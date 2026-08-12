//! Parquet's Split Block Bloom Filter (SBBF), read-only implementation.
//!
//! Handles the bitset body that follows the `BloomFilterHeader` (`meta.rs`).
//! The format is "`num_bytes / 32` blocks of 32 bytes (= 8 x u32) laid out in
//! sequence," where each block is an independent small Bloom filter with 8
//! masks (the Split Block Bloom Filter from the Apache Parquet Format spec's
//! `BloomFilter.md`).
//!
//! The hash is XXH64 (seed 0) applied to the PLAIN-encoded value byte
//! sequence. The `ahiru-zstd` crate has an XXH64 implementation, but per the
//! design policy in DESIGN.md §6 of keeping `ahiru-zstd`/`ahiru-core` free of
//! dependencies on each other, this duplicates the same algorithm in about 40
//! lines (adding one cross-crate dependency just for this isn't worth losing
//! the property that either crate keeps working without the other).

// --- XXH64 (seed = 0) -------------------------------------------------------
// A duplicate of the same algorithm as `crates/ahiru-zstd/src/xxh64.rs`. The
// decompressed output is always fully available here, so a one-shot
// computation suffices instead of an incremental streaming version.

const P1: u64 = 0x9E37_79B1_85EB_CA87;
const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const P3: u64 = 0x1656_67B1_9E37_79F9;
const P4: u64 = 0x85EB_CA77_C2B2_AE63;
const P5: u64 = 0x27D4_EB2F_1656_67C5;

#[inline]
fn round(acc: u64, v: u64) -> u64 {
    acc.wrapping_add(v.wrapping_mul(P2)).rotate_left(31).wrapping_mul(P1)
}

#[inline]
fn merge(h: u64, v: u64) -> u64 {
    (h ^ round(0, v)).wrapping_mul(P1).wrapping_add(P4)
}

#[inline]
fn u64le(b: &[u8]) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(b);
    u64::from_le_bytes(a)
}

#[inline]
fn u32le(b: &[u8]) -> u32 {
    let mut a = [0u8; 4];
    a.copy_from_slice(b);
    u32::from_le_bytes(a)
}

fn xxh64(data: &[u8]) -> u64 {
    let mut h;
    let mut p = 0usize;
    if data.len() >= 32 {
        let (mut v1, mut v2, mut v3, mut v4) =
            (P1.wrapping_add(P2), P2, 0u64, 0u64.wrapping_sub(P1));
        while p + 32 <= data.len() {
            v1 = round(v1, u64le(&data[p..p + 8]));
            v2 = round(v2, u64le(&data[p + 8..p + 16]));
            v3 = round(v3, u64le(&data[p + 16..p + 24]));
            v4 = round(v4, u64le(&data[p + 24..p + 32]));
            p += 32;
        }
        h = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));
        h = merge(h, v1);
        h = merge(h, v2);
        h = merge(h, v3);
        h = merge(h, v4);
    } else {
        h = P5;
    }
    h = h.wrapping_add(data.len() as u64);

    while p + 8 <= data.len() {
        h = (h ^ round(0, u64le(&data[p..p + 8])))
            .rotate_left(27)
            .wrapping_mul(P1)
            .wrapping_add(P4);
        p += 8;
    }
    if p + 4 <= data.len() {
        h = (h ^ (u32le(&data[p..p + 4]) as u64).wrapping_mul(P1))
            .rotate_left(23)
            .wrapping_mul(P2)
            .wrapping_add(P3);
        p += 4;
    }
    while p < data.len() {
        h = (h ^ (data[p] as u64).wrapping_mul(P5)).rotate_left(11).wrapping_mul(P1);
        p += 1;
    }

    h ^= h >> 33;
    h = h.wrapping_mul(P2);
    h ^= h >> 29;
    h = h.wrapping_mul(P3);
    h ^= h >> 32;
    h
}

// --- Split Block Bloom Filter -----------------------------------------------

/// The 8 salt constants used within each block of the SBBF.
/// These fixed values come from the Impala block-split Bloom filter
/// implementation referenced by the Apache Parquet Format spec
/// (`BloomFilter.md`'s "Block-based Bloom filter" section); the parquet-mr /
/// parquet-cpp / arrow-rs reference implementations use the same values.
const SALT: [u32; 8] = [
    0x47b6_137b,
    0x4497_4d91,
    0x8824_ad5b,
    0xa2b7_289d,
    0x7054_95c7,
    0x2df1_424b,
    0x9efc_4947,
    0x5c6b_fb31,
];

/// Byte size of one block (8 u32 words).
const BLOCK_BYTES: usize = 32;

/// A read-only Split Block Bloom Filter.
///
/// `bits` is the bitset body of `BloomFilterHeader.num_bytes` bytes (not
/// including the header). Its length must be a multiple of `BLOCK_BYTES`
/// (`decode_bloom_filter_header` has already verified `num_bytes % 32 == 0`).
pub struct BloomFilter<'a> {
    bits: &'a [u8],
}

impl<'a> BloomFilter<'a> {
    /// `None` if `bits.len()` is not 0 or a multiple of 32 (the caller can
    /// safely fall back to "not used").
    pub fn new(bits: &'a [u8]) -> Option<Self> {
        if bits.is_empty() || !bits.len().is_multiple_of(BLOCK_BYTES) {
            return None;
        }
        Some(BloomFilter { bits })
    }

    /// Whether `key_bytes_in_plain_encoding` may have been inserted.
    ///
    /// False positives are possible (returning `true` doesn't guarantee it
    /// was actually inserted), but there are no false negatives (this never
    /// returns `false` for a value that was actually inserted).
    pub fn contains(&self, key_bytes_in_plain_encoding: &[u8]) -> bool {
        let h = xxh64(key_bytes_in_plain_encoding);
        let num_blocks = (self.bits.len() / BLOCK_BYTES) as u64;
        // Block selection: maps the hash's upper 32 bits into [0, num_blocks)
        // using the "multiply-shift" technique (Lemire's method). Same formula as the spec's reference implementation.
        let block_idx = (((h >> 32).wrapping_mul(num_blocks)) >> 32) as usize;
        let base = block_idx * BLOCK_BYTES;
        let block = &self.bits[base..base + BLOCK_BYTES];
        let lo = h as u32;
        for (i, salt) in SALT.iter().enumerate() {
            let bit = salt.wrapping_mul(lo) >> 27;
            let mask = 1u32 << bit;
            let w = &block[i * 4..i * 4 + 4];
            let word = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
            if word & mask == 0 {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A test-only builder. Sets the corresponding bit using the same mask
    /// calculation as `contains` (production code never writes out a Bloom
    /// filter, so this is kept limited to building fixtures).
    fn insert(bits: &mut [u8], key: &[u8]) {
        let h = xxh64(key);
        let num_blocks = (bits.len() / BLOCK_BYTES) as u64;
        let block_idx = (((h >> 32).wrapping_mul(num_blocks)) >> 32) as usize;
        let base = block_idx * BLOCK_BYTES;
        let lo = h as u32;
        for (i, salt) in SALT.iter().enumerate() {
            let bit = salt.wrapping_mul(lo) >> 27;
            let mask = 1u32 << bit;
            let w = &mut bits[base + i * 4..base + i * 4 + 4];
            let word = u32::from_le_bytes([w[0], w[1], w[2], w[3]]) | mask;
            w.copy_from_slice(&word.to_le_bytes());
        }
    }

    #[test]
    fn xxh64_matches_known_vectors() {
        // Official reference values for XXH64(seed=0) (xxHash test vectors).
        assert_eq!(xxh64(b""), 0xEF46_DB37_51D8_E999);
        assert_eq!(xxh64(b"a"), 0xd24e_c4f1_a98c_6e5b);
    }

    #[test]
    fn new_rejects_non_multiple_of_32() {
        assert!(BloomFilter::new(&[]).is_none());
        assert!(BloomFilter::new(&[0u8; 31]).is_none());
        assert!(BloomFilter::new(&[0u8; 32]).is_some());
    }

    #[test]
    fn inserted_keys_are_never_reported_absent() {
        // Insert keys of several different types into 2 blocks (64 bytes).
        let mut bits = [0u8; 64];
        let keys: &[&[u8]] = &[
            &42i32.to_le_bytes(),
            &12345i64.to_le_bytes(),
            b"hello",
            b"a-somewhat-longer-key-value",
            &0i32.to_le_bytes(),
            &(-1i32).to_le_bytes(),
        ];
        for k in keys {
            insert(&mut bits, k);
        }
        let bf = BloomFilter::new(&bits).unwrap();
        for k in keys {
            assert!(bf.contains(k), "inserted key must never be reported absent");
        }
    }

    #[test]
    fn absent_key_is_reported_absent_in_a_sparse_filter() {
        // In a sparse filter (only one entry inserted), the odds of an
        // unrelated value colliding by chance are negligible in practice. Confirm one concrete non-collision here.
        let mut bits = [0u8; 32];
        insert(&mut bits, &1i32.to_le_bytes());
        let bf = BloomFilter::new(&bits).unwrap();
        assert!(bf.contains(&1i32.to_le_bytes()));
        assert!(!bf.contains(&999_999i32.to_le_bytes()));
    }
}
