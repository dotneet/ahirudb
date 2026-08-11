//! Parquet の Split Block Bloom Filter (SBBF) 読み取り専用実装。
//!
//! `BloomFilterHeader`（`meta.rs`）に続くビットセット本体を扱う。フォーマットは
//! 「32 バイト（= 8 × u32）のブロックが `num_bytes / 32` 個並ぶ」もので、各ブロック
//! は 8 個のマスクを持つ独立した小さい Bloom フィルタ（Apache Parquet Format 仕様
//! `BloomFilter.md` の Split Block Bloom Filter）。
//!
//! ハッシュは XXH64（seed 0）を PLAIN エンコードした値バイト列に対して掛ける。
//! `ahiru-zstd` crate に XXH64 実装があるが、DESIGN.md §6 の通り
//! `ahiru-zstd`/`ahiru-core` は互いに依存させない設計方針のため、ここでは
//! 同じアルゴリズムを ~40 行だけ複製する（クレートをまたいだ依存を 1 本足す
//! ためだけに、双方が「他方が無くても壊れない」という性質を失うのは割に合わない）。

// --- XXH64 (seed = 0) -------------------------------------------------------
// `crates/ahiru-zstd/src/xxh64.rs` と同一アルゴリズムの複製。展開結果は
// 全て手元にあるので、逐次更新版ではなく一発計算で足りる。

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

/// SBBF の各ブロック内で使う 8 個のソルト定数。
/// Apache Parquet Format 仕様 (`BloomFilter.md` の "Block-based Bloom filter"
/// 節) が参照する Impala のブロック分割 Bloom フィルタ実装に由来する固定値で、
/// parquet-mr / parquet-cpp / arrow-rs のリファレンス実装も同じ値を使う。
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

/// 1 ブロックのバイト数（8 個の u32 ワード）。
const BLOCK_BYTES: usize = 32;

/// 読み取り専用の Split Block Bloom Filter。
///
/// `bits` は `BloomFilterHeader.num_bytes` ぶんのビットセット本体（ヘッダは
/// 含まない）。長さは `BLOCK_BYTES` の倍数でなければならない
/// （`decode_bloom_filter_header` が `num_bytes % 32 == 0` を検証済み）。
pub struct BloomFilter<'a> {
    bits: &'a [u8],
}

impl<'a> BloomFilter<'a> {
    /// `bits.len()` が 0 または 32 の倍数でなければ `None`
    /// （呼び出し側は「使わない」で安全側に倒せる）。
    pub fn new(bits: &'a [u8]) -> Option<Self> {
        if bits.is_empty() || !bits.len().is_multiple_of(BLOCK_BYTES) {
            return None;
        }
        Some(BloomFilter { bits })
    }

    /// `key_bytes_in_plain_encoding` が挿入されている可能性があるか。
    ///
    /// 偽陽性はあり得る（`true` を返しても実際には無いことがある）が、
    /// 偽陰性は無い（挿入済みの値に対して `false` を返すことは無い）。
    pub fn contains(&self, key_bytes_in_plain_encoding: &[u8]) -> bool {
        let h = xxh64(key_bytes_in_plain_encoding);
        let num_blocks = (self.bits.len() / BLOCK_BYTES) as u64;
        // ブロック選択: ハッシュの上位 32 ビットを [0, num_blocks) に写像する
        // "multiply-shift" 手法（Lemire 法）。仕様の参照実装と同じ式。
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

    /// テスト専用のビルダ。`contains` と同じマスク計算を使って該当ビットを
    /// 立てる（本番コードは Bloom フィルタを書き出さないので、これは
    /// フィクスチャ構築用に留める）。
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
        // XXH64(seed=0) の公式リファレンス値（xxHash テストベクタ）。
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
        // 2 ブロック(64 バイト)に、複数の異なる型のキーを挿入する。
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
        // 疎なフィルタ（1 件だけ挿入）なら、無関係な値が偶然衝突する確率は
        // 実用上無視できる。具体的な非衝突を 1 つ確認しておく。
        let mut bits = [0u8; 32];
        insert(&mut bits, &1i32.to_le_bytes());
        let bf = BloomFilter::new(&bits).unwrap();
        assert!(bf.contains(&1i32.to_le_bytes()));
        assert!(!bf.contains(&999_999i32.to_le_bytes()));
    }
}
