//! Hash functions.
//!
//! Neither cryptographically strong nor HashDoS resistant. An embedded analytics
//! engine does not assume adversarial keys, so speed and code size come first.

/// The splitmix64 finalizer. Used to mix integer keys.
#[inline]
pub fn hash_u64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

/// An FxHash-style byte-sequence hash. Reads 8 bytes at a time and mixes by multiplication and rotation.
pub fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h: u64 = SEED ^ (bytes.len() as u64);
    let mut chunks = bytes.chunks_exact(8);
    for c in &mut chunks {
        let v = u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
        h = (h.rotate_left(5) ^ v).wrapping_mul(SEED);
    }
    let rem = chunks.remainder();
    if !rem.is_empty() {
        let mut buf = [0u8; 8];
        buf[..rem.len()].copy_from_slice(rem);
        let v = u64::from_le_bytes(buf);
        h = (h.rotate_left(5) ^ v).wrapping_mul(SEED);
    }
    hash_u64(h)
}

/// A hash that ignores ASCII case. For SQL identifiers and keywords.
pub fn hash_ascii_ci(bytes: &[u8]) -> u64 {
    let mut h: u64 = SEED ^ (bytes.len() as u64);
    for &b in bytes {
        h = (h.rotate_left(5) ^ (b | 0x20) as u64).wrapping_mul(SEED);
    }
    hash_u64(h)
}

/// ASCII case-insensitive comparison.
#[inline]
pub fn eq_ascii_ci(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.eq_ignore_ascii_case(y))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ci_hash_matches_case_variants() {
        assert_eq!(hash_ascii_ci(b"SELECT"), hash_ascii_ci(b"select"));
        assert_eq!(hash_ascii_ci(b"FoO"), hash_ascii_ci(b"foO"));
        assert_ne!(hash_ascii_ci(b"foo"), hash_ascii_ci(b"bar"));
    }

    #[test]
    fn bytes_hash_is_length_sensitive() {
        assert_ne!(hash_bytes(b"a"), hash_bytes(b"a\0"));
        assert_eq!(hash_bytes(b"hello world"), hash_bytes(b"hello world"));
    }

    #[test]
    fn ci_eq() {
        assert!(eq_ascii_ci(b"Name", b"nAMe"));
        assert!(!eq_ascii_ci(b"Name", b"Names"));
    }
}
