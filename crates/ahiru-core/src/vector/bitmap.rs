//! Bitmaps. Used for validity masks and to store BOOLEAN values.
//!
//! Bit order is LSB-first (the same as Arrow / Parquet).

use crate::prelude::*;

#[derive(Clone, PartialEq, Eq)]
pub struct Bitmap {
    words: Vec<u64>,
    len: usize,
}

#[inline]
fn nwords(len: usize) -> usize {
    len.div_ceil(64)
}

impl Bitmap {
    pub fn new() -> Self {
        Bitmap { words: Vec::new(), len: 0 }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Bitmap { words: Vec::with_capacity(nwords(cap)), len: 0 }
    }

    /// A bitmap with every bit set.
    pub fn ones(len: usize) -> Self {
        let mut b = Bitmap { words: vec![u64::MAX; nwords(len)], len };
        b.clear_tail();
        b
    }

    /// A bitmap with every bit clear.
    pub fn zeros(len: usize) -> Self {
        Bitmap { words: vec![0u64; nwords(len)], len }
    }

    /// Reads from LSB-first packed bytes.
    /// The same layout as Parquet's PLAIN BOOLEAN / bit-packed runs.
    pub fn from_lsb_bytes(bytes: &[u8], len: usize) -> Self {
        let mut b = Bitmap::zeros(len);
        let n = core::cmp::min(bytes.len(), len.div_ceil(8));
        // The index serves as both the word position and the shift amount. An iterator
        // would read worse here, so this stays an index loop.
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let w = i / 8;
            let shift = (i % 8) * 8;
            b.words[w] |= (bytes[i] as u64) << shift;
        }
        b.clear_tail();
        b
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn get(&self, i: usize) -> bool {
        debug_assert!(i < self.len);
        (self.words[i >> 6] >> (i & 63)) & 1 != 0
    }

    #[inline]
    pub fn set(&mut self, i: usize, v: bool) {
        debug_assert!(i < self.len);
        let w = i >> 6;
        let mask = 1u64 << (i & 63);
        if v {
            self.words[w] |= mask;
        } else {
            self.words[w] &= !mask;
        }
    }

    #[inline]
    pub fn push(&mut self, v: bool) {
        if self.len.is_multiple_of(64) {
            self.words.push(0);
        }
        let i = self.len;
        self.len += 1;
        if v {
            self.words[i >> 6] |= 1u64 << (i & 63);
        }
    }

    /// Appends `n` copies of the same value at once. Used when expanding an RLE run.
    pub fn push_n(&mut self, v: bool, n: usize) {
        // Even a naive loop drops to word granularity once it reaches a 64-bit boundary.
        let mut rest = n;
        while rest > 0 && !self.len.is_multiple_of(64) {
            self.push(v);
            rest -= 1;
        }
        let fill = if v { u64::MAX } else { 0 };
        while rest >= 64 {
            self.words.push(fill);
            self.len += 64;
            rest -= 64;
        }
        for _ in 0..rest {
            self.push(v);
        }
    }

    /// Concatenates another bitmap onto the end. Used to stack per-page validity
    /// into the bitmap for the whole column chunk.
    pub fn extend(&mut self, other: &Bitmap) {
        if self.len.is_multiple_of(64) {
            // Word-aligned, so this can copy a word at a time.
            let base = self.words.len();
            self.words.extend_from_slice(&other.words);
            self.len += other.len;
            // The trailing padding on the other side is 0, so it stays consistent as is.
            let _ = base;
        } else {
            for i in 0..other.len {
                self.push(other.get(i));
            }
        }
    }

    /// Grows the length to `len`, filling the addition with `v`.
    pub fn resize(&mut self, len: usize, v: bool) {
        if len <= self.len {
            self.len = len;
            self.words.truncate(nwords(len));
            self.clear_tail();
        } else {
            self.push_n(v, len - self.len);
        }
    }

    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Whether every bit is set. Used to decide whether validity is effectively unnecessary.
    pub fn all_set(&self) -> bool {
        self.count_ones() == self.len
    }

    pub fn and_assign(&mut self, other: &Bitmap) {
        debug_assert_eq!(self.len, other.len);
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a &= *b;
        }
    }

    pub fn or_assign(&mut self, other: &Bitmap) {
        debug_assert_eq!(self.len, other.len);
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a |= *b;
        }
        self.clear_tail();
    }

    /// Inverts every bit, keeping the trailing padding bits at 0.
    pub fn negate(&mut self) {
        for w in self.words.iter_mut() {
            *w = !*w;
        }
        self.clear_tail();
    }

    /// Appends the positions of the set bits to `out`, in ascending order.
    /// Used to build a selection vector from a filter result.
    pub fn append_set_indices(&self, out: &mut Vec<u32>) {
        for (wi, &w) in self.words.iter().enumerate() {
            let mut bits = w;
            while bits != 0 {
                let t = bits.trailing_zeros() as usize;
                out.push((wi * 64 + t) as u32);
                bits &= bits - 1;
            }
        }
    }

    /// Clears the leftover trailing bits, so `count_ones` and friends stay correct.
    fn clear_tail(&mut self) {
        let rem = self.len % 64;
        if rem != 0 {
            if let Some(last) = self.words.last_mut() {
                *last &= (1u64 << rem) - 1;
            }
        }
    }

    pub fn as_words(&self) -> &[u64] {
        &self.words
    }
}

impl Default for Bitmap {
    fn default() -> Self {
        Bitmap::new()
    }
}

/// The empty bitmap returned by accessors on a type mismatch.
/// `Vec::new()` is a const fn, so neither allocation nor a static initializer is needed.
static EMPTY: Bitmap = Bitmap { words: Vec::new(), len: 0 };

impl Bitmap {
    pub fn empty_ref() -> &'static Bitmap {
        &EMPTY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_get() {
        let mut b = Bitmap::new();
        for i in 0..200 {
            b.push(i % 3 == 0);
        }
        assert_eq!(b.len(), 200);
        for i in 0..200 {
            assert_eq!(b.get(i), i % 3 == 0, "bit {i}");
        }
    }

    #[test]
    fn push_n_spans_word_boundaries() {
        let mut b = Bitmap::new();
        b.push_n(true, 5);
        b.push_n(false, 130);
        b.push_n(true, 3);
        assert_eq!(b.len(), 138);
        assert_eq!(b.count_ones(), 8);
        for i in 0..5 {
            assert!(b.get(i));
        }
        for i in 5..135 {
            assert!(!b.get(i));
        }
        for i in 135..138 {
            assert!(b.get(i));
        }
    }

    #[test]
    fn ones_clears_tail() {
        let b = Bitmap::ones(10);
        assert_eq!(b.count_ones(), 10);
        assert!(b.all_set());
    }

    #[test]
    fn from_lsb_bytes_matches_parquet_layout() {
        // 0b1010_1100 = bits 2, 3, 5, and 7 are set
        let b = Bitmap::from_lsb_bytes(&[0b1010_1100], 8);
        let got: Vec<bool> = (0..8).map(|i| b.get(i)).collect();
        assert_eq!(got, vec![false, false, true, true, false, true, false, true]);
    }

    #[test]
    fn negate_keeps_tail_clear() {
        let mut b = Bitmap::zeros(10);
        b.negate();
        assert_eq!(b.count_ones(), 10);
    }

    #[test]
    fn set_indices() {
        let mut b = Bitmap::zeros(130);
        b.set(0, true);
        b.set(64, true);
        b.set(129, true);
        let mut out = Vec::new();
        b.append_set_indices(&mut out);
        assert_eq!(out, vec![0, 64, 129]);
    }
}
