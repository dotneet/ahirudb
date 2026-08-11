//! ビットマップ。validity マスクと BOOLEAN 値の格納に使う。
//!
//! ビット順は LSB-first（Arrow / Parquet と同じ）。

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

    /// 全ビット 1 のビットマップ。
    pub fn ones(len: usize) -> Self {
        let mut b = Bitmap { words: vec![u64::MAX; nwords(len)], len };
        b.clear_tail();
        b
    }

    /// 全ビット 0 のビットマップ。
    pub fn zeros(len: usize) -> Self {
        Bitmap { words: vec![0u64; nwords(len)], len }
    }

    /// LSB-first でパックされたバイト列から読み込む。
    /// Parquet の PLAIN BOOLEAN / bit-packed run と同じレイアウト。
    pub fn from_lsb_bytes(bytes: &[u8], len: usize) -> Self {
        let mut b = Bitmap::zeros(len);
        let n = core::cmp::min(bytes.len(), len.div_ceil(8));
        // 添字はワード位置とシフト量の両方に使う。イテレータにすると
        // かえって読みにくくなるので添字ループのままにする。
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

    /// `n` 個の同じ値をまとめて追加する。RLE run の展開で使う。
    pub fn push_n(&mut self, v: bool, n: usize) {
        // 素朴なループでも 64 ビット境界まで進めば word 単位に落ちる。
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

    /// 別のビットマップを末尾に連結する。ページ単位の validity を
    /// 列チャンク全体のビットマップに積むのに使う。
    pub fn extend(&mut self, other: &Bitmap) {
        if self.len.is_multiple_of(64) {
            // ワード境界に揃っているので word 単位でコピーできる。
            let base = self.words.len();
            self.words.extend_from_slice(&other.words);
            self.len += other.len;
            // other 側の末尾パディングは 0 なので、そのままで整合する。
            let _ = base;
        } else {
            for i in 0..other.len {
                self.push(other.get(i));
            }
        }
    }

    /// 長さを `len` に伸ばし、追加分を `v` で埋める。
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

    /// 全ビットが 1 か。validity が実質不要かの判定に使う。
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
    }

    /// ビット反転。末尾のパディングビットは 0 のまま保つ。
    pub fn negate(&mut self) {
        for w in self.words.iter_mut() {
            *w = !*w;
        }
        self.clear_tail();
    }

    /// 立っているビットの位置を昇順に `out` へ追記する。
    /// フィルタ結果から selection vector を作るのに使う。
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

    /// 末尾の余りビットを 0 にする。`count_ones` などが狂わないように。
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

/// 型不一致のアクセサが返すための空ビットマップ。
/// `Vec::new()` は const fn なので確保も静的初期化子も不要。
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
        // 0b1010_1100 = ビット 2,3,5,7 が 1
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
