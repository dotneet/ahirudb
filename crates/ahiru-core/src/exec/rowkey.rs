//! 行キーの正規化とハッシュ索引。
//!
//! 集約（GROUP BY）と結合（等値結合）はどちらも「複数列の値をキーにして
//! 引く」動作をする。**別々に実装すると等価判定の細部（NULL の扱い、-0.0、
//! NaN）がずれて静かに壊れる**ので、1 か所に寄せて共有する。
//! コードサイズの点でも 2 つ持つ理由がない。
//!
//! キーは**等価判定専用**で、順序は持たない。ソートは値を直接比較する。

use crate::prelude::*;
use crate::rt::hash::hash_u64;
use crate::vector::{Data, Vector};

/// 1 行分のキーを `out` に書く。
///
/// 各列は「1 バイトの有無フラグ + 値」。NULL はフラグ 0 のみで値を持たない。
/// SQL の `=` では NULL は NULL と等しくないが、**GROUP BY と DISTINCT では
/// 同じグループに入る**。この関数は後者の意味論で符号化するので、結合側は
/// NULL キーの行を投入する前に自分で弾くこと。
pub fn encode_key(cols: &[&Vector], row: usize, out: &mut Vec<u8>) {
    out.clear();
    for c in cols {
        if !c.is_valid(row) {
            out.push(0);
            continue;
        }
        out.push(1);
        match c.data() {
            Data::Bool(b) => out.push(b.get(row) as u8),
            Data::I32(v) => out.extend_from_slice(&v[row].to_le_bytes()),
            Data::I64(v) => out.extend_from_slice(&v[row].to_le_bytes()),
            Data::I128(v) => out.extend_from_slice(&v[row].to_le_bytes()),
            Data::F64(v) => out.extend_from_slice(&canonical_f64(v[row]).to_le_bytes()),
            Data::Bytes(b) => {
                let s = b.get(row);
                out.extend_from_slice(&(s.len() as u32).to_le_bytes());
                out.extend_from_slice(s);
            }
        }
    }
}

/// 浮動小数のビット表現を正規化する。
///
/// `-0.0` と `0.0` は同じ値なので同じグループに入れる。NaN は比較演算では
/// 常に偽だが、グループ化では 1 つにまとめる（DuckDB と同じ挙動）。
/// ビット列をそのままキーにすると、この 2 つが別グループに割れてしまう。
#[inline]
pub fn canonical_f64(v: f64) -> u64 {
    if v.is_nan() {
        0x7ff8_0000_0000_0000
    } else if v == 0.0 {
        0
    } else {
        v.to_bits()
    }
}

/// キーが 1 つでも NULL を含むか。等値結合では NULL キーは決して一致しない
/// ので、投入・探索の前に弾くのに使う。
pub fn key_has_null(cols: &[&Vector], row: usize) -> bool {
    cols.iter().any(|c| !c.is_valid(row))
}

/// バイト列キー → `u32` 値のオープンアドレッシング表。
///
/// キー本体は 1 本のアリーナに連結して置く。エントリごとに `Vec` を持つと
/// 確保回数が行数に比例して増えるため。
pub struct HashIndex {
    /// キーを連結したアリーナ。
    keys: Vec<u8>,
    /// `(キー開始, キー長, 値)`
    entries: Vec<(u32, u32, u32)>,
    /// 各エントリのハッシュ値。再ハッシュ時にキーを読み直さずに済む。
    hashes: Vec<u64>,
    /// バケット。0 は空、それ以外は `entries` の添字 + 1。
    buckets: Vec<u32>,
    mask: usize,
}

/// バケットの初期容量（2 のべき乗）。
const INITIAL_BUCKETS: usize = 1024;
/// この使用率を超えたら倍にする。
const LOAD_NUM: usize = 7;
const LOAD_DEN: usize = 10;

impl HashIndex {
    pub fn new() -> Self {
        HashIndex {
            keys: Vec::new(),
            entries: Vec::new(),
            hashes: Vec::new(),
            buckets: vec![0; INITIAL_BUCKETS],
            mask: INITIAL_BUCKETS - 1,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 保持しているキーの総バイト数。メモリ上限の判定に使う。
    pub fn key_bytes(&self) -> usize {
        self.keys.len()
    }

    #[inline]
    fn hash(key: &[u8]) -> u64 {
        crate::rt::hash::hash_bytes(key)
    }

    #[inline]
    fn entry_key(&self, i: usize) -> &[u8] {
        let (off, len, _) = self.entries[i];
        &self.keys[off as usize..(off + len) as usize]
    }

    /// 空きスロット、または一致するエントリのスロットを線形探査で探す。
    #[inline]
    fn probe(&self, key: &[u8], h: u64) -> (usize, Option<usize>) {
        let mut slot = (hash_u64(h) as usize) & self.mask;
        loop {
            let b = self.buckets[slot];
            if b == 0 {
                return (slot, None);
            }
            let i = (b - 1) as usize;
            if self.hashes[i] == h && self.entry_key(i) == key {
                return (slot, Some(i));
            }
            slot = (slot + 1) & self.mask;
        }
    }

    /// 集約用。キーが無ければ「次のスロット番号」を値として挿入する。
    /// 返り値は `(値, 新規に挿入したか)`。
    pub fn get_or_insert(&mut self, key: &[u8]) -> (u32, bool) {
        let h = Self::hash(key);
        let (slot, hit) = self.probe(key, h);
        if let Some(i) = hit {
            return (self.entries[i].2, false);
        }
        let value = self.entries.len() as u32;
        self.insert_at(slot, key, h, value);
        (value, true)
    }

    /// 探索のみ。無ければ `None`。
    pub fn lookup(&self, key: &[u8]) -> Option<u32> {
        let h = Self::hash(key);
        let (_, hit) = self.probe(key, h);
        hit.map(|i| self.entries[i].2)
    }

    /// 結合のビルド側用。同じキーが複数あってよい。
    ///
    /// 既存の値を返しつつ新しい値で置き換えるので、呼び出し側は
    /// `next[value] = 返り値` としてチェーンを張れる。
    pub fn insert_chained(&mut self, key: &[u8], value: u32) -> Option<u32> {
        let h = Self::hash(key);
        let (slot, hit) = self.probe(key, h);
        match hit {
            Some(i) => {
                let prev = self.entries[i].2;
                self.entries[i].2 = value;
                Some(prev)
            }
            None => {
                self.insert_at(slot, key, h, value);
                None
            }
        }
    }

    fn insert_at(&mut self, slot: usize, key: &[u8], h: u64, value: u32) {
        let off = self.keys.len() as u32;
        self.keys.extend_from_slice(key);
        self.entries.push((off, key.len() as u32, value));
        self.hashes.push(h);
        self.buckets[slot] = self.entries.len() as u32;
        if self.entries.len() * LOAD_DEN > self.buckets.len() * LOAD_NUM {
            self.grow();
        }
    }

    fn grow(&mut self) {
        let n = self.buckets.len() * 2;
        self.buckets.clear();
        self.buckets.resize(n, 0);
        self.mask = n - 1;
        for i in 0..self.entries.len() {
            let mut slot = (hash_u64(self.hashes[i]) as usize) & self.mask;
            while self.buckets[slot] != 0 {
                slot = (slot + 1) & self.mask;
            }
            self.buckets[slot] = (i + 1) as u32;
        }
    }
}

impl Default for HashIndex {
    fn default() -> Self {
        HashIndex::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::{Ty, Value, Vector};

    fn vec_i32(vals: &[Option<i32>]) -> Vector {
        let mut v = Vector::new(Ty::Int);
        for x in vals {
            match x {
                Some(x) => v.push_value(&Value::I32(*x)),
                None => v.push_null(),
            }
        }
        v
    }

    fn vec_str(vals: &[Option<&str>]) -> Vector {
        let mut v = Vector::new(Ty::Varchar);
        for x in vals {
            match x {
                Some(x) => v.push_value(&Value::Bytes(x.as_bytes().to_vec())),
                None => v.push_null(),
            }
        }
        v
    }

    fn key(cols: &[&Vector], row: usize) -> Vec<u8> {
        let mut out = Vec::new();
        encode_key(cols, row, &mut out);
        out
    }

    #[test]
    fn equal_values_encode_equally() {
        let a = vec_i32(&[Some(1), Some(1), Some(2)]);
        assert_eq!(key(&[&a], 0), key(&[&a], 1));
        assert_ne!(key(&[&a], 0), key(&[&a], 2));
    }

    #[test]
    fn nulls_group_together_and_differ_from_values() {
        let a = vec_i32(&[None, None, Some(0)]);
        assert_eq!(key(&[&a], 0), key(&[&a], 1));
        assert_ne!(key(&[&a], 0), key(&[&a], 2));
    }

    #[test]
    fn multi_column_keys_are_not_confusable() {
        // ("a", "bc") と ("ab", "c") が同じキーにならないこと。
        // 長さを前置していなければ衝突する。
        let a1 = vec_str(&[Some("a")]);
        let b1 = vec_str(&[Some("bc")]);
        let a2 = vec_str(&[Some("ab")]);
        let b2 = vec_str(&[Some("c")]);
        assert_ne!(key(&[&a1, &b1], 0), key(&[&a2, &b2], 0));
    }

    #[test]
    fn negative_zero_and_nan_are_canonicalised() {
        let mut v = Vector::new(Ty::Double);
        v.push_value(&Value::F64(0.0));
        v.push_value(&Value::F64(-0.0));
        v.push_value(&Value::F64(f64::NAN));
        v.push_value(&Value::F64(-f64::NAN));
        assert_eq!(key(&[&v], 0), key(&[&v], 1), "-0.0 は 0.0 と同じグループ");
        assert_eq!(key(&[&v], 2), key(&[&v], 3), "NaN は 1 つのグループ");
        assert_ne!(key(&[&v], 0), key(&[&v], 2));
    }

    #[test]
    fn key_null_detection() {
        let a = vec_i32(&[Some(1), None]);
        let b = vec_i32(&[Some(1), Some(1)]);
        assert!(!key_has_null(&[&a, &b], 0));
        assert!(key_has_null(&[&a, &b], 1));
    }

    #[test]
    fn get_or_insert_assigns_sequential_slots() {
        let mut h = HashIndex::new();
        assert_eq!(h.get_or_insert(b"a"), (0, true));
        assert_eq!(h.get_or_insert(b"b"), (1, true));
        assert_eq!(h.get_or_insert(b"a"), (0, false));
        assert_eq!(h.len(), 2);
        assert_eq!(h.lookup(b"b"), Some(1));
        assert_eq!(h.lookup(b"zzz"), None);
    }

    #[test]
    fn growth_preserves_all_entries() {
        let mut h = HashIndex::new();
        // 初期容量を大きく超える件数を入れて再ハッシュを起こす。
        let n = 5000u32;
        for i in 0..n {
            let k = i.to_le_bytes();
            assert_eq!(h.get_or_insert(&k), (i, true));
        }
        assert_eq!(h.len(), n as usize);
        for i in 0..n {
            let k = i.to_le_bytes();
            assert_eq!(h.lookup(&k), Some(i), "再ハッシュ後に {i} が引けない");
        }
    }

    #[test]
    fn chained_insert_returns_previous_head() {
        let mut h = HashIndex::new();
        assert_eq!(h.insert_chained(b"k", 0), None);
        assert_eq!(h.insert_chained(b"k", 1), Some(0));
        assert_eq!(h.insert_chained(b"k", 2), Some(1));
        // 最後に入れたものが先頭。
        assert_eq!(h.lookup(b"k"), Some(2));
        assert_eq!(h.len(), 1);
    }

    #[test]
    fn empty_key_is_a_valid_key() {
        // GROUP BY を持たない集約は空キーの 1 グループになる。
        let mut h = HashIndex::new();
        assert_eq!(h.get_or_insert(b""), (0, true));
        assert_eq!(h.get_or_insert(b""), (0, false));
    }
}
