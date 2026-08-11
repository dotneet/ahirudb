//! テーブルカタログとバイト供給元。
//!
//! `Source` は「取得済みのバイト範囲の集合」として表現する。この 1 つの型で
//! 2 つの経路を統一できるのが要点:
//!
//! - メモリ上に全体がある場合 … `[0, len)` を覆う範囲が 1 本だけある状態
//! - ホストからレンジ取得する場合 … 必要な範囲が届くたびに増えていく状態
//!
//! 実行側は `get()` を呼ぶだけでよく、`None` が返ったら「その範囲を取ってきて
//! ほしい」と要求を出す（DESIGN.md §6 の RowGroup 境界バリア）。

use crate::format::{self, FormatKind, TableFormat};
use crate::prelude::*;
use crate::rt::hash::eq_ascii_ci;
use crate::vector::Field;

/// 取得済みバイト範囲の集合。
pub struct Source {
    pub total_len: u64,
    /// `(開始オフセット, データ)`。開始オフセット昇順に保つ。
    chunks: Vec<(u64, Vec<u8>)>,
    /// ホストに展開してもらったページ。キーは圧縮ページ本体の
    /// `(ファイル上のオフセット, 長さ)`（DESIGN.md §6 のコーデック委譲）。
    decoded: Vec<((u64, u32), Vec<u8>)>,
}

impl Source {
    /// ファイル全体がメモリにある場合。
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Source { total_len: bytes.len() as u64, chunks: vec![(0, bytes)], decoded: Vec::new() }
    }

    /// ホストが保持していて、レンジ取得で読み出す場合。
    pub fn remote(total_len: u64) -> Self {
        Source { total_len, chunks: Vec::new(), decoded: Vec::new() }
    }

    /// `[off, off+len)` が取得済みならそのスライスを返す。
    pub fn get(&self, off: u64, len: usize) -> Option<&[u8]> {
        let end = off.checked_add(len as u64)?;
        // 範囲は多くても数十本なので線形探索で足りる。
        for (start, data) in &self.chunks {
            let cend = start + data.len() as u64;
            if *start <= off && end <= cend {
                let s = (off - start) as usize;
                return Some(&data[s..s + len]);
            }
        }
        None
    }

    /// 取得したバイト列を登録する。隣接・重複する範囲は結合する。
    pub fn insert(&mut self, off: u64, data: Vec<u8>) {
        if data.is_empty() {
            return;
        }
        self.chunks.push((off, data));
        self.chunks.sort_by_key(|(o, _)| *o);
        // 隣接・重複を 1 本にまとめる。放っておくと `get` の線形探索が伸びる。
        let mut merged: Vec<(u64, Vec<u8>)> = Vec::with_capacity(self.chunks.len());
        for (off, data) in self.chunks.drain(..) {
            match merged.last_mut() {
                Some((po, pd)) if off <= *po + pd.len() as u64 => {
                    let overlap = (*po + pd.len() as u64 - off) as usize;
                    if overlap < data.len() {
                        pd.extend_from_slice(&data[overlap..]);
                    }
                }
                _ => merged.push((off, data)),
            }
        }
        self.chunks = merged;
    }

    /// 要求範囲のうち、まだ取得できていない部分を返す。
    pub fn missing(&self, off: u64, len: u64) -> Option<(u64, u64)> {
        if len == 0 || self.get(off, len as usize).is_some() {
            None
        } else {
            Some((off, len.min(self.total_len.saturating_sub(off))))
        }
    }

    pub fn is_complete(&self) -> bool {
        matches!(self.chunks.first(), Some((0, d)) if d.len() as u64 == self.total_len)
    }

    /// ホストが展開したページを登録する。
    pub fn insert_decoded(&mut self, offset: u64, len: u32, data: Vec<u8>) {
        if let Some(e) = self.decoded.iter_mut().find(|(k, _)| *k == (offset, len)) {
            e.1 = data;
            return;
        }
        self.decoded.push(((offset, len), data));
    }

    /// 展開済みページが既にあるか。
    pub fn has_decoded(&self, offset: u64, len: u32) -> bool {
        self.decoded.iter().any(|(k, _)| *k == (offset, len))
    }

    pub fn decoded_bytes(&self) -> usize {
        self.decoded.iter().map(|(_, d)| d.len()).sum()
    }

    /// 展開済みページを捨てる。分割を 1 つ処理し終えたら呼ぶ。
    /// 溜め込むと圧縮前のファイルより大きなメモリを抱えることになる。
    pub fn clear_decoded(&mut self) {
        self.decoded.clear();
    }
}

impl crate::parquet::reader::PageCache for Source {
    fn get(&self, offset: u64, len: u32) -> Option<&[u8]> {
        self.decoded.iter().find(|(k, _)| *k == (offset, len)).map(|(_, d)| d.as_slice())
    }
}

/// 登録済みテーブル。
///
/// データ源（`source`）と読み方（`format`）を分けて持つ。Parquet でも CSV でも
/// 「バイト範囲を要求して読む」という形は同じなので、`Source` は共通で使える。
pub struct Table {
    pub name: String,
    pub source: Source,
    pub format: Box<dyn TableFormat>,
}

impl Table {
    /// スキーマを解決する。まだバイトが足りなければ必要な範囲を返す。
    pub fn resolve(&mut self) -> Result<core::result::Result<(), (u64, u64)>> {
        self.format.resolve(&self.source)
    }

    pub fn schema(&self) -> &[Field] {
        self.format.schema()
    }
}

#[derive(Default)]
pub struct Catalog {
    tables: Vec<Table>,
}

impl Catalog {
    pub fn new() -> Self {
        Catalog { tables: Vec::new() }
    }

    /// テーブルを登録する。同名があれば置き換える（再登録をエラーにしない）。
    ///
    /// この時点では I/O を行わない。スキーマ解決は最初のクエリまで遅延する。
    pub fn register(&mut self, name: &str, source: Source, kind: FormatKind) -> Result<usize> {
        let t = Table { name: name.into(), source, format: format::make(kind, name)? };
        Ok(match self.index_of(name) {
            Some(i) => {
                self.tables[i] = t;
                i
            }
            None => {
                self.tables.push(t);
                self.tables.len() - 1
            }
        })
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.tables.iter().position(|t| eq_ascii_ci(t.name.as_bytes(), name.as_bytes()))
    }

    pub fn get(&self, i: usize) -> Option<&Table> {
        self.tables.get(i)
    }

    pub fn get_mut(&mut self, i: usize) -> Option<&mut Table> {
        self.tables.get_mut(i)
    }

    pub fn len(&self) -> usize {
        self.tables.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tables.iter().map(|t| t.name.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_source_serves_any_range() {
        let s = Source::from_bytes((0u8..=255).collect());
        assert_eq!(s.get(0, 4), Some(&[0u8, 1, 2, 3][..]));
        assert_eq!(s.get(250, 6), Some(&[250u8, 251, 252, 253, 254, 255][..]));
        assert_eq!(s.get(250, 7), None);
        assert!(s.is_complete());
    }

    #[test]
    fn remote_source_reports_missing_then_serves() {
        let mut s = Source::remote(1000);
        assert_eq!(s.missing(100, 50), Some((100, 50)));
        s.insert(100, vec![7u8; 50]);
        assert_eq!(s.missing(100, 50), None);
        assert_eq!(s.get(120, 10), Some(&[7u8; 10][..]));
        // 別の範囲はまだ無い。
        assert_eq!(s.missing(300, 10), Some((300, 10)));
    }

    #[test]
    fn adjacent_ranges_merge() {
        let mut s = Source::remote(300);
        s.insert(100, vec![1u8; 50]);
        s.insert(150, vec![2u8; 50]);
        // 結合されているので 1 本のスライスとして取り出せる。
        let got = s.get(140, 20).unwrap();
        assert_eq!(&got[..10], &[1u8; 10]);
        assert_eq!(&got[10..], &[2u8; 10]);
    }

    #[test]
    fn overlapping_ranges_merge_without_duplication() {
        let mut s = Source::remote(300);
        s.insert(0, vec![1u8; 100]);
        s.insert(50, vec![2u8; 100]);
        assert_eq!(s.get(0, 150).unwrap().len(), 150);
        // 重なった部分は先着を優先し、後続の非重複部分だけを足す。
        assert_eq!(s.get(0, 150).unwrap()[99], 1);
        assert_eq!(s.get(0, 150).unwrap()[100], 2);
    }

    #[test]
    fn missing_range_is_clamped_to_file_length() {
        let s = Source::remote(120);
        assert_eq!(s.missing(100, 500), Some((100, 20)));
    }

    #[test]
    fn register_replaces_same_name_case_insensitively() {
        let mut c = Catalog::new();
        let a = c.register("Trips", Source::from_bytes(vec![1]), FormatKind::Parquet).unwrap();
        let b = c.register("trips", Source::from_bytes(vec![2]), FormatKind::Parquet).unwrap();
        assert_eq!(a, b);
        assert_eq!(c.len(), 1);
        assert_eq!(c.index_of("TRIPS"), Some(0));
    }
}
