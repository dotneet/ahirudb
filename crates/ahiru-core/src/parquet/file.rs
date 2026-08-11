//! ファイル単位の入口: フッタの位置決めとメタデータの解決。
//!
//! フッタは末尾にあるので、まず末尾 64 KiB を投機的に取得する。多くの
//! ファイルはこれ 1 往復でフッタを読み切れる（DESIGN.md §5）。足りなければ
//! `footer_retry_range` が示す正確な範囲をもう一度取りに行く。

use crate::parquet::meta::{decode_file_metadata, FileMetaData};
use crate::parquet::schema::{resolve_schema, ParquetSchema};
use crate::parquet::*;
use crate::prelude::*;

/// 末尾の投機取得サイズ。
pub const FOOTER_PROBE: u64 = 64 * 1024;
/// フッタ長の上限。壊れた長さフィールドで巨大確保しないため。
pub const MAX_FOOTER_LEN: usize = 256 * 1024 * 1024;
/// マジック 4 バイト + 長さ 4 バイト。
const TRAILER: usize = 8;

/// 解決済みの Parquet ファイル。
pub struct ParquetFile {
    pub meta: FileMetaData,
    pub schema: ParquetSchema,
}

impl ParquetFile {
    pub fn num_rows(&self) -> i64 {
        self.meta.num_rows
    }

    pub fn num_row_groups(&self) -> usize {
        self.meta.row_groups.len()
    }
}

/// 末尾の投機取得で読むべき範囲 `[offset, len)`。
pub fn footer_probe_range(file_len: u64) -> (u64, u64) {
    let len = FOOTER_PROBE.min(file_len);
    (file_len - len, len)
}

/// 投機取得したテール部分からフッタを解決する。
///
/// `tail` はファイル末尾のバイト列、`tail_offset` はそのファイル上の開始位置。
/// テールにフッタ全体が収まっていなければ、必要な範囲を `Err` ではなく
/// `Ok(Err(range))` として返す（呼び出し側が再取得する）。
pub fn parse_footer(
    tail: &[u8],
    tail_offset: u64,
    file_len: u64,
) -> Result<core::result::Result<ParquetFile, (u64, u64)>> {
    ensure!(file_len >= (TRAILER + 4) as u64, BadMagic);
    ensure!(tail.len() >= TRAILER, UnexpectedEof);
    ensure!(tail_offset + tail.len() as u64 == file_len, Internal);

    let n = tail.len();
    let magic = &tail[n - 4..];
    if magic == MAGIC_ENCRYPTED {
        err!(EncryptionUnsupported);
    }
    ensure!(magic == MAGIC, BadMagic);

    let meta_len =
        u32::from_le_bytes([tail[n - 8], tail[n - 7], tail[n - 6], tail[n - 5]]) as usize;
    ensure!(meta_len <= MAX_FOOTER_LEN, LimitExceeded);
    let meta_end = file_len - TRAILER as u64;
    ensure!(meta_len as u64 <= meta_end, BadMagic);
    let meta_start = meta_end - meta_len as u64;

    if meta_start < tail_offset {
        // テールに収まらなかった。フッタ全体 + トレーラを取り直す。
        return Ok(Err((meta_start, file_len - meta_start)));
    }

    let s = (meta_start - tail_offset) as usize;
    let e = (meta_end - tail_offset) as usize;
    let meta = decode_file_metadata(&tail[s..e])?;
    let schema = resolve_schema(&meta)?;
    Ok(Ok(ParquetFile { meta, schema }))
}

/// ファイル全体がメモリにある場合の入口。
pub fn open_bytes(bytes: &[u8]) -> Result<ParquetFile> {
    let file_len = bytes.len() as u64;
    let (off, _) = footer_probe_range(file_len);
    match parse_footer(&bytes[off as usize..], off, file_len)? {
        Ok(f) => Ok(f),
        // メモリ上に全体があるので、再取得ではなくその場で読み直せばよい。
        Err((start, _)) => match parse_footer(&bytes[start as usize..], start, file_len)? {
            Ok(f) => Ok(f),
            // 2 度目は必ずフッタ全体を含む範囲なので、ここに来たら長さが矛盾している。
            Err(_) => err!(BadMagic),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::code_of;

    #[test]
    fn probe_range_clamps_to_file_size() {
        assert_eq!(footer_probe_range(1000), (0, 1000));
        assert_eq!(footer_probe_range(100_000), (100_000 - FOOTER_PROBE, FOOTER_PROBE));
    }

    #[test]
    fn rejects_non_parquet() {
        let buf = vec![0u8; 64];
        assert_eq!(code_of(open_bytes(&buf)), Some(Code::BadMagic));
    }

    #[test]
    fn rejects_encrypted_footer() {
        let mut buf = vec![0u8; 64];
        buf[60..].copy_from_slice(MAGIC_ENCRYPTED);
        assert_eq!(code_of(open_bytes(&buf)), Some(Code::EncryptionUnsupported));
    }

    #[test]
    fn rejects_footer_length_past_start_of_file() {
        let mut buf = vec![0u8; 64];
        buf[60..].copy_from_slice(MAGIC);
        buf[56..60].copy_from_slice(&1_000_000u32.to_le_bytes());
        assert_eq!(code_of(open_bytes(&buf)), Some(Code::BadMagic));
    }

    #[test]
    fn requests_refetch_when_footer_exceeds_probe() {
        // ファイルは 200 KiB、フッタは 100 KiB。64 KiB の投機取得では届かない。
        let file_len = 200 * 1024u64;
        let mut tail = vec![0u8; FOOTER_PROBE as usize];
        let n = tail.len();
        tail[n - 4..].copy_from_slice(MAGIC);
        tail[n - 8..n - 4].copy_from_slice(&(100 * 1024u32).to_le_bytes());
        let r = parse_footer(&tail, file_len - FOOTER_PROBE, file_len).unwrap();
        match r {
            Err((off, len)) => {
                assert_eq!(off, file_len - 8 - 100 * 1024);
                assert_eq!(off + len, file_len);
            }
            Ok(_) => panic!("expected a refetch request"),
        }
    }
}
