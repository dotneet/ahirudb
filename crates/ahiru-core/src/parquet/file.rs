//! File-level entry point: locating the footer and resolving metadata.
//!
//! The footer sits at the end of the file, so we speculatively fetch the
//! last 64 KiB first. Most files can have their footer fully read in this
//! single round trip (DESIGN.md §5). If that's not enough, we go fetch the
//! exact range indicated by `footer_retry_range`.

use crate::parquet::meta::{decode_file_metadata, FileMetaData};
use crate::parquet::schema::{resolve_schema, ParquetSchema};
use crate::parquet::*;
use crate::prelude::*;

/// Speculative fetch size for the tail.
pub const FOOTER_PROBE: u64 = 64 * 1024;
/// Upper bound on footer length. Prevents a huge allocation from a corrupted length field.
pub const MAX_FOOTER_LEN: usize = 256 * 1024 * 1024;
/// Magic (4 bytes) + length (4 bytes).
const TRAILER: usize = 8;

/// A resolved Parquet file.
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

/// The range `[offset, len)` to read for the tail's speculative fetch.
pub fn footer_probe_range(file_len: u64) -> (u64, u64) {
    let len = FOOTER_PROBE.min(file_len);
    (file_len - len, len)
}

/// Resolve the footer from a speculatively fetched tail portion.
///
/// `tail` is the byte range at the end of the file, and `tail_offset` is
/// where that starts in the file. If the entire footer doesn't fit within
/// the tail, the needed range is returned as `Ok(Err(range))` rather than
/// `Err` (so the caller can refetch it).
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
        // Didn't fit in the tail. Refetch the entire footer + trailer.
        return Ok(Err((meta_start, file_len - meta_start)));
    }

    let s = (meta_start - tail_offset) as usize;
    let e = (meta_end - tail_offset) as usize;
    let meta = decode_file_metadata(&tail[s..e])?;
    let schema = resolve_schema(&meta)?;
    Ok(Ok(ParquetFile { meta, schema }))
}

/// Entry point for when the entire file is already in memory.
pub fn open_bytes(bytes: &[u8]) -> Result<ParquetFile> {
    let file_len = bytes.len() as u64;
    let (off, _) = footer_probe_range(file_len);
    match parse_footer(&bytes[off as usize..], off, file_len)? {
        Ok(f) => Ok(f),
        // The whole file is already in memory, so just re-read it in place instead of refetching.
        Err((start, _)) => match parse_footer(&bytes[start as usize..], start, file_len)? {
            Ok(f) => Ok(f),
            // The second attempt always covers a range that includes the entire footer, so reaching here means the lengths are inconsistent.
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
        // The file is 200 KiB and the footer is 100 KiB. A 64 KiB speculative fetch doesn't reach it.
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
