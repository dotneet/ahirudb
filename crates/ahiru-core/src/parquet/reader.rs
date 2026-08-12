//! Column chunk (dictionary page + data pages) -> `Vector`.
//!
//! v1 only handles flat schemas, so the repetition level is always 0, and
//! the definition level can be represented as a single 0/1 bit. This is
//! exactly the validity bitmap, so we decode straight into a bitmap without
//! ever materializing a level array.
//!
//! Branching per logical type is confined to this module. Everything
//! downstream from here (the expression VM, operators) sees only the 6
//! physical types (DESIGN.md §8).

use alloc::borrow::Cow;

use crate::parquet::codec;
use crate::parquet::encoding::{self, RleDecoder};
use crate::parquet::meta::{decode_page_header, ColumnMetaData, PageHeader};
use crate::parquet::schema::ColumnDesc;
use crate::parquet::*;
use crate::prelude::*;
use crate::vector::{Bitmap, Data, PhysType, Ty, Vector};

/// Julian day number of the Unix epoch (1970-01-01). Used to convert INT96.
const JULIAN_EPOCH: i64 = 2_440_588;
const MICROS_PER_DAY: i64 = 86_400_000_000;

/// Upper bound on the number of values per page. Prevents a huge allocation from a corrupted header.
const MAX_PAGE_VALUES: usize = 1 << 26;

/// Storage for pages the host has decompressed.
///
/// Codecs that aren't built in (GZIP / ZSTD) delegate decompression to the
/// host (DESIGN.md §6). Delegated results are looked up here. The key is the
/// **absolute file offset and length of the compressed page body**, with one
/// entry per page.
pub trait PageCache {
    fn get(&self, offset: u64, len: u32) -> Option<&[u8]>;
}

/// A dummy for when delegation isn't used. Sufficient for files that only use built-in codecs.
pub struct NoPageCache;

impl PageCache for NoPageCache {
    fn get(&self, _offset: u64, _len: u32) -> Option<&[u8]> {
        None
    }
}

/// A page that needs the host to decompress it.
#[derive(Clone, Copy)]
pub struct CodecPage {
    pub codec: Compression,
    /// File position and length of the compressed data body.
    pub offset: u64,
    pub len: u32,
    /// Decompressed size as declared by the page header.
    pub out_len: u32,
}

/// Decode an entire column chunk into a single vector.
///
/// `buf` is the byte range indicated by `ColumnMetaData::byte_range()`.
/// `num_rows` is that RowGroup's row count.
pub fn read_column_chunk(
    desc: &ColumnDesc,
    meta: &ColumnMetaData,
    buf: &[u8],
    chunk_start: u64,
    num_rows: usize,
    cache: &dyn PageCache,
) -> Result<Vector> {
    let mut out = Vector::with_capacity(desc.ty, num_rows);
    // Only accumulate validity for columns that have a definition level.
    let mut validity =
        if desc.max_def_level > 0 { Some(Bitmap::with_capacity(num_rows)) } else { None };
    let mut dict: Option<Vector> = None;
    let mut pos = 0usize;
    let mut rows_done = 0usize;

    while rows_done < num_rows {
        ensure!(pos < buf.len(), UnexpectedEof, pos);
        let (hdr, hlen) = decode_page_header(&buf[pos..])?;
        pos += hlen;
        let clen = hdr.compressed_page_size as usize;
        ensure!(clen <= buf.len() - pos, UnexpectedEof, pos);
        let raw_off = chunk_start + pos as u64;
        let raw = &buf[pos..pos + clen];
        pos += clen;

        match hdr.ptype {
            PageType::DictionaryPage => {
                dict = Some(decode_dictionary_page(desc, meta, &hdr, raw, raw_off, cache)?);
            }
            _ => {
                rows_done += decode_one_page(
                    desc,
                    meta,
                    &hdr,
                    raw,
                    raw_off,
                    dict.as_ref(),
                    &mut out,
                    &mut validity,
                    cache,
                )?;
            }
        }
    }

    ensure!(rows_done == num_rows, BadCompressedData, pos);
    if let Some(v) = validity {
        debug_assert_eq!(v.len(), out.len());
        out.set_validity(Some(v));
        out.compact_validity();
    }
    Ok(out)
}

/// Decode a single data page (v1/v2) according to its page header, and
/// return the number of rows added. `IndexPage` is a variant that's never
/// actually written, so this does nothing for it. The caller handles
/// dictionary pages separately (because subsequent pages reference the
/// decoded result).
#[allow(clippy::too_many_arguments)]
fn decode_one_page(
    desc: &ColumnDesc,
    meta: &ColumnMetaData,
    hdr: &PageHeader,
    raw: &[u8],
    raw_off: u64,
    dict: Option<&Vector>,
    out: &mut Vector,
    validity: &mut Option<Bitmap>,
    cache: &dyn PageCache,
) -> Result<usize> {
    match hdr.ptype {
        PageType::DataPage => {
            read_data_page_v1(desc, meta, hdr, raw, raw_off, dict, out, validity, cache)
        }
        PageType::DataPageV2 => {
            read_data_page_v2(desc, meta, hdr, raw, raw_off, dict, out, validity, cache)
        }
        // A page type that is never actually written. Skip it.
        PageType::IndexPage => Ok(0),
        PageType::DictionaryPage => err!(BadPageHeader),
    }
}

pub(crate) fn decode_dictionary_page(
    desc: &ColumnDesc,
    meta: &ColumnMetaData,
    hdr: &PageHeader,
    raw: &[u8],
    raw_off: u64,
    cache: &dyn PageCache,
) -> Result<Vector> {
    let d = hdr.dict_page.as_ref().ok_or(err_at(Code::BadPageHeader, 0))?;
    let n = check_count(d.num_values)?;
    let page = decompress(meta.codec, raw, hdr.uncompressed_page_size, raw_off, cache)?;
    let mut v = Vector::with_capacity(desc.ty, n);
    decode_plain(desc, &page, n, &mut v)?;
    ensure!(v.len() == n, BadCompressedData, 0);
    Ok(v)
}

/// Read only the non-contiguous set of pages narrowed down by page
/// selection, and return `(value vector, each row's absolute row number
/// within the RowGroup)`.
///
/// While `read_column_chunk` scans headers sequentially, assuming the
/// entire column chunk is one contiguous byte range, this function assumes
/// the caller (`format::parquet`) has already fetched the byte range of each
/// individual page indicated by the `OffsetIndex`, and decodes each page
/// independently. Page boundaries coincide with row boundaries (since v1 only
/// handles flat schemas, one page = a contiguous run of rows), so each row's
/// absolute position can be derived directly from `first_row_index`.
///
/// `dict_buf` is the dictionary page's `(raw bytes including the page
/// header, file offset where it starts)`. `pages` is a list of the same for
/// each data page -- `(raw bytes including the page header, start offset,
/// absolute row number within the RowGroup of that page's first row)` --
/// sorted by ascending row number.
pub fn read_selected_pages(
    desc: &ColumnDesc,
    meta: &ColumnMetaData,
    dict_buf: Option<(&[u8], u64)>,
    pages: &[(&[u8], u64, i64)],
    cache: &dyn PageCache,
) -> Result<(Vector, Vec<u64>)> {
    let mut dict: Option<Vector> = None;
    if let Some((buf, start)) = dict_buf {
        ensure!(!buf.is_empty(), UnexpectedEof, 0);
        let (hdr, hlen) = decode_page_header(buf)?;
        ensure!(hdr.ptype == PageType::DictionaryPage, BadPageHeader, 0);
        let clen = hdr.compressed_page_size as usize;
        ensure!(clen <= buf.len() - hlen, UnexpectedEof, hlen);
        let raw = &buf[hlen..hlen + clen];
        let raw_off = start + hlen as u64;
        dict = Some(decode_dictionary_page(desc, meta, &hdr, raw, raw_off, cache)?);
    }

    let cap = pages.len().max(1);
    let mut out = Vector::with_capacity(desc.ty, cap);
    let mut validity = if desc.max_def_level > 0 { Some(Bitmap::with_capacity(cap)) } else { None };
    let mut abs_rows: Vec<u64> = Vec::with_capacity(cap);

    for &(buf, start, first_row) in pages {
        ensure!(!buf.is_empty(), UnexpectedEof, 0);
        ensure!(first_row >= 0, BadPageHeader, 0);
        let (hdr, hlen) = decode_page_header(buf)?;
        ensure!(hdr.ptype != PageType::DictionaryPage, BadPageHeader, 0);
        let clen = hdr.compressed_page_size as usize;
        ensure!(clen <= buf.len() - hlen, UnexpectedEof, hlen);
        let raw = &buf[hlen..hlen + clen];
        let raw_off = start + hlen as u64;
        let before = out.len();
        decode_one_page(
            desc,
            meta,
            &hdr,
            raw,
            raw_off,
            dict.as_ref(),
            &mut out,
            &mut validity,
            cache,
        )?;
        let added = out.len() - before;
        for k in 0..added {
            abs_rows.push(first_row as u64 + k as u64);
        }
    }

    if let Some(v) = validity {
        debug_assert_eq!(v.len(), out.len());
        out.set_validity(Some(v));
        out.compact_validity();
    }
    ensure!(abs_rows.len() == out.len(), Internal, 0);
    Ok((out, abs_rows))
}

pub(crate) fn err_at(code: Code, pos: usize) -> Error {
    Error::at(code, pos)
}

pub(crate) fn check_count(n: i32) -> Result<usize> {
    ensure!(n >= 0 && (n as usize) <= MAX_PAGE_VALUES, BadPageHeader);
    Ok(n as usize)
}

/// The decompressed page byte slice. Borrowed without copying when uncompressed.
///
/// A codec that isn't built in should already have been delegated to the
/// host, so we look it up in the cache. Failing to find it means the calls
/// were made in the wrong order (no request was issued via `codec_pages`).
pub(crate) fn decompress<'a>(
    codec: Compression,
    raw: &'a [u8],
    out_len: i32,
    raw_off: u64,
    cache: &'a dyn PageCache,
) -> Result<Cow<'a, [u8]>> {
    if codec == Compression::Uncompressed {
        return Ok(Cow::Borrowed(raw));
    }
    if codec.is_builtin() {
        return Ok(Cow::Owned(codec::decompress(codec, raw, out_len.max(0) as usize)?));
    }
    match cache.get(raw_off, raw.len() as u32) {
        Some(d) => Ok(Cow::Borrowed(d)),
        None => err!(UnsupportedCodec),
    }
}

/// Data page v1. Levels and values are packed into the same compressed block.
#[allow(clippy::too_many_arguments)]
fn read_data_page_v1(
    desc: &ColumnDesc,
    meta: &ColumnMetaData,
    hdr: &PageHeader,
    raw: &[u8],
    raw_off: u64,
    dict: Option<&Vector>,
    out: &mut Vector,
    validity: &mut Option<Bitmap>,
    cache: &dyn PageCache,
) -> Result<usize> {
    let dp = hdr.data_page.as_ref().ok_or(err_at(Code::BadPageHeader, 0))?;
    let n = check_count(dp.num_values)?;
    let page = decompress(meta.codec, raw, hdr.uncompressed_page_size, raw_off, cache)?;
    let mut off = 0usize;

    // v1 levels are prefixed with a 4-byte little-endian length.
    let page_validity = if desc.max_def_level > 0 {
        ensure!(dp.definition_level_encoding == Encoding::Rle, UnsupportedEncoding);
        let (bm, used) = read_levels_prefixed(&page, n, desc.max_def_level as u32)?;
        off += used;
        Some(bm)
    } else {
        None
    };

    let present = page_validity.as_ref().map_or(n, |b| b.count_ones());
    ensure!(off <= page.len(), UnexpectedEof, off);
    decode_and_append(
        desc,
        dp.encoding,
        &page[off..],
        n,
        present,
        dict,
        page_validity,
        out,
        validity,
    )?;
    Ok(n)
}

/// Data page v2. Levels sit uncompressed at the start of the page; only the value portion is compressed.
#[allow(clippy::too_many_arguments)]
fn read_data_page_v2(
    desc: &ColumnDesc,
    meta: &ColumnMetaData,
    hdr: &PageHeader,
    raw: &[u8],
    raw_off: u64,
    dict: Option<&Vector>,
    out: &mut Vector,
    validity: &mut Option<Bitmap>,
    cache: &dyn PageCache,
) -> Result<usize> {
    let dp = hdr.data_page_v2.as_ref().ok_or(err_at(Code::BadPageHeader, 0))?;
    let n = check_count(dp.num_values)?;
    let rep_len = check_len(dp.repetition_levels_byte_length)?;
    let def_len = check_len(dp.definition_levels_byte_length)?;
    ensure!(rep_len == 0, UnsupportedNested);
    ensure!(def_len <= raw.len(), UnexpectedEof, 0);

    let page_validity = if desc.max_def_level > 0 {
        ensure!(def_len > 0, BadPageHeader);
        let mut bm = Bitmap::with_capacity(n);
        let bw = encoding::bit_width(desc.max_def_level as u32);
        let mut d = RleDecoder::new(&raw[..def_len], bw);
        d.read_levels_into(n, desc.max_def_level as u32, &mut bm)?;
        Some(bm)
    } else {
        None
    };

    let values_raw = &raw[def_len..];
    let values = if dp.is_compressed {
        // v2 levels sit uncompressed at the start of the page. Only the value
        // portion is compressed, so subtract the levels' share from the decompressed size.
        let want = (hdr.uncompressed_page_size as i64) - (rep_len + def_len) as i64;
        ensure!(want >= 0, BadPageHeader);
        decompress(meta.codec, values_raw, want as i32, raw_off + def_len as u64, cache)?
    } else {
        Cow::Borrowed(values_raw)
    };

    let present = page_validity.as_ref().map_or(n, |b| b.count_ones());
    decode_and_append(desc, dp.encoding, &values, n, present, dict, page_validity, out, validity)?;
    Ok(n)
}

pub(crate) fn check_len(v: i32) -> Result<usize> {
    ensure!(v >= 0, BadPageHeader);
    Ok(v as usize)
}

/// Read a 4-byte-length-prefixed RLE level stream. Returns `(validity, bytes consumed)`.
fn read_levels_prefixed(page: &[u8], n: usize, max_level: u32) -> Result<(Bitmap, usize)> {
    ensure!(page.len() >= 4, UnexpectedEof, 0);
    let len = u32::from_le_bytes([page[0], page[1], page[2], page[3]]) as usize;
    ensure!(len <= page.len() - 4, UnexpectedEof, 4);
    let mut bm = Bitmap::with_capacity(n);
    let bw = encoding::bit_width(max_level);
    let mut d = RleDecoder::new(&page[4..4 + len], bw);
    d.read_levels_into(n, max_level, &mut bm)?;
    Ok((bm, 4 + len))
}

/// Decode the value portion and append it to `out`, filling in NULL positions.
#[allow(clippy::too_many_arguments)]
fn decode_and_append(
    desc: &ColumnDesc,
    enc: Encoding,
    data: &[u8],
    n: usize,
    present: usize,
    dict: Option<&Vector>,
    page_validity: Option<Bitmap>,
    out: &mut Vector,
    validity: &mut Option<Bitmap>,
) -> Result<()> {
    // First build the dense value column (containing no NULLs).
    let dense = decode_dense(desc, enc, data, present, dict)?;
    ensure!(dense.len() == present, BadCompressedData);

    match (&page_validity, validity.as_mut()) {
        (Some(pv), Some(acc)) => acc.extend(pv),
        (None, Some(acc)) => acc.push_n(true, n),
        // A column with no definition level. validity stays None the whole way through.
        (_, None) => {}
    }
    append_scattered(out, &dense, page_validity.as_ref(), n)
}

/// Build a dense value column according to the encoding.
pub(crate) fn decode_dense(
    desc: &ColumnDesc,
    enc: Encoding,
    data: &[u8],
    present: usize,
    dict: Option<&Vector>,
) -> Result<Vector> {
    if enc.is_dictionary() {
        let dict = match dict {
            Some(d) => d,
            None => err!(BadCompressedData),
        };
        // The first byte of the value portion is the bit width.
        ensure!(!data.is_empty(), UnexpectedEof, 0);
        let bw = data[0];
        ensure!(bw <= 32, BadCompressedData, 0);
        let mut idx = Vec::with_capacity(present);
        RleDecoder::new(&data[1..], bw).read_u32(present, &mut idx)?;
        // An index out of the dictionary's range is corruption. Check before gathering.
        let dlen = dict.len() as u32;
        for &i in &idx {
            ensure!(i < dlen, BadCompressedData);
        }
        return Ok(dict.gather(&idx));
    }

    let mut v = Vector::with_capacity(desc.ty, present);
    match enc {
        Encoding::Plain => decode_plain(desc, data, present, &mut v)?,
        Encoding::DeltaBinaryPacked => decode_delta(desc, data, present, &mut v)?,
        Encoding::DeltaLengthByteArray => {
            ensure!(desc.ty.phys() == PhysType::Bytes, UnsupportedEncoding);
            if let Data::Bytes(b) = v.data_mut() {
                encoding::decode_delta_length_byte_array(data, present, b)?;
            }
        }
        Encoding::DeltaByteArray => {
            ensure!(desc.ty.phys() == PhysType::Bytes, UnsupportedEncoding);
            if let Data::Bytes(b) = v.data_mut() {
                encoding::decode_delta_byte_array(data, present, b)?;
            }
        }
        // RLE is also used as the value encoding for BOOLEAN columns.
        Encoding::Rle => {
            ensure!(desc.ptype == PType::Boolean, UnsupportedEncoding);
            ensure!(data.len() >= 4, UnexpectedEof, 0);
            let len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
            ensure!(len <= data.len() - 4, UnexpectedEof, 4);
            let mut bm = Bitmap::with_capacity(present);
            RleDecoder::new(&data[4..4 + len], 1).read_levels_into(present, 1, &mut bm)?;
            *v.data_mut() = Data::Bool(bm);
        }
        _ => err!(UnsupportedEncoding),
    }
    Ok(v)
}

fn decode_delta(desc: &ColumnDesc, data: &[u8], n: usize, out: &mut Vector) -> Result<()> {
    match desc.ptype {
        PType::Int32 => {
            let mut tmp = Vec::with_capacity(n);
            encoding::decode_delta_binary_packed_i32(data, n, &mut tmp)?;
            push_i32_values(desc, &tmp, out)
        }
        PType::Int64 => {
            let mut tmp = Vec::with_capacity(n);
            encoding::decode_delta_binary_packed_i64(data, n, &mut tmp)?;
            push_i64_values(desc, &tmp, out)
        }
        _ => err!(UnsupportedEncoding),
    }
}

// --- PLAIN decoding ---------------------------------------------------------

/// Read `n` PLAIN-encoded values and push them into `out` according to the logical type.
fn decode_plain(desc: &ColumnDesc, data: &[u8], n: usize, out: &mut Vector) -> Result<()> {
    match desc.ptype {
        PType::Boolean => {
            ensure!(data.len() >= n.div_ceil(8), UnexpectedEof, 0);
            *out.data_mut() = Data::Bool(Bitmap::from_lsb_bytes(data, n));
            Ok(())
        }
        PType::Int32 => {
            ensure!(data.len() >= n * 4, UnexpectedEof, 0);
            let vals: Vec<i32> = data[..n * 4]
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            push_i32_values(desc, &vals, out)
        }
        PType::Int64 => {
            ensure!(data.len() >= n * 8, UnexpectedEof, 0);
            let vals: Vec<i64> = data[..n * 8]
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            push_i64_values(desc, &vals, out)
        }
        PType::Int96 => {
            ensure!(data.len() >= n * 12, UnexpectedEof, 0);
            let d = as_i64_vec(out, n)?;
            for c in data[..n * 12].chunks_exact(12) {
                let nanos = i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
                let julian = i32::from_le_bytes([c[8], c[9], c[10], c[11]]) as i64;
                d.push(
                    (julian - JULIAN_EPOCH)
                        .saturating_mul(MICROS_PER_DAY)
                        .saturating_add(nanos / 1000),
                );
            }
            Ok(())
        }
        PType::Float => {
            ensure!(data.len() >= n * 4, UnexpectedEof, 0);
            let d = as_f64_vec(out, n)?;
            for c in data[..n * 4].chunks_exact(4) {
                d.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64);
            }
            Ok(())
        }
        PType::Double => {
            ensure!(data.len() >= n * 8, UnexpectedEof, 0);
            let d = as_f64_vec(out, n)?;
            for c in data[..n * 8].chunks_exact(8) {
                d.push(f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]));
            }
            Ok(())
        }
        PType::ByteArray => {
            let mut off = 0usize;
            let mut items = Vec::with_capacity(n);
            for _ in 0..n {
                ensure!(data.len() - off >= 4, UnexpectedEof, off);
                let len =
                    u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
                        as usize;
                off += 4;
                ensure!(len <= data.len() - off, UnexpectedEof, off);
                items.push(&data[off..off + len]);
                off += len;
            }
            push_byte_values(desc, &items, out)
        }
        PType::FixedLenByteArray => {
            let w = desc.type_length;
            ensure!(w > 0, UnsupportedType);
            ensure!(data.len() >= n * w, UnexpectedEof, 0);
            let items: Vec<&[u8]> = data[..n * w].chunks_exact(w).collect();
            push_byte_values(desc, &items, out)
        }
    }
}

// --- Repacking physical values into logical types ---------------------------

fn as_i32_vec(out: &mut Vector, cap: usize) -> Result<&mut Vec<i32>> {
    match out.data_mut() {
        Data::I32(v) => {
            v.reserve(cap);
            Ok(v)
        }
        _ => err!(Internal),
    }
}

fn as_i64_vec(out: &mut Vector, cap: usize) -> Result<&mut Vec<i64>> {
    match out.data_mut() {
        Data::I64(v) => {
            v.reserve(cap);
            Ok(v)
        }
        _ => err!(Internal),
    }
}

fn as_i128_vec(out: &mut Vector, cap: usize) -> Result<&mut Vec<i128>> {
    match out.data_mut() {
        Data::I128(v) => {
            v.reserve(cap);
            Ok(v)
        }
        _ => err!(Internal),
    }
}

fn as_f64_vec(out: &mut Vector, cap: usize) -> Result<&mut Vec<f64>> {
    match out.data_mut() {
        Data::F64(v) => {
            v.reserve(cap);
            Ok(v)
        }
        _ => err!(Internal),
    }
}

/// Whether `ty` is one of the unsigned integer logical types. Also used by
/// `format::parquet::stat_value`/`plain_encode_for_bloom`, which need to apply the
/// exact same zero-extension this module applies when decoding actual column values,
/// so that pruning compares values on the same scale the reader would produce.
pub(crate) fn is_unsigned(ty: Ty) -> bool {
    matches!(ty, Ty::UTinyInt | Ty::USmallInt | Ty::UInt | Ty::UBigInt)
}

/// Repacking from the INT32 physical type.
fn push_i32_values(desc: &ColumnDesc, vals: &[i32], out: &mut Vector) -> Result<()> {
    match desc.ty.phys() {
        PhysType::I32 => {
            let d = as_i32_vec(out, vals.len())?;
            d.extend_from_slice(vals);
        }
        PhysType::I64 => {
            let unsigned = is_unsigned(desc.ty);
            let unit = desc.time_unit;
            let d = as_i64_vec(out, vals.len())?;
            for &v in vals {
                // UINT32 is widened to 64 bits as unsigned.
                let x = if unsigned { v as u32 as i64 } else { v as i64 };
                // TIME_MILLIS is stored as INT32. Normalize it to microseconds.
                d.push(match unit {
                    Some(u) => u.to_micros(x),
                    None => x,
                });
            }
        }
        PhysType::I128 => {
            let unsigned = is_unsigned(desc.ty);
            let d = as_i128_vec(out, vals.len())?;
            for &v in vals {
                d.push(if unsigned { v as u32 as i128 } else { v as i128 });
            }
        }
        PhysType::F64 => {
            let d = as_f64_vec(out, vals.len())?;
            for &v in vals {
                d.push(v as f64);
            }
        }
        _ => err!(UnsupportedType),
    }
    Ok(())
}

/// Repacking from the INT64 physical type.
fn push_i64_values(desc: &ColumnDesc, vals: &[i64], out: &mut Vector) -> Result<()> {
    match desc.ty.phys() {
        PhysType::I64 => {
            let unit = desc.time_unit;
            let d = as_i64_vec(out, vals.len())?;
            match unit {
                None => d.extend_from_slice(vals),
                Some(u) => {
                    for &v in vals {
                        d.push(u.to_micros(v));
                    }
                }
            }
        }
        PhysType::I128 => {
            let unsigned = is_unsigned(desc.ty);
            let d = as_i128_vec(out, vals.len())?;
            for &v in vals {
                d.push(if unsigned { v as u64 as i128 } else { v as i128 });
            }
        }
        PhysType::F64 => {
            let d = as_f64_vec(out, vals.len())?;
            for &v in vals {
                d.push(v as f64);
            }
        }
        _ => err!(UnsupportedType),
    }
    Ok(())
}

/// Repacking from BYTE_ARRAY / FIXED_LEN_BYTE_ARRAY.
/// DECIMAL is converted to an integer as a big-endian two's complement value.
fn push_byte_values(desc: &ColumnDesc, items: &[&[u8]], out: &mut Vector) -> Result<()> {
    match desc.ty.phys() {
        PhysType::Bytes => {
            match out.data_mut() {
                Data::Bytes(b) => {
                    let total: usize = items.iter().map(|i| i.len()).sum();
                    b.data.reserve(total);
                    b.offsets.reserve(items.len());
                    for it in items {
                        b.push(it);
                    }
                }
                _ => err!(Internal),
            }
            Ok(())
        }
        PhysType::I64 => {
            let d = as_i64_vec(out, items.len())?;
            for it in items {
                let v = be_signed(it)?;
                match i64::try_from(v) {
                    Ok(x) => d.push(x),
                    Err(_) => err!(ValueOutOfRange),
                }
            }
            Ok(())
        }
        PhysType::I128 => {
            let d = as_i128_vec(out, items.len())?;
            for it in items {
                d.push(be_signed(it)?);
            }
            Ok(())
        }
        _ => err!(UnsupportedType),
    }
}

/// Convert a big-endian two's complement representation into an i128.
fn be_signed(b: &[u8]) -> Result<i128> {
    ensure!(b.len() <= 16, ValueOutOfRange);
    if b.is_empty() {
        return Ok(0);
    }
    // If the most significant bit is set, start from all-1 bits and sign-extend.
    let mut v: i128 = if b[0] & 0x80 != 0 { -1 } else { 0 };
    for &x in b {
        v = (v << 8) | (x as i128 & 0xff);
    }
    Ok(v)
}

// --- Appending while filling in NULL positions -------------------------------

/// Push the dense value column `src` into `out`, skipping over the NULL
/// positions indicated by `validity`. Dummy values are inserted at NULL
/// positions (since validity marks them invalid, the actual value doesn't matter).
fn append_scattered(
    out: &mut Vector,
    src: &Vector,
    validity: Option<&Bitmap>,
    n: usize,
) -> Result<()> {
    let validity = match validity {
        // A page with no NULLs at all can be appended wholesale.
        None => return append_all(out, src, n),
        Some(v) if v.count_ones() == n => return append_all(out, src, n),
        Some(v) => v,
    };

    macro_rules! scatter {
        ($ov:expr, $sv:expr, $zero:expr) => {{
            $ov.reserve(n);
            let mut k = 0usize;
            for i in 0..n {
                if validity.get(i) {
                    ensure!(k < $sv.len(), BadCompressedData);
                    $ov.push($sv[k]);
                    k += 1;
                } else {
                    $ov.push($zero);
                }
            }
        }};
    }

    match (out.data_mut(), src.data()) {
        (Data::I32(o), Data::I32(s)) => scatter!(o, s, 0),
        (Data::I64(o), Data::I64(s)) => scatter!(o, s, 0),
        (Data::F64(o), Data::F64(s)) => scatter!(o, s, 0.0),
        (Data::I128(o), Data::I128(s)) => scatter!(o, s, 0),
        (Data::Bool(o), Data::Bool(s)) => {
            let mut k = 0usize;
            for i in 0..n {
                if validity.get(i) {
                    ensure!(k < s.len(), BadCompressedData);
                    o.push(s.get(k));
                    k += 1;
                } else {
                    o.push(false);
                }
            }
        }
        (Data::Bytes(o), Data::Bytes(s)) => {
            let mut k = 0usize;
            for i in 0..n {
                if validity.get(i) {
                    ensure!(k < s.len(), BadCompressedData);
                    o.push(s.get(k));
                    k += 1;
                } else {
                    o.push_empty();
                }
            }
        }
        _ => err!(Internal),
    }
    Ok(())
}

pub(crate) fn append_all(out: &mut Vector, src: &Vector, n: usize) -> Result<()> {
    ensure!(src.len() == n, BadCompressedData);
    match (out.data_mut(), src.data()) {
        (Data::I32(o), Data::I32(s)) => o.extend_from_slice(s),
        (Data::I64(o), Data::I64(s)) => o.extend_from_slice(s),
        (Data::F64(o), Data::F64(s)) => o.extend_from_slice(s),
        (Data::I128(o), Data::I128(s)) => o.extend_from_slice(s),
        (Data::Bool(o), Data::Bool(s)) => o.extend(s),
        (Data::Bytes(o), Data::Bytes(s)) => {
            o.data.reserve(s.data.len());
            o.offsets.reserve(n);
            for i in 0..n {
                o.push(s.get(i));
            }
        }
        _ => err!(Internal),
    }
    Ok(())
}

/// Scan a column chunk and enumerate the pages that need decompression delegated to the host.
///
/// Page headers are uncompressed, so scanning is possible before decoding as
/// long as the bytes are available. This is the key to preserving the
/// property that "the work needed is determined up front, at the start of the
/// split" -- so execution never has to stop partway through (DESIGN.md §6).
pub fn collect_codec_pages(
    meta: &ColumnMetaData,
    buf: &[u8],
    chunk_start: u64,
    num_rows: usize,
    out: &mut Vec<CodecPage>,
) -> Result<()> {
    if meta.codec.is_builtin() {
        return Ok(());
    }
    let mut pos = 0usize;
    let mut rows_done = 0usize;
    while rows_done < num_rows {
        ensure!(pos < buf.len(), UnexpectedEof, pos);
        let (hdr, hlen) = decode_page_header(&buf[pos..])?;
        pos += hlen;
        let clen = hdr.compressed_page_size as usize;
        ensure!(clen <= buf.len() - pos, UnexpectedEof, pos);
        let raw_off = chunk_start + pos as u64;

        push_codec_page(meta, &hdr, raw_off, clen, out)?;

        match hdr.ptype {
            PageType::DataPage => {
                rows_done += hdr.data_page.as_ref().map_or(0, |d| d.num_values.max(0) as usize)
            }
            PageType::DataPageV2 => {
                rows_done += hdr.data_page_v2.as_ref().map_or(0, |d| d.num_values.max(0) as usize)
            }
            _ => {}
        }
        pos += clen;
    }
    Ok(())
}

/// The nested-column variant of `collect_codec_pages`. A REPEATED column's
/// per-page value count doesn't match the row count (it varies with the
/// number of array elements), so the loop ends not by reaching `num_rows` but
/// by exhausting the column chunk's byte range.
pub fn collect_codec_pages_all(
    meta: &ColumnMetaData,
    buf: &[u8],
    chunk_start: u64,
    out: &mut Vec<CodecPage>,
) -> Result<()> {
    if meta.codec.is_builtin() {
        return Ok(());
    }
    let mut pos = 0usize;
    while pos < buf.len() {
        let (hdr, hlen) = decode_page_header(&buf[pos..])?;
        pos += hlen;
        let clen = hdr.compressed_page_size as usize;
        ensure!(clen <= buf.len().saturating_sub(pos), UnexpectedEof, pos);
        let raw_off = chunk_start + pos as u64;
        push_codec_page(meta, &hdr, raw_off, clen, out)?;
        pos += clen;
    }
    Ok(())
}

/// Given one page's header, push a host-delegation entry into `out` if
/// needed. This is the shared core used by both `collect_codec_pages`
/// (sequential scanning of a contiguous buffer) and
/// `collect_codec_pages_selected` (the non-contiguous set of buffers after
/// page selection).
fn push_codec_page(
    meta: &ColumnMetaData,
    hdr: &PageHeader,
    raw_off: u64,
    clen: usize,
    out: &mut Vec<CodecPage>,
) -> Result<()> {
    // v2 has uncompressed levels at the start, so only the range excluding that is delegated.
    let (off, len, out_len) = match (&hdr.data_page_v2, hdr.ptype) {
        (Some(dp), PageType::DataPageV2) => {
            let skip = check_len(dp.repetition_levels_byte_length)?
                + check_len(dp.definition_levels_byte_length)?;
            ensure!(skip <= clen, BadPageHeader);
            if !dp.is_compressed {
                (0, 0, 0)
            } else {
                (
                    raw_off + skip as u64,
                    (clen - skip) as u32,
                    (hdr.uncompressed_page_size as usize).saturating_sub(skip) as u32,
                )
            }
        }
        _ => (raw_off, clen as u32, hdr.uncompressed_page_size.max(0) as u32),
    };
    if len > 0 {
        out.push(CodecPage { codec: meta.codec, offset: off, len, out_len });
    }
    Ok(())
}

/// Enumerate the pages that need decompression delegated to the host, from
/// the non-contiguous set of pages after page selection. The page-selection
/// variant of `collect_codec_pages`. `dict` is the dictionary page's `(raw
/// bytes, start offset)`, and `pages` is the same pair for each data page.
pub fn collect_codec_pages_selected(
    meta: &ColumnMetaData,
    dict: Option<(&[u8], u64)>,
    pages: &[(&[u8], u64)],
    out: &mut Vec<CodecPage>,
) -> Result<()> {
    if meta.codec.is_builtin() {
        return Ok(());
    }
    if let Some((buf, start)) = dict {
        push_codec_page_for(meta, buf, start, out)?;
    }
    for &(buf, start) in pages {
        push_codec_page_for(meta, buf, start, out)?;
    }
    Ok(())
}

fn push_codec_page_for(
    meta: &ColumnMetaData,
    buf: &[u8],
    start: u64,
    out: &mut Vec<CodecPage>,
) -> Result<()> {
    ensure!(!buf.is_empty(), UnexpectedEof, 0);
    let (hdr, hlen) = decode_page_header(buf)?;
    let clen = hdr.compressed_page_size as usize;
    ensure!(clen <= buf.len() - hlen, UnexpectedEof, hlen);
    push_codec_page(meta, &hdr, start + hlen as u64, clen, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn be_signed_handles_sign_extension() {
        assert_eq!(be_signed(&[0x00]).unwrap(), 0);
        assert_eq!(be_signed(&[0x7f]).unwrap(), 127);
        assert_eq!(be_signed(&[0xff]).unwrap(), -1);
        assert_eq!(be_signed(&[0x80]).unwrap(), -128);
        assert_eq!(be_signed(&[0x01, 0x00]).unwrap(), 256);
        assert_eq!(be_signed(&[0xff, 0xff]).unwrap(), -1);
        assert_eq!(be_signed(&[0x00, 0x00, 0x30, 0x39]).unwrap(), 12345);
        assert_eq!(be_signed(&[0xff, 0xff, 0xcf, 0xc7]).unwrap(), -12345);
        assert!(be_signed(&[0u8; 17]).is_err());
    }

    #[test]
    fn int96_epoch_conversion() {
        // Julian day number 2440588 = 1970-01-01. Nanoseconds of 0 means the epoch.
        let mut data = Vec::new();
        data.extend_from_slice(&0i64.to_le_bytes());
        data.extend_from_slice(&(JULIAN_EPOCH as i32).to_le_bytes());
        // One second into the next day
        data.extend_from_slice(&1_000_000_000i64.to_le_bytes());
        data.extend_from_slice(&(JULIAN_EPOCH as i32 + 1).to_le_bytes());

        let desc = ColumnDesc {
            name: "t".into(),
            ty: Ty::Timestamp,
            nullable: false,
            max_def_level: 0,
            ptype: PType::Int96,
            type_length: 0,
            time_unit: None,
            phys_cols: Vec::new(),
            leaves: Vec::new(),
            nested: None,
        };
        let mut v = Vector::with_capacity(Ty::Timestamp, 2);
        decode_plain(&desc, &data, 2, &mut v).unwrap();
        assert_eq!(v.i64s(), &[0, MICROS_PER_DAY + 1_000_000]);
    }

    #[test]
    fn unsigned_int32_widens_without_sign_extension() {
        let desc = ColumnDesc {
            name: "u".into(),
            ty: Ty::UInt,
            nullable: false,
            max_def_level: 0,
            ptype: PType::Int32,
            type_length: 0,
            time_unit: None,
            phys_cols: Vec::new(),
            leaves: Vec::new(),
            nested: None,
        };
        let mut v = Vector::with_capacity(Ty::UInt, 2);
        // -1i32 is 4294967295 as a u32.
        push_i32_values(&desc, &[-1, 7], &mut v).unwrap();
        assert_eq!(v.i64s(), &[4_294_967_295, 7]);
    }

    #[test]
    fn millis_are_normalised_to_micros() {
        let desc = ColumnDesc {
            name: "ts".into(),
            ty: Ty::Timestamp,
            nullable: false,
            max_def_level: 0,
            ptype: PType::Int64,
            type_length: 0,
            time_unit: Some(TimeUnit::Millis),
            phys_cols: Vec::new(),
            leaves: Vec::new(),
            nested: None,
        };
        let mut v = Vector::with_capacity(Ty::Timestamp, 2);
        push_i64_values(&desc, &[1, 1_700_000_000_000], &mut v).unwrap();
        assert_eq!(v.i64s(), &[1_000, 1_700_000_000_000_000]);
    }

    #[test]
    fn scatter_inserts_placeholders_at_nulls() {
        let mut src = Vector::with_capacity(Ty::Int, 3);
        if let Data::I32(d) = src.data_mut() {
            d.extend_from_slice(&[10, 20, 30]);
        }
        // Rows 1 and 3 (of 5) are NULL
        let mut bm = Bitmap::with_capacity(5);
        for b in [true, false, true, false, true] {
            bm.push(b);
        }
        let mut out = Vector::with_capacity(Ty::Int, 5);
        append_scattered(&mut out, &src, Some(&bm), 5).unwrap();
        assert_eq!(out.i32s(), &[10, 0, 20, 0, 30]);
    }

    #[test]
    fn scatter_detects_missing_dense_values() {
        let mut src = Vector::with_capacity(Ty::Int, 1);
        if let Data::I32(d) = src.data_mut() {
            d.push(1);
        }
        let mut bm = Bitmap::with_capacity(3);
        for b in [true, true, true] {
            bm.push(b);
        }
        let mut out = Vector::with_capacity(Ty::Int, 3);
        // validity requires 3 values, but the dense column has only 1.
        assert!(append_scattered(&mut out, &src, Some(&bm), 3).is_err());
    }

    #[test]
    fn plain_byte_array_bounds_are_checked() {
        let desc = ColumnDesc {
            name: "s".into(),
            ty: Ty::Varchar,
            nullable: false,
            max_def_level: 0,
            ptype: PType::ByteArray,
            type_length: 0,
            time_unit: None,
            phys_cols: Vec::new(),
            leaves: Vec::new(),
            nested: None,
        };
        // Declares length 100, but the actual data is only 2 bytes.
        let data = [100u8, 0, 0, 0, b'a', b'b'];
        let mut v = Vector::with_capacity(Ty::Varchar, 1);
        assert_eq!(
            crate::error::code_of(decode_plain(&desc, &data, 1, &mut v)),
            Some(Code::UnexpectedEof)
        );
    }
}
