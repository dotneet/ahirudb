//! 列チャンク（辞書ページ + データページ群）→ `Vector`。
//!
//! v1 はフラットスキーマのみを扱うので repetition level は常に 0 であり、
//! definition level は 0/1 の 1 ビットで表せる。これは validity ビットマップ
//! そのものなので、レベル配列を実体化せず直接ビットマップに落とす。
//!
//! 論理型ごとの分岐はここに閉じ込める。ここから下流（式 VM、オペレータ）は
//! 物理型 6 種しか見ない（DESIGN.md §8）。

use alloc::borrow::Cow;

use crate::parquet::codec;
use crate::parquet::encoding::{self, RleDecoder};
use crate::parquet::meta::{decode_page_header, ColumnMetaData, PageHeader};
use crate::parquet::schema::ColumnDesc;
use crate::parquet::*;
use crate::prelude::*;
use crate::vector::{Bitmap, BytesData, Data, PhysType, Ty, Vector};

/// Unix エポック (1970-01-01) のユリウス通日。INT96 の変換に使う。
const JULIAN_EPOCH: i64 = 2_440_588;
const MICROS_PER_DAY: i64 = 86_400_000_000;

/// 1 ページあたりの値数の上限。破損したヘッダによる巨大確保を防ぐ。
const MAX_PAGE_VALUES: usize = 1 << 26;

/// ホストが展開したページの置き場。
///
/// 内蔵していないコーデック（GZIP / ZSTD）はホストに展開を委譲する
/// （DESIGN.md §6）。委譲した結果はここから引く。キーは圧縮ページ本体の
/// **ファイル上の絶対オフセットと長さ**で、ページ 1 つに 1 エントリ対応する。
pub trait PageCache {
    fn get(&self, offset: u64, len: u32) -> Option<&[u8]>;
}

/// 委譲を使わない場合のダミー。内蔵コーデックだけのファイルではこれで足りる。
pub struct NoPageCache;

impl PageCache for NoPageCache {
    fn get(&self, _offset: u64, _len: u32) -> Option<&[u8]> {
        None
    }
}

/// ホストに展開してもらう必要のあるページ。
#[derive(Clone, Copy)]
pub struct CodecPage {
    pub codec: Compression,
    /// 圧縮データ本体のファイル上の位置と長さ。
    pub offset: u64,
    pub len: u32,
    /// ページヘッダが宣言する展開後サイズ。
    pub out_len: u32,
}

/// 列チャンク全体をデコードして 1 本のベクタにする。
///
/// `buf` は `ColumnMetaData::byte_range()` が示す範囲のバイト列。
/// `num_rows` はその RowGroup の行数。
pub fn read_column_chunk(
    desc: &ColumnDesc,
    meta: &ColumnMetaData,
    buf: &[u8],
    chunk_start: u64,
    num_rows: usize,
    cache: &dyn PageCache,
) -> Result<Vector> {
    let mut out = Vector::with_capacity(desc.ty, num_rows);
    // definition level を持つ列だけ validity を積む。
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
                let d = hdr.dict_page.as_ref().ok_or(err_at(Code::BadPageHeader, pos))?;
                let n = check_count(d.num_values)?;
                let page = decompress(meta.codec, raw, hdr.uncompressed_page_size, raw_off, cache)?;
                let mut v = Vector::with_capacity(desc.ty, n);
                decode_plain(desc, &page, n, &mut v)?;
                ensure!(v.len() == n, BadCompressedData, pos);
                dict = Some(v);
            }
            PageType::DataPage => {
                rows_done += read_data_page_v1(
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
            PageType::DataPageV2 => {
                rows_done += read_data_page_v2(
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
            // 実際には書かれないページ種別。読み飛ばす。
            PageType::IndexPage => {}
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

fn err_at(code: Code, pos: usize) -> Error {
    Error::at(code, pos)
}

fn check_count(n: i32) -> Result<usize> {
    ensure!(n >= 0 && (n as usize) <= MAX_PAGE_VALUES, BadPageHeader);
    Ok(n as usize)
}

/// 展開後のページバイト列。非圧縮ならコピーせず借用する。
///
/// 内蔵していないコーデックはホストに委譲済みのはずなので、キャッシュから引く。
/// 引けなければ呼び出し順序の誤り（`codec_pages` で要求を出していない）。
fn decompress<'a>(
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

/// データページ v1。レベルと値が同じ圧縮ブロックに入っている。
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

    // v1 のレベルは 4 バイトのリトルエンディアン長が前置される。
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

/// データページ v2。レベルは非圧縮でページ先頭に置かれ、値部分だけが圧縮される。
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
        let mut d = RleDecoder::new(&raw[..def_len], 1);
        d.read_levels_into(n, desc.max_def_level as u32, &mut bm)?;
        Some(bm)
    } else {
        None
    };

    let values_raw = &raw[def_len..];
    let values = if dp.is_compressed {
        // v2 はレベルが非圧縮でページ先頭に置かれる。圧縮されているのは値の部分
        // だけなので、展開後サイズからレベル分を差し引く。
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

fn check_len(v: i32) -> Result<usize> {
    ensure!(v >= 0, BadPageHeader);
    Ok(v as usize)
}

/// 4 バイト長前置の RLE レベルストリームを読む。返り値は `(validity, 消費バイト数)`。
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

/// 値部分をデコードし、NULL 位置を埋めながら `out` に追記する。
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
    // 密な値列（NULL を含まない）をまず作る。
    let dense = decode_dense(desc, enc, data, present, dict)?;
    ensure!(dense.len() == present, BadCompressedData);

    match (&page_validity, validity.as_mut()) {
        (Some(pv), Some(acc)) => acc.extend(pv),
        (None, Some(acc)) => acc.push_n(true, n),
        // definition level が無い列。validity は最後まで None のまま。
        (_, None) => {}
    }
    append_scattered(out, &dense, page_validity.as_ref(), n)
}

/// エンコーディングに応じて密な値列を作る。
fn decode_dense(
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
        // 値部分の先頭 1 バイトがビット幅。
        ensure!(!data.is_empty(), UnexpectedEof, 0);
        let bw = data[0];
        ensure!(bw <= 32, BadCompressedData, 0);
        let mut idx = Vec::with_capacity(present);
        RleDecoder::new(&data[1..], bw).read_u32(present, &mut idx)?;
        // 辞書の範囲外インデックスは破損。gather 前に検査する。
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
        // RLE は BOOLEAN 列の値エンコーディングとしても使われる。
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

// --- PLAIN デコード ---------------------------------------------------------

/// PLAIN エンコードされた `n` 個の値を読み、論理型に合わせて `out` に積む。
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

// --- 物理値 → 論理型への詰め替え -------------------------------------------

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

fn is_unsigned(ty: Ty) -> bool {
    matches!(ty, Ty::UTinyInt | Ty::USmallInt | Ty::UInt | Ty::UBigInt)
}

/// INT32 物理型からの詰め替え。
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
                // UINT32 は符号なしとして 64 ビットへ広げる。
                let x = if unsigned { v as u32 as i64 } else { v as i64 };
                // TIME_MILLIS は INT32 に格納される。マイクロ秒へ正規化する。
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

/// INT64 物理型からの詰め替え。
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

/// BYTE_ARRAY / FIXED_LEN_BYTE_ARRAY からの詰め替え。
/// DECIMAL はビッグエンディアンの 2 の補数として整数に変換する。
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

/// ビッグエンディアンの 2 の補数表現を i128 にする。
fn be_signed(b: &[u8]) -> Result<i128> {
    ensure!(b.len() <= 16, ValueOutOfRange);
    if b.is_empty() {
        return Ok(0);
    }
    // 最上位ビットが立っていれば全ビット 1 から始めて符号拡張する。
    let mut v: i128 = if b[0] & 0x80 != 0 { -1 } else { 0 };
    for &x in b {
        v = (v << 8) | (x as i128 & 0xff);
    }
    Ok(v)
}

// --- NULL 位置を埋めながら追記 ----------------------------------------------

/// 密な値列 `src` を、`validity` が示す NULL 位置を飛ばしながら `out` に積む。
/// NULL 位置にはダミー値を入れる（validity 側で無効と分かるため値は問わない）。
fn append_scattered(
    out: &mut Vector,
    src: &Vector,
    validity: Option<&Bitmap>,
    n: usize,
) -> Result<()> {
    let validity = match validity {
        // NULL が 1 つも無いページは丸ごと連結できる。
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

fn append_all(out: &mut Vector, src: &Vector, n: usize) -> Result<()> {
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

/// 列チャンク内を走査し、ホストに展開を委譲すべきページを列挙する。
///
/// ページヘッダは非圧縮なので、バイトさえ揃っていれば復号前に走査できる。
/// これが「分割の開始時点で必要な作業が確定する」性質を保つ鍵で、実行の
/// 途中で止まらずに済む（DESIGN.md §6）。
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

        // v2 はレベルが非圧縮で先頭に載るので、その分を除いた範囲だけを委譲する。
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

/// `BytesData` を直接触る必要がある箇所のための再エクスポート。
pub(crate) type _BytesData = BytesData;

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
        // ユリウス通日 2440588 = 1970-01-01。ナノ秒 0 ならエポック。
        let mut data = Vec::new();
        data.extend_from_slice(&0i64.to_le_bytes());
        data.extend_from_slice(&(JULIAN_EPOCH as i32).to_le_bytes());
        // 翌日の 1 秒後
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
        };
        let mut v = Vector::with_capacity(Ty::UInt, 2);
        // -1i32 は u32 では 4294967295。
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
        // 5 行中 1,3 行目が NULL
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
        // validity は 3 個の値を要求するが密な列には 1 個しかない。
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
        };
        // 長さ 100 を宣言しているが実データは 2 バイトしかない。
        let data = [100u8, 0, 0, 0, b'a', b'b'];
        let mut v = Vector::with_capacity(Ty::Varchar, 1);
        assert_eq!(
            crate::error::code_of(decode_plain(&desc, &data, 1, &mut v)),
            Some(Code::UnexpectedEof)
        );
    }
}
