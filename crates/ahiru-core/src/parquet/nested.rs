//! Assemble columns containing REPEATED fields (LIST/MAP, etc.) via the Dremel
//! algorithm into a single `Ty::Json` column (physical representation: UTF-8
//! JSON text, `PhysType::Bytes`).
//!
//! ## Key points of Dremel assembly
//!
//! Each value of a Parquet REPEATED field carries a repetition level (which
//! level a new repetition started at) and a definition level (where along the
//! path it broke off, if NULL). Whether a given node "is present" can be
//! determined by `def_level >= node.def_depth`, and whether a REPEATED node
//! "is still the same array" can be determined by
//! `next.rep_level >= node.rep_depth` (`node.def_depth`/`node.rep_depth` are
//! precomputed during schema resolution by accumulating the entire path in
//! `schema::build_nested_node`).
//!
//! Using this, the content of one leaf reduces to a simple recursion: "peek
//! at one representative leaf; if it's absent, discard one boundary entry
//! from every leaf underneath and produce NULL; if it's present, assemble the
//! contents (and if REPEATED, repeat this as an array)". Multiple leaves
//! (STRUCT fields, MAP key/value) never reference each other's cursors at
//! all -- each independently decides how many entries to consume, looking
//! only at its own def_level/rep_level (guaranteed by Parquet's shredding
//! convention).
//!
//! ## Relation to the I/O barrier
//!
//! Page reading itself follows the existing mechanism (page headers are
//! uncompressed, so they can be scanned before decoding; codecs that aren't
//! built in are delegated to the host). Since a nested column spans multiple
//! physical column chunks, `format::parquet` determines the byte range for
//! every leaf up front, at the start of the split, and always passes "the
//! entire column chunk for that leaf" here (there is no per-page narrowing:
//! REPEATED columns don't have a 1:1 ratio between a page's value count and
//! the row count, so the existing page-selection logic, which assumes
//! `first_row_index`, can't be reused as-is).

use alloc::borrow::Cow;

use crate::expr::{funcs, kernels};
use crate::parquet::encoding::{self, RleDecoder};
use crate::parquet::meta::{decode_page_header, ColumnMetaData, PageHeader};
use crate::parquet::reader::{self, PageCache};
use crate::parquet::schema::{ColumnDesc, LeafDecodeInfo, NestedContent, NestedNode};
use crate::parquet::*;
use crate::prelude::*;
use crate::vector::{Ty, Value, Vector};

/// A JSON value assembled dynamically. Numbers are embedded directly as
/// pre-formatted token byte sequences (we reuse the existing
/// `expr::kernels::fmt_int`/`fmt_f64` rather than rolling our own decimal
/// conversion, and `expr::funcs::fmt_*` for date/time; both are existing
/// `pub(crate)` implementations within this crate, and neither uses
/// `format!`/`core::fmt`).
enum JsonValue {
    Null,
    Bool(bool),
    /// A valid JSON number token (sign, digits, decimal point, exponent only).
    Num(Vec<u8>),
    /// Raw byte string. Escaped at serialization time.
    Str(Vec<u8>),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

// --- Serialization ------------------------------------------------------

fn write_json(v: &JsonValue, out: &mut Vec<u8>) {
    match v {
        JsonValue::Null => out.extend_from_slice(b"null"),
        JsonValue::Bool(b) => out.extend_from_slice(if *b { b"true" } else { b"false" }),
        JsonValue::Num(tok) => out.extend_from_slice(tok),
        JsonValue::Str(bytes) => write_json_string(bytes, out),
        JsonValue::Array(items) => {
            out.push(b'[');
            for (i, it) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_json(it, out);
            }
            out.push(b']');
        }
        JsonValue::Object(fields) => {
            out.push(b'{');
            for (i, (k, fv)) in fields.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_json_string(k.as_bytes(), out);
                out.push(b':');
                write_json(fv, out);
            }
            out.push(b'}');
        }
    }
}

fn write_json_string(bytes: &[u8], out: &mut Vec<u8>) {
    out.push(b'"');
    for &b in bytes {
        match b {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            0x08 => out.extend_from_slice(b"\\b"),
            0x0C => out.extend_from_slice(b"\\f"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            0x00..=0x1F => {
                out.extend_from_slice(b"\\u00");
                out.push(hex_digit(b >> 4));
                out.push(hex_digit(b & 0xF));
            }
            _ => out.push(b),
        }
    }
    out.push(b'"');
}

fn hex_digit(n: u8) -> u8 {
    if n < 10 {
        b'0' + n
    } else {
        b'a' + (n - 10)
    }
}

fn hex_encode(bytes: &[u8], out: &mut Vec<u8>) {
    for &b in bytes {
        out.push(hex_digit(b >> 4));
        out.push(hex_digit(b & 0xF));
    }
}

/// Convert a leaf's `Value` into its JSON representation according to its
/// logical type. Number formatting is delegated to `expr::kernels`, and
/// date/time to the existing `expr::funcs` implementations (using the same
/// foundation as this crate's CAST and CSV/JSONL export avoids discrepancies
/// from a separate rounding/formatting implementation).
fn leaf_value_to_json(ty: Ty, v: Value) -> JsonValue {
    match v {
        Value::Null => JsonValue::Null,
        Value::Bool(b) => JsonValue::Bool(b),
        Value::I32(x) => int_like_to_json(ty, x as i128),
        Value::I64(x) => match ty {
            Ty::Time => {
                let mut b = Vec::new();
                funcs::fmt_time(x, &mut b);
                JsonValue::Str(b)
            }
            Ty::Timestamp => {
                let mut b = Vec::new();
                funcs::fmt_timestamp(x, &mut b);
                JsonValue::Str(b)
            }
            Ty::Timestamptz => {
                let mut b = Vec::new();
                funcs::fmt_timestamptz(x, &mut b);
                JsonValue::Str(b)
            }
            _ => int_like_to_json(ty, x as i128),
        },
        Value::I128(x) => int_like_to_json(ty, x),
        Value::F64(x) => {
            let mut b = Vec::new();
            if x.is_finite() {
                kernels::fmt_f64(x, &mut b);
            } else {
                // JSON has no NaN/Infinity. DuckDB's to_json collapses these to NULL too.
                b.extend_from_slice(b"null");
            }
            JsonValue::Num(b)
        }
        Value::Bytes(b) => match ty {
            Ty::Varchar => JsonValue::Str(b),
            Ty::Uuid => {
                let mut h = Vec::new();
                if let Ok(raw) = <[u8; 16]>::try_from(b.as_slice()) {
                    funcs::fmt_uuid(&raw, &mut h);
                }
                JsonValue::Str(h)
            }
            // BLOB has no direct JSON equivalent, so it becomes a hex string.
            _ => {
                let mut h = Vec::new();
                hex_encode(&b, &mut h);
                JsonValue::Str(h)
            }
        },
    }
}

fn int_like_to_json(ty: Ty, x: i128) -> JsonValue {
    match ty {
        Ty::Decimal { scale, .. } => {
            let mut b = Vec::new();
            kernels::fmt_int(x.unsigned_abs(), x < 0, scale, &mut b);
            JsonValue::Num(b)
        }
        Ty::Date => {
            let mut b = Vec::new();
            funcs::fmt_date(x as i64, &mut b);
            JsonValue::Str(b)
        }
        _ => {
            let mut b = Vec::new();
            kernels::fmt_int(x.unsigned_abs(), x < 0, 0, &mut b);
            JsonValue::Num(b)
        }
    }
}

// --- Reading levels + values per leaf --------------------------------------

/// The repetition/definition levels across every page for one leaf column,
/// plus a dense value vector packed with only the entries that have a value.
struct LeafRuns {
    rep: Vec<u16>,
    def: Vec<u16>,
    values: Vector,
}

/// v1: read a 4-byte-length-prefixed RLE level stream. When `max_level == 0`
/// the stream itself is omitted (every entry is assumed to be 0).
fn read_levels_v1(page: &[u8], n: usize, max_level: u16) -> Result<(Vec<u16>, usize)> {
    if max_level == 0 {
        return Ok((vec![0u16; n], 0));
    }
    ensure!(page.len() >= 4, UnexpectedEof, 0);
    let len = u32::from_le_bytes([page[0], page[1], page[2], page[3]]) as usize;
    ensure!(len <= page.len() - 4, UnexpectedEof, 4);
    let bw = encoding::bit_width(max_level as u32);
    let mut d = RleDecoder::new(&page[4..4 + len], bw);
    let mut raw = Vec::with_capacity(n);
    d.read_u32(n, &mut raw)?;
    Ok((raw.into_iter().map(|v| v as u16).collect(), 4 + len))
}

/// v2: the length is already known from a `DataPageHeaderV2` field, so there is no prefix.
fn read_levels_v2(data: &[u8], n: usize, max_level: u16) -> Result<Vec<u16>> {
    if max_level == 0 {
        return Ok(vec![0u16; n]);
    }
    let bw = encoding::bit_width(max_level as u32);
    let mut d = RleDecoder::new(data, bw);
    let mut raw = Vec::with_capacity(n);
    d.read_u32(n, &mut raw)?;
    Ok(raw.into_iter().map(|v| v as u16).collect())
}

/// Read a single v1 data page, consuming repetition level -> definition
/// level -> values in that order (the v1 in-page layout). Unlike
/// `reader::read_data_page_v1`, this also reads the repetition level, and it
/// appends values densely instead of scattering them with NULLs interleaved.
#[allow(clippy::too_many_arguments)]
fn read_nested_page_v1(
    desc: &ColumnDesc,
    meta: &ColumnMetaData,
    hdr: &PageHeader,
    raw: &[u8],
    raw_off: u64,
    dict: Option<&Vector>,
    max_def_level: u16,
    max_rep_level: u16,
    rep_out: &mut Vec<u16>,
    def_out: &mut Vec<u16>,
    values_out: &mut Vector,
    cache: &dyn PageCache,
) -> Result<()> {
    let dp = hdr.data_page.as_ref().ok_or(reader::err_at(Code::BadPageHeader, 0))?;
    let n = reader::check_count(dp.num_values)?;
    let page = reader::decompress(meta.codec, raw, hdr.uncompressed_page_size, raw_off, cache)?;
    let mut off = 0usize;

    ensure!(dp.repetition_level_encoding == Encoding::Rle, UnsupportedEncoding);
    ensure!(off <= page.len(), UnexpectedEof, off);
    let (rep_levels, used) = read_levels_v1(&page[off..], n, max_rep_level)?;
    off += used;

    ensure!(dp.definition_level_encoding == Encoding::Rle, UnsupportedEncoding);
    ensure!(off <= page.len(), UnexpectedEof, off);
    let (def_levels, used2) = read_levels_v1(&page[off..], n, max_def_level)?;
    off += used2;

    let present = def_levels.iter().filter(|&&d| d == max_def_level).count();
    ensure!(off <= page.len(), UnexpectedEof, off);
    let dense = reader::decode_dense(desc, dp.encoding, &page[off..], present, dict)?;
    ensure!(dense.len() == present, BadCompressedData);
    reader::append_all(values_out, &dense, present)?;
    rep_out.extend_from_slice(&rep_levels);
    def_out.extend_from_slice(&def_levels);
    Ok(())
}

/// Read a single v2 data page. The levels sit uncompressed at the start of
/// the page, and only the value portion is compressed (if at all).
#[allow(clippy::too_many_arguments)]
fn read_nested_page_v2(
    desc: &ColumnDesc,
    meta: &ColumnMetaData,
    hdr: &PageHeader,
    raw: &[u8],
    raw_off: u64,
    dict: Option<&Vector>,
    max_def_level: u16,
    max_rep_level: u16,
    rep_out: &mut Vec<u16>,
    def_out: &mut Vec<u16>,
    values_out: &mut Vector,
    cache: &dyn PageCache,
) -> Result<()> {
    let dp = hdr.data_page_v2.as_ref().ok_or(reader::err_at(Code::BadPageHeader, 0))?;
    let n = reader::check_count(dp.num_values)?;
    let rep_len = reader::check_len(dp.repetition_levels_byte_length)?;
    let def_len = reader::check_len(dp.definition_levels_byte_length)?;
    ensure!(rep_len <= raw.len(), UnexpectedEof, 0);
    ensure!(def_len <= raw.len() - rep_len, UnexpectedEof, rep_len);
    let rep_levels = read_levels_v2(&raw[..rep_len], n, max_rep_level)?;
    let def_levels = read_levels_v2(&raw[rep_len..rep_len + def_len], n, max_def_level)?;

    let present = def_levels.iter().filter(|&&d| d == max_def_level).count();
    let skip = rep_len + def_len;
    let values_raw = &raw[skip..];
    let values = if dp.is_compressed {
        let want = (hdr.uncompressed_page_size as i64) - skip as i64;
        ensure!(want >= 0, BadPageHeader);
        reader::decompress(meta.codec, values_raw, want as i32, raw_off + skip as u64, cache)?
    } else {
        Cow::Borrowed(values_raw)
    };
    let dense = reader::decode_dense(desc, dp.encoding, &values, present, dict)?;
    ensure!(dense.len() == present, BadCompressedData);
    reader::append_all(values_out, &dense, present)?;
    rep_out.extend_from_slice(&rep_levels);
    def_out.extend_from_slice(&def_levels);
    Ok(())
}

/// Read one leaf's entire column chunk (dictionary page + data pages) into
/// repetition/definition levels and a dense value vector. Unlike the flat
/// column's `reader::read_column_chunk`, the loop's termination condition is
/// exhausting the buffer rather than reaching the row count (because a
/// REPEATED column's per-page value count doesn't map directly onto the row
/// count).
fn read_nested_leaf_chunk(
    info: &LeafDecodeInfo,
    meta: &ColumnMetaData,
    buf: &[u8],
    chunk_start: u64,
    cache: &dyn PageCache,
) -> Result<LeafRuns> {
    // A throwaway ColumnDesc, only for reusing decode_dense and friends.
    // Only the information needed for physical decoding (ty/ptype/type_length/time_unit) is filled in.
    let desc = ColumnDesc {
        name: String::new(),
        ty: info.ty,
        nullable: true,
        max_def_level: info.max_def_level,
        ptype: info.ptype,
        type_length: info.type_length,
        time_unit: info.time_unit,
        phys_cols: Vec::new(),
        leaves: Vec::new(),
        nested: None,
    };
    let mut rep = Vec::new();
    let mut def = Vec::new();
    let mut values = Vector::with_capacity(info.ty, 0);
    let mut dict: Option<Vector> = None;
    let mut pos = 0usize;

    while pos < buf.len() {
        let (hdr, hlen) = decode_page_header(&buf[pos..])?;
        pos += hlen;
        let clen = hdr.compressed_page_size as usize;
        ensure!(clen <= buf.len().saturating_sub(pos), UnexpectedEof, pos);
        let raw_off = chunk_start + pos as u64;
        let raw = &buf[pos..pos + clen];
        pos += clen;

        match hdr.ptype {
            PageType::DictionaryPage => {
                dict =
                    Some(reader::decode_dictionary_page(&desc, meta, &hdr, raw, raw_off, cache)?);
            }
            PageType::DataPage => read_nested_page_v1(
                &desc,
                meta,
                &hdr,
                raw,
                raw_off,
                dict.as_ref(),
                info.max_def_level,
                info.max_rep_level,
                &mut rep,
                &mut def,
                &mut values,
                cache,
            )?,
            PageType::DataPageV2 => read_nested_page_v2(
                &desc,
                meta,
                &hdr,
                raw,
                raw_off,
                dict.as_ref(),
                info.max_def_level,
                info.max_rep_level,
                &mut rep,
                &mut def,
                &mut values,
                cache,
            )?,
            // A page type that is never actually written. Skip it.
            PageType::IndexPage => {}
        }
    }
    Ok(LeafRuns { rep, def, values })
}

// --- Dremel assembly -----------------------------------------------------

/// A cursor for advancing through one leaf. `pos` indexes raw entries
/// (including NULLs); `val_idx` is a separate index counting only entries
/// that have a value.
struct LeafCursor<'a> {
    ty: Ty,
    rep: &'a [u16],
    def: &'a [u16],
    values: &'a Vector,
    pos: usize,
    val_idx: usize,
}

impl<'a> LeafCursor<'a> {
    /// The `(repetition level, definition level)` at the current position.
    /// Running out mid-row means a corrupted file (the declared row count
    /// disagrees with the actual level arrays).
    #[inline]
    fn peek(&self) -> Result<(u16, u16)> {
        match (self.rep.get(self.pos), self.def.get(self.pos)) {
            (Some(&r), Some(&d)) => Ok((r, d)),
            _ => err!(BadCompressedData),
        }
    }

    /// Used to decide whether repetition continues. Reaching the end of the
    /// column (no next leaf/row) is a normal condition, so this returns
    /// `None`.
    #[inline]
    fn peek_opt(&self) -> Option<(u16, u16)> {
        match (self.rep.get(self.pos), self.def.get(self.pos)) {
            (Some(&r), Some(&d)) => Some((r, d)),
            _ => None,
        }
    }

    #[inline]
    fn advance_raw(&mut self) {
        self.pos += 1;
    }

    #[inline]
    fn take_value(&mut self) -> Value {
        let v = self.values.value_at(self.val_idx);
        self.val_idx += 1;
        v
    }
}

/// Assemble one node. An array if REPEATED, otherwise a single value
/// (which may be NULL).
fn assemble(node: &NestedNode, cursors: &mut [LeafCursor]) -> Result<JsonValue> {
    if node.repetition == Repetition::Repeated {
        assemble_repeated(node, cursors)
    } else {
        assemble_single(node, cursors)
    }
}

/// A non-REPEATED node. If the representative leaf's definition level
/// doesn't reach this node's `def_depth`, consume one boundary entry from
/// every leaf underneath and return NULL.
fn assemble_single(node: &NestedNode, cursors: &mut [LeafCursor]) -> Result<JsonValue> {
    let (_, def) = cursors[node.rep_leaf].peek()?;
    if def < node.def_depth {
        consume_boundary(node, cursors);
        return Ok(JsonValue::Null);
    }
    render_present(node, cursors)
}

/// A REPEATED node. Zero elements yields an empty array (just consuming one
/// boundary entry). With one or more elements, keep reading elements as
/// "continuations of the same array" for as long as the representative
/// leaf's repetition level is at least this node's `rep_depth`.
fn assemble_repeated(node: &NestedNode, cursors: &mut [LeafCursor]) -> Result<JsonValue> {
    let mut items = Vec::new();
    loop {
        let (_, def) = cursors[node.rep_leaf].peek()?;
        if def < node.def_depth {
            consume_boundary(node, cursors);
            break;
        }
        items.push(render_element(node, cursors)?);
        match cursors[node.rep_leaf].peek_opt() {
            Some((next_rep, _)) if next_rep >= node.rep_depth => continue,
            _ => break,
        }
    }
    Ok(JsonValue::Array(items))
}

/// Once a node is determined to be "absent", consume exactly one boundary
/// entry from every leaf underneath it (Parquet's shredding convention
/// guarantees that, no matter where along the path things broke off, every
/// leaf underneath always has this one entry).
fn consume_boundary(node: &NestedNode, cursors: &mut [LeafCursor]) {
    match &node.content {
        NestedContent::Leaf(idx) => cursors[*idx].advance_raw(),
        NestedContent::Group(children) => {
            for c in children {
                consume_boundary(c, cursors);
            }
        }
    }
}

/// Render the "present" contents of a non-REPEATED node. If it has exactly
/// one child and that child is itself REPEATED (a LIST/MAP wrapper group),
/// delegate straight through without creating a name (otherwise we'd get an
/// extra layer of nesting like `{"list": [...]}`).
fn render_present(node: &NestedNode, cursors: &mut [LeafCursor]) -> Result<JsonValue> {
    match &node.content {
        NestedContent::Leaf(idx) => Ok(take_leaf_value(*idx, cursors)),
        NestedContent::Group(children) => {
            if children.len() == 1 && children[0].repetition == Repetition::Repeated {
                return assemble(&children[0], cursors);
            }
            let mut obj = Vec::with_capacity(children.len());
            for c in children {
                obj.push((c.name.clone(), assemble(c, cursors)?));
            }
            Ok(JsonValue::Object(obj))
        }
    }
}

/// Render "one element's worth" of a REPEATED node. If it has exactly one
/// child (the intermediate group of 3-level/2-level encoding, or the element
/// of a LIST<STRUCT>), use that child directly as the element (unlike
/// `render_present`, this doesn't care whether it's REPEATED -- the sole
/// child of a repeated node is always passed through unwrapped). If it has
/// two or more children (e.g. a MAP's key/value), it becomes a named object.
fn render_element(node: &NestedNode, cursors: &mut [LeafCursor]) -> Result<JsonValue> {
    match &node.content {
        NestedContent::Leaf(idx) => Ok(take_leaf_value(*idx, cursors)),
        NestedContent::Group(children) => {
            if children.len() == 1 {
                return assemble(&children[0], cursors);
            }
            let mut obj = Vec::with_capacity(children.len());
            for c in children {
                obj.push((c.name.clone(), assemble(c, cursors)?));
            }
            Ok(JsonValue::Object(obj))
        }
    }
}

fn take_leaf_value(idx: usize, cursors: &mut [LeafCursor]) -> JsonValue {
    let ty = cursors[idx].ty;
    let v = cursors[idx].take_value();
    cursors[idx].advance_raw();
    leaf_value_to_json(ty, v)
}

// --- Entry point ----------------------------------------------------------

/// Assemble a nested column (a subtree containing REPEATED fields like
/// LIST/MAP) into a single `Ty::Json` vector.
///
/// `chunks` is in the same order as `desc.leaves`/`desc.phys_cols`: for each
/// leaf, `(column metadata, entire column chunk byte range, file offset where
/// it starts)`. No page selection happens here (the caller always passes the
/// whole column chunk; see the comment at the top of the module for why).
pub fn read_nested_column(
    desc: &ColumnDesc,
    chunks: &[(&ColumnMetaData, &[u8], u64)],
    num_rows: usize,
    cache: &dyn PageCache,
) -> Result<Vector> {
    let root = match &desc.nested {
        Some(n) => n.as_ref(),
        None => err!(Internal),
    };
    ensure!(chunks.len() == desc.leaves.len(), Internal);
    ensure!(chunks.len() == desc.phys_cols.len(), Internal);

    let mut runs: Vec<LeafRuns> = Vec::with_capacity(chunks.len());
    for (i, &(meta, buf, start)) in chunks.iter().enumerate() {
        runs.push(read_nested_leaf_chunk(&desc.leaves[i], meta, buf, start, cache)?);
    }

    let mut cursors: Vec<LeafCursor> = runs
        .iter()
        .zip(desc.leaves.iter())
        .map(|(r, info)| LeafCursor {
            ty: info.ty,
            rep: &r.rep,
            def: &r.def,
            values: &r.values,
            pos: 0,
            val_idx: 0,
        })
        .collect();

    let mut out = Vector::with_capacity(Ty::Json, num_rows);
    for _ in 0..num_rows {
        match assemble(root, &mut cursors)? {
            JsonValue::Null => out.push_null(),
            other => {
                let mut buf = Vec::new();
                write_json(&other, &mut buf);
                out.push_value(&Value::Bytes(buf));
            }
        }
    }

    // Verify that every leaf's cursor has been exactly exhausted (that
    // consumption matched the declared row count with nothing left over or
    // missing). Used to detect corrupted files.
    for c in &cursors {
        ensure!(c.pos == c.rep.len() && c.pos == c.def.len(), BadCompressedData);
    }
    Ok(out)
}
