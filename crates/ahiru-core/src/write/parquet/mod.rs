//! Parquet writing (`export-parquet` feature).
//!
//! Deliberately minimal: one uncompressed PLAIN-encoded data page (v1) per
//! column per row group, RLE definition levels, no dictionary, no
//! statistics, no column/offset index, no bloom filters. That is the
//! smallest thing that produces files DuckDB (and this crate's own reader)
//! accept, which is what the 1 MiB budget can afford for an opt-in writer
//! (DESIGN.md §15). Readers treat all of the omitted parts as optional, so
//! the output is a valid Parquet file — just not a well-tuned one.
//!
//! Structurally this is the mirror of `crate::parquet`: `thrift` (metadata
//! → bytes) sits under this module and only serves it, exactly as the
//! read-side `parquet::thrift` only serves `parquet::meta`. The read path
//! is untouched.
//!
//! ## Buffering
//!
//! Unlike the CSV/JSONL sinks, this one cannot stream: a Parquet row group
//! stores each column contiguously, so all of a row group's rows have to be
//! held before any of them can be written. `write_batch` therefore appends
//! into per-column buffers and `flush_row_group` emits them once
//! `ROW_GROUP_ROWS` have accumulated. The rest of the export driver is
//! already fully materializing (`write::export_all` collects the whole file
//! into a `Vec<u8>`), so this adds a bounded amount on top rather than a
//! new class of memory use.
//!
//! ## Type mapping
//!
//! Every column is written as OPTIONAL, because the sink sees batches one
//! at a time and cannot know up front whether a NULL is still coming.
//!
//! | SQL type | Parquet |
//! |---|---|
//! | BOOLEAN | BOOLEAN |
//! | TINYINT / SMALLINT / INTEGER | INT32 (+ `INT(8\|16)` where narrower) |
//! | BIGINT | INT64 |
//! | UTINYINT / USMALLINT / UINTEGER | INT32 + `INT(n, unsigned)` |
//! | UBIGINT | INT64 + `INT(64, unsigned)` |
//! | HUGEINT | FIXED_LEN_BYTE_ARRAY(16) + `DECIMAL(38, 0)` |
//! | FLOAT | FLOAT (narrowed from the f64 in-memory form) |
//! | DOUBLE | DOUBLE |
//! | DECIMAL(p, s) | INT64 for p <= 18, else FLBA(16), + `DECIMAL(p, s)` |
//! | VARCHAR | BYTE_ARRAY + `STRING` |
//! | BLOB | BYTE_ARRAY |
//! | JSON | BYTE_ARRAY + `JSON` |
//! | DATE | INT32 + `DATE` |
//! | TIME | INT64 + `TIME(micros)` |
//! | TIMESTAMP | INT64 + `TIMESTAMP(micros, utc = false)` |
//! | TIMESTAMPTZ | INT64 + `TIMESTAMP(micros, utc = true)` |
//! | UUID | FIXED_LEN_BYTE_ARRAY(16) + `UUID` |
//! | INTERVAL | BYTE_ARRAY + `STRING` (see below) |
//! | NULL | INT32 + `UNKNOWN`, all values NULL |
//!
//! Two of those are not round-trip exact, and both are deliberate:
//!
//! - **INTERVAL** is written as its text rendering (`fmt_interval`, the
//!   same form the CSV/JSONL sinks use) rather than the FLBA(12) legacy
//!   `INTERVAL` converted type. The legacy encoding cannot represent the
//!   month/day/micros triple's signs, and this crate's own reader rejects
//!   it outright (`parquet::schema::map_converted`), so writing it would
//!   produce files we could not read back. Reading the export back gives
//!   VARCHAR.
//! - **JSON** reads back as VARCHAR, because the reader maps the `JSON`
//!   logical type to VARCHAR (its in-memory form is JSON text either way).

#[cfg(test)]
mod tests;
mod thrift;

use crate::parquet::{Compression, ConvertedType, Encoding, PType, PageType, Repetition};
use crate::prelude::*;
use crate::vector::{fmt_interval, unpack_interval, Batch, Bitmap, Field, Ty, Vector};
use crate::write::{validate_batch, TableSink};
use thrift::{ttype, Writer};

/// Rows per row group. Matches DuckDB's default, which keeps row groups
/// large enough that per-group metadata is negligible but small enough
/// that a reader can prune at a useful granularity.
const ROW_GROUP_ROWS: usize = 122_880;

/// The definition level of a present value. Every column is a top-level
/// OPTIONAL leaf, so levels are only ever 0 (NULL) or 1 (present) and the
/// RLE stream is 1 bit wide.
const MAX_DEF_LEVEL: u8 = 1;

// --- column plan -------------------------------------------------------------

/// How a column's values turn into PLAIN-encoded bytes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Enc {
    /// BOOLEAN: bit-packed, so it accumulates in a `Bitmap` rather than
    /// in the byte buffer.
    Bool,
    I32,
    /// UINTEGER: held as i64, written as the low 32 bits.
    U32,
    I64,
    /// UBIGINT: held as i128, written as the low 64 bits.
    U64,
    /// HUGEINT / DECIMAL(p > 18): big-endian two's complement, 16 bytes.
    I128Be,
    /// FLOAT: narrowed from the f64 in-memory form.
    F32,
    F64,
    /// Length-prefixed bytes, as-is.
    Bytes,
    /// FIXED_LEN_BYTE_ARRAY(16), zero-padded / truncated to width.
    Uuid,
    /// INTERVAL, rendered as text (see the module doc).
    IntervalText,
}

/// `SchemaElement.logicalType`, restricted to what this writer emits.
#[derive(Clone, Copy)]
enum Lg {
    String,
    Decimal { precision: i32, scale: i32 },
    Date,
    TimeMicros,
    TimestampMicros { utc: bool },
    Integer { bits: i8, signed: bool },
    Unknown,
    Json,
    Uuid,
}

/// Everything needed to declare one column in the footer and to encode
/// its values.
struct Plan {
    ptype: PType,
    /// Only set for FIXED_LEN_BYTE_ARRAY.
    type_length: Option<i32>,
    /// Legacy `ConvertedType`, written alongside `logical` for readers
    /// that predate logical types. Absent where no equivalent exists.
    converted: Option<ConvertedType>,
    logical: Option<Lg>,
    enc: Enc,
}

fn plan_column(ty: Ty) -> Result<Plan> {
    let plan =
        |ptype, enc, converted, logical| Plan { ptype, type_length: None, converted, logical, enc };
    let int = |bits: i8, signed: bool| {
        let converted = match (bits, signed) {
            (8, true) => ConvertedType::Int8,
            (16, true) => ConvertedType::Int16,
            (64, true) => ConvertedType::Int64,
            (8, false) => ConvertedType::Uint8,
            (16, false) => ConvertedType::Uint16,
            (32, false) => ConvertedType::Uint32,
            (64, false) => ConvertedType::Uint64,
            // Only the widths listed above are ever passed; 32-bit signed
            // is INTEGER, which goes out unannotated.
            _ => ConvertedType::Int32,
        };
        (Some(converted), Some(Lg::Integer { bits, signed }))
    };
    let decimal = |precision: u8, scale: u8| {
        let logical = Lg::Decimal { precision: i32::from(precision), scale: i32::from(scale) };
        (Some(ConvertedType::Decimal), Some(logical))
    };

    Ok(match ty {
        Ty::Boolean => plan(PType::Boolean, Enc::Bool, None, None),
        // INTEGER / BIGINT are the physical types' own meaning, so they
        // need no annotation at all.
        Ty::Int => plan(PType::Int32, Enc::I32, None, None),
        Ty::BigInt => plan(PType::Int64, Enc::I64, None, None),
        Ty::TinyInt => {
            let (c, l) = int(8, true);
            plan(PType::Int32, Enc::I32, c, l)
        }
        Ty::SmallInt => {
            let (c, l) = int(16, true);
            plan(PType::Int32, Enc::I32, c, l)
        }
        Ty::UTinyInt => {
            let (c, l) = int(8, false);
            plan(PType::Int32, Enc::I32, c, l)
        }
        Ty::USmallInt => {
            let (c, l) = int(16, false);
            plan(PType::Int32, Enc::I32, c, l)
        }
        Ty::UInt => {
            let (c, l) = int(32, false);
            plan(PType::Int32, Enc::U32, c, l)
        }
        Ty::UBigInt => {
            let (c, l) = int(64, false);
            plan(PType::Int64, Enc::U64, c, l)
        }
        // No Parquet type is 128 bits wide, so HUGEINT goes out as the
        // widest decimal that fits in FLBA(16). i128 values with 39 digits
        // exceed DECIMAL(38, 0)'s declared precision; they still round-trip
        // through this crate (the reader only uses the byte width), but a
        // strict reader may reject them.
        Ty::HugeInt => {
            let (c, l) = decimal(38, 0);
            Plan {
                ptype: PType::FixedLenByteArray,
                type_length: Some(16),
                converted: c,
                logical: l,
                enc: Enc::I128Be,
            }
        }
        Ty::Float => plan(PType::Float, Enc::F32, None, None),
        Ty::Double => plan(PType::Double, Enc::F64, None, None),
        Ty::Decimal { precision, scale } => {
            let (c, l) = decimal(precision, scale);
            if precision <= 18 {
                plan(PType::Int64, Enc::I64, c, l)
            } else {
                Plan {
                    ptype: PType::FixedLenByteArray,
                    type_length: Some(16),
                    converted: c,
                    logical: l,
                    enc: Enc::I128Be,
                }
            }
        }
        Ty::Varchar => {
            plan(PType::ByteArray, Enc::Bytes, Some(ConvertedType::Utf8), Some(Lg::String))
        }
        Ty::Blob => plan(PType::ByteArray, Enc::Bytes, None, None),
        Ty::Json => plan(PType::ByteArray, Enc::Bytes, Some(ConvertedType::Json), Some(Lg::Json)),
        Ty::Date => plan(PType::Int32, Enc::I32, Some(ConvertedType::Date), Some(Lg::Date)),
        Ty::Time => {
            plan(PType::Int64, Enc::I64, Some(ConvertedType::TimeMicros), Some(Lg::TimeMicros))
        }
        Ty::Timestamp => plan(
            PType::Int64,
            Enc::I64,
            Some(ConvertedType::TimestampMicros),
            Some(Lg::TimestampMicros { utc: false }),
        ),
        // The legacy converted type has no "is adjusted to UTC" bit, so a
        // reader that only understands converted types sees a plain
        // TIMESTAMP. The logical type carries the distinction and wins
        // wherever both are understood.
        Ty::Timestamptz => plan(
            PType::Int64,
            Enc::I64,
            Some(ConvertedType::TimestampMicros),
            Some(Lg::TimestampMicros { utc: true }),
        ),
        Ty::Uuid => Plan {
            ptype: PType::FixedLenByteArray,
            type_length: Some(16),
            converted: None,
            logical: Some(Lg::Uuid),
            enc: Enc::Uuid,
        },
        // Text, not the legacy FLBA(12) INTERVAL — see the module doc.
        Ty::Interval => {
            plan(PType::ByteArray, Enc::IntervalText, Some(ConvertedType::Utf8), Some(Lg::String))
        }
        // An untyped NULL literal. Every value is NULL, so the physical
        // type only has to exist; `UNKNOWN` is what says so.
        Ty::Null => plan(PType::Int32, Enc::I32, None, Some(Lg::Unknown)),
    })
}

// --- per-column buffer -------------------------------------------------------

/// One column's pending row group.
struct ColBuf {
    /// PLAIN bytes of the non-NULL values (unused for BOOLEAN).
    plain: Vec<u8>,
    /// BOOLEAN values, bit-packed at flush time. Empty for other types.
    bools: Bitmap,
    /// One bit per row: set = present, clear = NULL.
    validity: Bitmap,
}

impl ColBuf {
    fn new() -> Self {
        ColBuf { plain: Vec::new(), bools: Bitmap::new(), validity: Bitmap::new() }
    }

    fn clear(&mut self) {
        self.plain.clear();
        self.bools = Bitmap::new();
        self.validity = Bitmap::new();
    }
}

/// One column chunk's position in the file, kept until the footer is written.
struct ChunkMeta {
    offset: i64,
    /// Data page header + page body.
    size: i64,
}

struct RowGroupMeta {
    columns: Vec<ChunkMeta>,
    num_rows: i64,
}

// --- sink --------------------------------------------------------------------

pub struct ParquetSink {
    out: Vec<u8>,
    began: bool,
    schema: Vec<Field>,
    names: Vec<String>,
    plans: Vec<Plan>,
    cols: Vec<ColBuf>,
    pending_rows: usize,
    row_groups: Vec<RowGroupMeta>,
    total_rows: i64,
}

impl ParquetSink {
    pub fn new() -> Self {
        ParquetSink {
            out: Vec::new(),
            began: false,
            schema: Vec::new(),
            names: Vec::new(),
            plans: Vec::new(),
            cols: Vec::new(),
            pending_rows: 0,
            row_groups: Vec::new(),
            total_rows: 0,
        }
    }

    /// Emit every buffered column as one row group and reset the buffers.
    fn flush_row_group(&mut self) -> Result<()> {
        if self.pending_rows == 0 {
            return Ok(());
        }
        let rows = self.pending_rows;
        // `PageHeader.num_values` and the page sizes are i32 on the wire.
        // Nothing realistic reaches either bound — a row group holds at
        // most `ROW_GROUP_ROWS` rows, so only a column of multi-kilobyte
        // blobs could approach the size one — but truncating silently
        // would corrupt the file, so refuse up front instead. The level
        // stream costs at most a byte per 8 rows plus its run headers, so
        // `rows` bounds it with room to spare.
        ensure!(rows <= i32::MAX as usize, ValueOutOfRange);
        for buf in &self.cols {
            let page = buf.plain.len().saturating_add(rows).saturating_add(16);
            ensure!(page <= i32::MAX as usize, ValueOutOfRange);
        }
        // Taken out so the per-column loop can append to `self.out` while
        // still holding a mutable borrow of the buffers.
        let mut cols = core::mem::take(&mut self.cols);
        let mut columns = Vec::with_capacity(cols.len());
        for (buf, plan) in cols.iter_mut().zip(&self.plans) {
            let mut page = encode_def_levels(&buf.validity, rows);
            if plan.enc == Enc::Bool {
                push_bitmap_lsb(&mut page, &buf.bools);
            } else {
                page.extend_from_slice(&buf.plain);
            }
            let header = data_page_header(rows, page.len());
            let offset = self.out.len() as i64;
            self.out.extend_from_slice(&header);
            self.out.extend_from_slice(&page);
            columns.push(ChunkMeta { offset, size: (header.len() + page.len()) as i64 });
            buf.clear();
        }
        self.cols = cols;
        self.row_groups.push(RowGroupMeta { columns, num_rows: rows as i64 });
        self.total_rows += rows as i64;
        self.pending_rows = 0;
        Ok(())
    }

    /// Serialize the `FileMetaData` footer.
    fn footer(&self) -> Result<Vec<u8>> {
        let mut w = Writer::new();
        w.begin_struct();
        w.field_i32(1, 1); // version
        w.begin_field_list(2, ttype::STRUCT, self.names.len() + 1);
        write_root_schema_element(&mut w, self.names.len());
        for (name, plan) in self.names.iter().zip(&self.plans) {
            write_schema_element(&mut w, name, plan);
        }
        w.field_i64(3, self.total_rows);
        w.begin_field_list(4, ttype::STRUCT, self.row_groups.len());
        for rg in &self.row_groups {
            self.write_row_group(&mut w, rg);
        }
        w.field_binary(6, b"ahirudb");
        w.end_struct();
        let footer = w.into_bytes();
        validate_footer_len(footer.len())?;
        Ok(footer)
    }

    fn write_row_group(&self, w: &mut Writer, rg: &RowGroupMeta) {
        let total: i64 = rg.columns.iter().map(|c| c.size).sum();
        w.begin_struct();
        w.begin_field_list(1, ttype::STRUCT, rg.columns.len());
        for (chunk, (name, plan)) in rg.columns.iter().zip(self.names.iter().zip(&self.plans)) {
            w.begin_struct();
            // `file_offset` is deprecated but still read by some tools;
            // point it at the chunk's first page like every other writer.
            w.field_i64(2, chunk.offset);
            w.begin_field_struct(3); // meta_data
            w.field_i32(1, plan.ptype as i32);
            w.begin_field_list(2, ttype::I32, 2);
            w.elem_i32(Encoding::Plain as i32);
            w.elem_i32(Encoding::Rle as i32);
            w.begin_field_list(3, ttype::BINARY, 1);
            w.elem_binary(name.as_bytes());
            w.field_i32(4, Compression::Uncompressed as i32);
            w.field_i64(5, rg.num_rows);
            // Uncompressed and compressed are equal because we never
            // compress; both include the page headers, which is what the
            // reader's `ColumnMetaData::byte_range` assumes.
            w.field_i64(6, chunk.size);
            w.field_i64(7, chunk.size);
            w.field_i64(9, chunk.offset); // data_page_offset
            w.end_struct();
            w.end_struct();
        }
        w.field_i64(2, total);
        w.field_i64(3, rg.num_rows);
        w.end_struct();
    }
}

impl Default for ParquetSink {
    fn default() -> Self {
        ParquetSink::new()
    }
}

impl TableSink for ParquetSink {
    fn begin(&mut self, schema: &[Field]) -> Result<()> {
        ensure!(!self.began, Internal);
        // A Parquet row group needs at least one column; there is no
        // meaningful file to write for a zero-column result.
        ensure!(!schema.is_empty(), UnsupportedFeature);
        let plans = schema.iter().map(|f| plan_column(f.ty)).collect::<Result<Vec<_>>>()?;
        self.out.clear();
        self.schema = schema.to_vec();
        self.names = schema.iter().map(|f| f.name.clone()).collect();
        self.plans = plans;
        self.cols = schema.iter().map(|_| ColBuf::new()).collect();
        self.pending_rows = 0;
        self.row_groups.clear();
        self.total_rows = 0;
        self.out.extend_from_slice(crate::parquet::MAGIC);
        self.began = true;
        Ok(())
    }

    fn write_batch(&mut self, schema: &[Field], batch: &Batch) -> Result<()> {
        ensure!(self.began, Internal);
        ensure!(self.schema.len() == schema.len(), Internal);
        ensure!(
            self.schema
                .iter()
                .zip(schema)
                .all(|(a, b)| { a.name == b.name && a.ty == b.ty && a.nullable == b.nullable }),
            Internal
        );
        validate_batch(schema, batch)?;
        ensure!(batch.cols.len() == self.plans.len(), Internal);
        ensure!(schema.len() == self.plans.len(), Internal);
        let rows = batch.num_rows();
        // Parquet DECIMAL precision is part of the file schema, not merely a
        // display hint. A 16-byte value can hold all of i128, but DECIMAL(38)
        // only permits magnitudes below 10^38. Reject an out-of-range HUGEINT
        // (or DECIMAL(p > 18)) before appending anything to the row-group
        // buffers; otherwise a strict Parquet reader can reject the file.
        for (col, plan) in batch.cols.iter().zip(&self.plans) {
            let Some(Lg::Decimal { precision, .. }) = plan.logical else { continue };
            for r in 0..rows {
                if col.is_valid(r) {
                    let fits = match plan.enc {
                        Enc::I64 => col
                            .i64s()
                            .get(r)
                            .copied()
                            .map(|value| decimal_value_fits(value as i128, precision))
                            .unwrap_or(false),
                        Enc::I128Be => col
                            .i128s()
                            .get(r)
                            .copied()
                            .map(|value| decimal_value_fits(value, precision))
                            .unwrap_or(false),
                        _ => true,
                    };
                    ensure!(fits, ValueOutOfRange);
                }
            }
        }
        for ((col, plan), buf) in batch.cols.iter().zip(&self.plans).zip(&mut self.cols) {
            for r in 0..rows {
                if !col.is_valid(r) {
                    buf.validity.push(false);
                    continue;
                }
                buf.validity.push(true);
                encode_value(plan.enc, col, r, buf);
            }
        }
        self.pending_rows += rows;
        if self.pending_rows >= ROW_GROUP_ROWS {
            self.flush_row_group()?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<Vec<u8>> {
        ensure!(self.began, Internal);
        self.flush_row_group()?;
        let footer = self.footer()?;
        // Keep the framing check at the point where the footer is appended
        // and its length is narrowed to the trailing u32.
        validate_footer_len(footer.len())?;
        self.out.extend_from_slice(&footer);
        self.out.extend_from_slice(&(footer.len() as u32).to_le_bytes());
        self.out.extend_from_slice(crate::parquet::MAGIC);
        self.began = false;
        Ok(core::mem::take(&mut self.out))
    }
}

/// Keep writer output compatible with the reader's bounded footer parser and
/// prevent the trailing u32 length field from truncating silently.
fn validate_footer_len(len: usize) -> Result<()> {
    ensure!(len <= crate::parquet::file::MAX_FOOTER_LEN, LimitExceeded);
    ensure!(len <= u32::MAX as usize, LimitExceeded);
    Ok(())
}

// --- value encoding ----------------------------------------------------------

/// Append row `r` of `col` to `buf` in PLAIN encoding.
///
/// The accessors return an empty slice on a physical-type mismatch (which
/// would be a bug in `plan_column`), so every read goes through `get` and
/// falls back to a zero value rather than indexing.
fn encode_value(enc: Enc, col: &Vector, r: usize, buf: &mut ColBuf) {
    match enc {
        Enc::Bool => buf.bools.push(col.bools().get(r)),
        Enc::I32 => {
            let v = col.i32s().get(r).copied().unwrap_or(0);
            buf.plain.extend_from_slice(&v.to_le_bytes());
        }
        Enc::U32 => {
            let v = col.i64s().get(r).copied().unwrap_or(0);
            buf.plain.extend_from_slice(&(v as u32).to_le_bytes());
        }
        Enc::I64 => {
            let v = col.i64s().get(r).copied().unwrap_or(0);
            buf.plain.extend_from_slice(&v.to_le_bytes());
        }
        Enc::U64 => {
            let v = col.i128s().get(r).copied().unwrap_or(0);
            buf.plain.extend_from_slice(&(v as u64).to_le_bytes());
        }
        Enc::I128Be => {
            let v = col.i128s().get(r).copied().unwrap_or(0);
            buf.plain.extend_from_slice(&v.to_be_bytes());
        }
        Enc::F32 => {
            let v = col.f64s().get(r).copied().unwrap_or(0.0);
            buf.plain.extend_from_slice(&(v as f32).to_le_bytes());
        }
        Enc::F64 => {
            let v = col.f64s().get(r).copied().unwrap_or(0.0);
            buf.plain.extend_from_slice(&v.to_le_bytes());
        }
        Enc::Bytes => {
            let v = bytes_at(col, r);
            buf.plain.extend_from_slice(&(v.len() as u32).to_le_bytes());
            buf.plain.extend_from_slice(v);
        }
        Enc::Uuid => {
            // FIXED_LEN_BYTE_ARRAY carries no length, so the width has to
            // be exactly 16 whatever the value happens to hold.
            let v = bytes_at(col, r);
            let mut raw = [0u8; 16];
            let n = v.len().min(16);
            raw[..n].copy_from_slice(&v[..n]);
            buf.plain.extend_from_slice(&raw);
        }
        Enc::IntervalText => {
            let v = col.i128s().get(r).copied().unwrap_or(0);
            let (months, days, micros) = unpack_interval(v);
            let mut text = Vec::new();
            fmt_interval(months, days, micros, &mut text);
            buf.plain.extend_from_slice(&(text.len() as u32).to_le_bytes());
            buf.plain.extend_from_slice(&text);
        }
    }
}

/// Whether an unscaled integer fits a Parquet DECIMAL with the given precision.
/// Precision `p` permits values in `-(10^p - 1)..=(10^p - 1)`.
fn decimal_value_fits(value: i128, precision: i32) -> bool {
    if !(1..=38).contains(&precision) {
        return false;
    }
    let mut limit = 1u128;
    for _ in 0..precision {
        limit *= 10;
    }
    value.unsigned_abs() < limit
}

/// Row `r` of a `Bytes` column, without the bounds assumption `BytesData::get`
/// makes.
fn bytes_at(col: &Vector, r: usize) -> &[u8] {
    let b = col.bytes();
    if r + 1 < b.offsets.len() {
        b.get(r)
    } else {
        &[]
    }
}

/// Bit-pack a `Bitmap` LSB-first, the layout PLAIN BOOLEAN uses.
fn push_bitmap_lsb(out: &mut Vec<u8>, bm: &Bitmap) {
    let want = bm.len().div_ceil(8);
    let start = out.len();
    for w in bm.as_words() {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out.truncate(start + want);
}

// --- definition levels -------------------------------------------------------

/// Encode `rows` definition levels as a v1 level stream: a 4-byte
/// little-endian length followed by an RLE/bit-packing hybrid run sequence
/// 1 bit wide.
///
/// Runs of 8 or more equal levels become RLE runs (a column with no NULLs
/// at all collapses to a handful of bytes); anything shorter is bit-packed
/// in groups of 8, which caps the worst case at 1 byte per 8 rows instead
/// of the ~2 bytes per row a pure-RLE encoder would emit for alternating
/// NULLs.
fn encode_def_levels(validity: &Bitmap, rows: usize) -> Vec<u8> {
    let mut body = Vec::new();
    let mut i = 0usize;
    while i < rows {
        let run = run_len(validity, i, rows);
        if run >= 8 {
            // Low bit clear = RLE run, followed by the repeated level in
            // ceil(bit_width / 8) = 1 byte.
            push_uleb(&mut body, (run as u64) << 1);
            body.push(if validity.get(i) { MAX_DEF_LEVEL } else { 0 });
            i += run;
            continue;
        }
        // Bit-packed stretch. `j` only ever advances by 8, so the segment
        // is a whole number of groups except when it runs into the end of
        // the page — which is the one place padding bits are harmless,
        // because the reader stops after `rows` values.
        let start = i;
        let mut j = i;
        while j < rows {
            if j > start && run_len(validity, j, rows) >= 8 {
                break;
            }
            j = (j + 8).min(rows);
        }
        let groups = (j - start).div_ceil(8);
        push_uleb(&mut body, ((groups as u64) << 1) | 1);
        for g in 0..groups {
            let mut byte = 0u8;
            for b in 0..8 {
                let idx = start + g * 8 + b;
                if idx < j && validity.get(idx) {
                    byte |= 1 << b;
                }
            }
            body.push(byte);
        }
        i = j;
    }
    let mut out = Vec::with_capacity(body.len() + 4);
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

/// Length of the run of equal bits starting at `i`.
fn run_len(bm: &Bitmap, i: usize, rows: usize) -> usize {
    let bit = bm.get(i);
    let mut j = i + 1;
    while j < rows && bm.get(j) == bit {
        j += 1;
    }
    j - i
}

fn push_uleb(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

// --- metadata ----------------------------------------------------------------

/// The page header for a single uncompressed v1 data page.
fn data_page_header(rows: usize, page_len: usize) -> Vec<u8> {
    let mut w = Writer::new();
    w.begin_struct();
    w.field_i32(1, PageType::DataPage as i32);
    // Uncompressed and compressed sizes are equal: this writer never
    // compresses. Both cover the page body only, not this header.
    w.field_i32(2, page_len as i32);
    w.field_i32(3, page_len as i32);
    w.begin_field_struct(5); // data_page_header
    w.field_i32(1, rows as i32);
    w.field_i32(2, Encoding::Plain as i32);
    w.field_i32(3, Encoding::Rle as i32); // definition levels
    w.field_i32(4, Encoding::Rle as i32); // repetition levels (none, flat)
    w.end_struct();
    w.end_struct();
    w.into_bytes()
}

/// The root of the schema tree: a group with one child per column and no
/// type of its own.
fn write_root_schema_element(w: &mut Writer, num_columns: usize) {
    w.begin_struct();
    w.field_binary(4, b"schema");
    w.field_i32(5, num_columns as i32);
    w.end_struct();
}

fn write_schema_element(w: &mut Writer, name: &str, plan: &Plan) {
    w.begin_struct();
    w.field_i32(1, plan.ptype as i32);
    if let Some(len) = plan.type_length {
        w.field_i32(2, len);
    }
    w.field_i32(3, Repetition::Optional as i32);
    w.field_binary(4, name.as_bytes());
    // No `num_children`: its presence is what marks an element as a group.
    if let Some(c) = plan.converted {
        w.field_i32(6, c as i32);
    }
    // `scale`/`precision` are the converted-type spelling of DECIMAL, and
    // are required whenever `ConvertedType::Decimal` is set.
    if let Some(Lg::Decimal { precision, scale }) = plan.logical {
        w.field_i32(7, scale);
        w.field_i32(8, precision);
    }
    if let Some(l) = plan.logical {
        w.begin_field_struct(10);
        write_logical_type(w, l);
        w.end_struct();
    }
    w.end_struct();
}

/// `LogicalType` is a Thrift union: the field id picks the member.
fn write_logical_type(w: &mut Writer, l: Lg) {
    match l {
        Lg::String => w.field_empty_struct(1),
        Lg::Decimal { precision, scale } => {
            w.begin_field_struct(5);
            w.field_i32(1, scale);
            w.field_i32(2, precision);
            w.end_struct();
        }
        Lg::Date => w.field_empty_struct(6),
        Lg::TimeMicros => {
            w.begin_field_struct(7);
            w.field_bool(1, false); // isAdjustedToUTC
            write_micros_unit(w);
            w.end_struct();
        }
        Lg::TimestampMicros { utc } => {
            w.begin_field_struct(8);
            w.field_bool(1, utc);
            write_micros_unit(w);
            w.end_struct();
        }
        Lg::Integer { bits, signed } => {
            w.begin_field_struct(10);
            w.field_i8(1, bits);
            w.field_bool(2, signed);
            w.end_struct();
        }
        Lg::Unknown => w.field_empty_struct(11),
        Lg::Json => w.field_empty_struct(12),
        Lg::Uuid => w.field_empty_struct(14),
    }
}

/// `TimeUnit` is itself a union; MICROS is member 2. Every temporal column
/// is written in microseconds because that is the engine's internal
/// resolution (DESIGN.md §8).
fn write_micros_unit(w: &mut Writer) {
    w.begin_field_struct(2); // unit
    w.field_empty_struct(2); // MICROS
    w.end_struct();
}
