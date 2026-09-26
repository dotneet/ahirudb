//! JSON (the equivalent of `read_json`/`read_json_auto`) -- the whole file is a single JSON value.
//!
//! Unlike `format::jsonl` (NDJSON, one object per line), this reads input where the whole file is
//! **one JSON document** (a top-level array, or a single object). The separators are `[`/`,`/`]`
//! rather than newlines, so record boundaries are not settled partway through a split.
//!
//! ## Top-level rules (confirmed by measuring `duckdb`'s `read_json_auto`)
//!
//! - A top-level array `[...]` -> each element is read as one row.
//! - A top-level single object `{...}` -> it alone is read as a one-row table (confirmed with
//!   `duckdb -c "SELECT * FROM read_json_auto('single_obj.json')"`: it becomes a one-row table).
//! - When an array element is not an object (a scalar, or an array itself), and when the top level
//!   is a bare scalar rather than an object or array, the raw value goes into a single column
//!   named `"json"` (the same idea as `duckdb`'s `read_json_auto('[1,2,3]')` becoming a one-column
//!   table `json BIGINT`). **This decision is made independently per row**, so even a mixture of
//!   object rows and scalar rows (input `duckdb` does not normally anticipate) does not crash.
//!   A non-object row has a value only in the `"json"` column, and the other columns are NULL.
//! - An empty array `[]` becomes, following `duckdb`, a one-column (type JSON), zero-row table
//!   named `"json"` (easier to handle than a degenerate table with no columns at all). A document
//!   whose objects carry **no keys at all** (`[{}, {}]`) takes the same shape, with each object
//!   itself as that column's value: a table with no columns cannot report a row count, so
//!   `count(*)` used to read 0 and `SELECT *` used to be a syntax error.
//! - An empty file (0 bytes) has no top-level value and gives `UnexpectedEof`. Unlike JSONL's
//!   "an empty file is an empty table", this premises reading one JSON document, so empty is a
//!   syntax violation to begin with.
//!
//! ## A non-streaming design (a v1 limitation)
//!
//! The other formats (Parquet/CSV/JSONL) follow the design of "waiting on I/O only at split
//! boundaries" (the `format` module docs, DESIGN.md §6), but a JSON array's element separators are
//! not structurally settled from the middle bytes alone (where one element ends depends on the
//! array depth and string escapes). So `resolve` takes the trade-off of **returning `NeedIo` until
//! the whole file's bytes are present, then parsing them all at once**. In the same spirit as
//! `write::export_all` documenting its "non-resumable design that fails rather than resuming a
//! `NEED_IO` during query execution", the split-boundary barrier's shape is at least preserved
//! here by simplifying to "a split = the whole file, one of them" (`num_splits` is always at most
//! 1). Reading itself walks the whole file once more in `read_split` (a separate implementation
//! from `resolve`'s scan; a simplification to avoid a self-referential struct).
//!
//! ## The memory safety valve
//!
//! Since the whole file is loaded into memory at once, a file exceeding `MAX_JSON_BYTES` is
//! rejected as `Oom` at `resolve` time (the same "fail explicitly rather than quietly" policy as
//! `exec::join`'s `MAX_BUILD_BYTES` and friends. It is of a piece with DESIGN.md's "memory cap:
//! 512 MB by default", but set more conservatively still, since the whole file is retained).
//!
//! ## Schema inference
//!
//! The column set is the union of the objects' keys (in order of first appearance), and the types
//! widen along the lattice NULL -> BOOLEAN -> BIGINT -> DOUBLE -> VARCHAR/JSON --
//! so far the same idea as `format::jsonl`. But `jsonl` was designed when `Ty::Json` did not
//! exist, so it fell back to `Ty::Varchar` (raw text) for nested values and incompatible type
//! mixtures. Here `Ty::Json` is available, so it falls that way instead, matching what was
//! measured with `duckdb` (below):
//!
//! - Nested values (arrays and objects) are always `Ty::Json`
//!   (confirmed with `duckdb -c "SELECT * FROM read_json_auto('nested_mixed.json')"` that a column
//!   mixing `[1,2,3]` and `5` becomes type `JSON`).
//! - A mixture within the string family (plain strings, and strings determined to be DATE or
//!   TIMESTAMP) settles on `Ty::Varchar`, since all of them can be expressed as decoded text
//!   (the same judgment as `format::jsonl`'s `widen`).
//! - Any other incompatible mixture (numbers with strings, booleans with strings, nesting with
//!   scalars, and so on) has raw JSON text as its only physical representation and falls to
//!   `Ty::Json` (confirmed with `duckdb -c "SELECT * FROM read_json_auto('widen.json')"` that a
//!   column mixing int/double/bool/string becomes type `JSON`).
//! - A column that was all NULL in the sample becomes the safe `Ty::Varchar`, as in
//!   `format::jsonl` (`duckdb` makes it `JSON`, but consistency with `jsonl` won out).
//!
//! The low-level tokenizer (skipping values, scanning strings, decoding escapes, reading numbers)
//! is `crate::json`'s, shared with `format::jsonl` and the JSON functions. The object/array
//! iterators, the inference lattice and the date parser follow `format::jsonl`'s but are kept
//! separate, since `jsonl`'s are private and tuned to its line-at-a-time reading.
//!
//! ## Newline-delimited `.json`
//!
//! `COPY ... TO 'x.json'` writes one object per line (as DuckDB does), and many other tools write
//! `.json` files the same way. Such a file is not one JSON document, so reading it back used to
//! fail with a syntax error. DuckDB auto-detects the layout, and so does this: when the first
//! top-level value is followed, after a line break, by another value, the file is read by
//! `format::jsonl` instead (see `JsonFormat::ndjson`) -- which also brings back streaming,
//! split-by-split reading and lifts the `MAX_JSON_BYTES` cap for it.
//!
//! ## The input is untrusted
//!
//! The same policy as `format::jsonl`. Broken JSON gives `Err`. A value beyond `SAMPLE_ELEMENTS`
//! whose type falls outside the inferred one is `InvalidCast`, not a quiet NULL -- dropping it
//! would be the "silently wrong answer" `docs/DESIGN.md` §15 rules out, and it is what
//! `format::csv`, `format::jsonl` and DuckDB all report for the same input. A *key* beyond the
//! sample is not dropped either: the whole document is resident, so every element contributes its
//! keys and the column set is complete no matter where a key first appears.
//! Scanning does not recurse, so nesting depth is not limited (see `crate::json::skip_value`).

use crate::catalog::Source;
use crate::format::jsonl::JsonlFormat;
use crate::format::{get_or_internal, ResolveStep, TableFormat};
use crate::json::{byte_at, decode_string, parse_f64, parse_i64, scan_string, skip_value, skip_ws};
use crate::prelude::*;
use crate::vector::{Bitmap, Data, Field, Ty, Vector};

/// The maximum leading rows used for schema inference (how column types widen).
/// The same idea as `format::jsonl::SAMPLE_LINES`. Elements beyond it are still syntax-checked
/// (the design has `resolve` read the whole file through) and counted toward the row count, and
/// they still contribute their **keys**, so the column set is always complete. What they do not do
/// is widen the type of a column the sample already settled on.
pub const SAMPLE_ELEMENTS: usize = 1000;

/// The cap on the number of columns inference creates. The same as `format::jsonl::MAX_COLUMNS`.
const MAX_COLUMNS: usize = 1024;

/// The safety valve for the design of loading the whole file into memory at once. Set to the same
/// order of magnitude as `exec::join`'s `MAX_BUILD_BYTES` (128 MiB).
const MAX_JSON_BYTES: u64 = 128 * 1024 * 1024;

pub struct JsonFormat {
    schema: Vec<Field>,
    resolved: bool,
    total_len: u64,
    row_count: u64,
    /// Set when the objects in the document contribute **no columns at all** (a document of `{}`
    /// elements, or an empty array).
    ///
    /// A zero-column table reports `count(*)` as 0 and makes `SELECT *` a syntax error, which is a
    /// silently wrong answer for a file that plainly holds rows. DuckDB gives such a document one
    /// column named `json` holding each record whole, and so does this: every row goes into that
    /// column as raw JSON text, objects included.
    raw_json: bool,
    /// Per column: the sample held only `null` for it, so its `Ty::Varchar` is a default rather
    /// than a reading of the data. See `TableFormat::column_has_no_evidence`.
    no_evidence: Vec<bool>,
    /// Per column: the object key it holds (see `format::unique_column_names` for why a column's
    /// name can differ from its key).
    keys: Vec<String>,
    /// Set in `resolve` when the file turns out to be newline-delimited (see the module docs);
    /// every method then forwards to it.
    ndjson: Option<JsonlFormat>,
}

impl JsonFormat {
    pub fn new() -> Self {
        JsonFormat {
            schema: Vec::new(),
            resolved: false,
            total_len: 0,
            row_count: 0,
            raw_json: false,
            no_evidence: Vec::new(),
            keys: Vec::new(),
            ndjson: None,
        }
    }

    /// Switches this file over to the JSONL reader and resolves it there.
    fn resolve_as_ndjson(&mut self, src: &Source) -> Result<ResolveStep> {
        let j = self.ndjson.insert(JsonlFormat::new());
        j.resolve(src)
    }
}

/// Whether `b` (a prefix of the file, or all of it) starts like newline-delimited JSON: its first
/// top-level value is complete and followed, after a line break, by more content.
///
/// A single document never has anything but whitespace after its top-level value, so this cannot
/// misfire on one. A first value that does not end within `b` gives `false`; `resolve` asks again
/// once the whole file is in hand.
fn looks_like_ndjson(b: &[u8]) -> bool {
    let i = skip_ws(b, skip_bom(b));
    let Ok(end) = skip_value(b, i) else { return false };
    let next = skip_ws(b, end);
    next < b.len() && b[end..next].contains(&b'\n')
}

impl Default for JsonFormat {
    fn default() -> Self {
        JsonFormat::new()
    }
}

impl TableFormat for JsonFormat {
    fn resolve(&mut self, src: &Source) -> Result<ResolveStep> {
        if let Some(j) = &mut self.ndjson {
            return j.resolve(src);
        }
        if self.resolved {
            return Ok(Ok(()));
        }
        // An empty file has no top-level value and so is invalid as JSON to begin with
        // (unlike JSONL, this premises reading "one JSON document").
        ensure!(src.total_len > 0, UnexpectedEof);
        // A leading sample first, to tell a newline-delimited file (read by `format::jsonl`,
        // split by split) from a single document (which needs the whole file) before committing
        // to fetching all of it.
        let n = src.total_len.min(crate::format::jsonl::SAMPLE_BYTES);
        let head = match src.get(0, n as usize) {
            Some(b) => b,
            None => return Ok(Err((0, n))),
        };
        if looks_like_ndjson(head) {
            return self.resolve_as_ndjson(src);
        }
        ensure!(src.total_len <= MAX_JSON_BYTES, Oom);
        self.total_len = src.total_len;
        let buf = match src.get(0, src.total_len as usize) {
            Some(b) => b,
            // It requests "the whole file" rather than a split boundary barrier (see the module docs).
            None => return Ok(Err((0, src.total_len))),
        };
        // A first record longer than the sample is only seen whole now.
        if n < src.total_len && looks_like_ndjson(buf) {
            return self.resolve_as_ndjson(src);
        }
        let (keys, schema, row_count, raw_json, no_evidence) = parse_schema(buf)?;
        self.keys = keys;
        self.schema = schema;
        self.row_count = row_count;
        self.raw_json = raw_json;
        self.no_evidence = no_evidence;
        self.resolved = true;
        Ok(Ok(()))
    }

    fn is_resolved(&self) -> bool {
        match &self.ndjson {
            Some(j) => j.is_resolved(),
            None => self.resolved,
        }
    }

    fn schema(&self) -> &[Field] {
        match &self.ndjson {
            Some(j) => j.schema(),
            None => &self.schema,
        }
    }

    fn num_splits(&self) -> usize {
        if let Some(j) = &self.ndjson {
            return j.num_splits();
        }
        // The non-streaming design: the split is always the whole file, exactly one (see the module docs).
        if self.resolved {
            1
        } else {
            0
        }
    }

    fn split_rows(&self, split: usize) -> Option<u64> {
        if let Some(j) = &self.ndjson {
            return j.split_rows(split);
        }
        if split == 0 {
            Some(self.row_count)
        } else {
            None
        }
    }

    fn split_ranges(
        &self,
        split: usize,
        projection: &[usize],
        out: &mut Vec<(u64, u64)>,
    ) -> Result<()> {
        if let Some(j) = &self.ndjson {
            return j.split_ranges(split, projection, out);
        }
        ensure!(split < self.num_splits(), Internal);
        // The structure is not settled, so the whole file is always needed even with a projection.
        out.push((0, self.total_len));
        Ok(())
    }

    fn read_split(&self, src: &Source, split: usize, projection: &[usize]) -> Result<Vec<Vector>> {
        if let Some(j) = &self.ndjson {
            return j.read_split(src, split, projection);
        }
        ensure!(self.resolved, Internal);
        ensure!(split < self.num_splits(), Internal);
        let buf = get_or_internal(src, 0, self.total_len)?;

        let mut names: Vec<&[u8]> = Vec::with_capacity(projection.len());
        let mut builders: Vec<Builder> = Vec::with_capacity(projection.len());
        for &c in projection {
            let (Some(f), Some(k)) = (self.schema.get(c), self.keys.get(c)) else { err!(Internal) };
            names.push(k.as_bytes());
            builders.push(Builder::new(f.ty, 64));
        }

        let mut slots: Vec<Option<Member>> = vec![None; projection.len()];
        let mut key = Vec::new();
        let mut val = Vec::new();

        let i0 = skip_ws(buf, skip_bom(buf));
        if byte_at(buf, i0)? == b'[' {
            let mut it = Elements::new(buf, i0)?;
            while let Some((s, e)) = it.next()? {
                process_row(
                    &buf[s..e],
                    &names,
                    self.raw_json,
                    &mut slots,
                    &mut builders,
                    &mut key,
                    &mut val,
                )?;
            }
            // Only whitespace may remain after the array's closing bracket.
            ensure!(skip_ws(buf, it.i) == buf.len(), SyntaxError, it.i);
        } else {
            let end = skip_value(buf, i0)?;
            ensure!(skip_ws(buf, end) == buf.len(), SyntaxError, end);
            process_row(
                &buf[i0..end],
                &names,
                self.raw_json,
                &mut slots,
                &mut builders,
                &mut key,
                &mut val,
            )?;
        }

        Ok(builders.into_iter().map(|b| b.finish()).collect())
    }

    /// JSON carries no schema, so every column type here is a guess from the leading sample.
    fn schema_is_inferred(&self) -> bool {
        true
    }

    fn column_has_no_evidence(&self, col: usize) -> bool {
        if let Some(j) = &self.ndjson {
            return j.column_has_no_evidence(col);
        }
        self.no_evidence.get(col).copied().unwrap_or(false)
    }
}

// --- Schema inference ----------------------------------------------------------

/// What `parse_schema` settles:
/// `(per-column object keys, schema, row count, raw-JSON mode, per-column "no evidence" flags)`.
type Resolved = (Vec<String>, Vec<Field>, u64, bool, Vec<bool>);

/// Resolves all of `buf` as one JSON document.
/// The row count is exact even beyond `SAMPLE_ELEMENTS` (every element is walked for the syntax
/// check anyway, so there is no extra cost). Only the widen computation is limited to the sample.
fn parse_schema(buf: &[u8]) -> Result<Resolved> {
    let i = skip_ws(buf, skip_bom(buf));
    let c = byte_at(buf, i)?;

    let mut names: Vec<String> = Vec::new();
    let mut infs: Vec<Inf> = Vec::new();
    let mut key = Vec::new();
    let mut row_count: u64 = 0;

    if c == b'[' {
        // The number of columns the sample settled on. `None` while still inside the
        // sample. Past it, those columns' types are frozen, but *every* element still
        // contributes its keys: the whole document is resident and already walked for
        // the syntax check, and a key that first appears past the sample would
        // otherwise vanish from the schema entirely -- an entire column of data
        // dropped without a word, which `docs/DESIGN.md` §15 rules out.
        let mut frozen: Option<usize> = None;
        let mut it = Elements::new(buf, i)?;
        while let Some((s, e)) = it.next()? {
            row_count += 1;
            if frozen.is_none() && row_count as usize > SAMPLE_ELEMENTS {
                frozen = Some(names.len());
            }
            accumulate_row(&buf[s..e], &mut names, &mut infs, &mut key, frozen.unwrap_or(0))?;
        }
        ensure!(skip_ws(buf, it.i) == buf.len(), SyntaxError, it.i);
    } else {
        // The top level is not an array: a single object, or a bare scalar.
        // Both are treated as "one row" (see the module docs).
        let end = skip_value(buf, i)?;
        ensure!(skip_ws(buf, end) == buf.len(), SyntaxError, end);
        row_count = 1;
        accumulate_row(&buf[i..end], &mut names, &mut infs, &mut key, 0)?;
    }

    // No column came out of the document at all: an empty array, or elements that are all `{}`.
    // Matching duckdb's one JSON-typed column named `"json"` is both easier to handle than a
    // degenerate table with no columns and the only way `count(*)` can report the real row count.
    let raw_json = names.is_empty();
    if raw_json {
        names.push(String::from("json"));
        infs.push(Inf::Json);
    }

    let no_evidence = infs.iter().map(|i| *i == Inf::Null).collect();
    let cols = crate::format::unique_column_names(&names);
    let schema = cols.into_iter().zip(infs).map(|(n, i)| Field::new(n, i.ty(), true)).collect();
    Ok((names, schema, row_count, raw_json, no_evidence))
}

/// Folds one row's worth (one array element, or a single top-level value) into the column set.
/// `row` is exactly that value's bytes (with no surrounding whitespace).
fn accumulate_row(
    row: &[u8],
    names: &mut Vec<String>,
    infs: &mut Vec<Inf>,
    key: &mut Vec<u8>,
    frozen: usize,
) -> Result<()> {
    let c = byte_at(row, 0)?;
    if c == b'{' {
        let mut it = Members::new(row)?;
        while let Some(m) = it.next()? {
            let name = member_key(&m, key)?;
            merge(names, infs, name, infer(&m), frozen)?;
        }
    } else if c == b'n' {
        // A `null` element is a row of NULLs, as in DuckDB (`[null, {"a":1}]` is `a`: NULL, 1),
        // so it contributes no column. It used to add a phantom `json` column holding only NULL.
        // `process_row` gives it NULL in every column.
    } else {
        // A non-object row puts the raw value into a single `"json"` column
        // (see the module docs).
        let m = Member { key: b"json", key_escaped: false, val: row, kind: kind_of(c) };
        merge(names, infs, b"json", infer(&m), frozen)?;
    }
    Ok(())
}

/// Widens column `name`'s inferred type with `inf`. Creates the column if it does not exist.
///
/// `frozen` is the number of leading columns whose type is already settled -- the column
/// count at the end of the inference sample, or 0 while still inside it. A column below
/// that index no longer widens, so an out-of-sample value that does not fit an
/// established column stays an `InvalidCast` (see the module docs) instead of quietly
/// reshaping the schema. Columns discovered *after* the sample have no sample evidence
/// at all, so they keep widening across every element in which they appear.
fn merge(
    names: &mut Vec<String>,
    infs: &mut Vec<Inf>,
    name: &[u8],
    inf: Inf,
    frozen: usize,
) -> Result<()> {
    let idx = match names.iter().position(|s| s.as_bytes() == name) {
        Some(i) => i,
        None => {
            ensure!(names.len() < MAX_COLUMNS, LimitExceeded);
            names.push(String::from_utf8_lossy(name).into_owned());
            infs.push(Inf::Null);
            names.len() - 1
        }
    };
    if idx >= frozen {
        infs[idx] = widen(infs[idx], inf);
    }
    Ok(())
}

/// The column type during inference. `Ty` is not used directly because DATE/TIMESTAMP/JSON need to
/// be distinguished within the lattice as "specially promoted from a string or from nesting".
#[derive(Clone, Copy, PartialEq, Eq)]
enum Inf {
    Null,
    Bool,
    Int,
    Double,
    Date,
    Timestamp,
    /// A plain string that could not be interpreted as a date or timestamp.
    Str,
    /// A nested value, or a mixture of incompatible types. It is carried in a `Ty::Json` column as
    /// raw JSON text.
    Json,
}

impl Inf {
    fn ty(self) -> Ty {
        match self {
            // Everything in the sample was NULL. As in `format::jsonl`, the safe VARCHAR is chosen
            // (a difference from duckdb noted in the module docs).
            Inf::Null => Ty::Varchar,
            Inf::Bool => Ty::Boolean,
            Inf::Int => Ty::BigInt,
            Inf::Double => Ty::Double,
            Inf::Date => Ty::Date,
            Inf::Timestamp => Ty::Timestamp,
            Inf::Str => Ty::Varchar,
            Inf::Json => Ty::Json,
        }
    }
}

/// The lattice NULL -> BIGINT -> DOUBLE; the string family (Str/Date/Timestamp) falls to
/// VARCHAR among themselves, and any other incompatible pair falls to JSON
/// (see the duckdb measurements in the module docs).
///
/// BOOLEAN is a sibling of the numbers, not a step below them: `true` has no numeric reading, so
/// widening a `true`/`1` column to BIGINT dropped the boolean rows the sample itself had just
/// seen. `duckdb -c "SELECT * FROM read_json_auto('[{\"a\":true},{\"a\":1}]')"` types that column
/// `JSON`, and so does this.
fn widen(a: Inf, b: Inf) -> Inf {
    use Inf::*;
    if a == b {
        return a;
    }
    match (a, b) {
        (Null, x) | (x, Null) => x,
        (Int, Double) | (Double, Int) => Double,
        (Date, Timestamp) | (Timestamp, Date) => Timestamp,
        // The remaining pairs within the string family (Str/Date/Timestamp) can all be expressed as
        // decoded text, so they settle on the VARCHAR side (Str).
        (Str, Date) | (Date, Str) => Str,
        (Str, Timestamp) | (Timestamp, Str) => Str,
        // A number or boolean mixed with a string, and any mixture involving nesting, can only be
        // expressed as raw JSON text, so they fall to JSON.
        _ => Json,
    }
}

fn infer(m: &Member<'_>) -> Inf {
    match m.kind {
        Kind::Null => Inf::Null,
        Kind::Bool => Inf::Bool,
        Kind::Num => {
            // A decimal point or exponent means DOUBLE unconditionally. An integer not fitting i64 is DOUBLE too.
            if m.val.iter().any(|&c| c == b'.' || c == b'e' || c == b'E') {
                Inf::Double
            } else if parse_i64(m.val).is_some() {
                Inf::Int
            } else {
                Inf::Double
            }
        }
        Kind::Str => {
            // A string containing escapes is essentially never a date, so it is not decoded.
            if m.value_escaped() {
                return Inf::Str;
            }
            let s = m.str_body();
            if parse_date(s).is_some() {
                Inf::Date
            } else if parse_timestamp(s).is_some() {
                Inf::Timestamp
            } else {
                Inf::Str
            }
        }
        // Nesting is always JSON (see the module docs).
        Kind::Nested => Inf::Json,
    }
}

// --- Column builders ------------------------------------------------------------

enum Cell<'a> {
    Bool(bool),
    I32(i32),
    I64(i64),
    F64(f64),
    Bytes(&'a [u8]),
}

/// Building one column. The only rule it upholds is advancing the data length and the validity length 1:1.
struct Builder {
    ty: Ty,
    data: Data,
    valid: Bitmap,
    any_null: bool,
}

impl Builder {
    fn new(ty: Ty, cap: usize) -> Self {
        Builder {
            ty,
            data: Data::with_capacity(ty.phys(), cap),
            valid: Bitmap::with_capacity(cap),
            any_null: false,
        }
    }

    /// `None` is NULL. A physical type mismatch also lands as NULL here, but the caller
    /// (`push_member`) turns it into `InvalidCast` -- see there for why.
    fn push(&mut self, cell: Option<Cell<'_>>) -> bool {
        let ok = match (&mut self.data, &cell) {
            (Data::Bool(d), Some(Cell::Bool(v))) => {
                d.push(*v);
                true
            }
            (Data::I32(d), Some(Cell::I32(v))) => {
                d.push(*v);
                true
            }
            (Data::I64(d), Some(Cell::I64(v))) => {
                d.push(*v);
                true
            }
            (Data::F64(d), Some(Cell::F64(v))) => {
                d.push(*v);
                true
            }
            (Data::Bytes(d), Some(Cell::Bytes(v))) => {
                d.push(v);
                true
            }
            _ => false,
        };
        if !ok {
            match &mut self.data {
                Data::Bool(d) => d.push(false),
                Data::I32(d) => d.push(0),
                Data::I64(d) => d.push(0),
                Data::F64(d) => d.push(0.0),
                Data::I128(d) => d.push(0),
                Data::Bytes(d) => d.push_empty(),
            }
            self.any_null = true;
        }
        self.valid.push(ok);
        ok
    }

    fn finish(self) -> Vector {
        let v = if self.any_null { Some(self.valid) } else { None };
        Vector::from_data(self.ty, self.data, v)
    }
}

/// Distributes one row's values to their positions among the projected columns (`names`).
///
/// `raw_json` puts the whole row into the `json` column even when it *is* an object, which is the
/// shape a file whose objects carry no keys at all takes (see `JsonFormat::raw_json`).
fn process_row<'a>(
    row: &'a [u8],
    names: &[&[u8]],
    raw_json: bool,
    slots: &mut [Option<Member<'a>>],
    builders: &mut [Builder],
    key: &mut Vec<u8>,
    val: &mut Vec<u8>,
) -> Result<()> {
    for s in slots.iter_mut() {
        *s = None;
    }
    let c = byte_at(row, 0)?;
    if c == b'{' && !raw_json {
        let mut it = Members::new(row)?;
        while let Some(m) = it.next()? {
            let name = member_key(&m, key)?;
            if let Some(j) = names.iter().position(|n| *n == name) {
                slots[j] = Some(m);
            }
        }
    } else if let Some(j) = names.iter().position(|n| *n == b"json") {
        slots[j] = Some(Member { key: b"json", key_escaped: false, val: row, kind: kind_of(c) });
    }
    for (j, b) in builders.iter_mut().enumerate() {
        push_member(b, slots[j].as_ref(), val)?;
    }
    Ok(())
}

/// Pushes one member according to the column's type. A missing key (`None`) and `null` are both NULL.
///
/// A value that is *present* but does not fit the column's type is `InvalidCast`. The schema comes
/// from a bounded leading sample (`SAMPLE_ELEMENTS`), so a later element can genuinely hold a
/// string where the sample only ever showed integers -- but silently turning that cell into NULL
/// loses data the file plainly contains, which is the "silently wrong answer" `docs/DESIGN.md` §15
/// says the engine never produces. `format::csv` and `format::jsonl` report the same situation the
/// same way, and so does DuckDB.
fn push_member(b: &mut Builder, m: Option<&Member<'_>>, scratch: &mut Vec<u8>) -> Result<()> {
    let m = match m {
        Some(m) if m.kind != Kind::Null => m,
        _ => {
            b.push(None);
            return Ok(());
        }
    };
    let ok = match b.ty {
        Ty::Boolean => b.push(match m.kind {
            Kind::Bool => Some(Cell::Bool(m.val.first() == Some(&b't'))),
            _ => None,
        }),
        Ty::BigInt => b.push(match m.kind {
            Kind::Num => parse_i64(m.val).map(Cell::I64),
            _ => None,
        }),
        Ty::Double => b.push(match m.kind {
            Kind::Num => parse_f64(m.val).map(Cell::F64),
            _ => None,
        }),
        Ty::Date => b.push(match m.kind {
            Kind::Str => parse_date(m.str_body()).map(Cell::I32),
            _ => None,
        }),
        Ty::Timestamp => b.push(match m.kind {
            Kind::Str => parse_timestamp(m.str_body()).map(Cell::I64),
            _ => None,
        }),
        // JSON always stays raw JSON text (even a string keeps its quotes, because `Ty::Json`'s
        // physical representation is "UTF-8 JSON text" rather than the decoded string -- see the
        // `vector::Ty::Json` docs).
        Ty::Json => b.push(Some(Cell::Bytes(m.val))),
        // VARCHAR accepts anything. Strings are decoded, and everything else is raw JSON text.
        _ => match m.kind {
            Kind::Str => {
                scratch.clear();
                decode_string(m.str_body(), scratch)?;
                b.push(Some(Cell::Bytes(scratch)))
            }
            _ => b.push(Some(Cell::Bytes(m.val))),
        },
    };
    ensure!(ok, InvalidCast);
    Ok(())
}

// --- JSON scanning --------------------------------------------------------------
//
// The iterators over one object's members and one array's elements. The tokenizer underneath
// them is `crate::json`'s (see the module docs).

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Null,
    Bool,
    Num,
    Str,
    /// An array or an object.
    Nested,
}

#[derive(Clone, Copy)]
struct Member<'a> {
    /// The key's body (inside the quotes, escapes unexpanded). When a non-object row is synthesized
    /// as the `"json"` column, `b"json"` (unescaped) is used.
    key: &'a [u8],
    key_escaped: bool,
    /// The value's span. For a string it includes the quotes.
    val: &'a [u8],
    kind: Kind,
}

impl<'a> Member<'a> {
    /// A string value's body (without the quotes). Meaningful only when `kind == Str`.
    fn str_body(&self) -> &'a [u8] {
        let n = self.val.len();
        if n >= 2 {
            &self.val[1..n - 1]
        } else {
            &[]
        }
    }

    /// Whether the string value contains escapes.
    fn value_escaped(&self) -> bool {
        self.str_body().contains(&b'\\')
    }
}

/// Returns a top-level object's members in order. It does not recurse.
/// `obj` is exactly one object (with no extra bytes on either side).
struct Members<'a> {
    b: &'a [u8],
    i: usize,
    /// Whether at least one has already been returned. Used to decide whether a comma is required.
    started: bool,
    done: bool,
}

impl<'a> Members<'a> {
    fn new(obj: &'a [u8]) -> Result<Self> {
        let i = skip_ws(obj, 0);
        ensure!(byte_at(obj, i)? == b'{', SyntaxError, i);
        Ok(Members { b: obj, i: i + 1, started: false, done: false })
    }

    fn next(&mut self) -> Result<Option<Member<'a>>> {
        if self.done {
            return Ok(None);
        }
        let b = self.b;
        let mut i = skip_ws(b, self.i);
        let mut c = byte_at(b, i)?;
        if c == b'}' {
            self.done = true;
            // Only whitespace may remain after the close. Input like `{}{}` is rejected.
            ensure!(skip_ws(b, i + 1) == b.len(), SyntaxError, i + 1);
            return Ok(None);
        }
        if self.started {
            ensure!(c == b',', SyntaxError, i);
            i = skip_ws(b, i + 1);
            c = byte_at(b, i)?;
        }
        ensure!(c == b'"', SyntaxError, i);
        let (key, key_escaped, ni) = scan_string(b, i)?;
        i = skip_ws(b, ni);
        ensure!(byte_at(b, i)? == b':', SyntaxError, i);
        i = skip_ws(b, i + 1);
        let kind = kind_of(byte_at(b, i)?);
        let v0 = i;
        i = skip_value(b, i)?;
        self.i = i;
        self.started = true;
        Ok(Some(Member { key, key_escaped, val: &b[v0..i], kind }))
    }
}

/// Returns a top-level array's elements in order. Each element's `(start, end)` byte positions
/// (the value's own span, excluding surrounding whitespace and commas).
struct Elements<'a> {
    b: &'a [u8],
    /// The next position to read. After the iteration ends it points just past the `]`.
    i: usize,
    started: bool,
    done: bool,
}

impl<'a> Elements<'a> {
    /// `at` is the position of the `[`.
    fn new(b: &'a [u8], at: usize) -> Result<Self> {
        ensure!(byte_at(b, at)? == b'[', SyntaxError, at);
        Ok(Elements { b, i: at + 1, started: false, done: false })
    }

    fn next(&mut self) -> Result<Option<(usize, usize)>> {
        if self.done {
            return Ok(None);
        }
        let b = self.b;
        let mut i = skip_ws(b, self.i);
        let mut c = byte_at(b, i)?;
        if c == b']' {
            self.done = true;
            self.i = i + 1;
            return Ok(None);
        }
        if self.started {
            ensure!(c == b',', SyntaxError, i);
            i = skip_ws(b, i + 1);
            c = byte_at(b, i)?;
        }
        let _ = c;
        let start = i;
        let end = skip_value(b, start)?;
        self.i = end;
        self.started = true;
        Ok(Some((start, end)))
    }
}

fn kind_of(c: u8) -> Kind {
    match c {
        b'{' | b'[' => Kind::Nested,
        b'"' => Kind::Str,
        b't' | b'f' => Kind::Bool,
        b'n' => Kind::Null,
        _ => Kind::Num,
    }
}

/// Gets a member's key as bytes. Only an escaped one is decoded into `scratch`.
fn member_key<'k>(m: &Member<'k>, scratch: &'k mut Vec<u8>) -> Result<&'k [u8]> {
    if !m.key_escaped {
        return Ok(m.key);
    }
    scratch.clear();
    decode_string(m.key, scratch)?;
    Ok(scratch)
}

// --- Numbers and date-times -----------------------------------------------------

/// Reads the leading `n` bytes as a decimal number. `None` if anything but digits is mixed in.
fn digits(s: &[u8], n: usize) -> Option<u32> {
    if s.len() < n {
        return None;
    }
    let mut v = 0u32;
    for &c in &s[..n] {
        if !c.is_ascii_digit() {
            return None;
        }
        v = v * 10 + (c - b'0') as u32;
    }
    Some(v)
}

fn is_leap(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i32, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap(y) => 29,
        2 => 28,
        _ => 0,
    }
}

/// days-from-civil (Howard Hinnant). Days since the epoch 1970-01-01.
fn days_from_civil(y: i32, m: u32, d: u32) -> i32 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u32; // [0, 399]
    let mp = if m > 2 { m - 3 } else { m + 9 }; // shift to a March-based year
    let doy = (153 * mp + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe as i32 - 719468
}

/// `YYYY-MM-DD` as days since the epoch.
fn parse_date(s: &[u8]) -> Option<i32> {
    if s.len() != 10 || s[4] != b'-' || s[7] != b'-' {
        return None;
    }
    let y = digits(s, 4)? as i32;
    let m = digits(&s[5..], 2)?;
    let d = digits(&s[8..], 2)?;
    if !(1..=12).contains(&m) || d < 1 || d > days_in_month(y, m) {
        return None;
    }
    Some(days_from_civil(y, m, d))
}

/// `YYYY-MM-DD[ T]HH:MM:SS[.ffffff]` as microseconds since the epoch.
/// A date-only string is accepted as midnight (for the DATE -> TIMESTAMP promotion).
fn parse_timestamp(s: &[u8]) -> Option<i64> {
    let days = parse_date(s.get(..10)?)? as i64;
    if s.len() == 10 {
        return Some(days * 86_400_000_000);
    }
    if s.len() < 19 || (s[10] != b' ' && s[10] != b'T') || s[13] != b':' || s[16] != b':' {
        return None;
    }
    let hh = digits(&s[11..], 2)?;
    let mi = digits(&s[14..], 2)?;
    let ss = digits(&s[17..], 2)?;
    if hh > 23 || mi > 59 || ss > 59 {
        return None;
    }
    let mut micros = 0u32;
    let mut rest = if s.len() > 19 { &s[19..] } else { &[][..] };
    if rest.first() == Some(&b'.') {
        let frac = &rest[1..];
        if frac.is_empty() {
            return None;
        }
        let mut k = 0usize;
        while k < frac.len() && frac[k].is_ascii_digit() {
            if k < 6 {
                micros = micros * 10 + (frac[k] - b'0') as u32;
            }
            k += 1;
        }
        if k == 0 {
            return None;
        }
        while k < 6 {
            micros *= 10;
            k += 1;
        }
        let mut n = 1;
        while n < rest.len() && rest[n].is_ascii_digit() {
            n += 1;
        }
        rest = &rest[n..];
    }
    let off = if rest.is_empty() {
        0
    } else {
        let (o, n) = crate::format::scan_tz_suffix(rest)?;
        if n != rest.len() {
            return None;
        }
        o
    };
    let secs = days * 86_400 + (hh as i64) * 3600 + (mi as i64) * 60 + ss as i64;
    Some(secs * 1_000_000 + micros as i64 - off)
}

// --- Byte-sequence utilities ------------------------------------------------------

/// Skips a UTF-8 BOM if present.
fn skip_bom(b: &[u8]) -> usize {
    if b.starts_with(&[0xEF, 0xBB, 0xBF]) {
        3
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{code_of, Code};
    use crate::vector::Value;

    fn data_file(name: &str) -> Vec<u8> {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
        std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    fn resolve(bytes: &[u8]) -> (JsonFormat, Source) {
        let src = Source::from_bytes(bytes.to_vec());
        let mut f = JsonFormat::new();
        match f.resolve(&src).expect("resolve") {
            Ok(()) => {}
            Err(r) => panic!("a range request {r:?} came back from an in-memory Source"),
        }
        (f, src)
    }

    fn schema_of(text: &str) -> Vec<(String, Ty)> {
        let (f, _) = resolve(text.as_bytes());
        f.schema().iter().map(|c| (c.name.clone(), c.ty)).collect()
    }

    fn read_all(text: &str) -> Vec<Vec<Value>> {
        let (f, src) = resolve(text.as_bytes());
        let projection: Vec<usize> = (0..f.schema().len()).collect();
        let mut ranges = Vec::new();
        f.split_ranges(0, &projection, &mut ranges).expect("split_ranges");
        let cols = f.read_split(&src, 0, &projection).expect("read_split");
        let mut out: Vec<Vec<Value>> = projection.iter().map(|_| Vec::new()).collect();
        for (j, c) in cols.iter().enumerate() {
            for i in 0..c.len() {
                out[j].push(c.value_at(i));
            }
        }
        out
    }

    fn s(v: &str) -> Value {
        Value::Bytes(v.as_bytes().to_vec())
    }

    fn infer_ty(values: &[&str]) -> Ty {
        let mut text = String::from("[");
        for (i, v) in values.iter().enumerate() {
            if i > 0 {
                text.push(',');
            }
            text.push_str("{\"a\":");
            text.push_str(v);
            text.push('}');
        }
        text.push(']');
        schema_of(&text)[0].1
    }

    // --- Real files -------------------------------------------------------

    #[test]
    fn basic_json_array_schema_matches_duckdb() {
        // duckdb: DESCRIBE SELECT * FROM read_json_auto('tests/data/basic_array.json')
        let (f, _) = resolve(&data_file("basic_array.json"));
        let got: Vec<(&str, Ty)> = f.schema().iter().map(|c| (c.name.as_str(), c.ty)).collect();
        assert_eq!(
            got,
            [
                ("id", Ty::BigInt),
                ("name", Ty::Varchar),
                ("score", Ty::Double),
                ("flag", Ty::Boolean),
                ("big", Ty::BigInt),
                ("d", Ty::Timestamp),
            ]
        );
        assert!(f.schema().iter().all(|c| c.nullable));
    }

    #[test]
    fn basic_json_array_values_match_the_jsonl_reference() {
        let cols = read_all_file("basic_array.json");
        assert!(cols.iter().all(|c| c.len() == 1000));
        assert_eq!(cols[0][0], Value::I64(0));
        assert_eq!(cols[1][0], s("name_0"));
        assert_eq!(cols[2][0], Value::F64(0.0));
        assert_eq!(cols[3][0], Value::Bool(true));
        assert_eq!(cols[4][0], Value::Null);
        assert_eq!(cols[2][1], Value::F64(1.5));
        assert_eq!(cols[4][1], Value::I64(100));
        assert_eq!(cols[5][0], Value::I64(1_704_067_200_000_000));
        assert_eq!(cols[0][999], Value::I64(999));
        assert_eq!(cols[1][999], s("name_5"));
        assert_eq!(cols[4][999], Value::I64(99_900));
    }

    fn read_all_file(name: &str) -> Vec<Vec<Value>> {
        let bytes = data_file(name);
        let src = Source::from_bytes(bytes);
        let mut f = JsonFormat::new();
        f.resolve(&src).expect("resolve").expect("bytes");
        let projection: Vec<usize> = (0..f.schema().len()).collect();
        let mut ranges = Vec::new();
        f.split_ranges(0, &projection, &mut ranges).expect("split_ranges");
        let cols = f.read_split(&src, 0, &projection).expect("read_split");
        let mut out: Vec<Vec<Value>> = projection.iter().map(|_| Vec::new()).collect();
        for (j, c) in cols.iter().enumerate() {
            for i in 0..c.len() {
                out[j].push(c.value_at(i));
            }
        }
        out
    }

    // --- Top-level rules --------------------------------------------------

    #[test]
    fn top_level_object_is_a_single_row_table() {
        // duckdb: SELECT * FROM read_json_auto('single_obj.json') -> 1 row.
        let cols = read_all("{\"a\":1,\"b\":\"hello\"}");
        assert_eq!(cols[0], [Value::I64(1)]);
        assert_eq!(cols[1], [s("hello")]);
    }

    #[test]
    fn array_of_scalars_becomes_a_single_json_named_column() {
        // duckdb: SELECT * FROM read_json_auto('[1,2,3]') -> the column name "json".
        let (f, _) = resolve(b"[1,2,3]");
        assert_eq!(f.schema().len(), 1);
        assert_eq!(f.schema()[0].name, "json");
        assert_eq!(f.schema()[0].ty, Ty::BigInt);
        let cols = read_all("[1,2,3]");
        assert_eq!(cols[0], [Value::I64(1), Value::I64(2), Value::I64(3)]);
    }

    #[test]
    fn empty_array_is_a_single_json_column_with_no_rows() {
        // duckdb: SELECT * FROM read_json_auto('[]') -> json JSON, 0 rows.
        let (f, _) = resolve(b"[]");
        assert_eq!(f.schema().len(), 1);
        assert_eq!(f.schema()[0].name, "json");
        assert_eq!(f.schema()[0].ty, Ty::Json);
        let cols = read_all("[]");
        assert_eq!(cols[0].len(), 0);
    }

    #[test]
    fn a_document_of_empty_objects_is_one_json_column_rather_than_no_columns() {
        // `[{},{}]` used to infer zero columns, so `count(*)` read 0 (a `Batch` with no columns
        // has no row count to report) and `SELECT *` was a syntax error. duckdb gives the document
        // one column named `json` holding each record whole.
        let (f, _) = resolve(b"[{},{}]");
        assert_eq!(f.schema().len(), 1);
        assert_eq!(f.schema()[0].name, "json");
        assert_eq!(f.schema()[0].ty, Ty::Json);
        assert_eq!(f.split_rows(0), Some(2));
        let cols = read_all("[{},{}]");
        assert_eq!(cols[0], [s("{}"), s("{}")]);

        // A single top-level `{}` is a one-row table for the same reason.
        let cols = read_all("{}");
        assert_eq!(cols[0], [s("{}")]);
    }

    #[test]
    fn a_bool_and_number_mixture_becomes_a_json_column_instead_of_nulling_rows() {
        // `[{"a":true},{"a":1}]` inferred BIGINT and then turned the `true` row into NULL, so
        // `count(a)` was 1 for a file with two values. duckdb types the column JSON.
        let (f, _) = resolve(b"[{\"a\":true},{\"a\":1}]");
        assert_eq!(f.schema()[0].ty, Ty::Json);
        let cols = read_all("[{\"a\":true},{\"a\":1}]");
        assert_eq!(cols[0], [s("true"), s("1")]);
    }

    #[test]
    fn mixed_object_and_scalar_rows_do_not_crash() {
        // duckdb does not normally anticipate this input, but this confirms it is handled without breaking.
        // A non-object row goes only into the "json" column, and the other columns are NULL.
        let cols = read_all("[{\"a\":1},5,{\"a\":2}]");
        let names: Vec<_> = {
            let (f, _) = resolve(b"[{\"a\":1},5,{\"a\":2}]");
            f.schema().iter().map(|c| c.name.clone()).collect()
        };
        let a = names.iter().position(|n| n == "a").unwrap();
        let j = names.iter().position(|n| n == "json").unwrap();
        assert_eq!(cols[a], [Value::I64(1), Value::Null, Value::I64(2)]);
        // The "json" column is contributed to only by row 1's `5`, so it is inferred as BIGINT
        // (the other rows are objects with no "json" key, so they are NULL).
        assert_eq!(cols[j], [Value::Null, Value::I64(5), Value::Null]);
    }

    // --- The union of column sets, and NULLs ------------------------------

    #[test]
    fn schema_is_the_union_of_keys_in_first_seen_order() {
        let got = schema_of("[{\"b\":1,\"a\":\"x\"},{\"c\":true},{\"a\":\"y\",\"b\":2}]");
        let names: Vec<&str> = got.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["b", "a", "c"]);
    }

    #[test]
    fn missing_key_and_explicit_null_are_both_null() {
        let cols = read_all("[{\"a\":1,\"b\":2},{\"a\":3},{\"a\":null,\"b\":4}]");
        assert_eq!(cols[0], [Value::I64(1), Value::I64(3), Value::Null]);
        assert_eq!(cols[1], [Value::I64(2), Value::Null, Value::I64(4)]);
    }

    // --- Type inference -----------------------------------------------------

    #[test]
    fn inference_lattice_covers_every_transition() {
        assert_eq!(infer_ty(&["null"]), Ty::Varchar);
        assert_eq!(infer_ty(&["true"]), Ty::Boolean);
        assert_eq!(infer_ty(&["null", "true"]), Ty::Boolean);
        assert_eq!(infer_ty(&["1"]), Ty::BigInt);
        // BOOLEAN mixed with a number falls to JSON too: `true` has no numeric reading, so BIGINT
        // dropped the boolean rows the sample itself had seen. duckdb types the column JSON.
        assert_eq!(infer_ty(&["true", "1"]), Ty::Json);
        assert_eq!(infer_ty(&["1", "1.5"]), Ty::Double);
        assert_eq!(infer_ty(&["true", "1.5"]), Ty::Json);
        // A number or boolean mixed with a string falls to JSON, matching duckdb
        // (`format::jsonl` falls to VARCHAR, a difference stemming from the constraints of the time
        // when `Ty::Json` did not exist -- see the module docs).
        assert_eq!(infer_ty(&["1", "\"x\""]), Ty::Json);
        assert_eq!(infer_ty(&["1.5", "\"x\""]), Ty::Json);
        assert_eq!(infer_ty(&["true", "\"x\""]), Ty::Json);
    }

    #[test]
    fn date_and_timestamp_strings_get_temporal_types() {
        assert_eq!(infer_ty(&["\"2024-01-01\""]), Ty::Date);
        assert_eq!(infer_ty(&["\"2024-01-01 12:34:56\""]), Ty::Timestamp);
        assert_eq!(infer_ty(&["\"2024-01-01T12:34:56\""]), Ty::Timestamp);
        assert_eq!(infer_ty(&["\"2024-01-01T12:34:56.123456\""]), Ty::Timestamp);
        assert_eq!(infer_ty(&["\"2024-01-01\"", "\"2024-01-01 00:00:00\""]), Ty::Timestamp);
        // A mixture within the string family is VARCHAR (the same judgment as `format::jsonl`).
        assert_eq!(infer_ty(&["\"2024-01-01\"", "\"hello\""]), Ty::Varchar);
        assert_eq!(infer_ty(&["\"2024-01-01 00:00:00\"", "\"hello\""]), Ty::Varchar);
        assert_eq!(infer_ty(&["\"2024-02-30\""]), Ty::Varchar);
        assert_eq!(infer_ty(&["\"2024-13-01\""]), Ty::Varchar);
        // Mixed with numbers it is JSON (unlike a mixture within the string family).
        assert_eq!(infer_ty(&["\"2024-01-01\"", "1"]), Ty::Json);
    }

    #[test]
    fn temporal_values_use_the_internal_representation() {
        let cols = read_all("[{\"a\":\"1970-01-02\"},{\"a\":\"2024-01-01\"}]");
        assert_eq!(cols[0], [Value::I32(1), Value::I32(19723)]);

        let cols = read_all(
            "[{\"a\":\"1970-01-01 00:00:01\"},\
              {\"a\":\"1969-12-31 23:59:59\"},\
              {\"a\":\"2024-01-01T00:00:00.000123\"}]",
        );
        assert_eq!(
            cols[0],
            [Value::I64(1_000_000), Value::I64(-1_000_000), Value::I64(1_704_067_200_000_123),]
        );
    }

    #[test]
    fn numbers_cover_integer_negative_fraction_exponent_and_overflow() {
        assert_eq!(infer_ty(&["-3"]), Ty::BigInt);
        assert_eq!(infer_ty(&["1e3"]), Ty::Double);
        assert_eq!(infer_ty(&["-1.5E-2"]), Ty::Double);
        assert_eq!(infer_ty(&["9223372036854775808"]), Ty::Double);
        assert_eq!(infer_ty(&["-9223372036854775809"]), Ty::Double);
        assert_eq!(infer_ty(&["-9223372036854775808"]), Ty::BigInt);

        let cols = read_all("[{\"a\":-3},{\"a\":9223372036854775807}]");
        assert_eq!(cols[0], [Value::I64(-3), Value::I64(i64::MAX)]);

        let cols = read_all("[{\"a\":1e3},{\"a\":-1.5E-2},{\"a\":7}]");
        assert_eq!(cols[0], [Value::F64(1000.0), Value::F64(-0.015), Value::F64(7.0)]);

        // Inference sees only the sample (SAMPLE_ELEMENTS of them). An element outside it whose
        // value does not fit the inferred type used to become NULL, silently dropping data the
        // file plainly contains -- exactly the "silently wrong answer" DESIGN.md §15 rules out.
        // It is now `InvalidCast`, as `format::csv` / `format::jsonl` and duckdb all report.
        let mut text = String::from("[");
        for i in 0..SAMPLE_ELEMENTS {
            if i > 0 {
                text.push(',');
            }
            text.push_str("{\"a\":1}");
        }
        text.push_str(",{\"a\":\"x\"},{\"a\":2}]");
        let (f, src) = resolve(text.as_bytes());
        assert_eq!(f.schema()[0].ty, Ty::BigInt);
        assert_eq!(code_of(f.read_split(&src, 0, &[0])), Some(Code::InvalidCast));
    }

    // A key that first appears past the inference sample used to be dropped from the
    // schema entirely: `SELECT *` returned only the sampled columns and referring to
    // the missing one was a bind error, with nothing anywhere saying the file had it.
    // The whole document is resident and already walked for the syntax check, so there
    // was never a reason to stop collecting keys -- only the *types* of columns the
    // sample settled on stay frozen (that is what keeps the InvalidCast above).
    #[test]
    fn a_key_first_seen_past_the_sample_still_becomes_a_column() {
        let mut text = String::from("[");
        for _ in 0..SAMPLE_ELEMENTS {
            text.push_str("{\"a\":1},");
        }
        text.push_str("{\"a\":1,\"b\":2},{\"a\":1,\"b\":2.5}]");
        let (f, _) = resolve(text.as_bytes());
        let names: Vec<&str> = f.schema().iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["a", "b"]);
        // `b` has no sample evidence at all, so it keeps widening over every element it
        // appears in -- 2 and 2.5 together give DOUBLE, not a BIGINT that then fails.
        assert_eq!(f.schema()[1].ty, Ty::Double);

        let cols = read_all(&text);
        assert_eq!(cols[1].len(), SAMPLE_ELEMENTS + 2);
        assert_eq!(cols[1][0], Value::Null);
        assert_eq!(cols[1][SAMPLE_ELEMENTS], Value::F64(2.0));
        assert_eq!(cols[1][SAMPLE_ELEMENTS + 1], Value::F64(2.5));
    }

    // --- Strings ----------------------------------------------------------

    #[test]
    fn string_escapes_are_decoded() {
        let cols = read_all("[{\"a\":\"q\\\"b\\\\s\\/t\\bf\\f\\nn\\rr\\tt\"}]");
        assert_eq!(cols[0][0], s("q\"b\\s/t\u{8}f\u{c}\nn\rr\tt"));
    }

    #[test]
    fn unicode_escapes_and_surrogate_pairs() {
        let cols = read_all("[{\"a\":\"\\u00e9\\u65e5\\uD83D\\uDE00\"}]");
        assert_eq!(cols[0][0], s("\u{e9}\u{65e5}\u{1f600}"));
    }

    #[test]
    fn lone_surrogates_become_the_replacement_character() {
        let cols =
            read_all("[{\"a\":\"\\uD800\"},{\"a\":\"\\uDC00x\"},{\"a\":\"\\uD800\\u0041\"}]");
        assert_eq!(cols[0][0], s("\u{FFFD}"));
        assert_eq!(cols[0][1], s("\u{FFFD}x"));
        assert_eq!(cols[0][2], s("\u{FFFD}A"));
    }

    #[test]
    fn escaped_keys_match_their_decoded_name() {
        let (f, _) = resolve(b"[{\"a\\u0062\":1},{\"ab\":2}]");
        assert_eq!(f.schema().len(), 1);
        assert_eq!(f.schema()[0].name, "ab");
        let cols = read_all("[{\"a\\u0062\":1},{\"ab\":2}]");
        assert_eq!(cols[0], [Value::I64(1), Value::I64(2)]);
    }

    // --- Nesting ------------------------------------------------------------

    #[test]
    fn nested_values_become_json_typed_columns() {
        // duckdb infers struct/list types, but this engine has no LIST/STRUCT and falls to Ty::Json
        // (raw JSON text) (see the module docs and the `Ty::Json` docs).
        let text = "[{\"a\":[1,2,{\"x\":\"y\"}],\"b\":{\"k\":[true,null]}},\
                     {\"a\":[],\"b\":{}}]";
        let sch = schema_of(text);
        assert_eq!(sch[0].1, Ty::Json);
        assert_eq!(sch[1].1, Ty::Json);
        let cols = read_all(text);
        // Ty::Json does not decode and keeps the raw text, quotes included.
        assert_eq!(cols[0][0], s("[1,2,{\"x\":\"y\"}]"));
        assert_eq!(cols[1][0], s("{\"k\":[true,null]}"));
        assert_eq!(cols[0][1], s("[]"));
        assert_eq!(cols[1][1], s("{}"));
    }

    #[test]
    fn nested_mixed_with_scalar_is_json() {
        // duckdb: the behavior confirmed with read_json_auto('nested_mixed.json').
        assert_eq!(schema_of("[{\"a\":[1,2,3]},{\"a\":5}]")[0].1, Ty::Json);
        let cols = read_all("[{\"a\":[1,2,3]},{\"a\":5}]");
        assert_eq!(cols[0][0], s("[1,2,3]"));
        assert_eq!(cols[0][1], s("5"));
    }

    #[test]
    fn json_column_preserves_raw_text_including_quotes_for_strings() {
        // duckdb: confirmed with read_json_auto('str_nested.json') that "hello" enters a JSON-typed
        // column with its quotes intact.
        let cols = read_all("[{\"a\":\"hello\"},{\"a\":[1,2,3]}]");
        assert_eq!(schema_of("[{\"a\":\"hello\"},{\"a\":[1,2,3]}]")[0].1, Ty::Json);
        assert_eq!(cols[0][0], s("\"hello\""));
        assert_eq!(cols[0][1], s("[1,2,3]"));
    }

    #[test]
    fn varchar_column_keeps_decoded_text_for_string_family_mix() {
        let cols =
            read_all("[{\"a\":\"2024-01-01\"},{\"a\":\"2024-01-01 00:00:00\"},{\"a\":\"hello\"}]");
        assert_eq!(cols[0], [s("2024-01-01"), s("2024-01-01 00:00:00"), s("hello")]);
    }

    // --- Projection ---------------------------------------------------------

    #[test]
    fn projection_returns_requested_columns_in_order() {
        let bytes = data_file("basic_array.json");
        let src = Source::from_bytes(bytes);
        let mut f = JsonFormat::new();
        f.resolve(&src).unwrap().unwrap();
        let proj = [4usize, 1usize];
        let mut ranges = Vec::new();
        f.split_ranges(0, &proj, &mut ranges).unwrap();
        let cols = f.read_split(&src, 0, &proj).unwrap();
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0].value_at(1), Value::I64(100)); // big
        assert_eq!(cols[1].value_at(1), s("name_1")); // name
        assert_eq!(cols[0].value_at(0), Value::Null);
        // It does not break with an empty projection either.
        let cols = f.read_split(&src, 0, &[]).unwrap();
        assert!(cols.is_empty());
    }

    // --- Splits -------------------------------------------------------------

    #[test]
    fn split_count_is_always_one_when_resolved() {
        let bytes = data_file("basic_array.json");
        let src = Source::from_bytes(bytes.clone());
        let mut f = JsonFormat::new();
        f.resolve(&src).unwrap().unwrap();
        assert_eq!(f.num_splits(), 1);
        let mut out = Vec::new();
        f.split_ranges(0, &[], &mut out).unwrap();
        assert_eq!(out, [(0, bytes.len() as u64)]);
        assert_eq!(code_of(f.split_ranges(1, &[], &mut out)), Some(Code::Internal));
        assert_eq!(f.split_rows(0), Some(1000));
        assert_eq!(f.split_rows(1), None);
    }

    // --- resolve's I/O requests ---------------------------------------------

    #[test]
    fn resolve_requests_the_whole_file_when_bytes_are_missing() {
        // A leading sample first (to tell NDJSON from a single document), then -- for a single
        // document -- the whole file.
        let mut doc = b"[".to_vec();
        while doc.len() < 1_000_000 {
            doc.extend_from_slice(b"{\"a\":1},");
        }
        doc.extend_from_slice(b"{\"a\":1}]");
        let mut f = JsonFormat::new();
        let mut src = Source::remote(doc.len() as u64);
        let sample = crate::format::jsonl::SAMPLE_BYTES;
        assert_eq!(f.resolve(&src).unwrap(), Err((0, sample)));
        src.insert(0, doc[..sample as usize].to_vec());
        assert_eq!(f.resolve(&src).unwrap(), Err((0, doc.len() as u64)));
        assert!(!f.is_resolved());
        assert_eq!(f.num_splits(), 0);
        src.insert(0, doc.clone());
        assert_eq!(f.resolve(&src).unwrap(), Ok(()));
        assert_eq!(f.num_splits(), 1);
    }

    #[test]
    fn oversized_file_is_rejected_before_reading_it_whole() {
        let mut f = JsonFormat::new();
        let mut src = Source::remote(MAX_JSON_BYTES + 1);
        let sample = crate::format::jsonl::SAMPLE_BYTES;
        assert_eq!(f.resolve(&src).unwrap(), Err((0, sample)));
        // The sample shows a single document (its first value does not even end within it), which
        // would have to be held whole.
        src.insert(0, vec![b'['; sample as usize]);
        assert_eq!(code_of(f.resolve(&src)), Some(Code::Oom));
    }

    #[test]
    fn oversized_ndjson_file_is_read_split_by_split() {
        // The whole-file cap does not apply once the file is known to be newline-delimited: it is
        // streamed like any `.jsonl`.
        let mut f = JsonFormat::new();
        let mut src = Source::remote(MAX_JSON_BYTES + 1);
        let sample = crate::format::jsonl::SAMPLE_BYTES as usize;
        let mut head = b"{\"a\":1}\n".repeat(sample / 8 + 1);
        head.truncate(sample);
        src.insert(0, head);
        assert_eq!(f.resolve(&src).unwrap(), Ok(()));
        assert!(f.num_splits() > 1);
    }

    #[test]
    fn empty_file_is_rejected() {
        let src = Source::from_bytes(Vec::new());
        let mut f = JsonFormat::new();
        assert_eq!(code_of(f.resolve(&src)), Some(Code::UnexpectedEof));
    }

    #[test]
    fn unresolved_format_reports_no_splits() {
        let f = JsonFormat::new();
        assert!(!f.is_resolved());
        assert_eq!(f.num_splits(), 0);
        assert!(f.schema().is_empty());
    }

    // --- Corrupt input ------------------------------------------------------

    fn resolve_err(text: &str) -> Option<Code> {
        let src = Source::from_bytes(text.as_bytes().to_vec());
        let mut f = JsonFormat::new();
        match f.resolve(&src) {
            Err(e) => Some(e.code),
            Ok(Err(_)) => None,
            Ok(Ok(())) => None,
        }
    }

    #[test]
    fn malformed_input_is_an_error_not_a_panic() {
        assert_eq!(resolve_err("[{\"a\":\"abc}]"), Some(Code::UnexpectedEof));
        assert_eq!(resolve_err("[{\"a\":1]"), Some(Code::SyntaxError));
        assert_eq!(resolve_err("[{\"a\":1}"), Some(Code::UnexpectedEof));
        assert_eq!(resolve_err("not json"), Some(Code::SyntaxError));
        assert_eq!(resolve_err("[1,2,3"), Some(Code::UnexpectedEof));
        assert_eq!(resolve_err("[1,2,3] garbage"), Some(Code::SyntaxError));
        assert_eq!(resolve_err("{\"a\":1,}"), Some(Code::SyntaxError));
        assert_eq!(resolve_err("[{\"a\":1},]"), Some(Code::SyntaxError));
        assert_eq!(resolve_err("{\"a\":\"x\ny\"}"), Some(Code::SyntaxError));
        assert_eq!(resolve_err(r#"{"a":"\q"}"#), Some(Code::SyntaxError));
        assert_eq!(resolve_err(r#"{"a":"\u12xz"}"#), Some(Code::SyntaxError));
        assert_eq!(resolve_err("[01]"), Some(Code::SyntaxError));
        assert_eq!(resolve_err("[-01]"), Some(Code::SyntaxError));
    }

    #[test]
    fn deeply_nested_values_are_read() {
        // A value 40 levels deep used to fail the whole file with NestingTooDeep, even in a
        // column the query never selected. DuckDB reads it.
        let deep = "[".repeat(40) + &"]".repeat(40);
        let text = format!("[{{\"a\":1,\"n\":{deep}}}]");
        let got = read_all(&text);
        assert_eq!(got[0], vec![Value::I64(1)]);
        assert_eq!(got[1], vec![s(&deep)]);
        // Elements themselves nested far deeper, too.
        let deeper = "[".repeat(5000) + "1" + &"]".repeat(5000);
        assert_eq!(read_all(&format!("[{deeper}]"))[0], vec![s(&deeper)]);
    }

    #[test]
    fn a_null_element_is_a_row_of_nulls() {
        // DuckDB: `[null, {"a":1}]` is column `a` holding NULL, 1. The `null` used to add a phantom
        // `json` column.
        assert_eq!(schema_of("[null, {\"a\":1}]"), vec![("a".to_owned(), Ty::BigInt)]);
        assert_eq!(
            read_all("[null, {\"a\":1}, null]")[0],
            vec![Value::Null, Value::I64(1), Value::Null]
        );
        // With nothing but `null` there is no key to make a column of: one NULL `json` row each.
        assert_eq!(schema_of("[null, null]"), vec![("json".to_owned(), Ty::Json)]);
        assert_eq!(read_all("[null, null]")[0], vec![Value::Null, Value::Null]);
        // A top-level `null` document is one such row.
        assert_eq!(read_all("null")[0], vec![Value::Null]);
    }

    #[test]
    fn keys_differing_only_in_case_get_distinct_columns() {
        // DuckDB renames the later key `A_1`; both used to be named as spelled, which made each
        // of them ambiguous to a (case-insensitive) column reference.
        assert_eq!(
            schema_of("[{\"a\":1,\"A\":\"x\"},{\"A\":\"y\"}]"),
            vec![("a".to_owned(), Ty::BigInt), ("A_1".to_owned(), Ty::Varchar)]
        );
        let got = read_all("[{\"a\":1,\"A\":\"x\"},{\"A\":\"y\"}]");
        assert_eq!(got[0], vec![Value::I64(1), Value::Null]);
        assert_eq!(got[1], vec![s("x"), s("y")]);
    }

    /// Resolves and reads every split, for a file that may be read as NDJSON.
    fn read_every_split(text: &str) -> (Vec<String>, Vec<Vec<Value>>) {
        let (f, src) = resolve(text.as_bytes());
        let projection: Vec<usize> = (0..f.schema().len()).collect();
        let mut out: Vec<Vec<Value>> = projection.iter().map(|_| Vec::new()).collect();
        for split in 0..f.num_splits() {
            let mut ranges = Vec::new();
            f.split_ranges(split, &projection, &mut ranges).expect("split_ranges");
            let cols = f.read_split(&src, split, &projection).expect("read_split");
            for (j, c) in cols.iter().enumerate() {
                for i in 0..c.len() {
                    out[j].push(c.value_at(i));
                }
            }
        }
        (f.schema().iter().map(|c| c.name.clone()).collect(), out)
    }

    #[test]
    fn newline_delimited_json_is_read_as_jsonl() {
        // `COPY ... TO 'x.json'` writes one object per line, as DuckDB does, and reading it back
        // used to fail with a syntax error. DuckDB auto-detects the layout.
        let (names, got) = read_every_split("{\"a\":1,\"b\":\"x\"}\n{\"a\":2,\"b\":\"y\"}\n");
        assert_eq!(names, ["a", "b"]);
        assert_eq!(got[0], vec![Value::I64(1), Value::I64(2)]);
        assert_eq!(got[1], vec![s("x"), s("y")]);
        // CRLF, a BOM and blank lines are all JSONL's business once detected.
        let (_, got) = read_every_split("\u{feff}{\"a\":1}\r\n\r\n{\"a\":2}\r\n");
        assert_eq!(got[0], vec![Value::I64(1), Value::I64(2)]);

        // A first record longer than the detection sample is still recognized.
        let long = "x".repeat(crate::format::jsonl::SAMPLE_BYTES as usize);
        let text = format!("{{\"a\":\"{long}\"}}\n{{\"a\":\"y\"}}\n");
        let (_, got) = read_every_split(&text);
        assert_eq!(got[0], vec![s(&long), s("y")]);

        // A single document is unaffected, including one followed by trailing newlines, and two
        // values on one line are still a syntax error rather than silently NDJSON.
        let (names, got) = read_every_split("[{\"a\":1},\n{\"a\":2}]\n\n");
        assert_eq!(names, ["a"]);
        assert_eq!(got[0], vec![Value::I64(1), Value::I64(2)]);
        assert_eq!(resolve_err("{\"a\":1} {\"a\":2}"), Some(Code::SyntaxError));
    }

    #[test]
    fn too_many_columns_is_rejected() {
        let mut text = String::from("[");
        for chunk in 0..2 {
            if chunk > 0 {
                text.push(',');
            }
            text.push('{');
            for i in 0..MAX_COLUMNS {
                if i > 0 {
                    text.push(',');
                }
                text.push_str(&format!("\"c{}\":1", chunk * MAX_COLUMNS + i));
            }
            text.push('}');
        }
        text.push(']');
        let src = Source::from_bytes(text.into_bytes());
        let mut f = JsonFormat::new();
        assert_eq!(code_of(f.resolve(&src)), Some(Code::LimitExceeded));
    }

    // --- Helper functions ---------------------------------------------------

    #[test]
    fn days_from_civil_matches_known_dates() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(1969, 12, 31), -1);
        assert_eq!(days_from_civil(2000, 3, 1), 11017);
        assert_eq!(days_from_civil(2024, 1, 1), 19723);
        assert_eq!(days_from_civil(2024, 2, 29), 19782);
    }
}
