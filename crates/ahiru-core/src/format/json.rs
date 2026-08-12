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
//!   named `"json"` (easier to handle than a degenerate table with no columns at all).
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
//! The low-level JSON scanner (scanning strings and numbers, decoding escapes, detecting
//! dates/timestamps) follows the same idea as `format::jsonl`'s implementation, but `jsonl`'s types
//! are private and cannot be reused from this module, and changing `jsonl.rs` itself is outside
//! this task's scope (its existing implementation is optimized for the split-boundary I/O barrier
//! and is left alone), so this is written up as an independent implementation. The structure is
//! kept as close as possible, so unifying them into one later should be feasible.
//!
//! ## The input is untrusted
//!
//! The same policy as `format::jsonl`. Broken JSON gives `Err` (reinterpreting as a NULL cell
//! applies only after the schema is settled, and only to values whose type falls outside the sample).
//! Scanning does not recurse, and nesting is cut off at `MAX_DEPTH`.

use crate::catalog::Source;
use crate::format::{get_or_internal, ResolveStep, TableFormat};
use crate::prelude::*;
use crate::vector::{Bitmap, Data, Field, Ty, Vector};

/// The maximum leading rows used for schema inference (how column types widen).
/// The same idea as `format::jsonl::SAMPLE_LINES`. Elements beyond it are still syntax-checked
/// (the design has `resolve` read the whole file through) and counted toward the row count, but
/// they do not enter the widen computation.
pub const SAMPLE_ELEMENTS: usize = 1000;

/// The nesting limit for values. The same reason and the same value as `format::jsonl::MAX_DEPTH`.
const MAX_DEPTH: u32 = 32;

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
}

impl JsonFormat {
    pub fn new() -> Self {
        JsonFormat { schema: Vec::new(), resolved: false, total_len: 0, row_count: 0 }
    }
}

impl Default for JsonFormat {
    fn default() -> Self {
        JsonFormat::new()
    }
}

impl TableFormat for JsonFormat {
    fn resolve(&mut self, src: &Source) -> Result<ResolveStep> {
        if self.resolved {
            return Ok(Ok(()));
        }
        ensure!(src.total_len <= MAX_JSON_BYTES, Oom);
        // An empty file has no top-level value and so is invalid as JSON to begin with
        // (unlike JSONL, this premises reading "one JSON document").
        ensure!(src.total_len > 0, UnexpectedEof);
        self.total_len = src.total_len;
        let buf = match src.get(0, src.total_len as usize) {
            Some(b) => b,
            // It requests "the whole file" rather than a split boundary barrier (see the module docs).
            None => return Ok(Err((0, src.total_len))),
        };
        let (schema, row_count) = parse_schema(buf)?;
        self.schema = schema;
        self.row_count = row_count;
        self.resolved = true;
        Ok(Ok(()))
    }

    fn is_resolved(&self) -> bool {
        self.resolved
    }

    fn schema(&self) -> &[Field] {
        &self.schema
    }

    fn num_splits(&self) -> usize {
        // The non-streaming design: the split is always the whole file, exactly one (see the module docs).
        if self.resolved {
            1
        } else {
            0
        }
    }

    fn split_rows(&self, split: usize) -> Option<u64> {
        if split == 0 {
            Some(self.row_count)
        } else {
            None
        }
    }

    fn split_ranges(
        &self,
        split: usize,
        _projection: &[usize],
        out: &mut Vec<(u64, u64)>,
    ) -> Result<()> {
        ensure!(split < self.num_splits(), Internal);
        // The structure is not settled, so the whole file is always needed even with a projection.
        out.push((0, self.total_len));
        Ok(())
    }

    fn read_split(&self, src: &Source, split: usize, projection: &[usize]) -> Result<Vec<Vector>> {
        ensure!(self.resolved, Internal);
        ensure!(split < self.num_splits(), Internal);
        let buf = get_or_internal(src, 0, self.total_len)?;

        let mut names: Vec<&[u8]> = Vec::with_capacity(projection.len());
        let mut builders: Vec<Builder> = Vec::with_capacity(projection.len());
        for &c in projection {
            let f = match self.schema.get(c) {
                Some(f) => f,
                None => err!(Internal),
            };
            names.push(f.name.as_bytes());
            builders.push(Builder::new(f.ty, 64));
        }

        let mut slots: Vec<Option<Member>> = vec![None; projection.len()];
        let mut key = Vec::new();
        let mut val = Vec::new();

        let i0 = skip_ws(buf, skip_bom(buf));
        if byte_at(buf, i0)? == b'[' {
            let mut it = Elements::new(buf, i0)?;
            while let Some((s, e)) = it.next()? {
                process_row(&buf[s..e], &names, &mut slots, &mut builders, &mut key, &mut val)?;
            }
            // Only whitespace may remain after the array's closing bracket.
            ensure!(skip_ws(buf, it.i) == buf.len(), SyntaxError, it.i);
        } else {
            let end = skip_value(buf, i0)?;
            ensure!(skip_ws(buf, end) == buf.len(), SyntaxError, end);
            process_row(&buf[i0..end], &names, &mut slots, &mut builders, &mut key, &mut val)?;
        }

        Ok(builders.into_iter().map(|b| b.finish()).collect())
    }
}

// --- Schema inference ----------------------------------------------------------

/// Resolves all of `buf` as one JSON document and returns `(schema, row count)`.
/// The row count is exact even beyond `SAMPLE_ELEMENTS` (every element is walked for the syntax
/// check anyway, so there is no extra cost). Only the widen computation is limited to the sample.
fn parse_schema(buf: &[u8]) -> Result<(Vec<Field>, u64)> {
    let i = skip_ws(buf, skip_bom(buf));
    let c = byte_at(buf, i)?;

    let mut names: Vec<String> = Vec::new();
    let mut infs: Vec<Inf> = Vec::new();
    let mut key = Vec::new();
    let mut row_count: u64 = 0;

    if c == b'[' {
        let mut it = Elements::new(buf, i)?;
        while let Some((s, e)) = it.next()? {
            row_count += 1;
            if row_count as usize <= SAMPLE_ELEMENTS {
                accumulate_row(&buf[s..e], &mut names, &mut infs, &mut key)?;
            }
        }
        ensure!(skip_ws(buf, it.i) == buf.len(), SyntaxError, it.i);
        if row_count == 0 {
            // An empty array. Matching duckdb is easier to handle than a degenerate table with no
            // columns at all, so it becomes a JSON-typed column named `"json"`.
            names.push(String::from("json"));
            infs.push(Inf::Json);
        }
    } else {
        // The top level is not an array: a single object, or a bare scalar.
        // Both are treated as "one row" (see the module docs).
        let end = skip_value(buf, i)?;
        ensure!(skip_ws(buf, end) == buf.len(), SyntaxError, end);
        row_count = 1;
        accumulate_row(&buf[i..end], &mut names, &mut infs, &mut key)?;
    }

    let schema = names.into_iter().zip(infs).map(|(n, i)| Field::new(n, i.ty(), true)).collect();
    Ok((schema, row_count))
}

/// Folds one row's worth (one array element, or a single top-level value) into the column set.
/// `row` is exactly that value's bytes (with no surrounding whitespace).
fn accumulate_row(
    row: &[u8],
    names: &mut Vec<String>,
    infs: &mut Vec<Inf>,
    key: &mut Vec<u8>,
) -> Result<()> {
    let c = byte_at(row, 0)?;
    if c == b'{' {
        let mut it = Members::new(row)?;
        while let Some(m) = it.next()? {
            let name = member_key(&m, key)?;
            merge(names, infs, name, infer(&m))?;
        }
    } else {
        // A non-object row puts the raw value into a single `"json"` column
        // (see the module docs).
        let m = Member { key: b"json", key_escaped: false, val: row, kind: kind_of(c) };
        merge(names, infs, b"json", infer(&m))?;
    }
    Ok(())
}

/// Widens column `name`'s inferred type with `inf`. Creates the column if it does not exist.
fn merge(names: &mut Vec<String>, infs: &mut Vec<Inf>, name: &[u8], inf: Inf) -> Result<()> {
    let idx = match names.iter().position(|s| s.as_bytes() == name) {
        Some(i) => i,
        None => {
            ensure!(names.len() < MAX_COLUMNS, LimitExceeded);
            names.push(String::from_utf8_lossy(name).into_owned());
            infs.push(Inf::Null);
            names.len() - 1
        }
    };
    infs[idx] = widen(infs[idx], inf);
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

/// The lattice NULL -> BOOLEAN -> BIGINT -> DOUBLE; the string family (Str/Date/Timestamp) falls to
/// VARCHAR among themselves, and any other incompatible pair falls to JSON
/// (see the duckdb measurements in the module docs).
fn widen(a: Inf, b: Inf) -> Inf {
    use Inf::*;
    if a == b {
        return a;
    }
    match (a, b) {
        (Null, x) | (x, Null) => x,
        (Bool, Int) | (Int, Bool) => Int,
        (Bool, Double) | (Double, Bool) => Double,
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

    /// Both `None` and a physical type mismatch give NULL. Discarding a value of the wrong type is
    /// because inference comes from a sample and later rows can fall outside (it is not an error).
    fn push(&mut self, cell: Option<Cell<'_>>) {
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
    }

    fn finish(self) -> Vector {
        let v = if self.any_null { Some(self.valid) } else { None };
        Vector::from_data(self.ty, self.data, v)
    }
}

/// Distributes one row's values to their positions among the projected columns (`names`).
fn process_row<'a>(
    row: &'a [u8],
    names: &[&[u8]],
    slots: &mut [Option<Member<'a>>],
    builders: &mut [Builder],
    key: &mut Vec<u8>,
    val: &mut Vec<u8>,
) -> Result<()> {
    for s in slots.iter_mut() {
        *s = None;
    }
    let c = byte_at(row, 0)?;
    if c == b'{' {
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
fn push_member(b: &mut Builder, m: Option<&Member<'_>>, scratch: &mut Vec<u8>) -> Result<()> {
    let m = match m {
        Some(m) if m.kind != Kind::Null => m,
        _ => {
            b.push(None);
            return Ok(());
        }
    };
    match b.ty {
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
                b.push(Some(Cell::Bytes(scratch)));
            }
            _ => b.push(Some(Cell::Bytes(m.val))),
        },
    }
    Ok(())
}

// --- JSON scanning --------------------------------------------------------------
//
// An independent implementation following the same idea as `format::jsonl`'s private scanner (see the module docs).

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

/// Skips one value starting at `b[i]` and returns the position just after it.
///
/// Non-recursive. The kinds of open containers live in a `u32` bit stack, and depth is capped by
/// `MAX_DEPTH` (kept equal to the bit width of `u32`).
fn skip_value(b: &[u8], start: usize) -> Result<usize> {
    let mut i = start;
    // Bit 1 = object, 0 = array. The lowest bit is the current container.
    let mut stack: u32 = 0;
    let mut depth: u32 = 0;

    'value: loop {
        i = skip_ws(b, i);
        let c = byte_at(b, i)?;
        if c == b'{' || c == b'[' {
            let obj = c == b'{';
            ensure!(depth < MAX_DEPTH, NestingTooDeep, i);
            stack = (stack << 1) | obj as u32;
            depth += 1;
            i = skip_ws(b, i + 1);
            let n = byte_at(b, i)?;
            if (obj && n == b'}') || (!obj && n == b']') {
                // An empty container. Falls through to the closing logic below as though one value had been consumed.
                i += 1;
                stack >>= 1;
                depth -= 1;
            } else {
                if obj {
                    i = skip_member_key(b, i)?;
                }
                continue 'value;
            }
        } else {
            i = skip_scalar(b, i)?;
        }

        // One value consumed. Handle the separator or closing bracket.
        loop {
            if depth == 0 {
                return Ok(i);
            }
            i = skip_ws(b, i);
            let c = byte_at(b, i)?;
            let obj = stack & 1 == 1;
            match c {
                b',' => {
                    i += 1;
                    if obj {
                        i = skip_member_key(b, skip_ws(b, i))?;
                    }
                    continue 'value;
                }
                b'}' if obj => {
                    i += 1;
                    stack >>= 1;
                    depth -= 1;
                }
                b']' if !obj => {
                    i += 1;
                    stack >>= 1;
                    depth -= 1;
                }
                _ => err!(SyntaxError, i),
            }
        }
    }
}

/// Skips `"key" :` and returns the position where the value starts.
fn skip_member_key(b: &[u8], i: usize) -> Result<usize> {
    let i = skip_ws(b, i);
    ensure!(byte_at(b, i)? == b'"', SyntaxError, i);
    let (_, _, ni) = scan_string(b, i)?;
    let ni = skip_ws(b, ni);
    ensure!(byte_at(b, ni)? == b':', SyntaxError, ni);
    Ok(ni + 1)
}

fn skip_scalar(b: &[u8], i: usize) -> Result<usize> {
    match byte_at(b, i)? {
        b'"' => Ok(scan_string(b, i)?.2),
        b't' => skip_lit(b, i, b"true"),
        b'f' => skip_lit(b, i, b"false"),
        b'n' => skip_lit(b, i, b"null"),
        _ => scan_number(b, i),
    }
}

fn skip_lit(b: &[u8], i: usize, w: &[u8]) -> Result<usize> {
    let e = i + w.len();
    ensure!(b.len() >= e, UnexpectedEof, i);
    ensure!(&b[i..e] == w, SyntaxError, i);
    Ok(e)
}

fn scan_number(b: &[u8], start: usize) -> Result<usize> {
    let mut i = start;
    if b.get(i) == Some(&b'-') {
        i += 1;
    }
    let d0 = i;
    while matches!(b.get(i), Some(c) if c.is_ascii_digit()) {
        i += 1;
    }
    ensure!(i > d0, SyntaxError, start);
    if b.get(i) == Some(&b'.') {
        i += 1;
        let f0 = i;
        while matches!(b.get(i), Some(c) if c.is_ascii_digit()) {
            i += 1;
        }
        ensure!(i > f0, SyntaxError, start);
    }
    if matches!(b.get(i), Some(b'e') | Some(b'E')) {
        i += 1;
        if matches!(b.get(i), Some(b'+') | Some(b'-')) {
            i += 1;
        }
        let e0 = i;
        while matches!(b.get(i), Some(c) if c.is_ascii_digit()) {
            i += 1;
        }
        ensure!(i > e0, SyntaxError, start);
    }
    Ok(i)
}

/// `b[i]` is `"`. Returns `(body, whether escaped, the position after the closing quote)`.
fn scan_string(b: &[u8], i: usize) -> Result<(&[u8], bool, usize)> {
    let mut j = i + 1;
    let mut esc = false;
    loop {
        match byte_at(b, j)? {
            b'"' => return Ok((&b[i + 1..j], esc, j + 1)),
            b'\\' => {
                // The next byte is always consumed, so `\"` is not mistaken for the terminator.
                byte_at(b, j + 1)?;
                esc = true;
                j += 2;
            }
            _ => j += 1,
        }
    }
}

/// Expands the escapes in a string body and writes it to `out` as UTF-8.
fn decode_string(body: &[u8], out: &mut Vec<u8>) -> Result<()> {
    let mut i = 0;
    while i < body.len() {
        let c = body[i];
        if c != b'\\' {
            out.push(c);
            i += 1;
            continue;
        }
        i += 1;
        let e = byte_at(body, i)?;
        i += 1;
        match e {
            b'"' => out.push(b'"'),
            b'\\' => out.push(b'\\'),
            b'/' => out.push(b'/'),
            b'b' => out.push(0x08),
            b'f' => out.push(0x0c),
            b'n' => out.push(b'\n'),
            b'r' => out.push(b'\r'),
            b't' => out.push(b'\t'),
            b'u' => {
                let hi = hex4(body, i)?;
                i += 4;
                let cp = if (0xD800..0xDC00).contains(&hi) {
                    // A high surrogate. Combine if a low surrogate follows immediately.
                    match (body.get(i), body.get(i + 1)) {
                        (Some(b'\\'), Some(b'u')) => match hex4(body, i + 2) {
                            Ok(lo) if (0xDC00..0xE000).contains(&lo) => {
                                i += 6;
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            }
                            // An unpaired high surrogate is collapsed to U+FFFD rather than made an
                            // error (losing a whole row to broken input is worse).
                            _ => 0xFFFD,
                        },
                        _ => 0xFFFD,
                    }
                } else if (0xDC00..0xE000).contains(&hi) {
                    // A lone low surrogate.
                    0xFFFD
                } else {
                    hi
                };
                let ch = char::from_u32(cp).unwrap_or(char::REPLACEMENT_CHARACTER);
                let mut buf = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
            _ => err!(SyntaxError, i),
        }
    }
    Ok(())
}

fn hex4(b: &[u8], i: usize) -> Result<u32> {
    let mut v = 0u32;
    for k in 0..4 {
        let c = byte_at(b, i + k)?;
        let d = match c {
            b'0'..=b'9' => (c - b'0') as u32,
            b'a'..=b'f' => (c - b'a' + 10) as u32,
            b'A'..=b'F' => (c - b'A' + 10) as u32,
            _ => err!(SyntaxError, i + k),
        };
        v = (v << 4) | d;
    }
    Ok(v)
}

// --- Numbers and date-times -----------------------------------------------------

/// An integer literal as i64. `None` when it has a decimal point or exponent, or is out of range.
fn parse_i64(s: &[u8]) -> Option<i64> {
    let (neg, ds) = match s.first() {
        Some(b'-') => (true, &s[1..]),
        _ => (false, s),
    };
    if ds.is_empty() {
        return None;
    }
    // Accumulated on the negative side. That avoids special-casing i64::MIN.
    let mut acc: i64 = 0;
    for &c in ds {
        if !c.is_ascii_digit() {
            return None;
        }
        acc = acc.checked_mul(10)?.checked_sub((c - b'0') as i64)?;
    }
    if neg {
        Some(acc)
    } else {
        acc.checked_neg()
    }
}

/// Floating point is left to core's `str::parse::<f64>` (dec2flt).
/// It is smaller than writing one in-house and rounds correctly.
fn parse_f64(s: &[u8]) -> Option<f64> {
    core::str::from_utf8(s).ok()?.parse::<f64>().ok()
}

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

fn byte_at(b: &[u8], i: usize) -> Result<u8> {
    match b.get(i) {
        Some(&c) => Ok(c),
        None => err!(UnexpectedEof, i),
    }
}

fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while matches!(b.get(i), Some(b' ') | Some(b'\t') | Some(b'\r') | Some(b'\n')) {
        i += 1;
    }
    i
}

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
        assert_eq!(infer_ty(&["true", "1"]), Ty::BigInt);
        assert_eq!(infer_ty(&["1", "1.5"]), Ty::Double);
        assert_eq!(infer_ty(&["true", "1.5"]), Ty::Double);
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

        // Inference sees only the sample (SAMPLE_ELEMENTS of them). When a row outside it carries a
        // value of the wrong type, that row alone becomes NULL rather than an error.
        let mut text = String::from("[");
        for i in 0..SAMPLE_ELEMENTS {
            if i > 0 {
                text.push(',');
            }
            text.push_str("{\"a\":1}");
        }
        text.push_str(",{\"a\":1.5},{\"a\":\"x\"},{\"a\":2}]");
        let cols = read_all(&text);
        assert_eq!(cols[0].len(), SAMPLE_ELEMENTS + 3);
        assert_eq!(cols[0][SAMPLE_ELEMENTS], Value::Null);
        assert_eq!(cols[0][SAMPLE_ELEMENTS + 1], Value::Null);
        assert_eq!(cols[0][SAMPLE_ELEMENTS + 2], Value::I64(2));
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
        let mut f = JsonFormat::new();
        let src = Source::remote(1_000_000);
        match f.resolve(&src).unwrap() {
            Err((off, len)) => {
                assert_eq!(off, 0);
                assert_eq!(len, 1_000_000);
            }
            Ok(()) => panic!("it cannot possibly resolve with no bytes"),
        }
        assert!(!f.is_resolved());
        assert_eq!(f.num_splits(), 0);
    }

    #[test]
    fn oversized_file_is_rejected_before_reading_any_bytes() {
        let mut f = JsonFormat::new();
        let src = Source::remote(MAX_JSON_BYTES + 1);
        assert_eq!(code_of(f.resolve(&src)), Some(Code::Oom));
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
    }

    #[test]
    fn deeply_nested_input_hits_the_depth_cap() {
        // Nests the array elements themselves deeply. `Elements::next` calls `skip_value` directly
        // from the element's start (its own `[`/`{`) to find the element's span, so depth is counted
        // from this element's outermost shell (when embedded as an object member, that member's
        // line -- the element's own `{` -- consumes one level, so one less depth is available.
        // `format::jsonl`'s `Members` calls `skip_value` only on a member's value, so its origin
        // does not shift -- this difference is exactly the "independent implementation" note in the
        // module docs).
        let mut text = String::from("[");
        let deep = MAX_DEPTH as usize + 1;
        for _ in 0..deep {
            text.push('[');
        }
        text.push('1');
        for _ in 0..deep {
            text.push(']');
        }
        text.push(']');
        assert_eq!(resolve_err(&text), Some(Code::NestingTooDeep));

        let mut ok = String::from("[");
        for _ in 0..MAX_DEPTH {
            ok.push('[');
        }
        ok.push('1');
        for _ in 0..MAX_DEPTH {
            ok.push(']');
        }
        ok.push(']');
        assert_eq!(resolve_err(&ok), None);
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
