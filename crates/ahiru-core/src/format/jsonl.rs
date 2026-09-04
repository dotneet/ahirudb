//! JSONL (NDJSON) -- one JSON object per line.
//!
//! ## Splits
//!
//! Being row-oriented it has no statistics, and splits are fixed-length byte chunks
//! (`TEXT_SPLIT_BYTES`). A projection cannot reduce the fetched bytes, but `read_split` can skip
//! decoding unneeded columns (only the second of the two stages in the `format` module's docs applies).
//!
//! ## Handling nested values (a design decision)
//!
//! `crate::vector::Ty` has no LIST / STRUCT. The Parquet side rejects nested schemas with
//! `UnsupportedNested`, but doing the same for JSONL would make "a file with just one array
//! column mixed in" entirely unreadable. Nesting shows up routinely in JSON, so here **nested
//! values are treated as VARCHAR columns holding their raw JSON text**.
//!
//! That is, `SELECT tags FROM t` returns the string `["a","b"]`.
//! **Structural access into the nesting (extracting or expanding elements) is unsupported.**
//! Needing it would mean adding LIST/STRUCT to the type system.
//!
//! ## Lines that are not objects
//!
//! NDJSON permits any JSON value per line, not just objects. When the inference sample holds such
//! a line, the whole file reads as a single VARCHAR column named `json` carrying each line's raw
//! JSON text -- the shape DuckDB gives the same file. A file whose objects are all empty (`{}`)
//! takes the same shape, since a table with no columns at all cannot report a row count.
//!
//! ## Type inference
//!
//! Looking at up to `SAMPLE_LINES` rows within the leading `SAMPLE_BYTES` (grown up to
//! `MAX_SAMPLE_BYTES` when that does not hold one complete line), each column widens along the
//! lattice NULL -> BIGINT -> DOUBLE -> VARCHAR, with BOOLEAN a sibling of the numbers rather than
//! a step below them (`true` mixed with `1` gives VARCHAR -- see `widen`). The column set is the
//! **union** of the per-row keys (in order of first appearance), and a key absent from a row is
//! NULL for that row alone. Unlike CSV, columns can be missing per row -- that is this format's essence.
//!
//! The schema is fixed before any split is read (the file is consumed split by split, so the
//! complete key set is not knowable up front), which means a key that first appears past the
//! sample has no column to go into. That is an error (`ColumnNotFound`, positioned at the line
//! carrying it), not a silent drop -- see `read_split`. `format::json` has the whole document
//! resident and so has no such limit: there, every element contributes its keys.
//!
//! ## The input is untrusted
//!
//! Broken JSON gives `Err` or a NULL cell. It never panics, reads out of range, or recurses
//! infinitely. Scanning does not recurse, and nesting is cut off at `MAX_DEPTH`.

use crate::catalog::Source;
use crate::format::{get_or_internal, ResolveStep, TableFormat, TEXT_MAX_RECORD, TEXT_SPLIT_BYTES};
use crate::json::{byte_at, decode_string, kind_of, scan_string, skip_value, skip_ws, Kind};
use crate::prelude::*;
use crate::vector::{Bitmap, Data, Field, Ty, Vector};

/// How many leading bytes are read for schema inference.
pub const SAMPLE_BYTES: u64 = 256 * 1024;

/// The maximum rows used for schema inference. Looking at more is unlikely to change the types.
pub const SAMPLE_LINES: usize = 1000;

/// The cap on the number of columns inference creates. A stop against unbounded allocation on
/// pathological input whose keys differ per row.
const MAX_COLUMNS: usize = 1024;

/// The ceiling the inference sample may grow to when [`SAMPLE_BYTES`] holds no complete line.
///
/// A first line longer than 256 KiB used to fail the whole file with `LimitExceeded` -- reordering
/// the very same lines made it readable. The ceiling is `TEXT_MAX_RECORD`, the longest line the
/// reader accepts anyway, so asking for more would buy nothing.
const MAX_SAMPLE_BYTES: u64 = TEXT_MAX_RECORD;

pub struct JsonlFormat {
    schema: Vec<Field>,
    resolved: bool,
    total_len: u64,
    split_bytes: u64,
    /// Set when the sample holds a line whose top-level value is not an object.
    ///
    /// NDJSON does not actually require objects -- a line may be any JSON value -- so a file of
    /// bare scalars or arrays used to be rejected outright with `SyntaxError`. DuckDB instead
    /// gives the whole file a single column named `json` holding each line's raw JSON text, and so
    /// does this: one non-object line anywhere in the sample switches the file into that shape
    /// (object lines then come back as their raw text too, exactly as DuckDB renders them).
    ///
    /// It is also switched on when the sample's objects contribute **no columns at all** (a file of
    /// `{}` lines): a zero-column table reports `count(*)` as 0 and makes `SELECT *` a syntax
    /// error, which is a silently wrong answer. DuckDB gives such a file one `json` column holding
    /// each record, and so does this.
    raw_json: bool,
    /// How many leading bytes the schema inference has asked for so far. Starts at
    /// [`SAMPLE_BYTES`] and grows toward [`MAX_SAMPLE_BYTES`] while no complete line fits.
    sample_bytes: u64,
    /// Per column: the sample held only `null` for it, so its `Ty::Varchar` is a default rather
    /// than a reading of the data. See `TableFormat::column_has_no_evidence`.
    no_evidence: Vec<bool>,
}

impl JsonlFormat {
    pub fn new() -> Self {
        JsonlFormat {
            schema: Vec::new(),
            resolved: false,
            total_len: 0,
            split_bytes: TEXT_SPLIT_BYTES,
            raw_json: false,
            sample_bytes: SAMPLE_BYTES,
            no_evidence: Vec::new(),
        }
    }

    /// The split size. Defaults to `TEXT_SPLIT_BYTES` (8 MiB).
    pub fn split_bytes(&self) -> u64 {
        self.split_bytes
    }

    /// Overrides the split size, so even a small file can exercise the multi-split path
    /// (split-boundary handling could be broken without being noticed unless it is tested).
    /// 0 is rounded to 1.
    pub fn set_split_bytes(&mut self, n: u64) {
        self.split_bytes = n.max(1);
    }

    /// Enlarges the inference sample, returning whether there was any room left to do so.
    ///
    /// The sample strictly grows on every successful call, so `resolve`'s retry loop always makes
    /// progress and cannot spin.
    fn grow_sample(&mut self, total_len: u64) -> bool {
        if self.sample_bytes >= MAX_SAMPLE_BYTES || self.sample_bytes >= total_len {
            return false;
        }
        self.sample_bytes = self.sample_bytes.saturating_mul(4).min(MAX_SAMPLE_BYTES);
        true
    }

    /// The byte range `[start, end)` split `split` is responsible for.
    fn chunk(&self, split: usize) -> (u64, u64) {
        let start = (split as u64).saturating_mul(self.split_bytes).min(self.total_len);
        let end = start.saturating_add(self.split_bytes).min(self.total_len);
        (start, end)
    }
}

impl Default for JsonlFormat {
    fn default() -> Self {
        JsonlFormat::new()
    }
}

impl TableFormat for JsonlFormat {
    fn resolve(&mut self, src: &Source) -> Result<ResolveStep> {
        if self.resolved {
            return Ok(Ok(()));
        }
        self.total_len = src.total_len;
        if src.total_len == 0 {
            // An empty file. There are neither columns nor splits, but resolution itself succeeds.
            self.resolved = true;
            return Ok(Ok(()));
        }
        // The loop only ever repeats after `grow_sample` has strictly increased `sample_bytes`
        // (and only up to `MAX_SAMPLE_BYTES`), so it terminates.
        loop {
            let n = self.sample_bytes.min(src.total_len);
            let buf = match src.get(0, n as usize) {
                Some(b) => b,
                None => return Ok(Err((0, n))),
            };
            // If the sample is not the whole file, the last line may be cut off partway.
            let partial = n < src.total_len;

            let mut names: Vec<String> = Vec::new();
            let mut infs: Vec<Inf> = Vec::new();
            let mut key = Vec::new();

            let mut pos = skip_bom(buf);
            let mut lines = 0usize;
            let mut truncated = false;
            let mut raw = false;
            while pos < buf.len() && lines < SAMPLE_LINES {
                let (line, next, terminated) = next_line(buf, pos);
                pos = next;
                if !terminated && partial {
                    truncated = true;
                    break;
                }
                if is_blank(line) {
                    continue;
                }
                lines += 1;
                // A line that is not an object puts the whole file into raw-JSON mode (see `raw_json`).
                if byte_at(line, skip_ws(line, 0))? != b'{' {
                    raw = true;
                    break;
                }
                let mut it = Members::new(line)?;
                while let Some(m) = it.next()? {
                    let name = member_key(&m, &mut key)?;
                    let idx = match names.iter().position(|s| s.as_bytes() == name) {
                        Some(i) => i,
                        None => {
                            ensure!(names.len() < MAX_COLUMNS, LimitExceeded);
                            names.push(String::from_utf8_lossy(name).into_owned());
                            infs.push(Inf::Null);
                            names.len() - 1
                        }
                    };
                    infs[idx] = widen(infs[idx], infer(&m));
                }
            }
            if truncated && lines == 0 {
                // Not one line fit within the sample. Ask for more before giving up -- a first line
                // longer than `SAMPLE_BYTES` used to fail the whole file even though moving it
                // further down made the very same file readable.
                ensure!(self.grow_sample(src.total_len), LimitExceeded);
                continue;
            }

            // Split indexes are `usize` throughout the execution layer. Do not
            // silently truncate a huge remote file's u64 split count on 32-bit
            // WASM (or for a deliberately tiny test split size).
            let split_count = self.total_len.div_ceil(self.split_bytes.max(1));
            ensure!(usize::try_from(split_count).is_ok(), LimitExceeded);

            // Objects that contribute no columns at all (a file of `{}` lines) would give a
            // zero-column table, whose `count(*)` reads 0 and whose `SELECT *` is a syntax error.
            // Fall back to the same raw-JSON shape DuckDB uses (see `raw_json`).
            self.raw_json = raw || names.is_empty();
            self.schema = if self.raw_json {
                // One VARCHAR column holding each line's raw JSON text, named as DuckDB names it.
                self.no_evidence = vec![false];
                vec![Field::new(String::from("json"), Ty::Varchar, true)]
            } else {
                self.no_evidence = infs.iter().map(|i| *i == Inf::Null).collect();
                // A JSON value can be missing at any time, so every column is nullable.
                names.into_iter().zip(infs).map(|(n, i)| Field::new(n, i.ty(), true)).collect()
            };
            self.resolved = true;
            return Ok(Ok(()));
        }
    }

    fn is_resolved(&self) -> bool {
        self.resolved
    }

    fn schema(&self) -> &[Field] {
        &self.schema
    }

    fn num_splits(&self) -> usize {
        if !self.resolved || self.total_len == 0 {
            return 0;
        }
        self.total_len.div_ceil(self.split_bytes) as usize
    }

    fn split_rows(&self, _split: usize) -> Option<u64> {
        // The row count is unknown until it is read.
        None
    }

    fn split_ranges(
        &self,
        split: usize,
        _projection: &[usize],
        out: &mut Vec<(u64, u64)>,
    ) -> Result<()> {
        ensure!(split < self.num_splits(), Internal);
        let (start, end) = self.chunk(split);
        // Being row-oriented, a projection does not reduce bytes. A split's bytes are always all needed.
        // An extra TEXT_MAX_RECORD is taken so a record straddling the boundary can be finished.
        let read_end = end.saturating_add(TEXT_MAX_RECORD).min(self.total_len);
        out.push((start, read_end));
        Ok(())
    }

    fn read_split(&self, src: &Source, split: usize, projection: &[usize]) -> Result<Vec<Vector>> {
        ensure!(self.resolved, Internal);
        ensure!(split < self.num_splits(), Internal);
        let (chunk_start, chunk_end) = self.chunk(split);
        let read_end = chunk_end.saturating_add(TEXT_MAX_RECORD).min(self.total_len);
        let buf = get_or_internal(src, chunk_start, read_end)?;

        let mut names: Vec<&[u8]> = Vec::with_capacity(projection.len());
        let mut builders: Vec<Builder> = Vec::with_capacity(projection.len());
        for &c in projection {
            let f = match self.schema.get(c) {
                Some(f) => f,
                None => err!(Internal),
            };
            names.push(f.name.as_bytes());
            // A rough reservation assuming around 64 bytes per line. Being off does not affect correctness.
            builders.push(Builder::new(f.ty, (buf.len() / 64).min(1 << 16)));
        }

        // A raw newline cannot appear inside a JSON string (it is escaped as `\n`), so record
        // boundaries are uniquely determined by looking for `\n` alone. That is why, unlike CSV's
        // quoting, splitting needs no state.
        let mut pos = if split == 0 {
            skip_bom(buf)
        } else {
            // Everything up to the first newline is a line the previous split finishes.
            match find_nl(buf, 0) {
                Some(i) => i + 1,
                // No newline within the window = not one line starts in this split.
                None => return Ok(builders.into_iter().map(|b| b.finish()).collect()),
            }
        };

        // Line ownership: split k(>0)'s first line starts "just after the first newline at or past
        // chunk_start", so its start position is always greater than chunk_start. The previous
        // split must therefore cover lines whose start is at or **below** chunk_end, or a line
        // starting exactly at the boundary would be read by no one and vanish.
        let mut slots: Vec<Option<Member>> = vec![None; projection.len()];
        let mut key = Vec::new();
        let mut val = Vec::new();
        while pos < buf.len() {
            let abs = chunk_start + pos as u64;
            if abs > chunk_end {
                break;
            }
            let (line, next) = match find_nl(buf, pos) {
                Some(i) => (strip_cr(&buf[pos..i]), i + 1),
                None => {
                    // No newline. At end of file it is the final line; otherwise it is a record too
                    // long, exceeding TEXT_MAX_RECORD.
                    ensure!(read_end == self.total_len, LimitExceeded);
                    (strip_cr(&buf[pos..]), buf.len())
                }
            };
            pos = next;
            if is_blank(line) {
                continue;
            }

            if self.raw_json {
                // The line is still validated as one complete JSON value; only its shape is free.
                let start = skip_ws(line, 0);
                let end = skip_value(line, start)?;
                ensure!(skip_ws(line, end) == line.len(), SyntaxError, end);
                let body = &line[start..end];
                // A bare `null` line is SQL NULL, not the four-character text "null".
                // `format::json`'s reader already gives NULL for a `null` element, and so
                // does DuckDB; pushing the raw span for every kind made this the one place
                // JSON's own NULL could not be spelled.
                let is_null = kind_of(byte_at(line, start)?) == Kind::Null;
                for b in builders.iter_mut() {
                    b.push(if is_null { None } else { Some(Cell::Bytes(body)) });
                }
                continue;
            }

            for s in slots.iter_mut() {
                *s = None;
            }
            let mut it = Members::new(line)?;
            while let Some(m) = it.next()? {
                let name = member_key(&m, &mut key)?;
                // Only the projected columns have their positions remembered. Other values are not
                // decoded (Members merely skips their value spans).
                if let Some(j) = names.iter().position(|n| *n == name) {
                    slots[j] = Some(m);
                } else if !self.schema.iter().any(|f| f.name.as_bytes() == name) {
                    // A key the schema has never heard of. The schema comes from a bounded
                    // leading sample (`SAMPLE_BYTES` / `SAMPLE_LINES`) and, unlike `format::json`,
                    // is fixed before any split is read -- the file is consumed split by split, so
                    // a complete key set is not available up front. Skipping the key would drop a
                    // whole column the file plainly contains, which is the "silently wrong answer"
                    // `docs/DESIGN.md` §15 rules out (and it is exactly what a later *value* of an
                    // unexpected type is already refused for, above). `pos` is the offset of the
                    // line carrying the key, so the offending record can be found.
                    err!(ColumnNotFound, abs as usize);
                }
            }
            for (j, b) in builders.iter_mut().enumerate() {
                push_member(b, slots[j].as_ref(), &mut val)?;
            }
        }

        Ok(builders.into_iter().map(|b| b.finish()).collect())
    }

    /// JSONL carries no schema, so every column type here is a guess from the leading sample.
    fn schema_is_inferred(&self) -> bool {
        true
    }

    fn column_has_no_evidence(&self, col: usize) -> bool {
        self.no_evidence.get(col).copied().unwrap_or(false)
    }
}

// --- Type inference -----------------------------------------------------------

/// The column type during inference. `Ty` is not used directly because DATE/TIMESTAMP need to be
/// distinguished within the lattice as "specially promoted from a string".
#[derive(Clone, Copy, PartialEq, Eq)]
enum Inf {
    Null,
    Bool,
    Int,
    Double,
    Date,
    Timestamp,
    Varchar,
}

impl Inf {
    fn ty(self) -> Ty {
        match self {
            // Everything in the sample was NULL. VARCHAR is chosen so no later value, whatever it
            // is, gets dropped.
            Inf::Null | Inf::Varchar => Ty::Varchar,
            Inf::Bool => Ty::Boolean,
            Inf::Int => Ty::BigInt,
            Inf::Double => Ty::Double,
            Inf::Date => Ty::Date,
            Inf::Timestamp => Ty::Timestamp,
        }
    }
}

/// The lattice NULL -> BIGINT -> DOUBLE -> VARCHAR, with BOOLEAN a sibling of the numbers rather
/// than a step below them. A mixture across families (numbers and dates, say) can only fall to text.
///
/// BOOLEAN mixed with a number widens to **VARCHAR**, not to the numeric type: `true` has no
/// numeric reading, so choosing BIGINT made the reader fail with `InvalidCast` on the very sample
/// rows that produced the guess. `format::csv`'s `widen` resolves the same mixture the same way.
/// (DuckDB gives the column its `JSON` type; this reader predates `Ty::Json` and carries raw JSON
/// text in a VARCHAR, which is the closest thing it has -- see `docs/sql/data-sources.md`.)
fn widen(a: Inf, b: Inf) -> Inf {
    use Inf::*;
    if a == b {
        return a;
    }
    match (a, b) {
        (Null, x) | (x, Null) => x,
        (Int, Double) | (Double, Int) => Double,
        (Date, Timestamp) | (Timestamp, Date) => Timestamp,
        _ => Varchar,
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
                return Inf::Varchar;
            }
            let s = m.str_body();
            if parse_date(s).is_some() {
                Inf::Date
            } else if parse_timestamp(s).is_some() {
                Inf::Timestamp
            } else {
                Inf::Varchar
            }
        }
        // Nesting rides in a VARCHAR as raw JSON text (see the module docs).
        Kind::Object | Kind::Array => Inf::Varchar,
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

/// Pushes one member according to the column's type. A missing key (`None`) and `null` are both NULL.
///
/// A value that is *present* but does not fit the column's type is `InvalidCast`. The schema comes
/// from a bounded leading sample (`SAMPLE_BYTES` / `SAMPLE_LINES`), so a later line can genuinely
/// hold a `2.5` where the sample only ever showed integers -- but silently turning that cell into
/// NULL loses data the file plainly contains, which is the "silently wrong answer" `docs/DESIGN.md`
/// §15 says the engine never produces. DuckDB reports the same situation as a conversion error.
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
// The tokenizer itself (`byte_at`/`skip_ws`/`skip_value` and so on) lives in `crate::json`.
// What remains here is only the iterator returning a top-level object's members in order,
// premised on NDJSON's one object per line.

#[derive(Clone, Copy)]
struct Member<'a> {
    /// The key's body (inside the quotes, escapes unexpanded).
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
struct Members<'a> {
    b: &'a [u8],
    i: usize,
    /// Whether at least one has already been returned. Used to decide whether a comma is required.
    started: bool,
    done: bool,
}

impl<'a> Members<'a> {
    fn new(line: &'a [u8]) -> Result<Self> {
        let i = skip_ws(line, 0);
        // A line that is not an object is not accepted.
        ensure!(byte_at(line, i)? == b'{', SyntaxError, i);
        Ok(Members { b: line, i: i + 1, started: false, done: false })
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
            // Only whitespace may remain after the close. A line like `{}{}` is rejected.
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
        // Skip extra fractional digits, then a time-zone suffix may follow.
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
// `byte_at`/`skip_ws` come straight from `crate::json` (imported above).

/// Skips a UTF-8 BOM if present.
fn skip_bom(b: &[u8]) -> usize {
    if b.starts_with(&[0xEF, 0xBB, 0xBF]) {
        3
    } else {
        0
    }
}

fn find_nl(b: &[u8], from: usize) -> Option<usize> {
    b.get(from..)?.iter().position(|&c| c == b'\n').map(|i| i + from)
}

/// Drops CRLF's `\r`.
fn strip_cr(line: &[u8]) -> &[u8] {
    match line.split_last() {
        Some((b'\r', rest)) => rest,
        _ => line,
    }
}

fn is_blank(line: &[u8]) -> bool {
    line.iter().all(|&c| matches!(c, b' ' | b'\t' | b'\r'))
}

/// Takes one line from `pos`. `(the line, the next position, whether it ended with a newline)`.
fn next_line(b: &[u8], pos: usize) -> (&[u8], usize, bool) {
    match find_nl(b, pos) {
        Some(i) => (strip_cr(&b[pos..i]), i + 1, true),
        None => (strip_cr(&b[pos..]), b.len(), false),
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

    /// Resolves and returns the schema.
    fn resolve(bytes: &[u8]) -> (JsonlFormat, Source) {
        let src = Source::from_bytes(bytes.to_vec());
        let mut f = JsonlFormat::new();
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

    /// Reads every split and returns per-column `Value` sequences.
    fn read_cols(bytes: &[u8], split_bytes: u64, proj: Option<&[usize]>) -> Vec<Vec<Value>> {
        let src = Source::from_bytes(bytes.to_vec());
        let mut f = JsonlFormat::new();
        if split_bytes > 0 {
            f.set_split_bytes(split_bytes);
        }
        f.resolve(&src).expect("resolve").expect("bytes");
        let projection: Vec<usize> = match proj {
            Some(p) => p.to_vec(),
            None => (0..f.schema().len()).collect(),
        };
        let mut out: Vec<Vec<Value>> = projection.iter().map(|_| Vec::new()).collect();
        let mut ranges = Vec::new();
        for s in 0..f.num_splits() {
            ranges.clear();
            f.split_ranges(s, &projection, &mut ranges).expect("split_ranges");
            assert_eq!(ranges.len(), 1, "being row-oriented, there should be exactly one range");
            let cols = f.read_split(&src, s, &projection).expect("read_split");
            assert_eq!(cols.len(), projection.len());
            let rows = cols.first().map_or(0, |c| c.len());
            assert!(cols.iter().all(|c| c.len() == rows), "the column lengths do not agree");
            for (j, c) in cols.iter().enumerate() {
                for i in 0..c.len() {
                    out[j].push(c.value_at(i));
                }
            }
        }
        out
    }

    fn read_all(text: &str) -> Vec<Vec<Value>> {
        read_cols(text.as_bytes(), 0, None)
    }

    fn s(v: &str) -> Value {
        Value::Bytes(v.as_bytes().to_vec())
    }

    /// Infers a single-column file and gets its type.
    fn infer_ty(values: &[&str]) -> Ty {
        let mut text = String::new();
        for v in values {
            text.push_str("{\"a\":");
            text.push_str(v);
            text.push_str("}\n");
        }
        schema_of(&text)[0].1
    }

    // --- Real files ---------------------------------------------------------

    #[test]
    fn basic_jsonl_schema_matches_duckdb() {
        // duckdb: DESCRIBE SELECT * FROM read_json_auto('tests/data/basic.jsonl')
        let (f, _) = resolve(&data_file("basic.jsonl"));
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
    fn basic_jsonl_values_match_the_parquet_reference() {
        let cols = read_cols(&data_file("basic.jsonl"), 0, None);
        assert!(cols.iter().all(|c| c.len() == 1000));
        assert_eq!(cols[0][0], Value::I64(0));
        assert_eq!(cols[1][0], s("name_0"));
        assert_eq!(cols[2][0], Value::F64(0.0));
        assert_eq!(cols[3][0], Value::Bool(true));
        assert_eq!(cols[4][0], Value::Null);
        assert_eq!(cols[2][1], Value::F64(1.5));
        assert_eq!(cols[4][1], Value::I64(100));
        // duckdb: SELECT epoch_us(d) FROM ... LIMIT 1 -> 1704067200000000
        assert_eq!(cols[5][0], Value::I64(1_704_067_200_000_000));
        // The end.
        assert_eq!(cols[0][999], Value::I64(999));
        assert_eq!(cols[1][999], s("name_5"));
        assert_eq!(cols[4][999], Value::I64(99_900));
    }

    #[test]
    fn null_positions_match_the_parquet_file() {
        let cols = read_cols(&data_file("basic.jsonl"), 0, None);
        #[allow(clippy::needless_range_loop)]
        for i in 0..1000 {
            assert_eq!(
                cols[4][i] == Value::Null,
                i % 5 == 0,
                "the NULL determination for big in row {i} is off"
            );
        }
    }

    // --- The union of column sets -------------------------------------------

    #[test]
    fn schema_is_the_union_of_keys_in_first_seen_order() {
        let got = schema_of(
            "{\"b\":1,\"a\":\"x\"}\n\
             {\"c\":true}\n\
             {\"a\":\"y\",\"b\":2}\n",
        );
        let names: Vec<&str> = got.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["b", "a", "c"]);
    }

    #[test]
    fn missing_key_and_explicit_null_are_both_null() {
        let cols = read_all(
            "{\"a\":1,\"b\":2}\n\
             {\"a\":3}\n\
             {\"a\":null,\"b\":4}\n",
        );
        assert_eq!(cols[0], [Value::I64(1), Value::I64(3), Value::Null]);
        // The second row does not have the key b at all.
        assert_eq!(cols[1], [Value::I64(2), Value::Null, Value::I64(4)]);
    }

    // --- Type inference -----------------------------------------------------

    #[test]
    fn inference_lattice_covers_every_transition() {
        assert_eq!(infer_ty(&["null"]), Ty::Varchar); // all-NULL falls to VARCHAR
        assert_eq!(infer_ty(&["true"]), Ty::Boolean);
        assert_eq!(infer_ty(&["null", "true"]), Ty::Boolean);
        assert_eq!(infer_ty(&["1"]), Ty::BigInt);
        // BOOLEAN mixed with a number is VARCHAR: choosing the numeric type used to make the
        // reader fail with `InvalidCast` on the very sample rows the guess came from.
        // (duckdb types the column JSON; VARCHAR carrying raw JSON text is this reader's
        // equivalent -- `Ty::Json` postdates it.)
        assert_eq!(infer_ty(&["true", "1"]), Ty::Varchar);
        assert_eq!(infer_ty(&["1", "1.5"]), Ty::Double);
        assert_eq!(infer_ty(&["true", "1.5"]), Ty::Varchar);
        assert_eq!(infer_ty(&["1", "\"x\""]), Ty::Varchar);
        assert_eq!(infer_ty(&["1.5", "\"x\""]), Ty::Varchar);
        assert_eq!(infer_ty(&["true", "\"x\""]), Ty::Varchar);
    }

    #[test]
    fn date_and_timestamp_strings_get_temporal_types() {
        assert_eq!(infer_ty(&["\"2024-01-01\""]), Ty::Date);
        assert_eq!(infer_ty(&["\"2024-01-01 12:34:56\""]), Ty::Timestamp);
        assert_eq!(infer_ty(&["\"2024-01-01T12:34:56\""]), Ty::Timestamp);
        assert_eq!(infer_ty(&["\"2024-01-01T12:34:56.123456\""]), Ty::Timestamp);
        // A mixture of DATE and TIMESTAMP settles on TIMESTAMP.
        assert_eq!(infer_ty(&["\"2024-01-01\"", "\"2024-01-01 00:00:00\""]), Ty::Timestamp);
        // With another string mixed in it falls to VARCHAR.
        assert_eq!(infer_ty(&["\"2024-01-01\"", "\"hello\""]), Ty::Varchar);
        assert_eq!(infer_ty(&["\"2024-01-01 00:00:00\"", "\"hello\""]), Ty::Varchar);
        // Something invalid as a date does not become a date.
        assert_eq!(infer_ty(&["\"2024-02-30\""]), Ty::Varchar);
        assert_eq!(infer_ty(&["\"2024-13-01\""]), Ty::Varchar);
        // Mixed with numbers it is VARCHAR.
        assert_eq!(infer_ty(&["\"2024-01-01\"", "1"]), Ty::Varchar);
    }

    #[test]
    fn temporal_values_use_the_internal_representation() {
        let cols = read_all("{\"a\":\"1970-01-02\"}\n{\"a\":\"2024-01-01\"}\n");
        assert_eq!(cols[0], [Value::I32(1), Value::I32(19723)]);

        let cols = read_all(
            "{\"a\":\"1970-01-01 00:00:01\"}\n\
             {\"a\":\"1969-12-31 23:59:59\"}\n\
             {\"a\":\"2024-01-01T00:00:00.000123\"}\n",
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
        // An integer not fitting i64 widens to DOUBLE.
        assert_eq!(infer_ty(&["9223372036854775808"]), Ty::Double);
        assert_eq!(infer_ty(&["-9223372036854775809"]), Ty::Double);
        assert_eq!(infer_ty(&["-9223372036854775808"]), Ty::BigInt);

        let cols = read_all("{\"a\":-3}\n{\"a\":9223372036854775807}\n");
        assert_eq!(cols[0], [Value::I64(-3), Value::I64(i64::MAX)]);

        let cols = read_all("{\"a\":1e3}\n{\"a\":-1.5E-2}\n{\"a\":7}\n");
        assert_eq!(cols[0], [Value::F64(1000.0), Value::F64(-0.015), Value::F64(7.0)]);

        // Inference sees only the sample (SAMPLE_LINES rows). A row outside it carrying a value of
        // the wrong type used to become NULL, silently dropping data the file plainly contains; it
        // is now a clear `InvalidCast`.
        for tail in ["{\"a\":1.5}", "{\"a\":\"x\"}", "{\"a\":[1]}"] {
            let mut text = String::new();
            for _ in 0..SAMPLE_LINES {
                text.push_str("{\"a\":1}\n");
            }
            text.push_str(tail);
            text.push('\n');
            let mut f = JsonlFormat::new();
            let src = Source::from_bytes(text.into_bytes());
            f.resolve(&src).unwrap().unwrap();
            assert_eq!(code_of(f.read_split(&src, 0, &[0])), Some(Code::InvalidCast), "{tail}");
        }
        // A missing key and an explicit `null` are still NULL, not an error.
        let mut text = String::new();
        for _ in 0..SAMPLE_LINES {
            text.push_str("{\"a\":1}\n");
        }
        text.push_str("{}\n{\"a\":null}\n{\"a\":2}\n");
        let cols = read_all(&text);
        assert_eq!(cols[0].len(), SAMPLE_LINES + 3);
        assert_eq!(cols[0][SAMPLE_LINES], Value::Null);
        assert_eq!(cols[0][SAMPLE_LINES + 1], Value::Null);
        assert_eq!(cols[0][SAMPLE_LINES + 2], Value::I64(2));
    }

    // A key that first appears past the inference sample has no column to go into. It
    // used to be skipped without a word, so a whole column of data the file plainly
    // contains simply vanished: `SELECT *` showed only the sampled columns and no error
    // was raised anywhere. The schema is fixed before the splits are read, so it cannot
    // be repaired here -- but it must be loud (`ColumnNotFound`, positioned at the
    // offending line) rather than silent.
    #[test]
    fn a_key_first_seen_past_the_sample_is_an_error_not_a_silent_drop() {
        let mut text = String::new();
        for _ in 0..SAMPLE_LINES {
            text.push_str("{\"a\":1}\n");
        }
        let tail_at = text.len();
        text.push_str("{\"a\":1,\"b\":2}\n");
        let (f, src) = resolve(text.as_bytes());
        assert_eq!(f.schema().len(), 1, "only `a` was sampled");
        let e = match f.read_split(&src, 0, &[0]) {
            Ok(_) => panic!("the late key `b` was silently dropped"),
            Err(e) => e,
        };
        assert_eq!(e.code, Code::ColumnNotFound);
        assert_eq!(e.pos as usize, tail_at, "the position should locate the offending line");
    }

    // The check must not fire for a key that *is* in the schema but is simply not part
    // of this query's projection -- `read_split` only remembers the projected columns'
    // positions, so an unprojected key looks identical there without the schema lookup.
    #[test]
    fn an_unprojected_but_known_key_is_not_reported_as_unknown() {
        let text = "{\"a\":1,\"b\":2}\n{\"a\":3,\"b\":4}\n";
        let (f, src) = resolve(text.as_bytes());
        assert_eq!(f.schema().len(), 2);
        let cols = f.read_split(&src, 0, &[1]).unwrap();
        assert_eq!(cols[0].len(), 2);
    }

    // --- Strings ------------------------------------------------------------

    #[test]
    fn string_escapes_are_decoded() {
        let cols = read_all("{\"a\":\"q\\\"b\\\\s\\/t\\bf\\f\\nn\\rr\\tt\"}\n");
        assert_eq!(cols[0][0], s("q\"b\\s/t\u{8}f\u{c}\nn\rr\tt"));
    }

    #[test]
    fn unicode_escapes_and_surrogate_pairs() {
        // \u00e9 = e-acute, \u65e5 = a CJK character, \uD83D\uDE00 = an emoji
        let cols = read_all("{\"a\":\"\\u00e9\\u65e5\\uD83D\\uDE00\"}\n");
        assert_eq!(cols[0][0], s("\u{e9}\u{65e5}\u{1f600}"));
    }

    #[test]
    fn lone_surrogates_become_the_replacement_character() {
        // Unpaired high/low surrogates become U+FFFD rather than an error.
        let cols = read_all(
            "{\"a\":\"\\uD800\"}\n\
             {\"a\":\"\\uDC00x\"}\n\
             {\"a\":\"\\uD800\\u0041\"}\n",
        );
        assert_eq!(cols[0][0], s("\u{FFFD}"));
        assert_eq!(cols[0][1], s("\u{FFFD}x"));
        assert_eq!(cols[0][2], s("\u{FFFD}A"));
    }

    #[test]
    fn escaped_keys_match_their_decoded_name() {
        let (f, _) = resolve(b"{\"a\\u0062\":1}\n{\"ab\":2}\n");
        // They resolve to the same name, so there is one column.
        assert_eq!(f.schema().len(), 1);
        assert_eq!(f.schema()[0].name, "ab");
        let cols = read_all("{\"a\\u0062\":1}\n{\"ab\":2}\n");
        assert_eq!(cols[0], [Value::I64(1), Value::I64(2)]);
    }

    // --- Nesting ------------------------------------------------------------

    #[test]
    fn nested_values_are_kept_as_raw_json_text() {
        let text = "{\"a\":[1,2,{\"x\":\"y\"}],\"b\":{\"k\":[true,null]}}\n\
                    {\"a\":[],\"b\":{}}\n";
        let sch = schema_of(text);
        assert_eq!(sch[0].1, Ty::Varchar);
        assert_eq!(sch[1].1, Ty::Varchar);
        let cols = read_all(text);
        assert_eq!(cols[0][0], s("[1,2,{\"x\":\"y\"}]"));
        assert_eq!(cols[1][0], s("{\"k\":[true,null]}"));
        assert_eq!(cols[0][1], s("[]"));
        assert_eq!(cols[1][1], s("{}"));
    }

    #[test]
    fn varchar_column_keeps_raw_text_for_non_strings() {
        // A column that became VARCHAR through mixed types carries numbers and booleans as raw text too.
        let cols = read_all("{\"a\":1}\n{\"a\":\"x\"}\n{\"a\":true}\n{\"a\":2.5}\n{\"a\":null}\n");
        assert_eq!(cols[0], [s("1"), s("x"), s("true"), s("2.5"), Value::Null]);
    }

    // --- Row handling -------------------------------------------------------

    #[test]
    fn blank_lines_are_skipped() {
        let cols = read_all("\n{\"a\":1}\n\n   \n{\"a\":2}\n\n");
        assert_eq!(cols[0], [Value::I64(1), Value::I64(2)]);
    }

    #[test]
    fn bom_and_crlf_are_handled() {
        let mut bytes = vec![0xEF, 0xBB, 0xBF];
        bytes.extend_from_slice(b"{\"a\":1}\r\n\r\n{\"a\":2}\r\n");
        let (f, _) = resolve(&bytes);
        assert_eq!(f.schema()[0].name, "a");
        let cols = read_cols(&bytes, 0, None);
        assert_eq!(cols[0], [Value::I64(1), Value::I64(2)]);
    }

    #[test]
    fn last_line_without_a_newline_is_read() {
        let cols = read_all("{\"a\":1}\n{\"a\":2}");
        assert_eq!(cols[0], [Value::I64(1), Value::I64(2)]);
    }

    // --- Projection ---------------------------------------------------------

    #[test]
    fn projection_returns_requested_columns_in_order() {
        let bytes = data_file("basic.jsonl");
        let cols = read_cols(&bytes, 0, Some(&[4, 1]));
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0][1], Value::I64(100)); // big
        assert_eq!(cols[1][1], s("name_1")); // name
        assert_eq!(cols[0][0], Value::Null);
        // It does not break with an empty projection either.
        let cols = read_cols(&bytes, 0, Some(&[]));
        assert!(cols.is_empty());
    }

    // --- Splits -------------------------------------------------------------

    #[test]
    fn split_count_and_ranges_cover_the_file() {
        let bytes = data_file("basic.jsonl");
        let src = Source::from_bytes(bytes.clone());
        let mut f = JsonlFormat::new();
        f.set_split_bytes(1024);
        f.resolve(&src).unwrap().unwrap();
        let total = bytes.len() as u64;
        assert_eq!(f.num_splits() as u64, total.div_euclid(1024) + 1);
        let mut out = Vec::new();
        for i in 0..f.num_splits() {
            out.clear();
            f.split_ranges(i, &[], &mut out).unwrap();
            let (a, b) = out[0];
            assert_eq!(a, i as u64 * 1024);
            // The trailing overread is capped by the file length.
            assert_eq!(b, (a + 1024 + TEXT_MAX_RECORD).min(total));
        }
        // An out-of-range split cannot be requested.
        assert_eq!(code_of(f.split_ranges(f.num_splits(), &[], &mut out)), Some(Code::Internal));
    }

    #[test]
    fn multi_split_reads_every_row_exactly_once() {
        let bytes = data_file("basic.jsonl");
        let reference = read_cols(&bytes, 0, None);
        // Changing the split size must not change the row count, the values, or the order.
        // At 84 bytes per line, every value below cuts partway through a line.
        for &sb in &[1u64, 7, 63, 84, 85, 128, 1000, 4096] {
            let cols = read_cols(&bytes, sb, None);
            assert_eq!(cols[0].len(), 1000, "the row count is off at split_bytes={sb}");
            for (j, col) in cols.iter().enumerate() {
                assert_eq!(col, &reference[j], "column {j} is off at split_bytes={sb}");
            }
        }
    }

    #[test]
    fn split_boundary_values_are_not_duplicated_or_dropped() {
        // Lines are fixed at 11 bytes (`{"a":1000}\n`), and split sizes are tried both as multiples
        // of the line length (boundaries landing exactly at a line start = the case most prone to
        // dropping rows) and as non-multiples.
        let mut text = String::new();
        for i in 1000..1500 {
            text.push_str(&format!("{{\"a\":{i}}}\n"));
        }
        let bytes = text.into_bytes();
        assert_eq!(bytes.len(), 500 * 11);
        let expect: Vec<Value> = (1000..1500).map(Value::I64).collect();
        for &sb in &[5u64, 10, 11, 12, 22, 23, 110, 121] {
            let cols = read_cols(&bytes, sb, None);
            assert_eq!(cols[0], expect, "a boundary row is off at split_bytes={sb}");
        }
    }

    #[test]
    fn oversized_record_is_rejected_instead_of_being_silently_split() {
        // A line exceeding TEXT_MAX_RECORD cannot be overread. A line starting before the boundary
        // and continuing past the read window gives LimitExceeded.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"{\"a\":1}\n{\"a\":");
        bytes.resize(bytes.len() + (TEXT_MAX_RECORD as usize + 16), b'0');
        bytes.extend_from_slice(b"}\n");
        let src = Source::from_bytes(bytes);
        let mut f = JsonlFormat::new();
        f.set_split_bytes(8);
        f.resolve(&src).unwrap().unwrap();
        assert_eq!(code_of(f.read_split(&src, 0, &[0])), Some(Code::LimitExceeded));
    }

    // --- Corrupt input ------------------------------------------------------

    fn read_err(text: &str) -> Option<Code> {
        let src = Source::from_bytes(text.as_bytes().to_vec());
        let mut f = JsonlFormat::new();
        match f.resolve(&src) {
            Err(e) => return Some(e.code),
            Ok(Err(_)) => return None,
            Ok(Ok(())) => {}
        }
        let p: Vec<usize> = (0..f.schema().len()).collect();
        code_of(f.read_split(&src, 0, &p))
    }

    #[test]
    fn non_object_lines_read_as_one_raw_json_column() {
        // NDJSON permits any JSON value per line. Such a file used to be rejected with
        // SyntaxError; duckdb gives it a single column named `json` holding each line's raw text.
        let mut f = JsonlFormat::new();
        let src = Source::from_bytes(b"1\n[1,2]\n\"s\"\n".to_vec());
        f.resolve(&src).unwrap().unwrap();
        assert_eq!(f.schema().len(), 1);
        assert_eq!(f.schema()[0].name, "json");
        assert_eq!(f.schema()[0].ty, Ty::Varchar);
        let cols = read_all("1\n[1,2]\n\"s\"\n");
        assert_eq!(cols[0], [s("1"), s("[1,2]"), s("\"s\"")]);

        // One non-object line anywhere puts the whole file into that shape, objects included --
        // duckdb behaves the same way.
        let cols = read_all("{\"a\":1}\n5\n");
        assert_eq!(cols[0], [s("{\"a\":1}"), s("5")]);

        // The raw text is still validated as one complete JSON value per line.
        assert_eq!(read_err("[1,2\n"), Some(Code::UnexpectedEof));
        assert_eq!(read_err("1 2\n"), Some(Code::SyntaxError));

        // A bare `null` line is SQL NULL, not the four-character text "null". Pushing the
        // raw span for every kind made this the only place JSON's own NULL could not be
        // spelled: `SELECT json IS NULL` came back false. `format::json` and duckdb both
        // give NULL here.
        let cols = read_all("1\nnull\n\"s\"\n");
        assert_eq!(cols[0], [s("1"), Value::Null, s("\"s\"")]);
    }

    #[test]
    fn malformed_input_is_an_error_not_a_panic() {
        // An unclosed string.
        assert_eq!(read_err("{\"a\":\"abc}\n"), Some(Code::UnexpectedEof));
        // An unclosed object.
        assert_eq!(read_err("{\"a\":1\n"), Some(Code::UnexpectedEof));
        assert_eq!(read_err("{\"a\":{\"b\":1}\n"), Some(Code::UnexpectedEof));
        // A line that is not an object is no longer an error -- it puts the file into raw-JSON
        // mode (see `raw_json` and `non_object_lines_read_as_one_raw_json_column`).
        // Garbage following the close.
        assert_eq!(read_err("{\"a\":1}{\"a\":2}\n"), Some(Code::SyntaxError));
        // A missing separator or colon, and a trailing comma.
        assert_eq!(read_err("{\"a\":1 \"b\":2}\n"), Some(Code::SyntaxError));
        assert_eq!(read_err("{\"a\" 1}\n"), Some(Code::SyntaxError));
        assert_eq!(read_err("{\"a\":1,}\n"), Some(Code::SyntaxError));
        // A malformed number or literal.
        assert_eq!(read_err("{\"a\":1.}\n"), Some(Code::SyntaxError));
        assert_eq!(read_err("{\"a\":-}\n"), Some(Code::SyntaxError));
        assert_eq!(read_err("{\"a\":tru}\n"), Some(Code::SyntaxError));
        assert_eq!(read_err("{\"a\":01}\n"), Some(Code::SyntaxError));
        // Unbalanced brackets.
        assert_eq!(read_err("{\"a\":[1,2}}\n"), Some(Code::SyntaxError));
        // \u's digits are not hex, or are too few.
        assert_eq!(read_err("{\"a\":\"\\u12zz\"}\n"), Some(Code::SyntaxError));
        assert_eq!(read_err("{\"a\":\"\\u12\"}\n"), Some(Code::SyntaxError));
        // An unknown escape.
        assert_eq!(read_err("{\"a\":\"\\x\"}\n"), Some(Code::SyntaxError));
    }

    #[test]
    fn deeply_nested_input_hits_the_depth_cap() {
        use crate::json::MAX_DEPTH;
        let mut text = String::from("{\"a\":");
        let deep = MAX_DEPTH as usize + 1;
        for _ in 0..deep {
            text.push('[');
        }
        text.push('1');
        for _ in 0..deep {
            text.push(']');
        }
        text.push_str("}\n");
        assert_eq!(read_err(&text), Some(Code::NestingTooDeep));

        // Exactly at the limit passes (the raw text rides in a VARCHAR column).
        let mut ok = String::from("{\"a\":");
        for _ in 0..MAX_DEPTH {
            ok.push('[');
        }
        ok.push('1');
        for _ in 0..MAX_DEPTH {
            ok.push(']');
        }
        ok.push_str("}\n");
        assert_eq!(read_err(&ok), None);
    }

    #[test]
    fn too_many_columns_is_rejected() {
        // A few rows each carrying a great many keys, shaped so the column cap is reached before
        // SAMPLE_LINES caps things.
        let mut text = String::new();
        for chunk in 0..2 {
            text.push('{');
            for i in 0..MAX_COLUMNS {
                if i > 0 {
                    text.push(',');
                }
                text.push_str(&format!("\"c{}\":1", chunk * MAX_COLUMNS + i));
            }
            text.push_str("}\n");
        }
        let src = Source::from_bytes(text.into_bytes());
        let mut f = JsonlFormat::new();
        assert_eq!(code_of(f.resolve(&src)), Some(Code::LimitExceeded));
    }

    #[test]
    fn a_file_of_empty_objects_is_one_json_column_rather_than_no_columns() {
        // `{}\n{}\n` used to infer zero columns, so `count(*)` read 0 (a `Batch` with no columns
        // has no row count to report) and `SELECT *` was a syntax error. duckdb gives the file one
        // column named `json` holding each record.
        let (f, _) = resolve(b"{}\n{}\n");
        assert_eq!(schema_of("{}\n{}\n"), vec![(String::from("json"), Ty::Varchar)]);
        assert_eq!(f.num_splits(), 1);
        let cols = read_all("{}\n{}\n");
        assert_eq!(cols[0], [s("{}"), s("{}")]);
    }

    #[test]
    fn a_bool_and_number_mixture_widens_to_text_instead_of_failing_its_own_sample() {
        // `true` has no numeric reading, so BIGINT made the reader fail with InvalidCast on the
        // very sample lines that produced the guess. duckdb types the column JSON; raw JSON text
        // in a VARCHAR is this reader's equivalent (see the module docs).
        assert_eq!(schema_of("{\"a\":true}\n{\"a\":1}\n"), vec![(String::from("a"), Ty::Varchar)]);
        let cols = read_all("{\"a\":true}\n{\"a\":1}\n");
        assert_eq!(cols[0], [s("true"), s("1")]);
    }

    #[test]
    fn a_first_line_longer_than_the_initial_sample_grows_it_instead_of_failing() {
        // A first line over SAMPLE_BYTES used to fail the whole file with LimitExceeded -- moving
        // the very same line further down made the file readable, which is no rule at all.
        let pad = "x".repeat(SAMPLE_BYTES as usize);
        let text = format!("{{\"a\":1,\"b\":\"{pad}\"}}\n{{\"a\":2,\"b\":\"y\"}}\n");
        let (f, _) = resolve(text.as_bytes());
        assert!(f.sample_bytes > SAMPLE_BYTES);
        assert_eq!(f.schema()[0].ty, Ty::BigInt);
        let cols = read_cols(text.as_bytes(), 0, Some(&[0]));
        assert_eq!(cols[0], [Value::I64(1), Value::I64(2)]);

        // A line longer than the ceiling is still a limit rather than an endless growth loop.
        let huge = "x".repeat(MAX_SAMPLE_BYTES as usize);
        let text = format!("{{\"a\":\"{huge}\"}}\n{{\"a\":\"y\"}}\n");
        let src = Source::from_bytes(text.into_bytes());
        let mut f = JsonlFormat::new();
        assert_eq!(code_of(f.resolve(&src)), Some(Code::LimitExceeded));
    }

    // --- resolve's I/O requests ---------------------------------------------

    #[test]
    fn resolve_requests_the_sample_range_when_bytes_are_missing() {
        let mut f = JsonlFormat::new();
        let src = Source::remote(1_000_000);
        match f.resolve(&src).unwrap() {
            Err((off, len)) => {
                assert_eq!(off, 0);
                assert_eq!(len, SAMPLE_BYTES);
            }
            Ok(()) => panic!("it cannot possibly resolve with no bytes"),
        }
        assert!(!f.is_resolved());
        assert_eq!(f.num_splits(), 0);

        // It does not request a range longer than the file.
        let mut f = JsonlFormat::new();
        let src = Source::remote(100);
        match f.resolve(&src).unwrap() {
            Err((off, len)) => assert_eq!((off, len), (0, 100)),
            Ok(()) => panic!("it cannot possibly resolve with no bytes"),
        }
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn resolve_rejects_a_split_count_that_does_not_fit_usize() {
        let mut f = JsonlFormat::new();
        f.set_split_bytes(1);
        let mut src = Source::remote(u64::MAX);
        let mut sample = vec![0u8; SAMPLE_BYTES as usize];
        sample[..8].copy_from_slice(b"{\"a\":1}\n");
        src.insert(0, sample);
        assert_eq!(code_of(f.resolve(&src)), Some(Code::LimitExceeded));
    }

    #[test]
    fn empty_file_resolves_to_an_empty_schema() {
        let (f, _) = resolve(b"");
        assert!(f.is_resolved());
        assert!(f.schema().is_empty());
        assert_eq!(f.num_splits(), 0);
        assert_eq!(f.split_rows(0), None);
    }

    #[test]
    fn unresolved_format_reports_no_splits() {
        let f = JsonlFormat::new();
        assert!(!f.is_resolved());
        assert_eq!(f.num_splits(), 0);
        assert!(f.schema().is_empty());
        assert_eq!(f.split_bytes(), TEXT_SPLIT_BYTES);
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
