//! A reader adapting CSV / TSV to `TableFormat`.
//!
//! Being a row-oriented format, unlike the Parquet adapter it can neither "drop splits by
//! statistics" nor "reduce fetched bytes by projection". Only two things are possible:
//!
//! 1. Split into fixed-length byte chunks, cutting I/O at split boundaries
//! 2. Skip **converting** columns that are not projected (scanning is needed regardless)
//!
//! The schema is inferred from the leading `SAMPLE_BYTES`, so later rows can fall outside it.
//! A value that falls outside is an error (`InvalidCast`), not a silent NULL: turning it into
//! NULL would drop data the file plainly contains, which is exactly the "silently wrong answer"
//! `docs/DESIGN.md` §15 says the engine never produces. DuckDB reports the same situation as a
//! conversion error. An empty (unquoted) field keeps its NULL meaning, since that is the only way
//! CSV can spell NULL at all.
//!
//! Splitting a file that uses RFC 4180 quoting is unsafe in general: an embedded `\n` inside a
//! quoted field is not a record boundary, but a fixed-size split cut has no way to know, from its
//! own bytes alone, whether the byte right at its boundary sits inside an open quote (the field
//! carrying it can start arbitrarily far back). So `resolve` checks for a `"` byte — in the
//! leading sample, and in the rest of the file when it is already resident — and, if found,
//! forces the whole file to be read as a single split (`quoted_sample` on `CsvFormat`) rather
//! than risk mis-resynchronizing and emitting a phantom row. A remote file whose first sample
//! has no `"` but a later split does is rejected with `UnsupportedFeature` rather than
//! returning a wrong row count.

use crate::catalog::Source;
use crate::format::{get_or_internal, ResolveStep, TableFormat, TEXT_MAX_RECORD, TEXT_SPLIT_BYTES};
use crate::prelude::*;
use crate::rt::hash::eq_ascii_ci;
use crate::vector::{Bitmap, Data, Field, Ty, Vector};

/// How many leading bytes are requested for schema inference.
///
/// The larger it is the better the inference, but the first query's wait grows accordingly.
const SAMPLE_BYTES: u64 = 256 * 1024;

/// The maximum number of rows used for type inference. The sample bytes may run out first.
const SAMPLE_ROWS: usize = 1000;

/// The initial row capacity of a column buffer. The input is untrusted, so the allocation is not
/// sized from an estimated row count (so a huge allocation cannot be planted).
const ROW_CAP: usize = 256;

pub struct CsvFormat {
    delimiter: u8,
    schema: Vec<Field>,
    /// The absolute offset just after the header row (= the start of the first data record).
    data_start: u64,
    total_len: u64,
    resolved: bool,
    /// The bytes per split. Defaults to [`TEXT_SPLIT_BYTES`].
    ///
    /// Making it smaller reduces one split's memory at the cost of more I/O round trips.
    /// It is also used in tests to create several splits.
    pub split_bytes: u64,
    /// The maximum bytes in one record. Defaults to [`TEXT_MAX_RECORD`].
    ///
    /// It also doubles as the "overread amount" for finishing a record that straddles a split
    /// boundary, so a file with a record exceeding it cannot be read split (`LimitExceeded`).
    pub max_record: u64,
    /// Set once, in `resolve`, when a `"` byte is found in the leading sample, or in the
    /// rest of a fully-resident file.
    ///
    /// A split boundary can only ever be mis-resynchronized (§ the module doc's quoting caveat)
    /// when the file uses RFC 4180 quoting: an unquoted CSV/TSV field can never contain the
    /// delimiter, `\n`, or `\r`, so the naive "resync at the first `\n`" scan in `read_split` is
    /// exact for such files. Determining whether a `\n` found *at* a split boundary sits inside an
    /// open quote would need unbounded backward context (the field carrying it can start anywhere
    /// earlier in the file) -- context a single split step does not have, by the split-boundary
    /// I/O barrier design (`docs/DESIGN.md` §6). Rather than guess, quoting forces the whole file
    /// to be read as a single split, which sidesteps the ambiguity entirely (at the cost of the
    /// per-split I/O parallelism, per `docs/sql/limitations.md`).
    quoted_sample: bool,
    /// Set in `resolve` when the leading sample contains a `\r` but no `\n` at all: a file written
    /// with classic Mac (CR-only) line endings.
    ///
    /// Such a file used to be read as one giant record -- a garbage header and zero data rows, with
    /// no error at all. When this is set, a lone `\r` terminates a record (see `Scanner::cr_term`).
    ///
    /// Like `quoted_sample` it also forces the file to be read as a single split: `read_split`'s
    /// boundary resynchronization scans for `\n`, which a CR-only file never contains, so every
    /// split past the first would otherwise resolve to "no record starts here" and silently drop
    /// its rows. A file that has any `\n` keeps the RFC 4180 reading exactly as before, so CRLF and
    /// a `\r` inside a quoted field are unaffected.
    cr_only: bool,
}

impl CsvFormat {
    pub fn new(delimiter: u8) -> Self {
        CsvFormat {
            delimiter,
            schema: Vec::new(),
            data_start: 0,
            total_len: 0,
            resolved: false,
            split_bytes: TEXT_SPLIT_BYTES,
            max_record: TEXT_MAX_RECORD,
            quoted_sample: false,
            cr_only: false,
        }
    }

    /// The byte that ends a record for this file: `\r` for a CR-only file, `\n` otherwise.
    fn term_byte(&self) -> u8 {
        if self.cr_only {
            b'\r'
        } else {
            b'\n'
        }
    }

    /// Whether the whole file must be read as one split. See `quoted_sample` and `cr_only`.
    fn single_split(&self) -> bool {
        self.quoted_sample || self.cr_only
    }

    pub fn delimiter(&self) -> u8 {
        self.delimiter
    }

    /// The byte count of the data region (everything but the header).
    fn data_len(&self) -> u64 {
        self.total_len.saturating_sub(self.data_start)
    }

    /// 0 would break the split-count computation, so it is always rounded up to at least 1.
    ///
    /// When the leading sample looked quoted (`quoted_sample`) or used CR-only line endings
    /// (`cr_only`), this returns the whole data region so `num_splits` collapses to (at most) 1 --
    /// see those fields' doc comments for why such a file cannot be safely resynchronized at an
    /// arbitrary split boundary.
    fn chunk_size(&self) -> u64 {
        if self.single_split() {
            self.data_len().max(1)
        } else {
            self.split_bytes.max(1)
        }
    }

    /// The boundaries of split `split`.
    ///
    /// It returns `(owned interval start, owned interval end, fetch start, fetch end)`.
    /// The owned interval is half-open, and records **starting within it** belong to this split.
    ///
    /// There are two reasons the fetch range is wider than the owned interval:
    ///
    /// - On the back: the overread needed to finish a record that started at the owned interval's end.
    /// - One byte on the front: to know "whether the preceding byte was a line terminator".
    ///   Without it, when a record starts exactly at the owned interval's beginning, that one row
    ///   belongs to no split and vanishes (because "skip to the first line terminator" and the
    ///   half-open ownership rule would disagree).
    fn chunk(&self, split: usize) -> Result<(u64, u64, u64, u64)> {
        ensure!(split < self.num_splits(), Internal);
        let size = self.chunk_size();
        let off = match (split as u64).checked_mul(size) {
            Some(v) => v,
            None => err!(Internal),
        };
        let start = match self.data_start.checked_add(off) {
            Some(v) => v,
            None => err!(Internal),
        };
        let end = start.saturating_add(size).min(self.total_len);
        let fetch_start = if split == 0 { start } else { start - 1 };
        let fetch_end = end.saturating_add(self.max_record).min(self.total_len);
        Ok((start, end, fetch_start, fetch_end))
    }
}

impl TableFormat for CsvFormat {
    fn resolve(&mut self, src: &Source) -> Result<ResolveStep> {
        if self.resolved {
            return Ok(Ok(()));
        }
        self.total_len = src.total_len;
        if src.total_len == 0 {
            // An empty file. Resolved as 0 columns and 0 splits. Not an error.
            self.data_start = 0;
            self.resolved = true;
            return Ok(Ok(()));
        }

        let n = SAMPLE_BYTES.min(src.total_len);
        let sample = match src.get(0, n as usize) {
            Some(b) => b,
            None => return Ok(Err((0, n))),
        };

        // A UTF-8 BOM is dropped. Leaving it would mix into the first column's name and break column references.
        let bom = if sample.starts_with(&[0xEF, 0xBB, 0xBF]) { 3 } else { 0 };
        let body = &sample[bom..];

        // See `quoted_sample`'s doc comment: any `"` is treated as evidence the file may
        // use RFC 4180 quoting, which forces reading it as a single split. The leading
        // sample is checked first; when the whole file is already resident, the rest is
        // scanned too so a quote that first appears after `SAMPLE_BYTES` is not missed.
        self.quoted_sample = body.contains(&b'"');
        if !self.quoted_sample && src.is_complete() && src.total_len > n {
            if let Some(all) = src.get(0, src.total_len as usize) {
                self.quoted_sample = all[n as usize..].contains(&b'"');
            }
        }

        // A file whose sample carries `\r` but not a single `\n` uses classic Mac (CR-only) line
        // endings. Reading it under the RFC 4180 rules would swallow the whole file as one record,
        // so a lone `\r` becomes the record terminator instead (see `cr_only`).
        self.cr_only = !body.contains(&b'\n') && body.contains(&b'\r');

        // --- The header row -------------------------------------------------
        let mut sc = Scanner::new(body, self.delimiter).with_cr_term(self.cr_only);
        let mut raw: Vec<Vec<u8>> = Vec::new();
        let mut scratch = Vec::new();
        let terminated = loop {
            let f = sc.field()?;
            raw.push(field_value(body, &f, &mut scratch).to_vec());
            if f.term != Term::Field {
                break f.term == Term::Record;
            }
        };
        // A header not fitting the sample is anomalous input. Rejecting it as a limit is better
        // than growing the requested range forever and creating a loop that never advances.
        ensure!(terminated || n == src.total_len, LimitExceeded);
        self.data_start = bom as u64 + sc.pos() as u64;

        let names = column_names(&raw);

        // `num_splits` returns a `usize` because the execution layer indexes
        // splits with `usize`. Reject a remote file whose split count cannot
        // be represented instead of truncating the u64 division on 32-bit
        // WASM (or after a caller intentionally chooses a tiny test split).
        let split_count = self.data_len().div_ceil(self.chunk_size());
        ensure!(usize::try_from(split_count).is_ok(), LimitExceeded);

        // --- Type inference ---------------------------------------------------
        // Cut at the last line terminator so a truncated trailing record does not enter the inference.
        // (It may cut at a newline inside quotes, but all that happens then is "that column widens
        //  to VARCHAR", which errs safe.)
        let rest = &body[sc.pos()..];
        let term = self.term_byte();
        let rest = if n < src.total_len {
            match rest.iter().rposition(|&c| c == term) {
                Some(i) => &rest[..i + 1],
                None => &rest[..0],
            }
        } else {
            rest
        };

        let mut cands = vec![Cand::Empty; names.len()];
        let mut sc = Scanner::new(rest, self.delimiter).with_cr_term(self.cr_only);
        let mut rows = 0;
        'sample: while rows < SAMPLE_ROWS && !sc.at_end() {
            if sc.skip_blank_line() {
                continue;
            }
            let mut fi = 0;
            loop {
                // A broken record merely stops the inference. When actually reading it later,
                // it will fail at the same spot, so there's no need to bail out here.
                let f = match sc.field() {
                    Ok(f) => f,
                    Err(_) => break 'sample,
                };
                if let Some(slot) = cands.get_mut(fi) {
                    let v = field_value(rest, &f, &mut scratch);
                    // An empty cell is treated as NULL, so it does not count as evidence for a type.
                    if !v.is_empty() {
                        *slot = widen(*slot, classify(v));
                    }
                }
                fi += 1;
                if f.term != Term::Field {
                    break;
                }
            }
            rows += 1;
        }

        // CSV has no way to express NOT NULL, so every column is nullable.
        self.schema =
            names.into_iter().zip(cands).map(|(name, c)| Field::new(name, c.ty(), true)).collect();
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
        if !self.resolved {
            return 0;
        }
        let len = self.data_len();
        if len == 0 {
            return 0;
        }
        let size = self.chunk_size();
        len.div_ceil(size) as usize
    }

    /// The row count is unknown until the data is read. Since this is only used for estimation, `None` is fine.
    fn split_rows(&self, _split: usize) -> Option<u64> {
        None
    }

    fn split_ranges(
        &self,
        split: usize,
        _projection: &[usize],
        out: &mut Vec<(u64, u64)>,
    ) -> Result<()> {
        // Projection is ignored: since this is row-oriented, all bytes of the split are needed
        // Without scanning, not even the field boundaries are settled.
        let (_, _, fetch_start, fetch_end) = self.chunk(split)?;
        out.push((fetch_start, fetch_end));
        Ok(())
    }

    fn read_split(&self, src: &Source, split: usize, projection: &[usize]) -> Result<Vec<Vector>> {
        ensure!(self.resolved, Internal);
        let (own_start, own_end, fetch_start, fetch_end) = self.chunk(split)?;
        let buf = get_or_internal(src, fetch_start, fetch_end)?;
        let ncols = self.schema.len();

        // So that columns of equal length can be returned even when the projection names the same
        // column twice, it is built over a deduplicated set first and reordered at the end.
        let mut uniq: Vec<usize> = Vec::with_capacity(projection.len());
        for &c in projection {
            ensure!(c < ncols, Internal);
            if !uniq.contains(&c) {
                uniq.push(c);
            }
        }
        let mut slot_of: Vec<Option<usize>> = vec![None; ncols];
        let mut cols: Vec<ColBuf> = Vec::with_capacity(uniq.len());
        for (slot, &c) in uniq.iter().enumerate() {
            match self.schema.get(c) {
                Some(f) => cols.push(ColBuf::new(f.ty, ROW_CAP)),
                None => err!(Internal),
            }
            if let Some(s) = slot_of.get_mut(c) {
                *s = Some(slot);
            }
        }

        // A quote past the leading sample on a remote (incomplete-at-resolve) file
        // cannot be turned into a single-split plan after the fact. Refusing beats
        // mis-resynchronizing a quoted newline at a later split boundary.
        ensure!(self.single_split() || split == 0 || !buf.contains(&b'"'), UnsupportedFeature);

        let lead = (own_start - fetch_start) as usize;
        let mut sc = Scanner::new(buf, self.delimiter).with_cr_term(self.cr_only);
        if lead > 0 {
            // If the byte just before the owned interval is a line terminator, a record starts at
            // the beginning. Otherwise it skips to just after the first line terminator (that
            // record was finished by the previous split).
            //
            // This scan cannot know the quoting state at this position in general -- it may
            // mistake a newline inside quotes for a line terminator, and "record length <=
            // max_record" alone cannot prevent that (a limitation every implementation that reads
            // RFC 4180 CSV in parallel shares). That is exactly why `resolve` forces `num_splits`
            // down to 1 whenever the leading sample looked quoted (`quoted_sample`): `lead > 0`
            // (i.e. `split > 0`) is unreachable for such a file, since there is only ever one
            // split. This byte scan is only ever live for files confirmed (by that sample) not to
            // use quoting, where an unquoted field can never contain the delimiter/`\n`/`\r`, so
            // scanning for a raw `\n` here is exact, not a heuristic. A CR-only file (`cr_only`)
            // likewise collapses to one split, so this `\n` scan never runs for one.
            match buf.first() {
                Some(&b'\n') => sc.seek(1),
                _ => match buf.iter().skip(1).position(|&c| c == b'\n') {
                    Some(i) => sc.seek(i + 2),
                    // No line terminator within the overread range = no record starts in this split
                    // (or the record is too long). Return empty.
                    None => return finish(cols, &uniq, projection),
                },
            }
        }

        let limit = (own_end - fetch_start) as usize;
        let at_file_end = fetch_end >= self.total_len;
        let mut scratch = Vec::new();
        while sc.pos() < limit && !sc.at_end() {
            // Blank lines do not count as rows (the same treatment as a trailing newline not adding a row).
            if sc.skip_blank_line() {
                continue;
            }
            let mut fi = 0;
            loop {
                let f = match sc.field() {
                    Ok(f) => f,
                    Err(e) => {
                        // Cut off at the buffer's end means not a syntax error but too long a record.
                        ensure!(at_file_end, LimitExceeded);
                        return Err(e);
                    }
                };
                if let Some(Some(slot)) = slot_of.get(fi) {
                    let v = field_value(buf, &f, &mut scratch);
                    if let Some(col) = cols.get_mut(*slot) {
                        col.push(v, f.quoted)?;
                    }
                } // Columns outside the projection are not converted. Only the scan is done.
                fi += 1;
                if f.term != Term::Field {
                    // If the buffer runs out without meeting a line terminator: at end of file that
                    // is the final record; otherwise the overread was insufficient.
                    ensure!(f.term != Term::Eof || at_file_end, LimitExceeded);
                    break;
                }
            }
            // A row with too few fields has the rest set to NULL (surplus fields are discarded).
            for c in fi..ncols {
                if let Some(Some(slot)) = slot_of.get(c) {
                    if let Some(col) = cols.get_mut(*slot) {
                        col.push_null();
                    }
                }
            }
        }

        finish(cols, &uniq, projection)
    }

    /// CSV carries no schema, so every column type here is a guess from the leading sample.
    fn schema_is_inferred(&self) -> bool {
        true
    }
}

/// Restores the column buffers into the projection's order.
fn finish(cols: Vec<ColBuf>, uniq: &[usize], projection: &[usize]) -> Result<Vec<Vector>> {
    let vecs: Vec<Vector> = cols.into_iter().map(|c| c.finish()).collect();
    if uniq.len() == projection.len() {
        // With no duplicates, uniq is the projection itself. No copy is needed.
        return Ok(vecs);
    }
    let mut out = Vec::with_capacity(projection.len());
    for &c in projection {
        match uniq.iter().position(|&u| u == c).and_then(|i| vecs.get(i)) {
            Some(v) => out.push(v.clone()),
            None => err!(Internal),
        }
    }
    Ok(out)
}

// --- Column names --------------------------------------------------------------

/// Decides the column names from the header.
///
/// Empty, non-UTF-8, and duplicate names are replaced with `columnN`. Leaving duplicates would
/// make it impossible to decide which one a column reference means, so they are always made unique.
fn column_names(raw: &[Vec<u8>]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(raw.len());
    for (i, r) in raw.iter().enumerate() {
        let mut name = match core::str::from_utf8(r) {
            Ok(s) if !s.is_empty() => s.to_owned(),
            _ => generated_name(i),
        };
        if out.iter().any(|p| eq_ascii_ci(p.as_bytes(), name.as_bytes())) {
            name = generated_name(i);
        }
        // In case a generated name collides again (an earlier column was literally "column3", say).
        // The length grows each time, so it always terminates.
        while out.iter().any(|p| eq_ascii_ci(p.as_bytes(), name.as_bytes())) {
            name.push('_');
        }
        out.push(name);
    }
    out
}

/// Builds `column0`, `column1`, and so on. `format!` is unavailable, so the digits are written by hand.
fn generated_name(i: usize) -> String {
    let mut s = String::from("column");
    let mut digits = [0u8; 20];
    let mut n = 0;
    let mut v = i;
    loop {
        if n >= digits.len() {
            break;
        }
        digits[n] = b'0' + (v % 10) as u8;
        n += 1;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    while n > 0 {
        n -= 1;
        s.push(digits[n] as char);
    }
    s
}

// --- Scanning ------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Term {
    /// A delimiter. More fields follow.
    Field,
    /// A line terminator.
    Record,
    /// The end of the buffer.
    Eof,
}

/// One field's position. `start..end` points at the value body with quotes removed.
struct Span {
    start: usize,
    end: usize,
    quoted: bool,
    /// It contains `""`, so it must be folded when extracted.
    escaped: bool,
    term: Term,
}

struct Scanner<'a> {
    buf: &'a [u8],
    p: usize,
    delim: u8,
    /// Whether a lone `\r` ends a record (a CR-only file -- see `CsvFormat::cr_only`).
    /// Off by default, which is the RFC 4180 reading: only `\n` and `\r\n` end a record.
    cr_term: bool,
}

impl<'a> Scanner<'a> {
    fn new(buf: &'a [u8], delim: u8) -> Self {
        Scanner { buf, p: 0, delim, cr_term: false }
    }

    fn with_cr_term(mut self, cr_term: bool) -> Self {
        self.cr_term = cr_term;
        self
    }

    fn pos(&self) -> usize {
        self.p
    }

    fn seek(&mut self, p: usize) {
        self.p = p.min(self.buf.len());
    }

    fn at_end(&self) -> bool {
        self.p >= self.buf.len()
    }

    /// If the current position is a blank line, skips it and returns true.
    fn skip_blank_line(&mut self) -> bool {
        match self.buf.get(self.p) {
            Some(&b'\n') => {
                self.p += 1;
                true
            }
            Some(&b'\r') if self.buf.get(self.p + 1) == Some(&b'\n') => {
                self.p += 2;
                true
            }
            Some(&b'\r') if self.cr_term => {
                self.p += 1;
                true
            }
            _ => false,
        }
    }

    /// Scans one field.
    ///
    /// RFC 4180: a field starting with `"` is quoted, an inner `""` is one `"`, and the delimiter,
    /// `\n`, and `\r` become part of the value. **A newline inside quotes is not a line
    /// terminator**.
    fn field(&mut self) -> Result<Span> {
        if self.buf.get(self.p) == Some(&b'"') {
            self.p += 1;
            let start = self.p;
            let mut escaped = false;
            loop {
                match self.buf.get(self.p) {
                    // An unclosed quote. Without cutting off here the whole remainder would be held
                    // as one cell, so it is a syntax error.
                    None => err!(SyntaxError, self.p),
                    Some(&b'"') => {
                        if self.buf.get(self.p + 1) == Some(&b'"') {
                            escaped = true;
                            self.p += 2;
                        } else {
                            let end = self.p;
                            self.p += 1;
                            let term = self.skip_to_term();
                            return Ok(Span { start, end, quoted: true, escaped, term });
                        }
                    }
                    Some(_) => self.p += 1,
                }
            }
        }

        let start = self.p;
        while let Some(&c) = self.buf.get(self.p) {
            if c == self.delim {
                let end = self.p;
                self.p += 1;
                return Ok(Span { start, end, quoted: false, escaped: false, term: Term::Field });
            }
            if c == b'\n' {
                let end = self.p;
                self.p += 1;
                return Ok(Span { start, end, quoted: false, escaped: false, term: Term::Record });
            }
            // CRLF. A lone `\r` is not a line terminator (the split-boundary scan looks only at
            // `\n`, so the two interpretations are kept consistent) -- unless this is a CR-only
            // file, which has no `\n` to scan for and is therefore read as a single split.
            if c == b'\r' {
                if self.buf.get(self.p + 1) == Some(&b'\n') {
                    let end = self.p;
                    self.p += 2;
                    return Ok(Span {
                        start,
                        end,
                        quoted: false,
                        escaped: false,
                        term: Term::Record,
                    });
                }
                if self.cr_term {
                    let end = self.p;
                    self.p += 1;
                    return Ok(Span {
                        start,
                        end,
                        quoted: false,
                        escaped: false,
                        term: Term::Record,
                    });
                }
            }
            self.p += 1;
        }
        Ok(Span { start, end: self.buf.len(), quoted: false, escaped: false, term: Term::Eof })
    }

    /// Advances from after a closing quote to the delimiter or line terminator.
    /// RFC 4180 forbids bytes after a closing quote, but they are discarded and it moves on.
    fn skip_to_term(&mut self) -> Term {
        while let Some(&c) = self.buf.get(self.p) {
            if c == self.delim {
                self.p += 1;
                return Term::Field;
            }
            if c == b'\n' {
                self.p += 1;
                return Term::Record;
            }
            if c == b'\r' {
                if self.buf.get(self.p + 1) == Some(&b'\n') {
                    self.p += 2;
                    return Term::Record;
                }
                if self.cr_term {
                    self.p += 1;
                    return Term::Record;
                }
            }
            self.p += 1;
        }
        Term::Eof
    }
}

/// A field's value bytes. Folded into `scratch` and returned only when it contains `""`.
fn field_value<'b>(buf: &'b [u8], f: &Span, scratch: &'b mut Vec<u8>) -> &'b [u8] {
    if !f.escaped {
        return buf.get(f.start..f.end).unwrap_or(&[]);
    }
    scratch.clear();
    let mut i = f.start;
    while i < f.end {
        match buf.get(i) {
            Some(&b'"') => {
                scratch.push(b'"');
                i += 2; // fold `""` into one
            }
            Some(&c) => {
                scratch.push(c);
                i += 1;
            }
            None => break,
        }
    }
    scratch
}

// --- Type inference ------------------------------------------------------------

/// The candidate type during inference.
///
/// It widens in the order BOOLEAN -> BIGINT -> DOUBLE -> VARCHAR. DATE and TIMESTAMP are a
/// separate branch, and mixing them settles on TIMESTAMP. VARCHAR is an absorbing state.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
enum Cand {
    /// Not a single non-empty value has been seen yet.
    Empty,
    Bool,
    Int,
    Double,
    Date,
    Ts,
    Text,
}

impl Cand {
    fn ty(self) -> Ty {
        match self {
            // An all-empty column is left as VARCHAR. There is no basis for deciding it is numeric.
            Cand::Empty | Cand::Text => Ty::Varchar,
            Cand::Bool => Ty::Boolean,
            Cand::Int => Ty::BigInt,
            Cand::Double => Ty::Double,
            Cand::Date => Ty::Date,
            Cand::Ts => Ty::Timestamp,
        }
    }
}

/// Whether the value's leading digit run is zero padded (`007`, `0123`, `007.5`).
///
/// Such a value is text that merely looks numeric -- a zip code, an account number, a product ID --
/// and reading it as a number destroys it (`007` becomes `7`). DuckDB's sniffer refuses it for the
/// same reason, so inference keeps the column VARCHAR. A single `0`, and a fraction like `0.5`,
/// have no padding and stay numeric.
///
/// This governs **inference only**. `parse_i64`/`parse_f64` stay permissive, so an explicitly
/// numeric column still reads `007` as 7 (as DuckDB's `'007'::BIGINT` does).
fn is_zero_padded(v: &[u8]) -> bool {
    let ds = match v.first() {
        Some(b'+') | Some(b'-') => v.get(1..).unwrap_or(&[]),
        _ => v,
    };
    matches!((ds.first(), ds.get(1)), (Some(b'0'), Some(d)) if d.is_ascii_digit())
}

fn classify(v: &[u8]) -> Cand {
    let padded = is_zero_padded(v);
    if parse_bool(v).is_some() {
        Cand::Bool
    } else if !padded && parse_i64(v).is_some() {
        Cand::Int
    } else if !padded && parse_f64(v).is_some() {
        Cand::Double
    } else if parse_date(v).is_some() {
        Cand::Date
    } else if parse_ts(v).is_some() {
        Cand::Ts
    } else {
        Cand::Text
    }
}

/// Widens the candidate type. Any combination leaving the lattice becomes VARCHAR.
///
/// BOOLEAN mixed with a numeric type widens to **VARCHAR**, not to the numeric type: `true` has no
/// numeric reading, so picking BIGINT would drop a value the inference sample itself just saw.
/// DuckDB resolves the same mixture to VARCHAR.
///
/// DATE mixed with TIMESTAMP widens to TIMESTAMP, and a date-only value then reads as midnight
/// (see `parse_ts`), so nothing is lost there.
fn widen(a: Cand, b: Cand) -> Cand {
    use Cand::*;
    if a == b {
        return a;
    }
    match (a, b) {
        (Empty, x) | (x, Empty) => x,
        (Int, Double) | (Double, Int) => Double,
        (Date, Ts) | (Ts, Date) => Ts,
        _ => Text,
    }
}

// --- Value conversion ----------------------------------------------------------

fn parse_bool(v: &[u8]) -> Option<bool> {
    if eq_ascii_ci(v, b"true") {
        Some(true)
    } else if eq_ascii_ci(v, b"false") {
        Some(false)
    } else {
        None
    }
}

fn parse_i64(v: &[u8]) -> Option<i64> {
    let (neg, ds) = match v.first()? {
        b'-' => (true, v.get(1..)?),
        b'+' => (false, v.get(1..)?),
        _ => (false, v),
    };
    if ds.is_empty() {
        return None;
    }
    // Accumulated on the negative side; otherwise i64::MIN is not representable.
    let mut acc: i64 = 0;
    for &c in ds {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        acc = acc.checked_mul(10)?.checked_sub(d as i64)?;
    }
    if neg {
        Some(acc)
    } else {
        acc.checked_neg()
    }
}

fn parse_f64(v: &[u8]) -> Option<f64> {
    // IEEE specials. Both spellings of the infinities are accepted: the writer emits
    // `inf` / `-inf` (matching DuckDB, whose reader sniffs `Infinity` as a DATE), but
    // files written before that -- and by other tools -- use the long form.
    if eq_ascii_ci(v, b"nan") {
        return Some(f64::NAN);
    }
    let signed = match v.first() {
        Some(b'+') => &v[1..],
        Some(b'-') => &v[1..],
        _ => v,
    };
    if eq_ascii_ci(signed, b"inf") || eq_ascii_ci(signed, b"infinity") {
        return Some(if v.first() == Some(&b'-') { f64::NEG_INFINITY } else { f64::INFINITY });
    }
    // Hex is still rejected so only decimal CSV numbers pass.
    let mut i = if matches!(v.first(), Some(b'+') | Some(b'-')) { 1 } else { 0 };
    let mut digits = 0usize;
    while let Some(&c) = v.get(i) {
        if !c.is_ascii_digit() {
            break;
        }
        digits += 1;
        i += 1;
    }
    if v.get(i) == Some(&b'.') {
        i += 1;
        while let Some(&c) = v.get(i) {
            if !c.is_ascii_digit() {
                break;
            }
            digits += 1;
            i += 1;
        }
    }
    if digits == 0 {
        return None;
    }
    if matches!(v.get(i), Some(b'e') | Some(b'E')) {
        i += 1;
        if matches!(v.get(i), Some(b'+') | Some(b'-')) {
            i += 1;
        }
        let mut exp = 0usize;
        while let Some(&c) = v.get(i) {
            if !c.is_ascii_digit() {
                break;
            }
            exp += 1;
            i += 1;
        }
        if exp == 0 {
            return None;
        }
    }
    if i != v.len() {
        return None;
    }
    // The value conversion itself is left to core's `f64::from_str`. Writing a correctly rounding
    // decimal-to-binary conversion in-house would cost either accuracy or code size.
    core::str::from_utf8(v).ok()?.parse::<f64>().ok()
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap(y) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Days since the epoch (1970-01-01). Howard Hinnant's days_from_civil.
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    // Shifting to a March-based year puts the leap day at the year's end and removes the case split.
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let mp = ((m + 9) % 12) as i64; // March = 0
    let doy = (153 * mp + 2) / 5 + d as i64 - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}

/// Reads a decimal digit string. `None` if anything but digits is mixed in.
fn dec(b: &[u8]) -> Option<u32> {
    if b.is_empty() {
        return None;
    }
    let mut a = 0u32;
    for &c in b {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        a = a * 10 + d as u32;
    }
    Some(a)
}

/// Reads the leading 10 bytes as `YYYY-MM-DD`.
fn parse_ymd(v: &[u8]) -> Option<(i64, u32, u32)> {
    if v.len() < 10 || v.get(4) != Some(&b'-') || v.get(7) != Some(&b'-') {
        return None;
    }
    let y = dec(v.get(0..4)?)? as i64;
    let m = dec(v.get(5..7)?)?;
    let d = dec(v.get(8..10)?)?;
    if !(1..=12).contains(&m) || d < 1 || d > days_in_month(y, m) {
        return None;
    }
    Some((y, m, d))
}

/// DATE is days since the epoch (I32).
fn parse_date(v: &[u8]) -> Option<i32> {
    if v.len() != 10 {
        return None;
    }
    let (y, m, d) = parse_ymd(v)?;
    let days = days_from_civil(y, m, d);
    if days < i32::MIN as i64 || days > i32::MAX as i64 {
        return None;
    }
    Some(days as i32)
}

/// TIMESTAMP is microseconds since the epoch (I64).
/// The accepted form is `YYYY-MM-DD[[ T]HH:MM:SS[.ffffff]]`.
///
/// A date-only value is accepted as that date's midnight, matching DuckDB's
/// `TIMESTAMP '2024-01-02'`. Rejecting it would silently NULL every date-only row of a column that
/// mixes dates and timestamps (`widen` settles such a column on TIMESTAMP).
fn parse_ts(v: &[u8]) -> Option<i64> {
    if v.len() == 10 {
        let (y, m, d) = parse_ymd(v)?;
        return days_from_civil(y, m, d).checked_mul(86_400)?.checked_mul(1_000_000);
    }
    if v.len() < 19 {
        return None;
    }
    let (y, m, d) = parse_ymd(v)?;
    if !matches!(v.get(10), Some(b' ') | Some(b'T')) {
        return None;
    }
    if v.get(13) != Some(&b':') || v.get(16) != Some(&b':') {
        return None;
    }
    let h = dec(v.get(11..13)?)?;
    let mi = dec(v.get(14..16)?)?;
    let s = dec(v.get(17..19)?)?;
    if h > 23 || mi > 59 || s > 59 {
        return None;
    }
    let mut us = 0i64;
    let mut i = 19;
    if v.get(19) == Some(&b'.') {
        i = 20;
        let mut scale = 100_000i64;
        while let Some(&c) = v.get(i) {
            if !c.is_ascii_digit() {
                break;
            }
            // The 7th digit onward is discarded (normalized to microsecond precision).
            if scale > 0 {
                us += (c - b'0') as i64 * scale;
                scale /= 10;
            }
            i += 1;
        }
        if i == 20 {
            return None;
        }
    }
    let mut off = 0i64;
    if i < v.len() {
        let (o, n) = crate::format::scan_tz_suffix(&v[i..])?;
        i += n;
        off = o;
    }
    if i != v.len() {
        return None;
    }
    let secs = days_from_civil(y, m, d)
        .checked_mul(86_400)?
        .checked_add((h * 3600 + mi * 60 + s) as i64)?;
    // Even for negative times, `seconds * 1e6 + fraction` gets the direction right (-1 s + 999999 us = -1 us).
    secs.checked_mul(1_000_000)?.checked_add(us)?.checked_sub(off)
}

// --- Column buffers ------------------------------------------------------------

/// The under-construction buffer for one column. It does not go through `Value`, so there is no per-row allocation.
struct ColBuf {
    ty: Ty,
    data: Data,
    validity: Bitmap,
}

impl ColBuf {
    fn new(ty: Ty, cap: usize) -> Self {
        ColBuf {
            ty,
            data: Data::with_capacity(ty.phys(), cap),
            validity: Bitmap::with_capacity(cap),
        }
    }

    fn push_null(&mut self) {
        match &mut self.data {
            Data::Bool(b) => b.push(false),
            Data::I32(v) => v.push(0),
            Data::I64(v) => v.push(0),
            Data::F64(v) => v.push(0.0),
            Data::I128(v) => v.push(0),
            Data::Bytes(b) => b.push_empty(),
        }
        self.validity.push(false);
    }

    /// Pushes one cell.
    ///
    /// An unquoted empty cell is NULL and `""` is the empty string. Being able to distinguish them
    /// is the only way CSV can express NULL, so that is not broken here. In a non-VARCHAR column
    /// there is no such thing as an empty value, so `""` is NULL there too.
    ///
    /// A non-empty value that does not fit the inferred type is `InvalidCast`. The schema comes
    /// from a bounded leading sample, so a later row can genuinely fall outside it -- but turning
    /// that cell into NULL loses data the file plainly contains, without a word to the caller.
    /// DuckDB reports the same situation as a conversion error, and suggests widening the sample.
    fn push(&mut self, v: &[u8], quoted: bool) -> Result<()> {
        if v.is_empty() && (!quoted || !matches!(self.data, Data::Bytes(_))) {
            self.push_null();
            return Ok(());
        }
        let ok = match &mut self.data {
            Data::Bool(b) => match parse_bool(v) {
                Some(x) => {
                    b.push(x);
                    true
                }
                None => {
                    b.push(false);
                    false
                }
            },
            Data::I32(d) => match if self.ty == Ty::Date { parse_date(v) } else { None } {
                Some(x) => {
                    d.push(x);
                    true
                }
                None => {
                    d.push(0);
                    false
                }
            },
            Data::I64(d) => {
                let x = if self.ty == Ty::Timestamp { parse_ts(v) } else { parse_i64(v) };
                match x {
                    Some(x) => {
                        d.push(x);
                        true
                    }
                    None => {
                        d.push(0);
                        false
                    }
                }
            }
            Data::F64(d) => match parse_f64(v) {
                Some(x) => {
                    d.push(x);
                    true
                }
                None => {
                    d.push(0.0);
                    false
                }
            },
            Data::I128(d) => {
                // Inference never picks I128, so reaching this is a bug rather than bad input.
                d.push(0);
                false
            }
            Data::Bytes(b) => {
                b.push(v);
                true
            }
        };
        // The value has already been appended, so the buffer stays consistent even though the
        // caller aborts the whole split here.
        self.validity.push(ok);
        ensure!(ok, InvalidCast);
        Ok(())
    }

    fn finish(self) -> Vector {
        let mut v = Vector::from_data(self.ty, self.data, Some(self.validity));
        // With no NULLs the bitmap is dropped to lighten later operations.
        v.compact_validity();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{code_of, Code};
    use crate::vector::Value;

    fn data_path(name: &str) -> String {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
        std::format!("{p}{name}")
    }

    fn read_data(name: &str) -> Vec<u8> {
        std::fs::read(data_path(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    /// Reads every split and concatenates them vertically. The columns follow `projection`'s order.
    fn read_all(fmt: &CsvFormat, src: &Source, projection: &[usize]) -> Vec<Vec<Value>> {
        let mut out: Vec<Vec<Value>> = projection.iter().map(|_| Vec::new()).collect();
        for s in 0..fmt.num_splits() {
            // It also checks that the bytes contracted by the fetch range are present.
            let mut ranges = Vec::new();
            fmt.split_ranges(s, projection, &mut ranges).expect("split_ranges");
            for (a, b) in &ranges {
                assert!(src.get(*a, (b - a) as usize).is_some(), "split {s} range missing");
            }
            let cols = fmt.read_split(src, s, projection).expect("read_split");
            assert_eq!(cols.len(), projection.len());
            let n = cols.first().map_or(0, |c| c.len());
            for c in &cols {
                assert_eq!(c.len(), n, "split {s}: the column lengths do not agree");
            }
            for (i, c) in cols.iter().enumerate() {
                for r in 0..n {
                    out[i].push(c.value_at(r));
                }
            }
        }
        out
    }

    /// Resolves an in-memory CSV.
    fn open(bytes: &[u8], delim: u8) -> (CsvFormat, Source) {
        let src = Source::from_bytes(bytes.to_vec());
        let mut f = CsvFormat::new(delim);
        match f.resolve(&src).expect("resolve") {
            Ok(()) => {}
            Err(r) => panic!("an I/O request {r:?} from an in-memory source"),
        }
        (f, src)
    }

    fn open_split(bytes: &[u8], delim: u8, split_bytes: u64) -> (CsvFormat, Source) {
        let (mut f, src) = open(bytes, delim);
        f.split_bytes = split_bytes;
        (f, src)
    }

    fn names(f: &CsvFormat) -> Vec<&str> {
        f.schema().iter().map(|c| c.name.as_str()).collect()
    }

    fn types(f: &CsvFormat) -> Vec<Ty> {
        f.schema().iter().map(|c| c.ty).collect()
    }

    fn text(v: &Value) -> String {
        match v {
            Value::Bytes(b) => String::from_utf8_lossy(b).into_owned(),
            _ => panic!("not bytes"),
        }
    }

    // --- Headers ------------------------------------------------------------

    #[test]
    fn header_names_are_made_unique() {
        let (f, _) = open(b"a,,a,B,column2\n1,2,3,4,5\n", b',');
        // Empty -> column1; a duplicate (ignoring case) -> column2. The fifth column collides with
        // the "column2" generated before it, so it becomes column4.
        assert_eq!(names(&f), vec!["a", "column1", "column2", "B", "column4"]);
        // When generated names collide with one another, a `_` is added to make them unique.
        let (g, _) = open(b"x,column1,x\n1,2,3\n", b',');
        assert_eq!(names(&g), vec!["x", "column1", "column2"]);
        let (h, _) = open(b"column2,a,a\n1,2,3\n", b',');
        assert_eq!(names(&h), vec!["column2", "a", "column2_"]);
    }

    #[test]
    fn header_only_file_has_schema_but_no_splits() {
        let (f, _) = open(b"a,b,c\n", b',');
        assert_eq!(names(&f), vec!["a", "b", "c"]);
        assert_eq!(f.num_splits(), 0);
        assert!(f.is_resolved());
    }

    #[test]
    fn header_without_trailing_newline() {
        let (f, _) = open(b"a,b", b',');
        assert_eq!(names(&f), vec!["a", "b"]);
        assert_eq!(f.num_splits(), 0);
    }

    #[test]
    fn empty_file_resolves_to_empty_schema() {
        let (f, _) = open(b"", b',');
        assert!(f.is_resolved());
        assert!(f.schema().is_empty());
        assert_eq!(f.num_splits(), 0);
    }

    #[test]
    fn quoted_header_names() {
        let (f, _) = open(b"\"a,1\",\"b\"\"c\"\n1,2\n", b',');
        assert_eq!(names(&f), vec!["a,1", "b\"c"]);
    }

    #[test]
    fn tab_delimiter() {
        let (f, src) = open(b"a\tb\n1\tx\n", b'\t');
        assert_eq!(names(&f), vec!["a", "b"]);
        assert_eq!(types(&f), vec![Ty::BigInt, Ty::Varchar]);
        let got = read_all(&f, &src, &[0, 1]);
        assert_eq!(got[0], vec![Value::I64(1)]);
    }

    // --- BOM / CRLF ---------------------------------------------------------

    #[test]
    fn bom_is_stripped_from_first_column_name() {
        let (f, src) = open("\u{feff}id,name\n1,x\n".as_bytes(), b',');
        assert_eq!(names(&f), vec!["id", "name"]);
        let got = read_all(&f, &src, &[0, 1]);
        assert_eq!(got[0], vec![Value::I64(1)]);
    }

    #[test]
    fn crlf_line_endings() {
        let (f, src) = open(b"a,b\r\n1,x\r\n2,y\r\n", b',');
        assert_eq!(names(&f), vec!["a", "b"]);
        assert_eq!(types(&f), vec![Ty::BigInt, Ty::Varchar]);
        let got = read_all(&f, &src, &[0, 1]);
        assert_eq!(got[0], vec![Value::I64(1), Value::I64(2)]);
        // No `\r` remains in the value.
        assert_eq!(text(&got[1][0]), "x");
        assert_eq!(text(&got[1][1]), "y");
    }

    #[test]
    fn bom_with_crlf_and_split_boundaries() {
        let mut s = String::from("\u{feff}a,b\r\n");
        for i in 0..200 {
            s.push_str(&std::format!("{i},v{i}\r\n"));
        }
        let (f, src) = open_split(s.as_bytes(), b',', 16);
        assert!(f.num_splits() > 5);
        let got = read_all(&f, &src, &[0]);
        assert_eq!(got[0].len(), 200);
        assert_eq!(got[0][199], Value::I64(199));
    }

    #[test]
    fn blank_lines_are_skipped() {
        let (f, src) = open(b"a,b\n1,x\n\n2,y\n\n", b',');
        let got = read_all(&f, &src, &[0]);
        assert_eq!(got[0], vec![Value::I64(1), Value::I64(2)]);
    }

    // --- Type inference -----------------------------------------------------

    #[test]
    fn infers_each_type() {
        let csv = b"b,i,f,s,d,t\ntrue,1,1.5,x,2024-01-02,2024-01-02 03:04:05\n\
                    false,-2,-0.5e1,y,2000-02-29,1999-12-31T23:59:59.5\n";
        let (f, _) = open(csv, b',');
        assert_eq!(
            types(&f),
            vec![Ty::Boolean, Ty::BigInt, Ty::Double, Ty::Varchar, Ty::Date, Ty::Timestamp]
        );
    }

    #[test]
    fn widening_transitions() {
        // BIGINT -> DOUBLE -> VARCHAR, and DATE -> TIMESTAMP.
        let cases: &[(&[u8], Ty)] = &[
            (b"c\ntrue\nfalse\n", Ty::Boolean),
            // BOOLEAN mixed with a number is VARCHAR (duckdb agrees): `true` has no numeric
            // reading, so widening to the numeric type would drop the boolean rows.
            (b"c\ntrue\n1\n", Ty::Varchar),
            (b"c\ntrue\n1.5\n", Ty::Varchar),
            (b"c\ntrue\nx\n", Ty::Varchar),
            (b"c\n1\n2\n", Ty::BigInt),
            (b"c\n1\n2.5\n", Ty::Double),
            (b"c\n1\nx\n", Ty::Varchar),
            (b"c\n1.5\nx\n", Ty::Varchar),
            (b"c\n2024-01-01\n2024-01-02\n", Ty::Date),
            (b"c\n2024-01-01\n2024-01-02 00:00:00\n", Ty::Timestamp),
            (b"c\n2024-01-01\n1\n", Ty::Varchar),
            (b"c\n2024-01-01 00:00:00\nx\n", Ty::Varchar),
            // An integer that does not fit i64 falls to DOUBLE.
            (b"c\n99999999999999999999\n", Ty::Double),
            // All-empty stays VARCHAR.
            (b"c\n\n\n", Ty::Varchar),
            // A zero-padded digit run is text, not a number (duckdb agrees).
            (b"c\n007\n0123\n", Ty::Varchar),
            (b"c\n01\n02\n", Ty::Varchar),
            (b"c\n007.5\n1\n", Ty::Varchar),
            // ... but a lone `0` and a fraction below one are still numeric.
            (b"c\n0\n1\n", Ty::BigInt),
            (b"c\n0\n0.5\n", Ty::Double),
            (b"c\n0e3\n1\n", Ty::Double),
            // A zero-padded year must not be mistaken for a padded number.
            (b"c\n0001-02-03\n", Ty::Date),
        ];
        for (csv, ty) in cases {
            let (f, _) = open(csv, b',');
            assert_eq!(types(&f), vec![*ty], "{}", String::from_utf8_lossy(csv));
        }
    }

    #[test]
    fn inference_uses_only_the_first_rows() {
        // The first SAMPLE_ROWS rows are integers, with strings mixed in afterwards.
        let mut s = String::from("c\n");
        for i in 0..SAMPLE_ROWS {
            s.push_str(&std::format!("{i}\n"));
        }
        s.push_str("oops\n");
        let (f, src) = open(s.as_bytes(), b',');
        assert_eq!(types(&f), vec![Ty::BigInt]);
        // A row the inference missed is a clear error, not a silently dropped value.
        assert_eq!(code_of(f.read_split(&src, 0, &[0])), Some(Code::InvalidCast));
    }

    #[test]
    fn dates_and_timestamps_in_one_column_keep_both() {
        // duckdb reads the date-only row as that date's midnight rather than NULL.
        let (f, src) = open(b"c\n2024-01-01 10:00:00\n2024-01-02\n", b',');
        assert_eq!(types(&f), vec![Ty::Timestamp]);
        let got = read_all(&f, &src, &[0]);
        // duckdb: epoch_us(TIMESTAMP '2024-01-02') = 1704153600000000
        assert_eq!(got[0][1], Value::I64(1_704_153_600_000_000));
    }

    #[test]
    fn zero_padded_digits_keep_their_padding() {
        let (f, src) = open(b"a\n007\n0123\n", b',');
        assert_eq!(types(&f), vec![Ty::Varchar]);
        let got = read_all(&f, &src, &[0]);
        assert_eq!(text(&got[0][0]), "007");
        assert_eq!(text(&got[0][1]), "0123");
    }

    #[test]
    fn booleans_mixed_with_numbers_keep_both() {
        let (f, src) = open(b"a\ntrue\n1\n", b',');
        assert_eq!(types(&f), vec![Ty::Varchar]);
        let got = read_all(&f, &src, &[0]);
        assert_eq!(text(&got[0][0]), "true");
        assert_eq!(text(&got[0][1]), "1");
    }

    #[test]
    fn value_parsers() {
        assert_eq!(parse_i64(b"-9223372036854775808"), Some(i64::MIN));
        assert_eq!(parse_i64(b"9223372036854775808"), None);
        assert_eq!(parse_i64(b"+7"), Some(7));
        assert_eq!(parse_i64(b""), None);
        assert_eq!(parse_i64(b"-"), None);
        assert_eq!(parse_i64(b"1 "), None);
        assert_eq!(parse_f64(b"1.5"), Some(1.5));
        assert_eq!(parse_f64(b"-.5"), Some(-0.5));
        assert_eq!(parse_f64(b"1e3"), Some(1000.0));
        // Both spellings of the infinities read back, whichever wrote the file: the CSV
        // writer emits `inf` / `-inf`, older files and other tools the long form.
        assert!(parse_f64(b"inf").is_some_and(|f| f.is_infinite() && f > 0.0));
        assert!(parse_f64(b"-inf").is_some_and(|f| f.is_infinite() && f < 0.0));
        assert!(parse_f64(b"Infinity").is_some_and(|f| f.is_infinite() && f > 0.0));
        assert!(parse_f64(b"-Infinity").is_some_and(|f| f.is_infinite() && f < 0.0));
        assert!(parse_f64(b"NaN").is_some_and(|f| f.is_nan()));
        // ... and a column of them is inferred as DOUBLE either way, so a re-read of an
        // exported file keeps its type.
        assert_eq!(classify(b"inf"), Cand::Double);
        assert_eq!(classify(b"-inf"), Cand::Double);
        assert_eq!(classify(b"Infinity"), Cand::Double);
        assert_eq!(parse_f64(b"1e"), None);
        assert_eq!(parse_f64(b"0x10"), None);
        // duckdb: datediff('day', DATE '1970-01-01', DATE '2024-01-01') = 19723
        assert_eq!(parse_date(b"2024-01-01"), Some(19723));
        assert_eq!(parse_date(b"1900-03-01"), Some(-25508));
        assert_eq!(parse_date(b"2023-02-29"), None);
        assert_eq!(parse_date(b"2024-02-29"), Some(19782));
        assert_eq!(parse_date(b"2024-1-1"), None);
        // duckdb: epoch_us(TIMESTAMP '2024-01-01 00:00:00') = 1704067200000000
        assert_eq!(parse_ts(b"2024-01-01 00:00:00"), Some(1_704_067_200_000_000));
        assert_eq!(parse_ts(b"2024-01-01T00:00:00"), Some(1_704_067_200_000_000));
        assert_eq!(parse_ts(b"1969-12-31 23:59:59.999999"), Some(-1));
        // The 7th digit onward is truncated.
        assert_eq!(parse_ts(b"1970-01-01 00:00:00.1234567"), Some(123_456));
        assert_eq!(parse_ts(b"2024-01-01 24:00:00"), None);
        assert_eq!(parse_ts(b"2024-01-01 00:00:00Z"), Some(1_704_067_200_000_000));
        assert_eq!(parse_ts(b"2024-01-01 00:00:00+00"), Some(1_704_067_200_000_000));
        // 09:00+09 is 00:00 UTC.
        assert_eq!(parse_ts(b"2024-01-01 09:00:00+09:00"), Some(1_704_067_200_000_000));
        // A date-only value is that date's midnight (duckdb: TIMESTAMP '2024-01-01').
        assert_eq!(parse_ts(b"2024-01-01"), Some(1_704_067_200_000_000));
        assert_eq!(parse_ts(b"1969-12-31"), Some(-86_400_000_000));
        assert_eq!(parse_ts(b"2023-02-29"), None);
        assert_eq!(parse_ts(b"2024-01-01 "), None);

        assert!(is_zero_padded(b"007"));
        assert!(is_zero_padded(b"0123"));
        assert!(is_zero_padded(b"007.5"));
        assert!(is_zero_padded(b"-007"));
        assert!(!is_zero_padded(b"0"));
        assert!(!is_zero_padded(b"0.5"));
        assert!(!is_zero_padded(b"0e3"));
        assert!(!is_zero_padded(b""));
        // The permissive parsers are untouched: an explicitly numeric column still reads `007`
        // as 7, exactly as duckdb's `'007'::BIGINT` does.
        assert_eq!(parse_i64(b"007"), Some(7));
        assert_eq!(parse_f64(b"007.5"), Some(7.5));
    }

    // --- Quotes -------------------------------------------------------------

    #[test]
    fn quoted_file() {
        let bytes = read_data("quoted.csv");
        let (f, src) = open(&bytes, b',');
        assert_eq!(names(&f), vec!["a", "b", "c"]);
        assert_eq!(types(&f), vec![Ty::BigInt, Ty::Varchar, Ty::Double]);
        let got = read_all(&f, &src, &[0, 1, 2]);
        assert_eq!(got[0], vec![Value::I64(1), Value::I64(2), Value::I64(3)]);
        assert_eq!(text(&got[1][0]), "he said \"hi\"");
        // A newline inside quotes is not a line terminator.
        assert_eq!(text(&got[1][1]), "multi\nline");
        assert_eq!(got[1][2], Value::Null);
        assert_eq!(got[2], vec![Value::F64(2.5), Value::F64(3.5), Value::F64(4.5)]);
    }

    #[test]
    fn quoted_fields_contain_delimiter_and_crlf() {
        let (f, src) = open(b"a,b\n\"x,y\",\"p\r\nq\"\n", b',');
        let got = read_all(&f, &src, &[0, 1]);
        assert_eq!(text(&got[0][0]), "x,y");
        assert_eq!(text(&got[1][0]), "p\r\nq");
    }

    #[test]
    fn empty_versus_quoted_empty() {
        let (f, src) = open(b"a,b\n,\"\"\n", b',');
        let got = read_all(&f, &src, &[0, 1]);
        // An unquoted empty is NULL and `""` is the empty string.
        assert_eq!(got[0][0], Value::Null);
        assert_eq!(got[1][0], Value::Bytes(vec![]));
    }

    #[test]
    fn quoted_numbers_still_infer_numeric() {
        let (f, src) = open(b"a\n\"1\"\n\"2\"\n", b',');
        assert_eq!(types(&f), vec![Ty::BigInt]);
        let got = read_all(&f, &src, &[0]);
        assert_eq!(got[0], vec![Value::I64(1), Value::I64(2)]);
    }

    // --- Splits -------------------------------------------------------------

    #[test]
    fn split_ranges_ask_for_the_overhang() {
        let mut s = String::from("a\n");
        for i in 0..1000 {
            s.push_str(&std::format!("{i}\n"));
        }
        let (mut f, _) = open(s.as_bytes(), b',');
        f.split_bytes = 100;
        f.max_record = 50;
        let n = f.num_splits();
        assert!(n > 5);
        let mut r = Vec::new();
        f.split_ranges(1, &[0], &mut r).unwrap();
        let (a, b) = r[0];
        let ds = f.data_start;
        // It includes the preceding byte and overreads max_record on the back.
        assert_eq!(a, ds + 100 - 1);
        assert_eq!(b, ds + 200 + 50);
        // Changing the projection does not change the range (row-oriented, so it cannot be reduced).
        let mut r2 = Vec::new();
        f.split_ranges(1, &[], &mut r2).unwrap();
        assert_eq!(r2[0], (a, b));
    }

    #[test]
    fn single_split_reads_basic_csv() {
        let bytes = read_data("basic.csv");
        let (f, src) = open(&bytes, b',');
        assert_eq!(f.num_splits(), 1, "it fits in one split at 8 MiB per split");
        assert_eq!(names(&f), vec!["id", "name", "score", "flag", "big", "d"]);
        // It agrees with duckdb's DESCRIBE (id is inferred as BIGINT).
        assert_eq!(
            types(&f),
            vec![Ty::BigInt, Ty::Varchar, Ty::Double, Ty::Boolean, Ty::BigInt, Ty::Timestamp]
        );
        let got = read_all(&f, &src, &[0, 4]);
        assert_eq!(got[0].len(), 1000);
        // duckdb: count(big) = 800, with a NULL every five rows.
        let nulls = got[1].iter().filter(|v| **v == Value::Null).count();
        assert_eq!(nulls, 200);
    }

    #[test]
    fn basic_csv_matches_the_parquet_file() {
        // The Parquet side is read through `TableFormat` too. Being able to compare both through
        // the same entry point is the point of this abstraction, so the test takes that shape.
        let psrc = Source::from_bytes(read_data("basic.parquet"));
        let mut pf = crate::format::parquet::ParquetFormat::new();
        assert_eq!(pf.resolve(&psrc).expect("parquet resolve"), Ok(()));
        let proj: Vec<usize> = (0..pf.schema().len()).collect();
        let mut expect: Vec<Vec<Value>> = proj.iter().map(|_| Vec::new()).collect();
        for s in 0..pf.num_splits() {
            let cols = pf.read_split(&psrc, s, &proj).expect("parquet split");
            for (i, c) in cols.iter().enumerate() {
                for r in 0..c.len() {
                    expect[i].push(c.value_at(r));
                }
            }
        }

        let bytes = read_data("basic.csv");
        // It must agree across splits too (detecting rows dropped at boundaries).
        let (f, src) = open_split(&bytes, b',', 4096);
        assert!(f.num_splits() > 5);
        let got = read_all(&f, &src, &[0, 1, 2, 3, 4, 5]);

        for c in 0..6 {
            assert_eq!(got[c].len(), 1000, "the row count of column {c}");
            for r in 0..1000 {
                // The Parquet side's id is INTEGER, so it is aligned to i64 for comparison.
                let e = match &expect[c][r] {
                    Value::I32(x) if c == 0 => Value::I64(*x as i64),
                    v => v.clone(),
                };
                assert_eq!(got[c][r], e, "column {c} row {r}");
            }
        }
    }

    #[test]
    fn multi_split_row_counts_and_boundary_values() {
        let bytes = read_data("multi_rg.csv");
        // At the default split size it is one split, so it is shrunk to create boundaries.
        let (f, src) = open_split(&bytes, b',', 64 * 1024);
        assert!(f.num_splits() >= 16, "split count {}", f.num_splits());

        // The per-split row counts must add up to the whole, with no boundary row missing or duplicated.
        let mut total = 0usize;
        let mut next = 0i64;
        for s in 0..f.num_splits() {
            let cols = f.read_split(&src, s, &[0, 3]).expect("read_split");
            let n = cols[0].len();
            for r in 0..n {
                // id runs consecutively from 0. Any deviation is a boundary-handling error.
                assert_eq!(cols[0].value_at(r), Value::I64(next), "split {s} row {r}");
                assert_eq!(cols[1].value_at(r), Value::F64(next as f64 * 0.25));
                next += 1;
            }
            total += n;
        }
        // duckdb: count(*) = 50000
        assert_eq!(total, 50_000);
    }

    #[test]
    fn split_size_does_not_change_the_result() {
        let bytes = read_data("multi_rg.csv");
        let (f1, s1) = open_split(&bytes, b',', 1 << 20);
        let (f2, s2) = open_split(&bytes, b',', 997);
        let a = read_all(&f1, &s1, &[0, 1, 2, 3]);
        let b = read_all(&f2, &s2, &[0, 1, 2, 3]);
        assert!(f2.num_splits() > f1.num_splits());
        for c in 0..4 {
            assert_eq!(a[c].len(), 50_000);
            assert_eq!(a[c], b[c], "column {c}");
        }
    }

    #[test]
    fn records_straddling_every_boundary() {
        // With 8-byte records and 8-byte splits, boundaries coincide with record starts.
        // Without honoring the half-open ownership rule, one row would vanish at each boundary.
        let mut s = String::from("a,b\n");
        for i in 0..64 {
            s.push_str(&std::format!("{i:03},{i:03}\n"));
        }
        for size in 1..=17u64 {
            let (f, src) = open_split(s.as_bytes(), b',', size);
            // The values are zero padded so every record is the same length; that also makes the
            // column VARCHAR (`is_zero_padded`), which is beside the point of this test.
            assert_eq!(types(&f), vec![Ty::Varchar, Ty::Varchar]);
            let got = read_all(&f, &src, &[0]);
            assert_eq!(got[0].len(), 64, "split_bytes={size}");
            for (i, v) in got[0].iter().enumerate() {
                assert_eq!(text(v), std::format!("{i:03}"), "split_bytes={size}");
            }
        }
    }

    #[test]
    fn quoted_sample_is_detected_from_the_leading_bytes() {
        let (plain, _) = open(b"a,b\n1,x\n2,y\n", b',');
        assert!(!plain.quoted_sample);
        let (quoted, _) = open(b"a,b\n1,\"x\"\n2,y\n", b',');
        assert!(quoted.quoted_sample);
        // A quote in the header alone is enough (quoting could start on any row).
        let (quoted_header, _) = open(b"\"a\",b\n1,x\n", b',');
        assert!(quoted_header.quoted_sample);
    }

    #[test]
    fn quoted_newline_straddling_a_split_boundary_forces_a_single_split() {
        // Regression test: a quoted field whose embedded `\n` used to land exactly on a split
        // boundary. Before the fix, `read_split`'s "resync at the first raw `\n`" scan could not
        // tell that newline was inside an open quote, so the trailing half of the quoted value
        // was misread as a new (bogus) record -- typically surfacing as an extra `(NULL, NULL)`
        // row and one row too many overall.
        let bytes = b"a,b\n1,\"multi\nline\"\n2,plain\n3,\"another\nembedded\nnewline\"\n4,end\n";
        // split_bytes is set absurdly small so that, without the fix, several split boundaries
        // would fall inside the quoted values above.
        let (f, src) = open_split(bytes, b',', 8);
        assert!(f.quoted_sample);
        assert_eq!(f.num_splits(), 1, "a quoted file must always resolve to exactly one split");

        let got = read_all(&f, &src, &[0, 1]);
        // Exactly 4 rows -- no phantom row from a misinterpreted embedded newline.
        assert_eq!(got[0], vec![Value::I64(1), Value::I64(2), Value::I64(3), Value::I64(4)]);
        assert!(!got[0].contains(&Value::Null), "no phantom NULL row");
        assert_eq!(text(&got[1][0]), "multi\nline");
        assert_eq!(text(&got[1][1]), "plain");
        assert_eq!(text(&got[1][2]), "another\nembedded\nnewline");
        assert_eq!(text(&got[1][3]), "end");
    }

    #[test]
    fn quote_after_the_sample_on_a_complete_source_still_forces_a_single_split() {
        // Residual gap that used to drop rows: the leading SAMPLE_BYTES had no `"`,
        // so the file was split, and a quoted embedded newline on a later boundary
        // was treated as a record terminator. A fully-resident source now scans
        // past the sample, so quoting still collapses to one split.
        let mut s = String::from("a,b\n");
        let mut pad_rows = 0usize;
        while s.len() < SAMPLE_BYTES as usize + 64 {
            // Keep column `b` VARCHAR so the late quoted value is not
            // coerced to NULL by integer inference on the leading sample.
            s.push_str("0,a\n");
            pad_rows += 1;
        }
        s.push_str("1,\"x\ny\"\n2,z\n");
        let (f, src) = open_split(s.as_bytes(), b',', 8192);
        assert!(f.quoted_sample);
        assert_eq!(f.num_splits(), 1);
        let got = read_all(&f, &src, &[0, 1]);
        assert_eq!(got[0].len(), pad_rows + 2);
        assert_eq!(got[0][pad_rows], Value::I64(1));
        assert_eq!(text(&got[1][pad_rows]), "x\ny");
        assert_eq!(got[0][pad_rows + 1], Value::I64(2));
        assert_eq!(text(&got[1][pad_rows + 1]), "z");
    }

    #[test]
    fn quote_in_a_later_split_of_an_unquoted_sample_is_rejected() {
        // Remote resolve only sees the sample. A `"` in a later split must not be
        // parsed as unquoted CSV (wrong row count); it is UnsupportedFeature.
        let mut bytes = b"a,b\n".to_vec();
        while bytes.len() < SAMPLE_BYTES as usize + 64 {
            bytes.extend_from_slice(b"0,0\n");
        }
        bytes.extend_from_slice(b"1,\"x\ny\"\n");
        let mut src = Source::remote(bytes.len() as u64);
        let sample_n = SAMPLE_BYTES.min(bytes.len() as u64) as usize;
        src.insert(0, bytes[..sample_n].to_vec());
        let mut f = CsvFormat::new(b',');
        f.split_bytes = 8192;
        match f.resolve(&src).expect("resolve") {
            Ok(()) => {}
            Err(_) => panic!("sample is present"),
        }
        assert!(!f.quoted_sample);
        assert!(f.num_splits() > 1);
        src.insert(sample_n as u64, bytes[sample_n..].to_vec());
        let saw = (1..f.num_splits())
            .any(|i| code_of(f.read_split(&src, i, &[0, 1])) == Some(Code::UnsupportedFeature));
        assert!(saw, "a later split containing a quote must be rejected");
    }

    #[test]
    fn last_record_without_trailing_newline() {
        let (f, src) = open_split(b"a\n1\n2\n3", b',', 2);
        let got = read_all(&f, &src, &[0]);
        assert_eq!(got[0], vec![Value::I64(1), Value::I64(2), Value::I64(3)]);
    }

    #[test]
    fn record_longer_than_max_record_is_rejected() {
        let mut s = String::from("a,b\n");
        s.push_str("1,");
        for _ in 0..200 {
            s.push('x');
        }
        s.push('\n');
        s.push_str("2,y\n");
        let (mut f, src) = open(s.as_bytes(), b',');
        f.split_bytes = 8;
        f.max_record = 16;
        // The split that runs into a long record returns LimitExceeded.
        let bad =
            (0..f.num_splits()).filter_map(|s| code_of(f.read_split(&src, s, &[0, 1]))).next();
        assert_eq!(bad, Some(Code::LimitExceeded));
    }

    // --- Projection ---------------------------------------------------------

    #[test]
    fn projection_returns_requested_columns_in_order() {
        let (f, src) = open(b"a,b,c\n1,x,2.5\n3,y,4.5\n", b',');
        let got = read_all(&f, &src, &[2, 0]);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0], vec![Value::F64(2.5), Value::F64(4.5)]);
        assert_eq!(got[1], vec![Value::I64(1), Value::I64(3)]);
        // An empty projection returns 0 columns too (the caller has no way to know the row count).
        let cols = f.read_split(&src, 0, &[]).unwrap();
        assert!(cols.is_empty());
        // Requesting the same column twice still returns columns of equal length.
        let dup = read_all(&f, &src, &[1, 1]);
        assert_eq!(dup[0], dup[1]);
        assert_eq!(text(&dup[0][0]), "x");
        // An out-of-range column is an internal error.
        assert_eq!(code_of(f.read_split(&src, 0, &[9])), Some(Code::Internal));
    }

    // --- Anomalous input ----------------------------------------------------

    #[test]
    fn unterminated_quote_is_a_syntax_error() {
        let (f, src) = open(b"a,b\n1,\"oops\n", b',');
        assert_eq!(code_of(f.read_split(&src, 0, &[0, 1])), Some(Code::SyntaxError));
    }

    #[test]
    fn short_and_long_rows() {
        // Missing fields are NULL and surplus ones are discarded.
        let (f, src) = open(b"a,b,c\n1,x,2.5\n2\n3,y,4.5,extra\n", b',');
        let got = read_all(&f, &src, &[0, 1, 2]);
        assert_eq!(got[0], vec![Value::I64(1), Value::I64(2), Value::I64(3)]);
        assert_eq!(got[1][1], Value::Null);
        assert_eq!(got[2][1], Value::Null);
        assert_eq!(got[2][2], Value::F64(4.5));
    }

    #[test]
    fn values_that_do_not_match_the_inferred_type_are_an_error() {
        // Inference sees only the sample rows, so the rows that fall outside are placed after the
        // sample. Such a value used to be turned into NULL, silently dropping data the file
        // plainly contains; duckdb reports it as a conversion error and so does this.
        let head = "i,b,d,t\n";
        let row = "1,true,2024-01-01,2024-01-01 00:00:00\n";
        for col in 0..4 {
            let mut s = String::from(head);
            for _ in 0..SAMPLE_ROWS {
                s.push_str(row);
            }
            // One bad cell in one column at a time.
            let bad: Vec<&str> = (0..4).map(|c| if c == col { "x" } else { "1" }).collect();
            s.push_str(&bad.join(","));
            s.push('\n');
            let (f, src) = open(s.as_bytes(), b',');
            assert_eq!(types(&f), vec![Ty::BigInt, Ty::Boolean, Ty::Date, Ty::Timestamp]);
            assert_eq!(
                code_of(f.read_split(&src, 0, &[col])),
                Some(Code::InvalidCast),
                "column {col}"
            );
        }
    }

    #[test]
    fn empty_cells_stay_null_in_every_type() {
        // The one thing CSV can spell is NULL, and it must keep working: an unquoted empty field
        // is NULL, and so is `""` in a column that is not VARCHAR (there is no empty BIGINT).
        let mut s = String::from("i,b,d,t\n");
        for _ in 0..4 {
            s.push_str("1,true,2024-01-01,2024-01-01 00:00:00\n");
        }
        s.push_str(",,,\n");
        s.push_str("\"\",\"\",\"\",\"\"\n");
        let (f, src) = open(s.as_bytes(), b',');
        assert_eq!(types(&f), vec![Ty::BigInt, Ty::Boolean, Ty::Date, Ty::Timestamp]);
        let got = read_all(&f, &src, &[0, 1, 2, 3]);
        for (c, col) in got.iter().enumerate() {
            assert_ne!(col[0], Value::Null, "column {c}");
            assert_eq!(col[4], Value::Null, "column {c}");
            assert_eq!(col[5], Value::Null, "column {c}");
        }
    }

    #[test]
    fn cr_only_line_endings_are_read_as_records() {
        // A classic Mac (CR-only) file used to parse as one giant record: a garbage header and
        // zero rows, with no error at all. duckdb reads two rows here.
        let (f, src) = open(b"a,b\r1,2\r3,4\r", b',');
        assert!(f.cr_only);
        assert_eq!(names(&f), vec!["a", "b"]);
        assert_eq!(types(&f), vec![Ty::BigInt, Ty::BigInt]);
        assert_eq!(f.num_splits(), 1, "a CR-only file is always a single split");
        let got = read_all(&f, &src, &[0, 1]);
        assert_eq!(got[0], vec![Value::I64(1), Value::I64(3)]);
        assert_eq!(got[1], vec![Value::I64(2), Value::I64(4)]);
    }

    #[test]
    fn a_lone_cr_stays_data_when_the_file_has_newlines() {
        // CRLF and a `\r` inside a quoted field must be untouched by the CR-only support.
        let (crlf, csrc) = open(b"a,b\r\n1,x\r\n2,y\r\n", b',');
        assert!(!crlf.cr_only);
        let got = read_all(&crlf, &csrc, &[0, 1]);
        assert_eq!(got[0], vec![Value::I64(1), Value::I64(2)]);
        assert_eq!(text(&got[1][0]), "x");

        // An unquoted `\r` in a `\n`-terminated file is still an ordinary byte.
        let (f, src) = open(b"a,b\n1,x\ry\n", b',');
        assert!(!f.cr_only);
        let got = read_all(&f, &src, &[0, 1]);
        assert_eq!(got[0], vec![Value::I64(1)]);
        assert_eq!(text(&got[1][0]), "x\ry");

        // A quoted `\r` in a CR-only file belongs to the value, not to the record boundary.
        let (q, qsrc) = open(b"a,b\r1,\"x\ry\"\r2,z\r", b',');
        assert!(q.cr_only);
        let got = read_all(&q, &qsrc, &[0, 1]);
        assert_eq!(got[0], vec![Value::I64(1), Value::I64(2)]);
        assert_eq!(text(&got[1][0]), "x\ry");
        assert_eq!(text(&got[1][1]), "z");
    }

    #[test]
    fn garbage_after_a_closing_quote_is_dropped() {
        let (f, src) = open(b"a,b\n\"x\"junk,1\n", b',');
        let got = read_all(&f, &src, &[0, 1]);
        assert_eq!(text(&got[0][0]), "x");
        assert_eq!(got[1][0], Value::I64(1));
    }

    #[test]
    fn binary_garbage_does_not_panic() {
        let mut bytes = std::vec![b'a', b',', b'b', b'\n'];
        for i in 0u32..4096 {
            bytes.push((i.wrapping_mul(2654435761) >> 13) as u8);
        }
        let src = Source::from_bytes(bytes);
        let mut f = CsvFormat::new(b',');
        if f.resolve(&src).is_ok() && f.is_resolved() {
            f.split_bytes = 64;
            for s in 0..f.num_splits() {
                let _ = f.read_split(&src, s, &[0, 1]);
            }
        }
    }

    // --- Resolution I/O -----------------------------------------------------

    #[test]
    fn resolve_requests_the_sample_range_first() {
        let mut f = CsvFormat::new(b',');
        let src = Source::remote(10 * 1024 * 1024);
        match f.resolve(&src).unwrap() {
            Err((off, len)) => {
                assert_eq!(off, 0);
                assert_eq!(len, SAMPLE_BYTES);
            }
            Ok(()) => panic!("it cannot possibly resolve with no bytes"),
        }
        assert!(!f.is_resolved());
        assert_eq!(f.num_splits(), 0);
        assert!(f.schema().is_empty());
    }

    #[test]
    fn resolve_requests_only_the_file_when_shorter_than_the_sample() {
        let mut f = CsvFormat::new(b',');
        let src = Source::remote(10);
        assert_eq!(f.resolve(&src).unwrap(), Err((0, 10)));
    }

    #[test]
    fn resolve_progresses_once_the_sample_arrives() {
        let body = b"a,b\n1,x\n";
        let mut src = Source::remote(body.len() as u64);
        let mut f = CsvFormat::new(b',');
        let (off, len) = f.resolve(&src).unwrap().expect_err("a request the first time");
        src.insert(off, body[off as usize..(off + len) as usize].to_vec());
        assert_eq!(f.resolve(&src).unwrap(), Ok(()));
        assert_eq!(names(&f), vec!["a", "b"]);
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn resolve_rejects_a_split_count_that_does_not_fit_usize() {
        let mut f = CsvFormat::new(b',');
        f.split_bytes = 1;
        let mut src = Source::remote(u64::MAX);
        let mut sample = vec![0u8; SAMPLE_BYTES as usize];
        sample[..2].copy_from_slice(b"a\n");
        src.insert(0, sample);
        assert_eq!(code_of(f.resolve(&src)), Some(Code::LimitExceeded));
    }

    #[test]
    fn header_longer_than_the_sample_is_rejected() {
        // A file exceeding SAMPLE_BYTES with no newline in the header.
        let mut bytes = std::vec![b'a'; (SAMPLE_BYTES + 10) as usize];
        bytes[SAMPLE_BYTES as usize + 5] = b'\n';
        let src = Source::from_bytes(bytes);
        let mut f = CsvFormat::new(b',');
        assert_eq!(code_of(f.resolve(&src)), Some(Code::LimitExceeded));
    }
}
