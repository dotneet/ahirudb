//! The table format abstraction.
//!
//! The execution engine sees data sources only through this trait. Parquet-specific concepts
//! (RowGroups, column chunks, Thrift statistics) stay inside `format::parquet`.
//!
//! ## The unit called a split
//!
//! The key to absorbing format differences is concentrating them into the single notion of a split.
//!
//! | format | what a split is | statistics | bytes saved by projection |
//! |---|---|---|---|
//! | Parquet | a RowGroup | yes | yes (fetched per column chunk) |
//! | CSV / JSONL | a fixed-length byte chunk | no | no (row-oriented, so everything is read) |
//!
//! DESIGN.md §6's "RowGroup boundary I/O barrier" is more precisely a **split boundary** barrier.
//! The only requirement is that the needed byte ranges are settled at the start of a split;
//! being Parquet is not a requirement. That is why the same execution loop works for CSV.
//!
//! ## Why projection handling is split into two stages
//!
//! The projection is passed to `split_ranges` because a columnar format can **reduce the bytes
//! fetched** with it. A row-oriented format has to read every byte regardless, but `read_split`
//! can skip converting unneeded columns. Expressing these two stages with one argument keeps the
//! caller from having to know the format's nature.
//!
//! ## Why page-level filtering is a separate hook
//!
//! `may_match` (RowGroup statistics) can decide with no extra I/O, but page-level filtering
//! (Parquet's ColumnIndex/OffsetIndex/Bloom filter) itself requires fetching another byte range.
//! `split_ranges`'s inputs (the projection and pruners) alone cannot settle "which bytes are
//! needed", so an extra round trip is required: `index_ranges` -> I/O ->
//! `refine_with_index` (decode and pick pages) -> `split_ranges`
//! (returning the byte ranges of only the chosen pages).
//! The default implementation is "no such bytes", so formats without support (CSV/JSONL) keep
//! working exactly as before -- "a split = one I/O decision" -- without a single line changed.

pub mod parquet;
pub mod partitioned;

#[cfg(feature = "csv")]
pub mod csv;

#[cfg(feature = "jsonl")]
pub mod jsonl;

// A file whose top level is a single JSON array/object. It is a close relative of JSONL, so rather
// than adding a dedicated Cargo feature it rides `jsonl`.
#[cfg(feature = "jsonl")]
pub mod json;

use crate::catalog::Source;
use crate::prelude::*;
use crate::rt::hash::eq_ascii_ci;
use crate::vector::{Field, Value, Vector};

/// A simple range predicate usable for statistics-based pruning.
///
/// It holds only the `column <op> constant` shapes extractable from `WHERE`.
/// `column` refers to the **post-projection column number** (= the position in the scan's output).
/// `Clone` is needed for cloning `Node::Aggregate` (and hence `Node`)
/// (GROUPING SETS duplicates the input plan once per grouping set).
#[derive(Clone)]
pub struct Pruner {
    pub column: usize,
    pub op: PruneOp,
    pub value: Value,
    /// The remaining candidate values, used only for `PruneOp::In` (`value` holds the first one).
    /// Always empty for other operators.
    pub in_values: Vec<Value>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum PruneOp {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
    /// `column IN (constant, ...)`. It suffices to equal any of `value` + `in_values`
    /// (= several `Eq`s bundled with OR).
    In,
}

/// Whether the statistics' `[min, max]` could satisfy this predicate. If false, the whole split can be skipped.
///
/// When it cannot decide it always returns `true` (erring toward reading). A pruning mistake
/// breaks things in the worst way -- rows vanish -- so the safe definition is pinned in one place.
pub fn range_may_match(p: &Pruner, min: &Value, max: &Value) -> bool {
    use core::cmp::Ordering::*;
    // If stats are inverted (min > max, e.g. corrupted metadata), err safe and do not prune.
    if min.partial_cmp_same(max) == Some(Greater) {
        return true;
    }
    if p.op == PruneOp::In {
        // OR semantics: if any one candidate could fall in the range, this split is kept.
        return core::iter::once(&p.value).chain(p.in_values.iter()).any(|v| {
            match (min.partial_cmp_same(v), max.partial_cmp_same(v)) {
                (Some(cmp_min), Some(cmp_max)) => cmp_min != Greater && cmp_max != Less,
                // A candidate that cannot be compared errs safe (treated as possible).
                _ => true,
            }
        });
    }
    let (cmp_min, cmp_max) = match (min.partial_cmp_same(&p.value), max.partial_cmp_same(&p.value))
    {
        (Some(a), Some(b)) => (a, b),
        // If it cannot be compared (different types, NULL) there is no pruning.
        _ => return true,
    };
    match p.op {
        PruneOp::Eq => cmp_min != Greater && cmp_max != Less,
        PruneOp::Lt => cmp_min == Less,
        PruneOp::Le => cmp_min != Greater,
        PruneOp::Gt => cmp_max == Greater,
        PruneOp::Ge => cmp_max != Less,
        PruneOp::In => unreachable!(),
    }
}

/// The result of schema resolution. When bytes are missing it returns the ranges needed.
pub type ResolveStep = core::result::Result<(), (u64, u64)>;

/// A compressed block whose decompression is delegated to the host.
///
/// Codecs the wasm core does not carry (GZIP / ZSTD) are decompressed on the host side
/// (DESIGN.md §6). GZIP is handled by the browser's `DecompressionStream` and ZSTD by a separate
/// wasm module. From the engine's point of view both are the same "work asked of the host", so
/// they are unified into one path.
#[derive(Clone, Copy)]
pub struct CodecTask {
    pub codec: crate::parquet::Compression,
    /// The compressed data's position and length in the file. Also the cache key.
    pub offset: u64,
    pub len: u32,
    /// The decompressed size. The host must not return output exceeding it.
    pub out_len: u32,
}

/// Format-independent table reading.
pub trait TableFormat {
    /// Resolves the schema. It performs no I/O and returns after requesting the ranges it lacks.
    /// When called again with the same ranges satisfied, it must make progress.
    fn resolve(&mut self, src: &Source) -> Result<ResolveStep>;

    fn is_resolved(&self) -> bool;

    /// The resolved columns. Empty until `resolve` returns `Ok(Ok(()))`.
    fn schema(&self) -> &[Field];

    /// The total number of scan splits.
    fn num_splits(&self) -> usize;

    /// A split's row count. `None` for formats where it is not known in advance.
    /// It is used only for join-order estimation and progress display, so accuracy does not matter.
    fn split_rows(&self, split: usize) -> Option<u64>;

    /// Pushes into `out` the byte ranges needed to read a split.
    ///
    /// `projection` holds schema column numbers. A columnar format uses it to narrow the fetched
    /// ranges. A row-oriented format may ignore it.
    fn split_ranges(
        &self,
        split: usize,
        projection: &[usize],
        out: &mut Vec<(u64, u64)>,
    ) -> Result<()>;

    /// Enumerates the host-side decompression work needed to decode this split.
    ///
    /// Called **after** the bytes `split_ranges` named are present. Page headers are uncompressed,
    /// so all the necessary work can be settled at this point. That property is why execution never
    /// has to stop midway (DESIGN.md §6).
    ///
    /// Formats using only built-in codecs can leave the default implementation.
    fn codec_tasks(
        &self,
        _src: &Source,
        _split: usize,
        _projection: &[usize],
        _out: &mut Vec<CodecTask>,
    ) -> Result<()> {
        Ok(())
    }

    /// Whether statistics can rule this split out. Formats without statistics can leave the default
    /// implementation returning `true`.
    ///
    /// `pruners`' `column` refers to a position within `projection`.
    fn may_match(&self, _split: usize, _pruners: &[Pruner], _projection: &[usize]) -> bool {
        true
    }

    /// Pushes into `out` the byte ranges used for page-level filtering
    /// (ColumnIndex/OffsetIndex/Bloom filter). A hook for one extra round trip before
    /// `split_ranges`, for when `may_match` (RowGroup statistics, no extra I/O) cannot decide on
    /// its own.
    ///
    /// Formats with no such data (unsupported, or where the pruners are unusable) can leave the
    /// default implementation and push nothing. With nothing pushed, the subsequent
    /// `refine_with_index` stays at its default and merely returns `true`, matching the earlier
    /// behavior of "page-level filtering disabled" exactly.
    fn index_ranges(
        &self,
        _split: usize,
        _pruners: &[Pruner],
        _projection: &[usize],
        _out: &mut Vec<(u64, u64)>,
    ) -> Result<()> {
        Ok(())
    }

    /// Called after the bytes `index_ranges` requested are present. The page selection result
    /// (which pages to read) may be cached internally (`&mut self`).
    ///
    /// A return value of `false` means the whole split can be skipped
    /// (a Bloom filter negative, or no page can match by statistics).
    /// The default implementation always returns `true` (no filtering = fetching whole column chunks as before).
    fn refine_with_index(
        &mut self,
        _src: &Source,
        _split: usize,
        _pruners: &[Pruner],
        _projection: &[usize],
    ) -> Result<bool> {
        Ok(true)
    }

    /// Decodes a split and returns the column vectors.
    ///
    /// The returned columns must be in the same order and of the same count as `projection`, and
    /// all of the same length. The caller guarantees the ranges `split_ranges` named are present in `src`.
    fn read_split(&self, src: &Source, split: usize, projection: &[usize]) -> Result<Vec<Vector>>;
}

/// Extracts `[start, end)` from `src`. The caller is contracted to have the ranges (named by
/// `split_ranges`/`index_ranges`) already present in `src`, so their absence is a caller contract
/// violation and gives `Internal` (rather than waiting on I/O).
///
/// A small helper shared by every format's `read_split`/`codec_tasks`.
pub(crate) fn get_or_internal(src: &Source, start: u64, end: u64) -> Result<&[u8]> {
    let len = match end.checked_sub(start).and_then(|n| usize::try_from(n).ok()) {
        Some(len) => len,
        None => err!(Internal),
    };
    match src.get(start, len) {
        Some(b) => Ok(b),
        None => err!(Internal),
    }
}

/// The supported formats.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum FormatKind {
    /// Inferred from the extension of the name (file name or URL).
    Auto,
    Parquet,
    Csv,
    /// Tab-separated. The same implementation as CSV with a different delimiter.
    Tsv,
    /// One JSON object per line (NDJSON).
    Jsonl,
    /// The whole file is a single JSON value (a top-level array, or a single object).
    /// The equivalent of `read_json`/`read_json_auto` (see `format::json`).
    Json,
}

impl FormatKind {
    /// Infers the format from the name's extension.
    ///
    /// A URL's query string and fragment are dropped before inspection. When it cannot decide it
    /// counts as Parquet (the primary target format). A misdetection surfaces not as a magic-byte
    /// check but as a footer-resolution failure, and `BadMagic` is returned.
    pub fn detect(name: &str) -> FormatKind {
        let path = name.split(['?', '#']).next().unwrap_or(name);
        let ext = match path.rfind('.') {
            Some(i) => &path[i + 1..],
            None => return FormatKind::Parquet,
        };
        let e = ext.as_bytes();
        if eq_ascii_ci(e, b"csv") {
            FormatKind::Csv
        } else if eq_ascii_ci(e, b"tsv") || eq_ascii_ci(e, b"tab") {
            FormatKind::Tsv
        } else if eq_ascii_ci(e, b"jsonl") || eq_ascii_ci(e, b"ndjson") {
            FormatKind::Jsonl
        } else if eq_ascii_ci(e, b"json") {
            FormatKind::Json
        } else {
            FormatKind::Parquet
        }
    }
}

/// Builds a format implementation.
///
/// `Auto` infers from `name`. An unsupported format (its feature disabled) gives
/// `UnsupportedFeature`. Saying it is unsupported beats silently trying to read it as Parquet.
pub fn make(kind: FormatKind, name: &str) -> Result<Box<dyn TableFormat>> {
    let kind = match kind {
        FormatKind::Auto => FormatKind::detect(name),
        k => k,
    };
    match kind {
        FormatKind::Parquet => Ok(Box::new(parquet::ParquetFormat::new())),
        #[cfg(feature = "csv")]
        FormatKind::Csv => Ok(Box::new(csv::CsvFormat::new(b','))),
        #[cfg(feature = "csv")]
        FormatKind::Tsv => Ok(Box::new(csv::CsvFormat::new(b'\t'))),
        #[cfg(feature = "jsonl")]
        FormatKind::Jsonl => Ok(Box::new(jsonl::JsonlFormat::new())),
        #[cfg(feature = "jsonl")]
        FormatKind::Json => Ok(Box::new(json::JsonFormat::new())),
        #[allow(unreachable_patterns)]
        _ => err!(UnsupportedFeature),
    }
}

/// The chunk size a row-oriented format uses when cutting splits.
///
/// Too large and one split's memory balloons; too small and range-fetch round trips multiply.
/// It leans smaller than Parquet's typical RowGroup (tens of MB).
///
/// `format::csv` overrides this back down to "one split for the whole file" whenever the file
/// looks like it uses RFC 4180 quoting (see `CsvFormat::quoted_sample`) -- a quoted field's
/// embedded newline cannot be told apart from a real record boundary at an arbitrary split cut,
/// so this constant only actually governs splitting for CSV/TSV files confirmed unquoted (by
/// their leading sample), and for JSONL (whose records are single JSON lines and so can never
/// contain a raw, unescaped `\n`, making this scan always exact there).
#[cfg(any(feature = "csv", feature = "jsonl"))]
pub const TEXT_SPLIT_BYTES: u64 = 8 * 1024 * 1024;

/// The maximum bytes a row-oriented format may read past a split boundary looking for the next
/// newline when a record straddles one.
#[cfg(any(feature = "csv", feature = "jsonl"))]
pub const TEXT_MAX_RECORD: u64 = 1024 * 1024;

/// Time-zone suffix: `Z` / `z` or `[+-]HH[:]MM` / `[+-]HH`.
///
/// Returns `(offset_micros east of UTC, bytes consumed)`. Used by the text
/// readers so `2020-01-01T00:00:00Z` and a `COPY` TIMESTAMPTZ `+00` suffix
/// parse as timestamps instead of falling through to VARCHAR / NULL.
#[cfg(any(feature = "csv", feature = "jsonl"))]
pub(crate) fn scan_tz_suffix(s: &[u8]) -> Option<(i64, usize)> {
    match s.first()? {
        b'Z' | b'z' => Some((0, 1)),
        &sign @ (b'+' | b'-') => {
            if s.len() < 3 || !s[1].is_ascii_digit() || !s[2].is_ascii_digit() {
                return None;
            }
            let h = ((s[1] - b'0') as i64) * 10 + (s[2] - b'0') as i64;
            let (m, n) = if s.get(3) == Some(&b':') {
                if s.len() < 6 || !s[4].is_ascii_digit() || !s[5].is_ascii_digit() {
                    return None;
                }
                (((s[4] - b'0') as i64) * 10 + (s[5] - b'0') as i64, 6)
            } else {
                (0, 3)
            };
            if h > 23 || m > 59 {
                return None;
            }
            let micros = (h * 3600 + m * 60) * 1_000_000;
            Some((if sign == b'-' { -micros } else { micros }, n))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_detection() {
        assert_eq!(FormatKind::detect("a.parquet"), FormatKind::Parquet);
        assert_eq!(FormatKind::detect("a.CSV"), FormatKind::Csv);
        assert_eq!(FormatKind::detect("a.tsv"), FormatKind::Tsv);
        assert_eq!(FormatKind::detect("a.jsonl"), FormatKind::Jsonl);
        assert_eq!(FormatKind::detect("a.ndjson"), FormatKind::Jsonl);
        assert_eq!(FormatKind::detect("a.json"), FormatKind::Json);
        assert_eq!(FormatKind::detect("a.JSON"), FormatKind::Json);
        // No extension, or an unknown one, counts as Parquet.
        assert_eq!(FormatKind::detect("data"), FormatKind::Parquet);
        assert_eq!(FormatKind::detect("a.bin"), FormatKind::Parquet);
    }

    #[test]
    fn url_query_and_fragment_are_ignored() {
        assert_eq!(FormatKind::detect("https://x/y/trips.csv?token=abc"), FormatKind::Csv);
        assert_eq!(FormatKind::detect("https://x/y/trips.jsonl#frag"), FormatKind::Jsonl);
        // It must not be fooled by an extension in the query string.
        assert_eq!(FormatKind::detect("https://x/y/data.parquet?name=a.csv"), FormatKind::Parquet);
    }

    #[test]
    fn pruning_is_safe_when_statistics_are_unusable() {
        let p = Pruner { column: 0, op: PruneOp::Eq, value: Value::I64(1), in_values: Vec::new() };
        // Statistics whose types do not line up do not prune.
        assert!(range_may_match(&p, &Value::Bytes(vec![]), &Value::Bytes(vec![])));
        assert!(range_may_match(&p, &Value::Null, &Value::Null));
    }

    #[test]
    fn pruning_boundaries() {
        let gt =
            Pruner { column: 0, op: PruneOp::Gt, value: Value::I64(100), in_values: Vec::new() };
        assert!(!range_may_match(&gt, &Value::I64(0), &Value::I64(100)));
        assert!(range_may_match(&gt, &Value::I64(0), &Value::I64(101)));

        let ge =
            Pruner { column: 0, op: PruneOp::Ge, value: Value::I64(100), in_values: Vec::new() };
        assert!(range_may_match(&ge, &Value::I64(0), &Value::I64(100)));
        assert!(!range_may_match(&ge, &Value::I64(0), &Value::I64(99)));

        let eq =
            Pruner { column: 0, op: PruneOp::Eq, value: Value::I64(100), in_values: Vec::new() };
        assert!(range_may_match(&eq, &Value::I64(100), &Value::I64(100)));
        assert!(!range_may_match(&eq, &Value::I64(101), &Value::I64(200)));
    }

    #[test]
    fn pruning_in_list_matches_if_any_candidate_overlaps() {
        let p = Pruner {
            column: 0,
            op: PruneOp::In,
            value: Value::I64(5),
            in_values: vec![Value::I64(50), Value::I64(500)],
        };
        // 50 falls within [0,100], so it is kept.
        assert!(range_may_match(&p, &Value::I64(0), &Value::I64(100)));
        // No candidate falls within [1000,2000], so it can be skipped.
        assert!(!range_may_match(&p, &Value::I64(1000), &Value::I64(2000)));
    }

    #[test]
    fn invalid_source_range_returns_internal_without_panicking() {
        let src = crate::catalog::Source::from_bytes(vec![1, 2, 3]);
        assert_eq!(crate::error::code_of(get_or_internal(&src, 2, 1)), Some(Code::Internal));
    }

    #[test]
    fn pruning_in_list_is_safe_when_a_candidate_is_incomparable() {
        let p = Pruner {
            column: 0,
            op: PruneOp::In,
            value: Value::I64(5),
            in_values: vec![Value::Bytes(vec![1])],
        };
        // If even one candidate is incomparable it errs safe and is kept.
        assert!(range_may_match(&p, &Value::I64(1000), &Value::I64(2000)));
    }
}
