//! A decorator that virtually appends Hive-style partition columns.
//!
//! It extracts `key=value` segments from the directory part (before the file name) of a path such
//! as `s3://bucket/year=2024/month=01/part.parquet` and adds them to the schema and read results
//! as columns that do not exist inside the file.
//!
//! It is a pure decorator delegating everything to `inner: Box<dyn TableFormat>`, so
//! `parquet.rs` / `csv.rs` / `jsonl.rs` need not change by a single line.
//! Keeping it contained to this one file is the design's key point.

use alloc::string::ToString;

use crate::catalog::Source;
use crate::format::{range_may_match, CodecTask, Pruner, ResolveStep, TableFormat};
use crate::prelude::*;
use crate::vector::{Field, Ty, Value, Vector};

pub struct PartitionedFormat {
    inner: Box<dyn TableFormat>,
    /// `(column name, raw value)`. Extracted from the file's path exactly once, percent-decoded but
    /// left untyped: a partition key's type is a property of the **table**, not of one directory,
    /// so it is settled across every part by `catalog::register_multi` (`set_hive_types`).
    keys: Vec<(String, String)>,
    /// The column type of each key, parallel to `keys`. Defaults to this part's own reading of the
    /// raw value, which is already right for a single-part table.
    types: Vec<Ty>,
    /// `inner`'s schema plus the partition columns. Empty until `resolve` completes.
    schema: Vec<Field>,
}

impl PartitionedFormat {
    pub fn new(inner: Box<dyn TableFormat>, keys: Vec<(String, String)>) -> Self {
        let types = keys.iter().map(|(_, v)| hive_value_ty(v)).collect();
        PartitionedFormat { inner, keys, types, schema: Vec::new() }
    }

    /// Extracts `key=value` from the path's directory part (each segment except the file name).
    /// Returns empty when nothing is found (= not Hive partitioned).
    ///
    /// A URL's query string and fragment are dropped beforehand, as in `FormatKind::detect`.
    /// A segment with an empty `key` or `value` is ignored (so a merely decorative directory name
    /// containing an `=` is not misdetected).
    pub fn parse_hive_path(path: &str) -> Vec<(String, String)> {
        let path = crate::format::strip_url_query(path);
        let segs: Vec<&str> = path.split('/').collect();
        if segs.len() < 2 {
            return Vec::new();
        }
        // The last segment is the file name and is not examined.
        let mut out = Vec::new();
        for seg in &segs[..segs.len() - 1] {
            let Some(eq) = seg.find('=') else { continue };
            let (k, v) = (&seg[..eq], &seg[eq + 1..]);
            if k.is_empty() || v.is_empty() {
                continue;
            }
            out.push((k.to_string(), percent_decode(v)));
        }
        out
    }

    /// The constant value of partition key `i`, built to match its settled column type.
    fn value_of(&self, i: usize) -> Value {
        let Some((_, raw)) = self.keys.get(i) else { return Value::Null };
        typed_value(raw, self.types.get(i).copied().unwrap_or(Ty::Varchar))
    }

    /// A projection keeping only `inner`'s columns. Partition columns (indices at or beyond
    /// `inner`'s column count) do not exist in the real file and are not shown to `inner`.
    ///
    /// Even when none remain (a query selecting only partition columns), at least one of `inner`'s
    /// columns is still requested so the row count is known. The same idea as `plan/bind.rs`'s
    /// "add column 0 when the projection is empty" for `COUNT(*)`, applied here to the
    /// inner/partition column boundary.
    fn inner_projection(&self, projection: &[usize]) -> Vec<usize> {
        let inner_n = self.inner.schema().len();
        let mut v: Vec<usize> = projection.iter().copied().filter(|&c| c < inner_n).collect();
        if v.is_empty() && inner_n > 0 {
            v.push(0);
        }
        v
    }

    /// Remaps `pruners`/`projection` into `inner`'s column space. A pruner touching a partition
    /// column cannot be shown to `inner` (the real file has no such column), so it is decided on
    /// the spot -- a partition column is one constant for the whole file, so it can be run through
    /// `range_may_match` as `min = max = that constant`.
    ///
    /// The return value is `(inner projection, pruners remapped for inner, definitely no match)`.
    /// When the third is `true`, the partition-column pruners alone already show this split can be
    /// skipped entirely (without even asking `inner`).
    ///
    /// The same remapping is needed in three places -- `may_match` / `index_ranges` /
    /// `refine_with_index` -- so it is written once here.
    fn remap_pruners(
        &self,
        pruners: &[Pruner],
        projection: &[usize],
    ) -> (Vec<usize>, Vec<Pruner>, bool) {
        let inner_n = self.inner.schema().len();
        let inner_proj = self.inner_projection(projection);
        let mut inner_pruners: Vec<Pruner> = Vec::new();
        for p in pruners {
            let Some(&col) = projection.get(p.column) else { continue };
            if col < inner_n {
                if let Some(pos) = inner_proj.iter().position(|&c| c == col) {
                    inner_pruners.push(Pruner {
                        column: pos,
                        op: p.op,
                        value: p.value.clone(),
                        in_values: p.in_values.clone(),
                    });
                }
            } else {
                let v = self.value_of(col - inner_n);
                if !range_may_match(p, &v, &v) {
                    return (inner_proj, inner_pruners, true);
                }
            }
        }
        (inner_proj, inner_pruners, false)
    }
}

/// Decodes `%XX` escapes in a partition value back to the original bytes. Spark and others URL-
/// encode partition values containing spaces or symbols when writing them out (as in
/// `region=us%20east`), and DuckDB decodes them here as well (confirmed by measuring with the
/// `duckdb` CLI). An invalid `%` sequence (not hex, or the string ending partway) gives up on
/// decoding and passes through unchanged.
fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    if !b.contains(&b'%') {
        return s.to_string();
    }
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let (Some(hi), Some(lo)) = (hex_val(b[i + 1]), hex_val(b[i + 2])) {
                out.push((hi << 4) | lo);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| s.to_string())
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Matches DuckDB's Hive partition type inference: a value of digits only is INTEGER (or BIGINT if
/// it does not fit 32 bits), and anything else is VARCHAR.
/// Signs and decimal points are excluded from "digits only" (partitions containing `-1` or `1.5`
/// are rare in practice, and wrongly treating them as numeric is the more dangerous mistake).
///
/// A zero-padded run of digits (`k=007`, `month=01`) stays VARCHAR, for the same reason
/// `format::csv::is_zero_padded` keeps such a CSV column VARCHAR: reading it as a number destroys
/// the value. DuckDB keeps `month=01` as `01` too.
///
/// This types one directory's value in isolation. `catalog::register_multi` widens the result
/// across every part of the table before anything is read, so a key that is `k=1` under one
/// directory and `k=x` under another comes out VARCHAR everywhere rather than making the table
/// unreadable.
pub(crate) fn hive_value_ty(s: &str) -> Ty {
    let digits_only = !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let zero_padded = s.len() > 1 && s.starts_with('0');
    if digits_only && !zero_padded {
        if let Ok(v) = s.parse::<i64>() {
            return if i32::try_from(v).is_ok() { Ty::Int } else { Ty::BigInt };
        }
    }
    Ty::Varchar
}

/// Builds the constant for a partition value under an already-settled column type.
/// A value that does not fit the type falls back to its text, which cannot lose information.
fn typed_value(s: &str, ty: Ty) -> Value {
    match ty {
        Ty::Int => match s.parse::<i32>() {
            Ok(v) => Value::I32(v),
            Err(_) => Value::Bytes(s.as_bytes().to_vec()),
        },
        Ty::BigInt => match s.parse::<i64>() {
            Ok(v) => Value::I64(v),
            Err(_) => Value::Bytes(s.as_bytes().to_vec()),
        },
        _ => Value::Bytes(s.as_bytes().to_vec()),
    }
}

/// Builds a `rows`-row vector filled with a constant value.
fn constant_vector(ty: Ty, v: &Value, rows: usize) -> Vector {
    let mut out = Vector::with_capacity(ty, rows);
    for _ in 0..rows {
        out.push_value(v);
    }
    out
}

impl TableFormat for PartitionedFormat {
    fn resolve(&mut self, src: &Source) -> Result<ResolveStep> {
        let step = self.inner.resolve(src)?;
        if step.is_ok() && self.schema.is_empty() {
            let mut schema = self.inner.schema().to_vec();
            for (i, (name, _)) in self.keys.iter().enumerate() {
                // If a partition column collides with one of the file's own column names, the schema
                // ends up with two columns of the same name. Which one is looked up would depend on
                // column-resolution implementation details, and the unintended value could silently
                // come back, so it is clearly rejected -- the same policy as `catalog::unify_schema`
                // rejecting misaligned joins.
                ensure!(
                    !schema
                        .iter()
                        .any(|f| crate::rt::hash::eq_ascii_ci(f.name.as_bytes(), name.as_bytes())),
                    DuplicateColumn
                );
                let ty = self.types.get(i).copied().unwrap_or(Ty::Varchar);
                schema.push(Field::new(name.clone(), ty, false));
            }
            self.schema = schema;
        }
        Ok(step)
    }

    fn is_resolved(&self) -> bool {
        self.inner.is_resolved()
    }

    fn schema(&self) -> &[Field] {
        &self.schema
    }

    fn num_splits(&self) -> usize {
        self.inner.num_splits()
    }

    fn split_rows(&self, split: usize) -> Option<u64> {
        self.inner.split_rows(split)
    }

    fn split_ranges(
        &self,
        split: usize,
        projection: &[usize],
        out: &mut Vec<(u64, u64)>,
    ) -> Result<()> {
        self.inner.split_ranges(split, &self.inner_projection(projection), out)
    }

    fn codec_tasks(
        &self,
        src: &Source,
        split: usize,
        projection: &[usize],
        out: &mut Vec<CodecTask>,
    ) -> Result<()> {
        self.inner.codec_tasks(src, split, &self.inner_projection(projection), out)
    }

    fn may_match(&self, split: usize, pruners: &[Pruner], projection: &[usize]) -> bool {
        let (inner_proj, inner_pruners, reject) = self.remap_pruners(pruners, projection);
        if reject {
            return false;
        }
        self.inner.may_match(split, &inner_pruners, &inner_proj)
    }

    fn index_ranges(
        &self,
        split: usize,
        pruners: &[Pruner],
        projection: &[usize],
        out: &mut Vec<(u64, u64)>,
    ) -> Result<()> {
        let (inner_proj, inner_pruners, reject) = self.remap_pruners(pruners, projection);
        if reject {
            // The partition-column pruners alone already settle a non-match.
            // Not even the page-selection bytes are needed.
            return Ok(());
        }
        self.inner.index_ranges(split, &inner_pruners, &inner_proj, out)
    }

    fn refine_with_index(
        &mut self,
        src: &Source,
        split: usize,
        pruners: &[Pruner],
        projection: &[usize],
    ) -> Result<bool> {
        let (inner_proj, inner_pruners, reject) = self.remap_pruners(pruners, projection);
        if reject {
            return Ok(false);
        }
        self.inner.refine_with_index(src, split, &inner_pruners, &inner_proj)
    }

    fn read_split(&self, src: &Source, split: usize, projection: &[usize]) -> Result<Vec<Vector>> {
        let inner_n = self.inner.schema().len();
        let inner_proj = self.inner_projection(projection);
        let inner_cols = self.inner.read_split(src, split, &inner_proj)?;
        ensure!(inner_cols.len() == inner_proj.len(), Internal);
        let rows = inner_cols.first().map_or(0, |c| c.len());

        let mut out = Vec::with_capacity(projection.len());
        for &col in projection {
            if col < inner_n {
                let pos = match inner_proj.iter().position(|&c| c == col) {
                    Some(p) => p,
                    // `inner_projection` contains every inner column of projection, so this should
                    // be unreachable.
                    None => err!(Internal),
                };
                out.push(inner_cols[pos].clone());
            } else {
                let i = col - inner_n;
                let ty = self.types.get(i).copied().unwrap_or(Ty::Varchar);
                out.push(constant_vector(ty, &self.value_of(i), rows));
            }
        }
        Ok(out)
    }

    fn hive_keys(&self) -> &[(String, String)] {
        &self.keys
    }

    fn set_hive_types(&mut self, types: &[Ty]) {
        if types.len() == self.types.len() {
            self.types.copy_from_slice(types);
        }
    }

    /// A partitioned table's schema is only as declared as the file it wraps.
    fn schema_is_inferred(&self) -> bool {
        self.inner.schema_is_inferred()
    }

    /// The appended partition-key columns always have a value, so only the wrapped file's own
    /// columns (which come first, at the same indexes) can be evidence-free.
    fn column_has_no_evidence(&self, col: usize) -> bool {
        col < self.inner.schema().len() && self.inner.column_has_no_evidence(col)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::Value;

    #[test]
    fn hive_segments_are_parsed_from_directories_only() {
        let cols = PartitionedFormat::parse_hive_path("data/year=2024/month=01/part.parquet");
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0].0, "year");
        assert_eq!(cols[0].1, "2024");
        assert_eq!(hive_value_ty(&cols[0].1), Ty::Int);
        assert_eq!(cols[1].0, "month");
        // `01` keeps its padding, so it is VARCHAR rather than the integer 1 (duckdb agrees).
        assert_eq!(cols[1].1, "01");
        assert_eq!(hive_value_ty(&cols[1].1), Ty::Varchar);
    }

    #[test]
    fn hive_value_types() {
        assert_eq!(hive_value_ty("2024"), Ty::Int);
        assert_eq!(hive_value_ty("0"), Ty::Int);
        assert_eq!(hive_value_ty("99999999999"), Ty::BigInt);
        assert_eq!(hive_value_ty("007"), Ty::Varchar);
        assert_eq!(hive_value_ty("01"), Ty::Varchar);
        assert_eq!(hive_value_ty("1e3"), Ty::Varchar);
        assert_eq!(hive_value_ty("true"), Ty::Varchar);
        assert_eq!(hive_value_ty("-1"), Ty::Varchar);
        assert_eq!(hive_value_ty(""), Ty::Varchar);
    }

    #[test]
    fn a_local_path_with_a_hash_is_not_truncated() {
        // `#` and `?` are ordinary file-name characters; only a URL carries a fragment/query.
        let cols = PartitionedFormat::parse_hive_path("data/k=a#b/part.parquet");
        assert_eq!(cols[0].1, "a#b");
        let cols = PartitionedFormat::parse_hive_path("https://h/k=a/f.parquet?token=1");
        assert_eq!(cols[0].1, "a");
    }

    #[test]
    fn filename_is_never_treated_as_a_partition_segment() {
        // An `=` in the file name is not examined.
        let cols = PartitionedFormat::parse_hive_path("a=b.parquet");
        assert!(cols.is_empty());
    }

    #[test]
    fn no_key_value_segments_means_no_partitions() {
        assert!(PartitionedFormat::parse_hive_path("data/plain/file.parquet").is_empty());
        assert!(PartitionedFormat::parse_hive_path("file.parquet").is_empty());
    }

    #[test]
    fn non_numeric_values_stay_varchar() {
        let cols = PartitionedFormat::parse_hive_path("data/region=us-east/f.parquet");
        assert_eq!(cols[0].1, "us-east");
        assert_eq!(hive_value_ty(&cols[0].1), Ty::Varchar);
    }

    #[test]
    fn large_numeric_values_become_bigint() {
        let cols = PartitionedFormat::parse_hive_path("data/ts=99999999999/f.parquet");
        assert_eq!(hive_value_ty(&cols[0].1), Ty::BigInt);
        assert_eq!(typed_value(&cols[0].1, Ty::BigInt), Value::I64(99_999_999_999));
    }

    #[test]
    fn percent_encoded_values_are_decoded() {
        // Measured with duckdb: `region=us%20east` becomes "us east".
        let cols = PartitionedFormat::parse_hive_path("data/region=us%20east/f.parquet");
        assert_eq!(cols[0].1, "us east");
    }

    #[test]
    fn malformed_percent_escape_is_left_as_is() {
        // When what follows `%` is not hex, or is cut off partway, it passes through.
        let cols = PartitionedFormat::parse_hive_path("data/x=100%off/f.parquet");
        assert_eq!(cols[0].1, "100%off");
        let cols = PartitionedFormat::parse_hive_path("data/x=abc%2/f.parquet");
        assert_eq!(cols[0].1, "abc%2");
    }

    // --- A fake for verifying behavior against an isolated TableFormat -------

    struct FakeFormat {
        schema: Vec<Field>,
        rows: usize,
    }

    impl TableFormat for FakeFormat {
        fn resolve(&mut self, _src: &Source) -> Result<ResolveStep> {
            Ok(Ok(()))
        }
        fn is_resolved(&self) -> bool {
            true
        }
        fn schema(&self) -> &[Field] {
            &self.schema
        }
        fn num_splits(&self) -> usize {
            1
        }
        fn split_rows(&self, _split: usize) -> Option<u64> {
            Some(self.rows as u64)
        }
        fn split_ranges(
            &self,
            _split: usize,
            _projection: &[usize],
            _out: &mut Vec<(u64, u64)>,
        ) -> Result<()> {
            Ok(())
        }
        fn read_split(
            &self,
            _src: &Source,
            _split: usize,
            projection: &[usize],
        ) -> Result<Vec<Vector>> {
            Ok(projection
                .iter()
                .map(|&c| {
                    let mut v = Vector::with_capacity(self.schema[c].ty, self.rows);
                    for i in 0..self.rows {
                        v.push_value(&Value::I32(i as i32));
                    }
                    v
                })
                .collect())
        }
    }

    fn fake_source() -> Source {
        Source::from_bytes(Vec::new())
    }

    #[test]
    fn schema_is_extended_with_partition_columns() {
        let inner =
            Box::new(FakeFormat { schema: vec![Field::new("id", Ty::Int, false)], rows: 3 });
        let mut f = PartitionedFormat::new(
            inner,
            vec![
                ("year".to_string(), "2024".to_string()),
                ("region".to_string(), "us".to_string()),
            ],
        );
        assert!(f.resolve(&fake_source()).unwrap().is_ok());
        let schema = f.schema();
        assert_eq!(schema.len(), 3);
        assert_eq!(schema[0].name, "id");
        assert_eq!(schema[1].name, "year");
        assert_eq!(schema[1].ty, Ty::Int);
        assert_eq!(schema[2].name, "region");
        assert_eq!(schema[2].ty, Ty::Varchar);
    }

    #[test]
    fn read_split_appends_constant_partition_columns() {
        let inner =
            Box::new(FakeFormat { schema: vec![Field::new("id", Ty::Int, false)], rows: 4 });
        let mut f = PartitionedFormat::new(inner, vec![("year".to_string(), "2024".to_string())]);
        f.resolve(&fake_source()).unwrap().unwrap();

        let cols = f.read_split(&fake_source(), 0, &[0, 1]).unwrap();
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0].len(), 4);
        assert_eq!(cols[1].len(), 4);
        for i in 0..4 {
            assert!(matches!(cols[1].value_at(i), Value::I32(2024)));
        }
    }

    #[test]
    fn partition_column_colliding_with_a_real_file_column_is_rejected() {
        // If the file itself has a "year" column and the path also has `year=...`, the schema ends
        // up with two columns of the same name. Rather than leaving which one is looked up
        // undefined, it should be rejected with a clear error.
        let inner = Box::new(FakeFormat {
            schema: vec![Field::new("id", Ty::Int, false), Field::new("year", Ty::Int, false)],
            rows: 3,
        });
        let mut f = PartitionedFormat::new(inner, vec![("year".to_string(), "2024".to_string())]);
        assert!(f.resolve(&fake_source()).is_err());
    }

    #[test]
    fn read_split_works_when_only_partition_columns_are_projected() {
        // The row count is still known even when none of inner's columns are selected.
        let inner =
            Box::new(FakeFormat { schema: vec![Field::new("id", Ty::Int, false)], rows: 5 });
        let mut f = PartitionedFormat::new(inner, vec![("year".to_string(), "2024".to_string())]);
        f.resolve(&fake_source()).unwrap().unwrap();

        let cols = f.read_split(&fake_source(), 0, &[1]).unwrap();
        assert_eq!(cols.len(), 1);
        assert_eq!(cols[0].len(), 5);
    }
}
