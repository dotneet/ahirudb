//! The table catalog and byte sources.
//!
//! A `Source` is represented as "the set of byte ranges fetched so far". The point is
//! that this one type unifies two paths:
//!
//! - the whole thing is in memory ... a state with exactly one range covering `[0, len)`
//! - range-fetched from the host ... a state that grows as each needed range arrives
//!
//! Execution just calls `get()`, and when `None` comes back it raises a request to
//! fetch that range (the RowGroup boundary barrier in DESIGN.md §6).

use crate::format::{self, FormatKind, TableFormat};
use crate::prelude::*;
use crate::rt::hash::eq_ascii_ci;
#[cfg(feature = "ddl")]
use crate::vector::Value;
use crate::vector::{Field, Ty};

/// The set of byte ranges fetched so far.
pub struct Source {
    pub total_len: u64,
    /// `(start offset, data)`. Kept sorted by ascending start offset.
    chunks: Vec<(u64, Vec<u8>)>,
    /// Pages the host decompressed for us. The key is the compressed page body's
    /// `(offset in file, length)` (codec delegation, DESIGN.md §6).
    decoded: Vec<((u64, u32), Vec<u8>)>,
}

impl Source {
    /// The whole file is in memory.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Source { total_len: bytes.len() as u64, chunks: vec![(0, bytes)], decoded: Vec::new() }
    }

    /// The host holds it and it is read via range fetching.
    pub fn remote(total_len: u64) -> Self {
        Source { total_len, chunks: Vec::new(), decoded: Vec::new() }
    }

    /// Returns the slice for `[off, off+len)` if it has been fetched.
    pub fn get(&self, off: u64, len: usize) -> Option<&[u8]> {
        let end = off.checked_add(len as u64)?;
        // There are at most a few dozen ranges, so a linear scan suffices.
        for (start, data) in &self.chunks {
            let cend = start + data.len() as u64;
            if *start <= off && end <= cend {
                let s = (off - start) as usize;
                return Some(&data[s..s + len]);
            }
        }
        None
    }

    /// Registers fetched bytes. Adjacent and overlapping ranges are coalesced.
    pub fn insert(&mut self, off: u64, data: Vec<u8>) {
        if data.is_empty() {
            return;
        }
        self.chunks.push((off, data));
        self.chunks.sort_by_key(|(o, _)| *o);
        // Coalesce adjacent and overlapping ranges. Left alone, `get`'s linear scan grows.
        let mut merged: Vec<(u64, Vec<u8>)> = Vec::with_capacity(self.chunks.len());
        for (off, data) in self.chunks.drain(..) {
            match merged.last_mut() {
                Some((po, pd)) if off <= *po + pd.len() as u64 => {
                    let overlap = (*po + pd.len() as u64 - off) as usize;
                    if overlap < data.len() {
                        pd.extend_from_slice(&data[overlap..]);
                    }
                }
                _ => merged.push((off, data)),
            }
        }
        self.chunks = merged;
    }

    /// Returns the parts of the requested range that have not been fetched yet.
    pub fn missing(&self, off: u64, len: u64) -> Option<(u64, u64)> {
        if len == 0 || self.get(off, len as usize).is_some() {
            None
        } else {
            Some((off, len.min(self.total_len.saturating_sub(off))))
        }
    }

    pub fn is_complete(&self) -> bool {
        matches!(self.chunks.first(), Some((0, d)) if d.len() as u64 == self.total_len)
    }

    /// Registers a page the host decompressed.
    pub fn insert_decoded(&mut self, offset: u64, len: u32, data: Vec<u8>) {
        if let Some(e) = self.decoded.iter_mut().find(|(k, _)| *k == (offset, len)) {
            e.1 = data;
            return;
        }
        self.decoded.push(((offset, len), data));
    }

    /// Whether a decompressed page is already present.
    pub fn has_decoded(&self, offset: u64, len: u32) -> bool {
        self.decoded.iter().any(|(k, _)| *k == (offset, len))
    }

    pub fn decoded_bytes(&self) -> usize {
        self.decoded.iter().map(|(_, d)| d.len()).sum()
    }

    /// Discards decompressed pages. Called once a split has been fully processed.
    /// Hoarding them would hold more memory than the pre-compression file.
    pub fn clear_decoded(&mut self) {
        self.decoded.clear();
    }
}

impl crate::parquet::reader::PageCache for Source {
    fn get(&self, offset: u64, len: u32) -> Option<&[u8]> {
        self.decoded.iter().find(|(k, _)| *k == (offset, len)).map(|(_, d)| d.as_slice())
    }
}

/// One file making up a table.
///
/// `path` is the name passed at registration (usually a file path or URL) and is used
/// both for automatic format detection and for Hive partition parsing (which is
/// `session.rs`'s job). Here it is held as a plain identifier with no interpretation.
pub struct TablePart {
    pub path: String,
    pub source: Source,
    pub format: Box<dyn TableFormat>,
}

/// A registered table.
///
/// One logical table consists of one or more `TablePart`s (so that cases where several
/// files make up one table, such as Hive-style partition directories, can be
/// expressed). Each part holds and resolves "its own byte ranges" independently. From
/// the `Scan` operator's point of view this looks like a flat sequence that merely
/// renumbers splits across part boundaries (see `exec::Scan`).
pub struct Table {
    pub name: String,
    pub parts: Vec<TablePart>,
    /// The unified schema, fixed once every part is resolved. `None` while unresolved.
    schema: Option<Vec<Field>>,
}

/// Per-part I/O still needed before `Table::resolve` can finish: `(part index,
/// offset, len)` triples, one per part still missing bytes.
type PendingPartReads = Vec<(usize, u64, u64)>;

impl Table {
    /// Resolves the schemas of every part.
    ///
    /// If some parts lack bytes, it does not stop at the first one but **looks at every
    /// part** and returns the requests together. This lets the host fetch every file's
    /// footer in parallel (a round trip per part would serialize as many round trips as there are parts).
    ///
    /// Once every part is present, schema compatibility is checked and the unified
    /// schema is computed and cached exactly once.
    pub fn resolve(&mut self) -> Result<core::result::Result<(), PendingPartReads>> {
        let mut need = Vec::new();
        for (i, part) in self.parts.iter_mut().enumerate() {
            match part.format.resolve(&part.source)? {
                Ok(()) => {}
                Err((offset, len)) => need.push((i, offset, len)),
            }
        }
        if !need.is_empty() {
            return Ok(Err(need));
        }
        if self.schema.is_none() {
            self.schema = Some(unify_schema(&self.parts)?);
        }
        Ok(Ok(()))
    }

    /// Whether the unified schema is settled.
    pub fn is_resolved(&self) -> bool {
        self.schema.is_some()
    }

    /// The unified schema. Column names, order, and nullability follow the first part;
    /// types are widened with `Ty::unify`, and the whole becomes nullable if any single
    /// part allows NULL (see `unify_schema`). Empty while unresolved.
    pub fn schema(&self) -> &[Field] {
        self.schema.as_deref().unwrap_or(&[])
    }

    /// The total number of splits across all parts. Used for little more than progress reporting.
    pub fn num_splits(&self) -> usize {
        self.parts.iter().map(|p| p.format.num_splits()).sum()
    }
}

/// Merges the schemas of every part into one.
///
/// A differing column count, column names that do not line up at the same position
/// (ignoring case), or a combination of column types that cannot be unified all give
/// `TypeMismatch`. The same idea as `unify_setop_schema` (type reconciliation for
/// `UNION`) in `plan/bind.rs`, kept independently here because `catalog` should not depend on `plan`.
///
/// Looking at column names too is deliberate: matching on type alone by position would
/// silently merge columns with different meanings into one whenever parts order their
/// columns differently (but the types happen to be compatible). `Scan` (`exec::mod.rs`)
/// uses the unified column number directly as each part's physical column number, and
/// `Pruner` (statistics pruning) embeds column numbers too, so **parts cannot be
/// reordered relative to one another on a column-number basis** (reordering would
/// require per-part renumbering that includes `Pruner`, a large change bearing on the
/// correctness of statistics pruning). The design therefore errs on the safe side:
/// files with a different ordering are clearly rejected rather than accepted and reordered.
fn unify_schema(parts: &[TablePart]) -> Result<Vec<Field>> {
    let mut iter = parts.iter();
    let first = match iter.next() {
        Some(p) => p,
        None => return Ok(Vec::new()),
    };
    let mut out: Vec<Field> = first.format.schema().to_vec();
    for part in iter {
        let s = part.format.schema();
        ensure!(s.len() == out.len(), TypeMismatch);
        for (o, f) in out.iter_mut().zip(s) {
            ensure!(eq_ascii_ci(o.name.as_bytes(), f.name.as_bytes()), TypeMismatch);
            let ty = match Ty::unify(o.ty, f.ty) {
                Some(t) => t,
                None => err!(TypeMismatch),
            };
            o.ty = ty;
            o.nullable = o.nullable || f.nullable;
        }
    }
    Ok(out)
}

/// An in-memory table, exclusive to the `ddl`/`dml` features.
///
/// Entirely separate from `Table` (file-backed and read-only). This design realizes DML
/// without touching `Source`'s invariant (bytes never change once they are in), and
/// inserts, updates, and deletes only ever apply here (DESIGN.md §16).
///
/// Held row-wise: DML centers on per-row updates and deletes, and going columnar would
/// not help this engine's main arena (reading large Parquet files), so implementation
/// simplicity won out.
#[cfg(feature = "ddl")]
pub struct MemTable {
    pub name: String,
    pub schema: Vec<Field>,
    pub rows: Vec<Vec<Value>>,
}

#[cfg(feature = "ddl")]
impl MemTable {
    /// Converts rows `[start, end)` into a `Batch`. This centralizes the row-to-vector
    /// conversion used by `Scan` (`exec::MemScan`) and DML (`dml::update`/`dml::delete`).
    /// It only ever builds from in-memory data, so waiting on a split (`NeedIo`) as with
    /// `Source` cannot happen in principle.
    pub fn batch(&self, start: usize, end: usize) -> crate::vector::Batch {
        // With zero columns (right after `ALTER TABLE ... DROP COLUMN` removes the last
        // one) `cols` is empty and `Batch::new` cannot track the row count
        // (`num_rows()` consults `empty_rows` -- 0 by default -- when `cols.first()` is
        // absent, silently misreporting the real row count as 0). In that case
        // `Batch::rows_only` carries the row count explicitly.
        if self.schema.is_empty() {
            return crate::vector::Batch::rows_only(end - start);
        }
        let mut cols: Vec<crate::vector::Vector> = self
            .schema
            .iter()
            .map(|f| crate::vector::Vector::with_capacity(f.ty, end - start))
            .collect();
        for row in &self.rows[start..end] {
            for (c, v) in cols.iter_mut().zip(row.iter()) {
                c.push_value(v);
            }
        }
        crate::vector::Batch::new(cols)
    }
}

#[derive(Default)]
pub struct Catalog {
    tables: Vec<Table>,
    #[cfg(feature = "ddl")]
    mem: Vec<MemTable>,
    /// Views are `(name, the raw SQL of the query body)`. They are reparsed at bind time
    /// (`plan::bind::flatten_from`) on every reference. Holding an `ExprArena`/`QueryStmt`
    /// would make `catalog` depend on `sql::ast`, which is avoided.
    #[cfg(feature = "ddl")]
    views: Vec<(String, String)>,
}

/// A case-insensitive linear search by name. The lookup rule shared by
/// `index_of`/`mem_index_of`/`view_index_of` (identical across tables and views).
fn find_ci_index<'a>(mut names: impl Iterator<Item = &'a str>, target: &str) -> Option<usize> {
    names.position(|n| eq_ascii_ci(n.as_bytes(), target.as_bytes()))
}

impl Catalog {
    pub fn new() -> Self {
        Catalog {
            tables: Vec::new(),
            #[cfg(feature = "ddl")]
            mem: Vec::new(),
            #[cfg(feature = "ddl")]
            views: Vec::new(),
        }
    }

    /// Registers a single-file table. An existing table of the same name is replaced
    /// (re-registration is not an error).
    ///
    /// No I/O happens at this point. Schema resolution is deferred to the first query.
    pub fn register(&mut self, name: &str, source: Source, kind: FormatKind) -> Result<usize> {
        let fmt = format::make(kind, name)?;
        let part = TablePart { path: name.into(), source, format: fmt };
        self.register_multi(name, vec![part])
    }

    /// Registers several files as one logical table.
    ///
    /// Assumes the caller (`session.rs`) has already assembled each part's format
    /// (including wrapping for Hive partition columns). `catalog` merely bundles them
    /// as given and needs no knowledge that `format::partitioned` exists.
    pub fn register_multi(&mut self, name: &str, parts: Vec<TablePart>) -> Result<usize> {
        ensure!(!parts.is_empty(), Internal);
        // A file table must not silently shadow an in-memory table or view of
        // the same name (`CREATE TABLE t` then `register("t", …)` used to make
        // `SELECT` read the file and `INSERT` write the mem table).
        #[cfg(feature = "ddl")]
        {
            ensure!(self.mem_index_of(name).is_none(), DuplicateTable);
            ensure!(self.view_index_of(name).is_none(), DuplicateTable);
        }
        let t = Table { name: name.into(), parts, schema: None };
        Ok(match self.index_of(name) {
            Some(i) => {
                self.tables[i] = t;
                i
            }
            None => {
                self.tables.push(t);
                self.tables.len() - 1
            }
        })
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        find_ci_index(self.tables.iter().map(|t| t.name.as_str()), name)
    }

    pub fn get(&self, i: usize) -> Option<&Table> {
        self.tables.get(i)
    }

    pub fn get_mut(&mut self, i: usize) -> Option<&mut Table> {
        self.tables.get_mut(i)
    }

    pub fn len(&self) -> usize {
        self.tables.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tables.iter().map(|t| t.name.as_str())
    }

    // --- In-memory tables (`ddl`) --------------------------------------------

    #[cfg(feature = "ddl")]
    pub fn mem_index_of(&self, name: &str) -> Option<usize> {
        find_ci_index(self.mem.iter().map(|t| t.name.as_str()), name)
    }

    #[cfg(feature = "ddl")]
    pub fn mem_get(&self, i: usize) -> Option<&MemTable> {
        self.mem.get(i)
    }

    #[cfg(feature = "ddl")]
    pub fn mem_get_mut(&mut self, i: usize) -> Option<&mut MemTable> {
        self.mem.get_mut(i)
    }

    #[cfg(feature = "ddl")]
    pub fn mem_names(&self) -> impl Iterator<Item = &str> {
        self.mem.iter().map(|t| t.name.as_str())
    }

    /// Whether this name is free as a writable container (not taken by another
    /// file-backed table or view). Used by the collision check in `CREATE TABLE`/`CREATE VIEW`.
    #[cfg(feature = "ddl")]
    fn name_taken_by_other(&self, name: &str) -> bool {
        self.index_of(name).is_some() || self.view_index_of(name).is_some()
    }

    /// Case-insensitive uniqueness of schema column names.
    #[cfg(feature = "ddl")]
    fn unique_field_names(schema: &[Field]) -> bool {
        schema.iter().enumerate().all(|(i, f)| {
            schema[..i].iter().all(|g| !eq_ascii_ci(g.name.as_bytes(), f.name.as_bytes()))
        })
    }

    /// `CREATE TABLE t (...)` / `CREATE TABLE t AS SELECT ...`.
    /// With `replace`, an existing in-memory table of the same name is silently replaced.
    #[cfg(feature = "ddl")]
    pub fn mem_create(&mut self, name: &str, schema: Vec<Field>, replace: bool) -> Result<usize> {
        ensure!(!self.name_taken_by_other(name), DuplicateTable);
        ensure!(Self::unique_field_names(&schema), DuplicateColumn);
        match self.mem_index_of(name) {
            Some(i) => {
                ensure!(replace, DuplicateTable);
                self.mem[i] = MemTable { name: name.into(), schema, rows: Vec::new() };
                Ok(i)
            }
            None => {
                self.mem.push(MemTable { name: name.into(), schema, rows: Vec::new() });
                Ok(self.mem.len() - 1)
            }
        }
    }

    /// `DROP TABLE t`. File-backed tables are out of scope (being read-only, they always
    /// give `TableNotFound`).
    #[cfg(feature = "ddl")]
    pub fn mem_drop(&mut self, name: &str) -> Result<()> {
        match self.mem_index_of(name) {
            Some(i) => {
                self.mem.remove(i);
                Ok(())
            }
            None => err!(TableNotFound),
        }
    }

    /// Confirms the name refers to a writable in-memory table and returns its index.
    /// A file-backed table (read-only) gives `ReadOnlyTable`; a name in neither gives
    /// `TableNotFound`. Both `dml::insert`/`update`/`delete` and `ALTER TABLE` share this rule.
    #[cfg(feature = "ddl")]
    pub fn mem_index_writable(&self, name: &str) -> Result<usize> {
        if let Some(i) = self.mem_index_of(name) {
            return Ok(i);
        }
        if self.index_of(name).is_some() {
            err!(ReadOnlyTable);
        }
        err!(TableNotFound)
    }

    /// `ALTER TABLE t ADD COLUMN col ty ...`. Duplicate column names (ignoring case) are
    /// rejected. `value` is the already-evaluated DEFAULT from the caller
    /// (`ddl::alter_table`), or `Value::Null` if there is none, and the same value is
    /// appended to every existing row.
    #[cfg(feature = "ddl")]
    pub fn mem_add_column(&mut self, idx: usize, field: Field, value: Value) -> Result<()> {
        let mt = match self.mem.get_mut(idx) {
            Some(t) => t,
            None => err!(TableNotFound),
        };
        ensure!(
            !mt.schema.iter().any(|f| eq_ascii_ci(f.name.as_bytes(), field.name.as_bytes())),
            DuplicateColumn
        );
        mt.schema.push(field);
        for row in &mut mt.rows {
            row.push(value.clone());
        }
        Ok(())
    }

    /// `ALTER TABLE t DROP COLUMN col`. A missing column gives `ColumnNotFound`.
    #[cfg(feature = "ddl")]
    pub fn mem_drop_column(&mut self, idx: usize, col_name: &str) -> Result<()> {
        let mt = match self.mem.get_mut(idx) {
            Some(t) => t,
            None => err!(TableNotFound),
        };
        let pos = match mt
            .schema
            .iter()
            .position(|f| eq_ascii_ci(f.name.as_bytes(), col_name.as_bytes()))
        {
            Some(p) => p,
            None => err!(ColumnNotFound),
        };
        mt.schema.remove(pos);
        for row in &mut mt.rows {
            row.remove(pos);
        }
        Ok(())
    }

    /// `ALTER TABLE t RENAME COLUMN old TO new`. A missing `old` gives `ColumnNotFound`,
    /// and a `new` colliding with another existing column gives `DuplicateColumn`.
    #[cfg(feature = "ddl")]
    pub fn mem_rename_column(&mut self, idx: usize, old: &str, new: &str) -> Result<()> {
        let mt = match self.mem.get_mut(idx) {
            Some(t) => t,
            None => err!(TableNotFound),
        };
        let pos =
            match mt.schema.iter().position(|f| eq_ascii_ci(f.name.as_bytes(), old.as_bytes())) {
                Some(p) => p,
                None => err!(ColumnNotFound),
            };
        let conflict = mt
            .schema
            .iter()
            .enumerate()
            .any(|(i, f)| i != pos && eq_ascii_ci(f.name.as_bytes(), new.as_bytes()));
        ensure!(!conflict, DuplicateColumn);
        mt.schema[pos].name = new.into();
        Ok(())
    }

    /// `ALTER TABLE t RENAME TO new_name`. If `new_name` is taken by another file-backed
    /// table, a view, or a different in-memory table, it gives `DuplicateTable` (renaming
    /// to itself, i.e. changing only the case, is allowed).
    #[cfg(feature = "ddl")]
    pub fn mem_rename_table(&mut self, idx: usize, new_name: &str) -> Result<()> {
        let taken_by_other_mem = match self.mem_index_of(new_name) {
            Some(i) => i != idx,
            None => false,
        };
        ensure!(!self.name_taken_by_other(new_name) && !taken_by_other_mem, DuplicateTable);
        self.mem[idx].name = new_name.into();
        Ok(())
    }

    // --- Views (`ddl`) --------------------------------------------------------

    #[cfg(feature = "ddl")]
    pub fn view_index_of(&self, name: &str) -> Option<usize> {
        find_ci_index(self.views.iter().map(|(n, _)| n.as_str()), name)
    }

    /// The view body (the raw text of `SELECT ...`).
    #[cfg(feature = "ddl")]
    pub fn view_get(&self, i: usize) -> Option<&str> {
        self.views.get(i).map(|(_, sql)| sql.as_str())
    }

    #[cfg(feature = "ddl")]
    pub fn view_names(&self) -> impl Iterator<Item = &str> {
        self.views.iter().map(|(n, _)| n.as_str())
    }

    #[cfg(feature = "ddl")]
    pub fn view_create(
        &mut self,
        name: &str,
        query_sql: String,
        or_replace: bool,
    ) -> Result<usize> {
        ensure!(self.index_of(name).is_none() && self.mem_index_of(name).is_none(), DuplicateTable);
        match self.view_index_of(name) {
            Some(i) => {
                ensure!(or_replace, DuplicateTable);
                self.views[i] = (name.into(), query_sql);
                Ok(i)
            }
            None => {
                self.views.push((name.into(), query_sql));
                Ok(self.views.len() - 1)
            }
        }
    }

    #[cfg(feature = "ddl")]
    pub fn view_drop(&mut self, name: &str) -> Result<()> {
        match self.view_index_of(name) {
            Some(i) => {
                self.views.remove(i);
                Ok(())
            }
            None => err!(TableNotFound),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{CodecTask, Pruner, ResolveStep};
    use crate::vector::Vector;

    #[test]
    fn in_memory_source_serves_any_range() {
        let s = Source::from_bytes((0u8..=255).collect());
        assert_eq!(s.get(0, 4), Some(&[0u8, 1, 2, 3][..]));
        assert_eq!(s.get(250, 6), Some(&[250u8, 251, 252, 253, 254, 255][..]));
        assert_eq!(s.get(250, 7), None);
        assert!(s.is_complete());
    }

    #[test]
    fn remote_source_reports_missing_then_serves() {
        let mut s = Source::remote(1000);
        assert_eq!(s.missing(100, 50), Some((100, 50)));
        s.insert(100, vec![7u8; 50]);
        assert_eq!(s.missing(100, 50), None);
        assert_eq!(s.get(120, 10), Some(&[7u8; 10][..]));
        // No other range exists yet.
        assert_eq!(s.missing(300, 10), Some((300, 10)));
    }

    #[test]
    fn adjacent_ranges_merge() {
        let mut s = Source::remote(300);
        s.insert(100, vec![1u8; 50]);
        s.insert(150, vec![2u8; 50]);
        // Coalesced, so it can be taken as a single slice.
        let got = s.get(140, 20).unwrap();
        assert_eq!(&got[..10], &[1u8; 10]);
        assert_eq!(&got[10..], &[2u8; 10]);
    }

    #[test]
    fn overlapping_ranges_merge_without_duplication() {
        let mut s = Source::remote(300);
        s.insert(0, vec![1u8; 100]);
        s.insert(50, vec![2u8; 100]);
        assert_eq!(s.get(0, 150).unwrap().len(), 150);
        // The overlapping part keeps whichever arrived first, and only the non-overlapping remainder is added.
        assert_eq!(s.get(0, 150).unwrap()[99], 1);
        assert_eq!(s.get(0, 150).unwrap()[100], 2);
    }

    #[test]
    fn missing_range_is_clamped_to_file_length() {
        let s = Source::remote(120);
        assert_eq!(s.missing(100, 500), Some((100, 20)));
    }

    #[test]
    fn register_replaces_same_name_case_insensitively() {
        let mut c = Catalog::new();
        let a = c.register("Trips", Source::from_bytes(vec![1]), FormatKind::Parquet).unwrap();
        let b = c.register("trips", Source::from_bytes(vec![2]), FormatKind::Parquet).unwrap();
        assert_eq!(a, b);
        assert_eq!(c.len(), 1);
        assert_eq!(c.index_of("TRIPS"), Some(0));
    }

    // --- A mock TableFormat for multi-file tests ------------------------------
    //
    // We want to unit-test only how several parts are bundled, without dragging in the
    // real Parquet/CSV parsers. So this defines a minimal format: "there are `total`
    // bytes in all, and until they are present it requests `(0, total)`".

    struct MockFormat {
        schema: Vec<Field>,
        total: u64,
        resolved: bool,
        rows_per_split: u64,
        splits: usize,
    }

    impl MockFormat {
        fn new(schema: Vec<Field>, total: u64, splits: usize, rows_per_split: u64) -> Self {
            MockFormat { schema, total, resolved: false, rows_per_split, splits }
        }
    }

    impl TableFormat for MockFormat {
        fn resolve(&mut self, src: &Source) -> Result<ResolveStep> {
            if self.resolved {
                return Ok(Ok(()));
            }
            if src.get(0, self.total as usize).is_none() {
                return Ok(Err((0, self.total)));
            }
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
            self.splits
        }

        fn split_rows(&self, _split: usize) -> Option<u64> {
            Some(self.rows_per_split)
        }

        fn split_ranges(
            &self,
            _split: usize,
            _projection: &[usize],
            out: &mut Vec<(u64, u64)>,
        ) -> Result<()> {
            out.push((0, self.total));
            Ok(())
        }

        fn codec_tasks(
            &self,
            _src: &Source,
            _split: usize,
            _projection: &[usize],
            _out: &mut Vec<CodecTask>,
        ) -> Result<()> {
            Ok(())
        }

        fn may_match(&self, _split: usize, _pruners: &[Pruner], _projection: &[usize]) -> bool {
            true
        }

        fn read_split(
            &self,
            _src: &Source,
            _split: usize,
            projection: &[usize],
        ) -> Result<Vec<Vector>> {
            let n = self.rows_per_split as usize;
            Ok(projection
                .iter()
                .map(|&c| {
                    let mut v = Vector::with_capacity(self.schema[c].ty, n);
                    for i in 0..n {
                        v.push_value(&crate::vector::Value::I64(i as i64));
                    }
                    v
                })
                .collect())
        }
    }

    fn mock_part(path: &str, schema: Vec<Field>, total: u64) -> TablePart {
        TablePart {
            path: path.into(),
            source: Source::remote(total),
            format: Box::new(MockFormat::new(schema, total, 1, 4)),
        }
    }

    #[test]
    fn resolve_unions_needio_across_parts() {
        let schema = vec![Field::new("id", Ty::BigInt, false)];
        let parts = vec![
            mock_part("a.parquet", schema.clone(), 100),
            mock_part("b.parquet", schema.clone(), 200),
            mock_part("c.parquet", schema, 300),
        ];
        let mut c = Catalog::new();
        let i = c.register_multi("t", parts).unwrap();
        let t = c.get_mut(i).unwrap();

        // No part has any bytes yet, so requests for all three come back together.
        let need = match t.resolve().unwrap() {
            Err(need) => need,
            Ok(()) => panic!("expected NeedIo"),
        };
        assert_eq!(need.len(), 3);
        let mut sorted = need.clone();
        sorted.sort_by_key(|(p, _, _)| *p);
        assert_eq!(sorted, vec![(0, 0, 100), (1, 0, 200), (2, 0, 300)]);
        assert!(!t.is_resolved());
    }

    #[test]
    fn resolve_progresses_as_parts_arrive_incrementally() {
        let schema = vec![Field::new("id", Ty::BigInt, false)];
        let parts =
            vec![mock_part("a.parquet", schema.clone(), 100), mock_part("b.parquet", schema, 200)];
        let mut c = Catalog::new();
        let i = c.register_multi("t", parts).unwrap();

        // Deliver only the first part.
        {
            let t = c.get_mut(i).unwrap();
            t.parts[0].source.insert(0, vec![0u8; 100]);
            let need = match t.resolve().unwrap() {
                Err(need) => need,
                Ok(()) => panic!("expected NeedIo"),
            };
            // Only the second part is still needed.
            assert_eq!(need, vec![(1, 0, 200)]);
        }

        // Deliver the second part too.
        {
            let t = c.get_mut(i).unwrap();
            t.parts[1].source.insert(0, vec![0u8; 200]);
            assert!(t.resolve().unwrap().is_ok());
            assert!(t.is_resolved());
            assert_eq!(t.schema().len(), 1);
        }
    }

    #[test]
    fn schema_mismatch_across_parts_is_rejected() {
        let a_schema = vec![Field::new("id", Ty::BigInt, false)];
        let b_schema =
            vec![Field::new("id", Ty::Varchar, false), Field::new("extra", Ty::Int, true)];
        let mut c = Catalog::new();
        let i = c
            .register_multi(
                "t",
                vec![mock_part("a.parquet", a_schema, 10), mock_part("b.parquet", b_schema, 10)],
            )
            .unwrap();
        let t = c.get_mut(i).unwrap();
        for p in &mut t.parts {
            p.source.insert(0, vec![0u8; 10]);
        }
        // Different column counts, hence TypeMismatch.
        assert!(t.resolve().is_err());
    }

    #[test]
    fn schema_type_mismatch_is_rejected_when_unify_fails() {
        // VARCHAR and INTEGER are a combination that cannot be unified.
        let a_schema = vec![Field::new("id", Ty::Varchar, false)];
        let b_schema = vec![Field::new("id", Ty::Int, false)];
        let mut c = Catalog::new();
        let i = c
            .register_multi(
                "t",
                vec![mock_part("a.parquet", a_schema, 10), mock_part("b.parquet", b_schema, 10)],
            )
            .unwrap();
        let t = c.get_mut(i).unwrap();
        for p in &mut t.parts {
            p.source.insert(0, vec![0u8; 10]);
        }
        assert!(t.resolve().is_err());
    }

    #[test]
    fn nullable_widens_and_type_unifies_across_parts() {
        // INT and BIGINT widen to BIGINT, and NOT NULL with NULL widens to nullable.
        let a_schema = vec![Field::new("id", Ty::Int, false)];
        let b_schema = vec![Field::new("id", Ty::BigInt, true)];
        let mut c = Catalog::new();
        let i = c
            .register_multi(
                "t",
                vec![mock_part("a.parquet", a_schema, 10), mock_part("b.parquet", b_schema, 10)],
            )
            .unwrap();
        let t = c.get_mut(i).unwrap();
        for p in &mut t.parts {
            p.source.insert(0, vec![0u8; 10]);
        }
        assert!(t.resolve().unwrap().is_ok());
        let s = t.schema();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].ty, Ty::BigInt);
        assert!(s[0].nullable);
        // Column names follow the first part.
        assert_eq!(s[0].name, "id");
    }

    #[test]
    fn columns_swapped_across_parts_are_rejected_even_when_types_would_unify() {
        // a.parquet: (id INT, region VARCHAR) / b.parquet: (region VARCHAR, id INT).
        // Aligning by position alone would still be noticed here, since INT<->VARCHAR
        // cannot be unified, but for a combination where the types happen to be
        // compatible (both VARCHAR, say), aligning by position without looking at names
        // would silently merge columns of different meaning. Requiring the names to line
        // up positionally rejects that.
        let a_schema =
            vec![Field::new("id", Ty::Varchar, false), Field::new("region", Ty::Varchar, false)];
        let b_schema =
            vec![Field::new("region", Ty::Varchar, false), Field::new("id", Ty::Varchar, false)];
        let mut c = Catalog::new();
        let i = c
            .register_multi(
                "t",
                vec![mock_part("a.parquet", a_schema, 10), mock_part("b.parquet", b_schema, 10)],
            )
            .unwrap();
        let t = c.get_mut(i).unwrap();
        for p in &mut t.parts {
            p.source.insert(0, vec![0u8; 10]);
        }
        assert!(t.resolve().is_err(), "parts with a different column order must be rejected even when the types are compatible");
    }

    #[test]
    fn column_name_case_differs_across_parts_still_unifies() {
        // Column name comparison ignores case (the same convention as the other name
        // comparisons in this file, such as `index_of`).
        let a_schema = vec![Field::new("ID", Ty::Int, false)];
        let b_schema = vec![Field::new("id", Ty::Int, false)];
        let mut c = Catalog::new();
        let i = c
            .register_multi(
                "t",
                vec![mock_part("a.parquet", a_schema, 10), mock_part("b.parquet", b_schema, 10)],
            )
            .unwrap();
        let t = c.get_mut(i).unwrap();
        for p in &mut t.parts {
            p.source.insert(0, vec![0u8; 10]);
        }
        assert!(t.resolve().unwrap().is_ok());
    }

    // CSV is used as the fixture, so `csv` is required (resolving `FormatKind::Csv`
    // gives UnsupportedFeature without it).
    #[cfg(feature = "csv")]
    #[test]
    fn single_part_register_is_unchanged() {
        let mut c = Catalog::new();
        let i = c.register("t", Source::from_bytes(vec![1, 2, 3]), FormatKind::Csv).unwrap();
        let t = c.get(i).unwrap();
        assert_eq!(t.parts.len(), 1);
        assert_eq!(t.parts[0].path, "t");
    }

    // --- Direct unit tests of MemTable (`ddl`) --------------------------------
    //
    // The ADD/DROP/RENAME COLUMN tests up to here all went through `ddl.rs` or the
    // integration tests (by throwing SQL strings at it). These call the `Catalog::mem_*`
    // methods themselves and check the invariant that the schema and every row's length stay equal.

    #[cfg(feature = "ddl")]
    fn mem_catalog_with_two_rows() -> (Catalog, usize) {
        let mut c = Catalog::new();
        let i = c
            .mem_create(
                "t",
                vec![Field::new("a", Ty::Int, false), Field::new("b", Ty::Varchar, true)],
                false,
            )
            .unwrap();
        c.mem_get_mut(i).unwrap().rows = vec![
            vec![Value::I32(1), Value::Bytes(b"x".to_vec())],
            vec![Value::I32(2), Value::Bytes(b"y".to_vec())],
        ];
        (c, i)
    }

    #[cfg(feature = "ddl")]
    fn assert_schema_and_rows_stay_in_sync(c: &Catalog, i: usize) {
        let mt = c.mem_get(i).unwrap();
        for (r, row) in mt.rows.iter().enumerate() {
            assert_eq!(
                row.len(),
                mt.schema.len(),
                "row {r} has length {} but the schema has {} columns",
                row.len(),
                mt.schema.len()
            );
        }
    }

    #[cfg(feature = "ddl")]
    #[test]
    fn mem_add_column_backfills_existing_rows_and_rejects_duplicate_name() {
        let (mut c, i) = mem_catalog_with_two_rows();
        c.mem_add_column(i, Field::new("c", Ty::Boolean, true), Value::Bool(true)).unwrap();
        let mt = c.mem_get(i).unwrap();
        assert_eq!(mt.schema.len(), 3);
        assert_eq!(mt.rows[0][2], Value::Bool(true));
        assert_eq!(mt.rows[1][2], Value::Bool(true));
        assert_schema_and_rows_stay_in_sync(&c, i);

        // A collision with an existing column name (ignoring case) is rejected.
        let r = c.mem_add_column(i, Field::new("A", Ty::Int, true), Value::Null);
        assert!(r.is_err());
    }

    #[cfg(feature = "ddl")]
    #[test]
    fn mem_drop_column_removes_slot_from_every_row_down_to_zero_columns() {
        let (mut c, i) = mem_catalog_with_two_rows();
        c.mem_drop_column(i, "b").unwrap();
        assert_eq!(c.mem_get(i).unwrap().schema.len(), 1);
        assert_schema_and_rows_stay_in_sync(&c, i);

        // Even the last remaining column can be dropped (unlike DuckDB, no constraint is
        // imposed, since this is not columnar).
        c.mem_drop_column(i, "a").unwrap();
        let mt = c.mem_get(i).unwrap();
        assert_eq!(mt.schema.len(), 0);
        assert!(mt.rows.iter().all(|r| r.is_empty()));
        assert_schema_and_rows_stay_in_sync(&c, i);

        // A column that does not exist.
        assert!(c.mem_drop_column(i, "nope").is_err());
    }

    #[cfg(feature = "ddl")]
    #[test]
    fn mem_rename_column_updates_name_without_touching_data() {
        let (mut c, i) = mem_catalog_with_two_rows();
        c.mem_rename_column(i, "a", "a2").unwrap();
        let mt = c.mem_get(i).unwrap();
        assert_eq!(mt.schema[0].name, "a2");
        assert_eq!(mt.rows[0][0], Value::I32(1), "the data is unchanged");

        // An old name that does not exist.
        assert!(c.mem_rename_column(i, "nope", "x").is_err());
        // A collision with another existing column name.
        assert!(c.mem_rename_column(i, "a2", "b").is_err());
    }

    #[cfg(feature = "ddl")]
    // CSV is used as the fixture, so `csv` is required (resolving `FormatKind::Csv`
    // gives UnsupportedFeature without it).
    #[cfg(feature = "csv")]
    #[test]
    fn mem_rename_table_allows_case_only_change_but_rejects_real_collisions() {
        let (mut c, i) = mem_catalog_with_two_rows();
        // A change of case only (renaming to itself) is allowed.
        c.mem_rename_table(i, "T").unwrap();
        assert_eq!(c.mem_get(i).unwrap().name, "T");

        // Collisions with another in-memory table or a file-backed table are rejected.
        c.mem_create("other", vec![Field::new("x", Ty::Int, true)], false).unwrap();
        assert!(c.mem_rename_table(i, "other").is_err());
        c.register("filetable", Source::from_bytes(vec![1]), FormatKind::Csv).unwrap();
        assert!(c.mem_rename_table(i, "filetable").is_err());
    }

    #[cfg(feature = "ddl")]
    #[test]
    fn mem_schema_row_length_invariant_survives_add_drop_rename_sequence() {
        let (mut c, i) = mem_catalog_with_two_rows();
        c.mem_add_column(i, Field::new("c", Ty::Boolean, true), Value::Null).unwrap();
        assert_schema_and_rows_stay_in_sync(&c, i);
        c.mem_rename_column(i, "b", "b2").unwrap();
        assert_schema_and_rows_stay_in_sync(&c, i);
        c.mem_drop_column(i, "a").unwrap();
        assert_schema_and_rows_stay_in_sync(&c, i);
        c.mem_add_column(i, Field::new("d", Ty::Varchar, true), Value::Bytes(b"z".to_vec()))
            .unwrap();
        assert_schema_and_rows_stay_in_sync(&c, i);
        let mt = c.mem_get(i).unwrap();
        assert_eq!(mt.schema.len(), 3);
        assert_eq!(mt.schema[0].name, "b2");
        assert_eq!(mt.schema[1].name, "c");
        assert_eq!(mt.schema[2].name, "d");
    }
}
