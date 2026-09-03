//! Writing (`export` feature).
//!
//! Does not touch the read path (`Source` / `TableFormat`) at all. It is built
//! entirely on top of `Session`'s existing public API (`prepare` / `step`),
//! driven from the outside, so it cannot break the read side's invariants
//! (a byte range never changes once loaded; I/O is only awaited at split
//! boundaries). That is what "opt-out-able" means here: disable the `export`
//! feature and this whole module disappears, with no effect anywhere else
//! (DESIGN.md §15).
//!
//! ## v1 limitation: non-resumable design
//!
//! The read engine's core design is to "stop and return a request when bytes
//! are missing" (DESIGN.md §6), but this write driver does not expose that
//! type. If `export_all` hits `NEED_IO` while executing the query, it fails
//! with `Err(IoFailed)`. It only works when all data is already in memory
//! (CLI usage, or when the JS side has already fully fetched the table). A
//! resumable write path that cooperates with the host's fetch loop would need
//! an ABI shaped like `ahiru_query_step` for the write side too; that is
//! deferred past v1.
//!
//! `NEED_CODEC` is **not** part of that limitation, even though it used to be
//! treated as one. A codec request never means "bytes are missing": the
//! compressed bytes were already delivered by the preceding `NEED_IO`, and
//! only the decompression itself is delegated (DESIGN.md §6, "Codec
//! delegation protocol"). Since nothing has to be fetched, the export driver
//! answers it in place through [`Session::set_codec_hook`] and carries on --
//! so a GZIP-compressed source works here exactly as it does on the streaming
//! query path. A host that registers no hook gets `UnsupportedCodec`, which
//! at least names the real problem.

#[cfg(feature = "csv")]
pub mod csv;
// Shared `f64` shortest-round-trip formatting for `csv`/`jsonl`. The module
// itself lives in `expr/`, which is always compiled, because the `CAST(<double>
// AS VARCHAR)` kernel needs the identical rendering and is not gated behind
// `export`; this is a re-export so `super::float::write_f64_finite` keeps
// resolving inside the writers. Only bound when at least one writer is built,
// since it has no other caller here.
#[cfg(any(feature = "csv", feature = "jsonl"))]
pub(crate) use crate::expr::float;
#[cfg(feature = "jsonl")]
pub mod jsonl;
#[cfg(feature = "export-parquet")]
pub mod parquet;

use crate::prelude::*;
use crate::rt::hash::eq_ascii_ci;
use crate::session::{Prepared, Query, QueryStep, Session};
use crate::sql::ast::{ExprArena, QueryStmt};
use crate::vector::{Batch, Field, Value};

/// Output format.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum ExportFormat {
    #[cfg(feature = "csv")]
    Csv,
    #[cfg(feature = "csv")]
    Tsv,
    #[cfg(feature = "jsonl")]
    Jsonl,
    #[cfg(feature = "export-parquet")]
    Parquet,
}

/// Write destination. A thin abstraction that just converts a `Batch` to bytes.
///
/// Symmetric with the read side's `TableFormat` (one produces `Batch`, the
/// other consumes it). Shares the core types (`Batch`/`Vector`/`Field`) but
/// does not depend on any read-side type. Call `begin` exactly once, followed
/// by zero or more `write_batch` calls and one `finish`; the built-in sinks
/// reject calls outside that order and may be started again after finishing.
pub trait TableSink {
    /// Header-equivalent information. Implementations may write the first bytes here if needed.
    fn begin(&mut self, schema: &[Field]) -> Result<()>;
    /// Writes the rows of one batch. `selection` is passed already resolved (after `materialize`).
    fn write_batch(&mut self, schema: &[Field], batch: &Batch) -> Result<()>;
    /// Finalizes (footer, closing brackets, etc.) and returns the completed byte sequence.
    fn finish(&mut self) -> Result<Vec<u8>>;
}

/// Validate the dense batch contract shared by all sinks.
///
/// `export_all` materializes selections before calling a sink, but the sink
/// implementations are public and can also be driven directly. Without this
/// check a short/mismatched column is silently truncated by `zip` (CSV/JSONL)
/// or padded with a physical zero (Parquet), producing plausible but corrupt
/// output.
pub(crate) fn validate_batch(schema: &[Field], batch: &Batch) -> Result<()> {
    ensure!(batch.sel.is_none(), Internal);
    ensure!(batch.cols.len() == schema.len(), Internal);
    ensure!(batch.cols.iter().zip(schema).all(|(column, field)| column.ty() == field.ty), Internal);
    let rows = batch.num_rows();
    ensure!(batch.cols.iter().all(|c| c.len() == rows), Internal);
    Ok(())
}

/// Executes a query and writes the whole result to `sink`.
///
/// **Non-resumable**: fails with `IoFailed` if `NEED_IO` occurs during
/// execution. `NEED_CODEC` is answered in place via the session's codec hook
/// (see module doc).
pub fn export_all(
    session: &mut Session,
    sql: &str,
    params: &[Value],
    sink: &mut dyn TableSink,
) -> Result<Vec<u8>> {
    let mut q = match session.prepare(sql, params)? {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => err!(IoFailed),
    };
    sink.begin(&q.schema)?;
    loop {
        match session.step(&mut q)? {
            QueryStep::Batch(mut b) => {
                // Resolve the selection before passing it on, so the sink side only ever sees dense columns.
                b.materialize();
                sink.write_batch(&q.schema, &b)?;
            }
            QueryStep::NeedIo(_) => err!(IoFailed),
            // The compressed bytes are already in memory; only the inflate is
            // missing. Service it in place through the session's codec hook,
            // the same work the host does between two `step` calls
            // (DESIGN.md §6). With no hook registered this reports
            // `UnsupportedCodec`.
            QueryStep::NeedCodec(reqs) => session.service_codec(&reqs)?,
            QueryStep::Done => break,
        }
    }
    sink.finish()
}

/// The "from an already-parsed `QueryStmt`" version of `export_all`.
///
/// `Stmt::Copy` already has an `ExprArena`/`QueryStmt` from inside
/// `Session::prepare`. To avoid having to turn the tree back into SQL text
/// just to pass it to `export_all`, this twin function takes only the input
/// as an AST instead. It is built the same way as `export_all`, through
/// `Session`'s public `prepare`/`step` API (see the module doc's
/// "independence" section; it never touches `plan::bind`/`exec::build`).
///
/// **Non-resumable**: same constraint as `export_all` (see module doc).
fn export_query(
    session: &mut Session,
    arena: &ExprArena,
    query: &QueryStmt,
    params: &[Value],
    sink: &mut dyn TableSink,
) -> Result<Vec<u8>> {
    let mut q = match session.prepare_query(arena, query, params)? {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => err!(IoFailed),
    };
    sink.begin(&q.schema)?;
    loop {
        match session.step(&mut q)? {
            QueryStep::Batch(mut b) => {
                b.materialize();
                sink.write_batch(&q.schema, &b)?;
            }
            QueryStep::NeedIo(_) => err!(IoFailed),
            // The compressed bytes are already in memory; only the inflate is
            // missing. Service it in place through the session's codec hook,
            // the same work the host does between two `step` calls
            // (DESIGN.md §6). With no hook registered this reports
            // `UnsupportedCodec`.
            QueryStep::NeedCodec(reqs) => session.service_codec(&reqs)?,
            QueryStep::Done => break,
        }
    }
    sink.finish()
}

/// Execution body for `Stmt::Copy`. Called from `Session::prepare`.
///
/// The format comes from an explicit `FORMAT csv|jsonl|json|parquet` when
/// given, and otherwise from the extension of `path` via
/// `format::FormatKind::detect`. Which formats can actually be written
/// depends on the enabled features (`csv`, `jsonl`, `export-parquet`);
/// anything else is `UnsupportedFeature`.
///
/// **Does not write to a file**: the result is wrapped in `Query::copy_result`
/// and returned. Actually writing to `path` is the caller's responsibility
/// (`ahiru-cli` on native; see the module doc and DESIGN.md §15).
pub(crate) fn copy(
    session: &mut Session,
    arena: &ExprArena,
    query: &QueryStmt,
    path: &str,
    format: Option<&str>,
    params: &[Value],
) -> Result<Prepared> {
    let fmt = resolve_format(path, format)?;
    let data = match fmt {
        #[cfg(feature = "csv")]
        ExportFormat::Csv => {
            let mut sink = csv::CsvSink::new();
            export_query(session, arena, query, params, &mut sink)?
        }
        #[cfg(feature = "csv")]
        ExportFormat::Tsv => {
            let mut sink = csv::CsvSink::with_delimiter(b'\t');
            export_query(session, arena, query, params, &mut sink)?
        }
        #[cfg(feature = "jsonl")]
        ExportFormat::Jsonl => {
            let mut sink = jsonl::JsonlSink::new();
            export_query(session, arena, query, params, &mut sink)?
        }
        #[cfg(feature = "export-parquet")]
        ExportFormat::Parquet => {
            let mut sink = parquet::ParquetSink::new();
            export_query(session, arena, query, params, &mut sink)?
        }
    };
    Ok(Prepared::Ready(Query::copy_result(path.to_owned(), data)))
}

/// Resolves `format` if given, otherwise infers it from `path`'s extension.
///
/// Only a *recognized* extension picks the format. The read side's `FormatKind::detect`
/// resolves an unknown or absent extension to Parquet, which is the right guess when opening
/// an existing file but the wrong one when creating a new one: `COPY ... TO 'report'` would
/// silently produce a Parquet file under a name that reads as text. DuckDB writes CSV there,
/// and CSV is the format a reader can still make sense of without the extension, so an
/// unrecognized extension takes that route here too. Ask for `(FORMAT PARQUET)` to write
/// Parquet under an unusual name.
fn resolve_format(path: &str, format: Option<&str>) -> Result<ExportFormat> {
    match format {
        Some(f) => format_by_name(f),
        // `FormatKind::Csv`/`Tsv` map to the CSV sink, and `Jsonl`/`Json` map to the
        // JSONL sink. DuckDB's `COPY ... (FORMAT JSON)` also writes newline-delimited
        // JSON rather than an array (confirmed empirically), so the write side's
        // meaning of `Json` diverges from the read side's "one JSON value per file".
        // That is acceptable since v1 has no separate JSON-array sink for writing;
        // this is the appropriate mapping.
        None => match crate::format::FormatKind::detect(path) {
            #[cfg(feature = "csv")]
            crate::format::FormatKind::Csv => Ok(ExportFormat::Csv),
            #[cfg(feature = "csv")]
            crate::format::FormatKind::Tsv => Ok(ExportFormat::Tsv),
            #[cfg(feature = "jsonl")]
            crate::format::FormatKind::Jsonl | crate::format::FormatKind::Json => {
                Ok(ExportFormat::Jsonl)
            }
            // `detect` resolves `.parquet` *and* any unknown extension to `Parquet`, so the
            // two are separated here: a literal `.parquet` writes Parquet, anything else
            // falls through to the CSV default described above. Without the sink for the
            // chosen format there is nothing to write with, so say so rather than quietly
            // picking another one.
            #[cfg(feature = "export-parquet")]
            crate::format::FormatKind::Parquet if has_parquet_extension(path) => {
                Ok(ExportFormat::Parquet)
            }
            #[cfg(feature = "csv")]
            crate::format::FormatKind::Parquet => Ok(ExportFormat::Csv),
            _ => err!(UnsupportedFeature),
        },
    }
}

/// Whether `path` ends in `.parquet` (case-insensitively). Used to tell a real Parquet
/// destination apart from the unknown extension that `FormatKind::detect` also reports as
/// `Parquet`.
#[cfg(feature = "export-parquet")]
fn has_parquet_extension(path: &str) -> bool {
    let path = crate::format::strip_url_query(path);
    match path.rfind('.') {
        Some(i) => eq_ascii_ci(path[i + 1..].as_bytes(), b"parquet"),
        None => false,
    }
}

/// Maps a `(FORMAT <name>)` value to an `ExportFormat`.
fn format_by_name(name: &str) -> Result<ExportFormat> {
    #[cfg(feature = "csv")]
    if eq_ascii_ci(name.as_bytes(), b"csv") {
        return Ok(ExportFormat::Csv);
    }
    #[cfg(feature = "csv")]
    if eq_ascii_ci(name.as_bytes(), b"tsv") || eq_ascii_ci(name.as_bytes(), b"tab") {
        return Ok(ExportFormat::Tsv);
    }
    #[cfg(feature = "jsonl")]
    if eq_ascii_ci(name.as_bytes(), b"jsonl") || eq_ascii_ci(name.as_bytes(), b"json") {
        return Ok(ExportFormat::Jsonl);
    }
    #[cfg(feature = "export-parquet")]
    if eq_ascii_ci(name.as_bytes(), b"parquet") {
        return Ok(ExportFormat::Parquet);
    }
    err!(UnsupportedFeature)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Recording sink for tests. Accumulates written rows as a matrix of `Value`.
    struct RecordSink {
        rows: Vec<Vec<Value>>,
        began: bool,
        finished: bool,
    }

    impl RecordSink {
        fn new() -> Self {
            RecordSink { rows: Vec::new(), began: false, finished: false }
        }
    }

    impl TableSink for RecordSink {
        fn begin(&mut self, _schema: &[Field]) -> Result<()> {
            self.began = true;
            Ok(())
        }
        fn write_batch(&mut self, schema: &[Field], batch: &Batch) -> Result<()> {
            for r in 0..batch.num_rows() {
                let mut row = Vec::with_capacity(schema.len());
                for c in &batch.cols {
                    row.push(c.value_at(r));
                }
                self.rows.push(row);
            }
            Ok(())
        }
        fn finish(&mut self) -> Result<Vec<u8>> {
            self.finished = true;
            Ok(Vec::new())
        }
    }

    #[test]
    #[cfg(feature = "csv")]
    fn drives_begin_write_finish_in_order() {
        // CSV can be assembled by hand, so this module can be tested in isolation
        // without depending on other formats' test data or implementations.
        let mut s = Session::new();
        s.register_bytes_as("t", b"id\n1\n2\n3\n".to_vec(), crate::format::FormatKind::Csv)
            .unwrap();
        let mut sink = RecordSink::new();
        export_all(&mut s, "SELECT id FROM t ORDER BY id", &[], &mut sink).unwrap();
        assert!(sink.began);
        assert!(sink.finished);
        assert_eq!(sink.rows.len(), 3);
        assert_eq!(sink.rows[0][0], Value::I64(1));
        assert_eq!(sink.rows[2][0], Value::I64(3));
    }

    #[test]
    fn need_io_fails_clearly_for_unresolved_remote_table() {
        let mut s = Session::new();
        s.register_remote("t", 100).unwrap();
        let mut sink = RecordSink::new();
        let r = export_all(&mut s, "SELECT * FROM t", &[], &mut sink);
        assert_eq!(crate::error::code_of(r), Some(crate::error::Code::IoFailed));
    }

    // --- COPY (integration tests via `Session::prepare`) ---------------------
    // Verifies the whole path from the parser through to `write::copy`. Since
    // `ahiru-core` never writes to a file, what we check here is only that the
    // correct path and correct bytes come back in the `Query` (verifying actual
    // file writes is `ahiru-cli`'s job).

    #[cfg(feature = "csv")]
    fn copy_ready(sql: &str) -> crate::session::CopyResult {
        let mut s = Session::new();
        s.register_bytes_as("t", b"id,name\n2,b\n1,a\n".to_vec(), crate::format::FormatKind::Csv)
            .unwrap();
        match s.prepare(sql, &[]).unwrap() {
            Prepared::Ready(q) => q.copy.expect("no COPY result"),
            Prepared::NeedIo(_) => panic!("unexpected NeedIo"),
        }
    }

    #[test]
    #[cfg(feature = "csv")]
    fn copy_subquery_infers_csv_from_extension() {
        let r = copy_ready("COPY (SELECT id, name FROM t ORDER BY id) TO 'out.csv'");
        assert_eq!(r.path, "out.csv");
        assert_eq!(String::from_utf8(r.data).unwrap(), "id,name\n1,a\n2,b\n");
    }

    #[test]
    #[cfg(feature = "jsonl")]
    fn copy_explicit_format_overrides_extension() {
        let r = copy_ready("COPY (SELECT id, name FROM t ORDER BY id) TO 'out.dat' (FORMAT jsonl)");
        assert_eq!(r.path, "out.dat");
        let text = String::from_utf8(r.data).unwrap();
        assert_eq!(text, "{\"id\":1,\"name\":\"a\"}\n{\"id\":2,\"name\":\"b\"}\n");
    }

    #[test]
    #[cfg(feature = "jsonl")]
    fn copy_format_json_writes_ndjson_like_duckdb() {
        // DuckDB's `COPY ... (FORMAT JSON)` writes newline-delimited JSON (NDJSON)
        // rather than a JSON array (confirmed empirically with the local duckdb
        // CLI). Since v1 has no separate JSON-array sink for writing, mapping to
        // the JSONL sink matches this behavior.
        let r = copy_ready("COPY (SELECT id FROM t ORDER BY id) TO 'out.json'");
        assert_eq!(String::from_utf8(r.data).unwrap(), "{\"id\":1}\n{\"id\":2}\n");
    }

    #[test]
    #[cfg(feature = "csv")]
    fn copy_table_form_is_select_star_from_table() {
        let r = copy_ready("COPY t TO 'out.csv'");
        assert_eq!(String::from_utf8(r.data).unwrap(), "id,name\n2,b\n1,a\n");
    }

    /// Without `export-parquet` there is no Parquet sink, so both the
    /// extension route and the explicit `FORMAT parquet` route have to say
    /// so instead of quietly falling back to some other format.
    #[test]
    #[cfg(all(feature = "csv", not(feature = "export-parquet")))]
    fn copy_unsupported_format_is_rejected() {
        let mut s = Session::new();
        s.register_bytes_as("t", b"id\n1\n".to_vec(), crate::format::FormatKind::Csv).unwrap();
        let r = s.prepare("COPY (SELECT id FROM t) TO 'out.parquet'", &[]);
        assert_eq!(crate::error::code_of(r), Some(crate::error::Code::UnsupportedFeature));

        let r = s.prepare("COPY (SELECT id FROM t) TO 'out.csv' (FORMAT parquet)", &[]);
        assert_eq!(crate::error::code_of(r), Some(crate::error::Code::UnsupportedFeature));
    }

    /// A format name no build ever supports is rejected whatever features
    /// are on (the `export-parquet` counterpart of the test above, which
    /// can no longer use `parquet` as its example).
    #[test]
    #[cfg(feature = "csv")]
    fn copy_unknown_format_name_is_rejected() {
        let mut s = Session::new();
        s.register_bytes_as("t", b"id\n1\n".to_vec(), crate::format::FormatKind::Csv).unwrap();
        let r = s.prepare("COPY (SELECT id FROM t) TO 'out.csv' (FORMAT orc)", &[]);
        assert_eq!(crate::error::code_of(r), Some(crate::error::Code::UnsupportedFeature));
    }

    #[test]
    #[cfg(feature = "csv")]
    fn copy_need_io_fails_clearly_for_unresolved_remote_table() {
        let mut s = Session::new();
        s.register_remote("t", 100).unwrap();
        let r = s.prepare("COPY (SELECT * FROM t) TO 'out.csv'", &[]);
        assert_eq!(crate::error::code_of(r), Some(crate::error::Code::IoFailed));
    }
}
