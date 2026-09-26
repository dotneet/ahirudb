//! The session: holds the catalog, takes SQL, and returns batches.
//!
//! Asynchronous I/O is expressed as "stop execution and return the byte ranges needed".
//! Avoiding Asyncify keeps the wasm code size down (DESIGN.md §6).

use crate::catalog::{Catalog, Source, TablePart};
use crate::exec::{build, CodecRequest, ExecContext, IoRequest, Operator, Step, Values};
use crate::expr::vm::Vm;
use crate::format::{partitioned::PartitionedFormat, FormatKind, TableFormat};
use crate::plan::bind::{bind_query_at, desugar_pivot, desugar_unpivot, referenced_in_query};
use crate::prelude::*;
use crate::sql::ast::{FromItem, Stmt};
use crate::sql::parse;
use crate::vector::{Batch, Field, Ty, Value, Vector};

/// The bytes `COPY` produced, plus the path it was told to write to.
///
/// `ahiru-core` is `no_std` and cannot touch the filesystem, so actually writing
/// `data` to `path` is the caller's job (`ahiru-cli` on native)
/// (see the `write` module docs and DESIGN.md §15).
#[cfg(feature = "export")]
pub struct CopyResult {
    pub path: String,
    pub data: Vec<u8>,
}

/// A prepared query.
pub struct Query {
    root: Box<dyn Operator>,
    pub schema: Vec<Field>,
    /// The result of executing `COPY` (the `export` feature). When it is `Some`,
    /// `root`/`schema` are meaningless empty placeholders and `step` need not be
    /// called. The real data is here.
    #[cfg(feature = "export")]
    pub copy: Option<CopyResult>,
}

impl Query {
    /// Builds a result of exactly one predetermined batch.
    /// Used by `ddl`/`dml` to return a completion notice (affected row count and the
    /// like), the same trick as `one_column`/`describe_result` for `SHOW TABLES`/`DESCRIBE`.
    #[cfg(feature = "ddl")]
    pub(crate) fn single_batch(schema: Vec<Field>, batch: Batch) -> Self {
        Query {
            root: Box::new(Values::new(batch)),
            schema,
            #[cfg(feature = "export")]
            copy: None,
        }
    }

    /// Builds a `Query` that merely holds a `COPY` result.
    /// Called from `write::copy`.
    #[cfg(feature = "export")]
    pub(crate) fn copy_result(path: String, data: Vec<u8>) -> Self {
        Query {
            root: Box::new(Values::new(Batch::new(Vec::new()))),
            schema: Vec::new(),
            copy: Some(CopyResult { path, data }),
        }
    }
}

/// The result of `Session::prepare`.
pub enum Prepared {
    Ready(Query),
    /// Not enough bytes to read the footer.
    NeedIo(Vec<IoRequest>),
}

/// The result of `Session::step`.
pub enum QueryStep {
    Batch(Batch),
    NeedIo(Vec<IoRequest>),
    /// Asks the host to decompress a codec that is not built in (DESIGN.md §6).
    NeedCodec(Vec<CodecRequest>),
    Done,
}

/// A host-supplied decompressor for codecs the core does not carry.
///
/// `NEED_CODEC` is normally answered by the host between two `step` calls
/// (DESIGN.md §6). The one-shot statements — `COPY`, `CREATE TABLE AS`,
/// `INSERT ... SELECT` — run to completion *inside* `prepare`, so there is no
/// step boundary at which the host could be re-entered; without a hook they
/// can only fail. Registering one lets those paths service the request
/// in-process, using bytes the preceding `NEED_IO` already delivered.
///
/// It is a plain `fn` pointer, not a boxed closure: it costs one word on
/// `Session` and needs no allocation, which matters for the wasm budget
/// (DESIGN.md §3). The source bytes are handed in by the engine, so the hook
/// never has to reach back into the catalog.
///
/// A host that has no such decompressor simply leaves it unset, and those
/// statements report `UnsupportedCodec` — the same behaviour as before, but
/// with an error that names the actual problem.
#[cfg(any(feature = "ddl", feature = "export"))]
pub type CodecHook =
    fn(codec: crate::parquet::Compression, src: &[u8], out_len: usize) -> Result<Vec<u8>>;

pub struct Session {
    pub catalog: Catalog,
    /// `pub(crate)`: used by the `ddl`/`dml` modules for per-row expression evaluation
    /// (VALUES/SET/WHERE). Not exposed outside the crate (no effect on the existing ABI/JS surface).
    pub(crate) vm: Vm,
    /// The host decompressor used by the non-streaming statements. See [`CodecHook`].
    #[cfg(any(feature = "ddl", feature = "export"))]
    codec_hook: Option<CodecHook>,
    /// The query start time for `CURRENT_DATE`/`CURRENT_TIMESTAMP`/`now()`
    /// (microseconds since the epoch, UTC). The wasm core has no clock, so the host
    /// passes it explicitly via `set_now`. Unset, it is the epoch (1970-01-01) --
    /// the same reasoning as the other defensive-parsing choices: if the time is
    /// unknown, return a conspicuously broken value rather than quietly lying.
    pub(crate) now_micros: i64,
    /// The reads a one-shot statement (`COPY`, CTAS, `INSERT ... SELECT`) was
    /// waiting on when it gave up with `IoFailed`. Those statements run to
    /// completion inside `prepare`, so they cannot pause for I/O; `prepare` turns
    /// the failure into `Prepared::NeedIo` with these reads instead, and a host
    /// that answers them and calls `prepare` again gets further each time
    /// (see [`Session::stash_io`]).
    #[cfg(any(feature = "ddl", feature = "export"))]
    pending_io: Vec<IoRequest>,
}

impl Session {
    pub fn new() -> Self {
        Session {
            catalog: Catalog::new(),
            vm: Vm::new(),
            #[cfg(any(feature = "ddl", feature = "export"))]
            codec_hook: None,
            now_micros: 0,
            #[cfg(any(feature = "ddl", feature = "export"))]
            pending_io: Vec::new(),
        }
    }

    /// Remembers the reads a one-shot statement is about to fail on (see
    /// `pending_io`). Called right before such a statement returns `IoFailed`.
    #[cfg(any(feature = "ddl", feature = "export"))]
    pub(crate) fn stash_io(&mut self, io: Vec<IoRequest>) {
        self.pending_io = io;
    }

    /// Registers the host decompressor described by [`CodecHook`].
    ///
    /// Only the one-shot statements (`COPY`, `CREATE TABLE AS`,
    /// `INSERT ... SELECT`) consult it; the streaming `prepare`/`step` path is
    /// unaffected and keeps returning `NEED_CODEC` to the host as before.
    #[cfg(any(feature = "ddl", feature = "export"))]
    pub fn set_codec_hook(&mut self, hook: CodecHook) {
        self.codec_hook = Some(hook);
    }

    /// Answers a batch of `NEED_CODEC` requests in-process via the registered
    /// [`CodecHook`], caching each result exactly as `provide_decoded` would.
    ///
    /// The compressed bytes were already delivered by the preceding `NEED_IO`
    /// (DESIGN.md §6), so they are read straight out of the `Source` rather
    /// than asked for again.
    #[cfg(any(feature = "ddl", feature = "export"))]
    pub(crate) fn service_codec(&mut self, reqs: &[CodecRequest]) -> Result<()> {
        let hook = match self.codec_hook {
            Some(hook) => hook,
            None => err!(UnsupportedCodec),
        };
        for r in reqs {
            let part = match self.catalog.get(r.table).and_then(|t| t.parts.get(r.part)) {
                Some(part) => part,
                None => err!(TableNotFound),
            };
            let src = match part.source.get(r.offset, r.len as usize) {
                Some(src) => src,
                // The bytes should already be here; if they are not, this is a
                // genuine I/O gap rather than a codec problem.
                None => err!(IoFailed),
            };
            let data = hook(r.codec, src, r.out_len as usize)?;
            self.provide_decoded(r.table, r.part, r.offset, r.len, data)?;
        }
        Ok(())
    }

    /// Sets the query start time. Later `prepare` calls use it as the value of
    /// `CURRENT_DATE`/`CURRENT_TIMESTAMP`/`CURRENT_TIME`/`now()`/`today()`
    /// The host (JS/CLI) is expected to call it with the current time on every query
    /// (DESIGN.md §2, "do on the host what the host can do").
    pub fn set_now(&mut self, now_micros: i64) {
        self.now_micros = now_micros;
    }

    /// Registers a table whose whole file is held in memory.
    /// The format is inferred from the name (its extension).
    pub fn register_bytes(&mut self, name: &str, bytes: Vec<u8>) -> Result<usize> {
        self.register_bytes_as(name, bytes, FormatKind::Auto)
    }

    pub fn register_bytes_as(
        &mut self,
        name: &str,
        bytes: Vec<u8>,
        kind: FormatKind,
    ) -> Result<usize> {
        self.catalog.register(name, Source::from_bytes(bytes), kind)
    }

    /// Registers a table the host supplies via range fetching. No I/O happens.
    pub fn register_remote(&mut self, name: &str, total_len: u64) -> Result<usize> {
        self.register_remote_as(name, total_len, FormatKind::Auto)
    }

    pub fn register_remote_as(
        &mut self,
        name: &str,
        total_len: u64,
        kind: FormatKind,
    ) -> Result<usize> {
        self.catalog.register(name, Source::remote(total_len), kind)
    }

    /// Registers a host-served table under a path a SQL string literal named
    /// (case-sensitive; see `Catalog::register_path`). No I/O happens.
    pub fn register_remote_path(
        &mut self,
        path: &str,
        total_len: u64,
        kind: FormatKind,
    ) -> Result<usize> {
        self.catalog.register_path(path, Source::remote(total_len), kind)
    }

    /// Answers a size request (`IoRequest` with offset `catalog::SIZE_REQUEST`)
    /// for a table declared with `catalog::SIZE_UNKNOWN`.
    pub fn set_size(&mut self, table: usize, part: usize, total_len: u64) -> Result<()> {
        match self.catalog.get_mut(table) {
            Some(t) => t.set_size(part, total_len),
            None => err!(TableNotFound),
        }
    }

    /// Drops the raw bytes fetched for host-served tables. The host calls this
    /// (through the ABI) once no query is running; the next query asks for what
    /// it needs again, which the host answers from its own range cache.
    pub fn release_fetched(&mut self) {
        self.catalog.release();
    }

    /// Registers several files as one logical table by handing over their bytes.
    ///
    /// `files` is a sequence of `(path, bytes)`. `path` is used both for automatic
    /// format detection (the extension) and for extracting Hive partition columns
    /// (`key=value` directories). Parts whose path has no `key=value` segment are left
    /// alone, and only those that do are wrapped in `PartitionedFormat` -- wrapping
    /// every part uniformly would permanently insert a pointless indirection in the
    /// majority case where there are no partitions.
    pub fn register_multi_bytes(
        &mut self,
        name: &str,
        files: Vec<(String, Vec<u8>)>,
        kind: FormatKind,
    ) -> Result<usize> {
        let files: Vec<(String, Source)> =
            files.into_iter().map(|(p, b)| (p, Source::from_bytes(b))).collect();
        self.register_multi(name, files, kind)
    }

    /// Registers several files as one logical table, served by the host's range fetching.
    /// `files` is a sequence of `(path, total byte length)`. No I/O happens.
    pub fn register_multi_remote(
        &mut self,
        name: &str,
        files: Vec<(String, u64)>,
        kind: FormatKind,
    ) -> Result<usize> {
        let files: Vec<(String, Source)> =
            files.into_iter().map(|(p, len)| (p, Source::remote(len))).collect();
        self.register_multi(name, files, kind)
    }

    fn register_multi(
        &mut self,
        name: &str,
        files: Vec<(String, Source)>,
        kind: FormatKind,
    ) -> Result<usize> {
        ensure!(!files.is_empty(), Internal);
        let mut parts = Vec::with_capacity(files.len());
        for (path, source) in files {
            let inner = crate::format::make(kind, &path)?;
            let hive_cols = PartitionedFormat::parse_hive_path(&path);
            let format: Box<dyn TableFormat> = if hive_cols.is_empty() {
                inner
            } else {
                Box::new(PartitionedFormat::new(inner, hive_cols))
            };
            parts.push(TablePart { path, source, format });
        }
        self.catalog.register_multi(name, parts)
    }

    /// Hands over the bytes requested by `NeedIo`.
    pub fn provide(&mut self, table: usize, part: usize, offset: u64, data: Vec<u8>) -> Result<()> {
        let t = match self.catalog.get_mut(table) {
            Some(t) => t,
            None => err!(TableNotFound),
        };
        match t.parts.get_mut(part) {
            Some(p) => {
                // Hosts must not be able to install bytes beyond the file's declared
                // length.  Without this check a malformed offset/length was silently
                // retained by `Source`, leaving the requested in-file range missing
                // forever (or, for an offset near `u64::MAX`, reaching overflowing range
                // arithmetic while chunks were coalesced).
                let data_len = data.len() as u64;
                let end = match offset.checked_add(data_len) {
                    Some(end) => end,
                    None => err!(ValueOutOfRange),
                };
                ensure!(offset <= p.source.total_len && end <= p.source.total_len, ValueOutOfRange);
                p.source.insert(offset, data);
                Ok(())
            }
            None => err!(TableNotFound),
        }
    }

    /// Lowers SQL into a plan. Requests byte ranges and returns if a schema is unresolved.
    pub fn prepare(&mut self, sql: &str, params: &[Value]) -> Result<Prepared> {
        // Both are per-statement: a path miss or a stashed read left over from an
        // earlier statement must not be reported for this one.
        drop(self.catalog.take_missing());
        #[cfg(any(feature = "ddl", feature = "export"))]
        self.pending_io.clear();
        let r = self.prepare_stmt(sql, params);
        #[cfg(any(feature = "ddl", feature = "export"))]
        if let Err(e) = &r {
            if e.code == crate::error::Code::IoFailed && !self.pending_io.is_empty() {
                return Ok(Prepared::NeedIo(core::mem::take(&mut self.pending_io)));
            }
        }
        r
    }

    fn prepare_stmt(&mut self, sql: &str, params: &[Value]) -> Result<Prepared> {
        let mut parsed = parse(sql)?;
        // Placeholders are numbered by appearance, so the highest index + 1 is the
        // number of values the statement takes. Too few is caught where a
        // placeholder is compiled (`WrongArgCount`); too many was silently ignored,
        // which hides a caller binding its values to the wrong statement.
        let used = parsed
            .arena
            .iter_mut()
            .filter_map(|e| match e {
                crate::sql::ast::Expr::Param(i) => Some(*i as usize + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        ensure!(params.len() <= used, WrongArgCount);
        // `CURRENT_DATE`/`CURRENT_TIMESTAMP`/`now()` and friends are replaced with
        // constants before binding (see `sql::now`). This also matches the SQL standard's
        // contract of evaluating them exactly once per query.
        crate::sql::substitute_now(&mut parsed.arena, self.now_micros);

        // `PIVOT`/`UNPIVOT` are syntactic sugar, so they are expanded into an ordinary
        // `SELECT` before joining the branches below (see `plan::bind::desugar_pivot`/
        // `desugar_unpivot`). Expansion sometimes needs the target table's schema (when
        // `GROUP BY` is omitted, for instance), so schema resolution and expansion are
        // both finished here, leaving the large `match &parsed.stmt` below untouched
        // (`Stmt::Pivot`/`Stmt::Unpivot` consume `PivotStmt`/`UnpivotStmt` by ownership,
        // so they cannot be built from a borrow of `&parsed.stmt` -- `mem::replace`
        // extracts just `parsed.stmt`).
        if matches!(parsed.stmt, Stmt::Pivot(_) | Stmt::Unpivot(_)) {
            let stmt = core::mem::replace(&mut parsed.stmt, Stmt::ShowTables);
            let q = match stmt {
                Stmt::Pivot(p) => {
                    let from_schema = if p.group_by.is_empty() {
                        match self.describe(&p.from)? {
                            Ok(f) => f,
                            Err(io) => return Ok(Prepared::NeedIo(io)),
                        }
                    } else {
                        Vec::new()
                    };
                    desugar_pivot(&mut parsed.arena, *p, &from_schema)?
                }
                Stmt::Unpivot(u) => {
                    let from_schema = match self.describe(&u.from)? {
                        Ok(f) => f,
                        Err(io) => return Ok(Prepared::NeedIo(io)),
                    };
                    desugar_unpivot(&mut parsed.arena, *u, &from_schema)?
                }
                _ => unreachable!(),
            };
            return self.prepare_query(&parsed.arena, &q, params);
        }

        match &parsed.stmt {
            Stmt::Select(q) => self.prepare_query(&parsed.arena, q, params),
            // EXPLAIN builds the plan and then renders it as text. It does not execute.
            Stmt::Explain(q) => {
                if let Some(io) = self.resolve_query(&parsed.arena, q)? {
                    return Ok(Prepared::NeedIo(io));
                }
                let plan = bind_query_at(&self.catalog, &parsed.arena, q, params, self.now_micros)?;
                let lines = crate::plan::explain::explain(&plan.root);
                Ok(Prepared::Ready(one_column("plan", lines)))
            }
            Stmt::Describe(from) => {
                let fields = match self.describe(from)? {
                    Ok(f) => f,
                    Err(io) => return Ok(Prepared::NeedIo(io)),
                };
                Ok(Prepared::Ready(describe_result(&fields)))
            }
            Stmt::ShowTables => Ok(Prepared::Ready(one_column("name", self.table_names()))),
            // Always already consumed by the early return above (`Stmt::Pivot`/`Stmt::Unpivot`
            // are expanded before joining `prepare_query`, so they never reach here).
            Stmt::Pivot(_) | Stmt::Unpivot(_) => unreachable!(),
            // DDL/DML are one-shot statements with side effects (catalog changes) and do
            // not ride the Volcano streaming execution. The `ddl`/`dml` modules finish
            // them here and return the result as a one-row `Query` (affected row count
            // and so on) -- the same "drive the existing public path from outside" idea
            // as `export::export_all`, except this is inside Session, so it calls directly.
            #[cfg(feature = "ddl")]
            Stmt::CreateTable { name, or_replace, if_not_exists, columns, as_select } => {
                crate::ddl::create_table(
                    self,
                    &parsed.arena,
                    name,
                    *or_replace,
                    *if_not_exists,
                    columns,
                    as_select.as_deref(),
                    params,
                )
            }
            #[cfg(feature = "ddl")]
            Stmt::DropTable { name, if_exists } => crate::ddl::drop_table(self, name, *if_exists),
            #[cfg(feature = "ddl")]
            Stmt::AlterTable { name, action } => {
                crate::ddl::alter_table(self, &parsed.arena, name, action, params)
            }
            #[cfg(feature = "ddl")]
            Stmt::CreateView { name, or_replace, query_sql } => {
                crate::ddl::create_view(self, name, query_sql.clone(), *or_replace)
            }
            #[cfg(feature = "ddl")]
            Stmt::DropView { name, if_exists } => crate::ddl::drop_view(self, name, *if_exists),
            #[cfg(feature = "dml")]
            Stmt::Insert { table, columns, source } => {
                crate::dml::insert(self, &parsed.arena, table, columns, source, params)
            }
            #[cfg(feature = "dml")]
            Stmt::Update { table, assignments, filter } => {
                crate::dml::update(self, &parsed.arena, table, assignments, *filter, params)
            }
            #[cfg(feature = "dml")]
            Stmt::Delete { table, filter } => {
                crate::dml::delete(self, &parsed.arena, table, *filter, params)
            }
            // `COPY` is a one-shot statement like DDL/DML, but its side effect is
            // assembling bytes rather than changing the catalog. Actually writing the file
            // happens outside `ahiru-core` (see the `write` module docs and DESIGN.md §15).
            #[cfg(feature = "export")]
            Stmt::Copy { query, path, format } => {
                crate::write::copy(self, &parsed.arena, query, path, format.as_deref(), params)
            }
        }
    }

    /// Binds a `SELECT` and lowers it into a plan. The shared path used both by
    /// `Stmt::Select` and by callers holding an AST directly (such as the inner query of
    /// `COPY`, which wants to avoid a round trip through a SQL string; see `write::export_query`).
    pub(crate) fn prepare_query(
        &mut self,
        arena: &crate::sql::ast::ExprArena,
        q: &crate::sql::ast::QueryStmt,
        params: &[Value],
    ) -> Result<Prepared> {
        if let Some(io) = self.resolve_query(arena, q)? {
            // `COPY` drives this directly and turns `NeedIo` into `IoFailed`.
            #[cfg(feature = "export")]
            self.stash_io(io.clone());
            return Ok(Prepared::NeedIo(io));
        }
        let plan = bind_query_at(&self.catalog, arena, q, params, self.now_micros)?;
        let schema = plan.root.schema().to_vec();
        Ok(Prepared::Ready(Query {
            root: build(plan.root)?,
            schema,
            #[cfg(feature = "export")]
            copy: None,
        }))
    }

    /// Resolves the schemas of every table the query references.
    /// Any missing ranges are returned together (there can be several with joins, set
    /// operations, or multi-file tables). To keep it to one round trip per table, the
    /// ranges every part of that table needs are bundled at `Table::resolve` before being
    /// collected (see the docs on `catalog::Table::resolve`).
    pub(crate) fn resolve_query(
        &mut self,
        arena: &crate::sql::ast::ExprArena,
        q: &crate::sql::ast::QueryStmt,
    ) -> Result<Option<Vec<IoRequest>>> {
        let mut tables = Vec::new();
        referenced_in_query(&self.catalog, arena, q, &mut tables, 0)?;
        let mut io = Vec::new();
        for &table in &tables {
            let t = match self.catalog.get_mut(table) {
                Some(t) => t,
                None => err!(TableNotFound),
            };
            if let Err(need) = t.resolve()? {
                for (part, offset, len) in need {
                    io.push(IoRequest { table, part, offset, len });
                }
            }
        }
        Ok(if io.is_empty() { None } else { Some(io) })
    }

    /// Hands over a compressed block the host decompressed.
    pub fn provide_decoded(
        &mut self,
        table: usize,
        part: usize,
        offset: u64,
        len: u32,
        data: Vec<u8>,
    ) -> Result<()> {
        let t = match self.catalog.get_mut(table) {
            Some(t) => t,
            None => err!(TableNotFound),
        };
        match t.parts.get_mut(part) {
            Some(p) => {
                let end = match offset.checked_add(len as u64) {
                    Some(end) => end,
                    None => err!(ValueOutOfRange),
                };
                ensure!(offset <= p.source.total_len && end <= p.source.total_len, ValueOutOfRange);
                p.source.insert_decoded(offset, len, data)?;
                Ok(())
            }
            None => err!(TableNotFound),
        }
    }

    /// Pulls the next batch.
    pub fn step(&mut self, q: &mut Query) -> Result<QueryStep> {
        let mut ctx = ExecContext {
            catalog: &mut self.catalog,
            vm: &mut self.vm,
            io: Vec::new(),
            codec: Vec::new(),
        };
        let step = q.root.next(&mut ctx)?;
        let (io, codec) = (ctx.io, ctx.codec);
        match step {
            Step::Ready(b) => Ok(QueryStep::Batch(b)),
            Step::NeedIo => {
                // `COPY` steps through here and gives up with `IoFailed`; keep
                // the reads so `prepare` can hand them to the host instead.
                #[cfg(feature = "export")]
                self.stash_io(io.clone());
                Ok(QueryStep::NeedIo(io))
            }
            Step::NeedCodec => Ok(QueryStep::NeedCodec(codec)),
            Step::Done => Ok(QueryStep::Done),
        }
    }

    /// Enumerates table names (`SHOW TABLES`). In-memory tables and views are included (`ddl`).
    pub fn table_names(&self) -> Vec<String> {
        #[allow(unused_mut)]
        let mut names: Vec<String> = self.catalog.names().map(String::from).collect();
        #[cfg(feature = "ddl")]
        {
            names.extend(self.catalog.mem_names().map(String::from));
            names.extend(self.catalog.view_names().map(String::from));
        }
        names
    }

    /// For `DESCRIBE`. Returns requests if the footer is unresolved
    /// (bundled across parts for a multi-file table).
    pub fn describe(
        &mut self,
        from: &FromItem,
    ) -> Result<core::result::Result<Vec<Field>, Vec<IoRequest>>> {
        match from {
            FromItem::Table { name, .. } => {
                if let Some(i) = self.catalog.index_of(name) {
                    return self.describe_file_table(i);
                }
                #[cfg(feature = "ddl")]
                {
                    if let Some(i) = self.catalog.mem_index_of(name) {
                        let schema = match self.catalog.mem_get(i) {
                            Some(t) => t.schema.clone(),
                            None => err!(TableNotFound),
                        };
                        return Ok(Ok(schema));
                    }
                    if let Some(i) = self.catalog.view_index_of(name) {
                        return self.describe_view(i);
                    }
                }
                err!(TableNotFound)
            }
            FromItem::File { path, format, .. } => {
                let i = match self.catalog.path_index_of(path, *format) {
                    Some(i) => i,
                    None => err!(TableNotFound),
                };
                self.describe_file_table(i)
            }
            FromItem::Join { .. }
            | FromItem::Subquery { .. }
            | FromItem::Unnest { .. }
            | FromItem::GenerateSeries { .. } => err!(UnsupportedFeature),
        }
    }

    fn describe_file_table(
        &mut self,
        table: usize,
    ) -> Result<core::result::Result<Vec<Field>, Vec<IoRequest>>> {
        let t = match self.catalog.get_mut(table) {
            Some(t) => t,
            None => err!(TableNotFound),
        };
        if let Err(need) = t.resolve()? {
            let io = need
                .into_iter()
                .map(|(part, offset, len)| IoRequest { table, part, offset, len })
                .collect();
            return Ok(Err(io));
        }
        Ok(Ok(t.schema().to_vec()))
    }

    #[cfg(feature = "ddl")]
    fn describe_view(
        &mut self,
        i: usize,
    ) -> Result<core::result::Result<Vec<Field>, Vec<IoRequest>>> {
        let sql = match self.catalog.view_get(i) {
            Some(s) => s.to_owned(),
            None => err!(TableNotFound),
        };
        let mut parsed = parse(&sql)?;
        crate::sql::substitute_now(&mut parsed.arena, self.now_micros);
        let q = match &parsed.stmt {
            Stmt::Select(q) => q,
            _ => err!(Internal),
        };
        if let Some(io) = self.resolve_query(&parsed.arena, q)? {
            return Ok(Err(io));
        }
        let plan = bind_query_at(&self.catalog, &parsed.arena, q, &[], self.now_micros)?;
        Ok(Ok(plan.root.schema().to_vec()))
    }
}

/// Builds a result of a single string column. For `SHOW TABLES` and `EXPLAIN`.
fn one_column(name: &str, rows: Vec<String>) -> Query {
    let mut v = Vector::with_capacity(Ty::Varchar, rows.len());
    for r in &rows {
        v.push_value(&Value::Bytes(r.as_bytes().to_vec()));
    }
    let schema = vec![Field::new(name, Ty::Varchar, false)];
    Query {
        root: Box::new(Values::new(Batch::new(vec![v]))),
        schema,
        #[cfg(feature = "export")]
        copy: None,
    }
}

/// The result of `DESCRIBE`. Three columns: name, type, and nullability.
fn describe_result(fields: &[Field]) -> Query {
    let mut names = Vector::with_capacity(Ty::Varchar, fields.len());
    let mut types = Vector::with_capacity(Ty::Varchar, fields.len());
    let mut nulls = Vector::with_capacity(Ty::Varchar, fields.len());
    for f in fields {
        names.push_value(&Value::Bytes(f.name.as_bytes().to_vec()));
        types.push_value(&Value::Bytes(f.ty.name().as_bytes().to_vec()));
        nulls.push_value(&Value::Bytes(if f.nullable { b"YES".to_vec() } else { b"NO".to_vec() }));
    }
    let schema = vec![
        Field::new("column_name", Ty::Varchar, false),
        Field::new("column_type", Ty::Varchar, false),
        Field::new("null", Ty::Varchar, false),
    ];
    Query {
        root: Box::new(Values::new(Batch::new(vec![names, types, nulls]))),
        schema,
        #[cfg(feature = "export")]
        copy: None,
    }
}

impl Default for Session {
    fn default() -> Self {
        Session::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{code_of, Code};
    use crate::format::FormatKind;

    #[test]
    fn provide_rejects_ranges_outside_declared_file_length() {
        let mut session = Session::new();
        let table = session.register_remote_as("t.parquet", 10, FormatKind::Parquet).unwrap();

        assert_eq!(code_of(session.provide(table, 0, 9, vec![1, 2])), Some(Code::ValueOutOfRange));
        assert_eq!(
            code_of(session.provide(table, 0, u64::MAX, vec![1])),
            Some(Code::ValueOutOfRange)
        );
        assert_eq!(
            code_of(session.provide_decoded(table, 0, 9, 2, vec![1])),
            Some(Code::ValueOutOfRange)
        );
        // A valid in-file range remains accepted after rejected deliveries.
        assert!(session.provide(table, 0, 9, vec![1]).is_ok());
    }

    #[test]
    fn extra_parameters_are_rejected_like_missing_ones() {
        let mut s = Session::new();
        let one = [Value::I64(1)];
        let two = [Value::I64(1), Value::I64(2)];
        assert!(s.prepare("SELECT ? FROM range(1)", &one).is_ok());
        assert_eq!(code_of(s.prepare("SELECT ? FROM range(1)", &two)), Some(Code::WrongArgCount));
        assert_eq!(code_of(s.prepare("SELECT 1 FROM range(1)", &one)), Some(Code::WrongArgCount));
        assert_eq!(
            code_of(s.prepare("SELECT ?, ? FROM range(1)", &one)),
            Some(Code::WrongArgCount)
        );
        assert!(s.prepare("SELECT 1 FROM range(1)", &[]).is_ok());
    }

    /// Answers every `NeedIo` from `bytes` (sizes included) until `prepare` is ready.
    #[cfg(feature = "csv")]
    fn prepare_serving(s: &mut Session, sql: &str, bytes: &[u8]) -> (Query, usize) {
        let mut rounds = 0;
        loop {
            match s.prepare(sql, &[]).unwrap() {
                Prepared::Ready(q) => return (q, rounds),
                Prepared::NeedIo(io) => {
                    for r in io {
                        if r.offset == crate::catalog::SIZE_REQUEST {
                            s.set_size(r.table, r.part, bytes.len() as u64).unwrap();
                        } else {
                            let (o, l) = (r.offset as usize, r.len as usize);
                            s.provide(r.table, r.part, r.offset, bytes[o..o + l].to_vec()).unwrap();
                        }
                    }
                }
            }
            rounds += 1;
            assert!(rounds < 20, "no progress");
        }
    }

    #[test]
    #[cfg(feature = "csv")]
    fn a_table_declared_without_a_size_asks_for_it_then_reads() {
        let csv = b"id\n1\n2\n";
        let mut s = Session::new();
        s.register_remote_as("t.csv", crate::catalog::SIZE_UNKNOWN, FormatKind::Auto).unwrap();
        let (mut q, rounds) = prepare_serving(&mut s, "SELECT sum(id) FROM \"t.csv\"", csv);
        assert!(rounds >= 2, "a size round and a sample round");
        let mut total = 0;
        loop {
            match s.step(&mut q).unwrap() {
                QueryStep::Batch(b) => total += b.num_rows(),
                QueryStep::NeedIo(io) => {
                    for r in io {
                        let (o, l) = (r.offset as usize, r.len as usize);
                        s.provide(r.table, r.part, r.offset, csv[o..o + l].to_vec()).unwrap();
                    }
                }
                QueryStep::NeedCodec(_) => panic!("csv has no codec"),
                QueryStep::Done => break,
            }
        }
        assert_eq!(total, 1);

        // Releasing the fetched bytes keeps the resolved schema, and the next
        // query simply asks for the data again.
        s.release_fetched();
        let (mut q, _) = prepare_serving(&mut s, "SELECT id FROM \"t.csv\"", csv);
        assert!(matches!(s.step(&mut q).unwrap(), QueryStep::NeedIo(_)));
    }

    #[test]
    #[cfg(all(feature = "csv", feature = "ddl"))]
    fn create_table_as_over_a_remote_table_reports_its_reads_instead_of_failing() {
        let csv = b"id\n1\n2\n3\n";
        let mut s = Session::new();
        s.register_remote_as("t.csv", csv.len() as u64, FormatKind::Auto).unwrap();
        let (_, rounds) =
            prepare_serving(&mut s, "CREATE TABLE m AS SELECT id FROM \"t.csv\"", csv);
        assert!(rounds >= 1);
        let m = s.catalog.mem_index_of("m").unwrap();
        assert_eq!(s.catalog.mem_get(m).unwrap().rows.len(), 3);
    }

    #[test]
    fn a_missing_path_is_remembered_for_the_host() {
        let mut s = Session::new();
        let r = s.prepare("SELECT * FROM parquet('http://h/Data.parquet')", &[]);
        assert_eq!(code_of(r), Some(Code::TableNotFound));
        let missing = s.catalog.take_missing();
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].0, "http://h/Data.parquet");
        assert!(missing[0].1 == FormatKind::Parquet);
        // A later statement starts with a clean list.
        assert!(s.prepare("SELECT 1 FROM range(1)", &[]).is_ok());
        assert!(s.catalog.take_missing().is_empty());
    }
}
