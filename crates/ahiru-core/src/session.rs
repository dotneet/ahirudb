//! The session: holds the catalog, takes SQL, and returns batches.
//!
//! Asynchronous I/O is expressed as "stop execution and return the byte ranges needed".
//! Avoiding Asyncify keeps the wasm code size down (DESIGN.md §6).

use crate::catalog::{Catalog, Source, TablePart};
use crate::exec::{build, CodecRequest, ExecContext, IoRequest, Operator, Step, Values};
use crate::expr::vm::Vm;
use crate::format::{partitioned::PartitionedFormat, FormatKind, TableFormat};
use crate::plan::bind::{
    bind_query, desugar_pivot, desugar_unpivot, referenced_in_query, resolve_from,
};
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

pub struct Session {
    pub catalog: Catalog,
    /// `pub(crate)`: used by the `ddl`/`dml` modules for per-row expression evaluation
    /// (VALUES/SET/WHERE). Not exposed outside the crate (no effect on the existing ABI/JS surface).
    pub(crate) vm: Vm,
    /// The query start time for `CURRENT_DATE`/`CURRENT_TIMESTAMP`/`now()`
    /// (microseconds since the epoch, UTC). The wasm core has no clock, so the host
    /// passes it explicitly via `set_now`. Unset, it is the epoch (1970-01-01) --
    /// the same reasoning as the other defensive-parsing choices: if the time is
    /// unknown, return a conspicuously broken value rather than quietly lying.
    now_micros: i64,
}

impl Session {
    pub fn new() -> Self {
        Session { catalog: Catalog::new(), vm: Vm::new(), now_micros: 0 }
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
                p.source.insert(offset, data);
                Ok(())
            }
            None => err!(TableNotFound),
        }
    }

    /// Lowers SQL into a plan. Requests byte ranges and returns if a schema is unresolved.
    pub fn prepare(&mut self, sql: &str, params: &[Value]) -> Result<Prepared> {
        let mut parsed = parse(sql)?;
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
                let plan = bind_query(&self.catalog, &parsed.arena, q, params)?;
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
            return Ok(Prepared::NeedIo(io));
        }
        let plan = bind_query(&self.catalog, arena, q, params)?;
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
    fn resolve_query(
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
                p.source.insert_decoded(offset, len, data);
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
        match q.root.next(&mut ctx)? {
            Step::Ready(b) => Ok(QueryStep::Batch(b)),
            Step::NeedIo => Ok(QueryStep::NeedIo(ctx.io)),
            Step::NeedCodec => Ok(QueryStep::NeedCodec(ctx.codec)),
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
        let table = resolve_from(&self.catalog, from)?;
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
