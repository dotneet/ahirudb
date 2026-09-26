//! DDL execution (the `ddl` feature).
//!
//! `CREATE TABLE` / `CREATE TABLE AS SELECT` / `DROP TABLE` / `CREATE VIEW` /
//! `DROP VIEW`. Every effect lands only on `catalog::MemTable` and the view table (the
//! `(name, SQL)` table inside `Catalog`). The read-only `Source`/`TableFormat` are
//! never touched (DESIGN.md §16).
//!
//! Called directly from `Session::prepare` (DDL/DML are one-shot statements and do not
//! ride the Volcano streaming execution).
//!
//! ## `CREATE TABLE AS SELECT` is not resumable
//!
//! Same reason and same constraint as `write::export_all`: a `NEED_IO`/`NEED_CODEC`
//! during execution fails with `IoFailed`. It can only be used when all the data is in
//! memory (see the `write` module docs).

use crate::error::Code;
use crate::exec::{build, ExecContext, Step};
use crate::plan::bind::bind_query_at;
use crate::plan::compile::{cast_program, compile};
use crate::plan::Scope;
use crate::prelude::*;
use crate::session::{Prepared, Query, Session};
use crate::sql::ast::{AlterTableAction, ColumnDef, ExprArena, ExprId, QueryStmt};
use crate::vector::{Batch, Field, Ty, Value, Vector};

#[allow(clippy::too_many_arguments)]
pub(crate) fn create_table(
    session: &mut Session,
    arena: &ExprArena,
    name: &str,
    or_replace: bool,
    if_not_exists: bool,
    columns: &[ColumnDef],
    as_select: Option<&QueryStmt>,
    params: &[Value],
) -> Result<Prepared> {
    if if_not_exists && !or_replace && table_name_exists(session, name) {
        return Ok(Prepared::Ready(count_result(0)));
    }
    let (schema, rows) = match as_select {
        Some(q) => {
            let (mut schema, rows) = run_query_to_rows(session, arena, q, params)?;
            // A column whose expression is an untyped `NULL` (`SELECT NULL AS n`) has
            // the internal type `Ty::Null`, which is not a storable column type:
            // later INSERTs would keep the raw value while casts, comparisons and
            // COPY treated the column as always-NULL. DuckDB types such a column
            // INTEGER; do the same. Every existing row holds `Value::Null` there, so
            // no data needs converting.
            for f in &mut schema {
                if f.ty == Ty::Null {
                    f.ty = Ty::Int;
                }
            }
            (schema, rows)
        }
        None => {
            let schema =
                columns.iter().map(|c| Field::new(c.name.clone(), c.ty, c.nullable)).collect();
            (schema, Vec::new())
        }
    };
    let n = rows.len();
    let idx = session.catalog.mem_create(name, schema, or_replace)?;
    session.catalog.mem_get_mut(idx).unwrap().rows = rows;
    Ok(Prepared::Ready(count_result(n as i64)))
}

fn table_name_exists(session: &Session, name: &str) -> bool {
    session.catalog.index_of(name).is_some()
        || session.catalog.mem_index_of(name).is_some()
        || session.catalog.view_index_of(name).is_some()
}

pub(crate) fn drop_table(session: &mut Session, name: &str, if_exists: bool) -> Result<Prepared> {
    let existed = table_name_exists(session, name);
    match session.catalog.mem_drop(name) {
        Ok(()) => Ok(Prepared::Ready(count_result(0))),
        Err(e) if if_exists && !existed && e.code == Code::TableNotFound => {
            Ok(Prepared::Ready(count_result(0)))
        }
        Err(e) => Err(e),
    }
}

/// `ALTER TABLE t <action>`. Applies only to `catalog::MemTable`. File-backed tables
/// are rejected with `ReadOnlyTable` by `Catalog::mem_index_writable` (the same rule as
/// `dml::mem_index_writable`).
///
/// The actual rewriting of schema and rows is delegated to `catalog::Catalog`'s
/// `mem_add_column` and friends (the same division of labor as `CREATE TABLE`/`DROP
/// TABLE` delegating to `mem_create`/`mem_drop`). This function only evaluates the
/// DEFAULT expression (which needs the VM) and assembles the affected row count.
pub(crate) fn alter_table(
    session: &mut Session,
    arena: &ExprArena,
    name: &str,
    action: &AlterTableAction,
    params: &[Value],
) -> Result<Prepared> {
    let idx = session.catalog.mem_index_writable(name)?;
    match action {
        AlterTableAction::AddColumn { name: col_name, ty, nullable, default } => {
            add_column(session, arena, idx, col_name, *ty, *nullable, *default, params)?;
        }
        AlterTableAction::DropColumn { name: col_name } => {
            session.catalog.mem_drop_column(idx, col_name)?;
        }
        AlterTableAction::RenameColumn { old, new } => {
            session.catalog.mem_rename_column(idx, old, new)?;
        }
        AlterTableAction::RenameTable { new_name } => {
            session.catalog.mem_rename_table(idx, new_name)?;
        }
    }
    // As with CREATE VIEW/DROP TABLE/DROP VIEW, "affected rows" is meaningless for a
    // statement that only changes the schema, so this always returns 0.
    Ok(Prepared::Ready(count_result(0)))
}

/// `ADD COLUMN col ty [NOT NULL] [DEFAULT expr]`. `DEFAULT` is evaluated exactly once
/// using the existing bytecode VM, in the same pattern as `dml::insert`'s value
/// evaluation (no dedicated scalar evaluator is written), and the same value is
/// appended to every existing row.
///
/// **NOT NULL without DEFAULT**: checking with the `duckdb` CLI, DuckDB rejects a
/// `NOT NULL` constraint on `ADD COLUMN` outright as unsupported (the same with a
/// `DEFAULT`: "Adding columns with constraints not yet supported"). This engine has no
/// reason to reject uniformly -- including combined with a DEFAULT, or when there are
/// zero existing rows so no row actually receives NULL -- so it follows the same rule
/// as `dml::insert`/`dml::update`: an error only when a NOT NULL column actually
/// receives NULL. That is, `TypeMismatch` when the value being appended to the new
/// column (the DEFAULT if present, NULL otherwise) is NULL and there is at least one existing row.
#[allow(clippy::too_many_arguments)]
fn add_column(
    session: &mut Session,
    arena: &ExprArena,
    idx: usize,
    col_name: &str,
    ty: Ty,
    nullable: bool,
    default: Option<ExprId>,
    params: &[Value],
) -> Result<()> {
    let value = match default {
        // Strict, like `INSERT`: a DEFAULT that does not fit the column type (e.g.
        // `TINYINT DEFAULT 1000`, `DATE DEFAULT 'x'`) is a conversion error at DDL time,
        // as in DuckDB, rather than a NULL silently stored into every existing row.
        Some(expr_id) => eval_value_strict(session, arena, expr_id, params, ty)?,
        None => Value::Null,
    };
    let has_rows = !session.catalog.mem_get(idx).unwrap().rows.is_empty();
    ensure!(nullable || !value.is_null() || !has_rows, TypeMismatch);
    session.catalog.mem_add_column(idx, Field::new(col_name, ty, nullable), value)
}

/// Compiles a single expression in an empty scope (no column references), casts it to
/// `target_ty`, and evaluates it against a one-row batch. The cast is the lenient
/// `SELECT` one (an unconvertible value becomes NULL), so anything that *stores* the
/// result must go through [`eval_value_strict`]/[`cast_value`] instead; this is only
/// their building block.
fn eval_scalar(
    session: &mut Session,
    arena: &ExprArena,
    expr_id: ExprId,
    params: &[Value],
    target_ty: Ty,
) -> Result<Value> {
    let scope = Scope::new();
    let prog = compile(arena, &scope, params, expr_id)?;
    let prog = if prog.result_ty != target_ty { cast_program(prog, target_ty)? } else { prog };
    let batch = Batch::rows_only(1);
    let v = session.vm.eval(&prog, &batch)?;
    Ok(v.value_at(0))
}

/// Casts an already-determined `Value` of type `src_ty` to `target_ty`. Wrapping it as
/// an `Expr::TypedLiteral` and running it through `eval_scalar` shares CAST semantics
/// (DECIMAL scale adjustment, DATE/TIMESTAMP, and so on) fully with `SELECT`'s CAST.
///
/// `src_ty` must be the type of the *source column* the value came from, not a type
/// guessed from the `Value` variant. A plain `Expr::Literal` would do the latter, and
/// several logical types are indistinguishable from their physical representation:
/// `DECIMAL(10,2)` 12.50 is `I64(1250)`, which re-infers as `BIGINT 1250` and then
/// rescales to `1250.00`; `UUID`/`INTERVAL`/`DATE`/`TIME`/`TIMESTAMP` likewise lose
/// their identity and become NULL or a `TypeMismatch`.
///
/// The conversion is **strict**: the `Cast` opcode's documented per-row behaviour is
/// "a row that fails to convert becomes NULL" (`expr::kernels`), which is right for
/// `SELECT` but wrong here, where the NULL would be *stored*. DuckDB raises a
/// Conversion Error and stores nothing; a non-NULL value that casts to NULL is
/// therefore rejected with `ValueOutOfRange` rather than silently written. Because
/// both `insert` and `update` validate the entire statement before touching any row,
/// the failing statement mutates nothing.
pub(crate) fn cast_value(
    session: &mut Session,
    v: Value,
    src_ty: Ty,
    target_ty: Ty,
) -> Result<Value> {
    if v.is_null() {
        return Ok(Value::Null);
    }
    // No stored column has type `Ty::Null` (`create_table` maps it to INTEGER), and a
    // non-NULL value cannot be represented in it; refuse rather than store it.
    ensure!(target_ty != Ty::Null, TypeMismatch);
    // Identical types need no conversion at all -- and skipping the VM here keeps the
    // common `INSERT INTO t SELECT * FROM t` path free of per-value program compilation.
    if src_ty == target_ty {
        return Ok(v);
    }
    let mut arena = ExprArena::new();
    let id = arena.push(crate::sql::ast::Expr::TypedLiteral(v, src_ty));
    let out = eval_scalar(session, &arena, id, &[], target_ty)?;
    // `v` was not NULL, so a NULL here can only mean the conversion failed.
    ensure!(!out.is_null(), ValueOutOfRange);
    Ok(out)
}

/// Evaluates a single constant expression (an `INSERT ... VALUES` item or an
/// `ADD COLUMN ... DEFAULT`) and coerces it to the target column type, strictly
/// (see [`cast_value`]).
///
/// The expression is compiled and evaluated at its *natural* type first, so the
/// pre-cast value is known; [`cast_value`] then performs the conversion itself.
/// `eval_scalar` cannot be used directly because it folds the cast into the same
/// program and so cannot tell "the expression was NULL" from "the cast failed".
pub(crate) fn eval_value_strict(
    session: &mut Session,
    arena: &ExprArena,
    expr_id: ExprId,
    params: &[Value],
    target_ty: Ty,
) -> Result<Value> {
    let prog = compile(arena, &Scope::new(), params, expr_id)?;
    let src_ty = prog.result_ty;
    let v = session.vm.eval(&prog, &Batch::rows_only(1))?.value_at(0);
    cast_value(session, v, src_ty, target_ty)
}

pub(crate) fn create_view(
    session: &mut Session,
    name: &str,
    query_sql: String,
    or_replace: bool,
) -> Result<Prepared> {
    session.catalog.view_create(name, query_sql, or_replace)?;
    Ok(Prepared::Ready(count_result(0)))
}

pub(crate) fn drop_view(session: &mut Session, name: &str, if_exists: bool) -> Result<Prepared> {
    let existed = table_name_exists(session, name);
    match session.catalog.view_drop(name) {
        Ok(()) => Ok(Prepared::Ready(count_result(0))),
        Err(e) if if_exists && !existed && e.code == Code::TableNotFound => {
            Ok(Prepared::Ready(count_result(0)))
        }
        Err(e) => Err(e),
    }
}

/// Runs a `SELECT` to completion without resuming and extracts the result as rows.
/// Used by both `CREATE TABLE AS` and `INSERT INTO ... SELECT` (`dml`).
///
/// **Not resumable across the host boundary**: this runs to completion inside
/// `Session::prepare`, so a `NEED_IO` (bytes that were never fetched) gives
/// `IoFailed`. The reads it was waiting on are stashed on the session first,
/// and `Session::prepare` hands them to the host as `Prepared::NeedIo`, so a
/// host that answers them and prepares again gets further each time (the
/// statement restarts from scratch). A `NEED_CODEC` is different — the compressed bytes are already
/// in memory and only need inflating, so it is serviced in place through the
/// session's [`Session::set_codec_hook`] hook, the same way the host services
/// it between two `step` calls. Without a hook registered it is reported as
/// `UnsupportedCodec`.
pub(crate) fn run_query_to_rows(
    session: &mut Session,
    arena: &ExprArena,
    q: &QueryStmt,
    params: &[Value],
) -> Result<(Vec<Field>, Vec<Vec<Value>>)> {
    // Resolve file-backed table schemas first. Anything missing gives IoFailed, since
    // this is not resumable; the reads go to the session for `prepare` to report.
    if let Some(io) = session.resolve_query(arena, q)? {
        session.stash_io(io);
        err!(IoFailed);
    }
    let plan = bind_query_at(&session.catalog, arena, q, params, session.now_micros)?;
    let schema = plan.root.schema().to_vec();
    let mut op = build(plan.root)?;
    let mut rows = Vec::new();
    loop {
        let mut ctx = ExecContext {
            catalog: &mut session.catalog,
            vm: &mut session.vm,
            io: Vec::new(),
            codec: Vec::new(),
        };
        let step = op.next(&mut ctx)?;
        // Ends the `&mut session` borrow held by `ctx` so the codec arm below
        // can hand the requests back to the session.
        let pending = core::mem::take(&mut ctx.codec);
        let io = core::mem::take(&mut ctx.io);
        match step {
            Step::Ready(mut b) => {
                b.materialize();
                for r in 0..b.num_rows() {
                    rows.push(b.cols.iter().map(|c| c.value_at(r)).collect());
                }
            }
            Step::NeedIo => {
                session.stash_io(io);
                err!(IoFailed)
            }
            Step::NeedCodec => session.service_codec(&pending)?,
            Step::Done => break,
        }
    }
    Ok((schema, rows))
}

/// Returns the affected row count and the like as one row, one column (`count`). Used as the DDL/DML completion notice.
pub(crate) fn count_result(n: i64) -> Query {
    let mut v = Vector::with_capacity(Ty::BigInt, 1);
    v.push_value(&Value::I64(n));
    Query::single_batch(vec![Field::new("count", Ty::BigInt, false)], Batch::new(vec![v]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::Prepared;

    fn ready_rows(session: &mut Session, sql: &str) -> Vec<Vec<Value>> {
        let mut q = match session.prepare(sql, &[]).unwrap() {
            Prepared::Ready(q) => q,
            Prepared::NeedIo(_) => panic!("unexpected NeedIo"),
        };
        let mut out = Vec::new();
        loop {
            match session.step(&mut q).unwrap() {
                crate::session::QueryStep::Batch(mut b) => {
                    b.materialize();
                    for r in 0..b.num_rows() {
                        out.push(b.cols.iter().map(|c| c.value_at(r)).collect());
                    }
                }
                crate::session::QueryStep::Done => break,
                _ => panic!("mem table scan should never need io/codec"),
            }
        }
        out
    }

    #[test]
    fn create_table_registers_empty_mem_table() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER, name VARCHAR)", &[]).unwrap();
        assert!(s.table_names().iter().any(|n| n == "t"));
        let rows = ready_rows(&mut s, "SELECT * FROM t");
        assert!(rows.is_empty());
    }

    // CSV is used as the fixture, so `csv` is required (resolving `FormatKind::Csv`
    // gives UnsupportedFeature without it).
    #[cfg(feature = "csv")]
    #[test]
    fn create_table_as_select_materializes_rows() {
        let mut s = Session::new();
        // This engine does not handle `SELECT 1` (without FROM), so CTAS uses a
        // registered table as its source.
        s.register_bytes_as("u", b"x,y\n1,a\n2,b\n".to_vec(), crate::format::FormatKind::Csv)
            .unwrap();
        s.prepare("CREATE TABLE t AS SELECT x, y FROM u WHERE x = 1", &[]).unwrap();
        let rows = ready_rows(&mut s, "SELECT x, y FROM t");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0].as_i64(), Some(1));
    }

    #[test]
    fn drop_table_removes_it() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
        s.prepare("DROP TABLE t", &[]).unwrap();
        assert!(!s.table_names().iter().any(|n| n == "t"));
        assert_eq!(
            crate::error::code_of(s.prepare("SELECT * FROM t", &[])),
            Some(Code::TableNotFound)
        );
    }

    #[test]
    fn drop_table_if_exists_is_noop_when_missing() {
        let mut s = Session::new();
        s.prepare("DROP TABLE IF EXISTS nope", &[]).unwrap();
        assert_eq!(
            crate::error::code_of(s.prepare("DROP TABLE nope", &[])),
            Some(Code::TableNotFound)
        );
    }

    #[test]
    fn drop_if_exists_does_not_hide_a_same_name_object_of_another_kind() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
        s.prepare("CREATE VIEW v AS SELECT id FROM t", &[]).unwrap();

        assert_eq!(
            crate::error::code_of(s.prepare("DROP TABLE IF EXISTS v", &[])),
            Some(Code::TableNotFound)
        );
        assert_eq!(
            crate::error::code_of(s.prepare("DROP VIEW IF EXISTS t", &[])),
            Some(Code::TableNotFound)
        );
        assert!(s.table_names().iter().any(|name| name == "t"));
        assert!(s.table_names().iter().any(|name| name == "v"));
    }

    #[test]
    fn create_view_is_queryable_and_reflects_underlying_table() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (x INTEGER)", &[]).unwrap();
        s.prepare("CREATE VIEW v AS SELECT x FROM t", &[]).unwrap();
        assert!(ready_rows(&mut s, "SELECT x FROM v").is_empty());
        #[cfg(feature = "dml")]
        {
            s.prepare("INSERT INTO t VALUES (1)", &[]).unwrap();
            let rows = ready_rows(&mut s, "SELECT x FROM v");
            assert_eq!(rows, vec![vec![Value::I32(1)]]);
        }
    }

    #[cfg(feature = "csv")]
    #[test]
    fn view_over_unresolved_file_table_is_queryable() {
        let mut s = Session::new();
        s.register_bytes_as("src", b"id\n1\n2\n".to_vec(), crate::format::FormatKind::Csv).unwrap();
        s.prepare("CREATE VIEW v AS SELECT id FROM src", &[]).unwrap();
        assert_eq!(
            ready_rows(&mut s, "SELECT id FROM v ORDER BY id"),
            vec![vec![Value::I64(1)], vec![Value::I64(2)]]
        );
    }

    #[test]
    fn view_does_not_see_an_outer_cte_of_the_same_name_as_its_base_table() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
        #[cfg(feature = "dml")]
        {
            s.prepare("INSERT INTO t VALUES (1), (2)", &[]).unwrap();
        }
        s.prepare("CREATE VIEW v AS SELECT id FROM t", &[]).unwrap();
        let rows = ready_rows(&mut s, "WITH t AS (SELECT 99 AS id FROM range(1)) SELECT * FROM v");
        #[cfg(feature = "dml")]
        {
            assert_eq!(rows, vec![vec![Value::I32(1)], vec![Value::I32(2)]]);
        }
        #[cfg(not(feature = "dml"))]
        {
            assert!(rows.is_empty());
        }
    }

    #[test]
    fn describe_works_on_mem_tables_and_views() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER, name VARCHAR)", &[]).unwrap();
        let rows = ready_rows(&mut s, "DESCRIBE t");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][0], Value::Bytes(b"id".to_vec()));
        s.prepare("CREATE VIEW v AS SELECT id FROM t", &[]).unwrap();
        let rows = ready_rows(&mut s, "DESCRIBE v");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::Bytes(b"id".to_vec()));
    }

    #[test]
    fn create_view_with_a_placeholder_is_rejected() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
        assert_eq!(
            crate::error::code_of(s.prepare("CREATE VIEW v AS SELECT * FROM t WHERE id = ?", &[])),
            Some(Code::UnsupportedFeature)
        );
    }

    #[test]
    fn duplicate_create_table_is_rejected() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
        assert_eq!(
            crate::error::code_of(s.prepare("CREATE TABLE t (id INTEGER)", &[])),
            Some(Code::DuplicateTable)
        );
    }

    // --- ALTER TABLE ---------------------------------------------------------

    #[test]
    #[cfg(feature = "dml")]
    fn alter_table_add_column_fills_default_then_null() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
        s.prepare("INSERT INTO t VALUES (1), (2)", &[]).unwrap();

        s.prepare("ALTER TABLE t ADD COLUMN score INTEGER DEFAULT 7", &[]).unwrap();
        assert_eq!(
            ready_rows(&mut s, "SELECT id, score FROM t ORDER BY id"),
            vec![vec![Value::I32(1), Value::I32(7)], vec![Value::I32(2), Value::I32(7)]]
        );

        // Without a DEFAULT, existing rows are filled with NULL.
        s.prepare("ALTER TABLE t ADD COLUMN note VARCHAR", &[]).unwrap();
        let rows = ready_rows(&mut s, "SELECT note FROM t");
        assert!(rows.iter().all(|r| r[0].is_null()));
    }

    #[test]
    fn alter_table_add_column_default_with_aggregate_is_rejected_not_ice() {
        // DEFAULT is passed straight to `compile()` with `Scope::new()` (an empty scope
        // with no column references), so aggregate functions cannot even be resolved
        // syntactically (`count`/`sum` are only recognized on the binder's aggregate
        // binding path). This confirms it becomes a clear error rather than an internal inconsistency (Internal) or a panic.
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
        let r = s.prepare("ALTER TABLE t ADD COLUMN n INTEGER DEFAULT count(*)", &[]);
        assert!(
            crate::error::code_of(r).is_some(),
            "a DEFAULT containing an aggregate should be a clear error"
        );
    }

    #[test]
    fn alter_table_add_column_rejects_duplicate_name() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
        assert_eq!(
            crate::error::code_of(s.prepare("ALTER TABLE t ADD COLUMN id VARCHAR", &[])),
            Some(Code::DuplicateColumn)
        );
    }

    #[test]
    #[cfg(feature = "dml")]
    fn alter_table_add_not_null_column_without_default_needs_empty_table() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
        // With no rows, adding a NOT NULL column is fine (no row receives NULL).
        s.prepare("ALTER TABLE t ADD COLUMN score INTEGER NOT NULL", &[]).unwrap();

        s.prepare("INSERT INTO t VALUES (1, 10)", &[]).unwrap();
        // With rows present, NOT NULL without a DEFAULT is rejected, since existing rows would become NULL.
        assert_eq!(
            crate::error::code_of(s.prepare("ALTER TABLE t ADD COLUMN note VARCHAR NOT NULL", &[])),
            Some(Code::TypeMismatch)
        );
    }

    #[test]
    #[cfg(feature = "dml")]
    fn alter_table_drop_column_removes_slot_from_every_row() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (a INTEGER, b INTEGER)", &[]).unwrap();
        s.prepare("INSERT INTO t VALUES (1, 10), (2, 20)", &[]).unwrap();

        s.prepare("ALTER TABLE t DROP COLUMN a", &[]).unwrap();
        assert_eq!(
            ready_rows(&mut s, "SELECT b FROM t ORDER BY b"),
            vec![vec![Value::I32(10)], vec![Value::I32(20)]]
        );
        assert_eq!(
            crate::error::code_of(s.prepare("SELECT a FROM t", &[])),
            Some(Code::ColumnNotFound)
        );
    }

    #[test]
    #[cfg(feature = "dml")]
    fn view_referencing_a_dropped_or_renamed_column_fails_cleanly_not_a_panic() {
        // Views are held as raw SQL text and reparsed and rebound on every reference
        // (`catalog::views`). Querying a view after DROP/RENAME of a base table column
        // should simply fail with ColumnNotFound at bind time, and must not panic or
        // leave a dangling reference.
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (a INTEGER, b INTEGER)", &[]).unwrap();
        s.prepare("INSERT INTO t VALUES (1, 10)", &[]).unwrap();
        s.prepare("CREATE VIEW v AS SELECT a, b FROM t", &[]).unwrap();
        assert_eq!(
            ready_rows(&mut s, "SELECT a, b FROM v"),
            vec![vec![Value::I32(1), Value::I32(10)]]
        );

        s.prepare("ALTER TABLE t DROP COLUMN b", &[]).unwrap();
        assert_eq!(
            crate::error::code_of(s.prepare("SELECT a, b FROM v", &[])),
            Some(Code::ColumnNotFound)
        );

        s.prepare("ALTER TABLE t RENAME COLUMN a TO a2", &[]).unwrap();
        assert_eq!(
            crate::error::code_of(s.prepare("SELECT a2 FROM v", &[])),
            Some(Code::ColumnNotFound),
            "the view body still refers to the old column name `a`, so it cannot be queried through the new name `a2`"
        );
    }

    #[test]
    fn alter_table_drop_missing_column_is_column_not_found() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
        assert_eq!(
            crate::error::code_of(s.prepare("ALTER TABLE t DROP COLUMN nope", &[])),
            Some(Code::ColumnNotFound)
        );
    }

    #[test]
    #[cfg(feature = "dml")]
    fn alter_table_rename_column_keeps_data_and_type() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (a INTEGER)", &[]).unwrap();
        s.prepare("INSERT INTO t VALUES (5)", &[]).unwrap();

        s.prepare("ALTER TABLE t RENAME COLUMN a TO b", &[]).unwrap();
        assert_eq!(ready_rows(&mut s, "SELECT b FROM t"), vec![vec![Value::I32(5)]]);
        assert_eq!(
            crate::error::code_of(s.prepare("SELECT a FROM t", &[])),
            Some(Code::ColumnNotFound)
        );
    }

    #[test]
    fn alter_table_rename_column_to_existing_name_is_rejected() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (a INTEGER, b INTEGER)", &[]).unwrap();
        assert_eq!(
            crate::error::code_of(s.prepare("ALTER TABLE t RENAME COLUMN a TO b", &[])),
            Some(Code::DuplicateColumn)
        );
    }

    #[test]
    fn alter_table_rename_to_renames_without_moving_data() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
        s.prepare("ALTER TABLE t RENAME TO u", &[]).unwrap();
        assert!(!s.table_names().iter().any(|n| n == "t"));
        assert!(s.table_names().iter().any(|n| n == "u"));
        assert!(ready_rows(&mut s, "SELECT * FROM u").is_empty());
    }

    #[test]
    fn alter_table_rename_to_existing_table_is_rejected() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
        s.prepare("CREATE TABLE u (id INTEGER)", &[]).unwrap();
        assert_eq!(
            crate::error::code_of(s.prepare("ALTER TABLE t RENAME TO u", &[])),
            Some(Code::DuplicateTable)
        );
    }

    // CSV is used as the fixture, so `csv` is required (resolving `FormatKind::Csv`
    // gives UnsupportedFeature without it).
    #[cfg(feature = "csv")]
    #[test]
    fn alter_table_on_file_backed_table_is_read_only() {
        let mut s = Session::new();
        s.register_bytes_as("t", b"id\n1\n".to_vec(), crate::format::FormatKind::Csv).unwrap();
        assert_eq!(
            crate::error::code_of(s.prepare("ALTER TABLE t ADD COLUMN x INTEGER", &[])),
            Some(Code::ReadOnlyTable)
        );
        assert_eq!(
            crate::error::code_of(s.prepare("ALTER TABLE t RENAME TO u", &[])),
            Some(Code::ReadOnlyTable)
        );
    }
}
