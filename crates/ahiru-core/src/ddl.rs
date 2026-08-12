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
use crate::plan::bind::{bind_query, referenced_in_query};
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
        Some(q) => run_query_to_rows(session, arena, q, params)?,
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
    match session.catalog.mem_drop(name) {
        Ok(()) => Ok(Prepared::Ready(count_result(0))),
        Err(e) if if_exists && e.code == Code::TableNotFound => {
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
        Some(expr_id) => eval_scalar(session, arena, expr_id, params, ty)?,
        None => Value::Null,
    };
    let has_rows = !session.catalog.mem_get(idx).unwrap().rows.is_empty();
    ensure!(nullable || !value.is_null() || !has_rows, TypeMismatch);
    session.catalog.mem_add_column(idx, Field::new(col_name, ty, nullable), value)
}

/// Compiles a single expression in an empty scope (no column references), casts it to
/// `target_ty`, and evaluates it against a one-row batch. Shared with `dml` (value
/// evaluation for `INSERT ... VALUES`, and value-level type conversion).
pub(crate) fn eval_scalar(
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
    match session.catalog.view_drop(name) {
        Ok(()) => Ok(Prepared::Ready(count_result(0))),
        Err(e) if if_exists && e.code == Code::TableNotFound => {
            Ok(Prepared::Ready(count_result(0)))
        }
        Err(e) => Err(e),
    }
}

/// Runs a `SELECT` to completion without resuming and extracts the result as rows.
/// Used by both `CREATE TABLE AS` and `INSERT INTO ... SELECT` (`dml`).
///
/// **Not resumable**: see the module docs. A `NEED_IO`/`NEED_CODEC` during schema
/// resolution or scanning gives `IoFailed`.
pub(crate) fn run_query_to_rows(
    session: &mut Session,
    arena: &ExprArena,
    q: &QueryStmt,
    params: &[Value],
) -> Result<(Vec<Field>, Vec<Vec<Value>>)> {
    // Resolve file-backed table schemas first. Anything missing gives IoFailed, since
    // this is not resumable (a simplified version of what `resolve_query` does in
    // `Session::prepare`).
    let mut tables = Vec::new();
    referenced_in_query(&session.catalog, arena, q, &mut tables, 0)?;
    for t in tables {
        if let Some(table) = session.catalog.get_mut(t) {
            if table.resolve()?.is_err() {
                err!(IoFailed);
            }
        }
    }
    let plan = bind_query(&session.catalog, arena, q, params)?;
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
        match op.next(&mut ctx)? {
            Step::Ready(mut b) => {
                b.materialize();
                for r in 0..b.num_rows() {
                    rows.push(b.cols.iter().map(|c| c.value_at(r)).collect());
                }
            }
            Step::NeedIo | Step::NeedCodec => err!(IoFailed),
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
