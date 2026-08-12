//! Regression tests for schema resolution of tables that a query only
//! reaches through a subquery inside an *expression*.
//!
//! `Session::prepare` resolves the schema of every table a query touches
//! before binding it (`plan::bind::referenced_in_query` collects them,
//! `resolve_query` resolves them). `plan::bind::from::push_table_rel` then
//! asserts that invariant with an `Internal` error, so a table the collector
//! misses does not degrade gracefully — it surfaces to the user as a bare
//! "internal error" (E900) with nothing actionable in it.
//!
//! The collector used to walk only the `FROM` clause, so any table reachable
//! *only* through a subquery in an expression was missed:
//!
//! ```sql
//! SELECT (SELECT max(c) FROM u) FROM t   -- u never resolved -> E900
//! SELECT a FROM t WHERE a IN (SELECT c FROM u)
//! SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u)
//! SELECT a FROM t ORDER BY (SELECT max(c) FROM u)
//! ```
//!
//! The same query with `u` also named in `FROM` worked, which is why this
//! went unnoticed: the `FROM` walk resolved it as a side effect.
//!
//! These tests deliberately use *file-backed* tables (registered CSV bytes).
//! In-memory `ddl` tables are always already resolved, so they cannot
//! reproduce the bug. Expected values were checked against the `duckdb` CLI.

use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

/// `t(a, b)` = (1, 2) and `u(c)` = (9), both file-backed.
fn session() -> Session {
    let mut s = Session::new();
    s.register_bytes_as("t", b"a,b\n1,2\n".to_vec(), FormatKind::Csv).unwrap();
    s.register_bytes_as("u", b"c\n9\n".to_vec(), FormatKind::Csv).unwrap();
    s
}

/// Runs `sql` to completion. All bytes are in memory, so a `NeedIo` here
/// would itself be a bug.
fn run(s: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    let mut q = match s.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
    };
    let mut rows = Vec::new();
    loop {
        match s.step(&mut q).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
            QueryStep::Batch(mut b) => {
                b.materialize();
                for r in 0..b.num_rows() {
                    rows.push(b.cols.iter().map(|c| c.value_at(r)).collect());
                }
            }
            QueryStep::Done => break,
            QueryStep::NeedIo(_) | QueryStep::NeedCodec(_) => {
                panic!("{sql}: unexpected NeedIo/NeedCodec")
            }
        }
    }
    rows
}

fn i64(v: i64) -> Value {
    Value::I64(v)
}

/// duckdb: `SELECT (SELECT max(c) FROM u) FROM t` → 9.
#[test]
fn scalar_subquery_over_table_absent_from_outer_from() {
    let mut db = session();
    assert_eq!(run(&mut db, "SELECT (SELECT max(c) FROM u) FROM t"), vec![vec![i64(9)]]);
}

/// The select-list position is not special — the same subquery alongside an
/// ordinary column has to resolve `u` too.
#[test]
fn scalar_subquery_beside_a_plain_column() {
    let mut db = session();
    assert_eq!(run(&mut db, "SELECT a, (SELECT max(c) FROM u) FROM t"), vec![vec![i64(1), i64(9)]]);
}

/// duckdb: `SELECT a FROM t WHERE a < (SELECT max(c) FROM u)` → 1.
#[test]
fn scalar_subquery_in_where() {
    let mut db = session();
    assert_eq!(
        run(&mut db, "SELECT a FROM t WHERE a < (SELECT max(c) FROM u)"),
        vec![vec![i64(1)]]
    );
}

/// duckdb: `EXISTS` over a table only named inside the subquery → 1 row.
#[test]
fn exists_subquery_over_absent_table() {
    let mut db = session();
    assert_eq!(run(&mut db, "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u)"), vec![vec![i64(1)]]);
}

/// duckdb: `1 IN (9)` is false, so this is 0 rows — the point of the test is
/// that it *binds*, not that it matches.
#[test]
fn in_subquery_over_absent_table() {
    let mut db = session();
    assert!(run(&mut db, "SELECT a FROM t WHERE a IN (SELECT c FROM u)").is_empty());
    assert_eq!(
        run(&mut db, "SELECT a FROM t WHERE a NOT IN (SELECT c FROM u)"),
        vec![vec![i64(1)]]
    );
}

/// `= ANY` takes the same semijoin path as `IN`.
#[test]
fn quantified_comparison_over_absent_table() {
    let mut db = session();
    assert!(run(&mut db, "SELECT a FROM t WHERE a = ANY (SELECT c FROM u)").is_empty());
}

/// `ORDER BY` lives outside the select list and had its own gap.
#[test]
fn scalar_subquery_in_order_by() {
    let mut db = session();
    assert_eq!(run(&mut db, "SELECT a FROM t ORDER BY (SELECT max(c) FROM u)"), vec![vec![i64(1)]]);
}

/// A `GenerateSeries` outer FROM resolves no tables at all, so the subquery
/// was the query's only route to `t`. This is the shape the native CLI
/// produces when it rewrites a FROM-less `SELECT` as `... FROM range(1)`.
#[test]
fn scalar_subquery_with_generate_series_outer_from() {
    let mut db = session();
    assert_eq!(run(&mut db, "SELECT (SELECT max(a) FROM t) FROM range(1)"), vec![vec![i64(1)]]);
    assert_eq!(
        run(&mut db, "SELECT (SELECT max(a) FROM t) FROM generate_series(1, 2)"),
        vec![vec![i64(1)], vec![i64(1)]]
    );
}

/// The walk has to reach through derived tables, CTEs and set operations,
/// since each nests another `QueryStmt`.
#[test]
fn nested_query_forms_reach_the_subquery() {
    let mut db = session();
    assert_eq!(
        run(&mut db, "SELECT * FROM (SELECT (SELECT max(c) FROM u) AS z FROM t) s"),
        vec![vec![i64(9)]]
    );
    assert_eq!(
        run(&mut db, "WITH w AS (SELECT (SELECT max(c) FROM u) AS z FROM t) SELECT * FROM w"),
        vec![vec![i64(9)]]
    );
    assert_eq!(
        run(&mut db, "SELECT (SELECT max(c) FROM u) FROM t UNION ALL SELECT a FROM t"),
        vec![vec![i64(9)], vec![i64(1)]]
    );
}

/// A table named *only* inside a subquery still has to be reported as
/// missing with `TableNotFound`, not silently ignored.
#[test]
fn unknown_table_inside_subquery_is_still_an_error() {
    let mut db = session();
    let r = db.prepare("SELECT (SELECT max(c) FROM nope) FROM t", &[]);
    assert_eq!(
        ahiru_core::error::code_of(r.map(|_| ())),
        Some(ahiru_core::error::Code::TableNotFound)
    );
}
