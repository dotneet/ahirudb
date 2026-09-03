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

/// Each nested query block binds its own WITH list. The local CTE shadows an
/// outer CTE with the same name, while a different local CTE can still refer
/// to an outer definition, matching DuckDB's nested CTE scope rules.
#[test]
fn nested_query_blocks_bind_and_shadow_ctes() {
    let mut db = session();
    assert_eq!(
        run(
            &mut db,
            "SELECT * FROM (WITH inner_cte AS (SELECT c FROM u) SELECT * FROM inner_cte) d",
        ),
        vec![vec![i64(9)]]
    );
    assert_eq!(
        run(
            &mut db,
            "SELECT (WITH inner_cte AS (SELECT c FROM u) SELECT max(c) FROM inner_cte) FROM t",
        ),
        vec![vec![i64(9)]]
    );
    assert_eq!(
        run(
            &mut db,
            "WITH outer_cte AS (SELECT c FROM u) SELECT * FROM \
             (WITH inner_cte AS (SELECT * FROM outer_cte) SELECT * FROM inner_cte) d",
        ),
        vec![vec![i64(9)]]
    );
    assert_eq!(
        run(
            &mut db,
            "WITH same_name AS (SELECT c FROM u) SELECT * FROM \
             (WITH same_name AS (SELECT a FROM t) SELECT * FROM same_name) d",
        ),
        vec![vec![i64(1)]]
    );
    assert_eq!(
        ahiru_core::error::code_of(db.prepare(
            "SELECT * FROM (WITH local_cte AS (SELECT a FROM t), \
             LOCAL_CTE AS (SELECT a FROM t) SELECT * FROM local_cte) d",
            &[],
        )),
        Some(ahiru_core::error::Code::UnsupportedFeature)
    );
}

/// A CTE on the *left* of a join hid the file table on the right.
///
/// The FROM walk's `Join` arm propagated the left side's `TableNotFound` (a
/// CTE name is not in the catalog) with `?` before it ever visited the right
/// side, so `u` was never resolved and binding died with `Internal`. Naming
/// `u` anywhere else in the statement made the identical join work, which is
/// what kept this hidden. Values checked against the `duckdb` CLI.
#[test]
fn cte_on_the_left_of_a_join_still_resolves_the_right_table() {
    let mut db = session();
    assert_eq!(
        run(
            &mut db,
            "WITH agg AS (SELECT a * 9 AS k FROM t) \
             SELECT agg.k, u.c FROM agg JOIN u ON agg.k = u.c",
        ),
        vec![vec![i64(9), i64(9)]]
    );
    assert_eq!(
        run(
            &mut db,
            "WITH agg AS (SELECT a * 9 AS k FROM t) \
             SELECT agg.k, u.c FROM agg LEFT JOIN u ON agg.k = u.c",
        ),
        vec![vec![i64(9), i64(9)]]
    );
    assert_eq!(
        run(
            &mut db,
            "WITH agg AS (SELECT a FROM t) SELECT agg.a, u.c FROM agg, u WHERE agg.a < u.c",
        ),
        vec![vec![i64(1), i64(9)]]
    );
}

/// The same shape inside a recursive CTE: the recursive term lists the
/// working table first, so the join's left side is again a name the catalog
/// does not have. duckdb returns 1..9 for this.
#[test]
fn recursive_term_with_the_working_table_first_resolves_the_joined_table() {
    let mut db = session();
    assert_eq!(
        run(
            &mut db,
            "WITH RECURSIVE r(n) AS (SELECT a FROM t UNION ALL \
             SELECT r.n + 1 FROM r JOIN u ON r.n < u.c) SELECT * FROM r",
        ),
        (1..=9).map(|n| vec![i64(n)]).collect::<Vec<_>>()
    );
}

/// `SELECT x FROM (SELECT x FROM (... FROM t ...))`, nested `n` deep.
fn nested_derived(n: usize) -> String {
    let mut sql = String::from("SELECT x FROM ");
    for _ in 0..n {
        sql.push_str("(SELECT x FROM ");
    }
    sql.push_str("(SELECT a AS x FROM t)");
    for _ in 0..n {
        sql.push(')');
    }
    sql
}

/// A left-deep chain of `n` derived-table joins.
fn join_chain(n: usize) -> String {
    let mut sql = String::from("SELECT d0.x FROM (SELECT a AS x FROM t) d0");
    for i in 1..=n {
        sql.push_str(&format!(" JOIN (SELECT a AS x FROM t) d{i} ON d{i}.x = d0.x"));
    }
    sql
}

/// The pre-bind reference walk used to spend three frames per nesting level
/// against the same 64-unit budget the binder charges one for, so around 20
/// nested derived tables (or 60 join links) it gave up — and the error was
/// discarded, leaving the innermost table unresolved and binding failing with
/// `Internal`. Every depth must now either work or say "too deep";
/// `Internal` is never an acceptable answer to a well-formed query.
#[test]
fn deep_nesting_binds_or_fails_cleanly() {
    // Binding a query nested this deep recurses once per level, and an
    // unoptimized test build spends far more stack per frame than the
    // shipping `wasm` profile does, so the harness's default thread stack is
    // not enough. The stack size is a property of this test, not of the
    // engine: the release CLI runs the same statements on its main thread.
    std::thread::Builder::new()
        .stack_size(32 << 20)
        .spawn(|| {
            let mut db = session();
            // Depths that used to trip the old budget.
            assert_eq!(run(&mut db, &nested_derived(25)), vec![vec![i64(1)]]);
            assert_eq!(run(&mut db, &join_chain(62)), vec![vec![i64(1)]]);
            // Across the parser's cap the only acceptable failure is "too deep".
            for n in 0..=70 {
                for sql in [nested_derived(n), join_chain(n)] {
                    match ahiru_core::error::code_of(db.prepare(&sql, &[]).map(|_| ())) {
                        None | Some(ahiru_core::error::Code::ExpressionTooDeep) => {}
                        other => panic!("depth {n}: unexpected {other:?} for {sql}"),
                    }
                }
            }
        })
        .unwrap()
        .join()
        .unwrap();
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
