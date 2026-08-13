//! Integration tests for correlated subqueries.
//!
//! Creates `customers`/`orders` as in-memory tables (the `ddl`/`dml` features), and verifies
//! correlated scalar subqueries, correlated `EXISTS`/`NOT EXISTS`, and correlated
//! `IN`/`NOT IN`. All expected values are decided by cross-checking against the actual
//! output of `duckdb -c "SELECT ..."`.
//!
//! Also verifies that patterns outside the supported scope (non-equality correlation,
//! correlation inside `OR`, a correlated `NOT IN` where NULL is possible, a correlated
//! aggregate subquery with its own GROUP BY, and correlation nested 2+ levels deep) are
//! rejected with a clear error rather than panicking.

#![cfg(feature = "dml")]

use ahiru_core::error::{code_of, Code};
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

/// Runs `sql` and extracts the result as `Vec<Vec<Value>>`.
/// Since only in-memory tables are used, `NeedIo`/`NeedCodec` never occur.
fn run(session: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    let mut q = match session.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
    };
    let mut rows = Vec::new();
    loop {
        match session.step(&mut q).unwrap() {
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

fn i32(v: i32) -> Value {
    Value::I32(v)
}
fn i64(v: i64) -> Value {
    Value::I64(v)
}
fn f64(v: f64) -> Value {
    Value::F64(v)
}
fn s(v: &str) -> Value {
    Value::Bytes(v.as_bytes().to_vec())
}
const NULL: Value = Value::Null;

/// Creates customers(id, name, region) / orders(id, customer_id, amount, region), with the
/// same content verified with DuckDB:
///
/// ```text
/// customers: (1,alice,east) (2,bob,west) (3,carol,east) (4,dave,NULL)
/// orders:    (10,1,100.0,east) (11,1,50.0,NULL) (12,2,200.0,west) (13,NULL,10.0,east)
/// ```
///
/// alice has 2 orders (east, including one with a NULL correlation target), bob has 1,
/// and carol/dave have 0.
fn session_with_customers_orders() -> Session {
    let mut s = Session::new();
    s.prepare("CREATE TABLE customers (id INT, name VARCHAR, region VARCHAR)", &[]).unwrap();
    s.prepare("CREATE TABLE orders (id INT, customer_id INT, amount DOUBLE, region VARCHAR)", &[])
        .unwrap();
    s.prepare(
        "INSERT INTO customers VALUES \
         (1, 'alice', 'east'), (2, 'bob', 'west'), (3, 'carol', 'east'), (4, 'dave', NULL)",
        &[],
    )
    .unwrap();
    s.prepare(
        "INSERT INTO orders VALUES \
         (10, 1, 100.0, 'east'), (11, 1, 50.0, NULL), (12, 2, 200.0, 'west'), \
         (13, NULL, 10.0, 'east')",
        &[],
    )
    .unwrap();
    s
}

// --- Correlated scalar subqueries -----------------------------------------------

/// A correlated scalar subquery without aggregation: picks the "first row" per correlation key.
/// duckdb: `SELECT c.id, c.name, (SELECT o.amount FROM orders o WHERE
/// o.customer_id = c.id AND o.id = 10) FROM customers c ORDER BY c.id`
#[test]
fn correlated_scalar_subquery_picks_first_row_per_key() {
    let mut db = session_with_customers_orders();
    let rows = run(
        &mut db,
        "SELECT c.id, c.name, \
         (SELECT o.amount FROM orders o WHERE o.customer_id = c.id AND o.id = 10) \
         FROM customers c ORDER BY c.id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i32(1), s("alice"), f64(100.0)],
            vec![i32(2), s("bob"), NULL],
            vec![i32(3), s("carol"), NULL],
            vec![i32(4), s("dave"), NULL],
        ]
    );
}

/// A correlated scalar subquery with aggregation (`max`). The basic form of magic
/// decorrelation (GROUP BY on the correlation key, then LEFT JOIN).
/// duckdb: `SELECT c.id, c.name, (SELECT max(o.amount) FROM orders o
/// WHERE o.customer_id = c.id) FROM customers c ORDER BY c.id`
#[test]
fn correlated_scalar_subquery_aggregate_max() {
    let mut db = session_with_customers_orders();
    let rows = run(
        &mut db,
        "SELECT c.id, c.name, (SELECT max(o.amount) FROM orders o WHERE o.customer_id = c.id) \
         FROM customers c ORDER BY c.id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i32(1), s("alice"), f64(100.0)],
            vec![i32(2), s("bob"), f64(200.0)],
            vec![i32(3), s("carol"), NULL],
            vec![i32(4), s("dave"), NULL],
        ]
    );
}

/// A correlated scalar subquery with aggregation (`count`) returns 0 rather than NULL when
/// there is no matching inner row (confirmed with DuckDB; if you naively merge the
/// correlation key into the GROUP BY, "that group doesn't exist" turns into NULL via the
/// LEFT JOIN, so COUNT alone needs a correction).
/// duckdb: `SELECT c.id, c.name, (SELECT count(*) FROM orders o WHERE
/// o.customer_id = c.id) FROM customers c ORDER BY c.id`
#[test]
fn correlated_scalar_subquery_count_defaults_to_zero_not_null() {
    let mut db = session_with_customers_orders();
    let rows = run(
        &mut db,
        "SELECT c.id, c.name, (SELECT count(*) FROM orders o WHERE o.customer_id = c.id) \
         FROM customers c ORDER BY c.id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i32(1), s("alice"), i64(2)],
            vec![i32(2), s("bob"), i64(1)],
            vec![i32(3), s("carol"), i64(0)],
            vec![i32(4), s("dave"), i64(0)],
        ]
    );
}

// --- Correlated EXISTS / NOT EXISTS -----------------------------------------

/// Even when a correlated `EXISTS` has an extra non-correlated condition mixed in
/// (`o.amount > 60`), only the correlated equality predicate is extracted as the join key,
/// and the rest is evaluated normally as the inner WHERE.
/// duckdb: `SELECT c.id, c.name FROM customers c WHERE EXISTS
/// (SELECT 1 FROM orders o WHERE o.customer_id = c.id AND o.amount > 60)
/// ORDER BY c.id`
#[test]
fn correlated_exists_with_extra_local_predicate() {
    let mut db = session_with_customers_orders();
    let rows = run(
        &mut db,
        "SELECT c.id, c.name FROM customers c WHERE EXISTS \
         (SELECT 1 FROM orders o WHERE o.customer_id = c.id AND o.amount > 60) \
         ORDER BY c.id",
    );
    assert_eq!(rows, vec![vec![i32(1), s("alice")], vec![i32(2), s("bob")]]);
}

/// Correlated `NOT EXISTS`.
/// duckdb: `SELECT c.id, c.name FROM customers c WHERE NOT EXISTS
/// (SELECT 1 FROM orders o WHERE o.customer_id = c.id) ORDER BY c.id`
#[test]
fn correlated_not_exists() {
    let mut db = session_with_customers_orders();
    let rows = run(
        &mut db,
        "SELECT c.id, c.name FROM customers c WHERE NOT EXISTS \
         (SELECT 1 FROM orders o WHERE o.customer_id = c.id) ORDER BY c.id",
    );
    assert_eq!(rows, vec![vec![i32(3), s("carol")], vec![i32(4), s("dave")]]);
}

// --- Correlated IN / NOT IN ---------------------------------------------------

/// Correlated `IN`. Becomes a semi-join on the composite key of `IN`'s target column
/// (`customer_id`) and the correlation key (`region`).
/// duckdb: `SELECT c.id, c.name FROM customers c WHERE c.id IN
/// (SELECT o.customer_id FROM orders o WHERE o.region = c.region) ORDER BY c.id`
#[test]
fn correlated_in() {
    let mut db = session_with_customers_orders();
    let rows = run(
        &mut db,
        "SELECT c.id, c.name FROM customers c WHERE c.id IN \
         (SELECT o.customer_id FROM orders o WHERE o.region = c.region) ORDER BY c.id",
    );
    assert_eq!(rows, vec![vec![i32(1), s("alice")], vec![i32(2), s("bob")]]);
}

/// A correlated `NOT IN` needs a three-valued NULL judgment scoped per correlation key, but
/// the existing `AntiNullAware` (`NOT IN`'s NULL handling) only judges over the whole join.
/// There is also no way to precisely determine the target column's NULL possibility at bind
/// time (a SELECT list's output column is always treated as `nullable = true`), so it could
/// return a wrong result regardless of whether correlation is present. Rather than return an
/// ambiguous result, always reject clearly.
#[test]
fn correlated_not_in_is_always_rejected() {
    let mut db = session_with_customers_orders();
    let err = db.prepare(
        "SELECT c.id FROM customers c WHERE c.id NOT IN \
         (SELECT o.customer_id FROM orders o WHERE o.region = c.region \
          AND o.customer_id IS NOT NULL)",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

// --- Clear rejection of unsupported patterns ---------------------------------

/// A non-equality correlation predicate (`>`) cannot be extracted into a join key. Rejected clearly.
#[test]
fn non_equality_correlation_is_rejected() {
    let mut db = session_with_customers_orders();
    let err = db.prepare(
        "SELECT c.id FROM customers c WHERE EXISTS \
         (SELECT 1 FROM orders o WHERE o.customer_id > c.id)",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// If a correlation reference sits inside an `OR`, AND-decomposition cannot extract it. Rejected clearly.
#[test]
fn correlation_inside_or_is_rejected() {
    let mut db = session_with_customers_orders();
    let err = db.prepare(
        "SELECT c.id FROM customers c WHERE EXISTS \
         (SELECT 1 FROM orders o WHERE o.customer_id = c.id OR o.amount > 1000)",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// A correlated aggregate subquery with its own GROUP BY is clearly rejected, since there is
/// no way to merge the correlation key into that grouping (silently ignoring the correlation
/// would let the aggregate mix across outer rows, producing a wrong result).
#[test]
fn correlated_distinct_on_is_rejected() {
    let mut db = session_with_customers_orders();
    let err = db.prepare(
        "SELECT c.id FROM customers c WHERE EXISTS \
         (SELECT DISTINCT ON (o.region) o.region FROM orders o WHERE o.customer_id = c.id)",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

#[test]
fn correlated_aggregate_with_own_group_by_is_rejected() {
    let mut db = session_with_customers_orders();
    let err = db.prepare(
        "SELECT c.id, (SELECT count(*) FROM orders o WHERE o.customer_id = c.id \
         GROUP BY o.region) FROM customers c",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// Correlation nested 2+ levels deep (the innermost level reaches past the middle level to
/// reference the outermost column) only propagates one level of outer scope, so it does not
/// `panic`; it fails as an error (failing in the form of "column not found").
#[test]
fn two_level_correlation_skip_fails_cleanly() {
    let mut db = session_with_customers_orders();
    let err = db.prepare(
        "SELECT c.id FROM customers c WHERE EXISTS ( \
           SELECT 1 FROM orders o1 WHERE EXISTS ( \
             SELECT 1 FROM orders o2 WHERE o2.id = c.id \
           ) \
         )",
        &[],
    );
    // Must not panic; must fail with some clear error code.
    assert!(err.is_err());
}

// --- Regression check for uncorrelated subqueries -----------------------------

/// A subquery with no correlation (does not reference the outer scope) still works as before.
#[test]
fn uncorrelated_subqueries_are_unaffected() {
    let mut db = session_with_customers_orders();
    let rows = run(&mut db, "SELECT (SELECT max(amount) FROM orders) FROM customers WHERE id = 1");
    assert_eq!(rows, vec![vec![f64(200.0)]]);

    let rows =
        run(&mut db, "SELECT id FROM customers WHERE EXISTS (SELECT 1 FROM orders) ORDER BY id");
    assert_eq!(rows.len(), 4);

    let rows = run(
        &mut db,
        "SELECT id FROM customers WHERE id IN (SELECT customer_id FROM orders) ORDER BY id",
    );
    assert_eq!(rows, vec![vec![i32(1)], vec![i32(2)]]);
}
