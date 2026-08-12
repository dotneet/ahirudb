//! Cardinality tests for scalar subqueries: `SELECT (SELECT ... FROM t)`.
//!
//! A scalar subquery must produce at most one row. Zero rows is a valid `NULL` (standard SQL);
//! exactly one row yields that value; two or more rows is a cardinality error -- DuckDB raises
//! "More than one row returned by a subquery used as an expression" for this. Earlier, this
//! engine silently kept the first row instead of erroring on 2+ rows (`Node::Limit { limit:
//! Some(1), .. }` truncated the subquery's plan with no check), which meant a query with an
//! unintentionally multi-row scalar subquery returned a different answer than the SQL asked
//! for, without any indication anything was wrong.
//!
//! These tests use file-backed CSV tables (`register_bytes_as`) rather than the `ddl`/`dml`
//! in-memory tables used in `correlated_subqueries.rs`, so they compile and run under the
//! crate's default features (no `--features ddl,dml` needed) -- matching how this test binary
//! is expected to be run (`cargo test -p ahiru-core subquery`).

use ahiru_core::error::{code_of, Code};
use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn session_with(tables: &[(&str, &[u8])]) -> Session {
    let mut s = Session::new();
    for (name, bytes) in tables {
        s.register_bytes_as(name, bytes.to_vec(), FormatKind::Csv).unwrap();
    }
    s
}

/// Runs `sql` to completion. All data is registered in memory, so a `NeedIo` here would
/// itself be a bug.
fn run(s: &mut Session, sql: &str) -> ahiru_core::error::Result<Vec<Vec<Value>>> {
    let mut q = match s.prepare(sql, &[])? {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
    };
    let mut rows = Vec::new();
    loop {
        match s.step(&mut q)? {
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
    Ok(rows)
}

fn i64(v: i64) -> Value {
    Value::I64(v)
}
const NULL: Value = Value::Null;

// --- Uncorrelated scalar subquery, in the SELECT list -------------------------------

#[test]
fn scalar_subquery_two_rows_is_a_cardinality_error() {
    let mut db = session_with(&[("t", b"id\n1\n2\n")]);
    let err = run(&mut db, "SELECT (SELECT id FROM t ORDER BY id) FROM t WHERE id = 1");
    assert_eq!(code_of(err), Some(Code::MultipleRowsSubquery));
}

/// The exact repro from the bug report: an explicit `LIMIT 2` still leaves 2 rows for the
/// scalar subquery to return, which must still be a cardinality error, not "take row 1 of 2".
/// (`FROM range(1)` is just the required anchor -- there is no bare `SELECT <expr>` without a
/// `FROM` clause; see docs/sql/queries.md.)
#[test]
fn scalar_subquery_two_rows_via_an_explicit_limit_is_still_a_cardinality_error() {
    let mut db = session_with(&[("t", b"id\n0\n1\n2\n")]);
    let err = run(&mut db, "SELECT (SELECT id FROM t ORDER BY id LIMIT 2) FROM range(1)");
    assert_eq!(code_of(err), Some(Code::MultipleRowsSubquery));
}

/// More than two rows (so the `Limit(2)` bound the executor adds internally must not itself
/// leak into the observed cardinality: this must fail the same way as exactly two rows).
#[test]
fn scalar_subquery_many_rows_is_the_same_cardinality_error() {
    let mut db = session_with(&[("t", b"id\n1\n2\n3\n4\n5\n")]);
    let err = run(&mut db, "SELECT (SELECT id FROM t) FROM range(1)");
    assert_eq!(code_of(err), Some(Code::MultipleRowsSubquery));
}

#[test]
fn scalar_subquery_exactly_one_row_yields_that_value() {
    let mut db = session_with(&[("t", b"id\n1\n2\n3\n")]);
    let rows = run(&mut db, "SELECT (SELECT id FROM t WHERE id = 2) FROM range(1)").unwrap();
    assert_eq!(rows, vec![vec![i64(2)]]);
}

#[test]
fn scalar_subquery_zero_rows_yields_null() {
    let mut db = session_with(&[("t", b"id\n1\n2\n3\n")]);
    let rows = run(&mut db, "SELECT (SELECT id FROM t WHERE id = 999) FROM range(1)").unwrap();
    assert_eq!(rows, vec![vec![NULL]]);
}

// --- Same check, other syntactic positions -------------------------------------------
//
// The fix operates on every `Expr::ScalarSubquery`, collected uniformly by
// `plan::bind::subquery::collect_scalar_subqueries` regardless of where it appears in the
// expression tree, so a subquery buried in a `WHERE` clause or a function argument gets the
// same cardinality check as one directly in the SELECT list.

#[test]
fn scalar_subquery_two_rows_in_a_where_clause_is_a_cardinality_error() {
    let mut db = session_with(&[("t", b"id\n1\n2\n"), ("u", b"id\n10\n20\n")]);
    let err = run(&mut db, "SELECT id FROM t WHERE id < (SELECT id FROM u)");
    assert_eq!(code_of(err), Some(Code::MultipleRowsSubquery));
}

#[test]
fn scalar_subquery_one_row_in_a_where_clause_still_filters_correctly() {
    let mut db = session_with(&[("t", b"id\n1\n2\n3\n"), ("u", b"id\n2\n")]);
    let rows = run(&mut db, "SELECT id FROM t WHERE id < (SELECT id FROM u) ORDER BY id").unwrap();
    assert_eq!(rows, vec![vec![i64(1)]]);
}

#[test]
fn scalar_subquery_two_rows_as_a_function_argument_is_a_cardinality_error() {
    let mut db = session_with(&[("t", b"id\n-1\n-2\n")]);
    let err = run(&mut db, "SELECT abs((SELECT id FROM t)) FROM range(1)");
    assert_eq!(code_of(err), Some(Code::MultipleRowsSubquery));
}

#[test]
fn scalar_subquery_one_row_as_a_function_argument_still_computes_correctly() {
    let mut db = session_with(&[("t", b"id\n-5\n")]);
    let rows = run(&mut db, "SELECT abs((SELECT id FROM t)) FROM range(1)").unwrap();
    assert_eq!(rows, vec![vec![i64(5)]]);
}

// --- Correlated scalar subquery -------------------------------------------------------
//
// Same cardinality rule, scoped per correlation key: two-or-more rows for the *same* outer
// row's correlation key is an error, even though other outer rows' subqueries only ever
// produce one row each.

#[test]
fn correlated_scalar_subquery_two_rows_for_one_key_is_a_cardinality_error() {
    // customer 1 has two orders, customer 2 has exactly one.
    let mut db = session_with(&[
        ("customers", b"id\n1\n2\n"),
        ("orders", b"id,customer_id,amount\n10,1,100\n11,1,50\n12,2,200\n"),
    ]);
    let err = run(
        &mut db,
        "SELECT c.id, (SELECT o.amount FROM orders o WHERE o.customer_id = c.id) \
         FROM customers c ORDER BY c.id",
    );
    assert_eq!(code_of(err), Some(Code::MultipleRowsSubquery));
}

/// Regression: a correlated scalar subquery that never produces more than one row per key
/// (the common, correct case) still works exactly as before.
#[test]
fn correlated_scalar_subquery_one_row_per_key_is_unaffected() {
    let mut db = session_with(&[
        ("customers", b"id\n1\n2\n3\n"),
        ("orders", b"id,customer_id,amount\n10,1,100\n12,2,200\n"),
    ]);
    let rows = run(
        &mut db,
        "SELECT c.id, (SELECT o.amount FROM orders o WHERE o.customer_id = c.id) \
         FROM customers c ORDER BY c.id",
    )
    .unwrap();
    assert_eq!(
        rows,
        vec![
            vec![i64(1), i64(100)],
            vec![i64(2), i64(200)],
            vec![i64(3), NULL], // no matching order -> NULL, not an error
        ]
    );
}
