//! Integration tests for `WINDOW name AS (...)` / `OVER name`.
//!
//! All expected values are decided by cross-checking against the actual output of
//! `duckdb -c "SELECT ..."` (`tests/data/basic.parquet` is a real file written by DuckDB. Columns are
//! `id INTEGER, name VARCHAR, score DOUBLE, flag BOOLEAN, big BIGINT,
//! d TIMESTAMP`). Read-only Parquet is all that's needed, so this always runs under the default
//! `cargo test` features too.

use ahiru_core::error::{code_of, Code};
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn data(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn session_with_basic() -> Session {
    let mut s = Session::new();
    s.register_bytes("t", data("basic.parquet")).unwrap();
    s
}

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
fn b(v: bool) -> Value {
    Value::Bool(v)
}

/// Multiple window functions share the same named definition.
/// duckdb:
/// SELECT id, flag, sum(score) OVER w AS s, avg(score) OVER w AS a
/// FROM 'basic.parquet' WHERE id < 6
/// WINDOW w AS (PARTITION BY flag ORDER BY id) ORDER BY id
#[test]
fn named_window_shared_by_multiple_calls() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT id, flag, sum(score) OVER w AS s, avg(score) OVER w AS a \
         FROM t WHERE id < 6 \
         WINDOW w AS (PARTITION BY flag ORDER BY id) ORDER BY id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i32(0), b(true), f64(0.0), f64(0.0)],
            vec![i32(1), b(false), f64(1.5), f64(1.5)],
            vec![i32(2), b(false), f64(4.5), f64(2.25)],
            vec![i32(3), b(true), f64(4.5), f64(2.25)],
            vec![i32(4), b(false), f64(10.5), f64(3.5)],
            vec![i32(5), b(false), f64(18.0), f64(4.5)],
        ]
    );
}

/// A named reference (`OVER w`) and an inline spec (`OVER (...)`) can be used together in the same query.
/// duckdb:
/// SELECT id, flag, row_number() OVER w AS rn, count(*) OVER () AS total
/// FROM 'basic.parquet' WHERE id < 6
/// WINDOW w AS (PARTITION BY flag ORDER BY id) ORDER BY id
#[test]
fn named_and_inline_window_can_be_mixed() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT id, flag, row_number() OVER w AS rn, count(*) OVER () AS total \
         FROM t WHERE id < 6 \
         WINDOW w AS (PARTITION BY flag ORDER BY id) ORDER BY id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i32(0), b(true), i64(1), i64(6)],
            vec![i32(1), b(false), i64(1), i64(6)],
            vec![i32(2), b(false), i64(2), i64(6)],
            vec![i32(3), b(true), i64(2), i64(6)],
            vec![i32(4), b(false), i64(3), i64(6)],
            vec![i32(5), b(false), i64(4), i64(6)],
        ]
    );
}

/// Multiple named windows can be defined and used selectively.
#[test]
fn multiple_named_windows() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT id, rank() OVER w1 AS r1, row_number() OVER w2 AS r2 \
         FROM t WHERE id < 4 \
         WINDOW w1 AS (ORDER BY id), w2 AS (PARTITION BY flag ORDER BY id) \
         ORDER BY id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i32(0), i64(1), i64(1)],
            vec![i32(1), i64(2), i64(1)],
            vec![i32(2), i64(3), i64(2)],
            vec![i32(3), i64(4), i64(2)],
        ]
    );
}

/// Referencing an undefined name with `OVER` is rejected at bind time
/// (`duckdb` rejects it as "window ... does not exist").
#[test]
fn undefined_named_window_is_rejected() {
    let mut s = session_with_basic();
    let err = s.prepare("SELECT sum(score) OVER w FROM t", &[]);
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// Defining the same name twice within the same `WINDOW` clause is rejected before binding
/// (at parse time) (`duckdb`: an error equivalent to "Duplicate window name").
#[test]
fn duplicate_window_name_in_the_same_clause_is_rejected() {
    let mut s = session_with_basic();
    let err = s.prepare("SELECT sum(score) OVER w FROM t WINDOW w AS (), w AS ()", &[]);
    assert_eq!(code_of(err), Some(Code::SyntaxError));
}

/// Trying to use `WINDOW` as a plain identifier rather than a reserved word (as a column name
/// `window`) is rejected, since unlike this construct's context-sensitive keywords,
/// `Kw::Window` is a global reserved word (see the comment in `sql::lexer`; treated the same
/// as `QUALIFY`).
#[test]
fn window_is_a_reserved_word_and_cannot_be_a_bare_column_name() {
    let mut s = session_with_basic();
    let err = s.prepare("SELECT window FROM t", &[]);
    assert!(code_of(err).is_some());
}

/// A `WINDOW` clause is syntactically allowed even if empty (i.e. containing a definition that's never actually used).
#[test]
fn an_unused_named_window_definition_does_not_error() {
    let mut s = session_with_basic();
    let rows = run(&mut s, "SELECT id FROM t WHERE id < 2 WINDOW w AS (ORDER BY id) ORDER BY id");
    assert_eq!(rows, vec![vec![i32(0)], vec![i32(1)]]);
}

/// A named window is computed over rows after filtering through `WHERE`/`GROUP BY`
/// (same as an ordinary `OVER (...)`; a regression check for ordinary window-function
/// semantics).
/// duckdb: SELECT id, flag, sum(score) OVER w AS s FROM 'basic.parquet'
///         WHERE id < 6 AND flag = false WINDOW w AS (ORDER BY id) ORDER BY id
#[test]
fn named_window_operates_on_rows_after_the_where_filter() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT id, flag, sum(score) OVER w AS s FROM t \
         WHERE id < 6 AND flag = false WINDOW w AS (ORDER BY id) ORDER BY id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i32(1), b(false), f64(1.5)],
            vec![i32(2), b(false), f64(4.5)],
            vec![i32(4), b(false), f64(10.5)],
            vec![i32(5), b(false), f64(18.0)],
        ]
    );
}

/// Further filters, in an outer `SELECT`, a result that went through a `WINDOW` clause
/// (combining the new feature with a subquery).
#[test]
fn named_window_result_can_be_filtered_by_an_outer_query() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT * FROM (SELECT id, sum(score) OVER w AS s FROM t WHERE id < 6 \
         WINDOW w AS (PARTITION BY flag ORDER BY id)) WHERE s > 4.0 ORDER BY id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i32(2), f64(4.5)],
            vec![i32(3), f64(4.5)],
            vec![i32(4), f64(10.5)],
            vec![i32(5), f64(18.0)],
        ]
    );
}

/// An ordinary `OVER (...)` still works as before even without a `WINDOW` clause itself
/// (regression check).
#[test]
fn plain_over_clause_is_unaffected() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT id, sum(score) OVER (PARTITION BY flag ORDER BY id) AS s \
         FROM t WHERE id < 4 ORDER BY id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i32(0), f64(0.0)],
            vec![i32(1), f64(1.5)],
            vec![i32(2), f64(4.5)],
            vec![i32(3), f64(4.5)],
        ]
    );
}
