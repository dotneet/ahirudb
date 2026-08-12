//! Integration tests for the `generate_series`/`range` table functions.
//!
//! Expected values are decided by cross-checking against the actual output of the `duckdb` CLI:
//! - `range(stop)`/`range(start, stop)`/`range(start, stop, step)` are half-open
//!   intervals (`stop` excluded).
//! - `generate_series(stop)`/`generate_series(start, stop)`/
//!   `generate_series(start, stop, step)` are closed intervals (`stop` included).
//! - Without an alias, the column name is `"range"`/`"generate_series"` respectively.
//! - If `step`'s direction contradicts the ordering of `start`/`stop`, the result is 0 rows (not an error).
//! - `step = 0` is a bind-time error (`duckdb`: "interval cannot be 0!").

use ahiru_core::error::{code_of, Code};
use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::{Field, Ty, Value};

/// A session with `dual` (a dummy table with a single row) registered. This works around v1
/// excluding a bare `SELECT` of literals with no `FROM` (same as `unnest.rs`).
fn session_with_dual() -> Session {
    let mut s = Session::new();
    s.register_bytes_as("dual", b"x\n1\n".to_vec(), FormatKind::Csv).unwrap();
    s
}

/// Runs a query to completion where all data is in memory.
fn run(s: &mut Session, sql: &str) -> (Vec<Field>, Vec<Vec<Value>>) {
    let mut q = match s.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
    };
    let schema = q.schema.clone();
    let mut rows = Vec::new();
    loop {
        match s.step(&mut q).unwrap() {
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
    (schema, rows)
}

fn i64s(vals: impl IntoIterator<Item = i64>) -> Vec<Vec<Value>> {
    vals.into_iter().map(|v| vec![Value::I64(v)]).collect()
}

// --- range ---------------------------------------------------------------------

/// duckdb: `SELECT * FROM range(5)` -> 0,1,2,3,4 (`stop` excluded).
#[test]
fn range_single_arg_starts_at_zero_and_excludes_stop() {
    let mut db = session_with_dual();
    let (schema, rows) = run(&mut db, "SELECT * FROM range(5)");
    assert_eq!(schema[0].name, "range");
    assert_eq!(schema[0].ty, Ty::BigInt);
    assert_eq!(rows, i64s(0..5));
}

/// duckdb: `SELECT * FROM range(0, 100, 5)` -> 0,5,10,...,95.
#[test]
fn range_three_args_honors_start_stop_step() {
    let mut db = session_with_dual();
    let (_, rows) = run(&mut db, "SELECT * FROM range(0, 100, 5)");
    assert_eq!(rows, i64s((0..100).step_by(5)));
}

/// duckdb: `SELECT * FROM range(10, 0, -2)` -> 10,8,6,4,2.
#[test]
fn range_negative_step_counts_down() {
    let mut db = session_with_dual();
    let (_, rows) = run(&mut db, "SELECT * FROM range(10, 0, -2)");
    assert_eq!(rows, i64s([10, 8, 6, 4, 2]));
}

/// duckdb: 0 rows when the direction contradicts (e.g. a positive step with start > stop).
#[test]
fn range_mismatched_direction_yields_zero_rows() {
    let mut db = session_with_dual();
    let (_, rows) = run(&mut db, "SELECT * FROM range(10, 0, 1)");
    assert!(rows.is_empty());
    let (_, rows) = run(&mut db, "SELECT * FROM range(0, 10, -1)");
    assert!(rows.is_empty());
}

// --- generate_series -------------------------------------------------------------

/// duckdb: `SELECT * FROM generate_series(5)` -> 0,1,2,3,4,5 (`stop` included).
#[test]
fn generate_series_single_arg_starts_at_zero_and_includes_stop() {
    let mut db = session_with_dual();
    let (schema, rows) = run(&mut db, "SELECT * FROM generate_series(5)");
    assert_eq!(schema[0].name, "generate_series");
    assert_eq!(rows, i64s(0..=5));
}

/// duckdb: `SELECT * FROM generate_series(1, 10)` -> 1..=10.
#[test]
fn generate_series_two_args() {
    let mut db = session_with_dual();
    let (_, rows) = run(&mut db, "SELECT * FROM generate_series(1, 10)");
    assert_eq!(rows, i64s(1..=10));
}

/// duckdb: `SELECT * FROM generate_series(0, 10, 2)` -> 0,2,4,6,8,10.
#[test]
fn generate_series_three_args_with_step() {
    let mut db = session_with_dual();
    let (_, rows) = run(&mut db, "SELECT * FROM generate_series(0, 10, 2)");
    assert_eq!(rows, i64s([0, 2, 4, 6, 8, 10]));
}

// --- Alias ---------------------------------------------------------------------

#[test]
fn column_alias_renames_the_output_column() {
    let mut db = session_with_dual();
    let (schema, rows) = run(&mut db, "SELECT x FROM range(3) AS t(x)");
    assert_eq!(schema[0].name, "x");
    assert_eq!(rows, i64s(0..3));
}

#[test]
fn table_alias_qualifies_the_column_reference() {
    let mut db = session_with_dual();
    let (_, rows) = run(&mut db, "SELECT t.x FROM range(3) AS t(x) WHERE t.x > 0");
    assert_eq!(rows, i64s(1..3));
}

// --- Combinations --------------------------------------------------------------

#[test]
fn works_with_where_and_order_by() {
    let mut db = session_with_dual();
    let (_, rows) = run(&mut db, "SELECT * FROM range(20) WHERE range % 3 = 0 ORDER BY range DESC");
    assert_eq!(rows, i64s([18, 15, 12, 9, 6, 3, 0]));
}

#[test]
fn can_join_two_generated_series() {
    let mut db = session_with_dual();
    let (_, rows) =
        run(&mut db, "SELECT a.x, b.y FROM range(2) AS a(x), range(2) AS b(y) ORDER BY a.x, b.y");
    assert_eq!(
        rows,
        vec![
            vec![Value::I64(0), Value::I64(0)],
            vec![Value::I64(0), Value::I64(1)],
            vec![Value::I64(1), Value::I64(0)],
            vec![Value::I64(1), Value::I64(1)],
        ]
    );
}

/// Verify by row count and endpoint values that generation stays correct even over a large
/// range (without expanding it all into memory at once) (the internals -- generating in
/// `BATCH_SIZE`-sized chunks in `exec::range::GenerateSeries` -- are already verified in detail
/// by the unit tests in `exec/range.rs`).
#[test]
fn a_large_range_still_produces_the_correct_count_and_endpoints() {
    let mut db = session_with_dual();
    let (_, rows) = run(&mut db, "SELECT count(*), min(range), max(range) FROM range(500000)");
    assert_eq!(rows, vec![vec![Value::I64(500000), Value::I64(0), Value::I64(499999)]]);
}

// --- Errors ----------------------------------------------------------------------

/// duckdb: `step = 0` is a bind-time error ("interval cannot be 0!").
#[test]
fn zero_step_is_rejected() {
    let mut db = session_with_dual();
    let err = db.prepare("SELECT * FROM range(0, 10, 0)", &[]);
    assert_eq!(code_of(err), Some(Code::DivideByZero));
}

/// A call with no arguments is rejected (`duckdb` also turns `range()` into a function resolution error).
#[test]
fn zero_arguments_is_rejected() {
    let mut db = session_with_dual();
    let err = db.prepare("SELECT * FROM range()", &[]);
    assert_eq!(code_of(err), Some(Code::WrongArgCount));
}

/// A call with too many arguments is also rejected.
#[test]
fn too_many_arguments_is_rejected() {
    let mut db = session_with_dual();
    let err = db.prepare("SELECT * FROM range(1, 2, 3, 4)", &[]);
    assert_eq!(code_of(err), Some(Code::WrongArgCount));
}

/// Fractional arguments are unsupported (`sql::parser::base_rel` only reads arguments via
/// `signed_int_lit`). `duckdb` likewise rejects anything other than
/// `generate_series(BIGINT, ...)` by failing overload resolution, so the direction matches
/// (the only difference is whether the error happens at parse time or bind time).
#[test]
fn float_arguments_are_rejected() {
    let mut db = session_with_dual();
    let err = db.prepare("SELECT * FROM generate_series(1.5, 5.5)", &[]);
    assert!(code_of(err).is_some());
    let err = db.prepare("SELECT * FROM range(1.5)", &[]);
    assert!(code_of(err).is_some());
}

// --- Keyword vs. column/table name collisions -----------------------------------
//
// `range`/`generate_series` get special treatment as table functions, but are not reserved
// words (see the doc on `base_rel`: reserving them would repeat a past incident where
// reserving a word broke a column/table with the same name). Verify that real data with
// columns or tables using these names is not broken.

#[test]
fn a_real_column_named_range_is_not_shadowed_by_the_table_function() {
    let mut db = Session::new();
    db.register_bytes_as("t2", b"range\n7\n".to_vec(), FormatKind::Csv).unwrap();
    let (_, rows) = run(&mut db, "SELECT range FROM t2");
    assert_eq!(rows, vec![vec![Value::I64(7)]]);
    let (_, rows) = run(&mut db, "SELECT t2.range FROM t2");
    assert_eq!(rows, vec![vec![Value::I64(7)]]);
}

/// If a real table named `range`/`generate_series` is registered, `FROM range`
/// (without parentheses) resolves as a table reference. `range(...)` as a table function
/// always comes with a `(`, so there is no syntactic collision
/// (see the `if self.is(Tok::LParen)` branch in `base_rel`).
#[test]
fn a_real_table_named_range_is_queryable_by_name() {
    let mut db = Session::new();
    db.register_bytes_as("range", b"x\n9\n".to_vec(), FormatKind::Csv).unwrap();
    let (_, rows) = run(&mut db, "SELECT x FROM range");
    assert_eq!(rows, vec![vec![Value::I64(9)]]);
}

// --- Interaction: JOIN with real data --------------------------------------------

/// Uses `generate_series`/`range` on one side of a `JOIN` with a real table (Parquet).
/// duckdb: SELECT b.id FROM range(3) a JOIN 'basic.parquet' b ON a.range = b.id
///         ORDER BY b.id
///
/// Since `range`'s column is `BIGINT` and `basic.parquet`'s `id` is `INTEGER`, making the
/// `ON` clause an uncast `a.range = b.id` produces 0 rows
/// (a known, separate bug where `plan::bind`'s equi-join key extraction fails to unify
/// different integer physical types.
/// `query_composition_extra.rs::join_on_mixed_numeric_key_types_compares_by_value` already
/// reproduces and documents it in detail, and it's carved out for whoever owns
/// `plan::bind`. This isn't a bug in `generate_series`/`range` itself, so here we just
/// align the types with an explicit cast before checking).
#[test]
fn range_can_join_a_real_parquet_table() {
    let mut db = session_with_dual();
    db.register_bytes(
        "basic",
        std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/basic.parquet"))
            .unwrap(),
    )
    .unwrap();
    let (_, rows) = run(
        &mut db,
        "SELECT b.id FROM range(3) a JOIN basic b ON CAST(a.range AS INTEGER) = b.id \
         ORDER BY b.id",
    );
    assert_eq!(rows, vec![vec![Value::I32(0)], vec![Value::I32(1)], vec![Value::I32(2)]]);
}
