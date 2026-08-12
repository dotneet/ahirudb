//! Integration tests for `PIVOT`/`UNPIVOT`.
//!
//! All expected values are decided by cross-checking against the actual output of
//! `duckdb -c "..."`
//! (`tests/data/pivot.parquet`/`pivot_small.parquet` are real files written by DuckDB.
//! See `scripts/gen-testdata.sh` for how they were generated).
//! The `ddl`/`dml` features are not needed (read-only Parquet is enough), so this always
//! runs even under the default-features `cargo test`.

use ahiru_core::error::{code_of, Code};
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn data(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn session_with(file: &str) -> Session {
    let mut s = Session::new();
    s.register_bytes("t", data(file)).unwrap();
    s
}

/// Runs `sql` and extracts the result as `Vec<Vec<Value>>`.
/// Files used by the tests fit entirely in memory, so `NeedIo`/`NeedCodec` never occur.
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
fn i128(v: i128) -> Value {
    Value::I128(v)
}
fn s(v: &str) -> Value {
    Value::Bytes(v.as_bytes().to_vec())
}
const NULL: Value = Value::Null;

// --- PIVOT -------------------------------------------------------------------

/// `PIVOT t ON category IN (...) USING sum(amount) GROUP BY region`:
/// The basic form: explicit `GROUP BY` + explicit `IN`.
/// duckdb: PIVOT 'pivot.parquet' ON category IN ('a','b','c') USING sum(amount)
///         GROUP BY region ORDER BY region;
#[test]
fn pivot_explicit_group_by_and_in_list() {
    let mut sess = session_with("pivot.parquet");
    let rows = run(
        &mut sess,
        "PIVOT t ON category IN ('a', 'b', 'c') USING sum(amount) GROUP BY region \
         ORDER BY region",
    );
    assert_eq!(
        rows,
        vec![
            vec![s("east"), i128(1500), i128(1700), i128(1300)],
            vec![s("north"), i128(1200), i128(1400), i128(1600)],
            vec![s("south"), i128(1650), i128(1250), i128(1450)],
            vec![s("west"), i128(1350), i128(1550), i128(1750)],
        ]
    );
}

/// When `GROUP BY` is omitted, "every column other than those referenced by `ON`/`USING`"
/// becomes the default grouping target (same rule as DuckDB).
/// duckdb: PIVOT 'pivot_small.parquet' ON category IN ('a','b','c') USING sum(amount)
///         ORDER BY region;
#[test]
fn pivot_default_group_by_uses_all_other_columns() {
    let mut sess = session_with("pivot_small.parquet");
    let rows =
        run(&mut sess, "PIVOT t ON category IN ('a', 'b', 'c') USING sum(amount) ORDER BY region");
    assert_eq!(
        rows,
        vec![
            vec![s("east"), i128(10), i128(20), NULL],
            vec![s("west"), i128(30), i128(40), i128(5)],
        ]
    );
}

/// When `USING` is omitted, the default is `count(*)`, same as DuckDB.
/// duckdb: PIVOT 'pivot_small.parquet' ON category IN ('a','b','c') GROUP BY region
///         ORDER BY region;
#[test]
fn pivot_without_using_defaults_to_count_star() {
    let mut sess = session_with("pivot_small.parquet");
    let rows =
        run(&mut sess, "PIVOT t ON category IN ('a', 'b', 'c') GROUP BY region ORDER BY region");
    assert_eq!(
        rows,
        vec![vec![s("east"), i64(1), i64(1), i64(0)], vec![s("west"), i64(1), i64(1), i64(1)],]
    );
}

/// `IN (value AS alias, ...)`: if an alias is given, the column name becomes that alias
/// (the value is not stringified).
/// duckdb: PIVOT 'pivot_small.parquet' ON category IN ('a' AS alpha, 'b' AS beta)
///         USING sum(amount) GROUP BY region ORDER BY region;
#[test]
fn pivot_in_list_aliases_become_column_names() {
    let mut sess = session_with("pivot_small.parquet");
    let rows = run(
        &mut sess,
        "PIVOT t ON category IN ('a' AS alpha, 'b' AS beta) USING sum(amount) \
         GROUP BY region ORDER BY region",
    );
    assert_eq!(
        rows,
        vec![vec![s("east"), i128(10), i128(20)], vec![s("west"), i128(30), i128(40)],]
    );
}

/// `ON` accepts any expression, not just a bare column.
/// duckdb: PIVOT 'pivot.parquet' ON id % 2 IN (0, 1) USING sum(amount) GROUP BY region
///         ORDER BY region;
#[test]
fn pivot_on_accepts_arbitrary_expression() {
    let mut sess = session_with("pivot.parquet");
    let rows = run(
        &mut sess,
        "PIVOT t ON id % 2 IN (0, 1) USING sum(amount) GROUP BY region ORDER BY region",
    );
    assert_eq!(
        rows,
        vec![
            vec![s("east"), i128(4500), NULL],
            vec![s("north"), i128(4200), NULL],
            vec![s("south"), NULL, i128(4350)],
            vec![s("west"), NULL, i128(4650)],
        ]
    );
}

/// `PIVOT` accepts a trailing `ORDER BY`/`LIMIT`/`OFFSET` (same as DuckDB).
#[test]
fn pivot_supports_trailing_order_by_limit_offset() {
    let mut sess = session_with("pivot.parquet");
    let rows = run(
        &mut sess,
        "PIVOT t ON category IN ('a', 'b', 'c') USING sum(amount) GROUP BY region \
         ORDER BY region LIMIT 2 OFFSET 1",
    );
    assert_eq!(
        rows,
        vec![
            vec![s("north"), i128(1200), i128(1400), i128(1600)],
            vec![s("south"), i128(1650), i128(1250), i128(1450)],
        ]
    );
}

/// Auto-detecting values (omitting `IN`) is unsupported because at bind time the target
/// column's actual data cannot be read (only schema resolution has happened), so DISTINCT
/// cannot be taken. Verify it clearly returns `UnsupportedFeature`.
#[test]
fn pivot_without_in_list_is_unsupported() {
    let mut sess = session_with("pivot_small.parquet");
    let err = sess.prepare("PIVOT t ON category USING sum(amount) GROUP BY region", &[]);
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// Multiple aggregate functions (`USING sum(a), avg(a)`) are unsupported because determining
/// the column name would require stringifying the expression.
#[test]
fn pivot_multiple_using_aggregates_is_unsupported() {
    let mut sess = session_with("pivot_small.parquet");
    let err = sess.prepare(
        "PIVOT t ON category IN ('a', 'b') USING sum(amount), avg(amount) GROUP BY region",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

// --- UNPIVOT -----------------------------------------------------------------

/// `UNPIVOT t ON col1, col2, ... INTO NAME .. VALUE ..`: folds multiple target columns into
/// two columns, a "column name" column and a "value" column. Non-target columns
/// (id/region/category) pass through unchanged.
/// duckdb: SELECT * FROM (UNPIVOT 'pivot.parquet' ON q1, q2, q3, q4
///         INTO NAME quarter VALUE amt) WHERE id < 2 ORDER BY id, quarter;
#[test]
fn unpivot_basic_folds_columns_into_name_value_pairs() {
    let mut sess = session_with("pivot.parquet");
    let rows = run(
        &mut sess,
        "UNPIVOT t ON q1, q2, q3, q4 INTO NAME quarter VALUE amt \
         ORDER BY id, quarter LIMIT 8",
    );
    assert_eq!(
        rows,
        vec![
            vec![i32(0), s("north"), s("a"), i32(0), s("q1"), i32(0)],
            vec![i32(0), s("north"), s("a"), i32(0), s("q2"), i32(0)],
            vec![i32(0), s("north"), s("a"), i32(0), s("q3"), i32(0)],
            vec![i32(0), s("north"), s("a"), i32(0), s("q4"), i32(0)],
            vec![i32(1), s("south"), s("b"), i32(10), s("q1"), i32(1)],
            vec![i32(1), s("south"), s("b"), i32(10), s("q2"), i32(2)],
            vec![i32(1), s("south"), s("b"), i32(10), s("q3"), i32(3)],
            vec![i32(1), s("south"), s("b"), i32(10), s("q4"), i32(4)],
        ]
    );
}

/// The row count is "original row count x number of target columns" (4 columns q1..q4 x 60
/// rows).
#[test]
fn unpivot_row_count_multiplies_by_target_column_count() {
    let mut sess = session_with("pivot.parquet");
    let rows = run(&mut sess, "UNPIVOT t ON q1, q2, q3, q4 INTO NAME quarter VALUE amt");
    assert_eq!(rows.len(), 60 * 4);
}

/// Omitting `INTO NAME .. VALUE ..` makes `name`/`value` the default column names, same as
/// DuckDB.
/// duckdb: UNPIVOT 'pivot_small.parquet' ON amount ORDER BY region, category;
#[test]
fn unpivot_default_name_and_value_columns() {
    let mut sess = session_with("pivot_small.parquet");
    let rows = run(&mut sess, "UNPIVOT t ON amount ORDER BY region, category");
    assert_eq!(
        rows,
        vec![
            vec![s("east"), s("a"), s("amount"), i32(10)],
            vec![s("east"), s("b"), s("amount"), i32(20)],
            vec![s("west"), s("a"), s("amount"), i32(30)],
            vec![s("west"), s("b"), s("amount"), i32(40)],
            vec![s("west"), s("c"), s("amount"), i32(5)],
        ]
    );
}

/// `UNPIVOT` also accepts a trailing `ORDER BY`/`LIMIT`/`OFFSET`.
#[test]
fn unpivot_supports_trailing_order_by_limit_offset() {
    let mut sess = session_with("pivot_small.parquet");
    let rows = run(&mut sess, "UNPIVOT t ON amount ORDER BY region, category LIMIT 2");
    assert_eq!(
        rows,
        vec![
            vec![s("east"), s("a"), s("amount"), i32(10)],
            vec![s("east"), s("b"), s("amount"), i32(20)],
        ]
    );
}

/// Target columns must be bare, unqualified column references only. Expressions are
/// unsupported.
#[test]
fn unpivot_target_must_be_a_bare_column_reference() {
    let mut sess = session_with("pivot.parquet");
    let err = sess.prepare("UNPIVOT t ON q1 + q2 INTO NAME k VALUE v", &[]);
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// If the target columns' types are incompatible (no implicit conversion), it becomes the
/// same type error as an ordinary `UNION ALL` (DuckDB also rejects the same situation with
/// "an explicit cast is required"; the error code's meaning matches ordinary set operations).
#[test]
fn unpivot_incompatible_column_types_is_type_mismatch() {
    let mut sess = session_with("pivot.parquet");
    let err = sess.prepare("UNPIVOT t ON region, amount INTO NAME k VALUE v", &[]);
    assert_eq!(code_of(err), Some(Code::TypeMismatch));
}

// --- Regression tests for discovered bugs --------------------------------------

/// Writing the same value twice in `IN (...)`, regardless of whether aliases are given,
/// `duckdb` rejects with
/// "The value ... was specified multiple times in the IN clause" (confirmed with
/// `duckdb -c "PIVOT ... ON category IN ('a','a') ..."`).
///
/// Bug before the fix: this check was missing, so `PIVOT t ON category IN ('a', 'a')
/// USING sum(amount) GROUP BY region` silently went through, producing two columns with
/// the same `FILTER` condition both named `"a"` (fixed by adding duplicate detection to
/// `plan::bind::desugar_pivot`).
#[test]
fn pivot_rejects_duplicate_values_in_the_in_list() {
    let mut sess = session_with("pivot_small.parquet");
    let err =
        sess.prepare("PIVOT t ON category IN ('a', 'a') USING sum(amount) GROUP BY region", &[]);
    assert_eq!(code_of(err), Some(Code::SyntaxError));
    // Even with aliases given, it is likewise rejected if the underlying values are
    // duplicated (`duckdb` only auto-renames a pure alias collision with a `_1` suffix,
    // but does not allow a duplicate of the value itself).
    let err = sess.prepare(
        "PIVOT t ON category IN ('a' AS x, 'a' AS y) USING sum(amount) GROUP BY region",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::SyntaxError));
}

/// Distinct values pass through fine (confirms the duplicate check has no false positives).
#[test]
fn pivot_distinct_values_in_the_in_list_are_fine() {
    let mut sess = session_with("pivot_small.parquet");
    let rows = run(
        &mut sess,
        "PIVOT t ON category IN ('a', 'b') USING sum(amount) GROUP BY region ORDER BY region",
    );
    assert_eq!(
        rows,
        vec![vec![s("east"), i128(10), i128(20)], vec![s("west"), i128(30), i128(40)],]
    );
}

/// `PIVOT`/`UNPIVOT` are syntax sugar that, in this engine, is only expanded at the start of a
/// statement, so it cannot be used as a derived table (`FROM (PIVOT ...)`), a CTE body, or a
/// term of a set operation (`plan::bind::desugar_pivot`/`desugar_unpivot` is designed to
/// expand exactly once at the entry point of `Session::prepare`; see `session.rs`). `duckdb`
/// allows this, but it is out of scope here.
///
/// Bug before the fix: in this case, `sql::parser::select_body` only turned into an
/// `UnexpectedToken` after reading ahead to the point where it expected `SELECT`, which made
/// it confusing why "`PIVOT` itself can be written, but a subquery gives a syntax error".
/// Fixed by detecting `PIVOT`/`UNPIVOT` at the start of `select_body` and returning a
/// meaningful `UnsupportedFeature` instead.
#[test]
fn pivot_as_a_derived_table_is_a_clear_unsupported_error_not_a_confusing_syntax_error() {
    let mut sess = session_with("pivot.parquet");
    let err = sess.prepare(
        "SELECT * FROM (PIVOT t ON category IN ('a', 'b', 'c') USING sum(amount) \
         GROUP BY region) WHERE a > 1300",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

#[test]
fn unpivot_as_a_cte_body_is_a_clear_unsupported_error() {
    let mut sess = session_with("pivot.parquet");
    let err = sess.prepare(
        "WITH u AS (UNPIVOT t ON q1, q2, q3, q4 INTO NAME quarter VALUE amt) \
         SELECT quarter, sum(amt) FROM u GROUP BY quarter",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// Meanwhile, top-level `PIVOT`/`UNPIVOT` itself still works exactly as before, without
/// regression.
#[test]
fn top_level_pivot_is_unaffected_by_the_derived_table_check() {
    let mut sess = session_with("pivot.parquet");
    let rows = run(
        &mut sess,
        "PIVOT t ON category IN ('a', 'b', 'c') USING sum(amount) GROUP BY region \
         ORDER BY region",
    );
    assert_eq!(rows.len(), 4);
}

/// `PIVOT`'s `FROM` accepts not just a table name but any derived table, including one
/// containing a `JOIN` (`desugar_pivot` just passes `from` straight through to
/// `SelectStmt::from`, so there's no constraint since it doesn't need to duplicate it the
/// way `UNPIVOT` does).
#[test]
fn pivot_from_accepts_a_derived_table_containing_a_join() {
    let mut sess = session_with("pivot.parquet");
    let rows = run(
        &mut sess,
        "PIVOT (SELECT t.* FROM t JOIN t AS t2 ON t.id = t2.id) \
         ON category IN ('a', 'b') USING sum(amount) GROUP BY region ORDER BY region",
    );
    assert!(!rows.is_empty());
}

/// `UNPIVOT` needs to duplicate `from` once per target column (see `clone_from_item`), and
/// since `JOIN`/`Subquery` can have plans that cannot be duplicated, it is explicitly
/// rejected. Verify it produces a clean error instead of crashing.
#[test]
fn unpivot_from_a_join_is_rejected_cleanly() {
    let mut sess = session_with("pivot.parquet");
    let err =
        sess.prepare("UNPIVOT t JOIN t AS t2 ON t.id = t2.id ON q1, q2 INTO NAME k VALUE v", &[]);
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}
