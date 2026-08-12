//! Integration tests for `GROUP BY GROUPING SETS`/`ROLLUP`/`CUBE`/`GROUPING()`.
//!
//! All expected values are decided by cross-checking against the actual output of
//! `duckdb -c "SELECT ..."` (`tests/data/basic.parquet` is a real file written by DuckDB).
//! The `ddl`/`dml` features are not needed (read-only Parquet is enough), so this always
//! runs even under the default-features `cargo test`.

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

/// Runs `sql` and extracts the result as `Vec<Vec<Value>>`.
/// Since `basic.parquet` fits entirely in memory, `NeedIo`/`NeedCodec` never occur.
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
fn b(v: bool) -> Value {
    Value::Bool(v)
}
const NULL: Value = Value::Null;

/// A plain GROUP BY without GROUPING SETS/ROLLUP/CUBE still works as before
/// (regression check).
#[test]
fn plain_group_by_is_unaffected() {
    let mut s = session_with_basic();
    let rows = run(&mut s, "SELECT flag, count(*) c FROM t GROUP BY flag ORDER BY flag");
    // duckdb: SELECT flag, count(*) c FROM 'basic.parquet' GROUP BY flag ORDER BY flag
    assert_eq!(rows, vec![vec![b(false), i64(666)], vec![b(true), i64(334)],]);
}

/// `GROUPING SETS ((flag), ())`: a simple subtotal + grand total. A column absent from a set becomes NULL.
#[test]
fn grouping_sets_basic() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT flag, count(*) c, sum(id) s FROM t \
         GROUP BY GROUPING SETS ((flag), ()) ORDER BY flag",
    );
    assert_eq!(
        rows,
        vec![
            vec![b(false), i64(666), i128(332667)],
            vec![b(true), i64(334), i128(166833)],
            vec![NULL, i64(1000), i128(499500)],
        ]
    );
}

/// `ROLLUP (flag, id % 3)`: expands into the hierarchical subsets
/// `(flag, id%3), (flag), ()`.
#[test]
fn rollup_expands_to_hierarchical_subsets() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT flag, id % 3 AS m, count(*) c FROM t \
         GROUP BY ROLLUP (flag, id % 3) ORDER BY 1, 2",
    );
    assert_eq!(
        rows,
        vec![
            vec![b(false), i32(1), i64(333)],
            vec![b(false), i32(2), i64(333)],
            vec![b(false), NULL, i64(666)],
            vec![b(true), i32(0), i64(334)],
            vec![b(true), NULL, i64(334)],
            vec![NULL, NULL, i64(1000)],
        ]
    );
}

/// `CUBE (flag, id % 3)`: expands into all subsets (2^2 = 4 sets).
#[test]
fn cube_expands_to_all_subsets() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT flag, id % 3 AS m, count(*) c FROM t \
         GROUP BY CUBE (flag, id % 3) ORDER BY 1, 2",
    );
    assert_eq!(
        rows,
        vec![
            vec![b(false), i32(1), i64(333)],
            vec![b(false), i32(2), i64(333)],
            vec![b(false), NULL, i64(666)],
            vec![b(true), i32(0), i64(334)],
            vec![b(true), NULL, i64(334)],
            vec![NULL, i32(0), i64(334)],
            vec![NULL, i32(1), i64(333)],
            vec![NULL, i32(2), i64(333)],
            vec![NULL, NULL, i64(1000)],
        ]
    );
}

/// `GROUPING()`/`GROUPING_ID()`: 0 if the column is alive in that set, 1 if it was collapsed
/// by aggregation into NULL. With multiple arguments, it's a bitmask with the first argument
/// as the highest bit.
#[test]
fn grouping_function_reports_which_columns_were_rolled_up() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT flag, id % 3 AS m, count(*) c, \
                grouping(flag) gf, grouping(id % 3) gm, grouping(flag, id % 3) gid \
         FROM t GROUP BY CUBE (flag, id % 3) ORDER BY 1, 2",
    );
    assert_eq!(
        rows,
        vec![
            vec![b(false), i32(1), i64(333), i64(0), i64(0), i64(0)],
            vec![b(false), i32(2), i64(333), i64(0), i64(0), i64(0)],
            vec![b(false), NULL, i64(666), i64(0), i64(1), i64(1)],
            vec![b(true), i32(0), i64(334), i64(0), i64(0), i64(0)],
            vec![b(true), NULL, i64(334), i64(0), i64(1), i64(1)],
            vec![NULL, i32(0), i64(334), i64(1), i64(0), i64(2)],
            vec![NULL, i32(1), i64(333), i64(1), i64(0), i64(2)],
            vec![NULL, i32(2), i64(333), i64(1), i64(0), i64(2)],
            vec![NULL, NULL, i64(1000), i64(1), i64(1), i64(3)],
        ]
    );
    // `GROUPING_ID` is also an alias for `GROUPING` in DuckDB (same bitmask semantics).
    let rows2 = run(
        &mut s,
        "SELECT flag, id % 3 AS m, grouping_id(flag, id % 3) gid \
         FROM t GROUP BY CUBE (flag, id % 3) ORDER BY 1, 2",
    );
    let last = rows2.last().unwrap();
    assert_eq!(last, &vec![NULL, NULL, i64(3)]);
}

/// `HAVING` applies to the final result after all grouping sets are combined with UNION ALL.
#[test]
fn having_filters_after_all_grouping_sets_are_combined() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT flag, id % 3 AS m, count(*) c FROM t \
         GROUP BY GROUPING SETS ((flag, id % 3), (flag), ()) \
         HAVING count(*) > 400 ORDER BY 1, 2",
    );
    assert_eq!(rows, vec![vec![b(false), NULL, i64(666)], vec![NULL, NULL, i64(1000)],]);
}

/// `GROUPING()` can also be used inside `HAVING`.
#[test]
fn having_can_reference_grouping() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT flag, id % 3 AS m, count(*) c FROM t \
         GROUP BY GROUPING SETS ((flag, id % 3), (flag), ()) \
         HAVING grouping(flag) = 0 ORDER BY 1, 2",
    );
    // Only rows where `flag` is alive (not collapsed) remain -- the grand-total row is dropped.
    assert_eq!(
        rows,
        vec![
            vec![b(false), i32(1), i64(333)],
            vec![b(false), i32(2), i64(333)],
            vec![b(false), NULL, i64(666)],
            vec![b(true), i32(0), i64(334)],
            vec![b(true), NULL, i64(334)],
        ]
    );
}

/// `GROUPING()`'s argument must be a grouping column.
#[test]
fn grouping_of_a_non_grouped_column_is_rejected() {
    let mut s = session_with_basic();
    let err = s.prepare(
        "SELECT flag, count(*), grouping(id) FROM t GROUP BY GROUPING SETS ((flag), ())",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::NotGrouped));
}

/// `GROUPING()` cannot be used in a query with no aggregation.
#[test]
fn grouping_without_aggregation_is_rejected() {
    let mut s = session_with_basic();
    let err = s.prepare("SELECT grouping(id) FROM t", &[]);
    assert_eq!(code_of(err), Some(Code::NotAggregate));
}

/// `GROUPING SETS` treats the union of all sets as the "grouping columns". A column absent
/// from a given set can still be referenced bare in SELECT, becoming NULL for that row
/// (not an error).
#[test]
fn columns_missing_from_a_set_are_still_selectable() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT flag, id % 3 AS m FROM t GROUP BY GROUPING SETS ((flag), (id % 3)) ORDER BY 1, 2",
    );
    // In the (flag)-only set, `m` is always NULL; in the (id % 3)-only set, `flag` is always NULL.
    assert!(rows.iter().any(|r| r[0] != NULL && r[1] == NULL));
    assert!(rows.iter().any(|r| r[0] == NULL && r[1] != NULL));
}
