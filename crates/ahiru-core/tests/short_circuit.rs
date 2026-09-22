//! Regression tests for guarded evaluation: `CASE`, `AND`, `OR`, `COALESCE` and `IFNULL` must
//! not raise an error from an operand that no row actually reaches.
//!
//! These used to evaluate every branch over every row and then pick, so
//! `CASE WHEN i < 30 THEN factorial(i) END` failed with "value out of range" on the rows the
//! WHEN had already excluded. Every expectation is the output of `duckdb -csv -c "..."` for the
//! same query unless a comment says otherwise. `range(5000)` and `tests/data/multi_rg.parquet`
//! (50000 rows) span more than one 2048-row vector batch, so the row-subset evaluation is also
//! exercised across batch boundaries and under a selection vector left by a filter.

use ahiru_core::error::Code;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn data(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn session() -> Session {
    let mut s = Session::new();
    s.register_bytes("m", data("multi_rg.parquet")).unwrap();
    s
}

/// Runs `sql`, returning the rows or the error code it fails with (at prepare or run time).
fn try_run(sql: &str) -> Result<Vec<Vec<Value>>, Code> {
    let mut session = session();
    let mut q = match session.prepare(sql, &[]).map_err(|e| e.code)? {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
    };
    let mut rows = Vec::new();
    loop {
        match session.step(&mut q).map_err(|e| e.code)? {
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

fn run(sql: &str) -> Vec<Vec<Value>> {
    try_run(sql).unwrap_or_else(|c| panic!("{sql}: {c:?}"))
}

/// Renders a value the way `duckdb -csv` prints it, for the types these tests produce.
fn text(v: &Value) -> String {
    match v {
        Value::Null => "NULL".into(),
        Value::I32(x) => x.to_string(),
        Value::I64(x) => x.to_string(),
        Value::I128(x) => x.to_string(),
        Value::Bytes(b) => String::from_utf8(b.clone()).expect("utf8"),
        Value::Bool(b) => b.to_string(),
        v => panic!("unexpected value {v:?}"),
    }
}

/// Every row, as comma-joined text.
fn csv(sql: &str) -> Vec<String> {
    run(sql).iter().map(|r| r.iter().map(text).collect::<Vec<_>>().join(",")).collect()
}

/// The single value of a one-row, one-column result, as text.
fn one(sql: &str) -> String {
    let rows = csv(sql);
    assert_eq!(rows.len(), 1, "{sql}");
    rows[0].clone()
}

// --- The reported cases ---------------------------------------------------------

#[test]
fn case_then_is_only_evaluated_for_rows_its_when_takes() {
    assert_eq!(
        one("SELECT sum(CASE WHEN i < 30 THEN factorial(i::INTEGER) ELSE 0 END) FROM range(40) t(i)"),
        "9157958657951075573395300940314"
    );
    // repeat() would build a 2 GB string and fail with LimitExceeded; no row reaches it.
    assert_eq!(
        one("SELECT count(CASE WHEN i < 0 THEN repeat('x', 2000000000) END) FROM range(3) t(i)"),
        "0"
    );
}

#[test]
fn coalesce_and_ifnull_stop_at_the_first_non_null_argument() {
    assert_eq!(one("SELECT count(coalesce(1, factorial(i::INTEGER))) FROM range(40) t(i)"), "40");
    assert_eq!(one("SELECT count(ifnull(1, factorial(i::INTEGER))) FROM range(40) t(i)"), "40");
    // Every row has a non-NULL `i`, so the fallible argument is never reached.
    assert_eq!(
        one("SELECT count(coalesce(i, factorial(i::INTEGER))) FROM range(5000) t(i)"),
        "5000"
    );
}

#[test]
fn where_and_or_guard_their_right_hand_side() {
    assert_eq!(
        one("SELECT count(*) FROM range(40) t(i) WHERE i < 30 AND factorial(i::INTEGER) > 0"),
        "30"
    );
    assert_eq!(
        one("SELECT count(*) FROM range(40) t(i) WHERE i >= 30 OR factorial(i::INTEGER) > 0"),
        "40"
    );
    assert_eq!(
        one("SELECT count(*) FROM range(40) t(i) \
             WHERE i < 30 AND i < 35 AND factorial(i::INTEGER) > 0"),
        "30"
    );
}

// --- CASE shapes -----------------------------------------------------------------

#[test]
fn nested_case_over_several_batches() {
    assert_eq!(
        one("SELECT sum(CASE WHEN i % 100 < 30 THEN \
                CASE WHEN i % 100 < 10 THEN factorial((i % 100)::INTEGER) * 2 \
                     WHEN i % 100 >= 25 THEN NULL \
                     ELSE factorial((i % 100)::INTEGER) END \
             ELSE 7 END) FROM range(5000) t(i)"),
        "32373903573478392267495900"
    );
}

#[test]
fn a_later_when_condition_is_only_evaluated_for_rows_no_earlier_when_took() {
    assert_eq!(
        one("SELECT sum(CASE WHEN i % 100 >= 30 THEN 1 \
                            WHEN factorial((i % 100)::INTEGER) > 1 THEN 2 END) \
             FROM range(5000) t(i)"),
        "6300"
    );
}

#[test]
fn null_conditions_fall_through_to_the_next_branch() {
    // A NULL WHEN is not a match: its THEN is never evaluated, the next WHEN is.
    assert_eq!(
        one("SELECT sum(CASE WHEN i % 100 >= 30 THEN NULL \
                            WHEN NULL THEN factorial(99) \
                            WHEN i % 2 = 0 THEN factorial((i % 100)::INTEGER) \
                            ELSE 1 END) FROM range(5000) t(i)"),
        "15264612882384112627323412934100"
    );
    assert_eq!(
        one("SELECT sum(CASE WHEN NULL THEN factorial(50) ELSE 1 END) FROM range(40) t(i)"),
        "40"
    );
    // A condition that is NULL on some rows (`NULLIF` makes every tenth one NULL).
    assert_eq!(
        one("SELECT sum(CASE WHEN nullif(i % 10, 0) > 3 THEN 1 \
                            WHEN i % 100 < 30 THEN factorial((i % 100)::INTEGER) \
                            ELSE 0 END) FROM range(5000) t(i)"),
        "1351477065542460998403500"
    );
}

#[test]
fn simple_case_compares_the_operand_once_per_when() {
    assert_eq!(
        one("SELECT sum(CASE i % 100 WHEN 1 THEN 10 WHEN 2 THEN factorial((i % 100)::INTEGER) \
                                 WHEN 3 THEN NULL ELSE 0 END) FROM range(5000) t(i)"),
        "600"
    );
    assert_eq!(
        one("SELECT sum(CASE i WHEN 1 THEN 10 WHEN 2 THEN factorial(i::INTEGER) ELSE 0 END) \
             FROM range(40) t(i)"),
        "12"
    );
}

#[test]
fn case_in_a_projection_under_a_filter_keeps_every_row_in_place() {
    // The filter leaves a selection vector; the guarded THEN/ELSE results must land back on the
    // right rows across the 2048-row batch boundary.
    assert_eq!(
        csv("SELECT i, CASE WHEN i % 100 < 25 THEN factorial((i % 100)::INTEGER) \
                            ELSE -i END \
             FROM range(5000) t(i) \
             WHERE (i BETWEEN 2045 AND 2051 OR i BETWEEN 4120 AND 4122) AND i % 3 <> 0 \
             ORDER BY i"),
        vec![
            "2045,-2045",
            "2047,-2047",
            "2048,-2048",
            "2050,-2050",
            "2051,-2051",
            "4120,2432902008176640000",
            "4121,51090942171709440000",
        ]
    );
    assert_eq!(
        one("SELECT sum(CASE WHEN i % 100 < 30 THEN factorial((i % 100)::INTEGER) END) \
             FROM range(5000) t(i) WHERE i % 7 <> 3"),
        "393792221671447847567071069348696"
    );
}

#[test]
fn case_with_string_results_over_several_batches() {
    assert_eq!(
        one("SELECT count(x), count(DISTINCT x), min(x), max(x), sum(length(x)) FROM ( \
               SELECT CASE WHEN i % 7 = 0 THEN NULL \
                           WHEN i % 100 < 20 THEN 'f' || factorial((i % 100)::INTEGER)::VARCHAR \
                           ELSE coalesce(NULL, 'v' || i::VARCHAR, factorial(99)::VARCHAR) END AS x \
               FROM range(5000) t(i) WHERE i % 5 <> 1)"),
        "3428,2759,f1,v999,19025"
    );
}

#[test]
fn case_inside_a_window_and_over_a_window_result() {
    assert_eq!(
        one("SELECT sum(r) FROM (SELECT sum(CASE WHEN i % 100 < 30 \
                 THEN factorial((i % 100)::INTEGER) ELSE 0 END) OVER (PARTITION BY i % 3) AS r \
             FROM range(5000) t(i))"),
        "763160179732526930675181867035447769"
    );
}

#[test]
fn case_over_an_aggregate_result() {
    assert_eq!(
        one("SELECT CASE WHEN count(*) > 100 THEN 0 ELSE factorial(count(*)::INTEGER) END \
             FROM range(5000) t(i)"),
        "0"
    );
    assert_eq!(
        csv("SELECT g, count(*) FROM (SELECT i % 100 AS g FROM range(5000) t(i)) GROUP BY g \
             HAVING g < 30 AND factorial(g::INTEGER) > 1 ORDER BY g DESC LIMIT 1"),
        vec!["29,50"]
    );
}

#[test]
fn join_condition_guards_its_later_conjuncts() {
    assert_eq!(
        one("SELECT count(*) FROM range(5000) a(i) JOIN range(100) b(j) \
             ON a.i = b.j AND b.j < 30 AND factorial(b.j::INTEGER) > 0"),
        "30"
    );
}

#[test]
fn nested_and_or_inside_where() {
    // DuckDB evaluates these forms eagerly and raises; the guarantee here is the stronger one
    // (each AND/OR guards its own right-hand side), so the expectations are the SQL results.
    assert_eq!(
        one("SELECT count(*) FROM range(5000) t(i) \
             WHERE i % 100 < 30 AND (i % 2 = 0 OR factorial((i % 100)::INTEGER) > 0)"),
        "1500"
    );
    assert_eq!(
        one("SELECT count(*) FROM range(5000) t(i) \
             WHERE NOT (i % 100 >= 30 OR factorial((i % 100)::INTEGER) < 0)"),
        "1500"
    );
}

#[test]
fn iif_qualify_and_a_fallible_cast_are_guarded_too() {
    // IF/IIF desugar to CASE in the parser.
    assert_eq!(one("SELECT count(if(i < 30, factorial(i::INTEGER), 0)) FROM range(40) t(i)"), "40");
    assert_eq!(
        one("SELECT count(*) FROM (SELECT i FROM range(40) t(i) \
             QUALIFY row_number() OVER (ORDER BY i) < 31 AND factorial(i::INTEGER) > 0)"),
        "30"
    );
    // An invalid VARCHAR -> JSON cast raises (duckdb: Conversion Error) -- unless no row
    // reaches it.
    assert_eq!(
        one("SELECT count(*) FROM range(40) t(i) WHERE i < 0 AND CAST('x' AS JSON) IS NULL"),
        "0"
    );
}

// --- Pushed into a Parquet scan ----------------------------------------------------

#[test]
fn where_conjuncts_pushed_into_a_parquet_scan_keep_their_order() {
    assert_eq!(
        one("SELECT count(*), sum(k) FROM m WHERE id % 100 < 30 AND factorial(id % 100) > 0"),
        "15000,714375"
    );
    assert_eq!(
        one("SELECT count(*) FROM m WHERE id % 100 >= 30 OR factorial(id % 100) > 1"),
        "49000"
    );
    assert_eq!(
        one("SELECT sum(CASE WHEN id % 100 < 30 THEN factorial(id % 100) END) \
             FROM m WHERE id > 1000"),
        "4487399742396027030963697460753859"
    );
}

// --- The guard only hides errors from rows that never reach the operand -------------

#[test]
fn a_row_that_does_reach_the_operand_still_raises() {
    // 34! overflows HUGEINT; rows 34..39 reach the THEN.
    assert_eq!(
        try_run("SELECT sum(CASE WHEN i < 40 THEN factorial(i::INTEGER) END) FROM range(40) t(i)")
            .err(),
        Some(Code::ValueOutOfRange)
    );
    assert_eq!(
        try_run("SELECT count(*) FROM range(40) t(i) WHERE i < 35 AND factorial(i::INTEGER) > 0")
            .err(),
        Some(Code::ValueOutOfRange)
    );
    assert_eq!(
        try_run("SELECT count(coalesce(NULL, factorial(i::INTEGER))) FROM range(40) t(i)").err(),
        Some(Code::ValueOutOfRange)
    );
    // The first WHEN is reached by every row.
    assert_eq!(
        try_run("SELECT count(CASE WHEN factorial(i::INTEGER) > 0 THEN 1 END) FROM range(40) t(i)")
            .err(),
        Some(Code::ValueOutOfRange)
    );
}
