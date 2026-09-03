//! Regression tests for the physical-operator fixes: window `product()` over DECIMAL,
//! `list()`/`array_agg()` rendering, `mode()`'s tie-break, INTERVAL comparison keys, and
//! `unnest` element typing.
//!
//! Every expectation is the output of `duckdb -csv -c "..."` for the same query unless a
//! comment says otherwise. Values whose *spelling* matters (DECIMAL scale, INTERVAL text,
//! the JSON-ish text `list()` produces) are compared as text.

use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

/// `id` orders the rows, `n` feeds `mode()` (1,2,2,1 -- a tie broken by first appearance),
/// `d` feeds the DECIMAL products, and `f` the float rendering.
const ROWS: &str = "id,n,d,f\n1,1,1.5,0.5\n2,2,2.5,0.1\n3,2,4.5,0.001\n4,1,1.0,0.0001\n";

fn session() -> Session {
    let mut s = Session::new();
    s.register_bytes_as("t", ROWS.as_bytes().to_vec(), FormatKind::Csv).unwrap();
    s
}

fn run(session: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    let mut q = match session.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
    };
    let mut rows = Vec::new();
    loop {
        match session.step(&mut q).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
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

/// Every row's first column, rendered as text.
fn texts(sql: &str) -> Vec<String> {
    let mut s = session();
    run(&mut s, sql)
        .into_iter()
        .map(|r| match &r[0] {
            Value::Bytes(b) => String::from_utf8(b.clone()).expect("utf8"),
            Value::Null => "<null>".into(),
            v => panic!("{sql}: expected VARCHAR, got {v:?}"),
        })
        .collect()
}

/// The first column of the first row, as text.
fn text(sql: &str) -> String {
    let v = texts(sql);
    assert_eq!(v.len(), 1, "{sql}: expected exactly one row, got {v:?}");
    v.into_iter().next().unwrap_or_default()
}

fn one(sql: &str) -> Value {
    let mut s = session();
    let rows = run(&mut s, sql);
    assert_eq!(rows.len(), 1, "{sql}: expected exactly one row");
    rows[0][0].clone()
}

// ---- window product() over DECIMAL ---------------------------------------

#[test]
fn window_product_over_decimal_applies_the_scale() {
    // DECIMAL is stored as an integer scaled by 10^scale. Multiplying the raw integers made
    // every factor 100x too large: 150, 37500, ... duckdb: 1.5, 3.75, 16.875, 16.875.
    let got = texts(
        "SELECT CAST(product(CAST(d AS DECIMAL(6,2))) OVER (ORDER BY id) AS VARCHAR) \
         FROM t ORDER BY id",
    );
    assert_eq!(got, vec!["1.5", "3.75", "16.875", "16.875"]);
}

#[test]
fn window_product_matches_the_grouped_product() {
    // The blocking `product()` was already right; the two must not disagree.
    let windowed =
        texts("SELECT CAST(product(CAST(d AS DECIMAL(6,2))) OVER () AS VARCHAR) FROM t LIMIT 1");
    let grouped = texts("SELECT CAST(product(CAST(d AS DECIMAL(6,2))) AS VARCHAR) FROM t");
    assert_eq!(windowed, grouped);
    // duckdb: 16.875
    assert_eq!(grouped, vec!["16.875"]);
}

// ---- list() / array_agg() rendering --------------------------------------

#[test]
fn list_of_decimal_keeps_the_decimal_point() {
    // duckdb: [1.50, 2.50, 4.50, 1.00]. This used to print the raw scaled integers.
    assert_eq!(text("SELECT list(CAST(d AS DECIMAL(6,2))) FROM t"), "[1.50, 2.50, 4.50, 1.00]");
}

#[test]
fn list_of_temporal_values_renders_as_quoted_text() {
    // DATE/TIME/TIMESTAMP are stored as day and microsecond counts; `list()` used to print
    // those integers (18263, 45296000000, 1622550896000000). Elements are JSON text here, so
    // they go out quoted -- the same rendering `to_json` gives them.
    assert_eq!(text("SELECT list(DATE '2020-01-02') FROM t WHERE id = 1"), "[\"2020-01-02\"]");
    assert_eq!(text("SELECT list(TIME '12:34:56') FROM t WHERE id = 1"), "[\"12:34:56\"]");
    assert_eq!(
        text("SELECT list(TIMESTAMP '2021-06-01 12:34:56') FROM t WHERE id = 1"),
        "[\"2021-06-01 12:34:56\"]"
    );
    // INTERVAL and UUID were equally unreadable (a packed i128 / raw 16 bytes).
    assert_eq!(text("SELECT list(INTERVAL 1 DAY) FROM t WHERE id = 1"), "[\"1 day\"]");
    assert_eq!(
        text(
            "SELECT list(CAST('11111111-2222-3333-4444-555555555555' AS UUID)) \
             FROM t WHERE id = 1"
        ),
        "[\"11111111-2222-3333-4444-555555555555\"]"
    );
}

#[test]
fn list_of_doubles_uses_the_cast_to_varchar_spelling() {
    // duckdb: [0.5, 0.1, 0.001, 0.0001]. A private renderer here printed 5e-1, 1e-1, 1e-3.
    assert_eq!(text("SELECT list(f) FROM t"), "[0.5, 0.1, 0.001, 0.0001]");
    // ... and it still switches to exponential form exactly where `CAST(x AS VARCHAR)` does.
    assert_eq!(text("SELECT list(f * 1e18) FROM t WHERE id = 1"), "[5e+17]");
    assert_eq!(text("SELECT CAST(f * 1e18 AS VARCHAR) FROM t WHERE id = 1"), "5e+17");
}

#[test]
fn array_agg_of_json_stays_nested() {
    // The element is already JSON text, so it is embedded rather than quoted as a string.
    assert_eq!(text("SELECT array_agg(CAST('[1,2]' AS JSON)) FROM t WHERE id = 1"), "[[1,2]]");
}

// ---- mode() tie-break -----------------------------------------------------

#[test]
fn mode_breaks_ties_on_first_appearance() {
    // n is 1,2,2,1: both values end on two votes. duckdb answers 1, the value seen first.
    // Tracking only "who reached the top count first" answered 2.
    assert_eq!(one("SELECT mode(n) FROM t"), Value::I64(1));
    // Feeding the same two values in the opposite order flips the answer, which is what
    // "first appearance" means. duckdb answers 2 for 2,1,1,2.
    let reversed = "SELECT mode(n) FROM (\
          SELECT 2 AS n FROM t WHERE id = 1 UNION ALL SELECT 1 FROM t WHERE id = 1 \
          UNION ALL SELECT 1 FROM t WHERE id = 1 UNION ALL SELECT 2 FROM t WHERE id = 1\
        ) s";
    // The literals bind as INTEGER, so the result comes back as I32 here.
    assert_eq!(one(reversed), Value::I32(2));
    // A clear winner is unaffected.
    assert_eq!(
        one("SELECT mode(n) FROM (SELECT n FROM t UNION ALL SELECT 2 FROM t) s"),
        Value::I64(2)
    );
}

// ---- INTERVAL comparison keys ---------------------------------------------
//
// INTERVAL is months/days/microseconds packed into one i128. Comparing that bit pattern made
// `1 day` and `24 hours` different values everywhere keys are built. duckdb normalizes with
// 1 month = 30 days and 1 day = 24 hours; so do we now.

#[test]
fn interval_equality_normalizes_the_components() {
    // duckdb: true, true, false
    assert_eq!(
        one("SELECT INTERVAL 1 DAY = INTERVAL 24 HOUR FROM t WHERE id = 1"),
        Value::Bool(true)
    );
    assert_eq!(
        one("SELECT INTERVAL 1 MONTH = INTERVAL 30 DAY FROM t WHERE id = 1"),
        Value::Bool(true)
    );
    assert_eq!(
        one("SELECT INTERVAL 1 DAY <> INTERVAL 24 HOUR FROM t WHERE id = 1"),
        Value::Bool(false)
    );
    assert_eq!(
        one("SELECT INTERVAL 1 DAY = INTERVAL 25 HOUR FROM t WHERE id = 1"),
        Value::Bool(false)
    );
}

/// Three spans that the packed representation ranks wrongly: `1 day` has a non-zero days
/// field, so its bit pattern beats both hour-only values.
const THREE_SPANS: &str = "SELECT INTERVAL 25 HOUR AS x FROM t WHERE id = 1 \
     UNION ALL SELECT INTERVAL 1 DAY FROM t WHERE id = 1 \
     UNION ALL SELECT INTERVAL 23 HOUR FROM t WHERE id = 1";

#[test]
fn interval_order_by_uses_the_normalized_span() {
    // duckdb: 23:00:00, 1 day, 25:00:00
    let sql = format!("SELECT CAST(x AS VARCHAR) FROM ({THREE_SPANS}) s ORDER BY x");
    assert_eq!(texts(&sql), vec!["23:00:00", "1 day", "25:00:00"]);
}

#[test]
fn interval_min_max_use_the_normalized_span() {
    // duckdb: 23:00:00 / 25:00:00. The packed comparison answered 23:00:00 / 1 day.
    let sql = format!(
        "SELECT CAST(min(x) AS VARCHAR) || ' / ' || CAST(max(x) AS VARCHAR) \
         FROM ({THREE_SPANS}) s"
    );
    assert_eq!(text(&sql), "23:00:00 / 25:00:00");
}

#[test]
fn interval_window_min_max_use_the_normalized_span() {
    // The window accumulator has its own comparison; it must agree with the grouped one.
    let sql = format!("SELECT CAST(max(x) OVER () AS VARCHAR) FROM ({THREE_SPANS}) s LIMIT 1");
    assert_eq!(text(&sql), "25:00:00");
}

#[test]
fn interval_window_order_by_uses_the_normalized_span() {
    let sql = format!(
        "SELECT CAST(x AS VARCHAR) FROM (\
           SELECT x, row_number() OVER (ORDER BY x) AS rn FROM ({THREE_SPANS}) s\
         ) w ORDER BY rn"
    );
    assert_eq!(texts(&sql), vec!["23:00:00", "1 day", "25:00:00"]);
}

#[test]
fn interval_equi_join_matches_equal_spans() {
    // duckdb: 1. The hash key came straight from the bit pattern, so this found nothing.
    let sql = "SELECT count(*) FROM (SELECT INTERVAL 1 DAY AS x FROM t WHERE id = 1) a \
               JOIN (SELECT INTERVAL 24 HOUR AS x FROM t WHERE id = 1) b ON a.x = b.x";
    assert_eq!(one(sql), Value::I64(1));
}

#[test]
fn interval_set_operations_and_grouping_dedup_equal_spans() {
    // duckdb: 1 row each.
    let union = "SELECT count(*) FROM (SELECT INTERVAL 1 MONTH AS x FROM t WHERE id = 1 \
                 UNION SELECT INTERVAL 30 DAY FROM t WHERE id = 1) s";
    assert_eq!(one(union), Value::I64(1));
    let group = "SELECT count(*) FROM (SELECT x FROM (\
                   SELECT INTERVAL 1 DAY AS x FROM t WHERE id = 1 \
                   UNION ALL SELECT INTERVAL 24 HOUR FROM t WHERE id = 1\
                 ) s GROUP BY x) g";
    assert_eq!(one(group), Value::I64(1));
    let distinct = "SELECT count(*) FROM (SELECT DISTINCT x FROM (\
                      SELECT INTERVAL 1 DAY AS x FROM t WHERE id = 1 \
                      UNION ALL SELECT INTERVAL 24 HOUR FROM t WHERE id = 1\
                    ) s) d";
    assert_eq!(one(distinct), Value::I64(1));
}

// ---- unnest element typing ------------------------------------------------

#[test]
fn unnest_keeps_integers_that_do_not_fit_bigint() {
    // duckdb yields HUGEINT here: 1, 9223372036854775808. The element type was pinned to
    // BIGINT, so the second element silently became NULL.
    let got = texts("SELECT CAST(unnest([1, 9223372036854775808]) AS VARCHAR) FROM t WHERE id = 1");
    assert_eq!(got, vec!["1", "9223372036854775808"]);
    // Unsigned values past i64 need the same width.
    let got =
        texts("SELECT CAST(unnest([1, 18446744073709551615]) AS VARCHAR) FROM t WHERE id = 1");
    assert_eq!(got, vec!["1", "18446744073709551615"]);
    // An all-BIGINT array must keep its narrow element type (and its NULLs).
    let got = texts("SELECT CAST(unnest([1, NULL, 3]) AS VARCHAR) FROM t WHERE id = 1");
    assert_eq!(got, vec!["1", "<null>", "3"]);
}
