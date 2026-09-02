//! Regression tests for a batch of residual correctness fixes.
//!
//! Every expectation is the actual output of `duckdb -csv -c "SELECT ..."` unless a
//! comment says otherwise. Values are compared through `CAST(... AS VARCHAR)` where the
//! spelling matters (DECIMAL is stored as a scaled integer, so the `Value` alone does not
//! say where the point goes), and `Ty::unify` is exercised directly where the *type*
//! matters, since this crate's `typeof` reports a bare `DECIMAL` with no precision/scale.

use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::{Ty, Value};

fn data(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// A `SELECT <expr>` with no `FROM` is unsupported (`plan::bind`), so scalar expressions
/// run against `tests/data/basic.parquet` with `LIMIT 1` -- the same convention as
/// `scalar_function_fixes.rs`.
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

/// The first column of the first row of `sql`.
fn first(sql: &str) -> Value {
    let mut s = session_with_basic();
    let rows = run(&mut s, sql);
    assert!(!rows.is_empty(), "{sql}: expected at least one row");
    rows[0][0].clone()
}

/// `expr`, cast to VARCHAR so the exact spelling is what gets compared.
fn spelled(expr: &str) -> String {
    let sql = format!("SELECT CAST(({expr}) AS VARCHAR) AS x FROM t LIMIT 1");
    match first(&sql) {
        Value::Bytes(b) => String::from_utf8(b).expect("utf8"),
        v => panic!("{sql}: expected a VARCHAR result, got {v:?}"),
    }
}

fn boolean(expr: &str) -> Option<bool> {
    let sql = format!("SELECT ({expr}) AS x FROM t LIMIT 1");
    match first(&sql) {
        Value::Bool(b) => Some(b),
        Value::Null => None,
        v => panic!("{sql}: expected BOOLEAN, got {v:?}"),
    }
}

fn double(expr: &str) -> f64 {
    double_of(&format!("SELECT ({expr}) AS x FROM t LIMIT 1"))
}

/// The DOUBLE a whole query returns. Used where the query shape matters (window frames).
fn double_of(sql: &str) -> f64 {
    match first(sql) {
        Value::F64(x) => x,
        v => panic!("{sql}: expected DOUBLE, got {v:?}"),
    }
}

// ---- DECIMAL mixed with an integer type ----------------------------------
//
// `Ty::rank` puts DECIMAL between BIGINT and HUGEINT, so the rank-based widening in
// `Ty::unify` used to resolve `DECIMAL(4,1) + HUGEINT` to HUGEINT and round 7.5 to 8
// before adding. `*` and `/` escaped that because `plan::compile::decimal_arith`
// intercepts them first, which left the operators mutually inconsistent.

#[test]
fn decimal_plus_hugeint_keeps_the_fractional_part() {
    // duckdb: 8.5
    assert_eq!(spelled("CAST('7.5' AS DECIMAL(4,1)) + 1::HUGEINT"), "8.5");
    // duckdb: 6.5
    assert_eq!(spelled("CAST('7.5' AS DECIMAL(4,1)) - 1::HUGEINT"), "6.5");
    // Multiplication and division were already correct; they must stay so.
    // duckdb: 15.0 / 3.75
    assert_eq!(spelled("CAST('7.5' AS DECIMAL(4,1)) * 2::HUGEINT"), "15.0");
    assert_eq!(spelled("CAST('7.5' AS DECIMAL(4,1)) / 2::HUGEINT"), "3.75");
}

#[test]
fn decimal_plus_integer_widens_enough_for_the_integer_digits() {
    // The result type has to leave room for the integer side's digits, not just reuse
    // the DECIMAL's own precision. duckdb: 10998.9 (DECIMAL(21,1)); this used to
    // overflow DECIMAL(4,1) and come back NULL.
    assert_eq!(spelled("CAST('999.9' AS DECIMAL(4,1)) + 9999::BIGINT"), "10998.9");
    // duckdb: -8999.1
    assert_eq!(spelled("CAST('999.9' AS DECIMAL(4,1)) - 9999::INTEGER"), "-8999.1");
}

#[test]
fn comparing_a_decimal_with_an_integer_does_not_round_the_decimal() {
    // duckdb: true, false, true, false
    assert_eq!(boolean("CAST('7.5' AS DECIMAL(4,1)) < 8::HUGEINT"), Some(true));
    assert_eq!(boolean("CAST('7.5' AS DECIMAL(4,1)) = 8::HUGEINT"), Some(false));
    assert_eq!(boolean("CAST('7.5' AS DECIMAL(4,1)) > 7::HUGEINT"), Some(true));
    assert_eq!(boolean("CAST('7.5' AS DECIMAL(4,1)) <= 7::HUGEINT"), Some(false));
}

/// The exact result types, cross-checked against `duckdb -c "SELECT typeof(...)"`.
///
/// `unify` is what both the arithmetic and the comparison paths in `plan::compile` call,
/// so pinning it here covers every operator at once.
#[test]
fn unify_matches_duckdbs_decimal_result_types() {
    let d = |p, s| Ty::Decimal { precision: p, scale: s };
    // duckdb: DECIMAL(4,1) + INTEGER -> DECIMAL(12,1)
    assert_eq!(Ty::unify(d(4, 1), Ty::Int), Some(d(12, 1)));
    // duckdb: DECIMAL(4,1) + BIGINT -> DECIMAL(21,1)
    assert_eq!(Ty::unify(d(4, 1), Ty::BigInt), Some(d(21, 1)));
    // duckdb: DECIMAL(4,1) + HUGEINT -> DECIMAL(38,1), i.e. capped at the maximum
    // precision rather than the 40 the formula asks for.
    assert_eq!(Ty::unify(d(4, 1), Ty::HugeInt), Some(d(38, 1)));
    // Order must not matter.
    assert_eq!(Ty::unify(Ty::HugeInt, d(4, 1)), Some(d(38, 1)));
    // Unsigned integers carry the same digit counts as their signed partners.
    assert_eq!(Ty::unify(d(4, 1), Ty::UBigInt), Some(d(21, 1)));
    assert_eq!(Ty::unify(d(4, 1), Ty::TinyInt), Some(d(5, 1)));
}

/// The cases the fix must leave alone.
#[test]
fn unify_still_widens_integers_and_floats_the_way_it_did() {
    let d = |p, s| Ty::Decimal { precision: p, scale: s };
    // Two integers: plain rank-based widening, no DECIMAL anywhere.
    assert_eq!(Ty::unify(Ty::Int, Ty::BigInt), Some(Ty::BigInt));
    assert_eq!(Ty::unify(Ty::BigInt, Ty::HugeInt), Some(Ty::HugeInt));
    assert_eq!(Ty::unify(Ty::TinyInt, Ty::SmallInt), Some(Ty::SmallInt));
    // Two DECIMALs: align the scales and carry one digit. duckdb: DECIMAL(6,2).
    assert_eq!(Ty::unify(d(4, 1), d(5, 2)), Some(d(6, 2)));
    assert_eq!(Ty::unify(d(4, 1), d(4, 1)), Some(d(4, 1)));
    // DECIMAL with floating point still drops to DOUBLE, in both orders.
    assert_eq!(Ty::unify(d(4, 1), Ty::Double), Some(Ty::Double));
    assert_eq!(Ty::unify(Ty::Float, d(4, 1)), Some(Ty::Double));
    // NULL still passes the other side through untouched.
    assert_eq!(Ty::unify(Ty::Null, d(4, 1)), Some(d(4, 1)));
}

#[test]
fn decimal_and_decimal_arithmetic_is_unchanged() {
    // duckdb: 8.75
    assert_eq!(spelled("CAST('1.25' AS DECIMAL(5,2)) + CAST('7.5' AS DECIMAL(4,1))"), "8.75");
    // duckdb: 3.125 (multiplication adds the scales)
    assert_eq!(spelled("CAST('1.25' AS DECIMAL(5,2)) * CAST('2.5' AS DECIMAL(4,1))"), "3.125");
}

// ---- window SUM/AVG over DOUBLE ------------------------------------------
//
// `exec::agg` sums doubles with Neumaier compensation; `exec::window` had its own
// accumulator that did not, so the same rows gave different answers depending on which
// operator ran them.

#[test]
fn window_sum_over_doubles_is_compensated() {
    // duckdb: 1.0. Naive summation lands on 0.9999999999999999.
    assert_eq!(
        double_of("SELECT sum(x) OVER () FROM (SELECT 0.1 AS x FROM range(10)) LIMIT 1"),
        1.0
    );
    // The blocking aggregate over the identical rows must agree.
    assert_eq!(double_of("SELECT sum(x) FROM (SELECT 0.1 AS x FROM range(10))"), 1.0);
}

#[test]
fn window_avg_over_doubles_is_compensated() {
    // duckdb: 0.1
    assert_eq!(
        double_of("SELECT avg(x) OVER () FROM (SELECT 0.1 AS x FROM range(10)) LIMIT 1"),
        0.1
    );
    assert_eq!(double_of("SELECT avg(x) FROM (SELECT 0.1 AS x FROM range(10))"), 0.1);
}

/// The frame advances one peer group at a time and the accumulator is reused across
/// groups, so the compensation term must be folded in without being consumed: every
/// emitted running total has to stay compensated, not just the first.
///
/// The invariant asserted is "the running window sum after `n` rows equals the blocking
/// `sum` over those same `n` rows", which is exactly what the fix is for. Pinning literal
/// decimals here would be pinning something else: the engine's compensated total for
/// three 0.1s lands on the exact midpoint between two doubles and rounds to even
/// (0.30000000000000004), while DuckDB answers 0.3 -- a pre-existing difference in
/// `exec::agg`'s summation that this fix neither introduces nor is scoped to remove. What
/// the fix does guarantee is that both of this engine's paths now say the same thing.
#[test]
fn a_running_window_sum_stays_compensated_at_every_step() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT sum(x) OVER (ORDER BY i) AS s \
         FROM (SELECT i, 0.1 AS x FROM range(10) t(i)) ORDER BY s",
    );
    let running: Vec<f64> = rows
        .iter()
        .map(|r| match r[0] {
            Value::F64(x) => x,
            ref v => panic!("expected DOUBLE, got {v:?}"),
        })
        .collect();
    assert_eq!(running.len(), 10);
    for (i, got) in running.iter().enumerate() {
        let n = i + 1;
        let blocking = double_of(&format!("SELECT sum(x) FROM (SELECT 0.1 AS x FROM range({n}))"));
        assert_eq!(*got, blocking, "running sum after {n} rows");
    }
    // The full frame is the headline case: 1.0, not the naive 0.9999999999999999.
    assert_eq!(running[9], 1.0);
}

/// A large value followed by tiny ones. Naive summation drops the small ones entirely
/// (1e16 + 1.0 == 1e16), and plain Kahan is also lossy when one addend dominates the
/// running total; Neumaier is not. Again the assertion is window-equals-blocking, plus
/// "not the naive answer".
#[test]
fn window_sum_recovers_bits_from_a_dominant_first_value() {
    const ROWS: &str = "(SELECT CASE WHEN i = 0 THEN 1e16 ELSE 1.0 END AS x FROM range(4) t(i))";
    let windowed = double_of(&format!("SELECT sum(x) OVER () AS s FROM {ROWS} LIMIT 1"));
    let blocking = double_of(&format!("SELECT sum(x) AS s FROM {ROWS}"));
    assert_eq!(windowed, blocking);
    assert_ne!(windowed, 1e16, "the naive total loses all three small addends");
}

// ---- CAST(<double> AS VARCHAR) -------------------------------------------

/// The formatter the CLI renderer now shares. Pinned here so a change to either side
/// shows up as a failure rather than as a silent divergence between the two spellings.
#[test]
fn double_to_varchar_uses_exponential_notation_at_the_extremes() {
    // duckdb: 1e+30, 1e-07, 1000000000000000.0, 1e+16, 0.1
    assert_eq!(spelled("1e30::DOUBLE"), "1e+30");
    assert_eq!(spelled("1e-7::DOUBLE"), "1e-07");
    assert_eq!(spelled("1e15::DOUBLE"), "1000000000000000.0");
    assert_eq!(spelled("1e16::DOUBLE"), "1e+16");
    assert_eq!(spelled("0.1::DOUBLE"), "0.1");
}

/// Non-finite doubles use DuckDB's lowercase spellings, and the cast is reversible.
///
/// The CSV and JSONL writers deliberately keep their own (`NaN` / `Infinity` /
/// `-Infinity`, quoted as JSON strings in JSONL) so their output stays readable by both
/// engines; `write_and_codec.rs` pins those. Only the display/cast path is aligned here.
#[test]
fn non_finite_doubles_use_duckdbs_spellings_and_round_trip() {
    // duckdb: nan / inf / -inf
    assert_eq!(spelled("'nan'::DOUBLE"), "nan");
    assert_eq!(spelled("'inf'::DOUBLE"), "inf");
    assert_eq!(spelled("'-inf'::DOUBLE"), "-inf");
    // `parse_special_f64` is case-insensitive, so the text the cast writes reads back.
    assert_eq!(double("CAST(CAST('inf'::DOUBLE AS VARCHAR) AS DOUBLE)"), f64::INFINITY);
    assert_eq!(double("CAST(CAST('-inf'::DOUBLE AS VARCHAR) AS DOUBLE)"), f64::NEG_INFINITY);
    assert!(double("CAST(CAST('nan'::DOUBLE AS VARCHAR) AS DOUBLE)").is_nan());
}
