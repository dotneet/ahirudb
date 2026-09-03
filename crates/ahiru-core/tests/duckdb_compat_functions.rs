//! Coverage for the DuckDB-compatibility functions added on top of the original
//! set: the string/numeric/date-time/LIST scalar functions in
//! `crates/ahiru-core/src/expr/funcs.rs`, the aggregates in
//! `crates/ahiru-core/src/plan/mod.rs` + `crates/ahiru-core/src/exec/agg.rs`,
//! and the ranking window functions in `crates/ahiru-core/src/exec/window.rs`.
//!
//! Expected values were cross-checked against the `duckdb` CLI. Where this
//! engine deliberately diverges (an undefined argument gives NULL rather than
//! raising, `quantile` is the continuous version, ...) the divergence is
//! called out inline instead of asserted against DuckDB's output.

use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn session_with_csv(name: &str, csv: &str) -> Session {
    let mut s = Session::new();
    s.register_bytes_as(name, csv.as_bytes().to_vec(), FormatKind::Csv).unwrap();
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

/// Like `run`, but hands back the error instead of unwrapping. Some rejections
/// only happen once an operator runs, not at prepare time.
fn try_run(session: &mut Session, sql: &str) -> Result<(), ahiru_core::error::Error> {
    let mut q = match session.prepare(sql, &[])? {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
    };
    while !matches!(session.step(&mut q)?, QueryStep::Done) {}
    Ok(())
}

/// A one-row session to evaluate constant expressions against (the same
/// CSV-`dual` trick the other integration tests use).
fn dual() -> Session {
    session_with_csv("dual", "x\n1\n")
}

fn one(session: &mut Session, expr: &str) -> Value {
    run(session, &format!("SELECT {expr} AS v FROM dual")).remove(0).remove(0)
}

fn s(v: &str) -> Value {
    Value::Bytes(v.as_bytes().to_vec())
}

fn f64_of(v: &Value) -> f64 {
    match v {
        Value::F64(x) => *x,
        other => panic!("not f64: {other:?}"),
    }
}

// =============================================================================
// Strings
// =============================================================================

#[test]
fn concat_ws_skips_null_values_but_propagates_a_null_separator() {
    let mut sess = dual();
    // duckdb: `a-b` -- the NULL argument drops the separator that would precede it.
    assert_eq!(one(&mut sess, "concat_ws('-', 'a', NULL, 'b')"), s("a-b"));
    assert_eq!(one(&mut sess, "concat_ws('-', 'a')"), s("a"));
    // A NULL separator makes the whole row NULL (unlike a NULL value).
    assert_eq!(one(&mut sess, "concat_ws(NULL, 'a', 'b')"), Value::Null);
}

#[test]
fn left_and_right_count_code_points_and_accept_negative_counts() {
    let mut sess = dual();
    assert_eq!(one(&mut sess, "left('abcde', 2)"), s("ab"));
    assert_eq!(one(&mut sess, "right('abcde', 2)"), s("de"));
    // Negative means "all but the last / first |n| characters" (duckdb).
    assert_eq!(one(&mut sess, "left('abcde', -2)"), s("abc"));
    assert_eq!(one(&mut sess, "right('abcde', -2)"), s("cde"));
    // Beyond the string's length is the whole string, not an error.
    assert_eq!(one(&mut sess, "left('abc', 99)"), s("abc"));
    assert_eq!(one(&mut sess, "left('abc', -99)"), s(""));
    // Code points, not bytes.
    assert_eq!(one(&mut sess, "left('あいう', 2)"), s("あい"));
    assert_eq!(one(&mut sess, "right('あいう', 1)"), s("う"));
}

#[test]
fn ascii_and_chr_round_trip_through_code_points() {
    let mut sess = dual();
    assert_eq!(one(&mut sess, "ascii('A')"), Value::I64(65));
    // The code point of the first character, not its first byte.
    assert_eq!(one(&mut sess, "ascii('あ')"), Value::I64(12354));
    assert_eq!(one(&mut sess, "chr(12354)"), s("あ"));
    assert_eq!(one(&mut sess, "chr(ascii('☃'))"), s("☃"));
    // The empty string has no first character, and duckdb answers 0 there rather than NULL.
    assert_eq!(one(&mut sess, "ascii('')"), Value::I64(0));
    // An unencodable code point gives NULL rather than raising.
    assert_eq!(one(&mut sess, "chr(-1)"), Value::Null);
}

#[test]
fn hex_dumps_bytes_and_to_hex_renders_integers() {
    let mut sess = dual();
    assert_eq!(one(&mut sess, "hex('AB')"), s("4142"));
    // An integer argument picks the base-16 rendering, matching duckdb's overload.
    assert_eq!(one(&mut sess, "hex(255)"), s("FF"));
    assert_eq!(one(&mut sess, "to_hex(255)"), s("FF"));
    assert_eq!(one(&mut sess, "to_hex(0)"), s("0"));
    // Negatives go through their two's-complement bit pattern (duckdb does the same).
    assert_eq!(one(&mut sess, "to_hex(-1)"), s("FFFFFFFFFFFFFFFF"));
}

#[test]
fn string_split_produces_a_json_list() {
    let mut sess = dual();
    assert_eq!(one(&mut sess, "string_split('a,b,c', ',')"), s(r#"["a","b","c"]"#));
    assert_eq!(one(&mut sess, "str_split('a', ',')"), s(r#"["a"]"#));
    // Adjacent separators keep the empty pieces (duckdb: `[a, , b]`).
    assert_eq!(one(&mut sess, "string_split('a,,b', ',')"), s(r#"["a","","b"]"#));
    // An empty separator splits into characters (code points), as duckdb does. The empty
    // string still yields one (empty) piece.
    assert_eq!(one(&mut sess, "string_split('abc', '')"), s(r#"["a","b","c"]"#));
    assert_eq!(one(&mut sess, "string_split('あい', '')"), s(r#"["あ","い"]"#));
    assert_eq!(one(&mut sess, "string_split('', '')"), s(r#"[""]"#));
    // `split_part` agrees with it piece for piece.
    assert_eq!(one(&mut sess, "split_part('abc', '', 2)"), s("b"));
    assert_eq!(one(&mut sess, "split_part('abc', '', -1)"), s("c"));
    assert_eq!(one(&mut sess, "split_part('あい', '', 1)"), s("あ"));
    assert_eq!(one(&mut sess, "split_part('abc', '', 4)"), s(""));
    // It really is a list: the existing LIST functions accept the result.
    assert_eq!(one(&mut sess, "array_length(string_split('a,b,c', ','))"), Value::I64(3));
}

// =============================================================================
// Numbers
// =============================================================================

#[test]
fn logarithms_and_roots() {
    let mut sess = dual();
    assert_eq!(f64_of(&one(&mut sess, "log2(8)")), 3.0);
    // The two-argument form is base `b`.
    assert_eq!(f64_of(&one(&mut sess, "log(2, 8)")), 3.0);
    // The one-argument form stays the common logarithm. (`log10` derives from the in-house
    // `ln`, so it is accurate to a few ulps rather than exact.)
    assert!((f64_of(&one(&mut sess, "log(100)")) - 2.0).abs() < 1e-12);
    assert!((f64_of(&one(&mut sess, "cbrt(27)")) - 3.0).abs() < 1e-12);
    // Unlike sqrt, cbrt is defined for negative input.
    assert!((f64_of(&one(&mut sess, "cbrt(-8)")) + 2.0).abs() < 1e-12);
    // Undefined domains give NULL rather than raising (this engine's convention).
    assert_eq!(one(&mut sess, "log2(0)"), Value::Null);
    assert_eq!(one(&mut sess, "log(1, 8)"), Value::Null);
}

#[test]
fn angle_conversion_and_pi() {
    let mut sess = dual();
    assert!((f64_of(&one(&mut sess, "pi()")) - core::f64::consts::PI).abs() < 1e-15);
    assert!((f64_of(&one(&mut sess, "radians(180)")) - core::f64::consts::PI).abs() < 1e-15);
    assert!((f64_of(&one(&mut sess, "degrees(pi())")) - 180.0).abs() < 1e-12);
}

#[test]
fn integer_helpers() {
    let mut sess = dual();
    assert_eq!(one(&mut sess, "gcd(12, 18)"), Value::I64(6));
    // Always non-negative (duckdb: `gcd(-4, 6)` -> 2).
    assert_eq!(one(&mut sess, "gcd(-4, 6)"), Value::I64(2));
    assert_eq!(one(&mut sess, "gcd(0, 0)"), Value::I64(0));
    assert_eq!(one(&mut sess, "lcm(4, 6)"), Value::I64(12));
    assert_eq!(one(&mut sess, "lcm(0, 5)"), Value::I64(0));
    assert_eq!(one(&mut sess, "bit_count(7)"), Value::I64(3));
    assert_eq!(one(&mut sess, "xor(5, 3)"), Value::I64(6));
    // The mathematical gcd of i64::MIN and zero is 2^63, which does not fit
    // BIGINT. Do not silently clamp it to i64::MAX.
    assert_eq!(one(&mut sess, "gcd(-9223372036854775808, 0)"), Value::Null);
    // LCM can still resolve the same boundary input without taking abs(MIN):
    // DuckDB short-circuits either zero factor to zero.
    assert_eq!(one(&mut sess, "lcm(-9223372036854775808, 0)"), Value::I64(0));
    assert_eq!(one(&mut sess, "lcm(0, -9223372036854775808)"), Value::I64(0));
}

#[test]
fn integer_and_decimal_casts_enforce_logical_ranges() {
    let mut sess = dual();
    assert_eq!(one(&mut sess, "TRY_CAST('128' AS TINYINT)"), Value::Null);
    assert_eq!(one(&mut sess, "TRY_CAST(-1 AS UTINYINT)"), Value::Null);
    assert_eq!(one(&mut sess, "TRY_CAST(12345 AS DECIMAL(4, 0))"), Value::Null);
    assert_eq!(one(&mut sess, "TRY_CAST(9999 AS DECIMAL(4, 0))"), Value::I64(9999));
}

#[test]
fn boolean_text_cast_accepts_duckdb_yes_no_spellings() {
    let mut sess = dual();
    assert_eq!(one(&mut sess, "TRY_CAST('yes' AS BOOLEAN)"), Value::Bool(true));
    assert_eq!(one(&mut sess, "TRY_CAST('y' AS BOOLEAN)"), Value::Bool(true));
    assert_eq!(one(&mut sess, "TRY_CAST('no' AS BOOLEAN)"), Value::Bool(false));
    assert_eq!(one(&mut sess, "TRY_CAST('n' AS BOOLEAN)"), Value::Bool(false));
    // These are not DuckDB boolean spellings and remain NULL. The parser is
    // intentionally exact rather than trimming arbitrary text.
    assert_eq!(one(&mut sess, "TRY_CAST(' yes ' AS BOOLEAN)"), Value::Null);
    assert_eq!(one(&mut sess, "TRY_CAST('on' AS BOOLEAN)"), Value::Null);
}

#[test]
fn float_classification_predicates() {
    let mut sess = dual();
    assert_eq!(one(&mut sess, "isnan(0.0 / 0.0)"), Value::Bool(true));
    assert_eq!(one(&mut sess, "isnan(1.0)"), Value::Bool(false));
    assert_eq!(one(&mut sess, "isfinite(1.0)"), Value::Bool(true));
    assert_eq!(one(&mut sess, "isinf(1.0)"), Value::Bool(false));
    // NULL in, NULL out (the default propagation, as in duckdb).
    assert_eq!(one(&mut sess, "isnan(NULL)"), Value::Null);
}

#[test]
fn typeof_reports_the_static_type_including_for_null() {
    let mut sess = dual();
    assert_eq!(one(&mut sess, "typeof(1.5)"), s("DOUBLE"));
    assert_eq!(one(&mut sess, "typeof('a')"), s("VARCHAR"));
    assert_eq!(one(&mut sess, "typeof(DATE '2024-01-01')"), s("DATE"));
    assert_eq!(one(&mut sess, "typeof([1, 2])"), s("JSON"));
    // `typeof` names the type; it is never itself NULL.
    assert_eq!(one(&mut sess, "typeof(NULL)"), s("NULL"));
    // A real column's declared type, not the row's value.
    assert_eq!(one(&mut sess, "typeof(x)"), s("BIGINT"));
}

// =============================================================================
// Date and time
// =============================================================================

#[test]
fn day_and_month_names() {
    let mut sess = dual();
    assert_eq!(one(&mut sess, "dayname(DATE '2024-08-14')"), s("Wednesday"));
    assert_eq!(one(&mut sess, "monthname(DATE '2024-08-14')"), s("August"));
}

#[test]
fn make_date_and_make_timestamp_validate_their_components() {
    let mut sess = dual();
    assert_eq!(one(&mut sess, "CAST(make_date(2024, 2, 29) AS VARCHAR)"), s("2024-02-29"));
    assert_eq!(
        one(&mut sess, "CAST(make_timestamp(2024, 8, 14, 13, 45, 30) AS VARCHAR)"),
        s("2024-08-14 13:45:30")
    );
    // 2023 is not a leap year. duckdb raises here; NULL is this engine's
    // convention for an out-of-range argument (see docs/sql/limitations.md).
    assert_eq!(one(&mut sess, "make_date(2023, 2, 29)"), Value::Null);
    assert_eq!(one(&mut sess, "make_date(2024, 13, 1)"), Value::Null);

    // duckdb accepts a time of day anywhere in [00:00:00, 24:00:00] and normalizes what
    // the components add up to; only what falls outside that is an error (NULL here).
    // duckdb: 2024-01-02 00:00:00 / 2024-06-05 07:09:00.
    assert_eq!(
        one(&mut sess, "CAST(make_timestamp(2024, 1, 1, 24, 0, 0) AS VARCHAR)"),
        s("2024-01-02 00:00:00")
    );
    assert_eq!(
        one(&mut sess, "CAST(make_timestamp(2024, 6, 5, 7, 8, 60) AS VARCHAR)"),
        s("2024-06-05 07:09:00")
    );
    // duckdb: "Time out of range" for each of these.
    assert_eq!(one(&mut sess, "make_timestamp(2024, 1, 1, 24, 0, 1)"), Value::Null);
    assert_eq!(one(&mut sess, "make_timestamp(2024, 1, 1, 25, 0, 0)"), Value::Null);
    assert_eq!(one(&mut sess, "make_timestamp(2024, 1, 1, 0, 60, 0)"), Value::Null);
    assert_eq!(one(&mut sess, "make_timestamp(2024, 1, 1, 0, 0, 61)"), Value::Null);
    assert_eq!(one(&mut sess, "make_timestamp(2024, 1, 1, -1, 0, 0)"), Value::Null);
}

#[test]
fn date_arithmetic_rejects_duckdb_infinity_sentinels() {
    let mut sess = dual();
    // DuckDB reserves three i32 values for DATE special values. AhiruDB does
    // not expose those literals, so arithmetic reaching any of them must be
    // the existing per-row NULL for an out-of-range DATE, not a fake date.
    assert_eq!(one(&mut sess, "CAST(DATE '1970-01-01' + 2147483647 AS VARCHAR)"), Value::Null);
    assert_eq!(one(&mut sess, "CAST(DATE '1970-01-01' - 2147483647 AS VARCHAR)"), Value::Null);
    assert_eq!(
        one(&mut sess, "CAST(DATE '1970-01-01' + 2147483646 AS VARCHAR)"),
        s("5881580-07-10")
    );
    assert_eq!(
        one(&mut sess, "CAST(DATE '1970-01-01' - 2147483646 AS VARCHAR)"),
        s("-5877641-06-25")
    );
    // The difference between the two finite endpoints is wider than I32;
    // DATE - DATE is an INTEGER/BIGINT day count, not a wrapping I32 result.
    assert_eq!(
        one(&mut sess, "(DATE '1970-01-01' + 2147483646) - (DATE '1970-01-01' - 2147483646)"),
        Value::I64(4_294_967_292)
    );
}

#[test]
fn time_endpoint_24_hours_round_trips_without_wrapping() {
    let mut sess = dual();
    // DuckDB preserves the valid TIME endpoint rather than formatting it as midnight.
    assert_eq!(one(&mut sess, "CAST(CAST('24:00:00' AS TIME) AS VARCHAR)"), s("24:00:00"));
}

#[test]
fn sub_second_parts_include_the_seconds_field() {
    let mut sess = dual();
    let ts = "TIMESTAMP '2021-08-03 11:59:44.123456'";
    // duckdb: 44123 / 44123456 -- both roll the whole seconds in.
    assert_eq!(one(&mut sess, &format!("millisecond({ts})")), Value::I64(44123));
    assert_eq!(one(&mut sess, &format!("microsecond({ts})")), Value::I64(44123456));
    // The same parts are reachable through date_part, short aliases included.
    assert_eq!(one(&mut sess, &format!("date_part('ms', {ts})")), Value::I64(44123));
    assert_eq!(one(&mut sess, &format!("date_part('microseconds', {ts})")), Value::I64(44123456));
}

#[test]
fn calendar_parts_isodow_century_and_decade() {
    let mut sess = dual();
    // 2024-08-11 is a Sunday: dow 0, isodow 7.
    assert_eq!(one(&mut sess, "dayofweek(DATE '2024-08-11')"), Value::I64(0));
    assert_eq!(one(&mut sess, "isodow(DATE '2024-08-11')"), Value::I64(7));
    // duckdb: century(2021) = 21, century(2000) = 20, decade(2021) = 202.
    assert_eq!(one(&mut sess, "century(DATE '2021-01-01')"), Value::I64(21));
    assert_eq!(one(&mut sess, "century(DATE '2000-01-01')"), Value::I64(20));
    assert_eq!(one(&mut sess, "decade(DATE '2021-01-01')"), Value::I64(202));
}

#[test]
fn timestamp_text_cast_accepts_and_ignores_iso8601_zone_suffixes() {
    let mut sess = dual();
    // DuckDB's TIMESTAMP is without time zone: Z/offset suffixes are
    // accepted, validated, and ignored rather than applied to the wall clock.
    let plain = one(&mut sess, "CAST('2024-01-05 10:20:30' AS TIMESTAMP)");
    assert_eq!(one(&mut sess, "CAST('2024-01-05 10:20:30Z' AS TIMESTAMP)"), plain);
    assert_eq!(one(&mut sess, "CAST('2024-01-05 10:20:30+09:00' AS TIMESTAMP)"), plain);
    assert_eq!(one(&mut sess, "CAST('2024-01-05 10:20:30+0900' AS TIMESTAMP)"), plain);
    assert_eq!(one(&mut sess, "CAST('2024-01-05 10:20:30+99:99' AS TIMESTAMP)"), plain);
    assert_eq!(one(&mut sess, "CAST('2024-01-05 10:20:30-03:30' AS TIMESTAMP)"), plain);
    assert_eq!(
        one(&mut sess, "TRY_CAST('2024-01-05 10:20:30+09:00junk' AS TIMESTAMP)"),
        Value::Null
    );
    assert_eq!(one(&mut sess, "TRY_CAST('2024-01-05 10:20:30+9' AS TIMESTAMP)"), Value::Null);
    assert_eq!(one(&mut sess, "TRY_CAST('2024-01-05 10:20:30z' AS TIMESTAMP)"), Value::Null);
}

#[test]
fn epoch_variants_rescale_the_same_timestamp() {
    let mut sess = dual();
    let ts = "TIMESTAMP '1970-01-01 00:00:01.5'";
    assert_eq!(one(&mut sess, &format!("epoch({ts})")), Value::I64(1));
    assert_eq!(one(&mut sess, &format!("epoch_ms({ts})")), Value::I64(1_500));
    assert_eq!(one(&mut sess, &format!("epoch_us({ts})")), Value::I64(1_500_000));
    assert_eq!(one(&mut sess, &format!("epoch_ns({ts})")), Value::I64(1_500_000_000));
}

#[test]
fn epoch_milliseconds_truncate_negative_subseconds_toward_zero() {
    let mut sess = dual();
    // DuckDB: epoch_ms(TIMESTAMP '1969-12-31 23:59:59.999999') = 0.
    // Floor division would incorrectly return -1 here.
    assert_eq!(one(&mut sess, "epoch_ms(TIMESTAMP '1969-12-31 23:59:59.999999')",), Value::I64(0));
    assert_eq!(
        one(&mut sess, "epoch_ms(TIMESTAMP '1969-12-31 23:59:58.999999')",),
        Value::I64(-1000)
    );
}

#[test]
fn sub_second_parts_work_with_trunc_diff_and_add() {
    let mut sess = dual();
    let a = "TIMESTAMP '2024-01-01 00:00:00.123456'";
    assert_eq!(
        one(&mut sess, &format!("CAST(date_trunc('millisecond', {a}) AS VARCHAR)")),
        s("2024-01-01 00:00:00.123")
    );
    assert_eq!(
        one(&mut sess, &format!("date_diff('microsecond', TIMESTAMP '2024-01-01', {a})")),
        Value::I64(123_456)
    );
    assert_eq!(
        one(&mut sess, "CAST(date_add('millisecond', 250, TIMESTAMP '2024-01-01') AS VARCHAR)"),
        s("2024-01-01 00:00:00.25")
    );
}

// =============================================================================
// LIST functions (over the JSON representation)
// =============================================================================

#[test]
fn list_membership_and_position() {
    let mut sess = dual();
    assert_eq!(one(&mut sess, "list_contains([1, 2, 3], 2)"), Value::Bool(true));
    assert_eq!(one(&mut sess, "list_contains([1, 2, 3], 9)"), Value::Bool(false));
    assert_eq!(one(&mut sess, "list_contains(['a', 'b'], 'b')"), Value::Bool(true));
    assert_eq!(one(&mut sess, "list_position(['a', 'b'], 'b')"), Value::I64(2));
    // duckdb returns NULL, not 0, when the element is absent.
    assert_eq!(one(&mut sess, "list_position(['a', 'b'], 'z')"), Value::Null);
    // Unlike list_contains, list_position treats a NULL needle as a
    // searchable NULL list element.
    assert_eq!(one(&mut sess, "list_position([1, 2, NULL], NULL)"), Value::I64(3));
    // A non-list argument is NULL rather than false.
    assert_eq!(one(&mut sess, "list_contains(CAST('5' AS JSON), 5)"), Value::Null);
}

#[test]
fn list_sort_distinct_and_reverse() {
    let mut sess = dual();
    assert_eq!(one(&mut sess, "list_sort([3, 1, 2])"), s("[1,2,3]"));
    assert_eq!(one(&mut sess, "list_sort(['b', 'a'])"), s(r#"["a","b"]"#));
    // Numbers sort numerically, not lexicographically.
    assert_eq!(one(&mut sess, "list_sort([10, 9])"), s("[9,10]"));
    // First-occurrence order is kept.
    assert_eq!(one(&mut sess, "list_distinct([1, 2, 1, 3])"), s("[1,2,3]"));
    // SQL NULL elements are discarded, matching DuckDB.
    assert_eq!(one(&mut sess, "list_distinct([1, NULL, 1, NULL])"), s("[1]"));
    assert_eq!(one(&mut sess, "list_reverse([1, 2, 3])"), s("[3,2,1]"));
    assert_eq!(one(&mut sess, "array_length(list_sort([3, 1, 2]))"), Value::I64(3));
}

// =============================================================================
// Aggregates
// =============================================================================

fn agg_session() -> Session {
    session_with_csv("t", "x,s\n1,a\n2,b\n3,c\n4,d\n5,e\n")
}

#[test]
fn value_carrying_aggregates() {
    let mut sess = agg_session();
    let rows = run(&mut sess, "SELECT any_value(x), first(x), last(x) FROM t");
    assert_eq!(rows[0][0], Value::I64(1));
    assert_eq!(rows[0][1], Value::I64(1));
    assert_eq!(rows[0][2], Value::I64(5));
}

#[test]
fn boolean_aggregates_and_count_if() {
    let mut sess = agg_session();
    let rows = run(
        &mut sess,
        "SELECT bool_and(x > 0), bool_and(x > 1), bool_or(x > 4), count_if(x > 2) FROM t",
    );
    assert_eq!(rows[0][0], Value::Bool(true));
    assert_eq!(rows[0][1], Value::Bool(false));
    assert_eq!(rows[0][2], Value::Bool(true));
    assert_eq!(rows[0][3], Value::I64(3));
    // A group with no true row counts 0, not NULL.
    let rows = run(&mut sess, "SELECT count_if(x > 99) FROM t");
    assert_eq!(rows[0][0], Value::I64(0));
    // Only NULL inputs leave bool_and/bool_or NULL.
    let mut nulls = session_with_csv("t", "b\n\n\n");
    let rows = run(&mut nulls, "SELECT bool_and(CAST(b AS BOOLEAN)) FROM t");
    assert_eq!(rows[0][0], Value::Null);
}

#[test]
fn product_and_population_statistics() {
    let mut sess = session_with_csv("t", "x\n1\n2\n3\n4\n");
    // duckdb over [1,2,3,4]: product=24, var_pop=1.25, stddev_pop=1.118033988749895.
    let rows = run(&mut sess, "SELECT product(x), var_pop(x), stddev_pop(x) FROM t");
    assert!((f64_of(&rows[0][0]) - 24.0).abs() < 1e-9);
    assert!((f64_of(&rows[0][1]) - 1.25).abs() < 1e-9);
    assert!((f64_of(&rows[0][2]) - 1.118033988749895).abs() < 1e-9);
    // The population versions are defined from a single row (the sample ones are NULL there).
    let mut one_row = session_with_csv("t", "x\n7\n");
    let rows = run(&mut one_row, "SELECT var_pop(x), stddev_pop(x), variance(x) FROM t");
    assert!((f64_of(&rows[0][0])).abs() < 1e-12);
    assert!((f64_of(&rows[0][1])).abs() < 1e-12);
    assert_eq!(rows[0][2], Value::Null);
}

#[test]
fn quantile_cont_interpolates_and_matches_median_at_half() {
    let mut sess = agg_session();
    // duckdb over [1,2,3,4,5]: quantile_cont(x, 0.25) = 2, 0.5 = 3, 0.9 = 4.6.
    let rows = run(
        &mut sess,
        "SELECT quantile_cont(x, 0.25), quantile_cont(x, 0.5), quantile_cont(x, 0.9), median(x) FROM t",
    );
    assert!((f64_of(&rows[0][0]) - 2.0).abs() < 1e-9);
    assert!((f64_of(&rows[0][1]) - 3.0).abs() < 1e-9);
    assert!((f64_of(&rows[0][2]) - 4.6).abs() < 1e-9);
    assert_eq!(f64_of(&rows[0][1]), f64_of(&rows[0][3]));
    // The endpoints are the min and the max.
    let rows = run(&mut sess, "SELECT quantile_cont(x, 0.0), quantile_cont(x, 1.0) FROM t");
    assert!((f64_of(&rows[0][0]) - 1.0).abs() < 1e-9);
    assert!((f64_of(&rows[0][1]) - 5.0).abs() < 1e-9);
    // A fraction outside [0, 1] is rejected at prepare time rather than clamped.
    assert!(sess.prepare("SELECT quantile_cont(x, 2) FROM t", &[]).is_err());
    // So is a non-constant fraction.
    assert!(sess.prepare("SELECT quantile_cont(x, x) FROM t", &[]).is_err());
}

#[test]
fn arg_min_and_arg_max_pick_the_value_at_the_extreme_key() {
    let mut sess = agg_session();
    let rows =
        run(&mut sess, "SELECT arg_max(s, x), arg_min(s, x), max_by(s, x), min_by(s, x) FROM t");
    assert_eq!(rows[0][0], s("e"));
    assert_eq!(rows[0][1], s("a"));
    assert_eq!(rows[0][2], s("e"));
    assert_eq!(rows[0][3], s("a"));
    // A row whose key is NULL takes no part.
    let mut with_null = session_with_csv("t", "x,s\n1,a\n,z\n3,c\n");
    let rows = run(&mut with_null, "SELECT arg_max(s, x) FROM t");
    assert_eq!(rows[0][0], s("c"));
    // DISTINCT would have to deduplicate on the pair, so it is rejected rather
    // than silently deduplicating on one column.
    assert!(sess.prepare("SELECT arg_max(DISTINCT s, x) FROM t", &[]).is_err());
    // Both arguments are required.
    assert!(sess.prepare("SELECT arg_max(s) FROM t", &[]).is_err());
}

#[test]
fn new_aggregates_group_and_filter_like_the_existing_ones() {
    let mut sess = session_with_csv("t", "g,x\na,1\na,3\nb,2\nb,8\n");
    let rows = run(
        &mut sess,
        "SELECT g, count_if(x > 2), bool_or(x > 5), product(x) FROM t GROUP BY g ORDER BY g",
    );
    assert_eq!(rows[0][0], s("a"));
    assert_eq!(rows[0][1], Value::I64(1));
    assert_eq!(rows[0][2], Value::Bool(false));
    assert!((f64_of(&rows[0][3]) - 3.0).abs() < 1e-9);
    assert_eq!(rows[1][1], Value::I64(1));
    assert_eq!(rows[1][2], Value::Bool(true));
    assert!((f64_of(&rows[1][3]) - 16.0).abs() < 1e-9);
    // FILTER applies here too.
    let rows = run(&mut sess, "SELECT any_value(x) FILTER (WHERE x > 2) FROM t");
    assert_eq!(rows[0][0], Value::I64(3));
}

// =============================================================================
// Window functions
// =============================================================================

/// `[1, 2, 2, 3]` -- the repeated 2 makes the peer-group handling visible.
fn window_session() -> Session {
    session_with_csv("t", "x\n1\n2\n2\n3\n")
}

#[test]
fn ntile_splits_the_partition_into_even_buckets() {
    let mut sess = window_session();
    let rows = run(&mut sess, "SELECT ntile(2) OVER (ORDER BY x) FROM t");
    let got: Vec<Value> = rows.iter().map(|r| r[0].clone()).collect();
    assert_eq!(got, vec![Value::I64(1), Value::I64(1), Value::I64(2), Value::I64(2)]);
    // 4 rows into 3 buckets: the first bucket takes the extra row (duckdb: 1,1,2,3).
    let rows = run(&mut sess, "SELECT ntile(3) OVER (ORDER BY x) FROM t");
    let got: Vec<Value> = rows.iter().map(|r| r[0].clone()).collect();
    assert_eq!(got, vec![Value::I64(1), Value::I64(1), Value::I64(2), Value::I64(3)]);
    // More buckets than rows gives one row each.
    let rows = run(&mut sess, "SELECT ntile(99) OVER (ORDER BY x) FROM t");
    let got: Vec<Value> = rows.iter().map(|r| r[0].clone()).collect();
    assert_eq!(got, vec![Value::I64(1), Value::I64(2), Value::I64(3), Value::I64(4)]);
    // A non-positive bucket count is NULL (duckdb raises; NULL is this engine's convention).
    let rows = run(&mut sess, "SELECT ntile(0) OVER (ORDER BY x) FROM t");
    assert_eq!(rows[0][0], Value::Null);
}

#[test]
fn percent_rank_and_cume_dist_share_ranks_within_a_peer_group() {
    let mut sess = window_session();
    let rows = run(
        &mut sess,
        "SELECT percent_rank() OVER (ORDER BY x), cume_dist() OVER (ORDER BY x) FROM t",
    );
    // duckdb over [1,2,2,3]: percent_rank = 0, 1/3, 1/3, 1; cume_dist = .25, .75, .75, 1.
    let pr: Vec<f64> = rows.iter().map(|r| f64_of(&r[0])).collect();
    let cd: Vec<f64> = rows.iter().map(|r| f64_of(&r[1])).collect();
    assert!((pr[0]).abs() < 1e-12);
    assert!((pr[1] - 1.0 / 3.0).abs() < 1e-12);
    assert!((pr[2] - 1.0 / 3.0).abs() < 1e-12);
    assert!((pr[3] - 1.0).abs() < 1e-12);
    assert_eq!(cd, vec![0.25, 0.75, 0.75, 1.0]);
    // A single-row partition is percent_rank 0 / cume_dist 1.
    let mut single = session_with_csv("t", "x\n7\n");
    let rows = run(
        &mut single,
        "SELECT percent_rank() OVER (ORDER BY x), cume_dist() OVER (ORDER BY x) FROM t",
    );
    assert!(f64_of(&rows[0][0]).abs() < 1e-12);
    assert!((f64_of(&rows[0][1]) - 1.0).abs() < 1e-12);
}

#[test]
fn nth_value_follows_the_frame() {
    let mut sess = window_session();
    // With ORDER BY the frame grows peer group by peer group, so the second row
    // is not visible from the first (duckdb: NULL, 2, 2, 2).
    let rows = run(&mut sess, "SELECT nth_value(x, 2) OVER (ORDER BY x) FROM t");
    let got: Vec<Value> = rows.iter().map(|r| r[0].clone()).collect();
    assert_eq!(got, vec![Value::Null, Value::I64(2), Value::I64(2), Value::I64(2)]);
    // Without ORDER BY the frame is the whole partition.
    let rows = run(&mut sess, "SELECT nth_value(x, 2) OVER () FROM t");
    assert_eq!(rows[0][0], Value::I64(2));
    // Out of range is NULL.
    let rows = run(&mut sess, "SELECT nth_value(x, 99) OVER () FROM t");
    assert_eq!(rows[0][0], Value::Null);
}

#[test]
fn new_aggregates_also_work_as_window_functions() {
    let mut sess = window_session();
    let rows = run(
        &mut sess,
        "SELECT bool_or(x > 2) OVER (ORDER BY x), count_if(x > 1) OVER (ORDER BY x), \
         any_value(x) OVER (ORDER BY x), last(x) OVER (ORDER BY x) FROM t",
    );
    // The frame runs from the partition start through the current peer group.
    assert_eq!(rows[0][0], Value::Bool(false));
    assert_eq!(rows[3][0], Value::Bool(true));
    assert_eq!(rows[0][1], Value::I64(0));
    assert_eq!(rows[3][1], Value::I64(3));
    assert_eq!(rows[3][2], Value::I64(1));
    assert_eq!(rows[3][3], Value::I64(3));
    // Aggregates that would need to *remove* from the frame still have no window version.
    // The rejection happens in the window operator, so it surfaces while stepping rather
    // than at prepare time.
    assert!(try_run(&mut sess, "SELECT median(x) OVER (ORDER BY x) FROM t").is_err());
}
