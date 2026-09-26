//! Regression tests for a batch of DATE/TIME/TIMESTAMP/TIMESTAMPTZ/INTERVAL fixes.
//!
//! Expected values were taken from DuckDB 1.4.4
//! (`duckdb -c "SET TimeZone='UTC'; SELECT ..."`) unless a comment says otherwise. The one
//! deliberate difference that shows up below is that a result out of its type's range is
//! `NULL` here where DuckDB raises.
//!
//! A `SELECT <expr>` with no `FROM` is unsupported (`plan::bind`), so these run against
//! `range(1)`.

use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn run(session: &mut Session, sql: &str) -> ahiru_core::error::Result<Vec<Vec<Value>>> {
    let mut q = match session.prepare(sql, &[])? {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
    };
    let mut rows = Vec::new();
    loop {
        match session.step(&mut q)? {
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

/// The value of `expr` as text, via `CAST(... AS VARCHAR)`, so the engine's own rendering is
/// what is asserted on. `NULL` for a NULL.
fn text(expr: &str) -> String {
    let sql = format!("SELECT CAST({expr} AS VARCHAR) FROM range(1)");
    let rows = run(&mut Session::new(), &sql).unwrap_or_else(|e| panic!("{expr}: {e:?}"));
    match &rows[0][0] {
        Value::Bytes(b) => String::from_utf8(b.clone()).unwrap(),
        Value::Null => "NULL".into(),
        other => panic!("{expr}: not text: {other:?}"),
    }
}

/// The logical type name of `expr`.
fn type_of(expr: &str) -> String {
    text(&format!("typeof({expr})"))
}

fn fails(expr: &str) -> bool {
    run(&mut Session::new(), &format!("SELECT {expr} FROM range(1)")).is_err()
}

fn check(cases: &[(&str, &str)]) {
    for (expr, want) in cases {
        assert_eq!(text(expr), *want, "{expr}");
    }
}

#[test]
fn timestamptz_plus_minus_interval_stays_timestamptz() {
    check(&[
        ("TIMESTAMPTZ '2024-01-01 10:00:00' - INTERVAL 1 DAY", "2023-12-31 10:00:00+00"),
        ("INTERVAL '1 hour' + TIMESTAMPTZ '2024-01-01 10:00:00'", "2024-01-01 11:00:00+00"),
        ("TIMESTAMPTZ '2024-01-31 10:00:00' + INTERVAL 1 MONTH", "2024-02-29 10:00:00+00"),
    ]);
    assert_eq!(type_of("TIMESTAMPTZ '2024-01-01' + INTERVAL 1 DAY"), "TIMESTAMPTZ");
    assert_eq!(type_of("DATE '2024-01-01' + INTERVAL 1 DAY"), "TIMESTAMP");
    // `now()` is a TIMESTAMPTZ too.
    assert_eq!(type_of("now() - INTERVAL 1 DAY"), "TIMESTAMPTZ");
    assert_eq!(type_of("CURRENT_TIMESTAMP - INTERVAL '1 hour'"), "TIMESTAMPTZ");
}

#[test]
fn timestamp_minus_timestamp_is_an_interval() {
    check(&[
        ("TIMESTAMP '2024-01-02' - TIMESTAMP '2024-01-01 10:00:00'", "14:00:00"),
        ("TIMESTAMP '2024-03-02' - TIMESTAMP '2024-01-01 10:00:00'", "60 days 14:00:00"),
        // Days and microseconds both truncate toward zero.
        ("TIMESTAMP '2024-01-01' - TIMESTAMP '2024-03-02 10:00:00'", "-61 days -10:00:00"),
        ("DATE '2024-01-02' - TIMESTAMP '2024-01-01 01:00'", "23:00:00"),
        ("TIMESTAMP '2024-01-02' - DATE '2024-01-01'", "1 day"),
        ("TIMESTAMPTZ '2024-01-02' - TIMESTAMPTZ '2024-01-01 10:00:00'", "14:00:00"),
        ("TIMESTAMPTZ '2024-01-02' - DATE '2024-01-01'", "1 day"),
        // DuckDB raises "Timestamp difference is out of bounds".
        ("TIMESTAMP '290000-01-01' - TIMESTAMP '-290000-01-01'", "NULL"),
        // DATE - DATE stays a day count.
        ("DATE '2024-01-02' - DATE '2024-01-01'", "1"),
    ]);
    assert_eq!(type_of("TIMESTAMP '2024-01-02' - TIMESTAMP '2024-01-01'"), "INTERVAL");
}

#[test]
fn time_casts_parts_and_arithmetic() {
    check(&[
        ("CAST(TIMESTAMP '2024-01-01 10:20:30.5' AS TIME)", "10:20:30.5"),
        ("CAST(TIMESTAMP '1960-01-01 10:20:30.5' AS TIME)", "10:20:30.5"),
        ("hour(TIME '10:20:30.5')", "10"),
        ("minute(TIME '10:20:30.5')", "20"),
        ("second(TIME '10:20:30.5')", "30"),
        ("millisecond(TIME '10:20:30.5')", "30500"),
        ("extract(hour FROM TIME '10:20:30')", "10"),
        ("date_part('minute', TIME '10:20:30')", "20"),
        ("date_part('epoch', TIME '10:20:30')", "37230"),
        ("date_diff('minute', TIME '10:00', TIME '11:30')", "90"),
        ("TIME '10:00' + INTERVAL 1 HOUR", "11:00:00"),
        ("INTERVAL 1 HOUR + TIME '10:00'", "11:00:00"),
        // The clock wraps; the day and month fields are ignored.
        ("TIME '23:00' + INTERVAL 2 HOUR", "01:00:00"),
        ("TIME '01:00' - INTERVAL 2 HOUR", "23:00:00"),
        ("TIME '10:00' + INTERVAL '1 day 1 hour'", "11:00:00"),
        ("TIME '10:00' + INTERVAL '1 month'", "10:00:00"),
        ("DATE '2024-01-01' + TIME '10:00'", "2024-01-01 10:00:00"),
        ("TIME '10:00' + DATE '2024-01-01'", "2024-01-01 10:00:00"),
        // A zone offset on a TIME is accepted and ignored.
        ("'12:00:00+05'::TIME", "12:00:00"),
        ("'12:00:00-05:30'::TIME", "12:00:00"),
        ("'12:00:00Z'::TIME", "12:00:00"),
        ("TIME '12:00:00+05'", "12:00:00"),
    ]);
    assert_eq!(type_of("DATE '2024-01-01' + TIME '10:00'"), "TIMESTAMP");
    // A TIME has no date parts (DuckDB: binder/not-implemented errors).
    assert!(fails("year(TIME '10:00')"));
    assert!(fails("date_part('day', TIME '10:00')"));
    assert!(fails("TIME '10:00' - TIME '09:00'"));
}

#[test]
fn date_part_of_an_interval() {
    check(&[
        ("date_part('day', INTERVAL '5 days')", "5"),
        ("date_part('epoch', INTERVAL '1 day')", "86400"),
        // A year is 365.25 days and a month 30 days here.
        ("date_part('epoch', INTERVAL '13 months')", "34149600"),
        ("date_part('epoch', INTERVAL '-13 month -1 day')", "-34236000"),
        ("epoch(INTERVAL '1 day')", "86400"),
        ("year(INTERVAL '30 months')", "2"),
        ("month(INTERVAL '30 months')", "6"),
        ("date_part('year', INTERVAL '-30 months')", "-2"),
        ("date_part('quarter', INTERVAL '7 months')", "3"),
        ("date_part('quarter', INTERVAL '-7 months')", "-1"),
        ("date_part('decade', INTERVAL '250 months')", "2"),
        ("date_part('century', INTERVAL '2500 years')", "25"),
        ("date_part('millennium', INTERVAL '2500 years')", "2"),
        // The day field is not folded into the hours.
        ("date_part('hour', INTERVAL '1 day 25 hours 61 minutes')", "26"),
        ("date_part('minute', INTERVAL '25 hours 61 minutes')", "1"),
        ("date_part('second', INTERVAL '61.5 seconds')", "1"),
        ("date_part('millisecond', INTERVAL '61.5 seconds')", "1500"),
        ("date_part('microsecond', INTERVAL '61.5 seconds')", "1500000"),
    ]);
    assert!(fails("date_part('dow', INTERVAL '5 days')"));
    assert!(fails("dayofweek(INTERVAL '5 days')"));
}

#[test]
fn interval_times_and_divided_by_a_number() {
    check(&[
        ("INTERVAL '1 day' * 1.5", "1 day 12:00:00"),
        ("1.5 * INTERVAL '1 day'", "1 day 12:00:00"),
        ("INTERVAL '1 month' * 1.5", "1 month 15 days"),
        ("INTERVAL '1 month' * 1.3", "1 month 9 days"),
        ("INTERVAL '1 month' * 0.01", "07:12:00"),
        ("INTERVAL '1 day' / 2", "12:00:00"),
        ("INTERVAL '1 month' / 2", "15 days"),
        ("INTERVAL '1 day' / 1.5", "16:00:00"),
        ("INTERVAL '1 day' / 0", "NULL"),
        // PostgreSQL's cascade rounds each fraction to a microsecond, hence the odd digits.
        ("INTERVAL '1 month 1 day 1 hour' / 7", "4 days 10:25:42.832457"),
        ("INTERVAL '1 year' / 7", "1 month 21 days 10:17:08.5344"),
        // Rounded half to even.
        ("INTERVAL '3 microsecond' / 2", "00:00:00.000002"),
        ("INTERVAL '1 microsecond' / 2", "00:00:00"),
        ("INTERVAL '1 day' * -2", "-2 days"),
    ]);
}

#[test]
fn interval_overflow_is_null_not_wrapped() {
    // DuckDB raises for each of these; the result used to wrap (`-1294967296 days`).
    check(&[
        ("INTERVAL '1 day' * 3000000000", "NULL"),
        ("INTERVAL '1 microsecond' * 3000000000", "NULL"),
        ("INTERVAL '2000000000 days' + INTERVAL '2000000000 days'", "NULL"),
        ("INTERVAL '1 day' * 1e300", "NULL"),
    ]);
}

#[test]
fn date_trunc_and_date_diff_take_the_day_level_parts() {
    check(&[
        ("date_diff('dow', DATE '2024-01-01', DATE '2024-01-10')", "9"),
        ("date_diff('weekday', DATE '2024-01-01', DATE '2024-01-10')", "9"),
        ("date_diff('isodow', DATE '2024-01-01', DATE '2024-01-10')", "9"),
        ("date_diff('doy', DATE '2024-01-01', DATE '2025-01-10')", "375"),
        ("date_diff('dow', TIMESTAMP '2024-01-01 23:00', TIMESTAMP '2024-01-02 01:00')", "1"),
        ("date_trunc('dow', TIMESTAMP '2024-01-03 10:00')", "2024-01-03 00:00:00"),
        ("date_trunc('doy', TIMESTAMP '2024-01-03 10:00')", "2024-01-03 00:00:00"),
        ("date_trunc('isodow', DATE '2024-01-03')", "2024-01-03 00:00:00"),
        ("date_trunc('epoch', TIMESTAMP '2024-01-03 10:00:01.5')", "2024-01-03 10:00:01"),
        // A TIMESTAMPTZ truncates to a TIMESTAMPTZ.
        ("date_trunc('day', TIMESTAMPTZ '2024-01-01 10:00')", "2024-01-01 00:00:00+00"),
    ]);
    assert_eq!(type_of("date_trunc('day', TIMESTAMPTZ '2024-01-01 10:00')"), "TIMESTAMPTZ");
}

#[test]
fn date_functions_work_past_the_timestamp_range() {
    check(&[
        ("year(DATE '300000-01-01')", "300000"),
        ("dayname(DATE '300000-01-01')", "Saturday"),
        ("monthname(DATE '300000-03-01')", "March"),
        ("last_day(DATE '300000-02-01')", "300000-02-29"),
        ("week(DATE '300000-01-01')", "52"),
        ("isoyear(DATE '-300000-01-01')", "-300001"),
        ("date_part('epoch', DATE '300000-01-01')", "9404918380800"),
        ("date_diff('day', DATE '300000-01-01', DATE '300001-01-01')", "366"),
        ("date_diff('hour', DATE '300000-01-01', DATE '300001-01-01')", "8784"),
        ("date_diff('day', DATE '-5000000-01-01', DATE '5000000-01-01')", "3652425000"),
        ("strftime(DATE '300000-01-05', '%Y-%m-%d')", "300000-01-05"),
        // DuckDB's DATE range ends at 5881580-07-10.
        ("DATE '5881580-07-10'", "5881580-07-10"),
        ("TRY_CAST('5881580-07-11' AS DATE)", "NULL"),
        ("year(DATE '-5000000-01-01')", "-5000000"),
    ]);
}

#[test]
fn strftime_takes_either_argument_order_and_prints_bc_years_unpadded() {
    check(&[
        ("strftime('%Y', DATE '2024-01-05')", "2024"),
        ("strftime('%Y-%m-%d %H', TIMESTAMP '2024-01-05 03:04:05')", "2024-01-05 03"),
        ("strftime(DATE '2024-01-05', '%Y/%m')", "2024/01"),
        ("strftime(TIMESTAMP '-0044-01-05 03:04:05', '%Y')", "-44"),
        ("strftime(TIMESTAMP '-0001-01-05 03:04:05', '%Y|%m|%d')", "-1|01|05"),
        ("strftime(TIMESTAMP '0044-01-05 03:04:05', '%Y')", "0044"),
    ]);
}

#[test]
fn text_to_timestamptz_accepts_what_timestamp_accepts() {
    check(&[
        ("'2024-01-01 10:00:00+0530'::TIMESTAMPTZ", "2024-01-01 04:30:00+00"),
        ("'2024-01-01 10:00:00+05:30:15'::TIMESTAMPTZ", "2024-01-01 04:29:45+00"),
        ("'2024-01-01 10:00:00 UTC'::TIMESTAMPTZ", "2024-01-01 10:00:00+00"),
        ("'2024-01-01 10:00:00 utc'::TIMESTAMPTZ", "2024-01-01 10:00:00+00"),
        ("'2024-01-01  10:00:00'::TIMESTAMPTZ", "2024-01-01 10:00:00+00"),
        ("TIMESTAMPTZ '2024-01-01 10:00:00-0530'", "2024-01-01 15:30:00+00"),
        // Two-digit fields are not range-checked, as in DuckDB.
        ("'2024-01-01 10:00:00+99'::TIMESTAMPTZ", "2023-12-28 07:00:00+00"),
        ("'2024-01-01 10:00:00+05:60'::TIMESTAMPTZ", "2024-01-01 04:00:00+00"),
        // A single-digit hour and a lowercase `z` are rejected (DuckDB: conversion error).
        ("TRY_CAST('2024-01-01 10:00:00+5' AS TIMESTAMPTZ)", "NULL"),
        ("TRY_CAST('2024-01-01 10:00:00z' AS TIMESTAMPTZ)", "NULL"),
        // A plain TIMESTAMP reads the same suffixes and ignores them.
        ("'2024-01-01 10:00:00+05:30:15'::TIMESTAMP", "2024-01-01 10:00:00"),
    ]);
}

#[test]
fn text_dates_take_slash_space_and_backslash_separators() {
    check(&[
        ("TRY_CAST('2024/01/05' AS DATE)", "2024-01-05"),
        ("TRY_CAST('2024/1/5' AS DATE)", "2024-01-05"),
        ("TRY_CAST('2024 01 05' AS DATE)", "2024-01-05"),
        ("TRY_CAST('2024 1 5' AS DATE)", "2024-01-05"),
        (r"TRY_CAST('2024\01\05' AS DATE)", "2024-01-05"),
        ("TRY_CAST('2024/01/05 10:00:00' AS TIMESTAMP)", "2024-01-05 10:00:00"),
        ("TRY_CAST('2024/01/05T10:00' AS TIMESTAMP)", "2024-01-05 10:00:00"),
        ("TRY_CAST('2024 01 05 10:00' AS TIMESTAMP)", "2024-01-05 10:00:00"),
        ("TRY_CAST('2024/01/05' AS TIMESTAMPTZ)", "2024-01-05 00:00:00+00"),
        // Both separators must be the same one.
        ("TRY_CAST('2024/01-05' AS DATE)", "NULL"),
        ("TRY_CAST('2024.01.05' AS DATE)", "NULL"),
    ]);
}

#[test]
fn interval_text_follows_duckdb() {
    check(&[
        // A fraction cascades one field down only, truncated there.
        ("CAST('1.25 months' AS INTERVAL)", "1 month 7 days"),
        ("CAST('-1.25 months' AS INTERVAL)", "-1 month -7 days"),
        ("CAST('1.3 years' AS INTERVAL)", "1 year 3 months"),
        ("CAST('1.5 quarters' AS INTERVAL)", "4 months 15 days"),
        ("CAST('1.1 weeks' AS INTERVAL)", "7 days 16:48:00"),
        ("CAST('1.123456789 hours' AS INTERVAL)", "01:07:24.4416"),
        ("CAST('2.5 microseconds' AS INTERVAL)", "00:00:00.000003"),
        // Abbreviated and irregular units, with or without a space.
        ("CAST('1h' AS INTERVAL)", "01:00:00"),
        ("CAST('1d' AS INTERVAL)", "1 day"),
        ("CAST('1 day1 hour' AS INTERVAL)", "1 day 01:00:00"),
        ("CAST('2 mon' AS INTERVAL)", "2 months"),
        ("CAST('1 y' AS INTERVAL)", "1 year"),
        ("CAST('1 d 1 m' AS INTERVAL)", "1 day 00:01:00"),
        ("CAST('0.5 ms' AS INTERVAL)", "00:00:00.0005"),
        ("CAST('1 millennium' AS INTERVAL)", "1000 years"),
        ("CAST('2 centuries' AS INTERVAL)", "200 years"),
        ("CAST('1.5 decade' AS INTERVAL)", "15 years"),
        ("CAST('1 quarter' AS INTERVAL)", "3 months"),
        // `ago` negates, `@` is ignored, a bare decimal is seconds.
        ("CAST('1 day 2 hours ago' AS INTERVAL)", "-1 day -02:00:00"),
        ("CAST('-1 day -2 hours ago' AS INTERVAL)", "1 day 02:00:00"),
        ("CAST('@ 1 day' AS INTERVAL)", "1 day"),
        ("CAST('1.5' AS INTERVAL)", "00:00:01.5"),
        // A `-` before the time component negates the microseconds accumulated so far.
        ("CAST('1 hour -01:00' AS INTERVAL)", "-02:00:00"),
        ("INTERVAL 3 WEEKS", "21 days"),
        // Rejected, as in DuckDB.
        ("TRY_CAST('3 wk' AS INTERVAL)", "NULL"),
        ("TRY_CAST('1 dow' AS INTERVAL)", "NULL"),
        ("TRY_CAST('+1 day' AS INTERVAL)", "NULL"),
        ("TRY_CAST('.5 day' AS INTERVAL)", "NULL"),
        ("TRY_CAST('01:30:61' AS INTERVAL)", "NULL"),
        ("TRY_CAST('01:30 1 day' AS INTERVAL)", "NULL"),
        ("TRY_CAST('1 day agox' AS INTERVAL)", "NULL"),
    ]);
}

#[test]
fn interval_comparison_normalizes_like_duckdb() {
    // DuckDB normalizes with truncating carries and compares (months, days, micros)
    // lexicographically, so mixed-sign intervals do not compare as flattened spans.
    check(&[
        ("INTERVAL '2 months -45 days' < INTERVAL '16 days'", "false"),
        ("INTERVAL '-1 day 1 hour' = INTERVAL '-23 hours'", "false"),
        ("INTERVAL '-1 day 1 hour' < INTERVAL '-23 hours'", "true"),
        ("INTERVAL '-22 hours -60 minutes' = INTERVAL '-23 hours'", "true"),
        ("INTERVAL '1 day' = INTERVAL '24 hours'", "true"),
        ("INTERVAL '1 month' = INTERVAL '30 days'", "true"),
        ("INTERVAL '1 month' > INTERVAL '29 days'", "true"),
        ("INTERVAL '-1 month 40 days' > INTERVAL '5 days'", "true"),
    ]);
    let mut s = Session::new();
    let rows = run(
        &mut s,
        "WITH v AS (SELECT INTERVAL '2 months -45 days' AS x FROM range(1) \
         UNION ALL SELECT INTERVAL '16 days' FROM range(1) \
         UNION ALL SELECT INTERVAL '-1 day 1 hour' FROM range(1) \
         UNION ALL SELECT INTERVAL '-23 hours' FROM range(1) \
         UNION ALL SELECT INTERVAL '-22 hours -60 minutes' FROM range(1)) \
         SELECT CAST(x AS VARCHAR), count(*) OVER (PARTITION BY x) FROM v ORDER BY x",
    )
    .unwrap();
    let got: Vec<(String, Value)> = rows
        .into_iter()
        .map(|r| match &r[0] {
            Value::Bytes(b) => (String::from_utf8(b.clone()).unwrap(), r[1].clone()),
            other => panic!("{other:?}"),
        })
        .collect();
    let want = [
        ("-1 day 01:00:00", 1),
        ("-23:00:00", 2),
        ("-23:00:00", 2),
        ("16 days", 1),
        ("2 months -45 days", 1),
    ];
    assert_eq!(got.len(), want.len());
    for ((t, n), (wt, wn)) in got.iter().zip(want) {
        assert_eq!(t, wt);
        assert_eq!(n, &Value::I64(wn));
    }
}

#[test]
fn boolean_mixes_with_numbers() {
    check(&[
        ("true = 1", "true"),
        ("true = 2", "false"),
        ("true IN (1, 2)", "true"),
        ("true < 2", "true"),
        ("true = 1.5", "false"),
        ("false = 0::UBIGINT", "true"),
        ("coalesce(NULL::BOOLEAN, 0)", "0"),
        ("CASE WHEN 1 = 1 THEN true ELSE 1 END", "1"),
        ("greatest(true, 5)", "5"),
    ]);
    assert_eq!(type_of("coalesce(true, 0)"), "INTEGER");
    assert_eq!(type_of("coalesce(true, 2::TINYINT)"), "TINYINT");
    // Arithmetic on a BOOLEAN is still an error, as in DuckDB.
    assert!(fails("true + 1"));
    assert!(fails("true * 2"));
}

#[test]
fn nullif_keeps_the_first_argument_type() {
    check(&[
        ("nullif(DATE '2024-01-01', TIMESTAMP '2024-01-01')", "NULL"),
        ("nullif(DATE '2024-01-01', TIMESTAMP '2024-01-02')", "2024-01-01"),
        ("nullif(1, 1.0)", "NULL"),
        ("nullif(2, NULL)", "2"),
    ]);
    assert_eq!(type_of("nullif(DATE '2024-01-01', TIMESTAMP '2024-01-02')"), "DATE");
    assert_eq!(type_of("nullif(1::INTEGER, 2::BIGINT)"), "INTEGER");
}
