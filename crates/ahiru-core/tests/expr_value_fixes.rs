//! End-to-end regressions for a batch of scalar-expression value fixes: narrow
//! integer overflow, `//` on floating point, integer-to-temporal conversions,
//! whitespace around cast text, `FLOAT` parsing and list spelling, and the
//! calendar parts of years before 1.
//!
//! Expected values come from `duckdb -csv -c "SELECT ..."` (v1.4.4) unless a
//! comment says this engine deliberately differs.

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

fn try_run(session: &mut Session, sql: &str) -> Result<Vec<Vec<Value>>, ahiru_core::error::Error> {
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
            _ => panic!("{sql}: unexpected step"),
        }
    }
    Ok(rows)
}

fn run(session: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    try_run(session, sql).unwrap_or_else(|e| panic!("{sql}: {e:?}"))
}

/// The value as text, via `CAST(... AS VARCHAR)` so the engine's own rendering
/// is what is asserted on.
fn text(session: &mut Session, expr: &str) -> String {
    let sql = format!("SELECT CAST({expr} AS VARCHAR) AS x FROM t LIMIT 1");
    match &run(session, &sql)[0][0] {
        Value::Bytes(b) => String::from_utf8(b.clone()).unwrap(),
        Value::Null => "NULL".into(),
        other => panic!("{expr}: not text: {other:?}"),
    }
}

fn fails(session: &mut Session, expr: &str) -> bool {
    try_run(session, &format!("SELECT {expr} AS x FROM t LIMIT 1")).is_err()
}

#[test]
fn narrow_integer_overflow_wraps_inside_the_declared_type() {
    let mut s = session_with_basic();
    // DuckDB raises an out-of-range error for each of these; this engine wraps, its
    // documented rule for integer overflow -- but now at the declared width. They used
    // to produce 128 / 32768 while still calling themselves TINYINT / SMALLINT.
    assert_eq!(text(&mut s, "CAST(127 AS TINYINT) + CAST(1 AS TINYINT)"), "-128");
    assert_eq!(text(&mut s, "CAST(32767 AS SMALLINT) + CAST(1 AS SMALLINT)"), "-32768");
    assert_eq!(text(&mut s, "CAST(-128 AS TINYINT) * CAST(-1 AS TINYINT)"), "-128");
    assert_eq!(text(&mut s, "-CAST(-128 AS TINYINT)"), "-128");
    assert_eq!(text(&mut s, "typeof(CAST(127 AS TINYINT) + CAST(1 AS TINYINT))"), "TINYINT");
    // `MIN // -1` is NULL at every width, like `i32::MIN // -1`.
    assert_eq!(text(&mut s, "CAST(-128 AS TINYINT) // CAST(-1 AS TINYINT)"), "NULL");
    assert_eq!(text(&mut s, "CAST(-32768 AS SMALLINT) % CAST(-1 AS SMALLINT)"), "NULL");
    // In range, nothing changes; and `abs` widens to BIGINT, so 128 is a valid answer.
    assert_eq!(text(&mut s, "CAST(100 AS TINYINT) + CAST(27 AS TINYINT)"), "127");
    assert_eq!(text(&mut s, "abs(CAST(-128 AS TINYINT))"), "128");
}

#[cfg(feature = "export-parquet")]
#[test]
fn a_wrapped_tinyint_survives_a_parquet_round_trip() {
    use ahiru_core::write::{export_all, parquet::ParquetSink};
    let mut s = session_with_basic();
    // A value outside the declared type used to be written as-is, producing a file this
    // engine's own reader then rejected.
    let bytes = export_all(
        &mut s,
        "SELECT CAST(127 AS TINYINT) + CAST(1 AS TINYINT) AS x FROM t LIMIT 1",
        &[],
        &mut ParquetSink::new(),
    )
    .unwrap();
    let mut back = Session::new();
    back.register_bytes_as("p", bytes, ahiru_core::FormatKind::Parquet).unwrap();
    assert_eq!(run(&mut back, "SELECT x FROM p"), vec![vec![Value::I32(-128)]]);
}

#[test]
fn int_div_by_zero_is_null_for_floats() {
    let mut s = session_with_basic();
    assert_eq!(text(&mut s, "5 // CAST(0.0 AS DOUBLE)"), "NULL");
    assert_eq!(text(&mut s, "CAST(1.5 AS FLOAT) // 0"), "NULL");
    assert_eq!(text(&mut s, "CAST(7.5 AS DECIMAL(3,1)) // 0"), "NULL");
    // `//` does not floor a float (DuckDB), and `/` keeps IEEE semantics.
    assert_eq!(text(&mut s, "CAST(7.0 AS DOUBLE) // 2"), "3.5");
    assert_eq!(text(&mut s, "CAST(5.0 AS DOUBLE) / 0"), "inf");
    assert_eq!(text(&mut s, "7 // 2"), "3");
}

#[test]
fn numbers_are_not_timestamps() {
    let mut s = session_with_basic();
    // DuckDB: binder error / unsupported cast for each.
    assert!(fails(&mut s, "year(1500)"));
    assert!(fails(&mut s, "dayname(3)"));
    assert!(fails(&mut s, "CAST(5 AS DATE)"));
    assert!(fails(&mut s, "CAST(1500 AS TIMESTAMP)"));
    // The explicit conversions.
    assert_eq!(text(&mut s, "epoch_ms(1500)"), "1970-01-01 00:00:01.5");
    assert_eq!(text(&mut s, "epoch_ms(-1500)"), "1969-12-31 23:59:58.5");
    assert_eq!(text(&mut s, "make_timestamp(1500)"), "1970-01-01 00:00:00.0015");
    assert_eq!(text(&mut s, "to_timestamp(1.5)"), "1970-01-01 00:00:01.5+00");
    assert_eq!(text(&mut s, "typeof(to_timestamp(1500))"), "TIMESTAMPTZ");
    // A TIMESTAMP still goes the other way, and a string still parses.
    assert_eq!(text(&mut s, "epoch_ms(TIMESTAMP '1970-01-01 00:00:01.5')"), "1500");
    assert_eq!(text(&mut s, "to_timestamp('2024-05-01 10:00:00')"), "2024-05-01 10:00:00");
    // `DATE - DATE` (which widens DATEs to BIGINT internally) is unaffected.
    assert_eq!(text(&mut s, "DATE '2024-01-10' - DATE '2024-01-01'"), "9");
}

#[test]
fn text_casts_ignore_surrounding_ascii_whitespace() {
    let mut s = session_with_basic();
    assert_eq!(text(&mut s, "CAST('5' || chr(13) AS BIGINT)"), "5");
    assert_eq!(text(&mut s, "CAST('5.5' || chr(10) AS DOUBLE)"), "5.5");
    assert_eq!(text(&mut s, "CAST('5' || chr(10) AS DECIMAL(3,1))"), "5.0");
    assert_eq!(text(&mut s, "CAST('2024-01-01' || chr(13) AS DATE)"), "2024-01-01");
    assert_eq!(text(&mut s, "CAST(chr(11) || '5' || chr(12) AS INTEGER)"), "5");
    assert_eq!(text(&mut s, "CAST(chr(10) || '12:00:00' || chr(13) AS TIME)"), "12:00:00");
}

#[test]
fn float_values_parse_and_print_at_f32_precision() {
    let mut s = session_with_basic();
    assert_eq!(text(&mut s, "CAST('1.00000005960464477539062500001' AS FLOAT)"), "1.0000001");
    assert_eq!(text(&mut s, "list_value(CAST(0.1 AS FLOAT))"), "[0.1]");
    assert_eq!(text(&mut s, "[CAST(0.1 AS FLOAT)]"), "[0.1]");
    assert_eq!(text(&mut s, "[CAST(1e30 AS FLOAT), CAST(0.1 AS DOUBLE)]"), "[1e+30,0.1]");
    let rows =
        run(&mut s, "SELECT CAST(list(CAST(i / 10.0 AS FLOAT)) AS VARCHAR) FROM range(3) t(i)");
    assert_eq!(rows, vec![vec![Value::Bytes(b"[0.0, 0.1, 0.2]".to_vec())]]);
}

#[test]
fn doubles_print_their_shortest_round_trip() {
    let mut s = session_with_basic();
    assert_eq!(text(&mut s, "CAST('0.09999999999999999' AS DOUBLE)"), "0.09999999999999999");
    assert_eq!(text(&mut s, "power(10, 2.5)"), "316.22776601683796");
    assert_eq!(text(&mut s, "power(1.1, 1000)"), "2.4699329180060256e+41");
    assert_eq!(text(&mut s, "power(10, -310)"), "1e-310");
}

#[test]
fn calendar_parts_of_years_before_one() {
    let mut s = session_with_basic();
    assert_eq!(text(&mut s, "date_part('century', DATE '-0001-07-27')"), "-1");
    assert_eq!(text(&mut s, "date_part('millennium', DATE '0000-07-27')"), "-1");
    assert_eq!(text(&mut s, "date_part('decade', DATE '-0084-07-27')"), "-8");
    assert_eq!(text(&mut s, "year(date_trunc('decade', DATE '-0084-07-27'))"), "-80");
    assert_eq!(text(&mut s, "date_diff('decade', DATE '-0005-06-01', DATE '0005-06-01')"), "0");
}

#[test]
fn printf_precision() {
    let mut s = session_with_basic();
    assert_eq!(text(&mut s, "printf('%.3s|', 'abcdef')"), "abc|");
    assert_eq!(text(&mut s, "printf('%5.1s|', 'abc')"), "    a|");
    assert_eq!(text(&mut s, "printf('%.0s|', 'abc')"), "|");
    assert_eq!(text(&mut s, "printf('%.3d|', 5)"), "005|");
}
