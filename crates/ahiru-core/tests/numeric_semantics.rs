//! Regression tests for DECIMAL / FLOAT / numeric-literal semantics.
//!
//! Expected values were taken from `duckdb -csv -c "SELECT ..."` (DuckDB 1.4.4) unless
//! a comment says otherwise. A `SELECT <expr>` with no `FROM` is unsupported, so the
//! expressions run against `tests/data/basic.parquet` with `LIMIT 1`.

use ahiru_core::error::Code;
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

fn try_run(session: &mut Session, sql: &str) -> Result<Vec<Vec<Value>>, Code> {
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

fn run(session: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    try_run(session, sql).unwrap_or_else(|e| panic!("{sql}: {e:?}"))
}

/// One expression's value as the engine spells it (`CAST(... AS VARCHAR)`).
fn text(expr: &str) -> String {
    let sql = format!("SELECT CAST(({expr}) AS VARCHAR) FROM t LIMIT 1");
    match &run(&mut session_with("basic.parquet"), &sql)[0][0] {
        Value::Bytes(b) => String::from_utf8(b.clone()).unwrap(),
        Value::Null => "NULL".into(),
        v => panic!("{sql}: expected VARCHAR, got {v:?}"),
    }
}

/// Several expressions at once, each spelled as text, joined with `|`.
fn texts(exprs: &[&str]) -> String {
    exprs.iter().map(|e| text(e)).collect::<Vec<_>>().join("|")
}

fn error_of(expr: &str) -> Code {
    let sql = format!("SELECT {expr} FROM t LIMIT 1");
    match try_run(&mut session_with("basic.parquet"), &sql) {
        Err(c) => c,
        Ok(rows) => panic!("{sql}: expected an error, got {rows:?}"),
    }
}

// --- DECIMAL arithmetic overflow ---------------------------------------------

#[test]
fn decimal_38_overflow_is_an_error_not_a_wrapped_value() {
    // duckdb: "Overflow in multiplication of DECIMAL(38)". The raw product exceeds 2^127
    // and used to wrap to -168.67...
    let d = "CAST('13.1' AS DECIMAL(38,18))";
    assert_eq!(error_of(&format!("{d} * {d}")), Code::ValueOutOfRange);
    // Past 38 digits but still inside i128: used to print a 39-digit DECIMAL(38).
    let max = "CAST('99999999999999999999999999999999999.999' AS DECIMAL(38,3))";
    assert_eq!(error_of(&format!("{max} + CAST('1' AS DECIMAL(38,3))")), Code::ValueOutOfRange);
    assert_eq!(error_of(&format!("-{max} - CAST('1' AS DECIMAL(38,3))")), Code::ValueOutOfRange);
    let big = "CAST('60000000000000000000000000000000000000' AS DECIMAL(38,0))";
    assert_eq!(error_of(&format!("{big} * 2")), Code::ValueOutOfRange);
    let sql = format!("SELECT sum(d * 2) FROM (SELECT {big} AS d FROM t)");
    assert_eq!(try_run(&mut session_with("basic.parquet"), &sql), Err(Code::ValueOutOfRange));
    // In-range results are unchanged. duckdb: 1.21000..., 99999999999999999980000000000000000001
    assert_eq!(text(&format!("{d} * 0.1")), "1.3100000000000000000");
    assert_eq!(
        text("CAST('1.1' AS DECIMAL(38,18)) * CAST('1.1' AS DECIMAL(38,18))"),
        "1.210000000000000000000000000000000000"
    );
    assert_eq!(
        text("CAST('9999999999999999999' AS DECIMAL(19,0)) * CAST('9999999999999999999' AS DECIMAL(19,0))"),
        "99999999999999999980000000000000000001"
    );
    // A NULL operand is not an overflow.
    assert_eq!(text(&format!("{d} * CAST(NULL AS DECIMAL(38,18))")), "NULL");
}

// --- Merging DECIMALs whose widths add up past 38 digits ---------------------

#[test]
fn merged_decimals_give_up_scale_not_integer_digits() {
    // duckdb: coalesce/greatest/CASE/UNION of DECIMAL(38,0) and DECIMAL(38,18) is
    // DECIMAL(38,0). Keeping the scale left room for only 20 integer digits, so the
    // 30-digit value came back NULL and coalesce fell through to the second argument.
    let big = "CAST('123456789012345678901234567890' AS DECIMAL(38,0))";
    let frac = "CAST('1.5' AS DECIMAL(38,18))";
    assert_eq!(text(&format!("coalesce({big}, {frac})")), "123456789012345678901234567890");
    assert_eq!(text(&format!("greatest({big}, {frac})")), "123456789012345678901234567890");
    assert_eq!(text(&format!("least({big}, {frac})")), "2");
    assert_eq!(
        text(&format!("CASE WHEN id = 0 THEN {big} ELSE {frac} END")),
        "123456789012345678901234567890"
    );
    let mut s = session_with("basic.parquet");
    let rows = run(
        &mut s,
        &format!(
            "SELECT CAST(x AS VARCHAR) FROM (SELECT {big} AS x FROM t WHERE id = 0 \
             UNION ALL SELECT {frac} FROM t WHERE id = 0) ORDER BY 1"
        ),
    );
    assert_eq!(
        rows,
        vec![
            vec![Value::Bytes(b"123456789012345678901234567890".to_vec())],
            vec![Value::Bytes(b"2".to_vec())]
        ]
    );
    // Comparisons keep the larger scale, so a fraction is never rounded away.
    // duckdb: false, true
    assert_eq!(
        texts(&[
            "1.5::DECIMAL(38,18) = 2::DECIMAL(38,0)",
            "1.5::DECIMAL(38,18) < 2::DECIMAL(38,0)"
        ]),
        "false|true"
    );
}

// --- FLOAT against DECIMAL / integer constants --------------------------------

#[test]
fn float_compares_with_decimal_and_integer_literals_in_float() {
    // duckdb: true, false, true, false. `1.1` is a DECIMAL literal and FLOAT with a
    // DECIMAL stays FLOAT, so the literal rounds to the column's f32; `1.1::DOUBLE`
    // widens the FLOAT instead and does not match.
    let f = "1.1::FLOAT";
    assert_eq!(
        texts(&[
            &format!("{f} = 1.1"),
            &format!("{f} = 1.1::DOUBLE"),
            &format!("{f} IN (0.1, 1.1)"),
            &format!("{f} > 1.1"),
        ]),
        "true|false|true|false"
    );
    // duckdb: FLOAT, FLOAT, FLOAT, DOUBLE, and 2.2 (an f32 sum, printed as one).
    assert_eq!(
        texts(&[
            &format!("typeof({f} + 1.1)"),
            &format!("typeof({f} + 1)"),
            &format!("typeof({f} * 1::HUGEINT)"),
            &format!("typeof({f} + 1.1::DOUBLE)"),
            &format!("{f} + 1.1"),
        ]),
        "FLOAT|FLOAT|FLOAT|DOUBLE|2.2"
    );
}

#[test]
fn float_statistics_are_pruned_with_the_literal_rounded_to_float() {
    // `float_stats.parquet`: four RowGroups of 2048 rows holding only 1.1, 0.1, 3.3 and
    // 16777216 (as FLOAT). A pruner that compared the literal as a DOUBLE against the
    // f32 statistics would skip the RowGroup that matches. Counts are duckdb's.
    let count = |pred: &str| match &run(
        &mut session_with("float_stats.parquet"),
        &format!("SELECT count(*) FROM t WHERE {pred}"),
    )[0][0]
    {
        Value::I64(n) => *n,
        v => panic!("{pred}: {v:?}"),
    };
    assert_eq!(count("f = 1.1"), 2048);
    assert_eq!(count("f IN (0.1, 3.3)"), 4096);
    assert_eq!(count("f > 1.1"), 4096);
    assert_eq!(count("f <= 0.1"), 2048);
    assert_eq!(count("f BETWEEN 1.1 AND 3.3"), 4096);
    // 16777217 has no f32; it rounds to 16777216 as the comparison does.
    assert_eq!(count("f = 16777217"), 2048);
    assert_eq!(count("f >= 16777217"), 2048);
    // A DOUBLE literal still compares in DOUBLE.
    assert_eq!(count("f = 1.1e0"), 0);
    assert_eq!(count("f < 1.1::DOUBLE"), 2048);
}

// --- Decimal literals ---------------------------------------------------------

#[test]
fn decimal_literals_are_exact() {
    // duckdb: every one of these.
    assert_eq!(
        text("123456789012345678901234567.89::DECIMAL(38,2)"),
        "123456789012345678901234567.89"
    );
    assert_eq!(text("CAST(123456789012345678.005 AS DECIMAL(30,3))"), "123456789012345678.005");
    assert_eq!(text("123456789012345678901234567.89::HUGEINT"), "123456789012345678901234568");
    assert_eq!(text("0.1 + 0.2 = 0.3"), "true");
    assert_eq!(text("1.50"), "1.50");
    assert_eq!(text("typeof(1.5e3)"), "DOUBLE");
    // More than 38 digits is a DOUBLE.
    assert_eq!(text("typeof(0.000000000000000000000000000000000000001)"), "DOUBLE");
    // A DECIMAL rounds half away from zero into an integer, a DOUBLE half to even.
    assert_eq!(texts(&["CAST(4.5 AS INTEGER)", "CAST(4.5::DOUBLE AS INTEGER)"]), "5|4");
    assert_eq!(texts(&["CAST(-2.5 AS BIGINT)", "CAST(-2.5::DOUBLE AS BIGINT)"]), "-3|-2");
}

#[cfg(feature = "dml")]
#[test]
fn insert_rounds_a_decimal_literal_into_an_integer_column_half_away() {
    // duckdb: 3, -3, 2
    let mut s = Session::new();
    run(&mut s, "CREATE TABLE r (a INTEGER)");
    run(&mut s, "INSERT INTO r VALUES (2.5), (-2.5), (2.5e0)");
    let rows = run(&mut s, "SELECT a FROM r");
    assert_eq!(rows, vec![vec![Value::I32(3)], vec![Value::I32(-3)], vec![Value::I32(2)]]);
}

// --- DOUBLE -> DECIMAL --------------------------------------------------------

#[test]
fn double_to_decimal_rounds_the_double_not_its_shortest_text() {
    // duckdb: 1.00, 0.28, 0.13, 2.68, 1.01, 5, -0.13. `1.005` is really
    // 1.00499999999999989..., which its shortest text `1.005` hides.
    assert_eq!(
        texts(&[
            "1.005::DOUBLE::DECIMAL(10,2)",
            "0.285::DOUBLE::DECIMAL(10,2)",
            "0.125::DOUBLE::DECIMAL(10,2)",
            "2.675::DOUBLE::DECIMAL(10,2)",
            "1.015::DOUBLE::DECIMAL(10,2)",
            "4.5::DOUBLE::DECIMAL(3,0)",
            "-0.125::DOUBLE::DECIMAL(3,2)",
        ]),
        "1.00|0.28|0.13|2.68|1.01|5|-0.13"
    );
}

// --- Text -> DECIMAL / HUGEINT past 38 digits ---------------------------------

#[test]
fn text_to_decimal_rounds_on_the_digits_past_the_mantissa() {
    // duckdb: the digit past the 38th used to be dropped rather than rounded on.
    assert_eq!(
        text("CAST('8999999999999999999.99999999999999999995' AS DECIMAL(38,19))"),
        "9000000000000000000.0000000000000000000"
    );
    assert_eq!(
        text("CAST('8999999999999999999.99999999999999999995' AS DECIMAL(38,18))"),
        "9000000000000000000.000000000000000000"
    );
    assert_eq!(
        text("'99999999999999999999999999999999999999.5'::HUGEINT"),
        "100000000000000000000000000000000000000"
    );
    assert_eq!(
        text("'-99999999999999999999999999999999999999.5'::HUGEINT"),
        "-100000000000000000000000000000000000000"
    );
    // 10^38 does not fit DECIMAL(38,0): NULL here, a conversion error in DuckDB.
    assert_eq!(
        text("TRY_CAST('99999999999999999999999999999999999999.5' AS DECIMAL(38,0))"),
        "NULL"
    );
    // A tail below one half never rounds up, even when the dropped digits are many.
    assert_eq!(text("'1.4999999999999999999999999999999999999999'::DECIMAL(38,0)"), "1");
    assert_eq!(text("'0.49999999999999999999999999999999999999999'::INTEGER"), "0");
}

// --- Text -> number spellings -------------------------------------------------

#[test]
fn text_to_number_accepts_separators_and_radix_prefixes() {
    // duckdb: 1000, 16, 31, 5, 10.55, 10.550, 11, 15000000000.0, 18446744073709551615, 16
    assert_eq!(
        texts(&[
            "'1_000'::INTEGER",
            "'0x10'::INTEGER",
            "'0X1f'::BIGINT",
            "'0b101'::INTEGER",
            "'1_0.5_5'::DOUBLE",
            "'1_0.5_5'::DECIMAL(10,3)",
            "'1_0.5_5'::INTEGER",
            "'1.5e1_0'::DOUBLE",
            "'0xFFFFFFFFFFFFFFFF'::UBIGINT",
            "'0x1_0'::INTEGER",
        ]),
        "1000|16|31|5|10.55|10.550|11|15000000000.0|18446744073709551615|16"
    );
    // Misplaced separators, a sign or whitespace before a radix prefix, a radix prefix
    // on a non-integer target or HUGEINT, and a value past the target: all NULL.
    for e in [
        "'1__000'::INTEGER",
        "'_1'::INTEGER",
        "'1_'::INTEGER",
        "'1_.5'::DOUBLE",
        "'-0x10'::INTEGER",
        "' 0x10 '::INTEGER",
        "'0x10'::HUGEINT",
        "'0x10'::DECIMAL(5,1)",
        "'0x10'::DOUBLE",
        "'0x8000000000000000'::BIGINT",
        "'0xFF'::TINYINT",
        "'0x'::BIGINT",
        "'0o17'::INTEGER",
    ] {
        assert_eq!(text(e), "NULL", "{e}");
    }
}

#[test]
fn text_past_the_float_range_is_infinite() {
    // duckdb: inf, inf, -inf, inf, 0.0
    assert_eq!(
        texts(&[
            "'1e400'::FLOAT",
            "'1e400'::DOUBLE",
            "'-1e400'::DOUBLE",
            "'1e39'::FLOAT",
            "'1e-400'::DOUBLE"
        ]),
        "inf|inf|-inf|inf|0.0"
    );
    // A DOUBLE value too large for FLOAT is still NULL (DuckDB raises).
    assert_eq!(text("TRY_CAST(1e39 AS FLOAT)"), "NULL");
}

#[test]
fn text_to_boolean_accepts_only_the_boolean_spellings() {
    // duckdb: true, false, true, false, true; everything in the loop is a conversion
    // error there (NULL here, the engine's cast convention).
    assert_eq!(
        texts(&[
            "'1'::BOOLEAN",
            "'0'::BOOLEAN",
            "'TRUE'::BOOLEAN",
            "'no'::BOOLEAN",
            "'Y'::BOOLEAN"
        ]),
        "true|false|true|false|true"
    );
    for e in ["'0.4'", "'2'", "'-1'", "'1.0'", "'00'", "' 1 '", "' true '", "'on'"] {
        assert_eq!(text(&format!("TRY_CAST({e} AS BOOLEAN)")), "NULL", "{e}");
    }
}

#[test]
fn integer_literal_past_hugeint_is_a_double() {
    // duckdb: 1e+42 (DOUBLE); it has a UHUGEINT step first, which this engine does not.
    let big = "1000000000000000000000000000000000000000000";
    assert_eq!(texts(&[big, &format!("typeof({big})")]), "1e+42|DOUBLE");
    assert_eq!(text("-170141183460469231731687303715884105729"), "-1.7014118346046923e+38");
    // The HUGEINT extremes are still HUGEINT.
    assert_eq!(text("typeof(-170141183460469231731687303715884105728)"), "HUGEINT");
}

// --- Unary minus and `::` -----------------------------------------------------

#[test]
fn a_cast_binds_tighter_than_unary_minus() {
    // duckdb: 251 (-(5::UTINYINT) wraps in UTINYINT) and -5; the typing of a bare
    // negative literal is unchanged (BIGINT, BIGINT).
    assert_eq!(
        texts(&[
            "-5::UTINYINT",
            "-5::TINYINT",
            "typeof(-2147483648)",
            "typeof(-9223372036854775808)",
            "-9223372036854775808",
        ]),
        "251|-5|BIGINT|BIGINT|-9223372036854775808"
    );
}

// --- round(DOUBLE, d) ---------------------------------------------------------

#[test]
fn round_double_is_exact_at_large_digit_counts() {
    // duckdb: every value below. The power of ten used to be built by repeated
    // multiplication, off by an ulp past 10^22, and an overflowing multiplier gave 0.
    assert_eq!(
        texts(&[
            "round(1e300::DOUBLE, -300)",
            "round(1.5e-300::DOUBLE, 300)",
            "round(1.23e-310::DOUBLE, 312)",
            "round(5e-324::DOUBLE, 324)",
            "round(1.5e300::DOUBLE, -400)",
            "round(1.7976931348623157e308::DOUBLE, -308)",
            "round('inf'::DOUBLE, -2)",
            "round('inf'::DOUBLE, 2)",
            "round(123.456::DOUBLE, -1)",
            "round(-2.5::DOUBLE)",
        ]),
        "1e+300|2e-300|1.23e-310|5e-324|0.0|0.0|0.0|inf|120.0|-3.0"
    );
}

// --- gcd / lcm ----------------------------------------------------------------

#[test]
fn gcd_and_lcm_take_hugeint_arguments() {
    // duckdb: 5, HUGEINT, 1, 36893488147419103230, BIGINT, 2, 12
    assert_eq!(
        texts(&[
            "gcd(18446744073709551615::UBIGINT, 5)",
            "typeof(gcd(18446744073709551615::UBIGINT, 5))",
            "gcd(170141183460469231731687303715884105727::HUGEINT, 7)",
            "lcm(18446744073709551615::UBIGINT, 2)",
            "typeof(lcm(1::INTEGER, 2))",
            "gcd(-9223372036854775808, 6)",
            "lcm(-4::HUGEINT, 6)",
        ]),
        "5|HUGEINT|1|36893488147419103230|BIGINT|2|12"
    );
    // No positive BIGINT holds 2^63 (DuckDB raises); an lcm past HUGEINT is NULL too.
    assert_eq!(text("gcd(-9223372036854775808, 0)"), "NULL");
    assert_eq!(text("lcm(170141183460469231731687303715884105727::HUGEINT, 2)"), "NULL");
}
