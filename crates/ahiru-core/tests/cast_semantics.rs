//! Regression tests for a batch of `CAST`/conversion and numeric-semantics fixes.
//!
//! Expected values were taken from `duckdb -csv -c "SELECT ..."` unless a
//! comment says otherwise. Two places where this engine deliberately differs
//! from DuckDB are called out at the point they appear:
//!
//!   - An undecorated decimal literal (`0.1`, `9007199254740993.0`) is a
//!     `DOUBLE` here and a `DECIMAL` in DuckDB, so the DuckDB reference values
//!     below always cast explicitly to `DOUBLE` where that matters.
//!   - Unsigned integer arithmetic **wraps inside the unsigned domain** here;
//!     DuckDB raises an out-of-range error instead. See `wrap_unsigned` in
//!     `expr::kernels` for why wrapping was chosen.
//!
//! A `SELECT <expr>` with no `FROM` is unsupported (`plan::bind`), so these run
//! against `tests/data/basic.parquet` with `LIMIT 1`, the same convention as
//! `scalar_function_fixes.rs`.

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

fn one(session: &mut Session, expr: &str) -> Value {
    run(session, &format!("SELECT {expr} AS x FROM t LIMIT 1"))[0][0].clone()
}

/// The value as text, via `CAST(... AS VARCHAR)` so the engine's own rendering
/// is what is asserted on.
fn text(session: &mut Session, expr: &str) -> String {
    match one(session, &format!("CAST({expr} AS VARCHAR)")) {
        Value::Bytes(b) => String::from_utf8(b).unwrap(),
        Value::Null => "NULL".into(),
        other => panic!("{expr}: not text: {other:?}"),
    }
}

fn f64_of(v: &Value) -> f64 {
    match v {
        Value::F64(x) => *x,
        other => panic!("not a double: {other:?}"),
    }
}

// --- 1/2. INTERVAL <-> VARCHAR ------------------------------------------------

#[test]
fn interval_to_varchar_uses_the_interval_text_form() {
    let mut s = session_with_basic();
    // Was the raw packed i128 (`18446744073709551616` for one day).
    assert_eq!(text(&mut s, "INTERVAL '1 day'"), "1 day");
    assert_eq!(text(&mut s, "INTERVAL '1 month'"), "1 month");
    assert_eq!(text(&mut s, "INTERVAL '2 hours 30 minutes'"), "02:30:00");
    assert_eq!(text(&mut s, "INTERVAL '-1 day'"), "-1 day");
    // The `||`/`concat` path shares the same conversion.
    assert_eq!(one(&mut s, "concat(INTERVAL '1 day', 'x')"), Value::Bytes(b"1 dayx".to_vec()));
}

#[test]
fn varchar_to_interval_reuses_the_literal_parser() {
    let mut s = session_with_basic();
    assert_eq!(text(&mut s, "CAST('1 day' AS INTERVAL)"), "1 day");
    assert_eq!(text(&mut s, "CAST('1 year 2 months' AS INTERVAL)"), "1 year 2 months");
    assert_eq!(text(&mut s, "CAST('04:05:06' AS INTERVAL)"), "04:05:06");
    // Same value either way in.
    assert_eq!(one(&mut s, "CAST('1 day' AS INTERVAL) = INTERVAL '1 day'"), Value::Bool(true));
    // Unreadable text is NULL, matching the DATE/TIME/TIMESTAMP convention.
    assert_eq!(one(&mut s, "CAST('not an interval' AS INTERVAL)"), Value::Null);
}

#[test]
fn interval_text_round_trips() {
    let mut s = session_with_basic();
    for lit in [
        "1 day",
        "1 month",
        "1 year 2 months 3 days 04:05:06.5",
        "-3 days",
        "00:00:00.000001",
        "10 years",
    ] {
        let expr = format!("CAST(CAST(INTERVAL '{lit}' AS VARCHAR) AS INTERVAL)");
        assert_eq!(
            one(&mut s, &format!("{expr} = INTERVAL '{lit}'")),
            Value::Bool(true),
            "round trip of INTERVAL '{lit}'"
        );
    }
}

// --- 3. DOUBLE <-> VARCHAR is shortest-round-trip -----------------------------

#[test]
fn double_to_varchar_is_shortest_round_trip() {
    let mut s = session_with_basic();
    // Previously "9007199254740990" / "1234567890123460": a 15-digit
    // approximation mangled integral doubles inside f64's exact range.
    // (The literal is a DOUBLE here, so the value is the nearest double,
    // 9007199254740992; DuckDB's DECIMAL literal keeps ...993.)
    assert_eq!(text(&mut s, "9007199254740993.0"), "9007199254740992.0");
    assert_eq!(text(&mut s, "1234567890123456.0"), "1234567890123456.0");
    assert_eq!(text(&mut s, "0.1 + 0.2"), "0.30000000000000004");
    // Fixed vs exponential notation, and the exponent spelling, match DuckDB.
    assert_eq!(text(&mut s, "CAST(100 AS DOUBLE)"), "100.0");
    assert_eq!(text(&mut s, "1e30"), "1e+30");
    assert_eq!(text(&mut s, "1e-30"), "1e-30");
    assert_eq!(text(&mut s, "1e-5"), "1e-05");
    assert_eq!(text(&mut s, "1e-4"), "0.0001");
}

#[test]
fn double_survives_a_varchar_round_trip() {
    let mut s = session_with_basic();
    for expr in [
        "0.1 + 0.2",
        "1.7976931348623157e308",
        "-1.7976931348623157e308",
        "9007199254740993.0",
        "1e-300",
        "CAST(0 AS DOUBLE)",
        "1.0 / 3.0",
    ] {
        assert_eq!(
            one(&mut s, &format!("CAST(CAST({expr} AS VARCHAR) AS DOUBLE) = {expr}")),
            Value::Bool(true),
            "round trip of {expr}"
        );
    }
    // Was `inf`: the old text was a rounded-up 15-digit value, and the old
    // text-to-double parser overflowed on the way back in.
    let back = one(&mut s, "CAST(CAST(1.7976931348623157e308 AS VARCHAR) AS DOUBLE)");
    assert_eq!(f64_of(&back), f64::MAX);
}

#[test]
fn text_to_double_is_correctly_rounded() {
    let mut s = session_with_basic();
    // 17 significant digits: the old mantissa*10^e path drifted by ULPs here.
    assert_eq!(f64_of(&one(&mut s, "CAST('0.30000000000000004' AS DOUBLE)")), 0.1 + 0.2);
    assert_eq!(f64_of(&one(&mut s, "CAST('5e-324' AS DOUBLE)")), 5e-324);
    assert_eq!(f64_of(&one(&mut s, "CAST('1.7976931348623157e+308' AS DOUBLE)")), f64::MAX);
    // The IEEE spellings still work, and junk is still NULL.
    assert!(f64_of(&one(&mut s, "CAST('inf' AS DOUBLE)")).is_infinite());
    assert!(f64_of(&one(&mut s, "CAST('nan' AS DOUBLE)")).is_nan());
    assert_eq!(one(&mut s, "CAST('not a number' AS DOUBLE)"), Value::Null);
}

// --- 4. DOUBLE -> DECIMAL no longer scales in f64 -----------------------------

#[test]
fn double_to_decimal_does_not_scale_in_floating_point() {
    let mut s = session_with_basic();
    // The double nearest 12345678901234567890.5 is 12345678901234567168.
    // `f *= 10.0` used to round that a second time into 12345678901234566758.4,
    // more than a thousand off; the exact rescale keeps every digit the
    // shortest round-trip rendering carries. (DuckDB prints
    // 12345678901234566758.4 here, from the very same f64 multiply.)
    assert_eq!(
        text(&mut s, "CAST(12345678901234567890.5 AS DECIMAL(38,1))"),
        "12345678901234567000.0"
    );
    // Casting through text gives the same answer, by construction.
    assert_eq!(
        one(
            &mut s,
            "CAST(12345678901234567890.5 AS DECIMAL(38,1)) \
             = CAST(CAST(12345678901234567890.5 AS VARCHAR) AS DECIMAL(38,1))"
        ),
        Value::Bool(true)
    );
    // Ordinary cases are unchanged and match DuckDB.
    assert_eq!(text(&mut s, "CAST(1.5 AS DECIMAL(4,1))"), "1.5");
    assert_eq!(
        text(&mut s, "CAST(CAST(0.1 AS DOUBLE) AS DECIMAL(38,20))"),
        "0.10000000000000000000"
    );
    assert_eq!(
        text(&mut s, "CAST(1e-300 AS DECIMAL(38,37))"),
        "0.0000000000000000000000000000000000000"
    );
    // Out of the target's range is still NULL.
    assert_eq!(one(&mut s, "CAST(1e300 AS DECIMAL(38,1))"), Value::Null);
}

// --- 5. Casting to BOOLEAN tests "not zero" -----------------------------------

#[test]
fn float_and_decimal_to_boolean_test_non_zero() {
    let mut s = session_with_basic();
    // All of these were `false` (or NULL for NaN): the value was rounded to an
    // integer first and then tested, so anything under 0.5 vanished.
    assert_eq!(one(&mut s, "CAST(0.4 AS BOOLEAN)"), Value::Bool(true));
    assert_eq!(one(&mut s, "CAST(-0.4 AS BOOLEAN)"), Value::Bool(true));
    assert_eq!(one(&mut s, "CAST(1e-300 AS BOOLEAN)"), Value::Bool(true));
    assert_eq!(one(&mut s, "CAST(CAST('nan' AS DOUBLE) AS BOOLEAN)"), Value::Bool(true));
    assert_eq!(one(&mut s, "CAST(CAST('inf' AS DOUBLE) AS BOOLEAN)"), Value::Bool(true));
    assert_eq!(one(&mut s, "CAST(CAST('0.1' AS DECIMAL(3,1)) AS BOOLEAN)"), Value::Bool(true));
    // Zero, in every spelling, is still false.
    assert_eq!(one(&mut s, "CAST(CAST(0 AS DOUBLE) AS BOOLEAN)"), Value::Bool(false));
    assert_eq!(one(&mut s, "CAST(-0.0 AS BOOLEAN)"), Value::Bool(false));
    assert_eq!(one(&mut s, "CAST(CAST('0.0' AS DECIMAL(3,1)) AS BOOLEAN)"), Value::Bool(false));
    // A NULL input stays NULL rather than becoming false.
    assert_eq!(one(&mut s, "CAST(CAST(NULL AS DOUBLE) AS BOOLEAN)"), Value::Null);
}

// --- 6. Unsigned arithmetic stays inside the unsigned domain ------------------

#[test]
fn unsigned_arithmetic_wraps_within_its_own_width() {
    let mut s = session_with_basic();
    // Each of these used to produce -1 while still calling itself unsigned.
    // DuckDB raises an out-of-range error; this engine wraps, the documented
    // rule for signed integer overflow, now applied at the declared width.
    assert_eq!(text(&mut s, "CAST(1 AS UTINYINT) - CAST(2 AS UTINYINT)"), "255");
    assert_eq!(text(&mut s, "CAST(1 AS USMALLINT) - CAST(2 AS USMALLINT)"), "65535");
    assert_eq!(text(&mut s, "CAST(1 AS UINTEGER) - CAST(2 AS UINTEGER)"), "4294967295");
    assert_eq!(text(&mut s, "CAST(1 AS UBIGINT) - CAST(2 AS UBIGINT)"), "18446744073709551615");
    // The declared type is unchanged, so the result is now actually inside it.
    assert_eq!(
        one(&mut s, "typeof(CAST(1 AS UINTEGER) - CAST(2 AS UINTEGER))"),
        Value::Bytes(b"UINTEGER".to_vec())
    );
    assert_eq!(
        one(&mut s, "CAST(1 AS UINTEGER) - CAST(2 AS UINTEGER) >= CAST(0 AS UINTEGER)"),
        Value::Bool(true)
    );
    // Wrapping at the top of the range too, and unary negation.
    assert_eq!(text(&mut s, "CAST(255 AS UTINYINT) + CAST(1 AS UTINYINT)"), "0");
    assert_eq!(text(&mut s, "-CAST(1 AS UTINYINT)"), "255");
    // In-range arithmetic is untouched.
    assert_eq!(text(&mut s, "CAST(5 AS UINTEGER) - CAST(2 AS UINTEGER)"), "3");
    assert_eq!(text(&mut s, "CAST(3 AS UBIGINT) * CAST(4 AS UBIGINT)"), "12");
    // And so is signed arithmetic.
    assert_eq!(text(&mut s, "CAST(1 AS INTEGER) - CAST(2 AS INTEGER)"), "-1");
}

#[test]
fn unsigned_wrap_applies_to_columns_too() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT CAST(CAST(id AS UINTEGER) - CAST(2 AS UINTEGER) AS VARCHAR) AS x \
         FROM t ORDER BY id LIMIT 3",
    );
    let got: Vec<String> = rows
        .iter()
        .map(|r| match &r[0] {
            Value::Bytes(b) => String::from_utf8(b.clone()).unwrap(),
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(got, vec!["4294967294", "4294967295", "0"]);
}

// --- 7. Oversized expressions fail cleanly ------------------------------------

#[test]
fn a_huge_in_list_reports_a_limit_rather_than_aliasing_registers() {
    let mut s = session_with_basic();
    let mut sql = String::from("SELECT count(*) FROM t WHERE id IN (0");
    for i in 1..70_000 {
        sql.push(',');
        sql.push_str(&i.to_string());
    }
    sql.push(')');
    // Debug builds used to panic with "attempt to add with overflow"; the
    // shipping wasm profile (overflow checks off) silently aliased registers
    // and answered wrongly. Either a clean error or a correct answer is fine;
    // a panic or a wrong answer is not.
    let result = match s.prepare(&sql, &[]) {
        Ok(Prepared::Ready(mut q)) => loop {
            match s.step(&mut q) {
                Ok(QueryStep::Batch(_)) => continue,
                Ok(_) => break Ok(()),
                Err(e) => break Err(e.code),
            }
        },
        Ok(Prepared::NeedIo(_)) => panic!("unexpected NeedIo"),
        Err(e) => Err(e.code),
    };
    assert!(result.is_err(), "an expression this large must not silently succeed");
}

// --- 8. NaN is self-consistent across comparison and join paths ---------------

#[test]
fn nan_equals_nan_everywhere() {
    let mut s = session_with_basic();
    assert_eq!(one(&mut s, "CAST('nan' AS DOUBLE) = CAST('nan' AS DOUBLE)"), Value::Bool(true));
    assert_eq!(one(&mut s, "CAST('nan' AS DOUBLE) > 1.0"), Value::Bool(true));
    assert_eq!(one(&mut s, "CAST('nan' AS DOUBLE) < 1.0"), Value::Bool(false));
    assert_eq!(one(&mut s, "CAST('nan' AS DOUBLE) >= CAST('inf' AS DOUBLE)"), Value::Bool(true));
    assert_eq!(one(&mut s, "CAST('nan' AS DOUBLE) <> 1.0"), Value::Bool(true));
    assert_eq!(
        one(&mut s, "CAST('nan' AS DOUBLE) IN (CAST('nan' AS DOUBLE), 1.0)"),
        Value::Bool(true)
    );
    // A NULL comparison is still NULL, not true.
    assert_eq!(one(&mut s, "CAST('nan' AS DOUBLE) = CAST(NULL AS DOUBLE)"), Value::Null);
}

#[test]
fn nan_joins_the_same_way_on_both_physical_plans() {
    let mut s = session_with_basic();
    let hash = run(
        &mut s,
        "SELECT count(*) AS c FROM (SELECT CAST('nan' AS DOUBLE) AS x FROM t LIMIT 3) a \
         JOIN (SELECT CAST('nan' AS DOUBLE) AS x FROM t LIMIT 2) b ON a.x = b.x",
    );
    // `OR false` is not an equi-condition, so this takes the nested-loop path.
    let loops = run(
        &mut s,
        "SELECT count(*) AS c FROM (SELECT CAST('nan' AS DOUBLE) AS x FROM t LIMIT 3) a \
         JOIN (SELECT CAST('nan' AS DOUBLE) AS x FROM t LIMIT 2) b ON a.x = b.x OR false",
    );
    assert_eq!(hash, loops, "the physical plan must not change the answer");
    assert_eq!(hash[0][0], Value::I64(6));
}

#[test]
fn nan_and_negative_zero_still_group_as_one() {
    let mut s = session_with_basic();
    // Every row is a NaN, produced independently, and they land in one group.
    let rows = run(
        &mut s,
        "SELECT count(*) AS c FROM (SELECT CAST('nan' AS DOUBLE) + id AS x FROM t) g GROUP BY x",
    );
    assert_eq!(rows.len(), 1, "all NaNs are one group");
    // Half the rows produce -0.0 and half 0.0; DISTINCT still sees one value.
    let rows = run(
        &mut s,
        "SELECT count(DISTINCT CASE WHEN id % 2 = 0 THEN CAST(0 AS DOUBLE) \
                                    ELSE -CAST(0 AS DOUBLE) END) AS c FROM t",
    );
    assert_eq!(rows[0][0], Value::I64(1), "-0.0 and 0.0 are one value");
}

// --- 9. DOUBLE -> FLOAT overflow ----------------------------------------------

#[test]
fn double_to_float_overflow_is_null_not_infinity() {
    let mut s = session_with_basic();
    // Was `inf`, which claims the value survived the narrowing.
    assert_eq!(one(&mut s, "TRY_CAST(1e39 AS FLOAT)"), Value::Null);
    assert_eq!(one(&mut s, "CAST(1e308 AS FLOAT)"), Value::Null);
    assert_eq!(one(&mut s, "CAST(-1e308 AS FLOAT)"), Value::Null);
    assert_eq!(one(&mut s, "CAST('1e39' AS FLOAT)"), Value::Null);
    // An infinity that was already in the input still passes through.
    assert!(f64_of(&one(&mut s, "CAST(CAST('inf' AS DOUBLE) AS FLOAT)")).is_infinite());
    assert!(f64_of(&one(&mut s, "CAST('-inf' AS FLOAT)")).is_infinite());
    assert!(f64_of(&one(&mut s, "CAST(CAST('nan' AS DOUBLE) AS FLOAT)")).is_nan());
    // In-range narrowing is unchanged (1.1 is not representable in f32).
    assert_eq!(f64_of(&one(&mut s, "CAST(1.1 AS FLOAT)")), 1.1f32 as f64);
    assert_eq!(f64_of(&one(&mut s, "CAST(3.4e38 AS FLOAT)")), 3.4e38f32 as f64);
}
