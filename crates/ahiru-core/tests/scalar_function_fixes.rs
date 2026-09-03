//! Regression tests for a batch of scalar-function correctness fixes.
//!
//! Every expectation is the actual output of `duckdb -csv -c "SELECT ..."` unless a comment says
//! otherwise (`%f`'s exact expansion of a large double follows C's `printf`, which DuckDB's own
//! `fmt`-based implementation does not -- that divergence is called out where it appears).
//!
//! A `SELECT <expr>` with no `FROM` is unsupported (`plan::bind`), so these run against
//! `tests/data/basic.parquet` with `LIMIT 1`, the same convention as
//! `printf_format_glob_similar_array.rs`.

use ahiru_core::error::Code;
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
    let rows = run(session, &format!("SELECT {expr} AS x FROM t LIMIT 1"));
    rows[0][0].clone()
}

/// The error code a query fails with at execution time.
fn exec_err(session: &mut Session, expr: &str) -> Option<Code> {
    let sql = format!("SELECT {expr} AS x FROM t LIMIT 1");
    let mut q = match session.prepare(&sql, &[]).ok()? {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => return None,
    };
    loop {
        match session.step(&mut q) {
            Ok(QueryStep::Batch(_)) => continue,
            Ok(_) => return None,
            Err(e) => return Some(e.code),
        }
    }
}

fn s(v: &str) -> Value {
    Value::Bytes(v.as_bytes().to_vec())
}

// --- repeat -------------------------------------------------------------------

#[test]
fn repeat_short_circuits_on_an_empty_string() {
    let mut sess = session_with_basic();
    // duckdb: repeat('', 9223372036854775807) -> '' (instantly). This used to spin forever:
    // the length guard passes because the product is zero, and the loop then ran i64::MAX times
    // appending nothing.
    assert_eq!(one(&mut sess, "repeat('', 9223372036854775807)"), s(""));
    assert_eq!(one(&mut sess, "repeat('', 0)"), s(""));
    assert_eq!(one(&mut sess, "repeat('ab', 3)"), s("ababab"));
    // duckdb: repeat('a', -1) -> ''
    assert_eq!(one(&mut sess, "repeat('a', -1)"), s(""));
    // A non-empty string with an unbounded count is a clean error, not an unbounded allocation.
    assert_eq!(exec_err(&mut sess, "repeat('a', 9223372036854775807)"), Some(Code::LimitExceeded));
}

// --- printf / format on logical types -----------------------------------------

#[test]
fn printf_and_format_render_logical_types_like_the_text_cast() {
    let mut sess = session_with_basic();
    // These all used to print the physical integer: 19723, 36000000000, 1704103200500000, 15.
    // duckdb: 2024-01-01 / 10:00:00 / 2024-01-01 10:00:00.5 / 1.50 / 1.5
    assert_eq!(one(&mut sess, "printf('%s', DATE '2024-01-01')"), s("2024-01-01"));
    assert_eq!(one(&mut sess, "format('{}', DATE '2024-01-01')"), s("2024-01-01"));
    assert_eq!(one(&mut sess, "format('{}', TIME '10:00:00')"), s("10:00:00"));
    assert_eq!(
        one(&mut sess, "format('{}', TIMESTAMP '2024-01-01 10:00:00.5')"),
        s("2024-01-01 10:00:00.5")
    );
    assert_eq!(one(&mut sess, "printf('%.2f', 1.5::DECIMAL(3,1))"), s("1.50"));
    assert_eq!(one(&mut sess, "format('{}', 1.5::DECIMAL(3,1))"), s("1.5"));
    // HUGEINT used to be rejected outright (UnsupportedFeature).
    // duckdb: 12345678901234567890123
    assert_eq!(
        one(&mut sess, "format('{}', 12345678901234567890123::HUGEINT)"),
        s("12345678901234567890123")
    );
    assert_eq!(one(&mut sess, "printf('%d', 1::HUGEINT)"), s("1"));
    assert_eq!(
        one(&mut sess, "printf('%d', 170141183460469231731687303715884105727::HUGEINT)"),
        s("170141183460469231731687303715884105727")
    );
    // The types that already worked must not shift.
    assert_eq!(one(&mut sess, "printf('%s|%s|%s', 'x', 42, true)"), s("x|42|true"));
    assert_eq!(one(&mut sess, "format('{} {}', 'a', 3)"), s("a 3"));
    // ... nor on a real table column (docs/sql/functions-string.md advertises exactly this).
    // (`id`/`flag` rather than the DOUBLE `score`: how a float renders is `kernels::fmt_f64`'s
    // business, not this fix's.)
    assert_eq!(one(&mut sess, "printf('%s|%s|%s', d, id, flag)"), s("2024-01-01 00:00:00|0|true"));
}

// --- printf %f ----------------------------------------------------------------

#[test]
fn printf_f_is_correctly_rounded_fixed_point() {
    let mut sess = session_with_basic();
    // duckdb: 100000000000000000000.000000. The old f64 `x * 10^prec` product printed
    // 100000000000000004764.729344 instead.
    assert_eq!(one(&mut sess, "printf('%f', 1e20)"), s("100000000000000000000.000000"));
    // Halves round to even, the way C's printf does. duckdb: 2 / 4 / 0.2
    assert_eq!(one(&mut sess, "printf('%.0f', 2.5)"), s("2"));
    assert_eq!(one(&mut sess, "printf('%.0f', 3.5)"), s("4"));
    assert_eq!(one(&mut sess, "printf('%.1f', 0.25)"), s("0.2"));
    assert_eq!(one(&mut sess, "printf('%.1f', 0.35)"), s("0.3"));
    // The true binary expansion of 0.1, not an f64-shaped run of zeros.
    // duckdb: 0.10000000000000000555
    assert_eq!(one(&mut sess, "printf('%.20f', 0.1)"), s("0.10000000000000000555"));
    // Unchanged shapes.
    assert_eq!(one(&mut sess, "printf('%f', 3.5)"), s("3.500000"));
    assert_eq!(one(&mut sess, "printf('%.2f', 3.14159)"), s("3.14"));
    assert_eq!(one(&mut sess, "printf('%08.2f|%-8.2f|', -3.14159, 2.0)"), s("-0003.14|2.00    |"));
    assert_eq!(one(&mut sess, "printf('%.2f', 5e-324)"), s("0.00"));
    assert_eq!(one(&mut sess, "printf('%f', 1.0/0.0)"), s("inf"));
    // The whole exact decimal expansion of the double nearest 1e300 -- what C's `printf("%f")`
    // prints. DuckDB's fmt-based implementation stops at the shortest round-trip digits and pads
    // with zeros instead, so this one line deliberately does not match `duckdb -csv`.
    let big = match one(&mut sess, "printf('%f', 1e300)") {
        Value::Bytes(b) => String::from_utf8(b).unwrap(),
        v => panic!("expected text, got {v:?}"),
    };
    assert!(big.starts_with("1000000000000000052504760255204420248704468581108159154915854115511"));
    assert!(big.ends_with(".000000"));
    assert_eq!(big.len(), 301 + 7);
}

// --- lambdas over string elements ---------------------------------------------

#[test]
fn lambda_parameters_bound_to_string_elements_behave_like_that_string() {
    let mut sess = session_with_basic();
    // duckdb: [bb] -- i.e. only 'bb' survives, because length('bb') is 2. Binding the element as
    // its JSON serialization made `CAST(x AS VARCHAR)` yield `"a"` / `"bb"`, so length() counted
    // the quotes and both elements passed.
    assert_eq!(
        one(&mut sess, "list_filter(['a','bb'], x -> length(CAST(x AS VARCHAR)) > 1)"),
        s(r#"["bb"]"#)
    );
    // duckdb: [A]. This used to produce the doubly-serialized ["\"A\""].
    assert_eq!(
        one(&mut sess, "list_transform(['a'], x -> upper(CAST(x AS VARCHAR)))"),
        s(r#"["A"]"#)
    );
    // Quotes and escapes inside an element survive the round trip (no string trimming here).
    assert_eq!(
        one(&mut sess, r#"list_transform(['a"b'], x -> upper(CAST(x AS VARCHAR)))"#),
        s(r#"["A\"B"]"#)
    );
    // The identity transform still re-serializes to valid JSON.
    assert_eq!(one(&mut sess, "list_transform(['a','b'], x -> x)"), s(r#"["a","b"]"#));
    // Non-string elements are untouched: they keep arriving as JSON text.
    assert_eq!(
        one(&mut sess, "list_transform([1,2,3], x -> CAST(CAST(x AS VARCHAR) AS INTEGER) + 1)"),
        s("[2,3,4]")
    );
    assert_eq!(one(&mut sess, "list_transform([1,NULL,3], x -> x)"), s("[1,null,3]"));
    assert_eq!(one(&mut sess, "list_transform([[1,2],[3]], y -> y)"), s("[[1,2],[3]]"));
    // list_reduce threads the accumulator through the same binding.
    assert_eq!(
        one(
            &mut sess,
            "list_reduce(['a','b','c'], (a,b) -> CAST(a AS VARCHAR) || CAST(b AS VARCHAR))"
        ),
        s(r#""abc""#)
    );
}

// --- math ---------------------------------------------------------------------

#[test]
fn math_kernels_satisfy_exact_identities() {
    let mut sess = session_with_basic();
    // duckdb: 1.4142135623730951 / 17.320508075688775 / 2.718281828459045
    assert_eq!(one(&mut sess, "sqrt(2)"), Value::F64(core::f64::consts::SQRT_2));
    assert_eq!(one(&mut sess, "sqrt(300)"), Value::F64(17.320_508_075_688_775));
    assert_eq!(one(&mut sess, "exp(1)"), Value::F64(core::f64::consts::E));
    assert_eq!(one(&mut sess, "ln(10)"), Value::F64(core::f64::consts::LN_10));
    // duckdb: true for all of these; they were false before.
    assert_eq!(one(&mut sess, "pow(10,-2) = 0.01"), Value::Bool(true));
    assert_eq!(one(&mut sess, "cbrt(8) = 2"), Value::Bool(true));
    assert_eq!(one(&mut sess, "cbrt(1000) = 10"), Value::Bool(true));
    assert_eq!(one(&mut sess, "cbrt(-27) = -3"), Value::Bool(true));
    assert_eq!(one(&mut sess, "sqrt(4) = 2 AND log10(100) = 2 AND log2(8) = 3"), Value::Bool(true));
    // Subnormal inputs: the exponent-based seeds are unusable there and used to be off by
    // several orders of magnitude. duckdb: 2.2227587494850775e-162
    assert_eq!(one(&mut sess, "sqrt(5e-324)"), Value::F64(2.222_758_749_485_077_5e-162));
    assert_eq!(one(&mut sess, "cbrt(1e-310)"), Value::F64(4.641_588_833_612_774e-104));
}

#[test]
fn abs_and_sign_of_negative_zero_are_positive_zero() {
    let mut sess = session_with_basic();
    // duckdb: abs(-0.0) -> 0.0, sign(-0.0) -> 0. `-0.0 < 0.0` is false, so the comparison-based
    // forms returned the negative zero unchanged and it printed as `-0`.
    assert_eq!(one(&mut sess, "1.0 / abs(-0.0) > 0"), Value::Bool(true));
    assert_eq!(one(&mut sess, "1.0 / sign(-0.0) > 0"), Value::Bool(true));
    assert_eq!(one(&mut sess, "abs(-0.0) = 0 AND sign(-0.0) = 0"), Value::Bool(true));
    assert_eq!(one(&mut sess, "abs(-2.5)"), Value::F64(2.5));
    assert_eq!(one(&mut sess, "sign(-5.0)"), Value::F64(-1.0));
}

// --- hex ----------------------------------------------------------------------

#[test]
fn hex_covers_the_full_width_of_its_argument() {
    let mut sess = session_with_basic();
    // duckdb: 32 F's for HUGEINT, 16 for everything narrower.
    assert_eq!(one(&mut sess, "hex(-1::HUGEINT)"), s("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"));
    assert_eq!(one(&mut sess, "to_hex(-1::HUGEINT)"), s("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"));
    assert_eq!(
        one(&mut sess, "hex(170141183460469231731687303715884105727::HUGEINT)"),
        s("7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
    );
    // A UBIGINT above i64::MAX used to be NULL, lost in the cast to BIGINT.
    assert_eq!(one(&mut sess, "hex(18446744073709551615::UBIGINT)"), s("FFFFFFFFFFFFFFFF"));
    // Unchanged: narrow types widen to 64 bits, positives drop leading zeros, bytes are dumped.
    assert_eq!(one(&mut sess, "hex(-1::TINYINT)"), s("FFFFFFFFFFFFFFFF"));
    assert_eq!(one(&mut sess, "hex(1::TINYINT)"), s("1"));
    assert_eq!(one(&mut sess, "hex(0)"), s("0"));
    assert_eq!(one(&mut sess, "hex(255)"), s("FF"));
    assert_eq!(one(&mut sess, "to_hex(-2::SMALLINT)"), s("FFFFFFFFFFFFFFFE"));
    assert_eq!(one(&mut sess, "hex('AB')"), s("4142"));
    assert_eq!(one(&mut sess, "hex(NULL::INT)"), Value::Null);
}

// --- FLOAT arithmetic ---------------------------------------------------------

#[test]
fn float_arithmetic_is_rounded_back_to_f32() {
    let mut sess = session_with_basic();
    // FLOAT shares DOUBLE's physical f64 register, and `FLOAT op FLOAT` stays FLOAT, so the
    // f64 result has to be narrowed. Without that, every consumer that assumes a FLOAT value
    // is exactly an f32 misbehaved.
    // duckdb: 16777216.0 / true
    assert_eq!(one(&mut sess, "CAST(16777216::FLOAT + 1::FLOAT AS VARCHAR)"), s("16777216.0"));
    assert_eq!(one(&mut sess, "(16777216::FLOAT + 1::FLOAT) = 16777216::FLOAT"), Value::Bool(true));
    // f32 overflow becomes infinity, exactly as the f32 computation would.
    // duckdb: inf / true
    assert_eq!(one(&mut sess, "CAST(3.4028235e38::FLOAT * 2::FLOAT AS VARCHAR)"), s("inf"));
    assert_eq!(one(&mut sess, "isinf(3.4028235e38::FLOAT * 2::FLOAT)"), Value::Bool(true));
    // duckdb: 0.33333334 (the f32 quotient), not the f64 0.33333333333333331's f32 spelling.
    assert_eq!(one(&mut sess, "CAST(1::FLOAT / 3::FLOAT AS VARCHAR)"), s("0.33333334"));
    // Unary negation runs through the same kernel.
    assert_eq!(one(&mut sess, "CAST(-(0.1::FLOAT) AS VARCHAR)"), s("-0.1"));
    // The result is still a FLOAT, and CAST(.. AS VARCHAR) -> CAST(.. AS FLOAT) round-trips.
    assert_eq!(one(&mut sess, "typeof(1::FLOAT + 1::FLOAT)"), s("FLOAT"));
    assert_eq!(
        one(
            &mut sess,
            "CAST(CAST(0.1::FLOAT + 0.2::FLOAT AS VARCHAR) AS FLOAT) = 0.1::FLOAT + 0.2::FLOAT"
        ),
        Value::Bool(true)
    );
    // Mixing FLOAT with another numeric type still widens to DOUBLE and is left alone.
    assert_eq!(one(&mut sess, "typeof(1::FLOAT + 1::DOUBLE)"), s("DOUBLE"));
}

// --- ln / log / pow on non-finite input ---------------------------------------

#[test]
fn the_log_family_handles_infinity_and_nan() {
    let mut sess = session_with_basic();
    // `f_ln` decodes the exponent field bitwise; for inf/NaN that field is 0x7ff, so it used
    // to answer 1024 * ln 2 (709.78...) -- a plausible finite number -- for every one of these.
    // duckdb: inf for all four.
    for e in [
        "ln('inf'::DOUBLE)",
        "log10('inf'::DOUBLE)",
        "log2('inf'::DOUBLE)",
        "log(2, 'inf'::DOUBLE)",
    ] {
        assert_eq!(one(&mut sess, e), Value::F64(f64::INFINITY), "{e}");
    }
    // duckdb: nan for all three.
    for e in ["ln('nan'::DOUBLE)", "log2('nan'::DOUBLE)", "log(2, 'nan'::DOUBLE)"] {
        match one(&mut sess, e) {
            Value::F64(v) => assert!(v.is_nan(), "{e}: {v}"),
            other => panic!("{e}: {other:?}"),
        }
    }
    // duckdb: log(inf, 2) -> 0.0 (a finite numerator over an infinite denominator).
    assert_eq!(one(&mut sess, "log('inf'::DOUBLE, 2.0)"), Value::F64(0.0));
    // `pow` uses exp(y * ln|x|) for a non-integer exponent and inherited the bug.
    // duckdb: inf / 0.0 / inf / 0.0.
    assert_eq!(one(&mut sess, "pow('inf'::DOUBLE, 0.5)"), Value::F64(f64::INFINITY));
    assert_eq!(one(&mut sess, "pow('inf'::DOUBLE, -0.5)"), Value::F64(0.0));
    // An infinite *negative* base with a non-integer exponent is IEEE's one exception to
    // "a negative base needs an integer exponent". duckdb: inf / 0.0.
    assert_eq!(one(&mut sess, "pow('-inf'::DOUBLE, 0.5)"), Value::F64(f64::INFINITY));
    assert_eq!(one(&mut sess, "pow('-inf'::DOUBLE, -0.5)"), Value::F64(0.0));
    // Unchanged: a finite negative base with a non-integer exponent is still NaN (duckdb: nan),
    // a non-positive argument is still NULL (duckdb errors), and the ordinary values still work.
    match one(&mut sess, "pow(-2.0, 0.5)") {
        Value::F64(v) => assert!(v.is_nan()),
        other => panic!("{other:?}"),
    }
    assert_eq!(one(&mut sess, "ln(0.0)"), Value::Null);
    assert_eq!(one(&mut sess, "ln(-1.0)"), Value::Null);
    assert_eq!(one(&mut sess, "ln(-1.0 / 0.0)"), Value::Null);
    assert_eq!(one(&mut sess, "log2(8.0)"), Value::F64(3.0));
    assert_eq!(one(&mut sess, "ln(1.0)"), Value::F64(0.0));
}

// --- DATE +- INTEGER overflow -------------------------------------------------

#[test]
fn date_arithmetic_that_wraps_i32_is_null_not_a_fictitious_date() {
    let mut sess = session_with_basic();
    // The i32 lane wraps, and the old guard only caught a sum landing *exactly* on one of
    // DuckDB's three reserved sentinels. A sum that wrapped past one came back as an ordinary
    // negative day count and printed as a date five million years in the past.
    // duckdb: "Out of Range Error: Date out of range" for all three; NULL is this engine's
    // convention for an out-of-range argument (see docs/sql/limitations.md).
    assert_eq!(one(&mut sess, "DATE '2024-01-01' + 2147483647"), Value::Null);
    assert_eq!(one(&mut sess, "DATE '2024-01-01' + 2147480000"), Value::Null);
    assert_eq!(one(&mut sess, "DATE '2024-01-01' + 2147463924"), Value::Null);
    assert_eq!(one(&mut sess, "make_date(2024, 1, 1) + 2147483000"), Value::Null);
    // The same wrap in the other direction, from the smallest date duckdb has.
    assert_eq!(one(&mut sess, "(DATE '1970-01-01' - 2147483646) - 100"), Value::Null);
    // Unchanged: everything inside the range duckdb accepts is still a date.
    assert_eq!(
        one(&mut sess, "CAST(DATE '1970-01-01' + 2147483646 AS VARCHAR)"),
        s("5881580-07-10")
    );
    assert_eq!(one(&mut sess, "CAST(DATE '2024-01-01' + 1 AS VARCHAR)"), s("2024-01-02"));
    assert_eq!(one(&mut sess, "CAST(DATE '2024-01-01' - 1 AS VARCHAR)"), s("2023-12-31"));
}

// --- bit shifts ---------------------------------------------------------------

#[test]
fn the_right_shift_saturates_to_zero_instead_of_null() {
    let mut sess = session_with_basic();
    // duckdb: 0, 0, 0, 0 (the right shift is defined for every amount; only the *left*
    // shift errors out of range, which this engine deliberately answers with NULL).
    assert_eq!(one(&mut sess, "8 >> 64"), Value::I64(0));
    assert_eq!(one(&mut sess, "8 >> -1"), Value::I64(0));
    assert_eq!(one(&mut sess, "-8 >> 64"), Value::I64(0));
    assert_eq!(one(&mut sess, "bit_shift_right(8, 1000000)"), Value::I64(0));
    // Unchanged: an arithmetic (sign-extending) shift inside the range, and `<<`'s NULL.
    assert_eq!(one(&mut sess, "-8 >> 1"), Value::I64(-4));
    assert_eq!(one(&mut sess, "8 >> 3"), Value::I64(1));
    assert_eq!(one(&mut sess, "1 << 64"), Value::Null);
}

// --- ascii / unicode / ord ----------------------------------------------------

#[test]
fn unicode_and_ord_answer_minus_one_for_the_empty_string() {
    let mut sess = session_with_basic();
    // duckdb: unicode('') and ord('') are -1, ascii('') is 0. All three used to share one
    // kernel and answered 0.
    assert_eq!(one(&mut sess, "unicode('')"), Value::I64(-1));
    assert_eq!(one(&mut sess, "ord('')"), Value::I64(-1));
    assert_eq!(one(&mut sess, "ascii('')"), Value::I64(0));
    // Unchanged everywhere else: the code point of the first character, NULL for NULL.
    assert_eq!(one(&mut sess, "unicode('abc')"), Value::I64(97));
    assert_eq!(one(&mut sess, "ord('abc')"), Value::I64(97));
    assert_eq!(one(&mut sess, "ascii('abc')"), Value::I64(97));
    assert_eq!(one(&mut sess, "unicode('\u{00e9}')"), Value::I64(233));
    assert_eq!(one(&mut sess, "unicode(NULL)"), Value::Null);
}

// --- floor / ceil / trunc / round on negative zero -----------------------------

#[test]
fn rounding_toward_zero_keeps_the_sign_of_zero() {
    let mut sess = session_with_basic();
    // These all go through `f_trunc`, which routed through i64 and dropped the sign bit.
    // duckdb: -0.0 for each of these on a DOUBLE.
    for e in [
        "CAST(floor(-0.0::DOUBLE) AS VARCHAR)",
        "CAST(ceil(-0.3::DOUBLE) AS VARCHAR)",
        "CAST(trunc(-0.3::DOUBLE) AS VARCHAR)",
        "CAST(round(-0.3::DOUBLE) AS VARCHAR)",
    ] {
        assert_eq!(one(&mut sess, e), s("-0.0"), "{e}");
    }
    // Unchanged: a non-zero result, a positive zero, and the integer cast of a negative zero.
    assert_eq!(one(&mut sess, "floor(-0.3::DOUBLE)"), Value::F64(-1.0));
    assert_eq!(one(&mut sess, "CAST(floor(0.3::DOUBLE) AS VARCHAR)"), s("0.0"));
    assert_eq!(one(&mut sess, "CAST(-0.3::DOUBLE AS INTEGER)"), Value::I32(0));
    assert_eq!(one(&mut sess, "trunc(2.7::DOUBLE)"), Value::F64(2.0));
}

// --- DECIMAL -> DOUBLE --------------------------------------------------------

#[test]
fn a_wide_decimal_converts_to_double_without_rounding_twice() {
    let mut sess = session_with_basic();
    // `scaled_i128 as f64 / 10^scale` rounds twice and landed one ulp low
    // (12345678901234565120). duckdb: 12345678901234567168.0.
    assert_eq!(
        one(
            &mut sess,
            "printf('%.1f', CAST(12345678901234567890.123456789::DECIMAL(38,9) AS DOUBLE))"
        ),
        s("12345678901234567168.0")
    );
    // Unchanged: the narrow cases the fast path still handles.
    assert_eq!(one(&mut sess, "CAST(1.5::DECIMAL(3,1) AS DOUBLE)"), Value::F64(1.5));
    assert_eq!(one(&mut sess, "CAST(-0.25::DECIMAL(5,2) AS DOUBLE)"), Value::F64(-0.25));
    assert_eq!(one(&mut sess, "CAST(0::DECIMAL(3,1) AS DOUBLE)"), Value::F64(0.0));
    assert_eq!(one(&mut sess, "CAST(123::HUGEINT AS DOUBLE)"), Value::F64(123.0));
}

// --- VARCHAR <-> BLOB ---------------------------------------------------------

#[test]
fn the_blob_text_form_decodes_and_encodes_hex_escapes() {
    let mut sess = session_with_basic();
    // duckdb: '\x00ab'::BLOB is three bytes, and its VARCHAR form escapes the NUL again.
    // Both directions used to be a straight copy, so the escapes stayed as six literal
    // characters -- which also meant a BLOB written by this engine's own CSV/JSONL writers
    // (they emit \xHH) doubled up when read back through ::BLOB.
    assert_eq!(one(&mut sess, "CAST('\\x41\\x42'::BLOB AS VARCHAR)"), s("AB"));
    assert_eq!(one(&mut sess, "length('\\x41\\x42'::BLOB)"), Value::I64(2));
    assert_eq!(one(&mut sess, "CAST('A\\x00B'::BLOB AS VARCHAR)"), s("A\\x00B"));
    // Printable ASCII passes through untouched in both directions.
    assert_eq!(one(&mut sess, "CAST('abc'::BLOB AS VARCHAR)"), s("abc"));
    assert_eq!(one(&mut sess, "length('abc'::BLOB)"), Value::I64(3));
    // The round trip is the identity, which is what the straight copy used to buy and what
    // escaping only one direction would have broken.
    assert_eq!(
        one(
            &mut sess,
            "CAST(CAST('A\\x00B\\x5CD'::BLOB AS VARCHAR) AS BLOB) = 'A\\x00B\\x5CD'::BLOB"
        ),
        Value::Bool(true)
    );
    // A malformed escape is NULL (duckdb raises "Invalid hex escape code" instead).
    assert_eq!(one(&mut sess, "'\\xZZ'::BLOB"), Value::Null);
    assert_eq!(one(&mut sess, "'\\x4'::BLOB"), Value::Null);
    assert_eq!(one(&mut sess, "'a\\\\b'::BLOB"), Value::Null);
    assert_eq!(one(&mut sess, "CAST(''::BLOB AS VARCHAR)"), s(""));
}
