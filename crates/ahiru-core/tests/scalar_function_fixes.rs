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
