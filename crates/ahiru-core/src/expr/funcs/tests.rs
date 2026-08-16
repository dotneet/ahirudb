// The re-export of `civil_from_days` exists only with the `export` feature (its only consumers are
// `write::csv`/`write::jsonl`). Here it is taken directly from the implementation so the tests pass
// regardless of the feature.
use super::datetime::civil_from_days;
use super::numeric::{f_exp, f_ln, f_pow, f_sqrt};
use super::*;
use crate::error::code_of;
use crate::vector::Value;

// --- Construction helpers -----------------------------------------------

fn vs(vals: &[Option<&str>]) -> Vector {
    let mut v = Vector::new(Ty::Varchar);
    for x in vals {
        match x {
            Some(s) => v.push_value(&Value::Bytes(s.as_bytes().to_vec())),
            None => v.push_null(),
        }
    }
    v
}

fn vi(ty: Ty, vals: &[Option<i64>]) -> Vector {
    let mut v = Vector::new(ty);
    for x in vals {
        match x {
            Some(n) if ty.phys() == PhysType::I32 => v.push_value(&Value::I32(*n as i32)),
            Some(n) => v.push_value(&Value::I64(*n)),
            None => v.push_null(),
        }
    }
    v
}

fn vf(vals: &[Option<f64>]) -> Vector {
    let mut v = Vector::new(Ty::Double);
    for x in vals {
        match x {
            Some(n) => v.push_value(&Value::F64(*n)),
            None => v.push_null(),
        }
    }
    v
}

/// `resolve`s and then `call`s. The arguments are passed already in the resolved types.
fn run(name: &str, args: &[&Vector]) -> Result<Vector> {
    let tys: Vec<Ty> = args.iter().map(|a| a.ty()).collect();
    let (id, want, ret) = resolve(name, &tys)?;
    assert_eq!(want.len(), args.len());
    for (i, w) in want.iter().enumerate() {
        assert_eq!(*w, tys[i], "{name}: argument {i} has the wrong type");
    }
    call(id, ret, args)
}

fn str_at(v: &Vector, i: usize) -> Option<String> {
    if !v.is_valid(i) {
        return None;
    }
    Some(String::from_utf8(v.bytes().get(i).to_vec()).unwrap())
}

fn bytes_at(v: &Vector, i: usize) -> Option<Vec<u8>> {
    if !v.is_valid(i) {
        return None;
    }
    Some(v.bytes().get(i).to_vec())
}

fn int_at(v: &Vector, i: usize) -> Option<i64> {
    if !v.is_valid(i) {
        return None;
    }
    match v.data() {
        Data::I32(d) => Some(d[i] as i64),
        Data::I64(d) => Some(d[i]),
        _ => None,
    }
}

fn flt_at(v: &Vector, i: usize) -> Option<f64> {
    if v.is_valid(i) {
        Some(v.f64s()[i])
    } else {
        None
    }
}

fn i128_at(v: &Vector, i: usize) -> Option<i128> {
    if v.is_valid(i) {
        Some(v.i128s()[i])
    } else {
        None
    }
}

/// A shortcut for extracting one row's result from a one-string-argument (plus extras) call.
fn s1(name: &str, s: Option<&str>) -> Option<String> {
    str_at(&run(name, &[&vs(&[s])]).unwrap(), 0)
}

fn ts(lit: &str) -> Vector {
    vi(Ty::Timestamp, &[Some(parse_timestamp(lit.as_bytes()).unwrap())])
}

fn part(name: &str, p: &str, lit: &str) -> Option<i64> {
    int_at(&run(name, &[&vs(&[Some(p)]), &ts(lit)]).unwrap(), 0)
}

// --- Strings ------------------------------------------------------------

#[test]
fn string_basics() {
    assert_eq!(s1("upper", Some("aBc")).as_deref(), Some("ABC"));
    assert_eq!(s1("lower", Some("aBc")).as_deref(), Some("abc"));
    assert_eq!(s1("upper", Some("")).as_deref(), Some(""));
    assert_eq!(s1("upper", None), None);
    assert_eq!(s1("reverse", Some("abc")).as_deref(), Some("cba"));
    // reverse works in code points (duckdb: reverse('\u{3b1}\u{3b2}\u{3b3}') = '\u{3b3}\u{3b2}\u{3b1}').
    assert_eq!(
        s1("reverse", Some("\u{3b1}\u{3b2}\u{3b3}")).as_deref(),
        Some("\u{3b3}\u{3b2}\u{3b1}")
    );
    assert_eq!(s1("reverse", Some("")).as_deref(), Some(""));
    // duckdb: trim('  ab  ') = 'ab'; the default is spaces only (tabs remain).
    assert_eq!(s1("trim", Some("  ab  ")).as_deref(), Some("ab"));
    assert_eq!(s1("trim", Some("\tab\t")).as_deref(), Some("\tab\t"));
    assert_eq!(s1("ltrim", Some("  ab  ")).as_deref(), Some("ab  "));
    assert_eq!(s1("rtrim", Some("  ab  ")).as_deref(), Some("  ab"));
}

#[test]
fn unhex_uses_a_leading_zero_nibble_for_odd_lengths_and_errors_on_bad_digits() {
    // duckdb: unhex('ABC')/from_hex('ABC') = BLOB '\x0A\xBC'.
    let expected = Some(vec![0x0a, 0xbc]);
    assert_eq!(bytes_at(&run("unhex", &[&vs(&[Some("ABC")])]).unwrap(), 0), expected);
    assert_eq!(bytes_at(&run("from_hex", &[&vs(&[Some("ABC")])]).unwrap(), 0), expected);

    // NULL still propagates through the ordinary function path, while malformed
    // hexadecimal is a hard conversion error (DuckDB reports Invalid Input Error).
    let null = run("unhex", &[&vs(&[None])]).unwrap();
    assert_eq!(bytes_at(&null, 0), None);
    assert_eq!(code_of(run("unhex", &[&vs(&[Some("ABCG")])])), Some(Code::InvalidCast));
}

#[test]
fn trim_with_charset() {
    let f = |s: &str, c: &str, name: &str| {
        str_at(&run(name, &[&vs(&[Some(s)]), &vs(&[Some(c)])]).unwrap(), 0)
    };
    // duckdb: ltrim('xxabxx','x')='abxx', rtrim(..)='xxab', trim('xyabyx','xy')='ab'
    assert_eq!(f("xxabxx", "x", "ltrim").as_deref(), Some("abxx"));
    assert_eq!(f("xxabxx", "x", "rtrim").as_deref(), Some("xxab"));
    assert_eq!(f("xyabyx", "xy", "trim").as_deref(), Some("ab"));
    assert_eq!(f("abc", "", "trim").as_deref(), Some("abc"));
    assert_eq!(f("", "x", "trim").as_deref(), Some(""));
}

#[test]
fn length_and_strpos_are_codepoint_based() {
    // duckdb: length('\u{3b1}\u{3b2}\u{3b3}') = 3, strpos('\u{3b1}\u{3b2}\u{3b3}','\u{3b2}') = 2
    assert_eq!(
        int_at(&run("length", &[&vs(&[Some("\u{3b1}\u{3b2}\u{3b3}")])]).unwrap(), 0),
        Some(3)
    );
    assert_eq!(int_at(&run("length", &[&vs(&[Some("")])]).unwrap(), 0), Some(0));
    assert_eq!(int_at(&run("length", &[&vs(&[None])]).unwrap(), 0), None);
    let p =
        |s: &str, n: &str| int_at(&run("strpos", &[&vs(&[Some(s)]), &vs(&[Some(n)])]).unwrap(), 0);
    assert_eq!(p("\u{3b1}\u{3b2}\u{3b3}", "\u{3b2}"), Some(2));
    assert_eq!(p("abc", "b"), Some(2));
    assert_eq!(p("abc", "z"), Some(0));
    // duckdb: strpos('abc','') = 1
    assert_eq!(p("abc", ""), Some(1));
    assert_eq!(p("", "a"), Some(0));
}

#[test]
fn substr_edges_match_duckdb() {
    let f = |s: &str, a: i64, b: Option<i64>| {
        let sv = vs(&[Some(s)]);
        let av = vi(Ty::BigInt, &[Some(a)]);
        let out = match b {
            Some(b) => run("substr", &[&sv, &av, &vi(Ty::BigInt, &[Some(b)])]),
            None => run("substr", &[&sv, &av]),
        };
        str_at(&out.unwrap(), 0).unwrap()
    };
    // All of them are values measured with duckdb v1.4.
    assert_eq!(f("hello", 2, None), "ello");
    assert_eq!(f("hello", 2, Some(3)), "ell");
    assert_eq!(f("hello", -2, None), "lo");
    assert_eq!(f("hello", -2, Some(3)), "lo");
    assert_eq!(f("hello", 0, None), "hello");
    assert_eq!(f("hello", 0, Some(3)), "he");
    assert_eq!(f("hello", 10, None), "");
    assert_eq!(f("hello", 2, Some(0)), "");
    assert_eq!(f("hello", 2, Some(100)), "ello");
    assert_eq!(f("hello", -10, Some(3)), "");
    assert_eq!(f("hello", 2, Some(-1)), "h");
    assert_eq!(f("", 1, Some(3)), "");
    // duckdb: substr('\u{3b1}\u{3b2}\u{3b3}\u{3b4}\u{3b5}',2,2) = '\u{3b2}\u{3b3}' (code points)
    assert_eq!(f("\u{3b1}\u{3b2}\u{3b3}\u{3b4}\u{3b5}", 2, Some(2)), "\u{3b2}\u{3b3}");
    // A NULL argument gives NULL.
    let n = run("substr", &[&vs(&[None]), &vi(Ty::BigInt, &[Some(1)])]).unwrap();
    assert_eq!(str_at(&n, 0), None);
}

#[test]
fn replace_repeat_split_pad() {
    let s3 = |name: &str, a: &str, b: &str, c: &str| {
        str_at(&run(name, &[&vs(&[Some(a)]), &vs(&[Some(b)]), &vs(&[Some(c)])]).unwrap(), 0)
            .unwrap()
    };
    assert_eq!(s3("replace", "aaa", "a", "bb"), "bbbbbb");
    assert_eq!(s3("replace", "abc", "", "x"), "abc");
    assert_eq!(s3("replace", "", "a", "b"), "");

    let pad = |name: &str, s: &str, n: i64, p: &str| {
        str_at(
            &run(name, &[&vs(&[Some(s)]), &vi(Ty::BigInt, &[Some(n)]), &vs(&[Some(p)])]).unwrap(),
            0,
        )
        .unwrap()
    };
    assert_eq!(pad("lpad", "ab", 5, "xy"), "xyxab");
    assert_eq!(pad("rpad", "ab", 5, "xy"), "abxyx");
    assert_eq!(pad("lpad", "abcdef", 3, "x"), "abc");
    assert_eq!(pad("lpad", "ab", 0, "x"), "");
    // duckdb: lpad('\u{3b1}',3,'x') = 'xx\u{3b1}' (code points)
    assert_eq!(pad("lpad", "\u{3b1}", 3, "x"), "xx\u{3b1}");

    let sp = |s: &str, d: &str, k: i64| {
        str_at(
            &run("split_part", &[&vs(&[Some(s)]), &vs(&[Some(d)]), &vi(Ty::BigInt, &[Some(k)])])
                .unwrap(),
            0,
        )
        .unwrap()
    };
    assert_eq!(sp("a,b,c", ",", 2), "b");
    assert_eq!(sp("a,b,c", ",", 0), "");
    assert_eq!(sp("a,b,c", ",", 5), "");
    assert_eq!(sp("a,b,c", ",", -1), "c");
    assert_eq!(sp("a,b,c", ",", 1), "a");
    assert_eq!(sp("", ",", 1), "");

    let rp = |s: &str, k: i64| {
        str_at(&run("repeat", &[&vs(&[Some(s)]), &vi(Ty::BigInt, &[Some(k)])]).unwrap(), 0).unwrap()
    };
    assert_eq!(rp("ab", 3), "ababab");
    assert_eq!(rp("ab", 0), "");
    assert_eq!(rp("ab", -1), "");
}

#[test]
fn concat_ignores_nulls() {
    // duckdb: concat('a', NULL, 'b') = 'ab'
    let out = run("concat", &[&vs(&[Some("a")]), &vs(&[None]), &vs(&[Some("b")])]).unwrap();
    assert_eq!(str_at(&out, 0).as_deref(), Some("ab"));
    let out = run("concat", &[&vs(&[None]), &vs(&[None])]).unwrap();
    assert_eq!(str_at(&out, 0).as_deref(), Some(""));
}

#[test]
fn prefix_suffix_contains() {
    let f = |name: &str, s: &str, p: &str| {
        let v = run(name, &[&vs(&[Some(s)]), &vs(&[Some(p)])]).unwrap();
        v.is_valid(0).then(|| v.bools().get(0))
    };
    assert_eq!(f("starts_with", "abc", "ab"), Some(true));
    assert_eq!(f("starts_with", "abc", "b"), Some(false));
    // duckdb: starts_with('abc','') = true, contains('abc','') = true
    assert_eq!(f("starts_with", "abc", ""), Some(true));
    assert_eq!(f("ends_with", "abc", "bc"), Some(true));
    assert_eq!(f("ends_with", "abc", ""), Some(true));
    assert_eq!(f("contains", "abc", "b"), Some(true));
    assert_eq!(f("contains", "abc", ""), Some(true));
    assert_eq!(f("contains", "", "a"), Some(false));
    let n = run("contains", &[&vs(&[None]), &vs(&[Some("a")])]).unwrap();
    assert!(!n.is_valid(0));
}

// --- Numbers ------------------------------------------------------------

#[test]
fn numeric_integer_domain() {
    let f = |name: &str, x: i64| int_at(&run(name, &[&vi(Ty::BigInt, &[Some(x)])]).unwrap(), 0);
    assert_eq!(f("abs", -3), Some(3));
    assert_eq!(f("abs", 0), Some(0));
    assert_eq!(f("sign", -5), Some(-1));
    assert_eq!(f("sign", 0), Some(0));
    assert_eq!(f("ceil", 3), Some(3));
    assert_eq!(f("floor", 3), Some(3));
    assert_eq!(f("trunc", 3), Some(3));
    assert_eq!(f("round", 3), Some(3));
    assert_eq!(int_at(&run("abs", &[&vi(Ty::BigInt, &[None])]).unwrap(), 0), None);
    // duckdb: round(12345,-2)=12300, round(-12345,-2)=-12300, round(15,-1)=20, round(25,-1)=30
    let r = |x: i64, d: i64| {
        int_at(
            &run("round", &[&vi(Ty::BigInt, &[Some(x)]), &vi(Ty::BigInt, &[Some(d)])]).unwrap(),
            0,
        )
    };
    assert_eq!(r(12345, -2), Some(12300));
    assert_eq!(r(-12345, -2), Some(-12300));
    assert_eq!(r(15, -1), Some(20));
    assert_eq!(r(25, -1), Some(30));
    assert_eq!(r(12345, 2), Some(12345));
    // duckdb: mod(-7,3) = -1, mod(7,-3) = 1; division by zero gives NULL.
    let m = |x: i64, y: i64| {
        int_at(&run("mod", &[&vi(Ty::BigInt, &[Some(x)]), &vi(Ty::BigInt, &[Some(y)])]).unwrap(), 0)
    };
    assert_eq!(m(-7, 3), Some(-1));
    assert_eq!(m(7, -3), Some(1));
    assert_eq!(m(7, 0), None);
}

// `factorial`: HUGEINT (I128) output, the postfix `!` desugaring site
// (`sql::parser::Parser::primary`/`cast_postfix`). Every value below is
// cross-checked against a real `duckdb` CLI.
#[test]
fn factorial_matches_duckdb() {
    let f = |x: i64| i128_at(&run("factorial", &[&vi(Ty::BigInt, &[Some(x)])]).unwrap(), 0);
    assert_eq!(f(0), Some(1));
    assert_eq!(f(1), Some(1));
    assert_eq!(f(20), Some(2_432_902_008_176_640_000));
    // duckdb: factorial(33) = 8683317618811886495518194401280000000, the
    // largest factorial that fits in i128 (i128::MAX ~= 1.7e38, 33! ~=
    // 8.68e36, 34! ~= 2.95e38 overflows -- see the next test).
    assert_eq!(f(33), Some(8_683_317_618_811_886_495_518_194_401_280_000_000));
    // duckdb: factorial(-1) = 1 -- negative n isn't an "undefined" case,
    // it's just defined to be 1.
    assert_eq!(f(-1), Some(1));
    assert_eq!(f(-100), Some(1));
    assert_eq!(i128_at(&run("factorial", &[&vi(Ty::BigInt, &[None])]).unwrap(), 0), None);
}

#[test]
fn factorial_overflow_is_a_hard_error_not_a_silent_wrap() {
    // duckdb: factorial(34) -> "Out of Range Error". This engine's usual
    // convention is that integer arithmetic overflow wraps silently (see
    // docs/sql/limitations.md's rounding-conventions note) -- but `SUM` is
    // the existing exception (`exec::agg`'s `Code::ValueOutOfRange`), and
    // `factorial` joins it here rather than silently wrapping to a
    // nonsense negative HUGEINT.
    assert_eq!(
        code_of(run("factorial", &[&vi(Ty::BigInt, &[Some(34)])])),
        Some(Code::ValueOutOfRange)
    );
    // A huge `n` must fail fast (33 multiplications at most before
    // overflowing), not loop for a very long time.
    assert_eq!(
        code_of(run("factorial", &[&vi(Ty::BigInt, &[Some(i64::MAX)])])),
        Some(Code::ValueOutOfRange)
    );
}

#[test]
fn factorial_resolve_accepts_only_integer_input() {
    assert_eq!(resolve("factorial", &[Ty::Int]).unwrap().1, vec![Ty::BigInt]);
    assert_eq!(resolve("factorial", &[Ty::Int]).unwrap().2, Ty::HugeInt);
    assert_eq!(resolve("factorial", &[Ty::Null]).unwrap().2, Ty::HugeInt);
    // duckdb rejects a DOUBLE argument to `factorial`/`!` as a type error
    // (`duckdb -c "select factorial(4.0)"` / `select 4.5!`), and VARCHAR
    // was never a numeric type to begin with.
    assert_eq!(code_of(resolve("factorial", &[Ty::Double])), Some(Code::TypeMismatch));
    assert_eq!(code_of(resolve("factorial", &[Ty::Varchar])), Some(Code::TypeMismatch));
    assert_eq!(code_of(resolve("factorial", &[])), Some(Code::WrongArgCount));
    assert_eq!(code_of(resolve("factorial", &[Ty::Int, Ty::Int])), Some(Code::WrongArgCount));
}

#[test]
fn numeric_float_domain() {
    let f = |name: &str, x: f64| flt_at(&run(name, &[&vf(&[Some(x)])]).unwrap(), 0);
    // duckdb: round(2.5)=3, round(3.5)=4, round(-2.5)=-3 (away from zero)
    assert_eq!(f("round", 2.5), Some(3.0));
    assert_eq!(f("round", 3.5), Some(4.0));
    assert_eq!(f("round", -2.5), Some(-3.0));
    assert_eq!(f("round", 0.5), Some(1.0));
    assert_eq!(f("round", 1.5), Some(2.0));
    assert_eq!(f("abs", -2.5), Some(2.5));
    assert_eq!(f("sign", -5.0), Some(-1.0));
    assert_eq!(f("ceil", 3.2), Some(4.0));
    assert_eq!(f("ceil", -3.2), Some(-3.0));
    assert_eq!(f("floor", 3.8), Some(3.0));
    assert_eq!(f("floor", -3.2), Some(-4.0));
    assert_eq!(f("trunc", -2.7), Some(-2.0));
    assert_eq!(f("sqrt", 4.0), Some(2.0));
    assert_eq!(f("sqrt", 0.0), Some(0.0));
    // duckdb errors, but here it gives NULL (it does not fail mid-expression).
    assert_eq!(f("sqrt", -1.0), None);
    assert_eq!(f("ln", 0.0), None);
    assert_eq!(f("log", 0.0), None);
    assert!((f("log", 100.0).unwrap() - 2.0).abs() < 1e-15);
    assert_eq!(f("exp", 0.0), Some(1.0));
    assert_eq!(f("abs", 0.0), Some(0.0));
    assert_eq!(flt_at(&run("abs", &[&vf(&[None])]).unwrap(), 0), None);

    // duckdb: round(2.675,2)=2.68, round(1.005,2)=1.0, round(2.345,2)=2.35
    let r = |x: f64, d: i64| {
        flt_at(&run("round", &[&vf(&[Some(x)]), &vi(Ty::BigInt, &[Some(d)])]).unwrap(), 0)
    };
    assert_eq!(r(2.675, 2), Some(2.68));
    assert_eq!(r(1.005, 2), Some(1.0));
    assert_eq!(r(2.345, 2), Some(2.35));
    assert_eq!(r(1234.5678, -2), Some(1200.0));

    let p = |x: f64, y: f64| flt_at(&run("pow", &[&vf(&[Some(x)]), &vf(&[Some(y)])]).unwrap(), 0);
    // Integer exponents use repeated squaring and match exactly.
    assert_eq!(p(2.0, 3.0), Some(8.0));
    assert_eq!(p(2.0, -2.0), Some(0.25));
    assert_eq!(p(5.0, 0.0), Some(1.0));
    // Non-integer exponents go through exp(y ln x) and differ by a few ulps.
    assert!((p(9.0, 0.5).unwrap() - 3.0).abs() < 1e-14);
}

#[test]
fn transcendental_accuracy() {
    // How closely the in-house exp / ln / sqrt agree in double precision (relative error within 1e-15).
    let close = |a: f64, b: f64| ((a - b) / b).abs() < 1e-15;
    assert!(close(f_exp(1.0), core::f64::consts::E));
    assert!(close(f_exp(10.0), 22026.465794806718));
    assert!(close(f_exp(-5.0), 0.006737946999085467));
    assert!(close(f_ln(core::f64::consts::E), 1.0));
    assert!(close(f_ln(10.0), core::f64::consts::LN_10));
    assert!(close(f_ln(1e-300), -690.7755278982137));
    assert!(close(f_sqrt(2.0), core::f64::consts::SQRT_2));
    assert!(close(f_sqrt(1e300), 1e150));
    assert_eq!(f_sqrt(0.0), 0.0);
    assert_eq!(f_sqrt(4.0), 2.0);
    // Subnormals (exponent 0) are handled by scaling by 2^64 too.
    assert!(f_ln(1e-310) < -713.0 && f_ln(1e-310) > -714.0);
    assert!((f_pow(2.0, 0.5) - core::f64::consts::SQRT_2).abs() < 1e-15);
}

// --- The NULL family ----------------------------------------------------

#[test]
fn null_handling_functions() {
    let n = || vi(Ty::Int, &[None]);
    let k = |x: i64| vi(Ty::Int, &[Some(x)]);
    // duckdb: coalesce(NULL,NULL,3)=3, ifnull(NULL,5)=5
    assert_eq!(int_at(&run("coalesce", &[&n(), &n(), &k(3)]).unwrap(), 0), Some(3));
    assert_eq!(int_at(&run("coalesce", &[&n(), &n()]).unwrap(), 0), None);
    assert_eq!(int_at(&run("ifnull", &[&n(), &k(5)]).unwrap(), 0), Some(5));
    assert_eq!(int_at(&run("ifnull", &[&k(1), &k(5)]).unwrap(), 0), Some(1));
    // duckdb: nullif(1,1)=NULL, nullif(1,2)=1, nullif(1,NULL)=1, nullif(NULL,1)=NULL
    assert_eq!(int_at(&run("nullif", &[&k(1), &k(1)]).unwrap(), 0), None);
    assert_eq!(int_at(&run("nullif", &[&k(1), &k(2)]).unwrap(), 0), Some(1));
    assert_eq!(int_at(&run("nullif", &[&k(1), &n()]).unwrap(), 0), Some(1));
    assert_eq!(int_at(&run("nullif", &[&n(), &k(1)]).unwrap(), 0), None);
}

#[test]
fn greatest_least_skip_nulls() {
    let n = || vi(Ty::Int, &[None]);
    let k = |x: i64| vi(Ty::Int, &[Some(x)]);
    // duckdb: greatest(1,NULL,3)=3, least(NULL,2,1)=1, greatest(NULL,NULL)=NULL
    assert_eq!(int_at(&run("greatest", &[&k(1), &n(), &k(3)]).unwrap(), 0), Some(3));
    assert_eq!(int_at(&run("least", &[&n(), &k(2), &k(1)]).unwrap(), 0), Some(1));
    assert_eq!(int_at(&run("greatest", &[&n(), &n()]).unwrap(), 0), None);
    assert_eq!(int_at(&run("least", &[&k(5), &n()]).unwrap(), 0), Some(5));
    assert_eq!(int_at(&run("greatest", &[&n(), &k(7)]).unwrap(), 0), Some(7));
    // NaN is greater than every finite value (same total order as MIN/MAX / DuckDB).
    let nan = vf(&[Some(f64::NAN)]);
    let one = vf(&[Some(1.0)]);
    let g1 = run("greatest", &[&one, &nan]).unwrap();
    let g2 = run("greatest", &[&nan, &one]).unwrap();
    assert!(g1.f64s()[0].is_nan(), "greatest(1, nan)");
    assert!(g2.f64s()[0].is_nan(), "greatest(nan, 1)");
    let l1 = run("least", &[&one, &nan]).unwrap();
    let l2 = run("least", &[&nan, &one]).unwrap();
    assert_eq!(l1.f64s()[0], 1.0);
    assert_eq!(l2.f64s()[0], 1.0);
    // duckdb: greatest('a','b') = 'b'
    let g = run("greatest", &[&vs(&[Some("a")]), &vs(&[Some("b")])]).unwrap();
    assert_eq!(str_at(&g, 0).as_deref(), Some("b"));
}

// --- Date and time ------------------------------------------------------

#[test]
fn civil_roundtrip() {
    for &d in &[-719_468i64, -1, 0, 1, 19_000, 100_000, -50_000] {
        let (y, m, dd) = civil_from_days(d);
        assert_eq!(days_from_civil(y, m, dd), d);
    }
}

#[test]
fn date_trunc_matches_duckdb() {
    let f = |p: &str, lit: &str| {
        let v = run("date_trunc", &[&vs(&[Some(p)]), &ts(lit)]).unwrap();
        let mut o = Vec::new();
        fmt_timestamp(int_at(&v, 0).unwrap(), &mut o);
        String::from_utf8(o).unwrap()
    };
    // Before the epoch (negative microseconds). The boundary where truncation would be off by one unit.
    assert_eq!(f("year", "1969-07-20 20:17:40"), "1969-01-01 00:00:00");
    assert_eq!(f("quarter", "1969-07-20 20:17:40"), "1969-07-01 00:00:00");
    assert_eq!(f("month", "1969-07-20 20:17:40"), "1969-07-01 00:00:00");
    assert_eq!(f("week", "1969-07-20 20:17:40"), "1969-07-14 00:00:00");
    assert_eq!(f("day", "1969-07-20 20:17:40"), "1969-07-20 00:00:00");
    assert_eq!(f("hour", "1969-07-20 20:17:40"), "1969-07-20 20:00:00");
    assert_eq!(f("minute", "1969-07-20 20:17:40"), "1969-07-20 20:17:00");
    assert_eq!(f("second", "1969-07-20 20:17:40.123456"), "1969-07-20 20:17:40");
    // Leap days and year boundaries.
    assert_eq!(f("week", "2024-02-29 12:34:56"), "2024-02-26 00:00:00");
    assert_eq!(f("YEAR", "2024-12-31 23:59:59"), "2024-01-01 00:00:00");
    assert_eq!(f("month", "2024-01-01 00:00:00"), "2024-01-01 00:00:00");
}

#[test]
fn date_part_matches_duckdb() {
    assert_eq!(part("date_part", "dow", "2024-02-29 12:00:00"), Some(4));
    assert_eq!(part("date_part", "doy", "2024-02-29 12:00:00"), Some(60));
    assert_eq!(part("date_part", "quarter", "2024-02-29 00:00:00"), Some(1));
    assert_eq!(part("date_part", "epoch", "1969-07-20 20:17:40"), Some(-14_182_940));
    assert_eq!(part("date_part", "dow", "1969-07-20 00:00:00"), Some(0));
    // ISO week numbers. Checks the boundary where a week spanning years becomes week 52/53 of the previous year.
    assert_eq!(part("date_part", "week", "2024-01-01 00:00:00"), Some(1));
    assert_eq!(part("date_part", "week", "2023-01-01 00:00:00"), Some(52));
    assert_eq!(part("date_part", "week", "2021-01-03 00:00:00"), Some(53));
    assert_eq!(part("date_part", "week", "2024-12-30 00:00:00"), Some(1));
    assert_eq!(part("date_part", "week", "1969-07-20 00:00:00"), Some(29));
    assert_eq!(part("date_part", "minute", "1969-07-20 20:17:40"), Some(17));
    assert_eq!(part("date_part", "second", "2024-01-01 10:20:30.5"), Some(30));
    // Uppercase, plurals, and extract's aliases are accepted too.
    assert_eq!(part("extract", "HOUR", "1969-07-20 20:17:40"), Some(20));
    assert_eq!(part("date_part", "years", "1900-01-01 00:00:00"), Some(1900));
    // The shorthands.
    let sh = |name: &str, lit: &str| int_at(&run(name, &[&ts(lit)]).unwrap(), 0);
    assert_eq!(sh("year", "1900-01-01 00:00:00"), Some(1900));
    assert_eq!(sh("month", "2024-02-29 00:00:00"), Some(2));
    assert_eq!(sh("day", "2024-02-29 00:00:00"), Some(29));
    assert_eq!(sh("hour", "1969-07-20 20:17:40"), Some(20));
    assert_eq!(sh("minute", "1969-07-20 20:17:40"), Some(17));
    assert_eq!(sh("second", "2024-01-01 10:20:30.5"), Some(30));
    // NULL propagation.
    let n = run("year", &[&vi(Ty::Timestamp, &[None])]).unwrap();
    assert_eq!(int_at(&n, 0), None);
}

#[test]
fn date_diff_and_add() {
    let d = |p: &str, a: &str, b: &str| {
        int_at(&run("date_diff", &[&vs(&[Some(p)]), &ts(a), &ts(b)]).unwrap(), 0)
    };
    // All are values measured with duckdb (how many boundaries were crossed).
    assert_eq!(d("day", "2024-01-01 00:00:00", "2024-03-01 00:00:00"), Some(60));
    assert_eq!(d("month", "2024-01-31 00:00:00", "2024-03-01 00:00:00"), Some(2));
    assert_eq!(d("year", "2020-12-31 00:00:00", "2021-01-01 00:00:00"), Some(1));
    assert_eq!(d("hour", "2024-01-01 01:59:00", "2024-01-01 03:00:00"), Some(2));
    assert_eq!(d("week", "2024-01-01 00:00:00", "2024-01-15 00:00:00"), Some(2));
    assert_eq!(d("week", "2024-01-02 00:00:00", "2024-01-08 00:00:00"), Some(0));
    assert_eq!(d("quarter", "2024-01-01 00:00:00", "2024-07-01 00:00:00"), Some(2));
    assert_eq!(d("day", "2024-01-01 23:00:00", "2024-01-02 01:00:00"), Some(1));
    assert_eq!(d("second", "2024-01-01 00:00:00.9", "2024-01-01 00:00:01.1"), Some(1));
    // Crossing the epoch.
    assert_eq!(d("day", "1969-12-31 00:00:00", "1970-01-01 00:00:00"), Some(1));

    let a = |p: &str, k: i64, lit: &str| {
        let v = run("date_add", &[&vs(&[Some(p)]), &vi(Ty::BigInt, &[Some(k)]), &ts(lit)]).unwrap();
        let mut o = Vec::new();
        fmt_timestamp(int_at(&v, 0).unwrap(), &mut o);
        String::from_utf8(o).unwrap()
    };
    // As with duckdb's INTERVAL addition, the day is clamped to the month's end.
    assert_eq!(a("month", 1, "2024-01-31 00:00:00"), "2024-02-29 00:00:00");
    assert_eq!(a("month", -1, "2024-03-31 00:00:00"), "2024-02-29 00:00:00");
    assert_eq!(a("year", 1, "2024-02-29 00:00:00"), "2025-02-28 00:00:00");
    assert_eq!(a("day", 1, "1969-12-31 23:00:00"), "1970-01-01 23:00:00");
    assert_eq!(a("hour", -1, "1970-01-01 00:00:00"), "1969-12-31 23:00:00");
}

#[test]
fn add_interval_to_ts_matches_duckdb() {
    let a = |lit: &str, months: i32, days: i32, micros: i64| {
        let us = int_at(&ts(lit), 0).unwrap();
        let v = add_interval_to_ts(us, months, days, micros).unwrap();
        let mut o = Vec::new();
        fmt_timestamp(v, &mut o);
        String::from_utf8(o).unwrap()
    };
    // All are values measured with duckdb.
    assert_eq!(a("2024-01-01 00:00:00", 0, 3, 0), "2024-01-04 00:00:00");
    assert_eq!(a("2024-01-01 00:00:00", 0, 0, 3_600_000_000), "2024-01-01 01:00:00");
    // Month-end clamping: January has 31 days and February only 29, so it does not carry and stops at the 29th.
    assert_eq!(a("2024-01-31 00:00:00", 1, 0, 0), "2024-02-29 00:00:00");
    assert_eq!(a("2024-01-31 00:00:00", 2, 0, 0), "2024-03-31 00:00:00");
    assert_eq!(a("2024-01-31 00:00:00", 1, 1, 0), "2024-03-01 00:00:00");
    assert_eq!(a("2024-01-31 00:00:00", 1, -1, 0), "2024-02-28 00:00:00");
    assert_eq!(a("2024-03-31 00:00:00", -1, 0, 0), "2024-02-29 00:00:00");
    assert_eq!(a("2024-01-31 00:00:00", -1, -1, 0), "2023-12-30 00:00:00");
    assert_eq!(a("1970-01-01 00:00:00", 0, -1, 0), "1969-12-31 00:00:00");
    // After adding months the time is preserved, and adding microseconds crosses into the next day.
    assert_eq!(a("2024-01-31 23:30:00", 1, 0, 45 * 60 * 1_000_000), "2024-03-01 00:15:00");
}

#[test]
fn last_day_and_strftime() {
    let l = int_at(&run("last_day", &[&ts("2024-02-10 00:00:00")]).unwrap(), 0).unwrap();
    let mut o = Vec::new();
    fmt_date(l, &mut o);
    assert_eq!(String::from_utf8(o).unwrap(), "2024-02-29");
    let f =
        run("strftime", &[&ts("2024-02-29 01:02:03"), &vs(&[Some("%Y-%m-%d %H:%M:%S")])]).unwrap();
    assert_eq!(str_at(&f, 0).as_deref(), Some("2024-02-29 01:02:03"));
    let f = run("strftime", &[&ts("2024-02-29 01:02:03"), &vs(&[Some("%Y/%%/%q")])]).unwrap();
    assert_eq!(str_at(&f, 0).as_deref(), Some("2024/%/%q"));
}

#[test]
fn to_date_and_to_timestamp() {
    let d = |s: &str| int_at(&run("to_date", &[&vs(&[Some(s)])]).unwrap(), 0);
    assert_eq!(d("1970-01-01"), Some(0));
    assert_eq!(d("2024-1-5"), Some(days_from_civil(2024, 1, 5)));
    assert_eq!(d("nope"), None);
    assert_eq!(d("2024-13-01"), None);
    let t = |s: &str| int_at(&run("to_timestamp", &[&vs(&[Some(s)])]).unwrap(), 0);
    assert_eq!(t("1970-01-01 00:00:01"), Some(1_000_000));
    assert_eq!(t("1970-01-01"), Some(0));
    assert_eq!(t("bad"), None);
}

// --- Parsers / formatters (used by kernels' casts too) ------------------

#[test]
fn parse_and_format_roundtrip() {
    let fmt = |us: i64| {
        let mut o = Vec::new();
        fmt_timestamp(us, &mut o);
        String::from_utf8(o).unwrap()
    };
    for lit in [
        "2024-01-01 00:00:00",
        "1969-07-20 20:17:40",
        "0001-01-01 00:00:00",
        "2024-02-29 23:59:59.123456",
        "1900-12-31 12:00:00.1",
    ] {
        let us = parse_timestamp(lit.as_bytes()).unwrap();
        assert_eq!(fmt(us), lit, "round trip failed: {lit}");
    }
    // Matches duckdb's VARCHAR representation.
    assert_eq!(fmt(parse_timestamp(b"2024-01-01").unwrap()), "2024-01-01 00:00:00");
    assert_eq!(fmt(parse_timestamp(b"2024-01-05T10:20:30").unwrap()), "2024-01-05 10:20:30");
    assert_eq!(fmt(parse_timestamp(b" 2024-01-05 ").unwrap()), "2024-01-05 00:00:00");
}

#[test]
fn parse_rejects_bad_input() {
    assert_eq!(parse_date(b"2024-13-01"), None);
    assert_eq!(parse_date(b"2024-02-30"), None);
    assert_eq!(parse_date(b"2024-00-01"), None);
    assert_eq!(parse_date(b"2024-01-00"), None);
    assert_eq!(parse_date(b""), None);
    assert_eq!(parse_date(b"abc"), None);
    assert_eq!(parse_date(b"2024-01-01x"), None);
    // Leap years pass.
    assert!(parse_date(b"2024-02-29").is_some());
    assert_eq!(parse_date(b"2023-02-29"), None);
    assert_eq!(parse_timestamp(b"2024-01-01 25:00:00"), None);
    assert_eq!(parse_timestamp(b"2024-01-01 10:61:00"), None);
    assert_eq!(parse_timestamp(b"2024-01-01 10"), None);
    assert_eq!(parse_time(b"01:02:03"), Some(3_723_000_000));
    assert_eq!(parse_time(b"01:02"), Some(3_720_000_000));
    assert_eq!(parse_time(b"01:02:03.5"), Some(3_723_500_000));
    assert_eq!(parse_time(b"1:2:3"), Some(3_723_000_000));
    assert_eq!(parse_time(b"24:00:00"), Some(86_400_000_000));
    assert_eq!(parse_time(b"24:01:00"), None);
    assert_eq!(parse_time(b"24:00:00.000001"), None);
    assert_eq!(parse_time(b"xx"), None);
    let mut o = Vec::new();
    fmt_time(3_723_500_000, &mut o);
    assert_eq!(String::from_utf8(o).unwrap(), "01:02:03.5");
}

// --- String <-> date-time casts (the kernels-side implementation is verified from here) ---

#[test]
fn string_to_temporal_cast() {
    let src =
        vs(&[Some("2024-02-29"), Some("not a date"), Some("2024-13-01"), Some("2024-02-30"), None]);
    let out = kernels::cast(Ty::Varchar, Ty::Date, &src).unwrap();
    assert_eq!(int_at(&out, 0), Some(days_from_civil(2024, 2, 29)));
    // Unreadable or out of range is not an error; just that row becomes NULL.
    assert_eq!(int_at(&out, 1), None);
    assert_eq!(int_at(&out, 2), None);
    assert_eq!(int_at(&out, 3), None);
    assert_eq!(int_at(&out, 4), None);

    let src = vs(&[Some("2024-01-05 10:20:30"), Some("2024-01-05"), Some("x")]);
    let out = kernels::cast(Ty::Varchar, Ty::Timestamp, &src).unwrap();
    assert_eq!(int_at(&out, 0), Some(parse_timestamp(b"2024-01-05 10:20:30").unwrap()));
    // A date-only string is read as midnight.
    assert_eq!(int_at(&out, 1), Some(days_from_civil(2024, 1, 5) * US_PER_DAY));
    assert_eq!(int_at(&out, 2), None);

    let src = vs(&[Some("01:02:03.5"), Some("nope")]);
    let out = kernels::cast(Ty::Varchar, Ty::Time, &src).unwrap();
    assert_eq!(int_at(&out, 0), Some(3_723_500_000));
    assert_eq!(int_at(&out, 1), None);
}

#[test]
fn temporal_to_string_cast() {
    // Made to match duckdb's VARCHAR representation.
    let d = vi(Ty::Date, &[Some(days_from_civil(2024, 1, 1)), Some(0), None]);
    let out = kernels::cast(Ty::Date, Ty::Varchar, &d).unwrap();
    assert_eq!(str_at(&out, 0).as_deref(), Some("2024-01-01"));
    assert_eq!(str_at(&out, 1).as_deref(), Some("1970-01-01"));
    assert_eq!(str_at(&out, 2), None);

    let t = vi(
        Ty::Timestamp,
        &[
            Some(parse_timestamp(b"1969-07-20 20:17:40").unwrap()),
            Some(parse_timestamp(b"2024-01-01").unwrap()),
            Some(parse_timestamp(b"2024-01-01 00:00:00.123456").unwrap()),
            Some(parse_timestamp(b"2024-01-01 00:00:00.1").unwrap()),
        ],
    );
    let out = kernels::cast(Ty::Timestamp, Ty::Varchar, &t).unwrap();
    assert_eq!(str_at(&out, 0).as_deref(), Some("1969-07-20 20:17:40"));
    assert_eq!(str_at(&out, 1).as_deref(), Some("2024-01-01 00:00:00"));
    assert_eq!(str_at(&out, 2).as_deref(), Some("2024-01-01 00:00:00.123456"));
    assert_eq!(str_at(&out, 3).as_deref(), Some("2024-01-01 00:00:00.1"));

    let tm = vi(Ty::Time, &[Some(3_723_000_000), Some(3_723_500_000)]);
    let out = kernels::cast(Ty::Time, Ty::Varchar, &tm).unwrap();
    assert_eq!(str_at(&out, 0).as_deref(), Some("01:02:03"));
    assert_eq!(str_at(&out, 1).as_deref(), Some("01:02:03.5"));
}

#[test]
fn timestamp_string_roundtrip_through_cast() {
    let lits = ["2024-02-29 23:59:59.123456", "1969-07-20 20:17:40", "0001-01-01 00:00:00"];
    let v =
        vi(Ty::Timestamp, &lits.iter().map(|l| parse_timestamp(l.as_bytes())).collect::<Vec<_>>());
    let s = kernels::cast(Ty::Timestamp, Ty::Varchar, &v).unwrap();
    let back = kernels::cast(Ty::Varchar, Ty::Timestamp, &s).unwrap();
    for (i, lit) in lits.iter().enumerate() {
        assert_eq!(int_at(&back, i), int_at(&v, i), "round trip failed for {lit}");
    }
}

// --- Handling vectors and constants -------------------------------------

#[test]
fn constant_folding_keeps_length_one() {
    // If every argument has length 1 the result has length 1 too (it is not collapsed to stride 0).
    let out = run("upper", &[&vs(&[Some("ab")])]).unwrap();
    assert_eq!(out.len(), 1);
    let out = run("substr", &[&vs(&[Some("hello")]), &vi(Ty::BigInt, &[Some(2)])]).unwrap();
    assert_eq!(out.len(), 1);
    let out = run("coalesce", &[&vi(Ty::Int, &[None]), &vi(Ty::Int, &[Some(1)])]).unwrap();
    assert_eq!(out.len(), 1);
}

#[test]
fn constant_operand_broadcasts() {
    // Mixing a length-3 column with a length-1 constant gives 3 rows.
    let col = vs(&[Some("aa"), None, Some("cc")]);
    let k = vi(Ty::BigInt, &[Some(2)]);
    let out = run("substr", &[&col, &k]).unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(str_at(&out, 0).as_deref(), Some("a"));
    assert_eq!(str_at(&out, 1), None);
    assert_eq!(str_at(&out, 2).as_deref(), Some("c"));
    // An empty vector gives an empty result.
    let empty = Vector::new(Ty::Varchar);
    let out = run("upper", &[&empty]).unwrap();
    assert_eq!(out.len(), 0);
}

#[test]
fn multi_row_dates() {
    let col = vi(
        Ty::Timestamp,
        &[
            Some(parse_timestamp(b"2024-02-29 00:00:00").unwrap()),
            None,
            Some(parse_timestamp(b"1969-07-20 20:17:40").unwrap()),
        ],
    );
    let out = run("year", &[&col]).unwrap();
    assert_eq!(int_at(&out, 0), Some(2024));
    assert_eq!(int_at(&out, 1), None);
    assert_eq!(int_at(&out, 2), Some(1969));
}

// --- JSON -----------------------------------------------------------

fn vj(vals: &[Option<&str>]) -> Vector {
    let mut v = Vector::new(Ty::Json);
    for x in vals {
        match x {
            Some(s) => v.push_value(&Value::Bytes(s.as_bytes().to_vec())),
            None => v.push_null(),
        }
    }
    v
}

#[test]
fn json_extract_follows_dollar_paths_like_duckdb() {
    // duckdb: json_extract('{"a":{"b":[1,2,3]}}', '$.a.b[1]') -> 2
    let doc = vj(&[Some(r#"{"a":{"b":[1,2,3]}}"#)]);
    let path = vs(&[Some("$.a.b[1]")]);
    let out = run("json_extract", &[&doc, &path]).unwrap();
    assert_eq!(out.ty(), Ty::Json);
    assert_eq!(str_at(&out, 0).as_deref(), Some("2"));

    // duckdb: json_extract('{"a":1}', '$.b') -> NULL (no such key)
    let doc = vj(&[Some(r#"{"a":1}"#)]);
    let path = vs(&[Some("$.b")]);
    assert_eq!(str_at(&run("json_extract", &[&doc, &path]).unwrap(), 0), None);
}

#[test]
fn json_extract_operator_desugars_at_parse_time_not_tested_here() {
    // `->`/`->>` are merely expanded by sql::parser into json_extract(_string); this function itself
    // has no operator-specific logic (the parser-side test `sql::parser::tests` checks it together
    // with precedence).
    let doc = vj(&[Some(r#"{"a":1}"#)]);
    let path = vs(&[Some("$.a")]);
    assert_eq!(str_at(&run("json_extract", &[&doc, &path]).unwrap(), 0).as_deref(), Some("1"));
}

#[test]
fn json_extract_string_unquotes_and_nulls_json_null() {
    // duckdb: json_extract_string('{"a":"hello"}', '$.a') -> 'hello'
    let doc = vj(&[Some(r#"{"a":"hello"}"#)]);
    let path = vs(&[Some("$.a")]);
    let out = run("json_extract_string", &[&doc, &path]).unwrap();
    assert_eq!(out.ty(), Ty::Varchar);
    assert_eq!(str_at(&out, 0).as_deref(), Some("hello"));

    // duckdb: json_extract_string('{"a":1}', '$.a') -> '1'
    let doc = vj(&[Some(r#"{"a":1}"#)]);
    assert_eq!(
        str_at(&run("json_extract_string", &[&doc, &path]).unwrap(), 0).as_deref(),
        Some("1")
    );

    // duckdb: json_extract_string('{"a":null}', '$.a') -> NULL
    let doc = vj(&[Some(r#"{"a":null}"#)]);
    assert_eq!(str_at(&run("json_extract_string", &[&doc, &path]).unwrap(), 0), None);
}

#[test]
fn json_type_matches_duckdb_strings() {
    // duckdb: json_type('{"a":1}')='OBJECT', json_type('[1,2]')='ARRAY',
    // json_type('"x"')='VARCHAR', json_type('true')='BOOLEAN',
    // json_type('null')='NULL', json_type('1.5')='DOUBLE'
    let cases: &[(&str, &str)] = &[
        (r#"{"a":1}"#, "OBJECT"),
        ("[1,2]", "ARRAY"),
        ("\"x\"", "VARCHAR"),
        ("true", "BOOLEAN"),
        ("null", "NULL"),
        ("1.5", "DOUBLE"),
        ("1", "BIGINT"),
        ("-1", "BIGINT"),
    ];
    for (doc, want) in cases {
        let d = vj(&[Some(doc)]);
        assert_eq!(str_at(&run("json_type", &[&d]).unwrap(), 0).as_deref(), Some(*want));
    }
    // The two-argument form with a path.
    let doc = vj(&[Some(r#"{"a":{"b":1}}"#)]);
    let path = vs(&[Some("$.a")]);
    assert_eq!(str_at(&run("json_type", &[&doc, &path]).unwrap(), 0).as_deref(), Some("OBJECT"));
}

#[test]
fn json_array_length_is_zero_for_non_arrays_and_null_for_missing_path() {
    // duckdb: json_array_length('[1,2,3]')=3, json_array_length('{"a":1}')=0
    let a = vj(&[Some("[1,2,3]")]);
    assert_eq!(int_at(&run("json_array_length", &[&a]).unwrap(), 0), Some(3));
    let o = vj(&[Some(r#"{"a":1}"#)]);
    assert_eq!(int_at(&run("json_array_length", &[&o]).unwrap(), 0), Some(0));
    // duckdb: json_array_length('{"a":1}', '$.b') -> NULL (no such path)
    let path = vs(&[Some("$.b")]);
    assert_eq!(int_at(&run("json_array_length", &[&o, &path]).unwrap(), 0), None);
}

#[test]
fn list_extract_is_one_based_with_negative_from_end() {
    // duckdb: list_extract([10,20,30], 1) -> 10, list_extract(.., -1) -> 30,
    // list_extract(.., 0) -> NULL, list_extract(.., 100) -> NULL
    let a = vj(&[Some("[10,20,30]")]);
    let assert_idx = |idx: i64, want: Option<&str>| {
        let i = vi(Ty::BigInt, &[Some(idx)]);
        assert_eq!(str_at(&run("list_extract", &[&a, &i]).unwrap(), 0).as_deref(), want);
    };
    assert_idx(1, Some("10"));
    assert_idx(-1, Some("30"));
    assert_idx(0, None);
    assert_idx(100, None);
}

#[test]
fn list_concat_joins_arrays_and_reads_null_as_an_empty_list() {
    // duckdb: list_concat([1,2],[3]) -> [1, 2, 3];
    //         list_concat([], [1]) -> [1]; list_concat([1]) -> [1];
    //         list_concat([1,2],[3],NULL::INT[],[4]) -> [1, 2, 3, 4];
    //         list_concat([1], NULL::INTEGER[]) -> [1];
    //         list_concat(NULL::INTEGER[], NULL::INTEGER[]) -> [].
    // A NULL list reads as an empty one, so the function never returns NULL.
    let one = |args: &[&Vector]| str_at(&run("list_concat", args).unwrap(), 0);
    let a = vj(&[Some("[1,2]")]);
    let b = vj(&[Some("[3]")]);
    let empty = vj(&[Some("[]")]);
    let null = vj(&[None]);
    assert_eq!(one(&[&a, &b]).as_deref(), Some("[1,2,3]"));
    assert_eq!(one(&[&empty, &b]).as_deref(), Some("[3]"));
    assert_eq!(one(&[&b, &empty]).as_deref(), Some("[3]"));
    assert_eq!(one(&[&a]).as_deref(), Some("[1,2]"));
    assert_eq!(one(&[&a, &b, &null, &vj(&[Some("[4]")])]).as_deref(), Some("[1,2,3,4]"));
    assert_eq!(one(&[&a, &null]).as_deref(), Some("[1,2]"));
    assert_eq!(one(&[&null, &null]).as_deref(), Some("[]"));
    assert_eq!(code_of(resolve("list_concat", &[])), Some(Code::WrongArgCount));
}

#[test]
fn list_concat_aliases_resolve_to_the_same_function() {
    // duckdb: list_cat([1],[2]) = array_concat([1],[2]) = array_cat([1],[2])
    //         = list_concat([1],[2]) -> [1, 2]
    let (a, b) = (vj(&[Some("[1]")]), vj(&[Some("[2]")]));
    for name in ["list_concat", "list_cat", "array_concat", "array_cat"] {
        let out = run(name, &[&a, &b]).unwrap();
        assert_eq!(out.ty(), Ty::Json);
        assert_eq!(str_at(&out, 0).as_deref(), Some("[1,2]"), "{name}");
    }
}

#[test]
fn list_concat_function_is_null_when_an_operand_is_not_a_json_array() {
    // This case cannot exist in DuckDB (LIST and JSON are separate types
    // there, so a non-list argument is a binder error). Here a list *is* a
    // JSON value, so it has to be decided at run time. The *function* keeps
    // the leniency `list_extract`/`list_slice`/`list_transform` already have
    // — SQL NULL, never a text concatenation that would produce invalid JSON.
    // The *operator* raises instead, see
    // `list_concat_operator_raises_type_mismatch_on_a_non_array`.
    let obj = vj(&[Some(r#"{"a":1}"#)]);
    let arr = vj(&[Some("[1]")]);
    assert_eq!(str_at(&run("list_concat", &[&obj, &arr]).unwrap(), 0), None);
    assert_eq!(str_at(&run("list_concat", &[&arr, &obj]).unwrap(), 0), None);
    assert_eq!(str_at(&run("list_concat", &[&vj(&[Some("5")]), &arr]).unwrap(), 0), None);
}

/// `||` on JSON, which `resolve` cannot reach (an operator has no name) —
/// `plan::compile::binary` emits `F_LIST_CONCAT_OP` directly, so call it the
/// same way here.
fn concat_op(args: &[&Vector]) -> Result<Vector> {
    call(F_LIST_CONCAT_OP, Ty::Json, args)
}

#[test]
fn list_concat_operator_propagates_null_where_the_function_absorbs_it() {
    // duckdb: [1,2] || [3] -> [1, 2, 3]; [1] || NULL::INTEGER[] -> NULL;
    //         NULL || [1] -> NULL.
    // Contrast `list_concat_joins_arrays_and_reads_null_as_an_empty_list`:
    // the function returns `[1]`/`[]` for the same NULL operands.
    let (a, b) = (vj(&[Some("[1,2]")]), vj(&[Some("[3]")]));
    let null = vj(&[None]);
    assert_eq!(str_at(&concat_op(&[&a, &b]).unwrap(), 0).as_deref(), Some("[1,2,3]"));
    assert_eq!(str_at(&concat_op(&[&a, &null]).unwrap(), 0), None);
    assert_eq!(str_at(&concat_op(&[&null, &a]).unwrap(), 0), None);
    assert_eq!(str_at(&concat_op(&[&null, &null]).unwrap(), 0), None);
}

#[test]
fn list_concat_operator_raises_type_mismatch_on_a_non_array() {
    // A non-array JSON operand is a hard error for the operator, not NULL:
    // NULL would trade one silent wrong answer for another, and this whole
    // code path exists to stop `||` from quietly returning `{"a":1}{"b":2}`.
    // `TypeMismatch` is what the VARCHAR `||` kernel (`kernels::concat`) and
    // the other undefined JSON operator (ordering comparison) already raise.
    let obj = vj(&[Some(r#"{"a":1}"#)]);
    let arr = vj(&[Some("[1]")]);
    let scalar = vj(&[Some("5")]);
    let null = vj(&[None]);
    for args in [
        [&obj, &obj],
        [&obj, &arr], // mixed: duckdb rejects `[1] || '{"a":1}'::JSON` too
        [&arr, &obj],
        [&scalar, &arr],
        // The error wins over NULL propagation, in either operand order, so
        // the result never depends on which side is scanned first.
        [&obj, &null],
        [&null, &obj],
    ] {
        assert_eq!(code_of(concat_op(&args)), Some(Code::TypeMismatch));
    }
}

#[test]
fn list_concat_operator_keeps_rows_independent() {
    // A whole-batch raise, not a per-row one: any row with a non-array
    // operand fails the query. Rows that are all arrays still concatenate
    // when no row is bad.
    let a = vj(&[Some("[1]"), Some("[]"), None]);
    let b = vj(&[Some("[2]")]);
    let out = concat_op(&[&a, &b]).unwrap();
    assert_eq!(str_at(&out, 0).as_deref(), Some("[1,2]"));
    assert_eq!(str_at(&out, 1).as_deref(), Some("[2]"));
    assert_eq!(str_at(&out, 2), None);

    let bad = vj(&[Some("[1]"), Some(r#"{"a":1}"#)]);
    assert_eq!(code_of(concat_op(&[&bad, &b])), Some(Code::TypeMismatch));
}

#[test]
fn list_concat_keeps_rows_independent() {
    // Per-row NULL, not a whole-vector one: row 1's non-array operand must
    // not disturb rows 0/2. Also exercises the constant (stride 0) path.
    let a = vj(&[Some("[1]"), Some(r#"{"a":1}"#), Some("[]")]);
    let b = vj(&[Some("[2]")]);
    let out = run("list_concat", &[&a, &b]).unwrap();
    assert_eq!(str_at(&out, 0).as_deref(), Some("[1,2]"));
    assert_eq!(str_at(&out, 1), None);
    assert_eq!(str_at(&out, 2).as_deref(), Some("[2]"));
}

#[test]
fn list_concat_rejects_malformed_json_instead_of_truncating() {
    // A broken array is an error, not a silently shortened list.
    let bad = vj(&[Some("[1,")]);
    let ok = vj(&[Some("[2]")]);
    assert!(run("list_concat", &[&bad, &ok]).is_err());
}

#[test]
fn map_extract_looks_up_a_key_or_returns_null() {
    let m = vj(&[Some(r#"{"a":1,"b":2}"#)]);
    assert_eq!(
        str_at(&run("map_extract", &[&m, &vs(&[Some("a")])]).unwrap(), 0).as_deref(),
        Some("1")
    );
    assert_eq!(str_at(&run("map_extract", &[&m, &vs(&[Some("z")])]).unwrap(), 0), None);
}

#[test]
fn to_json_covers_the_documented_scalar_types() {
    // duckdb: to_json(1)='1', to_json(1.5)='1.5', to_json('hello')='"hello"',
    // to_json(true)='true', to_json(DATE '2024-01-01')='"2024-01-01"'
    assert_eq!(
        str_at(&run("to_json", &[&vi(Ty::BigInt, &[Some(1)])]).unwrap(), 0).as_deref(),
        Some("1")
    );
    assert_eq!(str_at(&run("to_json", &[&vf(&[Some(1.5)])]).unwrap(), 0).as_deref(), Some("1.5"));
    assert_eq!(
        str_at(&run("to_json", &[&vs(&[Some("hello")])]).unwrap(), 0).as_deref(),
        Some("\"hello\"")
    );
    let mut b = Vector::new(Ty::Boolean);
    b.push_value(&Value::Bool(true));
    assert_eq!(str_at(&run("to_json", &[&b]).unwrap(), 0).as_deref(), Some("true"));
    let mut d = Vector::new(Ty::Date);
    d.push_value(&Value::I32(19723)); // 2024-01-01
    assert_eq!(str_at(&run("to_json", &[&d]).unwrap(), 0).as_deref(), Some("\"2024-01-01\""));
    // duckdb: to_json(NULL) stays SQL NULL (the default NULL propagation).
    let n = vi(Ty::BigInt, &[None]);
    assert_eq!(str_at(&run("to_json", &[&n]).unwrap(), 0), None);
}

#[test]
fn to_json_escapes_quotes_in_strings() {
    let out = run("to_json", &[&vs(&[Some("a\"b\\c")])]).unwrap();
    assert_eq!(str_at(&out, 0).as_deref(), Some(r#""a\"b\\c""#));
}

#[test]
fn json_array_and_json_object_match_duckdb_construction() {
    // duckdb: json_array(1, 'x', true, NULL) -> [1,"x",true,null]
    let out = run(
        "json_array",
        &[
            &vi(Ty::BigInt, &[Some(1)]),
            &vs(&[Some("x")]),
            &{
                let mut b = Vector::new(Ty::Boolean);
                b.push_value(&Value::Bool(true));
                b
            },
            &vi(Ty::BigInt, &[None]),
        ],
    )
    .unwrap();
    assert_eq!(out.ty(), Ty::Json);
    assert_eq!(str_at(&out, 0).as_deref(), Some(r#"[1,"x",true,null]"#));

    // duckdb: json_object('a', 1, 'b', 'x') -> {"a":1,"b":"x"}
    let out = run(
        "json_object",
        &[&vs(&[Some("a")]), &vi(Ty::BigInt, &[Some(1)]), &vs(&[Some("b")]), &vs(&[Some("x")])],
    )
    .unwrap();
    assert_eq!(str_at(&out, 0).as_deref(), Some(r#"{"a":1,"b":"x"}"#));

    // duckdb: json_object('a', NULL) -> {"a":null} (a NULL value is embedded)
    let out = run("json_object", &[&vs(&[Some("a")]), &vi(Ty::BigInt, &[None])]).unwrap();
    assert_eq!(str_at(&out, 0).as_deref(), Some(r#"{"a":null}"#));
}

#[test]
fn list_value_is_an_alias_for_json_array() {
    let out =
        run("list_value", &[&vi(Ty::BigInt, &[Some(1)]), &vi(Ty::BigInt, &[Some(2)])]).unwrap();
    assert_eq!(str_at(&out, 0).as_deref(), Some("[1,2]"));
}

#[test]
fn malformed_json_source_is_an_error_not_a_silent_null() {
    // duckdb: json_extract('{not valid json', '$.a') -> an error
    let doc = vj(&[Some("{not valid json")]);
    let path = vs(&[Some("$.a")]);
    assert_eq!(code_of(run("json_extract", &[&doc, &path])), Some(Code::SyntaxError));
}

#[test]
fn json_resolve_errors() {
    assert_eq!(code_of(resolve("json_extract", &[Ty::Json])), Some(Code::WrongArgCount));
    assert_eq!(code_of(resolve("to_json", &[])), Some(Code::WrongArgCount));
    // BLOB/INTERVAL are outside the scope of to_json/json_array.
    assert_eq!(code_of(resolve("to_json", &[Ty::Blob])), Some(Code::TypeMismatch));
    assert_eq!(code_of(resolve("to_json", &[Ty::Interval])), Some(Code::TypeMismatch));
    assert_eq!(
        code_of(resolve("json_object", &[Ty::Varchar])),
        Some(Code::WrongArgCount),
        "an odd number of arguments is rejected"
    );
    assert_eq!(code_of(resolve("json_array", &[])), Some(Code::WrongArgCount));
}

// --- Errors -------------------------------------------------------------

#[test]
fn resolve_errors() {
    assert_eq!(code_of(resolve("upper", &[])), Some(Code::WrongArgCount));
    assert_eq!(code_of(resolve("upper", &[Ty::Varchar, Ty::Varchar])), Some(Code::WrongArgCount));
    assert_eq!(code_of(resolve("substr", &[Ty::Varchar])), Some(Code::WrongArgCount));
    assert_eq!(code_of(resolve("pow", &[Ty::Double])), Some(Code::WrongArgCount));
    assert_eq!(code_of(resolve("abs", &[Ty::Varchar])), Some(Code::TypeMismatch));
    assert_eq!(code_of(resolve("mod", &[Ty::Int, Ty::Varchar])), Some(Code::TypeMismatch));
    assert_eq!(code_of(resolve("greatest", &[Ty::Int, Ty::Varchar])), Some(Code::TypeMismatch));
    assert_eq!(code_of(resolve("nosuchfunc", &[])), Some(Code::FunctionNotFound));
    // Not implemented, since there is no clock (see the comments at the top of the module).
    assert_eq!(code_of(resolve("now", &[])), Some(Code::FunctionNotFound));
    assert_eq!(code_of(resolve("current_timestamp", &[])), Some(Code::FunctionNotFound));
    assert_eq!(code_of(resolve("current_date", &[])), Some(Code::FunctionNotFound));
}

#[test]
fn resolve_coerces_argument_types() {
    // INT arguments settle on BIGINT, and a FLOAT mixed in settles on DOUBLE.
    assert_eq!(resolve("abs", &[Ty::Int]).unwrap().1, vec![Ty::BigInt]);
    assert_eq!(resolve("abs", &[Ty::Double]).unwrap().1, vec![Ty::Double]);
    assert_eq!(resolve("abs", &[Ty::Decimal { precision: 9, scale: 2 }]).unwrap().2, Ty::Double);
    assert_eq!(resolve("length", &[Ty::Varchar]).unwrap().2, Ty::BigInt);
    // Date-time functions settle on TIMESTAMP (DATE columns and VARCHARs both arrive via a Cast).
    assert_eq!(resolve("year", &[Ty::Date]).unwrap().1, vec![Ty::Timestamp]);
    assert_eq!(
        resolve("date_trunc", &[Ty::Varchar, Ty::Date]).unwrap().1,
        vec![Ty::Varchar, Ty::Timestamp]
    );
    assert_eq!(resolve("coalesce", &[Ty::Int, Ty::BigInt]).unwrap().2, Ty::BigInt);
}

#[test]
fn call_errors_on_bad_part() {
    let (id, _, ret) = resolve("date_part", &[Ty::Varchar, Ty::Timestamp]).unwrap();
    let bad = vs(&[Some("fortnight")]);
    let t = ts("2024-01-01 00:00:00");
    assert_eq!(code_of(call(id, ret, &[&bad, &t])), Some(Code::TypeMismatch));
    // It does not panic even when called with no arguments.
    assert_eq!(code_of(call(id, ret, &[])), Some(Code::WrongArgCount));
}

#[test]
fn oversized_output_is_rejected() {
    let (id, _, ret) = resolve("repeat", &[Ty::Varchar, Ty::BigInt]).unwrap();
    let s = vs(&[Some("abcdefgh")]);
    let k = vi(Ty::BigInt, &[Some(1 << 40)]);
    assert_eq!(code_of(call(id, ret, &[&s, &k])), Some(Code::LimitExceeded));
}

// --- GLOB / SIMILAR TO ---------------------------------------------------

fn bool_at(v: &Vector, i: usize) -> Option<bool> {
    v.is_valid(i).then(|| v.bools().get(i))
}

#[test]
fn glob_matches_shell_patterns_like_duckdb() {
    let g =
        |s: &str, p: &str| bool_at(&run("glob", &[&vs(&[Some(s)]), &vs(&[Some(p)])]).unwrap(), 0);
    // duckdb: 'abc' glob 'a*c' = true, 'abc' glob 'a?c' = true
    assert_eq!(g("abc", "a*c"), Some(true));
    assert_eq!(g("abc", "a?c"), Some(true));
    assert_eq!(g("abc", "ab"), Some(false));
    // duckdb: 'ABC' glob 'a*c' = false (case-sensitive)
    assert_eq!(g("ABC", "a*c"), Some(false));
    // duckdb: 'abc' glob 'a[bx]c' = true, 'abc' glob '[!x]bc' = true,
    // 'abc' glob '[^x]bc' = false (`^` does not negate and is treated as a literal)
    assert_eq!(g("abc", "a[bx]c"), Some(true));
    assert_eq!(g("abc", "[!x]bc"), Some(true));
    assert_eq!(g("abc", "[^x]bc"), Some(false));
    // duckdb: 'a]c' glob 'a[]]c' = true (a leading `]` is a literal)
    assert_eq!(g("a]c", "a[]]c"), Some(true));
    // duckdb: 'a1c' glob 'a[0-9]c' = true
    assert_eq!(g("a1c", "a[0-9]c"), Some(true));
    // duckdb: 'a*c' glob 'a\*c' = true, 'abc' glob 'a\*c' = false
    assert_eq!(g("a*c", "a\\*c"), Some(true));
    assert_eq!(g("abc", "a\\*c"), Some(false));
    // duckdb: '' glob '' = true, '' glob '*' = true
    assert_eq!(g("", ""), Some(true));
    assert_eq!(g("", "*"), Some(true));
    // duckdb: 'a[bc' glob 'a[bc' = false (an unclosed `[` never matches)
    assert_eq!(g("a[bc", "a[bc"), Some(false));
    // NULL propagates from either argument.
    let n = run("glob", &[&vs(&[None]), &vs(&[Some("a*")])]).unwrap();
    assert!(!n.is_valid(0));
}

#[test]
fn regexp_full_match_backs_similar_to() {
    let f = |s: &str, p: &str| {
        bool_at(&run("regexp_full_match", &[&vs(&[Some(s)]), &vs(&[Some(p)])]).unwrap(), 0)
    };
    // duckdb: 'abc' similar to 'a.c' = true, 'Xabc' similar to 'a.c' = false
    // (a full match rather than a partial one).
    assert_eq!(f("abc", "a.c"), Some(true));
    assert_eq!(f("Xabc", "a.c"), Some(false));
    // duckdb: 'a%c' similar to 'a%c' = true, 'aXc' similar to 'a%c' = false
    // (in SIMILAR TO's regex dialect, `%`/`_` are not special).
    assert_eq!(f("a%c", "a%c"), Some(true));
    assert_eq!(f("aXc", "a%c"), Some(false));
    // duckdb: 'aaa' similar to 'a{1,2}' = false (3 is out of range, so it does not match fully).
    assert_eq!(f("aaa", "a{1,2}"), Some(false));
    assert_eq!(f("aa", "a{1,2}"), Some(true));
    // Even a pathological pattern returns an error rather than panicking
    // (confirming it goes straight through `expr::regex`'s limits).
    let huge = "a{1001}";
    assert_eq!(
        code_of(run("regexp_full_match", &[&vs(&[Some("a")]), &vs(&[Some(huge)])])),
        Some(Code::LimitExceeded)
    );
}

// --- printf / format ------------------------------------------------------

#[test]
// 3.14159 is a test value for checking printf's `%.2f` rounding and is not meant as pi
// (a false positive from clippy::approx_constant).
#[allow(clippy::approx_constant)]
fn printf_supports_documented_specifiers() {
    let f = |fmt: &str, args: &[&Vector]| {
        let fmt_v = vs(&[Some(fmt)]);
        let tys: Vec<Ty> =
            core::iter::once(Ty::Varchar).chain(args.iter().map(|a| a.ty())).collect();
        let (id, want, ret) = resolve("printf", &tys).unwrap();
        let mut refs: Vec<&Vector> = vec![&fmt_v];
        refs.extend_from_slice(args);
        assert_eq!(want, tys);
        str_at(&call(id, ret, &refs).unwrap(), 0)
    };
    let i = vi(Ty::BigInt, &[Some(42)]);
    let s = vs(&[Some("x")]);
    let d = vf(&[Some(3.14159)]);
    // duckdb: printf('%d-%s', 42, 'x') = '42-x'
    assert_eq!(f("%d-%s", &[&i, &s]), Some("42-x".to_string()));
    // duckdb: printf('%%') = '%'
    assert_eq!(f("%%", &[]), Some("%".to_string()));
    // duckdb: printf('%05d', 3) = '00003'
    let three = vi(Ty::BigInt, &[Some(3)]);
    assert_eq!(f("%05d", &[&three]), Some("00003".to_string()));
    // duckdb: printf('%-5d|', 3) = '3    |'
    assert_eq!(f("%-5d|", &[&three]), Some("3    |".to_string()));
    // duckdb: printf('%05d', -3) = '-0003'
    let neg3 = vi(Ty::BigInt, &[Some(-3)]);
    assert_eq!(f("%05d", &[&neg3]), Some("-0003".to_string()));
    // duckdb: printf('%.2f', 3.14159) = '3.14'
    assert_eq!(f("%.2f", &[&d]), Some("3.14".to_string()));
    // duckdb: printf('%f', 3.5) = '3.500000' (the default precision of 6 digits)
    let half = vf(&[Some(3.5)]);
    assert_eq!(f("%f", &[&half]), Some("3.500000".to_string()));
    // If any argument is NULL the whole result is NULL (the default propagation).
    let null_s = vs(&[None]);
    assert_eq!(f("%s", &[&null_s]), None);
}

#[test]
fn printf_rejects_malformed_or_unsupported_specifiers() {
    let (id, _, ret) = resolve("printf", &[Ty::Varchar]).unwrap();
    let fmt = vs(&[Some("%")]);
    assert_eq!(code_of(call(id, ret, &[&fmt])), Some(Code::SyntaxError));
    let fmt = vs(&[Some("%x")]);
    assert_eq!(code_of(call(id, ret, &[&fmt])), Some(Code::UnsupportedFeature));
    // Fewer actual arguments than specifiers gives WrongArgCount.
    let (id, _, ret) = resolve("printf", &[Ty::Varchar]).unwrap();
    let fmt = vs(&[Some("%d")]);
    assert_eq!(code_of(call(id, ret, &[&fmt])), Some(Code::WrongArgCount));
}

#[test]
fn format_supports_positional_and_auto_placeholders() {
    let f = |fmt: &str, args: &[&Vector]| {
        let all = vs(&[Some(fmt)]);
        let tys: Vec<Ty> =
            core::iter::once(Ty::Varchar).chain(args.iter().map(|a| a.ty())).collect();
        let (id, _, ret) = resolve("format", &tys).unwrap();
        let mut refs: Vec<&Vector> = vec![&all];
        refs.extend_from_slice(args);
        str_at(&call(id, ret, &refs).unwrap(), 0)
    };
    let i = vi(Ty::BigInt, &[Some(42)]);
    let s = vs(&[Some("x")]);
    // duckdb: format('{}-{}', 42, 'x') = '42-x'
    assert_eq!(f("{}-{}", &[&i, &s]), Some("42-x".to_string()));
    // duckdb: format('{{literal}}') = '{literal}'
    assert_eq!(f("{{literal}}", &[]), Some("{literal}".to_string()));
    // duckdb: format('{1}-{0}', 'a', 'b') = 'b-a'
    let a = vs(&[Some("a")]);
    let b = vs(&[Some("b")]);
    assert_eq!(f("{1}-{0}", &[&a, &b]), Some("b-a".to_string()));
}

#[test]
fn numeric_min_edge_cases_do_not_panic() {
    let min_i = vi(Ty::BigInt, &[Some(i64::MIN)]);
    let one = vi(Ty::BigInt, &[Some(1)]);
    let hundred = vi(Ty::BigInt, &[Some(100)]);

    // lcm(i64::MIN, 1) should not panic, returns None (overflow)
    let res = run("lcm", &[&min_i, &one]).unwrap();
    assert_eq!(int_at(&res, 0), None);

    // round(100, i64::MIN) should not panic, returns 0
    let res = run("round", &[&hundred, &min_i]).unwrap();
    assert_eq!(int_at(&res, 0), Some(0));
}
