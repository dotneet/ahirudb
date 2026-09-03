//! End-to-end regression tests for four silently-wrong-answer bugs in the
//! expression kernels and the type system. Every expected value here was
//! cross-checked against a real `duckdb` 1.4.4 CLI; the query that produced it
//! is quoted above each test.
//!
//! - `LIKE`'s `%` was matched literally whenever the subject byte at the same
//!   position happened to be `%`, so `'50% off' LIKE '50%'` was false.
//! - A signed and an unsigned integer type of the same width shared a rank in
//!   `Ty::unify`, so the tie-break kept whichever operand was written first and
//!   turned the other side's valid values into NULL.
//! - The runtime vector type of a DECIMAL `*` was recomputed with `Ty::unify`,
//!   giving scale `max(s1, s2)` where the raw product is scaled by `s1 + s2`.
//! - A DECIMAL `*` whose result scale exceeded 38 clamped the type without
//!   rescaling the value.
//!
//! Constant-only expressions and expressions over real columns take different
//! code paths in this engine (constant folding vs. the vectorized kernels), so
//! the interesting cases are checked both ways.

use ahiru_core::error::{code_of, Code};
use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn session_with_dual() -> Session {
    let mut s = Session::new();
    s.register_bytes_as("dual", b"x\n1\n".to_vec(), FormatKind::Csv).unwrap();
    s
}

fn run(s: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    let mut q = match s.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
    };
    let mut rows = Vec::new();
    loop {
        match s.step(&mut q).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
            QueryStep::Batch(mut b) => {
                b.materialize();
                for r in 0..b.num_rows() {
                    rows.push(b.cols.iter().map(|c| c.value_at(r)).collect());
                }
            }
            QueryStep::Done => break,
            QueryStep::NeedIo(_) | QueryStep::NeedCodec(_) => panic!("unexpected suspend"),
        }
    }
    rows
}

/// The single row's values, as a `Vec`.
fn row(s: &mut Session, sql: &str) -> Vec<Value> {
    let rows = run(s, sql);
    assert_eq!(rows.len(), 1, "{sql}: expected exactly one row");
    rows.into_iter().next().unwrap()
}

fn text(v: &Value) -> String {
    match v {
        Value::Bytes(b) => String::from_utf8(b.clone()).unwrap(),
        other => panic!("expected text, got {other:?}"),
    }
}

// --- LIKE: `%` is always the wildcard -----------------------------------------

/// duckdb: `SELECT '50% off' LIKE '50%', '%abc' LIKE '%bc', '%%' LIKE '%',
///          '%ABC' ILIKE '%bc'` -> true, true, true, true
#[test]
fn like_percent_wildcard_against_a_literal_percent_subject() {
    let mut db = session_with_dual();
    assert_eq!(
        row(&mut db, "SELECT '50% off' LIKE '50%', '%abc' LIKE '%bc', '%%' LIKE '%', '%ABC' ILIKE '%bc' FROM dual"),
        vec![Value::Bool(true), Value::Bool(true), Value::Bool(true), Value::Bool(true)]
    );
}

/// The same patterns applied to a real column, which goes through the
/// vectorized `like` kernel rather than constant folding.
///
/// duckdb: over the same four subjects, `s LIKE '50%'` -> true, false, false, false
/// and `s LIKE '%bc'` -> false, true, false, true.
#[test]
fn like_percent_wildcard_on_a_column() {
    let mut db = Session::new();
    db.register_bytes_as("t", b"s\n50% off\n%abc\n%%\nabc\n".to_vec(), FormatKind::Csv).unwrap();
    let rows = run(&mut db, "SELECT s LIKE '50%', s LIKE '%bc', s NOT LIKE '%bc' FROM t");
    assert_eq!(
        rows,
        vec![
            vec![Value::Bool(true), Value::Bool(false), Value::Bool(true)],
            vec![Value::Bool(false), Value::Bool(true), Value::Bool(false)],
            vec![Value::Bool(false), Value::Bool(false), Value::Bool(true)],
            vec![Value::Bool(false), Value::Bool(true), Value::Bool(false)],
        ]
    );
}

/// duckdb: `SELECT '50% off' LIKE '50!%%' ESCAPE '!', '50% off' LIKE '50!%' ESCAPE '!',
///          '%abc' LIKE '!%%' ESCAPE '!', 'xabc' LIKE '!%%' ESCAPE '!',
///          '50% off' LIKE '50%' ESCAPE '!'` -> true, false, true, false, true
#[test]
fn like_escape_keeps_unescaped_percent_a_wildcard() {
    let mut db = session_with_dual();
    assert_eq!(
        row(
            &mut db,
            "SELECT '50% off' LIKE '50!%%' ESCAPE '!', '50% off' LIKE '50!%' ESCAPE '!', \
             '%abc' LIKE '!%%' ESCAPE '!', 'xabc' LIKE '!%%' ESCAPE '!', \
             '50% off' LIKE '50%' ESCAPE '!' FROM dual"
        ),
        vec![
            Value::Bool(true),
            Value::Bool(false),
            Value::Bool(true),
            Value::Bool(false),
            Value::Bool(true),
        ]
    );
}

/// `_` matches one Unicode character, not one byte, and an escaped `_` matches
/// the literal byte only -- both still true next to the reordered `%` branch.
///
/// duckdb: `SELECT '%あ' LIKE '%_', 'aあc' LIKE 'a_c', 'aあc' LIKE 'a!_c' ESCAPE '!',
///          'a_c' LIKE 'a!_c' ESCAPE '!'` -> true, true, false, true
#[test]
fn like_underscore_stays_character_aligned() {
    let mut db = session_with_dual();
    assert_eq!(
        row(
            &mut db,
            "SELECT '%あ' LIKE '%_', 'aあc' LIKE 'a_c', 'aあc' LIKE 'a!_c' ESCAPE '!', \
             'a_c' LIKE 'a!_c' ESCAPE '!' FROM dual"
        ),
        vec![Value::Bool(true), Value::Bool(true), Value::Bool(false), Value::Bool(true)]
    );
}

// --- signed / unsigned unification --------------------------------------------

/// duckdb: `SELECT 0::UTINYINT + (-1)::TINYINT, 1::UBIGINT = (-1)::BIGINT,
///          1::UBIGINT > (-1)::BIGINT, 100::TINYINT + 200::UTINYINT,
///          200::UTINYINT + 100::TINYINT` -> -1, false, true, 300, 300
#[test]
fn signed_and_unsigned_operands_widen_to_a_signed_type() {
    let mut db = session_with_dual();
    let r = row(
        &mut db,
        "SELECT 0::UTINYINT + (-1)::TINYINT, 1::UBIGINT = (-1)::BIGINT, \
         1::UBIGINT > (-1)::BIGINT, 100::TINYINT + 200::UTINYINT, \
         200::UTINYINT + 100::TINYINT FROM dual",
    );
    assert_eq!(r[0].as_i64(), Some(-1));
    assert_eq!(r[1], Value::Bool(false));
    assert_eq!(r[2], Value::Bool(true));
    assert_eq!(r[3].as_i64(), Some(300));
    assert_eq!(r[4].as_i64(), Some(300));
}

/// The unified type must be reached from either operand order, and the wider
/// signed side must win when it is already wide enough.
///
/// duckdb: `SELECT 1::UBIGINT < (-1)::BIGINT, (-1)::BIGINT < 1::UBIGINT,
///          (-1)::TINYINT IN (255::UTINYINT, (-1)::TINYINT),
///          300::USMALLINT IN (300::SMALLINT),
///          18446744073709551615::UBIGINT + 0::BIGINT`
///          -> false, true, true, true, 18446744073709551615
#[test]
fn signed_and_unsigned_comparisons_and_in_lists() {
    let mut db = session_with_dual();
    let r = row(
        &mut db,
        "SELECT 1::UBIGINT < (-1)::BIGINT, (-1)::BIGINT < 1::UBIGINT, \
         (-1)::TINYINT IN (255::UTINYINT, (-1)::TINYINT), \
         300::USMALLINT IN (300::SMALLINT), \
         18446744073709551615::UBIGINT + 0::BIGINT FROM dual",
    );
    assert_eq!(r[0], Value::Bool(false));
    assert_eq!(r[1], Value::Bool(true));
    assert_eq!(r[2], Value::Bool(true));
    assert_eq!(r[3], Value::Bool(true));
    assert_eq!(r[4], Value::I128(18_446_744_073_709_551_615));
}

/// The same over real columns, so the values ride the vectorized cast/compare
/// kernels rather than being folded at compile time. A `WHERE` predicate that
/// silently NULLed the negative side used to drop the row entirely.
///
/// duckdb, over `u` = 0/200 (UTINYINT) and `s` = -1/100 (TINYINT):
/// `SELECT u + s` -> -1, 300; `SELECT count(*) WHERE u > s` -> 2.
#[test]
fn signed_and_unsigned_columns_do_not_null_out() {
    let mut db = Session::new();
    db.register_bytes_as("t", b"u,s\n0,-1\n200,100\n".to_vec(), FormatKind::Csv).unwrap();
    let rows = run(&mut db, "SELECT u::UTINYINT + s::TINYINT FROM t");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0][0].as_i64(), Some(-1));
    assert_eq!(rows[1][0].as_i64(), Some(300));
    let cnt = row(&mut db, "SELECT count(*) FROM t WHERE u::UTINYINT > s::TINYINT");
    assert_eq!(cnt[0].as_i64(), Some(2));
}

// --- DECIMAL multiplication ----------------------------------------------------

/// The result vector's logical type has to carry the *summed* scale, or every
/// consumer that renders the raw integer (JSON, printf, list building, the text
/// cast) is off by 10^min(s1,s2).
///
/// duckdb: `SELECT to_json(x), printf('%s', x), list_value(x), x::VARCHAR
///          FROM (SELECT 1.5::DECIMAL(4,1) * 1.25::DECIMAL(3,2) AS x)`
///          -> 1.875, 1.875, [1.875], 1.875
#[test]
fn decimal_multiplication_keeps_the_summed_scale_everywhere() {
    let mut db = session_with_dual();
    let r = row(
        &mut db,
        "SELECT to_json(x), printf('%s', x), list_value(x), x::VARCHAR \
         FROM (SELECT 1.5::DECIMAL(4,1) * 1.25::DECIMAL(3,2) AS x FROM dual)",
    );
    assert_eq!(text(&r[0]), "1.875");
    assert_eq!(text(&r[1]), "1.875");
    assert_eq!(text(&r[2]), "[1.875]");
    assert_eq!(text(&r[3]), "1.875");
}

/// DECIMAL times an integer takes the same path (the integer counts as a
/// scale-0 DECIMAL), and it must survive a round trip through a column.
///
/// duckdb: `SELECT (1.5::DECIMAL(4,1) * 2)::VARCHAR` -> 3.0
#[test]
fn decimal_times_integer_renders_at_the_planned_scale() {
    let mut db = Session::new();
    db.register_bytes_as("t", b"a,b\n1.5,1.25\n2.5,0.5\n".to_vec(), FormatKind::Csv).unwrap();
    let mut d = session_with_dual();
    assert_eq!(text(&row(&mut d, "SELECT (1.5::DECIMAL(4,1) * 2)::VARCHAR FROM dual")[0]), "3.0");
    // Over columns: 1.5 * 1.25 = 1.875 and 2.5 * 0.5 = 1.250 at scale 3.
    let rows = run(&mut db, "SELECT (a::DECIMAL(4,1) * b::DECIMAL(3,2))::VARCHAR FROM t");
    assert_eq!(text(&rows[0][0]), "1.875");
    assert_eq!(text(&rows[1][0]), "1.250");
}

/// A product whose scale would exceed the maximum DECIMAL scale of 38 is an
/// error, not a silently unrescaled value. DuckDB raises "Out of Range Error:
/// Needed scale 40 ... Max scale is 38" for exactly this expression; the
/// documented workarounds (cast to DOUBLE, or to a smaller scale) still work.
#[test]
fn decimal_multiplication_scale_over_38_is_an_error() {
    let mut db = session_with_dual();
    assert_eq!(
        code_of(db.prepare("SELECT 0.01::DECIMAL(25,20) * 0.01::DECIMAL(25,20) FROM dual", &[])),
        Some(Code::ValueOutOfRange)
    );
    // Casting one side to DOUBLE is the escape hatch, and it is exact here.
    let r = row(&mut db, "SELECT 0.01::DECIMAL(25,20) * 0.01::DECIMAL(25,20)::DOUBLE FROM dual");
    assert_eq!(r[0], Value::F64(0.0001));
    // A precision (rather than scale) overflow still just clamps to 38, as in
    // duckdb: `typeof(1.0::DECIMAL(20,2) * 1.0::DECIMAL(19,2))` -> DECIMAL(38,4).
    assert_eq!(
        text(
            &row(&mut db, "SELECT (1.5::DECIMAL(20,2) * 2.5::DECIMAL(19,2))::VARCHAR FROM dual")[0]
        ),
        "3.7500"
    );
}
