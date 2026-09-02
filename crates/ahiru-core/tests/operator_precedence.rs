//! End-to-end checks for the operator-precedence and literal-lexing fixes:
//! prefix `~`/`@` moved into PostgreSQL's "any other operator" band,
//! `IN`/`BETWEEN`/`LIKE` moved one notch above comparison, the bitwise
//! operators merged into `||`'s band, underscore digit separators,
//! leading-dot floats, reserved words as aliases after `AS`, `INTERVAL` as a
//! type name, and the extra INTERVAL literal spellings.
//!
//! The parse *trees* are pinned by unit tests in
//! `crates/ahiru-core/src/sql/parser/tests.rs`; this file checks that the
//! reshaped trees actually evaluate to what a real `duckdb` CLI answers,
//! including on real table columns (constant-only expressions take a
//! different code path in this engine). Every expected value in this file was
//! measured with `duckdb -csv -c "..."`.

use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

/// A one-row table `t(id)` with `id = 3`, matching the shape the hand repros used.
fn session_with_id3() -> Session {
    let mut s = Session::new();
    s.register_bytes_as("t", b"id\n3\n".to_vec(), FormatKind::Csv).unwrap();
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

/// The single value of a one-row, one-column query.
fn one(s: &mut Session, sql: &str) -> Value {
    let rows = run(s, sql);
    assert_eq!(rows.len(), 1, "{sql}: expected one row");
    assert_eq!(rows[0].len(), 1, "{sql}: expected one column");
    rows[0][0].clone()
}

// --- Prefix `~`/`@` sit in the binary ladder, not above it ------------------

#[test]
fn bitwise_not_and_abs_prefix_bind_below_multiplication() {
    // duckdb: SELECT ~1 * 2 -> -3 (i.e. `~(1 * 2)`), not -4.
    let mut db = session_with_id3();
    assert_eq!(one(&mut db, "SELECT ~1 * 2 FROM t"), Value::I64(-3));
    // duckdb: SELECT ~id + 1, @id - 5 FROM t (id = 3) -> -5, 2.
    let rows = run(&mut db, "SELECT ~id + 1, @id - 5 FROM t");
    assert_eq!(rows, vec![vec![Value::I64(-5), Value::I64(2)]]);
    // ... but the operand still stops at the operators of its own band and at
    // comparison (duckdb: `~1 || 'a'` -> '-2a', `~1 = -2` -> true).
    assert_eq!(one(&mut db, "SELECT ~1 || 'a' FROM t"), Value::Bytes(b"-2a".to_vec()));
    assert_eq!(one(&mut db, "SELECT ~1 = -2 FROM t"), Value::Bool(true));
    // duckdb: SELECT @ -3 + 1 -> 2 (i.e. `@(-3 + 1)`).
    assert_eq!(one(&mut db, "SELECT @ -3 + 1 FROM t"), Value::I64(2));
}

// --- IN / BETWEEN / LIKE bind tighter than comparison -----------------------

#[test]
fn predicates_bind_tighter_than_comparison() {
    let mut db = session_with_id3();
    // duckdb: all four -> false, false, true, true.
    assert_eq!(one(&mut db, "SELECT false = true IN (false, true) FROM t"), Value::Bool(false));
    assert_eq!(
        one(&mut db, "SELECT false = true BETWEEN false AND true FROM t"),
        Value::Bool(false)
    );
    assert_eq!(one(&mut db, "SELECT true = 1 IN (1, 2) FROM t"), Value::Bool(true));
    assert_eq!(one(&mut db, "SELECT true = 'abc' LIKE 'a%' FROM t"), Value::Bool(true));
    // With non-boolean operands the old parse produced a spurious type mismatch
    // rather than a wrong answer; duckdb answers false for this one.
    assert_eq!(one(&mut db, "SELECT false = 3 BETWEEN 1 AND 5 FROM t"), Value::Bool(false));
    // The `IS` family stays at comparison strength, so it still composes left to
    // right (duckdb: `1 = 1 ISNULL` -> false, `3 IN (1,2,3) IS NULL` -> false).
    assert_eq!(one(&mut db, "SELECT 1 = 1 ISNULL FROM t"), Value::Bool(false));
    assert_eq!(one(&mut db, "SELECT 3 IN (1, 2, 3) IS NULL FROM t"), Value::Bool(false));
}

// --- Bitwise and `||` share one band ---------------------------------------

#[test]
fn bitwise_operators_share_the_concat_band() {
    let mut db = session_with_id3();
    // duckdb: `1 & 2 || 3` -> '03' (i.e. `(1 & 2) || 3`), not `1 & ('2' || '3')`.
    assert_eq!(one(&mut db, "SELECT 1 & 2 || 3 FROM t"), Value::Bytes(b"03".to_vec()));
    // duckdb: `1 + 2 & 3` -> 3, `3 & 2 = 2` -> true.
    assert_eq!(one(&mut db, "SELECT 1 + 2 & 3 FROM t"), Value::I64(3));
    assert_eq!(one(&mut db, "SELECT 3 & 2 = 2 FROM t"), Value::Bool(true));
}

#[test]
fn between_bounds_reach_into_the_bitwise_band() {
    let mut db = session_with_id3();
    // duckdb: all three -> true. The last one used to be an outright
    // "unexpected token", the first two silently answered `1` and `0`.
    assert_eq!(one(&mut db, "SELECT 1 BETWEEN 0 AND 1 & 1 FROM t"), Value::Bool(true));
    assert_eq!(one(&mut db, "SELECT 2 BETWEEN 1 AND 1 << 2 FROM t"), Value::Bool(true));
    assert_eq!(one(&mut db, "SELECT 5 BETWEEN 1 << 1 AND 10 FROM t"), Value::Bool(true));
}

// --- Numeric literals -------------------------------------------------------

#[test]
fn underscore_separators_and_leading_dot_floats() {
    let mut db = session_with_id3();
    // duckdb: 1000, 1001, 1000000, 1000.5. `1_000` used to lex as `1` plus an
    // implicit alias `_000`, and `1_000 + 1` was a syntax error.
    assert_eq!(one(&mut db, "SELECT 1_000 FROM t"), Value::I32(1000));
    assert_eq!(one(&mut db, "SELECT 1_000 + 1 FROM t"), Value::I32(1001));
    assert_eq!(one(&mut db, "SELECT 1_000_000 FROM t"), Value::I32(1_000_000));
    assert_eq!(one(&mut db, "SELECT 1_000.5 FROM t"), Value::F64(1000.5));
    // duckdb: `.5` -> 0.5, `.5 + 1` -> 1.5.
    assert_eq!(one(&mut db, "SELECT .5 FROM t"), Value::F64(0.5));
    assert_eq!(one(&mut db, "SELECT .5 + 1 FROM t"), Value::F64(1.5));
    // `LIMIT` goes through a separate integer reader, which skips separators too.
    assert_eq!(run(&mut db, "SELECT id FROM t LIMIT 1_000").len(), 1);
}

// --- Aliases ----------------------------------------------------------------

#[test]
fn reserved_words_work_as_aliases_after_as() {
    let mut db = session_with_id3();
    // duckdb accepts every one of these. Checked through the binder (not just
    // the parser) so the alias really lands on the output column.
    for kw in ["limit", "offset", "all", "end", "distinct"] {
        let sql = alloc_format(kw);
        assert_eq!(one(&mut db, &sql), Value::I32(1), "{sql}");
    }
}

/// `SELECT 1 AS <kw> FROM t` — kept out of the loop body so the test above reads
/// as a list of the keywords under test.
fn alloc_format(kw: &str) -> String {
    let mut s = String::from("SELECT 1 AS ");
    s.push_str(kw);
    s.push_str(" FROM t");
    s
}

// --- INTERVAL ---------------------------------------------------------------

#[test]
fn interval_is_usable_as_a_type_name() {
    let mut db = session_with_id3();
    // All three used to fail with `[E405] invalid cast`.
    assert_eq!(one(&mut db, "SELECT CAST(NULL AS INTERVAL) FROM t"), Value::Null);
    assert_eq!(one(&mut db, "SELECT typeof(CAST(NULL AS INTERVAL)) FROM t"), interval_name());
    // A no-op cast of a value that is already an INTERVAL round-trips.
    assert_eq!(
        one(&mut db, "SELECT CAST(INTERVAL '1 day' AS INTERVAL) FROM t"),
        one(&mut db, "SELECT INTERVAL '1 day' FROM t")
    );
}

fn interval_name() -> Value {
    Value::Bytes(b"INTERVAL".to_vec())
}

#[test]
fn interval_literals_this_engine_prints_can_be_read_back() {
    let mut db = session_with_id3();
    // Each pair is a spelling duckdb accepts and the equivalent this engine
    // already accepted, so the assertion is independent of the packed layout.
    let same = [
        ("INTERVAL '1:30:00'", "INTERVAL '90 minutes'"),
        ("INTERVAL '01:02:03.5'", "INTERVAL '3723 seconds 500 milliseconds'"),
        ("INTERVAL '01:02'", "INTERVAL '62 minutes'"),
        ("INTERVAL '100:00:00'", "INTERVAL '100 hours'"),
        ("INTERVAL '-1:30:00'", "INTERVAL '-90 minutes'"),
        ("INTERVAL '1.5 days'", "INTERVAL '1 day 12 hours'"),
        ("INTERVAL '0.5 months'", "INTERVAL '15 days'"),
        ("INTERVAL '1.25 years'", "INTERVAL '15 months'"),
        ("INTERVAL '1.5 hours'", "INTERVAL '90 minutes'"),
        ("INTERVAL '3 weeks'", "INTERVAL '21 days'"),
        ("INTERVAL '1.5 weeks'", "INTERVAL '10 days 12 hours'"),
        ("INTERVAL '1 day 01:02:03'", "INTERVAL '1 day 3723 seconds'"),
        ("INTERVAL '-2 days -03:04:05'", "INTERVAL '-2 days -11045 seconds'"),
    ];
    for (new, old) in same {
        let a = one(&mut db, &select_of(new));
        let b = one(&mut db, &select_of(old));
        assert_eq!(a, b, "{new} vs {old}");
    }
    // An interval also survives a round trip through arithmetic, which is the
    // point of accepting the printed `HH:MM:SS` shape at all.
    assert_eq!(
        one(&mut db, "SELECT DATE '2020-01-01' + INTERVAL '1 day 01:02:03' FROM t"),
        one(&mut db, "SELECT TIMESTAMP '2020-01-02 01:02:03' FROM t")
    );
}

fn select_of(expr: &str) -> String {
    let mut s = String::from("SELECT ");
    s.push_str(expr);
    s.push_str(" FROM t");
    s
}
