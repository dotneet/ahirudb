//! End-to-end SQL coverage for the regex engine's UTF-8 handling, `^`/`$` anchoring under
//! global `regexp_replace`, and POSIX bracket expressions
//! (`crates/ahiru-core/src/expr/regex.rs`).
//!
//! These are the SQL-layer counterparts of the unit tests inside `expr::regex`; they also cover
//! `SIMILAR TO` / `~`, which reach the same engine through
//! `expr::funcs::json::regexp_full_match_build` and would not be exercised by unit tests on
//! `expr::regex` alone.
//!
//! Every expected value here was cross-checked against the `duckdb` CLI
//! (`duckdb -csv -c "SELECT ..."`).
//!
//! Like the other function tests in this directory, a one-row CSV table stands in for a
//! `SELECT` with no `FROM`.

use ahiru_core::error::{code_of, Code};
use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn session() -> Session {
    let mut s = Session::new();
    s.register_bytes_as("t", b"x\n1\n".to_vec(), FormatKind::Csv).unwrap();
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

/// `run` without the unwraps, for the cases that are supposed to fail. The failure can surface
/// either at `prepare` (constant folding) or at `step`, so both are funnelled into one `Result`.
fn try_run(session: &mut Session, sql: &str) -> ahiru_core::error::Result<()> {
    let mut q = match session.prepare(sql, &[])? {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
    };
    while !matches!(session.step(&mut q)?, QueryStep::Done) {}
    Ok(())
}

fn one(session: &mut Session, expr: &str) -> Value {
    run(session, &format!("SELECT {expr} AS r FROM t LIMIT 1")).remove(0).remove(0)
}

fn s(v: &str) -> Value {
    Value::Bytes(v.as_bytes().to_vec())
}

/// Asserts that a VARCHAR-returning expression produced exactly `want` *and* that the bytes are
/// valid UTF-8 — the engine used to be able to slice a multi-byte character in half and hand the
/// fragment back as a string value.
fn assert_str(sess: &mut Session, expr: &str, want: &str) {
    let got = one(sess, expr);
    let Value::Bytes(b) = &got else { panic!("{expr}: not a string: {got:?}") };
    assert!(core::str::from_utf8(b).is_ok(), "{expr}: result is not valid UTF-8: {b:?}");
    assert_eq!(got, s(want), "{expr}");
}

// =============================================================================
// `^` / `$` under global replacement
// =============================================================================

#[test]
fn caret_anchors_to_the_string_not_to_each_restart_position() {
    let mut sess = session();
    // duckdb: 'Xaa' — `^a` can only match once, however many times the scan restarts.
    assert_str(&mut sess, "regexp_replace('aaa', '^a', 'X', 'g')", "Xaa");
    // duckdb: 'Xabc'
    assert_str(&mut sess, "regexp_replace('abc', '^', 'X', 'g')", "Xabc");
    // duckdb: 'abcX' — `$` was already correct, and must stay so.
    assert_str(&mut sess, "regexp_replace('abc', '$', 'X', 'g')", "abcX");
    // duckdb: 'XabcX'
    assert_str(&mut sess, "regexp_replace('abc', '^|$', 'X', 'g')", "XabcX");
    // duckdb: 'Xbcabc'
    assert_str(&mut sess, "regexp_replace('abcabc', '^a', 'X', 'g')", "Xbcabc");
    // duckdb: 'abcabX'
    assert_str(&mut sess, "regexp_replace('abcabc', 'c$', 'X', 'g')", "abcabX");
    // The non-global form must not regress. duckdb: 'Xaa'
    assert_str(&mut sess, "regexp_replace('aaa', '^a', 'X')", "Xaa");
    // A plain global replacement still replaces every occurrence. duckdb: 'aYcaYc'
    assert_str(&mut sess, "regexp_replace('abcabc', 'b', 'Y', 'g')", "aYcaYc");
    // Adjacent empty matches are still skipped the way DuckDB skips them. duckdb: '-a-b-c-'
    assert_str(&mut sess, "regexp_replace('aXbXc', 'X*', '-', 'g')", "-a-b-c-");
}

// =============================================================================
// UTF-8: `.`, classes and quantifiers operate on characters
// =============================================================================

#[test]
fn dot_and_classes_match_whole_characters() {
    let mut sess = session();
    // duckdb: true / false
    assert_eq!(one(&mut sess, "regexp_matches('héllo', 'h.llo')"), Value::Bool(true));
    assert_eq!(one(&mut sess, "regexp_full_match('héllo', 'h..llo')"), Value::Bool(false));
    assert_eq!(one(&mut sess, "regexp_full_match('héllo', 'h.llo')"), Value::Bool(true));
    // A negated class must not match one byte of a multi-byte character. duckdb: true
    assert_eq!(one(&mut sess, "regexp_matches('héllo', 'h[^x]llo')"), Value::Bool(true));
    // Code-point ranges. duckdb: true / false
    assert_eq!(one(&mut sess, "regexp_matches('héllo', 'h[à-ÿ]llo')"), Value::Bool(true));
    assert_eq!(one(&mut sess, "regexp_matches('héllo', 'h[à-å]llo')"), Value::Bool(false));
    // Quantifiers apply to the whole character. duckdb: true / false
    assert_eq!(one(&mut sess, "regexp_matches('ééé', '^é{3}$')"), Value::Bool(true));
    assert_eq!(one(&mut sess, "regexp_matches('éé', '^é{3}$')"), Value::Bool(false));
    // `SIMILAR TO` and `~` ride the same engine. duckdb: true / true
    assert_eq!(one(&mut sess, "'héllo' SIMILAR TO 'h.llo'"), Value::Bool(true));
    assert_eq!(one(&mut sess, "'héllo' ~ 'h.llo'"), Value::Bool(true));
    // ASCII behavior must not regress.
    assert_eq!(one(&mut sess, "regexp_matches('hello', 'h.llo')"), Value::Bool(true));
    assert_eq!(one(&mut sess, "regexp_matches('hello', '^h[a-z]+o$')"), Value::Bool(true));
}

#[test]
fn results_are_always_valid_utf8() {
    let mut sess = session();
    // duckdb: 'xxxxx' — five characters in, five out (it used to emit six).
    assert_str(&mut sess, "regexp_replace('héllo', '.', 'x', 'g')", "xxxxx");
    // duckdb: 'Xllo' — this used to leave a lone UTF-8 continuation byte in the result.
    assert_str(&mut sess, "regexp_replace('héllo', 'h.', 'X')", "Xllo");
    // duckdb: '---'
    assert_str(&mut sess, "regexp_replace('日本語', '.', '-', 'g')", "---");
    // Advancing past a zero-length match copies whole characters through.
    // duckdb: 'XhXéXlXlXoX'
    assert_str(&mut sess, "regexp_replace('héllo', '', 'X', 'g')", "XhXéXlXlXoX");
    // duckdb: 'aZb'
    assert_str(&mut sess, "regexp_replace('aéb', '[^a-z]', 'Z', 'g')", "aZb");
    // Extraction slices on character boundaries too. duckdb: 'hé' / 'é' / '日本語'
    assert_str(&mut sess, "regexp_extract('héllo', 'h.')", "hé");
    assert_str(&mut sess, "regexp_extract('héllo', 'h(.)l', 1)", "é");
    assert_str(&mut sess, "regexp_extract('日本語abc', '[^a-z]+')", "日本語");
}

// =============================================================================
// POSIX bracket expressions
// =============================================================================

#[test]
fn posix_bracket_expressions() {
    let mut sess = session();
    // duckdb: all true
    for expr in [
        "regexp_matches('abc', '[[:alpha:]]+')",
        "regexp_matches('123', '^[[:digit:]]+$')",
        "regexp_matches('a1', '^[[:alnum:]]+$')",
        "regexp_matches(' ', '[[:space:]]')",
        "regexp_matches('A', '[[:upper:]]')",
        "regexp_matches('a', '[[:lower:]]')",
        "regexp_matches('!', '[[:punct:]]')",
        "regexp_matches('deadBEEF', '^[[:xdigit:]]+$')",
        "regexp_matches('a', '[[:^digit:]]')",
    ] {
        assert_eq!(one(&mut sess, expr), Value::Bool(true), "{expr}");
    }
    // POSIX classes are ASCII-only, as in RE2/DuckDB.
    // duckdb: regexp_matches('é','[[:alpha:]]') = false
    assert_eq!(one(&mut sess, "regexp_matches('é', '[[:alpha:]]')"), Value::Bool(false));
    assert_eq!(one(&mut sess, "regexp_matches('g', '^[[:xdigit:]]$')"), Value::Bool(false));
    // duckdb: regexp_extract('foo123bar','[[:digit:]]+') = '123'
    assert_str(&mut sess, "regexp_extract('foo123bar', '[[:digit:]]+')", "123");
    // A `[` that opens nothing stays a literal member. duckdb: true
    assert_eq!(one(&mut sess, "regexp_matches('[', '^[[]$')"), Value::Bool(true));
}

#[test]
fn unknown_posix_class_is_an_error_not_a_wrong_answer() {
    let mut sess = session();
    let sql = "SELECT regexp_matches('a', '[[:nope:]]') AS r FROM t";
    // A well-formed `[[:...:]]` naming a class this engine does not implement is rejected
    // outright. Reinterpreting it as the ordinary set `[:nope]` (which is what used to happen)
    // would answer the query with a plausible-looking wrong value instead.
    assert_eq!(code_of(try_run(&mut sess, sql)), Some(Code::UnsupportedFeature));
}

// =============================================================================
// No catastrophic backtracking
// =============================================================================

#[test]
fn adversarial_pattern_terminates_quickly() {
    let mut sess = session();
    let start = std::time::Instant::now();
    // The classic exponential-backtracking pattern. A Thompson NFA answers it in linear time.
    assert_eq!(
        one(&mut sess, "regexp_matches(repeat('a', 30) || '!', '(a*)*b')"),
        Value::Bool(false)
    );
    // The same shape with a multi-byte subject, which now costs one step per character.
    assert_eq!(one(&mut sess, "regexp_matches(repeat('é', 2000), '(.*)*x')"), Value::Bool(false));
    let elapsed = start.elapsed();
    assert!(elapsed.as_millis() < 5000, "took too long: {elapsed:?}");
}
