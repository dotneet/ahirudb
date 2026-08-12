//! Integration tests for `CURRENT_DATE`/`CURRENT_TIMESTAMP`/`CURRENT_TIME`/`now()`/`today()`.
//! Passes a fixed value via `Session::set_now`, verifying that `sql::now::substitute_now`
//! actually works correctly when going through the real `Session::prepare` path
//! (the unit tests in `crates/ahiru-core/src/sql/now.rs` only verify at the AST level).

use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

// 2024-01-15 12:30:00 UTC.
const NOW: i64 = 1_705_321_800_000_000;
const TODAY_DAYS: i32 = 19737;

fn session_with_dual() -> Session {
    let mut s = Session::new();
    s.register_bytes_as("dual", b"x\n1\n".to_vec(), FormatKind::Csv).unwrap();
    s.set_now(NOW);
    s
}

fn run(s: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    let mut q = match s.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
    };
    let mut rows = Vec::new();
    loop {
        match s.step(&mut q).unwrap() {
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

#[test]
fn bare_forms_and_call_forms_all_resolve_to_the_configured_now() {
    let mut s = session_with_dual();
    let rows = run(
        &mut s,
        "SELECT CURRENT_DATE, CURRENT_TIMESTAMP, now(), today(), current_time FROM dual",
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], Value::I32(TODAY_DAYS), "CURRENT_DATE");
    assert_eq!(rows[0][1], Value::I64(NOW), "CURRENT_TIMESTAMP");
    assert_eq!(rows[0][2], Value::I64(NOW), "now()");
    assert_eq!(rows[0][3], Value::I32(TODAY_DAYS), "today()");
    assert_eq!(rows[0][4], Value::I64(12 * 3_600_000_000 + 30 * 60_000_000), "current_time");
}

#[test]
fn current_timestamp_is_evaluated_once_per_query_not_per_row() {
    // The SQL standard contract: CURRENT_TIMESTAMP is evaluated exactly once at the start of
    // the query, staying the same value across multiple rows (not re-evaluated per row).
    let mut s = Session::new();
    s.register_bytes_as("t", b"id\n1\n2\n3\n".to_vec(), FormatKind::Csv).unwrap();
    s.set_now(NOW);
    let rows = run(&mut s, "SELECT id, CURRENT_TIMESTAMP FROM t ORDER BY id");
    assert_eq!(rows.len(), 3);
    for r in &rows {
        assert_eq!(r[1], Value::I64(NOW));
    }
}

#[test]
fn typed_literals_can_be_used_in_expressions() {
    let mut s = session_with_dual();
    // CURRENT_DATE plus an integer day count should work just like existing DATE arithmetic.
    let rows = run(&mut s, "SELECT CURRENT_DATE + 1 FROM dual");
    assert_eq!(rows[0][0], Value::I32(TODAY_DAYS + 1));
}

#[cfg(feature = "ddl")]
#[test]
fn view_body_sees_now_at_query_time() {
    let mut s = session_with_dual();
    s.prepare("CREATE VIEW v AS SELECT now() AS ts, CURRENT_DATE AS d FROM dual", &[]).unwrap();
    let rows = run(&mut s, "SELECT ts, d FROM v");
    assert_eq!(rows[0][0], Value::I64(NOW));
    assert_eq!(rows[0][1], Value::I32(TODAY_DAYS));
    let mut q = match s.prepare("DESCRIBE v", &[]).unwrap() {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("describe view"),
    };
    let mut n = 0usize;
    loop {
        match s.step(&mut q).unwrap() {
            QueryStep::Batch(b) => n += b.num_rows(),
            QueryStep::Done => break,
            _ => panic!("suspend"),
        }
    }
    assert_eq!(n, 2);
}

#[test]
fn unset_now_defaults_to_the_unix_epoch() {
    // If `set_now` is never called, it defaults to the epoch (1970-01-01)
    // (a default so a core with no clock doesn't silently return a fake time).
    let mut s = Session::new();
    s.register_bytes_as("dual", b"x\n1\n".to_vec(), FormatKind::Csv).unwrap();
    let rows = run(&mut s, "SELECT CURRENT_DATE FROM dual");
    assert_eq!(rows[0][0], Value::I32(0));
}

#[test]
fn a_real_column_named_current_date_is_shadowed_by_the_bare_keyword_form() {
    // A known tradeoff: `current_date`/`current_timestamp`/`current_time` are treated
    // unconditionally as keywords the moment they appear as a bare identifier
    // (they are effectively reserved words under the SQL standard, and we judged the odds of
    // real data having a column with the same name to be extremely low; see the module doc on
    // `sql/now.rs`). As a result, this test explicitly pins down that the function-call
    // interpretation wins even when a real column shares the name (so nobody gets confused
    // later by "why isn't the table's value coming back").
    let mut s = Session::new();
    s.register_bytes_as("t", b"current_date\n2000-01-01\n".to_vec(), FormatKind::Csv).unwrap();
    s.set_now(NOW);
    let rows = run(&mut s, "SELECT current_date FROM t");
    assert_eq!(rows[0][0], Value::I32(TODAY_DAYS), "becomes today's date, not the column's value");
}

#[test]
fn current_date_works_as_a_join_and_where_condition() {
    // Since `CURRENT_DATE`/`CURRENT_TIMESTAMP` are just replaced with a constant at prepare
    // time, they can be used normally as part of any expression in `WHERE`/`JOIN ON`
    // (combining with the existing JOIN/WHERE pipeline, rather than two new features together).
    let mut s = Session::new();
    s.register_bytes_as("t", b"id\n1\n2\n3\n".to_vec(), FormatKind::Csv).unwrap();
    s.set_now(NOW);
    let rows = run(
        &mut s,
        "SELECT a.id FROM t a JOIN t b ON a.id = b.id \
         WHERE CURRENT_DATE > CAST('2000-01-01' AS DATE) ORDER BY a.id",
    );
    assert_eq!(rows.len(), 3);
}

#[test]
fn current_timestamp_stays_constant_across_group_by_aggregation() {
    // Stays the same, once-evaluated value even after aggregation (not re-evaluated per group).
    let mut s = Session::new();
    s.register_bytes_as("t", b"id,g\n1,a\n2,a\n3,b\n".to_vec(), FormatKind::Csv).unwrap();
    s.set_now(NOW);
    let rows =
        run(&mut s, "SELECT g, count(*), max(CURRENT_TIMESTAMP) FROM t GROUP BY g ORDER BY g");
    assert_eq!(rows.len(), 2);
    for r in &rows {
        assert_eq!(r[2], Value::I64(NOW));
    }
}

#[test]
fn a_real_column_named_today_or_now_is_not_shadowed() {
    // `today`/`now` get no special treatment as bare identifiers without parentheses
    // (only the function forms `today()`/`now()` are targeted). A deliberate line drawn to
    // avoid colliding with real data's column names.
    let mut s = Session::new();
    s.register_bytes_as("t", b"today,now\nhello,world\n".to_vec(), FormatKind::Csv).unwrap();
    s.set_now(NOW);
    let rows = run(&mut s, "SELECT today, now FROM t");
    assert_eq!(rows[0][0], Value::Bytes(b"hello".to_vec()));
    assert_eq!(rows[0][1], Value::Bytes(b"world".to_vec()));
}
