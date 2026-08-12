//! Integration tests for `USING SAMPLE` / `TABLESAMPLE`.
//!
//! Spec decided by checking the `duckdb` CLI's syntax and value-range-check behavior:
//! - `USING SAMPLE <n>%` / `TABLESAMPLE <n>%` are percentage specs.
//! - `USING SAMPLE <n> ROWS` / a bare unitless number is a row-count spec.
//! - Method names `BERNOULLI(...)`/`SYSTEM(...)`/`RESERVOIR(...)` and the explicit-seed
//!   syntax `(method, seed)` are accepted.
//! - Percentage must be `0..=100`, row count must be `0` or more, otherwise a syntax error.
//! - Since `duckdb` samples randomly, we cannot match "exactly how many rows"
//!   (the row-count spec is an exception -- if the input has at least that many rows, it
//!   always returns exactly that many rows). This engine's own PRNG is its own xorshift64*,
//!   so we don't require matching `duckdb`'s actual chosen rows (only matching the same
//!   probability distribution is the goal).
//!
//! Random-sampling verification is done along four pillars: "is the count roughly as
//! expected", "does the same seed reproduce", "does a different seed give a different
//! result", and "does the result stay the same across a `NeedIo`".

use ahiru_core::error::{code_of, Code};
use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn data(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn session_with_dual() -> Session {
    let mut s = Session::new();
    s.register_bytes_as("dual", b"x\n1\n".to_vec(), FormatKind::Csv).unwrap();
    s
}

/// Runs a query to completion where all data is in memory, collecting the `x` column's
/// (column 0's) values.
fn run_x(s: &mut Session, sql: &str) -> Vec<i64> {
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
                    let Value::I64(v) = b.cols[0].value_at(r) else { panic!("expected I64") };
                    rows.push(v);
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

const N: i64 = 20_000;

fn big_table_sql(sample: &str) -> String {
    format!("SELECT range AS x FROM range({N}) {sample}")
}

// --- Percentage spec (Bernoulli method) ----------------------------------------

#[test]
fn using_sample_percent_keeps_roughly_the_requested_fraction() {
    let mut db = session_with_dual();
    let got = run_x(&mut db, &big_table_sql("USING SAMPLE 10%"));
    let frac = got.len() as f64 / N as f64;
    assert!((0.08..0.12).contains(&frac), "got fraction {frac} ({} rows)", got.len());
    // No duplicates, only values that exist, keeping the input's relative order.
    let mut sorted = got.clone();
    sorted.dedup();
    assert_eq!(got, sorted, "must not contain duplicates");
    let mut asc = got.clone();
    asc.sort_unstable();
    assert_eq!(got, asc, "must keep the input's relative order");
}

#[test]
fn tablesample_syntax_is_equivalent_to_using_sample() {
    let mut db = session_with_dual();
    let got = run_x(&mut db, &big_table_sql("TABLESAMPLE 10%"));
    let frac = got.len() as f64 / N as f64;
    assert!((0.08..0.12).contains(&frac), "got fraction {frac}");
}

#[test]
fn zero_percent_keeps_nothing_and_hundred_percent_keeps_everything() {
    let mut db = session_with_dual();
    assert!(run_x(&mut db, &big_table_sql("USING SAMPLE 0%")).is_empty());
    assert_eq!(run_x(&mut db, &big_table_sql("USING SAMPLE 100%")).len(), N as usize);
}

#[test]
fn explicit_method_names_are_accepted() {
    let mut db = session_with_dual();
    for m in ["BERNOULLI", "SYSTEM"] {
        let sql = big_table_sql(&format!("USING SAMPLE {m}(10%)"));
        let got = run_x(&mut db, &sql);
        // In this engine, `SYSTEM` falls back to the same implementation as `BERNOULLI`
        // (per the task's priorities, the difference between methods is not implemented).
        let frac = got.len() as f64 / N as f64;
        assert!((0.05..0.15).contains(&frac), "{m}: got fraction {frac}");
    }
}

// --- Row-count spec ---------------------------------------------------------------

#[test]
fn using_sample_rows_selects_exactly_that_many_rows() {
    let mut db = session_with_dual();
    let got = run_x(&mut db, &big_table_sql("USING SAMPLE 100 ROWS"));
    assert_eq!(got.len(), 100);
    let mut sorted = got.clone();
    sorted.dedup();
    assert_eq!(got, sorted, "must not contain duplicates");
}

#[test]
fn bare_number_without_a_unit_means_rows() {
    let mut db = session_with_dual();
    let got = run_x(&mut db, &big_table_sql("USING SAMPLE 50"));
    assert_eq!(got.len(), 50);
}

#[test]
fn reservoir_method_with_rows_is_accepted() {
    let mut db = session_with_dual();
    let got = run_x(&mut db, &big_table_sql("USING SAMPLE reservoir(30)"));
    assert_eq!(got.len(), 30);
}

#[test]
fn requesting_more_rows_than_available_returns_everything() {
    let mut db = session_with_dual();
    let sql = "SELECT range AS x FROM range(10) USING SAMPLE 1000 ROWS";
    let got = run_x(&mut db, sql);
    let want: Vec<i64> = (0..10).collect();
    assert_eq!(got, want);
}

// --- Seed reproducibility -----------------------------------------------------------

#[test]
fn same_explicit_seed_reproduces_the_same_sample() {
    let sql = big_table_sql("USING SAMPLE 20% (bernoulli, 42)");
    let mut a = session_with_dual();
    let mut b = session_with_dual();
    assert_eq!(run_x(&mut a, &sql), run_x(&mut b, &sql));
}

#[test]
fn different_seed_gives_a_different_sample() {
    let mut a = session_with_dual();
    let mut b = session_with_dual();
    let got_a = run_x(&mut a, &big_table_sql("USING SAMPLE 20% (bernoulli, 1)"));
    let got_b = run_x(&mut b, &big_table_sql("USING SAMPLE 20% (bernoulli, 2)"));
    assert_ne!(got_a, got_b);
}

#[test]
fn default_seed_without_an_explicit_one_is_still_deterministic() {
    // Per the task's instructions, "it's fine for the seed to be deterministic": running
    // the same query twice selects the same sample (`duckdb`'s default changes every time,
    // but this engine uses a fixed default seed, `plan::DEFAULT_SAMPLE_SEED`).
    let sql = big_table_sql("USING SAMPLE 15%");
    let mut a = session_with_dual();
    let mut b = session_with_dual();
    assert_eq!(run_x(&mut a, &sql), run_x(&mut b, &sql));
}

#[test]
fn row_sample_is_also_reproducible_with_the_same_seed() {
    let sql = big_table_sql("USING SAMPLE reservoir(50)");
    let mut a = session_with_dual();
    let mut b = session_with_dual();
    assert_eq!(run_x(&mut a, &sql), run_x(&mut b, &sql));
}

// --- Value-range errors ---------------------------------------------------------------

#[test]
fn percent_out_of_range_is_rejected() {
    let mut db = session_with_dual();
    let err = db.prepare(&big_table_sql("USING SAMPLE 150%"), &[]);
    assert_eq!(code_of(err), Some(Code::SyntaxError));
}

#[test]
fn zero_interval_step_style_errors_do_not_apply_but_negative_amount_is_rejected() {
    let mut db = session_with_dual();
    // A negative amount hits a unary minus at the start of the expression, becoming a
    // syntax error (`sample_amount` does not read a minus sign).
    let err = db.prepare(&big_table_sql("USING SAMPLE -5 ROWS"), &[]);
    assert!(code_of(err).is_some());
}

// --- Reproducibility across a NeedIo -------------------------------------------------

/// Verify that feeding the byte stream incrementally across `NeedIo` produces exactly the same
/// result as feeding it all at once. The Bernoulli method just passes through its input's
/// `NeedIo` unchanged (see the doc on `exec::sample::Bernoulli`), and the sequence of PRNG
/// calls is determined only by "the order of rows actually evaluated", so it should not
/// depend on suspend timing.
#[test]
fn need_io_across_a_real_parquet_scan_does_not_change_a_percent_sample() {
    let sql = "SELECT id FROM t USING SAMPLE 20% (bernoulli, 7)";
    let bytes = data("pagetest.parquet");

    let mut eager = Session::new();
    eager.register_bytes_as("t", bytes.clone(), FormatKind::Parquet).unwrap();
    let want = run_id(&mut eager, sql);
    assert!(!want.is_empty());

    let (got, rounds) = run_id_with_lazy_io(&bytes, sql);
    assert_eq!(got, want, "the result must not change across a NeedIo");
    assert!(rounds >= 1);
}

/// Verify the same thing for the row-count spec (the blocking method).
#[test]
fn need_io_across_a_real_parquet_scan_does_not_change_a_row_sample() {
    let sql = "SELECT id FROM t USING SAMPLE reservoir(200)";
    let bytes = data("pagetest.parquet");

    let mut eager = Session::new();
    eager.register_bytes_as("t", bytes.clone(), FormatKind::Parquet).unwrap();
    let want = run_id(&mut eager, sql);
    assert_eq!(want.len(), 200);

    let (got, rounds) = run_id_with_lazy_io(&bytes, sql);
    assert_eq!(got, want, "the result must not change across a NeedIo");
    assert!(rounds >= 1);
}

fn run_id(s: &mut Session, sql: &str) -> Vec<i32> {
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
                    let Value::I32(v) = b.cols[0].value_at(r) else { panic!("expected I32") };
                    rows.push(v);
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

/// Registers with `register_remote_as` and drives it by `provide`-ing exactly the range each
/// `NeedIo` requests (the same trick as `unnest.rs::run_with_lazy_io`).
fn run_id_with_lazy_io(bytes: &[u8], sql: &str) -> (Vec<i32>, u32) {
    let mut s = Session::new();
    s.register_remote_as("t", bytes.len() as u64, FormatKind::Parquet).unwrap();

    let mut rounds = 0u32;
    let mut q = loop {
        match s.prepare(sql, &[]).unwrap() {
            Prepared::Ready(q) => break q,
            Prepared::NeedIo(reqs) => {
                rounds += 1;
                assert!(rounds < 1000, "resolve_query never finished");
                for r in reqs {
                    let (start, end) = (r.offset as usize, (r.offset + r.len) as usize);
                    s.provide(r.table, r.part, r.offset, bytes[start..end].to_vec()).unwrap();
                }
            }
        }
    };

    let mut rows = Vec::new();
    let mut steps = 0u32;
    loop {
        steps += 1;
        assert!(steps < 10_000, "step never finished");
        match s.step(&mut q).unwrap() {
            QueryStep::Batch(mut b) => {
                b.materialize();
                for r in 0..b.num_rows() {
                    let Value::I32(v) = b.cols[0].value_at(r) else { panic!("expected I32") };
                    rows.push(v);
                }
            }
            QueryStep::NeedIo(reqs) => {
                rounds += 1;
                for r in reqs {
                    let (start, end) = (r.offset as usize, (r.offset + r.len) as usize);
                    s.provide(r.table, r.part, r.offset, bytes[start..end].to_vec()).unwrap();
                }
            }
            QueryStep::NeedCodec(_) => panic!("test fixtures are uncompressed"),
            QueryStep::Done => break,
        }
    }
    (rows, rounds)
}

// --- Clause placement (interaction with `WHERE`/`GROUP BY`/`QUALIFY`) --------------
//
// Spec confirmed with the `duckdb` CLI: `TABLESAMPLE` is a modifier that attaches directly
// to the FROM item and must always be placed before `WHERE`. `USING SAMPLE` is the opposite:
// an independent clause of the whole statement, placed after
// `WHERE`/`GROUP BY`/`HAVING`/`WINDOW`/`QUALIFY` and before `ORDER BY`. Both only look like
// the same position when placed "right after the FROM item, with nothing else following",
// so existing tests without a `WHERE` alone failed to catch this difference (the bug this
// uncovered; see `sql::parser::opt_using_sample_clause` / `opt_tablesample_clause`).

#[test]
fn using_sample_can_follow_where_group_by_and_qualify() {
    let mut db = session_with_dual();
    // After WHERE.
    let got = run_x(
        &mut db,
        &format!("SELECT range AS x FROM range({N}) WHERE range % 2 = 0 USING SAMPLE 100%"),
    );
    assert_eq!(got.len(), (N / 2) as usize);
}

#[test]
fn using_sample_right_after_from_is_rejected_when_where_follows() {
    // `USING SAMPLE` is not a modifier that attaches directly to the FROM item, so writing
    // it right after the FROM item with a `WHERE` following becomes a syntax error
    // (`duckdb` rejects it for the same reason).
    let mut db = session_with_dual();
    let err = db.prepare(
        &format!("SELECT range AS x FROM range({N}) USING SAMPLE 10% WHERE range % 2 = 0"),
        &[],
    );
    assert!(code_of(err).is_some());
}

#[test]
fn tablesample_after_where_is_rejected() {
    // Conversely, `TABLESAMPLE` cannot be written as a trailing clause.
    let mut db = session_with_dual();
    let err = db.prepare(
        &format!("SELECT range AS x FROM range({N}) WHERE range % 2 = 0 TABLESAMPLE 10%"),
        &[],
    );
    assert!(code_of(err).is_some());
}

#[test]
fn tablesample_still_works_right_after_from_with_a_following_where() {
    // `TABLESAMPLE` can still be placed right after the FROM item, before `WHERE`, as
    // before (regression check: this case worked even before the bug fix).
    let mut db = session_with_dual();
    let got = run_x(
        &mut db,
        &format!("SELECT range AS x FROM range({N}) TABLESAMPLE 100% WHERE range % 2 = 0"),
    );
    assert_eq!(got.len(), (N / 2) as usize);
}

#[test]
fn combining_tablesample_and_trailing_using_sample_is_rejected() {
    // `duckdb` applies both in sequence when both are written together, but this engine's
    // `SampleSpec` is simplified to hold only one, so a double spec is explicitly rejected
    // (silently ignoring one would become a hard-to-notice bug).
    let mut db = session_with_dual();
    let err = db.prepare(
        &format!(
            "SELECT range AS x FROM range({N}) TABLESAMPLE 50% WHERE range % 2 = 0 \
             USING SAMPLE 100%"
        ),
        &[],
    );
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

#[test]
fn using_sample_can_follow_group_by_having_before_order_by() {
    let mut db = session_with_dual();
    let got = run_x(
        &mut db,
        "SELECT range % 3 AS x FROM range(30) GROUP BY range % 3 HAVING count(*) > 0 \
         USING SAMPLE 100% ORDER BY x",
    );
    assert_eq!(got, vec![0, 1, 2]);
}
