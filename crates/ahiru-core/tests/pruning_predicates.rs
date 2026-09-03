//! SQL-level verification that `WHERE`'s `IN`/`BETWEEN` predicates correctly go through
//! RowGroup/PageIndex/Bloom-filter pruning.
//!
//! `format::parquet`'s unit tests verify the byte reduction itself by hitting
//! `ParquetFormat` directly, but here we verify against real data that the execution path
//! through `Session` (bind -> pruner extraction -> execution) is wired correctly, especially
//! whether it has pruned too aggressively and dropped a row that should have hit. Expected
//! values are decided by cross-checking against the actual output of `duckdb -c "SELECT ..."`.
//!
//! `tests/data/pagetest.parquet` is a 50000-row file with multiple RowGroups/multiple pages
//! whose `id` densely fills `0..50000` (the same fixture used by `format::parquet`'s
//! per-page pruning tests).

use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn data(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn session_with_pagetest() -> Session {
    session_with("pagetest.parquet")
}

fn session_with(file: &str) -> Session {
    let mut s = Session::new();
    s.register_bytes_as("t", data(file), FormatKind::Parquet).unwrap();
    s
}

/// `count(*)` for one `WHERE` predicate.
fn count(file: &str, predicate: &str) -> i64 {
    let sql = format!("SELECT count(*) FROM t WHERE {predicate}");
    match run(&mut session_with(file), &sql).as_slice() {
        [row] => match row.as_slice() {
            [Value::I64(n)] => *n,
            v => panic!("{sql}: unexpected row {v:?}"),
        },
        v => panic!("{sql}: unexpected result {v:?}"),
    }
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

/// Registers with `register_remote_as` and drives it by `provide`-ing exactly the range each
/// `NeedIo` requests (the same trick as `sample.rs::run_id_with_lazy_io`). Also verifies
/// that driving `NeedIo` doesn't break even when pruning changes the byte fetch range.
fn run_with_lazy_io(bytes: &[u8], sql: &str) -> Vec<Vec<Value>> {
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
                    rows.push(b.cols.iter().map(|c| c.value_at(r)).collect());
                }
            }
            QueryStep::Done => break,
            QueryStep::NeedIo(reqs) => {
                rounds += 1;
                assert!(rounds < 1000, "step never finished");
                for r in reqs {
                    let (start, end) = (r.offset as usize, (r.offset + r.len) as usize);
                    s.provide(r.table, r.part, r.offset, bytes[start..end].to_vec()).unwrap();
                }
            }
            QueryStep::NeedCodec(_) => panic!("test fixtures are uncompressed"),
        }
    }
    rows
}

#[test]
fn in_list_finds_the_one_present_value_among_absent_decoys() {
    let mut s = session_with_pagetest();
    // duckdb: SELECT id FROM 'pagetest.parquet' WHERE id IN (12345, 999999999, -1) ORDER BY id
    let rows = run(&mut s, "SELECT id FROM t WHERE id IN (12345, 999999999, -1) ORDER BY id");
    assert_eq!(rows, vec![vec![Value::I32(12345)]]);
}

#[test]
fn in_list_with_multiple_present_values_returns_all_of_them() {
    let mut s = session_with_pagetest();
    let rows = run(&mut s, "SELECT id FROM t WHERE id IN (10, 20, 12345, 40000) ORDER BY id");
    assert_eq!(
        rows,
        vec![
            vec![Value::I32(10)],
            vec![Value::I32(20)],
            vec![Value::I32(12345)],
            vec![Value::I32(40000)],
        ]
    );
}

#[test]
fn not_in_is_unaffected_by_the_new_pruning_path() {
    let mut s = session_with_pagetest();
    // duckdb: SELECT count(*) FROM 'pagetest.parquet' WHERE id NOT IN (10,20,30) -> 49997
    let rows = run(&mut s, "SELECT count(*) FROM t WHERE id NOT IN (10, 20, 30)");
    assert_eq!(rows, vec![vec![Value::I64(49997)]]);
}

#[test]
fn between_end_to_end_matches_duckdb() {
    let mut s = session_with_pagetest();
    // duckdb: SELECT count(*) FROM 'pagetest.parquet' WHERE id BETWEEN 12000 AND 12010 -> 11
    let rows = run(&mut s, "SELECT count(*) FROM t WHERE id BETWEEN 12000 AND 12010");
    assert_eq!(rows, vec![vec![Value::I64(11)]]);
}

#[test]
fn in_list_result_is_identical_whether_or_not_pruning_narrows_the_io() {
    let bytes = data("pagetest.parquet");
    let sql = "SELECT id FROM t WHERE id IN (5, 12345, 39999, 999999999) ORDER BY id";

    let mut eager = Session::new();
    eager.register_bytes_as("t", bytes.clone(), FormatKind::Parquet).unwrap();
    let want = run(&mut eager, sql);
    assert_eq!(want, vec![vec![Value::I32(5)], vec![Value::I32(12345)], vec![Value::I32(39999)],]);

    let got = run_with_lazy_io(&bytes, sql);
    assert_eq!(got, want, "the IN-pruning result must not change across a NeedIo");
}

#[test]
fn in_list_on_a_non_literal_candidate_still_returns_correct_rows_without_pruning() {
    // A case where a non-literal (a column reference) is mixed into the candidates. No
    // pruner gets built (confirmed by
    // `plan::bind::tests::in_list_with_non_literal_element_is_not_pruned`), but the actual result must still be correct.
    let mut s = session_with_pagetest();
    let rows = run(&mut s, "SELECT id FROM t WHERE id IN (12345, id) ORDER BY id LIMIT 3");
    assert_eq!(rows, vec![vec![Value::I32(0)], vec![Value::I32(1)], vec![Value::I32(2)]]);
}

// --- DECIMAL columns: the literal has to be rescaled before it can be pruned with ---------
//
// `tests/data/decimal_pruning.parquet` is 800 rows in 4 RowGroups, written by DuckDB, with
// `d1 DECIMAL(5,1)` (physically INT32), `d2 DECIMAL(15,2)` (INT64) and `d DATE`. DuckDB
// writes a Bloom filter for the dictionary-encoded columns, so both the statistics path and
// the Bloom path are exercised. Expected values come from
// `duckdb -c "SELECT count(*) FROM 'decimal_pruning.parquet' WHERE ..."`.

const DEC: &str = "decimal_pruning.parquet";

#[test]
fn decimal_equality_against_an_integer_literal_keeps_its_rows() {
    // The column stores 1500 for 150.0; comparing the raw literal 150 against the statistics
    // (or hashing it into the Bloom filter) used to drop every RowGroup.
    assert_eq!(count(DEC, "d1 = 150"), 8);
    assert_eq!(count(DEC, "d1 + 0 = 150"), 8, "the answer must not depend on pruning");
    assert_eq!(count(DEC, "d2 = 3000000050"), 8);
    assert_eq!(count(DEC, "d2 + 0 = 3000000050"), 8);
}

#[test]
fn decimal_range_and_in_predicates_keep_their_rows() {
    assert_eq!(count(DEC, "d2 < 3000000050"), 400);
    assert_eq!(count(DEC, "d2 + 0 < 3000000050"), 400);
    assert_eq!(count(DEC, "d1 IN (150, 160)"), 16);
    assert_eq!(count(DEC, "d1 + 0 IN (150, 160)"), 16);
    assert_eq!(count(DEC, "d1 BETWEEN 150 AND 160"), 88);
    assert_eq!(count(DEC, "d1 + 0 BETWEEN 150 AND 160"), 88);
    assert_eq!(count(DEC, "d1 >= 195"), 42);
    assert_eq!(count(DEC, "d1 + 0 >= 195"), 42);
}

#[test]
fn decimal_against_a_fractional_literal_is_still_correct_without_a_pruner() {
    // No exact representation at the column's scale, so the pruner is dropped; the result
    // still has to be right.
    assert_eq!(count(DEC, "d1 > 150.5"), 392);
    assert_eq!(count(DEC, "d1 + 0 > 150.5"), 392);
    assert_eq!(count(DEC, "d1 = 150.5"), 0);
}

#[test]
fn typed_date_literals_prune_without_changing_the_answer() {
    // `DATE '...'` parses to `Expr::TypedLiteral`, which used to produce no pruner at all
    // (`plan::bind::tests::typed_date_literal_produces_a_pruner` covers that half).
    assert_eq!(count(DEC, "d > DATE '2024-06-01'"), 494);
    assert_eq!(count(DEC, "d = DATE '2024-06-01'"), 2);
    assert_eq!(count(DEC, "d BETWEEN DATE '2024-02-01' AND DATE '2024-02-10'"), 20);
    assert_eq!(count(DEC, "d IN (DATE '2024-02-01', DATE '2024-02-02')"), 4);
}

// --- FLOAT/DOUBLE columns: NaN sits outside the writer's min/max --------------------------
//
// `tests/data/nan_stats.parquet` is 800 rows in 4 RowGroups written by pyarrow, with one NaN
// per RowGroup among values 0.0..9.0. pyarrow (like most writers) leaves NaN out of the
// statistics, while this engine orders NaN above every other value, so `> x`/`>= x` must not
// prune on `max`. Expected values come from `duckdb -c "... WHERE d + 0 ..."`; DuckDB's own
// pruned answers on this file differ from its unpruned ones, so the pruned form is not a
// usable reference here.

const NANF: &str = "nan_stats.parquet";

#[test]
fn nan_rows_survive_greater_than_pruning() {
    assert_eq!(count(NANF, "d > 100.0"), 4);
    assert_eq!(count(NANF, "d + 0 > 100.0"), 4);
    assert_eq!(count(NANF, "d >= 100.0"), 4);
    assert_eq!(count(NANF, "d + 0 >= 100.0"), 4);
    assert_eq!(count(NANF, "d > 5.0"), 324);
    assert_eq!(count(NANF, "d + 0 > 5.0"), 324);
    assert_eq!(count(NANF, "d >= 5.0"), 404);
    assert_eq!(count(NANF, "d + 0 >= 5.0"), 404);
}

#[test]
fn a_part_whose_decimal_scale_differs_from_the_table_is_not_pruned() {
    // `decimal_scale3.parquet` holds the same columns as `decimal_pruning.parquet` but with
    // `d1` at DECIMAL(8,3); the union reads back as DECIMAL(9,3). A pruner's constant is
    // scaled into the *table's* type, so against the DECIMAL(5,1) part it would be 1000x
    // too large -- that part has to be read in full rather than pruned.
    let mut s = Session::new();
    s.register_multi_bytes(
        "t",
        vec![
            ("decimal_pruning.parquet".into(), data("decimal_pruning.parquet")),
            ("decimal_scale3.parquet".into(), data("decimal_scale3.parquet")),
        ],
        FormatKind::Parquet,
    )
    .unwrap();
    // duckdb: SELECT count(*) FROM read_parquet(['decimal_pruning.parquet',
    //         'decimal_scale3.parquet']) WHERE d1 = 150  ->  8 + 2
    let counts: Vec<i64> = ["d1 = 150", "d1 > 195", "d1 IN (150, 160)", "d1 BETWEEN 150 AND 160"]
        .iter()
        .map(|p| match run(&mut s, &format!("SELECT count(*) FROM t WHERE {p}")).as_slice() {
            [row] => match row.as_slice() {
                [Value::I64(n)] => *n,
                v => panic!("unexpected row {v:?}"),
            },
            v => panic!("unexpected result {v:?}"),
        })
        .collect();
    assert_eq!(counts, vec![10, 43, 20, 110]);
}

#[test]
fn nan_does_not_disable_the_other_float_pruners() {
    // NaN is never `<`, `<=` or `=` a real number, so those keep pruning normally.
    assert_eq!(count(NANF, "d < 5.0"), 396);
    assert_eq!(count(NANF, "d + 0 < 5.0"), 396);
    assert_eq!(count(NANF, "d = 5.0"), 80);
    assert_eq!(count(NANF, "d + 0 = 5.0"), 80);
    assert_eq!(count(NANF, "d IN (5.0, 6.0)"), 160);
    assert_eq!(count(NANF, "d + 0 IN (5.0, 6.0)"), 160);
}
