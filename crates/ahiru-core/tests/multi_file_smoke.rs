//! A manual smoke check for multiple files / Hive partitions.
//!
//! The agent should already have written unit tests on the `catalog.rs`/`session.rs` side,
//! so here we keep it to a simple check of "does this work against real data from the
//! coordinator's perspective".

use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn read(rel: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../");
    std::fs::read(format!("{p}{rel}")).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

fn run_all(sql: &str, s: &mut Session) -> Vec<Vec<Value>> {
    let mut q = match s.prepare(sql, &[]).unwrap() {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("NeedIo should not happen for in-memory data"),
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
            QueryStep::NeedIo(_) | QueryStep::NeedCodec(_) => panic!("unexpected suspend"),
            QueryStep::Done => break,
        }
    }
    rows
}

#[test]
fn three_plain_files_union_into_one_table() {
    let mut s = Session::new();
    s.register_multi_bytes(
        "t",
        vec![
            ("a.parquet".into(), read("tests/data/multi/a.parquet")),
            ("b.parquet".into(), read("tests/data/multi/b.parquet")),
            ("c.parquet".into(), read("tests/data/multi/c.parquet")),
        ],
        FormatKind::Parquet,
    )
    .unwrap();
    // All 3 files share the same column names and order: (id INTEGER, name VARCHAR).
    // 100 + 150 + 230 rows.
    let rows = run_all("SELECT count(*) AS n FROM t", &mut s);
    assert_eq!(rows, [[Value::I64(480)]]);
}

#[test]
fn parts_with_mismatched_column_names_are_rejected_not_silently_merged() {
    // small_a.parquet: (k, v) / small_b.parquet: (k, w). The 2nd column's name differs
    // (both happen to be type-compatible as INTEGER), so aligning by position alone would
    // silently merge columns with different meanings into one. `catalog::unify_schema`
    // also requires column names to match by position, so this should be a clear error.
    let mut s = Session::new();
    s.register_multi_bytes(
        "t",
        vec![
            ("a.parquet".into(), read("tests/data/small_a.parquet")),
            ("b.parquet".into(), read("tests/data/small_b.parquet")),
        ],
        FormatKind::Parquet,
    )
    .unwrap();
    let r = s.prepare("SELECT count(*) FROM t", &[]);
    assert_eq!(
        ahiru_core::error::code_of(r),
        Some(ahiru_core::error::Code::TypeMismatch),
        "a part with a mismatched column name should be clearly rejected"
    );
}

#[test]
fn hive_partitioned_files_expose_partition_columns() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/hive");
    if !std::path::Path::new(dir).exists() {
        eprintln!("skipping: no hive fixture present");
        return;
    }
    let mut s = Session::new();
    s.register_multi_bytes(
        "t",
        vec![
            (
                format!("{dir}/year=2024/month=01/part.parquet"),
                std::fs::read(format!("{dir}/year=2024/month=01/part.parquet")).unwrap(),
            ),
            (
                format!("{dir}/year=2024/month=02/part.parquet"),
                std::fs::read(format!("{dir}/year=2024/month=02/part.parquet")).unwrap(),
            ),
            (
                format!("{dir}/year=2025/month=01/part.parquet"),
                std::fs::read(format!("{dir}/year=2025/month=01/part.parquet")).unwrap(),
            ),
        ],
        FormatKind::Parquet,
    )
    .unwrap();

    let total = run_all("SELECT count(*) AS n FROM t", &mut s);
    assert_eq!(total[0][0], Value::I64(1000));

    // `month` is VARCHAR: the directories spell it `month=01`, and a zero-padded value read as a
    // number would lose its padding. `year` has none and stays INTEGER.
    let filtered =
        run_all("SELECT count(*) AS n FROM t WHERE year = 2024 AND month = '01'", &mut s);
    assert_eq!(filtered[0][0], Value::I64(300));
}
