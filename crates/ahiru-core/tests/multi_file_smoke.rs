//! 複数ファイル / Hive パーティションの手動疎通確認。
//!
//! ユニットテストはエージェントが `catalog.rs`/`session.rs` 側に書いている
//! はずなので、ここでは「コーディネータの視点で実データに対して動くか」を
//! 素朴に確認するだけに留める。

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
        Prepared::NeedIo(_) => panic!("メモリ上のデータで NeedIo は出ないはず"),
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
    // 3 ファイルとも (id INTEGER, name VARCHAR) で列名・並びが揃っている。
    // 100 + 150 + 230 行。
    let rows = run_all("SELECT count(*) AS n FROM t", &mut s);
    assert_eq!(rows, [[Value::I64(480)]]);
}

#[test]
fn parts_with_mismatched_column_names_are_rejected_not_silently_merged() {
    // small_a.parquet: (k, v) / small_b.parquet: (k, w)。2 列目の名前が違う
    // （型はどちらも INTEGER で偶然両立してしまう）ので、位置だけで揃えると
    // 意味の違う列を静かに 1 列として merge してしまう。`catalog::unify_schema`
    // は列名の位置一致も要求するので、ここは明確なエラーになるべき。
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
        "列名が食い違うパートは明確に拒否されるべき"
    );
}

#[test]
fn hive_partitioned_files_expose_partition_columns() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/hive");
    if !std::path::Path::new(dir).exists() {
        eprintln!("hive フィクスチャが無いので飛ばす");
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

    let filtered = run_all("SELECT count(*) AS n FROM t WHERE year = 2024 AND month = 1", &mut s);
    assert_eq!(filtered[0][0], Value::I64(300));
}
