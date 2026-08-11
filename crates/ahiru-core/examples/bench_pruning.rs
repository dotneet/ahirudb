//! IN / BETWEEN プルーニングのロードテスト。
//!
//! `crates/ahiru-core/src/format/mod.rs` の `PruneOp::In` 追加（IN リストを
//! RowGroup 統計・PageIndex・Bloom フィルタのプルーニングに乗せる修正）が、
//! 実際にどれだけバイト転送量を減らすかを大きめの Parquet ファイルで測る。
//!
//! `IN` は修正前、`extract_pruners`（`plan/bind.rs`）が `Expr::InList` を
//! 一切扱っておらず pruner が 1 つも作られなかった。つまり修正前の
//! `WHERE id IN (...)` は常に「述語なし」（列チャンク全体を読む）と同じ
//! バイト量になっていたはず。ここでは「述語なし」の実測値を「修正前の
//! IN」の代理として使い、「修正後の IN」と比較する。
//!
//! 使い方:
//! ```sh
//! cargo run --release --example bench_pruning -- /path/to/large.parquet
//! ```
//! 引数を省略すると `/tmp/ahiru_bench.parquet` を探す。

use std::time::Instant;

use ahiru_core::catalog::Source;
use ahiru_core::format::{PruneOp, Pruner, TableFormat};
use ahiru_core::vector::Value;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/ahiru_bench.parquet".to_string());
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("failed to read {path}: {e}");
        eprintln!("generate one first, e.g. via pyarrow (see scripts/gen-testdata.sh)");
        std::process::exit(1);
    });
    println!("dataset: {path} ({} bytes)", bytes.len());

    let full_scan = run_case(&bytes, "predicate なし（全件走査、旧 IN の代理）", &[]);

    run_case(
        &bytes,
        "id = X（等号、修正前から効いていた）",
        &[Pruner {
            column: 0,
            op: PruneOp::Eq,
            value: Value::I32(1_234_567),
            in_values: Vec::new(),
        }],
    );

    run_case(
        &bytes,
        "id BETWEEN a AND b（範囲、修正前から効いていた）",
        &[
            Pruner {
                column: 0,
                op: PruneOp::Ge,
                value: Value::I32(1_000_000),
                in_values: Vec::new(),
            },
            Pruner {
                column: 0,
                op: PruneOp::Le,
                value: Value::I32(1_000_100),
                in_values: Vec::new(),
            },
        ],
    );

    let in_small = run_case(
        &bytes,
        "id IN (少数 4 件) ← 今回の修正",
        &[Pruner {
            column: 0,
            op: PruneOp::In,
            value: Value::I32(100),
            in_values: vec![Value::I32(500_000), Value::I32(1_000_000), Value::I32(1_800_000)],
        }],
    );

    // ファイル全体に散らばる 50 件（ほぼ全 RowGroup に 1 つは当たる、
    // 選択性が低い最悪ケース）。
    let scattered: Vec<Value> = (0..50).map(|i| Value::I32(i * 37_000 + 7)).collect();
    let (first, rest) = scattered.split_first().unwrap();
    let in_scattered = run_case(
        &bytes,
        "id IN (ファイル全体に散らばる 50 件、低選択性) ← 今回の修正",
        &[Pruner { column: 0, op: PruneOp::In, value: first.clone(), in_values: rest.to_vec() }],
    );

    // 1 RowGroup（25 万行）の中に収まる 50 件（バッチ処理・特定顧客の注文
    // ID 群のような、より典型的な IN 述語）。
    let clustered: Vec<Value> = (0..50).map(|i| Value::I32(1_000_000 + i * 400)).collect();
    let (first, rest) = clustered.split_first().unwrap();
    let in_clustered = run_case(
        &bytes,
        "id IN (1 RowGroup 内に収まる 50 件、高選択性) ← 今回の修正",
        &[Pruner { column: 0, op: PruneOp::In, value: first.clone(), in_values: rest.to_vec() }],
    );

    println!();
    println!("=== まとめ ===");
    report_speedup("IN (少数 4 件)", &full_scan, &in_small);
    report_speedup("IN (散らばった 50 件、低選択性)", &full_scan, &in_scattered);
    report_speedup("IN (1 RowGroup 内の 50 件、高選択性)", &full_scan, &in_clustered);
}

struct CaseResult {
    bytes_fetched: u64,
    rows_matched: u64,
    elapsed_us: u128,
}

fn report_speedup(label: &str, baseline: &CaseResult, fixed: &CaseResult) {
    let byte_ratio = baseline.bytes_fetched as f64 / fixed.bytes_fetched.max(1) as f64;
    let time_ratio = baseline.elapsed_us as f64 / fixed.elapsed_us.max(1) as f64;
    println!(
        "{label}: バイト削減 {byte_ratio:.1}x（{} -> {}）、所要時間 {time_ratio:.1}x（{} us -> {} us）、走査行数 {} -> {}",
        baseline.bytes_fetched,
        fixed.bytes_fetched,
        baseline.elapsed_us,
        fixed.elapsed_us,
        baseline.rows_matched,
        fixed.rows_matched,
    );
}

fn run_case(bytes: &[u8], label: &str, pruners: &[Pruner]) -> CaseResult {
    use ahiru_core::format::parquet::ParquetFormat;

    let src = Source::from_bytes(bytes.to_vec());
    let mut fmt = ParquetFormat::new();
    fmt.resolve(&src).unwrap().unwrap();

    let projection = vec![0usize]; // id 列だけ射影
    let start = Instant::now();
    let mut bytes_fetched: u64 = 0;
    let mut rows_matched: u64 = 0;

    for split in 0..fmt.num_splits() {
        if !fmt.may_match(split, pruners, &projection) {
            continue;
        }

        let mut idx_ranges = Vec::new();
        fmt.index_ranges(split, pruners, &projection, &mut idx_ranges).unwrap();
        for (s, e) in &idx_ranges {
            bytes_fetched += e - s;
        }

        let keep = fmt.refine_with_index(&src, split, pruners, &projection).unwrap();
        if !keep {
            continue;
        }

        let mut data_ranges = Vec::new();
        fmt.split_ranges(split, &projection, &mut data_ranges).unwrap();
        for (s, e) in &data_ranges {
            bytes_fetched += e - s;
        }

        let cols = fmt.read_split(&src, split, &projection).unwrap();
        rows_matched += cols[0].len() as u64;
    }

    let elapsed_us = start.elapsed().as_micros();
    println!("{label}: bytes={bytes_fetched} rows_scanned={rows_matched} time={elapsed_us}us");
    CaseResult { bytes_fetched, rows_matched, elapsed_us }
}
