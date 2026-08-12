//! Load test for IN / BETWEEN pruning.
//!
//! Measures, on a sizable Parquet file, how much byte transfer the addition of
//! `PruneOp::In` in `crates/ahiru-core/src/format/mod.rs` (the fix that wires IN
//! lists into RowGroup-stats/PageIndex/Bloom-filter pruning) actually saves.
//!
//! Before the fix, `extract_pruners` (`plan/bind.rs`) did not handle
//! `Expr::InList` at all for `IN`, so no pruner was ever created. That means a
//! pre-fix `WHERE id IN (...)` should always have cost the same bytes as "no
//! predicate" (reading the whole column chunk). Here the measured "no
//! predicate" value stands in for "IN before the fix", compared against "IN
//! after the fix".
//!
//! Usage:
//! ```sh
//! cargo run --release --example bench_pruning -- /path/to/large.parquet
//! ```
//! If the argument is omitted, looks for `/tmp/ahiru_bench.parquet`.

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

    let full_scan = run_case(&bytes, "predicate none (full scan, stand-in for old IN)", &[]);

    run_case(
        &bytes,
        "id = X (equality, already worked before the fix)",
        &[Pruner {
            column: 0,
            op: PruneOp::Eq,
            value: Value::I32(1_234_567),
            in_values: Vec::new(),
        }],
    );

    run_case(
        &bytes,
        "id BETWEEN a AND b (range, already worked before the fix)",
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
        "id IN (small set of 4) <- fixed by this change",
        &[Pruner {
            column: 0,
            op: PruneOp::In,
            value: Value::I32(100),
            in_values: vec![Value::I32(500_000), Value::I32(1_000_000), Value::I32(1_800_000)],
        }],
    );

    // 50 values scattered across the whole file (almost every RowGroup gets at
    // least one hit -- the low-selectivity worst case).
    let scattered: Vec<Value> = (0..50).map(|i| Value::I32(i * 37_000 + 7)).collect();
    let (first, rest) = scattered.split_first().unwrap();
    let in_scattered = run_case(
        &bytes,
        "id IN (50 values scattered across the file, low selectivity) <- fixed by this change",
        &[Pruner { column: 0, op: PruneOp::In, value: first.clone(), in_values: rest.to_vec() }],
    );

    // 50 values that fit inside a single RowGroup (250k rows) -- a more typical
    // IN predicate, like a batch job or a specific customer's order IDs.
    let clustered: Vec<Value> = (0..50).map(|i| Value::I32(1_000_000 + i * 400)).collect();
    let (first, rest) = clustered.split_first().unwrap();
    let in_clustered = run_case(
        &bytes,
        "id IN (50 values fitting in one RowGroup, high selectivity) <- fixed by this change",
        &[Pruner { column: 0, op: PruneOp::In, value: first.clone(), in_values: rest.to_vec() }],
    );

    println!();
    println!("=== summary ===");
    report_speedup("IN (small set of 4)", &full_scan, &in_small);
    report_speedup("IN (50 scattered values, low selectivity)", &full_scan, &in_scattered);
    report_speedup(
        "IN (50 values within one RowGroup, high selectivity)",
        &full_scan,
        &in_clustered,
    );
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
        "{label}: bytes reduced {byte_ratio:.1}x ({} -> {}), elapsed {time_ratio:.1}x ({} us -> {} us), rows scanned {} -> {}",
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

    let projection = vec![0usize]; // project only the id column
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
