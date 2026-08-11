//! `COPY (SELECT ...) TO 'path' [(FORMAT csv|jsonl)]` のエンドツーエンドテスト。
//!
//! `sql_e2e.rs` と同じ「期待値を手で書かずに DuckDB と突き合わせる」方針だが、
//! ここでは標準出力の行を比べるのではなく、**実際に書き出されたファイルを
//! バイト単位で** DuckDB の出力と比較する。`ahiru-core` はファイルシステムに
//! 触れられない（no_std）ので、書き込みを実際に担う `ahiru-cli` の経路を
//! 通しで確認するのが狙い（`crates/ahiru-core/src/write/mod.rs`、
//! `crates/ahiru-core/src/session.rs` の `Query::copy_result` 参照）。
//!
//! DuckDB が入っていない環境ではテストごと飛ばす（CI では入れる）。

use std::path::PathBuf;
use std::process::Command;

fn duckdb_available() -> bool {
    Command::new("duckdb").arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn repo_path(rel: &str) -> String {
    format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel)
}

/// テストごとに衝突しないテンポラリファイルパスを作る。
fn tmp_path(tag: &str, ext: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("ahiru_copy_test_{tag}_{}.{ext}", std::process::id()));
    p
}

/// ahiru CLI で `COPY` 文を実行する。標準出力は捨てて成否だけ見る
/// （書き込み結果はファイルを直接読んで検証する）。
fn run_ahiru_copy(files: &[&str], sql: &str) {
    let mut args: Vec<String> = vec!["query".into()];
    args.extend(files.iter().map(|f| repo_path(f)));
    args.push(sql.into());
    let out = Command::new(env!("CARGO_BIN_EXE_ahiru"))
        .args(&args)
        .output()
        .expect("failed to run ahiru");
    assert!(
        out.status.success(),
        "ahiru failed for `{sql}`:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// DuckDB で `COPY` 文を実行する。
fn run_duckdb_copy(sql: &str) {
    let out = Command::new("duckdb").args(["-c", sql]).output().expect("failed to run duckdb");
    assert!(
        out.status.success(),
        "duckdb failed for `{sql}`:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// CSV ソースファイルを DuckDB から読む式（`sql_e2e.rs::duckdb_source` と同じ）。
fn duckdb_csv_source(file: &str) -> String {
    format!("read_csv_auto('{}')", repo_path(file))
}

macro_rules! skip_without_duckdb {
    () => {
        if !duckdb_available() {
            eprintln!("duckdb が無いので飛ばす");
            return;
        }
    };
}

#[test]
fn copy_csv_extension_matches_duckdb_byte_for_byte() {
    skip_without_duckdb!();
    let ahiru_out = tmp_path("csv_ext", "csv");
    let duckdb_out = tmp_path("csv_ext_duckdb", "csv");

    run_ahiru_copy(
        &["tests/data/basic.csv"],
        &format!(
            "COPY (SELECT id, name, flag, big FROM t ORDER BY id LIMIT 5) TO '{}'",
            ahiru_out.display()
        ),
    );
    run_duckdb_copy(&format!(
        "COPY (SELECT id, name, flag, big FROM {} ORDER BY id LIMIT 5) TO '{}'",
        duckdb_csv_source("tests/data/basic.csv"),
        duckdb_out.display()
    ));

    let a = std::fs::read(&ahiru_out).expect("ahiru の出力が無い");
    let d = std::fs::read(&duckdb_out).expect("duckdb の出力が無い");
    assert_eq!(a, d, "CSV 出力がバイト単位で一致しない");
    let _ = std::fs::remove_file(&ahiru_out);
    let _ = std::fs::remove_file(&duckdb_out);
}

#[test]
fn copy_jsonl_format_matches_duckdb_byte_for_byte() {
    skip_without_duckdb!();
    // 拡張子は敢えて CSV っぽくない `.ndjson` にして、明示 FORMAT が
    // 拡張子より優先されることも一緒に確認する。
    let ahiru_out = tmp_path("jsonl_fmt", "ndjson");
    let duckdb_out = tmp_path("jsonl_fmt_duckdb", "ndjson");

    run_ahiru_copy(
        &["tests/data/basic.csv"],
        &format!(
            "COPY (SELECT id, name, flag, big FROM t ORDER BY id LIMIT 5) TO '{}' (FORMAT jsonl)",
            ahiru_out.display()
        ),
    );
    // DuckDB は NDJSON も `FORMAT JSON` として書く（配列ではなく改行区切り。
    // 手元の duckdb CLI で実測して確認済み）。
    run_duckdb_copy(&format!(
        "COPY (SELECT id, name, flag, big FROM {} ORDER BY id LIMIT 5) TO '{}' (FORMAT JSON)",
        duckdb_csv_source("tests/data/basic.csv"),
        duckdb_out.display()
    ));

    let a = std::fs::read(&ahiru_out).expect("ahiru の出力が無い");
    let d = std::fs::read(&duckdb_out).expect("duckdb の出力が無い");
    assert_eq!(a, d, "JSONL 出力がバイト単位で一致しない");
    let _ = std::fs::remove_file(&ahiru_out);
    let _ = std::fs::remove_file(&duckdb_out);
}

/// `COPY <table> TO ...`（サブクエリでなく素のテーブル名）が
/// `SELECT * FROM <table>` と同じ結果になることを、DuckDB 側は明示
/// `SELECT *` で組んで突き合わせる。
#[test]
fn copy_table_form_matches_select_star_from_table() {
    skip_without_duckdb!();
    let ahiru_out = tmp_path("table_form", "csv");
    let duckdb_out = tmp_path("table_form_duckdb", "csv");

    run_ahiru_copy(&["tests/data/basic.csv"], &format!("COPY t TO '{}'", ahiru_out.display()));
    run_duckdb_copy(&format!(
        "COPY (SELECT * FROM {}) TO '{}'",
        duckdb_csv_source("tests/data/basic.csv"),
        duckdb_out.display()
    ));

    let a = std::fs::read(&ahiru_out).expect("ahiru の出力が無い");
    let d = std::fs::read(&duckdb_out).expect("duckdb の出力が無い");
    assert_eq!(a, d, "COPY <table> TO の出力がバイト単位で一致しない");
    let _ = std::fs::remove_file(&ahiru_out);
    let _ = std::fs::remove_file(&duckdb_out);
}

/// 拡張子だけで書き出せない・明示 `FORMAT` も未対応な場合は明確なエラーで
/// 落ちる（黙って Parquet 相当として扱おうとしたりしない）。
#[test]
fn copy_unsupported_extension_fails_clearly() {
    let out_path = tmp_path("unsupported", "parquet");
    let mut args: Vec<String> = vec!["query".into(), repo_path("tests/data/basic.csv")];
    args.push(format!("COPY (SELECT id FROM t) TO '{}'", out_path.display()));
    let out = Command::new(env!("CARGO_BIN_EXE_ahiru"))
        .args(&args)
        .output()
        .expect("failed to run ahiru");
    assert!(!out.status.success(), "拡張子が parquet の COPY は失敗するはず");
    assert!(!out_path.exists(), "失敗したのにファイルが作られている: {}", out_path.display());
}
