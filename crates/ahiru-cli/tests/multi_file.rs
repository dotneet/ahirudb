//! 複数ファイル 1 テーブル機能のエンドツーエンドテスト。
//!
//! CLI の `+` 連結記法（`main.rs` の usage 参照）で複数ファイルを 1 テーブル
//! として登録し、素の UNION と Hive パーティションの両方を DuckDB と突き合わせる。
//! `sql_e2e.rs` と同じ「期待値を手で書かずに DuckDB と比較する」方針を踏襲する。

use std::process::Command;

fn duckdb_available() -> bool {
    Command::new("duckdb").arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn repo_path(rel: &str) -> String {
    format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel)
}

/// `+` で連結した複数ファイルを 1 テーブル `t` として ahiru CLI で実行する。
fn run_ahiru_multi(files: &[&str], sql: &str) -> Vec<Vec<String>> {
    let group = files.iter().map(|f| repo_path(f)).collect::<Vec<_>>().join("+");
    let out = Command::new(env!("CARGO_BIN_EXE_ahiru"))
        .args(["query", &group, sql])
        .output()
        .expect("failed to run ahiru");
    assert!(
        out.status.success(),
        "ahiru failed for `{sql}`:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    parse(&String::from_utf8_lossy(&out.stdout), '\t')
}

fn run_duckdb(sql: &str) -> Vec<Vec<String>> {
    let out =
        Command::new("duckdb").args(["-csv", "-c", sql]).output().expect("failed to run duckdb");
    assert!(
        out.status.success(),
        "duckdb failed for `{sql}`:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    parse(&String::from_utf8_lossy(&out.stdout), ',')
}

fn parse(text: &str, sep: char) -> Vec<Vec<String>> {
    text.lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.split(sep).map(normalize_cell).collect())
        .collect()
}

fn normalize_cell(s: &str) -> String {
    let s = s.trim().trim_matches('"');
    if s.is_empty() || s == "NULL" {
        return "<null>".into();
    }
    if let Ok(f) = s.parse::<f64>() {
        if f.fract() == 0.0 && f.abs() < 9e15 {
            return format!("{}", f as i64);
        }
        return format!("{f:.9}");
    }
    s.into()
}

fn body(mut rows: Vec<Vec<String>>) -> Vec<Vec<String>> {
    if rows.is_empty() {
        return rows;
    }
    rows.remove(0);
    rows
}

const MULTI: [&str; 3] =
    ["tests/data/multi/a.parquet", "tests/data/multi/b.parquet", "tests/data/multi/c.parquet"];

const HIVE: [&str; 3] = [
    "tests/data/hive/year=2024/month=01/part.parquet",
    "tests/data/hive/year=2024/month=02/part.parquet",
    "tests/data/hive/year=2025/month=01/part.parquet",
];

/// 3 ファイルを 1 テーブルとして束ねた `COUNT(*)` が、各ファイルの行数の合計
/// と一致する。DuckDB の `read_parquet([...])`（複数ファイルの素の UNION）と
/// 突き合わせる。
#[test]
fn multi_file_union_count_matches_duckdb() {
    if !duckdb_available() {
        eprintln!("duckdb が無いので飛ばす");
        return;
    }
    let ahiru = run_ahiru_multi(&MULTI, "SELECT count(*) FROM t");
    let paths: Vec<String> = MULTI.iter().map(|f| format!("'{}'", repo_path(f))).collect();
    let duckdb = run_duckdb(&format!("SELECT count(*) FROM read_parquet([{}])", paths.join(",")));
    assert_eq!(body(ahiru), body(duckdb));
}

/// 各ファイルの行 (id, name) がすべて出てくる（欠落・重複が無い）。
#[test]
fn multi_file_union_rows_match_duckdb() {
    if !duckdb_available() {
        eprintln!("duckdb が無いので飛ばす");
        return;
    }
    let ahiru = run_ahiru_multi(&MULTI, "SELECT id, name FROM t ORDER BY id");
    let paths: Vec<String> = MULTI.iter().map(|f| format!("'{}'", repo_path(f))).collect();
    let duckdb = run_duckdb(&format!(
        "SELECT id, name FROM read_parquet([{}]) ORDER BY id",
        paths.join(",")
    ));
    assert_eq!(body(ahiru), body(duckdb));
}

/// Hive パーティション列 (`year`, `month`) がスキーマに出て、
/// `WHERE year = 2024 AND month = 1` が該当ファイル分の行数だけを返す。
#[test]
fn hive_partition_columns_are_queryable_and_filterable() {
    let all = run_ahiru_multi(&HIVE, "SELECT count(*) FROM t");
    assert_eq!(body(all), vec![vec!["1000".to_string()]]);

    let jan24 = run_ahiru_multi(&HIVE, "SELECT count(*) FROM t WHERE year = 2024 AND month = 1");
    assert_eq!(body(jan24), vec![vec!["300".to_string()]]);

    let feb24 = run_ahiru_multi(&HIVE, "SELECT count(*) FROM t WHERE year = 2024 AND month = 2");
    assert_eq!(body(feb24), vec![vec!["400".to_string()]]);

    let y25 = run_ahiru_multi(&HIVE, "SELECT count(*) FROM t WHERE year = 2025");
    assert_eq!(body(y25), vec![vec!["300".to_string()]]);

    // 値そのものも合っている。
    let rows = run_ahiru_multi(&HIVE, "SELECT year, month FROM t WHERE id = 0");
    assert_eq!(body(rows), vec![vec!["2024".to_string(), "1".to_string()]]);
}

/// DuckDB の `hive_partitioning=true` と件数を突き合わせる。
#[test]
fn hive_partition_filter_matches_duckdb() {
    if !duckdb_available() {
        eprintln!("duckdb が無いので飛ばす");
        return;
    }
    let glob = repo_path("tests/data/hive/*/*/*.parquet");
    let ahiru = run_ahiru_multi(&HIVE, "SELECT count(*) FROM t WHERE year = 2024 AND month = 1");
    let duckdb = run_duckdb(&format!(
        "SELECT count(*) FROM read_parquet('{glob}', hive_partitioning=true) WHERE year=2024 AND month=1"
    ));
    assert_eq!(body(ahiru), body(duckdb));
}

/// `+` 連結記法そのものの簡単な動作確認（usage 文言に書いた記法が実際に動く）。
#[test]
fn plus_separator_smoke_test() {
    let out = run_ahiru_multi(&MULTI, "SELECT count(*) FROM t");
    assert_eq!(body(out), vec![vec!["480".to_string()]]);
}
