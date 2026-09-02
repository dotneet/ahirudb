//! End-to-end tests for the multiple-files-one-table feature.
//!
//! Registers several files as one table using the CLI's `+` concatenation notation
//! (see the usage text in `main.rs`) and cross-checks both a plain UNION and Hive
//! partitioning against DuckDB. Follows the same "compare with DuckDB rather than hand-writing expected values" policy as `sql_e2e.rs`.

use std::process::Command;

fn duckdb_available() -> bool {
    Command::new("duckdb").arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn repo_path(rel: &str) -> String {
    format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel)
}

/// Runs the ahiru CLI with several `+`-joined files bound as the single table `t`.
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

/// `COUNT(*)` over three files bundled as one table equals the sum of the per-file
/// row counts. Cross-checked against DuckDB's `read_parquet([...])` (a plain
/// multi-file UNION).
#[test]
fn multi_file_union_count_matches_duckdb() {
    if !duckdb_available() {
        eprintln!("duckdb not found, skipping");
        return;
    }
    let ahiru = run_ahiru_multi(&MULTI, "SELECT count(*) FROM t");
    let paths: Vec<String> = MULTI.iter().map(|f| format!("'{}'", repo_path(f))).collect();
    let duckdb = run_duckdb(&format!("SELECT count(*) FROM read_parquet([{}])", paths.join(",")));
    assert_eq!(body(ahiru), body(duckdb));
}

/// Every (id, name) row of each file shows up (nothing missing, nothing duplicated).
#[test]
fn multi_file_union_rows_match_duckdb() {
    if !duckdb_available() {
        eprintln!("duckdb not found, skipping");
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

/// The Hive partition columns (`year`, `month`) appear in the schema, and
/// `WHERE year = 2024 AND month = '01'` returns only the rows of the matching files.
///
/// `month` is VARCHAR, not an integer: the directories spell it `month=01`, and a zero-padded
/// value read as a number would come back as `1` and lose its padding (DuckDB keeps `01` too).
/// `year=2024` has no padding and stays INTEGER, so a typed integer comparison still works there.
#[test]
fn hive_partition_columns_are_queryable_and_filterable() {
    let all = run_ahiru_multi(&HIVE, "SELECT count(*) FROM t");
    assert_eq!(body(all), vec![vec!["1000".to_string()]]);

    let jan24 = run_ahiru_multi(&HIVE, "SELECT count(*) FROM t WHERE year = 2024 AND month = '01'");
    assert_eq!(body(jan24), vec![vec!["300".to_string()]]);

    let feb24 = run_ahiru_multi(&HIVE, "SELECT count(*) FROM t WHERE year = 2024 AND month = '02'");
    assert_eq!(body(feb24), vec![vec!["400".to_string()]]);

    let y25 = run_ahiru_multi(&HIVE, "SELECT count(*) FROM t WHERE year = 2025");
    assert_eq!(body(y25), vec![vec!["300".to_string()]]);

    // The values themselves are right too. (`normalize_cell` reads a numeric-looking cell as a
    // number, so the padding of `01` is not visible here; that the VARCHAR value keeps its zero
    // is checked directly in `format::partitioned`'s unit tests.)
    let rows = run_ahiru_multi(&HIVE, "SELECT year, month FROM t WHERE id = 0");
    assert_eq!(body(rows), vec![vec!["2024".to_string(), "1".to_string()]]);
}

/// Cross-checks the count against DuckDB's `hive_partitioning=true`.
#[test]
fn hive_partition_filter_matches_duckdb() {
    if !duckdb_available() {
        eprintln!("duckdb not found, skipping");
        return;
    }
    let glob = repo_path("tests/data/hive/*/*/*.parquet");
    let ahiru = run_ahiru_multi(&HIVE, "SELECT count(*) FROM t WHERE year = 2024 AND month = '01'");
    let duckdb = run_duckdb(&format!(
        "SELECT count(*) FROM read_parquet('{glob}', hive_partitioning=true) WHERE year=2024 AND month=1"
    ));
    assert_eq!(body(ahiru), body(duckdb));
}

/// A simple smoke test of the `+` notation itself (the notation documented in the usage text really works).
#[test]
fn plus_separator_smoke_test() {
    let out = run_ahiru_multi(&MULTI, "SELECT count(*) FROM t");
    assert_eq!(body(out), vec![vec!["480".to_string()]]);
}
