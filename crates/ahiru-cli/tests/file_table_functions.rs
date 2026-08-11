//! End-to-end tests for referencing a file directly as a table in `FROM`,
//! without a prior host-side `register`/`register_as` call under a short
//! alias — the gap identified against DuckDB (see the task description this
//! change was written against): only `parquet('path')` used to work this
//! way; CSV/JSONL/JSON readers existed but had no equivalent SQL surface.
//!
//! Covered here:
//! - `FROM 'path'` (bare string literal, format inferred from the
//!   extension via `format::FormatKind::detect`)
//! - `read_parquet('path')` (alias for `parquet('path')`)
//! - `read_csv('path')` / `read_csv_auto('path')`
//! - `read_json('path')` / `read_json_auto('path')`
//! - aliases (`AS x`) on all of the above
//! - joining two file-table-function references together
//! - errors for a path that was never registered (host-side registration is
//!   still required — this engine is `no_std` and has no filesystem access
//!   of its own; see docs/sql/data-sources.md)
//!
//! This lives under `ahiru-cli/tests` (not `ahiru-core/tests`) because it
//! needs to prove the SQL syntax actually resolves against real files on
//! disk, which only the native CLI (with real filesystem access) can do —
//! `ahiru-core` itself never touches a filesystem.
//!
//! The native `ahiru` CLI (see `crates/ahiru-cli/src/main.rs::cmd_query`)
//! registers every file argument twice: once under a short name (`t`, `t2`,
//! ...) and once under its own literal path string (so that
//! `parquet('<path>')`-style references work). This test always passes the
//! exact same path string it embeds in the SQL text, so the two line up.

use std::process::Command;

fn repo_path(rel: &str) -> String {
    format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel)
}

/// Runs the CLI, asserting success, and returns the parsed TSV output
/// (including the header row).
fn run_ahiru(files: &[&str], sql: &str) -> Vec<Vec<String>> {
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
    parse(&String::from_utf8_lossy(&out.stdout))
}

/// Runs the CLI, asserting *failure*, and returns stderr (for inspecting the
/// error message).
fn run_ahiru_expect_failure(files: &[&str], sql: &str) -> String {
    let mut args: Vec<String> = vec!["query".into()];
    args.extend(files.iter().map(|f| repo_path(f)));
    args.push(sql.into());
    let out = Command::new(env!("CARGO_BIN_EXE_ahiru"))
        .args(&args)
        .output()
        .expect("failed to run ahiru");
    assert!(!out.status.success(), "expected ahiru to fail for `{sql}`, but it succeeded");
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn parse(text: &str) -> Vec<Vec<String>> {
    text.lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.split('\t').map(str::to_string).collect())
        .collect()
}

fn body(mut rows: Vec<Vec<String>>) -> Vec<Vec<String>> {
    if rows.is_empty() {
        return rows;
    }
    rows.remove(0);
    rows
}

const BASIC_PARQUET: &str = "tests/data/basic.parquet";
const BASIC_CSV: &str = "tests/data/basic.csv";
const BASIC_JSONL: &str = "tests/data/basic.jsonl";
const BASIC_JSON_ARRAY: &str = "tests/data/basic_array.json";

// --- `FROM 'path'` (bare string literal, extension-inferred format) --------

#[test]
fn bare_literal_from_reads_parquet() {
    let path = repo_path(BASIC_PARQUET);
    let rows = run_ahiru(&[BASIC_PARQUET], &format!("SELECT count(*) FROM '{path}'"));
    assert_eq!(body(rows), vec![vec!["1000".to_string()]]);
}

#[test]
fn bare_literal_from_reads_csv() {
    let path = repo_path(BASIC_CSV);
    let rows = run_ahiru(&[BASIC_CSV], &format!("SELECT count(*) FROM '{path}'"));
    assert_eq!(body(rows), vec![vec!["1000".to_string()]]);
}

#[test]
fn bare_literal_from_reads_jsonl() {
    let path = repo_path(BASIC_JSONL);
    let rows = run_ahiru(&[BASIC_JSONL], &format!("SELECT count(*) FROM '{path}'"));
    assert_eq!(body(rows), vec![vec!["1000".to_string()]]);
}

#[test]
fn bare_literal_from_reads_json_array() {
    let path = repo_path(BASIC_JSON_ARRAY);
    let rows = run_ahiru(&[BASIC_JSON_ARRAY], &format!("SELECT count(*) FROM '{path}'"));
    assert_eq!(body(rows), vec![vec!["1000".to_string()]]);
}

#[test]
fn bare_literal_from_accepts_alias_and_filters() {
    let path = repo_path(BASIC_PARQUET);
    let rows =
        run_ahiru(&[BASIC_PARQUET], &format!("SELECT p.id FROM '{path}' AS p WHERE p.id = 7"));
    assert_eq!(body(rows), vec![vec!["7".to_string()]]);
}

// --- `read_parquet(...)` -----------------------------------------------------

#[test]
fn read_parquet_reads_parquet() {
    let path = repo_path(BASIC_PARQUET);
    let rows = run_ahiru(&[BASIC_PARQUET], &format!("SELECT count(*) FROM read_parquet('{path}')"));
    assert_eq!(body(rows), vec![vec!["1000".to_string()]]);
}

// --- `read_csv(...)` / `read_csv_auto(...)` ---------------------------------

#[test]
fn read_csv_reads_csv() {
    let path = repo_path(BASIC_CSV);
    let rows = run_ahiru(&[BASIC_CSV], &format!("SELECT count(*) FROM read_csv('{path}')"));
    assert_eq!(body(rows), vec![vec!["1000".to_string()]]);
}

#[test]
fn read_csv_auto_reads_csv() {
    let path = repo_path(BASIC_CSV);
    let rows = run_ahiru(&[BASIC_CSV], &format!("SELECT count(*) FROM read_csv_auto('{path}')"));
    assert_eq!(body(rows), vec![vec!["1000".to_string()]]);
}

#[test]
fn read_csv_accepts_alias_and_filters() {
    let path = repo_path(BASIC_CSV);
    let rows = run_ahiru(
        &[BASIC_CSV],
        &format!("SELECT x.id FROM read_csv('{path}') AS x WHERE x.id = 5"),
    );
    assert_eq!(body(rows), vec![vec!["5".to_string()]]);
}

// --- `read_json(...)` / `read_json_auto(...)` -------------------------------

#[test]
fn read_json_reads_a_top_level_json_array() {
    let path = repo_path(BASIC_JSON_ARRAY);
    let rows = run_ahiru(&[BASIC_JSON_ARRAY], &format!("SELECT count(*) FROM read_json('{path}')"));
    assert_eq!(body(rows), vec![vec!["1000".to_string()]]);
}

#[test]
fn read_json_auto_reads_jsonl() {
    let path = repo_path(BASIC_JSONL);
    let rows = run_ahiru(&[BASIC_JSONL], &format!("SELECT count(*) FROM read_json_auto('{path}')"));
    assert_eq!(body(rows), vec![vec!["1000".to_string()]]);
}

// --- combining several file-table-function references ----------------------

/// Two different `FromItem::File` references (one CSV, one Parquet, both the
/// same underlying data) joined together. Confirms the new syntax composes
/// normally with the rest of the query planner, not just as a lone `FROM`.
#[test]
fn joining_two_file_table_functions() {
    let csv = repo_path(BASIC_CSV);
    let parquet = repo_path(BASIC_PARQUET);
    let sql = format!(
        "SELECT a.id FROM read_csv('{csv}') a JOIN read_parquet('{parquet}') b ON a.id = b.id WHERE a.id = 3"
    );
    let rows = run_ahiru(&[BASIC_CSV, BASIC_PARQUET], &sql);
    assert_eq!(body(rows), vec![vec!["3".to_string()]]);
}

// --- errors ------------------------------------------------------------------

/// A path that was never registered by the host (not passed as a CLI file
/// argument) fails clearly, for every syntax form — this engine cannot reach
/// out to the filesystem on its own (`no_std`), so the failure mode is
/// "table not found", the same as an unregistered `parquet('...')` today.
#[test]
fn unregistered_path_fails_for_every_syntax_form() {
    let missing = repo_path("tests/data/does_not_exist.csv");
    for sql in [
        format!("SELECT * FROM '{missing}'"),
        format!("SELECT * FROM read_csv('{missing}')"),
        format!("SELECT * FROM read_csv_auto('{missing}')"),
    ] {
        // Register an unrelated file so the CLI has something to do, but
        // never the path referenced in `sql` itself.
        let err = run_ahiru_expect_failure(&[BASIC_CSV], &sql);
        assert!(err.contains("table not found"), "unexpected error for `{sql}`: {err}");
    }
}

#[test]
fn unregistered_path_fails_for_read_parquet_and_read_json() {
    let missing_parquet = repo_path("tests/data/does_not_exist.parquet");
    let err = run_ahiru_expect_failure(
        &[BASIC_PARQUET],
        &format!("SELECT * FROM read_parquet('{missing_parquet}')"),
    );
    assert!(err.contains("table not found"), "unexpected error: {err}");

    let missing_json = repo_path("tests/data/does_not_exist.json");
    let err = run_ahiru_expect_failure(
        &[BASIC_JSON_ARRAY],
        &format!("SELECT * FROM read_json('{missing_json}')"),
    );
    assert!(err.contains("table not found"), "unexpected error: {err}");
}
