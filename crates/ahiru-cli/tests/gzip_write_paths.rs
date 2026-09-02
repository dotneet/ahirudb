//! End-to-end tests for the write statements over a GZIP-compressed Parquet
//! source.
//!
//! GZIP inflate is deliberately not built into `ahiru-core` (DESIGN.md §6): the
//! host performs it and hands the bytes back. `COPY`, `CREATE TABLE AS` and
//! `INSERT ... SELECT` all complete inside `Session::prepare`, so the CLI's
//! own `NEED_CODEC` loop never gets a turn — they went through
//! `Session::set_codec_hook` instead, and used to fail outright with
//! `[E504] io failed`. These tests drive the real binary so that wiring stays
//! connected.
//!
//! Environments without DuckDB skip the cross-check but still run the rest.

use std::path::PathBuf;
use std::process::Command;

const SOURCE: &str = "tests/data/gzip.parquet";
const EXPECTED_ROWS: usize = 5000;

fn duckdb_available() -> bool {
    Command::new("duckdb").arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn repo_path(rel: &str) -> String {
    format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel)
}

fn tmp_path(tag: &str, ext: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("ahiru_gzip_write_{tag}_{}.{ext}", std::process::id()));
    p
}

/// Runs one or more statements through the CLI and returns stdout.
fn ahiru(sql: &str) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_ahiru"))
        .args(["-batch", "-csv", "-c", sql])
        .output()
        .expect("failed to run ahiru");
    assert!(
        out.status.success(),
        "ahiru failed for `{sql}`:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// The single value of a one-row, one-column result.
fn scalar(sql: &str) -> String {
    let text = ahiru(sql);
    let mut lines = text.lines().filter(|l| !l.is_empty() && !l.starts_with('('));
    lines.next().expect("header");
    lines.next().expect("value").to_string()
}

#[test]
fn ctas_reads_a_gzip_compressed_parquet_source() {
    let src = repo_path(SOURCE);
    let text = ahiru(&format!(
        "CREATE TABLE g AS SELECT * FROM '{src}'; \
         SELECT count(*), count(DISTINCT id) FROM g"
    ));
    let mut values =
        text.lines().filter(|l| !l.is_empty() && !l.starts_with('(')).skip(1).step_by(2);
    // The CTAS statement's own result is the number of rows it stored, and the
    // follow-up query confirms the values themselves survived, not just the count.
    assert_eq!(values.next(), Some(EXPECTED_ROWS.to_string().as_str()), "{text}");
    assert_eq!(values.next(), Some(format!("{EXPECTED_ROWS},{EXPECTED_ROWS}").as_str()), "{text}");
}

#[test]
fn insert_select_reads_a_gzip_compressed_parquet_source() {
    let src = repo_path(SOURCE);
    let text = ahiru(&format!(
        "CREATE TABLE g AS SELECT * FROM '{src}' WHERE 1 = 0; \
         INSERT INTO g SELECT * FROM '{src}'; \
         SELECT count(*) FROM g"
    ));
    assert!(
        text.contains(&EXPECTED_ROWS.to_string()),
        "expected {EXPECTED_ROWS} rows somewhere in:\n{text}"
    );
}

#[test]
fn copy_from_a_gzip_compressed_parquet_source_matches_duckdb() {
    let src = repo_path(SOURCE);
    let ours = tmp_path("copy", "csv");
    let _ = std::fs::remove_file(&ours);
    ahiru(&format!("COPY (SELECT * FROM '{src}') TO '{}'", ours.display()));

    let text = std::fs::read_to_string(&ours).expect("ahiru wrote no file");
    assert_eq!(text.lines().count(), EXPECTED_ROWS + 1, "header plus one line per row");

    if duckdb_available() {
        let theirs = tmp_path("copy_duckdb", "csv");
        let _ = std::fs::remove_file(&theirs);
        let out = Command::new("duckdb")
            .args(["-c", &format!("COPY (SELECT * FROM '{src}') TO '{}'", theirs.display())])
            .output()
            .expect("failed to run duckdb");
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let mut mine: Vec<&str> = text.lines().collect();
        let reference = std::fs::read_to_string(&theirs).unwrap();
        let mut duck: Vec<&str> = reference.lines().collect();
        mine.sort_unstable();
        duck.sort_unstable();
        assert_eq!(mine, duck);
        let _ = std::fs::remove_file(&theirs);
    }
    let _ = std::fs::remove_file(&ours);
}

// The JSONL writer's non-finite and DECIMAL rendering, through the real
// `COPY ... TO` path: NaN and the infinities stay distinguishable from NULL
// (they used to be flattened to `null`) and a DECIMAL is a JSON number, as
// `duckdb` writes it.
#[test]
fn copy_to_jsonl_writes_non_finite_doubles_and_decimals_readably() {
    let path = tmp_path("jsonl", "jsonl");
    let _ = std::fs::remove_file(&path);
    ahiru(&format!(
        "COPY (SELECT 'nan'::DOUBLE AS n, 'inf'::DOUBLE AS p, '-inf'::DOUBLE AS m, \
         CAST(NULL AS DOUBLE) AS z, CAST(12.345 AS DECIMAL(10,3)) AS d FROM range(1)) TO '{}'",
        path.display()
    ));
    let text = std::fs::read_to_string(&path).unwrap();
    assert_eq!(
        text.trim_end(),
        r#"{"n":"NaN","p":"Infinity","m":"-Infinity","z":null,"d":12.345}"#
    );

    // And the engine reads its own output back without error.
    let back =
        scalar(&format!("SELECT CAST(p AS DOUBLE) = 'inf'::DOUBLE FROM '{}'", path.display()));
    assert_eq!(back, "true");
    let _ = std::fs::remove_file(&path);
}
