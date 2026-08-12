//! End-to-end tests for `COPY (SELECT ...) TO 'path' [(FORMAT csv|jsonl|parquet)]`.
//!
//! Same policy as `sql_e2e.rs` -- cross-check against DuckDB rather than writing
//! expected values by hand -- but here the comparison is not of stdout lines but of
//! **the actually written files, byte for byte**, against DuckDB's output.
//! `ahiru-core` cannot touch the filesystem (no_std), so the aim is to exercise the
//! `ahiru-cli` path that actually performs the write (see
//! `crates/ahiru-core/src/write/mod.rs` and `Query::copy_result` in `crates/ahiru-core/src/session.rs`).
//!
//! Environments without DuckDB skip these tests entirely (CI installs it).

use std::path::PathBuf;
use std::process::Command;

fn duckdb_available() -> bool {
    Command::new("duckdb").arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn repo_path(rel: &str) -> String {
    format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel)
}

/// Builds a temporary file path that will not collide across tests.
fn tmp_path(tag: &str, ext: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("ahiru_copy_test_{tag}_{}.{ext}", std::process::id()));
    p
}

/// Runs a `COPY` statement through the ahiru CLI. stdout is discarded and only
/// success is checked (the written result is verified by reading the file directly).
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

/// Runs a `COPY` statement through DuckDB.
fn run_duckdb_copy(sql: &str) {
    let out = Command::new("duckdb").args(["-c", sql]).output().expect("failed to run duckdb");
    assert!(
        out.status.success(),
        "duckdb failed for `{sql}`:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Runs a query returning one row and one column through DuckDB and returns that value as a string.
fn duckdb_scalar(sql: &str) -> String {
    let out = Command::new("duckdb")
        .args(["-noheader", "-list", "-c", sql])
        .output()
        .expect("failed to run duckdb");
    assert!(
        out.status.success(),
        "duckdb failed for `{sql}`:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// The expression for reading the CSV source file from DuckDB (same as `sql_e2e.rs::duckdb_source`).
fn duckdb_csv_source(file: &str) -> String {
    format!("read_csv_auto('{}')", repo_path(file))
}

macro_rules! skip_without_duckdb {
    () => {
        if !duckdb_available() {
            eprintln!("duckdb not found, skipping");
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

    let a = std::fs::read(&ahiru_out).expect("no output from ahiru");
    let d = std::fs::read(&duckdb_out).expect("no output from duckdb");
    assert_eq!(a, d, "CSV output does not match byte for byte");
    let _ = std::fs::remove_file(&ahiru_out);
    let _ = std::fs::remove_file(&duckdb_out);
}

#[test]
fn copy_jsonl_format_matches_duckdb_byte_for_byte() {
    skip_without_duckdb!();
    // The extension is deliberately the un-CSV-like `.ndjson`, so this also confirms
    // that an explicit FORMAT takes precedence over the extension.
    let ahiru_out = tmp_path("jsonl_fmt", "ndjson");
    let duckdb_out = tmp_path("jsonl_fmt_duckdb", "ndjson");

    run_ahiru_copy(
        &["tests/data/basic.csv"],
        &format!(
            "COPY (SELECT id, name, flag, big FROM t ORDER BY id LIMIT 5) TO '{}' (FORMAT jsonl)",
            ahiru_out.display()
        ),
    );
    // DuckDB writes NDJSON under `FORMAT JSON` too (newline-delimited, not an array;
    // confirmed by measuring the local duckdb CLI).
    run_duckdb_copy(&format!(
        "COPY (SELECT id, name, flag, big FROM {} ORDER BY id LIMIT 5) TO '{}' (FORMAT JSON)",
        duckdb_csv_source("tests/data/basic.csv"),
        duckdb_out.display()
    ));

    let a = std::fs::read(&ahiru_out).expect("no output from ahiru");
    let d = std::fs::read(&duckdb_out).expect("no output from duckdb");
    assert_eq!(a, d, "JSONL output does not match byte for byte");
    let _ = std::fs::remove_file(&ahiru_out);
    let _ = std::fs::remove_file(&duckdb_out);
}

/// Confirms that `COPY <table> TO ...` (a plain table name rather than a subquery)
/// produces the same result as `SELECT * FROM <table>`, cross-checked against a
/// DuckDB side written with an explicit `SELECT *`.
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

    let a = std::fs::read(&ahiru_out).expect("no output from ahiru");
    let d = std::fs::read(&duckdb_out).expect("no output from duckdb");
    assert_eq!(a, d, "the output of COPY <table> TO does not match byte for byte");
    let _ = std::fs::remove_file(&ahiru_out);
    let _ = std::fs::remove_file(&duckdb_out);
}

/// A format no build ever supports fails with a clear error and leaves no
/// file behind (rather than silently falling back to some other writer).
#[test]
fn copy_unsupported_format_fails_clearly() {
    let out_path = tmp_path("unsupported", "orc");
    let mut args: Vec<String> = vec!["query".into(), repo_path("tests/data/basic.csv")];
    args.push(format!("COPY (SELECT id FROM t) TO '{}' (FORMAT orc)", out_path.display()));
    let out = Command::new(env!("CARGO_BIN_EXE_ahiru"))
        .args(&args)
        .output()
        .expect("failed to run ahiru");
    assert!(!out.status.success(), "COPY with an unsupported format should fail");
    assert!(!out_path.exists(), "it failed yet a file was created: {}", out_path.display());
}

// --- Parquet -----------------------------------------------------------------
// Parquet output does not match DuckDB byte for byte (encodings, compression, and
// statistics choices differ; ahirudb writes uncompressed PLAIN only). What matters
// here is "can DuckDB read it", so the comparison is made by having DuckDB read the
// written file and matching that against the source data. The round trip through
// ahirudb's own reader is covered in `crates/ahiru-core/src/write/parquet/tests.rs`.

/// Has DuckDB read the written Parquet and confirms, via the size of the symmetric
/// difference, that not one row differs from the source CSV.
#[test]
fn copy_parquet_is_readable_by_duckdb_with_identical_rows() {
    skip_without_duckdb!();
    let ahiru_out = tmp_path("parquet_ext", "parquet");
    let cols = "id, name, score, flag, big, d";

    run_ahiru_copy(
        &["tests/data/basic.csv"],
        &format!("COPY (SELECT {cols} FROM t) TO '{}'", ahiru_out.display()),
    );

    let src = duckdb_csv_source("tests/data/basic.csv");
    let pq = format!("read_parquet('{}')", ahiru_out.display());
    let diff = duckdb_scalar(&format!(
        "SELECT count(*) FROM (
           (SELECT {cols} FROM {src} EXCEPT ALL SELECT {cols} FROM {pq})
           UNION ALL
           (SELECT {cols} FROM {pq} EXCEPT ALL SELECT {cols} FROM {src})
         )"
    ));
    assert_eq!(diff, "0", "the rows DuckDB sees differ from the source CSV");

    let _ = std::fs::remove_file(&ahiru_out);
}

/// The column types DuckDB infers must match the SQL types at write time.
/// Whether logical types (`DATE` / `DECIMAL` / `UUID` / `TIMESTAMP(isAdjustedToUTC)`
/// and so on) are written correctly into the footer cannot be told from values alone.
#[test]
fn copy_parquet_preserves_column_types_as_duckdb_sees_them() {
    skip_without_duckdb!();
    let ahiru_out = tmp_path("parquet_types", "parquet");

    run_ahiru_copy(
        &["tests/data/basic.csv"],
        &format!(
            "COPY (SELECT CAST(id AS TINYINT) AS c_tinyint, \
                          CAST(id AS SMALLINT) AS c_smallint, \
                          CAST(id AS INTEGER) AS c_integer, \
                          CAST(id AS BIGINT) AS c_bigint, \
                          CAST(id AS UTINYINT) AS c_utinyint, \
                          CAST(id AS USMALLINT) AS c_usmallint, \
                          CAST(id AS UINTEGER) AS c_uinteger, \
                          CAST(id AS UBIGINT) AS c_ubigint, \
                          CAST(score AS FLOAT) AS c_float, \
                          CAST(score AS DOUBLE) AS c_double, \
                          CAST(score AS DECIMAL(10,2)) AS c_decimal_small, \
                          CAST(score AS DECIMAL(30,4)) AS c_decimal_big, \
                          CAST(name AS VARCHAR) AS c_varchar, \
                          CAST(name AS BLOB) AS c_blob, \
                          CAST(d AS DATE) AS c_date, \
                          CAST(d AS TIMESTAMP) AS c_timestamp, \
                          CAST(d AS TIMESTAMPTZ) AS c_timestamptz, \
                          CAST('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11' AS UUID) AS c_uuid, \
                          flag AS c_boolean \
                   FROM t LIMIT 3) TO '{}'",
            ahiru_out.display()
        ),
    );

    let types = duckdb_scalar(&format!(
        "SELECT string_agg(column_name || '=' || column_type, ' ' ORDER BY column_name) \
         FROM (DESCRIBE SELECT * FROM read_parquet('{}'))",
        ahiru_out.display()
    ));
    assert_eq!(
        types,
        "c_bigint=BIGINT c_blob=BLOB c_boolean=BOOLEAN c_date=DATE \
         c_decimal_big=DECIMAL(30,4) c_decimal_small=DECIMAL(10,2) c_double=DOUBLE \
         c_float=FLOAT c_integer=INTEGER c_smallint=SMALLINT c_timestamp=TIMESTAMP \
         c_timestamptz=TIMESTAMP WITH TIME ZONE c_tinyint=TINYINT c_ubigint=UBIGINT \
         c_uinteger=UINTEGER c_usmallint=USMALLINT c_utinyint=UTINYINT c_uuid=UUID \
         c_varchar=VARCHAR"
    );

    let _ = std::fs::remove_file(&ahiru_out);
}

/// A file with multiple RowGroups must also be readable end to end from DuckDB.
/// The rows are generated inside the test (no wish to keep 120k rows of data in the repository).
#[test]
fn copy_parquet_spanning_multiple_row_groups_reads_back_intact() {
    skip_without_duckdb!();
    let src = tmp_path("parquet_multi_rg_src", "csv");
    let ahiru_out = tmp_path("parquet_multi_rg", "parquet");

    // `write::parquet::ROW_GROUP_ROWS` is 122880. Pick a row count that certainly crosses it.
    let rows = 130_000usize;
    let mut csv = String::from("id,name\n");
    for i in 0..rows {
        // Make one row in five NULL, so definition levels go through both the RLE
        // and bit-packed paths.
        if i % 5 == 0 {
            csv.push_str(&format!("{i},\n"));
        } else {
            csv.push_str(&format!("{i},n{i}\n"));
        }
    }
    std::fs::write(&src, csv).expect("cannot write the temporary CSV");

    let out = Command::new(env!("CARGO_BIN_EXE_ahiru"))
        .args([
            "query".to_string(),
            src.display().to_string(),
            format!("COPY (SELECT id, name FROM t) TO '{}'", ahiru_out.display()),
        ])
        .output()
        .expect("failed to run ahiru");
    assert!(out.status.success(), "ahiru failed:\n{}", String::from_utf8_lossy(&out.stderr));

    let summary = duckdb_scalar(&format!(
        "SELECT count(*) || ',' || sum(id) || ',' || count(name) FROM read_parquet('{}')",
        ahiru_out.display()
    ));
    let expect_sum: u64 = (0..rows as u64).sum();
    let expect_names = rows - rows.div_ceil(5);
    assert_eq!(summary, format!("{rows},{expect_sum},{expect_names}"));

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&ahiru_out);
}
