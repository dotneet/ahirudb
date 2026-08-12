//! `COPY (SELECT ...) TO 'path' [(FORMAT csv|jsonl|parquet)]` のエンドツーエンドテスト。
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

/// DuckDB で 1 行 1 列を返すクエリを実行し、その値を文字列で返す。
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
    assert!(!out.status.success(), "未対応フォーマットの COPY は失敗するはず");
    assert!(!out_path.exists(), "失敗したのにファイルが作られている: {}", out_path.display());
}

// --- Parquet -----------------------------------------------------------------
// Parquet の出力はバイト単位では DuckDB と一致しない（エンコーディング・
// 圧縮・統計の選び方が違う。ahirudb 側は非圧縮 PLAIN のみ）。ここで確かめ
// たいのは「DuckDB が読めるか」なので、比較は書き出したファイルを DuckDB に
// 読ませた結果と元データの突き合わせで行う。ahirudb 自身の reader との往復は
// `crates/ahiru-core/src/write/parquet/tests.rs` 側で見ている。

/// 書き出した Parquet を DuckDB に読ませ、元の CSV と 1 行も食い違わない
/// ことを対称差の件数で確かめる。
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
    assert_eq!(diff, "0", "DuckDB から見た行が元の CSV と食い違う");

    let _ = std::fs::remove_file(&ahiru_out);
}

/// DuckDB が推論する列型が、書き出し時の SQL 型と一致すること。
/// 論理型（`DATE` / `DECIMAL` / `UUID` / `TIMESTAMP(isAdjustedToUTC)` など）
/// をフッタに正しく書けているかは、値だけを比べても分からない。
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

/// 1 ファイルに複数の RowGroup が並ぶ場合も DuckDB から通しで読めること。
/// 行数はテスト内で生成する（リポジトリに 12 万行のデータを置きたくない）。
#[test]
fn copy_parquet_spanning_multiple_row_groups_reads_back_intact() {
    skip_without_duckdb!();
    let src = tmp_path("parquet_multi_rg_src", "csv");
    let ahiru_out = tmp_path("parquet_multi_rg", "parquet");

    // `write::parquet::ROW_GROUP_ROWS` は 122880。それを確実に跨ぐ行数。
    let rows = 130_000usize;
    let mut csv = String::from("id,name\n");
    for i in 0..rows {
        // 5 行に 1 行を NULL にして、definition level が RLE と bit-packed の
        // 両方を通るようにする。
        if i % 5 == 0 {
            csv.push_str(&format!("{i},\n"));
        } else {
            csv.push_str(&format!("{i},n{i}\n"));
        }
    }
    std::fs::write(&src, csv).expect("テンポラリ CSV を書けない");

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
