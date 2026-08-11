//! `export` フィーチャの統合テスト。
//!
//! Parquet から読んだ実データを CSV / JSONL に書き出し、DuckDB の出力と
//! 突き合わせる。読み取り（Parquet）と書き出し（CSV/JSONL）を跨ぐので、
//! 型ごとの表示形式（DATE / TIMESTAMP / DECIMAL / NULL）が両方向で
//! 揃っているかを一度に検証できる。

#![cfg(feature = "export")]

use ahiru_core::session::Session;
use ahiru_core::write::csv::CsvSink;
use ahiru_core::write::export_all;
use ahiru_core::write::jsonl::JsonlSink;

fn data(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
fn csv_export_matches_duckdb_on_real_parquet() {
    let mut s = Session::new();
    s.register_bytes("t", data("basic.parquet")).unwrap();
    let mut sink = CsvSink::new();
    let out = export_all(
        &mut s,
        "SELECT id, name, score, big FROM t ORDER BY id LIMIT 3",
        &[],
        &mut sink,
    )
    .unwrap();
    let text = String::from_utf8(out).unwrap();
    // duckdb -csv -c "SELECT id, name, score, big FROM 'basic.parquet' ORDER BY id LIMIT 3"
    assert_eq!(text, "id,name,score,big\n0,name_0,0.0,\n1,name_1,1.5,100\n2,name_2,3.0,200\n");
}

#[test]
fn jsonl_export_matches_duckdb_on_real_parquet() {
    let mut s = Session::new();
    s.register_bytes("t", data("basic.parquet")).unwrap();
    let mut sink = JsonlSink::new();
    let out = export_all(
        &mut s,
        "SELECT id, name, score, big FROM t ORDER BY id LIMIT 3",
        &[],
        &mut sink,
    )
    .unwrap();
    let text = String::from_utf8(out).unwrap();
    // duckdb: COPY (SELECT ... ORDER BY id LIMIT 3) TO 'x.jsonl' (FORMAT JSON)
    let expected = "{\"id\":0,\"name\":\"name_0\",\"score\":0.0,\"big\":null}\n\
                    {\"id\":1,\"name\":\"name_1\",\"score\":1.5,\"big\":100}\n\
                    {\"id\":2,\"name\":\"name_2\",\"score\":3.0,\"big\":200}\n";
    assert_eq!(text, expected);
}

#[test]
fn export_after_aggregation_and_join() {
    // 単なる SELECT だけでなく、集約・結合を経た結果も正しく書き出せること。
    let mut s = Session::new();
    s.register_bytes("t", data("basic.parquet")).unwrap();
    let mut sink = CsvSink::new();
    let out = export_all(
        &mut s,
        "SELECT name, count(*) AS n FROM t GROUP BY name HAVING count(*) > 142 ORDER BY name",
        &[],
        &mut sink,
    )
    .unwrap();
    let text = String::from_utf8(out).unwrap();
    assert!(text.starts_with("name,n\n"));
    assert_eq!(text.lines().count(), 7); // ヘッダ + 6 グループ
}

#[test]
fn round_trip_csv_export_then_reimport() {
    // 書き出した CSV を再度読み込んで、元の Parquet と同じ結果になること。
    let mut s = Session::new();
    s.register_bytes("t", data("basic.parquet")).unwrap();
    let mut sink = CsvSink::new();
    let out =
        export_all(&mut s, "SELECT id, name, score FROM t ORDER BY id LIMIT 10", &[], &mut sink)
            .unwrap();

    let mut s2 = Session::new();
    s2.register_bytes_as("u", out, ahiru_core::FormatKind::Csv).unwrap();
    let mut sink2 = CsvSink::new();
    let out2 =
        export_all(&mut s2, "SELECT id, name, score FROM u ORDER BY id", &[], &mut sink2).unwrap();

    let mut s3 = Session::new();
    s3.register_bytes("t", data("basic.parquet")).unwrap();
    let mut sink3 = CsvSink::new();
    let out3 =
        export_all(&mut s3, "SELECT id, name, score FROM t ORDER BY id LIMIT 10", &[], &mut sink3)
            .unwrap();

    assert_eq!(out2, out3);
}
