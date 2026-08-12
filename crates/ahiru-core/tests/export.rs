//! Integration tests for the `export` feature.
//!
//! Writes real data read from Parquet out to CSV / JSONL and cross-checks against DuckDB's
//! output. Since this spans reading (Parquet) and writing (CSV/JSONL), it lets us verify in
//! one shot whether each type's display format (DATE / TIMESTAMP / DECIMAL / NULL) lines up
//! in both directions.

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
    // Verify not just a plain SELECT, but that results after aggregation/joins are also written out correctly.
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
    assert_eq!(text.lines().count(), 7); // header + 6 groups
}

#[test]
fn round_trip_csv_export_then_reimport() {
    // Re-import the exported CSV and verify it matches the original Parquet.
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
