//! QA pass, continued: `COPY (SELECT ...) TO 'file' (FORMAT csv|jsonl)`
//! round-trip checks — export, then re-read with this crate's own reader
//! and confirm the data survived. Builds on `crates/ahiru-cli/tests/copy.rs`
//! (byte-for-byte comparisons against `duckdb`, not edited here) and the
//! `write::csv`/`write::jsonl` unit tests.
//!
//! This file exists specifically because three real round-trip bugs were
//! found and fixed during this QA pass:
//!
//! 1. `write::csv::push_field` didn't quote empty strings, so an empty
//!    VARCHAR value and SQL NULL both serialized as an unquoted empty
//!    field — indistinguishable from each other on re-read (this crate's
//!    own CSV reader treats unquoted-empty as NULL, quoted-`""` as `""`).
//! 2. `write::jsonl`'s `push_value` ran every `Value::Bytes` through the
//!    string escaper regardless of type, so a `Ty::Json` column (Parquet
//!    LIST/MAP/nested-STRUCT, or `format::json`'s nested-value columns)
//!    got double-encoded: `"xs":"[1,2,3]"` instead of `"xs":[1,2,3]`.
//! 3. Both `write::csv::push_value` and `write::jsonl::push_value` only
//!    special-cased DECIMAL formatting for the `Value::I128` storage arm,
//!    but `Ty::Decimal` with precision <= 18 is stored as `Value::I64`
//!    (`vector/types.rs`'s doc on `Decimal`). A DECIMAL(10,2) value of
//!    12.50 (stored as the I64 1250) exported as the bare integer `1250`
//!    in both formats, silently dropping the decimal point.
//!
//! All three are covered by regression tests in `write/csv.rs`/
//! `write/jsonl.rs` themselves; this file adds the equivalent checks at the
//! `Session`/SQL layer (`COPY ... TO`) plus a broader round-trip sweep over
//! other types.

#![cfg(all(feature = "export", feature = "csv", feature = "jsonl"))]

use ahiru_core::catalog::Source;
use ahiru_core::format::csv::CsvFormat;
use ahiru_core::format::jsonl::JsonlFormat;
use ahiru_core::format::{FormatKind, TableFormat};
use ahiru_core::session::{Prepared, Session};
use ahiru_core::vector::Value;

/// `COPY (SELECT ...) TO 'path'` is fully materialized synchronously by
/// `Session::prepare` (in-memory bytes only in this file, so it never needs
/// IO): `Query::copy` carries the written bytes, matching how
/// `crates/ahiru-cli/src/main.rs` handles it (`q.copy.take()`).
fn copy_bytes(sess: &mut Session, sql: &str) -> Vec<u8> {
    match sess.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
        Prepared::Ready(q) => q.copy.expect("COPY statement must set Query::copy").data,
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
    }
}

fn read_back_csv(bytes: Vec<u8>) -> (Vec<String>, Vec<Vec<Value>>) {
    let src = Source::from_bytes(bytes);
    let mut fmt = CsvFormat::new(b',');
    fmt.resolve(&src).unwrap().unwrap();
    let names: Vec<String> = fmt.schema().iter().map(|f| f.name.clone()).collect();
    let proj: Vec<usize> = (0..names.len()).collect();
    let mut rows: Vec<Vec<Value>> = Vec::new();
    for split in 0..fmt.num_splits() {
        let cols = fmt.read_split(&src, split, &proj).unwrap();
        let n = cols.first().map_or(0, |c| c.len());
        for r in 0..n {
            rows.push(cols.iter().map(|c| c.value_at(r)).collect());
        }
    }
    (names, rows)
}

fn read_back_jsonl(bytes: Vec<u8>) -> (Vec<String>, Vec<Vec<Value>>) {
    let src = Source::from_bytes(bytes);
    let mut fmt = JsonlFormat::new();
    fmt.resolve(&src).unwrap().unwrap();
    let names: Vec<String> = fmt.schema().iter().map(|f| f.name.clone()).collect();
    let proj: Vec<usize> = (0..names.len()).collect();
    let mut rows: Vec<Vec<Value>> = Vec::new();
    for split in 0..fmt.num_splits() {
        let cols = fmt.read_split(&src, split, &proj).unwrap();
        let n = cols.first().map_or(0, |c| c.len());
        for r in 0..n {
            rows.push(cols.iter().map(|c| c.value_at(r)).collect());
        }
    }
    (names, rows)
}

fn s(v: &str) -> Value {
    Value::Bytes(v.as_bytes().to_vec())
}

#[test]
fn csv_round_trip_distinguishes_empty_string_from_null() {
    let mut sess = Session::new();
    sess.register_bytes_as("t", b"a,b\n\"\",1\n,2\nx,3\n".to_vec(), FormatKind::Csv).unwrap();
    let out = copy_bytes(&mut sess, "COPY (SELECT a, b FROM t ORDER BY b) TO 'out.csv'");
    let (_, rows) = read_back_csv(out);
    assert_eq!(
        rows,
        vec![
            vec![s(""), Value::I64(1)],
            vec![Value::Null, Value::I64(2)],
            vec![s("x"), Value::I64(3)],
        ]
    );
}

#[test]
fn jsonl_round_trip_of_a_json_typed_parquet_list_column_preserves_nested_structure() {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/list1.parquet");
    let mut sess = Session::new();
    sess.register_bytes_as("t", std::fs::read(p).unwrap(), FormatKind::Parquet).unwrap();
    let out = copy_bytes(
        &mut sess,
        "COPY (SELECT id, xs FROM t WHERE id < 2 ORDER BY id) TO 'out.jsonl'",
    );
    let text = String::from_utf8(out.clone()).unwrap();
    // Embedded as real JSON structure, not a re-escaped string.
    assert!(text.contains("\"xs\":[1,2,3]"), "got: {text}");
    assert!(!text.contains("\"xs\":\"["), "must not be double-encoded, got: {text}");

    let (_, rows) = read_back_jsonl(out);
    assert_eq!(rows.len(), 2);
    // The JSONL reader treats nested values as raw-text VARCHAR
    // (`format::jsonl`'s module doc), so reading it back gives the JSON
    // text verbatim.
    assert_eq!(rows[0][1], s("[1,2,3]"));
}

#[test]
fn csv_round_trip_preserves_decimal_boolean_and_timestamp() {
    let mut sess = Session::new();
    sess.register_bytes_as(
        "t",
        b"d,b,ts\n12.50,true,2024-01-02 03:04:05\n-1.00,false,1999-12-31 23:59:59\n".to_vec(),
        FormatKind::Csv,
    )
    .unwrap();
    // The source CSV infers `d` as DOUBLE (no DECIMAL inference in
    // `format::csv`), so CAST it explicitly to exercise DECIMAL formatting
    // on the way out.
    let out = copy_bytes(
        &mut sess,
        "COPY (SELECT CAST(d AS DECIMAL(10,2)) AS d, b, ts FROM t ORDER BY d) TO 'out.csv'",
    );
    let (_, rows) = read_back_csv(out.clone());
    let text = String::from_utf8(out).unwrap();
    assert_eq!(text, "d,b,ts\n-1.00,false,1999-12-31 23:59:59\n12.50,true,2024-01-02 03:04:05\n");
    assert_eq!(rows.len(), 2);
}

#[test]
fn copy_formats_time_and_blob_as_duckdb_text() {
    let mut sess = Session::new();
    sess.register_bytes_as("t", b"id\n1\n".to_vec(), FormatKind::Csv).unwrap();

    let csv = copy_bytes(
        &mut sess,
        "COPY (SELECT TIME '12:34:56.123456' AS tm, unhex('00a1FEff') AS b FROM t) TO 'out.csv'",
    );
    assert_eq!(csv, b"tm,b\n12:34:56.123456,\\x00\\xA1\\xFE\\xFF\n");

    let jsonl = copy_bytes(
        &mut sess,
        "COPY (SELECT TIME '12:34:56.123456' AS tm, unhex('00a1FEff') AS b FROM t) TO 'out.jsonl'",
    );
    assert_eq!(jsonl, b"{\"tm\":\"12:34:56.123456\",\"b\":\"\\\\x00\\\\xA1\\\\xFE\\\\xFF\"}\n",);
}

#[test]
fn jsonl_round_trip_of_null_and_non_null_values_in_the_same_column() {
    let mut sess = Session::new();
    sess.register_bytes_as(
        "t",
        b"{\"a\":1}\n{\"a\":null}\n{\"a\":3}\n".to_vec(),
        FormatKind::Jsonl,
    )
    .unwrap();
    let out = copy_bytes(&mut sess, "COPY (SELECT a FROM t ORDER BY a) TO 'out.jsonl'");
    let (_, rows) = read_back_jsonl(out);
    // NULLs sort first in this engine's default ordering; just check the
    // multiset of values round-tripped, independent of NULL placement.
    let mut vals: Vec<Value> = rows.into_iter().map(|r| r[0].clone()).collect();
    vals.sort_by_key(|v| match v {
        Value::Null => -1,
        Value::I64(x) => *x,
        other => panic!("unexpected {other:?}"),
    });
    assert_eq!(vals, vec![Value::Null, Value::I64(1), Value::I64(3)]);
}

#[test]
fn csv_round_trip_of_a_field_containing_comma_quote_and_newline() {
    let mut sess = Session::new();
    sess.register_bytes_as(
        "t",
        "a\n\"has, comma\"\n\"has \"\"quote\"\"\"\n\"has\nnewline\"\n".as_bytes().to_vec(),
        FormatKind::Csv,
    )
    .unwrap();
    let out = copy_bytes(&mut sess, "COPY (SELECT a FROM t) TO 'out.csv'");
    let (_, rows) = read_back_csv(out);
    assert_eq!(
        rows,
        vec![vec![s("has, comma")], vec![s("has \"quote\"")], vec![s("has\nnewline")],]
    );
}
