//! Regression tests for the write path (`export` / `ddl` / `dml`), codec
//! delegation on that path, float aggregation, and unnamed-column naming.
//!
//! Every expectation here was cross-checked against `duckdb -csv`; where this
//! engine deliberately diverges (non-finite doubles in JSONL) the reason is on
//! the test itself.

use ahiru_core::error::Code;
use ahiru_core::parquet::Compression;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::{Field, Value};

fn data(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Runs a statement to completion, returning its output schema and rows.
///
/// Deliberately does *not* service `NEED_IO`/`NEED_CODEC`: every fixture here
/// is registered whole, and a codec request reaching the streaming loop would
/// mean the statement under test never used the hook.
fn run(session: &mut Session, sql: &str) -> (Vec<Field>, Vec<Vec<Value>>) {
    let mut q = match session.prepare(sql, &[]).unwrap() {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("unexpected NeedIo for an in-memory source"),
    };
    let schema = q.schema.clone();
    let mut rows = Vec::new();
    loop {
        match session.step(&mut q).unwrap() {
            QueryStep::Batch(mut b) => {
                b.materialize();
                for r in 0..b.num_rows() {
                    rows.push(b.cols.iter().map(|c| c.value_at(r)).collect());
                }
            }
            QueryStep::Done => break,
            _ => panic!("unexpected io/codec request"),
        }
    }
    (schema, rows)
}

fn names(schema: &[Field]) -> Vec<&str> {
    schema.iter().map(|f| f.name.as_str()).collect()
}

fn one_f64(session: &mut Session, sql: &str) -> f64 {
    let (_, rows) = run(session, sql);
    assert_eq!(rows.len(), 1, "{sql}");
    rows[0][0].as_f64().unwrap()
}

// ---------------------------------------------------------------------------
// Codec delegation on the non-streaming statements
// ---------------------------------------------------------------------------

/// A `Session::set_codec_hook` implementation backed by the system `gzip`,
/// mirroring what `ahiru-cli` registers. GZIP inflate is deliberately not in
/// the core (DESIGN.md §6), so a test host has to supply it just like a real
/// one does.
fn gzip_hook(codec: Compression, src: &[u8], out_len: usize) -> ahiru_core::error::Result<Vec<u8>> {
    use ahiru_core::error::{Code, Error};
    use std::io::Write;
    use std::process::{Command, Stdio};
    if codec != Compression::Gzip {
        return Err(Error::new(Code::UnsupportedCodec));
    }
    let mut child = Command::new("gzip")
        .arg("-dc")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| Error::new(Code::BadCompressedData))?;
    let mut stdin = child.stdin.take().unwrap();
    let input = src.to_vec();
    // Feed stdin on its own thread: writing everything first can deadlock once
    // gzip fills the stdout pipe (see `ahiru-cli`'s `gunzip`).
    let feeder = std::thread::spawn(move || stdin.write_all(&input));
    let out = child.wait_with_output().map_err(|_| Error::new(Code::BadCompressedData))?;
    feeder.join().unwrap().map_err(|_| Error::new(Code::BadCompressedData))?;
    if out.stdout.len() != out_len {
        return Err(Error::new(Code::BadCompressedData));
    }
    Ok(out.stdout)
}

fn gzip_session() -> Session {
    let mut s = Session::new();
    s.set_codec_hook(gzip_hook);
    s.register_bytes("g", data("gzip.parquet")).unwrap();
    s
}

// `CREATE TABLE AS` over a GZIP-compressed Parquet source used to fail with
// `[E504] io failed`. Nothing was actually missing: the compressed bytes were
// already in memory (a plain `SELECT count(*)` over the same file works), only
// the inflate was delegated, and `ddl::run_query_to_rows` treated any
// `NEED_CODEC` as fatal.
#[test]
#[cfg(feature = "ddl")]
fn ctas_over_a_gzip_source_services_the_codec_request() {
    let mut s = gzip_session();
    run(&mut s, "CREATE TABLE t AS SELECT * FROM g");
    let (_, rows) = run(&mut s, "SELECT count(*) FROM t");
    assert_eq!(rows[0][0], Value::I64(5000));
}

#[test]
#[cfg(feature = "dml")]
fn insert_select_over_a_gzip_source_services_the_codec_request() {
    let mut s = gzip_session();
    run(&mut s, "CREATE TABLE t AS SELECT * FROM g WHERE 1 = 0");
    let (_, inserted) = run(&mut s, "INSERT INTO t SELECT * FROM g");
    assert_eq!(inserted[0][0], Value::I64(5000));
    let (_, rows) = run(&mut s, "SELECT count(*) FROM t");
    assert_eq!(rows[0][0], Value::I64(5000));
}

#[test]
#[cfg(feature = "export")]
fn export_over_a_gzip_source_services_the_codec_request() {
    use ahiru_core::write::{csv::CsvSink, export_all};
    let mut s = gzip_session();
    let mut sink = CsvSink::new();
    let out = export_all(&mut s, "SELECT * FROM g", &[], &mut sink).unwrap();
    // One header line plus one line per row.
    assert_eq!(String::from_utf8(out).unwrap().lines().count(), 5001);
}

// A host that registers no decompressor still cannot service the request --
// but the failure must name the real problem. It used to report `IoFailed`,
// which points at byte fetching, when nothing was missing at all.
#[test]
#[cfg(feature = "export")]
fn a_codec_request_with_no_hook_reports_unsupported_codec_not_io_failed() {
    use ahiru_core::write::{csv::CsvSink, export_all};
    let mut s = Session::new();
    s.register_bytes("g", data("gzip.parquet")).unwrap();
    let mut sink = CsvSink::new();
    let r = export_all(&mut s, "SELECT * FROM g", &[], &mut sink);
    assert_eq!(ahiru_core::error::code_of(r), Some(Code::UnsupportedCodec));
}

// A codec the host cannot handle must surface the hook's own error rather than
// being reported as an I/O failure.
#[test]
#[cfg(feature = "export")]
fn a_hook_that_rejects_the_codec_propagates_its_error() {
    use ahiru_core::write::{csv::CsvSink, export_all};
    fn refuse(_: Compression, _: &[u8], _: usize) -> ahiru_core::error::Result<Vec<u8>> {
        Err(ahiru_core::error::Error::new(Code::BadCompressedData))
    }
    let mut s = Session::new();
    s.set_codec_hook(refuse);
    s.register_bytes("g", data("gzip.parquet")).unwrap();
    let mut sink = CsvSink::new();
    let r = export_all(&mut s, "SELECT * FROM g", &[], &mut sink);
    assert_eq!(ahiru_core::error::code_of(r), Some(Code::BadCompressedData));
}

// The streaming query path is untouched: it still hands `NEED_CODEC` to the
// host between two `step` calls even when a hook is registered, so an existing
// JS driving loop keeps working exactly as before.
#[test]
fn the_streaming_path_still_returns_need_codec_to_the_host() {
    let mut s = gzip_session();
    let mut q = match s.prepare("SELECT count(*) FROM g", &[]).unwrap() {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("unexpected NeedIo"),
    };
    let mut saw_codec = false;
    let mut total = None;
    loop {
        match s.step(&mut q).unwrap() {
            QueryStep::Batch(mut b) => {
                b.materialize();
                total = Some(b.cols[0].value_at(0));
            }
            QueryStep::NeedCodec(reqs) => {
                saw_codec = true;
                for r in reqs {
                    let src = data("gzip.parquet");
                    let start = r.offset as usize;
                    let end = start + r.len as usize;
                    let decoded = gzip_hook(r.codec, &src[start..end], r.out_len as usize).unwrap();
                    s.provide_decoded(r.table, r.part, r.offset, r.len, decoded).unwrap();
                }
            }
            QueryStep::NeedIo(_) => panic!("unexpected NeedIo"),
            QueryStep::Done => break,
        }
    }
    assert!(saw_codec, "a GZIP source must still ask the host on the streaming path");
    assert_eq!(total, Some(Value::I64(5000)));
}

// ---------------------------------------------------------------------------
// JSONL export: non-finite doubles and DECIMAL
// ---------------------------------------------------------------------------

#[cfg(all(feature = "export", feature = "jsonl"))]
fn jsonl(session: &mut Session, sql: &str) -> String {
    use ahiru_core::write::{export_all, jsonl::JsonlSink};
    let mut sink = JsonlSink::new();
    String::from_utf8(export_all(session, sql, &[], &mut sink).unwrap()).unwrap()
}

// NaN and +/-Infinity used to be written as `null`, which is indistinguishable
// from a real SQL NULL -- a silent value change. RFC 8259 has no literal for
// them, and `duckdb`'s unquoted `NaN` is not JSON and is rejected by this
// crate's own reader, so they are written as quoted strings instead (see
// `write::jsonl::push_f64`).
#[test]
#[cfg(all(feature = "export", feature = "jsonl", feature = "csv"))]
fn jsonl_export_writes_non_finite_doubles_as_quoted_strings() {
    let mut s = Session::new();
    s.register_bytes_as("t", b"id\n1\n".to_vec(), ahiru_core::FormatKind::Csv).unwrap();
    let out = jsonl(
        &mut s,
        "SELECT 'nan'::DOUBLE AS n, 'inf'::DOUBLE AS p, '-inf'::DOUBLE AS m, \
         CAST(NULL AS DOUBLE) AS z FROM t",
    );
    assert_eq!(out, "{\"n\":\"NaN\",\"p\":\"Infinity\",\"m\":\"-Infinity\",\"z\":null}\n");
}

// `duckdb -csv -c "COPY (SELECT CAST(12.345 AS DECIMAL(10,3)) d) TO 'x.jsonl'"`
// writes `{"d":12.345}`. This used to quote it, turning a numeric column into
// a string for every consumer of the file.
#[test]
#[cfg(all(feature = "export", feature = "jsonl", feature = "csv"))]
fn jsonl_export_writes_decimal_as_a_json_number() {
    let mut s = Session::new();
    s.register_bytes_as("t", b"id\n1\n".to_vec(), ahiru_core::FormatKind::Csv).unwrap();
    let out = jsonl(
        &mut s,
        "SELECT CAST(12.345 AS DECIMAL(10,3)) AS narrow, \
         CAST(-12.345 AS DECIMAL(30,4)) AS wide FROM t",
    );
    assert_eq!(out, "{\"narrow\":12.345,\"wide\":-12.3450}\n");
}

// What we write, we must be able to read. The non-finite values come back as
// their text (castable straight back to DOUBLE) and the decimal as a number.
#[test]
#[cfg(all(feature = "export", feature = "jsonl", feature = "csv"))]
fn jsonl_export_round_trips_through_this_engines_own_reader() {
    let mut s = Session::new();
    s.register_bytes_as("t", b"id\n1\n".to_vec(), ahiru_core::FormatKind::Csv).unwrap();
    let out = jsonl(
        &mut s,
        "SELECT 'nan'::DOUBLE AS n, 'inf'::DOUBLE AS p, '-inf'::DOUBLE AS m, \
         CAST(12.345 AS DECIMAL(10,3)) AS d FROM t",
    );

    let mut back = Session::new();
    back.register_bytes_as("u", out.into_bytes(), ahiru_core::FormatKind::Jsonl).unwrap();
    let (_, rows) =
        run(&mut back, "SELECT CAST(n AS DOUBLE), CAST(p AS DOUBLE), CAST(m AS DOUBLE), d FROM u");
    assert_eq!(rows.len(), 1);
    assert!(rows[0][0].as_f64().unwrap().is_nan());
    assert_eq!(rows[0][1].as_f64().unwrap(), f64::INFINITY);
    assert_eq!(rows[0][2].as_f64().unwrap(), f64::NEG_INFINITY);
    assert_eq!(rows[0][3].as_f64().unwrap(), 12.345);
}

// ---------------------------------------------------------------------------
// Compensated float summation
// ---------------------------------------------------------------------------

// `duckdb -csv -c "SELECT sum(x) FROM (SELECT 0.1 AS x FROM range(10))"` gives
// exactly 1.0; naive f64 accumulation gave 0.9999999999999999.
#[test]
fn sum_and_avg_over_doubles_are_compensated() {
    let mut s = Session::new();
    assert_eq!(one_f64(&mut s, "SELECT sum(x) FROM (SELECT 0.1 AS x FROM range(10))"), 1.0);
    assert_eq!(one_f64(&mut s, "SELECT avg(x) FROM (SELECT 0.1 AS x FROM range(10))"), 0.1);
    // 1000 terms: the naive error grows with the row count.
    assert_eq!(one_f64(&mut s, "SELECT sum(x) FROM (SELECT 0.1 AS x FROM range(1000))"), 100.0);
}

#[test]
fn grouped_sum_over_doubles_is_compensated_per_group() {
    let mut s = Session::new();
    let (_, rows) = run(
        &mut s,
        "SELECT g, sum(x) FROM (SELECT i % 2 AS g, 0.1 AS x FROM range(20) t(i)) \
         GROUP BY g ORDER BY g",
    );
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0][1].as_f64().unwrap(), 1.0);
    assert_eq!(rows[1][1].as_f64().unwrap(), 1.0);
}

// Neumaier's correction term is `inf - inf` = NaN once a value or the running
// total is infinite. Carrying that would turn a well-defined infinite sum into
// NaN, so it is dropped -- these all match `duckdb`.
#[test]
fn compensation_does_not_disturb_non_finite_sums() {
    let mut s = Session::new();
    let inf = "SELECT sum(x) FROM \
        (SELECT CASE WHEN i = 0 THEN 'inf'::DOUBLE ELSE i::DOUBLE END AS x FROM range(3) t(i))";
    assert_eq!(one_f64(&mut s, inf), f64::INFINITY);

    let nan = "SELECT sum(x) FROM \
        (SELECT CASE WHEN i = 0 THEN 'nan'::DOUBLE ELSE 1.0::DOUBLE END AS x FROM range(3) t(i))";
    assert!(one_f64(&mut s, nan).is_nan());

    let overflow = "SELECT sum(x) FROM (SELECT 1e308::DOUBLE AS x FROM range(3))";
    assert_eq!(one_f64(&mut s, overflow), f64::INFINITY);
}

// A zero compensation is not added back, so a sum of nothing but negative
// zeros keeps its sign instead of being flipped by `-0.0 + 0.0`.
#[test]
fn negative_zero_sum_keeps_its_sign() {
    let mut s = Session::new();
    let v = one_f64(&mut s, "SELECT sum(x) FROM (SELECT -0.0::DOUBLE AS x FROM range(3))");
    assert!(v == 0.0 && v.is_sign_negative(), "expected -0.0, got {v}");
}

#[test]
fn sum_of_no_rows_is_still_null() {
    let mut s = Session::new();
    let (_, rows) =
        run(&mut s, "SELECT sum(x) FROM (SELECT 0.1 AS x FROM range(10)) WHERE x > 100");
    assert_eq!(rows[0][0], Value::Null);
}

// ---------------------------------------------------------------------------
// Names of unnamed expression columns
// ---------------------------------------------------------------------------

// The names used to be derived from expression-arena ids, so they were gapped
// and moved whenever an unrelated part of the query changed: `SELECT 1, 2`
// gave `col0, col1` but `SELECT 1+1, 2` gave `col2, col3`. They are numbered
// by output position now.
#[test]
fn unnamed_expression_columns_are_numbered_by_output_position() {
    let mut s = Session::new();
    let (schema, _) = run(&mut s, "SELECT 1, 2 FROM range(1)");
    assert_eq!(names(&schema), ["col0", "col1"]);

    let (schema, _) = run(&mut s, "SELECT 1 + 1, 2 FROM range(1)");
    assert_eq!(names(&schema), ["col0", "col1"]);

    let (schema, _) = run(&mut s, "SELECT count(*), sum(i), sum(i + 1) FROM range(3) t(i)");
    assert_eq!(names(&schema), ["col0", "col1", "col2"]);
}

// Aliases, column references and `UNNEST` keep the names they already had; the
// positional numbering applies only to the columns that had none.
#[test]
fn named_columns_are_unaffected_by_positional_numbering() {
    let mut s = Session::new();
    let (schema, _) = run(&mut s, "SELECT i, 1 AS one, i + 1 FROM range(1) t(i)");
    assert_eq!(names(&schema), ["i", "one", "col2"]);
}

// `CREATE TABLE AS` persists these names, so an unstable one was baked into
// the stored schema and shown by `DESCRIBE`.
#[test]
#[cfg(feature = "ddl")]
fn ctas_persists_positional_column_names() {
    let mut s = Session::new();
    run(&mut s, "CREATE TABLE t AS SELECT 1 + 1, 2 FROM range(1)");
    let (_, rows) = run(&mut s, "DESCRIBE t");
    let got: Vec<String> = rows
        .iter()
        .map(|r| match &r[0] {
            Value::Bytes(b) => String::from_utf8_lossy(b).into_owned(),
            other => panic!("unexpected {other:?}"),
        })
        .collect();
    assert_eq!(got, ["col0", "col1"]);
}

// ---------------------------------------------------------------------------
// UPDATE / DELETE forms
// ---------------------------------------------------------------------------

#[cfg(feature = "dml")]
fn dml_session() -> Session {
    let mut s = Session::new();
    run(&mut s, "CREATE TABLE al (a INTEGER, b INTEGER)");
    run(&mut s, "INSERT INTO al VALUES (1, 10), (3, 30)");
    s
}

// `UPDATE al SET b = al.b + 1 WHERE al.a = 3` used to fail with `[E400] table
// not found`: the predicate was compiled against a scope with no qualifier at
// all, so `al.b` looked like a column of an unknown table.
#[test]
#[cfg(feature = "dml")]
fn update_accepts_table_qualified_columns() {
    let mut s = dml_session();
    let (_, n) = run(&mut s, "UPDATE al SET b = al.b + 1 WHERE al.a = 3");
    assert_eq!(n[0][0], Value::I64(1));
    let (_, rows) = run(&mut s, "SELECT a, b FROM al ORDER BY a");
    assert_eq!(rows[0], vec![Value::I32(1), Value::I32(10)]);
    assert_eq!(rows[1], vec![Value::I32(3), Value::I32(31)]);
}

#[test]
#[cfg(feature = "dml")]
fn delete_accepts_table_qualified_columns() {
    let mut s = dml_session();
    let (_, n) = run(&mut s, "DELETE FROM al WHERE al.a = 1");
    assert_eq!(n[0][0], Value::I64(1));
    let (_, rows) = run(&mut s, "SELECT a FROM al");
    assert_eq!(rows, vec![vec![Value::I32(3)]]);
}

// An unknown qualifier must still be an error, not silently ignored.
#[test]
#[cfg(feature = "dml")]
fn an_unknown_qualifier_is_still_rejected() {
    let mut s = dml_session();
    assert_eq!(
        ahiru_core::error::code_of(s.prepare("DELETE FROM al WHERE zz.a = 1", &[])),
        Some(Code::TableNotFound)
    );
}

// `WHERE NULL` is UNKNOWN, so it matches nothing -- the same rule `SELECT`
// already followed. DML used to raise `[E404] type mismatch` instead.
#[test]
#[cfg(feature = "dml")]
fn where_null_matches_no_rows_in_dml() {
    let mut s = dml_session();
    let (_, n) = run(&mut s, "DELETE FROM al WHERE NULL");
    assert_eq!(n[0][0], Value::I64(0));
    let (_, n) = run(&mut s, "UPDATE al SET b = 0 WHERE NULL");
    assert_eq!(n[0][0], Value::I64(0));
    let (_, rows) = run(&mut s, "SELECT count(*) FROM al");
    assert_eq!(rows[0][0], Value::I64(2));
}
