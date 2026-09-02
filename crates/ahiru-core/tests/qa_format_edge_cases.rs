//! QA pass over `format::csv`/`format::jsonl`/`format::json`: integration-level
//! edge cases exercised through `Session`/SQL, on top of the extensive
//! unit-level coverage already in each format module. See the module docs in
//! `crates/ahiru-core/src/format/{csv,jsonl,json}.rs` for the documented
//! behaviors this file cross-checks.
//!
//! Scope note: this file is part of the QA pass covering
//! `format/{csv,jsonl,json,partitioned}.rs`, `parquet/nested.rs`, `ddl.rs`,
//! `dml.rs`, `write/`, and `catalog.rs`. New file, not an edit to the shared
//! `crates/ahiru-core/tests/{multi_file_smoke,json_files,nested_files}.rs`.

use ahiru_core::catalog::Source;
use ahiru_core::error::{code_of, Code};
use ahiru_core::format::csv::CsvFormat;
use ahiru_core::format::{FormatKind, TableFormat};
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn run_all(sql: &str, s: &mut Session) -> Vec<Vec<Value>> {
    let mut q = match s.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: in-memory bytes never need IO"),
    };
    let mut rows = Vec::new();
    loop {
        match s.step(&mut q).unwrap() {
            QueryStep::Batch(mut b) => {
                b.materialize();
                for r in 0..b.num_rows() {
                    rows.push(b.cols.iter().map(|c| c.value_at(r)).collect());
                }
            }
            QueryStep::NeedIo(_) | QueryStep::NeedCodec(_) => panic!("unexpected suspend"),
            QueryStep::Done => break,
        }
    }
    rows
}

fn s(v: &str) -> Value {
    Value::Bytes(v.as_bytes().to_vec())
}

// --- CSV: split-boundary quote handling, verified at the byte level -------

/// Regression test for a real, fixed bug (previously pinned by this same
/// test under a different name as an "accepted limitation"): a quoted
/// newline landing right at/after a fixed-size split boundary used to be
/// misread as a record terminator, injecting one spurious all-NULL row. Full
/// quote-state recovery at an arbitrary split boundary needs unbounded
/// backward context (`format::csv`'s module doc, and `docs/sql/
/// limitations.md`), so the fix instead has `CsvFormat::resolve` check its
/// leading sample for a `"` byte and, if found, force the whole file to read
/// as a single split — sidestepping the ambiguity entirely rather than
/// guessing wrong. This test pins that down: with the fixture below, a
/// `split_bytes` small enough to have put a boundary exactly inside the
/// quoted embedded newline is now simply ignored, `num_splits()` stays 1,
/// and both records come back correctly with no spurious row.
#[test]
fn quoted_embedded_newline_no_longer_injects_a_spurious_row_at_a_split_boundary() {
    // Row 1's second field is quoted and contains a literal '\n'. Absolute
    // byte offset 12 (right after "line1") is exactly that embedded
    // newline; data_start (after the "a,b\n" header) is 4, so split_bytes=8
    // would have put a split boundary exactly there under the old,
    // quote-unaware resync.
    let data = "a,b\n1,\"line1\nline2\"\n2,x\n";
    let src = Source::from_bytes(data.as_bytes().to_vec());
    let mut f = CsvFormat::new(b',');
    f.resolve(&src).unwrap().unwrap();
    f.split_bytes = 8;
    assert_eq!(
        f.num_splits(),
        1,
        "a file whose leading sample contains a quote is always read as a single split"
    );

    let mut rows: Vec<(Value, Value)> = Vec::new();
    for split in 0..f.num_splits() {
        let cols = f.read_split(&src, split, &[0, 1]).expect("read_split must not error");
        for i in 0..cols[0].len() {
            rows.push((cols[0].value_at(i), cols[1].value_at(i)));
        }
    }
    // Exactly the two real records -- no spurious (NULL, NULL) row.
    assert_eq!(
        rows,
        vec![(Value::I64(1), s("line1\nline2")), (Value::I64(2), s("x"))],
        "no phantom row from the (now-unreachable) mid-quote resync"
    );
}

// --- CSV: through Session/SQL ----------------------------------------------

#[test]
fn tsv_kind_is_tab_delimited_through_sql() {
    let mut sess = Session::new();
    sess.register_bytes_as("t", b"id\tname\n1\tapple\n2\tpear\n".to_vec(), FormatKind::Tsv)
        .unwrap();
    let rows = run_all("SELECT id, name FROM t ORDER BY id", &mut sess);
    assert_eq!(rows, vec![vec![Value::I64(1), s("apple")], vec![Value::I64(2), s("pear")]]);
}

#[test]
fn crlf_and_lf_records_mixed_in_the_same_file_are_both_accepted() {
    let mut sess = Session::new();
    sess.register_bytes_as("t", b"a,b\r\n1,x\n2,y\r\n3,z\n".to_vec(), FormatKind::Csv).unwrap();
    let rows = run_all("SELECT a, b FROM t ORDER BY a", &mut sess);
    assert_eq!(
        rows,
        vec![vec![Value::I64(1), s("x")], vec![Value::I64(2), s("y")], vec![Value::I64(3), s("z")],]
    );
}

#[test]
fn csv_values_outside_the_sampled_type_are_a_conversion_error_not_a_silent_null() {
    // The inferred type comes from the first SAMPLE_ROWS rows only
    // (`format::csv`'s module doc), so a later row can genuinely fall
    // outside it. Such a value used to be turned into NULL, which quietly
    // dropped data the file plainly contains and made `count(n)` disagree
    // with `count(*)` for no visible reason -- the "silently wrong answer"
    // `docs/DESIGN.md` §15 says the engine never produces. It is now
    // `InvalidCast`, which is what DuckDB reports for the same file.
    let mut csv = String::from("n\n");
    for i in 0..1001 {
        csv.push_str(&format!("{i}\n"));
    }
    csv.push_str("not_a_number\n");
    let mut sess = Session::new();
    sess.register_bytes_as("t", csv.into_bytes(), FormatKind::Csv).unwrap();
    let mut q = match sess.prepare("SELECT count(*) FROM t", &[]).unwrap() {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("in-memory bytes never need IO"),
    };
    assert_eq!(code_of(sess.step(&mut q)), Some(Code::InvalidCast));

    // An empty field keeps its NULL meaning -- that is the only thing CSV
    // can spell, and it must not be caught up in the new error.
    let mut sess = Session::new();
    // (A second column keeps the empty cell on a line of its own from being
    // skipped as a blank line.)
    sess.register_bytes_as("t", b"k,n\n1,1\n2,\n3,3\n".to_vec(), FormatKind::Csv).unwrap();
    assert_eq!(
        run_all("SELECT count(*), count(n) FROM t", &mut sess),
        [[Value::I64(3), Value::I64(2)]]
    );
}

#[test]
fn multi_part_csv_table_with_mismatched_column_names_is_rejected() {
    // Same invariant `multi_file_smoke.rs` checks for Parquet
    // (`catalog::unify_schema` requires column names to line up, not just
    // types), exercised for CSV parts instead.
    let mut sess = Session::new();
    sess.register_multi_bytes(
        "t",
        vec![("a.csv".into(), b"k,v\n1,10\n".to_vec()), ("b.csv".into(), b"k,w\n2,20\n".to_vec())],
        FormatKind::Csv,
    )
    .unwrap();
    let r = sess.prepare("SELECT count(*) FROM t", &[]);
    assert_eq!(code_of(r), Some(Code::TypeMismatch));
}

#[test]
fn header_only_csv_part_with_all_varchar_columns_unifies_with_a_typed_data_part() {
    // A header-only part has no rows to sample, so every column infers as
    // VARCHAR (`Cand::Empty` -> `Ty::Varchar`, `format::csv`'s widening
    // lattice). When the sibling part's columns are also VARCHAR (as here:
    // "name" holds text in both), `Ty::unify` trivially agrees and the parts
    // combine, contributing zero rows from the empty part.
    let mut sess = Session::new();
    sess.register_multi_bytes(
        "t",
        vec![
            ("empty.csv".into(), b"name\n".to_vec()),
            ("data.csv".into(), b"name\na\nb\n".to_vec()),
        ],
        FormatKind::Csv,
    )
    .unwrap();
    let rows = run_all("SELECT count(*) FROM t", &mut sess);
    assert_eq!(rows, [[Value::I64(2)]]);
}

#[test]
fn header_only_csv_part_with_a_numeric_sibling_column_widens_to_varchar() {
    // Companion to the case above. An empty part's columns always default to
    // VARCHAR regardless of what the *other* parts' data looks like, so
    // unioning a header-only CSV with a part holding actual numeric data in
    // the same column position used to be rejected outright
    // (`Ty::unify(Varchar, BigInt)` is `None`) even though the empty part
    // contributes zero conflicting rows -- a placeholder or template file in
    // a glob made the whole table unreadable.
    //
    // Both parts' types are *guesses* from a leading sample, so
    // `catalog::unify_schema` now widens the disagreement to VARCHAR rather
    // than failing, which is what DuckDB does when it unions text files.
    // (Parquet, whose schema is declared rather than sniffed, still reports a
    // real type conflict -- see `multi_file_smoke.rs`.)
    let mut sess = Session::new();
    sess.register_multi_bytes(
        "t",
        vec![
            ("empty.csv".into(), b"id,name\n".to_vec()),
            ("data.csv".into(), b"id,name\n1,a\n2,b\n".to_vec()),
        ],
        FormatKind::Csv,
    )
    .unwrap();
    let rows = run_all("SELECT id FROM t ORDER BY id", &mut sess);
    assert_eq!(rows, [[s("1")], [s("2")]]);
}

#[test]
fn multi_part_csv_columns_that_sniff_differently_union_as_text() {
    // The repro that motivated the widening above: one file's column holds a
    // number and the next file's is empty, so the two parts sniff BIGINT and
    // VARCHAR. DuckDB unions such files to VARCHAR; this used to be
    // `TypeMismatch`.
    let mut sess = Session::new();
    sess.register_multi_bytes(
        "t",
        vec![("a.csv".into(), b"id,v\n1,10\n".to_vec()), ("b.csv".into(), b"id,v\n2,\n".to_vec())],
        FormatKind::Csv,
    )
    .unwrap();
    let rows = run_all("SELECT id, v FROM t ORDER BY id", &mut sess);
    assert_eq!(rows, [[Value::I64(1), s("10")], [Value::I64(2), Value::Null]]);
}

// --- JSONL / JSON: schema promotion through SQL ----------------------------

#[test]
fn jsonl_missing_key_and_explicit_null_are_indistinguishable_through_sql() {
    // `format::jsonl`'s module doc: a key absent from a row and a key
    // present with an explicit JSON `null` both read back as SQL NULL.
    // Confirm this holds through `IS NULL` in actual SQL, not just at the
    // `Value` level.
    let mut sess = Session::new();
    sess.register_bytes_as(
        "t",
        b"{\"a\":1,\"b\":2}\n{\"a\":3,\"b\":null}\n".to_vec(),
        FormatKind::Jsonl,
    )
    .unwrap();
    let rows = run_all("SELECT a FROM t WHERE b IS NULL ORDER BY a", &mut sess);
    assert_eq!(rows, [[Value::I64(3)]]);
}

#[test]
fn jsonl_int_then_double_rows_promote_the_whole_column_and_aggregate_correctly() {
    // Schema inference widens across the *union* of sampled rows
    // (BOOLEAN -> BIGINT -> DOUBLE -> VARCHAR). Confirm an aggregate over a
    // promoted column returns the DOUBLE-typed sum, not a truncated one.
    let mut sess = Session::new();
    sess.register_bytes_as("t", b"{\"n\":1}\n{\"n\":2.5}\n{\"n\":3}\n".to_vec(), FormatKind::Jsonl)
        .unwrap();
    let rows = run_all("SELECT sum(n) FROM t", &mut sess);
    assert_eq!(rows, [[Value::F64(6.5)]]);
}

#[test]
fn json_array_top_level_nested_and_scalar_widen_to_json_type_through_sql() {
    // `format::json`'s module doc: a column mixing nested values with a
    // scalar widens to `Ty::Json`, unlike jsonl's fallback to `Ty::Varchar`.
    // Confirm the resulting column is queryable as JSON through SQL
    // (`json_extract`), not just correctly typed.
    let mut sess = Session::new();
    sess.register_bytes_as(
        "t",
        b"[{\"id\":1,\"v\":[1,2,3]},{\"id\":2,\"v\":5}]".to_vec(),
        FormatKind::Json,
    )
    .unwrap();
    let rows = run_all("SELECT id, v FROM t ORDER BY id", &mut sess);
    assert_eq!(rows[0][1], s("[1,2,3]"));
    assert_eq!(rows[1][1], s("5"));
}

#[test]
fn empty_csv_file_registers_as_a_zero_row_table_not_an_error() {
    let mut sess = Session::new();
    sess.register_bytes_as("t", Vec::new(), FormatKind::Csv).unwrap();
    let rows = run_all("SELECT count(*) FROM t", &mut sess);
    assert_eq!(rows, [[Value::I64(0)]]);
}

#[test]
fn csv_nan_and_infinity_round_trip_as_double() {
    let mut sess = Session::new();
    sess.register_bytes_as("t", b"v\n1.5\nNaN\nInfinity\n-Infinity\n".to_vec(), FormatKind::Csv)
        .unwrap();
    let rows = run_all("SELECT v FROM t", &mut sess);
    assert_eq!(rows[0][0], Value::F64(1.5));
    match &rows[1][0] {
        Value::F64(x) => assert!(x.is_nan()),
        other => panic!("NaN row: {other:?}"),
    }
    match &rows[2][0] {
        Value::F64(x) => assert!(x.is_infinite() && *x > 0.0),
        other => panic!("inf row: {other:?}"),
    }
    match &rows[3][0] {
        Value::F64(x) => assert!(x.is_infinite() && *x < 0.0),
        other => panic!("-inf row: {other:?}"),
    }
}

#[test]
fn jsonl_z_suffix_is_a_timestamp() {
    let mut sess = Session::new();
    sess.register_bytes_as(
        "t",
        br#"{"t":"2020-01-01T00:00:00"}
{"t":"2020-01-01T00:00:00Z"}
"#
        .to_vec(),
        FormatKind::Jsonl,
    )
    .unwrap();
    let rows = run_all("SELECT t FROM t", &mut sess);
    assert_eq!(rows[0][0], Value::I64(1_577_836_800_000_000));
    assert_eq!(rows[1][0], Value::I64(1_577_836_800_000_000));
}

#[test]
fn cr_only_line_endings_are_read_instead_of_returning_zero_rows() {
    // A classic Mac (CR-only) file used to parse as one giant record: a garbage single-row header
    // and `(0 rows)`, with no error at all. DuckDB reads the two data rows.
    let mut sess = Session::new();
    sess.register_bytes_as("t", b"a,b\r1,x\r3,y\r".to_vec(), FormatKind::Csv).unwrap();
    let rows = run_all("SELECT a, b FROM t ORDER BY a", &mut sess);
    assert_eq!(rows, vec![vec![Value::I64(1), s("x")], vec![Value::I64(3), s("y")]]);
}

#[test]
fn zero_padded_digit_strings_keep_their_padding() {
    // `007` read as a number comes back as `7`, silently corrupting zip codes, account numbers and
    // product IDs. DuckDB keeps such a column VARCHAR; a lone `0` and `0.5` stay numeric.
    let mut sess = Session::new();
    sess.register_bytes_as("t", b"a\n007\n0123\n".to_vec(), FormatKind::Csv).unwrap();
    assert_eq!(run_all("SELECT a FROM t", &mut sess), vec![vec![s("007")], vec![s("0123")]]);

    let mut sess = Session::new();
    sess.register_bytes_as("t", b"a\n0\n1\n".to_vec(), FormatKind::Csv).unwrap();
    assert_eq!(
        run_all("SELECT a FROM t ORDER BY a", &mut sess),
        [[Value::I64(0)], [Value::I64(1)]]
    );
}

#[test]
fn booleans_mixed_with_numbers_widen_to_text_instead_of_nulling_the_booleans() {
    // Widening BOOLEAN + BIGINT to BIGINT nulled the `true` row -- data the inference sample
    // itself had just seen. DuckDB resolves the mixture to VARCHAR.
    for body in [&b"a\ntrue\n1\n"[..], &b"a\ntrue\n1.5\n"[..]] {
        let mut sess = Session::new();
        sess.register_bytes_as("t", body.to_vec(), FormatKind::Csv).unwrap();
        let rows = run_all("SELECT a FROM t", &mut sess);
        assert_eq!(rows[0], vec![s("true")], "{:?}", String::from_utf8_lossy(body));
        assert_ne!(rows[1][0], Value::Null);
    }
}

#[test]
fn dates_mixed_with_timestamps_read_the_date_as_midnight() {
    // The column widens to TIMESTAMP, and a date-only value used to fail to parse there and become
    // NULL. DuckDB reads it as that date's midnight.
    let mut sess = Session::new();
    sess.register_bytes_as("t", b"a\n2024-01-01 10:00:00\n2024-01-02\n".to_vec(), FormatKind::Csv)
        .unwrap();
    let rows = run_all("SELECT count(a) FROM t", &mut sess);
    assert_eq!(rows, [[Value::I64(2)]]);
    // duckdb: epoch_us(TIMESTAMP '2024-01-02') = 1704153600000000
    let rows = run_all("SELECT a FROM t ORDER BY a", &mut sess);
    assert_eq!(rows[1], vec![Value::I64(1_704_153_600_000_000)]);
}

#[test]
fn jsonl_values_outside_the_sampled_type_are_a_conversion_error() {
    // Same rule as CSV: inference sees a bounded leading sample, and a later line holding a value
    // the inferred type cannot express is `InvalidCast` rather than a silently NULLed row.
    let mut jsonl = String::new();
    for _ in 0..1001 {
        jsonl.push_str("{\"a\":1}\n");
    }
    jsonl.push_str("{\"a\":2.5}\n");
    let mut sess = Session::new();
    sess.register_bytes_as("t", jsonl.into_bytes(), FormatKind::Jsonl).unwrap();
    let mut q = match sess.prepare("SELECT count(*) FROM t", &[]).unwrap() {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("in-memory bytes never need IO"),
    };
    assert_eq!(code_of(sess.step(&mut q)), Some(Code::InvalidCast));

    // A missing key and an explicit `null` still mean NULL.
    let mut sess = Session::new();
    sess.register_bytes_as("t", b"{\"a\":1}\n{}\n{\"a\":null}\n".to_vec(), FormatKind::Jsonl)
        .unwrap();
    assert_eq!(
        run_all("SELECT count(*), count(a) FROM t", &mut sess),
        [[Value::I64(3), Value::I64(1)]]
    );
}

#[test]
fn ndjson_lines_that_are_not_objects_read_as_a_raw_json_column() {
    // NDJSON permits any JSON value per line. Such a file used to be rejected with a syntax
    // error; DuckDB gives it a single column named `json` holding each line's raw text.
    let mut sess = Session::new();
    sess.register_bytes_as("t", b"1\n[1,2]\n\"s\"\n".to_vec(), FormatKind::Jsonl).unwrap();
    let rows = run_all("SELECT json FROM t", &mut sess);
    assert_eq!(rows, vec![vec![s("1")], vec![s("[1,2]")], vec![s("\"s\"")]]);
}
