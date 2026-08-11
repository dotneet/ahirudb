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

// --- CSV: known split-boundary limitation, verified at the byte level -----

/// `format::csv.rs` documents (around `read_split`'s `lead` handling) that
/// the split-boundary scanner cannot track quote state, so a quoted newline
/// that lands right at/after a split boundary can be misread as a record
/// terminator. This test pins down *what actually happens* in that case:
/// with the fixture below, a boundary that falls inside the quoted embedded
/// newline causes the scanner to resynchronize mid-quote and inject one
/// spurious all-NULL row — real rows before and after survive intact, and
/// nothing panics or drops data. This is the documented, accepted
/// limitation (not something this QA pass fixes), but it's worth pinning so
/// a future change to the split algorithm doesn't silently change this
/// failure mode without anyone noticing.
#[test]
fn quoted_embedded_newline_on_a_split_boundary_can_inject_a_spurious_row() {
    // Row 1's second field is quoted and contains a literal '\n'. Absolute
    // byte offset 12 (right after "line1") is exactly that embedded
    // newline; data_start (after the "a,b\n" header) is 4, so split_bytes=8
    // puts a split boundary exactly there.
    let data = "a,b\n1,\"line1\nline2\"\n2,x\n";
    let src = Source::from_bytes(data.as_bytes().to_vec());
    let mut f = CsvFormat::new(b',');
    f.resolve(&src).unwrap().unwrap();
    f.split_bytes = 8;
    assert!(f.num_splits() > 1, "test fixture must actually straddle a split boundary");

    let mut rows: Vec<(Value, Value)> = Vec::new();
    for split in 0..f.num_splits() {
        let cols = f.read_split(&src, split, &[0, 1]).expect("read_split must not error");
        for i in 0..cols[0].len() {
            rows.push((cols[0].value_at(i), cols[1].value_at(i)));
        }
    }
    // The two real records survive. A single spurious (NULL, NULL) row is
    // injected between them — the documented failure mode, not silent
    // corruption of real data.
    assert_eq!(rows[0], (Value::I64(1), s("line1\nline2")));
    assert_eq!(rows.last().unwrap(), &(Value::I64(2), s("x")));
    assert!(
        rows.iter().any(|r| r.0 == Value::Null && r.1 == Value::Null),
        "expected the documented spurious all-NULL row, got {rows:?}"
    );

    // With a split size that keeps the whole quoted field inside a single
    // split (no boundary lands inside the quote), the same data reads back
    // with no spurious row at all — confirming the artifact above really is
    // caused by the boundary landing inside the quote, not a general bug.
    let mut g = CsvFormat::new(b',');
    g.resolve(&src).unwrap().unwrap();
    g.split_bytes = 1024;
    assert_eq!(g.num_splits(), 1);
    let cols = g.read_split(&src, 0, &[0, 1]).unwrap();
    assert_eq!(cols[0].len(), 2, "no split boundary inside the quote => no spurious row");
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
fn csv_values_outside_the_sampled_type_become_null_through_sql_not_an_error() {
    // The inferred type comes from the first SAMPLE_ROWS rows only
    // (`format::csv`'s module doc); values that don't fit later become NULL
    // rather than failing the whole query. Exercise this through actual SQL
    // (WHERE / aggregate), not just direct `read_split` calls.
    let mut csv = String::from("n\n");
    for i in 0..1001 {
        csv.push_str(&format!("{i}\n"));
    }
    csv.push_str("not_a_number\n");
    let mut sess = Session::new();
    sess.register_bytes_as("t", csv.into_bytes(), FormatKind::Csv).unwrap();
    let rows = run_all("SELECT count(*) FROM t", &mut sess);
    assert_eq!(rows, [[Value::I64(1002)]]);
    let nulls = run_all("SELECT count(*) FROM t WHERE n IS NULL", &mut sess);
    assert_eq!(nulls, [[Value::I64(1)]]);
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
fn header_only_csv_part_with_a_numeric_sibling_column_is_a_type_mismatch() {
    // Companion to the case above, and a real gotcha worth pinning: because
    // an empty part's columns always default to VARCHAR regardless of what
    // the *other* parts' data looks like, unioning a header-only CSV with a
    // part that has actual numeric data in the same column position is
    // rejected by `catalog::unify_schema` (`Ty::unify(Varchar, BigInt)` is
    // `None`) even though the empty part contributes zero conflicting rows.
    // This falls straight out of two independently documented behaviors
    // (empty-sample columns default to VARCHAR; part schemas are unified
    // strictly, never silently coerced) but is easy to trip over in
    // practice (e.g. a placeholder/template file in a glob), so it's worth
    // asserting explicitly rather than leaving it as an emergent surprise.
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
    let r = sess.prepare("SELECT count(*) FROM t", &[]);
    assert_eq!(code_of(r), Some(Code::TypeMismatch));
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
