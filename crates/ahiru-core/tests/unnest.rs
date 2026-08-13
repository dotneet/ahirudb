//! Integration tests for `UNNEST`.
//!
//! Expected values are decided by cross-checking against the actual output of the `duckdb` CLI
//! (`tests/data/list_varied.parquet`/`list1.parquet` are existing fixtures;
//! see `scripts/gen-testdata.sh`. Do not change how they are generated or their contents).
//!
//! - Both the SELECT-list and FROM-clause syntaxes (task patterns (a)/(b)).
//! - When the target column is a table's `Ty::Json` column itself, the elements are not
//!   restored to a native type (it cannot be safely decided without looking at the actual
//!   data; see the `narrow_unnest_elem_ty` doc in `plan::bind`), so expected values are
//!   compared as the raw JSON token byte sequence.
//! - For calls like `UNNEST(list_value(...))`/`UNNEST(json_array(...))`, where the target
//!   builds a list on the spot and every element is the same non-nested scalar type,
//!   verify with both the schema and the values that the result is restored to a native
//!   type (BIGINT/DOUBLE/VARCHAR/BOOLEAN).
//! - Resuming across a NeedIo does not change the result (feed a real Parquet file
//!   incrementally via `register_remote_as` + `provide`, and cross-check against feeding it
//!   all at once via `register_bytes`).

use ahiru_core::error::{code_of, Code};
use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::{Field, Ty, Value};

fn data(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// A session with `dual` (a dummy table with a single row) registered. This works around v1
/// excluding a bare `SELECT` of literals with no `FROM`
/// (same reason as `session_with_dual` in `recursive_cte.rs`).
fn session_with_dual() -> Session {
    let mut s = Session::new();
    s.register_bytes_as("dual", b"x\n1\n".to_vec(), FormatKind::Csv).unwrap();
    s
}

/// Runs a query to completion where all data is in memory.
/// Assumes `NeedIo`/`NeedCodec` never happen.
fn run(s: &mut Session, sql: &str) -> (Vec<Field>, Vec<Vec<Value>>) {
    let mut q = match s.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
    };
    let schema = q.schema.clone();
    let mut rows = Vec::new();
    loop {
        match s.step(&mut q).unwrap() {
            QueryStep::Batch(mut b) => {
                b.materialize();
                for r in 0..b.num_rows() {
                    rows.push(b.cols.iter().map(|c| c.value_at(r)).collect());
                }
            }
            QueryStep::Done => break,
            QueryStep::NeedIo(_) | QueryStep::NeedCodec(_) => {
                panic!("{sql}: unexpected NeedIo/NeedCodec")
            }
        }
    }
    (schema, rows)
}

fn i32v(v: i32) -> Value {
    Value::I32(v)
}
fn i64v(v: i64) -> Value {
    Value::I64(v)
}
fn s(v: &str) -> Value {
    Value::Bytes(v.as_bytes().to_vec())
}
/// Expected value for an element that comes back still as `Ty::Json`. Compared as the raw JSON
/// token byte sequence (for numbers, unquoted numeric text).
fn json_tok(v: &str) -> Value {
    Value::Bytes(v.as_bytes().to_vec())
}
const NULL: Value = Value::Null;

// --- SELECT list --------------------------------------------------------------

/// duckdb: `SELECT id, UNNEST(xs) AS x FROM 'list1.parquet' WHERE id < 2` ->
/// (0,1) (0,2) (0,3) (1,1) (1,2) (1,3) (every row is `[1,2,3]`).
/// Since the target is the table's JSON column itself, also verify that the value is not
/// restored and stays as a JSON token (`schema[1].ty == Ty::Json`).
#[test]
fn select_list_unnest_duplicates_other_columns_per_element() {
    let mut sess = Session::new();
    sess.register_bytes_as("t", data("list1.parquet"), FormatKind::Parquet).unwrap();
    let (schema, rows) = run(&mut sess, "SELECT id, UNNEST(xs) AS x FROM t WHERE id < 2");
    assert_eq!(schema[1].name, "x");
    assert_eq!(
        schema[1].ty,
        Ty::Json,
        "UNNEST of a table column itself does not restore a native type"
    );
    assert_eq!(
        rows,
        vec![
            vec![i32v(0), json_tok("1")],
            vec![i32v(0), json_tok("2")],
            vec![i32v(0), json_tok("3")],
            vec![i32v(1), json_tok("1")],
            vec![i32v(1), json_tok("2")],
            vec![i32v(1), json_tok("3")],
        ]
    );
}

/// duckdb: `list_varied.parquet` repeats NULL / empty array / arrays of 1-4 elements based on
/// `id % 5` (a rule already verified by
/// `nested_files.rs::list_varied_distinguishes_...`). NULL and empty-array rows produce 0
/// rows, and verify that the other column (`id`) is duplicated per row. Since the ordering
/// is just this engine reading a single Scan straight through from the top (no
/// parallelization or reordering at all), without an `ORDER BY` the result is exactly the
/// file's physical order.
#[test]
fn select_list_unnest_skips_null_and_empty_arrays() {
    let mut sess = Session::new();
    sess.register_bytes_as("t", data("list_varied.parquet"), FormatKind::Parquet).unwrap();
    let (_, rows) = run(&mut sess, "SELECT id, UNNEST(xs) AS x FROM t WHERE id < 10");
    let want: Vec<Vec<Value>> = vec![
        vec![i32v(2), json_tok("2")],
        vec![i32v(3), json_tok("3")],
        vec![i32v(3), NULL],
        vec![i32v(3), json_tok("6")],
        vec![i32v(4), json_tok("4")],
        vec![i32v(4), json_tok("5")],
        vec![i32v(4), json_tok("6")],
        vec![i32v(4), json_tok("7")],
        vec![i32v(7), json_tok("7")],
        vec![i32v(8), json_tok("8")],
        vec![i32v(8), NULL],
        vec![i32v(8), json_tok("16")],
        vec![i32v(9), json_tok("9")],
        vec![i32v(9), json_tok("10")],
        vec![i32v(9), json_tok("11")],
        vec![i32v(9), json_tok("12")],
    ];
    assert_eq!(rows, want);
}

/// Without an alias for UNNEST, the column name is "unnest", same as duckdb.
#[test]
fn select_list_unnest_default_column_name_is_unnest() {
    let mut db = session_with_dual();
    let (schema, _) = run(&mut db, "SELECT UNNEST(list_value(1,2,3)) FROM dual");
    assert_eq!(schema[0].name, "unnest");
}

#[test]
fn from_unnest_default_table_alias_is_unnest() {
    let mut db = session_with_dual();
    let (schema, rows) = run(&mut db, "SELECT unnest.* FROM dual, UNNEST(list_value(1, 2))");
    assert_eq!(schema[0].name, "unnest");
    assert_eq!(rows, vec![vec![i64v(1)], vec![i64v(2)]]);
}

// --- FROM clause (implicit LATERAL) --------------------------------------------

/// duckdb: `SELECT t.id, y.x FROM 'list_varied.parquet' t, UNNEST(t.xs) AS
/// y(x) WHERE t.id < 5` -> (2,2) (3,3) (3,NULL) (3,6) (4,4) (4,5) (4,6) (4,7).
#[test]
fn from_clause_unnest_is_implicit_lateral_cross_join() {
    let mut sess = Session::new();
    sess.register_bytes_as("t", data("list_varied.parquet"), FormatKind::Parquet).unwrap();
    let (schema, rows) =
        run(&mut sess, "SELECT t.id, y.x FROM t, UNNEST(t.xs) AS y(x) WHERE t.id < 5");
    assert_eq!(schema.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(), ["id", "x"]);
    assert_eq!(
        rows,
        vec![
            vec![i32v(2), json_tok("2")],
            vec![i32v(3), json_tok("3")],
            vec![i32v(3), NULL],
            vec![i32v(3), json_tok("6")],
            vec![i32v(4), json_tok("4")],
            vec![i32v(4), json_tok("5")],
            vec![i32v(4), json_tok("6")],
            vec![i32v(4), json_tok("7")],
        ]
    );
}

/// `SELECT *` lists `t`'s original columns (`id`, `xs`) followed by the expanded column
/// (same as duckdb, the target column itself is duplicated and kept).
#[test]
fn from_clause_unnest_star_keeps_source_array_column_too() {
    let mut sess = Session::new();
    sess.register_bytes_as("t", data("list1.parquet"), FormatKind::Parquet).unwrap();
    let (schema, rows) = run(&mut sess, "SELECT * FROM t, UNNEST(t.xs) AS y(x) WHERE t.id = 0");
    assert_eq!(schema.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(), ["id", "xs", "x"]);
    assert_eq!(
        rows,
        vec![
            vec![i32v(0), json_tok("[1,2,3]"), json_tok("1")],
            vec![i32v(0), json_tok("[1,2,3]"), json_tok("2")],
            vec![i32v(0), json_tok("[1,2,3]"), json_tok("3")],
        ]
    );
}

/// Without an alias, the column name is "unnest" same as duckdb, and other sibling columns
/// besides `t.tags` can be referenced normally too. Also verifies chaining multiple FROM-clause
/// UNNESTs (each independently multiplying rows -- the same cross product as DuckDB's LATERAL
/// chaining), while we're at it.
#[test]
fn chained_from_clause_unnests_cross_multiply_independently() {
    let mut db = session_with_dual();
    let (schema, rows) = run(
        &mut db,
        "SELECT a.v, b.v FROM dual, UNNEST(list_value(1,2)) AS a(v), UNNEST(list_value(10,20)) AS b(v)",
    );
    assert_eq!(schema[0].name, "v");
    assert_eq!(
        rows,
        vec![
            vec![i64v(1), i64v(10)],
            vec![i64v(1), i64v(20)],
            vec![i64v(2), i64v(10)],
            vec![i64v(2), i64v(20)],
        ]
    );
}

// --- Restoring to a native type -------------------------------------------------

/// duckdb: `SELECT UNNEST([1,2,3])` returns a BIGINT column. This engine has no array
/// literal syntax, so write it as `list_value(1,2,3)` (an alias for `json_array`). Since all
/// arguments are the same non-JSON scalar type, it can be determined to be BIGINT without
/// looking at the actual data (`plan::bind::narrow_unnest_elem_ty`).
#[test]
fn unnest_of_list_value_literal_restores_bigint() {
    let mut db = session_with_dual();
    let (schema, rows) = run(&mut db, "SELECT UNNEST(list_value(1,2,3)) AS x FROM dual");
    assert_eq!(schema[0].ty, Ty::BigInt);
    assert_eq!(rows, vec![vec![i64v(1)], vec![i64v(2)], vec![i64v(3)]]);
}

#[test]
fn unnest_of_list_value_literal_restores_varchar() {
    let mut db = session_with_dual();
    let (schema, rows) = run(&mut db, "SELECT UNNEST(list_value('a','b','c')) AS x FROM dual");
    assert_eq!(schema[0].ty, Ty::Varchar);
    assert_eq!(rows, vec![vec![s("a")], vec![s("b")], vec![s("c")]]);
}

#[test]
fn unnest_of_list_value_literal_restores_boolean() {
    let mut db = session_with_dual();
    let (schema, rows) = run(&mut db, "SELECT UNNEST(list_value(true,false)) AS x FROM dual");
    assert_eq!(schema[0].ty, Ty::Boolean);
    assert_eq!(rows, vec![vec![Value::Bool(true)], vec![Value::Bool(false)]]);
}

/// When the types don't match (integer and string mixed), it is not restored and stays as
/// `Ty::Json`.
#[test]
fn unnest_of_list_value_with_mixed_types_stays_json() {
    let mut db = session_with_dual();
    let (schema, rows) = run(&mut db, "SELECT UNNEST(list_value(1,'a')) AS x FROM dual");
    assert_eq!(schema[0].ty, Ty::Json);
    assert_eq!(rows, vec![vec![json_tok("1")], vec![json_tok("\"a\"")]]);
}

/// Also not restored when nested (an element that is itself an array).
#[test]
fn unnest_of_nested_list_value_stays_json() {
    let mut db = session_with_dual();
    let (schema, rows) =
        run(&mut db, "SELECT UNNEST(list_value(list_value(1,2), list_value(3))) AS x FROM dual");
    assert_eq!(schema[0].ty, Ty::Json);
    assert_eq!(rows, vec![vec![json_tok("[1,2]")], vec![json_tok("[3]")]]);
}

/// A CAST from a plain column reference (not a direct call to `json_array`/`list_value`) is
/// not restored, since the element type depends on the actual data.
#[test]
fn unnest_of_a_plain_json_cast_column_stays_json() {
    let mut db = session_with_dual();
    let (schema, rows) = run(&mut db, "SELECT UNNEST(CAST('[1,2,3]' AS JSON)) AS x FROM dual");
    assert_eq!(schema[0].ty, Ty::Json);
    assert_eq!(rows, vec![vec![json_tok("1")], vec![json_tok("2")], vec![json_tok("3")]]);
}

// --- NULL / empty array -----------------------------------------------------------

/// duckdb: `UNNEST(NULL)`/`UNNEST([])` both yield 0 rows.
#[test]
fn unnest_of_null_and_empty_array_yield_zero_rows() {
    let mut db = session_with_dual();
    let (_, rows) = run(&mut db, "SELECT UNNEST(CAST(NULL AS JSON)) AS x FROM dual");
    assert!(rows.is_empty());
    let (_, rows) = run(&mut db, "SELECT UNNEST(CAST('[]' AS JSON)) AS x FROM dual");
    assert!(rows.is_empty());
}

// --- Patterns explicitly rejected as unsupported -----------------------------------

/// Writing multiple `UNNEST`s in the same SELECT list (DuckDB does a complex per-column zip
/// behavior there) is rejected as out of scope.
#[test]
fn multiple_unnests_in_select_list_are_rejected() {
    let mut db = session_with_dual();
    let err = db.prepare("SELECT UNNEST(list_value(1,2)), UNNEST(list_value(3,4)) FROM dual", &[]);
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// Combining UNNEST with aggregation is rejected (the aggregation semantics over expanded rows
/// are not implemented).
#[test]
fn unnest_combined_with_aggregation_is_rejected() {
    let mut db = session_with_dual();
    let err = db.prepare("SELECT count(*), UNNEST(list_value(1,2)) FROM dual", &[]);
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// Writing `UNNEST` inside WHERE is unsupported (same as aggregation, the only position it can
/// appear in is the SELECT list).
#[test]
fn unnest_in_where_clause_is_rejected() {
    let mut db = session_with_dual();
    let err = db.prepare("SELECT 1 FROM dual WHERE UNNEST(list_value(1,2)) = 1", &[]);
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// A standalone FROM-clause `UNNEST` (with no preceding item) is rejected because implicit
/// LATERAL has no left neighbor.
#[test]
fn standalone_from_unnest_without_a_preceding_item_is_rejected() {
    let mut db = session_with_dual();
    let err = db.prepare("SELECT v FROM UNNEST(list_value(1,2,3)) AS x(v)", &[]);
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// A FROM-clause `UNNEST` on the left side of a JOIN is also rejected, since implicit LATERAL
/// only looks at its right neighbor.
#[test]
fn from_unnest_on_the_left_side_of_a_join_is_rejected() {
    let mut db = session_with_dual();
    let err = db.prepare("SELECT v FROM UNNEST(list_value(1,2,3)) AS x(v), dual", &[]);
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// If UNNEST's target is not `Ty::Json`, it is clearly rejected at bind time.
#[test]
fn unnest_of_a_non_json_expression_is_rejected() {
    let mut db = session_with_dual();
    let err = db.prepare("SELECT UNNEST(1) FROM dual", &[]);
    assert_eq!(code_of(err), Some(Code::TypeMismatch));
}

// --- Resuming across a NeedIo -----------------------------------------------------

/// Verify that feeding the byte stream incrementally across `NeedIo` produces exactly the same
/// result as feeding it all at once (`register_bytes`). `Unnest` itself just passes through its
/// input's `NeedIo` unchanged while preserving "the row/index currently being expanded" --
/// verification, against a real file, of this engine's most important invariant
/// (requirement 3).
#[test]
fn need_io_across_a_real_parquet_scan_does_not_change_the_result() {
    let sql = "SELECT id, UNNEST(xs) AS x FROM t";
    let bytes = data("list_varied.parquet");

    let mut eager = Session::new();
    eager.register_bytes_as("t", bytes.clone(), FormatKind::Parquet).unwrap();
    let (_, want) = run(&mut eager, sql);
    assert!(!want.is_empty());

    let (got, _) = run_with_lazy_io(&bytes, sql);
    assert_eq!(got, want, "the result must not change across a NeedIo");
}

/// Verify the same thing with the larger `list_pagetest.parquet` (2000 rows, an existing
/// fixture already used in `nested_files.rs`). For both of these files, the requested byte
/// range for footer resolution alone happens to cover the whole file (because they are small
/// test fixtures), so the scenario where another `NeedIo` gets inserted partway through
/// `step()` cannot be reproduced with a real file -- that scenario (where `Unnest` resumes
/// across multiple batches while preserving "the row/index currently being expanded") is
/// verified rigorously with a mock input operator in `exec::unnest::tests`
/// (`need_io_between_input_batches_does_not_change_the_result`/
/// `need_io_mid_row_does_not_change_the_result`).
#[test]
fn need_io_with_a_larger_file_does_not_change_the_result() {
    let sql = "SELECT id, UNNEST(xs) AS x FROM t WHERE id < 50";
    let bytes = data("list_pagetest.parquet");

    let mut eager = Session::new();
    eager.register_bytes_as("t", bytes.clone(), FormatKind::Parquet).unwrap();
    let (_, want) = run(&mut eager, sql);
    assert!(!want.is_empty());

    let (got, rounds) = run_with_lazy_io(&bytes, sql);
    assert_eq!(got, want, "the result must not change across a NeedIo");
    assert!(
        rounds >= 1,
        "register_remote_as starts at 0 bytes, so at least one NeedIo must always occur"
    );
}

/// Registers with `register_remote_as` and drives it by `provide`-ing exactly the range each
/// `NeedIo` requests. This mirrors how a host would actually perform range fetches
/// (the same "only hand over what was requested" savings path as the `ahiru-cli`/wasm hosts).
/// Returns `(result rows, number of suspend/resume round trips)`.
fn run_with_lazy_io(bytes: &[u8], sql: &str) -> (Vec<Vec<Value>>, u32) {
    let mut s = Session::new();
    s.register_remote_as("t", bytes.len() as u64, FormatKind::Parquet).unwrap();

    let mut rounds = 0u32;
    let mut q = loop {
        match s.prepare(sql, &[]).unwrap() {
            Prepared::Ready(q) => break q,
            Prepared::NeedIo(reqs) => {
                rounds += 1;
                assert!(rounds < 1000, "resolve_query never finished");
                for r in reqs {
                    let (start, end) = (r.offset as usize, (r.offset + r.len) as usize);
                    s.provide(r.table, r.part, r.offset, bytes[start..end].to_vec()).unwrap();
                }
            }
        }
    };

    let mut rows = Vec::new();
    let mut steps = 0u32;
    loop {
        steps += 1;
        assert!(steps < 10_000, "step never finished");
        match s.step(&mut q).unwrap() {
            QueryStep::Batch(mut b) => {
                b.materialize();
                for r in 0..b.num_rows() {
                    rows.push(b.cols.iter().map(|c| c.value_at(r)).collect());
                }
            }
            QueryStep::NeedIo(reqs) => {
                rounds += 1;
                for r in reqs {
                    let (start, end) = (r.offset as usize, (r.offset + r.len) as usize);
                    s.provide(r.table, r.part, r.offset, bytes[start..end].to_vec()).unwrap();
                }
            }
            QueryStep::NeedCodec(_) => panic!("test fixtures are uncompressed"),
            QueryStep::Done => break,
        }
    }
    (rows, rounds)
}
