//! Integration tests for `expr[i]` (subscript access) / `expr[i:j]` (slicing).
//!
//! The syntax and its desugaring target live in `sql::parser::Parser::subscript`
//! (`crates/ahiru-core/src/sql/parser.rs`); execution itself is
//! `crate::json::list_index` (the existing `list_extract`, unchanged here) and
//! `crate::json::list_slice` (new here, in `crates/ahiru-core/src/json.rs`).
//!
//! All expected values are decided by cross-checking against the actual output of the
//! `duckdb` CLI (each test's comment also records the command run).
//!
//! Out of scope: `[i]`/`[i:j]` on VARCHAR (substring extraction). In DuckDB,
//! `'hello'[1]` becomes `'h'`, but this implementation unifies LIST/MAP/JSON into a single
//! `Ty::Json` physical type (`docs/DESIGN.md` §8), and `[i]`/`[i:j]` are implemented only as
//! sugar for `list_extract`/`list_slice`, which are specific to that JSON representation.
//! VARCHAR is implicitly CAST to `Ty::Json` as `list_extract`/`list_slice`'s first argument,
//! but that CAST validates "is this byte string valid JSON"
//! (`expr::kernels::cast_str_to_json`), so `'hello'[1]` does not become NULL; it's a CAST
//! error (the actual behavior is recorded by
//! `varchar_subscript_is_a_cast_error_not_substring`).

use ahiru_core::error::{code_of, Code};
use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::{Field, Value};

fn data(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// A session with `basic.parquet` (1000 rows, `id` 0..999) registered as `t`.
/// A `SELECT <expr>` with no `FROM` is unsupported in v1 (`plan::bind`), so tests that just
/// want to evaluate an expression use `one()`, which goes through `LIMIT 1` (same workaround
/// as `json_functions.rs`).
fn session_with_basic() -> Session {
    let mut s = Session::new();
    s.register_bytes("t", data("basic.parquet")).unwrap();
    s
}

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

fn one(session: &mut Session, expr: &str) -> Value {
    let (_, rows) = run(session, &format!("SELECT {expr} AS x FROM t LIMIT 1"));
    rows[0][0].clone()
}

/// A helper that compares against the raw JSON token byte sequence (for numbers, unquoted
/// numeric text). `Ty::Json` values are not restored to a native type
/// (same reason as `unnest.rs::json_tok`).
fn j(v: &str) -> Value {
    Value::Bytes(v.as_bytes().to_vec())
}

fn i32v(v: i32) -> Value {
    Value::I32(v)
}

const NULL: Value = Value::Null;

// =========================================================================
// Basic subscript access (`[i]` -> `list_extract`)
// =========================================================================

#[test]
fn subscript_is_one_based_with_negative_from_end_and_null_out_of_range() {
    let mut sess = session_with_basic();
    // duckdb -c "select [1,2,3][1], [1,2,3][-1], [1,2,3][10], [1,2,3][0]"
    // -> 1 / 3 / NULL / NULL
    assert_eq!(one(&mut sess, "[1,2,3][1]"), j("1"));
    assert_eq!(one(&mut sess, "[1,2,3][-1]"), j("3"));
    assert_eq!(one(&mut sess, "[1,2,3][10]"), NULL);
    assert_eq!(one(&mut sess, "[1,2,3][0]"), NULL);
}

#[test]
fn subscript_index_can_be_an_arbitrary_expression() {
    let mut sess = session_with_basic();
    assert_eq!(one(&mut sess, "[10,20,30][1 + 1]"), j("20"));
}

#[test]
fn subscript_on_nested_lists_chains_left_to_right() {
    let mut sess = session_with_basic();
    // duckdb -c "select [[1,2],[3,4]][1], [[1,2],[3,4]][1][2]" -> [1, 2] / 2
    assert_eq!(one(&mut sess, "[[1,2],[3,4]][1]"), j("[1,2]"));
    assert_eq!(one(&mut sess, "[[1,2],[3,4]][1][2]"), j("2"));
}

#[test]
fn subscript_and_cast_interleave_by_written_order() {
    let mut sess = session_with_basic();
    // duckdb -c "select [1,2,3][1]::varchar" -> '1' (added on top of the
    // extracted element, not on the whole list)
    assert_eq!(one(&mut sess, "[1,2,3][1]::varchar"), Value::Bytes(b"1".to_vec()));
    // duckdb -c "select ([1,2,3]::json)[1]" -> 2 in real DuckDB, because
    // DuckDB's JSON subscript follows 0-based JSON-path semantics while its
    // LIST subscript is 1-based. This engine unifies LIST/MAP/JSON into one
    // physical representation and does not track that logical distinction
    // (docs/DESIGN.md §8), so `[i]` always means the 1-based list_extract
    // rule regardless of whether the value was cast to JSON first. This is a
    // deliberate, documented deviation from DuckDB for this one case.
    assert_eq!(one(&mut sess, "([1,2,3]::json)[1]"), j("1"));
}

// =========================================================================
// Slicing (`[i:j]` -> `list_slice`)
// =========================================================================

#[test]
fn slice_is_inclusive_on_both_ends() {
    let mut sess = session_with_basic();
    // duckdb -c "select [1,2,3,4,5][2:3]" -> [2, 3]
    assert_eq!(one(&mut sess, "[1,2,3,4,5][2:3]"), j("[2,3]"));
}

#[test]
fn slice_negative_bounds_count_from_the_end() {
    let mut sess = session_with_basic();
    // duckdb -c "select [1,2,3,4,5][-2:-1]" -> [4, 5]
    assert_eq!(one(&mut sess, "[1,2,3,4,5][-2:-1]"), j("[4,5]"));
    // duckdb -c "select [1,2,3,4,5][-1:-3]" -> [] (start after conversion (5)
    // is past end after conversion (3))
    assert_eq!(one(&mut sess, "[1,2,3,4,5][-1:-3]"), j("[]"));
}

#[test]
fn slice_out_of_range_clamps_to_empty_or_partial_instead_of_null() {
    let mut sess = session_with_basic();
    // Unlike `list_extract`, `list_slice` never returns NULL for an
    // out-of-range bound; it clamps (`duckdb -c "select [1,2,3,4,5][10:20],
    // [1,2,3,4,5][-10:3], [1,2,3,4,5][3:1]"` -> `[]` / `[1, 2, 3]` / `[]`).
    assert_eq!(one(&mut sess, "[1,2,3,4,5][10:20]"), j("[]"));
    assert_eq!(one(&mut sess, "[1,2,3,4,5][-10:3]"), j("[1,2,3]"));
    assert_eq!(one(&mut sess, "[1,2,3,4,5][3:1]"), j("[]"));
}

#[test]
fn slice_zero_bound_behaves_like_one() {
    let mut sess = session_with_basic();
    // duckdb -c "select [1,2,3,4,5][0:2]" -> [1, 2] (0 behaves like 1, unlike
    // list_extract's [0] which is NULL)
    assert_eq!(one(&mut sess, "[1,2,3,4,5][0:2]"), j("[1,2]"));
}

#[test]
fn slice_start_and_end_are_each_omittable() {
    let mut sess = session_with_basic();
    // duckdb -c "select [1,2,3,4,5][:3], [1,2,3,4,5][2:], [1,2,3,4,5][:]"
    // -> [1, 2, 3] / [2, 3, 4, 5] / [1, 2, 3, 4, 5]
    assert_eq!(one(&mut sess, "[1,2,3,4,5][:3]"), j("[1,2,3]"));
    assert_eq!(one(&mut sess, "[1,2,3,4,5][2:]"), j("[2,3,4,5]"));
    assert_eq!(one(&mut sess, "[1,2,3,4,5][:]"), j("[1,2,3,4,5]"));
}

#[test]
fn slice_of_empty_list_is_empty_not_null() {
    let mut sess = session_with_basic();
    // duckdb -c "select [][1:2]" -> []
    assert_eq!(one(&mut sess, "[][1:2]"), j("[]"));
}

#[test]
fn list_slice_function_propagates_null_bounds_unlike_omitted_ones() {
    let mut sess = session_with_basic();
    // duckdb -c "select list_slice([1,2,3,4,5], NULL, 3)" -> NULL. This is
    // exactly why the parser desugars an *omitted* bound to a literal
    // sentinel (1 / i64::MAX) rather than to SQL NULL — `[:3]` must stay
    // non-NULL (previous test), while an *explicit* NULL bound propagates.
    assert_eq!(one(&mut sess, "list_slice([1,2,3,4,5], NULL, 3)"), NULL);
    assert_eq!(one(&mut sess, "list_slice([1,2,3,4,5], 2, NULL)"), NULL);
    // `array_slice` is the DuckDB alias, also supported.
    assert_eq!(one(&mut sess, "array_slice([1,2,3,4,5], 2, 3)"), j("[2,3]"));
}

// =========================================================================
// Subscript/slice on a Parquet LIST-typed column
// =========================================================================

#[test]
fn subscript_on_a_parquet_list_column() {
    // duckdb: list1.parquet has 10 rows, xs = [1,2,3] on every row
    // (verified in nested_files.rs::list_of_int_matches_duckdb).
    let mut sess = Session::new();
    sess.register_bytes_as("t", data("list1.parquet"), FormatKind::Parquet).unwrap();
    let (_, rows) = run(&mut sess, "SELECT id, xs[1], xs[2:3] FROM t WHERE id < 2 ORDER BY id");
    assert_eq!(rows, vec![vec![i32v(0), j("1"), j("[2,3]")], vec![i32v(1), j("1"), j("[2,3]")]]);
}

#[test]
fn subscript_on_a_parquet_list_column_with_null_and_empty_rows() {
    // duckdb -c "select id, xs[1], xs[1:2] from 'list_varied.parquet' order
    // by id limit 5" ->
    //   0 NULL NULL / 1 NULL [] / 2 2 [2] / 3 3 [3,NULL] / 4 4 [4,5]
    // (list_varied.parquet's id%5 pattern is documented in
    // nested_files.rs::list_varied_distinguishes_null_empty_and_inner_nulls).
    let mut sess = Session::new();
    sess.register_bytes_as("t", data("list_varied.parquet"), FormatKind::Parquet).unwrap();
    let (_, rows) = run(&mut sess, "SELECT id, xs[1], xs[1:2] FROM t WHERE id < 5 ORDER BY id");
    assert_eq!(
        rows,
        vec![
            vec![i32v(0), NULL, NULL],    // list itself is SQL NULL
            vec![i32v(1), NULL, j("[]")], // empty list: [1] is NULL, [1:2] is []
            vec![i32v(2), j("2"), j("[2]")],
            vec![i32v(3), j("3"), j("[3,null]")],
            vec![i32v(4), j("4"), j("[4,5]")],
        ]
    );
}

#[test]
fn subscript_and_slice_on_a_nested_parquet_list_of_list_column() {
    // duckdb -c "select id, xss[1], xss[2], xss[3], xss[1][2], xss[1:2] from
    // 'list_of_list.parquet' order by id limit 3" (row 0 of xss is
    // [[0,1],[],[0]], per nested_files.rs::list_of_list_handles_nested_repetition):
    //   xss[1]=[0,1] xss[2]=[] xss[3]=[0] xss[1][2]=1 xss[1:2]=[[0,1],[]]
    let mut sess = Session::new();
    sess.register_bytes_as("t", data("list_of_list.parquet"), FormatKind::Parquet).unwrap();
    let (_, rows) =
        run(&mut sess, "SELECT xss[1], xss[2], xss[3], xss[1][2], xss[1:2] FROM t WHERE id = 0");
    assert_eq!(rows, vec![vec![j("[0,1]"), j("[]"), j("[0]"), j("1"), j("[[0,1],[]]")]]);
}

// =========================================================================
// Combined with JOIN / GROUP BY / aggregation
// =========================================================================

#[test]
fn subscript_in_a_join() {
    // duckdb -c "select a.id, b.name, a.xs[1] from 'list1.parquet' a join
    // 'basic.parquet' b on a.id=b.id where a.id < 3 order by a.id" ->
    //   0 name_0 1 / 1 name_1 1 / 2 name_2 1
    let mut sess = Session::new();
    sess.register_bytes_as("a", data("list1.parquet"), FormatKind::Parquet).unwrap();
    sess.register_bytes_as("b", data("basic.parquet"), FormatKind::Parquet).unwrap();
    let (_, rows) = run(
        &mut sess,
        "SELECT a.id, b.name, a.xs[1] FROM a JOIN b ON a.id = b.id WHERE a.id < 3 ORDER BY a.id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i32v(0), Value::Bytes(b"name_0".to_vec()), j("1")],
            vec![i32v(1), Value::Bytes(b"name_1".to_vec()), j("1")],
            vec![i32v(2), Value::Bytes(b"name_2".to_vec()), j("1")],
        ]
    );
}

#[test]
fn subscript_result_can_be_grouped_by() {
    // duckdb -c "select xs[1] g, count(*) from 'list1.parquet' group by 1"
    // -> single group (1, 10), since every row's xs is [1,2,3].
    let mut sess = Session::new();
    sess.register_bytes_as("t", data("list1.parquet"), FormatKind::Parquet).unwrap();
    let (_, rows) = run(&mut sess, "SELECT xs[1] AS g, COUNT(*) FROM t GROUP BY 1");
    assert_eq!(rows, vec![vec![j("1"), Value::I64(10)]]);
}

#[test]
fn slice_applied_to_array_agg_output() {
    // duckdb -c "select list_slice(array_agg(id order by id), 1, 3) from
    // 'basic.parquet'" -> [0, 1, 2]. This engine's array_agg has no ORDER BY
    // support inside the call, but a plain sequential Scan over a single
    // RowGroup with a WHERE filter is deterministic (no sort/parallelism is
    // ever introduced), so restricting to id < 3 makes the aggregation
    // order predictable without relying on ORDER BY inside array_agg.
    let mut sess = session_with_basic();
    let (_, rows) = run(&mut sess, "SELECT array_agg(id)[1:3] FROM t WHERE id < 3");
    // `array_agg`'s own JSON serialization uses ", " (space after comma,
    // see `exec::agg`), unlike the parser's array-literal/Parquet-LIST JSON
    // text used elsewhere in this file which has no spaces. `list_slice`
    // just copies the underlying bytes verbatim (see its doc comment in
    // `crate::json`), so the slice inherits whichever spacing its source
    // document used.
    assert_eq!(rows, vec![vec![j("[0, 1, 2]")]]);
}

// =========================================================================
// Errors / out of scope
// =========================================================================

#[test]
fn list_slice_requires_exactly_three_arguments() {
    let mut sess = session_with_basic();
    // duckdb -c "select list_slice([1,2,3], 1)" is also rejected (Binder
    // Error: no 2-arg overload). This engine's list_slice has a single fixed
    // 3-argument signature (crate::expr::funcs::resolve), so a 2-argument
    // call is WrongArgCount.
    assert_eq!(
        code_of(sess.prepare("SELECT list_slice([1,2,3], 1) FROM t LIMIT 1", &[]).map(|_| ())),
        Some(Code::WrongArgCount)
    );
}

#[test]
fn varchar_subscript_is_a_cast_error_not_substring() {
    // Deliberately out of scope (see module doc): `[i]`/`[i:j]` only desugar
    // to list_extract/list_slice, whose first argument is Ty::Json. A bare
    // VARCHAR implicitly casts to JSON, and that cast validates the bytes
    // are actually JSON (`expr::kernels::cast_str_to_json`) — 'hello' is not
    // valid JSON, so this is a CAST error at runtime, not silently NULL and
    // not DuckDB's character-indexing behavior (`duckdb -c "select
    // 'hello'[1]"` -> 'h').
    let mut sess = session_with_basic();
    assert_eq!(
        code_of(sess.prepare("SELECT 'hello'[1] FROM t LIMIT 1", &[]).map(|_| ())),
        None,
        "expected this to bind fine (the CAST error, if any, is a runtime row failure)"
    );
    let mut q = match sess.prepare("SELECT 'hello'[1] AS x FROM t LIMIT 1", &[]).unwrap() {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("unexpected NeedIo"),
    };
    let err = loop {
        match sess.step(&mut q) {
            Ok(QueryStep::Batch(_)) => continue,
            Ok(QueryStep::Done) => panic!("expected a runtime error, got a result"),
            Ok(QueryStep::NeedIo(_)) | Ok(QueryStep::NeedCodec(_)) => panic!("unexpected NeedIo"),
            Err(e) => break e,
        }
    };
    assert_eq!(err.code, Code::InvalidCast);
}
