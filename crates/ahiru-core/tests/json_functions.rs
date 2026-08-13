//! Integration tests for JSON path operators, construction functions, and LIST/MAP accessors.
//!
//! Expected values are decided by cross-checking against the actual output of `duckdb -c "SELECT ..."`.
//! For supported scope and known limitations, see the comments on `ahiru_core::json`
//! (internal-only, so see the doc at the top of `crates/ahiru-core/src/json.rs`) and `expr::funcs`.
//!
//! A `SELECT <expr>` with no `FROM` is unsupported in v1 (`plan::bind`), so we use
//! `tests/data/basic.parquet` with `LIMIT 1` to get exactly one row.

use ahiru_core::error::{code_of, Code};
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn data(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn session_with_basic() -> Session {
    let mut s = Session::new();
    s.register_bytes("t", data("basic.parquet")).unwrap();
    s
}

fn run(session: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    let mut q = match session.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
    };
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
            QueryStep::NeedIo(_) | QueryStep::NeedCodec(_) => {
                panic!("{sql}: unexpected NeedIo/NeedCodec")
            }
        }
    }
    rows
}

fn one(session: &mut Session, expr: &str) -> Value {
    let rows = run(session, &format!("SELECT {expr} AS x FROM t LIMIT 1"));
    rows[0][0].clone()
}

fn s(v: &str) -> Value {
    Value::Bytes(v.as_bytes().to_vec())
}

// --- json_extract / json_extract_string / -> / ->> --------------------------

#[test]
fn json_extract_matches_duckdb_paths() {
    let mut sess = session_with_basic();
    // duckdb: json_extract('{"a":{"b":[1,2,3]}}', '$.a.b[1]') -> 2
    assert_eq!(one(&mut sess, r#"json_extract('{"a":{"b":[1,2,3]}}', '$.a.b[1]')"#), s("2"));
    // duckdb: json_extract('[1,2,3]', '$[-1]') -> 3
    assert_eq!(one(&mut sess, r#"json_extract('[1,2,3]', '$[-1]')"#), s("3"));
    // A path that isn't found is SQL NULL.
    assert_eq!(one(&mut sess, r#"json_extract('{"a":1}', '$.b')"#), Value::Null);
}

#[test]
fn arrow_operators_are_sugar_for_json_extract_functions() {
    let mut sess = session_with_basic();
    // A VARCHAR literal is implicitly Cast to json_extract's first argument (expects
    // Ty::Json). Works without an explicit CAST(... AS JSON).
    // duckdb: '{"a":1}'::JSON -> '$.a' -> 1 (json), ->> '$.a' -> '1' (varchar)
    assert_eq!(one(&mut sess, r#"'{"a":1}' -> '$.a'"#), s("1"));
    assert_eq!(one(&mut sess, r#"'{"a":1}' ->> '$.a'"#), s("1"));
    // Binds tighter than comparison operators, so no parentheses are needed.
    assert_eq!(one(&mut sess, r#"'{"a":1}' ->> '$.a' = '1'"#), Value::Bool(true));
}

#[test]
fn json_extract_string_unquotes_strings_and_nulls_json_null() {
    let mut sess = session_with_basic();
    assert_eq!(one(&mut sess, r#"json_extract_string('{"a":"hi"}', '$.a')"#), s("hi"));
    assert_eq!(one(&mut sess, r#"json_extract_string('{"a":null}', '$.a')"#), Value::Null);
}

// --- json_type / json_array_length ------------------------------------------

#[test]
fn json_type_matches_duckdb_type_names() {
    let mut sess = session_with_basic();
    for (doc, want) in [
        (r#"{"a":1}"#, "OBJECT"),
        ("[1,2]", "ARRAY"),
        ("\"x\"", "VARCHAR"),
        ("true", "BOOLEAN"),
        ("null", "NULL"),
        ("1.5", "DOUBLE"),
    ] {
        assert_eq!(one(&mut sess, &format!("json_type('{doc}')")), s(want), "doc={doc}");
    }
}

#[test]
fn json_array_length_matches_duckdb() {
    let mut sess = session_with_basic();
    assert_eq!(one(&mut sess, "json_array_length('[1,2,3]')"), Value::I64(3));
    // A non-array is 0 (duckdb: json_array_length('{"a":1}') -> 0).
    assert_eq!(one(&mut sess, r#"json_array_length('{"a":1}')"#), Value::I64(0));
}

// --- Construction functions: to_json / json_object / json_array / list_value --------------

#[test]
fn to_json_covers_scalar_types_like_duckdb() {
    let mut sess = session_with_basic();
    assert_eq!(one(&mut sess, "to_json(1)"), s("1"));
    assert_eq!(one(&mut sess, "to_json(1.5)"), s("1.5"));
    assert_eq!(one(&mut sess, "to_json('hello')"), s("\"hello\""));
    assert_eq!(one(&mut sess, "to_json(true)"), s("true"));
    assert_eq!(one(&mut sess, "to_json(CAST('2024-01-01' AS DATE))"), s("\"2024-01-01\""));
    // duckdb: to_json(NULL) -> SQL NULL.
    assert_eq!(one(&mut sess, "to_json(NULL)"), Value::Null);
}

#[test]
fn json_object_and_json_array_match_duckdb_construction() {
    let mut sess = session_with_basic();
    // duckdb: json_object('a', 1, 'b', 'x') -> {"a":1,"b":"x"}
    assert_eq!(one(&mut sess, "json_object('a', 1, 'b', 'x')"), s(r#"{"a":1,"b":"x"}"#));
    // duckdb: json_array(1, 'x', true, NULL) -> [1,"x",true,null]
    assert_eq!(one(&mut sess, "json_array(1, 'x', true, NULL)"), s(r#"[1,"x",true,null]"#));
    // list_value is an alias for json_array.
    assert_eq!(one(&mut sess, "list_value(1, 2, 3)"), s("[1,2,3]"));
}

#[test]
fn json_functions_work_over_table_columns_not_just_literals() {
    let mut sess = session_with_basic();
    // `id` is a BigInt column. Verify to_json/json_object also work on a column value, not just a literal.
    let rows = run(
        &mut sess,
        "SELECT json_object('id', id, 'flag', flag) FROM t WHERE id IN (0, 1) ORDER BY id",
    );
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0][0], s(r#"{"id":0,"flag":true}"#));
    assert_eq!(rows[1][0], s(r#"{"id":1,"flag":false}"#));
}

// --- list_extract / map_extract ----------------------------------------------

#[test]
fn list_extract_is_one_based_like_duckdb() {
    let mut sess = session_with_basic();
    // The 1st argument expects Ty::Json, so a VARCHAR literal is implicitly Cast.
    // duckdb: list_extract([10,20,30], 1) -> 10, list_extract(.., -1) -> 30
    assert_eq!(one(&mut sess, "list_extract('[10,20,30]', 1)"), s("10"));
    assert_eq!(one(&mut sess, "list_extract('[10,20,30]', -1)"), s("30"));
    assert_eq!(one(&mut sess, "list_extract('[10,20,30]', 0)"), Value::Null);
}

#[test]
fn map_extract_looks_up_object_keys() {
    let mut sess = session_with_basic();
    assert_eq!(one(&mut sess, r#"map_extract('{"a":1,"b":2}', 'a')"#), s("1"));
    assert_eq!(one(&mut sess, r#"map_extract('{"a":1}', 'z')"#), Value::Null);
}

#[test]
fn map_extract_reads_parquet_map_pair_arrays() {
    // Parquet MAP is stored as `[{"key":...,"value":...}, ...]`.
    // map_basic row id=1: a=1, b=2, c=null.
    let mut sess = Session::new();
    sess.register_bytes_as("t", data("map_basic.parquet"), ahiru_core::format::FormatKind::Parquet)
        .unwrap();
    let rows = run(
        &mut sess,
        "SELECT map_extract(m, 'a'), map_extract(m, 'b'), map_extract(m, 'z') FROM t WHERE id = 1",
    );
    assert_eq!(rows, vec![vec![s("1"), s("2"), Value::Null]]);

    let mut sess = Session::new();
    sess.register_bytes_as(
        "t",
        data("map_int_key.parquet"),
        ahiru_core::format::FormatKind::Parquet,
    )
    .unwrap();
    let rows = run(&mut sess, "SELECT map_extract(m, '1') FROM t WHERE id = 1");
    assert_eq!(rows, vec![vec![s("\"v1\"")]]);
}

// --- list_concat / `||` on lists ---------------------------------------------

#[test]
fn list_concat_matches_duckdb() {
    let mut sess = session_with_basic();
    // duckdb: list_concat([1,2],[3]) -> [1, 2, 3]; list_concat([1]) -> [1];
    //         list_concat([],[1]) -> [1]
    assert_eq!(one(&mut sess, "list_concat([1,2], [3])"), s("[1,2,3]"));
    assert_eq!(one(&mut sess, "list_concat([1])"), s("[1]"));
    assert_eq!(one(&mut sess, "list_concat([], [1])"), s("[1]"));
    // duckdb: list_cat/array_concat/array_cat are aliases of list_concat.
    assert_eq!(one(&mut sess, "list_cat([1], [2])"), s("[1,2]"));
    assert_eq!(one(&mut sess, "array_concat([1], [2])"), s("[1,2]"));
    assert_eq!(one(&mut sess, "array_cat([1], [2])"), s("[1,2]"));
    // duckdb: list_concat([1], NULL::INTEGER[]) -> [1];
    //         list_concat(NULL::INTEGER[], NULL::INTEGER[]) -> [].
    // The *function* reads NULL as an empty list and never returns NULL —
    // unlike the `||` *operator*, see `list_concat_operator_matches_duckdb`.
    assert_eq!(one(&mut sess, "list_concat([1], NULL)"), s("[1]"));
    assert_eq!(one(&mut sess, "list_concat(NULL, NULL)"), s("[]"));
    assert_eq!(one(&mut sess, "list_concat([1,2], [3], NULL, [4])"), s("[1,2,3,4]"));
}

#[test]
fn list_concat_operator_matches_duckdb() {
    let mut sess = session_with_basic();
    // The bug this replaced: `||` used to cast both sides to VARCHAR, so
    // `[1,2] || [3]` silently returned the string `[1,2][3]`.
    // duckdb: [1,2] || [3] -> [1, 2, 3] (INTEGER[])
    assert_eq!(one(&mut sess, "[1,2] || [3]"), s("[1,2,3]"));
    // duckdb: [] || [1] -> [1]; [[1]] || [[3]] -> [[1], [3]] (no flattening)
    assert_eq!(one(&mut sess, "[] || [1]"), s("[1]"));
    assert_eq!(one(&mut sess, "[1] || []"), s("[1]"));
    assert_eq!(one(&mut sess, "[[1]] || [[3]]"), s("[[1],[3]]"));
    // Left-associative chaining stays a list all the way through.
    assert_eq!(one(&mut sess, "[1,2] || [3] || [4]"), s("[1,2,3,4]"));
    // duckdb: [1] || NULL::INTEGER[] -> NULL, NULL || [1] -> NULL.
    // The operator propagates NULL (the function does not).
    assert_eq!(one(&mut sess, "[1] || NULL"), Value::Null);
    assert_eq!(one(&mut sess, "NULL || [1]"), Value::Null);
    // An explicitly JSON-typed operand behaves the same as a list literal —
    // in this engine they are the same type (`docs/DESIGN.md` §5/§8).
    assert_eq!(one(&mut sess, "CAST('[1,2]' AS JSON) || CAST('[3]' AS JSON)"), s("[1,2,3]"));
}

#[test]
fn concat_operator_keeps_varchar_behavior_when_not_both_json() {
    let mut sess = session_with_basic();
    // duckdb: 'a' || 'b' -> 'ab', 'a' || 1 -> 'a1', 'a' || NULL -> NULL
    assert_eq!(one(&mut sess, "'a' || 'b'"), s("ab"));
    assert_eq!(one(&mut sess, "'a' || 1"), s("a1"));
    assert_eq!(one(&mut sess, "'a' || NULL"), Value::Null);
    assert_eq!(one(&mut sess, "NULL || NULL"), Value::Null);
    // JSON on one side only stays text concatenation. DuckDB rejects
    // `[1] || 2` outright ("Cannot concatenate types INTEGER[] and
    // INTEGER"), but it can tell a LIST from a JSON document and this engine
    // cannot, and `json_col || 'suffix'` is legal DuckDB. Documented in
    // `docs/sql/limitations.md`.
    assert_eq!(one(&mut sess, "[1] || 2"), s("[1]2"));
    assert_eq!(one(&mut sess, "1 || [2]"), s("1[2]"));
    // Casting out of JSON is the documented escape hatch for concatenating
    // JSON documents as text (which is what DuckDB's `JSON || JSON` does).
    assert_eq!(one(&mut sess, "CAST([1,2] AS VARCHAR) || CAST([3] AS VARCHAR)"), s("[1,2][3]"));
}

/// Run `SELECT <expr> FROM t LIMIT 1` and return the error code it fails
/// with, or `None` if it succeeds. The failures this is used for are
/// *runtime* ones — whether a JSON value is an array can't be known until the
/// value is read — so they surface from `Session::step`, not `prepare`, the
/// same way `cast_invalid_json_text_errors_the_whole_query` does.
fn err_code(session: &mut Session, expr: &str) -> Option<Code> {
    let sql = format!("SELECT {expr} AS x FROM t LIMIT 1");
    let mut q = match session.prepare(&sql, &[]) {
        Ok(Prepared::Ready(q)) => q,
        Ok(Prepared::NeedIo(_)) => panic!("{sql}: unexpected NeedIo"),
        Err(e) => return Some(e.code),
    };
    loop {
        match session.step(&mut q) {
            Ok(QueryStep::Batch(_)) => continue,
            Ok(QueryStep::Done) => return None,
            Ok(QueryStep::NeedIo(_)) | Ok(QueryStep::NeedCodec(_)) => panic!("{sql}: unexpected"),
            Err(e) => return Some(e.code),
        }
    }
}

#[test]
fn concat_operator_on_non_array_json_is_a_type_error() {
    let mut sess = session_with_basic();
    // Deliberate divergence from DuckDB, where JSON is a distinct type from
    // LIST and `'{"a":1}'::JSON || '{"b":2}'::JSON` is VARCHAR text
    // concatenation (`{"a":1}{"b":2}`). Here a list *is* a JSON value, so the
    // two cases cannot be told apart, and `||` raises rather than guessing —
    // returning NULL would just be a second silent wrong answer. See
    // `docs/sql/limitations.md`.
    let obj = r#"CAST('{"a":1}' AS JSON)"#;
    assert_eq!(
        err_code(&mut sess, &format!(r#"{obj} || CAST('{{"b":2}}' AS JSON)"#)),
        Some(Code::TypeMismatch)
    );
    // Mixed array/non-array, in both orders. DuckDB rejects these too, at
    // bind time: `duckdb -c "select [1] || '{\"a\":1}'::JSON"` -> "Cannot
    // concatenate types INTEGER[] and JSON - an explicit cast is required".
    assert_eq!(err_code(&mut sess, &format!("[1] || {obj}")), Some(Code::TypeMismatch));
    assert_eq!(err_code(&mut sess, &format!("{obj} || [1]")), Some(Code::TypeMismatch));
    // A JSON scalar is not an array either.
    assert_eq!(err_code(&mut sess, "CAST('5' AS JSON) || [1]"), Some(Code::TypeMismatch));
    // The error takes priority over NULL propagation, in either order, so the
    // result never depends on which operand is inspected first.
    assert_eq!(err_code(&mut sess, &format!("{obj} || NULL")), Some(Code::TypeMismatch));
    assert_eq!(err_code(&mut sess, &format!("NULL || {obj}")), Some(Code::TypeMismatch));
    // The escape hatch stays available and is the only way to get DuckDB's
    // text concatenation of two JSON documents.
    assert_eq!(
        one(
            &mut sess,
            &format!(r#"CAST({obj} AS VARCHAR) || CAST(CAST('{{"b":2}}' AS JSON) AS VARCHAR)"#)
        ),
        s(r#"{"a":1}{"b":2}"#)
    );
}

#[test]
fn list_concat_function_on_non_array_json_stays_null() {
    let mut sess = session_with_basic();
    // Unlike the operator above, the *function* keeps the leniency the rest
    // of the `list_*` family has for non-array JSON. DuckDB's own
    // function-vs-operator split (NULL as an empty list vs NULL propagation)
    // already treats them as different things.
    assert_eq!(one(&mut sess, r#"list_concat(CAST('{"a":1}' AS JSON), [1])"#), Value::Null);
    assert_eq!(one(&mut sess, r#"array_concat([1], CAST('5' AS JSON))"#), Value::Null);
}

#[test]
fn list_concat_works_over_table_columns() {
    let mut sess = session_with_basic();
    // Per-row evaluation, not just constant folding: one operand varies per
    // row and one row's list is NULL (so the operator's row is NULL too).
    let rows = run(
        &mut sess,
        "SELECT [id] || CASE WHEN id = 0 THEN NULL ELSE [99] END \
         FROM t WHERE id IN (0, 1) ORDER BY id",
    );
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0][0], Value::Null);
    assert_eq!(rows[1][0], s("[1,99]"));
}

// --- CAST --------------------------------------------------------------------

#[test]
fn cast_varchar_to_json_round_trips_and_try_cast_is_lenient() {
    let mut sess = session_with_basic();
    assert_eq!(one(&mut sess, r#"CAST('{"a":1}' AS JSON)"#), s(r#"{"a":1}"#));
    assert_eq!(one(&mut sess, r#"CAST(CAST('{"a":1}' AS JSON) AS VARCHAR)"#), s(r#"{"a":1}"#));
    // duckdb: TRY_CAST('not json' AS JSON) -> NULL.
    assert_eq!(one(&mut sess, "TRY_CAST('not json' AS JSON)"), Value::Null);
}

#[test]
fn cast_invalid_json_text_errors_the_whole_query() {
    let mut sess = session_with_basic();
    // duckdb: CAST('not json' AS JSON) -> Conversion Error (not rounded down to NULL).
    // The `name` column holds non-JSON text like "name_0", so CAST(name AS JSON)
    // fails at runtime (`Session::step`) with InvalidCast. CAST itself cannot detect the
    // invalidity from type checking alone (it's not known until the value is read), so
    // note that this does not error at `prepare` time.
    let mut q = match sess.prepare("SELECT CAST(name AS JSON) FROM t", &[]).unwrap() {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("unexpected NeedIo"),
    };
    let mut saw_error = false;
    loop {
        match sess.step(&mut q) {
            Ok(QueryStep::Batch(_)) => continue,
            Ok(QueryStep::Done) => break,
            Ok(QueryStep::NeedIo(_)) | Ok(QueryStep::NeedCodec(_)) => panic!("unexpected"),
            Err(e) => {
                assert_eq!(e.code, Code::InvalidCast);
                saw_error = true;
                break;
            }
        }
    }
    assert!(saw_error, "CAST(name AS JSON) should error on invalid JSON");
}

// --- Comparison ---------------------------------------------------------------------

#[test]
fn json_equality_is_byte_comparison_ordering_is_rejected() {
    let mut sess = session_with_basic();
    // Differences in key order or whitespace cause a mismatch, since it's a byte-sequence
    // comparison (a known limitation of v1). JSON doesn't `Ty::unify` with any other type
    // (see the module doc), so both comparison sides are explicitly CAST to Ty::Json.
    assert_eq!(
        one(&mut sess, r#"CAST('{"a":1,"b":2}' AS JSON) = CAST('{"a":1,"b":2}' AS JSON)"#),
        Value::Bool(true)
    );
    assert_eq!(
        one(&mut sess, r#"CAST('{"a": 1}' AS JSON) = CAST('{"a":1}' AS JSON)"#),
        Value::Bool(false)
    );
    // Ordering comparisons are TypeMismatch.
    assert_eq!(
        code_of(
            sess.prepare("SELECT CAST('1' AS JSON) < CAST('2' AS JSON) FROM t", &[]).map(|_| ())
        ),
        Some(Code::TypeMismatch)
    );
}
