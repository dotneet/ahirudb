//! Integration tests for lambda expressions (`x -> expr` / `(a, b) -> expr`) and
//! `list_transform`/`list_filter`/`list_reduce`.
//!
//! Expected values are decided by cross-checking against the actual output of
//! `duckdb -c "SELECT ..."`.
//! However, this engine implements LIST as a "dynamically typed JSON value"
//! (`Ty::Json`; see the doc on `crates/ahiru-core/src/vector/types.rs`), so unlike duckdb,
//! list elements do not have a native numeric type.
//! To perform arithmetic/comparison on a parameter inside a lambda body, an explicit
//! conversion through VARCHAR like `CAST(CAST(x AS VARCHAR) AS INTEGER)` is required, the
//! same as the existing restriction on `json_extract`/`list_extract` results (this is not a
//! constraint specific to lambdas).
//! A lambda body can only reference its own parameters and cannot reference columns from the
//! outer SQL scope (a known limitation; see the doc on
//! `plan::compile::Compiler::lambda_call`).
//!
//! A `SELECT <expr>` with no `FROM` is unsupported in v1 (`plan::bind`), so we use
//! `tests/data/basic.parquet` with `LIMIT 1` to get exactly one row
//! (same convention as `json_functions.rs`).

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

/// The common idiom in this engine for doing arithmetic/comparison on JSON array elements.
/// Since `Ty::Json` doesn't `Ty::unify` with any other type, first go through VARCHAR before
/// converting to a number.
fn int_cast(x: &str) -> String {
    format!("CAST(CAST({x} AS VARCHAR) AS INTEGER)")
}

// --- Syntax: `x -> expr` / `(a, b) -> expr` ----------------------------------

#[test]
fn lambda_single_param_needs_no_parens() {
    let mut sess = session_with_basic();
    // duckdb: list_transform([1,2,3], x -> x + 1) -> [2,3,4]
    let e = format!("list_transform(json_array(1,2,3), x -> {} + 1)", int_cast("x"));
    assert_eq!(one(&mut sess, &e), s("[2,3,4]"));
}

#[test]
fn lambda_multi_param_needs_parens() {
    let mut sess = session_with_basic();
    // duckdb: list_reduce([1,2,3,4], (acc, x) -> acc + x) -> 10
    let e = format!(
        "list_reduce(json_array(1,2,3,4), (acc, x) -> {} + {})",
        int_cast("acc"),
        int_cast("x")
    );
    assert_eq!(one(&mut sess, &e), s("10"));
}

#[test]
fn arrow_outside_a_lambda_taking_function_stays_the_json_path_operator() {
    let mut sess = session_with_basic();
    // Since `coalesce` does not accept a lambda, `->` stays the JSON path operator as usual
    // even in an argument position (confirmed with the duckdb CLI:
    // `coalesce(doc -> 'a', 'x')` is not interpreted as a lambda and is resolved as JSON extraction).
    assert_eq!(one(&mut sess, r#"coalesce('{"a":1}' -> '$.a', to_json('x'))"#), s("1"));
}

// --- list_transform -----------------------------------------------------------

#[test]
fn list_transform_maps_each_element() {
    let mut sess = session_with_basic();
    // duckdb: list_transform([1,2,3], x -> x + 1) -> [2,3,4]
    let e = format!("list_transform(json_array(1,2,3), x -> {} + 1)", int_cast("x"));
    assert_eq!(one(&mut sess, &e), s("[2,3,4]"));
}

#[test]
fn list_transform_identity_needs_no_cast() {
    let mut sess = session_with_basic();
    assert_eq!(one(&mut sess, "list_transform(json_array(1,2,3), x -> x)"), s("[1,2,3]"));
    // String elements come back as the JSON text as-is (with quotes).
    assert_eq!(one(&mut sess, "list_transform(json_array('a','b'), x -> x)"), s(r#"["a","b"]"#));
}

#[test]
fn list_transform_null_element_matches_duckdb() {
    let mut sess = session_with_basic();
    // duckdb: list_transform([1,2,NULL,4], x -> x + 1) -> [2,3,NULL,5]
    // A NULL argument to `json_array` is embedded as JSON `null`
    // (the SQL-NULL representation for list elements; see the module-level doc).
    let e = format!("list_transform(json_array(1,2,NULL,4), x -> {} + 1)", int_cast("x"));
    assert_eq!(one(&mut sess, &e), s("[2,3,null,5]"));
}

#[test]
fn list_transform_empty_array_and_null_list() {
    let mut sess = session_with_basic();
    // duckdb: list_transform([]::INTEGER[], x -> x + 1) -> []
    assert_eq!(one(&mut sess, "list_transform(CAST('[]' AS JSON), x -> x)"), s("[]"));
    // duckdb: list_transform(NULL, x -> x + 1) -> NULL
    assert_eq!(one(&mut sess, "list_transform(NULL, x -> x)"), Value::Null);
}

#[test]
fn list_transform_non_array_json_is_null() {
    let mut sess = session_with_basic();
    // duckdb is statically typed, so a non-array cannot even be written there in the first
    // place. This engine treats LIST as a dynamically typed JSON value, so a non-array can
    // arrive at runtime. It rounds down to SQL NULL with the same leniency as other list_*
    // functions (e.g. `list_extract`) (a known incompatibility).
    assert_eq!(one(&mut sess, r#"list_transform(CAST('{"a":1}' AS JSON), x -> x)"#), Value::Null);
}

#[test]
fn list_transform_nested_lambda() {
    let mut sess = session_with_basic();
    // duckdb: list_transform([[1,2],[3,4]], y -> list_transform(y, x -> x*2))
    //   -> [[2,4],[6,8]]
    let inner = format!("list_transform(y, x -> {} * 2)", int_cast("x"));
    let e = format!("list_transform(json_array(json_array(1,2), json_array(3,4)), y -> {inner})");
    assert_eq!(one(&mut sess, &e), s("[[2,4],[6,8]]"));
}

#[test]
fn list_transform_over_table_rows_with_a_null_list() {
    let mut sess = session_with_basic();
    // The id=0 row's list is itself NULL, and the id=1 row is a single-element array. Verify
    // that even when multiple rows pass through in one query, each row is processed correctly
    // (`one()` only looks at one row, so verify multiple rows separately here).
    let rows = run(
        &mut sess,
        "SELECT list_transform(CASE WHEN id = 0 THEN NULL ELSE json_array(id) END, x -> x) \
         FROM t WHERE id IN (0, 1) ORDER BY id",
    );
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0][0], Value::Null);
    assert_eq!(rows[1][0], s("[1]"));
}

// --- list_filter ----------------------------------------------------------------

#[test]
fn list_filter_keeps_elements_matching_the_predicate() {
    let mut sess = session_with_basic();
    // duckdb: list_filter([1,2,3,4,5], x -> x > 2) -> [3,4,5]
    let e = format!("list_filter(json_array(1,2,3,4,5), x -> {} > 2)", int_cast("x"));
    assert_eq!(one(&mut sess, &e), s("[3,4,5]"));
}

#[test]
fn list_filter_treats_null_predicate_as_false() {
    let mut sess = session_with_basic();
    // duckdb: list_filter([1,2,NULL,4], x -> x > 1) -> [2,4]
    // (NULL > 1) becomes NULL, which is excluded as false under SQL three-valued logic.
    let e = format!("list_filter(json_array(1,2,NULL,4), x -> {} > 1)", int_cast("x"));
    assert_eq!(one(&mut sess, &e), s("[2,4]"));
}

#[test]
fn list_filter_equality_needs_no_cast() {
    let mut sess = session_with_basic();
    // An equality comparison between two JSON values can be used as-is without a cast
    // (`Ty::Json` specially allows only `Eq`/`Ne`; see `plan::compile::Compiler::binary`).
    assert_eq!(
        one(&mut sess, "list_filter(json_array(1,2,3), x -> x = CAST('2' AS JSON))"),
        s("[2]")
    );
}

#[test]
fn list_filter_requires_a_boolean_body() {
    let mut sess = session_with_basic();
    // A TypeMismatch at compile time if the predicate is not BOOLEAN.
    assert_eq!(
        code_of(sess.prepare("SELECT list_filter(json_array(1,2,3), x -> x) FROM t", &[])),
        Some(Code::TypeMismatch)
    );
}

#[test]
fn list_filter_empty_result_is_an_empty_array_not_null() {
    let mut sess = session_with_basic();
    let e = format!("list_filter(json_array(1,2,3), x -> {} > 100)", int_cast("x"));
    assert_eq!(one(&mut sess, &e), s("[]"));
}

// --- list_reduce ------------------------------------------------------------------

#[test]
fn list_reduce_folds_without_initial_value() {
    let mut sess = session_with_basic();
    // duckdb: list_reduce([1,2,3,4], (acc, x) -> acc + x) -> 10
    let e = format!(
        "list_reduce(json_array(1,2,3,4), (acc, x) -> {} + {})",
        int_cast("acc"),
        int_cast("x")
    );
    assert_eq!(one(&mut sess, &e), s("10"));
}

#[test]
fn list_reduce_with_initial_value() {
    let mut sess = session_with_basic();
    // duckdb: list_reduce([]::INTEGER[], (acc, x) -> acc + x, 100) -> 100
    let e = format!(
        "list_reduce(CAST('[]' AS JSON), (acc, x) -> {} + {}, to_json(100))",
        int_cast("acc"),
        int_cast("x")
    );
    assert_eq!(one(&mut sess, &e), s("100"));
}

#[test]
fn list_reduce_single_element_without_initial_returns_the_element() {
    let mut sess = session_with_basic();
    // duckdb: list_reduce([5], (acc, x) -> acc + x) -> 5
    let e =
        format!("list_reduce(json_array(5), (acc, x) -> {} + {})", int_cast("acc"), int_cast("x"));
    assert_eq!(one(&mut sess, &e), s("5"));
}

#[test]
fn list_reduce_null_list_is_null() {
    let mut sess = session_with_basic();
    // duckdb: list_reduce(NULL, (acc, x) -> acc + x) -> NULL
    let e = format!("list_reduce(NULL, (acc, x) -> {} + {})", int_cast("acc"), int_cast("x"));
    assert_eq!(one(&mut sess, &e), Value::Null);
}

#[test]
fn list_reduce_null_element_poisons_the_result() {
    let mut sess = session_with_basic();
    // duckdb: list_reduce([1,2,NULL,4], (acc, x) -> acc + x) -> NULL
    let e = format!(
        "list_reduce(json_array(1,2,NULL,4), (acc, x) -> {} + {})",
        int_cast("acc"),
        int_cast("x")
    );
    assert_eq!(one(&mut sess, &e), Value::Null);
}

#[test]
fn list_reduce_coalesce_recovers_from_a_null_accumulator() {
    let mut sess = session_with_basic();
    // duckdb: list_reduce([NULL, 1, 2], (acc, x) -> coalesce(acc, x)) -> 1
    assert_eq!(
        one(&mut sess, "list_reduce(json_array(NULL, 1, 2), (acc, x) -> coalesce(acc, x))"),
        s("1")
    );
    assert_eq!(
        one(&mut sess, "list_reduce(json_array(1, 2), (acc, x) -> coalesce(acc, x), NULL)"),
        s("1")
    );
}

#[test]
fn list_reduce_empty_without_initial_is_null_unlike_duckdb() {
    let mut sess = session_with_basic();
    // duckdb: list_reduce([]::INTEGER[], (acc, x) -> acc + x) is an error
    // ("Cannot perform list_reduce on an empty input list"). This engine prioritizes the
    // same "round down leniently to NULL" policy as the other list_* functions, returning SQL
    // NULL rather than failing the whole query (a known incompatibility).
    let e = format!(
        "list_reduce(CAST('[]' AS JSON), (acc, x) -> {} + {})",
        int_cast("acc"),
        int_cast("x")
    );
    assert_eq!(one(&mut sess, &e), Value::Null);
}

// --- Known limitation: cannot reference columns from the outer scope ----------------

// --- Edge cases: argument count / error paths ----------------------------------

#[test]
fn list_transform_rejects_a_lambda_with_the_wrong_param_count() {
    let mut sess = session_with_basic();
    // `list_transform` only accepts a single-argument lambda.
    assert_eq!(
        code_of(sess.prepare("SELECT list_transform(json_array(1,2,3), (x,y) -> x) FROM t", &[])),
        Some(Code::WrongArgCount)
    );
}

#[test]
fn list_reduce_rejects_a_lambda_with_the_wrong_param_count() {
    let mut sess = session_with_basic();
    // `list_reduce` requires a two-argument (acc, x) lambda.
    assert_eq!(
        code_of(sess.prepare("SELECT list_reduce(json_array(1,2,3), x -> x) FROM t", &[])),
        Some(Code::WrongArgCount)
    );
}

#[test]
fn list_filter_rejects_a_lambda_with_the_wrong_param_count() {
    let mut sess = session_with_basic();
    assert_eq!(
        code_of(sess.prepare("SELECT list_filter(json_array(1,2,3), (a,b) -> a) FROM t", &[])),
        Some(Code::WrongArgCount)
    );
}

#[test]
fn lambda_taking_function_requires_the_lambda_argument() {
    let mut sess = session_with_basic();
    // Omitting the lambda argument itself is an argument-count error.
    assert_eq!(
        code_of(sess.prepare("SELECT list_transform(json_array(1,2,3)) FROM t", &[])),
        Some(Code::WrongArgCount)
    );
    // Extra arguments are likewise rejected.
    assert_eq!(
        code_of(
            sess.prepare("SELECT list_transform(json_array(1,2,3), x -> x, x -> x) FROM t", &[])
        ),
        Some(Code::WrongArgCount)
    );
}

#[test]
fn list_transform_on_a_non_json_argument_is_a_type_error_at_prepare_time() {
    let mut sess = session_with_basic();
    // The 1st argument must be JSON (LIST). `5` is an integer literal whose type is known
    // statically, so it becomes `TypeMismatch` at `prepare` time
    // (a different case from `list_transform_non_array_json_is_null`, where a non-array JSON
    // value arrives at runtime: there the type is JSON but the content isn't an array).
    assert_eq!(
        code_of(sess.prepare("SELECT list_transform(5, x -> x) FROM t", &[])),
        Some(Code::TypeMismatch)
    );
}

// --- Parameter shadowing with nested lambdas ------------------------------------

#[test]
fn nested_lambda_params_with_the_same_name_shadow_correctly() {
    let mut sess = session_with_basic();
    // The inner lambda's `x` shadows the outer `x`. Regardless of the outer element's value,
    // the inner one should always return `[9]`, the conversion of `json_array(9)`.
    let e = "list_transform(json_array(1,2,3), x -> list_transform(json_array(9), x -> x))";
    assert_eq!(one(&mut sess, e), s("[[9],[9],[9]]"));
}

#[test]
fn lambda_body_cannot_reference_outer_scope_columns() {
    let mut sess = session_with_basic();
    // `id` is a column from the outer scope (FROM t), not a lambda parameter. A lambda body
    // can only reference its own parameters (see the doc on
    // `plan::compile::Compiler::lambda_call`), so the outer column reference cannot resolve and
    // becomes ColumnNotFound.
    assert_eq!(
        code_of(sess.prepare("SELECT list_transform(json_array(1,2,3), x -> x + id) FROM t", &[])),
        Some(Code::ColumnNotFound)
    );
}

// --- Merging with a pushed-down predicate -----------------------------------------

/// A lambda call sitting in a `WHERE` conjunct next to a pushdown-able
/// equality. The equality is consumed into the scan's pruner and compiled as
/// its own program, then merged with the residual conjunct by
/// `plan::compile::and_programs`. That merge has to carry the residual side's
/// lambda table over — it used to be dropped, so the query failed with
/// `Internal` while the same conjuncts in the opposite order worked, which is
/// why both orders are checked here.
#[test]
fn lambda_in_a_conjunct_next_to_a_pushed_down_equality() {
    let mut sess = session_with_basic();
    let gt = |n: &str| {
        format!(
            "json_array_length(list_filter(json_array(1, 2, 3), x -> {} > {n})) ",
            int_cast("x")
        )
    };
    let pred = format!("{}= 2", gt("1"));
    let expect = vec![vec![Value::I32(1)]];
    assert_eq!(run(&mut sess, &format!("SELECT id FROM t WHERE id = 1 AND {pred}")), expect);
    assert_eq!(run(&mut sess, &format!("SELECT id FROM t WHERE {pred} AND id = 1")), expect);
    // Lambdas on both sides of a merge: the two lambda tables must stay
    // distinct rather than the second aliasing the first.
    let other = format!("{}= 1", gt("2"));
    assert_eq!(
        run(&mut sess, &format!("SELECT id FROM t WHERE id = 1 AND {pred} AND {other}")),
        expect
    );
}
