//! ラムダ式（`x -> expr` / `(a, b) -> expr`）と `list_transform`/`list_filter`/
//! `list_reduce` の統合テスト。
//!
//! 期待値は `duckdb -c "SELECT ..."` の実際の出力と突き合わせて決めている。
//! ただしこのエンジンは LIST を「動的型付けの JSON 値」として実装している
//! （`Ty::Json`。`crates/ahiru-core/src/vector/types.rs` の doc 参照）ため、
//! duckdb のようにリスト要素がネイティブな数値型を持つわけではない。
//! ラムダ本体でパラメータに算術・比較を行うには、`json_extract`/
//! `list_extract` の結果に対する既存の制限と同じく
//! `CAST(CAST(x AS VARCHAR) AS INTEGER)` のように一度 VARCHAR を経由して
//! 明示的に変換する必要がある（ラムダ固有の制約ではない）。
//! ラムダ本体は自分のパラメータだけを参照でき、外側の SQL スコープの列は
//! 参照できない（既知の制限、`plan::compile::Compiler::lambda_call` の doc
//! 参照）。
//!
//! `FROM` 無しの `SELECT <expr>` は v1 未対応（`plan::bind`）なので、行を
//! 1 本だけ得るために `tests/data/basic.parquet` を `LIMIT 1` で使う
//! （`json_functions.rs` と同じ流儀）。

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

/// このエンジンで JSON 配列の要素に算術・比較を行うときの共通イディオム。
/// `Ty::Json` は他のどの型とも `Ty::unify` しないので、いったん VARCHAR を
/// 経由してから数値へ変換する。
fn int_cast(x: &str) -> String {
    format!("CAST(CAST({x} AS VARCHAR) AS INTEGER)")
}

// --- 構文: `x -> expr` / `(a, b) -> expr` ------------------------------------

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
    // `coalesce` はラムダを受け取らないので、引数位置でも `->` は今まで通り
    // JSON パス演算子のまま（duckdb CLI で実測: `coalesce(doc -> 'a', 'x')` は
    // ラムダとしては解釈されず JSON 抽出として解決される）。
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
    // 文字列要素はそのままの JSON テキスト（引用符付き）で返る。
    assert_eq!(one(&mut sess, "list_transform(json_array('a','b'), x -> x)"), s(r#"["a","b"]"#));
}

#[test]
fn list_transform_null_element_matches_duckdb() {
    let mut sess = session_with_basic();
    // duckdb: list_transform([1,2,NULL,4], x -> x + 1) -> [2,3,NULL,5]
    // `json_array` の NULL 引数は JSON `null` として埋め込まれる
    // （リスト要素の SQL NULL 表現。モジュール冒頭 doc 参照）。
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
    // duckdb は静的型付けなので非配列は最初から書けない。このエンジンは
    // LIST を動的型付けの JSON 値として扱うため、実行時に非配列が来ることが
    // ありうる。他の list_* 関数（`list_extract` 等）と同じ寛容さで SQL NULL
    // に丸める（既知の非互換）。
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
    // id=0 の行はリスト自体が NULL、id=1 の行は 1 要素の配列。1 回のクエリで
    // 複数行を通しても正しく行ごとに処理されることを確認する
    // （`one()` は 1 行しか見ないので別途複数行で確認する）。
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
    // (NULL > 1) は NULL になり、SQL の 3 値論理で偽として除外される。
    let e = format!("list_filter(json_array(1,2,NULL,4), x -> {} > 1)", int_cast("x"));
    assert_eq!(one(&mut sess, &e), s("[2,4]"));
}

#[test]
fn list_filter_equality_needs_no_cast() {
    let mut sess = session_with_basic();
    // JSON 同士の等価比較はキャスト無しでそのまま使える（`Ty::Json` は
    // `Eq`/`Ne` だけは特別に許す。`plan::compile::Compiler::binary` 参照）。
    assert_eq!(
        one(&mut sess, "list_filter(json_array(1,2,3), x -> x = CAST('2' AS JSON))"),
        s("[2]")
    );
}

#[test]
fn list_filter_requires_a_boolean_body() {
    let mut sess = session_with_basic();
    // 述語が BOOLEAN でなければコンパイル時に TypeMismatch。
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
fn list_reduce_empty_without_initial_is_null_unlike_duckdb() {
    let mut sess = session_with_basic();
    // duckdb: list_reduce([]::INTEGER[], (acc, x) -> acc + x) はエラー
    // （"Cannot perform list_reduce on an empty input list"）。このエンジンは
    // 他の list_* 関数と同じ「寛容に NULL へ丸める」方針を優先し、クエリ全体を
    // 失敗させず SQL NULL を返す（既知の非互換）。
    let e = format!(
        "list_reduce(CAST('[]' AS JSON), (acc, x) -> {} + {})",
        int_cast("acc"),
        int_cast("x")
    );
    assert_eq!(one(&mut sess, &e), Value::Null);
}

// --- 既知の制限: 外側スコープの列は参照できない ----------------------------------

// --- 引数個数・エラー系のエッジケース ------------------------------------------

#[test]
fn list_transform_rejects_a_lambda_with_the_wrong_param_count() {
    let mut sess = session_with_basic();
    // `list_transform` は 1 引数のラムダしか受け付けない。
    assert_eq!(
        code_of(sess.prepare("SELECT list_transform(json_array(1,2,3), (x,y) -> x) FROM t", &[])),
        Some(Code::WrongArgCount)
    );
}

#[test]
fn list_reduce_rejects_a_lambda_with_the_wrong_param_count() {
    let mut sess = session_with_basic();
    // `list_reduce` は 2 引数（acc, x）のラムダが必要。
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
    // ラムダ引数そのものを省略した呼び出しは引数個数エラー。
    assert_eq!(
        code_of(sess.prepare("SELECT list_transform(json_array(1,2,3)) FROM t", &[])),
        Some(Code::WrongArgCount)
    );
    // 余分な引数も同様に拒否される。
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
    // 第 1 引数は JSON（LIST）でなければならない。`5` は整数リテラルで
    // 型が静的に分かるので、`prepare` 時点で `TypeMismatch` になる
    // （非配列の JSON 値が実行時に来る `list_transform_non_array_json_is_null`
    // とは違うケース: あちらは型は JSON だが中身が配列でない場合）。
    assert_eq!(
        code_of(sess.prepare("SELECT list_transform(5, x -> x) FROM t", &[])),
        Some(Code::TypeMismatch)
    );
}

// --- ネストしたラムダでのパラメータ隠蔽 ----------------------------------------

#[test]
fn nested_lambda_params_with_the_same_name_shadow_correctly() {
    let mut sess = session_with_basic();
    // 内側のラムダの `x` は外側の `x` を隠す。外側の要素の値に関わらず、
    // 内側は常に `json_array(9)` を変換した `[9]` を返すはず。
    let e = "list_transform(json_array(1,2,3), x -> list_transform(json_array(9), x -> x))";
    assert_eq!(one(&mut sess, e), s("[[9],[9],[9]]"));
}

#[test]
fn lambda_body_cannot_reference_outer_scope_columns() {
    let mut sess = session_with_basic();
    // `id` は外側（FROM t）の列で、ラムダのパラメータではない。ラムダ本体は
    // 自分のパラメータだけを参照できる（`plan::compile::Compiler::lambda_call`
    // の doc 参照）ので、外側の列参照は解決できず ColumnNotFound になる。
    assert_eq!(
        code_of(sess.prepare("SELECT list_transform(json_array(1,2,3), x -> x + id) FROM t", &[])),
        Some(Code::ColumnNotFound)
    );
}
