//! 配列リテラル `[..]`・`printf`/`format`・`GLOB`/`SIMILAR TO` の統合テスト。
//!
//! 期待値は `duckdb -c "SELECT ..."` の実際の出力と突き合わせて決めている。
//! サポート範囲・既知の制限は `sql::parser`（`array_literal`/`similar_to`）・
//! `expr::funcs`（`printf_scan`/`format_scan`/`glob_match`/
//! `regexp_full_match_build`）のコメントを参照。
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

// --- 配列リテラル ------------------------------------------------------------

#[test]
fn array_literal_is_sugar_for_list_value() {
    let mut sess = session_with_basic();
    // duckdb: [1,2,3] = list_value(1,2,3) -> true
    assert_eq!(one(&mut sess, "[1, 2, 3] = list_value(1, 2, 3)"), Value::Bool(true));
    // duckdb: [1,2,3][1] -> 1（1 始まり。list_extract 経由で確認）。
    // `list_extract` の結果は Ty::Json（生の JSON テキスト）を返す設計
    // なので、期待値もテキストの "10" になる（`list_extract` のモジュール
    // 冒頭 doc・`funcs.rs` の `list_extract_is_one_based_with_negative_from_end`
    // ユニットテスト参照）。
    assert_eq!(one(&mut sess, "list_extract([10, 20, 30], 1)"), s("10"));
    // duckdb: [] は有効な式（空配列）。json_array_length([]) = 0。
    assert_eq!(one(&mut sess, "json_array_length([])"), Value::I64(0));
    // 混在型も許す（`list_value`/`json_array` と同じ）。
    assert_eq!(one(&mut sess, "to_json([1, 'x', true])"), s(r#"[1,"x",true]"#));
}

#[test]
fn array_literal_only_at_expression_start() {
    let mut sess = session_with_basic();
    // 添字アクセス `expr[i]` は今回のスコープ外。列名の直後の `[` は
    // 配列リテラルとして解釈されず、構文エラーになる。
    assert_eq!(
        code_of(sess.prepare("SELECT id[1] FROM t", &[]).map(|_| ())),
        Some(Code::UnexpectedToken)
    );
}

// --- printf / format ----------------------------------------------------------

#[test]
fn printf_matches_duckdb_on_common_specifiers() {
    let mut sess = session_with_basic();
    // duckdb: printf('%d-%s', 42, 'x') = '42-x'
    assert_eq!(one(&mut sess, "printf('%d-%s', 42, 'x')"), s("42-x"));
    // duckdb: printf('%%') = '%'
    assert_eq!(one(&mut sess, "printf('%%')"), s("%"));
    // duckdb: printf('%05d', 3) = '00003'、printf('%05d', -3) = '-0003'
    assert_eq!(one(&mut sess, "printf('%05d', 3)"), s("00003"));
    assert_eq!(one(&mut sess, "printf('%05d', -3)"), s("-0003"));
    // duckdb: printf('%-5d|', 3) = '3    |'
    assert_eq!(one(&mut sess, "printf('%-5d|', 3)"), s("3    |"));
    // duckdb: printf('%.2f', 3.14159) = '3.14'、printf('%f', 3.5) = '3.500000'
    assert_eq!(one(&mut sess, "printf('%.2f', 3.14159)"), s("3.14"));
    assert_eq!(one(&mut sess, "printf('%f', 3.5)"), s("3.500000"));
    // NULL 引数は結果全体を NULL にする（既定の NULL 伝播）。
    assert_eq!(one(&mut sess, "printf('%s', NULL)"), Value::Null);
    // 表の列を実引数として使える（定数畳み込みだけでなく実データでも動く）。
    assert_eq!(one(&mut sess, "printf('id=%d name=%s', id, name)"), s("id=0 name=name_0"));
}

#[test]
fn printf_rejects_unsupported_specifiers_and_short_arg_lists() {
    let mut sess = session_with_basic();
    // 書式文字列は定数とは限らない（列にもできる）ので、`%x` のような
    // 対応外の変換文字が混ざっていても `prepare`（型検査だけ）の時点では
    // 分からず、実行時（`step`）で初めてエラーになる。
    let mut q = match sess.prepare("SELECT printf('%x', 1) FROM t", &[]).unwrap() {
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
                assert_eq!(e.code, Code::UnsupportedFeature);
                saw_error = true;
                break;
            }
        }
    }
    assert!(saw_error, "printf('%x', ..) は対応外の変換文字なのでエラーになるはず");
}

#[test]
fn format_supports_python_style_placeholders() {
    let mut sess = session_with_basic();
    // duckdb: format('{}-{}', 42, 'x') = '42-x'
    assert_eq!(one(&mut sess, "format('{}-{}', 42, 'x')"), s("42-x"));
    // duckdb: format('{{literal}}') = '{literal}'
    assert_eq!(one(&mut sess, "format('{{literal}}')"), s("{literal}"));
    // duckdb: format('{1}-{0}', 'a', 'b') = 'b-a'
    assert_eq!(one(&mut sess, "format('{1}-{0}', 'a', 'b')"), s("b-a"));
}

// --- GLOB / SIMILAR TO ---------------------------------------------------------

#[test]
fn glob_matches_shell_patterns_over_table_rows() {
    let mut sess = session_with_basic();
    // duckdb: 'name_0' GLOB 'name_?' = true, 'name_0' GLOB 'name_1' = false
    assert_eq!(one(&mut sess, "name GLOB 'name_?'"), Value::Bool(true));
    assert_eq!(one(&mut sess, "'name_0' GLOB 'name_1'"), Value::Bool(false));
    // 実データに対しても正しくフィルタできる（`WHERE` 句での利用）。
    // `basic.parquet` は `name_0`..`name_6` の 7 種類の値を 1000 行に
    // わたって繰り返すだけの表なので、`DISTINCT` で候補集合だけを見る。
    let rows =
        run(&mut sess, "SELECT DISTINCT name FROM t WHERE name GLOB 'name_[01]' ORDER BY name");
    assert_eq!(rows, vec![vec![s("name_0")], vec![s("name_1")]]);
    // `NOT (x GLOB y)` は書けるが `x NOT GLOB y` は DuckDB 同様に構文エラー。
    assert_eq!(one(&mut sess, "NOT ('abc' GLOB 'x*')"), Value::Bool(true));
    assert_eq!(
        code_of(sess.prepare("SELECT 'abc' NOT GLOB 'x*' FROM t", &[]).map(|_| ())),
        Some(Code::UnexpectedToken)
    );
}

#[test]
fn similar_to_is_full_match_regexp() {
    let mut sess = session_with_basic();
    // duckdb: 'abc' similar to 'a.c' = true、'Xabc' similar to 'a.c' = false
    // （部分一致ではなく完全一致）。
    assert_eq!(one(&mut sess, "'abc' SIMILAR TO 'a.c'"), Value::Bool(true));
    assert_eq!(one(&mut sess, "'Xabc' SIMILAR TO 'a.c'"), Value::Bool(false));
    assert_eq!(one(&mut sess, "'Xabc' NOT SIMILAR TO 'a.c'"), Value::Bool(true));
    // 実データに対するフィルタ（`basic.parquet` は `name_0`..`name_6` の
    // 7 種類の値を繰り返すだけなので `DISTINCT` で候補集合だけを見る）。
    let rows = run(
        &mut sess,
        "SELECT DISTINCT name FROM t WHERE name SIMILAR TO 'name_[0-1]' ORDER BY name",
    );
    assert_eq!(rows, vec![vec![s("name_0")], vec![s("name_1")]]);
    // ESCAPE 句は DuckDB 自身も未実装として拒否するので、ここでも拒否する。
    assert_eq!(
        code_of(sess.prepare(r"SELECT 'a' SIMILAR TO 'a' ESCAPE '\' FROM t", &[]).map(|_| ())),
        Some(Code::UnsupportedFeature)
    );
}
