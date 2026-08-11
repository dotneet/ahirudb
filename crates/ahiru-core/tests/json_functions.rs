//! JSON パス演算子・構築関数・LIST/MAP アクセサの統合テスト。
//!
//! 期待値は `duckdb -c "SELECT ..."` の実際の出力と突き合わせて決めている。
//! サポート範囲・既知の制限は `ahiru_core::json`（内部専用のためドキュメントは
//! `crates/ahiru-core/src/json.rs` 冒頭）・`expr::funcs` のコメントを参照。
//!
//! `FROM` 無しの `SELECT <expr>` は v1 未対応（`plan::bind`）なので、行を
//! 1 本だけ得るために `tests/data/basic.parquet` を `LIMIT 1` で使う。

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
    // 見つからないパスは SQL NULL。
    assert_eq!(one(&mut sess, r#"json_extract('{"a":1}', '$.b')"#), Value::Null);
}

#[test]
fn arrow_operators_are_sugar_for_json_extract_functions() {
    let mut sess = session_with_basic();
    // VARCHAR リテラルは json_extract の第 1 引数（Ty::Json 期待）へ暗黙に
    // Cast される。CAST(... AS JSON) を明示しなくても動く。
    // duckdb: '{"a":1}'::JSON -> '$.a' -> 1 (json), ->> '$.a' -> '1' (varchar)
    assert_eq!(one(&mut sess, r#"'{"a":1}' -> '$.a'"#), s("1"));
    assert_eq!(one(&mut sess, r#"'{"a":1}' ->> '$.a'"#), s("1"));
    // 比較演算子より強く結合するので括弧が要らない。
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
    // 非配列は 0（duckdb: json_array_length('{"a":1}') -> 0）。
    assert_eq!(one(&mut sess, r#"json_array_length('{"a":1}')"#), Value::I64(0));
}

// --- 構築関数: to_json / json_object / json_array / list_value --------------

#[test]
fn to_json_covers_scalar_types_like_duckdb() {
    let mut sess = session_with_basic();
    assert_eq!(one(&mut sess, "to_json(1)"), s("1"));
    assert_eq!(one(&mut sess, "to_json(1.5)"), s("1.5"));
    assert_eq!(one(&mut sess, "to_json('hello')"), s("\"hello\""));
    assert_eq!(one(&mut sess, "to_json(true)"), s("true"));
    assert_eq!(one(&mut sess, "to_json(CAST('2024-01-01' AS DATE))"), s("\"2024-01-01\""));
    // duckdb: to_json(NULL) -> SQL NULL。
    assert_eq!(one(&mut sess, "to_json(NULL)"), Value::Null);
}

#[test]
fn json_object_and_json_array_match_duckdb_construction() {
    let mut sess = session_with_basic();
    // duckdb: json_object('a', 1, 'b', 'x') -> {"a":1,"b":"x"}
    assert_eq!(one(&mut sess, "json_object('a', 1, 'b', 'x')"), s(r#"{"a":1,"b":"x"}"#));
    // duckdb: json_array(1, 'x', true, NULL) -> [1,"x",true,null]
    assert_eq!(one(&mut sess, "json_array(1, 'x', true, NULL)"), s(r#"[1,"x",true,null]"#));
    // list_value は json_array の別名。
    assert_eq!(one(&mut sess, "list_value(1, 2, 3)"), s("[1,2,3]"));
}

#[test]
fn json_functions_work_over_table_columns_not_just_literals() {
    let mut sess = session_with_basic();
    // id は BigInt 列。to_json/json_object が列値でも動くことを確認する。
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
    // 第 1 引数は Ty::Json 期待なので VARCHAR リテラルは暗黙に Cast される。
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

// --- CAST --------------------------------------------------------------------

#[test]
fn cast_varchar_to_json_round_trips_and_try_cast_is_lenient() {
    let mut sess = session_with_basic();
    assert_eq!(one(&mut sess, r#"CAST('{"a":1}' AS JSON)"#), s(r#"{"a":1}"#));
    assert_eq!(one(&mut sess, r#"CAST(CAST('{"a":1}' AS JSON) AS VARCHAR)"#), s(r#"{"a":1}"#));
    // duckdb: TRY_CAST('not json' AS JSON) -> NULL。
    assert_eq!(one(&mut sess, "TRY_CAST('not json' AS JSON)"), Value::Null);
}

#[test]
fn cast_invalid_json_text_errors_the_whole_query() {
    let mut sess = session_with_basic();
    // duckdb: CAST('not json' AS JSON) -> Conversion Error（NULL に丸めない）。
    // `name` 列は "name_0" のような非 JSON テキストなので、CAST(name AS JSON)
    // は実行時（`Session::step`）に InvalidCast で失敗する。CAST 自体は
    // 型検査だけでは不正さが分からない（値を読むまで分からない）ので、
    // `prepare` の時点ではエラーにならない点に注意。
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
    assert!(saw_error, "CAST(name AS JSON) は不正な JSON でエラーになるはず");
}

// --- 比較 ---------------------------------------------------------------------

#[test]
fn json_equality_is_byte_comparison_ordering_is_rejected() {
    let mut sess = session_with_basic();
    // キー順序・空白の違いはバイト列比較なので不一致になる（v1 の既知の制限）。
    // JSON は他のどの型とも `Ty::unify` しない（モジュール doc 参照）ので、
    // 比較する両辺を明示的に CAST して Ty::Json 同士にする。
    assert_eq!(
        one(&mut sess, r#"CAST('{"a":1,"b":2}' AS JSON) = CAST('{"a":1,"b":2}' AS JSON)"#),
        Value::Bool(true)
    );
    assert_eq!(
        one(&mut sess, r#"CAST('{"a": 1}' AS JSON) = CAST('{"a":1}' AS JSON)"#),
        Value::Bool(false)
    );
    // 大小比較は TypeMismatch。
    assert_eq!(
        code_of(
            sess.prepare("SELECT CAST('1' AS JSON) < CAST('2' AS JSON) FROM t", &[]).map(|_| ())
        ),
        Some(Code::TypeMismatch)
    );
}
