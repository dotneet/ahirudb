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
fn subscript_on_a_non_json_column_binds_fine_but_fails_at_execution() {
    // `expr[i]` is no longer scope-limited to array literals — it now
    // desugars to `list_extract(expr, i)` for any `expr` (see
    // `sql::parser::Parser::subscript`, added alongside this test's rename).
    // `id` is `Ty::Int`, and `list_extract`'s first parameter is `Ty::Json`,
    // so the call still needs an implicit Int -> Json coercion. That
    // coercion is *not* rejected at bind time — `plan::compile::Compiler::
    // coerce` unconditionally emits a `Cast` instruction without checking
    // in advance whether the specific (from, to) pair is even a legal cast
    // (this is pre-existing behavior of the generic function-argument
    // coercion path, not something this change introduced). The rejection
    // instead happens at the `Cast` kernel, which only allows VARCHAR <->
    // JSON (`expr::kernels::cast_impl`), i.e. at *execution* time.
    let mut sess = session_with_basic();
    assert_eq!(code_of(sess.prepare("SELECT id[1] FROM t", &[]).map(|_| ())), None);
    let mut q = match sess.prepare("SELECT id[1] AS x FROM t", &[]).unwrap() {
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

// --- 追加のエッジケース ---------------------------------------------------------

#[test]
fn printf_too_few_args_is_a_runtime_error() {
    let mut sess = session_with_basic();
    // duckdb: printf('%d-%d', 1) => Invalid Input Error ("Argument index ...
    // out of range")。書式文字列は列にもなり得るので、このエンジンでも
    // 実行時（`step`）エラーとして検出される（`printf_rejects_unsupported_
    // specifiers_and_short_arg_lists` と同じ理由）。
    let mut q = match sess.prepare("SELECT printf('%d-%d', 1) FROM t", &[]).unwrap() {
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
                assert_eq!(e.code, Code::WrongArgCount);
                saw_error = true;
                break;
            }
        }
    }
    assert!(saw_error, "printf('%d-%d', 1) は引数不足なのでエラーになるはず");
}

#[test]
fn printf_extra_args_are_ignored_like_duckdb() {
    let mut sess = session_with_basic();
    // duckdb: printf('%d', 1, 2) = '1'（余った引数は無視される）。
    assert_eq!(one(&mut sess, "printf('%d', 1, 2)"), s("1"));
}

#[test]
fn format_too_few_args_and_out_of_range_index_are_runtime_errors() {
    let mut sess = session_with_basic();
    // duckdb: format('{} {} {}', 1, 2) => 引数が足りずエラー。書式文字列は
    // 列にもなり得るので `prepare`（型検査）自体はここでは成功し、実行時
    // （`step`）で初めて検出される。
    let mut q = match sess.prepare("SELECT format('{} {} {}', 1, 2) FROM t", &[]).unwrap() {
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
                assert_eq!(e.code, Code::WrongArgCount);
                saw_error = true;
                break;
            }
        }
    }
    assert!(saw_error);
    // duckdb: format('{2}', 1, 2) => インデックス 2 は範囲外でエラー
    // （0 始まりなので有効なのは {0}/{1} のみ）。
    let mut q2 = match sess.prepare("SELECT format('{2}', 1, 2) FROM t", &[]).unwrap() {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("unexpected NeedIo"),
    };
    let mut saw_error2 = false;
    loop {
        match sess.step(&mut q2) {
            Ok(QueryStep::Batch(_)) => continue,
            Ok(QueryStep::Done) => break,
            Ok(QueryStep::NeedIo(_)) | Ok(QueryStep::NeedCodec(_)) => panic!("unexpected"),
            Err(e) => {
                assert_eq!(e.code, Code::WrongArgCount);
                saw_error2 = true;
                break;
            }
        }
    }
    assert!(saw_error2);
}

#[test]
fn glob_supports_negated_character_classes_and_escaping() {
    let mut sess = session_with_basic();
    // duckdb: 'abc' GLOB '[!a]*' = false（先頭が 'a' なので否定クラスに
    // マッチしない）、'xbc' GLOB '[!a]*' = true。
    assert_eq!(one(&mut sess, "'abc' GLOB '[!a]*'"), Value::Bool(false));
    assert_eq!(one(&mut sess, "'xbc' GLOB '[!a]*'"), Value::Bool(true));
    // duckdb: 'a*b' GLOB 'a\*b' = true（バックスラッシュで `*` をリテラル
    // 扱いにエスケープできる）。
    assert_eq!(one(&mut sess, r"'a*b' GLOB 'a\*b'"), Value::Bool(true));
    // 文字クラスの範囲指定も動く。
    assert_eq!(one(&mut sess, "'c' GLOB '[a-z]'"), Value::Bool(true));
    assert_eq!(one(&mut sess, "'C' GLOB '[a-z]'"), Value::Bool(false));
}

#[test]
fn similar_to_supports_alternation_and_quantifiers() {
    let mut sess = session_with_basic();
    // duckdb: 'abc' similar to '(a|x)bc' = true、'aaab' similar to
    // 'a{2,3}b' = true、'ab' similar to 'a+b?' = true。
    assert_eq!(one(&mut sess, "'abc' SIMILAR TO '(a|x)bc'"), Value::Bool(true));
    assert_eq!(one(&mut sess, "'aaab' SIMILAR TO 'a{2,3}b'"), Value::Bool(true));
    assert_eq!(one(&mut sess, "'ab' SIMILAR TO 'a+b?'"), Value::Bool(true));
}

#[test]
fn array_literal_allows_mixed_numeric_types_and_nesting() {
    let mut sess = session_with_basic();
    // 整数と浮動小数の混在も特に変換せずそのまま JSON へ乗る
    // （`Ty::Json` は動的型付けなので duckdb のような数値統一は行わない。
    // `array_literal_is_sugar_for_list_value` の混在型テストと同じ方針）。
    assert_eq!(one(&mut sess, "to_json([1, 2.5])"), s("[1,2.5]"));
    // NULL 要素を含む配列。
    assert_eq!(one(&mut sess, "to_json([1, NULL, 3])"), s("[1,null,3]"));
    // 配列の配列（ネスト）。
    assert_eq!(one(&mut sess, "to_json([1, [2, 3]])"), s("[1,[2,3]]"));
}

// --- 相互作用: WHERE/JOIN/集約との組み合わせ ------------------------------------

#[test]
fn glob_and_similar_to_work_as_join_conditions() {
    let mut sess = session_with_basic();
    // `GLOB`/`SIMILAR TO` は普通の bool 式なので `JOIN ... ON` にも書ける。
    let rows = run(
        &mut sess,
        "SELECT DISTINCT a.name FROM t a JOIN t b ON a.name GLOB b.name \
         WHERE a.name = 'name_0' ORDER BY a.name",
    );
    assert_eq!(rows, vec![vec![s("name_0")]]);
}

#[test]
fn array_literal_can_feed_an_aggregate_input() {
    let mut sess = session_with_basic();
    // 配列リテラルの要素数（`json_array_length`）を集約する
    // — 新機能（配列リテラル）を既存の集約パイプラインと組み合わせる。
    let rows = run(&mut sess, "SELECT sum(json_array_length([id, id, id])) FROM t WHERE id < 4");
    assert_eq!(rows, vec![vec![Value::I128(12)]]);
}
