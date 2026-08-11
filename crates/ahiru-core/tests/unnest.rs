//! `UNNEST` の統合テスト。
//!
//! 期待値は `duckdb` CLI の実際の出力と突き合わせて決めている
//! （`tests/data/list_varied.parquet`/`list1.parquet` は既存フィクスチャ、
//! `scripts/gen-testdata.sh` 参照。生成方法・内容は変更しない）。
//!
//! - SELECT リスト / FROM 句の両構文（タスクのパターン (a)/(b)）
//! - 対象列がテーブルの `Ty::Json` 列そのものの場合は要素をネイティブ型へ
//!   復元しない（実データを見ないと安全に判定できないため。`plan::bind`
//!   の `narrow_unnest_elem_ty` ドキュメント参照）ので、期待値は生の JSON
//!   トークンのバイト列で比較する。
//! - `UNNEST(list_value(...))`/`UNNEST(json_array(...))` のように、対象が
//!   その場でリストを組み立てる呼び出しで、かつ全要素が入れ子を持たない
//!   同じスカラ型なら、ネイティブ型（BIGINT/DOUBLE/VARCHAR/BOOLEAN）へ
//!   復元されることをスキーマと値の両方で確認する。
//! - NeedIo をまたいだ再開が結果を変えないこと（実 Parquet ファイルを
//!   `register_remote_as` + `provide` で少しずつ供給し、`register_bytes`
//!   で一括供給した場合と結果を突き合わせる）。

use ahiru_core::error::{code_of, Code};
use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::{Field, Ty, Value};

fn data(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// `dual`（1 行だけのダミー表）を登録したセッション。`FROM` 無しの
/// リテラルだけの `SELECT` を v1 が対象外にしているための迂回
/// （`recursive_cte.rs` の `session_with_dual` と同じ理由）。
fn session_with_dual() -> Session {
    let mut s = Session::new();
    s.register_bytes_as("dual", b"x\n1\n".to_vec(), FormatKind::Csv).unwrap();
    s
}

/// 全データがメモリ上にあるクエリを最後まで実行する。
/// `NeedIo`/`NeedCodec` は絶対に起きない前提。
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
/// `Ty::Json` のまま返る要素の期待値。JSON トークンのバイト列そのもの
/// （数値ならクォート無しの数字テキスト）で比較する。
fn json_tok(v: &str) -> Value {
    Value::Bytes(v.as_bytes().to_vec())
}
const NULL: Value = Value::Null;

// --- SELECT リスト -----------------------------------------------------------

/// duckdb: `SELECT id, UNNEST(xs) AS x FROM 'list1.parquet' WHERE id < 2` →
/// (0,1) (0,2) (0,3) (1,1) (1,2) (1,3)（全行 `[1,2,3]`）。
/// 対象がテーブルの JSON 列そのものなので、値は復元されず JSON トークンの
/// ままであることも合わせて確認する（`schema[1].ty == Ty::Json`）。
#[test]
fn select_list_unnest_duplicates_other_columns_per_element() {
    let mut sess = Session::new();
    sess.register_bytes_as("t", data("list1.parquet"), FormatKind::Parquet).unwrap();
    let (schema, rows) = run(&mut sess, "SELECT id, UNNEST(xs) AS x FROM t WHERE id < 2");
    assert_eq!(schema[1].name, "x");
    assert_eq!(schema[1].ty, Ty::Json, "テーブル列そのものの UNNEST はネイティブ型へ復元しない");
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

/// duckdb: `list_varied.parquet` は `id % 5` で NULL / 空配列 / 1〜4 要素の
/// 配列を繰り返す（`nested_files.rs::list_varied_distinguishes_...` で
/// 検証済みの規則）。NULL・空配列の行は 0 行になり、他の列（`id`）は行ごと
/// 複製されることを確認する。並び順はこのエンジンが単一の Scan を素直に
/// 頭から読むだけ（並列化・並べ替えを一切挟まない）なので、`ORDER BY` 無しで
/// ファイルの物理順そのままになる。
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

/// UNNEST の別名が無ければ duckdb と同じく列名は "unnest" になる。
#[test]
fn select_list_unnest_default_column_name_is_unnest() {
    let mut db = session_with_dual();
    let (schema, _) = run(&mut db, "SELECT UNNEST(list_value(1,2,3)) FROM dual");
    assert_eq!(schema[0].name, "unnest");
}

// --- FROM 句（暗黙 LATERAL） --------------------------------------------------

/// duckdb: `SELECT t.id, y.x FROM 'list_varied.parquet' t, UNNEST(t.xs) AS
/// y(x) WHERE t.id < 5` → (2,2) (3,3) (3,NULL) (3,6) (4,4) (4,5) (4,6) (4,7)。
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

/// `SELECT *` は `t` の元の列（`id`, `xs`）に続けて展開列が並ぶ
/// （duckdb と同じく、対象列自体は複製されて残る）。
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

/// 別名を付けなければ duckdb と同じく列名は "unnest"、`t.tags` 以外の他の
/// 兄弟列も普通に参照できる。複数の FROM 句 UNNEST の連鎖（各々が独立に
/// 行を掛け合わせる、DuckDB の LATERAL 連鎖と同じクロス積）もついでに確認。
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

// --- ネイティブ型への復元 ------------------------------------------------------

/// duckdb: `SELECT UNNEST([1,2,3])` は BIGINT 列を返す。このエンジンには
/// 配列リテラル構文が無いので `list_value(1,2,3)`（`json_array` の別名）で
/// 書く。全引数が同じ非 JSON スカラ型なので、実データを見ずに BIGINT だと
/// 判定できる（`plan::bind::narrow_unnest_elem_ty`）。
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

/// 型が揃わない（整数と文字列が混ざる）場合は復元せず `Ty::Json` のまま。
#[test]
fn unnest_of_list_value_with_mixed_types_stays_json() {
    let mut db = session_with_dual();
    let (schema, rows) = run(&mut db, "SELECT UNNEST(list_value(1,'a')) AS x FROM dual");
    assert_eq!(schema[0].ty, Ty::Json);
    assert_eq!(rows, vec![vec![json_tok("1")], vec![json_tok("\"a\"")]]);
}

/// 入れ子（配列を要素に持つ）場合も復元しない。
#[test]
fn unnest_of_nested_list_value_stays_json() {
    let mut db = session_with_dual();
    let (schema, rows) =
        run(&mut db, "SELECT UNNEST(list_value(list_value(1,2), list_value(3))) AS x FROM dual");
    assert_eq!(schema[0].ty, Ty::Json);
    assert_eq!(rows, vec![vec![json_tok("[1,2]")], vec![json_tok("[3]")]]);
}

/// 単純な列参照からの CAST（`json_array`/`list_value` の直接呼び出しでは
/// ない）は、要素の型が実データ依存なので復元しない。
#[test]
fn unnest_of_a_plain_json_cast_column_stays_json() {
    let mut db = session_with_dual();
    let (schema, rows) = run(&mut db, "SELECT UNNEST(CAST('[1,2,3]' AS JSON)) AS x FROM dual");
    assert_eq!(schema[0].ty, Ty::Json);
    assert_eq!(rows, vec![vec![json_tok("1")], vec![json_tok("2")], vec![json_tok("3")]]);
}

// --- NULL / 空配列 -------------------------------------------------------------

/// duckdb: `UNNEST(NULL)`/`UNNEST([])` はどちらも 0 行。
#[test]
fn unnest_of_null_and_empty_array_yield_zero_rows() {
    let mut db = session_with_dual();
    let (_, rows) = run(&mut db, "SELECT UNNEST(CAST(NULL AS JSON)) AS x FROM dual");
    assert!(rows.is_empty());
    let (_, rows) = run(&mut db, "SELECT UNNEST(CAST('[]' AS JSON)) AS x FROM dual");
    assert!(rows.is_empty());
}

// --- 対応外として明確に拒否するパターン -----------------------------------------

/// 複数の `UNNEST` を同じ SELECT リストに書く（DuckDB は列ごとの zip という
/// 複雑な挙動をする）のはスコープ外として拒否する。
#[test]
fn multiple_unnests_in_select_list_are_rejected() {
    let mut db = session_with_dual();
    let err = db.prepare("SELECT UNNEST(list_value(1,2)), UNNEST(list_value(3,4)) FROM dual", &[]);
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// UNNEST と集約の組み合わせ（展開後の行に対する集約の意味論を実装して
/// いない）は拒否する。
#[test]
fn unnest_combined_with_aggregation_is_rejected() {
    let mut db = session_with_dual();
    let err = db.prepare("SELECT count(*), UNNEST(list_value(1,2)) FROM dual", &[]);
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// WHERE の中に `UNNEST` を書くのは対応外（集約と同じく、書ける位置が
/// SELECT リストに限られる）。
#[test]
fn unnest_in_where_clause_is_rejected() {
    let mut db = session_with_dual();
    let err = db.prepare("SELECT 1 FROM dual WHERE UNNEST(list_value(1,2)) = 1", &[]);
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// FROM 句単独の `UNNEST`（先行する項目が無い）は暗黙 LATERAL の左隣が
/// 無いので拒否する。
#[test]
fn standalone_from_unnest_without_a_preceding_item_is_rejected() {
    let mut db = session_with_dual();
    let err = db.prepare("SELECT v FROM UNNEST(list_value(1,2,3)) AS x(v)", &[]);
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// FROM 句の `UNNEST` が JOIN の左側に来る場合も、暗黙 LATERAL は右隣しか
/// 見ないので拒否する。
#[test]
fn from_unnest_on_the_left_side_of_a_join_is_rejected() {
    let mut db = session_with_dual();
    let err = db.prepare("SELECT v FROM UNNEST(list_value(1,2,3)) AS x(v), dual", &[]);
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// UNNEST の対象が `Ty::Json` でなければ束縛時に明確に拒否する。
#[test]
fn unnest_of_a_non_json_expression_is_rejected() {
    let mut db = session_with_dual();
    let err = db.prepare("SELECT UNNEST(1) FROM dual", &[]);
    assert_eq!(code_of(err), Some(Code::TypeMismatch));
}

// --- NeedIo をまたいだ再開 ------------------------------------------------------

/// `NeedIo` を挟んでバイト列を少しずつ供給しても、一括供給（`register_bytes`）
/// と全く同じ結果になることを確認する。`Unnest` 自身は入力の `NeedIo` を
/// そのまま素通しするだけで「今展開中の行/添字」を保つ、というこのエンジン
/// 最重要の不変条件（要求 3）の実ファイル版の検証。
#[test]
fn need_io_across_a_real_parquet_scan_does_not_change_the_result() {
    let sql = "SELECT id, UNNEST(xs) AS x FROM t";
    let bytes = data("list_varied.parquet");

    let mut eager = Session::new();
    eager.register_bytes_as("t", bytes.clone(), FormatKind::Parquet).unwrap();
    let (_, want) = run(&mut eager, sql);
    assert!(!want.is_empty());

    let (got, _) = run_with_lazy_io(&bytes, sql);
    assert_eq!(got, want, "NeedIo をまたいでも結果が変わってはいけない");
}

/// より大きな `list_pagetest.parquet`（2000 行、`nested_files.rs` で使用済み
/// の既存フィクスチャ）でも同じことを確認する。この 2 ファイルはどちらも
/// フッタ解決の要求バイト範囲だけでファイル全体を賄えてしまい（小さい
/// テストフィクスチャゆえ）、`step()` の途中でさらに `NeedIo` が挟まる
/// 場面までは実ファイルでは再現できていない ―― その場面（`Unnest` が
/// 複数バッチにまたがって「今展開中の行/添字」を保ったまま再開する）は
/// `exec::unnest::tests` 側でモック入力オペレータを使って厳密に検証して
/// ある（`need_io_between_input_batches_does_not_change_the_result`/
/// `need_io_mid_row_does_not_change_the_result`）。
#[test]
fn need_io_with_a_larger_file_does_not_change_the_result() {
    let sql = "SELECT id, UNNEST(xs) AS x FROM t WHERE id < 50";
    let bytes = data("list_pagetest.parquet");

    let mut eager = Session::new();
    eager.register_bytes_as("t", bytes.clone(), FormatKind::Parquet).unwrap();
    let (_, want) = run(&mut eager, sql);
    assert!(!want.is_empty());

    let (got, rounds) = run_with_lazy_io(&bytes, sql);
    assert_eq!(got, want, "NeedIo をまたいでも結果が変わってはいけない");
    assert!(
        rounds >= 1,
        "register_remote_as は 0 バイトから始まるので必ず 1 回は NeedIo が起きるはず"
    );
}

/// `register_remote_as` で登録し、`NeedIo` が要求した範囲だけをそのつど
/// `provide` して駆動する。ホストが実際にレンジ取得を行う流れそのもの
/// （`ahiru-cli`/wasm ホストと同じ「要求された分だけ渡す」節約経路）。
/// 戻り値は `(結果行, 中断・再開の往復回数)`。
fn run_with_lazy_io(bytes: &[u8], sql: &str) -> (Vec<Vec<Value>>, u32) {
    let mut s = Session::new();
    s.register_remote_as("t", bytes.len() as u64, FormatKind::Parquet).unwrap();

    let mut rounds = 0u32;
    let mut q = loop {
        match s.prepare(sql, &[]).unwrap() {
            Prepared::Ready(q) => break q,
            Prepared::NeedIo(reqs) => {
                rounds += 1;
                assert!(rounds < 1000, "resolve_query が終わらない");
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
        assert!(steps < 10_000, "step が終わらない");
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
