//! `generate_series`/`range` テーブル関数の統合テスト。
//!
//! 期待値は `duckdb` CLI の実際の出力と突き合わせて決めている:
//! - `range(stop)`/`range(start, stop)`/`range(start, stop, step)` は半開
//!   区間（`stop` を含まない）。
//! - `generate_series(stop)`/`generate_series(start, stop)`/
//!   `generate_series(start, stop, step)` は閉区間（`stop` を含む）。
//! - 別名を付けなければ列名はそれぞれ `"range"`/`"generate_series"`。
//! - `step` の向きと `start`/`stop` の大小が矛盾すれば 0 行（エラーにならない）。
//! - `step = 0` は束縛時エラー（`duckdb`: "interval cannot be 0!"）。

use ahiru_core::error::{code_of, Code};
use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::{Field, Ty, Value};

/// `dual`（1 行だけのダミー表）を登録したセッション。`FROM` 無しのリテラル
/// だけの `SELECT` を v1 が対象外にしているための迂回（`unnest.rs` と同じ）。
fn session_with_dual() -> Session {
    let mut s = Session::new();
    s.register_bytes_as("dual", b"x\n1\n".to_vec(), FormatKind::Csv).unwrap();
    s
}

/// 全データがメモリ上にあるクエリを最後まで実行する。
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

fn i64s(vals: impl IntoIterator<Item = i64>) -> Vec<Vec<Value>> {
    vals.into_iter().map(|v| vec![Value::I64(v)]).collect()
}

// --- range ---------------------------------------------------------------------

/// duckdb: `SELECT * FROM range(5)` → 0,1,2,3,4（`stop` を含まない）。
#[test]
fn range_single_arg_starts_at_zero_and_excludes_stop() {
    let mut db = session_with_dual();
    let (schema, rows) = run(&mut db, "SELECT * FROM range(5)");
    assert_eq!(schema[0].name, "range");
    assert_eq!(schema[0].ty, Ty::BigInt);
    assert_eq!(rows, i64s(0..5));
}

/// duckdb: `SELECT * FROM range(0, 100, 5)` → 0,5,10,...,95。
#[test]
fn range_three_args_honors_start_stop_step() {
    let mut db = session_with_dual();
    let (_, rows) = run(&mut db, "SELECT * FROM range(0, 100, 5)");
    assert_eq!(rows, i64s((0..100).step_by(5)));
}

/// duckdb: `SELECT * FROM range(10, 0, -2)` → 10,8,6,4,2。
#[test]
fn range_negative_step_counts_down() {
    let mut db = session_with_dual();
    let (_, rows) = run(&mut db, "SELECT * FROM range(10, 0, -2)");
    assert_eq!(rows, i64s([10, 8, 6, 4, 2]));
}

/// duckdb: 向きが矛盾する（正の step で start > stop、など）と 0 行。
#[test]
fn range_mismatched_direction_yields_zero_rows() {
    let mut db = session_with_dual();
    let (_, rows) = run(&mut db, "SELECT * FROM range(10, 0, 1)");
    assert!(rows.is_empty());
    let (_, rows) = run(&mut db, "SELECT * FROM range(0, 10, -1)");
    assert!(rows.is_empty());
}

// --- generate_series -------------------------------------------------------------

/// duckdb: `SELECT * FROM generate_series(5)` → 0,1,2,3,4,5（`stop` を含む）。
#[test]
fn generate_series_single_arg_starts_at_zero_and_includes_stop() {
    let mut db = session_with_dual();
    let (schema, rows) = run(&mut db, "SELECT * FROM generate_series(5)");
    assert_eq!(schema[0].name, "generate_series");
    assert_eq!(rows, i64s(0..=5));
}

/// duckdb: `SELECT * FROM generate_series(1, 10)` → 1..=10。
#[test]
fn generate_series_two_args() {
    let mut db = session_with_dual();
    let (_, rows) = run(&mut db, "SELECT * FROM generate_series(1, 10)");
    assert_eq!(rows, i64s(1..=10));
}

/// duckdb: `SELECT * FROM generate_series(0, 10, 2)` → 0,2,4,6,8,10。
#[test]
fn generate_series_three_args_with_step() {
    let mut db = session_with_dual();
    let (_, rows) = run(&mut db, "SELECT * FROM generate_series(0, 10, 2)");
    assert_eq!(rows, i64s([0, 2, 4, 6, 8, 10]));
}

// --- 別名 -------------------------------------------------------------------------

#[test]
fn column_alias_renames_the_output_column() {
    let mut db = session_with_dual();
    let (schema, rows) = run(&mut db, "SELECT x FROM range(3) AS t(x)");
    assert_eq!(schema[0].name, "x");
    assert_eq!(rows, i64s(0..3));
}

#[test]
fn table_alias_qualifies_the_column_reference() {
    let mut db = session_with_dual();
    let (_, rows) = run(&mut db, "SELECT t.x FROM range(3) AS t(x) WHERE t.x > 0");
    assert_eq!(rows, i64s(1..3));
}

// --- 組み合わせ --------------------------------------------------------------------

#[test]
fn works_with_where_and_order_by() {
    let mut db = session_with_dual();
    let (_, rows) = run(&mut db, "SELECT * FROM range(20) WHERE range % 3 = 0 ORDER BY range DESC");
    assert_eq!(rows, i64s([18, 15, 12, 9, 6, 3, 0]));
}

#[test]
fn can_join_two_generated_series() {
    let mut db = session_with_dual();
    let (_, rows) =
        run(&mut db, "SELECT a.x, b.y FROM range(2) AS a(x), range(2) AS b(y) ORDER BY a.x, b.y");
    assert_eq!(
        rows,
        vec![
            vec![Value::I64(0), Value::I64(0)],
            vec![Value::I64(0), Value::I64(1)],
            vec![Value::I64(1), Value::I64(0)],
            vec![Value::I64(1), Value::I64(1)],
        ]
    );
}

/// 大きな範囲でも（メモリへ一括展開せず）正しく生成できることを、件数と
/// 端の値で確認する（内部実装は `exec::range::GenerateSeries` の
/// `BATCH_SIZE` ずつの生成、`exec/range.rs` の単体テストで詳しく検証済み）。
#[test]
fn a_large_range_still_produces_the_correct_count_and_endpoints() {
    let mut db = session_with_dual();
    let (_, rows) = run(&mut db, "SELECT count(*), min(range), max(range) FROM range(500000)");
    assert_eq!(rows, vec![vec![Value::I64(500000), Value::I64(0), Value::I64(499999)]]);
}

// --- エラー -------------------------------------------------------------------------

/// duckdb: `step = 0` は束縛時エラー（"interval cannot be 0!"）。
#[test]
fn zero_step_is_rejected() {
    let mut db = session_with_dual();
    let err = db.prepare("SELECT * FROM range(0, 10, 0)", &[]);
    assert_eq!(code_of(err), Some(Code::DivideByZero));
}

/// 引数が無い呼び出しは拒否する（`duckdb` も `range()` を関数解決エラーにする）。
#[test]
fn zero_arguments_is_rejected() {
    let mut db = session_with_dual();
    let err = db.prepare("SELECT * FROM range()", &[]);
    assert_eq!(code_of(err), Some(Code::WrongArgCount));
}

/// 引数が多すぎる呼び出しも拒否する。
#[test]
fn too_many_arguments_is_rejected() {
    let mut db = session_with_dual();
    let err = db.prepare("SELECT * FROM range(1, 2, 3, 4)", &[]);
    assert_eq!(code_of(err), Some(Code::WrongArgCount));
}

/// 小数の引数は非対応（`sql::parser::base_rel` は `signed_int_lit` でしか
/// 引数を読まない）。`duckdb` も `generate_series(BIGINT, ...)` 以外は
/// オーバーロード解決に失敗して拒否するので、方向性は同じ（エラーの段階が
/// 構文解析時か束縛時かの違いだけ）。
#[test]
fn float_arguments_are_rejected() {
    let mut db = session_with_dual();
    let err = db.prepare("SELECT * FROM generate_series(1.5, 5.5)", &[]);
    assert!(code_of(err).is_some());
    let err = db.prepare("SELECT * FROM range(1.5)", &[]);
    assert!(code_of(err).is_some());
}

// --- キーワードと列名/表名の衝突 -------------------------------------------------
//
// `range`/`generate_series` はテーブル関数としては特別扱いされるが、予約語
// ではない（`base_rel` の doc 参照: 予約語化すると同名の列/表を壊す過去の
// 事故と同じ理由）。実データにこれらの名前の列や表があっても壊れないことを
// 確認する。

#[test]
fn a_real_column_named_range_is_not_shadowed_by_the_table_function() {
    let mut db = Session::new();
    db.register_bytes_as("t2", b"range\n7\n".to_vec(), FormatKind::Csv).unwrap();
    let (_, rows) = run(&mut db, "SELECT range FROM t2");
    assert_eq!(rows, vec![vec![Value::I64(7)]]);
    let (_, rows) = run(&mut db, "SELECT t2.range FROM t2");
    assert_eq!(rows, vec![vec![Value::I64(7)]]);
}

/// `range`/`generate_series` という名前の実表を登録した場合、`FROM range`
/// （括弧なし）は表参照として解決される。テーブル関数としての `range(...)`
/// は必ず `(` を伴うので構文上は衝突しない
/// （`base_rel` の `if self.is(Tok::LParen)` 分岐参照）。
#[test]
fn a_real_table_named_range_is_queryable_by_name() {
    let mut db = Session::new();
    db.register_bytes_as("range", b"x\n9\n".to_vec(), FormatKind::Csv).unwrap();
    let (_, rows) = run(&mut db, "SELECT x FROM range");
    assert_eq!(rows, vec![vec![Value::I64(9)]]);
}

// --- 相互作用: 実データとの JOIN -------------------------------------------------

/// `generate_series`/`range` を実表（Parquet）との `JOIN` の片側に使う。
/// duckdb: SELECT b.id FROM range(3) a JOIN 'basic.parquet' b ON a.range = b.id
///         ORDER BY b.id
///
/// `range` の列は `BIGINT`、`basic.parquet` の `id` は `INTEGER` なので、
/// `ON` 句をキャスト無しの `a.range = b.id` にすると 0 行になってしまう
/// （`plan::bind` の等結合キー抽出が異なる整数物理型を単一化していない、
/// 既知の別件バグ。`query_composition_extra.rs::
/// join_on_mixed_numeric_key_types_compares_by_value` が詳しく再現・記録
/// 済みで、`plan::bind` の担当者向けに切り分けてある。ここは
/// `generate_series`/`range` 自体の不具合ではないので、素直に明示キャスト
/// で型を揃えて確認する）。
#[test]
fn range_can_join_a_real_parquet_table() {
    let mut db = session_with_dual();
    db.register_bytes(
        "basic",
        std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/basic.parquet"))
            .unwrap(),
    )
    .unwrap();
    let (_, rows) = run(
        &mut db,
        "SELECT b.id FROM range(3) a JOIN basic b ON CAST(a.range AS INTEGER) = b.id \
         ORDER BY b.id",
    );
    assert_eq!(rows, vec![vec![Value::I32(0)], vec![Value::I32(1)], vec![Value::I32(2)]]);
}
