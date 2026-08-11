//! `PIVOT`/`UNPIVOT` の統合テスト。
//!
//! 期待値はすべて `duckdb -c "..."` の実際の出力と突き合わせて決めている
//! （`tests/data/pivot.parquet`/`pivot_small.parquet` は DuckDB が書いた実
//! ファイル。生成手順は `scripts/gen-testdata.sh` 参照）。
//! `ddl`/`dml` フィーチャは要らない（読み取り専用の Parquet だけで足りる）ので、
//! 既定フィーチャの `cargo test` でも必ず走る。

use ahiru_core::error::{code_of, Code};
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn data(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn session_with(file: &str) -> Session {
    let mut s = Session::new();
    s.register_bytes("t", data(file)).unwrap();
    s
}

/// `sql` を実行し、結果を `Vec<Vec<Value>>` として取り出す。
/// テストで使うファイルはメモリ上にまるごと乗るので `NeedIo`/`NeedCodec` は
/// 出ない。
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

fn i32(v: i32) -> Value {
    Value::I32(v)
}
fn i64(v: i64) -> Value {
    Value::I64(v)
}
fn i128(v: i128) -> Value {
    Value::I128(v)
}
fn s(v: &str) -> Value {
    Value::Bytes(v.as_bytes().to_vec())
}
const NULL: Value = Value::Null;

// --- PIVOT -------------------------------------------------------------------

/// `PIVOT t ON category IN (...) USING sum(amount) GROUP BY region`:
/// 明示 `GROUP BY` + 明示 `IN` の基本形。
/// duckdb: PIVOT 'pivot.parquet' ON category IN ('a','b','c') USING sum(amount)
///         GROUP BY region ORDER BY region;
#[test]
fn pivot_explicit_group_by_and_in_list() {
    let mut sess = session_with("pivot.parquet");
    let rows = run(
        &mut sess,
        "PIVOT t ON category IN ('a', 'b', 'c') USING sum(amount) GROUP BY region \
         ORDER BY region",
    );
    assert_eq!(
        rows,
        vec![
            vec![s("east"), i128(1500), i128(1700), i128(1300)],
            vec![s("north"), i128(1200), i128(1400), i128(1600)],
            vec![s("south"), i128(1650), i128(1250), i128(1450)],
            vec![s("west"), i128(1350), i128(1550), i128(1750)],
        ]
    );
}

/// `GROUP BY` 省略時は「`ON`/`USING` が参照する列以外の全列」が既定の
/// グルーピング対象になる（DuckDB と同じ規則）。
/// duckdb: PIVOT 'pivot_small.parquet' ON category IN ('a','b','c') USING sum(amount)
///         ORDER BY region;
#[test]
fn pivot_default_group_by_uses_all_other_columns() {
    let mut sess = session_with("pivot_small.parquet");
    let rows =
        run(&mut sess, "PIVOT t ON category IN ('a', 'b', 'c') USING sum(amount) ORDER BY region");
    assert_eq!(
        rows,
        vec![
            vec![s("east"), i128(10), i128(20), NULL],
            vec![s("west"), i128(30), i128(40), i128(5)],
        ]
    );
}

/// `USING` 省略時は DuckDB と同じく既定で `count(*)`。
/// duckdb: PIVOT 'pivot_small.parquet' ON category IN ('a','b','c') GROUP BY region
///         ORDER BY region;
#[test]
fn pivot_without_using_defaults_to_count_star() {
    let mut sess = session_with("pivot_small.parquet");
    let rows =
        run(&mut sess, "PIVOT t ON category IN ('a', 'b', 'c') GROUP BY region ORDER BY region");
    assert_eq!(
        rows,
        vec![vec![s("east"), i64(1), i64(1), i64(0)], vec![s("west"), i64(1), i64(1), i64(1)],]
    );
}

/// `IN (値 AS 別名, ...)`: 別名を指定すれば列名はそれになる（値の文字列化は
/// 使わない）。
/// duckdb: PIVOT 'pivot_small.parquet' ON category IN ('a' AS alpha, 'b' AS beta)
///         USING sum(amount) GROUP BY region ORDER BY region;
#[test]
fn pivot_in_list_aliases_become_column_names() {
    let mut sess = session_with("pivot_small.parquet");
    let rows = run(
        &mut sess,
        "PIVOT t ON category IN ('a' AS alpha, 'b' AS beta) USING sum(amount) \
         GROUP BY region ORDER BY region",
    );
    assert_eq!(
        rows,
        vec![vec![s("east"), i128(10), i128(20)], vec![s("west"), i128(30), i128(40)],]
    );
}

/// `ON` は裸の列だけでなく任意の式でよい。
/// duckdb: PIVOT 'pivot.parquet' ON id % 2 IN (0, 1) USING sum(amount) GROUP BY region
///         ORDER BY region;
#[test]
fn pivot_on_accepts_arbitrary_expression() {
    let mut sess = session_with("pivot.parquet");
    let rows = run(
        &mut sess,
        "PIVOT t ON id % 2 IN (0, 1) USING sum(amount) GROUP BY region ORDER BY region",
    );
    assert_eq!(
        rows,
        vec![
            vec![s("east"), i128(4500), NULL],
            vec![s("north"), i128(4200), NULL],
            vec![s("south"), NULL, i128(4350)],
            vec![s("west"), NULL, i128(4650)],
        ]
    );
}

/// `PIVOT` は末尾の `ORDER BY`/`LIMIT`/`OFFSET` を受け付ける（DuckDB と同じ）。
#[test]
fn pivot_supports_trailing_order_by_limit_offset() {
    let mut sess = session_with("pivot.parquet");
    let rows = run(
        &mut sess,
        "PIVOT t ON category IN ('a', 'b', 'c') USING sum(amount) GROUP BY region \
         ORDER BY region LIMIT 2 OFFSET 1",
    );
    assert_eq!(
        rows,
        vec![
            vec![s("north"), i128(1200), i128(1400), i128(1600)],
            vec![s("south"), i128(1650), i128(1250), i128(1450)],
        ]
    );
}

/// 値の自動検出（`IN` 省略）は、束縛時点では対象列の実データを読めず
/// （スキーマ解決までしかしていない）DISTINCT が取れないため非対応。
/// 明確に `UnsupportedFeature` を返すことを確認する。
#[test]
fn pivot_without_in_list_is_unsupported() {
    let mut sess = session_with("pivot_small.parquet");
    let err = sess.prepare("PIVOT t ON category USING sum(amount) GROUP BY region", &[]);
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// 複数集約関数（`USING sum(a), avg(a)`）は列名決定に式の文字列化が要り
/// 非対応。
#[test]
fn pivot_multiple_using_aggregates_is_unsupported() {
    let mut sess = session_with("pivot_small.parquet");
    let err = sess.prepare(
        "PIVOT t ON category IN ('a', 'b') USING sum(amount), avg(amount) GROUP BY region",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

// --- UNPIVOT -----------------------------------------------------------------

/// `UNPIVOT t ON col1, col2, ... INTO NAME .. VALUE ..`: 複数対象列を
/// 「列名の列 + 値の列」の2列に畳み込む。対象以外の列（id/region/category）は
/// そのまま素通しする。
/// duckdb: SELECT * FROM (UNPIVOT 'pivot.parquet' ON q1, q2, q3, q4
///         INTO NAME quarter VALUE amt) WHERE id < 2 ORDER BY id, quarter;
#[test]
fn unpivot_basic_folds_columns_into_name_value_pairs() {
    let mut sess = session_with("pivot.parquet");
    let rows = run(
        &mut sess,
        "UNPIVOT t ON q1, q2, q3, q4 INTO NAME quarter VALUE amt \
         ORDER BY id, quarter LIMIT 8",
    );
    assert_eq!(
        rows,
        vec![
            vec![i32(0), s("north"), s("a"), i32(0), s("q1"), i32(0)],
            vec![i32(0), s("north"), s("a"), i32(0), s("q2"), i32(0)],
            vec![i32(0), s("north"), s("a"), i32(0), s("q3"), i32(0)],
            vec![i32(0), s("north"), s("a"), i32(0), s("q4"), i32(0)],
            vec![i32(1), s("south"), s("b"), i32(10), s("q1"), i32(1)],
            vec![i32(1), s("south"), s("b"), i32(10), s("q2"), i32(2)],
            vec![i32(1), s("south"), s("b"), i32(10), s("q3"), i32(3)],
            vec![i32(1), s("south"), s("b"), i32(10), s("q4"), i32(4)],
        ]
    );
}

/// 行数は「元の行数 × 対象列数」になる（q1..q4 の 4 列 × 60 行）。
#[test]
fn unpivot_row_count_multiplies_by_target_column_count() {
    let mut sess = session_with("pivot.parquet");
    let rows = run(&mut sess, "UNPIVOT t ON q1, q2, q3, q4 INTO NAME quarter VALUE amt");
    assert_eq!(rows.len(), 60 * 4);
}

/// `INTO NAME .. VALUE ..` を省略すると DuckDB と同じく `name`/`value` が
/// 既定の列名になる。
/// duckdb: UNPIVOT 'pivot_small.parquet' ON amount ORDER BY region, category;
#[test]
fn unpivot_default_name_and_value_columns() {
    let mut sess = session_with("pivot_small.parquet");
    let rows = run(&mut sess, "UNPIVOT t ON amount ORDER BY region, category");
    assert_eq!(
        rows,
        vec![
            vec![s("east"), s("a"), s("amount"), i32(10)],
            vec![s("east"), s("b"), s("amount"), i32(20)],
            vec![s("west"), s("a"), s("amount"), i32(30)],
            vec![s("west"), s("b"), s("amount"), i32(40)],
            vec![s("west"), s("c"), s("amount"), i32(5)],
        ]
    );
}

/// `UNPIVOT` も末尾の `ORDER BY`/`LIMIT`/`OFFSET` を受け付ける。
#[test]
fn unpivot_supports_trailing_order_by_limit_offset() {
    let mut sess = session_with("pivot_small.parquet");
    let rows = run(&mut sess, "UNPIVOT t ON amount ORDER BY region, category LIMIT 2");
    assert_eq!(
        rows,
        vec![
            vec![s("east"), s("a"), s("amount"), i32(10)],
            vec![s("east"), s("b"), s("amount"), i32(20)],
        ]
    );
}

/// 対象列は修飾子なしの裸の列参照のみ。式は非対応。
#[test]
fn unpivot_target_must_be_a_bare_column_reference() {
    let mut sess = session_with("pivot.parquet");
    let err = sess.prepare("UNPIVOT t ON q1 + q2 INTO NAME k VALUE v", &[]);
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// 対象列の型が非互換（暗黙変換できない）なら、通常の `UNION ALL` と同じ
/// 型エラーになる（DuckDB も同じ状況で "an explicit cast is required" と拒否
/// する。エラーコードの意味は通常の集合演算と揃える）。
#[test]
fn unpivot_incompatible_column_types_is_type_mismatch() {
    let mut sess = session_with("pivot.parquet");
    let err = sess.prepare("UNPIVOT t ON region, amount INTO NAME k VALUE v", &[]);
    assert_eq!(code_of(err), Some(Code::TypeMismatch));
}

// --- 発見した不具合の回帰テスト -----------------------------------------------

/// `IN (...)` に同じ値を 2 回書くと、別名の有無に関わらず `duckdb` は
/// "The value ... was specified multiple times in the IN clause" として
/// 拒否する（`duckdb -c "PIVOT ... ON category IN ('a','a') ..."` で確認済み）。
///
/// 修正前の不具合: このチェックが無く、`PIVOT t ON category IN ('a', 'a')
/// USING sum(amount) GROUP BY region` が黙って通ってしまい、同じ `FILTER`
/// 条件を持つ列が `"a"` という同じ名前で 2 本できていた（`plan::bind::
/// desugar_pivot` に重複検出を追加して修正）。
#[test]
fn pivot_rejects_duplicate_values_in_the_in_list() {
    let mut sess = session_with("pivot_small.parquet");
    let err =
        sess.prepare("PIVOT t ON category IN ('a', 'a') USING sum(amount) GROUP BY region", &[]);
    assert_eq!(code_of(err), Some(Code::SyntaxError));
    // 別名を付けていても、元の値が重複していれば同じく拒否される
    // （`duckdb` は別名の衝突自体は `_1` サフィックスで自動リネームするだけ
    // だが、値そのものの重複は許さない）。
    let err = sess.prepare(
        "PIVOT t ON category IN ('a' AS x, 'a' AS y) USING sum(amount) GROUP BY region",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::SyntaxError));
}

/// 値が異なれば問題なく通る（重複チェックが誤爆していないことの確認）。
#[test]
fn pivot_distinct_values_in_the_in_list_are_fine() {
    let mut sess = session_with("pivot_small.parquet");
    let rows = run(
        &mut sess,
        "PIVOT t ON category IN ('a', 'b') USING sum(amount) GROUP BY region ORDER BY region",
    );
    assert_eq!(
        rows,
        vec![vec![s("east"), i128(10), i128(20)], vec![s("west"), i128(30), i128(40)],]
    );
}

/// `PIVOT`/`UNPIVOT` はこのエンジンでは文の先頭でしか展開されない糖衣構文
/// なので、派生表（`FROM (PIVOT ...)`）や CTE 本体、集合演算の項としては
/// 使えない（`plan::bind::desugar_pivot`/`desugar_unpivot` が `Session::
/// prepare` の入り口で 1 回だけ展開する設計。`session.rs` 参照）。`duckdb`
/// はこれを許すが、対応範囲外。
///
/// 修正前の不具合: このケースは `sql::parser::select_body` が `SELECT` を
/// 期待するところまで読み進んでから初めて `UnexpectedToken` になっており、
/// 「`PIVOT` 自体は書けるのになぜ subquery だと構文エラーになるのか」が
/// 分かりにくかった。`select_body` の先頭で `PIVOT`/`UNPIVOT` を検出し、
/// 意味の分かる `UnsupportedFeature` を返すように修正した。
#[test]
fn pivot_as_a_derived_table_is_a_clear_unsupported_error_not_a_confusing_syntax_error() {
    let mut sess = session_with("pivot.parquet");
    let err = sess.prepare(
        "SELECT * FROM (PIVOT t ON category IN ('a', 'b', 'c') USING sum(amount) \
         GROUP BY region) WHERE a > 1300",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

#[test]
fn unpivot_as_a_cte_body_is_a_clear_unsupported_error() {
    let mut sess = session_with("pivot.parquet");
    let err = sess.prepare(
        "WITH u AS (UNPIVOT t ON q1, q2, q3, q4 INTO NAME quarter VALUE amt) \
         SELECT quarter, sum(amt) FROM u GROUP BY quarter",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// 一方、top-level の `PIVOT`/`UNPIVOT` 自体はリグレッションなく今までどおり
/// 動く。
#[test]
fn top_level_pivot_is_unaffected_by_the_derived_table_check() {
    let mut sess = session_with("pivot.parquet");
    let rows = run(
        &mut sess,
        "PIVOT t ON category IN ('a', 'b', 'c') USING sum(amount) GROUP BY region \
         ORDER BY region",
    );
    assert_eq!(rows.len(), 4);
}

/// `PIVOT` の `FROM` は表名だけでなく、`JOIN` を含む任意の派生表でもよい
/// （`desugar_pivot` は `from` をそのまま `SelectStmt::from` へ渡すだけで、
/// `UNPIVOT` のように複製する必要が無いため制約が無い）。
#[test]
fn pivot_from_accepts_a_derived_table_containing_a_join() {
    let mut sess = session_with("pivot.parquet");
    let rows = run(
        &mut sess,
        "PIVOT (SELECT t.* FROM t JOIN t AS t2 ON t.id = t2.id) \
         ON category IN ('a', 'b') USING sum(amount) GROUP BY region ORDER BY region",
    );
    assert!(!rows.is_empty());
}

/// `UNPIVOT` は対象列 1 本ごとに `from` を複製する必要があり（`clone_from_item`
/// 参照）、`JOIN`/`Subquery` は複製できないプランを持ちうるため明示的に
/// 拒否する。クラッシュせずクリーンなエラーになることを確認する。
#[test]
fn unpivot_from_a_join_is_rejected_cleanly() {
    let mut sess = session_with("pivot.parquet");
    let err =
        sess.prepare("UNPIVOT t JOIN t AS t2 ON t.id = t2.id ON q1, q2 INTO NAME k VALUE v", &[]);
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}
