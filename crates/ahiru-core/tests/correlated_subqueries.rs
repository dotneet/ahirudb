//! 相関サブクエリ（correlated subqueries）の統合テスト。
//!
//! `customers`/`orders` をインメモリ表として作り（`ddl`/`dml` フィーチャ)、
//! 相関スカラサブクエリ・相関 `EXISTS`/`NOT EXISTS`・相関 `IN`/`NOT IN` を
//! 検証する。期待値はすべて `duckdb -c "SELECT ..."` の実際の出力と
//! 突き合わせて決めている。
//!
//! サポート範囲外のパターン（非等価相関・OR の中の相関・NULL の可能性が
//! ある相関 `NOT IN`・独自の GROUP BY を持つ相関集約サブクエリ・2 階層以上
//! ネストした相関）が `panic` せず明確なエラーで拒否されることも確認する。

#![cfg(feature = "dml")]

use ahiru_core::error::{code_of, Code};
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

/// `sql` を実行し、結果を `Vec<Vec<Value>>` として取り出す。
/// インメモリ表しか使わないので `NeedIo`/`NeedCodec` は絶対に起きない。
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
fn f64(v: f64) -> Value {
    Value::F64(v)
}
fn s(v: &str) -> Value {
    Value::Bytes(v.as_bytes().to_vec())
}
const NULL: Value = Value::Null;

/// customers(id, name, region) / orders(id, customer_id, amount, region) を
/// 作り、DuckDB で検証したのと同じ内容を入れる:
///
/// ```text
/// customers: (1,alice,east) (2,bob,west) (3,carol,east) (4,dave,NULL)
/// orders:    (10,1,100.0,east) (11,1,50.0,NULL) (12,2,200.0,west) (13,NULL,10.0,east)
/// ```
///
/// alice は 2 件（東, 相関先 NULL 込み）、bob は 1 件、carol/dave は 0 件。
fn session_with_customers_orders() -> Session {
    let mut s = Session::new();
    s.prepare("CREATE TABLE customers (id INT, name VARCHAR, region VARCHAR)", &[]).unwrap();
    s.prepare("CREATE TABLE orders (id INT, customer_id INT, amount DOUBLE, region VARCHAR)", &[])
        .unwrap();
    s.prepare(
        "INSERT INTO customers VALUES \
         (1, 'alice', 'east'), (2, 'bob', 'west'), (3, 'carol', 'east'), (4, 'dave', NULL)",
        &[],
    )
    .unwrap();
    s.prepare(
        "INSERT INTO orders VALUES \
         (10, 1, 100.0, 'east'), (11, 1, 50.0, NULL), (12, 2, 200.0, 'west'), \
         (13, NULL, 10.0, 'east')",
        &[],
    )
    .unwrap();
    s
}

// --- 相関スカラサブクエリ -----------------------------------------------------

/// 集約を伴わない相関スカラサブクエリ: 相関キーごとに「先頭の 1 行」を採る。
/// duckdb: `SELECT c.id, c.name, (SELECT o.amount FROM orders o WHERE
/// o.customer_id = c.id AND o.id = 10) FROM customers c ORDER BY c.id`
#[test]
fn correlated_scalar_subquery_picks_first_row_per_key() {
    let mut db = session_with_customers_orders();
    let rows = run(
        &mut db,
        "SELECT c.id, c.name, \
         (SELECT o.amount FROM orders o WHERE o.customer_id = c.id AND o.id = 10) \
         FROM customers c ORDER BY c.id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i32(1), s("alice"), f64(100.0)],
            vec![i32(2), s("bob"), NULL],
            vec![i32(3), s("carol"), NULL],
            vec![i32(4), s("dave"), NULL],
        ]
    );
}

/// 集約を伴う相関スカラサブクエリ（`max`）。マジックデコリレーション
/// （相関キーで GROUP BY してから LEFT JOIN）の基本形。
/// duckdb: `SELECT c.id, c.name, (SELECT max(o.amount) FROM orders o
/// WHERE o.customer_id = c.id) FROM customers c ORDER BY c.id`
#[test]
fn correlated_scalar_subquery_aggregate_max() {
    let mut db = session_with_customers_orders();
    let rows = run(
        &mut db,
        "SELECT c.id, c.name, (SELECT max(o.amount) FROM orders o WHERE o.customer_id = c.id) \
         FROM customers c ORDER BY c.id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i32(1), s("alice"), f64(100.0)],
            vec![i32(2), s("bob"), f64(200.0)],
            vec![i32(3), s("carol"), NULL],
            vec![i32(4), s("dave"), NULL],
        ]
    );
}

/// 集約を伴う相関スカラサブクエリ（`count`）は、一致する内側行が無い
/// ときに NULL ではなく 0 を返す（DuckDB で確認済み。素朴に GROUP BY へ
/// 相関キーを合流させただけだと「その組が無い」→ LEFT JOIN で NULL に
/// なってしまうので、COUNT 系だけは補正が要る）。
/// duckdb: `SELECT c.id, c.name, (SELECT count(*) FROM orders o WHERE
/// o.customer_id = c.id) FROM customers c ORDER BY c.id`
#[test]
fn correlated_scalar_subquery_count_defaults_to_zero_not_null() {
    let mut db = session_with_customers_orders();
    let rows = run(
        &mut db,
        "SELECT c.id, c.name, (SELECT count(*) FROM orders o WHERE o.customer_id = c.id) \
         FROM customers c ORDER BY c.id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i32(1), s("alice"), i64(2)],
            vec![i32(2), s("bob"), i64(1)],
            vec![i32(3), s("carol"), i64(0)],
            vec![i32(4), s("dave"), i64(0)],
        ]
    );
}

// --- 相関 EXISTS / NOT EXISTS -------------------------------------------------

/// 相関 `EXISTS` に非相関の追加条件（`o.amount > 60`）が混じっていても、
/// 相関等価述語だけを結合キーに取り出し、残りは内側の WHERE として
/// 普通に評価される。
/// duckdb: `SELECT c.id, c.name FROM customers c WHERE EXISTS
/// (SELECT 1 FROM orders o WHERE o.customer_id = c.id AND o.amount > 60)
/// ORDER BY c.id`
#[test]
fn correlated_exists_with_extra_local_predicate() {
    let mut db = session_with_customers_orders();
    let rows = run(
        &mut db,
        "SELECT c.id, c.name FROM customers c WHERE EXISTS \
         (SELECT 1 FROM orders o WHERE o.customer_id = c.id AND o.amount > 60) \
         ORDER BY c.id",
    );
    assert_eq!(rows, vec![vec![i32(1), s("alice")], vec![i32(2), s("bob")]]);
}

/// 相関 `NOT EXISTS`。
/// duckdb: `SELECT c.id, c.name FROM customers c WHERE NOT EXISTS
/// (SELECT 1 FROM orders o WHERE o.customer_id = c.id) ORDER BY c.id`
#[test]
fn correlated_not_exists() {
    let mut db = session_with_customers_orders();
    let rows = run(
        &mut db,
        "SELECT c.id, c.name FROM customers c WHERE NOT EXISTS \
         (SELECT 1 FROM orders o WHERE o.customer_id = c.id) ORDER BY c.id",
    );
    assert_eq!(rows, vec![vec![i32(3), s("carol")], vec![i32(4), s("dave")]]);
}

// --- 相関 IN / NOT IN ---------------------------------------------------------

/// 相関 `IN`。`IN` の対象列（`customer_id`）と相関キー（`region`）の
/// 複合キーによる半結合になる。
/// duckdb: `SELECT c.id, c.name FROM customers c WHERE c.id IN
/// (SELECT o.customer_id FROM orders o WHERE o.region = c.region) ORDER BY c.id`
#[test]
fn correlated_in() {
    let mut db = session_with_customers_orders();
    let rows = run(
        &mut db,
        "SELECT c.id, c.name FROM customers c WHERE c.id IN \
         (SELECT o.customer_id FROM orders o WHERE o.region = c.region) ORDER BY c.id",
    );
    assert_eq!(rows, vec![vec![i32(1), s("alice")], vec![i32(2), s("bob")]]);
}

/// 相関 `NOT IN` は、相関キーごとにスコープされた NULL 三値論理の判定が
/// 必要だが、既存の `AntiNullAware`（`NOT IN` の NULL 対応）は結合全体に
/// 対する判定しか持たない。対象列の NULL 可能性を束縛時に正確に判定する
/// 手段も無い（SELECT リストの出力列は常に `nullable = true` として扱われる）
/// ため、相関の有無に関わらず誤った結果を返しうる。曖昧な結果を返すくらい
/// なら常に明確に拒否する。
#[test]
fn correlated_not_in_is_always_rejected() {
    let mut db = session_with_customers_orders();
    let err = db.prepare(
        "SELECT c.id FROM customers c WHERE c.id NOT IN \
         (SELECT o.customer_id FROM orders o WHERE o.region = c.region \
          AND o.customer_id IS NOT NULL)",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

// --- サポート外パターンの明確な拒否 -------------------------------------------

/// 相関述語が非等価（`>`）だと結合キーに取り出せない。明確に拒否する。
#[test]
fn non_equality_correlation_is_rejected() {
    let mut db = session_with_customers_orders();
    let err = db.prepare(
        "SELECT c.id FROM customers c WHERE EXISTS \
         (SELECT 1 FROM orders o WHERE o.customer_id > c.id)",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// 相関参照が `OR` の中にあると、AND 分解では取り出せない。明確に拒否する。
#[test]
fn correlation_inside_or_is_rejected() {
    let mut db = session_with_customers_orders();
    let err = db.prepare(
        "SELECT c.id FROM customers c WHERE EXISTS \
         (SELECT 1 FROM orders o WHERE o.customer_id = c.id OR o.amount > 1000)",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// 独自の GROUP BY を持つ相関集約サブクエリは、相関キーをそのグルーピングに
/// 合流させる手段が無いため明確に拒否する（黙って相関を無視すると集約が
/// 外側の行をまたいで混ざり、誤った結果になってしまうため）。
#[test]
fn correlated_aggregate_with_own_group_by_is_rejected() {
    let mut db = session_with_customers_orders();
    let err = db.prepare(
        "SELECT c.id, (SELECT count(*) FROM orders o WHERE o.customer_id = c.id \
         GROUP BY o.region) FROM customers c",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// 2 階層以上ネストした相関（内側の内側が、間の階層を飛び越して一番外側の
/// 列を参照する）は 1 階層分の外側スコープしか伝播しないので、`panic` せず
/// エラーになる（列が見つからない、という形で失敗する）。
#[test]
fn two_level_correlation_skip_fails_cleanly() {
    let mut db = session_with_customers_orders();
    let err = db.prepare(
        "SELECT c.id FROM customers c WHERE EXISTS ( \
           SELECT 1 FROM orders o1 WHERE EXISTS ( \
             SELECT 1 FROM orders o2 WHERE o2.id = c.id \
           ) \
         )",
        &[],
    );
    // パニックせず、何らかの明確なエラーコードで失敗すること。
    assert!(err.is_err());
}

// --- 非相関サブクエリの回帰確認 -----------------------------------------------

/// 相関のない（外側スコープを参照しない）サブクエリは従来どおり動く。
#[test]
fn uncorrelated_subqueries_are_unaffected() {
    let mut db = session_with_customers_orders();
    let rows = run(&mut db, "SELECT (SELECT max(amount) FROM orders) FROM customers WHERE id = 1");
    assert_eq!(rows, vec![vec![f64(200.0)]]);

    let rows =
        run(&mut db, "SELECT id FROM customers WHERE EXISTS (SELECT 1 FROM orders) ORDER BY id");
    assert_eq!(rows.len(), 4);

    let rows = run(
        &mut db,
        "SELECT id FROM customers WHERE id IN (SELECT customer_id FROM orders) ORDER BY id",
    );
    assert_eq!(rows, vec![vec![i32(1)], vec![i32(2)]]);
}
