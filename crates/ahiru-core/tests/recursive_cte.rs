//! `WITH RECURSIVE`（再帰 CTE）の統合テスト。
//!
//! このエンジンは `SELECT 1`（FROM 無し）を v1 の対象外としている
//! （`plan::bind::bind_select_in` 参照）ので、リテラルだけのアンカーは
//! `dual`（1 行だけの CSV バイト列で作った、Oracle の `DUAL` 相当のダミー
//! 表）を経由する。`csv` フィーチャは既定で有効なので、`ddl`/`dml` フィーチャ
//! を要らず `cargo test --workspace` がそのまま拾う。期待値はすべて
//! `duckdb -c "..."` の実際の出力と突き合わせて決めている。

use ahiru_core::error::{code_of, Code};
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;
use ahiru_core::FormatKind;

/// `dual` テーブル（1 行 1 列、値は使わない）を登録したセッション。
/// リテラルだけのアンカー（`SELECT 0, 0, 1 FROM dual` のように）を FROM 句
/// 付きにするためだけに使う。
fn session_with_dual() -> Session {
    let mut s = Session::new();
    s.register_bytes_as("dual", b"x\n1\n".to_vec(), FormatKind::Csv).unwrap();
    s
}

/// `nodes(id, parent_id, name)` を 4 行のツリー構造（`root` の下に
/// `child1`/`child2`、`child1` の下に `grandchild`）で登録したセッション。
/// CSV の型推定で `id`/`parent_id` は `BIGINT` になる（`format::csv` の
/// 整数推定規則）。
fn session_with_nodes() -> Session {
    let mut s = Session::new();
    let csv = b"id,parent_id,name\n1,,root\n2,1,child1\n3,1,child2\n4,2,grandchild\n".to_vec();
    s.register_bytes_as("nodes", csv, FormatKind::Csv).unwrap();
    s
}

/// `sql` を実行し、結果を `Vec<Vec<Value>>` として取り出す。
/// バイト列に丸ごと乗っているデータしか読まないので `NeedIo`/`NeedCodec` は
/// 絶対に起きない。
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
fn s(v: &str) -> Value {
    Value::Bytes(v.as_bytes().to_vec())
}
const NULL: Value = Value::Null;

// --- 数列生成（フィボナッチ） -------------------------------------------------

/// duckdb:
/// ```sql
/// WITH RECURSIVE fib(n, a, b) AS (
///     SELECT 0, 0, 1
///     UNION ALL
///     SELECT n+1, b, a+b FROM fib WHERE n < 10
/// )
/// SELECT * FROM fib;
/// ```
/// 0,0,1 / 1,1,1 / 2,1,2 / 3,2,3 / 4,3,5 / 5,5,8 / 6,8,13 / 7,13,21 /
/// 8,21,34 / 9,34,55 / 10,55,89 の 11 行。
#[test]
fn fibonacci_union_all() {
    let mut db = session_with_dual();
    let rows = run(
        &mut db,
        "WITH RECURSIVE fib(n, a, b) AS ( \
           SELECT 0, 0, 1 FROM dual \
           UNION ALL \
           SELECT n+1, b, a+b FROM fib WHERE n < 10 \
         ) \
         SELECT * FROM fib",
    );
    assert_eq!(
        rows,
        vec![
            vec![i32(0), i32(0), i32(1)],
            vec![i32(1), i32(1), i32(1)],
            vec![i32(2), i32(1), i32(2)],
            vec![i32(3), i32(2), i32(3)],
            vec![i32(4), i32(3), i32(5)],
            vec![i32(5), i32(5), i32(8)],
            vec![i32(6), i32(8), i32(13)],
            vec![i32(7), i32(13), i32(21)],
            vec![i32(8), i32(21), i32(34)],
            vec![i32(9), i32(34), i32(55)],
            vec![i32(10), i32(55), i32(89)],
        ]
    );
}

/// 単純な連番。`n < 5` を渡すたびに 1 行ずつ増える一番素朴な形。
/// duckdb: `WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM t
/// WHERE n < 5) SELECT * FROM t` → 1..5。
#[test]
fn simple_counter_union_all() {
    let mut db = session_with_dual();
    let rows = run(
        &mut db,
        "WITH RECURSIVE t(n) AS ( \
           SELECT 1 FROM dual UNION ALL SELECT n+1 FROM t WHERE n < 5 \
         ) SELECT * FROM t",
    );
    assert_eq!(rows, vec![vec![i32(1)], vec![i32(2)], vec![i32(3)], vec![i32(4)], vec![i32(5)]]);
}

// --- 階層データ（自己結合） ---------------------------------------------------

/// `nodes` は実テーブル（`session_with_nodes` 参照）、`tree` がそれと自分
/// 自身を JOIN して根から順にたどる。duckdb:
/// ```sql
/// WITH RECURSIVE tree AS (
///     SELECT id, parent_id, name FROM nodes WHERE parent_id IS NULL
///     UNION ALL
///     SELECT n.id, n.parent_id, n.name FROM nodes n JOIN tree t ON n.parent_id = t.id
/// )
/// SELECT * FROM tree ORDER BY id;
/// ```
#[test]
fn hierarchy_self_join() {
    let mut db = session_with_nodes();
    let rows = run(
        &mut db,
        "WITH RECURSIVE tree AS ( \
           SELECT id, parent_id, name FROM nodes WHERE parent_id IS NULL \
           UNION ALL \
           SELECT n.id, n.parent_id, n.name FROM nodes n JOIN tree t ON n.parent_id = t.id \
         ) \
         SELECT * FROM tree ORDER BY id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i64(1), NULL, s("root")],
            vec![i64(2), i64(1), s("child1")],
            vec![i64(3), i64(1), s("child2")],
            vec![i64(4), i64(2), s("grandchild")],
        ]
    );
}

// --- UNION（重複排除） ---------------------------------------------------------

/// `UNION`（`ALL` 無し）は全イテレーションを通して重複を除く。`n % 3 + 1` は
/// 1→2→3→1→2→3… と巡回するので、`UNION ALL` なら無限に続くが `UNION` なら
/// 3 回目で既出の行しか出さず不動点に達して止まる。
/// duckdb: `WITH RECURSIVE t(n) AS (SELECT 1 UNION SELECT (n % 3) + 1 FROM t)
/// SELECT * FROM t ORDER BY n` → 1,2,3。
#[test]
fn union_distinct_dedups_across_iterations_and_terminates() {
    let mut db = session_with_dual();
    let rows = run(
        &mut db,
        "WITH RECURSIVE t(n) AS ( \
           SELECT 1 FROM dual UNION SELECT (n % 3) + 1 FROM t \
         ) SELECT * FROM t ORDER BY n",
    );
    assert_eq!(rows, vec![vec![i32(1)], vec![i32(2)], vec![i32(3)]]);
}

// --- 列名指定 -----------------------------------------------------------------

/// `WITH RECURSIVE` の下では非再帰 CTE にも列名リストを付けられる。
#[test]
fn column_list_applies_to_non_recursive_member_too() {
    let mut db = session_with_dual();
    let rows = run(
        &mut db,
        "WITH RECURSIVE base(y) AS (SELECT 1 FROM dual), \
         t AS (SELECT y AS n FROM base UNION ALL SELECT n+1 FROM t WHERE n < 3) \
         SELECT * FROM t ORDER BY n",
    );
    assert_eq!(rows, vec![vec![i32(1)], vec![i32(2)], vec![i32(3)]]);
}

// --- 安全弁 --------------------------------------------------------------------

/// 停止条件を書き忘れた再帰 CTE（`WHERE` が無いので毎回 1 行ずつ無限に
/// 増える）は、パニックせず `RecursionLimitExceeded` で止まる。
#[test]
fn runaway_recursion_is_rejected_not_panicking() {
    let mut db = session_with_dual();
    let mut q = match db.prepare(
        "WITH RECURSIVE t(n) AS (SELECT 1 FROM dual UNION ALL SELECT n+1 FROM t) SELECT * FROM t",
        &[],
    ) {
        Ok(Prepared::Ready(q)) => q,
        Ok(Prepared::NeedIo(_)) => panic!("unexpected NeedIo"),
        Err(e) => {
            assert_eq!(e.code, Code::RecursionLimitExceeded);
            return;
        }
    };
    let last = loop {
        match db.step(&mut q) {
            Ok(QueryStep::Batch(_)) => {}
            Ok(QueryStep::Done) => panic!("runaway recursion must not terminate on its own"),
            Ok(QueryStep::NeedIo(_)) | Ok(QueryStep::NeedCodec(_)) => {
                panic!("unexpected NeedIo/NeedCodec")
            }
            Err(e) => break e.code,
        }
    };
    assert_eq!(last, Code::RecursionLimitExceeded);
}

/// 1行あたりの増分が一定ではなく、幾何級数的に膨張する再帰 CTE（毎回 10 倍）
/// は `MAX_RECURSIVE_ITERATIONS`（10万回）に達するよりずっと早く、作業集合の
/// バイト数上限（`MAX_WORKING_BYTES`）で `Oom` になるはず。
/// `RecursiveCte::process` はバッチ単位でこの上限を見ているので、
/// 最終的に天文学的な行数になる結合を最後まで実体化させずに途中で
/// 打ち切れることも合わせて確認する（テストが長時間かからないことがその証拠）。
#[test]
fn geometric_growth_hits_the_working_set_byte_limit_not_the_iteration_limit() {
    let mut db = session_with_dual();
    let ten = "(SELECT 0 AS k FROM dual UNION ALL SELECT 1 FROM dual UNION ALL SELECT 2 FROM dual \
               UNION ALL SELECT 3 FROM dual UNION ALL SELECT 4 FROM dual UNION ALL SELECT 5 FROM dual \
               UNION ALL SELECT 6 FROM dual UNION ALL SELECT 7 FROM dual UNION ALL SELECT 8 FROM dual \
               UNION ALL SELECT 9 FROM dual)";
    let sql = format!(
        "WITH RECURSIVE t(n) AS ( \
           SELECT 1 FROM dual \
           UNION ALL \
           SELECT t.n * 10 + m.k FROM t, {ten} AS m \
         ) SELECT count(*) FROM t"
    );
    let mut q = match db.prepare(&sql, &[]) {
        Ok(Prepared::Ready(q)) => q,
        Ok(Prepared::NeedIo(_)) => panic!("unexpected NeedIo"),
        Err(e) => {
            assert_eq!(e.code, Code::Oom);
            return;
        }
    };
    let last = loop {
        match db.step(&mut q) {
            Ok(QueryStep::Batch(_)) => {}
            Ok(QueryStep::Done) => panic!("幾何級数的な膨張はどこかで Oom になるべき"),
            Ok(QueryStep::NeedIo(_)) | Ok(QueryStep::NeedCodec(_)) => {
                panic!("unexpected NeedIo/NeedCodec")
            }
            Err(e) => break e.code,
        }
    };
    assert_eq!(last, Code::Oom, "反復回数上限より先にバイト数上限で止まるべき");
}

/// 2 つの独立した再帰 CTE を同じクエリ内で同時に使う。それぞれの
/// `WorkingTable` の差し替えが正しい方の CTE に対応しないと、値が
/// 混線したり無限ループになったりするはず。
#[test]
fn two_independent_recursive_ctes_in_the_same_query_do_not_cross_contaminate() {
    let mut db = session_with_dual();
    let rows = run(
        &mut db,
        "WITH RECURSIVE \
           a(n) AS (SELECT 1 FROM dual UNION ALL SELECT n+1 FROM a WHERE n < 3), \
           b(n) AS (SELECT 100 FROM dual UNION ALL SELECT n+1 FROM b WHERE n < 103) \
         SELECT a.n, b.n FROM a, b WHERE a.n = b.n - 99 ORDER BY a.n",
    );
    assert_eq!(rows, vec![vec![i32(1), i32(100)], vec![i32(2), i32(101)], vec![i32(3), i32(102)],]);
}

// --- 束縛時に明確に拒否するパターン -------------------------------------------

/// `WITH RECURSIVE` でも、自分自身を参照する CTE の本体が
/// `<anchor> UNION [ALL] <recursive_term>` の形になっていなければ拒否する
/// （例: 自己参照がアンカー側にある）。
#[test]
fn self_reference_in_anchor_is_rejected() {
    let mut db = Session::new();
    let err = db.prepare(
        "WITH RECURSIVE t(n) AS (SELECT n FROM t UNION ALL SELECT 1) SELECT * FROM t",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// 列名リストの列数が本体と合わなければ明確に拒否する（アンカーが 1 列しか
/// 出さないのに `t(a, b)` と 2 列指定している）。
#[test]
fn column_list_arity_mismatch_is_rejected() {
    let mut db = session_with_dual();
    let err = db.prepare(
        "WITH RECURSIVE t(a, b) AS ( \
           SELECT 1 FROM dual UNION ALL SELECT n+1 FROM t WHERE n < 3 \
         ) SELECT * FROM t",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::ColumnCountMismatch));
}
