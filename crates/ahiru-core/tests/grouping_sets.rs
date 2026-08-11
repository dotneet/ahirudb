//! `GROUP BY GROUPING SETS`/`ROLLUP`/`CUBE`/`GROUPING()` の統合テスト。
//!
//! 期待値はすべて `duckdb -c "SELECT ..."` の実際の出力と突き合わせて決めている
//! （`tests/data/basic.parquet` は DuckDB が書いた実ファイル）。
//! `ddl`/`dml` フィーチャは要らない（読み取り専用の Parquet だけで足りる）ので、
//! 既定フィーチャの `cargo test` でも必ず走る。

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

/// `sql` を実行し、結果を `Vec<Vec<Value>>` として取り出す。
/// `basic.parquet` はメモリ上にまるごと乗るので `NeedIo`/`NeedCodec` は出ない。
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
fn b(v: bool) -> Value {
    Value::Bool(v)
}
const NULL: Value = Value::Null;

/// GROUPING SETS/ROLLUP/CUBE を挟まない、素の GROUP BY が今まで通り動くこと
/// （リグレッション確認）。
#[test]
fn plain_group_by_is_unaffected() {
    let mut s = session_with_basic();
    let rows = run(&mut s, "SELECT flag, count(*) c FROM t GROUP BY flag ORDER BY flag");
    // duckdb: SELECT flag, count(*) c FROM 'basic.parquet' GROUP BY flag ORDER BY flag
    assert_eq!(rows, vec![vec![b(false), i64(666)], vec![b(true), i64(334)],]);
}

/// `GROUPING SETS ((flag), ())`: 単純な小計 + 総計。セットに無い列は NULL になる。
#[test]
fn grouping_sets_basic() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT flag, count(*) c, sum(id) s FROM t \
         GROUP BY GROUPING SETS ((flag), ()) ORDER BY flag",
    );
    assert_eq!(
        rows,
        vec![
            vec![b(false), i64(666), i128(332667)],
            vec![b(true), i64(334), i128(166833)],
            vec![NULL, i64(1000), i128(499500)],
        ]
    );
}

/// `ROLLUP (flag, id % 3)`: 階層的な部分集合
/// `(flag, id%3), (flag), ()` に展開される。
#[test]
fn rollup_expands_to_hierarchical_subsets() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT flag, id % 3 AS m, count(*) c FROM t \
         GROUP BY ROLLUP (flag, id % 3) ORDER BY 1, 2",
    );
    assert_eq!(
        rows,
        vec![
            vec![b(false), i32(1), i64(333)],
            vec![b(false), i32(2), i64(333)],
            vec![b(false), NULL, i64(666)],
            vec![b(true), i32(0), i64(334)],
            vec![b(true), NULL, i64(334)],
            vec![NULL, NULL, i64(1000)],
        ]
    );
}

/// `CUBE (flag, id % 3)`: 全部分集合（2^2 = 4 セット）に展開される。
#[test]
fn cube_expands_to_all_subsets() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT flag, id % 3 AS m, count(*) c FROM t \
         GROUP BY CUBE (flag, id % 3) ORDER BY 1, 2",
    );
    assert_eq!(
        rows,
        vec![
            vec![b(false), i32(1), i64(333)],
            vec![b(false), i32(2), i64(333)],
            vec![b(false), NULL, i64(666)],
            vec![b(true), i32(0), i64(334)],
            vec![b(true), NULL, i64(334)],
            vec![NULL, i32(0), i64(334)],
            vec![NULL, i32(1), i64(333)],
            vec![NULL, i32(2), i64(333)],
            vec![NULL, NULL, i64(1000)],
        ]
    );
}

/// `GROUPING()`/`GROUPING_ID()`: 列がそのセットで生きていれば 0、集約で潰されて
/// NULL になっていれば 1。複数引数は先頭を最上位ビットにしたビットマスク。
#[test]
fn grouping_function_reports_which_columns_were_rolled_up() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT flag, id % 3 AS m, count(*) c, \
                grouping(flag) gf, grouping(id % 3) gm, grouping(flag, id % 3) gid \
         FROM t GROUP BY CUBE (flag, id % 3) ORDER BY 1, 2",
    );
    assert_eq!(
        rows,
        vec![
            vec![b(false), i32(1), i64(333), i64(0), i64(0), i64(0)],
            vec![b(false), i32(2), i64(333), i64(0), i64(0), i64(0)],
            vec![b(false), NULL, i64(666), i64(0), i64(1), i64(1)],
            vec![b(true), i32(0), i64(334), i64(0), i64(0), i64(0)],
            vec![b(true), NULL, i64(334), i64(0), i64(1), i64(1)],
            vec![NULL, i32(0), i64(334), i64(1), i64(0), i64(2)],
            vec![NULL, i32(1), i64(333), i64(1), i64(0), i64(2)],
            vec![NULL, i32(2), i64(333), i64(1), i64(0), i64(2)],
            vec![NULL, NULL, i64(1000), i64(1), i64(1), i64(3)],
        ]
    );
    // `GROUPING_ID` は DuckDB でも `GROUPING` の別名（同じビットマスク意味論）。
    let rows2 = run(
        &mut s,
        "SELECT flag, id % 3 AS m, grouping_id(flag, id % 3) gid \
         FROM t GROUP BY CUBE (flag, id % 3) ORDER BY 1, 2",
    );
    let last = rows2.last().unwrap();
    assert_eq!(last, &vec![NULL, NULL, i64(3)]);
}

/// `HAVING` はグルーピングセットを UNION ALL で束ねた後の最終結果に対して効く。
#[test]
fn having_filters_after_all_grouping_sets_are_combined() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT flag, id % 3 AS m, count(*) c FROM t \
         GROUP BY GROUPING SETS ((flag, id % 3), (flag), ()) \
         HAVING count(*) > 400 ORDER BY 1, 2",
    );
    assert_eq!(rows, vec![vec![b(false), NULL, i64(666)], vec![NULL, NULL, i64(1000)],]);
}

/// `HAVING` の中で `GROUPING()` も使える。
#[test]
fn having_can_reference_grouping() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT flag, id % 3 AS m, count(*) c FROM t \
         GROUP BY GROUPING SETS ((flag, id % 3), (flag), ()) \
         HAVING grouping(flag) = 0 ORDER BY 1, 2",
    );
    // flag が生きている（潰れていない）行だけが残る = 総計行が落ちる。
    assert_eq!(
        rows,
        vec![
            vec![b(false), i32(1), i64(333)],
            vec![b(false), i32(2), i64(333)],
            vec![b(false), NULL, i64(666)],
            vec![b(true), i32(0), i64(334)],
            vec![b(true), NULL, i64(334)],
        ]
    );
}

/// `GROUPING()` の引数はグルーピング列でなければならない。
#[test]
fn grouping_of_a_non_grouped_column_is_rejected() {
    let mut s = session_with_basic();
    let err = s.prepare(
        "SELECT flag, count(*), grouping(id) FROM t GROUP BY GROUPING SETS ((flag), ())",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::NotGrouped));
}

/// `GROUPING()` は集約が無いクエリでは使えない。
#[test]
fn grouping_without_aggregation_is_rejected() {
    let mut s = session_with_basic();
    let err = s.prepare("SELECT grouping(id) FROM t", &[]);
    assert_eq!(code_of(err), Some(Code::NotAggregate));
}

/// `GROUPING SETS` は全セットの和集合を「グルーピング列」として扱う。
/// あるセットに無い列でも SELECT で裸参照でき、その行では NULL になる
/// （エラーにはならない）。
#[test]
fn columns_missing_from_a_set_are_still_selectable() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT flag, id % 3 AS m FROM t GROUP BY GROUPING SETS ((flag), (id % 3)) ORDER BY 1, 2",
    );
    // (flag) だけのセットでは m は必ず NULL、(id % 3) だけのセットでは flag は必ず NULL。
    assert!(rows.iter().any(|r| r[0] != NULL && r[1] == NULL));
    assert!(rows.iter().any(|r| r[0] == NULL && r[1] != NULL));
}
