//! `WINDOW name AS (...)` / `OVER name` の統合テスト。
//!
//! 期待値はすべて `duckdb -c "SELECT ..."` の実際の出力と突き合わせて決めている
//! （`tests/data/basic.parquet` は DuckDB が書いた実ファイル。列は
//! `id INTEGER, name VARCHAR, score DOUBLE, flag BOOLEAN, big BIGINT,
//! d TIMESTAMP`）。読み取り専用の Parquet だけで足りるので、既定フィーチャの
//! `cargo test` でも必ず走る。

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

fn i32(v: i32) -> Value {
    Value::I32(v)
}
fn i64(v: i64) -> Value {
    Value::I64(v)
}
fn f64(v: f64) -> Value {
    Value::F64(v)
}
fn b(v: bool) -> Value {
    Value::Bool(v)
}

/// 複数のウィンドウ関数が同じ名前付き定義を共有する。
/// duckdb:
/// SELECT id, flag, sum(score) OVER w AS s, avg(score) OVER w AS a
/// FROM 'basic.parquet' WHERE id < 6
/// WINDOW w AS (PARTITION BY flag ORDER BY id) ORDER BY id
#[test]
fn named_window_shared_by_multiple_calls() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT id, flag, sum(score) OVER w AS s, avg(score) OVER w AS a \
         FROM t WHERE id < 6 \
         WINDOW w AS (PARTITION BY flag ORDER BY id) ORDER BY id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i32(0), b(true), f64(0.0), f64(0.0)],
            vec![i32(1), b(false), f64(1.5), f64(1.5)],
            vec![i32(2), b(false), f64(4.5), f64(2.25)],
            vec![i32(3), b(true), f64(4.5), f64(2.25)],
            vec![i32(4), b(false), f64(10.5), f64(3.5)],
            vec![i32(5), b(false), f64(18.0), f64(4.5)],
        ]
    );
}

/// 名前付き参照（`OVER w`）とその場の指定（`OVER (...)`）は同じクエリで併用できる。
/// duckdb:
/// SELECT id, flag, row_number() OVER w AS rn, count(*) OVER () AS total
/// FROM 'basic.parquet' WHERE id < 6
/// WINDOW w AS (PARTITION BY flag ORDER BY id) ORDER BY id
#[test]
fn named_and_inline_window_can_be_mixed() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT id, flag, row_number() OVER w AS rn, count(*) OVER () AS total \
         FROM t WHERE id < 6 \
         WINDOW w AS (PARTITION BY flag ORDER BY id) ORDER BY id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i32(0), b(true), i64(1), i64(6)],
            vec![i32(1), b(false), i64(1), i64(6)],
            vec![i32(2), b(false), i64(2), i64(6)],
            vec![i32(3), b(true), i64(2), i64(6)],
            vec![i32(4), b(false), i64(3), i64(6)],
            vec![i32(5), b(false), i64(4), i64(6)],
        ]
    );
}

/// 複数の名前付きウィンドウを定義して使い分けられる。
#[test]
fn multiple_named_windows() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT id, rank() OVER w1 AS r1, row_number() OVER w2 AS r2 \
         FROM t WHERE id < 4 \
         WINDOW w1 AS (ORDER BY id), w2 AS (PARTITION BY flag ORDER BY id) \
         ORDER BY id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i32(0), i64(1), i64(1)],
            vec![i32(1), i64(2), i64(1)],
            vec![i32(2), i64(3), i64(2)],
            vec![i32(3), i64(4), i64(2)],
        ]
    );
}

/// 定義されていない名前を `OVER` で参照すると束縛時に拒否される
/// （`duckdb` は "window ... does not exist" として拒否する）。
#[test]
fn undefined_named_window_is_rejected() {
    let mut s = session_with_basic();
    let err = s.prepare("SELECT sum(score) OVER w FROM t", &[]);
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// `WINDOW` 句自体が無くても普通の `OVER (...)` は今までどおり動く
/// （リグレッション確認）。
#[test]
fn plain_over_clause_is_unaffected() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT id, sum(score) OVER (PARTITION BY flag ORDER BY id) AS s \
         FROM t WHERE id < 4 ORDER BY id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i32(0), f64(0.0)],
            vec![i32(1), f64(1.5)],
            vec![i32(2), f64(4.5)],
            vec![i32(3), f64(4.5)],
        ]
    );
}
