//! `SELECT * EXCLUDE (...)` / `SELECT * REPLACE (...)` の統合テスト。
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

fn schema_names(session: &mut Session, sql: &str) -> Vec<String> {
    match session.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
        Prepared::Ready(q) => q.schema.iter().map(|f| f.name.clone()).collect(),
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
    }
}

fn i32(v: i32) -> Value {
    Value::I32(v)
}
fn f64(v: f64) -> Value {
    Value::F64(v)
}
fn vc(v: &str) -> Value {
    Value::Bytes(v.as_bytes().to_vec())
}
fn b(v: bool) -> Value {
    Value::Bool(v)
}

/// `EXCLUDE` は列挙結果から名前だけで除く。
/// duckdb: SELECT * EXCLUDE (score, big, d) FROM 'basic.parquet' WHERE id < 4 ORDER BY id
#[test]
fn exclude_drops_named_columns() {
    let mut s = session_with_basic();
    assert_eq!(
        schema_names(&mut s, "SELECT * EXCLUDE (score, big, d) FROM t"),
        vec!["id", "name", "flag"]
    );
    let rows = run(&mut s, "SELECT * EXCLUDE (score, big, d) FROM t WHERE id < 4 ORDER BY id");
    assert_eq!(
        rows,
        vec![
            vec![i32(0), vc("name_0"), b(true)],
            vec![i32(1), vc("name_1"), b(false)],
            vec![i32(2), vc("name_2"), b(false)],
            vec![i32(3), vc("name_3"), b(true)],
        ]
    );
}

/// `REPLACE` は列名・位置を変えずに値だけ差し替える。
/// duckdb: SELECT * REPLACE (score * 2 AS score) FROM 'basic.parquet' WHERE id < 4 ORDER BY id
#[test]
fn replace_substitutes_value_keeping_column_name_and_position() {
    let mut s = session_with_basic();
    assert_eq!(
        schema_names(&mut s, "SELECT * REPLACE (score * 2 AS score) FROM t"),
        vec!["id", "name", "score", "flag", "big", "d"]
    );
    let rows =
        run(&mut s, "SELECT id, name, score FROM (SELECT * REPLACE (score * 2 AS score) FROM t) WHERE id < 4 ORDER BY id");
    assert_eq!(
        rows,
        vec![
            vec![i32(0), vc("name_0"), f64(0.0)],
            vec![i32(1), vc("name_1"), f64(3.0)],
            vec![i32(2), vc("name_2"), f64(6.0)],
            vec![i32(3), vc("name_3"), f64(9.0)],
        ]
    );
}

/// `EXCLUDE` と `REPLACE` は同じ `*` に両方効かせられる（EXCLUDE が先）。
/// duckdb: SELECT * EXCLUDE (name, big, d) REPLACE (score * 2 AS score) FROM 'basic.parquet' WHERE id < 4 ORDER BY id
#[test]
fn exclude_and_replace_combine() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT * EXCLUDE (name, big, d) REPLACE (score * 2 AS score) FROM t WHERE id < 4 ORDER BY id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i32(0), f64(0.0), b(true)],
            vec![i32(1), f64(3.0), b(false)],
            vec![i32(2), f64(6.0), b(false)],
            vec![i32(3), f64(9.0), b(true)],
        ]
    );
}

/// `t.* EXCLUDE (...)` のように修飾子付きの `*` にも効く。
/// duckdb: SELECT t.* EXCLUDE (name, big, d) FROM 'basic.parquet' t WHERE id < 4 ORDER BY id
#[test]
fn qualified_star_supports_exclude() {
    let mut s = session_with_basic();
    let rows = run(&mut s, "SELECT t.* EXCLUDE (name, big, d) FROM t WHERE id < 4 ORDER BY id");
    assert_eq!(
        rows,
        vec![
            vec![i32(0), f64(0.0), b(true)],
            vec![i32(1), f64(1.5), b(false)],
            vec![i32(2), f64(3.0), b(false)],
            vec![i32(3), f64(4.5), b(true)],
        ]
    );
}

/// 括弧を省略した 1 個だけの形も動く（`duckdb` と同じ挙動）。
#[test]
fn exclude_and_replace_allow_bare_single_item() {
    let mut s = session_with_basic();
    assert_eq!(
        schema_names(&mut s, "SELECT * EXCLUDE score FROM t"),
        vec!["id", "name", "flag", "big", "d"]
    );
    let rows = run(&mut s, "SELECT id, name FROM (SELECT * REPLACE 99 AS id FROM t WHERE id = 1)");
    assert_eq!(rows, vec![vec![i32(99), vc("name_1")]]);
}

/// 存在しない列を EXCLUDE/REPLACE に書くと束縛時に拒否される
/// （`duckdb`: "Column ... not found"）。
#[test]
fn exclude_of_unknown_column_is_rejected() {
    let mut s = session_with_basic();
    let err = s.prepare("SELECT * EXCLUDE (nope) FROM t", &[]);
    assert_eq!(code_of(err), Some(Code::ColumnNotFound));
}

#[test]
fn replace_of_unknown_column_is_rejected() {
    let mut s = session_with_basic();
    let err = s.prepare("SELECT * REPLACE (1 AS nope) FROM t", &[]);
    assert_eq!(code_of(err), Some(Code::ColumnNotFound));
}

/// 集約後は `*` を展開できない（元の行が残っていない）のは EXCLUDE/REPLACE
/// を付けても変わらない。
#[test]
fn exclude_after_aggregation_is_still_rejected() {
    let mut s = session_with_basic();
    let err = s.prepare("SELECT * EXCLUDE (id) FROM t GROUP BY flag", &[]);
    assert_eq!(code_of(err), Some(Code::NotGrouped));
}

// --- 重複指定・境界値 -----------------------------------------------------------

/// `EXCLUDE`/`REPLACE` のリストに同じ列名が 2 回現れると `duckdb` は
/// "Duplicate entry ... in EXCLUDE/REPLACE list" として拒否する
/// (`duckdb -c "SELECT * EXCLUDE (id, id) FROM ..."` で確認済み)。
#[test]
fn exclude_rejects_a_duplicate_column_name() {
    let mut s = session_with_basic();
    let err = s.prepare("SELECT * EXCLUDE (id, id) FROM t", &[]);
    assert_eq!(code_of(err), Some(Code::SyntaxError));
}

#[test]
fn replace_rejects_a_duplicate_target_column() {
    let mut s = session_with_basic();
    let err = s.prepare("SELECT * REPLACE (1 AS id, 2 AS id) FROM t", &[]);
    assert_eq!(code_of(err), Some(Code::SyntaxError));
}

/// 全列を `EXCLUDE` すると選択リストが空になり、`duckdb` も
/// "SELECT list is empty after resolving * expressions!" として拒否する。
#[test]
fn excluding_every_column_is_rejected() {
    let mut s = session_with_basic();
    let err = s.prepare("SELECT * EXCLUDE (id, name, score, flag, big, d) FROM t", &[]);
    assert!(code_of(err).is_some());
}

/// 列名は大文字小文字を区別しない（このエンジンの通常の列参照と同じ規則、
/// `duckdb` も同様）。
#[test]
fn exclude_column_name_matching_is_case_insensitive() {
    let mut s = session_with_basic();
    assert_eq!(
        schema_names(&mut s, "SELECT * EXCLUDE (ID) FROM t"),
        vec!["name", "score", "flag", "big", "d"]
    );
}

// --- JOIN との相互作用 ----------------------------------------------------------

/// 修飾された `t.* EXCLUDE (...)` は、その表由来の列だけに効く
/// （`JOIN` の相手側の同名列は残る）。
/// duckdb: SELECT a.* EXCLUDE (id) FROM 'basic.parquet' a JOIN 'basic.parquet' b
///         ON a.id = b.id LIMIT 1
#[test]
fn qualified_star_exclude_only_affects_its_own_table_in_a_join() {
    let mut s = session_with_basic();
    let rows =
        run(&mut s, "SELECT a.* EXCLUDE (id) FROM t a JOIN t b ON a.id = b.id WHERE a.id = 0");
    assert_eq!(
        rows,
        vec![vec![vc("name_0"), f64(0.0), b(true), Value::Null, i64_ts(1704067200000000)]]
    );
}

/// 修飾されない `* EXCLUDE (...)` は両方の表から同名列を除く
/// （`duckdb` の実測どおり: 曖昧な列参照と同じ扱いで両側に効く）。
/// duckdb: SELECT * EXCLUDE (id) FROM 'basic.parquet' a JOIN 'basic.parquet' b
///         ON a.id = b.id LIMIT 1
#[test]
fn unqualified_star_exclude_affects_both_sides_of_a_join() {
    let mut s = session_with_basic();
    let rows = run(&mut s, "SELECT * EXCLUDE (id) FROM t a JOIN t b ON a.id = b.id WHERE a.id = 0");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].len(), 10, "両側から id を除いた残り 5 列 x 2 表");
}

fn i64_ts(v: i64) -> Value {
    Value::I64(v)
}

// --- 集約の入力として使う --------------------------------------------------------

/// `EXCLUDE`/`REPLACE` を含む `SELECT *` をサブクエリにして、その結果を
/// さらに集約する（新機能同士ではなく既存の集約パイプラインとの組み合わせ）。
#[test]
fn exclude_and_replace_result_can_feed_an_aggregate() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT flag, sum(score) FROM (SELECT * EXCLUDE (id, name, big, d) FROM t) \
         GROUP BY flag ORDER BY flag",
    );
    assert_eq!(rows, vec![vec![b(false), f64(499000.5)], vec![b(true), f64(250249.5)]]);
    // REPLACE は値だけ差し替わる（列名・位置は変わらない）ので、集約結果も
    // 差し替えた値どおりに 2 倍になる。
    let rows2 = run(
        &mut s,
        "SELECT sum(score) FROM (SELECT * REPLACE (score * 2 AS score) FROM t WHERE id < 4)",
    );
    assert_eq!(rows2, vec![vec![f64(18.0)]]);
}
