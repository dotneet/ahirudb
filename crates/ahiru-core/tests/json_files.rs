//! `format::json`（`read_json`/`read_json_auto` 相当）の統合テスト。
//!
//! 単体テストは `format::json` 側で自前のバイト列に対して行っているので、
//! ここでは「実際に `Session`/`Catalog` 経由の SQL パイプラインとして動くか」
//! と「同じ内容を持つ他フォーマット（Parquet/JSONL）と同じ結果を返すか」を
//! 確認する。期待値は `duckdb -c "SELECT ..."` の出力と突き合わせて決めている。

use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn data(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// クエリを最後まで実行し、行を集めて返す。全データがメモリ上にあるので
/// `NeedIo`/`NeedCodec` は起こらない前提。
fn run_all(sql: &str, s: &mut Session) -> Vec<Vec<Value>> {
    let mut q = match s.prepare(sql, &[]).unwrap() {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("メモリ上のデータで NeedIo は出ないはず"),
    };
    let mut rows = Vec::new();
    loop {
        match s.step(&mut q).unwrap() {
            QueryStep::Batch(mut b) => {
                b.materialize();
                for r in 0..b.num_rows() {
                    rows.push(b.cols.iter().map(|c| c.value_at(r)).collect());
                }
            }
            QueryStep::NeedIo(_) | QueryStep::NeedCodec(_) => panic!("unexpected suspend"),
            QueryStep::Done => break,
        }
    }
    rows
}

fn s(v: &str) -> Value {
    Value::Bytes(v.as_bytes().to_vec())
}

#[test]
fn json_array_file_is_auto_detected_from_extension() {
    let mut sess = Session::new();
    // register_bytes は拡張子から FormatKind::Auto → Json を選ぶ。
    sess.register_bytes("basic_array.json", data("basic_array.json")).unwrap();
    let rows = run_all("SELECT count(*) AS n FROM \"basic_array.json\"", &mut sess);
    assert_eq!(rows, [[Value::I64(1000)]]);
}

#[test]
fn json_array_file_matches_duckdb_output_through_sql() {
    // duckdb: SELECT * FROM read_json_auto('basic_array.json') ORDER BY id LIMIT 3
    let mut sess = Session::new();
    sess.register_bytes_as("t", data("basic_array.json"), FormatKind::Json).unwrap();
    let rows =
        run_all("SELECT id, name, score, flag, big, d FROM t ORDER BY id LIMIT 3", &mut sess);
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0][0], Value::I64(0));
    assert_eq!(rows[0][1], s("name_0"));
    assert_eq!(rows[0][2], Value::F64(0.0));
    assert_eq!(rows[0][3], Value::Bool(true));
    assert_eq!(rows[0][4], Value::Null); // i % 5 == 0
    assert_eq!(rows[1][2], Value::F64(1.5));
    assert_eq!(rows[1][4], Value::I64(100));
    // duckdb: epoch_us(d) の先頭行 → 1704067200000000
    assert_eq!(rows[0][5], Value::I64(1_704_067_200_000_000));
}

#[test]
fn json_array_and_jsonl_agree_on_aggregates() {
    // 同じ内容（basic.parquet 由来）を JSON 配列と JSONL の両方で読み、
    // 集約結果が一致することを確認する。フォーマットが違っても論理的な
    // 中身は同じであるべき、という不変条件のテスト。
    let mut sess = Session::new();
    sess.register_bytes_as("arr", data("basic_array.json"), FormatKind::Json).unwrap();
    sess.register_bytes_as("lines", data("basic.jsonl"), FormatKind::Jsonl).unwrap();

    let arr = run_all("SELECT count(*), sum(score), count(big) FROM arr", &mut sess);
    let lines = run_all("SELECT count(*), sum(score), count(big) FROM lines", &mut sess);
    assert_eq!(arr, lines);
}

#[test]
fn where_clause_and_projection_work_over_json_array() {
    let mut sess = Session::new();
    sess.register_bytes_as("t", data("basic_array.json"), FormatKind::Json).unwrap();
    let rows = run_all("SELECT name FROM t WHERE id = 999", &mut sess);
    assert_eq!(rows, [[s("name_5")]]);
}

#[test]
fn top_level_object_and_scalar_array_round_trip_through_sql() {
    // register_bytes_as はメモリ上のバイト列をそのまま渡せるので、実ファイルを
    // 経由せずにトップレベルの規則（モジュール doc 参照）を SQL からも確認する。
    let mut sess = Session::new();
    sess.register_bytes_as("obj", br#"{"a":1,"b":"hello"}"#.to_vec(), FormatKind::Json).unwrap();
    let rows = run_all("SELECT a, b FROM obj", &mut sess);
    assert_eq!(rows, [[Value::I64(1), s("hello")]]);

    let mut sess = Session::new();
    sess.register_bytes_as("scalars", b"[1,2,3]".to_vec(), FormatKind::Json).unwrap();
    let rows = run_all("SELECT sum(json) AS total FROM scalars", &mut sess);
    // SUM(INT) はこのエンジンでは常に HUGEINT (I128) に広がる（他の SUM テストと同じ規約）。
    assert_eq!(rows, [[Value::I128(6)]]);
}
