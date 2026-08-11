//! `WHERE` の `IN` / `BETWEEN` 述語が RowGroup/PageIndex/Bloom フィルタの
//! プルーニングを正しく通ることの SQL レベルでの確認。
//!
//! `format::parquet` の単体テストは `ParquetFormat` を直接叩いてバイト削減
//! そのものを検証しているが、ここでは `Session` 越しの実行経路（束縛 →
//! pruner 抽出 → 実行）が正しく配線されていること、特に「絞り込みすぎて
//! 本来ヒットすべき行を消していないか」を実データで確認する。期待値は
//! `duckdb -c "SELECT ..."` の実際の出力と突き合わせて決めている。
//!
//! `tests/data/pagetest.parquet` は id が `0..50000` を隙間なく埋める
//! 50000 行・複数 RowGroup/複数ページのファイル（`format::parquet` の
//! ページ単位プルーニングのテストで使っているものと同じ）。

use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn data(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn session_with_pagetest() -> Session {
    let mut s = Session::new();
    s.register_bytes_as("t", data("pagetest.parquet"), FormatKind::Parquet).unwrap();
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

/// `register_remote_as` で登録し、`NeedIo` が要求した範囲だけをそのつど
/// `provide` して駆動する（`sample.rs::run_id_with_lazy_io` と同じ手口）。
/// プルーニングでバイト取得範囲が変わっても `NeedIo` の駆動が壊れないことを
/// 一緒に確認する。
fn run_with_lazy_io(bytes: &[u8], sql: &str) -> Vec<Vec<Value>> {
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
            QueryStep::Done => break,
            QueryStep::NeedIo(reqs) => {
                rounds += 1;
                assert!(rounds < 1000, "step が終わらない");
                for r in reqs {
                    let (start, end) = (r.offset as usize, (r.offset + r.len) as usize);
                    s.provide(r.table, r.part, r.offset, bytes[start..end].to_vec()).unwrap();
                }
            }
            QueryStep::NeedCodec(_) => panic!("test fixtures are uncompressed"),
        }
    }
    rows
}

#[test]
fn in_list_finds_the_one_present_value_among_absent_decoys() {
    let mut s = session_with_pagetest();
    // duckdb: SELECT id FROM 'pagetest.parquet' WHERE id IN (12345, 999999999, -1) ORDER BY id
    let rows = run(&mut s, "SELECT id FROM t WHERE id IN (12345, 999999999, -1) ORDER BY id");
    assert_eq!(rows, vec![vec![Value::I32(12345)]]);
}

#[test]
fn in_list_with_multiple_present_values_returns_all_of_them() {
    let mut s = session_with_pagetest();
    let rows = run(&mut s, "SELECT id FROM t WHERE id IN (10, 20, 12345, 40000) ORDER BY id");
    assert_eq!(
        rows,
        vec![
            vec![Value::I32(10)],
            vec![Value::I32(20)],
            vec![Value::I32(12345)],
            vec![Value::I32(40000)],
        ]
    );
}

#[test]
fn not_in_is_unaffected_by_the_new_pruning_path() {
    let mut s = session_with_pagetest();
    // duckdb: SELECT count(*) FROM 'pagetest.parquet' WHERE id NOT IN (10,20,30) -> 49997
    let rows = run(&mut s, "SELECT count(*) FROM t WHERE id NOT IN (10, 20, 30)");
    assert_eq!(rows, vec![vec![Value::I64(49997)]]);
}

#[test]
fn between_end_to_end_matches_duckdb() {
    let mut s = session_with_pagetest();
    // duckdb: SELECT count(*) FROM 'pagetest.parquet' WHERE id BETWEEN 12000 AND 12010 -> 11
    let rows = run(&mut s, "SELECT count(*) FROM t WHERE id BETWEEN 12000 AND 12010");
    assert_eq!(rows, vec![vec![Value::I64(11)]]);
}

#[test]
fn in_list_result_is_identical_whether_or_not_pruning_narrows_the_io() {
    let bytes = data("pagetest.parquet");
    let sql = "SELECT id FROM t WHERE id IN (5, 12345, 39999, 999999999) ORDER BY id";

    let mut eager = Session::new();
    eager.register_bytes_as("t", bytes.clone(), FormatKind::Parquet).unwrap();
    let want = run(&mut eager, sql);
    assert_eq!(want, vec![vec![Value::I32(5)], vec![Value::I32(12345)], vec![Value::I32(39999)],]);

    let got = run_with_lazy_io(&bytes, sql);
    assert_eq!(got, want, "NeedIo を挟んでも IN プルーニングの結果は変わってはいけない");
}

#[test]
fn in_list_on_a_non_literal_candidate_still_returns_correct_rows_without_pruning() {
    // 候補にリテラル以外（列参照）が混ざるケース。pruner は作られない
    // （`plan::bind::tests::in_list_with_non_literal_element_is_not_pruned`
    // で確認済み）が、実行結果そのものは正しくなければならない。
    let mut s = session_with_pagetest();
    let rows = run(&mut s, "SELECT id FROM t WHERE id IN (12345, id) ORDER BY id LIMIT 3");
    assert_eq!(rows, vec![vec![Value::I32(0)], vec![Value::I32(1)], vec![Value::I32(2)]]);
}
