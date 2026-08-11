//! `CURRENT_DATE`/`CURRENT_TIMESTAMP`/`CURRENT_TIME`/`now()`/`today()` の
//! 統合テスト。`Session::set_now` で固定値を渡し、`sql::now::substitute_now`
//! が実際の `Session::prepare` 経路を通しても正しく効くことを確認する
//! （`crates/ahiru-core/src/sql/now.rs` の単体テストは AST レベルの確認のみ）。

use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

// 2024-01-15 12:30:00 UTC。
const NOW: i64 = 1_705_321_800_000_000;
const TODAY_DAYS: i32 = 19737;

fn session_with_dual() -> Session {
    let mut s = Session::new();
    s.register_bytes_as("dual", b"x\n1\n".to_vec(), FormatKind::Csv).unwrap();
    s.set_now(NOW);
    s
}

fn run(s: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    let mut q = match s.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
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
            QueryStep::Done => break,
            QueryStep::NeedIo(_) | QueryStep::NeedCodec(_) => panic!("unexpected suspend"),
        }
    }
    rows
}

#[test]
fn bare_forms_and_call_forms_all_resolve_to_the_configured_now() {
    let mut s = session_with_dual();
    let rows = run(
        &mut s,
        "SELECT CURRENT_DATE, CURRENT_TIMESTAMP, now(), today(), current_time FROM dual",
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], Value::I32(TODAY_DAYS), "CURRENT_DATE");
    assert_eq!(rows[0][1], Value::I64(NOW), "CURRENT_TIMESTAMP");
    assert_eq!(rows[0][2], Value::I64(NOW), "now()");
    assert_eq!(rows[0][3], Value::I32(TODAY_DAYS), "today()");
    assert_eq!(rows[0][4], Value::I64(12 * 3_600_000_000 + 30 * 60_000_000), "current_time");
}

#[test]
fn current_timestamp_is_evaluated_once_per_query_not_per_row() {
    // SQL 標準の契約: CURRENT_TIMESTAMP はクエリ開始時に 1 回だけ評価され、
    // 複数行に渡って同じ値になる（行ごとに再評価されない）。
    let mut s = Session::new();
    s.register_bytes_as("t", b"id\n1\n2\n3\n".to_vec(), FormatKind::Csv).unwrap();
    s.set_now(NOW);
    let rows = run(&mut s, "SELECT id, CURRENT_TIMESTAMP FROM t ORDER BY id");
    assert_eq!(rows.len(), 3);
    for r in &rows {
        assert_eq!(r[1], Value::I64(NOW));
    }
}

#[test]
fn typed_literals_can_be_used_in_expressions() {
    let mut s = session_with_dual();
    // CURRENT_DATE + 整数日数は既存の DATE 演算と同じように使えるべき。
    let rows = run(&mut s, "SELECT CURRENT_DATE + 1 FROM dual");
    assert_eq!(rows[0][0], Value::I32(TODAY_DAYS + 1));
}

#[test]
fn unset_now_defaults_to_the_unix_epoch() {
    // set_now を一度も呼ばなければエポック（1970-01-01）になる
    // （時計を持たないコアが黙って嘘の時刻を返さないようにする既定値）。
    let mut s = Session::new();
    s.register_bytes_as("dual", b"x\n1\n".to_vec(), FormatKind::Csv).unwrap();
    let rows = run(&mut s, "SELECT CURRENT_DATE FROM dual");
    assert_eq!(rows[0][0], Value::I32(0));
}

#[test]
fn a_real_column_named_current_date_is_shadowed_by_the_bare_keyword_form() {
    // 既知のトレードオフ: `current_date`/`current_timestamp`/`current_time`
    // は裸の識別子として現れた時点で無条件にキーワード扱いする
    // （SQL標準でこれらは実質予約語であり、実データに同名の列が来る可能性は
    // 極めて低いと判断した。`sql/now.rs` のモジュール doc 参照）。
    // その結果、同名の実列があっても関数呼び出し扱いが優先されることを
    // 明示的に固定しておく（「なぜかテーブルの値が返らない」と将来
    // 混乱しないように）。
    let mut s = Session::new();
    s.register_bytes_as("t", b"current_date\n2000-01-01\n".to_vec(), FormatKind::Csv).unwrap();
    s.set_now(NOW);
    let rows = run(&mut s, "SELECT current_date FROM t");
    assert_eq!(rows[0][0], Value::I32(TODAY_DAYS), "列の値ではなく今日の日付になる");
}

#[test]
fn a_real_column_named_today_or_now_is_not_shadowed() {
    // `today`/`now` は括弧を伴わない裸の識別子としては特別扱いしない
    // （関数形は `today()`/`now()` のみ対象）。実データの列名との衝突を
    // 避けるための意図的な線引き。
    let mut s = Session::new();
    s.register_bytes_as("t", b"today,now\nhello,world\n".to_vec(), FormatKind::Csv).unwrap();
    s.set_now(NOW);
    let rows = run(&mut s, "SELECT today, now FROM t");
    assert_eq!(rows[0][0], Value::Bytes(b"hello".to_vec()));
    assert_eq!(rows[0][1], Value::Bytes(b"world".to_vec()));
}
