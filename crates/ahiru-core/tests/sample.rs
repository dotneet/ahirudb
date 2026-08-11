//! `USING SAMPLE` / `TABLESAMPLE` の統合テスト。
//!
//! `duckdb` CLI で構文・値域チェックの挙動を確認して決めた仕様:
//! - `USING SAMPLE <n>%` / `TABLESAMPLE <n>%` はパーセント指定。
//! - `USING SAMPLE <n> ROWS` / 単位無しの裸の数値は行数指定。
//! - `BERNOULLI(...)`/`SYSTEM(...)`/`RESERVOIR(...)` という手法名や
//!   `(method, seed)` という明示シードの構文は受理する。
//! - パーセントは `0..=100`、行数は `0` 以上でなければ構文エラー。
//! - `duckdb` は乱択なので「ちょうど何行」までは一致させられない
//!   （行数指定は例外 ―― 入力がその行数以上あれば必ずちょうどその行数）。
//!   このエンジン自身の PRNG は自前の xorshift64* なので、`duckdb` の
//!   選ぶ行そのものとの一致は求めない（同じ確率分布にすることだけが目標）。
//!
//! 乱択の検証は「件数がだいたい期待通りか」「同じシードで再現するか」
//! 「異なるシードなら違う結果になるか」「`NeedIo` をまたいでも結果が
//! 変わらないか」の 4 本柱で行う。

use ahiru_core::error::{code_of, Code};
use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn data(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn session_with_dual() -> Session {
    let mut s = Session::new();
    s.register_bytes_as("dual", b"x\n1\n".to_vec(), FormatKind::Csv).unwrap();
    s
}

/// 全データがメモリ上にあるクエリを最後まで実行し、`x` 列（0 列目）の値を集める。
fn run_x(s: &mut Session, sql: &str) -> Vec<i64> {
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
                    let Value::I64(v) = b.cols[0].value_at(r) else { panic!("expected I64") };
                    rows.push(v);
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

const N: i64 = 20_000;

fn big_table_sql(sample: &str) -> String {
    format!("SELECT range AS x FROM range({N}) {sample}")
}

// --- パーセント指定（ベルヌーイ法） -------------------------------------------

#[test]
fn using_sample_percent_keeps_roughly_the_requested_fraction() {
    let mut db = session_with_dual();
    let got = run_x(&mut db, &big_table_sql("USING SAMPLE 10%"));
    let frac = got.len() as f64 / N as f64;
    assert!((0.08..0.12).contains(&frac), "got fraction {frac} ({} rows)", got.len());
    // 重複が無く、実在する値のみで、入力の相対順序のまま。
    let mut sorted = got.clone();
    sorted.dedup();
    assert_eq!(got, sorted, "重複してはいけない");
    let mut asc = got.clone();
    asc.sort_unstable();
    assert_eq!(got, asc, "入力の相対順序を保つ");
}

#[test]
fn tablesample_syntax_is_equivalent_to_using_sample() {
    let mut db = session_with_dual();
    let got = run_x(&mut db, &big_table_sql("TABLESAMPLE 10%"));
    let frac = got.len() as f64 / N as f64;
    assert!((0.08..0.12).contains(&frac), "got fraction {frac}");
}

#[test]
fn zero_percent_keeps_nothing_and_hundred_percent_keeps_everything() {
    let mut db = session_with_dual();
    assert!(run_x(&mut db, &big_table_sql("USING SAMPLE 0%")).is_empty());
    assert_eq!(run_x(&mut db, &big_table_sql("USING SAMPLE 100%")).len(), N as usize);
}

#[test]
fn explicit_method_names_are_accepted() {
    let mut db = session_with_dual();
    for m in ["BERNOULLI", "SYSTEM"] {
        let sql = big_table_sql(&format!("USING SAMPLE {m}(10%)"));
        let got = run_x(&mut db, &sql);
        // `SYSTEM` はこのエンジンでは `BERNOULLI` と同じ実装に落ちる
        // （タスクの優先度どおり、手法の違いは実装しない）。
        let frac = got.len() as f64 / N as f64;
        assert!((0.05..0.15).contains(&frac), "{m}: got fraction {frac}");
    }
}

// --- 行数指定 -------------------------------------------------------------------

#[test]
fn using_sample_rows_selects_exactly_that_many_rows() {
    let mut db = session_with_dual();
    let got = run_x(&mut db, &big_table_sql("USING SAMPLE 100 ROWS"));
    assert_eq!(got.len(), 100);
    let mut sorted = got.clone();
    sorted.dedup();
    assert_eq!(got, sorted, "重複してはいけない");
}

#[test]
fn bare_number_without_a_unit_means_rows() {
    let mut db = session_with_dual();
    let got = run_x(&mut db, &big_table_sql("USING SAMPLE 50"));
    assert_eq!(got.len(), 50);
}

#[test]
fn reservoir_method_with_rows_is_accepted() {
    let mut db = session_with_dual();
    let got = run_x(&mut db, &big_table_sql("USING SAMPLE reservoir(30)"));
    assert_eq!(got.len(), 30);
}

#[test]
fn requesting_more_rows_than_available_returns_everything() {
    let mut db = session_with_dual();
    let sql = "SELECT range AS x FROM range(10) USING SAMPLE 1000 ROWS";
    let got = run_x(&mut db, sql);
    let want: Vec<i64> = (0..10).collect();
    assert_eq!(got, want);
}

// --- シードの再現性 ---------------------------------------------------------------

#[test]
fn same_explicit_seed_reproduces_the_same_sample() {
    let sql = big_table_sql("USING SAMPLE 20% (bernoulli, 42)");
    let mut a = session_with_dual();
    let mut b = session_with_dual();
    assert_eq!(run_x(&mut a, &sql), run_x(&mut b, &sql));
}

#[test]
fn different_seed_gives_a_different_sample() {
    let mut a = session_with_dual();
    let mut b = session_with_dual();
    let got_a = run_x(&mut a, &big_table_sql("USING SAMPLE 20% (bernoulli, 1)"));
    let got_b = run_x(&mut b, &big_table_sql("USING SAMPLE 20% (bernoulli, 2)"));
    assert_ne!(got_a, got_b);
}

#[test]
fn default_seed_without_an_explicit_one_is_still_deterministic() {
    // タスクの指示どおり「シードは決定的でよい」: 同じクエリを 2 回実行すれば
    // 同じサンプルが選ばれる（`duckdb` の既定は毎回変わるが、このエンジンは
    // 固定の既定シードを使う。`plan::DEFAULT_SAMPLE_SEED`）。
    let sql = big_table_sql("USING SAMPLE 15%");
    let mut a = session_with_dual();
    let mut b = session_with_dual();
    assert_eq!(run_x(&mut a, &sql), run_x(&mut b, &sql));
}

#[test]
fn row_sample_is_also_reproducible_with_the_same_seed() {
    let sql = big_table_sql("USING SAMPLE reservoir(50)");
    let mut a = session_with_dual();
    let mut b = session_with_dual();
    assert_eq!(run_x(&mut a, &sql), run_x(&mut b, &sql));
}

// --- 値域エラー --------------------------------------------------------------------

#[test]
fn percent_out_of_range_is_rejected() {
    let mut db = session_with_dual();
    let err = db.prepare(&big_table_sql("USING SAMPLE 150%"), &[]);
    assert_eq!(code_of(err), Some(Code::SyntaxError));
}

#[test]
fn zero_interval_step_style_errors_do_not_apply_but_negative_amount_is_rejected() {
    let mut db = session_with_dual();
    // 負の量は式の先頭で単項マイナスにぶつかり構文エラーになる
    // （`sample_amount` はマイナス記号を読まない）。
    let err = db.prepare(&big_table_sql("USING SAMPLE -5 ROWS"), &[]);
    assert!(code_of(err).is_some());
}

// --- NeedIo をまたいだ再現性 -------------------------------------------------------

/// `NeedIo` を挟んでバイト列を少しずつ供給しても、一括供給と全く同じ結果に
/// なることを確認する。ベルヌーイ法は入力の `NeedIo` をそのまま素通しする
/// だけ（`exec::sample::Bernoulli` の doc 参照）で、PRNG の呼び出し列は
/// 「実際に評価された行の並び」だけで決まるので、中断のタイミングに依存
/// しないはず。
#[test]
fn need_io_across_a_real_parquet_scan_does_not_change_a_percent_sample() {
    let sql = "SELECT id FROM t USING SAMPLE 20% (bernoulli, 7)";
    let bytes = data("pagetest.parquet");

    let mut eager = Session::new();
    eager.register_bytes_as("t", bytes.clone(), FormatKind::Parquet).unwrap();
    let want = run_id(&mut eager, sql);
    assert!(!want.is_empty());

    let (got, rounds) = run_id_with_lazy_io(&bytes, sql);
    assert_eq!(got, want, "NeedIo をまたいでも結果が変わってはいけない");
    assert!(rounds >= 1);
}

/// 行数指定（ブロッキング方式）でも同じことを確認する。
#[test]
fn need_io_across_a_real_parquet_scan_does_not_change_a_row_sample() {
    let sql = "SELECT id FROM t USING SAMPLE reservoir(200)";
    let bytes = data("pagetest.parquet");

    let mut eager = Session::new();
    eager.register_bytes_as("t", bytes.clone(), FormatKind::Parquet).unwrap();
    let want = run_id(&mut eager, sql);
    assert_eq!(want.len(), 200);

    let (got, rounds) = run_id_with_lazy_io(&bytes, sql);
    assert_eq!(got, want, "NeedIo をまたいでも結果が変わってはいけない");
    assert!(rounds >= 1);
}

fn run_id(s: &mut Session, sql: &str) -> Vec<i32> {
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
                    let Value::I32(v) = b.cols[0].value_at(r) else { panic!("expected I32") };
                    rows.push(v);
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
/// `provide` して駆動する（`unnest.rs::run_with_lazy_io` と同じ手口）。
fn run_id_with_lazy_io(bytes: &[u8], sql: &str) -> (Vec<i32>, u32) {
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
                    let Value::I32(v) = b.cols[0].value_at(r) else { panic!("expected I32") };
                    rows.push(v);
                }
            }
            QueryStep::NeedIo(reqs) => {
                rounds += 1;
                for r in reqs {
                    let (start, end) = (r.offset as usize, (r.offset + r.len) as usize);
                    s.provide(r.table, r.part, r.offset, bytes[start..end].to_vec()).unwrap();
                }
            }
            QueryStep::NeedCodec(_) => panic!("test fixtures are uncompressed"),
            QueryStep::Done => break,
        }
    }
    (rows, rounds)
}

// --- 句の置き場所（`WHERE`/`GROUP BY`/`QUALIFY` との相互作用） ------------------
//
// `duckdb` CLI で確認した仕様: `TABLESAMPLE` は FROM 項目に直接くっつく修飾子
// で必ず `WHERE` より前に置く。`USING SAMPLE` はその逆で文全体の独立した句
// として `WHERE`/`GROUP BY`/`HAVING`/`WINDOW`/`QUALIFY` の後・`ORDER BY` の
// 前に置く。両方とも「FROM 項目の直後、かつ他に何も続かない」ときだけ同じ
// 位置に見えるため、`WHERE` を伴わない既存テストだけではこの違いに気づけ
// なかった（これが原因で見つかった不具合。`sql::parser::opt_using_sample_clause`
// / `opt_tablesample_clause` 参照）。

#[test]
fn using_sample_can_follow_where_group_by_and_qualify() {
    let mut db = session_with_dual();
    // WHERE の後。
    let got = run_x(
        &mut db,
        &format!("SELECT range AS x FROM range({N}) WHERE range % 2 = 0 USING SAMPLE 100%"),
    );
    assert_eq!(got.len(), (N / 2) as usize);
}

#[test]
fn using_sample_right_after_from_is_rejected_when_where_follows() {
    // `USING SAMPLE` は FROM 項目に直接くっつく修飾子ではないので、
    // `WHERE` が後に続く形で FROM 項目の直後に書くと構文エラーになる
    // （`duckdb` も同じ理由で拒否する）。
    let mut db = session_with_dual();
    let err = db.prepare(
        &format!("SELECT range AS x FROM range({N}) USING SAMPLE 10% WHERE range % 2 = 0"),
        &[],
    );
    assert!(code_of(err).is_some());
}

#[test]
fn tablesample_after_where_is_rejected() {
    // 逆に `TABLESAMPLE` は文末側の句としては書けない。
    let mut db = session_with_dual();
    let err = db.prepare(
        &format!("SELECT range AS x FROM range({N}) WHERE range % 2 = 0 TABLESAMPLE 10%"),
        &[],
    );
    assert!(code_of(err).is_some());
}

#[test]
fn tablesample_still_works_right_after_from_with_a_following_where() {
    // `TABLESAMPLE` は今までどおり FROM 項目の直後、`WHERE` より前に置ける
    // （リグレッション確認: このケースはバグ修正の前から動いていた）。
    let mut db = session_with_dual();
    let got = run_x(
        &mut db,
        &format!("SELECT range AS x FROM range({N}) TABLESAMPLE 100% WHERE range % 2 = 0"),
    );
    assert_eq!(got.len(), (N / 2) as usize);
}

#[test]
fn combining_tablesample_and_trailing_using_sample_is_rejected() {
    // `duckdb` は両方を同時に書くと順番に両方適用するが、このエンジンの
    // `SampleSpec` は 1 個しか持てない単純化のため、二重指定は明示的に
    // 拒否する（黙って片方を無視すると気づきにくいバグになるため）。
    let mut db = session_with_dual();
    let err = db.prepare(
        &format!(
            "SELECT range AS x FROM range({N}) TABLESAMPLE 50% WHERE range % 2 = 0 \
             USING SAMPLE 100%"
        ),
        &[],
    );
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

#[test]
fn using_sample_can_follow_group_by_having_before_order_by() {
    let mut db = session_with_dual();
    let got = run_x(
        &mut db,
        "SELECT range % 3 AS x FROM range(30) GROUP BY range % 3 HAVING count(*) > 0 \
         USING SAMPLE 100% ORDER BY x",
    );
    assert_eq!(got, vec![0, 1, 2]);
}
