//! SQL のエンドツーエンド回帰テスト。
//!
//! **期待値を書かずに DuckDB と突き合わせる。** 手で書いた期待値は書き間違いが
//! そのまま仕様になってしまうし、クエリを増やすたびに手計算が要る。DuckDB を
//! 参照実装として使えば、クエリを 1 行足すだけで検証が増える。
//!
//! DuckDB が入っていない環境ではテストごと飛ばす（CI では入れる）。

use std::process::Command;

fn duckdb_available() -> bool {
    Command::new("duckdb").arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn repo_path(rel: &str) -> String {
    format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel)
}

/// ahirudb の CLI で実行し、行 × 列に分解する（ヘッダ含む）。
fn run_ahiru(files: &[&str], sql: &str) -> Vec<Vec<String>> {
    let mut args: Vec<String> = vec!["query".into()];
    args.extend(files.iter().map(|f| repo_path(f)));
    args.push(sql.into());
    let out = Command::new(env!("CARGO_BIN_EXE_ahiru"))
        .args(&args)
        .output()
        .expect("failed to run ahiru");
    assert!(
        out.status.success(),
        "ahiru failed for `{sql}`:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    parse(&text, '\t')
}

fn duckdb_source(file: &str) -> String {
    let path = repo_path(file);
    if file.ends_with(".csv") {
        format!("read_csv_auto('{path}')")
    } else if file.ends_with(".jsonl") {
        format!("read_json_auto('{path}')")
    } else {
        format!("'{path}'")
    }
}

/// DuckDB で同じクエリを実行する。テーブル名 `t`, `t2`, ... をファイル参照に置き換える。
fn run_duckdb(files: &[&str], sql: &str) -> Vec<Vec<String>> {
    let sql = replace_tables(sql, files);
    let out =
        Command::new("duckdb").args(["-csv", "-c", &sql]).output().expect("failed to run duckdb");
    assert!(
        out.status.success(),
        "duckdb failed for `{sql}`:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    parse(&String::from_utf8_lossy(&out.stdout), ',')
}

/// `t`, `t2`, `t3`, ... をそれぞれのファイル参照に差し替える。
///
/// 単語境界で切って一致を見る。`text` の中の `t` や、`t` が `t2` の一部で
/// あるケースを壊さないため。
fn replace_tables(sql: &str, files: &[&str]) -> String {
    let names: Vec<String> =
        (0..files.len()).map(|i| if i == 0 { "t".into() } else { format!("t{}", i + 1) }).collect();
    let b = sql.as_bytes();
    let is_word = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut out = String::with_capacity(sql.len() + 64);
    let mut i = 0usize;
    while i < b.len() {
        if is_word(b[i]) && (i == 0 || !is_word(b[i - 1])) {
            let mut j = i;
            while j < b.len() && is_word(b[j]) {
                j += 1;
            }
            let word = &sql[i..j];
            match names.iter().position(|n| n == word) {
                Some(k) => out.push_str(&duckdb_source(files[k])),
                None => out.push_str(word),
            }
            i = j;
            continue;
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

fn parse(text: &str, sep: char) -> Vec<Vec<String>> {
    text.lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.split(sep).map(normalize_cell).collect())
        .collect()
}

/// セルを比較可能な形に揃える。
///
/// - NULL の表記が違う（ahirudb は `NULL`、DuckDB の CSV は空）
/// - 浮動小数の桁の出し方が違う（`747` と `747.0`）
fn normalize_cell(s: &str) -> String {
    let s = s.trim().trim_matches('"');
    if s.is_empty() || s == "NULL" {
        return "<null>".into();
    }
    if let Ok(f) = s.parse::<f64>() {
        // 整数として表せる値は整数表記に寄せる。
        if f.fract() == 0.0 && f.abs() < 9e15 {
            return format!("{}", f as i64);
        }
        return format!("{f:.9}");
    }
    s.into()
}

fn check(file: &str, sql: &str) {
    check_multi(&[file], sql)
}

/// 1 つのクエリを両方で実行して突き合わせる。
fn check_multi(files: &[&str], sql: &str) {
    // ヘッダは比較しない。別名の付け方までは合わせないし、DuckDB は結果が
    // 0 行のときヘッダ自体を出さない。データ行だけを突き合わせる。
    let a = body(run_ahiru(files, sql));
    let d = body(run_duckdb(files, sql));
    assert_eq!(a.len(), d.len(), "行数が違う\nSQL: {sql}\nahiru: {a:?}\nduckdb: {d:?}");
    for (i, (ra, rd)) in a.iter().zip(&d).enumerate() {
        assert_eq!(ra, rd, "{} 行目が違う\nSQL: {sql}", i + 1);
    }
}

fn body(mut rows: Vec<Vec<String>>) -> Vec<Vec<String>> {
    if rows.is_empty() {
        return rows;
    }
    rows.remove(0);
    rows
}

macro_rules! e2e_multi {
    ($name:ident, [$($file:expr),* $(,)?], [$($sql:expr),* $(,)?]) => {
        #[test]
        fn $name() {
            if !duckdb_available() {
                eprintln!("duckdb が無いので飛ばす");
                return;
            }
            let files = [$($file),*];
            $( check_multi(&files, $sql); )*
        }
    };
}

macro_rules! e2e {
    ($name:ident, $file:expr, [$($sql:expr),* $(,)?]) => {
        #[test]
        fn $name() {
            if !duckdb_available() {
                eprintln!("duckdb が無いので飛ばす");
                return;
            }
            $( check($file, $sql); )*
        }
    };
}

e2e!(
    projection_and_filter,
    "tests/data/basic.parquet",
    [
        "SELECT id, name FROM t ORDER BY id LIMIT 5",
        "SELECT * FROM t ORDER BY id LIMIT 3",
        "SELECT id FROM t WHERE id > 995 ORDER BY id",
        "SELECT id, big FROM t WHERE big IS NULL ORDER BY id LIMIT 5",
        "SELECT id, big FROM t WHERE big IS NOT NULL ORDER BY id LIMIT 5",
        "SELECT id FROM t WHERE id BETWEEN 10 AND 14 ORDER BY id",
        "SELECT id FROM t WHERE id IN (1, 3, 5) ORDER BY id",
        "SELECT id FROM t WHERE id NOT IN (1, 3, 5) ORDER BY id LIMIT 5",
        "SELECT name FROM t WHERE name LIKE 'name_1' ORDER BY name LIMIT 3",
        "SELECT id FROM t WHERE NOT (id > 5) ORDER BY id",
    ]
);

e2e!(expressions, "tests/data/basic.parquet", [
    "SELECT id + 1, id * 2, id - 1 FROM t ORDER BY id LIMIT 5",
    "SELECT score / 2 FROM t ORDER BY id LIMIT 5",
    "SELECT id % 7 FROM t ORDER BY id LIMIT 10",
    "SELECT CASE WHEN id < 3 THEN 'a' WHEN id < 6 THEN 'b' ELSE 'c' END FROM t ORDER BY id LIMIT 8",
    "SELECT CAST(id AS DOUBLE) FROM t ORDER BY id LIMIT 3",
    "SELECT CAST(score AS BIGINT) FROM t ORDER BY id LIMIT 5",
    "SELECT id, flag FROM t WHERE flag ORDER BY id LIMIT 5",
    "SELECT id FROM t WHERE big > 100 AND id < 20 ORDER BY id",
    "SELECT id FROM t WHERE big IS NULL OR id > 997 ORDER BY id LIMIT 5",
    // NULL を含む算術は NULL になる。
    "SELECT id, big + 1 FROM t ORDER BY id LIMIT 6",
]);

e2e!(
    aggregates,
    "tests/data/basic.parquet",
    [
        "SELECT count(*) FROM t",
        "SELECT count(big) FROM t",
        "SELECT sum(id), min(id), max(id), avg(id) FROM t",
        "SELECT sum(big), avg(big) FROM t",
        "SELECT min(name), max(name) FROM t",
        "SELECT name, count(*) FROM t GROUP BY name ORDER BY name",
        "SELECT name, sum(big), avg(score) FROM t GROUP BY name ORDER BY name",
        "SELECT name, count(*) FROM t GROUP BY name HAVING count(*) > 142 ORDER BY name",
        "SELECT count(DISTINCT name) FROM t",
        "SELECT flag, count(*) FROM t GROUP BY flag ORDER BY flag",
        "SELECT id % 3, count(*) FROM t GROUP BY 1 ORDER BY 1",
        // 1 行も残らない集約。
        "SELECT count(*) FROM t WHERE id > 100000",
        "SELECT name, count(*) FROM t WHERE id > 100000 GROUP BY name ORDER BY name",
    ]
);

e2e!(
    sorting_and_limits,
    "tests/data/basic.parquet",
    [
        "SELECT id FROM t ORDER BY id DESC LIMIT 5",
        "SELECT id, name FROM t ORDER BY name DESC, id ASC LIMIT 10",
        "SELECT id FROM t ORDER BY id LIMIT 5 OFFSET 10",
        "SELECT big FROM t ORDER BY big NULLS FIRST LIMIT 5",
        "SELECT big FROM t ORDER BY big NULLS LAST LIMIT 5",
        "SELECT big FROM t ORDER BY big DESC NULLS FIRST LIMIT 5",
        "SELECT DISTINCT name FROM t ORDER BY name",
        "SELECT DISTINCT flag FROM t ORDER BY flag",
        // 出力に無い式で並べ替える（隠しソート列）。
        "SELECT name FROM t ORDER BY id DESC LIMIT 5",
    ]
);

e2e!(
    csv_matches_parquet_semantics,
    "tests/data/basic.csv",
    [
        "SELECT id, name FROM t ORDER BY id LIMIT 5",
        "SELECT count(*) FROM t",
        "SELECT count(big) FROM t",
        "SELECT name, count(*) FROM t GROUP BY name ORDER BY name",
        "SELECT id FROM t WHERE id > 995 ORDER BY id",
    ]
);

e2e!(
    jsonl_matches_parquet_semantics,
    "tests/data/basic.jsonl",
    [
        "SELECT id, name FROM t ORDER BY id LIMIT 5",
        "SELECT count(*) FROM t",
        "SELECT count(big) FROM t",
        "SELECT name, count(*) FROM t GROUP BY name ORDER BY name",
    ]
);

// GZIP はホストに展開を委譲する（DESIGN.md §6）。CLI ではシステムの `gzip` が
// その役目を果たす。ここが通れば、コーデック委譲プロトコルが端から端まで
// 動いていることになる。
e2e!(
    gzip_via_host_codec_delegation,
    "tests/data/gzip.parquet",
    [
        "SELECT id, s, f FROM t ORDER BY id LIMIT 5",
        "SELECT count(*) FROM t",
        "SELECT s, count(*) FROM t GROUP BY s ORDER BY s",
        "SELECT sum(f), avg(f) FROM t",
        "SELECT id FROM t WHERE id > 4995 ORDER BY id",
    ]
);

e2e!(
    multi_row_group_scan,
    "tests/data/multi_rg.parquet",
    [
        "SELECT count(*) FROM t",
        "SELECT sum(id), min(id), max(id) FROM t",
        "SELECT k, count(*) FROM t GROUP BY k ORDER BY k LIMIT 10",
        // 統計による枝刈りが効く範囲でも結果は変わらない。
        "SELECT count(*) FROM t WHERE id > 40000",
        "SELECT id FROM t WHERE id BETWEEN 8190 AND 8195 ORDER BY id",
    ]
);

// ページ単位の枝刈り（ColumnIndex/OffsetIndex/Bloom フィルタ）。
// pagetest.parquet は pyarrow (parquet-cpp) が ColumnIndex/OffsetIndex/Bloom
// フィルタ付きで書いた、id が 0..50000 の昇順・小さいページのファイル
// （DuckDB のこの環境の版は Bloom フィルタを書けないので pyarrow を使った。
// `scripts/gen-testdata.sh` にどちらでどう作ったかを記している）。
// DuckDB はこのファイルを読めるので、結果の正しさはいつも通り突き合わせる。
// 「実際にページ単位で絞り込めているか」自体は
// `crates/ahiru-core/src/format/parquet.rs` の
// `selective_equality_predicate_reads_a_small_fraction_of_the_column` が見る。
e2e!(
    page_level_pruning,
    "tests/data/pagetest.parquet",
    [
        "SELECT id, s FROM t WHERE id = 12345",
        "SELECT id, s FROM t WHERE id = 0",
        "SELECT id, s FROM t WHERE id = 49999",
        "SELECT count(*) FROM t WHERE id = 999999999",
        "SELECT id FROM t WHERE id > 49990 ORDER BY id",
        "SELECT id FROM t WHERE id < 5 ORDER BY id",
        "SELECT id, s FROM t WHERE id BETWEEN 12340 AND 12345 ORDER BY id",
        "SELECT count(*) FROM t",
        "SELECT sum(id) FROM t",
    ]
);

// 結合。DuckDB 側でもファイル参照に別名を付ける必要があるので、
// クエリには常に明示的な別名を書く。
// FROM 句の派生表。集約結果をさらに絞り込む形。
e2e!(derived_tables, "tests/data/basic.parquet", [
    "SELECT s.name, s.n FROM (SELECT name, count(*) AS n FROM t GROUP BY name) AS s WHERE s.n > 142 ORDER BY s.name",
    "SELECT count(*) FROM (SELECT id FROM t WHERE id < 10) AS s",
    "SELECT max(s.m) FROM (SELECT name, max(big) AS m FROM t GROUP BY name) AS s",
]);

// ZSTD は別 wasm モジュールに委譲する（DESIGN.md §6）。CLI では `ahiru-zstd`
// を直接リンクしてその役目を果たす。GZIP と同じ 1 本の経路を通る。
e2e!(
    zstd_via_host_codec_delegation,
    "tests/data/zstd2.parquet",
    [
        "SELECT id, s, f FROM t ORDER BY id LIMIT 5",
        "SELECT count(*) FROM t",
        "SELECT s, count(*) FROM t GROUP BY s ORDER BY s",
        "SELECT sum(f), avg(f) FROM t",
        "SELECT id FROM t WHERE id > 4995 ORDER BY id",
    ]
);

e2e_multi!(joins, ["tests/data/small_a.parquet", "tests/data/small_b.parquet"], [
    "SELECT a.k, a.v, b.w FROM t AS a JOIN t2 AS b ON a.k = b.k ORDER BY a.k",
    "SELECT a.k, b.w FROM t AS a LEFT JOIN t2 AS b ON a.k = b.k ORDER BY a.k",
    "SELECT a.k, b.w FROM t AS a RIGHT JOIN t2 AS b ON a.k = b.k ORDER BY b.k",
    "SELECT a.k, b.k FROM t AS a FULL JOIN t2 AS b ON a.k = b.k ORDER BY a.k NULLS LAST, b.k NULLS LAST",
    "SELECT a.k, b.k FROM t AS a CROSS JOIN t2 AS b ORDER BY a.k, b.k",
    // 等値に落ちない条件は残余述語として評価される。
    "SELECT a.k, b.k FROM t AS a JOIN t2 AS b ON a.k < b.k ORDER BY a.k, b.k",
    // 結合キーに加えて片側だけの条件（押し下げの対象）。
    "SELECT a.k, b.w FROM t AS a JOIN t2 AS b ON a.k = b.k WHERE a.v > 4 ORDER BY a.k",
    // 結合の上での集約。
    "SELECT count(*) FROM t AS a JOIN t2 AS b ON a.k = b.k",
    "SELECT a.k, sum(b.w) FROM t AS a JOIN t2 AS b ON a.k = b.k GROUP BY a.k ORDER BY a.k",
]);

// 大きい表と小さい表の結合。射影プッシュダウンと枝刈りが結合下でも効くこと。
e2e_multi!(join_with_dimension, ["tests/data/basic.parquet", "tests/data/dim.parquet"], [
    "SELECT count(*) FROM t AS f JOIN t2 AS d ON f.name = d.label",
    "SELECT d.nid, count(*) FROM t AS f JOIN t2 AS d ON f.name = d.label GROUP BY d.nid ORDER BY d.nid",
    "SELECT f.id, d.w FROM t AS f JOIN t2 AS d ON f.name = d.label WHERE f.id < 5 ORDER BY f.id",
]);

// --- SQL 糖衣構文（FILTER / QUALIFY / ILIKE / TRY_CAST / IIF / DISTINCT ON） ---

e2e!(
    filter_clause,
    "tests/data/basic.parquet",
    [
        "SELECT count(*) FILTER (WHERE id > 500) FROM t",
        "SELECT count(*) FILTER (WHERE id > 500), count(*) FROM t",
        "SELECT flag, count(*) FILTER (WHERE big IS NOT NULL) FROM t GROUP BY flag ORDER BY flag",
        "SELECT sum(id) FILTER (WHERE flag), sum(id) FILTER (WHERE NOT flag) FROM t",
        // 全行が条件に落ちる集約は SUM が NULL、COUNT(*) が 0 になる。
        "SELECT count(*) FILTER (WHERE id > 100000), sum(id) FILTER (WHERE id > 100000) FROM t",
        // FILTER 無しの集約と混在しても互いに影響しない。
        "SELECT count(*), count(*) FILTER (WHERE flag) FROM t GROUP BY name ORDER BY name LIMIT 5",
    ]
);

e2e!(
    qualify_clause,
    "tests/data/basic.parquet",
    [
        // ウィンドウ関数をそのまま QUALIFY に書く形。
        // ウィンドウ関数は行を並べ替えないので、出力順を突き合わせるには
        // 明示的な ORDER BY が要る（QUALIFY 自体は並びに関与しない）。
        "SELECT id FROM t QUALIFY row_number() OVER (ORDER BY id DESC) <= 3 ORDER BY id DESC",
        // SELECT の別名を QUALIFY から参照する形。
        "SELECT id, flag, row_number() OVER (PARTITION BY flag ORDER BY id) AS rn FROM t QUALIFY rn <= 2 ORDER BY flag, id",
        "SELECT id, flag, rank() OVER (PARTITION BY flag ORDER BY id) AS rk FROM t QUALIFY rk = 1 ORDER BY flag",
    ]
);

e2e!(
    ilike_predicate,
    "tests/data/basic.parquet",
    [
        "SELECT id FROM t WHERE name ILIKE 'NAME_1' ORDER BY id",
        "SELECT id FROM t WHERE name ILIKE 'Name_1%' ORDER BY id LIMIT 20",
        "SELECT id FROM t WHERE name NOT ILIKE 'NAME_1%' ORDER BY id LIMIT 5",
    ]
);

e2e!(
    try_cast_expr,
    "tests/data/basic.parquet",
    [
        // 数字でない文字列は CAST ならエラーだが TRY_CAST は NULL。
        "SELECT TRY_CAST(name AS INTEGER) FROM t ORDER BY id LIMIT 5",
        // 変換できる場合は普通の CAST と同じ結果になる。
        "SELECT TRY_CAST(CAST(id AS VARCHAR) AS INTEGER) FROM t ORDER BY id LIMIT 5",
        "SELECT TRY_CAST(big AS VARCHAR) FROM t ORDER BY id LIMIT 5",
    ]
);

e2e!(
    distinct_on_clause,
    "tests/data/basic.parquet",
    [
        // 各 flag グループで id が最大の行だけを残す。
        "SELECT DISTINCT ON (flag) flag, id FROM t ORDER BY flag, id DESC",
        // ON に無い列（`*` の一部）も普通に出力される。
        "SELECT DISTINCT ON (flag) * FROM t ORDER BY flag, id",
    ]
);

/// `ORDER BY` を省略した `DISTINCT ON` は「到着順で最初の行」を残す。
/// DuckDB も同じ規則だが、どちらを「到着順の先頭」とみなすかはスキャンの
/// 実装依存（ページ読み出し順など）で、別エンジン同士を突き合わせる意味が
/// 無い。ここでは行数（= ON キーの種類数）と、返る値が実在するグループの
/// 値であることだけ確認する。
#[test]
fn distinct_on_without_order_by_keeps_first_arrival_per_key() {
    let files = ["tests/data/basic.parquet"];
    let rows = body(run_ahiru(&files, "SELECT DISTINCT ON (flag) flag FROM t"));
    let mut vals: Vec<&str> = rows.iter().map(|r| r[0].as_str()).collect();
    vals.sort_unstable();
    assert_eq!(vals, vec!["false", "true"], "flag は 2 値なので 2 行になるはず: {rows:?}");
}

/// `IIF` は `CASE WHEN` への脱糖なので、この環境の DuckDB に `iif` が
/// 無くても等価な `CASE WHEN` と突き合わせれば検証できる。
#[test]
fn iif_desugars_like_case_when() {
    if !duckdb_available() {
        eprintln!("duckdb が無いので飛ばす");
        return;
    }
    let cases = [
        (
            "SELECT id, IIF(flag, 'yes', 'no') FROM t ORDER BY id LIMIT 10",
            "SELECT id, CASE WHEN flag THEN 'yes' ELSE 'no' END FROM t ORDER BY id LIMIT 10",
        ),
        (
            "SELECT IIF(id < 5, id, -1) FROM t ORDER BY id LIMIT 10",
            "SELECT CASE WHEN id < 5 THEN id ELSE -1 END FROM t ORDER BY id LIMIT 10",
        ),
        (
            "SELECT IIF(big IS NULL, 0, big) FROM t ORDER BY id LIMIT 10",
            "SELECT CASE WHEN big IS NULL THEN 0 ELSE big END FROM t ORDER BY id LIMIT 10",
        ),
    ];
    let files = ["tests/data/basic.parquet"];
    for (ahiru_sql, duckdb_sql) in cases {
        let a = body(run_ahiru(&files, ahiru_sql));
        let d = body(run_duckdb(&files, duckdb_sql));
        assert_eq!(a, d, "IIF: {ahiru_sql}\nCASE WHEN: {duckdb_sql}");
    }
}

#[test]
fn table_name_replacement_respects_word_boundaries() {
    // `t2` や `text` の中の t を書き換えてしまうと、比較対象の SQL が壊れる。
    let f = ["a.parquet", "b.parquet"];
    let got = replace_tables("SELECT text, t.a FROM t JOIN t2 ON t.a = t2.a", &f);
    // `text` の先頭の t を置換していないこと。
    assert!(got.contains("text,"), "text が壊れている: {got}");
    // `t` は 3 箇所、`t2` は 2 箇所。数が合っていれば取り違えていない。
    assert_eq!(got.matches("a.parquet").count(), 3, "{got}");
    assert_eq!(got.matches("b.parquet").count(), 2, "{got}");
    // 置換後に裸の `t` / `t2` が残っていないこと。
    assert!(!got.split_whitespace().any(|w| w == "t" || w == "t2"), "{got}");
}

#[test]
fn cell_normalisation() {
    assert_eq!(normalize_cell(""), "<null>");
    assert_eq!(normalize_cell("NULL"), "<null>");
    // 浮動小数の表記ゆれを吸収する。
    assert_eq!(normalize_cell("747.0"), normalize_cell("747"));
    assert_eq!(normalize_cell("1.5"), normalize_cell("1.50"));
    assert_ne!(normalize_cell("1.5"), normalize_cell("1.6"));
    assert_eq!(normalize_cell("\"abc\""), "abc");
}
