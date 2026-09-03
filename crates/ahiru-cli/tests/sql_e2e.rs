//! End-to-end SQL regression tests.
//!
//! **Cross-check against DuckDB instead of writing expected values.** Hand-written
//! expectations turn typos into the specification, and every new query needs manual
//! computation. With DuckDB as the reference implementation, one more query line means one more verified case.
//!
//! Environments without DuckDB skip these tests entirely (CI installs it).

use std::process::Command;

fn duckdb_available() -> bool {
    Command::new("duckdb").arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn repo_path(rel: &str) -> String {
    format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel)
}

/// Runs through the ahirudb CLI and splits the output into rows x columns (header included).
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

/// Runs the same query through DuckDB, replacing the table names `t`, `t2`, ... with file references.
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

/// Substitutes `t`, `t2`, `t3`, ... with their respective file references.
///
/// Matching is done on word boundaries, so the `t` inside `text`, and cases where
/// `t` is part of `t2`, are not broken.
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

/// Normalizes a cell into a comparable form.
///
/// - NULL is spelled differently (ahirudb writes `NULL`; DuckDB's CSV writes empty)
/// - floats are formatted differently (`747` vs `747.0`)
fn normalize_cell(s: &str) -> String {
    let s = s.trim().trim_matches('"');
    if s.is_empty() || s == "NULL" {
        return "<null>".into();
    }
    if let Ok(f) = s.parse::<f64>() {
        // Values representable as integers are normalized to integer form.
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

/// Runs one query through both and cross-checks the results.
fn check_multi(files: &[&str], sql: &str) {
    // Headers are not compared. Alias naming is not kept in sync, and DuckDB omits
    // the header entirely when the result has 0 rows. Only data rows are compared.
    let a = body(run_ahiru(files, sql));
    let d = body(run_duckdb(files, sql));
    assert_eq!(a.len(), d.len(), "row counts differ\nSQL: {sql}\nahiru: {a:?}\nduckdb: {d:?}");
    for (i, (ra, rd)) in a.iter().zip(&d).enumerate() {
        assert_eq!(ra, rd, "row {} differs\nSQL: {sql}", i + 1);
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
                eprintln!("duckdb not found, skipping");
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
                eprintln!("duckdb not found, skipping");
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
    // Arithmetic involving NULL yields NULL.
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
        // An aggregate over zero surviving rows.
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
        // Ordering by an expression absent from the output (a hidden sort column).
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

// GZIP decompression is delegated to the host (DESIGN.md §6). In the CLI the system
// `gzip` fills that role. If this passes, the codec delegation protocol works end to end.
//
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
        // Results are unchanged even where statistics pruning applies.
        "SELECT count(*) FROM t WHERE id > 40000",
        "SELECT id FROM t WHERE id BETWEEN 8190 AND 8195 ORDER BY id",
    ]
);

// Page-level pruning (ColumnIndex/OffsetIndex/Bloom filter).
// pagetest.parquet was written by pyarrow (parquet-cpp) with ColumnIndex/OffsetIndex/Bloom
// filters, with id ascending over 0..50000 and small pages (the DuckDB version in
// this environment cannot write Bloom filters, hence pyarrow; `scripts/gen-testdata.sh`
// records which tool built what and how).
// DuckDB can read this file, so result correctness is cross-checked as usual.
// Whether pruning actually happens at page granularity is covered by
// `selective_equality_predicate_reads_a_small_fraction_of_the_column` in
// `crates/ahiru-core/src/format/parquet.rs`.
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

// Joins. The DuckDB side needs aliases on its file references too, so the queries
// always spell out explicit aliases.
// Derived tables in the FROM clause, filtering an aggregate result further.
e2e!(derived_tables, "tests/data/basic.parquet", [
    "SELECT s.name, s.n FROM (SELECT name, count(*) AS n FROM t GROUP BY name) AS s WHERE s.n > 142 ORDER BY s.name",
    "SELECT count(*) FROM (SELECT id FROM t WHERE id < 10) AS s",
    "SELECT max(s.m) FROM (SELECT name, max(big) AS m FROM t GROUP BY name) AS s",
]);

// ZSTD is part of `ahiru-core`'s default features (`zstd`) and the core decompresses
// it in-process (DESIGN.md §6). The CLI links `ahiru-zstd` directly, but that is the
// host-delegation fallback for builds without the `zstd` feature, and the default
// build never goes through it.
e2e!(
    zstd_is_decompressed_by_the_core,
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
    // Conditions that do not reduce to equality are evaluated as residual predicates.
    "SELECT a.k, b.k FROM t AS a JOIN t2 AS b ON a.k < b.k ORDER BY a.k, b.k",
    // A one-sided condition on top of the join key (a pushdown candidate).
    "SELECT a.k, b.w FROM t AS a JOIN t2 AS b ON a.k = b.k WHERE a.v > 4 ORDER BY a.k",
    // Aggregation on top of a join.
    "SELECT count(*) FROM t AS a JOIN t2 AS b ON a.k = b.k",
    "SELECT a.k, sum(b.w) FROM t AS a JOIN t2 AS b ON a.k = b.k GROUP BY a.k ORDER BY a.k",
]);

// Joining a large table with a small one. Projection pushdown and pruning must work under a join too.
e2e_multi!(join_with_dimension, ["tests/data/basic.parquet", "tests/data/dim.parquet"], [
    "SELECT count(*) FROM t AS f JOIN t2 AS d ON f.name = d.label",
    "SELECT d.nid, count(*) FROM t AS f JOIN t2 AS d ON f.name = d.label GROUP BY d.nid ORDER BY d.nid",
    "SELECT f.id, d.w FROM t AS f JOIN t2 AS d ON f.name = d.label WHERE f.id < 5 ORDER BY f.id",
]);

// --- SQL sugar (FILTER / QUALIFY / ILIKE / TRY_CAST / IIF / DISTINCT ON) ------

e2e!(
    filter_clause,
    "tests/data/basic.parquet",
    [
        "SELECT count(*) FILTER (WHERE id > 500) FROM t",
        "SELECT count(*) FILTER (WHERE id > 500), count(*) FROM t",
        "SELECT flag, count(*) FILTER (WHERE big IS NOT NULL) FROM t GROUP BY flag ORDER BY flag",
        "SELECT sum(id) FILTER (WHERE flag), sum(id) FILTER (WHERE NOT flag) FROM t",
        // An aggregate whose rows are all filtered out gives NULL for SUM and 0 for COUNT(*).
        "SELECT count(*) FILTER (WHERE id > 100000), sum(id) FILTER (WHERE id > 100000) FROM t",
        // Mixing with aggregates that have no FILTER leaves both unaffected.
        "SELECT count(*), count(*) FILTER (WHERE flag) FROM t GROUP BY name ORDER BY name LIMIT 5",
    ]
);

e2e!(
    qualify_clause,
    "tests/data/basic.parquet",
    [
        // A window function written directly in QUALIFY.
        // Window functions do not reorder rows, so cross-checking output order needs an
        // explicit ORDER BY (QUALIFY itself plays no part in ordering).
        "SELECT id FROM t QUALIFY row_number() OVER (ORDER BY id DESC) <= 3 ORDER BY id DESC",
        // Referring to a SELECT alias from QUALIFY.
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

// A conjunct that predicate pushdown consumes into the scan's pruner
// (`id = <literal>`) next to one it cannot (a scalar function call, `ILIKE`'s
// implicit `lower()` calls) gets compiled as two separate programs and merged
// by `plan::compile::and_programs`. That merge used to drop the right-hand
// side's `calls`/`lambdas` tables, so the residual predicate either failed
// with `Internal` or silently ran the wrong function. Conjunct order matters
// here: with the pushdown-able side first the call table being merged into is
// empty, which is exactly the case that broke — so each query is listed in
// both orders.
e2e!(
    pushdown_residual_predicate_with_function_calls,
    "tests/data/basic.parquet",
    [
        "SELECT id, name FROM t WHERE id = 1 AND upper(name) = 'NAME_1' ORDER BY id",
        "SELECT id, name FROM t WHERE upper(name) = 'NAME_1' AND id = 1 ORDER BY id",
        "SELECT id, name FROM t WHERE id = 1 AND name ILIKE 'name_1' ORDER BY id",
        "SELECT id, name FROM t WHERE name ILIKE 'name_1' AND id = 1 ORDER BY id",
        "SELECT id, name FROM t WHERE big = 100 AND name ILIKE 'N%' ORDER BY id",
        "SELECT id, name FROM t WHERE name ILIKE 'N%' AND big = 100 ORDER BY id",
        // Function calls on both sides: the two call tables must stay distinct
        // rather than the second one aliasing the first.
        "SELECT id, name FROM t WHERE lower(name) = 'name_1' AND upper(name) = 'NAME_1' ORDER BY id",
        "SELECT id, name FROM t WHERE upper(name) = 'NAME_1' AND lower(name) = 'name_1' ORDER BY id",
        // Three conjuncts, so the merge happens twice and the second merge sees
        // a non-empty left-hand call table.
        "SELECT id, name FROM t WHERE id = 1 AND upper(name) = 'NAME_1' AND lower(name) = 'name_1' ORDER BY id",
        // Predicates the pruner never consumed still work (guarding against a
        // fix that just turns pushdown off).
        "SELECT id FROM t WHERE id = 1 ORDER BY id",
        "SELECT id FROM t WHERE upper(name) = 'NAME_1' ORDER BY id",
        "SELECT id FROM t WHERE id = 1 AND name <> 'x' ORDER BY id",
        "SELECT id FROM t WHERE id = 1 AND name LIKE 'name%' ORDER BY id",
    ]
);

// The bug lives in the expression compiler, not in any one format's scan, so
// it reproduces on CSV too (`tests/data/basic.csv` is the same data).
e2e!(
    pushdown_residual_predicate_with_function_calls_csv,
    "tests/data/basic.csv",
    [
        "SELECT id, name FROM t WHERE id = 1 AND upper(name) = 'NAME_1' ORDER BY id",
        "SELECT id, name FROM t WHERE upper(name) = 'NAME_1' AND id = 1 ORDER BY id",
        "SELECT id, name FROM t WHERE id = 1 AND name ILIKE 'name_1' ORDER BY id",
        "SELECT id, name FROM t WHERE lower(name) = 'name_1' AND upper(name) = 'NAME_1' ORDER BY id",
    ]
);

e2e!(
    try_cast_expr,
    "tests/data/basic.parquet",
    [
        // A non-numeric string is an error under CAST but NULL under TRY_CAST.
        "SELECT TRY_CAST(name AS INTEGER) FROM t ORDER BY id LIMIT 5",
        // When the conversion succeeds the result matches a plain CAST.
        "SELECT TRY_CAST(CAST(id AS VARCHAR) AS INTEGER) FROM t ORDER BY id LIMIT 5",
        "SELECT TRY_CAST(big AS VARCHAR) FROM t ORDER BY id LIMIT 5",
    ]
);

e2e!(
    distinct_on_clause,
    "tests/data/basic.parquet",
    [
        // Keeps only the row with the largest id in each flag group.
        "SELECT DISTINCT ON (flag) flag, id FROM t ORDER BY flag, id DESC",
        // Columns not in ON (part of `*`) are output normally.
        "SELECT DISTINCT ON (flag) * FROM t ORDER BY flag, id",
    ]
);

// DuckDB's `COLUMNS(...)` star expression. Headers are not compared by this
// harness, so the value here is in the row *content* and column order —
// which is exactly where the two easy ways to get `COLUMNS` wrong show up
// (expanding the explicit list in list order instead of schema order, and
// treating the regex as an anchored full match instead of a search).
e2e!(
    columns_star_expression,
    "tests/data/basic.parquet",
    [
        "SELECT COLUMNS(*) FROM t ORDER BY id LIMIT 3",
        "SELECT COLUMNS(* EXCLUDE (score, big, d)) FROM t ORDER BY id LIMIT 3",
        "SELECT COLUMNS(* REPLACE (score * 2 AS score)) FROM t ORDER BY id LIMIT 3",
        // Unanchored search: `a` matches `name` and `flag`.
        "SELECT COLUMNS('a') FROM t ORDER BY id LIMIT 3",
        "SELECT COLUMNS('^s') FROM t ORDER BY id LIMIT 3",
        // Schema order, not list order.
        "SELECT COLUMNS(['score', 'id']) FROM t ORDER BY id LIMIT 3",
        "SELECT COLUMNS(['ID']) FROM t ORDER BY id LIMIT 3",
        // Capture-group renaming; the 1-character `d` column doesn't match.
        r"SELECT COLUMNS('(\w{2}).*') AS '\1' FROM t ORDER BY id LIMIT 3",
        "SELECT COLUMNS(*) AS 'x_\\0' FROM t ORDER BY id LIMIT 3",
        "SELECT 1 AS z, COLUMNS(['id', 'name']) FROM t ORDER BY id LIMIT 3",
    ]
);

/// `DISTINCT ON` without `ORDER BY` keeps "the first row in arrival order".
/// DuckDB follows the same rule, but which row counts as "first in arrival order"
/// depends on the scan implementation (page read order and so on), so cross-checking
/// two engines is meaningless. This only checks the row count (= the number of
/// distinct ON keys) and that the returned value belongs to a group that exists.
#[test]
fn distinct_on_without_order_by_keeps_first_arrival_per_key() {
    let files = ["tests/data/basic.parquet"];
    let rows = body(run_ahiru(&files, "SELECT DISTINCT ON (flag) flag FROM t"));
    let mut vals: Vec<&str> = rows.iter().map(|r| r[0].as_str()).collect();
    vals.sort_unstable();
    assert_eq!(
        vals,
        vec!["false", "true"],
        "flag is two-valued, so there should be 2 rows: {rows:?}"
    );
}

/// `IIF` desugars to `CASE WHEN`, so it can be verified against an equivalent
/// `CASE WHEN` even if this environment's DuckDB has no `iif`.
#[test]
fn iif_desugars_like_case_when() {
    if !duckdb_available() {
        eprintln!("duckdb not found, skipping");
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

// --- DuckDB expression sugar (typed literals / `^@` / IS UNKNOWN / SQL
// --- standard function syntax / GROUP BY ALL / ORDER BY ALL) ---------------

// Typed time literals. `d` is a TIMESTAMP column, so comparisons and arithmetic
// against `DATE`/`TIMESTAMP` literals can be cross-checked directly.
e2e!(
    typed_temporal_literals,
    "tests/data/basic.parquet",
    [
        "SELECT id FROM t WHERE d >= TIMESTAMP '2024-01-05 00:00:00' AND d < TIMESTAMP '2024-01-09 00:00:00' ORDER BY id",
        "SELECT id FROM t WHERE CAST(d AS DATE) = DATE '2024-01-03' ORDER BY id",
        "SELECT count(*) FROM t WHERE d > TIMESTAMP '2024-06-01 00:00:00'",
        "SELECT DATE '2020-02-29' AS a, TIME '01:02:03' AS b FROM t LIMIT 1",
        // A form that rides on constant pruning (the result is of course unchanged).
        "SELECT id, d FROM t WHERE d = TIMESTAMP '2024-01-10 00:00:00'",
    ]
);

e2e!(
    prefix_operator_and_is_unknown,
    "tests/data/basic.parquet",
    [
        "SELECT id FROM t WHERE name ^@ 'name_1' ORDER BY id LIMIT 10",
        "SELECT count(*) FROM t WHERE NOT name ^@ 'name_'",
        "SELECT id, name ^@ 'name_0' FROM t ORDER BY id LIMIT 5",
        // NULL propagates through as NULL.
        "SELECT id, big IS UNKNOWN, big IS NOT UNKNOWN FROM t ORDER BY id LIMIT 6",
        "SELECT count(*) FROM t WHERE big IS UNKNOWN",
    ]
);

e2e!(
    sql_standard_function_syntax,
    "tests/data/basic.parquet",
    [
        "SELECT id, position('_' IN name) FROM t ORDER BY id LIMIT 5",
        "SELECT count(*) FROM t WHERE position('9' IN name) > 0",
        "SELECT substring(name FROM 6) FROM t ORDER BY id LIMIT 5",
        "SELECT substring(name FROM 1 FOR 4) FROM t ORDER BY id LIMIT 5",
        "SELECT substring(name FOR 4) FROM t ORDER BY id LIMIT 5",
        "SELECT trim(BOTH 'e' FROM name) FROM t ORDER BY id LIMIT 5",
        "SELECT trim(LEADING 'n' FROM name) FROM t ORDER BY id LIMIT 5",
        "SELECT trim(TRAILING '0' FROM name) FROM t ORDER BY id LIMIT 5",
        "SELECT trim(FROM name) FROM t ORDER BY id LIMIT 5",
    ]
);

e2e!(
    group_by_all_and_order_by_all,
    "tests/data/basic.parquet",
    [
        "SELECT name, count(*) FROM t GROUP BY ALL ORDER BY ALL",
        "SELECT flag, name, count(*) FROM t GROUP BY ALL ORDER BY ALL",
        // Expressions containing an aggregate are excluded from GROUP BY ALL.
        "SELECT id % 3, sum(id) + 1 FROM t GROUP BY ALL ORDER BY ALL",
        // With no aggregates at all it behaves like DISTINCT.
        "SELECT name FROM t GROUP BY ALL ORDER BY ALL",
        // With only aggregates there are zero grouping columns.
        "SELECT count(*), sum(id) FROM t GROUP BY ALL",
        "SELECT id, name FROM t ORDER BY ALL LIMIT 5",
        "SELECT id, name FROM t ORDER BY ALL DESC LIMIT 5",
        // Ordering must agree even for columns containing NULL.
        "SELECT big FROM t ORDER BY ALL LIMIT 8",
        "SELECT big FROM t ORDER BY ALL NULLS FIRST LIMIT 8",
        // The expansion of `*` is subject to ORDER BY ALL too.
        "SELECT * FROM t ORDER BY ALL LIMIT 5",
        // It applies to the whole result of a set operation.
        "SELECT id FROM t WHERE id < 3 UNION ALL SELECT id FROM t WHERE id < 2 ORDER BY ALL",
    ]
);

// `%` in a LIKE pattern used to be matched literally whenever the subject byte
// at the same position was `%` too -- and, because the wildcard branch was then
// skipped, no backtrack point was recorded either, so `'50% off' LIKE '50%'`
// came back false.
e2e!(
    like_percent_against_percent_in_the_subject,
    "tests/data/basic.parquet",
    [
        "SELECT '50% off' LIKE '50%', '%abc' LIKE '%bc', '%%' LIKE '%', '%ABC' ILIKE '%bc' FROM t LIMIT 1",
        "SELECT '50% off' LIKE '50!%%' ESCAPE '!', '50% off' LIKE '50!%' ESCAPE '!', '%abc' LIKE '!%%' ESCAPE '!', 'xabc' LIKE '!%%' ESCAPE '!' FROM t LIMIT 1",
        // `_` and an escaped `_` still behave, next to the reordered `%` branch.
        // (Non-ASCII literals are avoided here: `replace_tables` rebuilds the SQL
        // byte-by-byte as `char`, which mangles multibyte text before DuckDB sees
        // it. The multibyte cases are covered by
        // `ahiru-core/tests/kernel_type_fixes.rs` instead.)
        "SELECT '%a' LIKE '%_', 'a_c' LIKE 'a_c', 'abc' LIKE 'a!_c' ESCAPE '!', 'a_c' LIKE 'a!_c' ESCAPE '!' FROM t LIMIT 1",
    ]
);

// A signed and an unsigned integer type of the same width shared a rank in
// `Ty::unify`, so the tie-break kept the left operand's type and cast the other
// side's valid values to NULL. They now widen to the smallest signed type that
// covers both, whichever order they are written in.
e2e!(
    signed_and_unsigned_integer_mixing,
    "tests/data/basic.parquet",
    [
        "SELECT 0::UTINYINT + (-1)::TINYINT, 1::UBIGINT = (-1)::BIGINT, 1::UBIGINT > (-1)::BIGINT FROM t LIMIT 1",
        "SELECT 100::TINYINT + 200::UTINYINT, 200::UTINYINT + 100::TINYINT FROM t LIMIT 1",
        "SELECT 1::UBIGINT < (-1)::BIGINT, (-1)::BIGINT < 1::UBIGINT FROM t LIMIT 1",
        "SELECT (-1)::TINYINT IN (255::UTINYINT, (-1)::TINYINT), 300::USMALLINT IN (300::SMALLINT) FROM t LIMIT 1",
        "SELECT 18446744073709551615::UBIGINT + 0::BIGINT, (-1)::INTEGER + 1::UINTEGER FROM t LIMIT 1",
    ]
);

// DECIMAL `*` adds the operands' scales, but the VM used to relabel the result
// with `Ty::unify`'s `max(s1, s2)` -- rendering the raw product 10^min(s1,s2)
// times too large everywhere the text form is produced.
e2e!(
    decimal_multiplication_scale,
    "tests/data/basic.parquet",
    [
        "SELECT (1.5::DECIMAL(4,1) * 1.25::DECIMAL(3,2))::VARCHAR FROM t LIMIT 1",
        "SELECT (1.5::DECIMAL(4,1) * 2)::VARCHAR, (2.5::DECIMAL(4,1) * 0.5::DECIMAL(3,2))::VARCHAR FROM t LIMIT 1",
        // Precision past 38 still just clamps, as in DuckDB.
        "SELECT (1.5::DECIMAL(20,2) * 2.5::DECIMAL(19,2))::VARCHAR FROM t LIMIT 1",
    ]
);

#[test]
fn table_name_replacement_respects_word_boundaries() {
    // Rewriting the t inside `t2` or `text` would break the SQL being compared.
    let f = ["a.parquet", "b.parquet"];
    let got = replace_tables("SELECT text, t.a FROM t JOIN t2 ON t.a = t2.a", &f);
    // The leading t of `text` must not have been replaced.
    assert!(got.contains("text,"), "text was mangled: {got}");
    // `t` appears 3 times and `t2` twice. Matching counts means nothing was mixed up.
    assert_eq!(got.matches("a.parquet").count(), 3, "{got}");
    assert_eq!(got.matches("b.parquet").count(), 2, "{got}");
    // No bare `t` / `t2` may remain after substitution.
    assert!(!got.split_whitespace().any(|w| w == "t" || w == "t2"), "{got}");
}

#[test]
fn cell_normalisation() {
    assert_eq!(normalize_cell(""), "<null>");
    assert_eq!(normalize_cell("NULL"), "<null>");
    // Absorbs differences in float formatting.
    assert_eq!(normalize_cell("747.0"), normalize_cell("747"));
    assert_eq!(normalize_cell("1.5"), normalize_cell("1.50"));
    assert_ne!(normalize_cell("1.5"), normalize_cell("1.6"));
    assert_eq!(normalize_cell("\"abc\""), "abc");
}
