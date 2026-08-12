//! End-to-end tests for DuckDB's `COLUMNS(...)` star expression
//! (<https://duckdb.org/docs/lts/sql/expressions/star>).
//!
//! Every expected value below is cross-checked against a real
//! `duckdb` v1.4.4 CLI run over the same file — `tests/data/basic.parquet`
//! was written by DuckDB and has columns
//! `id INTEGER, name VARCHAR, score DOUBLE, flag BOOLEAN, big BIGINT,
//! d TIMESTAMP`. Read-only Parquet is all this needs, so it runs under a
//! plain default-feature `cargo test`.

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

// --- COLUMNS(*) --------------------------------------------------------------

/// `COLUMNS(*)` is exactly a plain `*`.
/// duckdb: SELECT COLUMNS(*) FROM 'basic.parquet' WHERE id < 2 ORDER BY id
#[test]
fn columns_star_expands_to_every_column() {
    let mut s = session_with_basic();
    assert_eq!(
        schema_names(&mut s, "SELECT COLUMNS(*) FROM t"),
        vec!["id", "name", "score", "flag", "big", "d"]
    );
    let rows = run(&mut s, "SELECT COLUMNS(*) FROM t WHERE id < 2 ORDER BY id");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[1][0], i32(1));
    assert_eq!(rows[1][1], vc("name_1"));
    assert_eq!(rows[1][2], f64(1.5));
}

/// The star modifiers go *inside* the parentheses for `COLUMNS(...)` — that
/// is the only placement `duckdb` accepts (`COLUMNS(*) EXCLUDE (...)` is a
/// parser error there) — and they behave exactly as they do on a bare `*`.
/// duckdb: SELECT COLUMNS(* EXCLUDE (score, big, d)) FROM 'basic.parquet'
///         WHERE id < 2 ORDER BY id
#[test]
fn columns_star_supports_the_star_modifiers_inside_the_parens() {
    let mut s = session_with_basic();
    assert_eq!(
        schema_names(&mut s, "SELECT COLUMNS(* EXCLUDE (score, big, d)) FROM t"),
        vec!["id", "name", "flag"]
    );
    let rows =
        run(&mut s, "SELECT COLUMNS(* EXCLUDE (score, big, d)) FROM t WHERE id < 2 ORDER BY id");
    assert_eq!(
        rows,
        vec![vec![i32(0), vc("name_0"), b(true)], vec![i32(1), vc("name_1"), b(false)]]
    );

    assert_eq!(
        schema_names(&mut s, "SELECT COLUMNS(* RENAME (score AS points)) FROM t"),
        vec!["id", "name", "points", "flag", "big", "d"]
    );
    let rows2 = run(
        &mut s,
        "SELECT id, score FROM (SELECT COLUMNS(* REPLACE (score * 2 AS score)) FROM t) \
         WHERE id < 2 ORDER BY id",
    );
    assert_eq!(rows2, vec![vec![i32(0), f64(0.0)], vec![i32(1), f64(3.0)]]);
}

/// Aggregation makes `*` unexpandable (the original rows are gone); adding
/// `COLUMNS` around it doesn't change that.
#[test]
fn columns_star_after_aggregation_is_still_rejected() {
    let mut s = session_with_basic();
    let err = s.prepare("SELECT COLUMNS(*) FROM t GROUP BY flag", &[]);
    assert_eq!(code_of(err), Some(Code::NotGrouped));
}

// --- COLUMNS('regex') --------------------------------------------------------

/// The regex is an unanchored search over each column name.
/// duckdb: SELECT COLUMNS('a') FROM 'basic.parquet' WHERE id < 2 ORDER BY id
///         -- name, flag
#[test]
fn columns_regex_matches_column_names_as_a_search() {
    let mut s = session_with_basic();
    assert_eq!(schema_names(&mut s, "SELECT COLUMNS('a') FROM t"), vec!["name", "flag"]);
    let rows = run(&mut s, "SELECT COLUMNS('a') FROM t WHERE id < 2 ORDER BY id");
    assert_eq!(rows, vec![vec![vc("name_0"), b(true)], vec![vc("name_1"), b(false)]]);
    // Anchors work the way they do everywhere else in `expr::regex`.
    assert_eq!(schema_names(&mut s, "SELECT COLUMNS('^s') FROM t"), vec!["score"]);
    assert_eq!(schema_names(&mut s, "SELECT COLUMNS('g$') FROM t"), vec!["flag", "big"]);
}

/// A regex that matches no column is an error, not an empty result
/// (`duckdb`: "No matching columns found that match regex"). The same goes
/// for a match that only fails because of case — the regex comparison is
/// case-sensitive, unlike ordinary column-name resolution here.
#[test]
fn columns_regex_matching_nothing_is_rejected() {
    let mut s = session_with_basic();
    assert_eq!(code_of(s.prepare("SELECT COLUMNS('zzz') FROM t", &[])), Some(Code::ColumnNotFound));
    assert_eq!(
        code_of(s.prepare("SELECT COLUMNS('NAME') FROM t", &[])),
        Some(Code::ColumnNotFound)
    );
}

/// A `COLUMNS('regex')` expansion narrows projection pushdown to the columns
/// it actually selects, rather than falling back to reading everything the
/// way a bare `*` has to (DESIGN.md §2: never read a byte that isn't needed).
#[test]
fn columns_regex_narrows_the_scan_projection() {
    let mut s = session_with_basic();
    let plan = run(&mut s, "EXPLAIN SELECT COLUMNS('a') FROM t");
    let text: Vec<String> = plan
        .iter()
        .map(|r| match &r[0] {
            Value::Bytes(b) => String::from_utf8_lossy(b).into_owned(),
            v => panic!("unexpected plan value {v:?}"),
        })
        .collect();
    assert!(
        text.iter().any(|l| l.contains("Scan") && l.contains("columns=2")),
        "expected a 2-column scan, got {text:?}"
    );
}

// --- COLUMNS(['a', 'b']) -----------------------------------------------------

/// The explicit list expands in *schema* order, not list order, and matches
/// names case-insensitively.
/// duckdb: SELECT COLUMNS(['score', 'id']) FROM 'basic.parquet' WHERE id < 2
///         ORDER BY id   -- id, score
#[test]
fn columns_list_expands_in_schema_order() {
    let mut s = session_with_basic();
    assert_eq!(schema_names(&mut s, "SELECT COLUMNS(['score', 'id']) FROM t"), vec!["id", "score"]);
    assert_eq!(schema_names(&mut s, "SELECT COLUMNS(['ID']) FROM t"), vec!["id"]);
    let rows = run(&mut s, "SELECT COLUMNS(['score', 'id']) FROM t WHERE id < 2 ORDER BY id");
    assert_eq!(rows, vec![vec![i32(0), f64(0.0)], vec![i32(1), f64(1.5)]]);
}

/// A listed name that no column has is an error — `COLUMNS` follows
/// `EXCLUDE` here, not `RENAME` (which silently ignores unknown names).
/// duckdb: Binder Error: Column "nope" was selected but was not found in the
/// FROM clause.
#[test]
fn columns_list_with_an_unknown_name_is_rejected() {
    let mut s = session_with_basic();
    assert_eq!(
        code_of(s.prepare("SELECT COLUMNS(['id', 'nope']) FROM t", &[])),
        Some(Code::ColumnNotFound)
    );
}

// --- AS '<template>' ---------------------------------------------------------

/// `COLUMNS('regex') AS '\1'` renames each expanded column from the pattern's
/// capture groups.
/// duckdb: SELECT COLUMNS('(\w{2}).*') AS '\1' FROM 'basic.parquet' WHERE id < 2
///         -- id, na, sc, fl, bi  (the 1-character `d` column doesn't match)
#[test]
fn columns_regex_alias_template_renames_from_capture_groups() {
    let mut s = session_with_basic();
    assert_eq!(
        schema_names(&mut s, "SELECT COLUMNS('(\\w{2}).*') AS '\\1' FROM t"),
        vec!["id", "na", "sc", "fl", "bi"]
    );
    let rows = run(&mut s, "SELECT COLUMNS('(\\w{2}).*') AS '\\1' FROM t WHERE id < 2 ORDER BY id");
    assert_eq!(
        rows,
        vec![
            vec![i32(0), vc("name_0"), f64(0.0), b(true), Value::Null],
            vec![i32(1), vc("name_1"), f64(1.5), b(false), Value::I64(100)],
        ]
    );
}

/// `\0` is the whole column name (not the matched substring), and the
/// template works on the `COLUMNS(*)` / `COLUMNS([...])` forms too.
/// duckdb: SELECT COLUMNS(*) AS 'x_\0' FROM 'basic.parquet'
///         / SELECT COLUMNS('o') AS 'x_\0' FROM 'basic.parquet'  -- x_score
#[test]
fn columns_alias_template_backslash_zero_is_the_whole_column_name() {
    let mut s = session_with_basic();
    assert_eq!(
        schema_names(&mut s, "SELECT COLUMNS(*) AS 'x_\\0' FROM t"),
        vec!["x_id", "x_name", "x_score", "x_flag", "x_big", "x_d"]
    );
    assert_eq!(schema_names(&mut s, "SELECT COLUMNS('o') AS 'x_\\0' FROM t"), vec!["x_score"]);
    assert_eq!(
        schema_names(&mut s, "SELECT COLUMNS(['id', 'big']) AS 'c\\0' FROM t"),
        vec!["cid", "cbig"]
    );
}

/// A group the pattern doesn't have expands to the empty string; a template
/// that expands to nothing at all leaves the original name in place.
/// duckdb: SELECT COLUMNS('^n') AS 'q\1' / AS '\1' FROM 'basic.parquet'
#[test]
fn columns_alias_template_handles_a_missing_capture_group() {
    let mut s = session_with_basic();
    assert_eq!(schema_names(&mut s, "SELECT COLUMNS('^n') AS 'q\\1' FROM t"), vec!["q"]);
    assert_eq!(schema_names(&mut s, "SELECT COLUMNS('^n') AS '\\1' FROM t"), vec!["name"]);
}

/// A plain (unquoted) alias is just a template with no escapes, so every
/// expanded column gets that same name — matching `duckdb`, which produces
/// duplicate output names here rather than rejecting it.
#[test]
fn columns_plain_alias_names_every_expanded_column_the_same() {
    let mut s = session_with_basic();
    assert_eq!(schema_names(&mut s, "SELECT COLUMNS(['id', 'big']) AS x FROM t"), vec!["x", "x"]);
}

/// The `AS '<template>'` name wins over `RENAME` and is built from the
/// *original* column name.
/// duckdb: DESCRIBE SELECT COLUMNS(* RENAME (id AS ident)) AS 'x_\0'
///         FROM 'basic.parquet'  -- x_id, not x_ident
#[test]
fn columns_alias_template_overrides_rename() {
    let mut s = session_with_basic();
    assert_eq!(
        schema_names(&mut s, "SELECT COLUMNS(* RENAME (id AS ident)) AS 'x_\\0' FROM t"),
        vec!["x_id", "x_name", "x_score", "x_flag", "x_big", "x_d"]
    );
}

// --- combinations with the rest of the engine --------------------------------

/// A `COLUMNS(...)` item sits alongside ordinary select items, and its output
/// feeds an enclosing query like any other projection.
#[test]
fn columns_combines_with_other_select_items_and_subqueries() {
    let mut s = session_with_basic();
    assert_eq!(
        schema_names(&mut s, "SELECT 1 AS z, COLUMNS(['id', 'name']) FROM t"),
        vec!["z", "id", "name"]
    );
    let rows = run(
        &mut s,
        "SELECT count(*), sum(score) FROM (SELECT COLUMNS(['id', 'score']) FROM t) WHERE id < 4",
    );
    assert_eq!(rows, vec![vec![Value::I64(4), f64(9.0)]]);
}

/// An unqualified expansion covers both sides of a join, exactly like a bare
/// `*` does.
#[test]
fn columns_expands_across_both_sides_of_a_join() {
    let mut s = session_with_basic();
    assert_eq!(
        schema_names(&mut s, "SELECT COLUMNS(['id']) FROM t a JOIN t b ON a.id = b.id"),
        vec!["id", "id"]
    );
}

// --- explicitly unsupported forms --------------------------------------------
//
// Each of these is a real DuckDB feature this engine does not implement. They
// are checked here so they fail as a clear `UnsupportedFeature` rather than
// as a confusing token error or, worse, a wrong result.

/// Distributing an enclosing function over the expansion (`min(COLUMNS(*))`).
/// This does not fall out of the bind-time expansion: the binder would have
/// to synthesize one call node per expanded column, but the expression arena
/// is already immutable by the time the input schema is known, and aggregates
/// are collected from the raw AST in an earlier pass.
#[test]
fn function_distribution_over_columns_is_rejected() {
    let mut s = session_with_basic();
    assert_eq!(
        code_of(s.prepare("SELECT min(COLUMNS(*)) FROM t", &[])),
        Some(Code::UnsupportedFeature)
    );
    assert_eq!(
        code_of(s.prepare("SELECT COLUMNS(*) + 1 FROM t", &[])),
        Some(Code::UnsupportedFeature)
    );
}

/// `UNPACK(...)` and `*COLUMNS(...)`, DuckDB's two unpacking forms.
#[test]
fn unpacking_a_columns_expansion_is_rejected() {
    let mut s = session_with_basic();
    assert_eq!(
        code_of(s.prepare("SELECT UNPACK(COLUMNS(*)) FROM t", &[])),
        Some(Code::UnsupportedFeature)
    );
    assert_eq!(
        code_of(s.prepare("SELECT *COLUMNS(*) FROM t", &[])),
        Some(Code::UnsupportedFeature)
    );
}

/// The `COLUMNS(lambda)` predicate form.
#[test]
fn columns_lambda_form_is_rejected() {
    let mut s = session_with_basic();
    assert_eq!(
        code_of(s.prepare("SELECT COLUMNS(c -> c LIKE 'n%') FROM t", &[])),
        Some(Code::UnsupportedFeature)
    );
}

/// `* LIKE 'pat'` / `* GLOB` / `* SIMILAR TO` star filtering.
#[test]
fn star_filtering_operators_are_rejected() {
    let mut s = session_with_basic();
    for sql in [
        "SELECT * LIKE 'n%' FROM t",
        "SELECT * ILIKE 'n%' FROM t",
        "SELECT * NOT LIKE 'n%' FROM t",
        "SELECT * GLOB 'n*' FROM t",
        "SELECT * SIMILAR TO 'n.*' FROM t",
    ] {
        assert_eq!(code_of(s.prepare(sql, &[])), Some(Code::UnsupportedFeature), "{sql}");
    }
}

/// A table-qualified `t.COLUMNS(*)` — `duckdb` rejects this too ("Scalar
/// Function with name columns does not exist").
#[test]
fn qualified_columns_is_rejected() {
    let mut s = session_with_basic();
    assert_eq!(
        code_of(s.prepare("SELECT t.COLUMNS(*) FROM t", &[])),
        Some(Code::UnsupportedFeature)
    );
}

/// `COLUMNS` outside a select-list item has nowhere to expand to.
#[test]
fn columns_outside_the_select_list_is_rejected() {
    let mut s = session_with_basic();
    assert_eq!(
        code_of(s.prepare("SELECT id FROM t WHERE COLUMNS('id') > 1", &[])),
        Some(Code::UnsupportedFeature)
    );
    assert_eq!(
        code_of(s.prepare("SELECT id FROM t ORDER BY COLUMNS('id')", &[])),
        Some(Code::UnsupportedFeature)
    );
}

// --- `columns` is not a reserved word ----------------------------------------

/// The whole point of recognizing `COLUMNS` only at the start of a select
/// item, followed by `(`: a column named `columns` has to keep working
/// unquoted, since column names come from data files (`sql/lexer.rs`'s
/// `KEYWORDS` comment). `duckdb` behaves the same way. Uses a derived table
/// so this stays a Parquet-only test like the rest of the file.
#[test]
fn columns_keyword_does_not_break_a_columns_named_column() {
    let mut s = session_with_basic();
    assert_eq!(schema_names(&mut s, "SELECT id AS columns FROM t"), vec!["columns"]);
    let rows = run(
        &mut s,
        "SELECT columns FROM (SELECT id AS columns FROM t) WHERE columns < 2 ORDER BY columns",
    );
    assert_eq!(rows, vec![vec![i32(0)], vec![i32(1)]]);
    // Including as the sole select item, where the parser has to decide
    // between the star expression and a plain column reference.
    let rows2 = run(&mut s, "SELECT columns FROM (SELECT id AS columns FROM t) WHERE columns = 3");
    assert_eq!(rows2, vec![vec![i32(3)]]);
}
