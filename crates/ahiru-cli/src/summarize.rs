//! `SUMMARIZE <table>` support.
//!
//! Implemented as a rewrite into ordinary SQL rather than as an engine
//! feature: the aggregates it needs (`min`/`max`/`avg`/`stddev`/`median`/
//! `approx_count_distinct`/`count`) all exist already, and keeping the rewrite
//! in the CLI costs the wasm build nothing (`docs/DESIGN.md` §3).
//!
//! The output shape matches DuckDB's `SUMMARIZE` with one deliberate
//! omission: DuckDB also emits `q25`/`q75` (the 0.25/0.75 quantiles), but
//! this engine only has `median` (the 0.5 quantile) — there's no general
//! `quantile_cont`/`percentile` aggregate to build them from — so those two
//! columns are left out entirely rather than faked.

/// Recognises `SUMMARIZE <target>` and returns the target text as written
/// (a table name, a quoted path, a table function call, or a parenthesised
/// query). Leading whitespace and `--`/`/* */` comments before the keyword
/// are skipped; a trailing `;` is stripped. Returns `None` if the statement
/// doesn't start with `SUMMARIZE` (as a whole word) or if there's no target
/// text left after it.
pub fn parse(sql: &str) -> Option<String> {
    let s = skip_ws_and_comments(sql);

    // `get` (rather than slicing directly) avoids panicking if `s` is
    // shorter than the keyword or the byte offset lands mid-codepoint.
    let head = s.get(0..9)?;
    if !head.eq_ignore_ascii_case("SUMMARIZE") {
        return None;
    }
    let after = &s[9..];

    // Word boundary: `SUMMARIZED` or `SUMMARIZE_FOO` must not match.
    if let Some(c) = after.chars().next() {
        if c.is_ascii_alphanumeric() || c == '_' {
            return None;
        }
    }

    let mut target = after.trim();
    if let Some(stripped) = target.strip_suffix(';') {
        target = stripped.trim_end();
    }
    if target.is_empty() {
        return None;
    }
    Some(target.to_string())
}

/// Repeatedly strips leading whitespace, `-- ...` line comments and
/// `/* ... */` block comments from `s`. An unterminated block/line comment
/// consumes the rest of the input rather than panicking.
fn skip_ws_and_comments(mut s: &str) -> &str {
    loop {
        let trimmed = s.trim_start();
        if trimmed.len() != s.len() {
            s = trimmed;
            continue;
        }
        if let Some(rest) = s.strip_prefix("--") {
            s = match rest.find('\n') {
                Some(i) => &rest[i + 1..],
                None => "",
            };
            continue;
        }
        if let Some(rest) = s.strip_prefix("/*") {
            s = match rest.find("*/") {
                Some(i) => &rest[i + 2..],
                None => "",
            };
            continue;
        }
        break;
    }
    s
}

/// Column-name/type-name pairs from [`Ty::name`](ahiru_core::vector::Ty::name)
/// for which `avg`/`stddev`/`median` make sense. `type_name` may be a bare
/// name (`"BIGINT"`) or carry parameters (`"DECIMAL(10,2)"`); only the part
/// before `(` is checked.
fn is_numeric_type(type_name: &str) -> bool {
    let base = type_name.split('(').next().unwrap_or(type_name);
    matches!(
        base,
        "TINYINT"
            | "SMALLINT"
            | "INTEGER"
            | "BIGINT"
            | "HUGEINT"
            | "UTINYINT"
            | "USMALLINT"
            | "UINTEGER"
            | "UBIGINT"
            | "FLOAT"
            | "DOUBLE"
            | "DECIMAL"
    )
}

/// Double-quotes a SQL identifier, doubling any embedded `"`.
fn quote_ident(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 2);
    out.push('"');
    for c in name.chars() {
        if c == '"' {
            out.push('"');
        }
        out.push(c);
    }
    out.push('"');
    out
}

/// Single-quotes a SQL string literal, doubling any embedded `'`.
fn quote_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        if c == '\'' {
            out.push('\'');
        }
        out.push(c);
    }
    out.push('\'');
    out
}

/// Output columns, in order, of the SQL built by [`summarize_sql`].
const OUTPUT_ALIASES: [&str; 10] = [
    "column_name",
    "column_type",
    "min",
    "max",
    "approx_unique",
    "avg",
    "std",
    "q50",
    "count",
    "null_percentage",
];

/// Builds the SQL producing DuckDB's `SUMMARIZE` shape (minus `q25`/`q75`,
/// see the module doc) for `target`, whose columns are `(name, type_name,
/// nullable)`. `target` is pasted verbatim into a `FROM` clause, so it may
/// be a bare table name, a quoted path, a table function call, or a
/// parenthesised subquery.
pub fn summarize_sql(target: &str, columns: &[(String, String, bool)]) -> String {
    if columns.is_empty() {
        // No columns to describe: yield the right schema with zero rows,
        // without needing to touch `target` at all.
        let arms: Vec<String> = OUTPUT_ALIASES
            .iter()
            .map(|alias| {
                let ty = if *alias == "approx_unique" || *alias == "count" {
                    "BIGINT"
                } else {
                    "VARCHAR"
                };
                format!("CAST(NULL AS {ty}) AS {}", quote_ident(alias))
            })
            .collect();
        return format!("SELECT {} FROM range(1) WHERE false", arms.join(", "));
    }

    let arms: Vec<String> = columns
        .iter()
        .enumerate()
        .map(|(i, (name, ty, _nullable))| column_arm(name, ty, target, i == 0))
        .collect();
    arms.join("\nUNION ALL\n")
}

/// Builds one `SELECT ... FROM <target>` arm of the `UNION ALL` for a single
/// column. `first` controls whether the output columns carry `AS <alias>`
/// labels (only needed once, on the first arm — the rest are positional).
fn column_arm(name: &str, ty: &str, target: &str, first: bool) -> String {
    let ident = quote_ident(name);
    let numeric = is_numeric_type(ty);

    let (avg_expr, std_expr, q50_expr) = if numeric {
        (
            format!("CAST(round(avg({ident}), 2) AS VARCHAR)"),
            format!("CAST(round(stddev({ident}), 2) AS VARCHAR)"),
            format!("CAST(median({ident}) AS VARCHAR)"),
        )
    } else {
        let null_varchar = "CAST(NULL AS VARCHAR)".to_string();
        (null_varchar.clone(), null_varchar.clone(), null_varchar)
    };

    // `(count(*) - count(col)) * 100.0 / count(*)`, guarded against a
    // zero-row table (which would otherwise divide by zero).
    let null_pct_expr = format!(
        "CAST(CASE WHEN count(*) = 0 THEN 0.0 ELSE \
         round((count(*) - count({ident})) * 100.0 / count(*), 2) END AS VARCHAR)"
    );

    let exprs = [
        quote_literal(name),
        quote_literal(ty),
        format!("CAST(min({ident}) AS VARCHAR)"),
        format!("CAST(max({ident}) AS VARCHAR)"),
        format!("approx_count_distinct({ident})"),
        avg_expr,
        std_expr,
        q50_expr,
        format!("count({ident})"),
        null_pct_expr,
    ];

    if first {
        let parts: Vec<String> = exprs
            .iter()
            .zip(OUTPUT_ALIASES.iter())
            .map(|(expr, alias)| format!("{expr} AS {}", quote_ident(alias)))
            .collect();
        format!("SELECT {} FROM {target}", parts.join(", "))
    } else {
        format!("SELECT {} FROM {target}", exprs.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- parse -------------------------------------------------------------

    #[test]
    fn parse_bare_identifier() {
        assert_eq!(parse("SUMMARIZE t"), Some("t".to_string()));
    }

    #[test]
    fn parse_is_case_insensitive() {
        assert_eq!(parse("summarize t"), Some("t".to_string()));
        assert_eq!(parse("SuMmArIzE t"), Some("t".to_string()));
    }

    #[test]
    fn parse_strips_trailing_semicolon_and_whitespace() {
        assert_eq!(parse("SUMMARIZE t;"), Some("t".to_string()));
        assert_eq!(parse("SUMMARIZE t ; "), Some("t".to_string()));
        assert_eq!(parse("  SUMMARIZE   t  "), Some("t".to_string()));
    }

    #[test]
    fn parse_skips_leading_whitespace_and_comments() {
        assert_eq!(parse("-- a comment\nSUMMARIZE t"), Some("t".to_string()));
        assert_eq!(parse("/* a comment */ SUMMARIZE t"), Some("t".to_string()));
        assert_eq!(parse("  -- c1\n/* c2 */  -- c3\n  SUMMARIZE t"), Some("t".to_string()));
    }

    #[test]
    fn parse_quoted_path_literal() {
        assert_eq!(parse("SUMMARIZE 'data/x.parquet'"), Some("'data/x.parquet'".to_string()));
    }

    #[test]
    fn parse_table_function_call() {
        assert_eq!(parse("SUMMARIZE read_csv('x.csv')"), Some("read_csv('x.csv')".to_string()));
    }

    #[test]
    fn parse_parenthesised_subquery() {
        assert_eq!(
            parse("SUMMARIZE (SELECT 1 AS a FROM range(1))"),
            Some("(SELECT 1 AS a FROM range(1))".to_string())
        );
    }

    #[test]
    fn parse_rejects_non_summarize_statements() {
        assert_eq!(parse("SELECT 1"), None);
        assert_eq!(parse(""), None);
        assert_eq!(parse("SUM"), None);
    }

    #[test]
    fn parse_rejects_keyword_prefix_without_word_boundary() {
        assert_eq!(parse("SUMMARIZED t"), None);
        assert_eq!(parse("SUMMARIZE_FOO t"), None);
    }

    #[test]
    fn parse_rejects_missing_target() {
        assert_eq!(parse("SUMMARIZE"), None);
        assert_eq!(parse("SUMMARIZE   ;  "), None);
        assert_eq!(parse("SUMMARIZE ;"), None);
    }

    // -- summarize_sql -------------------------------------------------------

    #[test]
    fn summarize_sql_two_columns() {
        let columns = vec![
            ("id".to_string(), "BIGINT".to_string(), false),
            ("name".to_string(), "VARCHAR".to_string(), true),
        ];
        let sql = summarize_sql("t", &columns);
        let expected = "SELECT 'id' AS \"column_name\", 'BIGINT' AS \"column_type\", \
CAST(min(\"id\") AS VARCHAR) AS \"min\", CAST(max(\"id\") AS VARCHAR) AS \"max\", \
approx_count_distinct(\"id\") AS \"approx_unique\", \
CAST(round(avg(\"id\"), 2) AS VARCHAR) AS \"avg\", \
CAST(round(stddev(\"id\"), 2) AS VARCHAR) AS \"std\", \
CAST(median(\"id\") AS VARCHAR) AS \"q50\", count(\"id\") AS \"count\", \
CAST(CASE WHEN count(*) = 0 THEN 0.0 ELSE round((count(*) - count(\"id\")) * 100.0 / count(*), 2) END AS VARCHAR) AS \"null_percentage\" FROM t\n\
UNION ALL\n\
SELECT 'name', 'VARCHAR', CAST(min(\"name\") AS VARCHAR), CAST(max(\"name\") AS VARCHAR), \
approx_count_distinct(\"name\"), CAST(NULL AS VARCHAR), CAST(NULL AS VARCHAR), CAST(NULL AS VARCHAR), \
count(\"name\"), CAST(CASE WHEN count(*) = 0 THEN 0.0 ELSE round((count(*) - count(\"name\")) * 100.0 / count(*), 2) END AS VARCHAR) FROM t";
        assert_eq!(sql, expected);
    }

    #[test]
    fn summarize_sql_quotes_column_name_with_embedded_quote() {
        let columns = vec![("wei\"rd".to_string(), "VARCHAR".to_string(), true)];
        let sql = summarize_sql("t", &columns);
        assert!(sql.contains("'wei\"rd' AS \"column_name\""));
        assert!(sql.contains("min(\"wei\"\"rd\")"));
    }

    #[test]
    fn summarize_sql_quotes_column_name_with_space() {
        let columns = vec![("my col".to_string(), "BIGINT".to_string(), false)];
        let sql = summarize_sql("t", &columns);
        assert!(sql.contains("'my col' AS \"column_name\""));
        assert!(sql.contains("min(\"my col\")"));
    }

    #[test]
    fn summarize_sql_empty_columns_yields_zero_row_schema() {
        let sql = summarize_sql("t", &[]);
        assert!(sql.contains("FROM range(1) WHERE false"));
        assert!(sql.contains("AS \"column_name\""));
        assert!(sql.contains("AS \"null_percentage\""));
        assert!(sql.contains("CAST(NULL AS BIGINT) AS \"count\""));
    }

    #[test]
    fn is_numeric_type_recognises_numeric_families() {
        assert!(is_numeric_type("BIGINT"));
        assert!(is_numeric_type("DOUBLE"));
        assert!(is_numeric_type("DECIMAL"));
        assert!(is_numeric_type("DECIMAL(10,2)"));
        assert!(!is_numeric_type("VARCHAR"));
        assert!(!is_numeric_type("TIMESTAMP"));
        assert!(!is_numeric_type("JSON"));
    }
}
