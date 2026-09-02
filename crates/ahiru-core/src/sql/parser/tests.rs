use super::*;
use crate::error::Code;
#[cfg(feature = "ddl")]
use crate::sql::ast::AlterTableAction;
#[cfg(feature = "dml")]
use crate::sql::ast::InsertSource;
use crate::sql::lexer::{keyword, KEYWORDS};

// --- Test helpers -------------------------------------------------------

/// Renders an expression tree back to a fully parenthesized string. Literals are printed in a form that shows their type.
fn r(a: &ExprArena, id: ExprId) -> String {
    match a.get(id) {
        // Window specifications are printed in the order `PARTITION BY .. ORDER BY .. <frame>`.
        // The frame is always drawn, so a wrong default is detected.
        Expr::Window { name, args, star, window_ref, partition_by, order_by, frame } => {
            let inner = if *star {
                "*".to_string()
            } else {
                let items: Vec<String> = args.iter().map(|i| r(a, *i)).collect();
                items.join(", ")
            };
            if let Some(w) = window_ref {
                return format!("{}({}) OVER {}", name, inner, w);
            }
            let mut spec: Vec<String> = Vec::new();
            if !partition_by.is_empty() {
                let ps: Vec<String> = partition_by.iter().map(|e| r(a, *e)).collect();
                spec.push(format!("PARTITION BY {}", ps.join(", ")));
            }
            if !order_by.is_empty() {
                spec.push(format!("ORDER BY {}", order_list(a, order_by)));
            }
            spec.push(
                match frame {
                    WindowFrame::WholePartition => "WHOLE",
                    WindowFrame::RangeUnboundedPreceding => "RANGE",
                }
                .to_string(),
            );
            format!("{}({}) OVER ({})", name, inner, spec.join(" "))
        }
        Expr::ScalarSubquery(q) => format!("({})", query_str(a, q)),
        Expr::Exists { query, negated } => {
            format!("{}EXISTS ({})", if *negated { "NOT " } else { "" }, query_str(a, query))
        }
        Expr::InSubquery { arg, query, negated } => format!(
            "({}{} IN ({}))",
            r(a, *arg),
            if *negated { " NOT" } else { "" },
            query_str(a, query)
        ),
        Expr::QuantifiedComparison { op, arg, all, query } => format!(
            "({} {} {} ({}))",
            r(a, *arg),
            op_name(*op),
            if *all { "ALL" } else { "ANY" },
            query_str(a, query)
        ),
        Expr::Literal(v) => lit(v),
        Expr::IntervalLiteral(v) => format!("INTERVAL({}i128)", v),
        Expr::TypedLiteral(v, ty) => format!("{v:?}::{}", ty.name()),
        Expr::Param(n) => format!("?{}", n),
        Expr::ColumnRef { qualifier, name } => match qualifier {
            Some(q) => format!("{}.{}", q, name),
            None => name.clone(),
        },
        Expr::Star { qualifier, columns, exclude, replace, rename } => {
            let mut s = match (qualifier, columns) {
                (_, Some(ColumnsSpec::All)) => "COLUMNS(*)".to_string(),
                (_, Some(ColumnsSpec::Regex(p))) => format!("COLUMNS('{}')", p),
                (_, Some(ColumnsSpec::Names(ns))) => {
                    let items: Vec<String> = ns.iter().map(|n| format!("'{}'", n)).collect();
                    format!("COLUMNS([{}])", items.join(", "))
                }
                (Some(q), None) => format!("{}.*", q),
                (None, None) => "*".to_string(),
            };
            if !exclude.is_empty() {
                s.push_str(&format!(" EXCLUDE ({})", exclude.join(", ")));
            }
            if !replace.is_empty() {
                let items: Vec<String> =
                    replace.iter().map(|(e, n)| format!("{} AS {}", r(a, *e), n)).collect();
                s.push_str(&format!(" REPLACE ({})", items.join(", ")));
            }
            if !rename.is_empty() {
                let items: Vec<String> =
                    rename.iter().map(|(old, new)| format!("{} AS {}", old, new)).collect();
                s.push_str(&format!(" RENAME ({})", items.join(", ")));
            }
            s
        }
        Expr::Unary { op, arg } => {
            let o = if *op == UnaryOp::Neg { "-" } else { "NOT" };
            format!("({} {})", o, r(a, *arg))
        }
        Expr::Binary { op, lhs, rhs } => {
            format!("({} {} {})", r(a, *lhs), op_name(*op), r(a, *rhs))
        }
        Expr::Cast { arg, ty, try_ } => {
            format!(
                "{}({} AS {})",
                if *try_ { "TRY_CAST" } else { "CAST" },
                r(a, *arg),
                ty_name(*ty)
            )
        }
        Expr::Case { operand, whens, else_ } => {
            let mut s = String::from("CASE");
            if let Some(o) = operand {
                s.push_str(&format!(" {}", r(a, *o)));
            }
            for (w, t) in whens {
                s.push_str(&format!(" WHEN {} THEN {}", r(a, *w), r(a, *t)));
            }
            if let Some(e) = else_ {
                s.push_str(&format!(" ELSE {}", r(a, *e)));
            }
            s.push_str(" END");
            s
        }
        Expr::InList { arg, list, negated } => {
            let items: Vec<String> = list.iter().map(|i| r(a, *i)).collect();
            format!(
                "({}{} IN [{}])",
                r(a, *arg),
                if *negated { " NOT" } else { "" },
                items.join(", ")
            )
        }
        Expr::Between { arg, low, high, negated } => format!(
            "({}{} BETWEEN {} AND {})",
            r(a, *arg),
            if *negated { " NOT" } else { "" },
            r(a, *low),
            r(a, *high)
        ),
        Expr::IsNull { arg, negated } => {
            format!("({} IS{} NULL)", r(a, *arg), if *negated { " NOT" } else { "" })
        }
        Expr::Like { arg, pattern, negated, ci } => {
            format!(
                "({}{} {} {})",
                r(a, *arg),
                if *negated { " NOT" } else { "" },
                if *ci { "ILIKE" } else { "LIKE" },
                r(a, *pattern),
            )
        }
        Expr::Function { name, args, distinct, star, filter } => {
            let inner = if *star {
                "*".to_string()
            } else {
                let items: Vec<String> = args.iter().map(|i| r(a, *i)).collect();
                format!("{}{}", if *distinct { "DISTINCT " } else { "" }, items.join(", "))
            };
            let f = match filter {
                Some(f) => format!(" FILTER (WHERE {})", r(a, *f)),
                None => String::new(),
            };
            format!("{}({}){}", name, inner, f)
        }
        Expr::Unnest(arg) => format!("UNNEST({})", r(a, *arg)),
        Expr::Lambda { params, body } => {
            let p = if params.len() == 1 {
                params[0].clone()
            } else {
                format!("({})", params.join(", "))
            };
            format!("{} -> {}", p, r(a, *body))
        }
    }
}

fn lit(v: &Value) -> String {
    match v {
        Value::Null => "NULL".to_string(),
        Value::Bool(b) => format!("{}", b),
        Value::I32(x) => format!("{}i32", x),
        Value::I64(x) => format!("{}i64", x),
        Value::I128(x) => format!("{}i128", x),
        Value::F64(x) => format!("{}f64", x),
        Value::Bytes(b) => format!("'{}'", String::from_utf8_lossy(b)),
    }
}

fn op_name(op: BinaryOp) -> &'static str {
    use BinaryOp::*;
    match op {
        Add => "+",
        Sub => "-",
        Mul => "*",
        Div => "/",
        Mod => "%",
        Eq => "=",
        Ne => "!=",
        Lt => "<",
        Le => "<=",
        Gt => ">",
        Ge => ">=",
        And => "AND",
        Or => "OR",
        Concat => "||",
    }
}

fn ty_name(ty: Ty) -> String {
    match ty {
        Ty::Decimal { precision, scale } => format!("DECIMAL({},{})", precision, scale),
        other => other.name().to_string(),
    }
}

fn from_str(a: &ExprArena, f: &FromItem) -> String {
    fn alias(s: &Option<String>) -> String {
        match s {
            Some(x) => format!(" AS {}", x),
            None => String::new(),
        }
    }
    match f {
        FromItem::Table { name, alias: al } => format!("{}{}", name, alias(al)),
        FromItem::File { path, format, alias: al } => {
            let fn_name = match format {
                FormatKind::Csv | FormatKind::Tsv => "read_csv",
                FormatKind::Json | FormatKind::Jsonl => "read_json",
                FormatKind::Parquet | FormatKind::Auto => "parquet",
            };
            format!("{}('{}'){}", fn_name, path, alias(al))
        }
        FromItem::Subquery { query, alias: al } => {
            format!("({}){}", query_str(a, query), alias(al))
        }
        FromItem::Join { left, right, kind, on } => {
            let k = match kind {
                JoinKind::Inner => "INNER",
                JoinKind::Left => "LEFT",
                JoinKind::Right => "RIGHT",
                JoinKind::Full => "FULL",
                JoinKind::Cross => "CROSS",
                // The SEMI / ANTI family is never produced by syntax (binder only).
                // They are caught together so parser tests do not break as more kinds are added.
                _ => "BINDER-ONLY",
            };
            let on_s = match on {
                Some(e) => format!(" ON {}", r(a, *e)),
                None => String::new(),
            };
            format!("({} {} JOIN {}{})", from_str(a, left), k, from_str(a, right), on_s)
        }
        FromItem::Unnest { expr, alias: al, column_alias } => {
            let col = match column_alias {
                Some(c) => format!("({})", c),
                None => String::new(),
            };
            format!("UNNEST({}){}{}", r(a, *expr), alias(al), col)
        }
        FromItem::GenerateSeries { start, stop, step, inclusive, alias: al, column_alias } => {
            let col = match column_alias {
                Some(c) => format!("({})", c),
                None => String::new(),
            };
            let name = if *inclusive { "generate_series" } else { "range" };
            format!("{}({},{},{}){}{}", name, start, stop, step, alias(al), col)
        }
    }
}

/// Flattens a SELECT statement onto one line. For comparing structure; not valid SQL.
fn select_str(a: &ExprArena, s: &SelectStmt) -> String {
    let mut out = String::from("SELECT");
    if s.distinct {
        out.push_str(" DISTINCT");
    }
    if !s.distinct_on.is_empty() {
        let on: Vec<String> = s.distinct_on.iter().map(|e| r(a, *e)).collect();
        out.push_str(&format!(" DISTINCT ON ({})", on.join(", ")));
    }
    let items: Vec<String> = s
        .items
        .iter()
        .map(|i| match &i.alias {
            Some(al) => format!("{} AS {}", r(a, i.expr), al),
            None => r(a, i.expr),
        })
        .collect();
    out.push_str(&format!(" {}", items.join(", ")));
    if let Some(f) = &s.from {
        out.push_str(&format!(" FROM {}", from_str(a, f)));
    }
    if let Some(w) = s.filter {
        out.push_str(&format!(" WHERE {}", r(a, w)));
    }
    if s.group_by_all {
        out.push_str(" GROUP BY ALL");
    }
    if !s.group_by.is_empty() {
        let g: Vec<String> = s.group_by.iter().map(|e| r(a, *e)).collect();
        out.push_str(&format!(" GROUP BY {}", g.join(", ")));
    }
    if let Some(sets) = &s.grouping_sets {
        let sets_str: Vec<String> = sets
            .iter()
            .map(|set| {
                let cols: Vec<String> = set.iter().map(|e| r(a, *e)).collect();
                format!("({})", cols.join(", "))
            })
            .collect();
        out.push_str(&format!(" GROUP BY GROUPING SETS ({})", sets_str.join(", ")));
    }
    if let Some(h) = s.having {
        out.push_str(&format!(" HAVING {}", r(a, h)));
    }
    if !s.windows.is_empty() {
        let ws: Vec<String> = s
            .windows
            .iter()
            .map(|(name, def)| {
                let mut spec: Vec<String> = Vec::new();
                if !def.partition_by.is_empty() {
                    let ps: Vec<String> = def.partition_by.iter().map(|e| r(a, *e)).collect();
                    spec.push(format!("PARTITION BY {}", ps.join(", ")));
                }
                if !def.order_by.is_empty() {
                    spec.push(format!("ORDER BY {}", order_list(a, &def.order_by)));
                }
                format!("{} AS ({})", name, spec.join(" "))
            })
            .collect();
        out.push_str(&format!(" WINDOW {}", ws.join(", ")));
    }
    if let Some(q) = s.qualify {
        out.push_str(&format!(" QUALIFY {}", r(a, q)));
    }
    if let Some(oa) = &s.order_by_all {
        out.push_str(&format!(" ORDER BY ALL {}", order_all_str(oa)));
    }
    if !s.order_by.is_empty() {
        out.push_str(&format!(" ORDER BY {}", order_list(a, &s.order_by)));
    }
    if let Some(l) = s.limit {
        out.push_str(&format!(" LIMIT {}", l));
    }
    if let Some(o) = s.offset {
        out.push_str(&format!(" OFFSET {}", o));
    }
    out
}

fn order_all_str(oa: &OrderByAll) -> String {
    format!(
        "{} NULLS {}",
        if oa.desc { "DESC" } else { "ASC" },
        if oa.nulls_first { "FIRST" } else { "LAST" }
    )
}

fn order_list(a: &ExprArena, items: &[OrderByItem]) -> String {
    let o: Vec<String> = items
        .iter()
        .map(|i| {
            format!(
                "{} {} NULLS {}",
                r(a, i.expr),
                if i.desc { "DESC" } else { "ASC" },
                if i.nulls_first { "FIRST" } else { "LAST" }
            )
        })
        .collect();
    o.join(", ")
}

/// The set-operation tree. Parentheses are always added, so the associativity reads directly.
fn setexpr_str(a: &ExprArena, e: &SetExpr) -> String {
    match e {
        SetExpr::Select(s) => select_str(a, s),
        SetExpr::SetOp { op, all, left, right } => {
            let name = match op {
                SetOp::Union => "UNION",
                SetOp::Intersect => "INTERSECT",
                SetOp::Except => "EXCEPT",
            };
            format!(
                "({} {}{} {})",
                setexpr_str(a, left),
                name,
                if *all { " ALL" } else { "" },
                setexpr_str(a, right)
            )
        }
    }
}

/// The whole query (CTEs + body + the outer ORDER BY / LIMIT).
fn query_str(a: &ExprArena, q: &QueryStmt) -> String {
    let mut out = String::new();
    if !q.ctes.is_empty() {
        let recursive = q.ctes.iter().any(|c| c.recursive);
        let cs: Vec<String> = q
            .ctes
            .iter()
            .map(|c| {
                let cols = if c.columns.is_empty() {
                    String::new()
                } else {
                    format!("({})", c.columns.join(", "))
                };
                format!("{}{} AS ({})", c.name, cols, query_str(a, &c.query))
            })
            .collect();
        let kw = if recursive { "WITH RECURSIVE " } else { "WITH " };
        out.push_str(&format!("{}{} ", kw, cs.join(", ")));
    }
    out.push_str(&setexpr_str(a, &q.body));
    if let Some(oa) = &q.order_by_all {
        out.push_str(&format!(" ORDER BY ALL {}", order_all_str(oa)));
    }
    if !q.order_by.is_empty() {
        out.push_str(&format!(" ORDER BY {}", order_list(a, &q.order_by)));
    }
    if let Some(l) = q.limit {
        out.push_str(&format!(" LIMIT {}", l));
    }
    if let Some(o) = q.offset {
        out.push_str(&format!(" OFFSET {}", o));
    }
    out
}

/// Extracts the plain `SelectStmt` from a `QueryStmt` (assuming no set operations or CTEs).
fn plain(q: &QueryStmt) -> &SelectStmt {
    match &q.body {
        SetExpr::Select(s) => s,
        _ => panic!("query contains a set operation"),
    }
}

fn sel(sql: &str) -> String {
    let p = parse(sql).expect("parse failed");
    match &p.stmt {
        Stmt::Select(q) => select_str(&p.arena, plain(q)),
        _ => panic!("not a SELECT"),
    }
}

/// Renders the whole query. Statements with set operations or CTEs are inspected with this.
fn qs(sql: &str) -> String {
    let p = parse(sql).expect("parse failed");
    match &p.stmt {
        Stmt::Select(q) => query_str(&p.arena, q),
        _ => panic!("not a SELECT"),
    }
}

/// Parses a statement and extracts the `QueryStmt`. For tests that inspect the tree shape directly.
fn parsed(sql: &str) -> (ExprArena, Box<QueryStmt>) {
    let p = parse(sql).expect("parse failed");
    match p.stmt {
        Stmt::Select(q) => (p.arena, q),
        _ => panic!("not a SELECT"),
    }
}

/// Renders just the expression by running it through `SELECT <expr>`.
fn ex(expr: &str) -> String {
    let sql = format!("SELECT {}", expr);
    let p = parse(&sql).expect("parse failed");
    match &p.stmt {
        Stmt::Select(q) => r(&p.arena, plain(q).items[0].expr),
        _ => panic!("not a SELECT"),
    }
}

fn code(sql: &str) -> u16 {
    match parse(sql) {
        Ok(_) => 0,
        Err(e) => e.code_u16(),
    }
}

fn err_at(sql: &str) -> (u16, u32) {
    match parse(sql) {
        Ok(_) => (0, 0),
        Err(e) => (e.code_u16(), e.pos),
    }
}

// --- Lexing -------------------------------------------------------------

#[test]
fn keyword_table_is_sorted_and_complete() {
    // The binary search's precondition: (length, lowercased first byte) is monotonically non-decreasing.
    let key = |n: &[u8]| ((n.len() as u32) << 8) | (n[0] | 0x20) as u32;
    for w in KEYWORDS.windows(2) {
        assert!(key(w[0].0) <= key(w[1].0), "unsorted near {:?}", w[0].0);
    }
    for &(name, kw) in KEYWORDS {
        let upper = name.to_ascii_uppercase();
        assert_eq!(keyword(name), Some(kw));
        assert_eq!(keyword(&upper), Some(kw));
    }
    assert_eq!(keyword(b"nope"), None);
    assert_eq!(keyword(b"a"), None);
}

#[test]
fn comments_and_whitespace() {
    assert_eq!(
        sel("SELECT /* column */ a -- trailing comment\n FROM t /* middle */ WHERE a > 1"),
        "SELECT a FROM t WHERE (a > 1i32)"
    );
    assert_eq!(sel("SELECT 1 --x"), "SELECT 1i32");
    // Block comments do not nest: the first */ closes them.
    assert_eq!(sel("SELECT /* /* */ 1"), "SELECT 1i32");
    assert_eq!(code("SELECT /* unclosed 1"), Code::SyntaxError as u16);
}

// --- Precedence and associativity ---------------------------------------

#[test]
fn precedence() {
    assert_eq!(ex("a + b * c"), "(a + (b * c))");
    assert_eq!(ex("a * b + c"), "((a * b) + c)");
    assert_eq!(ex("a OR b AND c"), "(a OR (b AND c))");
    assert_eq!(ex("NOT a = b"), "(NOT (a = b))");
    assert_eq!(ex("NOT a AND b"), "((NOT a) AND b)");
    assert_eq!(ex("a = b AND c = d"), "((a = b) AND (c = d))");
    assert_eq!(ex("-a + b"), "((- a) + b)");
    assert_eq!(ex("a || b = c"), "((a || b) = c)");
    assert_eq!(ex("a % b + c"), "((a % b) + c)");
    assert_eq!(ex("(a + b) * c"), "((a + b) * c)");
    // == and <> are aliases for = and !=.
    assert_eq!(ex("a == b"), "(a = b)");
    assert_eq!(ex("a <> b"), "(a != b)");
    assert_eq!(ex("a != b"), "(a != b)");
    assert_eq!(ex("a <= b"), "(a <= b)");
    assert_eq!(ex("a >= b"), "(a >= b)");
}

#[test]
fn left_associativity() {
    assert_eq!(ex("1 - 2 - 3"), "((1i32 - 2i32) - 3i32)");
    assert_eq!(ex("8 / 4 / 2"), "((8i32 / 4i32) / 2i32)");
    assert_eq!(ex("a AND b AND c"), "((a AND b) AND c)");
    assert_eq!(ex("a || b || c"), "((a || b) || c)");
}

#[test]
fn unary() {
    assert_eq!(ex("-x"), "(- x)");
    assert_eq!(ex("+x"), "x");
    assert_eq!(ex("- -x"), "(- (- x))");
    assert_eq!(ex("-x * y"), "((- x) * y)");
    assert_eq!(ex("NOT NOT a"), "(NOT (NOT a))");
}

// --- Predicates ---------------------------------------------------------

#[test]
fn predicates() {
    assert_eq!(ex("x IS NULL"), "(x IS NULL)");
    assert_eq!(ex("x IS NOT NULL"), "(x IS NOT NULL)");
    assert_eq!(ex("x IN (1, 2)"), "(x IN [1i32, 2i32])");
    assert_eq!(ex("x NOT IN (1)"), "(x NOT IN [1i32])");
    assert_eq!(ex("x BETWEEN 1 AND 2"), "(x BETWEEN 1i32 AND 2i32)");
    assert_eq!(ex("x NOT BETWEEN 1 AND 2"), "(x NOT BETWEEN 1i32 AND 2i32)");
    assert_eq!(ex("x LIKE 'a%'"), "(x LIKE 'a%')");
    assert_eq!(ex("x NOT LIKE 'a%'"), "(x NOT LIKE 'a%')");
    // A custom escape desugars into a `like_escape` call (DuckDB's own shape).
    assert_eq!(ex("x LIKE 'a!%' ESCAPE '!'"), "like_escape(x, 'a!%', '!')");
    // The AND separating BETWEEN's bounds is not swallowed as a logical operator.
    assert_eq!(ex("a BETWEEN 1 AND 2 AND b"), "((a BETWEEN 1i32 AND 2i32) AND b)");
    assert_eq!(ex("a BETWEEN 1 + 1 AND 2 * 3"), "(a BETWEEN (1i32 + 1i32) AND (2i32 * 3i32))");
    // Predicates have the same binding power as comparison. They bind tighter than AND.
    assert_eq!(ex("a IS NULL AND b IS NOT NULL"), "((a IS NULL) AND (b IS NOT NULL))");
    assert_eq!(ex("NOT a IS NULL"), "(NOT (a IS NULL))");
    // `IS TRUE`/`IS FALSE` are now supported (see the `is_true_desugars_to_
    // cast_and_coalesce`/`is_false_desugars_to_negated_cast_and_coalesce`
    // tests near the end of this file); an actually-unsupported `IS`
    // right-hand side is still rejected.
    assert_eq!(code("SELECT x IS 5"), Code::UnsupportedFeature as u16);
    assert_eq!(code("SELECT x NOT IS NULL"), Code::UnexpectedToken as u16);
}

#[test]
fn ilike_predicate() {
    assert_eq!(ex("x ILIKE 'a%'"), "(x ILIKE 'a%')");
    assert_eq!(ex("x NOT ILIKE 'a%'"), "(x NOT ILIKE 'a%')");
    assert_eq!(ex("x ilike 'a%'"), "(x ILIKE 'a%')");
    // With ESCAPE, ILIKE lowers both sides and then calls `like_escape`.
    assert_eq!(ex("x ILIKE 'a!%' ESCAPE '!'"), "like_escape(lower(x), lower('a!%'), '!')");
    // ILIKE is a predicate too, so it binds tighter than AND.
    assert_eq!(ex("a ILIKE 'x' AND b"), "((a ILIKE 'x') AND b)");
    // `ILIKE` is fully reserved (unusable as a column name).
    assert_eq!(code("SELECT ilike FROM t"), Code::UnexpectedToken as u16);
}

#[test]
fn glob_operator_desugars_to_glob_function() {
    assert_eq!(ex("a GLOB 'x*'"), "glob(a, 'x*')");
    assert_eq!(ex("a glob 'x*'"), "glob(a, 'x*')");
    // GLOB has the same binding power as a predicate (tighter than AND).
    assert_eq!(ex("a GLOB 'x' AND b"), "(glob(a, 'x') AND b)");
    // DuckDB cannot write `NOT GLOB` (confirmed that `duckdb -c "select 'a' NOT GLOB
    // 'b'"` is a syntax error). `NOT (x GLOB y)` is an ordinary prefix NOT and writes fine.
    assert_eq!(code("SELECT a NOT GLOB 'x'"), Code::UnexpectedToken as u16);
    assert_eq!(ex("NOT (a GLOB 'x')"), "(NOT glob(a, 'x'))");
    // `glob` is not reserved (the same "context-dependent keyword" scheme as
    // ROWS/RANGE/QUALIFY), so it remains usable as a column name.
    assert_eq!(ex("glob"), "glob");
    assert_eq!(code("SELECT glob FROM t"), 0);
}

#[test]
fn similar_to_desugars_to_regexp_full_match() {
    assert_eq!(ex("a SIMILAR TO 'x.y'"), "regexp_full_match(a, 'x.y')");
    assert_eq!(ex("a similar to 'x.y'"), "regexp_full_match(a, 'x.y')");
    assert_eq!(ex("a NOT SIMILAR TO 'x.y'"), "(NOT regexp_full_match(a, 'x.y'))");
    // The same predicate binding power as LIKE (tighter than AND).
    assert_eq!(ex("a SIMILAR TO 'x' AND b"), "(regexp_full_match(a, 'x') AND b)");
    // DuckDB itself rejects the `ESCAPE` clause as unimplemented
    // (confirmed with `duckdb -c "select 'a' similar to 'a' escape '\\'"`).
    assert_eq!(code(r"SELECT a SIMILAR TO 'x' ESCAPE '\'"), Code::UnsupportedFeature as u16);
    // Neither `similar` nor `to` is reserved, so both remain usable as column names
    // (a design informed by the past ROWS/RANGE/QUALIFY lessons).
    assert_eq!(ex("similar"), "similar");
    assert_eq!(code("SELECT similar, glob FROM t"), 0);
}

#[test]
fn distinct_from_desugars_to_null_safe_equality() {
    assert_eq!(
            ex("a IS DISTINCT FROM b"),
            "(NOT (((a IS NULL) AND (b IS NULL)) OR (((a IS NOT NULL) AND (b IS NOT NULL)) AND (a = b))))"
        );
    assert_eq!(
        ex("a IS NOT DISTINCT FROM b"),
        "(((a IS NULL) AND (b IS NULL)) OR (((a IS NOT NULL) AND (b IS NOT NULL)) AND (a = b)))"
    );
    // Both `distinct` and `from` are existing reserved words, so they need quotes to be
    // column names (they are not context-dependent keywords like
    // `similar`/`glob`/`to`).
    assert_eq!(code(r#"SELECT "distinct" FROM t"#), 0);
}

#[test]
fn cast_shorthand_desugars_to_cast() {
    assert_eq!(ex("x::INTEGER"), "CAST(x AS INTEGER)");
    assert_eq!(ex("'42'::INTEGER"), "CAST('42' AS INTEGER)");
    // `::` binds tighter than prefix operators (confirmed by `duckdb -c "select -1::varchar"`
    // being interpreted as `-(1::VARCHAR)` and thus a type error).
    assert_eq!(ex("-1::VARCHAR"), "CAST(-1i32 AS VARCHAR)");
    assert_eq!(ex("(1 + 2)::VARCHAR"), "CAST((1i32 + 2i32) AS VARCHAR)");
    // Repeated application folds too.
    assert_eq!(ex("x::INTEGER::VARCHAR"), "CAST(CAST(x AS INTEGER) AS VARCHAR)");
}

#[test]
fn power_operator_desugars_to_pow() {
    assert_eq!(ex("2 ^ 10"), "pow(2i32, 10i32)");
    assert_eq!(ex("2 ** 10"), "pow(2i32, 10i32)");
    // Left-associative (confirmed `duckdb -c "select 2^3^2"` = 64; see the BP_POW docs).
    assert_eq!(ex("2 ^ 3 ^ 2"), "pow(pow(2i32, 3i32), 2i32)");
    // Tighter than `*`/`/`, looser than unary `-`.
    assert_eq!(ex("2 + 3 ^ 2"), "(2i32 + pow(3i32, 2i32))");
    assert_eq!(ex("-2 ^ 2"), "pow(-2i32, 2i32)");
}

#[test]
fn bitwise_operators_desugar_to_bit_functions() {
    assert_eq!(ex("a & b"), "bit_and(a, b)");
    assert_eq!(ex("a | b"), "bit_or(a, b)");
    assert_eq!(ex("a << b"), "bit_shift_left(a, b)");
    assert_eq!(ex("a >> b"), "bit_shift_right(a, b)");
    assert_eq!(ex("~a"), "bit_not(a)");
    // `&`/`|` bind tighter than comparison and looser than `+`/`-` (confirmed
    // `duckdb -c "select 1 + 2 & 3"` = `(1 + 2) & 3` and `duckdb -c "select 1 & 2 = 0"` =
    // `(1 & 2) = 0`).
    assert_eq!(ex("1 + 2 & 3"), "bit_and((1i32 + 2i32), 3i32)");
    assert_eq!(ex("1 & 2 = 0"), "(bit_and(1i32, 2i32) = 0i32)");
}

#[test]
fn tilde_operators_desugar_to_regexp_full_match() {
    // Infix `~`/`!~` (regex match). Prefix `~` (bitwise NOT) is covered above in
    // `bitwise_operators_desugar_to_bit_functions` -- the same token, but its meaning
    // changes depending on whether it is at the head or in the middle of an expression
    // (the same pattern as `-`; see the `prefix`/`expr_body` docs).
    assert_eq!(ex("a ~ 'x.y'"), "regexp_full_match(a, 'x.y')");
    assert_eq!(ex("a !~ 'x.y'"), "(NOT regexp_full_match(a, 'x.y'))");
}

#[test]
fn array_literal_desugars_to_list_value() {
    assert_eq!(ex("[1, 2, 3]"), "list_value(1i32, 2i32, 3i32)");
    assert_eq!(ex("['a', 'b']"), "list_value('a', 'b')");
    assert_eq!(ex("[1 + 1]"), "list_value((1i32 + 1i32))");
    // The empty array bypasses `list_value()` (whose `resolve` is designed to reject zero
    // arguments) and embeds an empty JSON array directly as a TypedLiteral. Confirmed with
    // `duckdb -c "select []"` that it is a valid expression.
    assert_eq!(ex("[]"), "Bytes([91, 93])::JSON");
}

// --- Subscripting / slicing (`primary`/`subscript`) -------------------------

#[test]
fn subscript_desugars_to_list_extract() {
    // `expr[i]` is sugar for `list_extract(expr, i)`. It is distinguished by position from
    // a `[` at the head of an expression (an array literal) (see the comment at the top of
    // `primary_atom`).
    assert_eq!(ex("a[1]"), "list_extract(a, 1i32)");
    assert_eq!(ex("a[-1]"), "list_extract(a, -1i32)");
    // The subscript itself may be any expression.
    assert_eq!(ex("a[b + 1]"), "list_extract(a, (b + 1i32))");
    // Nesting does not break down: confirmed that the same shape as
    // `duckdb -c "select [[1,2],[3,4]][1]"` still works.
    assert_eq!(
        ex("[[1, 2], [3, 4]][1]"),
        "list_extract(list_value(list_value(1i32, 2i32), list_value(3i32, 4i32)), 1i32)"
    );
    assert_eq!(ex("a[1][2]"), "list_extract(list_extract(a, 1i32), 2i32)");
}

#[test]
fn slice_desugars_to_list_slice_with_omittable_bounds() {
    // `expr[i:j]` is sugar for `list_slice(expr, i, j)`.
    assert_eq!(ex("a[2:3]"), "list_slice(a, 2i32, 3i32)");
    // An omitted start desugars to `1` and an omitted end to `i64::MAX` (see the
    // `subscript` doc comment; it does not desugar to SQL NULL -- `list_slice` itself
    // propagates NULL arguments as NULL, so desugaring to NULL would make `[:3]` NULL).
    assert_eq!(ex("a[:3]"), "list_slice(a, 1i64, 3i32)");
    assert_eq!(ex("a[2:]"), format!("list_slice(a, 2i32, {}i64)", i64::MAX));
    assert_eq!(ex("a[:]"), format!("list_slice(a, 1i64, {}i64)", i64::MAX));
}

#[test]
fn postfix_cast_and_subscript_interleave_left_to_right() {
    // `[1,2,3][1]::varchar` means "subscript first, then cast"
    // (confirmed with `duckdb -c "select [1,2,3][1]::varchar"`).
    assert_eq!(ex("a[1]::varchar"), "CAST(list_extract(a, 1i32) AS VARCHAR)");
    // `a::json[1]` means "cast first, then subscript"
    // (confirmed with `duckdb -c "select [1,2,3]::json[1]"`. In DuckDB this syntax is
    // ambiguous with the ARRAY type literal `json[1]`, but this implementation has no
    // fixed-length ARRAY type, so it reads unambiguously as a postfix subscript on `a::json`).
    assert_eq!(ex("a::json[1]"), "list_extract(CAST(a AS JSON), 1i32)");
    // It binds tighter than unary `-`: `duckdb -c "select -[1,2,3][1]"` means
    // `-(list[1])`.
    assert_eq!(ex("-a[1]"), "(- list_extract(a, 1i32))");
    // The prefix immediate negative-literal folding path (`prefix`) does not fold as far as
    // a subscript (confirmed that `duckdb -c "select -5[1]"` is a syntax error; see the
    // `cast_postfix` doc comment).
    assert_eq!(code("SELECT -5[1]"), Code::UnexpectedToken as u16);
}

// --- CASE / CAST / functions --------------------------------------------

#[test]
fn case_expr() {
    assert_eq!(
        ex("CASE WHEN a > 1 THEN 'x' ELSE 'y' END"),
        "CASE WHEN (a > 1i32) THEN 'x' ELSE 'y' END"
    );
    assert_eq!(ex("CASE WHEN a THEN 1 END"), "CASE WHEN a THEN 1i32 END");
    assert_eq!(
        ex("CASE a WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'many' END"),
        "CASE a WHEN 1i32 THEN 'one' WHEN 2i32 THEN 'two' ELSE 'many' END"
    );
    assert_eq!(code("SELECT CASE a END"), Code::UnexpectedToken as u16);
    assert_eq!(code("SELECT CASE WHEN a THEN 1"), Code::UnexpectedToken as u16);
}

#[test]
fn cast_expr() {
    assert_eq!(ex("CAST(x AS DECIMAL(10,2))"), "CAST(x AS DECIMAL(10,2))");
    assert_eq!(ex("CAST(x AS decimal)"), "CAST(x AS DECIMAL(18,3))");
    assert_eq!(ex("CAST(x AS NUMERIC(38,0))"), "CAST(x AS DECIMAL(38,0))");
    assert_eq!(ex("CAST(x AS INT)"), "CAST(x AS INTEGER)");
    assert_eq!(ex("CAST(x AS integer)"), "CAST(x AS INTEGER)");
    assert_eq!(ex("CAST(x AS BigInt)"), "CAST(x AS BIGINT)");
    assert_eq!(ex("CAST(x AS TEXT)"), "CAST(x AS VARCHAR)");
    assert_eq!(ex("CAST(x AS bytea)"), "CAST(x AS BLOB)");
    assert_eq!(ex("CAST(x AS DATETIME)"), "CAST(x AS TIMESTAMP)");
    assert_eq!(ex("CAST(x AS BOOL)"), "CAST(x AS BOOLEAN)");
    assert_eq!(ex("CAST(x AS REAL)"), "CAST(x AS FLOAT)");
    assert_eq!(ex("CAST(x AS UBIGINT)"), "CAST(x AS UBIGINT)");
    assert_eq!(code("SELECT CAST(x AS FROB)"), Code::InvalidCast as u16);
    assert_eq!(code("SELECT CAST(x AS DECIMAL(99,1))"), Code::InvalidCast as u16);
    assert_eq!(code("SELECT CAST(x AS NULL)"), Code::InvalidCast as u16);
    assert_eq!(ex("CAST(x AS JSON)"), "CAST(x AS JSON)");
    assert_eq!(ex("CAST(x AS json)"), "CAST(x AS JSON)");
}

#[test]
fn json_arrow_operators_desugar_to_function_calls() {
    // `->`/`->>` add no new BinaryOp and expand as sugar for
    // `json_extract`/`json_extract_string` calls.
    assert_eq!(ex("a -> b"), "json_extract(a, b)");
    assert_eq!(ex("a ->> b"), "json_extract_string(a, b)");
    // They chain left-associatively (`a -> b -> c` = `(a -> b) -> c`).
    assert_eq!(ex("a -> b -> c"), "json_extract(json_extract(a, b), c)");
    // Like Postgres's "other operators" band they have the same binding power as `||`, so
    // they bind tighter than comparison (`doc -> 'a' = 1` needs no parentheses).
    assert_eq!(ex("a -> b = c"), "(json_extract(a, b) = c)");
    assert_eq!(ex("a ->> b = c"), "(json_extract_string(a, b) = c)");
    // Distinguished from a plain `-` subtraction.
    assert_eq!(ex("a - b"), "(a - b)");
}

#[test]
fn lambda_is_recognized_only_in_list_transform_filter_reduce_arg_position() {
    // A single parameter needs no parentheses.
    assert_eq!(ex("list_transform(a, x -> x + 1)"), "list_transform(a, x -> (x + 1i32))");
    // Several parameters need parentheses.
    assert_eq!(ex("list_reduce(a, (acc, x) -> acc + x)"), "list_reduce(a, (acc, x) -> (acc + x))");
    // A parenthesized single parameter is allowed too (same as duckdb).
    assert_eq!(ex("list_filter(a, (x) -> x > 1)"), "list_filter(a, x -> (x > 1i32))");
    // Nested lambdas.
    assert_eq!(
        ex("list_transform(a, y -> list_transform(y, x -> x * 2))"),
        "list_transform(a, y -> list_transform(y, x -> (x * 2i32)))"
    );
    // list_filter gets the same treatment.
    assert_eq!(ex("list_filter(a, x -> x > 5)"), "list_filter(a, x -> (x > 5i32))");
    // The function-name check is case-insensitive.
    assert_eq!(ex("LIST_TRANSFORM(a, x -> x)"), "LIST_TRANSFORM(a, x -> x)");

    // In the argument positions of any other function, `->` stays the JSON path operator as
    // before and is not interpreted as a lambda (measured with the duckdb CLI:
    // `coalesce(doc -> 'a', 'x')` still resolves as JSON extraction).
    assert_eq!(ex("coalesce(doc -> 'a', x)"), "coalesce(json_extract(doc, 'a'), x)");
}

#[test]
fn try_cast_expr() {
    assert_eq!(ex("TRY_CAST(x AS INTEGER)"), "TRY_CAST(x AS INTEGER)");
    assert_eq!(ex("try_cast(x AS DECIMAL(10,2))"), "TRY_CAST(x AS DECIMAL(10,2))");
    // Distinguished as a separate node from an ordinary CAST.
    assert_ne!(ex("CAST(x AS INTEGER)"), ex("TRY_CAST(x AS INTEGER)"));
    assert_eq!(code("SELECT TRY_CAST(x AS FROB)"), Code::InvalidCast as u16);
    // `try_cast` is not reserved, so it remains usable as an ordinary identifier (column or
    // function name); without a following `(` it is a plain column reference.
    assert_eq!(ex("try_cast"), "try_cast");
}

#[test]
fn iif_desugars_to_case() {
    assert_eq!(ex("IIF(a > 1, 'x', 'y')"), "CASE WHEN (a > 1i32) THEN 'x' ELSE 'y' END");
    assert_eq!(ex("iif(a, 1, 2)"), "CASE WHEN a THEN 1i32 ELSE 2i32 END");
    assert_eq!(code("SELECT IIF(a, 1)"), Code::UnexpectedToken as u16);
    // `iif` is not reserved either.
    assert_eq!(ex("iif"), "iif");
}

#[test]
fn interval_literals() {
    // The compound string form. The packed result is compared directly.
    let lit =
        |m: i32, d: i32, u: i64| format!("INTERVAL({}i128)", crate::vector::pack_interval(m, d, u));
    assert_eq!(ex("INTERVAL '3 days'"), lit(0, 3, 0));
    assert_eq!(ex("INTERVAL '1 year 2 months 3 days'"), ex("INTERVAL '14 months 3 days'"),);
    // Singular, plural, and case are all accepted.
    assert_eq!(ex("interval '1 DAY'"), ex("INTERVAL '1 days'"));
    assert_eq!(ex("INTERVAL '1 month'"), ex("INTERVAL '1 months'"));
    // Negative values (a sign inside the string).
    assert_eq!(ex("INTERVAL '-3 days'"), lit(0, -3, 0));
    // A unary minus is wrapped in `Unary::Neg` (a separate node), so whether it expands to
    // a dedicated kernel is checked on the `plan::compile` side. Here only that it parses
    // is checked.
    assert_eq!(ex("-INTERVAL '3 days'"), format!("(- {})", lit(0, 3, 0)));
    // The `'n' UNIT` form.
    assert_eq!(ex("INTERVAL '3' DAY"), ex("INTERVAL '3 days'"));
    assert_eq!(ex("INTERVAL '1' MONTH"), ex("INTERVAL '1 month'"));
    // The unquoted `n UNIT` form.
    assert_eq!(ex("INTERVAL 3 DAY"), ex("INTERVAL '3 days'"));
    // Sub-second units.
    assert_eq!(ex("INTERVAL '1500 milliseconds'"), ex("INTERVAL '1500000 microseconds'"));
    // `interval` is not reserved, so it is usable as a column reference.
    assert_eq!(ex("interval"), "interval");
    assert_eq!(ex("interval + 1"), "(interval + 1i32)");
    assert_eq!(code("SELECT INTERVAL 'nonsense'"), Code::SyntaxError as u16);
    assert_eq!(code("SELECT INTERVAL '3 fortnights'"), Code::SyntaxError as u16);
}

#[test]
fn functions() {
    assert_eq!(ex("count(*)"), "count(*)");
    assert_eq!(ex("count(DISTINCT x)"), "count(DISTINCT x)");
    assert_eq!(ex("now()"), "now()");
    assert_eq!(ex("upper(substr(s, 1, 3))"), "upper(substr(s, 1i32, 3i32))");
    assert_eq!(ex("abs(-a) + round(b, 2)"), "(abs((- a)) + round(b, 2i32))");
    assert_eq!(code("SELECT f(1,"), Code::UnexpectedToken as u16);
}

#[test]
fn filter_clause() {
    assert_eq!(ex("count(*) FILTER (WHERE a > 1)"), "count(*) FILTER (WHERE (a > 1i32))");
    assert_eq!(ex("sum(x) FILTER (WHERE a > 1 AND b)"), "sum(x) FILTER (WHERE ((a > 1i32) AND b))");
    assert_eq!(ex("count(*)"), "count(*)", "without FILTER, unchanged");
    // `FILTER` is not reserved, so without a following `(` it passes as an ordinary alias.
    assert_eq!(sel("SELECT count(*) filter FROM t"), "SELECT count(*) AS filter FROM t");
    // `FILTER` is unsupported on window functions (out of scope).
    assert_eq!(
        code("SELECT count(*) FILTER (WHERE a > 1) OVER () FROM t"),
        Code::UnsupportedFeature as u16
    );
}

#[test]
fn column_refs_and_params() {
    assert_eq!(ex("a"), "a");
    assert_eq!(ex("t.a"), "t.a");
    // Flattened STRUCT leaves: dots after the first become the column name.
    assert_eq!(ex("address.city"), "address.city");
    assert_eq!(ex("nested.a.b.c"), "nested.a.b.c");
    assert_eq!(ex("\"Mixed Case\".\"x\"\"y\""), "Mixed Case.x\"y");
    let p = parse("SELECT ? WHERE a = ? AND b = ?").expect("parse");
    assert_eq!(p.num_params, 3);
    match &p.stmt {
        Stmt::Select(q) => {
            let s = plain(q);
            assert_eq!(r(&p.arena, s.items[0].expr), "?0");
            assert_eq!(r(&p.arena, s.filter.expect("filter")), "((a = ?1) AND (b = ?2))");
        }
        _ => panic!("not select"),
    }
    assert_eq!(parse("SELECT 1").expect("parse").num_params, 0);
}

// --- Literals -----------------------------------------------------------

#[test]
fn integer_literal_widths() {
    assert_eq!(ex("2147483647"), "2147483647i32");
    assert_eq!(ex("2147483648"), "2147483648i64");
    assert_eq!(ex("-2147483648"), "-2147483648i32");
    assert_eq!(ex("-2147483649"), "-2147483649i64");
    assert_eq!(ex("9223372036854775807"), "9223372036854775807i64");
    assert_eq!(ex("-9223372036854775808"), "-9223372036854775808i64");
    assert_eq!(ex("9223372036854775808"), "9223372036854775808i128");
    assert_eq!(
        ex("170141183460469231731687303715884105727"),
        "170141183460469231731687303715884105727i128"
    );
    assert_eq!(
        ex("-170141183460469231731687303715884105728"),
        "-170141183460469231731687303715884105728i128"
    );
    assert_eq!(code("SELECT 170141183460469231731687303715884105728"), Code::NumberOverflow as u16);
    assert_eq!(
        code("SELECT 99999999999999999999999999999999999999999"),
        Code::NumberOverflow as u16
    );
    // A unary minus folds, but against an expression it stays an ordinary operator.
    assert_eq!(ex("-(1)"), "(- 1i32)");
}

#[test]
fn other_literals() {
    assert_eq!(ex("1.5"), "1.5f64");
    assert_eq!(ex("1e3"), "1000f64");
    assert_eq!(ex("1.5e-2"), "0.015f64");
    assert_eq!(ex("TRUE"), "true");
    assert_eq!(ex("false"), "false");
    assert_eq!(ex("NULL"), "NULL");
    assert_eq!(ex("'abc'"), "'abc'");
    assert_eq!(ex("''"), "''");
    assert_eq!(ex("'it''s'"), "'it's'");
    assert_eq!(ex("'a''''b'"), "'a''b'");
    assert_eq!(code("SELECT 'abc"), Code::UnterminatedString as u16);
    assert_eq!(code("SELECT 'it''s"), Code::UnterminatedString as u16);
}

// --- Whole statements ---------------------------------------------------

#[test]
fn full_select() {
    assert_eq!(
            sel("SELECT DISTINCT a AS x, b y, t.*, count(*) FROM t WHERE a > 1 GROUP BY a, b HAVING count(*) > 2 ORDER BY a DESC, b ASC NULLS FIRST LIMIT 10 OFFSET 5"),
            "SELECT DISTINCT a AS x, b AS y, t.*, count(*) FROM t WHERE (a > 1i32) \
             GROUP BY a, b HAVING (count(*) > 2i32) \
             ORDER BY a DESC NULLS LAST, b ASC NULLS FIRST LIMIT 10 OFFSET 5"
        );
    // The default NULL order is always LAST, for both ASC and DESC
    // (matches DuckDB, not the SQL-standard/PostgreSQL convention).
    assert_eq!(sel("SELECT a FROM t ORDER BY a"), "SELECT a FROM t ORDER BY a ASC NULLS LAST");
    assert_eq!(
        sel("SELECT a FROM t ORDER BY a DESC NULLS LAST"),
        "SELECT a FROM t ORDER BY a DESC NULLS LAST"
    );
    assert_eq!(sel("SELECT * FROM t;"), "SELECT * FROM t");
    assert_eq!(sel("select 1"), "SELECT 1i32");
}

#[test]
fn qualify_clause() {
    // Even with `QUALIFY` right after the table reference and no intervening
    // WHERE/GROUP BY/HAVING, it is correctly read as a clause rather than eaten as a table
    // alias (the same trap as the past ROWS/RANGE incident; the reason `QUALIFY` was made fully reserved).
    assert_eq!(sel("SELECT a FROM t QUALIFY a > 1"), "SELECT a FROM t QUALIFY (a > 1i32)");
    // Its placement: after GROUP BY / HAVING and before ORDER BY.
    assert_eq!(
            sel("SELECT a, count(*) FROM t GROUP BY a HAVING count(*) > 1 QUALIFY a > 0 ORDER BY a"),
            "SELECT a, count(*) FROM t GROUP BY a HAVING (count(*) > 1i32) QUALIFY (a > 0i32) ORDER BY a ASC NULLS LAST"
        );
    // Being fully reserved, it is unusable as a column name.
    assert_eq!(code("SELECT qualify FROM t"), Code::UnexpectedToken as u16);
}

#[test]
fn star_exclude_replace() {
    // The basic form. The order is fixed as EXCLUDE then REPLACE (confirmed with `duckdb`).
    assert_eq!(sel("SELECT * EXCLUDE (b) FROM t"), "SELECT * EXCLUDE (b) FROM t");
    assert_eq!(
        sel("SELECT * REPLACE (a + 1 AS a) FROM t"),
        "SELECT * REPLACE ((a + 1i32) AS a) FROM t"
    );
    assert_eq!(
        sel("SELECT * EXCLUDE (b) REPLACE (a + 1 AS a) FROM t"),
        "SELECT * EXCLUDE (b) REPLACE ((a + 1i32) AS a) FROM t"
    );
    // The reverse order (EXCLUDE after REPLACE) is a syntax error, as in `duckdb`.
    assert_eq!(
        code("SELECT * REPLACE (a + 1 AS a) EXCLUDE (b) FROM t"),
        Code::UnexpectedToken as u16
    );
    // A single entry may omit the parentheses (`duckdb`'s behavior).
    assert_eq!(sel("SELECT * EXCLUDE b FROM t"), "SELECT * EXCLUDE (b) FROM t");
    assert_eq!(sel("SELECT * REPLACE 1 AS a FROM t"), "SELECT * REPLACE (1i32 AS a) FROM t");
    // Several columns require parentheses.
    assert_eq!(sel("SELECT * EXCLUDE (a, b) FROM t"), "SELECT * EXCLUDE (a, b) FROM t");
    assert_eq!(
        sel("SELECT * REPLACE (1 AS a, 2 AS b) FROM t"),
        "SELECT * REPLACE (1i32 AS a, 2i32 AS b) FROM t"
    );
    // They attach to `t.*` in the same way.
    assert_eq!(sel("SELECT t.* EXCLUDE (b) FROM t"), "SELECT t.* EXCLUDE (b) FROM t");
    // Putting the same column in both EXCLUDE and REPLACE is meaningless, so it is rejected.
    assert_eq!(code("SELECT * EXCLUDE (a) REPLACE (1 AS a) FROM t"), Code::SyntaxError as u16);
    // Duplicates within EXCLUDE and within REPLACE are each rejected too.
    assert_eq!(code("SELECT * EXCLUDE (a, a) FROM t"), Code::SyntaxError as u16);
    assert_eq!(code("SELECT * REPLACE (1 AS a, 2 AS a) FROM t"), Code::SyntaxError as u16);
    // `EXCLUDE` is a keyword only in the context right after `*`. That is the same class of
    // trap as the past ROWS/RANGE/QUALIFY incidents, so this confirms it is still usable as
    // an ordinary column name or alias. `REPLACE` becomes a separate global reserved word
    // for `CREATE OR REPLACE` when the `ddl` feature is on, so it is excluded in that case
    // (see the comment on `is_star_replace_kw`).
    assert_eq!(sel("SELECT exclude FROM t"), "SELECT exclude FROM t");
    assert_eq!(sel("SELECT a AS exclude FROM t"), "SELECT a AS exclude FROM t");
    #[cfg(not(feature = "ddl"))]
    assert_eq!(sel("SELECT exclude, replace FROM t"), "SELECT exclude, replace FROM t");
}

/// `SELECT * RENAME (...)`, the third star modifier. All expected forms and
/// error cases below are cross-checked against a real `duckdb` CLI.
#[test]
fn star_rename() {
    // Basic parenthesized form.
    assert_eq!(sel("SELECT * RENAME (a AS z) FROM t"), "SELECT * RENAME (a AS z) FROM t");
    // Bare form (no parens) for a single entry, same convention as
    // EXCLUDE/REPLACE.
    assert_eq!(sel("SELECT * RENAME a AS z FROM t"), "SELECT * RENAME (a AS z) FROM t");
    // Multiple entries require parens.
    assert_eq!(
        sel("SELECT * RENAME (a AS z, b AS y) FROM t"),
        "SELECT * RENAME (a AS z, b AS y) FROM t"
    );
    // Qualified star.
    assert_eq!(sel("SELECT t.* RENAME (a AS z) FROM t"), "SELECT t.* RENAME (a AS z) FROM t");
    // Combines with EXCLUDE / REPLACE. Fixed order is
    // EXCLUDE -> REPLACE -> RENAME (verified against `duckdb`).
    assert_eq!(
        sel("SELECT * EXCLUDE (b) RENAME (a AS z) FROM t"),
        "SELECT * EXCLUDE (b) RENAME (a AS z) FROM t"
    );
    assert_eq!(
        sel("SELECT * REPLACE (b + 10 AS b) RENAME (a AS z) FROM t"),
        "SELECT * REPLACE ((b + 10i32) AS b) RENAME (a AS z) FROM t"
    );
    assert_eq!(
        sel("SELECT * EXCLUDE (c) REPLACE (b + 10 AS b) RENAME (a AS z) FROM t"),
        "SELECT * EXCLUDE (c) REPLACE ((b + 10i32) AS b) RENAME (a AS z) FROM t"
    );

    // RENAME must come last: EXCLUDE/REPLACE after RENAME is a parser error,
    // same as the already-tested REPLACE-before-EXCLUDE case.
    assert_eq!(code("SELECT * RENAME (a AS z) EXCLUDE (b) FROM t"), Code::UnexpectedToken as u16);
    assert_eq!(
        code("SELECT * RENAME (a AS z) REPLACE (1 AS b) FROM t"),
        Code::UnexpectedToken as u16
    );

    // The same source column named twice in one RENAME list is rejected.
    assert_eq!(code("SELECT * RENAME (a AS z, a AS y) FROM t"), Code::SyntaxError as u16);
    // A column in both EXCLUDE and RENAME, or both REPLACE and RENAME, is
    // rejected (`duckdb`: "Column ... cannot occur in both ... list").
    assert_eq!(code("SELECT * EXCLUDE (a) RENAME (a AS z) FROM t"), Code::SyntaxError as u16);
    assert_eq!(code("SELECT * REPLACE (1 AS a) RENAME (a AS z) FROM t"), Code::SyntaxError as u16);
    // Renaming a source column onto a name that is itself immediately
    // renamed elsewhere is a duplicate *source*, not a duplicate *target*,
    // so it is allowed at parse time (target-name collisions are only
    // checked/allowed at bind time; see the `star_rename_*` integration
    // tests).
    assert_eq!(
        sel("SELECT * RENAME (a AS z, b AS z) FROM t"),
        "SELECT * RENAME (a AS z, b AS z) FROM t"
    );

    // `RENAME` is only a keyword right after `*`/`t.*` (same convention as
    // EXCLUDE/REPLACE), so it stays usable as an ordinary identifier
    // elsewhere. Under `ddl`, `rename` is a real reserved word (used by
    // `ALTER TABLE ... RENAME`, see `is_star_rename_kw`), so this part is
    // only checked without that feature — mirroring how the REPLACE test
    // above handles the same `ddl` interaction.
    #[cfg(not(feature = "ddl"))]
    {
        assert_eq!(sel("SELECT rename FROM t"), "SELECT rename FROM t");
        assert_eq!(sel("SELECT a AS rename FROM t"), "SELECT a AS rename FROM t");
    }
}

/// DuckDB's `COLUMNS(...)` star expression. Every accepted and rejected form
/// below is cross-checked against a real `duckdb` v1.4.4 CLI.
///
/// Note the round-trip strings put the star modifiers *after* the closing
/// paren (`COLUMNS(*) EXCLUDE (b)`) while the real syntax puts them inside
/// it. `select_str` is a structural dump, not valid SQL (see its doc), and
/// `COLUMNS(...)` shares the `Expr::Star` node with a plain `*`.
#[test]
fn columns_star_expression() {
    // The three supported argument forms.
    assert_eq!(sel("SELECT COLUMNS(*) FROM t"), "SELECT COLUMNS(*) FROM t");
    assert_eq!(sel("SELECT COLUMNS('n.*') FROM t"), "SELECT COLUMNS('n.*') FROM t");
    assert_eq!(sel("SELECT COLUMNS(['id', 'num']) FROM t"), "SELECT COLUMNS(['id', 'num']) FROM t");
    // A single-element list is still a list, not the regex form.
    assert_eq!(sel("SELECT COLUMNS(['id']) FROM t"), "SELECT COLUMNS(['id']) FROM t");
    // The star modifiers go *inside* the parens (`duckdb` rejects
    // `COLUMNS(*) EXCLUDE (b)` as a parser error), and all three still apply
    // in their usual fixed order.
    assert_eq!(sel("SELECT COLUMNS(* EXCLUDE (b)) FROM t"), "SELECT COLUMNS(*) EXCLUDE (b) FROM t");
    assert_eq!(
        sel("SELECT COLUMNS(* EXCLUDE (c) REPLACE (b + 1 AS b) RENAME (a AS z)) FROM t"),
        "SELECT COLUMNS(*) EXCLUDE (c) REPLACE ((b + 1i32) AS b) RENAME (a AS z) FROM t"
    );
    assert_eq!(code("SELECT COLUMNS(*) EXCLUDE (b) FROM t"), Code::UnexpectedToken as u16);
    // `AS '<template>'` is the capture-group renaming form, so the alias may
    // be a string literal here — `opt_alias` alone would reject that.
    assert_eq!(
        sel("SELECT COLUMNS('(a)b') AS '\\1' FROM t"),
        "SELECT COLUMNS('(a)b') AS \\1 FROM t"
    );
    assert_eq!(sel("SELECT COLUMNS(*) AS x FROM t"), "SELECT COLUMNS(*) AS x FROM t");

    // --- forms that must fail loudly ---------------------------------------
    // Distributing an enclosing function over the expansion.
    assert_eq!(code("SELECT min(COLUMNS(*)) FROM t"), Code::UnsupportedFeature as u16);
    // ... including over a plain operator, which `duckdb` also distributes
    // (`COLUMNS(*) + 1` yields one `+ 1` column per input column there).
    assert_eq!(code("SELECT COLUMNS(*) + 1 FROM t"), Code::UnsupportedFeature as u16);
    // `UNPACK(...)` / `*COLUMNS(...)` unpacking into a parent expression.
    assert_eq!(code("SELECT UNPACK(COLUMNS(*)) FROM t"), Code::UnsupportedFeature as u16);
    assert_eq!(code("SELECT *COLUMNS(*) FROM t"), Code::UnsupportedFeature as u16);
    // The lambda predicate form.
    assert_eq!(code("SELECT COLUMNS(c -> c LIKE 'n%') FROM t"), Code::UnsupportedFeature as u16);
    // `* LIKE`/`GLOB`/`SIMILAR TO` star filtering.
    assert_eq!(code("SELECT * LIKE 'n%' FROM t"), Code::UnsupportedFeature as u16);
    assert_eq!(code("SELECT * ILIKE 'n%' FROM t"), Code::UnsupportedFeature as u16);
    assert_eq!(code("SELECT * NOT LIKE 'n%' FROM t"), Code::UnsupportedFeature as u16);
    assert_eq!(code("SELECT * GLOB 'n*' FROM t"), Code::UnsupportedFeature as u16);
    assert_eq!(code("SELECT * SIMILAR TO 'n.*' FROM t"), Code::UnsupportedFeature as u16);
    // A qualified `t.COLUMNS(*)` — `duckdb` rejects this too.
    assert_eq!(code("SELECT t.COLUMNS(*) FROM t"), Code::UnsupportedFeature as u16);
    // `COLUMNS` outside the select list.
    assert_eq!(code("SELECT a FROM t WHERE COLUMNS('a') > 1"), Code::UnsupportedFeature as u16);

    // `COLUMNS` is not a reserved word: a column (or alias, or table) that
    // happens to be named `columns` still works unquoted, the same guarantee
    // EXCLUDE/REPLACE/RENAME carry. `duckdb` behaves the same way.
    assert_eq!(sel("SELECT columns FROM t"), "SELECT columns FROM t");
    assert_eq!(sel("SELECT a AS columns FROM t"), "SELECT a AS columns FROM t");
    assert_eq!(sel("SELECT columns.a FROM columns"), "SELECT columns.a FROM columns");
    assert_eq!(sel("SELECT columns.* FROM columns"), "SELECT columns.* FROM columns");
    assert_eq!(sel("SELECT a FROM t WHERE columns > 1"), "SELECT a FROM t WHERE (columns > 1i32)");
    assert_eq!(sel("SELECT unpack FROM t"), "SELECT unpack FROM t");
}

#[test]
fn grouping_sets_rollup_cube_syntax() {
    // `GROUPING SETS` keeps the sequence of sets as written. The empty set `()` is one element too.
    assert_eq!(
        sel("SELECT a, b, sum(c) FROM t GROUP BY GROUPING SETS ((a, b), (a), ())"),
        "SELECT a, b, sum(c) FROM t GROUP BY GROUPING SETS ((a, b), (a), ())"
    );
    // `ROLLUP (a, b, c)` expands into hierarchical subsets from more columns to fewer.
    assert_eq!(
        sel("SELECT a, b, c, sum(d) FROM t GROUP BY ROLLUP (a, b, c)"),
        "SELECT a, b, c, sum(d) FROM t GROUP BY GROUPING SETS ((a, b, c), (a, b), (a), ())"
    );
    // `CUBE (a, b)` expands into every subset (2^n of them).
    assert_eq!(
        sel("SELECT a, b, sum(c) FROM t GROUP BY CUBE (a, b)"),
        "SELECT a, b, sum(c) FROM t GROUP BY GROUPING SETS ((a, b), (a), (b), ())"
    );
    // Single-column ROLLUP/CUBE.
    assert_eq!(
        sel("SELECT a, sum(c) FROM t GROUP BY ROLLUP (a)"),
        "SELECT a, sum(c) FROM t GROUP BY GROUPING SETS ((a), ())"
    );

    // `GROUPING`/`SETS`/`ROLLUP`/`CUBE` are keywords only in the context right after GROUP
    // BY. That is the same class of trap as the past ROWS/RANGE/QUALIFY incidents, so this
    // confirms they are still usable as ordinary column names and aliases
    // (`SETS` needs no check, as it is not special in any other context).
    assert_eq!(sel("SELECT grouping FROM t"), "SELECT grouping FROM t");
    assert_eq!(sel("SELECT rollup, cube FROM t"), "SELECT rollup, cube FROM t");
    assert_eq!(sel("SELECT a AS rollup FROM t"), "SELECT a AS rollup FROM t");
    assert_eq!(
        sel("SELECT a FROM t WHERE grouping > 0"),
        "SELECT a FROM t WHERE (grouping > 0i32)"
    );

    // `GROUPING(...)` passes as an ordinary function call.
    assert_eq!(ex("grouping(a)"), "grouping(a)");
    assert_eq!(ex("grouping(a, b)"), "grouping(a, b)");
}

#[test]
fn distinct_on_clause() {
    assert_eq!(sel("SELECT DISTINCT ON (a) a, b FROM t"), "SELECT DISTINCT ON (a) a, b FROM t");
    assert_eq!(sel("SELECT DISTINCT ON (a, b) * FROM t"), "SELECT DISTINCT ON (a, b) * FROM t");
    // `DISTINCT ON` is mutually exclusive with a plain `DISTINCT` (never both at once).
    let p = parse("SELECT DISTINCT ON (a) a FROM t").expect("parse");
    match &p.stmt {
        Stmt::Select(q) => {
            let s = plain(q);
            assert!(!s.distinct);
            assert_eq!(s.distinct_on.len(), 1);
        }
        _ => panic!("not select"),
    }
    assert_eq!(code("SELECT DISTINCT ON () a FROM t"), Code::UnexpectedToken as u16);
}

#[test]
fn from_items_and_joins() {
    assert_eq!(sel("SELECT * FROM t a"), "SELECT * FROM t AS a");
    assert_eq!(sel("SELECT * FROM t AS a"), "SELECT * FROM t AS a");
    assert_eq!(
        sel("SELECT * FROM parquet('path/to.parquet') AS t"),
        "SELECT * FROM parquet('path/to.parquet') AS t"
    );
    assert_eq!(
        sel("SELECT * FROM PARQUET('a''b.parquet')"),
        "SELECT * FROM parquet('a'b.parquet')"
    );
    assert_eq!(
        sel("SELECT * FROM (SELECT a FROM t WHERE a > 0) s"),
        "SELECT * FROM (SELECT a FROM t WHERE (a > 0i32)) AS s"
    );
    assert_eq!(
        sel("SELECT * FROM a JOIN b ON a.x = b.x"),
        "SELECT * FROM (a INNER JOIN b ON (a.x = b.x))"
    );
    assert_eq!(
        sel("SELECT * FROM a INNER JOIN b ON a.x = b.x"),
        "SELECT * FROM (a INNER JOIN b ON (a.x = b.x))"
    );
    assert_eq!(
        sel("SELECT * FROM a LEFT OUTER JOIN b ON a.x = b.x"),
        "SELECT * FROM (a LEFT JOIN b ON (a.x = b.x))"
    );
    assert_eq!(
        sel("SELECT * FROM a RIGHT JOIN b ON a.x = b.x"),
        "SELECT * FROM (a RIGHT JOIN b ON (a.x = b.x))"
    );
    assert_eq!(
        sel("SELECT * FROM a FULL OUTER JOIN b ON a.x = b.x"),
        "SELECT * FROM (a FULL JOIN b ON (a.x = b.x))"
    );
    assert_eq!(sel("SELECT * FROM a CROSS JOIN b"), "SELECT * FROM (a CROSS JOIN b)");
    // A comma join is an implicit CROSS JOIN. Stacked left-deep.
    assert_eq!(sel("SELECT * FROM a, b, c"), "SELECT * FROM ((a CROSS JOIN b) CROSS JOIN c)");
    assert_eq!(
        sel("SELECT * FROM a x JOIN b y ON x.k = y.k LEFT JOIN c ON c.k = x.k"),
        "SELECT * FROM ((a AS x INNER JOIN b AS y ON (x.k = y.k)) LEFT JOIN c ON (c.k = x.k))"
    );
    assert_eq!(code("SELECT * FROM a JOIN b"), Code::UnexpectedToken as u16);
    assert_eq!(code("SELECT * FROM foo(1)"), Code::UnsupportedFeature as u16);
}

// --- CTE ----------------------------------------------------------------

#[test]
fn ctes() {
    assert_eq!(
        qs("WITH a AS (SELECT 1) SELECT * FROM a"),
        "WITH a AS (SELECT 1i32) SELECT * FROM a"
    );
    // Several CTEs line up in definition order.
    assert_eq!(
        qs("WITH a AS (SELECT x FROM t), b AS (SELECT y FROM u) SELECT * FROM a, b"),
        "WITH a AS (SELECT x FROM t), b AS (SELECT y FROM u) \
             SELECT * FROM (a CROSS JOIN b)"
    );
    // A later CTE can reference an earlier one (forward references are the binder's responsibility).
    assert_eq!(
        qs("WITH a AS (SELECT x FROM t), b AS (SELECT x FROM a) SELECT * FROM b"),
        "WITH a AS (SELECT x FROM t), b AS (SELECT x FROM a) SELECT * FROM b"
    );
    // A CTE's contents are a complete query too. Set operations and ORDER BY may be written.
    assert_eq!(
        qs("WITH a AS (SELECT 1 UNION SELECT 2) SELECT * FROM a"),
        "WITH a AS ((SELECT 1i32 UNION SELECT 2i32)) SELECT * FROM a"
    );
    assert_eq!(
        qs("WITH a AS (SELECT x FROM t ORDER BY x LIMIT 3) SELECT * FROM a"),
        "WITH a AS (SELECT x FROM t ORDER BY x ASC NULLS LAST LIMIT 3) SELECT * FROM a"
    );
    // Nested CTEs.
    assert_eq!(
        qs("WITH a AS (WITH b AS (SELECT 1) SELECT * FROM b) SELECT * FROM a"),
        "WITH a AS (WITH b AS (SELECT 1i32) SELECT * FROM b) SELECT * FROM a"
    );
    // A CTE can sit under EXPLAIN too.
    assert!(matches!(
        parse("EXPLAIN WITH a AS (SELECT 1) SELECT * FROM a").expect("parse").stmt,
        Stmt::Explain(_)
    ));
}

// --- WITH RECURSIVE -------------------------------------------------------

#[test]
fn with_recursive_parses() {
    // `RECURSIVE` is a context-dependent keyword only right after WITH. Whether a CTE
    // actually references itself is decided at bind time (`plan::bind`), so the parser only
    // sets the flag and does not care about the body's shape.
    assert_eq!(
        qs("WITH RECURSIVE x AS (SELECT 1) SELECT 1"),
        "WITH RECURSIVE x AS (SELECT 1i32) SELECT 1i32"
    );
    // Column lists are allowed only under `WITH RECURSIVE`.
    assert_eq!(
        qs("WITH RECURSIVE fib(n, a, b) AS \
                 (SELECT 0, 0, 1 UNION ALL SELECT n+1, b, a+b FROM fib WHERE n < 10) \
                 SELECT * FROM fib"),
        "WITH RECURSIVE fib(n, a, b) AS \
             ((SELECT 0i32, 0i32, 1i32 UNION ALL SELECT (n + 1i32), b, (a + b) FROM fib \
             WHERE (n < 10i32))) SELECT * FROM fib"
    );
    // `RECURSIVE` applies to the whole list. Some CTEs may in fact be non-recursive.
    assert_eq!(
        qs("WITH RECURSIVE base AS (SELECT 1 AS x), \
                 t AS (SELECT x AS n FROM base UNION ALL SELECT n+1 FROM t WHERE n < 5) \
                 SELECT * FROM t"),
        "WITH RECURSIVE base AS (SELECT 1i32 AS x), \
             t AS ((SELECT x AS n FROM base UNION ALL SELECT (n + 1i32) FROM t \
             WHERE (n < 5i32))) SELECT * FROM t"
    );
    // A CTE named `recursive` is still writable as before (`AS` follows immediately, so it
    // is not consumed as a keyword).
    assert_eq!(
        qs("WITH recursive AS (SELECT 1) SELECT * FROM recursive"),
        "WITH recursive AS (SELECT 1i32) SELECT * FROM recursive"
    );
    // A column list, however, is indistinguishable from "the name after `RECURSIVE`", so
    // `WITH recursive(a) AS (...)` falls to the column-list branch (unsupported without
    // `RECURSIVE`). DuckDB allows it, but this engine does not support column lists on
    // non-recursive CTEs at all, so it consistently errors.
    assert_eq!(
        code("WITH recursive(a) AS (SELECT 1) SELECT * FROM recursive"),
        Code::UnsupportedFeature as u16
    );
}

// --- Set operations -----------------------------------------------------

#[test]
fn set_operations() {
    assert_eq!(
        qs("SELECT a FROM t UNION SELECT b FROM u"),
        "(SELECT a FROM t UNION SELECT b FROM u)"
    );
    assert_eq!(
        qs("SELECT a FROM t UNION ALL SELECT b FROM u"),
        "(SELECT a FROM t UNION ALL SELECT b FROM u)"
    );
    assert_eq!(qs("SELECT 1 INTERSECT SELECT 2"), "(SELECT 1i32 INTERSECT SELECT 2i32)");
    assert_eq!(qs("SELECT 1 INTERSECT ALL SELECT 2"), "(SELECT 1i32 INTERSECT ALL SELECT 2i32)");
    assert_eq!(qs("SELECT 1 EXCEPT SELECT 2"), "(SELECT 1i32 EXCEPT SELECT 2i32)");
    assert_eq!(qs("SELECT 1 EXCEPT ALL SELECT 2"), "(SELECT 1i32 EXCEPT ALL SELECT 2i32)");
    // Parenthesized terms.
    assert_eq!(qs("(SELECT 1) UNION (SELECT 2)"), "(SELECT 1i32 UNION SELECT 2i32)");
}

#[test]
fn set_operation_precedence() {
    // INTERSECT binds tighter than UNION / EXCEPT.
    assert_eq!(
        qs("SELECT 1 UNION SELECT 2 INTERSECT SELECT 3"),
        "(SELECT 1i32 UNION (SELECT 2i32 INTERSECT SELECT 3i32))"
    );
    assert_eq!(
        qs("SELECT 1 INTERSECT SELECT 2 UNION SELECT 3"),
        "((SELECT 1i32 INTERSECT SELECT 2i32) UNION SELECT 3i32)"
    );
    assert_eq!(
        qs("SELECT 1 EXCEPT SELECT 2 INTERSECT SELECT 3"),
        "(SELECT 1i32 EXCEPT (SELECT 2i32 INTERSECT SELECT 3i32))"
    );
    // UNION / EXCEPT are left-associative. That matters here because EXCEPT is not associative.
    assert_eq!(
        qs("SELECT 1 EXCEPT SELECT 2 EXCEPT SELECT 3"),
        "((SELECT 1i32 EXCEPT SELECT 2i32) EXCEPT SELECT 3i32)"
    );
    assert_eq!(
        qs("SELECT 1 UNION SELECT 2 EXCEPT SELECT 3"),
        "((SELECT 1i32 UNION SELECT 2i32) EXCEPT SELECT 3i32)"
    );
    assert_eq!(
        qs("SELECT 1 INTERSECT SELECT 2 INTERSECT SELECT 3"),
        "((SELECT 1i32 INTERSECT SELECT 2i32) INTERSECT SELECT 3i32)"
    );
    // Parentheses can change the associativity.
    assert_eq!(
        qs("SELECT 1 EXCEPT (SELECT 2 EXCEPT SELECT 3)"),
        "(SELECT 1i32 EXCEPT (SELECT 2i32 EXCEPT SELECT 3i32))"
    );
    assert_eq!(
        qs("(SELECT 1 UNION SELECT 2) INTERSECT SELECT 3"),
        "((SELECT 1i32 UNION SELECT 2i32) INTERSECT SELECT 3i32)"
    );

    // The tree shape is checked directly (not relying on the rendering's parentheses alone).
    let (_, q) = parsed("SELECT 1 UNION SELECT 2 INTERSECT SELECT 3");
    match &q.body {
        SetExpr::SetOp { op, left, right, .. } => {
            assert_eq!(*op, SetOp::Union);
            assert!(matches!(**left, SetExpr::Select(_)));
            assert!(matches!(**right, SetExpr::SetOp { op: SetOp::Intersect, .. }));
        }
        _ => panic!("not a set operation"),
    }
    let (_, q) = parsed("SELECT 1 EXCEPT SELECT 2 EXCEPT SELECT 3");
    match &q.body {
        SetExpr::SetOp { op, left, right, .. } => {
            assert_eq!(*op, SetOp::Except);
            assert!(matches!(**left, SetExpr::SetOp { op: SetOp::Except, .. }));
            assert!(matches!(**right, SetExpr::Select(_)));
        }
        _ => panic!("not a set operation"),
    }
}

#[test]
fn trailing_clauses_placement() {
    // An ORDER BY / LIMIT / OFFSET after a set operation attaches to the outer QueryStmt.
    let (_, q) = parsed("SELECT a FROM t UNION SELECT b FROM u ORDER BY 1 LIMIT 5 OFFSET 2");
    assert_eq!(q.order_by.len(), 1);
    assert_eq!(q.limit, Some(5));
    assert_eq!(q.offset, Some(2));
    match &q.body {
        SetExpr::SetOp { left, right, .. } => {
            for side in [left, right] {
                match &**side {
                    SetExpr::Select(s) => {
                        assert!(s.order_by.is_empty());
                        assert!(s.limit.is_none());
                        assert!(s.offset.is_none());
                    }
                    _ => panic!("not a SELECT"),
                }
            }
        }
        _ => panic!("not a set operation"),
    }
    assert_eq!(
        qs("SELECT a FROM t UNION SELECT b FROM u ORDER BY 1 LIMIT 5 OFFSET 2"),
        "(SELECT a FROM t UNION SELECT b FROM u) ORDER BY 1i32 ASC NULLS LAST LIMIT 5 OFFSET 2"
    );

    // For a single unparenthesized SELECT it attaches to the SelectStmt side (the binder's existing path).
    let (_, q) = parsed("SELECT a FROM t ORDER BY a LIMIT 5 OFFSET 1");
    assert!(q.order_by.is_empty());
    assert_eq!(q.limit, None);
    assert_eq!(q.offset, None);
    assert_eq!(plain(&q).order_by.len(), 1);
    assert_eq!(plain(&q).limit, Some(5));
    assert_eq!(plain(&q).offset, Some(1));

    // A parenthesized term carrying its own ORDER BY is not collapsed into the outer one.
    assert_eq!(
        qs("(SELECT a FROM t ORDER BY a LIMIT 1) LIMIT 9"),
        "SELECT a FROM t ORDER BY a ASC NULLS LAST LIMIT 1 LIMIT 9"
    );
    // A per-term LIMIT in a set operation is expressed with parentheses.
    assert_eq!(
        qs("(SELECT a FROM t LIMIT 1) UNION SELECT b FROM u"),
        "(SELECT a FROM t LIMIT 1 UNION SELECT b FROM u)"
    );
    // A parenthesized query with a CTE or an outer LIMIT is wrapped in a derived table to become a term.
    assert_eq!(
        qs("(SELECT 1 UNION SELECT 2 LIMIT 3) UNION SELECT 4"),
        "(SELECT * FROM ((SELECT 1i32 UNION SELECT 2i32) LIMIT 3) UNION SELECT 4i32)"
    );
}

#[test]
fn derived_table_with_set_operation() {
    assert_eq!(
        sel("SELECT * FROM (SELECT a FROM t UNION SELECT b FROM u) AS x"),
        "SELECT * FROM ((SELECT a FROM t UNION SELECT b FROM u)) AS x"
    );
    assert_eq!(
        sel("SELECT * FROM (WITH c AS (SELECT 1) SELECT * FROM c) AS x"),
        "SELECT * FROM (WITH c AS (SELECT 1i32) SELECT * FROM c) AS x"
    );
    assert_eq!(
        sel("SELECT * FROM (SELECT a FROM t ORDER BY a LIMIT 2) s JOIN u ON s.a = u.a"),
        "SELECT * FROM ((SELECT a FROM t ORDER BY a ASC NULLS LAST LIMIT 2) AS s \
             INNER JOIN u ON (s.a = u.a))"
    );
}

// --- Window functions ---------------------------------------------------

#[test]
fn window_functions() {
    assert_eq!(ex("row_number() OVER ()"), "row_number() OVER (WHOLE)");
    assert_eq!(ex("count(*) OVER ()"), "count(*) OVER (WHOLE)");
    assert_eq!(ex("sum(x) OVER (PARTITION BY a)"), "sum(x) OVER (PARTITION BY a WHOLE)");
    assert_eq!(
        ex("sum(x) OVER (PARTITION BY a, b + 1)"),
        "sum(x) OVER (PARTITION BY a, (b + 1i32) WHOLE)"
    );
    assert_eq!(
        ex("rank() OVER (PARTITION BY a ORDER BY b DESC)"),
        "rank() OVER (PARTITION BY a ORDER BY b DESC NULLS LAST RANGE)"
    );
    assert_eq!(
        ex("sum(x) OVER (ORDER BY b, c NULLS LAST)"),
        "sum(x) OVER (ORDER BY b ASC NULLS LAST, c ASC NULLS LAST RANGE)"
    );
    // They can be written wherever an ordinary function call can.
    assert_eq!(
        sel("SELECT sum(x) OVER (PARTITION BY a) AS s FROM t"),
        "SELECT sum(x) OVER (PARTITION BY a WHOLE) AS s FROM t"
    );

    // `star` and the default frame are checked directly on the AST.
    let (a, q) = parsed("SELECT count(*) OVER (), rank() OVER (ORDER BY b) FROM t");
    match a.get(plain(&q).items[0].expr) {
        Expr::Window { name, args, star, window_ref, partition_by, order_by, frame } => {
            assert_eq!(name, "count");
            assert!(args.is_empty());
            assert!(*star);
            assert!(window_ref.is_none());
            assert!(partition_by.is_empty());
            assert!(order_by.is_empty());
            assert_eq!(*frame, WindowFrame::WholePartition);
        }
        _ => panic!("not a window function"),
    }
    match a.get(plain(&q).items[1].expr) {
        Expr::Window { star, frame, order_by, .. } => {
            assert!(!*star);
            assert_eq!(order_by.len(), 1);
            assert_eq!(*frame, WindowFrame::RangeUnboundedPreceding);
        }
        _ => panic!("not a window function"),
    }
}

#[test]
fn window_keywords_are_contextual() {
    // Column names come from data files and are not chosen by the user. Reserving common
    // words for window syntax would create columns unreadable without quotes.
    for name in ["rows", "range", "over", "partition"] {
        assert_eq!(sel(&format!("SELECT {} FROM t", name)), format!("SELECT {} FROM t", name));
        assert_eq!(sel(&format!("SELECT t.{} FROM t", name)), format!("SELECT t.{} FROM t", name));
        assert_eq!(
            sel(&format!("SELECT * FROM t WHERE {} > 1", name)),
            format!("SELECT * FROM t WHERE ({} > 1i32)", name)
        );
        // Usable as both a table alias and a column alias.
        assert_eq!(
            sel(&format!("SELECT * FROM t AS {}", name)),
            format!("SELECT * FROM t AS {}", name)
        );
        assert_eq!(
            sel(&format!("SELECT * FROM t {}", name)),
            format!("SELECT * FROM t AS {}", name)
        );
        assert_eq!(
            sel(&format!("SELECT a AS {} FROM t", name)),
            format!("SELECT a AS {} FROM t", name)
        );
        // The same in uppercase (identifier spelling is preserved).
        let upper = name.to_ascii_uppercase();
        assert_eq!(sel(&format!("SELECT {} FROM t", upper)), format!("SELECT {} FROM t", upper));
        // Readable as a column inside GROUP BY / ORDER BY too.
        assert_eq!(
            sel(&format!("SELECT {0} FROM t GROUP BY {0} ORDER BY {0}", name)),
            format!("SELECT {0} FROM t GROUP BY {0} ORDER BY {0} ASC NULLS LAST", name)
        );
    }

    // Without a `(` right after `over`, it is an alias rather than a window clause.
    assert_eq!(sel("SELECT count(*) over FROM t"), "SELECT count(*) AS over FROM t");
    assert_eq!(sel("SELECT count(*) over"), "SELECT count(*) AS over");
    assert_eq!(sel("SELECT count(*) AS over FROM t"), "SELECT count(*) AS over FROM t");
    // A `(` right after an `over` that is not a function call stays an alias plus a syntax error.
    assert_eq!(code("SELECT a over (b)"), Code::UnexpectedToken as u16);

    // Even as a context-dependent keyword, window specifications still work as before.
    assert_eq!(ex("count(*) OVER ()"), "count(*) OVER (WHOLE)");
    assert_eq!(
        ex("sum(rows) OVER (PARTITION BY partition ORDER BY range)"),
        "sum(rows) OVER (PARTITION BY partition ORDER BY range ASC NULLS LAST RANGE)"
    );
    assert_eq!(
        sel("SELECT rank() OVER (PARTITION BY over) FROM t"),
        "SELECT rank() OVER (PARTITION BY over WHOLE) FROM t"
    );
    // Rejecting frame specifications is decided by spelling, not by reservation.
    assert_eq!(
        code("SELECT sum(x) OVER (ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)"),
        Code::UnsupportedFeature as u16
    );
    assert_eq!(
        code("SELECT sum(x) OVER (rows BETWEEN 1 PRECEDING AND CURRENT ROW)"),
        Code::UnsupportedFeature as u16
    );
    // Quoted identifiers are not matched as context-dependent keywords.
    assert_eq!(sel("SELECT \"rows\" FROM t"), "SELECT rows FROM t");
    assert_eq!(code("SELECT sum(x) OVER (\"rows\")"), Code::UnexpectedToken as u16);
}

#[test]
fn window_rejections() {
    // Silently ignoring an explicit frame would change results, so it is always rejected.
    assert_eq!(
        code("SELECT sum(x) OVER (ORDER BY b ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)"),
        Code::UnsupportedFeature as u16
    );
    assert_eq!(
        code("SELECT sum(x) OVER (RANGE UNBOUNDED PRECEDING)"),
        Code::UnsupportedFeature as u16
    );
    assert_eq!(
        code("SELECT sum(x) OVER (PARTITION BY a ROWS UNBOUNDED PRECEDING)"),
        Code::UnsupportedFeature as u16
    );
    // A window aggregate with DISTINCT is out of scope too.
    assert_eq!(code("SELECT count(DISTINCT x) OVER ()"), Code::UnsupportedFeature as u16);
    // `OVER w` (a named reference) parses fine. Whether the definition exists is only known
    // at bind time (see the tests on the `plan::bind` side).
    assert_eq!(code("SELECT sum(x) OVER w"), 0);
    assert_eq!(code("SELECT sum(x) OVER (PARTITION a)"), Code::UnexpectedToken as u16);
}

#[test]
fn named_window_ref_parses() {
    // `OVER w` (a single identifier) is kept in the tree as a named window reference.
    // The definition (PARTITION BY/ORDER BY) lives in the `WINDOW` clause; this confirms the
    // parser only carries the name here.
    assert_eq!(ex("sum(x) OVER w"), "sum(x) OVER w");
    let (a, q) = parsed("SELECT sum(x) OVER w FROM t");
    match a.get(plain(&q).items[0].expr) {
        Expr::Window { name, window_ref, partition_by, order_by, .. } => {
            assert_eq!(name, "sum");
            assert_eq!(window_ref.as_deref(), Some("w"));
            assert!(partition_by.is_empty());
            assert!(order_by.is_empty());
        }
        _ => panic!("not a window function"),
    }
    // With neither `(` nor an identifier right after `OVER`, it is an alias as before.
    assert_eq!(sel("SELECT count(*) over FROM t"), "SELECT count(*) AS over FROM t");
    // But when an identifier does follow `OVER`, it can only be read as a named window
    // reference unless a comma intervenes (an alias plus another select item would require a
    // comma, so that reading does not even hold).
    assert_eq!(sel("SELECT sum(x) over w, y FROM t"), "SELECT sum(x) OVER w, y FROM t");
}

#[test]
fn window_clause_named_definitions() {
    // A simple named window. Several functions can share the same definition.
    assert_eq!(
            sel(
                "SELECT id, sum(x) OVER w, avg(x) OVER w FROM t WINDOW w AS (PARTITION BY id ORDER BY ts)"
            ),
            "SELECT id, sum(x) OVER w, avg(x) OVER w FROM t WINDOW w AS (PARTITION BY id ORDER BY ts ASC NULLS LAST)"
        );
    // Several definitions can be given, comma-separated.
    assert_eq!(
            sel(
                "SELECT sum(x) OVER w1, rank() OVER w2 FROM t WINDOW w1 AS (PARTITION BY id), w2 AS (ORDER BY x)"
            ),
            "SELECT sum(x) OVER w1, rank() OVER w2 FROM t WINDOW w1 AS (PARTITION BY id), w2 AS (ORDER BY x ASC NULLS LAST)"
        );
    // It can be combined with an inline `OVER (...)`.
    assert_eq!(
        sel("SELECT sum(x) OVER w, count(*) OVER () FROM t WINDOW w AS (PARTITION BY id)"),
        "SELECT sum(x) OVER w, count(*) OVER (WHOLE) FROM t WINDOW w AS (PARTITION BY id)"
    );
    // `WINDOW` goes after `GROUP BY`/`HAVING` and before `QUALIFY`/`ORDER BY`.
    assert_eq!(
            sel(
                "SELECT a, sum(x) OVER w FROM t WHERE a > 0 GROUP BY a, x HAVING x > 0 WINDOW w AS (ORDER BY a) QUALIFY sum(x) OVER w > 0 ORDER BY a"
            ),
            "SELECT a, sum(x) OVER w FROM t WHERE (a > 0i32) GROUP BY a, x HAVING (x > 0i32) WINDOW w AS (ORDER BY a ASC NULLS LAST) QUALIFY (sum(x) OVER w > 0i32) ORDER BY a ASC NULLS LAST"
        );
    // Defining the same name twice is an error (`duckdb` also rejects it as "already defined").
    assert_eq!(
        code("SELECT a FROM t WINDOW w AS (ORDER BY a), w AS (ORDER BY a)"),
        Code::SyntaxError as u16
    );
    // `WINDOW` is fully reserved (the same judgment as `QUALIFY`), so it is unusable as a column name.
    assert_eq!(code("SELECT window FROM t"), Code::UnexpectedToken as u16);
    // With quotes it remains usable as a column name.
    assert_eq!(sel("SELECT \"window\" FROM t"), "SELECT window FROM t");
}

// --- Subquery expressions -----------------------------------------------

#[test]
fn scalar_subqueries() {
    assert_eq!(ex("(SELECT 1)"), "(SELECT 1i32)");
    assert_eq!(
        sel("SELECT (SELECT max(x) FROM u) AS m FROM t"),
        "SELECT (SELECT max(x) FROM u) AS m FROM t"
    );
    assert_eq!(
        sel("SELECT * FROM t WHERE a > (SELECT avg(x) FROM u)"),
        "SELECT * FROM t WHERE (a > (SELECT avg(x) FROM u))"
    );
    assert_eq!(ex("(SELECT 1) + 1"), "((SELECT 1i32) + 1i32)");
    // Extra parentheses mean the same thing. The inner `(` gets the check afresh.
    assert_eq!(ex("((SELECT 1))"), "(SELECT 1i32)");
    // Parenthesized expressions behave as before.
    assert_eq!(ex("(1 + 2)"), "(1i32 + 2i32)");
    assert_eq!(ex("(SELECT 1 UNION SELECT 2)"), "((SELECT 1i32 UNION SELECT 2i32))");
    assert_eq!(
        ex("(WITH a AS (SELECT 1) SELECT * FROM a)"),
        "(WITH a AS (SELECT 1i32) SELECT * FROM a)"
    );
}

#[test]
fn exists_and_in_subquery() {
    assert_eq!(ex("EXISTS (SELECT 1)"), "EXISTS (SELECT 1i32)");
    assert_eq!(ex("NOT EXISTS (SELECT 1)"), "NOT EXISTS (SELECT 1i32)");
    assert_eq!(
        sel("SELECT * FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.a = t.a)"),
        "SELECT * FROM t WHERE EXISTS (SELECT 1i32 FROM u WHERE (u.a = t.a))"
    );
    assert_eq!(
        ex("a = 1 AND NOT EXISTS (SELECT 1 FROM u)"),
        "((a = 1i32) AND NOT EXISTS (SELECT 1i32 FROM u))"
    );
    assert_eq!(ex("x IN (SELECT a FROM t)"), "(x IN (SELECT a FROM t))");
    assert_eq!(ex("x NOT IN (SELECT a FROM t)"), "(x NOT IN (SELECT a FROM t))");
    assert_eq!(
        ex("x IN (SELECT a FROM t UNION SELECT b FROM u)"),
        "(x IN ((SELECT a FROM t UNION SELECT b FROM u)))"
    );
    // A value list stays a value list as before.
    assert_eq!(ex("x IN (1, 2, 3)"), "(x IN [1i32, 2i32, 3i32])");
    assert_eq!(ex("x IN ((1), (2))"), "(x IN [1i32, 2i32])");
    // `IN ((SELECT ...))` falls to the value-list side, with a scalar subquery as its element.
    assert_eq!(ex("x IN ((SELECT 1))"), "(x IN [(SELECT 1i32)])");
    assert_eq!(code("SELECT x IN ()"), Code::UnexpectedToken as u16);
}

#[test]
fn quantified_comparison_subquery() {
    // `ANY`/`SOME` parse to the same meaning (`all: false`).
    assert_eq!(ex("x = ANY (SELECT a FROM t)"), "(x = ANY (SELECT a FROM t))");
    assert_eq!(ex("x = SOME (SELECT a FROM t)"), "(x = ANY (SELECT a FROM t))");
    assert_eq!(ex("x <> ALL (SELECT a FROM t)"), "(x != ALL (SELECT a FROM t))");
    assert_eq!(ex("x > ANY (SELECT a FROM t)"), "(x > ANY (SELECT a FROM t))");
    assert_eq!(ex("x >= ALL (SELECT a FROM t)"), "(x >= ALL (SELECT a FROM t))");
    assert_eq!(ex("x < SOME (SELECT a FROM t)"), "(x < ANY (SELECT a FROM t))");
    assert_eq!(ex("x <= ALL (SELECT a FROM t)"), "(x <= ALL (SELECT a FROM t))");
    // The same binding power as a predicate (tighter than AND).
    assert_eq!(
        ex("a = 1 AND x > ANY (SELECT a FROM t)"),
        "((a = 1i32) AND (x > ANY (SELECT a FROM t)))"
    );
    // `any`/`some` are not reserved, so without a following `(` they are ordinary column names.
    assert_eq!(ex("x > any"), "(x > any)");
    assert_eq!(ex("x > some"), "(x > some)");
    // `ALL` stays an existing reserved word and does not collide with UNION ALL and friends.
    assert_eq!(ex("(SELECT 1 UNION ALL SELECT 2)"), "((SELECT 1i32 UNION ALL SELECT 2i32))");
}

#[test]
fn other_statements() {
    let p = parse("EXPLAIN SELECT 1").expect("parse");
    assert!(matches!(p.stmt, Stmt::Explain(_)));
    let p = parse("DESCRIBE t").expect("parse");
    match &p.stmt {
        Stmt::Describe(f) => assert_eq!(from_str(&p.arena, f), "t"),
        _ => panic!("not describe"),
    }
    let p = parse("DESCRIBE parquet('x.parquet')").expect("parse");
    match &p.stmt {
        Stmt::Describe(f) => assert_eq!(from_str(&p.arena, f), "parquet('x.parquet')"),
        _ => panic!("not describe"),
    }
    assert!(matches!(parse("SHOW TABLES;").expect("parse").stmt, Stmt::ShowTables));
    assert_eq!(code("SHOW COLUMNS"), Code::UnexpectedToken as u16);
}

// --- Errors -------------------------------------------------------------

#[test]
fn errors() {
    assert_eq!(code("SELECT FROM t"), Code::UnexpectedToken as u16);
    assert_eq!(code("SELECT * FROM"), Code::UnexpectedToken as u16);
    assert_eq!(code("SELECT (1 + 2"), Code::UnexpectedToken as u16);
    assert_eq!(code("SELECT 1)"), Code::UnexpectedToken as u16);
    assert_eq!(code("SELECT * FROM t WHERE"), Code::UnexpectedToken as u16);
    assert_eq!(code("SELECT * FROM t LIMIT x"), Code::UnexpectedToken as u16);
    assert_eq!(code("SELECT a b c FROM t"), Code::UnexpectedToken as u16);
    assert_eq!(code(""), Code::UnsupportedFeature as u16);
    // With the `ddl`/`dml` features on, INSERT/UPDATE/CREATE TABLE are valid statements.
    // The behavior in the default build with the features off is checked here, and the
    // behavior with them on is checked by the `ddl_dml` tests at the end of `sql/parser.rs`.
    #[cfg(not(feature = "dml"))]
    {
        assert_eq!(code("INSERT INTO t VALUES (1)"), Code::UnsupportedFeature as u16);
        assert_eq!(code("UPDATE t SET a = 1"), Code::UnsupportedFeature as u16);
    }
    #[cfg(not(feature = "ddl"))]
    {
        assert_eq!(code("CREATE TABLE t (a INT)"), Code::UnsupportedFeature as u16);
        // ALTER TABLE also remains unsupported in builds without `ddl`. The behavior with it
        // on is checked by the DDL/DML test group at the end of `sql/parser.rs`.
        assert_eq!(code("ALTER TABLE t ADD COLUMN x INT"), Code::UnsupportedFeature as u16);
    }
    assert_eq!(code("WITH x AS SELECT 1 SELECT * FROM x"), Code::UnexpectedToken as u16);
    // Column lists remain unsupported on non-recursive CTEs (allowed only under
    // `WITH RECURSIVE`; see the `recursive_cte` test group).
    assert_eq!(code("WITH x (a) AS (SELECT 1) SELECT 1"), Code::UnsupportedFeature as u16);
    assert_eq!(code("SELECT 1 UNION"), Code::UnexpectedToken as u16);
    assert_eq!(code("SELECT 1 INTERSECT 2"), Code::UnexpectedToken as u16);
    // `a & b` used to be a syntax error before `&` became the bitwise-AND
    // operator (see `bitwise_operators_desugar_to_bit_functions` below).
    assert_eq!(code("SELECT a &"), Code::UnexpectedToken as u16);
}

#[test]
fn error_positions() {
    // The position always points at the first byte of the offending token.
    assert_eq!(err_at("SELECT FROM t"), (Code::UnexpectedToken as u16, 7));
    assert_eq!(err_at("SELECT 'abc"), (Code::UnterminatedString as u16, 7));
    #[cfg(not(feature = "dml"))]
    assert_eq!(err_at("INSERT INTO t VALUES (1)"), (Code::UnsupportedFeature as u16, 0));
    #[cfg(not(feature = "ddl"))]
    assert_eq!(err_at("ALTER TABLE t ADD COLUMN x INT"), (Code::UnsupportedFeature as u16, 0));
    assert_eq!(err_at("SELECT a FROM t WHERE b @ 1"), (Code::UnexpectedToken as u16, 24));
    assert_eq!(err_at("SELECT CAST(x AS FROB)"), (Code::InvalidCast as u16, 17));
    // The position points at the start of the offending token for new syntax too.
    assert_eq!(err_at("SELECT 1 UNION 2"), (Code::UnexpectedToken as u16, 15));
    assert_eq!(err_at("SELECT sum(x) OVER (ROWS)"), (Code::UnsupportedFeature as u16, 20));
    // The position of the `(` a column list requires.
    assert_eq!(err_at("WITH x (a) AS (SELECT 1) SELECT 1"), (Code::UnsupportedFeature as u16, 7));
    assert_eq!(err_at("SELECT 1 WHERE EXISTS SELECT 1"), (Code::UnexpectedToken as u16, 22));
}

#[test]
fn deep_nesting_is_rejected_without_overflow() {
    let mut sql = String::from("SELECT ");
    for _ in 0..100_000 {
        sql.push('(');
    }
    sql.push('1');
    assert_eq!(code(&sql), Code::ExpressionTooDeep as u16);

    // Deep subqueries and comma joins stop at the same limit.
    let mut q = String::new();
    for _ in 0..200 {
        q.push_str("SELECT * FROM (");
    }
    q.push_str("SELECT 1");
    for _ in 0..200 {
        q.push(')');
    }
    assert_eq!(code(&q), Code::ExpressionTooDeep as u16);

    let mut j = String::from("SELECT * FROM a");
    for _ in 0..200 {
        j.push_str(", a");
    }
    assert_eq!(code(&j), Code::ExpressionTooDeep as u16);

    // Just below the limit passes.
    let ok = format!("SELECT {}1{}", "(".repeat(50), ")".repeat(50));
    assert_eq!(code(&ok), 0);
}

#[test]
fn deep_subquery_expressions_are_rejected() {
    // Nested scalar subqueries. Dropping the tree recurses too, so the limit must be at a
    // level where "the stack survives dropping a tree that parsed".
    let n = 200;
    let mut s = String::from("SELECT ");
    s.push_str(&"(SELECT ".repeat(n));
    s.push('1');
    s.push_str(&")".repeat(n));
    assert_eq!(code(&s), Code::ExpressionTooDeep as u16);

    // Nested EXISTS.
    let mut e = String::from("SELECT 1 WHERE ");
    e.push_str(&"EXISTS (SELECT 1 WHERE ".repeat(n));
    e.push_str("TRUE");
    e.push_str(&")".repeat(n));
    assert_eq!(code(&e), Code::ExpressionTooDeep as u16);

    // Nested IN (SELECT ...).
    let mut i = String::from("SELECT 1 WHERE ");
    i.push_str(&"1 IN (SELECT 1 WHERE ".repeat(n));
    i.push_str("TRUE");
    i.push_str(&")".repeat(n));
    assert_eq!(code(&i), Code::ExpressionTooDeep as u16);

    // Nested derived tables (via parenthesized queries).
    let mut d = String::from("SELECT * FROM ");
    d.push_str(&"(SELECT * FROM ".repeat(n));
    d.push('t');
    d.push_str(&")".repeat(n));
    assert_eq!(code(&d), Code::ExpressionTooDeep as u16);

    // Nested CTEs.
    let mut c = String::new();
    c.push_str(&"WITH a AS (".repeat(n));
    c.push_str("SELECT 1");
    for _ in 0..n {
        c.push_str(") SELECT * FROM a");
    }
    assert_eq!(code(&c), Code::ExpressionTooDeep as u16);

    // A scalar subquery just below the limit passes (each level consumes two of depth).
    let k = 30;
    let mut ok = String::from("SELECT ");
    ok.push_str(&"(SELECT ".repeat(k));
    ok.push('1');
    ok.push_str(&")".repeat(k));
    assert_eq!(code(&ok), 0);
}

#[test]
fn long_setop_chains_are_rejected() {
    // Set operations are a left-deep `Box` chain too. The limit stops the recursion on drop.
    let mut u = String::from("SELECT 1");
    u.push_str(&" UNION SELECT 1".repeat(200));
    assert_eq!(code(&u), Code::ExpressionTooDeep as u16);

    let mut x = String::from("SELECT 1");
    x.push_str(&" INTERSECT SELECT 1".repeat(200));
    assert_eq!(code(&x), Code::ExpressionTooDeep as u16);

    // Just below the limit passes.
    let mut ok = String::from("SELECT 1");
    ok.push_str(&" UNION SELECT 1".repeat(60));
    assert_eq!(code(&ok), 0);
}

// --- DDL/DML (the `ddl`/`dml` features) ----------------------------------

#[cfg(feature = "ddl")]
#[test]
fn create_table_variants_parse() {
    let p = parse("CREATE TABLE t (id INTEGER, name VARCHAR NOT NULL)").expect("parse");
    match p.stmt {
        Stmt::CreateTable { name, or_replace, if_not_exists, columns, as_select } => {
            assert_eq!(name, "t");
            assert!(!or_replace);
            assert!(!if_not_exists);
            assert!(as_select.is_none());
            assert_eq!(columns.len(), 2);
            assert!(columns[0].nullable);
            assert!(!columns[1].nullable);
        }
        _ => panic!("not CreateTable"),
    }

    let p = parse("CREATE TABLE IF NOT EXISTS t (id INTEGER)").expect("parse");
    match p.stmt {
        Stmt::CreateTable { if_not_exists, .. } => assert!(if_not_exists),
        _ => panic!("not CreateTable"),
    }

    let p = parse("CREATE OR REPLACE TABLE t AS SELECT * FROM u").expect("parse");
    match p.stmt {
        Stmt::CreateTable { or_replace, as_select, columns, .. } => {
            assert!(or_replace);
            assert!(as_select.is_some());
            assert!(columns.is_empty());
        }
        _ => panic!("not CreateTable"),
    }
}

#[cfg(feature = "ddl")]
#[test]
fn create_view_captures_body_as_raw_sql() {
    let p = parse("CREATE OR REPLACE VIEW v AS SELECT a, b FROM t WHERE a > 1").expect("parse");
    match p.stmt {
        Stmt::CreateView { name, or_replace, query_sql } => {
            assert_eq!(name, "v");
            assert!(or_replace);
            assert_eq!(query_sql, "SELECT a, b FROM t WHERE a > 1");
        }
        _ => panic!("not CreateView"),
    }
}

#[cfg(feature = "ddl")]
#[test]
fn create_view_rejects_placeholders() {
    assert_eq!(
        code("CREATE VIEW v AS SELECT * FROM t WHERE id = ?"),
        Code::UnsupportedFeature as u16
    );
}

#[cfg(feature = "ddl")]
#[test]
fn drop_table_and_view_parse() {
    let p = parse("DROP TABLE IF EXISTS t").expect("parse");
    match p.stmt {
        Stmt::DropTable { name, if_exists } => {
            assert_eq!(name, "t");
            assert!(if_exists);
        }
        _ => panic!("not DropTable"),
    }
    let p = parse("DROP VIEW v").expect("parse");
    match p.stmt {
        Stmt::DropView { name, if_exists } => {
            assert_eq!(name, "v");
            assert!(!if_exists);
        }
        _ => panic!("not DropView"),
    }
}

#[cfg(feature = "ddl")]
#[test]
fn alter_table_add_column_variants_parse() {
    let p = parse("ALTER TABLE t ADD COLUMN x INT").expect("parse");
    match p.stmt {
        Stmt::AlterTable { name, action } => {
            assert_eq!(name, "t");
            match action {
                AlterTableAction::AddColumn { name, ty, nullable, default } => {
                    assert_eq!(name, "x");
                    assert_eq!(ty, Ty::Int);
                    assert!(nullable);
                    assert!(default.is_none());
                }
                _ => panic!("not AddColumn"),
            }
        }
        _ => panic!("not AlterTable"),
    }

    // `COLUMN` may be omitted (the same as DuckDB, confirmed with the CLI).
    let p = parse("ALTER TABLE t ADD y INT NOT NULL DEFAULT 5").expect("parse");
    match p.stmt {
        Stmt::AlterTable { action, .. } => match action {
            AlterTableAction::AddColumn { name, nullable, default, .. } => {
                assert_eq!(name, "y");
                assert!(!nullable);
                assert!(default.is_some());
            }
            _ => panic!("not AddColumn"),
        },
        _ => panic!("not AlterTable"),
    }
}

#[cfg(feature = "ddl")]
#[test]
fn alter_table_drop_column_parse() {
    let p = parse("ALTER TABLE t DROP COLUMN x").expect("parse");
    match p.stmt {
        Stmt::AlterTable { name, action } => {
            assert_eq!(name, "t");
            match action {
                AlterTableAction::DropColumn { name } => assert_eq!(name, "x"),
                _ => panic!("not DropColumn"),
            }
        }
        _ => panic!("not AlterTable"),
    }
    // `COLUMN` may be omitted.
    let p = parse("ALTER TABLE t DROP x").expect("parse");
    assert!(matches!(p.stmt, Stmt::AlterTable { action: AlterTableAction::DropColumn { .. }, .. }));
}

#[cfg(feature = "ddl")]
#[test]
fn alter_table_rename_variants_parse() {
    let p = parse("ALTER TABLE t RENAME COLUMN a TO b").expect("parse");
    match p.stmt {
        Stmt::AlterTable { action, .. } => match action {
            AlterTableAction::RenameColumn { old, new } => {
                assert_eq!(old, "a");
                assert_eq!(new, "b");
            }
            _ => panic!("not RenameColumn"),
        },
        _ => panic!("not AlterTable"),
    }

    // `COLUMN` may be omitted.
    let p = parse("ALTER TABLE t RENAME a TO b").expect("parse");
    assert!(matches!(
        p.stmt,
        Stmt::AlterTable { action: AlterTableAction::RenameColumn { .. }, .. }
    ));

    let p = parse("ALTER TABLE t RENAME TO u").expect("parse");
    match p.stmt {
        Stmt::AlterTable { name, action } => {
            assert_eq!(name, "t");
            match action {
                AlterTableAction::RenameTable { new_name } => assert_eq!(new_name, "u"),
                _ => panic!("not RenameTable"),
            }
        }
        _ => panic!("not AlterTable"),
    }
}

#[cfg(feature = "dml")]
#[test]
fn insert_variants_parse() {
    let p = parse("INSERT INTO t VALUES (1, 'a'), (2, 'b')").expect("parse");
    match p.stmt {
        Stmt::Insert { table, columns, source } => {
            assert_eq!(table, "t");
            assert!(columns.is_empty());
            match source {
                InsertSource::Values(rows) => assert_eq!(rows.len(), 2),
                _ => panic!("not Values"),
            }
        }
        _ => panic!("not Insert"),
    }

    let p = parse("INSERT INTO t (id, name) SELECT id, name FROM u").expect("parse");
    match p.stmt {
        Stmt::Insert { columns, source, .. } => {
            assert_eq!(columns, vec!["id".to_string(), "name".to_string()]);
            assert!(matches!(source, InsertSource::Query(_)));
        }
        _ => panic!("not Insert"),
    }
}

#[cfg(feature = "dml")]
#[test]
fn update_and_delete_parse() {
    let p = parse("UPDATE t SET a = 1, b = a + 1 WHERE id = 5").expect("parse");
    match p.stmt {
        Stmt::Update { table, assignments, filter } => {
            assert_eq!(table, "t");
            assert_eq!(assignments.len(), 2);
            assert!(filter.is_some());
        }
        _ => panic!("not Update"),
    }

    let p = parse("DELETE FROM t").expect("parse");
    match p.stmt {
        Stmt::Delete { table, filter } => {
            assert_eq!(table, "t");
            assert!(filter.is_none());
        }
        _ => panic!("not Delete"),
    }
    let p = parse("DELETE FROM t WHERE id > 1").expect("parse");
    assert!(matches!(p.stmt, Stmt::Delete { filter: Some(_), .. }));
}

#[cfg(feature = "export")]
#[test]
fn copy_subquery_parses_path_and_optional_format() {
    let p = parse("COPY (SELECT a, b FROM t) TO 'out.csv'").expect("parse");
    match p.stmt {
        Stmt::Copy { query, path, format } => {
            assert_eq!(path, "out.csv");
            assert!(format.is_none());
            assert!(query.as_simple_select().is_some());
        }
        _ => panic!("not Copy"),
    }

    let p = parse("COPY (SELECT a FROM t) TO 'out.txt' (FORMAT csv)").expect("parse");
    match p.stmt {
        Stmt::Copy { path, format, .. } => {
            assert_eq!(path, "out.txt");
            assert_eq!(format.as_deref(), Some("csv"));
        }
        _ => panic!("not Copy"),
    }

    // The value of FORMAT is case-insensitive.
    let p = parse("COPY (SELECT a FROM t) TO 'out.bin' (FORMAT JSON)").expect("parse");
    assert!(matches!(p.stmt, Stmt::Copy { format: Some(f), .. } if f.eq_ignore_ascii_case("json")));
}

/// `COPY <table> TO ...` yields a tree equivalent to `SELECT * FROM <table>`.
#[cfg(feature = "export")]
#[test]
fn copy_table_form_desugars_to_select_star() {
    let p = parse("COPY t TO 'out.csv'").expect("parse");
    match p.stmt {
        Stmt::Copy { query, path, .. } => {
            assert_eq!(path, "out.csv");
            let sel = query.as_simple_select().expect("simple select");
            assert_eq!(sel.items.len(), 1);
            assert!(matches!(p.arena.get(sel.items[0].expr), Expr::Star { qualifier: None, .. }));
            match &sel.from {
                Some(FromItem::Table { name, alias }) => {
                    assert_eq!(name, "t");
                    assert!(alias.is_none());
                }
                _ => panic!("not Table"),
            }
        }
        _ => panic!("not Copy"),
    }
}

/// `format` is usable as an ordinary column name outside `COPY` (`export` does not
/// reserve the word globally -- it is a context-dependent keyword).
#[cfg(feature = "export")]
#[test]
fn format_remains_usable_as_a_column_name_outside_copy() {
    assert_eq!(sel("SELECT format FROM t"), "SELECT format FROM t");
}

/// `to` is likewise usable as an ordinary column name with `export` alone. With `ddl`
/// also on, it is separately reserved globally for `ALTER TABLE ... RENAME TO`
/// (`DDL_KEYWORDS` in `sql/lexer.rs`), so this check is meaningful only in configurations
/// without `ddl` (see the docs on `copy_stmt`/`expect_to`).
#[cfg(all(feature = "export", not(feature = "ddl")))]
#[test]
fn to_remains_usable_as_a_column_name_when_ddl_is_disabled() {
    assert_eq!(sel("SELECT to FROM t"), "SELECT to FROM t");
}

#[cfg(feature = "export")]
#[test]
fn copy_requires_to_and_a_string_path() {
    assert_eq!(code("COPY (SELECT 1) TO"), Code::UnexpectedToken as u16);
    assert_eq!(code("COPY (SELECT 1)"), Code::UnexpectedToken as u16);
    assert_eq!(code("COPY (SELECT 1) TO out.csv"), Code::UnexpectedToken as u16);
}

// --- File table functions / bare path literals (`FromItem::File`) --------
//
// Actual resolution (the exact-path name lookup via `catalog.index_of`) is
// `plan::bind::flatten_from`'s job, so this only checks the shape of the parsed AST
// (`FromItem::File`'s `path`/`format`/`alias`) and the round-trip string.
// The behavior was confirmed with the `duckdb` CLI (see the doc comments).

fn from_item(sql: &str) -> (ExprArena, FromItem) {
    let p = parse(sql).expect("parse failed");
    match p.stmt {
        Stmt::Select(q) => match q.body {
            SetExpr::Select(s) => (p.arena, s.from.expect("no FROM")),
            _ => panic!("not a plain SELECT"),
        },
        _ => panic!("not a SELECT"),
    }
}

#[test]
fn bare_string_literal_from_infers_format_from_extension() {
    let (a, f) = from_item("SELECT * FROM 'data.parquet'");
    assert!(matches!(&f, FromItem::File { path, format, alias }
            if path == "data.parquet" && *format == FormatKind::Parquet && alias.is_none()));
    assert_eq!(from_str(&a, &f), "parquet('data.parquet')");

    // An unknown extension is treated as Parquet, per the existing `FormatKind::detect` default.
    let (_, f) = from_item("SELECT * FROM 'data.bin'");
    assert!(matches!(&f, FromItem::File { format, .. } if *format == FormatKind::Parquet));

    #[cfg(feature = "csv")]
    {
        let (_, f) = from_item("SELECT * FROM 'data.csv'");
        assert!(matches!(&f, FromItem::File { format, .. } if *format == FormatKind::Csv));
        let (_, f) = from_item("SELECT * FROM 'data.tsv'");
        assert!(matches!(&f, FromItem::File { format, .. } if *format == FormatKind::Tsv));
    }
    #[cfg(feature = "jsonl")]
    {
        let (_, f) = from_item("SELECT * FROM 'data.jsonl'");
        assert!(matches!(&f, FromItem::File { format, .. } if *format == FormatKind::Jsonl));
        let (_, f) = from_item("SELECT * FROM 'data.json'");
        assert!(matches!(&f, FromItem::File { format, .. } if *format == FormatKind::Json));
    }
}

#[test]
fn bare_string_literal_from_accepts_alias() {
    let (_, f) = from_item("SELECT * FROM 'data.parquet' AS p");
    assert!(matches!(&f, FromItem::File { alias: Some(a), .. } if a == "p"));
    // A bare alias without `AS` is allowed as usual (`opt_alias` is shared).
    let (_, f) = from_item("SELECT * FROM 'data.parquet' p");
    assert!(matches!(&f, FromItem::File { alias: Some(a), .. } if a == "p"));
}

#[test]
fn read_parquet_is_an_alias_for_parquet() {
    let (a, f) = from_item("SELECT * FROM read_parquet('data.parquet')");
    assert!(matches!(&f, FromItem::File { path, format, .. }
            if path == "data.parquet" && *format == FormatKind::Parquet));
    // Both land in the same shape, so the round-trip rendering is normalized to
    // `parquet(...)` (see the `from_str` docs).
    assert_eq!(from_str(&a, &f), "parquet('data.parquet')");
}

#[cfg(feature = "csv")]
#[test]
fn read_csv_and_read_csv_auto_parse_the_same_way() {
    for func in ["read_csv", "read_csv_auto"] {
        let sql = format!("SELECT * FROM {func}('a.csv') AS x");
        let (a, f) = from_item(&sql);
        assert!(matches!(&f, FromItem::File { path, format, alias: Some(al) }
                if path == "a.csv" && *format == FormatKind::Csv && al == "x"));
        assert_eq!(from_str(&a, &f), "read_csv('a.csv') AS x");
    }
}

#[cfg(feature = "jsonl")]
#[test]
fn read_json_and_read_json_auto_parse_the_same_way() {
    for func in ["read_json", "read_json_auto"] {
        let sql = format!("SELECT * FROM {func}('a.json') AS x");
        let (a, f) = from_item(&sql);
        assert!(matches!(&f, FromItem::File { path, format, alias: Some(al) }
                if path == "a.json" && *format == FormatKind::Json && al == "x"));
        assert_eq!(from_str(&a, &f), "read_json('a.json') AS x");
    }
}

/// In builds with the `csv` feature off, the spellings `read_csv`/`read_csv_auto`
/// themselves are not recognized (they fall through at the syntax level and become
/// `UnsupportedFeature` -- the same pattern as `ddl`/`dml`/`export` statements falling
/// down the same path when their features are off; see the `base_rel` doc comment).
/// Parquet is always available and is unaffected.
#[cfg(not(feature = "csv"))]
#[test]
fn read_csv_is_unsupported_without_the_csv_feature() {
    assert_eq!(code("SELECT * FROM read_csv('a.csv')"), Code::UnsupportedFeature as u16);
    assert_eq!(code("SELECT * FROM read_csv_auto('a.csv')"), Code::UnsupportedFeature as u16);
    // A bare literal relying only on extension detection still parses (actual resolution is
    // left to the catalog lookup on the `plan::bind` side).
    assert!(parse("SELECT * FROM 'a.csv'").is_ok());
    assert!(parse("SELECT * FROM parquet('a.parquet')").is_ok());
}

#[cfg(not(feature = "jsonl"))]
#[test]
fn read_json_is_unsupported_without_the_jsonl_feature() {
    assert_eq!(code("SELECT * FROM read_json('a.json')"), Code::UnsupportedFeature as u16);
    assert_eq!(code("SELECT * FROM read_json_auto('a.json')"), Code::UnsupportedFeature as u16);
}

/// Named option arguments and multi-file arguments are out of scope for v1
/// (see the `FromItem::File` doc comment). Anything other than one argument is a syntax error.
#[test]
fn file_table_functions_reject_more_than_one_positional_argument() {
    assert_eq!(
        code("SELECT * FROM parquet('a.parquet', 'b.parquet')"),
        Code::UnexpectedToken as u16
    );
    #[cfg(feature = "csv")]
    assert_eq!(code("SELECT * FROM read_csv('a.csv', delim=',')"), Code::UnexpectedToken as u16);
    assert_eq!(code("SELECT * FROM parquet()"), Code::UnexpectedToken as u16);
}

/// An unknown function name gives `UnsupportedFeature`, like any other table function.
#[test]
fn unknown_table_function_name_is_unsupported() {
    assert_eq!(
        code("SELECT * FROM read_parquet_auto('a.parquet')"),
        Code::UnsupportedFeature as u16
    );
    assert_eq!(code("SELECT * FROM nonsense_fn('a.parquet')"), Code::UnsupportedFeature as u16);
}

// =========================================================================
// New operators: `~~`/`!~~`/`~~*`/`!~~*`/`~~~` (LIKE/ILIKE/GLOB aliases),
// `IS [NOT] TRUE/FALSE`, `ISNULL`/`NOTNULL`, `//` (integer division), `@`
// (absolute value), `!` (factorial). Parser-level desugaring only —
// end-to-end evaluated results (NULL propagation, real table columns) are
// covered by `crates/ahiru-core/tests/new_operators.rs`. Every expected
// value below is cross-checked against a real `duckdb` CLI.
// =========================================================================

// --- `~~`/`!~~`/`~~*`/`!~~*` (LIKE/ILIKE aliases) -------------------------

#[test]
fn like_alias_operators_desugar_to_like() {
    assert_eq!(ex("a ~~ 'x%'"), "(a LIKE 'x%')");
    assert_eq!(ex("a !~~ 'x%'"), "(a NOT LIKE 'x%')");
    assert_eq!(ex("a ~~* 'x%'"), "(a ILIKE 'x%')");
    assert_eq!(ex("a !~~* 'x%'"), "(a NOT ILIKE 'x%')");
}

#[test]
fn like_alias_operators_reject_escape() {
    // Only the `LIKE`/`ILIKE` *keyword* forms accept `ESCAPE`; duckdb
    // itself rejects `ESCAPE` after `~~` as a parse error (verified:
    // `duckdb -c "select 'a%c' ~~ 'a$%c' escape '$'"` -> `Parser Error`).
    assert_eq!(code("SELECT 'a%c' ~~ 'a$%c' ESCAPE '$' FROM t"), Code::UnexpectedToken as u16);
}

#[test]
fn like_alias_operators_bind_tighter_than_concat_unlike_the_like_keyword() {
    // This is a real, verified difference from the `LIKE` keyword, not a
    // copy-paste slip:
    //   duckdb: 'ab' LIKE 'a' || 'b' -> true,     i.e. 'ab' LIKE ('a'||'b')
    //   duckdb: 'ab' ~~   'a' || 'b' -> 'falseb', i.e. ('ab' ~~ 'a') || 'b'
    assert_eq!(ex("'ab' LIKE 'a' || 'b'"), "('ab' LIKE ('a' || 'b'))");
    assert_eq!(ex("'ab' ~~ 'a' || 'b'"), "(('ab' LIKE 'a') || 'b')");
    // Same quirk for the ILIKE spelling.
    assert_eq!(ex("'AB' ~~* 'a' || 'b'"), "(('AB' ILIKE 'a') || 'b')");
}

// --- `~~~` (GLOB alias) ----------------------------------------------------

#[test]
fn glob_alias_operator_desugars_to_glob_call() {
    assert_eq!(ex("a ~~~ 'x*'"), "glob(a, 'x*')");
}

#[test]
fn glob_alias_operator_binds_tighter_than_concat_unlike_the_glob_keyword() {
    // duckdb: 'ab' GLOB 'a' || '*'  -> true,   i.e. 'ab' GLOB ('a'||'*')
    // duckdb: 'ab' ~~~  'a' || '*'  -> 'false*', i.e. ('ab' ~~~ 'a') || '*'
    assert_eq!(ex("'ab' GLOB 'a' || '*'"), "glob('ab', ('a' || '*'))");
    assert_eq!(ex("'ab' ~~~ 'a' || '*'"), "(glob('ab', 'a') || '*')");
}

#[test]
fn tilde_regex_operators_are_unaffected_by_the_like_alias_family() {
    // Same lexer bytes (`~`), different meaning depending on how many
    // repeat: these must still desugar exactly the way they always have.
    assert_eq!(ex("a ~ 'x.y'"), "regexp_full_match(a, 'x.y')");
    assert_eq!(ex("a !~ 'x.y'"), "(NOT regexp_full_match(a, 'x.y'))");
    assert_eq!(ex("~5"), "bit_not(5i32)");
    // `~ ~5`: prefix bitwise NOT applied twice; the space is now required
    // — `~~5` lexes as the `~~` (LIKE-alias) operator followed by `5`,
    // which has no left-hand operand to attach to, so it's a parse error
    // rather than two bitwise NOTs (duckdb agrees: `~ ~5` -> `5`).
    assert_eq!(ex("~ ~5"), "bit_not(bit_not(5i32))");
    assert_eq!(code("SELECT ~~5 FROM t"), Code::UnexpectedToken as u16);
}

#[test]
fn ne_still_lexes_as_a_single_token_not_bang_eq() {
    assert_eq!(ex("4 != 5"), "(4i32 != 5i32)");
}

// --- `IS [NOT] TRUE` / `IS [NOT] FALSE` -----------------------------------

#[test]
fn is_true_desugars_to_cast_and_coalesce() {
    assert_eq!(ex("x IS TRUE"), "coalesce(CAST(x AS BOOLEAN), false)");
    assert_eq!(ex("x IS NOT TRUE"), "(NOT coalesce(CAST(x AS BOOLEAN), false))");
}

#[test]
fn is_false_desugars_to_negated_cast_and_coalesce() {
    assert_eq!(ex("x IS FALSE"), "coalesce((NOT CAST(x AS BOOLEAN)), false)");
    assert_eq!(ex("x IS NOT FALSE"), "(NOT coalesce((NOT CAST(x AS BOOLEAN)), false))");
}

// --- `ISNULL` / `NOTNULL` postfix -----------------------------------------

#[test]
fn isnull_notnull_desugar_to_is_null() {
    assert_eq!(ex("x ISNULL"), "(x IS NULL)");
    assert_eq!(ex("x NOTNULL"), "(x IS NOT NULL)");
}

#[test]
fn isnull_notnull_are_soft_keywords_not_reserved() {
    // A column named `isnull`/`notnull`: a bare reference is consumed
    // whole by `primary_atom` as the *operand*, so it never reaches the
    // postfix check in `expr_body`'s loop.
    assert_eq!(code("SELECT isnull FROM t"), 0);
    assert_eq!(code("SELECT notnull FROM t"), 0);
    // `AS isnull` reads the alias through a separate `ident()` call in
    // `opt_alias`, after `expr()` has already returned.
    assert_eq!(code("SELECT 1 AS isnull FROM t"), 0);
}

#[test]
fn isnull_binds_at_comparison_strength() {
    // duckdb: SELECT 1 + 2 ISNULL -> false, i.e. (1+2) ISNULL, not
    // 1 + (2 ISNULL).
    assert_eq!(ex("1 + 2 ISNULL"), "((1i32 + 2i32) IS NULL)");
}

// --- `//` integer division -------------------------------------------------

#[test]
fn integer_division_operator_desugars_to_div() {
    assert_eq!(ex("a // b"), "(a / b)");
}

#[test]
fn integer_division_binds_like_star_and_slash() {
    // duckdb: 2 + 5 // 2 -> 4, i.e. 2 + (5 // 2)
    assert_eq!(ex("2 + 5 // 2"), "(2i32 + (5i32 / 2i32))");
    // duckdb: 5 // 2 // 2 -> 1, left-associative: (5 // 2) // 2
    assert_eq!(ex("5 // 2 // 2"), "((5i32 / 2i32) / 2i32)");
}

// --- `@` absolute value -----------------------------------------------------

#[test]
fn at_prefix_desugars_to_abs() {
    assert_eq!(ex("@x"), "abs(x)");
    assert_eq!(ex("@(-5)"), "abs(-5i32)");
}

// --- `!` postfix factorial --------------------------------------------------

#[test]
fn factorial_postfix_desugars_to_factorial_call() {
    assert_eq!(ex("4!"), "factorial(4i32)");
    assert_eq!(ex("x!"), "factorial(x)");
    assert_eq!(ex("(2 + 2)!"), "factorial((2i32 + 2i32))");
}

#[test]
fn factorial_applies_after_unary_minus_for_literals_and_columns_alike() {
    // duckdb: SELECT -4! -> 1, and (confirmed independently against the
    // `duckdb` CLI) SELECT -x! FROM t (x=4) -> 1 too -- `!` binds looser
    // than prefix `-` (`BP_BANG`'s doc in `sql::parser` has the full
    // rationale), so both parse as `(-x)!`, never `-(x!)`.
    //
    // This closes a real bug an earlier version of this parser had: the
    // negative-*literal* fast path in `prefix()`'s `Tok::Minus` arm folded
    // `-4` into one literal before `!` got a chance to apply, so `-4!`
    // happened to come out right (`factorial(-4)`) -- but the *general*
    // `-x` path (any non-literal operand, reached via `expr_bp(BP_UNARY)`)
    // built `Unary::Neg(factorial(x))` instead, i.e. `-(x!)`. Same syntax,
    // two different parses depending only on whether the operand was a
    // literal. `BP_BANG` sitting below `BP_UNARY` (which is what `prefix`
    // always reads its operand at) fixes both paths at once, uniformly.
    assert_eq!(ex("-4!"), "factorial(-4i32)");
    assert_eq!(ex("-x!"), "factorial((- x))");
}

#[test]
fn factorial_and_cast_shorthand_interleave() {
    // duckdb: SELECT 4!::VARCHAR -> '24', i.e. CAST(4! AS VARCHAR). `!` no
    // longer joins `primary`'s postfix loop (see `BP_BANG`'s doc), so this
    // `::` is picked up by an explicit `cast_postfix` call made right
    // after `expr_body` folds the `!` -- without it, a `::` immediately
    // after `!` would be silently dropped instead of erroring or applying.
    assert_eq!(ex("4!::VARCHAR"), "CAST(factorial(4i32) AS VARCHAR)");
    assert_eq!(ex("x!::VARCHAR"), "CAST(factorial(x) AS VARCHAR)");
}

#[test]
fn factorial_precedence_is_self_consistent_but_diverges_from_duckdb_on_binary_operators() {
    // `!` binds looser than every prefix operator (`-`/`~`/`NOT`) but
    // tighter than every binary operator (`BP_BANG`'s doc has the full
    // rationale). DuckDB's own grammar for postfix `!` is internally
    // inconsistent Postgres legacy, verified against the `duckdb` CLI:
    //   3! ^ 2   -> 36.0    (works)
    //   2 ^ 3!   -> syntax error
    //   3! = 6   -> true    (works)
    //   2 + 3!   -> 120, i.e. (2+3)! silently
    //   3! + 1   -> syntax error
    // We deliberately do not replicate that inconsistency: `!` here always
    // applies to just the immediately preceding operand and never absorbs
    // a surrounding binary expression, so every one of these parses
    // (`2 + 3!`/`3! + 1` diverge from DuckDB on purpose -- see
    // docs/sql/limitations.md).
    assert_eq!(ex("3! ^ 2"), "pow(factorial(3i32), 2i32)");
    assert_eq!(ex("2 ^ 3!"), "pow(2i32, factorial(3i32))");
    assert_eq!(ex("3! = 6"), "(factorial(3i32) = 6i32)");
    assert_eq!(ex("2 + 3!"), "(2i32 + factorial(3i32))");
    assert_eq!(ex("3! + 1"), "(factorial(3i32) + 1i32)");
}

#[test]
fn factorial_binds_tighter_than_the_in_ladder_prefix_operators() {
    // `!` binds tighter than every binary operator, and `~`/`@` are prefix
    // operators that live *inside* the binary ladder (`BP_OTHER`, see
    // `prefix()`), so `!` binds tighter than them: `~5!` is `~(5!)` = -121
    // and `@5!` is `@(5!)` = 120.
    //
    // DuckDB itself is self-contradictory here -- `~5!` -> 1 (i.e. `(~5)!`)
    // but `@5!` -> 120 (i.e. `@(5!)`), for two identically-shaped
    // expressions -- so no single rule agrees with it on both. `BP_BANG`'s
    // doc in `sql::parser` has the full rationale. `@5!` matches DuckDB
    // either way (`(@5)!` and `@(5!)` are both 120); `~5!` is the one
    // deliberate divergence.
    assert_eq!(ex("~5!"), "bit_not(factorial(5i32))");
    assert_eq!(ex("@5!"), "abs(factorial(5i32))");
    // Unary `-`/`+` still sit *above* `BP_BANG`, so they are unaffected.
    assert_eq!(ex("-5!"), "factorial(-5i32)");
}

#[test]
fn bang_still_lexes_as_the_prefix_of_longer_operators() {
    // `!=`/`!~`/`!~~`/`!~~*` must still win over a bare `Bang`.
    assert_eq!(code("SELECT 4 != 5 FROM t"), 0);
    assert_eq!(code("SELECT a !~ 'x' FROM t"), 0);
    assert_eq!(code("SELECT a !~~ 'x' FROM t"), 0);
    assert_eq!(code("SELECT a !~~* 'x' FROM t"), 0);
}

// --- Typed temporal literals (`DATE '...'` and friends) ---------------------

#[test]
fn typed_temporal_literals_fold_to_typed_literal_constants() {
    // duckdb: SELECT DATE '2020-01-01' -> 2020-01-01 (type DATE).
    // 18262 = days from 1970-01-01 to 2020-01-01.
    assert_eq!(ex("DATE '2020-01-01'"), "I32(18262)::DATE");
    // duckdb: SELECT TIMESTAMP '2020-01-01 00:00:00' -> type TIMESTAMP.
    assert_eq!(ex("TIMESTAMP '2020-01-01 00:00:00'"), "I64(1577836800000000)::TIMESTAMP");
    // duckdb: SELECT TIME '01:00:00' -> type TIME.
    assert_eq!(ex("TIME '01:00:00'"), "I64(3600000000)::TIME");
    // duckdb: TIMESTAMPTZ '2020-01-01 09:00:00+09' is the same instant as
    // 2020-01-01 00:00:00 UTC.
    assert_eq!(ex("TIMESTAMPTZ '2020-01-01 09:00:00+09'"), "I64(1577836800000000)::TIMESTAMPTZ");
}

#[test]
fn typed_temporal_literal_spelling_is_case_insensitive() {
    assert_eq!(ex("date '2020-01-01'"), "I32(18262)::DATE");
    assert_eq!(ex("Timestamp '2020-01-01 00:00:00'"), "I64(1577836800000000)::TIMESTAMP");
}

#[test]
fn temporal_type_names_are_still_usable_as_column_names() {
    // The whole point of not reserving them: column names come from data
    // files. duckdb does not reserve these either (`select date, time from
    // (select 1 as date, 2 as time)` works there).
    assert_eq!(sel("SELECT date FROM t"), "SELECT date FROM t");
    assert_eq!(sel("SELECT time, timestamp FROM t"), "SELECT time, timestamp FROM t");
    assert_eq!(ex("date + 1"), "(date + 1i32)");
    assert_eq!(ex("t.date"), "t.date");
    assert_eq!(sel("SELECT 1 AS date FROM t"), "SELECT 1i32 AS date FROM t");
    assert_eq!(
        sel("SELECT a FROM t ORDER BY date"),
        "SELECT a FROM t ORDER BY date ASC NULLS LAST"
    );
    // A quoted identifier is never read as a type name.
    assert_eq!(ex("\"date\""), "date");
    // `date(x)` stays an ordinary function call, not a literal.
    assert_eq!(ex("date(x)"), "date(x)");
}

#[test]
fn unparseable_typed_temporal_literal_is_a_parse_error() {
    // duckdb raises `Conversion Error: invalid date field format` here.
    // A literal is fixed query text, so it fails loudly instead of
    // following this engine's "bad CAST input becomes NULL" rule.
    assert_eq!(code("SELECT DATE 'nonsense' FROM t"), Code::InvalidCast as u16);
    assert_eq!(code("SELECT TIMESTAMP '2020-13-01 00:00:00' FROM t"), Code::InvalidCast as u16);
    assert_eq!(code("SELECT TIME '99:99:99' FROM t"), Code::InvalidCast as u16);
}

// --- `^@` prefix operator ---------------------------------------------------

#[test]
fn caret_at_desugars_to_starts_with() {
    assert_eq!(ex("a ^@ 'x'"), "starts_with(a, 'x')");
}

#[test]
fn caret_at_binds_tighter_than_concat_on_the_right_only() {
    // duckdb: select 'a' || 'b' ^@ 'a'  -> true      i.e. ('a'||'b') ^@ 'a'
    assert_eq!(ex("'a' || 'b' ^@ 'a'"), "starts_with(('a' || 'b'), 'a')");
    // duckdb: select 'ab' ^@ 'a' || 'b' -> 'trueb'   i.e. ('ab' ^@ 'a') || 'b'
    assert_eq!(ex("'ab' ^@ 'a' || 'b'"), "(starts_with('ab', 'a') || 'b')");
    // duckdb: select 'ab' ^@ 'a' = true -> true      i.e. (...) = true
    assert_eq!(ex("'ab' ^@ 'a' = true"), "(starts_with('ab', 'a') = true)");
}

#[test]
fn caret_and_at_still_lex_separately_when_not_adjacent() {
    // `^` (pow) and prefix `@` (abs) must be unaffected by the new token.
    assert_eq!(ex("2 ^ 3"), "pow(2i32, 3i32)");
    assert_eq!(ex("2 ^ @x"), "pow(2i32, abs(x))");
}

// --- `IS [NOT] UNKNOWN` -----------------------------------------------------

#[test]
fn is_unknown_desugars_to_is_null() {
    // duckdb: NULL is unknown -> true, NULL is not unknown -> false,
    // 1 is unknown -> false. Exactly `IS [NOT] NULL`.
    assert_eq!(ex("x IS UNKNOWN"), "(x IS NULL)");
    assert_eq!(ex("x IS NOT UNKNOWN"), "(x IS NOT NULL)");
    assert_eq!(ex("x is unknown"), "(x IS NULL)");
}

#[test]
fn unknown_is_a_soft_keyword_not_reserved() {
    // A column literally named `unknown` still resolves everywhere.
    assert_eq!(sel("SELECT unknown FROM t"), "SELECT unknown FROM t");
    assert_eq!(sel("SELECT 1 AS unknown FROM t"), "SELECT 1i32 AS unknown FROM t");
    assert_eq!(ex("unknown + 1"), "(unknown + 1i32)");
    assert_eq!(ex("unknown IS NULL"), "(unknown IS NULL)");
    assert_eq!(ex("unknown IS UNKNOWN"), "(unknown IS NULL)");
    assert!(keyword(b"unknown").is_none(), "UNKNOWN must not be a reserved word");
}

// --- SQL-standard functional syntaxes ---------------------------------------

#[test]
fn position_in_form_desugars_to_strpos_with_swapped_arguments() {
    // duckdb: position('b' in 'abc') = strpos('abc','b') = 2. Note that the
    // argument order flips.
    assert_eq!(ex("position('b' IN 'abc')"), "strpos('abc', 'b')");
    assert_eq!(ex("position(needle IN haystack)"), "strpos(haystack, needle)");
    // `||` is stronger than the `IN` separator, so it stays inside the
    // searched-for operand.
    assert_eq!(ex("position('a' || 'b' IN s)"), "strpos(s, ('a' || 'b'))");
}

#[test]
fn position_without_in_stays_an_ordinary_call() {
    // This engine's own `position(a, b)` alias of `strpos` is untouched
    // (duckdb rejects that spelling, but the alias predates this change).
    assert_eq!(ex("position(a, b)"), "position(a, b)");
    // A column named `position` also still works.
    assert_eq!(sel("SELECT position FROM t"), "SELECT position FROM t");
    assert_eq!(ex("position + 1"), "(position + 1i32)");
}

#[test]
fn substring_from_for_desugars_to_positional_arguments() {
    // The written spelling (`substring`/`substr`) is preserved -- they are
    // the same function in `expr::funcs`; only the argument *shape* changes.
    // duckdb: substring('abcdef' from 2) -> 'bcdef'
    assert_eq!(ex("substring('abcdef' FROM 2)"), "substring('abcdef', 2i32)");
    // duckdb: substring('abcdef' from 2 for 3) -> 'bcd'
    assert_eq!(ex("substring('abcdef' FROM 2 FOR 3)"), "substring('abcdef', 2i32, 3i32)");
    // duckdb: substring('abcdef' for 3) -> 'abc' (start defaults to 1)
    assert_eq!(ex("substring('abcdef' FOR 3)"), "substring('abcdef', 1i32, 3i32)");
    // Bounds may be arbitrary expressions (duckdb: `... from 1+1 for 1+2`).
    assert_eq!(ex("substring(s FROM a + 1 FOR b + 2)"), "substring(s, (a + 1i32), (b + 2i32))");
    // `substr` accepts the same syntax.
    assert_eq!(ex("substr('abcdef' FROM 2)"), "substr('abcdef', 2i32)");
}

#[test]
fn substring_comma_form_and_for_as_a_column_name_still_work() {
    assert_eq!(ex("substring(s, 2, 3)"), "substring(s, 2i32, 3i32)");
    assert_eq!(sel("SELECT for FROM t"), "SELECT for FROM t");
    assert_eq!(ex("substring(for, 2)"), "substring(for, 2i32)");
}

#[test]
fn trim_from_form_desugars_to_trim_ltrim_rtrim() {
    // Every line here was verified against the `duckdb` CLI; see
    // `Parser::trim_from_call`'s doc comment for the measured results.
    assert_eq!(ex("trim(BOTH 'x' FROM s)"), "trim(s, 'x')");
    assert_eq!(ex("trim(LEADING 'x' FROM s)"), "ltrim(s, 'x')");
    assert_eq!(ex("trim(TRAILING 'x' FROM s)"), "rtrim(s, 'x')");
    assert_eq!(ex("trim('x' FROM s)"), "trim(s, 'x')");
    assert_eq!(ex("trim(FROM s)"), "trim(s)");
    assert_eq!(ex("trim(BOTH FROM s)"), "trim(s)");
    assert_eq!(ex("trim(leading from s)"), "ltrim(s)");
}

#[test]
fn trim_comma_form_and_direction_words_as_column_names_still_work() {
    assert_eq!(ex("trim(s)"), "trim(s)");
    assert_eq!(ex("trim(s, 'x')"), "trim(s, 'x')");
    // Without a top-level FROM inside the call, `both`/`leading`/`trailing`
    // are never read as direction words -- they stay ordinary columns.
    assert_eq!(ex("trim(leading)"), "trim(leading)");
    assert_eq!(ex("trim(leading, 'x')"), "trim(leading, 'x')");
    assert_eq!(sel("SELECT trailing FROM t"), "SELECT trailing FROM t");
}

#[test]
fn top_level_keyword_lookahead_ignores_nested_occurrences() {
    // A `FROM` belonging to a nested subquery must not turn this into the
    // SQL-standard `trim(... FROM ...)` form.
    assert_eq!(ex("trim((SELECT x FROM t))"), "trim((SELECT x FROM t))");
    // Likewise a nested `IN` inside parentheses.
    assert_eq!(ex("position((a IN (1, 2)), b)"), "position((a IN [1i32, 2i32]), b)");
}

// --- `GROUP BY ALL` / `ORDER BY ALL` ----------------------------------------

#[test]
fn group_by_all_and_order_by_all_parse_as_flags() {
    assert_eq!(sel("SELECT a, sum(b) FROM t GROUP BY ALL"), "SELECT a, sum(b) FROM t GROUP BY ALL");
    assert_eq!(sel("SELECT a FROM t ORDER BY ALL"), "SELECT a FROM t ORDER BY ALL ASC NULLS LAST");
    assert_eq!(
        sel("SELECT a FROM t ORDER BY ALL DESC"),
        "SELECT a FROM t ORDER BY ALL DESC NULLS LAST"
    );
    assert_eq!(
        sel("SELECT a FROM t ORDER BY ALL NULLS FIRST"),
        "SELECT a FROM t ORDER BY ALL ASC NULLS FIRST"
    );
    assert_eq!(
        sel("SELECT a FROM t ORDER BY ALL DESC NULLS LAST LIMIT 3"),
        "SELECT a FROM t ORDER BY ALL DESC NULLS LAST LIMIT 3"
    );
}

#[test]
fn order_by_all_after_a_set_operation_lands_on_the_query() {
    assert_eq!(
        qs("SELECT a FROM t UNION ALL SELECT a FROM t2 ORDER BY ALL"),
        "(SELECT a FROM t UNION ALL SELECT a FROM t2) ORDER BY ALL ASC NULLS LAST"
    );
}

#[test]
fn all_cannot_be_mixed_with_an_explicit_list() {
    // duckdb rejects both of these with a parser error at the comma.
    assert_eq!(code("SELECT a FROM t GROUP BY ALL, a"), Code::UnexpectedToken as u16);
    assert_eq!(code("SELECT a FROM t ORDER BY ALL, a"), Code::UnexpectedToken as u16);
}

#[test]
fn order_by_all_is_rejected_for_pivot_and_unpivot() {
    // The desugaring rebuilds the output column list, so the shorthand
    // cannot be carried through; refuse it rather than silently dropping it.
    assert_eq!(
        code("PIVOT t ON a IN (1) USING count(*) ORDER BY ALL"),
        Code::UnsupportedFeature as u16
    );
    assert_eq!(
        code("UNPIVOT t ON a, b INTO NAME n VALUE v ORDER BY ALL"),
        Code::UnsupportedFeature as u16
    );
}

// --- Operator precedence against the PostgreSQL/DuckDB ladder --------------
//
// Every expectation below was measured with the `duckdb` CLI (its `EXPLAIN`
// output prints the parse tree fully parenthesized, which is what pins the
// grouping rather than just the value). See the `BP_*` constants in
// `sql::parser` for the ladder these tests fix in place.

#[test]
fn non_arithmetic_prefix_operators_bind_below_multiplication() {
    // `~`/`@` are not high-precedence prefix operators: PostgreSQL gives every
    // prefix operator except `+`/`-` the "any other operator" precedence, so
    // their operand extends over `*`, `+` and `^` but stops at `||` and the
    // bitwise operators.
    //   duckdb: ~1 * 2 -> -3 (= ~(1*2)), ~1 + 1 -> -3, ~1 || 'a' -> '-2a',
    //           ~1 = -2 -> true, @ -3 + 1 -> 2
    assert_eq!(ex("~1 * 2"), "bit_not((1i32 * 2i32))");
    assert_eq!(ex("~id + 1"), "bit_not((id + 1i32))");
    assert_eq!(ex("@id - 5"), "abs((id - 5i32))");
    assert_eq!(ex("~2 ^ 2"), "bit_not(pow(2i32, 2i32))");
    // ... but stops before the operators of its own band, which are
    // left-associative, and before comparison.
    assert_eq!(ex("~1 || 'a'"), "(bit_not(1i32) || 'a')");
    assert_eq!(ex("~1 & 2"), "bit_and(bit_not(1i32), 2i32)");
    assert_eq!(ex("~1 = -2"), "(bit_not(1i32) = -2i32)");
    // Unary `-`/`+` keep their high precedence.
    assert_eq!(ex("-1 * 2"), "(-1i32 * 2i32)");
}

#[test]
fn in_between_like_bind_tighter_than_comparison() {
    // duckdb: `false = true IN (false, true)` -> false, i.e.
    // `false = (true IN (false, true))`; likewise for BETWEEN/LIKE/ILIKE and
    // their NOT-prefixed forms. Before this, they sat at `BP_CMP` and attached
    // to the *finished* comparison instead.
    assert_eq!(ex("a = b IN (c, d)"), "(a = (b IN [c, d]))");
    assert_eq!(ex("a = b NOT IN (c, d)"), "(a = (b NOT IN [c, d]))");
    assert_eq!(ex("a = b BETWEEN c AND d"), "(a = (b BETWEEN c AND d))");
    assert_eq!(ex("a = b NOT BETWEEN c AND d"), "(a = (b NOT BETWEEN c AND d))");
    assert_eq!(ex("a = b LIKE 'x'"), "(a = (b LIKE 'x'))");
    assert_eq!(ex("a = b ILIKE 'x'"), "(a = (b ILIKE 'x'))");
    // The predicate still binds looser than everything above it, so the left
    // operand absorbs arithmetic and `||` as before.
    assert_eq!(ex("a + 1 IN (c)"), "((a + 1i32) IN [c])");
    // And a predicate applied to a finished predicate stays left-associative.
    assert_eq!(ex("a IN (b) IN (c)"), "((a IN [b]) IN [c])");
}

#[test]
fn is_family_stays_at_comparison_strength() {
    // PostgreSQL puts `IS` one notch below comparison, DuckDB collapses the two.
    //   duckdb: `1 IS DISTINCT FROM 1 = 1` -> false, `2 = 1 IS DISTINCT FROM 1`
    //           -> true, `1 = 1 IS NOT NULL` -> true, `true = 1 ISNULL` -> false
    // All four are "one left-associative band", which is what `BP_CMP` is.
    assert_eq!(ex("a = b IS NULL"), "((a = b) IS NULL)");
    assert_eq!(ex("a = b ISNULL"), "((a = b) IS NULL)");
    assert_eq!(ex("a = b NOTNULL"), "((a = b) IS NOT NULL)");
    // A predicate binds tighter than `IS`, and `IS` tighter than nothing else
    // in that band, so the two compose left to right in source order.
    assert_eq!(ex("a IN (b) IS NULL"), "((a IN [b]) IS NULL)");
    assert_eq!(ex("a IS NULL IN (b)"), "((a IS NULL) IN [b])");
}

#[test]
fn bitwise_and_concat_share_one_left_associative_band() {
    // duckdb: `1 & 2 || 3` -> '03' (= `(1&2) || 3`), `1 || 2 & 3` -> a binder
    // error naming `&(VARCHAR, INTEGER)` (= `(1||2) & 3`), `1 + 2 & 3` -> 3
    // (= `(1+2) & 3`), `3 & 2 = 2` -> true (= `(3&2) = 2`).
    assert_eq!(ex("1 & 2 || 3"), "(bit_and(1i32, 2i32) || 3i32)");
    assert_eq!(ex("1 || 2 & 3"), "bit_and((1i32 || 2i32), 3i32)");
    assert_eq!(ex("1 + 2 & 3"), "bit_and((1i32 + 2i32), 3i32)");
    assert_eq!(ex("3 & 2 = 2"), "(bit_and(3i32, 2i32) = 2i32)");
    assert_eq!(ex("1 << 2 | 3"), "bit_or(bit_shift_left(1i32, 2i32), 3i32)");
}

#[test]
fn between_bounds_reach_into_the_bitwise_band() {
    // The bounds are read at `BP_OTHER`, so `||`, `&`, `<<` and arithmetic all
    // combine into a bound while the separating `AND` still terminates the low
    // one. duckdb: `1 BETWEEN 0 AND 1 & 1` -> true, `2 BETWEEN 1 AND 1 << 2` ->
    // true, `5 BETWEEN 1 << 1 AND 10` -> true (the last was an outright syntax
    // error here before).
    assert_eq!(ex("1 BETWEEN 0 AND 1 & 1"), "(1i32 BETWEEN 0i32 AND bit_and(1i32, 1i32))");
    assert_eq!(ex("2 BETWEEN 1 AND 1 << 2"), "(2i32 BETWEEN 1i32 AND bit_shift_left(1i32, 2i32))");
    assert_eq!(
        ex("5 BETWEEN 1 << 1 AND 10"),
        "(5i32 BETWEEN bit_shift_left(1i32, 1i32) AND 10i32)"
    );
    // The `AND` that separates the bounds is still not swallowed.
    assert_eq!(ex("a BETWEEN b AND c AND d"), "((a BETWEEN b AND c) AND d)");
}

#[test]
fn other_band_operators_bind_tighter_than_comparison() {
    // `^@`, the regex operators and the `~~` LIKE-punctuation family all live in
    // the "any other operator" band, so a comparison on their left is *not*
    // their left operand. duckdb: `true = 'ab' ^@ 'a'` -> true,
    // `true = 'a' ~ 'a'` -> true, `true = 'a' GLOB 'a'` -> true,
    // `'a' ~ 'a' || 'b'` -> 'trueb'.
    assert_eq!(ex("a = 'ab' ^@ 'a'"), "(a = starts_with('ab', 'a'))");
    assert_eq!(ex("a = 'x' ~ 'y'"), "(a = regexp_full_match('x', 'y'))");
    assert_eq!(ex("a = 'x' ~~ 'y'"), "(a = ('x' LIKE 'y'))");
    assert_eq!(ex("a = 'x' GLOB 'y'"), "(a = glob('x', 'y'))");
    // Their right operand still stops before `||`, so a trailing `||` applies to
    // the result (duckdb: `'a' ~ 'a' || 'b'` -> 'trueb').
    assert_eq!(ex("'x' ~ 'y' || 'z'"), "(regexp_full_match('x', 'y') || 'z')");
}

#[test]
fn position_and_pivot_still_stop_before_their_own_in_keyword() {
    // Both read their operand one notch tighter than `BP_PRED` so the `IN` that
    // separates the two halves is not eaten as the `x IN (...)` predicate. The
    // constant moved with `IN`'s precedence; this pins that they moved together.
    assert_eq!(ex("position('a' || 'b' in s)"), "strpos(s, ('a' || 'b'))");
    // `PIVOT ... ON <expr> IN (...)` is not a `Stmt::Select`, so only that it
    // still parses (rather than swallowing the `IN` into the `ON` expression)
    // is checked here.
    assert_eq!(code("PIVOT t ON a IN (1, 2) USING sum(b)"), 0);
    assert_eq!(code("PIVOT t ON a || 'x' IN (1, 2) USING sum(b)"), 0);
}

// --- Numeric literals ------------------------------------------------------

#[test]
fn underscore_digit_separators_in_numeric_literals() {
    // DuckDB accepts `_` *between digits* only. Before this, `number()` stopped
    // at the `_`, so `SELECT 1_000` silently became the literal `1` with an
    // implicit alias `_000`, and `SELECT 1_000 + 1` was a syntax error.
    assert_eq!(ex("1_000"), "1000i32");
    assert_eq!(ex("1_000 + 1"), "(1000i32 + 1i32)");
    assert_eq!(ex("1_0_0"), "100i32");
    assert_eq!(ex("1_000_000"), "1000000i32");
    assert_eq!(ex("1_000.5"), "1000.5f64");
    assert_eq!(ex("1.0_5"), "1.05f64");
    assert_eq!(ex("1e1_0"), "10000000000f64");
    // Leading, trailing, doubled, and point/exponent-adjacent underscores all
    // end the number, leaving an identifier behind -- exactly as in duckdb,
    // where `1__0`, `100_`, `1._5`, `1_e5` and `1e_5` all answer a bare `1`
    // or `100` with an implicit alias.
    assert_eq!(sel("SELECT 1__0"), "SELECT 1i32 AS __0");
    assert_eq!(sel("SELECT 100_"), "SELECT 100i32 AS _");
    assert_eq!(sel("SELECT 1e_5"), "SELECT 1i32 AS e_5");
    // `_100` is an ordinary identifier, not a number.
    assert_eq!(ex("_100"), "_100");
    // LIMIT/OFFSET go through `Parser::uint`, which skips separators too.
    assert_eq!(sel("SELECT a FROM t LIMIT 1_000"), "SELECT a FROM t LIMIT 1000");
}

#[test]
fn leading_dot_float_literals() {
    // duckdb: `SELECT .5` -> 0.5, `SELECT .5 + 1` -> 1.5. `5.` already worked.
    assert_eq!(ex(".5"), "0.5f64");
    assert_eq!(ex(".5 + 1"), "(0.5f64 + 1i32)");
    assert_eq!(ex(".5e1"), "5f64");
    assert_eq!(ex("5."), "5f64");
    // A `.` not followed by a digit is still the qualification separator, and a
    // digit after a qualified name is still a syntax error (as in duckdb).
    assert_eq!(ex("t.c"), "t.c");
    assert_eq!(ex("t.*"), "t.*");
    assert_eq!(code("SELECT t.5 FROM t"), Code::UnexpectedToken as u16);
}

// --- Aliases ---------------------------------------------------------------

#[test]
fn reserved_words_are_accepted_as_aliases_after_an_explicit_as() {
    // duckdb accepts every one of these; the quoted spellings already worked
    // here. `AS` has already fixed the position, so a keyword there can only be
    // a name.
    assert_eq!(sel("SELECT 1 AS limit"), "SELECT 1i32 AS limit");
    assert_eq!(sel("SELECT 1 AS offset"), "SELECT 1i32 AS offset");
    assert_eq!(sel("SELECT 1 AS all"), "SELECT 1i32 AS all");
    assert_eq!(sel("SELECT 1 AS end"), "SELECT 1i32 AS end");
    assert_eq!(sel("SELECT 1 AS distinct"), "SELECT 1i32 AS distinct");
    assert_eq!(sel("SELECT 1 AS select"), "SELECT 1i32 AS select");
    // The spelling keeps the case the user typed, like any identifier.
    assert_eq!(sel("SELECT 1 AS LIMIT"), "SELECT 1i32 AS LIMIT");
    // Table aliases take the same path.
    assert_eq!(sel("SELECT a FROM t AS order"), "SELECT a FROM t AS order");
    // A *bare* alias must still stop at a reserved word, or a clause boundary
    // would be eaten as a name.
    assert_eq!(sel("SELECT a FROM t WHERE b"), "SELECT a FROM t WHERE b");
    assert_eq!(sel("SELECT a FROM t LIMIT 1"), "SELECT a FROM t LIMIT 1");
}

// --- INTERVAL --------------------------------------------------------------

#[test]
fn interval_is_nameable_in_a_type_position() {
    // `INTERVAL` is a first-class type (DESIGN.md §8) but was missing from the
    // `TYPES` table, so it could not be *named* as one: every spelling below
    // used to fail with `InvalidCast`.
    assert_eq!(ex("CAST(NULL AS INTERVAL)"), "CAST(NULL AS INTERVAL)");
    assert_eq!(ex("CAST(x AS INTERVAL)"), "CAST(x AS INTERVAL)");
    assert_eq!(ex("x::INTERVAL"), "CAST(x AS INTERVAL)");
    assert_eq!(ex("TRY_CAST(x AS interval)"), "TRY_CAST(x AS INTERVAL)");
}

#[test]
fn interval_text_accepts_time_components_fractions_and_weeks() {
    let lit =
        |m: i32, d: i32, u: i64| format!("INTERVAL({}i128)", crate::vector::pack_interval(m, d, u));
    const US_PER_SEC: i64 = 1_000_000;
    // A bare `HH:MM[:SS[.frac]]` component -- the shape an interval *prints* as,
    // so an interval this engine emitted can now be read back in.
    // duckdb: '1:30:00' -> 01:30:00, '01:02:03.5' -> 01:02:03.5,
    //         '01:02' -> 01:02:00, '100:00:00' -> 100:00:00, '-1:30:00' -> -01:30:00
    assert_eq!(ex("INTERVAL '1:30:00'"), lit(0, 0, 90 * 60 * US_PER_SEC));
    assert_eq!(ex("INTERVAL '01:02:03.5'"), lit(0, 0, 3723 * US_PER_SEC + 500_000));
    assert_eq!(ex("INTERVAL '01:02'"), lit(0, 0, 62 * 60 * US_PER_SEC));
    assert_eq!(ex("INTERVAL '100:00:00'"), lit(0, 0, 100 * 60 * 60 * US_PER_SEC));
    assert_eq!(ex("INTERVAL '-1:30:00'"), lit(0, 0, -90 * 60 * US_PER_SEC));
    // Fractional amounts cascade into the next smaller field, as in duckdb:
    //   '1.5 days' -> 1 day 12:00:00     '0.5 months' -> 15 days
    //   '1.25 years' -> 1 year 3 months  '1.5 weeks' -> 10 days 12:00:00
    assert_eq!(ex("INTERVAL '1.5 days'"), lit(0, 1, 12 * 60 * 60 * US_PER_SEC));
    assert_eq!(ex("INTERVAL '0.5 months'"), lit(0, 15, 0));
    assert_eq!(ex("INTERVAL '1.25 years'"), lit(15, 0, 0));
    assert_eq!(ex("INTERVAL '1.5 hours'"), lit(0, 0, 90 * 60 * US_PER_SEC));
    assert_eq!(ex("INTERVAL '1.5 seconds'"), lit(0, 0, US_PER_SEC + 500_000));
    // `week`/`weeks`, singular and plural (duckdb: '3 weeks' -> 21 days).
    assert_eq!(ex("INTERVAL '3 weeks'"), lit(0, 21, 0));
    assert_eq!(ex("INTERVAL '1 week'"), lit(0, 7, 0));
    assert_eq!(ex("INTERVAL '1.5 weeks'"), lit(0, 10, 12 * 60 * 60 * US_PER_SEC));
    assert_eq!(ex("INTERVAL 3 WEEK"), lit(0, 21, 0));
    // Several terms in one string, including a time component alongside units.
    assert_eq!(ex("INTERVAL '1 day 01:02:03'"), lit(0, 1, 3723 * US_PER_SEC));
    assert_eq!(ex("INTERVAL '2 days 03:04:05.678'"), lit(0, 2, 11045 * US_PER_SEC + 678_000));
    assert_eq!(ex("INTERVAL '-2 days -03:04:05'"), lit(0, -2, -11045 * US_PER_SEC));
    assert_eq!(ex("INTERVAL '1 day 2 hours 3 minutes'"), lit(0, 1, 7380 * US_PER_SEC));
    // Malformed text is still rejected rather than silently truncated.
    assert_eq!(code("SELECT INTERVAL '1 bogus'"), Code::SyntaxError as u16);
    assert_eq!(code("SELECT INTERVAL '1 day 2'"), Code::SyntaxError as u16);
    assert_eq!(code("SELECT INTERVAL '1:2:3:4'"), Code::SyntaxError as u16);
    assert_eq!(code("SELECT INTERVAL ''"), Code::SyntaxError as u16);
}
