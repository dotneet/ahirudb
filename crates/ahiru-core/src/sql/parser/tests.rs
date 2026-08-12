use super::*;
use crate::error::Code;
#[cfg(feature = "ddl")]
use crate::sql::ast::AlterTableAction;
#[cfg(feature = "dml")]
use crate::sql::ast::InsertSource;
use crate::sql::lexer::{keyword, KEYWORDS};

// --- テスト用ヘルパ -----------------------------------------------------

/// 式木を完全括弧付きの文字列へ戻す。リテラルは型が分かる形で出す。
fn r(a: &ExprArena, id: ExprId) -> String {
    match a.get(id) {
        // ウィンドウ指定は `PARTITION BY .. ORDER BY .. <枠>` の順で出す。
        // 枠は既定値の取り違えを検出したいので必ず描画する。
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
        Expr::Like { arg, pattern, negated, escape, ci } => {
            let esc = match escape {
                Some(c) => format!(" ESCAPE '{}'", *c as char),
                None => String::new(),
            };
            format!(
                "({}{} {} {}{})",
                r(a, *arg),
                if *negated { " NOT" } else { "" },
                if *ci { "ILIKE" } else { "LIKE" },
                r(a, *pattern),
                esc
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
                // SEMI / ANTI 系は構文からは作られない（バインダ専用）。
                // 種類が増えてもパーサのテストが壊れないよう一括で受ける。
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

/// SELECT 文を 1 行に潰す。構造の比較用で、SQL として妥当な形ではない。
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

/// 集合演算の木。括弧を必ず付けるので結合の向きがそのまま読める。
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

/// クエリ全体（CTE + 本体 + 外側の ORDER BY / LIMIT）。
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

/// `QueryStmt` から素の `SelectStmt` を取り出す（集合演算・CTE 無しを前提）。
fn plain(q: &QueryStmt) -> &SelectStmt {
    match &q.body {
        SetExpr::Select(s) => s,
        _ => panic!("集合演算を含むクエリ"),
    }
}

fn sel(sql: &str) -> String {
    let p = parse(sql).expect("parse failed");
    match &p.stmt {
        Stmt::Select(q) => select_str(&p.arena, plain(q)),
        _ => panic!("not a SELECT"),
    }
}

/// クエリ全体を描画する。集合演算・CTE を含む文はこちらで見る。
fn qs(sql: &str) -> String {
    let p = parse(sql).expect("parse failed");
    match &p.stmt {
        Stmt::Select(q) => query_str(&p.arena, q),
        _ => panic!("not a SELECT"),
    }
}

/// 文をパースして `QueryStmt` を取り出す。木の形を直接見るテスト用。
fn parsed(sql: &str) -> (ExprArena, Box<QueryStmt>) {
    let p = parse(sql).expect("parse failed");
    match p.stmt {
        Stmt::Select(q) => (p.arena, q),
        _ => panic!("not a SELECT"),
    }
}

/// `SELECT <expr>` を通して式だけを描画する。
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

// --- 字句 ---------------------------------------------------------------

#[test]
fn keyword_table_is_sorted_and_complete() {
    // 二分探索の前提: (長さ, 小文字先頭バイト) が単調非減少。
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
        sel("SELECT /* 列 */ a -- 末尾コメント\n FROM t /* 途中 */ WHERE a > 1"),
        "SELECT a FROM t WHERE (a > 1i32)"
    );
    assert_eq!(sel("SELECT 1 --x"), "SELECT 1i32");
    // ブロックコメントは入れ子にしない: 最初の */ で閉じる。
    assert_eq!(sel("SELECT /* /* */ 1"), "SELECT 1i32");
    assert_eq!(code("SELECT /* 閉じない 1"), Code::SyntaxError as u16);
}

// --- 優先順位・結合 -----------------------------------------------------

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
    // == と <> は = / != の別名。
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

// --- 述語 ---------------------------------------------------------------

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
    assert_eq!(ex("x LIKE 'a!%' ESCAPE '!'"), "(x LIKE 'a!%' ESCAPE '!')");
    // BETWEEN の区切り AND は論理演算子として食わない。
    assert_eq!(ex("a BETWEEN 1 AND 2 AND b"), "((a BETWEEN 1i32 AND 2i32) AND b)");
    assert_eq!(ex("a BETWEEN 1 + 1 AND 2 * 3"), "(a BETWEEN (1i32 + 1i32) AND (2i32 * 3i32))");
    // 述語は比較と同じ強さ。AND より強く結合する。
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
    assert_eq!(ex("x ILIKE 'a!%' ESCAPE '!'"), "(x ILIKE 'a!%' ESCAPE '!')");
    // ILIKE も述語なので AND より強く結合する。
    assert_eq!(ex("a ILIKE 'x' AND b"), "((a ILIKE 'x') AND b)");
    // `ILIKE` は完全な予約語（列名としては使えない）。
    assert_eq!(code("SELECT ilike FROM t"), Code::UnexpectedToken as u16);
}

#[test]
fn glob_operator_desugars_to_glob_function() {
    assert_eq!(ex("a GLOB 'x*'"), "glob(a, 'x*')");
    assert_eq!(ex("a glob 'x*'"), "glob(a, 'x*')");
    // GLOB も述語と同じ強さ（AND より強く結合する）。
    assert_eq!(ex("a GLOB 'x' AND b"), "(glob(a, 'x') AND b)");
    // DuckDB は `NOT GLOB` を書けない（`duckdb -c "select 'a' NOT GLOB
    // 'b'"` が構文エラーになることを確認済み）。`NOT (x GLOB y)` は
    // 通常の前置 NOT なので問題なく書ける。
    assert_eq!(code("SELECT a NOT GLOB 'x'"), Code::UnexpectedToken as u16);
    assert_eq!(ex("NOT (a GLOB 'x')"), "(NOT glob(a, 'x'))");
    // `glob` は予約語ではない（ROWS/RANGE/QUALIFY と同じ「文脈依存
    // キーワード」方式）ので、列名としてそのまま使える。
    assert_eq!(ex("glob"), "glob");
    assert_eq!(code("SELECT glob FROM t"), 0);
}

#[test]
fn similar_to_desugars_to_regexp_full_match() {
    assert_eq!(ex("a SIMILAR TO 'x.y'"), "regexp_full_match(a, 'x.y')");
    assert_eq!(ex("a similar to 'x.y'"), "regexp_full_match(a, 'x.y')");
    assert_eq!(ex("a NOT SIMILAR TO 'x.y'"), "(NOT regexp_full_match(a, 'x.y'))");
    // LIKE と同じく述語の強さ（AND より強く結合する）。
    assert_eq!(ex("a SIMILAR TO 'x' AND b"), "(regexp_full_match(a, 'x') AND b)");
    // DuckDB 自身も `ESCAPE` 句は「未実装」として拒否する
    // （`duckdb -c "select 'a' similar to 'a' escape '\\'"` を確認済み）。
    assert_eq!(code(r"SELECT a SIMILAR TO 'x' ESCAPE '\'"), Code::UnsupportedFeature as u16);
    // `similar`/`to` はどちらも予約語ではないので、列名としてそのまま
    // 使える（過去の ROWS/RANGE/QUALIFY の教訓を踏まえた設計）。
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
    // `distinct`/`from` はどちらも既存の予約語なので、列名としては
    // 引用符が要る（`similar`/`glob`/`to` のような文脈依存キーワードでは
    // ない）。
    assert_eq!(code(r#"SELECT "distinct" FROM t"#), 0);
}

#[test]
fn cast_shorthand_desugars_to_cast() {
    assert_eq!(ex("x::INTEGER"), "CAST(x AS INTEGER)");
    assert_eq!(ex("'42'::INTEGER"), "CAST('42' AS INTEGER)");
    // `::` は前置演算子より強く結合する（`duckdb -c "select -1::varchar"`
    // が `-(1::VARCHAR)` と解釈されて型エラーになることで確認済み）。
    assert_eq!(ex("-1::VARCHAR"), "CAST(-1i32 AS VARCHAR)");
    assert_eq!(ex("(1 + 2)::VARCHAR"), "CAST((1i32 + 2i32) AS VARCHAR)");
    // 連続適用も畳み込める。
    assert_eq!(ex("x::INTEGER::VARCHAR"), "CAST(CAST(x AS INTEGER) AS VARCHAR)");
}

#[test]
fn power_operator_desugars_to_pow() {
    assert_eq!(ex("2 ^ 10"), "pow(2i32, 10i32)");
    assert_eq!(ex("2 ** 10"), "pow(2i32, 10i32)");
    // 左結合（`duckdb -c "select 2^3^2"` = 64 を確認済み。BP_POW の doc
    // 参照）。
    assert_eq!(ex("2 ^ 3 ^ 2"), "pow(pow(2i32, 3i32), 2i32)");
    // `*`/`/` より強く、単項 `-` より弱い。
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
    // `&`/`|` は比較より強く、`+`/`-` より弱い（`duckdb -c "select 1 + 2 &
    // 3"` = `(1 + 2) & 3`、`duckdb -c "select 1 & 2 = 0"` = `(1 & 2) = 0`
    // を確認済み）。
    assert_eq!(ex("1 + 2 & 3"), "bit_and((1i32 + 2i32), 3i32)");
    assert_eq!(ex("1 & 2 = 0"), "(bit_and(1i32, 2i32) = 0i32)");
}

#[test]
fn tilde_operators_desugar_to_regexp_full_match() {
    // 中置の `~`/`!~`（正規表現一致）。前置の `~`（ビット単位 NOT）は
    // 上の `bitwise_operators_desugar_to_bit_functions` で確認済み ――
    // 同じトークンだが式の途中か先頭かで意味が変わる（`-` と同じ
    // パターン、`prefix`/`expr_body` の doc 参照）。
    assert_eq!(ex("a ~ 'x.y'"), "regexp_full_match(a, 'x.y')");
    assert_eq!(ex("a !~ 'x.y'"), "(NOT regexp_full_match(a, 'x.y'))");
}

#[test]
fn array_literal_desugars_to_list_value() {
    assert_eq!(ex("[1, 2, 3]"), "list_value(1i32, 2i32, 3i32)");
    assert_eq!(ex("['a', 'b']"), "list_value('a', 'b')");
    assert_eq!(ex("[1 + 1]"), "list_value((1i32 + 1i32))");
    // 空配列は `list_value()` を経由せず（`resolve` が 0 引数を拒否する
    // 設計になっているため）、JSON の空配列を直接 TypedLiteral として
    // 埋め込む。`duckdb -c "select []"` が有効な式であることを確認済み。
    assert_eq!(ex("[]"), "Bytes([91, 93])::JSON");
}

// --- 添字アクセス / スライス（`primary`/`subscript`） -----------------------

#[test]
fn subscript_desugars_to_list_extract() {
    // `expr[i]` は `list_extract(expr, i)` への糖衣構文。式の先頭の `[`
    // （配列リテラル）とは位置で区別される（`primary_atom` 冒頭のコメント
    // 参照）。
    assert_eq!(ex("a[1]"), "list_extract(a, 1i32)");
    assert_eq!(ex("a[-1]"), "list_extract(a, -1i32)");
    // 添字自体が任意の式でよい。
    assert_eq!(ex("a[b + 1]"), "list_extract(a, (b + 1i32))");
    // 入れ子でも破綻しない: `duckdb -c "select [[1,2],[3,4]][1]"` の構文と
    // 同じ形が壊れないことを確認済み。
    assert_eq!(
        ex("[[1, 2], [3, 4]][1]"),
        "list_extract(list_value(list_value(1i32, 2i32), list_value(3i32, 4i32)), 1i32)"
    );
    assert_eq!(ex("a[1][2]"), "list_extract(list_extract(a, 1i32), 2i32)");
}

#[test]
fn slice_desugars_to_list_slice_with_omittable_bounds() {
    // `expr[i:j]` は `list_slice(expr, i, j)` への糖衣構文。
    assert_eq!(ex("a[2:3]"), "list_slice(a, 2i32, 3i32)");
    // 開始省略は `1`、終了省略は `i64::MAX` に脱糖する（`subscript` の
    // doc コメント参照。SQL NULL には脱糖しない — `list_slice` 自身は
    // 引数 NULL を NULL 伝播で処理するので、NULL に脱糖すると `[:3]` が
    // NULL になってしまう）。
    assert_eq!(ex("a[:3]"), "list_slice(a, 1i64, 3i32)");
    assert_eq!(ex("a[2:]"), format!("list_slice(a, 2i32, {}i64)", i64::MAX));
    assert_eq!(ex("a[:]"), format!("list_slice(a, 1i64, {}i64)", i64::MAX));
}

#[test]
fn postfix_cast_and_subscript_interleave_left_to_right() {
    // `[1,2,3][1]::varchar` は「先に添字、後にキャスト」
    // （`duckdb -c "select [1,2,3][1]::varchar"` で確認済み）。
    assert_eq!(ex("a[1]::varchar"), "CAST(list_extract(a, 1i32) AS VARCHAR)");
    // `a::json[1]` は「先にキャスト、後に添字」
    // （`duckdb -c "select [1,2,3]::json[1]"` で確認済み。この構文は
    // DuckDB では ARRAY 型リテラル `json[1]` と紛れるが、この実装に
    // 固定長 ARRAY 型は無いので `a::json` の後置添字として一意に読める）。
    assert_eq!(ex("a::json[1]"), "list_extract(CAST(a AS JSON), 1i32)");
    // 単項 `-` より強く結合する: `duckdb -c "select -[1,2,3][1]"` は
    // `-(list[1])` になる。
    assert_eq!(ex("-a[1]"), "(- list_extract(a, 1i32))");
    // 前置の負数リテラル即畳み込み経路（`prefix`）は添字までは畳まない
    // （`duckdb -c "select -5[1]"` が構文エラーになることで確認済み。
    // `cast_postfix` の doc コメント参照）。
    assert_eq!(code("SELECT -5[1]"), Code::UnexpectedToken as u16);
}

// --- CASE / CAST / 関数 -------------------------------------------------

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
    // `->`/`->>` は新しい BinaryOp を増やさず、`json_extract`/
    // `json_extract_string` 呼び出しへの糖衣構文として展開される。
    assert_eq!(ex("a -> b"), "json_extract(a, b)");
    assert_eq!(ex("a ->> b"), "json_extract_string(a, b)");
    // 左結合で連鎖できる（`a -> b -> c` = `(a -> b) -> c`）。
    assert_eq!(ex("a -> b -> c"), "json_extract(json_extract(a, b), c)");
    // Postgres の「その他の演算子」band と同じく `||` と同じ強さなので、
    // 比較より強く結合する（`doc -> 'a' = 1` に括弧が要らない）。
    assert_eq!(ex("a -> b = c"), "(json_extract(a, b) = c)");
    assert_eq!(ex("a ->> b = c"), "(json_extract_string(a, b) = c)");
    // `-` の単純な引き算とは区別される。
    assert_eq!(ex("a - b"), "(a - b)");
}

#[test]
fn lambda_is_recognized_only_in_list_transform_filter_reduce_arg_position() {
    // 単一引数は括弧無し。
    assert_eq!(ex("list_transform(a, x -> x + 1)"), "list_transform(a, x -> (x + 1i32))");
    // 複数引数は括弧付き。
    assert_eq!(ex("list_reduce(a, (acc, x) -> acc + x)"), "list_reduce(a, (acc, x) -> (acc + x))");
    // 括弧付き単一引数も許す（duckdb と同じ）。
    assert_eq!(ex("list_filter(a, (x) -> x > 1)"), "list_filter(a, x -> (x > 1i32))");
    // ネストしたラムダ。
    assert_eq!(
        ex("list_transform(a, y -> list_transform(y, x -> x * 2))"),
        "list_transform(a, y -> list_transform(y, x -> (x * 2i32)))"
    );
    // list_filter も同じ扱い。
    assert_eq!(ex("list_filter(a, x -> x > 5)"), "list_filter(a, x -> (x > 5i32))");
    // 大文字小文字を区別しない関数名判定。
    assert_eq!(ex("LIST_TRANSFORM(a, x -> x)"), "LIST_TRANSFORM(a, x -> x)");

    // それ以外の関数の引数位置では `->` は今まで通り JSON パス演算子の
    // ままで、ラムダとしては解釈されない（duckdb CLI で実測済み:
    // `coalesce(doc -> 'a', 'x')` は JSON 抽出のまま解決される）。
    assert_eq!(ex("coalesce(doc -> 'a', x)"), "coalesce(json_extract(doc, 'a'), x)");
}

#[test]
fn try_cast_expr() {
    assert_eq!(ex("TRY_CAST(x AS INTEGER)"), "TRY_CAST(x AS INTEGER)");
    assert_eq!(ex("try_cast(x AS DECIMAL(10,2))"), "TRY_CAST(x AS DECIMAL(10,2))");
    // 通常の CAST とは別ノードとして区別される。
    assert_ne!(ex("CAST(x AS INTEGER)"), ex("TRY_CAST(x AS INTEGER)"));
    assert_eq!(code("SELECT TRY_CAST(x AS FROB)"), Code::InvalidCast as u16);
    // `try_cast` は予約語ではないので、通常の識別子（列名・関数名）としても
    // 使い続けられる（`(` が続かなければ普通の列参照）。
    assert_eq!(ex("try_cast"), "try_cast");
}

#[test]
fn iif_desugars_to_case() {
    assert_eq!(ex("IIF(a > 1, 'x', 'y')"), "CASE WHEN (a > 1i32) THEN 'x' ELSE 'y' END");
    assert_eq!(ex("iif(a, 1, 2)"), "CASE WHEN a THEN 1i32 ELSE 2i32 END");
    assert_eq!(code("SELECT IIF(a, 1)"), Code::UnexpectedToken as u16);
    // `iif` も予約語ではない。
    assert_eq!(ex("iif"), "iif");
}

#[test]
fn interval_literals() {
    // 複合文字列形式。パック結果をそのまま比較する。
    let lit =
        |m: i32, d: i32, u: i64| format!("INTERVAL({}i128)", crate::vector::pack_interval(m, d, u));
    assert_eq!(ex("INTERVAL '3 days'"), lit(0, 3, 0));
    assert_eq!(ex("INTERVAL '1 year 2 months 3 days'"), ex("INTERVAL '14 months 3 days'"),);
    // 単数形・複数形・大文字小文字を区別しない。
    assert_eq!(ex("interval '1 DAY'"), ex("INTERVAL '1 days'"));
    assert_eq!(ex("INTERVAL '1 month'"), ex("INTERVAL '1 months'"));
    // 負値（文字列内の符号）。
    assert_eq!(ex("INTERVAL '-3 days'"), lit(0, -3, 0));
    // 単項マイナスは `Unary::Neg` で包まれる（別ノード）ので、専用カーネルに
    // 展開されることは `plan::compile` 側のテストで確認する。ここではパース
    // が通ることだけ見る。
    assert_eq!(ex("-INTERVAL '3 days'"), format!("(- {})", lit(0, 3, 0)));
    // `'n' UNIT` 形式。
    assert_eq!(ex("INTERVAL '3' DAY"), ex("INTERVAL '3 days'"));
    assert_eq!(ex("INTERVAL '1' MONTH"), ex("INTERVAL '1 month'"));
    // 引用符無しの `n UNIT` 形式。
    assert_eq!(ex("INTERVAL 3 DAY"), ex("INTERVAL '3 days'"));
    // 秒未満の単位。
    assert_eq!(ex("INTERVAL '1500 milliseconds'"), ex("INTERVAL '1500000 microseconds'"));
    // `interval` は予約語ではないので列参照としても使える。
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
    assert_eq!(ex("count(*)"), "count(*)", "FILTER 無しは今までどおり");
    // `FILTER` は予約語ではないので、次が `(` でなければ普通の別名として通る。
    assert_eq!(sel("SELECT count(*) filter FROM t"), "SELECT count(*) AS filter FROM t");
    // `FILTER` はウィンドウ関数には未対応（範囲外）。
    assert_eq!(
        code("SELECT count(*) FILTER (WHERE a > 1) OVER () FROM t"),
        Code::UnsupportedFeature as u16
    );
}

#[test]
fn column_refs_and_params() {
    assert_eq!(ex("a"), "a");
    assert_eq!(ex("t.a"), "t.a");
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

// --- リテラル -----------------------------------------------------------

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
    // 単項マイナスは畳むが、式に対しては通常の演算子のまま。
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

// --- 文全体 -------------------------------------------------------------

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
    // WHERE/GROUP BY/HAVING を挟まず、テーブル参照の直後に `QUALIFY` が
    // 来ても、テーブル別名として食われずに正しく句として解釈される
    // （過去の ROWS/RANGE 事故と同じ罠。`QUALIFY` を完全予約語にした理由）。
    assert_eq!(sel("SELECT a FROM t QUALIFY a > 1"), "SELECT a FROM t QUALIFY (a > 1i32)");
    // GROUP BY / HAVING の後、ORDER BY の前という置き場所。
    assert_eq!(
            sel("SELECT a, count(*) FROM t GROUP BY a HAVING count(*) > 1 QUALIFY a > 0 ORDER BY a"),
            "SELECT a, count(*) FROM t GROUP BY a HAVING (count(*) > 1i32) QUALIFY (a > 0i32) ORDER BY a ASC NULLS LAST"
        );
    // 完全な予約語なので列名としては使えない。
    assert_eq!(code("SELECT qualify FROM t"), Code::UnexpectedToken as u16);
}

#[test]
fn star_exclude_replace() {
    // 基本形。順序は EXCLUDE → REPLACE 固定（`duckdb` で確認済み）。
    assert_eq!(sel("SELECT * EXCLUDE (b) FROM t"), "SELECT * EXCLUDE (b) FROM t");
    assert_eq!(
        sel("SELECT * REPLACE (a + 1 AS a) FROM t"),
        "SELECT * REPLACE ((a + 1i32) AS a) FROM t"
    );
    assert_eq!(
        sel("SELECT * EXCLUDE (b) REPLACE (a + 1 AS a) FROM t"),
        "SELECT * EXCLUDE (b) REPLACE ((a + 1i32) AS a) FROM t"
    );
    // 逆順（REPLACE の後に EXCLUDE）は `duckdb` と同じく構文エラー。
    assert_eq!(
        code("SELECT * REPLACE (a + 1 AS a) EXCLUDE (b) FROM t"),
        Code::UnexpectedToken as u16
    );
    // 1 個だけなら括弧を省略できる（`duckdb` の挙動）。
    assert_eq!(sel("SELECT * EXCLUDE b FROM t"), "SELECT * EXCLUDE (b) FROM t");
    assert_eq!(sel("SELECT * REPLACE 1 AS a FROM t"), "SELECT * REPLACE (1i32 AS a) FROM t");
    // 複数列は括弧必須。
    assert_eq!(sel("SELECT * EXCLUDE (a, b) FROM t"), "SELECT * EXCLUDE (a, b) FROM t");
    assert_eq!(
        sel("SELECT * REPLACE (1 AS a, 2 AS b) FROM t"),
        "SELECT * REPLACE (1i32 AS a, 2i32 AS b) FROM t"
    );
    // `t.*` にも同様に付けられる。
    assert_eq!(sel("SELECT t.* EXCLUDE (b) FROM t"), "SELECT t.* EXCLUDE (b) FROM t");
    // 同じ列を EXCLUDE と REPLACE の両方に書くのは無意味なので拒否する。
    assert_eq!(code("SELECT * EXCLUDE (a) REPLACE (1 AS a) FROM t"), Code::SyntaxError as u16);
    // EXCLUDE 内の重複、REPLACE 内の重複もそれぞれ拒否する。
    assert_eq!(code("SELECT * EXCLUDE (a, a) FROM t"), Code::SyntaxError as u16);
    assert_eq!(code("SELECT * REPLACE (1 AS a, 2 AS a) FROM t"), Code::SyntaxError as u16);
    // `EXCLUDE` は `*` の直後という文脈でしかキーワードにならない。過去の
    // ROWS/RANGE/QUALIFY 事故と同種の罠なので、通常の列名・別名として
    // なお使えることを確認する。`REPLACE` は `ddl` フィーチャが有効だと
    // `CREATE OR REPLACE` 用に別途グローバル予約語になるため、その場合は
    // 対象外（`is_star_replace_kw` のコメント参照）。
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
    // `GROUPING SETS` はそのまま集合の並びを保つ。空集合 `()` も 1 要素。
    assert_eq!(
        sel("SELECT a, b, sum(c) FROM t GROUP BY GROUPING SETS ((a, b), (a), ())"),
        "SELECT a, b, sum(c) FROM t GROUP BY GROUPING SETS ((a, b), (a), ())"
    );
    // `ROLLUP (a, b, c)` は列の多い方から少ない方への階層的な部分集合に展開される。
    assert_eq!(
        sel("SELECT a, b, c, sum(d) FROM t GROUP BY ROLLUP (a, b, c)"),
        "SELECT a, b, c, sum(d) FROM t GROUP BY GROUPING SETS ((a, b, c), (a, b), (a), ())"
    );
    // `CUBE (a, b)` は全部分集合（2^n 個）に展開される。
    assert_eq!(
        sel("SELECT a, b, sum(c) FROM t GROUP BY CUBE (a, b)"),
        "SELECT a, b, sum(c) FROM t GROUP BY GROUPING SETS ((a, b), (a), (b), ())"
    );
    // 単一列の ROLLUP/CUBE。
    assert_eq!(
        sel("SELECT a, sum(c) FROM t GROUP BY ROLLUP (a)"),
        "SELECT a, sum(c) FROM t GROUP BY GROUPING SETS ((a), ())"
    );

    // `GROUPING`/`SETS`/`ROLLUP`/`CUBE` は GROUP BY 直後という文脈でしか
    // キーワードにならない。過去の ROWS/RANGE/QUALIFY 事故と同種の罠なので、
    // 通常の列名・別名としてなお使えることを確認する
    // （`SETS` は他のどの文脈でも特別扱いしないので確認不要）。
    assert_eq!(sel("SELECT grouping FROM t"), "SELECT grouping FROM t");
    assert_eq!(sel("SELECT rollup, cube FROM t"), "SELECT rollup, cube FROM t");
    assert_eq!(sel("SELECT a AS rollup FROM t"), "SELECT a AS rollup FROM t");
    assert_eq!(
        sel("SELECT a FROM t WHERE grouping > 0"),
        "SELECT a FROM t WHERE (grouping > 0i32)"
    );

    // `GROUPING(...)` は普通の関数呼び出しとして通る。
    assert_eq!(ex("grouping(a)"), "grouping(a)");
    assert_eq!(ex("grouping(a, b)"), "grouping(a, b)");
}

#[test]
fn distinct_on_clause() {
    assert_eq!(sel("SELECT DISTINCT ON (a) a, b FROM t"), "SELECT DISTINCT ON (a) a, b FROM t");
    assert_eq!(sel("SELECT DISTINCT ON (a, b) * FROM t"), "SELECT DISTINCT ON (a, b) * FROM t");
    // `DISTINCT ON` は普通の `DISTINCT` とは排他（同時には立たない）。
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
    // カンマ結合は暗黙の CROSS JOIN。左深に積む。
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
    // 複数 CTE は定義順に並ぶ。
    assert_eq!(
        qs("WITH a AS (SELECT x FROM t), b AS (SELECT y FROM u) SELECT * FROM a, b"),
        "WITH a AS (SELECT x FROM t), b AS (SELECT y FROM u) \
             SELECT * FROM (a CROSS JOIN b)"
    );
    // 後ろの CTE が前の CTE を参照できる（前方参照は binder 側の責務）。
    assert_eq!(
        qs("WITH a AS (SELECT x FROM t), b AS (SELECT x FROM a) SELECT * FROM b"),
        "WITH a AS (SELECT x FROM t), b AS (SELECT x FROM a) SELECT * FROM b"
    );
    // CTE の中身も完全なクエリ。集合演算も ORDER BY も書ける。
    assert_eq!(
        qs("WITH a AS (SELECT 1 UNION SELECT 2) SELECT * FROM a"),
        "WITH a AS ((SELECT 1i32 UNION SELECT 2i32)) SELECT * FROM a"
    );
    assert_eq!(
        qs("WITH a AS (SELECT x FROM t ORDER BY x LIMIT 3) SELECT * FROM a"),
        "WITH a AS (SELECT x FROM t ORDER BY x ASC NULLS LAST LIMIT 3) SELECT * FROM a"
    );
    // 入れ子の CTE。
    assert_eq!(
        qs("WITH a AS (WITH b AS (SELECT 1) SELECT * FROM b) SELECT * FROM a"),
        "WITH a AS (WITH b AS (SELECT 1i32) SELECT * FROM b) SELECT * FROM a"
    );
    // EXPLAIN の下にも CTE を置ける。
    assert!(matches!(
        parse("EXPLAIN WITH a AS (SELECT 1) SELECT * FROM a").expect("parse").stmt,
        Stmt::Explain(_)
    ));
}

// --- WITH RECURSIVE -------------------------------------------------------

#[test]
fn with_recursive_parses() {
    // `RECURSIVE` は WITH 直後だけの文脈依存キーワード。実際に自分自身を
    // 参照するかどうかは束縛時の判定（`plan::bind`）なので、パーサは
    // フラグを立てるだけで本文の形は問わない。
    assert_eq!(
        qs("WITH RECURSIVE x AS (SELECT 1) SELECT 1"),
        "WITH RECURSIVE x AS (SELECT 1i32) SELECT 1i32"
    );
    // 列名リストは `WITH RECURSIVE` の下でだけ許す。
    assert_eq!(
        qs("WITH RECURSIVE fib(n, a, b) AS \
                 (SELECT 0, 0, 1 UNION ALL SELECT n+1, b, a+b FROM fib WHERE n < 10) \
                 SELECT * FROM fib"),
        "WITH RECURSIVE fib(n, a, b) AS \
             ((SELECT 0i32, 0i32, 1i32 UNION ALL SELECT (n + 1i32), b, (a + b) FROM fib \
             WHERE (n < 10i32))) SELECT * FROM fib"
    );
    // `RECURSIVE` はリスト全体に効く。一部の CTE が実際には非再帰でもよい。
    assert_eq!(
        qs("WITH RECURSIVE base AS (SELECT 1 AS x), \
                 t AS (SELECT x AS n FROM base UNION ALL SELECT n+1 FROM t WHERE n < 5) \
                 SELECT * FROM t"),
        "WITH RECURSIVE base AS (SELECT 1i32 AS x), \
             t AS ((SELECT x AS n FROM base UNION ALL SELECT (n + 1i32) FROM t \
             WHERE (n < 5i32))) SELECT * FROM t"
    );
    // `recursive` という名前の CTE も従来どおり書ける（`AS` が直後に
    // 続くので、キーワードとしては消費されない）。
    assert_eq!(
        qs("WITH recursive AS (SELECT 1) SELECT * FROM recursive"),
        "WITH recursive AS (SELECT 1i32) SELECT * FROM recursive"
    );
    // ただし列名リストは「`RECURSIVE` の後の名前」と区別が付かないため、
    // `WITH recursive(a) AS (...)` は列名リスト側の判定
    // （`RECURSIVE` 無しでは未対応）に落ちる。DuckDB は許すが、この
    // エンジンでは非再帰 CTE の列名リストをそもそもサポートしないので
    // 一貫してエラーになる。
    assert_eq!(
        code("WITH recursive(a) AS (SELECT 1) SELECT * FROM recursive"),
        Code::UnsupportedFeature as u16
    );
}

// --- 集合演算 -----------------------------------------------------------

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
    // 括弧付きの項。
    assert_eq!(qs("(SELECT 1) UNION (SELECT 2)"), "(SELECT 1i32 UNION SELECT 2i32)");
}

#[test]
fn set_operation_precedence() {
    // INTERSECT は UNION / EXCEPT より強く結合する。
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
    // UNION / EXCEPT は左結合。EXCEPT は結合的でないのでここが要点。
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
    // 括弧で結合の向きを変えられる。
    assert_eq!(
        qs("SELECT 1 EXCEPT (SELECT 2 EXCEPT SELECT 3)"),
        "(SELECT 1i32 EXCEPT (SELECT 2i32 EXCEPT SELECT 3i32))"
    );
    assert_eq!(
        qs("(SELECT 1 UNION SELECT 2) INTERSECT SELECT 3"),
        "((SELECT 1i32 UNION SELECT 2i32) INTERSECT SELECT 3i32)"
    );

    // 木の形を直接確認する（描画の括弧だけに頼らない）。
    let (_, q) = parsed("SELECT 1 UNION SELECT 2 INTERSECT SELECT 3");
    match &q.body {
        SetExpr::SetOp { op, left, right, .. } => {
            assert_eq!(*op, SetOp::Union);
            assert!(matches!(**left, SetExpr::Select(_)));
            assert!(matches!(**right, SetExpr::SetOp { op: SetOp::Intersect, .. }));
        }
        _ => panic!("集合演算ではない"),
    }
    let (_, q) = parsed("SELECT 1 EXCEPT SELECT 2 EXCEPT SELECT 3");
    match &q.body {
        SetExpr::SetOp { op, left, right, .. } => {
            assert_eq!(*op, SetOp::Except);
            assert!(matches!(**left, SetExpr::SetOp { op: SetOp::Except, .. }));
            assert!(matches!(**right, SetExpr::Select(_)));
        }
        _ => panic!("集合演算ではない"),
    }
}

#[test]
fn trailing_clauses_placement() {
    // 集合演算の後ろの ORDER BY / LIMIT / OFFSET は外側の QueryStmt に付く。
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
                    _ => panic!("SELECT ではない"),
                }
            }
        }
        _ => panic!("集合演算ではない"),
    }
    assert_eq!(
        qs("SELECT a FROM t UNION SELECT b FROM u ORDER BY 1 LIMIT 5 OFFSET 2"),
        "(SELECT a FROM t UNION SELECT b FROM u) ORDER BY 1i32 ASC NULLS LAST LIMIT 5 OFFSET 2"
    );

    // 括弧無しの単一 SELECT なら SelectStmt 側に付く（binder の既存経路）。
    let (_, q) = parsed("SELECT a FROM t ORDER BY a LIMIT 5 OFFSET 1");
    assert!(q.order_by.is_empty());
    assert_eq!(q.limit, None);
    assert_eq!(q.offset, None);
    assert_eq!(plain(&q).order_by.len(), 1);
    assert_eq!(plain(&q).limit, Some(5));
    assert_eq!(plain(&q).offset, Some(1));

    // 括弧付きの項が自分の ORDER BY を持っていても外側に潰されない。
    assert_eq!(
        qs("(SELECT a FROM t ORDER BY a LIMIT 1) LIMIT 9"),
        "SELECT a FROM t ORDER BY a ASC NULLS LAST LIMIT 1 LIMIT 9"
    );
    // 集合演算の項ごとの LIMIT は括弧で表す。
    assert_eq!(
        qs("(SELECT a FROM t LIMIT 1) UNION SELECT b FROM u"),
        "(SELECT a FROM t LIMIT 1 UNION SELECT b FROM u)"
    );
    // CTE や外側 LIMIT を持つ括弧付きクエリは派生表に包んで項にする。
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

// --- ウィンドウ関数 -----------------------------------------------------

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
    // 通常の関数呼び出しと同じ場所に書ける。
    assert_eq!(
        sel("SELECT sum(x) OVER (PARTITION BY a) AS s FROM t"),
        "SELECT sum(x) OVER (PARTITION BY a WHOLE) AS s FROM t"
    );

    // `star` と既定枠を AST 上で直接確認する。
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
        _ => panic!("ウィンドウ関数ではない"),
    }
    match a.get(plain(&q).items[1].expr) {
        Expr::Window { star, frame, order_by, .. } => {
            assert!(!*star);
            assert_eq!(order_by.len(), 1);
            assert_eq!(*frame, WindowFrame::RangeUnboundedPreceding);
        }
        _ => panic!("ウィンドウ関数ではない"),
    }
}

#[test]
fn window_keywords_are_contextual() {
    // 列名はデータファイル由来で利用者が選べない。ウィンドウ構文のために
    // ありふれた語を予約語にすると、引用符無しでは読めない列ができる。
    for name in ["rows", "range", "over", "partition"] {
        assert_eq!(sel(&format!("SELECT {} FROM t", name)), format!("SELECT {} FROM t", name));
        assert_eq!(sel(&format!("SELECT t.{} FROM t", name)), format!("SELECT t.{} FROM t", name));
        assert_eq!(
            sel(&format!("SELECT * FROM t WHERE {} > 1", name)),
            format!("SELECT * FROM t WHERE ({} > 1i32)", name)
        );
        // 表別名としても列別名としても使える。
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
        // 大文字でも同じ（識別子の綴りはそのまま保つ）。
        let upper = name.to_ascii_uppercase();
        assert_eq!(sel(&format!("SELECT {} FROM t", upper)), format!("SELECT {} FROM t", upper));
        // GROUP BY / ORDER BY の中でも列として読める。
        assert_eq!(
            sel(&format!("SELECT {0} FROM t GROUP BY {0} ORDER BY {0}", name)),
            format!("SELECT {0} FROM t GROUP BY {0} ORDER BY {0} ASC NULLS LAST", name)
        );
    }

    // `over` の直後が `(` でなければウィンドウ句ではなく別名。
    assert_eq!(sel("SELECT count(*) over FROM t"), "SELECT count(*) AS over FROM t");
    assert_eq!(sel("SELECT count(*) over"), "SELECT count(*) AS over");
    assert_eq!(sel("SELECT count(*) AS over FROM t"), "SELECT count(*) AS over FROM t");
    // 関数呼び出しでない `over` の直後の `(` は別名 + 構文エラーのまま。
    assert_eq!(code("SELECT a over (b)"), Code::UnexpectedToken as u16);

    // 文脈依存にしてもウィンドウ指定は今までどおり効く。
    assert_eq!(ex("count(*) OVER ()"), "count(*) OVER (WHOLE)");
    assert_eq!(
        ex("sum(rows) OVER (PARTITION BY partition ORDER BY range)"),
        "sum(rows) OVER (PARTITION BY partition ORDER BY range ASC NULLS LAST RANGE)"
    );
    assert_eq!(
        sel("SELECT rank() OVER (PARTITION BY over) FROM t"),
        "SELECT rank() OVER (PARTITION BY over WHOLE) FROM t"
    );
    // 枠指定の拒否は予約語ではなく綴りで判定する。
    assert_eq!(
        code("SELECT sum(x) OVER (ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)"),
        Code::UnsupportedFeature as u16
    );
    assert_eq!(
        code("SELECT sum(x) OVER (rows BETWEEN 1 PRECEDING AND CURRENT ROW)"),
        Code::UnsupportedFeature as u16
    );
    // 引用符付き識別子は文脈依存キーワードとして照合しない。
    assert_eq!(sel("SELECT \"rows\" FROM t"), "SELECT rows FROM t");
    assert_eq!(code("SELECT sum(x) OVER (\"rows\")"), Code::UnexpectedToken as u16);
}

#[test]
fn window_rejections() {
    // 明示的な枠指定は黙って無視すると結果が変わるので必ず弾く。
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
    // DISTINCT 付きのウィンドウ集約も範囲外。
    assert_eq!(code("SELECT count(DISTINCT x) OVER ()"), Code::UnsupportedFeature as u16);
    // `OVER w`（名前付き参照）は構文としては通る。定義の有無は束縛時にしか
    // 分からない（`plan::bind` 側のテスト参照）。
    assert_eq!(code("SELECT sum(x) OVER w"), 0);
    assert_eq!(code("SELECT sum(x) OVER (PARTITION a)"), Code::UnexpectedToken as u16);
}

#[test]
fn named_window_ref_parses() {
    // `OVER w`（識別子 1 つだけ）は名前付きウィンドウの参照として木に残す。
    // 実体（PARTITION BY/ORDER BY）は `WINDOW` 句側にあり、ここでは
    // パーサが名前だけを持たせていることを確認する。
    assert_eq!(ex("sum(x) OVER w"), "sum(x) OVER w");
    let (a, q) = parsed("SELECT sum(x) OVER w FROM t");
    match a.get(plain(&q).items[0].expr) {
        Expr::Window { name, window_ref, partition_by, order_by, .. } => {
            assert_eq!(name, "sum");
            assert_eq!(window_ref.as_deref(), Some("w"));
            assert!(partition_by.is_empty());
            assert!(order_by.is_empty());
        }
        _ => panic!("ウィンドウ関数ではない"),
    }
    // `OVER` の直後が `(` でも識別子でもなければ、今までどおり別名扱い。
    assert_eq!(sel("SELECT count(*) over FROM t"), "SELECT count(*) AS over FROM t");
    // ただし `OVER` の直後に識別子が続けば、コンマを挟まない限り名前付き
    // ウィンドウ参照としてしか解釈できない（別名 + 別の select item は
    // コンマが要るので、そちらの解釈はそもそも成り立たない）。
    assert_eq!(sel("SELECT sum(x) over w, y FROM t"), "SELECT sum(x) OVER w, y FROM t");
}

#[test]
fn window_clause_named_definitions() {
    // 単純な名前付きウィンドウ。複数の関数から同じ定義を共有できる。
    assert_eq!(
            sel(
                "SELECT id, sum(x) OVER w, avg(x) OVER w FROM t WINDOW w AS (PARTITION BY id ORDER BY ts)"
            ),
            "SELECT id, sum(x) OVER w, avg(x) OVER w FROM t WINDOW w AS (PARTITION BY id ORDER BY ts ASC NULLS LAST)"
        );
    // カンマ区切りで複数定義できる。
    assert_eq!(
            sel(
                "SELECT sum(x) OVER w1, rank() OVER w2 FROM t WINDOW w1 AS (PARTITION BY id), w2 AS (ORDER BY x)"
            ),
            "SELECT sum(x) OVER w1, rank() OVER w2 FROM t WINDOW w1 AS (PARTITION BY id), w2 AS (ORDER BY x ASC NULLS LAST)"
        );
    // 通常の `OVER (...)` 直書きと併用できる。
    assert_eq!(
        sel("SELECT sum(x) OVER w, count(*) OVER () FROM t WINDOW w AS (PARTITION BY id)"),
        "SELECT sum(x) OVER w, count(*) OVER (WHOLE) FROM t WINDOW w AS (PARTITION BY id)"
    );
    // `WINDOW` は `GROUP BY`/`HAVING` の後、`QUALIFY`/`ORDER BY` の前。
    assert_eq!(
            sel(
                "SELECT a, sum(x) OVER w FROM t WHERE a > 0 GROUP BY a, x HAVING x > 0 WINDOW w AS (ORDER BY a) QUALIFY sum(x) OVER w > 0 ORDER BY a"
            ),
            "SELECT a, sum(x) OVER w FROM t WHERE (a > 0i32) GROUP BY a, x HAVING (x > 0i32) WINDOW w AS (ORDER BY a ASC NULLS LAST) QUALIFY (sum(x) OVER w > 0i32) ORDER BY a ASC NULLS LAST"
        );
    // 同じ名前を 2 回定義するのはエラー（`duckdb` も "already defined" として拒否）。
    assert_eq!(
        code("SELECT a FROM t WINDOW w AS (ORDER BY a), w AS (ORDER BY a)"),
        Code::SyntaxError as u16
    );
    // `WINDOW` は完全な予約語（`QUALIFY` と同じ判断）なので列名としては使えない。
    assert_eq!(code("SELECT window FROM t"), Code::UnexpectedToken as u16);
    // 引用符を付ければ引き続き列名として使える。
    assert_eq!(sel("SELECT \"window\" FROM t"), "SELECT window FROM t");
}

// --- サブクエリ式 -------------------------------------------------------

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
    // 括弧を重ねても意味は同じ。内側の `(` が改めて判定を受ける。
    assert_eq!(ex("((SELECT 1))"), "(SELECT 1i32)");
    // 括弧付き式は今までどおり。
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
    // 値リストは今までどおり値リストのまま。
    assert_eq!(ex("x IN (1, 2, 3)"), "(x IN [1i32, 2i32, 3i32])");
    assert_eq!(ex("x IN ((1), (2))"), "(x IN [1i32, 2i32])");
    // `IN ((SELECT ...))` は値リスト側に倒す。要素がスカラサブクエリになる。
    assert_eq!(ex("x IN ((SELECT 1))"), "(x IN [(SELECT 1i32)])");
    assert_eq!(code("SELECT x IN ()"), Code::UnexpectedToken as u16);
}

#[test]
fn quantified_comparison_subquery() {
    // `ANY`/`SOME` は同じ意味（`all: false`）でパースされる。
    assert_eq!(ex("x = ANY (SELECT a FROM t)"), "(x = ANY (SELECT a FROM t))");
    assert_eq!(ex("x = SOME (SELECT a FROM t)"), "(x = ANY (SELECT a FROM t))");
    assert_eq!(ex("x <> ALL (SELECT a FROM t)"), "(x != ALL (SELECT a FROM t))");
    assert_eq!(ex("x > ANY (SELECT a FROM t)"), "(x > ANY (SELECT a FROM t))");
    assert_eq!(ex("x >= ALL (SELECT a FROM t)"), "(x >= ALL (SELECT a FROM t))");
    assert_eq!(ex("x < SOME (SELECT a FROM t)"), "(x < ANY (SELECT a FROM t))");
    assert_eq!(ex("x <= ALL (SELECT a FROM t)"), "(x <= ALL (SELECT a FROM t))");
    // 述語と同じ強さ（AND より強く結合する）。
    assert_eq!(
        ex("a = 1 AND x > ANY (SELECT a FROM t)"),
        "((a = 1i32) AND (x > ANY (SELECT a FROM t)))"
    );
    // `any`/`some` は予約語ではないので、`(` を伴わなければ普通の列名。
    assert_eq!(ex("x > any"), "(x > any)");
    assert_eq!(ex("x > some"), "(x > some)");
    // `ALL` は既存の予約語のままで、UNION ALL 等と衝突しない。
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

// --- エラー -------------------------------------------------------------

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
    // `ddl`/`dml` フィーチャが有効だと INSERT/UPDATE/CREATE TABLE は正当な
    // 文になる。フィーチャが無効な既定ビルドでの挙動はここで確認し、
    // 有効時の挙動は `sql/parser.rs` 末尾の `ddl_dml` テストで確認する。
    #[cfg(not(feature = "dml"))]
    {
        assert_eq!(code("INSERT INTO t VALUES (1)"), Code::UnsupportedFeature as u16);
        assert_eq!(code("UPDATE t SET a = 1"), Code::UnsupportedFeature as u16);
    }
    #[cfg(not(feature = "ddl"))]
    {
        assert_eq!(code("CREATE TABLE t (a INT)"), Code::UnsupportedFeature as u16);
        // ALTER TABLE も `ddl` が無効なビルドでは引き続き未対応。有効時の
        // 挙動は `sql/parser.rs` 末尾の DDL/DML テスト群で確認する。
        assert_eq!(code("ALTER TABLE t ADD COLUMN x INT"), Code::UnsupportedFeature as u16);
    }
    assert_eq!(code("WITH x AS SELECT 1 SELECT * FROM x"), Code::UnexpectedToken as u16);
    // 列名リストは非再帰 CTE では引き続き未対応（`WITH RECURSIVE` の下でだけ
    // 許す。`recursive_cte` テスト群参照）。
    assert_eq!(code("WITH x (a) AS (SELECT 1) SELECT 1"), Code::UnsupportedFeature as u16);
    assert_eq!(code("SELECT 1 UNION"), Code::UnexpectedToken as u16);
    assert_eq!(code("SELECT 1 INTERSECT 2"), Code::UnexpectedToken as u16);
    // `a & b` used to be a syntax error before `&` became the bitwise-AND
    // operator (see `bitwise_operators_desugar_to_bit_functions` below).
    assert_eq!(code("SELECT a &"), Code::UnexpectedToken as u16);
}

#[test]
fn error_positions() {
    // 位置は必ず「問題のトークンの先頭バイト」を指す。
    assert_eq!(err_at("SELECT FROM t"), (Code::UnexpectedToken as u16, 7));
    assert_eq!(err_at("SELECT 'abc"), (Code::UnterminatedString as u16, 7));
    #[cfg(not(feature = "dml"))]
    assert_eq!(err_at("INSERT INTO t VALUES (1)"), (Code::UnsupportedFeature as u16, 0));
    #[cfg(not(feature = "ddl"))]
    assert_eq!(err_at("ALTER TABLE t ADD COLUMN x INT"), (Code::UnsupportedFeature as u16, 0));
    assert_eq!(err_at("SELECT a FROM t WHERE b @ 1"), (Code::UnexpectedToken as u16, 24));
    assert_eq!(err_at("SELECT CAST(x AS FROB)"), (Code::InvalidCast as u16, 17));
    // 新しい構文でも位置は問題のトークンの先頭を指す。
    assert_eq!(err_at("SELECT 1 UNION 2"), (Code::UnexpectedToken as u16, 15));
    assert_eq!(err_at("SELECT sum(x) OVER (ROWS)"), (Code::UnsupportedFeature as u16, 20));
    // 列名リストが要求する `(` の位置。
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

    // 深いサブクエリとカンマ結合も同じ上限で止まる。
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

    // 上限直下は通る。
    let ok = format!("SELECT {}1{}", "(".repeat(50), ")".repeat(50));
    assert_eq!(code(&ok), 0);
}

#[test]
fn deep_subquery_expressions_are_rejected() {
    // スカラサブクエリの入れ子。木の破棄も再帰するので、上限は
    // 「パースを通った木を落としてもスタックが持つ」水準でなければならない。
    let n = 200;
    let mut s = String::from("SELECT ");
    s.push_str(&"(SELECT ".repeat(n));
    s.push('1');
    s.push_str(&")".repeat(n));
    assert_eq!(code(&s), Code::ExpressionTooDeep as u16);

    // EXISTS の入れ子。
    let mut e = String::from("SELECT 1 WHERE ");
    e.push_str(&"EXISTS (SELECT 1 WHERE ".repeat(n));
    e.push_str("TRUE");
    e.push_str(&")".repeat(n));
    assert_eq!(code(&e), Code::ExpressionTooDeep as u16);

    // IN (SELECT ...) の入れ子。
    let mut i = String::from("SELECT 1 WHERE ");
    i.push_str(&"1 IN (SELECT 1 WHERE ".repeat(n));
    i.push_str("TRUE");
    i.push_str(&")".repeat(n));
    assert_eq!(code(&i), Code::ExpressionTooDeep as u16);

    // 派生表の入れ子（括弧付きクエリ経由）。
    let mut d = String::from("SELECT * FROM ");
    d.push_str(&"(SELECT * FROM ".repeat(n));
    d.push('t');
    d.push_str(&")".repeat(n));
    assert_eq!(code(&d), Code::ExpressionTooDeep as u16);

    // CTE の入れ子。
    let mut c = String::new();
    c.push_str(&"WITH a AS (".repeat(n));
    c.push_str("SELECT 1");
    for _ in 0..n {
        c.push_str(") SELECT * FROM a");
    }
    assert_eq!(code(&c), Code::ExpressionTooDeep as u16);

    // 上限直下のスカラサブクエリは通る（1 段あたり深さ 2 を消費する）。
    let k = 30;
    let mut ok = String::from("SELECT ");
    ok.push_str(&"(SELECT ".repeat(k));
    ok.push('1');
    ok.push_str(&")".repeat(k));
    assert_eq!(code(&ok), 0);
}

#[test]
fn long_setop_chains_are_rejected() {
    // 集合演算も左深の `Box` 連鎖。破棄時の再帰を上限で止める。
    let mut u = String::from("SELECT 1");
    u.push_str(&" UNION SELECT 1".repeat(200));
    assert_eq!(code(&u), Code::ExpressionTooDeep as u16);

    let mut x = String::from("SELECT 1");
    x.push_str(&" INTERSECT SELECT 1".repeat(200));
    assert_eq!(code(&x), Code::ExpressionTooDeep as u16);

    // 上限直下は通る。
    let mut ok = String::from("SELECT 1");
    ok.push_str(&" UNION SELECT 1".repeat(60));
    assert_eq!(code(&ok), 0);
}

// --- DDL/DML（`ddl`/`dml` フィーチャ） ------------------------------------

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

    // `COLUMN` は省略できる（DuckDB と同じ、CLI で確認済み）。
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
    // `COLUMN` は省略できる。
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

    // `COLUMN` は省略できる。
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

    // FORMAT の値は大文字小文字を問わない。
    let p = parse("COPY (SELECT a FROM t) TO 'out.bin' (FORMAT JSON)").expect("parse");
    assert!(matches!(p.stmt, Stmt::Copy { format: Some(f), .. } if f.eq_ignore_ascii_case("json")));
}

/// `COPY <table> TO ...` は `SELECT * FROM <table>` と等価な木になる。
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

/// `format` は `COPY` の外では普通の列名として使える（`export` はこの語を
/// グローバルには予約しない — 文脈依存キーワードなので）。
#[cfg(feature = "export")]
#[test]
fn format_remains_usable_as_a_column_name_outside_copy() {
    assert_eq!(sel("SELECT format FROM t"), "SELECT format FROM t");
}

/// `to` も `export` 単体では同じく普通の列名として使える。`ddl` が同時に
/// 有効だと `ALTER TABLE ... RENAME TO` 用に別途グローバル予約される
/// （`sql/lexer.rs` の `DDL_KEYWORDS`）ため、この確認は `ddl` が無効な
/// 構成でのみ意味を持つ（`copy_stmt`/`expect_to` のドキュメント参照）。
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

// --- ファイルテーブル関数 / 裸のパスリテラル (`FromItem::File`) -----------
//
// 実際の解決（`catalog.index_of` によるパス完全一致の名前引き）は
// `plan::bind::flatten_from` の担当なので、ここではパース結果の AST の形
// （`FromItem::File` の `path`/`format`/`alias`）とラウンドトリップ文字列
// だけを確認する。挙動は `duckdb` CLI で確認済み（doc コメント参照）。

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

    // 未知の拡張子は既存の `FormatKind::detect` の既定どおり Parquet 扱い。
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
    // `AS` 無しの裸別名も通常どおり許す（`opt_alias` を共有しているため）。
    let (_, f) = from_item("SELECT * FROM 'data.parquet' p");
    assert!(matches!(&f, FromItem::File { alias: Some(a), .. } if a == "p"));
}

#[test]
fn read_parquet_is_an_alias_for_parquet() {
    let (a, f) = from_item("SELECT * FROM read_parquet('data.parquet')");
    assert!(matches!(&f, FromItem::File { path, format, .. }
            if path == "data.parquet" && *format == FormatKind::Parquet));
    // 両方とも同じ形に落ちるので、ラウンドトリップの見た目は `parquet(...)`
    // へ正規化される（`from_str` の doc 参照）。
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

/// `csv` フィーチャが無効なビルドでは `read_csv`/`read_csv_auto` という
/// 綴りそのものを認識しない（構文レベルで falls through して
/// `UnsupportedFeature` になる — `ddl`/`dml`/`export` の文が feature 無効時
/// に同じ経路へ落ちるのと同じパターン。`base_rel` の doc コメント参照）。
/// Parquet は常に使えるので影響しない。
#[cfg(not(feature = "csv"))]
#[test]
fn read_csv_is_unsupported_without_the_csv_feature() {
    assert_eq!(code("SELECT * FROM read_csv('a.csv')"), Code::UnsupportedFeature as u16);
    assert_eq!(code("SELECT * FROM read_csv_auto('a.csv')"), Code::UnsupportedFeature as u16);
    // 拡張子検出だけの裸リテラルはパース自体は通る（実解決は
    // `plan::bind` 側でのカタログ引きに委ねる）。
    assert!(parse("SELECT * FROM 'a.csv'").is_ok());
    assert!(parse("SELECT * FROM parquet('a.parquet')").is_ok());
}

#[cfg(not(feature = "jsonl"))]
#[test]
fn read_json_is_unsupported_without_the_jsonl_feature() {
    assert_eq!(code("SELECT * FROM read_json('a.json')"), Code::UnsupportedFeature as u16);
    assert_eq!(code("SELECT * FROM read_json_auto('a.json')"), Code::UnsupportedFeature as u16);
}

/// 名前付きオプション引数・複数ファイル引数は v1 の範囲外
/// （`FromItem::File` の doc コメント参照）。1 引数以外はすべて構文エラー。
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

/// 未知の関数名は他のテーブル関数と同じく `UnsupportedFeature`。
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
fn factorial_applies_after_bitwise_not_too() {
    // duckdb: SELECT ~5! -> 1 (confirmed: same as `(~5)!`, matching
    // `(~5)! = factorial(-6) = 1`; `~(5!)` is a different value, `-121`).
    // Prefix `~` reads its operand at `BP_UNARY` exactly like unary `-`
    // does, so the same `BP_BANG < BP_UNARY` rule applies uniformly.
    assert_eq!(ex("~5!"), "factorial(bit_not(5i32))");
}

#[test]
fn bang_still_lexes_as_the_prefix_of_longer_operators() {
    // `!=`/`!~`/`!~~`/`!~~*` must still win over a bare `Bang`.
    assert_eq!(code("SELECT 4 != 5 FROM t"), 0);
    assert_eq!(code("SELECT a !~ 'x' FROM t"), 0);
    assert_eq!(code("SELECT a !~~ 'x' FROM t"), 0);
    assert_eq!(code("SELECT a !~~* 'x' FROM t"), 0);
}
