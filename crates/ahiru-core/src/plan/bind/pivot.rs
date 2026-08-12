//! `PIVOT`/`UNPIVOT` desugaring into plain `SELECT` statements. This runs as
//! an AST-level rewrite before binding proper (see the module doc below for
//! why), so it lives alongside the rest of the binder but does not call into
//! `bind_select_in` at all.

use super::refs::each_child;
use super::*;
use crate::sql::ast::{PivotStmt, SelectItem, UnpivotStmt};

// --- PIVOT/UNPIVOT -----------------------------------------------------------
//
// Both are implemented as "plan-level syntactic sugar expansion" (the same idea as GROUPING
// SETS bundling several `Node::Aggregate` with `Node::SetOp`). Unlike GROUPING SETS, though,
// the expanded form lands on exactly the same AST shape as the existing "`GROUP BY` +
// aggregate + `FILTER`" (PIVOT) or "projection + `UNION ALL`" (UNPIVOT), so binding
// (`bind_select_in`) itself is left completely untouched and everything is done by AST-level
// rewriting. The caller is `Session::prepare` (invoked right after detecting a PIVOT/UNPIVOT
// `Stmt`, once the target table's schema has been resolved). The expansion is passed on to
// the existing `prepare_query` as an ordinary `Stmt::Select`.
//
// At bind time the arena is an immutable reference (`&ExprArena`), so new expression nodes
// (the `on = value` equality, or a `FILTER`-bearing aggregate call) cannot be created during
// binding. So the expansion happens right after parsing and before binding, pushing new
// nodes onto a `&mut ExprArena` just as `substitute_now` (`sql::now`) does.

/// The set of literal types one PIVOT value given explicitly in `IN (...)` may take.
/// Only strings, integers, and booleans are supported (floating point is unsupported, since
/// `core::fmt` cannot be used to stringify a column name).
fn pivot_value_to_column_name(v: &Value) -> Result<String> {
    match v {
        Value::Bytes(b) => match core::str::from_utf8(b) {
            Ok(s) => Ok(String::from(s)),
            Err(_) => err!(UnsupportedFeature),
        },
        Value::Bool(x) => Ok(String::from(if *x { "true" } else { "false" })),
        Value::I32(n) => Ok(i128_to_decimal_string(*n as i128)),
        Value::I64(n) => Ok(i128_to_decimal_string(*n as i128)),
        Value::I128(n) => Ok(i128_to_decimal_string(*n)),
        Value::Null | Value::F64(_) => err!(UnsupportedFeature),
    }
}

/// Signed decimal stringification without `core::fmt`. The `i128` version of `push_u32`.
fn i128_to_decimal_string(v: i128) -> String {
    if v == 0 {
        return String::from("0");
    }
    let neg = v < 0;
    let mut uv = v.unsigned_abs();
    let mut buf = [0u8; 40];
    let mut n = 0;
    while uv > 0 {
        buf[n] = b'0' + (uv % 10) as u8;
        uv /= 10;
        n += 1;
    }
    let mut s = String::new();
    if neg {
        s.push('-');
    }
    for i in (0..n).rev() {
        s.push(buf[i] as char);
    }
    s
}

/// Collects every bare column name expression `id` references (recursively). Used to decide
/// the default columns when GROUP BY is omitted (all columns other than those `on`/`using`
/// reference). `each_child` does not walk into subqueries or windows in the first place (see
/// the module docs), so column references inside them are excluded -- a subquery or window
/// function in PIVOT's `ON`/`USING` is not anticipated, so this does no harm.
fn collect_colref_names(
    arena: &ExprArena,
    id: ExprId,
    out: &mut Vec<String>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    if let Expr::ColumnRef { name, .. } = arena.get(id) {
        out.push(name.clone());
        return Ok(());
    }
    let d = depth + 1;
    each_child(arena, id, &mut |c| collect_colref_names(arena, c, out, d))
}

/// Expands `PIVOT` into an ordinary `SELECT ... GROUP BY ...`.
///
/// `from_schema` is the column list of the target table (`stmt.from`). It is used only when
/// `stmt.group_by` is empty (everything except the columns `on`/`using` reference becomes the
/// default `GROUP BY` target -- the same rule as DuckDB). No real data is read.
pub fn desugar_pivot(
    arena: &mut ExprArena,
    stmt: PivotStmt,
    from_schema: &[Field],
) -> Result<QueryStmt> {
    let PivotStmt { from, on, in_list, using, group_by, order_by, limit, offset } = stmt;

    // Automatic value discovery (omitting `IN`) needs a DISTINCT over the target column's real
    // data. At bind time only the schema has been read (`Session::prepare` calls this function
    // right after schema resolution), so it is unsupported (see the module docs and the
    // `PivotStmt` docs).
    let in_list = match in_list {
        Some(v) => v,
        None => err!(UnsupportedFeature),
    };
    ensure!(!in_list.is_empty(), SyntaxError);
    ensure!(in_list.len() <= MAX_PIVOT_VALUES, ExpressionTooDeep);

    // With `USING` omitted, the default is `count(*)`, as in DuckDB.
    let using = if using.is_empty() {
        let f = arena.push(Expr::Function {
            name: String::from("count"),
            args: Vec::new(),
            distinct: false,
            star: true,
            filter: None,
        });
        vec![SelectItem { expr: f, alias: None }]
    } else {
        using
    };
    // Several aggregates (`USING sum(a), avg(b)`) would need expression stringification to
    // determine column names (the `a_sum(a)` scheme), which is not worth the cost under the ban
    // on `core::fmt`, so they are unsupported (see the `PivotStmt::using` docs).
    ensure!(using.len() == 1, UnsupportedFeature);
    let (agg_name, agg_args, agg_distinct, agg_star) = match arena.get(using[0].expr) {
        Expr::Function { name, args, distinct, star, filter } => {
            ensure!(filter.is_none(), UnsupportedFeature);
            (name.clone(), args.clone(), *distinct, *star)
        }
        _ => err!(SyntaxError),
    };

    // The GROUP BY target columns. Without an explicit list, "every column other than those on/using reference".
    let group_by_exprs: Vec<ExprId> = if !group_by.is_empty() {
        group_by
    } else {
        let mut excluded = Vec::new();
        collect_colref_names(arena, on, &mut excluded, 0)?;
        for a in &agg_args {
            collect_colref_names(arena, *a, &mut excluded, 0)?;
        }
        from_schema
            .iter()
            .filter(|f| !excluded.iter().any(|e| eq_ascii_ci(e.as_bytes(), f.name.as_bytes())))
            .map(|f| arena.push(Expr::ColumnRef { qualifier: None, name: f.name.clone() }))
            .collect()
    };

    let mut items: Vec<SelectItem> =
        group_by_exprs.iter().map(|&expr| SelectItem { expr, alias: None }).collect();

    // One column per value, as `agg(...) FILTER (WHERE on = value)`. It rides the existing
    // `FILTER (WHERE cond)`-bearing aggregate machinery (`Expr::Function.filter`,
    // `exec::agg::Agg.filter`) directly, so the aggregate binding logic is untouched.
    //
    // When the same value appears twice or more in `IN (...)` (with or without an alias),
    // `duckdb` rejects it as "The value ... was specified multiple times in the IN clause"
    // (confirmed with `duckdb -c "PIVOT ... ON category IN ('a','a') ..."`). Without a check
    // here, two columns with the same `FILTER` condition would be produced under duplicate
    // names (a known defect. The same judgment as the EXCLUDE/REPLACE duplicate check in
    // `star_exclude_replace.rs` and the duplicate-name check on the `WINDOW` clause: only
    // value-based duplication is refused -- when aliases collide `duckdb` merely auto-renames
    // with a `_1` suffix rather than erroring, so that is not pursued).
    let mut seen_values: Vec<Value> = Vec::with_capacity(in_list.len());
    for (val_expr, alias) in &in_list {
        let lit = match arena.get(*val_expr) {
            Expr::Literal(v) => v.clone(),
            // `TypedLiteral` and expressions in general are unsupported, since both a column name and constant folding would be needed.
            _ => err!(UnsupportedFeature),
        };
        ensure!(!seen_values.contains(&lit), SyntaxError);
        seen_values.push(lit.clone());
        let col_name = match alias {
            Some(a) => a.clone(),
            None => pivot_value_to_column_name(&lit)?,
        };
        let lit_expr = arena.push(Expr::Literal(lit));
        let pred = arena.push(Expr::Binary { op: BinaryOp::Eq, lhs: on, rhs: lit_expr });
        let f = arena.push(Expr::Function {
            name: agg_name.clone(),
            args: agg_args.clone(),
            distinct: agg_distinct,
            star: agg_star,
            filter: Some(pred),
        });
        items.push(SelectItem { expr: f, alias: Some(col_name) });
    }

    Ok(QueryStmt {
        ctes: Vec::new(),
        body: SetExpr::Select(Box::new(SelectStmt {
            items,
            from: Some(from),
            group_by: group_by_exprs,
            ..SelectStmt::empty()
        })),
        // It rides `bind_query_in`'s "outer ORDER BY/LIMIT" path (which supports column names
        // and ordinals only) directly. The expanded body is a plain single SELECT, but putting
        // it here changes nothing (the `UNPIVOT` side becomes a `SetOp`, so the placement is
        // kept consistent).
        order_by,
        // `ORDER BY ALL` is already rejected by the parser for `PIVOT`/`UNPIVOT`
        // (see `sql::parser::Parser::pivot_stmt`).
        order_by_all: None,
        limit,
        offset,
    })
}

/// The cap on how many values may be written in `PIVOT`'s `IN (...)`. It plays the same
/// "safety valve against overproduction" role as `MAX_CUBE_COLS` (`sql::parser`) for `CUBE`/`ROLLUP`.
const MAX_PIVOT_VALUES: usize = 128;

/// The cap on how many target columns `UNPIVOT` may fold at once.
const MAX_UNPIVOT_COLUMNS: usize = 128;

/// Clones a `FromItem`. PIVOT/UNPIVOT's `from` supports only `Table`/`File` (`UNPIVOT` needs
/// to duplicate the same `from` once per target column and hand one to each branch of the
/// `UNION ALL`, and `Subquery`/`Join` cannot be copied simply because a plan (`Node`) cannot
/// be cloned. The same constraint `plan::bind::resolve_from` imposes for `DESCRIBE`).
fn clone_from_item(f: &FromItem) -> Result<FromItem> {
    match f {
        FromItem::Table { name, alias } => {
            Ok(FromItem::Table { name: name.clone(), alias: alias.clone() })
        }
        FromItem::File { path, format, alias } => {
            Ok(FromItem::File { path: path.clone(), format: *format, alias: alias.clone() })
        }
        // A compute-only source holding no data, so unlike other derived tables cloning costs
        // nothing real.
        FromItem::GenerateSeries { start, stop, step, inclusive, alias, column_alias } => {
            Ok(FromItem::GenerateSeries {
                start: *start,
                stop: *stop,
                step: *step,
                inclusive: *inclusive,
                alias: alias.clone(),
                column_alias: column_alias.clone(),
            })
        }
        FromItem::Join { .. } | FromItem::Subquery { .. } | FromItem::Unnest { .. } => {
            err!(UnsupportedFeature)
        }
    }
}

/// Expands `UNPIVOT` into a `UNION ALL`.
///
/// For each target column it builds one `SELECT` that "passes through everything but the
/// target column, emits the target column's name as a string literal into `name_col`, and its
/// value into `value_col`", and bundles them all with `UNION ALL`. `from_schema` is the
/// target table's column list (used to decide the passed-through "non-target columns"; no real data is read).
pub fn desugar_unpivot(
    arena: &mut ExprArena,
    stmt: UnpivotStmt,
    from_schema: &[Field],
) -> Result<QueryStmt> {
    let UnpivotStmt { from, columns, name_col, value_col, order_by, limit, offset } = stmt;
    ensure!(!columns.is_empty(), SyntaxError);
    ensure!(columns.len() <= MAX_UNPIVOT_COLUMNS, ExpressionTooDeep);

    // Targets must be unqualified bare column references (see the `UnpivotStmt` docs).
    let mut target_names: Vec<String> = Vec::with_capacity(columns.len());
    for &c in &columns {
        match arena.get(c) {
            Expr::ColumnRef { qualifier: None, name } => target_names.push(name.clone()),
            _ => err!(UnsupportedFeature),
        }
    }

    // Non-target columns pass through unchanged (the same default as DuckDB).
    let other_names: Vec<String> = from_schema
        .iter()
        .map(|f| f.name.clone())
        .filter(|n| !target_names.iter().any(|t| eq_ascii_ci(t.as_bytes(), n.as_bytes())))
        .collect();

    let mut branches: Vec<SetExpr> = Vec::with_capacity(target_names.len());
    for name in &target_names {
        let mut items: Vec<SelectItem> = Vec::with_capacity(other_names.len() + 2);
        for n in &other_names {
            let e = arena.push(Expr::ColumnRef { qualifier: None, name: n.clone() });
            items.push(SelectItem { expr: e, alias: None });
        }
        let name_lit = arena.push(Expr::Literal(Value::Bytes(name.clone().into_bytes())));
        items.push(SelectItem { expr: name_lit, alias: Some(name_col.clone()) });
        let val_ref = arena.push(Expr::ColumnRef { qualifier: None, name: name.clone() });
        items.push(SelectItem { expr: val_ref, alias: Some(value_col.clone()) });

        let sel = SelectStmt { items, from: Some(clone_from_item(&from)?), ..SelectStmt::empty() };
        branches.push(SetExpr::Select(Box::new(sel)));
    }

    // Bundled into a left-associative `UNION ALL` chain. The same idea as GROUPING SETS
    // bundling several `Node::Aggregate` with `Node::SetOp` (see the module docs).
    let mut iter = branches.into_iter();
    let mut body = match iter.next() {
        Some(b) => b,
        None => err!(Internal), // cannot happen unless columns is empty
    };
    for b in iter {
        body = SetExpr::SetOp {
            op: SetOp::Union,
            all: true,
            left: Box::new(body),
            right: Box::new(b),
        };
    }

    Ok(QueryStmt { ctes: Vec::new(), body, order_by, order_by_all: None, limit, offset })
}
