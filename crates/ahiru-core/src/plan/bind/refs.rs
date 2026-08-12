//! Reference resolution and small shared helpers used across the binder:
//! `GROUP BY`/`ORDER BY`/`DISTINCT ON` ordinal and alias resolution, the
//! generic expression-tree walker (`each_child`), column-reference
//! collection, and name synthesis for unnamed output columns.

use super::from::FromTree;
use super::*;

// --- Reference resolution for ORDER BY / GROUP BY ---------------------------

/// Reads `GROUP BY 1` / `GROUP BY alias` as the corresponding SELECT expression.
pub(super) fn resolve_select_ref(
    arena: &ExprArena,
    sel: &SelectStmt,
    id: ExprId,
) -> Result<ExprId> {
    if let Some(n) = ordinal_of(arena, id) {
        let i = n as usize;
        ensure!(i >= 1 && i <= sel.items.len(), ColumnNotFound);
        return Ok(sel.items[i - 1].expr);
    }
    if let Expr::ColumnRef { qualifier: None, name } = arena.get(id) {
        for item in &sel.items {
            if let Some(a) = &item.alias {
                if eq_ascii_ci(a.as_bytes(), name.as_bytes()) {
                    return Ok(item.expr);
                }
            }
        }
    }
    Ok(id)
}

/// Returns the column number if an ORDER BY item points at an output column.
pub(super) fn order_output_column(
    arena: &ExprArena,
    sel: &SelectStmt,
    o: &OrderByItem,
    schema: &[Field],
) -> Result<Option<usize>> {
    if let Some(n) = ordinal_of(arena, o.expr) {
        let i = n as usize;
        ensure!(i >= 1 && i <= schema.len(), ColumnNotFound);
        return Ok(Some(i - 1));
    }
    if let Expr::ColumnRef { qualifier: None, name } = arena.get(o.expr) {
        for (i, item) in sel.items.iter().enumerate() {
            if let Some(a) = &item.alias {
                if eq_ascii_ci(a.as_bytes(), name.as_bytes()) && i < schema.len() {
                    return Ok(Some(i));
                }
            }
        }
        // A `*` `RENAME (old AS new, ...)` acts like a per-column alias for
        // `ORDER BY` purposes too (verified against `duckdb`: `ORDER BY new`
        // resolves to the renamed output column, and so does `ORDER BY old`
        // via the normal scope-resolution fallback below). Resolved by
        // matching the final schema name rather than select-item position,
        // since a `*` can expand into any number of columns and the two
        // don't line up.
        let is_rename_target = sel.items.iter().any(|it| {
            matches!(arena.get(it.expr), Expr::Star { rename, .. }
                if rename.iter().any(|(_, new)| eq_ascii_ci(new.as_bytes(), name.as_bytes())))
        });
        if is_rename_target {
            if let Some(i) =
                schema.iter().position(|f| eq_ascii_ci(f.name.as_bytes(), name.as_bytes()))
            {
                return Ok(Some(i));
            }
        }
    }
    // If it structurally matches an output expression, use that column (avoiding recomputation).
    for (col, item) in sel.items.iter().enumerate() {
        if matches!(arena.get(item.expr), Expr::Star { .. }) {
            // `*` expands to several columns, so positions cannot be lined up. Give up.
            return Ok(None);
        }
        if expr_eq(arena, item.expr, o.expr) && col < schema.len() {
            return Ok(Some(col));
        }
    }
    Ok(None)
}

/// Returns the column number if a `DISTINCT ON` expression points at an output column (by
/// alias match or structural match). The version of `order_output_column` without ordinals
/// (DISTINCT ON has no ordinal form such as `ON (1)`).
pub(super) fn distinct_on_output_column(
    arena: &ExprArena,
    sel: &SelectStmt,
    on_expr: ExprId,
    schema: &[Field],
) -> Option<usize> {
    if let Expr::ColumnRef { qualifier: None, name } = arena.get(on_expr) {
        for (i, item) in sel.items.iter().enumerate() {
            if let Some(a) = &item.alias {
                if eq_ascii_ci(a.as_bytes(), name.as_bytes()) && i < schema.len() {
                    return Some(i);
                }
            }
        }
    }
    for (col, item) in sel.items.iter().enumerate() {
        if matches!(arena.get(item.expr), Expr::Star { .. }) {
            return None;
        }
        if expr_eq(arena, item.expr, on_expr) && col < schema.len() {
            return Some(col);
        }
    }
    None
}

/// Returns the value if it is a positive integer literal.
pub(super) fn ordinal_of(arena: &ExprArena, id: ExprId) -> Option<u32> {
    match arena.get(id) {
        Expr::Literal(Value::I32(v)) if *v > 0 => Some(*v as u32),
        Expr::Literal(Value::I64(v)) if *v > 0 && *v <= u32::MAX as i64 => Some(*v as u32),
        _ => None,
    }
}

// --- Traversal helpers -------------------------------------------------------

/// Visits every direct child of an expression.
pub(super) fn each_child(
    arena: &ExprArena,
    id: ExprId,
    f: &mut dyn FnMut(ExprId) -> Result<()>,
) -> Result<()> {
    match arena.get(id) {
        Expr::Literal(_)
        | Expr::IntervalLiteral(_)
        | Expr::TypedLiteral(_, _)
        | Expr::Param(_)
        | Expr::Star { .. }
        | Expr::ColumnRef { .. } => {}
        Expr::Unary { arg, .. } | Expr::Cast { arg, .. } | Expr::IsNull { arg, .. } => f(*arg)?,
        Expr::Binary { lhs, rhs, .. } => {
            f(*lhs)?;
            f(*rhs)?;
        }
        Expr::Between { arg, low, high, .. } => {
            f(*arg)?;
            f(*low)?;
            f(*high)?;
        }
        Expr::InList { arg, list, .. } => {
            f(*arg)?;
            for i in list {
                f(*i)?;
            }
        }
        Expr::Like { arg, pattern, .. } => {
            f(*arg)?;
            f(*pattern)?;
        }
        Expr::Case { operand, whens, else_ } => {
            if let Some(o) = operand {
                f(*o)?;
            }
            for (c, v) in whens {
                f(*c)?;
                f(*v)?;
            }
            if let Some(e) = else_ {
                f(*e)?;
            }
        }
        Expr::Function { args, filter, .. } => {
            for a in args {
                f(*a)?;
            }
            if let Some(fl) = filter {
                f(*fl)?;
            }
        }
        Expr::Window { args, partition_by, order_by, .. } => {
            for a in args.iter().chain(partition_by) {
                f(*a)?;
            }
            for o in order_by {
                f(o.expr)?;
            }
        }
        // Expressions inside a subquery are resolved in a different scope, so they are not walked here.
        Expr::ScalarSubquery(_) | Expr::Exists { .. } => {}
        Expr::InSubquery { arg, .. } => f(*arg)?,
        // The `query` side is not walked, for the same reason as `InSubquery`. Only `arg`
        // belongs to this query's scope.
        Expr::QuantifiedComparison { arg, .. } => f(*arg)?,
        Expr::Unnest(arg) => f(*arg)?,
        // A lambda body can reference only its parameters, not columns of the enclosing scope
        // (see `plan::compile::Compiler::lambda_call`). Walking into it as a child would, when
        // a parameter name happens to match an outer scope column name, mistake it for an
        // outer column reference, so it is deliberately not walked (excluded from GROUP BY
        // validation and projection pushdown).
        Expr::Lambda { .. } => {}
    }
    Ok(())
}

/// Collects the scope column numbers an expression references. Nonexistent columns are detected here.
pub(super) fn collect_refs(
    arena: &ExprArena,
    scope: &Scope,
    id: ExprId,
    out: &mut Vec<usize>,
) -> Result<()> {
    collect_refs_at(arena, scope, id, out, 0)
}

fn collect_refs_at(
    arena: &ExprArena,
    scope: &Scope,
    id: ExprId,
    out: &mut Vec<usize>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    if let Expr::ColumnRef { qualifier, name } = arena.get(id) {
        out.push(scope.resolve(qualifier.as_deref(), name)?);
        return Ok(());
    }
    let d = depth + 1;
    each_child(arena, id, &mut |c| collect_refs_at(arena, scope, c, out, d))
}

pub(super) fn collect_join_refs(
    arena: &ExprArena,
    scope: &Scope,
    tree: &FromTree,
    out: &mut Vec<usize>,
) -> Result<()> {
    if let FromTree::Join { left, right, on, .. } = tree {
        if let Some(on) = on {
            collect_refs(arena, scope, *on, out)?;
        }
        collect_join_refs(arena, scope, left, out)?;
        collect_join_refs(arena, scope, right, out)?;
    }
    Ok(())
}

// --- Naming ------------------------------------------------------------------

pub(super) fn group_name(arena: &ExprArena, id: ExprId, i: usize) -> String {
    match arena.get(id) {
        Expr::ColumnRef { name, .. } => name.clone(),
        _ => {
            let mut s = String::from("group");
            push_u32(&mut s, i as u32);
            s
        }
    }
}

/// Builds a program returning only a constant. Used to fill grouping columns not in the set
/// with NULL under GROUPING SETS, and to carry the result of `GROUPING()`/`GROUPING_ID()`
/// (a bitmask) as a constant column.
pub(super) fn const_program(ty: Ty, v: Value) -> Program {
    let mut p = Program::new();
    let k = p.add_const(ty, v);
    let dst = p.alloc_reg();
    p.push(Instr::with_aux(OpCode::LoadConst, ty.phys(), dst, 0, 0, k));
    p.result = dst;
    p.result_ty = ty;
    p
}

/// The name of an output column with no alias. A column reference keeps its name; anything else gets a serial number.
pub(super) fn default_name(arena: &ExprArena, id: ExprId) -> String {
    match arena.get(id) {
        Expr::ColumnRef { name, .. } => name.clone(),
        // duckdb's `UNNEST(x)` (with no alias) also names its output column "unnest".
        Expr::Unnest(_) => String::from("unnest"),
        // Reconstructing the expression would need string assembly and cost size, so a number suffices.
        _ => {
            let mut s = String::from("col");
            push_u32(&mut s, id);
            s
        }
    }
}

pub(super) fn push_u32(s: &mut String, mut v: u32) {
    let mut buf = [0u8; 10];
    let mut n = 0;
    loop {
        buf[n] = b'0' + (v % 10) as u8;
        n += 1;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    for i in (0..n).rev() {
        s.push(buf[i] as char);
    }
}
