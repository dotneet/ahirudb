//! Reference resolution and small shared helpers used across the binder:
//! `GROUP BY`/`ORDER BY`/`DISTINCT ON` ordinal and alias resolution, the
//! generic expression-tree walker (`each_child`), column-reference
//! collection, and name synthesis for unnamed output columns.

use super::from::FromTree;
use super::*;

// --- Reference resolution for ORDER BY / GROUP BY ---------------------------

/// Reads `GROUP BY <name>` as either the input column of that name or, failing
/// that, the SELECT-list alias of that name.
///
/// Unlike ORDER BY (`order_output_column`), an *input* column wins over a
/// select-list alias that shadows it: `SELECT id % 2 AS id ... GROUP BY id`
/// groups by the table's `id`, not by `id % 2`. DuckDB and PostgreSQL both
/// bind the input column first here, and only fall back to the alias for a
/// name the FROM clause does not provide.
pub(super) fn resolve_group_ref(
    arena: &ExprArena,
    sel: &SelectStmt,
    scope: &Scope,
    id: ExprId,
) -> Result<ExprId> {
    if let Expr::ColumnRef { qualifier: None, name } = arena.get(id) {
        if scope.resolve(None, name).is_ok() {
            return Ok(id);
        }
    }
    resolve_select_ref(arena, sel, id)
}

/// Reads `ORDER BY 1` / `ORDER BY alias` as the corresponding SELECT expression.
/// Alias-first; see `resolve_group_ref` for the GROUP BY rule.
pub(super) fn resolve_select_ref(
    arena: &ExprArena,
    sel: &SelectStmt,
    id: ExprId,
) -> Result<ExprId> {
    if let Some(n) = numeric_ordinal_of(arena, id) {
        let i = ordinal_index(n, sel.items.len())?;
        return Ok(sel.items[i].expr);
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
    if let Some(n) = numeric_ordinal_of(arena, o.expr) {
        return Ok(Some(ordinal_index(n, schema.len())?));
    }
    if let Expr::ColumnRef { qualifier: None, name } = arena.get(o.expr) {
        // Resolve aliases against the *expanded* output schema, not the
        // select-item index. `SELECT *, expr AS extra ORDER BY extra` has
        // `extra` as item 1 but schema column N after the star expands.
        // Output aliases are resolved last-wins when a SELECT list contains
        // duplicate names. This is also the rule used by QUALIFY, and keeps
        // ORDER BY consistent with the post-projection output scope.
        if let Some(i) =
            schema.iter().rposition(|f| eq_ascii_ci(f.name.as_bytes(), name.as_bytes()))
        {
            return Ok(Some(i));
        }
    }
    // If it structurally matches an output expression, use that column (avoiding recomputation).
    // A `*` / `COLUMNS(...)` item expands to several columns, so select-item
    // positions no longer line up with the schema; fall through and recompile.
    if !sel.items.iter().any(|it| matches!(arena.get(it.expr), Expr::Star { .. })) {
        for (col, item) in sel.items.iter().enumerate() {
            if expr_eq(arena, item.expr, o.expr) && col < schema.len() {
                return Ok(Some(col));
            }
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
        if let Some(i) = schema.iter().position(|f| eq_ascii_ci(f.name.as_bytes(), name.as_bytes()))
        {
            return Some(i);
        }
    }
    if !sel.items.iter().any(|it| matches!(arena.get(it.expr), Expr::Star { .. })) {
        for (col, item) in sel.items.iter().enumerate() {
            if expr_eq(arena, item.expr, on_expr) && col < schema.len() {
                return Some(col);
            }
        }
    }
    None
}

/// Returns the value if it is a positive integer literal.
pub(super) fn ordinal_of(arena: &ExprArena, id: ExprId) -> Option<u32> {
    match numeric_ordinal_of(arena, id) {
        Some(NumericTerm::Int(v)) if v > 0 && v <= u32::MAX as i64 => Some(v as u32),
        _ => None,
    }
}

/// A numeric literal written where an `ORDER BY` / `GROUP BY` term goes.
///
/// Kept separate from `ordinal_of` because an out-of-range value must be an
/// *error*, not a fall-through to "an ordinary constant sort key": `ORDER BY 0`
/// and `ORDER BY 1.5` are rejected by DuckDB ("ORDER term out of range"), and
/// silently treating them as a constant would make the clause a no-op.
#[derive(Clone, Copy)]
pub(super) enum NumericTerm {
    Int(i64),
    /// A non-integer numeric literal (`1.5`). Never a valid position.
    Fractional,
}

/// Recognizes a numeric literal, including one behind a unary minus (the
/// parser keeps `-1` as `Unary(Neg, 1)` rather than folding it).
pub(super) fn numeric_ordinal_of(arena: &ExprArena, id: ExprId) -> Option<NumericTerm> {
    match arena.get(id) {
        Expr::Literal(Value::I32(v)) => Some(NumericTerm::Int(*v as i64)),
        Expr::Literal(Value::I64(v)) => Some(NumericTerm::Int(*v)),
        Expr::Literal(Value::I128(v)) => Some(match i64::try_from(*v) {
            Ok(v) => NumericTerm::Int(v),
            // Beyond i64 it cannot be a valid position either way.
            Err(_) => NumericTerm::Fractional,
        }),
        Expr::Literal(Value::F64(_)) => Some(NumericTerm::Fractional),
        Expr::Unary { op: UnaryOp::Neg, arg } => match numeric_ordinal_of(arena, *arg)? {
            NumericTerm::Int(v) => Some(NumericTerm::Int(v.checked_neg()?)),
            NumericTerm::Fractional => Some(NumericTerm::Fractional),
        },
        _ => None,
    }
}

/// Converts a positional term into a 0-based column index, rejecting anything
/// outside `1..=len`.
fn ordinal_index(n: NumericTerm, len: usize) -> Result<usize> {
    let v = match n {
        NumericTerm::Int(v) => v,
        NumericTerm::Fractional => err!(ColumnNotFound),
    };
    ensure!(v >= 1 && v as u64 <= len as u64, ColumnNotFound);
    Ok(v as usize - 1)
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

/// Collects, for projection pushdown only, every column reference in an expression **including
/// those inside subquery bodies** that resolves in `scope`.
///
/// `each_child` deliberately stops at a subquery boundary, because a subquery's expressions are
/// resolved in a different scope. That is right for name resolution but wrong for pushdown: an
/// outer column used *only* inside a correlated subquery — `SELECT id, (SELECT count(*) FROM t s
/// WHERE s.flag = t.flag) FROM t` — would be pruned from the scan, and binding the subquery
/// against the narrowed outer scope would then fail with `TableNotFound`.
///
/// References that do not resolve in `scope` belong to the subquery's own scope and are
/// ignored rather than reported (unlike `collect_refs`, this walker never validates: the
/// ordinary `collect_refs` pass over the same expressions does that). An inner column that
/// merely shares a name with an outer one is therefore over-collected, which costs one extra
/// column read and never a wrong answer.
pub(super) fn collect_outer_refs(
    arena: &ExprArena,
    scope: &Scope,
    id: ExprId,
    out: &mut Vec<usize>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    if let Expr::ColumnRef { qualifier, name } = arena.get(id) {
        if let Ok(i) = scope.resolve(qualifier.as_deref(), name) {
            out.push(i);
        }
        return Ok(());
    }
    let d = depth + 1;
    match arena.get(id) {
        Expr::ScalarSubquery(q)
        | Expr::Exists { query: q, .. }
        | Expr::InSubquery { query: q, .. }
        | Expr::QuantifiedComparison { query: q, .. } => {
            query_outer_refs(arena, scope, q, out, d)?;
        }
        _ => {}
    }
    each_child(arena, id, &mut |c| collect_outer_refs(arena, scope, c, out, d))
}

fn query_outer_refs(
    arena: &ExprArena,
    scope: &Scope,
    q: &QueryStmt,
    out: &mut Vec<usize>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    let d = depth + 1;
    for c in &q.ctes {
        query_outer_refs(arena, scope, &c.query, out, d)?;
    }
    set_expr_outer_refs(arena, scope, &q.body, out, d)?;
    for o in &q.order_by {
        collect_outer_refs(arena, scope, o.expr, out, d)?;
    }
    Ok(())
}

fn set_expr_outer_refs(
    arena: &ExprArena,
    scope: &Scope,
    body: &SetExpr,
    out: &mut Vec<usize>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    let d = depth + 1;
    match body {
        SetExpr::Select(s) => select_outer_refs(arena, scope, s, out, d),
        SetExpr::SetOp { left, right, .. } => {
            set_expr_outer_refs(arena, scope, left, out, d)?;
            set_expr_outer_refs(arena, scope, right, out, d)
        }
    }
}

fn select_outer_refs(
    arena: &ExprArena,
    scope: &Scope,
    s: &SelectStmt,
    out: &mut Vec<usize>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    let d = depth + 1;
    for item in &s.items {
        collect_outer_refs(arena, scope, item.expr, out, d)?;
    }
    for e in [s.filter, s.having, s.qualify].into_iter().flatten() {
        collect_outer_refs(arena, scope, e, out, d)?;
    }
    for &e in s.group_by.iter().chain(&s.distinct_on) {
        collect_outer_refs(arena, scope, e, out, d)?;
    }
    if let Some(sets) = &s.grouping_sets {
        for set in sets {
            for &e in set {
                collect_outer_refs(arena, scope, e, out, d)?;
            }
        }
    }
    for o in &s.order_by {
        collect_outer_refs(arena, scope, o.expr, out, d)?;
    }
    for (_, def) in &s.windows {
        for &p in &def.partition_by {
            collect_outer_refs(arena, scope, p, out, d)?;
        }
        for o in &def.order_by {
            collect_outer_refs(arena, scope, o.expr, out, d)?;
        }
    }
    match &s.from {
        Some(f) => from_outer_refs(arena, scope, f, out, d),
        None => Ok(()),
    }
}

fn from_outer_refs(
    arena: &ExprArena,
    scope: &Scope,
    f: &FromItem,
    out: &mut Vec<usize>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    let d = depth + 1;
    match f {
        FromItem::Join { left, right, on, .. } => {
            if let Some(on) = on {
                collect_outer_refs(arena, scope, *on, out, d)?;
            }
            from_outer_refs(arena, scope, left, out, d)?;
            from_outer_refs(arena, scope, right, out, d)
        }
        FromItem::Subquery { query, .. } => query_outer_refs(arena, scope, query, out, d),
        FromItem::Unnest { expr, .. } => collect_outer_refs(arena, scope, *expr, out, d),
        FromItem::Table { .. } | FromItem::File { .. } | FromItem::GenerateSeries { .. } => Ok(()),
    }
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
