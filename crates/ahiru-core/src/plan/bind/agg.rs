//! Aggregate extraction and rewriting: `GROUP BY`/`GROUPING SETS` support,
//! window-function binding, `UNNEST` element-type narrowing, and the
//! NULL/zero coalescing used when decorrelating correlated `COUNT`
//! aggregates.

use super::refs::{const_program, default_name, each_child};
use super::*;

/// Drops the trailing `k` columns used as correlation keys by the most recent join.
pub(super) fn drop_trailing_columns(node: Node, k: usize) -> Result<Node> {
    let s = Scope::from_fields(node.schema().to_vec());
    ensure!(s.len() >= k, Internal);
    let keep = s.len() - k;
    let mut exprs = Vec::with_capacity(keep);
    let mut schema = Vec::with_capacity(keep);
    for i in 0..keep {
        exprs.push(column_program(&s, i)?);
        schema.push(s.fields()[i].clone());
    }
    Ok(Node::Project { input: Box::new(node), exprs, schema })
}

/// Merges the registers, constants, and cast tables of two `Program`s into one and appends
/// `Coalesce(a.result, b.result)`. The same merge rules as `and_programs` (`compile.rs`)
/// (sharing `compile::merge_program_bodies`); only the final combining operator differs.
fn coalesce_programs(mut a: Program, b: Program) -> Program {
    let ty = a.result_ty;
    let (ra, rb) = crate::plan::compile::merge_program_bodies(&mut a, b);
    let dst = a.alloc_reg();
    a.push(Instr::new(OpCode::Coalesce, ty.phys(), dst, ra, rb));
    a.result = dst;
    a
}

/// Builds a `Program` that corrects column `i` of `scope` to 0 when it is NULL.
/// Used to correct `count(*)`/`count(x)` to "aggregate over 0 rows -> 0" when the pseudo-group
/// for a correlated GROUP BY does not exist (= there is no matching inner row)
/// (confirmed with DuckDB: it should match an ordinary uncorrelated `count`).
fn coalesce_zero(scope: &Scope, i: usize) -> Result<Program> {
    let ty = scope.fields()[i].ty;
    let col = column_program(scope, i)?;
    let zero = const_program(ty, Value::I64(0));
    Ok(coalesce_programs(col, zero))
}

/// Wraps a `Project` that corrects only the `target` column of the aggregate result node (a
/// COUNT-family result) with `coalesce_zero`, passing the other columns (the correlated GROUP BY key columns) through.
pub(super) fn coalesce_count_column(node: Node, target: usize) -> Result<Node> {
    let scope = Scope::from_fields(node.schema().to_vec());
    let mut exprs = Vec::with_capacity(scope.len());
    for i in 0..scope.len() {
        if i == target {
            exprs.push(coalesce_zero(&scope, i)?);
        } else {
            exprs.push(column_program(&scope, i)?);
        }
    }
    let schema = scope.fields().to_vec();
    Ok(Node::Project { input: Box::new(node), exprs, schema })
}

// --- Extracting aggregates ---------------------------------------------------

/// Collects the aggregate calls inside an expression, deduplicating already-seen ones by structural equality.
pub(super) fn collect_aggregates(
    arena: &ExprArena,
    id: ExprId,
    out: &mut Vec<ExprId>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    let d = depth + 1;
    if let Expr::Function { name, args, .. } = arena.get(id) {
        if AggKind::from_name(name).is_some() {
            // Nested aggregates are invalid SQL.
            for a in args {
                let mut nested = Vec::new();
                collect_aggregates(arena, *a, &mut nested, d)?;
                ensure!(nested.is_empty(), NotAggregate);
            }
            if !out.iter().any(|&e| expr_eq(arena, e, id)) {
                out.push(id);
            }
            return Ok(());
        }
    }
    each_child(arena, id, &mut |c| collect_aggregates(arena, c, out, d))
}

/// `GROUPING(col, ...)` / `GROUPING_ID(col, ...)`.
///
/// Unlike aggregate functions, these do not evaluate their arguments (they are simply
/// replaced by a bitmask constant fixed per grouping set), so they are collected separately
/// from `collect_aggregates`.
fn is_grouping_fn(name: &str) -> bool {
    eq_ascii_ci(name.as_bytes(), b"grouping") || eq_ascii_ci(name.as_bytes(), b"grouping_id")
}

/// Collects the `GROUPING`/`GROUPING_ID` calls inside an expression. Nesting is invalid
/// (the same reason as aggregates: a call appearing inside its own arguments is meaningless).
pub(super) fn collect_grouping_calls(
    arena: &ExprArena,
    id: ExprId,
    out: &mut Vec<ExprId>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    let d = depth + 1;
    if let Expr::Function { name, distinct, star, filter, .. } = arena.get(id) {
        if is_grouping_fn(name) {
            ensure!(!*distinct && !*star && filter.is_none(), UnsupportedFeature);
            if !out.iter().any(|&e| expr_eq(arena, e, id)) {
                out.push(id);
            }
            return Ok(());
        }
    }
    each_child(arena, id, &mut |c| collect_grouping_calls(arena, c, out, d))
}

/// Collects the window function calls inside an expression. Nesting is invalid.
pub(super) fn collect_windows(
    arena: &ExprArena,
    id: ExprId,
    out: &mut Vec<ExprId>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    let d = depth + 1;
    if let Expr::Window { args, partition_by, order_by, .. } = arena.get(id) {
        // Nested window functions are invalid SQL.
        for a in args.iter().chain(partition_by) {
            let mut nested = Vec::new();
            collect_windows(arena, *a, &mut nested, d)?;
            ensure!(nested.is_empty(), UnsupportedFeature);
        }
        for o in order_by {
            let mut nested = Vec::new();
            collect_windows(arena, o.expr, &mut nested, d)?;
            ensure!(nested.is_empty(), UnsupportedFeature);
        }
        if !out.iter().any(|&e| expr_eq(arena, e, id)) {
            out.push(id);
        }
        return Ok(());
    }
    each_child(arena, id, &mut |c| collect_windows(arena, c, out, d))
}

/// Collects the `Expr::Unnest` calls inside an expression. Unlike aggregates and windows,
/// nesting (a further `UNNEST` inside an `UNNEST(...)` argument) needs no special handling --
/// `each_child` walks into the arguments so it is found naturally, and if several are found
/// the caller rejects with `UnsupportedFeature` (the same "neither an aggregate nor an
/// ordinary expression" collection style as when `FILTER`/`QUALIFY` were implemented).
pub(super) fn collect_unnests(
    arena: &ExprArena,
    id: ExprId,
    out: &mut Vec<ExprId>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    let d = depth + 1;
    if matches!(arena.get(id), Expr::Unnest(_)) {
        out.push(id);
        return Ok(());
    }
    each_child(arena, id, &mut |c| collect_unnests(arena, c, out, d))
}

/// Recovers UNNEST's element type from `Ty::Json` to the actual scalar type where possible.
///
/// In this engine arrays and objects are all unified as `Ty::Json` (JSON text), and the
/// element type cannot be known without seeing the real data (see the
/// `vector::types::Ty::Json` docs). But a query's output column types must be settled at
/// bind time (before execution), so this narrows only when a native type can be determined
/// "safely" without reading the data.
///
/// The one case decidable without data is when `UNNEST`'s target is a direct call to
/// `json_array(...)`/`list_value(...)` (the equivalent of DuckDB's list literal; this engine
/// has no array literal syntax such as `[1,2,3]`, so this is how you write the equivalent of
/// `duckdb -c "SELECT UNNEST([1,2,3])"`): each argument's **compile-time type** is known, so
/// whether every argument settles on the same non-JSON scalar type without nesting can be
/// decided without reading a single row.
///
/// Everything else (the ordinary case of UNNESTing a table's JSON column itself) cannot be
/// decided without reading real data, so `Ty::Json` is returned unchanged (per the task's
/// requirement, no narrowing is needed in that case).
pub(super) fn narrow_unnest_elem_ty(
    arena: &ExprArena,
    scope: &Scope,
    params: &[Value],
    arg: ExprId,
) -> Ty {
    let (name, args) = match arena.get(arg) {
        Expr::Function { name, args, distinct: false, star: false, filter: None } => (name, args),
        _ => return Ty::Json,
    };
    let is_array_ctor =
        eq_ascii_ci(name.as_bytes(), b"json_array") || eq_ascii_ci(name.as_bytes(), b"list_value");
    if !is_array_ctor || args.is_empty() {
        return Ty::Json;
    }
    let mut common: Option<Ty> = None;
    for &a in args {
        let ty = match compile(arena, scope, params, a) {
            Ok(p) => p.result_ty,
            Err(_) => return Ty::Json,
        };
        // If the argument is itself JSON (possibly containing arrays or objects), the premise
        // that elements are "scalars without nesting" collapses, so give up.
        if ty == Ty::Json {
            return Ty::Json;
        }
        common = Some(match common {
            None => ty,
            Some(c) => match Ty::unify(c, ty) {
                Some(u) => u,
                None => return Ty::Json,
            },
        });
    }
    match common {
        Some(t) if t.is_integer() => Ty::BigInt,
        Some(Ty::Float) | Some(Ty::Double) => Ty::Double,
        Some(Ty::Varchar) => Ty::Varchar,
        Some(Ty::Boolean) => Ty::Boolean,
        // Recovery through JSON text is not implemented for DECIMAL/DATE/TIME/TIMESTAMP/NULL
        // and the like (out of scope. Round-tripping them while preserving precision and
        // formatting would require a dedicated JSON serialization convention for this engine,
        // so the scope was narrowed).
        _ => Ty::Json,
    }
}

pub(super) fn build_window(
    arena: &ExprArena,
    scope: &Scope,
    params: &[Value],
    subs: &[Substitution],
    windows: &[(String, WindowDef)],
    id: ExprId,
) -> Result<WindowSpec> {
    let (name, args, star, window_ref, inline_partition_by, inline_order_by, inline_frame) =
        match arena.get(id) {
            Expr::Window { name, args, star, window_ref, partition_by, order_by, frame } => {
                (name, args, *star, window_ref, partition_by, order_by, *frame)
            }
            _ => err!(Internal),
        };
    // `OVER w` (a named reference) looks up the definition from the `WINDOW` clause here.
    // The `WINDOW` clause syntactically follows the SELECT list, so the name cannot be
    // resolved at parse time; it is resolved at bind time in this one place.
    let (partition_by, order_by, frame): (&[ExprId], &[OrderByItem], WindowFrame) = match window_ref
    {
        Some(wname) => {
            let def = windows.iter().find(|(n, _)| eq_ascii_ci(n.as_bytes(), wname.as_bytes()));
            match def {
                Some((_, d)) => (&d.partition_by, &d.order_by, d.frame),
                // A reference to an undefined window name. `duckdb` rejects it at parse time as
                // `window "w" does not exist`, but this engine does not look ahead at the
                // `WINDOW` clause and so detects it at bind time.
                None => err!(UnsupportedFeature),
            }
        }
        None => (inline_partition_by, inline_order_by, inline_frame),
    };
    let kind = match WindowKind::from_name(name) {
        Some(k) => k,
        None => err!(FunctionNotFound),
    };

    let mut arg_progs = Vec::with_capacity(args.len());
    for a in args {
        arg_progs.push(compile_with_subs(arena, scope, params, subs, *a)?);
    }
    // `count(*) OVER ()` is treated as an aggregate taking no arguments.
    let kind = if star {
        ensure!(kind == WindowKind::Agg(AggKind::Count), WrongArgCount);
        WindowKind::Agg(AggKind::CountStar)
    } else {
        kind
    };
    if kind.is_nullary() || kind == WindowKind::Agg(AggKind::CountStar) {
        ensure!(arg_progs.is_empty(), WrongArgCount);
    } else {
        ensure!(!arg_progs.is_empty(), WrongArgCount);
    }

    let mut parts = Vec::with_capacity(partition_by.len());
    for p in partition_by {
        parts.push(compile_with_subs(arena, scope, params, subs, *p)?);
    }
    let mut keys = Vec::with_capacity(order_by.len());
    for o in order_by {
        keys.push(SortKey {
            expr: compile_with_subs(arena, scope, params, subs, o.expr)?,
            desc: o.desc,
            nulls_first: o.nulls_first,
        });
    }

    let arg_ty = arg_progs.first().map_or(Ty::Null, |p| p.result_ty);
    let result_ty = match kind {
        // Ranking is a 1-based running number.
        WindowKind::RowNumber | WindowKind::Rank | WindowKind::DenseRank => Ty::BigInt,
        // Functions that merely carry a value return the input type unchanged.
        WindowKind::Lag | WindowKind::Lead | WindowKind::FirstValue | WindowKind::LastValue => {
            arg_ty
        }
        // Aggregates follow the same rules as ordinary aggregates. The same function is used so the two cannot drift.
        WindowKind::Agg(a) => a.result_ty(arg_ty)?,
    };

    let mut label = String::from(name);
    label.push_str("_over");
    Ok(WindowSpec {
        kind,
        args: arg_progs,
        partition_by: parts,
        order_by: keys,
        frame,
        result_ty,
        name: label,
    })
}

/// If this query is the simple shape "SELECT exactly one bare aggregate call", returns that
/// aggregate's kind. Used so the caller (the scalar-subquery handling in `bind_select_in`)
/// can re-decide, without going through the bound `Plan`, whether a correlated scalar
/// subquery was an aggregate subquery (one going through the inner correlated GROUP BY
/// decorrelation). Once a correlated scalar subquery has bound successfully, anything with
/// an aggregate is guaranteed to be in this shape (see the `ensure!` on the early-return
/// path of `bind_select_in`), so this check and the actual binding result always agree.
pub(super) fn as_bare_aggregate(arena: &ExprArena, q: &QueryStmt) -> Option<AggKind> {
    let sel = q.as_simple_select()?;
    if sel.items.len() != 1 {
        return None;
    }
    let mut probe = Vec::new();
    collect_aggregates(arena, sel.items[0].expr, &mut probe, 0).ok()?;
    if probe.len() != 1 || probe[0] != sel.items[0].expr {
        return None;
    }
    match arena.get(probe[0]) {
        Expr::Function { name, star, .. } => {
            let kind = AggKind::from_name(name)?;
            Some(if *star { AggKind::CountStar } else { kind })
        }
        _ => None,
    }
}

pub(super) fn build_agg(
    arena: &ExprArena,
    scope: &Scope,
    params: &[Value],
    id: ExprId,
) -> Result<Agg> {
    let (name, args, distinct, star, filter) = match arena.get(id) {
        Expr::Function { name, args, distinct, star, filter } => {
            (name, args, *distinct, *star, *filter)
        }
        _ => err!(Internal),
    };
    let kind = match AggKind::from_name(name) {
        Some(k) => k,
        None => err!(FunctionNotFound),
    };
    // A FILTER condition is evaluated in the pre-aggregation input scope (the same as args).
    // An aggregate cannot be written there (`collect_aggregates` does not treat it specially,
    // so an aggregate inside a FILTER is caught by the ordinary "nested aggregate" detection).
    let filter_prog = match filter {
        Some(f) => Some(compile_predicate(arena, scope, params, f)?),
        None => None,
    };
    if star {
        ensure!(kind == AggKind::Count, WrongArgCount);
        ensure!(!distinct, UnsupportedFeature);
        return Ok(Agg {
            kind: AggKind::CountStar,
            arg: None,
            distinct: false,
            name: String::from("count_star()"),
            separator: Vec::new(),
            filter: filter_prog,
        });
    }

    // Only `string_agg(x, sep)` allows two arguments. sep must be a constant literal.
    let max_args = if kind.optional_arg_default().is_some() { 2 } else { 1 };
    ensure!(!args.is_empty() && args.len() <= max_args, WrongArgCount);
    let arg = compile(arena, scope, params, args[0])?;
    let separator = match args.get(1) {
        Some(&sep_id) => match arena.get(sep_id) {
            Expr::Literal(Value::Bytes(b)) => b.clone(),
            _ => err!(UnsupportedFeature),
        },
        None => kind.optional_arg_default().map(|d| d.to_vec()).unwrap_or_default(),
    };
    Ok(Agg {
        kind,
        arg: Some(arg),
        distinct,
        name: agg_name(name, arena, args[0]),
        separator,
        filter: filter_prog,
    })
}

fn agg_name(fname: &str, arena: &ExprArena, arg: ExprId) -> String {
    let mut s = String::from(fname);
    s.push('(');
    s.push_str(&default_name(arena, arg));
    s.push(')');
    s
}

/// Detects a bare column reference that is not in GROUP BY.
pub(super) fn check_grouped(
    arena: &ExprArena,
    scope: &Scope,
    id: ExprId,
    groups: &[ExprId],
    aggs: &[ExprId],
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    if groups.iter().any(|&g| expr_eq(arena, g, id)) {
        return Ok(());
    }
    if aggs.iter().any(|&a| expr_eq(arena, a, id)) {
        return Ok(());
    }
    match arena.get(id) {
        Expr::ColumnRef { qualifier, name } => {
            // A column that exists in the input reaching here = it is in neither GROUP BY nor an aggregate.
            if scope.resolve(qualifier.as_deref(), name).is_ok() {
                err!(NotGrouped);
            }
        }
        // Scalar subqueries are attached alongside as pre-aggregation columns, so referencing
        // one bare above the aggregate would shift the column numbers. Rejected like a column reference.
        Expr::ScalarSubquery(_) => err!(NotGrouped),
        _ => {}
    }
    let d = depth + 1;
    each_child(arena, id, &mut |c| check_grouped(arena, scope, c, groups, aggs, d))
}
