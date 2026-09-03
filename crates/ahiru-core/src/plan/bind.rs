//! AST -> logical plan.
//!
//! Three things happen here.
//!
//! 1. **Name resolution** (`Scope`). Joins produce several same-named columns, so names are handled with qualifiers.
//! 2. **Projection pushdown**. Only the columns actually referenced are read by the scan.
//!    Reading fewer bytes is the most effective optimization in this engine, so it happens
//!    during binding rather than in a later optimization pass (DESIGN.md §9).
//! 3. **Aggregate rewriting**. Aggregate calls and GROUP BY expressions inside
//!    SELECT / HAVING / ORDER BY are replaced with references to the aggregate operator's output columns.

use crate::catalog::Catalog;
use crate::expr::{Instr, OpCode, Program};
use crate::format::{PruneOp, Pruner};
use crate::plan::compile::{
    and_programs, cast_program, column_program, compile, compile_predicate,
    compile_predicate_with_subs, compile_with_subs, expr_eq, Substitution,
};
#[cfg(feature = "ddl")]
use crate::plan::MemScanSpec;
use crate::plan::SetOpKind;
use crate::plan::{
    Agg, AggKind, Node, Plan, ScanSpec, Scope, SecondArg, SortKey, WindowKind, WindowSpec,
};
use crate::prelude::*;
use crate::rt::hash::eq_ascii_ci;
use crate::sql::ast::{
    BinaryOp, Cte, Expr, ExprArena, ExprId, FromItem, JoinKind, OrderByItem, Parsed, QueryStmt,
    SelectStmt, SetExpr, SetOp, Stmt, UnaryOp, WindowDef, WindowFrame,
};
use crate::vector::{Field, Ty, Value};

use cte::{bind_one_cte, CteScope};
use refs::ordinal_of;
use select::bind_select_in;

/// The FROM-clause nesting limit. The parser limits it too; this is a second layer of defense.
const MAX_FROM_DEPTH: u32 = 64;
/// The recursion limit for the pre-bind table-reference walk
/// (`from::referenced_in_query` and friends).
///
/// Kept separate from `MAX_FROM_DEPTH` because that walk spends more frames per
/// syntactic level than the binder does: two per nested query block (the query
/// itself, then the FROM item or expression it descends into) plus one per
/// JOIN / set-operation link. The parser caps nesting at 64 levels and the whole
/// statement at 64 such links, so `2 * 64 + 64` frames is the most a parseable
/// statement can need; the budget must stay above that. A statement the binder
/// accepts but the walk rejects is not merely a stricter limit — the missed
/// table never gets its schema resolved and binding fails with `Internal` (see
/// `from::referenced_in_query`).
const MAX_REF_DEPTH: u32 = 4 * MAX_FROM_DEPTH;
/// The expression nesting limit.
const MAX_EXPR_DEPTH: u32 = 64;

pub fn bind(catalog: &Catalog, parsed: &Parsed, params: &[Value]) -> Result<Plan> {
    match &parsed.stmt {
        Stmt::Select(q) | Stmt::Explain(q) => bind_query(catalog, &parsed.arena, q, params),
        // DESCRIBE / SHOW are handled by the session layer.
        _ => err!(UnsupportedFeature),
    }
}

pub fn bind_query(
    catalog: &Catalog,
    arena: &ExprArena,
    q: &QueryStmt,
    params: &[Value],
) -> Result<Plan> {
    bind_query_at(catalog, arena, q, params, 0)
}

/// Like [`bind_query`], but substitutes `now()` / `CURRENT_DATE` inside view
/// bodies using `now_micros` (the same value `Session::set_now` recorded).
pub(crate) fn bind_query_at(
    catalog: &Catalog,
    arena: &ExprArena,
    q: &QueryStmt,
    params: &[Value],
    now_micros: i64,
) -> Result<Plan> {
    let mut ctes = CteScope::default();
    #[cfg(feature = "ddl")]
    {
        ctes.now_micros = now_micros;
    }
    let _ = now_micros;
    bind_query_in(catalog, arena, q, params, &mut ctes, None)
}

/// `outer_scope` is used only when binding a correlated subquery. Top-level queries, CTEs,
/// and FROM-clause derived tables always pass `None` (the backward-compatible default).
fn bind_query_in(
    catalog: &Catalog,
    arena: &ExprArena,
    q: &QueryStmt,
    params: &[Value],
    ctes: &mut CteScope,
    outer_scope: Option<&Scope>,
) -> Result<Plan> {
    // Every query block owns its WITH clause. Keep the entries only while this
    // block is being bound so nested derived/scalar subqueries can see outer CTEs
    // while their own definitions shadow them. `bind_query_at` used to process
    // only the top-level list, leaving nested WITH clauses invisible.
    let cte_start = ctes.len();
    for c in &q.ctes {
        // A CTE may reference CTEs defined earlier in the same query block. CTE
        // definitions are always uncorrelated (they have no outer scope).
        bind_one_cte(catalog, arena, c, params, ctes, cte_start)?;
    }
    let result = bind_query_body(catalog, arena, q, params, ctes, outer_scope);
    ctes.truncate(cte_start);
    result
}

/// Binds the body and trailing clauses after `bind_query_in` has installed the
/// query block's local CTE definitions.
fn bind_query_body(
    catalog: &Catalog,
    arena: &ExprArena,
    q: &QueryStmt,
    params: &[Value],
    ctes: &mut CteScope,
    outer_scope: Option<&Scope>,
) -> Result<Plan> {
    let (mut node, correlated) = bind_set_expr(catalog, arena, &q.body, params, ctes, outer_scope)?;
    // Correlation key columns are implementation columns appended at the end and are
    // invisible to ORDER BY ordinals and column names (they do not exist in SQL terms).
    let visible_len = node.schema().len() - correlated.len();
    // The case where a correlated subquery carries an ORDER BY/LIMIT on the `QueryStmt` side
    // (double parentheses, with a CTE, and so on) slips through the similar check in
    // `bind_select_in`. It is explicitly rejected here for the same reason (it would apply to
    // the whole thing rather than per correlation key).
    ensure!(
        correlated.is_empty()
            || (q.order_by.is_empty()
                && q.order_by_all.is_none()
                && q.limit.is_none()
                && q.offset.is_none()),
        UnsupportedFeature
    );

    // An outer ORDER BY / LIMIT applies to the whole set-operation result.
    if !q.order_by.is_empty() || q.order_by_all.is_some() {
        let scope = Scope::from_fields(node.schema()[..visible_len].to_vec());
        let mut keys = Vec::with_capacity(q.order_by.len().max(visible_len));
        // `ORDER BY ALL`: sorts by the output columns left to right, all in the same
        // direction and with the same NULL placement (confirmed that
        // `duckdb -c "select g,h from t union all select g,h from t order by all"` applies to
        // the whole set-operation result). "Output columns" here are the set operation's own
        // result schema, so unlike on the `bind_select_in` side, no projection needs rebuilding.
        if let Some(oa) = &q.order_by_all {
            for col in 0..visible_len {
                keys.push(SortKey {
                    expr: column_program(&scope, col)?,
                    desc: oa.desc,
                    nulls_first: oa.nulls_first,
                });
            }
        }
        for o in &q.order_by {
            // The original expressions are gone from a set-operation result, so only ordinals or output column names.
            let col = match ordinal_of(arena, o.expr) {
                Some(n) => {
                    ensure!(n >= 1 && (n as usize) <= scope.len(), ColumnNotFound);
                    n as usize - 1
                }
                None => match arena.get(o.expr) {
                    Expr::ColumnRef { qualifier, name } => {
                        match scope.resolve(qualifier.as_deref(), name) {
                            Ok(col) => col,
                            Err(e) if e.code == crate::error::Code::ColumnNotFound => {
                                // DuckDB also accepts an alias introduced by a
                                // non-first branch of a set operation. The
                                // output schema keeps the first branch's names,
                                // so recover the corresponding ordinal from
                                // the set-expression tree when the normal
                                // output-scope lookup misses it.
                                ensure!(qualifier.is_none(), ColumnNotFound);
                                match set_output_alias(arena, &q.body, name) {
                                    Some(col) if col < scope.len() => col,
                                    _ => err!(ColumnNotFound),
                                }
                            }
                            Err(e) => return Err(e),
                        }
                    }
                    _ => err!(UnsupportedFeature),
                },
            };
            keys.push(SortKey {
                expr: column_program(&scope, col)?,
                desc: o.desc,
                nulls_first: o.nulls_first,
            });
        }
        let topn = q
            .limit
            .map(|l| l.saturating_add(q.offset.unwrap_or(0)).min(usize::MAX as u64) as usize);
        node = Node::Sort { input: Box::new(node), keys, limit: topn };
    }
    if q.limit.is_some() || q.offset.unwrap_or(0) > 0 {
        node = Node::Limit { input: Box::new(node), limit: q.limit, offset: q.offset.unwrap_or(0) };
    }
    Ok(Plan { root: node, correlated })
}

/// Finds an explicit output alias in any leaf of a set-operation tree. Set
/// operation result names normally come from the first SELECT, but DuckDB lets
/// the trailing ORDER BY refer to an alias from another branch too. Star
/// expansion makes an AST item index insufficient, so only alias-only branches
/// are used here; ordinary output-scope resolution still handles their names.
fn set_output_alias(arena: &ExprArena, e: &SetExpr, name: &str) -> Option<usize> {
    match e {
        SetExpr::Select(s) => {
            if s.items.iter().any(|item| matches!(arena.get(item.expr), Expr::Star { .. })) {
                return None;
            }
            s.items.iter().enumerate().find_map(|(i, item)| {
                item.alias
                    .as_deref()
                    .filter(|alias| eq_ascii_ci(alias.as_bytes(), name.as_bytes()))
                    .map(|_| i)
            })
        }
        SetExpr::SetOp { left, right, .. } => {
            set_output_alias(arena, left, name).or_else(|| set_output_alias(arena, right, name))
        }
    }
}

/// The returned `Vec<ExprId>` is the correlation keys `bind_select_in` detected (the same
/// meaning as `Plan::correlated`). Correlation is not propagated to either side of
/// `UNION`/`INTERSECT`/`EXCEPT` (if each side could carry a different set of correlation
/// keys they could not be merged into one plan; a correlated reference fails with the usual `ColumnNotFound`).
fn bind_set_expr(
    catalog: &Catalog,
    arena: &ExprArena,
    e: &SetExpr,
    params: &[Value],
    ctes: &mut CteScope,
    outer_scope: Option<&Scope>,
) -> Result<(Node, Vec<ExprId>)> {
    match e {
        SetExpr::Select(s) => {
            let plan = bind_select_in(catalog, arena, s, params, ctes, outer_scope)?;
            Ok((plan.root, plan.correlated))
        }
        SetExpr::SetOp { op, all, left, right } => {
            let (l, _) = bind_set_expr(catalog, arena, left, params, ctes, None)?;
            let (r, _) = bind_set_expr(catalog, arena, right, params, ctes, None)?;
            let schema = unify_setop_schema(l.schema(), r.schema())?;
            let op = match op {
                SetOp::Union => SetOpKind::Union,
                SetOp::Intersect => SetOpKind::Intersect,
                SetOp::Except => SetOpKind::Except,
            };
            // Columns whose types differ are aligned by a projection before being passed on.
            // Giving the set-operation operator type conversion too would break the key-encoding assumptions.
            let l = coerce_to(l, &schema)?;
            let r = coerce_to(r, &schema)?;
            Ok((
                Node::SetOp { left: Box::new(l), right: Box::new(r), op, all: *all, schema },
                Vec::new(),
            ))
        }
    }
}

/// The output schema of a set operation. A differing column count is an error; types settle on a common type.
fn unify_setop_schema(l: &[Field], r: &[Field]) -> Result<Vec<Field>> {
    ensure!(l.len() == r.len(), TypeMismatch);
    let mut out = Vec::with_capacity(l.len());
    for (a, b) in l.iter().zip(r) {
        let ty = crate::vector::Ty::unify_or_mismatch(a.ty, b.ty)?;
        // Names come from the left (per the SQL standard).
        out.push(Field::new(a.name.clone(), ty, a.nullable || b.nullable));
    }
    Ok(out)
}

/// Interposes a projection to match the output schema. Does nothing if the types already agree.
///
/// Rejects a column-count mismatch with `ColumnCountMismatch` rather than
/// silently truncating (`have.len() > want.len()`) or hitting an unrelated
/// `Internal` error from `column_program`'s out-of-range access
/// (`have.len() < want.len()`). Without this check, `WITH RECURSIVE` queries
/// whose anchor and recursive term have different column counts would have
/// the extra/missing columns silently dropped, which can turn a query that
/// should fail to converge into one that spins until the iteration cap
/// (`crates/ahiru-core/src/exec/recursive.rs`) is hit instead of erroring
/// immediately.
fn coerce_to(node: Node, want: &[Field]) -> Result<Node> {
    let have = node.schema();
    ensure!(have.len() == want.len(), ColumnCountMismatch);
    if have.iter().zip(want).all(|(a, b)| a.ty == b.ty) {
        return Ok(node);
    }
    let scope = Scope::from_fields(have.to_vec());
    let mut exprs = Vec::with_capacity(want.len());
    for (i, f) in want.iter().enumerate() {
        let mut p = column_program(&scope, i)?;
        if p.result_ty != f.ty {
            p = crate::plan::compile::cast_program(p, f.ty)?;
        }
        exprs.push(p);
    }
    Ok(Node::Project { input: Box::new(node), exprs, schema: want.to_vec() })
}

// --- Submodules ----------------------------------------------------------
//
// This file stays a thin entry point: the doc comment, top-level imports,
// and the query-level dispatch (`bind`/`bind_query`/`bind_query_in`/
// `bind_set_expr`, plus the small `unify_setop_schema`/`coerce_to` helpers
// they share). Everything else lives in `plan::bind::*` submodules instead
// (same split strategy as `sql::parser` -> `sql::parser/` and
// `expr::funcs` -> `expr::funcs/`).

mod agg;
mod cte;
mod from;
mod pivot;
mod pruning;
mod refs;
mod select;
mod subquery;

#[cfg(test)]
mod tests;

// Re-exported at the original flat path so `crate::plan::bind::{resolve_from,
// referenced_in_query, referenced_tables, desugar_pivot, desugar_unpivot}`
// keep resolving exactly as before for callers outside this module
// (`session.rs`, `ddl.rs`) even though the definitions now live in
// submodules. These stay `pub use` (not `pub(crate) use`) because the
// originals were `pub fn` directly in this module, reachable as public API
// through the fully-`pub` `crate::plan::bind` chain; `pub(crate)` would
// narrow that.
pub use from::{referenced_in_query, referenced_tables, resolve_from};
pub use pivot::{desugar_pivot, desugar_unpivot};
