//! Split-pruning predicate extraction: pulls `column <op> literal`/`BETWEEN`/
//! `IN (literals)` comparisons out of a pushed-down filter and turns them
//! into `Pruner`s the format layer can use to skip whole splits.

use super::*;

// --- Extracting pruning predicates -------------------------------------------

/// Extracts the shape `column <op> constant`.
///
/// Only branches joined by AND are walked. Under an OR, "if either side is true it passes",
/// so dropping a split on an individual condition would change the result.
pub(super) fn extract_pruners(arena: &ExprArena, id: ExprId, scope: &Scope, out: &mut Vec<Pruner>) {
    match arena.get(id) {
        Expr::Binary { op: BinaryOp::And, lhs, rhs } => {
            extract_pruners(arena, *lhs, scope, out);
            extract_pruners(arena, *rhs, scope, out);
        }
        Expr::Binary { op, lhs, rhs } if op.is_comparison() => {
            if let Some(p) = as_pruner(arena, *op, *lhs, *rhs, scope) {
                out.push(p);
            } else if let Some(p) = as_pruner(arena, op.swapped(), *rhs, *lhs, scope) {
                out.push(p);
            }
        }
        Expr::Between { arg, low, high, negated: false } => {
            if let Some(p) = as_pruner(arena, BinaryOp::Ge, *arg, *low, scope) {
                out.push(p);
            }
            if let Some(p) = as_pruner(arena, BinaryOp::Le, *arg, *high, scope) {
                out.push(p);
            }
        }
        Expr::InList { arg, list, negated: false } => {
            if let Some(p) = as_in_pruner(arena, *arg, list, scope) {
                out.push(p);
            }
        }
        _ => {}
    }
}

/// Bundles `column IN (constant, ...)` into a single `PruneOp::In` pruner.
///
/// If even one element of the list is not a literal (a column reference or subexpression),
/// what it equals is unknown and the candidate set cannot be settled, so the whole pruner is
/// abandoned (erring safe = no pruning). A NULL literal can simply be dropped from the
/// candidates, since `x = NULL` can never be true.
fn as_in_pruner(arena: &ExprArena, arg: ExprId, list: &[ExprId], scope: &Scope) -> Option<Pruner> {
    let (qual, name) = match arena.get(arg) {
        Expr::ColumnRef { qualifier, name } => (qualifier.as_deref(), name),
        _ => return None,
    };
    let column = scope.resolve(qual, name).ok()?;
    let ty = scope.fields()[column].ty;
    if !(ty.is_numeric() || ty.is_temporal()) {
        return None;
    }
    let mut values = Vec::new();
    for &item in list {
        match arena.get(item) {
            Expr::Literal(v) if v.is_null() => {}
            Expr::Literal(v) => values.push(v.clone()),
            _ => return None,
        }
    }
    let mut iter = values.into_iter();
    let value = iter.next()?;
    Some(Pruner { column, op: PruneOp::In, value, in_values: iter.collect() })
}

fn as_pruner(
    arena: &ExprArena,
    op: BinaryOp,
    col: ExprId,
    lit: ExprId,
    scope: &Scope,
) -> Option<Pruner> {
    let (qual, name) = match arena.get(col) {
        Expr::ColumnRef { qualifier, name } => (qualifier.as_deref(), name),
        _ => return None,
    };
    let value = match arena.get(lit) {
        Expr::Literal(v) if !v.is_null() => v.clone(),
        _ => return None,
    };
    let column = scope.resolve(qual, name).ok()?;
    // String statistics can be truncated by the writer, so v1 handles only numeric and temporal types.
    let ty = scope.fields()[column].ty;
    if !(ty.is_numeric() || ty.is_temporal()) {
        return None;
    }
    let op = match op {
        BinaryOp::Eq => PruneOp::Eq,
        BinaryOp::Lt => PruneOp::Lt,
        BinaryOp::Le => PruneOp::Le,
        BinaryOp::Gt => PruneOp::Gt,
        BinaryOp::Ge => PruneOp::Ge,
        // `<>` cannot be pruned by statistics (some other value may exist anywhere in the range).
        _ => return None,
    };
    Some(Pruner { column, op, value, in_values: Vec::new() })
}
