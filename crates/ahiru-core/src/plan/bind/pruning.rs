//! Split-pruning predicate extraction: pulls `column <op> literal`/`BETWEEN`/
//! `IN (literals)` comparisons out of a pushed-down filter and turns them
//! into `Pruner`s the format layer can use to skip whole splits.

use super::*;

// --- 枝刈り述語の抽出 --------------------------------------------------------

/// `列 <op> 定数` の形を取り出す。
///
/// AND で連結された枝だけを辿る。OR の下では「片方が真なら通る」ため、
/// 個々の条件で分割を落とすと結果が変わってしまう。
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

/// `列 IN (定数, ...)` を 1 つの `PruneOp::In` pruner にまとめる。
///
/// リストの要素が 1 つでもリテラル以外（列参照・部分式）なら、その要素が
/// 何に等しくなるか分からず候補集合を確定できないため、pruner ごと諦める
/// （安全側 = 枝刈りしない）。NULL リテラルは `x = NULL` が真になり得ない
/// ので候補から除くだけでよい。
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
    // 文字列統計は writer による切り詰めがあるため、v1 では数値・時刻のみ扱う。
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
        // `<>` は統計で落とせない（範囲内のどこかに他の値がありうる）。
        _ => return None,
    };
    Some(Pruner { column, op, value, in_values: Vec::new() })
}
