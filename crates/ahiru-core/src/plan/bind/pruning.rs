//! Split-pruning predicate extraction: pulls `column <op> literal`/`BETWEEN`/
//! `IN (literals)` comparisons out of a pushed-down filter and turns them
//! into `Pruner`s the format layer can use to skip whole splits.

use super::*;
use crate::expr::kernels::{pow10_i128, MICROS_PER_DAY};
use crate::vector::PhysType;

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

/// The constant behind a literal node, with the type the binder gives it.
///
/// `Expr::TypedLiteral` is included: `DATE '2024-01-01'` / `TIMESTAMP '...'` and
/// `CURRENT_DATE` are folded to that node at parse time (`sql::parser::expr`,
/// `sql::now`), and matching only `Expr::Literal` here would silently give up
/// pruning on exactly the date/timestamp columns it helps most.
fn literal_of(arena: &ExprArena, id: ExprId) -> Option<(Value, Ty)> {
    match arena.get(id) {
        Expr::Literal(v) | Expr::TypedLiteral(v, _) if v.is_null() => None,
        Expr::Literal(v) => Some((v.clone(), v.default_ty())),
        Expr::TypedLiteral(v, ty) => Some((v.clone(), *ty)),
        _ => None,
    }
}

/// Whether widening a column of type `col` to `unified` leaves its raw (physical)
/// representation untouched.
///
/// Statistics hold the column's own unwidened value, so a pruner may only compare
/// against them when this holds -- otherwise the comparison the engine actually
/// performs happens in a different domain (DATE days versus TIMESTAMP microseconds,
/// a DECIMAL rescaled to a wider scale) and the min/max bounds mean nothing there.
fn widening_keeps_raw(col: Ty, unified: Ty) -> bool {
    if col == unified {
        return true;
    }
    match (col.as_decimal(), unified.as_decimal()) {
        // Integers and DECIMALs share one "integer scaled by 10^scale" model, so the raw
        // value survives exactly when the scale does (INTEGER -> BIGINT, DECIMAL(5,1) ->
        // DECIMAL(11,1)).
        (Some((_, s1)), Some((_, s2))) => s1 == s2,
        // FLOAT -> DOUBLE and TIMESTAMP -> TIMESTAMPTZ keep the value as it stands.
        _ => col.phys() == unified.phys(),
    }
}

/// Rewrites a literal into the column's own internal representation, the one
/// `format::parquet::stat_value` reconstructs from Parquet statistics and
/// `plain_encode_for_bloom` hashes: a DECIMAL's unscaled integer, a TIMESTAMP's
/// microseconds, and so on.
///
/// Without this the unscaled `150` of `d = 150` would be matched against the `1500`
/// a `DECIMAL(5,1)` column actually stores, and RowGroup/page/Bloom pruning would drop
/// rows that do match. The conversion mirrors what the comparison kernel does with the
/// same literal (`expr::kernels::int_conv`): both sides widen to `Ty::unify(col, lit)`,
/// and the literal is rescaled into that type.
///
/// Returns `None` whenever the conversion is not exact or not defined; the caller then
/// drops the pruner and the split is simply read. No pruning is always safe; a wrong
/// one silently loses rows.
fn to_column_repr(value: &Value, lit_ty: Ty, col_ty: Ty) -> Option<Value> {
    let unified = Ty::unify(col_ty, lit_ty)?;
    if !widening_keeps_raw(col_ty, unified) {
        return None;
    }
    if col_ty.phys() == PhysType::F64 {
        return match value {
            Value::F64(_) => Some(value.clone()),
            // An integer literal against a FLOAT/DOUBLE column is widened to DOUBLE the
            // same way by the comparison kernel, so the rounding matches.
            _ if lit_ty.is_integer() => value.as_f64().map(Value::F64),
            _ => None,
        };
    }
    let raw = match value {
        Value::I32(x) => *x as i128,
        Value::I64(x) => *x as i128,
        Value::I128(x) => *x,
        // A fractional literal has no exact representation at the column's scale
        // (and DECIMAL against a floating-point literal is compared as DOUBLE anyway,
        // which these integer-domain statistics cannot answer).
        _ => return None,
    };
    let scaled = if unified.is_temporal() {
        // DATE (days) compared against TIMESTAMP (microseconds).
        let mul = if lit_ty == Ty::Date && unified != Ty::Date { MICROS_PER_DAY } else { 1 };
        raw.checked_mul(mul)?
    } else {
        let (_, lit_scale) = lit_ty.as_decimal()?;
        let (_, col_scale) = unified.as_decimal()?;
        raw.checked_mul(pow10_i128(u32::from(col_scale.checked_sub(lit_scale)?))?)?
    };
    match col_ty.phys() {
        // A value the column's physical width cannot hold matches nothing, but saying so
        // would need per-operator reasoning; giving up is the cheaper safe answer.
        PhysType::I32 => i32::try_from(scaled).ok().map(Value::I32),
        PhysType::I64 => i64::try_from(scaled).ok().map(Value::I64),
        PhysType::I128 => Some(Value::I128(scaled)),
        _ => None,
    }
}

/// Bundles `column IN (constant, ...)` into a single `PruneOp::In` pruner.
///
/// If even one element of the list is not a literal (a column reference or subexpression),
/// what it equals is unknown and the candidate set cannot be settled, so the whole pruner is
/// abandoned (erring safe = no pruning). The same goes for a literal with no exact
/// representation in the column's own domain. A NULL literal can simply be dropped from the
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
            Expr::Literal(v) | Expr::TypedLiteral(v, _) if v.is_null() => continue,
            _ => {}
        }
        let (v, lit_ty) = literal_of(arena, item)?;
        values.push(to_column_repr(&v, lit_ty, ty)?);
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
    let (value, lit_ty) = literal_of(arena, lit)?;
    let column = scope.resolve(qual, name).ok()?;
    // String statistics can be truncated by the writer, so v1 handles only numeric and temporal types.
    let ty = scope.fields()[column].ty;
    if !(ty.is_numeric() || ty.is_temporal()) {
        return None;
    }
    let value = to_column_repr(&value, lit_ty, ty)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_literal_is_scaled_to_the_decimal_column() {
        let col = Ty::Decimal { precision: 5, scale: 1 };
        assert_eq!(to_column_repr(&Value::I32(150), Ty::Int, col), Some(Value::I64(1500)));
        let col = Ty::Decimal { precision: 15, scale: 2 };
        assert_eq!(
            to_column_repr(&Value::I64(3_000_000_050), Ty::BigInt, col),
            Some(Value::I64(300_000_005_000))
        );
    }

    #[test]
    fn fractional_literal_against_a_decimal_column_is_dropped() {
        // The kernel compares this pair as DOUBLE, which the integer statistics of the
        // column cannot answer, so there must be no pruner at all.
        let col = Ty::Decimal { precision: 5, scale: 1 };
        assert_eq!(to_column_repr(&Value::F64(150.5), Ty::Double, col), None);
    }

    #[test]
    fn literal_too_wide_for_the_column_is_dropped() {
        let col = Ty::Decimal { precision: 18, scale: 6 };
        // 10^18 * 10^6 overflows the i64 the column is stored in.
        assert_eq!(to_column_repr(&Value::I64(1_000_000_000_000_000_000), Ty::BigInt, col), None);
        // An INTEGER column's statistics are I32; a wider literal has no I32 form.
        assert_eq!(to_column_repr(&Value::I64(5_000_000_000), Ty::BigInt, Ty::Int), None);
    }

    #[test]
    fn integer_literal_takes_the_columns_physical_variant() {
        // `Value::partial_cmp_same` only compares like variants, so the literal has to end
        // up in the same one `stat_value` produces for that column.
        assert_eq!(to_column_repr(&Value::I32(7), Ty::Int, Ty::BigInt), Some(Value::I64(7)));
        assert_eq!(to_column_repr(&Value::I32(7), Ty::Int, Ty::Int), Some(Value::I32(7)));
        assert_eq!(to_column_repr(&Value::I32(7), Ty::Int, Ty::Double), Some(Value::F64(7.0)));
        assert_eq!(to_column_repr(&Value::F64(7.5), Ty::Double, Ty::Float), Some(Value::F64(7.5)));
    }

    #[test]
    fn date_literal_is_converted_for_a_timestamp_column() {
        assert_eq!(to_column_repr(&Value::I32(3), Ty::Date, Ty::Date), Some(Value::I32(3)));
        assert_eq!(
            to_column_repr(&Value::I32(3), Ty::Date, Ty::Timestamp),
            Some(Value::I64(3 * 86_400_000_000))
        );
        // The other way round the *column* would have to be rescaled, which statistics
        // cannot express, so no pruner is built.
        assert_eq!(to_column_repr(&Value::I64(0), Ty::Timestamp, Ty::Date), None);
    }
}
