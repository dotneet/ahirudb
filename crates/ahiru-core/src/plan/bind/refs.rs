//! Reference resolution and small shared helpers used across the binder:
//! `GROUP BY`/`ORDER BY`/`DISTINCT ON` ordinal and alias resolution, the
//! generic expression-tree walker (`each_child`), column-reference
//! collection, and name synthesis for unnamed output columns.

use super::from::FromTree;
use super::*;

// --- ORDER BY / GROUP BY の参照解決 -----------------------------------------

/// `GROUP BY 1` / `GROUP BY alias` を対応する SELECT 式に読み替える。
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

/// ORDER BY の項が出力列を指しているならその番号を返す。
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
    }
    // 出力式と構造が一致するならその列を使う（再計算を避ける）。
    for (col, item) in sel.items.iter().enumerate() {
        if matches!(arena.get(item.expr), Expr::Star { .. }) {
            // `*` は複数列に展開されるので位置合わせが取れない。諦める。
            return Ok(None);
        }
        if expr_eq(arena, item.expr, o.expr) && col < schema.len() {
            return Ok(Some(col));
        }
    }
    Ok(None)
}

/// `DISTINCT ON` の式が出力列（別名一致 or 構造一致）を指しているなら
/// その番号を返す。`order_output_column` の序数を除いた版
/// （DISTINCT ON に `ON (1)` のような序数指定は無い）。
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

/// 正の整数リテラルなら値を返す。
pub(super) fn ordinal_of(arena: &ExprArena, id: ExprId) -> Option<u32> {
    match arena.get(id) {
        Expr::Literal(Value::I32(v)) if *v > 0 => Some(*v as u32),
        Expr::Literal(Value::I64(v)) if *v > 0 && *v <= u32::MAX as i64 => Some(*v as u32),
        _ => None,
    }
}

// --- 走査ヘルパ --------------------------------------------------------------

/// 式の直接の子をすべて訪問する。
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
        // サブクエリの中の式は別スコープで解決するので、ここでは辿らない。
        Expr::ScalarSubquery(_) | Expr::Exists { .. } => {}
        Expr::InSubquery { arg, .. } => f(*arg)?,
        // `query` 側は `InSubquery` と同じ理由で辿らない。`arg` だけが
        // このクエリのスコープに属する式。
        Expr::QuantifiedComparison { arg, .. } => f(*arg)?,
        Expr::Unnest(arg) => f(*arg)?,
        // ラムダ本体はパラメータだけを参照でき、外側スコープの列は参照
        // できない（`plan::compile::Compiler::lambda_call` 参照）。ここで
        // 子として辿ると、パラメータ名がたまたま外側スコープの列名と
        // 一致したときに誤って外側の列参照とみなされてしまうので、
        // 意図的に辿らない（GROUP BY 検証・射影プッシュダウンの対象外）。
        Expr::Lambda { .. } => {}
    }
    Ok(())
}

/// 式が参照するスコープ上の列番号を集める。存在しない列はここで検出する。
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

// --- 名前付け ----------------------------------------------------------------

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

/// 定数だけを返すプログラムを作る。GROUPING SETS でセットに含まれない
/// グルーピング列を NULL で埋めるのと、`GROUPING()`/`GROUPING_ID()` の
/// 結果（ビットマスク）を定数列として載せるのに使う。
pub(super) fn const_program(ty: Ty, v: Value) -> Program {
    let mut p = Program::new();
    let k = p.add_const(ty, v);
    let dst = p.alloc_reg();
    p.push(Instr::with_aux(OpCode::LoadConst, ty.phys(), dst, 0, 0, k));
    p.result = dst;
    p.result_ty = ty;
    p
}

/// 別名が無い出力列の名前。列参照はその名前、それ以外は連番。
pub(super) fn default_name(arena: &ExprArena, id: ExprId) -> String {
    match arena.get(id) {
        Expr::ColumnRef { name, .. } => name.clone(),
        // duckdb の `UNNEST(x)`（別名無し）も出力列名は "unnest"。
        Expr::Unnest(_) => String::from("unnest"),
        // 式の再構成は文字列組み立てが要りサイズを食うので、番号で済ませる。
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
