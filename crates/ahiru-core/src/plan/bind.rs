//! AST → 論理プラン。
//!
//! ここでやることは 3 つ。
//!
//! 1. **名前解決**（`Scope`）。結合が入ると同名列が複数出るので修飾子込みで扱う。
//! 2. **射影プッシュダウン**。実際に参照される列だけをスキャンに読ませる。
//!    読むバイト数が減るのがこのエンジンで最も効く最適化なので、後段の
//!    最適化パスではなく束縛と同時に行う（DESIGN.md §9）。
//! 3. **集約の書き換え**。SELECT / HAVING / ORDER BY の中の集約呼び出しと
//!    GROUP BY 式を、集約オペレータの出力列への参照に差し替える。

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
use crate::plan::{Agg, AggKind, Node, Plan, ScanSpec, Scope, SortKey, WindowKind, WindowSpec};
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

/// FROM 句のネスト上限。パーサ側でも制限しているが二重に守る。
const MAX_FROM_DEPTH: u32 = 64;
/// 式のネスト上限。
const MAX_EXPR_DEPTH: u32 = 64;

pub fn bind(catalog: &Catalog, parsed: &Parsed, params: &[Value]) -> Result<Plan> {
    match &parsed.stmt {
        Stmt::Select(q) | Stmt::Explain(q) => bind_query(catalog, &parsed.arena, q, params),
        // DESCRIBE / SHOW はセッション層が処理する。
        _ => err!(UnsupportedFeature),
    }
}

pub fn bind_query(
    catalog: &Catalog,
    arena: &ExprArena,
    q: &QueryStmt,
    params: &[Value],
) -> Result<Plan> {
    let mut ctes = CteScope::default();
    for c in &q.ctes {
        // CTE 自身も CTE を参照できる（前方定義のみ）。CTE の定義は常に
        // 非相関（外側スコープを持たない）。
        bind_one_cte(catalog, arena, c, params, &mut ctes)?;
    }
    bind_query_in(catalog, arena, q, params, &mut ctes, None)
}

/// `outer_scope` は相関サブクエリのバインドにのみ使う。トップレベルの
/// クエリ・CTE・FROM 句の派生表は常に `None`（後方互換のデフォルト）。
fn bind_query_in(
    catalog: &Catalog,
    arena: &ExprArena,
    q: &QueryStmt,
    params: &[Value],
    ctes: &mut CteScope,
    outer_scope: Option<&Scope>,
) -> Result<Plan> {
    let (mut node, correlated) = bind_set_expr(catalog, arena, &q.body, params, ctes, outer_scope)?;
    // 相関キー列は末尾に付加された実装用の列で、ORDER BY の序数・列名
    // からは見えない（SQL 上は存在しない列なので）。
    let visible_len = node.schema().len() - correlated.len();
    // 相関サブクエリが `QueryStmt` 側の ORDER BY/LIMIT を持つケース（二重括弧・
    // CTE 付きなど）は、`bind_select_in` 側の同様のチェックの網から漏れる。
    // ここでも同じ理由で明確に拒否する（相関キーごとではなく全体に効いて
    // しまうため）。
    ensure!(
        correlated.is_empty() || (q.order_by.is_empty() && q.limit.is_none() && q.offset.is_none()),
        UnsupportedFeature
    );

    // 外側の ORDER BY / LIMIT は集合演算の結果全体に掛かる。
    if !q.order_by.is_empty() {
        let scope = Scope::from_fields(node.schema()[..visible_len].to_vec());
        let mut keys = Vec::with_capacity(q.order_by.len());
        for o in &q.order_by {
            // 集合演算の結果には元の式が残っていないので、序数か出力列名だけ。
            let col = match ordinal_of(arena, o.expr) {
                Some(n) => {
                    ensure!(n >= 1 && (n as usize) <= scope.len(), ColumnNotFound);
                    n as usize - 1
                }
                None => match arena.get(o.expr) {
                    Expr::ColumnRef { qualifier, name } => {
                        scope.resolve(qualifier.as_deref(), name)?
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

/// 戻り値の `Vec<ExprId>` は `bind_select_in` が検出した相関キー（`Plan::correlated`
/// と同じ意味）。`UNION`/`INTERSECT`/`EXCEPT` の両辺には相関を伝播しない
/// （各辺が別々の相関キー集合を持ちうると 1 本のプランに統合できないため。
/// 相関参照があれば通常どおり `ColumnNotFound` で失敗する）。
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
            // 型がずれている列は射影で揃えてから渡す。集合演算のオペレータに
            // 型変換まで持たせると、キー符号化の前提が崩れる。
            let l = coerce_to(l, &schema)?;
            let r = coerce_to(r, &schema)?;
            Ok((
                Node::SetOp { left: Box::new(l), right: Box::new(r), op, all: *all, schema },
                Vec::new(),
            ))
        }
    }
}

/// 集合演算の出力スキーマ。列数が違えばエラー、型は共通型に寄せる。
fn unify_setop_schema(l: &[Field], r: &[Field]) -> Result<Vec<Field>> {
    ensure!(l.len() == r.len(), TypeMismatch);
    let mut out = Vec::with_capacity(l.len());
    for (a, b) in l.iter().zip(r) {
        let ty = crate::vector::Ty::unify_or_mismatch(a.ty, b.ty)?;
        // 名前は左を採る（SQL 標準）。
        out.push(Field::new(a.name.clone(), ty, a.nullable || b.nullable));
    }
    Ok(out)
}

/// 出力スキーマに合わせて射影を挟む。型が既に一致していれば何もしない。
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
