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
    SelectStmt, SetExpr, SetOp, Stmt, WindowDef, WindowFrame,
};
use crate::vector::{Field, Ty, Value};

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

// --- CTE ---------------------------------------------------------------------

/// `WITH` で定義された表。
///
/// **通常の CTE は 1 回しか参照できない。** プラン木は所有権で持っており
/// `Node` は複製できないため、2 回目の参照はエラーにする。複数回参照したい
/// 場合は派生表を 2 つ書く必要がある。黙って壊れるより明示的に断る。
///
/// 唯一の例外は再帰 CTE の再帰項内での自己参照（`CtePlan::Recursive` /
/// `ResolvedCte::WorkingTable`）: こちらはスキーマだけを持つ軽い参照で
/// 実データを所有しないため、自己結合のように何度参照してもよい。
#[derive(Default)]
pub struct CteScope {
    entries: Vec<CteEntry>,
    /// `CREATE VIEW` の展開ネスト数（ビューが他のビューを参照するケースの
    /// 無限再帰を防ぐ）。`CteScope` はサブクエリ・CTE を跨いで同一インスタンス
    /// が使い回されるので、ここに積んでおけば深さがどこでも正しく引き継がれる。
    #[cfg(feature = "ddl")]
    view_depth: u32,
}

/// ビュー展開の最大ネスト数。`MAX_FROM_DEPTH` より小さくしてあるのは、
/// ビュー 1 段の展開が丸ごと 1 回分の `bind_query_in` 呼び出し（再帰下降の
/// フルスタック）に相当し、単純な FROM の入れ子より 1 段あたりのスタック
/// 消費が大きいため。
#[cfg(feature = "ddl")]
const MAX_VIEW_DEPTH: u32 = 16;

/// CTE の実体。
enum CtePlan {
    /// 束縛済みで、まだ取り出されていない。
    Ready(Box<Node>),
    /// 再帰 CTE 自身の再帰項を束縛している最中で、自己参照はまだ実プランを
    /// 持たない。参照側は `Node::WorkingTable`（実行時に直前イテレーションの
    /// 新規行へ差し替わる葉）を受け取る（`exec::recursive` 参照）。
    Recursive,
    /// 既に取り出し済み。
    Taken,
}

struct CteEntry {
    name: String,
    plan: CtePlan,
    schema: Vec<Field>,
}

/// `FromItem::Table` が CTE 名を解決した結果。
enum ResolvedCte {
    /// 通常の CTE。1 回だけ取り出せる。
    Plan(Box<Node>),
    /// 再帰 CTE 本体からの自己参照。実データは持たず、実行時に
    /// `RecursiveCte` オペレータが差し込む。
    WorkingTable,
}

impl CteScope {
    fn find(&self, name: &str) -> Option<usize> {
        self.entries.iter().position(|e| eq_ascii_ci(e.name.as_bytes(), name.as_bytes()))
    }

    /// `find` で見つけた添字を解決する。スキーマは常にクローンで返す
    /// （`ResolvedCte::WorkingTable` は何度参照されてもよいため、複製の
    /// コストを払ってでも「1 回しか取り出せない」制約を外している）。
    fn resolve(&mut self, i: usize) -> Result<(ResolvedCte, Vec<Field>)> {
        let e = &mut self.entries[i];
        let schema = e.schema.clone();
        match core::mem::replace(&mut e.plan, CtePlan::Taken) {
            CtePlan::Ready(node) => Ok((ResolvedCte::Plan(node), schema)),
            CtePlan::Recursive => {
                // 取り出していないので戻す。何度でも参照できる。
                e.plan = CtePlan::Recursive;
                Ok((ResolvedCte::WorkingTable, schema))
            }
            CtePlan::Taken => err!(UnsupportedFeature),
        }
    }
}

/// クエリ全体（CTE + 集合演算 + 外側の ORDER BY / LIMIT）を束縛する。
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

/// 1 個の CTE を束縛して `ctes` に積む。
///
/// `c.recursive`（`WITH RECURSIVE` の対象）が立っていても、本文が実際に
/// 自分自身を参照していなければ普通の CTE として扱う（標準 SQL 通り、
/// `WITH RECURSIVE` のリストに非再帰の CTE が混ざってよい）。
fn bind_one_cte(
    catalog: &Catalog,
    arena: &ExprArena,
    c: &Cte,
    params: &[Value],
    ctes: &mut CteScope,
) -> Result<()> {
    if c.recursive {
        if let Some((anchor_expr, union_all, recursive_sel)) =
            split_recursive_cte(&c.query, &c.name)?
        {
            // アンカー（左辺）は自己参照を含まないので普通に束縛できる。
            let (anchor, _) = bind_set_expr(catalog, arena, anchor_expr, params, ctes, None)?;
            let out_schema = apply_cte_columns(anchor.schema().to_vec(), &c.columns)?;

            // 自己参照の解決先（作業テーブル）を先に登録する。実プランは
            // まだ無いので `CtePlan::Recursive` のまま、束縛が終わったら
            // 完成した `Node::RecursiveCte` に差し替える。
            ctes.entries.push(CteEntry {
                name: c.name.clone(),
                plan: CtePlan::Recursive,
                schema: out_schema.clone(),
            });
            let recursive_plan = bind_select_in(catalog, arena, recursive_sel, params, ctes, None)?;
            // 自己参照サブクエリは相関を持てない（そもそも外側スコープを
            // 渡していないので通常は起きないが、念のため明示する）。
            ensure!(recursive_plan.correlated.is_empty(), UnsupportedFeature);

            let anchor = coerce_to(anchor, &out_schema)?;
            let recursive_term = coerce_to(recursive_plan.root, &out_schema)?;
            let entry = match ctes.entries.last_mut() {
                Some(e) => e,
                None => err!(Internal),
            };
            entry.plan = CtePlan::Ready(Box::new(Node::RecursiveCte {
                anchor: Box::new(anchor),
                recursive_term: Box::new(recursive_term),
                union_all,
                schema: out_schema,
            }));
            return Ok(());
        }
    }
    let plan = bind_query_in(catalog, arena, &c.query, params, ctes, None)?;
    let schema = apply_cte_columns(plan.root.schema().to_vec(), &c.columns)?;
    ctes.entries.push(CteEntry {
        name: c.name.clone(),
        plan: CtePlan::Ready(Box::new(plan.root)),
        schema,
    });
    Ok(())
}

/// `name(col, ...)` の明示列名を出力スキーマへ反映する。列数が合わなければ
/// エラー。型は元のまま、名前だけ差し替える。
fn apply_cte_columns(mut schema: Vec<Field>, columns: &[String]) -> Result<Vec<Field>> {
    if columns.is_empty() {
        return Ok(schema);
    }
    ensure!(columns.len() == schema.len(), ColumnCountMismatch);
    for (f, name) in schema.iter_mut().zip(columns) {
        f.name = name.clone();
    }
    Ok(schema)
}

/// `c.recursive` が立っている CTE の本文が、実際に自分自身（`name`）を
/// 参照しているかを見る。参照していなければ `None`（普通の CTE として
/// 扱ってよい）。参照していれば `<anchor> UNION [ALL] <recursive_term>` の
/// 形になっていることを検証し、`(アンカー, UNION ALL か, 再帰項)` を返す。
///
/// 再帰項は単一の `SELECT`（さらなる集合演算を含まない）に限る。DuckDB は
/// もっと複雑な形（アンカー側の入れ子集合演算、再帰項内の複数自己参照）も
/// 許すが、実行オペレータ（`exec::recursive::RecursiveCte`）は「再帰項を
/// 1 つの物理プランとして毎イテレーション作り直す」設計なので、素直に
/// 対応できるこの範囲に絞る。
fn split_recursive_cte<'a>(
    query: &'a QueryStmt,
    name: &str,
) -> Result<Option<(&'a SetExpr, bool, &'a SelectStmt)>> {
    if !set_expr_references(&query.body, name) {
        return Ok(None);
    }
    // 再帰 CTE の本文に入れ子の WITH を許すと自己参照検出が複雑になるうえ、
    // `bind_query_in` はそもそも入れ子の `q.ctes` を処理しない（派生表と
    // 同じ既存の制約）。範囲外として明確に断る。
    ensure!(query.ctes.is_empty(), UnsupportedFeature);
    match &query.body {
        SetExpr::SetOp { op: SetOp::Union, all, left, right } => {
            ensure!(!set_expr_references(left, name), UnsupportedFeature);
            match right.as_ref() {
                SetExpr::Select(s) => Ok(Some((left, *all, s))),
                _ => err!(UnsupportedFeature),
            }
        }
        _ => err!(UnsupportedFeature),
    }
}

/// `SetExpr` の中に、FROM 句（JOIN・派生表を含む）として `name` への参照が
/// あるか。式（WHERE/SELECT リストのスカラサブクエリ・`EXISTS`・`IN`）の
/// 中は見ない —— `flatten_from` の自己参照解決も FROM 句だけを対象にして
/// いるので、検出範囲をそれに揃える。
fn set_expr_references(e: &SetExpr, name: &str) -> bool {
    match e {
        SetExpr::Select(s) => match &s.from {
            Some(f) => from_item_references(f, name),
            None => false,
        },
        SetExpr::SetOp { left, right, .. } => {
            set_expr_references(left, name) || set_expr_references(right, name)
        }
    }
}

fn from_item_references(f: &FromItem, name: &str) -> bool {
    match f {
        FromItem::Table { name: n, .. } => eq_ascii_ci(n.as_bytes(), name.as_bytes()),
        FromItem::Parquet { .. } => false,
        FromItem::Subquery { query, .. } => set_expr_references(&query.body, name),
        FromItem::Join { left, right, .. } => {
            from_item_references(left, name) || from_item_references(right, name)
        }
        // `expr` はスカラ式で、CTE 自己参照は FROM 句の中の `FromItem` としてしか
        // 検出しない（モジュール doc 参照）ので、ここには現れない。
        FromItem::Unnest { .. } => false,
        // カタログを経由しない計算だけのソースなので、テーブル名は参照しない。
        FromItem::GenerateSeries { .. } => false,
    }
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

// --- FROM 句の平坦化 ---------------------------------------------------------

/// FROM に現れる 1 つの関係。
struct Rel {
    /// カタログ上のテーブル添字。サブクエリなら `None`。
    table: Option<usize>,
    /// 修飾子（別名、無ければテーブル名）。
    alias: String,
    /// 関係が持つ全列。
    all: Vec<Field>,
    /// 参照される列（`all` 上の添字、昇順・重複なし）。
    needed: Vec<usize>,
    /// サブクエリの場合のプラン。
    subplan: Option<Node>,
    /// FROM 句の `UNNEST(expr) AS alias(col)` の `expr`（未束縛の AST 式）。
    /// `Some` のときは他のフィールドは「1 列だけのダミーの関係」の形だけ
    /// 整えてあり（`table`/`subplan` は使わない）、実プランは `build_tree` が
    /// 左の兄弟ノードをラップして特別に組み立てる（LATERAL 相当。
    /// `flatten_from`/`build_tree` の該当コメント参照）。
    unnest: Option<ExprId>,
}

/// FROM の木構造。葉は `rels` の添字。
enum FromTree {
    Rel(usize),
    Join { left: Box<FromTree>, right: Box<FromTree>, kind: JoinKind, on: Option<ExprId> },
}

impl FromTree {
    /// 木に外部結合が 1 つでも含まれるか。述語の押し下げ可否の判定に使う。
    fn has_outer_join(&self) -> bool {
        match self {
            FromTree::Rel(_) => false,
            FromTree::Join { left, right, kind, .. } => {
                !matches!(kind, JoinKind::Inner | JoinKind::Cross)
                    || left.has_outer_join()
                    || right.has_outer_join()
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn flatten_from(
    catalog: &Catalog,
    arena: &ExprArena,
    params: &[Value],
    from: &FromItem,
    rels: &mut Vec<Rel>,
    ctes: &mut CteScope,
    depth: u32,
) -> Result<FromTree> {
    ensure!(depth < MAX_FROM_DEPTH, ExpressionTooDeep);
    match from {
        FromItem::Table { name, alias } => {
            // CTE を先に見る。同名のテーブルより CTE が優先される（SQL 標準）。
            if let Some(k) = ctes.find(name) {
                let (resolved, all) = ctes.resolve(k)?;
                let alias = alias.clone().unwrap_or_else(|| name.clone());
                let subplan = match resolved {
                    ResolvedCte::Plan(node) => *node,
                    // 再帰 CTE 自身の再帰項からの自己参照。実データは持たず、
                    // 実行時に `RecursiveCte` オペレータが直前イテレーションの
                    // 新規行を差し込む（`exec::recursive` 参照）。
                    ResolvedCte::WorkingTable => Node::WorkingTable { schema: all.clone() },
                };
                rels.push(Rel {
                    table: None,
                    alias,
                    needed: (0..all.len()).collect(),
                    all,
                    subplan: Some(subplan),
                    unnest: None,
                });
                return Ok(FromTree::Rel(rels.len() - 1));
            }
            if let Some(i) = catalog.index_of(name) {
                return push_table_rel(
                    catalog,
                    i,
                    alias.clone().unwrap_or_else(|| name.clone()),
                    rels,
                );
            }
            // ファイルテーブルに無ければインメモリ表・ビューを見る
            // （`ddl` フィーチャのみ）。優先順位: CTE > ファイルテーブル >
            // インメモリ表 > ビュー。
            #[cfg(feature = "ddl")]
            {
                if let Some(i) = catalog.mem_index_of(name) {
                    let alias = alias.clone().unwrap_or_else(|| name.clone());
                    return push_mem_table_rel(catalog, i, alias, rels);
                }
                if let Some(i) = catalog.view_index_of(name) {
                    let alias = alias.clone().unwrap_or_else(|| name.clone());
                    return push_view_rel(catalog, i, alias, params, rels, ctes);
                }
            }
            err!(TableNotFound)
        }
        FromItem::Parquet { path, alias } => {
            // `parquet('...')` はパスをテーブル名として登録済みであることを期待する。
            // ホストが URL をそのまま名前にして登録するので、SQL からは
            // `FROM parquet('https://…')` と書ける。
            let i = match catalog.index_of(path) {
                Some(i) => i,
                None => err!(TableNotFound),
            };
            push_table_rel(catalog, i, alias.clone().unwrap_or_else(|| path.clone()), rels)
        }
        FromItem::Subquery { query, alias } => {
            // FROM 句の派生表は外側スコープに相関できない（LATERAL 未対応）。
            let plan = bind_query_in(catalog, arena, query, params, ctes, None)?;
            let all = plan.root.schema().to_vec();
            // 別名の無い派生表は修飾して参照できないが、無修飾の参照は通す。
            let alias = alias.clone().unwrap_or_default();
            rels.push(Rel {
                table: None,
                alias,
                needed: (0..all.len()).collect(),
                all,
                subplan: Some(plan.root),
                unnest: None,
            });
            Ok(FromTree::Rel(rels.len() - 1))
        }
        FromItem::Join { left, right, kind, on } => {
            let l = flatten_from(catalog, arena, params, left, rels, ctes, depth + 1)?;
            let r = flatten_from(catalog, arena, params, right, rels, ctes, depth + 1)?;
            Ok(FromTree::Join { left: Box::new(l), right: Box::new(r), kind: *kind, on: *on })
        }
        FromItem::Unnest { expr, alias, column_alias } => {
            // 暗黙 LATERAL: `expr` は左の兄弟がここまでに積んだ列だけを
            // 参照できる。`rels` はここまでの兄弟を左から順に持つので、
            // その時点の `full_scope` がちょうど「見えてよい範囲」になる
            // （`build_tree` が実際に組み立てるときも同じスコープを
            // 再現する。射影プッシュダウンの都合で列の並びは変わりうるが、
            // 列は名前で解決するので影響しない）。
            let scope_so_far = full_scope(rels);
            let ty = compile(arena, &scope_so_far, params, *expr)?.result_ty;
            ensure!(ty == Ty::Json, TypeMismatch);
            let elem_ty = narrow_unnest_elem_ty(arena, &scope_so_far, params, *expr);
            let name = column_alias.clone().unwrap_or_else(|| String::from("unnest"));
            let all = vec![Field::new(name, elem_ty, true)];
            let alias = alias.clone().unwrap_or_default();
            rels.push(Rel {
                table: None,
                alias,
                needed: vec![0],
                all,
                subplan: None,
                unnest: Some(*expr),
            });
            Ok(FromTree::Rel(rels.len() - 1))
        }
        FromItem::GenerateSeries { start, stop, step, inclusive, alias, column_alias } => {
            // `step == 0` は `duckdb` も束縛時エラーにする
            // （"Binder Error: interval cannot be 0!"、`duckdb` CLI で確認済み）。
            ensure!(*step != 0, DivideByZero);
            let name = column_alias.clone().unwrap_or_else(|| {
                String::from(if *inclusive { "generate_series" } else { "range" })
            });
            // `duckdb` の `DESCRIBE` は BIGINT 列を NULL 許容として報告する
            // （実際には NULL を生まないが、宣言上は緩めておく。`Unnest` の
            // 展開要素と同じ判断）。
            let all = vec![Field::new(name, Ty::BigInt, true)];
            let node = Node::GenerateSeries {
                start: *start,
                stop: *stop,
                step: *step,
                inclusive: *inclusive,
                schema: all.clone(),
            };
            let alias = alias.clone().unwrap_or_default();
            rels.push(Rel {
                table: None,
                alias,
                needed: vec![0],
                all,
                subplan: Some(node),
                unnest: None,
            });
            Ok(FromTree::Rel(rels.len() - 1))
        }
    }
}

fn push_table_rel(
    catalog: &Catalog,
    i: usize,
    alias: String,
    rels: &mut Vec<Rel>,
) -> Result<FromTree> {
    let t = match catalog.get(i) {
        Some(t) => t,
        None => err!(TableNotFound),
    };
    // 呼び出し側がスキーマを解決してから来る約束。
    ensure!(t.is_resolved(), Internal);
    rels.push(Rel {
        table: Some(i),
        alias,
        all: t.schema().to_vec(),
        needed: Vec::new(),
        subplan: None,
        unnest: None,
    });
    Ok(FromTree::Rel(rels.len() - 1))
}

/// `catalog::MemTable` を `Rel` として差し込む。ファイルテーブルと違い、
/// スキャンそのものを `Node::MemScan` という「サブプラン」として持たせる
/// （`Rel::subplan` は元々 CTE/派生表用の仕組みだが、そのまま流用できる）。
/// 射影プッシュダウンはしない（全列を出す。`MemScanSpec` のドキュメント参照）。
#[cfg(feature = "ddl")]
fn push_mem_table_rel(
    catalog: &Catalog,
    i: usize,
    alias: String,
    rels: &mut Vec<Rel>,
) -> Result<FromTree> {
    let mt = match catalog.mem_get(i) {
        Some(t) => t,
        None => err!(TableNotFound),
    };
    let all = mt.schema.clone();
    let node = Node::MemScan(Box::new(MemScanSpec { table: i, schema: all.clone() }));
    rels.push(Rel {
        table: None,
        alias,
        needed: (0..all.len()).collect(),
        all,
        subplan: Some(node),
        unnest: None,
    });
    Ok(FromTree::Rel(rels.len() - 1))
}

/// `CREATE VIEW` で登録された生 SQL を再パース・再束縛して `Rel` として
/// 差し込む。ビューは実体を持たないので、参照されるたびに毎回この処理を
/// 行う（キャッシュしない。テーブルが小さく、ビューの再参照も稀な想定の
/// ための単純化）。
#[cfg(feature = "ddl")]
fn push_view_rel(
    catalog: &Catalog,
    i: usize,
    alias: String,
    params: &[Value],
    rels: &mut Vec<Rel>,
    ctes: &mut CteScope,
) -> Result<FromTree> {
    ensure!(ctes.view_depth < MAX_VIEW_DEPTH, ExpressionTooDeep);
    let sql = match catalog.view_get(i) {
        Some(s) => s.to_owned(),
        None => err!(TableNotFound),
    };
    let parsed = crate::sql::parse(&sql)?;
    let q = match parsed.stmt {
        Stmt::Select(q) => q,
        _ => err!(Internal),
    };
    ctes.view_depth += 1;
    // `ctes` を使い回すのは既存の派生表と同じ流儀（`FromItem::Subquery` 参照）。
    // ビュー本体から外側の CTE 名がたまたま同名で見えてしまう可能性はあるが、
    // ビューが CTE 名と衝突する実利用はまず無いので割り切っている。
    let plan = bind_query_in(catalog, &parsed.arena, &q, params, ctes, None);
    ctes.view_depth -= 1;
    let plan = plan?;
    let all = plan.root.schema().to_vec();
    rels.push(Rel {
        table: None,
        alias,
        needed: (0..all.len()).collect(),
        all,
        subplan: Some(plan.root),
        unnest: None,
    });
    Ok(FromTree::Rel(rels.len() - 1))
}

/// 関係の並びから全列のスコープを作る。
fn full_scope(rels: &[Rel]) -> Scope {
    let mut s = Scope::new();
    for r in rels {
        for f in &r.all {
            s.push(qual_of(r), f.clone());
        }
    }
    s
}

/// `needed` を反映した絞り込み後のスコープ。
fn narrow_scope(rels: &[Rel]) -> Scope {
    let mut s = Scope::new();
    for r in rels {
        for &i in &r.needed {
            s.push(qual_of(r), r.all[i].clone());
        }
    }
    s
}

fn qual_of(r: &Rel) -> Option<String> {
    if r.alias.is_empty() {
        None
    } else {
        Some(r.alias.clone())
    }
}

/// 絞り込み後スコープでの、各関係の列範囲 `[start, end)`。
fn rel_ranges(rels: &[Rel]) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(rels.len());
    let mut off = 0;
    for r in rels {
        out.push((off, off + r.needed.len()));
        off += r.needed.len();
    }
    out
}

// --- SAMPLE ------------------------------------------------------------------

/// AST の `sql::ast::SampleSpec` を実行用に解決する。
///
/// 手法（`BERNOULLI`/`SYSTEM`/`RESERVOIR`）の構文は受理するだけで、実装は
/// `is_rows` の 2 択に落ちる（`exec::sample` モジュール doc 参照。タスクの
/// 優先度どおり「パーセント指定 > 行数指定 > 手法の使い分け」という単純化）。
/// パーセント・行数の値域チェックは既に `sql::parser::Parser::sample_amount`
/// が済ませてあるので、ここではシードの解決だけを行う。
fn resolve_sample_spec(spec: &crate::sql::ast::SampleSpec) -> crate::plan::SampleSpec {
    let seed = match spec.seed {
        Some(s) => s as u64,
        None => crate::plan::DEFAULT_SAMPLE_SEED,
    };
    crate::plan::SampleSpec { is_rows: spec.is_rows, amount: spec.amount, seed }
}

// --- 本体 --------------------------------------------------------------------

/// `outer_scope` が `Some` のとき、この SELECT は相関サブクエリとして
/// バインドされる: WHERE の最上位 AND 節にある「外側スコープの式 = 内側の式」
/// という等価述語を相関キーとして検出し、通常の述語処理からは除外した上で
/// `Plan::correlated` に載せて返す（呼び出し側が結合キーとして使う）。
/// `None` のとき（トップレベル・CTE・FROM 句の派生表）は従来どおり。
fn bind_select_in(
    catalog: &Catalog,
    arena: &ExprArena,
    sel: &SelectStmt,
    params: &[Value],
    ctes: &mut CteScope,
    outer_scope: Option<&Scope>,
) -> Result<Plan> {
    let from = match &sel.from {
        Some(f) => f,
        // `SELECT 1` のような FROM 無しは v1 では扱わない。
        None => err!(UnsupportedFeature),
    };

    let mut rels: Vec<Rel> = Vec::new();
    let tree = flatten_from(catalog, arena, params, from, &mut rels, ctes, 0)?;
    let scope_all = full_scope(&rels);

    // --- 参照される列を集める（射影プッシュダウン） -------------------------
    let mut refs: Vec<usize> = Vec::new();
    let mut star_all = false;
    let mut star_quals: Vec<String> = Vec::new();
    for item in &sel.items {
        match arena.get(item.expr) {
            Expr::Star { qualifier: None, replace, .. } => {
                star_all = true;
                // `REPLACE (expr AS col, ...)` の `expr` は式木の外（`*` の
                // 展開の中）にあるので、他の select item と同じく参照列を
                // 拾っておかないと射影プッシュダウンから漏れる。
                // `EXCLUDE` の列自体は逆に読む必要が無いので何もしない。
                for &(e, _) in replace {
                    collect_refs(arena, &scope_all, e, &mut refs)?;
                }
            }
            Expr::Star { qualifier: Some(q), replace, .. } => {
                ensure!(scope_all.has_qualifier(q), TableNotFound);
                star_quals.push(q.clone());
                for &(e, _) in replace {
                    collect_refs(arena, &scope_all, e, &mut refs)?;
                }
            }
            _ => collect_refs(arena, &scope_all, item.expr, &mut refs)?,
        }
    }
    // 名前付きウィンドウ（`OVER w`）は `PARTITION BY`/`ORDER BY` の実体が
    // 式木の外（`sel.windows`）にあるため、通常の再帰では見つからない。
    // 実際に参照されている名前だけを拾って、その定義の列参照も射影
    // プッシュダウン対象に加える（未定義の名前はここではエラーにしない。
    // 後段の `build_window` が実際の束縛時に明確なエラーを出す）。
    if !sel.windows.is_empty() {
        let mut win_refs: Vec<ExprId> = Vec::new();
        for item in &sel.items {
            collect_windows(arena, item.expr, &mut win_refs, 0)?;
        }
        for o in &sel.order_by {
            collect_windows(arena, o.expr, &mut win_refs, 0)?;
        }
        if let Some(q) = sel.qualify {
            collect_windows(arena, q, &mut win_refs, 0)?;
        }
        for &w in &win_refs {
            let Expr::Window { window_ref: Some(wname), .. } = arena.get(w) else { continue };
            let Some((_, def)) =
                sel.windows.iter().find(|(n, _)| eq_ascii_ci(n.as_bytes(), wname.as_bytes()))
            else {
                continue;
            };
            for &p in &def.partition_by {
                collect_refs(arena, &scope_all, p, &mut refs)?;
            }
            for o in &def.order_by {
                collect_refs(arena, &scope_all, o.expr, &mut refs)?;
            }
        }
    }
    for e in [sel.filter, sel.having].into_iter().flatten() {
        // 相関サブクエリでは WHERE に外側スコープの列参照が混じりうる。
        // ここでは射影プッシュダウン用の列集めをしているだけなので、
        // 外側スコープで解決できる参照は黙ってプッシュダウン対象から外す
        // （相関等価述語の抽出自体は後段の WHERE 分解で行う）。
        collect_refs_tolerant(arena, &scope_all, outer_scope, e, &mut refs)?;
    }
    for e in &sel.group_by {
        // 序数指定（`GROUP BY 1`）はここでは列参照を持たない。
        if ordinal_of(arena, *e).is_none() {
            collect_refs(arena, &scope_all, *e, &mut refs)?;
        }
    }
    // `GROUPING SETS`/`ROLLUP`/`CUBE` も同様に射影プッシュダウンの対象に含める。
    if let Some(sets) = &sel.grouping_sets {
        for set in sets {
            for &e in set {
                if ordinal_of(arena, e).is_none() {
                    collect_refs(arena, &scope_all, e, &mut refs)?;
                }
            }
        }
    }
    for o in &sel.order_by {
        // ORDER BY は出力別名も指しうるので、解決できなくてもここでは許す。
        let _ = collect_refs(arena, &scope_all, o.expr, &mut refs);
    }
    if let Some(q) = sel.qualify {
        // QUALIFY も SELECT の出力別名を指しうるので、解決できなくてもここでは許す。
        let _ = collect_refs(arena, &scope_all, q, &mut refs);
    }
    for &e in &sel.distinct_on {
        // DISTINCT ON も出力別名を指しうるので、解決できなくてもここでは許す。
        let _ = collect_refs(arena, &scope_all, e, &mut refs);
    }
    collect_join_refs(arena, &scope_all, &tree, &mut refs)?;
    // FROM 句の `UNNEST(expr) AS ...` の `expr` は `FromTree` の一部ではなく
    // `Rel::unnest` に別置きなので、`collect_join_refs` の走査には乗らない。
    // ここで拾わないと、参照される列（`t.tags` の `tags` など）が
    // 射影プッシュダウンから漏れて `Scan` が読まなくなる。
    for r in rels.iter() {
        if let Some(u) = r.unnest {
            collect_refs(arena, &scope_all, u, &mut refs)?;
        }
    }

    if star_all {
        refs.extend(0..scope_all.len());
    }
    for q in &star_quals {
        refs.extend(scope_all.indices_for_qualifier(q));
    }
    refs.sort_unstable();
    refs.dedup();

    // グローバル添字を関係ごとに割り振る。
    {
        let mut off = 0usize;
        for r in rels.iter_mut() {
            let end = off + r.all.len();
            if r.subplan.is_none() {
                r.needed =
                    refs.iter().filter(|&&g| g >= off && g < end).map(|&g| g - off).collect();
                // 1 列も参照されない関係でも行数は要る（`COUNT(*)`）。最も安い列を
                // 1 本だけ読む。空の射影を渡すと、行指向フォーマットは行数を
                // 返す手段が無くなる。
                if r.needed.is_empty() && !r.all.is_empty() {
                    r.needed.push(0);
                }
            }
            off = end;
        }
    }

    let scope = narrow_scope(&rels);
    let ranges = rel_ranges(&rels);

    // --- WHERE の分解と押し下げ ---------------------------------------------
    // 外部結合があると、片側だけの条件を先に適用すると NULL 補完の結果が
    // 変わってしまう。安全側に倒して押し下げない。
    let pushdown_ok = !tree.has_outer_join();
    let mut conjuncts = Vec::new();
    if let Some(w) = sel.filter {
        split_conjuncts(arena, w, &mut conjuncts, 0)?;
    }
    let mut per_rel: Vec<Vec<ExprId>> = (0..rels.len()).map(|_| Vec::new()).collect();
    let mut leftover: Vec<ExprId> = Vec::new();
    // `EXISTS` / `IN (SELECT)` は半結合に書き換えるので、通常の述語とは分ける。
    let mut semijoins: Vec<ExprId> = Vec::new();
    // 相関等価述語: (内側の式, 外側スコープの式) の組。相関サブクエリを
    // 結合キー付きの結合に書き換えるために使う（下の「相関サブクエリ」節）。
    let mut correlated_eq: Vec<(ExprId, ExprId)> = Vec::new();
    for c in conjuncts {
        if is_semijoin_predicate(arena, c) {
            semijoins.push(c);
            continue;
        }
        if let Some(os) = outer_scope {
            if let ConjClass::Correlated { inner, outer } =
                classify_conjunct(arena, &scope_all, os, c)?
            {
                correlated_eq.push((inner, outer));
                continue;
            }
        }
        // サブクエリを含む述語は押し下げない。参照する関係を正しく数えられず、
        // 誤った側に落としてしまうため。
        let owner = if pushdown_ok && !contains_subquery(arena, c, 0) {
            single_rel_of(arena, &scope, &ranges, c)?
        } else {
            None
        };
        match owner {
            Some(i) => per_rel[i].push(c),
            None => leftover.push(c),
        }
    }
    if outer_scope.is_some() && !correlated_eq.is_empty() {
        // 相関キーは「先頭の 1 行/グループを選ぶ」処理（DistinctOn か、
        // 集約なら GROUP BY への合流）を経由するため、相関サブクエリ自身が
        // 独自の ORDER BY/LIMIT/QUALIFY/ウィンドウ関数を持つと、その意味が
        // 「相関キーごと」ではなく「デコリレーション後の全体」に対して
        // 適用されてしまい結果が変わる。安全側に倒して明確に拒否する。
        ensure!(
            sel.limit.is_none()
                && sel.offset.is_none()
                && sel.order_by.is_empty()
                && sel.qualify.is_none(),
            UnsupportedFeature
        );
        let mut win_probe = Vec::new();
        for item in &sel.items {
            collect_windows(arena, item.expr, &mut win_probe, 0)?;
        }
        ensure!(win_probe.is_empty(), UnsupportedFeature);
    }

    // --- スキャンと結合を組み立てる -----------------------------------------
    let mut node = build_tree(arena, &scope, &ranges, &mut rels, &tree, params, &per_rel, 0)?;

    // --- SAMPLE --------------------------------------------------------------
    // `duckdb` CLI で確認した意味論: `USING SAMPLE`/`TABLESAMPLE` は結合後・
    // フィルタ前の FROM 句の生データに効く（`a JOIN b USING SAMPLE 20 ROWS`
    // は 100 行の結合結果から 20 行選ぶ、`WHERE` を先に書いても行数の実測は
    // 変わらない）。ここで `WHERE`（`sel.filter`。下の「WHERE の適用」節）より
    // 前に挟むのはそのため。
    //
    // ただし単一テーブルに対する `WHERE` の一部は既に `per_rel` として
    // `build_tree` 内の `Node::Scan` 直後まで押し下げ済み（射影プッシュダウンと
    // 同時に行う設計、モジュール冒頭ドキュメント参照）なので、その部分だけは
    // 厳密には duckdb と逆順（フィルタ→サンプル）になる。単純化のための
    // 割り切り（タスクの優先度: パーセント指定 > 行数指定 > 手法の使い分け、
    // に倣い、フィルタとの相互作用の完全な一致までは求めない）。
    if let Some(spec) = &sel.sample {
        node = Node::Sample { input: Box::new(node), spec: resolve_sample_spec(spec) };
    }

    // --- サブクエリ ---------------------------------------------------------
    // 相関の無いサブクエリは結合に書き換えられる。スカラサブクエリは
    // 「1 行だけの表」との LEFT 結合、`EXISTS` / `IN` は半結合・反結合。
    let mut scope = scope;
    let mut subs: Vec<Substitution> = Vec::new();
    let mut scalars: Vec<ExprId> = Vec::new();
    for item in &sel.items {
        collect_scalar_subqueries(arena, item.expr, &mut scalars, 0)?;
    }
    for e in [sel.filter, sel.having].into_iter().flatten() {
        collect_scalar_subqueries(arena, e, &mut scalars, 0)?;
    }
    for e in &sel.group_by {
        collect_scalar_subqueries(arena, *e, &mut scalars, 0)?;
    }
    for o in &sel.order_by {
        collect_scalar_subqueries(arena, o.expr, &mut scalars, 0)?;
    }
    for (n, id) in scalars.iter().enumerate() {
        let q = match arena.get(*id) {
            Expr::ScalarSubquery(q) => q,
            _ => err!(Internal),
        };
        let plan = bind_query_in(catalog, arena, q, params, ctes, Some(&scope))?;
        let k = plan.correlated.len();
        // スカラサブクエリは（相関キー列を除いて）1 列でなければならない。
        ensure!(plan.root.schema().len() - k == 1, TypeMismatch);
        let ty = plan.root.schema()[0].ty;
        let mut label = String::from("subq");
        push_u32(&mut label, n as u32);

        let mut right = plan.root;
        let mut left_keys = Vec::new();
        let mut right_keys = Vec::new();
        if k == 0 {
            // 非相関: 1 行に絞ってから LEFT 結合する。空集合なら NULL になる。
            // 2 行以上返る場合、SQL 標準はエラーだがここでは先頭を採る（制限）。
            right = Node::Limit { input: Box::new(right), limit: Some(1), offset: 0 };
        } else {
            // 相関: 外側の行ごとに異なる相関キー値を持つので、右側全体を
            // 1 行に絞る（＝ブランケットな LIMIT 1）と他の外側行が軒並み
            // NULL になってしまう。相関キーの値ごとに「先頭の 1 行」に絞って
            // から、そのキーで LEFT 結合する（非相関版の「複数行なら先頭を
            // 採る」制限を相関キー単位に一般化したもの）。集約を経由した
            // 相関の場合は呼び出し元で既に GROUP BY 済みで 1 行/キーなので、
            // ここでの DistinctOn は実質的な no-op になる。
            let corr_scope = Scope::from_fields(right.schema().to_vec());
            let mut dkeys = Vec::with_capacity(k);
            for i in 0..k {
                dkeys.push(column_program(&corr_scope, 1 + i)?);
            }
            right = Node::DistinctOn { input: Box::new(right), keys: dkeys };
            for (i, &outer_e) in plan.correlated.iter().enumerate() {
                let lp = compile(arena, &scope, params, outer_e)?;
                let rp = column_program(&corr_scope, 1 + i)?;
                let want = Ty::unify_or_mismatch(lp.result_ty, rp.result_ty)?;
                left_keys.push(cast_program(lp, want)?);
                right_keys.push(cast_program(rp, want)?);
            }
        }

        let col = scope.len();
        let mut full_schema = node.schema().to_vec();
        full_schema.extend_from_slice(right.schema());
        node = Node::Join {
            left: Box::new(node),
            right: Box::new(right),
            kind: JoinKind::Left,
            left_keys,
            right_keys,
            residual: None,
            schema: full_schema,
        };
        if k > 0 {
            // 結合キーとして使い終えた相関キー列は落とす。
            node = drop_trailing_columns(node, k)?;
            // 相関キーに一致する内側行が 1 行も無い外側行は、この LEFT JOIN で
            // 値列が NULL になる。`count`/`count(*)` の相関スカラサブクエリ
            // だけは「0 行を集約 → 0」であるべき（DuckDB で確認済み）なので、
            // ここで NULL を 0 に補正する。
            if matches!(as_bare_aggregate(arena, q), Some(AggKind::Count | AggKind::CountStar)) {
                node = coalesce_count_column(node, col)?;
            }
        }
        scope.push(None, Field::new(label.clone(), ty, true));
        subs.push(Substitution { expr: *id, column: col, structural: false });
    }

    for c in semijoins {
        node = build_semijoin(catalog, arena, params, ctes, node, &scope, &subs, c)?;
    }

    if !leftover.is_empty() {
        let pred = and_all(arena, &scope, params, &subs, &leftover)?;
        node = Node::Filter { input: Box::new(node), pred };
    }

    // --- 相関を伴う集約スカラサブクエリ（限定パターン） ---------------------
    // 「裸の集約呼び出し 1 つだけを SELECT する」という限定形のみ、ここで
    // 完結させる（マジックデコリレーションの基本形: 相関キーで GROUP BY
    // した集約を組み立ててから、呼び出し元が相関キーで LEFT/SEMI/ANTI 結合
    // する）。この関数の残り（670 行以降の通常の集約セクション、GROUPING
    // SETS/ROLLUP/CUBE 実装と競合するため触れない）を経由せずに早期リターン
    // する。それ以外の組み合わせ（集約サブクエリが独自の GROUP BY/HAVING/
    // QUALIFY/DISTINCT/ウィンドウ関数を持つ場合など）は、相関キーを
    // 集約のグルーピングへ合流させる手段が無いため、明確に拒否する
    // （黙って相関述語を無視すると集約が外側行をまたいで混ざり、誤った
    // 結果になってしまう）。
    if outer_scope.is_some() && !correlated_eq.is_empty() {
        let mut corr_agg_probe: Vec<ExprId> = Vec::new();
        for item in &sel.items {
            collect_aggregates(arena, item.expr, &mut corr_agg_probe, 0)?;
        }
        if let Some(h) = sel.having {
            collect_aggregates(arena, h, &mut corr_agg_probe, 0)?;
        }
        let will_aggregate =
            !corr_agg_probe.is_empty() || !sel.group_by.is_empty() || sel.grouping_sets.is_some();
        if will_aggregate {
            ensure!(
                sel.items.len() == 1
                    && corr_agg_probe.len() == 1
                    && sel.items[0].expr == corr_agg_probe[0]
                    && sel.group_by.is_empty()
                    && sel.grouping_sets.is_none()
                    && sel.having.is_none()
                    && sel.qualify.is_none()
                    && !sel.distinct
                    && sel.distinct_on.is_empty(),
                UnsupportedFeature
            );
            let k = correlated_eq.len();
            let mut groups = Vec::with_capacity(k);
            let mut out_fields = Vec::with_capacity(k + 1);
            for &(inner_e, _) in &correlated_eq {
                let p = compile(arena, &scope, params, inner_e)?;
                out_fields.push(Field::new(String::new(), p.result_ty, true));
                groups.push(p);
            }
            let a = build_agg(arena, &scope, params, corr_agg_probe[0])?;
            out_fields.push(Field::new(a.name.clone(), a.result_ty()?, true));
            let agg_node = Node::Aggregate {
                input: Box::new(node),
                groups,
                aggs: vec![a],
                schema: out_fields,
                having: None,
            };
            // 「ある相関キー値に一致する内側行が 1 行も無い」場合は、GROUP BY
            // の結果自体にその組が現れない（NULL 値の行ではなく、行そのものが
            // 無い）。これは呼び出し側（スカラサブクエリなら LEFT JOIN）で
            // 相関キーが一致せず NULL になる形で表れるので、count 系の
            // 「0 行を集約 → 0」への補正はここではなく呼び出し側で行う
            // （`as_bare_aggregate` 参照）。
            // 呼び出し側の規約（index 0 = 値, 1.. = 相関キー）に列順を揃える。
            let s = Scope::from_fields(agg_node.schema().to_vec());
            let mut exprs = Vec::with_capacity(k + 1);
            let mut out_schema = Vec::with_capacity(k + 1);
            exprs.push(column_program(&s, k)?);
            out_schema.push(s.fields()[k].clone());
            for i in 0..k {
                exprs.push(column_program(&s, i)?);
                out_schema.push(s.fields()[i].clone());
            }
            let root = Node::Project { input: Box::new(agg_node), exprs, schema: out_schema };
            return Ok(Plan { root, correlated: correlated_eq.iter().map(|&(_, o)| o).collect() });
        }
    }

    // --- UNNEST（SELECT リスト） ---------------------------------------------
    // `FILTER`/`QUALIFY` と同様、集約でも通常のスカラ式でもない特殊な式として
    // 拾う。対象列の JSON 配列を要素数ぶんの行に展開して他の列を複製するため
    // 「行を増やす」効果があり、集約より前（FROM/WHERE の直後、GROUP BY の前）
    // に割り込ませる。展開後の行に対して集約する意味論は実装していないので
    // 集約とは同時に使えない。複数の `UNNEST` を同じ SELECT リストに書いた
    // ときの DuckDB の挙動（列ごとの zip、要素数が違えば NULL 埋め）は複雑
    // すぎるためスコープ外とし、1 個だけ許して残りは明確に拒否する。
    let mut unnest_calls: Vec<ExprId> = Vec::new();
    for item in &sel.items {
        collect_unnests(arena, item.expr, &mut unnest_calls, 0)?;
    }
    if !unnest_calls.is_empty() {
        // 相関サブクエリ内での UNNEST（デコリレーションとの相互作用が
        // 複雑になる）は対象外として明確に拒否する。
        ensure!(outer_scope.is_none(), UnsupportedFeature);
        ensure!(unnest_calls.len() == 1, UnsupportedFeature);
        for e in [sel.filter, sel.having, sel.qualify].into_iter().flatten() {
            let mut probe = Vec::new();
            collect_unnests(arena, e, &mut probe, 0)?;
            ensure!(probe.is_empty(), UnsupportedFeature);
        }
        for o in &sel.order_by {
            let mut probe = Vec::new();
            collect_unnests(arena, o.expr, &mut probe, 0)?;
            ensure!(probe.is_empty(), UnsupportedFeature);
        }
        ensure!(
            sel.group_by.is_empty() && sel.grouping_sets.is_none() && sel.having.is_none(),
            UnsupportedFeature
        );
        let mut agg_probe: Vec<ExprId> = Vec::new();
        for item in &sel.items {
            collect_aggregates(arena, item.expr, &mut agg_probe, 0)?;
        }
        ensure!(agg_probe.is_empty(), UnsupportedFeature);

        let unnest_id = unnest_calls[0];
        let arg = match arena.get(unnest_id) {
            Expr::Unnest(a) => *a,
            _ => err!(Internal),
        };
        let prog = compile(arena, &scope, params, arg)?;
        ensure!(prog.result_ty == Ty::Json, TypeMismatch);
        let elem_ty = narrow_unnest_elem_ty(arena, &scope, params, arg);
        let elem_field = Field::new(default_name(arena, unnest_id), elem_ty, true);
        let mut out_schema = scope.fields().to_vec();
        out_schema.push(elem_field);
        node =
            Node::Unnest { input: Box::new(node), expr: prog, elem_ty, schema: out_schema.clone() };
        scope = Scope::from_fields(out_schema);
        subs.push(Substitution { expr: unnest_id, column: scope.len() - 1, structural: false });
    }

    // --- 集約 ---------------------------------------------------------------
    let mut agg_calls: Vec<ExprId> = Vec::new();
    for item in &sel.items {
        collect_aggregates(arena, item.expr, &mut agg_calls, 0)?;
    }
    if let Some(h) = sel.having {
        collect_aggregates(arena, h, &mut agg_calls, 0)?;
    }
    for o in &sel.order_by {
        collect_aggregates(arena, o.expr, &mut agg_calls, 0)?;
    }
    // WHERE に集約は書けない（HAVING がそのためにある）。
    if let Some(w) = sel.filter {
        let mut in_where = Vec::new();
        collect_aggregates(arena, w, &mut in_where, 0)?;
        ensure!(in_where.is_empty(), NotAggregate);
    }

    // `GROUPING`/`GROUPING_ID` 呼び出しを集める。集約と違い引数は評価せず、
    // グルーピングセットごとに定まる定数（ビットマスク）に置き換える。
    let mut grouping_calls: Vec<ExprId> = Vec::new();
    for item in &sel.items {
        collect_grouping_calls(arena, item.expr, &mut grouping_calls, 0)?;
    }
    if let Some(h) = sel.having {
        collect_grouping_calls(arena, h, &mut grouping_calls, 0)?;
    }
    for o in &sel.order_by {
        collect_grouping_calls(arena, o.expr, &mut grouping_calls, 0)?;
    }

    let aggregating =
        !agg_calls.is_empty() || !sel.group_by.is_empty() || sel.grouping_sets.is_some();
    // GROUPING() は集約の外では意味を持たない。
    ensure!(aggregating || grouping_calls.is_empty(), NotAggregate);
    let mut item_scope = scope.clone();

    if aggregating && sel.grouping_sets.is_none() && grouping_calls.is_empty() {
        // 単純な `GROUP BY a, b, ...`（GROUPING SETS 系拡張なし）。
        // 既存の 1 本の Node::Aggregate に havings を直接埋め込む従来経路。
        let mut group_exprs = Vec::new();
        for g in &sel.group_by {
            group_exprs.push(resolve_select_ref(arena, sel, *g)?);
        }

        let mut groups = Vec::new();
        let mut out_fields = Vec::new();
        for (i, &g) in group_exprs.iter().enumerate() {
            let p = compile(arena, &scope, params, g)?;
            out_fields.push(Field::new(group_name(arena, g, i), p.result_ty, true));
            subs.push(Substitution { expr: g, column: i, structural: true });
            groups.push(p);
        }

        let ngroups = groups.len();
        let mut aggs = Vec::new();
        for (j, &call) in agg_calls.iter().enumerate() {
            let a = build_agg(arena, &scope, params, call)?;
            out_fields.push(Field::new(a.name.clone(), a.result_ty()?, true));
            subs.push(Substitution { expr: call, column: ngroups + j, structural: true });
            aggs.push(a);
        }

        // GROUP BY に無い裸の列参照を弾く。集約の外に出た列は値が定まらない。
        for item in &sel.items {
            check_grouped(arena, &scope, item.expr, &group_exprs, &agg_calls, 0)?;
        }
        if let Some(h) = sel.having {
            check_grouped(arena, &scope, h, &group_exprs, &agg_calls, 0)?;
        }

        let agg_scope = Scope::from_fields(out_fields.clone());
        let having = match sel.having {
            Some(h) => Some(compile_predicate_with_subs(arena, &agg_scope, params, &subs, h)?),
            None => None,
        };
        node = Node::Aggregate { input: Box::new(node), groups, aggs, schema: out_fields, having };
        item_scope = agg_scope;
    } else if aggregating {
        // `GROUPING SETS`/`ROLLUP`/`CUBE`、あるいは単純な GROUP BY に
        // `GROUPING()` が付いたケース。単純な GROUP BY も「グルーピングセットが
        // 1 個だけ」として同じ経路で扱う。
        //
        // グルーピングセットの数だけ Node::Aggregate を作り、UNION ALL
        // （Node::SetOp）で束ねる。あるセットに含まれない列は NULL 定数で
        // 埋める（DuckDB と同じ挙動）。FROM/WHERE 側の入力はどのセットでも
        // 同じなので、`node`（この時点で WHERE まで適用済み）をセットの数だけ
        // 複製する。
        let sets: Vec<Vec<ExprId>> = match &sel.grouping_sets {
            Some(sets) => sets.clone(),
            None => vec![sel.group_by.clone()],
        };
        ensure!(!sets.is_empty(), Internal);

        // グルーピング列の全セットにわたる和集合を「グルーピング列」として扱う
        // （構造的に等しい列は 1 本にまとめる）。GROUP BY 序数・別名も
        // 通常の GROUP BY と同じく解決する。
        let mut resolved_sets: Vec<Vec<ExprId>> = Vec::with_capacity(sets.len());
        let mut group_exprs: Vec<ExprId> = Vec::new();
        for set in &sets {
            let mut rs = Vec::with_capacity(set.len());
            for &g in set {
                let r = resolve_select_ref(arena, sel, g)?;
                if !group_exprs.iter().any(|&e| expr_eq(arena, e, r)) {
                    group_exprs.push(r);
                }
                rs.push(r);
            }
            resolved_sets.push(rs);
        }

        // 列ごとに 1 度だけコンパイルする。入力スコープ（`scope`）はどの
        // セットでも同じなので、そのままセット間で使い回せる（`Program` は
        // `Clone`）。
        let mut group_progs: Vec<Program> = Vec::with_capacity(group_exprs.len());
        for &g in &group_exprs {
            group_progs.push(compile(arena, &scope, params, g)?);
        }
        let ngroups = group_exprs.len();
        for (i, &g) in group_exprs.iter().enumerate() {
            subs.push(Substitution { expr: g, column: i, structural: true });
        }

        let mut out_fields: Vec<Field> = Vec::with_capacity(ngroups + agg_calls.len());
        for (i, &g) in group_exprs.iter().enumerate() {
            out_fields.push(Field::new(group_name(arena, g, i), group_progs[i].result_ty, true));
        }

        // 集約も入力スコープが共通なので 1 度だけ組み立て、セットごとに複製する。
        let mut aggs: Vec<Agg> = Vec::with_capacity(agg_calls.len());
        for (j, &call) in agg_calls.iter().enumerate() {
            let a = build_agg(arena, &scope, params, call)?;
            out_fields.push(Field::new(a.name.clone(), a.result_ty()?, true));
            subs.push(Substitution { expr: call, column: ngroups + j, structural: true });
            aggs.push(a);
        }
        let base_cols = ngroups + agg_calls.len();

        // GROUP BY に無い裸の列参照を弾く。GROUPING SETS では全セットの和集合が
        // 「グルーピング列」なので、あるセットに無い列を SELECT で裸参照しても
        // エラーにはしない（その行では NULL になるだけ。DuckDB と同じ）。
        for item in &sel.items {
            check_grouped(arena, &scope, item.expr, &group_exprs, &agg_calls, 0)?;
        }
        if let Some(h) = sel.having {
            check_grouped(arena, &scope, h, &group_exprs, &agg_calls, 0)?;
        }

        // GROUPING()/GROUPING_ID() の引数はグルーピング列でなければならない。
        // 引数がどの列（`group_exprs` の添字）を指すかだけ覚えておき、値は
        // セットごとに計算する（先頭引数が最上位ビット、DuckDB と同じ並び）。
        let mut grouping_arg_idx: Vec<Vec<usize>> = Vec::with_capacity(grouping_calls.len());
        for &gc in &grouping_calls {
            let args = match arena.get(gc) {
                Expr::Function { args, .. } => args.clone(),
                _ => err!(Internal),
            };
            ensure!(!args.is_empty(), WrongArgCount);
            let mut idxs = Vec::with_capacity(args.len());
            for a in args {
                let r = resolve_select_ref(arena, sel, a)?;
                let pos = group_exprs.iter().position(|&g| expr_eq(arena, g, r));
                idxs.push(match pos {
                    Some(p) => p,
                    None => err!(NotGrouped),
                });
            }
            grouping_arg_idx.push(idxs);
        }
        for (k, &gc) in grouping_calls.iter().enumerate() {
            out_fields.push(Field::new(default_name(arena, gc), Ty::BigInt, false));
            subs.push(Substitution { expr: gc, column: base_cols + k, structural: true });
        }

        // HAVING はグルーピング列・集約結果・GROUPING() 定数がすべて揃った
        // 最終スキーマに対して評価する。各セットの Aggregate には埋め込まず、
        // UNION ALL で束ねた後にまとめて 1 回だけ Filter として掛ける
        // （行ごとの評価なのでどちらでも結果は同じ）。
        let agg_scope = Scope::from_fields(out_fields.clone());
        let having = match sel.having {
            Some(h) => Some(compile_predicate_with_subs(arena, &agg_scope, params, &subs, h)?),
            None => None,
        };

        let base_fields: Vec<Field> = out_fields[..base_cols].to_vec();
        let mut branches: Vec<Node> = Vec::with_capacity(resolved_sets.len());
        for set in &resolved_sets {
            let mut set_groups = Vec::with_capacity(ngroups);
            for (i, &g) in group_exprs.iter().enumerate() {
                if set.iter().any(|&s| expr_eq(arena, s, g)) {
                    set_groups.push(group_progs[i].clone());
                } else {
                    set_groups.push(const_program(group_progs[i].result_ty, Value::Null));
                }
            }
            let agg_node = Node::Aggregate {
                input: Box::new(node.clone()),
                groups: set_groups,
                aggs: aggs.clone(),
                schema: base_fields.clone(),
                having: None,
            };
            let branch = if grouping_calls.is_empty() {
                agg_node
            } else {
                // GROUPING() の結果はグループキー・集約結果ではないので
                // Node::Aggregate のスキーマには乗らない。Project で定数列として
                // 追加で載せる。
                let branch_scope = Scope::from_fields(base_fields.clone());
                let mut exprs = Vec::with_capacity(out_fields.len());
                for i in 0..base_cols {
                    exprs.push(column_program(&branch_scope, i)?);
                }
                for idxs in &grouping_arg_idx {
                    let bits = idxs.len() as u32;
                    let mut v: i64 = 0;
                    for (bit_pos, &gi) in idxs.iter().enumerate() {
                        let in_set = set.iter().any(|&s| expr_eq(arena, s, group_exprs[gi]));
                        if !in_set {
                            v |= 1i64 << (bits - 1 - bit_pos as u32);
                        }
                    }
                    exprs.push(const_program(Ty::BigInt, Value::I64(v)));
                }
                Node::Project { input: Box::new(agg_node), exprs, schema: out_fields.clone() }
            };
            branches.push(branch);
        }

        let mut iter = branches.into_iter();
        // `sets` を空にしていない（上の ensure! 参照）ので必ず 1 個は取れる。
        let mut combined = iter.next().unwrap();
        for b in iter {
            combined = Node::SetOp {
                left: Box::new(combined),
                right: Box::new(b),
                op: SetOpKind::Union,
                all: true,
                schema: out_fields.clone(),
            };
        }
        node = match having {
            Some(pred) => Node::Filter { input: Box::new(combined), pred },
            None => combined,
        };
        item_scope = agg_scope;
    } else {
        ensure!(sel.having.is_none(), NotAggregate);
    }

    // --- ウィンドウ関数 -----------------------------------------------------
    // 集約の**後**に評価する。`sum(x) OVER ()` の x は集約結果を指しうるので、
    // 入力スコープは集約の出力（集約が無ければスキャン/結合の出力）になる。
    let mut win_calls: Vec<ExprId> = Vec::new();
    for item in &sel.items {
        collect_windows(arena, item.expr, &mut win_calls, 0)?;
    }
    for o in &sel.order_by {
        collect_windows(arena, o.expr, &mut win_calls, 0)?;
    }
    if let Some(q) = sel.qualify {
        // QUALIFY はウィンドウ関数の結果を直接書ける唯一の SELECT 外の場所。
        collect_windows(arena, q, &mut win_calls, 0)?;
    }
    // WHERE と HAVING にウィンドウ関数は書けない（評価順が後なので）。
    for e in [sel.filter, sel.having].into_iter().flatten() {
        let mut found = Vec::new();
        collect_windows(arena, e, &mut found, 0)?;
        ensure!(found.is_empty(), UnsupportedFeature);
    }

    if !win_calls.is_empty() {
        let base = item_scope.len();
        let mut fields = item_scope.fields().to_vec();
        let mut specs = Vec::with_capacity(win_calls.len());
        for (j, &w) in win_calls.iter().enumerate() {
            let spec = build_window(arena, &item_scope, params, &subs, &sel.windows, w)?;
            fields.push(Field::new(spec.name.clone(), spec.result_ty, true));
            subs.push(Substitution { expr: w, column: base + j, structural: true });
            specs.push(spec);
        }
        node = Node::Window { input: Box::new(node), windows: specs, schema: fields.clone() };
        // ウィンドウ出力は「入力の列 ++ ウィンドウ列」なので、既存の列番号は
        // ずれない。集約の置き換えもそのまま使える。
        item_scope = Scope::from_fields(fields);
    }

    // --- QUALIFY --------------------------------------------------------------
    // GROUP BY/HAVING の後・ORDER BY の前、ウィンドウ関数の結果に対して
    // 効くフィルタ。`item_scope`（集約・ウィンドウの出力）に対して評価する。
    //
    // QUALIFY は SELECT の出力別名（`... AS rn ... QUALIFY rn = 1`）も指せる。
    // 別名の実体（GROUP BY 式・集約呼び出し・ウィンドウ呼び出し）は既に
    // `subs` に列番号として登録済みなので、QUALIFY 側で見つけた同名の
    // 裸の列参照ノードにも同じ列番号を指す `Substitution` を追加で登録する
    // だけで済む（式木は複製せず、既存の置き換え機構にそのまま乗せる）。
    if let Some(q) = sel.qualify {
        let mut alias_refs = Vec::new();
        collect_unqualified_colrefs(arena, q, &mut alias_refs, 0)?;
        for (rid, rname) in alias_refs {
            let hit = sel.items.iter().find(|it| {
                it.alias.as_deref().is_some_and(|a| eq_ascii_ci(a.as_bytes(), rname.as_bytes()))
            });
            let Some(item) = hit else { continue };
            let col = subs
                .iter()
                .rev()
                .find(|s| {
                    if s.structural {
                        expr_eq(arena, s.expr, item.expr)
                    } else {
                        s.expr == item.expr
                    }
                })
                .map(|s| s.column)
                .or_else(|| match arena.get(item.expr) {
                    Expr::ColumnRef { qualifier, name } => {
                        item_scope.resolve(qualifier.as_deref(), name).ok()
                    }
                    _ => None,
                });
            if let Some(col) = col {
                subs.push(Substitution { expr: rid, column: col, structural: false });
            }
        }
        let pred = compile_predicate_with_subs(arena, &item_scope, params, &subs, q)?;
        node = Node::Filter { input: Box::new(node), pred };
    }

    // --- 射影 ---------------------------------------------------------------
    let mut exprs = Vec::new();
    let mut schema = Vec::new();
    for item in &sel.items {
        match arena.get(item.expr) {
            Expr::Star { qualifier, exclude, replace } => {
                // 集約後に `*` は展開できない（元の行が残っていない）。
                ensure!(!aggregating, NotGrouped);
                let idx: Vec<usize> = match qualifier {
                    Some(q) => scope.indices_for_qualifier(q),
                    None => (0..scope.len()).collect(),
                };
                // `EXCLUDE`/`REPLACE` に書かれた列名が実在するかを検証する
                // （`duckdb` は "Column ... not found" として束縛時に拒否する）。
                for name in exclude {
                    ensure!(
                        idx.iter().any(|&i| eq_ascii_ci(
                            scope.fields()[i].name.as_bytes(),
                            name.as_bytes()
                        )),
                        ColumnNotFound
                    );
                }
                for (_, name) in replace {
                    ensure!(
                        idx.iter().any(|&i| eq_ascii_ci(
                            scope.fields()[i].name.as_bytes(),
                            name.as_bytes()
                        )),
                        ColumnNotFound
                    );
                }
                for i in idx {
                    let fname = scope.fields()[i].name.clone();
                    if exclude.iter().any(|e| eq_ascii_ci(e.as_bytes(), fname.as_bytes())) {
                        continue;
                    }
                    let rexpr: Option<ExprId> = replace
                        .iter()
                        .find(|(_, n)| eq_ascii_ci(n.as_bytes(), fname.as_bytes()))
                        .map(|&(e, _)| e);
                    match rexpr {
                        Some(rexpr) => {
                            // REPLACE の式は通常の select item と同じスコープ
                            // （集約・ウィンドウ出力を含みうる `item_scope`）で
                            // コンパイルする。列名自体は元のまま変えない。
                            let p = compile_with_subs(arena, &item_scope, params, &subs, rexpr)?;
                            schema.push(Field::new(fname, p.result_ty, true));
                            exprs.push(p);
                        }
                        None => {
                            exprs.push(column_program(&scope, i)?);
                            schema.push(scope.fields()[i].clone());
                        }
                    }
                }
            }
            _ => {
                let p = compile_with_subs(arena, &item_scope, params, &subs, item.expr)?;
                let name = match &item.alias {
                    Some(a) => a.clone(),
                    None => default_name(arena, item.expr),
                };
                schema.push(Field::new(name, p.result_ty, true));
                exprs.push(p);
            }
        }
    }
    ensure!(!exprs.is_empty(), SyntaxError);
    // 相関キー列を出力の末尾に付加する（非集約の場合。集約を伴う相関は
    // 上の早期リターン経路で完結しているのでここには来ない）。呼び出し側
    // （相関スカラサブクエリ / `EXISTS` / `IN` の束縛）が結合キーとして
    // 使い、`Plan::correlated` の個数分だけ末尾から読む。DISTINCT や
    // ORDER BY の隠し列と同じ「`visible` を超えた分は最後に落とす」機構に
    // 相乗りさせないよう、`visible` はこれらの列を含めてから確定させる
    // （そうしないと最後の隠し列トリムで一緒に落ちてしまう）。
    for &(inner_e, _) in &correlated_eq {
        let p = compile(arena, &scope, params, inner_e)?;
        schema.push(Field::new(String::new(), p.result_ty, true));
        exprs.push(p);
    }
    let visible = exprs.len();

    // --- ORDER BY -----------------------------------------------------------
    // 出力に無い式で並べ替えるときは、隠し列として射影に足してから落とす。
    let mut keys = Vec::new();
    for o in &sel.order_by {
        let col = match order_output_column(arena, sel, o, &schema)? {
            Some(c) => c,
            None => {
                let p = compile_with_subs(arena, &item_scope, params, &subs, o.expr)?;
                schema.push(Field::new(String::new(), p.result_ty, true));
                exprs.push(p);
                exprs.len() - 1
            }
        };
        keys.push((col, o.desc, o.nulls_first));
    }

    // --- DISTINCT ON ---------------------------------------------------------
    // ON 式も ORDER BY と同じ「まず出力列と構造的に一致するか見て、無ければ
    // 隠し列として足す」規則で解決する。実際の重複除去は Sort の後、
    // 「入力の並びでキーごとに最初の行だけ通す」ストリーミングフィルタで行う
    // （DuckDB で確認済み: ORDER BY が無ければ到着順が「最初の行」になる）。
    let mut distinct_on_cols: Vec<usize> = Vec::with_capacity(sel.distinct_on.len());
    for &on_expr in &sel.distinct_on {
        let col = match distinct_on_output_column(arena, sel, on_expr, &schema) {
            Some(c) => c,
            None => {
                let p = compile_with_subs(arena, &item_scope, params, &subs, on_expr)?;
                schema.push(Field::new(String::new(), p.result_ty, true));
                exprs.push(p);
                exprs.len() - 1
            }
        };
        distinct_on_cols.push(col);
    }

    let project_schema: Vec<Field> = schema.clone();
    node = Node::Project { input: Box::new(node), exprs, schema: project_schema.clone() };
    let project_scope = Scope::from_fields(project_schema);

    // --- DISTINCT -----------------------------------------------------------
    if sel.distinct {
        // 出力列すべてをグループキーにした集約に落とす。隠しソート列もキーに
        // 含める（含めないと、可視部分が同じ行が複数残ることがある）。
        let mut groups = Vec::new();
        let mut out_fields = Vec::new();
        for i in 0..project_scope.len() {
            groups.push(column_program(&project_scope, i)?);
            out_fields.push(project_scope.fields()[i].clone());
        }
        node = Node::Aggregate {
            input: Box::new(node),
            groups,
            aggs: Vec::new(),
            schema: out_fields,
            having: None,
        };
    }

    // --- ソート -------------------------------------------------------------
    if !keys.is_empty() {
        let mut sort_keys = Vec::with_capacity(keys.len());
        for (col, desc, nulls_first) in keys {
            sort_keys.push(SortKey {
                expr: column_program(&project_scope, col)?,
                desc,
                nulls_first,
            });
        }
        // `ORDER BY ... LIMIT n OFFSET k` は上位 n+k 件だけ持てば足りる。
        // Top-N に落とすことで全件バッファせずに済む。ただし DISTINCT ON が
        // 有る場合は「並べ替えた後の重複除去」でどの行が勝つかが決まるので、
        // 先に Top-N で切り捨てると別グループの正しい代表行を取りこぼす。
        let topn = if sel.distinct_on.is_empty() {
            sel.limit
                .map(|l| l.saturating_add(sel.offset.unwrap_or(0)).min(usize::MAX as u64) as usize)
        } else {
            None
        };
        node = Node::Sort { input: Box::new(node), keys: sort_keys, limit: topn };
    }

    // --- DISTINCT ON（実体） ---------------------------------------------------
    if !distinct_on_cols.is_empty() {
        let s = Scope::from_fields(node.schema().to_vec());
        let mut on_progs = Vec::with_capacity(distinct_on_cols.len());
        for c in distinct_on_cols {
            on_progs.push(column_program(&s, c)?);
        }
        node = Node::DistinctOn { input: Box::new(node), keys: on_progs };
    }

    // --- LIMIT / OFFSET -----------------------------------------------------
    if sel.limit.is_some() || sel.offset.unwrap_or(0) > 0 {
        node = Node::Limit {
            input: Box::new(node),
            limit: sel.limit,
            offset: sel.offset.unwrap_or(0),
        };
    }

    // 隠しソート列を落とす。
    if visible < node.schema().len() {
        let s = Scope::from_fields(node.schema().to_vec());
        let mut trim = Vec::with_capacity(visible);
        let mut fields = Vec::with_capacity(visible);
        for i in 0..visible {
            trim.push(column_program(&s, i)?);
            fields.push(s.fields()[i].clone());
        }
        node = Node::Project { input: Box::new(node), exprs: trim, schema: fields };
    }

    Ok(Plan { root: node, correlated: correlated_eq.iter().map(|&(_, o)| o).collect() })
}

// --- 木の構築 ----------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn build_tree(
    arena: &ExprArena,
    scope: &Scope,
    ranges: &[(usize, usize)],
    rels: &mut [Rel],
    tree: &FromTree,
    params: &[Value],
    per_rel: &[Vec<ExprId>],
    depth: u32,
) -> Result<Node> {
    ensure!(depth < MAX_FROM_DEPTH, ExpressionTooDeep);
    match tree {
        FromTree::Rel(i) => {
            // FROM 句の `UNNEST` は独立した関係を持たない（`Rel::unnest` の
            // ドキュメント参照）。ここに来るのは、`FromTree::Join` 側の特別
            // 分岐（下記）を経ずに `UNNEST` が単独 (`FROM UNNEST(...)`) や
            // JOIN の左側に置かれた場合 ―― 対応範囲外として明確に拒否する。
            ensure!(rels[*i].unnest.is_none(), UnsupportedFeature);
            let (start, end) = ranges[*i];
            // この関係だけを見るスコープ。押し下げた述語はここで評価する。
            let rel_scope = sub_scope(scope, start, end);

            let mut node = match rels[*i].subplan.take() {
                // 派生表はそのままプランを差し込む。列は既に確定している。
                Some(p) => p,
                None => {
                    let table = match rels[*i].table {
                        Some(t) => t,
                        None => err!(Internal),
                    };
                    let mut pruners = Vec::new();
                    for &c in &per_rel[*i] {
                        extract_pruners(arena, c, &rel_scope, &mut pruners);
                    }
                    Node::Scan(Box::new(ScanSpec {
                        table,
                        columns: rels[*i].needed.clone(),
                        schema: rel_scope.fields().to_vec(),
                        pruners,
                    }))
                }
            };
            if !per_rel[*i].is_empty() {
                let pred = and_all(arena, &rel_scope, params, &[], &per_rel[*i])?;
                node = Node::Filter { input: Box::new(node), pred };
            }
            Ok(node)
        }
        FromTree::Join { left, right, kind, on } => {
            // FROM 句の `UNNEST` は独立した関係ではなく、左の兄弟が生む行
            // ごとに配列を展開する LATERAL 相当の演算。対称な `Node::Join`
            // には落とせないので、右が `UNNEST` 関係なら特別に組み立てる
            // （`Rel::unnest`/`FromItem::Unnest` のドキュメント参照）。
            if let FromTree::Rel(ri) = right.as_ref() {
                if let Some(arg) = rels[*ri].unnest {
                    ensure!(*kind == JoinKind::Cross && on.is_none(), UnsupportedFeature);
                    let ln =
                        build_tree(arena, scope, ranges, rels, left, params, per_rel, depth + 1)?;
                    let lspan = leaf_span(ranges, left)?;
                    let lscope = sub_scope(scope, lspan.0, lspan.1);
                    let prog = compile(arena, &lscope, params, arg)?;
                    ensure!(prog.result_ty == Ty::Json, TypeMismatch);
                    let elem_field = rels[*ri].all[0].clone();
                    let elem_ty = elem_field.ty;
                    let mut schema = lscope.fields().to_vec();
                    schema.push(elem_field);
                    let mut node = Node::Unnest {
                        input: Box::new(ln),
                        expr: prog,
                        elem_ty,
                        schema: schema.clone(),
                    };
                    if !per_rel[*ri].is_empty() {
                        let joined = Scope::from_fields(schema);
                        let pred = and_all(arena, &joined, params, &[], &per_rel[*ri])?;
                        node = Node::Filter { input: Box::new(node), pred };
                    }
                    return Ok(node);
                }
            }
            let ln = build_tree(arena, scope, ranges, rels, left, params, per_rel, depth + 1)?;
            let rn = build_tree(arena, scope, ranges, rels, right, params, per_rel, depth + 1)?;
            let lw = ln.schema().len();
            let rw = rn.schema().len();

            // 結合の入出力スコープ。左のスキーマ ++ 右のスキーマ。
            let lspan = leaf_span(ranges, left)?;
            let rspan = leaf_span(ranges, right)?;
            let lscope = sub_scope(scope, lspan.0, lspan.1);
            let rscope = sub_scope(scope, rspan.0, rspan.1);
            let mut joined = lscope.clone();
            joined.extend(&rscope);
            ensure!(joined.len() == lw + rw, Internal);

            let mut left_keys = Vec::new();
            let mut right_keys = Vec::new();
            let mut residual_parts = Vec::new();

            if let Some(on) = on {
                let mut parts = Vec::new();
                split_conjuncts(arena, *on, &mut parts, 0)?;
                for c in parts {
                    match equi_key(arena, &joined, lw, c)? {
                        Some((l, r)) => {
                            let lp = compile(arena, &lscope, params, l)?;
                            let rp = compile(arena, &rscope, params, r)?;
                            let (lp, rp) = unify_key_types(lp, rp)?;
                            left_keys.push(lp);
                            right_keys.push(rp);
                        }
                        None => residual_parts.push(c),
                    }
                }
            }
            let residual = if residual_parts.is_empty() {
                None
            } else {
                Some(and_all(arena, &joined, params, &[], &residual_parts)?)
            };

            let mut schema = lscope.fields().to_vec();
            schema.extend_from_slice(rscope.fields());
            Ok(Node::Join {
                left: Box::new(ln),
                right: Box::new(rn),
                kind: *kind,
                left_keys,
                right_keys,
                residual,
                schema,
            })
        }
    }
}

/// 部分木が占めるスコープ上の範囲。葉は連続して並んでいる前提。
fn leaf_span(ranges: &[(usize, usize)], t: &FromTree) -> Result<(usize, usize)> {
    match t {
        FromTree::Rel(i) => match ranges.get(*i) {
            Some(r) => Ok(*r),
            None => err!(Internal),
        },
        FromTree::Join { left, right, .. } => {
            let l = leaf_span(ranges, left)?;
            let r = leaf_span(ranges, right)?;
            Ok((l.0, r.1))
        }
    }
}

fn sub_scope(scope: &Scope, start: usize, end: usize) -> Scope {
    let mut s = Scope::new();
    for i in start..end {
        s.push(scope.qualifier(i).map(String::from), scope.fields()[i].clone());
    }
    s
}

/// `EXISTS` / `IN (SELECT ...)` を半結合・反結合に書き換える。
#[allow(clippy::too_many_arguments)]
fn build_semijoin(
    catalog: &Catalog,
    arena: &ExprArena,
    params: &[Value],
    ctes: &mut CteScope,
    left: Node,
    scope: &Scope,
    subs: &[Substitution],
    pred: ExprId,
) -> Result<Node> {
    let schema = scope.fields().to_vec();
    match arena.get(pred) {
        Expr::Exists { query, negated } => {
            // `scope` はこの `EXISTS`/`IN` を含むクエリ自身のスコープなので、
            // そのまま「1 段外側」として渡す（相関は 1 階層のみ対応。
            // これより深い相関は内側の `scope` に存在しない列参照になり、
            // 自然に `ColumnNotFound` で失敗する）。
            let plan = bind_query_in(catalog, arena, query, params, ctes, Some(scope))?;
            // 相関が無ければ「右が 1 行でもあるか」だけが問題でキーは要らない。
            // 相関があれば、相関キーが一致する行が 1 つでもあるかを見る。
            // `EXISTS` は集合ではなく行の存在を問うだけなので、`IN`/`NOT IN`
            // のような NULL の三値論理の落とし穴が無く、通常の
            // Semi/Anti（NULL キーは互いに一致しない）でそのまま正しい
            // （DuckDB で確認済み）。
            let kind = if *negated { JoinKind::Anti } else { JoinKind::Semi };
            let (left_keys, right_keys) = correlation_keys(arena, scope, params, &plan)?;
            Ok(Node::Join {
                left: Box::new(left),
                right: Box::new(plan.root),
                kind,
                left_keys,
                right_keys,
                residual: None,
                schema,
            })
        }
        Expr::InSubquery { arg, query, negated } => {
            let plan = bind_query_in(catalog, arena, query, params, ctes, Some(scope))?;
            let k = plan.correlated.len();
            ensure!(plan.root.schema().len() - k == 1, TypeMismatch);
            let rf = plan.root.schema()[0].clone();

            let lp = compile_with_subs(arena, scope, params, subs, *arg)?;
            let rscope = Scope::from_fields(plan.root.schema().to_vec());
            let rp = column_program(&rscope, 0)?;
            // キーはバイト列に符号化して突き合わせるので、物理型を揃える。
            let want = Ty::unify_or_mismatch(lp.result_ty, rf.ty)?;
            let mut left_keys = vec![cast_program(lp, want)?];
            let mut right_keys = vec![cast_program(rp, want)?];
            let (corr_left, corr_right) = correlation_keys(arena, scope, params, &plan)?;
            left_keys.extend(corr_left);
            right_keys.extend(corr_right);

            // `NOT IN` は右に NULL が 1 つでもあると三値論理で結果が空になる。
            // 通常の反結合ではこの意味を再現できないので、NULL 対応版
            // （右に NULL キーが 1 つでもあれば無条件に空を返す）を使う。
            // `AntiNullAware` は「右全体に NULL があるか」を見る実装
            // （相関キーごとにスコープされていない）なので、相関がある場合に
            // そのまま使うと、ある外側行の相関先に NULL があるだけで無関係な
            // 別の外側行の結果まで空になりかねない。相関キーごとに絞った
            // null 対応判定を行う実行系のプリミティブが無い以上、相関
            // `NOT IN` は安全に評価できない（対象列の NULL 可能性を束縛時に
            // 正確に判定する手段も無い: SELECT リストの出力列は常に
            // `nullable = true` として扱われるため）。誤った結果を返すより
            // 明確に拒否する。
            ensure!(!(*negated && k > 0), UnsupportedFeature);
            let kind = match (*negated, rf.nullable) {
                (false, _) => JoinKind::Semi,
                (true, false) => JoinKind::Anti,
                (true, true) => JoinKind::AntiNullAware,
            };
            Ok(Node::Join {
                left: Box::new(left),
                right: Box::new(plan.root),
                kind,
                left_keys,
                right_keys,
                residual: None,
                schema,
            })
        }
        _ => err!(Internal),
    }
}

/// `plan.correlated` の各項目に対応する結合キー `(left, right)` の組を作る。
/// `plan.root` の末尾 `plan.correlated.len()` 列が、`bind_select_in` が
/// 付加した相関キー列（その手前の列数はサブクエリの種類によって違うが、
/// 常に末尾にまとまっている）。相関が無ければ両方とも空を返す。
fn correlation_keys(
    arena: &ExprArena,
    outer_scope: &Scope,
    params: &[Value],
    plan: &Plan,
) -> Result<(Vec<crate::expr::Program>, Vec<crate::expr::Program>)> {
    let k = plan.correlated.len();
    if k == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let rscope = Scope::from_fields(plan.root.schema().to_vec());
    ensure!(rscope.len() >= k, Internal);
    let base = rscope.len() - k;
    let mut left_keys = Vec::with_capacity(k);
    let mut right_keys = Vec::with_capacity(k);
    for (i, &outer_e) in plan.correlated.iter().enumerate() {
        let lp = compile(arena, outer_scope, params, outer_e)?;
        let rp = column_program(&rscope, base + i)?;
        let want = Ty::unify_or_mismatch(lp.result_ty, rp.result_ty)?;
        left_keys.push(cast_program(lp, want)?);
        right_keys.push(cast_program(rp, want)?);
    }
    Ok((left_keys, right_keys))
}

// --- 述語の分解 --------------------------------------------------------------

/// AND で連結された述語を要素に分ける。
fn split_conjuncts(arena: &ExprArena, id: ExprId, out: &mut Vec<ExprId>, depth: u32) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    match arena.get(id) {
        Expr::Binary { op: BinaryOp::And, lhs, rhs } => {
            split_conjuncts(arena, *lhs, out, depth + 1)?;
            split_conjuncts(arena, *rhs, out, depth + 1)?;
        }
        _ => out.push(id),
    }
    Ok(())
}

fn and_all(
    arena: &ExprArena,
    scope: &Scope,
    params: &[Value],
    subs: &[Substitution],
    parts: &[ExprId],
) -> Result<crate::expr::Program> {
    ensure!(!parts.is_empty(), Internal);
    let mut prog = compile_predicate_with_subs(arena, scope, params, subs, parts[0])?;
    for &p in &parts[1..] {
        let rhs = compile_predicate_with_subs(arena, scope, params, subs, p)?;
        prog = and_programs(prog, rhs)?;
    }
    Ok(prog)
}

/// 式がサブクエリを含むか。押し下げの可否判定に使う。
fn contains_subquery(arena: &ExprArena, id: ExprId, depth: u32) -> bool {
    if depth >= MAX_EXPR_DEPTH {
        return true;
    }
    if matches!(
        arena.get(id),
        Expr::ScalarSubquery(_) | Expr::Exists { .. } | Expr::InSubquery { .. }
    ) {
        return true;
    }
    let mut found = false;
    let _ = each_child(arena, id, &mut |c| {
        if contains_subquery(arena, c, depth + 1) {
            found = true;
        }
        Ok(())
    });
    found
}

/// 述語そのものが `EXISTS` / `IN (SELECT)` か。半結合に書き換えられる形。
fn is_semijoin_predicate(arena: &ExprArena, id: ExprId) -> bool {
    matches!(arena.get(id), Expr::Exists { .. } | Expr::InSubquery { .. })
}

/// 式の中のスカラサブクエリを集める。
fn collect_scalar_subqueries(
    arena: &ExprArena,
    id: ExprId,
    out: &mut Vec<ExprId>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    if matches!(arena.get(id), Expr::ScalarSubquery(_)) {
        if !out.contains(&id) {
            out.push(id);
        }
        return Ok(());
    }
    let d = depth + 1;
    each_child(arena, id, &mut |c| collect_scalar_subqueries(arena, c, out, d))
}

/// 修飾子の無い列参照ノードを (ノードの ExprId, 名前) の組で集める。
/// QUALIFY の出力別名解決に使う（`each_child` で式木全体を辿るので、
/// どんな深さに書かれていても見つかる）。
fn collect_unqualified_colrefs(
    arena: &ExprArena,
    id: ExprId,
    out: &mut Vec<(ExprId, String)>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    if let Expr::ColumnRef { qualifier: None, name } = arena.get(id) {
        out.push((id, name.clone()));
        return Ok(());
    }
    let d = depth + 1;
    each_child(arena, id, &mut |c| collect_unqualified_colrefs(arena, c, out, d))
}

/// この述語が参照する関係が 1 つだけならその添字を返す。
fn single_rel_of(
    arena: &ExprArena,
    scope: &Scope,
    ranges: &[(usize, usize)],
    id: ExprId,
) -> Result<Option<usize>> {
    let mut cols = Vec::new();
    // 解決できない参照を含む述語は押し下げない。
    if collect_refs(arena, scope, id, &mut cols).is_err() {
        return Ok(None);
    }
    if cols.is_empty() {
        return Ok(None);
    }
    let mut owner = None;
    for c in cols {
        let r = ranges.iter().position(|(s, e)| c >= *s && c < *e);
        match (owner, r) {
            (_, None) => return Ok(None),
            (None, Some(r)) => owner = Some(r),
            (Some(a), Some(b)) if a == b => {}
            _ => return Ok(None),
        }
    }
    Ok(owner)
}

/// `左の式 = 右の式` の形なら等値結合キーとして取り出す。
fn equi_key(
    arena: &ExprArena,
    joined: &Scope,
    left_width: usize,
    id: ExprId,
) -> Result<Option<(ExprId, ExprId)>> {
    let (lhs, rhs) = match arena.get(id) {
        Expr::Binary { op: BinaryOp::Eq, lhs, rhs } => (*lhs, *rhs),
        _ => return Ok(None),
    };
    let mut lc = Vec::new();
    let mut rc = Vec::new();
    if collect_refs(arena, joined, lhs, &mut lc).is_err()
        || collect_refs(arena, joined, rhs, &mut rc).is_err()
    {
        return Ok(None);
    }
    // 定数だけの辺は結合キーにならない（WHERE 相当の条件）。
    if lc.is_empty() || rc.is_empty() {
        return Ok(None);
    }
    let all_left = |v: &[usize]| v.iter().all(|&c| c < left_width);
    let all_right = |v: &[usize]| v.iter().all(|&c| c >= left_width);
    if all_left(&lc) && all_right(&rc) {
        Ok(Some((lhs, rhs)))
    } else if all_right(&lc) && all_left(&rc) {
        Ok(Some((rhs, lhs)))
    } else {
        Ok(None)
    }
}

/// 結合キーの両辺を同じ物理型に揃える。
///
/// `equi_key` は左右をそれぞれ独立にコンパイルするため（`WHERE a.k = d.k`
/// の1本のプログラムとしてコンパイルされる相関等価述語や通常の比較式とは
/// 違い）、`Ty::unify` による暗黙変換が自動では効かない。揃えないと
/// `BIGINT` 列と `DOUBLE` 列のように論理的には比較可能な型同士でも物理表現
/// （`I64` の生ビット列 対 `F64` の生ビット列）が食い違い、ハッシュ結合の
/// キー比較が常に不一致になって行が消える（クラッシュもエラーも起きない
/// ぶん、誤りに気付きにくい）。
fn unify_key_types(l: Program, r: Program) -> Result<(Program, Program)> {
    if l.result_ty == r.result_ty {
        return Ok((l, r));
    }
    let t = Ty::unify_or_mismatch(l.result_ty, r.result_ty)?;
    Ok((cast_program(l, t)?, cast_program(r, t)?))
}

// --- 相関サブクエリの WHERE 分類 ---------------------------------------------

/// WHERE の最上位 conjunct（`split_conjuncts` で AND を分解した 1 片）の分類。
enum ConjClass {
    /// このクエリ自身のスコープだけで解決できる、普通の述語。
    Local,
    /// `内側の式 = 外側スコープの式` という相関等価述語。
    Correlated { inner: ExprId, outer: ExprId },
}

/// WHERE の 1 つの最上位 conjunct を分類する。
///
/// ローカルスコープだけで解決できるなら `Local`。外側スコープの列を含む
/// `内側 = 外側` の等価述語（両辺どちらも複合式でよい）なら `Correlated`。
/// それ以外の形で外側スコープの列を参照している場合（非等価比較、`OR` の
/// 中、`NOT` で包まれている、など）は結合キーとして取り出せないので
/// `UnsupportedFeature` を返す。不正確な結果を返すより明確に拒否する方針
/// （このプロジェクト全体の方針、DESIGN.md 参照）。
fn classify_conjunct(
    arena: &ExprArena,
    local: &Scope,
    outer: &Scope,
    id: ExprId,
) -> Result<ConjClass> {
    // まずローカルだけで解決できるか（＝相関を含まない普通の述語か）を見る。
    let mut tmp = Vec::new();
    if collect_refs(arena, local, id, &mut tmp).is_ok() {
        return Ok(ConjClass::Local);
    }
    if let Expr::Binary { op: BinaryOp::Eq, lhs, rhs } = arena.get(id) {
        let (l, r) = (*lhs, *rhs);
        if is_pure_scope(arena, local, l) && is_pure_outer_only(arena, local, outer, r) {
            return Ok(ConjClass::Correlated { inner: l, outer: r });
        }
        if is_pure_scope(arena, local, r) && is_pure_outer_only(arena, local, outer, l) {
            return Ok(ConjClass::Correlated { inner: r, outer: l });
        }
    }
    // 等価の相関キーとしては取り出せなかった。外側スコープへの参照が
    // どこかに残っているなら、サポート外の相関（非等価・OR の中など）。
    if references_outer(arena, local, outer, id, 0)? {
        err!(UnsupportedFeature);
    }
    // 外側スコープに触れていないなら、相関とは無関係の（別の理由で
    // ローカル解決に失敗した）述語。通常どおりの経路に委ねる。
    Ok(ConjClass::Local)
}

/// 式が丸ごと `scope` だけで解決できるか。
fn is_pure_scope(arena: &ExprArena, scope: &Scope, id: ExprId) -> bool {
    let mut tmp = Vec::new();
    collect_refs(arena, scope, id, &mut tmp).is_ok()
}

/// 式が「ローカルでは解決できないが外側スコープでは解決できる」か
/// （＝純粋に外側だけを参照している）。
fn is_pure_outer_only(arena: &ExprArena, local: &Scope, outer: &Scope, id: ExprId) -> bool {
    if is_pure_scope(arena, local, id) {
        return false;
    }
    is_pure_scope(arena, outer, id)
}

/// 式のどこかに、外側スコープでのみ解決できる列参照があるか。
/// ローカルで解決できる同名列はシャドーイングにより除外する
/// （SQL の名前解決規則: 内側のスコープが外側より優先される）。
fn references_outer(
    arena: &ExprArena,
    local: &Scope,
    outer: &Scope,
    id: ExprId,
    depth: u32,
) -> Result<bool> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    if let Expr::ColumnRef { qualifier, name } = arena.get(id) {
        if local.resolve(qualifier.as_deref(), name).is_ok() {
            return Ok(false);
        }
        return Ok(outer.resolve(qualifier.as_deref(), name).is_ok());
    }
    let mut found = false;
    let d = depth + 1;
    each_child(arena, id, &mut |c| {
        if references_outer(arena, local, outer, c, d)? {
            found = true;
        }
        Ok(())
    })?;
    Ok(found)
}

/// `collect_refs` の相関対応版。ローカルで解決できない列参照が外側スコープで
/// 解決できるなら（相関参照とみなし）黙って無視する。ローカルにも外側にも
/// 無い列参照はこれまでどおりエラーにする。`outer_scope` が `None`（相関で
/// ない通常のクエリ）なら `collect_refs` と完全に同じ挙動。
fn collect_refs_tolerant(
    arena: &ExprArena,
    scope: &Scope,
    outer_scope: Option<&Scope>,
    id: ExprId,
    out: &mut Vec<usize>,
) -> Result<()> {
    match outer_scope {
        None => collect_refs(arena, scope, id, out),
        Some(outer) => collect_refs_tolerant_at(arena, scope, outer, id, out, 0),
    }
}

fn collect_refs_tolerant_at(
    arena: &ExprArena,
    scope: &Scope,
    outer_scope: &Scope,
    id: ExprId,
    out: &mut Vec<usize>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    if let Expr::ColumnRef { qualifier, name } = arena.get(id) {
        match scope.resolve(qualifier.as_deref(), name) {
            Ok(i) => out.push(i),
            Err(e) => {
                if outer_scope.resolve(qualifier.as_deref(), name).is_err() {
                    return Err(e);
                }
            }
        }
        return Ok(());
    }
    let d = depth + 1;
    each_child(arena, id, &mut |c| collect_refs_tolerant_at(arena, scope, outer_scope, c, out, d))
}

/// 直近の結合で相関キーとして使った末尾 `k` 列を落とす。
fn drop_trailing_columns(node: Node, k: usize) -> Result<Node> {
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

/// 2 つの `Program` のレジスタ・定数・キャスト表を 1 本に併合し、
/// `Coalesce(a.result, b.result)` を末尾に足す。`and_programs`（`compile.rs`）
/// と同じ併合規則（`compile::merge_program_bodies` を共有）だが、末尾の
/// 合成演算子だけが違う。
fn coalesce_programs(mut a: Program, b: Program) -> Program {
    let ty = a.result_ty;
    let (ra, rb) = crate::plan::compile::merge_program_bodies(&mut a, b);
    let dst = a.alloc_reg();
    a.push(Instr::new(OpCode::Coalesce, ty.phys(), dst, ra, rb));
    a.result = dst;
    a
}

/// `scope` の `i` 列目が NULL なら 0 に補正する `Program` を作る。
/// 相関 GROUP BY 分の疑似グループが存在しない（＝一致する内側行が無い）
/// 場合に `count(*)`/`count(x)` を「0 行を集約 → 0」に補正するために使う
/// （DuckDB で確認済み: 通常の非相関の `count` と同じ結果になるようにする）。
fn coalesce_zero(scope: &Scope, i: usize) -> Result<Program> {
    let ty = scope.fields()[i].ty;
    let col = column_program(scope, i)?;
    let zero = const_program(ty, Value::I64(0));
    Ok(coalesce_programs(col, zero))
}

/// 集約結果ノードの `target` 列（COUNT 系の集約結果）だけを `coalesce_zero`
/// で補正し、他の列（相関 GROUP BY のキー列）はそのまま通す `Project` を被せる。
fn coalesce_count_column(node: Node, target: usize) -> Result<Node> {
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

// --- 集約の抽出 --------------------------------------------------------------

/// 式の中の集約呼び出しを集める。既出のものは構造の一致で重複を除く。
fn collect_aggregates(
    arena: &ExprArena,
    id: ExprId,
    out: &mut Vec<ExprId>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    let d = depth + 1;
    if let Expr::Function { name, args, .. } = arena.get(id) {
        if AggKind::from_name(name).is_some() {
            // 集約の入れ子は SQL として無効。
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

/// `GROUPING(col, ...)` / `GROUPING_ID(col, ...)`。
///
/// 集約関数とは違って引数を評価しない（グルーピングセットごとに定まる
/// ビットマスク定数に置き換わるだけ）ので `collect_aggregates` とは
/// 別に集める。
fn is_grouping_fn(name: &str) -> bool {
    eq_ascii_ci(name.as_bytes(), b"grouping") || eq_ascii_ci(name.as_bytes(), b"grouping_id")
}

/// 式の中の `GROUPING`/`GROUPING_ID` 呼び出しを集める。入れ子は不正
/// （集約と同じ理由: 引数の中で自分自身が現れることは意味を持たない）。
fn collect_grouping_calls(
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

/// 式の中のウィンドウ関数呼び出しを集める。入れ子は不正。
fn collect_windows(arena: &ExprArena, id: ExprId, out: &mut Vec<ExprId>, depth: u32) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    let d = depth + 1;
    if let Expr::Window { args, partition_by, order_by, .. } = arena.get(id) {
        // ウィンドウ関数の入れ子は SQL として無効。
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

/// 式の中の `Expr::Unnest` 呼び出しを集める。集約・ウィンドウと違って
/// 入れ子（`UNNEST(...)` の引数の中にさらに `UNNEST`）を特別扱いする必要は
/// 無い ―― `each_child` が引数の中まで辿るので自然に見つかり、複数見つかれば
/// 呼び出し側が `UnsupportedFeature` で拒否する（`FILTER`/`QUALIFY`
/// 実装時と同じ「集約でも通常の式でもない特殊な式」としての拾い方）。
fn collect_unnests(arena: &ExprArena, id: ExprId, out: &mut Vec<ExprId>, depth: u32) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    let d = depth + 1;
    if matches!(arena.get(id), Expr::Unnest(_)) {
        out.push(id);
        return Ok(());
    }
    each_child(arena, id, &mut |c| collect_unnests(arena, c, out, d))
}

/// UNNEST の要素型を、可能なら `Ty::Json` から実際のスカラ型へ復元する。
///
/// このエンジンでは配列・オブジェクトはすべて `Ty::Json`（JSON テキスト）
/// に統一されており、要素の型は実データを見ないと分からない
/// （`vector::types::Ty::Json` のドキュメント参照）。一方でクエリの出力列の
/// 型はバインド時（実行前）に確定していなければならないので、実データを
/// 読まずに「安全に」ネイティブ型だと判定できる場合だけ絞り込む。
///
/// 唯一データを見ずに判定できるのは、`UNNEST` の対象がその場で
/// `json_array(...)`/`list_value(...)`（DuckDB のリストリテラル相当。この
/// エンジンには `[1,2,3]` のような配列リテラル構文自体が無いので、
/// `duckdb -c "SELECT UNNEST([1,2,3])"` に対応する書き方はこちらになる）を
/// 直接呼んでいるケース: 各引数の**コンパイル時の型**が分かっているので、
/// 全引数が入れ子を持たない同じ非 JSON スカラ型に揃うかどうかを、行データを
/// 一切読まずに判定できる。
///
/// それ以外（テーブルの JSON 列そのものを UNNEST する、通常のケース）は
/// 実データを読まないと判定できないので `Ty::Json` のまま返す
/// （タスクの要求どおり、この場合は絞り込まなくてよい）。
fn narrow_unnest_elem_ty(arena: &ExprArena, scope: &Scope, params: &[Value], arg: ExprId) -> Ty {
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
        // 引数自体が JSON（配列・オブジェクトを含みうる）なら、要素が
        // 「入れ子を持たないスカラー」という前提が崩れるので諦める。
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
        // DECIMAL/DATE/TIME/TIMESTAMP/NULL 等は JSON テキスト経由での復元を
        // 実装していない（範囲外。精度・書式を保って往復させるにはこの
        // エンジンの JSON シリアライズ規約を専用に決める必要があり、
        // スコープを絞った）。
        _ => Ty::Json,
    }
}

fn build_window(
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
    // `OVER w`（名前付き参照）は `WINDOW` 句で定義された実体をここで引く。
    // `WINDOW` 句は構文上 SELECT リストより後に現れるため、パース時点では
    // まだ名前を解決できず、束縛時にこの一箇所でだけ解決する。
    let (partition_by, order_by, frame): (&[ExprId], &[OrderByItem], WindowFrame) = match window_ref
    {
        Some(wname) => {
            let def = windows.iter().find(|(n, _)| eq_ascii_ci(n.as_bytes(), wname.as_bytes()));
            match def {
                Some((_, d)) => (&d.partition_by, &d.order_by, d.frame),
                // 未定義のウィンドウ名を参照した。`duckdb` はパース時に
                // `window "w" does not exist` として拒否するが、このエンジンは
                // `WINDOW` 句を先読みしないため束縛時に検出する。
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
    // `count(*) OVER ()` は引数を取らない集約として扱う。
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
        // 順位付けは 1 始まりの通し番号。
        WindowKind::RowNumber | WindowKind::Rank | WindowKind::DenseRank => Ty::BigInt,
        // 値を運ぶだけの関数は入力型をそのまま返す。
        WindowKind::Lag | WindowKind::Lead | WindowKind::FirstValue | WindowKind::LastValue => {
            arg_ty
        }
        // 集約は通常の集約と同じ規則。両方でずれないよう同じ関数を通す。
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

/// このクエリが「裸の集約呼び出し 1 つだけを SELECT する」単純な形なら、
/// その集約の種類を返す。相関スカラサブクエリが（内側の相関 GROUP BY
/// 決コリレーションを経由する）集約サブクエリだったかどうかを、束縛結果の
/// `Plan` を介さずに呼び出し側（`bind_select_in` のスカラサブクエリ処理）
/// から再判定するために使う。相関スカラサブクエリの束縛が成功している
/// 時点で、集約を伴うなら必ずこの形（`bind_select_in` の早期リターン経路の
/// `ensure!` 参照）であることが保証されているので、ここでの判定と実際の
/// 束縛結果は必ず一致する。
fn as_bare_aggregate(arena: &ExprArena, q: &QueryStmt) -> Option<AggKind> {
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

fn build_agg(arena: &ExprArena, scope: &Scope, params: &[Value], id: ExprId) -> Result<Agg> {
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
    // FILTER の条件は集約前の入力スコープ（args と同じ）で評価する。
    // 集約自身は書けない（`collect_aggregates` が別扱いにしないので、
    // FILTER の中身に集約を書くと通常の「集約の入れ子」検出に掛かって弾かれる）。
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

    // `string_agg(x, sep)` だけが 2 引数を許す。sep は定数リテラルのみ。
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

/// GROUP BY に無い裸の列参照を検出する。
fn check_grouped(
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
            // 入力に存在する列がここまで来た = GROUP BY にも集約にも入っていない。
            if scope.resolve(qualifier.as_deref(), name).is_ok() {
                err!(NotGrouped);
            }
        }
        // スカラサブクエリは集約前の列として横に付けてあるので、集約の上で
        // 裸で参照されると列番号がずれる。列参照と同じ扱いで弾く。
        Expr::ScalarSubquery(_) => err!(NotGrouped),
        _ => {}
    }
    let d = depth + 1;
    each_child(arena, id, &mut |c| check_grouped(arena, scope, c, groups, aggs, d))
}

// --- ORDER BY / GROUP BY の参照解決 -----------------------------------------

/// `GROUP BY 1` / `GROUP BY alias` を対応する SELECT 式に読み替える。
fn resolve_select_ref(arena: &ExprArena, sel: &SelectStmt, id: ExprId) -> Result<ExprId> {
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
fn order_output_column(
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
fn distinct_on_output_column(
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
fn ordinal_of(arena: &ExprArena, id: ExprId) -> Option<u32> {
    match arena.get(id) {
        Expr::Literal(Value::I32(v)) if *v > 0 => Some(*v as u32),
        Expr::Literal(Value::I64(v)) if *v > 0 && *v <= u32::MAX as i64 => Some(*v as u32),
        _ => None,
    }
}

// --- 走査ヘルパ --------------------------------------------------------------

/// 式の直接の子をすべて訪問する。
fn each_child(
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
fn collect_refs(arena: &ExprArena, scope: &Scope, id: ExprId, out: &mut Vec<usize>) -> Result<()> {
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

fn collect_join_refs(
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

fn group_name(arena: &ExprArena, id: ExprId, i: usize) -> String {
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
fn const_program(ty: Ty, v: Value) -> Program {
    let mut p = Program::new();
    let k = p.add_const(ty, v);
    let dst = p.alloc_reg();
    p.push(Instr::with_aux(OpCode::LoadConst, ty.phys(), dst, 0, 0, k));
    p.result = dst;
    p.result_ty = ty;
    p
}

/// 別名が無い出力列の名前。列参照はその名前、それ以外は連番。
fn default_name(arena: &ExprArena, id: ExprId) -> String {
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

fn push_u32(s: &mut String, mut v: u32) {
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

// --- 枝刈り述語の抽出 --------------------------------------------------------

/// `列 <op> 定数` の形を取り出す。
///
/// AND で連結された枝だけを辿る。OR の下では「片方が真なら通る」ため、
/// 個々の条件で分割を落とすと結果が変わってしまう。
fn extract_pruners(arena: &ExprArena, id: ExprId, scope: &Scope, out: &mut Vec<Pruner>) {
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

/// FROM 句からテーブル添字を得る（`DESCRIBE` 用）。
pub fn resolve_from(catalog: &Catalog, from: &FromItem) -> Result<usize> {
    match from {
        FromItem::Table { name, .. } => match catalog.index_of(name) {
            Some(i) => Ok(i),
            None => err!(TableNotFound),
        },
        FromItem::Parquet { path, .. } => match catalog.index_of(path) {
            Some(i) => Ok(i),
            None => err!(TableNotFound),
        },
        // `GenerateSeries` はカタログに実体を持たない計算だけのソースなので、
        // `DESCRIBE` が期待する「テーブル添字」を返しようがない。
        FromItem::Join { .. }
        | FromItem::Subquery { .. }
        | FromItem::Unnest { .. }
        | FromItem::GenerateSeries { .. } => {
            err!(UnsupportedFeature)
        }
    }
}

/// クエリ木の中で参照されるテーブルを集める。
pub fn referenced_in_query(
    catalog: &Catalog,
    q: &QueryStmt,
    out: &mut Vec<usize>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_FROM_DEPTH, ExpressionTooDeep);
    for c in &q.ctes {
        referenced_in_query(catalog, &c.query, out, depth + 1)?;
    }
    referenced_in_set_expr(catalog, &q.body, out, depth + 1)
}

fn referenced_in_set_expr(
    catalog: &Catalog,
    e: &SetExpr,
    out: &mut Vec<usize>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_FROM_DEPTH, ExpressionTooDeep);
    match e {
        SetExpr::Select(s) => match &s.from {
            // CTE 名はカタログに無いので、見つからなくても後段のバインドに任せる。
            Some(f) => {
                let _ = referenced_tables_at(catalog, f, out, depth + 1);
                Ok(())
            }
            None => Ok(()),
        },
        SetExpr::SetOp { left, right, .. } => {
            referenced_in_set_expr(catalog, left, out, depth + 1)?;
            referenced_in_set_expr(catalog, right, out, depth + 1)
        }
    }
}

/// SQL が参照するテーブルをすべて集める。スキーマ解決の対象を知るために使う。
pub fn referenced_tables(catalog: &Catalog, from: &FromItem, out: &mut Vec<usize>) -> Result<()> {
    referenced_tables_at(catalog, from, out, 0)
}

fn referenced_tables_at(
    catalog: &Catalog,
    from: &FromItem,
    out: &mut Vec<usize>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_FROM_DEPTH, ExpressionTooDeep);
    let mut push = |i: usize| {
        if !out.contains(&i) {
            out.push(i);
        }
    };
    match from {
        FromItem::Table { name, .. } => match catalog.index_of(name) {
            Some(i) => {
                push(i);
                Ok(())
            }
            None => err!(TableNotFound),
        },
        FromItem::Parquet { path, .. } => match catalog.index_of(path) {
            Some(i) => {
                push(i);
                Ok(())
            }
            None => err!(TableNotFound),
        },
        FromItem::Join { left, right, .. } => {
            referenced_tables_at(catalog, left, out, depth + 1)?;
            referenced_tables_at(catalog, right, out, depth + 1)
        }
        FromItem::Subquery { query, .. } => referenced_in_query(catalog, query, out, depth + 1),
        // `expr` は列参照のみを含みうるスカラ式で、そこが指すテーブルは
        // 必ず左の兄弟が別の `FromItem` として持つ（暗黙 LATERAL の制約、
        // `FromItem::Unnest` のドキュメント参照）ので、ここで新たに解決が
        // 要るテーブルは無い。
        FromItem::Unnest { .. } => Ok(()),
        // カタログを経由しない計算だけのソースなので、解決が要るテーブルは無い。
        FromItem::GenerateSeries { .. } => Ok(()),
    }
}

// --- PIVOT/UNPIVOT -----------------------------------------------------------
//
// どちらも「計画レベルの構文糖衣展開」として実装する（GROUPING SETS が複数の
// `Node::Aggregate` を `Node::SetOp` で束ねた方針と同じ発想）。ただし
// GROUPING SETS と違い、展開後の形が既存の「`GROUP BY` + 集約 + `FILTER`」
// （PIVOT）や「射影 + `UNION ALL`」（UNPIVOT）とちょうど同じ AST 形に落ちる
// ので、束縛（`bind_select_in`）そのものには一切手を入れず、AST レベルの
// 書き換えだけで完結させる。呼び出し元は `Session::prepare`
// （PIVOT/UNPIVOT の `Stmt` を検出した直後、対象表のスキーマを解決してから
// 呼ぶ）。展開結果は通常の `Stmt::Select` として、そのまま既存の
// `prepare_query` に渡る。
//
// アリーナは束縛時点では不変参照（`&ExprArena`）なので、新しい式ノード
// （`on = 値` の等号や、`FILTER` 付き集約呼び出し）を束縛の最中に作ることは
// できない。そのため展開はパース直後・束縛前の段階で行い、
// `substitute_now`（`sql::now`）と同じように `&mut ExprArena` へ新規ノードを
// 積む。

use crate::sql::ast::{PivotStmt, SelectItem, UnpivotStmt};

/// `IN (...)` で明示された 1 つの PIVOT 値がとりうるリテラル型の上限。
/// 文字列・整数・真偽値のみ対応（列名の文字列化に `core::fmt` を使えない
/// ため、浮動小数点数は非対応）。
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

/// `core::fmt` を使わない符号付き 10 進数文字列化。`push_u32` の `i128` 版。
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

/// 式 `id` が（再帰的に）参照する裸の列名をすべて集める。GROUP BY 省略時の
/// デフォルト列（`on`/`using` が参照する列以外の全列）を決めるのに使う。
/// サブクエリ・ウィンドウの中は `each_child` がそもそも辿らない
/// （モジュール doc 参照）ので、それらの中の列参照は対象外になる —
/// PIVOT の `ON`/`USING` にサブクエリやウィンドウ関数が来ることは想定して
/// いないので実害はない。
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

/// `PIVOT` を通常の `SELECT ... GROUP BY ...` へ展開する。
///
/// `from_schema` は対象表（`stmt.from`）の列一覧。`stmt.group_by` が空の
/// ときだけ使う（`on`/`using` が参照する列を除いた残り全部が既定の
/// `GROUP BY` 対象になる — DuckDB と同じ規則）。実データは一切読まない。
pub fn desugar_pivot(
    arena: &mut ExprArena,
    stmt: PivotStmt,
    from_schema: &[Field],
) -> Result<QueryStmt> {
    let PivotStmt { from, on, in_list, using, group_by, order_by, limit, offset } = stmt;

    // 値の自動検出（`IN` 省略）には対象列の実データの DISTINCT が要る。
    // 束縛時点ではスキーマしか読めていない（`Session::prepare` がスキーマ
    // 解決の直後にこの関数を呼ぶ）ので非対応（モジュール doc / `PivotStmt`
    // doc 参照）。
    let in_list = match in_list {
        Some(v) => v,
        None => err!(UnsupportedFeature),
    };
    ensure!(!in_list.is_empty(), SyntaxError);
    ensure!(in_list.len() <= MAX_PIVOT_VALUES, ExpressionTooDeep);

    // `USING` 省略時は DuckDB と同じく既定で `count(*)`。
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
    // 複数集約関数（`USING sum(a), avg(b)`）は列名（`a_sum(a)` 方式）の決定に
    // 式の文字列化が要り、`core::fmt` 禁止の下でのコストに見合わないため
    // 対応しない（`PivotStmt::using` doc 参照）。
    ensure!(using.len() == 1, UnsupportedFeature);
    let (agg_name, agg_args, agg_distinct, agg_star) = match arena.get(using[0].expr) {
        Expr::Function { name, args, distinct, star, filter } => {
            ensure!(filter.is_none(), UnsupportedFeature);
            (name.clone(), args.clone(), *distinct, *star)
        }
        _ => err!(SyntaxError),
    };

    // GROUP BY 対象列。明示が無ければ「on/using が参照する列以外の全列」。
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

    // 値ごとに `agg(...) FILTER (WHERE on = 値)` を 1 列作る。既存の
    // `FILTER (WHERE cond)` 付き集約の仕組み（`Expr::Function.filter`、
    // `exec::agg::Agg.filter`）にそのまま乗るので、集約束縛ロジックには
    // 手を入れていない。
    //
    // `IN (...)` に同じ値が 2 回以上現れると（別名の有無に関わらず）
    // `duckdb` は "The value ... was specified multiple times in the IN
    // clause" として拒否する（`duckdb -c "PIVOT ... ON category IN
    // ('a','a') ..."` で確認済み）。ここでチェックしないと、同じ
    // `FILTER` 条件を持つ列が重複した名前で 2 本できてしまう
    // （既知の不具合。`star_exclude_replace.rs` の EXCLUDE/REPLACE 重複
    // チェックや `WINDOW` 句の重複名チェックと同じ判断で、値ベースの
    // 重複だけを拒む — 別名が衝突しても `duckdb` は `_1` サフィックスで
    // 自動的にリネームするだけでエラーにしないので、そちらは追わない）。
    let mut seen_values: Vec<Value> = Vec::with_capacity(in_list.len());
    for (val_expr, alias) in &in_list {
        let lit = match arena.get(*val_expr) {
            Expr::Literal(v) => v.clone(),
            // `TypedLiteral`/式一般は非対応。列名も定数畳み込みも要るため。
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
        // `bind_query_in` の「外側の ORDER BY/LIMIT」経路（列名・序数参照のみ
        // 対応）にそのまま乗せる。展開後の本体は素の単一 SELECT だが、こちら
        // に置いても意味は変わらない（`UNPIVOT` 側は `SetOp` になるので
        // 同じ置き場所で揃えてある）。
        order_by,
        limit,
        offset,
    })
}

/// `PIVOT` の `IN (...)` に書ける値の個数の上限。`CUBE`/`ROLLUP` の
/// `MAX_CUBE_COLS`（`sql::parser`）と同じ「作りすぎを防ぐ安全弁」の役割。
const MAX_PIVOT_VALUES: usize = 128;

/// `UNPIVOT` で同時に畳み込める対象列数の上限。
const MAX_UNPIVOT_COLUMNS: usize = 128;

/// `FromItem` を複製する。PIVOT/UNPIVOT の `from` は `Table`/`Parquet` に
/// しか対応しない（`UNPIVOT` は対象列数ぶん同じ `from` を複製して
/// `UNION ALL` の各枝に配る必要があり、`Subquery`/`Join` はプラン
/// （`Node`）を複製できないため単純にコピーできない。`plan::bind::resolve_from`
/// が `DESCRIBE` 向けに課している制約と同じ）。
fn clone_from_item(f: &FromItem) -> Result<FromItem> {
    match f {
        FromItem::Table { name, alias } => {
            Ok(FromItem::Table { name: name.clone(), alias: alias.clone() })
        }
        FromItem::Parquet { path, alias } => {
            Ok(FromItem::Parquet { path: path.clone(), alias: alias.clone() })
        }
        // データを持たない計算だけのソースなので、他の派生表と違って複製に
        // 実コストが無い。
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

/// `UNPIVOT` を `UNION ALL` へ展開する。
///
/// 対象列 1 本につき「対象列以外をそのまま通し、対象列名を文字列リテラル
/// として `name_col` に、対象列の値を `value_col` に出す」`SELECT` を 1 本
/// 作り、全部を `UNION ALL` で束ねる。`from_schema` は対象表の列一覧
/// （素通しする「対象列以外」を決めるのに使う。実データは読まない）。
pub fn desugar_unpivot(
    arena: &mut ExprArena,
    stmt: UnpivotStmt,
    from_schema: &[Field],
) -> Result<QueryStmt> {
    let UnpivotStmt { from, columns, name_col, value_col, order_by, limit, offset } = stmt;
    ensure!(!columns.is_empty(), SyntaxError);
    ensure!(columns.len() <= MAX_UNPIVOT_COLUMNS, ExpressionTooDeep);

    // 対象は修飾子なしの裸の列参照のみ（`UnpivotStmt` doc 参照）。
    let mut target_names: Vec<String> = Vec::with_capacity(columns.len());
    for &c in &columns {
        match arena.get(c) {
            Expr::ColumnRef { qualifier: None, name } => target_names.push(name.clone()),
            _ => err!(UnsupportedFeature),
        }
    }

    // 対象列以外はそのまま素通しする（DuckDB の既定と同じ）。
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

    // 左結合の `UNION ALL` 連鎖に束ねる。GROUPING SETS が `Node::SetOp` で
    // 複数の `Node::Aggregate` を束ねたのと同じ発想（モジュール doc 参照）。
    let mut iter = branches.into_iter();
    let mut body = match iter.next() {
        Some(b) => b,
        None => err!(Internal), // columns が空でない限り起きない
    };
    for b in iter {
        body = SetExpr::SetOp {
            op: SetOp::Union,
            all: true,
            left: Box::new(body),
            right: Box::new(b),
        };
    }

    Ok(QueryStmt { ctes: Vec::new(), body, order_by, limit, offset })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::code_of;
    use crate::vector::Ty;

    fn scope() -> Scope {
        Scope::from_fields(vec![
            Field::new("id", Ty::Int, false),
            Field::new("score", Ty::Double, true),
            Field::new("name", Ty::Varchar, true),
        ])
    }

    fn col(a: &mut ExprArena, n: &str) -> ExprId {
        a.push(Expr::ColumnRef { qualifier: None, name: n.into() })
    }

    #[test]
    fn pruner_extracted_from_simple_comparison() {
        let mut a = ExprArena::new();
        let l = col(&mut a, "id");
        let r = a.push(Expr::Literal(Value::I32(10)));
        let e = a.push(Expr::Binary { op: BinaryOp::Gt, lhs: l, rhs: r });
        let mut out = Vec::new();
        extract_pruners(&a, e, &scope(), &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].column, 0);
        assert_eq!(out[0].op, PruneOp::Gt);
    }

    #[test]
    fn pruner_handles_reversed_operands() {
        let mut a = ExprArena::new();
        let l = a.push(Expr::Literal(Value::I32(10)));
        let r = col(&mut a, "id");
        let e = a.push(Expr::Binary { op: BinaryOp::Lt, lhs: l, rhs: r });
        let mut out = Vec::new();
        extract_pruners(&a, e, &scope(), &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].op, PruneOp::Gt);
    }

    #[test]
    fn or_branches_are_not_pruned() {
        let mut a = ExprArena::new();
        let l1 = col(&mut a, "id");
        let r1 = a.push(Expr::Literal(Value::I32(10)));
        let c1 = a.push(Expr::Binary { op: BinaryOp::Gt, lhs: l1, rhs: r1 });
        let l2 = col(&mut a, "id");
        let r2 = a.push(Expr::Literal(Value::I32(20)));
        let c2 = a.push(Expr::Binary { op: BinaryOp::Lt, lhs: l2, rhs: r2 });
        let e = a.push(Expr::Binary { op: BinaryOp::Or, lhs: c1, rhs: c2 });
        let mut out = Vec::new();
        extract_pruners(&a, e, &scope(), &mut out);
        assert!(out.is_empty(), "OR の下の条件で枝刈りしてはいけない");
    }

    #[test]
    fn string_columns_are_not_pruned() {
        let mut a = ExprArena::new();
        let l = col(&mut a, "name");
        let r = a.push(Expr::Literal(Value::Bytes(b"x".to_vec())));
        let e = a.push(Expr::Binary { op: BinaryOp::Gt, lhs: l, rhs: r });
        let mut out = Vec::new();
        extract_pruners(&a, e, &scope(), &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn between_is_decomposed_into_ge_and_le_pruners() {
        let mut a = ExprArena::new();
        let arg = col(&mut a, "id");
        let low = a.push(Expr::Literal(Value::I32(10)));
        let high = a.push(Expr::Literal(Value::I32(20)));
        let e = a.push(Expr::Between { arg, low, high, negated: false });
        let mut out = Vec::new();
        extract_pruners(&a, e, &scope(), &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].op, PruneOp::Ge);
        assert_eq!(out[1].op, PruneOp::Le);
    }

    #[test]
    fn negated_between_is_not_pruned() {
        let mut a = ExprArena::new();
        let arg = col(&mut a, "id");
        let low = a.push(Expr::Literal(Value::I32(10)));
        let high = a.push(Expr::Literal(Value::I32(20)));
        let e = a.push(Expr::Between { arg, low, high, negated: true });
        let mut out = Vec::new();
        extract_pruners(&a, e, &scope(), &mut out);
        assert!(out.is_empty(), "NOT BETWEEN は範囲の外側なので統計だけでは絞れない");
    }

    #[test]
    fn in_list_of_literals_becomes_a_single_in_pruner() {
        let mut a = ExprArena::new();
        let arg = col(&mut a, "id");
        let v1 = a.push(Expr::Literal(Value::I32(1)));
        let v2 = a.push(Expr::Literal(Value::I32(2)));
        let v3 = a.push(Expr::Literal(Value::I32(3)));
        let e = a.push(Expr::InList { arg, list: vec![v1, v2, v3], negated: false });
        let mut out = Vec::new();
        extract_pruners(&a, e, &scope(), &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].op, PruneOp::In);
        assert_eq!(out[0].column, 0);
        // 先頭が `value`、残りが `in_values` に入る。
        assert_eq!(out[0].in_values.len(), 2);
    }

    #[test]
    fn in_list_with_null_literal_drops_the_null_but_keeps_the_rest() {
        let mut a = ExprArena::new();
        let arg = col(&mut a, "id");
        let v1 = a.push(Expr::Literal(Value::I32(1)));
        let vn = a.push(Expr::Literal(Value::Null));
        let e = a.push(Expr::InList { arg, list: vec![v1, vn], negated: false });
        let mut out = Vec::new();
        extract_pruners(&a, e, &scope(), &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].op, PruneOp::In);
        assert!(out[0].in_values.is_empty());
    }

    #[test]
    fn in_list_with_non_literal_element_is_not_pruned() {
        let mut a = ExprArena::new();
        let arg = col(&mut a, "id");
        let v1 = a.push(Expr::Literal(Value::I32(1)));
        let other_col = col(&mut a, "score");
        let e = a.push(Expr::InList { arg, list: vec![v1, other_col], negated: false });
        let mut out = Vec::new();
        extract_pruners(&a, e, &scope(), &mut out);
        assert!(out.is_empty(), "候補に非リテラルがあれば集合が確定しないので枝刈りしない");
    }

    #[test]
    fn negated_in_list_is_not_pruned() {
        let mut a = ExprArena::new();
        let arg = col(&mut a, "id");
        let v1 = a.push(Expr::Literal(Value::I32(1)));
        let e = a.push(Expr::InList { arg, list: vec![v1], negated: true });
        let mut out = Vec::new();
        extract_pruners(&a, e, &scope(), &mut out);
        assert!(out.is_empty(), "NOT IN は候補以外の全てにマッチしうるので統計だけでは絞れない");
    }

    #[test]
    fn in_list_on_string_column_is_not_pruned() {
        let mut a = ExprArena::new();
        let arg = col(&mut a, "name");
        let v1 = a.push(Expr::Literal(Value::Bytes(b"x".to_vec())));
        let e = a.push(Expr::InList { arg, list: vec![v1], negated: false });
        let mut out = Vec::new();
        extract_pruners(&a, e, &scope(), &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn conjuncts_are_split_on_and_only() {
        let mut a = ExprArena::new();
        let x = col(&mut a, "id");
        let y = col(&mut a, "score");
        let z = col(&mut a, "name");
        let and1 = a.push(Expr::Binary { op: BinaryOp::And, lhs: x, rhs: y });
        let e = a.push(Expr::Binary { op: BinaryOp::And, lhs: and1, rhs: z });
        let mut out = Vec::new();
        split_conjuncts(&a, e, &mut out, 0).unwrap();
        assert_eq!(out, vec![x, y, z]);

        let or = a.push(Expr::Binary { op: BinaryOp::Or, lhs: x, rhs: y });
        let mut out2 = Vec::new();
        split_conjuncts(&a, or, &mut out2, 0).unwrap();
        assert_eq!(out2, vec![or], "OR は分解しない");
    }

    fn join_scope() -> Scope {
        let mut s = Scope::new();
        s.push(Some("l".into()), Field::new("k", Ty::Int, false));
        s.push(Some("l".into()), Field::new("v", Ty::Int, false));
        s.push(Some("r".into()), Field::new("k", Ty::Int, false));
        s.push(Some("r".into()), Field::new("w", Ty::Int, false));
        s
    }

    #[test]
    fn equi_key_detects_both_orientations() {
        let s = join_scope();
        let mut a = ExprArena::new();
        let lk = a.push(Expr::ColumnRef { qualifier: Some("l".into()), name: "k".into() });
        let rk = a.push(Expr::ColumnRef { qualifier: Some("r".into()), name: "k".into() });
        let e = a.push(Expr::Binary { op: BinaryOp::Eq, lhs: lk, rhs: rk });
        assert_eq!(equi_key(&a, &s, 2, e).unwrap(), Some((lk, rk)));

        // 逆向きに書かれていても左右を入れ替えて拾う。
        let e2 = a.push(Expr::Binary { op: BinaryOp::Eq, lhs: rk, rhs: lk });
        assert_eq!(equi_key(&a, &s, 2, e2).unwrap(), Some((lk, rk)));
    }

    #[test]
    fn same_side_equality_is_not_a_join_key() {
        let s = join_scope();
        let mut a = ExprArena::new();
        let lk = a.push(Expr::ColumnRef { qualifier: Some("l".into()), name: "k".into() });
        let lv = a.push(Expr::ColumnRef { qualifier: Some("l".into()), name: "v".into() });
        let e = a.push(Expr::Binary { op: BinaryOp::Eq, lhs: lk, rhs: lv });
        assert_eq!(equi_key(&a, &s, 2, e).unwrap(), None);

        // 定数との比較も結合キーではない。
        let lit = a.push(Expr::Literal(Value::I32(1)));
        let e2 = a.push(Expr::Binary { op: BinaryOp::Eq, lhs: lk, rhs: lit });
        assert_eq!(equi_key(&a, &s, 2, e2).unwrap(), None);

        // 不等号も等値結合にはならない（残余述語に回る）。
        let rk = a.push(Expr::ColumnRef { qualifier: Some("r".into()), name: "k".into() });
        let e3 = a.push(Expr::Binary { op: BinaryOp::Lt, lhs: lk, rhs: rk });
        assert_eq!(equi_key(&a, &s, 2, e3).unwrap(), None);
    }

    #[test]
    fn single_rel_ownership_for_pushdown() {
        let s = join_scope();
        let ranges = [(0usize, 2usize), (2, 4)];
        let mut a = ExprArena::new();
        let lk = a.push(Expr::ColumnRef { qualifier: Some("l".into()), name: "k".into() });
        let lit = a.push(Expr::Literal(Value::I32(1)));
        let only_left = a.push(Expr::Binary { op: BinaryOp::Gt, lhs: lk, rhs: lit });
        assert_eq!(single_rel_of(&a, &s, &ranges, only_left).unwrap(), Some(0));

        let rk = a.push(Expr::ColumnRef { qualifier: Some("r".into()), name: "k".into() });
        let both = a.push(Expr::Binary { op: BinaryOp::Eq, lhs: lk, rhs: rk });
        assert_eq!(single_rel_of(&a, &s, &ranges, both).unwrap(), None);

        // 定数だけの条件はどの関係にも属さない。
        let c = a.push(Expr::Binary { op: BinaryOp::Eq, lhs: lit, rhs: lit });
        assert_eq!(single_rel_of(&a, &s, &ranges, c).unwrap(), None);
    }

    #[test]
    fn aggregates_are_collected_and_deduplicated() {
        let mut a = ExprArena::new();
        let x = col(&mut a, "id");
        let c1 = a.push(Expr::Function {
            name: "sum".into(),
            args: vec![x],
            distinct: false,
            star: false,
            filter: None,
        });
        let x2 = col(&mut a, "id");
        let c2 = a.push(Expr::Function {
            name: "SUM".into(),
            args: vec![x2],
            distinct: false,
            star: false,
            filter: None,
        });
        let e = a.push(Expr::Binary { op: BinaryOp::Add, lhs: c1, rhs: c2 });
        let mut out = Vec::new();
        collect_aggregates(&a, e, &mut out, 0).unwrap();
        // 構造が同じ集約は 1 つにまとめる（2 度計算しない）。
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn nested_aggregates_are_rejected() {
        let mut a = ExprArena::new();
        let x = col(&mut a, "id");
        let inner = a.push(Expr::Function {
            name: "sum".into(),
            args: vec![x],
            distinct: false,
            star: false,
            filter: None,
        });
        let outer = a.push(Expr::Function {
            name: "max".into(),
            args: vec![inner],
            distinct: false,
            star: false,
            filter: None,
        });
        let mut out = Vec::new();
        assert_eq!(code_of(collect_aggregates(&a, outer, &mut out, 0)), Some(Code::NotAggregate));
    }

    #[test]
    fn ungrouped_column_is_rejected() {
        let mut a = ExprArena::new();
        let g = col(&mut a, "id");
        let other = col(&mut a, "score");
        // GROUP BY id なのに score を裸で参照している。
        assert_eq!(
            code_of(check_grouped(&a, &scope(), other, &[g], &[], 0)),
            Some(Code::NotGrouped)
        );
        // GROUP BY した列そのものは通る。
        assert!(check_grouped(&a, &scope(), g, &[g], &[], 0).is_ok());
    }

    #[test]
    fn grouped_expression_matches_structurally() {
        // GROUP BY id + 1 に対して SELECT id + 1 は許される。
        let mut a = ExprArena::new();
        let c1 = col(&mut a, "id");
        let l1 = a.push(Expr::Literal(Value::I32(1)));
        let g = a.push(Expr::Binary { op: BinaryOp::Add, lhs: c1, rhs: l1 });
        let c2 = col(&mut a, "ID");
        let l2 = a.push(Expr::Literal(Value::I32(1)));
        let e = a.push(Expr::Binary { op: BinaryOp::Add, lhs: c2, rhs: l2 });
        assert!(check_grouped(&a, &scope(), e, &[g], &[], 0).is_ok());
    }

    #[test]
    fn ordinal_recognition() {
        let mut a = ExprArena::new();
        let one = a.push(Expr::Literal(Value::I32(1)));
        let zero = a.push(Expr::Literal(Value::I32(0)));
        let neg = a.push(Expr::Literal(Value::I32(-1)));
        let s = a.push(Expr::Literal(Value::Bytes(b"1".to_vec())));
        assert_eq!(ordinal_of(&a, one), Some(1));
        assert_eq!(ordinal_of(&a, zero), None);
        assert_eq!(ordinal_of(&a, neg), None);
        assert_eq!(ordinal_of(&a, s), None);
    }

    #[test]
    fn refs_are_collected_and_unknown_columns_rejected() {
        let mut a = ExprArena::new();
        let l = col(&mut a, "score");
        let r = col(&mut a, "id");
        let e = a.push(Expr::Binary { op: BinaryOp::Add, lhs: l, rhs: r });
        let mut out = Vec::new();
        collect_refs(&a, &scope(), e, &mut out).unwrap();
        out.sort_unstable();
        assert_eq!(out, vec![0, 1]);

        let bad = col(&mut a, "nope");
        let mut out2 = Vec::new();
        assert_eq!(code_of(collect_refs(&a, &scope(), bad, &mut out2)), Some(Code::ColumnNotFound));
    }
}
