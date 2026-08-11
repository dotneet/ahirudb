//! CTE (`WITH` / `WITH RECURSIVE`) binding: the `CteScope` registry, per-CTE
//! binding, and recursive-CTE anchor/recursive-term splitting.

use super::select::bind_select_in;
use super::*;

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
    // `pub(super)`: read/updated directly by `from::push_view_rel`, which lives
    // in a sibling submodule (view expansion is part of FROM-clause handling,
    // not CTE handling proper).
    #[cfg(feature = "ddl")]
    pub(super) view_depth: u32,
}

/// ビュー展開の最大ネスト数。`MAX_FROM_DEPTH` より小さくしてあるのは、
/// ビュー 1 段の展開が丸ごと 1 回分の `bind_query_in` 呼び出し（再帰下降の
/// フルスタック）に相当し、単純な FROM の入れ子より 1 段あたりのスタック
/// 消費が大きいため。
///
/// `pub(super)`: consulted by `from::push_view_rel` alongside `view_depth`.
#[cfg(feature = "ddl")]
pub(super) const MAX_VIEW_DEPTH: u32 = 16;

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
///
/// `pub(super)`: matched by `from::flatten_from`, a sibling submodule.
pub(super) enum ResolvedCte {
    /// 通常の CTE。1 回だけ取り出せる。
    Plan(Box<Node>),
    /// 再帰 CTE 本体からの自己参照。実データは持たず、実行時に
    /// `RecursiveCte` オペレータが差し込む。
    WorkingTable,
}

impl CteScope {
    // `pub(super)`: called from `from::flatten_from`, a sibling submodule,
    // to resolve `FromItem::Table` names against in-scope CTEs.
    pub(super) fn find(&self, name: &str) -> Option<usize> {
        self.entries.iter().position(|e| eq_ascii_ci(e.name.as_bytes(), name.as_bytes()))
    }

    /// `find` で見つけた添字を解決する。スキーマは常にクローンで返す
    /// （`ResolvedCte::WorkingTable` は何度参照されてもよいため、複製の
    /// コストを払ってでも「1 回しか取り出せない」制約を外している）。
    pub(super) fn resolve(&mut self, i: usize) -> Result<(ResolvedCte, Vec<Field>)> {
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

/// 1 個の CTE を束縛して `ctes` に積む。
///
/// `c.recursive`（`WITH RECURSIVE` の対象）が立っていても、本文が実際に
/// 自分自身を参照していなければ普通の CTE として扱う（標準 SQL 通り、
/// `WITH RECURSIVE` のリストに非再帰の CTE が混ざってよい）。
pub(super) fn bind_one_cte(
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
        FromItem::File { .. } => false,
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
