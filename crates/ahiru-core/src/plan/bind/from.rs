//! FROM-clause flattening: turns the `FromItem` AST into a flat list of
//! `Rel`s plus a `FromTree` describing how joins combine them, builds the
//! physical scan/join tree (`build_tree`), and resolves table references for
//! `DESCRIBE`/DDL callers (`resolve_from`, `referenced_in_query`,
//! `referenced_tables`).

use super::agg::narrow_unnest_elem_ty;
#[cfg(feature = "ddl")]
use super::cte::MAX_VIEW_DEPTH;
use super::cte::{CteScope, ResolvedCte};
use super::pruning::extract_pruners;
use super::subquery::{and_all, equi_key, split_conjuncts, unify_key_types};
use super::*;

// --- FROM 句の平坦化 ---------------------------------------------------------

/// FROM に現れる 1 つの関係。
///
/// `pub(super)`: `select::bind_select_in` builds/reads `Rel`s directly
/// (projection pushdown, `UNNEST` column collection), so the type and every
/// field it touches must be visible to that sibling submodule.
pub(super) struct Rel {
    /// カタログ上のテーブル添字。サブクエリなら `None`。
    pub(super) table: Option<usize>,
    /// 修飾子（別名、無ければテーブル名）。
    pub(super) alias: String,
    /// 関係が持つ全列。
    pub(super) all: Vec<Field>,
    /// 参照される列（`all` 上の添字、昇順・重複なし）。
    pub(super) needed: Vec<usize>,
    /// サブクエリの場合のプラン。
    pub(super) subplan: Option<Node>,
    /// FROM 句の `UNNEST(expr) AS alias(col)` の `expr`（未束縛の AST 式）。
    /// `Some` のときは他のフィールドは「1 列だけのダミーの関係」の形だけ
    /// 整えてあり（`table`/`subplan` は使わない）、実プランは `build_tree` が
    /// 左の兄弟ノードをラップして特別に組み立てる（LATERAL 相当。
    /// `flatten_from`/`build_tree` の該当コメント参照）。
    pub(super) unnest: Option<ExprId>,
}

/// FROM の木構造。葉は `rels` の添字。
///
/// `pub(super)`: matched directly by `refs::collect_join_refs`.
pub(super) enum FromTree {
    Rel(usize),
    Join { left: Box<FromTree>, right: Box<FromTree>, kind: JoinKind, on: Option<ExprId> },
}

impl FromTree {
    /// 木に外部結合が 1 つでも含まれるか。述語の押し下げ可否の判定に使う。
    pub(super) fn has_outer_join(&self) -> bool {
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
pub(super) fn flatten_from(
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
        // `parquet(...)`/`read_parquet(...)`/`read_csv[_auto](...)`/
        // `read_json[_auto](...)`/bare `'path'` (see `FromItem::File` doc).
        // All resolve the same way: `path` must already be registered as a
        // table name (the host registers the URL/path verbatim, so SQL can
        // write e.g. `FROM parquet('https://…')` or `FROM read_csv('a.csv')`
        // once the host has registered a table under that exact string).
        // `format` is intentionally not consulted here — it only records
        // which surface syntax was used; this engine cannot re-dispatch
        // parsing at bind time (see `FromItem::File` doc for why).
        FromItem::File { path, alias, .. } => {
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
pub(super) fn full_scope(rels: &[Rel]) -> Scope {
    let mut s = Scope::new();
    for r in rels {
        for f in &r.all {
            s.push(qual_of(r), f.clone());
        }
    }
    s
}

/// `needed` を反映した絞り込み後のスコープ。
pub(super) fn narrow_scope(rels: &[Rel]) -> Scope {
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
pub(super) fn rel_ranges(rels: &[Rel]) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(rels.len());
    let mut off = 0;
    for r in rels {
        out.push((off, off + r.needed.len()));
        off += r.needed.len();
    }
    out
}

// --- 木の構築 ----------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(super) fn build_tree(
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

/// FROM 句からテーブル添字を得る（`DESCRIBE` 用）。
pub fn resolve_from(catalog: &Catalog, from: &FromItem) -> Result<usize> {
    match from {
        FromItem::Table { name, .. } => match catalog.index_of(name) {
            Some(i) => Ok(i),
            None => err!(TableNotFound),
        },
        FromItem::File { path, .. } => match catalog.index_of(path) {
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
        FromItem::File { path, .. } => match catalog.index_of(path) {
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
