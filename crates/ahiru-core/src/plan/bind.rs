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
use crate::format::{PruneOp, Pruner};
use crate::plan::compile::{
    and_programs, column_program, compile, compile_predicate, compile_predicate_with_subs,
    compile_with_subs, expr_eq, Substitution,
};
use crate::plan::{Agg, AggKind, Node, Plan, ScanSpec, Scope, SortKey};
use crate::prelude::*;
use crate::rt::hash::eq_ascii_ci;
use crate::sql::ast::{
    BinaryOp, Expr, ExprArena, ExprId, FromItem, JoinKind, OrderByItem, Parsed, SelectStmt, Stmt,
};
use crate::vector::{Field, Value};

/// FROM 句のネスト上限。パーサ側でも制限しているが二重に守る。
const MAX_FROM_DEPTH: u32 = 64;
/// 式のネスト上限。
const MAX_EXPR_DEPTH: u32 = 64;

pub fn bind(catalog: &Catalog, parsed: &Parsed, params: &[Value]) -> Result<Plan> {
    match &parsed.stmt {
        Stmt::Select(s) | Stmt::Explain(s) => bind_select(catalog, &parsed.arena, s, params),
        // DESCRIBE / SHOW はセッション層が処理する。
        _ => err!(UnsupportedFeature),
    }
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

fn flatten_from(
    catalog: &Catalog,
    arena: &ExprArena,
    params: &[Value],
    from: &FromItem,
    rels: &mut Vec<Rel>,
    depth: u32,
) -> Result<FromTree> {
    ensure!(depth < MAX_FROM_DEPTH, ExpressionTooDeep);
    match from {
        FromItem::Table { name, alias } => {
            let i = match catalog.index_of(name) {
                Some(i) => i,
                None => err!(TableNotFound),
            };
            push_table_rel(catalog, i, alias.clone().unwrap_or_else(|| name.clone()), rels)
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
            let plan = bind_select(catalog, arena, query, params)?;
            let all = plan.root.schema().to_vec();
            // 別名の無い派生表は修飾して参照できないが、無修飾の参照は通す。
            let alias = alias.clone().unwrap_or_default();
            rels.push(Rel {
                table: None,
                alias,
                needed: (0..all.len()).collect(),
                all,
                subplan: Some(plan.root),
            });
            Ok(FromTree::Rel(rels.len() - 1))
        }
        FromItem::Join { left, right, kind, on } => {
            let l = flatten_from(catalog, arena, params, left, rels, depth + 1)?;
            let r = flatten_from(catalog, arena, params, right, rels, depth + 1)?;
            Ok(FromTree::Join { left: Box::new(l), right: Box::new(r), kind: *kind, on: *on })
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
    ensure!(t.format.is_resolved(), Internal);
    rels.push(Rel {
        table: Some(i),
        alias,
        all: t.schema().to_vec(),
        needed: Vec::new(),
        subplan: None,
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

// --- 本体 --------------------------------------------------------------------

pub fn bind_select(
    catalog: &Catalog,
    arena: &ExprArena,
    sel: &SelectStmt,
    params: &[Value],
) -> Result<Plan> {
    let from = match &sel.from {
        Some(f) => f,
        // `SELECT 1` のような FROM 無しは v1 では扱わない。
        None => err!(UnsupportedFeature),
    };

    let mut rels: Vec<Rel> = Vec::new();
    let tree = flatten_from(catalog, arena, params, from, &mut rels, 0)?;
    let scope_all = full_scope(&rels);

    // --- 参照される列を集める（射影プッシュダウン） -------------------------
    let mut refs: Vec<usize> = Vec::new();
    let mut star_all = false;
    let mut star_quals: Vec<String> = Vec::new();
    for item in &sel.items {
        match arena.get(item.expr) {
            Expr::Star { qualifier: None } => star_all = true,
            Expr::Star { qualifier: Some(q) } => {
                ensure!(scope_all.has_qualifier(q), TableNotFound);
                star_quals.push(q.clone());
            }
            _ => collect_refs(arena, &scope_all, item.expr, &mut refs)?,
        }
    }
    for e in [sel.filter, sel.having].into_iter().flatten() {
        collect_refs(arena, &scope_all, e, &mut refs)?;
    }
    for e in &sel.group_by {
        // 序数指定（`GROUP BY 1`）はここでは列参照を持たない。
        if ordinal_of(arena, *e).is_none() {
            collect_refs(arena, &scope_all, *e, &mut refs)?;
        }
    }
    for o in &sel.order_by {
        // ORDER BY は出力別名も指しうるので、解決できなくてもここでは許す。
        let _ = collect_refs(arena, &scope_all, o.expr, &mut refs);
    }
    collect_join_refs(arena, &scope_all, &tree, &mut refs)?;

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
    for c in conjuncts {
        let owner = if pushdown_ok { single_rel_of(arena, &scope, &ranges, c)? } else { None };
        match owner {
            Some(i) => per_rel[i].push(c),
            None => leftover.push(c),
        }
    }

    // --- スキャンと結合を組み立てる -----------------------------------------
    let mut node = build_tree(arena, &scope, &ranges, &mut rels, &tree, params, &per_rel, 0)?;

    if !leftover.is_empty() {
        let pred = and_all(arena, &scope, params, &leftover)?;
        node = Node::Filter { input: Box::new(node), pred };
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

    let aggregating = !agg_calls.is_empty() || !sel.group_by.is_empty();
    let mut subs: Vec<Substitution> = Vec::new();
    let mut item_scope = scope.clone();

    if aggregating {
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
    } else {
        ensure!(sel.having.is_none(), NotAggregate);
    }

    // --- 射影 ---------------------------------------------------------------
    let mut exprs = Vec::new();
    let mut schema = Vec::new();
    for item in &sel.items {
        match arena.get(item.expr) {
            Expr::Star { qualifier } => {
                // 集約後に `*` は展開できない（元の行が残っていない）。
                ensure!(!aggregating, NotGrouped);
                let idx: Vec<usize> = match qualifier {
                    Some(q) => scope.indices_for_qualifier(q),
                    None => (0..scope.len()).collect(),
                };
                for i in idx {
                    exprs.push(column_program(&scope, i)?);
                    schema.push(scope.fields()[i].clone());
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
        // Top-N に落とすことで全件バッファせずに済む。
        let topn = sel
            .limit
            .map(|l| l.saturating_add(sel.offset.unwrap_or(0)).min(usize::MAX as u64) as usize);
        node = Node::Sort { input: Box::new(node), keys: sort_keys, limit: topn };
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

    Ok(Plan { root: node })
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
                let pred = and_all(arena, &rel_scope, params, &per_rel[*i])?;
                node = Node::Filter { input: Box::new(node), pred };
            }
            Ok(node)
        }
        FromTree::Join { left, right, kind, on } => {
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
                            left_keys.push(compile(arena, &lscope, params, l)?);
                            right_keys.push(compile(arena, &rscope, params, r)?);
                        }
                        None => residual_parts.push(c),
                    }
                }
            }
            let residual = if residual_parts.is_empty() {
                None
            } else {
                Some(and_all(arena, &joined, params, &residual_parts)?)
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
    parts: &[ExprId],
) -> Result<crate::expr::Program> {
    ensure!(!parts.is_empty(), Internal);
    let mut prog = compile_predicate(arena, scope, params, parts[0])?;
    for &p in &parts[1..] {
        let rhs = compile_predicate(arena, scope, params, p)?;
        prog = and_programs(prog, rhs)?;
    }
    Ok(prog)
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

fn build_agg(arena: &ExprArena, scope: &Scope, params: &[Value], id: ExprId) -> Result<Agg> {
    let (name, args, distinct, star) = match arena.get(id) {
        Expr::Function { name, args, distinct, star } => (name, args, *distinct, *star),
        _ => err!(Internal),
    };
    let kind = match AggKind::from_name(name) {
        Some(k) => k,
        None => err!(FunctionNotFound),
    };
    if star {
        ensure!(kind == AggKind::Count, WrongArgCount);
        ensure!(!distinct, UnsupportedFeature);
        return Ok(Agg {
            kind: AggKind::CountStar,
            arg: None,
            distinct: false,
            name: String::from("count_star()"),
        });
    }
    ensure!(args.len() == 1, WrongArgCount);
    let arg = compile(arena, scope, params, args[0])?;
    Ok(Agg { kind, arg: Some(arg), distinct, name: agg_name(name, arena, args[0]) })
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
    if let Expr::ColumnRef { qualifier, name } = arena.get(id) {
        // 入力に存在する列がここまで来た = GROUP BY にも集約にも入っていない。
        if scope.resolve(qualifier.as_deref(), name).is_ok() {
            err!(NotGrouped);
        }
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
        Expr::Literal(_) | Expr::Param(_) | Expr::Star { .. } | Expr::ColumnRef { .. } => {}
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
        Expr::Function { args, .. } => {
            for a in args {
                f(*a)?;
            }
        }
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

/// 別名が無い出力列の名前。列参照はその名前、それ以外は連番。
fn default_name(arena: &ExprArena, id: ExprId) -> String {
    match arena.get(id) {
        Expr::ColumnRef { name, .. } => name.clone(),
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
        _ => {}
    }
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
    Some(Pruner { column, op, value })
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
        FromItem::Join { .. } | FromItem::Subquery { .. } => err!(UnsupportedFeature),
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
        FromItem::Subquery { query, .. } => match &query.from {
            Some(f) => referenced_tables_at(catalog, f, out, depth + 1),
            None => Ok(()),
        },
    }
}

// 統計の枝刈り判定はフォーマット層にある。既存の呼び出し元のために再エクスポート。
pub use crate::format::parquet::stat_value;
pub use crate::format::range_may_match;

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
        });
        let x2 = col(&mut a, "id");
        let c2 = a.push(Expr::Function {
            name: "SUM".into(),
            args: vec![x2],
            distinct: false,
            star: false,
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
        });
        let outer = a.push(Expr::Function {
            name: "max".into(),
            args: vec![inner],
            distinct: false,
            star: false,
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
