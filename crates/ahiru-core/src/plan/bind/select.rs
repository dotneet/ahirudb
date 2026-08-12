//! The core `SELECT` binder (`bind_select_in`): projection pushdown,
//! WHERE decomposition and pushdown, scalar-subquery and quantified-
//! comparison decorrelation, aggregate/window rewriting, QUALIFY, and the
//! final projection/DISTINCT/ORDER BY/LIMIT assembly.

use super::agg::{
    as_bare_aggregate, build_agg, build_window, check_grouped, coalesce_count_column,
    collect_aggregates, collect_grouping_calls, collect_unnests, collect_windows,
    drop_trailing_columns, narrow_unnest_elem_ty,
};
use super::cte::CteScope;
use super::from::{build_tree, flatten_from, full_scope, narrow_scope, rel_ranges, Rel};
use super::refs::{
    collect_join_refs, collect_refs, const_program, default_name, distinct_on_output_column,
    group_name, order_output_column, push_u32, resolve_select_ref,
};
use super::subquery::{
    and_all, build_quantified_comparison, build_semijoin, classify_conjunct,
    collect_quantified_comparisons, collect_refs_tolerant, collect_scalar_subqueries,
    collect_unqualified_colrefs, contains_subquery, is_semijoin_predicate, single_rel_of,
    split_conjuncts, ConjClass,
};
use super::*;
use crate::expr::regex;
use crate::sql::ast::ColumnsSpec;

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

// --- COLUMNS(...) ------------------------------------------------------------

/// Resolve DuckDB's `COLUMNS(...)` star expression against an input scope.
///
/// Returns `(column index, output name override)` pairs **in schema order**,
/// which is what `duckdb` does even for the explicit-list form
/// (`COLUMNS(['name','id'])` yields `id, name`, not `name, id`). The name
/// override is `Some` only when the item carried an `AS '<template>'` alias;
/// see `expr::regex::expand_name_template` for the template rules.
///
/// This runs twice per `COLUMNS(...)` item — once over the full scope while
/// collecting columns for projection pushdown (so a `COLUMNS('regex')` over a
/// wide table only ever reads the columns it actually expands to), and once
/// over the narrowed scope when building the projection. Both passes apply
/// the same predicate to the same names, so they always agree on the set;
/// only the indices differ.
///
/// Failure modes are DuckDB's, verified against `duckdb` v1.4.4:
/// a regex matching nothing is an error ("No matching columns found that
/// match regex"), and so is a listed name that no column has ("Column ...
/// was selected but was not found in the FROM clause"). That puts `COLUMNS`
/// on the `EXCLUDE` side of the asymmetry documented on `Expr::Star::rename`,
/// not the `RENAME` side.
pub(super) fn expand_columns(
    spec: &ColumnsSpec,
    scope: &Scope,
    template: Option<&str>,
) -> Result<Vec<(usize, Option<String>)>> {
    let named = |name: &str, saves: Option<&[u32]>| -> Option<String> {
        let t = template?;
        let bytes = regex::expand_name_template(name.as_bytes(), saves, t.as_bytes());
        Some(String::from_utf8(bytes).unwrap_or_else(|_| name.into()))
    };
    let mut out: Vec<(usize, Option<String>)> = Vec::new();
    match spec {
        ColumnsSpec::All => {
            for i in 0..scope.len() {
                out.push((i, named(&scope.fields()[i].name, None)));
            }
        }
        ColumnsSpec::Regex(pattern) => {
            // The match is an unanchored search over the column name, and it
            // is case-sensitive — unlike every other column-name comparison
            // in this engine, which is case-insensitive. Both verified
            // against `duckdb`: `COLUMNS('um')` matches a `num` column, while
            // `COLUMNS('N.*')` matches nothing at all.
            let prog = regex::compile(pattern.as_bytes())?;
            for i in 0..scope.len() {
                let name = &scope.fields()[i].name;
                if let Some(saves) = regex::find(&prog, name.as_bytes())? {
                    out.push((i, named(name, Some(&saves[..]))));
                }
            }
            ensure!(!out.is_empty(), ColumnNotFound);
        }
        ColumnsSpec::Names(names) => {
            for i in 0..scope.len() {
                let name = &scope.fields()[i].name;
                if names.iter().any(|n| eq_ascii_ci(n.as_bytes(), name.as_bytes())) {
                    out.push((i, named(name, None)));
                }
            }
            // A name listed twice, or one that matches columns from two
            // different relations of a join, is not an error — the loop above
            // already emits each *column* exactly once, which matches
            // `duckdb` (`COLUMNS(['id','id'])` yields one `id`).
            for n in names {
                ensure!(
                    scope.fields().iter().any(|f| eq_ascii_ci(f.name.as_bytes(), n.as_bytes())),
                    ColumnNotFound
                );
            }
        }
    }
    Ok(out)
}

// --- 本体 --------------------------------------------------------------------

/// `outer_scope` が `Some` のとき、この SELECT は相関サブクエリとして
/// バインドされる: WHERE の最上位 AND 節にある「外側スコープの式 = 内側の式」
/// という等価述語を相関キーとして検出し、通常の述語処理からは除外した上で
/// `Plan::correlated` に載せて返す（呼び出し側が結合キーとして使う）。
/// `None` のとき（トップレベル・CTE・FROM 句の派生表）は従来どおり。
pub(super) fn bind_select_in(
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
            Expr::Star { qualifier: None, columns, replace, .. } => {
                match columns {
                    // `COLUMNS('regex')` / `COLUMNS([...])` expand to a
                    // *subset* of the input, so only that subset has to be
                    // read. Resolving the spec here rather than falling back
                    // to "read everything" is what keeps the engine's
                    // never-read-a-byte-you-don't-need property (DESIGN.md
                    // §2) intact for a `COLUMNS('regex')` over a wide table.
                    Some(spec @ (ColumnsSpec::Regex(_) | ColumnsSpec::Names(_))) => {
                        for (i, _) in expand_columns(spec, &scope_all, None)? {
                            refs.push(i);
                        }
                    }
                    // `COLUMNS(*)` is exactly a plain `*`.
                    None | Some(ColumnsSpec::All) => star_all = true,
                }
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

    // `>`/`<`/`>=`/`<=` を伴う `ANY`/`ALL`（非相関のみ）。`= ANY`/`<> ALL` は
    // `is_semijoin_predicate` 経由で下の `semijoins` ループが処理するので、
    // ここには来ない（`collect_quantified_comparisons` の doc 参照）。
    // スカラサブクエリと同じ「結合で 1 列足して `Substitution` で差し替える」
    // パターンなので、位置は独立（`semijoins`/`leftover` より前ならどこでもよい）。
    let mut quantifieds: Vec<ExprId> = Vec::new();
    for item in &sel.items {
        collect_quantified_comparisons(arena, item.expr, &mut quantifieds, 0)?;
    }
    for e in [sel.filter, sel.having].into_iter().flatten() {
        collect_quantified_comparisons(arena, e, &mut quantifieds, 0)?;
    }
    for e in &sel.group_by {
        collect_quantified_comparisons(arena, *e, &mut quantifieds, 0)?;
    }
    for o in &sel.order_by {
        collect_quantified_comparisons(arena, o.expr, &mut quantifieds, 0)?;
    }
    for (n, id) in quantifieds.iter().enumerate() {
        let (op, arg, all, q) = match arena.get(*id) {
            Expr::QuantifiedComparison { op, arg, all, query } => (*op, *arg, *all, query.as_ref()),
            _ => err!(Internal),
        };
        node = build_quantified_comparison(
            catalog, arena, params, ctes, node, &scope, &subs, n, op, arg, all, q,
        )?;
        let out_field = match node.schema().last() {
            Some(f) => f.clone(),
            None => err!(Internal),
        };
        scope.push(None, out_field);
        subs.push(Substitution { expr: *id, column: scope.len() - 1, structural: false });
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
            Expr::Star { qualifier, columns, exclude, replace, rename } => {
                // 集約後に `*` は展開できない（元の行が残っていない）。
                ensure!(!aggregating, NotGrouped);
                // For a `COLUMNS(...)` item the select-item alias is a name
                // *template* applied per expanded column, not a single output
                // name — see `expr::regex::expand_name_template`.
                let expanded: Vec<(usize, Option<String>)> = match columns {
                    Some(spec) => expand_columns(spec, &scope, item.alias.as_deref())?,
                    None => {
                        let idx: Vec<usize> = match qualifier {
                            Some(q) => scope.indices_for_qualifier(q),
                            None => (0..scope.len()).collect(),
                        };
                        idx.into_iter().map(|i| (i, None)).collect()
                    }
                };
                // `EXCLUDE`/`REPLACE` に書かれた列名が実在するかを検証する
                // （`duckdb` は "Column ... not found" として束縛時に拒否する）。
                for name in exclude {
                    ensure!(
                        expanded.iter().any(|&(i, _)| eq_ascii_ci(
                            scope.fields()[i].name.as_bytes(),
                            name.as_bytes()
                        )),
                        ColumnNotFound
                    );
                }
                for (_, name) in replace {
                    ensure!(
                        expanded.iter().any(|&(i, _)| eq_ascii_ci(
                            scope.fields()[i].name.as_bytes(),
                            name.as_bytes()
                        )),
                        ColumnNotFound
                    );
                }
                for (i, templated) in expanded {
                    let fname = scope.fields()[i].name.clone();
                    if exclude.iter().any(|e| eq_ascii_ci(e.as_bytes(), fname.as_bytes())) {
                        continue;
                    }
                    let rexpr: Option<ExprId> = replace
                        .iter()
                        .find(|(_, n)| eq_ascii_ci(n.as_bytes(), fname.as_bytes()))
                        .map(|&(e, _)| e);
                    // `RENAME (old AS new, ...)` only relabels the OUTPUT
                    // column name — it is applied last (after EXCLUDE/
                    // REPLACE above), and everything else in the query still
                    // resolves the column by its original name (`fname`).
                    // Unlike EXCLUDE/REPLACE, an `old` that matches no column
                    // here is silently ignored rather than an error; this
                    // deliberately mirrors `duckdb`'s real behavior, which is
                    // asymmetric with EXCLUDE on this point.
                    //
                    // A `COLUMNS(...) AS '<template>'` name wins over RENAME
                    // and is built from the ORIGINAL column name (verified
                    // against `duckdb`: `COLUMNS(* RENAME (id AS ident)) AS
                    // 'x_\0'` yields `x_id`, not `x_ident`).
                    let out_name = templated.unwrap_or_else(|| {
                        rename
                            .iter()
                            .find(|(old, _)| eq_ascii_ci(old.as_bytes(), fname.as_bytes()))
                            .map(|(_, new)| new.clone())
                            .unwrap_or_else(|| fname.clone())
                    });
                    match rexpr {
                        Some(rexpr) => {
                            // REPLACE の式は通常の select item と同じスコープ
                            // （集約・ウィンドウ出力を含みうる `item_scope`）で
                            // コンパイルする。列名自体は元のまま変えない。
                            let p = compile_with_subs(arena, &item_scope, params, &subs, rexpr)?;
                            schema.push(Field::new(out_name, p.result_ty, true));
                            exprs.push(p);
                        }
                        None => {
                            exprs.push(column_program(&scope, i)?);
                            let mut field = scope.fields()[i].clone();
                            field.name = out_name;
                            schema.push(field);
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
