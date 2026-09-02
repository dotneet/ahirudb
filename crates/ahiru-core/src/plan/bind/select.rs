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
    collect_join_refs, collect_outer_refs, collect_refs, const_program, default_name,
    distinct_on_output_column, group_name, order_output_column, push_u32, resolve_group_ref,
};
use super::subquery::{
    and_all, build_quantified_comparison, build_semijoin, classify_conjunct, collect_colrefs,
    collect_quantified_comparisons, collect_refs_tolerant, collect_scalar_subqueries,
    contains_subquery, is_semijoin_predicate, single_rel_of, split_conjuncts, ConjClass,
};
use super::*;
use crate::expr::regex;
use crate::sql::ast::ColumnsSpec;

// --- SAMPLE ------------------------------------------------------------------

/// Resolves the AST's `sql::ast::SampleSpec` for execution.
///
/// The method syntax (`BERNOULLI`/`SYSTEM`/`RESERVOIR`) is only accepted; the implementation
/// reduces to the two cases of `is_rows` (see the `exec::sample` module docs; the
/// simplification "percentage > row count > distinguishing methods", per the task's priorities).
/// Range checks on the percentage and row count are already done by
/// `sql::parser::Parser::sample_amount`, so only the seed is resolved here.
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

// --- GROUP BY ALL ------------------------------------------------------------

/// `GROUP BY ALL` (DuckDB shorthand) -> the concrete list of grouping
/// expressions. Returns `sel.group_by` unchanged when the shorthand isn't
/// used, so every caller downstream stays on the ordinary code path.
///
/// The rule, verified against the `duckdb` CLI: group by every select-list
/// expression that does **not** contain an aggregate anywhere inside it.
///
/// ```text
/// duckdb: select g, h, sum(x) from t group by all      -- groups by g, h
/// duckdb: select g+1, sum(x)+1 from t group by all     -- groups by g+1 only
/// duckdb: select sum(x) from t group by all            -- no grouping columns
/// duckdb: select g from t group by all                 -- groups by g (a DISTINCT)
/// ```
///
/// Note the second line: `sum(x)+1` is excluded even though the item itself
/// is not an aggregate call — containing one anywhere is what counts.
///
/// `SELECT *` combined with `GROUP BY ALL` is rejected rather than expanded.
/// duckdb supports it (it groups by every column of the star expansion), but
/// this engine cannot: the binder's arena is immutable, so it has no way to
/// materialise the per-column expressions the star stands for, and `*` after
/// an aggregate is already refused by the projection below (`ensure!
/// (!aggregating, NotGrouped)`). Failing loudly here beats silently
/// grouping by nothing.
fn resolve_group_by_all(arena: &ExprArena, sel: &SelectStmt) -> Result<Vec<ExprId>> {
    if !sel.group_by_all {
        return Ok(sel.group_by.clone());
    }
    // `GROUP BY ALL` and `GROUPING SETS`/`ROLLUP`/`CUBE` are syntactically exclusive too (the
    // parser sets only one of them). If that premise breaks, fail rather than silently ignoring one.
    ensure!(sel.grouping_sets.is_none(), Internal);
    let mut out = Vec::new();
    for item in &sel.items {
        ensure!(!matches!(arena.get(item.expr), Expr::Star { .. }), UnsupportedFeature);
        let mut aggs = Vec::new();
        collect_aggregates(arena, item.expr, &mut aggs, 0)?;
        if aggs.is_empty() {
            out.push(item.expr);
        }
    }
    Ok(out)
}

/// Makes a column reference spelled differently from its grouping expression resolve to the
/// aggregate's grouping column.
///
/// `GROUP BY emp.dept` with `SELECT dept` (or the reverse) names one and the same input column,
/// but the `structural: true` substitution installed for a grouping expression matches raw
/// syntax, so the qualifier difference would leave `dept` compiled against the aggregate's
/// output scope — where the input column no longer exists. Each such reference gets its own
/// exact-node (`structural: false`) substitution instead, which cannot mis-match a same-named
/// column of another relation the way a loosened structural comparison would.
fn add_equivalent_group_subs(
    arena: &ExprArena,
    scope: &Scope,
    sel: &SelectStmt,
    group_exprs: &[ExprId],
    subs: &mut Vec<Substitution>,
) -> Result<()> {
    // (input column, grouping column) for every grouping expression that is a plain column ref.
    let mut gcols: Vec<(usize, usize)> = Vec::new();
    for (i, &g) in group_exprs.iter().enumerate() {
        if let Expr::ColumnRef { qualifier, name } = arena.get(g) {
            if let Ok(c) = scope.resolve(qualifier.as_deref(), name) {
                gcols.push((c, i));
            }
        }
    }
    if gcols.is_empty() {
        return Ok(());
    }
    let mut refs = Vec::new();
    for item in &sel.items {
        collect_colrefs(arena, item.expr, &mut refs, 0)?;
    }
    for e in [sel.having, sel.qualify].into_iter().flatten() {
        collect_colrefs(arena, e, &mut refs, 0)?;
    }
    for o in &sel.order_by {
        collect_colrefs(arena, o.expr, &mut refs, 0)?;
    }
    for (rid, qual, name) in refs {
        if group_exprs.contains(&rid) {
            continue;
        }
        let Ok(c) = scope.resolve(qual.as_deref(), &name) else { continue };
        if let Some(&(_, gi)) = gcols.iter().find(|&&(gc, _)| gc == c) {
            subs.push(Substitution { expr: rid, column: gi, structural: false });
        }
    }
    Ok(())
}

// --- Main --------------------------------------------------------------------

/// When `outer_scope` is `Some`, this SELECT is bound as a correlated subquery: an equality
/// predicate of the form "outer-scope expression = inner expression" in a top-level AND clause
/// of WHERE is detected as a correlation key, excluded from the ordinary predicate handling,
/// and returned on `Plan::correlated` (the caller uses it as a join key).
/// With `None` (top level, a CTE, a FROM-clause derived table) behavior is as before.
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
        // A missing FROM, as in `SELECT 1`, is not handled in v1.
        None => err!(UnsupportedFeature),
    };

    // `GROUP BY ALL` is resolved once here into a concrete expression list, and from then on
    // takes exactly the same path as an ordinary `GROUP BY a, b, ...`.
    let group_by = resolve_group_by_all(arena, sel)?;

    let mut rels: Vec<Rel> = Vec::new();
    let tree = flatten_from(catalog, arena, params, from, &mut rels, ctes, 0)?;
    let scope_all = full_scope(&rels);

    // --- Collect the referenced columns (projection pushdown) ---------------
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
                // The `expr` of `REPLACE (expr AS col, ...)` lives outside the expression tree
                // (inside the `*` expansion), so unless its referenced columns are collected
                // like any other select item, it would be missed by projection pushdown.
                // `EXCLUDE`'s columns, conversely, need not be read, so nothing is done for them.
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
    // A named window (`OVER w`) keeps the substance of its `PARTITION BY`/`ORDER BY` outside
    // the expression tree (in `sel.windows`), so ordinary recursion does not find it.
    // Only the names actually referenced are picked up, and their definitions' column
    // references are added to projection pushdown as well (an undefined name is not an error
    // here; the later `build_window` raises a clear error at actual bind time).
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
        // In a correlated subquery, WHERE may contain outer-scope column references.
        // This is only collecting columns for projection pushdown, so references resolvable in
        // the outer scope are silently excluded from pushdown (extracting the correlated
        // equality predicates themselves happens in the later WHERE decomposition).
        collect_refs_tolerant(arena, &scope_all, outer_scope, e, &mut refs)?;
    }
    for e in &group_by {
        // Resolve `GROUP BY 1` / `GROUP BY alias` to the select-list expression
        // first so pushdown reads the columns that expression actually uses.
        // Collecting the bare alias (`GROUP BY k` for `SELECT v+1 AS k`) would
        // look `k` up in the input scope and fail with `ColumnNotFound`.
        let e = resolve_group_ref(arena, sel, &scope_all, *e)?;
        if ordinal_of(arena, e).is_none() {
            collect_refs(arena, &scope_all, e, &mut refs)?;
        }
    }
    // `GROUPING SETS`/`ROLLUP`/`CUBE` are likewise included in projection pushdown.
    if let Some(sets) = &sel.grouping_sets {
        for set in sets {
            for &e in set {
                let e = resolve_group_ref(arena, sel, &scope_all, e)?;
                if ordinal_of(arena, e).is_none() {
                    collect_refs(arena, &scope_all, e, &mut refs)?;
                }
            }
        }
    }
    for o in &sel.order_by {
        // ORDER BY may point at an output alias, so failing to resolve is allowed here.
        let _ = collect_refs(arena, &scope_all, o.expr, &mut refs);
    }
    if let Some(q) = sel.qualify {
        // QUALIFY may point at a SELECT output alias too, so failing to resolve is allowed here.
        let _ = collect_refs(arena, &scope_all, q, &mut refs);
    }
    for &e in &sel.distinct_on {
        // DISTINCT ON may point at an output alias too, so failing to resolve is allowed here.
        let _ = collect_refs(arena, &scope_all, e, &mut refs);
    }
    collect_join_refs(arena, &scope_all, &tree, &mut refs)?;
    // The `expr` of a FROM-clause `UNNEST(expr) AS ...` is not part of `FromTree` but is kept
    // separately in `Rel::unnest`, so it is not covered by `collect_join_refs`'s traversal.
    // Without collecting it here, the referenced columns (the `tags` of `t.tags`, say) would be
    // missed by projection pushdown and the `Scan` would not read them.
    for r in rels.iter() {
        if let Some(u) = r.unnest {
            collect_refs(arena, &scope_all, u, &mut refs)?;
        }
    }
    // Outer columns a *subquery body* references (`... WHERE s.flag = t.flag` inside a
    // correlated subquery). Every pass above stops at the subquery boundary, so without this
    // the outer column would be pruned from the scan and the subquery would then fail to bind
    // against the narrowed scope. See `collect_outer_refs`.
    for item in &sel.items {
        collect_outer_refs(arena, &scope_all, item.expr, &mut refs, 0)?;
    }
    for e in [sel.filter, sel.having, sel.qualify].into_iter().flatten() {
        collect_outer_refs(arena, &scope_all, e, &mut refs, 0)?;
    }
    for &e in group_by.iter().chain(&sel.distinct_on) {
        collect_outer_refs(arena, &scope_all, e, &mut refs, 0)?;
    }
    for o in &sel.order_by {
        collect_outer_refs(arena, &scope_all, o.expr, &mut refs, 0)?;
    }

    if star_all {
        refs.extend(0..scope_all.len());
    }
    for q in &star_quals {
        refs.extend(scope_all.indices_for_qualifier(q));
    }
    refs.sort_unstable();
    refs.dedup();

    // Global indices are assigned per relation.
    {
        let mut off = 0usize;
        for r in rels.iter_mut() {
            let end = off + r.all.len();
            if r.subplan.is_none() {
                r.needed =
                    refs.iter().filter(|&&g| g >= off && g < end).map(|&g| g - off).collect();
                // Even a relation with no referenced columns needs its row count (`COUNT(*)`).
                // Exactly one, the cheapest, column is read. Passing an empty projection would
                // leave a row-oriented format with no way to return the row count.
                if r.needed.is_empty() && !r.all.is_empty() {
                    r.needed.push(0);
                }
            }
            off = end;
        }
    }

    let scope = narrow_scope(&rels);
    let ranges = rel_ranges(&rels);

    // --- Decomposing and pushing down WHERE ---------------------------------
    // With an outer join, applying a one-sided condition first would change the result of NULL
    // padding. Erring safe, nothing is pushed down.
    //
    // When `USING SAMPLE`/`TABLESAMPLE` is present, pushdown is disabled too: SAMPLE applies to
    // the joined FROM result before WHERE (confirmed against the `duckdb` CLI -- see
    // docs/sql/queries.md's "SAMPLE / TABLESAMPLE" section and the comment on `Node::Sample`'s
    // insertion point below). A single-relation conjunct pushed straight into the Scan (for
    // statistics pruning, ordinarily a pure win) would then run *before* the `Node::Sample`
    // interposed below, inverting that order -- the sample would end up drawn from the
    // filtered rows instead of the full FROM result. The predicate is still applied, just as
    // ordinary `leftover` filtering after Sample instead of pushed into the Scan; this only
    // costs statistics-pruning opportunities, and only for queries that combine WHERE with SAMPLE.
    let pushdown_ok = !tree.has_outer_join() && sel.sample.is_none();
    let mut conjuncts = Vec::new();
    if let Some(w) = sel.filter {
        split_conjuncts(arena, w, &mut conjuncts, 0)?;
    }
    let mut per_rel: Vec<Vec<ExprId>> = (0..rels.len()).map(|_| Vec::new()).collect();
    let mut leftover: Vec<ExprId> = Vec::new();
    // `EXISTS` / `IN (SELECT)` are rewritten into semi-joins, so they are kept apart from ordinary predicates.
    let mut semijoins: Vec<ExprId> = Vec::new();
    // Correlated equality predicates: pairs of (inner expression, outer-scope expression). Used
    // to rewrite a correlated subquery as a join with join keys (see the "correlated subquery" section below).
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
        // A predicate containing a subquery is not pushed down: the relations it references
        // cannot be counted correctly, and it could be dropped on the wrong side.
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
        // A correlation key goes through a "pick the first row/group" step (DistinctOn, or
        // merging into GROUP BY for an aggregate), so if the correlated subquery itself carries
        // its own ORDER BY/LIMIT/QUALIFY/window function, that meaning would apply to "the
        // whole thing after decorrelation" rather than "per correlation key" and change the
        // result. Erring safe, it is explicitly rejected.
        ensure!(
            sel.limit.is_none()
                && sel.offset.is_none()
                && sel.order_by.is_empty()
                && sel.qualify.is_none()
                // `DISTINCT ON` keys only the user columns, not the correlation
                // keys appended during decorrelation, so it can drop the wrong
                // outer rows. Plain `DISTINCT` is safe: it keys `0..visible`,
                // which already includes those columns.
                && sel.distinct_on.is_empty(),
            UnsupportedFeature
        );
        let mut win_probe = Vec::new();
        for item in &sel.items {
            collect_windows(arena, item.expr, &mut win_probe, 0)?;
        }
        ensure!(win_probe.is_empty(), UnsupportedFeature);
    }

    // --- Assembling scans and joins -----------------------------------------
    let mut node = build_tree(arena, &scope, &ranges, &mut rels, &tree, params, &per_rel, 0)?;

    // --- SAMPLE --------------------------------------------------------------
    // The semantics confirmed with the `duckdb` CLI: `USING SAMPLE`/`TABLESAMPLE` applies to the
    // FROM clause's raw data after joining and before filtering (`a JOIN b USING SAMPLE 20 ROWS`
    // picks 20 rows out of a 100-row join result, and writing `WHERE` first does not change the
    // measured row count). That is why it is interposed before `WHERE` (`sel.filter`; see the
    // "applying WHERE" section below).
    //
    // A single-relation `WHERE` conjunct would ordinarily already be pushed down into `per_rel`,
    // all the way to just after `Node::Scan` inside `build_tree` above (by design it happens
    // together with projection pushdown; see the module docs at the top), which would run
    // strictly before this `Node::Sample` -- the opposite of `duckdb`'s order. `pushdown_ok`
    // (see "Decomposing and pushing down WHERE" above) is set to disable that pushdown whenever
    // `sel.sample` is present, specifically to avoid that inversion: every `WHERE` conjunct ends
    // up in `leftover` instead and is applied via `Node::Filter` after this point (see the
    // "applying WHERE" section below), matching `duckdb`'s order at the cost of the
    // statistics-pruning benefit of pushing that predicate into the Scan.
    if let Some(spec) = &sel.sample {
        node = Node::Sample { input: Box::new(node), spec: resolve_sample_spec(spec) };
    }

    // --- Subqueries ---------------------------------------------------------
    // An uncorrelated subquery can be rewritten as a join. A scalar subquery becomes a LEFT
    // join against a "one-row table"; `EXISTS` / `IN` become semi-joins and anti-joins.
    let mut scope = scope;
    let mut subs: Vec<Substitution> = Vec::new();
    let mut scalars: Vec<ExprId> = Vec::new();
    for item in &sel.items {
        collect_scalar_subqueries(arena, item.expr, &mut scalars, 0)?;
    }
    for e in [sel.filter, sel.having].into_iter().flatten() {
        collect_scalar_subqueries(arena, e, &mut scalars, 0)?;
    }
    for e in &group_by {
        collect_scalar_subqueries(arena, *e, &mut scalars, 0)?;
    }
    for o in &sel.order_by {
        collect_scalar_subqueries(arena, o.expr, &mut scalars, 0)?;
    }
    // Whether this query aggregates. The authoritative value is `aggregating`, computed further
    // down; this early probe is needed here because an *uncorrelated* scalar subquery in an
    // aggregating query must be attached after the aggregate rather than before it. Attached
    // before, its column is neither a grouping key nor an aggregate result, so it cannot
    // survive the grouping — which is why such a query used to fail with `NotGrouped`.
    let will_aggregate = {
        let mut probe: Vec<ExprId> = Vec::new();
        for item in &sel.items {
            collect_aggregates(arena, item.expr, &mut probe, 0)?;
        }
        for e in [sel.having, sel.qualify].into_iter().flatten() {
            collect_aggregates(arena, e, &mut probe, 0)?;
        }
        for o in &sel.order_by {
            collect_aggregates(arena, o.expr, &mut probe, 0)?;
        }
        !probe.is_empty() || !group_by.is_empty() || sel.grouping_sets.is_some()
    };
    // Subqueries that WHERE or GROUP BY reads are consumed before the aggregate exists, so they
    // can never be deferred past it.
    let mut pre_only: Vec<ExprId> = Vec::new();
    if will_aggregate {
        if let Some(w) = sel.filter {
            collect_scalar_subqueries(arena, w, &mut pre_only, 0)?;
        }
        for e in &group_by {
            collect_scalar_subqueries(arena, *e, &mut pre_only, 0)?;
        }
    }
    // Uncorrelated scalar subqueries held back until after the aggregate: `(expr, plan, label)`.
    let mut deferred: Vec<(ExprId, Node, String)> = Vec::new();
    for (n, id) in scalars.iter().enumerate() {
        let q = match arena.get(*id) {
            Expr::ScalarSubquery(q) => q,
            _ => err!(Internal),
        };
        let plan = bind_query_in(catalog, arena, q, params, ctes, Some(&scope))?;
        let k = plan.correlated.len();
        // A scalar subquery must have exactly one column (excluding correlation key columns).
        ensure!(plan.root.schema().len() - k == 1, TypeMismatch);
        let ty = plan.root.schema()[0].ty;
        let mut label = String::from("subq");
        if k == 0 && will_aggregate && !pre_only.contains(id) {
            push_u32(&mut label, n as u32);
            deferred.push((*id, plan.root, label));
            continue;
        }
        push_u32(&mut label, n as u32);

        let mut right = plan.root;
        let mut left_keys = Vec::new();
        let mut right_keys = Vec::new();
        if k == 0 {
            // Uncorrelated: zero rows still gives NULL via the LEFT JOIN below. Two or more
            // rows is a cardinality error (matching the SQL standard and DuckDB), not silently
            // taking the first -- `Node::AssertMaxOneRow` raises `MultipleRowsSubquery` rather
            // than truncating. `Limit(2)` first bounds the cost of proving that to "one row
            // beyond the first", instead of `AssertMaxOneRow` alone potentially having to drain
            // the whole subquery to prove there is no second row.
            right = Node::Limit { input: Box::new(right), limit: Some(2), offset: 0 };
            right = Node::AssertMaxOneRow { input: Box::new(right), keys: Vec::new() };
        } else {
            // Correlated: each outer row has a different correlation key value, so a single
            // `AssertMaxOneRow` over the whole right side (as in the uncorrelated case above)
            // would wrongly reject as soon as any two *different* outer rows' subqueries each
            // produced one row. The check is instead per correlation key value, then LEFT joined
            // on that key: two-or-more rows for the *same* key is the cardinality error: it
            // means that one outer row's scalar subquery produced more than one row, which is
            // still a `MultipleRowsSubquery` error, just scoped per key instead of globally. For
            // correlation via an aggregate, the caller has already grouped so there is one row
            // per key, and this is effectively a no-op (never triggers).
            let corr_scope = Scope::from_fields(right.schema().to_vec());
            let mut dkeys = Vec::with_capacity(k);
            for i in 0..k {
                dkeys.push(column_program(&corr_scope, 1 + i)?);
            }
            right = Node::AssertMaxOneRow { input: Box::new(right), keys: dkeys };
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
            // The correlation key columns, having served as join keys, are dropped.
            node = drop_trailing_columns(node, k)?;
            // An outer row with no matching inner row for its correlation key gets a NULL value
            // column from this LEFT JOIN. Only for a `count`/`count(*)` correlated scalar
            // subquery should it be "aggregate over 0 rows -> 0" (confirmed with DuckDB), so the
            // NULL is corrected to 0 here.
            if matches!(as_bare_aggregate(arena, q), Some(AggKind::Count | AggKind::CountStar)) {
                node = coalesce_count_column(node, col)?;
            }
        }
        scope.push(None, Field::new(label.clone(), ty, true));
        subs.push(Substitution { expr: *id, column: col, structural: false });
    }

    // `ANY`/`ALL` with `>`/`<`/`>=`/`<=` (uncorrelated only). `= ANY`/`<> ALL` are handled by
    // the `semijoins` loop below via `is_semijoin_predicate` and never reach here (see the
    // `collect_quantified_comparisons` docs).
    // It follows the same "add one column with a join and swap it in with `Substitution`"
    // pattern as a scalar subquery, so its position is independent (anywhere before `semijoins`/`leftover`).
    let mut quantifieds: Vec<ExprId> = Vec::new();
    for item in &sel.items {
        collect_quantified_comparisons(arena, item.expr, &mut quantifieds, 0)?;
    }
    for e in [sel.filter, sel.having].into_iter().flatten() {
        collect_quantified_comparisons(arena, e, &mut quantifieds, 0)?;
    }
    for e in &group_by {
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

    // --- Correlated aggregate scalar subqueries (a restricted pattern) ------
    // Only the restricted form "SELECT exactly one bare aggregate call" is completed here (the
    // basic form of magic decorrelation: build an aggregate grouped by the correlation key, then
    // have the caller LEFT/SEMI/ANTI join on that key). It returns early without going through
    // the rest of this function (the ordinary aggregate section from line 670 on, left untouched
    // because it would conflict with the GROUPING SETS/ROLLUP/CUBE implementation). Every other
    // combination (an aggregate subquery with its own GROUP BY/HAVING/QUALIFY/DISTINCT/window
    // function, and so on) is explicitly rejected, since there is no way to merge the correlation
    // key into the aggregate's grouping (silently ignoring the correlation predicate would mix
    // the aggregate across outer rows and give a wrong result).
    if outer_scope.is_some() && !correlated_eq.is_empty() {
        let mut corr_agg_probe: Vec<ExprId> = Vec::new();
        for item in &sel.items {
            collect_aggregates(arena, item.expr, &mut corr_agg_probe, 0)?;
        }
        if let Some(h) = sel.having {
            collect_aggregates(arena, h, &mut corr_agg_probe, 0)?;
        }
        let corr_aggregates =
            !corr_agg_probe.is_empty() || !group_by.is_empty() || sel.grouping_sets.is_some();
        if corr_aggregates {
            ensure!(
                sel.items.len() == 1
                    && corr_agg_probe.len() == 1
                    && sel.items[0].expr == corr_agg_probe[0]
                    && group_by.is_empty()
                    && sel.grouping_sets.is_none()
                    && sel.having.is_none()
                    && sel.qualify.is_none()
                    && !sel.distinct
                    && sel.distinct_on.is_empty()
                    // This path returns early, before deferred scalar subqueries are attached.
                    && deferred.is_empty(),
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
            // When "no inner row matches a given correlation key value", that combination does
            // not appear in the GROUP BY result at all (not as a NULL-valued row -- the row
            // itself is absent). That surfaces on the caller's side (a LEFT JOIN for a scalar
            // subquery) as the correlation key not matching and becoming NULL, so the COUNT
            // family's "aggregate over 0 rows -> 0" correction is done by the caller rather than
            // here (see `as_bare_aggregate`).
            // The column order matches the caller's convention (index 0 = value, 1.. = correlation keys).
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

    // --- UNNEST (in the SELECT list) ----------------------------------------
    // Like `FILTER`/`QUALIFY`, it is picked up as a special expression that is neither an
    // aggregate nor an ordinary scalar expression. Because it expands the target column's JSON
    // array into as many rows as it has elements and duplicates the other columns, it "adds
    // rows", so it is interposed before aggregation (right after FROM/WHERE, before GROUP BY).
    // The semantics of aggregating over the expanded rows is not implemented, so it cannot be
    // used together with aggregation. DuckDB's behavior for several `UNNEST`s in one SELECT list
    // (per-column zip, NULL padding when element counts differ) is too complex, so it is out of
    // scope: exactly one is allowed and the rest are explicitly rejected.
    let mut unnest_calls: Vec<ExprId> = Vec::new();
    for item in &sel.items {
        collect_unnests(arena, item.expr, &mut unnest_calls, 0)?;
    }
    if !unnest_calls.is_empty() {
        // UNNEST inside a correlated subquery (whose interaction with decorrelation gets
        // complicated) is out of scope and explicitly rejected.
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
            group_by.is_empty() && sel.grouping_sets.is_none() && sel.having.is_none(),
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

    // --- Aggregation --------------------------------------------------------
    let mut agg_calls: Vec<ExprId> = Vec::new();
    for item in &sel.items {
        collect_aggregates(arena, item.expr, &mut agg_calls, 0)?;
    }
    if let Some(h) = sel.having {
        collect_aggregates(arena, h, &mut agg_calls, 0)?;
    }
    if let Some(q) = sel.qualify {
        collect_aggregates(arena, q, &mut agg_calls, 0)?;
    }
    for o in &sel.order_by {
        collect_aggregates(arena, o.expr, &mut agg_calls, 0)?;
    }
    // Aggregates cannot be written in WHERE (HAVING exists for that).
    if let Some(w) = sel.filter {
        let mut in_where = Vec::new();
        collect_aggregates(arena, w, &mut in_where, 0)?;
        ensure!(in_where.is_empty(), NotAggregate);
    }

    // Collects the `GROUPING`/`GROUPING_ID` calls. Unlike aggregates, their arguments are not
    // evaluated; they are replaced by a constant (a bitmask) fixed per grouping set.
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

    let aggregating = !agg_calls.is_empty() || !group_by.is_empty() || sel.grouping_sets.is_some();
    // GROUPING() is meaningless outside aggregation.
    ensure!(aggregating || grouping_calls.is_empty(), NotAggregate);
    let mut item_scope = scope.clone();
    // The expressions of the deferred (uncorrelated, hence grouping-invariant) scalar
    // subqueries, which `check_grouped` must let through and which are attached to the plan
    // right after the aggregate below.
    let const_subs: Vec<ExprId> = deferred.iter().map(|&(id, _, _)| id).collect();
    // A HAVING reading one of them cannot be embedded into `Node::Aggregate` (the column does
    // not exist yet at that point); it becomes a Filter applied after the attachment instead.
    let having_deferred = match sel.having {
        Some(h) if !const_subs.is_empty() => {
            let mut hs = Vec::new();
            collect_scalar_subqueries(arena, h, &mut hs, 0)?;
            hs.iter().any(|e| const_subs.contains(e))
        }
        _ => false,
    };

    if aggregating && sel.grouping_sets.is_none() && grouping_calls.is_empty() {
        // A plain `GROUP BY a, b, ...` (no GROUPING SETS-family extension).
        // The conventional path, embedding havings directly into a single existing Node::Aggregate.
        let mut group_exprs = Vec::new();
        for g in &group_by {
            group_exprs.push(resolve_group_ref(arena, sel, &scope, *g)?);
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

        // Rejects bare column references absent from GROUP BY. A column outside the aggregate has no determined value.
        for item in &sel.items {
            check_grouped(arena, &scope, item.expr, &group_exprs, &agg_calls, &const_subs, 0)?;
        }
        if let Some(h) = sel.having {
            check_grouped(arena, &scope, h, &group_exprs, &agg_calls, &const_subs, 0)?;
        }
        add_equivalent_group_subs(arena, &scope, sel, &group_exprs, &mut subs)?;

        let agg_scope = Scope::from_fields(out_fields.clone());
        let having = match sel.having {
            Some(h) if !having_deferred => {
                Some(compile_predicate_with_subs(arena, &agg_scope, params, &subs, h)?)
            }
            _ => None,
        };
        node = Node::Aggregate { input: Box::new(node), groups, aggs, schema: out_fields, having };
        item_scope = agg_scope;
    } else if aggregating {
        // `GROUPING SETS`/`ROLLUP`/`CUBE`, or a plain GROUP BY carrying a `GROUPING()`. A plain
        // GROUP BY is handled by the same path as "exactly one grouping set".
        //
        // One Node::Aggregate is built per grouping set and they are bundled with UNION ALL
        // (Node::SetOp). Columns not in a given set are filled with a NULL constant (the same
        // behavior as DuckDB). The FROM/WHERE input is identical for every set, so `node` (with
        // WHERE already applied at this point) is duplicated once per set.
        let sets: Vec<Vec<ExprId>> = match &sel.grouping_sets {
            Some(sets) => sets.clone(),
            None => vec![group_by.clone()],
        };
        ensure!(!sets.is_empty(), Internal);

        // The union of the grouping columns across all sets is treated as "the grouping columns"
        // (structurally equal columns are merged into one). GROUP BY ordinals and aliases are
        // resolved as in an ordinary GROUP BY.
        let mut resolved_sets: Vec<Vec<ExprId>> = Vec::with_capacity(sets.len());
        let mut group_exprs: Vec<ExprId> = Vec::new();
        for set in &sets {
            let mut rs = Vec::with_capacity(set.len());
            for &g in set {
                let r = resolve_group_ref(arena, sel, &scope, g)?;
                if !group_exprs.iter().any(|&e| expr_eq(arena, e, r)) {
                    group_exprs.push(r);
                }
                rs.push(r);
            }
            resolved_sets.push(rs);
        }

        // Each column is compiled exactly once. The input scope (`scope`) is the same for every
        // set, so it can be reused across them as is (`Program` is `Clone`).
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

        // Aggregates share the same input scope too, so they are built once and cloned per set.
        let mut aggs: Vec<Agg> = Vec::with_capacity(agg_calls.len());
        for (j, &call) in agg_calls.iter().enumerate() {
            let a = build_agg(arena, &scope, params, call)?;
            out_fields.push(Field::new(a.name.clone(), a.result_ty()?, true));
            subs.push(Substitution { expr: call, column: ngroups + j, structural: true });
            aggs.push(a);
        }
        let base_cols = ngroups + agg_calls.len();

        // Rejects bare column references absent from GROUP BY. Under GROUPING SETS the union
        // across sets counts as "the grouping columns", so referencing bare in SELECT a column
        // absent from one set is not an error (it is simply NULL in those rows, as in DuckDB).
        for item in &sel.items {
            check_grouped(arena, &scope, item.expr, &group_exprs, &agg_calls, &const_subs, 0)?;
        }
        if let Some(h) = sel.having {
            check_grouped(arena, &scope, h, &group_exprs, &agg_calls, &const_subs, 0)?;
        }
        add_equivalent_group_subs(arena, &scope, sel, &group_exprs, &mut subs)?;

        // The arguments of GROUPING()/GROUPING_ID() must be grouping columns.
        // Only which column (an index into `group_exprs`) each argument points at is remembered;
        // the value is computed per set (the first argument is the highest bit, as in DuckDB).
        let mut grouping_arg_idx: Vec<Vec<usize>> = Vec::with_capacity(grouping_calls.len());
        for &gc in &grouping_calls {
            let args = match arena.get(gc) {
                Expr::Function { args, .. } => args.clone(),
                _ => err!(Internal),
            };
            ensure!(!args.is_empty(), WrongArgCount);
            let mut idxs = Vec::with_capacity(args.len());
            for a in args {
                let r = resolve_group_ref(arena, sel, &scope, a)?;
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

        // HAVING is evaluated against the final schema, once the grouping columns, aggregate
        // results, and GROUPING() constants are all present. It is not embedded into each set's
        // Aggregate but applied once as a Filter after the UNION ALL bundle (evaluation is
        // per-row, so either gives the same result).
        let agg_scope = Scope::from_fields(out_fields.clone());
        let having = match sel.having {
            Some(h) if !having_deferred => {
                Some(compile_predicate_with_subs(arena, &agg_scope, params, &subs, h)?)
            }
            _ => None,
        };

        let base_fields: Vec<Field> = out_fields[..base_cols].to_vec();
        let mut branches: Vec<Node> = Vec::with_capacity(resolved_sets.len());
        for set in &resolved_sets {
            // `()` is "group by nothing": one row even on empty input (`count(*) = 0`).
            // Planning it as `GROUP BY NULL, NULL, …` would make HashAggregate treat it
            // as a real GROUP BY and emit zero rows for empty input.
            let empty_set = set.is_empty() && ngroups > 0;
            let agg_node = if empty_set {
                Node::Aggregate {
                    input: Box::new(node.clone()),
                    groups: Vec::new(),
                    aggs: aggs.clone(),
                    schema: base_fields[ngroups..].to_vec(),
                    having: None,
                }
            } else {
                let mut set_groups = Vec::with_capacity(ngroups);
                for (i, &g) in group_exprs.iter().enumerate() {
                    if set.iter().any(|&s| expr_eq(arena, s, g)) {
                        set_groups.push(group_progs[i].clone());
                    } else {
                        set_groups.push(const_program(group_progs[i].result_ty, Value::Null));
                    }
                }
                Node::Aggregate {
                    input: Box::new(node.clone()),
                    groups: set_groups,
                    aggs: aggs.clone(),
                    schema: base_fields.clone(),
                    having: None,
                }
            };
            let need_project = empty_set || !grouping_calls.is_empty();
            let branch = if !need_project {
                agg_node
            } else {
                // GROUPING()'s result is neither a group key nor an aggregate result, so it does
                // not appear in Node::Aggregate's schema. It is added as a constant column by a
                // Project. The empty-set branch also uses a Project to put NULL in every
                // grouping column in front of the no-key aggregate's output.
                let mut exprs = Vec::with_capacity(out_fields.len());
                if empty_set {
                    for p in group_progs.iter().take(ngroups) {
                        exprs.push(const_program(p.result_ty, Value::Null));
                    }
                    let agg_scope = Scope::from_fields(base_fields[ngroups..].to_vec());
                    for i in 0..aggs.len() {
                        exprs.push(column_program(&agg_scope, i)?);
                    }
                } else {
                    let branch_scope = Scope::from_fields(base_fields.clone());
                    for i in 0..base_cols {
                        exprs.push(column_program(&branch_scope, i)?);
                    }
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
        // `sets` was never emptied (see the ensure! above), so at least one is always available.
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

    // --- Uncorrelated scalar subqueries held back past the aggregate --------
    // Attached here, between the aggregate and the window functions, by the same keyless LEFT
    // JOIN used before the aggregate for the non-aggregating case. Being uncorrelated, the
    // subquery yields the same single value for every row, so where the join sits does not
    // change the result — only whether the value survives the grouping.
    for (id, right, label) in deferred {
        let ty = right.schema()[0].ty;
        // The same cardinality check as the pre-aggregate path: two or more rows is an error,
        // and `Limit(2)` bounds the cost of proving it.
        let right = Node::Limit { input: Box::new(right), limit: Some(2), offset: 0 };
        let right = Node::AssertMaxOneRow { input: Box::new(right), keys: Vec::new() };
        let col = item_scope.len();
        let mut full_schema = node.schema().to_vec();
        full_schema.extend_from_slice(right.schema());
        node = Node::Join {
            left: Box::new(node),
            right: Box::new(right),
            kind: JoinKind::Left,
            left_keys: Vec::new(),
            right_keys: Vec::new(),
            residual: None,
            schema: full_schema,
        };
        item_scope.push(None, Field::new(label, ty, true));
        subs.push(Substitution { expr: id, column: col, structural: false });
    }
    if having_deferred {
        if let Some(h) = sel.having {
            let pred = compile_predicate_with_subs(arena, &item_scope, params, &subs, h)?;
            node = Node::Filter { input: Box::new(node), pred };
        }
    }

    // --- Window functions ---------------------------------------------------
    // Evaluated **after** aggregation. The x in `sum(x) OVER ()` may point at an aggregate
    // result, so the input scope is the aggregate's output (or the scan/join output if there is no aggregate).
    let mut win_calls: Vec<ExprId> = Vec::new();
    for item in &sel.items {
        collect_windows(arena, item.expr, &mut win_calls, 0)?;
    }
    for o in &sel.order_by {
        collect_windows(arena, o.expr, &mut win_calls, 0)?;
    }
    if let Some(q) = sel.qualify {
        // QUALIFY is the only place outside SELECT where a window function's result can be written directly.
        collect_windows(arena, q, &mut win_calls, 0)?;
    }
    // Window functions cannot be written in WHERE or HAVING (they are evaluated later).
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
        // The window output is "the input's columns ++ the window columns", so existing column
        // numbers do not shift and the aggregate replacements still apply. The input columns
        // are appended onto the existing scope rather than rebuilt with `Scope::from_fields`,
        // which would drop their table qualifiers and make `SELECT e.id, row_number() OVER ()`
        // fail to resolve `e`.
        for f in fields.into_iter().skip(base) {
            item_scope.push(None, f);
        }
    }

    // --- Projection ---------------------------------------------------------
    let mut exprs = Vec::new();
    let mut schema = Vec::new();
    for item in &sel.items {
        match arena.get(item.expr) {
            Expr::Star { qualifier, columns, exclude, replace, rename } => {
                // `*` cannot be expanded after aggregation (the original rows are gone).
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
                // Validates that the column names written in `EXCLUDE`/`REPLACE` really exist
                // (`duckdb` rejects them at bind time as "Column ... not found").
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
                            // A REPLACE expression is compiled in the same scope as an ordinary
                            // select item (`item_scope`, which may include aggregate and window
                            // output). The column name itself is left unchanged.
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
    // How many columns `ORDER BY ALL` sorts by. It is settled here so the correlation key
    // columns (implementation hidden columns appended below) are not included.
    let projected = exprs.len();
    // The correlation key columns are appended at the end of the output (in the non-aggregate
    // case; correlation with aggregation is completed on the early-return path above and never
    // reaches here). The caller (binding of a correlated scalar subquery / `EXISTS` / `IN`) uses
    // them as join keys, reading as many from the end as `Plan::correlated` has. To keep them
    // off the "anything beyond `visible` is dropped at the end" mechanism used by DISTINCT and
    // ORDER BY hidden columns, `visible` is settled after including these columns (otherwise the
    // final hidden-column trim would drop them too).
    for &(inner_e, _) in &correlated_eq {
        let p = compile(arena, &scope, params, inner_e)?;
        schema.push(Field::new(String::new(), p.result_ty, true));
        exprs.push(p);
    }
    let visible = exprs.len();

    // --- QUALIFY hidden columns ---------------------------------------------
    // QUALIFY is evaluated against the SELECT output (after `*` REPLACE/RENAME
    // and last-wins aliases), so it is compiled after this Project. Window
    // functions, aggregates, and input columns that are not themselves output
    // columns are added here as hidden columns and dropped by the trailing
    // trim — the same mechanism ORDER BY uses for sort keys absent from SELECT.
    let mut qualify_subs: Vec<Substitution> = Vec::new();
    if let Some(q) = sel.qualify {
        let mut q_wins = Vec::new();
        collect_windows(arena, q, &mut q_wins, 0)?;
        for &w in &q_wins {
            let p = compile_with_subs(arena, &item_scope, params, &subs, w)?;
            schema.push(Field::new(String::new(), p.result_ty, true));
            exprs.push(p);
            qualify_subs.push(Substitution { expr: w, column: exprs.len() - 1, structural: true });
        }
        let mut q_aggs = Vec::new();
        collect_aggregates(arena, q, &mut q_aggs, 0)?;
        for &a in &q_aggs {
            let p = compile_with_subs(arena, &item_scope, params, &subs, a)?;
            schema.push(Field::new(String::new(), p.result_ty, true));
            exprs.push(p);
            qualify_subs.push(Substitution { expr: a, column: exprs.len() - 1, structural: true });
        }
        let mut q_refs = Vec::new();
        collect_colrefs(arena, q, &mut q_refs, 0)?;
        for (rid, qual, rname) in q_refs {
            let out_hit = if qual.is_none() {
                schema.iter().rposition(|f| {
                    !f.name.is_empty() && eq_ascii_ci(f.name.as_bytes(), rname.as_bytes())
                })
            } else {
                None
            };
            if let Some(col) = out_hit {
                qualify_subs.push(Substitution { expr: rid, column: col, structural: false });
                continue;
            }
            let p = compile_with_subs(arena, &item_scope, params, &subs, rid)?;
            schema.push(Field::new(String::new(), p.result_ty, true));
            exprs.push(p);
            qualify_subs.push(Substitution {
                expr: rid,
                column: exprs.len() - 1,
                structural: false,
            });
        }
    }

    // --- ORDER BY -----------------------------------------------------------
    // When sorting by an expression absent from the output, it is added to the projection as a hidden column and dropped afterwards.
    let mut keys = Vec::new();
    // `ORDER BY ALL`: sorts by the output columns left to right, all in the same direction and
    // with the same NULL placement. By this point `*` is already expanded, so
    // `SELECT * ... ORDER BY ALL` targets every expanded column too (as in DuckDB). Aggregate
    // result columns are included as well (confirmed that `duckdb -c "select h, sum(x) from t
    // group by h order by all"` sorts by h then sum(x)).
    if let Some(oa) = &sel.order_by_all {
        for col in 0..projected {
            keys.push((col, oa.desc, oa.nulls_first));
        }
    }
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
    // ON expressions are resolved by the same rule as ORDER BY -- first check for a structural
    // match with an output column, and add a hidden column otherwise. The actual deduplication
    // happens after the Sort, in a streaming filter that passes only the first row per key in
    // the input's order (confirmed with DuckDB: without ORDER BY, arrival order is "the first row").
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

    // QUALIFY filters the projected rows. Unqualified names resolve to the
    // last output column of that name (`* REPLACE`, a trailing `AS` that
    // shadows a star column, `RENAME`), matching DuckDB.
    if let Some(q) = sel.qualify {
        let pred = compile_predicate_with_subs(arena, &project_scope, params, &qualify_subs, q)?;
        node = Node::Filter { input: Box::new(node), pred };
    }

    // `SELECT DISTINCT a ORDER BY b` adds `b` as a hidden column. Deduplicating
    // on that hidden key as well would keep one row per (a, b) and then drop
    // `b`, returning duplicate `a`s. When a sort key sits past `visible`,
    // DISTINCT is applied *after* the sort as first-row-wins on the visible
    // columns (the same operator as `DISTINCT ON`).
    let distinct_after_sort = sel.distinct && keys.iter().any(|(col, _, _)| *col >= visible);

    // --- DISTINCT -----------------------------------------------------------
    if sel.distinct && !distinct_after_sort {
        // Group only the visible output (plus correlation keys). Hidden sort
        // columns are not present on this path.
        let mut groups = Vec::new();
        let mut out_fields = Vec::new();
        for i in 0..visible {
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

    // --- Sorting ------------------------------------------------------------
    if !keys.is_empty() {
        let mut sort_keys = Vec::with_capacity(keys.len());
        for (col, desc, nulls_first) in keys {
            sort_keys.push(SortKey {
                expr: column_program(&project_scope, col)?,
                desc,
                nulls_first,
            });
        }
        // `ORDER BY ... LIMIT n OFFSET k` only needs to hold the top n+k. Lowering to a Top-N
        // avoids buffering everything. With DISTINCT ON (or DISTINCT that must wait until
        // after the sort) present, which row wins is decided by "deduplication after
        // sorting", so truncating with Top-N first would drop the correct representative.
        let topn = if sel.distinct_on.is_empty() && !distinct_after_sort {
            sel.limit.and_then(|l| usize::try_from(l.saturating_add(sel.offset.unwrap_or(0))).ok())
        } else {
            None
        };
        node = Node::Sort { input: Box::new(node), keys: sort_keys, limit: topn };
    }

    // --- DISTINCT ON (the substance) ------------------------------------------
    if !distinct_on_cols.is_empty() {
        let s = Scope::from_fields(node.schema().to_vec());
        let mut on_progs = Vec::with_capacity(distinct_on_cols.len());
        for c in distinct_on_cols {
            on_progs.push(column_program(&s, c)?);
        }
        node = Node::DistinctOn { input: Box::new(node), keys: on_progs };
    } else if distinct_after_sort {
        let s = Scope::from_fields(node.schema().to_vec());
        let mut on_progs = Vec::with_capacity(visible);
        for i in 0..visible {
            on_progs.push(column_program(&s, i)?);
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

    // Drops the hidden sort columns.
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
