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
use super::refs::each_child;
use super::subquery::{and_all, equi_key, split_conjuncts, unify_key_types};
use super::*;

// --- Flattening the FROM clause ----------------------------------------------

/// One relation appearing in FROM.
///
/// `pub(super)`: `select::bind_select_in` builds/reads `Rel`s directly
/// (projection pushdown, `UNNEST` column collection), so the type and every
/// field it touches must be visible to that sibling submodule.
pub(super) struct Rel {
    /// The table index in the catalog. `None` for a subquery.
    pub(super) table: Option<usize>,
    /// The qualifier (the alias, or the table name if there is none).
    pub(super) alias: String,
    /// Every column the relation has.
    pub(super) all: Vec<Field>,
    /// The referenced columns (indices into `all`, ascending and without duplicates).
    pub(super) needed: Vec<usize>,
    /// The plan, for a subquery.
    pub(super) subplan: Option<Node>,
    /// The `expr` of a FROM-clause `UNNEST(expr) AS alias(col)` (an unbound AST expression).
    /// When `Some`, the other fields are only shaped as a "dummy one-column relation"
    /// (`table`/`subplan` are unused), and the real plan is assembled specially by
    /// `build_tree` wrapping the left sibling node (the LATERAL equivalent; see the
    /// corresponding comments in `flatten_from`/`build_tree`).
    pub(super) unnest: Option<ExprId>,
}

/// The tree structure of FROM. Leaves are indices into `rels`.
///
/// `pub(super)`: matched directly by `refs::collect_join_refs`.
pub(super) enum FromTree {
    Rel(usize),
    Join { left: Box<FromTree>, right: Box<FromTree>, kind: JoinKind, on: Option<ExprId> },
}

impl FromTree {
    /// Whether the tree contains at least one outer join. Used to decide whether predicates may be pushed down.
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
            // CTEs are checked first. A CTE wins over a table of the same name (per the SQL standard).
            if let Some(k) = ctes.find(name) {
                let (resolved, all) = ctes.resolve(k)?;
                let alias = alias.clone().unwrap_or_else(|| name.clone());
                let subplan = match resolved {
                    ResolvedCte::Plan(node) => *node,
                    // A self-reference from a recursive CTE's own recursive term. It carries
                    // no real data; at runtime the `RecursiveCte` operator feeds in the
                    // previous iteration's new rows (see `exec::recursive`).
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
            // If not among the file tables, in-memory tables and views are checked
            // (`ddl` feature only). Priority: CTE > file table > in-memory table > view.
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
            // A FROM-clause derived table cannot correlate with the outer scope (LATERAL is unsupported).
            let plan = bind_query_in(catalog, arena, query, params, ctes, None)?;
            let all = plan.root.schema().to_vec();
            // A derived table without an alias cannot be referenced with a qualifier, but unqualified references pass.
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
            // Implicit LATERAL: `expr` may reference only the columns the left siblings have
            // contributed so far. `rels` holds the siblings so far in left-to-right order, so
            // the `full_scope` at that moment is exactly "the range that may be visible"
            // (`build_tree` reproduces the same scope when it actually assembles things.
            // Column order may change due to projection pushdown, but columns are resolved by
            // name, so that does not matter).
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
            // `step == 0` is a bind-time error in `duckdb` too
            // ("Binder Error: interval cannot be 0!", confirmed with the `duckdb` CLI).
            ensure!(*step != 0, DivideByZero);
            let name = column_alias.clone().unwrap_or_else(|| {
                String::from(if *inclusive { "generate_series" } else { "range" })
            });
            // `duckdb`'s `DESCRIBE` reports the BIGINT column as nullable (it never actually
            // produces NULL, but the declaration is left loose -- the same judgment as
            // `Unnest`'s expanded element).
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
    // By contract the caller resolves the schema before getting here.
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

/// Inserts a `catalog::MemTable` as a `Rel`. Unlike a file table, the scan itself is carried
/// as a "subplan", `Node::MemScan` (`Rel::subplan` was originally a mechanism for CTEs and
/// derived tables, but it can be reused as is).
/// There is no projection pushdown (every column is emitted; see the `MemScanSpec` docs).
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

/// Reparses and rebinds the raw SQL registered by `CREATE VIEW` and inserts it as a `Rel`.
/// A view has no substance, so this happens on every reference (no caching -- a
/// simplification assuming tables are small and re-referencing a view is rare).
#[cfg(feature = "ddl")]
fn push_view_rel(
    catalog: &Catalog,
    i: usize,
    alias: String,
    params: &[Value],
    rels: &mut Vec<Rel>,
    ctes: &CteScope,
) -> Result<FromTree> {
    ensure!(ctes.view_depth < MAX_VIEW_DEPTH, ExpressionTooDeep);
    let sql = match catalog.view_get(i) {
        Some(s) => s.to_owned(),
        None => err!(TableNotFound),
    };
    let parsed = crate::sql::parse(&sql)?;
    // Views are stored queries, not prepared statements. A `?` in the body would be
    // re-numbered from zero on this separate parse and steal the outer query's
    // parameters, so it is rejected rather than silently binding the wrong values.
    ensure!(parsed.num_params == 0, UnsupportedFeature);
    let q = match parsed.stmt {
        Stmt::Select(q) => q,
        _ => err!(Internal),
    };
    // A view is bound in its own CTE scope so an outer `WITH t AS (...)` cannot
    // shadow the view's base table `t`. `view_depth` is copied so nested view
    // references still share the expansion-depth cap.
    let mut view_ctes = CteScope::default();
    view_ctes.view_depth = ctes.view_depth + 1;
    let plan = bind_query_in(catalog, &parsed.arena, &q, params, &mut view_ctes, None)?;
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

/// Builds the scope of all columns from a sequence of relations.
pub(super) fn full_scope(rels: &[Rel]) -> Scope {
    let mut s = Scope::new();
    for r in rels {
        for f in &r.all {
            s.push(qual_of(r), f.clone());
        }
    }
    s
}

/// The narrowed scope reflecting `needed`.
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

/// Each relation's column range `[start, end)` in the narrowed scope.
pub(super) fn rel_ranges(rels: &[Rel]) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(rels.len());
    let mut off = 0;
    for r in rels {
        out.push((off, off + r.needed.len()));
        off += r.needed.len();
    }
    out
}

// --- Building the tree -------------------------------------------------------

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
            // A FROM-clause `UNNEST` has no independent relation (see the `Rel::unnest` docs).
            // Reaching here means `UNNEST` was placed alone (`FROM UNNEST(...)`) or on the
            // left of a JOIN, without going through the special `FromTree::Join` branch below
            // -- explicitly rejected as out of scope.
            ensure!(rels[*i].unnest.is_none(), UnsupportedFeature);
            let (start, end) = ranges[*i];
            // A scope seeing only this relation. Pushed-down predicates are evaluated here.
            let rel_scope = sub_scope(scope, start, end);

            let mut node = match rels[*i].subplan.take() {
                // A derived table's plan is inserted as is. Its columns are already settled.
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
            // A FROM-clause `UNNEST` is not an independent relation but a LATERAL-equivalent
            // operation expanding an array per row produced by the left sibling. It cannot be
            // lowered to a symmetric `Node::Join`, so if the right side is an `UNNEST` relation
            // it is assembled specially (see the `Rel::unnest`/`FromItem::Unnest` docs).
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

            // The join's input/output scope. The left schema ++ the right schema.
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

/// The scope range a subtree occupies, assuming leaves are laid out contiguously.
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

/// Gets the table index from a FROM clause (for `DESCRIBE`).
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
        // `GenerateSeries` is a compute-only source with no substance in the catalog, so there
        // is no way to return the "table index" `DESCRIBE` expects.
        FromItem::Join { .. }
        | FromItem::Subquery { .. }
        | FromItem::Unnest { .. }
        | FromItem::GenerateSeries { .. } => {
            err!(UnsupportedFeature)
        }
    }
}

/// Collects the tables referenced within a query tree.
///
/// This must find *every* table the query will end up binding against,
/// including ones that only appear inside a subquery in an expression
/// (`SELECT (SELECT max(c) FROM u) FROM t`, `WHERE x IN (SELECT ...)`,
/// `WHERE EXISTS (...)`, `ORDER BY (SELECT ...)`, ...). The caller
/// (`Session::prepare` via `resolve_query`, or `ddl::ctas_rows`) resolves
/// the schema of everything reported here *before* binding, and
/// `push_table_rel` asserts that invariant with an `Internal` error — so a
/// table missed here does not degrade gracefully, it surfaces as a bare
/// "internal error" to the user.
pub fn referenced_in_query(
    catalog: &Catalog,
    arena: &ExprArena,
    q: &QueryStmt,
    out: &mut Vec<usize>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_FROM_DEPTH, ExpressionTooDeep);
    for c in &q.ctes {
        referenced_in_query(catalog, arena, &c.query, out, depth + 1)?;
    }
    for o in &q.order_by {
        referenced_in_expr(catalog, arena, o.expr, out, depth + 1)?;
    }
    referenced_in_set_expr(catalog, arena, &q.body, out, depth + 1)
}

fn referenced_in_set_expr(
    catalog: &Catalog,
    arena: &ExprArena,
    e: &SetExpr,
    out: &mut Vec<usize>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_FROM_DEPTH, ExpressionTooDeep);
    let d = depth + 1;
    match e {
        SetExpr::Select(s) => {
            if let Some(f) = &s.from {
                // CTE names are not in the catalog, so not finding one is left to the later binding.
                let _ = referenced_tables_at(catalog, arena, f, out, d);
            }
            // Every expression position of the SELECT can host a subquery.
            for item in &s.items {
                referenced_in_expr(catalog, arena, item.expr, out, d)?;
            }
            for e in [s.filter, s.having, s.qualify].into_iter().flatten() {
                referenced_in_expr(catalog, arena, e, out, d)?;
            }
            for &e in s.group_by.iter().chain(&s.distinct_on) {
                referenced_in_expr(catalog, arena, e, out, d)?;
            }
            if let Some(sets) = &s.grouping_sets {
                for set in sets {
                    for &e in set {
                        referenced_in_expr(catalog, arena, e, out, d)?;
                    }
                }
            }
            for (_, def) in &s.windows {
                for &e in &def.partition_by {
                    referenced_in_expr(catalog, arena, e, out, d)?;
                }
                for o in &def.order_by {
                    referenced_in_expr(catalog, arena, o.expr, out, d)?;
                }
            }
            for o in &s.order_by {
                referenced_in_expr(catalog, arena, o.expr, out, d)?;
            }
            Ok(())
        }
        SetExpr::SetOp { left, right, .. } => {
            referenced_in_set_expr(catalog, arena, left, out, d)?;
            referenced_in_set_expr(catalog, arena, right, out, d)
        }
    }
}

/// Collects the tables referenced by subqueries inside an expression.
///
/// Deliberately *not* built on `refs::each_child`, which stops at a
/// subquery boundary (correct for scope resolution — the inside of a
/// subquery resolves against its own scope — but exactly backwards here,
/// where the whole point is to reach the tables inside). The two star and
/// lambda positions `each_child` also skips are walked here for the same
/// reason: a subquery is syntactically valid in both
/// (`SELECT * REPLACE ((SELECT max(c) FROM u) AS b) FROM t`).
fn referenced_in_expr(
    catalog: &Catalog,
    arena: &ExprArena,
    id: ExprId,
    out: &mut Vec<usize>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_FROM_DEPTH, ExpressionTooDeep);
    let d = depth + 1;
    match arena.get(id) {
        Expr::ScalarSubquery(q) => referenced_in_query(catalog, arena, q, out, d),
        Expr::Exists { query, .. } => referenced_in_query(catalog, arena, query, out, d),
        Expr::InSubquery { arg, query, .. } | Expr::QuantifiedComparison { arg, query, .. } => {
            referenced_in_expr(catalog, arena, *arg, out, d)?;
            referenced_in_query(catalog, arena, query, out, d)
        }
        Expr::Star { replace, .. } => {
            for &(e, _) in replace {
                referenced_in_expr(catalog, arena, e, out, d)?;
            }
            Ok(())
        }
        Expr::Lambda { body, .. } => referenced_in_expr(catalog, arena, *body, out, d),
        _ => each_child(arena, id, &mut |c| referenced_in_expr(catalog, arena, c, out, d)),
    }
}

/// Collects every table the SQL references. Used to know what schemas to resolve.
pub fn referenced_tables(
    catalog: &Catalog,
    arena: &ExprArena,
    from: &FromItem,
    out: &mut Vec<usize>,
) -> Result<()> {
    referenced_tables_at(catalog, arena, from, out, 0)
}

fn referenced_tables_at(
    catalog: &Catalog,
    arena: &ExprArena,
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
        FromItem::Table { name, .. } => {
            if let Some(i) = catalog.index_of(name) {
                push(i);
                return Ok(());
            }
            // In-memory tables need no byte-range resolution. Views do: the file
            // tables they select from must be collected or `push_table_rel` hits
            // Internal on an unresolved schema.
            #[cfg(feature = "ddl")]
            {
                if catalog.mem_index_of(name).is_some() {
                    return Ok(());
                }
                if let Some(i) = catalog.view_index_of(name) {
                    return referenced_in_view(catalog, i, out, depth);
                }
            }
            err!(TableNotFound)
        }
        FromItem::File { path, .. } => match catalog.index_of(path) {
            Some(i) => {
                push(i);
                Ok(())
            }
            None => err!(TableNotFound),
        },
        FromItem::Join { left, right, on, .. } => {
            referenced_tables_at(catalog, arena, left, out, depth + 1)?;
            referenced_tables_at(catalog, arena, right, out, depth + 1)?;
            // `ON` can hold a subquery too (`JOIN u ON u.c = (SELECT ...)`).
            match on {
                Some(e) => referenced_in_expr(catalog, arena, *e, out, depth + 1),
                None => Ok(()),
            }
        }
        FromItem::Subquery { query, .. } => {
            referenced_in_query(catalog, arena, query, out, depth + 1)
        }
        // The tables that `expr`'s column references point at are always held by a left sibling
        // as a separate `FromItem` (the implicit-LATERAL constraint; see the
        // `FromItem::Unnest` docs), so no newly resolvable table comes from there. It may
        // still contain a subquery, as in `UNNEST((SELECT ...))`, so the expression itself is
        // walked.
        FromItem::Unnest { expr, .. } => referenced_in_expr(catalog, arena, *expr, out, depth + 1),
        // A compute-only source that bypasses the catalog, so there is no table to resolve.
        FromItem::GenerateSeries { .. } => Ok(()),
    }
}

/// Walks a `CREATE VIEW` body so schema resolution sees the file tables it
/// selects from. Without this, `SELECT * FROM v` never calls `Table::resolve`
/// on those tables and binding fails with `Internal`.
#[cfg(feature = "ddl")]
fn referenced_in_view(catalog: &Catalog, i: usize, out: &mut Vec<usize>, depth: u32) -> Result<()> {
    ensure!(depth < MAX_FROM_DEPTH, ExpressionTooDeep);
    let parsed = crate::sql::parse(match catalog.view_get(i) {
        Some(s) => s,
        None => err!(TableNotFound),
    })?;
    let q = match &parsed.stmt {
        Stmt::Select(q) => q,
        _ => err!(Internal),
    };
    referenced_in_query(catalog, &parsed.arena, q, out, depth + 1)
}
