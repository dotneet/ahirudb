//! CTE (`WITH` / `WITH RECURSIVE`) binding: the `CteScope` registry, per-CTE
//! binding, and recursive-CTE anchor/recursive-term splitting.

use super::select::bind_select_in;
use super::*;

// --- CTE ---------------------------------------------------------------------

/// A table defined by `WITH`.
///
/// **An ordinary CTE can be referenced only once.** The plan tree is held by ownership and
/// `Node` cannot be cloned, so a second reference is an error. Referencing it several times
/// requires writing two derived tables. Refusing explicitly beats breaking silently.
///
/// The sole exception is a self-reference inside a recursive CTE's recursive term
/// (`CtePlan::Recursive` / `ResolvedCte::WorkingTable`): that is a light reference carrying
/// only a schema and owning no real data, so it may be referenced any number of times, as in a self-join.
#[derive(Default)]
pub struct CteScope {
    entries: Vec<CteEntry>,
    /// The `CREATE VIEW` expansion nesting count (guarding against infinite recursion when a
    /// view references other views). The same `CteScope` instance is reused across subqueries
    /// and CTEs, so keeping it here carries the depth correctly everywhere.
    // `pub(super)`: read/updated directly by `from::push_view_rel`, which lives
    // in a sibling submodule (view expansion is part of FROM-clause handling,
    // not CTE handling proper).
    #[cfg(feature = "ddl")]
    pub(super) view_depth: u32,
    /// Query start time (UTC microseconds) for `now()` / `CURRENT_DATE` substitution
    /// when a view body is reparsed. Copied onto nested view scopes.
    #[cfg(feature = "ddl")]
    pub(super) now_micros: i64,
}

/// The maximum view expansion nesting. It is smaller than `MAX_FROM_DEPTH` because expanding
/// one view level amounts to a whole `bind_query_in` call (a full recursive-descent stack),
/// consuming more stack per level than a plain nested FROM.
///
/// `pub(super)`: consulted by `from::push_view_rel` alongside `view_depth`.
#[cfg(feature = "ddl")]
pub(super) const MAX_VIEW_DEPTH: u32 = 16;

/// The substance of a CTE.
enum CtePlan {
    /// Bound and not yet taken.
    Ready(Box<Node>),
    /// In the middle of binding a recursive CTE's own recursive term, where the self-reference
    /// has no real plan yet. The referencing side receives a `Node::WorkingTable` (a leaf
    /// swapped at runtime for the previous iteration's new rows; see `exec::recursive`).
    Recursive,
    /// Already taken.
    Taken,
}

struct CteEntry {
    name: String,
    plan: CtePlan,
    schema: Vec<Field>,
}

/// The result of `FromItem::Table` resolving a CTE name.
///
/// `pub(super)`: matched by `from::flatten_from`, a sibling submodule.
pub(super) enum ResolvedCte {
    /// An ordinary CTE. Can be taken exactly once.
    Plan(Box<Node>),
    /// A self-reference from within a recursive CTE's body. It carries no real data; the
    /// `RecursiveCte` operator feeds it at runtime.
    WorkingTable,
}

impl CteScope {
    // `pub(super)`: called from `from::flatten_from`, a sibling submodule,
    // to resolve `FromItem::Table` names against in-scope CTEs.
    pub(super) fn find(&self, name: &str) -> Option<usize> {
        self.entries.iter().position(|e| eq_ascii_ci(e.name.as_bytes(), name.as_bytes()))
    }

    /// Resolves the index `find` produced. The schema is always returned as a clone
    /// (`ResolvedCte::WorkingTable` may be referenced any number of times, so the
    /// "take only once" constraint is lifted at the cost of a copy).
    pub(super) fn resolve(&mut self, i: usize) -> Result<(ResolvedCte, Vec<Field>)> {
        let e = &mut self.entries[i];
        let schema = e.schema.clone();
        match core::mem::replace(&mut e.plan, CtePlan::Taken) {
            CtePlan::Ready(node) => Ok((ResolvedCte::Plan(node), schema)),
            CtePlan::Recursive => {
                // Not taken, so it is put back. It can be referenced any number of times.
                e.plan = CtePlan::Recursive;
                Ok((ResolvedCte::WorkingTable, schema))
            }
            CtePlan::Taken => err!(UnsupportedFeature),
        }
    }
}

/// Binds one CTE and pushes it onto `ctes`.
///
/// Even with `c.recursive` set (targeted by `WITH RECURSIVE`), a body that does not actually
/// reference itself is treated as an ordinary CTE (per standard SQL, non-recursive CTEs may
/// be mixed into a `WITH RECURSIVE` list).
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
            // The anchor (the left side) contains no self-reference and binds normally.
            let (anchor, _) = bind_set_expr(catalog, arena, anchor_expr, params, ctes, None)?;
            let out_schema = apply_cte_columns(anchor.schema().to_vec(), &c.columns)?;

            // The resolution target of the self-reference (the working table) is registered
            // first. There is no real plan yet, so it stays `CtePlan::Recursive` and is swapped
            // for the finished `Node::RecursiveCte` once binding completes.
            ctes.entries.push(CteEntry {
                name: c.name.clone(),
                plan: CtePlan::Recursive,
                schema: out_schema.clone(),
            });
            let recursive_plan = bind_select_in(catalog, arena, recursive_sel, params, ctes, None)?;
            // A self-referencing subquery cannot be correlated (no outer scope is passed in the
            // first place, so it normally cannot happen, but this makes it explicit).
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

/// Applies the explicit column names of `name(col, ...)` to the output schema. A mismatched
/// column count is an error. Types are unchanged; only the names are replaced.
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

/// Checks whether the body of a CTE with `c.recursive` set actually references itself
/// (`name`). If it does not, `None` (it may be treated as an ordinary CTE). If it does, this
/// validates that it has the shape `<anchor> UNION [ALL] <recursive_term>` and returns
/// `(anchor, whether UNION ALL, recursive term)`.
///
/// The recursive term is limited to a single `SELECT` (containing no further set operation).
/// DuckDB allows more complex shapes (nested set operations on the anchor side, several
/// self-references in the recursive term), but the execution operator
/// (`exec::recursive::RecursiveCte`) is designed to "rebuild the recursive term as one
/// physical plan every iteration", so this is narrowed to the range it handles cleanly.
fn split_recursive_cte<'a>(
    query: &'a QueryStmt,
    name: &str,
) -> Result<Option<(&'a SetExpr, bool, &'a SelectStmt)>> {
    if !set_expr_references(&query.body, name) {
        return Ok(None);
    }
    // Allowing a nested WITH in a recursive CTE's body would complicate self-reference
    // detection, and `bind_query_in` does not process a nested `q.ctes` in the first place (the
    // same existing constraint as derived tables). It is clearly refused as out of scope.
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

/// Whether a `SetExpr` contains a reference to `name` as a FROM clause (including JOINs and
/// derived tables). Expressions (scalar subqueries in WHERE/the SELECT list, `EXISTS`, `IN`)
/// are not examined -- `flatten_from`'s self-reference resolution also targets only the FROM
/// clause, so the detection scope is kept aligned with it.
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
        // `expr` is a scalar expression, and a CTE self-reference is only detected as a
        // `FromItem` inside a FROM clause (see the module docs), so it never appears here.
        FromItem::Unnest { .. } => false,
        // A compute-only source that bypasses the catalog, so it references no table name.
        FromItem::GenerateSeries { .. } => false,
    }
}
