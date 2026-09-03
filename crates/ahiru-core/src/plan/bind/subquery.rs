//! Subquery decorrelation into joins: `EXISTS`/`IN` -> semi/anti joins
//! (`build_semijoin`), scalar-subquery correlation-key extraction
//! (`correlation_keys`), quantified `ANY`/`ALL` comparison desugaring
//! (`build_quantified_comparison`), and the WHERE-clause conjunct
//! classification used to detect correlated equality predicates
//! (`classify_conjunct`/`ConjClass`).

use super::agg::{
    as_bare_aggregate, coalesce_count_column, collect_aggregates, drop_trailing_columns,
};
use super::cte::CteScope;
use super::refs::{collect_refs, each_child_flat, push_u32};
use super::*;

/// Whether this subquery is guaranteed to produce **exactly one row**, whatever its input.
///
/// An aggregate with no `GROUP BY` does: over an empty input it still emits a single row
/// (`count(*)` = 0, `max(x)` = NULL — the `empty_input_ungrouped_emits_one_row` premise in
/// `exec::agg`). That makes `EXISTS (SELECT count(*) ...)` unconditionally true and turns
/// `x IN (SELECT count(*) ...)` into a plain comparison against that one value; rewriting
/// either as an ordinary semi-join on the correlation key would instead drop every outer row
/// whose correlated group happens to be empty.
///
/// Every clause that could remove the row (`HAVING`, `QUALIFY`, `LIMIT`/`OFFSET`) or split it
/// into several (`GROUP BY`, `GROUP BY ALL`, `GROUPING SETS`) disqualifies the shape.
fn always_one_row(arena: &ExprArena, q: &QueryStmt) -> Result<bool> {
    if !q.ctes.is_empty() || q.limit.is_some() || q.offset.is_some() {
        return Ok(false);
    }
    let sel = match &q.body {
        SetExpr::Select(s) => s,
        SetExpr::SetOp { .. } => return Ok(false),
    };
    if sel.from.is_none()
        || !sel.group_by.is_empty()
        || sel.group_by_all
        || sel.grouping_sets.is_some()
        || sel.having.is_some()
        || sel.qualify.is_some()
        || sel.limit.is_some()
        || sel.offset.is_some()
    {
        return Ok(false);
    }
    let mut aggs = Vec::new();
    for item in &sel.items {
        collect_aggregates(arena, item.expr, &mut aggs, 0)?;
    }
    Ok(!aggs.is_empty())
}

/// Rewrites `EXISTS` / `IN (SELECT ...)` into semi-joins and anti-joins.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_semijoin(
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
            // `scope` is the scope of the query containing this `EXISTS`/`IN`, so it is passed
            // straight through as "one level out" (only one level of correlation is supported.
            // Deeper correlation becomes a column reference absent from the inner `scope` and
            // naturally fails with `ColumnNotFound`).
            let plan = bind_query_in(catalog, arena, query, params, ctes, Some(scope))?;
            // An ungrouped aggregate always yields exactly one row, so `EXISTS` over it is a
            // constant: TRUE, or FALSE under `NOT EXISTS`. The subquery is still bound above
            // so that its own errors are reported, then discarded. Rewriting it as a semi-join
            // on the correlation key would wrongly drop the outer rows whose correlated group
            // is empty — precisely the rows for which `count(*)` is 0 and `EXISTS` is still true.
            if always_one_row(arena, query)? {
                return Ok(if *negated {
                    Node::Limit { input: Box::new(left), limit: Some(0), offset: 0 }
                } else {
                    left
                });
            }
            // Without correlation, all that matters is "is there at least one row on the right",
            // and no key is needed. With correlation, it checks whether at least one row matches
            // the correlation key. `EXISTS` asks only about row existence rather than about a
            // set, so it has none of the NULL three-valued-logic pitfalls of `IN`/`NOT IN` and is
            // correct with an ordinary Semi/Anti (where NULL keys do not match one another)
            // (confirmed with DuckDB).
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
        Expr::InSubquery { arg, query, negated } => build_in_style_semijoin(
            catalog, arena, params, ctes, left, scope, subs, schema, *arg, query, *negated,
        ),
        // `= ANY (SELECT ...)` / `<> ALL (SELECT ...)` mean exactly the same as `IN`/`NOT IN`
        // (see the `is_semijoin_predicate` docs). Other operator and `SOME`/`ALL` combinations
        // never reach here (`is_semijoin_predicate` filters them out).
        Expr::QuantifiedComparison { op, arg, all: _, query } => {
            let negated = matches!(op, BinaryOp::Ne);
            build_in_style_semijoin(
                catalog, arena, params, ctes, left, scope, subs, schema, *arg, query, negated,
            )
        }
        _ => err!(Internal),
    }
}

/// Builds the semi-join/anti-join for the `x [NOT] IN (SELECT ...)` shape. The shared
/// implementation called both from `InSubquery` and from a `QuantifiedComparison` rewritten
/// as `= ANY` / `<> ALL` (see `build_semijoin`).
#[allow(clippy::too_many_arguments)]
fn build_in_style_semijoin(
    catalog: &Catalog,
    arena: &ExprArena,
    params: &[Value],
    ctes: &mut CteScope,
    left: Node,
    scope: &Scope,
    subs: &[Substitution],
    schema: Vec<Field>,
    arg: ExprId,
    query: &QueryStmt,
    negated: bool,
) -> Result<Node> {
    let plan = bind_query_in(catalog, arena, query, params, ctes, Some(scope))?;
    let k = plan.correlated.len();
    ensure!(plan.root.schema().len() - k == 1, TypeMismatch);
    let rf = plan.root.schema()[0].clone();

    // An ungrouped aggregate yields exactly one row, so membership in it is a plain comparison
    // against that single value — including the NULL three-valued logic, which `=`/`<>` already
    // carry. A semi-join on the correlation key would instead drop every outer row whose
    // correlated group is empty, even though `count(*)` there is 0, not "no row".
    if always_one_row(arena, query)? {
        let is_count =
            matches!(as_bare_aggregate(arena, query), Some(AggKind::Count | AggKind::CountStar));
        return build_one_row_membership(
            arena, params, left, scope, subs, schema, arg, negated, plan, is_count,
        );
    }

    let lp = compile_with_subs(arena, scope, params, subs, arg)?;
    let rscope = Scope::from_fields(plan.root.schema().to_vec());
    let rp = column_program(&rscope, 0)?;
    // Keys are compared as encoded byte sequences, so the physical types are aligned.
    let want = Ty::unify_or_mismatch(lp.result_ty, rf.ty)?;
    let mut left_keys = vec![cast_program(lp, want)?];
    let mut right_keys = vec![cast_program(rp, want)?];
    let (corr_left, corr_right) = correlation_keys(arena, scope, params, &plan)?;
    left_keys.extend(corr_left);
    right_keys.extend(corr_right);

    // With `NOT IN`, a single NULL on the right makes the result empty under three-valued
    // logic. An ordinary anti-join cannot reproduce that, so the NULL-aware version is used
    // (if there is even one NULL key on the right, it returns empty unconditionally).
    // `AntiNullAware` is implemented as "is there a NULL anywhere on the right" (not scoped per
    // correlation key), so using it as is under correlation could empty the result for an
    // unrelated outer row merely because some other outer row's correlated side had a NULL.
    // Without an execution primitive for a NULL-aware check narrowed per correlation key, a
    // correlated `NOT IN` cannot be evaluated safely (there is also no way to determine the
    // target column's nullability accurately at bind time: a SELECT list's output columns are
    // always treated as `nullable = true`). Rejecting clearly beats returning a wrong result.
    ensure!(!(negated && k > 0), UnsupportedFeature);
    let kind = match (negated, rf.nullable) {
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

/// `x [NOT] IN (<subquery yielding exactly one row>)` as a comparison against that row's value.
///
/// The value is attached with a LEFT JOIN — keyless when uncorrelated, on the correlation key
/// when correlated (the correlated form arrives already grouped by that key, one row per key,
/// from `bind_select_in`'s aggregate decorrelation). A key with no matching inner rows is
/// absent from that grouping, so the join leaves NULL, which is the right answer for every
/// aggregate except the COUNT family — `is_count` patches those back to 0, exactly as the
/// correlated scalar-subquery path does.
///
/// `x = v` / `x <> v` then give the correct three-valued result for `IN` / `NOT IN` over a
/// one-element set, so no NULL-aware anti-join is involved and the correlated `NOT IN` that
/// `build_in_style_semijoin` has to reject is expressible here.
#[allow(clippy::too_many_arguments)]
fn build_one_row_membership(
    arena: &ExprArena,
    params: &[Value],
    left: Node,
    scope: &Scope,
    subs: &[Substitution],
    out_schema: Vec<Field>,
    arg: ExprId,
    negated: bool,
    plan: Plan,
    is_count: bool,
) -> Result<Node> {
    let k = plan.correlated.len();
    let orig_len = scope.len();

    // x, the left-hand value, materialized as one extra column.
    let x_prog = compile_with_subs(arena, scope, params, subs, arg)?;
    let x_ty = x_prog.result_ty;
    let mut exprs = Vec::with_capacity(orig_len + 1);
    for i in 0..orig_len {
        exprs.push(column_program(scope, i)?);
    }
    exprs.push(x_prog);
    let mut ext_schema = out_schema.clone();
    ext_schema.push(Field::new(String::from("__inagg_x"), x_ty, true));
    let mut node = Node::Project { input: Box::new(left), exprs, schema: ext_schema };

    // The subquery's single row, joined on the correlation key (keyless when uncorrelated).
    // `plan.root`'s column order is [value, key0, key1, ...] — the convention `Plan::correlated`
    // carries, shared with the scalar-subquery path.
    let rscope = Scope::from_fields(plan.root.schema().to_vec());
    let mut left_keys = Vec::with_capacity(k);
    let mut right_keys = Vec::with_capacity(k);
    for (i, &outer_e) in plan.correlated.iter().enumerate() {
        let lp = compile_with_subs(arena, scope, params, subs, outer_e)?;
        let rp = column_program(&rscope, 1 + i)?;
        let want = Ty::unify_or_mismatch(lp.result_ty, rp.result_ty)?;
        left_keys.push(cast_program(lp, want)?);
        right_keys.push(cast_program(rp, want)?);
    }
    let value_col = orig_len + 1;
    let mut full_schema = node.schema().to_vec();
    full_schema.extend_from_slice(plan.root.schema());
    node = Node::Join {
        left: Box::new(node),
        right: Box::new(plan.root),
        kind: JoinKind::Left,
        left_keys,
        right_keys,
        residual: None,
        schema: full_schema,
    };
    if k > 0 {
        node = drop_trailing_columns(node, k)?;
        if is_count {
            node = coalesce_count_column(node, value_col)?;
        }
    }

    // The comparison is assembled in a temporary arena over a renamed view of the current
    // schema, so that `compile` handles operator type dispatch and NULL semantics as usual.
    // Only the two synthetic names are looked up, and both are unique within that view.
    let mut cmp_fields = node.schema().to_vec();
    ensure!(cmp_fields.len() > value_col, Internal);
    cmp_fields[orig_len].name = String::from("__inagg_x");
    cmp_fields[value_col].name = String::from("__inagg_v");
    let cmp_scope = Scope::from_fields(cmp_fields);
    let mut fa = ExprArena::new();
    let x_ref = fa.push(Expr::ColumnRef { qualifier: None, name: String::from("__inagg_x") });
    let v_ref = fa.push(Expr::ColumnRef { qualifier: None, name: String::from("__inagg_v") });
    let op = if negated { BinaryOp::Ne } else { BinaryOp::Eq };
    let cmp = fa.push(Expr::Binary { op, lhs: x_ref, rhs: v_ref });
    let pred = compile_predicate(&fa, &cmp_scope, params, cmp)?;
    node = Node::Filter { input: Box::new(node), pred };

    // Drop the two intermediate columns, restoring the caller's schema.
    let s = Scope::from_fields(node.schema().to_vec());
    let mut trim = Vec::with_capacity(orig_len);
    for i in 0..orig_len {
        trim.push(column_program(&s, i)?);
    }
    Ok(Node::Project { input: Box::new(node), exprs: trim, schema: out_schema })
}

/// Builds the join-key pairs `(left, right)` corresponding to each item of `plan.correlated`.
/// The trailing `plan.correlated.len()` columns of `plan.root` are the correlation key columns
/// `bind_select_in` appended (how many columns precede them depends on the kind of subquery,
/// but they are always grouped at the end). Without correlation, both come back empty.
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

// --- Quantified comparison (ANY/ALL/SOME, the `>`/`<`/`>=`/`<=` side) --------
//
// `= ANY`/`<> ALL` rewrite to the same semi-join as `IN`/`NOT IN` (see
// `is_semijoin_predicate`/`build_semijoin`). The remaining eight combinations
// (`<`/`<=`/`>`/`>=` x `ANY`/`ALL`) ask about "a comparison against the whole set" rather than
// "is there a matching row", so they do not lower to a semi-join. Instead, following
// `duckdb`'s internal rewrite (measured: EXPLAINing `SELECT 5 > ALL(...)` yields
// `NOT (5 <= ANY(...))`), `x <op> ALL (q)` is reduced to the single `ANY` form
// `NOT (x <negate(op)> ANY (q))`.
//
// `x <op> ANY (q)` itself is not correct as a simple substitution of `MIN`/`MAX` (empty sets
// and NULL three-valued logic are involved -- the SQL standard's pitfall that `ANY` over an
// empty set is always `FALSE` while `ALL` over an empty set is always `TRUE`. While working on
// this module the following identity was confirmed by measurement with the `duckdb` CLI):
//
//   x <op> ANY (q) ==
//     CASE
//       WHEN COUNT(*) = 0        THEN FALSE  -- the empty set
//       WHEN x IS NULL           THEN NULL   -- x NULL makes every row UNKNOWN
//       WHEN x <op> extreme(q)   THEN TRUE   -- decidable from the non-NULL measured values alone
//       WHEN COUNT(col) < COUNT(*) THEN NULL -- lost against the non-NULLs, but NULL rows exist
//       ELSE FALSE
//     END
//
// Here `extreme` is `MIN(col)` when `op` is `>`/`>=` and `MAX(col)` when it is `<`/`<=` (the
// idea being that pitting x against "the easiest candidate to beat" suffices: `x > ANY(q)`
// means "greater than at least one element of q", so being greater than the minimum guarantees
// one exists). In the third branch, if `extreme` is NULL (= there are no non-NULL rows) the
// comparison itself becomes NULL and falls through naturally, so there is no need to check
// `COUNT(col) > 0` separately.
//
// **Supported scope**: uncorrelated subqueries only. When the inner query references a column
// of the outer scope it is explicitly rejected with `UnsupportedFeature` (the "reject clearly
// rather than break silently" policy at the top of DESIGN.md). Supporting correlation gets
// decisively complex (x varies per outer row, but `extreme`/`COUNT` are currently designed to
// be computed once, so generalizing to per-correlation-key aggregation would take additional
// work), so it was deferred this time. See the tests in `tests/any_all_subquery.rs` and the
// final report for details.
// `= ALL (q)`/`<> ANY (q)` are deferred for the same reason (see the
// `collect_quantified_comparisons` docs):
// that shape asks whether "the set's elements are all exactly one value", which reduces neither
// to a single `MIN`/`MAX` nor to a semi-join.

/// Picks the `(whether to use MAX, comparison operator)` used to rewrite `x <op> ANY (q)`.
/// `op` is always one of `Lt`/`Le`/`Gt`/`Ge` (guaranteed by the caller).
fn quantified_any_extreme(op: BinaryOp) -> (bool, BinaryOp) {
    match op {
        BinaryOp::Gt => (false, BinaryOp::Gt), // greater than MIN(q)?
        BinaryOp::Ge => (false, BinaryOp::Ge), // at least MIN(q)?
        BinaryOp::Lt => (true, BinaryOp::Lt),  // less than MAX(q)?
        BinaryOp::Le => (true, BinaryOp::Le),  // at most MAX(q)?
        _ => unreachable!("caller only passes order comparisons"),
    }
}

/// The logically negated comparison operator (the one where negating `x <op> y` gives
/// `x <negate(op)> y`; distinct from `BinaryOp::swapped`, which swaps the operands).
/// Used to lower `x <op> ALL (q)` into `NOT (x <negate(op)> ANY (q))`.
fn negate_comparison(op: BinaryOp) -> BinaryOp {
    match op {
        BinaryOp::Gt => BinaryOp::Le,
        BinaryOp::Le => BinaryOp::Gt,
        BinaryOp::Lt => BinaryOp::Ge,
        BinaryOp::Ge => BinaryOp::Lt,
        _ => unreachable!("caller only passes order comparisons"),
    }
}

/// A synthesized composite name (`__qc{idx}_{suffix}`). `Scope::resolve` searches every column
/// unqualified, so a prefix is added to avoid colliding with the user's column names
/// (the same trade-off as the existing `subqN` labels. Not a complete guarantee, but SQL using
/// this prefix as a real column name is essentially unheard of).
fn quantified_label(idx: usize, suffix: &str) -> String {
    let mut s = String::from("__qc");
    push_u32(&mut s, idx as u32);
    s.push('_');
    s.push_str(suffix);
    s
}

/// Rewrites one quantified comparison involving `>`/`<`/`>=`/`<=` into a single row of three
/// aggregates -- `COUNT(*)`/`COUNT(col)`/`MIN` or `MAX` -- plus a (keyless) `LEFT JOIN`
/// attaching it to every outer row. The aggregate has no `GROUP BY`, so it always returns one
/// row even when `q` is empty (the same premise as `empty_input_ungrouped_emits_one_row` in
/// `exec::agg`; COUNT becomes 0 and `MIN`/`MAX` become NULL), so the empty-set branch is also
/// decided correctly by the CASE expression from that one row's values alone.
///
/// After the join, a small `CASE` expression reading the three aggregate columns and x's value
/// (the expression at the top of the module) is assembled in a separate temporary `ExprArena`
/// (dedicated to this subquery's computation, unrelated to the outer `arena`) and run through
/// the existing `compile()` (so comparison-operator type dispatch and `CASE`/`IS NULL`
/// semantics are not reimplemented). Finally the intermediate columns (x and the three
/// aggregates) are dropped, leaving the original columns plus one result column.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_quantified_comparison(
    catalog: &Catalog,
    arena: &ExprArena,
    params: &[Value],
    ctes: &mut CteScope,
    node: Node,
    scope: &Scope,
    subs: &[Substitution],
    idx: usize,
    op: BinaryOp,
    arg: ExprId,
    all: bool,
    query: &QueryStmt,
) -> Result<Node> {
    let plan = bind_query_in(catalog, arena, query, params, ctes, Some(scope))?;
    // Correlated subqueries are out of scope (see the module docs at the top).
    ensure!(plan.correlated.is_empty(), UnsupportedFeature);
    ensure!(plan.root.schema().len() == 1, TypeMismatch);
    let col_ty = plan.root.schema()[0].ty;

    // `ALL` reduces to the `ANY` rewrite as `NOT (x <negate(op)> ANY (q))`.
    let (want_max, cmp_op, wrap_not) = if all {
        let (wm, co) = quantified_any_extreme(negate_comparison(op));
        (wm, co, true)
    } else {
        let (wm, co) = quantified_any_extreme(op);
        (wm, co, false)
    };

    let rscope = Scope::from_fields(plan.root.schema().to_vec());
    let col_prog = column_program(&rscope, 0)?;
    let aggs = vec![
        Agg {
            kind: AggKind::CountStar,
            arg: None,
            distinct: false,
            name: "cnt".into(),
            separator: Vec::new(),
            quantile: 0.5,
            arg2: None,
            filter: None,
        },
        Agg {
            kind: AggKind::Count,
            arg: Some(col_prog.clone()),
            distinct: false,
            name: "nonnull".into(),
            separator: Vec::new(),
            quantile: 0.5,
            arg2: None,
            filter: None,
        },
        Agg {
            kind: if want_max { AggKind::Max } else { AggKind::Min },
            arg: Some(col_prog),
            distinct: false,
            name: "extreme".into(),
            separator: Vec::new(),
            quantile: 0.5,
            arg2: None,
            filter: None,
        },
    ];
    let agg_schema = vec![
        Field::new(quantified_label(idx, "cnt"), Ty::BigInt, false),
        Field::new(quantified_label(idx, "nonnull"), Ty::BigInt, false),
        Field::new(quantified_label(idx, "extreme"), col_ty, true),
    ];
    let agg_node = Node::Aggregate {
        input: Box::new(plan.root),
        groups: Vec::new(),
        aggs,
        schema: agg_schema,
        having: None,
    };

    // Computes x's value and appends one column after all the existing ones.
    let cur_scope = Scope::from_fields(node.schema().to_vec());
    let orig_len = cur_scope.len();
    let x_prog = compile_with_subs(arena, scope, params, subs, arg)?;
    let x_ty = x_prog.result_ty;
    let x_name = quantified_label(idx, "x");
    let mut exprs = Vec::with_capacity(orig_len + 1);
    for i in 0..orig_len {
        exprs.push(column_program(&cur_scope, i)?);
    }
    exprs.push(x_prog);
    let mut ext_schema = cur_scope.fields().to_vec();
    ext_schema.push(Field::new(x_name, x_ty, true));
    let node = Node::Project { input: Box::new(node), exprs, schema: ext_schema };

    // The one aggregate row is attached to every row with a keyless LEFT JOIN (= a cross join).
    let mut full_schema = node.schema().to_vec();
    full_schema.extend_from_slice(agg_node.schema());
    let node = Node::Join {
        left: Box::new(node),
        right: Box::new(agg_node),
        kind: JoinKind::Left,
        left_keys: Vec::new(),
        right_keys: Vec::new(),
        residual: None,
        schema: full_schema,
    };

    // The CASE expression is assembled and compiled over a temporary scope covering the whole
    // post-join schema. The last four columns are [x, cnt, nonnull, extreme].
    let combined_scope = Scope::from_fields(node.schema().to_vec());
    let len = combined_scope.len();
    ensure!(len >= 4, Internal);
    let (x_i, cnt_i, nonnull_i, extreme_i) = (len - 4, len - 3, len - 2, len - 1);

    let mut fa = ExprArena::new();
    let col_ref = |fa: &mut ExprArena, i: usize| {
        fa.push(Expr::ColumnRef { qualifier: None, name: combined_scope.fields()[i].name.clone() })
    };
    let x_ref = col_ref(&mut fa, x_i);
    let cnt_ref = col_ref(&mut fa, cnt_i);
    let nonnull_ref = col_ref(&mut fa, nonnull_i);
    let extreme_ref = col_ref(&mut fa, extreme_i);
    let zero = fa.push(Expr::Literal(Value::I64(0)));
    let cnt_eq0 = fa.push(Expr::Binary { op: BinaryOp::Eq, lhs: cnt_ref, rhs: zero });
    let x_isnull = fa.push(Expr::IsNull { arg: x_ref, negated: false });
    let cmp = fa.push(Expr::Binary { op: cmp_op, lhs: x_ref, rhs: extreme_ref });
    let nonnull_lt_cnt = fa.push(Expr::Binary { op: BinaryOp::Lt, lhs: nonnull_ref, rhs: cnt_ref });
    let v_false_empty = fa.push(Expr::Literal(Value::Bool(false)));
    let v_null_x = fa.push(Expr::Literal(Value::Null));
    let v_true = fa.push(Expr::Literal(Value::Bool(true)));
    let v_null_has_null = fa.push(Expr::Literal(Value::Null));
    let v_false_else = fa.push(Expr::Literal(Value::Bool(false)));
    let case_id = fa.push(Expr::Case {
        operand: None,
        whens: vec![
            (cnt_eq0, v_false_empty),
            (x_isnull, v_null_x),
            (cmp, v_true),
            (nonnull_lt_cnt, v_null_has_null),
        ],
        else_: Some(v_false_else),
    });
    let result_id =
        if wrap_not { fa.push(Expr::Unary { op: UnaryOp::Not, arg: case_id }) } else { case_id };
    let formula = compile(&fa, &combined_scope, params, result_id)?;

    // The intermediate columns (x, cnt, nonnull, extreme) are dropped, leaving the original columns plus one result column.
    let mut exprs2 = Vec::with_capacity(orig_len + 1);
    for i in 0..orig_len {
        exprs2.push(column_program(&combined_scope, i)?);
    }
    exprs2.push(formula);
    let mut out_schema = combined_scope.fields()[..orig_len].to_vec();
    out_schema.push(Field::new(quantified_label(idx, "result"), Ty::Boolean, true));
    Ok(Node::Project { input: Box::new(node), exprs: exprs2, schema: out_schema })
}

// --- Decomposing predicates --------------------------------------------------

/// Splits a predicate joined by AND into its parts.
pub(super) fn split_conjuncts(
    arena: &ExprArena,
    id: ExprId,
    out: &mut Vec<ExprId>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    // `a AND b AND ... AND z` is a left-deep chain: recursing into `lhs` would cost one stack
    // frame and one unit of the nesting budget per conjunct, so a flat 100-term WHERE would
    // be rejected as "expression nesting too deep". Descend the spine in a loop instead and
    // recurse only into the right operands, preserving the left-to-right output order.
    let mut rights: Vec<ExprId> = Vec::new();
    let mut cur = id;
    while let Expr::Binary { op: BinaryOp::And, lhs, rhs } = arena.get(cur) {
        rights.push(*rhs);
        cur = *lhs;
    }
    if rights.is_empty() {
        out.push(id);
        return Ok(());
    }
    let d = depth + 1;
    split_conjuncts(arena, cur, out, d)?;
    for &r in rights.iter().rev() {
        split_conjuncts(arena, r, out, d)?;
    }
    Ok(())
}

pub(super) fn and_all(
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

/// Whether an expression contains a subquery. Used to decide whether pushdown is allowed.
pub(super) fn contains_subquery(arena: &ExprArena, id: ExprId, depth: u32) -> bool {
    if depth >= MAX_EXPR_DEPTH {
        return true;
    }
    if matches!(
        arena.get(id),
        Expr::ScalarSubquery(_)
            | Expr::Exists { .. }
            | Expr::InSubquery { .. }
            | Expr::QuantifiedComparison { .. }
    ) {
        return true;
    }
    let mut found = false;
    let _ = each_child_flat(arena, id, &mut |c| {
        if contains_subquery(arena, c, depth + 1) {
            found = true;
        }
        Ok(())
    });
    found
}

/// Whether the predicate itself is an `EXISTS` / `IN (SELECT)` (a shape rewritable to a semi-join).
/// `= ANY (SELECT ...)` / `<> ALL (SELECT ...)` mean exactly the same as `IN`/`NOT IN`, so they
/// are treated as the same semi-join here (see `build_semijoin`).
/// Quantified comparisons involving `>`/`<`/`>=`/`<=` (every other `QuantifiedComparison`)
/// cannot be rewritten as a semi-join (being a comparison against the whole set, they do not
/// lower to the semi-join shape that only asks "is there a matching row"). They are handled by
/// `collect_quantified_comparisons` on the `bind_select_in` side, via a separate path.
pub(super) fn is_semijoin_predicate(arena: &ExprArena, id: ExprId) -> bool {
    match arena.get(id) {
        Expr::Exists { .. } | Expr::InSubquery { .. } => true,
        Expr::QuantifiedComparison { op, all, .. } => {
            matches!((op, all), (BinaryOp::Eq, false) | (BinaryOp::Ne, true))
        }
        _ => false,
    }
}

/// Collects the scalar subqueries inside an expression.
pub(super) fn collect_scalar_subqueries(
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
    each_child_flat(arena, id, &mut |c| collect_scalar_subqueries(arena, c, out, d))
}

/// Collects the "quantified comparisons via aggregation" inside an expression (`ANY`/`ALL` with
/// `>`/`<`/`>=`/`<=`). `= ANY`/`<> ALL` are handled separately via `is_semijoin_predicate` and
/// are not picked up here (left in, they would have `build_quantified_comparison` called with
/// an operator that cannot rely on `MIN`/`MAX`). `= ALL`/`<> ANY` (both out-of-scope
/// combinations, rewritable to neither a semi-join nor `MIN`/`MAX`) are not picked up here
/// either: they match none of `bind_select_in`'s paths, pass straight through, and finally land
/// in `plan::compile`'s net
/// (`Expr::QuantifiedComparison { .. } => err!(UnsupportedFeature)`), becoming a clear error.
pub(super) fn collect_quantified_comparisons(
    arena: &ExprArena,
    id: ExprId,
    out: &mut Vec<ExprId>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    if let Expr::QuantifiedComparison { op, .. } = arena.get(id) {
        if matches!(op, BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge)
            && !out.contains(&id)
        {
            out.push(id);
        }
    }
    // `arg` belongs to the same scope as this expression (unlike `InSubquery` it carries no
    // `query`, but another quantified comparison may be nested inside `arg`: for example
    // `(a > ANY (q1)) = (b < ANY (q2))`), so the children are always walked even after a match
    // (`each_child` passes only `arg` for a `QuantifiedComparison`).
    let d = depth + 1;
    each_child_flat(arena, id, &mut |c| collect_quantified_comparisons(arena, c, out, d))
}

/// Collects every column reference, qualified or not, as
/// `(ExprId, qualifier, name)`. Used when compiling QUALIFY after projection:
/// unqualified names that match an output column stay on the projected
/// schema; everything else is added as a hidden input column.
pub(super) fn collect_colrefs(
    arena: &ExprArena,
    id: ExprId,
    out: &mut Vec<(ExprId, Option<String>, String)>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    if let Expr::ColumnRef { qualifier, name } = arena.get(id) {
        out.push((id, qualifier.clone(), name.clone()));
        return Ok(());
    }
    let d = depth + 1;
    each_child_flat(arena, id, &mut |c| collect_colrefs(arena, c, out, d))
}

/// If this predicate references exactly one relation, returns its index.
pub(super) fn single_rel_of(
    arena: &ExprArena,
    scope: &Scope,
    ranges: &[(usize, usize)],
    id: ExprId,
) -> Result<Option<usize>> {
    let mut cols = Vec::new();
    // A predicate containing an unresolvable reference is not pushed down.
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

/// If it has the shape `left expression = right expression`, extracts it as an equi-join key.
pub(super) fn equi_key(
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
    // A side made only of constants is not a join key (it is a WHERE-style condition).
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

/// Aligns both sides of a join key to the same physical type.
///
/// Because `equi_key` compiles the two sides independently (unlike a correlated equality
/// predicate or an ordinary comparison expression, which compile as a single program such as
/// `WHERE a.k = d.k`), implicit conversion via `Ty::unify` does not happen automatically.
/// Without alignment, even logically comparable types such as a `BIGINT` column and a `DOUBLE`
/// column would have disagreeing physical representations (the raw bits of `I64` versus those
/// of `F64`), making the hash join's key comparison always mismatch and rows disappear (with
/// neither a crash nor an error, which makes the mistake hard to notice).
pub(super) fn unify_key_types(l: Program, r: Program) -> Result<(Program, Program)> {
    if l.result_ty == r.result_ty {
        return Ok((l, r));
    }
    let t = Ty::unify_or_mismatch(l.result_ty, r.result_ty)?;
    Ok((cast_program(l, t)?, cast_program(r, t)?))
}

// --- Classifying a correlated subquery's WHERE -------------------------------

/// The classification of a top-level WHERE conjunct (one piece after `split_conjuncts` splits the ANDs).
pub(super) enum ConjClass {
    /// An ordinary predicate resolvable within this query's own scope alone.
    Local,
    /// A correlated equality predicate of the form `inner expression = outer-scope expression`.
    Correlated { inner: ExprId, outer: ExprId },
}

/// Classifies one top-level WHERE conjunct.
///
/// `Local` if it resolves within the local scope alone. `Correlated` if it is an equality
/// predicate `inner = outer` involving an outer-scope column (either side may be a compound
/// expression). Referencing an outer-scope column in any other shape (a non-equality
/// comparison, inside an `OR`, wrapped in `NOT`, and so on) cannot be extracted as a join key,
/// so it returns `UnsupportedFeature`. The policy is to reject clearly rather than return an
/// inaccurate result (the policy of this project as a whole; see DESIGN.md).
pub(super) fn classify_conjunct(
    arena: &ExprArena,
    local: &Scope,
    outer: &Scope,
    id: ExprId,
) -> Result<ConjClass> {
    // First check whether it resolves locally (= an ordinary predicate with no correlation).
    if resolves_in(arena, local, id)? {
        return Ok(ConjClass::Local);
    }
    if let Expr::Binary { op: BinaryOp::Eq, lhs, rhs } = arena.get(id) {
        let (l, r) = (*lhs, *rhs);
        if resolves_in(arena, local, l)? && is_pure_outer_only(arena, local, outer, r)? {
            return Ok(ConjClass::Correlated { inner: l, outer: r });
        }
        if resolves_in(arena, local, r)? && is_pure_outer_only(arena, local, outer, l)? {
            return Ok(ConjClass::Correlated { inner: r, outer: l });
        }
    }
    // It could not be extracted as an equality correlation key. If a reference to the outer
    // scope remains anywhere, this is unsupported correlation (non-equality, inside an OR, ...).
    if references_outer(arena, local, outer, id, 0)? {
        err!(UnsupportedFeature);
    }
    // If it does not touch the outer scope, it is a predicate unrelated to correlation (which
    // failed local resolution for some other reason). Left to the ordinary path.
    Ok(ConjClass::Local)
}

/// Whether a name lookup failed merely because this scope does not offer the name, as
/// opposed to failing for a reason that must stay an error.
///
/// Only "this scope does not have it" may fall through to the outer scope. An
/// `AmbiguousColumn` in particular means the name *is* here, more than once, and treating it
/// like a missing name would silently re-bind it as a correlation key against the outer
/// query — returning a plausible wrong answer where DuckDB raises "ambiguous reference".
fn is_absent_here(e: Error) -> bool {
    matches!(e.code, Code::ColumnNotFound | Code::TableNotFound)
}

/// Whether an expression resolves entirely within `scope`.
///
/// `Ok(false)` only for names this scope does not have; any other resolution error
/// (`AmbiguousColumn`, `ExpressionTooDeep`, ...) is propagated.
fn resolves_in(arena: &ExprArena, scope: &Scope, id: ExprId) -> Result<bool> {
    let mut tmp = Vec::new();
    match collect_refs(arena, scope, id, &mut tmp) {
        Ok(()) => Ok(true),
        Err(e) if is_absent_here(e) => Ok(false),
        Err(e) => Err(e),
    }
}

/// Whether an expression "does not resolve locally but does resolve in the outer scope"
/// (= it references purely the outer scope).
fn is_pure_outer_only(arena: &ExprArena, local: &Scope, outer: &Scope, id: ExprId) -> Result<bool> {
    if resolves_in(arena, local, id)? {
        return Ok(false);
    }
    resolves_in(arena, outer, id)
}

/// Whether the expression contains, anywhere, a column reference resolvable only in the outer
/// scope. Same-named columns resolvable locally are excluded by shadowing
/// (SQL's name resolution rule: the inner scope wins over the outer).
fn references_outer(
    arena: &ExprArena,
    local: &Scope,
    outer: &Scope,
    id: ExprId,
    depth: u32,
) -> Result<bool> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    if let Expr::ColumnRef { qualifier, name } = arena.get(id) {
        match local.resolve(qualifier.as_deref(), name) {
            Ok(_) => return Ok(false),
            Err(e) if is_absent_here(e) => {}
            Err(e) => return Err(e),
        }
        return Ok(outer.resolve(qualifier.as_deref(), name).is_ok());
    }
    let mut found = false;
    let d = depth + 1;
    each_child_flat(arena, id, &mut |c| {
        if references_outer(arena, local, outer, c, d)? {
            found = true;
        }
        Ok(())
    })?;
    Ok(found)
}

/// The correlation-aware version of `collect_refs`. A column reference that does not resolve
/// locally but does resolve in the outer scope is silently ignored (treated as a correlated
/// reference). A column reference in neither is still an error as before. With `outer_scope`
/// as `None` (an ordinary uncorrelated query), the behavior is exactly `collect_refs`.
pub(super) fn collect_refs_tolerant(
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
            // Only a name this scope does not have may be re-read as a correlated
            // reference; `AmbiguousColumn` stays an error even when the outer scope
            // happens to offer the same name.
            Err(e) if !is_absent_here(e) => return Err(e),
            Err(e) => {
                if outer_scope.resolve(qualifier.as_deref(), name).is_err() {
                    return Err(e);
                }
            }
        }
        return Ok(());
    }
    let d = depth + 1;
    each_child_flat(arena, id, &mut |c| {
        collect_refs_tolerant_at(arena, scope, outer_scope, c, out, d)
    })
}
