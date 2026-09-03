use super::agg::{check_grouped, collect_aggregates};
use super::pruning::extract_pruners;
use super::refs::{collect_refs, ordinal_of};
use super::select::expand_columns;
use super::subquery::{equi_key, single_rel_of, split_conjuncts};
use super::*;
use crate::error::code_of;
use crate::sql::ast::ColumnsSpec;
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
    assert!(out.is_empty(), "a condition under an OR must not be used for pruning");
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
    assert!(
        out.is_empty(),
        "NOT BETWEEN falls outside the range, so statistics alone cannot narrow it"
    );
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
    // The first goes into `value`; the rest go into `in_values`.
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
    assert!(out.is_empty(), "with a non-literal candidate the set is not settled, so no pruning");
}

#[test]
fn negated_in_list_is_not_pruned() {
    let mut a = ExprArena::new();
    let arg = col(&mut a, "id");
    let v1 = a.push(Expr::Literal(Value::I32(1)));
    let e = a.push(Expr::InList { arg, list: vec![v1], negated: true });
    let mut out = Vec::new();
    extract_pruners(&a, e, &scope(), &mut out);
    assert!(
        out.is_empty(),
        "NOT IN can match anything outside the candidates, so statistics alone cannot narrow it"
    );
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

/// A scope shaped like a Parquet file with a scaled DECIMAL column and a DATE column,
/// for the pruner-literal normalization below.
fn typed_scope() -> Scope {
    Scope::from_fields(vec![
        Field::new("d1", Ty::Decimal { precision: 5, scale: 1 }, true),
        Field::new("dt", Ty::Date, true),
        Field::new("f", Ty::Double, true),
    ])
}

#[test]
fn decimal_pruner_holds_the_literal_scaled_to_the_column() {
    // `d1 = 150` on a DECIMAL(5,1) column: statistics and the Bloom filter both speak in
    // the stored unscaled integer, so the pruner has to carry 1500, not 150.
    let mut a = ExprArena::new();
    let l = col(&mut a, "d1");
    let r = a.push(Expr::Literal(Value::I32(150)));
    let e = a.push(Expr::Binary { op: BinaryOp::Eq, lhs: l, rhs: r });
    let mut out = Vec::new();
    extract_pruners(&a, e, &typed_scope(), &mut out);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].value, Value::I64(1500));
}

#[test]
fn decimal_in_list_pruner_scales_every_candidate() {
    let mut a = ExprArena::new();
    let arg = col(&mut a, "d1");
    let v1 = a.push(Expr::Literal(Value::I32(150)));
    let v2 = a.push(Expr::Literal(Value::I32(160)));
    let e = a.push(Expr::InList { arg, list: vec![v1, v2], negated: false });
    let mut out = Vec::new();
    extract_pruners(&a, e, &typed_scope(), &mut out);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].value, Value::I64(1500));
    assert_eq!(out[0].in_values, vec![Value::I64(1600)]);
}

#[test]
fn decimal_pruner_is_dropped_when_the_literal_has_no_exact_scaled_form() {
    // A fractional literal makes the comparison a DOUBLE one, which the column's integer
    // statistics cannot answer, so there must be no pruner rather than a wrong one.
    let mut a = ExprArena::new();
    let l = col(&mut a, "d1");
    let r = a.push(Expr::Literal(Value::F64(150.5)));
    let e = a.push(Expr::Binary { op: BinaryOp::Eq, lhs: l, rhs: r });
    let mut out = Vec::new();
    extract_pruners(&a, e, &typed_scope(), &mut out);
    assert!(out.is_empty());
}

#[test]
fn typed_date_literal_produces_a_pruner() {
    // `DATE '...'` is folded to `Expr::TypedLiteral` at parse time; matching only
    // `Expr::Literal` used to give up pruning on every date column.
    let mut a = ExprArena::new();
    let l = col(&mut a, "dt");
    let r = a.push(Expr::TypedLiteral(Value::I32(19783), Ty::Date));
    let e = a.push(Expr::Binary { op: BinaryOp::Gt, lhs: l, rhs: r });
    let mut out = Vec::new();
    extract_pruners(&a, e, &typed_scope(), &mut out);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].op, PruneOp::Gt);
    assert_eq!(out[0].value, Value::I32(19783));

    // ... and through BETWEEN, which becomes two pruners.
    let mut a = ExprArena::new();
    let arg = col(&mut a, "dt");
    let low = a.push(Expr::TypedLiteral(Value::I32(19783), Ty::Date));
    let high = a.push(Expr::TypedLiteral(Value::I32(19790), Ty::Date));
    let e = a.push(Expr::Between { arg, low, high, negated: false });
    let mut out = Vec::new();
    extract_pruners(&a, e, &typed_scope(), &mut out);
    assert_eq!(out.len(), 2);
}

#[test]
fn integer_literal_on_a_double_column_becomes_a_float_pruner() {
    // `Value::partial_cmp_same` only compares like variants, so an unconverted `I32` would
    // silently disable pruning on FLOAT/DOUBLE columns.
    let mut a = ExprArena::new();
    let l = col(&mut a, "f");
    let r = a.push(Expr::Literal(Value::I32(5)));
    let e = a.push(Expr::Binary { op: BinaryOp::Lt, lhs: l, rhs: r });
    let mut out = Vec::new();
    extract_pruners(&a, e, &typed_scope(), &mut out);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].value, Value::F64(5.0));
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
    assert_eq!(out2, vec![or], "an OR is not decomposed");
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

    // Picked up with the operands swapped even when written the other way around.
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

    // A comparison against a constant is not a join key either.
    let lit = a.push(Expr::Literal(Value::I32(1)));
    let e2 = a.push(Expr::Binary { op: BinaryOp::Eq, lhs: lk, rhs: lit });
    assert_eq!(equi_key(&a, &s, 2, e2).unwrap(), None);

    // An inequality is not an equi-join either (it becomes a residual predicate).
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

    // A condition made only of constants belongs to no relation.
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
    // Structurally identical aggregates are merged into one (not computed twice).
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
    // GROUP BY id, yet score is referenced bare.
    assert_eq!(
        code_of(check_grouped(&a, &scope(), other, &[g], &[], &[], 0)),
        Some(Code::NotGrouped)
    );
    // The grouped column itself passes.
    assert!(check_grouped(&a, &scope(), g, &[g], &[], &[], 0).is_ok());
}

#[test]
fn grouped_expression_matches_structurally() {
    // SELECT id + 1 is allowed against GROUP BY id + 1.
    let mut a = ExprArena::new();
    let c1 = col(&mut a, "id");
    let l1 = a.push(Expr::Literal(Value::I32(1)));
    let g = a.push(Expr::Binary { op: BinaryOp::Add, lhs: c1, rhs: l1 });
    let c2 = col(&mut a, "ID");
    let l2 = a.push(Expr::Literal(Value::I32(1)));
    let e = a.push(Expr::Binary { op: BinaryOp::Add, lhs: c2, rhs: l2 });
    assert!(check_grouped(&a, &scope(), e, &[g], &[], &[], 0).is_ok());
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

// --- COLUMNS(...) expansion ---------------------------------------------------
//
// `expand_columns` is the whole of `COLUMNS(...)`'s semantics; the cases below
// pin the parts that were resolved by running them against a real `duckdb`
// v1.4.4 CLI rather than guessed. The scope here is `id, score, name`.

/// Column names only (no template), for the common assertions below.
fn expanded_names(spec: &ColumnsSpec) -> Vec<String> {
    let s = scope();
    expand_columns(spec, &s, None)
        .unwrap()
        .into_iter()
        .map(|(i, _)| s.fields()[i].name.clone())
        .collect()
}

#[test]
fn columns_all_expands_like_a_plain_star() {
    assert_eq!(expanded_names(&ColumnsSpec::All), vec!["id", "score", "name"]);
}

/// The regex is an unanchored search, not a full match: `COLUMNS('am')`
/// matches `name` (`duckdb`: `COLUMNS('um')` matches a `num` column).
#[test]
fn columns_regex_is_an_unanchored_search() {
    assert_eq!(expanded_names(&ColumnsSpec::Regex("am".into())), vec!["name"]);
    assert_eq!(expanded_names(&ColumnsSpec::Regex("^n".into())), vec!["name"]);
    assert_eq!(expanded_names(&ColumnsSpec::Regex("e$".into())), vec!["score", "name"]);
}

/// Unlike every other column-name comparison in this engine, the regex match
/// is case-sensitive (`duckdb`: `COLUMNS('N.*')` matches nothing on a `name`
/// column).
#[test]
fn columns_regex_is_case_sensitive() {
    assert_eq!(
        code_of(expand_columns(&ColumnsSpec::Regex("N.*".into()), &scope(), None)),
        Some(Code::ColumnNotFound)
    );
}

/// A regex matching no column is an error, not an empty expansion
/// (`duckdb`: "No matching columns found that match regex").
#[test]
fn columns_regex_matching_nothing_is_an_error() {
    assert_eq!(
        code_of(expand_columns(&ColumnsSpec::Regex("zzz".into()), &scope(), None)),
        Some(Code::ColumnNotFound)
    );
}

/// The explicit list expands in *schema* order, matches names
/// case-insensitively, and emits each column once even if listed twice
/// (all three verified against `duckdb`).
#[test]
fn columns_list_follows_schema_order_and_dedups() {
    assert_eq!(
        expanded_names(&ColumnsSpec::Names(vec!["name".into(), "id".into()])),
        vec!["id", "name"]
    );
    assert_eq!(expanded_names(&ColumnsSpec::Names(vec!["ID".into()])), vec!["id"]);
    assert_eq!(expanded_names(&ColumnsSpec::Names(vec!["id".into(), "id".into()])), vec!["id"]);
}

/// A listed name that no column has is an error — `COLUMNS` sides with
/// `EXCLUDE`, not with `RENAME`, on the asymmetry documented on
/// `Expr::Star::rename` (`duckdb`: "Column "nope" was selected but was not
/// found in the FROM clause").
#[test]
fn columns_list_with_an_unknown_name_is_an_error() {
    assert_eq!(
        code_of(expand_columns(
            &ColumnsSpec::Names(vec!["id".into(), "nope".into()]),
            &scope(),
            None
        )),
        Some(Code::ColumnNotFound)
    );
}

/// `AS '<template>'`: `\0` is the whole column name and `\1`.. are the
/// pattern's capture groups. An expansion that comes out empty leaves the
/// original name alone.
#[test]
fn columns_alias_template_substitutes_capture_groups() {
    let s = scope();
    let names = |spec: &ColumnsSpec, t: &str| -> Vec<String> {
        expand_columns(spec, &s, Some(t))
            .unwrap()
            .into_iter()
            .map(|(i, n)| n.unwrap_or_else(|| s.fields()[i].name.clone()))
            .collect()
    };
    assert_eq!(names(&ColumnsSpec::All, "x_\\0"), vec!["x_id", "x_score", "x_name"]);
    assert_eq!(names(&ColumnsSpec::Regex("(..).*".into()), "\\1"), vec!["id", "sc", "na"]);
    // A group the pattern doesn't have expands to the empty string, and a
    // fully empty result falls back to the original name.
    assert_eq!(names(&ColumnsSpec::Regex("^n".into()), "q\\1"), vec!["q"]);
    assert_eq!(names(&ColumnsSpec::Regex("^n".into()), "\\1"), vec!["name"]);
}
