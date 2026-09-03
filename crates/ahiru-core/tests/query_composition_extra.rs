//! Extra edge-case coverage for query composition (JOIN, subqueries, CTE,
//! set operations, window functions, GROUPING SETS/ROLLUP/CUBE).
//!
//! This file is additive to the existing suite (`correlated_subqueries.rs`,
//! `recursive_cte.rs`, `grouping_sets.rs`, `named_window.rs`, `unnest.rs`) —
//! it focuses on edge cases not already exercised there: empty join sides,
//! all-NULL join keys, mixed numeric join key types, self-joins, multi-
//! condition ON clauses, RANGE-vs-ROWS frame semantics on tied ORDER BY
//! keys, out-of-range ROWS frame bounds, duplicate/empty GROUPING SETS, and
//! set-operation column-count/type-mismatch errors.
//!
//! Expected values are cross-checked against the real `duckdb` CLI
//! (`/opt/homebrew/bin/duckdb`), following the same convention as
//! `crates/ahiru-cli/tests/sql_e2e.rs` and the sibling test files in this
//! directory. Each test documents the equivalent `duckdb -c "..."` query it
//! was checked against.
//!
//! Tables are registered as in-memory CSV byte sources (`FormatKind::Csv`)
//! rather than DDL/DML in-memory tables, so this whole file runs under the
//! default feature set (`cargo test --workspace`, no `--features dml`
//! needed) — matching the pattern in `recursive_cte.rs`/`unnest.rs`.

use ahiru_core::error::{code_of, Code};
use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

/// Runs `sql` to completion and collects all rows as `Vec<Vec<Value>>`.
/// Every fixture in this file lives entirely in memory (registered via
/// `register_bytes_as`), so `NeedIo`/`NeedCodec` must never occur.
fn run(session: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    let mut q = match session.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
    };
    let mut rows = Vec::new();
    loop {
        match session.step(&mut q).unwrap() {
            QueryStep::Batch(mut b) => {
                b.materialize();
                for r in 0..b.num_rows() {
                    rows.push(b.cols.iter().map(|c| c.value_at(r)).collect());
                }
            }
            QueryStep::Done => break,
            QueryStep::NeedIo(_) | QueryStep::NeedCodec(_) => {
                panic!("{sql}: unexpected NeedIo/NeedCodec")
            }
        }
    }
    rows
}

fn i64v(v: i64) -> Value {
    Value::I64(v)
}
fn i128v(v: i128) -> Value {
    Value::I128(v)
}
fn f64v(v: f64) -> Value {
    Value::F64(v)
}
fn s(v: &str) -> Value {
    Value::Bytes(v.as_bytes().to_vec())
}
const NULL: Value = Value::Null;

fn csv(session: &mut Session, name: &str, content: &str) {
    session.register_bytes_as(name, content.as_bytes().to_vec(), FormatKind::Csv).unwrap();
}

/// `a`/`b`: two tables sharing an integer key column, deliberately built to
/// exercise duplicate keys on both sides (1:many and many:many), a NULL key
/// on both sides, and an unmatched key on each side.
///
/// ```text
/// a: (1,a1) (2,a2) (2,a2b) (NULL,aN) (4,a4)
/// b: (1,b1) (2,b2) (2,b2b) (NULL,bN) (3,b3) (2,b2c)
/// ```
/// Also registers `d` (same key domain but `DOUBLE`-typed) for the mixed
/// numeric-key-type join test.
fn session_with_ab() -> Session {
    let mut sess = Session::new();
    csv(&mut sess, "a", "k,v\n1,a1\n2,a2\n2,a2b\n,aN\n4,a4\n");
    csv(&mut sess, "b", "k,w\n1,b1\n2,b2\n2,b2b\n,bN\n3,b3\n2,b2c\n");
    csv(&mut sess, "d", "k,w\n1.0,d1\n2.0,d2\n2.5,d2b\n");
    sess
}

// =========================================================================
// JOIN: empty sides, NULL keys, mixed numeric key types, self-join,
// multi-condition ON.
// =========================================================================

/// Baseline: INNER JOIN with duplicate keys on both sides produces the full
/// cross-product of matching rows (many-to-many). NULL keys never match.
/// duckdb: `SELECT a.k, a.v, b.w FROM a JOIN b ON a.k = b.k ORDER BY a.k, a.v, b.w`
#[test]
fn inner_join_many_to_many_excludes_null_keys() {
    let mut db = session_with_ab();
    let rows =
        run(&mut db, "SELECT a.k, a.v, b.w FROM a JOIN b ON a.k = b.k ORDER BY a.k, a.v, b.w");
    assert_eq!(
        rows,
        vec![
            vec![i64v(1), s("a1"), s("b1")],
            vec![i64v(2), s("a2"), s("b2")],
            vec![i64v(2), s("a2"), s("b2b")],
            vec![i64v(2), s("a2"), s("b2c")],
            vec![i64v(2), s("a2b"), s("b2")],
            vec![i64v(2), s("a2b"), s("b2b")],
            vec![i64v(2), s("a2b"), s("b2c")],
        ]
    );
}

/// LEFT JOIN keeps the NULL-keyed left row and the unmatched-key left row,
/// both NULL-extended on the right side.
/// duckdb: `SELECT a.k, a.v, b.w FROM a LEFT JOIN b ON a.k = b.k
///          ORDER BY a.k NULLS FIRST, a.v, b.w NULLS FIRST`
#[test]
fn left_join_keeps_unmatched_and_null_key_rows() {
    let mut db = session_with_ab();
    let rows = run(
        &mut db,
        "SELECT a.k, a.v, b.w FROM a LEFT JOIN b ON a.k = b.k \
         ORDER BY a.k NULLS FIRST, a.v, b.w NULLS FIRST",
    );
    assert_eq!(
        rows,
        vec![
            vec![NULL, s("aN"), NULL],
            vec![i64v(1), s("a1"), s("b1")],
            vec![i64v(2), s("a2"), s("b2")],
            vec![i64v(2), s("a2"), s("b2b")],
            vec![i64v(2), s("a2"), s("b2c")],
            vec![i64v(2), s("a2b"), s("b2")],
            vec![i64v(2), s("a2b"), s("b2b")],
            vec![i64v(2), s("a2b"), s("b2c")],
            vec![i64v(4), s("a4"), NULL],
        ]
    );
}

/// FULL JOIN: unmatched rows from both sides are NULL-extended, plus the
/// matched rows in the middle. Both a's NULL key and b's NULL key are kept
/// as separate unmatched rows (a NULL key never matches b's NULL key).
/// duckdb: `SELECT a.k, a.v, b.w FROM a FULL JOIN b ON a.k = b.k
///          ORDER BY a.k NULLS FIRST, a.v NULLS FIRST, b.w NULLS FIRST`
#[test]
fn full_join_null_keys_on_both_sides_never_match_each_other() {
    let mut db = session_with_ab();
    let rows = run(
        &mut db,
        "SELECT a.k, a.v, b.w FROM a FULL JOIN b ON a.k = b.k \
         ORDER BY a.k NULLS FIRST, a.v NULLS FIRST, b.w NULLS FIRST",
    );
    assert_eq!(
        rows,
        vec![
            vec![NULL, NULL, s("b3")],
            vec![NULL, NULL, s("bN")],
            vec![NULL, s("aN"), NULL],
            vec![i64v(1), s("a1"), s("b1")],
            vec![i64v(2), s("a2"), s("b2")],
            vec![i64v(2), s("a2"), s("b2b")],
            vec![i64v(2), s("a2"), s("b2c")],
            vec![i64v(2), s("a2b"), s("b2")],
            vec![i64v(2), s("a2b"), s("b2b")],
            vec![i64v(2), s("a2b"), s("b2c")],
            vec![i64v(4), s("a4"), NULL],
        ]
    );
}

/// INNER JOIN where the right side is empty (a derived table filtered down
/// to zero rows, preserving its declared column types) yields zero rows,
/// not an error.
/// duckdb: `SELECT a.k, a.v FROM a JOIN (SELECT k, v FROM a WHERE 1=0) e ON a.k = e.k`
#[test]
fn inner_join_with_empty_right_side_yields_no_rows() {
    let mut db = session_with_ab();
    let rows =
        run(&mut db, "SELECT a.k, a.v FROM a JOIN (SELECT k, v FROM a WHERE 1=0) e ON a.k = e.k");
    assert!(rows.is_empty());
}

/// LEFT JOIN where the right side is empty must NULL-extend every left row
/// (regression for the "build side has zero rows" path specifically, as
/// distinct from "build side has some rows but none match").
/// duckdb: `SELECT a.k, a.v FROM a LEFT JOIN (SELECT k, v FROM a WHERE 1=0) e
///          ON a.k = e.k ORDER BY a.k NULLS FIRST, a.v`
#[test]
fn left_join_with_empty_right_side_null_extends_every_left_row() {
    let mut db = session_with_ab();
    let rows = run(
        &mut db,
        "SELECT a.k, a.v FROM a LEFT JOIN (SELECT k, v FROM a WHERE 1=0) e \
         ON a.k = e.k ORDER BY a.k NULLS FIRST, a.v",
    );
    assert_eq!(
        rows,
        vec![
            vec![NULL, s("aN")],
            vec![i64v(1), s("a1")],
            vec![i64v(2), s("a2")],
            vec![i64v(2), s("a2b")],
            vec![i64v(4), s("a4")],
        ]
    );
}

/// FULL JOIN where the *left* side is empty degenerates to "every right row
/// NULL-extended on the left" (the mirror image of the previous case, and a
/// check that `DrainingUnmatched` still runs when the probe side never
/// produces a single batch).
/// duckdb: `SELECT e.k, e.v, b.w FROM (SELECT k, v FROM a WHERE 1=0) e
///          FULL JOIN b ON e.k = b.k ORDER BY b.k`
#[test]
fn full_join_with_empty_left_side_null_extends_every_right_row() {
    let mut db = session_with_ab();
    let rows = run(
        &mut db,
        "SELECT e.k, e.v, b.w FROM (SELECT k, v FROM a WHERE 1=0) e \
         FULL JOIN b ON e.k = b.k ORDER BY b.k",
    );
    assert_eq!(
        rows,
        vec![
            vec![NULL, NULL, s("b1")],
            vec![NULL, NULL, s("b2")],
            vec![NULL, NULL, s("b2b")],
            vec![NULL, NULL, s("b2c")],
            vec![NULL, NULL, s("b3")],
            vec![NULL, NULL, s("bN")],
        ]
    );
}

/// A join predicate between two differently-typed numeric columns (`BIGINT`
/// key vs `DOUBLE` key) should unify to a common numeric type and compare by
/// value, matching plain `WHERE`-clause equality between the same two
/// columns.
///
/// Previously, `SELECT ... FROM a JOIN d ON a.k = d.k` incorrectly returned
/// zero rows instead of 3: `plan::bind`'s explicit-`ON` equi-key extraction
/// (`build_tree`'s `Node::Join` arm, `crates/ahiru-core/src/plan/bind.rs`)
/// compiled each side of the key separately and never unified their types,
/// unlike every other equi-join-key-building call site in the same file
/// (scalar/`EXISTS`/`IN` subquery decorrelation). `HashJoin` then
/// hash-encoded the raw `I64` bytes on one side and the raw `F64` bytes on
/// the other for what is logically the same key value, so no bucket ever
/// matched. Fixed by unifying and casting both sides via the same
/// `Ty::unify` + `cast_program` idiom already used elsewhere in this file
/// (see `unify_key_types`).
/// duckdb: `SELECT a.k, a.v, d.k, d.w FROM a JOIN d ON a.k = d.k ORDER BY a.k, a.v`
#[test]
fn join_on_mixed_numeric_key_types_compares_by_value() {
    let mut db = session_with_ab();
    let want = vec![
        vec![i64v(1), s("a1"), f64v(1.0), s("d1")],
        vec![i64v(2), s("a2"), f64v(2.0), s("d2")],
        vec![i64v(2), s("a2b"), f64v(2.0), s("d2")],
    ];
    // The comma-join / WHERE-clause form compares by value correctly.
    let via_where =
        run(&mut db, "SELECT a.k, a.v, d.k, d.w FROM a, d WHERE a.k = d.k ORDER BY a.k, a.v");
    assert_eq!(via_where, want);
    // The equivalent explicit `ON`-clause form must produce the same rows.
    let via_on =
        run(&mut db, "SELECT a.k, a.v, d.k, d.w FROM a JOIN d ON a.k = d.k ORDER BY a.k, a.v");
    assert_eq!(via_on, want);
}

/// Self-join: `a` joined to itself on the key, keeping only pairs where the
/// two `v` values differ, surfaces the one within-key duplicate pair
/// (`a2`/`a2b`, both under `k=2`).
/// duckdb: `SELECT a1.k, a1.v, a2.v FROM a a1 JOIN a a2 ON a1.k = a2.k
///          WHERE a1.v < a2.v ORDER BY a1.k, a1.v, a2.v`
#[test]
fn self_join_finds_duplicate_keys_within_the_same_table() {
    let mut db = session_with_ab();
    let rows = run(
        &mut db,
        "SELECT a1.k, a1.v, a2.v FROM a a1 JOIN a a2 ON a1.k = a2.k \
         WHERE a1.v < a2.v ORDER BY a1.k, a1.v, a2.v",
    );
    assert_eq!(rows, vec![vec![i64v(2), s("a2"), s("a2b")]]);
}

/// Multiple ON conditions (`AND`-connected equality + non-equality) are
/// both applied as part of the same join, not just the leading equality.
/// duckdb: `SELECT a.k, a.v, b.w FROM a JOIN b ON a.k = b.k AND a.v < b.w
///          ORDER BY a.k, a.v, b.w`
#[test]
fn join_with_equality_and_non_equality_on_conditions() {
    let mut db = session_with_ab();
    let rows = run(
        &mut db,
        "SELECT a.k, a.v, b.w FROM a JOIN b ON a.k = b.k AND a.v < b.w \
         ORDER BY a.k, a.v, b.w",
    );
    // Every `a.v < b.w` pair happens to hold lexically ('a...' < 'b...'), so
    // this must match the plain equi-join result exactly (regression: the
    // residual predicate must not accidentally drop or duplicate rows).
    assert_eq!(
        rows,
        vec![
            vec![i64v(1), s("a1"), s("b1")],
            vec![i64v(2), s("a2"), s("b2")],
            vec![i64v(2), s("a2"), s("b2b")],
            vec![i64v(2), s("a2"), s("b2c")],
            vec![i64v(2), s("a2b"), s("b2")],
            vec![i64v(2), s("a2b"), s("b2b")],
            vec![i64v(2), s("a2b"), s("b2c")],
        ]
    );
}

/// `CROSS JOIN` row count is the plain product of both sides' row counts,
/// independent of key values (5 rows in `a` x 6 rows in `b`).
/// duckdb: `SELECT count(*) FROM a CROSS JOIN b`
#[test]
fn cross_join_row_count_is_the_full_product() {
    let mut db = session_with_ab();
    let rows = run(&mut db, "SELECT count(*) FROM a CROSS JOIN b");
    assert_eq!(rows, vec![vec![i64v(30)]]);
}

/// Non-equi JOIN (`<`, no equality condition at all) falls back to nested
/// loop and must still respect NULL: a NULL key never satisfies `<` either.
/// duckdb: `SELECT a.k, b.k FROM a JOIN b ON a.k < b.k ORDER BY a.k, b.k`
#[test]
fn non_equi_join_excludes_null_keys_from_either_side() {
    let mut db = session_with_ab();
    let rows = run(&mut db, "SELECT a.k, b.k FROM a JOIN b ON a.k < b.k ORDER BY a.k, b.k");
    assert_eq!(
        rows,
        vec![
            vec![i64v(1), i64v(2)],
            vec![i64v(1), i64v(2)],
            vec![i64v(1), i64v(2)],
            vec![i64v(1), i64v(3)],
            vec![i64v(2), i64v(3)],
            vec![i64v(2), i64v(3)],
        ]
    );
}

// =========================================================================
// Subqueries: OR of two independent correlated EXISTS at the top level;
// sequential (non-recursive) CTE chaining where one CTE references another.
// =========================================================================

/// Two independently-correlated `EXISTS` subqueries combined with `OR` in
/// the outer `WHERE` (as opposed to a single `EXISTS` with `OR` *inside*
/// it, which `correlated_subqueries.rs::correlation_inside_or_is_rejected`
/// already confirms is rejected). Even though each `EXISTS` *could* in
/// principle decorrelate independently, this engine's decorrelation only
/// handles a single magic-decorrelated join per `WHERE` clause — combining
/// two of them via `OR` at the outer level is rejected too, cleanly (not a
/// panic), rather than silently mishandling one side.
/// duckdb (for reference — DuckDB *does* support this and returns 5):
/// `SELECT count(*) FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.k = a.k)
///  OR EXISTS (SELECT 1 FROM b WHERE b.w = 'b3')`
#[test]
fn or_of_two_independently_correlated_exists_subqueries_is_rejected() {
    let mut db = session_with_ab();
    let err = db.prepare(
        "SELECT count(*) FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.k = a.k) \
         OR EXISTS (SELECT 1 FROM b WHERE b.w = 'b3')",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// A later CTE referencing an earlier (non-recursive) CTE by name, then
/// joined with it, works like any other named relation. Two independent
/// CTEs joined with each other also works (not just a CTE joined with a
/// base table).
/// duckdb: `WITH c1 AS (SELECT k FROM a), c2 AS (SELECT k FROM a WHERE k > 1)
///          SELECT c1.k FROM c1 JOIN c2 ON c1.k = c2.k ORDER BY c1.k`
#[test]
fn later_cte_references_earlier_cte() {
    let mut db = session_with_ab();
    let rows = run(
        &mut db,
        "WITH c1 AS (SELECT k FROM a), c2 AS (SELECT k FROM a WHERE k > 1) \
         SELECT c1.k FROM c1 JOIN c2 ON c1.k = c2.k ORDER BY c1.k",
    );
    assert_eq!(
        rows,
        vec![vec![i64v(2)], vec![i64v(2)], vec![i64v(2)], vec![i64v(2)], vec![i64v(4)]]
    );
}

/// CTE names are case-insensitive and cannot be defined twice in one WITH
/// clause. DuckDB rejects both exact and case-variant duplicates at bind time;
/// accepting the first definition would silently hide the second one.
#[test]
fn duplicate_cte_names_are_rejected_case_insensitively() {
    let mut db = session_with_ab();
    let err =
        db.prepare("WITH c AS (SELECT k FROM a), C AS (SELECT k FROM b) SELECT * FROM c", &[]);
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

// =========================================================================
// Set operations: UNION/UNION ALL/INTERSECT/EXCEPT, NULL handling,
// column-count mismatch, incompatible-type mismatch.
// =========================================================================

/// `UNION` (implicit DISTINCT) treats NULL as equal to NULL for dedup
/// purposes, unlike ordinary `=`. Both `a.k` and `b.k` contribute one NULL;
/// the result carries exactly one NULL row, not two.
/// duckdb: `SELECT k FROM a UNION SELECT k FROM b ORDER BY k NULLS FIRST`
#[test]
fn union_distinct_treats_null_as_equal_to_null() {
    let mut db = session_with_ab();
    let rows = run(&mut db, "SELECT k FROM a UNION SELECT k FROM b ORDER BY k NULLS FIRST");
    assert_eq!(rows, vec![vec![NULL], vec![i64v(1)], vec![i64v(2)], vec![i64v(3)], vec![i64v(4)]]);
}

/// `UNION ALL` keeps every row including duplicate NULLs (no dedup at all).
/// duckdb: `SELECT k FROM a UNION ALL SELECT k FROM b ORDER BY k NULLS FIRST`
#[test]
fn union_all_keeps_all_duplicates_including_null() {
    let mut db = session_with_ab();
    let rows = run(&mut db, "SELECT k FROM a UNION ALL SELECT k FROM b ORDER BY k NULLS FIRST");
    assert_eq!(
        rows,
        vec![
            vec![NULL],
            vec![NULL],
            vec![i64v(1)],
            vec![i64v(1)],
            vec![i64v(2)],
            vec![i64v(2)],
            vec![i64v(2)],
            vec![i64v(2)],
            vec![i64v(2)],
            vec![i64v(3)],
            vec![i64v(4)],
        ]
    );
}

/// `INTERSECT` also treats NULL as equal to NULL: since both `a.k` and
/// `b.k` contain a NULL, NULL is part of the intersection.
/// duckdb: `SELECT k FROM a INTERSECT SELECT k FROM b ORDER BY k NULLS FIRST`
#[test]
fn intersect_treats_null_as_equal_to_null() {
    let mut db = session_with_ab();
    let rows = run(&mut db, "SELECT k FROM a INTERSECT SELECT k FROM b ORDER BY k NULLS FIRST");
    assert_eq!(rows, vec![vec![NULL], vec![i64v(1)], vec![i64v(2)]]);
}

/// `EXCEPT` (a minus b): `4` is the only key present in `a` but not `b`.
/// NULL and 1/2 are excluded because `b` also has a NULL and 1/2.
/// duckdb: `SELECT k FROM a EXCEPT SELECT k FROM b ORDER BY k NULLS FIRST`
#[test]
fn except_removes_rows_present_on_the_right_including_null() {
    let mut db = session_with_ab();
    let rows = run(&mut db, "SELECT k FROM a EXCEPT SELECT k FROM b ORDER BY k NULLS FIRST");
    assert_eq!(rows, vec![vec![i64v(4)]]);
}

/// A `UNION ALL` between an integer literal column and a `DOUBLE` literal
/// column unifies to the wider numeric type (`DOUBLE`) rather than erroring.
/// (`FROM dual` works around this engine's lack of support for a bare
/// `SELECT <expr>` with no `FROM` at all — see `recursive_cte.rs`'s
/// `session_with_dual` doc comment for the same workaround.)
/// duckdb: `SELECT 1 AS x UNION ALL SELECT CAST(2.5 AS DOUBLE)` -> DOUBLE.
#[test]
fn union_all_unifies_mismatched_numeric_literal_types() {
    let mut db = session_with_ab();
    csv(&mut db, "dual", "x\n1\n");
    let rows =
        run(&mut db, "SELECT 1 AS x FROM dual UNION ALL SELECT CAST(2.5 AS DOUBLE) FROM dual");
    assert_eq!(rows, vec![vec![f64v(1.0)], vec![f64v(2.5)]]);
}

/// DuckDB permits the final ORDER BY of a set operation to use an explicit
/// alias introduced by a non-first branch. The result name still comes from
/// the first branch, so the alias must resolve to its matching output ordinal.
#[test]
fn set_operation_order_by_can_use_a_later_branch_alias() {
    let mut db = session_with_ab();
    let rows = run(
        &mut db,
        "SELECT x AS first_name FROM range(3) t(x) UNION ALL \
         SELECT x + 10 AS later_name FROM range(3) t(x) \
         ORDER BY later_name",
    );
    assert_eq!(
        rows,
        vec![
            vec![i64v(0)],
            vec![i64v(1)],
            vec![i64v(2)],
            vec![i64v(10)],
            vec![i64v(11)],
            vec![i64v(12)]
        ],
    );
}

/// Set operations with a mismatched column *count* are rejected with a
/// clear error, not a panic or a silently truncated/padded result. This
/// engine's `unify_setop_schema` (`plan::bind`) uses the generic
/// `TypeMismatch` code for a column-count mismatch rather than the more
/// specific `ColumnCountMismatch` used elsewhere in the same file (e.g. for
/// `WITH RECURSIVE`'s column-list arity check, per `recursive_cte.rs`'s
/// `column_list_arity_mismatch_is_rejected`) — a minor error-code
/// inconsistency, not a functional bug: the query is still cleanly
/// rejected either way, never silently truncated/padded or panicking.
/// duckdb: `SELECT 1, 2 UNION SELECT 1` -> Binder Error (column count mismatch).
#[test]
fn union_with_mismatched_column_count_is_rejected() {
    let mut db = session_with_ab();
    let err = db.prepare("SELECT k, v FROM a UNION SELECT k FROM b", &[]);
    assert_eq!(code_of(err), Some(Code::TypeMismatch));
}

/// Set operations between two columns whose types cannot be unified at all
/// (here `BIGINT` vs a `JSON`-typed expression, which never implicitly
/// converts to/from numeric types per `Ty::unify`) are rejected with a
/// clear error rather than silently reinterpreting bytes.
#[test]
fn union_with_incompatible_column_types_is_rejected() {
    let mut db = session_with_ab();
    let err = db.prepare("SELECT k FROM a UNION SELECT CAST('[1]' AS JSON) FROM b", &[]);
    assert!(err.is_err(), "incompatible UNION types must fail cleanly, not panic or misparse");
}

// =========================================================================
// Window functions: default-frame semantics (this engine supports exactly
// two frames — see the "explicit frame specs are always rejected" test
// below — chosen automatically from whether `ORDER BY` is present:
// `RANGE UNBOUNDED PRECEDING AND CURRENT ROW` with `ORDER BY`,
// `ROWS UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` (whole partition)
// without it), RANGE peer-grouping on tied ORDER BY keys, no PARTITION BY,
// multiple functions mixed.
// =========================================================================

/// `w(id, grp, val)`: two partitions, one containing a NULL `val` in the
/// middle, used across the window tests below. `val` is written with a
/// decimal point so CSV type inference gives `DOUBLE`, not `BIGINT`
/// (`format::csv`'s numeric inference only picks `BigInt` for values that
/// look like whole numbers everywhere in the sampled rows).
///
/// ```text
/// (1,1,10.0) (2,1,20.0) (3,1,30.0) (4,2,5.0) (5,2,NULL) (6,2,15.0)
/// ```
fn session_with_w() -> Session {
    let mut sess = Session::new();
    csv(&mut sess, "w", "id,grp,val\n1,1,10.0\n2,1,20.0\n3,1,30.0\n4,2,5.0\n5,2,\n6,2,15.0\n");
    sess
}

/// This engine's `WindowFrame` (`sql::ast::WindowFrame`) has exactly two
/// variants — `RangeUnboundedPreceding` and `WholePartition` — chosen
/// automatically from whether `ORDER BY` is present; there is no support
/// for an explicit, arbitrary `ROWS`/`RANGE BETWEEN <n> PRECEDING/FOLLOWING`
/// frame. The parser deliberately rejects any explicit frame spec — bare
/// `ROWS`/`RANGE`, with or without `BETWEEN` — rather than silently
/// falling back to a default that would change the query's meaning
/// (`sql::parser::window_def_body`, and its `window_rejections` unit test).
/// This is a real, intentional gap relative to `docs/DESIGN.md`'s "Window
/// functions with `ROWS`/`RANGE` frames" claim, worth a docs correction —
/// but implementing arbitrary frame bounds would be new functionality, out
/// of scope for this test-and-fix pass. This test just pins down today's
/// behavior: rejected cleanly, not silently ignored or panicking.
#[test]
fn explicit_rows_or_range_frame_specs_are_always_rejected() {
    let mut db = session_with_w();
    for sql in [
        "SELECT sum(val) OVER (PARTITION BY grp ORDER BY id \
          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM w",
        "SELECT sum(val) OVER (PARTITION BY grp ORDER BY id \
          ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM w",
        "SELECT sum(val) OVER (PARTITION BY grp ORDER BY id \
          RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM w",
        "SELECT sum(val) OVER (PARTITION BY grp ROWS) FROM w",
    ] {
        let err = db.prepare(sql, &[]);
        assert_eq!(code_of(err), Some(Code::UnsupportedFeature), "{sql}");
    }
}

/// The implicit default frame when `ORDER BY` is present (`RANGE UNBOUNDED
/// PRECEDING AND CURRENT ROW`) is a running total per partition; a NULL
/// `val` contributes nothing to the sum but still occupies a row.
/// duckdb: `SELECT id, sum(val) OVER (PARTITION BY grp ORDER BY id) s
///          FROM w ORDER BY id` (no explicit frame — same default as DuckDB
///          uses when `ORDER BY` is present).
#[test]
fn default_frame_with_order_by_is_a_running_total() {
    let mut db = session_with_w();
    let rows = run(
        &mut db,
        "SELECT id, sum(val) OVER (PARTITION BY grp ORDER BY id) s FROM w ORDER BY id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i64v(1), f64v(10.0)],
            vec![i64v(2), f64v(30.0)],
            vec![i64v(3), f64v(60.0)],
            vec![i64v(4), f64v(5.0)],
            vec![i64v(5), f64v(5.0)],
            vec![i64v(6), f64v(20.0)],
        ]
    );
}

/// No `PARTITION BY` and no `ORDER BY` at all means the whole table is one
/// partition under the `WholePartition` default frame, so every row sees
/// the same grand total.
#[test]
fn window_without_partition_by_treats_whole_table_as_one_partition() {
    let mut db = session_with_w();
    let rows = run(&mut db, "SELECT id, sum(val) OVER () s FROM w ORDER BY id");
    for row in &rows {
        assert_eq!(row[1], f64v(80.0));
    }
    assert_eq!(rows.len(), 6);
}

/// The default `RANGE`-with-`ORDER BY` frame groups *peers* (rows with
/// equal `ORDER BY` value) into the same frame, unlike a positional `ROWS`
/// frame would. With two rows tied at `ord=1` and two more tied at `ord=2`,
/// the cumulative sum for either row in a peer group must include *all*
/// peers, not stop partway through the tie.
/// duckdb: `SELECT id, sum(val) OVER (PARTITION BY grp ORDER BY ord) rg
///          FROM tie ORDER BY id` (`tie`: (1,1,1,10) (2,1,1,20) (3,1,2,30) (4,1,2,40)).
#[test]
fn default_range_frame_groups_peers_by_order_key() {
    let mut db = Session::new();
    csv(&mut db, "tie", "id,grp,ord,val\n1,1,1,10.0\n2,1,1,20.0\n3,1,2,30.0\n4,1,2,40.0\n");
    let rows = run(
        &mut db,
        "SELECT id, sum(val) OVER (PARTITION BY grp ORDER BY ord) rg FROM tie ORDER BY id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i64v(1), f64v(30.0)],
            vec![i64v(2), f64v(30.0)],
            vec![i64v(3), f64v(100.0)],
            vec![i64v(4), f64v(100.0)],
        ]
    );
}

/// `lag`/`lead` return NULL past partition boundaries and correctly skip
/// over a NULL `val` in the middle of the partition (NULL is a legitimate
/// "previous value", not "no previous row").
#[test]
fn lag_and_lead_return_null_at_partition_edges() {
    let mut db = session_with_w();
    let rows = run(
        &mut db,
        "SELECT id, lag(val) OVER (PARTITION BY grp ORDER BY id) l, \
                lead(val) OVER (PARTITION BY grp ORDER BY id) le \
         FROM w ORDER BY id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i64v(1), NULL, f64v(20.0)],
            vec![i64v(2), f64v(10.0), f64v(30.0)],
            vec![i64v(3), f64v(20.0), NULL],
            vec![i64v(4), NULL, NULL],
            vec![i64v(5), f64v(5.0), f64v(15.0)],
            vec![i64v(6), NULL, NULL],
        ]
    );
}

/// `lag`/`lead`'s third argument is the value used outside the partition, and
/// the function's result type is the *first* argument's. The default used to
/// be pushed into the output column by physical type alone, so an INTEGER `1`
/// landed in a DECIMAL(6,2) column as 0.01 and a DATE in a TIMESTAMP column as
/// a microsecond count, while a physically incompatible default (an INTEGER
/// default for a VARCHAR column, a DOUBLE one for an INTEGER column) failed
/// with a type mismatch instead of being cast.
///
/// duckdb, on `wd(id, s, x) = (1,'a',1.5) (2,'b',2.25) (3,'c',0.5)`:
/// ```text
/// SELECT id, lag(CAST(x AS DECIMAL(6,2)), 1, 1) OVER (ORDER BY id),
///            lag(s, 1, 0) OVER (ORDER BY id),
///            lead(id, 1, 1.5) OVER (ORDER BY id),
///            lag(CAST('2020-01-05 00:00:00' AS TIMESTAMP), 1, DATE '1999-12-31')
///              OVER (ORDER BY id)
///   FROM wd ORDER BY id
/// 1|1.00|0|2|1999-12-31 00:00:00
/// 2|1.50|a|3|2020-01-05 00:00:00
/// 3|2.25|b|2|2020-01-05 00:00:00
/// ```
#[test]
fn lag_and_lead_cast_the_default_to_the_value_type() {
    let mut db = Session::new();
    csv(&mut db, "wd", "id,s,x\n1,a,1.5\n2,b,2.25\n3,c,0.5\n");
    let rows = run(
        &mut db,
        "SELECT id, lag(CAST(x AS DECIMAL(6,2)), 1, 1) OVER (ORDER BY id) d, \
                lag(s, 1, 0) OVER (ORDER BY id) sd, \
                lead(id, 1, 1.5) OVER (ORDER BY id) nd, \
                lag(CAST('2020-01-05 00:00:00' AS TIMESTAMP), 1, DATE '1999-12-31') \
                  OVER (ORDER BY id) td \
         FROM wd ORDER BY id",
    );
    // DECIMAL(6,2) is the unscaled integer; TIMESTAMP is epoch microseconds.
    let ts_1999 = 946_598_400_000_000;
    let ts_2020 = 1_578_182_400_000_000;
    assert_eq!(
        rows,
        vec![
            vec![i64v(1), i64v(100), s("0"), i64v(2), i64v(ts_1999)],
            vec![i64v(2), i64v(150), s("a"), i64v(3), i64v(ts_2020)],
            vec![i64v(3), i64v(225), s("b"), i64v(2), i64v(ts_2020)],
        ]
    );
}

/// `first_value` under the default `RANGE`-with-`ORDER BY` frame is
/// constant within a partition (the frame always starts at the partition's
/// first row), while `last_value` under the default `WholePartition` frame
/// (no `ORDER BY` at all — a *different* window in the same query) is the
/// partition's physically-last row.
/// duckdb: see module doc; both forms individually verified against
/// `SELECT id, first_value(val) OVER (PARTITION BY grp ORDER BY id) fv,
///         last_value(val) OVER (PARTITION BY grp) lv FROM w ORDER BY id`.
#[test]
fn first_value_with_order_by_and_last_value_without_it() {
    let mut db = session_with_w();
    let rows = run(
        &mut db,
        "SELECT id, first_value(val) OVER (PARTITION BY grp ORDER BY id) fv, \
                last_value(val) OVER (PARTITION BY grp) lv \
         FROM w ORDER BY id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i64v(1), f64v(10.0), f64v(30.0)],
            vec![i64v(2), f64v(10.0), f64v(30.0)],
            vec![i64v(3), f64v(10.0), f64v(30.0)],
            vec![i64v(4), f64v(5.0), f64v(15.0)],
            vec![i64v(5), f64v(5.0), f64v(15.0)],
            vec![i64v(6), f64v(5.0), f64v(15.0)],
        ]
    );
}

/// `QUALIFY` filters on a window function result computed over the *whole*
/// partition set, applied after all window functions are evaluated —
/// picking the max-`val` row per partition here.
/// duckdb: `SELECT id, grp, val FROM w QUALIFY row_number() OVER
///          (PARTITION BY grp ORDER BY val DESC) = 1 ORDER BY id`
#[test]
fn qualify_filters_on_a_window_function_result() {
    let mut db = session_with_w();
    // `NULLS LAST` used to need spelling out explicitly here: `sql::parser`
    // defaulted `nulls_first` to `desc` (i.e. `NULLS FIRST` under `DESC`,
    // the SQL-standard/Postgres-style "NULL is the largest value"
    // convention), which diverged from DuckDB's actual default of always
    // `NULLS LAST` (verified against a real `duckdb` CLI). Now fixed in
    // `sql::parser::order_item`, so the explicit `NULLS LAST` below is
    // redundant but left in place as a belt-and-suspenders regression
    // pin — see `default_nulls_ordering_matches_duckdb` below for a test of
    // the default itself.
    let rows = run(
        &mut db,
        "SELECT id, grp, val FROM w \
         QUALIFY row_number() OVER (PARTITION BY grp ORDER BY val DESC NULLS LAST) = 1 \
         ORDER BY id",
    );
    assert_eq!(rows, vec![vec![i64v(3), i64v(1), f64v(30.0)], vec![i64v(6), i64v(2), f64v(15.0)]]);
}

/// Confirms the `ORDER BY` `NULLS` default without spelling it out:
/// DuckDB always puts `NULL` last, for both `ASC` and `DESC` — not the
/// SQL-standard "NULL is the largest value" convention, which would put it
/// last for `ASC` but *first* for `DESC`.
/// duckdb: `SELECT x FROM (VALUES (1),(NULL),(3)) t(x) ORDER BY x DESC`
/// duckdb: `SELECT x FROM (VALUES (1),(NULL),(3)) t(x) ORDER BY x ASC`
#[test]
fn default_nulls_ordering_matches_duckdb() {
    let mut db = Session::new();
    // A wholly-blank line parses as "no row," not a NULL-valued row
    // (`format::csv`), so the NULL needs a non-blank sibling column.
    db.register_bytes_as("t", b"x,y\n1,a\n,b\n3,c\n".to_vec(), FormatKind::Csv).unwrap();
    let desc = run(&mut db, "SELECT x FROM t ORDER BY x DESC");
    assert_eq!(desc, vec![vec![i64v(3)], vec![i64v(1)], vec![Value::Null]]);
    let asc = run(&mut db, "SELECT x FROM t ORDER BY x ASC");
    assert_eq!(asc, vec![vec![i64v(1)], vec![i64v(3)], vec![Value::Null]]);
}

// =========================================================================
// GROUPING SETS / ROLLUP / CUBE: duplicate sets, an all-empty grouping set
// list, ROLLUP/CUBE cross-checked against their explicit GROUPING SETS
// expansion.
// =========================================================================

/// `gs(cat, sub, amt)`: two categories, with `B`/`x` appearing twice (so
/// per-set aggregates differ from a naive "distinct combinations" count).
///
/// ```text
/// (A,x,10) (A,y,20) (B,x,5) (B,y,15) (B,x,25)
/// ```
fn session_with_gs() -> Session {
    let mut sess = Session::new();
    csv(&mut sess, "gs", "cat,sub,amt\nA,x,10\nA,y,20\nB,x,5\nB,y,15\nB,x,25\n");
    sess
}

/// Listing the same grouping set twice in `GROUPING SETS` is not
/// deduplicated: each occurrence produces its own copy of every group's
/// row (DuckDB does the same — it is not a distinct-sets operation).
/// duckdb: `SELECT cat, count(*) c FROM gs GROUP BY GROUPING SETS ((cat), (cat)) ORDER BY cat`
#[test]
fn duplicate_grouping_sets_are_not_deduplicated() {
    let mut db = session_with_gs();
    let rows = run(
        &mut db,
        "SELECT cat, count(*) c FROM gs GROUP BY GROUPING SETS ((cat), (cat)) ORDER BY cat",
    );
    assert_eq!(
        rows,
        vec![
            vec![s("A"), i64v(2)],
            vec![s("A"), i64v(2)],
            vec![s("B"), i64v(3)],
            vec![s("B"), i64v(3)],
        ]
    );
}

/// A `GROUPING SETS` list containing only the empty set `()` collapses to a
/// single grand-total row; referencing a grouping column bare in the
/// SELECT list is rejected, exactly like a plain `GROUP BY` with no
/// matching column — the empty set means no column is ever "live".
/// duckdb: `SELECT cat, sub, sum(amt) s FROM gs GROUP BY GROUPING SETS (())`
/// -> Binder Error ("cat" must appear in GROUP BY or be aggregated).
#[test]
fn grouping_sets_with_only_the_empty_set_rejects_bare_grouping_columns() {
    let mut db = session_with_gs();
    let err = db.prepare("SELECT cat, sub, sum(amt) s FROM gs GROUP BY GROUPING SETS (())", &[]);
    assert_eq!(code_of(err), Some(Code::NotGrouped));
}

/// The same query with only aggregates in the SELECT list (no bare grouping
/// columns) works and returns exactly one grand-total row.
#[test]
fn grouping_sets_with_only_the_empty_set_returns_one_grand_total_row() {
    let mut db = session_with_gs();
    let rows = run(&mut db, "SELECT sum(amt) s FROM gs GROUP BY GROUPING SETS (())");
    assert_eq!(rows, vec![vec![i128v(75)]]);
}

/// `ROLLUP(cat, sub)` must produce exactly the same rows (same values, same
/// multiset) as its explicit `GROUPING SETS` expansion
/// `((cat, sub), (cat), ())`.
#[test]
fn rollup_matches_its_explicit_grouping_sets_expansion() {
    let mut db = session_with_gs();
    let rollup =
        run(&mut db, "SELECT cat, sub, count(*) c FROM gs GROUP BY ROLLUP(cat, sub) ORDER BY 1, 2");
    let explicit = run(
        &mut db,
        "SELECT cat, sub, count(*) c FROM gs \
         GROUP BY GROUPING SETS ((cat, sub), (cat), ()) ORDER BY 1, 2",
    );
    assert_eq!(rollup, explicit);
    // duckdb: SELECT cat, sub, count(*) c FROM gs GROUP BY ROLLUP(cat, sub) ORDER BY 1, 2
    assert_eq!(
        rollup,
        vec![
            vec![s("A"), s("x"), i64v(1)],
            vec![s("A"), s("y"), i64v(1)],
            vec![s("A"), NULL, i64v(2)],
            vec![s("B"), s("x"), i64v(2)],
            vec![s("B"), s("y"), i64v(1)],
            vec![s("B"), NULL, i64v(3)],
            vec![NULL, NULL, i64v(5)],
        ]
    );
}

/// `CUBE(cat, sub)` must produce exactly the same rows as its explicit
/// `GROUPING SETS` expansion of all 4 subsets, and must be a strict
/// superset of the `ROLLUP` rows (the two extra CUBE-only subsets:
/// `(sub)` and `()` already shared, plus `(sub)` alone).
#[test]
fn cube_matches_its_explicit_grouping_sets_expansion() {
    let mut db = session_with_gs();
    let cube =
        run(&mut db, "SELECT cat, sub, count(*) c FROM gs GROUP BY CUBE(cat, sub) ORDER BY 1, 2");
    let explicit = run(
        &mut db,
        "SELECT cat, sub, count(*) c FROM gs \
         GROUP BY GROUPING SETS ((cat, sub), (cat), (sub), ()) ORDER BY 1, 2",
    );
    assert_eq!(cube, explicit);
    // duckdb: SELECT cat, sub, count(*) c FROM gs GROUP BY CUBE(cat, sub) ORDER BY 1, 2
    assert_eq!(
        cube,
        vec![
            vec![s("A"), s("x"), i64v(1)],
            vec![s("A"), s("y"), i64v(1)],
            vec![s("A"), NULL, i64v(2)],
            vec![s("B"), s("x"), i64v(2)],
            vec![s("B"), s("y"), i64v(1)],
            vec![s("B"), NULL, i64v(3)],
            vec![NULL, s("x"), i64v(3)],
            vec![NULL, s("y"), i64v(2)],
            vec![NULL, NULL, i64v(5)],
        ]
    );
}

#[test]
fn distinct_does_not_keep_hidden_order_by_keys() {
    let mut db = Session::new();
    db.register_bytes_as("t", b"a,b\n1,10\n1,20\n2,5\n".to_vec(), FormatKind::Csv).unwrap();
    let rows = run(&mut db, "SELECT DISTINCT a FROM t ORDER BY b");
    assert_eq!(rows.len(), 2, "got {rows:?}");
    // After sorting by b, first-row-wins DISTINCT keeps a=2 (b=5) then a=1 (b=10).
    assert_eq!(rows[0][0], Value::I64(2));
    assert_eq!(rows[1][0], Value::I64(1));
}

#[test]
fn order_by_alias_after_star_uses_the_alias_column() {
    let mut db = Session::new();
    db.register_bytes_as("t", b"id,name,score\n1,b,10\n2,a,20\n".to_vec(), FormatKind::Csv)
        .unwrap();
    let rows = run(&mut db, "SELECT *, score * 2 AS extra FROM t ORDER BY extra");
    assert_eq!(rows[0][0], Value::I64(1), "sorted by extra, not name: {rows:?}");
    assert_eq!(rows[1][0], Value::I64(2));
}

/// An unqualified `ORDER BY` alias resolves to the last output column when
/// duplicate aliases are present, matching DuckDB's post-projection scope.
/// duckdb: `SELECT x AS y, -x AS y FROM range(3) t(x) ORDER BY y`
#[test]
fn order_by_duplicate_alias_uses_last_output_alias() {
    let mut db = Session::new();
    let rows = run(&mut db, "SELECT x AS y, -x AS y FROM range(3) t(x) ORDER BY y");
    assert_eq!(
        rows,
        vec![vec![i64v(2), i64v(-2)], vec![i64v(1), i64v(-1)], vec![i64v(0), i64v(0)],]
    );
}

#[test]
fn date_plus_bigint_adds_days() {
    let mut db = Session::new();
    db.register_bytes_as("dual", b"x\n1\n".to_vec(), FormatKind::Csv).unwrap();
    let rows = run(&mut db, "SELECT DATE '2024-01-01' + CAST(1 AS BIGINT) FROM dual");
    assert_eq!(rows[0][0], Value::I32(19724));
    let rows = run(&mut db, "SELECT DATE '2024-01-01' + NULL FROM dual");
    assert_eq!(rows[0][0], Value::Null);
}

/// `GROUP BY` may name a select-list alias. Projection pushdown has to
/// resolve that alias to the underlying expression (otherwise `k` is
/// looked up in the input and `ColumnNotFound`).
/// duckdb: `SELECT id+1 AS k, count(*) c FROM t GROUP BY k ORDER BY k`
#[test]
fn group_by_select_list_alias_is_resolved() {
    let mut db = Session::new();
    db.register_bytes_as("t", b"id,v\n1,10\n2,10\n3,20\n".to_vec(), FormatKind::Csv).unwrap();
    let rows = run(&mut db, "SELECT v + 1 AS k, count(*) c FROM t GROUP BY k ORDER BY k");
    assert_eq!(rows, vec![vec![i64v(11), i64v(2)], vec![i64v(21), i64v(1)]]);
}

/// QUALIFY sees the SELECT output, including `* REPLACE`, a trailing alias
/// that shadows a star column, and `RENAME`.
#[test]
fn qualify_uses_post_projection_names() {
    let mut db = Session::new();
    db.register_bytes_as("t", b"id,score\n1,10\n2,20\n".to_vec(), FormatKind::Csv).unwrap();

    // REPLACE: filter on the replaced `score` (20, 40), not the input (10, 20).
    let rows =
        run(&mut db, "SELECT * REPLACE (score * 2 AS score) FROM t QUALIFY score > 15 ORDER BY id");
    assert_eq!(rows, vec![vec![i64v(1), i64v(20)], vec![i64v(2), i64v(40)]]);

    // Shadowed alias: last `id` wins (`id+10` → 11, 12).
    let rows = run(&mut db, "SELECT *, id + 10 AS id FROM t QUALIFY id > 5 ORDER BY id");
    assert_eq!(rows.len(), 2, "got {rows:?}");
    assert_eq!(rows[0][0], i64v(1));
    assert_eq!(rows[0][2], i64v(11));

    // RENAME: QUALIFY can use the new name.
    let rows = run(&mut db, "SELECT * RENAME (id AS pk) FROM t QUALIFY pk > 1 ORDER BY pk");
    assert_eq!(rows, vec![vec![i64v(2), i64v(20)]]);
}

// =========================================================================
// Binder regressions: projection pushdown across a subquery boundary,
// qualified references around window functions, `EXISTS`/`IN` over an
// ungrouped aggregate, GROUP BY name resolution, scalar subqueries in an
// aggregating query, and ORDER BY positional-term validation.
//
// Every expected value below was cross-checked against the `duckdb` CLI over
// the same CSV fixtures.
// =========================================================================

/// `p(id, flag, name)`: two rows share a `flag`, every `name` is distinct.
///
/// ```text
/// p: (1,x,a) (2,y,b) (3,x,c)
/// ```
/// Registered as a CSV byte source, i.e. a *file-backed* table, which is what
/// projection pushdown prunes — an in-memory DDL table is never narrowed and
/// would not exercise the pushdown path at all.
fn session_with_p() -> Session {
    let mut sess = Session::new();
    csv(&mut sess, "p", "id,flag,name\n1,x,a\n2,y,b\n3,x,c\n");
    sess
}

/// An outer column referenced *only* inside a correlated subquery must survive
/// projection pushdown. `collect_refs` stops at the subquery boundary, so
/// `p.flag` used to be pruned from the scan and the subquery then failed to
/// bind against the narrowed scope with `TableNotFound`.
/// duckdb: `SELECT id, (SELECT count(*) FROM p s WHERE s.flag = p.flag) FROM p ORDER BY id`
#[test]
fn outer_column_used_only_inside_a_subquery_is_not_pruned() {
    let mut db = session_with_p();
    let rows = run(
        &mut db,
        "SELECT id, (SELECT count(*) FROM p s WHERE s.flag = p.flag) FROM p ORDER BY id",
    );
    assert_eq!(rows, vec![vec![i64v(1), i64v(2)], vec![i64v(2), i64v(1)], vec![i64v(3), i64v(2)],]);

    // The same for EXISTS ...
    let rows = run(
        &mut db,
        "SELECT id FROM p WHERE EXISTS (SELECT 1 FROM p s WHERE s.name = p.name) ORDER BY id",
    );
    assert_eq!(rows, vec![vec![i64v(1)], vec![i64v(2)], vec![i64v(3)]]);

    // ... and for IN.
    let rows = run(
        &mut db,
        "SELECT id FROM p WHERE id IN (SELECT s.id FROM p s WHERE s.flag = p.flag) ORDER BY id",
    );
    assert_eq!(rows, vec![vec![i64v(1)], vec![i64v(2)], vec![i64v(3)]]);
}

/// A table-qualified column in the SELECT list or QUALIFY of a query that also
/// contains a window function. The scope rebuilt after the window columns used
/// to drop table qualifiers, so `e.id` no longer resolved.
/// duckdb: `SELECT e.id, row_number() OVER (ORDER BY e.id) FROM p e ORDER BY 1`
#[test]
fn qualified_columns_resolve_alongside_window_functions() {
    let mut db = session_with_p();
    let rows = run(&mut db, "SELECT e.id, row_number() OVER (ORDER BY e.id) FROM p e ORDER BY 1");
    assert_eq!(rows, vec![vec![i64v(1), i64v(1)], vec![i64v(2), i64v(2)], vec![i64v(3), i64v(3)],]);

    // The table's own name works as a qualifier too.
    let rows = run(&mut db, "SELECT p.id, row_number() OVER () FROM p ORDER BY 1");
    assert_eq!(rows.len(), 3);

    // QUALIFY compiles against the same scope.
    let rows = run(&mut db, "SELECT p.id FROM p QUALIFY row_number() OVER (ORDER BY p.id) = 1");
    assert_eq!(rows, vec![vec![i64v(1)]]);
}

/// `nums`/`dup`: `nums` carries a NULL, `dup` has a duplicated value.
///
/// ```text
/// nums: 1 2 3 NULL      dup: 1 1 2
/// ```
/// `nums` gets a second column purely so the NULL row is a line with a
/// separator on it rather than a blank line at end of file.
fn session_with_nums_dup() -> Session {
    let mut sess = Session::new();
    csv(&mut sess, "nums", "x,tag\n1,a\n2,b\n3,c\n,d\n");
    csv(&mut sess, "dup", "x\n1\n1\n2\n");
    sess
}

/// An aggregate with no GROUP BY produces exactly one row even over an empty
/// input, so `EXISTS` over it is unconditionally true. Rewriting it as a
/// semi-join on the correlation key instead dropped the outer rows whose
/// correlated group is empty — the very rows for which `count(*)` is 0.
/// duckdb: `SELECT x FROM nums WHERE EXISTS (SELECT count(*) FROM dup WHERE dup.x = nums.x) ORDER BY x`
#[test]
fn exists_over_an_ungrouped_aggregate_is_always_true() {
    let mut db = session_with_nums_dup();
    let rows = run(
        &mut db,
        "SELECT x FROM nums WHERE EXISTS (SELECT count(*) FROM dup WHERE dup.x = nums.x) ORDER BY x",
    );
    assert_eq!(rows, vec![vec![i64v(1)], vec![i64v(2)], vec![i64v(3)], vec![NULL]]);

    // ... and `NOT EXISTS` is therefore unconditionally false.
    let rows = run(
        &mut db,
        "SELECT count(*) FROM nums WHERE NOT EXISTS (SELECT count(*) FROM dup WHERE dup.x = nums.x)",
    );
    assert_eq!(rows, vec![vec![i64v(0)]]);
}

/// `IN` over an ungrouped aggregate is a comparison against that one value,
/// with a missing correlated group counting as `count(*) = 0` rather than as
/// "no row to match".
/// duckdb: `SELECT x FROM nums WHERE 0 IN (SELECT count(*) FROM dup WHERE dup.x = nums.x + 10) ORDER BY x`
#[test]
fn in_over_an_ungrouped_aggregate_compares_against_the_single_value() {
    let mut db = session_with_nums_dup();
    // No `dup` row ever matches, so every group is empty and `count(*)` is 0.
    let rows = run(
        &mut db,
        "SELECT x FROM nums WHERE 0 IN (SELECT count(*) FROM dup WHERE dup.x = nums.x + 10) ORDER BY x",
    );
    assert_eq!(rows, vec![vec![i64v(1)], vec![i64v(2)], vec![i64v(3)], vec![NULL]]);

    // Only `x = 1` has two matching `dup` rows.
    let rows = run(
        &mut db,
        "SELECT x FROM nums WHERE 2 IN (SELECT count(*) FROM dup WHERE dup.x = nums.x) ORDER BY x",
    );
    assert_eq!(rows, vec![vec![i64v(1)]]);

    // An uncorrelated one-row aggregate, including the NULL-propagating `NOT IN`.
    let rows = run(&mut db, "SELECT x FROM nums WHERE x IN (SELECT max(x) FROM dup) ORDER BY x");
    assert_eq!(rows, vec![vec![i64v(2)]]);
    let rows =
        run(&mut db, "SELECT x FROM nums WHERE x NOT IN (SELECT max(x) FROM dup) ORDER BY x");
    assert_eq!(rows, vec![vec![i64v(1)], vec![i64v(3)]]);
}

/// `GROUP BY <name>` binds an *input* column before a SELECT-list alias of the
/// same name, as DuckDB and PostgreSQL do. ORDER BY keeps the opposite
/// (alias-first) rule.
/// duckdb: `SELECT id % 2 AS id, count(*) FROM p GROUP BY id ORDER BY 1, 2`
#[test]
fn group_by_name_prefers_the_input_column_over_a_select_alias() {
    let mut db = session_with_p();
    // Grouping by the table's `id` gives one group per row, not two groups of `id % 2`.
    let rows = run(&mut db, "SELECT id % 2 AS id, count(*) FROM p GROUP BY id ORDER BY 1, 2");
    assert_eq!(rows, vec![vec![i64v(0), i64v(1)], vec![i64v(1), i64v(1)], vec![i64v(1), i64v(1)]]);

    // Consequently a select item that is not grouped is now correctly rejected,
    // where the alias-first rule used to return a silently wrong answer.
    // duckdb: `Binder Error: column "flag" must appear in the GROUP BY clause`
    let err = db.prepare("SELECT flag AS id, count(*) FROM p GROUP BY id", &[]);
    assert_eq!(code_of(err), Some(Code::NotGrouped));

    // A name the FROM clause does not provide still falls back to the alias.
    let rows = run(&mut db, "SELECT id % 2 AS k, count(*) FROM p GROUP BY k ORDER BY 1");
    assert_eq!(rows, vec![vec![i64v(0), i64v(1)], vec![i64v(1), i64v(2)]]);

    // ORDER BY stays alias-first: `id` here is the output `id % 2`.
    let rows = run(&mut db, "SELECT id % 2 AS id, count(*) FROM p GROUP BY 1 ORDER BY id");
    assert_eq!(rows, vec![vec![i64v(0), i64v(1)], vec![i64v(1), i64v(2)]]);
}

/// An uncorrelated scalar subquery is a constant with respect to the grouping,
/// so it is legal in the SELECT list and in HAVING of an aggregating query.
/// duckdb: `SELECT flag, count(*) + (SELECT max(id) FROM p) FROM p GROUP BY flag ORDER BY 1`
#[test]
fn uncorrelated_scalar_subquery_is_allowed_in_an_aggregating_query() {
    let mut db = session_with_p();
    let rows = run(
        &mut db,
        "SELECT flag, count(*) + (SELECT max(id) FROM p) FROM p GROUP BY flag ORDER BY 1",
    );
    assert_eq!(rows, vec![vec![s("x"), i64v(5)], vec![s("y"), i64v(4)]]);

    // HAVING reads it too (`min(id)` is 1, so only the two-row group survives).
    let rows = run(
        &mut db,
        "SELECT flag, count(*) FROM p GROUP BY flag HAVING count(*) > (SELECT min(id) FROM p) \
         ORDER BY 1",
    );
    assert_eq!(rows, vec![vec![s("x"), i64v(2)]]);

    // With no rows at all, the ungrouped aggregate must still emit its single
    // row — the constant is joined on after the aggregate, not folded into it.
    let rows = run(&mut db, "SELECT count(*) + (SELECT max(id) FROM p) FROM p WHERE id > 100");
    assert_eq!(rows, vec![vec![i64v(3)]]);
}

/// A qualified and an unqualified spelling of one column match between SELECT
/// and GROUP BY. Structural equality alone compares raw syntax, so both
/// directions used to fail with `NotGrouped`.
/// duckdb: `SELECT flag, count(*) FROM p GROUP BY p.flag ORDER BY 1`
#[test]
fn qualified_and_unqualified_group_by_refer_to_the_same_column() {
    let mut db = session_with_p();
    let rows = run(&mut db, "SELECT flag, count(*) FROM p GROUP BY p.flag ORDER BY 1");
    assert_eq!(rows, vec![vec![s("x"), i64v(2)], vec![s("y"), i64v(1)]]);

    let rows = run(&mut db, "SELECT e.flag, count(*) FROM p e GROUP BY flag ORDER BY 1");
    assert_eq!(rows, vec![vec![s("x"), i64v(2)], vec![s("y"), i64v(1)]]);

    // A join still keeps the two same-named columns apart.
    let rows = run(
        &mut db,
        "SELECT a.flag, count(*) FROM p a JOIN p b ON a.id = b.id GROUP BY a.flag ORDER BY 1",
    );
    assert_eq!(rows, vec![vec![s("x"), i64v(2)], vec![s("y"), i64v(1)]]);
}

/// A positional ORDER BY / GROUP BY term outside `1..=<output columns>` is an
/// error, not a silently ignored constant sort key.
/// duckdb: `Binder Error: ORDER term out of range - should be between 1 and 1`
#[test]
fn out_of_range_positional_order_terms_are_rejected() {
    let mut db = session_with_p();
    for sql in [
        "SELECT id FROM p ORDER BY 0",
        "SELECT id FROM p ORDER BY -1",
        "SELECT id FROM p ORDER BY 1.5",
        "SELECT id FROM p ORDER BY 99",
        "SELECT count(*) FROM p GROUP BY 0",
    ] {
        assert_eq!(code_of(db.prepare(sql, &[])), Some(Code::ColumnNotFound), "{sql}");
    }
    // A valid position is unaffected.
    let rows = run(&mut db, "SELECT id FROM p ORDER BY 1");
    assert_eq!(rows, vec![vec![i64v(1)], vec![i64v(2)], vec![i64v(3)]]);
}

/// A non-recursive CTE may be referenced more than once; each reference gets
/// its own copy of the bound plan and recomputes the body.
/// duckdb: `WITH c AS (SELECT id FROM p) SELECT c1.id FROM c c1 JOIN c c2 ON c1.id = c2.id ORDER BY 1`
#[test]
fn a_non_recursive_cte_can_be_referenced_more_than_once() {
    let mut db = session_with_p();
    let rows = run(
        &mut db,
        "WITH c AS (SELECT id FROM p) SELECT c1.id FROM c c1 JOIN c c2 ON c1.id = c2.id ORDER BY 1",
    );
    assert_eq!(rows, vec![vec![i64v(1)], vec![i64v(2)], vec![i64v(3)]]);

    // Three references, and through a set operation as well.
    let rows = run(
        &mut db,
        "WITH c AS (SELECT id FROM p WHERE id < 3) \
         SELECT id FROM c UNION ALL SELECT id FROM c UNION ALL SELECT id FROM c ORDER BY 1",
    );
    assert_eq!(rows.len(), 6);
}
