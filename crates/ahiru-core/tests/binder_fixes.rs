//! Regression tests for binder fixes: ambiguous references inside correlated
//! subqueries, `ORDER BY` ordinal range checking against the projected columns,
//! flat (left-deep) operator chains, qualified predicates on an `UNNEST` column,
//! `GROUPING(alias)`, and SELECT-list aliases in `HAVING`.
//!
//! All expected values are cross-checked against `duckdb -c "SELECT ..."`
//! (`tests/data/basic.parquet` is a real file written by DuckDB).

use ahiru_core::error::{code_of, Code};
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn data(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn session_with_basic() -> Session {
    let mut s = Session::new();
    s.register_bytes("t", data("basic.parquet")).unwrap();
    s
}

/// Runs `sql` and extracts the result as `Vec<Vec<Value>>`.
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

fn err(session: &mut Session, sql: &str) -> Option<Code> {
    code_of(session.prepare(sql, &[]))
}

// --- Ambiguous references inside a correlated subquery ------------------------

/// An ambiguous name inside a subquery used to be silently re-bound as a
/// correlation key against the outer query, quietly returning a wrong answer.
/// DuckDB raises "Ambiguous reference"; so do we.
#[test]
fn an_ambiguous_column_in_a_subquery_is_an_error_not_a_correlation_key() {
    let mut s = session_with_basic();
    let sql = "SELECT count(*) FROM t WHERE t.id IN \
               (SELECT a.id FROM t a JOIN t b ON a.id = b.id WHERE id = 5)";
    assert_eq!(err(&mut s, sql), Some(Code::AmbiguousColumn));
}

/// The same shape in `EXISTS`, where the ambiguous name sits in a conjunct next
/// to a genuine correlation predicate.
#[test]
fn an_ambiguous_column_beside_a_correlation_predicate_is_still_an_error() {
    let mut s = session_with_basic();
    let sql = "SELECT count(*) FROM t WHERE EXISTS \
               (SELECT 1 FROM t a JOIN t b ON a.id = b.id WHERE a.id = t.id AND id < 5)";
    assert_eq!(err(&mut s, sql), Some(Code::AmbiguousColumn));
}

/// A genuine correlated subquery keeps working: a name the inner scope simply
/// does not have still resolves against the outer one.
#[test]
fn a_genuine_correlated_reference_still_binds() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT count(*) FROM t WHERE EXISTS (SELECT 1 FROM t s WHERE s.id = t.id AND s.id < 5)",
    );
    assert_eq!(rows, vec![vec![Value::I64(5)]]);
}

/// `GROUP BY` falls back to a select-list alias only for a name the input does
/// not have. An *ambiguous* input column is an error, as in DuckDB, not a
/// silent fall-through to an alias that happens to share the name.
#[test]
fn an_ambiguous_group_by_name_is_not_resolved_as_an_alias() {
    let mut s = session_with_basic();
    let sql = "SELECT a.id % 2 AS id, count(*) FROM t a JOIN t b ON a.id = b.id GROUP BY id";
    assert_eq!(err(&mut s, sql), Some(Code::AmbiguousColumn));
}

// --- ORDER BY ordinal range ---------------------------------------------------

/// `QUALIFY` appends hidden columns to the projection. An `ORDER BY` ordinal
/// must not be able to reach them: DuckDB rejects the position as out of range.
#[test]
fn an_order_by_ordinal_cannot_reach_a_hidden_qualify_column() {
    let mut s = session_with_basic();
    let sql = "SELECT id FROM t QUALIFY row_number() OVER (ORDER BY id DESC) <= 3 ORDER BY 2";
    assert_eq!(err(&mut s, sql), Some(Code::ColumnNotFound));
}

/// An earlier `ORDER BY` term that is not an output column also becomes a hidden
/// column, and must stay unreachable by a later ordinal.
#[test]
fn an_order_by_ordinal_cannot_reach_an_earlier_sort_key() {
    let mut s = session_with_basic();
    assert_eq!(err(&mut s, "SELECT id FROM t ORDER BY id % 7, 2"), Some(Code::ColumnNotFound));
}

/// Ordinals within the projection still work, including with QUALIFY present.
#[test]
fn order_by_ordinals_within_the_projection_still_work() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT id FROM t QUALIFY row_number() OVER (ORDER BY id DESC) <= 3 ORDER BY 1",
    );
    assert_eq!(rows, vec![vec![Value::I32(997)], vec![Value::I32(998)], vec![Value::I32(999)]]);
}

/// An aggregate inside QUALIFY over a column that is not a grouping key.
///
/// QUALIFY's windows and aggregates become hidden columns, but the reference
/// walk that follows kept descending into them and collected the bare `score`
/// inside `max(score)`. That column is consumed by the aggregate operator, so
/// compiling it against the post-aggregate scope failed with `ColumnNotFound`
/// — while `count(*)` or a grouping column in the same position worked.
///
/// duckdb (on `tests/data/basic.parquet`):
/// ```text
/// SELECT flag, max(score) m FROM t GROUP BY flag
///   QUALIFY row_number() OVER (ORDER BY max(score)) = 1     -> false|1497.0
/// SELECT flag FROM t GROUP BY flag
///   QUALIFY row_number() OVER (ORDER BY max(score)) = 1     -> false
/// SELECT flag, sum(id) s FROM t GROUP BY flag HAVING sum(id) > 0
///   QUALIFY row_number() OVER (ORDER BY sum(id)) > 1 ORDER BY flag
///                                                           -> false|332667
/// ```
#[test]
fn an_aggregate_over_a_non_grouped_column_works_inside_qualify() {
    let mut s = session_with_basic();
    assert_eq!(
        run(
            &mut s,
            "SELECT flag, max(score) m FROM t GROUP BY flag \
             QUALIFY row_number() OVER (ORDER BY max(score)) = 1",
        ),
        vec![vec![Value::Bool(false), Value::F64(1497.0)]]
    );
    // The aggregate appears only in QUALIFY, so it is not already substituted
    // by a SELECT-list item.
    assert_eq!(
        run(
            &mut s,
            "SELECT flag FROM t GROUP BY flag \
             QUALIFY row_number() OVER (ORDER BY max(score)) = 1",
        ),
        vec![vec![Value::Bool(false)]]
    );
    assert_eq!(
        run(
            &mut s,
            "SELECT flag, sum(id) s FROM t GROUP BY flag HAVING sum(id) > 0 \
             QUALIFY row_number() OVER (ORDER BY sum(id)) > 1 ORDER BY flag",
        ),
        vec![vec![Value::Bool(false), Value::I128(332_667)]]
    );
}

// --- Flat left-deep operator chains -------------------------------------------

/// `WHERE a AND b AND ... ` is left-deep but not nested. A 200-term predicate
/// must bind, not be rejected as "expression nesting too deep".
#[test]
fn a_flat_where_of_200_conjuncts_binds() {
    let mut s = session_with_basic();
    let mut sql = String::from("SELECT count(*) FROM t WHERE ");
    for i in 0..200 {
        if i > 0 {
            sql.push_str(" AND ");
        }
        sql.push_str(&format!("id <> {i}"));
    }
    assert_eq!(run(&mut s, &sql), vec![vec![Value::I64(800)]]);
}

/// The same for a flat arithmetic chain, which the *compiler* walks rather than
/// the conjunct splitter.
#[test]
fn a_flat_sum_of_200_terms_compiles() {
    let mut s = session_with_basic();
    let mut sql = String::from("SELECT ");
    for i in 0..200 {
        if i > 0 {
            sql.push('+');
        }
        sql.push('1');
    }
    sql.push_str(" AS n FROM t LIMIT 1");
    assert_eq!(run(&mut s, &sql), vec![vec![Value::I32(200)]]);
}

/// And for a flat `||` chain, which takes a different branch of the compiler.
#[test]
fn a_flat_concat_of_200_terms_compiles() {
    let mut s = session_with_basic();
    let mut sql = String::from("SELECT ");
    for i in 0..200 {
        if i > 0 {
            sql.push_str("||");
        }
        sql.push_str("'a'");
    }
    sql.push_str(" AS c FROM t LIMIT 1");
    let rows = run(&mut s, &sql);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], Value::Bytes("a".repeat(200).into_bytes()));
}

/// Genuine nesting is still rejected. `1+(1+(1+...))` is 200 levels deep.
#[test]
fn genuinely_nested_expressions_are_still_rejected() {
    let mut s = session_with_basic();
    let mut sql = String::from("SELECT ");
    sql.push_str(&"1+(".repeat(200));
    sql.push('1');
    sql.push_str(&")".repeat(200));
    sql.push_str(" FROM t");
    assert_eq!(err(&mut s, &sql), Some(Code::ExpressionTooDeep));
}

/// Set operations are capped separately, by the parser's `MAX_LINKS`: each link
/// of a `UNION ALL` chain is a `Box` that recurses once per link when dropped
/// and once per link in the binder, so the cap stays where it is. 100 branches
/// are rejected; a chain within the cap still works.
#[test]
fn set_operation_chains_are_capped_by_max_links() {
    let mut s = session_with_basic();
    let branch = "SELECT min(id) FROM t";
    let long = [branch; 100].join(" UNION ALL ");
    assert_eq!(err(&mut s, &long), Some(Code::ExpressionTooDeep));
    let short = [branch; 40].join(" UNION ALL ");
    assert_eq!(run(&mut s, &short).len(), 40);
}

// --- Qualified predicate on an UNNEST column ----------------------------------

/// `WHERE u.x > 1` is pushed down to the implicit-LATERAL `UNNEST` relation, and
/// the scope it is compiled against has to keep the `u` qualifier.
#[test]
fn a_qualified_predicate_on_an_unnest_column_binds() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT u.x FROM t, UNNEST([1,2,3]) AS u(x) WHERE u.x > 1 AND t.id = 0 ORDER BY 1",
    );
    assert_eq!(rows, vec![vec![Value::I64(2)], vec![Value::I64(3)]]);
}

/// The unqualified spelling keeps working, and so does a query that filters on
/// the UNNEST column without selecting it.
#[test]
fn unqualified_and_unselected_unnest_predicates_still_work() {
    let mut s = session_with_basic();
    let rows =
        run(&mut s, "SELECT x FROM t, UNNEST([1,2,3]) AS u(x) WHERE x > 2 AND t.id = 0 ORDER BY 1");
    assert_eq!(rows, vec![vec![Value::I64(3)]]);
    let rows =
        run(&mut s, "SELECT t.id FROM t, UNNEST([1,2,3]) AS u(x) WHERE u.x > 2 AND t.id = 0");
    assert_eq!(rows, vec![vec![Value::I32(0)]]);
}

// --- GROUPING(alias) ----------------------------------------------------------

/// `GROUPING()`'s argument may be a SELECT-list alias, as in DuckDB. Projection
/// pushdown used to reject it because the alias is not an input column.
#[test]
fn grouping_accepts_a_select_list_alias() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT id % 2 AS m, count(*), grouping(m) FROM t \
         GROUP BY GROUPING SETS ((m), ()) ORDER BY 1 NULLS FIRST",
    );
    assert_eq!(
        rows,
        vec![
            vec![Value::Null, Value::I64(1000), Value::I64(1)],
            vec![Value::I32(0), Value::I64(500), Value::I64(0)],
            vec![Value::I32(1), Value::I64(500), Value::I64(0)],
        ]
    );
}

// --- SELECT-list aliases in HAVING --------------------------------------------

/// DuckDB accepts an aggregate's alias in `HAVING`.
#[test]
fn having_accepts_an_aggregate_alias() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT id % 2 AS a, sum(id) AS s FROM t GROUP BY a HAVING s > 249600 ORDER BY 1",
    );
    assert_eq!(rows, vec![vec![Value::I32(1), Value::I128(250000)]]);
}

/// And a grouping expression's alias.
#[test]
fn having_accepts_a_grouping_alias() {
    let mut s = session_with_basic();
    let rows = run(&mut s, "SELECT id % 3 AS m, count(*) AS c FROM t GROUP BY m HAVING m = 1");
    assert_eq!(rows, vec![vec![Value::I32(1), Value::I64(333)]]);
}

/// The alias also resolves on the `GROUPING SETS` path, which builds its
/// aggregate output schema separately.
#[test]
fn having_accepts_an_alias_under_grouping_sets() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT id % 2 AS m, count(*) AS c FROM t GROUP BY GROUPING SETS ((m), ()) HAVING c > 600",
    );
    assert_eq!(rows, vec![vec![Value::Null, Value::I64(1000)]]);
}

/// An input column of the same name still wins over the alias, matching the
/// precedence `GROUP BY` uses.
#[test]
fn an_input_column_still_wins_over_a_having_alias() {
    let mut s = session_with_basic();
    let rows = run(
        &mut s,
        "SELECT id AS flag, count(*) AS c FROM t GROUP BY id, flag HAVING flag ORDER BY 1 LIMIT 1",
    );
    assert_eq!(rows, vec![vec![Value::I32(0), Value::I64(1)]]);
}
