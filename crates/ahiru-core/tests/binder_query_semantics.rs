//! Regression tests for binder/planner query semantics: star expansion hiding the binder's
//! helper columns, `DISTINCT ON (<ordinal>)`, `UNPIVOT` dropping NULL values, the position of
//! a select-list `UNNEST` relative to window functions and `QUALIFY`, `HAVING` alias
//! precedence, the plan-depth limit, and nullability through outer joins.
//!
//! Every expected value was cross-checked against `duckdb` v1.4.4.

use ahiru_core::error::{code_of, Code};
use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::{Field, Value};

fn data(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// `t` = small_a.parquet (`k, v`: `(0,0) (1,2) (2,4) (3,6) (4,8)`),
/// `t2` = small_b.parquet (`k, w`: `(2,0) (3,10) (4,20) (5,30) (6,40)`).
fn session_ab() -> Session {
    let mut s = Session::new();
    s.register_bytes("t", data("small_a.parquet")).unwrap();
    s.register_bytes("t2", data("small_b.parquet")).unwrap();
    s
}

fn session_with(file: &str) -> Session {
    let mut s = Session::new();
    s.register_bytes_as("t", data(file), FormatKind::Parquet).unwrap();
    s
}

fn run(s: &mut Session, sql: &str) -> (Vec<Field>, Vec<Vec<Value>>) {
    let mut q = match s.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
    };
    let schema = q.schema.clone();
    let mut rows = Vec::new();
    loop {
        match s.step(&mut q).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
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
    (schema, rows)
}

fn names(schema: &[Field]) -> Vec<&str> {
    schema.iter().map(|f| f.name.as_str()).collect()
}

fn i32v(v: i32) -> Value {
    Value::I32(v)
}
fn i64v(v: i64) -> Value {
    Value::I64(v)
}
fn s(v: &str) -> Value {
    Value::Bytes(v.as_bytes().to_vec())
}

// --- `*` never exposes helper columns -------------------------------------------

/// A scalar subquery, a quantified comparison and a select-list UNNEST each add a helper
/// column to the binder's scope. `*` / `COLUMNS(*)` / `t.*` must expand to the FROM clause's
/// columns only (duckdb: `k,v,m` / `k,v`).
#[test]
fn star_does_not_expose_scalar_subquery_or_quantified_helper_columns() {
    let mut db = session_ab();
    let cases: &[(&str, &[&str])] = &[
        ("SELECT *, (SELECT max(w) FROM t2) AS m FROM t", &["k", "v", "m"]),
        ("SELECT * FROM t WHERE v > (SELECT avg(v) FROM t)", &["k", "v"]),
        ("SELECT COLUMNS(*) FROM t WHERE v > (SELECT avg(v) FROM t)", &["k", "v"]),
        ("SELECT t.* FROM t WHERE v > (SELECT avg(v) FROM t)", &["k", "v"]),
        ("SELECT * EXCLUDE (k) FROM t WHERE v > (SELECT avg(v) FROM t)", &["v"]),
        ("SELECT * REPLACE (v + 1 AS v) FROM t WHERE v > (SELECT avg(v) FROM t)", &["k", "v"]),
        ("SELECT COLUMNS('.*') FROM t WHERE k > ANY (SELECT k FROM t2)", &["k", "v"]),
        ("SELECT * FROM t WHERE k > ALL (SELECT k - 3 FROM t2)", &["k", "v"]),
    ];
    for (sql, want) in cases {
        let (schema, _) = run(&mut db, sql);
        assert_eq!(names(&schema), *want, "{sql}");
    }
    let (_, rows) = run(&mut db, "SELECT * FROM t WHERE k > ALL (SELECT k - 3 FROM t2)");
    assert_eq!(rows, vec![vec![i32v(4), i32v(8)]]);
    // A helper column is not a column the regex can match either.
    assert_eq!(
        code_of(db.prepare("SELECT COLUMNS('subq') FROM t WHERE v > (SELECT 1)", &[])),
        Some(Code::ColumnNotFound)
    );
}

/// The extra column used to break a set operation over such a query ("type mismatch").
#[test]
fn star_with_a_subquery_feeds_a_set_operation() {
    let mut db = session_ab();
    let (schema, rows) = run(
        &mut db,
        "SELECT * FROM (SELECT * FROM t WHERE v > (SELECT avg(v) FROM t)) \
         UNION ALL SELECT * FROM t WHERE k > ANY (SELECT k + 1 FROM t2) ORDER BY 1, 2",
    );
    assert_eq!(names(&schema), ["k", "v"]);
    assert_eq!(rows, vec![vec![i32v(3), i32v(6)], vec![i32v(4), i32v(8)], vec![i32v(4), i32v(8)],]);
}

/// `*` next to a select-list UNNEST: `id, xs, u` (duckdb), not `id, xs, unnest, u`; the
/// qualified form keeps working (it used to lose the `t` qualifier and expand to nothing).
#[test]
fn star_does_not_expose_the_unnest_helper_column() {
    let mut db = session_with("list1.parquet");
    for sql in [
        "SELECT *, UNNEST(xs) AS u FROM t WHERE id < 2",
        "SELECT t.*, UNNEST(xs) AS u FROM t WHERE id < 2",
    ] {
        let (schema, rows) = run(&mut db, sql);
        assert_eq!(names(&schema), ["id", "xs", "u"], "{sql}");
        assert_eq!(rows.len(), 6, "{sql}");
    }
}

// --- DISTINCT ON (<ordinal>) ------------------------------------------------------

/// `DISTINCT ON (1)` names the first output column, like `ORDER BY 1`; it used to be a
/// constant key that collapsed the result to one row.
#[test]
fn distinct_on_accepts_ordinals() {
    let mut db = session_with("pivot_small.parquet");
    let (_, rows) = run(&mut db, "SELECT DISTINCT ON (1) region, amount FROM t ORDER BY 1, 2");
    assert_eq!(rows, vec![vec![s("east"), i32v(10)], vec![s("west"), i32v(5)]]);
    let (_, rows) = run(&mut db, "SELECT DISTINCT ON (2) * FROM t ORDER BY 2 DESC, 1");
    assert_eq!(rows.len(), 3);
    // Out of range / non-integer: an error, as in duckdb ("ORDER term out of range").
    for sql in [
        "SELECT DISTINCT ON (3) region, amount FROM t",
        "SELECT DISTINCT ON (0) region, amount FROM t",
        "SELECT DISTINCT ON (1.5) region, amount FROM t",
    ] {
        assert_eq!(code_of(db.prepare(sql, &[])), Some(Code::ColumnNotFound), "{sql}");
    }
}

// --- UNPIVOT drops NULL values ----------------------------------------------------

/// DuckDB's `UNPIVOT` statement emits no row for a NULL value. `basic.parquet` has 1000
/// rows, 200 of whose `big` values are NULL.
#[test]
fn unpivot_drops_null_values() {
    let mut s = Session::new();
    s.register_bytes("t", data("basic.parquet")).unwrap();
    let (_, rows) = run(&mut s, "UNPIVOT t ON big, id INTO NAME n VALUE v");
    assert_eq!(rows.len(), 1800);
    assert!(rows.iter().all(|r| r.last() != Some(&Value::Null)));
}

// --- Select-list UNNEST runs after window functions and QUALIFY ----------------------

/// duckdb evaluates window functions (and QUALIFY) on the rows *before* a select-list UNNEST
/// expands them: `count(*) OVER ()` is 2 here, not 6.
#[test]
fn select_list_unnest_expands_after_window_functions() {
    let mut db = session_with("list1.parquet");
    let (_, rows) =
        run(&mut db, "SELECT id, UNNEST(xs) AS x, count(*) OVER () AS n FROM t WHERE id < 2");
    assert_eq!(rows.len(), 6);
    assert!(rows.iter().all(|r| r[2] == i64v(2)), "{rows:?}");
}

#[test]
fn select_list_unnest_expands_after_qualify() {
    let mut db = session_with("list1.parquet");
    let (_, rows) = run(
        &mut db,
        "SELECT id, UNNEST(xs) AS x FROM t QUALIFY row_number() OVER (ORDER BY id) = 2",
    );
    let ids: Vec<&Value> = rows.iter().map(|r| &r[0]).collect();
    assert_eq!(ids, [&i32v(1), &i32v(1), &i32v(1)]);
    // ORDER BY / DISTINCT / LIMIT still see the expanded rows.
    let (_, rows) = run(
        &mut db,
        "SELECT DISTINCT id, UNNEST(xs) AS x FROM t WHERE id < 3 ORDER BY id DESC LIMIT 4",
    );
    assert_eq!(rows.len(), 4);
    assert!(rows[..3].iter().all(|r| r[0] == i32v(2)));
    // The UNNEST output does not exist yet when QUALIFY filters (duckdb: "UNNEST not
    // supported here").
    assert!(db.prepare("SELECT UNNEST(xs) AS x FROM t QUALIFY x = 2", &[]).is_err());
}

// --- HAVING alias precedence ----------------------------------------------------------

/// A HAVING name that is not a grouping column resolves to the SELECT-list alias even when
/// an (ungrouped) input column has the same name; a grouped input column still wins.
#[test]
fn having_prefers_the_alias_over_an_ungrouped_input_column() {
    let mut db = session_ab();
    let (_, rows) =
        run(&mut db, "SELECT k % 2 AS k2, sum(v) AS v FROM t GROUP BY k % 2 HAVING v > 8");
    assert_eq!(rows, vec![vec![i32v(0), Value::I128(12)]]);
    let (_, rows) = run(
        &mut db,
        "SELECT k % 2 AS k2, sum(v) AS v FROM t GROUP BY GROUPING SETS ((k % 2), ()) \
         HAVING v > 8 ORDER BY 1 NULLS LAST",
    );
    assert_eq!(rows, vec![vec![i32v(0), Value::I128(12)], vec![Value::Null, Value::I128(20)]]);
    // `v` is grouped here, so it is the input column (duckdb: rows 6 and 8).
    let (_, rows) = run(&mut db, "SELECT v, sum(k) AS v FROM t GROUP BY v HAVING v > 4 ORDER BY 1");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0][0], i32v(6));
    // Inside an aggregate the name is always the input column.
    let (_, rows) = run(
        &mut db,
        "SELECT k % 2 AS k2, sum(v) AS v FROM t GROUP BY k % 2 HAVING sum(v) > 8 AND v > 0",
    );
    assert_eq!(rows, vec![vec![i32v(0), Value::I128(12)]]);
    // A qualified reference is never an alias.
    assert_eq!(
        code_of(db.prepare("SELECT k % 2, sum(v) AS v FROM t GROUP BY k % 2 HAVING t.v > 8", &[])),
        Some(Code::NotGrouped)
    );
}

// --- Plan depth limit -------------------------------------------------------------------

fn cte_chain(n: usize) -> String {
    let mut sql = String::from("WITH c0 AS (SELECT 0 x FROM range(1))");
    for i in 1..n {
        sql.push_str(&format!(", c{i} AS (SELECT {i} x FROM c{})", i - 1));
    }
    sql.push_str(&format!(" SELECT * FROM c{}", n - 1));
    sql
}

/// Plans deeper than the binder's limit used to overflow the 1 MiB wasm stack (an instance
/// trap) while being built or executed: a long chain of CTEs each reading the previous one,
/// or hundreds of subquery conjuncts, nest nothing in the SQL text. They are now rejected
/// with `ExpressionTooDeep` before anything recurses over them; a moderate chain still runs.
#[test]
fn deep_plans_fail_cleanly_instead_of_overflowing_the_stack() {
    // Debug builds spend far more stack per frame than the `wasm` profile.
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            let mut db = session_ab();
            let (_, rows) = run(&mut db, &cte_chain(150));
            assert_eq!(rows, vec![vec![i32v(149)]]);
            let too_deep = Some(Code::ExpressionTooDeep);
            assert_eq!(code_of(db.prepare(&cte_chain(800), &[])), too_deep);
            let conj = |c: &str, n: usize| {
                format!("SELECT count(*) FROM t WHERE {}", vec![c; n].join(" AND "))
            };
            for c in [
                "k IN (SELECT k FROM t2)",
                "EXISTS (SELECT 1 FROM t2 WHERE t2.k = t.k)",
                "k > ANY (SELECT k - 1 FROM t2)",
            ] {
                assert_eq!(code_of(db.prepare(&conj(c, 1000), &[])), too_deep, "{c}");
                let (_, rows) = run(&mut db, &conj(c, 20));
                assert_eq!(rows.len(), 1, "{c}");
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

/// `CUBE` over 8 columns is 256 grouping sets. They are bundled as a balanced UNION ALL tree,
/// so the plan stays shallow enough for the depth limit.
#[test]
fn a_full_cube_stays_within_the_plan_depth_limit() {
    let mut db = session_ab();
    let (_, rows) = run(
        &mut db,
        "SELECT count(*) FROM (SELECT k, count(*) AS c FROM t \
         GROUP BY CUBE (k, v, k + 1, v + 1, k + 2, v + 2, k + 3, v + 3))",
    );
    // duckdb: 256 sets; the empty set yields 1 row, the other 255 yield 5 rows each.
    assert_eq!(rows, vec![vec![i64v(1276)]]);
}

// --- Nullability through outer joins ------------------------------------------------------

/// A `NOT NULL` column on the NULL-padded side of an outer join is nullable in the join's
/// output. It used to keep `NOT NULL`, which `DESCRIBE` reported and which made `NOT IN`
/// pick the plain (not NULL-aware) anti-join.
#[cfg(all(feature = "ddl", feature = "dml"))]
#[test]
fn outer_join_star_columns_are_nullable() {
    let mut db = Session::new();
    for sql in [
        "CREATE TABLE r (k INTEGER NOT NULL, w INTEGER NOT NULL)",
        "INSERT INTO r VALUES (1, 2)",
        "CREATE TABLE s (k INTEGER NOT NULL)",
        "INSERT INTO s VALUES (1), (5)",
    ] {
        run(&mut db, sql);
    }
    let nullable = |db: &mut Session, sql: &str| -> Vec<bool> {
        run(db, sql).0.iter().map(|f| f.nullable).collect()
    };
    assert_eq!(nullable(&mut db, "SELECT * FROM s LEFT JOIN r ON s.k = r.k"), [false, true, true]);
    assert_eq!(nullable(&mut db, "SELECT * FROM r RIGHT JOIN s ON s.k = r.k"), [true, true, false]);
    assert_eq!(nullable(&mut db, "SELECT * FROM r FULL JOIN s ON s.k = r.k"), [true, true, true]);
    assert_eq!(nullable(&mut db, "SELECT * FROM r JOIN s ON s.k = r.k"), [false, false, false]);
    // Nested: only the padded subtree changes, whatever joins sit above or below it.
    assert_eq!(
        nullable(
            &mut db,
            "SELECT r.*, s2.* FROM s LEFT JOIN r ON s.k = r.k JOIN s s2 ON s2.k = s.k"
        ),
        [true, true, false]
    );
    assert_eq!(
        nullable(&mut db, "SELECT * FROM s JOIN s s2 ON s.k = s2.k RIGHT JOIN r ON r.k = s.k"),
        [true, true, false, false]
    );
    // `5 NOT IN (1, NULL)` is NULL, so no row qualifies (duckdb: 0 rows).
    let (_, rows) = run(
        &mut db,
        "SELECT * FROM s WHERE k NOT IN (SELECT r.* EXCLUDE (w) FROM s LEFT JOIN r ON s.k = r.k)",
    );
    assert!(rows.is_empty(), "{rows:?}");
}
