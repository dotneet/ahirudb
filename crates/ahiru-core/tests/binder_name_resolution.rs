//! Regression tests for binder name-resolution and grouping fixes:
//! intermediate sub-expressions of a flattened operator chain matching a
//! `GROUP BY` expression, the width of a negative integer literal, and the
//! other items listed per section below.
//!
//! Every expected value was measured with the `duckdb` CLI against the same
//! data (`tests/data/orders.csv`, and the small inline table `v`).

use ahiru_core::error::{code_of, Code};
use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn data(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// `o` = `tests/data/orders.csv` (order_id, customer_id, amount, status) and
/// `v(a, b, s)`, a six-row table with NULLs in every column.
fn session() -> Session {
    let mut s = Session::new();
    s.register_bytes_as("o", data("orders.csv"), FormatKind::Csv).unwrap();
    let v = b"a,b,s\n1,5,a\n2,5,c\n3,1,b\n4,1,\n,7,z\n6,,y\n".to_vec();
    s.register_bytes_as("v", v, FormatKind::Csv).unwrap();
    s
}

fn run(s: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    let mut q = match s.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
    };
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
            QueryStep::NeedIo(_) | QueryStep::NeedCodec(_) => panic!("{sql}: unexpected suspend"),
        }
    }
    rows
}

/// The first column of every row.
fn col0(s: &mut Session, sql: &str) -> Vec<Value> {
    run(s, sql).into_iter().map(|r| r[0].clone()).collect()
}

fn err(s: &mut Session, sql: &str) -> Option<Code> {
    code_of(s.prepare(sql, &[]))
}

fn i(v: i64) -> Value {
    Value::I64(v)
}

// --- A GROUP BY expression nested inside a flattened operator chain ---------

#[test]
fn a_group_by_expression_inside_an_operator_chain_is_recognized() {
    let mut s = session();
    // duckdb: 3 5 7 9 11
    let got = col0(&mut s, "SELECT customer_id*2 + 1 FROM o GROUP BY customer_id*2 ORDER BY 1");
    assert_eq!(got, vec![i(3), i(5), i(7), i(9), i(11)]);
    // duckdb: 8 14 20 26 32
    let got =
        col0(&mut s, "SELECT (customer_id*2 + 1) * 3 - 1 FROM o GROUP BY customer_id*2 ORDER BY 1");
    assert_eq!(got, vec![i(8), i(14), i(20), i(26), i(32)]);
    // duckdb: 4 5 6
    let got = col0(
        &mut s,
        "SELECT customer_id + 1 FROM o GROUP BY customer_id + 1 \
         HAVING customer_id + 1 > 3 ORDER BY 1",
    );
    assert_eq!(got, vec![i(4), i(5), i(6)]);
    // A bare column beside the grouped sub-expression is still rejected (duckdb:
    // "column customer_id must appear in the GROUP BY clause").
    let sql = "SELECT customer_id*2 + customer_id FROM o GROUP BY customer_id*2";
    assert_eq!(err(&mut s, sql), Some(Code::NotGrouped));
    let sql = "SELECT customer_id*2 + amount FROM o GROUP BY customer_id*2";
    assert_eq!(err(&mut s, sql), Some(Code::NotGrouped));
}

#[test]
fn a_long_flat_chain_over_a_grouped_expression_still_binds() {
    // The chain walk must stay iterative: 200 terms is far past the nesting cap.
    let mut s = session();
    let mut sql = String::from("SELECT customer_id*2");
    for _ in 0..200 {
        sql.push_str(" + 1");
    }
    sql.push_str(" FROM o GROUP BY customer_id*2 ORDER BY 1 LIMIT 1");
    assert_eq!(col0(&mut s, &sql), vec![i(202)]);
}

// --- Negative integer literal width ------------------------------------------

#[test]
fn a_negative_literal_whose_magnitude_overflows_integer_is_bigint() {
    // duckdb: `SELECT -2147483648 - 1, typeof(-2147483648)` -> -2147483649, BIGINT
    let mut s = session();
    assert_eq!(col0(&mut s, "SELECT -2147483648 - 1 FROM o LIMIT 1"), vec![i(-2147483649)]);
    assert_eq!(
        col0(&mut s, "SELECT -2147483647 - 1 FROM o LIMIT 1"),
        vec![Value::I32(-2147483648)]
    );
}

// --- GROUP BY ALL leaves constant items out of the grouping -------------------

#[test]
fn group_by_all_ignores_constant_select_items() {
    let mut s = session();
    let n = Value::Null;
    // duckdb: 1,42,2 / 5,42,2 / 7,42,1 / NULL,42,1
    let rows = run(&mut s, "SELECT b, 42, count(*) FROM v GROUP BY ALL ORDER BY ALL");
    let k = Value::I32(42);
    assert_eq!(
        rows,
        vec![
            vec![i(1), k.clone(), i(2)],
            vec![i(5), k.clone(), i(2)],
            vec![i(7), k.clone(), i(1)],
            vec![n.clone(), k, i(1)],
        ]
    );
    // duckdb: 3,1,2 / 3,5,2 / 3,7,1 / 3,NULL,1 (`3` is not an ordinal here)
    let rows = run(&mut s, "SELECT 3, b, count(*) FROM v GROUP BY ALL ORDER BY 2");
    let bs: Vec<Value> = rows.iter().map(|r| r[1].clone()).collect();
    assert_eq!(bs, vec![i(1), i(5), i(7), n]);
    assert!(rows.iter().all(|r| r[0] == Value::I32(3)));
    // Only constants and aggregates: one row for the whole input (duckdb: 1,6).
    let rows = run(&mut s, "SELECT 1, count(*) FROM v GROUP BY ALL");
    assert_eq!(rows, vec![vec![Value::I32(1), i(6)]]);
}

/// An all-integer result as `Option<i64>` cells (`None` = NULL), for compact expectations.
fn grid(s: &mut Session, sql: &str) -> Vec<Vec<Option<i64>>> {
    let cell = |v: &Value| match v {
        Value::Null => None,
        Value::I32(x) => Some(*x as i64),
        Value::I64(x) => Some(*x),
        Value::I128(x) => Some(*x as i64),
        other => panic!("{sql}: unexpected value {other:?}"),
    };
    run(s, sql).iter().map(|r| r.iter().map(cell).collect()).collect()
}

const N: Option<i64> = None;

fn g<const K: usize>(rows: &[[Option<i64>; K]]) -> Vec<Vec<Option<i64>>> {
    rows.iter().map(|r| r.to_vec()).collect()
}

// --- One column spelled two ways is one grouping key ---------------------------

#[test]
fn grouping_keys_compare_resolved_columns_not_spelling() {
    let mut s = session();
    let (s1, s2, s5, s6, s7, s8) = (Some(1), Some(2), Some(5), Some(6), Some(7), Some(8));
    // duckdb: 1,2 1,2 5,2 5,2 7,1 7,1 NULL,1 NULL,1 -- both sets group by b.
    let sql = "SELECT b, count(*) FROM v GROUP BY GROUPING SETS ((b), (v.b)) ORDER BY ALL";
    let want = [[s1, s2], [s1, s2], [s5, s2], [s5, s2], [s7, s1], [s7, s1], [N, s1], [N, s1]];
    assert_eq!(grid(&mut s, sql), g(&want));
    // duckdb: 2,2 2,2 6,2 6,2 8,1 8,1 NULL,1 NULL,1 NULL,6
    let sql = "SELECT b + 1 AS x, count(*) FROM v GROUP BY ROLLUP(v.b + 1, b) ORDER BY ALL";
    let want =
        [[s2, s2], [s2, s2], [s6, s2], [s6, s2], [s8, s1], [s8, s1], [N, s1], [N, s1], [N, s6]];
    assert_eq!(grid(&mut s, sql), g(&want));
    // duckdb: 1,0,2 5,0,2 7,0,1 NULL,0,1 NULL,1,6
    let sql = "SELECT b, grouping(v.b), count(*) FROM v GROUP BY ROLLUP(b) ORDER BY ALL";
    let z = Some(0);
    let want = [[s1, z, s2], [s5, z, s2], [s7, z, s1], [N, z, s1], [N, s1, s6]];
    assert_eq!(grid(&mut s, sql), g(&want));
    // The plain GROUP BY path too. duckdb: 2,2 6,2 8,1 NULL,1
    let sql = "SELECT b+1, count(*) FROM v GROUP BY v.b+1 ORDER BY ALL";
    assert_eq!(grid(&mut s, sql), g(&[[s2, s2], [s6, s2], [s8, s1], [N, s1]]));
    // duckdb: 3,4 / 6,1 (b+1 > 2 drops the b = 1 group)
    let sql = "SELECT b + 1, sum(a) FROM v GROUP BY v.b + 1 HAVING b + 1 > 2 ORDER BY b + 1";
    assert_eq!(grid(&mut s, sql), g(&[[s6, Some(3)], [s8, N]]));
}

// --- Plain keys mixed with ROLLUP/CUBE/GROUPING SETS ---------------------------

#[test]
fn plain_keys_and_rollup_combine_as_a_cross_product() {
    let mut s = session();
    let (z, s1, s2, s3, s4, s5, s6, s7) =
        (Some(0), Some(1), Some(2), Some(3), Some(4), Some(5), Some(6), Some(7));
    // duckdb: GROUP BY b, ROLLUP(a) = GROUPING SETS ((b, a), (b))
    let want = [
        [s1, s3, z, z, s1],
        [s1, s4, z, z, s1],
        [s1, N, s1, z, s2],
        [s5, s1, z, z, s1],
        [s5, s2, z, z, s1],
        [s5, N, s1, z, s2],
        [s7, N, z, z, s1],
        [s7, N, s1, z, s1],
        [N, s6, z, z, s1],
        [N, N, s1, z, s1],
    ];
    for group_by in ["b, ROLLUP(a)", "ROLLUP(a), b", "b, GROUPING SETS ((a), ())"] {
        let sql = format!(
            "SELECT b, a, grouping(a), grouping(b), count(*) FROM v GROUP BY {group_by} \
             ORDER BY ALL"
        );
        assert_eq!(grid(&mut s, &sql), g(&want), "{sql}");
    }
    // Two constructs: ROLLUP(b) x ROLLUP(a) = ((b, a), (b), (a), ()); 6 + 4 + 6 + 1 rows
    // (duckdb).
    let sql = "SELECT b, a, count(*) FROM v GROUP BY ROLLUP(b), ROLLUP(a)";
    assert_eq!(run(&mut s, sql).len(), 17);
}
