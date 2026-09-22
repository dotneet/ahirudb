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
