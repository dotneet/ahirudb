//! Integration tests for `WITH RECURSIVE` (recursive CTEs).
//!
//! This engine excludes `SELECT 1` (with no `FROM`) from v1's scope
//! (see `plan::bind::bind_select_in`), so a literal-only anchor goes through
//! `dual` (a dummy table equivalent to Oracle's `DUAL`, built from a single-row CSV byte
//! string). The `csv` feature is enabled by default, so this is picked up as-is by
//! `cargo test --workspace` without needing the `ddl`/`dml` features. All expected values
//! are decided by cross-checking against the actual output of `duckdb -c "..."`.

use ahiru_core::error::{code_of, Code};
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;
use ahiru_core::FormatKind;

/// A session with the `dual` table (1 row, 1 column, value unused) registered. Used only to
/// give a literal-only anchor (like `SELECT 0, 0, 1 FROM dual`) a FROM clause.
fn session_with_dual() -> Session {
    let mut s = Session::new();
    s.register_bytes_as("dual", b"x\n1\n".to_vec(), FormatKind::Csv).unwrap();
    s
}

/// A session with `nodes(id, parent_id, name)` registered as a 4-row tree structure
/// (`child1`/`child2` under `root`, `grandchild` under `child1`). Under CSV type
/// inference, `id`/`parent_id` become `BIGINT` (per `format::csv`'s integer-inference
/// rule).
fn session_with_nodes() -> Session {
    let mut s = Session::new();
    let csv = b"id,parent_id,name\n1,,root\n2,1,child1\n3,1,child2\n4,2,grandchild\n".to_vec();
    s.register_bytes_as("nodes", csv, FormatKind::Csv).unwrap();
    s
}

/// Runs `sql` and extracts the result as `Vec<Vec<Value>>`.
/// Since it only reads data that fits entirely in a byte string, `NeedIo`/`NeedCodec` never
/// occur.
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

fn i32(v: i32) -> Value {
    Value::I32(v)
}
fn i64(v: i64) -> Value {
    Value::I64(v)
}
fn s(v: &str) -> Value {
    Value::Bytes(v.as_bytes().to_vec())
}
const NULL: Value = Value::Null;

// --- Sequence generation (Fibonacci) --------------------------------------------

/// duckdb:
/// ```sql
/// WITH RECURSIVE fib(n, a, b) AS (
///     SELECT 0, 0, 1
///     UNION ALL
///     SELECT n+1, b, a+b FROM fib WHERE n < 10
/// )
/// SELECT * FROM fib;
/// ```
/// 0,0,1 / 1,1,1 / 2,1,2 / 3,2,3 / 4,3,5 / 5,5,8 / 6,8,13 / 7,13,21 /
/// 8,21,34 / 9,34,55 / 10,55,89 -- 11 rows total.
#[test]
fn fibonacci_union_all() {
    let mut db = session_with_dual();
    let rows = run(
        &mut db,
        "WITH RECURSIVE fib(n, a, b) AS ( \
           SELECT 0, 0, 1 FROM dual \
           UNION ALL \
           SELECT n+1, b, a+b FROM fib WHERE n < 10 \
         ) \
         SELECT * FROM fib",
    );
    assert_eq!(
        rows,
        vec![
            vec![i32(0), i32(0), i32(1)],
            vec![i32(1), i32(1), i32(1)],
            vec![i32(2), i32(1), i32(2)],
            vec![i32(3), i32(2), i32(3)],
            vec![i32(4), i32(3), i32(5)],
            vec![i32(5), i32(5), i32(8)],
            vec![i32(6), i32(8), i32(13)],
            vec![i32(7), i32(13), i32(21)],
            vec![i32(8), i32(21), i32(34)],
            vec![i32(9), i32(34), i32(55)],
            vec![i32(10), i32(55), i32(89)],
        ]
    );
}

/// A simple counter. The most basic form, growing by one row each time `n < 5` passes.
/// duckdb: `WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM t
/// WHERE n < 5) SELECT * FROM t` -> 1..5.
#[test]
fn simple_counter_union_all() {
    let mut db = session_with_dual();
    let rows = run(
        &mut db,
        "WITH RECURSIVE t(n) AS ( \
           SELECT 1 FROM dual UNION ALL SELECT n+1 FROM t WHERE n < 5 \
         ) SELECT * FROM t",
    );
    assert_eq!(rows, vec![vec![i32(1)], vec![i32(2)], vec![i32(3)], vec![i32(4)], vec![i32(5)]]);
}

// --- Hierarchical data (self-join) --------------------------------------------

/// `nodes` is a real table (see `session_with_nodes`), and `tree` JOINs it with itself to
/// walk from the root downward. duckdb:
/// ```sql
/// WITH RECURSIVE tree AS (
///     SELECT id, parent_id, name FROM nodes WHERE parent_id IS NULL
///     UNION ALL
///     SELECT n.id, n.parent_id, n.name FROM nodes n JOIN tree t ON n.parent_id = t.id
/// )
/// SELECT * FROM tree ORDER BY id;
/// ```
#[test]
fn hierarchy_self_join() {
    let mut db = session_with_nodes();
    let rows = run(
        &mut db,
        "WITH RECURSIVE tree AS ( \
           SELECT id, parent_id, name FROM nodes WHERE parent_id IS NULL \
           UNION ALL \
           SELECT n.id, n.parent_id, n.name FROM nodes n JOIN tree t ON n.parent_id = t.id \
         ) \
         SELECT * FROM tree ORDER BY id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i64(1), NULL, s("root")],
            vec![i64(2), i64(1), s("child1")],
            vec![i64(3), i64(1), s("child2")],
            vec![i64(4), i64(2), s("grandchild")],
        ]
    );
}

// --- UNION (dedup) --------------------------------------------------------------

/// `UNION` (without `ALL`) removes duplicates across all iterations. `n % 3 + 1` cycles
/// 1->2->3->1->2->3..., so `UNION ALL` would go on forever, but `UNION` only produces
/// already-seen rows by the 3rd iteration, reaching a fixed point and stopping.
/// duckdb: `WITH RECURSIVE t(n) AS (SELECT 1 UNION SELECT (n % 3) + 1 FROM t)
/// SELECT * FROM t ORDER BY n` -> 1,2,3.
#[test]
fn union_distinct_dedups_across_iterations_and_terminates() {
    let mut db = session_with_dual();
    let rows = run(
        &mut db,
        "WITH RECURSIVE t(n) AS ( \
           SELECT 1 FROM dual UNION SELECT (n % 3) + 1 FROM t \
         ) SELECT * FROM t ORDER BY n",
    );
    assert_eq!(rows, vec![vec![i32(1)], vec![i32(2)], vec![i32(3)]]);
}

// --- Column naming --------------------------------------------------------------

/// Under `WITH RECURSIVE`, a column-name list can also be attached to a non-recursive CTE.
#[test]
fn column_list_applies_to_non_recursive_member_too() {
    let mut db = session_with_dual();
    let rows = run(
        &mut db,
        "WITH RECURSIVE base(y) AS (SELECT 1 FROM dual), \
         t AS (SELECT y AS n FROM base UNION ALL SELECT n+1 FROM t WHERE n < 3) \
         SELECT * FROM t ORDER BY n",
    );
    assert_eq!(rows, vec![vec![i32(1)], vec![i32(2)], vec![i32(3)]]);
}

// --- Safety valve -----------------------------------------------------------------

/// A recursive CTE where forgetting the stop condition (no `WHERE`, so it grows by one row
/// forever each time) does not panic; instead, it stops with `RecursionLimitExceeded`.
#[test]
fn runaway_recursion_is_rejected_not_panicking() {
    let mut db = session_with_dual();
    let mut q = match db.prepare(
        "WITH RECURSIVE t(n) AS (SELECT 1 FROM dual UNION ALL SELECT n+1 FROM t) SELECT * FROM t",
        &[],
    ) {
        Ok(Prepared::Ready(q)) => q,
        Ok(Prepared::NeedIo(_)) => panic!("unexpected NeedIo"),
        Err(e) => {
            assert_eq!(e.code, Code::RecursionLimitExceeded);
            return;
        }
    };
    let last = loop {
        match db.step(&mut q) {
            Ok(QueryStep::Batch(_)) => {}
            Ok(QueryStep::Done) => panic!("runaway recursion must not terminate on its own"),
            Ok(QueryStep::NeedIo(_)) | Ok(QueryStep::NeedCodec(_)) => {
                panic!("unexpected NeedIo/NeedCodec")
            }
            Err(e) => break e.code,
        }
    };
    assert_eq!(last, Code::RecursionLimitExceeded);
}

/// A recursive CTE whose per-row growth is not constant but expands geometrically (10x each
/// time) should hit `Oom` from the working-set byte limit (`MAX_WORKING_BYTES`) far sooner
/// than reaching `MAX_RECURSIVE_ITERATIONS` (100,000 iterations).
/// `RecursiveCte::process` checks this limit on a per-batch basis, so also verify that a
/// join that would eventually produce an astronomical row count can be cut off partway
/// through without ever being fully materialized (the test not taking a long time is the
/// evidence of that).
#[test]
fn geometric_growth_hits_the_working_set_byte_limit_not_the_iteration_limit() {
    let mut db = session_with_dual();
    let ten = "(SELECT 0 AS k FROM dual UNION ALL SELECT 1 FROM dual UNION ALL SELECT 2 FROM dual \
               UNION ALL SELECT 3 FROM dual UNION ALL SELECT 4 FROM dual UNION ALL SELECT 5 FROM dual \
               UNION ALL SELECT 6 FROM dual UNION ALL SELECT 7 FROM dual UNION ALL SELECT 8 FROM dual \
               UNION ALL SELECT 9 FROM dual)";
    let sql = format!(
        "WITH RECURSIVE t(n) AS ( \
           SELECT 1 FROM dual \
           UNION ALL \
           SELECT t.n * 10 + m.k FROM t, {ten} AS m \
         ) SELECT count(*) FROM t"
    );
    let mut q = match db.prepare(&sql, &[]) {
        Ok(Prepared::Ready(q)) => q,
        Ok(Prepared::NeedIo(_)) => panic!("unexpected NeedIo"),
        Err(e) => {
            assert_eq!(e.code, Code::Oom);
            return;
        }
    };
    let last = loop {
        match db.step(&mut q) {
            Ok(QueryStep::Batch(_)) => {}
            Ok(QueryStep::Done) => panic!("geometric growth should hit Oom somewhere"),
            Ok(QueryStep::NeedIo(_)) | Ok(QueryStep::NeedCodec(_)) => {
                panic!("unexpected NeedIo/NeedCodec")
            }
            Err(e) => break e.code,
        }
    };
    assert_eq!(last, Code::Oom, "must stop at the byte limit before the iteration limit");
}

/// Uses two independent recursive CTEs at the same time in the same query. If the
/// `WorkingTable` swap for each doesn't correspond to the right CTE, values should get
/// crossed or it should loop forever.
#[test]
fn two_independent_recursive_ctes_in_the_same_query_do_not_cross_contaminate() {
    let mut db = session_with_dual();
    let rows = run(
        &mut db,
        "WITH RECURSIVE \
           a(n) AS (SELECT 1 FROM dual UNION ALL SELECT n+1 FROM a WHERE n < 3), \
           b(n) AS (SELECT 100 FROM dual UNION ALL SELECT n+1 FROM b WHERE n < 103) \
         SELECT a.n, b.n FROM a, b WHERE a.n = b.n - 99 ORDER BY a.n",
    );
    assert_eq!(rows, vec![vec![i32(1), i32(100)], vec![i32(2), i32(101)], vec![i32(3), i32(102)],]);
}

// --- Patterns explicitly rejected at bind time -----------------------------------

/// Even under `WITH RECURSIVE`, a CTE body that references itself is rejected unless it has
/// the form `<anchor> UNION [ALL] <recursive_term>`
/// (e.g. the self-reference is on the anchor side).
#[test]
fn self_reference_in_anchor_is_rejected() {
    let mut db = Session::new();
    let err = db.prepare(
        "WITH RECURSIVE t(n) AS (SELECT n FROM t UNION ALL SELECT 1) SELECT * FROM t",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// Rejected clearly when the column-name list's column count doesn't match the body (the
/// anchor produces only 1 column, but `t(a, b)` specifies 2).
#[test]
fn column_list_arity_mismatch_is_rejected() {
    let mut db = session_with_dual();
    let err = db.prepare(
        "WITH RECURSIVE t(a, b) AS ( \
           SELECT 1 FROM dual UNION ALL SELECT n+1 FROM t WHERE n < 3 \
         ) SELECT * FROM t",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::ColumnCountMismatch));
}

/// The anchor and the recursive term must have the same column count as
/// each other, independent of whether an explicit column list is given.
/// Previously this wasn't checked: `coerce_to` (`plan::bind`) silently
/// truncated the wider side to match the narrower one, so a typo like an
/// extra SELECT-list column in the recursive term wouldn't fail cleanly —
/// it would coerce column shapes together and, since the truncation can
/// throw off whether the recursion actually converges, potentially spin
/// until the iteration/byte safety cap instead of erroring immediately.
#[test]
fn anchor_and_recursive_term_column_count_mismatch_is_rejected() {
    let mut db = session_with_dual();
    // Anchor has 1 column, recursive term has 2 — no explicit column list,
    // so this must be caught by the anchor/recursive-term shape check.
    let err = db.prepare(
        "WITH RECURSIVE t(n) AS ( \
           SELECT 1 FROM dual UNION ALL SELECT n, n+1 FROM t WHERE n < 3 \
         ) SELECT * FROM t",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::ColumnCountMismatch));

    // And the reverse direction: anchor has 2 columns, recursive term has 1.
    let err = db.prepare(
        "WITH RECURSIVE t(n, m) AS ( \
           SELECT 1, 1 FROM dual UNION ALL SELECT n+1 FROM t WHERE n < 3 \
         ) SELECT * FROM t",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::ColumnCountMismatch));
}
