//! `expr <op> ANY (subquery)` / `expr <op> ALL (subquery)` / `... SOME (subquery)`
//! quantified-comparison subqueries.
//!
//! Expected values throughout this file were checked against the actual
//! output of `duckdb -c "SELECT ..."` (DuckDB CLI installed at
//! `/opt/homebrew/bin/duckdb`), in particular:
//!   - `= ANY` / `<> ALL` are exactly equivalent to `IN` / `NOT IN`.
//!   - `>`/`<`/`>=`/`<=` combined with `ANY`/`ALL` reduce to a comparison
//!     against `MIN`/`MAX` of the subquery, *except* for the empty-set and
//!     NULL corner cases (see the module doc of
//!     `plan::bind::build_quantified_comparison` for the exact three-valued
//!     formula and its derivation).
//!   - An empty subquery makes `ANY` always `FALSE` and `ALL` always `TRUE`,
//!     regardless of the left-hand side (even if it is NULL).
//!   - `SOME` is parsed and behaves exactly like `ANY`.
//!
//! This file also documents what's *not* implemented: `= ALL (subquery)` /
//! `<> ANY (subquery)` (rejected, see the "unsupported combinations" test),
//! and `>`/`<`/`>=`/`<=` `ANY`/`ALL` over a correlated subquery (rejected;
//! only `= ANY`/`<> ALL` inherit correlated support from the existing
//! `IN`/`NOT IN` semijoin machinery).

#![cfg(feature = "dml")]

use ahiru_core::error::{code_of, Code};
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

/// Runs `sql` against in-memory tables only, so `NeedIo`/`NeedCodec` never happen.
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
fn b(v: bool) -> Value {
    Value::Bool(v)
}
const NULL: Value = Value::Null;

/// `products(id, price)`: (1,10) (2,20) (3,NULL).
/// `competitor_prices(price)`: (5) (15) (25) — non-empty, no NULLs.
/// `empty_prices(price)`: no rows.
/// `null_prices(price)`: (5) (NULL) (25) — contains a NULL.
/// `eq_prices(price)`: (10) (999) — used for `=`/`<>` equivalence checks.
///
/// Matches the tables used to gather the `duckdb` reference values in this
/// file's module doc.
fn session_with_products() -> Session {
    let mut s = Session::new();
    s.prepare("CREATE TABLE products (id INT, price INT)", &[]).unwrap();
    s.prepare("INSERT INTO products VALUES (1, 10), (2, 20), (3, NULL)", &[]).unwrap();
    s.prepare("CREATE TABLE competitor_prices (price INT)", &[]).unwrap();
    s.prepare("INSERT INTO competitor_prices VALUES (5), (15), (25)", &[]).unwrap();
    s.prepare("CREATE TABLE empty_prices (price INT)", &[]).unwrap();
    s.prepare("CREATE TABLE null_prices (price INT)", &[]).unwrap();
    s.prepare("INSERT INTO null_prices VALUES (5), (NULL), (25)", &[]).unwrap();
    s.prepare("CREATE TABLE eq_prices (price INT)", &[]).unwrap();
    s.prepare("INSERT INTO eq_prices VALUES (10), (999)", &[]).unwrap();
    s
}

// --- `=`/`<>` desugar to `IN`/`NOT IN` ---------------------------------------

/// `duckdb -c "SELECT id, price FROM products WHERE price = ANY (SELECT price FROM eq_prices)"`
/// gives the same result as the `IN` form; `<> ALL` matches `NOT IN`.
///
/// Both forms are compared as a top-level `WHERE` conjunct (not in the
/// SELECT list): like `IN (subquery)` itself, the desugared `InSubquery` node
/// is only ever rewritten into a semi-join at that position (see
/// `plan::bind::is_semijoin_predicate`/`build_semijoin`) — `plan::compile`
/// rejects a stray `InSubquery` anywhere else. `= ANY`/`<> ALL` inherit this
/// exact restriction since they desugar straight into `InSubquery`.
#[test]
fn eq_any_matches_in_and_ne_all_matches_not_in() {
    let mut db = session_with_products();
    let rows = run(
        &mut db,
        "SELECT id FROM products WHERE price = ANY (SELECT price FROM eq_prices) ORDER BY id",
    );
    let in_rows = run(
        &mut db,
        "SELECT id FROM products WHERE price IN (SELECT price FROM eq_prices) ORDER BY id",
    );
    assert_eq!(rows, vec![vec![i32(1)]]);
    assert_eq!(rows, in_rows);

    let rows = run(
        &mut db,
        "SELECT id FROM products WHERE price <> ALL (SELECT price FROM eq_prices) ORDER BY id",
    );
    let not_in_rows = run(
        &mut db,
        "SELECT id FROM products WHERE price NOT IN (SELECT price FROM eq_prices) ORDER BY id",
    );
    assert_eq!(rows, vec![vec![i32(2)]]);
    assert_eq!(rows, not_in_rows);
}

/// Unlike `>`/`<`/`>=`/`<=` `ANY`/`ALL` (which desugar into a plain boolean
/// expression usable anywhere, see `order_comparison_quantified_subquery_works_in_select_list_and_where`),
/// `= ANY`/`<> ALL` desugar into `InSubquery`, which `plan::compile` only
/// accepts once the binder has rewritten it into a semi-join — i.e. only as a
/// top-level `WHERE`/`HAVING` conjunct. Used in the SELECT list, it's
/// rejected clearly instead of silently doing the wrong thing.
#[test]
fn eq_any_in_select_list_is_rejected_like_in_subquery_itself() {
    let mut db = session_with_products();
    let err = db.prepare("SELECT price = ANY (SELECT price FROM eq_prices) FROM products", &[]);
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
    // Same restriction already applies to plain `IN (subquery)` — not a
    // regression introduced by `ANY`/`ALL`.
    let err = db.prepare("SELECT price IN (SELECT price FROM eq_prices) FROM products", &[]);
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// `SOME` is accepted as a spelling alias of `ANY` (same semantics).
#[test]
fn some_is_an_alias_of_any() {
    let mut db = session_with_products();
    let any_rows = run(
        &mut db,
        "SELECT price, price > ANY (SELECT price FROM eq_prices) FROM products ORDER BY price",
    );
    let some_rows = run(
        &mut db,
        "SELECT price, price > SOME (SELECT price FROM eq_prices) FROM products ORDER BY price",
    );
    assert_eq!(any_rows, some_rows);
    // duckdb -c "SELECT price, price > SOME (SELECT price FROM eq_prices) FROM products
    //             ORDER BY price" (NULLS LAST, matching this engine's default order).
    assert_eq!(some_rows, vec![vec![i32(10), b(false)], vec![i32(20), b(true)], vec![NULL, NULL]]);
}

// --- `>`/`<`/`>=`/`<=` via MIN/MAX ---------------------------------------------

/// The four order comparisons against a non-empty, NULL-free subquery.
/// duckdb reference values gathered in the module doc's setup query.
#[test]
fn order_comparisons_against_min_max() {
    let mut db = session_with_products();
    let rows = run(
        &mut db,
        "SELECT id, price, \
         price > ANY (SELECT price FROM competitor_prices), \
         price > ALL (SELECT price FROM competitor_prices), \
         price < ANY (SELECT price FROM competitor_prices), \
         price < ALL (SELECT price FROM competitor_prices), \
         price >= ANY (SELECT price FROM competitor_prices), \
         price <= ALL (SELECT price FROM competitor_prices) \
         FROM products ORDER BY id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i32(1), i32(10), b(true), b(false), b(true), b(false), b(true), b(false)],
            vec![i32(2), i32(20), b(true), b(false), b(true), b(false), b(true), b(false)],
            vec![i32(3), NULL, NULL, NULL, NULL, NULL, NULL, NULL],
        ]
    );
}

/// Empty subquery: `ANY` is always `FALSE`, `ALL` is always `TRUE` —
/// regardless of the left-hand side, even when it is NULL (SQL standard
/// pitfall explicitly called out in the task; verified against
/// `duckdb -c "SELECT price > ANY (SELECT price FROM empty_prices) ..."`).
#[test]
fn empty_subquery_any_is_false_all_is_true() {
    let mut db = session_with_products();
    let rows = run(
        &mut db,
        "SELECT id, price, \
         price > ANY (SELECT price FROM empty_prices), \
         price > ALL (SELECT price FROM empty_prices) \
         FROM products ORDER BY id",
    );
    assert_eq!(
        rows,
        vec![
            vec![i32(1), i32(10), b(false), b(true)],
            vec![i32(2), i32(20), b(false), b(true)],
            vec![i32(3), NULL, b(false), b(true)],
        ]
    );
}

/// A NULL inside the subquery's result: when a definite match already
/// decides the outcome the NULL is irrelevant, but when it doesn't, the
/// three-valued fold surfaces as NULL rather than FALSE/TRUE.
#[test]
fn null_inside_subquery_result_set() {
    let mut db = session_with_products();
    let rows = run(
        &mut db,
        "SELECT id, price, \
         price > ANY (SELECT price FROM null_prices), \
         price > ALL (SELECT price FROM null_prices) \
         FROM products ORDER BY id",
    );
    assert_eq!(
        rows,
        vec![
            // 10 > 5 (non-null candidate) is already true, so the NULL row
            // doesn't matter for `ANY`; for `ALL` it does (10 > 25 is false
            // anyway, so ALL is false regardless here).
            vec![i32(1), i32(10), b(true), b(false)],
            vec![i32(2), i32(20), b(true), b(false)],
            vec![i32(3), NULL, NULL, NULL],
        ]
    );

    // A case where the NULL actually flips the answer from FALSE to NULL:
    // `3 > ANY (5, NULL, 25)` — no non-null candidate is beaten, but a NULL
    // row means "maybe". duckdb: `SELECT 3 > ANY (SELECT price FROM null_prices)`.
    // (This engine has no FROM-less `SELECT <expr>`, so `FROM products WHERE
    // id = 1` stands in as a trivial single-row source.)
    let rows =
        run(&mut db, "SELECT 3 > ANY (SELECT price FROM null_prices) FROM products WHERE id = 1");
    assert_eq!(rows, vec![vec![NULL]]);
    // Symmetric ALL case: `30 > ALL (5, NULL, 25)` — every non-null
    // candidate is beaten, but the NULL row means "maybe".
    let rows =
        run(&mut db, "SELECT 30 > ALL (SELECT price FROM null_prices) FROM products WHERE id = 1");
    assert_eq!(rows, vec![vec![NULL]]);
}

/// The formula works in the SELECT list, not only in WHERE (unlike
/// `IN (subquery)`, which is restricted to a top-level WHERE/HAVING conjunct
/// because it's rewritten into a semi-join).
#[test]
fn order_comparison_quantified_subquery_works_in_select_list_and_where() {
    let mut db = session_with_products();
    let rows = run(
        &mut db,
        "SELECT id FROM products \
         WHERE price > ANY (SELECT price FROM competitor_prices) ORDER BY id",
    );
    assert_eq!(rows, vec![vec![i32(1)], vec![i32(2)]]);

    let rows = run(
        &mut db,
        "SELECT price > ALL (SELECT price FROM competitor_prices) OR price IS NULL \
         FROM products ORDER BY id",
    );
    assert_eq!(rows, vec![vec![b(false)], vec![b(false)], vec![b(true)]]);
}

// --- Unsupported combinations: rejected, not silently wrong ------------------

/// `= ALL (subquery)` / `<> ANY (subquery)` are valid DuckDB syntax but are
/// not reducible to `IN`/`NOT IN` (semi-join) nor to a `MIN`/`MAX` comparison
/// (they ask "does the whole set equal x" / "does some row differ from x",
/// which needs a per-row equality test against every row, not an order
/// statistic). Scoped out; rejected clearly rather than silently wrong.
#[test]
fn eq_all_and_ne_any_are_rejected() {
    let mut db = session_with_products();
    let err = db.prepare("SELECT price = ALL (SELECT price FROM eq_prices) FROM products", &[]);
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));

    let err = db.prepare("SELECT price <> ANY (SELECT price FROM eq_prices) FROM products", &[]);
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// `>`/`<`/`>=`/`<=` `ANY`/`ALL` over a *correlated* subquery (the subquery
/// itself references the outer row) is rejected: the `MIN`/`MAX` aggregate is
/// computed once, non-correlated, and reused for every outer row, which is
/// incompatible with a value that must be recomputed per outer row.
#[test]
fn order_comparison_over_correlated_subquery_is_rejected() {
    let mut db = session_with_products();
    db.prepare("CREATE TABLE thresholds (product_id INT, price INT)", &[]).unwrap();
    db.prepare("INSERT INTO thresholds VALUES (1, 1), (1, 100)", &[]).unwrap();
    let err = db.prepare(
        "SELECT p.id FROM products p WHERE p.price > ALL \
         (SELECT t.price FROM thresholds t WHERE t.product_id = p.id)",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}

/// `= ANY`/`<> ALL`, on the other hand, desugar straight into the existing
/// `IN`/`NOT IN` semi-join machinery, so they inherit its correlated-subquery
/// support (`= ANY`) and its correlated-`NOT IN` rejection (`<> ALL`) for
/// free — no special-casing needed in `build_quantified_comparison`.
#[test]
fn eq_any_inherits_correlated_in_support_ne_all_inherits_correlated_not_in_rejection() {
    let mut db = session_with_products();
    db.prepare("CREATE TABLE thresholds (product_id INT, price INT)", &[]).unwrap();
    db.prepare("INSERT INTO thresholds VALUES (1, 10), (2, 999)", &[]).unwrap();

    let rows = run(
        &mut db,
        "SELECT p.id FROM products p WHERE p.price = ANY \
         (SELECT t.price FROM thresholds t WHERE t.product_id = p.id) ORDER BY p.id",
    );
    assert_eq!(rows, vec![vec![i32(1)]]);

    let err = db.prepare(
        "SELECT p.id FROM products p WHERE p.price <> ALL \
         (SELECT t.price FROM thresholds t WHERE t.product_id = p.id)",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::UnsupportedFeature));
}
