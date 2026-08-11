//! QA pass, continued: DDL/DML edge cases exercised through `Session`/SQL,
//! on top of the existing `crates/ahiru-core/tests/ddl_dml.rs` (not edited
//! here — see that file's `create_table_conflicts_are_rejected_unless_if_
//! not_exists` for the base `IF NOT EXISTS`/`OR REPLACE` case this file
//! deepens) and the extensive unit tests in `src/ddl.rs`/`src/dml.rs`.
//!
//! Expected values for anything not directly checkable against `duckdb`
//! (DuckDB's own DDL/DML semantics differ in places, per DESIGN.md §16) are
//! computed from the documented behavior in `src/ddl.rs`/`src/dml.rs`'s doc
//! comments.

#![cfg(feature = "dml")]

use ahiru_core::error::{code_of, Code};
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn run(sess: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    let mut q = match sess.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: mem tables never need IO"),
    };
    let mut rows = Vec::new();
    loop {
        match sess.step(&mut q).unwrap() {
            QueryStep::Batch(mut b) => {
                b.materialize();
                for r in 0..b.num_rows() {
                    rows.push(b.cols.iter().map(|c| c.value_at(r)).collect());
                }
            }
            QueryStep::Done => break,
            QueryStep::NeedIo(_) | QueryStep::NeedCodec(_) => panic!("unexpected suspend"),
        }
    }
    rows
}

fn affected(sess: &mut Session, sql: &str) -> i64 {
    let rows = run(sess, sql);
    rows[0][0].as_i64().unwrap()
}

// --- CREATE TABLE: OR REPLACE + IF NOT EXISTS combined ---------------------

#[test]
fn create_or_replace_table_if_not_exists_always_replaces() {
    // `ddl::create_table`: the `IF NOT EXISTS` short-circuit only fires when
    // `!or_replace`. With both flags present, `OR REPLACE` wins and the
    // table is always (re)created — the opposite of what `IF NOT EXISTS`
    // alone would do. (DuckDB's grammar doesn't even accept this
    // combination — verified with the `duckdb` CLI, "syntax error at or
    // near NOT" — so there's no reference behavior to match; this pins the
    // engine's own documented resolution instead.)
    let mut sess = Session::new();
    sess.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
    sess.prepare("INSERT INTO t VALUES (1), (2)", &[]).unwrap();

    sess.prepare("CREATE OR REPLACE TABLE IF NOT EXISTS t (id INTEGER, name VARCHAR)", &[])
        .unwrap();
    // Schema changed and data was cleared, proving REPLACE happened rather
    // than IF NOT EXISTS's no-op.
    assert!(run(&mut sess, "SELECT * FROM t").is_empty());
    sess.prepare("INSERT INTO t VALUES (1, 'a')", &[]).unwrap();
    assert_eq!(
        run(&mut sess, "SELECT id, name FROM t"),
        vec![vec![Value::I32(1), Value::Bytes(b"a".to_vec())]]
    );
}

#[test]
fn create_or_replace_table_as_select_replaces_schema_and_data() {
    let mut sess = Session::new();
    sess.register_bytes_as("src", b"x\n1\n2\n3\n".to_vec(), ahiru_core::format::FormatKind::Csv)
        .unwrap();
    sess.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
    sess.prepare("INSERT INTO t VALUES (99)", &[]).unwrap();

    sess.prepare("CREATE OR REPLACE TABLE t AS SELECT x FROM src WHERE x > 1", &[]).unwrap();
    let rows = run(&mut sess, "SELECT x FROM t ORDER BY x");
    assert_eq!(rows, vec![vec![Value::I64(2)], vec![Value::I64(3)]]);
}

// --- INSERT: type mismatches --------------------------------------------

#[test]
fn insert_values_with_a_value_that_cannot_cast_to_the_column_type_becomes_null() {
    // Not a hard error: `expr::kernels`'s doc on `cast`/`try_cast` says a
    // per-row conversion failure (out of range / unparsable) always becomes
    // NULL, for ordinary CAST just as much as TRY_CAST, everywhere in the
    // engine — including the implicit CAST `dml::insert`'s `eval_scalar`
    // applies to each VALUES expression. This differs from `duckdb`, which
    // raises a hard Conversion Error for the same input (verified with the
    // `duckdb` CLI): a deliberate, engine-wide behavior difference, not a
    // DML-specific bug. The column stays nullable here, so nothing rejects
    // the resulting NULL.
    let mut sess = Session::new();
    sess.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
    sess.prepare("INSERT INTO t VALUES ('not_a_number')", &[]).unwrap();
    assert_eq!(run(&mut sess, "SELECT id FROM t"), vec![vec![Value::Null]]);
}

#[test]
fn insert_values_with_an_unconvertable_value_into_a_not_null_column_is_rejected() {
    // Same CAST-failure-becomes-NULL rule as above, but the column is NOT
    // NULL, so the resulting NULL must still be caught by `dml::insert`'s
    // `check_not_null` — the cast leniency and the NOT NULL check are two
    // independent layers, and both need to be exercised together to confirm
    // neither one silently swallows the other's job.
    let mut sess = Session::new();
    sess.prepare("CREATE TABLE t (id INTEGER NOT NULL)", &[]).unwrap();
    let r = sess.prepare("INSERT INTO t VALUES ('not_a_number')", &[]);
    assert_eq!(code_of(r), Some(Code::TypeMismatch));
}

#[test]
fn insert_select_casts_source_values_to_the_destination_column_type() {
    // `dml::insert`'s `InsertSource::Query` branch runs every source value
    // through `cast_value` into the destination type. A source INTEGER
    // column feeding a destination DECIMAL column should come out scaled,
    // not just reinterpreted.
    let mut sess = Session::new();
    sess.prepare("CREATE TABLE src (n INTEGER)", &[]).unwrap();
    sess.prepare("INSERT INTO src VALUES (5)", &[]).unwrap();
    sess.prepare("CREATE TABLE dst (n DECIMAL(10,2))", &[]).unwrap();
    sess.prepare("INSERT INTO dst SELECT n FROM src", &[]).unwrap();
    let rows = run(&mut sess, "SELECT n FROM dst");
    // DECIMAL(10,2) stores scaled I64 (same convention as ddl_dml.rs's
    // `full_ddl_dml_lifecycle` test): 5 -> 500.
    assert_eq!(rows, vec![vec![Value::I64(500)]]);
}

// --- UPDATE: simultaneous-assignment semantics across 3+ columns ----------

#[test]
fn update_three_way_rotation_uses_pre_update_values_for_every_set_expression() {
    // Deepens `dml.rs`'s own `update_same_batch_sees_pre_update_values_for_
    // all_set_expressions` (a 2-column swap) with a 3-column rotation, which
    // a naive left-to-right in-place update would get wrong (b would see
    // a's *new* value instead of a's original value).
    let mut sess = Session::new();
    sess.prepare("CREATE TABLE t (a INTEGER, b INTEGER, c INTEGER)", &[]).unwrap();
    sess.prepare("INSERT INTO t VALUES (1, 2, 3)", &[]).unwrap();
    sess.prepare("UPDATE t SET a = b, b = c, c = a", &[]).unwrap();
    let rows = run(&mut sess, "SELECT a, b, c FROM t");
    assert_eq!(rows, vec![vec![Value::I32(2), Value::I32(3), Value::I32(1)]]);
}

#[test]
fn update_set_expression_referencing_the_same_column_it_assigns_still_reads_pre_update_value() {
    let mut sess = Session::new();
    sess.prepare("CREATE TABLE t (a INTEGER)", &[]).unwrap();
    sess.prepare("INSERT INTO t VALUES (10)", &[]).unwrap();
    sess.prepare("UPDATE t SET a = a * 2", &[]).unwrap();
    assert_eq!(run(&mut sess, "SELECT a FROM t"), vec![vec![Value::I32(20)]]);
}

// --- ALTER TABLE: operating on nonexistent objects -------------------------

#[test]
fn alter_table_on_a_nonexistent_table_is_table_not_found() {
    let mut sess = Session::new();
    let r = sess.prepare("ALTER TABLE nope ADD COLUMN x INTEGER", &[]);
    assert_eq!(code_of(r), Some(Code::TableNotFound));
}

#[test]
fn alter_table_rename_column_from_a_nonexistent_column_is_column_not_found() {
    let mut sess = Session::new();
    sess.prepare("CREATE TABLE t (a INTEGER)", &[]).unwrap();
    let r = sess.prepare("ALTER TABLE t RENAME COLUMN nope TO x", &[]);
    assert_eq!(code_of(r), Some(Code::ColumnNotFound));
}

#[test]
fn alter_table_drop_the_only_column_then_add_a_new_one_restores_a_usable_table() {
    // Regression-shaped check for `catalog::MemTable::batch`'s special case
    // when `schema.is_empty()` (documented in `catalog.rs`): dropping every
    // column must not corrupt the table's row count bookkeeping, and adding
    // a column back afterwards must produce a normal, queryable table.
    let mut sess = Session::new();
    sess.prepare("CREATE TABLE t (only_col INTEGER)", &[]).unwrap();
    sess.prepare("INSERT INTO t VALUES (1), (2), (3)", &[]).unwrap();
    sess.prepare("ALTER TABLE t DROP COLUMN only_col", &[]).unwrap();
    assert_eq!(
        affected(&mut sess, "SELECT count(*) FROM t"),
        3,
        "row count must survive a 0-column schema"
    );

    sess.prepare("ALTER TABLE t ADD COLUMN new_col INTEGER DEFAULT 7", &[]).unwrap();
    let rows = run(&mut sess, "SELECT new_col FROM t");
    assert_eq!(rows, vec![vec![Value::I32(7)]; 3]);
}

// --- DROP VIEW / DROP TABLE: IF EXISTS ------------------------------------

#[test]
fn drop_view_if_exists_is_a_noop_when_missing_but_a_hard_error_without_it() {
    let mut sess = Session::new();
    sess.prepare("DROP VIEW IF EXISTS nope", &[]).unwrap();
    assert_eq!(code_of(sess.prepare("DROP VIEW nope", &[])), Some(Code::TableNotFound));
}

// --- Table-name collisions across the mem/view/file namespaces ------------

#[test]
fn create_view_colliding_with_an_existing_mem_table_name_is_rejected() {
    let mut sess = Session::new();
    sess.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
    let r = sess.prepare("CREATE VIEW t AS SELECT 1 AS x FROM t", &[]);
    assert_eq!(code_of(r), Some(Code::DuplicateTable));
}

#[test]
fn create_table_colliding_with_an_existing_view_name_is_rejected() {
    let mut sess = Session::new();
    sess.prepare("CREATE TABLE base (id INTEGER)", &[]).unwrap();
    sess.prepare("CREATE VIEW v AS SELECT id FROM base", &[]).unwrap();
    let r = sess.prepare("CREATE TABLE v (id INTEGER)", &[]);
    assert_eq!(code_of(r), Some(Code::DuplicateTable));
}

// --- DELETE / UPDATE with subqueries in the filter -------------------------

#[test]
fn delete_where_filter_with_a_scalar_subquery_fails_clearly_not_a_panic_or_wrong_answer() {
    // Finding from this QA pass: `dml::delete`'s WHERE filter (and
    // `dml::update`'s SET/WHERE) compile directly through
    // `plan::compile::compile()`, which explicitly rejects
    // `Expr::ScalarSubquery`/`Exists`/`InSubquery` — that rewriting into a
    // plan node only happens in the full binder (`plan::bind::bind_query`),
    // which `SELECT` goes through but `UPDATE`/`DELETE` do not. So
    // subqueries in a DML WHERE/SET clause are unsupported here, unlike
    // DuckDB (and unlike a plain `SELECT ... WHERE id <= (SELECT ...)`,
    // which works fine — see `correlated_subqueries.rs`). This isn't listed
    // in DESIGN.md §15's limitations table, so it's a real gap this QA pass
    // surfaced; adding subquery support to DML would mean routing
    // `dml.rs`'s filter/SET compilation through the binder, which is a
    // capability addition well beyond a minimal QA-pass fix. This test
    // instead pins the current, *safe* behavior — a clear
    // `UnsupportedFeature` error, not a crash or a silently wrong row count
    // — so a future change can't regress it to something worse unnoticed.
    let mut sess = Session::new();
    sess.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
    sess.prepare("INSERT INTO t VALUES (1), (2), (3), (4)", &[]).unwrap();
    sess.prepare("CREATE TABLE cutoff (n INTEGER)", &[]).unwrap();
    sess.prepare("INSERT INTO cutoff VALUES (2)", &[]).unwrap();

    let r = sess.prepare("DELETE FROM t WHERE id <= (SELECT n FROM cutoff)", &[]);
    assert_eq!(code_of(r), Some(Code::UnsupportedFeature));
    // And the table must be untouched — a rejected statement must not have
    // partially applied.
    assert_eq!(affected(&mut sess, "SELECT count(*) FROM t"), 4);
}
