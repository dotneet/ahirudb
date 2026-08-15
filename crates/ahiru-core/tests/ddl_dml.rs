//! Integration tests for the `ddl`/`dml` features.
//!
//! Verifies the full flow of CREATE TABLE -> INSERT -> SELECT -> UPDATE -> SELECT ->
//! DELETE -> SELECT end-to-end, using only the public API (`Session::prepare`/
//! `Session::step`). Never touches read-only Parquet/CSV tables, and also verifies that
//! DDL/DML only ever affects in-memory tables (DESIGN.md §16).

#![cfg(feature = "dml")]

use ahiru_core::error::{code_of, Code};
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;
use ahiru_core::FormatKind;

/// Runs `sql` and extracts the result as `Vec<Vec<Value>>`.
/// Since DDL/DML only ever handles in-memory data, `NeedIo`/`NeedCodec` never occur
/// (if they do, it's an implementation bug).
fn run(session: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    let mut q = match session.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo (mem tables never need io)"),
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

/// Extracts the affected row count (the 1-row, 1-column result `count_result` returns).
fn affected(session: &mut Session, sql: &str) -> i64 {
    let rows = run(session, sql);
    assert_eq!(rows.len(), 1);
    rows[0][0].as_i64().unwrap()
}

#[test]
fn full_ddl_dml_lifecycle() {
    let mut s = Session::new();

    // CREATE TABLE
    s.prepare("CREATE TABLE accounts (id INTEGER, name VARCHAR, balance DECIMAL(10,2))", &[])
        .unwrap();
    assert!(s.table_names().iter().any(|n| n == "accounts"));
    assert!(run(&mut s, "SELECT * FROM accounts").is_empty());

    // INSERT
    let n = affected(
        &mut s,
        "INSERT INTO accounts VALUES (1, 'alice', 100.00), (2, 'bob', 50.00), (3, 'carol', 0.00)",
    );
    assert_eq!(n, 3);

    // SELECT
    let rows = run(&mut s, "SELECT id, name, balance FROM accounts ORDER BY id");
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0][1], Value::Bytes(b"alice".to_vec()));

    // UPDATE
    let updated = affected(&mut s, "UPDATE accounts SET balance = balance + 25.00 WHERE id <= 2");
    assert_eq!(updated, 2);
    let rows = run(&mut s, "SELECT balance FROM accounts WHERE id = 1");
    match &rows[0][0] {
        Value::I64(v) => assert_eq!(*v, 12500), // DECIMAL(10,2) is stored as I64, with the last 2 digits as scale
        other => panic!("expected decimal-as-i64, got {other:?}"),
    }

    // SELECT (after the update)
    let rows = run(&mut s, "SELECT id FROM accounts WHERE balance > 100.00 ORDER BY id");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], Value::I32(1));

    // DELETE
    let deleted = affected(&mut s, "DELETE FROM accounts WHERE balance = 0.00");
    assert_eq!(deleted, 1);

    // SELECT (after the delete)
    let rows = run(&mut s, "SELECT id FROM accounts ORDER BY id");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0][0], Value::I32(1));
    assert_eq!(rows[1][0], Value::I32(2));

    // DROP TABLE
    s.prepare("DROP TABLE accounts", &[]).unwrap();
    assert!(!s.table_names().iter().any(|n| n == "accounts"));
    assert_eq!(code_of(s.prepare("SELECT * FROM accounts", &[])), Some(Code::TableNotFound));
}

#[test]
fn create_table_as_select_then_insert_select_and_view() {
    let mut s = Session::new();
    s.register_bytes_as("src", b"id,val\n1,10\n2,20\n3,30\n".to_vec(), FormatKind::Csv).unwrap();

    // CTAS
    s.prepare("CREATE TABLE snap AS SELECT id, val FROM src WHERE val >= 20", &[]).unwrap();
    let rows = run(&mut s, "SELECT id, val FROM snap ORDER BY id");
    assert_eq!(rows.len(), 2);

    // Feed into another in-memory table via INSERT INTO ... SELECT.
    s.prepare("CREATE TABLE dst (id INTEGER, val INTEGER)", &[]).unwrap();
    let n = affected(&mut s, "INSERT INTO dst SELECT id, val FROM snap");
    assert_eq!(n, 2);
    assert_eq!(run(&mut s, "SELECT count(*) FROM dst")[0][0].as_i64(), Some(2));

    // A view always reflects the current state of the in-memory table.
    s.prepare("CREATE VIEW dst_v AS SELECT id, val FROM dst", &[]).unwrap();
    assert_eq!(run(&mut s, "SELECT count(*) FROM dst_v")[0][0].as_i64(), Some(2));
    affected(&mut s, "INSERT INTO dst VALUES (99, 999)");
    assert_eq!(run(&mut s, "SELECT count(*) FROM dst_v")[0][0].as_i64(), Some(3));

    s.prepare("DROP VIEW dst_v", &[]).unwrap();
    assert_eq!(code_of(s.prepare("SELECT * FROM dst_v", &[])), Some(Code::TableNotFound));
}

#[test]
fn dml_on_file_backed_table_is_rejected_as_read_only() {
    let mut s = Session::new();
    s.register_bytes_as("t", b"id\n1\n2\n".to_vec(), FormatKind::Csv).unwrap();

    assert_eq!(code_of(s.prepare("INSERT INTO t VALUES (3)", &[])), Some(Code::ReadOnlyTable));
    assert_eq!(code_of(s.prepare("UPDATE t SET id = 9", &[])), Some(Code::ReadOnlyTable));
    assert_eq!(code_of(s.prepare("DELETE FROM t", &[])), Some(Code::ReadOnlyTable));
    // The Parquet/CSV-side data is left untouched.
    assert_eq!(run(&mut s, "SELECT count(*) FROM t")[0][0].as_i64(), Some(2));
}

#[test]
fn create_table_conflicts_are_rejected_unless_if_not_exists() {
    let mut s = Session::new();
    s.register_bytes_as("t", b"id\n1\n".to_vec(), FormatKind::Csv).unwrap();

    // Cannot create an in-memory table with the same name as a file-backed table.
    assert_eq!(code_of(s.prepare("CREATE TABLE t (id INTEGER)", &[])), Some(Code::DuplicateTable));

    s.prepare("CREATE TABLE u (id INTEGER)", &[]).unwrap();
    assert_eq!(code_of(s.prepare("CREATE TABLE u (id INTEGER)", &[])), Some(Code::DuplicateTable));
    // With IF NOT EXISTS, it silently succeeds.
    s.prepare("CREATE TABLE IF NOT EXISTS u (id INTEGER)", &[]).unwrap();
    // With OR REPLACE, it's replaced (also verify the schema can change).
    s.prepare("CREATE OR REPLACE TABLE u (id INTEGER, name VARCHAR)", &[]).unwrap();
    assert!(run(&mut s, "SELECT * FROM u").is_empty());
}

#[test]
fn insert_rejects_null_into_not_null_column() {
    let mut s = Session::new();
    s.prepare("CREATE TABLE t (id INTEGER NOT NULL)", &[]).unwrap();
    assert_eq!(code_of(s.prepare("INSERT INTO t VALUES (NULL)", &[])), Some(Code::TypeMismatch));
}

#[test]
fn alter_table_add_column_variants() {
    let mut s = Session::new();
    s.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
    affected(&mut s, "INSERT INTO t VALUES (1), (2), (3)");

    // With a DEFAULT, all existing rows get that value.
    affected(&mut s, "ALTER TABLE t ADD COLUMN grade INTEGER DEFAULT 100");
    let rows = run(&mut s, "SELECT id, grade FROM t ORDER BY id");
    assert_eq!(
        rows,
        vec![
            vec![Value::I32(1), Value::I32(100)],
            vec![Value::I32(2), Value::I32(100)],
            vec![Value::I32(3), Value::I32(100)],
        ]
    );

    // Without a DEFAULT, existing rows become NULL. A new INSERT works as-is.
    affected(&mut s, "ALTER TABLE t ADD COLUMN note VARCHAR");
    let rows = run(&mut s, "SELECT note FROM t ORDER BY id");
    assert!(rows.iter().all(|r| r[0] == Value::Null));
    affected(&mut s, "INSERT INTO t VALUES (4, 50, 'ok')");
    let rows = run(&mut s, "SELECT note FROM t WHERE id = 4");
    assert_eq!(rows, vec![vec![Value::Bytes(b"ok".to_vec())]]);

    // Duplicate column names are rejected.
    assert_eq!(
        code_of(s.prepare("ALTER TABLE t ADD COLUMN grade VARCHAR", &[])),
        Some(Code::DuplicateColumn)
    );
}

#[test]
fn alter_table_drop_column_removes_it_from_schema_and_every_row() {
    let mut s = Session::new();
    s.prepare("CREATE TABLE t (a INTEGER, b INTEGER, c INTEGER)", &[]).unwrap();
    affected(&mut s, "INSERT INTO t VALUES (1, 2, 3), (4, 5, 6)");

    affected(&mut s, "ALTER TABLE t DROP COLUMN b");
    // The remaining columns' values shift correctly (dropping `b` leaves `a`/`c` intact).
    let rows = run(&mut s, "SELECT a, c FROM t ORDER BY a");
    assert_eq!(rows, vec![vec![Value::I32(1), Value::I32(3)], vec![Value::I32(4), Value::I32(6)]]);
    // The dropped column can no longer be referenced.
    assert_eq!(code_of(s.prepare("SELECT b FROM t", &[])), Some(Code::ColumnNotFound));
    // Trying to DROP a nonexistent column is an error.
    assert_eq!(
        code_of(s.prepare("ALTER TABLE t DROP COLUMN nope", &[])),
        Some(Code::ColumnNotFound)
    );
}

#[test]
fn alter_table_rename_column_and_table() {
    let mut s = Session::new();
    s.prepare("CREATE TABLE accounts (id INTEGER, balance INTEGER)", &[]).unwrap();
    affected(&mut s, "INSERT INTO accounts VALUES (1, 100)");

    // RENAME COLUMN: only the name changes; type and data are unchanged.
    affected(&mut s, "ALTER TABLE accounts RENAME COLUMN balance TO bal");
    assert_eq!(run(&mut s, "SELECT bal FROM accounts")[0][0], Value::I32(100));
    assert_eq!(code_of(s.prepare("SELECT balance FROM accounts", &[])), Some(Code::ColumnNotFound));
    // Cannot rename to a name already used by another column.
    assert_eq!(
        code_of(s.prepare("ALTER TABLE accounts RENAME COLUMN bal TO id", &[])),
        Some(Code::DuplicateColumn)
    );

    // RENAME TO: only the table's name changes; its position in the catalog is unchanged.
    affected(&mut s, "ALTER TABLE accounts RENAME TO ledger");
    assert!(!s.table_names().iter().any(|n| n == "accounts"));
    assert!(s.table_names().iter().any(|n| n == "ledger"));
    assert_eq!(run(&mut s, "SELECT id, bal FROM ledger")[0], vec![Value::I32(1), Value::I32(100)]);
    assert_eq!(code_of(s.prepare("SELECT * FROM accounts", &[])), Some(Code::TableNotFound));

    // Cannot rename to a table name that's already in use.
    s.prepare("CREATE TABLE other (x INTEGER)", &[]).unwrap();
    assert_eq!(
        code_of(s.prepare("ALTER TABLE ledger RENAME TO other", &[])),
        Some(Code::DuplicateTable)
    );
}

#[test]
fn alter_table_on_file_backed_table_is_rejected_as_read_only() {
    let mut s = Session::new();
    s.register_bytes_as("t", b"id\n1\n2\n".to_vec(), FormatKind::Csv).unwrap();

    assert_eq!(
        code_of(s.prepare("ALTER TABLE t ADD COLUMN x INTEGER", &[])),
        Some(Code::ReadOnlyTable)
    );
    assert_eq!(code_of(s.prepare("ALTER TABLE t DROP COLUMN id", &[])), Some(Code::ReadOnlyTable));
    assert_eq!(
        code_of(s.prepare("ALTER TABLE t RENAME COLUMN id TO x", &[])),
        Some(Code::ReadOnlyTable)
    );
    assert_eq!(code_of(s.prepare("ALTER TABLE t RENAME TO u", &[])), Some(Code::ReadOnlyTable));
    // The file-side data is left untouched.
    assert_eq!(run(&mut s, "SELECT count(*) FROM t")[0][0].as_i64(), Some(2));
}

#[test]
fn alter_table_default_value_preserved_on_insert() {
    let mut s = Session::new();
    s.prepare("CREATE TABLE t (id INT)", &[]).unwrap();
    affected(&mut s, "INSERT INTO t VALUES (1)");

    affected(&mut s, "ALTER TABLE t ADD COLUMN score INT NOT NULL DEFAULT 100");
    // Existing row should be backfilled with 100.
    assert_eq!(run(&mut s, "SELECT id, score FROM t")[0], vec![Value::I32(1), Value::I32(100)]);

    // Subsequent INSERT omitting 'score' should receive the DEFAULT (100) rather than NULL.
    affected(&mut s, "INSERT INTO t (id) VALUES (2)");
    let rows = run(&mut s, "SELECT id, score FROM t ORDER BY id");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0], vec![Value::I32(1), Value::I32(100)]);
    assert_eq!(rows[1], vec![Value::I32(2), Value::I32(100)]);
}
