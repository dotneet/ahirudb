//! DML execution (the `dml` feature).
//!
//! `INSERT` / `UPDATE` / `DELETE`. These apply only to `catalog::MemTable` (in-memory
//! tables created by `ddl`'s `CREATE TABLE`). Read-only tables backed by
//! Parquet/CSV/JSONL are rejected with `ReadOnlyTable`
//! (DESIGN.md §16).
//!
//! Every expression is evaluated with the existing bytecode VM (`expr::vm::Vm`). Rows
//! are gathered into batches (up to `BATCH_SIZE` rows) and run through
//! `Vm::eval`/`eval_filter`; no dedicated scalar evaluator is written -- that keeps type
//! conversion, NULLs, and three-valued logic exactly the same as in `SELECT`, and adds no code size.

use crate::ddl::{count_result, eval_scalar, run_query_to_rows};
use crate::plan::compile::{cast_program, compile, compile_predicate};
use crate::plan::Scope;
use crate::prelude::*;
use crate::rt::hash::eq_ascii_ci;
use crate::session::{Prepared, Session};
use crate::sql::ast::{ExprArena, ExprId, InsertSource};
use crate::vector::{Field, Ty, Value, Vector, BATCH_SIZE};

/// Casts an already-determined `Value` of type `src_ty` to `target_ty`. Wrapping it as
/// an `Expr::TypedLiteral` and running it through `eval_scalar` shares CAST semantics
/// (DECIMAL scale adjustment, DATE/TIMESTAMP, and so on) fully with `SELECT`'s CAST.
///
/// `src_ty` must be the type of the *source column* the value came from, not a type
/// guessed from the `Value` variant. A plain `Expr::Literal` would do the latter, and
/// several logical types are indistinguishable from their physical representation:
/// `DECIMAL(10,2)` 12.50 is `I64(1250)`, which re-infers as `BIGINT 1250` and then
/// rescales to `1250.00`; `UUID`/`INTERVAL`/`DATE`/`TIME`/`TIMESTAMP` likewise lose
/// their identity and become NULL or a `TypeMismatch`.
fn cast_value(session: &mut Session, v: Value, src_ty: Ty, target_ty: Ty) -> Result<Value> {
    if v.is_null() {
        return Ok(Value::Null);
    }
    // Identical types need no conversion at all -- and skipping the VM here keeps the
    // common `INSERT INTO t SELECT * FROM t` path free of per-value program compilation.
    if src_ty == target_ty {
        return Ok(v);
    }
    let mut arena = ExprArena::new();
    let id = arena.push(crate::sql::ast::Expr::TypedLiteral(v, src_ty));
    eval_scalar(session, &arena, id, &[], target_ty)
}

/// The name-resolution scope for `UPDATE`/`DELETE` against `table`.
///
/// Every column carries the target table as its qualifier, so a `SET`
/// expression or `WHERE` clause may write `t.c` as well as bare `c` -- the
/// spelling `duckdb` accepts, and the natural one when the predicate was
/// copied out of a `SELECT`. A plain `Scope::from_fields` leaves the qualifier
/// unset, which made `t.c` resolve as "column `c` of an unknown table `t`" and
/// fail with `TableNotFound`.
fn table_scope(table: &str, schema: &[Field]) -> Scope {
    let mut scope = Scope::new();
    for f in schema {
        scope.push(Some(String::from(table)), f.clone());
    }
    scope
}

/// Resolves `columns` (the column-name list of `INSERT INTO t (a, b) ...`) into indices
/// in the target table's schema. When omitted (an empty list), all columns are used in schema order.
fn resolve_insert_columns(schema: &[Field], columns: &[String]) -> Result<Vec<usize>> {
    if columns.is_empty() {
        return Ok((0..schema.len()).collect());
    }
    let mut out = Vec::with_capacity(columns.len());
    for c in columns {
        match schema.iter().position(|f| eq_ascii_ci(f.name.as_bytes(), c.as_bytes())) {
            Some(i) => {
                ensure!(!out.contains(&i), DuplicateColumn);
                out.push(i);
            }
            None => err!(ColumnNotFound),
        }
    }
    Ok(out)
}

/// Checks NOT NULL over every column of a fully assembled row.
fn check_not_null(schema: &[Field], row: &[Value]) -> Result<()> {
    for (f, v) in schema.iter().zip(row) {
        ensure!(f.nullable || !v.is_null(), TypeMismatch);
    }
    Ok(())
}

pub(crate) fn insert(
    session: &mut Session,
    arena: &ExprArena,
    table: &str,
    columns: &[String],
    source: &InsertSource,
    params: &[Value],
) -> Result<Prepared> {
    let idx = session.catalog.mem_index_writable(table)?;
    let (schema, defaults) = {
        let mt = session.catalog.mem_get(idx).unwrap();
        (mt.schema.clone(), mt.defaults.clone())
    };
    let col_idx = resolve_insert_columns(&schema, columns)?;

    // NOT NULL is checked over all schema columns at once, after the row is assembled.
    // Checking only `col_idx` would let an omitted column (left as Value::Null), as in
    // `INSERT INTO t (a) ...`, slip through even when it is NOT NULL.
    let new_rows: Vec<Vec<Value>> = match source {
        InsertSource::Values(value_rows) => {
            let mut out = Vec::with_capacity(value_rows.len());
            for row_exprs in value_rows {
                ensure!(row_exprs.len() == col_idx.len(), ColumnCountMismatch);
                let mut row = defaults.clone();
                for (&slot, &expr_id) in col_idx.iter().zip(row_exprs) {
                    row[slot] = eval_scalar(session, arena, expr_id, params, schema[slot].ty)?;
                }
                check_not_null(&schema, &row)?;
                out.push(row);
            }
            out
        }
        InsertSource::Query(q) => {
            let (src_schema, src_rows) = run_query_to_rows(session, arena, q, params)?;
            ensure!(src_schema.len() == col_idx.len(), ColumnCountMismatch);
            let mut out = Vec::with_capacity(src_rows.len());
            for r in src_rows {
                let mut row = defaults.clone();
                for ((&slot, v), sf) in col_idx.iter().zip(r).zip(&src_schema) {
                    row[slot] = cast_value(session, v, sf.ty, schema[slot].ty)?;
                }
                check_not_null(&schema, &row)?;
                out.push(row);
            }
            out
        }
    };
    let n = new_rows.len();
    session.catalog.mem_get_mut(idx).unwrap().rows.extend(new_rows);
    Ok(Prepared::Ready(count_result(n as i64)))
}

pub(crate) fn update(
    session: &mut Session,
    arena: &ExprArena,
    table: &str,
    assignments: &[(String, ExprId)],
    filter: Option<ExprId>,
    params: &[Value],
) -> Result<Prepared> {
    let idx = session.catalog.mem_index_writable(table)?;
    let schema = session.catalog.mem_get(idx).unwrap().schema.clone();
    let scope = table_scope(table, &schema);

    let mut set_cols = Vec::with_capacity(assignments.len());
    let mut set_progs = Vec::with_capacity(assignments.len());
    for (col, expr_id) in assignments {
        let ci = match schema.iter().position(|f| eq_ascii_ci(f.name.as_bytes(), col.as_bytes())) {
            Some(i) => i,
            None => err!(ColumnNotFound),
        };
        // DuckDB rejects assigning the same target column more than once. Check the
        // resolved index (rather than the spelling) so this is case-insensitive and
        // cannot be bypassed with quoted/unquoted variants of the same name.
        ensure!(!set_cols.contains(&ci), DuplicateColumn);
        let prog = compile(arena, &scope, params, *expr_id)?;
        let prog =
            if prog.result_ty != schema[ci].ty { cast_program(prog, schema[ci].ty)? } else { prog };
        set_cols.push(ci);
        set_progs.push(prog);
    }
    // `compile_predicate`, not `compile`: it is what `SELECT`'s WHERE uses, so
    // `WHERE NULL` becomes a BOOLEAN NULL (matching nothing) instead of a
    // `TypeMismatch`, and a non-boolean predicate is rejected identically.
    let pred = match filter {
        Some(e) => Some(compile_predicate(arena, &scope, params, e)?),
        None => None,
    };

    let total = session.catalog.mem_get(idx).unwrap().rows.len();

    // Pass 1 (validate): walk every batch, evaluate the SET expressions, and check NOT
    // NULL for every matched row -- but don't touch `mt.rows` yet. A statement must be
    // all-or-nothing (docs/sql/limitations.md, "UPDATE atomicity"): rows are batched in
    // groups of BATCH_SIZE, and without this split, a constraint violation discovered in
    // a later batch (e.g. row 2049 of a 2050-row UPDATE) would leave the earlier batch's
    // rows already mutated in place. Buffering the validated new values costs memory
    // proportional to the rows actually being updated (already bounded by the in-memory
    // table itself, per DESIGN.md §16), not the whole table.
    let mut planned: Vec<(usize, Vec<Value>)> = Vec::new();
    let mut pos = 0;
    while pos < total {
        let end = (pos + BATCH_SIZE).min(total);
        let batch = session.catalog.mem_get(idx).unwrap().batch(pos, end);

        let mut mask = vec![false; end - pos];
        match &pred {
            Some(p) => {
                let mut sel = Vec::new();
                session.vm.eval_filter(p, &batch, &mut sel)?;
                for i in sel {
                    mask[i as usize] = true;
                }
            }
            None => mask.iter_mut().for_each(|m| *m = true),
        }

        // Each SET expression is evaluated in bulk against the "original" batch (the
        // pre-update values). SQL's UPDATE assigns simultaneously, so a later SET must
        // not see the result of an earlier one.
        let mut new_cols: Vec<Vector> = Vec::with_capacity(set_progs.len());
        for p in &set_progs {
            new_cols.push(session.vm.eval(p, &batch)?);
        }

        for (local, &matched) in mask.iter().enumerate() {
            if !matched {
                continue;
            }
            let mut row_values = Vec::with_capacity(set_cols.len());
            for (k, &ci) in set_cols.iter().enumerate() {
                let v = new_cols[k].value_at(local);
                ensure!(schema[ci].nullable || !v.is_null(), TypeMismatch);
                row_values.push(v);
            }
            planned.push((pos + local, row_values));
        }
        pos = end;
    }

    // Pass 2 (apply): every row already validated above, so this loop does no
    // fallible work -- just moves the precomputed values into the table.
    let updated = planned.len() as u64;
    let mt = session.catalog.mem_get_mut(idx).unwrap();
    for (global, row_values) in planned {
        for (&ci, v) in set_cols.iter().zip(row_values) {
            mt.rows[global][ci] = v;
        }
    }
    Ok(Prepared::Ready(count_result(updated as i64)))
}

pub(crate) fn delete(
    session: &mut Session,
    arena: &ExprArena,
    table: &str,
    filter: Option<ExprId>,
    params: &[Value],
) -> Result<Prepared> {
    let idx = session.catalog.mem_index_writable(table)?;
    let pred = match filter {
        Some(e) => {
            let schema = session.catalog.mem_get(idx).unwrap().schema.clone();
            let scope = table_scope(table, &schema);
            // Same as `update`: see the comment on its `compile_predicate` call.
            Some(compile_predicate(arena, &scope, params, e)?)
        }
        None => None,
    };

    let total = session.catalog.mem_get(idx).unwrap().rows.len();
    let mut deleted: u64 = 0;
    // Deletion involves compacting rows, so this processes batch by batch and drops
    // matched rows in a stable-order, `retain`-like way rather than `swap_remove`.
    // It moves forward while re-accumulating "rows to keep" into a separate vector.
    let mut keep: Vec<Vec<Value>> = Vec::with_capacity(total);
    let mut pos = 0;
    while pos < total {
        let end = (pos + BATCH_SIZE).min(total);
        let batch = session.catalog.mem_get(idx).unwrap().batch(pos, end);
        let mut mask = vec![true; end - pos];
        if let Some(p) = &pred {
            let mut sel = Vec::new();
            session.vm.eval_filter(p, &batch, &mut sel)?;
            mask.iter_mut().for_each(|m| *m = false);
            for i in sel {
                mask[i as usize] = true;
            }
        }
        let mt = session.catalog.mem_get(idx).unwrap();
        for (local, global) in (0..end - pos).zip(pos..end) {
            if mask[local] {
                deleted += 1;
            } else {
                keep.push(mt.rows[global].clone());
            }
        }
        pos = end;
    }
    session.catalog.mem_get_mut(idx).unwrap().rows = keep;
    Ok(Prepared::Ready(count_result(deleted as i64)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Code;
    use crate::session::QueryStep;

    fn ready_rows(session: &mut Session, sql: &str) -> Vec<Vec<Value>> {
        let mut q = match session.prepare(sql, &[]).unwrap() {
            Prepared::Ready(q) => q,
            Prepared::NeedIo(_) => panic!("unexpected NeedIo"),
        };
        let mut out = Vec::new();
        loop {
            match session.step(&mut q).unwrap() {
                QueryStep::Batch(mut b) => {
                    b.materialize();
                    for r in 0..b.num_rows() {
                        out.push(b.cols.iter().map(|c| c.value_at(r)).collect());
                    }
                }
                QueryStep::Done => break,
                _ => panic!("mem table scan should never need io/codec"),
            }
        }
        out
    }

    #[test]
    fn insert_values_then_select() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER, name VARCHAR)", &[]).unwrap();
        s.prepare("INSERT INTO t VALUES (1, 'a'), (2, 'b')", &[]).unwrap();
        let rows = ready_rows(&mut s, "SELECT id, name FROM t ORDER BY id");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][0], Value::I32(1));
    }

    #[test]
    fn insert_rejects_wrong_column_count() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER, name VARCHAR)", &[]).unwrap();
        let r = s.prepare("INSERT INTO t VALUES (1)", &[]);
        assert_eq!(crate::error::code_of(r), Some(Code::ColumnCountMismatch));
    }

    // CSV is used as the fixture, so `csv` is required (resolving `FormatKind::Csv`
    // gives UnsupportedFeature without it).
    #[cfg(feature = "csv")]
    #[test]
    fn insert_into_file_table_is_read_only() {
        let mut s = Session::new();
        s.register_bytes_as("t", b"id\n1\n".to_vec(), crate::format::FormatKind::Csv).unwrap();
        let r = s.prepare("INSERT INTO t VALUES (1)", &[]);
        assert_eq!(crate::error::code_of(r), Some(Code::ReadOnlyTable));
    }

    // A regression test confirming that an INSERT naming only some columns does not let
    // NOT NULL on the omitted columns slip through. Checking only `col_idx` used to be a
    // bug that passed omitted columns (left as Value::Null) even when they were NOT NULL.
    #[test]
    fn insert_with_partial_column_list_enforces_not_null_on_omitted_columns() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (a INTEGER, b INTEGER NOT NULL)", &[]).unwrap();
        let r = s.prepare("INSERT INTO t (a) VALUES (1)", &[]);
        assert_eq!(crate::error::code_of(r), Some(Code::TypeMismatch));
    }

    #[test]
    fn insert_with_partial_column_list_allows_omitting_nullable_columns() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (a INTEGER NOT NULL, b INTEGER)", &[]).unwrap();
        s.prepare("INSERT INTO t (a) VALUES (1)", &[]).unwrap();
        let rows = ready_rows(&mut s, "SELECT a, b FROM t");
        assert_eq!(rows, vec![vec![Value::I32(1), Value::Null]]);
    }

    #[test]
    fn insert_select_with_partial_column_list_enforces_not_null_on_omitted_columns() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE src (x INTEGER)", &[]).unwrap();
        s.prepare("INSERT INTO src VALUES (1)", &[]).unwrap();
        s.prepare("CREATE TABLE t (a INTEGER, b INTEGER NOT NULL)", &[]).unwrap();
        let r = s.prepare("INSERT INTO t (a) SELECT x FROM src", &[]);
        assert_eq!(crate::error::code_of(r), Some(Code::TypeMismatch));
    }

    #[test]
    fn update_same_batch_sees_pre_update_values_for_all_set_expressions() {
        // UPDATE assigns simultaneously. A later SET must not see an earlier SET's updated value.
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (a INTEGER, b INTEGER)", &[]).unwrap();
        s.prepare("INSERT INTO t VALUES (1, 2)", &[]).unwrap();
        s.prepare("UPDATE t SET a = b, b = a", &[]).unwrap();
        let rows = ready_rows(&mut s, "SELECT a, b FROM t");
        assert_eq!(rows, vec![vec![Value::I32(2), Value::I32(1)]]);
    }

    #[test]
    fn update_rejects_duplicate_target_columns_case_insensitively() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (a INTEGER)", &[]).unwrap();
        s.prepare("INSERT INTO t VALUES (1)", &[]).unwrap();

        // The first assignment must not be applied before the duplicate target is
        // rejected, and changing only the target's case must not evade the check.
        let r = s.prepare("UPDATE t SET a = 2, A = 3", &[]);
        assert_eq!(crate::error::code_of(r), Some(Code::DuplicateColumn));
        assert_eq!(ready_rows(&mut s, "SELECT a FROM t"), vec![vec![Value::I32(1)]]);
    }

    #[test]
    fn update_rejects_setting_not_null_column_to_null() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (a INTEGER NOT NULL)", &[]).unwrap();
        s.prepare("INSERT INTO t VALUES (1)", &[]).unwrap();
        let r = s.prepare("UPDATE t SET a = NULL", &[]);
        assert_eq!(crate::error::code_of(r), Some(Code::TypeMismatch));
    }

    // Regression test for a bug where UPDATE applied its SET expressions batch-by-batch
    // (BATCH_SIZE = 2048 rows), mutating each batch's rows in `mem_get_mut` as it went. A
    // NOT NULL violation discovered in a later batch still errored the whole statement,
    // but the earlier batch(es) had already been written -- so a 2050-row UPDATE where row
    // 2048 (the first row of the second batch) violated NOT NULL left rows 0..2047
    // mutated despite the statement as a whole reporting failure. UPDATE must be
    // all-or-nothing: this crosses the BATCH_SIZE boundary on purpose and asserts both
    // that the statement errors AND that no row was modified.
    #[test]
    fn update_crossing_batch_boundary_is_atomic_on_constraint_violation() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER, val INTEGER NOT NULL)", &[]).unwrap();
        let n = BATCH_SIZE * 2 + 2; // 4098 rows: several full batches plus a partial one
        let values: Vec<String> = (0..n).map(|i| format!("({i}, {i})")).collect();
        s.prepare(&format!("INSERT INTO t VALUES {}", values.join(",")), &[]).unwrap();

        // Row `BATCH_SIZE` (id 2048) is the first row of the *second* batch; setting its
        // new value to NULL is what used to leave the first batch's rows mutated.
        let sql =
            format!("UPDATE t SET val = CASE WHEN id = {BATCH_SIZE} THEN NULL ELSE val + 1 END");
        let r = s.prepare(&sql, &[]);
        assert_eq!(crate::error::code_of(r), Some(Code::TypeMismatch));

        // No row should have been touched -- every `val` must still equal its original `id`.
        let rows = ready_rows(&mut s, "SELECT id, val FROM t ORDER BY id");
        assert_eq!(rows.len(), n);
        for row in rows {
            assert_eq!(row[0], row[1], "row {:?} was mutated despite the statement failing", row);
        }
    }

    // A successful UPDATE that spans multiple batches (more than BATCH_SIZE rows) should
    // still update every matching row correctly -- the validate-then-apply split
    // shouldn't change behavior for the non-error path.
    #[test]
    fn update_crossing_batch_boundary_succeeds_for_all_rows() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER, val INTEGER)", &[]).unwrap();
        let n = BATCH_SIZE * 2 + 5;
        let values: Vec<String> = (0..n).map(|i| format!("({i}, {i})")).collect();
        s.prepare(&format!("INSERT INTO t VALUES {}", values.join(",")), &[]).unwrap();

        let updated = s.prepare("UPDATE t SET val = val + 1", &[]).unwrap();
        match updated {
            Prepared::Ready(q) => {
                let mut q = q;
                match s.step(&mut q).unwrap() {
                    QueryStep::Batch(mut b) => {
                        b.materialize();
                        assert_eq!(b.cols[0].value_at(0), Value::I64(n as i64));
                    }
                    _ => panic!("expected a count result batch"),
                }
            }
            Prepared::NeedIo(_) => panic!("unexpected NeedIo"),
        }

        let rows = ready_rows(&mut s, "SELECT id, val FROM t ORDER BY id");
        assert_eq!(rows.len(), n);
        for row in rows {
            let id = match &row[0] {
                Value::I32(i) => *i as i64,
                other => panic!("unexpected id value: {other:?}"),
            };
            assert_eq!(row[1], Value::I32((id + 1) as i32));
        }
    }

    #[test]
    fn delete_with_null_filter_result_deletes_nothing() {
        // Three-valued logic: rows where WHERE is UNKNOWN (NULL) are not deleted.
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (a INTEGER)", &[]).unwrap();
        s.prepare("INSERT INTO t VALUES (1), (NULL)", &[]).unwrap();
        let deleted = s.prepare("DELETE FROM t WHERE a = NULL", &[]).unwrap();
        match deleted {
            Prepared::Ready(q) => assert_eq!(q.schema.len(), 1),
            Prepared::NeedIo(_) => panic!("unexpected NeedIo"),
        }
        let rows = ready_rows(&mut s, "SELECT count(*) FROM t");
        assert_eq!(rows, vec![vec![Value::I64(2)]]);
    }

    #[test]
    fn delete_spanning_multiple_batches_keeps_correct_rows() {
        // Confirms that a delete spanning BATCH_SIZE does not corrupt the order or contents of the rows to keep.
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (a INTEGER)", &[]).unwrap();
        let values: Vec<String> = (0..(BATCH_SIZE * 2 + 5)).map(|i| format!("({i})")).collect();
        s.prepare(&format!("INSERT INTO t VALUES {}", values.join(",")), &[]).unwrap();
        s.prepare("DELETE FROM t WHERE a % 2 = 0", &[]).unwrap();
        let rows = ready_rows(&mut s, "SELECT count(*) FROM t");
        let expected_remaining = (0..(BATCH_SIZE * 2 + 5)).filter(|i| i % 2 != 0).count();
        assert_eq!(rows, vec![vec![Value::I64(expected_remaining as i64)]]);
    }
}
