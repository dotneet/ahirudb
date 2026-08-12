//! DML 実行（`dml` フィーチャ）。
//!
//! `INSERT` / `UPDATE` / `DELETE`。効くのは `catalog::MemTable`（`ddl` の
//! `CREATE TABLE` で作ったインメモリ表）だけ。Parquet/CSV/JSONL 由来の
//! 読み取り専用テーブルに対しては `ReadOnlyTable` で拒否する
//! （DESIGN.md §16）。
//!
//! 式評価はすべて既存のバイトコード VM（`expr::vm::Vm`）を使う。行を
//! バッチ（最大 `BATCH_SIZE` 行）にまとめてから `Vm::eval`/`eval_filter` に
//! 通し、専用のスカラ評価器は書かない — 型変換・NULL・3 値論理を
//! `SELECT` と完全に同じ規則にできるうえ、コードサイズも増えない。

use crate::ddl::{count_result, eval_scalar, run_query_to_rows};
use crate::plan::compile::{cast_program, compile};
use crate::plan::Scope;
use crate::prelude::*;
use crate::rt::hash::eq_ascii_ci;
use crate::session::{Prepared, Session};
use crate::sql::ast::{ExprArena, ExprId, InsertSource};
use crate::vector::{Field, Ty, Value, Vector, BATCH_SIZE};

/// 既に確定している `Value` を対象型へキャストする。`Expr::Literal` として
/// 包んでから `eval_scalar` に通すことで、CAST の意味論（DECIMAL の
/// スケール調整、DATE/TIMESTAMP など）を `SELECT` の CAST と完全に共有する。
fn cast_value(session: &mut Session, v: Value, target_ty: Ty) -> Result<Value> {
    if v.is_null() {
        return Ok(Value::Null);
    }
    let mut arena = ExprArena::new();
    let id = arena.push(crate::sql::ast::Expr::Literal(v));
    eval_scalar(session, &arena, id, &[], target_ty)
}

/// `columns`（`INSERT INTO t (a, b) ...` の列名リスト）を、対象表のスキーマ
/// 上の添字に解決する。省略時（空リスト）は全列をスキーマ順で使う。
fn resolve_insert_columns(schema: &[Field], columns: &[String]) -> Result<Vec<usize>> {
    if columns.is_empty() {
        return Ok((0..schema.len()).collect());
    }
    let mut out = Vec::with_capacity(columns.len());
    for c in columns {
        match schema.iter().position(|f| eq_ascii_ci(f.name.as_bytes(), c.as_bytes())) {
            Some(i) => out.push(i),
            None => err!(ColumnNotFound),
        }
    }
    Ok(out)
}

/// 組み立て終えた行の全列に対して NOT NULL を検査する。
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
    let schema = session.catalog.mem_get(idx).unwrap().schema.clone();
    let col_idx = resolve_insert_columns(&schema, columns)?;

    // NOT NULL は行を組み立てた後、スキーマ全列に対してまとめて検査する。
    // `col_idx` だけを見て検査すると、`INSERT INTO t (a) ...` のように
    // 省略した列（= Value::Null のまま残る）が NOT NULL でもすり抜けてしまう。
    let new_rows: Vec<Vec<Value>> = match source {
        InsertSource::Values(value_rows) => {
            let mut out = Vec::with_capacity(value_rows.len());
            for row_exprs in value_rows {
                ensure!(row_exprs.len() == col_idx.len(), ColumnCountMismatch);
                let mut row = vec![Value::Null; schema.len()];
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
                let mut row = vec![Value::Null; schema.len()];
                for (&slot, v) in col_idx.iter().zip(r) {
                    row[slot] = cast_value(session, v, schema[slot].ty)?;
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
    let scope = Scope::from_fields(schema.clone());

    let mut set_cols = Vec::with_capacity(assignments.len());
    let mut set_progs = Vec::with_capacity(assignments.len());
    for (col, expr_id) in assignments {
        let ci = match schema.iter().position(|f| eq_ascii_ci(f.name.as_bytes(), col.as_bytes())) {
            Some(i) => i,
            None => err!(ColumnNotFound),
        };
        let prog = compile(arena, &scope, params, *expr_id)?;
        let prog =
            if prog.result_ty != schema[ci].ty { cast_program(prog, schema[ci].ty)? } else { prog };
        set_cols.push(ci);
        set_progs.push(prog);
    }
    let pred = match filter {
        Some(e) => Some(compile(arena, &scope, params, e)?),
        None => None,
    };

    let total = session.catalog.mem_get(idx).unwrap().rows.len();
    let mut updated: u64 = 0;
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

        // 各 SET 式は「元の」バッチ（更新前の値）に対して一括評価する。
        // SQL の UPDATE は同時代入なので、後続の SET が先行の SET の結果を
        // 見てはいけない。
        let mut new_cols: Vec<Vector> = Vec::with_capacity(set_progs.len());
        for p in &set_progs {
            new_cols.push(session.vm.eval(p, &batch)?);
        }

        let mt = session.catalog.mem_get_mut(idx).unwrap();
        for (local, global) in (0..end - pos).zip(pos..end) {
            if !mask[local] {
                continue;
            }
            for (k, &ci) in set_cols.iter().enumerate() {
                let v = new_cols[k].value_at(local);
                ensure!(mt.schema[ci].nullable || !v.is_null(), TypeMismatch);
                mt.rows[global][ci] = v;
            }
            updated += 1;
        }
        pos = end;
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
            let scope = Scope::from_fields(schema);
            Some(compile(arena, &scope, params, e)?)
        }
        None => None,
    };

    let total = session.catalog.mem_get(idx).unwrap().rows.len();
    let mut deleted: u64 = 0;
    // 削除は行の詰め直しを伴うので、後ろから前へバッチ単位で処理し、
    // マッチした行だけを `swap_remove` しない安定順序の `retain` 的な方法で
    // 落とす。前から進めつつ「保持する行」を別のベクタに積み直す。
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

    // CSV をフィクスチャに使うので `csv` が要る（`FormatKind::Csv` の
    // 解決は `csv` 無しだと UnsupportedFeature になる）。
    #[cfg(feature = "csv")]
    #[test]
    fn insert_into_file_table_is_read_only() {
        let mut s = Session::new();
        s.register_bytes_as("t", b"id\n1\n".to_vec(), crate::format::FormatKind::Csv).unwrap();
        let r = s.prepare("INSERT INTO t VALUES (1)", &[]);
        assert_eq!(crate::error::code_of(r), Some(Code::ReadOnlyTable));
    }

    // 列を一部だけ指定した INSERT で、省略した列の NOT NULL がすり抜けない
    // ことを確認する回帰テスト。`col_idx` だけを見て検査すると、省略列
    // （Value::Null のまま残る）が NOT NULL でも通ってしまうバグがあった。
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
        // UPDATE は同時代入。後続の SET が先行の SET の更新後の値を見てはいけない。
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (a INTEGER, b INTEGER)", &[]).unwrap();
        s.prepare("INSERT INTO t VALUES (1, 2)", &[]).unwrap();
        s.prepare("UPDATE t SET a = b, b = a", &[]).unwrap();
        let rows = ready_rows(&mut s, "SELECT a, b FROM t");
        assert_eq!(rows, vec![vec![Value::I32(2), Value::I32(1)]]);
    }

    #[test]
    fn update_rejects_setting_not_null_column_to_null() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (a INTEGER NOT NULL)", &[]).unwrap();
        s.prepare("INSERT INTO t VALUES (1)", &[]).unwrap();
        let r = s.prepare("UPDATE t SET a = NULL", &[]);
        assert_eq!(crate::error::code_of(r), Some(Code::TypeMismatch));
    }

    #[test]
    fn delete_with_null_filter_result_deletes_nothing() {
        // 3値論理: WHERE が UNKNOWN (NULL) になる行は削除対象に入らない。
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
        // BATCH_SIZE を跨ぐ削除で、保持すべき行の順序と内容が壊れないことを確認。
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
