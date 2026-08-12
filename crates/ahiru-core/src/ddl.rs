//! DDL 実行（`ddl` フィーチャ）。
//!
//! `CREATE TABLE` / `CREATE TABLE AS SELECT` / `DROP TABLE` / `CREATE VIEW` /
//! `DROP VIEW`。効果はすべて `catalog::MemTable`・ビュー表（`Catalog` 内の
//! `(名前, SQL)` 表）にしか及ばない。読み取り専用の `Source`/`TableFormat`
//! には一切触れない（DESIGN.md §16）。
//!
//! `Session::prepare` から直接呼ばれる（DDL/DML は 1 発実行の文で、
//! Volcano のストリーミング実行に乗らないため）。
//!
//! ## `CREATE TABLE AS SELECT` は非再開設計
//!
//! `write::export_all` と同じ理由・同じ制約: 実行中に `NEED_IO`/`NEED_CODEC`
//! が起きたら `IoFailed` で失敗する。全データがメモリ上にある場合にしか
//! 使えない（`write` モジュールのドキュメント参照）。

use crate::error::Code;
use crate::exec::{build, ExecContext, Step};
use crate::plan::bind::{bind_query, referenced_in_query};
use crate::plan::compile::{cast_program, compile};
use crate::plan::Scope;
use crate::prelude::*;
use crate::session::{Prepared, Query, Session};
use crate::sql::ast::{AlterTableAction, ColumnDef, ExprArena, ExprId, QueryStmt};
use crate::vector::{Batch, Field, Ty, Value, Vector};

#[allow(clippy::too_many_arguments)]
pub(crate) fn create_table(
    session: &mut Session,
    arena: &ExprArena,
    name: &str,
    or_replace: bool,
    if_not_exists: bool,
    columns: &[ColumnDef],
    as_select: Option<&QueryStmt>,
    params: &[Value],
) -> Result<Prepared> {
    if if_not_exists && !or_replace && table_name_exists(session, name) {
        return Ok(Prepared::Ready(count_result(0)));
    }
    let (schema, rows) = match as_select {
        Some(q) => run_query_to_rows(session, arena, q, params)?,
        None => {
            let schema =
                columns.iter().map(|c| Field::new(c.name.clone(), c.ty, c.nullable)).collect();
            (schema, Vec::new())
        }
    };
    let n = rows.len();
    let idx = session.catalog.mem_create(name, schema, or_replace)?;
    session.catalog.mem_get_mut(idx).unwrap().rows = rows;
    Ok(Prepared::Ready(count_result(n as i64)))
}

fn table_name_exists(session: &Session, name: &str) -> bool {
    session.catalog.index_of(name).is_some()
        || session.catalog.mem_index_of(name).is_some()
        || session.catalog.view_index_of(name).is_some()
}

pub(crate) fn drop_table(session: &mut Session, name: &str, if_exists: bool) -> Result<Prepared> {
    match session.catalog.mem_drop(name) {
        Ok(()) => Ok(Prepared::Ready(count_result(0))),
        Err(e) if if_exists && e.code == Code::TableNotFound => {
            Ok(Prepared::Ready(count_result(0)))
        }
        Err(e) => Err(e),
    }
}

/// `ALTER TABLE t <action>`。効くのは `catalog::MemTable` のみ。ファイル
/// テーブルは `Catalog::mem_index_writable`（`dml::mem_index_writable` と
/// 同じ規則）が `ReadOnlyTable` で拒否する。
///
/// スキーマ・行の実際の書き換えは `catalog::Catalog` の `mem_add_column` 等に
/// 委譲する（`CREATE TABLE`/`DROP TABLE` が `mem_create`/`mem_drop` に
/// 委譲するのと同じ分担）。ここでは DEFAULT 式の評価（VM が要る）と、
/// 影響行数の組み立てだけを担当する。
pub(crate) fn alter_table(
    session: &mut Session,
    arena: &ExprArena,
    name: &str,
    action: &AlterTableAction,
    params: &[Value],
) -> Result<Prepared> {
    let idx = session.catalog.mem_index_writable(name)?;
    match action {
        AlterTableAction::AddColumn { name: col_name, ty, nullable, default } => {
            add_column(session, arena, idx, col_name, *ty, *nullable, *default, params)?;
        }
        AlterTableAction::DropColumn { name: col_name } => {
            session.catalog.mem_drop_column(idx, col_name)?;
        }
        AlterTableAction::RenameColumn { old, new } => {
            session.catalog.mem_rename_column(idx, old, new)?;
        }
        AlterTableAction::RenameTable { new_name } => {
            session.catalog.mem_rename_table(idx, new_name)?;
        }
    }
    // CREATE VIEW/DROP TABLE/DROP VIEW と同じく、スキーマだけを変える文には
    // 「影響行数」に意味が無いので常に 0 を返す。
    Ok(Prepared::Ready(count_result(0)))
}

/// `ADD COLUMN col ty [NOT NULL] [DEFAULT expr]`。`DEFAULT` は
/// `dml::insert` の値評価と同じパターンで既存のバイトコード VM を使って
/// 1 度だけ評価し（専用のスカラ評価器は書かない）、同じ値を既存の全行に
/// 積む。
///
/// **NOT NULL かつ DEFAULT 無しの扱い**: `duckdb` CLI で確認したところ、
/// DuckDB は `ADD COLUMN` への `NOT NULL` 制約そのものを未対応として
/// 一律に拒否する（`DEFAULT` を付けても同様、"Adding columns with
/// constraints not yet supported"）。このエンジンでは DEFAULT と組み合わせた
/// 場合や、既存行が 0 件で実際には NULL がどの行にも入らない場合まで
/// 一律拒否する理由が無いため、`dml::insert`/`dml::update` と同じ
/// 「NOT NULL 列に実際に NULL が入るときだけエラー」という規則に合わせる:
/// 新しい列に実際に積む値（DEFAULT があればその値、無ければ NULL）が
/// NULL で、かつ既存行が 1 行以上あれば `TypeMismatch`。
#[allow(clippy::too_many_arguments)]
fn add_column(
    session: &mut Session,
    arena: &ExprArena,
    idx: usize,
    col_name: &str,
    ty: Ty,
    nullable: bool,
    default: Option<ExprId>,
    params: &[Value],
) -> Result<()> {
    let value = match default {
        Some(expr_id) => eval_scalar(session, arena, expr_id, params, ty)?,
        None => Value::Null,
    };
    let has_rows = !session.catalog.mem_get(idx).unwrap().rows.is_empty();
    ensure!(nullable || !value.is_null() || !has_rows, TypeMismatch);
    session.catalog.mem_add_column(idx, Field::new(col_name, ty, nullable), value)
}

/// 単一の式を空スコープ（列参照なし）でコンパイルし、`target_ty` へ
/// キャストしたうえで 1 行のバッチに対して評価する。`dml`（`INSERT ...
/// VALUES` の値評価、値レベルの型変換）とも共有する。
pub(crate) fn eval_scalar(
    session: &mut Session,
    arena: &ExprArena,
    expr_id: ExprId,
    params: &[Value],
    target_ty: Ty,
) -> Result<Value> {
    let scope = Scope::new();
    let prog = compile(arena, &scope, params, expr_id)?;
    let prog = if prog.result_ty != target_ty { cast_program(prog, target_ty)? } else { prog };
    let batch = Batch::rows_only(1);
    let v = session.vm.eval(&prog, &batch)?;
    Ok(v.value_at(0))
}

pub(crate) fn create_view(
    session: &mut Session,
    name: &str,
    query_sql: String,
    or_replace: bool,
) -> Result<Prepared> {
    session.catalog.view_create(name, query_sql, or_replace)?;
    Ok(Prepared::Ready(count_result(0)))
}

pub(crate) fn drop_view(session: &mut Session, name: &str, if_exists: bool) -> Result<Prepared> {
    match session.catalog.view_drop(name) {
        Ok(()) => Ok(Prepared::Ready(count_result(0))),
        Err(e) if if_exists && e.code == Code::TableNotFound => {
            Ok(Prepared::Ready(count_result(0)))
        }
        Err(e) => Err(e),
    }
}

/// `SELECT` を非再開で最後まで実行し、結果を行列として取り出す。
/// `CREATE TABLE AS` と `INSERT INTO ... SELECT`（`dml`）の両方から使う。
///
/// **非再開**: モジュール doc 参照。スキーマ解決・スキャンの途中で
/// `NEED_IO`/`NEED_CODEC` が起きたら `IoFailed`。
pub(crate) fn run_query_to_rows(
    session: &mut Session,
    arena: &ExprArena,
    q: &QueryStmt,
    params: &[Value],
) -> Result<(Vec<Field>, Vec<Vec<Value>>)> {
    // ファイルテーブルのスキーマを先に解決する。足りなければ非再開なので
    // IoFailed（`Session::prepare` の `resolve_query` に相当する処理を
    // ここで簡略化して行う）。
    let mut tables = Vec::new();
    referenced_in_query(&session.catalog, q, &mut tables, 0)?;
    for t in tables {
        if let Some(table) = session.catalog.get_mut(t) {
            if table.resolve()?.is_err() {
                err!(IoFailed);
            }
        }
    }
    let plan = bind_query(&session.catalog, arena, q, params)?;
    let schema = plan.root.schema().to_vec();
    let mut op = build(plan.root)?;
    let mut rows = Vec::new();
    loop {
        let mut ctx = ExecContext {
            catalog: &mut session.catalog,
            vm: &mut session.vm,
            io: Vec::new(),
            codec: Vec::new(),
        };
        match op.next(&mut ctx)? {
            Step::Ready(mut b) => {
                b.materialize();
                for r in 0..b.num_rows() {
                    rows.push(b.cols.iter().map(|c| c.value_at(r)).collect());
                }
            }
            Step::NeedIo | Step::NeedCodec => err!(IoFailed),
            Step::Done => break,
        }
    }
    Ok((schema, rows))
}

/// 影響行数などを 1 行 1 列（`count`）で返す。DDL/DML の完了通知に使う。
pub(crate) fn count_result(n: i64) -> Query {
    let mut v = Vector::with_capacity(Ty::BigInt, 1);
    v.push_value(&Value::I64(n));
    Query::single_batch(vec![Field::new("count", Ty::BigInt, false)], Batch::new(vec![v]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::Prepared;

    fn ready_rows(session: &mut Session, sql: &str) -> Vec<Vec<Value>> {
        let mut q = match session.prepare(sql, &[]).unwrap() {
            Prepared::Ready(q) => q,
            Prepared::NeedIo(_) => panic!("unexpected NeedIo"),
        };
        let mut out = Vec::new();
        loop {
            match session.step(&mut q).unwrap() {
                crate::session::QueryStep::Batch(mut b) => {
                    b.materialize();
                    for r in 0..b.num_rows() {
                        out.push(b.cols.iter().map(|c| c.value_at(r)).collect());
                    }
                }
                crate::session::QueryStep::Done => break,
                _ => panic!("mem table scan should never need io/codec"),
            }
        }
        out
    }

    #[test]
    fn create_table_registers_empty_mem_table() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER, name VARCHAR)", &[]).unwrap();
        assert!(s.table_names().iter().any(|n| n == "t"));
        let rows = ready_rows(&mut s, "SELECT * FROM t");
        assert!(rows.is_empty());
    }

    // CSV をフィクスチャに使うので `csv` が要る（`FormatKind::Csv` の
    // 解決は `csv` 無しだと UnsupportedFeature になる）。
    #[cfg(feature = "csv")]
    #[test]
    fn create_table_as_select_materializes_rows() {
        let mut s = Session::new();
        // このエンジンは `SELECT 1`（FROM 無し）を扱わないので、CTAS の
        // ソースには登録済みテーブルを使う。
        s.register_bytes_as("u", b"x,y\n1,a\n2,b\n".to_vec(), crate::format::FormatKind::Csv)
            .unwrap();
        s.prepare("CREATE TABLE t AS SELECT x, y FROM u WHERE x = 1", &[]).unwrap();
        let rows = ready_rows(&mut s, "SELECT x, y FROM t");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0].as_i64(), Some(1));
    }

    #[test]
    fn drop_table_removes_it() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
        s.prepare("DROP TABLE t", &[]).unwrap();
        assert!(!s.table_names().iter().any(|n| n == "t"));
        assert_eq!(
            crate::error::code_of(s.prepare("SELECT * FROM t", &[])),
            Some(Code::TableNotFound)
        );
    }

    #[test]
    fn drop_table_if_exists_is_noop_when_missing() {
        let mut s = Session::new();
        s.prepare("DROP TABLE IF EXISTS nope", &[]).unwrap();
        assert_eq!(
            crate::error::code_of(s.prepare("DROP TABLE nope", &[])),
            Some(Code::TableNotFound)
        );
    }

    #[test]
    fn create_view_is_queryable_and_reflects_underlying_table() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (x INTEGER)", &[]).unwrap();
        s.prepare("CREATE VIEW v AS SELECT x FROM t", &[]).unwrap();
        assert!(ready_rows(&mut s, "SELECT x FROM v").is_empty());
        #[cfg(feature = "dml")]
        {
            s.prepare("INSERT INTO t VALUES (1)", &[]).unwrap();
            let rows = ready_rows(&mut s, "SELECT x FROM v");
            assert_eq!(rows, vec![vec![Value::I32(1)]]);
        }
    }

    #[test]
    fn duplicate_create_table_is_rejected() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
        assert_eq!(
            crate::error::code_of(s.prepare("CREATE TABLE t (id INTEGER)", &[])),
            Some(Code::DuplicateTable)
        );
    }

    // --- ALTER TABLE ---------------------------------------------------------

    #[test]
    #[cfg(feature = "dml")]
    fn alter_table_add_column_fills_default_then_null() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
        s.prepare("INSERT INTO t VALUES (1), (2)", &[]).unwrap();

        s.prepare("ALTER TABLE t ADD COLUMN score INTEGER DEFAULT 7", &[]).unwrap();
        assert_eq!(
            ready_rows(&mut s, "SELECT id, score FROM t ORDER BY id"),
            vec![vec![Value::I32(1), Value::I32(7)], vec![Value::I32(2), Value::I32(7)]]
        );

        // DEFAULT 無しは既存行に NULL を詰める。
        s.prepare("ALTER TABLE t ADD COLUMN note VARCHAR", &[]).unwrap();
        let rows = ready_rows(&mut s, "SELECT note FROM t");
        assert!(rows.iter().all(|r| r[0].is_null()));
    }

    #[test]
    fn alter_table_add_column_default_with_aggregate_is_rejected_not_ice() {
        // DEFAULT は `Scope::new()`（列参照なしの空スコープ）で `compile()` に
        // 直接通す設計なので、集約関数はそもそも構文的に解決できない
        // （`count`/`sum` はバインダの集約束縛経路でのみ認識される）。
        // 明確なエラーになり、内部矛盾（Internal）や panic にならないことを確認する。
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
        let r = s.prepare("ALTER TABLE t ADD COLUMN n INTEGER DEFAULT count(*)", &[]);
        assert!(crate::error::code_of(r).is_some(), "集約を含む DEFAULT は明確なエラーになるべき");
    }

    #[test]
    fn alter_table_add_column_rejects_duplicate_name() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
        assert_eq!(
            crate::error::code_of(s.prepare("ALTER TABLE t ADD COLUMN id VARCHAR", &[])),
            Some(Code::DuplicateColumn)
        );
    }

    #[test]
    #[cfg(feature = "dml")]
    fn alter_table_add_not_null_column_without_default_needs_empty_table() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
        // 行が無ければ NOT NULL を新規追加してよい（NULL がどの行にも入らない）。
        s.prepare("ALTER TABLE t ADD COLUMN score INTEGER NOT NULL", &[]).unwrap();

        s.prepare("INSERT INTO t VALUES (1, 10)", &[]).unwrap();
        // 行があるのに DEFAULT 無しの NOT NULL は、既存行が NULL になるので拒否。
        assert_eq!(
            crate::error::code_of(s.prepare("ALTER TABLE t ADD COLUMN note VARCHAR NOT NULL", &[])),
            Some(Code::TypeMismatch)
        );
    }

    #[test]
    #[cfg(feature = "dml")]
    fn alter_table_drop_column_removes_slot_from_every_row() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (a INTEGER, b INTEGER)", &[]).unwrap();
        s.prepare("INSERT INTO t VALUES (1, 10), (2, 20)", &[]).unwrap();

        s.prepare("ALTER TABLE t DROP COLUMN a", &[]).unwrap();
        assert_eq!(
            ready_rows(&mut s, "SELECT b FROM t ORDER BY b"),
            vec![vec![Value::I32(10)], vec![Value::I32(20)]]
        );
        assert_eq!(
            crate::error::code_of(s.prepare("SELECT a FROM t", &[])),
            Some(Code::ColumnNotFound)
        );
    }

    #[test]
    #[cfg(feature = "dml")]
    fn view_referencing_a_dropped_or_renamed_column_fails_cleanly_not_a_panic() {
        // ビューは生 SQL テキストとして持ち、参照されるたびに再パース・再束縛
        // する設計（`catalog::views`）。ベース表の列を DROP/RENAME した後に
        // ビューを引いたときも、束縛時に普通に ColumnNotFound で失敗する
        // べきで、panic やダングリング参照になってはいけない。
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (a INTEGER, b INTEGER)", &[]).unwrap();
        s.prepare("INSERT INTO t VALUES (1, 10)", &[]).unwrap();
        s.prepare("CREATE VIEW v AS SELECT a, b FROM t", &[]).unwrap();
        assert_eq!(
            ready_rows(&mut s, "SELECT a, b FROM v"),
            vec![vec![Value::I32(1), Value::I32(10)]]
        );

        s.prepare("ALTER TABLE t DROP COLUMN b", &[]).unwrap();
        assert_eq!(
            crate::error::code_of(s.prepare("SELECT a, b FROM v", &[])),
            Some(Code::ColumnNotFound)
        );

        s.prepare("ALTER TABLE t RENAME COLUMN a TO a2", &[]).unwrap();
        assert_eq!(
            crate::error::code_of(s.prepare("SELECT a2 FROM v", &[])),
            Some(Code::ColumnNotFound),
            "ビュー本体はまだ古い列名 `a` を参照しているので、新しい名前 `a2` 越しでは引けない"
        );
    }

    #[test]
    fn alter_table_drop_missing_column_is_column_not_found() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
        assert_eq!(
            crate::error::code_of(s.prepare("ALTER TABLE t DROP COLUMN nope", &[])),
            Some(Code::ColumnNotFound)
        );
    }

    #[test]
    #[cfg(feature = "dml")]
    fn alter_table_rename_column_keeps_data_and_type() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (a INTEGER)", &[]).unwrap();
        s.prepare("INSERT INTO t VALUES (5)", &[]).unwrap();

        s.prepare("ALTER TABLE t RENAME COLUMN a TO b", &[]).unwrap();
        assert_eq!(ready_rows(&mut s, "SELECT b FROM t"), vec![vec![Value::I32(5)]]);
        assert_eq!(
            crate::error::code_of(s.prepare("SELECT a FROM t", &[])),
            Some(Code::ColumnNotFound)
        );
    }

    #[test]
    fn alter_table_rename_column_to_existing_name_is_rejected() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (a INTEGER, b INTEGER)", &[]).unwrap();
        assert_eq!(
            crate::error::code_of(s.prepare("ALTER TABLE t RENAME COLUMN a TO b", &[])),
            Some(Code::DuplicateColumn)
        );
    }

    #[test]
    fn alter_table_rename_to_renames_without_moving_data() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
        s.prepare("ALTER TABLE t RENAME TO u", &[]).unwrap();
        assert!(!s.table_names().iter().any(|n| n == "t"));
        assert!(s.table_names().iter().any(|n| n == "u"));
        assert!(ready_rows(&mut s, "SELECT * FROM u").is_empty());
    }

    #[test]
    fn alter_table_rename_to_existing_table_is_rejected() {
        let mut s = Session::new();
        s.prepare("CREATE TABLE t (id INTEGER)", &[]).unwrap();
        s.prepare("CREATE TABLE u (id INTEGER)", &[]).unwrap();
        assert_eq!(
            crate::error::code_of(s.prepare("ALTER TABLE t RENAME TO u", &[])),
            Some(Code::DuplicateTable)
        );
    }

    // CSV をフィクスチャに使うので `csv` が要る（`FormatKind::Csv` の
    // 解決は `csv` 無しだと UnsupportedFeature になる）。
    #[cfg(feature = "csv")]
    #[test]
    fn alter_table_on_file_backed_table_is_read_only() {
        let mut s = Session::new();
        s.register_bytes_as("t", b"id\n1\n".to_vec(), crate::format::FormatKind::Csv).unwrap();
        assert_eq!(
            crate::error::code_of(s.prepare("ALTER TABLE t ADD COLUMN x INTEGER", &[])),
            Some(Code::ReadOnlyTable)
        );
        assert_eq!(
            crate::error::code_of(s.prepare("ALTER TABLE t RENAME TO u", &[])),
            Some(Code::ReadOnlyTable)
        );
    }
}
