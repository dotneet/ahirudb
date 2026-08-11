//! DDL statement parsing: `CREATE TABLE`/`CREATE VIEW`/`DROP`/`ALTER TABLE`.
//! Compiled only when the `ddl` feature is enabled.
use super::*;
use crate::sql::ast::{AlterTableAction, ColumnDef};

impl<'a> Parser<'a> {
    // --- DDL（`ddl` フィーチャ） ----------------------------------------------

    #[cfg(feature = "ddl")]
    fn if_not_exists(&mut self) -> Result<bool> {
        if self.eat_kw(Kw::If)? {
            self.expect_kw(Kw::Not)?;
            self.expect_kw(Kw::Exists)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    #[cfg(feature = "ddl")]
    fn if_exists(&mut self) -> Result<bool> {
        if self.eat_kw(Kw::If)? {
            self.expect_kw(Kw::Exists)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// `CREATE [OR REPLACE] (TABLE | VIEW) ...`
    #[cfg(feature = "ddl")]
    pub(super) fn create_stmt(&mut self, sql: &'a str) -> Result<Stmt> {
        self.bump()?; // CREATE
        let or_replace = if self.eat_kw(Kw::Or)? {
            self.expect_kw(Kw::Replace)?;
            true
        } else {
            false
        };
        if self.eat_kw(Kw::View)? {
            return self.create_view_stmt(sql, or_replace);
        }
        self.expect_kw(Kw::Table)?;
        self.create_table_stmt(or_replace)
    }

    #[cfg(feature = "ddl")]
    fn column_def(&mut self) -> Result<ColumnDef> {
        let name = self.ident()?;
        let ty = self.type_name()?;
        let nullable = if self.eat_kw(Kw::Not)? {
            self.expect_kw(Kw::Null)?;
            false
        } else {
            // 明示 `NULL` は既定と同じ意味なので読み捨てる。
            self.eat_kw(Kw::Null)?;
            true
        };
        Ok(ColumnDef { name, ty, nullable })
    }

    #[cfg(feature = "ddl")]
    fn create_table_stmt(&mut self, or_replace: bool) -> Result<Stmt> {
        let if_not_exists = self.if_not_exists()?;
        let name = self.ident()?;
        if self.eat_kw(Kw::As)? {
            let q = self.query_stmt()?;
            return Ok(Stmt::CreateTable {
                name,
                or_replace,
                if_not_exists,
                columns: Vec::new(),
                as_select: Some(Box::new(q)),
            });
        }
        self.expect(Tok::LParen)?;
        let mut columns = Vec::new();
        loop {
            columns.push(self.column_def()?);
            if !self.eat(Tok::Comma)? {
                break;
            }
        }
        self.expect(Tok::RParen)?;
        ensure!(!columns.is_empty(), UnexpectedToken, self.pos);
        Ok(Stmt::CreateTable { name, or_replace, if_not_exists, columns, as_select: None })
    }

    /// ビュー本体は AST ではなく原文の切り出しで保持する
    /// （`sql::ast::Stmt::CreateView` のドキュメント参照）。ここでは構文検証を
    /// 兼ねて一度パースし、消費したトークン範囲をそのまま文字列として拾う。
    #[cfg(feature = "ddl")]
    fn create_view_stmt(&mut self, sql: &'a str, or_replace: bool) -> Result<Stmt> {
        let name = self.ident()?;
        self.expect_kw(Kw::As)?;
        let start = self.pos;
        self.query_stmt()?;
        let end = self.pos;
        let query_sql = sql[start..end].trim().to_owned();
        Ok(Stmt::CreateView { name, or_replace, query_sql })
    }

    #[cfg(feature = "ddl")]
    pub(super) fn drop_stmt(&mut self) -> Result<Stmt> {
        self.bump()?; // DROP
        if self.eat_kw(Kw::View)? {
            let if_exists = self.if_exists()?;
            let name = self.ident()?;
            return Ok(Stmt::DropView { name, if_exists });
        }
        self.expect_kw(Kw::Table)?;
        let if_exists = self.if_exists()?;
        let name = self.ident()?;
        Ok(Stmt::DropTable { name, if_exists })
    }

    /// `ALTER TABLE t ADD [COLUMN] col ty [NOT NULL] [DEFAULT expr]` /
    /// `ALTER TABLE t DROP [COLUMN] col` /
    /// `ALTER TABLE t RENAME [COLUMN] old TO new` /
    /// `ALTER TABLE t RENAME TO new_name`。
    ///
    /// `COLUMN` キーワードは DuckDB と同じく省略可（CLI で確認済み）。
    #[cfg(feature = "ddl")]
    pub(super) fn alter_table_stmt(&mut self) -> Result<Stmt> {
        self.bump()?; // ALTER
        self.expect_kw(Kw::Table)?;
        let name = self.ident()?;
        let action = if self.eat_kw(Kw::Add)? {
            self.eat_kw(Kw::Column)?;
            let col = self.column_def()?;
            let default = if self.eat_kw(Kw::Default)? { Some(self.expr()?) } else { None };
            AlterTableAction::AddColumn {
                name: col.name,
                ty: col.ty,
                nullable: col.nullable,
                default,
            }
        } else if self.eat_kw(Kw::Drop)? {
            self.eat_kw(Kw::Column)?;
            AlterTableAction::DropColumn { name: self.ident()? }
        } else if self.eat_kw(Kw::Rename)? {
            if self.eat_kw(Kw::To)? {
                AlterTableAction::RenameTable { new_name: self.ident()? }
            } else {
                self.eat_kw(Kw::Column)?;
                let old = self.ident()?;
                self.expect_kw(Kw::To)?;
                let new = self.ident()?;
                AlterTableAction::RenameColumn { old, new }
            }
        } else {
            err!(UnexpectedToken, self.pos);
        };
        Ok(Stmt::AlterTable { name, action })
    }
}
