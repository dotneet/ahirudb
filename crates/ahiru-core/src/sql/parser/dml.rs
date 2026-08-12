//! DML statement parsing: `INSERT`/`UPDATE`/`DELETE`.
//! Compiled only when the `dml` feature is enabled.
use super::*;
use crate::sql::ast::InsertSource;

impl<'a> Parser<'a> {
    // --- DML (the `dml` feature) ----------------------------------------------

    #[cfg(feature = "dml")]
    pub(super) fn insert_stmt(&mut self) -> Result<Stmt> {
        self.bump()?; // INSERT
        self.expect_kw(Kw::Into)?;
        let table = self.ident()?;
        let mut columns = Vec::new();
        if self.eat(Tok::LParen)? {
            loop {
                columns.push(self.ident()?);
                if !self.eat(Tok::Comma)? {
                    break;
                }
            }
            self.expect(Tok::RParen)?;
        }
        let source = if self.eat_kw(Kw::Values)? {
            let mut rows = Vec::new();
            loop {
                self.expect(Tok::LParen)?;
                let mut row = Vec::new();
                loop {
                    row.push(self.expr()?);
                    if !self.eat(Tok::Comma)? {
                        break;
                    }
                }
                self.expect(Tok::RParen)?;
                rows.push(row);
                if !self.eat(Tok::Comma)? {
                    break;
                }
            }
            InsertSource::Values(rows)
        } else {
            InsertSource::Query(Box::new(self.query_stmt()?))
        };
        Ok(Stmt::Insert { table, columns, source })
    }

    #[cfg(feature = "dml")]
    pub(super) fn update_stmt(&mut self) -> Result<Stmt> {
        self.bump()?; // UPDATE
        let table = self.ident()?;
        self.expect_kw(Kw::Set)?;
        let mut assignments = Vec::new();
        loop {
            let col = self.ident()?;
            self.expect(Tok::Eq)?;
            let e = self.expr()?;
            assignments.push((col, e));
            if !self.eat(Tok::Comma)? {
                break;
            }
        }
        let filter = if self.eat_kw(Kw::Where)? { Some(self.expr()?) } else { None };
        Ok(Stmt::Update { table, assignments, filter })
    }

    #[cfg(feature = "dml")]
    pub(super) fn delete_stmt(&mut self) -> Result<Stmt> {
        self.bump()?; // DELETE
        self.expect_kw(Kw::From)?;
        let table = self.ident()?;
        let filter = if self.eat_kw(Kw::Where)? { Some(self.expr()?) } else { None };
        Ok(Stmt::Delete { table, filter })
    }
}
