//! SQL パーサ。
//!
//! 文は再帰下降、式は Pratt（優先順位登り）で解く（DESIGN.md §7）。
//! 式は `Box` ではなくアリーナへ積むので、木の破棄で再帰が起きない。
//!
//! 再帰は必ず `MAX_DEPTH` で頭打ちにする。wasm ではスタック枯渇が
//! 回復不能なトラップになるため、深い入力は必ずエラーとして返す。

use crate::prelude::*;
use crate::rt::hash::eq_ascii_ci;
#[cfg(feature = "dml")]
use crate::sql::ast::InsertSource;
#[cfg(feature = "ddl")]
use crate::sql::ast::{AlterTableAction, ColumnDef};
use crate::sql::ast::{
    BinaryOp, Cte, Expr, ExprArena, ExprId, FromItem, JoinKind, OrderByItem, Parsed, PivotStmt,
    QueryStmt, SampleMethod, SampleSpec, SelectItem, SelectStmt, SetExpr, SetOp, Stmt, UnaryOp,
    UnpivotStmt, WindowDef, WindowFrame,
};
use crate::sql::lexer::{Kw, Lexer, Tok};
use crate::vector::{Ty, Value};

/// 再帰下降・Pratt の深さ上限。括弧の入れ子・サブクエリ・クエリの入れ子に効く。
///
/// 式はアリーナに積むので破棄で再帰しないが、`QueryStmt` は `Box` 再帰なので
/// **パースに成功した木でも破棄時に再帰する**。上限はその破棄も含めて安全な
/// 値でなければならない。
const MAX_DEPTH: u16 = 64;

/// `Box` の左深連鎖（JOIN と集合演算）の総数上限。
///
/// どちらも構文上はループで組み立てるので解析時の再帰は無いが、木の破棄では
/// 連鎖の長さぶんだけ再帰する。深さ上限とは別枠で、文全体を通した通し数として
/// 数える（入れ子のクエリごとに上限を与えると積で効いてしまうため）。
const MAX_LINKS: u16 = 64;

/// `Parser::star_modifiers` の戻り値: `(EXCLUDE する列名, REPLACE する (式, 列名))`。
type StarModifiers = (Vec<String>, Vec<(ExprId, String)>);

// 結合強度。大きいほど強く結合する。
const BP_OR: u8 = 1;
const BP_AND: u8 = 2;
const BP_NOT: u8 = 3;
const BP_CMP: u8 = 4;
const BP_CONCAT: u8 = 5;
const BP_ADD: u8 = 6;
const BP_MUL: u8 = 7;
const BP_UNARY: u8 = 8;

pub fn parse(sql: &str) -> Result<Parsed> {
    let mut p = Parser::new(sql)?;
    let stmt = p.stmt(sql)?;
    Ok(Parsed { arena: p.arena, stmt, num_params: p.num_params })
}

struct Parser<'a> {
    lex: Lexer<'a>,
    /// 先読み 1 トークン。
    cur: Tok<'a>,
    /// `cur` の入力バイト位置。エラーはすべてこれを添えて返す。
    pos: usize,
    arena: ExprArena,
    depth: u16,
    /// JOIN・集合演算で作った `Box` 連鎖の通し数。
    links: u16,
    num_params: u16,
}

impl<'a> Parser<'a> {
    fn new(sql: &'a str) -> Result<Self> {
        let mut lex = Lexer::new(sql);
        let t = lex.next_token()?;
        Ok(Parser {
            lex,
            cur: t.tok,
            pos: t.pos,
            arena: ExprArena::new(),
            depth: 0,
            links: 0,
            num_params: 0,
        })
    }

    /// `Box` 連鎖を 1 本伸ばす前に呼ぶ。上限を超えたらエラー。
    fn link(&mut self) -> Result<()> {
        self.links += 1;
        ensure!(self.links <= MAX_LINKS, ExpressionTooDeep, self.pos);
        Ok(())
    }

    // --- トークン操作 -------------------------------------------------------

    fn bump(&mut self) -> Result<()> {
        let t = self.lex.next_token()?;
        self.cur = t.tok;
        self.pos = t.pos;
        Ok(())
    }

    #[inline]
    fn is(&self, t: Tok<'a>) -> bool {
        self.cur == t
    }

    /// 2 トークン目を先読みする。字句器を複製して 1 個進めるだけで、
    /// `self` の位置は動かさない。`OVER (` の判定にだけ使う。
    fn peek(&self) -> Result<Tok<'a>> {
        let mut lx = self.lex.clone();
        Ok(lx.next_token()?.tok)
    }

    /// 文脈依存キーワードの照合。引用符付き識別子（`"over"`）は対象外で、
    /// 明示的に引用すれば常に列名として扱える。
    #[inline]
    fn is_soft_kw(&self, word: &[u8]) -> bool {
        matches!(self.cur, Tok::Ident(s) if eq_ascii_ci(s.as_bytes(), word))
    }

    /// 2 トークン目が文脈依存キーワードに一致するか（`GROUPING SETS` の
    /// `SETS` 判定用。`is_soft_kw` の 2 トークン先読み版）。
    #[inline]
    fn peek_is_soft_kw(&self, word: &[u8]) -> Result<bool> {
        Ok(matches!(self.peek()?, Tok::Ident(s) if eq_ascii_ci(s.as_bytes(), word)))
    }

    fn eat(&mut self, t: Tok<'a>) -> Result<bool> {
        if self.cur == t {
            self.bump()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn expect(&mut self, t: Tok<'a>) -> Result<()> {
        ensure!(self.cur == t, UnexpectedToken, self.pos);
        self.bump()
    }

    fn eat_kw(&mut self, k: Kw) -> Result<bool> {
        self.eat(Tok::Kw(k))
    }

    fn expect_kw(&mut self, k: Kw) -> Result<()> {
        self.expect(Tok::Kw(k))
    }

    /// 識別子を 1 個取り出す。
    ///
    /// 引用符なし識別子は原文の綴りのまま保持する。比較は束縛時に大小無視で
    /// 行う（`rt::hash::hash_ascii_ci`）ので、結果の列名は入力の見た目を保てる。
    fn ident(&mut self) -> Result<String> {
        let s = match self.cur {
            Tok::Ident(s) => s.to_owned(),
            Tok::QIdent(s) => unquote(s, b'"'),
            _ => err!(UnexpectedToken, self.pos),
        };
        self.bump()?;
        Ok(s)
    }

    /// `[AS] alias`。予約語は `Tok::Kw` なので、句の切れ目を別名と誤認しない。
    fn opt_alias(&mut self) -> Result<Option<String>> {
        if self.eat_kw(Kw::As)? {
            return Ok(Some(self.ident()?));
        }
        match self.cur {
            // `AS` 無しの裸の別名は、それが実は `USING SAMPLE`/`TABLESAMPLE`
            // 節の先頭だった場合に食ってしまうと `opt_sample_clause` へ
            // 二度と辿り着けない（`FROM t USING SAMPLE 10%` のような、ごく
            // ふつうに書かれる形が壊れる）。`SAMPLE`/`QUALIFY` と同じ「事故に
            // なりやすい文脈」なので、この 2 語だけはここで先読みして除外する
            // （引用すれば `AS "using"` 等で常に別名として使える）。
            Tok::Ident(s) if self.is_using_sample_or_tablesample(s.as_bytes())? => Ok(None),
            Tok::Ident(_) | Tok::QIdent(_) => Ok(Some(self.ident()?)),
            _ => Ok(None),
        }
    }

    /// `self.cur` が `USING SAMPLE`/`TABLESAMPLE` 節の先頭かどうか。
    fn is_using_sample_or_tablesample(&self, word: &[u8]) -> Result<bool> {
        if eq_ascii_ci(word, b"tablesample") {
            return Ok(true);
        }
        if eq_ascii_ci(word, b"using") {
            return self.peek_is_soft_kw(b"sample");
        }
        Ok(false)
    }

    /// 非負整数。LIMIT / OFFSET と DECIMAL の精度指定で使う。
    fn uint(&mut self) -> Result<u64> {
        let pos = self.pos;
        let text = match self.cur {
            Tok::Int(s) => s,
            _ => err!(UnexpectedToken, pos),
        };
        let mut v: u64 = 0;
        for &d in text.as_bytes() {
            v = match v.checked_mul(10).and_then(|x| x.checked_add((d - b'0') as u64)) {
                Some(x) => x,
                None => err!(NumberOverflow, pos),
            };
        }
        self.bump()?;
        Ok(v)
    }

    /// 符号付き整数リテラル 1 個。`generate_series`/`range` の引数と
    /// `USING SAMPLE ... (method, seed)` のシードで使う。`prefix()` の単項
    /// マイナス処理と同じく、負号は数値リテラルへ畳んでから範囲を見る
    /// （`int_literal` は `i32`/`i64`/`i128` の順で収まる型を返す）。
    fn signed_int_lit(&mut self) -> Result<i64> {
        let pos = self.pos;
        let neg = self.eat(Tok::Minus)?;
        let text = match self.cur {
            Tok::Int(s) => s,
            _ => err!(UnexpectedToken, pos),
        };
        let v = int_literal(text, neg, pos)?;
        self.bump()?;
        match v {
            Value::I32(x) => Ok(x as i64),
            Value::I64(x) => Ok(x),
            _ => err!(NumberOverflow, pos),
        }
    }

    // --- 文 -----------------------------------------------------------------

    /// `sql` は入力全体の原文。`ddl` の `CREATE VIEW` がビュー本体を生テキストの
    /// まま保持するために使う（`create_view_stmt` 参照）。フィーチャが OFF の
    /// ときは使わないが、シグネチャを cfg で分けるとこの関数だけ 2 通り持つ
    /// ことになり複雑なので、常に受け取って未使用警告だけ潰す。
    fn stmt(&mut self, sql: &'a str) -> Result<Stmt> {
        let _ = sql;
        let s = match self.cur {
            // クエリは SELECT / WITH / 括弧のいずれかで始まる。
            Tok::Kw(Kw::Select | Kw::With) | Tok::LParen => {
                Stmt::Select(Box::new(self.query_stmt()?))
            }
            Tok::Kw(Kw::Explain) => {
                self.bump()?;
                Stmt::Explain(Box::new(self.query_stmt()?))
            }
            Tok::Kw(Kw::Describe) => {
                self.bump()?;
                Stmt::Describe(self.parse_from_item()?)
            }
            Tok::Kw(Kw::Show) => {
                self.bump()?;
                self.expect_kw(Kw::Tables)?;
                Stmt::ShowTables
            }
            #[cfg(feature = "ddl")]
            Tok::Kw(Kw::Create) => self.create_stmt(sql)?,
            #[cfg(feature = "ddl")]
            Tok::Kw(Kw::Drop) => self.drop_stmt()?,
            #[cfg(feature = "ddl")]
            Tok::Kw(Kw::Alter) => self.alter_table_stmt()?,
            #[cfg(feature = "dml")]
            Tok::Kw(Kw::Insert) => self.insert_stmt()?,
            #[cfg(feature = "dml")]
            Tok::Kw(Kw::Update) => self.update_stmt()?,
            #[cfg(feature = "dml")]
            Tok::Kw(Kw::Delete) => self.delete_stmt()?,
            #[cfg(feature = "export")]
            Tok::Kw(Kw::Copy) => self.copy_stmt()?,
            // `PIVOT`/`UNPIVOT` は予約語にせず、文の先頭というこの文脈でだけ
            // キーワード扱いする（`ROWS`/`RANGE`/`QUALIFY` を巡る列名破壊
            // 事故と同じ理由 — 有効な文は必ずここに列挙したキーワードで
            // 始まるので、先頭が裸の識別子 "pivot"/"unpivot" になるのは
            // この 2 構文しかなく、`FROM pivot` のような通常の列/表名としての
            // 利用は一切妨げない）。
            _ if self.is_soft_kw(b"pivot") => self.pivot_stmt()?,
            _ if self.is_soft_kw(b"unpivot") => self.unpivot_stmt()?,
            // v1 は上記のみ。その他の未対応の文はここで一括で弾く。
            _ => err!(UnsupportedFeature, self.pos),
        };
        self.eat(Tok::Semi)?;
        ensure!(self.is(Tok::Eof), UnexpectedToken, self.pos);
        Ok(s)
    }

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
    fn create_stmt(&mut self, sql: &'a str) -> Result<Stmt> {
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
    fn drop_stmt(&mut self) -> Result<Stmt> {
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
    fn alter_table_stmt(&mut self) -> Result<Stmt> {
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

    // --- DML（`dml` フィーチャ） ----------------------------------------------

    #[cfg(feature = "dml")]
    fn insert_stmt(&mut self) -> Result<Stmt> {
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
    fn update_stmt(&mut self) -> Result<Stmt> {
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
    fn delete_stmt(&mut self) -> Result<Stmt> {
        self.bump()?; // DELETE
        self.expect_kw(Kw::From)?;
        let table = self.ident()?;
        let filter = if self.eat_kw(Kw::Where)? { Some(self.expr()?) } else { None };
        Ok(Stmt::Delete { table, filter })
    }

    // --- COPY（`export` フィーチャ）-------------------------------------------

    /// `COPY (<query>) TO '<path>' [(FORMAT csv|jsonl|json)]` /
    /// `COPY <table> TO '<path>' [...]`
    ///
    /// `TO`/`FORMAT` は `export` 単体では文脈依存キーワードとして綴りで
    /// 照合する（`is_soft_kw`）。`export` のグローバルな予約語表には入れない
    /// — さもないと `to`/`format` という名前の列（出発/到着の `to`、ファイル
    /// フォーマットを表す `format` 列など、実データにありふれた名前）が
    /// 引用符無しでは参照できなくなってしまう。`OVER`/`RECURSIVE` と同じ
    /// 理由（`sql/lexer.rs` の `KEYWORDS` コメント参照）。`COPY` 自体は
    /// 文の先頭専用の統語頭語なので `CREATE`/`DROP` と同じく普通に予約する。
    ///
    /// `ddl` フィーチャが同時に有効だと `TO` は `ALTER TABLE ... RENAME TO`
    /// 用に別途グローバル予約されている（`DDL_KEYWORDS`）ので、その場合は
    /// `Tok::Kw(Kw::To)` として来る。`expect_to` で両方を受け付ける。
    #[cfg(feature = "export")]
    fn copy_stmt(&mut self) -> Result<Stmt> {
        self.bump()?; // COPY
        let query = if self.eat(Tok::LParen)? {
            let q = self.query_stmt()?;
            self.expect(Tok::RParen)?;
            Box::new(q)
        } else {
            // `COPY <table> TO ...`。`SELECT * FROM <table>` と等価な木を
            // 組み立てて、以降はサブクエリ形と同じ経路（`write::copy`）に
            // 流す（`base_rel` の派生表化と同じ発想）。
            let name = self.ident()?;
            let star = self.arena.push(Expr::Star {
                qualifier: None,
                exclude: Vec::new(),
                replace: Vec::new(),
            });
            let mut s = SelectStmt::empty();
            s.items.push(SelectItem { expr: star, alias: None });
            s.from = Some(FromItem::Table { name, alias: None });
            Box::new(QueryStmt {
                ctes: Vec::new(),
                body: SetExpr::Select(Box::new(s)),
                order_by: Vec::new(),
                limit: None,
                offset: None,
            })
        };
        self.expect_to()?;
        let path = match self.cur {
            Tok::Str(s) => unquote(s, b'\''),
            _ => err!(UnexpectedToken, self.pos),
        };
        self.bump()?;
        let format = if self.eat(Tok::LParen)? {
            ensure!(self.is_soft_kw(b"format"), UnexpectedToken, self.pos);
            self.bump()?;
            let f = match self.cur {
                Tok::Ident(s) => s.to_owned(),
                Tok::Str(s) => unquote(s, b'\''),
                _ => err!(UnexpectedToken, self.pos),
            };
            self.bump()?;
            self.expect(Tok::RParen)?;
            Some(f)
        } else {
            None
        };
        Ok(Stmt::Copy { query, path, format })
    }

    /// `COPY` の `TO`。`ddl` フィーチャの有無で `to` の字句が変わる
    /// （上の `copy_stmt` ドキュメント参照）ので両方を受け付ける。
    #[cfg(feature = "export")]
    fn expect_to(&mut self) -> Result<()> {
        #[cfg(feature = "ddl")]
        if self.eat_kw(Kw::To)? {
            return Ok(());
        }
        ensure!(self.is_soft_kw(b"to"), UnexpectedToken, self.pos);
        self.bump()
    }

    // --- クエリ（CTE + 集合演算 + 外側の ORDER BY / LIMIT）--------------------

    /// 深さを 1 段消費するクエリ。派生表・サブクエリ式経由で再帰しうる。
    ///
    /// クエリの入れ子はすべてここを通るので、深さの計上はこの 1 か所で足りる
    /// （`select_body` は必ず `query_body` の下から呼ばれる）。
    fn query_stmt(&mut self) -> Result<QueryStmt> {
        ensure!(self.depth < MAX_DEPTH, ExpressionTooDeep, self.pos);
        self.depth += 1;
        let r = self.query_body();
        self.depth -= 1;
        r
    }

    fn query_body(&mut self) -> Result<QueryStmt> {
        let ctes = if self.eat_kw(Kw::With)? {
            let recursive = self.eat_recursive_kw()?;
            self.cte_list(recursive)?
        } else {
            Vec::new()
        };
        let (body, bare) = self.set_expr()?;

        let mut order_by = Vec::new();
        if self.eat_kw(Kw::Order)? {
            self.expect_kw(Kw::By)?;
            loop {
                let it = self.order_item()?;
                order_by.push(it);
                if !self.eat(Tok::Comma)? {
                    break;
                }
            }
        }
        let limit = if self.eat_kw(Kw::Limit)? { Some(self.uint()?) } else { None };
        let offset = if self.eat_kw(Kw::Offset)? { Some(self.uint()?) } else { None };

        let mut q = QueryStmt { ctes, body, order_by: Vec::new(), limit: None, offset: None };
        // 末尾の ORDER BY / LIMIT / OFFSET の置き場所は 1 つの規則で決める:
        // **本体が括弧無しの単一 SELECT なら `SelectStmt` 側**、それ以外
        // （集合演算がある、または本体が括弧付きクエリ）なら `QueryStmt` 側。
        // 括弧付きを除くのは、内側で既に自分の ORDER BY を持ちうるため。
        match (&mut q.body, bare) {
            (SetExpr::Select(s), true) => {
                s.order_by = order_by;
                s.limit = limit;
                s.offset = offset;
            }
            _ => {
                q.order_by = order_by;
                q.limit = limit;
                q.offset = offset;
            }
        }
        Ok(q)
    }

    /// `WITH` の直後だけで意味を持つ文脈依存キーワード `RECURSIVE`。
    ///
    /// `OVER`/`ROWS`/`RANGE` と同じ理由でグローバルには予約しない
    /// （`recursive` という名前の CTE も書けなければならない）。`RECURSIVE`
    /// の後には必ず CTE 名（識別子）が続くので、次のトークンが識別子で
    /// なければ「`recursive` という名前の CTE」の側だと判断して消費しない
    /// （`WITH recursive AS (...)`／`WITH recursive(a) AS (...)` はどちらも
    /// 妥当な SQL で、DuckDB の挙動と一致させてある）。
    fn eat_recursive_kw(&mut self) -> Result<bool> {
        if self.is_soft_kw(b"recursive") && matches!(self.peek()?, Tok::Ident(_) | Tok::QIdent(_)) {
            self.bump()?;
            return Ok(true);
        }
        Ok(false)
    }

    /// `WITH` の本体。`name [(col, ...)] AS ( query )` のカンマ区切り。
    ///
    /// `recursive` は `WITH RECURSIVE` かどうか（`eat_recursive_kw` の結果）。
    /// 標準 SQL 通り、このフラグはリスト中の CTE 全部に等しく効く
    /// （実際に自分自身を参照するかどうかは束縛時に個別に判定する）。
    fn cte_list(&mut self, recursive: bool) -> Result<Vec<Cte>> {
        let mut out = Vec::new();
        loop {
            let name = self.ident()?;
            // `name (a, b) AS (...)` の列名リストは `WITH RECURSIVE` の下でのみ
            // 許す（非再帰 CTE の列名リストは引き続き未対応のまま）。
            let mut columns = Vec::new();
            if self.is(Tok::LParen) {
                ensure!(recursive, UnsupportedFeature, self.pos);
                self.bump()?;
                loop {
                    columns.push(self.ident()?);
                    if !self.eat(Tok::Comma)? {
                        break;
                    }
                }
                self.expect(Tok::RParen)?;
            }
            self.expect_kw(Kw::As)?;
            self.expect(Tok::LParen)?;
            let query = self.query_stmt()?;
            self.expect(Tok::RParen)?;
            out.push(Cte { name, columns, recursive, query: Box::new(query) });
            if !self.eat(Tok::Comma)? {
                break;
            }
        }
        Ok(out)
    }

    /// `UNION` / `EXCEPT` の段。左結合で積む。
    ///
    /// 戻り値の `bool` は「括弧無しの単一 SELECT だったか」。末尾の
    /// ORDER BY / LIMIT を `SelectStmt` へ落として良いかの判定に使う。
    fn set_expr(&mut self) -> Result<(SetExpr, bool)> {
        let (mut left, mut bare) = self.intersect_expr()?;
        loop {
            let op = match self.cur {
                Tok::Kw(Kw::Union) => SetOp::Union,
                Tok::Kw(Kw::Except) => SetOp::Except,
                _ => break,
            };
            self.bump()?;
            let all = self.eat_kw(Kw::All)?;
            self.link()?;
            let (right, _) = self.intersect_expr()?;
            left = SetExpr::SetOp { op, all, left: Box::new(left), right: Box::new(right) };
            bare = false;
        }
        Ok((left, bare))
    }

    /// `INTERSECT` の段。SQL 標準どおり `UNION` / `EXCEPT` より強く結合する。
    fn intersect_expr(&mut self) -> Result<(SetExpr, bool)> {
        let (mut left, mut bare) = self.select_or_paren()?;
        while self.is(Tok::Kw(Kw::Intersect)) {
            self.bump()?;
            let all = self.eat_kw(Kw::All)?;
            self.link()?;
            let (right, _) = self.select_or_paren()?;
            left = SetExpr::SetOp {
                op: SetOp::Intersect,
                all,
                left: Box::new(left),
                right: Box::new(right),
            };
            bare = false;
        }
        Ok((left, bare))
    }

    fn select_or_paren(&mut self) -> Result<(SetExpr, bool)> {
        if self.is(Tok::LParen) {
            self.bump()?;
            let q = self.query_stmt()?;
            self.expect(Tok::RParen)?;
            let body = self.paren_body(q);
            return Ok((body, false));
        }
        Ok((SetExpr::Select(Box::new(self.select_body()?)), true))
    }

    /// 括弧付きクエリを集合演算の項に落とす。
    ///
    /// `SetExpr` には CTE も ORDER BY / LIMIT も置けないので、それらを持つ
    /// 場合だけ `SELECT * FROM (...)` に包む。持たない場合は本体をそのまま使う
    /// （余計な射影を挟まない）。
    fn paren_body(&mut self, q: QueryStmt) -> SetExpr {
        if q.ctes.is_empty() && q.order_by.is_empty() && q.limit.is_none() && q.offset.is_none() {
            return q.body;
        }
        let star = self.arena.push(Expr::Star {
            qualifier: None,
            exclude: Vec::new(),
            replace: Vec::new(),
        });
        let mut s = SelectStmt::empty();
        s.items.push(SelectItem { expr: star, alias: None });
        s.from = Some(FromItem::Subquery { query: Box::new(q), alias: None });
        SetExpr::Select(Box::new(s))
    }

    /// 先読み: `(` の直後がクエリの始まりか。
    ///
    /// スカラサブクエリ / `IN (SELECT ...)` / 括弧付き式の切り分けはこの 1 語
    /// だけで行う。`(` が続く場合（`((SELECT 1))`）は式・値リスト側に倒し、
    /// 内側の `(` が改めてこの判定を受ける。
    #[inline]
    fn starts_query(&self) -> bool {
        matches!(self.cur, Tok::Kw(Kw::Select | Kw::With))
    }

    fn select_body(&mut self) -> Result<SelectStmt> {
        // `PIVOT`/`UNPIVOT` はこのエンジンでは文の先頭（`stmt`）でしか認識
        // しない糖衣構文で、展開（`plan::bind::desugar_pivot`/
        // `desugar_unpivot`）は `Session::prepare` の入り口で対象表の
        // スキーマを解決したうえで一度だけ行う設計になっている
        // （`session.rs` の該当コメント参照）。そのため `FROM (PIVOT ...)`
        // のような派生表・CTE本体・集合演算の項としては使えない
        // （`duckdb` はこれを許すが、ここでは対応範囲外）。素通しすると
        // この先で `SELECT` を期待して素の `UnexpectedToken` になり
        // 原因が分かりにくいので、ここで先に分かりやすい
        // `UnsupportedFeature` にしておく。
        ensure!(
            !self.is_soft_kw(b"pivot") && !self.is_soft_kw(b"unpivot"),
            UnsupportedFeature,
            self.pos
        );
        self.expect_kw(Kw::Select)?;
        let mut st = SelectStmt::empty();
        st.distinct = self.eat_kw(Kw::Distinct)?;
        // `DISTINCT ON (expr, ...)`。PostgreSQL/DuckDB 拡張。`ON` は JOIN の
        // `ON` と同じ予約語なので、列名として使われる心配なく普通に照合できる。
        if st.distinct && self.is(Tok::Kw(Kw::On)) {
            self.bump()?; // ON
            self.expect(Tok::LParen)?;
            loop {
                let e = self.expr()?;
                st.distinct_on.push(e);
                if !self.eat(Tok::Comma)? {
                    break;
                }
            }
            self.expect(Tok::RParen)?;
            // ON 側が実質の重複除去を行うので、グループキー全体を使う
            // 通常の DISTINCT フラグは同時には立てない。
            st.distinct = false;
        }
        loop {
            let item = self.select_item()?;
            st.items.push(item);
            if !self.eat(Tok::Comma)? {
                break;
            }
        }
        if self.eat_kw(Kw::From)? {
            st.from = Some(self.parse_from_item()?);
            st.sample = self.opt_tablesample_clause()?;
        }
        if self.eat_kw(Kw::Where)? {
            st.filter = Some(self.expr()?);
        }
        if self.eat_kw(Kw::Group)? {
            self.expect_kw(Kw::By)?;
            // `GROUPING SETS`/`ROLLUP`/`CUBE` は列名としても使える一般語
            // （`ROWS`/`RANGE`/`QUALIFY` を巡る事故と同種）なので予約語にせず、
            // `GROUP BY` 直後というこの文脈でだけキーワードとして扱う。
            // 2 語連続の `GROUPING SETS` は 2 トークン先読みで見分ける。
            if self.is_soft_kw(b"grouping") && self.peek_is_soft_kw(b"sets")? {
                self.bump()?; // grouping
                self.bump()?; // sets
                st.grouping_sets = Some(self.grouping_sets_body()?);
            } else if self.is_soft_kw(b"rollup") && self.peek()? == Tok::LParen {
                self.bump()?; // rollup
                let cols = self.paren_expr_list()?;
                st.grouping_sets = Some(rollup_sets(cols));
            } else if self.is_soft_kw(b"cube") && self.peek()? == Tok::LParen {
                self.bump()?; // cube
                let cols = self.paren_expr_list()?;
                st.grouping_sets = Some(cube_sets(cols, self.pos)?);
            } else {
                loop {
                    let e = self.expr()?;
                    st.group_by.push(e);
                    if !self.eat(Tok::Comma)? {
                        break;
                    }
                }
            }
        }
        if self.eat_kw(Kw::Having)? {
            st.having = Some(self.expr()?);
        }
        // `WINDOW name AS (...), ...`。`GROUP BY`/`ORDER BY` と同格の句
        // キーワードとして扱う（`Kw::Window` は通常の予約語。`window` を
        // 一般語として文脈依存にできない理由は `sql::lexer` のコメント参照）。
        if self.eat_kw(Kw::Window)? {
            loop {
                let wname = self.ident()?;
                let pos = self.pos;
                ensure!(
                    !st.windows.iter().any(|(n, _): &(String, WindowDef)| eq_ascii_ci(
                        n.as_bytes(),
                        wname.as_bytes()
                    )),
                    SyntaxError,
                    pos
                );
                self.expect_kw(Kw::As)?;
                self.expect(Tok::LParen)?;
                let def = self.window_def_body()?;
                st.windows.push((wname, def));
                if !self.eat(Tok::Comma)? {
                    break;
                }
            }
        }
        // `QUALIFY` は `OVER`/`PARTITION`/`ROWS`/`RANGE` と違い、実データの
        // 列名になり得る一般語ではない（Teradata/SQL:1999 由来の専用語）ので
        // 通常の予約語にする。文脈依存キーワードにすると、`FROM t QUALIFY ...`
        // のように `WHERE`/`GROUP BY` を挟まず直後に置いた場合、`opt_alias`
        // が `QUALIFY` をテーブル別名として食ってしまい構文が壊れる
        // （ここは `ON` を挟まず `LIKE` を予約語にしているのと同じ判断）。
        if self.eat_kw(Kw::Qualify)? {
            st.qualify = Some(self.expr()?);
        }
        // `USING SAMPLE` は文末側の独立した句（`opt_using_sample_clause` の
        // doc 参照）。`FROM` 項目に直接くっつく `TABLESAMPLE` とは受理位置が
        // 違うので、両方が同時に書かれた場合は二重指定として拒否する
        // （`duckdb` は両方を順番に適用できるが、このエンジンの `SampleSpec`
        // は 1 個しか持てない単純化なので、サポート範囲外として明示的に
        // 拒否する）。
        if let Some(spec) = self.opt_using_sample_clause()? {
            ensure!(st.sample.is_none(), UnsupportedFeature, self.pos);
            st.sample = Some(spec);
        }
        // ORDER BY / LIMIT / OFFSET はここでは読まない。集合演算の右項が
        // 外側の ORDER BY を食ってしまうため、`query_body` 側で一括して扱う。
        Ok(st)
    }

    /// `GROUPING SETS` の本体 `( (expr, ...), (expr, ...), () )`。
    /// 空集合 `()` も 1 セットとして許す。
    fn grouping_sets_body(&mut self) -> Result<Vec<Vec<ExprId>>> {
        self.expect(Tok::LParen)?;
        let mut sets = Vec::new();
        loop {
            sets.push(self.paren_expr_list()?);
            if !self.eat(Tok::Comma)? {
                break;
            }
        }
        self.expect(Tok::RParen)?;
        Ok(sets)
    }

    /// `( expr, expr, ... )`。空リスト `()` も許す（`ROLLUP ()` 等は書けないが
    /// `GROUPING SETS` の 1 要素としては現れる）。
    fn paren_expr_list(&mut self) -> Result<Vec<ExprId>> {
        self.expect(Tok::LParen)?;
        let mut list = Vec::new();
        if !self.is(Tok::RParen) {
            loop {
                list.push(self.expr()?);
                if !self.eat(Tok::Comma)? {
                    break;
                }
            }
        }
        self.expect(Tok::RParen)?;
        Ok(list)
    }

    /// `REPLACE`。`ddl` フィーチャ有効時は `CREATE OR REPLACE` 用にすでに
    /// グローバル予約語（`Kw::Replace`）になっているため、その形も受け付ける。
    /// 無効時は `EXCLUDE` と同じく綴りだけで判定する文脈依存キーワード
    /// （実データに `replace`/`exclude` という列名が現れても壊さないため）。
    #[inline]
    fn is_star_replace_kw(&self) -> bool {
        #[cfg(feature = "ddl")]
        if self.is(Tok::Kw(Kw::Replace)) {
            return true;
        }
        self.is_soft_kw(b"replace")
    }

    /// `*`/`t.*` の直後に続きうる `EXCLUDE (col, ...)` / `REPLACE (expr AS
    /// col, ...)`（DuckDB 拡張）。この 2 語は `ROWS`/`RANGE`/`QUALIFY` と同種の
    /// 「実データにありふれた列名」なので、`*` の直後というこの文脈でだけ
    /// キーワードとして読む。順序は EXCLUDE → REPLACE 固定（`duckdb` で
    /// `REPLACE (...) EXCLUDE (...)` の逆順を試すと構文エラーになることを
    /// 確認済み）。カンマ区切りの複数指定は括弧必須だが、1 個だけなら括弧を
    /// 省略できる（`duckdb` の挙動に合わせた）。
    fn star_modifiers(&mut self) -> Result<StarModifiers> {
        let mut exclude: Vec<String> = Vec::new();
        if self.is_soft_kw(b"exclude") {
            self.bump()?;
            if self.eat(Tok::LParen)? {
                loop {
                    let pos = self.pos;
                    let name = self.ident()?;
                    ensure!(
                        !exclude.iter().any(|e| eq_ascii_ci(e.as_bytes(), name.as_bytes())),
                        SyntaxError,
                        pos
                    );
                    exclude.push(name);
                    if !self.eat(Tok::Comma)? {
                        break;
                    }
                }
                self.expect(Tok::RParen)?;
            } else {
                exclude.push(self.ident()?);
            }
        }
        let mut replace: Vec<(ExprId, String)> = Vec::new();
        if self.is_star_replace_kw() {
            self.bump()?;
            if self.eat(Tok::LParen)? {
                loop {
                    let e = self.expr()?;
                    self.expect_kw(Kw::As)?;
                    let pos = self.pos;
                    let name = self.ident()?;
                    ensure!(
                        !replace.iter().any(|(_, n): &(ExprId, String)| eq_ascii_ci(
                            n.as_bytes(),
                            name.as_bytes()
                        )),
                        SyntaxError,
                        pos
                    );
                    replace.push((e, name));
                    if !self.eat(Tok::Comma)? {
                        break;
                    }
                }
                self.expect(Tok::RParen)?;
            } else {
                let e = self.expr()?;
                self.expect_kw(Kw::As)?;
                let name = self.ident()?;
                replace.push((e, name));
            }
        }
        // 同じ列を EXCLUDE と REPLACE の両方に置くのは無意味（`duckdb` も
        // 拒否する）ので、ここで検出する。
        let pos = self.pos;
        for (_, name) in &replace {
            ensure!(
                !exclude.iter().any(|e| eq_ascii_ci(e.as_bytes(), name.as_bytes())),
                SyntaxError,
                pos
            );
        }
        Ok((exclude, replace))
    }

    fn select_item(&mut self) -> Result<SelectItem> {
        // 先頭の `*` だけは式ではなく列挙として扱う。`t.*` は primary 側。
        if self.is(Tok::Star) {
            self.bump()?;
            let (exclude, replace) = self.star_modifiers()?;
            let expr = self.arena.push(Expr::Star { qualifier: None, exclude, replace });
            return Ok(SelectItem { expr, alias: None });
        }
        let expr = self.expr()?;
        let alias = self.opt_alias()?;
        Ok(SelectItem { expr, alias })
    }

    fn order_item(&mut self) -> Result<OrderByItem> {
        let expr = self.expr()?;
        let mut desc = false;
        if self.eat_kw(Kw::Desc)? {
            desc = true;
        } else {
            self.eat_kw(Kw::Asc)?;
        }
        // Default matches DuckDB's actual behavior: NULLS LAST regardless of
        // ASC/DESC (verified against a real `duckdb` CLI) — not the
        // SQL-standard/PostgreSQL convention of "NULL is the largest value"
        // (NULLS LAST for ASC, NULLS FIRST for DESC), which this used to
        // implement and which silently disagreed with the reference
        // implementation this project cross-checks against.
        let mut nulls_first = false;
        if self.eat_kw(Kw::Nulls)? {
            if self.eat_kw(Kw::First)? {
                nulls_first = true;
            } else {
                self.expect_kw(Kw::Last)?;
                nulls_first = false;
            }
        }
        Ok(OrderByItem { expr, desc, nulls_first })
    }

    // --- FROM ---------------------------------------------------------------

    fn parse_from_item(&mut self) -> Result<FromItem> {
        let mut left = self.base_rel()?;
        loop {
            // ON を要求するのは明示 JOIN のみ。CROSS と暗黙のカンマ結合は無条件。
            let (kind, needs_on) = match self.cur {
                Tok::Comma => {
                    self.bump()?;
                    (JoinKind::Cross, false)
                }
                Tok::Kw(Kw::Cross) => {
                    self.bump()?;
                    self.expect_kw(Kw::Join)?;
                    (JoinKind::Cross, false)
                }
                Tok::Kw(Kw::Join) => {
                    self.bump()?;
                    (JoinKind::Inner, true)
                }
                Tok::Kw(k @ (Kw::Inner | Kw::Left | Kw::Right | Kw::Full)) => {
                    self.bump()?;
                    self.eat_kw(Kw::Outer)?;
                    self.expect_kw(Kw::Join)?;
                    let kind = match k {
                        Kw::Left => JoinKind::Left,
                        Kw::Right => JoinKind::Right,
                        Kw::Full => JoinKind::Full,
                        _ => JoinKind::Inner,
                    };
                    (kind, true)
                }
                _ => break,
            };
            // 左深の `Box` 連鎖が伸びすぎると破棄時の再帰でスタックを使い切る。
            self.link()?;
            let right = self.base_rel()?;
            let on = if needs_on {
                self.expect_kw(Kw::On)?;
                Some(self.expr()?)
            } else {
                None
            };
            left = FromItem::Join { left: Box::new(left), right: Box::new(right), kind, on };
        }
        Ok(left)
    }

    fn base_rel(&mut self) -> Result<FromItem> {
        if self.is(Tok::LParen) {
            self.bump()?;
            // 派生表はクエリ全体を取る。集合演算も CTE も書ける。
            let query = self.query_stmt()?;
            self.expect(Tok::RParen)?;
            let alias = self.opt_alias()?;
            return Ok(FromItem::Subquery { query: Box::new(query), alias });
        }
        let pos = self.pos;
        let is_parquet = matches!(self.cur, Tok::Ident(s) if eq_ascii_ci(s.as_bytes(), b"parquet"));
        // `UNNEST` も `parquet(...)` と同じ「非予約語だが `(` が続けば特殊構文」
        // という扱い。予約語化すると同名の列参照を壊す事故が過去にあった
        // （`ROWS`/`RANGE`/`QUALIFY`/`RECURSIVE`）ので踏襲する。
        let is_unnest = matches!(self.cur, Tok::Ident(s) if eq_ascii_ci(s.as_bytes(), b"unnest"));
        // `RANGE` はウィンドウ枠（`OVER (... RANGE BETWEEN ...)`）でも文脈依存
        // キーワードとして使われるが、そちらは `parse_window` の別の構文位置
        // （`ORDER BY` の直後）でしか見ないので、ここでテーブル関数として
        // 扱っても衝突しない（`is_soft_kw` の呼び出し元がそれぞれ独立している
        // ことを確認済み）。
        let is_generate_series =
            matches!(self.cur, Tok::Ident(s) if eq_ascii_ci(s.as_bytes(), b"generate_series"));
        let is_range = matches!(self.cur, Tok::Ident(s) if eq_ascii_ci(s.as_bytes(), b"range"));
        let name = self.ident()?;
        if self.is(Tok::LParen) {
            if is_generate_series || is_range {
                self.bump()?; // '('
                let mut args = Vec::with_capacity(3);
                if !self.is(Tok::RParen) {
                    args.push(self.signed_int_lit()?);
                    while self.eat(Tok::Comma)? {
                        args.push(self.signed_int_lit()?);
                    }
                }
                self.expect(Tok::RParen)?;
                ensure!(!args.is_empty() && args.len() <= 3, WrongArgCount, pos);
                // `range` は半開区間・単項なら start=0、`generate_series` は
                // 閉区間・単項なら stop=args[0]（`duckdb` CLI で確認済み）。
                let (start, stop, step) = match args.len() {
                    1 => (0, args[0], 1),
                    2 => (args[0], args[1], 1),
                    _ => (args[0], args[1], args[2]),
                };
                let alias = self.opt_alias()?;
                // `AS t(x)`。列名リストは 1 個だけ（`UNNEST` と同じ形）。
                let column_alias = if self.is(Tok::LParen) {
                    self.bump()?;
                    let col = self.ident()?;
                    self.expect(Tok::RParen)?;
                    Some(col)
                } else {
                    None
                };
                return Ok(FromItem::GenerateSeries {
                    start,
                    stop,
                    step,
                    inclusive: is_generate_series,
                    alias,
                    column_alias,
                });
            }
            if is_unnest {
                self.bump()?; // '('
                let expr = self.expr()?;
                self.expect(Tok::RParen)?;
                let alias = self.opt_alias()?;
                // `AS x(col)`。列名リストは 1 個だけ（`UNNEST` は常に 1 列を
                // 生む。複数列を返す DuckDB の `UNNEST(struct)` 展開は未対応）。
                let column_alias = if self.is(Tok::LParen) {
                    self.bump()?;
                    let col = self.ident()?;
                    self.expect(Tok::RParen)?;
                    Some(col)
                } else {
                    None
                };
                return Ok(FromItem::Unnest { expr, alias, column_alias });
            }
            // テーブル関数は parquet('...') だけ。ほかは v1 の範囲外。
            ensure!(is_parquet, UnsupportedFeature, pos);
            self.bump()?;
            let path = match self.cur {
                Tok::Str(s) => unquote(s, b'\''),
                _ => err!(UnexpectedToken, self.pos),
            };
            self.bump()?;
            self.expect(Tok::RParen)?;
            let alias = self.opt_alias()?;
            return Ok(FromItem::Parquet { path, alias });
        }
        let alias = self.opt_alias()?;
        Ok(FromItem::Table { name, alias })
    }

    // --- SAMPLE ---------------------------------------------------------------
    //
    // `SAMPLE`/`USING`/`TABLESAMPLE`/`BERNOULLI`/`SYSTEM`/`RESERVOIR`/`ROWS`/
    // `PERCENT` はどれも予約語にしない。`ROWS`/`RANGE`/`QUALIFY` の事故
    // （ファイル冒頭コメント参照）と同じ理由で、`FROM <item>` の直後という
    // 決まった位置だけで綴りを見て判定する。`duckdb` CLI で確認したところ
    // `SAMPLE` 単体（`USING`/`TABLESAMPLE` を伴わない）は文法上どこにも
    // 現れないので、`SAMPLE` という列名が壊れる心配も無い。

    /// `TABLESAMPLE <body>`。`duckdb` CLI で確認した位置の制約: `TABLESAMPLE`
    /// は FROM 項目に直接くっつく修飾子で、`FROM t TABLESAMPLE 10% WHERE ...`
    /// のように必ず `WHERE`/`GROUP BY`/... より前（FROM 項目の直後）に置く。
    /// `WHERE` の後に置くと `duckdb` も構文エラーになる
    /// (`duckdb -c "... WHERE ... TABLESAMPLE 10%"` → `syntax error at or near
    /// "TABLESAMPLE"`) ので、呼び出し元（FROM 項目の直後）でだけ呼ぶ。
    fn opt_tablesample_clause(&mut self) -> Result<Option<SampleSpec>> {
        if !self.is_soft_kw(b"tablesample") {
            return Ok(None);
        }
        self.bump()?; // tablesample
        Ok(Some(self.sample_body()?))
    }

    /// `USING SAMPLE <body>`。`TABLESAMPLE` と違い、こちらは文全体に対する
    /// 独立した句で、`WHERE`/`GROUP BY`/`HAVING`/`WINDOW`/`QUALIFY` の後・
    /// `ORDER BY` の前に置く（`duckdb` CLI で確認済み: `FROM t USING SAMPLE
    /// 10% WHERE ...` は構文エラーになるが `FROM t WHERE ... USING SAMPLE
    /// 10%` は通る）。FROM 項目の直後に `WHERE` 等を挟まず直接書いた場合は
    /// この関数と `opt_tablesample_clause` のどちらでも同じ位置になるが、
    /// 呼び出しは常にこちら（文末側、`QUALIFY` の直後）だけで行う。
    fn opt_using_sample_clause(&mut self) -> Result<Option<SampleSpec>> {
        if !(self.is_soft_kw(b"using") && self.peek_is_soft_kw(b"sample")?) {
            return Ok(None);
        }
        self.bump()?; // using
        self.bump()?; // sample
        Ok(Some(self.sample_body()?))
    }

    /// `<method>(<amount>[unit])` / `(<amount>[unit])` / `<amount>[unit]
    /// [(<method>, <seed>)]` のいずれか（`duckdb` CLI で確認した 3 通りの形）。
    fn sample_body(&mut self) -> Result<SampleSpec> {
        if let Tok::Ident(s) = self.cur {
            if let Some(method) = sample_method_from_ident(s.as_bytes()) {
                if self.peek()? == Tok::LParen {
                    self.bump()?; // method 名
                    self.bump()?; // '('
                    let (amount, is_rows) = self.sample_amount()?;
                    self.expect(Tok::RParen)?;
                    return Ok(SampleSpec { method, amount, is_rows, seed: None });
                }
            }
        }
        if self.is(Tok::LParen) {
            self.bump()?;
            let (amount, is_rows) = self.sample_amount()?;
            self.expect(Tok::RParen)?;
            return Ok(SampleSpec { method: SampleMethod::System, amount, is_rows, seed: None });
        }
        let (amount, is_rows) = self.sample_amount()?;
        // 単位無し（裸の数値）は行数指定として扱う（`duckdb` の実測挙動）。
        let mut method = if is_rows { SampleMethod::Reservoir } else { SampleMethod::System };
        let mut seed = None;
        if self.is(Tok::LParen) {
            self.bump()?;
            let m = match self.cur {
                Tok::Ident(s) => s.as_bytes(),
                _ => err!(UnexpectedToken, self.pos),
            };
            method = match sample_method_from_ident(m) {
                Some(m) => m,
                None => err!(UnexpectedToken, self.pos),
            };
            self.bump()?;
            self.expect(Tok::Comma)?;
            seed = Some(self.signed_int_lit()?);
            self.expect(Tok::RParen)?;
        }
        Ok(SampleSpec { method, amount, is_rows, seed })
    }

    /// `<数値>['%' | PERCENT | ROWS]`。単位が無ければ行数指定として扱う
    /// （`duckdb` の実測挙動: `USING SAMPLE 100` は 100 行、`USING SAMPLE
    /// 0.1` はパーセントではなく行数として解釈される）。
    fn sample_amount(&mut self) -> Result<(f64, bool)> {
        let pos = self.pos;
        let v = match self.cur {
            Tok::Int(s) => int_literal(s, false, pos)?,
            Tok::Float(s) => float_literal(s, pos)?,
            _ => err!(UnexpectedToken, pos),
        };
        self.bump()?;
        let amount = match v {
            Value::I32(x) => x as f64,
            Value::I64(x) => x as f64,
            Value::F64(x) => x,
            _ => err!(NumberOverflow, pos),
        };
        let is_rows = if self.eat(Tok::Percent)? {
            false
        } else if self.is_soft_kw(b"percent") {
            self.bump()?;
            false
        } else if self.is_soft_kw(b"rows") {
            self.bump()?;
            true
        } else {
            true
        };
        if is_rows {
            // `Tok::Int`/`Tok::Float` は符号を含まない（`-` は独立したトークン
            // なのでここまで来ない）ので、`amount` は常に 0 以上。負の行数を
            // 拒否するチェックは不要。
        } else {
            // `duckdb`: "Sample sample_size ... out of range, must be between 0 and 100"
            ensure!((0.0..=100.0).contains(&amount), SyntaxError, pos);
        }
        Ok((amount, is_rows))
    }

    // --- PIVOT/UNPIVOT -------------------------------------------------------
    //
    // `PIVOT`/`UNPIVOT`/`USING`/`INTO`/`NAME`/`VALUE` はどれも予約語にしない。
    // 文の先頭（`PIVOT`/`UNPIVOT`）、または各構文の中の決まった位置
    // （`ON` の直後の `USING`、列リストの直後の `INTO NAME .. VALUE ..`）
    // でだけ綴りを見て判定する。`ROWS`/`RANGE`/`QUALIFY` を予約語にして
    // 同名列を壊した過去の事故と同じ理由（`sql::lexer` 冒頭コメント参照）。

    /// `PIVOT <from> ON <on> [IN (...)] USING <agg>[, ...] [GROUP BY <cols>]`。
    /// `PIVOT` キーワード自体は呼び出し元（`stmt`）が確認済みで、まだ消費
    /// していない。
    fn pivot_stmt(&mut self) -> Result<Stmt> {
        self.bump()?; // PIVOT
        let from = self.parse_from_item()?;
        self.expect_kw(Kw::On)?;
        // `IN (...)` の直前で止めたいので、比較演算子と同じ強さの `IN` 述語を
        // 飲み込ませない（`BP_CMP + 1` 未満は結合させない）。`ON a, b` の
        // ような複数列指定は非対応 — カンマが残るので、この後 IN/USING の
        // どちらにも一致せず自然に構文エラーになる。
        let on = self.expr_bp(BP_CMP + 1)?;
        let in_list = if self.eat_kw(Kw::In)? {
            self.expect(Tok::LParen)?;
            let mut vals = Vec::new();
            loop {
                let e = self.expr()?;
                let alias = self.opt_alias()?;
                vals.push((e, alias));
                if !self.eat(Tok::Comma)? {
                    break;
                }
            }
            self.expect(Tok::RParen)?;
            Some(vals)
        } else {
            None
        };
        let using = if self.is_soft_kw(b"using") {
            self.bump()?;
            let mut items = Vec::new();
            loop {
                let expr = self.expr()?;
                let alias = self.opt_alias()?;
                items.push(SelectItem { expr, alias });
                if !self.eat(Tok::Comma)? {
                    break;
                }
            }
            items
        } else {
            // DuckDB と同じく、`USING` 省略時は `count(*)` が既定
            // （`plan::bind::desugar_pivot` が実際に補う）。
            Vec::new()
        };
        let mut group_by = Vec::new();
        if self.eat_kw(Kw::Group)? {
            self.expect_kw(Kw::By)?;
            loop {
                group_by.push(self.expr()?);
                if !self.eat(Tok::Comma)? {
                    break;
                }
            }
        }
        let (order_by, limit, offset) = self.order_limit_offset_tail()?;
        Ok(Stmt::Pivot(Box::new(PivotStmt {
            from,
            on,
            in_list,
            using,
            group_by,
            order_by,
            limit,
            offset,
        })))
    }

    /// `UNPIVOT <from> ON <col, ...> [INTO NAME <name> VALUE <value>]`。
    /// `UNPIVOT` キーワード自体は呼び出し元が確認済みで、まだ消費していない。
    fn unpivot_stmt(&mut self) -> Result<Stmt> {
        self.bump()?; // UNPIVOT
        let from = self.parse_from_item()?;
        self.expect_kw(Kw::On)?;
        // `(a, b), (c, d)` のような複数列同時畳み込みは非対応。裸の列参照の
        // カンマ区切りのみ受理する（式や `t.col` も構文上は通ってしまうが、
        // 展開時（`desugar_unpivot`）に裸の列参照でなければ拒否する）。
        let mut columns = Vec::new();
        loop {
            columns.push(self.expr()?);
            if !self.eat(Tok::Comma)? {
                break;
            }
        }
        let (name_col, value_col) = if self.is_soft_kw(b"into") {
            self.bump()?;
            ensure!(self.is_soft_kw(b"name"), UnexpectedToken, self.pos);
            self.bump()?;
            let name_col = self.ident()?;
            ensure!(self.is_soft_kw(b"value"), UnexpectedToken, self.pos);
            self.bump()?;
            let value_col = self.ident()?;
            (name_col, value_col)
        } else {
            (String::from("name"), String::from("value"))
        };
        let (order_by, limit, offset) = self.order_limit_offset_tail()?;
        Ok(Stmt::Unpivot(Box::new(UnpivotStmt {
            from,
            columns,
            name_col,
            value_col,
            order_by,
            limit,
            offset,
        })))
    }

    /// 末尾の `ORDER BY <items> [LIMIT n] [OFFSET n]`。`PIVOT`/`UNPIVOT` は
    /// 集合演算も CTE も持たない単純な文なので、`query_body` の同種の処理
    /// （こちらは `SetExpr`/`WITH` の分岐まで持つ）を簡略化した専用版。
    fn order_limit_offset_tail(&mut self) -> Result<(Vec<OrderByItem>, Option<u64>, Option<u64>)> {
        let mut order_by = Vec::new();
        if self.eat_kw(Kw::Order)? {
            self.expect_kw(Kw::By)?;
            loop {
                order_by.push(self.order_item()?);
                if !self.eat(Tok::Comma)? {
                    break;
                }
            }
        }
        let limit = if self.eat_kw(Kw::Limit)? { Some(self.uint()?) } else { None };
        let offset = if self.eat_kw(Kw::Offset)? { Some(self.uint()?) } else { None };
        Ok((order_by, limit, offset))
    }

    // --- 式 -----------------------------------------------------------------

    fn expr(&mut self) -> Result<ExprId> {
        self.expr_bp(0)
    }

    /// 深さを 1 段消費してから本体へ。エラー経路でも必ず戻すため薄く包む。
    fn expr_bp(&mut self, min_bp: u8) -> Result<ExprId> {
        ensure!(self.depth < MAX_DEPTH, ExpressionTooDeep, self.pos);
        self.depth += 1;
        let r = self.expr_body(min_bp);
        self.depth -= 1;
        r
    }

    fn expr_body(&mut self, min_bp: u8) -> Result<ExprId> {
        let mut lhs = self.prefix()?;
        loop {
            // `GLOB`/`SIMILAR TO` は ROWS/RANGE/QUALIFY と同じ理由で予約語表
            // には入れず（ファイル冒頭 `sql/lexer.rs` のコメント参照）、この
            // 中置演算子の構文位置でだけ綴りを見て判定する。こうすれば
            // `glob`/`similar` という名前の列も引用符無しでそのまま使える。
            //
            // DuckDB は `GLOB` に `NOT` を前置できない（`NOT (x GLOB y)` と
            // 書く必要がある。`duckdb -c "select 'a' NOT GLOB 'b'"` が構文
            // エラーになることを確認済み）ので、ここでは `LIKE` と違って
            // `predicate()` を経由させない。`x NOT GLOB y` は `Tok::Kw(Kw::Not)`
            // 分岐から `predicate()` に入り、そこに `glob` の腕が無いので
            // 自然に `UnexpectedToken` になる。
            if self.is_soft_kw(b"glob") {
                if BP_CMP < min_bp {
                    break;
                }
                self.bump()?; // glob
                let pattern = self.expr_bp(BP_CONCAT)?;
                lhs = self.arena.push(Expr::Function {
                    name: "glob".into(),
                    args: vec![lhs, pattern],
                    distinct: false,
                    star: false,
                    filter: None,
                });
                continue;
            }
            // `SIMILAR TO` は `LIKE` 同様 `[NOT]` を前置できるので、`predicate()`
            // 側（`Tok::Kw(Kw::Not)` 分岐）にも同じ判定を足してある。
            if self.is_soft_kw(b"similar") && self.peek_is_soft_kw(b"to")? {
                if BP_CMP < min_bp {
                    break;
                }
                lhs = self.similar_to(lhs, false)?;
                continue;
            }
            let (op, bp) = match self.cur {
                Tok::Kw(Kw::Or) => (BinaryOp::Or, BP_OR),
                Tok::Kw(Kw::And) => (BinaryOp::And, BP_AND),
                Tok::Eq => (BinaryOp::Eq, BP_CMP),
                Tok::Ne => (BinaryOp::Ne, BP_CMP),
                Tok::Lt => (BinaryOp::Lt, BP_CMP),
                Tok::Le => (BinaryOp::Le, BP_CMP),
                Tok::Gt => (BinaryOp::Gt, BP_CMP),
                Tok::Ge => (BinaryOp::Ge, BP_CMP),
                Tok::Concat => (BinaryOp::Concat, BP_CONCAT),
                Tok::Plus => (BinaryOp::Add, BP_ADD),
                Tok::Minus => (BinaryOp::Sub, BP_ADD),
                Tok::Star => (BinaryOp::Mul, BP_MUL),
                Tok::Slash => (BinaryOp::Div, BP_MUL),
                Tok::Percent => (BinaryOp::Mod, BP_MUL),
                // `->`/`->>` は新しい BinaryOp を増やさず、`json_extract`/
                // `json_extract_string` 呼び出しへの糖衣構文として展開する
                // （`expr::funcs` の型解決・実行にそのまま乗る）。
                // 優先順位は Postgres の「その他の演算子」band に倣い `||` と
                // 同じ強さにする（比較より強く結合するので `doc->'a' = 1` が
                // 括弧無しで書ける）。
                Tok::Arrow | Tok::LongArrow => {
                    if BP_CONCAT < min_bp {
                        break;
                    }
                    let is_text = self.cur == Tok::LongArrow;
                    self.bump()?;
                    let rhs = self.expr_bp(BP_CONCAT + 1)?;
                    let name = if is_text { "json_extract_string" } else { "json_extract" };
                    lhs = self.arena.push(Expr::Function {
                        name: name.into(),
                        args: vec![lhs, rhs],
                        distinct: false,
                        star: false,
                        filter: None,
                    });
                    continue;
                }
                // 述語（IS NULL / IN / BETWEEN / LIKE / ILIKE）は比較と同じ強さの後置。
                Tok::Kw(Kw::Is | Kw::In | Kw::Between | Kw::Like | Kw::Ilike | Kw::Not) => {
                    if BP_CMP < min_bp {
                        break;
                    }
                    lhs = self.predicate(lhs)?;
                    continue;
                }
                _ => break,
            };
            if bp < min_bp {
                break;
            }
            self.bump()?;
            // すべて左結合なので、右辺は 1 段強い下限で読む。
            let rhs = self.expr_bp(bp + 1)?;
            lhs = self.arena.push(Expr::Binary { op, lhs, rhs });
        }
        Ok(lhs)
    }

    fn prefix(&mut self) -> Result<ExprId> {
        match self.cur {
            Tok::Minus => {
                self.bump()?;
                // 負の整数はリテラル 1 個に畳む。そうしないと
                // -9223372036854775808 のように正側へ収まらない値が書けない。
                if let Tok::Int(text) = self.cur {
                    let v = int_literal(text, true, self.pos)?;
                    self.bump()?;
                    return Ok(self.arena.push(Expr::Literal(v)));
                }
                let arg = self.expr_bp(BP_UNARY)?;
                Ok(self.arena.push(Expr::Unary { op: UnaryOp::Neg, arg }))
            }
            // 単項 + は恒等。ノードを作らない。
            Tok::Plus => {
                self.bump()?;
                self.expr_bp(BP_UNARY)
            }
            Tok::Kw(Kw::Not) => {
                self.bump()?;
                // `NOT EXISTS` は `Unary::Not` で包まず negated に落とす。
                if self.is(Tok::Kw(Kw::Exists)) {
                    return self.exists(true);
                }
                let arg = self.expr_bp(BP_NOT)?;
                Ok(self.arena.push(Expr::Unary { op: UnaryOp::Not, arg }))
            }
            _ => self.primary(),
        }
    }

    /// `IS [NOT] NULL` / `[NOT] IN` / `[NOT] BETWEEN` / `[NOT] LIKE`。
    /// 否定は `Unary::Not` で包まず、各ノードの `negated` に落とす。
    fn predicate(&mut self, arg: ExprId) -> Result<ExprId> {
        let negated = self.eat_kw(Kw::Not)?;
        // `SIMILAR` は予約語ではない（`expr_body` 冒頭のコメント参照）ので、
        // 通常の `match self.cur` には乗せられない。ここだけ先に判定する。
        if self.is_soft_kw(b"similar") && self.peek_is_soft_kw(b"to")? {
            return self.similar_to(arg, negated);
        }
        let node = match self.cur {
            Tok::Kw(Kw::Is) => {
                ensure!(!negated, UnexpectedToken, self.pos);
                self.bump()?;
                let neg = self.eat_kw(Kw::Not)?;
                // v1 は IS [NOT] NULL のみ。IS TRUE などは範囲外。
                if !self.is(Tok::Kw(Kw::Null)) {
                    err!(UnsupportedFeature, self.pos);
                }
                self.bump()?;
                Expr::IsNull { arg, negated: neg }
            }
            Tok::Kw(Kw::In) => {
                self.bump()?;
                self.expect(Tok::LParen)?;
                // `IN (SELECT ...)` は副問い合わせ、それ以外は値リスト。
                // `IN ((SELECT 1))` は値リスト側（要素がスカラサブクエリ）。
                if self.starts_query() {
                    let query = self.query_stmt()?;
                    self.expect(Tok::RParen)?;
                    return Ok(self.arena.push(Expr::InSubquery {
                        arg,
                        query: Box::new(query),
                        negated,
                    }));
                }
                let mut list = Vec::new();
                loop {
                    let e = self.expr()?;
                    list.push(e);
                    if !self.eat(Tok::Comma)? {
                        break;
                    }
                }
                self.expect(Tok::RParen)?;
                Expr::InList { arg, list, negated }
            }
            Tok::Kw(Kw::Between) => {
                self.bump()?;
                // 境界は AND より強い下限で読む。区切りの AND を食わないため。
                let low = self.expr_bp(BP_CONCAT)?;
                self.expect_kw(Kw::And)?;
                let high = self.expr_bp(BP_CONCAT)?;
                Expr::Between { arg, low, high, negated }
            }
            Tok::Kw(k @ (Kw::Like | Kw::Ilike)) => {
                self.bump()?;
                let pattern = self.expr_bp(BP_CONCAT)?;
                let mut escape = None;
                if self.eat_kw(Kw::Escape)? {
                    let pos = self.pos;
                    let raw = match self.cur {
                        Tok::Str(s) => s,
                        _ => err!(UnexpectedToken, pos),
                    };
                    let bytes = unquote(raw, b'\'').into_bytes();
                    // エスケープ文字は 1 バイトのみ受け付ける。
                    ensure!(bytes.len() == 1, SyntaxError, pos);
                    escape = bytes.first().copied();
                    self.bump()?;
                }
                Expr::Like { arg, pattern, negated, escape, ci: k == Kw::Ilike }
            }
            _ => err!(UnexpectedToken, self.pos),
        };
        Ok(self.arena.push(node))
    }

    /// `[NOT] SIMILAR TO pattern`。DuckDB は `SIMILAR TO` を単に
    /// `regexp_full_match` への糖衣構文として扱う（`duckdb -c "explain select
    /// 'a' similar to 'a'"` で確認済み）。SQL 標準の `SIMILAR TO` と違い
    /// `_`/`%` は特別扱いされず、素の POSIX 風正規表現として渡る。
    /// このエンジンには `regexp_full_match` という名前の関数は無いが、
    /// `expr::funcs` 側にこの糖衣構文専用として実装してある
    /// （新しい正規表現エンジンは書かず、既存の `expr::regex` を
    /// アンカー付きで再利用する。詳細はそちらのモジュール doc）。
    ///
    /// `ESCAPE` 句は DuckDB 自身も「未実装」として拒否する
    /// （`duckdb -c "select 'a' similar to 'a' escape '\\'"` が
    /// `Not implemented Error: Custom escape in SIMILAR TO` になることを
    /// 確認済み）ので、ここでも明示的に拒否する。
    fn similar_to(&mut self, arg: ExprId, negated: bool) -> Result<ExprId> {
        self.bump()?; // similar
        self.bump()?; // to
        let pattern = self.expr_bp(BP_CONCAT)?;
        ensure!(!self.is(Tok::Kw(Kw::Escape)), UnsupportedFeature, self.pos);
        let call = self.arena.push(Expr::Function {
            name: "regexp_full_match".into(),
            args: vec![arg, pattern],
            distinct: false,
            star: false,
            filter: None,
        });
        if negated {
            Ok(self.arena.push(Expr::Unary { op: UnaryOp::Not, arg: call }))
        } else {
            Ok(call)
        }
    }

    fn primary(&mut self) -> Result<ExprId> {
        let pos = self.pos;
        let node = match self.cur {
            Tok::Int(t) => {
                let v = int_literal(t, false, pos)?;
                self.bump()?;
                Expr::Literal(v)
            }
            Tok::Float(t) => {
                let v = float_literal(t, pos)?;
                self.bump()?;
                Expr::Literal(v)
            }
            Tok::Str(s) => {
                let v = Value::Bytes(unquote(s, b'\'').into_bytes());
                self.bump()?;
                Expr::Literal(v)
            }
            Tok::Kw(Kw::True) => {
                self.bump()?;
                Expr::Literal(Value::Bool(true))
            }
            Tok::Kw(Kw::False) => {
                self.bump()?;
                Expr::Literal(Value::Bool(false))
            }
            Tok::Kw(Kw::Null) => {
                self.bump()?;
                Expr::Literal(Value::Null)
            }
            Tok::Param => {
                // 番号は出現順。ホスト側のバインド配列の添字になる。
                ensure!(self.num_params < u16::MAX, UnsupportedFeature, pos);
                let n = self.num_params;
                self.num_params += 1;
                self.bump()?;
                Expr::Param(n)
            }
            Tok::LParen => {
                self.bump()?;
                // `(SELECT ...)` はスカラサブクエリ、それ以外は括弧付きの式。
                if self.starts_query() {
                    let query = self.query_stmt()?;
                    self.expect(Tok::RParen)?;
                    return Ok(self.arena.push(Expr::ScalarSubquery(Box::new(query))));
                }
                let e = self.expr()?;
                self.expect(Tok::RParen)?;
                return Ok(e);
            }
            // `[expr, ...]` 配列リテラル。式の開始位置でしか出てこないので、
            // `expr[i]` のような添字アクセス（今回のスコープ外。ファイル冒頭
            // `sql/lexer.rs` の `Tok::LBracket` のコメント参照）とは衝突しない。
            Tok::LBracket => return self.array_literal(),
            Tok::Kw(Kw::Exists) => return self.exists(false),
            Tok::Kw(Kw::Cast) => return self.cast(),
            Tok::Kw(Kw::Case) => return self.case(),
            // `INTERVAL` は予約語にしていない（`interval` という列名も書けて
            // 欲しいため）ので、ここで綴りを見てから中身を先読みして判定する。
            Tok::Ident(s) if eq_ascii_ci(s.as_bytes(), b"interval") => {
                return self.interval_literal_or_ident();
            }
            Tok::Ident(_) | Tok::QIdent(_) => return self.name_ref(),
            _ => err!(UnexpectedToken, pos),
        };
        Ok(self.arena.push(node))
    }

    /// `[expr, expr, ...]`。`list_value(expr, expr, ...)`（`json_array` の
    /// 別名）へそのまま脱糖する糖衣構文で、意味は完全に同じ。
    ///
    /// 空配列 `[]` だけは `list_value()` を経由しない: `list_value`/`json_array`
    /// の `resolve` は 0 引数を `WrongArgCount` で拒否する設計になっている
    /// （引数が無いと `call()` が行数を決められないため）ので、代わりに
    /// JSON の空配列を直接 `TypedLiteral` として埋め込む。`duckdb -c "select
    /// [], typeof([])"` で `[]` 自体は有効な式であることを確認済み。
    fn array_literal(&mut self) -> Result<ExprId> {
        self.bump()?; // '['
        if self.eat(Tok::RBracket)? {
            return Ok(self.arena.push(Expr::TypedLiteral(Value::Bytes(b"[]".to_vec()), Ty::Json)));
        }
        let mut args = Vec::new();
        loop {
            args.push(self.expr()?);
            if !self.eat(Tok::Comma)? {
                break;
            }
        }
        self.expect(Tok::RBracket)?;
        Ok(self.arena.push(Expr::Function {
            name: "list_value".into(),
            args,
            distinct: false,
            star: false,
            filter: None,
        }))
    }

    /// `INTERVAL '<n> <unit> ...'` / `INTERVAL '<n>' <unit>` /
    /// `INTERVAL <n> <unit>` リテラル。`INTERVAL` の直後がこれらの形でなければ
    /// 列参照 (`interval` という名前の列) として `name_ref` に委ねる。
    ///
    /// `call()` の `EXTRACT` と同じ流儀: `Lexer` を複製して形を確定させてから
    /// 本物のトークン列を同じ回数だけ進める。
    fn interval_literal_or_ident(&mut self) -> Result<ExprId> {
        let mut lx = self.lex.clone();
        let t1 = lx.next_token()?.tok;
        match t1 {
            Tok::Str(raw) => {
                let text = unquote(raw, b'\'');
                // 文字列全体が符号付き整数 1 個で、直後に単位語が続くなら
                // `INTERVAL '3' DAY` 形式。
                let t2 = lx.next_token()?.tok;
                if let (Some(n), Tok::Ident(u)) = (parse_signed_int(&text), t2) {
                    if let Some(unit) = lookup_interval_unit(u.as_bytes()) {
                        let pos = self.pos;
                        self.bump()?; // INTERVAL
                        self.bump()?; // 文字列
                        self.bump()?; // 単位語
                        let packed = unit_to_interval(unit, n, pos)?;
                        return Ok(self.arena.push(Expr::IntervalLiteral(packed)));
                    }
                }
                let pos = self.pos;
                self.bump()?; // INTERVAL
                self.bump()?; // 文字列
                let packed = parse_interval_text(&text, pos)?;
                Ok(self.arena.push(Expr::IntervalLiteral(packed)))
            }
            Tok::Int(text) => {
                // 引用符無しの `INTERVAL <n> <unit>`。単位語が続かなければ
                // 列参照（`interval` という名前の列に整数が続く式はどのみち
                // 構文上ありえないが、安全側に倒して列参照へ逃がす）。
                let t2 = lx.next_token()?.tok;
                let Tok::Ident(u) = t2 else {
                    return self.name_ref();
                };
                let Some(unit) = lookup_interval_unit(u.as_bytes()) else {
                    return self.name_ref();
                };
                let pos = self.pos;
                let Some(n) = parse_signed_int(text) else { err!(NumberOverflow, pos) };
                self.bump()?; // INTERVAL
                self.bump()?; // 数値
                self.bump()?; // 単位語
                let packed = unit_to_interval(unit, n, pos)?;
                Ok(self.arena.push(Expr::IntervalLiteral(packed)))
            }
            _ => self.name_ref(),
        }
    }

    /// 識別子始まりの primary: 関数呼び出し / `q.name` / `q.*` / 列参照。
    fn name_ref(&mut self) -> Result<ExprId> {
        let name = self.ident()?;
        if self.is(Tok::LParen) {
            return self.call(name);
        }
        if self.eat(Tok::Dot)? {
            if self.is(Tok::Star) {
                self.bump()?;
                let (exclude, replace) = self.star_modifiers()?;
                return Ok(self.arena.push(Expr::Star { qualifier: Some(name), exclude, replace }));
            }
            let col = self.ident()?;
            return Ok(self.arena.push(Expr::ColumnRef { qualifier: Some(name), name: col }));
        }
        Ok(self.arena.push(Expr::ColumnRef { qualifier: None, name }))
    }

    /// `[NOT] EXISTS ( query )`。呼び出し時の `cur` は `EXISTS`。
    fn exists(&mut self, negated: bool) -> Result<ExprId> {
        self.bump()?; // EXISTS
        self.expect(Tok::LParen)?;
        let query = self.query_stmt()?;
        self.expect(Tok::RParen)?;
        Ok(self.arena.push(Expr::Exists { query: Box::new(query), negated }))
    }

    fn call(&mut self, name: String) -> Result<ExprId> {
        // `EXTRACT(part FROM ts)` は `FROM` を挟む特殊構文で、通常のカンマ
        // 区切り引数列とは形が違う。`date_part(part, ts)` と等価なので、
        // ここで構文だけ吸収して同じ `Function` ノードに畳む（実行側は
        // `extract` を知らなくてよい）。
        //
        // `Parser` 自体は clone できない（`ExprArena` を持つため）ので、
        // `Lexer` の clone だけで「`( IDENT FROM` の形か」を確定させてから
        // `self` を進める。`peek()` と同じ先読みの流儀。
        // `TRY_CAST(expr AS type)` は CAST と同じ特殊構文（`AS` を挟む）。
        // 通常のカンマ引数列とは形が違うので、CAST と同じ経路に落とす。
        if eq_ascii_ci(name.as_bytes(), b"try_cast") && self.cur == Tok::LParen {
            return self.cast_body(true);
        }
        // `UNNEST(expr)`（SELECT リスト用）。DISTINCT/`*`/FILTER/OVER は
        // 意味を持たないので、通常の関数呼び出しとは別の単純な形で読む
        // （FROM 句版は `base_rel` 参照）。
        if eq_ascii_ci(name.as_bytes(), b"unnest") && self.cur == Tok::LParen {
            self.bump()?; // '('
            let arg = self.expr()?;
            self.expect(Tok::RParen)?;
            return Ok(self.arena.push(Expr::Unnest(arg)));
        }
        // `IIF(cond, then, else)` は `CASE WHEN cond THEN then ELSE else END`
        // の糖衣構文。パーサレベルで CASE 式へ脱糖する（`EXTRACT` → `date_part`
        // と同じ判断）ので、実行側は `iif` を一切知らなくてよい。
        if eq_ascii_ci(name.as_bytes(), b"iif") && self.cur == Tok::LParen {
            self.bump()?; // '('
            let cond = self.expr()?;
            self.expect(Tok::Comma)?;
            let then_ = self.expr()?;
            self.expect(Tok::Comma)?;
            let else_ = self.expr()?;
            self.expect(Tok::RParen)?;
            return Ok(self.arena.push(Expr::Case {
                operand: None,
                whens: vec![(cond, then_)],
                else_: Some(else_),
            }));
        }
        if eq_ascii_ci(name.as_bytes(), b"extract") && self.cur == Tok::LParen {
            let mut lx = self.lex.clone();
            let t1 = lx.next_token()?.tok;
            if let Tok::Ident(part) = t1 {
                let t2 = lx.next_token()?.tok;
                if t2 == Tok::Kw(Kw::From) {
                    // ここまで来たら確定。本物のトークン列を同じ回数だけ進める。
                    self.bump()?; // '('
                    self.bump()?; // part
                    self.bump()?; // FROM
                    let part_lit =
                        self.arena.push(Expr::Literal(Value::Bytes(part.as_bytes().to_vec())));
                    let ts = self.expr()?;
                    self.expect(Tok::RParen)?;
                    return Ok(self.arena.push(Expr::Function {
                        name: String::from("date_part"),
                        args: vec![part_lit, ts],
                        distinct: false,
                        star: false,
                        filter: None,
                    }));
                }
            }
        }
        self.bump()?; // '('
        let mut args = Vec::new();
        let mut distinct = false;
        let mut star = false;
        if self.is(Tok::Star) {
            // 引数位置の `*` は COUNT(*) 系だけ。意味付けは binder に任せる。
            star = true;
            self.bump()?;
        } else if !self.is(Tok::RParen) {
            distinct = self.eat_kw(Kw::Distinct)?;
            // ラムダ（`x -> expr` / `(a, b) -> expr`）は `list_transform` /
            // `list_filter` / `list_reduce` の引数位置でだけ認識する。
            // `->` は通常 JSON パス演算子の糖衣構文（`json_extract` への
            // 展開、`expr_body` 参照）なので、他の関数の引数では
            // 今までどおりその意味のまま残す（`coalesce(doc -> 'a', 'x')` の
            // ような既存の使い方を壊さないため。duckdb CLI で実測した
            // 限りでも、`->` がラムダとして解釈されるのは list_transform 等
            // 「ラムダを受け取ると分かっている関数」の引数位置だけで、
            // 例えば `coalesce(x -> 5, 3)` は `x` が列として解決できず
            // `x -> 5` は JSON 演算子のままエラーになる）。
            let lambda_fn = is_lambda_func(&name);
            loop {
                let e = if lambda_fn && self.looks_like_lambda_params()? {
                    self.lambda_expr()?
                } else {
                    self.expr()?
                };
                args.push(e);
                if !self.eat(Tok::Comma)? {
                    break;
                }
            }
        }
        self.expect(Tok::RParen)?;
        // `agg(...) FILTER (WHERE cond)`。`FILTER` も予約語ではなく、関数呼び出し
        // 直後で次が `(` のときだけキーワードとして扱う（`OVER` と同じ判断）。
        let mut filter = None;
        if self.is_soft_kw(b"filter") && self.peek()? == Tok::LParen {
            self.bump()?; // filter
            self.bump()?; // '('
            self.expect_kw(Kw::Where)?;
            filter = Some(self.expr()?);
            self.expect(Tok::RParen)?;
        }
        // `OVER` は予約語ではないので、次が `(` か識別子のときだけウィンドウ句
        // と見なす。`SELECT count(*) over FROM t` の `over` は別名のまま通る
        // （次が `FROM` などキーワードなら、下の分岐に入らず素通りする）。
        if self.is_soft_kw(b"over") {
            match self.peek()? {
                Tok::LParen => {
                    // `f(DISTINCT x) OVER (...)` / `f(...) FILTER (...) OVER (...)` は
                    // 範囲外。無視すると結果が変わるので弾く。
                    ensure!(!distinct, UnsupportedFeature, self.pos);
                    ensure!(filter.is_none(), UnsupportedFeature, self.pos);
                    return self.window(name, args, star);
                }
                Tok::Ident(_) | Tok::QIdent(_) => {
                    // `OVER w`（名前付きウィンドウの参照）。定義の実体は
                    // `WINDOW` 句にしか無く、SELECT リストの方が構文上先に
                    // 来るのでここでは名前だけ持たせ、束縛時に
                    // `SelectStmt::windows` から引く（`plan::bind` 参照）。
                    ensure!(!distinct, UnsupportedFeature, self.pos);
                    ensure!(filter.is_none(), UnsupportedFeature, self.pos);
                    self.bump()?; // OVER
                    let wname = self.ident()?;
                    return Ok(self.arena.push(Expr::Window {
                        name,
                        args,
                        star,
                        window_ref: Some(wname),
                        partition_by: Vec::new(),
                        order_by: Vec::new(),
                        frame: WindowFrame::WholePartition,
                    }));
                }
                _ => {}
            }
        }
        Ok(self.arena.push(Expr::Function { name, args, distinct, star, filter }))
    }

    /// 現在位置がラムダの仮引数リストの形（`IDENT ->` または
    /// `( IDENT [, IDENT]* ) ->`）をしているか。`Lexer` の clone だけで先読みし
    /// `self` は動かさない（`peek` と同じ流儀）。
    ///
    /// 形が違えば `false` を返すだけで構文エラーにはしない（呼び出し側が
    /// 通常の `expr()` へフォールバックする）。
    fn looks_like_lambda_params(&self) -> Result<bool> {
        match self.cur {
            Tok::Ident(_) => Ok(self.peek()? == Tok::Arrow),
            Tok::LParen => {
                let mut lx = self.lex.clone();
                loop {
                    match lx.next_token()?.tok {
                        Tok::Ident(_) => {}
                        _ => return Ok(false),
                    }
                    match lx.next_token()?.tok {
                        Tok::Comma => continue,
                        Tok::RParen => return Ok(lx.next_token()?.tok == Tok::Arrow),
                        _ => return Ok(false),
                    }
                }
            }
            _ => Ok(false),
        }
    }

    /// `x -> expr` / `(a, b) -> expr`。`looks_like_lambda_params` が `true` を
    /// 返した直後にだけ呼ぶ。
    fn lambda_expr(&mut self) -> Result<ExprId> {
        let mut params = Vec::new();
        if self.eat(Tok::LParen)? {
            loop {
                params.push(self.ident()?);
                if !self.eat(Tok::Comma)? {
                    break;
                }
            }
            self.expect(Tok::RParen)?;
        } else {
            params.push(self.ident()?);
        }
        self.expect(Tok::Arrow)?;
        // 本体は通常の式。カンマは引数区切りとして上位ループが見るので
        // ここでは（`expr_body` に元々カンマの扱いが無いことも合わせて）
        // 何も特別な下限を与えず読むだけでよい。
        let body = self.expr()?;
        Ok(self.arena.push(Expr::Lambda { params, body }))
    }

    /// `OVER ( [PARTITION BY ...] [ORDER BY ...] )`。呼び出し時の `cur` は
    /// `OVER` で、次が `(` であることは確認済み。
    fn window(&mut self, name: String, args: Vec<ExprId>, star: bool) -> Result<ExprId> {
        self.bump()?; // OVER
        self.expect(Tok::LParen)?;
        let def = self.window_def_body()?;
        Ok(self.arena.push(Expr::Window {
            name,
            args,
            star,
            window_ref: None,
            partition_by: def.partition_by,
            order_by: def.order_by,
            frame: def.frame,
        }))
    }

    /// `[PARTITION BY ...] [ORDER BY ...] )` の共通本体。呼び出し時に開き
    /// 括弧は消費済みで、閉じ括弧まで読んで消費する。`OVER (...)` の直書きと
    /// `WINDOW name AS (...)` の名前付き定義の両方から使う。
    fn window_def_body(&mut self) -> Result<WindowDef> {
        let mut partition_by = Vec::new();
        // `PARTITION` もウィンドウ指定の先頭でだけキーワードとして扱う。
        if self.is_soft_kw(b"partition") {
            self.bump()?;
            self.expect_kw(Kw::By)?;
            loop {
                let e = self.expr()?;
                partition_by.push(e);
                if !self.eat(Tok::Comma)? {
                    break;
                }
            }
        }
        let mut order_by = Vec::new();
        if self.eat_kw(Kw::Order)? {
            self.expect_kw(Kw::By)?;
            loop {
                let it = self.order_item()?;
                order_by.push(it);
                if !self.eat(Tok::Comma)? {
                    break;
                }
            }
        }
        // 明示的な枠指定は未対応。黙って既定枠を使うと結果が変わるので弾く。
        // `ROWS` / `RANGE` も予約語ではないため、ここで綴りを見て判定する。
        if self.is_soft_kw(b"rows") || self.is_soft_kw(b"range") {
            err!(UnsupportedFeature, self.pos);
        }
        self.expect(Tok::RParen)?;
        // 既定枠は SQL 標準どおり ORDER BY の有無で決まる。
        let frame = if order_by.is_empty() {
            WindowFrame::WholePartition
        } else {
            WindowFrame::RangeUnboundedPreceding
        };
        Ok(WindowDef { partition_by, order_by, frame })
    }

    fn cast(&mut self) -> Result<ExprId> {
        self.bump()?; // CAST
        self.cast_body(false)
    }

    /// `( expr AS type )`。`CAST` と `TRY_CAST` の共通部分。呼び出し時の
    /// `cur` は開き括弧。
    fn cast_body(&mut self, try_: bool) -> Result<ExprId> {
        self.expect(Tok::LParen)?;
        let arg = self.expr()?;
        self.expect_kw(Kw::As)?;
        let ty = self.type_name()?;
        self.expect(Tok::RParen)?;
        Ok(self.arena.push(Expr::Cast { arg, ty, try_ }))
    }

    fn type_name(&mut self) -> Result<Ty> {
        let pos = self.pos;
        let name = match self.cur {
            Tok::Ident(s) => s,
            _ => err!(InvalidCast, pos),
        };
        let Some(ty) = lookup_type(name.as_bytes()) else {
            err!(InvalidCast, pos);
        };
        self.bump()?;
        if matches!(ty, Ty::Decimal { .. }) && self.eat(Tok::LParen)? {
            let p = self.uint()?;
            self.expect(Tok::Comma)?;
            let s = self.uint()?;
            self.expect(Tok::RParen)?;
            ensure!((1..=38).contains(&p) && s <= p, InvalidCast, pos);
            return Ok(Ty::Decimal { precision: p as u8, scale: s as u8 });
        }
        Ok(ty)
    }

    fn case(&mut self) -> Result<ExprId> {
        self.bump()?; // CASE
                      // CASE の直後が WHEN でなければ、比較対象の式が付いている形。
        let operand = if self.is(Tok::Kw(Kw::When)) { None } else { Some(self.expr()?) };
        let mut whens = Vec::new();
        while self.eat_kw(Kw::When)? {
            let cond = self.expr()?;
            self.expect_kw(Kw::Then)?;
            let then = self.expr()?;
            whens.push((cond, then));
        }
        ensure!(!whens.is_empty(), UnexpectedToken, self.pos);
        let else_ = if self.eat_kw(Kw::Else)? { Some(self.expr()?) } else { None };
        self.expect_kw(Kw::End)?;
        Ok(self.arena.push(Expr::Case { operand, whens, else_ }))
    }
}

// --- GROUP BY の拡張構文 -----------------------------------------------------

/// `CUBE` の列数上限。2^n 個のグルーピングセット（= `Node::Aggregate` を
/// UNION ALL で束ねた本数）に展開されるため、無制限だとプランが爆発する。
const MAX_CUBE_COLS: usize = 8;

/// `ROLLUP (a, b, c)` を `GROUPING SETS ((a,b,c),(a,b),(a),())` に展開する。
/// 列の多い方から少ない方へ、階層的な部分集合を作る。
fn rollup_sets(cols: Vec<ExprId>) -> Vec<Vec<ExprId>> {
    let mut sets = Vec::with_capacity(cols.len() + 1);
    for k in (0..=cols.len()).rev() {
        sets.push(cols[..k].to_vec());
    }
    sets
}

/// `CUBE (a, b)` を `GROUPING SETS ((a,b),(a),(b),())` に展開する。
/// 全部分集合（2^n 個）を作る。
fn cube_sets(cols: Vec<ExprId>, pos: usize) -> Result<Vec<Vec<ExprId>>> {
    ensure!(cols.len() <= MAX_CUBE_COLS, ExpressionTooDeep, pos);
    let n = cols.len();
    let mut sets = Vec::with_capacity(1usize << n);
    // 先頭の列ほど上位ビットに割り当てる。こうすると `(a,b),(a),(b),()` の
    // ように「先頭に近い列を優先して残す」順になり、`ROLLUP` の階層的な
    // 部分集合の並びとも感覚が揃う（実行結果には影響しない: どの順で
    // UNION ALL しても集合として同じ）。
    for mask in (0..(1usize << n)).rev() {
        let mut set = Vec::new();
        for (i, &c) in cols.iter().enumerate() {
            if mask & (1 << (n - 1 - i)) != 0 {
                set.push(c);
            }
        }
        sets.push(set);
    }
    Ok(sets)
}

// --- ラムダ ------------------------------------------------------------------

/// 引数位置の `->` をラムダとして解釈してよい関数名か。
///
/// duckdb CLI で実測した限り、ラムダとして解釈されるのは「ラムダを受け取ると
/// 分かっている関数」の引数位置だけで、他の関数（`coalesce` 等）の引数では
/// `->` は素通りして通常の JSON パス演算子のままになる（`coalesce(doc -> 'a',
/// 'x')` は JSON 抽出として解決される一方、`abs(x -> x+1)` はラムダとして
/// 解釈されて「この関数はラムダを受け取らない」というエラーになる）。
/// この実装では関数名を固定集合として持つことで同じ区別を再現する。
fn is_lambda_func(name: &str) -> bool {
    eq_ascii_ci(name.as_bytes(), b"list_transform")
        || eq_ascii_ci(name.as_bytes(), b"list_filter")
        || eq_ascii_ci(name.as_bytes(), b"list_reduce")
}

// --- リテラル・型名 ---------------------------------------------------------

/// 引用符の中身を展開する。二重化された引用符を 1 個に畳むだけ。
fn unquote(raw: &str, q: u8) -> String {
    let b = raw.as_bytes();
    let mut out = String::new();
    let (mut i, mut start) = (0usize, 0usize);
    while i < b.len() {
        if b[i] == q {
            // 引用符は ASCII なので、この範囲は必ず文字境界に乗る。
            out.push_str(&raw[start..i + 1]);
            i += 2;
            start = i;
        } else {
            i += 1;
        }
    }
    if start < b.len() {
        out.push_str(&raw[start..]);
    }
    out
}

/// 整数リテラル。収まる最小の型（I32 → I64 → I128）を選ぶ。
fn int_literal(text: &str, negative: bool, pos: usize) -> Result<Value> {
    let mut mag: u128 = 0;
    for &d in text.as_bytes() {
        mag = match mag.checked_mul(10).and_then(|v| v.checked_add((d - b'0') as u128)) {
            Some(v) => v,
            None => err!(NumberOverflow, pos),
        };
    }
    // i128::MIN は絶対値が i128::MAX より 1 大きい。符号を見て上限を変える。
    let limit = if negative { 1u128 << 127 } else { (1u128 << 127) - 1 };
    ensure!(mag <= limit, NumberOverflow, pos);
    let v = if negative { (mag as i128).wrapping_neg() } else { mag as i128 };
    Ok(if let Ok(x) = i32::try_from(v) {
        Value::I32(x)
    } else if let Ok(x) = i64::try_from(v) {
        Value::I64(x)
    } else {
        Value::I128(v)
    })
}

fn float_literal(text: &str, pos: usize) -> Result<Value> {
    match text.parse::<f64>() {
        Ok(v) => Ok(Value::F64(v)),
        Err(_) => err!(NumberOverflow, pos),
    }
}

/// `USING SAMPLE`/`TABLESAMPLE` の手法名。一致しなければ `None`
/// （呼び出し側が「サンプル手法ではなく別の何か」として扱う）。
fn sample_method_from_ident(word: &[u8]) -> Option<SampleMethod> {
    if eq_ascii_ci(word, b"bernoulli") {
        Some(SampleMethod::Bernoulli)
    } else if eq_ascii_ci(word, b"system") {
        Some(SampleMethod::System)
    } else if eq_ascii_ci(word, b"reservoir") {
        Some(SampleMethod::Reservoir)
    } else {
        None
    }
}

/// CAST の型名表。CAST はホットパスではないので、(長さ, 先頭バイト) で
/// 絞り込む線形走査で十分。予約語表と違い二分探索の順序制約を持たない。
static TYPES: &[(&[u8], Ty)] = &[
    (b"boolean", Ty::Boolean),
    (b"bool", Ty::Boolean),
    (b"tinyint", Ty::TinyInt),
    (b"smallint", Ty::SmallInt),
    (b"int", Ty::Int),
    (b"integer", Ty::Int),
    (b"bigint", Ty::BigInt),
    (b"hugeint", Ty::HugeInt),
    (b"utinyint", Ty::UTinyInt),
    (b"usmallint", Ty::USmallInt),
    (b"uinteger", Ty::UInt),
    (b"ubigint", Ty::UBigInt),
    (b"float", Ty::Float),
    (b"real", Ty::Float),
    (b"double", Ty::Double),
    // 括弧なしの DECIMAL は (18,3)。I64 に収まる精度を既定にする。
    (b"decimal", Ty::Decimal { precision: 18, scale: 3 }),
    (b"numeric", Ty::Decimal { precision: 18, scale: 3 }),
    (b"varchar", Ty::Varchar),
    (b"text", Ty::Varchar),
    (b"string", Ty::Varchar),
    (b"char", Ty::Varchar),
    (b"blob", Ty::Blob),
    (b"bytea", Ty::Blob),
    (b"date", Ty::Date),
    (b"time", Ty::Time),
    (b"timestamp", Ty::Timestamp),
    (b"datetime", Ty::Timestamp),
    (b"json", Ty::Json),
];

fn lookup_type(name: &[u8]) -> Option<Ty> {
    if name.is_empty() {
        return None;
    }
    let head = name[0] | 0x20;
    for &(n, ty) in TYPES {
        if n.len() == name.len() && n[0] == head && eq_ascii_ci(n, name) {
            return Some(ty);
        }
    }
    None
}

// --- INTERVAL リテラル -------------------------------------------------------
// DESIGN.md §7 に載る 8 単位（年・月・日・時・分・秒・ミリ秒・マイクロ秒）だけ
// を単数形・複数形の両方で受け付ける。DuckDB にある他の略記（`mon`/`y`/`wk` 等）
// は対象外（テーブルを絞ることでコードサイズを増やさない）。

#[derive(Clone, Copy)]
enum IntervalUnit {
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
    Millisecond,
    Microsecond,
}

static INTERVAL_UNITS: &[(&[u8], IntervalUnit)] = &[
    (b"year", IntervalUnit::Year),
    (b"years", IntervalUnit::Year),
    (b"month", IntervalUnit::Month),
    (b"months", IntervalUnit::Month),
    (b"day", IntervalUnit::Day),
    (b"days", IntervalUnit::Day),
    (b"hour", IntervalUnit::Hour),
    (b"hours", IntervalUnit::Hour),
    (b"minute", IntervalUnit::Minute),
    (b"minutes", IntervalUnit::Minute),
    (b"second", IntervalUnit::Second),
    (b"seconds", IntervalUnit::Second),
    (b"millisecond", IntervalUnit::Millisecond),
    (b"milliseconds", IntervalUnit::Millisecond),
    (b"microsecond", IntervalUnit::Microsecond),
    (b"microseconds", IntervalUnit::Microsecond),
];

fn lookup_interval_unit(name: &[u8]) -> Option<IntervalUnit> {
    for &(n, u) in INTERVAL_UNITS {
        if eq_ascii_ci(n, name) {
            return Some(u);
        }
    }
    None
}

/// 符号付き 10 進整数。前後の空白は許す（`INTERVAL` の数値片用）。
fn parse_signed_int(s: &str) -> Option<i64> {
    let b = s.trim().as_bytes();
    if b.is_empty() {
        return None;
    }
    let (neg, digits) = match b[0] {
        b'-' => (true, &b[1..]),
        b'+' => (false, &b[1..]),
        _ => (false, b),
    };
    if digits.is_empty() || !digits.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let mut v: i64 = 0;
    for &d in digits {
        v = v.checked_mul(10)?.checked_add((d - b'0') as i64)?;
    }
    Some(if neg { -v } else { v })
}

/// 1 単位ぶんを `(months, days, micros)` の累積へ足し込む。
fn add_interval_unit(
    u: IntervalUnit,
    n: i64,
    months: &mut i64,
    days: &mut i64,
    micros: &mut i64,
    pos: usize,
) -> Result<()> {
    fn add(acc: &mut i64, delta: Option<i64>, pos: usize) -> Result<()> {
        match delta.and_then(|d| acc.checked_add(d)) {
            Some(v) => {
                *acc = v;
                Ok(())
            }
            None => err!(NumberOverflow, pos),
        }
    }
    const US_PER_SEC: i64 = 1_000_000;
    const US_PER_MIN: i64 = 60 * US_PER_SEC;
    const US_PER_HOUR: i64 = 60 * US_PER_MIN;
    match u {
        IntervalUnit::Year => add(months, n.checked_mul(12), pos),
        IntervalUnit::Month => add(months, Some(n), pos),
        IntervalUnit::Day => add(days, Some(n), pos),
        IntervalUnit::Hour => add(micros, n.checked_mul(US_PER_HOUR), pos),
        IntervalUnit::Minute => add(micros, n.checked_mul(US_PER_MIN), pos),
        IntervalUnit::Second => add(micros, n.checked_mul(US_PER_SEC), pos),
        IntervalUnit::Millisecond => add(micros, n.checked_mul(1_000), pos),
        IntervalUnit::Microsecond => add(micros, Some(n), pos),
    }
}

/// `months`/`days` が `i32` に収まることを確認してから詰める。
fn pack_interval_checked(months: i64, days: i64, micros: i64, pos: usize) -> Result<i128> {
    let m = match i32::try_from(months) {
        Ok(v) => v,
        Err(_) => err!(NumberOverflow, pos),
    };
    let d = match i32::try_from(days) {
        Ok(v) => v,
        Err(_) => err!(NumberOverflow, pos),
    };
    Ok(crate::vector::pack_interval(m, d, micros))
}

/// `n UNIT` 1 個ぶんの INTERVAL。
fn unit_to_interval(u: IntervalUnit, n: i64, pos: usize) -> Result<i128> {
    let (mut months, mut days, mut micros) = (0i64, 0i64, 0i64);
    add_interval_unit(u, n, &mut months, &mut days, &mut micros, pos)?;
    pack_interval_checked(months, days, micros, pos)
}

/// `'<n> <unit> [<n> <unit> ...]'` の複合形式。同じ単位が複数回出てきても
/// 単純に加算する（DuckDB も `'1 month 1 month'` を `2 months` として扱う）。
fn parse_interval_text(text: &str, pos: usize) -> Result<i128> {
    let (mut months, mut days, mut micros) = (0i64, 0i64, 0i64);
    let mut any = false;
    let mut it = text.split_ascii_whitespace();
    while let Some(num_tok) = it.next() {
        let Some(n) = parse_signed_int(num_tok) else { err!(SyntaxError, pos) };
        let Some(unit_tok) = it.next() else { err!(SyntaxError, pos) };
        let Some(unit) = lookup_interval_unit(unit_tok.as_bytes()) else { err!(SyntaxError, pos) };
        add_interval_unit(unit, n, &mut months, &mut days, &mut micros, pos)?;
        any = true;
    }
    ensure!(any, SyntaxError, pos);
    pack_interval_checked(months, days, micros, pos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Code;
    use crate::sql::lexer::{keyword, KEYWORDS};

    // --- テスト用ヘルパ -----------------------------------------------------

    /// 式木を完全括弧付きの文字列へ戻す。リテラルは型が分かる形で出す。
    fn r(a: &ExprArena, id: ExprId) -> String {
        match a.get(id) {
            // ウィンドウ指定は `PARTITION BY .. ORDER BY .. <枠>` の順で出す。
            // 枠は既定値の取り違えを検出したいので必ず描画する。
            Expr::Window { name, args, star, window_ref, partition_by, order_by, frame } => {
                let inner = if *star {
                    "*".to_string()
                } else {
                    let items: Vec<String> = args.iter().map(|i| r(a, *i)).collect();
                    items.join(", ")
                };
                if let Some(w) = window_ref {
                    return format!("{}({}) OVER {}", name, inner, w);
                }
                let mut spec: Vec<String> = Vec::new();
                if !partition_by.is_empty() {
                    let ps: Vec<String> = partition_by.iter().map(|e| r(a, *e)).collect();
                    spec.push(format!("PARTITION BY {}", ps.join(", ")));
                }
                if !order_by.is_empty() {
                    spec.push(format!("ORDER BY {}", order_list(a, order_by)));
                }
                spec.push(
                    match frame {
                        WindowFrame::WholePartition => "WHOLE",
                        WindowFrame::RangeUnboundedPreceding => "RANGE",
                    }
                    .to_string(),
                );
                format!("{}({}) OVER ({})", name, inner, spec.join(" "))
            }
            Expr::ScalarSubquery(q) => format!("({})", query_str(a, q)),
            Expr::Exists { query, negated } => {
                format!("{}EXISTS ({})", if *negated { "NOT " } else { "" }, query_str(a, query))
            }
            Expr::InSubquery { arg, query, negated } => format!(
                "({}{} IN ({}))",
                r(a, *arg),
                if *negated { " NOT" } else { "" },
                query_str(a, query)
            ),
            Expr::Literal(v) => lit(v),
            Expr::IntervalLiteral(v) => format!("INTERVAL({}i128)", v),
            Expr::TypedLiteral(v, ty) => format!("{v:?}::{}", ty.name()),
            Expr::Param(n) => format!("?{}", n),
            Expr::ColumnRef { qualifier, name } => match qualifier {
                Some(q) => format!("{}.{}", q, name),
                None => name.clone(),
            },
            Expr::Star { qualifier, exclude, replace } => {
                let mut s = match qualifier {
                    Some(q) => format!("{}.*", q),
                    None => "*".to_string(),
                };
                if !exclude.is_empty() {
                    s.push_str(&format!(" EXCLUDE ({})", exclude.join(", ")));
                }
                if !replace.is_empty() {
                    let items: Vec<String> =
                        replace.iter().map(|(e, n)| format!("{} AS {}", r(a, *e), n)).collect();
                    s.push_str(&format!(" REPLACE ({})", items.join(", ")));
                }
                s
            }
            Expr::Unary { op, arg } => {
                let o = if *op == UnaryOp::Neg { "-" } else { "NOT" };
                format!("({} {})", o, r(a, *arg))
            }
            Expr::Binary { op, lhs, rhs } => {
                format!("({} {} {})", r(a, *lhs), op_name(*op), r(a, *rhs))
            }
            Expr::Cast { arg, ty, try_ } => {
                format!(
                    "{}({} AS {})",
                    if *try_ { "TRY_CAST" } else { "CAST" },
                    r(a, *arg),
                    ty_name(*ty)
                )
            }
            Expr::Case { operand, whens, else_ } => {
                let mut s = String::from("CASE");
                if let Some(o) = operand {
                    s.push_str(&format!(" {}", r(a, *o)));
                }
                for (w, t) in whens {
                    s.push_str(&format!(" WHEN {} THEN {}", r(a, *w), r(a, *t)));
                }
                if let Some(e) = else_ {
                    s.push_str(&format!(" ELSE {}", r(a, *e)));
                }
                s.push_str(" END");
                s
            }
            Expr::InList { arg, list, negated } => {
                let items: Vec<String> = list.iter().map(|i| r(a, *i)).collect();
                format!(
                    "({}{} IN [{}])",
                    r(a, *arg),
                    if *negated { " NOT" } else { "" },
                    items.join(", ")
                )
            }
            Expr::Between { arg, low, high, negated } => format!(
                "({}{} BETWEEN {} AND {})",
                r(a, *arg),
                if *negated { " NOT" } else { "" },
                r(a, *low),
                r(a, *high)
            ),
            Expr::IsNull { arg, negated } => {
                format!("({} IS{} NULL)", r(a, *arg), if *negated { " NOT" } else { "" })
            }
            Expr::Like { arg, pattern, negated, escape, ci } => {
                let esc = match escape {
                    Some(c) => format!(" ESCAPE '{}'", *c as char),
                    None => String::new(),
                };
                format!(
                    "({}{} {} {}{})",
                    r(a, *arg),
                    if *negated { " NOT" } else { "" },
                    if *ci { "ILIKE" } else { "LIKE" },
                    r(a, *pattern),
                    esc
                )
            }
            Expr::Function { name, args, distinct, star, filter } => {
                let inner = if *star {
                    "*".to_string()
                } else {
                    let items: Vec<String> = args.iter().map(|i| r(a, *i)).collect();
                    format!("{}{}", if *distinct { "DISTINCT " } else { "" }, items.join(", "))
                };
                let f = match filter {
                    Some(f) => format!(" FILTER (WHERE {})", r(a, *f)),
                    None => String::new(),
                };
                format!("{}({}){}", name, inner, f)
            }
            Expr::Unnest(arg) => format!("UNNEST({})", r(a, *arg)),
            Expr::Lambda { params, body } => {
                let p = if params.len() == 1 {
                    params[0].clone()
                } else {
                    format!("({})", params.join(", "))
                };
                format!("{} -> {}", p, r(a, *body))
            }
        }
    }

    fn lit(v: &Value) -> String {
        match v {
            Value::Null => "NULL".to_string(),
            Value::Bool(b) => format!("{}", b),
            Value::I32(x) => format!("{}i32", x),
            Value::I64(x) => format!("{}i64", x),
            Value::I128(x) => format!("{}i128", x),
            Value::F64(x) => format!("{}f64", x),
            Value::Bytes(b) => format!("'{}'", String::from_utf8_lossy(b)),
        }
    }

    fn op_name(op: BinaryOp) -> &'static str {
        use BinaryOp::*;
        match op {
            Add => "+",
            Sub => "-",
            Mul => "*",
            Div => "/",
            Mod => "%",
            Eq => "=",
            Ne => "!=",
            Lt => "<",
            Le => "<=",
            Gt => ">",
            Ge => ">=",
            And => "AND",
            Or => "OR",
            Concat => "||",
        }
    }

    fn ty_name(ty: Ty) -> String {
        match ty {
            Ty::Decimal { precision, scale } => format!("DECIMAL({},{})", precision, scale),
            other => other.name().to_string(),
        }
    }

    fn from_str(a: &ExprArena, f: &FromItem) -> String {
        fn alias(s: &Option<String>) -> String {
            match s {
                Some(x) => format!(" AS {}", x),
                None => String::new(),
            }
        }
        match f {
            FromItem::Table { name, alias: al } => format!("{}{}", name, alias(al)),
            FromItem::Parquet { path, alias: al } => format!("parquet('{}'){}", path, alias(al)),
            FromItem::Subquery { query, alias: al } => {
                format!("({}){}", query_str(a, query), alias(al))
            }
            FromItem::Join { left, right, kind, on } => {
                let k = match kind {
                    JoinKind::Inner => "INNER",
                    JoinKind::Left => "LEFT",
                    JoinKind::Right => "RIGHT",
                    JoinKind::Full => "FULL",
                    JoinKind::Cross => "CROSS",
                    // SEMI / ANTI 系は構文からは作られない（バインダ専用）。
                    // 種類が増えてもパーサのテストが壊れないよう一括で受ける。
                    _ => "BINDER-ONLY",
                };
                let on_s = match on {
                    Some(e) => format!(" ON {}", r(a, *e)),
                    None => String::new(),
                };
                format!("({} {} JOIN {}{})", from_str(a, left), k, from_str(a, right), on_s)
            }
            FromItem::Unnest { expr, alias: al, column_alias } => {
                let col = match column_alias {
                    Some(c) => format!("({})", c),
                    None => String::new(),
                };
                format!("UNNEST({}){}{}", r(a, *expr), alias(al), col)
            }
            FromItem::GenerateSeries { start, stop, step, inclusive, alias: al, column_alias } => {
                let col = match column_alias {
                    Some(c) => format!("({})", c),
                    None => String::new(),
                };
                let name = if *inclusive { "generate_series" } else { "range" };
                format!("{}({},{},{}){}{}", name, start, stop, step, alias(al), col)
            }
        }
    }

    /// SELECT 文を 1 行に潰す。構造の比較用で、SQL として妥当な形ではない。
    fn select_str(a: &ExprArena, s: &SelectStmt) -> String {
        let mut out = String::from("SELECT");
        if s.distinct {
            out.push_str(" DISTINCT");
        }
        if !s.distinct_on.is_empty() {
            let on: Vec<String> = s.distinct_on.iter().map(|e| r(a, *e)).collect();
            out.push_str(&format!(" DISTINCT ON ({})", on.join(", ")));
        }
        let items: Vec<String> = s
            .items
            .iter()
            .map(|i| match &i.alias {
                Some(al) => format!("{} AS {}", r(a, i.expr), al),
                None => r(a, i.expr),
            })
            .collect();
        out.push_str(&format!(" {}", items.join(", ")));
        if let Some(f) = &s.from {
            out.push_str(&format!(" FROM {}", from_str(a, f)));
        }
        if let Some(w) = s.filter {
            out.push_str(&format!(" WHERE {}", r(a, w)));
        }
        if !s.group_by.is_empty() {
            let g: Vec<String> = s.group_by.iter().map(|e| r(a, *e)).collect();
            out.push_str(&format!(" GROUP BY {}", g.join(", ")));
        }
        if let Some(sets) = &s.grouping_sets {
            let sets_str: Vec<String> = sets
                .iter()
                .map(|set| {
                    let cols: Vec<String> = set.iter().map(|e| r(a, *e)).collect();
                    format!("({})", cols.join(", "))
                })
                .collect();
            out.push_str(&format!(" GROUP BY GROUPING SETS ({})", sets_str.join(", ")));
        }
        if let Some(h) = s.having {
            out.push_str(&format!(" HAVING {}", r(a, h)));
        }
        if !s.windows.is_empty() {
            let ws: Vec<String> = s
                .windows
                .iter()
                .map(|(name, def)| {
                    let mut spec: Vec<String> = Vec::new();
                    if !def.partition_by.is_empty() {
                        let ps: Vec<String> = def.partition_by.iter().map(|e| r(a, *e)).collect();
                        spec.push(format!("PARTITION BY {}", ps.join(", ")));
                    }
                    if !def.order_by.is_empty() {
                        spec.push(format!("ORDER BY {}", order_list(a, &def.order_by)));
                    }
                    format!("{} AS ({})", name, spec.join(" "))
                })
                .collect();
            out.push_str(&format!(" WINDOW {}", ws.join(", ")));
        }
        if let Some(q) = s.qualify {
            out.push_str(&format!(" QUALIFY {}", r(a, q)));
        }
        if !s.order_by.is_empty() {
            out.push_str(&format!(" ORDER BY {}", order_list(a, &s.order_by)));
        }
        if let Some(l) = s.limit {
            out.push_str(&format!(" LIMIT {}", l));
        }
        if let Some(o) = s.offset {
            out.push_str(&format!(" OFFSET {}", o));
        }
        out
    }

    fn order_list(a: &ExprArena, items: &[OrderByItem]) -> String {
        let o: Vec<String> = items
            .iter()
            .map(|i| {
                format!(
                    "{} {} NULLS {}",
                    r(a, i.expr),
                    if i.desc { "DESC" } else { "ASC" },
                    if i.nulls_first { "FIRST" } else { "LAST" }
                )
            })
            .collect();
        o.join(", ")
    }

    /// 集合演算の木。括弧を必ず付けるので結合の向きがそのまま読める。
    fn setexpr_str(a: &ExprArena, e: &SetExpr) -> String {
        match e {
            SetExpr::Select(s) => select_str(a, s),
            SetExpr::SetOp { op, all, left, right } => {
                let name = match op {
                    SetOp::Union => "UNION",
                    SetOp::Intersect => "INTERSECT",
                    SetOp::Except => "EXCEPT",
                };
                format!(
                    "({} {}{} {})",
                    setexpr_str(a, left),
                    name,
                    if *all { " ALL" } else { "" },
                    setexpr_str(a, right)
                )
            }
        }
    }

    /// クエリ全体（CTE + 本体 + 外側の ORDER BY / LIMIT）。
    fn query_str(a: &ExprArena, q: &QueryStmt) -> String {
        let mut out = String::new();
        if !q.ctes.is_empty() {
            let recursive = q.ctes.iter().any(|c| c.recursive);
            let cs: Vec<String> = q
                .ctes
                .iter()
                .map(|c| {
                    let cols = if c.columns.is_empty() {
                        String::new()
                    } else {
                        format!("({})", c.columns.join(", "))
                    };
                    format!("{}{} AS ({})", c.name, cols, query_str(a, &c.query))
                })
                .collect();
            let kw = if recursive { "WITH RECURSIVE " } else { "WITH " };
            out.push_str(&format!("{}{} ", kw, cs.join(", ")));
        }
        out.push_str(&setexpr_str(a, &q.body));
        if !q.order_by.is_empty() {
            out.push_str(&format!(" ORDER BY {}", order_list(a, &q.order_by)));
        }
        if let Some(l) = q.limit {
            out.push_str(&format!(" LIMIT {}", l));
        }
        if let Some(o) = q.offset {
            out.push_str(&format!(" OFFSET {}", o));
        }
        out
    }

    /// `QueryStmt` から素の `SelectStmt` を取り出す（集合演算・CTE 無しを前提）。
    fn plain(q: &QueryStmt) -> &SelectStmt {
        match &q.body {
            SetExpr::Select(s) => s,
            _ => panic!("集合演算を含むクエリ"),
        }
    }

    fn sel(sql: &str) -> String {
        let p = parse(sql).expect("parse failed");
        match &p.stmt {
            Stmt::Select(q) => select_str(&p.arena, plain(q)),
            _ => panic!("not a SELECT"),
        }
    }

    /// クエリ全体を描画する。集合演算・CTE を含む文はこちらで見る。
    fn qs(sql: &str) -> String {
        let p = parse(sql).expect("parse failed");
        match &p.stmt {
            Stmt::Select(q) => query_str(&p.arena, q),
            _ => panic!("not a SELECT"),
        }
    }

    /// 文をパースして `QueryStmt` を取り出す。木の形を直接見るテスト用。
    fn parsed(sql: &str) -> (ExprArena, Box<QueryStmt>) {
        let p = parse(sql).expect("parse failed");
        match p.stmt {
            Stmt::Select(q) => (p.arena, q),
            _ => panic!("not a SELECT"),
        }
    }

    /// `SELECT <expr>` を通して式だけを描画する。
    fn ex(expr: &str) -> String {
        let sql = format!("SELECT {}", expr);
        let p = parse(&sql).expect("parse failed");
        match &p.stmt {
            Stmt::Select(q) => r(&p.arena, plain(q).items[0].expr),
            _ => panic!("not a SELECT"),
        }
    }

    fn code(sql: &str) -> u16 {
        match parse(sql) {
            Ok(_) => 0,
            Err(e) => e.code_u16(),
        }
    }

    fn err_at(sql: &str) -> (u16, u32) {
        match parse(sql) {
            Ok(_) => (0, 0),
            Err(e) => (e.code_u16(), e.pos),
        }
    }

    // --- 字句 ---------------------------------------------------------------

    #[test]
    fn keyword_table_is_sorted_and_complete() {
        // 二分探索の前提: (長さ, 小文字先頭バイト) が単調非減少。
        let key = |n: &[u8]| ((n.len() as u32) << 8) | (n[0] | 0x20) as u32;
        for w in KEYWORDS.windows(2) {
            assert!(key(w[0].0) <= key(w[1].0), "unsorted near {:?}", w[0].0);
        }
        for &(name, kw) in KEYWORDS {
            let upper = name.to_ascii_uppercase();
            assert_eq!(keyword(name), Some(kw));
            assert_eq!(keyword(&upper), Some(kw));
        }
        assert_eq!(keyword(b"nope"), None);
        assert_eq!(keyword(b"a"), None);
    }

    #[test]
    fn comments_and_whitespace() {
        assert_eq!(
            sel("SELECT /* 列 */ a -- 末尾コメント\n FROM t /* 途中 */ WHERE a > 1"),
            "SELECT a FROM t WHERE (a > 1i32)"
        );
        assert_eq!(sel("SELECT 1 --x"), "SELECT 1i32");
        // ブロックコメントは入れ子にしない: 最初の */ で閉じる。
        assert_eq!(sel("SELECT /* /* */ 1"), "SELECT 1i32");
        assert_eq!(code("SELECT /* 閉じない 1"), Code::SyntaxError as u16);
    }

    // --- 優先順位・結合 -----------------------------------------------------

    #[test]
    fn precedence() {
        assert_eq!(ex("a + b * c"), "(a + (b * c))");
        assert_eq!(ex("a * b + c"), "((a * b) + c)");
        assert_eq!(ex("a OR b AND c"), "(a OR (b AND c))");
        assert_eq!(ex("NOT a = b"), "(NOT (a = b))");
        assert_eq!(ex("NOT a AND b"), "((NOT a) AND b)");
        assert_eq!(ex("a = b AND c = d"), "((a = b) AND (c = d))");
        assert_eq!(ex("-a + b"), "((- a) + b)");
        assert_eq!(ex("a || b = c"), "((a || b) = c)");
        assert_eq!(ex("a % b + c"), "((a % b) + c)");
        assert_eq!(ex("(a + b) * c"), "((a + b) * c)");
        // == と <> は = / != の別名。
        assert_eq!(ex("a == b"), "(a = b)");
        assert_eq!(ex("a <> b"), "(a != b)");
        assert_eq!(ex("a != b"), "(a != b)");
        assert_eq!(ex("a <= b"), "(a <= b)");
        assert_eq!(ex("a >= b"), "(a >= b)");
    }

    #[test]
    fn left_associativity() {
        assert_eq!(ex("1 - 2 - 3"), "((1i32 - 2i32) - 3i32)");
        assert_eq!(ex("8 / 4 / 2"), "((8i32 / 4i32) / 2i32)");
        assert_eq!(ex("a AND b AND c"), "((a AND b) AND c)");
        assert_eq!(ex("a || b || c"), "((a || b) || c)");
    }

    #[test]
    fn unary() {
        assert_eq!(ex("-x"), "(- x)");
        assert_eq!(ex("+x"), "x");
        assert_eq!(ex("- -x"), "(- (- x))");
        assert_eq!(ex("-x * y"), "((- x) * y)");
        assert_eq!(ex("NOT NOT a"), "(NOT (NOT a))");
    }

    // --- 述語 ---------------------------------------------------------------

    #[test]
    fn predicates() {
        assert_eq!(ex("x IS NULL"), "(x IS NULL)");
        assert_eq!(ex("x IS NOT NULL"), "(x IS NOT NULL)");
        assert_eq!(ex("x IN (1, 2)"), "(x IN [1i32, 2i32])");
        assert_eq!(ex("x NOT IN (1)"), "(x NOT IN [1i32])");
        assert_eq!(ex("x BETWEEN 1 AND 2"), "(x BETWEEN 1i32 AND 2i32)");
        assert_eq!(ex("x NOT BETWEEN 1 AND 2"), "(x NOT BETWEEN 1i32 AND 2i32)");
        assert_eq!(ex("x LIKE 'a%'"), "(x LIKE 'a%')");
        assert_eq!(ex("x NOT LIKE 'a%'"), "(x NOT LIKE 'a%')");
        assert_eq!(ex("x LIKE 'a!%' ESCAPE '!'"), "(x LIKE 'a!%' ESCAPE '!')");
        // BETWEEN の区切り AND は論理演算子として食わない。
        assert_eq!(ex("a BETWEEN 1 AND 2 AND b"), "((a BETWEEN 1i32 AND 2i32) AND b)");
        assert_eq!(ex("a BETWEEN 1 + 1 AND 2 * 3"), "(a BETWEEN (1i32 + 1i32) AND (2i32 * 3i32))");
        // 述語は比較と同じ強さ。AND より強く結合する。
        assert_eq!(ex("a IS NULL AND b IS NOT NULL"), "((a IS NULL) AND (b IS NOT NULL))");
        assert_eq!(ex("NOT a IS NULL"), "(NOT (a IS NULL))");
        assert_eq!(code("SELECT x IS TRUE"), Code::UnsupportedFeature as u16);
        assert_eq!(code("SELECT x NOT IS NULL"), Code::UnexpectedToken as u16);
    }

    #[test]
    fn ilike_predicate() {
        assert_eq!(ex("x ILIKE 'a%'"), "(x ILIKE 'a%')");
        assert_eq!(ex("x NOT ILIKE 'a%'"), "(x NOT ILIKE 'a%')");
        assert_eq!(ex("x ilike 'a%'"), "(x ILIKE 'a%')");
        assert_eq!(ex("x ILIKE 'a!%' ESCAPE '!'"), "(x ILIKE 'a!%' ESCAPE '!')");
        // ILIKE も述語なので AND より強く結合する。
        assert_eq!(ex("a ILIKE 'x' AND b"), "((a ILIKE 'x') AND b)");
        // `ILIKE` は完全な予約語（列名としては使えない）。
        assert_eq!(code("SELECT ilike FROM t"), Code::UnexpectedToken as u16);
    }

    #[test]
    fn glob_operator_desugars_to_glob_function() {
        assert_eq!(ex("a GLOB 'x*'"), "glob(a, 'x*')");
        assert_eq!(ex("a glob 'x*'"), "glob(a, 'x*')");
        // GLOB も述語と同じ強さ（AND より強く結合する）。
        assert_eq!(ex("a GLOB 'x' AND b"), "(glob(a, 'x') AND b)");
        // DuckDB は `NOT GLOB` を書けない（`duckdb -c "select 'a' NOT GLOB
        // 'b'"` が構文エラーになることを確認済み）。`NOT (x GLOB y)` は
        // 通常の前置 NOT なので問題なく書ける。
        assert_eq!(code("SELECT a NOT GLOB 'x'"), Code::UnexpectedToken as u16);
        assert_eq!(ex("NOT (a GLOB 'x')"), "(NOT glob(a, 'x'))");
        // `glob` は予約語ではない（ROWS/RANGE/QUALIFY と同じ「文脈依存
        // キーワード」方式）ので、列名としてそのまま使える。
        assert_eq!(ex("glob"), "glob");
        assert_eq!(code("SELECT glob FROM t"), 0);
    }

    #[test]
    fn similar_to_desugars_to_regexp_full_match() {
        assert_eq!(ex("a SIMILAR TO 'x.y'"), "regexp_full_match(a, 'x.y')");
        assert_eq!(ex("a similar to 'x.y'"), "regexp_full_match(a, 'x.y')");
        assert_eq!(ex("a NOT SIMILAR TO 'x.y'"), "(NOT regexp_full_match(a, 'x.y'))");
        // LIKE と同じく述語の強さ（AND より強く結合する）。
        assert_eq!(ex("a SIMILAR TO 'x' AND b"), "(regexp_full_match(a, 'x') AND b)");
        // DuckDB 自身も `ESCAPE` 句は「未実装」として拒否する
        // （`duckdb -c "select 'a' similar to 'a' escape '\\'"` を確認済み）。
        assert_eq!(code(r"SELECT a SIMILAR TO 'x' ESCAPE '\'"), Code::UnsupportedFeature as u16);
        // `similar`/`to` はどちらも予約語ではないので、列名としてそのまま
        // 使える（過去の ROWS/RANGE/QUALIFY の教訓を踏まえた設計）。
        assert_eq!(ex("similar"), "similar");
        assert_eq!(code("SELECT similar, glob FROM t"), 0);
    }

    #[test]
    fn array_literal_desugars_to_list_value() {
        assert_eq!(ex("[1, 2, 3]"), "list_value(1i32, 2i32, 3i32)");
        assert_eq!(ex("['a', 'b']"), "list_value('a', 'b')");
        assert_eq!(ex("[1 + 1]"), "list_value((1i32 + 1i32))");
        // 空配列は `list_value()` を経由せず（`resolve` が 0 引数を拒否する
        // 設計になっているため）、JSON の空配列を直接 TypedLiteral として
        // 埋め込む。`duckdb -c "select []"` が有効な式であることを確認済み。
        assert_eq!(ex("[]"), "Bytes([91, 93])::JSON");
        // 添字アクセス（`expr[i]`）は今回のスコープ外。式の開始位置以外の
        // `[` は構文エラーになる（`list_value` への糖衣構文と衝突しない）。
        assert_eq!(code("SELECT a[1]"), Code::UnexpectedToken as u16);
    }

    // --- CASE / CAST / 関数 -------------------------------------------------

    #[test]
    fn case_expr() {
        assert_eq!(
            ex("CASE WHEN a > 1 THEN 'x' ELSE 'y' END"),
            "CASE WHEN (a > 1i32) THEN 'x' ELSE 'y' END"
        );
        assert_eq!(ex("CASE WHEN a THEN 1 END"), "CASE WHEN a THEN 1i32 END");
        assert_eq!(
            ex("CASE a WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'many' END"),
            "CASE a WHEN 1i32 THEN 'one' WHEN 2i32 THEN 'two' ELSE 'many' END"
        );
        assert_eq!(code("SELECT CASE a END"), Code::UnexpectedToken as u16);
        assert_eq!(code("SELECT CASE WHEN a THEN 1"), Code::UnexpectedToken as u16);
    }

    #[test]
    fn cast_expr() {
        assert_eq!(ex("CAST(x AS DECIMAL(10,2))"), "CAST(x AS DECIMAL(10,2))");
        assert_eq!(ex("CAST(x AS decimal)"), "CAST(x AS DECIMAL(18,3))");
        assert_eq!(ex("CAST(x AS NUMERIC(38,0))"), "CAST(x AS DECIMAL(38,0))");
        assert_eq!(ex("CAST(x AS INT)"), "CAST(x AS INTEGER)");
        assert_eq!(ex("CAST(x AS integer)"), "CAST(x AS INTEGER)");
        assert_eq!(ex("CAST(x AS BigInt)"), "CAST(x AS BIGINT)");
        assert_eq!(ex("CAST(x AS TEXT)"), "CAST(x AS VARCHAR)");
        assert_eq!(ex("CAST(x AS bytea)"), "CAST(x AS BLOB)");
        assert_eq!(ex("CAST(x AS DATETIME)"), "CAST(x AS TIMESTAMP)");
        assert_eq!(ex("CAST(x AS BOOL)"), "CAST(x AS BOOLEAN)");
        assert_eq!(ex("CAST(x AS REAL)"), "CAST(x AS FLOAT)");
        assert_eq!(ex("CAST(x AS UBIGINT)"), "CAST(x AS UBIGINT)");
        assert_eq!(code("SELECT CAST(x AS FROB)"), Code::InvalidCast as u16);
        assert_eq!(code("SELECT CAST(x AS DECIMAL(99,1))"), Code::InvalidCast as u16);
        assert_eq!(code("SELECT CAST(x AS NULL)"), Code::InvalidCast as u16);
        assert_eq!(ex("CAST(x AS JSON)"), "CAST(x AS JSON)");
        assert_eq!(ex("CAST(x AS json)"), "CAST(x AS JSON)");
    }

    #[test]
    fn json_arrow_operators_desugar_to_function_calls() {
        // `->`/`->>` は新しい BinaryOp を増やさず、`json_extract`/
        // `json_extract_string` 呼び出しへの糖衣構文として展開される。
        assert_eq!(ex("a -> b"), "json_extract(a, b)");
        assert_eq!(ex("a ->> b"), "json_extract_string(a, b)");
        // 左結合で連鎖できる（`a -> b -> c` = `(a -> b) -> c`）。
        assert_eq!(ex("a -> b -> c"), "json_extract(json_extract(a, b), c)");
        // Postgres の「その他の演算子」band と同じく `||` と同じ強さなので、
        // 比較より強く結合する（`doc -> 'a' = 1` に括弧が要らない）。
        assert_eq!(ex("a -> b = c"), "(json_extract(a, b) = c)");
        assert_eq!(ex("a ->> b = c"), "(json_extract_string(a, b) = c)");
        // `-` の単純な引き算とは区別される。
        assert_eq!(ex("a - b"), "(a - b)");
    }

    #[test]
    fn lambda_is_recognized_only_in_list_transform_filter_reduce_arg_position() {
        // 単一引数は括弧無し。
        assert_eq!(ex("list_transform(a, x -> x + 1)"), "list_transform(a, x -> (x + 1i32))");
        // 複数引数は括弧付き。
        assert_eq!(
            ex("list_reduce(a, (acc, x) -> acc + x)"),
            "list_reduce(a, (acc, x) -> (acc + x))"
        );
        // 括弧付き単一引数も許す（duckdb と同じ）。
        assert_eq!(ex("list_filter(a, (x) -> x > 1)"), "list_filter(a, x -> (x > 1i32))");
        // ネストしたラムダ。
        assert_eq!(
            ex("list_transform(a, y -> list_transform(y, x -> x * 2))"),
            "list_transform(a, y -> list_transform(y, x -> (x * 2i32)))"
        );
        // list_filter も同じ扱い。
        assert_eq!(ex("list_filter(a, x -> x > 5)"), "list_filter(a, x -> (x > 5i32))");
        // 大文字小文字を区別しない関数名判定。
        assert_eq!(ex("LIST_TRANSFORM(a, x -> x)"), "LIST_TRANSFORM(a, x -> x)");

        // それ以外の関数の引数位置では `->` は今まで通り JSON パス演算子の
        // ままで、ラムダとしては解釈されない（duckdb CLI で実測済み:
        // `coalesce(doc -> 'a', 'x')` は JSON 抽出のまま解決される）。
        assert_eq!(ex("coalesce(doc -> 'a', x)"), "coalesce(json_extract(doc, 'a'), x)");
    }

    #[test]
    fn try_cast_expr() {
        assert_eq!(ex("TRY_CAST(x AS INTEGER)"), "TRY_CAST(x AS INTEGER)");
        assert_eq!(ex("try_cast(x AS DECIMAL(10,2))"), "TRY_CAST(x AS DECIMAL(10,2))");
        // 通常の CAST とは別ノードとして区別される。
        assert_ne!(ex("CAST(x AS INTEGER)"), ex("TRY_CAST(x AS INTEGER)"));
        assert_eq!(code("SELECT TRY_CAST(x AS FROB)"), Code::InvalidCast as u16);
        // `try_cast` は予約語ではないので、通常の識別子（列名・関数名）としても
        // 使い続けられる（`(` が続かなければ普通の列参照）。
        assert_eq!(ex("try_cast"), "try_cast");
    }

    #[test]
    fn iif_desugars_to_case() {
        assert_eq!(ex("IIF(a > 1, 'x', 'y')"), "CASE WHEN (a > 1i32) THEN 'x' ELSE 'y' END");
        assert_eq!(ex("iif(a, 1, 2)"), "CASE WHEN a THEN 1i32 ELSE 2i32 END");
        assert_eq!(code("SELECT IIF(a, 1)"), Code::UnexpectedToken as u16);
        // `iif` も予約語ではない。
        assert_eq!(ex("iif"), "iif");
    }

    #[test]
    fn interval_literals() {
        // 複合文字列形式。パック結果をそのまま比較する。
        let lit = |m: i32, d: i32, u: i64| {
            format!("INTERVAL({}i128)", crate::vector::pack_interval(m, d, u))
        };
        assert_eq!(ex("INTERVAL '3 days'"), lit(0, 3, 0));
        assert_eq!(ex("INTERVAL '1 year 2 months 3 days'"), ex("INTERVAL '14 months 3 days'"),);
        // 単数形・複数形・大文字小文字を区別しない。
        assert_eq!(ex("interval '1 DAY'"), ex("INTERVAL '1 days'"));
        assert_eq!(ex("INTERVAL '1 month'"), ex("INTERVAL '1 months'"));
        // 負値（文字列内の符号）。
        assert_eq!(ex("INTERVAL '-3 days'"), lit(0, -3, 0));
        // 単項マイナスは `Unary::Neg` で包まれる（別ノード）ので、専用カーネルに
        // 展開されることは `plan::compile` 側のテストで確認する。ここではパース
        // が通ることだけ見る。
        assert_eq!(ex("-INTERVAL '3 days'"), format!("(- {})", lit(0, 3, 0)));
        // `'n' UNIT` 形式。
        assert_eq!(ex("INTERVAL '3' DAY"), ex("INTERVAL '3 days'"));
        assert_eq!(ex("INTERVAL '1' MONTH"), ex("INTERVAL '1 month'"));
        // 引用符無しの `n UNIT` 形式。
        assert_eq!(ex("INTERVAL 3 DAY"), ex("INTERVAL '3 days'"));
        // 秒未満の単位。
        assert_eq!(ex("INTERVAL '1500 milliseconds'"), ex("INTERVAL '1500000 microseconds'"));
        // `interval` は予約語ではないので列参照としても使える。
        assert_eq!(ex("interval"), "interval");
        assert_eq!(ex("interval + 1"), "(interval + 1i32)");
        assert_eq!(code("SELECT INTERVAL 'nonsense'"), Code::SyntaxError as u16);
        assert_eq!(code("SELECT INTERVAL '3 fortnights'"), Code::SyntaxError as u16);
    }

    #[test]
    fn functions() {
        assert_eq!(ex("count(*)"), "count(*)");
        assert_eq!(ex("count(DISTINCT x)"), "count(DISTINCT x)");
        assert_eq!(ex("now()"), "now()");
        assert_eq!(ex("upper(substr(s, 1, 3))"), "upper(substr(s, 1i32, 3i32))");
        assert_eq!(ex("abs(-a) + round(b, 2)"), "(abs((- a)) + round(b, 2i32))");
        assert_eq!(code("SELECT f(1,"), Code::UnexpectedToken as u16);
    }

    #[test]
    fn filter_clause() {
        assert_eq!(ex("count(*) FILTER (WHERE a > 1)"), "count(*) FILTER (WHERE (a > 1i32))");
        assert_eq!(
            ex("sum(x) FILTER (WHERE a > 1 AND b)"),
            "sum(x) FILTER (WHERE ((a > 1i32) AND b))"
        );
        assert_eq!(ex("count(*)"), "count(*)", "FILTER 無しは今までどおり");
        // `FILTER` は予約語ではないので、次が `(` でなければ普通の別名として通る。
        assert_eq!(sel("SELECT count(*) filter FROM t"), "SELECT count(*) AS filter FROM t");
        // `FILTER` はウィンドウ関数には未対応（範囲外）。
        assert_eq!(
            code("SELECT count(*) FILTER (WHERE a > 1) OVER () FROM t"),
            Code::UnsupportedFeature as u16
        );
    }

    #[test]
    fn column_refs_and_params() {
        assert_eq!(ex("a"), "a");
        assert_eq!(ex("t.a"), "t.a");
        assert_eq!(ex("\"Mixed Case\".\"x\"\"y\""), "Mixed Case.x\"y");
        let p = parse("SELECT ? WHERE a = ? AND b = ?").expect("parse");
        assert_eq!(p.num_params, 3);
        match &p.stmt {
            Stmt::Select(q) => {
                let s = plain(q);
                assert_eq!(r(&p.arena, s.items[0].expr), "?0");
                assert_eq!(r(&p.arena, s.filter.expect("filter")), "((a = ?1) AND (b = ?2))");
            }
            _ => panic!("not select"),
        }
        assert_eq!(parse("SELECT 1").expect("parse").num_params, 0);
    }

    // --- リテラル -----------------------------------------------------------

    #[test]
    fn integer_literal_widths() {
        assert_eq!(ex("2147483647"), "2147483647i32");
        assert_eq!(ex("2147483648"), "2147483648i64");
        assert_eq!(ex("-2147483648"), "-2147483648i32");
        assert_eq!(ex("-2147483649"), "-2147483649i64");
        assert_eq!(ex("9223372036854775807"), "9223372036854775807i64");
        assert_eq!(ex("-9223372036854775808"), "-9223372036854775808i64");
        assert_eq!(ex("9223372036854775808"), "9223372036854775808i128");
        assert_eq!(
            ex("170141183460469231731687303715884105727"),
            "170141183460469231731687303715884105727i128"
        );
        assert_eq!(
            ex("-170141183460469231731687303715884105728"),
            "-170141183460469231731687303715884105728i128"
        );
        assert_eq!(
            code("SELECT 170141183460469231731687303715884105728"),
            Code::NumberOverflow as u16
        );
        assert_eq!(
            code("SELECT 99999999999999999999999999999999999999999"),
            Code::NumberOverflow as u16
        );
        // 単項マイナスは畳むが、式に対しては通常の演算子のまま。
        assert_eq!(ex("-(1)"), "(- 1i32)");
    }

    #[test]
    fn other_literals() {
        assert_eq!(ex("1.5"), "1.5f64");
        assert_eq!(ex("1e3"), "1000f64");
        assert_eq!(ex("1.5e-2"), "0.015f64");
        assert_eq!(ex("TRUE"), "true");
        assert_eq!(ex("false"), "false");
        assert_eq!(ex("NULL"), "NULL");
        assert_eq!(ex("'abc'"), "'abc'");
        assert_eq!(ex("''"), "''");
        assert_eq!(ex("'it''s'"), "'it's'");
        assert_eq!(ex("'a''''b'"), "'a''b'");
        assert_eq!(code("SELECT 'abc"), Code::UnterminatedString as u16);
        assert_eq!(code("SELECT 'it''s"), Code::UnterminatedString as u16);
    }

    // --- 文全体 -------------------------------------------------------------

    #[test]
    fn full_select() {
        assert_eq!(
            sel("SELECT DISTINCT a AS x, b y, t.*, count(*) FROM t WHERE a > 1 GROUP BY a, b HAVING count(*) > 2 ORDER BY a DESC, b ASC NULLS FIRST LIMIT 10 OFFSET 5"),
            "SELECT DISTINCT a AS x, b AS y, t.*, count(*) FROM t WHERE (a > 1i32) \
             GROUP BY a, b HAVING (count(*) > 2i32) \
             ORDER BY a DESC NULLS LAST, b ASC NULLS FIRST LIMIT 10 OFFSET 5"
        );
        // The default NULL order is always LAST, for both ASC and DESC
        // (matches DuckDB, not the SQL-standard/PostgreSQL convention).
        assert_eq!(sel("SELECT a FROM t ORDER BY a"), "SELECT a FROM t ORDER BY a ASC NULLS LAST");
        assert_eq!(
            sel("SELECT a FROM t ORDER BY a DESC NULLS LAST"),
            "SELECT a FROM t ORDER BY a DESC NULLS LAST"
        );
        assert_eq!(sel("SELECT * FROM t;"), "SELECT * FROM t");
        assert_eq!(sel("select 1"), "SELECT 1i32");
    }

    #[test]
    fn qualify_clause() {
        // WHERE/GROUP BY/HAVING を挟まず、テーブル参照の直後に `QUALIFY` が
        // 来ても、テーブル別名として食われずに正しく句として解釈される
        // （過去の ROWS/RANGE 事故と同じ罠。`QUALIFY` を完全予約語にした理由）。
        assert_eq!(sel("SELECT a FROM t QUALIFY a > 1"), "SELECT a FROM t QUALIFY (a > 1i32)");
        // GROUP BY / HAVING の後、ORDER BY の前という置き場所。
        assert_eq!(
            sel("SELECT a, count(*) FROM t GROUP BY a HAVING count(*) > 1 QUALIFY a > 0 ORDER BY a"),
            "SELECT a, count(*) FROM t GROUP BY a HAVING (count(*) > 1i32) QUALIFY (a > 0i32) ORDER BY a ASC NULLS LAST"
        );
        // 完全な予約語なので列名としては使えない。
        assert_eq!(code("SELECT qualify FROM t"), Code::UnexpectedToken as u16);
    }

    #[test]
    fn star_exclude_replace() {
        // 基本形。順序は EXCLUDE → REPLACE 固定（`duckdb` で確認済み）。
        assert_eq!(sel("SELECT * EXCLUDE (b) FROM t"), "SELECT * EXCLUDE (b) FROM t");
        assert_eq!(
            sel("SELECT * REPLACE (a + 1 AS a) FROM t"),
            "SELECT * REPLACE ((a + 1i32) AS a) FROM t"
        );
        assert_eq!(
            sel("SELECT * EXCLUDE (b) REPLACE (a + 1 AS a) FROM t"),
            "SELECT * EXCLUDE (b) REPLACE ((a + 1i32) AS a) FROM t"
        );
        // 逆順（REPLACE の後に EXCLUDE）は `duckdb` と同じく構文エラー。
        assert_eq!(
            code("SELECT * REPLACE (a + 1 AS a) EXCLUDE (b) FROM t"),
            Code::UnexpectedToken as u16
        );
        // 1 個だけなら括弧を省略できる（`duckdb` の挙動）。
        assert_eq!(sel("SELECT * EXCLUDE b FROM t"), "SELECT * EXCLUDE (b) FROM t");
        assert_eq!(sel("SELECT * REPLACE 1 AS a FROM t"), "SELECT * REPLACE (1i32 AS a) FROM t");
        // 複数列は括弧必須。
        assert_eq!(sel("SELECT * EXCLUDE (a, b) FROM t"), "SELECT * EXCLUDE (a, b) FROM t");
        assert_eq!(
            sel("SELECT * REPLACE (1 AS a, 2 AS b) FROM t"),
            "SELECT * REPLACE (1i32 AS a, 2i32 AS b) FROM t"
        );
        // `t.*` にも同様に付けられる。
        assert_eq!(sel("SELECT t.* EXCLUDE (b) FROM t"), "SELECT t.* EXCLUDE (b) FROM t");
        // 同じ列を EXCLUDE と REPLACE の両方に書くのは無意味なので拒否する。
        assert_eq!(code("SELECT * EXCLUDE (a) REPLACE (1 AS a) FROM t"), Code::SyntaxError as u16);
        // EXCLUDE 内の重複、REPLACE 内の重複もそれぞれ拒否する。
        assert_eq!(code("SELECT * EXCLUDE (a, a) FROM t"), Code::SyntaxError as u16);
        assert_eq!(code("SELECT * REPLACE (1 AS a, 2 AS a) FROM t"), Code::SyntaxError as u16);
        // `EXCLUDE` は `*` の直後という文脈でしかキーワードにならない。過去の
        // ROWS/RANGE/QUALIFY 事故と同種の罠なので、通常の列名・別名として
        // なお使えることを確認する。`REPLACE` は `ddl` フィーチャが有効だと
        // `CREATE OR REPLACE` 用に別途グローバル予約語になるため、その場合は
        // 対象外（`is_star_replace_kw` のコメント参照）。
        assert_eq!(sel("SELECT exclude FROM t"), "SELECT exclude FROM t");
        assert_eq!(sel("SELECT a AS exclude FROM t"), "SELECT a AS exclude FROM t");
        #[cfg(not(feature = "ddl"))]
        assert_eq!(sel("SELECT exclude, replace FROM t"), "SELECT exclude, replace FROM t");
    }

    #[test]
    fn grouping_sets_rollup_cube_syntax() {
        // `GROUPING SETS` はそのまま集合の並びを保つ。空集合 `()` も 1 要素。
        assert_eq!(
            sel("SELECT a, b, sum(c) FROM t GROUP BY GROUPING SETS ((a, b), (a), ())"),
            "SELECT a, b, sum(c) FROM t GROUP BY GROUPING SETS ((a, b), (a), ())"
        );
        // `ROLLUP (a, b, c)` は列の多い方から少ない方への階層的な部分集合に展開される。
        assert_eq!(
            sel("SELECT a, b, c, sum(d) FROM t GROUP BY ROLLUP (a, b, c)"),
            "SELECT a, b, c, sum(d) FROM t GROUP BY GROUPING SETS ((a, b, c), (a, b), (a), ())"
        );
        // `CUBE (a, b)` は全部分集合（2^n 個）に展開される。
        assert_eq!(
            sel("SELECT a, b, sum(c) FROM t GROUP BY CUBE (a, b)"),
            "SELECT a, b, sum(c) FROM t GROUP BY GROUPING SETS ((a, b), (a), (b), ())"
        );
        // 単一列の ROLLUP/CUBE。
        assert_eq!(
            sel("SELECT a, sum(c) FROM t GROUP BY ROLLUP (a)"),
            "SELECT a, sum(c) FROM t GROUP BY GROUPING SETS ((a), ())"
        );

        // `GROUPING`/`SETS`/`ROLLUP`/`CUBE` は GROUP BY 直後という文脈でしか
        // キーワードにならない。過去の ROWS/RANGE/QUALIFY 事故と同種の罠なので、
        // 通常の列名・別名としてなお使えることを確認する
        // （`SETS` は他のどの文脈でも特別扱いしないので確認不要）。
        assert_eq!(sel("SELECT grouping FROM t"), "SELECT grouping FROM t");
        assert_eq!(sel("SELECT rollup, cube FROM t"), "SELECT rollup, cube FROM t");
        assert_eq!(sel("SELECT a AS rollup FROM t"), "SELECT a AS rollup FROM t");
        assert_eq!(
            sel("SELECT a FROM t WHERE grouping > 0"),
            "SELECT a FROM t WHERE (grouping > 0i32)"
        );

        // `GROUPING(...)` は普通の関数呼び出しとして通る。
        assert_eq!(ex("grouping(a)"), "grouping(a)");
        assert_eq!(ex("grouping(a, b)"), "grouping(a, b)");
    }

    #[test]
    fn distinct_on_clause() {
        assert_eq!(sel("SELECT DISTINCT ON (a) a, b FROM t"), "SELECT DISTINCT ON (a) a, b FROM t");
        assert_eq!(sel("SELECT DISTINCT ON (a, b) * FROM t"), "SELECT DISTINCT ON (a, b) * FROM t");
        // `DISTINCT ON` は普通の `DISTINCT` とは排他（同時には立たない）。
        let p = parse("SELECT DISTINCT ON (a) a FROM t").expect("parse");
        match &p.stmt {
            Stmt::Select(q) => {
                let s = plain(q);
                assert!(!s.distinct);
                assert_eq!(s.distinct_on.len(), 1);
            }
            _ => panic!("not select"),
        }
        assert_eq!(code("SELECT DISTINCT ON () a FROM t"), Code::UnexpectedToken as u16);
    }

    #[test]
    fn from_items_and_joins() {
        assert_eq!(sel("SELECT * FROM t a"), "SELECT * FROM t AS a");
        assert_eq!(sel("SELECT * FROM t AS a"), "SELECT * FROM t AS a");
        assert_eq!(
            sel("SELECT * FROM parquet('path/to.parquet') AS t"),
            "SELECT * FROM parquet('path/to.parquet') AS t"
        );
        assert_eq!(
            sel("SELECT * FROM PARQUET('a''b.parquet')"),
            "SELECT * FROM parquet('a'b.parquet')"
        );
        assert_eq!(
            sel("SELECT * FROM (SELECT a FROM t WHERE a > 0) s"),
            "SELECT * FROM (SELECT a FROM t WHERE (a > 0i32)) AS s"
        );
        assert_eq!(
            sel("SELECT * FROM a JOIN b ON a.x = b.x"),
            "SELECT * FROM (a INNER JOIN b ON (a.x = b.x))"
        );
        assert_eq!(
            sel("SELECT * FROM a INNER JOIN b ON a.x = b.x"),
            "SELECT * FROM (a INNER JOIN b ON (a.x = b.x))"
        );
        assert_eq!(
            sel("SELECT * FROM a LEFT OUTER JOIN b ON a.x = b.x"),
            "SELECT * FROM (a LEFT JOIN b ON (a.x = b.x))"
        );
        assert_eq!(
            sel("SELECT * FROM a RIGHT JOIN b ON a.x = b.x"),
            "SELECT * FROM (a RIGHT JOIN b ON (a.x = b.x))"
        );
        assert_eq!(
            sel("SELECT * FROM a FULL OUTER JOIN b ON a.x = b.x"),
            "SELECT * FROM (a FULL JOIN b ON (a.x = b.x))"
        );
        assert_eq!(sel("SELECT * FROM a CROSS JOIN b"), "SELECT * FROM (a CROSS JOIN b)");
        // カンマ結合は暗黙の CROSS JOIN。左深に積む。
        assert_eq!(sel("SELECT * FROM a, b, c"), "SELECT * FROM ((a CROSS JOIN b) CROSS JOIN c)");
        assert_eq!(
            sel("SELECT * FROM a x JOIN b y ON x.k = y.k LEFT JOIN c ON c.k = x.k"),
            "SELECT * FROM ((a AS x INNER JOIN b AS y ON (x.k = y.k)) LEFT JOIN c ON (c.k = x.k))"
        );
        assert_eq!(code("SELECT * FROM a JOIN b"), Code::UnexpectedToken as u16);
        assert_eq!(code("SELECT * FROM foo(1)"), Code::UnsupportedFeature as u16);
    }

    // --- CTE ----------------------------------------------------------------

    #[test]
    fn ctes() {
        assert_eq!(
            qs("WITH a AS (SELECT 1) SELECT * FROM a"),
            "WITH a AS (SELECT 1i32) SELECT * FROM a"
        );
        // 複数 CTE は定義順に並ぶ。
        assert_eq!(
            qs("WITH a AS (SELECT x FROM t), b AS (SELECT y FROM u) SELECT * FROM a, b"),
            "WITH a AS (SELECT x FROM t), b AS (SELECT y FROM u) \
             SELECT * FROM (a CROSS JOIN b)"
        );
        // 後ろの CTE が前の CTE を参照できる（前方参照は binder 側の責務）。
        assert_eq!(
            qs("WITH a AS (SELECT x FROM t), b AS (SELECT x FROM a) SELECT * FROM b"),
            "WITH a AS (SELECT x FROM t), b AS (SELECT x FROM a) SELECT * FROM b"
        );
        // CTE の中身も完全なクエリ。集合演算も ORDER BY も書ける。
        assert_eq!(
            qs("WITH a AS (SELECT 1 UNION SELECT 2) SELECT * FROM a"),
            "WITH a AS ((SELECT 1i32 UNION SELECT 2i32)) SELECT * FROM a"
        );
        assert_eq!(
            qs("WITH a AS (SELECT x FROM t ORDER BY x LIMIT 3) SELECT * FROM a"),
            "WITH a AS (SELECT x FROM t ORDER BY x ASC NULLS LAST LIMIT 3) SELECT * FROM a"
        );
        // 入れ子の CTE。
        assert_eq!(
            qs("WITH a AS (WITH b AS (SELECT 1) SELECT * FROM b) SELECT * FROM a"),
            "WITH a AS (WITH b AS (SELECT 1i32) SELECT * FROM b) SELECT * FROM a"
        );
        // EXPLAIN の下にも CTE を置ける。
        assert!(matches!(
            parse("EXPLAIN WITH a AS (SELECT 1) SELECT * FROM a").expect("parse").stmt,
            Stmt::Explain(_)
        ));
    }

    // --- WITH RECURSIVE -------------------------------------------------------

    #[test]
    fn with_recursive_parses() {
        // `RECURSIVE` は WITH 直後だけの文脈依存キーワード。実際に自分自身を
        // 参照するかどうかは束縛時の判定（`plan::bind`）なので、パーサは
        // フラグを立てるだけで本文の形は問わない。
        assert_eq!(
            qs("WITH RECURSIVE x AS (SELECT 1) SELECT 1"),
            "WITH RECURSIVE x AS (SELECT 1i32) SELECT 1i32"
        );
        // 列名リストは `WITH RECURSIVE` の下でだけ許す。
        assert_eq!(
            qs("WITH RECURSIVE fib(n, a, b) AS \
                 (SELECT 0, 0, 1 UNION ALL SELECT n+1, b, a+b FROM fib WHERE n < 10) \
                 SELECT * FROM fib"),
            "WITH RECURSIVE fib(n, a, b) AS \
             ((SELECT 0i32, 0i32, 1i32 UNION ALL SELECT (n + 1i32), b, (a + b) FROM fib \
             WHERE (n < 10i32))) SELECT * FROM fib"
        );
        // `RECURSIVE` はリスト全体に効く。一部の CTE が実際には非再帰でもよい。
        assert_eq!(
            qs("WITH RECURSIVE base AS (SELECT 1 AS x), \
                 t AS (SELECT x AS n FROM base UNION ALL SELECT n+1 FROM t WHERE n < 5) \
                 SELECT * FROM t"),
            "WITH RECURSIVE base AS (SELECT 1i32 AS x), \
             t AS ((SELECT x AS n FROM base UNION ALL SELECT (n + 1i32) FROM t \
             WHERE (n < 5i32))) SELECT * FROM t"
        );
        // `recursive` という名前の CTE も従来どおり書ける（`AS` が直後に
        // 続くので、キーワードとしては消費されない）。
        assert_eq!(
            qs("WITH recursive AS (SELECT 1) SELECT * FROM recursive"),
            "WITH recursive AS (SELECT 1i32) SELECT * FROM recursive"
        );
        // ただし列名リストは「`RECURSIVE` の後の名前」と区別が付かないため、
        // `WITH recursive(a) AS (...)` は列名リスト側の判定
        // （`RECURSIVE` 無しでは未対応）に落ちる。DuckDB は許すが、この
        // エンジンでは非再帰 CTE の列名リストをそもそもサポートしないので
        // 一貫してエラーになる。
        assert_eq!(
            code("WITH recursive(a) AS (SELECT 1) SELECT * FROM recursive"),
            Code::UnsupportedFeature as u16
        );
    }

    // --- 集合演算 -----------------------------------------------------------

    #[test]
    fn set_operations() {
        assert_eq!(
            qs("SELECT a FROM t UNION SELECT b FROM u"),
            "(SELECT a FROM t UNION SELECT b FROM u)"
        );
        assert_eq!(
            qs("SELECT a FROM t UNION ALL SELECT b FROM u"),
            "(SELECT a FROM t UNION ALL SELECT b FROM u)"
        );
        assert_eq!(qs("SELECT 1 INTERSECT SELECT 2"), "(SELECT 1i32 INTERSECT SELECT 2i32)");
        assert_eq!(
            qs("SELECT 1 INTERSECT ALL SELECT 2"),
            "(SELECT 1i32 INTERSECT ALL SELECT 2i32)"
        );
        assert_eq!(qs("SELECT 1 EXCEPT SELECT 2"), "(SELECT 1i32 EXCEPT SELECT 2i32)");
        assert_eq!(qs("SELECT 1 EXCEPT ALL SELECT 2"), "(SELECT 1i32 EXCEPT ALL SELECT 2i32)");
        // 括弧付きの項。
        assert_eq!(qs("(SELECT 1) UNION (SELECT 2)"), "(SELECT 1i32 UNION SELECT 2i32)");
    }

    #[test]
    fn set_operation_precedence() {
        // INTERSECT は UNION / EXCEPT より強く結合する。
        assert_eq!(
            qs("SELECT 1 UNION SELECT 2 INTERSECT SELECT 3"),
            "(SELECT 1i32 UNION (SELECT 2i32 INTERSECT SELECT 3i32))"
        );
        assert_eq!(
            qs("SELECT 1 INTERSECT SELECT 2 UNION SELECT 3"),
            "((SELECT 1i32 INTERSECT SELECT 2i32) UNION SELECT 3i32)"
        );
        assert_eq!(
            qs("SELECT 1 EXCEPT SELECT 2 INTERSECT SELECT 3"),
            "(SELECT 1i32 EXCEPT (SELECT 2i32 INTERSECT SELECT 3i32))"
        );
        // UNION / EXCEPT は左結合。EXCEPT は結合的でないのでここが要点。
        assert_eq!(
            qs("SELECT 1 EXCEPT SELECT 2 EXCEPT SELECT 3"),
            "((SELECT 1i32 EXCEPT SELECT 2i32) EXCEPT SELECT 3i32)"
        );
        assert_eq!(
            qs("SELECT 1 UNION SELECT 2 EXCEPT SELECT 3"),
            "((SELECT 1i32 UNION SELECT 2i32) EXCEPT SELECT 3i32)"
        );
        assert_eq!(
            qs("SELECT 1 INTERSECT SELECT 2 INTERSECT SELECT 3"),
            "((SELECT 1i32 INTERSECT SELECT 2i32) INTERSECT SELECT 3i32)"
        );
        // 括弧で結合の向きを変えられる。
        assert_eq!(
            qs("SELECT 1 EXCEPT (SELECT 2 EXCEPT SELECT 3)"),
            "(SELECT 1i32 EXCEPT (SELECT 2i32 EXCEPT SELECT 3i32))"
        );
        assert_eq!(
            qs("(SELECT 1 UNION SELECT 2) INTERSECT SELECT 3"),
            "((SELECT 1i32 UNION SELECT 2i32) INTERSECT SELECT 3i32)"
        );

        // 木の形を直接確認する（描画の括弧だけに頼らない）。
        let (_, q) = parsed("SELECT 1 UNION SELECT 2 INTERSECT SELECT 3");
        match &q.body {
            SetExpr::SetOp { op, left, right, .. } => {
                assert_eq!(*op, SetOp::Union);
                assert!(matches!(**left, SetExpr::Select(_)));
                assert!(matches!(**right, SetExpr::SetOp { op: SetOp::Intersect, .. }));
            }
            _ => panic!("集合演算ではない"),
        }
        let (_, q) = parsed("SELECT 1 EXCEPT SELECT 2 EXCEPT SELECT 3");
        match &q.body {
            SetExpr::SetOp { op, left, right, .. } => {
                assert_eq!(*op, SetOp::Except);
                assert!(matches!(**left, SetExpr::SetOp { op: SetOp::Except, .. }));
                assert!(matches!(**right, SetExpr::Select(_)));
            }
            _ => panic!("集合演算ではない"),
        }
    }

    #[test]
    fn trailing_clauses_placement() {
        // 集合演算の後ろの ORDER BY / LIMIT / OFFSET は外側の QueryStmt に付く。
        let (_, q) = parsed("SELECT a FROM t UNION SELECT b FROM u ORDER BY 1 LIMIT 5 OFFSET 2");
        assert_eq!(q.order_by.len(), 1);
        assert_eq!(q.limit, Some(5));
        assert_eq!(q.offset, Some(2));
        match &q.body {
            SetExpr::SetOp { left, right, .. } => {
                for side in [left, right] {
                    match &**side {
                        SetExpr::Select(s) => {
                            assert!(s.order_by.is_empty());
                            assert!(s.limit.is_none());
                            assert!(s.offset.is_none());
                        }
                        _ => panic!("SELECT ではない"),
                    }
                }
            }
            _ => panic!("集合演算ではない"),
        }
        assert_eq!(
            qs("SELECT a FROM t UNION SELECT b FROM u ORDER BY 1 LIMIT 5 OFFSET 2"),
            "(SELECT a FROM t UNION SELECT b FROM u) ORDER BY 1i32 ASC NULLS LAST LIMIT 5 OFFSET 2"
        );

        // 括弧無しの単一 SELECT なら SelectStmt 側に付く（binder の既存経路）。
        let (_, q) = parsed("SELECT a FROM t ORDER BY a LIMIT 5 OFFSET 1");
        assert!(q.order_by.is_empty());
        assert_eq!(q.limit, None);
        assert_eq!(q.offset, None);
        assert_eq!(plain(&q).order_by.len(), 1);
        assert_eq!(plain(&q).limit, Some(5));
        assert_eq!(plain(&q).offset, Some(1));

        // 括弧付きの項が自分の ORDER BY を持っていても外側に潰されない。
        assert_eq!(
            qs("(SELECT a FROM t ORDER BY a LIMIT 1) LIMIT 9"),
            "SELECT a FROM t ORDER BY a ASC NULLS LAST LIMIT 1 LIMIT 9"
        );
        // 集合演算の項ごとの LIMIT は括弧で表す。
        assert_eq!(
            qs("(SELECT a FROM t LIMIT 1) UNION SELECT b FROM u"),
            "(SELECT a FROM t LIMIT 1 UNION SELECT b FROM u)"
        );
        // CTE や外側 LIMIT を持つ括弧付きクエリは派生表に包んで項にする。
        assert_eq!(
            qs("(SELECT 1 UNION SELECT 2 LIMIT 3) UNION SELECT 4"),
            "(SELECT * FROM ((SELECT 1i32 UNION SELECT 2i32) LIMIT 3) UNION SELECT 4i32)"
        );
    }

    #[test]
    fn derived_table_with_set_operation() {
        assert_eq!(
            sel("SELECT * FROM (SELECT a FROM t UNION SELECT b FROM u) AS x"),
            "SELECT * FROM ((SELECT a FROM t UNION SELECT b FROM u)) AS x"
        );
        assert_eq!(
            sel("SELECT * FROM (WITH c AS (SELECT 1) SELECT * FROM c) AS x"),
            "SELECT * FROM (WITH c AS (SELECT 1i32) SELECT * FROM c) AS x"
        );
        assert_eq!(
            sel("SELECT * FROM (SELECT a FROM t ORDER BY a LIMIT 2) s JOIN u ON s.a = u.a"),
            "SELECT * FROM ((SELECT a FROM t ORDER BY a ASC NULLS LAST LIMIT 2) AS s \
             INNER JOIN u ON (s.a = u.a))"
        );
    }

    // --- ウィンドウ関数 -----------------------------------------------------

    #[test]
    fn window_functions() {
        assert_eq!(ex("row_number() OVER ()"), "row_number() OVER (WHOLE)");
        assert_eq!(ex("count(*) OVER ()"), "count(*) OVER (WHOLE)");
        assert_eq!(ex("sum(x) OVER (PARTITION BY a)"), "sum(x) OVER (PARTITION BY a WHOLE)");
        assert_eq!(
            ex("sum(x) OVER (PARTITION BY a, b + 1)"),
            "sum(x) OVER (PARTITION BY a, (b + 1i32) WHOLE)"
        );
        assert_eq!(
            ex("rank() OVER (PARTITION BY a ORDER BY b DESC)"),
            "rank() OVER (PARTITION BY a ORDER BY b DESC NULLS LAST RANGE)"
        );
        assert_eq!(
            ex("sum(x) OVER (ORDER BY b, c NULLS LAST)"),
            "sum(x) OVER (ORDER BY b ASC NULLS LAST, c ASC NULLS LAST RANGE)"
        );
        // 通常の関数呼び出しと同じ場所に書ける。
        assert_eq!(
            sel("SELECT sum(x) OVER (PARTITION BY a) AS s FROM t"),
            "SELECT sum(x) OVER (PARTITION BY a WHOLE) AS s FROM t"
        );

        // `star` と既定枠を AST 上で直接確認する。
        let (a, q) = parsed("SELECT count(*) OVER (), rank() OVER (ORDER BY b) FROM t");
        match a.get(plain(&q).items[0].expr) {
            Expr::Window { name, args, star, window_ref, partition_by, order_by, frame } => {
                assert_eq!(name, "count");
                assert!(args.is_empty());
                assert!(*star);
                assert!(window_ref.is_none());
                assert!(partition_by.is_empty());
                assert!(order_by.is_empty());
                assert_eq!(*frame, WindowFrame::WholePartition);
            }
            _ => panic!("ウィンドウ関数ではない"),
        }
        match a.get(plain(&q).items[1].expr) {
            Expr::Window { star, frame, order_by, .. } => {
                assert!(!*star);
                assert_eq!(order_by.len(), 1);
                assert_eq!(*frame, WindowFrame::RangeUnboundedPreceding);
            }
            _ => panic!("ウィンドウ関数ではない"),
        }
    }

    #[test]
    fn window_keywords_are_contextual() {
        // 列名はデータファイル由来で利用者が選べない。ウィンドウ構文のために
        // ありふれた語を予約語にすると、引用符無しでは読めない列ができる。
        for name in ["rows", "range", "over", "partition"] {
            assert_eq!(sel(&format!("SELECT {} FROM t", name)), format!("SELECT {} FROM t", name));
            assert_eq!(
                sel(&format!("SELECT t.{} FROM t", name)),
                format!("SELECT t.{} FROM t", name)
            );
            assert_eq!(
                sel(&format!("SELECT * FROM t WHERE {} > 1", name)),
                format!("SELECT * FROM t WHERE ({} > 1i32)", name)
            );
            // 表別名としても列別名としても使える。
            assert_eq!(
                sel(&format!("SELECT * FROM t AS {}", name)),
                format!("SELECT * FROM t AS {}", name)
            );
            assert_eq!(
                sel(&format!("SELECT * FROM t {}", name)),
                format!("SELECT * FROM t AS {}", name)
            );
            assert_eq!(
                sel(&format!("SELECT a AS {} FROM t", name)),
                format!("SELECT a AS {} FROM t", name)
            );
            // 大文字でも同じ（識別子の綴りはそのまま保つ）。
            let upper = name.to_ascii_uppercase();
            assert_eq!(
                sel(&format!("SELECT {} FROM t", upper)),
                format!("SELECT {} FROM t", upper)
            );
            // GROUP BY / ORDER BY の中でも列として読める。
            assert_eq!(
                sel(&format!("SELECT {0} FROM t GROUP BY {0} ORDER BY {0}", name)),
                format!("SELECT {0} FROM t GROUP BY {0} ORDER BY {0} ASC NULLS LAST", name)
            );
        }

        // `over` の直後が `(` でなければウィンドウ句ではなく別名。
        assert_eq!(sel("SELECT count(*) over FROM t"), "SELECT count(*) AS over FROM t");
        assert_eq!(sel("SELECT count(*) over"), "SELECT count(*) AS over");
        assert_eq!(sel("SELECT count(*) AS over FROM t"), "SELECT count(*) AS over FROM t");
        // 関数呼び出しでない `over` の直後の `(` は別名 + 構文エラーのまま。
        assert_eq!(code("SELECT a over (b)"), Code::UnexpectedToken as u16);

        // 文脈依存にしてもウィンドウ指定は今までどおり効く。
        assert_eq!(ex("count(*) OVER ()"), "count(*) OVER (WHOLE)");
        assert_eq!(
            ex("sum(rows) OVER (PARTITION BY partition ORDER BY range)"),
            "sum(rows) OVER (PARTITION BY partition ORDER BY range ASC NULLS LAST RANGE)"
        );
        assert_eq!(
            sel("SELECT rank() OVER (PARTITION BY over) FROM t"),
            "SELECT rank() OVER (PARTITION BY over WHOLE) FROM t"
        );
        // 枠指定の拒否は予約語ではなく綴りで判定する。
        assert_eq!(
            code("SELECT sum(x) OVER (ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)"),
            Code::UnsupportedFeature as u16
        );
        assert_eq!(
            code("SELECT sum(x) OVER (rows BETWEEN 1 PRECEDING AND CURRENT ROW)"),
            Code::UnsupportedFeature as u16
        );
        // 引用符付き識別子は文脈依存キーワードとして照合しない。
        assert_eq!(sel("SELECT \"rows\" FROM t"), "SELECT rows FROM t");
        assert_eq!(code("SELECT sum(x) OVER (\"rows\")"), Code::UnexpectedToken as u16);
    }

    #[test]
    fn window_rejections() {
        // 明示的な枠指定は黙って無視すると結果が変わるので必ず弾く。
        assert_eq!(
            code("SELECT sum(x) OVER (ORDER BY b ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)"),
            Code::UnsupportedFeature as u16
        );
        assert_eq!(
            code("SELECT sum(x) OVER (RANGE UNBOUNDED PRECEDING)"),
            Code::UnsupportedFeature as u16
        );
        assert_eq!(
            code("SELECT sum(x) OVER (PARTITION BY a ROWS UNBOUNDED PRECEDING)"),
            Code::UnsupportedFeature as u16
        );
        // DISTINCT 付きのウィンドウ集約も範囲外。
        assert_eq!(code("SELECT count(DISTINCT x) OVER ()"), Code::UnsupportedFeature as u16);
        // `OVER w`（名前付き参照）は構文としては通る。定義の有無は束縛時にしか
        // 分からない（`plan::bind` 側のテスト参照）。
        assert_eq!(code("SELECT sum(x) OVER w"), 0);
        assert_eq!(code("SELECT sum(x) OVER (PARTITION a)"), Code::UnexpectedToken as u16);
    }

    #[test]
    fn named_window_ref_parses() {
        // `OVER w`（識別子 1 つだけ）は名前付きウィンドウの参照として木に残す。
        // 実体（PARTITION BY/ORDER BY）は `WINDOW` 句側にあり、ここでは
        // パーサが名前だけを持たせていることを確認する。
        assert_eq!(ex("sum(x) OVER w"), "sum(x) OVER w");
        let (a, q) = parsed("SELECT sum(x) OVER w FROM t");
        match a.get(plain(&q).items[0].expr) {
            Expr::Window { name, window_ref, partition_by, order_by, .. } => {
                assert_eq!(name, "sum");
                assert_eq!(window_ref.as_deref(), Some("w"));
                assert!(partition_by.is_empty());
                assert!(order_by.is_empty());
            }
            _ => panic!("ウィンドウ関数ではない"),
        }
        // `OVER` の直後が `(` でも識別子でもなければ、今までどおり別名扱い。
        assert_eq!(sel("SELECT count(*) over FROM t"), "SELECT count(*) AS over FROM t");
        // ただし `OVER` の直後に識別子が続けば、コンマを挟まない限り名前付き
        // ウィンドウ参照としてしか解釈できない（別名 + 別の select item は
        // コンマが要るので、そちらの解釈はそもそも成り立たない）。
        assert_eq!(sel("SELECT sum(x) over w, y FROM t"), "SELECT sum(x) OVER w, y FROM t");
    }

    #[test]
    fn window_clause_named_definitions() {
        // 単純な名前付きウィンドウ。複数の関数から同じ定義を共有できる。
        assert_eq!(
            sel(
                "SELECT id, sum(x) OVER w, avg(x) OVER w FROM t WINDOW w AS (PARTITION BY id ORDER BY ts)"
            ),
            "SELECT id, sum(x) OVER w, avg(x) OVER w FROM t WINDOW w AS (PARTITION BY id ORDER BY ts ASC NULLS LAST)"
        );
        // カンマ区切りで複数定義できる。
        assert_eq!(
            sel(
                "SELECT sum(x) OVER w1, rank() OVER w2 FROM t WINDOW w1 AS (PARTITION BY id), w2 AS (ORDER BY x)"
            ),
            "SELECT sum(x) OVER w1, rank() OVER w2 FROM t WINDOW w1 AS (PARTITION BY id), w2 AS (ORDER BY x ASC NULLS LAST)"
        );
        // 通常の `OVER (...)` 直書きと併用できる。
        assert_eq!(
            sel("SELECT sum(x) OVER w, count(*) OVER () FROM t WINDOW w AS (PARTITION BY id)"),
            "SELECT sum(x) OVER w, count(*) OVER (WHOLE) FROM t WINDOW w AS (PARTITION BY id)"
        );
        // `WINDOW` は `GROUP BY`/`HAVING` の後、`QUALIFY`/`ORDER BY` の前。
        assert_eq!(
            sel(
                "SELECT a, sum(x) OVER w FROM t WHERE a > 0 GROUP BY a, x HAVING x > 0 WINDOW w AS (ORDER BY a) QUALIFY sum(x) OVER w > 0 ORDER BY a"
            ),
            "SELECT a, sum(x) OVER w FROM t WHERE (a > 0i32) GROUP BY a, x HAVING (x > 0i32) WINDOW w AS (ORDER BY a ASC NULLS LAST) QUALIFY (sum(x) OVER w > 0i32) ORDER BY a ASC NULLS LAST"
        );
        // 同じ名前を 2 回定義するのはエラー（`duckdb` も "already defined" として拒否）。
        assert_eq!(
            code("SELECT a FROM t WINDOW w AS (ORDER BY a), w AS (ORDER BY a)"),
            Code::SyntaxError as u16
        );
        // `WINDOW` は完全な予約語（`QUALIFY` と同じ判断）なので列名としては使えない。
        assert_eq!(code("SELECT window FROM t"), Code::UnexpectedToken as u16);
        // 引用符を付ければ引き続き列名として使える。
        assert_eq!(sel("SELECT \"window\" FROM t"), "SELECT window FROM t");
    }

    // --- サブクエリ式 -------------------------------------------------------

    #[test]
    fn scalar_subqueries() {
        assert_eq!(ex("(SELECT 1)"), "(SELECT 1i32)");
        assert_eq!(
            sel("SELECT (SELECT max(x) FROM u) AS m FROM t"),
            "SELECT (SELECT max(x) FROM u) AS m FROM t"
        );
        assert_eq!(
            sel("SELECT * FROM t WHERE a > (SELECT avg(x) FROM u)"),
            "SELECT * FROM t WHERE (a > (SELECT avg(x) FROM u))"
        );
        assert_eq!(ex("(SELECT 1) + 1"), "((SELECT 1i32) + 1i32)");
        // 括弧を重ねても意味は同じ。内側の `(` が改めて判定を受ける。
        assert_eq!(ex("((SELECT 1))"), "(SELECT 1i32)");
        // 括弧付き式は今までどおり。
        assert_eq!(ex("(1 + 2)"), "(1i32 + 2i32)");
        assert_eq!(ex("(SELECT 1 UNION SELECT 2)"), "((SELECT 1i32 UNION SELECT 2i32))");
        assert_eq!(
            ex("(WITH a AS (SELECT 1) SELECT * FROM a)"),
            "(WITH a AS (SELECT 1i32) SELECT * FROM a)"
        );
    }

    #[test]
    fn exists_and_in_subquery() {
        assert_eq!(ex("EXISTS (SELECT 1)"), "EXISTS (SELECT 1i32)");
        assert_eq!(ex("NOT EXISTS (SELECT 1)"), "NOT EXISTS (SELECT 1i32)");
        assert_eq!(
            sel("SELECT * FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.a = t.a)"),
            "SELECT * FROM t WHERE EXISTS (SELECT 1i32 FROM u WHERE (u.a = t.a))"
        );
        assert_eq!(
            ex("a = 1 AND NOT EXISTS (SELECT 1 FROM u)"),
            "((a = 1i32) AND NOT EXISTS (SELECT 1i32 FROM u))"
        );
        assert_eq!(ex("x IN (SELECT a FROM t)"), "(x IN (SELECT a FROM t))");
        assert_eq!(ex("x NOT IN (SELECT a FROM t)"), "(x NOT IN (SELECT a FROM t))");
        assert_eq!(
            ex("x IN (SELECT a FROM t UNION SELECT b FROM u)"),
            "(x IN ((SELECT a FROM t UNION SELECT b FROM u)))"
        );
        // 値リストは今までどおり値リストのまま。
        assert_eq!(ex("x IN (1, 2, 3)"), "(x IN [1i32, 2i32, 3i32])");
        assert_eq!(ex("x IN ((1), (2))"), "(x IN [1i32, 2i32])");
        // `IN ((SELECT ...))` は値リスト側に倒す。要素がスカラサブクエリになる。
        assert_eq!(ex("x IN ((SELECT 1))"), "(x IN [(SELECT 1i32)])");
        assert_eq!(code("SELECT x IN ()"), Code::UnexpectedToken as u16);
    }

    #[test]
    fn other_statements() {
        let p = parse("EXPLAIN SELECT 1").expect("parse");
        assert!(matches!(p.stmt, Stmt::Explain(_)));
        let p = parse("DESCRIBE t").expect("parse");
        match &p.stmt {
            Stmt::Describe(f) => assert_eq!(from_str(&p.arena, f), "t"),
            _ => panic!("not describe"),
        }
        let p = parse("DESCRIBE parquet('x.parquet')").expect("parse");
        match &p.stmt {
            Stmt::Describe(f) => assert_eq!(from_str(&p.arena, f), "parquet('x.parquet')"),
            _ => panic!("not describe"),
        }
        assert!(matches!(parse("SHOW TABLES;").expect("parse").stmt, Stmt::ShowTables));
        assert_eq!(code("SHOW COLUMNS"), Code::UnexpectedToken as u16);
    }

    // --- エラー -------------------------------------------------------------

    #[test]
    fn errors() {
        assert_eq!(code("SELECT FROM t"), Code::UnexpectedToken as u16);
        assert_eq!(code("SELECT * FROM"), Code::UnexpectedToken as u16);
        assert_eq!(code("SELECT (1 + 2"), Code::UnexpectedToken as u16);
        assert_eq!(code("SELECT 1)"), Code::UnexpectedToken as u16);
        assert_eq!(code("SELECT * FROM t WHERE"), Code::UnexpectedToken as u16);
        assert_eq!(code("SELECT * FROM t LIMIT x"), Code::UnexpectedToken as u16);
        assert_eq!(code("SELECT a b c FROM t"), Code::UnexpectedToken as u16);
        assert_eq!(code(""), Code::UnsupportedFeature as u16);
        // `ddl`/`dml` フィーチャが有効だと INSERT/UPDATE/CREATE TABLE は正当な
        // 文になる。フィーチャが無効な既定ビルドでの挙動はここで確認し、
        // 有効時の挙動は `sql/parser.rs` 末尾の `ddl_dml` テストで確認する。
        #[cfg(not(feature = "dml"))]
        {
            assert_eq!(code("INSERT INTO t VALUES (1)"), Code::UnsupportedFeature as u16);
            assert_eq!(code("UPDATE t SET a = 1"), Code::UnsupportedFeature as u16);
        }
        #[cfg(not(feature = "ddl"))]
        {
            assert_eq!(code("CREATE TABLE t (a INT)"), Code::UnsupportedFeature as u16);
            // ALTER TABLE も `ddl` が無効なビルドでは引き続き未対応。有効時の
            // 挙動は `sql/parser.rs` 末尾の DDL/DML テスト群で確認する。
            assert_eq!(code("ALTER TABLE t ADD COLUMN x INT"), Code::UnsupportedFeature as u16);
        }
        assert_eq!(code("WITH x AS SELECT 1 SELECT * FROM x"), Code::UnexpectedToken as u16);
        // 列名リストは非再帰 CTE では引き続き未対応（`WITH RECURSIVE` の下でだけ
        // 許す。`recursive_cte` テスト群参照）。
        assert_eq!(code("WITH x (a) AS (SELECT 1) SELECT 1"), Code::UnsupportedFeature as u16);
        assert_eq!(code("SELECT 1 UNION"), Code::UnexpectedToken as u16);
        assert_eq!(code("SELECT 1 INTERSECT 2"), Code::UnexpectedToken as u16);
        assert_eq!(code("SELECT a & b"), Code::UnexpectedToken as u16);
    }

    #[test]
    fn error_positions() {
        // 位置は必ず「問題のトークンの先頭バイト」を指す。
        assert_eq!(err_at("SELECT FROM t"), (Code::UnexpectedToken as u16, 7));
        assert_eq!(err_at("SELECT 'abc"), (Code::UnterminatedString as u16, 7));
        #[cfg(not(feature = "dml"))]
        assert_eq!(err_at("INSERT INTO t VALUES (1)"), (Code::UnsupportedFeature as u16, 0));
        #[cfg(not(feature = "ddl"))]
        assert_eq!(err_at("ALTER TABLE t ADD COLUMN x INT"), (Code::UnsupportedFeature as u16, 0));
        assert_eq!(err_at("SELECT a FROM t WHERE b @ 1"), (Code::UnexpectedToken as u16, 24));
        assert_eq!(err_at("SELECT CAST(x AS FROB)"), (Code::InvalidCast as u16, 17));
        // 新しい構文でも位置は問題のトークンの先頭を指す。
        assert_eq!(err_at("SELECT 1 UNION 2"), (Code::UnexpectedToken as u16, 15));
        assert_eq!(err_at("SELECT sum(x) OVER (ROWS)"), (Code::UnsupportedFeature as u16, 20));
        // 列名リストが要求する `(` の位置。
        assert_eq!(
            err_at("WITH x (a) AS (SELECT 1) SELECT 1"),
            (Code::UnsupportedFeature as u16, 7)
        );
        assert_eq!(err_at("SELECT 1 WHERE EXISTS SELECT 1"), (Code::UnexpectedToken as u16, 22));
    }

    #[test]
    fn deep_nesting_is_rejected_without_overflow() {
        let mut sql = String::from("SELECT ");
        for _ in 0..100_000 {
            sql.push('(');
        }
        sql.push('1');
        assert_eq!(code(&sql), Code::ExpressionTooDeep as u16);

        // 深いサブクエリとカンマ結合も同じ上限で止まる。
        let mut q = String::new();
        for _ in 0..200 {
            q.push_str("SELECT * FROM (");
        }
        q.push_str("SELECT 1");
        for _ in 0..200 {
            q.push(')');
        }
        assert_eq!(code(&q), Code::ExpressionTooDeep as u16);

        let mut j = String::from("SELECT * FROM a");
        for _ in 0..200 {
            j.push_str(", a");
        }
        assert_eq!(code(&j), Code::ExpressionTooDeep as u16);

        // 上限直下は通る。
        let ok = format!("SELECT {}1{}", "(".repeat(50), ")".repeat(50));
        assert_eq!(code(&ok), 0);
    }

    #[test]
    fn deep_subquery_expressions_are_rejected() {
        // スカラサブクエリの入れ子。木の破棄も再帰するので、上限は
        // 「パースを通った木を落としてもスタックが持つ」水準でなければならない。
        let n = 200;
        let mut s = String::from("SELECT ");
        s.push_str(&"(SELECT ".repeat(n));
        s.push('1');
        s.push_str(&")".repeat(n));
        assert_eq!(code(&s), Code::ExpressionTooDeep as u16);

        // EXISTS の入れ子。
        let mut e = String::from("SELECT 1 WHERE ");
        e.push_str(&"EXISTS (SELECT 1 WHERE ".repeat(n));
        e.push_str("TRUE");
        e.push_str(&")".repeat(n));
        assert_eq!(code(&e), Code::ExpressionTooDeep as u16);

        // IN (SELECT ...) の入れ子。
        let mut i = String::from("SELECT 1 WHERE ");
        i.push_str(&"1 IN (SELECT 1 WHERE ".repeat(n));
        i.push_str("TRUE");
        i.push_str(&")".repeat(n));
        assert_eq!(code(&i), Code::ExpressionTooDeep as u16);

        // 派生表の入れ子（括弧付きクエリ経由）。
        let mut d = String::from("SELECT * FROM ");
        d.push_str(&"(SELECT * FROM ".repeat(n));
        d.push('t');
        d.push_str(&")".repeat(n));
        assert_eq!(code(&d), Code::ExpressionTooDeep as u16);

        // CTE の入れ子。
        let mut c = String::new();
        c.push_str(&"WITH a AS (".repeat(n));
        c.push_str("SELECT 1");
        for _ in 0..n {
            c.push_str(") SELECT * FROM a");
        }
        assert_eq!(code(&c), Code::ExpressionTooDeep as u16);

        // 上限直下のスカラサブクエリは通る（1 段あたり深さ 2 を消費する）。
        let k = 30;
        let mut ok = String::from("SELECT ");
        ok.push_str(&"(SELECT ".repeat(k));
        ok.push('1');
        ok.push_str(&")".repeat(k));
        assert_eq!(code(&ok), 0);
    }

    #[test]
    fn long_setop_chains_are_rejected() {
        // 集合演算も左深の `Box` 連鎖。破棄時の再帰を上限で止める。
        let mut u = String::from("SELECT 1");
        u.push_str(&" UNION SELECT 1".repeat(200));
        assert_eq!(code(&u), Code::ExpressionTooDeep as u16);

        let mut x = String::from("SELECT 1");
        x.push_str(&" INTERSECT SELECT 1".repeat(200));
        assert_eq!(code(&x), Code::ExpressionTooDeep as u16);

        // 上限直下は通る。
        let mut ok = String::from("SELECT 1");
        ok.push_str(&" UNION SELECT 1".repeat(60));
        assert_eq!(code(&ok), 0);
    }

    // --- DDL/DML（`ddl`/`dml` フィーチャ） ------------------------------------

    #[cfg(feature = "ddl")]
    #[test]
    fn create_table_variants_parse() {
        let p = parse("CREATE TABLE t (id INTEGER, name VARCHAR NOT NULL)").expect("parse");
        match p.stmt {
            Stmt::CreateTable { name, or_replace, if_not_exists, columns, as_select } => {
                assert_eq!(name, "t");
                assert!(!or_replace);
                assert!(!if_not_exists);
                assert!(as_select.is_none());
                assert_eq!(columns.len(), 2);
                assert!(columns[0].nullable);
                assert!(!columns[1].nullable);
            }
            _ => panic!("not CreateTable"),
        }

        let p = parse("CREATE TABLE IF NOT EXISTS t (id INTEGER)").expect("parse");
        match p.stmt {
            Stmt::CreateTable { if_not_exists, .. } => assert!(if_not_exists),
            _ => panic!("not CreateTable"),
        }

        let p = parse("CREATE OR REPLACE TABLE t AS SELECT * FROM u").expect("parse");
        match p.stmt {
            Stmt::CreateTable { or_replace, as_select, columns, .. } => {
                assert!(or_replace);
                assert!(as_select.is_some());
                assert!(columns.is_empty());
            }
            _ => panic!("not CreateTable"),
        }
    }

    #[cfg(feature = "ddl")]
    #[test]
    fn create_view_captures_body_as_raw_sql() {
        let p = parse("CREATE OR REPLACE VIEW v AS SELECT a, b FROM t WHERE a > 1").expect("parse");
        match p.stmt {
            Stmt::CreateView { name, or_replace, query_sql } => {
                assert_eq!(name, "v");
                assert!(or_replace);
                assert_eq!(query_sql, "SELECT a, b FROM t WHERE a > 1");
            }
            _ => panic!("not CreateView"),
        }
    }

    #[cfg(feature = "ddl")]
    #[test]
    fn drop_table_and_view_parse() {
        let p = parse("DROP TABLE IF EXISTS t").expect("parse");
        match p.stmt {
            Stmt::DropTable { name, if_exists } => {
                assert_eq!(name, "t");
                assert!(if_exists);
            }
            _ => panic!("not DropTable"),
        }
        let p = parse("DROP VIEW v").expect("parse");
        match p.stmt {
            Stmt::DropView { name, if_exists } => {
                assert_eq!(name, "v");
                assert!(!if_exists);
            }
            _ => panic!("not DropView"),
        }
    }

    #[cfg(feature = "ddl")]
    #[test]
    fn alter_table_add_column_variants_parse() {
        let p = parse("ALTER TABLE t ADD COLUMN x INT").expect("parse");
        match p.stmt {
            Stmt::AlterTable { name, action } => {
                assert_eq!(name, "t");
                match action {
                    AlterTableAction::AddColumn { name, ty, nullable, default } => {
                        assert_eq!(name, "x");
                        assert_eq!(ty, Ty::Int);
                        assert!(nullable);
                        assert!(default.is_none());
                    }
                    _ => panic!("not AddColumn"),
                }
            }
            _ => panic!("not AlterTable"),
        }

        // `COLUMN` は省略できる（DuckDB と同じ、CLI で確認済み）。
        let p = parse("ALTER TABLE t ADD y INT NOT NULL DEFAULT 5").expect("parse");
        match p.stmt {
            Stmt::AlterTable { action, .. } => match action {
                AlterTableAction::AddColumn { name, nullable, default, .. } => {
                    assert_eq!(name, "y");
                    assert!(!nullable);
                    assert!(default.is_some());
                }
                _ => panic!("not AddColumn"),
            },
            _ => panic!("not AlterTable"),
        }
    }

    #[cfg(feature = "ddl")]
    #[test]
    fn alter_table_drop_column_parse() {
        let p = parse("ALTER TABLE t DROP COLUMN x").expect("parse");
        match p.stmt {
            Stmt::AlterTable { name, action } => {
                assert_eq!(name, "t");
                match action {
                    AlterTableAction::DropColumn { name } => assert_eq!(name, "x"),
                    _ => panic!("not DropColumn"),
                }
            }
            _ => panic!("not AlterTable"),
        }
        // `COLUMN` は省略できる。
        let p = parse("ALTER TABLE t DROP x").expect("parse");
        assert!(matches!(
            p.stmt,
            Stmt::AlterTable { action: AlterTableAction::DropColumn { .. }, .. }
        ));
    }

    #[cfg(feature = "ddl")]
    #[test]
    fn alter_table_rename_variants_parse() {
        let p = parse("ALTER TABLE t RENAME COLUMN a TO b").expect("parse");
        match p.stmt {
            Stmt::AlterTable { action, .. } => match action {
                AlterTableAction::RenameColumn { old, new } => {
                    assert_eq!(old, "a");
                    assert_eq!(new, "b");
                }
                _ => panic!("not RenameColumn"),
            },
            _ => panic!("not AlterTable"),
        }

        // `COLUMN` は省略できる。
        let p = parse("ALTER TABLE t RENAME a TO b").expect("parse");
        assert!(matches!(
            p.stmt,
            Stmt::AlterTable { action: AlterTableAction::RenameColumn { .. }, .. }
        ));

        let p = parse("ALTER TABLE t RENAME TO u").expect("parse");
        match p.stmt {
            Stmt::AlterTable { name, action } => {
                assert_eq!(name, "t");
                match action {
                    AlterTableAction::RenameTable { new_name } => assert_eq!(new_name, "u"),
                    _ => panic!("not RenameTable"),
                }
            }
            _ => panic!("not AlterTable"),
        }
    }

    #[cfg(feature = "dml")]
    #[test]
    fn insert_variants_parse() {
        let p = parse("INSERT INTO t VALUES (1, 'a'), (2, 'b')").expect("parse");
        match p.stmt {
            Stmt::Insert { table, columns, source } => {
                assert_eq!(table, "t");
                assert!(columns.is_empty());
                match source {
                    InsertSource::Values(rows) => assert_eq!(rows.len(), 2),
                    _ => panic!("not Values"),
                }
            }
            _ => panic!("not Insert"),
        }

        let p = parse("INSERT INTO t (id, name) SELECT id, name FROM u").expect("parse");
        match p.stmt {
            Stmt::Insert { columns, source, .. } => {
                assert_eq!(columns, vec!["id".to_string(), "name".to_string()]);
                assert!(matches!(source, InsertSource::Query(_)));
            }
            _ => panic!("not Insert"),
        }
    }

    #[cfg(feature = "dml")]
    #[test]
    fn update_and_delete_parse() {
        let p = parse("UPDATE t SET a = 1, b = a + 1 WHERE id = 5").expect("parse");
        match p.stmt {
            Stmt::Update { table, assignments, filter } => {
                assert_eq!(table, "t");
                assert_eq!(assignments.len(), 2);
                assert!(filter.is_some());
            }
            _ => panic!("not Update"),
        }

        let p = parse("DELETE FROM t").expect("parse");
        match p.stmt {
            Stmt::Delete { table, filter } => {
                assert_eq!(table, "t");
                assert!(filter.is_none());
            }
            _ => panic!("not Delete"),
        }
        let p = parse("DELETE FROM t WHERE id > 1").expect("parse");
        assert!(matches!(p.stmt, Stmt::Delete { filter: Some(_), .. }));
    }

    #[cfg(feature = "export")]
    #[test]
    fn copy_subquery_parses_path_and_optional_format() {
        let p = parse("COPY (SELECT a, b FROM t) TO 'out.csv'").expect("parse");
        match p.stmt {
            Stmt::Copy { query, path, format } => {
                assert_eq!(path, "out.csv");
                assert!(format.is_none());
                assert!(query.as_simple_select().is_some());
            }
            _ => panic!("not Copy"),
        }

        let p = parse("COPY (SELECT a FROM t) TO 'out.txt' (FORMAT csv)").expect("parse");
        match p.stmt {
            Stmt::Copy { path, format, .. } => {
                assert_eq!(path, "out.txt");
                assert_eq!(format.as_deref(), Some("csv"));
            }
            _ => panic!("not Copy"),
        }

        // FORMAT の値は大文字小文字を問わない。
        let p = parse("COPY (SELECT a FROM t) TO 'out.bin' (FORMAT JSON)").expect("parse");
        assert!(
            matches!(p.stmt, Stmt::Copy { format: Some(f), .. } if f.eq_ignore_ascii_case("json"))
        );
    }

    /// `COPY <table> TO ...` は `SELECT * FROM <table>` と等価な木になる。
    #[cfg(feature = "export")]
    #[test]
    fn copy_table_form_desugars_to_select_star() {
        let p = parse("COPY t TO 'out.csv'").expect("parse");
        match p.stmt {
            Stmt::Copy { query, path, .. } => {
                assert_eq!(path, "out.csv");
                let sel = query.as_simple_select().expect("simple select");
                assert_eq!(sel.items.len(), 1);
                assert!(matches!(
                    p.arena.get(sel.items[0].expr),
                    Expr::Star { qualifier: None, .. }
                ));
                match &sel.from {
                    Some(FromItem::Table { name, alias }) => {
                        assert_eq!(name, "t");
                        assert!(alias.is_none());
                    }
                    _ => panic!("not Table"),
                }
            }
            _ => panic!("not Copy"),
        }
    }

    /// `format` は `COPY` の外では普通の列名として使える（`export` はこの語を
    /// グローバルには予約しない — 文脈依存キーワードなので）。
    #[cfg(feature = "export")]
    #[test]
    fn format_remains_usable_as_a_column_name_outside_copy() {
        assert_eq!(sel("SELECT format FROM t"), "SELECT format FROM t");
    }

    /// `to` も `export` 単体では同じく普通の列名として使える。`ddl` が同時に
    /// 有効だと `ALTER TABLE ... RENAME TO` 用に別途グローバル予約される
    /// （`sql/lexer.rs` の `DDL_KEYWORDS`）ため、この確認は `ddl` が無効な
    /// 構成でのみ意味を持つ（`copy_stmt`/`expect_to` のドキュメント参照）。
    #[cfg(all(feature = "export", not(feature = "ddl")))]
    #[test]
    fn to_remains_usable_as_a_column_name_when_ddl_is_disabled() {
        assert_eq!(sel("SELECT to FROM t"), "SELECT to FROM t");
    }

    #[cfg(feature = "export")]
    #[test]
    fn copy_requires_to_and_a_string_path() {
        assert_eq!(code("COPY (SELECT 1) TO"), Code::UnexpectedToken as u16);
        assert_eq!(code("COPY (SELECT 1)"), Code::UnexpectedToken as u16);
        assert_eq!(code("COPY (SELECT 1) TO out.csv"), Code::UnexpectedToken as u16);
    }
}
