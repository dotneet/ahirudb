//! The SQL parser.
//!
//! Statements are handled by recursive descent, expressions by Pratt (precedence
//! climbing) (DESIGN.md §7). Expressions go into an arena rather than `Box`, so dropping the tree does not recurse.
//!
//! Recursion is always capped by `MAX_DEPTH`. On wasm, stack exhaustion is an
//! unrecoverable trap, so deep input is always returned as an error.

use crate::format::FormatKind;
use crate::prelude::*;
use crate::rt::hash::eq_ascii_ci;
use crate::sql::ast::{
    BinaryOp, ColumnsSpec, Cte, Expr, ExprArena, ExprId, FromItem, JoinKind, OrderByAll,
    OrderByItem, Parsed, PivotStmt, QueryStmt, SampleMethod, SampleSpec, SelectItem, SelectStmt,
    SetExpr, SetOp, Stmt, UnaryOp, UnpivotStmt, WindowDef, WindowFrame,
};
use crate::sql::lexer::{Kw, Lexer, Tok};
use crate::vector::{Ty, Value};

use types::{int_literal, unquote};

/// The depth limit for recursive descent and Pratt parsing. It bounds nested parentheses, subqueries, and nested queries.
///
/// Expressions go into the arena and do not recurse on drop, but `QueryStmt` recurses
/// through `Box`, so **even a successfully parsed tree recurses when dropped**. The
/// limit must be safe including that drop.
const MAX_DEPTH: u16 = 64;

/// The cap on the total number of left-deep `Box` chains (JOINs and set operations).
///
/// Both are built with loops syntactically, so parsing does not recurse, but dropping
/// the tree recurses once per link in the chain. It is counted separately from the
/// depth limit, as a running total across the whole statement (a per-nested-query limit would multiply).
const MAX_LINKS: u16 = 64;

/// The return value of `Parser::star_modifiers`: `(column names to EXCLUDE, the
/// (expression, column name) pairs to REPLACE, the (old name, new name) pairs to RENAME)`.
type StarModifiers = (Vec<String>, Vec<(ExprId, String)>, Vec<(String, String)>);

// Binding power. Larger binds tighter.
const BP_OR: u8 = 1;
const BP_AND: u8 = 2;
const BP_NOT: u8 = 3;
const BP_CMP: u8 = 4;
// `&`/`|`/`<<`/`>>`. Tighter than comparison, looser than `||` and arithmetic
// (confirmed with the `duckdb` CLI: `1 + 2 & 3` = `(1 + 2) & 3`, `1 & 2 = 0` = `(1 & 2) = 0`).
// The relative precedence among those four operators has not been measured, so they are
// simply collapsed into one level.
const BP_BITWISE: u8 = 5;
const BP_CONCAT: u8 = 6;
const BP_ADD: u8 = 7;
const BP_MUL: u8 = 8;
// `^`/`**`. Confirmed with the `duckdb` CLI: `2 + 3^2` = `2 + (3^2)`, `-2^2` = `(-2)^2`
// (tighter than `*`/`/`, looser than unary `-`). Left-associative
// (`2^3^2` = `(2^3)^2` = 64, not the right-associative 512).
const BP_POW: u8 = 9;
// Postfix `!` (factorial). DuckDB's own precedence for `!` is internally
// inconsistent Postgres legacy (`3! ^ 2` parses fine but `2 ^ 3!` is a
// syntax error; `2 + 3!` silently reads as `(2+3)!` while `3! + 1` is a
// syntax error) — not worth replicating. This engine instead picks a
// self-consistent rule: `!` binds looser than every prefix operator
// (`-`/`~`/`NOT`, all read at `BP_UNARY`) but tighter than every binary
// operator. Concretely, `BP_BANG` sits strictly between `BP_POW` (the
// strongest binary operator) and `BP_UNARY`, and `prefix()` always reads
// its operand at `BP_UNARY` — so a `!` following `-x`/`~x` is left alone
// by the inner (operand) parse and only picked up once the outer loop
// already has the completed `-x`/`~x` node as `lhs`. That makes `-4!` and
// `-x!` (literal and column operand alike) both parse as `(-x)!`, matching
// DuckDB's actual behavior there, while deliberately diverging from
// DuckDB on the binary-operator cases documented in
// docs/sql/limitations.md (`2 + 3!` = `2 + (3!)` = `8` here, not
// `(2+3)!` = `120`).
const BP_BANG: u8 = 10;
const BP_UNARY: u8 = 11;

pub fn parse(sql: &str) -> Result<Parsed> {
    let mut p = Parser::new(sql)?;
    let stmt = p.stmt(sql)?;
    Ok(Parsed { arena: p.arena, stmt, num_params: p.num_params })
}

struct Parser<'a> {
    lex: Lexer<'a>,
    /// One token of lookahead.
    cur: Tok<'a>,
    /// The input byte position of `cur`. Every error carries it.
    pos: usize,
    arena: ExprArena,
    depth: u16,
    /// The running count of `Box` chains built by JOINs and set operations.
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

    /// Called before extending a `Box` chain by one. An error once the limit is exceeded.
    fn link(&mut self) -> Result<()> {
        self.links += 1;
        ensure!(self.links <= MAX_LINKS, ExpressionTooDeep, self.pos);
        Ok(())
    }

    // --- Token operations ---------------------------------------------------

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

    /// Looks ahead at the second token. It merely clones the lexer and advances it once,
    /// leaving `self`'s position alone. Used only to detect `OVER (`.
    fn peek(&self) -> Result<Tok<'a>> {
        let mut lx = self.lex.clone();
        Ok(lx.next_token()?.tok)
    }

    /// Matches a context-dependent keyword. Quoted identifiers (`"over"`) are excluded,
    /// so quoting explicitly always makes it a column name.
    #[inline]
    fn is_soft_kw(&self, word: &[u8]) -> bool {
        matches!(self.cur, Tok::Ident(s) if eq_ascii_ci(s.as_bytes(), word))
    }

    /// Advances expecting the current token to be a single-quoted string literal. Used
    /// at positions where "the next thing is necessarily a string literal", such as the
    /// path argument of `COPY ... TO '<path>'` / `parquet('<path>')`.
    fn string_lit(&mut self) -> Result<String> {
        let s = match self.cur {
            Tok::Str(s) => unquote(s, b'\''),
            _ => err!(UnexpectedToken, self.pos),
        };
        self.bump()?;
        Ok(s)
    }

    /// Desugars operators and syntactic sugar into a plain function-call node.
    /// The shared form for simple calls without `DISTINCT`/`*`/`FILTER`, used by
    /// `GLOB`/`->`/`->>`/`SIMILAR TO`/array literals/`EXTRACT`.
    fn simple_call(&mut self, name: &str, args: Vec<ExprId>) -> ExprId {
        self.arena.push(Expr::Function {
            name: name.into(),
            args,
            distinct: false,
            star: false,
            filter: None,
        })
    }

    /// Reads a single-column alias of the form `AS name(col)`. The "column list of
    /// exactly one" shape shared by the `generate_series`/`range`/`UNNEST` FROM items.
    fn opt_single_col_alias(&mut self) -> Result<Option<String>> {
        if self.is(Tok::LParen) {
            self.bump()?;
            let col = self.ident()?;
            self.expect(Tok::RParen)?;
            Ok(Some(col))
        } else {
            Ok(None)
        }
    }

    /// Whether the second token matches a context-dependent keyword (for detecting `SETS`
    /// in `GROUPING SETS`; the two-token-lookahead version of `is_soft_kw`).
    #[inline]
    fn peek_is_soft_kw(&self, word: &[u8]) -> Result<bool> {
        Ok(matches!(self.peek()?, Tok::Ident(s) if eq_ascii_ci(s.as_bytes(), word)))
    }

    /// The `to`-aware version of `peek_is_soft_kw(b"to")`. Used for the two-token
    /// lookahead of `SIMILAR TO`. When the `ddl` feature is on, `to` arrives as
    /// `Tok::Kw(Kw::To)` via `DDL_KEYWORDS` (for `ALTER TABLE ... RENAME TO`), so
    /// `peek_is_soft_kw` alone (which only looks at `Tok::Ident`) would fail to recognize
    /// `SIMILAR TO` in a `ddl`-enabled build (see the doc comment on `expect_to`; the same
    /// dual-representation problem).
    #[inline]
    fn peek_is_to(&self) -> Result<bool> {
        #[cfg(feature = "ddl")]
        if matches!(self.peek()?, Tok::Kw(Kw::To)) {
            return Ok(true);
        }
        self.peek_is_soft_kw(b"to")
    }

    /// The `into`-aware version of `is_soft_kw(b"into")`. Used to detect the `INTO` of
    /// `UNPIVOT ... INTO NAME ...`. When the `dml` feature is on, `into` is separately
    /// reserved globally for `INSERT INTO` (`DML_KEYWORDS`), so `is_soft_kw` alone (which
    /// only looks at `Tok::Ident`) would fail to recognize `UNPIVOT ... INTO` in a
    /// `dml`-enabled build (see the doc comments on `expect_to`/`peek_is_to`; the same
    /// dual-representation problem).
    fn is_into(&self) -> bool {
        #[cfg(feature = "dml")]
        if self.is(Tok::Kw(Kw::Into)) {
            return true;
        }
        self.is_soft_kw(b"into")
    }

    /// Assuming `self.cur` is a comparison operator, decides without consuming anything
    /// whether what follows is `ANY|ALL|SOME (`.
    /// `Some(true)` = `ALL`, `Some(false)` = `ANY`/`SOME`, `None` = not a quantified
    /// comparison (the caller treats it as an ordinary infix operator).
    ///
    /// The decision waits until the `(` because `ANY`/`SOME` are not reserved words (see
    /// the docs on `is_soft_kw` and the ROWS/RANGE/QUALIFY incident comment at the top of
    /// `sql/lexer.rs`): a case where `any` is just a column name, as in `x > any`, must
    /// not be mistaken for `x > ANY (SELECT ...)`.
    /// It only clones `lex` for two tokens of lookahead and leaves `self`'s position
    /// alone (the same style as `peek`/`peek_is_soft_kw`).
    fn peek_quantifier(&self) -> Result<Option<bool>> {
        let mut lx = self.lex.clone();
        let t1 = lx.next_token()?.tok;
        let all = match t1 {
            Tok::Kw(Kw::All) => true,
            Tok::Ident(s)
                if eq_ascii_ci(s.as_bytes(), b"any") || eq_ascii_ci(s.as_bytes(), b"some") =>
            {
                false
            }
            _ => return Ok(None),
        };
        let t2 = lx.next_token()?.tok;
        if t2 == Tok::LParen {
            Ok(Some(all))
        } else {
            Ok(None)
        }
    }

    /// Assuming `self.cur` is the opening parenthesis of a function call, decides without
    /// consuming anything whether `want` appears at **the same nesting depth** before the
    /// matching closing parenthesis (like `peek_quantifier`, it only looks ahead on a clone of `lex`).
    ///
    /// Used to settle, before parsing the argument expressions themselves, whether this is
    /// a syntax that interposes a keyword in the argument list, as with the standard SQL
    /// `position(a IN b)` / `trim(BOTH x FROM s)`. Specifically for cases where `EXTRACT`'s
    /// two-token lookahead is not enough (an expression of arbitrary length precedes the keyword).
    ///
    /// The same word appearing in an inner call or subquery is rejected by depth (so a
    /// shape like `trim(f(x FROM y))` is not mistaken). The input is always finite, and the
    /// scan always stops at the closing parenthesis or `Eof`.
    fn call_has_top_level(&self, want: Tok<'a>) -> Result<bool> {
        let mut lx = self.lex.clone();
        // `self.lex` already points **past** `self.cur` (the opening parenthesis), so the
        // depth starts counting at 1.
        let mut depth = 1u32;
        loop {
            match lx.next_token()?.tok {
                Tok::Eof => return Ok(false),
                Tok::LParen | Tok::LBracket => depth += 1,
                Tok::RParen | Tok::RBracket => {
                    // Done once we reach the closing parenthesis matching `self.cur`.
                    if depth == 1 {
                        return Ok(false);
                    }
                    depth -= 1;
                }
                t if depth == 1 && t == want => return Ok(true),
                _ => {}
            }
        }
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

    /// Takes one identifier.
    ///
    /// Unquoted identifiers keep their original spelling. Comparison happens
    /// case-insensitively at bind time (`rt::hash::hash_ascii_ci`), so result column names keep the input's appearance.
    fn ident(&mut self) -> Result<String> {
        let s = match self.cur {
            Tok::Ident(s) => s.to_owned(),
            Tok::QIdent(s) => unquote(s, b'"'),
            _ => err!(UnexpectedToken, self.pos),
        };
        self.bump()?;
        Ok(s)
    }

    /// `[AS] alias`. Reserved words are `Tok::Kw`, so a clause boundary is never mistaken for an alias.
    fn opt_alias(&mut self) -> Result<Option<String>> {
        if self.eat_kw(Kw::As)? {
            return Ok(Some(self.ident()?));
        }
        match self.cur {
            // If a bare alias without `AS` swallowed what was actually the head of a
            // `USING SAMPLE`/`TABLESAMPLE` clause, `opt_sample_clause` could never be
            // reached (a perfectly ordinary form like `FROM t USING SAMPLE 10%` would
            // break). This is an accident-prone context like `SAMPLE`/`QUALIFY`, so just
            // these two words are looked ahead and excluded here (quoting still allows
            // them as aliases, as in `AS "using"`).
            Tok::Ident(s) if self.is_using_sample_or_tablesample(s.as_bytes())? => Ok(None),
            Tok::Ident(_) | Tok::QIdent(_) => Ok(Some(self.ident()?)),
            _ => Ok(None),
        }
    }

    /// Whether `self.cur` is the head of a `USING SAMPLE`/`TABLESAMPLE` clause.
    fn is_using_sample_or_tablesample(&self, word: &[u8]) -> Result<bool> {
        if eq_ascii_ci(word, b"tablesample") {
            return Ok(true);
        }
        if eq_ascii_ci(word, b"using") {
            return self.peek_is_soft_kw(b"sample");
        }
        Ok(false)
    }

    /// A non-negative integer. Used by LIMIT / OFFSET and by DECIMAL precision.
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

    /// One signed integer literal. Used by the arguments of `generate_series`/`range` and
    /// by the seed of `USING SAMPLE ... (method, seed)`. As with the unary-minus handling
    /// in `prefix()`, the sign is folded into the numeric literal before the range is
    /// checked (`int_literal` returns the first of `i32`/`i64`/`i128` that fits).
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

    // --- Statements ---------------------------------------------------------

    /// `sql` is the original text of the whole input. `ddl`'s `CREATE VIEW` uses it to
    /// keep the view body as raw text (see `create_view_stmt`). It is unused when the
    /// feature is OFF, but splitting the signature by cfg would mean two versions of just
    /// this function, so it is always taken and only the unused warning is suppressed.
    fn stmt(&mut self, sql: &'a str) -> Result<Stmt> {
        let _ = sql;
        let s = match self.cur {
            // A query starts with SELECT, WITH, or a parenthesis.
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
            // `PIVOT`/`UNPIVOT` are not reserved words; they are treated as keywords only
            // in this statement-head context (the same reason as the column-name-breaking
            // incidents around `ROWS`/`RANGE`/`QUALIFY` -- a valid statement always starts
            // with one of the keywords listed here, so a bare identifier "pivot"/"unpivot"
            // at the head can only be these two constructs, and ordinary uses as a column
            // or table name such as `FROM pivot` are not hindered at all).
            _ if self.is_soft_kw(b"pivot") => self.pivot_stmt()?,
            _ if self.is_soft_kw(b"unpivot") => self.unpivot_stmt()?,
            // v1 supports only the above. Every other unsupported statement is rejected here.
            _ => err!(UnsupportedFeature, self.pos),
        };
        self.eat(Tok::Semi)?;
        ensure!(self.is(Tok::Eof), UnexpectedToken, self.pos);
        Ok(s)
    }
}

impl<'a> Parser<'a> {
    // --- COPY (the `export` feature) -----------------------------------------

    /// `COPY (<query>) TO '<path>' [(FORMAT csv|jsonl|json)]` /
    /// `COPY <table> TO '<path>' [...]`
    ///
    /// With `export` alone, `TO`/`FORMAT` are matched by spelling as context-dependent
    /// keywords (`is_soft_kw`). They are not put in `export`'s global reserved-word table
    /// -- otherwise columns named `to`/`format` (a departure/arrival `to`, a `format`
    /// column naming a file format, and other names common in real data) would become
    /// unreferenceable without quotes. The same reason as `OVER`/`RECURSIVE` (see the
    /// `KEYWORDS` comment in `sql/lexer.rs`). `COPY` itself is a syntactic head used only
    /// at the start of a statement, so it is reserved normally like `CREATE`/`DROP`.
    ///
    /// When the `ddl` feature is on at the same time, `TO` is separately reserved globally
    /// for `ALTER TABLE ... RENAME TO` (`DDL_KEYWORDS`) and thus arrives as
    /// `Tok::Kw(Kw::To)`. `expect_to` accepts both.
    #[cfg(feature = "export")]
    fn copy_stmt(&mut self) -> Result<Stmt> {
        self.bump()?; // COPY
        let query = if self.eat(Tok::LParen)? {
            let q = self.query_stmt()?;
            self.expect(Tok::RParen)?;
            Box::new(q)
        } else {
            // `COPY <table> TO ...`. Builds a tree equivalent to `SELECT * FROM <table>`
            // and sends it down the same path as the subquery form (`write::copy`) from
            // there on (the same idea as turning `base_rel` into a derived table).
            let name = self.ident()?;
            let star = self.arena.push(Expr::Star {
                qualifier: None,
                columns: None,
                exclude: Vec::new(),
                replace: Vec::new(),
                rename: Vec::new(),
            });
            let mut s = SelectStmt::empty();
            s.items.push(SelectItem { expr: star, alias: None });
            s.from = Some(FromItem::Table { name, alias: None });
            Box::new(QueryStmt {
                ctes: Vec::new(),
                body: SetExpr::Select(Box::new(s)),
                order_by: Vec::new(),
                order_by_all: None,
                limit: None,
                offset: None,
            })
        };
        self.expect_to()?;
        let path = self.string_lit()?;
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

    /// The `TO` of `COPY`. The lexing of `to` changes depending on whether the `ddl`
    /// feature is on (see the `copy_stmt` docs above), so both forms are accepted.
    #[cfg(feature = "export")]
    fn expect_to(&mut self) -> Result<()> {
        #[cfg(feature = "ddl")]
        if self.eat_kw(Kw::To)? {
            return Ok(());
        }
        ensure!(self.is_soft_kw(b"to"), UnexpectedToken, self.pos);
        self.bump()
    }
}

mod expr;
mod select;
mod types;

#[cfg(feature = "ddl")]
mod ddl;
#[cfg(feature = "dml")]
mod dml;

#[cfg(test)]
mod tests;
