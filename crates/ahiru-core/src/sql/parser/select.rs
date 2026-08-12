//! SELECT/FROM/JOIN/WHERE/GROUP BY/HAVING/ORDER BY/LIMIT, window definitions,
//! CTEs, set operations (UNION/INTERSECT/EXCEPT), and PIVOT/UNPIVOT parsing.
use super::types::{cube_sets, float_literal, int_literal, rollup_sets, sample_method_from_ident};
use super::*;

impl<'a> Parser<'a> {
    // --- Queries (CTEs + set operations + the outer ORDER BY / LIMIT) ---------

    /// A query that consumes one level of depth. It can recurse via derived tables and subquery expressions.
    ///
    /// Every nested query passes through here, so accounting for depth in this one place
    /// suffices (`select_body` is always called from beneath `query_body`).
    pub(super) fn query_stmt(&mut self) -> Result<QueryStmt> {
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

        let (order_by, order_by_all, limit, offset) = self.order_limit_offset_tail()?;

        let mut q = QueryStmt {
            ctes,
            body,
            order_by: Vec::new(),
            order_by_all: None,
            limit: None,
            offset: None,
        };
        // Where the trailing ORDER BY / LIMIT / OFFSET goes is decided by one rule:
        // **the `SelectStmt` side if the body is a single unparenthesized SELECT**, and the
        // `QueryStmt` side otherwise (a set operation is present, or the body is a
        // parenthesized query). Parenthesized bodies are excluded because they may already carry their own ORDER BY inside.
        match (&mut q.body, bare) {
            (SetExpr::Select(s), true) => {
                s.order_by = order_by;
                s.order_by_all = order_by_all;
                s.limit = limit;
                s.offset = offset;
            }
            _ => {
                q.order_by = order_by;
                q.order_by_all = order_by_all;
                q.limit = limit;
                q.offset = offset;
            }
        }
        Ok(q)
    }

    /// `RECURSIVE`, a context-dependent keyword meaningful only right after `WITH`.
    ///
    /// Not reserved globally, for the same reason as `OVER`/`ROWS`/`RANGE` (a CTE named
    /// `recursive` must be writable). `RECURSIVE` is always followed by a CTE name (an
    /// identifier), so if the next token is not an identifier this decides in favor of "a
    /// CTE named `recursive`" and does not consume it (`WITH recursive AS (...)` and
    /// `WITH recursive(a) AS (...)` are both valid SQL, matching DuckDB's behavior).
    fn eat_recursive_kw(&mut self) -> Result<bool> {
        if self.is_soft_kw(b"recursive") && matches!(self.peek()?, Tok::Ident(_) | Tok::QIdent(_)) {
            self.bump()?;
            return Ok(true);
        }
        Ok(false)
    }

    /// The body of `WITH`. A comma-separated list of `name [(col, ...)] AS ( query )`.
    ///
    /// `recursive` is whether this is a `WITH RECURSIVE` (the result of `eat_recursive_kw`).
    /// Per standard SQL the flag applies equally to every CTE in the list (whether one
    /// actually references itself is decided individually at bind time).
    fn cte_list(&mut self, recursive: bool) -> Result<Vec<Cte>> {
        let mut out = Vec::new();
        loop {
            let name = self.ident()?;
            // The column list of `name (a, b) AS (...)` is allowed only under
            // `WITH RECURSIVE` (column lists on non-recursive CTEs remain unsupported).
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

    /// The `UNION` / `EXCEPT` level. Built left-associatively.
    ///
    /// The returned `bool` is "was it a single unparenthesized SELECT". It decides whether
    /// the trailing ORDER BY / LIMIT may be dropped onto `SelectStmt`.
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

    /// The `INTERSECT` level. It binds tighter than `UNION` / `EXCEPT`, per the SQL standard.
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

    /// Drops a parenthesized query into a set-operation term.
    ///
    /// `SetExpr` cannot carry a CTE or an ORDER BY / LIMIT, so it is wrapped in
    /// `SELECT * FROM (...)` only when it has them. Otherwise the body is used as is (no
    /// needless projection is interposed).
    fn paren_body(&mut self, q: QueryStmt) -> SetExpr {
        if q.ctes.is_empty()
            && q.order_by.is_empty()
            && q.order_by_all.is_none()
            && q.limit.is_none()
            && q.offset.is_none()
        {
            return q.body;
        }
        let star = self.arena.push(Expr::Star {
            qualifier: None,
            columns: None,
            exclude: Vec::new(),
            replace: Vec::new(),
            rename: Vec::new(),
        });
        let mut s = SelectStmt::empty();
        s.items.push(SelectItem { expr: star, alias: None });
        s.from = Some(FromItem::Subquery { query: Box::new(q), alias: None });
        SetExpr::Select(Box::new(s))
    }

    /// Lookahead: whether what follows `(` is the start of a query.
    ///
    /// Telling apart a scalar subquery / `IN (SELECT ...)` / a parenthesized expression is
    /// done on this one word alone. A following `(` (`((SELECT 1))`) falls to the
    /// expression/value-list side, and the inner `(` gets this check afresh.
    #[inline]
    pub(super) fn starts_query(&self) -> bool {
        matches!(self.cur, Tok::Kw(Kw::Select | Kw::With))
    }

    fn select_body(&mut self) -> Result<SelectStmt> {
        // In this engine `PIVOT`/`UNPIVOT` are sugar recognized only at the head of a
        // statement (`stmt`), and the expansion
        // (`plan::bind::desugar_pivot`/`desugar_unpivot`) is by design done exactly once at
        // the entrance of `Session::prepare`, after resolving the target table's schema
        // (see the corresponding comment in `session.rs`). They therefore cannot be used as
        // a derived table, a CTE body, or a set-operation term, as in `FROM (PIVOT ...)`
        // (`duckdb` allows this; here it is out of scope). Letting it through would lead to
        // a bare `UnexpectedToken` further on where `SELECT` was expected, obscuring the
        // cause, so a clearer `UnsupportedFeature` is raised here first.
        ensure!(
            !self.is_soft_kw(b"pivot") && !self.is_soft_kw(b"unpivot"),
            UnsupportedFeature,
            self.pos
        );
        self.expect_kw(Kw::Select)?;
        let mut st = SelectStmt::empty();
        st.distinct = self.eat_kw(Kw::Distinct)?;
        // `DISTINCT ON (expr, ...)`. A PostgreSQL/DuckDB extension. `ON` is the same
        // reserved word as JOIN's `ON`, so it can be matched normally without worrying about column names.
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
            // The ON side performs the effective deduplication, so the ordinary DISTINCT
            // flag, which uses the whole group key, is not set at the same time.
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
            // `GROUPING SETS`/`ROLLUP`/`CUBE` are common words usable as column names too
            // (the same class as the incidents around `ROWS`/`RANGE`/`QUALIFY`), so they are
            // not reserved and are treated as keywords only in this position right after
            // `GROUP BY`. The two-word `GROUPING SETS` is distinguished with two tokens of lookahead.
            // `GROUP BY ALL` (a DuckDB extension). `ALL` is already reserved (`Kw::All`) by
            // `UNION ALL` and friends, so no context-dependent keyword check is needed.
            // Which expressions are actually grouped on is decided at bind time (see the
            // docs on `SelectStmt::group_by_all`). Writing both, as in `GROUP BY ALL, x`, is
            // a syntax error in DuckDB too, and is rejected naturally here as well: no list
            // is read, so the following `,` becomes an `UnexpectedToken`.
            if self.eat_kw(Kw::All)? {
                st.group_by_all = true;
            } else if self.is_soft_kw(b"grouping") && self.peek_is_soft_kw(b"sets")? {
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
        // `WINDOW name AS (...), ...`. Treated as a clause keyword on a par with
        // `GROUP BY`/`ORDER BY` (`Kw::Window` is an ordinary reserved word; see the comment
        // in `sql::lexer` for why `window` cannot be made context-dependent).
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
        // Unlike `OVER`/`PARTITION`/`ROWS`/`RANGE`, `QUALIFY` is not a common word that
        // could be a real data column name (it is a dedicated term from Teradata/SQL:1999),
        // so it is an ordinary reserved word. As a context-dependent keyword, placing it
        // directly after FROM without an intervening `WHERE`/`GROUP BY`, as in
        // `FROM t QUALIFY ...`, would have `opt_alias` eat `QUALIFY` as a table alias and
        // break the syntax (the same judgment as reserving `LIKE` without an intervening `ON`).
        if self.eat_kw(Kw::Qualify)? {
            st.qualify = Some(self.expr()?);
        }
        // `USING SAMPLE` is an independent clause on the statement-tail side (see the docs
        // on `opt_using_sample_clause`). It is accepted at a different position from
        // `TABLESAMPLE`, which attaches directly to a FROM item, so writing both at once is
        // rejected as a double specification (`duckdb` can apply both in order, but this
        // engine's `SampleSpec` is simplified to hold only one, so it is explicitly
        // rejected as out of scope).
        if let Some(spec) = self.opt_using_sample_clause()? {
            ensure!(st.sample.is_none(), UnsupportedFeature, self.pos);
            st.sample = Some(spec);
        }
        // ORDER BY / LIMIT / OFFSET are not read here. The right term of a set operation
        // would swallow the outer ORDER BY, so `query_body` handles them all together.
        Ok(st)
    }

    /// The body of `GROUPING SETS`, `( (expr, ...), (expr, ...), () )`.
    /// The empty set `()` is allowed as one set.
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

    /// `( expr, expr, ... )`. The empty list `()` is allowed too (`ROLLUP ()` and the like
    /// cannot be written, but it does appear as one element of `GROUPING SETS`).
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

    /// `REPLACE`. With the `ddl` feature on it is already a global reserved word
    /// (`Kw::Replace`) for `CREATE OR REPLACE`, so that form is accepted too.
    /// With it off, it is matched by spelling as a context-dependent keyword like `EXCLUDE`
    /// (so real data with columns named `replace`/`exclude` is not broken).
    #[inline]
    fn is_star_replace_kw(&self) -> bool {
        #[cfg(feature = "ddl")]
        if self.is(Tok::Kw(Kw::Replace)) {
            return true;
        }
        self.is_soft_kw(b"replace")
    }

    /// `RENAME`. Same rationale as `is_star_replace_kw` just above: under the
    /// `ddl` feature it is already a real reserved `Kw::Rename` (used by
    /// `ALTER TABLE ... RENAME`), so that form is accepted too. Otherwise it
    /// is a context-dependent keyword matched by spelling only, so a data set
    /// with a `rename` column still works without quoting.
    #[inline]
    fn is_star_rename_kw(&self) -> bool {
        #[cfg(feature = "ddl")]
        if self.is(Tok::Kw(Kw::Rename)) {
            return true;
        }
        self.is_soft_kw(b"rename")
    }

    /// The `EXCLUDE (col, ...)` / `REPLACE (expr AS col, ...)` (DuckDB extensions) that may
    /// follow `*`/`t.*`. These two words are of the same class as `ROWS`/`RANGE`/`QUALIFY`
    /// -- "column names common in real data" -- so they are read as keywords only in this
    /// position right after `*`. The order is fixed as EXCLUDE then REPLACE (confirmed that
    /// the reverse `REPLACE (...) EXCLUDE (...)` is a syntax error in `duckdb`).
    /// A comma-separated list requires parentheses, but a single entry may omit them
    /// (matching `duckdb`'s behavior).
    ///
    /// A third modifier, `RENAME (old AS new, ...)`, follows the same two
    /// rules and must come last: the fixed order is
    /// EXCLUDE -> REPLACE -> RENAME (verified against `duckdb` — either
    /// modifier appearing after RENAME is a parser error). That order falls
    /// out naturally from parsing the three blocks sequentially below: once
    /// RENAME has been consumed, a trailing EXCLUDE/REPLACE keyword is just
    /// leftover input and fails in the caller as an unexpected token, exactly
    /// like the already-tested REPLACE-before-EXCLUDE case.
    pub(super) fn star_modifiers(&mut self) -> Result<StarModifiers> {
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
        // Putting the same column in both EXCLUDE and REPLACE is meaningless (`duckdb`
        // rejects it too), so it is detected here.
        let pos = self.pos;
        for (_, name) in &replace {
            ensure!(
                !exclude.iter().any(|e| eq_ascii_ci(e.as_bytes(), name.as_bytes())),
                SyntaxError,
                pos
            );
        }
        // `RENAME (old AS new, ...)`. Unlike EXCLUDE/REPLACE, `old` is
        // resolved (or silently ignored if unknown) at bind time, not here —
        // see the `Expr::Star::rename` doc comment. What *is* checked here,
        // matching `duckdb`'s parser errors, is: the same source column
        // named twice within RENAME, and a source column that also appears
        // in EXCLUDE or REPLACE.
        let mut rename: Vec<(String, String)> = Vec::new();
        if self.is_star_rename_kw() {
            self.bump()?;
            if self.eat(Tok::LParen)? {
                loop {
                    let pos = self.pos;
                    let old = self.ident()?;
                    ensure!(
                        !rename.iter().any(|(o, _): &(String, String)| eq_ascii_ci(
                            o.as_bytes(),
                            old.as_bytes()
                        )),
                        SyntaxError,
                        pos
                    );
                    self.expect_kw(Kw::As)?;
                    let new = self.ident()?;
                    rename.push((old, new));
                    if !self.eat(Tok::Comma)? {
                        break;
                    }
                }
                self.expect(Tok::RParen)?;
            } else {
                let old = self.ident()?;
                self.expect_kw(Kw::As)?;
                let new = self.ident()?;
                rename.push((old, new));
            }
        }
        let pos = self.pos;
        for (old, _) in &rename {
            ensure!(
                !exclude.iter().any(|e| eq_ascii_ci(e.as_bytes(), old.as_bytes())),
                SyntaxError,
                pos
            );
            ensure!(
                !replace
                    .iter()
                    .any(|(_, n): &(ExprId, String)| eq_ascii_ci(n.as_bytes(), old.as_bytes())),
                SyntaxError,
                pos
            );
        }
        Ok((exclude, replace, rename))
    }

    /// `self.cur` starts a `COLUMNS(...)` star expression.
    ///
    /// `COLUMNS` is a context-dependent keyword, recognized only at the start
    /// of a select-list item and only when immediately followed by `(` — the
    /// same rule `EXCLUDE`/`REPLACE`/`RENAME`/`FILTER`/`OVER` follow, and for
    /// the same reason (`sql/lexer.rs`'s `KEYWORDS` comment): column names
    /// come from data files, so a column literally named `columns` has to
    /// keep working unquoted. `duckdb` agrees — `SELECT columns FROM u` on a
    /// table with a `columns` column works there too.
    #[inline]
    fn is_columns_kw(&self) -> Result<bool> {
        Ok(self.is_soft_kw(b"columns") && self.peek()? == Tok::LParen)
    }

    /// The body of a `COLUMNS(...)` select-list item, positioned at `COLUMNS`.
    ///
    /// Three argument forms are supported, all verified against `duckdb`
    /// v1.4.4 (see `sql::ast::ColumnsSpec` for the resolved semantics of
    /// each): `COLUMNS(*)` — optionally with the ordinary star modifiers
    /// *inside* the parentheses — `COLUMNS('regex')`, and
    /// `COLUMNS(['a', 'b'])`.
    ///
    /// DuckDB's `COLUMNS(c -> <predicate>)` lambda form is deliberately
    /// rejected here as `UnsupportedFeature` rather than being left to fail
    /// as a confusing token error: the argument would have to be evaluated
    /// per column name at bind time, which is a different mechanism from the
    /// name matching the other three forms share.
    fn columns_item(&mut self) -> Result<SelectItem> {
        self.bump()?; // COLUMNS
        self.expect(Tok::LParen)?;
        let pos = self.pos;
        let (columns, exclude, replace, rename) = if self.eat(Tok::Star)? {
            let (exclude, replace, rename) = self.star_modifiers()?;
            (ColumnsSpec::All, exclude, replace, rename)
        } else if self.is(Tok::LBracket) {
            self.bump()?; // '['
            let mut names: Vec<String> = Vec::new();
            loop {
                names.push(self.string_lit()?);
                if !self.eat(Tok::Comma)? {
                    break;
                }
            }
            self.expect(Tok::RBracket)?;
            (ColumnsSpec::Names(names), Vec::new(), Vec::new(), Vec::new())
        } else if matches!(self.cur, Tok::Str(_)) {
            (ColumnsSpec::Regex(self.string_lit()?), Vec::new(), Vec::new(), Vec::new())
        } else {
            // Covers `COLUMNS(c -> ...)` (the lambda form) and anything else
            // that isn't one of the three supported arguments. `duckdb`
            // rejects an empty `COLUMNS()` too.
            err!(UnsupportedFeature, pos)
        };
        self.expect(Tok::RParen)?;
        // A `COLUMNS(...)` item is a whole select-list entry, never an operand:
        // DuckDB *does* distribute an enclosing expression over the expansion
        // (`COLUMNS(*) + 1` yields one `+ 1` column per input column), which
        // this engine cannot do — the binder would have to synthesize one
        // expression node per expanded column, and the arena is immutable by
        // the time the input schema is known. Everything that can legitimately
        // follow here is a keyword (`FROM`, `AS`, ...), an alias identifier, a
        // comma, or the end of the item list; an operator token means the
        // distributing form, so it gets the same `UnsupportedFeature` as
        // `min(COLUMNS(*))` instead of an "unexpected token" on the operator.
        let pos = self.pos;
        ensure!(
            matches!(
                self.cur,
                Tok::Eof
                    | Tok::Kw(_)
                    | Tok::Ident(_)
                    | Tok::QIdent(_)
                    | Tok::Comma
                    | Tok::RParen
                    | Tok::Semi
            ),
            UnsupportedFeature,
            pos
        );
        // `AS '<template>'` is the capture-group renaming form
        // (`COLUMNS('(\w{3}).*') AS '\1'`), so the alias here may be a string
        // literal, which the shared `opt_alias` deliberately does not accept.
        let alias = if self.eat_kw(Kw::As)? {
            Some(match self.cur {
                Tok::Str(_) => self.string_lit()?,
                _ => self.ident()?,
            })
        } else {
            self.opt_alias()?
        };
        let expr = self.arena.push(Expr::Star {
            qualifier: None,
            columns: Some(columns),
            exclude,
            replace,
            rename,
        });
        Ok(SelectItem { expr, alias })
    }

    fn select_item(&mut self) -> Result<SelectItem> {
        if self.is_columns_kw()? {
            return self.columns_item();
        }
        // Only a leading `*` is treated as an enumeration rather than an expression. `t.*` is handled on the primary side.
        if self.is(Tok::Star) {
            self.bump()?;
            // `*COLUMNS(...)` (DuckDB's unpacking form) and `* LIKE 'pat'` /
            // `* GLOB ...` / `* SIMILAR TO ...` (star filtering) are both
            // unsupported. Caught explicitly so they fail as
            // `UnsupportedFeature` instead of as an "unexpected token" from
            // the enclosing item list, which would read as if the syntax were
            // simply malformed.
            let pos = self.pos;
            ensure!(!self.is_columns_kw()?, UnsupportedFeature, pos);
            let (exclude, replace, rename) = self.star_modifiers()?;
            let pos = self.pos;
            ensure!(!self.is_star_filter_kw(), UnsupportedFeature, pos);
            let expr = self.arena.push(Expr::Star {
                qualifier: None,
                columns: None,
                exclude,
                replace,
                rename,
            });
            return Ok(SelectItem { expr, alias: None });
        }
        let expr = self.expr()?;
        let alias = self.opt_alias()?;
        Ok(SelectItem { expr, alias })
    }

    /// `self.cur` starts one of DuckDB's star-filtering operators
    /// (`* LIKE 'pat'`, `* ILIKE`, `* GLOB`, `* SIMILAR TO`), which this
    /// engine does not implement. Only used to turn those into a clear
    /// `UnsupportedFeature`; `NOT` is included so `* NOT LIKE ...` reports the
    /// same thing.
    #[inline]
    fn is_star_filter_kw(&self) -> bool {
        self.is(Tok::Kw(Kw::Like))
            || self.is(Tok::Kw(Kw::Ilike))
            || self.is(Tok::Kw(Kw::Not))
            || self.is_soft_kw(b"glob")
            || self.is_soft_kw(b"similar")
    }

    pub(super) fn order_item(&mut self) -> Result<OrderByItem> {
        let expr = self.expr()?;
        let (desc, nulls_first) = self.order_direction()?;
        Ok(OrderByItem { expr, desc, nulls_first })
    }

    /// `[ASC | DESC] [NULLS FIRST | NULLS LAST]`. Shared by `order_item` and
    /// by `ORDER BY ALL` (which takes the same modifiers and applies them to
    /// every output column).
    ///
    /// Default matches DuckDB's actual behavior: NULLS LAST regardless of
    /// ASC/DESC (verified against a real `duckdb` CLI) — not the
    /// SQL-standard/PostgreSQL convention of "NULL is the largest value"
    /// (NULLS LAST for ASC, NULLS FIRST for DESC), which this used to
    /// implement and which silently disagreed with the reference
    /// implementation this project cross-checks against.
    fn order_direction(&mut self) -> Result<(bool, bool)> {
        let mut desc = false;
        if self.eat_kw(Kw::Desc)? {
            desc = true;
        } else {
            self.eat_kw(Kw::Asc)?;
        }
        let mut nulls_first = false;
        if self.eat_kw(Kw::Nulls)? {
            if self.eat_kw(Kw::First)? {
                nulls_first = true;
            } else {
                self.expect_kw(Kw::Last)?;
                nulls_first = false;
            }
        }
        Ok((desc, nulls_first))
    }

    // --- FROM ---------------------------------------------------------------

    pub(super) fn parse_from_item(&mut self) -> Result<FromItem> {
        let mut left = self.base_rel()?;
        loop {
            // Only explicit JOINs require ON. CROSS and the implicit comma join are unconditional.
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
            // A left-deep `Box` chain that grows too long exhausts the stack when dropped.
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
            // A derived table takes a whole query. Set operations and CTEs may be written.
            let query = self.query_stmt()?;
            self.expect(Tok::RParen)?;
            let alias = self.opt_alias()?;
            return Ok(FromItem::Subquery { query: Box::new(query), alias });
        }
        // `FROM 'path'`: a file reference by a bare string literal
        // (syntax confirmed with `duckdb -c "SELECT * FROM 'x.csv'"`).
        // The format is inferred from the extension (`FormatKind::detect`, the same rule as
        // the existing Hive/multi-file registration). The resolution path is exactly the
        // same as `parquet(...)`/`read_csv(...)` below (see the `FromItem::File` docs).
        if let Tok::Str(s) = self.cur {
            let path = unquote(s, b'\'');
            self.bump()?;
            let format = FormatKind::detect(&path);
            let alias = self.opt_alias()?;
            return Ok(FromItem::File { path, format, alias });
        }
        let pos = self.pos;
        let is_parquet = self.is_soft_kw(b"parquet");
        let is_read_parquet = self.is_soft_kw(b"read_parquet");
        // CSV/JSON table functions. As far as the `duckdb` CLI shows, `read_csv` and
        // `read_csv_auto` (likewise `read_json`/`read_json_auto`) give the same result in
        // the basic one-argument form, so v1 treats them identically (named option
        // arguments are unsupported -- see the `FromItem::File` docs).
        // In builds where the `csv`/`jsonl` features are off, reading those formats does not
        // exist at all, so they are not recognized syntactically either (an unsupported soft
        // keyword falls through to the `ensure!` below and becomes `UnsupportedFeature` --
        // the same pattern as `ddl`/`dml`/`export` statements falling down the same path
        // when their features are off). Parquet is always available, so `parquet`/`read_parquet`
        // have no feature check.
        #[cfg(feature = "csv")]
        let is_read_csv = self.is_soft_kw(b"read_csv") || self.is_soft_kw(b"read_csv_auto");
        #[cfg(not(feature = "csv"))]
        let is_read_csv = false;
        #[cfg(feature = "jsonl")]
        let is_read_json = self.is_soft_kw(b"read_json") || self.is_soft_kw(b"read_json_auto");
        #[cfg(not(feature = "jsonl"))]
        let is_read_json = false;
        // `UNNEST` gets the same "not reserved, but special syntax if `(` follows" treatment
        // as `parquet(...)`. Reserving it has caused incidents breaking same-named column
        // references in the past (`ROWS`/`RANGE`/`QUALIFY`/`RECURSIVE`), so that is followed here.
        let is_unnest = self.is_soft_kw(b"unnest");
        // `RANGE` is also used as a context-dependent keyword in window frames
        // (`OVER (... RANGE BETWEEN ...)`), but that is only looked at in a different
        // syntactic position in `parse_window` (right after `ORDER BY`), so treating it as a
        // table function here does not collide (confirmed that the call sites of
        // `is_soft_kw` are independent of one another).
        let is_generate_series = self.is_soft_kw(b"generate_series");
        let is_range = self.is_soft_kw(b"range");
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
                // `range` is half-open, and unary means start=0; `generate_series` is closed,
                // and unary means stop=args[0] (confirmed with the `duckdb` CLI).
                let (start, stop, step) = match args.len() {
                    1 => (0, args[0], 1),
                    2 => (args[0], args[1], 1),
                    _ => (args[0], args[1], args[2]),
                };
                let alias = self.opt_alias()?;
                let column_alias = self.opt_single_col_alias()?;
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
                // `UNNEST` always produces one column. DuckDB's multi-column
                // `UNNEST(struct)` expansion is unsupported.
                let column_alias = self.opt_single_col_alias()?;
                return Ok(FromItem::Unnest { expr, alias, column_alias });
            }
            // File table functions: parquet/read_parquet/read_csv[_auto]/read_json[_auto].
            // All take one argument (a path string only); named option arguments
            // (`delim=`, `header=`, ...), multiple arguments, and glob expansion are
            // unsupported (see the `FromItem::File` docs; the actual format is decided by the
            // host at registration, so this read-only binding path cannot re-dispatch on it).
            let is_file_fn = is_parquet || is_read_parquet || is_read_csv || is_read_json;
            ensure!(is_file_fn, UnsupportedFeature, pos);
            self.bump()?;
            let path = self.string_lit()?;
            self.expect(Tok::RParen)?;
            let alias = self.opt_alias()?;
            let format = if is_parquet || is_read_parquet {
                FormatKind::Parquet
            } else if is_read_csv {
                FormatKind::Csv
            } else {
                FormatKind::Json
            };
            return Ok(FromItem::File { path, format, alias });
        }
        let alias = self.opt_alias()?;
        Ok(FromItem::Table { name, alias })
    }

    // --- SAMPLE ---------------------------------------------------------------
    //
    // `SAMPLE`/`USING`/`TABLESAMPLE`/`BERNOULLI`/`SYSTEM`/`RESERVOIR`/`ROWS`/
    // None of `PERCENT` and friends are reserved. For the same reason as the
    // `ROWS`/`RANGE`/`QUALIFY` incidents (see the comment at the top of the file), they are
    // matched by spelling only at the fixed position right after `FROM <item>`. As the
    // `duckdb` CLI shows, `SAMPLE` on its own (without `USING`/`TABLESAMPLE`) appears
    // nowhere in the grammar, so a column named `SAMPLE` is in no danger either.

    /// `TABLESAMPLE <body>`. The positional constraint confirmed with the `duckdb` CLI:
    /// `TABLESAMPLE` is a modifier attaching directly to a FROM item and must always come
    /// before `WHERE`/`GROUP BY`/... (right after the FROM item), as in
    /// `FROM t TABLESAMPLE 10% WHERE ...`. Placing it after `WHERE` is a syntax error in
    /// `duckdb` too (`duckdb -c "... WHERE ... TABLESAMPLE 10%"` -> `syntax error at or near`
    /// `"TABLESAMPLE"`), so it is called only from the caller right after a FROM item.
    fn opt_tablesample_clause(&mut self) -> Result<Option<SampleSpec>> {
        if !self.is_soft_kw(b"tablesample") {
            return Ok(None);
        }
        self.bump()?; // tablesample
        Ok(Some(self.sample_body()?))
    }

    /// `USING SAMPLE <body>`. Unlike `TABLESAMPLE`, this is an independent clause over the
    /// whole statement, placed after `WHERE`/`GROUP BY`/`HAVING`/`WINDOW`/`QUALIFY` and
    /// before `ORDER BY` (confirmed with the `duckdb` CLI: `FROM t USING SAMPLE 10% WHERE
    /// ...` is a syntax error while `FROM t WHERE ... USING SAMPLE 10%` parses). Written
    /// directly after a FROM item with no intervening `WHERE` and the like, it lands at the
    /// same position as `opt_tablesample_clause`, but the call is always made only from
    /// here (the statement-tail side, right after `QUALIFY`).
    fn opt_using_sample_clause(&mut self) -> Result<Option<SampleSpec>> {
        if !(self.is_soft_kw(b"using") && self.peek_is_soft_kw(b"sample")?) {
            return Ok(None);
        }
        self.bump()?; // using
        self.bump()?; // sample
        Ok(Some(self.sample_body()?))
    }

    /// `<method>(<amount>[unit])` / `(<amount>[unit])` / `<amount>[unit]
    /// `[(<method>, <seed>)]` -- one of the three forms confirmed with the `duckdb` CLI.
    fn sample_body(&mut self) -> Result<SampleSpec> {
        if let Tok::Ident(s) = self.cur {
            if let Some(method) = sample_method_from_ident(s.as_bytes()) {
                if self.peek()? == Tok::LParen {
                    self.bump()?; // the method name
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
        // Without a unit (a bare number) it is treated as a row count (`duckdb`'s measured behavior).
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

    /// `<number>['%' | PERCENT | ROWS]`. Without a unit it is treated as a row count
    /// (`duckdb`'s measured behavior: `USING SAMPLE 100` means 100 rows, and
    /// `USING SAMPLE 0.1` is interpreted as a row count, not a percentage).
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
            // `Tok::Int`/`Tok::Float` carry no sign (`-` is a separate token and never
            // reaches here), so `amount` is always non-negative. No check rejecting a
            // negative row count is needed.
        } else {
            // `duckdb`: "Sample sample_size ... out of range, must be between 0 and 100"
            ensure!((0.0..=100.0).contains(&amount), SyntaxError, pos);
        }
        Ok((amount, is_rows))
    }

    // --- PIVOT/UNPIVOT -------------------------------------------------------
    //
    // None of `PIVOT`/`UNPIVOT`/`USING`/`INTO`/`NAME`/`VALUE` are reserved.
    // They are matched by spelling only at the head of a statement (`PIVOT`/`UNPIVOT`) or
    // at fixed positions inside each construct (the `USING` right after `ON`, the
    // `INTO NAME .. VALUE ..` right after the column list). The same reason as the past
    // incident where reserving `ROWS`/`RANGE`/`QUALIFY` broke same-named columns (see the comment at the top of `sql::lexer`).

    /// `PIVOT <from> ON <on> [IN (...)] USING <agg>[, ...] [GROUP BY <cols>]`.
    /// The `PIVOT` keyword itself has been confirmed by the caller (`stmt`) and is not yet
    /// consumed.
    pub(super) fn pivot_stmt(&mut self) -> Result<Stmt> {
        self.bump()?; // PIVOT
        let from = self.parse_from_item()?;
        self.expect_kw(Kw::On)?;
        // We want to stop just before `IN (...)`, so the `IN` predicate (of the same binding
        // power as comparison) is not swallowed (nothing below `BP_CMP + 1` is combined).
        // Multiple columns as in `ON a, b` are unsupported -- the comma is left over, so it
        // matches neither IN nor USING afterwards and naturally becomes a syntax error.
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
            // As in DuckDB, omitting `USING` defaults to `count(*)`
            // (`plan::bind::desugar_pivot` actually supplies it).
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
        let (order_by, order_by_all, limit, offset) = self.order_limit_offset_tail()?;

        // The post-expansion query of `PIVOT`/`UNPIVOT` (`plan::bind::desugar_pivot`)
        // reassembles the output columns, so `ORDER BY ALL` cannot be carried through as is.
        // Silently ignoring it would change the ordering, so it is clearly rejected as unsupported.
        ensure!(order_by_all.is_none(), UnsupportedFeature, self.pos);
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

    /// `UNPIVOT <from> ON <col, ...> [INTO NAME <name> VALUE <value>]`.
    /// The `UNPIVOT` keyword itself has been confirmed by the caller and is not yet consumed.
    pub(super) fn unpivot_stmt(&mut self) -> Result<Stmt> {
        self.bump()?; // UNPIVOT
        let from = self.parse_from_item()?;
        self.expect_kw(Kw::On)?;
        // Folding several columns at once, as in `(a, b), (c, d)`, is unsupported. Only a
        // comma-separated list of bare column references is accepted (expressions and
        // `t.col` also parse, but expansion (`desugar_unpivot`) rejects anything that is not a bare column reference).
        let mut columns = Vec::new();
        loop {
            columns.push(self.expr()?);
            if !self.eat(Tok::Comma)? {
                break;
            }
        }
        let (name_col, value_col) = if self.is_into() {
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
        let (order_by, order_by_all, limit, offset) = self.order_limit_offset_tail()?;

        // The post-expansion query of `PIVOT`/`UNPIVOT` (`plan::bind::desugar_pivot`)
        // reassembles the output columns, so `ORDER BY ALL` cannot be carried through as is.
        // Silently ignoring it would change the ordering, so it is clearly rejected as unsupported.
        ensure!(order_by_all.is_none(), UnsupportedFeature, self.pos);
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

    /// The trailing `ORDER BY <items> | ORDER BY ALL [ASC|DESC] [NULLS ...]`
    /// `[LIMIT n] [OFFSET n]`. `PIVOT`/`UNPIVOT` are simple statements with neither set
    /// operations nor CTEs, so this is a simplified dedicated version of the equivalent
    /// handling in `query_body` (which also branches on `SetExpr`/`WITH`).
    ///
    /// `ORDER BY ALL` takes only the single word `ALL` (the existing reserved word
    /// `Kw::All`) and cannot be combined with an ordinary item list (DuckDB makes
    /// `ORDER BY ALL, h` a syntax error too. No list is read here, so the following `,`
    /// becomes an `UnexpectedToken`, giving the same result).
    #[allow(clippy::type_complexity)]
    fn order_limit_offset_tail(
        &mut self,
    ) -> Result<(Vec<OrderByItem>, Option<OrderByAll>, Option<u64>, Option<u64>)> {
        let mut order_by = Vec::new();
        let mut order_by_all = None;
        if self.eat_kw(Kw::Order)? {
            self.expect_kw(Kw::By)?;
            if self.eat_kw(Kw::All)? {
                let (desc, nulls_first) = self.order_direction()?;
                order_by_all = Some(OrderByAll { desc, nulls_first });
            } else {
                loop {
                    order_by.push(self.order_item()?);
                    if !self.eat(Tok::Comma)? {
                        break;
                    }
                }
            }
        }
        let limit = if self.eat_kw(Kw::Limit)? { Some(self.uint()?) } else { None };
        let offset = if self.eat_kw(Kw::Offset)? { Some(self.uint()?) } else { None };
        Ok((order_by, order_by_all, limit, offset))
    }
}
