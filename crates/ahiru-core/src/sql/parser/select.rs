//! SELECT/FROM/JOIN/WHERE/GROUP BY/HAVING/ORDER BY/LIMIT, window definitions,
//! CTEs, set operations (UNION/INTERSECT/EXCEPT), and PIVOT/UNPIVOT parsing.
use super::types::{cube_sets, float_literal, int_literal, rollup_sets, sample_method_from_ident};
use super::*;

impl<'a> Parser<'a> {
    // --- クエリ（CTE + 集合演算 + 外側の ORDER BY / LIMIT）--------------------

    /// 深さを 1 段消費するクエリ。派生表・サブクエリ式経由で再帰しうる。
    ///
    /// クエリの入れ子はすべてここを通るので、深さの計上はこの 1 か所で足りる
    /// （`select_body` は必ず `query_body` の下から呼ばれる）。
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
        // 末尾の ORDER BY / LIMIT / OFFSET の置き場所は 1 つの規則で決める:
        // **本体が括弧無しの単一 SELECT なら `SelectStmt` 側**、それ以外
        // （集合演算がある、または本体が括弧付きクエリ）なら `QueryStmt` 側。
        // 括弧付きを除くのは、内側で既に自分の ORDER BY を持ちうるため。
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

    /// 先読み: `(` の直後がクエリの始まりか。
    ///
    /// スカラサブクエリ / `IN (SELECT ...)` / 括弧付き式の切り分けはこの 1 語
    /// だけで行う。`(` が続く場合（`((SELECT 1))`）は式・値リスト側に倒し、
    /// 内側の `(` が改めてこの判定を受ける。
    #[inline]
    pub(super) fn starts_query(&self) -> bool {
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
            // `GROUP BY ALL`（DuckDB 拡張）。`ALL` は `UNION ALL` 等で既に
            // 予約語（`Kw::All`）なので、文脈依存キーワードの判定は要らない。
            // 実際にどの式でグルーピングするかは束縛時に決める
            // （`SelectStmt::group_by_all` の doc 参照）。`GROUP BY ALL, x`
            // のような併記は DuckDB も構文エラーにするので、ここでもリストを
            // 読まない＝続く `,` が `UnexpectedToken` になる形で自然に弾かれる。
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

    /// `*`/`t.*` の直後に続きうる `EXCLUDE (col, ...)` / `REPLACE (expr AS
    /// col, ...)`（DuckDB 拡張）。この 2 語は `ROWS`/`RANGE`/`QUALIFY` と同種の
    /// 「実データにありふれた列名」なので、`*` の直後というこの文脈でだけ
    /// キーワードとして読む。順序は EXCLUDE → REPLACE 固定（`duckdb` で
    /// `REPLACE (...) EXCLUDE (...)` の逆順を試すと構文エラーになることを
    /// 確認済み）。カンマ区切りの複数指定は括弧必須だが、1 個だけなら括弧を
    /// 省略できる（`duckdb` の挙動に合わせた）。
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
        // 先頭の `*` だけは式ではなく列挙として扱う。`t.*` は primary 側。
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
        // `FROM 'path'`: 裸の文字列リテラルによるファイル参照
        // （`duckdb -c "SELECT * FROM 'x.csv'"` で構文を確認済み）。
        // フォーマットは拡張子から推定する（`FormatKind::detect`、既存の
        // Hive/multi-file 登録と同じ規則）。解決経路は下の `parquet(...)`/
        // `read_csv(...)` 等とまったく同じ（`FromItem::File` の doc 参照）。
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
        // CSV/JSON テーブル関数。`duckdb` CLI で確認した限り、`read_csv` と
        // `read_csv_auto`（`read_json`/`read_json_auto` も同様）は 1 引数の
        // 基本形では同じ結果になるので、この v1 では両方を同じ扱いにする
        // （named オプション引数は非対応 — `FromItem::File` の doc 参照）。
        // `csv`/`jsonl` フィーチャが無効なビルドでは該当フォーマットの読み取り
        // 自体が存在しないので、構文としても認識しない（未対応の soft
        // keyword は下の `ensure!` に落ちて `UnsupportedFeature` になる —
        // `ddl`/`dml`/`export` の文が feature 無効時に同じ経路へ落ちるのと
        // 同じパターン）。Parquet は常に使えるので `parquet`/`read_parquet`
        // はフィーチャ判定なし。
        #[cfg(feature = "csv")]
        let is_read_csv = self.is_soft_kw(b"read_csv") || self.is_soft_kw(b"read_csv_auto");
        #[cfg(not(feature = "csv"))]
        let is_read_csv = false;
        #[cfg(feature = "jsonl")]
        let is_read_json = self.is_soft_kw(b"read_json") || self.is_soft_kw(b"read_json_auto");
        #[cfg(not(feature = "jsonl"))]
        let is_read_json = false;
        // `UNNEST` も `parquet(...)` と同じ「非予約語だが `(` が続けば特殊構文」
        // という扱い。予約語化すると同名の列参照を壊す事故が過去にあった
        // （`ROWS`/`RANGE`/`QUALIFY`/`RECURSIVE`）ので踏襲する。
        let is_unnest = self.is_soft_kw(b"unnest");
        // `RANGE` はウィンドウ枠（`OVER (... RANGE BETWEEN ...)`）でも文脈依存
        // キーワードとして使われるが、そちらは `parse_window` の別の構文位置
        // （`ORDER BY` の直後）でしか見ないので、ここでテーブル関数として
        // 扱っても衝突しない（`is_soft_kw` の呼び出し元がそれぞれ独立している
        // ことを確認済み）。
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
                // `range` は半開区間・単項なら start=0、`generate_series` は
                // 閉区間・単項なら stop=args[0]（`duckdb` CLI で確認済み）。
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
                // `UNNEST` は常に 1 列を生む。複数列を返す DuckDB の
                // `UNNEST(struct)` 展開は未対応。
                let column_alias = self.opt_single_col_alias()?;
                return Ok(FromItem::Unnest { expr, alias, column_alias });
            }
            // ファイルテーブル関数: parquet/read_parquet/read_csv[_auto]/
            // read_json[_auto]。すべて 1 引数（パス文字列のみ）で、
            // 名前付きオプション引数（`delim=`, `header=`, ...）や複数引数・
            // glob 展開は非対応（`FromItem::File` の doc 参照。フォーマットの
            // 実体は登録時にホストが決めるため、この読み取り専用の束縛経路では
            // 再ディスパッチできない）。
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
    pub(super) fn pivot_stmt(&mut self) -> Result<Stmt> {
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
        let (order_by, order_by_all, limit, offset) = self.order_limit_offset_tail()?;

        // `PIVOT`/`UNPIVOT` の展開後クエリ（`plan::bind::desugar_pivot`）は
        // 出力列を組み立て直すので、`ORDER BY ALL` をそのまま持ち回れない。
        // 黙って無視すると並びが変わるため、明確に未対応として拒否する。
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

    /// `UNPIVOT <from> ON <col, ...> [INTO NAME <name> VALUE <value>]`。
    /// `UNPIVOT` キーワード自体は呼び出し元が確認済みで、まだ消費していない。
    pub(super) fn unpivot_stmt(&mut self) -> Result<Stmt> {
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

        // `PIVOT`/`UNPIVOT` の展開後クエリ（`plan::bind::desugar_pivot`）は
        // 出力列を組み立て直すので、`ORDER BY ALL` をそのまま持ち回れない。
        // 黙って無視すると並びが変わるため、明確に未対応として拒否する。
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

    /// 末尾の `ORDER BY <items> | ORDER BY ALL [ASC|DESC] [NULLS ...]`
    /// `[LIMIT n] [OFFSET n]`。`PIVOT`/`UNPIVOT` は集合演算も CTE も持たない
    /// 単純な文なので、`query_body` の同種の処理（こちらは `SetExpr`/`WITH`
    /// の分岐まで持つ）を簡略化した専用版。
    ///
    /// `ORDER BY ALL` は `ALL`（既存の予約語 `Kw::All`）1 語だけを取り、
    /// 通常の項目リストとは併記できない（DuckDB も `ORDER BY ALL, h` を構文
    /// エラーにする。ここではリストを読まないので、続く `,` が
    /// `UnexpectedToken` になって同じ結果になる）。
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
