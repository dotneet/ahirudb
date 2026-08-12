//! Expression parsing: Pratt precedence climbing, prefix/primary/postfix,
//! CAST, CASE, window function calls, lambdas, and literal parsing helpers.
use super::types::{
    comparison_binop, float_literal, int_literal, is_lambda_func, lookup_interval_unit,
    lookup_type, parse_interval_text, parse_signed_int, unit_to_interval, unquote,
};
use super::*;

impl<'a> Parser<'a> {
    // --- 式 -----------------------------------------------------------------

    pub(super) fn expr(&mut self) -> Result<ExprId> {
        self.expr_bp(0)
    }

    /// 深さを 1 段消費してから本体へ。エラー経路でも必ず戻すため薄く包む。
    pub(super) fn expr_bp(&mut self, min_bp: u8) -> Result<ExprId> {
        ensure!(self.depth < MAX_DEPTH, ExpressionTooDeep, self.pos);
        self.depth += 1;
        let r = self.expr_body(min_bp);
        self.depth -= 1;
        r
    }

    fn expr_body(&mut self, min_bp: u8) -> Result<ExprId> {
        let mut lhs = self.prefix()?;
        loop {
            // `x <op> ANY|ALL|SOME (SELECT ...)`。比較演算子の直後にだけ現れる
            // 量化比較で、比較と同じ強さの中置演算子として扱う。`ALL` は既存の
            // 予約語（`UNION ALL` 等）だが、`ANY`/`SOME` は `glob`/`similar` と
            // 同様に予約語表に入れていない（列名としても使えるように）ので、
            // ここで 2 トークン先読みして `(` まで確認してから確定させる
            // （`x > any_col` のような普通の列参照を誤認しないため。
            // `peek_quantifier` の doc 参照）。
            if let Some(op) = comparison_binop(self.cur) {
                if BP_CMP >= min_bp {
                    if let Some(all) = self.peek_quantifier()? {
                        self.bump()?; // 比較演算子
                        self.bump()?; // ANY/ALL/SOME
                        self.expect(Tok::LParen)?;
                        let query = self.query_stmt()?;
                        self.expect(Tok::RParen)?;
                        lhs = self.arena.push(Expr::QuantifiedComparison {
                            op,
                            arg: lhs,
                            all,
                            query: Box::new(query),
                        });
                        continue;
                    }
                }
            }
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
                lhs = self.simple_call("glob", vec![lhs, pattern]);
                continue;
            }
            // `SIMILAR TO` は `LIKE` 同様 `[NOT]` を前置できるので、`predicate()`
            // 側（`Tok::Kw(Kw::Not)` 分岐）にも同じ判定を足してある。
            if self.is_soft_kw(b"similar") && self.peek_is_to()? {
                if BP_CMP < min_bp {
                    break;
                }
                lhs = self.similar_to(lhs, false)?;
                continue;
            }
            // `ISNULL`/`NOTNULL`: DuckDB's non-standard postfix aliases for
            // `IS [NOT] NULL`. Soft keywords for the same reason as
            // `glob`/`similar`/`filter`/`over` above (not in the reserved
            // table) — `isnull`/`notnull` must stay usable as column names.
            // That's safe here because both uses of the word are read
            // through different code paths: `SELECT isnull FROM t` never
            // reaches this loop (the bare identifier is consumed whole by
            // `primary_atom`/`name_ref` as the *operand*, before this
            // infix/postfix loop ever sees the token as `self.cur`), and
            // `SELECT 1 AS isnull` reads the alias via a separate `ident()`
            // call in `opt_alias` after `expr()` has already returned.
            if self.is_soft_kw(b"isnull") {
                if BP_CMP < min_bp {
                    break;
                }
                self.bump()?;
                lhs = self.arena.push(Expr::IsNull { arg: lhs, negated: false });
                continue;
            }
            if self.is_soft_kw(b"notnull") {
                if BP_CMP < min_bp {
                    break;
                }
                self.bump()?;
                lhs = self.arena.push(Expr::IsNull { arg: lhs, negated: true });
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
                // `//` (integer division). Sugar for plain `/`, not a new
                // `BinaryOp` variant: this engine's `/` is *already*
                // truncating integer division when both operands are
                // integers (`7/2` = 3, `-7/2` = -3, matching DuckDB's `//`
                // exactly), and stays real-valued division when either
                // operand is a float (`5.0/2` = 2.5). That's the same
                // behavior DuckDB gives `//` specifically (its plain `/`
                // instead always returns a float, e.g. `7/2` = 3.5 — a
                // pre-existing, out-of-scope divergence from DuckDB noted
                // in docs/sql/functions-numeric.md). If `/`'s semantics
                // ever change, this alias must be revisited.
                Tok::SlashSlash => (BinaryOp::Div, BP_MUL),
                Tok::Percent => (BinaryOp::Mod, BP_MUL),
                // `&`/`|`/`<<`/`>>`/`^`/`**` も `->`/`->>` と同じく新しい
                // `BinaryOp` を増やさず、既存のスカラ関数呼び出しへの糖衣構文
                // として展開する（`bit_and`/`bit_or`/`bit_shift_left`/
                // `bit_shift_right`/`pow` は `expr::funcs` に既存、または
                // このコミットで新設。カーネルを増やさない、という
                // DESIGN.md §11 の方針の適用）。
                Tok::Amp => {
                    if BP_BITWISE < min_bp {
                        break;
                    }
                    self.bump()?;
                    let rhs = self.expr_bp(BP_BITWISE + 1)?;
                    lhs = self.simple_call("bit_and", vec![lhs, rhs]);
                    continue;
                }
                Tok::Pipe => {
                    if BP_BITWISE < min_bp {
                        break;
                    }
                    self.bump()?;
                    let rhs = self.expr_bp(BP_BITWISE + 1)?;
                    lhs = self.simple_call("bit_or", vec![lhs, rhs]);
                    continue;
                }
                Tok::Shl => {
                    if BP_BITWISE < min_bp {
                        break;
                    }
                    self.bump()?;
                    let rhs = self.expr_bp(BP_BITWISE + 1)?;
                    lhs = self.simple_call("bit_shift_left", vec![lhs, rhs]);
                    continue;
                }
                Tok::Shr => {
                    if BP_BITWISE < min_bp {
                        break;
                    }
                    self.bump()?;
                    let rhs = self.expr_bp(BP_BITWISE + 1)?;
                    lhs = self.simple_call("bit_shift_right", vec![lhs, rhs]);
                    continue;
                }
                // 左結合（`duckdb` の `2^3^2` = `(2^3)^2` を確認済み、BP 定数の
                // doc 参照）なので、通常の演算子と同じく `expr_bp(bp + 1)` で
                // 右辺を読む。
                Tok::Pow => {
                    if BP_POW < min_bp {
                        break;
                    }
                    self.bump()?;
                    let rhs = self.expr_bp(BP_POW + 1)?;
                    lhs = self.simple_call("pow", vec![lhs, rhs]);
                    continue;
                }
                // 中置の `~`/`!~`。前置の `~`（ビット単位 NOT）は `prefix()` が
                // 別に処理するので、ここに来るのは必ず中置（正規表現一致）。
                // `SIMILAR TO` と同じ関数に展開する（`similar_to` の doc 参照）。
                Tok::Tilde | Tok::NotTilde => {
                    if BP_CMP < min_bp {
                        break;
                    }
                    let negate = self.cur == Tok::NotTilde;
                    self.bump()?;
                    let rhs = self.expr_bp(BP_CMP + 1)?;
                    let call = self.simple_call("regexp_full_match", vec![lhs, rhs]);
                    lhs = if negate {
                        self.arena.push(Expr::Unary { op: UnaryOp::Not, arg: call })
                    } else {
                        call
                    };
                    continue;
                }
                // `~~`/`!~~`/`~~*`/`!~~*`: PostgreSQL/DuckDB's punctuation
                // aliases for `LIKE`/`NOT LIKE`/`ILIKE`/`NOT ILIKE`. Desugar
                // straight into `Expr::Like` (same node the `LIKE`/`ILIKE`
                // keywords produce in `predicate()` below) rather than a
                // fresh AST variant.
                //
                // The right operand is read at `BP_CONCAT + 1`, one notch
                // *tighter* than `LIKE`'s own `BP_CONCAT` (see
                // `predicate()`'s `Kw::Like | Kw::Ilike` arm) — this is a
                // real, verified difference from the `LIKE` keyword, not a
                // copy-paste slip:
                //   duckdb: 'ab' LIKE 'a' || 'b'  -> true      i.e. 'ab' LIKE ('a'||'b')
                //   duckdb: 'ab' ~~   'a' || 'b'  -> 'falseb'  i.e. ('ab' ~~ 'a') || 'b'
                // So a `||` right after one of these operators binds to the
                // *result*, not into the pattern. `ESCAPE` is intentionally
                // not accepted here (only the `LIKE`/`ILIKE` keyword forms
                // take it) — duckdb rejects `'a%c' ~~ 'a$%c' ESCAPE '$'` as
                // a parse error, confirmed against the `duckdb` CLI.
                Tok::TildeTilde
                | Tok::NotTildeTilde
                | Tok::TildeTildeStar
                | Tok::NotTildeTildeStar => {
                    if BP_CMP < min_bp {
                        break;
                    }
                    let negated = matches!(self.cur, Tok::NotTildeTilde | Tok::NotTildeTildeStar);
                    let ci = matches!(self.cur, Tok::TildeTildeStar | Tok::NotTildeTildeStar);
                    self.bump()?;
                    let pattern = self.expr_bp(BP_CONCAT + 1)?;
                    lhs = self.arena.push(Expr::Like {
                        arg: lhs,
                        pattern,
                        negated,
                        escape: None,
                        ci,
                    });
                    continue;
                }
                // `~~~`: alias for `GLOB`. Same "bind tighter than `||`"
                // quirk as the `~~` family above (verified: `duckdb -c
                // "select 'ab' ~~~ 'a' || '*'"` -> `false*`, i.e. `('ab' ~~~
                // 'a') || '*'`, while `'ab' GLOB 'a' || '*'` -> `true`, i.e.
                // `'ab' GLOB ('a'||'*')` — the `GLOB` keyword itself reads
                // its pattern at `BP_CONCAT`, see the `is_soft_kw(b"glob")`
                // block above).
                Tok::TildeTildeTilde => {
                    if BP_CMP < min_bp {
                        break;
                    }
                    self.bump()?;
                    let pattern = self.expr_bp(BP_CONCAT + 1)?;
                    lhs = self.simple_call("glob", vec![lhs, pattern]);
                    continue;
                }
                // `!` (postfix factorial, sugar for `factorial(x)`) at
                // `BP_BANG` — see that constant's doc in `sql::parser` for
                // why it lives here (a dedicated strength between every
                // binary operator and the prefix operators) rather than in
                // `primary`'s postfix loop alongside `::`/`[...]`.
                //
                // One trap this creates: `primary`'s loop is what normally
                // picks up a `::` immediately following a postfix
                // expression, but `!` no longer goes through that loop, so
                // a `::` right after `!` would otherwise be left dangling.
                // Explicitly re-running `cast_postfix` on the folded
                // `factorial(...)` node closes that gap (verified:
                // `duckdb -c "select 4!::varchar"` -> `'24'`, i.e.
                // `CAST(4! AS VARCHAR)`).
                Tok::Bang => {
                    if BP_BANG < min_bp {
                        break;
                    }
                    self.bump()?;
                    let call = self.simple_call("factorial", vec![lhs]);
                    lhs = self.cast_postfix(call)?;
                    continue;
                }
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
                    lhs = self.simple_call(name, vec![lhs, rhs]);
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
                // このリテラルは `primary_atom` を経由しないので、後置 `::`
                // をここでも自分で畳み込む（`primary` の doc 参照。
                // `-1::VARCHAR` が `(-1)::VARCHAR` になることを
                // `duckdb -c "select -1::varchar"` で確認済み）。
                if let Tok::Int(text) = self.cur {
                    let v = int_literal(text, true, self.pos)?;
                    self.bump()?;
                    let node = self.arena.push(Expr::Literal(v));
                    return self.cast_postfix(node);
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
            // 前置の `~`（ビット単位 NOT）。中置の `~`/`!~`（正規表現一致）は
            // `expr_body` の中置ループ側が処理する（同じトークンだが位置で
            // 意味が決まる。`-` の前置/中置と同じパターン）。
            Tok::Tilde => {
                self.bump()?;
                let arg = self.expr_bp(BP_UNARY)?;
                Ok(self.simple_call("bit_not", vec![arg]))
            }
            // `@` (prefix only, no infix meaning): absolute value, sugar
            // for `abs(x)` (verified: `duckdb -c "select @(-5), @(-5.5)"`
            // -> `5`, `5.5`).
            Tok::At => {
                self.bump()?;
                let arg = self.expr_bp(BP_UNARY)?;
                Ok(self.simple_call("abs", vec![arg]))
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
        if self.is_soft_kw(b"similar") && self.peek_is_to()? {
            return self.similar_to(arg, negated);
        }
        let node = match self.cur {
            Tok::Kw(Kw::Is) => {
                ensure!(!negated, UnexpectedToken, self.pos);
                self.bump()?;
                let neg = self.eat_kw(Kw::Not)?;
                // `DISTINCT`/`FROM` はどちらも既存の予約語（`Kw::Distinct`/
                // `Kw::From`）なので、`similar`/`glob` のような文脈依存判定は
                // 要らない。ただし `IS DISTINCT` だけで終わる文はここには
                // 存在しない（`FROM` が必ず続く）ので、2 トークン先読みで
                // 確定させてから消費する。
                if self.is(Tok::Kw(Kw::Distinct)) && self.peek()? == Tok::Kw(Kw::From) {
                    self.bump()?; // distinct
                    self.bump()?; // from
                    let rhs = self.expr_bp(BP_CMP + 1)?;
                    return Ok(self.distinct_from(arg, rhs, neg));
                }
                // `IS [NOT] TRUE`/`IS [NOT] FALSE`. `Kw::True`/`Kw::False`
                // are already reserved keywords (used for the `TRUE`/
                // `FALSE` literals), so no soft-keyword lookahead is needed
                // here.
                if let Tok::Kw(want @ (Kw::True | Kw::False)) = self.cur {
                    self.bump()?;
                    return Ok(self.is_true_or_false(arg, want == Kw::True, neg));
                }
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
        let call = self.simple_call("regexp_full_match", vec![arg, pattern]);
        if negated {
            Ok(self.arena.push(Expr::Unary { op: UnaryOp::Not, arg: call }))
        } else {
            Ok(call)
        }
    }

    /// `[NOT] DISTINCT FROM`。NULL 同士は等しいとみなす等価比較（`=` と違い
    /// 3 値論理の `UNKNOWN` を経由しない、常に `TRUE`/`FALSE` の等価判定）。
    /// 専用のカーネル・`Expr` バリアントは増やさず、既存の `IS NULL`/`AND`/
    /// `OR`/`=` の組み合わせへ展開する:
    ///   `a IS NOT DISTINCT FROM b` ≡
    ///     `(a IS NULL AND b IS NULL) OR (a IS NOT NULL AND b IS NOT NULL AND a = b)`
    ///   `a IS DISTINCT FROM b` ≡ `NOT (上式)`
    /// （`(a IS NULL AND b IS NULL) OR a = b` のような一見同等の短い式は、
    /// 片方だけ NULL のとき `a = b` が `NULL` になり `OR` の結果も `NULL` に
    /// 引きずられてしまう ―― ここでは常に `TRUE`/`FALSE` だけを返す必要が
    /// あるので、両辺とも非 NULL であることを明示的に確認してから `=` へ
    /// 委ねる形にしてある）。
    fn distinct_from(&mut self, l: ExprId, r: ExprId, same: bool) -> ExprId {
        let l_null = self.arena.push(Expr::IsNull { arg: l, negated: false });
        let r_null = self.arena.push(Expr::IsNull { arg: r, negated: false });
        let both_null =
            self.arena.push(Expr::Binary { op: BinaryOp::And, lhs: l_null, rhs: r_null });
        let l_nn = self.arena.push(Expr::IsNull { arg: l, negated: true });
        let r_nn = self.arena.push(Expr::IsNull { arg: r, negated: true });
        let nn = self.arena.push(Expr::Binary { op: BinaryOp::And, lhs: l_nn, rhs: r_nn });
        let eq = self.arena.push(Expr::Binary { op: BinaryOp::Eq, lhs: l, rhs: r });
        let both_nn_and_eq = self.arena.push(Expr::Binary { op: BinaryOp::And, lhs: nn, rhs: eq });
        let same_value =
            self.arena.push(Expr::Binary { op: BinaryOp::Or, lhs: both_null, rhs: both_nn_and_eq });
        if same {
            same_value
        } else {
            self.arena.push(Expr::Unary { op: UnaryOp::Not, arg: same_value })
        }
    }

    /// `IS [NOT] TRUE`/`IS [NOT] FALSE`. Modeled on `distinct_from` above:
    /// no new `Expr`/`BinaryOp` variant, just existing nodes/functions
    /// (`Cast`, `coalesce`, `Unary::Not`) composed to match duckdb's
    /// verified semantics — a non-boolean operand is coerced rather than
    /// rejected, and `NULL` never propagates (`IS TRUE`/`IS FALSE` always
    /// return `TRUE`/`FALSE`, never `NULL`, on any input including `NULL`
    /// itself):
    ///   `x IS TRUE`  ≡ `coalesce(CAST(x AS BOOLEAN), false)`
    ///   `x IS FALSE` ≡ `coalesce(NOT CAST(x AS BOOLEAN), false)`
    /// The `CAST` is load-bearing, not decorative: this engine's `NOT`
    /// requires an already-boolean operand (`SELECT NOT 1` is a type
    /// error), while `CAST(3 AS BOOLEAN)` succeeds and `CAST(NULL AS
    /// BOOLEAN)` yields `NULL` (which `coalesce` then turns into `false`).
    /// The `IS NOT` forms wrap the whole thing in `Unary::Not` at the call
    /// site (`predicate()` above), same as every other `IS NOT ...` form.
    fn is_true_or_false(&mut self, arg: ExprId, want_true: bool, negated: bool) -> ExprId {
        let cast = self.arena.push(Expr::Cast { arg, ty: Ty::Boolean, try_: false });
        let cond = if want_true {
            cast
        } else {
            self.arena.push(Expr::Unary { op: UnaryOp::Not, arg: cast })
        };
        let false_lit = self.arena.push(Expr::Literal(Value::Bool(false)));
        let result = self.simple_call("coalesce", vec![cond, false_lit]);
        if negated {
            self.arena.push(Expr::Unary { op: UnaryOp::Not, arg: result })
        } else {
            result
        }
    }

    /// `primary_atom()` に続けて、任意個の postfix `::type` / `[i]` /
    /// `[i:j]` を、現れた順に畳み込む。いずれも前置演算子より強く結合する
    /// （`duckdb -c "select -1::varchar"` が `-(1::VARCHAR)` と解釈されて
    /// 型エラーになることで確認済み。添字も同様: `duckdb -c "select
    /// -[1,2,3][1]"` は `-(list[1])` になる）。`primary_atom` の分岐の
    /// 大半は結果を `self.arena.push` する前に早期 `return` するので、
    /// この処理はその外側に置く必要がある（内側に置くと `(expr)::ty` や
    /// `col::ty`、`(expr)[i]` などほとんどの実用例を取りこぼす）。
    ///
    /// 2 つの後置演算子は好きな順に交互に書ける（DuckDB と同じ、実測済み）:
    /// `duckdb -c "select [1,2,3][1]::varchar"` は「先に添字、後にキャスト」
    /// （`CAST(list[1] AS VARCHAR)`）、`duckdb -c "select ([1,2,3]::json)[1]"`
    /// は「先にキャスト、後に添字」になる。そのため 1 つのループで両方を
    /// 交互に受け付ける。
    ///
    /// `!` (factorial) deliberately does *not* join this loop, even though
    /// it's a postfix operator too — it lives in `expr_body`'s infix loop
    /// instead, at its own strength `BP_BANG` (see that constant's doc in
    /// `sql::parser`), which sits below the prefix operators but above
    /// every binary operator. That's what makes `-4!`/`-x!` parse as
    /// `(-x)!` without any special-casing here or in `prefix()`.
    fn primary(&mut self) -> Result<ExprId> {
        let mut node = self.primary_atom()?;
        loop {
            if self.eat(Tok::ColonColon)? {
                let ty = self.type_name()?;
                node = self.arena.push(Expr::Cast { arg: node, ty, try_: false });
            } else if self.is(Tok::LBracket) {
                node = self.subscript(node)?;
            } else {
                break;
            }
        }
        Ok(node)
    }

    /// 任意個の後置 `::type` を `node` へ畳み込む。`prefix` の負数リテラル
    /// 即畳み込み経路（`primary_atom`/`primary` を経由しないため別途ここを
    /// 呼ぶ必要がある）専用。`-5[1]` は DuckDB でも構文エラーになる
    /// （`duckdb -c "select -5[1]"` で確認済み）ので、この経路は `::` だけ
    /// 畳み込めばよく、`[i]`/`[i:j]` は付けない（`primary` 側のループとは
    /// 意図的に別実装）。
    ///
    /// `!` (factorial) doesn't belong here either, for the same reason it
    /// doesn't belong in `primary`'s loop: `-4!` folds `-4` to a literal
    /// via this function first, and it's `expr_body`'s outer loop —
    /// operating on the completed `-4` node — that then picks up the `!`
    /// at `BP_BANG` (see that constant's doc). That's also what makes
    /// `-4!` and `-x!` (a non-literal operand) parse identically.
    fn cast_postfix(&mut self, mut node: ExprId) -> Result<ExprId> {
        while self.eat(Tok::ColonColon)? {
            let ty = self.type_name()?;
            node = self.arena.push(Expr::Cast { arg: node, ty, try_: false });
        }
        Ok(node)
    }

    fn primary_atom(&mut self) -> Result<ExprId> {
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
            // `[expr, ...]` 配列リテラル。ここ（`primary_atom`）に来るのは
            // `[` が式の**先頭**にあるときだけ。`expr[i]`/`expr[i:j]` の添字
            // アクセス・スライスは `primary_atom` の結果に**後置**で続く形
            // なので、`primary`/`postfix_ops`（この関数の外）が別途処理する
            // （ファイル冒頭 `sql/lexer.rs` の `Tok::LBracket` のコメント参照）。
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
        Ok(self.simple_call("list_value", args))
    }

    /// `base[i]`（添字アクセス）/ `base[i:j]`（スライス、両端省略可）。
    /// `primary`（この関数の呼び出し元）の後置ループから、`[` を見た時点で
    /// 呼ばれる。`self.cur` はまだ `[` を指している。
    ///
    /// `expr[i]` は `list_extract(expr, i)`（DuckDB と同じく 1 始まり、
    /// 範囲外は NULL、負数は末尾から: `duckdb -c "select [1,2,3][1],
    /// [1,2,3][-1], [1,2,3][10]"` で `1`/`3`/`NULL` を確認済み。この
    /// エンジンの `list_extract` は既にこの規則で実装済み — `crate::json::
    /// list_index` 参照）、`expr[i:j]` は `list_slice(expr, i, j)`
    /// （両端含む。境界の詳しい規則は `list_slice` の doc 参照）へ脱糖する。
    ///
    /// 開始/終了はそれぞれ省略できる（`duckdb -c "select [1,2,3,4,5][:3],
    /// [1,2,3,4,5][2:]"` で `[1,2,3]`/`[2,3,4,5]` を確認済み、つまり
    /// `[:3]` == `[1:3]`、`[2:]` == 末尾まで）。省略した境界はそれぞれ
    /// リテラル `1` / `i64::MAX` に脱糖し、`list_slice` 側でクランプする
    /// （SQL `NULL` には脱糖しない — `duckdb -c "select list_slice([1,2,3],
    /// NULL, 3)"` が `NULL` を返すことから、`NULL` 境界は「省略」ではなく
    /// 「結果も NULL」を意味することが分かる。混同すると `[:3]` が NULL に
    /// なってしまう）。
    fn subscript(&mut self, base: ExprId) -> Result<ExprId> {
        self.bump()?; // '['
                      // `[:j]`: 開始省略。`[:]`（両方省略）にもここで対応する必要がある
                      // ——`]` が直後に来る場合、続けて `self.expr()` を呼ぶと `]` を式の
                      // 先頭として読もうとして構文エラーになる。
        if self.eat(Tok::Colon)? {
            let end = if self.is(Tok::RBracket) {
                self.arena.push(Expr::Literal(Value::I64(i64::MAX)))
            } else {
                self.expr()?
            };
            self.expect(Tok::RBracket)?;
            let start = self.arena.push(Expr::Literal(Value::I64(1)));
            return Ok(self.simple_call("list_slice", vec![base, start, end]));
        }
        let first = self.expr()?;
        if self.eat(Tok::Colon)? {
            // `[i:]`: 終了省略。`[i:j]`: 両方指定。
            let end = if self.is(Tok::RBracket) {
                self.arena.push(Expr::Literal(Value::I64(i64::MAX)))
            } else {
                self.expr()?
            };
            self.expect(Tok::RBracket)?;
            return Ok(self.simple_call("list_slice", vec![base, first, end]));
        }
        // `[i]`: 添字アクセス。
        self.expect(Tok::RBracket)?;
        Ok(self.simple_call("list_extract", vec![base, first]))
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
                let (exclude, replace, rename) = self.star_modifiers()?;
                return Ok(self.arena.push(Expr::Star {
                    qualifier: Some(name),
                    exclude,
                    replace,
                    rename,
                }));
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
                    return Ok(self.simple_call("date_part", vec![part_lit, ts]));
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
    pub(super) fn window_def_body(&mut self) -> Result<WindowDef> {
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

    pub(super) fn type_name(&mut self) -> Result<Ty> {
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
        // SQL 標準の綴り `TIMESTAMP WITH TIME ZONE`。単語 `timestamptz`
        // （`TYPES` 表）が普段使いの短縮形、こちらは標準準拠のための別綴り。
        // `WITH` はここでは常に CTE ではなくこの構文の意味しかありえない
        // 位置（型名の直後）なので、先読み無しでそのまま食ってよい。
        if ty == Ty::Timestamp && self.eat_kw(Kw::With)? {
            ensure!(self.is_soft_kw(b"time"), InvalidCast, pos);
            self.bump()?;
            ensure!(self.is_soft_kw(b"zone"), InvalidCast, pos);
            self.bump()?;
            return Ok(Ty::Timestamptz);
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
