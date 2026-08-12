//! Expression parsing: Pratt precedence climbing, prefix/primary/postfix,
//! CAST, CASE, window function calls, lambdas, and literal parsing helpers.
use super::types::{
    comparison_binop, float_literal, int_literal, is_lambda_func, lookup_interval_unit,
    lookup_type, parse_interval_text, parse_signed_int, temporal_literal_ty, unit_to_interval,
    unquote,
};
use super::*;
use crate::expr::funcs;

impl<'a> Parser<'a> {
    // --- Expressions --------------------------------------------------------

    pub(super) fn expr(&mut self) -> Result<ExprId> {
        self.expr_bp(0)
    }

    /// Consumes one level of depth before the body. Kept a thin wrapper so it is always restored even on error paths.
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
            // `x <op> ANY|ALL|SOME (SELECT ...)`. A quantified comparison that appears
            // only right after a comparison operator, treated as an infix operator of the
            // same binding power as comparison. `ALL` is an existing reserved word (`UNION
            // ALL` and so on), but `ANY`/`SOME` are kept out of the reserved-word table
            // like `glob`/`similar` (so they remain usable as column names), so this looks
            // ahead two tokens and settles only after seeing the `(` (so an ordinary
            // column reference like `x > any_col` is not misread; see the `peek_quantifier` docs).
            if let Some(op) = comparison_binop(self.cur) {
                if BP_CMP >= min_bp {
                    if let Some(all) = self.peek_quantifier()? {
                        self.bump()?; // the comparison operator
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
            // `GLOB`/`SIMILAR TO` are kept out of the reserved-word table for the same
            // reason as ROWS/RANGE/QUALIFY (see the comment at the top of `sql/lexer.rs`)
            // and are matched by spelling only in this infix-operator position. That way
            // columns named `glob`/`similar` remain usable unquoted.
            //
            // DuckDB does not allow `NOT` before `GLOB` (you must write `NOT (x GLOB y)`;
            // confirmed that `duckdb -c "select 'a' NOT GLOB 'b'"` is a syntax error), so
            // unlike `LIKE` this does not route through `predicate()`. `x NOT GLOB y`
            // enters `predicate()` from the `Tok::Kw(Kw::Not)` branch, which has no arm for
            // `glob`, so it naturally becomes `UnexpectedToken`.
            if self.is_soft_kw(b"glob") {
                if BP_CMP < min_bp {
                    break;
                }
                self.bump()?; // glob
                let pattern = self.expr_bp(BP_CONCAT)?;
                lhs = self.simple_call("glob", vec![lhs, pattern]);
                continue;
            }
            // `SIMILAR TO` can take a leading `[NOT]` like `LIKE`, so the same check is
            // also added on the `predicate()` side (the `Tok::Kw(Kw::Not)` branch).
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
                // Like `->`/`->>`, `&`/`|`/`<<`/`>>`/`^`/`**` add no new `BinaryOp` and
                // are expanded as sugar for existing scalar function calls
                // (`bit_and`/`bit_or`/`bit_shift_left`/`bit_shift_right`/`pow` either
                // already exist in `expr::funcs` or were added in this commit -- applying
                // DESIGN.md §11's policy of not adding kernels).
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
                // Left-associative (confirmed `duckdb`'s `2^3^2` = `(2^3)^2`; see the docs
                // on the BP constants), so the right operand is read with `expr_bp(bp + 1)`
                // like any ordinary operator.
                Tok::Pow => {
                    if BP_POW < min_bp {
                        break;
                    }
                    self.bump()?;
                    let rhs = self.expr_bp(BP_POW + 1)?;
                    lhs = self.simple_call("pow", vec![lhs, rhs]);
                    continue;
                }
                // Infix `~`/`!~`. Prefix `~` (bitwise NOT) is handled separately by
                // `prefix()`, so anything reaching here is necessarily infix (regex match).
                // It expands to the same function as `SIMILAR TO` (see the `similar_to` docs).
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
                // `^@`: PostgreSQL/DuckDB's prefix ("starts with") operator.
                // Desugars to the already-existing `starts_with(lhs, rhs)`
                // scalar function, so nothing downstream learns a new
                // concept (same treatment as `~~~` -> `glob(...)` below).
                //
                // Strength and operand handling are verified against the
                // `duckdb` CLI, and are deliberately asymmetric in exactly
                // the same way as the `~~` family:
                //   duckdb: select 'a' || 'b' ^@ 'a'  -> true    i.e. ('a'||'b') ^@ 'a'
                //   duckdb: select 'ab' ^@ 'a' || 'b' -> 'trueb' i.e. ('ab' ^@ 'a') || 'b'
                //   duckdb: select 'ab' ^@ 'a' = true -> true    i.e. ('ab' ^@ 'a') = true
                // So the operator itself sits at `BP_CMP` (the left side
                // therefore already absorbed any `||`/arithmetic), while the
                // right operand is read at `BP_CONCAT + 1` so a following
                // `||` binds to the *result*, not into the pattern.
                // NULL on either side yields NULL (`duckdb -c "select NULL
                // ^@ 'a', 'a' ^@ NULL"` -> both NULL), which is what
                // `starts_with` already does.
                Tok::CaretAt => {
                    if BP_CMP < min_bp {
                        break;
                    }
                    self.bump()?;
                    let prefix = self.expr_bp(BP_CONCAT + 1)?;
                    lhs = self.simple_call("starts_with", vec![lhs, prefix]);
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
                // `->`/`->>` add no new BinaryOp and are expanded as sugar for
                // `json_extract`/`json_extract_string` calls (which ride the existing type
                // resolution and execution in `expr::funcs`).
                // Their precedence follows Postgres's "other operators" band and matches
                // `||` (binding tighter than comparison, so `doc->'a' = 1` can be written
                // without parentheses).
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
                // Predicates (IS NULL / IN / BETWEEN / LIKE / ILIKE) are postfix at the same binding power as comparison.
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
            // All are left-associative, so the right operand is read one level tighter.
            let rhs = self.expr_bp(bp + 1)?;
            lhs = self.arena.push(Expr::Binary { op, lhs, rhs });
        }
        Ok(lhs)
    }

    fn prefix(&mut self) -> Result<ExprId> {
        match self.cur {
            Tok::Minus => {
                self.bump()?;
                // A negative integer folds into a single literal. Otherwise a value that
                // does not fit on the positive side, such as -9223372036854775808, could
                // not be written. This literal does not go through `primary_atom`, so the
                // postfix `::` is folded here as well (see the `primary` docs; confirmed
                // with `duckdb -c "select -1::varchar"` that `-1::VARCHAR` means
                // `(-1)::VARCHAR`).
                if let Tok::Int(text) = self.cur {
                    let v = int_literal(text, true, self.pos)?;
                    self.bump()?;
                    let node = self.arena.push(Expr::Literal(v));
                    return self.cast_postfix(node);
                }
                let arg = self.expr_bp(BP_UNARY)?;
                Ok(self.arena.push(Expr::Unary { op: UnaryOp::Neg, arg }))
            }
            // Unary + is the identity. No node is created.
            Tok::Plus => {
                self.bump()?;
                self.expr_bp(BP_UNARY)
            }
            Tok::Kw(Kw::Not) => {
                self.bump()?;
                // `NOT EXISTS` is folded into negated rather than wrapped in `Unary::Not`.
                if self.is(Tok::Kw(Kw::Exists)) {
                    return self.exists(true);
                }
                let arg = self.expr_bp(BP_NOT)?;
                Ok(self.arena.push(Expr::Unary { op: UnaryOp::Not, arg }))
            }
            // Prefix `~` (bitwise NOT). Infix `~`/`!~` (regex match) are handled by the
            // infix loop in `expr_body` (the same token, with meaning fixed by position --
            // the same pattern as prefix versus infix `-`).
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

    /// `IS [NOT] NULL` / `[NOT] IN` / `[NOT] BETWEEN` / `[NOT] LIKE`.
    /// Negation is folded into each node's `negated` rather than wrapped in `Unary::Not`.
    fn predicate(&mut self, arg: ExprId) -> Result<ExprId> {
        let negated = self.eat_kw(Kw::Not)?;
        // `SIMILAR` is not a reserved word (see the comment at the top of `expr_body`), so
        // it cannot ride the ordinary `match self.cur`. Only this case is checked first.
        if self.is_soft_kw(b"similar") && self.peek_is_to()? {
            return self.similar_to(arg, negated);
        }
        let node = match self.cur {
            Tok::Kw(Kw::Is) => {
                ensure!(!negated, UnexpectedToken, self.pos);
                self.bump()?;
                let neg = self.eat_kw(Kw::Not)?;
                // `DISTINCT`/`FROM` are both existing reserved words
                // (`Kw::Distinct`/`Kw::From`), so no context-dependent check like
                // `similar`/`glob` is needed. But no statement ends at just `IS DISTINCT`
                // (`FROM` always follows), so this settles it with two tokens of lookahead
                // before consuming.
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
                // `IS [NOT] UNKNOWN`. Exactly `IS [NOT] NULL`, so it folds
                // into the same `Expr::IsNull` node — no new AST variant and
                // nothing for the binder or the kernels to learn.
                //
                // The SQL standard only defines `IS UNKNOWN` on a boolean,
                // but duckdb accepts any operand and answers the plain
                // null test, without coercing first (`duckdb -c "select
                // NULL is unknown, 1 is unknown, 'x' is unknown, (1=1) is
                // unknown, NULL is not unknown"` -> true, false, false,
                // false, false). That is `IS NULL` on the nose, so no
                // `CAST(... AS BOOLEAN)` is inserted here (unlike
                // `is_true_or_false` above, where the cast is load-bearing).
                //
                // `UNKNOWN` stays a soft keyword rather than joining
                // `KEYWORDS`: it is a perfectly ordinary word for a data
                // column (`SELECT unknown FROM t` works in duckdb too), and
                // this syntactic position — right after `IS`/`IS NOT` — can
                // never hold a column reference, so spelling-matching here
                // is unambiguous.
                if self.is_soft_kw(b"unknown") {
                    self.bump()?;
                    return Ok(self.arena.push(Expr::IsNull { arg, negated: neg }));
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
                // `IN (SELECT ...)` is a subquery; anything else is a value list.
                // `IN ((SELECT 1))` is a value list (whose element is a scalar subquery).
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
                // The bounds are read one level tighter than AND, so the separating AND is not swallowed.
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
                    // Only a single-byte escape character is accepted.
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

    /// `[NOT] SIMILAR TO pattern`. DuckDB simply treats `SIMILAR TO` as sugar for
    /// `regexp_full_match` (confirmed with `duckdb -c "explain select 'a' similar to
    /// 'a'"`). Unlike the SQL standard's `SIMILAR TO`, `_`/`%` are not special and it is
    /// passed through as a plain POSIX-style regular expression.
    /// This engine has no function named `regexp_full_match`, but one is implemented in
    /// `expr::funcs` specifically for this sugar (no new regex engine is written; the
    /// existing `expr::regex` is reused with anchoring -- see that module's docs).
    ///
    /// DuckDB itself rejects the `ESCAPE` clause as unimplemented (confirmed that
    /// `duckdb -c "select 'a' similar to 'a' escape '\\'"` gives
    /// `Not implemented Error: Custom escape in SIMILAR TO`), so it is explicitly rejected
    /// here as well.
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

    /// `[NOT] DISTINCT FROM`. An equality comparison that treats two NULLs as equal
    /// (unlike `=`, it never goes through three-valued `UNKNOWN`; always TRUE/FALSE).
    /// Rather than adding a dedicated kernel or `Expr` variant, it expands into a
    /// combination of the existing `IS NULL`/`AND`/`OR`/`=`:
    ///   `a IS NOT DISTINCT FROM b` ≡
    ///     `(a IS NULL AND b IS NULL) OR (a IS NOT NULL AND b IS NOT NULL AND a = b)`
    ///   `a IS DISTINCT FROM b` == `NOT (the expression above)`
    /// (A seemingly equivalent shorter form such as `(a IS NULL AND b IS NULL) OR a = b`
    /// fails when only one side is NULL: `a = b` becomes `NULL` and drags the `OR` result
    /// to `NULL` too. Since this must always return only TRUE/FALSE, it explicitly
    /// confirms both sides are non-NULL before delegating to `=`.)
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

    /// Following `primary_atom()`, folds any number of postfix `::type` / `[i]` / `[i:j]`
    /// in the order they appear. All bind tighter than prefix operators (confirmed by
    /// `duckdb -c "select -1::varchar"` being interpreted as `-(1::VARCHAR)` and thus a
    /// type error; subscripting behaves the same: `duckdb -c "select -[1,2,3][1]"` means
    /// `-(list[1])`). Most branches of `primary_atom` return early before pushing the
    /// result via `self.arena.push`, so this handling must sit outside it (inside, it
    /// would miss almost every practical case, such as `(expr)::ty`, `col::ty`, and `(expr)[i]`).
    ///
    /// The two postfix operators can be written in any alternating order (the same as
    /// DuckDB, measured): `duckdb -c "select [1,2,3][1]::varchar"` is "subscript first,
    /// then cast" (`CAST(list[1] AS VARCHAR)`), while `duckdb -c "select
    /// ([1,2,3]::json)[1]"` is "cast first, then subscript". Hence a single loop accepts
    /// both, alternating.
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

    /// Folds any number of postfix `::type` into `node`. Specifically for `prefix`'s
    /// immediate negative-literal folding path (which bypasses `primary_atom`/`primary`
    /// and so must call this separately). `-5[1]` is a syntax error in DuckDB too
    /// (confirmed with `duckdb -c "select -5[1]"`), so this path only needs to fold `::`
    /// and does not attach `[i]`/`[i:j]` (deliberately implemented separately from the
    /// loop on the `primary` side).
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
                // The number is by order of appearance. It indexes the host's bind array.
                ensure!(self.num_params < u16::MAX, UnsupportedFeature, pos);
                let n = self.num_params;
                self.num_params += 1;
                self.bump()?;
                Expr::Param(n)
            }
            Tok::LParen => {
                self.bump()?;
                // `(SELECT ...)` is a scalar subquery; anything else is a parenthesized expression.
                if self.starts_query() {
                    let query = self.query_stmt()?;
                    self.expect(Tok::RParen)?;
                    return Ok(self.arena.push(Expr::ScalarSubquery(Box::new(query))));
                }
                let e = self.expr()?;
                self.expect(Tok::RParen)?;
                return Ok(e);
            }
            // An `[expr, ...]` array literal. This point (`primary_atom`) is reached only
            // when `[` is at the **head** of an expression. Subscripting and slicing,
            // `expr[i]`/`expr[i:j]`, follow the result of `primary_atom` as a **postfix**
            // and are handled separately by `primary`/`postfix_ops` (outside this
            // function; see the `Tok::LBracket` comment in `sql/lexer.rs`).
            Tok::LBracket => return self.array_literal(),
            Tok::Kw(Kw::Exists) => return self.exists(false),
            Tok::Kw(Kw::Cast) => return self.cast(),
            Tok::Kw(Kw::Case) => return self.case(),
            // `INTERVAL` is not a reserved word (a column named `interval` should be
            // writable), so the spelling is checked here and the contents looked ahead.
            Tok::Ident(s) if eq_ascii_ci(s.as_bytes(), b"interval") => {
                return self.interval_literal_or_ident();
            }
            // `DATE '...'` / `TIME '...'` / `TIMESTAMP '...'` /
            // `TIMESTAMPTZ '...'`. Like `INTERVAL` it is not reserved, so it is read as a
            // typed literal only when the spelling matches and the next token is a string
            // literal (`temporal_literal_or_ident`). Quoted identifiers (`"date"`) are
            // excluded and always become column references.
            Tok::Ident(s) => {
                if let Some(ty) = temporal_literal_ty(s.as_bytes()) {
                    return self.temporal_literal_or_ident(ty);
                }
                return self.name_ref();
            }
            Tok::QIdent(_) => return self.name_ref(),
            _ => err!(UnexpectedToken, pos),
        };
        Ok(self.arena.push(node))
    }

    /// `[expr, expr, ...]`. Sugar that desugars directly to `list_value(expr, expr, ...)`
    /// (an alias of `json_array`), with exactly the same meaning.
    ///
    /// Only the empty array `[]` bypasses `list_value()`: `resolve` for
    /// `list_value`/`json_array` is designed to reject zero arguments with `WrongArgCount`
    /// (with no arguments, `call()` cannot determine the row count), so an empty JSON
    /// array is embedded directly as a `TypedLiteral` instead. Confirmed with `duckdb -c
    /// "select [], typeof([])"` that `[]` is itself a valid expression.
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

    /// `base[i]` (subscripting) / `base[i:j]` (slicing, either bound optional).
    /// Called from the postfix loop of `primary` (this function's caller) upon seeing `[`.
    /// `self.cur` still points at `[`.
    ///
    /// `expr[i]` desugars to `list_extract(expr, i)` (1-based like DuckDB, NULL when out
    /// of range, negatives counting from the end: confirmed `1`/`3`/`NULL` with
    /// `duckdb -c "select [1,2,3][1], [1,2,3][-1], [1,2,3][10]"`. This engine's
    /// `list_extract` already implements that rule -- see `crate::json::list_index`), and
    /// `expr[i:j]` desugars to `list_slice(expr, i, j)` (inclusive on both ends; see the
    /// `list_slice` docs for the detailed boundary rules).
    ///
    /// Start and end may each be omitted (confirmed `[1,2,3]`/`[2,3,4,5]` with
    /// `duckdb -c "select [1,2,3,4,5][:3], [1,2,3,4,5][2:]"`, i.e. `[:3]` == `[1:3]` and
    /// `[2:]` runs to the end). An omitted bound desugars to the literal `1` /
    /// `i64::MAX` respectively and is clamped by `list_slice` (it does not desugar to SQL
    /// `NULL` -- `duckdb -c "select list_slice([1,2,3], NULL, 3)"` returning `NULL` shows
    /// that a `NULL` bound means "the result is NULL too", not "omitted". Conflating them
    /// would make `[:3]` NULL).
    fn subscript(&mut self, base: ExprId) -> Result<ExprId> {
        self.bump()?; // '['
                      // `[:j]`: start omitted. `[:]` (both omitted) must be handled here
                      // too -- if `]` comes next, calling `self.expr()` would try to read
                      // `]` as the head of an expression and give a syntax error.
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
            // `[i:]`: end omitted. `[i:j]`: both given.
            let end = if self.is(Tok::RBracket) {
                self.arena.push(Expr::Literal(Value::I64(i64::MAX)))
            } else {
                self.expr()?
            };
            self.expect(Tok::RBracket)?;
            return Ok(self.simple_call("list_slice", vec![base, first, end]));
        }
        // `[i]`: subscripting.
        self.expect(Tok::RBracket)?;
        Ok(self.simple_call("list_extract", vec![base, first]))
    }

    /// `INTERVAL '<n> <unit> ...'` / `INTERVAL '<n>' <unit>` /
    /// An `INTERVAL <n> <unit>` literal. If what follows `INTERVAL` is not one of these
    /// shapes, it is delegated to `name_ref` as a column reference (a column named `interval`).
    ///
    /// The same style as `EXTRACT` in `call()`: clone the `Lexer` to settle the shape,
    /// then advance the real token stream the same number of times.
    fn interval_literal_or_ident(&mut self) -> Result<ExprId> {
        let mut lx = self.lex.clone();
        let t1 = lx.next_token()?.tok;
        match t1 {
            Tok::Str(raw) => {
                let text = unquote(raw, b'\'');
                // If the whole string is one signed integer followed by a unit word, this
                // is the `INTERVAL '3' DAY` form.
                let t2 = lx.next_token()?.tok;
                if let (Some(n), Tok::Ident(u)) = (parse_signed_int(&text), t2) {
                    if let Some(unit) = lookup_interval_unit(u.as_bytes()) {
                        let pos = self.pos;
                        self.bump()?; // INTERVAL
                        self.bump()?; // the string
                        self.bump()?; // the unit word
                        let packed = unit_to_interval(unit, n, pos)?;
                        return Ok(self.arena.push(Expr::IntervalLiteral(packed)));
                    }
                }
                let pos = self.pos;
                self.bump()?; // INTERVAL
                self.bump()?; // the string
                let packed = parse_interval_text(&text, pos)?;
                Ok(self.arena.push(Expr::IntervalLiteral(packed)))
            }
            Tok::Int(text) => {
                // An unquoted `INTERVAL <n> <unit>`. Without a following unit word it is a
                // column reference (an expression where a column named `interval` is
                // followed by an integer is syntactically impossible anyway, but this errs on the safe side).
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
                self.bump()?; // the number
                self.bump()?; // the unit word
                let packed = unit_to_interval(unit, n, pos)?;
                Ok(self.arena.push(Expr::IntervalLiteral(packed)))
            }
            _ => self.name_ref(),
        }
    }

    /// `DATE '2020-01-01'` / `TIME '10:00:00'` / `TIMESTAMP '2020-01-01
    /// 10:00:00'` / `TIMESTAMPTZ '2020-01-01 00:00:00+09'`.
    ///
    /// Modeled directly on `interval_literal_or_ident` above, including its
    /// escape hatch: if the token after the type name is not a string
    /// literal, this is not a typed literal at all and the word goes back to
    /// being an ordinary column reference (`SELECT date FROM t`, `SELECT
    /// date + 1`, `ORDER BY time`). That fallback is the whole reason
    /// `DATE`/`TIME`/... are not reserved words — column names come from
    /// data files and users cannot rename them (`sql::lexer` module doc).
    ///
    /// The text is converted **here**, at parse time, into a folded
    /// `Expr::TypedLiteral` rather than being left as `CAST('...' AS DATE)`
    /// (which is literally what duckdb's EXPLAIN shows for this syntax).
    /// A constant matters downstream: RowGroup/page/Bloom pruning
    /// (`plan::bind::pruning`) extracts literal bounds out of `WHERE`, and a
    /// `Cast` node wrapping a string would not be recognised as one.
    ///
    /// A value the parser cannot read is a hard error at parse time,
    /// matching duckdb (`duckdb -c "select DATE 'nonsense'"` ->
    /// `Conversion Error: invalid date field format`). This is deliberately
    /// *not* the `CAST`-of-a-bad-string rule this engine uses elsewhere
    /// ("that row becomes NULL", see docs/sql/types.md): a literal is a
    /// fixed piece of query text, so silently turning the whole query's
    /// constant into NULL would hide a typo instead of reporting it.
    fn temporal_literal_or_ident(&mut self, ty: Ty) -> Result<ExprId> {
        let mut lx = self.lex.clone();
        let Tok::Str(raw) = lx.next_token()?.tok else {
            return self.name_ref();
        };
        let pos = self.pos;
        self.bump()?; // the type name
        self.bump()?; // the string
        let text = unquote(raw, b'\'');
        let b = text.as_bytes();
        // The physical representation is fixed per logical type (DESIGN.md §8): DATE is
        // I32 days, TIME/TIMESTAMP/TIMESTAMPTZ are I64 microseconds. Exactly the same
        // shape as the `TypedLiteral` `sql::now` builds for
        // `CURRENT_DATE`/`CURRENT_TIMESTAMP`.
        let value = match ty {
            Ty::Date => funcs::parse_date(b).and_then(|d| i32::try_from(d).ok()).map(Value::I32),
            Ty::Time => funcs::parse_time(b).map(Value::I64),
            Ty::Timestamptz => funcs::parse_timestamptz(b).map(Value::I64),
            _ => funcs::parse_timestamp(b).map(Value::I64),
        };
        let Some(value) = value else { err!(InvalidCast, pos) };
        Ok(self.arena.push(Expr::TypedLiteral(value, ty)))
    }

    /// A primary beginning with an identifier: function call / `q.name` / `q.*` / column reference.
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
                    columns: None,
                    exclude,
                    replace,
                    rename,
                }));
            }
            let col = self.ident()?;
            // `t.COLUMNS(*)` is not a thing: DuckDB rejects a qualified
            // `COLUMNS` too ("Scalar Function with name columns does not
            // exist"). Reported as `UnsupportedFeature` rather than as a
            // stray-token error on the `(`, which would read as a typo.
            ensure!(
                !(self.is(Tok::LParen) && eq_ascii_ci(col.as_bytes(), b"columns")),
                UnsupportedFeature,
                self.pos
            );
            return Ok(self.arena.push(Expr::ColumnRef { qualifier: Some(name), name: col }));
        }
        Ok(self.arena.push(Expr::ColumnRef { qualifier: None, name }))
    }

    /// `[NOT] EXISTS ( query )`. On entry `cur` is `EXISTS`.
    fn exists(&mut self, negated: bool) -> Result<ExprId> {
        self.bump()?; // EXISTS
        self.expect(Tok::LParen)?;
        let query = self.query_stmt()?;
        self.expect(Tok::RParen)?;
        Ok(self.arena.push(Expr::Exists { query: Box::new(query), negated }))
    }

    fn call(&mut self, name: String) -> Result<ExprId> {
        // `EXTRACT(part FROM ts)` is a special syntax interposing `FROM`, shaped
        // differently from an ordinary comma-separated argument list. It is equivalent to
        // `date_part(part, ts)`, so only the syntax is absorbed here and folded into the
        // same `Function` node (execution need not know about `extract`).
        //
        // `Parser` itself cannot be cloned (it holds an `ExprArena`), so only the `Lexer`
        // is cloned to settle "is this the `( IDENT FROM` shape" before advancing `self`.
        // The same lookahead style as `peek()`.
        // `TRY_CAST(expr AS type)` is the same special syntax as CAST (interposing `AS`).
        // Its shape differs from a comma-separated list, so it takes the same path as CAST.
        if eq_ascii_ci(name.as_bytes(), b"try_cast") && self.cur == Tok::LParen {
            return self.cast_body(true);
        }
        // `COLUMNS(...)` is only a star expression at the start of a
        // select-list item (`sql::parser::Parser::columns_item`); reaching it
        // here means it was written somewhere a star cannot expand — most
        // often `min(COLUMNS(*))`, DuckDB's "distribute the enclosing function
        // over the expansion" form. That would need the binder to synthesize
        // one function call per expanded column, which it cannot do: the
        // expression arena is immutable by the time the input schema is known,
        // and aggregates are collected from the raw AST in an earlier pass.
        // `UNPACK(...)` (the other unpacking form) is rejected alongside it.
        // Both are reported as `UnsupportedFeature` rather than left to
        // surface later as "function not found", which would suggest a typo.
        if (eq_ascii_ci(name.as_bytes(), b"columns") || eq_ascii_ci(name.as_bytes(), b"unpack"))
            && self.cur == Tok::LParen
        {
            err!(UnsupportedFeature, self.pos)
        }
        // `UNNEST(expr)` (for the SELECT list). DISTINCT/`*`/FILTER/OVER are meaningless
        // here, so it is read in a simple form distinct from an ordinary function call
        // (for the FROM-clause version see `base_rel`).
        if eq_ascii_ci(name.as_bytes(), b"unnest") && self.cur == Tok::LParen {
            self.bump()?; // '('
            let arg = self.expr()?;
            self.expect(Tok::RParen)?;
            return Ok(self.arena.push(Expr::Unnest(arg)));
        }
        // `IIF(cond, then, else)` is sugar for `CASE WHEN cond THEN then ELSE else END`.
        // It is desugared to a CASE expression at the parser level (the same judgment as
        // `EXTRACT` -> `date_part`), so execution need not know about `iif` at all.
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
        // --- Standard SQL function syntax (keywords interposed among arguments) -------
        //
        // Handled like `EXTRACT(part FROM ts)`: only the syntax is absorbed here and
        // desugared straight into existing scalar function calls. Neither execution nor
        // the kernels need to know about the "standard syntax" of `position`/`substring`/`trim`.
        //
        // Unlike `EXTRACT`, an expression of arbitrary length precedes the keyword, so two
        // tokens of lookahead cannot settle the shape. Instead `call_has_top_level` first
        // checks "is there an `IN`/`FROM` at the same depth" before branching (without
        // one, it is read as a conventional comma-separated argument list).
        if eq_ascii_ci(name.as_bytes(), b"position")
            && self.cur == Tok::LParen
            && self.call_has_top_level(Tok::Kw(Kw::In))?
        {
            return self.position_in_call();
        }
        if eq_ascii_ci(name.as_bytes(), b"trim")
            && self.cur == Tok::LParen
            && self.call_has_top_level(Tok::Kw(Kw::From))?
        {
            return self.trim_from_call();
        }
        if (eq_ascii_ci(name.as_bytes(), b"substring") || eq_ascii_ci(name.as_bytes(), b"substr"))
            && self.cur == Tok::LParen
        {
            return self.substring_call(&name);
        }
        if eq_ascii_ci(name.as_bytes(), b"extract") && self.cur == Tok::LParen {
            let mut lx = self.lex.clone();
            let t1 = lx.next_token()?.tok;
            if let Tok::Ident(part) = t1 {
                let t2 = lx.next_token()?.tok;
                if t2 == Tok::Kw(Kw::From) {
                    // Settled at this point. Advance the real token stream the same number of times.
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
            // A `*` in argument position occurs only in the COUNT(*) family. Interpretation is left to the binder.
            star = true;
            self.bump()?;
        } else if !self.is(Tok::RParen) {
            distinct = self.eat_kw(Kw::Distinct)?;
            // Lambdas (`x -> expr` / `(a, b) -> expr`) are recognized only in the argument
            // positions of `list_transform` / `list_filter` / `list_reduce`.
            // `->` is normally sugar for the JSON path operator (expanding to
            // `json_extract`; see `expr_body`), so in other functions' arguments it keeps
            // that meaning as before (so existing usage like `coalesce(doc -> 'a', 'x')` is
            // not broken. As measured with the duckdb CLI, `->` is interpreted as a lambda
            // only in the argument positions of functions known to take one, such as
            // list_transform; `coalesce(x -> 5, 3)`, for instance, fails to resolve `x` as
            // a column and errors with `x -> 5` still the JSON operator).
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
        // `agg(...) FILTER (WHERE cond)`. `FILTER` is not reserved either, and is treated
        // as a keyword only right after a function call when the next token is `(` (the same judgment as `OVER`).
        let mut filter = None;
        if self.is_soft_kw(b"filter") && self.peek()? == Tok::LParen {
            self.bump()?; // filter
            self.bump()?; // '('
            self.expect_kw(Kw::Where)?;
            filter = Some(self.expr()?);
            self.expect(Tok::RParen)?;
        }
        // `OVER` is not reserved, so it is taken as a window clause only when followed by
        // `(` or an identifier. The `over` in `SELECT count(*) over FROM t` still passes as
        // an alias (with a keyword such as `FROM` next, the branch below is not entered).
        if self.is_soft_kw(b"over") {
            match self.peek()? {
                Tok::LParen => {
                    // `f(DISTINCT x) OVER (...)` / `f(...) FILTER (...) OVER (...)` are out
                    // of scope. Ignoring them would change results, so they are rejected.
                    ensure!(!distinct, UnsupportedFeature, self.pos);
                    ensure!(filter.is_none(), UnsupportedFeature, self.pos);
                    return self.window(name, args, star);
                }
                Tok::Ident(_) | Tok::QIdent(_) => {
                    // `OVER w` (a named window reference). The definition itself lives only
                    // in the `WINDOW` clause, and the SELECT list comes first syntactically,
                    // so only the name is kept here and looked up from
                    // `SelectStmt::windows` at bind time (see `plan::bind`).
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

    /// `position(<search> IN <string>)` -- the standard SQL syntax. On entry `cur` is the
    /// opening parenthesis, and an `IN` at the same depth is already confirmed.
    ///
    /// Note that **the argument order is swapped**: against `strpos(string, search)`, the
    /// standard syntax writes the needle first (confirmed by `duckdb -c "select
    /// position('b' in 'abc'), strpos('abc','b')"` both returning `2`). It desugars
    /// straight to the existing `strpos` (`position`/`instr` share the same implementation).
    ///
    /// Not found gives `0`, searching for the empty string gives `1`, and either side
    /// being NULL gives NULL (`duckdb -c "select position('z' in 'abc'), position('' in
    /// 'abc'), position(NULL in 'abc'), position('b' in NULL)"` -> `0`/`1`/NULL/NULL)
    /// -- that is exactly the existing `strpos` behavior, so nothing is done here.
    ///
    /// The needle expression is read at `BP_CMP + 1`, so the separating `IN` is not
    /// swallowed as the `x IN (...)` predicate (the same trick as reading `BETWEEN`'s
    /// bounds at `BP_CONCAT`). `||` and arithmetic bind tighter than `BP_CMP` and so
    /// combine normally (`position('a' || 'b' in s)` works as intended).
    fn position_in_call(&mut self) -> Result<ExprId> {
        self.bump()?; // '('
        let search = self.expr_bp(BP_CMP + 1)?;
        self.expect_kw(Kw::In)?;
        let string = self.expr()?;
        self.expect(Tok::RParen)?;
        Ok(self.simple_call("strpos", vec![string, search]))
    }

    /// `trim([BOTH | LEADING | TRAILING] [<chars>] FROM <s>)` -- the standard SQL syntax.
    /// On entry `cur` is the opening parenthesis, and a `FROM` at the same depth is
    /// already confirmed. It desugars to the existing `trim`/`ltrim`/`rtrim` (the
    /// one-argument form trims whitespace, the two-argument form a set of characters).
    ///
    /// The accepted forms and results were confirmed with the `duckdb` CLI:
    ///   trim(both 'x' from 'xxabxx')     -> 'ab'      == trim('xxabxx','x')
    ///   trim(leading 'x' from 'xxabxx')  -> 'abxx'    == ltrim('xxabxx','x')
    ///   trim(trailing 'x' from 'xxabxx') -> 'xxab'    == rtrim('xxabxx','x')
    ///   trim('x' from 'xxabxx')          -> 'ab'      (omitting the direction means BOTH)
    ///   trim(from '  ab  ')              -> 'ab'      (omitting the character set means whitespace)
    ///   trim(both from '  ab  ')         -> 'ab'
    ///
    /// `BOTH`/`LEADING`/`TRAILING` are not reserved (the same reason as the column-name
    /// breakage incidents around `ROWS`/`RANGE`). Matching them by spelling alone is safe
    /// here only because the caller already settled that "there is a `FROM` at the same
    /// depth"; an ordinary `trim(leading)` / `trim(leading, 'x')` (passing a column named
    /// `leading`) never enters this function and is read as a function call as before.
    fn trim_from_call(&mut self) -> Result<ExprId> {
        self.bump()?; // '('
        let mut func = "trim";
        if self.is_soft_kw(b"both") {
            self.bump()?;
        } else if self.is_soft_kw(b"leading") {
            func = "ltrim";
            self.bump()?;
        } else if self.is_soft_kw(b"trailing") {
            func = "rtrim";
            self.bump()?;
        }
        // If `FROM` follows the direction word, the character set is omitted (= trim whitespace).
        let chars = if self.eat_kw(Kw::From)? {
            None
        } else {
            let c = self.expr()?;
            self.expect_kw(Kw::From)?;
            Some(c)
        };
        let s = self.expr()?;
        self.expect(Tok::RParen)?;
        let args = match chars {
            Some(c) => vec![s, c],
            None => vec![s],
        };
        Ok(self.simple_call(func, args))
    }

    /// The argument list of `substring`/`substr`. Reads both the standard SQL form
    /// `substring(<s> FROM <start> [FOR <len>])` / `substring(<s> FOR <len>)`
    /// and the conventional comma-separated `substring(<s>, <start>[, <len>])`
    /// in one place. On entry `cur` is the opening parenthesis.
    ///
    /// Confirmed with the `duckdb` CLI:
    ///   substring('abcdef' from 2)        -> 'bcdef'  == substring('abcdef',2)
    ///   substring('abcdef' from 2 for 3)  -> 'bcd'    == substring('abcdef',2,3)
    ///   substring('abcdef' for 3)         -> 'abc'    == substring('abcdef',1,3)
    /// A `FOR`-only form is the same as start position 1, so a literal `1` is supplied and
    /// it drops into the three-argument form.
    ///
    /// Unlike `position`/`trim`, no lookahead (`call_has_top_level`) is needed: the
    /// separator word can only appear **after the first argument**, and neither `FROM` (a
    /// reserved word) nor `FOR` (a bare identifier) is an infix operator, so reading the
    /// first argument with a plain `expr()` always stops just before it. Only then does it
    /// branch, and the path back to the comma form remains open (which matters so that
    /// passing a column named `for`, as in `substring(for, 2)`, is not broken).
    fn substring_call(&mut self, name: &str) -> Result<ExprId> {
        self.bump()?; // '('
        let mut args = vec![self.expr()?];
        if self.eat_kw(Kw::From)? {
            args.push(self.expr()?);
            if self.is_soft_kw(b"for") {
                self.bump()?;
                args.push(self.expr()?);
            }
        } else if self.is_soft_kw(b"for") {
            self.bump()?;
            args.push(self.arena.push(Expr::Literal(Value::I32(1))));
            args.push(self.expr()?);
        } else {
            while self.eat(Tok::Comma)? {
                args.push(self.expr()?);
            }
        }
        self.expect(Tok::RParen)?;
        Ok(self.simple_call(name, args))
    }

    /// Whether the current position has the shape of a lambda parameter list (`IDENT ->`
    /// or `( IDENT [, IDENT]* ) ->`). It looks ahead on a clone of the `Lexer` only and
    /// does not move `self` (the same style as `peek`).
    ///
    /// A different shape merely returns `false` rather than raising a syntax error (the
    /// caller falls back to an ordinary `expr()`).
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

    /// `x -> expr` / `(a, b) -> expr`. Called only right after `looks_like_lambda_params`
    /// returns `true`.
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
        // The body is an ordinary expression. The comma is seen by the enclosing loop as an
        // argument separator, so nothing special is needed here (also because `expr_body`
        // never handled commas in the first place) -- just read with no special lower bound.
        let body = self.expr()?;
        Ok(self.arena.push(Expr::Lambda { params, body }))
    }

    /// `OVER ( [PARTITION BY ...] [ORDER BY ...] )`. On entry `cur` is `OVER`, and the
    /// next token is already confirmed to be `(`.
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

    /// The shared body of `[PARTITION BY ...] [ORDER BY ...] )`. On entry the opening
    /// parenthesis is already consumed, and this reads and consumes through the closing
    /// one. Used both by an inline `OVER (...)` and by a named `WINDOW name AS (...)` definition.
    pub(super) fn window_def_body(&mut self) -> Result<WindowDef> {
        let mut partition_by = Vec::new();
        // `PARTITION` is likewise treated as a keyword only at the head of a window specification.
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
        // Explicit frame specifications are unsupported. Silently using the default frame
        // would change results, so they are rejected. `ROWS` / `RANGE` are not reserved either, so they are matched by spelling here.
        if self.is_soft_kw(b"rows") || self.is_soft_kw(b"range") {
            err!(UnsupportedFeature, self.pos);
        }
        self.expect(Tok::RParen)?;
        // The default frame is decided by the presence of ORDER BY, per the SQL standard.
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

    /// `( expr AS type )`. The shared part of `CAST` and `TRY_CAST`. On entry `cur` is the
    /// opening parenthesis.
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
        // The standard SQL spelling `TIMESTAMP WITH TIME ZONE`. The single word
        // `timestamptz` (the `TYPES` table) is the everyday shorthand; this is the
        // alternative spelling for standard conformance. `WITH` here is in a position (right
        // after a type name) where it can only ever mean this construct and never a CTE, so it can be consumed without lookahead.
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
                      // If WHEN does not immediately follow CASE, the form carries a subject expression to compare against.
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
