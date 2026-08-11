//! SQL パーサ。
//!
//! 文は再帰下降、式は Pratt（優先順位登り）で解く（DESIGN.md §7）。
//! 式は `Box` ではなくアリーナへ積むので、木の破棄で再帰が起きない。
//!
//! 再帰は必ず `MAX_DEPTH` で頭打ちにする。wasm ではスタック枯渇が
//! 回復不能なトラップになるため、深い入力は必ずエラーとして返す。

use crate::prelude::*;
use crate::rt::hash::eq_ascii_ci;
use crate::sql::ast::{
    BinaryOp, Expr, ExprArena, ExprId, FromItem, JoinKind, OrderByItem, Parsed, SelectItem,
    SelectStmt, Stmt, UnaryOp,
};
use crate::sql::lexer::{Kw, Lexer, Tok};
use crate::vector::{Ty, Value};

/// 再帰下降・Pratt の深さ上限。括弧の入れ子やサブクエリに効く。
const MAX_DEPTH: u16 = 64;

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
    let stmt = p.stmt()?;
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
    num_params: u16,
}

impl<'a> Parser<'a> {
    fn new(sql: &'a str) -> Result<Self> {
        let mut lex = Lexer::new(sql);
        let t = lex.next_token()?;
        Ok(Parser { lex, cur: t.tok, pos: t.pos, arena: ExprArena::new(), depth: 0, num_params: 0 })
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
            Tok::Ident(_) | Tok::QIdent(_) => Ok(Some(self.ident()?)),
            _ => Ok(None),
        }
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

    // --- 文 -----------------------------------------------------------------

    fn stmt(&mut self) -> Result<Stmt> {
        let s = match self.cur {
            Tok::Kw(Kw::Select) => Stmt::Select(Box::new(self.select_stmt()?)),
            Tok::Kw(Kw::Explain) => {
                self.bump()?;
                Stmt::Explain(Box::new(self.select_stmt()?))
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
            // v1 は問い合わせ系のみ。INSERT/CREATE などはここで一括で弾く。
            _ => err!(UnsupportedFeature, self.pos),
        };
        self.eat(Tok::Semi)?;
        ensure!(self.is(Tok::Eof), UnexpectedToken, self.pos);
        Ok(s)
    }

    /// 深さを 1 段消費する SELECT。FROM のサブクエリ経由で再帰しうる。
    fn select_stmt(&mut self) -> Result<SelectStmt> {
        ensure!(self.depth < MAX_DEPTH, ExpressionTooDeep, self.pos);
        self.depth += 1;
        let r = self.select_body();
        self.depth -= 1;
        r
    }

    fn select_body(&mut self) -> Result<SelectStmt> {
        self.expect_kw(Kw::Select)?;
        let mut st = SelectStmt::empty();
        st.distinct = self.eat_kw(Kw::Distinct)?;
        loop {
            let item = self.select_item()?;
            st.items.push(item);
            if !self.eat(Tok::Comma)? {
                break;
            }
        }
        if self.eat_kw(Kw::From)? {
            st.from = Some(self.parse_from_item()?);
        }
        if self.eat_kw(Kw::Where)? {
            st.filter = Some(self.expr()?);
        }
        if self.eat_kw(Kw::Group)? {
            self.expect_kw(Kw::By)?;
            loop {
                let e = self.expr()?;
                st.group_by.push(e);
                if !self.eat(Tok::Comma)? {
                    break;
                }
            }
        }
        if self.eat_kw(Kw::Having)? {
            st.having = Some(self.expr()?);
        }
        if self.eat_kw(Kw::Order)? {
            self.expect_kw(Kw::By)?;
            loop {
                let it = self.order_item()?;
                st.order_by.push(it);
                if !self.eat(Tok::Comma)? {
                    break;
                }
            }
        }
        if self.eat_kw(Kw::Limit)? {
            st.limit = Some(self.uint()?);
        }
        if self.eat_kw(Kw::Offset)? {
            st.offset = Some(self.uint()?);
        }
        Ok(st)
    }

    fn select_item(&mut self) -> Result<SelectItem> {
        // 先頭の `*` だけは式ではなく列挙として扱う。`t.*` は primary 側。
        if self.is(Tok::Star) {
            self.bump()?;
            let expr = self.arena.push(Expr::Star { qualifier: None });
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
        // 既定は SQL 標準どおり「NULL は最大値扱い」= ASC なら最後、DESC なら最初。
        let mut nulls_first = desc;
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
        let mut joins: u16 = 0;
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
            // 式と同じ上限で打ち切る。
            joins += 1;
            ensure!(joins <= MAX_DEPTH, ExpressionTooDeep, self.pos);
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
            let query = self.select_stmt()?;
            self.expect(Tok::RParen)?;
            let alias = self.opt_alias()?;
            return Ok(FromItem::Subquery { query: Box::new(query), alias });
        }
        let pos = self.pos;
        let is_parquet = matches!(self.cur, Tok::Ident(s) if eq_ascii_ci(s.as_bytes(), b"parquet"));
        let name = self.ident()?;
        if self.is(Tok::LParen) {
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
                // 述語（IS NULL / IN / BETWEEN / LIKE）は比較と同じ強さの後置。
                Tok::Kw(Kw::Is | Kw::In | Kw::Between | Kw::Like | Kw::Not) => {
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
                // IN (SELECT ...) は v2 送り。
                if self.is(Tok::Kw(Kw::Select)) {
                    err!(UnsupportedFeature, self.pos);
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
            Tok::Kw(Kw::Like) => {
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
                Expr::Like { arg, pattern, negated, escape }
            }
            _ => err!(UnexpectedToken, self.pos),
        };
        Ok(self.arena.push(node))
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
                // スカラサブクエリは v2 送り。
                if self.is(Tok::Kw(Kw::Select)) {
                    err!(UnsupportedFeature, self.pos);
                }
                let e = self.expr()?;
                self.expect(Tok::RParen)?;
                return Ok(e);
            }
            Tok::Kw(Kw::Cast) => return self.cast(),
            Tok::Kw(Kw::Case) => return self.case(),
            Tok::Ident(_) | Tok::QIdent(_) => return self.name_ref(),
            _ => err!(UnexpectedToken, pos),
        };
        Ok(self.arena.push(node))
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
                return Ok(self.arena.push(Expr::Star { qualifier: Some(name) }));
            }
            let col = self.ident()?;
            return Ok(self.arena.push(Expr::ColumnRef { qualifier: Some(name), name: col }));
        }
        Ok(self.arena.push(Expr::ColumnRef { qualifier: None, name }))
    }

    fn call(&mut self, name: String) -> Result<ExprId> {
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
            loop {
                let e = self.expr()?;
                args.push(e);
                if !self.eat(Tok::Comma)? {
                    break;
                }
            }
        }
        self.expect(Tok::RParen)?;
        Ok(self.arena.push(Expr::Function { name, args, distinct, star }))
    }

    fn cast(&mut self) -> Result<ExprId> {
        self.bump()?; // CAST
        self.expect(Tok::LParen)?;
        let arg = self.expr()?;
        self.expect_kw(Kw::As)?;
        let ty = self.type_name()?;
        self.expect(Tok::RParen)?;
        Ok(self.arena.push(Expr::Cast { arg, ty }))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Code;
    use crate::sql::lexer::{keyword, KEYWORDS};

    // --- テスト用ヘルパ -----------------------------------------------------

    /// 式木を完全括弧付きの文字列へ戻す。リテラルは型が分かる形で出す。
    fn r(a: &ExprArena, id: ExprId) -> String {
        match a.get(id) {
            Expr::Literal(v) => lit(v),
            Expr::Param(n) => format!("?{}", n),
            Expr::ColumnRef { qualifier, name } => match qualifier {
                Some(q) => format!("{}.{}", q, name),
                None => name.clone(),
            },
            Expr::Star { qualifier } => match qualifier {
                Some(q) => format!("{}.*", q),
                None => "*".to_string(),
            },
            Expr::Unary { op, arg } => {
                let o = if *op == UnaryOp::Neg { "-" } else { "NOT" };
                format!("({} {})", o, r(a, *arg))
            }
            Expr::Binary { op, lhs, rhs } => {
                format!("({} {} {})", r(a, *lhs), op_name(*op), r(a, *rhs))
            }
            Expr::Cast { arg, ty } => format!("CAST({} AS {})", r(a, *arg), ty_name(*ty)),
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
            Expr::Like { arg, pattern, negated, escape } => {
                let esc = match escape {
                    Some(c) => format!(" ESCAPE '{}'", *c as char),
                    None => String::new(),
                };
                format!(
                    "({}{} LIKE {}{})",
                    r(a, *arg),
                    if *negated { " NOT" } else { "" },
                    r(a, *pattern),
                    esc
                )
            }
            Expr::Function { name, args, distinct, star } => {
                let inner = if *star {
                    "*".to_string()
                } else {
                    let items: Vec<String> = args.iter().map(|i| r(a, *i)).collect();
                    format!("{}{}", if *distinct { "DISTINCT " } else { "" }, items.join(", "))
                };
                format!("{}({})", name, inner)
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
                format!("({}){}", select_str(a, query), alias(al))
            }
            FromItem::Join { left, right, kind, on } => {
                let k = match kind {
                    JoinKind::Inner => "INNER",
                    JoinKind::Left => "LEFT",
                    JoinKind::Right => "RIGHT",
                    JoinKind::Full => "FULL",
                    JoinKind::Cross => "CROSS",
                };
                let on_s = match on {
                    Some(e) => format!(" ON {}", r(a, *e)),
                    None => String::new(),
                };
                format!("({} {} JOIN {}{})", from_str(a, left), k, from_str(a, right), on_s)
            }
        }
    }

    /// SELECT 文を 1 行に潰す。構造の比較用で、SQL として妥当な形ではない。
    fn select_str(a: &ExprArena, s: &SelectStmt) -> String {
        let mut out = String::from("SELECT");
        if s.distinct {
            out.push_str(" DISTINCT");
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
        if let Some(h) = s.having {
            out.push_str(&format!(" HAVING {}", r(a, h)));
        }
        if !s.order_by.is_empty() {
            let o: Vec<String> = s
                .order_by
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
            out.push_str(&format!(" ORDER BY {}", o.join(", ")));
        }
        if let Some(l) = s.limit {
            out.push_str(&format!(" LIMIT {}", l));
        }
        if let Some(o) = s.offset {
            out.push_str(&format!(" OFFSET {}", o));
        }
        out
    }

    fn sel(sql: &str) -> String {
        let p = parse(sql).expect("parse failed");
        match &p.stmt {
            Stmt::Select(s) => select_str(&p.arena, s),
            _ => panic!("not a SELECT"),
        }
    }

    /// `SELECT <expr>` を通して式だけを描画する。
    fn ex(expr: &str) -> String {
        let sql = format!("SELECT {}", expr);
        let p = parse(&sql).expect("parse failed");
        match &p.stmt {
            Stmt::Select(s) => r(&p.arena, s.items[0].expr),
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
        assert_eq!(code("SELECT x IN (SELECT 1)"), Code::UnsupportedFeature as u16);
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
    fn column_refs_and_params() {
        assert_eq!(ex("a"), "a");
        assert_eq!(ex("t.a"), "t.a");
        assert_eq!(ex("\"Mixed Case\".\"x\"\"y\""), "Mixed Case.x\"y");
        let p = parse("SELECT ? WHERE a = ? AND b = ?").expect("parse");
        assert_eq!(p.num_params, 3);
        match &p.stmt {
            Stmt::Select(s) => {
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
             ORDER BY a DESC NULLS FIRST, b ASC NULLS FIRST LIMIT 10 OFFSET 5"
        );
        // 既定の NULL 順序は ASC=LAST / DESC=FIRST。
        assert_eq!(sel("SELECT a FROM t ORDER BY a"), "SELECT a FROM t ORDER BY a ASC NULLS LAST");
        assert_eq!(
            sel("SELECT a FROM t ORDER BY a DESC NULLS LAST"),
            "SELECT a FROM t ORDER BY a DESC NULLS LAST"
        );
        assert_eq!(sel("SELECT * FROM t;"), "SELECT * FROM t");
        assert_eq!(sel("select 1"), "SELECT 1i32");
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
        assert_eq!(code("INSERT INTO t VALUES (1)"), Code::UnsupportedFeature as u16);
        assert_eq!(code("UPDATE t SET a = 1"), Code::UnsupportedFeature as u16);
        assert_eq!(code("CREATE TABLE t (a INT)"), Code::UnsupportedFeature as u16);
        assert_eq!(code("WITH x AS (SELECT 1) SELECT * FROM x"), Code::UnsupportedFeature as u16);
        assert_eq!(code("SELECT 1 UNION SELECT 2"), Code::UnexpectedToken as u16);
        assert_eq!(code("SELECT a & b"), Code::UnexpectedToken as u16);
    }

    #[test]
    fn error_positions() {
        // 位置は必ず「問題のトークンの先頭バイト」を指す。
        assert_eq!(err_at("SELECT FROM t"), (Code::UnexpectedToken as u16, 7));
        assert_eq!(err_at("SELECT 'abc"), (Code::UnterminatedString as u16, 7));
        assert_eq!(err_at("INSERT INTO t VALUES (1)"), (Code::UnsupportedFeature as u16, 0));
        assert_eq!(err_at("SELECT a FROM t WHERE b @ 1"), (Code::UnexpectedToken as u16, 24));
        assert_eq!(err_at("SELECT CAST(x AS FROB)"), (Code::InvalidCast as u16, 17));
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
}
