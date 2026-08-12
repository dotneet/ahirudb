//! SQL パーサ。
//!
//! 文は再帰下降、式は Pratt（優先順位登り）で解く（DESIGN.md §7）。
//! 式は `Box` ではなくアリーナへ積むので、木の破棄で再帰が起きない。
//!
//! 再帰は必ず `MAX_DEPTH` で頭打ちにする。wasm ではスタック枯渇が
//! 回復不能なトラップになるため、深い入力は必ずエラーとして返す。

use crate::format::FormatKind;
use crate::prelude::*;
use crate::rt::hash::eq_ascii_ci;
use crate::sql::ast::{
    BinaryOp, Cte, Expr, ExprArena, ExprId, FromItem, JoinKind, OrderByAll, OrderByItem, Parsed,
    PivotStmt, QueryStmt, SampleMethod, SampleSpec, SelectItem, SelectStmt, SetExpr, SetOp, Stmt,
    UnaryOp, UnpivotStmt, WindowDef, WindowFrame,
};
use crate::sql::lexer::{Kw, Lexer, Tok};
use crate::vector::{Ty, Value};

use types::{int_literal, unquote};

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

/// `Parser::star_modifiers` の戻り値: `(EXCLUDE する列名, REPLACE する
/// (式, 列名), RENAME する (旧列名, 新列名))`。
type StarModifiers = (Vec<String>, Vec<(ExprId, String)>, Vec<(String, String)>);

// 結合強度。大きいほど強く結合する。
const BP_OR: u8 = 1;
const BP_AND: u8 = 2;
const BP_NOT: u8 = 3;
const BP_CMP: u8 = 4;
// `&`/`|`/`<<`/`>>`。比較より強く、`||`/算術より弱い
// （`duckdb` CLI で `1 + 2 & 3` = `(1 + 2) & 3`、`1 & 2 = 0` = `(1 & 2) = 0`
// を確認済み）。4 つの演算子間の相対順位までは実測していないので、
// 単純に 1 段にまとめてある。
const BP_BITWISE: u8 = 5;
const BP_CONCAT: u8 = 6;
const BP_ADD: u8 = 7;
const BP_MUL: u8 = 8;
// `^`/`**`。`duckdb` CLI で `2 + 3^2` = `2 + (3^2)`、`-2^2` = `(-2)^2` を
// 確認済み（`*`/`/` より強く、単項 `-` より弱い）。左結合
// （`2^3^2` = `(2^3)^2` = 64、右結合の `512` にはならない）。
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

    /// カレントトークンが単一引用符の文字列リテラルであることを期待して
    /// 読み進める。`COPY ... TO '<path>'` / `parquet('<path>')` のパス引数
    /// など、"次は必ず文字列リテラル" という位置で使う。
    fn string_lit(&mut self) -> Result<String> {
        let s = match self.cur {
            Tok::Str(s) => unquote(s, b'\''),
            _ => err!(UnexpectedToken, self.pos),
        };
        self.bump()?;
        Ok(s)
    }

    /// 演算子・糖衣構文をプレーンな関数呼び出しノードに desugar する。
    /// `GLOB`/`->`/`->>`/`SIMILAR TO`/配列リテラル/`EXTRACT` が使う
    /// `DISTINCT`/`*`/`FILTER` を持たない単純呼び出しの共通形。
    fn simple_call(&mut self, name: &str, args: Vec<ExprId>) -> ExprId {
        self.arena.push(Expr::Function {
            name: name.into(),
            args,
            distinct: false,
            star: false,
            filter: None,
        })
    }

    /// `AS name(col)` 形の単一列エイリアスを読む。`generate_series`/`range`/
    /// `UNNEST` の FROM 項目に共通する「列名リストは 1 個だけ」という形。
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

    /// 2 トークン目が文脈依存キーワードに一致するか（`GROUPING SETS` の
    /// `SETS` 判定用。`is_soft_kw` の 2 トークン先読み版）。
    #[inline]
    fn peek_is_soft_kw(&self, word: &[u8]) -> Result<bool> {
        Ok(matches!(self.peek()?, Tok::Ident(s) if eq_ascii_ci(s.as_bytes(), word)))
    }

    /// `peek_is_soft_kw(b"to")` の `to`-対応版。`SIMILAR TO` の 2 トークン目
    /// 先読みに使う。`to` は `ddl` フィーチャが有効なとき `DDL_KEYWORDS`
    /// （`ALTER TABLE ... RENAME TO` 用）により `Tok::Kw(Kw::To)` として来る
    /// ため、`peek_is_soft_kw` だけでは（`Tok::Ident` しか見ないので）
    /// `ddl` 有効ビルドで `SIMILAR TO` を誤って認識し損ねる
    /// （`expect_to` の doc コメント参照、同じ二重表現の問題）。
    #[inline]
    fn peek_is_to(&self) -> Result<bool> {
        #[cfg(feature = "ddl")]
        if matches!(self.peek()?, Tok::Kw(Kw::To)) {
            return Ok(true);
        }
        self.peek_is_soft_kw(b"to")
    }

    /// `is_soft_kw(b"into")` の `into`-対応版。`UNPIVOT ... INTO NAME ...` の
    /// `INTO` 判定に使う。`into` は `dml` フィーチャが有効なとき
    /// `INSERT INTO` 用に別途グローバル予約されている（`DML_KEYWORDS`）ので、
    /// `is_soft_kw` だけでは（`Tok::Ident` しか見ないので）`dml` 有効ビルドで
    /// `UNPIVOT ... INTO` を誤って認識し損ねる（`expect_to`/`peek_is_to` の
    /// doc コメント参照、同じ二重表現の問題）。
    fn is_into(&self) -> bool {
        #[cfg(feature = "dml")]
        if self.is(Tok::Kw(Kw::Into)) {
            return true;
        }
        self.is_soft_kw(b"into")
    }

    /// `self.cur` が比較演算子であるという前提で、その直後が
    /// `ANY|ALL|SOME (` の並びかどうかを、何も消費せずに判定する。
    /// `Some(true)` = `ALL`、`Some(false)` = `ANY`/`SOME`、`None` = 量化比較
    /// ではない（呼び出し側は通常の中置演算子として扱う）。
    ///
    /// `(` まで見てから確定させるのは、`ANY`/`SOME` を予約語にしていない
    /// ため（`is_soft_kw` の doc、ファイル冒頭 `sql/lexer.rs` の
    /// ROWS/RANGE/QUALIFY 事故のコメント参照）: `x > any` のように `any` が
    /// ただの列名の場合を、`x > ANY (SELECT ...)` と取り違えないようにする。
    /// `lex` を複製して 2 トークン先読みするだけで `self` の位置は動かさない
    /// （`peek`/`peek_is_soft_kw` と同じ流儀）。
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

    /// `self.cur` が関数呼び出しの開き括弧であるという前提で、対応する閉じ
    /// 括弧までの**同じ入れ子深さ**に `want` が現れるかを、何も消費せずに
    /// 判定する（`peek_quantifier` と同じく `lex` の複製で先読みするだけ）。
    ///
    /// SQL 標準の `position(a IN b)` / `trim(BOTH x FROM s)` のように、
    /// 引数リストの途中にキーワードを挟む構文かどうかを、引数の式そのものを
    /// パースする前に確定させるために使う。`EXTRACT` の 2 トークン先読みでは
    /// 足りない（キーワードの前に任意の長さの式が来る）ケース専用。
    ///
    /// 内側の呼び出し・部分クエリに現れた同じ語は深さで弾く
    /// （`trim(f(x FROM y))` のような形を取り違えない）。入力は常に有限で、
    /// 走査は閉じ括弧か `Eof` で必ず止まる。
    fn call_has_top_level(&self, want: Tok<'a>) -> Result<bool> {
        let mut lx = self.lex.clone();
        // `self.lex` は既に `self.cur`（開き括弧）の**次**を指しているので、
        // 深さは 1 から数え始める。
        let mut depth = 1u32;
        loop {
            match lx.next_token()?.tok {
                Tok::Eof => return Ok(false),
                Tok::LParen | Tok::LBracket => depth += 1,
                Tok::RParen | Tok::RBracket => {
                    // 開き括弧（`self.cur`）に対応する閉じ括弧まで来たら終わり。
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
}

impl<'a> Parser<'a> {
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
