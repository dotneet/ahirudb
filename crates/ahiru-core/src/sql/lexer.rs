//! SQL トークナイザ。
//!
//! トークンは入力のスライスを借用するだけで、`String` を一切作らない。
//! 引用符の畳み込み（`''` / `""`）や数値の変換は、AST ノードを組み立てる
//! パーサ側に寄せてある。字句解析は 1 パス・無確保で回るのが狙い。
//!
//! 入力は信用できない。境界検査を必ず行い、破損に対しては `Err` を返す
//! （パニックしない）。エラー位置は常に入力先頭からのバイト位置。

use crate::prelude::*;
use crate::rt::hash::eq_ascii_ci;

/// 予約語。
///
/// 型名（INTEGER / VARCHAR など）はここに入れない。予約語にすると同名の
/// 列が書けなくなるうえ、表が伸びてコードサイズに響くため、CAST の型名は
/// 識別子として受けてパーサ側で引く。
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum Kw {
    And,
    As,
    Asc,
    Between,
    By,
    Case,
    Cast,
    Cross,
    Desc,
    Describe,
    Distinct,
    Else,
    End,
    Escape,
    Explain,
    False,
    First,
    From,
    Full,
    Group,
    Having,
    In,
    Inner,
    Is,
    Join,
    Last,
    Left,
    Like,
    Limit,
    Not,
    Null,
    Nulls,
    Offset,
    On,
    Or,
    Order,
    Outer,
    Right,
    Select,
    Show,
    Tables,
    Then,
    True,
    When,
    Where,
}

/// 予約語表。**(長さ, 小文字化した先頭バイト) の昇順**に並べること。
/// この順序が二分探索の前提になっている（`keyword`）。
pub(crate) static KEYWORDS: &[(&[u8], Kw)] = &[
    // 2
    (b"as", Kw::As),
    (b"by", Kw::By),
    (b"in", Kw::In),
    (b"is", Kw::Is),
    (b"on", Kw::On),
    (b"or", Kw::Or),
    // 3
    (b"and", Kw::And),
    (b"asc", Kw::Asc),
    (b"end", Kw::End),
    (b"not", Kw::Not),
    // 4
    (b"case", Kw::Case),
    (b"cast", Kw::Cast),
    (b"desc", Kw::Desc),
    (b"else", Kw::Else),
    (b"from", Kw::From),
    (b"full", Kw::Full),
    (b"join", Kw::Join),
    (b"last", Kw::Last),
    (b"left", Kw::Left),
    (b"like", Kw::Like),
    (b"null", Kw::Null),
    (b"show", Kw::Show),
    (b"then", Kw::Then),
    (b"true", Kw::True),
    (b"when", Kw::When),
    // 5
    (b"cross", Kw::Cross),
    (b"false", Kw::False),
    (b"first", Kw::First),
    (b"group", Kw::Group),
    (b"inner", Kw::Inner),
    (b"limit", Kw::Limit),
    (b"nulls", Kw::Nulls),
    (b"order", Kw::Order),
    (b"outer", Kw::Outer),
    (b"right", Kw::Right),
    (b"where", Kw::Where),
    // 6
    (b"escape", Kw::Escape),
    (b"having", Kw::Having),
    (b"offset", Kw::Offset),
    (b"select", Kw::Select),
    (b"tables", Kw::Tables),
    // 7
    (b"between", Kw::Between),
    (b"explain", Kw::Explain),
    // 8
    (b"describe", Kw::Describe),
    (b"distinct", Kw::Distinct),
];

/// 探索キー: 長さと小文字化した先頭バイトを 1 語に詰めたもの。
#[inline]
fn kw_key(name: &[u8]) -> u32 {
    // 表は空文字列を含まないので添字 0 は常に有効。
    ((name.len() as u32) << 8) | (name[0] | 0x20) as u32
}

/// 予約語を引く。予約語でなければ `None`。
///
/// 文字列比較の連鎖を避けるため、まず (長さ, 先頭バイト) で二分探索して
/// 候補の区間を求め、その中（高々数個）だけを大小無視で比較する。
pub fn keyword(s: &[u8]) -> Option<Kw> {
    if s.len() < 2 || s.len() > 8 {
        return None;
    }
    let key = kw_key(s);
    let (mut lo, mut hi) = (0usize, KEYWORDS.len());
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if kw_key(KEYWORDS[mid].0) < key {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    while lo < KEYWORDS.len() && kw_key(KEYWORDS[lo].0) == key {
        if eq_ascii_ci(KEYWORDS[lo].0, s) {
            return Some(KEYWORDS[lo].1);
        }
        lo += 1;
    }
    None
}

/// トークン種別。文字列を持つ変種はいずれも**入力の生スライス**であり、
/// 引用符の中身は未展開（`''` / `""` がそのまま残る）。
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum Tok<'a> {
    Eof,
    Kw(Kw),
    /// 引用符なし識別子。大小を無視して比較する。
    Ident(&'a str),
    /// 二重引用符付き識別子。大小を区別する。
    QIdent(&'a str),
    /// 単一引用符付き文字列。
    Str(&'a str),
    /// 整数リテラルの生テキスト（数字のみ）。
    Int(&'a str),
    /// 小数点・指数を含む数値リテラルの生テキスト。
    Float(&'a str),
    /// `?` プレースホルダ。
    Param,
    LParen,
    RParen,
    Comma,
    Dot,
    Semi,
    /// `*`。乗算と `SELECT *` の両方に使う。
    Star,
    Plus,
    Minus,
    Slash,
    Percent,
    /// `||`
    Concat,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// トークンと、その先頭の入力バイト位置。位置はそのままエラー報告に使う。
#[derive(Clone, Copy)]
pub struct Token<'a> {
    pub tok: Tok<'a>,
    pub pos: usize,
}

pub struct Lexer<'a> {
    src: &'a str,
    pos: usize,
}

#[inline]
fn is_ident_start(c: u8) -> bool {
    // 非 ASCII バイトも識別子に許す。UTF-8 の継続バイトは 0x80 以上なので、
    // この判定だけで多バイト文字が丸ごと 1 識別子に収まり、境界も崩れない。
    c.is_ascii_alphabetic() || c == b'_' || c >= 0x80
}

#[inline]
fn is_ident_cont(c: u8) -> bool {
    is_ident_start(c) || c.is_ascii_digit()
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Self {
        Lexer { src, pos: 0 }
    }

    #[inline]
    fn b(&self) -> &'a [u8] {
        self.src.as_bytes()
    }

    /// 空白とコメントを読み飛ばす。
    fn skip_trivia(&mut self) -> Result<()> {
        let b = self.b();
        loop {
            while self.pos < b.len() && matches!(b[self.pos], b' ' | b'\t' | b'\r' | b'\n') {
                self.pos += 1;
            }
            if self.pos + 1 < b.len() && b[self.pos] == b'-' && b[self.pos + 1] == b'-' {
                self.pos += 2;
                while self.pos < b.len() && b[self.pos] != b'\n' {
                    self.pos += 1;
                }
                continue;
            }
            if self.pos + 1 < b.len() && b[self.pos] == b'/' && b[self.pos + 1] == b'*' {
                let start = self.pos;
                self.pos += 2;
                // ブロックコメントは入れ子にしない（SQL 標準どおり）。
                loop {
                    if self.pos + 1 >= b.len() {
                        self.pos = b.len();
                        err!(SyntaxError, start);
                    }
                    if b[self.pos] == b'*' && b[self.pos + 1] == b'/' {
                        self.pos += 2;
                        break;
                    }
                    self.pos += 1;
                }
                continue;
            }
            return Ok(());
        }
    }

    pub fn next_token(&mut self) -> Result<Token<'a>> {
        self.skip_trivia()?;
        let b = self.b();
        let start = self.pos;
        if start >= b.len() {
            return Ok(Token { tok: Tok::Eof, pos: start });
        }
        let c = b[start];
        let tok = if c.is_ascii_digit() {
            self.number()
        } else if c == b'\'' {
            self.quoted(b'\'')?
        } else if c == b'"' {
            self.quoted(b'"')?
        } else if is_ident_start(c) {
            self.ident()
        } else {
            self.punct()?
        };
        Ok(Token { tok, pos: start })
    }

    fn ident(&mut self) -> Tok<'a> {
        let b = self.b();
        let start = self.pos;
        let mut i = start;
        while i < b.len() && is_ident_cont(b[i]) {
            i += 1;
        }
        self.pos = i;
        let text = &self.src[start..i];
        match keyword(text.as_bytes()) {
            Some(k) => Tok::Kw(k),
            None => Tok::Ident(text),
        }
    }

    fn number(&mut self) -> Tok<'a> {
        let b = self.b();
        let start = self.pos;
        let mut i = start;
        let mut is_float = false;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i < b.len() && b[i] == b'.' {
            is_float = true;
            i += 1;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
        }
        if i < b.len() && (b[i] | 0x20) == b'e' {
            // 指数部は数字が 1 桁以上必要。無ければ `e` 以降は別トークン扱い。
            let mut j = i + 1;
            if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
                j += 1;
            }
            if j < b.len() && b[j].is_ascii_digit() {
                while j < b.len() && b[j].is_ascii_digit() {
                    j += 1;
                }
                i = j;
                is_float = true;
            }
        }
        self.pos = i;
        let text = &self.src[start..i];
        if is_float {
            Tok::Float(text)
        } else {
            Tok::Int(text)
        }
    }

    /// 引用符で囲まれた字句。引用符 2 個はエスケープとして読み飛ばすだけで、
    /// 展開はしない（確保を避けるため）。
    fn quoted(&mut self, q: u8) -> Result<Tok<'a>> {
        let b = self.b();
        let start = self.pos;
        let mut i = start + 1;
        loop {
            if i >= b.len() {
                self.pos = b.len();
                err!(UnterminatedString, start);
            }
            if b[i] == q {
                if i + 1 < b.len() && b[i + 1] == q {
                    i += 2;
                    continue;
                }
                break;
            }
            i += 1;
        }
        self.pos = i + 1;
        let raw = &self.src[start + 1..i];
        Ok(if q == b'\'' { Tok::Str(raw) } else { Tok::QIdent(raw) })
    }

    fn punct(&mut self) -> Result<Tok<'a>> {
        let b = self.b();
        let start = self.pos;
        let c = b[start];
        self.pos += 1;
        // 2 文字演算子は 2 バイト目を見てから伸ばす。
        let mut eat = |x: u8| -> bool {
            if self.pos < b.len() && b[self.pos] == x {
                self.pos += 1;
                true
            } else {
                false
            }
        };
        let t = match c {
            b'(' => Tok::LParen,
            b')' => Tok::RParen,
            b',' => Tok::Comma,
            b'.' => Tok::Dot,
            b';' => Tok::Semi,
            b'*' => Tok::Star,
            b'+' => Tok::Plus,
            b'-' => Tok::Minus,
            b'/' => Tok::Slash,
            b'%' => Tok::Percent,
            b'?' => Tok::Param,
            // `==` は `=` の別名。
            b'=' => {
                eat(b'=');
                Tok::Eq
            }
            b'!' => {
                if eat(b'=') {
                    Tok::Ne
                } else {
                    err!(UnexpectedToken, start)
                }
            }
            b'<' => {
                if eat(b'=') {
                    Tok::Le
                } else if eat(b'>') {
                    Tok::Ne
                } else {
                    Tok::Lt
                }
            }
            b'>' => {
                if eat(b'=') {
                    Tok::Ge
                } else {
                    Tok::Gt
                }
            }
            b'|' => {
                if eat(b'|') {
                    Tok::Concat
                } else {
                    err!(UnexpectedToken, start)
                }
            }
            _ => err!(UnexpectedToken, start),
        };
        Ok(t)
    }
}
