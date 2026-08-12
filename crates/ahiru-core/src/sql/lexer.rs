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
///
/// 同じ理由で `OVER` / `PARTITION` / `ROWS` / `RANGE` もここには入れない。
/// 列名はデータファイル由来で利用者が選べないため、ありふれた語を予約語に
/// すると引用符無しでは参照できない列ができてしまう。これらはウィンドウ指定
/// の中でだけ意味を持つ文脈依存キーワードとして、パーサが綴りで照合する。
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum Kw {
    // --- ddl/dml: `sql/parser.rs` の CREATE/INSERT/UPDATE/DELETE/ALTER 系
    // でのみ予約する語。基本語彙とは別枠にしてあるのは、フィーチャが OFF の
    // 間はこれらを普通の識別子（列名など）として使えるようにするため
    // （`KEYWORDS`/`DDL_KEYWORDS`/`DML_KEYWORDS` のコメント参照）。
    #[cfg(feature = "ddl")]
    Add,
    All,
    #[cfg(feature = "ddl")]
    Alter,
    And,
    As,
    Asc,
    Between,
    By,
    Case,
    Cast,
    #[cfg(feature = "ddl")]
    Column,
    // `export` フィーチャでのみ予約する語（`COPY (<query>) TO ...`）。
    // 文の先頭でしか出現しない一発実行文のキーワードなので、`Create`/`Drop`
    // など DDL の統語頭語と同じ扱いでよい（`TO`/`FORMAT` はここには入れない
    // — ファイル冒頭のコメント参照。`sql/parser.rs` の `copy_stmt` が
    // 文脈依存キーワードとして綴りで照合する）。
    #[cfg(feature = "export")]
    Copy,
    #[cfg(feature = "ddl")]
    Create,
    Cross,
    #[cfg(feature = "ddl")]
    Default,
    #[cfg(feature = "dml")]
    Delete,
    Desc,
    Describe,
    Distinct,
    #[cfg(feature = "ddl")]
    Drop,
    Else,
    End,
    Escape,
    Except,
    Exists,
    Explain,
    False,
    First,
    From,
    Full,
    Group,
    Having,
    #[cfg(feature = "ddl")]
    If,
    Ilike,
    In,
    Inner,
    #[cfg(feature = "dml")]
    Insert,
    Intersect,
    #[cfg(feature = "dml")]
    Into,
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
    Qualify,
    #[cfg(feature = "ddl")]
    Rename,
    #[cfg(feature = "ddl")]
    Replace,
    Right,
    Select,
    #[cfg(feature = "dml")]
    Set,
    Show,
    #[cfg(feature = "ddl")]
    Table,
    Tables,
    Then,
    #[cfg(feature = "ddl")]
    To,
    True,
    Union,
    #[cfg(feature = "dml")]
    Update,
    #[cfg(feature = "dml")]
    Values,
    #[cfg(feature = "ddl")]
    View,
    When,
    Where,
    /// `WINDOW name AS (...)` 句の先頭。`QUALIFY` と同じ理由で通常の予約語に
    /// する: 文脈依存キーワードにすると `FROM t WINDOW w AS (...)` のように
    /// 直前に別の句を挟まない形で、`opt_alias` が `WINDOW` をテーブル別名
    /// として食ってしまい構文が壊れる。DuckDB 自身も `WINDOW` を予約語として
    /// 扱っている（列名としては使えず、`AS window` のような別名にしか使えない）
    /// ので、実データの列名を壊す心配は薄いと判断した。
    Window,
    With,
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
    (b"all", Kw::All),
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
    (b"with", Kw::With),
    // 5
    (b"cross", Kw::Cross),
    (b"false", Kw::False),
    (b"first", Kw::First),
    (b"group", Kw::Group),
    (b"ilike", Kw::Ilike),
    (b"inner", Kw::Inner),
    (b"limit", Kw::Limit),
    (b"nulls", Kw::Nulls),
    (b"order", Kw::Order),
    (b"outer", Kw::Outer),
    (b"right", Kw::Right),
    (b"union", Kw::Union),
    (b"where", Kw::Where),
    // 6
    (b"escape", Kw::Escape),
    (b"except", Kw::Except),
    (b"exists", Kw::Exists),
    (b"having", Kw::Having),
    (b"offset", Kw::Offset),
    (b"select", Kw::Select),
    (b"tables", Kw::Tables),
    (b"window", Kw::Window),
    // 7
    (b"between", Kw::Between),
    (b"explain", Kw::Explain),
    (b"qualify", Kw::Qualify),
    // 8
    (b"describe", Kw::Describe),
    (b"distinct", Kw::Distinct),
    // 9
    (b"intersect", Kw::Intersect),
];

/// `ddl` フィーチャでのみ予約する語（CREATE TABLE / CREATE VIEW / DROP TABLE /
/// ALTER TABLE 系）。`KEYWORDS` と別表にしてあるのは、フィーチャが OFF の
/// ビルドではこれらを従来どおり普通の識別子（列名など）として使えるように
/// するため。昇順制約は `KEYWORDS` と同じ。
#[cfg(feature = "ddl")]
static DDL_KEYWORDS: &[(&[u8], Kw)] = &[
    (b"if", Kw::If),
    (b"to", Kw::To),
    (b"add", Kw::Add),
    (b"drop", Kw::Drop),
    (b"view", Kw::View),
    (b"alter", Kw::Alter),
    (b"table", Kw::Table),
    (b"column", Kw::Column),
    (b"create", Kw::Create),
    (b"rename", Kw::Rename),
    (b"default", Kw::Default),
    (b"replace", Kw::Replace),
];

/// `dml` フィーチャでのみ予約する語（INSERT / UPDATE / DELETE 系）。
/// `dml` は `ddl` を暗黙に含む（Cargo.toml）ので、`DDL_KEYWORDS` も同時に
/// 有効になる。
#[cfg(feature = "dml")]
static DML_KEYWORDS: &[(&[u8], Kw)] = &[
    (b"set", Kw::Set),
    (b"into", Kw::Into),
    (b"delete", Kw::Delete),
    (b"insert", Kw::Insert),
    (b"update", Kw::Update),
    (b"values", Kw::Values),
];

/// `export` フィーチャでのみ予約する語（`COPY (<query>) TO ...`）。
/// `ddl`/`dml` とは独立のフィーチャなので別表にしてある
/// （`export` だけを有効にしたビルドでも `COPY` を予約できるように）。
#[cfg(feature = "export")]
static EXPORT_KEYWORDS: &[(&[u8], Kw)] = &[(b"copy", Kw::Copy)];

/// 探索キー: 長さと小文字化した先頭バイトを 1 語に詰めたもの。
#[inline]
fn kw_key(name: &[u8]) -> u32 {
    // 表は空文字列を含まないので添字 0 は常に有効。
    ((name.len() as u32) << 8) | (name[0] | 0x20) as u32
}

/// `table` を (長さ, 先頭バイト) で二分探索し、候補区間だけ大小無視で比較する。
/// `keyword`/`keyword_in` の共通実装。`table` は `kw_key` の昇順であること。
fn keyword_in(table: &[(&[u8], Kw)], s: &[u8]) -> Option<Kw> {
    let key = kw_key(s);
    let (mut lo, mut hi) = (0usize, table.len());
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if kw_key(table[mid].0) < key {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    while lo < table.len() && kw_key(table[lo].0) == key {
        if eq_ascii_ci(table[lo].0, s) {
            return Some(table[lo].1);
        }
        lo += 1;
    }
    None
}

/// 予約語を引く。予約語でなければ `None`。
///
/// 文字列比較の連鎖を避けるため、まず (長さ, 先頭バイト) で二分探索して
/// 候補の区間を求め、その中（高々数個）だけを大小無視で比較する。
pub fn keyword(s: &[u8]) -> Option<Kw> {
    // 表に載る語長は 2..=9（最長は INTERSECT）。
    if s.len() < 2 || s.len() > 9 {
        return None;
    }
    if let Some(k) = keyword_in(KEYWORDS, s) {
        return Some(k);
    }
    #[cfg(feature = "ddl")]
    if let Some(k) = keyword_in(DDL_KEYWORDS, s) {
        return Some(k);
    }
    #[cfg(feature = "dml")]
    if let Some(k) = keyword_in(DML_KEYWORDS, s) {
        return Some(k);
    }
    #[cfg(feature = "export")]
    if let Some(k) = keyword_in(EXPORT_KEYWORDS, s) {
        return Some(k);
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
    /// `[`。式の先頭では配列リテラル `[expr, ...]` の開始、`primary_atom()`
    /// の結果に後置で続くときは添字アクセス `expr[i]`/スライス `expr[i:j]`
    /// （`sql::parser` の `primary`/`postfix_ops`/`subscript` 参照）。位置で
    /// 区別できるので、レキサ側は同じトークンのままでよい。
    LBracket,
    RBracket,
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
    /// `->`（JSON パス抽出。`json_extract` の糖衣構文）
    Arrow,
    /// `->>`（JSON パス抽出・テキスト化。`json_extract_string` の糖衣構文）
    LongArrow,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    /// `::`（`CAST(expr AS ty)` の糖衣構文）
    ColonColon,
    /// 単独の `:`。添字スライス `expr[i:j]`（境界省略可）の区切りにのみ使う。
    Colon,
    /// `^` または `**`（べき乗）
    Pow,
    /// `&`（ビット単位 AND、整数のみ）
    Amp,
    /// `|`（ビット単位 OR、整数のみ。`||` は別途 `Concat`）
    Pipe,
    /// `<<`（左シフト、整数のみ）
    Shl,
    /// `>>`（右シフト、整数のみ）
    Shr,
    /// `~`。前置なら整数のビット単位 NOT、中置なら正規表現一致
    /// （`regexp_full_match` への糖衣構文、`SIMILAR TO` と同じ）。
    Tilde,
    /// `!~`（`~` の否定、`NOT (a ~ b)` の糖衣構文）
    NotTilde,
    /// `~~`. PostgreSQL/DuckDB alias for `LIKE` (`sql::parser::expr_body`
    /// desugars this to `Expr::Like`). Two tildes, not to be confused with
    /// infix `~` (regex match) above.
    TildeTilde,
    /// `~~*`. Alias for `ILIKE` (case-insensitive `~~`).
    TildeTildeStar,
    /// `~~~`. Alias for `GLOB` (desugars to a `glob(...)` call, same as the
    /// `GLOB` keyword — see `sql::parser::expr_body`).
    TildeTildeTilde,
    /// `!~~`. Negated `~~`; folded into `Expr::Like`'s `negated` field
    /// rather than wrapped in `Unary::Not` (`sql::parser::expr_body`).
    NotTildeTilde,
    /// `!~~*`. Negated `~~*`.
    NotTildeTildeStar,
    /// `//`. Integer division: sugar for `BinaryOp::Div` (see the
    /// desugaring site in `sql::parser::expr_body` for why this is a
    /// correct alias for `/` specifically in this engine).
    SlashSlash,
    /// `@` (prefix only). Absolute value, sugar for `abs(x)`
    /// (`sql::parser::Parser::prefix`).
    At,
    /// `!` (postfix only). Factorial, sugar for `factorial(x)`
    /// (`sql::parser::Parser::primary`/`cast_postfix`). Also the leading
    /// byte of `!=`/`!~`/`!~~`/`!~~*`, which the lexer matches first as
    /// their own longer tokens — a lone `!` only remains when none of
    /// those match.
    Bang,
}

/// トークンと、その先頭の入力バイト位置。位置はそのままエラー報告に使う。
#[derive(Clone, Copy)]
pub struct Token<'a> {
    pub tok: Tok<'a>,
    pub pos: usize,
}

/// `Clone` は 2 トークン目の先読み用。借用と添字だけなので複製は無確保で、
/// 複製側を進めても本体の位置は動かない（`Parser::peek`）。
#[derive(Clone)]
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
            b'[' => Tok::LBracket,
            b']' => Tok::RBracket,
            b',' => Tok::Comma,
            b'.' => Tok::Dot,
            b';' => Tok::Semi,
            b'*' => {
                if eat(b'*') {
                    Tok::Pow
                } else {
                    Tok::Star
                }
            }
            b'+' => Tok::Plus,
            b'-' => {
                if eat(b'>') {
                    if eat(b'>') {
                        Tok::LongArrow
                    } else {
                        Tok::Arrow
                    }
                } else {
                    Tok::Minus
                }
            }
            // `//` (integer division). Not a comment: SQL line comments are
            // `--`, already consumed by `skip_trivia` before `punct` runs.
            b'/' => {
                if eat(b'/') {
                    Tok::SlashSlash
                } else {
                    Tok::Slash
                }
            }
            b'%' => Tok::Percent,
            b'?' => Tok::Param,
            b'^' => Tok::Pow,
            b'@' => Tok::At,
            b':' => {
                if eat(b':') {
                    Tok::ColonColon
                } else {
                    Tok::Colon
                }
            }
            // `==` は `=` の別名。
            b'=' => {
                eat(b'=');
                Tok::Eq
            }
            // Longest match first: `!~~*` / `!~~` / `!~` (existing
            // `NotTilde`) / `!=` (existing `Ne`) / bare `!` (`Bang`,
            // postfix factorial — see `Tok::Bang` doc).
            b'!' => {
                if eat(b'~') {
                    if eat(b'~') {
                        if eat(b'*') {
                            Tok::NotTildeTildeStar
                        } else {
                            Tok::NotTildeTilde
                        }
                    } else {
                        Tok::NotTilde
                    }
                } else if eat(b'=') {
                    Tok::Ne
                } else {
                    Tok::Bang
                }
            }
            // Longest match first: `~~~` / `~~*` / `~~` / bare `~`
            // (existing `Tilde`, prefix bitwise NOT or infix regex match —
            // see `Tok::Tilde` doc).
            b'~' => {
                if eat(b'~') {
                    if eat(b'~') {
                        Tok::TildeTildeTilde
                    } else if eat(b'*') {
                        Tok::TildeTildeStar
                    } else {
                        Tok::TildeTilde
                    }
                } else {
                    Tok::Tilde
                }
            }
            b'&' => Tok::Amp,
            b'<' => {
                if eat(b'=') {
                    Tok::Le
                } else if eat(b'>') {
                    Tok::Ne
                } else if eat(b'<') {
                    Tok::Shl
                } else {
                    Tok::Lt
                }
            }
            b'>' => {
                if eat(b'=') {
                    Tok::Ge
                } else if eat(b'>') {
                    Tok::Shr
                } else {
                    Tok::Gt
                }
            }
            b'|' => {
                if eat(b'|') {
                    Tok::Concat
                } else {
                    Tok::Pipe
                }
            }
            _ => err!(UnexpectedToken, start),
        };
        Ok(t)
    }
}
