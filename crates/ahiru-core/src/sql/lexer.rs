//! The SQL tokenizer.
//!
//! Tokens merely borrow slices of the input and never build a `String`. Folding quotes
//! (`''` / `""`) and converting numbers are pushed to the parser, which assembles the
//! AST nodes. The aim is for lexing to run in one pass with no allocation.
//!
//! The input is untrusted. Bounds are always checked, and corruption yields `Err`
//! (never a panic). Error positions are always byte offsets from the start of the input.

use crate::prelude::*;
use crate::rt::hash::eq_ascii_ci;

/// Reserved words.
///
/// Type names (INTEGER / VARCHAR and so on) do not belong here. Reserving them would
/// make same-named columns unwritable, and would grow the table and thus the code size,
/// so CAST type names are taken as identifiers and looked up by the parser.
///
/// For the same reason `OVER` / `PARTITION` / `ROWS` / `RANGE` are not here either.
/// Column names come from data files and are not chosen by the user, so reserving
/// common words would create columns unreferenceable without quotes. These are
/// context-dependent keywords meaningful only inside a window specification, matched by spelling in the parser.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum Kw {
    // --- ddl/dml: words reserved only by the CREATE/INSERT/UPDATE/DELETE/ALTER
    // statements in `sql/parser.rs`. They are kept separate from the base vocabulary so
    // that they remain usable as ordinary identifiers (column names and so on) while
    // the features are OFF (see the comments on `KEYWORDS`/`DDL_KEYWORDS`/`DML_KEYWORDS`).
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
    // Words reserved only by the `export` feature (`COPY (<query>) TO ...`).
    // They are keywords of a one-shot statement that appears only at the start of a
    // statement, so they can be treated like the DDL syntactic heads `Create`/`Drop`
    // (`TO`/`FORMAT` do not belong here -- see the comment at the top of the file;
    // `copy_stmt` in `sql/parser.rs` matches them by spelling as context-dependent keywords).
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
    /// The head of a `WINDOW name AS (...)` clause. An ordinary reserved word for the
    /// same reason as `QUALIFY`: as a context-dependent keyword, a form with no
    /// intervening clause such as `FROM t WINDOW w AS (...)` would have `opt_alias` eat
    /// `WINDOW` as a table alias and break the syntax. DuckDB itself treats `WINDOW` as
    /// reserved (unusable as a column name, only as an alias like `AS window`), so the
    /// risk of breaking real data column names was judged low.
    Window,
    With,
}

/// The reserved-word table. Must be sorted **ascending by (length, lowercased first byte)**.
/// That ordering is what the binary search (`keyword`) assumes.
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

/// Words reserved only by the `ddl` feature (CREATE TABLE / CREATE VIEW / DROP TABLE /
/// ALTER TABLE and friends). Kept in a separate table from `KEYWORDS` so builds with
/// the feature OFF can still use them as ordinary identifiers (column names and so on).
/// The ascending-order constraint is the same as for `KEYWORDS`.
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

/// Words reserved only by the `dml` feature (INSERT / UPDATE / DELETE and friends).
/// `dml` implies `ddl` (see Cargo.toml), so `DDL_KEYWORDS` becomes active at the same time.
#[cfg(feature = "dml")]
static DML_KEYWORDS: &[(&[u8], Kw)] = &[
    (b"set", Kw::Set),
    (b"into", Kw::Into),
    (b"delete", Kw::Delete),
    (b"insert", Kw::Insert),
    (b"update", Kw::Update),
    (b"values", Kw::Values),
];

/// Words reserved only by the `export` feature (`COPY (<query>) TO ...`).
/// It is an independent feature from `ddl`/`dml`, hence a separate table (so `COPY` can
/// be reserved even in a build that enables only `export`).
#[cfg(feature = "export")]
static EXPORT_KEYWORDS: &[(&[u8], Kw)] = &[(b"copy", Kw::Copy)];

/// The search key: the length and the lowercased first byte packed into one word.
#[inline]
fn kw_key(name: &[u8]) -> u32 {
    // The table contains no empty string, so index 0 is always valid.
    ((name.len() as u32) << 8) | (name[0] | 0x20) as u32
}

/// Binary-searches `table` by (length, first byte) and compares case-insensitively only
/// within the candidate range. The shared implementation of `keyword`/`keyword_in`. `table` must be sorted by `kw_key`.
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

/// Looks up a reserved word. `None` if it is not one.
///
/// To avoid a chain of string comparisons, it first binary-searches by (length, first
/// byte) to find the candidate range, then compares case-insensitively only within it (at most a handful).
pub fn keyword(s: &[u8]) -> Option<Kw> {
    // Word lengths in the table range over 2..=9 (the longest is INTERSECT).
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

/// Token kinds. Every variant carrying a string holds a **raw slice of the input**,
/// with quote contents unexpanded (`''` / `""` remain as they are).
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum Tok<'a> {
    Eof,
    Kw(Kw),
    /// An unquoted identifier. Compared case-insensitively.
    Ident(&'a str),
    /// A double-quoted identifier. Case-sensitive.
    QIdent(&'a str),
    /// A single-quoted string.
    Str(&'a str),
    /// The raw text of an integer literal (digits only).
    Int(&'a str),
    /// The raw text of a numeric literal containing a decimal point or exponent.
    Float(&'a str),
    /// A `?` placeholder.
    Param,
    LParen,
    RParen,
    /// `[`. At the start of an expression it opens an array literal `[expr, ...]`; when
    /// it follows the result of `primary_atom()` it is subscripting `expr[i]` / slicing
    /// `expr[i:j]` (see `primary`/`postfix_ops`/`subscript` in `sql::parser`). Position
    /// distinguishes them, so the lexer can emit the same token either way.
    LBracket,
    RBracket,
    Comma,
    Dot,
    Semi,
    /// `*`. Used for both multiplication and `SELECT *`.
    Star,
    Plus,
    Minus,
    Slash,
    Percent,
    /// `||`
    Concat,
    /// `->` (JSON path extraction; sugar for `json_extract`)
    Arrow,
    /// `->>` (JSON path extraction as text; sugar for `json_extract_string`)
    LongArrow,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    /// `::` (sugar for `CAST(expr AS ty)`)
    ColonColon,
    /// A lone `:`. Used only as the separator in a subscript slice `expr[i:j]` (either bound may be omitted).
    Colon,
    /// `^` or `**` (exponentiation)
    Pow,
    /// `^@`. PostgreSQL/DuckDB's "starts with" operator; sugar for
    /// `starts_with(lhs, rhs)` (`sql::parser::expr_body`). Matched before
    /// bare `^` (`Pow`) in `punct`, the same longest-match-first rule the
    /// `~~~`/`!~~*` families use — without it `'abc' ^@ 'a'` would lex as
    /// `^` followed by prefix `@` (absolute value) and die as a type error.
    CaretAt,
    /// `&` (bitwise AND, integers only)
    Amp,
    /// `|` (bitwise OR, integers only; `||` is `Concat` separately)
    Pipe,
    /// `<<` (left shift, integers only)
    Shl,
    /// `>>` (right shift, integers only)
    Shr,
    /// `~`. Prefix means integer bitwise NOT; infix means regular-expression match
    /// (sugar for `regexp_full_match`, the same as `SIMILAR TO`).
    Tilde,
    /// `!~` (the negation of `~`; sugar for `NOT (a ~ b)`)
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

/// A token plus the input byte position of its start. The position is used directly in error reports.
#[derive(Clone, Copy)]
pub struct Token<'a> {
    pub tok: Tok<'a>,
    pub pos: usize,
}

/// `Clone` exists for two-token lookahead. Since it is only borrows and indices, the
/// copy allocates nothing, and advancing the copy does not move the original (`Parser::peek`).
#[derive(Clone)]
pub struct Lexer<'a> {
    src: &'a str,
    pos: usize,
}

#[inline]
fn is_ident_start(c: u8) -> bool {
    // Non-ASCII bytes are allowed in identifiers too. UTF-8 continuation bytes are 0x80
    // and above, so this check alone keeps a multi-byte character within one identifier without breaking boundaries.
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

    /// Skips whitespace and comments.
    fn skip_trivia(&mut self) -> Result<()> {
        let b = self.b();
        loop {
            while self.pos < b.len() {
                if matches!(b[self.pos], b' ' | b'\t' | b'\r' | b'\n') {
                    self.pos += 1;
                } else if let Some(c) = self.src[self.pos..].chars().next() {
                    if c.is_whitespace() {
                        self.pos += c.len_utf8();
                    } else {
                        break;
                    }
                } else {
                    break;
                }
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
                // Block comments do not nest (per the SQL standard).
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
            // An exponent needs at least one digit. Without one, everything from `e` on is a separate token.
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

    /// A quoted lexeme. A doubled quote is merely skipped as an escape and is not
    /// expanded (to avoid allocating).
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
        // Two-character operators are extended after looking at the second byte.
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
            // Longest match first: `^@` (prefix/starts-with) before bare `^`
            // (power). See `Tok::CaretAt`.
            b'^' => {
                if eat(b'@') {
                    Tok::CaretAt
                } else {
                    Tok::Pow
                }
            }
            b'@' => Tok::At,
            b':' => {
                if eat(b':') {
                    Tok::ColonColon
                } else {
                    Tok::Colon
                }
            }
            // `==` is an alias for `=`.
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
