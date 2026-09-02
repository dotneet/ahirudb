//! The regular expression engine for `regexp_matches` / `regexp_extract` / `regexp_replace`.
//!
//! An in-house implementation using no external crate. Naive recursive backtracking can go
//! exponential on patterns like `(a+)+b`, so it is not used; instead this is a
//! **Thompson NFA + Pike's VM** (the technique explained in Russ Cox's series of articles).
//! The pattern is compiled into an NFA, and "the set of currently live states" advances in
//! parallel per input byte. The states grow by at most the instruction count per byte (reaching
//! the same instruction number counts only once), so the worst case stays within
//! `O(instructions x input length)` and a backtracking blowup cannot occur in principle.
//! The motivation is the same as `LIKE` (`like_match` in `expr/kernels.rs`) avoiding the same
//! problem with a two-pointer method.
//!
//! ## Supported syntax (anything else gives `Err(UnsupportedFeature)` or `Err(SyntaxError)`)
//!
//! - Literal characters (**per UTF-8 scalar value**. Both the pattern and the subject are decoded
//!   as UTF-8, so a multi-byte character is one unit of matching, exactly as in `upper`/`lower`)
//! - `.` (any single character other than a newline)
//! - `*` `+` `?` (greedy quantifiers)
//! - `{n}` `{n,}` `{n,m}` (bounded repetition. As in RE2, DuckDB's internal engine, `n` and `m`
//!   go up to 1000)
//! - Character classes `[abc]` `[^abc]`, ranges `[a-z]`, the shorthands `\d \D \w \W \s \S`, and
//!   the POSIX names `[[:alpha:]]` and friends (see `posix_class`). Class members are code points,
//!   so `[^a]` matches one whole multi-byte character rather than one of its bytes.
//! - Anchors `^` `$` (no `(?m)`-style multiline. `^` matches only the start of the input and `$`
//!   only its end)
//! - Alternation `|`
//! - Groups `(...)` (capturing) and `(?:...)` (non-capturing)
//! - Escaped metacharacters `\. \* \+ \? \( \) \[ \] \{ \} \| \^ \$ \\`
//!
//! **Unsupported** (always an error at `resolve`/compile time; silently interpreting them
//! differently would give the worst failure mode of all, "wrong search results"):
//! - Lookahead and lookbehind `(?=...)` `(?!...)` `(?<=...)` `(?<!...)`
//! - In-pattern backreferences `\1` (the `\1`/`\2` **in the replacement string** are a different
//!   thing and are supported as essential to `regexp_replace`; see `parse_repl`)
//! - Named groups `(?P<name>...)` `(?<name>...)`
//! - Lazy quantifiers `*?` `+?` `??` `{n,m}?`
//! - Word boundaries `\b` `\B`
//! - Case-insensitive matching (`(?i)` or a flags argument). DuckDB supports it, but it was
//!   judged not frequent enough to be worth the implementation cost, so it is deferred in v1.
//!   ASCII-only case folding alone could be added cheaply, but as with `LIKE`, nothing is added
//!   at this granularity until demand is confirmed.
//!
//! ## Size and runtime caps
//!
//! Every one is a value that "caps things reliably with little code".
//! See the comments on each constant for the individual reasons.
//!
//! ## Relationship to `resolve`/`call`
//!
//! `resolve` can only see types, not the pattern string (which is usually a constant across the
//! whole query). So "compile exactly once per query" cannot be arranged at planning time.
//! Instead, inside `call`, if the pattern column's stride is 0 (= a constant column) it
//! `compile`s once outside the row loop, and otherwise (the rare case where the pattern varies
//! per row) recompiles per row.
//! `LIKE` currently just rescans the pattern every row, so this is no worse than the existing level.

use crate::expr::funcs;
use crate::prelude::*;
use crate::vector::{Bitmap, BytesData, Data, Ty, Vector};

// ============================================================================
// Limits
// ============================================================================

/// The maximum byte length of a pattern string. Anything longer is rejected before compilation.
/// Given the later limits (the instruction count and so on), no meaningful pattern can be
/// written that is longer than this.
const MAX_PATTERN_LEN: usize = 4096;

/// The maximum length of the compiled instruction sequence. Set to 4096, following the
/// "a few thousand instructions" instruction. `{n,m}` expansion and nested repeats abort the
/// moment this value is exceeded, so even a nesting like `(a{1000}){1000}` allocates memory
/// bounded in proportion to this cap (see `Compiler::emit` for details).
const MAX_PROG_INSTS: usize = 4096;

/// The maximum number of capture groups. It sizes the per-thread capture-position array
/// `[u32; SLOTS]` (`SLOTS = (MAX_GROUPS+1)*2`), so raising it makes each step of the NFA
/// simulation heavier. 32 is a count practical patterns essentially never exceed.
const MAX_GROUPS: usize = 32;

/// The size of the capture-position array. Group 0 is reserved for the whole match.
const SLOTS: usize = (MAX_GROUPS + 1) * 2;

/// The maximum nesting depth of parentheses. Both the parser and the compiler are written
/// recursively, so this indirectly bounds the native stack depth (some wasm hosts have a small
/// stack).
const MAX_NEST_DEPTH: u32 = 64;

/// The maximum value of `n`/`m` in `{n,m}`. RE2, which DuckDB uses internally, caps at the same
/// 1000 (confirmed that `duckdb -c "select regexp_matches('a','a{1001}')"` is rejected with
/// `invalid repetition size`), so this matches it.
const MAX_REPEAT: u32 = 1000;

/// The cumulative number of "instruction reached" events allowed in one NFA simulation (one
/// `exec` call = one match attempt). A Thompson NFA's worst case is
/// `O(instructions x input length)`, so multiplied by the 4096 instruction cap, 8,000,000 was
/// chosen as a value allowing both "a pattern at the instruction cap applied to ~2000 bytes of
/// input" and "an ordinary ~20-instruction pattern applied to ~400 KB of input", while capping
/// things so that one match attempt does not exceed tens of milliseconds on wasm under odd input
/// (such as `regexp_replace`'s global `g` replacement calling `exec` repeatedly).
/// Note that it is a cap per `exec` call rather than per batch
/// (limiting the legitimate work of every row would stop ordinary queries from running).
const MAX_STEPS: u32 = 8_000_000;

// ============================================================================
// UTF-8 decoding
// ============================================================================

/// The largest Unicode scalar value.
const MAX_CP: u32 = 0x10_FFFF;

/// Decodes the UTF-8 scalar value starting at `s[i]` and returns `(code point, byte width)`.
///
/// Both the pattern and the subject go through this, so matching always advances by whole
/// characters and can never cut a well-formed sequence in half (which is what used to let
/// `regexp_replace` emit strings that were not valid UTF-8).
///
/// A byte that does not begin a well-formed sequence (only possible on input that was not valid
/// UTF-8 to begin with) is reported as a one-byte scalar equal to the byte itself. That keeps the
/// simulation advancing and keeps every match boundary on a byte the input already had, rather
/// than inventing one.
///
/// Callers must ensure `i < s.len()`.
fn decode_utf8(s: &[u8], i: usize) -> (u32, usize) {
    let b0 = s[i];
    if b0 < 0x80 {
        return (b0 as u32, 1);
    }
    let cont = |k: usize| -> Option<u32> {
        match s.get(i + k) {
            Some(&b) if b & 0xC0 == 0x80 => Some((b & 0x3F) as u32),
            _ => None,
        }
    };
    // 0xC0/0xC1 are excluded: they could only encode an overlong form.
    if (0xC2..0xE0).contains(&b0) {
        if let Some(c1) = cont(1) {
            return ((((b0 & 0x1F) as u32) << 6) | c1, 2);
        }
    } else if (0xE0..0xF0).contains(&b0) {
        if let (Some(c1), Some(c2)) = (cont(1), cont(2)) {
            let cp = (((b0 & 0x0F) as u32) << 12) | (c1 << 6) | c2;
            // Reject overlong forms and the UTF-16 surrogate range.
            if cp >= 0x800 && !(0xD800..0xE000).contains(&cp) {
                return (cp, 3);
            }
        }
    } else if (0xF0..0xF5).contains(&b0) {
        if let (Some(c1), Some(c2), Some(c3)) = (cont(1), cont(2), cont(3)) {
            let cp = (((b0 & 0x07) as u32) << 18) | (c1 << 12) | (c2 << 6) | c3;
            if (0x1_0000..=MAX_CP).contains(&cp) {
                return (cp, 4);
            }
        }
    }
    (b0 as u32, 1)
}

/// The byte width of the character starting at `s[i]` (`decode_utf8` without the code point).
fn char_width(s: &[u8], i: usize) -> usize {
    decode_utf8(s, i).1
}

// ============================================================================
// Character classes
// ============================================================================

/// A set of Unicode scalar values.
///
/// ASCII (which is what patterns overwhelmingly use) is a plain 128-bit mask, and everything above
/// it is a short list of sorted, disjoint, non-adjacent inclusive ranges. Keeping the two halves
/// apart is what makes negation cheap: `[^a]` must cover all 1.1M code points, which no bitmap
/// this crate can afford would.
#[derive(Clone, Default)]
struct ClassSet {
    /// Membership of `0..=0x7F`.
    ascii: [u64; 2],
    /// Membership of `0x80..=MAX_CP`, sorted and merged.
    hi: Vec<(u32, u32)>,
}

impl ClassSet {
    fn new() -> Self {
        ClassSet { ascii: [0; 2], hi: Vec::new() }
    }

    fn set(&mut self, cp: u32) {
        self.set_range(cp, cp);
    }

    fn set_range(&mut self, lo: u32, hi: u32) {
        if lo > hi {
            return;
        }
        let mut b = lo;
        while b <= hi && b < 0x80 {
            self.ascii[(b >> 6) as usize] |= 1u64 << (b & 63);
            b += 1;
        }
        if hi >= 0x80 {
            self.add_hi(lo.max(0x80), hi.min(MAX_CP));
        }
    }

    /// Inserts `lo..=hi` (both `>= 0x80`) into `hi`, merging with any range it touches so the list
    /// stays sorted, disjoint and non-adjacent. Linear rather than binary, since a class holds a
    /// handful of ranges at most.
    fn add_hi(&mut self, lo: u32, hi: u32) {
        if lo > hi {
            return;
        }
        let mut i = 0;
        while i < self.hi.len() && self.hi[i].1 + 1 < lo {
            i += 1;
        }
        let (mut nlo, mut nhi) = (lo, hi);
        while i < self.hi.len() && self.hi[i].0 <= nhi + 1 {
            nlo = nlo.min(self.hi[i].0);
            nhi = nhi.max(self.hi[i].1);
            self.hi.remove(i);
        }
        self.hi.insert(i, (nlo, nhi));
    }

    fn negate(&mut self) {
        self.ascii[0] = !self.ascii[0];
        self.ascii[1] = !self.ascii[1];
        let mut out = Vec::with_capacity(self.hi.len() + 1);
        let mut next = 0x80u32;
        for &(lo, hi) in &self.hi {
            if lo > next {
                out.push((next, lo - 1));
            }
            next = hi + 1;
        }
        if next <= MAX_CP {
            out.push((next, MAX_CP));
        }
        self.hi = out;
    }

    fn union(&mut self, other: &ClassSet) {
        self.ascii[0] |= other.ascii[0];
        self.ascii[1] |= other.ascii[1];
        for &(lo, hi) in &other.hi {
            self.add_hi(lo, hi);
        }
    }

    fn test(&self, cp: u32) -> bool {
        if cp < 0x80 {
            return (self.ascii[(cp >> 6) as usize] >> (cp & 63)) & 1 != 0;
        }
        self.hi.iter().any(|&(lo, hi)| cp >= lo && cp <= hi)
    }
}

fn digit_set() -> ClassSet {
    let mut s = ClassSet::new();
    s.set_range(b'0' as u32, b'9' as u32);
    s
}

fn word_set() -> ClassSet {
    let mut s = digit_set();
    s.set_range(b'a' as u32, b'z' as u32);
    s.set_range(b'A' as u32, b'Z' as u32);
    s.set(b'_' as u32);
    s
}

/// The POSIX bracket-expression classes (`[[:alpha:]]` and friends).
///
/// All of them are ASCII-only, matching RE2 (which is what DuckDB uses) with no Unicode tables:
/// `duckdb -c "select regexp_matches('é','[[:alpha:]]')"` is false. That is also why they cost
/// almost nothing in wasm size, so supporting them beats erroring on them — the previous behavior
/// silently reinterpreted `[[:alpha:]]` as the ordinary set `[:alph]` followed by a literal `]`,
/// which is the one failure mode this module is written to avoid.
fn posix_class(name: &[u8]) -> Option<ClassSet> {
    let mut s = ClassSet::new();
    let alpha = |s: &mut ClassSet| {
        s.set_range(b'a' as u32, b'z' as u32);
        s.set_range(b'A' as u32, b'Z' as u32);
    };
    let digit = |s: &mut ClassSet| s.set_range(b'0' as u32, b'9' as u32);
    let punct = |s: &mut ClassSet| {
        s.set_range(0x21, 0x2F);
        s.set_range(0x3A, 0x40);
        s.set_range(0x5B, 0x60);
        s.set_range(0x7B, 0x7E);
    };
    match name {
        b"alpha" => alpha(&mut s),
        b"digit" => digit(&mut s),
        b"alnum" => {
            alpha(&mut s);
            digit(&mut s);
        }
        b"upper" => s.set_range(b'A' as u32, b'Z' as u32),
        b"lower" => s.set_range(b'a' as u32, b'z' as u32),
        b"space" => s.union(&space_set()),
        b"blank" => {
            s.set(b' ' as u32);
            s.set(b'\t' as u32);
        }
        b"punct" => punct(&mut s),
        b"xdigit" => {
            digit(&mut s);
            s.set_range(b'a' as u32, b'f' as u32);
            s.set_range(b'A' as u32, b'F' as u32);
        }
        b"word" => s.union(&word_set()),
        b"cntrl" => {
            s.set_range(0x00, 0x1F);
            s.set(0x7F);
        }
        b"ascii" => s.set_range(0x00, 0x7F),
        b"graph" => s.set_range(0x21, 0x7E),
        b"print" => s.set_range(0x20, 0x7E),
        _ => return None,
    }
    Some(s)
}

/// RE2's `\s` is `[\t\n\f\r ]` (it does not include `\v`). Confirmed by measuring through DuckDB
/// (`chr(11)` (`\v`) does not match while `chr(12)` (`\f`) does).
fn space_set() -> ClassSet {
    let mut s = ClassSet::new();
    for &b in &[b' ', b'\t', b'\n', 0x0c, b'\r'] {
        s.set(b as u32);
    }
    s
}

// ============================================================================
// The AST and the parser
// ============================================================================

enum Ast {
    Empty,
    /// One literal Unicode scalar value.
    Char(u32),
    Any,
    Class(ClassSet),
    Concat(Vec<Ast>),
    Alt(Vec<Ast>),
    Star(Box<Ast>),
    Plus(Box<Ast>),
    Quest(Box<Ast>),
    Repeat(Box<Ast>, u32, Option<u32>),
    /// `Some(g)` is a capture group (`g` is 1-based); `None` is non-capturing.
    Group(Box<Ast>, Option<u16>),
    Bol,
    Eol,
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
    depth: u32,
    n_groups: u16,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn parse_alt(&mut self) -> Result<Ast> {
        let mut branches = vec![self.parse_concat()?];
        while self.peek() == Some(b'|') {
            self.i += 1;
            branches.push(self.parse_concat()?);
        }
        Ok(if branches.len() == 1 { branches.pop().unwrap() } else { Ast::Alt(branches) })
    }

    fn parse_concat(&mut self) -> Result<Ast> {
        let mut items = Vec::new();
        while let Some(c) = self.peek() {
            if c == b'|' || c == b')' {
                break;
            }
            items.push(self.parse_repeat()?);
        }
        Ok(match items.len() {
            0 => Ast::Empty,
            1 => items.pop().unwrap(),
            _ => Ast::Concat(items),
        })
    }

    /// Reads one atom plus a trailing quantifier (at most one, if present).
    fn parse_repeat(&mut self) -> Result<Ast> {
        let atom = self.parse_atom()?;
        // First, decide only "whether a quantifier applies", without consuming the atom.
        enum Q {
            None,
            Star,
            Plus,
            Quest,
            Repeat(u32, Option<u32>),
        }
        let q = match self.peek() {
            Some(b'*') => {
                self.i += 1;
                Q::Star
            }
            Some(b'+') => {
                self.i += 1;
                Q::Plus
            }
            Some(b'?') => {
                self.i += 1;
                Q::Quest
            }
            Some(b'{') => match self.try_parse_bound() {
                Some((n, m)) => {
                    ensure!(n <= MAX_REPEAT, LimitExceeded);
                    if let Some(mm) = m {
                        ensure!(mm <= MAX_REPEAT, LimitExceeded);
                        ensure!(mm >= n, SyntaxError);
                    }
                    Q::Repeat(n, m)
                }
                // A `{` not forming a valid `{n,m}` is just a literal character (RE2/DuckDB's
                // behavior. `self.i` is rewound to just before the '{', so the next parse_repeat
                // call rereads it as an ordinary atom).
                None => Q::None,
            },
            _ => Q::None,
        };
        if matches!(q, Q::None) {
            return Ok(atom);
        }
        // Another quantifier immediately after a quantifier is not allowed. Lazy notations such as
        // `*?` are explicitly rejected as unsupported (silently treating them as greedy would give
        // "wrong results", which is the one thing to avoid at all costs).
        match self.peek() {
            Some(b'?') => err!(UnsupportedFeature),
            Some(b'*') | Some(b'+') | Some(b'{') => err!(SyntaxError),
            _ => {}
        }
        Ok(match q {
            Q::None => unreachable!(),
            Q::Star => Ast::Star(Box::new(atom)),
            Q::Plus => Ast::Plus(Box::new(atom)),
            Q::Quest => Ast::Quest(Box::new(atom)),
            Q::Repeat(n, m) => Ast::Repeat(Box::new(atom), n, m),
        })
    }

    /// Reads `{n}` / `{n,}` / `{n,m}`. On a different shape it rewinds `self.i` and gives `None`.
    fn try_parse_bound(&mut self) -> Option<(u32, Option<u32>)> {
        let start = self.i;
        self.i += 1; // '{'
        let n = self.scan_u32()?;
        let bound = if self.peek() == Some(b',') {
            self.i += 1;
            if self.peek() == Some(b'}') {
                Some((n, None))
            } else {
                let m = self.scan_u32()?;
                Some((n, Some(m)))
            }
        } else {
            Some((n, Some(n)))
        };
        if self.peek() == Some(b'}') && bound.is_some() {
            self.i += 1;
            bound
        } else {
            self.i = start;
            None
        }
    }

    fn scan_u32(&mut self) -> Option<u32> {
        let start = self.i;
        let mut v: u32 = 0;
        while let Some(c) = self.peek() {
            if !c.is_ascii_digit() {
                break;
            }
            // Saturate early at a value well above MAX_REPEAT (avoiding overflow; the later
            // ensure! makes it LimitExceeded anyway).
            v = v.saturating_mul(10).saturating_add((c - b'0') as u32);
            self.i += 1;
        }
        if self.i == start {
            None
        } else {
            Some(v)
        }
    }

    fn parse_atom(&mut self) -> Result<Ast> {
        let c = match self.peek() {
            Some(c) => c,
            None => err!(SyntaxError),
        };
        match c {
            b'*' | b'+' | b'?' => err!(SyntaxError), // a quantifier with nothing to apply to
            b'.' => {
                self.i += 1;
                Ok(Ast::Any)
            }
            b'^' => {
                self.i += 1;
                Ok(Ast::Bol)
            }
            b'$' => {
                self.i += 1;
                Ok(Ast::Eol)
            }
            b'[' => self.parse_class(),
            b'(' => self.parse_group(),
            b'\\' => self.parse_escape(),
            // '{' is always a literal here (a valid {n,m} is handled earlier by parse_repeat.
            // Reaching here means a `{` at the head of an atom, with nothing to apply to, so
            // treating it as a literal is always right).
            _ => {
                // A multi-byte character is one atom, so that a quantifier after it applies to the
                // whole character rather than to its last byte.
                let (cp, w) = decode_utf8(self.s, self.i);
                self.i += w;
                Ok(Ast::Char(cp))
            }
        }
    }

    fn parse_group(&mut self) -> Result<Ast> {
        self.i += 1; // '('
        let capturing = if self.peek() == Some(b'?') {
            if self.s.get(self.i + 1) == Some(&b':') {
                self.i += 2;
                false
            } else {
                // (?=  (?!  (?<  (?P  and every other extension besides non-capturing are unsupported.
                err!(UnsupportedFeature);
            }
        } else {
            true
        };
        let g = if capturing {
            ensure!((self.n_groups as usize) < MAX_GROUPS, LimitExceeded);
            self.n_groups += 1;
            Some(self.n_groups)
        } else {
            None
        };
        self.depth += 1;
        ensure!(self.depth <= MAX_NEST_DEPTH, LimitExceeded);
        let inner = self.parse_alt()?;
        self.depth -= 1;
        ensure!(self.peek() == Some(b')'), SyntaxError);
        self.i += 1;
        Ok(Ast::Group(Box::new(inner), g))
    }

    fn parse_escape(&mut self) -> Result<Ast> {
        self.i += 1; // '\'
        let c = match self.peek() {
            Some(c) => c,
            None => err!(SyntaxError),
        };
        self.i += 1;
        Ok(match c {
            b'.' | b'*' | b'+' | b'?' | b'(' | b')' | b'[' | b']' | b'{' | b'}' | b'|' | b'^'
            | b'$' | b'\\' => Ast::Char(c as u32),
            b'd' => Ast::Class(digit_set()),
            b'D' => {
                let mut s = digit_set();
                s.negate();
                Ast::Class(s)
            }
            b'w' => Ast::Class(word_set()),
            b'W' => {
                let mut s = word_set();
                s.negate();
                Ast::Class(s)
            }
            b's' => Ast::Class(space_set()),
            b'S' => {
                let mut s = space_set();
                s.negate();
                Ast::Class(s)
            }
            // In-pattern backreferences and word boundaries are unsupported (see the top of the module).
            b'1'..=b'9' | b'b' | b'B' => err!(UnsupportedFeature),
            _ => err!(SyntaxError),
        })
    }

    /// Reads a POSIX bracket expression `[:name:]` (or the negated `[:^name:]`) at `self.i`,
    /// which must point at the `[`.
    ///
    /// `Ok(None)` means "this is not a bracket expression after all" (no closing `:]` before the
    /// end of the pattern), in which case `self.i` is untouched and the caller keeps treating the
    /// `[` as a literal. A well-formed bracket expression naming a class this engine does not know
    /// is an `UnsupportedFeature` error rather than a silent reinterpretation.
    fn try_parse_posix(&mut self) -> Result<Option<ClassSet>> {
        let start = self.i + 2; // past "[:"
        let mut j = start;
        while j + 1 < self.s.len() && !(self.s[j] == b':' && self.s[j + 1] == b']') {
            j += 1;
        }
        if j + 1 >= self.s.len() {
            return Ok(None);
        }
        let mut name = &self.s[start..j];
        let neg = name.first() == Some(&b'^');
        if neg {
            name = &name[1..];
        }
        match posix_class(name) {
            Some(mut set) => {
                if neg {
                    set.negate();
                }
                self.i = j + 2; // past ":]"
                Ok(Some(set))
            }
            None => err!(UnsupportedFeature),
        }
    }

    fn parse_class(&mut self) -> Result<Ast> {
        self.i += 1; // '['
        let negate = self.peek() == Some(b'^');
        if negate {
            self.i += 1;
        }
        let mut set = ClassSet::new();
        let mut first = true;
        loop {
            match self.peek() {
                None => err!(SyntaxError),
                Some(b']') if !first => {
                    self.i += 1;
                    break;
                }
                // A POSIX bracket expression `[:name:]` / `[:^name:]`. It is only a class when a
                // closing `:]` is actually present; otherwise the `[` stays an ordinary literal
                // member (`[[]` is still a class containing `[`).
                Some(b'[') if self.s.get(self.i + 1) == Some(&b':') => {
                    match self.try_parse_posix()? {
                        Some(t) => set.union(&t),
                        None => {
                            self.i += 1;
                            set.set(b'[' as u32);
                        }
                    }
                    first = false;
                }
                Some(b'\\') => {
                    self.i += 1;
                    let c = match self.peek() {
                        Some(c) => c,
                        None => err!(SyntaxError),
                    };
                    if c >= 0x80 {
                        // An escaped multi-byte character is a literal member of the class.
                        let (cp, w) = decode_utf8(self.s, self.i);
                        self.i += w;
                        set.set(cp);
                        first = false;
                        continue;
                    }
                    self.i += 1;
                    match c {
                        b'd' => set.union(&digit_set()),
                        b'D' => {
                            let mut t = digit_set();
                            t.negate();
                            set.union(&t);
                        }
                        b'w' => set.union(&word_set()),
                        b'W' => {
                            let mut t = word_set();
                            t.negate();
                            set.union(&t);
                        }
                        b's' => set.union(&space_set()),
                        b'S' => {
                            let mut t = space_set();
                            t.negate();
                            set.union(&t);
                        }
                        // Inside a class, escaping `]` `\` `^` `-` and so on is a practical need,
                        // so other characters are allowed as literals too (separate from the strict
                        // escape rules outside a class).
                        _ => set.set(c as u32),
                    }
                    first = false;
                }
                Some(_) => {
                    // Members are whole characters, so `[^é]` excludes the character rather than
                    // each of its bytes, and `[à-ÿ]` is a code-point range.
                    let (lo, w) = decode_utf8(self.s, self.i);
                    self.i += w;
                    first = false;
                    // A range `a-z`. A trailing `-` (with `]` next) or a leading `-` is treated as a
                    // literal.
                    if self.peek() == Some(b'-')
                        && self.s.get(self.i + 1).is_some()
                        && self.s.get(self.i + 1) != Some(&b']')
                    {
                        self.i += 1;
                        ensure!(self.s[self.i] != b'\\', SyntaxError);
                        let (hi, hw) = decode_utf8(self.s, self.i);
                        self.i += hw;
                        ensure!(lo <= hi, SyntaxError);
                        set.set_range(lo, hi);
                    } else {
                        set.set(lo);
                    }
                }
            }
        }
        if negate {
            set.negate();
        }
        Ok(Ast::Class(set))
    }
}

// ============================================================================
// Compilation (AST -> NFA instruction sequence)
// ============================================================================

#[derive(Clone, Copy)]
enum Inst {
    /// Consumes exactly the given Unicode scalar value (however many bytes it takes).
    Char(u32),
    Any,
    Class(u16),
    Split(u32, u32),
    Jmp(u32),
    Save(u8),
    Bol,
    Eol,
    Match,
}

pub struct Program {
    insts: Vec<Inst>,
    classes: Vec<ClassSet>,
    n_groups: u16,
}

struct Compiler {
    prog: Vec<Inst>,
    classes: Vec<ClassSet>,
}

impl Compiler {
    fn emit(&mut self, inst: Inst) -> Result<u32> {
        // This is the sole point of growth. `{n,m}` and nested-repeat expansion all pass through
        // here via recursive `compile` calls, so however deeply a pattern nests, the instructions
        // actually allocated are capped here (the design never duplicates and inflates the AST
        // itself).
        ensure!(self.prog.len() < MAX_PROG_INSTS, LimitExceeded);
        self.prog.push(inst);
        Ok((self.prog.len() - 1) as u32)
    }

    fn compile(&mut self, ast: &Ast) -> Result<()> {
        match ast {
            Ast::Empty => Ok(()),
            Ast::Char(c) => {
                self.emit(Inst::Char(*c))?;
                Ok(())
            }
            Ast::Any => {
                self.emit(Inst::Any)?;
                Ok(())
            }
            Ast::Class(set) => {
                let idx = self.classes.len() as u16;
                self.classes.push(set.clone());
                self.emit(Inst::Class(idx))?;
                Ok(())
            }
            Ast::Bol => {
                self.emit(Inst::Bol)?;
                Ok(())
            }
            Ast::Eol => {
                self.emit(Inst::Eol)?;
                Ok(())
            }
            Ast::Concat(items) => {
                for it in items {
                    self.compile(it)?;
                }
                Ok(())
            }
            Ast::Alt(branches) => self.compile_alt(branches),
            Ast::Star(x) => self.compile_star(x),
            Ast::Plus(x) => self.compile_plus(x),
            Ast::Quest(x) => self.compile_quest(x),
            Ast::Repeat(x, n, m) => self.compile_repeat(x, *n, *m),
            Ast::Group(x, Some(g)) => {
                self.emit(Inst::Save(2 * (*g as u8)))?;
                self.compile(x)?;
                self.emit(Inst::Save(2 * (*g as u8) + 1))?;
                Ok(())
            }
            Ast::Group(x, None) => self.compile(x),
        }
    }

    fn compile_alt(&mut self, branches: &[Ast]) -> Result<()> {
        if branches.len() == 1 {
            return self.compile(&branches[0]);
        }
        let split_pc = self.emit(Inst::Split(0, 0))?;
        self.compile(&branches[0])?;
        let jmp_pc = self.emit(Inst::Jmp(0))?;
        let l2 = self.prog.len() as u32;
        self.compile_alt(&branches[1..])?;
        let end = self.prog.len() as u32;
        self.prog[split_pc as usize] = Inst::Split(split_pc + 1, l2);
        self.prog[jmp_pc as usize] = Inst::Jmp(end);
        Ok(())
    }

    /// Greedy `x*`. A `Split` with "continue" first and "exit" second expresses greedy priority.
    fn compile_star(&mut self, x: &Ast) -> Result<()> {
        let l1 = self.emit(Inst::Split(0, 0))?;
        self.compile(x)?;
        self.emit(Inst::Jmp(l1))?;
        let end = self.prog.len() as u32;
        self.prog[l1 as usize] = Inst::Split(l1 + 1, end);
        Ok(())
    }

    fn compile_plus(&mut self, x: &Ast) -> Result<()> {
        let l1 = self.prog.len() as u32;
        self.compile(x)?;
        let split_pc = self.emit(Inst::Split(0, 0))?;
        self.prog[split_pc as usize] = Inst::Split(l1, split_pc + 1);
        Ok(())
    }

    fn compile_quest(&mut self, x: &Ast) -> Result<()> {
        let l1 = self.emit(Inst::Split(0, 0))?;
        self.compile(x)?;
        let end = self.prog.len() as u32;
        self.prog[l1 as usize] = Inst::Split(l1 + 1, end);
        Ok(())
    }

    /// `{n,m}` is expressed as `x` repeated `n` times followed by `x?` repeated `m-n` times
    /// (`{n,}` omits `m` and continues with `x*`).
    /// Each `x?` is an independent Split, so greedy priority is preserved naturally.
    /// Rather than duplicating and expanding the AST, `compile` is simply called repeatedly on a
    /// reference to the same `x`, so the expanded size shows up only as the instruction count
    /// (which `emit` counts) and never balloons in memory beforehand, even when nested.
    fn compile_repeat(&mut self, x: &Ast, n: u32, m: Option<u32>) -> Result<()> {
        for _ in 0..n {
            self.compile(x)?;
        }
        match m {
            None => self.compile_star(x),
            Some(mm) => {
                for _ in 0..(mm - n) {
                    self.compile_quest(x)?;
                }
                Ok(())
            }
        }
    }
}

/// Compiles a pattern string. The caller (`funcs.rs`) calls this once outside the row loop when
/// the pattern column's stride is 0 (a constant column).
pub fn compile(pattern: &[u8]) -> Result<Program> {
    ensure!(pattern.len() <= MAX_PATTERN_LEN, LimitExceeded);
    let mut p = Parser { s: pattern, i: 0, depth: 0, n_groups: 0 };
    let ast = p.parse_alt()?;
    // A leftover remainder means a `)` with no matching `(` was mixed in.
    ensure!(p.i == pattern.len(), SyntaxError);
    let mut c = Compiler { prog: Vec::new(), classes: Vec::new() };
    c.emit(Inst::Save(0))?;
    c.compile(&ast)?;
    c.emit(Inst::Save(1))?;
    c.emit(Inst::Match)?;
    Ok(Program { insts: c.prog, classes: c.classes, n_groups: p.n_groups })
}

// ============================================================================
// Simulation with Pike's VM
// ============================================================================

#[derive(Clone, Copy)]
struct Thread {
    pc: u32,
    saves: [u32; SLOTS],
}

/// Follows the epsilon transitions (`Split`/`Jmp`/`Save`/`Bol`/`Eol`) from `pc` and pushes onto
/// `list` the instructions that consume a byte (`Char`/`Any`/`Class`) and `Match` (the NFA's
/// epsilon closure).
///
/// `gen`/`gen_id` remember "whether this instruction number was already added at this position".
/// That prevents the same instruction number from being pushed twice in one step, so however
/// deeply `Split`s nest, one step's cost is capped by the instruction count (= no exponential
/// revisiting as in backtracking).
///
/// It uses an explicit stack rather than recursion, because following `Split` branches
/// recursively could consume native stack proportional to the instruction count depending on the
/// pattern (some wasm hosts have a small stack).
#[allow(clippy::too_many_arguments)]
fn add_thread(
    prog: &Program,
    list: &mut Vec<Thread>,
    gen: &mut [u32],
    gen_id: u32,
    pc0: u32,
    saves0: [u32; SLOTS],
    sp: usize,
    input_len: usize,
    steps: &mut u32,
) -> Result<()> {
    let mut stack: Vec<(u32, [u32; SLOTS])> = vec![(pc0, saves0)];
    while let Some((pc, mut saves)) = stack.pop() {
        if gen[pc as usize] == gen_id {
            continue;
        }
        gen[pc as usize] = gen_id;
        *steps += 1;
        ensure!(*steps <= MAX_STEPS, LimitExceeded);
        match prog.insts[pc as usize] {
            Inst::Jmp(x) => stack.push((x, saves)),
            Inst::Split(a, b) => {
                // Greedy priority: the list is LIFO, so what is pushed first is processed later;
                // the higher-priority branch `a` is pushed last so it is taken first.
                stack.push((b, saves));
                stack.push((a, saves));
            }
            Inst::Save(slot) => {
                if (slot as usize) < SLOTS {
                    saves[slot as usize] = sp as u32;
                }
                stack.push((pc + 1, saves));
            }
            Inst::Bol => {
                if sp == 0 {
                    stack.push((pc + 1, saves));
                }
            }
            Inst::Eol => {
                if sp == input_len {
                    stack.push((pc + 1, saves));
                }
            }
            Inst::Char(_) | Inst::Any | Inst::Class(_) | Inst::Match => {
                list.push(Thread { pc, saves });
            }
        }
    }
    Ok(())
}

/// Finds the leftmost match in the input. On success it returns the capture positions (group 0 =
/// the whole match; the slot value `u32::MAX` means "not captured").
pub fn find(prog: &Program, input: &[u8]) -> Result<Option<[u32; SLOTS]>> {
    find_from(prog, input, 0)
}

/// Finds the leftmost match that starts at or after `start`, reported in **absolute** offsets into
/// `input`.
///
/// Callers that scan a string repeatedly (`replace_into`'s global mode) must use this rather than
/// re-running `find` on `&input[pos..]`: the sub-slice would make `^` match at every restart
/// position, since `Inst::Bol` tests `sp == 0`. Passing the whole string plus a start cursor keeps
/// both `^` and `$` measured against the real ends of the string.
///
/// This is a "search" that shifts the start position one character at a time, not an anchored
/// match. Rather than running a dedicated loop, it simply keeps adding a new starting thread at
/// **lower priority than the existing threads** while no match has been found (the standard Pike's
/// VM technique). That automatically guarantees "the first one found = the leftmost".
pub fn find_from(prog: &Program, input: &[u8], start: usize) -> Result<Option<[u32; SLOTS]>> {
    let mut gen = vec![0u32; prog.insts.len()];
    let mut gen_id: u32 = 0;
    let mut steps: u32 = 0;
    let mut clist: Vec<Thread> = Vec::new();
    let mut nlist: Vec<Thread> = Vec::new();
    let mut matched: Option<[u32; SLOTS]> = None;
    let len = input.len();
    if start > len {
        return Ok(None);
    }

    gen_id += 1;
    add_thread(prog, &mut clist, &mut gen, gen_id, 0, [u32::MAX; SLOTS], start, len, &mut steps)?;

    let mut sp = start;
    loop {
        if clist.is_empty() && matched.is_some() {
            break;
        }
        // One step of the simulation consumes one whole UTF-8 scalar value, never one byte, so a
        // multi-byte character can neither be matched piecewise by `.` nor be split across a match
        // boundary (which would let the caller slice out invalid UTF-8).
        let (ch, next_sp) = if sp < len {
            let (cp, w) = decode_utf8(input, sp);
            (Some(cp), sp + w)
        } else {
            (None, sp)
        };
        gen_id += 1;
        let mut idx = 0;
        while idx < clist.len() {
            let th = clist[idx];
            match prog.insts[th.pc as usize] {
                Inst::Char(c) => {
                    if ch == Some(c) {
                        add_thread(
                            prog,
                            &mut nlist,
                            &mut gen,
                            gen_id,
                            th.pc + 1,
                            th.saves,
                            next_sp,
                            len,
                            &mut steps,
                        )?;
                    }
                }
                Inst::Any => {
                    if let Some(c) = ch {
                        if c != b'\n' as u32 {
                            add_thread(
                                prog,
                                &mut nlist,
                                &mut gen,
                                gen_id,
                                th.pc + 1,
                                th.saves,
                                next_sp,
                                len,
                                &mut steps,
                            )?;
                        }
                    }
                }
                Inst::Class(ci) => {
                    if let Some(c) = ch {
                        if prog.classes[ci as usize].test(c) {
                            add_thread(
                                prog,
                                &mut nlist,
                                &mut gen,
                                gen_id,
                                th.pc + 1,
                                th.saves,
                                next_sp,
                                len,
                                &mut steps,
                            )?;
                        }
                    }
                }
                Inst::Match => {
                    // Threads of lower priority (further back in clist) can no longer be chosen, so
                    // it breaks off. Higher-priority threads were already pushed into nlist before
                    // this loop.
                    matched = Some(th.saves);
                    break;
                }
                Inst::Jmp(_) | Inst::Split(_, _) | Inst::Save(_) | Inst::Bol | Inst::Eol => {
                    // Epsilon instructions are already expanded by add_thread and never appear as threads.
                }
            }
            idx += 1;
        }
        if ch.is_some() && matched.is_none() {
            add_thread(
                prog,
                &mut nlist,
                &mut gen,
                gen_id,
                0,
                [u32::MAX; SLOTS],
                next_sp,
                len,
                &mut steps,
            )?;
        }
        core::mem::swap(&mut clist, &mut nlist);
        nlist.clear();
        if ch.is_none() {
            break;
        }
        sp = next_sp;
    }
    Ok(matched)
}

pub fn is_match(prog: &Program, input: &[u8]) -> Result<bool> {
    Ok(find(prog, input)?.is_some())
}

// ============================================================================
// The SQL function bodies (per Vector)
// ============================================================================

fn geti(v: &Vector, i: usize) -> i64 {
    match v.data() {
        Data::I64(d) => d.get(i).copied().unwrap_or(0),
        Data::I32(d) => d.get(i).copied().unwrap_or(0) as i64,
        _ => 0,
    }
}

/// `regexp_matches(str, pattern)`.
pub fn eval_matches(args: &[&Vector]) -> Result<Vector> {
    ensure!(args.len() == 2, WrongArgCount);
    let (n, s) = funcs::strides(args)?;
    let valid = funcs::combine(args, &s, n);
    let live = |i: usize| valid.as_ref().is_none_or(|b| b.get(i));
    let (sv, pv) = (args[0], args[1]);
    let pat_const = s[1] == 0;
    let cached =
        if pat_const && n > 0 && pv.is_valid(0) { Some(compile(pv.bytes().get(0))?) } else { None };
    let mut bits = Bitmap::with_capacity(n);
    for i in 0..n {
        if !live(i) {
            bits.push(false);
            continue;
        }
        let compiled;
        let prog: &Program = match &cached {
            Some(p) => p,
            None => {
                compiled = compile(pv.bytes().get(i * s[1]))?;
                &compiled
            }
        };
        bits.push(is_match(prog, sv.bytes().get(i * s[0]))?);
    }
    let mut out = Vector::from_data(Ty::Boolean, Data::Bool(bits), valid);
    out.compact_validity();
    Ok(out)
}

/// `regexp_extract(str, pattern[, group])`.
///
/// DuckDB returns **the empty string rather than NULL** for all of "no match", "the group was not
/// captured in that match", and "the group number exceeds the pattern's capture count (but stays
/// within 0-9)" (measured with `duckdb -c "select regexp_extract('xxx','foo')"` and the like).
/// It becomes NULL only when the str/pattern/group argument itself is NULL.
pub fn eval_extract(args: &[&Vector]) -> Result<Vector> {
    ensure!(args.len() == 2 || args.len() == 3, WrongArgCount);
    let (n, s) = funcs::strides(args)?;
    let valid = funcs::combine(args, &s, n);
    let live = |i: usize| valid.as_ref().is_none_or(|b| b.get(i));
    let (sv, pv) = (args[0], args[1]);
    let pat_const = s[1] == 0;
    let cached =
        if pat_const && n > 0 && pv.is_valid(0) { Some(compile(pv.bytes().get(0))?) } else { None };
    let mut out = BytesData::with_capacity(n, n * 8);
    for i in 0..n {
        if !live(i) {
            out.push_empty();
            continue;
        }
        let compiled;
        let prog: &Program = match &cached {
            Some(p) => p,
            None => {
                compiled = compile(pv.bytes().get(i * s[1]))?;
                &compiled
            }
        };
        let g = if args.len() == 3 { geti(args[2], i * s[2]) } else { 0 };
        // Measured in DuckDB: group is allowed only in 0-9, and out of range is an error for the
        // query itself rather than for the row (`select regexp_extract('a','a',10)` is an
        // Invalid Input Error).
        ensure!((0..=9).contains(&g), ValueOutOfRange);
        let text = sv.bytes().get(i * s[0]);
        match find(prog, text)? {
            None => out.push_empty(),
            Some(saves) => {
                if g as u16 > prog.n_groups {
                    out.push_empty();
                } else {
                    let (st, en) = (saves[2 * g as usize], saves[2 * g as usize + 1]);
                    if st == u32::MAX || en == u32::MAX {
                        out.push_empty();
                    } else {
                        out.push(&text[st as usize..en as usize]);
                    }
                }
            }
        }
    }
    let mut v = Vector::from_data(Ty::Varchar, Data::Bytes(out), valid);
    v.compact_validity();
    Ok(v)
}

/// A token of the replacement string. `\0`-`\9` are group references, `\\` is a literal `\`, and
/// any other `\X` is invalid.
enum ReplTok {
    Lit(u8),
    Group(u8),
}

/// Parses the replacement string. Measured against DuckDB (RE2 internally): when the replacement
/// string is invalid (it ends with `\`, or `\` is followed by neither a digit nor `\`), or when a
/// referenced group number exceeds the pattern's actual capture count, **it is not an error; that
/// row's result is simply the input string** (no replacement happens at all).
/// Confirmed by `select regexp_replace('abc','b','X\9')` returning `'abc'` and
/// `select regexp_replace('a','(a)','X\2Y')` returning `'a'`, among others. Here that "invalid
/// gives `None`" maps directly onto `replace_into`'s "invalid passes everything through".
///
/// The second element of the return value is the largest group number referenced (the comparison
/// against `prog.n_groups` is the caller's job, since the pattern is unknown here).
fn parse_repl(repl: &[u8]) -> Option<(Vec<ReplTok>, u8)> {
    let mut toks = Vec::with_capacity(repl.len());
    let mut max_g: u8 = 0;
    let mut i = 0usize;
    while i < repl.len() {
        let c = repl[i];
        if c == b'\\' {
            let d = *repl.get(i + 1)?;
            if d == b'\\' {
                toks.push(ReplTok::Lit(b'\\'));
            } else if d.is_ascii_digit() {
                let g = d - b'0';
                toks.push(ReplTok::Group(g));
                if g > max_g {
                    max_g = g;
                }
            } else {
                return None;
            }
            i += 2;
        } else {
            toks.push(ReplTok::Lit(c));
            i += 1;
        }
    }
    Some((toks, max_g))
}

/// Builds one output column name for `COLUMNS(...) AS '<template>'`
/// (capture-group renaming in DuckDB's star expressions. Called from `plan::bind`).
///
/// Semantics below are all verified against `duckdb` v1.4.4:
///
/// - `\0` is the **whole column name**, not the matched substring
///   (`SELECT COLUMNS('u') AS 'x_\0'` over a `num` column yields `x_num`,
///   not `x_u`).
/// - `\1`..`\9` are the pattern's capture groups. `saves` is the slot array
///   `find` returned for this column name, or `None` for the
///   `COLUMNS(*)`/`COLUMNS([...])` forms, which have no pattern at all. A
///   group that did not participate in the match — including a reference
///   past the pattern's group count — expands to the empty string
///   (`COLUMNS('(n)(a)?.*') AS 'x\2'` yields `x` and `xa`).
/// - If the template is malformed, or if the whole expansion comes out
///   empty, the column keeps its original name (`COLUMNS('n.*') AS '\1'`
///   with a group-less pattern leaves `num`/`name` untouched, while
///   `AS 'q\1'` renames both to `q`). The malformed case follows the same
///   "invalid replacement means no substitution" rule `regexp_replace`
///   already uses — see `parse_repl`.
pub fn expand_name_template(name: &[u8], saves: Option<&[u32]>, template: &[u8]) -> Vec<u8> {
    let Some((toks, _)) = parse_repl(template) else { return name.to_vec() };
    let mut out = Vec::with_capacity(template.len() + name.len());
    for t in &toks {
        match t {
            ReplTok::Lit(b) => out.push(*b),
            ReplTok::Group(0) => out.extend_from_slice(name),
            ReplTok::Group(g) => {
                let slot = |k: usize| saves.and_then(|s| s.get(k).copied());
                if let (Some(st), Some(en)) = (slot(2 * *g as usize), slot(2 * *g as usize + 1)) {
                    if st != u32::MAX && en as usize <= name.len() && st <= en {
                        out.extend_from_slice(&name[st as usize..en as usize]);
                    }
                }
            }
        }
    }
    if out.is_empty() {
        return name.to_vec();
    }
    out
}

fn emit_repl(toks: &[ReplTok], text: &[u8], saves: &[u32; SLOTS], out: &mut Vec<u8>) {
    for t in toks {
        match t {
            ReplTok::Lit(b) => out.push(*b),
            ReplTok::Group(g) => {
                let (st, en) = (saves[2 * *g as usize], saves[2 * *g as usize + 1]);
                // A group that is declared but was not captured in this match (for example group 2
                // when `(a)|(b)` matched on the `a` side) counts as the empty string (measured in
                // DuckDB: `select regexp_replace('a', '(a)|(b)', 'X\2Y')` gives `'XY'`).
                if st != u32::MAX && en != u32::MAX {
                    out.extend_from_slice(&text[st as usize..en as usize]);
                }
            }
        }
    }
}

/// The body of `regexp_replace`. With `global=false` only the first match is replaced; with
/// `true`, every match.
///
/// When handling a pattern that can match the empty string (such as `'x*'`) under global
/// replacement, an empty match starting at the same position as the previously accepted match's
/// end is ignored, and one byte is emitted as is before advancing. Without that,
/// `regexp_replace(.., 'X*', '-', 'g')` on `'aXbXc'` would give `-a--b--c-`, whereas DuckDB
/// measures `-a-b-c-` (the same "an empty match adjacent to the previous match is not accepted"
/// rule as Python's `re.sub` and others), so this matches that.
fn replace_into(prog: &Program, text: &[u8], repl: &[u8], global: bool, out: &mut Vec<u8>) {
    let toks = match parse_repl(repl) {
        Some((t, mg)) if mg as u16 <= prog.n_groups => t,
        _ => {
            out.extend_from_slice(text);
            return;
        }
    };
    if !global {
        match find(prog, text) {
            Ok(Some(saves)) => {
                let (mstart, mend) = (saves[0] as usize, saves[1] as usize);
                out.extend_from_slice(&text[..mstart]);
                emit_repl(&toks, text, &saves, out);
                out.extend_from_slice(&text[mend..]);
            }
            _ => out.extend_from_slice(text),
        }
        return;
    }
    let len = text.len();
    let mut pos = 0usize;
    let mut prev_end: Option<usize> = None;
    while pos <= len {
        // The whole string is searched from a cursor rather than `&text[pos..]`, so that `^` keeps
        // meaning "the start of the string" instead of "the start of whatever is left".
        let saves = match find_from(prog, text, pos) {
            Ok(Some(sv)) => sv,
            _ => break,
        };
        let mstart = saves[0] as usize;
        let mend = saves[1] as usize;
        // Advancing past a zero-length match steps one whole character, so a multi-byte character
        // is copied through intact.
        let skip_one = |out: &mut Vec<u8>, at: usize| -> usize {
            if at < len {
                let w = char_width(text, at);
                out.extend_from_slice(&text[at..at + w]);
                at + w
            } else {
                at + 1
            }
        };
        if mstart == mend && Some(mstart) == prev_end {
            pos = skip_one(out, mstart);
            continue;
        }
        out.extend_from_slice(&text[pos..mstart]);
        emit_repl(&toks, text, &saves, out);
        pos = if mend > mstart { mend } else { skip_one(out, mend) };
        prev_end = Some(mend);
    }
    out.extend_from_slice(&text[pos.min(len)..]);
}

/// `regexp_replace(str, pattern, replacement[, 'g'])`.
pub fn eval_replace(args: &[&Vector]) -> Result<Vector> {
    ensure!(args.len() == 3 || args.len() == 4, WrongArgCount);
    let (n, s) = funcs::strides(args)?;
    let valid = funcs::combine(args, &s, n);
    let live = |i: usize| valid.as_ref().is_none_or(|b| b.get(i));
    let (sv, pv, rv) = (args[0], args[1], args[2]);
    let pat_const = s[1] == 0;
    let cached =
        if pat_const && n > 0 && pv.is_valid(0) { Some(compile(pv.bytes().get(0))?) } else { None };
    let mut out = BytesData::with_capacity(n, n * 8);
    let mut buf = Vec::new();
    for i in 0..n {
        if !live(i) {
            out.push_empty();
            continue;
        }
        let compiled;
        let prog: &Program = match &cached {
            Some(p) => p,
            None => {
                compiled = compile(pv.bytes().get(i * s[1]))?;
                &compiled
            }
        };
        let global = if args.len() == 4 {
            let f = args[3].bytes().get(i * s[3]);
            ensure!(f.is_empty() || f == b"g", UnsupportedFeature);
            f == b"g"
        } else {
            false
        };
        buf.clear();
        replace_into(prog, sv.bytes().get(i * s[0]), rv.bytes().get(i * s[2]), global, &mut buf);
        out.push(&buf);
    }
    let mut v = Vector::from_data(Ty::Varchar, Data::Bytes(out), valid);
    v.compact_validity();
    Ok(v)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::code_of;
    use crate::vector::Value;
    use std::time::Instant;

    fn m(s: &str, p: &str) -> bool {
        is_match(&compile(p.as_bytes()).unwrap(), s.as_bytes()).unwrap()
    }

    fn ext(s: &str, p: &str, g: u16) -> Option<String> {
        let prog = compile(p.as_bytes()).unwrap();
        let text = s.as_bytes();
        find(&prog, text).unwrap().map(|saves| {
            let (st, en) = (saves[2 * g as usize], saves[2 * g as usize + 1]);
            if st == u32::MAX || en == u32::MAX {
                String::new()
            } else {
                String::from_utf8(text[st as usize..en as usize].to_vec()).unwrap()
            }
        })
    }

    // --- Individual syntax ---------------------------------------------------

    #[test]
    fn literal_and_any() {
        assert!(m("foobar", "foo"));
        assert!(!m("foo", "bar"));
        assert!(m("a", "."));
        assert!(!m("\n", "."));
        assert!(m("axb", "a.b"));
    }

    #[test]
    fn quantifiers() {
        assert!(m("", "a*"));
        assert!(m("aaa", "a*"));
        assert!(!m("", "a+"));
        assert!(m("a", "a+"));
        assert!(m("", "a?"));
        assert!(m("a", "a?"));
        assert!(!m("aa", "^a?$"));
    }

    #[test]
    fn bounded_repeat() {
        assert!(m("aaa", "a{3}"));
        assert!(!m("aa", "^a{3}$"));
        assert!(m("aaaa", "^a{2,}$"));
        assert!(!m("a", "^a{2,}$"));
        assert!(m("aa", "^a{1,3}$"));
        assert!(m("aaa", "^a{1,3}$"));
        assert!(!m("aaaa", "^a{1,3}$"));
        assert!(m("b", "a{0}b"));
    }

    #[test]
    fn char_classes() {
        assert!(m("b", "[abc]"));
        assert!(!m("d", "[abc]"));
        assert!(m("d", "[^abc]"));
        assert!(m("m", "[a-z]"));
        assert!(!m("M", "[a-z]"));
        assert!(m("a.b", "a[.]b"));
        assert!(m("a-b", "[a-b-]"));
        assert!(m("-", "[-ab]"));
    }

    #[test]
    fn shorthand_classes() {
        assert!(m("5", "\\d"));
        assert!(!m("x", "^\\d$"));
        assert!(m("x", "\\D"));
        assert!(m("_", "\\w"));
        assert!(m(" ", "\\s"));
        assert!(!m("\u{0b}", "^\\s$")); // RE2's \s does not include \v (measured)
        assert!(m("\u{0c}", "^\\s$"));
        assert!(m("!", "\\S"));
        assert!(m("3.14", "\\d+\\.\\d+"));
    }

    #[test]
    fn anchors() {
        assert!(m("bar", "^bar"));
        assert!(!m("foobar", "^bar"));
        assert!(m("ab", "ab$"));
        assert!(!m("\n", "^$")); // $ matches only the end of the input (not before a newline)
        assert!(m("", "^$"));
    }

    // --- UTF-8 awareness (cross-checked with DuckDB) -------------------------

    #[test]
    fn dot_and_classes_consume_whole_characters() {
        // duckdb: regexp_matches('héllo','h.llo') = true, regexp_full_match('héllo','h..llo') = false
        assert!(m("héllo", "h.llo"));
        assert!(!m("héllo", "^h..llo$"));
        assert!(m("é", "^.$"));
        assert!(!m("é", "^..$"));
        // A negated class must not match half of a multi-byte character.
        assert!(m("héllo", "h[^x]llo"));
        assert!(m("あ", "^[^a]$"));
        assert!(!m("あ", "^[^a][^a]$"));
        // Code-point ranges, and the shorthand classes' complements.
        assert!(m("héllo", "h[à-ÿ]llo"));
        assert!(!m("héllo", "h[à-å]llo"));
        assert!(m("é", "^\\D$"));
        assert!(m("é", "^\\W$"));
        assert!(!m("é", "^\\w$"));
        // A literal multi-byte character is one atom, so a quantifier applies to all of it.
        assert!(m("ééé", "^é{3}$"));
        assert!(m("ééé", "^é+$"));
        assert!(!m("éé", "^é{3}$"));
        assert_eq!(ext("naïve", "^(.)(.)(.)", 3).as_deref(), Some("ï"));
        assert_eq!(ext("日本語abc", "[^a-z]+", 0).as_deref(), Some("日本語"));
    }

    #[test]
    fn match_boundaries_stay_on_character_boundaries() {
        // Every slice a caller takes out of `find`'s result must still be valid UTF-8; before the
        // UTF-8-aware VM, `h.` here matched `h` plus the first byte of `é`.
        // duckdb: regexp_extract('héllo','h.') = 'hé'
        let prog = compile(b"h.").unwrap();
        let text = "héllo".as_bytes();
        let saves = find(&prog, text).unwrap().unwrap();
        let piece = &text[saves[0] as usize..saves[1] as usize];
        assert_eq!(core::str::from_utf8(piece), Ok("hé"));
    }

    // --- POSIX bracket expressions (cross-checked with DuckDB) ---------------

    #[test]
    fn posix_classes() {
        assert!(m("abc", "[[:alpha:]]+"));
        assert!(m("123", "^[[:digit:]]+$"));
        assert!(m("a1", "^[[:alnum:]]+$"));
        assert!(m(" ", "[[:space:]]"));
        assert!(m("A", "[[:upper:]]"));
        assert!(!m("a", "^[[:upper:]]$"));
        assert!(m("a", "[[:lower:]]"));
        assert!(m("!", "[[:punct:]]"));
        assert!(m("deadBEEF", "^[[:xdigit:]]+$"));
        assert!(!m("g", "^[[:xdigit:]]$"));
        // Mixing a POSIX name with ordinary members inside one bracket expression.
        assert!(m("a-b", "^[[:alpha:]-]+$"));
        assert!(m("x", "[[:alpha:][:digit:]]"));
        // Negated form, and the ASCII-only semantics RE2/DuckDB use
        // (duckdb: regexp_matches('é','[[:alpha:]]') = false).
        assert!(m("a", "[[:^digit:]]"));
        assert!(!m("é", "[[:alpha:]]"));
        // A `[` that does not open a bracket expression is still a literal member.
        assert!(m("[", "^[[]$"));
        assert!(m("[:", "^[[:]+$"));
        // An unknown class name is an error, never a silently different set.
        assert_eq!(code_of(compile(b"[[:nope:]]").map(|_| ())), Some(Code::UnsupportedFeature));
    }

    #[test]
    fn alternation_and_groups() {
        assert!(m("cat", "cat|dog"));
        assert!(m("dog", "cat|dog"));
        assert!(!m("fish", "^(cat|dog)$"));
        assert_eq!(ext("foobar", "(foo)(bar)", 1).as_deref(), Some("foo"));
        assert_eq!(ext("foobar", "(foo)(bar)", 2).as_deref(), Some("bar"));
        assert_eq!(ext("abc", "(?:a)(b)", 1).as_deref(), Some("b"));
    }

    #[test]
    fn escapes() {
        for c in [".", "*", "+", "?", "(", ")", "[", "]", "{", "}", "|", "^", "$", "\\"] {
            let pat = format!("\\{c}");
            assert!(m(c, &pat), "escape {c} failed");
        }
    }

    // --- Greedy / leftmost-first semantics (cross-checked with DuckDB) -------

    #[test]
    fn greedy_and_leftmost() {
        // duckdb: regexp_extract('aaa','a*') = 'aaa'
        assert_eq!(ext("aaa", "a*", 0).as_deref(), Some("aaa"));
        // duckdb: regexp_extract('xaaay','a*') = '' (zero-length match at the leading 'x')
        assert_eq!(ext("xaaay", "a*", 0).as_deref(), Some(""));
        // duckdb: regexp_extract('abc','a|ab') = 'a' (alternation is leftmost-first, not longest)
        assert_eq!(ext("abc", "a|ab", 0).as_deref(), Some("a"));
        // duckdb: regexp_extract('xabcx','ab|abc') = 'ab'
        assert_eq!(ext("xabcx", "ab|abc", 0).as_deref(), Some("ab"));
    }

    // --- Compilation errors --------------------------------------------------

    #[test]
    fn rejects_unsupported_constructs() {
        assert_eq!(code_of(compile(b"(?=a)").map(|_| ())), Some(Code::UnsupportedFeature));
        assert_eq!(code_of(compile(b"(?!a)").map(|_| ())), Some(Code::UnsupportedFeature));
        assert_eq!(code_of(compile(b"(?<=a)").map(|_| ())), Some(Code::UnsupportedFeature));
        assert_eq!(code_of(compile(b"(?P<n>a)").map(|_| ())), Some(Code::UnsupportedFeature));
        assert_eq!(code_of(compile(b"(a)\\1").map(|_| ())), Some(Code::UnsupportedFeature));
        assert_eq!(code_of(compile(b"a*?").map(|_| ())), Some(Code::UnsupportedFeature));
        assert_eq!(code_of(compile(b"a+?").map(|_| ())), Some(Code::UnsupportedFeature));
        assert_eq!(code_of(compile(b"\\bfoo").map(|_| ())), Some(Code::UnsupportedFeature));
    }

    #[test]
    fn rejects_malformed_patterns() {
        assert_eq!(code_of(compile(b"(a").map(|_| ())), Some(Code::SyntaxError));
        assert_eq!(code_of(compile(b"a)").map(|_| ())), Some(Code::SyntaxError));
        assert_eq!(code_of(compile(b"[a").map(|_| ())), Some(Code::SyntaxError));
        assert_eq!(code_of(compile(b"*a").map(|_| ())), Some(Code::SyntaxError));
        assert_eq!(code_of(compile(b"a**").map(|_| ())), Some(Code::SyntaxError));
        assert_eq!(code_of(compile(b"a{2,1}").map(|_| ())), Some(Code::SyntaxError));
        assert_eq!(code_of(compile(b"\\").map(|_| ())), Some(Code::SyntaxError));
        assert!(compile(b"foobar").is_ok());
    }

    // --- Pathological input must still finish quickly ------------------------

    #[test]
    fn pathological_pattern_is_not_exponential() {
        // (a+)+b is the classic pattern that goes exponential under naive recursive backtracking.
        // A Thompson NFA should finish in O(instructions x input length).
        let subject = "a".repeat(5000);
        let prog = compile(b"(a+)+b").unwrap();
        let start = Instant::now();
        let hit = is_match(&prog, subject.as_bytes()).unwrap();
        let elapsed = start.elapsed();
        assert!(!hit);
        assert!(elapsed.as_millis() < 2000, "took too long: {elapsed:?}");
    }

    #[test]
    fn step_limit_triggers_on_adversarial_input() {
        // Applies the classic NFA thread-explosion pattern `a?{n}a{n}` to a long non-matching
        // input to reach MAX_STEPS. The instruction count itself stays within MAX_PROG_INSTS
        // (4096) at 1000*2 + 1000 + 3 = 3003, while the live thread count per step reaches ~1000,
        // so a long input exceeds MAX_STEPS early.
        let pat: String = "a?".repeat(1000) + &"a".repeat(1000);
        let prog = compile(pat.as_bytes()).unwrap();
        // A 'b' is interposed every 999 'a's so that consecutive 'a's never reach 1000 (no match).
        // That keeps the search endlessly "just short" all the way through, so it reaches the
        // MAX_STEPS check rather than being cut off early by finding a match.
        let block = "a".repeat(999) + "b";
        let subject = block.repeat(5000);
        let start = Instant::now();
        let r = is_match(&prog, subject.as_bytes());
        let elapsed = start.elapsed();
        assert_eq!(code_of(r.map(|_| ())), Some(Code::LimitExceeded));
        assert!(elapsed.as_millis() < 5000, "took too long: {elapsed:?}");
    }

    #[test]
    fn prog_size_limit_triggers() {
        // Several {1000}s concatenated to exceed the instruction cap.
        let pat = "a{1000}".repeat(10);
        assert_eq!(code_of(compile(pat.as_bytes()).map(|_| ())), Some(Code::LimitExceeded));
    }

    #[test]
    fn nested_repeat_does_not_blow_up_before_limit_check() {
        // Even a pattern that would expand to the equivalent of 1000^3 instructions ends
        // immediately with LimitExceeded (it does not OOM, since the AST is never duplicated).
        let start = Instant::now();
        let r = compile(b"((a{1000}){1000}){1000}");
        let elapsed = start.elapsed();
        assert_eq!(code_of(r.map(|_| ())), Some(Code::LimitExceeded));
        assert!(elapsed.as_millis() < 2000, "took too long: {elapsed:?}");
    }

    // --- The SQL functions, per Vector ---------------------------------------

    fn vs(vals: &[Option<&str>]) -> Vector {
        let mut v = Vector::new(Ty::Varchar);
        for x in vals {
            match x {
                Some(s) => v.push_value(&Value::Bytes(s.as_bytes().to_vec())),
                None => v.push_null(),
            }
        }
        v
    }

    fn vi(vals: &[Option<i64>]) -> Vector {
        let mut v = Vector::new(Ty::BigInt);
        for x in vals {
            match x {
                Some(n) => v.push_value(&Value::I64(*n)),
                None => v.push_null(),
            }
        }
        v
    }

    fn str_at(v: &Vector, i: usize) -> Option<String> {
        if !v.is_valid(i) {
            return None;
        }
        Some(String::from_utf8(v.bytes().get(i).to_vec()).unwrap())
    }

    fn bool_at(v: &Vector, i: usize) -> Option<bool> {
        if !v.is_valid(i) {
            return None;
        }
        match v.data() {
            Data::Bool(b) => Some(b.get(i)),
            _ => None,
        }
    }

    #[test]
    fn sql_matches_basics_and_nulls() {
        let s = vs(&[Some("foobar"), Some("xxx"), None, Some("abc")]);
        let p = vs(&[Some("o+b"), Some("foo"), Some("a"), None]);
        let r = eval_matches(&[&s, &p]).unwrap();
        assert_eq!(bool_at(&r, 0), Some(true));
        assert_eq!(bool_at(&r, 1), Some(false));
        assert_eq!(bool_at(&r, 2), None);
        assert_eq!(bool_at(&r, 3), None);
    }

    #[test]
    fn sql_matches_constant_pattern_len1_result() {
        let s = vs(&[Some("abc")]);
        let p = vs(&[Some("b")]);
        let r = eval_matches(&[&s, &p]).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(bool_at(&r, 0), Some(true));
    }

    #[test]
    fn sql_extract_no_match_and_group_range() {
        let s = vs(&[Some("xxx"), Some("foobar"), Some("foobar")]);
        let p = vs(&[Some("foo"), Some("(foo)(bar)"), Some("(foo)(bar)")]);
        let g = vi(&[Some(0), Some(1), Some(9)]);
        let r = eval_extract(&[&s, &p, &g]).unwrap();
        // A non-match also gives the empty string (not NULL), exactly as measured in DuckDB.
        assert_eq!(str_at(&r, 0).as_deref(), Some(""));
        assert_eq!(str_at(&r, 1).as_deref(), Some("foo"));
        // group=9 is within 0-9 but exceeds this pattern's capture count (2), so the empty string.
        assert_eq!(str_at(&r, 2).as_deref(), Some(""));
    }

    #[test]
    fn sql_extract_group_out_of_hard_range_errors() {
        let s = vs(&[Some("a")]);
        let p = vs(&[Some("a")]);
        let g = vi(&[Some(10)]);
        assert_eq!(code_of(eval_extract(&[&s, &p, &g]).map(|_| ())), Some(Code::ValueOutOfRange));
        let g2 = vi(&[Some(-1)]);
        assert_eq!(code_of(eval_extract(&[&s, &p, &g2]).map(|_| ())), Some(Code::ValueOutOfRange));
    }

    #[test]
    fn sql_extract_null_propagation() {
        let s = vs(&[None]);
        let p = vs(&[Some("a")]);
        let r = eval_extract(&[&s, &p]).unwrap();
        assert_eq!(str_at(&r, 0), None);
    }

    #[test]
    fn sql_replace_first_vs_global() {
        let s = vs(&[Some("hello world")]);
        let p = vs(&[Some("o")]);
        let r0 = vs(&[Some("0")]);
        let first = eval_replace(&[&s, &p, &r0]).unwrap();
        assert_eq!(str_at(&first, 0).as_deref(), Some("hell0 world"));
        let flag = vs(&[Some("g")]);
        let global = eval_replace(&[&s, &p, &r0, &flag]).unwrap();
        assert_eq!(str_at(&global, 0).as_deref(), Some("hell0 w0rld"));
    }

    #[test]
    fn sql_replace_backreferences() {
        let s = vs(&[Some("foobar")]);
        let p = vs(&[Some("(foo)(bar)")]);
        let r1 = vs(&[Some("\\2\\1")]);
        let out = eval_replace(&[&s, &p, &r1]).unwrap();
        assert_eq!(str_at(&out, 0).as_deref(), Some("barfoo"));

        let flag = vs(&[Some("g")]);
        let r2 = vs(&[Some("\\2-\\1")]);
        let out2 = eval_replace(&[&s, &p, &r2, &flag]).unwrap();
        assert_eq!(str_at(&out2, 0).as_deref(), Some("bar-foo"));
    }

    #[test]
    fn sql_replace_invalid_backreference_is_noop() {
        // Measured in DuckDB: when a group reference exceeds the pattern's capture count, no
        // replacement happens and the original string is returned.
        let s = vs(&[Some("abc")]);
        let p = vs(&[Some("b")]);
        let r = vs(&[Some("X\\9")]);
        let out = eval_replace(&[&s, &p, &r]).unwrap();
        assert_eq!(str_at(&out, 0).as_deref(), Some("abc"));
    }

    #[test]
    fn sql_replace_global_empty_match_adjacency() {
        // duckdb: regexp_replace('aXbXc','X*','-','g') = '-a-b-c-'
        let s = vs(&[Some("aXbXc")]);
        let p = vs(&[Some("X*")]);
        let r = vs(&[Some("-")]);
        let flag = vs(&[Some("g")]);
        let out = eval_replace(&[&s, &p, &r, &flag]).unwrap();
        assert_eq!(str_at(&out, 0).as_deref(), Some("-a-b-c-"));
    }

    #[test]
    fn sql_replace_null_propagation() {
        let s = vs(&[None]);
        let p = vs(&[Some("a")]);
        let r = vs(&[Some("b")]);
        let out = eval_replace(&[&s, &p, &r]).unwrap();
        assert_eq!(str_at(&out, 0), None);
    }

    #[test]
    fn sql_replace_bad_flag_errors() {
        let s = vs(&[Some("abc")]);
        let p = vs(&[Some("a")]);
        let r = vs(&[Some("x")]);
        let bad = vs(&[Some("i")]);
        assert_eq!(
            code_of(eval_replace(&[&s, &p, &r, &bad]).map(|_| ())),
            Some(Code::UnsupportedFeature)
        );
    }

    #[test]
    fn sql_matches_len1_result_for_constant_folding() {
        let s = vs(&[Some("abc")]);
        let p = vs(&[Some("b")]);
        let r = eval_matches(&[&s, &p]).unwrap();
        assert_eq!(r.len(), 1);
    }
}
