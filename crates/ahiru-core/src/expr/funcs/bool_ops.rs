//! Bool output
use super::string::find;
use super::*;

pub(super) fn eval_bool(id: FuncId, a: &A) -> Result<Option<bool>> {
    let (s, p) = (a.bytes(0), a.bytes(1));
    Ok(Some(match id {
        F_STARTS_WITH => s.len() >= p.len() && &s[..p.len()] == p,
        F_ENDS_WITH => s.len() >= p.len() && &s[s.len() - p.len()..] == p,
        F_CONTAINS => find(s, p).is_some(),
        // `glob_match` uses the same "remember only the most recent `*`" two-pointer method as
        // `like_match`, so it needs no compilation cache the way a regex does
        // (hence no per-call bypass directly under `call()`).
        F_GLOB => glob_match(s, p),
        _ => err!(Internal),
    }))
}

/// Shell glob pattern matching. The same meaning as DuckDB's `GLOB` operator (all of the
/// following confirmed with the `duckdb` CLI):
///
/// - `*` matches zero or more bytes and `?` matches exactly one byte.
/// - `[...]` is a character class. `[!...]` negates (`[^...]` does not negate and instead means
///   "a class containing the character `^`"; DuckDB does the same). Ranges such as `a-z` and a
///   leading `]` as a literal `]` (`[]]` matches the single character `]`) are supported.
/// - `\` makes the next byte a literal (there is no notion of an `ESCAPE` clause; it is always in
///   effect).
/// - A `[` with no closing `]` makes the whole rest of the pattern an element that "can never
///   match" (DuckDB does the same: even `'a[bc' GLOB 'a[bc'` is false, i.e. `[` does not fall back
///   to a literal). It does not panic.
///
/// Unlike the other string functions, multi-byte characters are handled **byte-wise** (the same
/// judgment as the `regexp` family: making character classes code-point-wise would grow the code
/// for little practical gain).
///
/// The same "remember one position of the previous `*`" two-pointer method as `like_match`
/// (`kernels.rs`). With only one backtracking point it stays within `O(|s| * |p|)` at worst, so
/// even a pathological pattern like `***...*` does not go exponential.
fn glob_match(s: &[u8], p: &[u8]) -> bool {
    let (mut si, mut pi) = (0usize, 0usize);
    let (mut star_p, mut star_s) = (usize::MAX, 0usize);
    loop {
        if pi < p.len() && p[pi] == b'*' {
            star_p = pi;
            star_s = si;
            pi += 1;
            continue;
        }
        if si < s.len() && pi < p.len() && glob_atom(p, &mut pi, s[si]) {
            si += 1;
            continue;
        }
        // Feed the previous `*` one more byte and retry. With no `*` seen at all, it is settled as
        // a non-match.
        if si < s.len() && star_p != usize::MAX {
            star_s += 1;
            si = star_s;
            pi = star_p + 1;
            continue;
        }
        break;
    }
    while pi < p.len() && p[pi] == b'*' {
        pi += 1;
    }
    si == s.len() && pi == p.len()
}

/// Decides whether the single element starting at `p[*pi]` (a literal, `?`, `\x`, or `[...]`)
/// matches `c`, and advances `*pi` past the element either way.
/// `*` never arrives here, since the caller (`glob_match`) handles it first.
fn glob_atom(p: &[u8], pi: &mut usize, c: u8) -> bool {
    match p[*pi] {
        b'?' => {
            *pi += 1;
            true
        }
        // A trailing `\` has nothing to escape, so it is plainly treated as a literal `\`
        // (a fallback; it does not panic).
        b'\\' if *pi + 1 < p.len() => {
            let want = p[*pi + 1];
            *pi += 2;
            want == c
        }
        b'[' => glob_class(p, pi, c),
        lit => {
            *pi += 1;
            lit == c
        }
    }
}

/// Reads the character class starting at `p[*pi]` (== `[`), decides whether `c` belongs to it, and
/// advances `*pi` past the closing `]`. Without a closing `]` it returns "never matches" and
/// advances `*pi` to the end of the pattern (as the struct comment says, DuckDB behaves the same
/// way).
fn glob_class(p: &[u8], pi: &mut usize, c: u8) -> bool {
    let mut i = *pi + 1;
    let negate = p.get(i) == Some(&b'!');
    if negate {
        i += 1;
    }
    let members_start = i;
    // A leading `]` is treated as a literal member rather than the class's terminator.
    if p.get(i) == Some(&b']') {
        i += 1;
    }
    while i < p.len() && p[i] != b']' {
        i += 1;
    }
    if i >= p.len() {
        *pi = p.len();
        return false;
    }
    let close = i;
    *pi = close + 1;
    let mut hit = false;
    let mut j = members_start;
    while j < close {
        // A `-` that is neither first nor last denotes a range (`a-z`).
        if p[j] == b'-' && j > members_start && j + 1 < close {
            let (lo, hi) = (p[j - 1], p[j + 1]);
            if lo <= c && c <= hi {
                hit = true;
            }
            j += 2;
        } else {
            if p[j] == c {
                hit = true;
            }
            j += 1;
        }
    }
    hit != negate
}
