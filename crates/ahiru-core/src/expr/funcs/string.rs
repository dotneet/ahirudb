//! Strings (Bytes output)
use super::datetime::{date_part, strftime};
use super::json::{json_extract_or_whole, write_json_scalar};
use super::*;

/// The number of UTF-8 code points. It merely does not count continuation bytes.
pub(super) fn cp_count(s: &[u8]) -> usize {
    s.iter().filter(|&&b| (b & 0xc0) != 0x80).count()
}

/// The byte position of the `k`-th code point boundary from the start. Out of range gives `s.len()`.
fn cp_byte(s: &[u8], k: usize) -> usize {
    let mut seen = 0usize;
    for (i, &b) in s.iter().enumerate() {
        if (b & 0xc0) != 0x80 {
            if seen == k {
                return i;
            }
            seen += 1;
        }
    }
    s.len()
}

/// The byte length of the code point starting at position `i`.
fn cp_width(s: &[u8], i: usize) -> usize {
    let mut w = 1;
    while i + w < s.len() && (s[i + w] & 0xc0) == 0x80 {
        w += 1;
    }
    w
}

/// The start position of the code point ending at `hi`.
fn cp_back(s: &[u8], hi: usize) -> usize {
    let mut lo = hi - 1;
    while lo > 0 && (s[lo] & 0xc0) == 0x80 {
        lo -= 1;
    }
    lo
}

/// Whether the code point is in `set` (`trim`'s character set).
fn in_set(set: &[u8], c: &[u8]) -> bool {
    let mut i = 0;
    while i < set.len() {
        let w = cp_width(set, i);
        if &set[i..i + w] == c {
            return true;
        }
        i += w;
    }
    false
}

/// The starting byte position of `needle` within `hay`. `None` if absent. An empty needle gives 0.
pub(super) fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if needle.len() > hay.len() {
        return None;
    }
    (0..=hay.len() - needle.len()).find(|&i| &hay[i..i + needle.len()] == needle)
}

/// One row's worth of a string function. A row returning `false` becomes NULL.
pub(super) fn eval_str(id: FuncId, a: &A, out: &mut Vec<u8>) -> Result<bool> {
    match id {
        F_UPPER | F_LOWER => {
            // ASCII only. Unicode case tables do not fit the 1 MiB budget.
            let s = a.bytes(0);
            out.extend_from_slice(s);
            for c in out.iter_mut() {
                *c = if id == F_UPPER { c.to_ascii_uppercase() } else { c.to_ascii_lowercase() };
            }
        }
        F_TRIM | F_LTRIM | F_RTRIM => {
            let s = a.bytes(0);
            // The one-argument form drops **whitespace only** (the same as DuckDB; tabs remain).
            let set: &[u8] = if a.n() >= 2 { a.bytes(1) } else { b" " };
            let (mut lo, mut hi) = (0usize, s.len());
            if id != F_RTRIM {
                while lo < hi {
                    let w = cp_width(s, lo);
                    if !in_set(set, &s[lo..lo + w]) {
                        break;
                    }
                    lo += w;
                }
            }
            if id != F_LTRIM {
                while hi > lo {
                    let st = cp_back(s, hi);
                    if !in_set(set, &s[st..hi]) {
                        break;
                    }
                    hi = st;
                }
            }
            out.extend_from_slice(&s[lo..hi]);
        }
        F_SUBSTR => {
            let s = a.bytes(0);
            let mut start = a.int(1);
            // A negative start position counts from the end (DuckDB).
            if start < 0 {
                start += cp_count(s) as i64 + 1;
            }
            let (mut from, mut to) = if a.n() >= 3 {
                let e = start.saturating_add(a.int(2));
                // A negative length means "backwards from the start position" (DuckDB reverses the range).
                if a.int(2) < 0 {
                    (e, start)
                } else {
                    (start, e)
                }
            } else {
                (start, i64::MAX)
            };
            from = from.max(1);
            to = to.max(from);
            let cap = cp_count(s) as i64 + 1;
            let b0 = cp_byte(s, (from - 1).min(cap) as usize);
            let b1 = cp_byte(s, (to - 1).min(cap) as usize);
            out.extend_from_slice(&s[b0..b1.max(b0)]);
        }
        F_REPLACE => {
            let (s, from, to) = (a.bytes(0), a.bytes(1), a.bytes(2));
            if from.is_empty() {
                out.extend_from_slice(s);
            } else {
                let mut i = 0;
                while i < s.len() {
                    if s.len() - i >= from.len() && &s[i..i + from.len()] == from {
                        out.extend_from_slice(to);
                        i += from.len();
                    } else {
                        out.push(s[i]);
                        i += 1;
                    }
                }
            }
        }
        F_LPAD | F_RPAD => {
            let s = a.bytes(0);
            let want = a.int(1);
            ensure!(want <= MAX_STR, LimitExceeded);
            let cl = cp_count(s) as i64;
            if want <= cl {
                // Longer than the target length is truncated (DuckDB).
                let cut = cp_byte(s, want.max(0) as usize);
                out.extend_from_slice(&s[..cut]);
                return Ok(true);
            }
            let pad = a.bytes(2);
            let mut fill = Vec::new();
            if !pad.is_empty() {
                // The padding repeats and is cut off partway at the end.
                let pl = cp_count(pad);
                let mut need = (want - cl) as usize;
                while need > 0 {
                    let take = need.min(pl);
                    fill.extend_from_slice(&pad[..cp_byte(pad, take)]);
                    need -= take;
                }
            }
            if id == F_LPAD {
                out.extend_from_slice(&fill);
                out.extend_from_slice(s);
            } else {
                out.extend_from_slice(s);
                out.extend_from_slice(&fill);
            }
        }
        F_LEFT | F_RIGHT => {
            let s = a.bytes(0);
            let cl = cp_count(s) as i64;
            let k = a.int(1);
            // A negative count means "all but the last (LEFT) / first (RIGHT) |k| characters".
            let take = if k < 0 { (cl + k).max(0) } else { k.min(cl) };
            let cut = cp_byte(s, take as usize);
            if id == F_LEFT {
                out.extend_from_slice(&s[..cut]);
            } else {
                // RIGHT takes `take` characters off the end, so it skips the first `cl - take`.
                let skip = cp_byte(s, (cl - take) as usize);
                out.extend_from_slice(&s[skip..]);
            }
        }
        F_CHR => {
            let cp = a.int(0);
            // Out of Unicode's range, or a surrogate half, has no UTF-8 encoding -> NULL
            // (this engine's "undefined argument gives NULL" convention).
            return match u32::try_from(cp).ok().and_then(char::from_u32) {
                Some(c) => {
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                    Ok(true)
                }
                None => Ok(false),
            };
        }
        // Uppercase hex, matching DuckDB. `hex` dumps the argument's bytes; `to_hex` renders an
        // integer in base 16 (negative values through their two's-complement bit pattern, again
        // matching DuckDB: `select to_hex(-1)` -> `FFFFFFFFFFFFFFFF`).
        F_HEX => {
            for &b in a.bytes(0) {
                out.push(hex_digit(b >> 4));
                out.push(hex_digit(b & 0x0f));
            }
        }
        F_TO_HEX => {
            // The bit pattern is read at the argument's own width, not through BIGINT: casting
            // first truncated `hex(-1::HUGEINT)` to 16 digits and turned anything above
            // `i64::MAX` (a large UBIGINT, say) into NULL. Like DuckDB, only HUGEINT is rendered
            // at 128 bits; every narrower integer type is widened to 64
            // (`hex(-1::TINYINT)` is `FFFFFFFFFFFFFFFF` there too).
            let nibbles = match a.at(0) {
                Some((v, _)) if v.ty() == Ty::HugeInt => 32,
                _ => 16,
            };
            let x = a.i128(0);
            let mask = if nibbles == 32 { u128::MAX } else { u64::MAX as u128 };
            let v = (x as u128) & mask;
            let mut started = false;
            for shift in (0..nibbles).rev() {
                let d = ((v >> (shift * 4)) & 0x0f) as u8;
                if d != 0 || started || shift == 0 {
                    out.push(hex_digit(d));
                    started = true;
                }
            }
        }
        F_UNHEX => {
            let s = a.bytes(0);
            // DuckDB treats an odd-length input as if it had a leading zero nibble
            // (`unhex('ABC')` -> `\x0A\xBC`). An invalid digit is an invalid conversion,
            // not a per-row NULL; `Err` propagates through `call` as a query error.
            let mut i = 0;
            if !s.len().is_multiple_of(2) {
                let Some(v) = hex_val(s[0]) else { err!(InvalidCast) };
                out.push(v);
                i = 1;
            }
            while i < s.len() {
                let Some(hi) = hex_val(s[i]) else { err!(InvalidCast) };
                let Some(lo) = hex_val(s[i + 1]) else { err!(InvalidCast) };
                out.push((hi << 4) | lo);
                i += 2;
            }
        }
        F_DAYNAME | F_MONTHNAME => {
            let name = if id == F_DAYNAME {
                const D: [&[u8]; 7] = [
                    b"Sunday",
                    b"Monday",
                    b"Tuesday",
                    b"Wednesday",
                    b"Thursday",
                    b"Friday",
                    b"Saturday",
                ];
                D[date_part(P_DOW, a.int(0)).unwrap_or(0).clamp(0, 6) as usize]
            } else {
                const M: [&[u8]; 12] = [
                    b"January",
                    b"February",
                    b"March",
                    b"April",
                    b"May",
                    b"June",
                    b"July",
                    b"August",
                    b"September",
                    b"October",
                    b"November",
                    b"December",
                ];
                M[date_part(P_MONTH, a.int(0)).unwrap_or(1).clamp(1, 12) as usize - 1]
            };
            out.extend_from_slice(name);
        }
        F_STRING_SPLIT => string_split(a.bytes(0), a.bytes(1), out),
        F_LIST_SORT | F_LIST_DISTINCT | F_LIST_REVERSE => {
            return super::json::list_rearrange(id, a.bytes(0), out);
        }
        F_REVERSE => {
            let s = a.bytes(0);
            let mut hi = s.len();
            while hi > 0 {
                let lo = cp_back(s, hi);
                out.extend_from_slice(&s[lo..hi]);
                hi = lo;
            }
        }
        F_SPLIT_PART => {
            let (s, d, k) = (a.bytes(0), a.bytes(1), a.int(2));
            // The index is 1-based. Negative counts from the end. 0 and out of range give the empty string.
            if k == 0 {
                return Ok(true);
            }
            // An empty delimiter splits into characters, matching DuckDB
            // (`split_part('abc', '', 2)` -> `b`); an empty string still has exactly one
            // (empty) part. The parts are code points, not bytes, like every other
            // position-taking string function here.
            if d.is_empty() {
                let cnt = cp_count(s).max(1) as i64;
                let idx = if k < 0 { cnt + k } else { k - 1 };
                if (0..cnt).contains(&idx) {
                    let lo = cp_byte(s, idx as usize);
                    out.extend_from_slice(&s[lo..(lo + cp_width(s, lo)).min(s.len())]);
                }
                return Ok(true);
            }
            let mut cnt = 1i64;
            let mut i = 0usize;
            while i + d.len() <= s.len() {
                if &s[i..i + d.len()] == d {
                    cnt += 1;
                    i += d.len();
                } else {
                    i += 1;
                }
            }
            let idx = if k < 0 { cnt + k } else { k - 1 };
            if idx < 0 || idx >= cnt {
                return Ok(true);
            }
            let (mut cur, mut st) = (0i64, 0usize);
            i = 0;
            while i + d.len() <= s.len() {
                if &s[i..i + d.len()] == d {
                    if cur == idx {
                        out.extend_from_slice(&s[st..i]);
                        return Ok(true);
                    }
                    cur += 1;
                    i += d.len();
                    st = i;
                } else {
                    i += 1;
                }
            }
            out.extend_from_slice(&s[st..]);
        }
        F_REPEAT => {
            let s = a.bytes(0);
            let k = a.int(1);
            // The output is empty whenever the input is empty or the count is not positive, and
            // the loop below must not be entered in that case: `repeat('', 9223372036854775807)`
            // passes the length check (the product is zero) and would then spin for effectively
            // forever appending nothing. DuckDB returns `''` immediately.
            if s.is_empty() || k <= 0 {
                return Ok(true);
            }
            ensure!(k.saturating_mul(s.len() as i64) <= MAX_STR, LimitExceeded);
            for _ in 0..k {
                out.extend_from_slice(s);
            }
        }
        F_STRFTIME => strftime(a.int(0), a.bytes(1), out),
        F_JSON_EXTRACT => {
            return match crate::json::extract(a.bytes(0), a.bytes(1))? {
                Some((span, _)) => {
                    out.extend_from_slice(span);
                    Ok(true)
                }
                None => Ok(false),
            };
        }
        F_JSON_EXTRACT_STRING => {
            return match crate::json::extract(a.bytes(0), a.bytes(1))? {
                Some((span, kind)) => crate::json::write_extracted_text(span, kind, out),
                None => Ok(false),
            };
        }
        F_JSON_TYPE => {
            let found = json_extract_or_whole(a)?;
            return match found {
                Some((span, kind)) => {
                    out.extend_from_slice(crate::json::type_name(kind, span).as_bytes());
                    Ok(true)
                }
                None => Ok(false),
            };
        }
        F_TO_JSON => {
            if let Some((v, j)) = a.at(0) {
                write_json_scalar(v, j, out);
            }
        }
        F_LIST_EXTRACT => {
            return match crate::json::list_index(a.bytes(0), a.int(1))? {
                Some(span) => {
                    out.extend_from_slice(span);
                    Ok(true)
                }
                None => Ok(false),
            };
        }
        F_LIST_SLICE => {
            let doc = a.bytes(0);
            return match crate::json::list_slice(doc, a.int(1), a.int(2))? {
                Some((lo, hi)) => {
                    out.push(b'[');
                    out.extend_from_slice(&doc[lo..hi]);
                    out.push(b']');
                    Ok(true)
                }
                None => Ok(false),
            };
        }
        F_MAP_EXTRACT => {
            return match crate::json::map_get(a.bytes(0), a.bytes(1))? {
                Some(span) => {
                    out.extend_from_slice(span);
                    Ok(true)
                }
                None => Ok(false),
            };
        }
        // NULL propagation can be left to the default in the caller's `call()` (the `live(i)`
        // check), so unlike `json_array`/`concat` there is no need to bypass directly under
        // `call()` (even with variadic arguments, `A` holds all of `args: &[&Vector]`, so it stays
        // contained within `eval_str`). See the comments on `printf_scan`/`format_scan` for the
        // supported format specifiers.
        F_PRINTF => printf_scan(a.bytes(0), a, out)?,
        F_FORMAT => format_scan(a.bytes(0), a, out)?,
        _ => err!(Internal),
    }
    Ok(true)
}

/// One uppercase hex digit. `json.rs` and `exec::agg` carry their own copies for `\uXXXX`
/// escapes (lowercase there); this one exists for `hex`/`to_hex`.
fn hex_digit(n: u8) -> u8 {
    if n < 10 {
        b'0' + n
    } else {
        b'A' + (n - 10)
    }
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// `string_split(s, sep)`. Writes a JSON array of strings, which is how this engine spells a
/// LIST (see `expr::funcs`'s module docs).
///
fn string_split(s: &[u8], sep: &[u8], out: &mut Vec<u8>) {
    out.push(b'[');
    // An empty separator splits into characters, matching DuckDB
    // (`string_split('abc', '')` -> `[a, b, c]`); an empty string still yields a
    // one-element list holding the empty string. Kept deliberately in step with
    // `split_part`'s empty-delimiter rule above -- the two used to share the old
    // "the whole string is one part" divergence, and letting only one of them move
    // would leave `split_part(s, '', k)` disagreeing with `string_split(s, '')[k]`.
    if sep.is_empty() {
        let mut i = 0usize;
        while i < s.len() {
            let w = cp_width(s, i);
            crate::json::write_json_string(&s[i..i + w], out);
            i += w;
            if i < s.len() {
                out.push(b',');
            }
        }
        if s.is_empty() {
            crate::json::write_json_string(s, out);
        }
        out.push(b']');
        return;
    }
    let mut start = 0usize;
    let mut i = 0usize;
    while i + sep.len() <= s.len() {
        if &s[i..i + sep.len()] == sep {
            crate::json::write_json_string(&s[start..i], out);
            out.push(b',');
            i += sep.len();
            start = i;
        } else {
            i += 1;
        }
    }
    crate::json::write_json_string(&s[start..], out);
    out.push(b']');
}

/// `printf`'s maximum precision (the `N` of `%.<N>f`). A larger precision is silently clamped to
/// this. `fmt_fixed` itself is exact at any precision, so the cap is purely a bound on how much
/// output one specifier may produce; C's `printf` would keep going. (Known difference from
/// DuckDB, which follows C: `printf('%.40f', 0.1)` prints 40 fractional digits there and 32
/// here.)
const MAX_PRINTF_PREC: u32 = 32;

/// The cap on `printf`/`format`'s width specification. A limit so a malicious query cannot eat
/// memory and time by writing a width like `%<huge number>d` (the same motivation as `MAX_STR` for
/// `repeat`/`lpad`).
const MAX_PRINTF_WIDTH: usize = 1 << 16;

/// `printf(fmt, args...)`. Of C's `%` formats, only the following are supported:
///
/// - `%%`  a literal `%`
/// - `%[-][0][<width>]d`  an integer. The corresponding actual argument is BOOLEAN (0/1) or an
///   integer type. Floating point is truncated toward zero.
/// - `%[-][0][<width>][.<precision>]f`  fixed-point notation for floating point. The precision
///   defaults to 6 digits (the same as C's `printf`; confirmed that
///   `duckdb -c "select printf('%f', 3.5)"` gives `3.500000`) and can go up to `MAX_PRINTF_PREC`.
/// - `%[-][<width>]s`  stringified by the same rules as `write_display`.
///
/// A known difference from DuckDB: DuckDB errors when a FLOAT is passed to `%d` or an INTEGER to
/// `%s` (the `fmt` library's type strictness), whereas here practicality wins and both are accepted
/// (`%s` stringifies any supported physical type). Base conversions such as `%x`/`%o`, a width
/// given by an actual argument via `*`, and `%1$d`-style argument numbering are unsupported
/// (`UnsupportedFeature`). Fewer actual arguments than specifiers gives `WrongArgCount` (the same
/// as DuckDB. Surplus arguments may be ignored: `duckdb -c "select printf('%d', 1, 2)"` gives `1`).
fn printf_scan(fmt: &[u8], a: &A, out: &mut Vec<u8>) -> Result<()> {
    let mut ai = 1usize; // args[0] is fmt itself
    let mut i = 0usize;
    while i < fmt.len() {
        if fmt[i] != b'%' {
            out.push(fmt[i]);
            i += 1;
            continue;
        }
        i += 1;
        ensure!(i < fmt.len(), SyntaxError);
        if fmt[i] == b'%' {
            out.push(b'%');
            i += 1;
            continue;
        }
        let (mut left, mut zero) = (false, false);
        loop {
            match fmt.get(i) {
                Some(b'-') => {
                    left = true;
                    i += 1;
                }
                Some(b'0') if !zero => {
                    zero = true;
                    i += 1;
                }
                _ => break,
            }
        }
        let mut width = 0usize;
        while let Some(&b) = fmt.get(i) {
            if !b.is_ascii_digit() {
                break;
            }
            width = (width.saturating_mul(10) + (b - b'0') as usize).min(MAX_PRINTF_WIDTH);
            i += 1;
        }
        let mut prec: u32 = 6;
        if fmt.get(i) == Some(&b'.') {
            i += 1;
            prec = 0;
            while let Some(&b) = fmt.get(i) {
                if !b.is_ascii_digit() {
                    break;
                }
                prec = (prec.saturating_mul(10) + (b - b'0') as u32).min(MAX_PRINTF_PREC);
                i += 1;
            }
        }
        ensure!(i < fmt.len(), SyntaxError);
        let conv = fmt[i];
        i += 1;
        // An unsupported conversion character is checked before whether there are enough actual
        // arguments (`%x` should be `UnsupportedFeature` no matter how many arguments are supplied).
        ensure!(matches!(conv, b'd' | b'f' | b's'), UnsupportedFeature);
        ensure!(ai < a.n(), WrongArgCount);
        let (v, row) = match a.at(ai) {
            Some(x) => x,
            None => err!(Internal),
        };
        ai += 1;
        let mut body = Vec::new();
        match conv {
            b'd' => {
                let x = numeric_i128(v, row)?;
                kernels::fmt_int(x.unsigned_abs(), x < 0, 0, &mut body);
            }
            b'f' => {
                let x = numeric_f64(v, row)?;
                fmt_fixed(x, prec as u8, &mut body);
            }
            b's' => write_display(v, row, &mut body)?,
            _ => err!(UnsupportedFeature),
        }
        pad_field(out, &body, width, left, zero && conv != b's');
    }
    Ok(())
}

/// `format(fmt, args...)`. Only Python-style `{}`/`{<n>}` placeholders are supported (a format
/// mini-language such as `{:.2f}` is unsupported: `UnsupportedFeature`).
/// `{{`/`}}` become a literal `{`/`}` respectively (confirmed that
/// `duckdb -c "select format('{{literal}}')"` gives `{literal}`).
/// Values are stringified with the same `write_display` as printf's `%s`
/// (`format` has no per-type specifiers, so it is always this one).
///
/// `{}` consumes actual arguments in order, and `{<n>}` is an explicit 0-based index (counted
/// independently of the automatic numbering). A specifier pointing at a missing argument gives `WrongArgCount`.
fn format_scan(fmt: &[u8], a: &A, out: &mut Vec<u8>) -> Result<()> {
    let mut auto_idx = 0usize;
    let mut i = 0usize;
    while i < fmt.len() {
        match fmt[i] {
            b'{' if fmt.get(i + 1) == Some(&b'{') => {
                out.push(b'{');
                i += 2;
            }
            b'{' => {
                i += 1;
                let start = i;
                while i < fmt.len() && fmt[i] != b'}' {
                    ensure!(fmt[i].is_ascii_digit(), UnsupportedFeature);
                    i += 1;
                }
                ensure!(i < fmt.len(), SyntaxError);
                let idx = if start == i {
                    let cur = auto_idx;
                    auto_idx += 1;
                    cur
                } else {
                    parse_format_index(&fmt[start..i])
                };
                i += 1; // '}'
                let ai = idx.saturating_add(1); // args[0] is fmt itself
                ensure!(ai < a.n(), WrongArgCount);
                let (v, row) = match a.at(ai) {
                    Some(x) => x,
                    None => err!(Internal),
                };
                write_display(v, row, out)?;
            }
            b'}' if fmt.get(i + 1) == Some(&b'}') => {
                out.push(b'}');
                i += 2;
            }
            b'}' => err!(SyntaxError),
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    Ok(())
}

/// Reads the index part of `{<digits>}`. Overflow saturates (the later `ai < a.n()` check gives
/// `WrongArgCount` anyway, so overflow need not be worried about at parse time).
fn parse_format_index(digits: &[u8]) -> usize {
    let mut v: usize = 0;
    for &b in digits {
        v = v.saturating_mul(10).saturating_add((b - b'0') as usize);
    }
    v
}

/// DECIMAL's scale, or 0 for every other type. `%d`/`%f` have to divide it out: the physical
/// value of `1.5::DECIMAL(3,1)` is the integer 15.
///
/// INTERVAL is rejected outright instead: it shares I128 with HUGEINT but its physical value is
/// three packed fields, not a number (DuckDB rejects `%d`/`%f` on an interval too). `%s`/`{}`
/// render it properly through `write_display`.
fn scale_of(v: &Vector) -> Result<u8> {
    Ok(match v.ty() {
        Ty::Decimal { scale, .. } => scale,
        Ty::Interval => err!(TypeMismatch),
        _ => 0,
    })
}

/// `%d`. BOOLEAN and every integer type (HUGEINT included) are supported; DECIMAL is truncated
/// toward zero, and so is floating point (Rust's `f64 as i128` merely saturates out of range and
/// does not panic). The result is i128 so HUGEINT keeps its full range.
fn numeric_i128(v: &Vector, row: usize) -> Result<i128> {
    let s = scale_of(v)?;
    let x = match v.data() {
        Data::Bool(b) => b.get(row) as i128,
        Data::I32(d) => d[row] as i128,
        Data::I64(d) => d[row] as i128,
        Data::I128(d) => d[row],
        Data::F64(d) => d[row] as i128,
        Data::Bytes(_) => err!(TypeMismatch),
    };
    Ok(if s > 0 { x / pow10_i128(s) } else { x })
}

/// `%f`. BOOLEAN, every integer type, DECIMAL, and floating point are supported.
fn numeric_f64(v: &Vector, row: usize) -> Result<f64> {
    let s = scale_of(v)?;
    let x = match v.data() {
        Data::Bool(b) => b.get(row) as i64 as f64,
        Data::I32(d) => d[row] as f64,
        Data::I64(d) => d[row] as f64,
        Data::I128(d) => d[row] as f64,
        Data::F64(d) => d[row],
        Data::Bytes(_) => err!(TypeMismatch),
    };
    // DuckDB also routes DECIMAL through a double for `%f`, so the f64 division matches it
    // (`printf('%.2f', 1234567890123456789.5::DECIMAL(20,1))` loses the same digits there).
    Ok(if s > 0 { x / pow10_i128(s) as f64 } else { x })
}

/// `10^k` as an integer, for DECIMAL scales (which never exceed 38).
fn pow10_i128(k: u8) -> i128 {
    let mut r: i128 = 1;
    for _ in 0..k.min(38) {
        r *= 10;
    }
    r
}

/// The general stringification used by `%s` (printf) and `{}` (format).
///
/// VARCHAR/JSON/BLOB are emitted verbatim; every other type is rendered exactly the way
/// `CAST(x AS VARCHAR)` renders it, by running that very cast (`kernels::cast`) on the one row.
/// Going through the cast rather than re-deriving the text here is what keeps the two spellings
/// in step, and it is what fixes the logical types whose physical value used to leak out:
/// `printf('%s', DATE '2024-01-01')` printed `19723`, `format('{}', TIME '10:00:00')` printed
/// `36000000000`, and DECIMAL dropped its decimal point (`1.5::DECIMAL(3,1)` -> `15`).
/// HUGEINT and INTERVAL, which used to be rejected outright, come along for free.
fn write_display(v: &Vector, row: usize, out: &mut Vec<u8>) -> Result<()> {
    ensure!(row < v.len(), Internal);
    let ty = v.ty();
    if matches!(ty, Ty::Varchar | Ty::Json | Ty::Blob) {
        out.extend_from_slice(v.bytes().get(row));
        return Ok(());
    }
    let text = kernels::cast(ty, Ty::Varchar, &v.gather(&[row as u32]))?;
    if text.is_valid(0) {
        out.extend_from_slice(text.bytes().get(0));
    }
    Ok(())
}

/// One limb of the little-endian base-10^9 magnitudes `fmt_fixed` works in.
const BIG_BASE: u64 = 1_000_000_000;

/// Multiplies a base-10^9 magnitude in place by a small factor (only 2, 5 and 10 are used).
fn big_mul_small(v: &mut Vec<u32>, m: u32) {
    let mut carry: u64 = 0;
    for limb in v.iter_mut() {
        let t = *limb as u64 * m as u64 + carry;
        *limb = (t % BIG_BASE) as u32;
        carry = t / BIG_BASE;
    }
    while carry > 0 {
        v.push((carry % BIG_BASE) as u32);
        carry /= BIG_BASE;
    }
}

/// Divides a base-10^9 magnitude in place by ten, returning the decimal digit dropped.
fn big_div10(v: &mut Vec<u32>) -> u8 {
    let mut rem: u64 = 0;
    for limb in v.iter_mut().rev() {
        let cur = rem * BIG_BASE + *limb as u64;
        *limb = (cur / 10) as u32;
        rem = cur % 10;
    }
    while v.len() > 1 && v[v.len() - 1] == 0 {
        v.pop();
    }
    rem as u8
}

/// Adds one to a base-10^9 magnitude in place (the carry of a round-up).
fn big_inc(v: &mut Vec<u32>) {
    for limb in v.iter_mut() {
        *limb += 1;
        if (*limb as u64) < BIG_BASE {
            return;
        }
        *limb = 0;
    }
    v.push(1);
}

/// `%f`'s fixed-point notation, correctly rounded.
///
/// The value is expanded exactly: an f64 is `mant * 2^e2` with `mant` a 53-bit integer, so
/// `|x| * 10^prec` is `mant * 2^e2 * 10^prec`, which is a whole number once the negative powers
/// of two are turned into powers of five (`2^-k = 5^k / 10^k`). Only when `prec` is smaller than
/// `k` does anything have to be discarded, and that single rounding is round-half-to-even on the
/// true binary value -- the rule C's `printf` follows.
///
/// The previous implementation formed `|x| * 10^prec` in f64 and printed the resulting integer,
/// which reported whatever noise that product carried: `printf('%f', 1e20)` came out as
/// `100000000000000004764.729344`, `printf('%.0f', 2.5)` as `3` (rounding halves away from zero
/// rather than to even), and `printf('%.20f', 0.1)` as an f64-shaped `0.10000000000000000000`
/// instead of the true `0.10000000000000000555`.
fn fmt_fixed(x: f64, prec: u8, out: &mut Vec<u8>) {
    if x.is_nan() {
        out.extend_from_slice(b"nan");
        return;
    }
    let neg = x.is_sign_negative();
    let ax = f_abs(x);
    if ax.is_infinite() {
        if neg {
            out.push(b'-');
        }
        out.extend_from_slice(b"inf");
        return;
    }
    let bits = ax.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & 0x000f_ffff_ffff_ffff;
    // Subnormals have no implicit leading one and a fixed exponent.
    let (mant, e2) = if biased == 0 { (frac, -1074) } else { (frac | (1u64 << 52), biased - 1075) };
    let mut v = vec![(mant % BIG_BASE) as u32, (mant / BIG_BASE) as u32];
    while v.len() > 1 && v[v.len() - 1] == 0 {
        v.pop();
    }
    let p = prec as i32;
    if e2 >= 0 {
        // A whole number already; scaling by 10^prec only appends zeros.
        for _ in 0..e2 {
            big_mul_small(&mut v, 2);
        }
        for _ in 0..p {
            big_mul_small(&mut v, 10);
        }
    } else {
        let k = -e2;
        for _ in 0..k {
            big_mul_small(&mut v, 5);
        }
        // `v` now holds |x| * 10^k exactly.
        if p >= k {
            for _ in 0..(p - k) {
                big_mul_small(&mut v, 10);
            }
        } else {
            // Discard k - p digits, remembering the highest one dropped (the guard) and whether
            // anything below it was non-zero (the sticky bit), then round half to even.
            let mut guard = 0u8;
            let mut sticky = false;
            for _ in 0..(k - p) {
                sticky |= guard != 0;
                guard = big_div10(&mut v);
            }
            if guard > 5 || (guard == 5 && (sticky || v[0] % 2 == 1)) {
                big_inc(&mut v);
            }
        }
    }
    write_scaled(&v, neg, prec as usize, out);
}

/// Writes a base-10^9 magnitude as decimal with a point `prec` digits from the right.
fn write_scaled(v: &[u32], neg: bool, prec: usize, out: &mut Vec<u8>) {
    let mut digits: Vec<u8> = Vec::with_capacity(v.len() * 9);
    let mut top = v[v.len() - 1];
    let mut buf = [0u8; 9];
    let mut n = 0usize;
    while top > 0 {
        buf[n] = b'0' + (top % 10) as u8;
        top /= 10;
        n += 1;
    }
    if n == 0 {
        digits.push(b'0');
    }
    for i in (0..n).rev() {
        digits.push(buf[i]);
    }
    for &limb in v[..v.len() - 1].iter().rev() {
        let mut w = limb;
        let mut b = [0u8; 9];
        for slot in b.iter_mut().rev() {
            *slot = b'0' + (w % 10) as u8;
            w /= 10;
        }
        digits.extend_from_slice(&b);
    }
    if neg {
        out.push(b'-');
    }
    if digits.len() > prec {
        out.extend_from_slice(&digits[..digits.len() - prec]);
    } else {
        out.push(b'0');
    }
    if prec > 0 {
        out.push(b'.');
        for _ in digits.len()..prec {
            out.push(b'0');
        }
        out.extend_from_slice(&digits[digits.len().saturating_sub(prec)..]);
    }
}

/// Applies width, zero padding, and left alignment (shared by `%d`/`%f`/`%s`). `body` is the
/// converted content (including the sign). `%s` can be given multi-byte characters, so the width is
/// counted in code points (the same judgment as `lpad`/`rpad`).
fn pad_field(out: &mut Vec<u8>, body: &[u8], width: usize, left: bool, zero: bool) {
    let len = cp_count(body);
    if len >= width {
        out.extend_from_slice(body);
        return;
    }
    let pad_n = width - len;
    if left {
        out.extend_from_slice(body);
        for _ in 0..pad_n {
            out.push(b' ');
        }
    } else if zero {
        // The sign is kept before the zeros are packed in (giving forms like `-0003`).
        let (sign, rest): (&[u8], &[u8]) = match body.first() {
            Some(b'-') | Some(b'+') => (&body[..1], &body[1..]),
            _ => (&body[..0], body),
        };
        out.extend_from_slice(sign);
        for _ in 0..pad_n {
            out.push(b'0');
        }
        out.extend_from_slice(rest);
    } else {
        for _ in 0..pad_n {
            out.push(b' ');
        }
        out.extend_from_slice(body);
    }
}
