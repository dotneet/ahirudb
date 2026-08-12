//! Strings (Bytes output)
use super::datetime::strftime;
use super::json::{json_extract_or_whole, write_json_scalar};
use super::numeric::{pow10, round_half_up};
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
            let cap = s.len() as i64 + 1;
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
            if d.is_empty() {
                if k == 1 || k == -1 {
                    out.extend_from_slice(s);
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
            ensure!(k.saturating_mul(s.len() as i64) <= MAX_STR, LimitExceeded);
            for _ in 0..k.max(0) {
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

/// `printf`'s maximum precision (the `N` of `%.<N>f`). `kernels::fmt_int`'s internal buffer is only
/// 48 bytes, so passing a scale much larger than this would index the buffer out of range (=
/// panic). 32 leaves ample safety margin.
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
                let x = numeric_i64(v, row)?;
                kernels::fmt_int(x.unsigned_abs() as u128, x < 0, 0, &mut body);
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

/// `%d`. Only BOOLEAN and integer types are supported. Floating point is truncated toward zero
/// (Rust's `f64 as i64` merely saturates out of range and does not panic).
fn numeric_i64(v: &Vector, row: usize) -> Result<i64> {
    Ok(match v.data() {
        Data::Bool(b) => b.get(row) as i64,
        Data::I32(d) => d[row] as i64,
        Data::I64(d) => d[row],
        Data::F64(d) => d[row] as i64,
        Data::I128(_) | Data::Bytes(_) => err!(TypeMismatch),
    })
}

/// `%f`. BOOLEAN, integer types, and floating point are supported.
fn numeric_f64(v: &Vector, row: usize) -> Result<f64> {
    Ok(match v.data() {
        Data::Bool(b) => b.get(row) as i64 as f64,
        Data::I32(d) => d[row] as f64,
        Data::I64(d) => d[row] as f64,
        Data::F64(d) => d[row],
        Data::I128(_) | Data::Bytes(_) => err!(TypeMismatch),
    })
}

/// The general stringification used by `%s` (printf) and `{}` (format). The supported physical
/// types are BOOLEAN, integers (I32/I64), floating point (F64), and byte sequences (VARCHAR/JSON
/// are emitted as is). DATE/TIME/TIMESTAMP/DECIMAL/INTERVAL/HUGEINT (physical type I128, or
/// DECIMAL's scale) merely emit the internal representation as a number, with no calendar or
/// decimal-point conversion (a design decision to narrow the scope. If a calendar rendering is
/// needed, the caller should `CAST(.. AS VARCHAR)` first).
fn write_display(v: &Vector, row: usize, out: &mut Vec<u8>) -> Result<()> {
    match v.data() {
        Data::Bool(b) => out.extend_from_slice(if b.get(row) { b"true" } else { b"false" }),
        Data::I32(d) => kernels::fmt_int(d[row].unsigned_abs() as u128, d[row] < 0, 0, out),
        Data::I64(d) => kernels::fmt_int(d[row].unsigned_abs() as u128, d[row] < 0, 0, out),
        Data::F64(d) => kernels::fmt_f64(d[row], out),
        Data::Bytes(b) => out.extend_from_slice(b.get(row)),
        Data::I128(_) => err!(UnsupportedFeature),
    }
    Ok(())
}

/// `%f`'s fixed-point notation. It multiplies by `10^prec`, turns it into an integer by rounding
/// away from zero, and hands it to `kernels::fmt_int` (which takes care of inserting the decimal point).
/// `prec` is already clamped to `MAX_PRINTF_PREC` by the caller (`printf_scan`), so `fmt_int`'s
/// internal 48-byte buffer cannot overflow.
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
    let scaled = round_half_up(ax * pow10(prec as u32));
    // A value too large to fit u128 gives up and falls back to a simple rendering
    // (being within u128's range is confirmed before handing it to `fmt_int`).
    if scaled >= 1.0e33 {
        if neg {
            out.push(b'-');
        }
        kernels::fmt_f64(ax, out);
        return;
    }
    kernels::fmt_int(scaled as u128, neg, prec, out);
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
