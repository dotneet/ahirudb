//! Shared JSON foundation: tokenizer, path operators, and serialization.
//!
//! The physical representation of `Ty::Json` is UTF-8 JSON text (see the docs in `vector::types`).
//! Rather than reparsing that value, this "skips along in place and extracts only
//! the part wanted", which avoids adding nested data structures (LIST/STRUCT-like
//! physical types) just to build a DOM.
//!
//! ## Relationship to `format::jsonl`
//!
//! `format::jsonl` reads NDJSON one object per line and assembles columnar data; its
//! key/value traversal (`Members`/`Member`) is specialized for that use case. What
//! this module needs instead is navigation that "skips partway into a value and
//! extracts one particular subtree". Only the **tokenizer itself** below those
//! higher-level iterators -- skipping whitespace, strings, numbers, scalars, and any
//! single value, plus escape expansion -- is entirely common to both. So only the
//! tokenizer lives here, and `format::jsonl` `use`s it (the NDJSON-specific parts --
//! `Members`/`Member`, the type lattice for schema inference, the date parser, and so
//! on -- stay in `format::jsonl` as they were).
//!
//! ## The input is untrusted
//!
//! Broken JSON yields `Err` (under no_std/panic=abort there are no panics and no
//! out-of-bounds accesses). Path traversal does carry a design simplification: it
//! validates **only the parts it visits** (see "Known limitations" below).
//!
//! ## Path syntax (what is supported)
//!
//! DuckDB's `$.a.b[0]` form, implemented straightforwardly, plus the ability to omit
//! the leading `$` (`a.b[0]` means the same; a deliberate simplification that is not DuckDB-compatible).
//!
//! - `$` or omitted: the whole root
//! - `.key` / a bare `key`: an object member
//! - `."quoted key"`: for keys containing `.`/`[`. Only `\"` and `\\` escapes are supported
//!   (a simplification: the full JSON string escape set, `\uXXXX` and friends, is not supported)
//! - `[N]`: an array element. 0-based, and negative counts from the end (as in DuckDB)
//! - any number of the above can be chained (`$.a[0].b` and so on)
//!
//! **Unsupported**: JSON Pointer notation (`/a/b/0`).
//!
//! ## Known limitations
//!
//! - Path traversal only validates structure as far as "the siblings along the path
//!   walked". Siblings following the target value, or the contents of subtrees never
//!   opened, may be broken without being detected (`whole`/`validate`, by contrast,
//!   validate the whole document; CAST-time validation uses those).
//! - No equality comparison of objects (`=`) and no key-order normalization
//!   (the caller, `expr::kernels::compare`, merely compares bytes).

use crate::prelude::*;

/// The nesting limit for values. Same value and same reason as `format::jsonl`
/// (the kinds of open containers fit in a `u32` bit stack).
pub(crate) const MAX_DEPTH: u32 = 32;

/// The kind of a JSON value.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub(crate) enum Kind {
    Null,
    Bool,
    Num,
    Str,
    Object,
    Array,
}

pub(crate) fn kind_of(c: u8) -> Kind {
    match c {
        b'{' => Kind::Object,
        b'[' => Kind::Array,
        b'"' => Kind::Str,
        b't' | b'f' => Kind::Bool,
        b'n' => Kind::Null,
        _ => Kind::Num,
    }
}

// =========================================================================
// The tokenizer (the lower layer shared with `format::jsonl`)
// =========================================================================

pub(crate) fn byte_at(b: &[u8], i: usize) -> Result<u8> {
    match b.get(i) {
        Some(&c) => Ok(c),
        None => err!(UnexpectedEof, i),
    }
}

pub(crate) fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while matches!(b.get(i), Some(b' ') | Some(b'\t') | Some(b'\r') | Some(b'\n')) {
        i += 1;
    }
    i
}

/// `b[i]` is `"`. Returns `(body, whether escaped, the position after the closing quote)`.
pub(crate) fn scan_string(b: &[u8], i: usize) -> Result<(&[u8], bool, usize)> {
    let mut j = i + 1;
    let mut esc = false;
    loop {
        match byte_at(b, j)? {
            b'"' => {
                ensure!(core::str::from_utf8(&b[i + 1..j]).is_ok(), SyntaxError, i);
                return Ok((&b[i + 1..j], esc, j + 1));
            }
            b'\\' => {
                // Validate the escape while scanning. `skip_value`, used by CAST validation,
                // does not necessarily decode the string later, so accepting an arbitrary
                // escaped byte here would admit invalid JSON such as `"\\q"`.
                match byte_at(b, j + 1)? {
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => j += 2,
                    b'u' => {
                        hex4(b, j + 2)?;
                        j += 6;
                    }
                    _ => err!(SyntaxError, j + 1),
                }
                esc = true;
            }
            0x00..=0x1f => err!(SyntaxError, j),
            _ => j += 1,
        }
    }
}

pub(crate) fn scan_number(b: &[u8], start: usize) -> Result<usize> {
    let mut i = start;
    if b.get(i) == Some(&b'-') {
        i += 1;
    }
    let d0 = i;
    if b.get(i) == Some(&b'0') {
        i += 1;
        ensure!(!matches!(b.get(i), Some(c) if c.is_ascii_digit()), SyntaxError, i);
    } else {
        while matches!(b.get(i), Some(c) if c.is_ascii_digit()) {
            i += 1;
        }
    }
    ensure!(i > d0, SyntaxError, start);
    if b.get(i) == Some(&b'.') {
        i += 1;
        let f0 = i;
        while matches!(b.get(i), Some(c) if c.is_ascii_digit()) {
            i += 1;
        }
        ensure!(i > f0, SyntaxError, start);
    }
    if matches!(b.get(i), Some(b'e') | Some(b'E')) {
        i += 1;
        if matches!(b.get(i), Some(b'+') | Some(b'-')) {
            i += 1;
        }
        let e0 = i;
        while matches!(b.get(i), Some(c) if c.is_ascii_digit()) {
            i += 1;
        }
        ensure!(i > e0, SyntaxError, start);
    }
    Ok(i)
}

pub(crate) fn skip_lit(b: &[u8], i: usize, w: &[u8]) -> Result<usize> {
    let e = i + w.len();
    ensure!(b.len() >= e, UnexpectedEof, i);
    ensure!(&b[i..e] == w, SyntaxError, i);
    Ok(e)
}

pub(crate) fn skip_scalar(b: &[u8], i: usize) -> Result<usize> {
    match byte_at(b, i)? {
        b'"' => Ok(scan_string(b, i)?.2),
        b't' => skip_lit(b, i, b"true"),
        b'f' => skip_lit(b, i, b"false"),
        b'n' => skip_lit(b, i, b"null"),
        _ => scan_number(b, i),
    }
}

/// Skips `"key" :` and returns the position where the value starts.
pub(crate) fn skip_member_key(b: &[u8], i: usize) -> Result<usize> {
    let i = skip_ws(b, i);
    ensure!(byte_at(b, i)? == b'"', SyntaxError, i);
    let (_, _, ni) = scan_string(b, i)?;
    let ni = skip_ws(b, ni);
    ensure!(byte_at(b, ni)? == b':', SyntaxError, ni);
    Ok(ni + 1)
}

/// Skips one value starting at `b[i]` and returns the position just after it.
///
/// Non-recursive. The kinds of open containers live in a `u32` bit stack, and depth
/// is capped by `MAX_DEPTH` (kept equal to the bit width of `u32`).
pub(crate) fn skip_value(b: &[u8], start: usize) -> Result<usize> {
    let mut i = start;
    // Bit 1 = object, 0 = array. The lowest bit is the current container.
    let mut stack: u32 = 0;
    let mut depth: u32 = 0;

    'value: loop {
        i = skip_ws(b, i);
        let c = byte_at(b, i)?;
        if c == b'{' || c == b'[' {
            let obj = c == b'{';
            ensure!(depth < MAX_DEPTH, NestingTooDeep, i);
            stack = (stack << 1) | obj as u32;
            depth += 1;
            i = skip_ws(b, i + 1);
            let n = byte_at(b, i)?;
            if (obj && n == b'}') || (!obj && n == b']') {
                // An empty container. Falls through to the closing logic below as though one value had been consumed.
                i += 1;
                stack >>= 1;
                depth -= 1;
            } else {
                if obj {
                    i = skip_member_key(b, i)?;
                }
                continue 'value;
            }
        } else {
            i = skip_scalar(b, i)?;
        }

        // One value consumed. Handle the separator or closing bracket.
        loop {
            if depth == 0 {
                return Ok(i);
            }
            i = skip_ws(b, i);
            let c = byte_at(b, i)?;
            let obj = stack & 1 == 1;
            match c {
                b',' => {
                    i += 1;
                    if obj {
                        i = skip_member_key(b, skip_ws(b, i))?;
                    }
                    continue 'value;
                }
                b'}' if obj => {
                    i += 1;
                    stack >>= 1;
                    depth -= 1;
                }
                b']' if !obj => {
                    i += 1;
                    stack >>= 1;
                    depth -= 1;
                }
                _ => err!(SyntaxError, i),
            }
        }
    }
}

pub(crate) fn hex4(b: &[u8], i: usize) -> Result<u32> {
    let mut v = 0u32;
    for k in 0..4 {
        let c = byte_at(b, i + k)?;
        let d = match c {
            b'0'..=b'9' => (c - b'0') as u32,
            b'a'..=b'f' => (c - b'a' + 10) as u32,
            b'A'..=b'F' => (c - b'A' + 10) as u32,
            _ => err!(SyntaxError, i + k),
        };
        v = (v << 4) | d;
    }
    Ok(v)
}

/// Expands the escapes in a string body and writes it to `out` as UTF-8.
pub(crate) fn decode_string(body: &[u8], out: &mut Vec<u8>) -> Result<()> {
    let mut i = 0;
    while i < body.len() {
        let c = body[i];
        if c != b'\\' {
            out.push(c);
            i += 1;
            continue;
        }
        i += 1;
        let e = byte_at(body, i)?;
        i += 1;
        match e {
            b'"' => out.push(b'"'),
            b'\\' => out.push(b'\\'),
            b'/' => out.push(b'/'),
            b'b' => out.push(0x08),
            b'f' => out.push(0x0c),
            b'n' => out.push(b'\n'),
            b'r' => out.push(b'\r'),
            b't' => out.push(b'\t'),
            b'u' => {
                let hi = hex4(body, i)?;
                i += 4;
                let cp = if (0xD800..0xDC00).contains(&hi) {
                    // A high surrogate. Combine if a low surrogate follows immediately.
                    match (body.get(i), body.get(i + 1)) {
                        (Some(b'\\'), Some(b'u')) => match hex4(body, i + 2) {
                            Ok(lo) if (0xDC00..0xE000).contains(&lo) => {
                                i += 6;
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            }
                            // An unpaired high surrogate is collapsed to U+FFFD rather
                            // than made an error (losing a whole row to broken input is worse).
                            _ => 0xFFFD,
                        },
                        _ => 0xFFFD,
                    }
                } else if (0xDC00..0xE000).contains(&hi) {
                    // A lone low surrogate.
                    0xFFFD
                } else {
                    hi
                };
                let ch = char::from_u32(cp).unwrap_or(char::REPLACEMENT_CHARACTER);
                let mut buf = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
            _ => err!(SyntaxError, i),
        }
    }
    Ok(())
}

// =========================================================================
// Whole-document validation (for CAST)
// =========================================================================

/// Allowing leading and trailing whitespace, validates that the input consists of
/// exactly one JSON value and returns its span and kind. Used by the path-less
/// `json_type`/`json_array_length` and by `VARCHAR -> JSON` cast validation (the path
/// that validates **the whole document**; stricter than path-based navigation).
pub(crate) fn whole(doc: &[u8]) -> Result<(&[u8], Kind)> {
    let start = skip_ws(doc, 0);
    let end = skip_value(doc, start)?;
    ensure!(skip_ws(doc, end) == doc.len(), SyntaxError, end);
    let kind = kind_of(byte_at(doc, start)?);
    Ok((&doc[start..end], kind))
}

/// Only checks whether `doc` is exactly one piece of valid JSON text.
pub(crate) fn validate(doc: &[u8]) -> Result<()> {
    whole(doc)?;
    Ok(())
}

// =========================================================================
// Paths
// =========================================================================

enum Seg {
    Key(Vec<u8>),
    Index(i64),
}

/// Turns a path string into a sequence of segments. See "Path syntax" in the module docs.
fn parse_path(p: &[u8]) -> Result<Vec<Seg>> {
    let mut i = if p.first() == Some(&b'$') { 1 } else { 0 };
    let mut segs = Vec::new();
    while i < p.len() {
        match p[i] {
            b'.' => {
                i += 1;
                if p.get(i) == Some(&b'"') {
                    let (body, esc, ni) = scan_string(p, i)?;
                    let key = if esc {
                        let mut buf = Vec::new();
                        decode_string(body, &mut buf)?;
                        buf
                    } else {
                        body.to_vec()
                    };
                    segs.push(Seg::Key(key));
                    i = ni;
                } else {
                    let start = i;
                    while i < p.len() && !matches!(p[i], b'.' | b'[') {
                        i += 1;
                    }
                    ensure!(i > start, SyntaxError, start);
                    segs.push(Seg::Key(p[start..i].to_vec()));
                }
            }
            b'[' => {
                i += 1;
                let neg = p.get(i) == Some(&b'-');
                let dstart = if neg { i + 1 } else { i };
                let mut j = dstart;
                while j < p.len() && p[j].is_ascii_digit() {
                    j += 1;
                }
                ensure!(j > dstart, SyntaxError, i);
                ensure!(p.get(j) == Some(&b']'), SyntaxError, j);
                let mut n: i64 = 0;
                for &d in &p[dstart..j] {
                    n = n.saturating_mul(10).saturating_add((d - b'0') as i64);
                }
                if neg {
                    n = -n;
                }
                segs.push(Seg::Index(n));
                i = j + 1;
            }
            _ => {
                // A shorthand path whose first segment starts without `.`/`[` (`a.b[0]`).
                let start = i;
                while i < p.len() && !matches!(p[i], b'.' | b'[') {
                    i += 1;
                }
                ensure!(i > start, SyntaxError, start);
                segs.push(Seg::Key(p[start..i].to_vec()));
            }
        }
    }
    Ok(segs)
}

/// Searches the object at `b[obj_start]` (`{`) for the member matching `want` and
/// returns where its value starts. `None` if not found.
///
/// Members preceding the target key are structurally validated in order to be skipped
/// (broken ones give `Err`). Anything not preceding it is left unvalidated (see
/// "Known limitations" in the module docs).
fn find_member(b: &[u8], obj_start: usize, want: &[u8]) -> Result<Option<usize>> {
    let mut i = skip_ws(b, obj_start + 1);
    if byte_at(b, i)? == b'}' {
        return Ok(None);
    }
    let mut scratch = Vec::new();
    loop {
        ensure!(byte_at(b, i)? == b'"', SyntaxError, i);
        let (key, esc, ni) = scan_string(b, i)?;
        i = skip_ws(b, ni);
        ensure!(byte_at(b, i)? == b':', SyntaxError, i);
        let vs = skip_ws(b, i + 1);
        let matched = if esc {
            scratch.clear();
            decode_string(key, &mut scratch)?;
            scratch == want
        } else {
            key == want
        };
        if matched {
            return Ok(Some(vs));
        }
        let ve = skip_value(b, vs)?;
        i = skip_ws(b, ve);
        match byte_at(b, i)? {
            b',' => i = skip_ws(b, i + 1),
            b'}' => return Ok(None),
            _ => err!(SyntaxError, i),
        }
    }
}

/// The element count of the array at `b[arr_start]` (`[`).
fn count_elements(b: &[u8], arr_start: usize) -> Result<i64> {
    let mut i = skip_ws(b, arr_start + 1);
    if byte_at(b, i)? == b']' {
        return Ok(0);
    }
    let mut n = 0i64;
    loop {
        let ve = skip_value(b, i)?;
        n += 1;
        i = skip_ws(b, ve);
        match byte_at(b, i)? {
            b',' => i = skip_ws(b, i + 1),
            b']' => return Ok(n),
            _ => err!(SyntaxError, i),
        }
    }
}

/// Where element `want` (0-based, negative counting from the end) of the array at
/// `b[arr_start]` (`[`) starts. Out of range gives `None`.
fn nth_element(b: &[u8], arr_start: usize, want: i64) -> Result<Option<usize>> {
    let count = count_elements(b, arr_start)?;
    let real = if want < 0 { count + want } else { want };
    if real < 0 || real >= count {
        return Ok(None);
    }
    let mut i = skip_ws(b, arr_start + 1);
    let mut idx = 0i64;
    loop {
        if idx == real {
            return Ok(Some(i));
        }
        let ve = skip_value(b, i)?;
        idx += 1;
        i = skip_ws(b, ve);
        match byte_at(b, i)? {
            b',' => i = skip_ws(b, i + 1),
            b']' => return Ok(None),
            _ => err!(SyntaxError, i),
        }
    }
}

/// The body of the path-taking forms of
/// `json_extract`/`json_extract_string`/`json_type`/`json_array_length`. An empty
/// `path` (or just `$`) returns the whole document. If a segment's kind disagrees with
/// the actual container, or the key/index is not found, the result is `Ok(None)` (the caller turns it into SQL NULL).
pub(crate) fn extract<'a>(doc: &'a [u8], path: &[u8]) -> Result<Option<(&'a [u8], Kind)>> {
    let segs = parse_path(path)?;
    if segs.is_empty() {
        let (span, kind) = whole(doc)?;
        return Ok(Some((span, kind)));
    }
    let mut vs = skip_ws(doc, 0);
    for seg in &segs {
        let b0 = byte_at(doc, vs)?;
        match seg {
            Seg::Key(k) => {
                if b0 != b'{' {
                    return Ok(None);
                }
                match find_member(doc, vs, k)? {
                    Some(s) => vs = s,
                    None => return Ok(None),
                }
            }
            Seg::Index(idx) => {
                if b0 != b'[' {
                    return Ok(None);
                }
                match nth_element(doc, vs, *idx)? {
                    Some(s) => vs = s,
                    None => return Ok(None),
                }
            }
        }
    }
    let end = skip_value(doc, vs)?;
    let kind = kind_of(byte_at(doc, vs)?);
    Ok(Some((&doc[vs..end], kind)))
}

/// The body of `list_extract`. 1-based, with negatives counting from the end.
/// Out of range gives `None`.
pub(crate) fn list_index(doc: &[u8], idx: i64) -> Result<Option<&[u8]>> {
    let start = skip_ws(doc, 0);
    if idx == 0 || byte_at(doc, start)? != b'[' {
        return Ok(None);
    }
    let zero_based = if idx > 0 { idx - 1 } else { idx };
    match nth_element(doc, start, zero_based)? {
        Some(vs) => {
            let ve = skip_value(doc, vs)?;
            Ok(Some(&doc[vs..ve]))
        }
        None => Ok(None),
    }
}

/// The body of `list_slice`. 1-based, inclusive on both ends, with negatives counting
/// from the end. Unlike `list_index` (`list_extract`), out-of-range does **not** give
/// NULL but clamps (the same rule as DuckDB's `expr[i:j]` syntax; confirmed by
/// `duckdb -c "select [1,2,3,4,5][10:20], [1,2,3,4,5][-10:3]"` returning `[]`/`[1, 2, 3]`
/// -- fully out of range gives an empty array, partly out of range gives what remains).
/// A non-array gives `None` (SQL NULL at the call site).
///
/// The return value is the half-open interval `(lo, hi)` within `doc`. Wrapping
/// `doc[lo..hi]` in `[` `]` gives the result (`lo == hi` means the empty array `[]`).
/// Array elements are contiguous bytes, so rather than re-emitting element by element
/// this just copies the whole interval (`nth_element`/`skip_value` are used only to
/// locate the first and last elements).
pub(crate) fn list_slice(doc: &[u8], start: i64, end: i64) -> Result<Option<(usize, usize)>> {
    let arr_start = skip_ws(doc, 0);
    if byte_at(doc, arr_start)? != b'[' {
        return Ok(None);
    }
    let count = count_elements(doc, arr_start)?;
    // Where an empty array is written (just after `[`, at the end of the whitespace before `]`).
    let empty_at = skip_ws(doc, arr_start + 1);
    if count == 0 {
        return Ok(Some((empty_at, empty_at)));
    }
    // Normalize to a 1-based inclusive interval before clamping. 0 is treated as 1
    // (confirmed by `duckdb -c "select [1,2,3,4,5][0:2]"` returning the same `[1, 2]`
    // as `[1,2,3,4,5][1:2]`). Negatives use `saturating_add` to avoid overflow
    // (guarding against huge negative indices from untrusted input).
    let norm = |v: i64| -> i64 {
        if v == 0 {
            1
        } else if v < 0 {
            count.saturating_add(v).saturating_add(1)
        } else {
            v
        }
    };
    let s = norm(start).max(1);
    let e = norm(end).min(count);
    if s > e {
        return Ok(Some((empty_at, empty_at)));
    }
    // After clamping, s/e always fall within [1, count], so `nth_element` always
    // returns `Some` (converted to 0-based before being passed in).
    let (Some(lo), Some(hi_start)) =
        (nth_element(doc, arr_start, s - 1)?, nth_element(doc, arr_start, e - 1)?)
    else {
        err!(Internal)
    };
    let hi = skip_value(doc, hi_start)?;
    Ok(Some((lo, hi)))
}

/// The body of `map_extract`. A direct member-name lookup (no path syntax).
///
/// Accepts a JSON object (`{"a":1}`) and the Parquet MAP encoding — a JSON
/// array of `{"key":...,"value":...}` pairs. A missing key, or a root that
/// is neither an object nor such an array, gives `None`.
pub(crate) fn map_get<'a>(doc: &'a [u8], key: &[u8]) -> Result<Option<&'a [u8]>> {
    let start = skip_ws(doc, 0);
    match byte_at(doc, start)? {
        b'{' => match find_member(doc, start, key)? {
            Some(vs) => {
                let ve = skip_value(doc, vs)?;
                Ok(Some(&doc[vs..ve]))
            }
            None => Ok(None),
        },
        b'[' => map_get_pairs(doc, key),
        _ => Ok(None),
    }
}

/// Looks up `key` in a Parquet-style MAP: `[{"key":...,"value":...}, ...]`.
fn map_get_pairs<'a>(doc: &'a [u8], key: &[u8]) -> Result<Option<&'a [u8]>> {
    let Some(elems) = array_elements(doc)? else {
        return Ok(None);
    };
    for (span, kind) in elems {
        if kind != Kind::Object {
            continue;
        }
        let obj = skip_ws(span, 0);
        let Some(ks) = find_member(span, obj, b"key")? else {
            continue;
        };
        let ke = skip_value(span, ks)?;
        if !json_key_matches(&span[ks..ke], key)? {
            continue;
        }
        let Some(vs) = find_member(span, obj, b"value")? else {
            return Ok(None);
        };
        let ve = skip_value(span, vs)?;
        return Ok(Some(&span[vs..ve]));
    }
    Ok(None)
}

/// Whether a JSON key token equals the VARCHAR lookup key.
/// String keys are unquoted (and unescaped); numbers/bools/null compare as
/// their raw JSON text so `map_extract(m, '1')` hits a numeric MAP key `1`.
fn json_key_matches(span: &[u8], want: &[u8]) -> Result<bool> {
    let s = skip_ws(span, 0);
    if byte_at(span, s)? == b'"' {
        let (raw, esc, _) = scan_string(span, s)?;
        if esc {
            let mut scratch = Vec::new();
            decode_string(raw, &mut scratch)?;
            Ok(scratch == want)
        } else {
            Ok(raw == want)
        }
    } else {
        let e = skip_value(span, s)?;
        Ok(&span[s..e] == want)
    }
}

/// The body of `json_array_length`. 0 for anything that is not an array (as in DuckDB).
pub(crate) fn array_length(span: &[u8], kind: Kind) -> Result<i64> {
    if kind != Kind::Array {
        return Ok(0);
    }
    let start = skip_ws(span, 0);
    count_elements(span, start)
}

/// The span and kind of one array element.
pub(crate) type Elem<'a> = (&'a [u8], Kind);

/// If `doc` (surrounding whitespace allowed) is an array, returns its elements in order.
/// `None` otherwise (`exec::unnest::Unnest` treats that as "0 rows produced"). This
/// collects every element's span in one pass, avoiding repeating `nth_element`'s walk
/// once per element (O(n^2)).
pub(crate) fn array_elements(doc: &[u8]) -> Result<Option<Vec<Elem<'_>>>> {
    let start = skip_ws(doc, 0);
    if byte_at(doc, start)? != b'[' {
        return Ok(None);
    }
    let mut out = Vec::new();
    let mut i = skip_ws(doc, start + 1);
    if byte_at(doc, i)? == b']' {
        return Ok(Some(out));
    }
    loop {
        let kind = kind_of(byte_at(doc, i)?);
        let ve = skip_value(doc, i)?;
        out.push((&doc[i..ve], kind));
        i = skip_ws(doc, ve);
        match byte_at(doc, i)? {
            b',' => i = skip_ws(doc, i + 1),
            b']' => return Ok(Some(out)),
            _ => err!(SyntaxError, i),
        }
    }
}

/// A JSON number token as an `i64`. With a decimal point or exponent, or out of range,
/// gives `None` (the caller turns it into SQL NULL). The same judgment as
/// `format::jsonl::parse_i64`, but placed on the `json` side specifically for UNNEST's native type recovery.
pub(crate) fn parse_i64(s: &[u8]) -> Option<i64> {
    let (neg, ds) = match s.first() {
        Some(b'-') => (true, &s[1..]),
        _ => (false, s),
    };
    if ds.is_empty() {
        return None;
    }
    // Accumulate on the negative side. That avoids special-casing i64::MIN.
    let mut acc: i64 = 0;
    for &c in ds {
        if !c.is_ascii_digit() {
            return None;
        }
        acc = acc.checked_mul(10)?.checked_sub((c - b'0') as i64)?;
    }
    if neg {
        Some(acc)
    } else {
        acc.checked_neg()
    }
}

/// Floating point is left to core's `str::parse::<f64>()` (dec2flt).
pub(crate) fn parse_f64(s: &[u8]) -> Option<f64> {
    core::str::from_utf8(s).ok()?.parse::<f64>().ok()
}

/// The type name for `json_type`. Matches DuckDB's actual strings, except that
/// DuckDB's behavior of picking UBIGINT/DOUBLE for numbers based on sign and overflow
/// is not reproduced: every integer is simplified to `"BIGINT"` (an additional
/// simplification beyond the module docs; see the tests and docs in `expr::funcs`).
pub(crate) fn type_name(kind: Kind, span: &[u8]) -> &'static str {
    match kind {
        Kind::Null => "NULL",
        Kind::Bool => "BOOLEAN",
        Kind::Str => "VARCHAR",
        Kind::Object => "OBJECT",
        Kind::Array => "ARRAY",
        Kind::Num => {
            if span.iter().any(|&c| c == b'.' || c == b'e' || c == b'E') {
                "DOUBLE"
            } else {
                "BIGINT"
            }
        }
    }
}

/// The stringification for `json_extract_string`. JSON `null` becomes SQL NULL (`false`).
/// Strings have their escapes expanded; everything else has its span written verbatim
/// (numbers are not normalized: `1e3` stays `1e3`. DuckDB returns a normalized numeric
/// form; this is a simplification that does not).
pub(crate) fn write_extracted_text(span: &[u8], kind: Kind, out: &mut Vec<u8>) -> Result<bool> {
    match kind {
        Kind::Null => Ok(false),
        Kind::Str => {
            let body = if span.len() >= 2 { &span[1..span.len() - 1] } else { &[][..] };
            decode_string(body, out)?;
            Ok(true)
        }
        _ => {
            out.extend_from_slice(span);
            Ok(true)
        }
    }
}

// =========================================================================
// Serialization
// =========================================================================

fn hex_digit(n: u8) -> u8 {
    if n < 10 {
        b'0' + n
    } else {
        b'a' + (n - 10)
    }
}

/// Writes bytes as a JSON string literal.
///
/// Valid non-ASCII UTF-8 bytes are embedded as-is. Invalid UTF-8 sequences are
/// replaced with U+FFFD so the resulting JSON document is always valid UTF-8,
/// even when a permissive byte-oriented input format supplied a malformed
/// VARCHAR value.
pub(crate) fn write_json_string(s: &[u8], out: &mut Vec<u8>) {
    out.push(b'"');
    let mut pos = 0usize;
    while pos < s.len() {
        let (valid, consumed, error_len) = match core::str::from_utf8(&s[pos..]) {
            Ok(valid) => (valid.as_bytes(), s.len() - pos, None),
            Err(e) => {
                let n = e.valid_up_to();
                (&s[pos..pos + n], n, e.error_len())
            }
        };
        for &c in valid {
            match c {
                b'"' => out.extend_from_slice(b"\\\""),
                b'\\' => out.extend_from_slice(b"\\\\"),
                0x08 => out.extend_from_slice(b"\\b"),
                0x0c => out.extend_from_slice(b"\\f"),
                b'\n' => out.extend_from_slice(b"\\n"),
                b'\r' => out.extend_from_slice(b"\\r"),
                b'\t' => out.extend_from_slice(b"\\t"),
                0x00..=0x1f => {
                    out.extend_from_slice(b"\\u00");
                    out.push(hex_digit(c >> 4));
                    out.push(hex_digit(c & 0xf));
                }
                _ => out.push(c),
            }
        }
        pos += consumed;
        if pos < s.len() {
            // `error_len` is None only when the remaining bytes are a
            // truncated UTF-8 prefix. All of that prefix is one replacement
            // character; a non-empty error consumes exactly that many bytes.
            out.extend_from_slice("\u{FFFD}".as_bytes());
            pos += error_len.unwrap_or(s.len() - pos).max(1);
        }
    }
    out.push(b'"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{code_of, Code};

    fn ext<'a>(doc: &'a str, path: &str) -> Option<(&'a str, Kind)> {
        extract(doc.as_bytes(), path.as_bytes())
            .unwrap()
            .map(|(s, k)| (core::str::from_utf8(s).unwrap(), k))
    }

    #[test]
    fn extract_dollar_path_matches_duckdb() {
        // duckdb: SELECT json_extract('{"a":{"b":[1,2,3]}}', '$.a.b[1]') -> 2
        assert_eq!(ext(r#"{"a":{"b":[1,2,3]}}"#, "$.a.b[1]").unwrap().0, "2");
        assert_eq!(ext(r#"{"a":1}"#, "$.a").unwrap().0, "1");
        assert_eq!(ext(r#"{"a":1}"#, "$").unwrap().0, r#"{"a":1}"#);
        assert_eq!(ext("[1,2,3]", "$[0]").unwrap().0, "1");
        // duckdb: json_extract('[1,2,3]', '$[-1]') -> 3
        assert_eq!(ext("[1,2,3]", "$[-1]").unwrap().0, "3");
    }

    #[test]
    fn extract_bare_path_is_our_simplification() {
        // In DuckDB 'a.b[1]' is NULL, but here omitting $ is allowed.
        assert_eq!(ext(r#"{"a":{"b":[1,2,3]}}"#, "a.b[1]").unwrap().0, "2");
    }

    #[test]
    fn extract_quoted_key() {
        assert_eq!(ext(r#"{"a b":1}"#, r#"$."a b""#).unwrap().0, "1");
    }

    #[test]
    fn extract_missing_path_is_none() {
        assert!(ext(r#"{"a":1}"#, "$.b").is_none());
        assert!(ext("[1,2,3]", "$[10]").is_none());
        assert!(ext(r#"{"a":1}"#, "$.a.b").is_none());
        assert!(ext("[1,2,3]", "$.a").is_none());
    }

    #[test]
    fn extract_null_value_is_a_real_json_null_not_missing() {
        let (span, kind) = ext(r#"{"a":null}"#, "$.a").unwrap();
        assert_eq!(span, "null");
        assert!(kind == Kind::Null);
    }

    #[test]
    fn malformed_json_is_an_error() {
        assert_eq!(code_of(extract(b"{not json", b"$.a")), Some(Code::SyntaxError));
        assert_eq!(code_of(whole(b"{not json")), Some(Code::SyntaxError));
    }

    #[test]
    fn whole_rejects_trailing_garbage() {
        assert_eq!(code_of(whole(b"{}{}")), Some(Code::SyntaxError));
        assert!(whole(b"  {\"a\":1}  ").is_ok());
    }

    #[test]
    fn list_index_is_one_based_with_negative_from_end() {
        assert_eq!(list_index(b"[10,20,30]", 1).unwrap(), Some(&b"10"[..]));
        assert_eq!(list_index(b"[10,20,30]", -1).unwrap(), Some(&b"30"[..]));
        assert_eq!(list_index(b"[10,20,30]", 0).unwrap(), None);
        assert_eq!(list_index(b"[10,20,30]", 100).unwrap(), None);
        assert_eq!(list_index(b"{\"a\":1}", 1).unwrap(), None);
    }

    fn slice(doc: &str, start: i64, end: i64) -> Option<String> {
        list_slice(doc.as_bytes(), start, end).unwrap().map(|(lo, hi)| {
            let mut out = String::from("[");
            out.push_str(&doc[lo..hi]);
            out.push(']');
            out
        })
    }

    #[test]
    fn list_slice_clamps_out_of_range_instead_of_nulling() {
        // duckdb -c "select [1,2,3,4,5][2:3]" -> [2, 3]
        assert_eq!(slice("[1,2,3,4,5]", 2, 3).as_deref(), Some("[2,3]"));
        // duckdb -c "select [1,2,3,4,5][-2:-1]" -> [4, 5]
        assert_eq!(slice("[1,2,3,4,5]", -2, -1).as_deref(), Some("[4,5]"));
        // duckdb -c "select [1,2,3,4,5][10:20]" -> []  (fully out of range: empty, not NULL)
        assert_eq!(slice("[1,2,3,4,5]", 10, 20).as_deref(), Some("[]"));
        // duckdb -c "select [1,2,3,4,5][-10:3]" -> [1, 2, 3] (start clamps up to 1)
        assert_eq!(slice("[1,2,3,4,5]", -10, 3).as_deref(), Some("[1,2,3]"));
        // duckdb -c "select [1,2,3,4,5][3:1]" -> [] (start past end)
        assert_eq!(slice("[1,2,3,4,5]", 3, 1).as_deref(), Some("[]"));
        // duckdb -c "select [1,2,3,4,5][0:2]" -> [1, 2] (0 behaves like 1)
        assert_eq!(slice("[1,2,3,4,5]", 0, 2).as_deref(), Some("[1,2]"));
        // duckdb -c "select [][1:2]" -> []
        assert_eq!(slice("[]", 1, 2).as_deref(), Some("[]"));
        // Non-array base -> None (caller turns this into SQL NULL).
        assert_eq!(list_slice(b"{\"a\":1}", 1, 2).unwrap(), None);
        // Omitted-bound sentinels used by the parser desugaring.
        // duckdb -c "select [1,2,3,4,5][:3], [1,2,3,4,5][2:]" -> [1,2,3] / [2,3,4,5]
        assert_eq!(slice("[1,2,3,4,5]", 1, 3).as_deref(), Some("[1,2,3]"));
        assert_eq!(slice("[1,2,3,4,5]", 2, i64::MAX).as_deref(), Some("[2,3,4,5]"));
    }

    #[test]
    fn map_get_looks_up_a_direct_member() {
        assert_eq!(map_get(br#"{"a":1,"b":2}"#, b"a").unwrap(), Some(&b"1"[..]));
        assert_eq!(map_get(br#"{"a":1}"#, b"z").unwrap(), None);
        assert_eq!(map_get(b"[1,2]", b"a").unwrap(), None);
    }

    #[test]
    fn map_get_looks_up_parquet_map_pairs() {
        let doc = br#"[{"key":"a","value":0},{"key":"b","value":2},{"key":"c","value":null}]"#;
        assert_eq!(map_get(doc, b"a").unwrap(), Some(&b"0"[..]));
        assert_eq!(map_get(doc, b"b").unwrap(), Some(&b"2"[..]));
        assert_eq!(map_get(doc, b"c").unwrap(), Some(&b"null"[..]));
        assert_eq!(map_get(doc, b"z").unwrap(), None);
        // Numeric MAP keys compare as their JSON text.
        let nums = br#"[{"key":1,"value":"v1"},{"key":2,"value":"v2"}]"#;
        assert_eq!(map_get(nums, b"1").unwrap(), Some(&br#""v1""#[..]));
        assert_eq!(map_get(nums, b"2").unwrap(), Some(&br#""v2""#[..]));
    }

    #[test]
    fn array_length_is_zero_for_non_arrays() {
        assert_eq!(array_length(b"[1,2,3]", Kind::Array).unwrap(), 3);
        assert_eq!(array_length(b"[]", Kind::Array).unwrap(), 0);
        assert_eq!(array_length(b"5", Kind::Num).unwrap(), 0);
        assert_eq!(array_length(br#"{"a":1}"#, Kind::Object).unwrap(), 0);
    }

    #[test]
    fn type_name_matches_duckdb_strings() {
        assert_eq!(type_name(Kind::Object, b"{}"), "OBJECT");
        assert_eq!(type_name(Kind::Array, b"[]"), "ARRAY");
        assert_eq!(type_name(Kind::Str, b"\"x\""), "VARCHAR");
        assert_eq!(type_name(Kind::Bool, b"true"), "BOOLEAN");
        assert_eq!(type_name(Kind::Null, b"null"), "NULL");
        assert_eq!(type_name(Kind::Num, b"1"), "BIGINT");
        assert_eq!(type_name(Kind::Num, b"-1"), "BIGINT");
        assert_eq!(type_name(Kind::Num, b"1.5"), "DOUBLE");
        assert_eq!(type_name(Kind::Num, b"1e3"), "DOUBLE");
    }

    #[test]
    fn write_extracted_text_unquotes_strings_and_nulls_json_null() {
        let mut out = Vec::new();
        assert!(write_extracted_text(b"\"hi\"", Kind::Str, &mut out).unwrap());
        assert_eq!(out, b"hi");
        out.clear();
        assert!(!write_extracted_text(b"null", Kind::Null, &mut out).unwrap());
        out.clear();
        assert!(write_extracted_text(b"1.5", Kind::Num, &mut out).unwrap());
        assert_eq!(out, b"1.5");
    }

    #[test]
    fn write_json_string_escapes_control_chars_and_quotes() {
        let mut out = Vec::new();
        write_json_string(b"a\"b\\c\nd", &mut out);
        assert_eq!(out, b"\"a\\\"b\\\\c\\nd\"");
    }

    #[test]
    fn write_json_string_replaces_invalid_utf8() {
        let mut out = Vec::new();
        write_json_string(b"a\x80\xE2\x82b", &mut out);
        assert_eq!(out, "\"a\u{FFFD}\u{FFFD}b\"".as_bytes());
        assert!(core::str::from_utf8(&out).is_ok());
    }

    #[test]
    fn array_elements_splits_top_level_values_only() {
        let got = array_elements(b"[1,[2,3],\"a,b\",null]").unwrap().unwrap();
        assert_eq!(got.len(), 4);
        assert_eq!(got[0], (&b"1"[..], Kind::Num));
        assert_eq!(got[1], (&b"[2,3]"[..], Kind::Array));
        assert_eq!(got[2], (&b"\"a,b\""[..], Kind::Str));
        assert_eq!(got[3], (&b"null"[..], Kind::Null));
    }

    #[test]
    fn array_elements_empty_array_is_some_empty_vec() {
        assert_eq!(array_elements(b"[]").unwrap(), Some(Vec::new()));
        assert_eq!(array_elements(b"  [ ]  ").unwrap(), Some(Vec::new()));
    }

    #[test]
    fn array_elements_non_array_is_none() {
        assert_eq!(array_elements(b"5").unwrap(), None);
        assert_eq!(array_elements(br#"{"a":1}"#).unwrap(), None);
        assert_eq!(array_elements(b"null").unwrap(), None);
    }

    #[test]
    fn parse_i64_rejects_non_integer_tokens() {
        assert_eq!(parse_i64(b"42"), Some(42));
        assert_eq!(parse_i64(b"-42"), Some(-42));
        assert_eq!(parse_i64(b"4.2"), None);
        assert_eq!(parse_i64(b""), None);
    }

    #[test]
    fn parse_f64_parses_plain_and_exponent_forms() {
        assert_eq!(parse_f64(b"1.5"), Some(1.5));
        assert_eq!(parse_f64(b"1e3"), Some(1000.0));
        assert_eq!(parse_f64(b"nope"), None);
    }

    // --- Boundary values and corrupt input --------------------------------------

    #[test]
    fn nesting_exactly_at_max_depth_succeeds_one_more_fails() {
        // `skip_value` is non-recursive (a `u32` bit stack), so deep nesting cannot
        // overflow the stack, but MAX_DEPTH (32) itself should still act as an
        // explicit limit.
        let mut at_limit = vec![b'['; MAX_DEPTH as usize];
        at_limit.extend(vec![b']'; MAX_DEPTH as usize]);
        assert!(whole(&at_limit).is_ok(), "exactly MAX_DEPTH passes");

        let mut over_limit = vec![b'['; MAX_DEPTH as usize + 1];
        over_limit.extend(vec![b']'; MAX_DEPTH as usize + 1]);
        assert_eq!(code_of(whole(&over_limit)), Some(Code::NestingTooDeep));
    }

    #[test]
    fn very_deeply_nested_input_errors_without_panicking_or_hanging() {
        // Confirm that input many times deeper than MAX_DEPTH does not panic (the
        // implementation is non-recursive) and ends in a plain SyntaxError/NestingTooDeep.
        let deep: Vec<u8> = vec![b'['; 10_000];
        assert!(whole(&deep).is_err());
    }

    #[test]
    fn trailing_comma_is_rejected() {
        assert_eq!(code_of(whole(b"[1,2,]")), Some(Code::SyntaxError));
        assert_eq!(code_of(whole(br#"{"a":1,}"#)), Some(Code::SyntaxError));
    }

    #[test]
    fn missing_closing_bracket_is_unexpected_eof_not_a_panic() {
        assert!(whole(b"[1,2,3").is_err());
        assert!(whole(br#"{"a":1"#).is_err());
        assert!(whole(br#"{"a""#).is_err());
    }

    #[test]
    fn decode_string_reassembles_surrogate_pairs_into_one_codepoint() {
        // U+1F600 becomes a high/low surrogate pair in UTF-16.
        let mut out = Vec::new();
        decode_string(b"\\ud83d\\ude00", &mut out).unwrap();
        assert_eq!(core::str::from_utf8(&out).unwrap(), "\u{1F600}");
    }

    #[test]
    fn decode_string_replaces_unpaired_surrogates_with_replacement_char() {
        // Unpaired high/low surrogates are collapsed to U+FFFD rather than made errors
        // (losing the reads of other fields to broken input is worse).
        let mut out = Vec::new();
        decode_string(b"\\ud83d", &mut out).unwrap();
        assert_eq!(core::str::from_utf8(&out).unwrap(), "\u{FFFD}");
        out.clear();
        decode_string(b"\\udc00", &mut out).unwrap();
        assert_eq!(core::str::from_utf8(&out).unwrap(), "\u{FFFD}");
    }

    #[test]
    fn strings_reject_invalid_json_lexemes() {
        assert_eq!(code_of(whole(b"\"a\x01b\"")), Some(Code::SyntaxError));
        assert_eq!(code_of(whole(br#""a\q""#)), Some(Code::SyntaxError));
        assert_eq!(code_of(whole(br#""a\u12xz""#)), Some(Code::SyntaxError));
        assert_eq!(code_of(whole(b"\"a\xffb\"")), Some(Code::SyntaxError));
    }

    #[test]
    fn numbers_reject_leading_zeroes() {
        assert_eq!(code_of(whole(b"01")), Some(Code::SyntaxError));
        assert_eq!(code_of(whole(b"-01")), Some(Code::SyntaxError));
        assert_eq!(code_of(whole(b"[00]")), Some(Code::SyntaxError));
        assert!(whole(b"0").is_ok());
        assert!(whole(b"-0.1").is_ok());
    }
}
