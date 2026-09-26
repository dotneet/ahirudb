//! Free-standing parsing helpers shared across submodules: GROUPING SETS
//! expansion, literal/type-name lookups, and INTERVAL text parsing.
use super::*;

// --- Extended GROUP BY syntax ------------------------------------------------

/// The column-count cap for `CUBE`. It expands into 2^n grouping sets (= that many
/// `Node::Aggregate` bundled with UNION ALL), so unbounded input would blow up the plan.
const MAX_CUBE_COLS: usize = 8;

/// Expands `ROLLUP (a, b, c)` into `GROUPING SETS ((a,b,c),(a,b),(a),())`.
/// Builds hierarchical subsets from more columns to fewer.
pub(super) fn rollup_sets(cols: Vec<ExprId>) -> Vec<Vec<ExprId>> {
    let mut sets = Vec::with_capacity(cols.len() + 1);
    for k in (0..=cols.len()).rev() {
        sets.push(cols[..k].to_vec());
    }
    sets
}

/// Expands `CUBE (a, b)` into `GROUPING SETS ((a,b),(a),(b),())`.
/// Builds every subset (2^n of them).
pub(super) fn cube_sets(cols: Vec<ExprId>, pos: usize) -> Result<Vec<Vec<ExprId>>> {
    ensure!(cols.len() <= MAX_CUBE_COLS, ExpressionTooDeep, pos);
    let n = cols.len();
    let mut sets = Vec::with_capacity(1usize << n);
    // Earlier columns are assigned higher bits. That yields an order that "prefers to
    // keep the columns nearest the front", as in `(a,b),(a),(b),()`, matching the feel of
    // `ROLLUP`'s hierarchical subset ordering (it has no effect on results: any UNION ALL
    // order gives the same set).
    for mask in (0..(1usize << n)).rev() {
        let mut set = Vec::new();
        for (i, &c) in cols.iter().enumerate() {
            if mask & (1 << (n - 1 - i)) != 0 {
                set.push(c);
            }
        }
        sets.push(set);
    }
    Ok(sets)
}

/// The cap on the number of grouping sets a `GROUP BY` clause expands into, the same
/// bound `MAX_CUBE_COLS` puts on a single `CUBE` (each set becomes one `Node::Aggregate`).
const MAX_GROUPING_SETS: usize = 1 << MAX_CUBE_COLS;

/// The cross product of two lists of grouping sets: every set of `acc` concatenated with
/// every set of `elem`. This is how the comma-separated elements of a `GROUP BY` combine
/// (`GROUP BY b, ROLLUP (a)` = `(b)` x `((a), ())` = `((b, a), (b))`).
pub(super) fn cross_sets(
    acc: &[Vec<ExprId>],
    elem: &[Vec<ExprId>],
    pos: usize,
) -> Result<Vec<Vec<ExprId>>> {
    ensure!(acc.len().saturating_mul(elem.len()) <= MAX_GROUPING_SETS, ExpressionTooDeep, pos);
    let mut out = Vec::with_capacity(acc.len() * elem.len());
    for a in acc {
        for e in elem {
            let mut set = a.clone();
            set.extend_from_slice(e);
            out.push(set);
        }
    }
    Ok(out)
}

// --- Lambdas ------------------------------------------------------------------

/// Whether this function name may interpret a `->` in argument position as a lambda.
///
/// As measured with the duckdb CLI, `->` is interpreted as a lambda only in the argument
/// positions of functions known to take a lambda; in the arguments of other functions
/// (`coalesce` and so on) `->` passes through as the ordinary JSON path operator
/// (`coalesce(doc -> 'a', 'x')` resolves as JSON extraction, while `abs(x -> x+1)` is
/// interpreted as a lambda and errors with "this function does not take a lambda").
/// This implementation reproduces that distinction by keeping the function names as a fixed set.
pub(super) fn is_lambda_func(name: &str) -> bool {
    eq_ascii_ci(name.as_bytes(), b"list_transform")
        || eq_ascii_ci(name.as_bytes(), b"list_filter")
        || eq_ascii_ci(name.as_bytes(), b"list_reduce")
}

/// Returns the corresponding `BinaryOp` if `self.cur` is a comparison operator token.
/// The shared check used by `expr_body`'s infix loop and by `peek_quantifier` to
/// recognize the quantified comparison `x <op> ANY|ALL|SOME (SELECT ...)` (only the six
/// comparison operators may be followed by `ANY`/`ALL`/`SOME`).
pub(super) fn comparison_binop(t: Tok<'_>) -> Option<BinaryOp> {
    match t {
        Tok::Eq => Some(BinaryOp::Eq),
        Tok::Ne => Some(BinaryOp::Ne),
        Tok::Lt => Some(BinaryOp::Lt),
        Tok::Le => Some(BinaryOp::Le),
        Tok::Gt => Some(BinaryOp::Gt),
        Tok::Ge => Some(BinaryOp::Ge),
        _ => None,
    }
}

// --- Literals and type names -------------------------------------------------

/// Expands the contents of a quoted lexeme. It only folds doubled quotes into one.
pub(super) fn unquote(raw: &str, q: u8) -> String {
    let b = raw.as_bytes();
    let mut out = String::new();
    let (mut i, mut start) = (0usize, 0usize);
    while i < b.len() {
        if b[i] == q {
            // Quotes are ASCII, so this range always lands on a character boundary.
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

/// An integer literal. Picks the smallest type that fits (I32 -> I64 -> I128).
///
/// `text` comes straight from the lexer and may carry `_` digit separators
/// (`1_000`); the lexer has already validated their placement, so they are simply
/// skipped here rather than re-checked.
pub(super) fn int_literal(text: &str, negative: bool, pos: usize) -> Result<Value> {
    let mut mag: u128 = 0;
    for &d in text.as_bytes() {
        if d == b'_' {
            continue;
        }
        mag = match mag.checked_mul(10).and_then(|v| v.checked_add((d - b'0') as u128)) {
            Some(v) => v,
            None => err!(NumberOverflow, pos),
        };
    }
    // i128::MIN has an absolute value one greater than i128::MAX. The limit depends on the sign.
    let limit = if negative { 1u128 << 127 } else { (1u128 << 127) - 1 };
    ensure!(mag <= limit, NumberOverflow, pos);
    let v = if negative { (mag as i128).wrapping_neg() } else { mag as i128 };
    // INTEGER is chosen by the unsigned magnitude, as DuckDB does, so
    // `-2147483648` is BIGINT (its magnitude does not fit INTEGER) and
    // `-2147483648 - 1` does not wrap around in 32 bits. From BIGINT upwards
    // DuckDB goes by the signed value: `-9223372036854775808` is BIGINT there
    // too (`duckdb -c "select typeof(-2147483648), typeof(-9223372036854775808)"`
    // -> BIGINT, BIGINT).
    Ok(if mag <= i32::MAX as u128 {
        Value::I32(v as i32)
    } else if let Ok(x) = i64::try_from(v) {
        Value::I64(x)
    } else {
        Value::I128(v)
    })
}

/// A float literal. Like `int_literal`, `text` may carry `_` digit separators
/// (`1_000.5`, `1e1_0`); `f64::from_str` does not accept them, so they are stripped
/// first — but only on that rare path, so the ordinary literal still parses in place.
pub(super) fn float_literal(text: &str, pos: usize) -> Result<Value> {
    let parsed = if text.as_bytes().contains(&b'_') {
        let cleaned: String = text.chars().filter(|&c| c != '_').collect();
        cleaned.parse::<f64>()
    } else {
        text.parse::<f64>()
    };
    match parsed {
        Ok(v) => Ok(Value::F64(v)),
        Err(_) => err!(NumberOverflow, pos),
    }
}

/// The method name of `USING SAMPLE`/`TABLESAMPLE`. `None` when it does not match
/// (the caller then treats it as "something other than a sampling method").
pub(super) fn sample_method_from_ident(word: &[u8]) -> Option<SampleMethod> {
    if eq_ascii_ci(word, b"bernoulli") {
        Some(SampleMethod::Bernoulli)
    } else if eq_ascii_ci(word, b"system") {
        Some(SampleMethod::System)
    } else if eq_ascii_ci(word, b"reservoir") {
        Some(SampleMethod::Reservoir)
    } else {
        None
    }
}

/// The CAST type-name table. CAST is not a hot path, so a linear scan narrowed by
/// (length, first byte) is enough. Unlike the reserved-word table, it has no binary-search ordering constraint.
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
    // A DECIMAL without parentheses is (18,3). The default precision is one that fits in I64.
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
    (b"timestamptz", Ty::Timestamptz),
    (b"json", Ty::Json),
    (b"uuid", Ty::Uuid),
    // INTERVAL is a first-class type here (DESIGN.md §8 gives it its own `I128`
    // physical representation and docs/sql/types.md lists it), so it has to be
    // nameable in a type position: `CREATE TABLE t (x INTERVAL)`,
    // `CAST(NULL AS INTERVAL)`, `x::INTERVAL`. Without this entry `type_name`
    // rejected every one of them with `InvalidCast`, even though
    // `CREATE TABLE t AS SELECT INTERVAL '1 day' AS x ...` produced the type
    // perfectly well.
    (b"interval", Ty::Interval),
];

/// Type names that may prefix a single-quoted string as a **typed literal**
/// (`DATE '2020-01-01'`, `TIMESTAMP '2020-01-01 10:00:00'`, `TIME
/// '10:00:00'`, `TIMESTAMPTZ '2020-01-01 00:00:00+09'`).
///
/// Deliberately narrower than `lookup_type`. duckdb generalises the form to
/// every type name (`duckdb -c "select INTEGER '5'"` works, and its EXPLAIN
/// shows it is literally `CAST('5' AS INTEGER)`), but the four temporal
/// types are the ones that actually need the syntax: they are the only
/// types whose values have no other literal spelling. Restricting the table
/// keeps the "column named `text`/`blob`/... still resolves normally" blast
/// radius small and costs almost nothing in code size.
///
/// These stay out of the reserved-word table (`sql::lexer::KEYWORDS`) —
/// `date`/`time` are extremely common column names, and duckdb does not
/// reserve them either (`duckdb -c "select date, time from (select 1 as
/// date, 2 as time)"` works). The parser only reads them this way when a
/// string literal follows; see `Parser::temporal_literal_or_ident`.
pub(super) fn temporal_literal_ty(name: &[u8]) -> Option<Ty> {
    if eq_ascii_ci(name, b"date") {
        Some(Ty::Date)
    } else if eq_ascii_ci(name, b"time") {
        Some(Ty::Time)
    } else if eq_ascii_ci(name, b"timestamp") {
        Some(Ty::Timestamp)
    } else if eq_ascii_ci(name, b"timestamptz") {
        Some(Ty::Timestamptz)
    } else {
        None
    }
}

pub(super) fn lookup_type(name: &[u8]) -> Option<Ty> {
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

// --- INTERVAL literals -------------------------------------------------------
// The unit words are every `date_part` spelling that names an interval unit (`year`, `mons`,
// `h`, `millennia`, ...); `expr::funcs::add_interval_unit` resolves them and applies DuckDB's
// per-unit rules, so the literal, `CAST(... AS INTERVAL)` and the `INTERVAL n unit` form
// all accept the same words.

/// A signed decimal integer. Surrounding whitespace is allowed (for the numeric pieces of `INTERVAL`).
pub(super) fn parse_signed_int(s: &str) -> Option<i64> {
    let b = s.trim().as_bytes();
    if b.is_empty() {
        return None;
    }
    let (neg, digits) = match b[0] {
        b'-' => (true, &b[1..]),
        b'+' => (false, &b[1..]),
        _ => (false, b),
    };
    if digits.is_empty() || !digits.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let mut v: i64 = 0;
    if neg {
        for &d in digits {
            v = v.checked_mul(10)?.checked_sub((d - b'0') as i64)?;
        }
        Some(v)
    } else {
        for &d in digits {
            v = v.checked_mul(10)?.checked_add((d - b'0') as i64)?;
        }
        Some(v)
    }
}

/// Packs after confirming that `months`/`days` fit in `i32`.
fn pack_interval_checked(acc: [i64; 3], pos: usize) -> Result<i128> {
    match (i32::try_from(acc[0]), i32::try_from(acc[1])) {
        (Ok(m), Ok(d)) => Ok(crate::vector::pack_interval(m, d, acc[2])),
        _ => err!(NumberOverflow, pos),
    }
}

/// `n unit` as an INTERVAL (the `INTERVAL 3 DAY` / `INTERVAL '3' DAY` forms). `None` when
/// `unit` is not an interval unit, so the caller can fall back to another reading.
pub(super) fn unit_to_interval(unit: &str, n: i64, pos: usize) -> Option<Result<i128>> {
    let mut acc = [0i64; 3];
    match crate::expr::funcs::add_interval_unit(unit.as_bytes(), n, 0, &mut acc) {
        Ok(()) => Some(pack_interval_checked(acc, pos)),
        Err(e) if e.code == Code::SyntaxError => None,
        Err(e) => Some(Err(Error::at(e.code, pos))),
    }
}

/// A run of ASCII digits as an `i64`. Empty input, or any non-digit, is `None`.
fn digits_i64(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.is_empty() {
        return None;
    }
    let mut v: i64 = 0;
    for &c in b {
        if !c.is_ascii_digit() {
            return None;
        }
        v = v.checked_mul(10)?.checked_add((c - b'0') as i64)?;
    }
    Some(v)
}

/// A bare time component, `HH:MM[:SS[.frac]]`, in microseconds.
///
/// This is the shape an interval *prints* as, so accepting it is what makes an
/// interval this engine emitted readable back in. DuckDB accepts it inside any
/// INTERVAL string, on its own (`'1:30:00'`, `'01:02'`, `'01:02:03.5'`) or after unit
/// terms (`'1 day 01:02:03'`, `'-2 days -03:04:05'`). The hour field is not wrapped at 24
/// (`'100:00:00'` is 100 hours), but minutes and seconds must be below 60, and the
/// fraction is truncated at microsecond resolution.
fn parse_time_component(s: &str) -> Option<i64> {
    const US_PER_SEC: i64 = 1_000_000;
    let mut it = s.trim_end().split(':');
    let hours = digits_i64(it.next()?)?;
    let minutes = digits_i64(it.next()?)?;
    let (secs, frac_us) = match it.next() {
        None => (0, 0),
        Some(sec_tok) => {
            let (whole, frac) = match sec_tok.split_once('.') {
                Some((w, f)) => (w, Some(f)),
                None => (sec_tok, None),
            };
            let mut us = 0i64;
            if let Some(f) = frac {
                if f.is_empty() {
                    return None;
                }
                // Six digits of resolution; anything finer is truncated away.
                let mut scale = 100_000i64;
                for c in f.bytes() {
                    if !c.is_ascii_digit() {
                        return None;
                    }
                    us += (c - b'0') as i64 * scale;
                    scale /= 10;
                }
            }
            (digits_i64(whole)?, us)
        }
    };
    if it.next().is_some() || minutes >= 60 || secs >= 60 {
        return None;
    }
    hours
        .checked_mul(60 * 60 * US_PER_SEC)?
        .checked_add(minutes * 60 * US_PER_SEC + secs * US_PER_SEC + frac_us)
}

/// INTERVAL text, as DuckDB's `Interval::FromCString` reads it: an optional `@`, then terms
/// `[-]<n>[.<frac>] <unit>` (the space before the unit is optional: `'1h'`, `'1 day1 hour'`),
/// optionally a trailing `HH:MM[:SS[.frac]]` time component (a `-` before it negates the
/// microseconds accumulated so far, as in DuckDB), and optionally a final `ago` that negates
/// everything (`'1 day ago'` is `-1 day`). A bare number is seconds (`'1.5'`). Repeated units
/// accumulate (`'1 month 1 month'` is `2 months`). See `add_interval_unit` for how a fraction
/// cascades.
pub(crate) fn parse_interval_text(text: &str, pos: usize) -> Result<i128> {
    let s = text.as_bytes();
    let space = |c: &u8| matches!(c, b' ' | b'\t' | b'\n');
    let mut acc = [0i64; 3];
    let mut any = false;
    let mut i = usize::from(s.first() == Some(&b'@'));
    loop {
        while s.get(i).is_some_and(space) {
            i += 1;
        }
        let Some(&c) = s.get(i) else { break };
        if c == b'a' || c == b'A' {
            let ago = s.len() - i >= 3 && s[i + 1..i + 3].eq_ignore_ascii_case(b"go");
            ensure!(ago && s[i + 3..].iter().all(space), SyntaxError, pos);
            acc = [-acc[0], -acc[1], -acc[2]];
            break;
        }
        let neg = c == b'-';
        let start = i + usize::from(neg);
        let mut j = start;
        while s.get(j).is_some_and(u8::is_ascii_digit) {
            j += 1;
        }
        if s.get(j) == Some(&b':') {
            // The time component runs to the end of the text.
            let Some(t) = parse_time_component(&text[start..]) else { err!(SyntaxError, pos) };
            let Some(u) = acc[2].checked_add(t) else { err!(NumberOverflow, pos) };
            acc[2] = if neg { -u } else { u };
            any = true;
            break;
        }
        ensure!(j > start, SyntaxError, pos);
        let Some(mut n) = digits_i64(&text[start..j]) else { err!(NumberOverflow, pos) };
        let mut frac = 0i64;
        if s.get(j) == Some(&b'.') {
            j += 1;
            let mut mult = 100_000;
            while let Some(d) = s.get(j).filter(|d| d.is_ascii_digit()) {
                frac += (d - b'0') as i64 * mult;
                mult /= 10;
                j += 1;
            }
        }
        if neg {
            (n, frac) = (-n, -frac);
        }
        while s.get(j).is_some_and(space) {
            j += 1;
        }
        let w = j;
        while s.get(j).is_some_and(u8::is_ascii_alphabetic) {
            j += 1;
        }
        // A bare number is seconds, but only as the whole text.
        let unit = if j == w {
            ensure!(!any && j == s.len(), SyntaxError, pos);
            "second"
        } else {
            &text[w..j]
        };
        if let Err(e) = crate::expr::funcs::add_interval_unit(unit.as_bytes(), n, frac, &mut acc) {
            return Err(Error::at(e.code, pos));
        }
        any = true;
        i = j;
    }
    ensure!(any, SyntaxError, pos);
    pack_interval_checked(acc, pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_signed_int_boundaries() {
        assert_eq!(parse_signed_int("0"), Some(0));
        assert_eq!(parse_signed_int("9223372036854775807"), Some(i64::MAX));
        assert_eq!(parse_signed_int("+9223372036854775807"), Some(i64::MAX));
        assert_eq!(parse_signed_int("-9223372036854775808"), Some(i64::MIN));
        assert_eq!(parse_signed_int("9223372036854775808"), None);
        assert_eq!(parse_signed_int("-9223372036854775809"), None);
    }
}
