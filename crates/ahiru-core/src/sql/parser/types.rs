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
    Ok(if let Ok(x) = i32::try_from(v) {
        Value::I32(x)
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
// The 8 units listed in DESIGN.md §7 (year, month, day, hour, minute, second,
// millisecond, microsecond) plus `week`, in both singular and plural. DuckDB's other
// abbreviations (`mon`/`y`/`wk` and so on) are out of scope (a smaller table keeps code size down).

#[derive(Clone, Copy)]
pub(super) enum IntervalUnit {
    Year,
    Month,
    Week,
    Day,
    Hour,
    Minute,
    Second,
    Millisecond,
    Microsecond,
}

static INTERVAL_UNITS: &[(&[u8], IntervalUnit)] = &[
    (b"year", IntervalUnit::Year),
    (b"years", IntervalUnit::Year),
    (b"month", IntervalUnit::Month),
    (b"months", IntervalUnit::Month),
    // A week is exactly 7 days, and that is how DuckDB stores it too
    // (`INTERVAL '3 weeks'` -> `21 days`); no new field is needed.
    (b"week", IntervalUnit::Week),
    (b"weeks", IntervalUnit::Week),
    (b"day", IntervalUnit::Day),
    (b"days", IntervalUnit::Day),
    (b"hour", IntervalUnit::Hour),
    (b"hours", IntervalUnit::Hour),
    (b"minute", IntervalUnit::Minute),
    (b"minutes", IntervalUnit::Minute),
    (b"second", IntervalUnit::Second),
    (b"seconds", IntervalUnit::Second),
    (b"millisecond", IntervalUnit::Millisecond),
    (b"milliseconds", IntervalUnit::Millisecond),
    (b"microsecond", IntervalUnit::Microsecond),
    (b"microseconds", IntervalUnit::Microsecond),
];

pub(super) fn lookup_interval_unit(name: &[u8]) -> Option<IntervalUnit> {
    for &(n, u) in INTERVAL_UNITS {
        if eq_ascii_ci(n, name) {
            return Some(u);
        }
    }
    None
}

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

/// Adds one unit's worth into the `(months, days, micros)` accumulator.
fn add_interval_unit(
    u: IntervalUnit,
    n: i64,
    months: &mut i64,
    days: &mut i64,
    micros: &mut i64,
    pos: usize,
) -> Result<()> {
    add_scaled_unit(u, n as i128, 1, months, days, micros, pos)
}

/// Checked `i128` arithmetic that reports overflow as `NumberOverflow` at `pos`.
fn ck(v: Option<i128>, pos: usize) -> Result<i128> {
    match v {
        Some(x) => Ok(x),
        None => err!(NumberOverflow, pos),
    }
}

/// Adds `num / den` of one unit into the `(months, days, micros)` accumulator.
///
/// `den` is always a power of ten (see `parse_decimal_amount`), so `num / den` is
/// the exact decimal amount the user wrote; passing `den = 1` is the whole-number
/// case. Each field keeps only its integer part and cascades the remainder into the
/// next smaller one, using the same fixed conversions PostgreSQL and DuckDB use
/// (1 year = 12 months, 1 month = 30 days, 1 day = 24 hours). Verified against the
/// `duckdb` CLI:
///   `1.25 years` -> 1 year 3 months        `0.5 months` -> 15 days
///   `1.5 days`   -> 1 day 12:00:00         `1.5 weeks`  -> 10 days 12:00:00
///   `1.5 hours`  -> 01:30:00               `1.5 seconds` -> 00:00:01.5
fn add_scaled_unit(
    u: IntervalUnit,
    num: i128,
    den: i128,
    months: &mut i64,
    days: &mut i64,
    micros: &mut i64,
    pos: usize,
) -> Result<()> {
    const US_PER_SEC: i128 = 1_000_000;
    const US_PER_DAY: i128 = 24 * 60 * 60 * US_PER_SEC;
    // How much of each field one whole unit is worth.
    let (per_month, per_day, per_micro): (i128, i128, i128) = match u {
        IntervalUnit::Year => (12, 0, 0),
        IntervalUnit::Month => (1, 0, 0),
        IntervalUnit::Week => (0, 7, 0),
        IntervalUnit::Day => (0, 1, 0),
        IntervalUnit::Hour => (0, 0, 60 * 60 * US_PER_SEC),
        IntervalUnit::Minute => (0, 0, 60 * US_PER_SEC),
        IntervalUnit::Second => (0, 0, US_PER_SEC),
        IntervalUnit::Millisecond => (0, 0, 1_000),
        IntervalUnit::Microsecond => (0, 0, 1),
    };
    // Every intermediate is kept as a numerator over the shared `den`, so nothing is
    // rounded until a field's integer part is taken. `%` truncates toward zero in
    // Rust, which is what a negative amount wants (`-1.5 days` -> -1 day -12:00:00).
    let m_num = ck(num.checked_mul(per_month), pos)?;
    let d_num = ck(
        ck(num.checked_mul(per_day), pos)?.checked_add(ck((m_num % den).checked_mul(30), pos)?),
        pos,
    )?;
    let u_num = ck(
        ck(num.checked_mul(per_micro), pos)?
            .checked_add(ck((d_num % den).checked_mul(US_PER_DAY), pos)?),
        pos,
    )?;
    add(months, m_num / den, pos)?;
    add(days, d_num / den, pos)?;
    add(micros, u_num / den, pos)
}

/// Adds an `i128` delta into an `i64` accumulator, reporting overflow at `pos`.
fn add(acc: &mut i64, delta: i128, pos: usize) -> Result<()> {
    match i64::try_from(delta).ok().and_then(|d| acc.checked_add(d)) {
        Some(v) => {
            *acc = v;
            Ok(())
        }
        None => err!(NumberOverflow, pos),
    }
}

/// Packs after confirming that `months`/`days` fit in `i32`.
fn pack_interval_checked(months: i64, days: i64, micros: i64, pos: usize) -> Result<i128> {
    let m = match i32::try_from(months) {
        Ok(v) => v,
        Err(_) => err!(NumberOverflow, pos),
    };
    let d = match i32::try_from(days) {
        Ok(v) => v,
        Err(_) => err!(NumberOverflow, pos),
    };
    Ok(crate::vector::pack_interval(m, d, micros))
}

/// One `n UNIT` worth of INTERVAL.
pub(super) fn unit_to_interval(u: IntervalUnit, n: i64, pos: usize) -> Result<i128> {
    let (mut months, mut days, mut micros) = (0i64, 0i64, 0i64);
    add_interval_unit(u, n, &mut months, &mut days, &mut micros, pos)?;
    pack_interval_checked(months, days, micros, pos)
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

/// A signed decimal amount, as the exact rational `num / den` where `den` is a power
/// of ten (`"1.5"` -> `(15, 10)`, `"-2"` -> `(-2, 1)`). This is what lets
/// `INTERVAL '1.5 days'` be exact without any floating point.
fn parse_decimal_amount(s: &str) -> Option<(i128, i128)> {
    let b = s.as_bytes();
    let (neg, rest) = match b.first()? {
        b'-' => (true, &b[1..]),
        b'+' => (false, &b[1..]),
        _ => (false, b),
    };
    let (mut num, mut den, mut seen, mut in_frac) = (0i128, 1i128, false, false);
    for &c in rest {
        if c == b'.' {
            if in_frac {
                return None;
            }
            in_frac = true;
            continue;
        }
        if !c.is_ascii_digit() {
            return None;
        }
        num = num.checked_mul(10)?.checked_add((c - b'0') as i128)?;
        if in_frac {
            den = den.checked_mul(10)?;
        }
        seen = true;
    }
    if !seen {
        return None;
    }
    Some(if neg { (-num, den) } else { (num, den) })
}

/// A bare time component, `[+|-]HH:MM[:SS[.frac]]`, in microseconds.
///
/// This is the shape an interval *prints* as, so accepting it is what makes an
/// interval this engine emitted readable back in. DuckDB accepts it inside any
/// INTERVAL string, on its own (`'1:30:00'`, `'01:02'`, `'01:02:03.5'`) or mixed with
/// unit terms (`'1 day 01:02:03'`, `'-2 days -03:04:05'`). The hour field is not
/// wrapped at 24 (`'100:00:00'` is 100 hours), and the fraction is truncated at
/// microsecond resolution.
fn parse_time_component(s: &str) -> Option<i64> {
    const US_PER_SEC: i64 = 1_000_000;
    let (neg, rest) = match s.as_bytes().first()? {
        b'-' => (true, &s[1..]),
        b'+' => (false, &s[1..]),
        _ => (false, s),
    };
    let mut it = rest.split(':');
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
                    if scale == 0 {
                        break;
                    }
                }
            }
            (digits_i64(whole)?, us)
        }
    };
    if it.next().is_some() {
        return None;
    }
    let total = hours
        .checked_mul(60 * 60 * US_PER_SEC)?
        .checked_add(minutes.checked_mul(60 * US_PER_SEC)?)?
        .checked_add(secs.checked_mul(US_PER_SEC)?)?
        .checked_add(frac_us)?;
    Some(if neg { -total } else { total })
}

/// The compound form `'<n> <unit> [<n> <unit> ...] [HH:MM[:SS[.frac]]]'`. Terms are
/// simply added, so repeated units accumulate (DuckDB likewise treats `'1 month 1
/// month'` as `2 months`), and any term may be fractional or a bare time component.
pub(crate) fn parse_interval_text(text: &str, pos: usize) -> Result<i128> {
    let (mut months, mut days, mut micros) = (0i64, 0i64, 0i64);
    let mut any = false;
    let mut it = text.split_ascii_whitespace();
    while let Some(tok) = it.next() {
        // A time component carries its units in its own shape, so unlike every other
        // term it is not followed by a unit word.
        if tok.as_bytes().contains(&b':') {
            let Some(us) = parse_time_component(tok) else { err!(SyntaxError, pos) };
            add(&mut micros, us as i128, pos)?;
            any = true;
            continue;
        }
        let Some((num, den)) = parse_decimal_amount(tok) else { err!(SyntaxError, pos) };
        let Some(unit_tok) = it.next() else { err!(SyntaxError, pos) };
        let Some(unit) = lookup_interval_unit(unit_tok.as_bytes()) else { err!(SyntaxError, pos) };
        add_scaled_unit(unit, num, den, &mut months, &mut days, &mut micros, pos)?;
        any = true;
    }
    ensure!(any, SyntaxError, pos);
    pack_interval_checked(months, days, micros, pos)
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
