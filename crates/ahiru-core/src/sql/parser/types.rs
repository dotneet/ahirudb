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
pub(super) fn int_literal(text: &str, negative: bool, pos: usize) -> Result<Value> {
    let mut mag: u128 = 0;
    for &d in text.as_bytes() {
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

pub(super) fn float_literal(text: &str, pos: usize) -> Result<Value> {
    match text.parse::<f64>() {
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
// Only the 8 units listed in DESIGN.md §7 (year, month, day, hour, minute, second,
// millisecond, microsecond) are accepted, in both singular and plural. DuckDB's other
// abbreviations (`mon`/`y`/`wk` and so on) are out of scope (a smaller table keeps code size down).

#[derive(Clone, Copy)]
pub(super) enum IntervalUnit {
    Year,
    Month,
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
    fn add(acc: &mut i64, delta: Option<i64>, pos: usize) -> Result<()> {
        match delta.and_then(|d| acc.checked_add(d)) {
            Some(v) => {
                *acc = v;
                Ok(())
            }
            None => err!(NumberOverflow, pos),
        }
    }
    const US_PER_SEC: i64 = 1_000_000;
    const US_PER_MIN: i64 = 60 * US_PER_SEC;
    const US_PER_HOUR: i64 = 60 * US_PER_MIN;
    match u {
        IntervalUnit::Year => add(months, n.checked_mul(12), pos),
        IntervalUnit::Month => add(months, Some(n), pos),
        IntervalUnit::Day => add(days, Some(n), pos),
        IntervalUnit::Hour => add(micros, n.checked_mul(US_PER_HOUR), pos),
        IntervalUnit::Minute => add(micros, n.checked_mul(US_PER_MIN), pos),
        IntervalUnit::Second => add(micros, n.checked_mul(US_PER_SEC), pos),
        IntervalUnit::Millisecond => add(micros, n.checked_mul(1_000), pos),
        IntervalUnit::Microsecond => add(micros, Some(n), pos),
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

/// The compound form `'<n> <unit> [<n> <unit> ...]'`. Repeated units are simply added
/// (DuckDB likewise treats `'1 month 1 month'` as `2 months`).
pub(super) fn parse_interval_text(text: &str, pos: usize) -> Result<i128> {
    let (mut months, mut days, mut micros) = (0i64, 0i64, 0i64);
    let mut any = false;
    let mut it = text.split_ascii_whitespace();
    while let Some(num_tok) = it.next() {
        let Some(n) = parse_signed_int(num_tok) else { err!(SyntaxError, pos) };
        let Some(unit_tok) = it.next() else { err!(SyntaxError, pos) };
        let Some(unit) = lookup_interval_unit(unit_tok.as_bytes()) else { err!(SyntaxError, pos) };
        add_interval_unit(unit, n, &mut months, &mut days, &mut micros, pos)?;
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
