//! Free-standing parsing helpers shared across submodules: GROUPING SETS
//! expansion, literal/type-name lookups, and INTERVAL text parsing.
use super::*;

// --- GROUP BY の拡張構文 -----------------------------------------------------

/// `CUBE` の列数上限。2^n 個のグルーピングセット（= `Node::Aggregate` を
/// UNION ALL で束ねた本数）に展開されるため、無制限だとプランが爆発する。
const MAX_CUBE_COLS: usize = 8;

/// `ROLLUP (a, b, c)` を `GROUPING SETS ((a,b,c),(a,b),(a),())` に展開する。
/// 列の多い方から少ない方へ、階層的な部分集合を作る。
pub(super) fn rollup_sets(cols: Vec<ExprId>) -> Vec<Vec<ExprId>> {
    let mut sets = Vec::with_capacity(cols.len() + 1);
    for k in (0..=cols.len()).rev() {
        sets.push(cols[..k].to_vec());
    }
    sets
}

/// `CUBE (a, b)` を `GROUPING SETS ((a,b),(a),(b),())` に展開する。
/// 全部分集合（2^n 個）を作る。
pub(super) fn cube_sets(cols: Vec<ExprId>, pos: usize) -> Result<Vec<Vec<ExprId>>> {
    ensure!(cols.len() <= MAX_CUBE_COLS, ExpressionTooDeep, pos);
    let n = cols.len();
    let mut sets = Vec::with_capacity(1usize << n);
    // 先頭の列ほど上位ビットに割り当てる。こうすると `(a,b),(a),(b),()` の
    // ように「先頭に近い列を優先して残す」順になり、`ROLLUP` の階層的な
    // 部分集合の並びとも感覚が揃う（実行結果には影響しない: どの順で
    // UNION ALL しても集合として同じ）。
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

// --- ラムダ ------------------------------------------------------------------

/// 引数位置の `->` をラムダとして解釈してよい関数名か。
///
/// duckdb CLI で実測した限り、ラムダとして解釈されるのは「ラムダを受け取ると
/// 分かっている関数」の引数位置だけで、他の関数（`coalesce` 等）の引数では
/// `->` は素通りして通常の JSON パス演算子のままになる（`coalesce(doc -> 'a',
/// 'x')` は JSON 抽出として解決される一方、`abs(x -> x+1)` はラムダとして
/// 解釈されて「この関数はラムダを受け取らない」というエラーになる）。
/// この実装では関数名を固定集合として持つことで同じ区別を再現する。
pub(super) fn is_lambda_func(name: &str) -> bool {
    eq_ascii_ci(name.as_bytes(), b"list_transform")
        || eq_ascii_ci(name.as_bytes(), b"list_filter")
        || eq_ascii_ci(name.as_bytes(), b"list_reduce")
}

/// `self.cur` が比較演算子トークンなら対応する `BinaryOp` を返す。
/// `x <op> ANY|ALL|SOME (SELECT ...)` の量化比較を認識するための、
/// `expr_body` の中置ループ・`peek_quantifier` 共通の判定
/// （6 種類の比較演算子だけが `ANY`/`ALL`/`SOME` を続けられる）。
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

// --- リテラル・型名 ---------------------------------------------------------

/// 引用符の中身を展開する。二重化された引用符を 1 個に畳むだけ。
pub(super) fn unquote(raw: &str, q: u8) -> String {
    let b = raw.as_bytes();
    let mut out = String::new();
    let (mut i, mut start) = (0usize, 0usize);
    while i < b.len() {
        if b[i] == q {
            // 引用符は ASCII なので、この範囲は必ず文字境界に乗る。
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

/// 整数リテラル。収まる最小の型（I32 → I64 → I128）を選ぶ。
pub(super) fn int_literal(text: &str, negative: bool, pos: usize) -> Result<Value> {
    let mut mag: u128 = 0;
    for &d in text.as_bytes() {
        mag = match mag.checked_mul(10).and_then(|v| v.checked_add((d - b'0') as u128)) {
            Some(v) => v,
            None => err!(NumberOverflow, pos),
        };
    }
    // i128::MIN は絶対値が i128::MAX より 1 大きい。符号を見て上限を変える。
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

/// `USING SAMPLE`/`TABLESAMPLE` の手法名。一致しなければ `None`
/// （呼び出し側が「サンプル手法ではなく別の何か」として扱う）。
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

/// CAST の型名表。CAST はホットパスではないので、(長さ, 先頭バイト) で
/// 絞り込む線形走査で十分。予約語表と違い二分探索の順序制約を持たない。
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
    // 括弧なしの DECIMAL は (18,3)。I64 に収まる精度を既定にする。
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

// --- INTERVAL リテラル -------------------------------------------------------
// DESIGN.md §7 に載る 8 単位（年・月・日・時・分・秒・ミリ秒・マイクロ秒）だけ
// を単数形・複数形の両方で受け付ける。DuckDB にある他の略記（`mon`/`y`/`wk` 等）
// は対象外（テーブルを絞ることでコードサイズを増やさない）。

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

/// 符号付き 10 進整数。前後の空白は許す（`INTERVAL` の数値片用）。
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
    for &d in digits {
        v = v.checked_mul(10)?.checked_add((d - b'0') as i64)?;
    }
    Some(if neg { -v } else { v })
}

/// 1 単位ぶんを `(months, days, micros)` の累積へ足し込む。
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

/// `months`/`days` が `i32` に収まることを確認してから詰める。
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

/// `n UNIT` 1 個ぶんの INTERVAL。
pub(super) fn unit_to_interval(u: IntervalUnit, n: i64, pos: usize) -> Result<i128> {
    let (mut months, mut days, mut micros) = (0i64, 0i64, 0i64);
    add_interval_unit(u, n, &mut months, &mut days, &mut micros, pos)?;
    pack_interval_checked(months, days, micros, pos)
}

/// `'<n> <unit> [<n> <unit> ...]'` の複合形式。同じ単位が複数回出てきても
/// 単純に加算する（DuckDB も `'1 month 1 month'` を `2 months` として扱う）。
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
