//! Integer output and floating-point output
use super::datetime::{civil, date_add, date_diff, date_part, date_trunc, days_in_month};
use super::json::json_extract_or_whole;
use super::string::{cp_count, find};
use super::*;

pub(super) fn eval_int(id: FuncId, a: &A) -> Result<Option<i64>> {
    // Shorthands such as year(). The part number is embedded in the ID.
    if id >= F_PART_BASE {
        return Ok(date_part((id - F_PART_BASE) as u8, a.int(0)));
    }
    Ok(match id {
        F_LENGTH => Some(cp_count(a.bytes(0)) as i64),
        F_STRPOS => {
            // 1-based, or 0 if absent. The position is in code points.
            let (s, p) = (a.bytes(0), a.bytes(1));
            Some(match find(s, p) {
                Some(b) => cp_count(&s[..b]) as i64 + 1,
                None => 0,
            })
        }
        F_ABS_I => a.int(0).checked_abs(),
        F_SIGN_I => Some(a.int(0).signum()),
        F_ROUND_I => {
            let x = a.int(0);
            let d = if a.n() >= 2 { a.int(1) } else { 0 };
            if d >= 0 {
                Some(x)
            } else if let Some(k) = d.checked_neg() {
                round_int(x, k)
            } else {
                Some(0)
            }
        }
        F_MOD_I => {
            let (x, y) = (a.int(0), a.int(1));
            // Division by zero and MIN % -1 give NULL (the same judgment as kernels' Mod).
            if y == 0 || (y == -1 && x == i64::MIN) {
                None
            } else {
                Some(x % y)
            }
        }
        F_BIT_AND => Some(a.int(0) & a.int(1)),
        F_BIT_OR => Some(a.int(0) | a.int(1)),
        // A negative shift amount, or one at or above the word width (64), is undefined. It gives
        // NULL (the same "undefined operations are NULL" policy as division by zero).
        F_BIT_SHL => u32::try_from(a.int(1)).ok().and_then(|n| a.int(0).checked_shl(n)),
        F_BIT_SHR => u32::try_from(a.int(1)).ok().and_then(|n| a.int(0).checked_shr(n)),
        F_BIT_NOT => Some(!a.int(0)),
        // The **code point** of the first character, not its first byte, so `chr(ascii(s))`
        // round-trips (DuckDB's `ascii`/`unicode` behave the same). An empty string gives NULL,
        // matching `select ascii('')` in duckdb; invalid UTF-8 gives NULL too.
        F_ASCII => {
            let s = a.bytes(0);
            core::str::from_utf8(s).ok().and_then(|t| t.chars().next()).map(|c| c as i64)
        }
        F_BIT_XOR => Some(a.int(0) ^ a.int(1)),
        F_BIT_COUNT => Some(a.int(0).count_ones() as i64),
        // Always non-negative, matching DuckDB (`select gcd(-4, 6)` -> `2`).
        F_GCD => Some(gcd(a.int(0), a.int(1))),
        F_LCM => {
            let (x, y) = (a.int(0), a.int(1));
            let g = gcd(x, y);
            if g == 0 {
                Some(0)
            } else {
                // Divide first so the product does not overflow needlessly.
                (x / g).checked_mul(y).and_then(|v| v.checked_abs())
            }
        }
        // Out-of-range components give NULL rather than silently normalizing
        // (`make_date(2024, 13, 1)` is an error in DuckDB; NULL is this engine's convention
        // for an undefined argument, the same as `sqrt(-1)`).
        F_MAKE_DATE => make_date(a.int(0), a.int(1), a.int(2)),
        F_MAKE_TIMESTAMP => {
            let (h, mi, s) = (a.int(3), a.int(4), a.int(5));
            if !(0..24).contains(&h) || !(0..60).contains(&mi) || !(0..60).contains(&s) {
                None
            } else {
                let tod = h * US_PER_HOUR + mi * US_PER_MIN + s * US_PER_SEC;
                make_date(a.int(0), a.int(1), a.int(2))
                    .and_then(|d| d.checked_mul(US_PER_DAY))
                    .and_then(|us| us.checked_add(tod))
            }
        }
        // TIMESTAMP is already microseconds since the epoch, so these are pure rescalings.
        F_EPOCH_MS => Some(a.int(0).div_euclid(1_000)),
        F_EPOCH_US => Some(a.int(0)),
        F_EPOCH_NS => a.int(0).checked_mul(1_000),
        F_LIST_POSITION => match super::json::list_find(a)? {
            // DuckDB gives NULL rather than 0 when the element is absent.
            Some(0) | None => None,
            found => found,
        },
        F_DATE_PART => match part_id(a.bytes(0)) {
            Some(p) => date_part(p, a.int(1)),
            None => err!(TypeMismatch),
        },
        F_DATE_DIFF => match part_id(a.bytes(0)) {
            Some(p) => date_diff(p, a.int(1), a.int(2))?,
            None => err!(TypeMismatch),
        },
        F_DATE_TRUNC => match part_id(a.bytes(0)) {
            Some(p) => date_trunc(p, a.int(1))?,
            None => err!(TypeMismatch),
        },
        F_DATE_ADD => match part_id(a.bytes(0)) {
            Some(p) => date_add(p, a.int(1), a.int(2))?,
            None => err!(TypeMismatch),
        },
        F_TO_DATE => parse_date(a.bytes(0)),
        F_TO_TIMESTAMP => parse_timestamp(a.bytes(0)),
        F_LAST_DAY => {
            let c = civil(a.int(0));
            Some(days_from_civil(c.y, c.mo, days_in_month(c.y, c.mo)))
        }
        F_JSON_ARRAY_LENGTH => {
            let found = json_extract_or_whole(a)?;
            match found {
                Some((span, kind)) => Some(crate::json::array_length(span, kind)?),
                None => None,
            }
        }
        _ => err!(Internal),
    })
}

/// HUGEINT (I128) output. Currently only `factorial`/postfix `!`
/// (`F_FACTORIAL`, `sql::parser` desugars `!` to a call of this name).
pub(super) fn eval_i128(id: FuncId, a: &A) -> Result<Option<i128>> {
    Ok(match id {
        F_FACTORIAL => Some(factorial(a.int(0))?),
        _ => err!(Internal),
    })
}

/// `n!`. Matches duckdb: negative `n` returns `1` rather than erroring
/// (`duckdb -c "select factorial(-1)"` -> `1` — there's no "undefined"
/// case to report here, only genuine overflow is an error). `33!` is the
/// largest factorial that fits in `i128` (`8_683_317_618_811_886_495_518_
/// 194_401_280_000_000_000`, versus `i128::MAX` ≈ `1.7e38`); `34!` ≈
/// `2.95e38` overflows, and duckdb errors there too (`duckdb -c "select
/// factorial(34)"` -> `Out of Range Error`), so this raises the engine's
/// own overflow error rather than wrapping (see the doc comment on
/// `funcs::call`'s `PhysType::I128` arm for why this diverges from the
/// "integer arithmetic overflow wraps" default).
fn factorial(n: i64) -> Result<i128> {
    if n < 0 {
        return Ok(1);
    }
    let mut acc: i128 = 1;
    let mut k: i128 = 2;
    while k <= n as i128 {
        acc = match acc.checked_mul(k) {
            Some(v) => v,
            None => err!(ValueOutOfRange),
        };
        k += 1;
    }
    Ok(acc)
}

/// The greatest common divisor, always non-negative. `i64::MIN` has no positive absolute value,
/// so it is handled through `unsigned_abs` and clamped on the way back out.
fn gcd(a: i64, b: i64) -> i64 {
    let (mut x, mut y) = (a.unsigned_abs(), b.unsigned_abs());
    while y != 0 {
        let t = x % y;
        x = y;
        y = t;
    }
    i64::try_from(x).unwrap_or(i64::MAX)
}

/// `make_date(y, m, d)` -> days since the epoch. Out-of-range month or day gives `None`
/// (= SQL NULL); the day is checked against the real length of that month, so
/// `make_date(2023, 2, 29)` is NULL while `make_date(2024, 2, 29)` is a date.
fn make_date(y: i64, m: i64, d: i64) -> Option<i64> {
    if !(1..=12).contains(&m) {
        return None;
    }
    if d < 1 || d > days_in_month(y, m as u32) as i64 {
        return None;
    }
    Some(days_from_civil(y, m as u32, d as u32))
}

/// Rounds away from zero to a power of ten (`round(12345, -2)` -> 12300).
fn round_int(x: i64, k: i64) -> Option<i64> {
    if k > 18 {
        return Some(0);
    }
    let mut p: i64 = 1;
    for _ in 0..k {
        p = p.checked_mul(10)?;
    }
    let (q, r) = (x / p, x % p);
    let half = p / 2;
    let q = if r >= half {
        q.checked_add(1)?
    } else if r <= -half {
        q.checked_sub(1)?
    } else {
        q
    };
    q.checked_mul(p)
}

pub(super) fn eval_f64(id: FuncId, a: &A) -> Result<Option<f64>> {
    let x = a.flt(0);
    Ok(match id {
        F_ABS_F => Some(f_abs(x)),
        F_SIGN_F => Some(if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            x
        }),
        F_ROUND_F => {
            let d = if a.n() >= 2 { a.int(1) } else { 0 };
            Some(round_f64(x, d))
        }
        F_CEIL_F => Some(f_ceil(x)),
        F_FLOOR_F => Some(f_floor(x)),
        F_TRUNC_F => Some(f_trunc(x)),
        // DuckDB errors outside the domain; here it gives NULL like division by zero
        // (a failure mid-expression should not bring down the whole query).
        F_SQRT => {
            if x < 0.0 {
                None
            } else {
                Some(f_sqrt(x))
            }
        }
        F_EXP => Some(f_exp(x)),
        F_LN => {
            if x <= 0.0 {
                None
            } else {
                Some(f_ln(x))
            }
        }
        F_LOG10 => {
            if x <= 0.0 {
                None
            } else {
                Some(f_ln(x) / core::f64::consts::LN_10)
            }
        }
        F_LOG2 => {
            if x <= 0.0 {
                None
            } else {
                Some(f_ln(x) / core::f64::consts::LN_2)
            }
        }
        // `log(base, x)`. A base of 1 would divide by zero, so it joins the "undefined gives
        // NULL" cases rather than returning an infinity.
        F_LOG_BASE => {
            let b = x;
            let v = a.flt(1);
            if b <= 0.0 || b == 1.0 || v <= 0.0 {
                None
            } else {
                Some(f_ln(v) / f_ln(b))
            }
        }
        // Defined for negative input too (unlike `sqrt`), matching DuckDB.
        F_CBRT => Some(if x < 0.0 { -f_pow(-x, 1.0 / 3.0) } else { f_pow(x, 1.0 / 3.0) }),
        F_RADIANS => Some(x * (core::f64::consts::PI / 180.0)),
        F_DEGREES => Some(x * (180.0 / core::f64::consts::PI)),
        F_POW => Some(f_pow(x, a.flt(1))),
        F_MOD_F => Some(x % a.flt(1)),
        _ => err!(Internal),
    })
}

/// An implementation shared with `expr::kernels`.
pub(crate) fn f_abs(x: f64) -> f64 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// Truncation toward zero. `f64::trunc` is std-only, so this is in-house.
/// An implementation shared with `expr::kernels`.
pub(crate) fn f_trunc(x: f64) -> f64 {
    if f_abs(x) < 9_223_372_036_854_775_808.0 {
        (x as i64) as f64
    } else {
        // An f64 at or above 2^63 is already an integer.
        x
    }
}

fn f_floor(x: f64) -> f64 {
    let t = f_trunc(x);
    if t > x {
        t - 1.0
    } else {
        t
    }
}

fn f_ceil(x: f64) -> f64 {
    let t = f_trunc(x);
    if t < x {
        t + 1.0
    } else {
        t
    }
}

/// Rounding away from zero. DuckDB's `round` uses this, and its rule differs from the banker's
/// rounding used for casts (`kernels::f_round`) (`round(2.5)` = 3, `round(3.5)` = 4).
pub(super) fn round_half_up(x: f64) -> f64 {
    let t = f_trunc(x);
    let frac = x - t;
    if frac >= 0.5 {
        t + 1.0
    } else if frac <= -0.5 {
        t - 1.0
    } else {
        t
    }
}

/// `round(x, d)`. Like DuckDB it multiplies by 10^d, rounds, and divides back.
fn round_f64(x: f64, d: i64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    if d == 0 {
        return round_half_up(x);
    }
    let m = pow10(d.unsigned_abs().min(308) as u32);
    let r = if d > 0 { round_half_up(x * m) / m } else { round_half_up(x / m) * m };
    if r.is_finite() {
        r
    } else {
        x
    }
}

pub(super) fn pow10(k: u32) -> f64 {
    let mut r = 1.0f64;
    for _ in 0..k {
        r *= 10.0;
    }
    r
}

/// Square root. `core` has no `f64::sqrt` (it is in libm). Five Newton iterations from an initial
/// value with a halved exponent are exact in double precision.
pub(super) fn f_sqrt(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let mut y = f64::from_bits((x.to_bits() + (1023u64 << 52)) >> 1);
    for _ in 0..5 {
        y = 0.5 * (y + x / y);
    }
    y
}

/// The natural logarithm. Decomposed as `x = m * 2^e` (with `m` in `[sqrt(2)/2, sqrt(2))`), it is
/// computed with the series `ln(m) = 2*atanh((m-1)/(m+1))`. Since |z| <= 0.1716 and
/// z^2 <= 0.0295, truncating at 16 terms drops the error to 1e-22.
pub(super) fn f_ln(x: f64) -> f64 {
    let mut bits = x.to_bits();
    let mut e = 0i32;
    // Subnormals are scaled by 2^64 before being handled.
    if (bits >> 52) & 0x7ff == 0 {
        bits = (x * 18_446_744_073_709_551_616.0).to_bits();
        e -= 64;
    }
    e += ((bits >> 52) & 0x7ff) as i32 - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | (1023u64 << 52));
    if m > core::f64::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    let z = (m - 1.0) / (m + 1.0);
    let z2 = z * z;
    let mut s = 0.0f64;
    for k in (0..16).rev() {
        s = s * z2 + 1.0 / (2 * k + 1) as f64;
    }
    2.0 * z * s + e as f64 * core::f64::consts::LN_2
}

/// Multiplies by `2^k`. Split into two steps so the exponent does not overflow at once.
fn scale2(m: f64, k: i32) -> f64 {
    let p = |e: i32| f64::from_bits(((e + 1023) as u64) << 52);
    let k1 = k.clamp(-700, 700);
    m * p(k1) * p(k - k1)
}

/// The exponential function. Split as `x = k*ln2 + r`, `exp(r)` is computed by a Taylor expansion
/// (14 terms) and multiplied by `2^k`. Since |r| <= ln2/2, 14 terms reach double precision.
pub(super) fn f_exp(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x > 709.8 {
        return f64::INFINITY;
    }
    if x < -745.2 {
        return 0.0;
    }
    let k = round_half_up(x / core::f64::consts::LN_2);
    // Subtracting ln2 in a high and a low part avoids cancellation even for large x.
    const LN2_HI: f64 = 0.693_147_180_369_123_8;
    const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
    let r = (x - k * LN2_HI) - k * LN2_LO;
    let mut s = 1.0f64;
    for i in (1..=14).rev() {
        s = 1.0 + r * s / i as f64;
    }
    scale2(s, k as i32)
}

/// Exponentiation. Integer exponents are computed exactly by repeated squaring (so `pow(2,3)` does
/// not come out as 7.999...). Everything else uses `exp(y * ln x)`.
pub(super) fn f_pow(x: f64, y: f64) -> f64 {
    if y == 0.0 {
        return 1.0;
    }
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    let n = f_trunc(y);
    if n == y && f_abs(y) <= 1024.0 {
        let mut r = 1.0f64;
        let mut b = if n < 0.0 { 1.0 / x } else { x };
        let mut k = f_abs(n) as u32;
        while k > 0 {
            if k & 1 == 1 {
                r *= b;
            }
            b *= b;
            k >>= 1;
        }
        return r;
    }
    if x < 0.0 {
        // A negative base with a non-integer exponent has no real solution.
        return f64::NAN;
    }
    if x == 0.0 {
        return if y < 0.0 { f64::INFINITY } else { 0.0 };
    }
    f_exp(y * f_ln(x))
}
