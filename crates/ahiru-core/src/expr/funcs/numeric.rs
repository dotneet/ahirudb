//! Integer output and floating-point output
use super::datetime::{civil, date_add, date_diff, date_part, date_trunc, days_in_month};
use super::json::json_extract_or_whole;
use super::string::{cp_count, find};
use super::*;

/// `res` is the call's result type. Only the DECIMAL-aware arms consult it, for the scale the
/// result has to come back at (`round` on a DECIMAL drops the digits below the requested
/// position from the scale as well, matching DuckDB).
pub(super) fn eval_int(id: FuncId, a: &A, res: Ty) -> Result<Option<i64>> {
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
        // Reads the argument at full width so that a HUGEINT/UBIGINT (physically I128) is not
        // seen as zero; the answer is a plain -1/0/1 either way.
        F_SIGN_I => Some(a.i128(0).signum() as i64),
        F_ROUND_I => {
            let d = if a.n() >= 2 { a.int(1) } else { 0 };
            narrow(round_scaled(a.i128(0), a.dec_scale(0), scale_of(res), d))
        }
        F_CEIL_I | F_FLOOR_I | F_TRUNC_I => narrow(whole_scaled(id, a.i128(0), a.dec_scale(0))),
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
        // A negative left-shift amount, or one at or above the word width (64), is undefined.
        // It gives NULL (the same "undefined operations are NULL" policy as division by zero);
        // DuckDB raises an error there instead, and this engine prefers NULL to failing a whole
        // scan (docs/sql/functions-numeric.md).
        F_BIT_SHL => u32::try_from(a.int(1)).ok().and_then(|n| a.int(0).checked_shl(n)),
        // The right shift is *not* an error in DuckDB: shifting by 64 or more, or by a negative
        // amount, simply yields 0 (`select 8 >> 64` -> `0`, `select 8 >> -1` -> `0`), so there
        // is no reason to diverge here.
        F_BIT_SHR => Some(match u32::try_from(a.int(1)) {
            Ok(k) if k < 64 => a.int(0) >> k,
            _ => 0,
        }),
        F_BIT_NOT => Some(!a.int(0)),
        // The **code point** of the first character, not its first byte, so `chr(ascii(s))`
        // round-trips (DuckDB's `ascii`/`unicode` behave the same). The empty string is the one
        // place the two names part company: `ascii('')` is 0 and `unicode('')`/`ord('')` is -1,
        // as in DuckDB. Invalid UTF-8 still gives NULL.
        F_ASCII | F_UNICODE => {
            let s = a.bytes(0);
            if s.is_empty() {
                Some(if id == F_ASCII { 0 } else { -1 })
            } else {
                core::str::from_utf8(s).ok().and_then(|t| t.chars().next()).map(|c| c as i64)
            }
        }
        F_BIT_XOR => Some(a.int(0) ^ a.int(1)),
        F_BIT_COUNT => Some(a.int(0).count_ones() as i64),
        // Always non-negative, matching DuckDB (`select gcd(-4, 6)` -> `2`).
        F_GCD => gcd(a.int(0), a.int(1)),
        F_LCM => {
            let (x, y) = (a.int(0), a.int(1));
            if x == 0 || y == 0 {
                // Short-circuit before gcd: lcm(i64::MIN, 0) is exactly zero
                // even though abs(i64::MIN), and therefore its gcd with zero,
                // is not representable as a positive BIGINT.
                Some(0)
            } else {
                match gcd(x, y) {
                    None => None,
                    // Both inputs are non-zero, so their gcd cannot be zero.
                    Some(0) => None,
                    Some(g) => {
                        // Divide first so the product does not overflow needlessly.
                        (x / g).checked_mul(y).and_then(|v| v.checked_abs())
                    }
                }
            }
        }
        // Out-of-range components give NULL rather than silently normalizing
        // (`make_date(2024, 13, 1)` is an error in DuckDB; NULL is this engine's convention
        // for an undefined argument, the same as `sqrt(-1)`).
        F_MAKE_DATE => make_date(a.int(0), a.int(1), a.int(2)),
        // DuckDB accepts a time of day anywhere in `[00:00:00, 24:00:00]` with minutes below 60
        // and seconds at or below 60, and normalizes what that adds up to -- `24:00:00` is the
        // next day's midnight and `07:08:60` is `07:09:00`. Anything past 24:00:00, and any
        // negative or over-60 component, stays NULL (DuckDB errors there).
        F_MAKE_TIMESTAMP => {
            let (h, mi, s) = (a.int(3), a.int(4), a.int(5));
            let tod = h * US_PER_HOUR + mi * US_PER_MIN + s * US_PER_SEC;
            if !(0..=24).contains(&h)
                || !(0..60).contains(&mi)
                || !(0..=60).contains(&s)
                || tod > US_PER_DAY
            {
                None
            } else {
                make_date(a.int(0), a.int(1), a.int(2))
                    .and_then(|d| d.checked_mul(US_PER_DAY))
                    .and_then(|us| us.checked_add(tod))
            }
        }
        // TIMESTAMP is already microseconds since the epoch, so these are pure rescalings.
        // epoch_ms truncates toward zero, like DuckDB: one microsecond before
        // the epoch is 0 milliseconds, not -1. epoch itself intentionally
        // floors to whole seconds in date_part (a separate documented choice).
        F_EPOCH_MS => Some(a.int(0) / 1_000),
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

/// HUGEINT / UBIGINT / wide-DECIMAL (I128) output: `factorial`/postfix `!`
/// (`F_FACTORIAL`, `sql::parser` desugars `!` to a call of this name), plus the
/// exact-integer and exact-decimal arms of the numeric functions that `eval_int`
/// serves at `i64` width. Those share their function IDs with `eval_int`; which of
/// the two runs is decided purely by the result's physical type, exactly as for the
/// `f64` split (`see funcs::num_ty`).
///
/// `res` carries the result type, for the same reason as in [`eval_int`].
pub(super) fn eval_i128(id: FuncId, a: &A, res: Ty) -> Result<Option<i128>> {
    Ok(match id {
        F_FACTORIAL => Some(factorial(a.int(0))?),
        F_ABS_I => a.i128(0).checked_abs(),
        F_ROUND_I => {
            let d = if a.n() >= 2 { a.int(1) } else { 0 };
            round_scaled(a.i128(0), a.dec_scale(0), scale_of(res), d)
        }
        F_CEIL_I | F_FLOOR_I | F_TRUNC_I => whole_scaled(id, a.i128(0), a.dec_scale(0)),
        F_MOD_I => {
            let (x, y) = (a.i128(0), a.i128(1));
            // Division by zero and MIN % -1 give NULL, as in the `i64` arm above.
            if y == 0 || (y == -1 && x == i128::MIN) {
                None
            } else {
                Some(x % y)
            }
        }
        _ => err!(Internal),
    })
}

/// The DECIMAL scale of a type (0 for anything else).
fn scale_of(t: Ty) -> u32 {
    match t {
        Ty::Decimal { scale, .. } => scale as u32,
        _ => 0,
    }
}

/// Narrows an `i128` result back to the `i64` the caller's physical type holds.
/// Out of range gives NULL, the same as any other unrepresentable result.
fn narrow(v: Option<i128>) -> Option<i64> {
    v.and_then(|x| i64::try_from(x).ok())
}

/// `10^k` as an `i128`. `None` past `i128`'s range.
fn pow10_i128(k: u32) -> Option<i128> {
    let mut p: i128 = 1;
    for _ in 0..k {
        p = p.checked_mul(10)?;
    }
    Some(p)
}

/// `round(x, d)` on the scaled integer of a DECIMAL with scale `s_in`, returned at scale
/// `s_out`. An integer is a DECIMAL with scale 0, so this covers `round(<bigint>, -2)` too.
///
/// Rounding is half away from zero, matching DuckDB's `round` (and unlike the
/// round-half-to-even the float-to-integer *casts* use). `s_out` is never above `s_in`
/// (`funcs::resolve` picks it), so the final rescaling only ever divides, and it divides
/// exactly: the digits it drops were already zeroed by the rounding step.
///
/// `None` (= NULL) when the result does not fit, e.g. `round(<hugeint max>, -1)`.
fn round_scaled(x: i128, s_in: u32, s_out: u32, d: i64) -> Option<i128> {
    // How many of the scaled integer's digits fall below the requested position.
    // Saturating, so `d = i64::MIN` does not overflow on the way to "round everything away".
    let k = (s_in as i64).saturating_sub(d);
    let y = if k <= 0 { x } else { round_pow10(x, k)? };
    let drop = s_in.saturating_sub(s_out);
    if drop == 0 {
        Some(y)
    } else {
        Some(y / pow10_i128(drop)?)
    }
}

/// Rounds `x` away from zero to a multiple of `10^k` (`round_pow10(12345, 2)` -> 12300).
/// `k` is at least 1. `None` on overflow.
fn round_pow10(x: i128, k: i64) -> Option<i128> {
    // Every i128 is below 10^39 in magnitude, so at that point the answer is zero: even the
    // largest value is under half of `10^k` and rounds to nothing. (10^39 does not fit an
    // i128 either, so this also keeps `pow10_i128` in range.)
    if k >= 39 {
        return Some(0);
    }
    let p = pow10_i128(k as u32)?;
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

/// `ceil` / `floor` / `trunc` on the scaled integer of a DECIMAL with scale `s_in`. The
/// result is a whole number at scale 0, as in DuckDB (`ceil(1.5::DECIMAL(4,2))` is the
/// `DECIMAL(4,0)` `2`).
fn whole_scaled(id: FuncId, x: i128, s_in: u32) -> Option<i128> {
    let p = pow10_i128(s_in)?;
    let (q, r) = (x / p, x % p);
    Some(match id {
        F_CEIL_I if r > 0 => q + 1,
        F_FLOOR_I if r < 0 => q - 1,
        _ => q,
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

/// The greatest common divisor, always non-negative. i64::MIN has no positive
/// absolute value in BIGINT, so an unrepresentable result is returned as None.
fn gcd(a: i64, b: i64) -> Option<i64> {
    let (mut x, mut y) = (a.unsigned_abs(), b.unsigned_abs());
    while y != 0 {
        let t = x % y;
        x = y;
        y = t;
    }
    i64::try_from(x).ok()
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

pub(super) fn eval_f64(id: FuncId, a: &A) -> Result<Option<f64>> {
    let x = a.flt(0);
    Ok(match id {
        F_ABS_F => Some(f_abs(x)),
        // Zero (either sign) gives positive zero, matching DuckDB's `sign(-0.0)` -> `0`.
        // NaN has no sign to report and is passed through.
        F_SIGN_F => Some(if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else if x.is_nan() {
            x
        } else {
            0.0
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
        F_CBRT => Some(f_cbrt(x)),
        F_RADIANS => Some(x * (core::f64::consts::PI / 180.0)),
        F_DEGREES => Some(x * (180.0 / core::f64::consts::PI)),
        F_POW => Some(f_pow(x, a.flt(1))),
        F_MOD_F => Some(x % a.flt(1)),
        _ => err!(Internal),
    })
}

/// An implementation shared with `expr::kernels`. Clearing the sign bit rather than negating
/// matters for `-0.0`: `-0.0 < 0.0` is false, so the comparison form used to hand back the
/// negative zero unchanged and `abs(-0.0)` printed as `-0` (DuckDB prints `0`).
pub(crate) fn f_abs(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & !(1u64 << 63))
}

/// Truncation toward zero. `f64::trunc` is std-only, so this is in-house.
/// An implementation shared with `expr::kernels`.
pub(crate) fn f_trunc(x: f64) -> f64 {
    if f_abs(x) < 9_223_372_036_854_775_808.0 {
        let t = (x as i64) as f64;
        // The trip through `i64` loses the sign of zero, so `trunc(-0.3)` came back as `+0.0`
        // and `floor`/`ceil`/`round`, which are all built on this, printed `0` where DuckDB
        // prints `-0.0`. Restoring the sign bit is enough (there is no `copysign` in `core`).
        if t == 0.0 {
            f64::from_bits(x.to_bits() & (1u64 << 63))
        } else {
            t
        }
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

/// `ln(2)` split so that `k * LN2_HI` is exact for every exponent `k` a double can carry
/// (`LN2_HI`'s low mantissa bits are zero) and `LN2_HI + LN2_LO` reproduces `ln(2)` to about
/// 90 bits. Shared by `f_ln` and `f_exp`.
const LN2_HI: f64 = 6.931_471_803_691_238e-1;
const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;

/// Dekker's two-product: returns `(p, e)` with `a * b == p + e` exactly, `p` being the rounded
/// product. Used to form residuals that would otherwise cancel away (`f_sqrt`/`f_cbrt`).
/// `core` has no `f64::mul_add`, so the operands are split into 26-bit halves by hand.
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    // 2^27 + 1, the classic splitting factor.
    const S: f64 = 134_217_729.0;
    let t = a * S;
    let ah = t - (t - a);
    let al = a - ah;
    let u = b * S;
    let bh = u - (u - b);
    let bl = b - bh;
    let p = a * b;
    (p, (((ah * bh - p) + ah * bl + al * bh) + al * bl))
}

/// Square root. `core` has no `f64::sqrt` (it is in libm), so this is Newton's iteration from an
/// initial value with a halved exponent, plus one exact correction step.
///
/// The five Newton steps alone land within one ulp but are not correctly rounded (`sqrt(2)` came
/// out as `1.414213562373095`, one ulp below the true `1.4142135623730951`). The correction
/// computes the residual `x - y*y` exactly with [`two_prod`] -- the subtraction itself is exact by
/// Sterbenz's lemma, since `y*y` is within a factor of two of `x` -- and applies
/// `y += r / (2y)`. That single rounding at the end makes the result correctly rounded.
pub(super) fn f_sqrt(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    // A subnormal has no usable exponent field, so the bit-trick seed would be wildly off and
    // five iterations could not recover. Scale by 2^54 (an even power, so the square root scales
    // by exactly 2^27) and undo it afterwards.
    if x < f64::MIN_POSITIVE {
        return f_sqrt(x * 18_014_398_509_481_984.0) / 134_217_728.0;
    }
    let mut y = f64::from_bits((x.to_bits() + (1023u64 << 52)) >> 1);
    for _ in 0..5 {
        y = 0.5 * (y + x / y);
    }
    let (p, e) = two_prod(y, y);
    let r = (x - p) - e;
    y + r / (y + y)
}

/// Cube root, defined for negative input too. Newton's iteration from the classic
/// "divide the exponent by three" bit trick, finished with the same exact-residual correction as
/// [`f_sqrt`]: `r = x - y^3` (formed from two [`two_prod`]s so the cancellation is exact),
/// then `y += r / (3 y^2)`.
///
/// It used to be spelled `exp(ln(x)/3)`, which lost several ulps and broke exact identities --
/// `cbrt(8)` returned `1.9999999999999998`.
pub(super) fn f_cbrt(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let a = f_abs(x);
    // Subnormals: scale by 2^63 (a perfect cube of 2^21) so the seed below sees a real exponent.
    if a < f64::MIN_POSITIVE {
        let y = f_cbrt(a * 9_223_372_036_854_775_808.0) / 2_097_152.0;
        return if x < 0.0 { -y } else { y };
    }
    let mut y = f64::from_bits(a.to_bits() / 3 + 0x2a9f_7625_3119_d328);
    for _ in 0..4 {
        y -= (y - a / (y * y)) / 3.0;
    }
    let (p2, e2) = two_prod(y, y);
    let (p3, e3) = two_prod(y, p2);
    let r = (a - p3) - (e3 + y * e2);
    y += r / (3.0 * p2);
    if x < 0.0 {
        -y
    } else {
        y
    }
}

/// The natural logarithm. `x = 2^k * m` with `m` in `[sqrt(2)/2, sqrt(2))`; `ln(m)` uses the
/// atanh form `2*atanh(f/(2+f))` with `f = m - 1`, evaluated as fdlibm's `__ieee754_log` does:
/// the even and odd halves of the series are summed separately and the leading `f` is kept out of
/// the polynomial, so the dominant term is never rounded twice. `k*ln(2)` is added through the
/// `LN2_HI`/`LN2_LO` split for the same reason.
///
/// The previous straight Horner form was one ulp off for about a quarter of all inputs
/// (`ln(10)` among them); this form is one ulp off for about 0.1% and never more than that.
pub(super) fn f_ln(x: f64) -> f64 {
    // The exponent decoding below reads `0x7ff` for an infinity or a NaN and would answer
    // `1024 * ln 2` -- a plausible-looking finite number -- for both. `ln(inf)` is `inf`
    // and `ln(NaN)` is NaN, as in DuckDB. (Negative input, `-inf` included, never reaches
    // here: the callers turn `x <= 0` into NULL first.)
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    let mut bits = x.to_bits();
    let mut k = 0i32;
    // Subnormals are scaled by 2^64 before being handled.
    if (bits >> 52) & 0x7ff == 0 {
        bits = (x * 18_446_744_073_709_551_616.0).to_bits();
        k -= 64;
    }
    k += ((bits >> 52) & 0x7ff) as i32 - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | (1023u64 << 52));
    if m > core::f64::consts::SQRT_2 {
        m *= 0.5;
        k += 1;
    }
    // The minimax coefficients of fdlibm's `__ieee754_log` for `s^2` on this reduced range.
    const LG1: f64 = 6.666_666_666_666_735e-1;
    const LG2: f64 = 3.999_999_999_940_942e-1;
    const LG3: f64 = 2.857_142_874_366_239e-1;
    const LG4: f64 = 2.222_219_843_214_978_4e-1;
    const LG5: f64 = 1.818_357_216_161_805e-1;
    const LG6: f64 = 1.531_383_769_920_937_3e-1;
    const LG7: f64 = 1.479_819_860_511_658_6e-1;
    let f = m - 1.0;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    let r = t1 + t2;
    let hfsq = 0.5 * f * f;
    if k == 0 {
        f - (hfsq - s * (hfsq + r))
    } else {
        let kf = k as f64;
        kf * LN2_HI - ((hfsq - (s * (hfsq + r) + kf * LN2_LO)) - f)
    }
}

/// Multiplies by `2^k`. Split into two steps so the exponent does not overflow at once.
fn scale2(m: f64, k: i32) -> f64 {
    let p = |e: i32| f64::from_bits(((e + 1023) as u64) << 52);
    let k1 = k.clamp(-700, 700);
    m * p(k1) * p(k - k1)
}

/// The exponential function. Split as `x = k*ln2 + r` (the `ln2` subtraction done in a high and a
/// low part so it does not cancel for large `x`), `exp(r)` comes from fdlibm's `__ieee754_exp`
/// rational form `1 + r + r*c/(2-c)`, and the result is scaled by `2^k`.
///
/// The three terms of that sum are added with exact two-sums rather than left to round one after
/// another, so only the very last addition rounds. That is what makes `exp(1)` come out as
/// `2.718281828459045` instead of one ulp above it; a plain Taylor Horner loop was one ulp off
/// for about 10% of inputs, this form for about 1.5%.
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
    let hi = x - k * LN2_HI;
    let lo = k * LN2_LO;
    let r = hi - lo;
    // fdlibm's minimax coefficients for `R(z) = r*(exp(r)+1)/(exp(r)-1)` on |r| <= ln2/2.
    const P1: f64 = 1.666_666_666_666_660_2e-1;
    const P2: f64 = -2.777_777_777_701_559_3e-3;
    const P3: f64 = 6.613_756_321_437_934e-5;
    const P4: f64 = -1.653_390_220_546_525_2e-6;
    const P5: f64 = 4.138_136_797_057_238_5e-8;
    let t = r * r;
    let c = r - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    let q = (r * c) / (2.0 - c);
    // exp(r) = 1 + hi - lo + q. `1 >= |hi|` and `|1 + hi| >= |q|`, so both two-sums are exact.
    let s1 = 1.0 + hi;
    let e1 = hi - (s1 - 1.0);
    let s2 = s1 + q;
    let e2 = q - (s2 - s1);
    scale2(s2 + ((e1 + e2) - lo), k as i32)
}

/// Exponentiation. Integer exponents small enough to square up are computed exactly by repeated
/// squaring (so `pow(2,3)` does not come out as 7.999...). Everything else uses
/// `exp(y * ln |x|)`, with the sign restored from the exponent's parity when the base is
/// negative.
///
/// The IEEE 754 / C `pow` special cases come first: `pow(x, ±0)` and `pow(1, y)` are 1 even
/// when the other operand is NaN (`duckdb -c "select pow(1, 'nan'::DOUBLE)"` -> `1.0`).
pub(super) fn f_pow(x: f64, y: f64) -> f64 {
    if y == 0.0 || x == 1.0 {
        return 1.0;
    }
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    // `f_trunc` leaves an infinity alone, so `int_exp` is also true for `y = ±inf`. That is
    // what the parity test below wants: an infinite exponent has no parity (`inf % 2` is NaN),
    // and IEEE 754 gives `pow(negative, ±inf)` the same value as `pow(|negative|, ±inf)`.
    let n = f_trunc(y);
    let int_exp = n == y;
    if int_exp && f_abs(y) <= 1024.0 {
        let mut r = 1.0f64;
        let mut b = x;
        let mut k = f_abs(n) as u32;
        while k > 0 {
            if k & 1 == 1 {
                r *= b;
            }
            b *= b;
            k >>= 1;
        }
        // A negative exponent takes the reciprocal of the whole (exactly computed) positive
        // power. Squaring `1/x` instead would carry that first rounding into every step:
        // `pow(10, -2)` used to give `0.010000000000000002` rather than `0.01`.
        return if n < 0.0 { 1.0 / r } else { r };
    }
    // A negative base is defined only for an integer exponent; its magnitude comes from `|x|`
    // and its sign from whether that exponent is odd. `y % 2.0` is exact for every
    // integer-valued double (everything at or above 2^53 is even), which is what makes this
    // work for exponents far too large to square up: `pow(-2, 1025)` is `-inf`, `pow(-2, 2000)`
    // is `+inf`. Before this, both came out NaN.
    let negate = if x < 0.0 {
        if !int_exp {
            // A negative *finite* base with a non-integer exponent has no real solution.
            // `-inf` is the exception: IEEE 754 gives `pow(-inf, y)` the same value as
            // `pow(+inf, y)` whenever `y` is not an odd integer, and DuckDB agrees
            // (`pow(-inf, 0.5)` -> `inf`, `pow(-inf, -0.5)` -> `0`).
            if x != f64::NEG_INFINITY {
                return f64::NAN;
            }
            false
        } else {
            f_abs(y % 2.0) == 1.0
        }
    } else {
        false
    };
    let a = f_abs(x);
    let mag = if a == 0.0 {
        if y < 0.0 {
            f64::INFINITY
        } else {
            0.0
        }
    } else if a == 1.0 {
        // `x == -1.0` (the positive case returned at the top). `ln(1)` is 0, so the general
        // form below would evaluate `inf * 0` = NaN for an infinite exponent.
        1.0
    } else {
        f_exp(y * f_ln(a))
    };
    if negate {
        -mag
    } else {
        mag
    }
}
