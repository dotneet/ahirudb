//! 整数出力・浮動小数出力
use super::datetime::{civil, date_add, date_diff, date_part, date_trunc, days_in_month};
use super::json::json_extract_or_whole;
use super::string::{cp_count, find};
use super::*;

pub(super) fn eval_int(id: FuncId, a: &A) -> Result<Option<i64>> {
    // year() などの略記。part 番号は ID に埋まっている。
    if id >= F_PART_BASE {
        return Ok(date_part((id - F_PART_BASE) as u8, a.int(0)));
    }
    Ok(match id {
        F_LENGTH => Some(cp_count(a.bytes(0)) as i64),
        F_STRPOS => {
            // 1 始まり、無ければ 0。位置はコードポイント単位。
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
            } else {
                round_int(x, -d)
            }
        }
        F_MOD_I => {
            let (x, y) = (a.int(0), a.int(1));
            // 0 除算と MIN % -1 は NULL（kernels の Mod と同じ判断）。
            if y == 0 || (y == -1 && x == i64::MIN) {
                None
            } else {
                Some(x % y)
            }
        }
        F_BIT_AND => Some(a.int(0) & a.int(1)),
        F_BIT_OR => Some(a.int(0) | a.int(1)),
        // シフト量が負、または語幅（64）以上は未定義。NULL にする
        // （ゼロ除算と同じ「未定義演算は NULL」の方針）。
        F_BIT_SHL => u32::try_from(a.int(1)).ok().and_then(|n| a.int(0).checked_shl(n)),
        F_BIT_SHR => u32::try_from(a.int(1)).ok().and_then(|n| a.int(0).checked_shr(n)),
        F_BIT_NOT => Some(!a.int(0)),
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

/// 10 の冪へ 0 から遠ざかる向きに丸める（`round(12345, -2)` → 12300）。
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
        // DuckDB は定義域外をエラーにするが、ここは 0 除算と同じく NULL に
        // する（式の途中で失敗してもクエリ全体を落とさない）。
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
        F_POW => Some(f_pow(x, a.flt(1))),
        F_MOD_F => Some(x % a.flt(1)),
        _ => err!(Internal),
    })
}

/// `expr::kernels` とも共有する実装。
pub(crate) fn f_abs(x: f64) -> f64 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// 0 方向への切り捨て。`f64::trunc` は std 専用なので自前で持つ。
/// `expr::kernels` とも共有する実装。
pub(crate) fn f_trunc(x: f64) -> f64 {
    if f_abs(x) < 9_223_372_036_854_775_808.0 {
        (x as i64) as f64
    } else {
        // 2^63 以上の f64 は既に整数。
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

/// 0 から遠ざかる丸め。DuckDB の `round` はこちらで、キャストの銀行丸め
/// (`kernels::f_round`) とは規則が違う（`round(2.5)` = 3、`round(3.5)` = 4）。
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

/// `round(x, d)`。DuckDB と同じく 10^d を掛けて丸め、割り戻す。
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

/// 平方根。`core` に `f64::sqrt` は無い（libm 側）。指数部を半分にした
/// 初期値から Newton 法で 5 回反復すれば倍精度で厳密になる。
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

/// 自然対数。`x = m * 2^e`（`m ∈ [√2/2, √2)`）に分解し、
/// `ln(m) = 2*atanh((m-1)/(m+1))` の級数で求める。|z| <= 0.1716、
/// z² <= 0.0295 なので、16 項で打ち切り誤差が 1e-22 まで落ちる。
pub(super) fn f_ln(x: f64) -> f64 {
    let mut bits = x.to_bits();
    let mut e = 0i32;
    // 非正規化数は 2^64 倍してから扱う。
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

/// `2^k` を掛ける。指数部が一度に振り切れないよう 2 回に分ける。
fn scale2(m: f64, k: i32) -> f64 {
    let p = |e: i32| f64::from_bits(((e + 1023) as u64) << 52);
    let k1 = k.clamp(-700, 700);
    m * p(k1) * p(k - k1)
}

/// 指数関数。`x = k*ln2 + r` に分け、`exp(r)` を Taylor 展開（14 項）で
/// 求めて `2^k` を掛ける。|r| <= ln2/2 なので 14 項で倍精度に届く。
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
    // ln2 を上位・下位に分けて引くと、大きな x でも桁落ちしない。
    const LN2_HI: f64 = 0.693_147_180_369_123_8;
    const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
    let r = (x - k * LN2_HI) - k * LN2_LO;
    let mut s = 1.0f64;
    for i in (1..=14).rev() {
        s = 1.0 + r * s / i as f64;
    }
    scale2(s, k as i32)
}

/// べき乗。整数指数は繰り返し二乗で厳密に出す（`pow(2,3)` を
/// 7.999… にしないため）。それ以外は `exp(y * ln x)`。
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
        // 負の底に非整数の指数は実数解を持たない。
        return f64::NAN;
    }
    if x == 0.0 {
        return if y < 0.0 { f64::INFINITY } else { 0.0 };
    }
    f_exp(y * f_ln(x))
}
