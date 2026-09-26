//! The civil calendar and date-time input/output (used by `expr::kernels`'s casts too)
use super::*;

/// A calendar day. The result of `civil_from_days` plus the time part.
pub(super) struct Civil {
    // `y`/`mo` are read from `numeric::eval_int`'s `F_LAST_DAY` arm (a
    // sibling submodule), so they need `pub(super)`; `d`/`tod`/`days` stay
    // private since only this file's own date/time helpers use them.
    pub(super) y: i64,
    pub(super) mo: u32,
    d: u32,
    /// Microseconds since midnight.
    tod: i64,
    /// Days since the epoch.
    days: i64,
}

/// Days since the epoch -> `(year, month, day)`. Howard Hinnant's civil_from_days.
/// It carries no tables, so leap years and leap centuries are handled by formulas alone.
pub(crate) fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `(year, month, day)` -> days since the epoch. The inverse of civil_from_days.
pub(crate) fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = y - if m <= 2 { 1 } else { 0 };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = if m > 2 { m as i64 - 3 } else { m as i64 + 9 };
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Decomposes a TIMESTAMP (microseconds) into a calendar date. To avoid an
/// off-by-one-day error before the epoch, every division uses floor division (`div_euclid`).
pub(super) fn civil(us: i64) -> Civil {
    civil_at(us.div_euclid(US_PER_DAY), us.rem_euclid(US_PER_DAY))
}

/// A calendar day from a day count and a time of day. A DATE argument is decomposed from its
/// day count directly, so dates beyond TIMESTAMP's range (DuckDB's DATE reaches year 5881580,
/// TIMESTAMP only about 294247) still answer `year()`, `dayname()` and friends.
pub(super) fn civil_at(days: i64, tod: i64) -> Civil {
    let (y, mo, d) = civil_from_days(days);
    Civil { y, mo, d, tod, days }
}

/// Argument `k` of a date function as a `Civil`. `resolve` leaves DATE and TIME arguments
/// uncast (see `dt_arg` there), so the physical value is a day count or a time of day for
/// those, and microseconds since the epoch for TIMESTAMP/TIMESTAMPTZ.
pub(super) fn arg_civil(a: &A, k: usize) -> Civil {
    let v = a.int(k);
    match a.ty(k) {
        Ty::Date => civil_at(v, 0),
        Ty::Time => civil_at(0, v),
        _ => civil(v),
    }
}

impl Civil {
    /// Microseconds since the epoch, widened so a far-off DATE cannot overflow.
    fn micros(&self) -> i128 {
        self.days as i128 * US_PER_DAY as i128 + self.tod as i128
    }
}

/// `date_part` of argument `k`: an INTERVAL is split into its own fields, and a TIME only
/// has the time-of-day parts (DuckDB rejects `year(TIME ...)` and `dow` of an INTERVAL).
pub(super) fn part_of(a: &A, k: usize, p: u8) -> Result<Option<i64>> {
    match a.ty(k) {
        Ty::Interval => interval_part(p, a.i128(k)),
        Ty::Time if !time_part(p) => err!(TypeMismatch),
        _ => Ok(date_part(p, &arg_civil(a, k))),
    }
}

/// The parts a TIME has.
pub(super) fn time_part(p: u8) -> bool {
    matches!(p, P_HOUR | P_MINUTE | P_SECOND | P_MILLISECOND | P_MICROSECOND | P_EPOCH)
}

/// `date_part` of an INTERVAL, field by field and with truncating division, as DuckDB does:
/// `year` is `months / 12`, `hour` is the whole hours in the microsecond field (days are not
/// folded in), and `second`/`millisecond`/`microsecond` all count within the current minute.
pub(super) fn interval_part(p: u8, v: i128) -> Result<Option<i64>> {
    let (m, d, u) = crate::vector::unpack_interval(v);
    let (m, d) = (m as i64, d as i64);
    Ok(Some(match p {
        P_YEAR => m / 12,
        P_QUARTER => m % 12 / 3 + 1,
        P_MONTH => m % 12,
        P_DAY => d,
        P_DECADE => m / 120,
        P_CENTURY => m / 1_200,
        P_MILLENNIUM => m / 12_000,
        P_HOUR => u / US_PER_HOUR,
        P_MINUTE => u % US_PER_HOUR / US_PER_MIN,
        P_SECOND => u % US_PER_MIN / US_PER_SEC,
        P_MILLISECOND => u % US_PER_MIN / 1_000,
        P_MICROSECOND => u % US_PER_MIN,
        // A year counts 365.25 days and a month 30, DuckDB's convention. The result is whole
        // seconds, floored, like the TIMESTAMP `epoch` here (DuckDB returns a DOUBLE).
        P_EPOCH => {
            let secs = (m / 12) as i128 * 31_557_600 + ((m % 12) * 30 + d) as i128 * 86_400;
            (secs * US_PER_SEC as i128 + u as i128).div_euclid(US_PER_SEC as i128) as i64
        }
        _ => err!(TypeMismatch),
    }))
}

fn is_leap(y: i64) -> bool {
    (y.rem_euclid(4) == 0 && y.rem_euclid(100) != 0) || y.rem_euclid(400) == 0
}

pub(super) fn days_in_month(y: i64, m: u32) -> u32 {
    const N: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let i = (m.clamp(1, 12) - 1) as usize;
    if i == 1 && is_leap(y) {
        29
    } else {
        N[i]
    }
}

/// Day of week. 0 = Sunday (same as DuckDB's `dow`). 1970-01-01 is a Thursday.
fn weekday(days: i64) -> i64 {
    (days + 4).rem_euclid(7)
}

/// The Thursday of the ISO week containing `days`. The ISO year and week number are both defined
/// by which calendar year that Thursday falls in.
fn iso_thursday(days: i64) -> i64 {
    let iso_dow = match weekday(days) {
        0 => 7,
        w => w,
    };
    days + (4 - iso_dow)
}

/// ISO 8601 week number: which week of the year the Thursday of the week containing this day falls in.
fn iso_week(days: i64) -> i64 {
    let thursday = iso_thursday(days);
    let (y, _, _) = civil_from_days(thursday);
    (thursday - days_from_civil(y, 1, 1)) / 7 + 1
}

/// ISO 8601 week-numbering year (`date_part('isoyear', DATE '2024-12-30')` is 2025).
fn iso_year(days: i64) -> i64 {
    civil_from_days(iso_thursday(days)).0
}

/// The first day (a Monday) of the ISO week-numbering year containing `days`. `date_trunc`'s
/// `isoyear` unit.
fn iso_year_start(days: i64) -> i64 {
    // ISO week 1 always contains January 4th, so the Thursday of *that* week minus 3 days is the
    // Monday the ISO year starts on.
    iso_thursday(days_from_civil(iso_year(days), 1, 4)) - 3
}

/// DuckDB's 1-based century/millennium number for a `span`-year period: positive years
/// count up from 1, and year 0 and earlier count down from -1, so neither has a period 0.
fn one_based(y: i64, span: i64) -> i64 {
    if y > 0 {
        (y - 1) / span + 1
    } else {
        y / span - 1
    }
}

/// The body of `date_part` / `year()` and friends.
pub(super) fn date_part(p: u8, c: &Civil) -> Option<i64> {
    Some(match p {
        P_YEAR => c.y,
        P_QUARTER => (c.mo as i64 + 2) / 3,
        P_MONTH => c.mo as i64,
        P_WEEK => iso_week(c.days),
        P_DAY => c.d as i64,
        P_HOUR => c.tod / US_PER_HOUR,
        P_MINUTE => c.tod / US_PER_MIN % 60,
        P_SECOND => c.tod / US_PER_SEC % 60,
        P_DOW => weekday(c.days),
        P_DOY => c.days - days_from_civil(c.y, 1, 1) + 1,
        // Both include the whole seconds field, matching DuckDB
        // (`date_part('millisecond', TIMESTAMP '2021-08-03 11:59:44.123456')` -> 44123).
        P_MILLISECOND => c.tod.rem_euclid(US_PER_MIN) / 1_000,
        P_MICROSECOND => c.tod.rem_euclid(US_PER_MIN),
        // Monday = 1 .. Sunday = 7.
        P_ISODOW => match weekday(c.days) {
            0 => 7,
            w => w,
        },
        // Year 1-100 is century 1, 2021 is century 21 (DuckDB's definition). There is no
        // century 0: years 0 to -99 are century -1, -100 to -199 century -2, and so on
        // (DuckDB's `year / 100 - 1` for `year <= 0`, with truncating division).
        // Note `date_trunc`/`date_diff` deliberately do *not* share this 1-based definition;
        // see their own comments.
        P_CENTURY => one_based(c.y, 100),
        // Truncating toward zero, as DuckDB does: `-0084` is decade -8, and `-0009` decade 0.
        P_DECADE => c.y / 10,
        // Same 1-based definition as the century (`date_part('millennium', 2024)` is 3,
        // and year 0 is millennium -1).
        P_MILLENNIUM => one_based(c.y, 1000),
        P_ISOYEAR => iso_year(c.days),
        // DuckDB returns a DOUBLE including fractional seconds; here it is BIGINT seconds (floored).
        _ => c.days * 86_400 + c.tod / US_PER_SEC,
    })
}

/// `date_trunc`. Truncation is always toward the floor, so nothing is "off by one unit" even
/// before the epoch.
pub(super) fn date_trunc(p: u8, us: i64) -> Result<Option<i64>> {
    let c = civil(us);
    let day = |d: i64| d.checked_mul(US_PER_DAY);
    let unit = |u: i64| Some(us.div_euclid(u) * u);
    Ok(match p {
        P_YEAR => day(days_from_civil(c.y, 1, 1)),
        P_QUARTER => day(days_from_civil(c.y, (c.mo - 1) / 3 * 3 + 1, 1)),
        P_MONTH => day(days_from_civil(c.y, c.mo, 1)),
        // The week starts on Monday (the same as DuckDB).
        P_WEEK => day(c.days - (weekday(c.days) + 6).rem_euclid(7)),
        P_DAY => day(c.days),
        P_HOUR => unit(US_PER_HOUR),
        P_MINUTE => unit(US_PER_MIN),
        P_SECOND => unit(US_PER_SEC),
        P_MILLISECOND => unit(1_000),
        P_MICROSECOND => Some(us),
        // The day-level parts truncate to the day, and `epoch` to the second (DuckDB).
        P_DOW | P_DOY | P_ISODOW => day(c.days),
        P_EPOCH => unit(US_PER_SEC),
        // Toward zero like the century and millennium below, again as DuckDB does
        // (`date_trunc('decade', DATE '-0084-07-27')` is year -80, not -90).
        P_DECADE => day(days_from_civil(c.y / 10 * 10, 1, 1)),
        // DuckDB truncates to `year / 100 * 100`, not to the start of the 1-based century that
        // `date_part('century')` counts: `date_trunc('century', DATE '2024-05-05')` is
        // `2000-01-01` and `date_trunc('century', DATE '1900-12-31')` is `1900-01-01`.
        // The division truncates toward zero rather than flooring, again matching DuckDB
        // (`date_trunc('century', DATE '-0050-05-05')` is year 0, printed there as `0001-01-01 (BC)`).
        P_CENTURY => day(days_from_civil(c.y / 100 * 100, 1, 1)),
        P_MILLENNIUM => day(days_from_civil(c.y / 1000 * 1000, 1, 1)),
        P_ISOYEAR => day(iso_year_start(c.days)),
        _ => err!(TypeMismatch),
    })
}

/// `date_diff`. Like DuckDB it counts "how many boundaries were crossed"
/// (`date_diff('day', '..23:00', '..01:00')` is 1).
///
/// The units below a day are counted on `i128` microseconds, so two DATEs beyond TIMESTAMP's
/// range still compare (`date_diff('hour', DATE '300000-01-01', DATE '300001-01-01')`).
pub(super) fn date_diff(p: u8, ca: &Civil, cb: &Civil) -> Result<Option<i64>> {
    let (a, b) = (ca.micros(), cb.micros());
    let unit = |u: i64| (b.div_euclid(u as i128) - a.div_euclid(u as i128)) as i64;
    Ok(Some(match p {
        P_YEAR => cb.y - ca.y,
        P_QUARTER => (cb.y * 4 + (cb.mo as i64 - 1) / 3) - (ca.y * 4 + (ca.mo as i64 - 1) / 3),
        P_MONTH => (cb.y * 12 + cb.mo as i64) - (ca.y * 12 + ca.mo as i64),
        // A week difference is a day difference divided by 7 (DuckDB uses this definition too).
        P_WEEK => (cb.days - ca.days) / 7,
        // DuckDB counts the day-level parts as plain days
        // (`date_diff('dow', DATE '2024-01-01', DATE '2024-01-10')` is 9).
        P_DAY | P_DOW | P_DOY | P_ISODOW => cb.days - ca.days,
        P_HOUR => unit(US_PER_HOUR),
        P_MINUTE => unit(US_PER_MIN),
        P_SECOND => unit(US_PER_SEC),
        P_EPOCH => unit(US_PER_SEC),
        P_MILLISECOND => unit(1_000),
        // The only unit whose difference is not already divided down, so it is the only one that
        // can overflow i64 (`date_diff('microsecond', TIMESTAMP '-290000-01-01', TIMESTAMP
        // '290000-01-01')` used to wrap to a negative number). DuckDB raises an Out of Range
        // error for the same call.
        P_MICROSECOND => match i64::try_from(b - a) {
            Ok(v) => v,
            Err(_) => err!(ValueOutOfRange),
        },
        // Truncating toward zero, like `date_trunc` and DuckDB: years -9 to 9 are all one
        // decade, so `date_diff('decade', DATE '-0005-06-01', DATE '0005-06-01')` is 0.
        P_DECADE => cb.y / 10 - ca.y / 10,
        // Like `date_trunc`, DuckDB counts century and millennium boundaries with a plain
        // `year / 100` (truncating toward zero), not with the 1-based numbering
        // `date_part('century')` returns: `date_diff('century', DATE '1900-01-01',
        // DATE '2024-01-01')` is 1, and `date_diff('century', DATE '-0150-01-01',
        // DATE '0050-01-01')` is 1 as well.
        P_CENTURY => cb.y / 100 - ca.y / 100,
        P_MILLENNIUM => cb.y / 1000 - ca.y / 1000,
        P_ISOYEAR => iso_year(cb.days) - iso_year(ca.days),
        _ => err!(TypeMismatch),
    }))
}

/// `date_add(part, n, ts)`. DuckDB's `date_add` takes an INTERVAL; this is a deliberate
/// incompatibility because this implementation has no INTERVAL type (see the comment in resolve).
///
/// Adding years, quarters, and months clamps the day to the month's end (`2024-01-31` + 1 month =
/// `2024-02-29`). The same rule as DuckDB.
pub(super) fn date_add(p: u8, k: i64, us: i64) -> Result<Option<i64>> {
    let c = civil(us);
    let months = |m: i64| -> Option<i64> {
        let t = (c.y * 12 + c.mo as i64 - 1).checked_add(m)?;
        let (y, mo) = (t.div_euclid(12), t.rem_euclid(12) as u32 + 1);
        let d = c.d.min(days_in_month(y, mo));
        days_from_civil(y, mo, d).checked_mul(US_PER_DAY)?.checked_add(c.tod)
    };
    let unit = |u: i64| k.checked_mul(u).and_then(|d| us.checked_add(d));
    Ok(match p {
        P_YEAR => k.checked_mul(12).and_then(months),
        P_QUARTER => k.checked_mul(3).and_then(months),
        P_MONTH => months(k),
        P_WEEK => unit(7 * US_PER_DAY),
        P_DAY => unit(US_PER_DAY),
        P_HOUR => unit(US_PER_HOUR),
        P_MINUTE => unit(US_PER_MIN),
        P_SECOND => unit(US_PER_SEC),
        P_MILLISECOND => unit(1_000),
        P_MICROSECOND => unit(1),
        P_DECADE => k.checked_mul(120).and_then(months),
        P_CENTURY => k.checked_mul(1_200).and_then(months),
        P_MILLENNIUM => k.checked_mul(12_000).and_then(months),
        _ => err!(TypeMismatch),
    })
}

/// Adds INTERVAL's three components to a TIMESTAMP (microseconds). Called from
/// `expr::kernels::ts_add_interval`. Months follow the same rule as `date_add` (clamped to a day
/// not past the month's end), while days and microseconds are plain additions (carries appear
/// naturally in the timestamp's digits).
pub(crate) fn add_interval_to_ts(us: i64, months: i32, days: i32, micros: i64) -> Option<i64> {
    let c = civil(us);
    let t = (c.y * 12 + c.mo as i64 - 1).checked_add(months as i64)?;
    let (y, mo) = (t.div_euclid(12), t.rem_euclid(12) as u32 + 1);
    let d = c.d.min(days_in_month(y, mo));
    let total_days = days_from_civil(y, mo, d).checked_add(days as i64)?;
    let ts = total_days.checked_mul(US_PER_DAY)?.checked_add(c.tod)?;
    ts.checked_add(micros)
}

/// Adds `n` units plus `frac` millionths of a unit (`frac` carries `n`'s sign) of the interval
/// unit named `word` (any `date_part` spelling: `h`, `mons`, `millennia`, ...) into `acc`, which
/// is `[months, days, micros]`. This is DuckDB's `Interval::FromCString` per-unit rule: a
/// fraction cascades into the next smaller field only, truncated there, so `'1.25 months'` is
/// `1 month 7 days` (the remaining half day is dropped) and `'1.3 years'` is `1 year 3 months`;
/// a fraction of a quarter or a week cascades one field further, and a fraction of a
/// microsecond rounds half away from zero.
///
/// `SyntaxError` for a word that is not an interval unit (`dow`, `epoch`, ...), and
/// `NumberOverflow` when a field overflows.
pub(crate) fn add_interval_unit(word: &[u8], n: i64, frac: i64, acc: &mut [i64; 3]) -> Result<()> {
    const MIL: i64 = 1_000_000;
    let Some(p) = part_id(word) else { err!(SyntaxError) };
    // (field, units of that field per unit, units of the next field per unit of this one).
    let (field, mul, next) = match p {
        P_MILLENNIUM => (0, 12_000, 0),
        P_CENTURY => (0, 1_200, 0),
        P_DECADE => (0, 120, 0),
        P_YEAR => (0, 12, 0),
        P_QUARTER => (0, 3, 30),
        P_MONTH => (0, 1, 30),
        P_WEEK => (1, 7, US_PER_DAY),
        P_DAY => (1, 1, US_PER_DAY),
        P_HOUR => (2, US_PER_HOUR, 0),
        P_MINUTE => (2, US_PER_MIN, 0),
        P_SECOND => (2, US_PER_SEC, 0),
        P_MILLISECOND => (2, 1_000, 0),
        P_MICROSECOND => (2, 1, 0),
        _ => err!(SyntaxError),
    };
    let (n, frac) = if p == P_MICROSECOND { (n + frac * 2 / MIL, 0) } else { (n, frac) };
    let whole = n.checked_mul(mul).and_then(|v| v.checked_add(frac * mul / MIL));
    let Some(v) = whole.and_then(|v| acc[field].checked_add(v)) else { err!(NumberOverflow) };
    acc[field] = v;
    if next != 0 {
        let Some(v) = acc[field + 1].checked_add(frac * mul % MIL * next / MIL) else {
            err!(NumberOverflow)
        };
        acc[field + 1] = v;
    }
    Ok(())
}

/// The operator forms `plan::compile` lowers to function calls. A result that leaves its type's
/// range is NULL, where DuckDB raises (this engine's convention for an undefined value).
///
/// - `F_TS_SUB`: `TIMESTAMP - TIMESTAMP`, an INTERVAL of whole days plus the remaining
///   microseconds, both truncated toward zero (`-61 days -10:00:00`), as DuckDB builds it.
/// - `F_TIME_ADD_IV`: `TIME + INTERVAL`. Only the time-of-day part of the microseconds field
///   moves the clock, which then wraps around midnight; months and days are ignored (DuckDB).
/// - `F_DATE_ADD_TIME`: `DATE + TIME`, a TIMESTAMP.
/// - `F_IV_MUL_F` / `F_IV_DIV_F`: `INTERVAL * DOUBLE` and `INTERVAL / number`. DuckDB divides by
///   multiplying with `1.0 / n` (a zero divisor is NULL), and both go through PostgreSQL's
///   `interval_mul`, which cascades a fractional month into days (30 per month) and a fractional
///   day into microseconds, rounding each cascade to a microsecond: `INTERVAL '1 month' / 7` is
///   `4 days 06:51:25.6896`.
pub(super) fn temporal_op(id: FuncId, a: &A) -> Option<i128> {
    use crate::vector::{pack_interval, unpack_interval};
    Some(match id {
        F_TS_SUB => {
            let d = a.int(0).checked_sub(a.int(1))?;
            pack_interval(0, (d / US_PER_DAY) as i32, d % US_PER_DAY)
        }
        F_TIME_ADD_IV => {
            let (_, _, u) = unpack_interval(a.i128(1));
            (a.int(0) + u % US_PER_DAY).rem_euclid(US_PER_DAY) as i128
        }
        F_DATE_ADD_TIME => a.int(0).checked_mul(US_PER_DAY)?.checked_add(a.int(1))? as i128,
        _ => {
            let mut f = a.flt(1);
            if id == F_IV_DIV_F {
                if f == 0.0 {
                    return None;
                }
                f = 1.0 / f;
            }
            let (m, d, u) = unpack_interval(a.i128(0));
            let fits = |x: f64| (i32::MIN as f64..=i32::MAX as f64).contains(&x);
            let (mf, df) = (m as f64 * f, d as f64 * f);
            if !fits(mf) || !fits(df) {
                return None;
            }
            let (rm, mut rd) = (mf as i32, df as i32);
            let round_us = |x: f64| crate::expr::kernels::f_round(x * 1e6) / 1e6;
            let month_rem = round_us((mf - rm as f64) * 30.0);
            let day_rem = month_rem as i32;
            let mut sec_rem = round_us((df - rd as f64 + month_rem - day_rem as f64) * 86_400.0);
            if f_abs(sec_rem) >= 86_400.0 {
                let k = (sec_rem / 86_400.0) as i32;
                rd = rd.checked_add(k)?;
                sec_rem -= k as f64 * 86_400.0;
            }
            let um = crate::expr::kernels::f_round(u as f64 * f + sec_rem * 1e6);
            if !(-9.223_372_036_854_775e18..9.223_372_036_854_775e18).contains(&um) {
                return None;
            }
            pack_interval(rm, rd.checked_add(day_rem)?, um as i64)
        }
    })
}

// =========================================================================
// Date-time input/output (used by kernels' casts too)
// =========================================================================

/// Right-aligned zero-padded decimal. A leading `-` when negative.
fn pad(v: i64, w: usize, out: &mut Vec<u8>) {
    let mut buf = [0u8; 24];
    let mut u = v.unsigned_abs();
    let mut k = 0usize;
    loop {
        buf[k] = b'0' + (u % 10) as u8;
        u /= 10;
        k += 1;
        if u == 0 || k == buf.len() {
            break;
        }
    }
    if v < 0 {
        out.push(b'-');
    }
    for _ in k..w {
        out.push(b'0');
    }
    for i in (0..k).rev() {
        out.push(buf[i]);
    }
}

/// `YYYY-MM-DD`.
pub(crate) fn fmt_date(days: i64, out: &mut Vec<u8>) {
    let (y, m, d) = civil_from_days(days);
    pad(y, 4, out);
    out.push(b'-');
    pad(m as i64, 2, out);
    out.push(b'-');
    pad(d as i64, 2, out);
}

/// `HH:MM:SS[.ffffff]`. Trailing zeros are dropped from the fraction (the same as DuckDB).
pub(crate) fn fmt_time(us: i64, out: &mut Vec<u8>) {
    // TIME permits the single endpoint 24:00:00. Keep that spelling instead of
    // wrapping it to midnight: the parser deliberately accepts it and DuckDB
    // preserves the endpoint when casting it back to VARCHAR.
    if us == US_PER_DAY {
        out.extend_from_slice(b"24:00:00");
        return;
    }
    let t = us.rem_euclid(US_PER_DAY);
    pad(t / US_PER_HOUR, 2, out);
    out.push(b':');
    pad(t / US_PER_MIN % 60, 2, out);
    out.push(b':');
    pad(t / US_PER_SEC % 60, 2, out);
    let frac = t % US_PER_SEC;
    if frac != 0 {
        let mut buf = Vec::new();
        pad(frac, 6, &mut buf);
        while buf.last() == Some(&b'0') {
            buf.pop();
        }
        out.push(b'.');
        out.extend_from_slice(&buf);
    }
}

/// `YYYY-MM-DD HH:MM:SS[.ffffff]`.
pub(crate) fn fmt_timestamp(us: i64, out: &mut Vec<u8>) {
    fmt_date(us.div_euclid(US_PER_DAY), out);
    out.push(b' ');
    fmt_time(us.rem_euclid(US_PER_DAY), out);
}

/// Only `%Y %m %d %H %M %S %%` are interpreted. An unknown specifier is emitted verbatim, `%` and all.
pub(super) fn strftime(c: &Civil, f: &[u8], out: &mut Vec<u8>) {
    let mut i = 0usize;
    while i < f.len() {
        if f[i] != b'%' || i + 1 >= f.len() {
            out.push(f[i]);
            i += 1;
            continue;
        }
        i += 1;
        match f[i] {
            // A year before 0 is not zero-padded: DuckDB writes `-44`, not `-0044`.
            b'Y' => pad(c.y, if c.y < 0 { 0 } else { 4 }, out),
            b'm' => pad(c.mo as i64, 2, out),
            b'd' => pad(c.d as i64, 2, out),
            b'H' => pad(c.tod / US_PER_HOUR, 2, out),
            b'M' => pad(c.tod / US_PER_MIN % 60, 2, out),
            b'S' => pad(c.tod / US_PER_SEC % 60, 2, out),
            b'%' => out.push(b'%'),
            x => {
                out.push(b'%');
                out.push(x);
            }
        }
        i += 1;
    }
}

/// A numeric scan with a read position. Reads `lo..=hi` digits and returns the value and the next position.
fn scan(s: &[u8], i: usize, lo: usize, hi: usize) -> Option<(i64, usize)> {
    let mut v = 0i64;
    let mut k = 0usize;
    let mut j = i;
    while j < s.len() && k < hi && s[j].is_ascii_digit() {
        v = v * 10 + (s[j] - b'0') as i64;
        j += 1;
        k += 1;
    }
    if k < lo {
        None
    } else {
        Some((v, j))
    }
}

/// Reads `YYYY-MM-DD`. The position read up to is returned as well (for use from TIMESTAMP).
///
/// Like DuckDB, the separator may also be `/`, a space or `\`, as long as both separators are
/// the same one (`'2024/1/5'` and `'2024 01 05'` are dates, `'2024/01-05'` is not). The year
/// takes up to 7 digits, enough for DuckDB's last DATE (`5881580-07-10`); a value past the
/// DATE range is caught by the caller's range check.
fn scan_date(s: &[u8]) -> Option<(i64, usize)> {
    let neg = s.first() == Some(&b'-');
    let i = usize::from(neg);
    let (y, i) = scan(s, i, 1, 7)?;
    let sep = *s.get(i)?;
    if !matches!(sep, b'-' | b'/' | b' ' | b'\\') {
        return None;
    }
    let (m, i) = scan(s, i + 1, 1, 2)?;
    if s.get(i) != Some(&sep) {
        return None;
    }
    let (d, i) = scan(s, i + 1, 1, 2)?;
    let y = if neg { -y } else { y };
    // An out-of-range month or day is a read failure (= that row becomes NULL).
    if !(1..=12).contains(&m) || d < 1 || d > days_in_month(y, m as u32) as i64 {
        return None;
    }
    let days = days_from_civil(y, m as u32, d as u32);
    (DATE_MIN_DAYS..=DATE_MAX_DAYS).contains(&days).then_some((days, i))
}

/// Reads `HH:MM[:SS[.ffffff]]`.
fn scan_time(s: &[u8], i: usize) -> Option<(i64, usize)> {
    let (h, i) = scan(s, i, 1, 2)?;
    if s.get(i) != Some(&b':') {
        return None;
    }
    let (m, i) = scan(s, i + 1, 1, 2)?;
    let (sec, mut i) = if s.get(i) == Some(&b':') { scan(s, i + 1, 1, 2)? } else { (0, i) };
    let mut frac = 0i64;
    if s.get(i) == Some(&b'.') {
        let mut k = 0usize;
        i += 1;
        // The 7th digit onward is truncated (the internal representation is microseconds).
        while i < s.len() && s[i].is_ascii_digit() {
            if k < 6 {
                frac = frac * 10 + (s[i] - b'0') as i64;
                k += 1;
            }
            i += 1;
        }
        // A trailing `.` with no digits after it is accepted and means zero (measured:
        // `'2024-01-01 10:00:00.'::TIMESTAMP` and `'10:00:00.'::TIME` are both valid in DuckDB).
        while k < 6 {
            frac *= 10;
            k += 1;
        }
    }
    if h > 24 || m > 59 || sec > 59 {
        return None;
    }
    // SQL (and DuckDB) accept `24:00:00` only as midnight-at-the-end-of-the-day.
    // `24:01:00` used to slip through and wrap to `00:01:00` on display.
    if h == 24 && (m != 0 || sec != 0 || frac != 0) {
        return None;
    }
    Some((h * US_PER_HOUR + m * US_PER_MIN + sec * US_PER_SEC + frac, i))
}

/// VARCHAR -> DATE. `None` if unreadable (the caller makes that row NULL).
pub(crate) fn parse_date(s: &[u8]) -> Option<i64> {
    let s = crate::expr::kernels::trim_space(s);
    let (d, i) = scan_date(s)?;
    if i == s.len() {
        return Some(d);
    }
    // DuckDB's text -> DATE cast accepts a timestamp-shaped string and keeps only the date part
    // (`'2024-01-01T00:00:00'::DATE` and `'2024-01-01 10:00:00'::DATE` are both `2024-01-01`).
    // Rejecting those turned an entire ISO-timestamp column into NULLs.
    //
    // Only a well-formed time tail is accepted. DuckDB itself ignores *any* trailing text here
    // (even `'2024-01-01x'::DATE` gives `2024-01-01`); this stays stricter on purpose, so that
    // genuinely malformed input still becomes NULL -- see `docs/sql/limitations.md`.
    scan_time_tail(s, i).map(|_| d)
}

/// VARCHAR -> TIMESTAMP. `YYYY-MM-DD[ T]HH:MM[:SS[.ffffff]][zone]`.
/// A date alone counts as midnight. A zone suffix is accepted but not applied: DuckDB keeps
/// the wall-clock fields of a *without-time-zone* value unchanged.
pub(crate) fn parse_timestamp(s: &[u8]) -> Option<i64> {
    parse_ts(s, false)
}

/// VARCHAR -> TIMESTAMPTZ. The same text as `parse_timestamp`, but the zone offset is
/// subtracted to normalize "that locale's wall-clock time" into a UTC instant (for example
/// `12:00+09` is `03:00` in UTC). No offset counts as UTC (there is no session time zone; the
/// same simplification as `CURRENT_TIMESTAMP` in `sql::now`).
pub(crate) fn parse_timestamptz(s: &[u8]) -> Option<i64> {
    parse_ts(s, true)
}

fn parse_ts(s: &[u8], apply_zone: bool) -> Option<i64> {
    let s = crate::expr::kernels::trim_space(s);
    let (d, i) = scan_date(s)?;
    let base = d.checked_mul(US_PER_DAY)?;
    if i == s.len() {
        return Some(base);
    }
    let (t, zone) = scan_time_tail(s, i)?;
    base.checked_add(t)?.checked_sub(if apply_zone { zone } else { 0 })
}

/// Reads the `[ T]HH:MM[:SS[.ffffff]][zone]` tail that follows a date, with `i` at the separator,
/// up to the end of `s`. Returns the time of day and the zone offset, both in microseconds.
///
/// DuckDB allows more than one space between the date and the time
/// (`'2024-01-01  10:00:00'::TIMESTAMP` is valid), so the run of spaces is consumed as one
/// separator.
fn scan_time_tail(s: &[u8], i: usize) -> Option<(i64, i64)> {
    let mut j = i;
    match s.get(j) {
        Some(b'T' | b't') => j += 1,
        Some(b' ') => {
            while s.get(j) == Some(&b' ') {
                j += 1;
            }
        }
        _ => return None,
    }
    let (t, k) = scan_time(s, j)?;
    Some((t, scan_zone(s, k)?))
}

/// Reads the (possibly absent) zone suffix DuckDB accepts after a time, which must end the
/// text: nothing at all, `Z`, `[+-]HH`, `[+-]HHMM`, `[+-]HH:MM`, `[+-]HH:MM:SS`, or a separate
/// ` UTC` word. Returns the offset in microseconds (east is positive).
///
/// Every field is exactly two digits but not range-checked, as in DuckDB (`+99` and `+05:60`
/// are accepted there too). `UTC` is the only zone *name* accepted, matching DuckDB, which
/// rejects ` GMT` and every other named zone unless the ICU extension is loaded.
fn scan_zone(s: &[u8], i: usize) -> Option<i64> {
    let (secs, end) = match s.get(i) {
        None => (0, i),
        Some(b'Z') => (0, i + 1),
        Some(&sign @ (b'+' | b'-')) => {
            let (h, mut j) = scan(s, i + 1, 2, 2)?;
            let mut secs = h * 3600;
            if s.get(j) == Some(&b':') {
                let (m, k) = scan(s, j + 1, 2, 2)?;
                secs += m * 60;
                j = k;
                if s.get(j) == Some(&b':') {
                    let (x, k) = scan(s, j + 1, 2, 2)?;
                    secs += x;
                    j = k;
                }
            } else if let Some((m, k)) = scan(s, j, 2, 2) {
                secs += m * 60;
                j = k;
            }
            (if sign == b'-' { -secs } else { secs }, j)
        }
        Some(b' ') if s[i + 1..].eq_ignore_ascii_case(b"UTC") => (0, s.len()),
        _ => return None,
    };
    (end == s.len()).then_some(secs * US_PER_SEC)
}

/// VARCHAR -> TIME. A zone suffix is accepted and ignored, as DuckDB does for a plain TIME
/// (`'12:00:00+05'::TIME` is `12:00:00`).
pub(crate) fn parse_time(s: &[u8]) -> Option<i64> {
    let s = crate::expr::kernels::trim_space(s);
    let (t, i) = scan_time(s, 0)?;
    scan_zone(s, i).map(|_| t)
}

/// `YYYY-MM-DD HH:MM:SS[.ffffff]+00`. The physical representation is the same UTC microseconds as
/// `Ty::Timestamp`, but the `+00` suffix is added, as in DuckDB, to make explicit that the value
/// is already a UTC instant (this engine has no notion of a session time zone, so it always
/// displays in UTC).
pub(crate) fn fmt_timestamptz(us: i64, out: &mut Vec<u8>) {
    fmt_timestamp(us, out);
    out.extend_from_slice(b"+00");
}

fn hex_digit(n: u8) -> u8 {
    if n < 10 {
        b'0' + n
    } else {
        b'a' + (n - 10)
    }
}

fn hex_value(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// UUID -> `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` (lowercase, RFC 4122 byte order).
pub(crate) fn fmt_uuid(bytes: &[u8; 16], out: &mut Vec<u8>) {
    for (i, &b) in bytes.iter().enumerate() {
        if matches!(i, 4 | 6 | 8 | 10) {
            out.push(b'-');
        }
        out.push(hex_digit(b >> 4));
        out.push(hex_digit(b & 0xF));
    }
}

/// `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` -> 16 bytes. Case-insensitive.
/// Even the hyphen positions are checked strictly (DuckDB accepts only this form as well).
pub(crate) fn parse_uuid(s: &[u8]) -> Option<[u8; 16]> {
    let s = crate::expr::kernels::trim_space(s);
    if s.len() != 36 {
        return None;
    }
    for (i, &c) in s.iter().enumerate() {
        let want_dash = matches!(i, 8 | 13 | 18 | 23);
        if want_dash != (c == b'-') {
            return None;
        }
    }
    let mut out = [0u8; 16];
    let mut oi = 0usize;
    let mut i = 0usize;
    while i < s.len() {
        if s[i] == b'-' {
            i += 1;
            continue;
        }
        let hi = hex_value(s[i])?;
        let lo = hex_value(*s.get(i + 1)?)?;
        out[oi] = (hi << 4) | lo;
        oi += 1;
        i += 2;
    }
    Some(out)
}
