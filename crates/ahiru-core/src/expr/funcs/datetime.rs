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
    let days = us.div_euclid(US_PER_DAY);
    let (y, mo, d) = civil_from_days(days);
    Civil { y, mo, d, tod: us.rem_euclid(US_PER_DAY), days }
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

/// ISO 8601 week number: which week of the year the Thursday of the week containing this day falls in.
fn iso_week(days: i64) -> i64 {
    let iso_dow = match weekday(days) {
        0 => 7,
        w => w,
    };
    let thursday = days + (4 - iso_dow);
    let (y, _, _) = civil_from_days(thursday);
    (thursday - days_from_civil(y, 1, 1)) / 7 + 1
}

/// The body of `date_part` / `year()` and friends.
pub(super) fn date_part(p: u8, us: i64) -> Option<i64> {
    let c = civil(us);
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
        // Year 1-100 is century 1, 2021 is century 21 (DuckDB's definition).
        P_CENTURY => (c.y - 1).div_euclid(100) + 1,
        P_DECADE => c.y.div_euclid(10),
        // DuckDB returns a DOUBLE including fractional seconds; here it is BIGINT seconds (floored).
        _ => us.div_euclid(US_PER_SEC),
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
        P_DECADE => day(days_from_civil(c.y.div_euclid(10) * 10, 1, 1)),
        P_CENTURY => day(days_from_civil((c.y - 1).div_euclid(100) * 100 + 1, 1, 1)),
        _ => err!(TypeMismatch),
    })
}

/// `date_diff`. Like DuckDB it counts "how many boundaries were crossed"
/// (`date_diff('day', '..23:00', '..01:00')` is 1).
pub(super) fn date_diff(p: u8, a: i64, b: i64) -> Result<Option<i64>> {
    let (ca, cb) = (civil(a), civil(b));
    let unit = |u: i64| b.div_euclid(u) - a.div_euclid(u);
    Ok(Some(match p {
        P_YEAR => cb.y - ca.y,
        P_QUARTER => (cb.y * 4 + (cb.mo as i64 - 1) / 3) - (ca.y * 4 + (ca.mo as i64 - 1) / 3),
        P_MONTH => (cb.y * 12 + cb.mo as i64) - (ca.y * 12 + ca.mo as i64),
        // A week difference is a day difference divided by 7 (DuckDB uses this definition too).
        P_WEEK => (cb.days - ca.days) / 7,
        P_DAY => cb.days - ca.days,
        P_HOUR => unit(US_PER_HOUR),
        P_MINUTE => unit(US_PER_MIN),
        P_SECOND => unit(US_PER_SEC),
        P_EPOCH => unit(US_PER_SEC),
        P_MILLISECOND => unit(1_000),
        P_MICROSECOND => b - a,
        P_DECADE => cb.y.div_euclid(10) - ca.y.div_euclid(10),
        P_CENTURY => (cb.y - 1).div_euclid(100) - (ca.y - 1).div_euclid(100),
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
pub(super) fn strftime(us: i64, f: &[u8], out: &mut Vec<u8>) {
    let c = civil(us);
    let mut i = 0usize;
    while i < f.len() {
        if f[i] != b'%' || i + 1 >= f.len() {
            out.push(f[i]);
            i += 1;
            continue;
        }
        i += 1;
        match f[i] {
            b'Y' => pad(c.y, 4, out),
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

fn trim_ws(s: &[u8]) -> &[u8] {
    let mut lo = 0;
    let mut hi = s.len();
    while lo < hi && (s[lo] == b' ' || s[lo] == b'\t') {
        lo += 1;
    }
    while hi > lo && (s[hi - 1] == b' ' || s[hi - 1] == b'\t') {
        hi -= 1;
    }
    &s[lo..hi]
}

/// Reads `YYYY-MM-DD`. The position read up to is returned as well (for use from TIMESTAMP).
fn scan_date(s: &[u8]) -> Option<(i64, usize)> {
    let neg = s.first() == Some(&b'-');
    let i = usize::from(neg);
    let (y, i) = scan(s, i, 1, 6)?;
    if s.get(i) != Some(&b'-') {
        return None;
    }
    let (m, i) = scan(s, i + 1, 1, 2)?;
    if s.get(i) != Some(&b'-') {
        return None;
    }
    let (d, i) = scan(s, i + 1, 1, 2)?;
    let y = if neg { -y } else { y };
    // An out-of-range month or day is a read failure (= that row becomes NULL).
    if !(1..=12).contains(&m) || d < 1 || d > days_in_month(y, m as u32) as i64 {
        return None;
    }
    Some((days_from_civil(y, m as u32, d as u32), i))
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
        if k == 0 {
            return None;
        }
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
    let s = trim_ws(s);
    let (d, i) = scan_date(s)?;
    if i == s.len() {
        Some(d)
    } else {
        None
    }
}

/// VARCHAR -> TIMESTAMP. `YYYY-MM-DD[ T]HH:MM[:SS[.ffffff]]`.
/// A date alone counts as midnight.
pub(crate) fn parse_timestamp(s: &[u8]) -> Option<i64> {
    let s = trim_ws(s);
    let (d, i) = scan_date(s)?;
    let base = d.checked_mul(US_PER_DAY)?;
    if i == s.len() {
        return Some(base);
    }
    if s[i] != b' ' && s[i] != b'T' && s[i] != b't' {
        return None;
    }
    let (t, j) = scan_time(s, i + 1)?;
    if j != s.len() {
        return None;
    }
    base.checked_add(t)
}

/// VARCHAR -> TIME.
pub(crate) fn parse_time(s: &[u8]) -> Option<i64> {
    let s = trim_ws(s);
    let (t, i) = scan_time(s, 0)?;
    if i == s.len() {
        Some(t)
    } else {
        None
    }
}

/// `YYYY-MM-DD HH:MM:SS[.ffffff]+00`. The physical representation is the same UTC microseconds as
/// `Ty::Timestamp`, but the `+00` suffix is added, as in DuckDB, to make explicit that the value
/// is already a UTC instant (this engine has no notion of a session time zone, so it always
/// displays in UTC).
pub(crate) fn fmt_timestamptz(us: i64, out: &mut Vec<u8>) {
    fmt_timestamp(us, out);
    out.extend_from_slice(b"+00");
}

/// Reads a `[+-]HH[:MM]` or `Z` time zone offset.
/// Returns the offset in microseconds (east is positive). `None` if unreadable
/// (the caller falls back to "no offset" = treating it as UTC).
fn scan_offset(s: &[u8], i: usize) -> Option<(i64, usize)> {
    match s.get(i) {
        Some(b'Z') | Some(b'z') => Some((0, i + 1)),
        Some(&sign @ (b'+' | b'-')) => {
            let (h, j) = scan(s, i + 1, 1, 2)?;
            let (m, j) = if s.get(j) == Some(&b':') { scan(s, j + 1, 2, 2)? } else { (0, j) };
            if h > 23 || m > 59 {
                return None;
            }
            let micros = (h * 3600 + m * 60) * 1_000_000;
            Some((if sign == b'-' { -micros } else { micros }, j))
        }
        _ => None,
    }
}

/// VARCHAR -> TIMESTAMPTZ. The date and time parts take the same form as `parse_timestamp`
/// (`YYYY-MM-DD[ T]HH:MM[:SS[.ffffff]]`), optionally followed by a time zone offset
/// (`Z` or `[+-]HH[:MM]`). No offset counts as UTC (there is no session time zone; the same
/// simplification as `CURRENT_TIMESTAMP` in `sql::now`). The offset is subtracted to normalize
/// "that locale's wall-clock time" into a UTC instant (for example `12:00+09` is `03:00` in UTC).
pub(crate) fn parse_timestamptz(s: &[u8]) -> Option<i64> {
    let s = trim_ws(s);
    let (d, i) = scan_date(s)?;
    let base = d.checked_mul(US_PER_DAY)?;
    if i == s.len() {
        return Some(base);
    }
    if s[i] != b' ' && s[i] != b'T' && s[i] != b't' {
        return None;
    }
    let (t, j) = scan_time(s, i + 1)?;
    let local = base.checked_add(t)?;
    if j == s.len() {
        return Some(local);
    }
    let (offset, k) = scan_offset(s, j)?;
    if k != s.len() {
        return None;
    }
    local.checked_sub(offset)
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
    let s = trim_ws(s);
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
