//! Cell rendering: `Value` -> display string.
//!
//! Kept independent of the output *mode* (see `output.rs`): every mode renders
//! the same cell text and only differs in how cells are framed, quoted and
//! separated. The one exception is `NULL`, which each mode spells its own way,
//! so `render` takes the caller's null spelling.

use ahiru_core::vector::{Ty, Value};

/// The default rendering of `NULL`. Same as DuckDB's `.nullvalue` default.
pub const DEFAULT_NULL: &str = "NULL";

pub fn render(v: &Value, ty: Ty, null: &str) -> String {
    match v {
        Value::Null => null.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::I32(x) if ty == Ty::Date => fmt_date(*x as i64),
        Value::I32(x) => x.to_string(),
        Value::I64(x) if ty == Ty::Timestamp => fmt_timestamp(*x),
        Value::I64(x) if ty == Ty::Timestamptz => format!("{}+00", fmt_timestamp(*x)),
        Value::I64(x) if ty == Ty::Time => fmt_time(*x),
        Value::I64(x) => fmt_scaled(*x as i128, ty),
        Value::I128(x) if ty == Ty::Interval => fmt_interval_value(*x),
        Value::I128(x) => fmt_scaled(*x, ty),
        Value::F64(x) => x.to_string(),
        Value::Bytes(b) if ty == Ty::Uuid => fmt_uuid(b),
        Value::Bytes(b) => match std::str::from_utf8(b) {
            Ok(s) => s.to_string(),
            Err(_) => format!("<{} bytes>", b.len()),
        },
    }
}

/// Whether a type is numeric, for right-alignment in the boxed output modes.
pub fn is_numeric(ty: Ty) -> bool {
    use Ty::*;
    matches!(
        ty,
        TinyInt
            | SmallInt
            | Int
            | BigInt
            | HugeInt
            | UTinyInt
            | USmallInt
            | UInt
            | UBigInt
            | Float
            | Double
            | Decimal { .. }
    )
}

fn fmt_interval_value(packed: i128) -> String {
    let (months, days, micros) = ahiru_core::vector::unpack_interval(packed);
    let mut out = Vec::new();
    ahiru_core::vector::fmt_interval(months, days, micros, &mut out);
    String::from_utf8(out).unwrap_or_default()
}

/// DECIMAL is held as a scaled integer, so the decimal point is inserted at display time.
fn fmt_scaled(v: i128, ty: Ty) -> String {
    let scale = match ty {
        Ty::Decimal { scale, .. } => scale as usize,
        _ => return v.to_string(),
    };
    if scale == 0 {
        return v.to_string();
    }
    let neg = v < 0;
    let digits = v.unsigned_abs().to_string();
    // Prepend a 0 when there is no integer part (0.05 and the like).
    let padded = if digits.len() <= scale {
        format!("{}{}", "0".repeat(scale - digits.len() + 1), digits)
    } else {
        digits
    };
    let split = padded.len() - scale;
    format!("{}{}.{}", if neg { "-" } else { "" }, &padded[..split], &padded[split..])
}

/// Turns days since the epoch into `YYYY-MM-DD`. The civil_from_days algorithm.
fn fmt_date(days: i64) -> String {
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Displays a TIME (microseconds since midnight) as `HH:MM:SS`. A simplification
/// for CLI display only, dropping sub-microseconds (the same convention as the time part of `fmt_timestamp`).
fn fmt_time(micros: i64) -> String {
    let rem = micros.rem_euclid(86_400_000_000);
    let (h, mi, s) = (rem / 3_600_000_000, rem / 60_000_000 % 60, rem / 1_000_000 % 60);
    format!("{h:02}:{mi:02}:{s:02}")
}

fn fmt_timestamp(micros: i64) -> String {
    let days = micros.div_euclid(86_400_000_000);
    let rem = micros.rem_euclid(86_400_000_000);
    let (y, m, d) = civil_from_days(days);
    let (h, mi, s) = (rem / 3_600_000_000, rem / 60_000_000 % 60, rem / 1_000_000 % 60);
    format!("{y:04}-{m:02}-{d:02} {h:02}:{mi:02}:{s:02}")
}

/// Turns a 16-byte UUID into `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`.
/// A value whose length is not 16 (which should never happen) is shown as raw hex.
fn fmt_uuid(b: &[u8]) -> String {
    let Ok(b): Result<[u8; 16], _> = b.try_into() else {
        return b.iter().map(|x| format!("{x:02x}")).collect();
    };
    let mut s = String::with_capacity(36);
    for (i, byte) in b.iter().enumerate() {
        if matches!(i, 4 | 6 | 8 | 10) {
            s.push('-');
        }
        s.push_str(&format!("{byte:02x}"));
    }
    s
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
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
