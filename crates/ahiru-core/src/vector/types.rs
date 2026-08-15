//! The type system.
//!
//! Logical and physical types are separated, and **execution kernels are written
//! against the six physical types only**. This is the biggest lever against kernel monomorphization blowup (DESIGN.md §8, §11).
//!
//! Time types are normalized to microseconds at ingest. Not distinguishing
//! TIMESTAMP(ms/us/ns) by type means one set of comparison, arithmetic, and cast kernels suffices.

use crate::prelude::*;

/// The physical representations the execution kernels handle. There are no others.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PhysType {
    Bool,
    I32,
    I64,
    F64,
    I128,
    Bytes,
}

impl PhysType {
    #[inline]
    pub fn as_u8(self) -> u8 {
        self as u8
    }
    pub const COUNT: usize = 6;
}

/// The types visible from SQL.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ty {
    /// A NULL literal whose type is not yet decided.
    Null,
    Boolean,
    TinyInt,
    SmallInt,
    Int,
    BigInt,
    HugeInt,
    UTinyInt,
    USmallInt,
    UInt,
    UBigInt,
    Float,
    Double,
    /// precision <= 18 is held as I64; anything larger as I128.
    Decimal {
        precision: u8,
        scale: u8,
    },
    Varchar,
    Blob,
    /// Days since the epoch (I32).
    Date,
    /// Microseconds since midnight (I64).
    Time,
    /// Microseconds since the epoch (I64).
    Timestamp,
    /// Physically identical to `Timestamp` (UTC microseconds since the epoch, I64).
    /// The difference is purely in logical meaning: this value already denotes a UTC
    /// instant (matching `isAdjustedToUTC = true` on Parquet's `TIMESTAMP` logical
    /// type), whereas `Timestamp` denotes a plain date-time with no time zone
    /// (`isAdjustedToUTC = false`, or no annotation at all).
    /// This engine has no notion of a session time zone, so apart from normalizing
    /// offset-bearing strings (`+09:00`/`Z` and the like) to UTC microseconds when
    /// casting from `VARCHAR`, it behaves the same as `Timestamp`.
    Timestamptz,
    /// UUID (the raw 16 bytes, held in `Bytes`). Stored in RFC 4122 byte order;
    /// only display and parsing convert to and from the hyphenated hex text form
    /// `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`. Unlike `VARCHAR`/`BLOB`, it is the
    /// only `Bytes`-family logical type whose text and physical representations differ.
    Uuid,
    /// Months (i32) / days (i32) / microseconds (i64) packed into one I128
    /// (`pack_interval`). The same three-component model DuckDB / PostgreSQL use.
    Interval,
    /// A dynamically typed JSON value. Physically UTF-8 JSON text (Bytes).
    ///
    /// Adding a physical type per nested type (LIST / MAP / STRUCT) would cause the
    /// kernel monomorphization blowup that DESIGN.md §8 and §11 exist to avoid, so
    /// they are unified into this single type. Arrays and objects are kept as JSON
    /// text, and functions such as `json_extract`/`list_extract`/`unnest` parse them
    /// on the spot (a design decision that gives up static type checking of elements
    /// and leans on JSON's own dynamic typing).
    Json,
}

/// The maximum DECIMAL precision.
pub const MAX_DECIMAL_PRECISION: u8 = 38;

impl Ty {
    /// Builds a DECIMAL with the precision clamped to the maximum.
    pub fn decimal(precision: u8, scale: u8) -> Ty {
        let precision = precision.min(MAX_DECIMAL_PRECISION);
        Ty::Decimal { precision, scale: scale.min(precision) }
    }

    /// The `(precision, scale)` as a DECIMAL. Integer types count as DECIMALs with
    /// scale 0 (used when mixing DECIMAL and integers in one operation).
    pub fn as_decimal(self) -> Option<(u8, u8)> {
        use Ty::*;
        Some(match self {
            Decimal { precision, scale } => (precision, scale),
            TinyInt | UTinyInt => (3, 0),
            SmallInt | USmallInt => (5, 0),
            Int | UInt => (10, 0),
            BigInt | UBigInt => (19, 0),
            HugeInt => (38, 0),
            _ => return None,
        })
    }

    pub fn phys(self) -> PhysType {
        use Ty::*;
        match self {
            Boolean => PhysType::Bool,
            Null | TinyInt | SmallInt | Int | UTinyInt | USmallInt | Date => PhysType::I32,
            BigInt | UInt | Time | Timestamp | Timestamptz => PhysType::I64,
            HugeInt | UBigInt | Interval => PhysType::I128,
            Decimal { precision, .. } => {
                if precision <= 18 {
                    PhysType::I64
                } else {
                    PhysType::I128
                }
            }
            Float | Double => PhysType::F64,
            Varchar | Blob | Json | Uuid => PhysType::Bytes,
        }
    }

    pub fn is_numeric(self) -> bool {
        use Ty::*;
        matches!(
            self,
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

    pub fn is_integer(self) -> bool {
        use Ty::*;
        matches!(
            self,
            TinyInt | SmallInt | Int | BigInt | HugeInt | UTinyInt | USmallInt | UInt | UBigInt
        )
    }

    pub fn is_temporal(self) -> bool {
        matches!(self, Ty::Date | Ty::Time | Ty::Timestamp | Ty::Timestamptz)
    }

    pub fn is_interval(self) -> bool {
        matches!(self, Ty::Interval)
    }

    /// How "wide" a type is. The ordering implicit conversion widens along.
    fn rank(self) -> u8 {
        use Ty::*;
        match self {
            Null => 0,
            Boolean => 1,
            UTinyInt | TinyInt => 2,
            USmallInt | SmallInt => 3,
            UInt | Int => 4,
            UBigInt | BigInt => 5,
            Decimal { .. } => 6,
            HugeInt => 7,
            Float => 8,
            Double => 9,
            Date => 10,
            Time => 11,
            Timestamp => 12,
            Timestamptz => 13,
            Varchar => 14,
            Blob => 15,
            // No implicit conversion with any other type (addition and subtraction
            // against DATE/TIMESTAMP is handled by a dedicated path in `plan::compile`
            // that does not go through `Ty::unify`).
            Interval => 16,
            // JSON likewise has no implicit conversion with other types.
            Json => 17,
            // UUID likewise has no implicit conversion with any other type
            // (not even with VARCHAR/BLOB, which share the `Bytes` physical type; only the dedicated rule below aligns them).
            Uuid => 18,
        }
    }

    /// `unify`'s result as a `Result`. Combinations that cannot be decided become a
    /// `TypeMismatch` error. This is the form used by the majority of call sites in
    /// `plan::bind`/`plan::compile`, where "a type mismatch is an immediate error".
    /// (Special call sites that fall back to a default type on mismatch, as with
    /// `Ty::Json`, call `unify` directly instead.)
    pub fn unify_or_mismatch(a: Ty, b: Ty) -> Result<Ty> {
        match Ty::unify(a, b) {
            Some(t) => Ok(t),
            None => err!(TypeMismatch),
        }
    }

    /// Determines the common type of a binary operation. `None` when undecidable.
    pub fn unify(a: Ty, b: Ty) -> Option<Ty> {
        use Ty::*;
        if a == b {
            return Some(a);
        }
        if a == Null {
            return Some(b);
        }
        if b == Null {
            return Some(a);
        }
        // Between two DECIMALs, align precision/scale.
        // Addition and subtraction can carry into one more digit, so precision gets +1 (as in DuckDB).
        if let (Decimal { precision: p1, scale: s1 }, Decimal { precision: p2, scale: s2 }) = (a, b)
        {
            return Some(Ty::decimal(
                p1.saturating_sub(s1).max(p2.saturating_sub(s2)) + s1.max(s2) + 1,
                s1.max(s2),
            ));
        }
        // Between numerics, widen. DECIMAL with floating point drops to DOUBLE.
        if a.is_numeric() && b.is_numeric() {
            let (lo, hi) = if a.rank() < b.rank() { (a, b) } else { (b, a) };
            if matches!(lo, Decimal { .. }) && matches!(hi, Float | Double) {
                return Some(Double);
            }
            if matches!(hi, Decimal { .. }) && matches!(lo, Float | Double) {
                return Some(Double);
            }
            return Some(if hi == Float { Double } else { hi });
        }
        // Comparing DATE with TIMESTAMP settles on TIMESTAMP.
        if matches!((a, b), (Date, Timestamp) | (Timestamp, Date)) {
            return Some(Timestamp);
        }
        // Comparisons involving TIMESTAMPTZ settle on TIMESTAMPTZ (DATE/TIMESTAMP are
        // "without time zone", so they align to the more informative TIMESTAMPTZ side.
        // DuckDB also permits comparing DATE/TIMESTAMP with TIMESTAMPTZ and settles on
        // TIMESTAMPTZ).
        if matches!(
            (a, b),
            (Date, Timestamptz)
                | (Timestamptz, Date)
                | (Timestamp, Timestamptz)
                | (Timestamptz, Timestamp)
        ) {
            return Some(Timestamptz);
        }
        if matches!((a, b), (Varchar, Blob) | (Blob, Varchar)) {
            return Some(Blob);
        }
        None
    }

    /// The type name. Used by `DESCRIBE` and in result metadata.
    pub fn name(self) -> &'static str {
        use Ty::*;
        match self {
            Null => "NULL",
            Boolean => "BOOLEAN",
            TinyInt => "TINYINT",
            SmallInt => "SMALLINT",
            Int => "INTEGER",
            BigInt => "BIGINT",
            HugeInt => "HUGEINT",
            UTinyInt => "UTINYINT",
            USmallInt => "USMALLINT",
            UInt => "UINTEGER",
            UBigInt => "UBIGINT",
            Float => "FLOAT",
            Double => "DOUBLE",
            Decimal { .. } => "DECIMAL",
            Varchar => "VARCHAR",
            Blob => "BLOB",
            Date => "DATE",
            Time => "TIME",
            Timestamp => "TIMESTAMP",
            Timestamptz => "TIMESTAMPTZ",
            Interval => "INTERVAL",
            Json => "JSON",
            Uuid => "UUID",
        }
    }
}

/// Packs `months` (i32) into the top 32 bits, `days` (i32) into the next 32, and
/// `micros` (i64) into the low 64. Only the dedicated per-field kernels
/// (`expr::kernels::interval_*`) ever take this apart, so no carry can cross a
/// field boundary.
pub fn pack_interval(months: i32, days: i32, micros: i64) -> i128 {
    ((months as i128) << 96) | ((days as u32 as i128) << 64) | (micros as u64 as i128)
}

/// The inverse of `pack_interval`.
pub fn unpack_interval(v: i128) -> (i32, i32, i64) {
    let months = (v >> 96) as i32;
    let days = ((v >> 64) & 0xFFFF_FFFF) as u32 as i32;
    let micros = v as i64;
    (months, days, micros)
}

/// A minimal stringification of INTERVAL. `core::fmt` is unavailable, so this is
/// hand-rolled (used by CSV/JSONL writing and CLI display). It follows DuckDB's
/// rendering, `<N> years <N> months <N> days HH:MM:SS[.ffffff]`, but omits
/// zero-valued components.
pub fn fmt_interval(months: i32, days: i32, micros: i64, out: &mut Vec<u8>) {
    fn push_i64(out: &mut Vec<u8>, v: i64) {
        if v < 0 {
            out.push(b'-');
        }
        let mut buf = [0u8; 20];
        let mut n = 0usize;
        let mut u = v.unsigned_abs();
        loop {
            buf[n] = b'0' + (u % 10) as u8;
            n += 1;
            u /= 10;
            if u == 0 {
                break;
            }
        }
        for i in (0..n).rev() {
            out.push(buf[i]);
        }
    }
    fn push_padded(out: &mut Vec<u8>, v: i64, width: usize) {
        let mut buf = [0u8; 20];
        let mut n = 0usize;
        let mut u = v.unsigned_abs();
        loop {
            buf[n] = b'0' + (u % 10) as u8;
            n += 1;
            u /= 10;
            if u == 0 {
                break;
            }
        }
        for _ in n..width {
            out.push(b'0');
        }
        for i in (0..n).rev() {
            out.push(buf[i]);
        }
    }

    let mut wrote = false;
    if months != 0 {
        let years = months / 12;
        let rem_months = months % 12;
        if years != 0 {
            push_i64(out, years as i64);
            out.extend_from_slice(if years == 1 || years == -1 { b" year" } else { b" years" });
            wrote = true;
        }
        if rem_months != 0 {
            if wrote {
                out.push(b' ');
            }
            push_i64(out, rem_months as i64);
            out.extend_from_slice(if rem_months == 1 || rem_months == -1 {
                b" month"
            } else {
                b" months"
            });
            wrote = true;
        }
    }
    if days != 0 {
        if wrote {
            out.push(b' ');
        }
        push_i64(out, days as i64);
        out.extend_from_slice(if days == 1 || days == -1 { b" day" } else { b" days" });
        wrote = true;
    }
    if micros != 0 || !wrote {
        if wrote {
            out.push(b' ');
        }
        let neg = micros < 0;
        let mut u = micros.unsigned_abs();
        if neg {
            out.push(b'-');
        }
        let hh = u / 3_600_000_000;
        u %= 3_600_000_000;
        let mm = u / 60_000_000;
        u %= 60_000_000;
        let ss = u / 1_000_000;
        let frac = u % 1_000_000;
        push_padded(out, hh as i64, 2);
        out.push(b':');
        push_padded(out, mm as i64, 2);
        out.push(b':');
        push_padded(out, ss as i64, 2);
        if frac != 0 {
            out.push(b'.');
            // Drop trailing zeros.
            let mut digits = [0u8; 6];
            let mut f = frac;
            for i in (0..6).rev() {
                digits[i] = b'0' + (f % 10) as u8;
                f /= 10;
            }
            let mut end = 6;
            while end > 0 && digits[end - 1] == b'0' {
                end -= 1;
            }
            out.extend_from_slice(&digits[..end]);
        }
    }
}

/// One column of the output schema.
#[derive(Clone)]
pub struct Field {
    pub name: String,
    pub ty: Ty,
    pub nullable: bool,
}

impl Field {
    pub fn new(name: impl Into<String>, ty: Ty, nullable: bool) -> Self {
        Field { name: name.into(), ty, nullable }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phys_mapping_collapses_logical_types() {
        assert_eq!(Ty::Date.phys(), PhysType::I32);
        assert_eq!(Ty::Int.phys(), PhysType::I32);
        assert_eq!(Ty::Timestamp.phys(), PhysType::I64);
        assert_eq!(Ty::Time.phys(), PhysType::I64);
        // Unsigned types are promoted to the next larger signed type.
        assert_eq!(Ty::UInt.phys(), PhysType::I64);
        assert_eq!(Ty::UBigInt.phys(), PhysType::I128);
        assert_eq!(Ty::Float.phys(), PhysType::F64);
        assert_eq!(Ty::Decimal { precision: 18, scale: 2 }.phys(), PhysType::I64);
        assert_eq!(Ty::Decimal { precision: 19, scale: 2 }.phys(), PhysType::I128);
    }

    #[test]
    fn unify_numeric() {
        assert_eq!(Ty::unify(Ty::Int, Ty::BigInt), Some(Ty::BigInt));
        assert_eq!(Ty::unify(Ty::Int, Ty::Double), Some(Ty::Double));
        assert_eq!(Ty::unify(Ty::Float, Ty::Int), Some(Ty::Double));
        assert_eq!(Ty::unify(Ty::Null, Ty::Varchar), Some(Ty::Varchar));
        assert_eq!(Ty::unify(Ty::Date, Ty::Timestamp), Some(Ty::Timestamp));
        assert_eq!(Ty::unify(Ty::Varchar, Ty::Int), None);
    }

    #[test]
    fn integers_are_scale_zero_decimals() {
        assert_eq!(Ty::Int.as_decimal(), Some((10, 0)));
        assert_eq!(Ty::BigInt.as_decimal(), Some((19, 0)));
        assert_eq!(Ty::Decimal { precision: 9, scale: 3 }.as_decimal(), Some((9, 3)));
        assert_eq!(Ty::Double.as_decimal(), None);
        assert_eq!(Ty::Varchar.as_decimal(), None);
    }

    #[test]
    fn decimal_precision_is_capped() {
        assert_eq!(Ty::decimal(50, 4), Ty::Decimal { precision: 38, scale: 4 });
        // scale cannot exceed precision.
        assert_eq!(Ty::decimal(5, 9), Ty::Decimal { precision: 5, scale: 5 });
    }

    #[test]
    fn unify_decimal() {
        let a = Ty::Decimal { precision: 10, scale: 2 };
        let b = Ty::Decimal { precision: 12, scale: 4 };
        // Addition and subtraction gain one digit of precision for the carry (as in DuckDB).
        assert_eq!(Ty::unify(a, b), Some(Ty::Decimal { precision: 13, scale: 4 }));
        assert_eq!(Ty::unify(a, Ty::Double), Some(Ty::Double));
    }

    #[test]
    fn interval_is_i128_and_unifies_only_with_itself() {
        assert_eq!(Ty::Interval.phys(), PhysType::I128);
        assert_eq!(Ty::unify(Ty::Interval, Ty::Interval), Some(Ty::Interval));
        assert_eq!(Ty::unify(Ty::Interval, Ty::BigInt), None);
        assert_eq!(Ty::unify(Ty::Null, Ty::Interval), Some(Ty::Interval));
    }

    #[test]
    fn interval_pack_roundtrip() {
        let cases = [
            (0, 0, 0),
            (14, 3, 3_723_000_000),
            (-14, -3, -3_723_000_000),
            (i32::MAX, i32::MIN, i64::MAX),
            (i32::MIN, i32::MAX, i64::MIN),
        ];
        for (m, d, u) in cases {
            let packed = pack_interval(m, d, u);
            assert_eq!(unpack_interval(packed), (m, d, u), "m={m} d={d} u={u}");
        }
    }

    #[test]
    fn interval_format_matches_duckdb_style() {
        fn s(m: i32, d: i32, u: i64) -> String {
            let mut out = Vec::new();
            fmt_interval(m, d, u, &mut out);
            String::from_utf8(out).unwrap()
        }
        assert_eq!(s(0, 0, 0), "00:00:00");
        assert_eq!(s(0, 3, 0), "3 days");
        assert_eq!(s(1, 0, 0), "1 month");
        assert_eq!(s(14, 3, 3_723_000_000), "1 year 2 months 3 days 01:02:03");
        assert_eq!(s(0, -3, 0), "-3 days");
        assert_eq!(s(0, 0, -1), "-00:00:00.000001");
    }
}
