//! 型システム。
//!
//! 論理型と物理型を分離し、**実行カーネルは物理型 6 種に対してのみ**書く。
//! これがカーネル単相化爆発を抑える最大のレバー（DESIGN.md §8, §11）。
//!
//! 時刻系は取り込み時にマイクロ秒へ正規化する。TIMESTAMP(ms/us/ns) を型で
//! 区別しないことで、比較・算術・キャストのカーネルが 1 組で済む。

use crate::prelude::*;

/// 実行カーネルが扱う物理表現。この 6 種以外は存在しない。
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

/// SQL から見える型。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ty {
    /// 型がまだ決まっていない NULL リテラル。
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
    /// precision <= 18 は I64、それ以上は I128 で保持する。
    Decimal {
        precision: u8,
        scale: u8,
    },
    Varchar,
    Blob,
    /// エポックからの日数 (I32)。
    Date,
    /// 深夜からのマイクロ秒 (I64)。
    Time,
    /// エポックからのマイクロ秒 (I64)。
    Timestamp,
    /// 月 (i32) / 日 (i32) / マイクロ秒 (i64) を 1 個の I128 に詰めて持つ
    /// （`pack_interval`）。DuckDB / PostgreSQL と同じ 3 成分モデル。
    Interval,
    /// 動的型付けの JSON 値。物理表現は UTF-8 の JSON テキスト (Bytes)。
    ///
    /// LIST / MAP / STRUCT のような入れ子型を物理型ごとに増やすと
    /// カーネル単相化爆発（DESIGN.md §8, §11 が避けている問題そのもの）
    /// を起こすので、この 1 種類に統合してある。配列・オブジェクトは
    /// JSON テキストのまま持ち、`json_extract`/`list_extract`/`unnest`
    /// 等の関数がその場でパースして取り出す（要素の静的型チェックは
    /// 諦め、JSON 自身の動的型付けに委ねる設計判断）。
    Json,
}

/// DECIMAL の最大 precision。
pub const MAX_DECIMAL_PRECISION: u8 = 38;

impl Ty {
    /// precision を上限で丸めた DECIMAL を作る。
    pub fn decimal(precision: u8, scale: u8) -> Ty {
        let precision = precision.min(MAX_DECIMAL_PRECISION);
        Ty::Decimal { precision, scale: scale.min(precision) }
    }

    /// DECIMAL としての `(precision, scale)`。整数型は scale 0 の DECIMAL と
    /// みなす（DECIMAL と整数の混在演算で使う）。
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
            BigInt | UInt | Time | Timestamp => PhysType::I64,
            HugeInt | UBigInt | Interval => PhysType::I128,
            Decimal { precision, .. } => {
                if precision <= 18 {
                    PhysType::I64
                } else {
                    PhysType::I128
                }
            }
            Float | Double => PhysType::F64,
            Varchar | Blob | Json => PhysType::Bytes,
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
        matches!(self, Ty::Date | Ty::Time | Ty::Timestamp)
    }

    pub fn is_interval(self) -> bool {
        matches!(self, Ty::Interval)
    }

    /// 型の「広さ」。暗黙変換で広い方に寄せるための順序。
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
            Varchar => 13,
            Blob => 14,
            // 他のどの型とも暗黙変換しない（DATE/TIMESTAMP との加減算は
            // `plan::compile` が `Ty::unify` を経由しない専用経路で扱う）。
            Interval => 15,
            // JSON も同様に他の型と暗黙変換しない。
            Json => 16,
        }
    }

    /// 二項演算の共通型を決める。決められない組み合わせは `None`。
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
        // DECIMAL 同士は precision/scale を合わせる。
        // 加減算は桁上がりで 1 桁増えうるので precision に +1 する（DuckDB と同じ）。
        if let (Decimal { precision: p1, scale: s1 }, Decimal { precision: p2, scale: s2 }) = (a, b)
        {
            return Some(Ty::decimal((p1 - s1).max(p2 - s2) + s1.max(s2) + 1, s1.max(s2)));
        }
        // 数値同士は広い方へ。DECIMAL と浮動小数は DOUBLE に落とす。
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
        // DATE と TIMESTAMP の比較は TIMESTAMP に寄せる。
        if matches!((a, b), (Date, Timestamp) | (Timestamp, Date)) {
            return Some(Timestamp);
        }
        if matches!((a, b), (Varchar, Blob) | (Blob, Varchar)) {
            return Some(Blob);
        }
        None
    }

    /// 型名。`DESCRIBE` と結果メタデータで使う。
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
            Interval => "INTERVAL",
            Json => "JSON",
        }
    }
}

/// `months` (i32) を上位 32bit、`days` (i32) を次の 32bit、`micros` (i64) を
/// 下位 64bit に詰める。フィールドごとに演算する専用カーネル
/// （`expr::kernels::interval_*`）でしか解かないので、境界を跨ぐ桁上がりは
/// 起きない。
pub fn pack_interval(months: i32, days: i32, micros: i64) -> i128 {
    ((months as i128) << 96) | ((days as u32 as i128) << 64) | (micros as u64 as i128)
}

/// `pack_interval` の逆関数。
pub fn unpack_interval(v: i128) -> (i32, i32, i64) {
    let months = (v >> 96) as i32;
    let days = ((v >> 64) & 0xFFFF_FFFF) as u32 as i32;
    let micros = v as i64;
    (months, days, micros)
}

/// INTERVAL の最小限の文字列化。`core::fmt` は使えないので手組みする
/// （CSV/JSONL 書き出し・CLI 表示から使う）。DuckDB の表示に寄せて
/// `<N> years <N> months <N> days HH:MM:SS[.ffffff]` の形にするが、
/// ゼロの成分は省く。
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
            // 末尾 0 を落とす。
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

/// 出力スキーマの 1 列。
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
        // 符号なしは 1 段上の符号付きへ昇格させる。
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
        // scale は precision を超えられない。
        assert_eq!(Ty::decimal(5, 9), Ty::Decimal { precision: 5, scale: 5 });
    }

    #[test]
    fn unify_decimal() {
        let a = Ty::Decimal { precision: 10, scale: 2 };
        let b = Ty::Decimal { precision: 12, scale: 4 };
        // 加減算は桁上がりぶん precision が 1 増える（DuckDB と同じ）。
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
