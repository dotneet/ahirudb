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
}

impl Ty {
    pub fn phys(self) -> PhysType {
        use Ty::*;
        match self {
            Boolean => PhysType::Bool,
            Null | TinyInt | SmallInt | Int | UTinyInt | USmallInt | Date => PhysType::I32,
            BigInt | UInt | Time | Timestamp => PhysType::I64,
            HugeInt | UBigInt => PhysType::I128,
            Decimal { precision, .. } => {
                if precision <= 18 {
                    PhysType::I64
                } else {
                    PhysType::I128
                }
            }
            Float | Double => PhysType::F64,
            Varchar | Blob => PhysType::Bytes,
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
        if let (Decimal { precision: p1, scale: s1 }, Decimal { precision: p2, scale: s2 }) = (a, b)
        {
            let scale = s1.max(s2);
            let precision = (p1 - s1).max(p2 - s2) + scale;
            return Some(Decimal { precision: precision.min(38), scale });
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
    fn unify_decimal() {
        let a = Ty::Decimal { precision: 10, scale: 2 };
        let b = Ty::Decimal { precision: 12, scale: 4 };
        assert_eq!(Ty::unify(a, b), Some(Ty::Decimal { precision: 12, scale: 4 }));
        assert_eq!(Ty::unify(a, Ty::Double), Some(Ty::Double));
    }
}
