//! スカラ値。定数畳み込み、統計値の比較、結果の取り出しに使う。
//!
//! ホットループでは使わない（1 行ごとに `Value` を作るのは遅い）。
//! ベクタ化カーネルが扱えない境界だけで使うこと。

use crate::prelude::*;
use crate::vector::Ty;

#[derive(Clone, Debug)]
pub enum Value {
    Null,
    Bool(bool),
    I32(i32),
    I64(i64),
    F64(f64),
    I128(i128),
    Bytes(Vec<u8>),
}

impl Value {
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// 整数として取り出す。範囲外・型違いは `None`。
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::I32(v) => Some(*v as i64),
            Value::I64(v) => Some(*v),
            Value::I128(v) => i64::try_from(*v).ok(),
            Value::Bool(b) => Some(*b as i64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F64(v) => Some(*v),
            Value::I32(v) => Some(*v as f64),
            Value::I64(v) => Some(*v as f64),
            Value::I128(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Value::Bytes(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// リテラルから推論される既定の型。
    pub fn default_ty(&self) -> Ty {
        match self {
            Value::Null => Ty::Null,
            Value::Bool(_) => Ty::Boolean,
            Value::I32(_) => Ty::Int,
            Value::I64(_) => Ty::BigInt,
            Value::F64(_) => Ty::Double,
            Value::I128(_) => Ty::HugeInt,
            Value::Bytes(_) => Ty::Varchar,
        }
    }

    /// 同じ物理型同士の順序比較。NULL が絡む場合は `None`。
    /// 統計を使った枝刈り（min/max 比較）で使う。
    pub fn partial_cmp_same(&self, other: &Value) -> Option<core::cmp::Ordering> {
        use Value::*;
        match (self, other) {
            (Null, _) | (_, Null) => None,
            (Bool(a), Bool(b)) => Some(a.cmp(b)),
            (I32(a), I32(b)) => Some(a.cmp(b)),
            (I64(a), I64(b)) => Some(a.cmp(b)),
            (I128(a), I128(b)) => Some(a.cmp(b)),
            (F64(a), F64(b)) => a.partial_cmp(b),
            (Bytes(a), Bytes(b)) => Some(a.as_slice().cmp(b.as_slice())),
            _ => None,
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        use Value::*;
        match (self, other) {
            (Null, Null) => true,
            (Bool(a), Bool(b)) => a == b,
            (I32(a), I32(b)) => a == b,
            (I64(a), I64(b)) => a == b,
            (I128(a), I128(b)) => a == b,
            // 定数畳み込みの同一性判定用。NaN == NaN を真とする点で SQL の
            // 比較演算とは意図的に異なる。
            (F64(a), F64(b)) => a == b || (a.is_nan() && b.is_nan()),
            (Bytes(a), Bytes(b)) => a == b,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_conversions() {
        assert_eq!(Value::I32(5).as_i64(), Some(5));
        assert_eq!(Value::I128(i128::MAX).as_i64(), None);
        assert_eq!(Value::I64(7).as_f64(), Some(7.0));
        assert_eq!(Value::Bytes(b"x".to_vec()).as_i64(), None);
    }

    #[test]
    fn ordering_requires_same_phys_type() {
        use core::cmp::Ordering;
        assert_eq!(Value::I64(1).partial_cmp_same(&Value::I64(2)), Some(Ordering::Less));
        assert_eq!(Value::I64(1).partial_cmp_same(&Value::I32(2)), None);
        assert_eq!(Value::Null.partial_cmp_same(&Value::I64(2)), None);
        assert_eq!(
            Value::Bytes(b"abc".to_vec()).partial_cmp_same(&Value::Bytes(b"abd".to_vec())),
            Some(Ordering::Less)
        );
    }
}
