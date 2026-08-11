//! 論理プラン。
//!
//! 最適化はルールベースのみ（コストベースは持たない）。効果の大半は
//! 「読むバイト数を減らす」ことから来るので、射影プッシュダウンと述語による
//! 分割枝刈りの 2 つに集中する（DESIGN.md §9）。

pub mod bind;
pub mod compile;
pub mod scope;

use crate::expr::Program;
use crate::prelude::*;
use crate::sql::ast::JoinKind;
use crate::vector::{Field, Ty};

// 枝刈り述語はフォーマット層との契約なので `format` 側に置いてある。
// ここからは再エクスポートするだけ。
pub use crate::format::{range_may_match, PruneOp, Pruner};
pub use scope::Scope;

/// 集約関数。
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum AggKind {
    CountStar,
    Count,
    Sum,
    Min,
    Max,
    Avg,
}

impl AggKind {
    /// 引数の型から結果型を決める。
    ///
    /// **バインダと実行オペレータは必ずこの関数を通すこと。** 別々に決めると
    /// 出力スキーマと実データの型がずれ、結果の読み出しが静かに壊れる。
    pub fn result_ty(self, input: Ty) -> Result<Ty> {
        Ok(match self {
            AggKind::CountStar | AggKind::Count => Ty::BigInt,
            // 整数の合計は 64 ビットで溢れやすいので 128 ビットに広げる。
            AggKind::Sum => match input {
                t if t.is_integer() => Ty::HugeInt,
                Ty::Decimal { precision, scale } => {
                    Ty::Decimal { precision: precision.max(38), scale }
                }
                Ty::Float | Ty::Double => Ty::Double,
                Ty::Null => Ty::HugeInt,
                _ => err!(TypeMismatch),
            },
            AggKind::Avg => match input {
                t if t.is_numeric() || t == Ty::Null => Ty::Double,
                _ => err!(TypeMismatch),
            },
            // MIN/MAX は入力型をそのまま返す。比較さえできればよい。
            AggKind::Min | AggKind::Max => input,
        })
    }

    /// 引数を取らない集約か。
    pub fn is_nullary(self) -> bool {
        self == AggKind::CountStar
    }

    /// 名前から引く。大文字小文字は区別しない。
    pub fn from_name(name: &str) -> Option<AggKind> {
        use crate::rt::hash::eq_ascii_ci;
        let n = name.as_bytes();
        if eq_ascii_ci(n, b"count") {
            Some(AggKind::Count)
        } else if eq_ascii_ci(n, b"sum") {
            Some(AggKind::Sum)
        } else if eq_ascii_ci(n, b"min") {
            Some(AggKind::Min)
        } else if eq_ascii_ci(n, b"max") {
            Some(AggKind::Max)
        } else if eq_ascii_ci(n, b"avg") {
            Some(AggKind::Avg)
        } else {
            None
        }
    }
}

pub struct Agg {
    pub kind: AggKind,
    /// `COUNT(*)` では `None`。
    pub arg: Option<Program>,
    pub distinct: bool,
    pub name: String,
}

impl Agg {
    /// 引数の型。`COUNT(*)` は引数を持たないので `Ty::Null`。
    pub fn input_ty(&self) -> Ty {
        self.arg.as_ref().map_or(Ty::Null, |p| p.result_ty)
    }

    pub fn result_ty(&self) -> Result<Ty> {
        self.kind.result_ty(self.input_ty())
    }
}

pub struct SortKey {
    pub expr: Program,
    pub desc: bool,
    pub nulls_first: bool,
}

pub struct ScanSpec {
    /// カタログ上のテーブル添字。
    pub table: usize,
    /// 読み出す列の添字。射影プッシュダウン後。
    pub columns: Vec<usize>,
    /// スキャンが出力するスキーマ（`columns` と同じ並び）。
    pub schema: Vec<Field>,
    /// 分割の枝刈り用の述語。
    pub pruners: Vec<Pruner>,
}

pub enum Node {
    Scan(Box<ScanSpec>),
    Filter {
        input: Box<Node>,
        pred: Program,
    },
    Project {
        input: Box<Node>,
        exprs: Vec<Program>,
        schema: Vec<Field>,
    },
    Aggregate {
        input: Box<Node>,
        groups: Vec<Program>,
        aggs: Vec<Agg>,
        /// グループキー、続いて集約結果、の順。
        schema: Vec<Field>,
        /// `HAVING`。集約後のスキーマで評価する。
        having: Option<Program>,
    },
    Sort {
        input: Box<Node>,
        keys: Vec<SortKey>,
        /// `ORDER BY ... LIMIT n` は Top-N に落とす。
        limit: Option<usize>,
    },
    Join {
        left: Box<Node>,
        right: Box<Node>,
        kind: JoinKind,
        /// 等値結合のキー。左右で同じ個数。空ならネストループになる。
        left_keys: Vec<Program>,
        right_keys: Vec<Program>,
        /// 等値条件に落とせなかった残りの述語。結合後のスキーマで評価する。
        residual: Option<Program>,
        /// 左のスキーマ、続いて右のスキーマ。
        schema: Vec<Field>,
    },
    Limit {
        input: Box<Node>,
        limit: Option<u64>,
        offset: u64,
    },
}

impl Node {
    /// このノードが出力するスキーマ。
    pub fn schema(&self) -> &[Field] {
        match self {
            Node::Scan(s) => &s.schema,
            Node::Project { schema, .. } => schema,
            Node::Aggregate { schema, .. } => schema,
            Node::Join { schema, .. } => schema,
            Node::Filter { input, .. } | Node::Sort { input, .. } | Node::Limit { input, .. } => {
                input.schema()
            }
        }
    }
}

pub struct Plan {
    pub root: Node,
}
