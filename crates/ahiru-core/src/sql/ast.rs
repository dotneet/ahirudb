//! SQL の抽象構文木。
//!
//! 式は `Box`/`Rc` ではなくアリーナ上の `u32` インデックスで参照する
//! （DESIGN.md §7）。確保回数が減り、`Drop` の再帰も消えるので、
//! コードサイズと実行速度の両方に効く。

use crate::prelude::*;
use crate::vector::{Ty, Value};

/// `ExprArena` 内の式の位置。
pub type ExprId = u32;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    /// `||`
    Concat,
}

impl BinaryOp {
    pub fn is_comparison(self) -> bool {
        use BinaryOp::*;
        matches!(self, Eq | Ne | Lt | Le | Gt | Ge)
    }

    pub fn is_logical(self) -> bool {
        matches!(self, BinaryOp::And | BinaryOp::Or)
    }

    /// 左右を入れ替えたときに等価になる演算子。述語プッシュダウンで使う。
    pub fn swapped(self) -> BinaryOp {
        use BinaryOp::*;
        match self {
            Lt => Gt,
            Le => Ge,
            Gt => Lt,
            Ge => Le,
            other => other,
        }
    }
}

#[derive(Clone)]
pub enum Expr {
    Literal(Value),
    /// `?` プレースホルダ。0 始まり。
    Param(u16),
    ColumnRef {
        qualifier: Option<String>,
        name: String,
    },
    /// `*` または `t.*`。SELECT リストでのみ有効。
    Star {
        qualifier: Option<String>,
    },
    Unary {
        op: UnaryOp,
        arg: ExprId,
    },
    Binary {
        op: BinaryOp,
        lhs: ExprId,
        rhs: ExprId,
    },
    Cast {
        arg: ExprId,
        ty: Ty,
    },
    /// `CASE [operand] WHEN .. THEN .. [ELSE ..] END`
    Case {
        operand: Option<ExprId>,
        whens: Vec<(ExprId, ExprId)>,
        else_: Option<ExprId>,
    },
    InList {
        arg: ExprId,
        list: Vec<ExprId>,
        negated: bool,
    },
    Between {
        arg: ExprId,
        low: ExprId,
        high: ExprId,
        negated: bool,
    },
    IsNull {
        arg: ExprId,
        negated: bool,
    },
    Like {
        arg: ExprId,
        pattern: ExprId,
        negated: bool,
        escape: Option<u8>,
    },
    /// 集約関数もスカラ関数もここに入る。区別は binder が行う。
    Function {
        name: String,
        args: Vec<ExprId>,
        /// `COUNT(DISTINCT x)`
        distinct: bool,
        /// `COUNT(*)`
        star: bool,
    },
}

/// 式のアリーナ。
#[derive(Default)]
pub struct ExprArena {
    nodes: Vec<Expr>,
}

impl ExprArena {
    pub fn new() -> Self {
        ExprArena { nodes: Vec::new() }
    }

    pub fn push(&mut self, e: Expr) -> ExprId {
        let id = self.nodes.len() as ExprId;
        self.nodes.push(e);
        id
    }

    #[inline]
    pub fn get(&self, id: ExprId) -> &Expr {
        &self.nodes[id as usize]
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum JoinKind {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

pub enum FromItem {
    /// 登録済みテーブル名。
    Table {
        name: String,
        alias: Option<String>,
    },
    /// `parquet('...')` によるインライン参照。
    Parquet {
        path: String,
        alias: Option<String>,
    },
    Subquery {
        query: Box<SelectStmt>,
        alias: Option<String>,
    },
    Join {
        left: Box<FromItem>,
        right: Box<FromItem>,
        kind: JoinKind,
        on: Option<ExprId>,
    },
}

pub struct SelectItem {
    pub expr: ExprId,
    pub alias: Option<String>,
}

pub struct OrderByItem {
    pub expr: ExprId,
    pub desc: bool,
    pub nulls_first: bool,
}

pub struct SelectStmt {
    pub distinct: bool,
    pub items: Vec<SelectItem>,
    pub from: Option<FromItem>,
    pub filter: Option<ExprId>,
    pub group_by: Vec<ExprId>,
    pub having: Option<ExprId>,
    pub order_by: Vec<OrderByItem>,
    pub limit: Option<u64>,
    pub offset: Option<u64>,
}

impl SelectStmt {
    pub fn empty() -> Self {
        SelectStmt {
            distinct: false,
            items: Vec::new(),
            from: None,
            filter: None,
            group_by: Vec::new(),
            having: None,
            order_by: Vec::new(),
            limit: None,
            offset: None,
        }
    }
}

pub enum Stmt {
    Select(Box<SelectStmt>),
    /// `EXPLAIN <select>`
    Explain(Box<SelectStmt>),
    /// `DESCRIBE <table>` / `DESCRIBE parquet('...')`
    Describe(FromItem),
    /// `SHOW TABLES`
    ShowTables,
}

/// パース結果。式アリーナと文をまとめて持つ。
pub struct Parsed {
    pub arena: ExprArena,
    pub stmt: Stmt,
    /// `?` プレースホルダの個数。
    pub num_params: u16,
}
