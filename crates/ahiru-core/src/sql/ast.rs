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

// Clone は導出しない。サブクエリを含むようになったため、式 1 つの複製が
// クエリ木まるごとの複製になりかねない。必要な場所では子の Vec だけを複製する。
pub enum Expr {
    Literal(Value),
    /// `INTERVAL '...'` リテラル。`vector::pack_interval` で詰めた生の値。
    /// `Literal` と分けているのは、`Value::I128` だけでは既定の論理型が
    /// `HUGEINT` に決まってしまい `INTERVAL` と区別できないため。
    IntervalLiteral(i128),
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
        /// `TRY_CAST`。変換できない行はエラーにせず NULL にする。
        try_: bool,
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
        /// `ILIKE`。大文字小文字を無視して比較する。
        ci: bool,
    },
    /// 集約関数もスカラ関数もここに入る。区別は binder が行う。
    Function {
        name: String,
        args: Vec<ExprId>,
        /// `COUNT(DISTINCT x)`
        distinct: bool,
        /// `COUNT(*)`
        star: bool,
        /// `agg(...) FILTER (WHERE cond)`。集約関数にのみ意味を持つ。
        filter: Option<ExprId>,
    },
    /// ウィンドウ関数（`f(...) OVER (PARTITION BY .. ORDER BY ..)`）。
    Window {
        name: String,
        args: Vec<ExprId>,
        star: bool,
        partition_by: Vec<ExprId>,
        order_by: Vec<OrderByItem>,
        frame: WindowFrame,
    },
    /// スカラサブクエリ（`(SELECT ...)`）。1 行 1 列を返さなければならない。
    ScalarSubquery(Box<QueryStmt>),
    /// `EXISTS (SELECT ...)`
    Exists {
        query: Box<QueryStmt>,
        negated: bool,
    },
    /// `x IN (SELECT ...)`
    InSubquery {
        arg: ExprId,
        query: Box<QueryStmt>,
        negated: bool,
    },
    /// `UNNEST(expr)`。SELECT リストにのみ書ける（FROM 句の `UNNEST` は
    /// `FromItem::Unnest`）。対象は `Ty::Json`（配列）でなければならない。
    /// 集約でも通常のスカラ式でもない特殊な式として `plan::bind` が拾う
    /// （`FILTER`/`QUALIFY` と同じ扱い）。
    Unnest(ExprId),
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
    /// 左の行を、右に一致があれば 1 回だけ返す。`IN (SELECT)` / `EXISTS` の
    /// 書き換え先。構文からは作られず、バインダだけが生成する。
    Semi,
    /// 左の行を、右に一致が無ければ返す。`NOT IN` / `NOT EXISTS` 用。
    Anti,
    /// NULL 対応の ANTI。`NOT IN (SELECT ...)` の書き換え先で、バインダだけが
    /// 生成する。`Anti` との違いは SQL の 3 値論理をそのまま再現する点:
    /// 右のキーに NULL が 1 つでもあれば比較が UNKNOWN になり**結果は空**、
    /// 左のキーが NULL の行も（右が空でない限り）返さない。
    AntiNullAware,
}

impl JoinKind {
    /// 出力が左のスキーマだけになる結合か（右の列を返さない）。
    pub fn is_semi(self) -> bool {
        matches!(self, JoinKind::Semi | JoinKind::Anti | JoinKind::AntiNullAware)
    }
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
        query: Box<QueryStmt>,
        alias: Option<String>,
    },
    Join {
        left: Box<FromItem>,
        right: Box<FromItem>,
        kind: JoinKind,
        on: Option<ExprId>,
    },
    /// `UNNEST(expr) [AS alias[(col)]]`。DuckDB は明示的な `LATERAL` 無しで
    /// 先行する FROM 項目の列を参照できる（暗黙的に LATERAL）ので、`expr` は
    /// 左の兄弟が積んだ列だけを参照できるという制約付きで束縛する
    /// （`plan::bind::flatten_from`/`build_tree` 参照）。単独 (`FROM UNNEST(...)`)
    /// や JOIN の左側での出現は非対応として明確に拒否する。
    Unnest {
        expr: ExprId,
        alias: Option<String>,
        column_alias: Option<String>,
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

/// 集合演算。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SetOp {
    Union,
    Intersect,
    Except,
}

/// クエリ本体。集合演算で入れ子になる。
///
/// `EXCEPT` は結合的でない（`(a EXCEPT b) EXCEPT c` と
/// `a EXCEPT (b EXCEPT c)` は違う）ので、平坦なリストではなく木で持つ。
/// パーサは左結合で組み立てること。
pub enum SetExpr {
    Select(Box<SelectStmt>),
    SetOp {
        op: SetOp,
        /// `UNION ALL` のように重複を残すか。
        all: bool,
        left: Box<SetExpr>,
        right: Box<SetExpr>,
    },
}

/// 共通表式（`WITH name AS (...)`）。
pub struct Cte {
    pub name: String,
    /// `name(a, b, ...)` の明示列名。空なら本体のスキーマをそのまま使う。
    /// `WITH RECURSIVE` の下でのみパーサが許す（DESIGN.md 通り、通常の
    /// `WITH` では列リストは未対応のまま）。
    pub columns: Vec<String>,
    /// `WITH RECURSIVE` の対象になっている CTE か。
    ///
    /// 立っていても実際には自分自身を参照しない CTE（`WITH RECURSIVE base
    /// AS (...), t AS (... base ... UNION ALL ... t ...)` の `base` 側）が
    /// 混ざってよいのは標準 SQL 通り。実際に再帰が要るかどうかは束縛時
    /// （`plan::bind`）に本文を見て判定する。
    pub recursive: bool,
    pub query: Box<QueryStmt>,
}

/// 文としてのクエリ全体。
///
/// `ORDER BY` / `LIMIT` を `SelectStmt` と両方に持つのは、
/// `SELECT ... UNION SELECT ... ORDER BY x` の `ORDER BY` が
/// **集合演算の結果全体**に掛かるため。派生表の中の `SELECT` は自分の
/// `SelectStmt` 側を使う。
pub struct QueryStmt {
    pub ctes: Vec<Cte>,
    pub body: SetExpr,
    pub order_by: Vec<OrderByItem>,
    pub limit: Option<u64>,
    pub offset: Option<u64>,
}

impl QueryStmt {
    /// 集合演算も CTE も無い単純な SELECT ならそれを返す。
    pub fn as_simple_select(&self) -> Option<&SelectStmt> {
        if !self.ctes.is_empty() || !self.order_by.is_empty() || self.limit.is_some() {
            return None;
        }
        match &self.body {
            SetExpr::Select(s) => Some(s),
            SetExpr::SetOp { .. } => None,
        }
    }
}

/// ウィンドウ関数の枠。v1 は既定枠のみ扱う。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WindowFrame {
    /// `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`（ORDER BY あり時の既定）
    RangeUnboundedPreceding,
    /// `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`（ORDER BY 無し時の既定）
    WholePartition,
}

pub struct SelectStmt {
    pub distinct: bool,
    /// `DISTINCT ON (expr, ...)`。空なら未使用。`distinct` とは排他
    /// （パーサがどちらか一方だけを立てる）。
    pub distinct_on: Vec<ExprId>,
    pub items: Vec<SelectItem>,
    pub from: Option<FromItem>,
    pub filter: Option<ExprId>,
    pub group_by: Vec<ExprId>,
    /// `GROUP BY GROUPING SETS (...)` / `ROLLUP (...)` / `CUBE (...)`。
    /// `Some` のときは `group_by` は使わず、各要素が 1 つのグルーピングセット
    /// （その回のグルーピングに使う列の組）を表す。`ROLLUP`/`CUBE` はパーサが
    /// 対応するセット集合へ展開済み（`sql::parser` 参照）。
    pub grouping_sets: Option<Vec<Vec<ExprId>>>,
    pub having: Option<ExprId>,
    /// `QUALIFY`。ウィンドウ関数評価後・ORDER BY 前に効くフィルタ。
    pub qualify: Option<ExprId>,
    pub order_by: Vec<OrderByItem>,
    pub limit: Option<u64>,
    pub offset: Option<u64>,
}

impl SelectStmt {
    pub fn empty() -> Self {
        SelectStmt {
            distinct: false,
            distinct_on: Vec::new(),
            items: Vec::new(),
            from: None,
            filter: None,
            group_by: Vec::new(),
            grouping_sets: None,
            having: None,
            qualify: None,
            order_by: Vec::new(),
            limit: None,
            offset: None,
        }
    }
}

pub enum Stmt {
    Select(Box<QueryStmt>),
    /// `EXPLAIN <query>`
    Explain(Box<QueryStmt>),
    /// `DESCRIBE <table>` / `DESCRIBE parquet('...')`
    Describe(FromItem),
    /// `SHOW TABLES`
    ShowTables,
    /// `CREATE TABLE t (col ty, ...)` / `CREATE TABLE t AS SELECT ...`
    ///
    /// 効くのは `catalog::MemTable`（インメモリ表）のみ。読み取り専用の
    /// `Source`/`TableFormat` には一切触れない（DESIGN.md §16）。
    #[cfg(feature = "ddl")]
    CreateTable {
        name: String,
        or_replace: bool,
        if_not_exists: bool,
        /// 明示列定義。`as_select` があるときは空。
        columns: Vec<ColumnDef>,
        as_select: Option<Box<QueryStmt>>,
    },
    /// `DROP TABLE [IF EXISTS] t`
    #[cfg(feature = "ddl")]
    DropTable {
        name: String,
        if_exists: bool,
    },
    /// `ALTER TABLE t <action>`（`ADD COLUMN` / `DROP COLUMN` /
    /// `RENAME COLUMN` / `RENAME TO`）。
    ///
    /// 効くのは `catalog::MemTable`（インメモリ表）のみ。読み取り専用の
    /// `Source`/`TableFormat` には一切触れない（DESIGN.md §16、
    /// `CreateTable`/`DropTable` と同じ方針）。
    #[cfg(feature = "ddl")]
    AlterTable {
        name: String,
        action: AlterTableAction,
    },
    /// `CREATE [OR REPLACE] VIEW v AS <query>`
    ///
    /// ビュー本体は AST ではなく生 SQL テキストで保持する。参照されるたびに
    /// 束縛時（`plan::bind`）に再パース・再束縛することで、ビュー定義用の
    /// `ExprArena` を `Catalog` に持たせずに済む（`catalog` を `sql::ast` に
    /// 依存させたくないため）。
    #[cfg(feature = "ddl")]
    CreateView {
        name: String,
        or_replace: bool,
        query_sql: String,
    },
    /// `DROP VIEW [IF EXISTS] v`
    #[cfg(feature = "ddl")]
    DropView {
        name: String,
        if_exists: bool,
    },
    /// `INSERT INTO t [(col, ...)] VALUES (...), ... | INSERT INTO t [(...)] SELECT ...`
    #[cfg(feature = "dml")]
    Insert {
        table: String,
        columns: Vec<String>,
        source: InsertSource,
    },
    /// `UPDATE t SET col = expr, ... [WHERE cond]`
    #[cfg(feature = "dml")]
    Update {
        table: String,
        assignments: Vec<(String, ExprId)>,
        filter: Option<ExprId>,
    },
    /// `DELETE FROM t [WHERE cond]`
    #[cfg(feature = "dml")]
    Delete {
        table: String,
        filter: Option<ExprId>,
    },
    /// `COPY (<query>) TO '<path>' [(FORMAT csv|jsonl)]` /
    /// `COPY <table> TO '<path>' [...]`
    ///
    /// `ahiru-core` はファイルへは書かない（`no_std` でファイルシステムに
    /// 触れられない）。`Session::prepare` はこの文を最後まで実行して
    /// バイト列を組み立てるところまでを担い、`Query` にその結果（書き込み先
    /// パスとバイト列）を載せて返す。実際に `path` へ書き込むのは呼び出し側
    /// （ネイティブなら `ahiru-cli`）の役目（`write` モジュール doc、
    /// DESIGN.md §15 参照）。
    #[cfg(feature = "export")]
    Copy {
        query: Box<QueryStmt>,
        path: String,
        /// `(FORMAT csv|jsonl|json)`。省略時は `path` の拡張子から推定する。
        format: Option<String>,
    },
}

/// `CREATE TABLE` の列定義 1 個。
#[cfg(feature = "ddl")]
pub struct ColumnDef {
    pub name: String,
    pub ty: Ty,
    pub nullable: bool,
}

/// `ALTER TABLE t <action>` の具体的な操作。
#[cfg(feature = "ddl")]
pub enum AlterTableAction {
    /// `ADD [COLUMN] col ty [NOT NULL] [DEFAULT expr]`。
    /// `default` が無ければ既存の全行にその列の値として NULL を詰める。
    AddColumn { name: String, ty: Ty, nullable: bool, default: Option<ExprId> },
    /// `DROP [COLUMN] col`
    DropColumn { name: String },
    /// `RENAME [COLUMN] old TO new`
    RenameColumn { old: String, new: String },
    /// `RENAME TO new_name`
    RenameTable { new_name: String },
}

/// `INSERT` が行を得る方法。
#[cfg(feature = "dml")]
pub enum InsertSource {
    /// `VALUES (a, b), (c, d), ...`。各行は式の並び（列数はチェック時に検証）。
    Values(Vec<Vec<ExprId>>),
    /// `SELECT ...` からそのまま流し込む。
    Query(Box<QueryStmt>),
}

/// パース結果。式アリーナと文をまとめて持つ。
pub struct Parsed {
    pub arena: ExprArena,
    pub stmt: Stmt,
    /// `?` プレースホルダの個数。
    pub num_params: u16,
}
