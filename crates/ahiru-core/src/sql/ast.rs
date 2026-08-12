//! SQL の抽象構文木。
//!
//! 式は `Box`/`Rc` ではなくアリーナ上の `u32` インデックスで参照する
//! （DESIGN.md §7）。確保回数が減り、`Drop` の再帰も消えるので、
//! コードサイズと実行速度の両方に効く。

use crate::format::FormatKind;
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
    /// `CURRENT_DATE`/`CURRENT_TIMESTAMP`/`now()` 等が束縛前に置き換えられた
    /// 結果。`Literal` と分けているのは `IntervalLiteral` と同じ理由
    /// （`Value` の物理表現だけでは論理型を決められない: `Value::I32` は
    /// 既定で `INTEGER`、`Value::I64` は `BIGINT` になり `DATE`/`TIMESTAMP`
    /// と区別できない）。クエリ開始時刻はホストから 1 度だけ渡され、
    /// `Session::prepare` が構文木をこのノードへ置き換えてから束縛する
    /// （`sql::now::substitute_now` 参照）ので、`plan::compile` 以降は
    /// 単なる定数として扱うだけでよい。
    TypedLiteral(Value, Ty),
    /// `?` プレースホルダ。0 始まり。
    Param(u16),
    ColumnRef {
        qualifier: Option<String>,
        name: String,
    },
    /// `*` または `t.*`。SELECT リストでのみ有効。
    /// `EXCLUDE (col, ...)` / `REPLACE (expr AS col, ...)` は DuckDB 拡張。
    /// どちらも列名として使われうる一般語なので、パーサは `*`/`t.*` の直後
    /// という文脈でだけキーワードとして読む（`sql::parser` 参照）。
    Star {
        qualifier: Option<String>,
        /// 展開結果から除く列名（大小無視で比較）。
        exclude: Vec<String>,
        /// 展開結果のうち指定列を式の評価結果に差し替える。列名自体は変わらない。
        replace: Vec<(ExprId, String)>,
        /// `RENAME (old AS new, ...)`: relabels columns in the expansion,
        /// as `(old_name, new_name)` pairs. Applied after `exclude`/`replace`
        /// (DuckDB's fixed modifier order is EXCLUDE -> REPLACE -> RENAME).
        /// Only the OUTPUT name changes — `WHERE`/`ORDER BY`/etc. in the same
        /// query still see the original column name, and `ORDER BY` also
        /// accepts the new name via the usual output-alias lookup.
        /// Unlike `exclude`, an `old` name that doesn't match any column is
        /// silently ignored rather than an error — this asymmetry matches
        /// DuckDB's real behavior, it is not an oversight.
        rename: Vec<(String, String)>,
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
    /// ウィンドウ関数（`f(...) OVER (PARTITION BY .. ORDER BY ..)`、
    /// または `f(...) OVER w` の名前付き参照）。
    Window {
        name: String,
        args: Vec<ExprId>,
        star: bool,
        /// `OVER w`（識別子 1 つだけ）の場合に `Some(w)`。このとき
        /// `partition_by`/`order_by`/`frame` は未使用（空/既定値）のままで、
        /// 実体は束縛時に `SelectStmt::windows` から `w` を引いて使う
        /// （`WINDOW` 句は構文上 SELECT リストより後に現れるため、パース時点
        /// ではまだ定義を知りえない。`plan::bind` 参照）。
        window_ref: Option<String>,
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
    /// `x <op> ANY (SELECT ...)` / `x <op> ALL (SELECT ...)` / `x <op> SOME (SELECT ...)`.
    /// `SOME` is parsed as an alias of `ANY` (`all: false`); the parser does not
    /// keep the original spelling since the two are fully equivalent in DuckDB.
    /// `op` is always one of the six comparison operators
    /// (`=`/`<>`/`<`/`<=`/`>`/`>=`; see `sql::parser::Parser::expr_body`, the
    /// quantified-comparison lookahead only fires right after one of those
    /// tokens). `plan::bind` desugars `= ANY`/`<> ALL` into `InSubquery`
    /// (exactly equivalent to `IN`/`NOT IN`) and the remaining `<`/`<=`/`>`/`>=`
    /// combinations into a `MIN`/`MAX`-based aggregate comparison, restricted to
    /// non-correlated subqueries; `= ALL`/`<> ANY` are rejected as
    /// `UnsupportedFeature` (see `plan::bind` module doc for the full rationale).
    QuantifiedComparison {
        op: BinaryOp,
        arg: ExprId,
        /// `true` for `ALL`, `false` for `ANY`/`SOME`.
        all: bool,
        query: Box<QueryStmt>,
    },
    /// `UNNEST(expr)`。SELECT リストにのみ書ける（FROM 句の `UNNEST` は
    /// `FromItem::Unnest`）。対象は `Ty::Json`（配列）でなければならない。
    /// 集約でも通常のスカラ式でもない特殊な式として `plan::bind` が拾う
    /// （`FILTER`/`QUALIFY` と同じ扱い）。
    Unnest(ExprId),
    /// `x -> expr` / `(a, b) -> expr`。`list_transform`/`list_filter`/
    /// `list_reduce` の引数としてのみパーサが生成する
    /// （`sql::parser::Parser::call` 参照。関数呼び出しの引数位置以外では
    /// `->` は JSON パス演算子 (`json_extract`) の糖衣構文のまま）。
    ///
    /// `params` は本体の中だけで有効な仮引数名で、外側の SQL スコープの列とは
    /// 独立している。`plan::compile` はラムダ本体を外側とは別の孤立した
    /// スコープでコンパイルする（`Compiler::lambda_call` の doc 参照）ので、
    /// 通常の式の位置（`plan::compile::Compiler::expr_inner` の一般経路）に
    /// 出現した場合はエラーにする。
    Lambda {
        params: Vec<String>,
        body: ExprId,
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

    /// 全ノードを走査するイテレータ（`sql::now::substitute_now` 用）。
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, Expr> {
        self.nodes.iter_mut()
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
    /// An inline reference to a file-backed table, written as a table
    /// function (`parquet('...')`, `read_parquet('...')`, `read_csv('...')`,
    /// `read_csv_auto('...')`, `read_json('...')`, `read_json_auto('...')`)
    /// or as a bare string literal (`FROM 'path'`, format inferred from the
    /// extension via `format::FormatKind::detect`).
    ///
    /// All forms resolve identically: `plan::bind::flatten_from` looks up
    /// `path` verbatim in the `Catalog` (`Catalog::index_of`), exactly like
    /// the original `parquet(...)` did. The host is responsible for having
    /// registered a table under that exact name/path string beforehand
    /// (`Session::register*`) — this engine is `no_std` and has no
    /// filesystem access of its own (see docs/sql/data-sources.md). `format`
    /// therefore does *not* re-dispatch parsing at query time; it only
    /// records which surface syntax was used (for diagnostics/pretty
    /// printing). Named CSV/JSON options (`delim=`, `header=`, ...) and glob
    /// expansion are intentionally unsupported for the same reason — both
    /// would require re-parsing bytes against a different format at bind
    /// time, which is a host-side (`Catalog`-mutating) operation this read
    /// path does not have.
    File {
        path: String,
        format: FormatKind,
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
    /// `generate_series(start, stop[, step])` / `range([start,] stop[, step])`
    /// テーブル関数。DuckDB は任意の式を引数に取れるが、束縛時定数畳み込みの
    /// 仕組みが無い（`plan::bind` は列参照が要る式しかコンパイルできない）ので
    /// v1 はリテラル整数のみを受け付ける（`sql::parser::Parser::signed_int_lit`）。
    /// `stop` の扱いが 2 関数で違う: `range` は半開区間、`generate_series` は
    /// 閉区間（`duckdb` CLI で確認済み）。
    GenerateSeries {
        start: i64,
        stop: i64,
        step: i64,
        /// `true` なら `generate_series`（`stop` を含む）、`false` なら
        /// `range`（`stop` を含まない）。
        inclusive: bool,
        alias: Option<String>,
        column_alias: Option<String>,
    },
}

/// `USING SAMPLE`/`TABLESAMPLE` のサンプリング手法。
///
/// `duckdb` CLI で確認した限り、手法ごとに実際のアルゴリズムが違う
/// （`SYSTEM` はベクタ単位、`BERNOULLI` は行単位、`RESERVOIR` は行数指定
/// 専用）が、このエンジンは構文だけ受理して実装は 1 通りに単純化する
/// （タスクの優先度: パーセント指定 > 行数指定 > 手法の使い分け）。
/// `plan::bind::resolve_sample_spec` 参照。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SampleMethod {
    Bernoulli,
    System,
    Reservoir,
}

/// `USING SAMPLE <spec>` / `TABLESAMPLE <spec>` の構文を保持する。
pub struct SampleSpec {
    pub method: SampleMethod,
    /// `is_rows` が `false` なら 0.0..=100.0 のパーセント、`true` なら
    /// 残す行数（浮動小数のまま持つ。`duckdb` は端数を丸めて受け付ける）。
    pub amount: f64,
    pub is_rows: bool,
    /// `USING SAMPLE 10% (bernoulli, 42)` のような明示シード。省略時は
    /// 固定の既定値を使う（決定的でよい、というタスクの指示どおり）。
    pub seed: Option<i64>,
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

/// `ORDER BY ALL [ASC|DESC] [NULLS FIRST|LAST]` (DuckDB shorthand): sort by
/// every output column, left to right, all with this one direction and null
/// placement.
///
/// Kept as its own field rather than as items inside `order_by` for two
/// reasons. It is resolved at bind time, not parse time — the output column
/// list isn't known until `SELECT *` has been expanded, and `ORDER BY ALL`
/// must cover the expanded columns (verified: `duckdb -c "... select * from
/// t order by all"` sorts by every column). And it is mutually exclusive
/// with an ordinary list: DuckDB's parser rejects `ORDER BY ALL, h` outright,
/// so there is never a mix to represent.
#[derive(Clone, Copy)]
pub struct OrderByAll {
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
    /// `ORDER BY ALL`. Mutually exclusive with `order_by` (see `OrderByAll`).
    pub order_by_all: Option<OrderByAll>,
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

/// 名前付きウィンドウ定義（`WINDOW name AS (...)`）の本体。
/// `Expr::Window` の `partition_by`/`order_by`/`frame` と同じ形。
pub struct WindowDef {
    pub partition_by: Vec<ExprId>,
    pub order_by: Vec<OrderByItem>,
    pub frame: WindowFrame,
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
    /// `GROUP BY ALL` (DuckDB shorthand): group by every select-list
    /// expression that does **not** contain an aggregate.
    ///
    /// Resolved at bind time rather than by the parser, because "contains an
    /// aggregate" is a question about function *names* that only the binder
    /// (`plan::bind::agg::collect_aggregates`) can answer. When set,
    /// `group_by` is empty and unused. Verified against duckdb: `SELECT g+1,
    /// sum(x)+1 ... GROUP BY ALL` groups by `g+1` only — an expression is
    /// excluded when an aggregate appears anywhere inside it, not only when
    /// the item *is* an aggregate call.
    pub group_by_all: bool,
    /// `GROUP BY GROUPING SETS (...)` / `ROLLUP (...)` / `CUBE (...)`。
    /// `Some` のときは `group_by` は使わず、各要素が 1 つのグルーピングセット
    /// （その回のグルーピングに使う列の組）を表す。`ROLLUP`/`CUBE` はパーサが
    /// 対応するセット集合へ展開済み（`sql::parser` 参照）。
    pub grouping_sets: Option<Vec<Vec<ExprId>>>,
    pub having: Option<ExprId>,
    /// `WINDOW name AS (...), ...`。名前は定義順のまま保持し、束縛時に
    /// `OVER name` からこの名前を大小無視で引く（`plan::bind` 参照）。
    pub windows: Vec<(String, WindowDef)>,
    /// `QUALIFY`。ウィンドウ関数評価後・ORDER BY 前に効くフィルタ。
    pub qualify: Option<ExprId>,
    pub order_by: Vec<OrderByItem>,
    /// `ORDER BY ALL`. Mutually exclusive with `order_by` (see `OrderByAll`).
    pub order_by_all: Option<OrderByAll>,
    pub limit: Option<u64>,
    pub offset: Option<u64>,
    /// `USING SAMPLE` / `TABLESAMPLE`。`duckdb` CLI で確認した限り、
    /// 意味的には常に FROM 句の結合結果（フィルタ前）に効く ―― `a JOIN b
    /// USING SAMPLE 20 ROWS` は結合後 100 行から 20 行選ぶ。構文上は
    /// WHERE/GROUP BY/HAVING/QUALIFY のどこの後ろにでも置ける柔軟な文法だが
    /// （`FROM t WHERE x>1 USING SAMPLE 10%` も通る）、ここでは単純化して
    /// FROM 句の直後・WHERE の直前という 1 箇所だけで受理する
    /// （`sql::parser::Parser::select_body` 参照）。
    pub sample: Option<SampleSpec>,
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
            group_by_all: false,
            grouping_sets: None,
            having: None,
            windows: Vec::new(),
            qualify: None,
            order_by: Vec::new(),
            order_by_all: None,
            limit: None,
            offset: None,
            sample: None,
        }
    }
}

/// `PIVOT <from> ON <on> [IN (<値> [AS 別名], ...)] USING <agg>[, ...] [GROUP BY <列, ...>]`
/// の構文糖衣。
///
/// 意味的には「`on` の各値ごとに `agg(...) FILTER (WHERE on = 値)` を作り、
/// `GROUP BY` 対象列と束ねる」という通常の集約クエリに等価
/// （`plan::bind::desugar_pivot` 参照）。展開は束縛の直前
/// （`session::Session::prepare`）に行い、既存の集約束縛ロジックには一切
/// 手を入れない。
pub struct PivotStmt {
    pub from: FromItem,
    /// `ON` の対象式。DuckDB は `ON a, b` の複数列指定も許すが、v1 は
    /// 単一の式のみ対応する。
    pub on: ExprId,
    /// `IN (値 [AS 別名], ...)`。`None` なら値の自動検出が必要になる。
    /// 束縛時点ではまだ対象表のスキーマしか読めておらず（実データはまだ
    /// スキャンしていない）、DISTINCT を取るには本来の意味での実行が要る。
    /// `no_std`/ストリーミング実行の制約下でそれを二段階クエリとして
    /// 挟むのは大掛かりな変更になるため、v1 では明示 `IN` のみ対応し、
    /// 省略時は `desugar_pivot` が `UnsupportedFeature` を返す。
    pub in_list: Option<Vec<(ExprId, Option<String>)>>,
    /// `USING agg(expr) [AS 別名], ...`。空なら DuckDB と同じく既定で
    /// `count(*)`。複数集約関数（`USING sum(a), avg(b)`）は列名決定に式の
    /// 文字列化が要り（`no_std`/`core::fmt` 禁止のコストが見合わない）、
    /// v1 では単一集約関数のみ対応（`desugar_pivot` が `UnsupportedFeature`）。
    pub using: Vec<SelectItem>,
    /// 明示 `GROUP BY`。空なら「`on`/`using` が参照する列以外の全列」
    /// （DuckDB の既定と同じ）。
    pub group_by: Vec<ExprId>,
    /// 末尾の `ORDER BY`/`LIMIT`/`OFFSET`。展開後の `QueryStmt` にそのまま
    /// 移す（`plan::bind::desugar_pivot` 参照）。
    pub order_by: Vec<OrderByItem>,
    pub limit: Option<u64>,
    pub offset: Option<u64>,
}

/// `UNPIVOT <from> ON <col, ...> [INTO NAME <name> VALUE <value>]` の構文糖衣。
///
/// 対象列ごとに 1 本の `SELECT`（対象列以外をそのまま通し、対象列名を文字列
/// リテラルとして・対象列の値をそのまま出す）を作り `UNION ALL` で束ねる
/// 展開に等価（`plan::bind::desugar_unpivot` 参照。GROUPING SETS が複数の
/// `Node::Aggregate` を `Node::SetOp` で束ねたのと同じ発想）。
pub struct UnpivotStmt {
    pub from: FromItem,
    /// `ON` の対象列。DuckDB の `(a, b), (c, d)` のような複数列同時畳み込み
    /// （1 回の展開で複数の VALUE 列を作る形）は非対応。各要素は修飾子なしの
    /// 裸の列参照でなければならない。
    pub columns: Vec<ExprId>,
    /// `INTO NAME <name_col>`。省略時は `"name"`（DuckDB の既定と同じ）。
    pub name_col: String,
    /// `INTO ... VALUE <value_col>`。省略時は `"value"`。
    pub value_col: String,
    /// 末尾の `ORDER BY`/`LIMIT`/`OFFSET`。展開後の `QueryStmt` にそのまま
    /// 移す（`plan::bind::desugar_unpivot` 参照）。
    pub order_by: Vec<OrderByItem>,
    pub limit: Option<u64>,
    pub offset: Option<u64>,
}

pub enum Stmt {
    Select(Box<QueryStmt>),
    /// `EXPLAIN <query>`
    Explain(Box<QueryStmt>),
    /// `PIVOT ...`
    Pivot(Box<PivotStmt>),
    /// `UNPIVOT ...`
    Unpivot(Box<UnpivotStmt>),
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
