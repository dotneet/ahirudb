//! 論理プラン。
//!
//! 最適化はルールベースのみ（コストベースは持たない）。効果の大半は
//! 「読むバイト数を減らす」ことから来るので、射影プッシュダウンと述語による
//! 分割枝刈りの 2 つに集中する（DESIGN.md §9）。

pub mod bind;
pub mod compile;
pub mod explain;
pub mod scope;

use crate::expr::Program;
use crate::prelude::*;
use crate::sql::ast::{ExprId, JoinKind};
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
    /// 標本標準偏差（`stddev` / `stddev_samp`）。
    StdDev,
    /// 標本分散（`variance` / `var_samp`）。
    Variance,
    /// 連続分布の中央値（線形補間）。`quantile_cont(x, 0.5)` と同じ。
    Median,
    /// 最頻値。同数が複数あれば最初に見つかったものを返す（実装依存だが
    /// DuckDB も同じ立場）。
    Mode,
    /// 近似個数（実装は v1 では厳密カウントでよい。将来 HyperLogLog に
    /// 差し替える余地として名前だけ分けてある）。
    ApproxCountDistinct,
    /// カンマ等の区切り文字で連結する。第 2 引数は区切り文字（既定は空文字）。
    StringAgg,
    /// 値を集めて JSON 風のテキストにする。LIST 型が無いための代替表現
    /// （DESIGN.md のネスト型の扱いと同じ判断）。
    ArrayAgg,
}

impl AggKind {
    /// 引数の型から結果型を決める。
    ///
    /// **バインダと実行オペレータは必ずこの関数を通すこと。** 別々に決めると
    /// 出力スキーマと実データの型がずれ、結果の読み出しが静かに壊れる。
    pub fn result_ty(self, input: Ty) -> Result<Ty> {
        Ok(match self {
            AggKind::CountStar | AggKind::Count | AggKind::ApproxCountDistinct => Ty::BigInt,
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
            AggKind::Avg | AggKind::StdDev | AggKind::Variance | AggKind::Median => match input {
                t if t.is_numeric() || t == Ty::Null => Ty::Double,
                _ => err!(TypeMismatch),
            },
            // MIN/MAX/MODE は入力型をそのまま返す。
            AggKind::Min | AggKind::Max | AggKind::Mode => input,
            AggKind::StringAgg => Ty::Varchar,
            AggKind::ArrayAgg => Ty::Varchar,
        })
    }

    /// 引数を取らない集約か。
    pub fn is_nullary(self) -> bool {
        self == AggKind::CountStar
    }

    /// `StringAgg` のように 2 個目の引数（区切り文字など）を取りうるか。
    /// 取れる場合、省略時の既定引数を返す。
    ///
    /// `string_agg(x)`（区切り文字省略）は DuckDB では `','` がデフォルト
    /// （`duckdb -c "select string_agg(x) from (values ('p'),('q'),('r')) t(x)"`
    /// が `p,q,r` になることを実測済み。`group_concat` エイリアスも同じ）。
    /// 空文字列ではない。
    pub fn optional_arg_default(self) -> Option<&'static [u8]> {
        match self {
            AggKind::StringAgg => Some(b","),
            _ => None,
        }
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
        } else if eq_ascii_ci(n, b"avg") || eq_ascii_ci(n, b"mean") {
            Some(AggKind::Avg)
        } else if eq_ascii_ci(n, b"stddev") || eq_ascii_ci(n, b"stddev_samp") {
            Some(AggKind::StdDev)
        } else if eq_ascii_ci(n, b"variance") || eq_ascii_ci(n, b"var_samp") {
            Some(AggKind::Variance)
        } else if eq_ascii_ci(n, b"median") {
            Some(AggKind::Median)
        } else if eq_ascii_ci(n, b"mode") {
            Some(AggKind::Mode)
        } else if eq_ascii_ci(n, b"approx_count_distinct") {
            Some(AggKind::ApproxCountDistinct)
        } else if eq_ascii_ci(n, b"string_agg") || eq_ascii_ci(n, b"group_concat") {
            Some(AggKind::StringAgg)
        } else if eq_ascii_ci(n, b"array_agg") || eq_ascii_ci(n, b"list") {
            Some(AggKind::ArrayAgg)
        } else {
            None
        }
    }
}

#[derive(Clone)]
pub struct Agg {
    pub kind: AggKind,
    /// `COUNT(*)` では `None`。
    pub arg: Option<Program>,
    pub distinct: bool,
    pub name: String,
    /// `string_agg(x, sep)` の区切り文字。定数リテラルのみ許す
    /// （行ごとに変わる区切り文字は実用上ほぼ無く、実行を単純に保てる）。
    pub separator: Vec<u8>,
    /// `agg(...) FILTER (WHERE cond)`。集約前の入力スコープで評価する
    /// BOOLEAN 式。偽・NULL の行はこの集約の更新から除外する。
    pub filter: Option<Program>,
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

/// ウィンドウ関数の種類。
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum WindowKind {
    RowNumber,
    Rank,
    DenseRank,
    Lag,
    Lead,
    FirstValue,
    LastValue,
    /// 集約のウィンドウ版（`sum(x) OVER (...)`）。
    Agg(AggKind),
}

impl WindowKind {
    pub fn from_name(name: &str) -> Option<WindowKind> {
        use crate::rt::hash::eq_ascii_ci;
        let n = name.as_bytes();
        if eq_ascii_ci(n, b"row_number") {
            Some(WindowKind::RowNumber)
        } else if eq_ascii_ci(n, b"rank") {
            Some(WindowKind::Rank)
        } else if eq_ascii_ci(n, b"dense_rank") {
            Some(WindowKind::DenseRank)
        } else if eq_ascii_ci(n, b"lag") {
            Some(WindowKind::Lag)
        } else if eq_ascii_ci(n, b"lead") {
            Some(WindowKind::Lead)
        } else if eq_ascii_ci(n, b"first_value") {
            Some(WindowKind::FirstValue)
        } else if eq_ascii_ci(n, b"last_value") {
            Some(WindowKind::LastValue)
        } else {
            AggKind::from_name(name).map(WindowKind::Agg)
        }
    }

    /// 引数を取らない関数か。
    pub fn is_nullary(self) -> bool {
        matches!(self, WindowKind::RowNumber | WindowKind::Rank | WindowKind::DenseRank)
    }
}

/// 1 つのウィンドウ関数呼び出し。
#[derive(Clone)]
pub struct WindowSpec {
    pub kind: WindowKind,
    /// 関数の引数。`row_number()` は空、`lag(x, n, d)` は 3 個まで。
    pub args: Vec<Program>,
    pub partition_by: Vec<Program>,
    pub order_by: Vec<SortKey>,
    pub frame: crate::sql::ast::WindowFrame,
    pub result_ty: crate::vector::Ty,
    pub name: String,
}

/// 集合演算。
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum SetOpKind {
    Union,
    Intersect,
    Except,
}

#[derive(Clone)]
pub struct SortKey {
    pub expr: Program,
    pub desc: bool,
    pub nulls_first: bool,
}

#[derive(Clone)]
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

/// `catalog::MemTable`（インメモリ表）のスキャン。`ddl` フィーチャ専用。
///
/// 常に全列を出す（射影プッシュダウンはしない）。CTE/派生表の `Rel` と同じ
/// 扱いで、必要な列だけを選ぶのは上位の `Project` に任せる —
/// メモリ上に全データがあるので、Parquet の「読むバイト数を減らす」最適化
/// が効く場面がそもそも無い。
#[cfg(feature = "ddl")]
#[derive(Clone)]
pub struct MemScanSpec {
    /// `Catalog::mem_get` に渡す添字（ファイルテーブルの `table` とは
    /// 別の添字空間）。
    pub table: usize,
    pub schema: Vec<Field>,
}

/// `Clone` は GROUPING SETS 対応で要る: FROM/WHERE まで束ねたプラン木を
/// グルーピングセットの数だけ複製し、それぞれに別の `Node::Aggregate` を
/// 被せて `Node::SetOp`（UNION ALL）で束ねる（`plan::bind` 参照）。複製する
/// のは実行前の命令列であって実データではないが、結果として実行時には
/// 同じ入力をセットの数だけスキャンし直すことになる。実行オペレータ
/// （`exec/`）を増やさずに済む方を優先した割り切り。
#[derive(Clone)]
pub enum Node {
    Scan(Box<ScanSpec>),
    #[cfg(feature = "ddl")]
    MemScan(Box<MemScanSpec>),
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
    /// ウィンドウ関数。出力は入力の列に続けてウィンドウ列を並べる。
    Window {
        input: Box<Node>,
        windows: Vec<WindowSpec>,
        schema: Vec<Field>,
    },
    /// 集合演算。左右のスキーマは列数と型が一致していなければならない。
    SetOp {
        left: Box<Node>,
        right: Box<Node>,
        op: SetOpKind,
        /// `UNION ALL` のように重複を残すか。
        all: bool,
        schema: Vec<Field>,
    },
    Limit {
        input: Box<Node>,
        limit: Option<u64>,
        offset: u64,
    },
    /// `DISTINCT ON (keys)`。入力の並び順で最初に見た行だけをキーごとに通す
    /// ストリーミングフィルタ。呼び出し側が `ORDER BY` で希望の並びを先に
    /// 確定させておく（DESIGN.md の「既存インフラの再利用」方針）。
    DistinctOn {
        input: Box<Node>,
        keys: Vec<Program>,
    },
    /// `WITH RECURSIVE name AS (anchor UNION [ALL] recursive_term)`。
    ///
    /// アンカーを 1 度だけ読み切って初期の作業集合にし、それを入力に
    /// `recursive_term` を繰り返し実行して新規行が無くなるまで積み増す
    /// （不動点反復）。`recursive_term` の中にある自己参照は
    /// `Node::WorkingTable` として現れ、実行オペレータ
    /// （`exec::recursive::RecursiveCte`）がイテレーションごとに直前の
    /// 新規行を差し込んで再構築する（`plan::bind::split_recursive_cte`
    /// 参照）。
    RecursiveCte {
        anchor: Box<Node>,
        recursive_term: Box<Node>,
        /// `UNION ALL` なら重複を残す。`UNION` ならアンカー・全イテレーションを
        /// 通して重複排除する。
        union_all: bool,
        schema: Vec<Field>,
    },
    /// `RecursiveCte` の `recursive_term` の中で自分自身を参照する箇所。
    ///
    /// 論理プラン上はスキーマだけを持つ葉で、実データは持たない。
    /// `exec::build` から素で（`RecursiveCte` の外側で）組み立てられることは
    /// バインダのバグなので `Internal` エラーになる。
    WorkingTable {
        schema: Vec<Field>,
    },
    /// `UNNEST`。入力の 1 行を、`expr`（`Ty::Json` の配列）の要素数ぶんの行に
    /// 展開する（1 行 → N 行の set-returning オペレータ）。入力の他の列は
    /// そのまま複製する。SELECT リストの `UNNEST(x)` と FROM 句の
    /// `UNNEST(x) AS t(c)`（暗黙 LATERAL）はどちらもこのノードに落ちる
    /// （`plan::bind` 参照）。
    Unnest {
        input: Box<Node>,
        /// 展開対象の配列を入力行に対して評価する式。結果型は必ず `Ty::Json`。
        expr: Program,
        /// 展開後の要素列の宣言型。全行・全要素を通して常にこの型で出す
        /// （`plan::bind::narrow_unnest_elem_ty` が静的に安全と判定できた
        /// ときだけ `Ty::Json` 以外に絞り込む。実データを見ないと判定でき
        /// ない一般のケース、たとえばテーブルの JSON 列そのものを対象に
        /// する場合は `Ty::Json` のまま。実行側は宣言された型に厳密に
        /// 従う ―― 値が型と食い違えば NULL にする、決してパニックしない）。
        elem_ty: Ty,
        /// 入力のスキーマ ++ 展開要素 1 列。
        schema: Vec<Field>,
    },
    /// `generate_series(start, stop, step)` / `range(start, stop, step)`
    /// テーブル関数。カタログ・I/O を一切経由しない「計算だけのソース」
    /// （`exec::range::GenerateSeries` 参照）。
    GenerateSeries {
        start: i64,
        stop: i64,
        step: i64,
        /// `true` なら `stop` を含む（`generate_series`）。
        inclusive: bool,
        schema: Vec<Field>,
    },
    /// `USING SAMPLE` / `TABLESAMPLE`。入力の行を一定確率・一定行数で間引く。
    /// 列は変えないので `input.schema()` をそのまま使う
    /// （`exec::sample` 参照）。
    Sample {
        input: Box<Node>,
        spec: SampleSpec,
    },
}

/// `Node::Sample` の実行時パラメタ。手法（`BERNOULLI`/`SYSTEM`/`RESERVOIR`）
/// は構文として受理するだけで、実装は `is_rows` の 2 択に落ちる
/// （`plan::bind::resolve_sample_spec` のドキュメント参照）。
#[derive(Clone, Copy)]
pub struct SampleSpec {
    /// `false` ならパーセント指定（0.0..=100.0）、`true` なら行数指定。
    pub is_rows: bool,
    pub amount: f64,
    pub seed: u64,
}

/// `USING SAMPLE`/`TABLESAMPLE` にシード省略時の既定値。呼び出しごとに
/// 変える理由が無い（決定的な方がテストしやすく、`NeedIo` をまたいでも
/// 結果の再現性を保てる。タスクの指示どおり「シードは決定的でよい」）。
pub const DEFAULT_SAMPLE_SEED: u64 = 0x2545_F491_4F6C_DD1D;

impl Node {
    /// このノードが出力するスキーマ。
    pub fn schema(&self) -> &[Field] {
        match self {
            Node::Scan(s) => &s.schema,
            #[cfg(feature = "ddl")]
            Node::MemScan(s) => &s.schema,
            Node::Project { schema, .. } => schema,
            Node::Aggregate { schema, .. } => schema,
            Node::Join { schema, .. } => schema,
            Node::Window { schema, .. } => schema,
            Node::SetOp { schema, .. } => schema,
            Node::RecursiveCte { schema, .. } => schema,
            Node::WorkingTable { schema } => schema,
            Node::Unnest { schema, .. } => schema,
            Node::GenerateSeries { schema, .. } => schema,
            Node::Filter { input, .. }
            | Node::Sort { input, .. }
            | Node::Limit { input, .. }
            | Node::DistinctOn { input, .. }
            | Node::Sample { input, .. } => input.schema(),
        }
    }
}

pub struct Plan {
    pub root: Node,
    /// 相関サブクエリの場合、`root` のスキーマ末尾に付加された相関キー列に
    /// 対応する外側スコープ側の式（`plan::bind` の相関検出結果）。
    /// 非相関なら空で、`root` に余分な列は無い。
    pub correlated: Vec<ExprId>,
}
