//! The logical plan.
//!
//! Optimization is rule-based only (there is no cost model). Most of the benefit comes
//! from "reading fewer bytes", so it concentrates on two things: projection pushdown and
//! split pruning by predicate (DESIGN.md §9).

pub mod bind;
pub mod compile;
pub mod explain;
pub mod scope;

use crate::expr::Program;
use crate::prelude::*;
use crate::sql::ast::{ExprId, JoinKind};
use crate::vector::{Field, Ty};

// Pruning predicates are a contract with the format layer, so they live on the `format`
// side. This only re-exports them.
pub use crate::format::{range_may_match, PruneOp, Pruner};
pub use scope::Scope;

/// Aggregate functions.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum AggKind {
    CountStar,
    Count,
    Sum,
    Min,
    Max,
    Avg,
    /// Sample standard deviation (`stddev` / `stddev_samp`).
    StdDev,
    /// Sample variance (`variance` / `var_samp`).
    Variance,
    /// The continuous-distribution median (linear interpolation). The same as `quantile_cont(x, 0.5)`.
    Median,
    /// The mode. On a tie, the first one found is returned (implementation-defined, but
    /// DuckDB takes the same position).
    Mode,
    /// An approximate count (the v1 implementation may be an exact count; the name is kept
    /// separate to leave room for swapping in HyperLogLog later).
    ApproxCountDistinct,
    /// Concatenates with a separator such as a comma. The second argument is the separator (the default is the empty string).
    StringAgg,
    /// Collects values into JSON-like text. A substitute representation, since there is no
    /// LIST type (the same judgment as DESIGN.md's handling of nested types).
    ArrayAgg,
    /// `any_value` / `first` / `arbitrary`. The first non-NULL input seen.
    AnyValue,
    /// `last`. The last non-NULL input seen.
    Last,
    /// `bool_and` / `bool_or`. NULL inputs are skipped, so a group with only NULLs is NULL.
    BoolAnd,
    BoolOr,
    /// `count_if`. Counts the rows whose BOOLEAN argument is true.
    CountIf,
    /// `product`. Accumulated in f64 (an exact integer product would overflow immediately).
    Product,
    /// Population standard deviation and population variance (`stddev_pop` / `var_pop`).
    /// The same Welford accumulator as the sample versions, divided by `n` instead of `n-1`.
    StdDevPop,
    VarPop,
    /// `quantile_cont` / `percentile_cont` (and `quantile`, which DuckDB defines as the
    /// discrete version but is served here by the continuous one -- see
    /// `docs/sql/functions-aggregate.md`). Linear interpolation, exactly like `Median`; the
    /// fraction is `Agg::quantile`.
    Quantile,
    /// `arg_min` / `arg_max` (aliases `min_by` / `max_by`). Returns the first argument's value
    /// at the row where the second argument is smallest/largest. Both arguments have to be
    /// non-NULL for a row to take part.
    ArgMin,
    ArgMax,
}

/// What an aggregate's second argument means. Every aggregate here takes at most one, and it
/// is either a compile-time constant (folded into `Agg` at bind time) or a second per-row
/// expression (`Agg::arg2`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SecondArg {
    /// No second argument is accepted.
    None,
    /// An optional constant separator, with the default used when omitted (`string_agg`).
    Separator(&'static [u8]),
    /// A required constant fraction in `[0, 1]` (`quantile_cont`).
    Fraction,
    /// A required second expression, evaluated per row (`arg_min`/`arg_max`).
    Expr,
}

impl AggKind {
    /// Determines the result type from the argument type.
    ///
    /// **The binder and the execution operators must both go through this function.**
    /// Deciding separately would make the output schema disagree with the real data's type
    /// and silently break reading the result.
    pub fn result_ty(self, input: Ty) -> Result<Ty> {
        Ok(match self {
            AggKind::CountStar | AggKind::Count | AggKind::ApproxCountDistinct => Ty::BigInt,
            // Integer sums overflow 64 bits easily, so they widen to 128 bits.
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
            // MIN/MAX/MODE return the input type unchanged.
            AggKind::Min | AggKind::Max | AggKind::Mode => input,
            AggKind::StringAgg => Ty::Varchar,
            AggKind::ArrayAgg => Ty::Varchar,
            // These carry a value through rather than computing one, so the type is the
            // argument's (for ARG_MIN/ARG_MAX that is the *first* argument's).
            AggKind::AnyValue | AggKind::Last | AggKind::ArgMin | AggKind::ArgMax => input,
            AggKind::BoolAnd | AggKind::BoolOr => match input {
                Ty::Boolean | Ty::Null => Ty::Boolean,
                _ => err!(TypeMismatch),
            },
            AggKind::CountIf => match input {
                Ty::Boolean | Ty::Null => Ty::BigInt,
                _ => err!(TypeMismatch),
            },
            AggKind::Product | AggKind::StdDevPop | AggKind::VarPop | AggKind::Quantile => {
                match input {
                    t if t.is_numeric() || t == Ty::Null => Ty::Double,
                    _ => err!(TypeMismatch),
                }
            }
        })
    }

    /// What this aggregate's second argument is, if it takes one.
    pub fn second_arg(self) -> SecondArg {
        match self {
            // In DuckDB, `string_agg(x)` (separator omitted) defaults to `','` (measured:
            // `duckdb -c "select string_agg(x) from (values ('p'),('q'),('r')) t(x)"` gives
            // `p,q,r`; the `group_concat` alias behaves the same). It is not the empty string.
            AggKind::StringAgg => SecondArg::Separator(b","),
            AggKind::Quantile => SecondArg::Fraction,
            AggKind::ArgMin | AggKind::ArgMax => SecondArg::Expr,
            _ => SecondArg::None,
        }
    }

    /// Whether the aggregate takes no arguments.
    pub fn is_nullary(self) -> bool {
        self == AggKind::CountStar
    }

    /// Looks up by name. Case-insensitive.
    pub fn from_name(name: &str) -> Option<AggKind> {
        use crate::rt::hash::eq_ascii_ci;
        let n = name.as_bytes();
        // A flat table rather than a chain of `if`s: the list is long enough now that one
        // linear scan over `&'static [u8]` is both smaller and easier to extend.
        const NAMES: &[(&[u8], AggKind)] = &[
            (b"count", AggKind::Count),
            (b"sum", AggKind::Sum),
            (b"min", AggKind::Min),
            (b"max", AggKind::Max),
            (b"avg", AggKind::Avg),
            (b"mean", AggKind::Avg),
            (b"stddev", AggKind::StdDev),
            (b"stddev_samp", AggKind::StdDev),
            (b"variance", AggKind::Variance),
            (b"var_samp", AggKind::Variance),
            (b"stddev_pop", AggKind::StdDevPop),
            (b"var_pop", AggKind::VarPop),
            (b"median", AggKind::Median),
            (b"mode", AggKind::Mode),
            (b"approx_count_distinct", AggKind::ApproxCountDistinct),
            (b"string_agg", AggKind::StringAgg),
            (b"group_concat", AggKind::StringAgg),
            (b"listagg", AggKind::StringAgg),
            (b"array_agg", AggKind::ArrayAgg),
            (b"list", AggKind::ArrayAgg),
            (b"any_value", AggKind::AnyValue),
            (b"first", AggKind::AnyValue),
            (b"arbitrary", AggKind::AnyValue),
            (b"last", AggKind::Last),
            (b"bool_and", AggKind::BoolAnd),
            (b"bool_or", AggKind::BoolOr),
            (b"count_if", AggKind::CountIf),
            (b"countif", AggKind::CountIf),
            (b"product", AggKind::Product),
            (b"quantile", AggKind::Quantile),
            (b"quantile_cont", AggKind::Quantile),
            (b"percentile_cont", AggKind::Quantile),
            (b"arg_min", AggKind::ArgMin),
            (b"min_by", AggKind::ArgMin),
            (b"arg_max", AggKind::ArgMax),
            (b"max_by", AggKind::ArgMax),
        ];
        NAMES.iter().find(|(nm, _)| eq_ascii_ci(n, nm)).map(|&(_, k)| k)
    }
}

#[derive(Clone)]
pub struct Agg {
    pub kind: AggKind,
    /// `None` for `COUNT(*)`.
    pub arg: Option<Program>,
    pub distinct: bool,
    pub name: String,
    /// The separator of `string_agg(x, sep)`. Only a constant literal is allowed
    /// (a per-row separator has almost no practical use, and this keeps execution simple).
    pub separator: Vec<u8>,
    /// The fraction of `quantile_cont(x, frac)`. Only a constant literal is allowed, for the
    /// same reason as `separator`. Ignored (and left at 0.5) by every other aggregate.
    pub quantile: f64,
    /// The second per-row expression of `arg_min`/`arg_max` (the ordering key). `None` for
    /// every other aggregate.
    pub arg2: Option<Program>,
    /// `agg(...) FILTER (WHERE cond)`. A BOOLEAN expression evaluated in the pre-aggregation
    /// input scope. Rows that are false or NULL are excluded from updating this aggregate.
    pub filter: Option<Program>,
}

impl Agg {
    /// The argument type. `COUNT(*)` takes no argument, so it is `Ty::Null`.
    pub fn input_ty(&self) -> Ty {
        self.arg.as_ref().map_or(Ty::Null, |p| p.result_ty)
    }

    pub fn result_ty(&self) -> Result<Ty> {
        self.kind.result_ty(self.input_ty())
    }
}

/// The kind of window function.
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
    /// `nth_value(x, n)`. The n-th row of the frame, 1-based.
    NthValue,
    /// `ntile(n)`. Splits the partition into `n` buckets as evenly as possible.
    NTile,
    /// `percent_rank()`. `(rank - 1) / (rows - 1)`, so it spans 0..1.
    PercentRank,
    /// `cume_dist()`. The fraction of rows at or before this row's peer group.
    CumeDist,
    /// The window version of an aggregate (`sum(x) OVER (...)`).
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
        } else if eq_ascii_ci(n, b"nth_value") {
            Some(WindowKind::NthValue)
        } else if eq_ascii_ci(n, b"ntile") {
            Some(WindowKind::NTile)
        } else if eq_ascii_ci(n, b"percent_rank") {
            Some(WindowKind::PercentRank)
        } else if eq_ascii_ci(n, b"cume_dist") {
            Some(WindowKind::CumeDist)
        } else {
            AggKind::from_name(name).map(WindowKind::Agg)
        }
    }

    /// Whether the function takes no arguments.
    pub fn is_nullary(self) -> bool {
        matches!(
            self,
            WindowKind::RowNumber
                | WindowKind::Rank
                | WindowKind::DenseRank
                | WindowKind::PercentRank
                | WindowKind::CumeDist
        )
    }
}

/// One window function call.
#[derive(Clone)]
pub struct WindowSpec {
    pub kind: WindowKind,
    /// The function's arguments. `row_number()` takes none; `lag(x, n, d)` takes up to three.
    pub args: Vec<Program>,
    pub partition_by: Vec<Program>,
    pub order_by: Vec<SortKey>,
    pub frame: crate::sql::ast::WindowFrame,
    pub result_ty: crate::vector::Ty,
    pub name: String,
}

/// Set operations.
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
    /// The table index in the catalog.
    pub table: usize,
    /// The indices of the columns to read. After projection pushdown.
    pub columns: Vec<usize>,
    /// The schema the scan outputs (in the same order as `columns`).
    pub schema: Vec<Field>,
    /// The predicate used for split pruning.
    pub pruners: Vec<Pruner>,
}

/// A scan of a `catalog::MemTable` (an in-memory table). Exclusive to the `ddl` feature.
///
/// It always emits every column (no projection pushdown). Treated like the `Rel` of a
/// CTE or derived table, with choosing only the needed columns left to the `Project`
/// above -- all the data is in memory, so there is simply no place for Parquet's
/// "read fewer bytes" optimization to help.
#[cfg(feature = "ddl")]
#[derive(Clone)]
pub struct MemScanSpec {
    /// The index passed to `Catalog::mem_get` (a different index space from a file table's `table`).
    pub table: usize,
    pub schema: Vec<Field>,
}

/// `Clone` is needed for GROUPING SETS support: the plan tree bundled up to FROM/WHERE is
/// duplicated once per grouping set, each covered with its own `Node::Aggregate` and
/// bundled with `Node::SetOp` (UNION ALL) (see `plan::bind`). What is duplicated is the
/// pre-execution instruction sequence, not the real data, but as a consequence execution
/// does rescan the same input once per set. A deliberate trade-off, preferring not to add
/// execution operators (`exec/`).
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
        /// Group keys first, then the aggregate results.
        schema: Vec<Field>,
        /// `HAVING`. Evaluated against the post-aggregation schema.
        having: Option<Program>,
    },
    Sort {
        input: Box<Node>,
        keys: Vec<SortKey>,
        /// `ORDER BY ... LIMIT n` is lowered to a Top-N.
        limit: Option<usize>,
    },
    Join {
        left: Box<Node>,
        right: Box<Node>,
        kind: JoinKind,
        /// The equi-join keys. The same count on both sides. Empty means a nested loop.
        left_keys: Vec<Program>,
        right_keys: Vec<Program>,
        /// The remaining predicates that could not be reduced to equality. Evaluated against the post-join schema.
        residual: Option<Program>,
        /// The left schema, followed by the right schema.
        schema: Vec<Field>,
    },
    /// Window functions. The output places the window columns after the input's columns.
    Window {
        input: Box<Node>,
        windows: Vec<WindowSpec>,
        schema: Vec<Field>,
    },
    /// A set operation. The left and right schemas must agree in column count and type.
    SetOp {
        left: Box<Node>,
        right: Box<Node>,
        op: SetOpKind,
        /// Whether duplicates are kept, as with `UNION ALL`.
        all: bool,
        schema: Vec<Field>,
    },
    Limit {
        input: Box<Node>,
        limit: Option<u64>,
        offset: u64,
    },
    /// `DISTINCT ON (keys)`. A streaming filter that passes only the first row seen per key
    /// in the input's order. The caller settles the desired order first with `ORDER BY`
    /// (DESIGN.md's "reuse the existing infrastructure" policy).
    DistinctOn {
        input: Box<Node>,
        keys: Vec<Program>,
    },
    /// `WITH RECURSIVE name AS (anchor UNION [ALL] recursive_term)`.
    ///
    /// The anchor is read to completion once to form the initial working set, and
    /// `recursive_term` is then run repeatedly on that input, accumulating until no new rows
    /// appear (fixed-point iteration). Self-references inside `recursive_term` appear as
    /// `Node::WorkingTable`, and the execution operator (`exec::recursive::RecursiveCte`)
    /// feeds in the previous iteration's new rows and rebuilds each round (see
    /// `plan::bind::split_recursive_cte`).
    RecursiveCte {
        anchor: Box<Node>,
        recursive_term: Box<Node>,
        /// With `UNION ALL`, duplicates are kept. With `UNION`, duplicates are removed
        /// across the anchor and every iteration.
        union_all: bool,
        schema: Vec<Field>,
    },
    /// Where `recursive_term` references itself inside a `RecursiveCte`.
    ///
    /// In the logical plan it is a leaf carrying only a schema, with no real data.
    /// Being built bare by `exec::build` (outside a `RecursiveCte`) is a binder bug and
    /// becomes an `Internal` error.
    WorkingTable {
        schema: Vec<Field>,
    },
    /// `UNNEST`. Expands one input row into as many rows as `expr` (a `Ty::Json` array) has
    /// elements (a 1-row -> N-row set-returning operator). The input's other columns are
    /// duplicated as they are. Both `UNNEST(x)` in a SELECT list and
    /// `UNNEST(x) AS t(c)` in a FROM clause (implicitly LATERAL) lower to this node
    /// (see `plan::bind`).
    Unnest {
        input: Box<Node>,
        /// The expression evaluated per input row to produce the array to expand. Its result type is always `Ty::Json`.
        expr: Program,
        /// The declared type of the expanded element column. Every row and every element is
        /// emitted with this type (`plan::bind::narrow_unnest_elem_ty` narrows it to
        /// something other than `Ty::Json` only when it can prove that statically safe. The
        /// general case, which cannot be decided without seeing the real data -- targeting a
        /// table's JSON column itself, say -- stays `Ty::Json`. Execution follows the
        /// declared type strictly: a value disagreeing with the type becomes NULL, never a panic).
        elem_ty: Ty,
        /// The input schema ++ one expanded element column.
        schema: Vec<Field>,
    },
    /// `generate_series(start, stop, step)` / `range(start, stop, step)`
    /// A table function. A "compute-only source" that goes through neither the catalog nor
    /// I/O (see `exec::range::GenerateSeries`).
    GenerateSeries {
        start: i64,
        stop: i64,
        step: i64,
        /// `true` includes `stop` (`generate_series`).
        inclusive: bool,
        schema: Vec<Field>,
    },
    /// `USING SAMPLE` / `TABLESAMPLE`. Thins the input rows at a fixed probability or to a
    /// fixed row count. It does not change the columns, so `input.schema()` is used as is
    /// (see `exec::sample`).
    Sample {
        input: Box<Node>,
        spec: SampleSpec,
    },
    /// Enforces that `input` produces at most one row (per `keys`, if non-empty), raising
    /// `error::Code::MultipleRowsSubquery` instead of silently keeping only the first. This is
    /// how `plan::bind::select` gives a scalar subquery (`SELECT (SELECT x FROM t)`, or the
    /// correlated form `SELECT (SELECT x FROM t WHERE t.k = outer.k)`) correct SQL semantics:
    /// zero rows still becomes `NULL` through the `LEFT JOIN` placed above this node, exactly
    /// one row becomes that value, and two or more rows is a runtime error rather than a
    /// silently different answer.
    ///
    /// With `keys` empty (the uncorrelated case), every row belongs to the same group, so any
    /// second row is an error; the caller bounds the cost of proving that by first wrapping
    /// `input` in `Node::Limit(2)`, so at most one row beyond the first is ever pulled. With
    /// `keys` non-empty (the correlated case), the check is per correlation key -- the input is
    /// already read in full to build the decorrelating join, so this adds no extra scanning.
    AssertMaxOneRow {
        input: Box<Node>,
        keys: Vec<Program>,
    },
}

/// The runtime parameters of `Node::Sample`. The method
/// (`BERNOULLI`/`SYSTEM`/`RESERVOIR`) is accepted syntactically only; the implementation
/// reduces to the two cases of `is_rows` (see the docs on `plan::bind::resolve_sample_spec`).
#[derive(Clone, Copy)]
pub struct SampleSpec {
    /// `false` means a percentage (0.0..=100.0); `true` means a row count.
    pub is_rows: bool,
    pub amount: f64,
    pub seed: u64,
}

/// The default seed for `USING SAMPLE`/`TABLESAMPLE` when one is omitted. There is no
/// reason to vary it per call (determinism is easier to test and keeps results reproducible
/// across a `NeedIo`; per the task's instruction, "a deterministic seed is fine").
pub const DEFAULT_SAMPLE_SEED: u64 = 0x2545_F491_4F6C_DD1D;

impl Node {
    /// The schema this node outputs.
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
            | Node::Sample { input, .. }
            | Node::AssertMaxOneRow { input, .. } => input.schema(),
        }
    }
}

pub struct Plan {
    pub root: Node,
    /// For a correlated subquery, the expressions in the outer scope corresponding to the
    /// correlation key columns appended to the end of `root`'s schema (the result of
    /// `plan::bind`'s correlation detection). Empty when uncorrelated, with no extra columns on `root`.
    pub correlated: Vec<ExprId>,
}
