//! The SQL abstract syntax tree.
//!
//! Expressions are referenced by `u32` indices into an arena rather than `Box`/`Rc`
//! (DESIGN.md §7). That reduces allocations and removes recursive `Drop`, helping both
//! code size and execution speed.

use crate::format::FormatKind;
use crate::prelude::*;
use crate::vector::{Ty, Value};

/// The position of an expression within an `ExprArena`.
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

    /// Operators that stay equivalent when their operands are swapped. Used by predicate pushdown.
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

/// The argument of DuckDB's `COLUMNS(...)` star expression
/// (<https://duckdb.org/docs/lts/sql/expressions/star>).
///
/// A `COLUMNS(...)` item is a star that expands to a *subset* of the input
/// columns, so it is carried on `Expr::Star` rather than being its own node
/// — expansion happens at bind time against the resolved input schema,
/// exactly where `*` already expands.
///
/// Only the three argument forms below are supported. DuckDB's
/// `COLUMNS(lambda)` predicate form, `UNPACK(...)`/`*COLUMNS(...)`
/// unpacking, distributing an enclosing function over the expansion
/// (`min(COLUMNS(*))`), and `* LIKE 'pat'`-style star filtering are all
/// rejected with `UnsupportedFeature` — see `sql::parser::Parser::columns_item`.
pub enum ColumnsSpec {
    /// `COLUMNS(*)`. Same column set as a bare `*`; the `EXCLUDE`/`REPLACE`/
    /// `RENAME` modifiers are written *inside* the parentheses
    /// (`COLUMNS(* EXCLUDE (a))`), which is what DuckDB accepts.
    All,
    /// `COLUMNS('regex')`. Matched with `expr::regex` as an unanchored,
    /// case-sensitive search over each column name (both verified against
    /// `duckdb` v1.4.4: `COLUMNS('um')` matches `num`, `COLUMNS('N.*')`
    /// matches nothing). Matching no column at all is an error, not an
    /// empty expansion.
    Regex(String),
    /// `COLUMNS(['a', 'b'])`. Names are matched case-insensitively, and the
    /// expansion follows *schema* order, not list order (verified against
    /// `duckdb`: `COLUMNS(['name','id'])` yields `id, name`). A listed name
    /// that matches no column is an error — i.e. `COLUMNS` sides with
    /// `EXCLUDE`, not with `RENAME`, on the asymmetry documented on
    /// `Expr::Star::rename`.
    Names(Vec<String>),
}

// Clone is not derived. Now that subqueries are involved, cloning one expression could
// clone an entire query tree. Where needed, only the children Vec is cloned.
pub enum Expr {
    Literal(Value),
    /// An `INTERVAL '...'` literal. The raw value packed by `vector::pack_interval`.
    /// It is separate from `Literal` because `Value::I128` alone would fix the default
    /// logical type to `HUGEINT`, making it indistinguishable from `INTERVAL`.
    IntervalLiteral(i128),
    /// The result of substituting `CURRENT_DATE`/`CURRENT_TIMESTAMP`/`now()` and friends
    /// before binding. Separate from `Literal` for the same reason as `IntervalLiteral`
    /// (the physical representation of a `Value` alone cannot fix the logical type:
    /// `Value::I32` defaults to `INTEGER` and `Value::I64` to `BIGINT`, indistinguishable
    /// from `DATE`/`TIMESTAMP`). The query start time is passed once by the host, and
    /// `Session::prepare` rewrites the syntax tree into this node before binding (see
    /// `sql::now::substitute_now`), so from `plan::compile` onward it is just a constant.
    TypedLiteral(Value, Ty),
    /// A `?` placeholder. 0-based.
    Param(u16),
    ColumnRef {
        qualifier: Option<String>,
        name: String,
    },
    /// `*` or `t.*`. Valid only in a SELECT list.
    /// `EXCLUDE (col, ...)` / `REPLACE (expr AS col, ...)` are DuckDB extensions.
    /// Both are common words that could be column names, so the parser reads them as
    /// keywords only in the context immediately following `*`/`t.*` (see `sql::parser`).
    Star {
        qualifier: Option<String>,
        /// `Some(..)` when this star was written as DuckDB's `COLUMNS(...)`
        /// star expression rather than a bare `*`/`t.*`. It only narrows
        /// *which* columns the expansion produces; everything else
        /// (`exclude`/`replace`/`rename`, the bind-time expansion itself) is
        /// shared with the plain-`*` path. `COLUMNS` never carries a
        /// qualifier — `t.COLUMNS(*)` is not accepted by DuckDB either
        /// (verified: "Scalar Function with name columns does not exist").
        columns: Option<ColumnsSpec>,
        /// Column names excluded from the expansion (compared case-insensitively).
        exclude: Vec<String>,
        /// Replaces the named columns of the expansion with the result of an expression. The column names themselves are unchanged.
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
        /// `TRY_CAST`. Rows that cannot be converted become NULL instead of an error.
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
        /// `ILIKE`. Compares case-insensitively.
        ci: bool,
    },
    /// Both aggregate and scalar functions land here. The binder tells them apart.
    Function {
        name: String,
        args: Vec<ExprId>,
        /// `COUNT(DISTINCT x)`
        distinct: bool,
        /// `COUNT(*)`
        star: bool,
        /// `agg(...) FILTER (WHERE cond)`. Meaningful only for aggregate functions.
        filter: Option<ExprId>,
    },
    /// A window function (`f(...) OVER (PARTITION BY .. ORDER BY ..)`, or the named
    /// reference `f(...) OVER w`).
    Window {
        name: String,
        args: Vec<ExprId>,
        star: bool,
        /// `Some(w)` for `OVER w` (a single identifier). In that case
        /// `partition_by`/`order_by`/`frame` stay unused (empty/default) and the real
        /// definition is looked up from `SelectStmt::windows` by `w` at bind time (the
        /// `WINDOW` clause syntactically follows the SELECT list, so the definition
        /// cannot be known at parse time; see `plan::bind`).
        window_ref: Option<String>,
        partition_by: Vec<ExprId>,
        order_by: Vec<OrderByItem>,
        frame: WindowFrame,
    },
    /// A scalar subquery (`(SELECT ...)`). It must return one row and one column.
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
    /// `UNNEST(expr)`. Writable only in a SELECT list (`UNNEST` in a FROM clause is
    /// `FromItem::Unnest`). The target must be `Ty::Json` (an array). `plan::bind` picks
    /// it up as a special expression that is neither an aggregate nor an ordinary scalar
    /// expression (the same treatment as `FILTER`/`QUALIFY`).
    Unnest(ExprId),
    /// `x -> expr` / `(a, b) -> expr`. The parser produces this only as an argument to
    /// `list_transform`/`list_filter`/`list_reduce` (see `sql::parser::Parser::call`;
    /// anywhere other than an argument position, `->` remains sugar for the JSON path
    /// operator `json_extract`).
    ///
    /// `params` are formal parameter names valid only inside the body, independent of the
    /// columns of the enclosing SQL scope. `plan::compile` compiles a lambda body in a
    /// scope isolated from the enclosing one (see the docs on `Compiler::lambda_call`),
    /// so appearing in an ordinary expression position (the general path of
    /// `plan::compile::Compiler::expr_inner`) is an error.
    Lambda {
        params: Vec<String>,
        body: ExprId,
    },
}

/// The expression arena.
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

    /// An iterator over every node (for `sql::now::substitute_now`).
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
    /// Returns each left row exactly once if the right side has a match. The rewrite
    /// target of `IN (SELECT)` / `EXISTS`. Never produced by syntax; only the binder generates it.
    Semi,
    /// Returns each left row if the right side has no match. For `NOT IN` / `NOT EXISTS`.
    Anti,
    /// NULL-aware ANTI. The rewrite target of `NOT IN (SELECT ...)`, generated only by the
    /// binder. Unlike `Anti`, it reproduces SQL's three-valued logic exactly: if any right
    /// key is NULL the comparison is UNKNOWN and **the result is empty**, and rows whose
    /// left key is NULL are not returned either (unless the right side is empty).
    AntiNullAware,
}

impl JoinKind {
    /// Whether the output is just the left schema (the right columns are not returned).
    pub fn is_semi(self) -> bool {
        matches!(self, JoinKind::Semi | JoinKind::Anti | JoinKind::AntiNullAware)
    }
}

pub enum FromItem {
    /// A registered table name.
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
    /// `UNNEST(expr) [AS alias[(col)]]`. DuckDB can reference columns of preceding FROM
    /// items without an explicit `LATERAL` (implicitly LATERAL), so `expr` is bound with
    /// the constraint that it may only reference columns contributed by its left siblings
    /// (see `plan::bind::flatten_from`/`build_tree`). Appearing alone (`FROM UNNEST(...)`)
    /// or on the left of a JOIN is explicitly rejected as unsupported.
    Unnest {
        expr: ExprId,
        alias: Option<String>,
        column_alias: Option<String>,
    },
    /// `generate_series(start, stop[, step])` / `range([start,] stop[, step])`
    /// Table functions. DuckDB accepts arbitrary expressions as arguments, but there is
    /// no bind-time constant-folding machinery here (`plan::bind` can only compile
    /// expressions that need column references), so v1 accepts literal integers only
    /// (`sql::parser::Parser::signed_int_lit`). `stop` is treated differently by the two
    /// functions: `range` is half-open, `generate_series` is closed (confirmed with the `duckdb` CLI).
    GenerateSeries {
        start: i64,
        stop: i64,
        step: i64,
        /// `true` for `generate_series` (which includes `stop`), `false` for `range`
        /// (which does not).
        inclusive: bool,
        alias: Option<String>,
        column_alias: Option<String>,
    },
}

/// The sampling method of `USING SAMPLE`/`TABLESAMPLE`.
///
/// As far as the `duckdb` CLI shows, each method uses a genuinely different algorithm
/// (`SYSTEM` per vector, `BERNOULLI` per row, `RESERVOIR` only for a row count), but
/// this engine accepts only the syntax and simplifies the implementation to one
/// approach (task priority: percentage > row count > distinguishing methods).
/// See `plan::bind::resolve_sample_spec`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SampleMethod {
    Bernoulli,
    System,
    Reservoir,
}

/// Holds the syntax of `USING SAMPLE <spec>` / `TABLESAMPLE <spec>`.
pub struct SampleSpec {
    pub method: SampleMethod,
    /// A percentage in 0.0..=100.0 when `is_rows` is `false`, or the number of rows to
    /// keep when `true` (held as a float; `duckdb` rounds fractions and accepts them).
    pub amount: f64,
    pub is_rows: bool,
    /// An explicit seed, as in `USING SAMPLE 10% (bernoulli, 42)`. When omitted, a fixed
    /// default is used (determinism is fine, per the task's instruction).
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

/// Set operations.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SetOp {
    Union,
    Intersect,
    Except,
}

/// The body of a query. Nests through set operations.
///
/// `EXCEPT` is not associative (`(a EXCEPT b) EXCEPT c` differs from
/// `a EXCEPT (b EXCEPT c)`), so this is a tree rather than a flat list.
/// The parser must build it left-associatively.
pub enum SetExpr {
    Select(Box<SelectStmt>),
    SetOp {
        op: SetOp,
        /// Whether duplicates are kept, as with `UNION ALL`.
        all: bool,
        left: Box<SetExpr>,
        right: Box<SetExpr>,
    },
}

/// A common table expression (`WITH name AS (...)`).
pub struct Cte {
    pub name: String,
    /// The explicit column names of `name(a, b, ...)`. When empty, the body's schema is used as is.
    /// The parser allows this only under `WITH RECURSIVE` (as per DESIGN.md, column lists
    /// remain unsupported for ordinary `WITH`).
    pub columns: Vec<String>,
    /// Whether this CTE is under a `WITH RECURSIVE`.
    ///
    /// Per standard SQL, it is fine for CTEs that are flagged but do not actually
    /// reference themselves to be mixed in (the `base` side of `WITH RECURSIVE base
    /// AS (...), t AS (... base ... UNION ALL ... t ...)`). Whether recursion is really
    /// needed is decided at bind time (`plan::bind`) by looking at the body.
    pub recursive: bool,
    pub query: Box<QueryStmt>,
}

/// A whole query as a statement.
///
/// `ORDER BY` / `LIMIT` live both here and on `SelectStmt` because the `ORDER BY` of
/// `SELECT ... UNION SELECT ... ORDER BY x` applies to **the whole set-operation
/// result**. A `SELECT` inside a derived table uses its own `SelectStmt` side.
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
    /// Returns the plain SELECT if there is neither a set operation nor a CTE.
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

/// A window function frame. v1 handles only the default frames.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WindowFrame {
    /// `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` (the default when ORDER BY is present)
    RangeUnboundedPreceding,
    /// `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` (the default without ORDER BY)
    WholePartition,
}

/// The body of a named window definition (`WINDOW name AS (...)`).
/// The same shape as `Expr::Window`'s `partition_by`/`order_by`/`frame`.
pub struct WindowDef {
    pub partition_by: Vec<ExprId>,
    pub order_by: Vec<OrderByItem>,
    pub frame: WindowFrame,
}

pub struct SelectStmt {
    pub distinct: bool,
    /// `DISTINCT ON (expr, ...)`. Empty when unused. Mutually exclusive with `distinct`
    /// (the parser sets only one of them).
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
    /// `GROUP BY GROUPING SETS (...)` / `ROLLUP (...)` / `CUBE (...)`.
    /// When `Some`, `group_by` is unused and each element represents one grouping set
    /// (the set of columns used for that round of grouping). `ROLLUP`/`CUBE` are already
    /// expanded by the parser into the corresponding set collections (see `sql::parser`).
    pub grouping_sets: Option<Vec<Vec<ExprId>>>,
    pub having: Option<ExprId>,
    /// `WINDOW name AS (...), ...`. Names are kept in definition order and looked up
    /// case-insensitively from `OVER name` at bind time (see `plan::bind`).
    pub windows: Vec<(String, WindowDef)>,
    /// `QUALIFY`. A filter applied after window functions are evaluated and before ORDER BY.
    pub qualify: Option<ExprId>,
    pub order_by: Vec<OrderByItem>,
    /// `ORDER BY ALL`. Mutually exclusive with `order_by` (see `OrderByAll`).
    pub order_by_all: Option<OrderByAll>,
    pub limit: Option<u64>,
    pub offset: Option<u64>,
    /// `USING SAMPLE` / `TABLESAMPLE`. As far as the `duckdb` CLI shows, semantically it
    /// always applies to the join result of the FROM clause (before filtering) -- `a JOIN
    /// b USING SAMPLE 20 ROWS` picks 20 rows out of the 100 rows after the join.
    /// Syntactically the grammar is flexible enough to place it after any of
    /// WHERE/GROUP BY/HAVING/QUALIFY (`FROM t WHERE x>1 USING SAMPLE 10%` parses too), but
    /// this simplifies to accepting it in exactly one place: right after the FROM clause
    /// and right before WHERE (see `sql::parser::Parser::select_body`).
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

/// Sugar for `PIVOT <from> ON <on> [IN (<value> [AS alias], ...)] USING <agg>[, ...] [GROUP BY <col, ...>]`.
///
/// Semantically it is equivalent to an ordinary aggregate query that "builds
/// `agg(...) FILTER (WHERE on = value)` for each value of `on` and bundles it with the
/// `GROUP BY` columns" (see `plan::bind::desugar_pivot`). The expansion happens just
/// before binding (`session::Session::prepare`) and leaves the existing aggregate
/// binding logic completely untouched.
pub struct PivotStmt {
    pub from: FromItem,
    /// The target expression of `ON`. DuckDB also allows several columns as in `ON a, b`,
    /// but v1 supports only a single expression.
    pub on: ExprId,
    /// `IN (value [AS alias], ...)`. `None` would require automatic value discovery.
    /// At bind time only the target table's schema has been read (the real data has not
    /// been scanned yet), and taking a DISTINCT would require execution in the real sense.
    /// Interposing that as a two-stage query under the `no_std`/streaming-execution
    /// constraints would be a large change, so v1 supports only an explicit `IN`, and
    /// `desugar_pivot` returns `UnsupportedFeature` when it is omitted.
    pub in_list: Option<Vec<(ExprId, Option<String>)>>,
    /// `USING agg(expr) [AS alias], ...`. When empty, the default is `count(*)`, as in
    /// DuckDB. Several aggregates (`USING sum(a), avg(b)`) would require stringifying
    /// expressions to determine column names (not worth the cost given `no_std`/no
    /// `core::fmt`), so v1 supports a single aggregate only (`desugar_pivot` gives `UnsupportedFeature`).
    pub using: Vec<SelectItem>,
    /// An explicit `GROUP BY`. When empty, "every column other than those referenced by
    /// `on`/`using`" (the same default as DuckDB).
    pub group_by: Vec<ExprId>,
    /// The trailing `ORDER BY`/`LIMIT`/`OFFSET`. Carried over unchanged to the expanded
    /// `QueryStmt` (see `plan::bind::desugar_pivot`).
    pub order_by: Vec<OrderByItem>,
    pub limit: Option<u64>,
    pub offset: Option<u64>,
}

/// Sugar for `UNPIVOT <from> ON <col, ...> [INTO NAME <name> VALUE <value>]`.
///
/// Equivalent to an expansion that builds one `SELECT` per target column (passing
/// through the non-target columns, emitting the target column's name as a string literal
/// and its value as is) and bundles them with `UNION ALL` (see
/// `plan::bind::desugar_unpivot`; the same idea as GROUPING SETS bundling several `Node::Aggregate` with `Node::SetOp`).
pub struct UnpivotStmt {
    pub from: FromItem,
    /// The target columns of `ON`. Folding several columns at once, as in DuckDB's
    /// `(a, b), (c, d)` (producing several VALUE columns in one expansion), is
    /// unsupported. Each element must be an unqualified bare column reference.
    pub columns: Vec<ExprId>,
    /// `INTO NAME <name_col>`. Defaults to `"name"` when omitted (as in DuckDB).
    pub name_col: String,
    /// `INTO ... VALUE <value_col>`. Defaults to `"value"` when omitted.
    pub value_col: String,
    /// The trailing `ORDER BY`/`LIMIT`/`OFFSET`. Carried over unchanged to the expanded
    /// `QueryStmt` (see `plan::bind::desugar_unpivot`).
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
    /// Applies only to `catalog::MemTable` (in-memory tables). The read-only
    /// `Source`/`TableFormat` are never touched (DESIGN.md §16).
    #[cfg(feature = "ddl")]
    CreateTable {
        name: String,
        or_replace: bool,
        if_not_exists: bool,
        /// Explicit column definitions. Empty when `as_select` is present.
        columns: Vec<ColumnDef>,
        as_select: Option<Box<QueryStmt>>,
    },
    /// `DROP TABLE [IF EXISTS] t`
    #[cfg(feature = "ddl")]
    DropTable {
        name: String,
        if_exists: bool,
    },
    /// `ALTER TABLE t <action>` (`ADD COLUMN` / `DROP COLUMN` /
    /// `RENAME COLUMN` / `RENAME TO`).
    ///
    /// Applies only to `catalog::MemTable` (in-memory tables). The read-only
    /// `Source`/`TableFormat` are never touched (DESIGN.md §16; the same policy as
    /// `CreateTable`/`DropTable`).
    #[cfg(feature = "ddl")]
    AlterTable {
        name: String,
        action: AlterTableAction,
    },
    /// `CREATE [OR REPLACE] VIEW v AS <query>`
    ///
    /// The view body is held as raw SQL text rather than an AST. Reparsing and rebinding
    /// it at bind time (`plan::bind`) on every reference avoids giving `Catalog` an
    /// `ExprArena` for view definitions (since `catalog` should not depend on `sql::ast`).
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
    /// `ahiru-core` never writes to files (it is `no_std` and cannot touch the
    /// filesystem). `Session::prepare` runs this statement to completion and assembles
    /// the bytes, returning the result (destination path and bytes) on the `Query`.
    /// Actually writing to `path` is the caller's job (`ahiru-cli` on native) -- see the
    /// `write` module docs and DESIGN.md §15.
    #[cfg(feature = "export")]
    Copy {
        query: Box<QueryStmt>,
        path: String,
        /// `(FORMAT csv|jsonl|json)`. Inferred from `path`'s extension when omitted.
        format: Option<String>,
    },
}

/// One column definition of `CREATE TABLE`.
#[cfg(feature = "ddl")]
pub struct ColumnDef {
    pub name: String,
    pub ty: Ty,
    pub nullable: bool,
}

/// The concrete operation of `ALTER TABLE t <action>`.
#[cfg(feature = "ddl")]
pub enum AlterTableAction {
    /// `ADD [COLUMN] col ty [NOT NULL] [DEFAULT expr]`.
    /// Without a `default`, every existing row gets NULL as that column's value.
    AddColumn { name: String, ty: Ty, nullable: bool, default: Option<ExprId> },
    /// `DROP [COLUMN] col`
    DropColumn { name: String },
    /// `RENAME [COLUMN] old TO new`
    RenameColumn { old: String, new: String },
    /// `RENAME TO new_name`
    RenameTable { new_name: String },
}

/// How `INSERT` obtains its rows.
#[cfg(feature = "dml")]
pub enum InsertSource {
    /// `VALUES (a, b), (c, d), ...`. Each row is a list of expressions (the column count is validated during checking).
    Values(Vec<Vec<ExprId>>),
    /// Streamed straight in from `SELECT ...`.
    Query(Box<QueryStmt>),
}

/// The parse result. Holds the expression arena and the statement together.
pub struct Parsed {
    pub arena: ExprArena,
    pub stmt: Stmt,
    /// The number of `?` placeholders.
    pub num_params: u16,
}
