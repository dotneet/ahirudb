//! Window functions (`OVER (...)`).
//!
//! The output is **the input's columns followed by one column per `WindowSpec`**. The binder
//! builds the schema in that order, so the order is a contract and does not change.
//!
//! ## Blocking and resumption
//!
//! Not even the first row's value is settled until the partition's last row is seen (consider
//! `sum(x) OVER ()`), so this is a **blocking** operator that reads the input through.
//! A remote input can return `Step::NeedIo` / `NeedCodec` midway, so all the partial
//! accumulation state lives in `self`, interruptions are passed straight through, and the next
//! `next()` reads again from the same place (DESIGN.md §6). The unit of ingestion is one batch,
//! and `absorb` never bails out midway, structurally eliminating drops and double reads on resumption.
//! The phases are `Buffering -> Emitting -> Done`.
//!
//! ## Row order
//!
//! Computation happens per partition in ORDER BY order, but **the output must be in the input's
//! row order** (a window does not reorder rows). Values are accumulated in visit order and then
//! `gather`ed back with the inverse permutation "input row -> visit position". That allocates
//! less than scattering cell by cell through `Value`.
//!
//! ## Frames
//!
//! - `WholePartition` (the default without ORDER BY) is the whole partition.
//! - `RangeUnboundedPreceding` (the default with ORDER BY) runs from the partition's start
//!   **to the end of the current row's peers (rows with equal ORDER BY keys)**. It is RANGE
//!   rather than ROWS, so tied rows all share a frame and thus a value.
//!   That is why `sum(x) OVER (ORDER BY y)` returns the same running total for rows with equal y
//!   (confirmed with DuckDB).
//!
//! ## Memory
//!
//! There is no overflow handling. Once accumulation exceeds `MAX_BUFFER_BYTES` it returns `Oom`.

// Shared with the blocking aggregate path so window SUM/AVG over DOUBLE cannot
// drift from the grouped ones; see `Acc::comp`.
use crate::exec::agg::{compensated, neumaier_add};
use crate::exec::rowkey::{encode_key, interval_key, ord_f64, pow10, HashIndex};
use crate::exec::{ExecContext, Operator, Step};
use crate::plan::{AggKind, SortKey, WindowKind, WindowSpec};
use crate::prelude::*;
use crate::sql::ast::WindowFrame;
use crate::vector::{Batch, Bitmap, Data, PhysType, Ty, Value, Vector, BATCH_SIZE};

use core::cmp::Ordering;

/// With no overflow handling, exceeding this returns `Oom`.
/// It is lower than sorting's (256 MiB) because, on top of buffering the input, the window
/// columns and the per-partition index tables are held at the same time.
const MAX_BUFFER_BYTES: usize = 128 * 1024 * 1024;

enum Phase {
    /// Reading and buffering the input. It stays in this state across interruptions.
    Buffering,
    /// The window columns are settled. Returned in `BATCH_SIZE` slices.
    Emitting,
    Done,
}

pub struct Window {
    input: Box<dyn Operator>,
    windows: Vec<WindowSpec>,
    phase: Phase,

    /// The buffered input columns. The first half of the output streams these unchanged.
    cols: Vec<Vector>,
    /// The buffered row count. Even a zero-column input (`count(*) OVER ()`) needs the row count.
    rows: usize,
    /// Whether the column types were decided from the first batch. A zero-column input exists, so
    /// emptiness of `cols` cannot stand in for it.
    init: bool,

    /// The window columns. In the same order as `windows`, in **input row order**. Valid only from `Emitting` on.
    out: Vec<Vector>,
    /// The start of the next row to return.
    pos: usize,
}

impl Window {
    pub fn new(input: Box<dyn Operator>, windows: Vec<WindowSpec>) -> Result<Self> {
        Ok(Window {
            input,
            windows,
            phase: Phase::Buffering,
            cols: Vec::new(),
            rows: 0,
            init: false,
            out: Vec::new(),
            pos: 0,
        })
    }

    /// Takes one whole batch into the buffer. **It never bails out midway** (the unit of resumption is a batch).
    fn absorb(&mut self, mut batch: Batch) -> Result<()> {
        // From here on lookups are by row number, so selection is materialized now.
        batch.materialize();
        let rows = batch.num_rows();
        if rows == 0 {
            return Ok(());
        }
        if !self.init {
            self.cols = batch.cols.iter().map(|c| Vector::new(c.ty())).collect();
            self.init = true;
        }
        ensure!(batch.cols.len() == self.cols.len(), Internal);
        // Row numbers ride in a u32, so beyond that it gives up.
        ensure!(self.rows.saturating_add(rows) <= u32::MAX as usize, LimitExceeded);

        for (dst, src) in self.cols.iter_mut().zip(batch.cols.iter()) {
            append(dst, src)?;
        }
        self.rows += rows;

        let mut bytes = 0usize;
        for v in self.cols.iter() {
            bytes = bytes.saturating_add(vector_bytes(v));
        }
        ensure!(bytes <= MAX_BUFFER_BYTES, Oom);
        Ok(())
    }

    /// The input is read through. Builds every window column and moves to the output phase.
    fn finish(&mut self, ctx: &mut ExecContext) -> Result<()> {
        // Not a single row arrived (every partition was pruned, say). Not even the column types
        // are known, so it moves to output empty without evaluating expressions. `emit` immediately gives `Done`.
        if self.rows == 0 {
            self.phase = Phase::Emitting;
            return Ok(());
        }
        // The buffered columns are temporarily lent to a batch for expression evaluation. Cloning
        // would duplicate the whole input, so ownership is handed over and taken back afterwards.
        let cols = core::mem::take(&mut self.cols);
        let batch = if cols.is_empty() { Batch::rows_only(self.rows) } else { Batch::new(cols) };
        let mut out = Vec::with_capacity(self.windows.len());
        let mut bytes: usize = batch.cols.iter().map(vector_bytes).sum();
        for spec in &self.windows {
            let v = compute(spec, &batch, self.rows, ctx)?;
            bytes = bytes.saturating_add(vector_bytes(&v));
            ensure!(bytes <= MAX_BUFFER_BYTES, Oom);
            out.push(v);
        }
        self.cols = batch.cols;
        self.out = out;
        self.pos = 0;
        self.phase = Phase::Emitting;
        Ok(())
    }

    fn emit(&mut self) -> Result<Step> {
        if self.pos >= self.rows {
            self.phase = Phase::Done;
            self.cols = Vec::new();
            self.out = Vec::new();
            return Ok(Step::Done);
        }
        let end = (self.pos + BATCH_SIZE).min(self.rows);
        let idx: Vec<u32> = (self.pos as u32..end as u32).collect();
        let mut cols = Vec::with_capacity(self.cols.len() + self.out.len());
        for c in self.cols.iter().chain(self.out.iter()) {
            cols.push(c.gather(&idx));
        }
        self.pos = end;
        Ok(Step::Ready(if cols.is_empty() {
            Batch::rows_only(idx.len())
        } else {
            Batch::new(cols)
        }))
    }
}

impl Operator for Window {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        loop {
            match self.phase {
                Phase::Buffering => match self.input.next(ctx)? {
                    Step::Ready(b) => self.absorb(b)?,
                    // The interruption is returned straight up. The buffer stays in `self`, so the
                    // next call pulls the input again from here. Waiting on bytes and waiting on
                    // decompression are handled identically.
                    other @ (Step::NeedIo | Step::NeedCodec) => return Ok(other),
                    Step::Done => self.finish(ctx)?,
                },
                Phase::Emitting => return self.emit(),
                Phase::Done => return Ok(Step::Done),
            }
        }
    }
}

// --- Computing one window function -------------------------------------------

/// Builds `spec`'s result column in input row order.
fn compute(spec: &WindowSpec, batch: &Batch, rows: usize, ctx: &mut ExecContext) -> Result<Vector> {
    let mut pcols = Vec::with_capacity(spec.partition_by.len());
    for p in &spec.partition_by {
        pcols.push(ctx.vm.eval(p, batch)?);
    }
    let mut kcols = Vec::with_capacity(spec.order_by.len());
    for k in &spec.order_by {
        kcols.push(ctx.vm.eval(&k.expr, batch)?);
    }
    let mut acols = Vec::with_capacity(spec.args.len());
    for a in &spec.args {
        acols.push(ctx.vm.eval(a, batch)?);
    }

    // --- Partitioning -------------------------------------------------------
    // Key encoding is delegated to `rowkey`. As with GROUP BY, **NULL joins the same partition
    // as NULL** (that is `encode_key`'s semantics).
    let mut parts: Vec<Vec<u32>> = Vec::new();
    if spec.partition_by.is_empty() {
        parts.push((0..rows as u32).collect());
    } else {
        let refs: Vec<&Vector> = pcols.iter().collect();
        let mut index = HashIndex::new();
        let mut key = Vec::new();
        for r in 0..rows {
            encode_key(&refs, r, &mut key);
            let (slot, is_new) = index.get_or_insert(&key);
            if is_new {
                parts.push(Vec::new());
            }
            match parts.get_mut(slot as usize) {
                Some(p) => p.push(r as u32),
                None => err!(Internal),
            }
        }
    }

    // --- Building values per partition (in visit order) ---------------------
    let mut vals = Vector::with_capacity(spec.result_ty, rows);
    let mut order: Vec<u32> = Vec::with_capacity(rows);
    for part in parts.iter_mut() {
        if !spec.order_by.is_empty() {
            // The comparator follows the same discipline as the sort operator (NULLs by
            // nulls_first, f64 by a total-order key, ties broken by row number = stable).
            part.sort_by(|&a, &b| cmp_row(&spec.order_by, &kcols, a, b));
        }
        eval_partition(spec, part, &kcols, &acols, &mut vals)?;
        order.extend_from_slice(part);
    }
    // Every row is visited exactly once. If that breaks, the inverse permutation breaks.
    ensure!(vals.len() == rows && order.len() == rows, Internal);

    // --- Restoring input row order ------------------------------------------
    let mut inv = vec![0u32; rows];
    for (p, &r) in order.iter().enumerate() {
        inv[r as usize] = p as u32;
    }
    let mut v = vals.gather(&inv);
    v.compact_validity();
    Ok(v)
}

/// Pushes one partition's values into `out` in **visit order** (the order of `part`).
fn eval_partition(
    spec: &WindowSpec,
    part: &[u32],
    kcols: &[Vector],
    acols: &[Vector],
    out: &mut Vector,
) -> Result<()> {
    let n = part.len();
    if n == 0 {
        return Ok(());
    }
    let ty = spec.result_ty;
    // Without ORDER BY every row is a peer, so either frame covers the whole partition.
    let whole = spec.frame == WindowFrame::WholePartition || spec.order_by.is_empty();

    // The peer-group boundaries. `groups[g] = (start, end)` (a half-open interval over `part`).
    let mut groups: Vec<(usize, usize)> = Vec::new();
    if spec.order_by.is_empty() {
        groups.push((0, n));
    } else {
        let mut s = 0usize;
        for i in 1..=n {
            // `i == n` is checked first to avoid indexing `part[i]` out of range.
            if i == n || cmp_keys(&spec.order_by, kcols, part[i - 1], part[i]) != Ordering::Equal {
                groups.push((s, i));
                s = i;
            }
        }
    }

    match spec.kind {
        // A running number from 1, ignoring peers.
        WindowKind::RowNumber => {
            for p in 0..n {
                push_as(out, ty, &Value::I64(p as i64 + 1))?;
            }
        }
        // Ties share a rank, and the next one skips ahead.
        WindowKind::Rank => {
            for &(s, e) in groups.iter() {
                for _ in s..e {
                    push_as(out, ty, &Value::I64(s as i64 + 1))?;
                }
            }
        }
        // Ties share a rank, with no skipping.
        WindowKind::DenseRank => {
            for (gi, &(s, e)) in groups.iter().enumerate() {
                for _ in s..e {
                    push_as(out, ty, &Value::I64(gi as i64 + 1))?;
                }
            }
        }
        // Looked up by relative position in ORDER BY order rather than by frame.
        WindowKind::Lag | WindowKind::Lead => {
            let src = match acols.first() {
                Some(c) => c,
                None => err!(WrongArgCount),
            };
            let back = spec.kind == WindowKind::Lag;
            for p in 0..n {
                let row = part[p] as usize;
                let off = match acols.get(1) {
                    Some(c) => match c.value_at(row).as_i64() {
                        Some(x) => x,
                        // A NULL offset gives a NULL result (the same as DuckDB).
                        None if !c.is_valid(row) => {
                            out.push_null();
                            continue;
                        }
                        None => err!(TypeMismatch),
                    },
                    // Omitted means 1. A negative offset reverses the direction.
                    None => 1,
                };
                // An offset is user data. Checked arithmetic makes the out-of-range result
                // follow the default/NULL branch instead of panicking in debug builds (or
                // wrapping to an unrelated row in release/WASM builds).
                let target =
                    if back { (p as i64).checked_sub(off) } else { (p as i64).checked_add(off) };
                if let Some(target) = target
                    .and_then(|target| usize::try_from(target).ok())
                    .filter(|&target| target < n)
                {
                    let v = src.value_at(part[target] as usize);
                    push_as(out, ty, &v)?;
                } else {
                    // Outside the partition it is the default value, or NULL if none was given.
                    match acols.get(2) {
                        Some(c) => {
                            let v = c.value_at(row);
                            push_as(out, ty, &v)?;
                        }
                        None => out.push_null(),
                    }
                }
            }
        }
        // The frame's start is always the partition's start (UNBOUNDED PRECEDING under either frame).
        WindowKind::FirstValue => {
            let src = match acols.first() {
                Some(c) => c,
                None => err!(WrongArgCount),
            };
            let v = src.value_at(part[0] as usize);
            for _ in 0..n {
                push_as(out, ty, &v)?;
            }
        }
        // The frame's end. Under RANGE it is the last row of the peer group.
        WindowKind::LastValue => {
            let src = match acols.first() {
                Some(c) => c,
                None => err!(WrongArgCount),
            };
            if whole {
                let v = src.value_at(part[n - 1] as usize);
                for _ in 0..n {
                    push_as(out, ty, &v)?;
                }
            } else {
                for &(s, e) in groups.iter() {
                    let v = src.value_at(part[e - 1] as usize);
                    for _ in s..e {
                        push_as(out, ty, &v)?;
                    }
                }
            }
        }
        // `nth_value(x, n)`: the n-th row of the frame, 1-based. The frame always starts at the
        // partition's start (like FirstValue), so the n-th row of the frame is the n-th row of
        // the partition -- but only once the frame has actually reached it, which under a RANGE
        // frame happens at the end of the peer group containing that row.
        WindowKind::NthValue => {
            let src = match acols.first() {
                Some(c) => c,
                None => err!(WrongArgCount),
            };
            for p in 0..n {
                let row = part[p] as usize;
                let k = match acols.get(1) {
                    Some(c) => match c.value_at(row).as_i64() {
                        Some(x) => x,
                        None => {
                            out.push_null();
                            continue;
                        }
                    },
                    None => err!(WrongArgCount),
                };
                // The frame's end: the whole partition, or this row's peer group's last row.
                let end = if whole {
                    n
                } else {
                    match groups.iter().find(|&&(s, e)| p >= s && p < e) {
                        Some(&(_, e)) => e,
                        None => n,
                    }
                };
                match usize::try_from(k).ok().filter(|&k| (1..=end).contains(&k)) {
                    Some(k) => {
                        let v = src.value_at(part[k - 1] as usize);
                        push_as(out, ty, &v)?;
                    }
                    None => {
                        out.push_null();
                    }
                }
            }
        }
        // `ntile(n)`: buckets 1..=n, with the first `rows % n` buckets one row larger.
        WindowKind::NTile => {
            for (p, &r) in part.iter().enumerate() {
                let row = r as usize;
                let buckets = match acols.first().map(|c| c.value_at(row).as_i64()) {
                    Some(Some(b)) => b,
                    // A NULL or missing bucket count gives NULL (DuckDB errors on n < 1; NULL
                    // is this engine's convention for an undefined argument).
                    _ => {
                        out.push_null();
                        continue;
                    }
                };
                if buckets < 1 {
                    out.push_null();
                    continue;
                }
                // On 32-bit WASM, a direct cast can wrap a large positive count to zero and
                // make the division below panic. Counts larger than the partition are equivalent
                // to one bucket per row, so saturating at the partition size is sufficient.
                let b = usize::try_from(buckets).unwrap_or(usize::MAX).min(n.max(1));
                let (base, rem) = (n / b, n % b);
                // Rows 0..rem*(base+1) fall in the larger buckets; the rest in the smaller ones.
                let big = rem * (base + 1);
                let idx = if p < big { p / (base + 1) } else { rem + (p - big) / base.max(1) };
                push_as(out, ty, &Value::I64(idx as i64 + 1))?;
            }
        }
        // `(rank - 1) / (rows - 1)`. A single-row partition is 0 by definition (SQL standard).
        WindowKind::PercentRank => {
            let denom = if n > 1 { (n - 1) as f64 } else { 1.0 };
            for &(s, e) in groups.iter() {
                let v = if n > 1 { s as f64 / denom } else { 0.0 };
                for _ in s..e {
                    push_as(out, ty, &Value::F64(v))?;
                }
            }
        }
        // The fraction of the partition at or before this row's peer group.
        WindowKind::CumeDist => {
            for &(s, e) in groups.iter() {
                let v = e as f64 / n as f64;
                for _ in s..e {
                    push_as(out, ty, &Value::F64(v))?;
                }
            }
        }
        // Aggregation over the frame. A RANGE frame only ever extends forward, so an accumulation
        // with no removal suffices (advancing one peer group at a time).
        WindowKind::Agg(kind) => {
            let src = acols.first();
            // Only aggregates that need nothing but **adding into** the frame have a window
            // version. This implementation's premise is that the frame only extends forward, so
            // removal never has to be considered; median and mode would have to be rebuilt per frame, breaking that premise.
            ensure!(
                matches!(
                    kind,
                    AggKind::CountStar
                        | AggKind::Count
                        | AggKind::Sum
                        | AggKind::Min
                        | AggKind::Max
                        | AggKind::Avg
                        | AggKind::AnyValue
                        | AggKind::Last
                        | AggKind::BoolAnd
                        | AggKind::BoolOr
                        | AggKind::CountIf
                        | AggKind::Product
                ),
                UnsupportedFeature
            );
            let div = match spec.args.first().map(|a| a.result_ty) {
                // DECIMAL's internal representation is an integer. AVG divides back by 10^scale.
                Some(Ty::Decimal { scale, .. }) => pow10(scale),
                _ => 1.0,
            };
            let mut acc = Acc::new();
            if whole {
                for &r in part.iter() {
                    acc.add(kind, src, r as usize)?;
                }
                let v = acc.value(kind, div);
                for _ in 0..n {
                    push_as(out, ty, &v)?;
                }
            } else {
                for &(s, e) in groups.iter() {
                    for &r in part[s..e].iter() {
                        acc.add(kind, src, r as usize)?;
                    }
                    let v = acc.value(kind, div);
                    for _ in s..e {
                        push_as(out, ty, &v)?;
                    }
                }
            }
        }
    }
    Ok(())
}

// --- Accumulating an aggregate over a frame ----------------------------------

/// The aggregate state for one frame. The semantics match `exec::agg`
/// (SUM over integers in i128, AVG in f64, MIN/MAX under a total order with NaN as the maximum).
struct Acc {
    /// The count of non-NULL inputs. For `COUNT(*)`, every row.
    n: i64,
    /// The accumulated value. `Value::Null` means "no non-NULL input yet".
    acc: Value,
    /// The Neumaier compensation term for SUM/AVG over DOUBLE, mirroring
    /// `exec::agg::State::comp`. Without it `sum(x) OVER ()` over ten rows of
    /// `0.1` lands on 0.9999999999999999 while the blocking `sum(x)` over the
    /// same rows returns 1.0 -- the same query, two answers.
    ///
    /// Compensated summation is not exactly reversible, so this would be wrong
    /// for a frame that has to *remove* values as it advances. It is sound here
    /// because this accumulator is strictly additive: as the comment on the
    /// `WindowKind::Agg` arm above records, a frame only ever extends forward
    /// (one peer group at a time), which is also why MEDIAN/MODE have no window
    /// version at all. `value()` folds the term in without consuming it, so the
    /// running state stays intact for the next peer group.
    comp: f64,
}

impl Acc {
    fn new() -> Self {
        Acc { n: 0, acc: Value::Null, comp: 0.0 }
    }

    fn add(&mut self, kind: AggKind, col: Option<&Vector>, row: usize) -> Result<()> {
        if kind == AggKind::CountStar {
            // COUNT(*) counts rows that are all NULL too.
            self.n += 1;
            return Ok(());
        }
        let col = match col {
            Some(c) => c,
            // Everything but COUNT(*) always has an argument.
            None => err!(WrongArgCount),
        };
        // SUM/MIN/MAX/AVG/COUNT(x) ignore NULLs.
        if !col.is_valid(row) {
            return Ok(());
        }
        self.n += 1;
        match kind {
            AggKind::CountStar | AggKind::Count => {}
            AggKind::Sum | AggKind::Avg => match col.data() {
                Data::I32(v) => self.add_int(v[row] as i128)?,
                Data::I64(v) => self.add_int(v[row] as i128)?,
                Data::I128(v) => self.add_int(v[row])?,
                Data::F64(v) => {
                    let (s, comp) = match &self.acc {
                        Value::F64(s) => neumaier_add(*s, self.comp, v[row]),
                        _ => (v[row], 0.0),
                    };
                    self.acc = Value::F64(s);
                    self.comp = comp;
                }
                _ => err!(TypeMismatch),
            },
            AggKind::Min | AggKind::Max => {
                let v = col.value_at(row);
                let take = match &self.acc {
                    Value::Null => true,
                    a => {
                        let c = cmp_val(&v, a, col.ty());
                        if kind == AggKind::Min {
                            c.is_lt()
                        } else {
                            c.is_gt()
                        }
                    }
                };
                if take {
                    self.acc = v;
                }
            }
            // First arrival wins; `Last` keeps overwriting. Both only ever *add* to the frame,
            // so they satisfy the same forward-only premise as SUM/MIN/MAX.
            AggKind::AnyValue => {
                if matches!(self.acc, Value::Null) {
                    self.acc = col.value_at(row);
                }
            }
            AggKind::Last => self.acc = col.value_at(row),
            AggKind::BoolAnd | AggKind::BoolOr => {
                let x = match col.data() {
                    Data::Bool(b) => b.get(row),
                    _ => err!(TypeMismatch),
                };
                let v = match &self.acc {
                    Value::Bool(a) => {
                        if kind == AggKind::BoolAnd {
                            *a && x
                        } else {
                            *a || x
                        }
                    }
                    _ => x,
                };
                self.acc = Value::Bool(v);
            }
            AggKind::CountIf => {
                let hit = matches!(col.data(), Data::Bool(b) if b.get(row));
                let c = match &self.acc {
                    Value::I64(c) => *c,
                    _ => 0,
                };
                self.acc = Value::I64(c + i64::from(hit));
            }
            AggKind::Product => {
                // DECIMAL is physically an integer scaled by 10^scale, so it has to be divided
                // back before multiplying -- the same rule `exec::agg::as_f64_generic` applies to
                // the grouped `product()`. Without it every factor comes out 10^scale too large.
                let div = match col.ty() {
                    Ty::Decimal { scale, .. } => pow10(scale),
                    _ => 1.0,
                };
                let x = match col.data() {
                    Data::I32(v) => v[row] as f64 / div,
                    Data::I64(v) => v[row] as f64 / div,
                    Data::I128(v) => v[row] as f64 / div,
                    Data::F64(v) => v[row],
                    _ => err!(TypeMismatch),
                };
                let p = match &self.acc {
                    Value::F64(p) => p * x,
                    _ => x,
                };
                self.acc = Value::F64(p);
            }
            // An aggregate with no window version. The caller rejects it earlier.
            _ => err!(UnsupportedFeature),
        }
        Ok(())
    }

    fn add_int(&mut self, x: i128) -> Result<()> {
        let s = match &self.acc {
            Value::I128(s) => match s.checked_add(x) {
                Some(v) => v,
                // A sum that overflows even i128 is an error rather than a silent wraparound.
                None => err!(ValueOutOfRange),
            },
            _ => x,
        };
        self.acc = Value::I128(s);
        Ok(())
    }

    fn value(&self, kind: AggKind, div: f64) -> Value {
        match kind {
            AggKind::CountStar | AggKind::Count => Value::I64(self.n),
            // A frame with not a single non-NULL input is NULL.
            // SUM over DOUBLE folds the compensation term back in here, exactly
            // once per emitted value, without clearing it.
            AggKind::Sum => match &self.acc {
                Value::F64(s) => Value::F64(compensated(*s, self.comp)),
                v => v.clone(),
            },
            AggKind::Min
            | AggKind::Max
            | AggKind::AnyValue
            | AggKind::Last
            | AggKind::BoolAnd
            | AggKind::BoolOr
            | AggKind::Product => self.acc.clone(),
            // A frame with no true row is 0, not NULL (the same rule as the grouped version).
            AggKind::CountIf => match &self.acc {
                Value::I64(c) => Value::I64(*c),
                _ => Value::I64(0),
            },
            AggKind::Avg => match &self.acc {
                // Integers are summed exactly in i128 and divided exactly once.
                Value::I128(s) if self.n > 0 => Value::F64(*s as f64 / div / self.n as f64),
                Value::F64(s) if self.n > 0 => {
                    Value::F64(compensated(*s, self.comp) / self.n as f64)
                }
                _ => Value::Null,
            },
            // Aggregates with no window version never reach here (rejected before `add`).
            _ => Value::Null,
        }
    }
}

/// Compares two values of the same physical type. NaN is "greater than everything" (as in `exec::agg`).
fn cmp_val(a: &Value, b: &Value, ty: Ty) -> Ordering {
    match (a, b) {
        (Value::F64(x), Value::F64(y)) => ord_f64(*x, *y),
        // INTERVAL compares on its normalized microsecond span (`rowkey::interval_key`),
        // matching `exec::agg::cmp_at` and the sort comparator.
        (Value::I128(x), Value::I128(y)) if ty == Ty::Interval => {
            interval_key(*x).cmp(&interval_key(*y))
        }
        // A physical type mismatch is an upstream bug. No ordering is imposed; they count as equal.
        _ => a.partial_cmp_same(b).unwrap_or(Ordering::Equal),
    }
}

/// Pushes one value into a column of type `ty`.
///
/// **The binder's chosen output type is authoritative** for `result_ty`, so the value is adapted
/// to it. `Vector::push_value` silently does nothing on a physical type mismatch (which would
/// desynchronize the column length), so conversion always happens here first.
fn push_as(out: &mut Vector, ty: Ty, v: &Value) -> Result<()> {
    if v.is_null() {
        out.push_null();
        return Ok(());
    }
    match ty.phys() {
        PhysType::Bool => match v.as_bool() {
            Some(b) => out.push_value(&Value::Bool(b)),
            None => err!(TypeMismatch),
        },
        PhysType::I32 => match i32::try_from(int_of(v)?) {
            Ok(x) => out.push_value(&Value::I32(x)),
            Err(_) => err!(ValueOutOfRange),
        },
        PhysType::I64 => match i64::try_from(int_of(v)?) {
            Ok(x) => out.push_value(&Value::I64(x)),
            Err(_) => err!(ValueOutOfRange),
        },
        PhysType::I128 => out.push_value(&Value::I128(int_of(v)?)),
        PhysType::F64 => match v.as_f64() {
            Some(x) => out.push_value(&Value::F64(x)),
            None => err!(TypeMismatch),
        },
        PhysType::Bytes => match v.as_bytes() {
            Some(b) => out.push_value(&Value::Bytes(b.to_vec())),
            None => err!(TypeMismatch),
        },
    }
    Ok(())
}

fn int_of(v: &Value) -> Result<i128> {
    Ok(match v {
        Value::Bool(b) => *b as i128,
        Value::I32(x) => *x as i128,
        Value::I64(x) => *x as i128,
        Value::I128(x) => *x,
        _ => err!(TypeMismatch),
    })
}

// --- Comparison ---------------------------------------------------------------
//
// The same discipline as `exec::sort`'s comparator. Its `cmp_row` / `cmp_data` / `f64_key` are
// private, so they are **deliberately duplicated** (sort.rs is finished and, by agreement, not
// touched). When changing the behavior, always change both places.

/// A total-order comparison of two rows. Equal keys are broken by row number, so it is stable.
fn cmp_row(keys: &[SortKey], cols: &[Vector], a: u32, b: u32) -> Ordering {
    match cmp_keys(keys, cols, a, b) {
        Ordering::Equal => a.cmp(&b),
        o => o,
    }
}

/// Comparison of keys alone. `Equal` means "they are peers (tied)".
fn cmp_keys(keys: &[SortKey], cols: &[Vector], a: u32, b: u32) -> Ordering {
    let (ai, bi) = (a as usize, b as usize);
    for (k, c) in keys.iter().zip(cols.iter()) {
        let (va, vb) = (c.is_valid(ai), c.is_valid(bi));
        if !va || !vb {
            if !va && !vb {
                // Two NULLs are tied. They become peers.
                continue;
            }
            // NULL placement is decided by nulls_first alone. Applying desc here would apply the
            // binder's default (ASC->LAST / DESC->FIRST) twice.
            return if !va {
                if k.nulls_first {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            } else if k.nulls_first {
                Ordering::Greater
            } else {
                Ordering::Less
            };
        }
        let mut o = cmp_data(c, ai, bi);
        if k.desc {
            o = o.reverse();
        }
        if o != Ordering::Equal {
            return o;
        }
    }
    Ordering::Equal
}

fn cmp_data(c: &Vector, a: usize, b: usize) -> Ordering {
    match c.data() {
        Data::Bool(v) => v.get(a).cmp(&v.get(b)),
        Data::I32(v) => v[a].cmp(&v[b]),
        Data::I64(v) => v[a].cmp(&v[b]),
        // INTERVAL is normalized to microseconds first, like `exec::sort::cmp_data`.
        Data::I128(v) if c.ty() == Ty::Interval => interval_key(v[a]).cmp(&interval_key(v[b])),
        Data::I128(v) => v[a].cmp(&v[b]),
        Data::F64(v) => f64_key(v[a]).cmp(&f64_key(v[b])),
        Data::Bytes(v) => v.get(a).cmp(v.get(b)),
    }
}

/// Maps an f64 to an order-preserving `u64`. `partial_cmp` returns `None` for NaN, and
/// collapsing that into "equal" would break transitivity, so it is not used.
/// The order is `-inf < ... < -0.0 = 0.0 < ... < +inf < NaN`.
#[inline]
fn f64_key(v: f64) -> u64 {
    if v.is_nan() {
        return u64::MAX;
    }
    let b = if v == 0.0 { 0 } else { v.to_bits() };
    if b >> 63 != 0 {
        !b
    } else {
        b | (1 << 63)
    }
}

// --- Buffer operations --------------------------------------------------------
// These are the same as `exec::sort`'s private helpers.

/// Appends every row of `src` to the end of `dst`. Assumes the physical types match.
fn append(dst: &mut Vector, src: &Vector) -> Result<()> {
    let base = dst.len();
    let n = src.len();
    match (dst.data_mut(), src.data()) {
        (Data::Bool(d), Data::Bool(s)) => d.extend(s),
        (Data::I32(d), Data::I32(s)) => d.extend_from_slice(s),
        (Data::I64(d), Data::I64(s)) => d.extend_from_slice(s),
        (Data::F64(d), Data::F64(s)) => d.extend_from_slice(s),
        (Data::I128(d), Data::I128(s)) => d.extend_from_slice(s),
        (Data::Bytes(d), Data::Bytes(s)) => {
            let first = s.offsets.first().copied().unwrap_or(0);
            let shift = d.data.len() as u32;
            d.data.extend_from_slice(&s.data);
            for &o in s.offsets.iter().skip(1) {
                d.offsets.push(shift + (o - first));
            }
        }
        // The columns come from the same operator, so landing here is an assembly-side bug.
        _ => err!(Internal),
    }
    // If either side has NULLs, validity is aligned. Without extending it, its length would
    // desynchronize from the data and later `is_valid` calls would read out of range.
    if n > 0 && (src.has_nulls() || dst.has_nulls()) {
        let bm: &mut Bitmap = dst.validity_mut();
        if let Some(sv) = src.validity() {
            for i in 0..n {
                if !sv.get(i) {
                    bm.set(base + i, false);
                }
            }
        }
    }
    Ok(())
}

/// The approximate byte count of one vector. For the cap check, so it need not be exact.
fn vector_bytes(v: &Vector) -> usize {
    let n = v.len();
    let body = match v.data() {
        Data::Bool(_) => n / 8 + 1,
        Data::I32(_) => n * 4,
        Data::I64(_) | Data::F64(_) => n * 8,
        Data::I128(_) => n * 16,
        Data::Bytes(b) => b.data.len() + (n + 1) * 4,
    };
    body + if v.has_nulls() { n / 8 + 1 } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Catalog;
    use crate::error::code_of;
    use crate::expr::vm::Vm;
    use crate::expr::{Instr, OpCode, Program};

    // --- Construction helpers -----------------------------------------------

    fn col(ty: Ty, vals: &[Option<Value>]) -> Vector {
        let mut v = Vector::new(ty);
        for x in vals {
            match x {
                Some(x) => v.push_value(x),
                None => v.push_null(),
            }
        }
        v
    }

    fn ints(vals: &[Option<i32>]) -> Vector {
        col(Ty::Int, &vals.iter().map(|v| v.map(Value::I32)).collect::<Vec<_>>())
    }

    fn strs(vals: &[Option<&str>]) -> Vector {
        col(
            Ty::Varchar,
            &vals
                .iter()
                .map(|v| v.map(|s| Value::Bytes(s.as_bytes().to_vec())))
                .collect::<Vec<_>>(),
        )
    }

    /// A program that returns column `idx` unchanged.
    fn load(ty: Ty, idx: u16) -> Program {
        let mut p = Program::new();
        let r = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, ty.phys(), r, 0, 0, idx));
        p.result = r;
        p.result_ty = ty;
        p
    }

    /// A program returning a constant (for lag/lead's offset and default value).
    fn konst(ty: Ty, v: Value) -> Program {
        let mut p = Program::new();
        let r = p.alloc_reg();
        let c = p.add_const(ty, v);
        p.push(Instr::with_aux(OpCode::LoadConst, ty.phys(), r, 0, 0, c));
        p.result = r;
        p.result_ty = ty;
        p
    }

    fn skey(idx: u16, ty: Ty) -> SortKey {
        // The SQL default, ASC / NULLS LAST. Aligned with DuckDB's default.
        SortKey { expr: load(ty, idx), desc: false, nulls_first: false }
    }

    struct SpecBuilder {
        kind: WindowKind,
        args: Vec<Program>,
        partition_by: Vec<Program>,
        order_by: Vec<SortKey>,
        result_ty: Ty,
    }

    fn spec(kind: WindowKind, result_ty: Ty) -> SpecBuilder {
        SpecBuilder {
            kind,
            args: Vec::new(),
            partition_by: Vec::new(),
            order_by: Vec::new(),
            result_ty,
        }
    }

    impl SpecBuilder {
        fn args(mut self, a: Vec<Program>) -> Self {
            self.args = a;
            self
        }
        fn part(mut self, p: Vec<Program>) -> Self {
            self.partition_by = p;
            self
        }
        fn order(mut self, o: Vec<SortKey>) -> Self {
            self.order_by = o;
            self
        }
        fn build(self) -> WindowSpec {
            // The frame is decided the same way as the binder's default: RANGE with ORDER BY,
            // the whole partition without.
            let frame = if self.order_by.is_empty() {
                WindowFrame::WholePartition
            } else {
                WindowFrame::RangeUnboundedPreceding
            };
            WindowSpec {
                kind: self.kind,
                args: self.args,
                partition_by: self.partition_by,
                order_by: self.order_by,
                frame,
                result_ty: self.result_ty,
                name: String::from("w"),
            }
        }
        /// The combination of having ORDER BY but a whole-partition frame.
        fn build_whole(self) -> WindowSpec {
            let mut s = self.build();
            s.frame = WindowFrame::WholePartition;
            s
        }
    }

    // --- Mock inputs --------------------------------------------------------

    enum Script {
        Rows(Vec<Vector>),
        NeedIo,
        NeedCodec,
    }

    struct Mock {
        steps: Vec<Script>,
        pos: usize,
    }

    impl Mock {
        #[allow(clippy::new_ret_no_self)]
        fn new(steps: Vec<Script>) -> Box<dyn Operator> {
            Box::new(Mock { steps, pos: 0 })
        }
    }

    impl Operator for Mock {
        fn next(&mut self, _ctx: &mut ExecContext) -> Result<Step> {
            if self.pos >= self.steps.len() {
                return Ok(Step::Done);
            }
            let i = self.pos;
            self.pos += 1;
            // An interruption is consumed as though "the host's response was awaited".
            Ok(match &self.steps[i] {
                Script::NeedIo => Step::NeedIo,
                Script::NeedCodec => Step::NeedCodec,
                Script::Rows(cols) => Step::Ready(Batch::new(cols.clone())),
            })
        }
    }

    // --- Execution helpers --------------------------------------------------

    fn drive(steps: Vec<Script>, windows: Vec<WindowSpec>) -> Result<Vec<Vec<Value>>> {
        let mut op = Window::new(Mock::new(steps), windows)?;
        let mut cat = Catalog::new();
        let mut vm = Vm::new();
        let mut rows = Vec::new();
        for guard in 0..100_000 {
            assert!(guard < 99_999, "does not terminate");
            let mut ctx =
                ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
            match op.next(&mut ctx)? {
                Step::Ready(b) => {
                    assert!(b.card() <= BATCH_SIZE);
                    for i in 0..b.card() {
                        let r = match &b.sel {
                            Some(s) => s[i] as usize,
                            None => i,
                        };
                        rows.push(b.cols.iter().map(|c| c.value_at(r)).collect::<Vec<_>>());
                    }
                }
                Step::NeedIo | Step::NeedCodec => {}
                Step::Done => break,
            }
        }
        Ok(rows)
    }

    fn run(steps: Vec<Script>, windows: Vec<WindowSpec>) -> Vec<Vec<Value>> {
        drive(steps, windows).unwrap()
    }

    /// Extracts column `c` of the output as i64.
    fn ints_at(rows: &[Vec<Value>], c: usize) -> Vec<Option<i64>> {
        rows.iter().map(|r| r[c].as_i64()).collect()
    }

    fn dbls_at(rows: &[Vec<Value>], c: usize) -> Vec<Option<f64>> {
        rows.iter().map(|r| r[c].as_f64()).collect()
    }

    /// A one-batch, two-column (g, x) input.
    fn gx(g: &[Option<i32>], x: &[Option<i32>]) -> Vec<Script> {
        vec![Script::Rows(vec![ints(g), ints(x)])]
    }

    // --- Interruption and resumption (the most important) -------------------

    #[test]
    fn need_io_and_need_codec_mid_input_match_uninterrupted_run() {
        let chunk = |g: &[Option<i32>], x: &[Option<i32>]| Script::Rows(vec![ints(g), ints(x)]);
        let mk = |interrupted: bool| {
            let a = chunk(&[Some(1), Some(1)], &[Some(10), Some(20)]);
            let b = chunk(&[Some(1), Some(2)], &[Some(30), Some(1)]);
            let c = chunk(&[Some(2), Some(1)], &[Some(2), Some(20)]);
            if interrupted {
                // Both kinds of interruption are interposed mid-input (neither at the start nor at the end).
                vec![a, Script::NeedIo, b, Script::NeedCodec, c, Script::NeedIo]
            } else {
                vec![a, b, c]
            }
        };
        let ws = || {
            vec![
                spec(WindowKind::RowNumber, Ty::BigInt)
                    .part(vec![load(Ty::Int, 0)])
                    .order(vec![skey(1, Ty::Int)])
                    .build(),
                spec(WindowKind::Agg(AggKind::Sum), Ty::HugeInt)
                    .args(vec![load(Ty::Int, 1)])
                    .part(vec![load(Ty::Int, 0)])
                    .order(vec![skey(1, Ty::Int)])
                    .build(),
            ]
        };
        let plain = run(mk(false), ws());
        let broken = run(mk(true), ws());
        assert_eq!(plain.len(), 6);
        for c in 0..4 {
            assert_eq!(ints_at(&broken, c), ints_at(&plain, c), "column {c}");
        }
    }

    #[test]
    fn need_io_before_any_input_is_passed_through() {
        let steps = vec![
            Script::NeedIo,
            Script::NeedCodec,
            Script::Rows(vec![ints(&[Some(1), Some(1)]), ints(&[Some(5), Some(6)])]),
        ];
        let mut op = Window::new(
            Mock::new(steps),
            vec![spec(WindowKind::RowNumber, Ty::BigInt).order(vec![skey(1, Ty::Int)]).build()],
        )
        .unwrap();
        let mut cat = Catalog::new();
        let mut vm = Vm::new();
        let mut ctx =
            ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
        assert!(matches!(op.next(&mut ctx).unwrap(), Step::NeedIo));
        assert!(matches!(op.next(&mut ctx).unwrap(), Step::NeedCodec));
        let b = match op.next(&mut ctx).unwrap() {
            Step::Ready(b) => b,
            _ => panic!("rows should arrive"),
        };
        assert_eq!(b.card(), 2);
        assert_eq!(b.cols[2].value_at(1), Value::I64(2));
    }

    // --- The ranking family (cross-checked with DuckDB) ---------------------

    /// row_number / rank / dense_rank against y = 1,1,2,2,3.
    #[test]
    fn row_number_rank_dense_rank_with_ties() {
        let steps = gx(&[Some(1); 5], &[Some(1), Some(1), Some(2), Some(2), Some(3)]);
        let ws = vec![
            spec(WindowKind::RowNumber, Ty::BigInt).order(vec![skey(1, Ty::Int)]).build(),
            spec(WindowKind::Rank, Ty::BigInt).order(vec![skey(1, Ty::Int)]).build(),
            spec(WindowKind::DenseRank, Ty::BigInt).order(vec![skey(1, Ty::Int)]).build(),
        ];
        let rows = run(steps, ws);
        assert_eq!(ints_at(&rows, 2), vec![Some(1), Some(2), Some(3), Some(4), Some(5)]);
        assert_eq!(ints_at(&rows, 3), vec![Some(1), Some(1), Some(3), Some(3), Some(5)]);
        assert_eq!(ints_at(&rows, 4), vec![Some(1), Some(1), Some(2), Some(2), Some(3)]);
    }

    /// Combines a multi-column PARTITION BY with a DESC NULLS FIRST ORDER BY.
    /// The row_number/rank tests so far only exercised a single-column PARTITION BY with the
    /// default (ASC NULLS LAST) ORDER BY; this covers the paths `cmp_keys`'s multi-key loop,
    /// DESC inversion, and NULL-placement inversion.
    /// Cross-checked with DuckDB:
    ///   PARTITION BY (1,1): x=10,20 -> DESC gives 20,10 -> row_number 2,1
    ///   PARTITION BY (1,2): x=30,40 -> DESC gives 40,30 -> row_number 2,1
    ///   PARTITION BY (2,1): x=NULL,5 -> DESC NULLS FIRST gives NULL,5 -> row_number 1,2
    #[test]
    fn multi_column_partition_by_with_desc_nulls_first_order_by() {
        let steps = vec![Script::Rows(vec![
            ints(&[Some(1), Some(1), Some(1), Some(1), Some(2), Some(2)]), // p1
            ints(&[Some(1), Some(1), Some(2), Some(2), Some(1), Some(1)]), // p2
            ints(&[Some(10), Some(20), Some(30), Some(40), None, Some(5)]), // x
        ])];
        let ws = vec![spec(WindowKind::RowNumber, Ty::BigInt)
            .part(vec![load(Ty::Int, 0), load(Ty::Int, 1)])
            .order(vec![SortKey { expr: load(Ty::Int, 2), desc: true, nulls_first: true }])
            .build()];
        let rows = run(steps, ws);
        assert_eq!(ints_at(&rows, 3), vec![Some(2), Some(1), Some(2), Some(1), Some(1), Some(2)]);
    }

    // --- RANGE peer groups --------------------------------------------------

    /// Tied rows share a frame (RANGE, not ROWS).
    /// DuckDB:
    ///   the 2 rows with y=1 -> sum 30 / count 2 / avg 15
    ///   the 2 rows with y=2 -> sum 100 / count 4 / avg 25
    ///   y=3               -> sum 150 / count 5 / avg 30
    #[test]
    fn range_frame_shares_the_running_value_across_peers() {
        let steps = vec![Script::Rows(vec![
            ints(&[Some(1), Some(1), Some(2), Some(2), Some(3)]),
            ints(&[Some(10), Some(20), Some(30), Some(40), Some(50)]),
        ])];
        let w = |k: WindowKind, ty: Ty| {
            spec(k, ty).args(vec![load(Ty::Int, 1)]).order(vec![skey(0, Ty::Int)]).build()
        };
        let ws = vec![
            w(WindowKind::Agg(AggKind::Sum), Ty::HugeInt),
            spec(WindowKind::Agg(AggKind::CountStar), Ty::BigInt)
                .order(vec![skey(0, Ty::Int)])
                .build(),
            w(WindowKind::Agg(AggKind::Avg), Ty::Double),
            w(WindowKind::Agg(AggKind::Min), Ty::Int),
            w(WindowKind::Agg(AggKind::Max), Ty::Int),
            w(WindowKind::FirstValue, Ty::Int),
            w(WindowKind::LastValue, Ty::Int),
        ];
        let rows = run(steps, ws);
        assert_eq!(ints_at(&rows, 2), vec![Some(30), Some(30), Some(100), Some(100), Some(150)]);
        assert_eq!(ints_at(&rows, 3), vec![Some(2), Some(2), Some(4), Some(4), Some(5)]);
        assert_eq!(
            dbls_at(&rows, 4),
            vec![Some(15.0), Some(15.0), Some(25.0), Some(25.0), Some(30.0)]
        );
        assert_eq!(ints_at(&rows, 5), vec![Some(10); 5], "MIN comes from the frame's start");
        assert_eq!(ints_at(&rows, 6), vec![Some(20), Some(20), Some(40), Some(40), Some(50)]);
        assert_eq!(
            ints_at(&rows, 7),
            vec![Some(10); 5],
            "FIRST_VALUE is the partition's first row"
        );
        assert_eq!(
            ints_at(&rows, 8),
            vec![Some(20), Some(20), Some(40), Some(40), Some(50)],
            "LAST_VALUE is the end of the peer group"
        );
    }

    /// Actually combines FIRST_VALUE/LAST_VALUE with several partitions.
    /// The existing RANGE tests only covered a single partition without using `.part(...)`.
    #[test]
    fn first_value_last_value_are_scoped_per_partition() {
        let steps = vec![Script::Rows(vec![
            ints(&[Some(1), Some(1), Some(1), Some(2), Some(2)]), // g
            ints(&[Some(10), Some(20), Some(30), Some(100), Some(200)]), // x
        ])];
        let w = |k: WindowKind| {
            spec(k, Ty::Int)
                .args(vec![load(Ty::Int, 1)])
                .part(vec![load(Ty::Int, 0)])
                .order(vec![skey(1, Ty::Int)])
                .build_whole()
        };
        let ws = vec![w(WindowKind::FirstValue), w(WindowKind::LastValue)];
        let rows = run(steps, ws);
        assert_eq!(
            ints_at(&rows, 2),
            vec![Some(10), Some(10), Some(10), Some(100), Some(100)],
            "FIRST_VALUE must be separate per partition"
        );
        assert_eq!(
            ints_at(&rows, 3),
            vec![Some(30), Some(30), Some(30), Some(200), Some(200)],
            "LAST_VALUE must not leak across partitions either"
        );
    }

    /// Without ORDER BY (the frame is the whole partition). Every row gets the same value.
    #[test]
    fn whole_partition_frame_without_order_by() {
        let steps = gx(&[Some(1); 3], &[Some(1), Some(2), Some(3)]);
        let ws = vec![
            spec(WindowKind::Agg(AggKind::Sum), Ty::HugeInt).args(vec![load(Ty::Int, 1)]).build(),
            spec(WindowKind::Agg(AggKind::CountStar), Ty::BigInt).build(),
            spec(WindowKind::FirstValue, Ty::Int).args(vec![load(Ty::Int, 1)]).build(),
            spec(WindowKind::LastValue, Ty::Int).args(vec![load(Ty::Int, 1)]).build(),
        ];
        let rows = run(steps, ws);
        assert_eq!(ints_at(&rows, 2), vec![Some(6); 3]);
        assert_eq!(ints_at(&rows, 3), vec![Some(3); 3]);
        assert_eq!(ints_at(&rows, 4), vec![Some(1); 3]);
        assert_eq!(
            ints_at(&rows, 5),
            vec![Some(3); 3],
            "with a whole-partition frame, the last row"
        );
    }

    /// Even with ORDER BY, a whole-partition frame does not produce a running total.
    #[test]
    fn explicit_whole_partition_frame_with_order_by() {
        let steps = gx(&[Some(1); 3], &[Some(3), Some(1), Some(2)]);
        let ws = vec![spec(WindowKind::Agg(AggKind::Sum), Ty::HugeInt)
            .args(vec![load(Ty::Int, 1)])
            .order(vec![skey(1, Ty::Int)])
            .build_whole()];
        assert_eq!(ints_at(&run(steps, ws), 2), vec![Some(6); 3]);
    }

    // --- lag / lead ---------------------------------------------------------

    #[test]
    fn lag_and_lead_at_partition_edges() {
        let steps = gx(&[Some(1); 3], &[Some(1), Some(2), Some(3)]);
        let ws = vec![
            // lag(x)
            spec(WindowKind::Lag, Ty::Int)
                .args(vec![load(Ty::Int, 1)])
                .order(vec![skey(1, Ty::Int)])
                .build(),
            // lead(x, 2, -1)
            spec(WindowKind::Lead, Ty::Int)
                .args(vec![
                    load(Ty::Int, 1),
                    konst(Ty::BigInt, Value::I64(2)),
                    konst(Ty::Int, Value::I32(-1)),
                ])
                .order(vec![skey(1, Ty::Int)])
                .build(),
            // lag(x, 1, -9)
            spec(WindowKind::Lag, Ty::Int)
                .args(vec![
                    load(Ty::Int, 1),
                    konst(Ty::BigInt, Value::I64(1)),
                    konst(Ty::Int, Value::I32(-9)),
                ])
                .order(vec![skey(1, Ty::Int)])
                .build(),
            // lag(x, -1) is the same as lead(x, 1).
            spec(WindowKind::Lag, Ty::Int)
                .args(vec![load(Ty::Int, 1), konst(Ty::BigInt, Value::I64(-1))])
                .order(vec![skey(1, Ty::Int)])
                .build(),
            // lag(x, 0) is the row itself.
            spec(WindowKind::Lag, Ty::Int)
                .args(vec![load(Ty::Int, 1), konst(Ty::BigInt, Value::I64(0))])
                .order(vec![skey(1, Ty::Int)])
                .build(),
        ];
        let rows = run(steps, ws);
        assert_eq!(ints_at(&rows, 2), vec![None, Some(1), Some(2)]);
        assert_eq!(ints_at(&rows, 3), vec![Some(3), Some(-1), Some(-1)]);
        assert_eq!(ints_at(&rows, 4), vec![Some(-9), Some(1), Some(2)]);
        assert_eq!(ints_at(&rows, 5), vec![Some(2), Some(3), None]);
        assert_eq!(ints_at(&rows, 6), vec![Some(1), Some(2), Some(3)]);
    }

    /// Extreme signed offsets are out of range, not arithmetic traps or wrapped row indexes.
    #[test]
    fn lag_and_lead_extreme_offsets_yield_null() {
        let steps = gx(&[Some(1); 3], &[Some(10), Some(20), Some(30)]);
        let ws = vec![
            spec(WindowKind::Lag, Ty::Int)
                .args(vec![load(Ty::Int, 1), konst(Ty::BigInt, Value::I64(i64::MIN))])
                .order(vec![skey(1, Ty::Int)])
                .build(),
            spec(WindowKind::Lead, Ty::Int)
                .args(vec![load(Ty::Int, 1), konst(Ty::BigInt, Value::I64(i64::MAX))])
                .order(vec![skey(1, Ty::Int)])
                .build(),
        ];
        let rows = run(steps, ws);
        assert_eq!(ints_at(&rows, 2), vec![None, None, None]);
        assert_eq!(ints_at(&rows, 3), vec![None, None, None]);
    }

    /// A NULL offset gives a NULL result (the same as DuckDB).
    #[test]
    fn lag_with_null_offset_yields_null() {
        let steps = gx(&[Some(1); 2], &[Some(1), Some(2)]);
        let ws = vec![spec(WindowKind::Lag, Ty::Int)
            .args(vec![load(Ty::Int, 1), konst(Ty::BigInt, Value::Null)])
            .order(vec![skey(1, Ty::Int)])
            .build()];
        assert_eq!(ints_at(&run(steps, ws), 2), vec![None, None]);
    }

    /// lag does not cross partitions.
    #[test]
    fn lag_does_not_cross_partitions() {
        let steps =
            gx(&[Some(1), Some(1), Some(2), Some(2)], &[Some(1), Some(2), Some(3), Some(4)]);
        let ws = vec![spec(WindowKind::Lag, Ty::Int)
            .args(vec![load(Ty::Int, 1)])
            .part(vec![load(Ty::Int, 0)])
            .order(vec![skey(1, Ty::Int)])
            .build()];
        assert_eq!(ints_at(&run(steps, ws), 2), vec![None, Some(1), None, Some(3)]);
    }

    // --- Partitioning -------------------------------------------------------

    /// Several partitions. The same output as DuckDB's (g=2 includes the 2 rows where y is NULL).
    #[test]
    fn multiple_partitions_with_null_order_key() {
        // (g, y, x)
        let steps = vec![Script::Rows(vec![
            ints(&[Some(1), Some(1), Some(2), Some(2), Some(2)]),
            ints(&[Some(1), Some(2), Some(1), None, None]),
            ints(&[Some(10), Some(20), Some(1), Some(2), Some(3)]),
        ])];
        let ws = vec![
            spec(WindowKind::Rank, Ty::BigInt)
                .part(vec![load(Ty::Int, 0)])
                .order(vec![skey(1, Ty::Int)])
                .build(),
            spec(WindowKind::Agg(AggKind::Sum), Ty::HugeInt)
                .args(vec![load(Ty::Int, 2)])
                .part(vec![load(Ty::Int, 0)])
                .order(vec![skey(1, Ty::Int)])
                .build(),
        ];
        let rows = run(steps, ws);
        // With ASC/NULLS LAST, NULLs go last, and NULLs are peers with one another.
        assert_eq!(ints_at(&rows, 3), vec![Some(1), Some(2), Some(1), Some(2), Some(2)]);
        assert_eq!(ints_at(&rows, 4), vec![Some(10), Some(30), Some(1), Some(6), Some(6)]);
    }

    /// Without PARTITION BY = one partition.
    #[test]
    fn single_partition_without_partition_by() {
        let steps = gx(&[Some(9); 4], &[Some(4), Some(3), Some(2), Some(1)]);
        let ws =
            vec![spec(WindowKind::RowNumber, Ty::BigInt).order(vec![skey(1, Ty::Int)]).build()];
        // Emitted in input row order, so values 4,3,2,1 get ranks 4,3,2,1.
        assert_eq!(ints_at(&run(steps, ws), 2), vec![Some(4), Some(3), Some(2), Some(1)]);
    }

    /// A NULL partition key forms its own single partition (as with GROUP BY).
    #[test]
    fn null_partition_keys_form_their_own_partition() {
        let steps = gx(&[None, Some(1), None, Some(1)], &[Some(1), Some(2), Some(3), Some(4)]);
        let ws = vec![
            spec(WindowKind::Agg(AggKind::CountStar), Ty::BigInt)
                .part(vec![load(Ty::Int, 0)])
                .build(),
            spec(WindowKind::Agg(AggKind::Sum), Ty::HugeInt)
                .args(vec![load(Ty::Int, 1)])
                .part(vec![load(Ty::Int, 0)])
                .build(),
        ];
        let rows = run(steps, ws);
        assert_eq!(ints_at(&rows, 2), vec![Some(2); 4]);
        assert_eq!(ints_at(&rows, 3), vec![Some(4), Some(6), Some(4), Some(6)]);
    }

    // --- Row order ----------------------------------------------------------

    /// The output is always in the input's row order. Reordering within a partition must come back.
    #[test]
    fn output_keeps_input_row_order() {
        let g = [Some(1), Some(2), Some(1), Some(2), Some(1)];
        let x = [Some(50), Some(5), Some(10), Some(1), Some(30)];
        let steps = gx(&g, &x);
        let ws = vec![spec(WindowKind::RowNumber, Ty::BigInt)
            .part(vec![load(Ty::Int, 0)])
            .order(vec![skey(1, Ty::Int)])
            .build()];
        let rows = run(steps, ws);
        // The input columns must be laid out unchanged.
        assert_eq!(ints_at(&rows, 0), g.iter().map(|v| v.map(|x| x as i64)).collect::<Vec<_>>());
        assert_eq!(ints_at(&rows, 1), x.iter().map(|v| v.map(|x| x as i64)).collect::<Vec<_>>());
        // g=1 has 10 < 30 < 50; g=2 has 1 < 5.
        assert_eq!(ints_at(&rows, 2), vec![Some(3), Some(2), Some(1), Some(1), Some(2)]);
    }

    /// Tied rows keep the input order (because the comparator breaks ties by row number).
    #[test]
    fn ties_keep_input_order_in_row_number() {
        let steps = gx(&[Some(1); 4], &[Some(7), Some(7), Some(7), Some(7)]);
        let ws =
            vec![spec(WindowKind::RowNumber, Ty::BigInt).order(vec![skey(1, Ty::Int)]).build()];
        assert_eq!(ints_at(&run(steps, ws), 2), vec![Some(1), Some(2), Some(3), Some(4)]);
    }

    // --- Size ---------------------------------------------------------------

    #[test]
    fn more_rows_than_batch_size() {
        const N: usize = BATCH_SIZE * 2 + 37;
        let mut steps = Vec::new();
        let mut i = 0usize;
        while i < N {
            let end = (i + 500).min(N);
            // g partitions into 0/1, and x is a running number (reversed so it looks descending).
            let g: Vec<Option<i32>> = (i..end).map(|k| Some((k % 2) as i32)).collect();
            let x: Vec<Option<i32>> = (i..end).map(|k| Some((N - k) as i32)).collect();
            steps.push(Script::Rows(vec![ints(&g), ints(&x)]));
            i = end;
        }
        let ws = vec![
            spec(WindowKind::RowNumber, Ty::BigInt)
                .part(vec![load(Ty::Int, 0)])
                .order(vec![skey(1, Ty::Int)])
                .build(),
            spec(WindowKind::Agg(AggKind::CountStar), Ty::BigInt)
                .part(vec![load(Ty::Int, 0)])
                .build(),
        ];
        let rows = run(steps, ws);
        assert_eq!(rows.len(), N);
        // x decreases monotonically overall, so per-partition ranks get smaller toward the end.
        let n0 = N.div_ceil(2) as i64;
        let n1 = (N / 2) as i64;
        assert_eq!(rows[0][2].as_i64(), Some(n0));
        assert_eq!(rows[1][2].as_i64(), Some(n1));
        assert_eq!(rows[N - 1][2].as_i64(), Some(1));
        assert_eq!(rows[0][3].as_i64(), Some(n0));
        assert_eq!(rows[1][3].as_i64(), Some(n1));
    }

    #[test]
    fn output_is_chunked_to_batch_size() {
        const N: usize = BATCH_SIZE + 10;
        let x: Vec<Option<i32>> = (0..N as i32).map(Some).collect();
        let steps = vec![Script::Rows(vec![ints(&x)])];
        let mut op = Window::new(
            Mock::new(steps),
            vec![spec(WindowKind::RowNumber, Ty::BigInt).order(vec![skey(0, Ty::Int)]).build()],
        )
        .unwrap();
        let mut cat = Catalog::new();
        let mut vm = Vm::new();
        let mut sizes = Vec::new();
        loop {
            let mut ctx =
                ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
            match op.next(&mut ctx).unwrap() {
                Step::Ready(b) => sizes.push(b.card()),
                Step::NeedIo | Step::NeedCodec => {}
                Step::Done => break,
            }
        }
        assert_eq!(sizes, vec![BATCH_SIZE, 10]);
    }

    // --- Edge conditions ----------------------------------------------------

    #[test]
    fn empty_input_emits_nothing() {
        let ws =
            || vec![spec(WindowKind::RowNumber, Ty::BigInt).order(vec![skey(0, Ty::Int)]).build()];
        assert!(run(Vec::new(), ws()).is_empty());
        // The same when only 0-row batches arrive.
        assert!(run(vec![Script::Rows(vec![ints(&[])]), Script::NeedIo], ws()).is_empty());
    }

    /// NULL values drop out of SUM / COUNT(x) but count toward COUNT(*).
    #[test]
    fn nulls_in_the_value_are_ignored_by_aggregates() {
        let steps = gx(&[Some(1); 3], &[None, Some(2), None]);
        let ws = vec![
            spec(WindowKind::Agg(AggKind::Sum), Ty::HugeInt).args(vec![load(Ty::Int, 1)]).build(),
            spec(WindowKind::Agg(AggKind::Count), Ty::BigInt).args(vec![load(Ty::Int, 1)]).build(),
            spec(WindowKind::Agg(AggKind::CountStar), Ty::BigInt).build(),
        ];
        let rows = run(steps, ws);
        assert_eq!(ints_at(&rows, 2), vec![Some(2); 3]);
        assert_eq!(ints_at(&rows, 3), vec![Some(1); 3]);
        assert_eq!(ints_at(&rows, 4), vec![Some(3); 3]);
    }

    /// An all-NULL frame gives NULL for SUM and 0 for COUNT.
    #[test]
    fn all_null_frame_sums_to_null() {
        let steps = gx(&[Some(1); 2], &[None, None]);
        let ws = vec![
            spec(WindowKind::Agg(AggKind::Sum), Ty::HugeInt).args(vec![load(Ty::Int, 1)]).build(),
            spec(WindowKind::Agg(AggKind::Count), Ty::BigInt).args(vec![load(Ty::Int, 1)]).build(),
            spec(WindowKind::Agg(AggKind::Avg), Ty::Double).args(vec![load(Ty::Int, 1)]).build(),
        ];
        let rows = run(steps, ws);
        assert!(rows[0][2].is_null());
        assert_eq!(rows[0][3].as_i64(), Some(0));
        assert!(rows[0][4].is_null());
    }

    /// first/last/min/max work on string columns too (pushing through `Value`).
    #[test]
    fn string_values() {
        let steps = vec![Script::Rows(vec![
            ints(&[Some(1), Some(1), Some(1)]),
            strs(&[Some("b"), None, Some("a")]),
        ])];
        let ws = vec![
            spec(WindowKind::FirstValue, Ty::Varchar).args(vec![load(Ty::Varchar, 1)]).build(),
            spec(WindowKind::LastValue, Ty::Varchar).args(vec![load(Ty::Varchar, 1)]).build(),
            spec(WindowKind::Agg(AggKind::Min), Ty::Varchar)
                .args(vec![load(Ty::Varchar, 1)])
                .build(),
        ];
        let rows = run(steps, ws);
        assert_eq!(rows[0][2], Value::Bytes(b"b".to_vec()));
        assert_eq!(rows[0][3], Value::Bytes(b"a".to_vec()));
        assert_eq!(rows[0][4], Value::Bytes(b"a".to_vec()));
    }

    /// A zero-column input (the path selecting only `count(*) OVER ()`).
    #[test]
    fn zero_column_input() {
        struct RowsOnly(usize);
        impl Operator for RowsOnly {
            fn next(&mut self, _ctx: &mut ExecContext) -> Result<Step> {
                if self.0 == 0 {
                    return Ok(Step::Done);
                }
                let n = self.0;
                self.0 = 0;
                Ok(Step::Ready(Batch::rows_only(n)))
            }
        }
        let mut op = Window::new(
            Box::new(RowsOnly(3)),
            vec![spec(WindowKind::Agg(AggKind::CountStar), Ty::BigInt).build()],
        )
        .unwrap();
        let mut cat = Catalog::new();
        let mut vm = Vm::new();
        let mut ctx =
            ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
        let b = match op.next(&mut ctx).unwrap() {
            Step::Ready(b) => b,
            _ => panic!("rows should arrive"),
        };
        assert_eq!(b.card(), 3);
        assert_eq!(b.cols[0].value_at(0), Value::I64(3));
    }

    /// The f64 ordering key. NaN is the maximum, and -0.0 and 0.0 are tied (peers).
    #[test]
    fn f64_order_key_is_a_total_order() {
        let steps = vec![Script::Rows(vec![col(
            Ty::Double,
            &[
                Some(Value::F64(f64::NAN)),
                Some(Value::F64(0.0)),
                Some(Value::F64(-0.0)),
                Some(Value::F64(-1.0)),
            ],
        )])];
        let ws = vec![
            spec(WindowKind::Rank, Ty::BigInt).order(vec![skey(0, Ty::Double)]).build(),
            spec(WindowKind::RowNumber, Ty::BigInt).order(vec![skey(0, Ty::Double)]).build(),
        ];
        let rows = run(steps, ws);
        // The order is -1.0 < -0.0 = 0.0 < NaN.
        assert_eq!(ints_at(&rows, 1), vec![Some(4), Some(2), Some(2), Some(1)]);
        assert_eq!(ints_at(&rows, 2), vec![Some(4), Some(2), Some(3), Some(1)]);
    }

    /// SUM accumulates in i128 (a sum that overflows i64).
    #[test]
    fn sum_accumulates_in_i128() {
        let big = i64::MAX;
        let steps = vec![Script::Rows(vec![col(
            Ty::BigInt,
            &[Some(Value::I64(big)), Some(Value::I64(big)), Some(Value::I64(big))],
        )])];
        let ws = vec![spec(WindowKind::Agg(AggKind::Sum), Ty::HugeInt)
            .args(vec![load(Ty::BigInt, 0)])
            .build()];
        let rows = run(steps, ws);
        assert_eq!(rows[0][1], Value::I128(big as i128 * 3));
    }

    #[test]
    fn sum_overflowing_i128_errors() {
        let steps = vec![Script::Rows(vec![col(
            Ty::HugeInt,
            &[Some(Value::I128(i128::MAX)), Some(Value::I128(i128::MAX))],
        )])];
        let ws = vec![spec(WindowKind::Agg(AggKind::Sum), Ty::HugeInt)
            .args(vec![load(Ty::HugeInt, 0)])
            .build()];
        assert_eq!(code_of(drive(steps, ws).map(|_| ())), Some(Code::ValueOutOfRange));
    }

    /// An argument required but absent is rejected as a construction-side bug.
    #[test]
    fn missing_argument_is_rejected() {
        let steps = gx(&[Some(1)], &[Some(1)]);
        let ws = vec![spec(WindowKind::Lag, Ty::Int).order(vec![skey(1, Ty::Int)]).build()];
        assert_eq!(code_of(drive(steps, ws).map(|_| ())), Some(Code::WrongArgCount));
    }

    /// The input's selection vector is honored.
    #[test]
    fn selection_vector_on_input_is_respected() {
        struct Sel;
        impl Operator for Sel {
            fn next(&mut self, _ctx: &mut ExecContext) -> Result<Step> {
                let mut b = Batch::new(vec![ints(&[Some(5), Some(1), Some(9), Some(3)])]);
                b.sel = Some(vec![1, 3]);
                Ok(Step::Ready(b))
            }
        }
        // Wrapped in a mock that gives Done from the second call on, so it returns exactly once.
        struct Once(Option<Sel>);
        impl Operator for Once {
            fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
                match self.0.take() {
                    Some(mut s) => s.next(ctx),
                    None => Ok(Step::Done),
                }
            }
        }
        let mut op = Window::new(
            Box::new(Once(Some(Sel))),
            vec![spec(WindowKind::RowNumber, Ty::BigInt).order(vec![skey(0, Ty::Int)]).build()],
        )
        .unwrap();
        let mut cat = Catalog::new();
        let mut vm = Vm::new();
        let mut ctx =
            ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
        let b = match op.next(&mut ctx).unwrap() {
            Step::Ready(b) => b,
            _ => panic!("rows should arrive"),
        };
        assert_eq!(b.card(), 2);
        assert_eq!(b.cols[0].value_at(0), Value::I32(1));
        assert_eq!(b.cols[1].value_at(0), Value::I64(1));
        assert_eq!(b.cols[1].value_at(1), Value::I64(2));
    }
}
