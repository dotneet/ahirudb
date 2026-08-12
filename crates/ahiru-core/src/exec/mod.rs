//! Physical operators (pull-based Volcano, vectorized).
//!
//! Push-based would be faster, but pull-based takes less code and rides naturally on the
//! step-wise execution at RowGroup boundaries (DESIGN.md §6), so pull-based it is. One
//! `next()` returns 2048 rows, so Volcano's call overhead is relatively negligible.

pub mod agg;
pub mod join;
pub mod range;
pub mod recursive;
pub mod rowkey;
pub mod sample;
pub mod setop;
pub mod sort;
pub mod unnest;
pub mod window;

use crate::catalog::Catalog;
use crate::exec::rowkey::{encode_key, HashIndex};
use crate::expr::vm::Vm;
use crate::expr::Program;
use crate::plan::{Node, ScanSpec};
use crate::prelude::*;
use crate::vector::{Batch, Field, Vector, BATCH_SIZE};

/// A byte range requested from the host.
#[derive(Clone, Copy)]
pub struct IoRequest {
    /// The table index in the catalog.
    pub table: usize,
    /// The part index within the table (which file of a multi-file table).
    /// Always 0 for a single-file table.
    pub part: usize,
    pub offset: u64,
    pub len: u64,
}

/// A compressed block the host is asked to decompress.
///
/// Codecs the wasm core does not carry (GZIP / ZSTD) go through here. GZIP is handled by the
/// browser's `DecompressionStream` and ZSTD by a separate wasm module (DESIGN.md §6). From
/// the engine's point of view both are the same "work asked of the host".
#[derive(Clone, Copy)]
pub struct CodecRequest {
    pub table: usize,
    /// The same meaning as `IoRequest::part`.
    pub part: usize,
    pub codec: crate::parquet::Compression,
    pub offset: u64,
    pub len: u32,
    pub out_len: u32,
}

pub enum Step {
    Ready(Batch),
    /// The needed bytes are not fetched yet. The requests are queued on `ExecContext::io`.
    NeedIo,
    /// A codec that is not built in must be decompressed. The requests are on `ExecContext::codec`.
    NeedCodec,
    Done,
}

pub struct ExecContext<'a> {
    pub catalog: &'a mut Catalog,
    pub vm: &'a mut Vm,
    pub io: Vec<IoRequest>,
    pub codec: Vec<CodecRequest>,
}

pub trait Operator {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step>;
}

/// Builds the physical operator tree from the logical plan.
pub fn build(node: Node) -> Result<Box<dyn Operator>> {
    build_ctx(node, None)
}

/// The body of `build`. `working` says whether a recursive CTE's recursive term is being
/// assembled, and only when it is `Some` can `Node::WorkingTable` be resolved
/// (see `exec::recursive`). The top-level `build` always calls with `None`.
///
/// Even with nested recursive CTEs (one recursive CTE's recursive term referencing another,
/// already-completed one), passing `working` straight through is safe: `bind` only ever
/// places a `Node::WorkingTable` inside "the recursive term of the `RecursiveCte` that
/// produced it", so an outer working table straying into an inner tree can never be used by
/// mistake.
fn build_ctx(node: Node, working: Option<&[Batch]>) -> Result<Box<dyn Operator>> {
    Ok(match node {
        Node::Scan(spec) => Box::new(Scan::new(*spec)),
        #[cfg(feature = "ddl")]
        Node::MemScan(spec) => Box::new(MemScan::new(*spec)),
        Node::Filter { input, pred } => {
            Box::new(Filter { input: build_ctx(*input, working)?, pred })
        }
        Node::Project { input, exprs, .. } => {
            Box::new(Project { input: build_ctx(*input, working)?, exprs })
        }
        Node::Limit { input, limit, offset } => Box::new(Limit {
            input: build_ctx(*input, working)?,
            limit,
            offset,
            seen: 0,
            emitted: 0,
        }),
        Node::Aggregate { input, groups, aggs, having, .. } => {
            Box::new(agg::HashAggregate::new(build_ctx(*input, working)?, groups, aggs, having)?)
        }
        Node::Sort { input, keys, limit } => {
            Box::new(sort::Sort::new(build_ctx(*input, working)?, keys, limit)?)
        }
        Node::Join { left, right, kind, left_keys, right_keys, residual, .. } => {
            let lt: Vec<crate::vector::Ty> = left.schema().iter().map(|f| f.ty).collect();
            let rt: Vec<crate::vector::Ty> = right.schema().iter().map(|f| f.ty).collect();
            Box::new(join::HashJoin::new(
                build_ctx(*left, working)?,
                build_ctx(*right, working)?,
                kind,
                left_keys,
                right_keys,
                residual,
                lt,
                rt,
            )?)
        }
        Node::Window { input, windows, .. } => {
            Box::new(window::Window::new(build_ctx(*input, working)?, windows)?)
        }
        Node::SetOp { left, right, op, all, .. } => Box::new(setop::SetOp::new(
            build_ctx(*left, working)?,
            build_ctx(*right, working)?,
            op,
            all,
        )?),
        Node::DistinctOn { input, keys } => {
            Box::new(DistinctOn::new(build_ctx(*input, working)?, keys))
        }
        Node::RecursiveCte { anchor, recursive_term, union_all, .. } => Box::new(
            recursive::RecursiveCte::new(build_ctx(*anchor, working)?, *recursive_term, union_all),
        ),
        Node::WorkingTable { .. } => {
            // `bind` only places a `Node::WorkingTable` inside the recursive term of the
            // `RecursiveCte` that produced it, so a missing `working` is a binder bug (or a bug
            // calling `exec::build` directly on this node).
            let batches = match working {
                Some(w) => recursive::clone_batches(w),
                None => err!(Internal),
            };
            Box::new(recursive::WorkingTableScan::new(batches))
        }
        Node::Unnest { input, expr, elem_ty, .. } => {
            Box::new(unnest::Unnest::new(build_ctx(*input, working)?, expr, elem_ty))
        }
        Node::GenerateSeries { start, stop, step, inclusive, .. } => {
            Box::new(range::GenerateSeries::new(start, stop, step, inclusive))
        }
        Node::Sample { input, spec } => {
            let input = build_ctx(*input, working)?;
            if spec.is_rows {
                Box::new(sample::RowSample::new(input, &spec))
            } else {
                Box::new(sample::Bernoulli::new(input, &spec))
            }
        }
        Node::AssertMaxOneRow { input, keys } => {
            Box::new(AssertMaxOneRow::new(build_ctx(*input, working)?, keys))
        }
    })
}

// --- Scan -------------------------------------------------------------------

pub struct Scan {
    spec: ScanSpec,
    /// The index of the table part currently being read (which file of a multi-file table).
    /// For a single-file table it stays 0 throughout.
    part: usize,
    /// The next split to read within the current part (a RowGroup for Parquet, a byte chunk for
    /// CSV). It resets to 0 when crossing into a new part.
    split: usize,
    /// The decoded split.
    cur: Option<Decoded>,
}

struct Decoded {
    cols: Vec<Vector>,
    rows: usize,
    /// The start of the next row to return.
    pos: usize,
}

impl Scan {
    pub fn new(spec: ScanSpec) -> Self {
        Scan { spec, part: 0, split: 0, cur: None }
    }
}

impl Operator for Scan {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        loop {
            // If a decoded split remains, return from it.
            if let Some(d) = &mut self.cur {
                if d.pos < d.rows {
                    let n = (d.rows - d.pos).min(BATCH_SIZE);
                    // TODO: a contiguous-range copy would do, so a dedicated slice would make
                    // this index array unnecessary.
                    let idx: Vec<u32> = (d.pos as u32..(d.pos + n) as u32).collect();
                    let cols = d.cols.iter().map(|c| c.gather(&idx)).collect();
                    d.pos += n;
                    return Ok(Step::Ready(Batch::new(cols)));
                }
                self.cur = None;
            }

            let table = match ctx.catalog.get(self.spec.table) {
                Some(t) => t,
                None => err!(TableNotFound),
            };

            // Once the current part is read through, advance to the next. A `while` is used so
            // parts with zero splits (an empty file, say) can be skipped as well. That way the
            // operators above (Filter/Project/...) need not be aware of multi-file tables --
            // presenting split numbers as "one flat sequence across the whole table" is this
            // loop's job.
            while self.part < table.parts.len()
                && self.split >= table.parts[self.part].format.num_splits()
            {
                self.part += 1;
                self.split = 0;
            }
            if self.part >= table.parts.len() {
                return Ok(Step::Done);
            }
            let part = &table.parts[self.part];
            let fmt = &part.format;

            // If statistics can rule it out, move on without fetching a single byte.
            // Formats with no statistics always pass under the default implementation.
            if !fmt.may_match(self.split, &self.spec.pruners, &self.spec.columns) {
                self.split += 1;
                continue;
            }

            // Collects the bytes needed for page-level filtering (ColumnIndex/OffsetIndex/Bloom
            // filter). With nothing to target, nothing is queued (the default implementation), so
            // formats without support add not one round trip here.
            let mut idx_ranges = Vec::with_capacity(self.spec.columns.len());
            fmt.index_ranges(self.split, &self.spec.pruners, &self.spec.columns, &mut idx_ranges)?;
            let mut idx_missing = Vec::new();
            for (start, end) in &idx_ranges {
                if let Some((o, l)) = part.source.missing(*start, end - start) {
                    idx_missing.push(IoRequest {
                        table: self.spec.table,
                        part: self.part,
                        offset: o,
                        len: l,
                    });
                }
            }
            if !idx_missing.is_empty() {
                ctx.io.extend_from_slice(&idx_missing);
                return Ok(Step::NeedIo);
            }

            // Page selection is settled with the index bytes obtained. `false` means this whole
            // split can be skipped (a Bloom filter negative, or no page can match by statistics).
            // It needs `&mut self`, so only here is `catalog` reborrowed mutably
            // (`table`/`part`/`fmt` are re-taken afterwards).
            let keep = match ctx.catalog.get_mut(self.spec.table) {
                Some(t) => match t.parts.get_mut(self.part) {
                    Some(p) => p.format.refine_with_index(
                        &p.source,
                        self.split,
                        &self.spec.pruners,
                        &self.spec.columns,
                    )?,
                    None => err!(Internal),
                },
                None => err!(TableNotFound),
            };
            if !keep {
                self.split += 1;
                continue;
            }

            let table = match ctx.catalog.get(self.spec.table) {
                Some(t) => t,
                None => err!(TableNotFound),
            };
            let part = &table.parts[self.part];
            let fmt = &part.format;

            // Collects the byte ranges this split needs and requests them all at once if not yet
            // fetched. I/O is awaited only at split boundaries, so operators never think about
            // asynchrony (DESIGN.md §6). With page selection in effect, the ranges returned here
            // are narrowed to the surviving pages rather than whole column chunks.
            let mut ranges = Vec::with_capacity(self.spec.columns.len());
            fmt.split_ranges(self.split, &self.spec.columns, &mut ranges)?;
            let mut missing = Vec::new();
            for (start, end) in &ranges {
                if let Some((o, l)) = part.source.missing(*start, end - start) {
                    missing.push(IoRequest {
                        table: self.spec.table,
                        part: self.part,
                        offset: o,
                        len: l,
                    });
                }
            }
            if !missing.is_empty() {
                ctx.io.extend_from_slice(&missing);
                return Ok(Step::NeedIo);
            }

            // Codecs that are not built in are decompressed by the host. Page headers are
            // uncompressed, so all the necessary work can be enumerated at this point.
            let mut tasks = Vec::new();
            fmt.codec_tasks(&part.source, self.split, &self.spec.columns, &mut tasks)?;
            let mut pending = Vec::new();
            for t in &tasks {
                if !part.source.has_decoded(t.offset, t.len) {
                    pending.push(CodecRequest {
                        table: self.spec.table,
                        part: self.part,
                        codec: t.codec,
                        offset: t.offset,
                        len: t.len,
                        out_len: t.out_len,
                    });
                }
            }
            if !pending.is_empty() {
                ctx.codec.extend_from_slice(&pending);
                return Ok(Step::NeedCodec);
            }

            let cols = fmt.read_split(&part.source, self.split, &self.spec.columns)?;
            // The format implementation's contract: return as many columns, of the same length, as the projection.
            ensure!(cols.len() == self.spec.columns.len(), Internal);
            let rows = cols.first().map_or(0, |c| c.len());
            ensure!(cols.iter().all(|c| c.len() == rows), Internal);

            // Decompressed pages are used only within this split. Holding on to them would keep
            // more memory than the pre-compression file.
            if let Some(t) = ctx.catalog.get_mut(self.spec.table) {
                if let Some(p) = t.parts.get_mut(self.part) {
                    p.source.clear_decoded();
                }
            }

            self.split += 1;
            if rows == 0 {
                // An empty split (an empty CSV chunk, say) is skipped.
                continue;
            }
            self.cur = Some(Decoded { cols, rows, pos: 0 });
        }
    }
}

// --- MemScan (`ddl`) ---------------------------------------------------------

/// Scanning a `catalog::MemTable`. The data is already in memory, so unlike `Scan` it can
/// never in principle return `NeedIo`/`NeedCodec`
/// (DESIGN.md §16, "why it is split into four").
#[cfg(feature = "ddl")]
pub struct MemScan {
    spec: crate::plan::MemScanSpec,
    pos: usize,
}

#[cfg(feature = "ddl")]
impl MemScan {
    pub fn new(spec: crate::plan::MemScanSpec) -> Self {
        MemScan { spec, pos: 0 }
    }
}

#[cfg(feature = "ddl")]
impl Operator for MemScan {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        let mt = match ctx.catalog.mem_get(self.spec.table) {
            Some(t) => t,
            None => err!(TableNotFound),
        };
        if self.pos >= mt.rows.len() {
            return Ok(Step::Done);
        }
        let end = (self.pos + BATCH_SIZE).min(mt.rows.len());
        let batch = mt.batch(self.pos, end);
        self.pos = end;
        Ok(Step::Ready(batch))
    }
}

// --- Filter -----------------------------------------------------------------

pub struct Filter {
    input: Box<dyn Operator>,
    pred: crate::expr::Program,
}

impl Operator for Filter {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        loop {
            let mut batch = match self.input.next(ctx)? {
                Step::Ready(b) => b,
                other => return Ok(other),
            };
            let mut sel = Vec::new();
            ctx.vm.eval_filter(&self.pred, &batch, &mut sel)?;
            if sel.is_empty() {
                // A batch whose rows are all filtered out is not returned upstream; the next is pulled.
                continue;
            }
            batch.sel = Some(sel);
            return Ok(Step::Ready(batch));
        }
    }
}

// --- DistinctOn ---------------------------------------------------------------

/// `DISTINCT ON (keys)`.
///
/// A streaming filter passing, in the input's order, **only the first row seen** per key.
/// Deciding "which row comes first" is not this operator's job; the caller (the binder)
/// interposes a `Sort` first when the order matters.
/// It is not blocking, so `NeedIo`/`NeedCodec` are simply passed through and it resumes (the
/// same simplicity as `Filter`). State (`seen`) is kept across batches, but one batch's worth
/// of processing always finishes in full rather than bailing out midway.
const MAX_DISTINCT_ON_BYTES: usize = 64 << 20;

pub struct DistinctOn {
    input: Box<dyn Operator>,
    keys: Vec<Program>,
    seen: HashIndex,
}

impl DistinctOn {
    pub fn new(input: Box<dyn Operator>, keys: Vec<Program>) -> Self {
        DistinctOn { input, keys, seen: HashIndex::new() }
    }
}

impl Operator for DistinctOn {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        loop {
            let batch = match self.input.next(ctx)? {
                Step::Ready(b) => b,
                other => return Ok(other),
            };
            let rows = batch.card();
            let mut kcols = Vec::with_capacity(self.keys.len());
            for p in &self.keys {
                kcols.push(ctx.vm.eval(p, &batch)?);
            }
            let refs: Vec<&Vector> = kcols.iter().collect();
            let mut sel = Vec::with_capacity(rows);
            let mut key = Vec::new();
            for row in 0..rows {
                encode_key(&refs, row, &mut key);
                let (_, is_new) = self.seen.get_or_insert(&key);
                if is_new {
                    let phys = match &batch.sel {
                        Some(s) => s[row],
                        None => row as u32,
                    };
                    sel.push(phys);
                }
            }
            ensure!(self.seen.approx_bytes() <= MAX_DISTINCT_ON_BYTES, Oom);
            if sel.is_empty() {
                continue;
            }
            let mut out = batch;
            out.sel = Some(sel);
            return Ok(Step::Ready(out));
        }
    }
}

// --- AssertMaxOneRow ---------------------------------------------------------

/// Enforces `Node::AssertMaxOneRow`: at most one row overall (`keys` empty) or at most one row
/// per key (`keys` non-empty) may pass through, else `Code::MultipleRowsSubquery`.
///
/// Structurally this is `DistinctOn` with the outcome for a repeat key flipped from "silently
/// dropped" to "an error" -- deliberately a separate operator rather than a flag on
/// `DistinctOn`, since real `DISTINCT ON` queries (`plan::bind::select`'s other use of
/// `Node::DistinctOn`) must keep the "first row wins" behavior, not turn a duplicate into an
/// error.
pub struct AssertMaxOneRow {
    input: Box<dyn Operator>,
    keys: Vec<Program>,
    seen: HashIndex,
}

impl AssertMaxOneRow {
    pub fn new(input: Box<dyn Operator>, keys: Vec<Program>) -> Self {
        AssertMaxOneRow { input, keys, seen: HashIndex::new() }
    }
}

impl Operator for AssertMaxOneRow {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        loop {
            let batch = match self.input.next(ctx)? {
                Step::Ready(b) => b,
                other => return Ok(other),
            };
            let rows = batch.card();
            let mut kcols = Vec::with_capacity(self.keys.len());
            for p in &self.keys {
                kcols.push(ctx.vm.eval(p, &batch)?);
            }
            let refs: Vec<&Vector> = kcols.iter().collect();
            let mut key = Vec::new();
            for row in 0..rows {
                encode_key(&refs, row, &mut key);
                let (_, is_new) = self.seen.get_or_insert(&key);
                // With `keys` empty, `encode_key` produces the same (empty) key for every row,
                // so the second row seen anywhere -- not just the second with a matching key --
                // trips this.
                ensure!(is_new, MultipleRowsSubquery);
            }
            // Same cap as `DistinctOn`, whose `seen` index this mirrors byte-for-byte.
            ensure!(self.seen.approx_bytes() <= MAX_DISTINCT_ON_BYTES, Oom);
            if rows == 0 {
                continue;
            }
            return Ok(Step::Ready(batch));
        }
    }
}

// --- Project ----------------------------------------------------------------

pub struct Project {
    input: Box<dyn Operator>,
    exprs: Vec<crate::expr::Program>,
}

impl Operator for Project {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        let batch = match self.input.next(ctx)? {
            Step::Ready(b) => b,
            other => return Ok(other),
        };
        let rows = batch.card();
        let mut cols = Vec::with_capacity(self.exprs.len());
        for p in &self.exprs {
            cols.push(ctx.vm.eval(p, &batch)?);
        }
        if cols.is_empty() {
            return Ok(Step::Ready(Batch::rows_only(rows)));
        }
        Ok(Step::Ready(Batch::new(cols)))
    }
}

// --- Limit ------------------------------------------------------------------

pub struct Limit {
    input: Box<dyn Operator>,
    limit: Option<u64>,
    offset: u64,
    /// How many rows were skipped to consume the OFFSET.
    seen: u64,
    emitted: u64,
}

impl Operator for Limit {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        loop {
            if let Some(l) = self.limit {
                if self.emitted >= l {
                    return Ok(Step::Done);
                }
            }
            let mut batch = match self.input.next(ctx)? {
                Step::Ready(b) => b,
                other => return Ok(other),
            };
            let card = batch.card() as u64;

            // The OFFSET has not been fully consumed yet.
            if self.seen + card <= self.offset {
                self.seen += card;
                continue;
            }
            let skip = self.offset.saturating_sub(self.seen);
            self.seen += card;

            let mut take = card - skip;
            if let Some(l) = self.limit {
                take = take.min(l - self.emitted);
            }
            if take == 0 {
                continue;
            }
            self.emitted += take;

            if skip > 0 || take < card {
                let base: Vec<u32> = match &batch.sel {
                    Some(s) => s[skip as usize..(skip + take) as usize].to_vec(),
                    None => (skip as u32..(skip + take) as u32).collect(),
                };
                batch.sel = Some(base);
            }
            return Ok(Step::Ready(batch));
        }
    }
}

/// The schema of an execution result.
pub fn result_schema(node: &Node) -> Vec<Field> {
    node.schema().to_vec()
}

// --- Values -----------------------------------------------------------------

/// An operator returning a pre-built batch exactly once.
///
/// Used where the result is already settled without a scan, as with
/// `DESCRIBE` / `SHOW TABLES` / `EXPLAIN`. It saves building a dedicated execution path.
pub struct Values {
    batch: Option<Batch>,
}

impl Values {
    pub fn new(batch: Batch) -> Self {
        Values { batch: Some(batch) }
    }
}

impl Operator for Values {
    fn next(&mut self, _ctx: &mut ExecContext) -> Result<Step> {
        match self.batch.take() {
            Some(b) => Ok(Step::Ready(b)),
            None => Ok(Step::Done),
        }
    }
}

#[cfg(test)]
mod distinct_on_tests {
    use super::*;
    use crate::catalog::Catalog;
    use crate::expr::vm::Vm;
    use crate::expr::{Instr, OpCode, Program};
    use crate::vector::{Ty, Value};

    fn ints(vals: &[Option<i32>]) -> Vector {
        let mut v = Vector::new(Ty::Int);
        for x in vals {
            match x {
                Some(x) => v.push_value(&Value::I32(*x)),
                None => v.push_null(),
            }
        }
        v
    }

    /// A program that just returns column `idx`.
    fn load(idx: u16) -> Program {
        let mut p = Program::new();
        let r = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, crate::vector::PhysType::I32, r, 0, 0, idx));
        p.result = r;
        p.result_ty = Ty::Int;
        p
    }

    enum Script {
        Rows(Vec<Vector>),
        NeedIo,
    }

    struct Mock {
        steps: Vec<Script>,
        pos: usize,
    }

    impl Operator for Mock {
        fn next(&mut self, _ctx: &mut ExecContext) -> Result<Step> {
            if self.pos >= self.steps.len() {
                return Ok(Step::Done);
            }
            let i = self.pos;
            self.pos += 1;
            Ok(match &self.steps[i] {
                Script::NeedIo => Step::NeedIo,
                Script::Rows(cols) => Step::Ready(Batch::new(cols.clone())),
            })
        }
    }

    fn drive(steps: Vec<Script>, keys: Vec<Program>) -> Vec<Vec<Value>> {
        let mut op = DistinctOn::new(Box::new(Mock { steps, pos: 0 }), keys);
        let mut cat = Catalog::new();
        let mut vm = Vm::new();
        let mut rows = Vec::new();
        for guard in 0..10_000 {
            assert!(guard < 9_999, "does not terminate");
            let mut ctx =
                ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
            match op.next(&mut ctx).unwrap() {
                Step::Ready(b) => {
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
        rows
    }

    #[test]
    fn keeps_only_the_first_row_per_key_in_arrival_order() {
        // Three rows with key=1 and two with key=2 arrive. Only the first row of each key survives.
        let steps = vec![Script::Rows(vec![
            ints(&[Some(1), Some(2), Some(1), Some(2), Some(1)]),
            ints(&[Some(10), Some(20), Some(11), Some(21), Some(12)]),
        ])];
        let rows = drive(steps, vec![load(0)]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec![Value::I32(1), Value::I32(10)]);
        assert_eq!(rows[1], vec![Value::I32(2), Value::I32(20)]);
    }

    #[test]
    fn null_keys_form_their_own_group() {
        let steps = vec![Script::Rows(vec![
            ints(&[None, Some(1), None]),
            ints(&[Some(9), Some(8), Some(7)]),
        ])];
        let rows = drive(steps, vec![load(0)]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec![Value::Null, Value::I32(9)]);
        assert_eq!(rows[1], vec![Value::I32(1), Value::I32(8)]);
    }

    #[test]
    fn multi_column_keys_are_compared_as_a_whole() {
        let steps = vec![Script::Rows(vec![
            ints(&[Some(1), Some(1), Some(1)]),
            ints(&[Some(1), Some(2), Some(1)]),
            ints(&[Some(100), Some(200), Some(300)]),
        ])];
        let rows = drive(steps, vec![load(0), load(1)]);
        // (1,1) and (1,2) are different keys; the second (1,1) is a duplicate and is dropped.
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][2], Value::I32(100));
        assert_eq!(rows[1][2], Value::I32(200));
    }

    #[test]
    fn need_io_mid_input_does_not_change_the_result() {
        let make = |interrupt: bool| {
            let mut steps = Vec::new();
            steps.push(Script::Rows(vec![ints(&[Some(1), Some(2)]), ints(&[Some(10), Some(20)])]));
            if interrupt {
                steps.push(Script::NeedIo);
            }
            steps.push(Script::Rows(vec![ints(&[Some(1), Some(3)]), ints(&[Some(11), Some(30)])]));
            steps
        };
        let plain = drive(make(false), vec![load(0)]);
        let interrupted = drive(make(true), vec![load(0)]);
        assert_eq!(plain, interrupted, "the result must not change across a NeedIo");
        assert_eq!(plain.len(), 3);
    }

    #[test]
    fn all_duplicates_in_a_batch_yield_no_output_batch() {
        // If a whole batch is nothing but duplicates, that batch is not returned to the caller
        // and the next input is pulled (the same "empty means next" discipline as `Filter`).
        let steps = vec![
            Script::Rows(vec![ints(&[Some(1)]), ints(&[Some(10)])]),
            Script::Rows(vec![ints(&[Some(1), Some(1)]), ints(&[Some(99), Some(98)])]),
            Script::Rows(vec![ints(&[Some(2)]), ints(&[Some(20)])]),
        ];
        let rows = drive(steps, vec![load(0)]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec![Value::I32(1), Value::I32(10)]);
        assert_eq!(rows[1], vec![Value::I32(2), Value::I32(20)]);
    }
}
