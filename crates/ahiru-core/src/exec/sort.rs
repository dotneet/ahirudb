//! Sorting and Top-N.
//!
//! Sorting is a **blocking** operator that cannot return a single row until the input is read
//! through. With a remote (range-fetched) source, `Step::NeedIo` / `NeedCodec` come back midway,
//! so all the partial buffering state lives in `self`, interruptions are passed straight
//! through, and the next call resumes from the same place (DESIGN.md §6). Writing a "read
//! everything" loop without state would either discard input there or read it twice.
//!
//! Rows are buffered columnar. Building a `Value` per row would mean one allocation per cell,
//! making allocation dominate over comparison.
//!
//! ## How the order is decided
//!
//! - Keys are compared in the given order, and `desc` inverts **only the value comparison**.
//! - NULL placement is decided by `nulls_first` alone and is unaffected by `desc`.
//!   The SQL defaults ASC->NULLS LAST / DESC->NULLS FIRST are already lowered into flags by the
//!   binder, so reapplying them here would invert twice.
//! - F64's `partial_cmp` can return `None`. Collapsing that to "equal" in a comparator would
//!   break transitivity, so it is not used at all; values are mapped to a totally ordered `u64`
//!   key instead (`f64_key`). The order is `-inf < ... < -0.0 = 0.0 < ... < +inf < NaN`.
//! - Ties are finally broken by the row number in the buffer, so the comparison is a **total order**.
//!   That also yields stability (equal keys keep the input order).
//!
//! ## Memory
//!
//! There is no spilling. Once the buffer exceeds `MAX_BUFFER_BYTES` it returns `Oom` rather than
//! quietly ballooning. With a `limit` only the top n are held in the first place, so
//! `ORDER BY ... LIMIT 10` over 50M rows never touches the cap.

use crate::exec::rowkey::interval_key;
use crate::exec::{ExecContext, Operator, Step};
use crate::plan::SortKey;
use crate::prelude::*;
use crate::vector::{Batch, Bitmap, Data, Ty, Vector, BATCH_SIZE};

use core::cmp::Ordering;

/// With no spilling, exceeding this returns `Oom`.
/// wasm's linear memory caps at 4 GiB, but the host's buffers and decoded pages share it, so
/// sorting alone is held to 256 MiB.
const MAX_BUFFER_BYTES: usize = 256 * 1024 * 1024;

enum Phase {
    /// Reading and buffering the input. It stays in this state across interruptions.
    Buffering,
    /// The order is settled. `order` is returned in `BATCH_SIZE` slices.
    Emitting,
    Done,
}

pub struct Sort {
    input: Box<dyn Operator>,
    keys: Vec<SortKey>,
    /// How many rows Top-N keeps. `None` means all of them.
    limit: Option<usize>,
    phase: Phase,

    /// The buffered input columns. The schema is the input's as is (sorting does not change columns).
    cols: Vec<Vector>,
    /// The buffered sort keys. The same count and order as `keys`.
    key_cols: Vec<Vector>,
    /// The buffered row count. Even an input with no columns (`COUNT(*)` and the like) needs the row count.
    rows: usize,
    /// Whether the column types were decided from the first batch. A zero-column input exists, so emptiness of `cols` cannot stand in for it.
    init: bool,

    /// The settled output order. Valid only from `Emitting` on.
    order: Vec<u32>,
    /// The next position in `order` to return.
    pos: usize,
}

impl Sort {
    pub fn new(input: Box<dyn Operator>, keys: Vec<SortKey>, limit: Option<usize>) -> Result<Self> {
        // LIMIT 0 returns no rows. There is not even a need to pull the input.
        let phase = if limit == Some(0) { Phase::Done } else { Phase::Buffering };
        Ok(Sort {
            input,
            keys,
            limit,
            phase,
            cols: Vec::new(),
            key_cols: Vec::new(),
            rows: 0,
            init: false,
            order: Vec::new(),
            pos: 0,
        })
    }

    /// Takes one batch into the buffer.
    fn absorb(&mut self, mut batch: Batch, ctx: &mut ExecContext) -> Result<()> {
        // Selection is resolved first, to avoid gathering twice (once for key evaluation and once
        // for appending the columns).
        batch.materialize();
        let rows = batch.card();

        let mut kvs = Vec::with_capacity(self.keys.len());
        for k in &self.keys {
            kvs.push(ctx.vm.eval(&k.expr, &batch)?);
        }

        if !self.init {
            self.cols = batch.cols.iter().map(|c| Vector::new(c.ty())).collect();
            self.key_cols = kvs.iter().map(|v| Vector::new(v.ty())).collect();
            self.init = true;
        }
        ensure!(batch.cols.len() == self.cols.len(), Internal);

        // Row numbers ride in a u32, so beyond that it gives up.
        ensure!(self.rows.saturating_add(rows) <= u32::MAX as usize, LimitExceeded);

        for (dst, src) in self.key_cols.iter_mut().zip(kvs.iter()) {
            append(dst, src)?;
        }
        for (dst, src) in self.cols.iter_mut().zip(batch.cols.iter()) {
            append(dst, src)?;
        }
        self.rows += rows;

        // Compaction happens before the cap check. Top-N never touches the cap.
        self.compact()?;
        ensure!(self.buffered_bytes() <= MAX_BUFFER_BYTES, Oom);
        Ok(())
    }

    /// Only under Top-N, trims the buffer to the top `n` rows.
    ///
    /// Rather than touching a heap per row, it buffers up to `2n` rows and then selects in bulk.
    /// In a columnar buffer, replacing one row is not cheap (because of variable-length columns),
    /// so re-`gather`ing in bulk needs fewer allocations and fewer comparisons.
    /// One compaction discards at least `n` rows, so amortized it is O(log n) per row.
    fn compact(&mut self) -> Result<()> {
        let n = match self.limit {
            Some(n) => n,
            None => return Ok(()),
        };
        let cap = n.saturating_mul(2).max(BATCH_SIZE);
        if self.rows <= cap {
            return Ok(());
        }
        let mut order: Vec<u32> = (0..self.rows as u32).collect();
        order.sort_by(|&a, &b| cmp_row(&self.keys, &self.key_cols, a, b));
        order.truncate(n);
        // The retained rows are written back **in sorted order**. That preserves the invariant
        // "ascending buffer index matches input order among equal keys", so stability follows from
        // the comparator's index tiebreak alone.
        for c in self.cols.iter_mut() {
            *c = c.gather(&order);
        }
        for c in self.key_cols.iter_mut() {
            *c = c.gather(&order);
        }
        self.rows = order.len();
        Ok(())
    }

    /// The approximate bytes the buffer uses.
    fn buffered_bytes(&self) -> usize {
        let mut n = 0usize;
        for v in self.cols.iter().chain(self.key_cols.iter()) {
            n = n.saturating_add(vector_bytes(v));
        }
        n
    }

    /// The input is read through. Settles the order and moves to the output phase.
    fn finish(&mut self) -> Result<()> {
        let mut order: Vec<u32> = (0..self.rows as u32).collect();
        order.sort_by(|&a, &b| cmp_row(&self.keys, &self.key_cols, a, b));
        if let Some(n) = self.limit {
            order.truncate(n);
        }
        self.order = order;
        self.pos = 0;
        // The key columns are no longer needed. There is no reason to hold them during output.
        self.key_cols = Vec::new();
        self.phase = Phase::Emitting;
        Ok(())
    }

    fn emit(&mut self) -> Result<Step> {
        if self.pos >= self.order.len() {
            self.phase = Phase::Done;
            self.cols = Vec::new();
            self.order = Vec::new();
            return Ok(Step::Done);
        }
        let end = (self.pos + BATCH_SIZE).min(self.order.len());
        let idx = &self.order[self.pos..end];
        let out = if self.cols.is_empty() {
            Batch::rows_only(idx.len())
        } else {
            Batch::new(self.cols.iter().map(|c| c.gather(idx)).collect())
        };
        self.pos = end;
        Ok(Step::Ready(out))
    }
}

impl Operator for Sort {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        loop {
            match self.phase {
                Phase::Buffering => match self.input.next(ctx)? {
                    Step::Ready(b) => self.absorb(b, ctx)?,
                    // The interruption is returned straight up. The buffered rows stay in `self`,
                    // so the next call pulls the input again from here (nothing dropped and
                    // nothing read twice). Waiting on bytes and on decompression are handled alike.
                    Step::NeedIo => return Ok(Step::NeedIo),
                    Step::NeedCodec => return Ok(Step::NeedCodec),
                    Step::Done => self.finish()?,
                },
                Phase::Emitting => return self.emit(),
                Phase::Done => return Ok(Step::Done),
            }
        }
    }
}

// --- Comparison ---------------------------------------------------------------

/// A total-order comparison of two rows. Equal keys are broken by index, so `Equal` happens only for the same row.
fn cmp_row(keys: &[SortKey], cols: &[Vector], a: u32, b: u32) -> Ordering {
    let (ai, bi) = (a as usize, b as usize);
    for (k, c) in keys.iter().zip(cols.iter()) {
        let (va, vb) = (c.is_valid(ai), c.is_valid(bi));
        if !va || !vb {
            if !va && !vb {
                continue;
            }
            // NULL placement is decided by nulls_first alone. Applying desc here would apply the
            // binder's default (ASC->LAST / DESC->FIRST) twice.
            let null_is_first = k.nulls_first;
            return if !va {
                if null_is_first {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            } else if null_is_first {
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
    a.cmp(&b)
}

/// Value comparison per physical type. NULL checks are already done by the caller.
fn cmp_data(c: &Vector, a: usize, b: usize) -> Ordering {
    match c.data() {
        // false < true.
        Data::Bool(v) => v.get(a).cmp(&v.get(b)),
        Data::I32(v) => v[a].cmp(&v[b]),
        Data::I64(v) => v[a].cmp(&v[b]),
        // INTERVAL's three packed components are normalized to microseconds first (see
        // `rowkey::interval_key`); the raw bit pattern would rank `1 day` above `25 hours`.
        Data::I128(v) if c.ty() == Ty::Interval => interval_key(v[a]).cmp(&interval_key(v[b])),
        Data::I128(v) => v[a].cmp(&v[b]),
        Data::F64(v) => f64_key(v[a]).cmp(&f64_key(v[b])),
        // Lexicographic. On a common prefix the shorter one is smaller.
        Data::Bytes(v) => v.get(a).cmp(v.get(b)),
    }
}

/// Maps an f64 to an order-preserving `u64`.
///
/// `partial_cmp` returns `None` for NaN and so cannot be used in a comparator. The bit
/// representation is passed through a monotone mapping to give a total order:
/// `-inf < ... < -0.0 = 0.0 < ... < +inf < NaN`.
///
/// - NaN collapses into a single value "greater than every number". NaNs are equal to one
///   another, so several NaNs stay in input order (deterministic).
/// - `-0.0` and `0.0` are equal under `=`, and `rowkey::canonical_f64` identifies them too.
///   Treating them differently for ordering alone would be incoherent, so they are made equal here as well.
#[inline]
fn f64_key(v: f64) -> u64 {
    if v.is_nan() {
        return u64::MAX;
    }
    let b = if v == 0.0 { 0 } else { v.to_bits() };
    // For negatives, a larger bit pattern means a smaller value, so it is inverted. Positives get
    // the sign bit set to lift them above the negatives.
    if b >> 63 != 0 {
        !b
    } else {
        b | (1 << 63)
    }
}

// --- Buffer operations --------------------------------------------------------

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
            // The leading offset is already in `dst`, so it is skipped.
            for &o in s.offsets.iter().skip(1) {
                d.offsets.push(shift + (o - first));
            }
        }
        // The columns come from the same operator, so landing here is an assembly-side bug.
        _ => err!(Internal),
    }
    // If either side has NULLs, validity is aligned. Even when only `dst` has it, without
    // extending it, its length would desynchronize from the data and later `is_valid` calls would read out of range.
    if n > 0 && (src.has_nulls() || dst.has_nulls()) {
        // validity_mut materializes and extends with all-ones up to the appended length.
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
/// `exec::recursive` uses it too, for the byte cap on a recursive CTE's working table
/// (so the memory-estimation logic does not exist in two places).
pub(crate) fn vector_bytes(v: &Vector) -> usize {
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
    use crate::expr::vm::Vm;
    use crate::expr::{Instr, OpCode, Program};
    use crate::vector::{Ty, Value};

    // --- Construction helpers -----------------------------------------------

    /// A program that returns column `col` unchanged.
    fn col_expr(col: u16, ty: Ty) -> Program {
        let mut p = Program::new();
        let r = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, ty.phys(), r, 0, 0, col));
        p.result = r;
        p.result_ty = ty;
        p
    }

    fn key(col: u16, ty: Ty, desc: bool, nulls_first: bool) -> SortKey {
        SortKey { expr: col_expr(col, ty), desc, nulls_first }
    }

    fn vector(ty: Ty, vals: &[Option<Value>]) -> Vector {
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
        vector(Ty::Int, &vals.iter().map(|v| v.map(Value::I32)).collect::<Vec<_>>())
    }

    /// A running 0..n column. A "row ID" for verifying stability and Top-N.
    fn ids(n: usize) -> Vector {
        ints(&(0..n as i32).map(Some).collect::<Vec<_>>())
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
        // A test helper, so returning `Box<dyn Operator>` keeps the call sites shorter.
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
            // A real Scan likewise resumes the same split on the next call.
            Ok(match &self.steps[i] {
                Script::NeedIo => Step::NeedIo,
                Script::NeedCodec => Step::NeedCodec,
                Script::Rows(cols) => Step::Ready(Batch::new(cols.clone())),
            })
        }
    }

    /// Flattens one batch's output into per-row `Value` lists.
    fn rows_of(b: &Batch) -> Vec<Vec<Value>> {
        (0..b.card()).map(|i| b.cols.iter().map(|c| c.value_at(i)).collect()).collect()
    }

    /// Runs the sort to completion and returns the rows per batch.
    fn drive(steps: Vec<Script>, keys: Vec<SortKey>, limit: Option<usize>) -> Vec<Vec<Vec<Value>>> {
        let mut op = Sort::new(Mock::new(steps), keys, limit).unwrap();
        let mut cat = Catalog::default();
        let mut vm = Vm::new();
        let mut ctx =
            ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
        let mut out = Vec::new();
        for _ in 0..100_000 {
            match op.next(&mut ctx).unwrap() {
                Step::Ready(b) => out.push(rows_of(&b)),
                // Calls the same operator again, as though the host had responded.
                Step::NeedIo | Step::NeedCodec => continue,
                Step::Done => return out,
            }
        }
        panic!("sort did not terminate");
    }

    /// Every row with batch boundaries flattened away.
    fn flat(steps: Vec<Script>, keys: Vec<SortKey>, limit: Option<usize>) -> Vec<Vec<Value>> {
        drive(steps, keys, limit).into_iter().flatten().collect()
    }

    /// Extracts the given column of each row as i32.
    fn col_i32(rows: &[Vec<Value>], c: usize) -> Vec<Option<i32>> {
        rows.iter()
            .map(|r| match &r[c] {
                Value::I32(v) => Some(*v),
                _ => None,
            })
            .collect()
    }

    // --- Interruption and resumption (the most important) -------------------

    #[test]
    fn need_io_mid_input_matches_uninterrupted_run() {
        let mk = |interrupted: bool| {
            let a = Script::Rows(vec![ints(&[Some(5), Some(1)]), ints(&[Some(0), Some(1)])]);
            let b = Script::Rows(vec![ints(&[Some(3), Some(9)]), ints(&[Some(2), Some(3)])]);
            let c = Script::Rows(vec![ints(&[Some(2)]), ints(&[Some(4)])]);
            if interrupted {
                // Interruptions are interposed mid-input (between batches, and neither at the start nor at
                // the end). Both waiting on bytes and waiting on decompression are mixed in.
                vec![a, Script::NeedIo, b, Script::NeedCodec, c, Script::NeedIo]
            } else {
                vec![a, b, c]
            }
        };
        let ks = || vec![key(0, Ty::Int, false, false)];
        let plain = flat(mk(false), ks(), None);
        let broken = flat(mk(true), ks(), None);
        assert_eq!(col_i32(&plain, 0), vec![Some(1), Some(2), Some(3), Some(5), Some(9)]);
        // No row may have vanished or been doubled.
        assert_eq!(col_i32(&broken, 0), col_i32(&plain, 0));
        assert_eq!(col_i32(&broken, 1), col_i32(&plain, 1));
    }

    #[test]
    fn need_io_before_any_input_is_passed_through() {
        // It does not break even when the very first call is an interruption.
        let steps =
            vec![Script::NeedIo, Script::NeedCodec, Script::Rows(vec![ints(&[Some(2), Some(1)])])];
        let mut op =
            Sort::new(Mock::new(steps), vec![key(0, Ty::Int, false, false)], None).unwrap();
        let mut cat = Catalog::default();
        let mut vm = Vm::new();
        let mut ctx =
            ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
        assert!(matches!(op.next(&mut ctx).unwrap(), Step::NeedIo));
        assert!(matches!(op.next(&mut ctx).unwrap(), Step::NeedCodec));
        let b = match op.next(&mut ctx).unwrap() {
            Step::Ready(b) => b,
            _ => panic!("expected rows"),
        };
        assert_eq!(col_i32(&rows_of(&b), 0), vec![Some(1), Some(2)]);
    }

    #[test]
    fn need_io_during_top_n_compaction() {
        // Interrupting across a compaction does not change the top n.
        let chunk = |base: i32| {
            let vals: Vec<Option<i32>> = (0..1500).map(|i| Some((base + i * 37) % 5000)).collect();
            Script::Rows(vec![ints(&vals)])
        };
        let mk = |interrupted: bool| {
            let mut v = vec![chunk(0), chunk(1)];
            if interrupted {
                v.push(Script::NeedIo);
            }
            v.push(chunk(2));
            v
        };
        let ks = || vec![key(0, Ty::Int, false, false)];
        let plain = flat(mk(false), ks(), Some(5));
        let broken = flat(mk(true), ks(), Some(5));
        assert_eq!(col_i32(&broken, 0), col_i32(&plain, 0));
        assert_eq!(plain.len(), 5);
    }

    // --- Basic ordering -----------------------------------------------------

    #[test]
    fn single_key_asc_and_desc() {
        let rows = || vec![Script::Rows(vec![ints(&[Some(3), Some(1), Some(2)])])];
        let asc = flat(rows(), vec![key(0, Ty::Int, false, false)], None);
        assert_eq!(col_i32(&asc, 0), vec![Some(1), Some(2), Some(3)]);
        let desc = flat(rows(), vec![key(0, Ty::Int, true, false)], None);
        assert_eq!(col_i32(&desc, 0), vec![Some(3), Some(2), Some(1)]);
    }

    #[test]
    fn second_key_breaks_ties_with_its_own_direction() {
        // First key ASC, second key DESC.
        let cols = vec![
            ints(&[Some(1), Some(1), Some(0), Some(0)]),
            ints(&[Some(10), Some(20), Some(30), Some(40)]),
        ];
        let rows = flat(
            vec![Script::Rows(cols)],
            vec![key(0, Ty::Int, false, false), key(1, Ty::Int, true, false)],
            None,
        );
        assert_eq!(col_i32(&rows, 0), vec![Some(0), Some(0), Some(1), Some(1)]);
        assert_eq!(col_i32(&rows, 1), vec![Some(40), Some(30), Some(20), Some(10)]);
    }

    #[test]
    fn equal_keys_keep_input_order() {
        // Every key is the same. The ID column must come out in input order.
        let n = 200;
        let cols = vec![ints(&vec![Some(7); n]), ids(n)];
        let rows = flat(vec![Script::Rows(cols)], vec![key(0, Ty::Int, false, false)], None);
        assert_eq!(col_i32(&rows, 1), (0..n as i32).map(Some).collect::<Vec<_>>());

        // The same across batches.
        let steps = vec![
            Script::Rows(vec![ints(&[Some(7), Some(7)]), ints(&[Some(0), Some(1)])]),
            Script::Rows(vec![ints(&[Some(7), Some(7)]), ints(&[Some(2), Some(3)])]),
        ];
        let rows = flat(steps, vec![key(0, Ty::Int, true, false)], None);
        assert_eq!(col_i32(&rows, 1), vec![Some(0), Some(1), Some(2), Some(3)]);
    }

    // --- NULL placement -----------------------------------------------------

    #[test]
    fn null_placement_follows_flag_not_direction() {
        // The values are 2, NULL, 1 (with IDs 0, 1, 2).
        let cols = || vec![ints(&[Some(2), None, Some(1)]), ids(3)];
        let run = |desc: bool, nulls_first: bool| {
            let rows =
                flat(vec![Script::Rows(cols())], vec![key(0, Ty::Int, desc, nulls_first)], None);
            col_i32(&rows, 1)
        };
        // ASC: 1 then 2. NULL goes to whichever side the flag says.
        assert_eq!(run(false, false), vec![Some(2), Some(0), Some(1)]);
        assert_eq!(run(false, true), vec![Some(1), Some(2), Some(0)]);
        // DESC: 2 then 1. NULL placement follows the same flag as ASC (desc does not invert it).
        assert_eq!(run(true, false), vec![Some(0), Some(2), Some(1)]);
        assert_eq!(run(true, true), vec![Some(1), Some(0), Some(2)]);
    }

    #[test]
    fn nulls_are_equal_to_each_other_and_fall_through_to_next_key() {
        // When the first key is NULL on both sides, the second key decides.
        let cols = vec![ints(&[None, None]), ints(&[Some(9), Some(4)])];
        let rows = flat(
            vec![Script::Rows(cols)],
            vec![key(0, Ty::Int, false, true), key(1, Ty::Int, false, false)],
            None,
        );
        assert_eq!(col_i32(&rows, 1), vec![Some(4), Some(9)]);
    }

    // --- Comparison per physical type ---------------------------------------

    fn sorted_ids(col: Vector, n: usize, desc: bool) -> Vec<Option<i32>> {
        let ty = col.ty();
        let rows = flat(vec![Script::Rows(vec![col, ids(n)])], vec![key(0, ty, desc, false)], None);
        col_i32(&rows, 1)
    }

    #[test]
    fn sorts_bool() {
        let c = vector(Ty::Boolean, &[Some(Value::Bool(true)), Some(Value::Bool(false))]);
        // false < true.
        assert_eq!(sorted_ids(c, 2, false), vec![Some(1), Some(0)]);
    }

    #[test]
    fn sorts_i32_i64_i128() {
        let c = ints(&[Some(0), Some(i32::MIN), Some(i32::MAX)]);
        assert_eq!(sorted_ids(c, 3, false), vec![Some(1), Some(0), Some(2)]);

        let c = vector(
            Ty::BigInt,
            &[Some(Value::I64(0)), Some(Value::I64(i64::MIN)), Some(Value::I64(i64::MAX))],
        );
        assert_eq!(sorted_ids(c, 3, false), vec![Some(1), Some(0), Some(2)]);

        let c = vector(
            Ty::HugeInt,
            &[Some(Value::I128(0)), Some(Value::I128(i128::MIN)), Some(Value::I128(i128::MAX))],
        );
        assert_eq!(sorted_ids(c, 3, true), vec![Some(2), Some(0), Some(1)]);
    }

    #[test]
    fn sorts_bytes_lexicographically() {
        let b = |s: &str| Some(Value::Bytes(s.as_bytes().to_vec()));
        // "" < "ab" < "abc" < "b" (on a common prefix the shorter is smaller)
        let c = vector(Ty::Varchar, &[b("b"), b("abc"), b(""), b("ab")]);
        assert_eq!(sorted_ids(c, 4, false), vec![Some(2), Some(3), Some(1), Some(0)]);

        // Bytes 0x80 and above are treated as unsigned too.
        let raw = |v: &[u8]| Some(Value::Bytes(v.to_vec()));
        let c = vector(Ty::Blob, &[raw(&[0xff]), raw(&[0x01]), raw(&[0x80])]);
        assert_eq!(sorted_ids(c, 3, false), vec![Some(1), Some(2), Some(0)]);
    }

    #[test]
    fn f64_total_order_is_documented_and_deterministic() {
        let f = |v: f64| Some(Value::F64(v));
        // Input order: NaN, 0.0, -0.0, +inf, -inf, 1.0, NaN (negative sign)
        let c = vector(
            Ty::Double,
            &[
                f(f64::NAN),
                f(0.0),
                f(-0.0),
                f(f64::INFINITY),
                f(f64::NEG_INFINITY),
                f(1.0),
                f(-f64::NAN),
            ],
        );
        // -inf < -0.0 = 0.0 < 1.0 < +inf < NaN.
        // 0.0 and -0.0 are equal, so they stay in input order (IDs 1, 2). The same for the NaNs.
        assert_eq!(
            sorted_ids(c.clone(), 7, false),
            vec![Some(4), Some(1), Some(2), Some(5), Some(3), Some(0), Some(6)]
        );
        // DESC is the exact reverse (equal pairs stay in input order and do not swap).
        assert_eq!(
            sorted_ids(c, 7, true),
            vec![Some(0), Some(6), Some(3), Some(5), Some(1), Some(2), Some(4)]
        );
    }

    #[test]
    fn f64_nan_and_nulls_are_independent() {
        let f = |v: f64| Some(Value::F64(v));
        let c = vector(Ty::Double, &[f(f64::NAN), None, f(1.0)]);
        let ty = c.ty();
        // NaN is "the largest number" and NULL goes to the flag's side. The two do not mix.
        let rows =
            flat(vec![Script::Rows(vec![c.clone(), ids(3)])], vec![key(0, ty, false, true)], None);
        assert_eq!(col_i32(&rows, 1), vec![Some(1), Some(2), Some(0)]);
        let rows = flat(vec![Script::Rows(vec![c, ids(3)])], vec![key(0, ty, false, false)], None);
        assert_eq!(col_i32(&rows, 1), vec![Some(2), Some(0), Some(1)]);
    }

    // --- Top-N --------------------------------------------------------------

    #[test]
    fn limit_zero_emits_nothing() {
        let steps = vec![Script::Rows(vec![ints(&[Some(1), Some(2)])])];
        assert!(drive(steps, vec![key(0, Ty::Int, false, false)], Some(0)).is_empty());
    }

    #[test]
    fn limit_smaller_equal_and_larger_than_input() {
        let rows = || vec![Script::Rows(vec![ints(&[Some(3), Some(1), Some(2)])])];
        let ks = || vec![key(0, Ty::Int, false, false)];
        assert_eq!(col_i32(&flat(rows(), ks(), Some(2)), 0), vec![Some(1), Some(2)]);
        assert_eq!(col_i32(&flat(rows(), ks(), Some(3)), 0), vec![Some(1), Some(2), Some(3)]);
        assert_eq!(col_i32(&flat(rows(), ks(), Some(9)), 0), vec![Some(1), Some(2), Some(3)]);
    }

    #[test]
    fn top_n_over_many_rows_comes_out_in_order() {
        // 5000 rows are fed 2048 at a time, triggering compaction repeatedly.
        // The keys are a permutation of 0..4999 (gcd(37, 5000) = 1).
        const N: usize = 5000;
        let mut steps = Vec::new();
        let mut i = 0usize;
        while i < N {
            let end = (i + BATCH_SIZE).min(N);
            let k: Vec<Option<i32>> = (i..end).map(|j| Some(((j * 37) % N) as i32)).collect();
            let id: Vec<Option<i32>> = (i..end).map(|j| Some(j as i32)).collect();
            steps.push(Script::Rows(vec![ints(&k), ints(&id)]));
            i = end;
        }
        let rows = flat(steps, vec![key(0, Ty::Int, false, false)], Some(5));
        assert_eq!(col_i32(&rows, 0), vec![Some(0), Some(1), Some(2), Some(3), Some(4)]);
        // The original row numbers must match too (checking that gather is not misaligned). 37's
        // inverse modulo 5000 is 2973, so the row that produced key v is j = v * 2973 mod 5000.
        let expect: Vec<Option<i32>> = (0..5usize).map(|v| Some(((v * 2973) % N) as i32)).collect();
        assert_eq!(col_i32(&rows, 1), expect);
    }

    #[test]
    fn top_n_is_stable_across_compaction() {
        // Every key is the same. The first 3 rows (in input order) must survive.
        const N: usize = 5000;
        let mut steps = Vec::new();
        let mut i = 0usize;
        while i < N {
            let end = (i + BATCH_SIZE).min(N);
            let id: Vec<Option<i32>> = (i..end).map(|j| Some(j as i32)).collect();
            steps.push(Script::Rows(vec![ints(&vec![Some(1); end - i]), ints(&id)]));
            i = end;
        }
        let rows = flat(steps, vec![key(0, Ty::Int, false, false)], Some(3));
        assert_eq!(col_i32(&rows, 1), vec![Some(0), Some(1), Some(2)]);
    }

    // --- Output batches -----------------------------------------------------

    #[test]
    fn splits_output_into_batch_size_chunks() {
        const N: usize = 5000;
        let k: Vec<Option<i32>> = (0..N).map(|j| Some((N - 1 - j) as i32)).collect();
        let steps = vec![Script::Rows(vec![ints(&k)])];
        let batches = drive(steps, vec![key(0, Ty::Int, false, false)], None);
        assert_eq!(batches.iter().map(|b| b.len()).collect::<Vec<_>>(), vec![2048, 2048, 904]);
        // Ascending overall, even across batch boundaries.
        let all: Vec<Vec<Value>> = batches.into_iter().flatten().collect();
        assert_eq!(all.len(), N);
        assert_eq!(col_i32(&all, 0), (0..N as i32).map(Some).collect::<Vec<_>>());
    }

    #[test]
    fn empty_input_emits_nothing() {
        assert!(drive(Vec::new(), vec![key(0, Ty::Int, false, false)], None).is_empty());
        // The same when only 0-row batches arrive.
        let steps = vec![Script::Rows(vec![ints(&[])]), Script::NeedIo];
        assert!(drive(steps, vec![key(0, Ty::Int, false, false)], None).is_empty());
    }

    #[test]
    fn schema_passes_through_unchanged() {
        let cols = vec![
            ints(&[Some(2), Some(1)]),
            vector(Ty::Varchar, &[Some(Value::Bytes(b"b".to_vec())), None]),
            vector(Ty::Double, &[Some(Value::F64(1.5)), Some(Value::F64(2.5))]),
        ];
        let batches = drive(vec![Script::Rows(cols)], vec![key(0, Ty::Int, false, false)], None);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[0][0].len(), 3, "the column count is unchanged");
        assert_eq!(batches[0][0][0], Value::I32(1));
        assert_eq!(batches[0][0][1], Value::Null);
        assert_eq!(batches[0][0][2], Value::F64(2.5));
        assert_eq!(batches[0][1][1], Value::Bytes(b"b".to_vec()));
    }

    #[test]
    fn selection_vector_on_input_is_respected() {
        let mut op =
            Sort::new(Mock::new(Vec::new()), vec![key(0, Ty::Int, false, false)], None).unwrap();
        let mut cat = Catalog::default();
        let mut vm = Vm::new();
        let mut ctx =
            ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
        let mut batch = Batch::new(vec![ints(&[Some(5), Some(1), Some(9), Some(3)]), ids(4)]);
        batch.sel = Some(vec![2, 1]);
        op.absorb(batch, &mut ctx).unwrap();
        op.finish().unwrap();
        let b = match op.emit().unwrap() {
            Step::Ready(b) => b,
            _ => panic!("expected rows"),
        };
        assert_eq!(col_i32(&rows_of(&b), 0), vec![Some(1), Some(9)]);
        assert_eq!(col_i32(&rows_of(&b), 1), vec![Some(1), Some(2)]);
    }

    // --- Properties of the comparator ---------------------------------------

    #[test]
    fn comparator_is_a_total_order() {
        // Antisymmetry and transitivity must hold even with NULL / NaN / ties mixed in.
        let f = |v: f64| Some(Value::F64(v));
        let c = vector(
            Ty::Double,
            &[f(f64::NAN), None, f(0.0), f(-0.0), f(1.0), None, f(f64::NEG_INFINITY), f(f64::NAN)],
        );
        let keys = vec![SortKey { expr: col_expr(0, Ty::Double), desc: true, nulls_first: true }];
        let cols = vec![c];
        let n = 8u32;
        for a in 0..n {
            assert_eq!(cmp_row(&keys, &cols, a, a), Ordering::Equal);
            for b in 0..n {
                let ab = cmp_row(&keys, &cols, a, b);
                assert_eq!(ab.reverse(), cmp_row(&keys, &cols, b, a), "{a} vs {b}");
                if a != b {
                    assert_ne!(ab, Ordering::Equal, "{a} vs {b}");
                }
                for d in 0..n {
                    if ab == Ordering::Less && cmp_row(&keys, &cols, b, d) == Ordering::Less {
                        assert_eq!(cmp_row(&keys, &cols, a, d), Ordering::Less, "{a}<{b}<{d}");
                    }
                }
            }
        }
    }

    #[test]
    fn appended_vectors_keep_values_and_validity() {
        let mut dst = Vector::new(Ty::Varchar);
        let s1 = vector(Ty::Varchar, &[Some(Value::Bytes(b"ab".to_vec())), None]);
        let s2 = vector(
            Ty::Varchar,
            &[Some(Value::Bytes(b"".to_vec())), Some(Value::Bytes(b"cde".to_vec()))],
        );
        append(&mut dst, &s1).unwrap();
        append(&mut dst, &s2).unwrap();
        assert_eq!(dst.len(), 4);
        assert_eq!(dst.value_at(0), Value::Bytes(b"ab".to_vec()));
        assert_eq!(dst.value_at(1), Value::Null);
        assert_eq!(dst.value_at(2), Value::Bytes(Vec::new()));
        assert_eq!(dst.value_at(3), Value::Bytes(b"cde".to_vec()));

        // When NULLs arrive later (dst has no validity yet).
        let mut dst = Vector::new(Ty::Boolean);
        append(&mut dst, &vector(Ty::Boolean, &[Some(Value::Bool(true))])).unwrap();
        append(&mut dst, &vector(Ty::Boolean, &[None, Some(Value::Bool(false))])).unwrap();
        // Conversely, a vector without NULLs arriving later must not desynchronize validity's length.
        append(&mut dst, &vector(Ty::Boolean, &[Some(Value::Bool(true))])).unwrap();
        assert_eq!(dst.len(), 4);
        assert_eq!(dst.value_at(0), Value::Bool(true));
        assert_eq!(dst.value_at(1), Value::Null);
        assert_eq!(dst.value_at(2), Value::Bool(false));
        assert_eq!(dst.value_at(3), Value::Bool(true));
    }

    #[test]
    fn nulls_survive_batches_that_have_none() {
        // In the order: a batch with NULLs -> a batch without -> a batch with.
        let steps = vec![
            Script::Rows(vec![ints(&[Some(4), None]), ints(&[Some(0), Some(1)])]),
            Script::Rows(vec![ints(&[Some(2), Some(6)]), ints(&[Some(2), Some(3)])]),
            Script::Rows(vec![ints(&[None, Some(1)]), ints(&[Some(4), Some(5)])]),
        ];
        let rows = flat(steps, vec![key(0, Ty::Int, false, false)], None);
        assert_eq!(
            col_i32(&rows, 0),
            vec![Some(1), Some(2), Some(4), Some(6), None, None],
            "exactly two NULLs at the end"
        );
        assert_eq!(col_i32(&rows, 1), vec![Some(5), Some(2), Some(0), Some(3), Some(1), Some(4)]);
    }

    #[test]
    fn buffer_size_estimate_tracks_appends() {
        // Actually exceeding the cap (256 MiB) in a test is impractical, so this only confirms the
        // estimation function grows with the row count.
        let empty = vector(Ty::BigInt, &[]);
        assert_eq!(vector_bytes(&empty), 0);
        let mut big = Vector::new(Ty::BigInt);
        append(&mut big, &vector(Ty::BigInt, &[Some(Value::I64(1)), None])).unwrap();
        // 8 B per value x 2 rows, plus validity.
        assert!(vector_bytes(&big) >= 16);
        let mut s = Vector::new(Ty::Varchar);
        append(&mut s, &vector(Ty::Varchar, &[Some(Value::Bytes(b"abcdef".to_vec()))])).unwrap();
        assert!(vector_bytes(&s) >= 6);
    }
}
