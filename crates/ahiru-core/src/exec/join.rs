//! Hash join and nested-loop join.
//!
//! The build side is fixed to **always be the right input**. Which side goes right (= making
//! the smaller one the build side) is the binder's job and is not decided here. Swapping at
//! runtime would change the output column order (left columns then right) and break `residual` and the schema above.
//!
//! ## Interruption and resumption
//!
//! The inputs can return `NeedIo` / `NeedCodec`. Probing cannot start until the right input
//! is read through, so the phase (`Building` -> `Probing` -> `DrainingUnmatched` -> `Done`) is
//! held explicitly; on interruption it simply exits and the next `next()` resumes from the same
//! position. The hash table, the batch being probed, and the match bitmap all live in `self`.
//!
//! ## Memory
//!
//! The build side is held entirely in memory. There is no mechanism to spill overflow to disk
//! (a known limitation). Instead `MAX_BUILD_BYTES` caps it and returns `Oom`.
//!
//! ## Semi / Anti
//!
//! The rewrite target of `IN (SELECT)` / `EXISTS`. The output is **the left columns only**, and
//! one left row appears at most once. The probing machinery is shared: rather than returning
//! candidate pairs it only sets the match bit (`Probe::matched`), and the drain at the end of a
//! batch emits the matched rows for Semi and the unmatched rows for Anti.
//!
//! A left row with a NULL key matches no build row, so it disappears under Semi and survives
//! under Anti. `Anti` plainly does "emit if there is no match" and does not consider SQL's
//! three-valued logic (the semantics of `NOT EXISTS`).
//!
//! `NOT IN (SELECT ...)` is a different thing and uses `AntiNullAware`. The three-valued logic
//! confirmed with DuckDB is:
//!
//! - If the right keys contain even one NULL, every comparison with any left row is UNKNOWN.
//!   **The result is empty** (`2 NOT IN (1, NULL)` is UNKNOWN, not true).
//! - A left row with a NULL key is UNKNOWN unless the right side is empty, so it is not emitted.
//! - If the right side is empty, `x NOT IN ()` is true even for a NULL left. **Every row** is emitted.
//!
//! Which one to use is the binder's decision. The operator just follows the `kind` it is given.

use crate::exec::rowkey::{encode_key, key_has_null, HashIndex};
use crate::exec::{ExecContext, Operator, Step};
use crate::expr::Program;
use crate::prelude::*;
use crate::sql::ast::JoinKind;
use crate::vector::{Batch, Bitmap, Data, Ty, Vector, BATCH_SIZE};

/// The row number meaning "no counterpart". Also used as a chain terminator.
const NONE: u32 = u32::MAX;

/// How many bytes of the build side may be buffered. With no spilling, exceeding this fails
/// with an error instead of silently grabbing enormous memory.
const MAX_BUILD_BYTES: usize = 128 * 1024 * 1024;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Reading the right input through and building the hash table.
    Building,
    /// Probing the left input one batch at a time.
    Probing,
    /// For RIGHT/FULL, emitting the build rows that matched no left row.
    DrainingUnmatched,
    Done,
}

/// The left batch being probed, and its progress.
struct Probe {
    /// The left batch with selection materialized. Row numbers serve directly as indices.
    batch: Batch,
    /// The evaluated left keys. The same row count as `batch`.
    keys: Vec<Vector>,
    /// The left row being scanned.
    row: usize,
    /// The next build row to try. `None` means this left row has not been started yet,
    /// and `Some(NONE)` means the candidates are exhausted.
    cursor: Option<u32>,
    /// Per left row: "was there a match that made it through residual".
    matched: Bitmap,
    /// The next row to look at when NULL-extending unmatched left rows.
    drain: usize,
}

pub struct HashJoin {
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    left_keys: Vec<Program>,
    right_keys: Vec<Program>,
    residual: Option<Program>,
    left_types: Vec<Ty>,
    right_types: Vec<Ty>,

    /// `kind` affects only these flags, so it is folded rather than kept.
    /// Whether to emit unmatched left rows (LEFT / FULL NULL-extend; ANTI emits the left columns only).
    emit_unmatched_left: bool,
    /// Whether to emit a matched left row exactly once (rather than as join pairs) -- SEMI.
    emit_matched_left: bool,
    /// Whether the output is the left columns only (SEMI / ANTI / ANTI NULL AWARE).
    left_only: bool,
    /// Whether to reproduce `NOT IN`'s three-valued logic (ANTI NULL AWARE).
    null_aware: bool,
    /// Whether to NULL-extend and emit unmatched build rows (RIGHT / FULL).
    emit_unmatched_right: bool,

    phase: Phase,
    /// The buffered build side. Every batch of the right input concatenated into one.
    build_cols: Vec<Vector>,
    build_rows: usize,
    /// The approximate amount buffered. Used only for the cap check.
    build_bytes: usize,
    /// Key -> the build row number at the head of the chain.
    index: HashIndex,
    /// The chain of build rows sharing a key. Terminated by `NONE`.
    next: Vec<u32>,
    /// The match flag per build row. Allocated only for RIGHT/FULL.
    build_matched: Bitmap,
    /// Whether any build-side key row contained a NULL. Used only by ANTI NULL AWARE.
    build_has_null: bool,
    /// The destination `encode_key` writes to. Reused rather than allocated per row.
    keybuf: Vec<u8>,
    probe: Option<Probe>,
    /// The next build row to look at in `DrainingUnmatched`.
    drain: usize,
}

impl HashJoin {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        left: Box<dyn Operator>,
        right: Box<dyn Operator>,
        kind: JoinKind,
        left_keys: Vec<Program>,
        right_keys: Vec<Program>,
        residual: Option<Program>,
        left_types: Vec<Ty>,
        right_types: Vec<Ty>,
    ) -> Result<Self> {
        // Equality conditions pair up left and right. A mismatch is a binder bug.
        ensure!(left_keys.len() == right_keys.len(), Internal);
        let build_cols = right_types.iter().map(|t| Vector::new(*t)).collect();
        Ok(HashJoin {
            left,
            right,
            left_keys,
            right_keys,
            residual,
            left_types,
            right_types,
            // ANTI emits "left rows that did not match", so it sits on the same side as LEFT.
            // The only difference is not laying out the right columns (`left_only`).
            emit_unmatched_left: matches!(
                kind,
                JoinKind::Left | JoinKind::Full | JoinKind::Anti | JoinKind::AntiNullAware
            ),
            emit_matched_left: kind == JoinKind::Semi,
            left_only: kind.is_semi(),
            null_aware: kind == JoinKind::AntiNullAware,
            emit_unmatched_right: matches!(kind, JoinKind::Right | JoinKind::Full),
            phase: Phase::Building,
            build_cols,
            build_rows: 0,
            build_bytes: 0,
            index: HashIndex::new(),
            next: Vec::new(),
            build_matched: Bitmap::new(),
            build_has_null: false,
            keybuf: Vec::new(),
            probe: None,
            drain: 0,
        })
    }

    /// No equality keys = brute force with a nested loop. CROSS and non-equi joins such as
    /// `ON a.x < b.y` come here.
    #[inline]
    fn nested(&self) -> bool {
        self.right_keys.is_empty()
    }

    /// Takes one right batch into the build side.
    fn absorb(&mut self, ctx: &mut ExecContext, mut batch: Batch) -> Result<()> {
        // From here on lookups are by row number, so selection is materialized now.
        batch.materialize();
        let rows = batch.num_rows();
        if rows == 0 {
            return Ok(());
        }
        ensure!(batch.cols.len() == self.right_types.len(), Internal);

        // Keys are evaluated before concatenation (already materialized, so row numbers correspond directly).
        let mut keys = Vec::with_capacity(self.right_keys.len());
        for p in &self.right_keys {
            keys.push(ctx.vm.eval(p, &batch)?);
        }

        let base = self.build_rows;
        for (dst, src) in self.build_cols.iter_mut().zip(batch.cols.iter()) {
            append_all(dst, src)?;
            self.build_bytes += vector_bytes(src);
        }
        self.build_rows += rows;
        // Row numbers are held as u32. The byte cap should bite first, but a zero-column batch
        // (the COUNT(*) path) does not grow the byte count, so this is checked explicitly.
        ensure!(self.build_rows < NONE as usize, Oom);
        ensure!(self.build_bytes + self.index.key_bytes() <= MAX_BUILD_BYTES, Oom);

        if !self.nested() {
            self.next.resize(self.build_rows, NONE);
            let refs: Vec<&Vector> = keys.iter().collect();
            for r in 0..rows {
                // A key containing NULL matches nothing (SQL's `=` is not NULL-safe), so it is
                // not put in the table. The row itself remains and appears in the OUTER unmatched drain.
                if key_has_null(&refs, r) {
                    // Only for `NOT IN` does "whether the right side has a NULL" itself change
                    // the answer, so having seen one is remembered.
                    self.build_has_null = true;
                    continue;
                }
                encode_key(&refs, r, &mut self.keybuf);
                let id = (base + r) as u32;
                self.next[base + r] = self.index.insert_chained(&self.keybuf, id).unwrap_or(NONE);
            }
        }
        Ok(())
    }

    /// Advances the probe phase by one step. `None` means "no output, but state advanced".
    fn probe_step(&mut self, ctx: &mut ExecContext) -> Result<Option<Step>> {
        if self.probe.is_none() {
            let mut batch = match self.left.next(ctx)? {
                Step::Ready(b) => b,
                Step::Done => {
                    self.phase = if self.emit_unmatched_right {
                        Phase::DrainingUnmatched
                    } else {
                        Phase::Done
                    };
                    return Ok(None);
                }
                // Interrupted. The hash table stays, and probing exits still unstarted.
                other => return Ok(Some(other)),
            };
            batch.materialize();
            let rows = batch.num_rows();
            if rows == 0 {
                return Ok(None);
            }
            ensure!(batch.cols.len() == self.left_types.len(), Internal);
            let mut keys = Vec::with_capacity(self.left_keys.len());
            for p in &self.left_keys {
                keys.push(ctx.vm.eval(p, &batch)?);
            }
            self.probe = Some(Probe {
                batch,
                keys,
                row: 0,
                cursor: None,
                matched: Bitmap::zeros(rows),
                drain: 0,
            });
        }

        // --- Build up to BATCH_SIZE candidate pairs ---------------------------
        // One left row can hit many build rows, so `cursor` is kept so it can break off in the
        // middle of a left row.
        let mut lidx: Vec<u32> = Vec::new();
        let mut ridx: Vec<u32> = Vec::new();
        {
            let p = match self.probe.as_mut() {
                Some(p) => p,
                None => err!(Internal),
            };
            let rows = p.batch.num_rows();
            let refs: Vec<&Vector> = p.keys.iter().collect();
            while lidx.len() < BATCH_SIZE && p.row < rows {
                // SEMI / ANTI only need "is there at least one match". A left row whose match is
                // settled breaks off without walking the rest of the chain. `matched` is only set
                // by pairs that passed residual, so it is never too early.
                if self.left_only && p.matched.get(p.row) {
                    p.row += 1;
                    p.cursor = None;
                    continue;
                }
                let cur = match p.cursor {
                    Some(c) => c,
                    None => {
                        // The same check as `nested()`. Methods cannot be called while `probe` is
                        // borrowed, so the fields are read directly.
                        let head = if self.right_keys.is_empty() {
                            if self.build_rows == 0 {
                                NONE
                            } else {
                                0
                            }
                        } else if key_has_null(&refs, p.row) {
                            // The same on the probe side. A NULL key matches no build row.
                            //
                            // But under `NOT IN`, `NULL NOT IN (a non-empty set)` is UNKNOWN, so
                            // this left row must not be returned either. It is marked as matched to
                            // exclude it from the drain (when the right side is empty,
                            // `NULL NOT IN ()` is true, so only then is it emitted).
                            if self.null_aware && self.build_rows > 0 {
                                p.matched.set(p.row, true);
                            }
                            NONE
                        } else {
                            encode_key(&refs, p.row, &mut self.keybuf);
                            self.index.lookup(&self.keybuf).unwrap_or(NONE)
                        };
                        p.cursor = Some(head);
                        head
                    }
                };
                if cur == NONE {
                    p.row += 1;
                    p.cursor = None;
                    continue;
                }
                lidx.push(p.row as u32);
                ridx.push(cur);
                p.cursor = Some(if self.right_keys.is_empty() {
                    // A nested loop simply advances to the next build row.
                    if (cur as usize) + 1 < self.build_rows {
                        cur + 1
                    } else {
                        NONE
                    }
                } else {
                    self.next.get(cur as usize).copied().unwrap_or(NONE)
                });
            }
        }

        if !lidx.is_empty() {
            let mut out = {
                let p = match self.probe.as_ref() {
                    Some(p) => p,
                    None => err!(Internal),
                };
                // SEMI / ANTI do not return this batch. Without residual to consider there is no
                // need to assemble it at all, so a container carrying only the row count suffices.
                if self.left_only && self.residual.is_none() {
                    Batch::rows_only(lidx.len())
                } else {
                    assemble(
                        &p.batch.cols,
                        &self.left_types,
                        Some(&lidx),
                        &self.build_cols,
                        &self.right_types,
                        Some(&ridx),
                        lidx.len(),
                    )
                }
            };
            // residual is applied **before** counting a match. A pair whose keys agreed but that
            // residual rejected is not a match, so under OUTER that left row must still be
            // NULL-extended and emitted.
            let keep: Option<Vec<u32>> = match &self.residual {
                Some(r) => {
                    let mut sel = Vec::new();
                    ctx.vm.eval_filter(r, &out, &mut sel)?;
                    Some(sel)
                }
                None => None,
            };
            {
                let p = match self.probe.as_mut() {
                    Some(p) => p,
                    None => err!(Internal),
                };
                let mut mark = |i: usize| {
                    p.matched.set(lidx[i] as usize, true);
                    if self.emit_unmatched_right {
                        self.build_matched.set(ridx[i] as usize, true);
                    }
                };
                match &keep {
                    Some(sel) => {
                        for &i in sel.iter() {
                            mark(i as usize);
                        }
                    }
                    None => {
                        for i in 0..lidx.len() {
                            mark(i);
                        }
                    }
                }
            }
            // SEMI / ANTI do not return join pairs. Only the match bit is set, and output happens
            // together in the drain below (a left row appears at most once).
            if self.left_only {
                return Ok(None);
            }
            match keep {
                // The candidates are exhausted. An empty batch is not returned; move to the next chunk.
                Some(sel) if sel.is_empty() => return Ok(None),
                Some(sel) => out.sel = Some(sel),
                None => {}
            }
            return Ok(Some(Step::Ready(out)));
        }

        // --- Candidates exhausted. The stage that emits the left rows themselves --
        // LEFT/FULL NULL-extend unmatched rows, ANTI emits unmatched rows' left columns only, and
        // SEMI emits matched rows' left columns only. The pickup condition collapses to the single
        // test "match bit == emit_matched_left".
        let mut idx: Vec<u32> = Vec::new();
        if self.emit_unmatched_left || self.emit_matched_left {
            let p = match self.probe.as_mut() {
                Some(p) => p,
                None => err!(Internal),
            };
            let rows = p.batch.num_rows();
            while p.drain < rows && idx.len() < BATCH_SIZE {
                if p.matched.get(p.drain) == self.emit_matched_left {
                    idx.push(p.drain as u32);
                }
                p.drain += 1;
            }
        }
        if !idx.is_empty() {
            let p = match self.probe.as_ref() {
                Some(p) => p,
                None => err!(Internal),
            };
            return Ok(Some(Step::Ready(assemble(
                &p.batch.cols,
                &self.left_types,
                Some(&idx),
                &self.build_cols,
                // SEMI / ANTI output only the left schema.
                if self.left_only { &[] } else { &self.right_types },
                None,
                idx.len(),
            ))));
        }

        // This batch is finished. Pull the next left batch.
        self.probe = None;
        Ok(None)
    }
}

impl Operator for HashJoin {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        loop {
            match self.phase {
                Phase::Building => match self.right.next(ctx)? {
                    Step::Ready(b) => self.absorb(ctx, b)?,
                    Step::Done => {
                        // Only RIGHT/FULL emit unmatched build rows. It is a bitmap sized to the
                        // row count, so it is not allocated when unnecessary.
                        if self.emit_unmatched_right {
                            self.build_matched = Bitmap::zeros(self.build_rows);
                        }
                        // Under `NOT IN`, a NULL among the right keys makes every comparison with
                        // any left row UNKNOWN and the result empty. It finishes without pulling a single left row.
                        self.phase = if self.null_aware && self.build_has_null {
                            Phase::Done
                        } else {
                            Phase::Probing
                        };
                    }
                    // NeedIo / NeedCodec. Exits with the partially built hash table intact.
                    other => return Ok(other),
                },
                Phase::Probing => {
                    if let Some(step) = self.probe_step(ctx)? {
                        return Ok(step);
                    }
                }
                Phase::DrainingUnmatched => {
                    let mut idx: Vec<u32> = Vec::new();
                    while self.drain < self.build_rows && idx.len() < BATCH_SIZE {
                        if !self.build_matched.get(self.drain) {
                            idx.push(self.drain as u32);
                        }
                        self.drain += 1;
                    }
                    if idx.is_empty() {
                        self.phase = Phase::Done;
                        continue;
                    }
                    return Ok(Step::Ready(assemble(
                        &[],
                        &self.left_types,
                        None,
                        &self.build_cols,
                        &self.right_types,
                        Some(&idx),
                        idx.len(),
                    )));
                }
                Phase::Done => return Ok(Step::Done),
            }
        }
    }
}

// --- Assembling the output ---------------------------------------------------

/// Builds a batch laying the right columns after the left ones. A side whose index is `None`
/// is all NULL (OUTER NULL extension). The column order is a contract with the schema the binder builds, so it does not change.
fn assemble(
    lcols: &[Vector],
    ltys: &[Ty],
    lidx: Option<&[u32]>,
    rcols: &[Vector],
    rtys: &[Ty],
    ridx: Option<&[u32]>,
    n: usize,
) -> Batch {
    let mut cols = Vec::with_capacity(ltys.len() + rtys.len());
    push_side(&mut cols, lcols, ltys, lidx, n);
    push_side(&mut cols, rcols, rtys, ridx, n);
    if cols.is_empty() {
        // Two inputs with no columns (the COUNT(*) path). Only the row count is conveyed.
        return Batch::rows_only(n);
    }
    Batch::new(cols)
}

fn push_side(out: &mut Vec<Vector>, cols: &[Vector], tys: &[Ty], idx: Option<&[u32]>, n: usize) {
    for (i, ty) in tys.iter().enumerate() {
        match (idx, cols.get(i)) {
            (Some(ix), Some(c)) => out.push(gather_opt(c, ix, *ty)),
            _ => out.push(null_vector(*ty, n)),
        }
    }
}

/// `Vector::gather` extended to allow the "no counterpart" marker (`NONE`).
fn gather_opt(src: &Vector, idx: &[u32], ty: Ty) -> Vector {
    if !idx.contains(&NONE) {
        return src.gather(idx);
    }
    if src.is_empty() {
        // With an empty build side there is no row to pick up. Return all NULL.
        return null_vector(ty, idx.len());
    }
    // Pick up an arbitrary row and then clear validity. Avoids copying rows through `Value`.
    let safe: Vec<u32> = idx.iter().map(|&i| if i == NONE { 0 } else { i }).collect();
    let mut v = src.gather(&safe);
    let bm = v.validity_mut();
    for (k, &i) in idx.iter().enumerate() {
        if i == NONE {
            bm.set(k, false);
        }
    }
    v
}

fn null_vector(ty: Ty, n: usize) -> Vector {
    let mut v = Vector::with_capacity(ty, n);
    for _ in 0..n {
        v.push_null();
    }
    v
}

/// Concatenates every row of `src` onto the end of `dst`. Going through `Value` row by row
/// would make allocations proportional to the row count for variable-length columns, so it is done in batch units.
fn append_all(dst: &mut Vector, src: &Vector) -> Result<()> {
    let base = dst.len();
    let n = src.len();
    match (dst.data_mut(), src.data()) {
        (Data::Bool(d), Data::Bool(s)) => {
            for i in 0..n {
                d.push(s.get(i));
            }
        }
        (Data::I32(d), Data::I32(s)) => d.extend_from_slice(s),
        (Data::I64(d), Data::I64(s)) => d.extend_from_slice(s),
        (Data::I128(d), Data::I128(s)) => d.extend_from_slice(s),
        (Data::F64(d), Data::F64(s)) => d.extend_from_slice(s),
        (Data::Bytes(d), Data::Bytes(s)) => {
            for i in 0..n {
                d.push(s.get(i));
            }
        }
        // A physical type mismatch is an upstream bug.
        _ => err!(Internal),
    }
    if src.has_nulls() || dst.has_nulls() {
        // `validity_mut` fills any shortfall as "valid", so only the NULL positions are cleared.
        let bm = dst.validity_mut();
        for i in 0..n {
            if !src.is_valid(i) {
                bm.set(base + i, false);
            }
        }
    }
    Ok(())
}

/// An estimate of the buffered amount. Used only for the cap check, so it need not be exact.
fn vector_bytes(v: &Vector) -> usize {
    let d = match v.data() {
        Data::Bool(b) => b.len() / 8 + 1,
        Data::I32(x) => x.len() * 4,
        Data::I64(x) => x.len() * 8,
        Data::F64(x) => x.len() * 8,
        Data::I128(x) => x.len() * 16,
        Data::Bytes(b) => b.data.len() + (b.len() + 1) * 4,
    };
    // Plus validity and the join chain.
    d + v.len() / 8 + 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Catalog;
    use crate::expr::vm::Vm;
    use crate::expr::{Instr, OpCode};
    use crate::vector::Value;

    // --- Mock inputs --------------------------------------------------------

    /// An input returning `Step`s from a script. Interposing `NeedIo` reproduces exactly the
    /// situation of a remote input being interrupted midway.
    struct Mock {
        steps: Vec<Option<Step>>,
        pos: usize,
    }

    impl Mock {
        fn script(steps: Vec<Step>) -> Box<dyn Operator> {
            Box::new(Mock { steps: steps.into_iter().map(Some).collect(), pos: 0 })
        }
        fn empty() -> Box<dyn Operator> {
            Mock::script(Vec::new())
        }
    }

    impl Operator for Mock {
        fn next(&mut self, _ctx: &mut ExecContext) -> Result<Step> {
            if self.pos >= self.steps.len() {
                return Ok(Step::Done);
            }
            let s = self.steps[self.pos].take();
            self.pos += 1;
            Ok(s.unwrap_or(Step::Done))
        }
    }

    // --- Construction helpers -----------------------------------------------

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

    fn strs(vals: &[Option<&str>]) -> Vector {
        let mut v = Vector::new(Ty::Varchar);
        for x in vals {
            match x {
                Some(x) => v.push_value(&Value::Bytes(x.as_bytes().to_vec())),
                None => v.push_null(),
            }
        }
        v
    }

    fn dbls(vals: &[f64]) -> Vector {
        let mut v = Vector::new(Ty::Double);
        for x in vals {
            v.push_value(&Value::F64(*x));
        }
        v
    }

    fn ready(cols: Vec<Vector>) -> Step {
        Step::Ready(Batch::new(cols))
    }

    /// A program using column `i` directly as the key.
    fn col_prog(i: u16, ty: Ty) -> Program {
        let mut p = Program::new();
        let r = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, ty.phys(), r, 0, 0, i));
        p.result = r;
        p.result_ty = ty;
        p
    }

    /// A program returning `col a <op> col b` (for residual).
    fn cmp_prog(a: u16, b: u16, ty: Ty, op: OpCode) -> Program {
        let mut p = Program::new();
        let ra = p.alloc_reg();
        let rb = p.alloc_reg();
        let rd = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, ty.phys(), ra, 0, 0, a));
        p.push(Instr::with_aux(OpCode::LoadCol, ty.phys(), rb, 0, 0, b));
        p.push(Instr::new(op, ty.phys(), rd, ra, rb));
        p.result = rd;
        p.result_ty = Ty::Boolean;
        p
    }

    struct Runner {
        rows: Vec<Vec<Value>>,
        /// How many `NeedIo`s came back.
        interrupts: usize,
        batches: usize,
    }

    fn run(op: &mut dyn Operator) -> Runner {
        let mut cat = Catalog::new();
        let mut vm = Vm::new();
        let mut out = Runner { rows: Vec::new(), interrupts: 0, batches: 0 };
        for guard in 0..100_000 {
            assert!(guard < 99_999, "does not terminate");
            let mut ctx =
                ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
            match op.next(&mut ctx).unwrap() {
                Step::Ready(b) => {
                    let n = b.card();
                    assert!(n > 0, "an empty batch must not be returned");
                    assert!(n <= BATCH_SIZE, "{n} rows in one next exceeds the batch cap");
                    out.batches += 1;
                    for i in 0..n {
                        let r = match &b.sel {
                            Some(s) => s[i] as usize,
                            None => i,
                        };
                        out.rows.push(b.cols.iter().map(|c| c.value_at(r)).collect());
                    }
                }
                Step::Done => break,
                _ => out.interrupts += 1,
            }
        }
        out
    }

    /// Flattens rows to integers (NULL as `None`) and sorts them, since the order is not guaranteed.
    fn norm(rows: &[Vec<Value>]) -> Vec<Vec<Option<i64>>> {
        let mut v: Vec<Vec<Option<i64>>> =
            rows.iter().map(|r| r.iter().map(|x| x.as_i64()).collect()).collect();
        v.sort();
        v
    }

    fn text(rows: &[Vec<Value>]) -> Vec<Vec<Option<String>>> {
        let mut v: Vec<Vec<Option<String>>> = rows
            .iter()
            .map(|r| {
                r.iter()
                    .map(|x| x.as_bytes().map(|b| String::from_utf8_lossy(b).into_owned()))
                    .collect()
            })
            .collect();
        v.sort();
        v
    }

    fn join(
        left: Vec<Step>,
        right: Vec<Step>,
        kind: JoinKind,
        keys: usize,
        lty: Vec<Ty>,
        rty: Vec<Ty>,
    ) -> HashJoin {
        let lk = (0..keys).map(|i| col_prog(i as u16, lty[i])).collect();
        let rk = (0..keys).map(|i| col_prog(i as u16, rty[i])).collect();
        HashJoin::new(Mock::script(left), Mock::script(right), kind, lk, rk, None, lty, rty)
            .unwrap()
    }

    /// A join of one column each (an INT key only).
    fn join1(left: Vec<Step>, right: Vec<Step>, kind: JoinKind) -> HashJoin {
        join(left, right, kind, 1, vec![Ty::Int], vec![Ty::Int])
    }

    fn ints1(vals: &[Option<i32>]) -> Step {
        ready(vec![ints(vals)])
    }

    // --- Interruption and resumption ----------------------------------------

    /// The most important one. The result must not change when `NeedIo` interposes during build or probe.
    #[test]
    fn need_io_mid_build_and_mid_probe_is_transparent() {
        let l = || vec![ints1(&[Some(1), Some(2)]), ints1(&[Some(3), Some(1)])];
        let r = || vec![ints1(&[Some(1), Some(3)]), ints1(&[Some(1), Some(9)])];
        let clean = run(&mut join1(l(), r(), JoinKind::Full));

        let interrupted_left = vec![
            Step::NeedIo,
            ints1(&[Some(1), Some(2)]),
            Step::NeedIo,
            ints1(&[Some(3), Some(1)]),
            Step::NeedIo,
        ];
        let interrupted_right = vec![
            ints1(&[Some(1), Some(3)]),
            Step::NeedIo,
            Step::NeedIo,
            ints1(&[Some(1), Some(9)]),
            Step::NeedIo,
        ];
        let got = run(&mut join1(interrupted_left, interrupted_right, JoinKind::Full));
        assert!(got.interrupts >= 5, "the interruptions were not passed through");
        assert_eq!(norm(&got.rows), norm(&clean.rows));

        // The same for INNER / LEFT / RIGHT.
        for kind in [JoinKind::Inner, JoinKind::Left, JoinKind::Right] {
            let clean = run(&mut join1(l(), r(), kind));
            let noisy = run(&mut join1(
                vec![
                    Step::NeedIo,
                    ints1(&[Some(1), Some(2)]),
                    Step::NeedIo,
                    ints1(&[Some(3), Some(1)]),
                ],
                vec![
                    Step::NeedIo,
                    ints1(&[Some(1), Some(3)]),
                    Step::NeedIo,
                    ints1(&[Some(1), Some(9)]),
                ],
                kind,
            ));
            assert!(noisy.interrupts >= 4);
            assert_eq!(norm(&noisy.rows), norm(&clean.rows), "{kind:?}");
        }
    }

    /// When one left row produces more output than BATCH_SIZE, the chain position must not be
    /// lost even if the probe is interrupted midway.
    #[test]
    fn need_io_does_not_disturb_a_long_chain() {
        let big: Vec<Option<i32>> = (0..3000).map(|_| Some(7)).collect();
        let left = vec![Step::NeedIo, ints1(&[Some(7)]), Step::NeedIo];
        let right = vec![Step::NeedIo, ready(vec![ints(&big)]), Step::NeedIo];
        let got = run(&mut join1(left, right, JoinKind::Inner));
        assert_eq!(got.rows.len(), 3000);
        assert_eq!(got.batches, 2, "split at BATCH_SIZE");
    }

    // --- INNER --------------------------------------------------------------

    #[test]
    fn inner_no_match_is_empty() {
        let got = run(&mut join1(
            vec![ints1(&[Some(1), Some(2)])],
            vec![ints1(&[Some(3)])],
            JoinKind::Inner,
        ));
        assert!(got.rows.is_empty());
    }

    #[test]
    fn inner_one_to_one() {
        let got = run(&mut join1(
            vec![ints1(&[Some(1), Some(2), Some(3)])],
            vec![ints1(&[Some(2), Some(3), Some(4)])],
            JoinKind::Inner,
        ));
        assert_eq!(norm(&got.rows), vec![vec![Some(2), Some(2)], vec![Some(3), Some(3)]]);
    }

    #[test]
    fn inner_one_to_many() {
        let got = run(&mut join1(
            vec![ints1(&[Some(1), Some(2), Some(3)])],
            vec![ints1(&[Some(1), Some(1), Some(2)])],
            JoinKind::Inner,
        ));
        assert_eq!(got.rows.len(), 3);
        assert_eq!(
            norm(&got.rows),
            vec![vec![Some(1), Some(1)], vec![Some(1), Some(1)], vec![Some(2), Some(2)]]
        );
    }

    /// Many-to-many. Getting the chaining wrong would give the wrong count.
    #[test]
    fn inner_many_to_many_row_count() {
        // Left: 1,1,2,2,3 / right: 1,1,1,2 -> 2*3 + 2*1 = 8
        let got = run(&mut join1(
            vec![ints1(&[Some(1), Some(1), Some(2), Some(2), Some(3)])],
            vec![ints1(&[Some(1), Some(1), Some(1), Some(2)])],
            JoinKind::Inner,
        ));
        assert_eq!(got.rows.len(), 8);
        let ones = got.rows.iter().filter(|r| r[0].as_i64() == Some(1)).count();
        assert_eq!(ones, 6);
    }

    /// The chain must connect even when the right input is split across several batches.
    #[test]
    fn build_side_spanning_batches() {
        let got = run(&mut join1(
            vec![ints1(&[Some(1)])],
            vec![ints1(&[Some(1), Some(2)]), ints1(&[Some(1)]), ints1(&[Some(1), Some(3)])],
            JoinKind::Inner,
        ));
        assert_eq!(got.rows.len(), 3);
    }

    // --- OUTER --------------------------------------------------------------

    #[test]
    fn left_join_null_extends_unmatched_left_rows() {
        let got = run(&mut join1(
            vec![ints1(&[Some(1), Some(2), Some(3)])],
            vec![ints1(&[Some(2)])],
            JoinKind::Left,
        ));
        assert_eq!(
            norm(&got.rows),
            vec![vec![Some(1), None], vec![Some(2), Some(2)], vec![Some(3), None]]
        );
    }

    #[test]
    fn right_join_null_extends_unmatched_build_rows() {
        let got = run(&mut join1(
            vec![ints1(&[Some(2)])],
            vec![ints1(&[Some(1), Some(2), Some(3)])],
            JoinKind::Right,
        ));
        assert_eq!(
            norm(&got.rows),
            vec![vec![None, Some(1)], vec![None, Some(3)], vec![Some(2), Some(2)]]
        );
    }

    #[test]
    fn full_join_extends_both_sides() {
        let got = run(&mut join1(
            vec![ints1(&[Some(1), Some(2)])],
            vec![ints1(&[Some(2), Some(3)])],
            JoinKind::Full,
        ));
        assert_eq!(
            norm(&got.rows),
            vec![vec![None, Some(3)], vec![Some(1), None], vec![Some(2), Some(2)]]
        );
    }

    // --- NULL keys ----------------------------------------------------------

    /// NULL does not match NULL either. Under OUTER it does appear as an unmatched row.
    #[test]
    fn null_keys_never_match_each_other() {
        let left = || vec![ints1(&[None, Some(1)])];
        let right = || vec![ints1(&[None, Some(1)])];

        let got = run(&mut join1(left(), right(), JoinKind::Inner));
        assert_eq!(norm(&got.rows), vec![vec![Some(1), Some(1)]], "NULLs got connected");

        let got = run(&mut join1(left(), right(), JoinKind::Left));
        assert_eq!(norm(&got.rows), vec![vec![None, None], vec![Some(1), Some(1)]]);
        // The row whose left is NULL should appear as "left only valid".
        let extended = got.rows.iter().filter(|r| r[0].is_null() && r[1].is_null()).count();
        assert_eq!(extended, 1);

        let got = run(&mut join1(left(), right(), JoinKind::Right));
        assert_eq!(norm(&got.rows), vec![vec![None, None], vec![Some(1), Some(1)]]);

        let got = run(&mut join1(left(), right(), JoinKind::Full));
        // The left NULL row and the right NULL row, one each, separately.
        assert_eq!(got.rows.len(), 3);
        assert_eq!(
            norm(&got.rows),
            vec![vec![None, None], vec![None, None], vec![Some(1), Some(1)]]
        );
    }

    /// A composite key. If one part is NULL it does not connect even when the other matches.
    #[test]
    fn multi_column_keys_with_one_null() {
        let left = vec![ready(vec![
            ints(&[Some(1), Some(1), Some(2)]),
            ints(&[Some(10), None, Some(20)]),
        ])];
        let right = vec![ready(vec![
            ints(&[Some(1), Some(1), Some(2)]),
            ints(&[Some(10), None, Some(99)]),
        ])];
        let ty = vec![Ty::Int, Ty::Int];
        let got = run(&mut join(left, right, JoinKind::Left, 2, ty.clone(), ty));
        assert_eq!(
            norm(&got.rows),
            vec![
                vec![Some(1), None, None, None], // (1, NULL) matches nothing
                vec![Some(1), Some(10), Some(1), Some(10)],
                vec![Some(2), Some(20), None, None], // the second column differs
            ]
        );
    }

    // --- Keys per type ------------------------------------------------------

    #[test]
    fn string_keys() {
        let left = vec![ready(vec![strs(&[Some("a"), Some("bc"), None])])];
        let right = vec![ready(vec![strs(&[Some("bc"), Some("a"), Some("a")])])];
        let mut j = join(left, right, JoinKind::Left, 1, vec![Ty::Varchar], vec![Ty::Varchar]);
        let got = run(&mut j);
        assert_eq!(
            text(&got.rows),
            vec![
                vec![None, None],
                vec![Some("a".into()), Some("a".into())],
                vec![Some("a".into()), Some("a".into())],
                vec![Some("bc".into()), Some("bc".into())],
            ]
        );
    }

    /// `encode_key` normalizes -0.0 and NaN, so both join.
    #[test]
    fn float_keys_canonicalise_zero_and_nan() {
        let left = vec![ready(vec![dbls(&[-0.0, f64::NAN, 1.5])])];
        let right = vec![ready(vec![dbls(&[0.0, f64::NAN])])];
        let mut j = join(left, right, JoinKind::Inner, 1, vec![Ty::Double], vec![Ty::Double]);
        let got = run(&mut j);
        assert_eq!(got.rows.len(), 2, "2 rows, from -0.0=0.0 and NaN=NaN");
        let z = got.rows.iter().find(|r| r[0].as_f64() == Some(0.0)).unwrap();
        // The left is -0.0 and the right is 0.0. The bit patterns differ yet they join.
        assert_eq!(z[0].as_f64().unwrap().to_bits(), (-0.0f64).to_bits());
        assert_eq!(z[1].as_f64().unwrap().to_bits(), 0.0f64.to_bits());
        let n = got.rows.iter().find(|r| r[0].as_f64().unwrap().is_nan()).unwrap();
        assert!(n[1].as_f64().unwrap().is_nan());
    }

    // --- CROSS / non-equi ---------------------------------------------------

    #[test]
    fn cross_join_is_cartesian() {
        let mut j = HashJoin::new(
            Mock::script(vec![ints1(&[Some(1), Some(2)])]),
            Mock::script(vec![ints1(&[Some(10), Some(20), Some(30)])]),
            JoinKind::Cross,
            Vec::new(),
            Vec::new(),
            None,
            vec![Ty::Int],
            vec![Ty::Int],
        )
        .unwrap();
        let got = run(&mut j);
        assert_eq!(got.rows.len(), 6);
        assert_eq!(
            norm(&got.rows),
            vec![
                vec![Some(1), Some(10)],
                vec![Some(1), Some(20)],
                vec![Some(1), Some(30)],
                vec![Some(2), Some(10)],
                vec![Some(2), Some(20)],
                vec![Some(2), Some(30)],
            ]
        );
    }

    #[test]
    fn cross_join_with_empty_side_is_empty() {
        let cross = |l: Vec<Step>, r: Vec<Step>| {
            HashJoin::new(
                Mock::script(l),
                Mock::script(r),
                JoinKind::Cross,
                Vec::new(),
                Vec::new(),
                None,
                vec![Ty::Int],
                vec![Ty::Int],
            )
            .unwrap()
        };
        assert!(run(&mut cross(vec![ints1(&[Some(1)])], Vec::new())).rows.is_empty());
        assert!(run(&mut cross(Vec::new(), vec![ints1(&[Some(1)])])).rows.is_empty());
    }

    /// A predicate that does not reduce to equality (`a < b`). It must fall to a nested loop.
    #[test]
    fn non_equi_join_falls_back_to_nested_loop() {
        let mut j = HashJoin::new(
            Mock::script(vec![ints1(&[Some(1), Some(5)])]),
            Mock::script(vec![ints1(&[Some(2), Some(9)])]),
            JoinKind::Inner,
            Vec::new(),
            Vec::new(),
            // Post-join schema: 0 = left, 1 = right
            Some(cmp_prog(0, 1, Ty::Int, OpCode::Lt)),
            vec![Ty::Int],
            vec![Ty::Int],
        )
        .unwrap();
        let got = run(&mut j);
        assert_eq!(
            norm(&got.rows),
            vec![vec![Some(1), Some(2)], vec![Some(1), Some(9)], vec![Some(5), Some(9)]]
        );
    }

    /// Non-equi + LEFT. A left row that passed with no build row is NULL-extended.
    #[test]
    fn non_equi_left_join_keeps_unmatched() {
        let mut j = HashJoin::new(
            Mock::script(vec![ints1(&[Some(1), Some(50)])]),
            Mock::script(vec![ints1(&[Some(2), Some(9)])]),
            JoinKind::Left,
            Vec::new(),
            Vec::new(),
            Some(cmp_prog(0, 1, Ty::Int, OpCode::Lt)),
            vec![Ty::Int],
            vec![Ty::Int],
        )
        .unwrap();
        let got = run(&mut j);
        assert_eq!(
            norm(&got.rows),
            vec![vec![Some(1), Some(2)], vec![Some(1), Some(9)], vec![Some(50), None]]
        );
    }

    // --- residual -----------------------------------------------------------

    /// A pair whose keys matched but that residual rejected is "not matched". Under LEFT that
    /// left row is NULL-extended and emitted (it must not be dropped).
    #[test]
    fn residual_failure_still_yields_null_extended_left_row() {
        // Left (key, v) / right (key, w), residual: left's v < right's w
        let left = vec![ready(vec![ints(&[Some(1), Some(2)]), ints(&[Some(100), Some(0)])])];
        let right = vec![ready(vec![ints(&[Some(1), Some(2)]), ints(&[Some(5), Some(5)])])];
        let ty = vec![Ty::Int, Ty::Int];
        let mut j = HashJoin::new(
            Mock::script(left),
            Mock::script(right),
            JoinKind::Left,
            vec![col_prog(0, Ty::Int)],
            vec![col_prog(0, Ty::Int)],
            // Post-join schema: 0=left key 1=left v 2=right key 3=right w
            Some(cmp_prog(1, 3, Ty::Int, OpCode::Lt)),
            ty.clone(),
            ty,
        )
        .unwrap();
        let got = run(&mut j);
        assert_eq!(
            norm(&got.rows),
            vec![
                vec![Some(1), Some(100), None, None], // 100 < 5 is false -> NULL extension
                vec![Some(2), Some(0), Some(2), Some(5)],
            ]
        );
    }

    /// FULL + residual. A rejected pair counts as unmatched on both sides.
    #[test]
    fn residual_failure_marks_both_sides_unmatched() {
        let left = vec![ready(vec![ints(&[Some(1)]), ints(&[Some(100)])])];
        let right = vec![ready(vec![ints(&[Some(1)]), ints(&[Some(5)])])];
        let ty = vec![Ty::Int, Ty::Int];
        let mut j = HashJoin::new(
            Mock::script(left),
            Mock::script(right),
            JoinKind::Full,
            vec![col_prog(0, Ty::Int)],
            vec![col_prog(0, Ty::Int)],
            Some(cmp_prog(1, 3, Ty::Int, OpCode::Lt)),
            ty.clone(),
            ty,
        )
        .unwrap();
        let got = run(&mut j);
        assert_eq!(
            norm(&got.rows),
            vec![vec![None, None, Some(1), Some(5)], vec![Some(1), Some(100), None, None]]
        );
    }

    // --- Batch boundaries ---------------------------------------------------

    /// One left row produces more output than BATCH_SIZE. Without cutting mid-probe and resuming
    /// from there, rows would be lost.
    #[test]
    fn one_probe_row_spans_multiple_batches() {
        let big: Vec<Option<i32>> = (0..BATCH_SIZE + 500).map(|_| Some(4)).collect();
        let got = run(&mut join1(
            vec![ints1(&[Some(4)])],
            vec![ready(vec![ints(&big)])],
            JoinKind::Inner,
        ));
        assert_eq!(got.rows.len(), BATCH_SIZE + 500);
        assert_eq!(got.batches, 2);
    }

    /// The unmatched-row drain fits into batches too.
    #[test]
    fn unmatched_drain_spans_multiple_batches() {
        let many: Vec<Option<i32>> = (0..BATCH_SIZE as i32 + 10).map(Some).collect();
        let got = run(&mut join1(
            vec![ready(vec![ints(&many)])],
            vec![ints1(&[Some(0)])],
            JoinKind::Left,
        ));
        assert_eq!(got.rows.len(), BATCH_SIZE + 10);

        let got = run(&mut join1(
            vec![ints1(&[Some(0)])],
            vec![ready(vec![ints(&many)])],
            JoinKind::Right,
        ));
        assert_eq!(got.rows.len(), BATCH_SIZE + 10);
        assert!(got.batches >= 2);
    }

    // --- Empty inputs -------------------------------------------------------

    #[test]
    fn empty_build_side_for_each_kind() {
        for (kind, expect) in
            [(JoinKind::Inner, 0), (JoinKind::Left, 2), (JoinKind::Right, 0), (JoinKind::Full, 2)]
        {
            let got = run(&mut join1(vec![ints1(&[Some(1), Some(2)])], Vec::new(), kind));
            assert_eq!(got.rows.len(), expect, "{kind:?}");
            for r in &got.rows {
                assert!(r[1].is_null(), "the right should be NULL-extended");
            }
        }
    }

    #[test]
    fn empty_probe_side_for_each_kind() {
        for (kind, expect) in
            [(JoinKind::Inner, 0), (JoinKind::Left, 0), (JoinKind::Right, 2), (JoinKind::Full, 2)]
        {
            let got = run(&mut join1(Vec::new(), vec![ints1(&[Some(1), Some(2)])], kind));
            assert_eq!(got.rows.len(), expect, "{kind:?}");
            for r in &got.rows {
                assert!(r[0].is_null(), "the left should be NULL-extended");
            }
        }
    }

    #[test]
    fn both_sides_empty() {
        for kind in [JoinKind::Inner, JoinKind::Left, JoinKind::Right, JoinKind::Full] {
            let mut j = join1(Vec::new(), Vec::new(), kind);
            assert!(run(&mut j).rows.is_empty(), "{kind:?}");
        }
    }

    // --- SEMI / ANTI --------------------------------------------------------

    /// The output is the left schema only. Not a single right column is attached.
    #[test]
    fn semi_and_anti_emit_left_columns_only() {
        let left = || vec![ready(vec![ints(&[Some(1), Some(2)]), strs(&[Some("a"), Some("b")])])];
        let right = || vec![ready(vec![ints(&[Some(1)]), ints(&[Some(9)])])];
        let mk = |kind| {
            HashJoin::new(
                Mock::script(left()),
                Mock::script(right()),
                kind,
                vec![col_prog(0, Ty::Int)],
                vec![col_prog(0, Ty::Int)],
                None,
                vec![Ty::Int, Ty::Varchar],
                vec![Ty::Int, Ty::Int],
            )
            .unwrap()
        };
        let got = run(&mut mk(JoinKind::Semi));
        assert_eq!(got.rows.len(), 1);
        assert_eq!(got.rows[0].len(), 2, "just the left's 2 columns");
        assert_eq!(got.rows[0][0].as_i64(), Some(1));
        assert_eq!(got.rows[0][1].as_bytes(), Some(&b"a"[..]));

        let got = run(&mut mk(JoinKind::Anti));
        assert_eq!(got.rows.len(), 1);
        assert_eq!(got.rows[0].len(), 2);
        assert_eq!(got.rows[0][0].as_i64(), Some(2));
    }

    #[test]
    fn semi_and_anti_with_no_match_and_one_match() {
        // Absent from the right / present exactly once.
        let l = || vec![ints1(&[Some(1), Some(2), Some(3)])];
        let r = || vec![ints1(&[Some(2)])];
        assert_eq!(norm(&run(&mut join1(l(), r(), JoinKind::Semi)).rows), vec![vec![Some(2)]]);
        assert_eq!(
            norm(&run(&mut join1(l(), r(), JoinKind::Anti)).rows),
            vec![vec![Some(1)], vec![Some(3)]]
        );

        // With an empty right side, SEMI gives 0 rows and ANTI gives every row.
        assert!(run(&mut join1(l(), Vec::new(), JoinKind::Semi)).rows.is_empty());
        assert_eq!(run(&mut join1(l(), Vec::new(), JoinKind::Anti)).rows.len(), 3);
        // With an empty left side, both give 0 rows.
        assert!(run(&mut join1(Vec::new(), r(), JoinKind::Semi)).rows.is_empty());
        assert!(run(&mut join1(Vec::new(), r(), JoinKind::Anti)).rows.is_empty());
    }

    /// However many matches there are, a left row appears **exactly once**.
    #[test]
    fn semi_emits_a_left_row_at_most_once_with_many_matches() {
        let many: Vec<Option<i32>> = (0..BATCH_SIZE + 500).map(|_| Some(7)).collect();
        let got = run(&mut join1(
            vec![ints1(&[Some(7), Some(7), Some(8)])],
            vec![ready(vec![ints(&many)])],
            JoinKind::Semi,
        ));
        assert_eq!(
            norm(&got.rows),
            vec![vec![Some(7)], vec![Some(7)]],
            "the 2 left rows once each"
        );
        assert_eq!(got.batches, 1);

        // The ANTI side, with the same input, gives "only the unmatched 8".
        let many: Vec<Option<i32>> = (0..BATCH_SIZE + 500).map(|_| Some(7)).collect();
        let got = run(&mut join1(
            vec![ints1(&[Some(7), Some(7), Some(8)])],
            vec![ready(vec![ints(&many)])],
            JoinKind::Anti,
        ));
        assert_eq!(norm(&got.rows), vec![vec![Some(8)]]);
    }

    /// Even when matched left rows exceed BATCH_SIZE, the drain is merely split.
    #[test]
    fn semi_drain_spans_multiple_batches() {
        let many: Vec<Option<i32>> = (0..BATCH_SIZE as i32 + 10).map(Some).collect();
        let got = run(&mut join1(
            vec![ready(vec![ints(&many)])],
            vec![ready(vec![ints(&many)])],
            JoinKind::Semi,
        ));
        assert_eq!(got.rows.len(), BATCH_SIZE + 10);
        assert!(got.batches >= 2);
    }

    /// A left row with a NULL key matches no build row. It disappears under SEMI and survives under ANTI.
    /// `NOT IN`'s "return no rows if the right has a NULL" rule is the binder's responsibility
    /// and is not here.
    #[test]
    fn null_keys_are_dropped_by_semi_and_kept_by_anti() {
        let l = || vec![ints1(&[None, Some(1), Some(2)])];
        let r = || vec![ints1(&[None, Some(1)])];
        assert_eq!(norm(&run(&mut join1(l(), r(), JoinKind::Semi)).rows), vec![vec![Some(1)]]);
        assert_eq!(
            norm(&run(&mut join1(l(), r(), JoinKind::Anti)).rows),
            vec![vec![None], vec![Some(2)]],
            "ANTI plainly returns the unmatched rows even with a NULL on the right"
        );
    }

    /// If residual rejects the only match, SEMI does not emit that left row and ANTI does.
    #[test]
    fn residual_that_kills_the_only_match_flips_semi_and_anti() {
        let mk = |kind| {
            // Left (key, v) / right (key, w), residual: left's v < right's w
            HashJoin::new(
                Mock::script(vec![ready(vec![
                    ints(&[Some(1), Some(2)]),
                    ints(&[Some(100), Some(0)]),
                ])]),
                Mock::script(vec![ready(vec![
                    ints(&[Some(1), Some(2)]),
                    ints(&[Some(5), Some(5)]),
                ])]),
                kind,
                vec![col_prog(0, Ty::Int)],
                vec![col_prog(0, Ty::Int)],
                // Post-join schema: 0=left key 1=left v 2=right key 3=right w
                Some(cmp_prog(1, 3, Ty::Int, OpCode::Lt)),
                vec![Ty::Int, Ty::Int],
                vec![Ty::Int, Ty::Int],
            )
            .unwrap()
        };
        // key=1 has no match since 100 < 5 is false; key=2 matches with 0 < 5.
        assert_eq!(
            norm(&run(&mut mk(JoinKind::Semi)).rows),
            vec![vec![Some(2), Some(0)]],
            "a left row rejected by residual disappears from SEMI"
        );
        assert_eq!(
            norm(&run(&mut mk(JoinKind::Anti)).rows),
            vec![vec![Some(1), Some(100)]],
            "the same row survives under ANTI"
        );
    }

    /// SEMI without equality keys (the rewrite target of a correlated EXISTS). The match bit is
    /// set the same way with a nested loop plus residual.
    #[test]
    fn non_equi_semi_and_anti() {
        let mk = |kind| {
            HashJoin::new(
                Mock::script(vec![ints1(&[Some(1), Some(50)])]),
                Mock::script(vec![ints1(&[Some(2), Some(9)])]),
                kind,
                Vec::new(),
                Vec::new(),
                Some(cmp_prog(0, 1, Ty::Int, OpCode::Lt)),
                vec![Ty::Int],
                vec![Ty::Int],
            )
            .unwrap()
        };
        assert_eq!(norm(&run(&mut mk(JoinKind::Semi)).rows), vec![vec![Some(1)]]);
        assert_eq!(norm(&run(&mut mk(JoinKind::Anti)).rows), vec![vec![Some(50)]]);
    }

    /// Interruption passes straight through for SEMI / ANTI too, and the result matches the uninterrupted run.
    #[test]
    fn semi_and_anti_survive_need_io_and_need_codec() {
        let l = || vec![ints1(&[Some(1), Some(2)]), ints1(&[Some(3), Some(1)])];
        let r = || vec![ints1(&[Some(1), Some(3)]), ints1(&[Some(1), Some(9)])];
        for kind in [JoinKind::Semi, JoinKind::Anti] {
            let clean = run(&mut join1(l(), r(), kind));
            let noisy = run(&mut join1(
                vec![
                    Step::NeedIo,
                    ints1(&[Some(1), Some(2)]),
                    Step::NeedCodec,
                    ints1(&[Some(3), Some(1)]),
                    Step::NeedIo,
                ],
                vec![
                    ints1(&[Some(1), Some(3)]),
                    Step::NeedCodec,
                    Step::NeedIo,
                    ints1(&[Some(1), Some(9)]),
                ],
                kind,
            ));
            assert!(noisy.interrupts >= 5, "the interruptions were not passed through");
            assert_eq!(norm(&noisy.rows), norm(&clean.rows), "{kind:?}");
        }
        // The contents too. Left 1,2,3,1 against right 1,3,1,9.
        assert_eq!(
            norm(&run(&mut join1(l(), r(), JoinKind::Semi)).rows),
            vec![vec![Some(1)], vec![Some(1)], vec![Some(3)]]
        );
        assert_eq!(norm(&run(&mut join1(l(), r(), JoinKind::Anti)).rows), vec![vec![Some(2)]]);
    }

    // --- ANTI NULL AWARE (`NOT IN (SELECT ...)`) ----------------------------
    //
    // Every expectation was confirmed with DuckDB:
    //   NULL NOT IN (empty)      -> true    (the row survives)
    //   x    NOT IN (empty)      -> true
    //   NULL NOT IN (non-empty)  -> UNKNOWN (the row disappears)
    //   x    NOT IN (...NULL...) -> UNKNOWN (every row disappears)

    /// With a NULL among the right keys, the result is empty even with unmatched left rows.
    #[test]
    fn null_aware_anti_is_empty_when_build_side_has_a_null_key() {
        let got = run(&mut join1(
            vec![ints1(&[Some(1), Some(2), Some(3)])],
            vec![ints1(&[Some(1), None])],
            JoinKind::AntiNullAware,
        ));
        assert!(
            got.rows.is_empty(),
            "2 and 3 do not match but are UNKNOWN, so they are not emitted"
        );
        // Plain ANTI returns 2 rows for the same input (the difference shows here).
        let plain = run(&mut join1(
            vec![ints1(&[Some(1), Some(2), Some(3)])],
            vec![ints1(&[Some(1), None])],
            JoinKind::Anti,
        ));
        assert_eq!(norm(&plain.rows), vec![vec![Some(2)], vec![Some(3)]]);
    }

    /// With no NULL on the right it behaves like plain ANTI.
    #[test]
    fn null_aware_anti_matches_plain_anti_without_nulls() {
        let l = || vec![ints1(&[Some(1), Some(2)]), ints1(&[Some(3), Some(1)])];
        let r = || vec![ints1(&[Some(1), Some(9)])];
        let plain = run(&mut join1(l(), r(), JoinKind::Anti));
        let aware = run(&mut join1(l(), r(), JoinKind::AntiNullAware));
        assert_eq!(norm(&aware.rows), norm(&plain.rows));
        assert_eq!(norm(&aware.rows), vec![vec![Some(2)], vec![Some(3)]]);
    }

    /// A left row with a NULL key drops unless the right side is empty.
    #[test]
    fn null_aware_anti_drops_left_rows_with_null_keys() {
        let got = run(&mut join1(
            vec![ints1(&[None, Some(2), Some(1)])],
            vec![ints1(&[Some(1)])],
            JoinKind::AntiNullAware,
        ));
        assert_eq!(norm(&got.rows), vec![vec![Some(2)]], "the NULL left row is UNKNOWN");
        // Plain ANTI keeps the NULL left row.
        let plain = run(&mut join1(
            vec![ints1(&[None, Some(2), Some(1)])],
            vec![ints1(&[Some(1)])],
            JoinKind::Anti,
        ));
        assert_eq!(norm(&plain.rows), vec![vec![None], vec![Some(2)]]);
    }

    /// With an empty right side, `NOT IN ()` is always true. Every row is returned, including the left NULL row.
    #[test]
    fn null_aware_anti_with_empty_build_side_emits_every_left_row() {
        let got =
            run(&mut join1(vec![ints1(&[None, Some(2)])], Vec::new(), JoinKind::AntiNullAware));
        assert_eq!(norm(&got.rows), vec![vec![None], vec![Some(2)]]);
        // "The right is empty" also covers the case where only 0-row batches arrive.
        let got = run(&mut join1(
            vec![ints1(&[None, Some(2)])],
            vec![ints1(&[])],
            JoinKind::AntiNullAware,
        ));
        assert_eq!(got.rows.len(), 2);
    }

    /// With a composite key, a NULL in either column still counts as "the key contains a NULL".
    #[test]
    fn null_aware_anti_looks_at_every_key_column() {
        let left = vec![ready(vec![ints(&[Some(1), Some(2)]), ints(&[Some(10), Some(20)])])];
        let right = vec![ready(vec![ints(&[Some(9)]), ints(&[None])])];
        let ty = vec![Ty::Int, Ty::Int];
        let got = run(&mut join(left, right, JoinKind::AntiNullAware, 2, ty.clone(), ty));
        assert!(got.rows.is_empty(), "empty, since the right keys contain a NULL");
    }

    /// Short-circuiting still reaches `Done` (it does not hang). Not a single left row is pulled.
    #[test]
    fn null_aware_anti_short_circuit_terminates_without_reading_the_left() {
        let mut j = HashJoin::new(
            // The script is padded with Ready so that pulling the left would be noticeable.
            Mock::script(vec![ints1(&[Some(1)]), ints1(&[Some(2)])]),
            Mock::script(vec![ints1(&[None])]),
            JoinKind::AntiNullAware,
            vec![col_prog(0, Ty::Int)],
            vec![col_prog(0, Ty::Int)],
            None,
            vec![Ty::Int],
            vec![Ty::Int],
        )
        .unwrap();
        let mut cat = Catalog::new();
        let mut vm = Vm::new();
        for _ in 0..3 {
            let mut ctx =
                ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
            assert!(matches!(j.next(&mut ctx).unwrap(), Step::Done));
        }
    }

    /// The decision does not change with an interruption interposed.
    #[test]
    fn null_aware_anti_survives_interrupts() {
        let l = || vec![ints1(&[Some(1), Some(2)]), ints1(&[None, Some(3)])];
        let r = || vec![ints1(&[Some(1)]), ints1(&[Some(9)])];
        let clean = run(&mut join1(l(), r(), JoinKind::AntiNullAware));
        let noisy = run(&mut join1(
            vec![
                Step::NeedIo,
                ints1(&[Some(1), Some(2)]),
                Step::NeedCodec,
                ints1(&[None, Some(3)]),
            ],
            vec![ints1(&[Some(1)]), Step::NeedCodec, Step::NeedIo, ints1(&[Some(9)])],
            JoinKind::AntiNullAware,
        ));
        assert!(noisy.interrupts >= 4);
        assert_eq!(norm(&noisy.rows), norm(&clean.rows));
        assert_eq!(norm(&clean.rows), vec![vec![Some(2)], vec![Some(3)]]);

        // It must short-circuit when the right's NULL arrives in the second batch too.
        let got = run(&mut join1(
            vec![Step::NeedIo, ints1(&[Some(2)])],
            vec![ints1(&[Some(1)]), Step::NeedIo, ints1(&[None])],
            JoinKind::AntiNullAware,
        ));
        assert!(got.rows.is_empty());
    }

    /// If residual rejects the only match, the left row survives under NULL-aware ANTI as well.
    #[test]
    fn null_aware_anti_respects_the_residual() {
        let mut j = HashJoin::new(
            Mock::script(vec![ready(vec![ints(&[Some(1)]), ints(&[Some(100)])])]),
            Mock::script(vec![ready(vec![ints(&[Some(1)]), ints(&[Some(5)])])]),
            JoinKind::AntiNullAware,
            vec![col_prog(0, Ty::Int)],
            vec![col_prog(0, Ty::Int)],
            Some(cmp_prog(1, 3, Ty::Int, OpCode::Lt)),
            vec![Ty::Int, Ty::Int],
            vec![Ty::Int, Ty::Int],
        )
        .unwrap();
        let got = run(&mut j);
        assert_eq!(norm(&got.rows), vec![vec![Some(1), Some(100)]]);
    }

    #[test]
    fn mismatched_key_counts_are_rejected() {
        let e = HashJoin::new(
            Mock::empty(),
            Mock::empty(),
            JoinKind::Inner,
            vec![col_prog(0, Ty::Int)],
            Vec::new(),
            None,
            vec![Ty::Int],
            vec![Ty::Int],
        );
        assert_eq!(crate::error::code_of(e.map(|_| ())), Some(Code::Internal));
    }
}
