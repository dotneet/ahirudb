//! The fixed-point iteration of `WITH RECURSIVE` (recursive CTEs).
//!
//! ## The algorithm
//!
//! 1. Read the anchor (the left side of `UNION`) to completion and make all its rows the first
//!    "working table" (the previous iteration's new rows). They are also returned to the caller
//!    while being read (the anchor's result is part of the final result too).
//! 2. While the working table is non-empty, run the recursive term (the right side of `UNION`)
//!    once with it as input. Self-references inside the recursive term appear as the leaf node
//!    `Node::WorkingTable`, and each run rebuilds the physical operator tree with "this round's
//!    working table" plugged into `WorkingTableScan`
//!    (see `super::build_ctx`).
//! 3. Of the rows the recursive term produced, only those not yet emitted (under `UNION`) count
//!    as "new rows"; they become the next iteration's working table and are also returned to the
//!    caller. It ends once there are 0 new rows.
//!
//! ## Deduplication
//!
//! `UNION ALL` keeps duplicates and so holds no state beyond building the working table.
//! `UNION` (DISTINCT) uses the same `rowkey::encode_key` + `HashIndex` as
//! `exec::setop`/`exec::agg` and keeps exactly one set of seen keys spanning the anchor through
//! the final iteration (this engine's consistent policy of not rebuilding key encoding per
//! operator).
//!
//! ## Resumability
//!
//! The anchor and each iteration's recursive term can both return `Step::NeedIo`/`NeedCodec`.
//! Interruptions are returned straight up, and all the partial state -- `phase`/`working`/`seen`
//! and the rest -- lives in `self`, so the next `next()` resumes from the same place (the same
//! style as `exec::setop::SetOp`). One iteration's physical operator tree is held in
//! `self.current` and is not rebuilt except at an iteration boundary (swapping the working table).
//!
//! ## Safety valves
//!
//! To reliably turn a non-terminating recursive CTE (a `WHERE` that does not shrink, or a
//! forgotten stopping condition) into an error in finite time and finite memory, both the
//! iteration count and the working table's bytes per iteration are capped (see the comments on
//! the constants below).

use crate::exec::rowkey::{encode_key, HashIndex};
use crate::exec::sort::vector_bytes;
use crate::exec::{build_ctx, ExecContext, Operator, Step};
use crate::plan::Node;
use crate::prelude::*;
use crate::vector::{Batch, Vector};

/// The cap on fixed-point iterations.
///
/// DuckDB itself was confirmed on real hardware to spin on this kind of input (a recursive term
/// such as `SELECT n+1 FROM t` with no stopping condition) indefinitely, not stopping until
/// memory is exhausted (the `duckdb` CLI had not finished after 120 seconds).
/// Doing that on a wasm host would be unrecoverable, so 100,000 was chosen as a value where
/// realistic hierarchical data (org charts and category trees thousands to tens of thousands
/// deep) and graph traversals fit comfortably, while a runaway falls into a clear error in seconds.
const MAX_RECURSIVE_ITERATIONS: u32 = 100_000;

/// The approximate byte cap on what one iteration's working table (the previous iteration's new
/// rows) may use.
///
/// A second safety valve for detecting, without relying on the iteration cap, the case where
/// rows keep growing (and never shrink) each round. The same idea as
/// `exec::sort::Sort`/`exec::setop::SetOp`; no exact byte accounting is done.
const MAX_WORKING_BYTES: usize = 256 << 20;

/// The approximate byte cap on what `UNION`'s (deduplicating) seen-key set may use.
/// The same level as `exec::setop::SetOp`/`exec::mod::DistinctOn`.
const MAX_SEEN_BYTES: usize = 64 << 20;

enum Phase {
    /// Reading the anchor.
    Anchor,
    /// Reading the current iteration's recursive term (`current`).
    Iterate,
    Done,
}

pub struct RecursiveCte {
    anchor: Box<dyn Operator>,
    /// The recursive term's logical plan. A new physical operator tree is rebuilt each iteration,
    /// so it is kept owned even after execution (`Node: Clone`).
    recursive_term: Node,
    phase: Phase,
    /// `Some` only in the `Iterate` phase.
    current: Option<Box<dyn Operator>>,
    /// The rows newly added in the previous iteration. Plugged into the next iteration's
    /// `Node::WorkingTable`.
    working: Vec<Batch>,
    /// The rows newly found in this iteration (or the anchor).
    /// Once the phase ends it replaces `working`.
    next_working: Vec<Batch>,
    /// The approximate bytes `next_working` uses.
    next_working_bytes: usize,
    /// `Some` only under `UNION` (DISTINCT). `UNION ALL` does not look at duplicates.
    seen: Option<HashIndex>,
    keybuf: Vec<u8>,
    /// The number of completed iterations (a safety valve).
    iterations: u32,
}

impl RecursiveCte {
    pub fn new(anchor: Box<dyn Operator>, recursive_term: Node, union_all: bool) -> Self {
        RecursiveCte {
            anchor,
            recursive_term,
            phase: Phase::Anchor,
            current: None,
            working: Vec::new(),
            next_working: Vec::new(),
            next_working_bytes: 0,
            seen: if union_all { None } else { Some(HashIndex::new()) },
            keybuf: Vec::new(),
            iterations: 0,
        }
    }

    /// Processes one batch. Under `UNION` it removes duplicate rows and also pushes the survivors
    /// into "the next iteration's working table". The return value is the output for the caller
    /// (`None` for a batch of nothing but duplicates, or 0 rows).
    fn process(&mut self, mut batch: Batch) -> Result<Option<Batch>> {
        if batch.card() == 0 {
            return Ok(None);
        }
        // From here on lookups are by row number (both for duplicate checking and for storing into
        // the working table), so selection is materialized now.
        batch.materialize();
        let input_rows = batch.num_rows();
        let mut output_rows = input_rows;
        let cols = match &mut self.seen {
            None => batch.cols,
            Some(seen) => {
                let rows = batch.num_rows();
                let refs: Vec<&Vector> = batch.cols.iter().collect();
                let mut sel = Vec::with_capacity(rows);
                let mut keybuf = core::mem::take(&mut self.keybuf);
                for r in 0..rows {
                    encode_key(&refs, r, &mut keybuf);
                    if seen.get_or_insert(&keybuf).1 {
                        sel.push(r as u32);
                    }
                }
                self.keybuf = keybuf;
                ensure!(seen.approx_bytes() <= MAX_SEEN_BYTES, Oom);
                if sel.is_empty() {
                    return Ok(None);
                }
                output_rows = sel.len();
                if sel.len() == rows {
                    batch.cols
                } else {
                    batch.cols.iter().map(|c| c.gather(&sel)).collect()
                }
            }
        };

        let bytes: usize = cols.iter().map(vector_bytes).sum();
        self.next_working_bytes = self.next_working_bytes.saturating_add(bytes);
        ensure!(self.next_working_bytes <= MAX_WORKING_BYTES, Oom);

        // `Batch::new(Vec::new())` cannot carry a row count. Preserve it explicitly for
        // zero-column recursive relations (for example, a table after its only column was
        // dropped); otherwise UNION ALL would turn every rows-only batch into an empty batch.
        let out = if cols.is_empty() { Batch::rows_only(output_rows) } else { Batch::new(cols) };
        self.next_working.push(clone_batch(&out));
        Ok(Some(out))
    }

    /// The current phase's input is read through. Advances to the next iteration
    /// (`Done` if there are no new rows).
    fn begin_iteration(&mut self) -> Result<()> {
        self.working = core::mem::take(&mut self.next_working);
        self.next_working_bytes = 0;
        self.current = None;
        if self.working.is_empty() {
            self.phase = Phase::Done;
            return Ok(());
        }
        self.iterations += 1;
        ensure!(self.iterations <= MAX_RECURSIVE_ITERATIONS, RecursionLimitExceeded);
        self.current = Some(build_ctx(self.recursive_term.clone(), Some(&self.working))?);
        self.phase = Phase::Iterate;
        Ok(())
    }
}

impl Operator for RecursiveCte {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        loop {
            match self.phase {
                Phase::Anchor => match self.anchor.next(ctx)? {
                    Step::Ready(b) => {
                        if let Some(out) = self.process(b)? {
                            return Ok(Step::Ready(out));
                        }
                    }
                    Step::NeedIo => return Ok(Step::NeedIo),
                    Step::NeedCodec => return Ok(Step::NeedCodec),
                    Step::Done => self.begin_iteration()?,
                },
                Phase::Iterate => {
                    let op = match &mut self.current {
                        Some(op) => op,
                        // `Iterate` is entered only right after `begin_iteration` sets `current`,
                        // so it is always `Some`.
                        None => err!(Internal),
                    };
                    match op.next(ctx)? {
                        Step::Ready(b) => {
                            if let Some(out) = self.process(b)? {
                                return Ok(Step::Ready(out));
                            }
                        }
                        Step::NeedIo => return Ok(Step::NeedIo),
                        Step::NeedCodec => return Ok(Step::NeedCodec),
                        Step::Done => self.begin_iteration()?,
                    }
                }
                Phase::Done => return Ok(Step::Done),
            }
        }
    }
}

/// A self-reference inside a recursive CTE's recursive term (`Node::WorkingTable`). A leaf
/// operator that simply returns the rows newly added in the previous iteration.
/// The data is already in memory, so for the same reason as `MemScan` it can never in principle
/// return `NeedIo`/`NeedCodec`.
pub struct WorkingTableScan {
    batches: Vec<Batch>,
    pos: usize,
}

impl WorkingTableScan {
    pub fn new(batches: Vec<Batch>) -> Self {
        WorkingTableScan { batches, pos: 0 }
    }
}

impl Operator for WorkingTableScan {
    fn next(&mut self, _ctx: &mut ExecContext) -> Result<Step> {
        if self.pos >= self.batches.len() {
            return Ok(Step::Done);
        }
        let b = core::mem::replace(&mut self.batches[self.pos], Batch::new(Vec::new()));
        self.pos += 1;
        Ok(Step::Ready(b))
    }
}

/// Clones the working table. Even when `Node::WorkingTable` appears in several places (a
/// self-join), a fresh `Vec<Batch>` is built per reference so each can advance independently.
pub(crate) fn clone_batches(src: &[Batch]) -> Vec<Batch> {
    src.iter().map(clone_batch).collect()
}

/// `Batch` offers no way to clone `sel`/`empty_rows` from outside (they are private fields of
/// `vector::Batch`), so only the columns are cloned, assuming no selection is present. Every
/// caller passes only batches that have already been `materialize()`d.
fn clone_batch(b: &Batch) -> Batch {
    if b.cols.is_empty() {
        Batch::rows_only(b.num_rows())
    } else {
        Batch::new(b.cols.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_preserves_rows_only_batches() {
        let mut op = RecursiveCte::new(
            Box::new(crate::exec::Values::new(Batch::rows_only(3))),
            Node::WorkingTable { schema: Vec::new() },
            true,
        );
        let out = op.process(Batch::rows_only(3)).unwrap().unwrap();
        assert!(out.cols.is_empty());
        assert_eq!(out.num_rows(), 3);
        assert_eq!(op.next_working[0].num_rows(), 3);
    }
}
