//! `USING SAMPLE` / `TABLESAMPLE`.
//!
//! A percentage is implemented with the Bernoulli method (each row is kept independently with
//! probability `p`; a streaming operator shaped like `exec::Filter` that does not block upstream).
//! A row count is a blocking scheme that reads the input through and then picks N rows uniformly
//! at random (the same three phases as `exec::sort::Sort`: buffer -> settle -> return in
//! `BATCH_SIZE` slices). The method names `BERNOULLI`/`SYSTEM`/`RESERVOIR` are accepted
//! syntactically but make no difference in the implementation (the simplification the task
//! prescribed: percentage > row count > distinguishing methods).
//!
//! ## The random number generator
//! xorshift64* is implemented in-house with no dependency (`no_std` rules out the `rand` crate).
//! The seed is initialized deterministically from `plan::SampleSpec::seed` (given explicitly by
//! `USING SAMPLE ... (method, seed)`, or `plan::DEFAULT_SAMPLE_SEED` otherwise), so running the
//! same query any number of times picks the same rows.
//!
//! ## Why reproducibility survives a `NeedIo`/`NeedCodec`
//! - Bernoulli (`Bernoulli`): exactly one random draw per row. Interruptions happen on the input
//!   side and merely pass straight through (as in `Filter`), so the PRNG's call sequence is
//!   determined solely by "the sequence of rows actually evaluated" and does not depend on where
//!   an interruption occurred.
//! - Row count (`RowSample`): buffering continues in the `Buffering` phase, and only once the
//!   input reaches `Done` is a subset chosen randomly. Interruptions can only happen before that
//!   stage (as in `Sort`), so the selection itself is unaffected by whether one occurred.

use crate::exec::sort::vector_bytes;
use crate::exec::{ExecContext, Operator, Step};
use crate::plan::SampleSpec;
use crate::prelude::*;
use crate::vector::{Batch, Ty, Vector, BATCH_SIZE};

/// xorshift64*. A deterministic PRNG usable in a `no_std` environment with no dependencies.
/// Cryptographic strength is unnecessary (this is only for sampling).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // 0 is a fixed point (`0 ^ ... = 0`), so an offset is added.
        Rng(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A uniform random number in `[0, 1)`. The top 53 bits are used as the mantissa.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// A uniform index in `[0, n)`. Assumes it is never called with `n == 0`.
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

// --- Percentage (the Bernoulli method) ---------------------------------------

pub struct Bernoulli {
    input: Box<dyn Operator>,
    /// The probability of keeping a row (0.0..=1.0).
    p: f64,
    rng: Rng,
}

impl Bernoulli {
    pub fn new(input: Box<dyn Operator>, spec: &SampleSpec) -> Self {
        let p = (spec.amount / 100.0).clamp(0.0, 1.0);
        Bernoulli { input, p, rng: Rng::new(spec.seed) }
    }
}

impl Operator for Bernoulli {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        loop {
            let mut batch = match self.input.next(ctx)? {
                Step::Ready(b) => b,
                other => return Ok(other),
            };
            let rows = batch.card();
            let mut sel = Vec::with_capacity(rows);
            for row in 0..rows {
                if self.rng.next_f64() < self.p {
                    let phys = match &batch.sel {
                        Some(s) => s[row],
                        None => row as u32,
                    };
                    sel.push(phys);
                }
            }
            if sel.is_empty() {
                // A batch whose rows are all dropped is not returned upstream; the next is pulled (the same discipline as `Filter`).
                continue;
            }
            batch.sel = Some(sel);
            return Ok(Step::Ready(batch));
        }
    }
}

// --- Row count (N rows uniformly at random) -----------------------------------

/// With no overflow handling, exceeding this returns `Oom` (the same cap and the same reason as
/// `exec::sort::Sort`: it is held down on its own, given wasm's linear memory and the coexistence
/// of the other buffers held).
const MAX_BUFFER_BYTES: usize = 256 * 1024 * 1024;

/// One buffered input batch.
struct Buffered {
    cols: Vec<Vector>,
    rows: usize,
}

enum Phase {
    /// Reading and buffering the input. It stays in this state across interruptions.
    Buffering,
    /// Returning the selected rows in `BATCH_SIZE` slices.
    Emitting,
    Done,
}

pub struct RowSample {
    input: Box<dyn Operator>,
    /// How many rows to keep.
    target: u64,
    rng: Rng,
    phase: Phase,
    batches: Vec<Buffered>,
    total_rows: u64,
    buffered_bytes: usize,
    /// Valid only from `Emitting` on. Holds the selected rows columnar.
    out: Vec<Vector>,
    out_rows: usize,
    pos: usize,
}

impl RowSample {
    pub fn new(input: Box<dyn Operator>, spec: &SampleSpec) -> Self {
        // Fractions are rounded to nearest (`duckdb`'s `12.5 ROWS` gives 12 in some examples and
        // rounds in others, suggesting it is implementation-defined, so plain round-to-nearest is
        // fixed here). A negative value cannot occur syntactically (`sql::parser` rejects it).
        let target = (spec.amount.max(0.0) + 0.5) as u64;
        RowSample {
            input,
            target,
            rng: Rng::new(spec.seed),
            phase: Phase::Buffering,
            batches: Vec::new(),
            total_rows: 0,
            buffered_bytes: 0,
            out: Vec::new(),
            out_rows: 0,
            pos: 0,
        }
    }

    fn absorb(&mut self, mut batch: Batch) -> Result<()> {
        // Selection is resolved first, so `value_at` can be used later with plain row numbers.
        batch.materialize();
        let rows = batch.card();
        if rows == 0 {
            return Ok(());
        }
        self.total_rows = self.total_rows.saturating_add(rows as u64);
        let bytes: usize = batch.cols.iter().map(vector_bytes).sum();
        self.buffered_bytes = self.buffered_bytes.saturating_add(bytes);
        ensure!(self.buffered_bytes <= MAX_BUFFER_BYTES, Oom);
        self.batches.push(Buffered { cols: batch.cols, rows });
        Ok(())
    }

    /// The input is read through. Picks `k` uniformly at random from `0..total_rows` and moves the
    /// selected rows into the output vectors (preserving the input's relative order).
    fn finish(&mut self) {
        let n = self.total_rows;
        let k = self.target.min(n);
        // A partial Fisher-Yates: picks the first `k` uniformly at random.
        let mut idx: Vec<u64> = (0..n).collect();
        for i in 0..k {
            let j = i + self.rng.below(n - i);
            idx.swap(i as usize, j as usize);
        }
        idx.truncate(k as usize);
        idx.sort_unstable();

        let template: Vec<Ty> = self.batches[0].cols.iter().map(|c| c.ty()).collect();
        let mut out: Vec<Vector> =
            template.iter().map(|&ty| Vector::with_capacity(ty, k as usize)).collect();

        let mut offset: u64 = 0;
        let mut ii = 0usize;
        for b in &self.batches {
            let blen = b.rows as u64;
            while ii < idx.len() && idx[ii] < offset + blen {
                let local = (idx[ii] - offset) as usize;
                for (c, col) in out.iter_mut().enumerate() {
                    col.push_value(&b.cols[c].value_at(local));
                }
                ii += 1;
            }
            offset += blen;
        }

        self.batches = Vec::new();
        self.out = out;
        self.out_rows = k as usize;
        self.pos = 0;
        self.phase = Phase::Emitting;
    }

    fn emit(&mut self) -> Result<Step> {
        if self.pos >= self.out_rows {
            self.phase = Phase::Done;
            self.out = Vec::new();
            return Ok(Step::Done);
        }
        let end = (self.pos + BATCH_SIZE).min(self.out_rows);
        let out = if self.out.is_empty() {
            Batch::rows_only(end - self.pos)
        } else {
            let idx: Vec<u32> = (self.pos as u32..end as u32).collect();
            Batch::new(self.out.iter().map(|c| c.gather(&idx)).collect())
        };
        self.pos = end;
        Ok(Step::Ready(out))
    }
}

impl Operator for RowSample {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        loop {
            match self.phase {
                Phase::Buffering => match self.input.next(ctx)? {
                    Step::Ready(b) => self.absorb(b)?,
                    // The interruption is returned straight up. The buffered rows stay in `self`,
                    // so the next call pulls the input again from here.
                    Step::NeedIo => return Ok(Step::NeedIo),
                    Step::NeedCodec => return Ok(Step::NeedCodec),
                    Step::Done => {
                        if self.batches.is_empty() {
                            self.phase = Phase::Done;
                            return Ok(Step::Done);
                        }
                        self.finish();
                    }
                },
                Phase::Emitting => return self.emit(),
                Phase::Done => return Ok(Step::Done),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Catalog;
    use crate::expr::vm::Vm;
    use crate::vector::Value;

    fn ints(vals: &[i32]) -> Vector {
        let mut v = Vector::new(Ty::Int);
        for &x in vals {
            v.push_value(&Value::I32(x));
        }
        v
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

    fn drive(mut op: Box<dyn Operator>) -> Vec<i32> {
        let mut cat = Catalog::new();
        let mut vm = Vm::new();
        let mut out = Vec::new();
        for guard in 0..10_000 {
            assert!(guard < 9_999, "does not terminate");
            let mut ctx =
                ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
            match op.next(&mut ctx).unwrap() {
                Step::Ready(mut b) => {
                    b.materialize();
                    for r in 0..b.num_rows() {
                        let Value::I32(v) = b.cols[0].value_at(r) else { panic!("expected I32") };
                        out.push(v);
                    }
                }
                Step::NeedIo | Step::NeedCodec => {}
                Step::Done => break,
            }
        }
        out
    }

    fn spec(is_rows: bool, amount: f64, seed: u64) -> SampleSpec {
        SampleSpec { is_rows, amount, seed }
    }

    // --- The Bernoulli method (percentage) ----------------------------------

    #[test]
    fn zero_percent_keeps_nothing() {
        let steps = vec![Script::Rows(vec![ints(&(0..1000).collect::<Vec<_>>())])];
        let op = Box::new(Bernoulli::new(Box::new(Mock { steps, pos: 0 }), &spec(false, 0.0, 1)));
        assert!(drive(op).is_empty());
    }

    #[test]
    fn hundred_percent_keeps_everything() {
        let vals: Vec<i32> = (0..1000).collect();
        let steps = vec![Script::Rows(vec![ints(&vals)])];
        let op = Box::new(Bernoulli::new(Box::new(Mock { steps, pos: 0 }), &spec(false, 100.0, 1)));
        assert_eq!(drive(op), vals);
    }

    #[test]
    fn roughly_the_requested_fraction_survives() {
        let vals: Vec<i32> = (0..100_000).collect();
        let steps = vec![Script::Rows(vec![ints(&vals)])];
        let op = Box::new(Bernoulli::new(Box::new(Mock { steps, pos: 0 }), &spec(false, 10.0, 7)));
        let got = drive(op);
        let frac = got.len() as f64 / vals.len() as f64;
        assert!((0.08..0.12).contains(&frac), "got fraction {frac}");
        // The selected rows keep the input's relative order.
        let mut sorted = got.clone();
        sorted.sort_unstable();
        assert_eq!(got, sorted);
    }

    #[test]
    fn same_seed_reproduces_the_same_rows() {
        let vals: Vec<i32> = (0..500).collect();
        let mk = || {
            Box::new(Bernoulli::new(
                Box::new(Mock { steps: vec![Script::Rows(vec![ints(&vals)])], pos: 0 }),
                &spec(false, 30.0, 42),
            )) as Box<dyn Operator>
        };
        assert_eq!(drive(mk()), drive(mk()));
    }

    #[test]
    fn different_seed_gives_a_different_sample() {
        let vals: Vec<i32> = (0..500).collect();
        let mk = |seed: u64| {
            Box::new(Bernoulli::new(
                Box::new(Mock { steps: vec![Script::Rows(vec![ints(&vals)])], pos: 0 }),
                &spec(false, 30.0, seed),
            )) as Box<dyn Operator>
        };
        assert_ne!(drive(mk(1)), drive(mk(2)));
    }

    #[test]
    fn need_io_between_batches_does_not_change_the_result() {
        let make = |interrupt: bool| {
            let mut steps = vec![Script::Rows(vec![ints(&(0..500).collect::<Vec<_>>())])];
            if interrupt {
                steps.push(Script::NeedIo);
            }
            steps.push(Script::Rows(vec![ints(&(500..1000).collect::<Vec<_>>())]));
            steps
        };
        let mk = |interrupt: bool| {
            Box::new(Bernoulli::new(
                Box::new(Mock { steps: make(interrupt), pos: 0 }),
                &spec(false, 25.0, 99),
            )) as Box<dyn Operator>
        };
        assert_eq!(drive(mk(false)), drive(mk(true)), "the result must not change across a NeedIo");
    }

    // --- Row count (N rows uniformly at random) -----------------------------

    #[test]
    fn selects_exactly_the_requested_row_count() {
        let vals: Vec<i32> = (0..1000).collect();
        let steps = vec![Script::Rows(vec![ints(&vals)])];
        let op = Box::new(RowSample::new(Box::new(Mock { steps, pos: 0 }), &spec(true, 100.0, 7)));
        let got = drive(op);
        assert_eq!(got.len(), 100);
        // No duplicates, only values present in the input, and in input order.
        let mut sorted = got.clone();
        sorted.dedup();
        assert_eq!(got, sorted, "there must be no duplicates");
        assert!(got.iter().all(|v| vals.contains(v)));
        let mut asc = got.clone();
        asc.sort_unstable();
        assert_eq!(got, asc, "the selected rows keep the input's relative order");
    }

    #[test]
    fn requesting_more_rows_than_available_returns_everything() {
        let vals: Vec<i32> = (0..10).collect();
        let steps = vec![Script::Rows(vec![ints(&vals)])];
        let op = Box::new(RowSample::new(Box::new(Mock { steps, pos: 0 }), &spec(true, 1000.0, 1)));
        assert_eq!(drive(op), vals);
    }

    #[test]
    fn zero_rows_requested_yields_nothing() {
        let vals: Vec<i32> = (0..10).collect();
        let steps = vec![Script::Rows(vec![ints(&vals)])];
        let op = Box::new(RowSample::new(Box::new(Mock { steps, pos: 0 }), &spec(true, 0.0, 1)));
        assert!(drive(op).is_empty());
    }

    #[test]
    fn empty_input_yields_nothing() {
        let op = Box::new(RowSample::new(
            Box::new(Mock { steps: Vec::new(), pos: 0 }),
            &spec(true, 5.0, 1),
        ));
        assert!(drive(op).is_empty());
    }

    #[test]
    fn same_seed_reproduces_the_same_rows_across_batches() {
        let mk = |interrupt: bool| {
            let mut steps = vec![Script::Rows(vec![ints(&(0..500).collect::<Vec<_>>())])];
            if interrupt {
                steps.push(Script::NeedIo);
            }
            steps.push(Script::Rows(vec![ints(&(500..1000).collect::<Vec<_>>())]));
            Box::new(RowSample::new(Box::new(Mock { steps, pos: 0 }), &spec(true, 50.0, 123)))
                as Box<dyn Operator>
        };
        assert_eq!(drive(mk(false)), drive(mk(true)), "the result must not change across a NeedIo");
    }

    #[test]
    fn spans_multiple_output_batches() {
        let n = BATCH_SIZE * 2 + 300;
        let vals: Vec<i32> = (0..n as i32).collect();
        let steps = vec![Script::Rows(vec![ints(&vals)])];
        let target = (n / 2) as f64;
        let op = Box::new(RowSample::new(Box::new(Mock { steps, pos: 0 }), &spec(true, target, 5)));
        let got = drive(op);
        assert_eq!(got.len(), n / 2);
    }
}
