//! The `generate_series(start, stop, step)` / `range(start, stop, step)` table
//! functions.
//!
//! A "compute-only source" that goes through neither the catalog nor I/O. It generates and
//! returns `BATCH_SIZE` rows at a time -- even a huge range like `range(0, 100000000)` is never
//! expanded into memory; only "the next value to return" is kept in `self`. `NeedIo`/`NeedCodec`
//! can never be returned in principle (the same reason as `exec::MemScan`: the real data is
//! already in memory -- or rather, it is pure computation).

use crate::exec::{ExecContext, Operator, Step};
use crate::prelude::*;
use crate::vector::{Batch, Ty, Value, Vector, BATCH_SIZE};

pub struct GenerateSeries {
    /// The next value to return.
    cur: i64,
    stop: i64,
    step: i64,
    /// Whether `stop` is included (`generate_series` includes it; `range` does not).
    inclusive: bool,
    done: bool,
}

impl GenerateSeries {
    pub fn new(start: i64, stop: i64, step: i64, inclusive: bool) -> Self {
        GenerateSeries { cur: start, stop, step, inclusive, done: false }
    }

    /// Whether `cur` is still in range (whether one more may be emitted).
    fn in_range(&self) -> bool {
        if self.step > 0 {
            if self.inclusive {
                self.cur <= self.stop
            } else {
                self.cur < self.stop
            }
        } else if self.inclusive {
            self.cur >= self.stop
        } else {
            self.cur > self.stop
        }
    }
}

impl Operator for GenerateSeries {
    fn next(&mut self, _ctx: &mut ExecContext) -> Result<Step> {
        if self.done {
            return Ok(Step::Done);
        }
        let mut v = Vector::with_capacity(Ty::BigInt, BATCH_SIZE);
        while v.len() < BATCH_SIZE {
            if !self.in_range() {
                self.done = true;
                break;
            }
            v.push_value(&Value::I64(self.cur));
            // Wrapping on overflow would loop forever, so that itself is made a termination
            // condition (stop when `checked_add` gives `None`).
            match self.cur.checked_add(self.step) {
                Some(next) => self.cur = next,
                None => {
                    self.done = true;
                    break;
                }
            }
        }
        if v.is_empty() {
            return Ok(Step::Done);
        }
        Ok(Step::Ready(Batch::new(vec![v])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Catalog;
    use crate::expr::vm::Vm;

    fn drive(mut op: GenerateSeries) -> Vec<i64> {
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
                        let Value::I64(v) = b.cols[0].value_at(r) else { panic!("expected I64") };
                        out.push(v);
                    }
                }
                Step::NeedIo | Step::NeedCodec => panic!("impossible for a compute-only source"),
                Step::Done => break,
            }
        }
        out
    }

    #[test]
    fn range_excludes_the_stop_value() {
        assert_eq!(drive(GenerateSeries::new(0, 5, 1, false)), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn generate_series_includes_the_stop_value() {
        assert_eq!(drive(GenerateSeries::new(0, 5, 1, true)), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn step_and_negative_step_are_honored() {
        assert_eq!(drive(GenerateSeries::new(0, 10, 2, false)), vec![0, 2, 4, 6, 8]);
        assert_eq!(drive(GenerateSeries::new(10, 0, -2, false)), vec![10, 8, 6, 4, 2]);
    }

    #[test]
    fn mismatched_direction_yields_zero_rows() {
        assert_eq!(drive(GenerateSeries::new(10, 0, 1, false)), Vec::<i64>::new());
        assert_eq!(drive(GenerateSeries::new(0, 10, -1, false)), Vec::<i64>::new());
    }

    #[test]
    fn spans_multiple_batches_without_materializing_everything_up_front() {
        let n = BATCH_SIZE * 2 + 137;
        let got = drive(GenerateSeries::new(0, n as i64, 1, false));
        assert_eq!(got.len(), n);
        assert_eq!(got[0], 0);
        assert_eq!(got[n - 1], (n - 1) as i64);
    }

    #[test]
    fn overflow_at_the_boundary_stops_instead_of_looping_forever() {
        // When adding `step` would exceed `i64::MAX`, it stops without producing the next value.
        let got = drive(GenerateSeries::new(i64::MAX - 2, i64::MAX, 1, true));
        assert_eq!(got, vec![i64::MAX - 2, i64::MAX - 1, i64::MAX]);
    }
}
