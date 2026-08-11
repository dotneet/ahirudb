//! `generate_series(start, stop, step)` / `range(start, stop, step)` テーブル
//! 関数。
//!
//! カタログ・I/O を一切経由しない「計算だけのソース」。`BATCH_SIZE` 行ずつ
//! 生成して返す ―― `range(0, 100000000)` のような巨大な範囲でも、全体を
//! メモリへ展開せず「次に返す値」だけを `self` に持って進む。`NeedIo`/
//! `NeedCodec` は原理的に返らない（`exec::MemScan` と同じ理由: 実データが
//! 既にメモリ上、というより計算のみで完結する）。

use crate::exec::{ExecContext, Operator, Step};
use crate::prelude::*;
use crate::vector::{Batch, Ty, Value, Vector, BATCH_SIZE};

pub struct GenerateSeries {
    /// 次に返す値。
    cur: i64,
    stop: i64,
    step: i64,
    /// `stop` を含むか（`generate_series` は含む、`range` は含まない）。
    inclusive: bool,
    done: bool,
}

impl GenerateSeries {
    pub fn new(start: i64, stop: i64, step: i64, inclusive: bool) -> Self {
        GenerateSeries { cur: start, stop, step, inclusive, done: false }
    }

    /// `cur` がまだ範囲内か（次の 1 個を出してよいか）。
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
            // オーバーフローで折り返すと無限ループになるので、それ自体を
            // 終了条件にする（`checked_add` が `None` なら打ち切る）。
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
            assert!(guard < 9_999, "終わらない");
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
                Step::NeedIo | Step::NeedCodec => panic!("計算だけのソースなので起きないはず"),
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
        // `step` を足すと `i64::MAX` を超える場合、次の値を作らずに打ち切る。
        let got = drive(GenerateSeries::new(i64::MAX - 2, i64::MAX, 1, true));
        assert_eq!(got, vec![i64::MAX - 2, i64::MAX - 1, i64::MAX]);
    }
}
