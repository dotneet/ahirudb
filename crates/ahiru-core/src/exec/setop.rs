//! Set operations (UNION / INTERSECT / EXCEPT).
//!
//! Row identity is delegated to `exec::rowkey::encode_key`. Using the same function as
//! aggregation and joins keeps the handling of NULL / -0.0 / NaN from drifting. In particular,
//! set operations need **NULL to equal NULL** (unlike `=`), and `encode_key` implements exactly
//! those semantics. The hash table is `rowkey::HashIndex` too.
//! Carrying two tables would only cost code size with nothing gained.
//!
//! ## How much it blocks, and resumption
//!
//! - `UNION ALL` keeps duplicates and is a **pass-through**. It streams the left through and
//!   then the right, buffering not a single row (`Phase::Left -> Right`).
//! - `UNION` (DISTINCT) buffers no rows either. It keeps only the set of seen keys and streams
//!   both sides' batches while narrowing them with selection. Clearly lighter than a row buffer.
//! - `INTERSECT` / `EXCEPT` cannot judge the left's first row **until the right is read
//!   through**. `Phase::BuildRight` builds the right's keys and occurrence counts, after which
//!   `Phase::Left` streams the left.
//!
//! At any stage the input can return `Step::NeedIo` / `NeedCodec`. Interruptions are returned
//! straight up, and the partial state (phase, hash table, occurrence counts) stays in `self`, so
//! the next `next()` pulls the input again from the same place (DESIGN.md §6). A batch is either
//! "fully processed" or "not touched yet".
//!
//! ## Duplicate counts (cross-checked with DuckDB)
//!
//! - `INTERSECT ALL` keeps `min(left count, right count)`.
//! - `EXCEPT ALL` keeps `max(0, left count - right count)`.
//! - The DISTINCT versions also deduplicate the output itself.
//!
//! ## Memory
//!
//! There is no spilling. Once the key set exceeds `MAX_STATE_BYTES` it returns `Oom`.

use crate::exec::rowkey::{encode_key, HashIndex};
use crate::exec::{ExecContext, Operator, Step};
use crate::plan::SetOpKind;
use crate::prelude::*;
use crate::vector::{Batch, PhysType, Vector};

/// The approximate byte budget allowed for the key set. Exceeding it gives `Oom`.
/// Set to the same level as aggregation (64 MiB). Only keys, not the rows themselves, are held,
/// so any input for which this is not enough would not pass aggregation either.
const MAX_STATE_BYTES: usize = 64 << 20;

enum Phase {
    /// For INTERSECT / EXCEPT, building the right's key set.
    BuildRight,
    /// Streaming the left.
    Left,
    /// For UNION, streaming the right.
    Right,
    Done,
}

pub struct SetOp {
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    op: SetOpKind,
    all: bool,
    phase: Phase,

    /// The physical types of the left and right columns. Decided from the first batch seen and checked thereafter.
    /// The binder is assumed to have aligned the types, but a mismatch would change the key
    /// length and break things silently as "no match", so it is checked at runtime too.
    shape: Option<Vec<PhysType>>,

    /// The right's key -> an index into `counts` (INTERSECT / EXCEPT only).
    index: HashIndex,
    /// The remaining occurrences of a key on the right. Decremented to adjust counts for ALL.
    counts: Vec<u32>,
    /// Output deduplication (the DISTINCT family only).
    seen: HashIndex,
    /// The destination `encode_key` writes to. Reused rather than allocated per row.
    keybuf: Vec<u8>,
}

impl SetOp {
    pub fn new(
        left: Box<dyn Operator>,
        right: Box<dyn Operator>,
        op: SetOpKind,
        all: bool,
    ) -> Result<Self> {
        // UNION need not buffer the right, so it starts streaming from the left immediately.
        let phase = if op == SetOpKind::Union { Phase::Left } else { Phase::BuildRight };
        Ok(SetOp {
            left,
            right,
            op,
            all,
            phase,
            shape: None,
            index: HashIndex::new(),
            counts: Vec::new(),
            seen: HashIndex::new(),
            keybuf: Vec::new(),
        })
    }

    /// Whether it is a pass-through that never looks at duplicates.
    #[inline]
    fn pass_through(&self) -> bool {
        self.op == SetOpKind::Union && self.all
    }

    /// The approximate memory usage. Used only for the cap check.
    fn mem_used(&self) -> usize {
        self.index.approx_bytes() + self.counts.len() * 4 + self.seen.approx_bytes()
    }

    /// Confirms the column count and physical types agree between left and right.
    fn check_shape(&mut self, batch: &Batch) -> Result<()> {
        match &self.shape {
            Some(s) => {
                ensure!(batch.cols.len() == s.len(), Internal);
                for (c, p) in batch.cols.iter().zip(s.iter()) {
                    ensure!(c.data().phys() == *p, TypeMismatch);
                }
            }
            None => self.shape = Some(batch.cols.iter().map(|c| c.data().phys()).collect()),
        }
        Ok(())
    }

    /// Takes one right batch into the key set. **It never bails out midway**.
    fn absorb_right(&mut self, mut batch: Batch) -> Result<()> {
        if batch.card() == 0 {
            return Ok(());
        }
        self.check_shape(&batch)?;
        // From here on lookups are by row number, so selection is materialized now.
        batch.materialize();
        let rows = batch.num_rows();
        let refs: Vec<&Vector> = batch.cols.iter().collect();
        for r in 0..rows {
            encode_key(&refs, r, &mut self.keybuf);
            let (slot, is_new) = self.index.get_or_insert(&self.keybuf);
            if is_new {
                self.counts.push(0);
            }
            match self.counts.get_mut(slot as usize) {
                Some(c) => {
                    // The occurrence count is u32. Duplication beyond that is given up on.
                    ensure!(*c < u32::MAX, LimitExceeded);
                    *c += 1;
                }
                None => err!(Internal),
            }
        }
        ensure!(self.mem_used() <= MAX_STATE_BYTES, Oom);
        Ok(())
    }

    /// Narrows one batch of the left (or, for UNION, the right).
    /// `None` means "no output, but state advanced".
    fn filter(&mut self, mut batch: Batch) -> Result<Option<Step>> {
        if batch.card() == 0 {
            return Ok(None);
        }
        // The column-count and physical-type checks are independent of selection, so they run before materializing.
        self.check_shape(&batch)?;
        if self.pass_through() {
            return Ok(Some(Step::Ready(batch)));
        }
        batch.materialize();
        let rows = batch.num_rows();
        let refs: Vec<&Vector> = batch.cols.iter().collect();
        let mut sel: Vec<u32> = Vec::new();
        for r in 0..rows {
            encode_key(&refs, r, &mut self.keybuf);
            if self.keep() {
                sel.push(r as u32);
            }
        }
        ensure!(self.mem_used() <= MAX_STATE_BYTES, Oom);
        if sel.is_empty() {
            // An empty batch is not returned upstream.
            return Ok(None);
        }
        if sel.len() < rows {
            batch.sel = Some(sel);
        }
        Ok(Some(Step::Ready(batch)))
    }

    /// Whether to emit `keybuf`'s row. Under ALL it also consumes an occurrence.
    ///
    /// Under UNION the right side passes through here too, but UNION's decision is the same for
    /// both sides, so there is no need to say which side it is (INTERSECT / EXCEPT never stream the right).
    fn keep(&mut self) -> bool {
        let qualifies = match self.op {
            SetOpKind::Union => true,
            SetOpKind::Intersect => match self.index.lookup(&self.keybuf) {
                None => false,
                Some(slot) => {
                    if !self.all {
                        true
                    } else {
                        // Emit only while the right's stock lasts -> min(left, right) rows.
                        match self.counts.get_mut(slot as usize) {
                            Some(c) if *c > 0 => {
                                *c -= 1;
                                true
                            }
                            _ => false,
                        }
                    }
                }
            },
            SetOpKind::Except => match self.index.lookup(&self.keybuf) {
                None => true,
                Some(slot) => {
                    if !self.all {
                        false
                    } else {
                        // Consume the right's stock first and emit only the surplus
                        // -> max(0, left - right) rows.
                        match self.counts.get_mut(slot as usize) {
                            Some(c) if *c > 0 => {
                                *c -= 1;
                                false
                            }
                            _ => true,
                        }
                    }
                }
            },
        };
        if !qualifies {
            return false;
        }
        // The DISTINCT versions deduplicate the output itself too.
        self.all || self.seen.get_or_insert(&self.keybuf).1
    }
}

impl Operator for SetOp {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        loop {
            match self.phase {
                Phase::BuildRight => match self.right.next(ctx)? {
                    Step::Ready(b) => self.absorb_right(b)?,
                    // Exits with the partially built key set intact. Next time it resumes from here.
                    Step::NeedIo => return Ok(Step::NeedIo),
                    Step::NeedCodec => return Ok(Step::NeedCodec),
                    Step::Done => self.phase = Phase::Left,
                },
                Phase::Left => match self.left.next(ctx)? {
                    Step::Ready(b) => {
                        if let Some(s) = self.filter(b)? {
                            return Ok(s);
                        }
                    }
                    Step::NeedIo => return Ok(Step::NeedIo),
                    Step::NeedCodec => return Ok(Step::NeedCodec),
                    Step::Done => {
                        // Only UNION emits the right as well.
                        self.phase =
                            if self.op == SetOpKind::Union { Phase::Right } else { Phase::Done };
                    }
                },
                Phase::Right => match self.right.next(ctx)? {
                    Step::Ready(b) => {
                        if let Some(s) = self.filter(b)? {
                            return Ok(s);
                        }
                    }
                    Step::NeedIo => return Ok(Step::NeedIo),
                    Step::NeedCodec => return Ok(Step::NeedCodec),
                    Step::Done => self.phase = Phase::Done,
                },
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
    use crate::vector::{Ty, Value, BATCH_SIZE};

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

    /// A script of one column (INT) in one batch.
    fn one(vals: &[Option<i32>]) -> Vec<Script> {
        vec![Script::Rows(vec![ints(vals)])]
    }

    // --- Execution helpers --------------------------------------------------

    fn drive(l: Vec<Script>, r: Vec<Script>, op: SetOpKind, all: bool) -> Vec<Vec<Value>> {
        let mut o = SetOp::new(Mock::new(l), Mock::new(r), op, all).unwrap();
        let mut cat = Catalog::new();
        let mut vm = Vm::new();
        let mut rows = Vec::new();
        for guard in 0..100_000 {
            assert!(guard < 99_999, "does not terminate");
            let mut ctx =
                ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
            match o.next(&mut ctx).unwrap() {
                Step::Ready(b) => {
                    let n = b.card();
                    assert!(n > 0, "an empty batch must not be returned");
                    assert!(n <= BATCH_SIZE);
                    for i in 0..n {
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

    /// Flattens the output into an easily compared form, sorted ascending by the first column's value (NULLs last).
    fn sorted(rows: Vec<Vec<Value>>) -> Vec<Option<i64>> {
        let mut v: Vec<Option<i64>> = rows.iter().map(|r| r[0].as_i64()).collect();
        v.sort_by(|a, b| match (a, b) {
            (None, None) => core::cmp::Ordering::Equal,
            (None, _) => core::cmp::Ordering::Greater,
            (_, None) => core::cmp::Ordering::Less,
            (Some(x), Some(y)) => x.cmp(y),
        });
        v
    }

    fn run(l: &[Option<i32>], r: &[Option<i32>], op: SetOpKind, all: bool) -> Vec<Option<i64>> {
        sorted(drive(one(l), one(r), op, all))
    }

    // --- All six variants (cross-checked with DuckDB) -----------------------

    // a = [1,1,1,2,NULL,NULL,3] / b = [1,1,2,2,NULL,4]
    const A: [Option<i32>; 7] = [Some(1), Some(1), Some(1), Some(2), None, None, Some(3)];
    const B: [Option<i32>; 6] = [Some(1), Some(1), Some(2), Some(2), None, Some(4)];

    #[test]
    fn union_all_keeps_everything() {
        let got = run(&A, &B, SetOpKind::Union, true);
        assert_eq!(got.len(), 13);
        assert_eq!(
            got,
            vec![
                Some(1),
                Some(1),
                Some(1),
                Some(1),
                Some(1),
                Some(2),
                Some(2),
                Some(2),
                Some(3),
                Some(4),
                None,
                None,
                None
            ]
        );
    }

    #[test]
    fn union_distinct() {
        // NULL counts as the same row as NULL, so only one survives.
        assert_eq!(
            run(&A, &B, SetOpKind::Union, false),
            vec![Some(1), Some(2), Some(3), Some(4), None]
        );
    }

    #[test]
    fn intersect_all_keeps_min_count() {
        // 1: min(3,2)=2 / 2: min(1,2)=1 / NULL: min(2,1)=1 / 3,4: 0
        assert_eq!(run(&A, &B, SetOpKind::Intersect, true), vec![Some(1), Some(1), Some(2), None]);
    }

    #[test]
    fn intersect_distinct() {
        assert_eq!(run(&A, &B, SetOpKind::Intersect, false), vec![Some(1), Some(2), None]);
    }

    #[test]
    fn except_all_keeps_left_minus_right_count() {
        // 1: 3-2=1 / 2: 1-2->0 / NULL: 2-1=1 / 3: 1-0=1
        assert_eq!(run(&A, &B, SetOpKind::Except, true), vec![Some(1), Some(3), None]);
    }

    #[test]
    fn except_distinct() {
        assert_eq!(run(&A, &B, SetOpKind::Except, false), vec![Some(3)]);
    }

    // --- Interruption and resumption (the most important) -------------------

    #[test]
    fn need_io_and_need_codec_match_uninterrupted_run() {
        let chunks = |v: &[Option<i32>]| -> Vec<Vec<Option<i32>>> {
            v.chunks(3).map(|c| c.to_vec()).collect()
        };
        let script = |v: &[Option<i32>], interrupted: bool| {
            let mut out = Vec::new();
            for (i, c) in chunks(v).into_iter().enumerate() {
                // Both interruptions are interposed mid-input (neither at the start nor at the end).
                if interrupted && i == 1 {
                    out.push(Script::NeedIo);
                }
                out.push(Script::Rows(vec![ints(&c)]));
                if interrupted && i == 1 {
                    out.push(Script::NeedCodec);
                }
            }
            if interrupted {
                out.push(Script::NeedIo);
            }
            out
        };
        for op in [SetOpKind::Union, SetOpKind::Intersect, SetOpKind::Except] {
            for all in [true, false] {
                let plain = sorted(drive(script(&A, false), script(&B, false), op, all));
                let noisy = sorted(drive(script(&A, true), script(&B, true), op, all));
                assert_eq!(noisy, plain, "{op:?} all={all}");
            }
        }
    }

    #[test]
    fn need_io_before_any_input() {
        let l = vec![Script::NeedIo, Script::NeedCodec, Script::Rows(vec![ints(&[Some(1)])])];
        let r = vec![Script::NeedIo, Script::Rows(vec![ints(&[Some(1)])])];
        assert_eq!(sorted(drive(l, r, SetOpKind::Intersect, false)), vec![Some(1)]);
    }

    /// Interruptions must propagate to the caller as they are (not swallowed).
    #[test]
    fn interrupts_are_forwarded_unchanged() {
        let mut o = SetOp::new(
            Mock::new(vec![Script::NeedCodec, Script::Rows(vec![ints(&[Some(1)])])]),
            Mock::new(vec![Script::NeedIo, Script::Rows(vec![ints(&[Some(1)])])]),
            SetOpKind::Intersect,
            false,
        )
        .unwrap();
        let mut cat = Catalog::new();
        let mut vm = Vm::new();
        let mut ctx =
            ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
        // First an interruption on the right (the build side).
        assert!(matches!(o.next(&mut ctx).unwrap(), Step::NeedIo));
        // Then one on the left.
        assert!(matches!(o.next(&mut ctx).unwrap(), Step::NeedCodec));
        assert!(matches!(o.next(&mut ctx).unwrap(), Step::Ready(_)));
    }

    // --- Empty inputs -------------------------------------------------------

    #[test]
    fn empty_left_side() {
        let e: [Option<i32>; 0] = [];
        assert_eq!(run(&e, &B, SetOpKind::Union, true).len(), 6);
        assert_eq!(run(&e, &B, SetOpKind::Union, false).len(), 4);
        assert!(run(&e, &B, SetOpKind::Intersect, true).is_empty());
        assert!(run(&e, &B, SetOpKind::Intersect, false).is_empty());
        assert!(run(&e, &B, SetOpKind::Except, true).is_empty());
        assert!(run(&e, &B, SetOpKind::Except, false).is_empty());
    }

    #[test]
    fn empty_right_side() {
        let e: [Option<i32>; 0] = [];
        assert_eq!(run(&A, &e, SetOpKind::Union, true).len(), 7);
        assert_eq!(run(&A, &e, SetOpKind::Union, false).len(), 4);
        assert!(run(&A, &e, SetOpKind::Intersect, true).is_empty());
        assert!(run(&A, &e, SetOpKind::Intersect, false).is_empty());
        assert_eq!(
            run(&A, &e, SetOpKind::Except, true).len(),
            7,
            "an empty right leaves the left unchanged"
        );
        assert_eq!(run(&A, &e, SetOpKind::Except, false), vec![Some(1), Some(2), Some(3), None]);
    }

    #[test]
    fn both_sides_empty() {
        let e: [Option<i32>; 0] = [];
        for op in [SetOpKind::Union, SetOpKind::Intersect, SetOpKind::Except] {
            for all in [true, false] {
                assert!(run(&e, &e, op, all).is_empty(), "{op:?} all={all}");
            }
        }
    }

    /// It does not break when only 0-row batches arrive.
    #[test]
    fn zero_row_batches_are_ignored() {
        let l = vec![Script::Rows(vec![ints(&[])]), Script::Rows(vec![ints(&[Some(1)])])];
        let r = vec![Script::Rows(vec![ints(&[])])];
        assert_eq!(sorted(drive(l, r, SetOpKind::Except, true)), vec![Some(1)]);
    }

    // --- NULL ---------------------------------------------------------------

    /// In set operations NULL matches NULL (unlike `=`).
    #[test]
    fn nulls_match_each_other() {
        let n = [None, None];
        assert_eq!(run(&n, &[None], SetOpKind::Intersect, false), vec![None]);
        assert_eq!(run(&n, &[None], SetOpKind::Intersect, true), vec![None]);
        assert_eq!(run(&n, &[None], SetOpKind::Except, true), vec![None]);
        assert!(run(&n, &[None], SetOpKind::Except, false).is_empty());
        assert_eq!(run(&n, &[None], SetOpKind::Union, false), vec![None]);
    }

    // --- Several columns ----------------------------------------------------

    #[test]
    fn multi_column_rows() {
        let l = vec![Script::Rows(vec![
            ints(&[Some(1), Some(1), Some(2), None]),
            strs(&[Some("a"), Some("b"), Some("a"), Some("a")]),
        ])];
        let r = vec![Script::Rows(vec![
            ints(&[Some(1), Some(2), None]),
            strs(&[Some("a"), Some("z"), Some("a")]),
        ])];
        let got = drive(l, r, SetOpKind::Except, true);
        // (1,a) and (NULL,a) are on the right. (1,b) and (2,a) survive.
        let mut pairs: Vec<(Option<i64>, Option<String>)> = got
            .iter()
            .map(|row| {
                (
                    row[0].as_i64(),
                    row[1].as_bytes().map(|b| String::from_utf8_lossy(b).into_owned()),
                )
            })
            .collect();
        pairs.sort();
        assert_eq!(pairs, vec![(Some(1), Some("b".into())), (Some(2), Some("a".into()))]);
    }

    /// A key concatenated without length delimiters would make ("a","bc") and ("ab","c") collide.
    /// `encode_key` prefixes the length, so they stay separate.
    #[test]
    fn multi_column_keys_are_not_confusable() {
        let l = vec![Script::Rows(vec![strs(&[Some("a")]), strs(&[Some("bc")])])];
        let r = vec![Script::Rows(vec![strs(&[Some("ab")]), strs(&[Some("c")])])];
        assert_eq!(drive(l, r, SetOpKind::Except, false).len(), 1);
    }

    // --- Size ---------------------------------------------------------------

    #[test]
    fn more_rows_than_batch_size() {
        const N: i32 = BATCH_SIZE as i32 * 2 + 37;
        // Left: 0..N / right: the even numbers only.
        let mut l = Vec::new();
        let mut i = 0i32;
        while i < N {
            let end = (i + 500).min(N);
            l.push(Script::Rows(vec![ints(&(i..end).map(Some).collect::<Vec<_>>())]));
            // The result must not change even with an interruption between every batch.
            l.push(Script::NeedIo);
            i = end;
        }
        let evens: Vec<Option<i32>> = (0..N).filter(|x| x % 2 == 0).map(Some).collect();
        let r = vec![Script::Rows(vec![ints(&evens)])];

        let got = sorted(drive(l, r, SetOpKind::Except, true));
        let want: Vec<Option<i64>> =
            (0..N).filter(|x| x % 2 != 0).map(|x| Some(x as i64)).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn union_all_over_many_batches_is_streamed() {
        // Being a pass-through, the input's batch structure comes out as is (nothing is buffered).
        let l = vec![
            Script::Rows(vec![ints(&[Some(1), Some(2)])]),
            Script::Rows(vec![ints(&[Some(3)])]),
        ];
        let r = vec![Script::Rows(vec![ints(&[Some(1)])])];
        let mut o = SetOp::new(Mock::new(l), Mock::new(r), SetOpKind::Union, true).unwrap();
        let mut cat = Catalog::new();
        let mut vm = Vm::new();
        let mut sizes = Vec::new();
        loop {
            let mut ctx =
                ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
            match o.next(&mut ctx).unwrap() {
                Step::Ready(b) => sizes.push(b.card()),
                Step::NeedIo | Step::NeedCodec => {}
                Step::Done => break,
            }
        }
        assert_eq!(sizes, vec![2, 1, 1]);
    }

    // --- Detecting contract violations --------------------------------------

    #[test]
    fn mismatched_column_count_is_rejected() {
        let l = vec![Script::Rows(vec![ints(&[Some(1)]), ints(&[Some(2)])])];
        let r = vec![Script::Rows(vec![ints(&[Some(1)])])];
        let mut o = SetOp::new(Mock::new(l), Mock::new(r), SetOpKind::Except, true).unwrap();
        let mut cat = Catalog::new();
        let mut vm = Vm::new();
        let mut ctx =
            ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
        let mut last = None;
        for _ in 0..5 {
            match o.next(&mut ctx) {
                Ok(Step::Done) => break,
                Ok(_) => {}
                Err(e) => {
                    last = Some(e.code);
                    break;
                }
            }
        }
        assert_eq!(last, Some(Code::Internal));
    }

    #[test]
    fn mismatched_physical_type_is_rejected() {
        let l = vec![Script::Rows(vec![strs(&[Some("1")])])];
        let r = vec![Script::Rows(vec![ints(&[Some(1)])])];
        let mut o = SetOp::new(Mock::new(l), Mock::new(r), SetOpKind::Intersect, true).unwrap();
        let mut cat = Catalog::new();
        let mut vm = Vm::new();
        let mut ctx =
            ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
        let mut last = None;
        for _ in 0..5 {
            match o.next(&mut ctx) {
                Ok(Step::Done) => break,
                Ok(_) => {}
                Err(e) => {
                    last = Some(e.code);
                    break;
                }
            }
        }
        assert_eq!(last, Some(Code::TypeMismatch));
    }
}
