//! 集合演算（UNION / INTERSECT / EXCEPT）。
//!
//! 行の同一判定は `exec::rowkey::encode_key` に寄せる。集約・結合と同じ関数を
//! 使うことで NULL / -0.0 / NaN の扱いがずれない。とくに集合演算では
//! **NULL は NULL と等しい**（`=` とは違う）必要があり、`encode_key` は
//! ちょうどその意味論を実装している。ハッシュ表も `rowkey::HashIndex` を使う。
//! 表を 2 つ持つとコードサイズを損するだけで得が無い。
//!
//! ## ブロッキングの度合いと再開
//!
//! - `UNION ALL` は重複を残すので**素通し**。左を流し切ってから右を流すだけで、
//!   1 行も溜めない（`Phase::Left → Right`）。
//! - `UNION`（DISTINCT）も行は溜めない。既出キーの集合だけ持って、左右の
//!   バッチを selection で絞りながら流す。行バッファを持つより明確に軽い。
//! - `INTERSECT` / `EXCEPT` は**右を読み切ってから**でないと左の 1 行目を
//!   判定できない。`Phase::BuildRight` で右のキーと出現数を作り、その後
//!   `Phase::Left` で左を流す。
//!
//! どの段でも入力は `Step::NeedIo` / `NeedCodec` を返しうる。中断はそのまま
//! 上へ返し、途中状態（フェーズ・ハッシュ表・出現数）は `self` に残るので、
//! 次の `next()` は同じ場所から入力を引き直す（DESIGN.md §6）。1 バッチは
//! 「丸ごと処理した」か「まだ触っていない」かのどちらかしかない。
//!
//! ## 重複の数（DuckDB で照合済み）
//!
//! - `INTERSECT ALL` は `min(左の出現数, 右の出現数)` 件残す。
//! - `EXCEPT ALL` は `max(0, 左の出現数 - 右の出現数)` 件残す。
//! - DISTINCT 版は出力自体も重複除去する。
//!
//! ## メモリ
//!
//! スピルは持たない。キー集合が `MAX_STATE_BYTES` を超えたら `Oom` を返す。

use crate::exec::rowkey::{encode_key, HashIndex};
use crate::exec::{ExecContext, Operator, Step};
use crate::plan::SetOpKind;
use crate::prelude::*;
use crate::vector::{Batch, PhysType, Vector};

/// キー集合に許すおおよそのバイト数。超えたら `Oom`。
/// 集約（64MiB）と同じ水準に揃える。行本体は持たずキーだけなので、
/// これで足りない入力は集約でも通らない。
const MAX_STATE_BYTES: usize = 64 << 20;

enum Phase {
    /// INTERSECT / EXCEPT で、右のキー集合を作っている。
    BuildRight,
    /// 左を流している。
    Left,
    /// UNION で、右を流している。
    Right,
    Done,
}

pub struct SetOp {
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    op: SetOpKind,
    all: bool,
    phase: Phase,

    /// 左右の列の物理型。最初に見たバッチで決め、以降は照合する。
    /// バインダが型を揃えている前提だが、ずれるとキーの長さが変わって
    /// 「一致しない」という形で静かに壊れるので実行時にも見る。
    shape: Option<Vec<PhysType>>,

    /// 右のキー → `counts` の添字（INTERSECT / EXCEPT のみ）。
    index: HashIndex,
    /// 右におけるキーの残り出現数。ALL の件数調整で減らしていく。
    counts: Vec<u32>,
    /// 出力の重複除去（DISTINCT 系のみ）。
    seen: HashIndex,
    /// `encode_key` の書き込み先。行ごとに確保しないよう使い回す。
    keybuf: Vec<u8>,
}

impl SetOp {
    pub fn new(
        left: Box<dyn Operator>,
        right: Box<dyn Operator>,
        op: SetOpKind,
        all: bool,
    ) -> Result<Self> {
        // UNION は右を溜める必要が無いので、いきなり左から流し始める。
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

    /// 重複を一切見ない素通しか。
    #[inline]
    fn pass_through(&self) -> bool {
        self.op == SetOpKind::Union && self.all
    }

    /// おおよそのメモリ使用量。上限判定にしか使わない。
    fn mem_used(&self) -> usize {
        self.index.approx_bytes() + self.counts.len() * 4 + self.seen.approx_bytes()
    }

    /// 列数と物理型が左右で揃っていることを確かめる。
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

    /// 右バッチ 1 つをキー集合へ取り込む。**途中で抜けない**。
    fn absorb_right(&mut self, mut batch: Batch) -> Result<()> {
        if batch.card() == 0 {
            return Ok(());
        }
        self.check_shape(&batch)?;
        // 以降は行番号で引くので selection をここで畳む。
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
                    // 出現数は u32。これを超える重複は諦める。
                    ensure!(*c < u32::MAX, LimitExceeded);
                    *c += 1;
                }
                None => err!(Internal),
            }
        }
        ensure!(self.mem_used() <= MAX_STATE_BYTES, Oom);
        Ok(())
    }

    /// 左（または UNION の右）のバッチ 1 つを絞り込む。
    /// `None` は「出力は無いが状態は進んだ」。
    fn filter(&mut self, mut batch: Batch) -> Result<Option<Step>> {
        if batch.card() == 0 {
            return Ok(None);
        }
        // 列数・物理型の検査は selection と無関係なので畳む前に済ませる。
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
            // 空バッチは上へ返さない。
            return Ok(None);
        }
        if sel.len() < rows {
            batch.sel = Some(sel);
        }
        Ok(Some(Step::Ready(batch)))
    }

    /// `keybuf` の行を出力するか。ALL では出現数も同時に消費する。
    ///
    /// UNION では右側もここを通るが、UNION の判定は左右で同じなので
    /// どちら側かを渡す必要はない（INTERSECT / EXCEPT は右を流さない）。
    fn keep(&mut self) -> bool {
        let qualifies = match self.op {
            SetOpKind::Union => true,
            SetOpKind::Intersect => match self.index.lookup(&self.keybuf) {
                None => false,
                Some(slot) => {
                    if !self.all {
                        true
                    } else {
                        // 右の在庫がある間だけ出す → min(左, 右) 件。
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
                        // 右の在庫を先に食い潰し、余った分だけ出す
                        // → max(0, 左 - 右) 件。
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
        // DISTINCT 版は出力自体も重複除去する。
        self.all || self.seen.get_or_insert(&self.keybuf).1
    }
}

impl Operator for SetOp {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        loop {
            match self.phase {
                Phase::BuildRight => match self.right.next(ctx)? {
                    Step::Ready(b) => self.absorb_right(b)?,
                    // 作りかけのキー集合を保ったまま抜ける。次回はここから再開。
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
                        // UNION だけが右も出力する。
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

    // --- 組み立てヘルパ -----------------------------------------------------

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

    // --- モック入力 ---------------------------------------------------------

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
            // 中断は「ホストの応答を待った」ことにして消費する。
            Ok(match &self.steps[i] {
                Script::NeedIo => Step::NeedIo,
                Script::NeedCodec => Step::NeedCodec,
                Script::Rows(cols) => Step::Ready(Batch::new(cols.clone())),
            })
        }
    }

    /// 1 列（INT）1 バッチの台本。
    fn one(vals: &[Option<i32>]) -> Vec<Script> {
        vec![Script::Rows(vec![ints(vals)])]
    }

    // --- 実行ヘルパ ---------------------------------------------------------

    fn drive(l: Vec<Script>, r: Vec<Script>, op: SetOpKind, all: bool) -> Vec<Vec<Value>> {
        let mut o = SetOp::new(Mock::new(l), Mock::new(r), op, all).unwrap();
        let mut cat = Catalog::new();
        let mut vm = Vm::new();
        let mut rows = Vec::new();
        for guard in 0..100_000 {
            assert!(guard < 99_999, "終わらない");
            let mut ctx =
                ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
            match o.next(&mut ctx).unwrap() {
                Step::Ready(b) => {
                    let n = b.card();
                    assert!(n > 0, "空バッチを返してはいけない");
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

    /// 出力を「第 1 列の値」の昇順（NULL は末尾）で並べた比較しやすい形に落とす。
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

    // --- 6 通り（DuckDB で照合済み） ----------------------------------------

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
        // NULL は NULL と同じ行とみなすので 1 つだけ残る。
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
        // 1: 3-2=1 / 2: 1-2→0 / NULL: 2-1=1 / 3: 1-0=1
        assert_eq!(run(&A, &B, SetOpKind::Except, true), vec![Some(1), Some(3), None]);
    }

    #[test]
    fn except_distinct() {
        assert_eq!(run(&A, &B, SetOpKind::Except, false), vec![Some(3)]);
    }

    // --- 中断と再開（最重要） -----------------------------------------------

    #[test]
    fn need_io_and_need_codec_match_uninterrupted_run() {
        let chunks = |v: &[Option<i32>]| -> Vec<Vec<Option<i32>>> {
            v.chunks(3).map(|c| c.to_vec()).collect()
        };
        let script = |v: &[Option<i32>], interrupted: bool| {
            let mut out = Vec::new();
            for (i, c) in chunks(v).into_iter().enumerate() {
                // 入力の途中（先頭でも末尾でもない位置）に両方の中断を挟む。
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

    /// 中断がそのまま呼び出し元へ伝わること（握り潰していない）。
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
        // まず右（ビルド側）の中断。
        assert!(matches!(o.next(&mut ctx).unwrap(), Step::NeedIo));
        // 次に左の中断。
        assert!(matches!(o.next(&mut ctx).unwrap(), Step::NeedCodec));
        assert!(matches!(o.next(&mut ctx).unwrap(), Step::Ready(_)));
    }

    // --- 空入力 -------------------------------------------------------------

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
        assert_eq!(run(&A, &e, SetOpKind::Except, true).len(), 7, "右が空なら左そのまま");
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

    /// 0 行のバッチだけが来ても壊れない。
    #[test]
    fn zero_row_batches_are_ignored() {
        let l = vec![Script::Rows(vec![ints(&[])]), Script::Rows(vec![ints(&[Some(1)])])];
        let r = vec![Script::Rows(vec![ints(&[])])];
        assert_eq!(sorted(drive(l, r, SetOpKind::Except, true)), vec![Some(1)]);
    }

    // --- NULL ---------------------------------------------------------------

    /// 集合演算では NULL は NULL と一致する（`=` とは違う）。
    #[test]
    fn nulls_match_each_other() {
        let n = [None, None];
        assert_eq!(run(&n, &[None], SetOpKind::Intersect, false), vec![None]);
        assert_eq!(run(&n, &[None], SetOpKind::Intersect, true), vec![None]);
        assert_eq!(run(&n, &[None], SetOpKind::Except, true), vec![None]);
        assert!(run(&n, &[None], SetOpKind::Except, false).is_empty());
        assert_eq!(run(&n, &[None], SetOpKind::Union, false), vec![None]);
    }

    // --- 複数列 -------------------------------------------------------------

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
        // (1,a) と (NULL,a) は右にある。(1,b) と (2,a) が残る。
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

    /// 列の長さで区切らずに繋いだキーだと ("a","bc") と ("ab","c") が衝突する。
    /// `encode_key` が長さを前置しているので分かれること。
    #[test]
    fn multi_column_keys_are_not_confusable() {
        let l = vec![Script::Rows(vec![strs(&[Some("a")]), strs(&[Some("bc")])])];
        let r = vec![Script::Rows(vec![strs(&[Some("ab")]), strs(&[Some("c")])])];
        assert_eq!(drive(l, r, SetOpKind::Except, false).len(), 1);
    }

    // --- 大きさ -------------------------------------------------------------

    #[test]
    fn more_rows_than_batch_size() {
        const N: i32 = BATCH_SIZE as i32 * 2 + 37;
        // 左: 0..N / 右: 偶数のみ。
        let mut l = Vec::new();
        let mut i = 0i32;
        while i < N {
            let end = (i + 500).min(N);
            l.push(Script::Rows(vec![ints(&(i..end).map(Some).collect::<Vec<_>>())]));
            // バッチの間で毎回中断しても結果が変わらないこと。
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
        // 素通しなので入力のバッチ構成がそのまま出る（溜め込まない）。
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

    // --- 契約違反の検出 -----------------------------------------------------

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
