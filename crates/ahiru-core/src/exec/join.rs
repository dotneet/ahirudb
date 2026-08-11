//! ハッシュ結合とネストループ結合。
//!
//! ビルド側は**常に右入力**に固定する。どちらを右に置くか（＝小さいほうを
//! ビルドにする）はバインダの仕事で、ここでは判断しない。実行時に入れ替えると
//! 出力列順（左の列 → 右の列）が変わり、`residual` と上位のスキーマが崩れる。
//!
//! ## 中断と再開
//!
//! 入力は `NeedIo` / `NeedCodec` を返しうる。ビルドは右入力を**全部**読み切って
//! からでないと探索を始められないので、フェーズ（`Building` → `Probing` →
//! `DrainingUnmatched` → `Done`）を明示的に持ち、中断時はそのまま抜けて次の
//! `next()` で同じ位置から再開する。ハッシュ表・探索中のバッチ・一致ビットマップ
//! はすべて `self` に置く。
//!
//! ## メモリ
//!
//! ビルド側はメモリに全部載せる。溢れをディスクに逃がす仕組みは持たない
//! （既知の制限）。代わりに `MAX_BUILD_BYTES` で頭打ちにして `Oom` を返す。

use crate::exec::rowkey::{encode_key, key_has_null, HashIndex};
use crate::exec::{ExecContext, Operator, Step};
use crate::expr::Program;
use crate::prelude::*;
use crate::sql::ast::JoinKind;
use crate::vector::{Batch, Bitmap, Data, Ty, Vector, BATCH_SIZE};

/// 「相手が居ない」を表す行番号。チェーンの終端にも使う。
const NONE: u32 = u32::MAX;

/// バッファできるビルド側のバイト数。スピルを持たないので、これを超えたら
/// 静かに巨大なメモリを掴む代わりにエラーで落とす。
const MAX_BUILD_BYTES: usize = 128 * 1024 * 1024;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// 右入力を読み切ってハッシュ表を作っている。
    Building,
    /// 左入力を 1 バッチずつ探索している。
    Probing,
    /// RIGHT/FULL で、どの左行とも一致しなかったビルド行を吐いている。
    DrainingUnmatched,
    Done,
}

/// 探索中の左バッチと、その途中経過。
struct Probe {
    /// selection を畳んだ左バッチ。行番号がそのまま添字になる。
    batch: Batch,
    /// 左キーの評価結果。`batch` と同じ行数。
    keys: Vec<Vector>,
    /// 走査中の左行。
    row: usize,
    /// 次に試すビルド行。`None` はこの左行をまだ開始していないこと、
    /// `Some(NONE)` は候補を出し切ったことを表す。
    cursor: Option<u32>,
    /// 左行ごとの「residual まで通った一致があったか」。
    matched: Bitmap,
    /// 未一致左行の NULL 拡張で次に見る行。
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

    /// `kind` はこの 2 つのフラグにしか効かないので、そのまま持たずに畳む。
    /// 未一致の左行を NULL 拡張して出すか（LEFT / FULL）。
    emit_unmatched_left: bool,
    /// 未一致のビルド行を NULL 拡張して出すか（RIGHT / FULL）。
    emit_unmatched_right: bool,

    phase: Phase,
    /// バッファしたビルド側。右入力の全バッチを 1 本に連結したもの。
    build_cols: Vec<Vector>,
    build_rows: usize,
    /// 概算のバッファ量。上限判定にしか使わない。
    build_bytes: usize,
    /// キー → チェーン先頭のビルド行番号。
    index: HashIndex,
    /// 同じキーを持つビルド行の連鎖。終端は `NONE`。
    next: Vec<u32>,
    /// ビルド行ごとの一致フラグ。RIGHT/FULL のときだけ確保する。
    build_matched: Bitmap,
    /// `encode_key` の書き込み先。行ごとに確保しないよう使い回す。
    keybuf: Vec<u8>,
    probe: Option<Probe>,
    /// `DrainingUnmatched` で次に見るビルド行。
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
        // 等値条件は左右で対になっている。ずれていればバインダのバグ。
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
            emit_unmatched_left: matches!(kind, JoinKind::Left | JoinKind::Full),
            emit_unmatched_right: matches!(kind, JoinKind::Right | JoinKind::Full),
            phase: Phase::Building,
            build_cols,
            build_rows: 0,
            build_bytes: 0,
            index: HashIndex::new(),
            next: Vec::new(),
            build_matched: Bitmap::new(),
            keybuf: Vec::new(),
            probe: None,
            drain: 0,
        })
    }

    /// 等値キーが無い＝ネストループで総当たりする。CROSS と `ON a.x < b.y`
    /// のような非等値結合がここに来る。
    #[inline]
    fn nested(&self) -> bool {
        self.right_keys.is_empty()
    }

    /// 右バッチ 1 つをビルド側に取り込む。
    fn absorb(&mut self, ctx: &mut ExecContext, mut batch: Batch) -> Result<()> {
        // 以降は行番号で引くので selection をここで畳む。
        batch.materialize();
        let rows = batch.num_rows();
        if rows == 0 {
            return Ok(());
        }
        ensure!(batch.cols.len() == self.right_types.len(), Internal);

        // キーは連結前に評価する（materialize 済みなので行番号がそのまま対応）。
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
        // 行番号は u32 で持つ。上限バイト数のほうが先に効くはずだが、
        // 0 列のバッチ（COUNT(*) 経路）ではバイト数が増えないので明示的に見る。
        ensure!(self.build_rows < NONE as usize, Oom);
        ensure!(self.build_bytes + self.index.key_bytes() <= MAX_BUILD_BYTES, Oom);

        if !self.nested() {
            self.next.resize(self.build_rows, NONE);
            let refs: Vec<&Vector> = keys.iter().collect();
            for r in 0..rows {
                // NULL を含むキーは何とも一致しない（SQL の `=` は NULL 安全でない）
                // ので表に入れない。行自体は残るので OUTER の未一致ドレインには出る。
                if key_has_null(&refs, r) {
                    continue;
                }
                encode_key(&refs, r, &mut self.keybuf);
                let id = (base + r) as u32;
                self.next[base + r] = self.index.insert_chained(&self.keybuf, id).unwrap_or(NONE);
            }
        }
        Ok(())
    }

    /// 探索フェーズを 1 歩進める。`None` は「出力は無いが状態は進んだ」。
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
                // 中断。ハッシュ表はそのまま、探索も未開始のまま抜ける。
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

        // --- 候補ペアを最大 BATCH_SIZE 件ぶん作る -----------------------------
        // 1 つの左行が多数のビルド行に当たることがあるので、左行の途中で
        // 打ち切れるように `cursor` を残す。
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
                let cur = match p.cursor {
                    Some(c) => c,
                    None => {
                        // `nested()` と同じ判定。`probe` を借りている間はメソッドを
                        // 呼べないのでフィールドを直接見る。
                        let head = if self.right_keys.is_empty() {
                            if self.build_rows == 0 {
                                NONE
                            } else {
                                0
                            }
                        } else if key_has_null(&refs, p.row) {
                            // 探索側も同じ。NULL キーはどのビルド行とも一致しない。
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
                    // ネストループは次のビルド行へ進むだけ。
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
                assemble(
                    &p.batch.cols,
                    &self.left_types,
                    Some(&lidx),
                    &self.build_cols,
                    &self.right_types,
                    Some(&ridx),
                    lidx.len(),
                )
            };
            // residual は「一致した」と数える**前**に適用する。キーは合ったが
            // residual で落ちたペアは一致ではないので、OUTER ではその左行を
            // NULL 拡張して出さなければならない。
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
            match keep {
                // 候補が全滅した。空バッチは返さず次の塊へ進む。
                Some(sel) if sel.is_empty() => return Ok(None),
                Some(sel) => out.sel = Some(sel),
                None => {}
            }
            return Ok(Some(Step::Ready(out)));
        }

        // --- 候補を出し切った。LEFT/FULL は未一致の左行を NULL 拡張する ------
        let mut idx: Vec<u32> = Vec::new();
        if self.emit_unmatched_left {
            let p = match self.probe.as_mut() {
                Some(p) => p,
                None => err!(Internal),
            };
            let rows = p.batch.num_rows();
            while p.drain < rows && idx.len() < BATCH_SIZE {
                if !p.matched.get(p.drain) {
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
                &self.right_types,
                None,
                idx.len(),
            ))));
        }

        // このバッチは片付いた。次の左バッチを引く。
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
                        // 未一致ビルド行を出すのは RIGHT/FULL だけ。行数ぶんの
                        // ビットマップなので、要らないときは確保しない。
                        if self.emit_unmatched_right {
                            self.build_matched = Bitmap::zeros(self.build_rows);
                        }
                        self.phase = Phase::Probing;
                    }
                    // NeedIo / NeedCodec。作りかけのハッシュ表を保ったまま抜ける。
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

// --- 出力の組み立て ---------------------------------------------------------

/// 左の列に続けて右の列を並べたバッチを作る。添字が `None` の側は全行 NULL
/// （OUTER の NULL 拡張）。列順はバインダが作るスキーマとの契約なので変えない。
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
        // 列を持たない入力同士（COUNT(*) 経路）。行数だけを伝える。
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

/// `Vector::gather` に「相手が居ない」印（`NONE`）を許したもの。
fn gather_opt(src: &Vector, idx: &[u32], ty: Ty) -> Vector {
    if !idx.contains(&NONE) {
        return src.gather(idx);
    }
    if src.is_empty() {
        // ビルド側が空なら拾える行が無い。全行 NULL で返す。
        return null_vector(ty, idx.len());
    }
    // 適当な行を拾ってから validity を落とす。`Value` 経由の行コピーを避ける。
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

/// `dst` の末尾に `src` の全行を連結する。1 行ずつ `Value` を経由すると
/// 可変長列で確保回数が行数に比例するので、バッチ単位でまとめて積む。
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
        // 物理型が食い違うのは上流のバグ。
        _ => err!(Internal),
    }
    if src.has_nulls() || dst.has_nulls() {
        // `validity_mut` は不足分を「有効」で埋めるので、NULL の位置だけ落とす。
        let bm = dst.validity_mut();
        for i in 0..n {
            if !src.is_valid(i) {
                bm.set(base + i, false);
            }
        }
    }
    Ok(())
}

/// バッファ量の概算。上限判定にしか使わないので厳密でなくてよい。
fn vector_bytes(v: &Vector) -> usize {
    let d = match v.data() {
        Data::Bool(b) => b.len() / 8 + 1,
        Data::I32(x) => x.len() * 4,
        Data::I64(x) => x.len() * 8,
        Data::F64(x) => x.len() * 8,
        Data::I128(x) => x.len() * 16,
        Data::Bytes(b) => b.data.len() + (b.len() + 1) * 4,
    };
    // validity と結合用のチェーンぶん。
    d + v.len() / 8 + 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Catalog;
    use crate::expr::vm::Vm;
    use crate::expr::{Instr, OpCode};
    use crate::vector::Value;

    // --- モック入力 ---------------------------------------------------------

    /// 台本どおりに `Step` を返す入力。`NeedIo` を挟むと、リモート入力で
    /// 途中中断された状況をそのまま再現できる。
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

    /// 第 `i` 列をそのままキーにするプログラム。
    fn col_prog(i: u16, ty: Ty) -> Program {
        let mut p = Program::new();
        let r = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, ty.phys(), r, 0, 0, i));
        p.result = r;
        p.result_ty = ty;
        p
    }

    /// `col a <op> col b` を返すプログラム（residual 用）。
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
        /// 返ってきた `NeedIo` の回数。
        interrupts: usize,
        batches: usize,
    }

    fn run(op: &mut dyn Operator) -> Runner {
        let mut cat = Catalog::new();
        let mut vm = Vm::new();
        let mut out = Runner { rows: Vec::new(), interrupts: 0, batches: 0 };
        for guard in 0..100_000 {
            assert!(guard < 99_999, "終わらない");
            let mut ctx =
                ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
            match op.next(&mut ctx).unwrap() {
                Step::Ready(b) => {
                    let n = b.card();
                    assert!(n > 0, "空バッチを返してはいけない");
                    assert!(n <= BATCH_SIZE, "1 回の next で {n} 行はバッチ上限超え");
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

    /// 行を整数（NULL は `None`）に潰して並べ替える。順序は保証しないため。
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

    /// 1 列（INT のキーのみ）同士の結合。
    fn join1(left: Vec<Step>, right: Vec<Step>, kind: JoinKind) -> HashJoin {
        join(left, right, kind, 1, vec![Ty::Int], vec![Ty::Int])
    }

    fn ints1(vals: &[Option<i32>]) -> Step {
        ready(vec![ints(vals)])
    }

    // --- 中断と再開 ---------------------------------------------------------

    /// 最重要。ビルド中・探索中に `NeedIo` が挟まっても結果が変わらないこと。
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
        assert!(got.interrupts >= 5, "中断がそのまま伝わっていない");
        assert_eq!(norm(&got.rows), norm(&clean.rows));

        // INNER / LEFT / RIGHT でも同じ。
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

    /// 1 つの左行が BATCH_SIZE を超える出力を生む場合、探索の途中で中断されても
    /// チェーンの位置を見失わないこと。
    #[test]
    fn need_io_does_not_disturb_a_long_chain() {
        let big: Vec<Option<i32>> = (0..3000).map(|_| Some(7)).collect();
        let left = vec![Step::NeedIo, ints1(&[Some(7)]), Step::NeedIo];
        let right = vec![Step::NeedIo, ready(vec![ints(&big)]), Step::NeedIo];
        let got = run(&mut join1(left, right, JoinKind::Inner));
        assert_eq!(got.rows.len(), 3000);
        assert_eq!(got.batches, 2, "BATCH_SIZE で区切られる");
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

    /// 多対多。チェーンの張り方を間違えると件数が合わない。
    #[test]
    fn inner_many_to_many_row_count() {
        // 左: 1,1,2,2,3 / 右: 1,1,1,2 → 2*3 + 2*1 = 8
        let got = run(&mut join1(
            vec![ints1(&[Some(1), Some(1), Some(2), Some(2), Some(3)])],
            vec![ints1(&[Some(1), Some(1), Some(1), Some(2)])],
            JoinKind::Inner,
        ));
        assert_eq!(got.rows.len(), 8);
        let ones = got.rows.iter().filter(|r| r[0].as_i64() == Some(1)).count();
        assert_eq!(ones, 6);
    }

    /// 右入力が複数バッチに割れていてもチェーンが繋がること。
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

    // --- NULL キー ----------------------------------------------------------

    /// NULL は NULL とも一致しない。ただし OUTER では未一致行として出る。
    #[test]
    fn null_keys_never_match_each_other() {
        let left = || vec![ints1(&[None, Some(1)])];
        let right = || vec![ints1(&[None, Some(1)])];

        let got = run(&mut join1(left(), right(), JoinKind::Inner));
        assert_eq!(norm(&got.rows), vec![vec![Some(1), Some(1)]], "NULL 同士が繋がっている");

        let got = run(&mut join1(left(), right(), JoinKind::Left));
        assert_eq!(norm(&got.rows), vec![vec![None, None], vec![Some(1), Some(1)]]);
        // 左が NULL の行は「左だけ有効」として出ているはず。
        let extended = got.rows.iter().filter(|r| r[0].is_null() && r[1].is_null()).count();
        assert_eq!(extended, 1);

        let got = run(&mut join1(left(), right(), JoinKind::Right));
        assert_eq!(norm(&got.rows), vec![vec![None, None], vec![Some(1), Some(1)]]);

        let got = run(&mut join1(left(), right(), JoinKind::Full));
        // 左の NULL 行と右の NULL 行が別々に 1 行ずつ。
        assert_eq!(got.rows.len(), 3);
        assert_eq!(
            norm(&got.rows),
            vec![vec![None, None], vec![None, None], vec![Some(1), Some(1)]]
        );
    }

    /// 複合キー。片方が NULL なら他方が一致していても繋がらない。
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
                vec![Some(1), None, None, None], // (1, NULL) は何とも一致しない
                vec![Some(1), Some(10), Some(1), Some(10)],
                vec![Some(2), Some(20), None, None], // 第 2 列が違う
            ]
        );
    }

    // --- 型ごとのキー -------------------------------------------------------

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

    /// `encode_key` が -0.0 と NaN を正規化するので、どちらも結合する。
    #[test]
    fn float_keys_canonicalise_zero_and_nan() {
        let left = vec![ready(vec![dbls(&[-0.0, f64::NAN, 1.5])])];
        let right = vec![ready(vec![dbls(&[0.0, f64::NAN])])];
        let mut j = join(left, right, JoinKind::Inner, 1, vec![Ty::Double], vec![Ty::Double]);
        let got = run(&mut j);
        assert_eq!(got.rows.len(), 2, "-0.0=0.0 と NaN=NaN で 2 行");
        let z = got.rows.iter().find(|r| r[0].as_f64() == Some(0.0)).unwrap();
        // 左は -0.0、右は 0.0。ビット列は違うのに結合されている。
        assert_eq!(z[0].as_f64().unwrap().to_bits(), (-0.0f64).to_bits());
        assert_eq!(z[1].as_f64().unwrap().to_bits(), 0.0f64.to_bits());
        let n = got.rows.iter().find(|r| r[0].as_f64().unwrap().is_nan()).unwrap();
        assert!(n[1].as_f64().unwrap().is_nan());
    }

    // --- CROSS / 非等値 -----------------------------------------------------

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

    /// 等値に落ちない述語（`a < b`）。ネストループに落ちること。
    #[test]
    fn non_equi_join_falls_back_to_nested_loop() {
        let mut j = HashJoin::new(
            Mock::script(vec![ints1(&[Some(1), Some(5)])]),
            Mock::script(vec![ints1(&[Some(2), Some(9)])]),
            JoinKind::Inner,
            Vec::new(),
            Vec::new(),
            // 結合後スキーマ: 0 = 左, 1 = 右
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

    /// 非等値 + LEFT。どのビルド行とも通らなかった左行は NULL 拡張される。
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

    /// キーは一致したが residual で落ちたペアは「一致していない」。LEFT では
    /// その左行を NULL 拡張して出す（落として消してはいけない）。
    #[test]
    fn residual_failure_still_yields_null_extended_left_row() {
        // 左 (key, v) / 右 (key, w)、residual: 左の v < 右の w
        let left = vec![ready(vec![ints(&[Some(1), Some(2)]), ints(&[Some(100), Some(0)])])];
        let right = vec![ready(vec![ints(&[Some(1), Some(2)]), ints(&[Some(5), Some(5)])])];
        let ty = vec![Ty::Int, Ty::Int];
        let mut j = HashJoin::new(
            Mock::script(left),
            Mock::script(right),
            JoinKind::Left,
            vec![col_prog(0, Ty::Int)],
            vec![col_prog(0, Ty::Int)],
            // 結合後スキーマ: 0=左key 1=左v 2=右key 3=右w
            Some(cmp_prog(1, 3, Ty::Int, OpCode::Lt)),
            ty.clone(),
            ty,
        )
        .unwrap();
        let got = run(&mut j);
        assert_eq!(
            norm(&got.rows),
            vec![
                vec![Some(1), Some(100), None, None], // 100 < 5 は偽 → NULL 拡張
                vec![Some(2), Some(0), Some(2), Some(5)],
            ]
        );
    }

    /// FULL + residual。落ちたペアは左右どちらの側でも未一致として扱う。
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

    // --- バッチ境界 ---------------------------------------------------------

    /// 1 つの左行が BATCH_SIZE を超える出力を生む。探索の途中で切って続きから
    /// 再開できないと行が落ちる。
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

    /// 未一致行のドレインもバッチに収まる。
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

    // --- 空入力 -------------------------------------------------------------

    #[test]
    fn empty_build_side_for_each_kind() {
        for (kind, expect) in
            [(JoinKind::Inner, 0), (JoinKind::Left, 2), (JoinKind::Right, 0), (JoinKind::Full, 2)]
        {
            let got = run(&mut join1(vec![ints1(&[Some(1), Some(2)])], Vec::new(), kind));
            assert_eq!(got.rows.len(), expect, "{kind:?}");
            for r in &got.rows {
                assert!(r[1].is_null(), "右は NULL 拡張のはず");
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
                assert!(r[0].is_null(), "左は NULL 拡張のはず");
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
