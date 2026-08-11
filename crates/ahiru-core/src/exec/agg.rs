//! ハッシュ集約（GROUP BY / HAVING）。
//!
//! グループキーの符号化とハッシュ表は `exec::rowkey` を使う。集約と結合で
//! 等価判定がずれると静かに壊れるうえ、表を 2 つ持つ理由もないため。
//!
//! ## ブロッキングと再開
//!
//! 集約は入力を最後まで読まないと 1 行も出せない。一方でスキャンは
//! `Step::NeedIo` / `Step::NeedCodec` を返して中断しうるので、**構築の途中
//! 状態を保持したまま中断をそのまま上へ返し、次の `next()` で続きから読む**。
//! 取り込みの単位は 1 バッチで、「丸ごと入れる」か「まだ入れていない」かの
//! どちらかしかない（`consume` は途中で抜けない）。これで再開時の行の
//! 取りこぼしと二重計上を構造的に潰す。
//!
//! ## 溢れ
//!
//! スピル（外部集約）は持たない。ハッシュ表が上限を超えたら `Oom` を返す。
//! 1MiB のバイナリ予算でソート済み実行時のマージまで抱えるのは割に合わない
//! ため、既知の制限として受け入れる。

use crate::exec::rowkey::{encode_key, HashIndex};
use crate::exec::{ExecContext, Operator, Step};
use crate::expr::Program;
use crate::plan::{Agg, AggKind};
use crate::prelude::*;
use crate::vector::{Batch, Data, PhysType, Ty, Value, Vector, BATCH_SIZE};

/// ハッシュ表と状態に許すおおよそのバイト数。超えたら `Oom`。
///
/// ホストが渡す WASM 線形メモリは 32 ビット空間で、実行中はスキャンの
/// バッファと出力バッチも同居する。集約だけで食い潰さない値として 64MiB。
const MAX_STATE_BYTES: usize = 64 << 20;

/// エントリ 1 件あたりの `HashIndex` 側のおおよその固定費
/// （`entries` 12B + `hashes` 8B + バケット分の余裕）。
const INDEX_OVERHEAD: usize = 32;

/// 実行時に選ぶ更新規則。`AggKind` と入力の物理型から一度だけ決める。
#[derive(Clone, Copy, PartialEq)]
enum Op {
    CountStar,
    Count,
    /// 整数・DECIMAL の SUM。i128 で累積する。
    SumInt,
    SumF64,
    /// 整数・DECIMAL の AVG。i128 で累積し、出力時に f64 へ割る。
    AvgInt,
    AvgF64,
    Min,
    Max,
}

/// 1 グループ × 1 集約の状態。
struct State {
    /// 非 NULL 入力の個数。`COUNT(*)` では全行数。
    n: i64,
    /// 累積値。`Value::Null` は「まだ非 NULL 入力が無い」。
    acc: Value,
}

impl State {
    fn empty() -> Self {
        State { n: 0, acc: Value::Null }
    }
}

enum Phase {
    Building,
    Emitting,
    Done,
}

pub struct HashAggregate {
    input: Box<dyn Operator>,
    groups: Vec<Program>,
    aggs: Vec<Agg>,
    having: Option<Program>,

    /// 集約ごとの更新規則。`aggs` と同じ並び。
    ops: Vec<Op>,
    /// 集約ごとの出力型。**必ず `Agg::result_ty()` 由来**（バインダと同じ関数）。
    out_tys: Vec<Ty>,
    /// AVG(DECIMAL) の 10^scale。それ以外は 1.0。
    avg_div: Vec<f64>,

    /// グループキー → グループ番号。番号は 0 から連番で振られる。
    index: HashIndex,
    /// 出力用に保持するグループキーの実体。`groups` と同じ並び、1 行 1 グループ。
    key_cols: Vec<Vector>,
    /// 集約ごとの状態列。`states[agg][group]`。
    states: Vec<Vec<State>>,
    /// DISTINCT 集約の重複除去集合。キーは「グループ番号 ++ 値」。
    distinct: Vec<Option<HashIndex>>,

    /// MIN/MAX が抱えるバイト列の合計。メモリ判定に含める。
    acc_bytes: usize,
    /// グループキー列の実体のおおよそのバイト数。
    key_bytes: usize,

    phase: Phase,
    /// 次に出力するグループ番号。
    emit_pos: usize,
}

impl HashAggregate {
    pub fn new(
        input: Box<dyn Operator>,
        groups: Vec<Program>,
        aggs: Vec<Agg>,
        having: Option<Program>,
    ) -> Result<Self> {
        let mut ops = Vec::with_capacity(aggs.len());
        let mut out_tys = Vec::with_capacity(aggs.len());
        let mut avg_div = Vec::with_capacity(aggs.len());
        let mut states = Vec::with_capacity(aggs.len());
        let mut distinct = Vec::with_capacity(aggs.len());
        for a in &aggs {
            let ity = a.input_ty();
            // 型検査はここで済ませる。出力型は必ずバインダと同じ関数から取る。
            out_tys.push(a.result_ty()?);
            let float = ity.phys() == PhysType::F64;
            ops.push(match a.kind {
                AggKind::CountStar => Op::CountStar,
                AggKind::Count => Op::Count,
                AggKind::Sum => {
                    if float {
                        Op::SumF64
                    } else {
                        Op::SumInt
                    }
                }
                AggKind::Avg => {
                    if float {
                        Op::AvgF64
                    } else {
                        Op::AvgInt
                    }
                }
                AggKind::Min => Op::Min,
                AggKind::Max => Op::Max,
            });
            // DECIMAL は内部が整数なので、AVG では 10^scale で戻す。
            avg_div.push(match ity {
                Ty::Decimal { scale, .. } => pow10(scale),
                _ => 1.0,
            });
            states.push(Vec::new());
            // `COUNT(*)` に DISTINCT は付かない（引数が無い）。
            distinct.push(if a.distinct && a.arg.is_some() {
                Some(HashIndex::new())
            } else {
                None
            });
        }
        let key_cols = groups.iter().map(|g| Vector::new(g.result_ty)).collect();
        Ok(HashAggregate {
            input,
            groups,
            aggs,
            having,
            ops,
            out_tys,
            avg_div,
            index: HashIndex::new(),
            key_cols,
            states,
            distinct,
            acc_bytes: 0,
            key_bytes: 0,
            phase: Phase::Building,
            emit_pos: 0,
        })
    }

    fn num_groups(&self) -> usize {
        self.index.len()
    }

    /// おおよそのメモリ使用量。厳密な値は要らない（上限判定にしか使わない）。
    fn mem_used(&self) -> usize {
        let mut n = self.index.key_bytes() + self.index.len() * INDEX_OVERHEAD;
        n += self.key_bytes + self.acc_bytes;
        n += self.num_groups() * self.aggs.len() * core::mem::size_of::<State>();
        for d in self.distinct.iter().flatten() {
            n += d.key_bytes() + d.len() * INDEX_OVERHEAD;
        }
        n
    }

    /// 1 バッチを丸ごと取り込む。**途中で抜けない**（再開の単位はバッチ）。
    fn consume(&mut self, ctx: &mut ExecContext, batch: &Batch) -> Result<()> {
        let rows = batch.card();
        if rows == 0 {
            return Ok(());
        }
        // VM の結果は selection 解決済みの密ベクタなので、行番号は 0..rows。
        let mut gvs = Vec::with_capacity(self.groups.len());
        for (i, p) in self.groups.iter().enumerate() {
            let v = ctx.vm.eval(p, batch)?;
            // 宣言された型と実データがずれると出力列が壊れる。契約違反を検出する。
            ensure!(v.data().phys() == self.key_cols[i].ty().phys(), Internal);
            gvs.push(v);
        }
        let mut avs = Vec::with_capacity(self.aggs.len());
        for a in &self.aggs {
            avs.push(match &a.arg {
                Some(p) => Some(ctx.vm.eval(p, batch)?),
                None => None,
            });
        }

        let refs: Vec<&Vector> = gvs.iter().collect();
        let mut key = Vec::new();
        let mut dkey = Vec::new();
        let mut vkey = Vec::new();
        for row in 0..rows {
            encode_key(&refs, row, &mut key);
            let (slot, is_new) = self.index.get_or_insert(&key);
            if is_new {
                self.key_bytes += key.len();
                for (i, c) in self.key_cols.iter_mut().enumerate() {
                    c.push_value(&gvs[i].value_at(row));
                }
                for s in self.states.iter_mut() {
                    s.push(State::empty());
                }
            }
            let g = slot as usize;

            for (ai, av) in avs.iter().enumerate() {
                if self.ops[ai] == Op::CountStar {
                    // COUNT(*) は NULL だけの行も数える。
                    self.states[ai][g].n += 1;
                    continue;
                }
                let col = match av {
                    Some(c) => c,
                    // COUNT(*) 以外は必ず引数を持つ（`Agg` の契約）。
                    None => err!(Internal),
                };
                // SUM/MIN/MAX/AVG/COUNT(x) は NULL を無視する。
                if !col.is_valid(row) {
                    continue;
                }
                if let Some(seen) = &mut self.distinct[ai] {
                    // グループ番号を前置して 1 本の表で全グループを賄う。
                    // ネストした表を持たずに済むが、(グループ, 値) の組の数
                    // だけキーが残るのでメモリはその分増える。
                    encode_key(&[col], row, &mut vkey);
                    dkey.clear();
                    dkey.extend_from_slice(&slot.to_le_bytes());
                    dkey.extend_from_slice(&vkey);
                    if !seen.get_or_insert(&dkey).1 {
                        continue;
                    }
                }
                self.update(ai, g, col, row)?;
            }
        }

        ensure!(self.mem_used() <= MAX_STATE_BYTES, Oom);
        Ok(())
    }

    /// 非 NULL・重複除去済みの 1 値を状態へ畳み込む。
    fn update(&mut self, ai: usize, g: usize, col: &Vector, row: usize) -> Result<()> {
        let op = self.ops[ai];
        let st = &mut self.states[ai][g];
        st.n += 1;
        match op {
            Op::CountStar | Op::Count => {}
            Op::SumInt | Op::AvgInt => {
                let x = as_i128(col, row)?;
                let sum = match &st.acc {
                    Value::I128(s) => match s.checked_add(x) {
                        Some(v) => v,
                        // i128 でも溢れる合計は黙って巻き戻さずエラーにする。
                        None => err!(ValueOutOfRange),
                    },
                    _ => x,
                };
                st.acc = Value::I128(sum);
            }
            Op::SumF64 | Op::AvgF64 => {
                let x = as_f64(col, row)?;
                let sum = match &st.acc {
                    Value::F64(s) => s + x,
                    _ => x,
                };
                st.acc = Value::F64(sum);
            }
            Op::Min | Op::Max => {
                let take = match &st.acc {
                    Value::Null => true,
                    acc => {
                        let c = cmp_at(col, row, acc);
                        if op == Op::Min {
                            c.is_lt()
                        } else {
                            c.is_gt()
                        }
                    }
                };
                if take {
                    if let Value::Bytes(b) = &st.acc {
                        self.acc_bytes = self.acc_bytes.saturating_sub(b.len());
                    }
                    let v = col.value_at(row);
                    if let Value::Bytes(b) = &v {
                        self.acc_bytes += b.len();
                    }
                    self.states[ai][g].acc = v;
                }
            }
        }
        Ok(())
    }

    /// 1 グループぶんの集約結果を出力ベクタへ積む。
    fn push_result(&self, ai: usize, g: usize, out: &mut Vector) {
        let st = &self.states[ai][g];
        match self.ops[ai] {
            Op::CountStar | Op::Count => out.push_value(&Value::I64(st.n)),
            // 非 NULL 入力が 1 つも無いグループは NULL。
            Op::SumInt | Op::SumF64 | Op::Min | Op::Max => out.push_value(&st.acc),
            Op::AvgInt => {
                // 整数は i128 で正確に足してから 1 回だけ割る。f64 で足し込むと
                // 桁落ちが累積するため。
                match &st.acc {
                    Value::I128(s) if st.n > 0 => {
                        out.push_value(&Value::F64(*s as f64 / self.avg_div[ai] / st.n as f64))
                    }
                    _ => out.push_null(),
                }
            }
            Op::AvgF64 => match &st.acc {
                Value::F64(s) if st.n > 0 => out.push_value(&Value::F64(s / st.n as f64)),
                _ => out.push_null(),
            },
        }
    }

    fn emit(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        let total = self.num_groups();
        while self.emit_pos < total {
            let start = self.emit_pos;
            let end = (start + BATCH_SIZE).min(total);
            let idx: Vec<u32> = (start as u32..end as u32).collect();
            // 出力はグループ列が先、集約列が後（バインダのスキーマと同じ並び）。
            let mut cols = Vec::with_capacity(self.key_cols.len() + self.ops.len());
            for c in &self.key_cols {
                cols.push(c.gather(&idx));
            }
            for ai in 0..self.ops.len() {
                let mut v = Vector::with_capacity(self.out_tys[ai], end - start);
                for g in start..end {
                    self.push_result(ai, g, &mut v);
                }
                v.compact_validity();
                cols.push(v);
            }
            self.emit_pos = end;

            let mut batch =
                if cols.is_empty() { Batch::rows_only(end - start) } else { Batch::new(cols) };
            if let Some(h) = &self.having {
                // HAVING は集約後のスキーマで評価する。
                let mut sel = Vec::new();
                ctx.vm.eval_filter(h, &batch, &mut sel)?;
                if sel.is_empty() {
                    continue;
                }
                batch.sel = Some(sel);
            }
            return Ok(Step::Ready(batch));
        }
        self.phase = Phase::Done;
        Ok(Step::Done)
    }
}

impl Operator for HashAggregate {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        loop {
            match self.phase {
                Phase::Building => match self.input.next(ctx)? {
                    Step::Ready(b) => self.consume(ctx, &b)?,
                    // 途中状態をそのまま残して素通しする。次回はここから再開。
                    // 取り込みはバッチ単位なので、中断しても行の取りこぼしも
                    // 二重計上も起きない。
                    other @ (Step::NeedIo | Step::NeedCodec) => return Ok(other),
                    Step::Done => {
                        // GROUP BY が無い集約は入力が空でも 1 行返す
                        // （COUNT(*) は 0、それ以外は NULL）。GROUP BY がある
                        // 場合は 0 行。この非対称は SQL の規定どおり。
                        if self.groups.is_empty() && self.index.is_empty() {
                            self.index.get_or_insert(&[]);
                            for s in self.states.iter_mut() {
                                s.push(State::empty());
                            }
                        }
                        self.phase = Phase::Emitting;
                    }
                },
                Phase::Emitting => return self.emit(ctx),
                Phase::Done => return Ok(Step::Done),
            }
        }
    }
}

/// 10^scale。`f64::powi` は core に無いので掛け算で作る。
fn pow10(scale: u8) -> f64 {
    let mut d = 1.0f64;
    for _ in 0..scale {
        d *= 10.0;
    }
    d
}

/// 整数系の 1 値を i128 で取り出す。SUM の桁溢れを避けるため常に広げる。
fn as_i128(col: &Vector, row: usize) -> Result<i128> {
    Ok(match col.data() {
        Data::I32(v) => v[row] as i128,
        Data::I64(v) => v[row] as i128,
        Data::I128(v) => v[row],
        _ => err!(TypeMismatch),
    })
}

fn as_f64(col: &Vector, row: usize) -> Result<f64> {
    Ok(match col.data() {
        Data::F64(v) => v[row],
        _ => err!(TypeMismatch),
    })
}

/// 列の row 行目と累積値の比較。物理型が違う組み合わせは呼び出し側のバグ。
fn cmp_at(col: &Vector, row: usize, acc: &Value) -> core::cmp::Ordering {
    use core::cmp::Ordering;
    match (col.data(), acc) {
        (Data::Bool(b), Value::Bool(x)) => b.get(row).cmp(x),
        (Data::I32(v), Value::I32(x)) => v[row].cmp(x),
        (Data::I64(v), Value::I64(x)) => v[row].cmp(x),
        (Data::I128(v), Value::I128(x)) => v[row].cmp(x),
        (Data::F64(v), Value::F64(x)) => ord_f64(v[row], *x),
        (Data::Bytes(b), Value::Bytes(x)) => b.get(row).cmp(x.as_slice()),
        _ => Ordering::Equal,
    }
}

/// NaN を「すべてより大きい」とみなす全順序。
///
/// こうすると MAX は他に値が無いときだけ NaN を返し、MIN は NaN 以外を
/// 優先する（全部 NaN のグループだけ NaN になる）。`rowkey` が NaN を
/// 1 グループにまとめるのと矛盾しない。
fn ord_f64(a: f64, b: f64) -> core::cmp::Ordering {
    use core::cmp::Ordering::*;
    if a < b {
        Less
    } else if a > b {
        Greater
    } else if a == b {
        Equal
    } else {
        match (a.is_nan(), b.is_nan()) {
            (true, true) => Equal,
            (true, false) => Greater,
            _ => Less,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Catalog;
    use crate::error::code_of;
    use crate::expr::vm::Vm;
    use crate::expr::{Instr, OpCode};

    // --- 組み立てヘルパ -----------------------------------------------------

    fn col(ty: Ty, vals: &[Option<Value>]) -> Vector {
        let mut v = Vector::new(ty);
        for x in vals {
            match x {
                Some(x) => v.push_value(x),
                None => v.push_null(),
            }
        }
        v
    }

    fn i32s(vals: &[Option<i32>]) -> Vector {
        col(Ty::Int, &vals.iter().map(|v| v.map(Value::I32)).collect::<Vec<_>>())
    }

    fn i64s(vals: &[Option<i64>]) -> Vector {
        col(Ty::BigInt, &vals.iter().map(|v| v.map(Value::I64)).collect::<Vec<_>>())
    }

    fn f64s(vals: &[Option<f64>]) -> Vector {
        col(Ty::Double, &vals.iter().map(|v| v.map(Value::F64)).collect::<Vec<_>>())
    }

    fn strs(vals: &[Option<&str>]) -> Vector {
        col(
            Ty::Varchar,
            &vals
                .iter()
                .map(|v| v.map(|s| Value::Bytes(s.as_bytes().to_vec())))
                .collect::<Vec<_>>(),
        )
    }

    /// 入力バッチの `idx` 列をそのまま返すプログラム。
    fn load(ty: Ty, idx: u16) -> Program {
        let mut p = Program::new();
        let r = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, ty.phys(), r, 0, 0, idx));
        p.result = r;
        p.result_ty = ty;
        p
    }

    /// `col[idx] <op> const` の述語。HAVING 用。
    fn cmp_const(op: OpCode, ty: Ty, idx: u16, v: Value) -> Program {
        let mut p = Program::new();
        let r0 = p.alloc_reg();
        let r1 = p.alloc_reg();
        let r2 = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, ty.phys(), r0, 0, 0, idx));
        let c = p.add_const(ty, v);
        p.push(Instr::with_aux(OpCode::LoadConst, ty.phys(), r1, 0, 0, c));
        p.push(Instr::new(op, ty.phys(), r2, r0, r1));
        p.result = r2;
        p.result_ty = Ty::Boolean;
        p
    }

    fn agg(kind: AggKind, arg: Option<Program>) -> Agg {
        Agg { kind, arg, distinct: false, name: String::from("a") }
    }

    fn agg_distinct(kind: AggKind, arg: Program) -> Agg {
        Agg { kind, arg: Some(arg), distinct: true, name: String::from("a") }
    }

    // --- モックオペレータ ---------------------------------------------------

    enum MockStep {
        Rows(Batch),
        /// 入力の途中で I/O 待ちになる場面を再現する。
        NeedIo,
        /// ホストにコーデック展開を頼んで待つ場面。
        NeedCodec,
    }

    struct Mock {
        steps: Vec<MockStep>,
        pos: usize,
        /// Done を返したあとに呼ばれた回数。二重消費の検出用。
        after_done: usize,
    }

    impl Mock {
        fn new(steps: Vec<MockStep>) -> Box<Mock> {
            Box::new(Mock { steps, pos: 0, after_done: 0 })
        }
    }

    impl Operator for Mock {
        fn next(&mut self, _ctx: &mut ExecContext) -> Result<Step> {
            if self.pos >= self.steps.len() {
                self.after_done += 1;
                return Ok(Step::Done);
            }
            let i = self.pos;
            self.pos += 1;
            Ok(match &mut self.steps[i] {
                MockStep::NeedIo => Step::NeedIo,
                MockStep::NeedCodec => Step::NeedCodec,
                MockStep::Rows(b) => Step::Ready(core::mem::replace(b, Batch::rows_only(0))),
            })
        }
    }

    fn batches(cols: Vec<Vec<Vector>>) -> Vec<MockStep> {
        cols.into_iter().map(|c| MockStep::Rows(Batch::new(c))).collect()
    }

    // --- 実行ヘルパ ---------------------------------------------------------

    /// 出力を行の並びに落とす。`NeedIo` はホストが埋めた体で単に再開する。
    fn run(mut op: HashAggregate) -> Result<Vec<Vec<Value>>> {
        let mut catalog = Catalog::new();
        let mut vm = Vm::new();
        let mut rows = Vec::new();
        let mut guard = 0;
        loop {
            guard += 1;
            assert!(guard < 100_000, "終わらない");
            let mut ctx = ExecContext {
                catalog: &mut catalog,
                vm: &mut vm,
                io: Vec::new(),
                codec: Vec::new(),
            };
            match op.next(&mut ctx)? {
                Step::Ready(b) => {
                    let n = b.card();
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
        Ok(rows)
    }

    fn build(
        steps: Vec<MockStep>,
        groups: Vec<Program>,
        aggs: Vec<Agg>,
        having: Option<Program>,
    ) -> HashAggregate {
        HashAggregate::new(Mock::new(steps), groups, aggs, having).unwrap()
    }

    /// 出力行をキー列の文字列表現でソートして比較しやすくする。
    fn sorted(mut rows: Vec<Vec<Value>>) -> Vec<Vec<Value>> {
        rows.sort_by_key(|r| fmt_row(r));
        rows
    }

    fn fmt_row(r: &[Value]) -> String {
        let mut s = String::new();
        for v in r {
            s.push_str(&format!("{v:?}|"));
        }
        s
    }

    // --- 型と基本形 ---------------------------------------------------------

    #[test]
    fn result_types_come_from_agg_result_ty() {
        // SUM(INT) は HUGEINT、AVG は DOUBLE、COUNT は BIGINT。
        let a = HashAggregate::new(
            Mock::new(vec![]),
            vec![],
            vec![
                agg(AggKind::CountStar, None),
                agg(AggKind::Sum, Some(load(Ty::Int, 0))),
                agg(AggKind::Avg, Some(load(Ty::Int, 0))),
                agg(AggKind::Min, Some(load(Ty::Varchar, 0))),
            ],
            None,
        )
        .unwrap();
        assert_eq!(a.out_tys, vec![Ty::BigInt, Ty::HugeInt, Ty::Double, Ty::Varchar]);
    }

    #[test]
    fn invalid_input_type_is_rejected_at_construction() {
        let r = HashAggregate::new(
            Mock::new(vec![]),
            vec![],
            vec![agg(AggKind::Sum, Some(load(Ty::Varchar, 0)))],
            None,
        );
        assert_eq!(code_of(r.map(|_| ())), Some(Code::TypeMismatch));
    }

    #[test]
    fn ungrouped_count_sum_min_max_avg() {
        let steps = batches(vec![
            vec![i32s(&[Some(1), Some(2), None]), strs(&[Some("b"), Some("a"), Some("c")])],
            vec![i32s(&[Some(4)]), strs(&[Some("z")])],
        ]);
        let op = build(
            steps,
            vec![],
            vec![
                agg(AggKind::CountStar, None),
                agg(AggKind::Count, Some(load(Ty::Int, 0))),
                agg(AggKind::Sum, Some(load(Ty::Int, 0))),
                agg(AggKind::Min, Some(load(Ty::Int, 0))),
                agg(AggKind::Max, Some(load(Ty::Int, 0))),
                agg(AggKind::Avg, Some(load(Ty::Int, 0))),
                agg(AggKind::Min, Some(load(Ty::Varchar, 1))),
                agg(AggKind::Max, Some(load(Ty::Varchar, 1))),
            ],
            None,
        );
        let rows = run(op).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::I64(4), "COUNT(*) は NULL 行も数える");
        assert_eq!(rows[0][1], Value::I64(3), "COUNT(x) は NULL を数えない");
        assert_eq!(rows[0][2], Value::I128(7));
        assert_eq!(rows[0][3], Value::I32(1));
        assert_eq!(rows[0][4], Value::I32(4));
        assert_eq!(rows[0][5], Value::F64(7.0 / 3.0));
        assert_eq!(rows[0][6], Value::Bytes(b"a".to_vec()));
        assert_eq!(rows[0][7], Value::Bytes(b"z".to_vec()));
    }

    #[test]
    fn grouped_aggregates() {
        let steps = batches(vec![vec![
            strs(&[Some("a"), Some("b"), Some("a"), Some("b"), Some("a")]),
            i32s(&[Some(1), Some(10), Some(2), None, Some(3)]),
        ]]);
        let op = build(
            steps,
            vec![load(Ty::Varchar, 0)],
            vec![
                agg(AggKind::CountStar, None),
                agg(AggKind::Count, Some(load(Ty::Int, 1))),
                agg(AggKind::Sum, Some(load(Ty::Int, 1))),
                agg(AggKind::Min, Some(load(Ty::Int, 1))),
                agg(AggKind::Max, Some(load(Ty::Int, 1))),
                agg(AggKind::Avg, Some(load(Ty::Int, 1))),
            ],
            None,
        );
        let rows = sorted(run(op).unwrap());
        assert_eq!(rows.len(), 2);
        // グループ列が先、集約列が後。
        assert_eq!(rows[0][0], Value::Bytes(b"a".to_vec()));
        assert_eq!(rows[0][1], Value::I64(3));
        assert_eq!(rows[0][2], Value::I64(3));
        assert_eq!(rows[0][3], Value::I128(6));
        assert_eq!(rows[0][4], Value::I32(1));
        assert_eq!(rows[0][5], Value::I32(3));
        assert_eq!(rows[0][6], Value::F64(2.0));
        assert_eq!(rows[1][0], Value::Bytes(b"b".to_vec()));
        assert_eq!(rows[1][1], Value::I64(2));
        assert_eq!(rows[1][2], Value::I64(1));
        assert_eq!(rows[1][3], Value::I128(10));
    }

    // --- 再開（これが本命） -------------------------------------------------

    #[test]
    fn need_io_in_the_middle_resumes_with_identical_result() {
        let data: Vec<Vec<Option<i32>>> =
            vec![vec![Some(1), Some(2), None], vec![Some(3), Some(1)], vec![Some(2), Some(9)]];
        let keys: Vec<Vec<Option<i32>>> =
            vec![vec![Some(0), Some(1), Some(0)], vec![Some(1), Some(0)], vec![Some(1), Some(0)]];

        let make = |interrupt: bool| {
            let mut steps = Vec::new();
            for (i, (k, v)) in keys.iter().zip(data.iter()).enumerate() {
                if interrupt && i == 1 {
                    // バッチとバッチの間、かつ入力の途中で I/O 待ちにする。
                    steps.push(MockStep::NeedIo);
                }
                steps.push(MockStep::Rows(Batch::new(vec![i32s(k), i32s(v)])));
                if interrupt && i == 1 {
                    steps.push(MockStep::NeedIo);
                }
            }
            build(
                steps,
                vec![load(Ty::Int, 0)],
                vec![
                    agg(AggKind::CountStar, None),
                    agg(AggKind::Count, Some(load(Ty::Int, 1))),
                    agg(AggKind::Sum, Some(load(Ty::Int, 1))),
                    agg(AggKind::Max, Some(load(Ty::Int, 1))),
                ],
                None,
            )
        };
        let plain = sorted(run(make(false)).unwrap());
        let interrupted = sorted(run(make(true)).unwrap());
        assert_eq!(plain.len(), 2);
        assert_eq!(plain, interrupted, "NeedIo をまたいでも結果が変わってはいけない");
        // 中身も直接確認（行が消えても増えても気づけるように）。
        // key=0 の値は 1, NULL, 1, 9 / key=1 の値は 2, 3, 2。
        assert_eq!(plain[0][1], Value::I64(4)); // COUNT(*)
        assert_eq!(plain[0][2], Value::I64(3)); // COUNT(x) は NULL を除く
        assert_eq!(plain[0][3], Value::I128(1 + 1 + 9));
        assert_eq!(plain[0][4], Value::I32(9));
        assert_eq!(plain[1][1], Value::I64(3));
        assert_eq!(plain[1][3], Value::I128(2 + 3 + 2));
        assert_eq!(plain[1][4], Value::I32(3));
    }

    #[test]
    fn need_codec_is_passed_through_and_resumes() {
        // ホストにコーデック展開を頼む中断も NeedIo と同じ扱い。
        let mut steps = vec![MockStep::Rows(Batch::new(vec![i32s(&[Some(1), Some(2)])]))];
        steps.push(MockStep::NeedCodec);
        steps.push(MockStep::Rows(Batch::new(vec![i32s(&[Some(3)])])));
        let op = build(
            steps,
            vec![],
            vec![agg(AggKind::CountStar, None), agg(AggKind::Sum, Some(load(Ty::Int, 0)))],
            None,
        );
        let rows = run(op).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::I64(3));
        assert_eq!(rows[0][1], Value::I128(6));
    }

    #[test]
    fn need_io_before_any_input() {
        let mut steps = vec![MockStep::NeedIo, MockStep::NeedIo];
        steps.extend(batches(vec![vec![i32s(&[Some(5), Some(6)])]]));
        let op = build(steps, vec![], vec![agg(AggKind::Sum, Some(load(Ty::Int, 0)))], None);
        let rows = run(op).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::I128(11));
    }

    #[test]
    fn done_is_sticky_and_input_is_not_reconsumed() {
        let steps = batches(vec![vec![i32s(&[Some(1), Some(2)])]]);
        let mut op = build(steps, vec![], vec![agg(AggKind::CountStar, None)], None);
        let mut catalog = Catalog::new();
        let mut vm = Vm::new();
        let mut first = None;
        for _ in 0..5 {
            let mut ctx = ExecContext {
                catalog: &mut catalog,
                vm: &mut vm,
                io: Vec::new(),
                codec: Vec::new(),
            };
            match op.next(&mut ctx).unwrap() {
                Step::Ready(b) => {
                    assert!(first.is_none(), "出力バッチは 1 つだけ");
                    first = Some(b.cols[0].value_at(0));
                }
                Step::NeedIo | Step::NeedCodec => panic!("中断は起きないはず"),
                Step::Done => {}
            }
        }
        assert_eq!(first, Some(Value::I64(2)));
    }

    // --- 空入力と全 NULL ----------------------------------------------------

    #[test]
    fn empty_input_ungrouped_emits_one_row() {
        let op = build(
            vec![],
            vec![],
            vec![
                agg(AggKind::CountStar, None),
                agg(AggKind::Count, Some(load(Ty::Int, 0))),
                agg(AggKind::Sum, Some(load(Ty::Int, 0))),
                agg(AggKind::Min, Some(load(Ty::Int, 0))),
                agg(AggKind::Max, Some(load(Ty::Int, 0))),
                agg(AggKind::Avg, Some(load(Ty::Int, 0))),
            ],
            None,
        );
        let rows = run(op).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::I64(0));
        assert_eq!(rows[0][1], Value::I64(0));
        for v in &rows[0][2..] {
            assert_eq!(*v, Value::Null);
        }
    }

    #[test]
    fn empty_input_grouped_emits_nothing() {
        let op = build(vec![], vec![load(Ty::Int, 0)], vec![agg(AggKind::CountStar, None)], None);
        assert!(run(op).unwrap().is_empty());
    }

    #[test]
    fn empty_batches_are_ignored() {
        // card()==0 のバッチが混ざっても 1 行の結果は変わらない。
        let steps = vec![
            MockStep::Rows(Batch::new(vec![i32s(&[])])),
            MockStep::Rows(Batch::new(vec![i32s(&[Some(4)])])),
        ];
        let op = build(steps, vec![], vec![agg(AggKind::CountStar, None)], None);
        let rows = run(op).unwrap();
        assert_eq!(rows[0][0], Value::I64(1));
    }

    #[test]
    fn all_null_group_counts_zero_and_sums_null() {
        let steps = batches(vec![vec![
            strs(&[Some("a"), Some("a"), Some("b")]),
            i32s(&[None, None, Some(7)]),
        ]]);
        let op = build(
            steps,
            vec![load(Ty::Varchar, 0)],
            vec![
                agg(AggKind::CountStar, None),
                agg(AggKind::Count, Some(load(Ty::Int, 1))),
                agg(AggKind::Sum, Some(load(Ty::Int, 1))),
                agg(AggKind::Avg, Some(load(Ty::Int, 1))),
                agg(AggKind::Min, Some(load(Ty::Int, 1))),
            ],
            None,
        );
        let rows = sorted(run(op).unwrap());
        assert_eq!(rows[0][1], Value::I64(2), "COUNT(*) は 2");
        assert_eq!(rows[0][2], Value::I64(0), "COUNT(x) は 0");
        assert_eq!(rows[0][3], Value::Null);
        assert_eq!(rows[0][4], Value::Null);
        assert_eq!(rows[0][5], Value::Null);
    }

    // --- グループキー -------------------------------------------------------

    #[test]
    fn null_keys_form_their_own_group() {
        let steps = batches(vec![vec![
            i32s(&[None, Some(1), None, Some(1)]),
            i32s(&[Some(1), Some(2), Some(3), Some(4)]),
        ]]);
        let op = build(
            steps,
            vec![load(Ty::Int, 0)],
            vec![agg(AggKind::CountStar, None), agg(AggKind::Sum, Some(load(Ty::Int, 1)))],
            None,
        );
        let rows = sorted(run(op).unwrap());
        assert_eq!(rows.len(), 2);
        // NULL グループと 1 のグループがそれぞれ 2 行ずつ。
        let null_row = rows.iter().find(|r| r[0] == Value::Null).unwrap();
        assert_eq!(null_row[1], Value::I64(2));
        assert_eq!(null_row[2], Value::I128(4));
    }

    #[test]
    fn multiple_group_columns() {
        let steps = batches(vec![vec![
            strs(&[Some("a"), Some("a"), Some("b"), Some("a")]),
            i32s(&[Some(1), Some(2), Some(1), Some(1)]),
        ]]);
        let op = build(
            steps,
            vec![load(Ty::Varchar, 0), load(Ty::Int, 1)],
            vec![agg(AggKind::CountStar, None)],
            None,
        );
        let rows = sorted(run(op).unwrap());
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0][0], Value::Bytes(b"a".to_vec()));
        assert_eq!(rows[0][1], Value::I32(1));
        assert_eq!(rows[0][2], Value::I64(2));
    }

    #[test]
    fn float_group_key_merges_zeros_and_nans() {
        let steps = batches(vec![vec![
            f64s(&[Some(0.0), Some(-0.0), Some(f64::NAN), Some(-f64::NAN), Some(1.5)]),
            i32s(&[Some(1), Some(1), Some(1), Some(1), Some(1)]),
        ]]);
        let op = build(steps, vec![load(Ty::Double, 0)], vec![agg(AggKind::CountStar, None)], None);
        let rows = run(op).unwrap();
        assert_eq!(rows.len(), 3, "0.0/-0.0 と NaN/-NaN はそれぞれ 1 グループ");
        for r in &rows {
            let n = r[1].as_i64().unwrap();
            assert!(n == 2 || n == 1);
        }
        let total: i64 = rows.iter().map(|r| r[1].as_i64().unwrap()).sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn min_max_treat_nan_as_largest() {
        let steps = batches(vec![vec![
            i32s(&[Some(0), Some(0), Some(0), Some(1)]),
            f64s(&[Some(1.0), Some(f64::NAN), Some(-2.0), Some(f64::NAN)]),
        ]]);
        let op = build(
            steps,
            vec![load(Ty::Int, 0)],
            vec![
                agg(AggKind::Min, Some(load(Ty::Double, 1))),
                agg(AggKind::Max, Some(load(Ty::Double, 1))),
            ],
            None,
        );
        let rows = sorted(run(op).unwrap());
        assert_eq!(rows[0][1], Value::F64(-2.0));
        assert!(rows[0][2].as_f64().unwrap().is_nan(), "MAX は NaN");
        // 全部 NaN のグループは MIN も NaN。
        assert!(rows[1][1].as_f64().unwrap().is_nan());
    }

    #[test]
    fn min_max_over_bools() {
        let steps = batches(vec![vec![col(
            Ty::Boolean,
            &[Some(Value::Bool(true)), Some(Value::Bool(false)), None],
        )]]);
        let op = build(
            steps,
            vec![],
            vec![
                agg(AggKind::Min, Some(load(Ty::Boolean, 0))),
                agg(AggKind::Max, Some(load(Ty::Boolean, 0))),
            ],
            None,
        );
        let rows = run(op).unwrap();
        assert_eq!(rows[0][0], Value::Bool(false));
        assert_eq!(rows[0][1], Value::Bool(true));
    }

    // --- 桁溢れ -------------------------------------------------------------

    #[test]
    fn sum_of_i64_accumulates_in_i128() {
        let big = i64::MAX;
        let steps = batches(vec![vec![i64s(&[Some(big), Some(big), Some(big)])]]);
        let op = build(steps, vec![], vec![agg(AggKind::Sum, Some(load(Ty::BigInt, 0)))], None);
        let rows = run(op).unwrap();
        assert_eq!(rows[0][0], Value::I128(big as i128 * 3), "i64 では溢れる合計");
    }

    #[test]
    fn sum_overflowing_i128_errors() {
        let huge = col(Ty::HugeInt, &[Some(Value::I128(i128::MAX)), Some(Value::I128(i128::MAX))]);
        let steps = vec![MockStep::Rows(Batch::new(vec![huge]))];
        let op = build(steps, vec![], vec![agg(AggKind::Sum, Some(load(Ty::HugeInt, 0)))], None);
        assert_eq!(code_of(run(op)), Some(Code::ValueOutOfRange));
    }

    // --- DISTINCT -----------------------------------------------------------

    #[test]
    fn distinct_aggregates() {
        let steps = batches(vec![
            vec![
                strs(&[Some("a"), Some("a"), Some("a"), Some("b")]),
                i32s(&[Some(1), Some(1), Some(2), Some(5)]),
            ],
            vec![strs(&[Some("a"), Some("b")]), i32s(&[Some(2), Some(5)])],
        ]);
        let op = build(
            steps,
            vec![load(Ty::Varchar, 0)],
            vec![
                agg_distinct(AggKind::Count, load(Ty::Int, 1)),
                agg_distinct(AggKind::Sum, load(Ty::Int, 1)),
                agg(AggKind::Count, Some(load(Ty::Int, 1))),
                agg(AggKind::Sum, Some(load(Ty::Int, 1))),
                agg_distinct(AggKind::Avg, load(Ty::Int, 1)),
            ],
            None,
        );
        let rows = sorted(run(op).unwrap());
        // a: 値は 1,1,2,2 → distinct は {1,2}
        assert_eq!(rows[0][1], Value::I64(2));
        assert_eq!(rows[0][2], Value::I128(3));
        assert_eq!(rows[0][3], Value::I64(4), "非 DISTINCT は重複を数える");
        assert_eq!(rows[0][4], Value::I128(6));
        assert_eq!(rows[0][5], Value::F64(1.5));
        // b: 値は 5,5 → distinct は {5}
        assert_eq!(rows[1][1], Value::I64(1));
        assert_eq!(rows[1][2], Value::I128(5));
    }

    #[test]
    fn distinct_dedup_is_per_group() {
        let steps = batches(vec![vec![
            i32s(&[Some(0), Some(1), Some(0), Some(1)]),
            i32s(&[Some(7), Some(7), Some(7), Some(7)]),
        ]]);
        let op = build(
            steps,
            vec![load(Ty::Int, 0)],
            vec![agg_distinct(AggKind::Count, load(Ty::Int, 1))],
            None,
        );
        let rows = sorted(run(op).unwrap());
        assert_eq!(rows.len(), 2);
        // 同じ値 7 でもグループが違えば別々に数える。
        assert_eq!(rows[0][1], Value::I64(1));
        assert_eq!(rows[1][1], Value::I64(1));
    }

    #[test]
    fn distinct_ignores_nulls() {
        let steps = batches(vec![vec![i32s(&[None, None, Some(3), Some(3)])]]);
        let op = build(steps, vec![], vec![agg_distinct(AggKind::Count, load(Ty::Int, 0))], None);
        let rows = run(op).unwrap();
        assert_eq!(rows[0][0], Value::I64(1));
    }

    // --- HAVING -------------------------------------------------------------

    #[test]
    fn having_filters_groups() {
        let steps =
            batches(vec![vec![i32s(&[Some(1), Some(1), Some(2), Some(3), Some(3), Some(3)])]]);
        // HAVING COUNT(*) > 1。出力スキーマは [key, count]。
        let op = build(
            steps,
            vec![load(Ty::Int, 0)],
            vec![agg(AggKind::CountStar, None)],
            Some(cmp_const(OpCode::Gt, Ty::BigInt, 1, Value::I64(1))),
        );
        let rows = sorted(run(op).unwrap());
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][0], Value::I32(1));
        assert_eq!(rows[1][0], Value::I32(3));
    }

    #[test]
    fn having_can_remove_everything() {
        let steps = batches(vec![vec![i32s(&[Some(1), Some(2), Some(3)])]]);
        let op = build(
            steps,
            vec![load(Ty::Int, 0)],
            vec![agg(AggKind::CountStar, None)],
            Some(cmp_const(OpCode::Gt, Ty::BigInt, 1, Value::I64(100))),
        );
        assert!(run(op).unwrap().is_empty());
    }

    #[test]
    fn having_on_ungrouped_row() {
        let steps = batches(vec![vec![i32s(&[Some(1), Some(2)])]]);
        let op = build(
            steps,
            vec![],
            vec![agg(AggKind::CountStar, None)],
            Some(cmp_const(OpCode::Gt, Ty::BigInt, 0, Value::I64(5))),
        );
        assert!(run(op).unwrap().is_empty());
    }

    // --- 大きさ -------------------------------------------------------------

    #[test]
    fn more_groups_than_batch_size_emits_multiple_batches() {
        // BATCH_SIZE を超えるグループ数。ハッシュ表の再ハッシュも複数回起きる。
        let n = BATCH_SIZE * 2 + 37;
        let mut steps = Vec::new();
        let mut i = 0usize;
        while i < n {
            let end = (i + 500).min(n);
            let keys: Vec<Option<i32>> = (i..end).map(|k| Some(k as i32)).collect();
            let vals: Vec<Option<i32>> = (i..end).map(|_| Some(2)).collect();
            steps.push(MockStep::Rows(Batch::new(vec![i32s(&keys), i32s(&vals)])));
            // 毎回 I/O 待ちを挟んでも、グループが増減しないこと。
            steps.push(MockStep::NeedIo);
            i = end;
        }
        // 同じキーをもう一周入れて、各グループが 2 行ずつになるようにする。
        let keys: Vec<Option<i32>> = (0..n).map(|k| Some(k as i32)).collect();
        let vals: Vec<Option<i32>> = (0..n).map(|_| Some(3)).collect();
        steps.push(MockStep::Rows(Batch::new(vec![i32s(&keys), i32s(&vals)])));

        let op = build(
            steps,
            vec![load(Ty::Int, 0)],
            vec![agg(AggKind::CountStar, None), agg(AggKind::Sum, Some(load(Ty::Int, 1)))],
            None,
        );
        let rows = run(op).unwrap();
        assert_eq!(rows.len(), n, "グループが増減していない");
        let mut seen = vec![false; n];
        for r in &rows {
            let k = r[0].as_i64().unwrap() as usize;
            assert!(!seen[k], "グループ {k} が重複");
            seen[k] = true;
            assert_eq!(r[1], Value::I64(2));
            assert_eq!(r[2], Value::I128(5));
        }
        assert!(seen.iter().all(|s| *s));
    }

    #[test]
    fn output_is_chunked_to_batch_size() {
        let n = BATCH_SIZE + 10;
        let keys: Vec<Option<i32>> = (0..n).map(|k| Some(k as i32)).collect();
        let steps = vec![MockStep::Rows(Batch::new(vec![i32s(&keys)]))];
        let mut op =
            build(steps, vec![load(Ty::Int, 0)], vec![agg(AggKind::CountStar, None)], None);
        let mut catalog = Catalog::new();
        let mut vm = Vm::new();
        let mut sizes = Vec::new();
        loop {
            let mut ctx = ExecContext {
                catalog: &mut catalog,
                vm: &mut vm,
                io: Vec::new(),
                codec: Vec::new(),
            };
            match op.next(&mut ctx).unwrap() {
                Step::Ready(b) => sizes.push(b.card()),
                Step::NeedIo | Step::NeedCodec => {}
                Step::Done => break,
            }
        }
        assert_eq!(sizes, vec![BATCH_SIZE, 10]);
    }

    #[test]
    fn memory_limit_is_enforced() {
        // 長いキーを大量に入れて上限に当てる。スピルは無いので Oom を返す。
        let mut steps = Vec::new();
        let mut i = 0u32;
        // 1 グループあたり 1KiB 強のキー → 64MiB 到達には十分な件数を流す。
        let filler = "x".repeat(1024);
        for _ in 0..80 {
            let keys: Vec<Option<String>> = (0..1000)
                .map(|k| {
                    i += 1;
                    Some(format!("{filler}{k}-{i}"))
                })
                .collect();
            let refs: Vec<Option<&str>> = keys.iter().map(|s| s.as_deref()).collect();
            steps.push(MockStep::Rows(Batch::new(vec![strs(&refs)])));
        }
        let op =
            build(steps, vec![load(Ty::Varchar, 0)], vec![agg(AggKind::CountStar, None)], None);
        assert_eq!(code_of(run(op)), Some(Code::Oom));
    }

    #[test]
    fn phys_type_mismatch_between_program_and_data_is_detected() {
        // 宣言 INT なのに実データが VARCHAR。契約違反として Internal。
        let steps = vec![MockStep::Rows(Batch::new(vec![strs(&[Some("a")])]))];
        let op = build(steps, vec![load(Ty::Int, 0)], vec![agg(AggKind::CountStar, None)], None);
        assert_eq!(code_of(run(op)), Some(Code::Internal));
    }
}
