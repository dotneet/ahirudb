//! ウィンドウ関数（`OVER (...)`）。
//!
//! 出力は**入力の列に続けて `WindowSpec` 1 個につき 1 列**。バインダはこの
//! 並びでスキーマを組むので、順番は契約として変えない。
//!
//! ## ブロッキングと再開
//!
//! 分割の最後の行を見るまで最初の行の値すら決まらない（`sum(x) OVER ()` を
//! 考えればよい）ので、入力を全部読み切る**ブロッキング**オペレータになる。
//! リモート入力は途中で `Step::NeedIo` / `NeedCodec` を返すため、蓄積の途中
//! 状態はすべて `self` に置き、中断はそのまま素通しして次の `next()` で同じ
//! 場所から読み直す（DESIGN.md §6）。取り込みの単位は 1 バッチで、`absorb`
//! は途中で抜けない。これで再開時の取りこぼしと二重取りを構造的に潰す。
//! フェーズは `Buffering → Emitting → Done`。
//!
//! ## 行順
//!
//! 計算は分割ごとに ORDER BY 順で行うが、**出力は入力の行順**でなければ
//! ならない（ウィンドウは行を並べ替えない）。訪問順に値を積んでから、
//! 「入力行 → 訪問位置」の逆置換で `gather` して戻す。1 セルずつ `Value` を
//! 経由して散らすより確保が少ない。
//!
//! ## 枠（frame）
//!
//! - `WholePartition`（ORDER BY 無しの既定）は分割全体。
//! - `RangeUnboundedPreceding`（ORDER BY ありの既定）は分割先頭から
//!   **現在行のピア（ORDER BY キーが等しい行）の末尾まで**。ROWS ではなく
//!   RANGE なので、同順の行はすべて同じ枠＝同じ値になる。
//!   `sum(x) OVER (ORDER BY y)` で y が同値の行が同じ累計を返すのはこのため
//!   （DuckDB で確認済み）。
//!
//! ## メモリ
//!
//! 溢れ処理は持たない。蓄積が `MAX_BUFFER_BYTES` を超えたら `Oom` を返す。

use crate::exec::rowkey::{encode_key, ord_f64, pow10, HashIndex};
use crate::exec::{ExecContext, Operator, Step};
use crate::plan::{AggKind, SortKey, WindowKind, WindowSpec};
use crate::prelude::*;
use crate::sql::ast::WindowFrame;
use crate::vector::{Batch, Bitmap, Data, PhysType, Ty, Value, Vector, BATCH_SIZE};

use core::cmp::Ordering;

/// 溢れ処理を持たないので、これを超えたら `Oom` を返す。
/// ソート（256MiB）より低いのは、入力の蓄積に加えてウィンドウ列と分割ごとの
/// 添字表を同時に抱えるため。
const MAX_BUFFER_BYTES: usize = 128 * 1024 * 1024;

enum Phase {
    /// 入力を読んで溜めている。中断を跨いでもこの状態のまま。
    Buffering,
    /// ウィンドウ列が確定した。`BATCH_SIZE` ずつ切って返す。
    Emitting,
    Done,
}

pub struct Window {
    input: Box<dyn Operator>,
    windows: Vec<WindowSpec>,
    phase: Phase,

    /// 入力列の蓄積。出力の前半はこれをそのまま流す。
    cols: Vec<Vector>,
    /// 蓄積した行数。0 列の入力（`count(*) OVER ()`）でも行数だけは要る。
    rows: usize,
    /// 最初のバッチで列型を決めたか。0 列の入力があるので `cols` の空判定では
    /// 代用できない。
    init: bool,

    /// ウィンドウ列。`windows` と同じ並び、**入力行順**。`Emitting` 以降のみ有効。
    out: Vec<Vector>,
    /// 次に返す行の先頭。
    pos: usize,
}

impl Window {
    pub fn new(input: Box<dyn Operator>, windows: Vec<WindowSpec>) -> Result<Self> {
        Ok(Window {
            input,
            windows,
            phase: Phase::Buffering,
            cols: Vec::new(),
            rows: 0,
            init: false,
            out: Vec::new(),
            pos: 0,
        })
    }

    /// 1 バッチを丸ごと蓄積へ取り込む。**途中で抜けない**（再開の単位はバッチ）。
    fn absorb(&mut self, mut batch: Batch) -> Result<()> {
        // 以降は行番号で引くので selection をここで畳む。
        batch.materialize();
        let rows = batch.num_rows();
        if rows == 0 {
            return Ok(());
        }
        if !self.init {
            self.cols = batch.cols.iter().map(|c| Vector::new(c.ty())).collect();
            self.init = true;
        }
        ensure!(batch.cols.len() == self.cols.len(), Internal);
        // 行番号を u32 に載せるので、そこを超えたら諦める。
        ensure!(self.rows.saturating_add(rows) <= u32::MAX as usize, LimitExceeded);

        for (dst, src) in self.cols.iter_mut().zip(batch.cols.iter()) {
            append(dst, src)?;
        }
        self.rows += rows;

        let mut bytes = 0usize;
        for v in self.cols.iter() {
            bytes = bytes.saturating_add(vector_bytes(v));
        }
        ensure!(bytes <= MAX_BUFFER_BYTES, Oom);
        Ok(())
    }

    /// 入力を読み切った。全ウィンドウ列を作って出力フェーズへ移る。
    fn finish(&mut self, ctx: &mut ExecContext) -> Result<()> {
        // 1 行も来なかった（全分割が枝刈りされた等）。列型すら分からないので
        // 式を評価せずに空のまま出力へ移る。`emit` は即 `Done` になる。
        if self.rows == 0 {
            self.phase = Phase::Emitting;
            return Ok(());
        }
        // 蓄積列を一度バッチへ預けて式評価に使う。clone すると入力全体の複製に
        // なるので所有権ごと渡して後で取り返す。
        let cols = core::mem::take(&mut self.cols);
        let batch = if cols.is_empty() { Batch::rows_only(self.rows) } else { Batch::new(cols) };
        let mut out = Vec::with_capacity(self.windows.len());
        for spec in &self.windows {
            out.push(compute(spec, &batch, self.rows, ctx)?);
        }
        self.cols = batch.cols;
        self.out = out;
        self.pos = 0;
        self.phase = Phase::Emitting;
        Ok(())
    }

    fn emit(&mut self) -> Result<Step> {
        if self.pos >= self.rows {
            self.phase = Phase::Done;
            self.cols = Vec::new();
            self.out = Vec::new();
            return Ok(Step::Done);
        }
        let end = (self.pos + BATCH_SIZE).min(self.rows);
        let idx: Vec<u32> = (self.pos as u32..end as u32).collect();
        let mut cols = Vec::with_capacity(self.cols.len() + self.out.len());
        for c in self.cols.iter().chain(self.out.iter()) {
            cols.push(c.gather(&idx));
        }
        self.pos = end;
        Ok(Step::Ready(if cols.is_empty() {
            Batch::rows_only(idx.len())
        } else {
            Batch::new(cols)
        }))
    }
}

impl Operator for Window {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        loop {
            match self.phase {
                Phase::Buffering => match self.input.next(ctx)? {
                    Step::Ready(b) => self.absorb(b)?,
                    // 中断はそのまま上へ返す。蓄積は `self` に残るので、次回の
                    // 呼び出しはここから入力を引き直す。バイト待ちも展開待ちも
                    // 扱いは同じ。
                    other @ (Step::NeedIo | Step::NeedCodec) => return Ok(other),
                    Step::Done => self.finish(ctx)?,
                },
                Phase::Emitting => return self.emit(),
                Phase::Done => return Ok(Step::Done),
            }
        }
    }
}

// --- 1 つのウィンドウ関数の計算 ---------------------------------------------

/// `spec` の結果列を入力行順で作る。
fn compute(spec: &WindowSpec, batch: &Batch, rows: usize, ctx: &mut ExecContext) -> Result<Vector> {
    let mut pcols = Vec::with_capacity(spec.partition_by.len());
    for p in &spec.partition_by {
        pcols.push(ctx.vm.eval(p, batch)?);
    }
    let mut kcols = Vec::with_capacity(spec.order_by.len());
    for k in &spec.order_by {
        kcols.push(ctx.vm.eval(&k.expr, batch)?);
    }
    let mut acols = Vec::with_capacity(spec.args.len());
    for a in &spec.args {
        acols.push(ctx.vm.eval(a, batch)?);
    }

    // --- 分割 ---------------------------------------------------------------
    // キーの符号化は `rowkey` に寄せる。GROUP BY と同じく **NULL は NULL と
    // 同じ分割**に入る（`encode_key` がその意味論）。
    let mut parts: Vec<Vec<u32>> = Vec::new();
    if spec.partition_by.is_empty() {
        parts.push((0..rows as u32).collect());
    } else {
        let refs: Vec<&Vector> = pcols.iter().collect();
        let mut index = HashIndex::new();
        let mut key = Vec::new();
        for r in 0..rows {
            encode_key(&refs, r, &mut key);
            let (slot, is_new) = index.get_or_insert(&key);
            if is_new {
                parts.push(Vec::new());
            }
            match parts.get_mut(slot as usize) {
                Some(p) => p.push(r as u32),
                None => err!(Internal),
            }
        }
    }

    // --- 分割ごとに値を作る（訪問順） ---------------------------------------
    let mut vals = Vector::with_capacity(spec.result_ty, rows);
    let mut order: Vec<u32> = Vec::with_capacity(rows);
    for part in parts.iter_mut() {
        if !spec.order_by.is_empty() {
            // 比較器はソート演算子と同じ規律（NULL は nulls_first、f64 は
            // 全順序キー、最後は行番号で決着＝安定）。
            part.sort_by(|&a, &b| cmp_row(&spec.order_by, &kcols, a, b));
        }
        eval_partition(spec, part, &kcols, &acols, &mut vals)?;
        order.extend_from_slice(part);
    }
    // 各行はちょうど 1 回訪問される。ここが崩れると逆置換が壊れる。
    ensure!(vals.len() == rows && order.len() == rows, Internal);

    // --- 入力行順へ戻す -----------------------------------------------------
    let mut inv = vec![0u32; rows];
    for (p, &r) in order.iter().enumerate() {
        inv[r as usize] = p as u32;
    }
    let mut v = vals.gather(&inv);
    v.compact_validity();
    Ok(v)
}

/// 1 分割ぶんの値を `out` へ**訪問順**（`part` の並び）で積む。
fn eval_partition(
    spec: &WindowSpec,
    part: &[u32],
    kcols: &[Vector],
    acols: &[Vector],
    out: &mut Vector,
) -> Result<()> {
    let n = part.len();
    if n == 0 {
        return Ok(());
    }
    let ty = spec.result_ty;
    // ORDER BY が無ければ全行がピアなので、どちらの枠でも分割全体になる。
    let whole = spec.frame == WindowFrame::WholePartition || spec.order_by.is_empty();

    // ピア群の境界。`groups[g] = (start, end)`（`part` 上の半開区間）。
    let mut groups: Vec<(usize, usize)> = Vec::new();
    if spec.order_by.is_empty() {
        groups.push((0, n));
    } else {
        let mut s = 0usize;
        for i in 1..=n {
            // `i == n` を先に見て `part[i]` の範囲外参照を避ける。
            if i == n || cmp_keys(&spec.order_by, kcols, part[i - 1], part[i]) != Ordering::Equal {
                groups.push((s, i));
                s = i;
            }
        }
    }

    match spec.kind {
        // ピアを無視して 1 から通し番号。
        WindowKind::RowNumber => {
            for p in 0..n {
                push_as(out, ty, &Value::I64(p as i64 + 1))?;
            }
        }
        // 同順は同じ順位、次は飛ぶ。
        WindowKind::Rank => {
            for &(s, e) in groups.iter() {
                for _ in s..e {
                    push_as(out, ty, &Value::I64(s as i64 + 1))?;
                }
            }
        }
        // 同順は同じ順位、飛ばない。
        WindowKind::DenseRank => {
            for (gi, &(s, e)) in groups.iter().enumerate() {
                for _ in s..e {
                    push_as(out, ty, &Value::I64(gi as i64 + 1))?;
                }
            }
        }
        // 枠ではなく ORDER BY 順の相対位置で引く。
        WindowKind::Lag | WindowKind::Lead => {
            let src = match acols.first() {
                Some(c) => c,
                None => err!(WrongArgCount),
            };
            let back = spec.kind == WindowKind::Lag;
            for p in 0..n {
                let row = part[p] as usize;
                let off = match acols.get(1) {
                    Some(c) => match c.value_at(row).as_i64() {
                        Some(x) => x,
                        // オフセットが NULL なら結果も NULL（DuckDB と同じ）。
                        None if !c.is_valid(row) => {
                            out.push_null();
                            continue;
                        }
                        None => err!(TypeMismatch),
                    },
                    // 省略時は 1。負のオフセットは向きが反転する。
                    None => 1,
                };
                let target = if back { p as i64 - off } else { p as i64 + off };
                if target >= 0 && (target as usize) < n {
                    let v = src.value_at(part[target as usize] as usize);
                    push_as(out, ty, &v)?;
                } else {
                    // 分割の外は既定値。既定値の指定が無ければ NULL。
                    match acols.get(2) {
                        Some(c) => {
                            let v = c.value_at(row);
                            push_as(out, ty, &v)?;
                        }
                        None => out.push_null(),
                    }
                }
            }
        }
        // 枠の先頭は常に分割の先頭（どちらの枠でも UNBOUNDED PRECEDING）。
        WindowKind::FirstValue => {
            let src = match acols.first() {
                Some(c) => c,
                None => err!(WrongArgCount),
            };
            let v = src.value_at(part[0] as usize);
            for _ in 0..n {
                push_as(out, ty, &v)?;
            }
        }
        // 枠の末尾。RANGE ではピアの最後の行になる。
        WindowKind::LastValue => {
            let src = match acols.first() {
                Some(c) => c,
                None => err!(WrongArgCount),
            };
            if whole {
                let v = src.value_at(part[n - 1] as usize);
                for _ in 0..n {
                    push_as(out, ty, &v)?;
                }
            } else {
                for &(s, e) in groups.iter() {
                    let v = src.value_at(part[e - 1] as usize);
                    for _ in s..e {
                        push_as(out, ty, &v)?;
                    }
                }
            }
        }
        // 枠に対する集約。RANGE の枠は前方に伸びる一方なので、削除の要らない
        // 累積で足りる（ピア群を 1 単位として進める）。
        WindowKind::Agg(kind) => {
            let src = acols.first();
            // ウィンドウ版で持つのは枠に対して**足し込むだけ**で済む集約に限る。
            // 枠は前へ伸びる一方なので削除を考えずに済むのがこの実装の前提で、
            // 中央値や最頻値は枠ごとに作り直すことになり前提が崩れる。
            ensure!(
                matches!(
                    kind,
                    AggKind::CountStar
                        | AggKind::Count
                        | AggKind::Sum
                        | AggKind::Min
                        | AggKind::Max
                        | AggKind::Avg
                ),
                UnsupportedFeature
            );
            let div = match spec.args.first().map(|a| a.result_ty) {
                // DECIMAL の内部表現は整数。AVG では 10^scale で戻す。
                Some(Ty::Decimal { scale, .. }) => pow10(scale),
                _ => 1.0,
            };
            let mut acc = Acc::new();
            if whole {
                for &r in part.iter() {
                    acc.add(kind, src, r as usize)?;
                }
                let v = acc.value(kind, div);
                for _ in 0..n {
                    push_as(out, ty, &v)?;
                }
            } else {
                for &(s, e) in groups.iter() {
                    for &r in part[s..e].iter() {
                        acc.add(kind, src, r as usize)?;
                    }
                    let v = acc.value(kind, div);
                    for _ in s..e {
                        push_as(out, ty, &v)?;
                    }
                }
            }
        }
    }
    Ok(())
}

// --- 枠に対する集約の累積 ---------------------------------------------------

/// 1 枠ぶんの集約状態。意味論は `exec::agg` と同じに揃える
/// （SUM(整数) は i128、AVG は f64、MIN/MAX は NaN を最大とする全順序）。
struct Acc {
    /// 非 NULL 入力の個数。`COUNT(*)` では全行数。
    n: i64,
    /// 累積値。`Value::Null` は「まだ非 NULL 入力が無い」。
    acc: Value,
}

impl Acc {
    fn new() -> Self {
        Acc { n: 0, acc: Value::Null }
    }

    fn add(&mut self, kind: AggKind, col: Option<&Vector>, row: usize) -> Result<()> {
        if kind == AggKind::CountStar {
            // COUNT(*) は NULL だけの行も数える。
            self.n += 1;
            return Ok(());
        }
        let col = match col {
            Some(c) => c,
            // COUNT(*) 以外は必ず引数を持つ。
            None => err!(WrongArgCount),
        };
        // SUM/MIN/MAX/AVG/COUNT(x) は NULL を無視する。
        if !col.is_valid(row) {
            return Ok(());
        }
        self.n += 1;
        match kind {
            AggKind::CountStar | AggKind::Count => {}
            AggKind::Sum | AggKind::Avg => match col.data() {
                Data::I32(v) => self.add_int(v[row] as i128)?,
                Data::I64(v) => self.add_int(v[row] as i128)?,
                Data::I128(v) => self.add_int(v[row])?,
                Data::F64(v) => {
                    let s = match &self.acc {
                        Value::F64(s) => s + v[row],
                        _ => v[row],
                    };
                    self.acc = Value::F64(s);
                }
                _ => err!(TypeMismatch),
            },
            AggKind::Min | AggKind::Max => {
                let v = col.value_at(row);
                let take = match &self.acc {
                    Value::Null => true,
                    a => {
                        let c = cmp_val(&v, a);
                        if kind == AggKind::Min {
                            c.is_lt()
                        } else {
                            c.is_gt()
                        }
                    }
                };
                if take {
                    self.acc = v;
                }
            }
            // ウィンドウ版を持たない集約。呼び出し元が先に弾いている。
            _ => err!(UnsupportedFeature),
        }
        Ok(())
    }

    fn add_int(&mut self, x: i128) -> Result<()> {
        let s = match &self.acc {
            Value::I128(s) => match s.checked_add(x) {
                Some(v) => v,
                // i128 でも溢れる合計は黙って巻き戻さずエラーにする。
                None => err!(ValueOutOfRange),
            },
            _ => x,
        };
        self.acc = Value::I128(s);
        Ok(())
    }

    fn value(&self, kind: AggKind, div: f64) -> Value {
        match kind {
            AggKind::CountStar | AggKind::Count => Value::I64(self.n),
            // 非 NULL 入力が 1 つも無い枠は NULL。
            AggKind::Sum | AggKind::Min | AggKind::Max => self.acc.clone(),
            AggKind::Avg => match &self.acc {
                // 整数は i128 で正確に足してから 1 回だけ割る。
                Value::I128(s) if self.n > 0 => Value::F64(*s as f64 / div / self.n as f64),
                Value::F64(s) if self.n > 0 => Value::F64(s / self.n as f64),
                _ => Value::Null,
            },
            // ウィンドウ版を持たない集約はここへ来ない（`add` より前に弾く）。
            _ => Value::Null,
        }
    }
}

/// 同じ物理型の 2 値の比較。NaN は「すべてより大きい」（`exec::agg` と同じ）。
fn cmp_val(a: &Value, b: &Value) -> Ordering {
    match (a, b) {
        (Value::F64(x), Value::F64(y)) => ord_f64(*x, *y),
        // 物理型が食い違う組み合わせは上流のバグ。順序を付けずに等しいとみなす。
        _ => a.partial_cmp_same(b).unwrap_or(Ordering::Equal),
    }
}

/// 1 値を `ty` の列へ積む。
///
/// `result_ty` は**バインダが決めた出力型が正**なので、値のほうを合わせる。
/// `Vector::push_value` は物理型が違うと何もせずに落ちる（列の長さがずれる）
/// ため、ここで必ず変換してから渡す。
fn push_as(out: &mut Vector, ty: Ty, v: &Value) -> Result<()> {
    if v.is_null() {
        out.push_null();
        return Ok(());
    }
    match ty.phys() {
        PhysType::Bool => match v.as_bool() {
            Some(b) => out.push_value(&Value::Bool(b)),
            None => err!(TypeMismatch),
        },
        PhysType::I32 => match i32::try_from(int_of(v)?) {
            Ok(x) => out.push_value(&Value::I32(x)),
            Err(_) => err!(ValueOutOfRange),
        },
        PhysType::I64 => match i64::try_from(int_of(v)?) {
            Ok(x) => out.push_value(&Value::I64(x)),
            Err(_) => err!(ValueOutOfRange),
        },
        PhysType::I128 => out.push_value(&Value::I128(int_of(v)?)),
        PhysType::F64 => match v.as_f64() {
            Some(x) => out.push_value(&Value::F64(x)),
            None => err!(TypeMismatch),
        },
        PhysType::Bytes => match v.as_bytes() {
            Some(b) => out.push_value(&Value::Bytes(b.to_vec())),
            None => err!(TypeMismatch),
        },
    }
    Ok(())
}

fn int_of(v: &Value) -> Result<i128> {
    Ok(match v {
        Value::Bool(b) => *b as i128,
        Value::I32(x) => *x as i128,
        Value::I64(x) => *x as i128,
        Value::I128(x) => *x,
        _ => err!(TypeMismatch),
    })
}

// --- 比較 -------------------------------------------------------------------
//
// `exec::sort` の比較器と同じ規律。あちらの `cmp_row` / `cmp_data` / `f64_key`
// は private なので**意図的に複製**している（sort.rs は完成済みで手を入れない
// 約束のため）。挙動を変えるときは 2 か所を必ず揃えること。

/// 2 行の全順序比較。同値キーは行番号で決着させるので安定。
fn cmp_row(keys: &[SortKey], cols: &[Vector], a: u32, b: u32) -> Ordering {
    match cmp_keys(keys, cols, a, b) {
        Ordering::Equal => a.cmp(&b),
        o => o,
    }
}

/// キーだけの比較。`Equal` は「ピア（同順）である」ことを意味する。
fn cmp_keys(keys: &[SortKey], cols: &[Vector], a: u32, b: u32) -> Ordering {
    let (ai, bi) = (a as usize, b as usize);
    for (k, c) in keys.iter().zip(cols.iter()) {
        let (va, vb) = (c.is_valid(ai), c.is_valid(bi));
        if !va || !vb {
            if !va && !vb {
                // NULL 同士は同順。ピアになる。
                continue;
            }
            // NULL の位置は nulls_first だけで決まる。ここで desc を掛けると
            // バインダが入れた既定（ASC→LAST / DESC→FIRST）を二重に適用する。
            return if !va {
                if k.nulls_first {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            } else if k.nulls_first {
                Ordering::Greater
            } else {
                Ordering::Less
            };
        }
        let mut o = cmp_data(c.data(), ai, bi);
        if k.desc {
            o = o.reverse();
        }
        if o != Ordering::Equal {
            return o;
        }
    }
    Ordering::Equal
}

fn cmp_data(d: &Data, a: usize, b: usize) -> Ordering {
    match d {
        Data::Bool(v) => v.get(a).cmp(&v.get(b)),
        Data::I32(v) => v[a].cmp(&v[b]),
        Data::I64(v) => v[a].cmp(&v[b]),
        Data::I128(v) => v[a].cmp(&v[b]),
        Data::F64(v) => f64_key(v[a]).cmp(&f64_key(v[b])),
        Data::Bytes(v) => v.get(a).cmp(v.get(b)),
    }
}

/// f64 を順序を保つ `u64` へ写す。`partial_cmp` は NaN で `None` を返し、
/// それを「等しい」に潰すと推移律が壊れるため使わない。
/// 順序は `-inf < … < -0.0 = 0.0 < … < +inf < NaN`。
#[inline]
fn f64_key(v: f64) -> u64 {
    if v.is_nan() {
        return u64::MAX;
    }
    let b = if v == 0.0 { 0 } else { v.to_bits() };
    if b >> 63 != 0 {
        !b
    } else {
        b | (1 << 63)
    }
}

// --- バッファ操作 -----------------------------------------------------------
// これも `exec::sort` の private ヘルパと同じもの。

/// `src` の全行を `dst` の末尾に足す。物理型が同じであることが前提。
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
            for &o in s.offsets.iter().skip(1) {
                d.offsets.push(shift + (o - first));
            }
        }
        // 同じオペレータから来る列なので、ここに落ちるのは組み立て側のバグ。
        _ => err!(Internal),
    }
    // どちらかに NULL があれば validity を揃える。伸ばさないと長さが本体と
    // ずれ、以降の `is_valid` が範囲外を読む。
    if n > 0 && (src.has_nulls() || dst.has_nulls()) {
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

/// ベクタ 1 本のおおよそのバイト数。上限判定用なので厳密でなくてよい。
fn vector_bytes(v: &Vector) -> usize {
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
    use crate::error::code_of;
    use crate::expr::vm::Vm;
    use crate::expr::{Instr, OpCode, Program};

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

    fn ints(vals: &[Option<i32>]) -> Vector {
        col(Ty::Int, &vals.iter().map(|v| v.map(Value::I32)).collect::<Vec<_>>())
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

    /// `idx` 列をそのまま返すプログラム。
    fn load(ty: Ty, idx: u16) -> Program {
        let mut p = Program::new();
        let r = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, ty.phys(), r, 0, 0, idx));
        p.result = r;
        p.result_ty = ty;
        p
    }

    /// 定数を返すプログラム（lag/lead のオフセットと既定値用）。
    fn konst(ty: Ty, v: Value) -> Program {
        let mut p = Program::new();
        let r = p.alloc_reg();
        let c = p.add_const(ty, v);
        p.push(Instr::with_aux(OpCode::LoadConst, ty.phys(), r, 0, 0, c));
        p.result = r;
        p.result_ty = ty;
        p
    }

    fn skey(idx: u16, ty: Ty) -> SortKey {
        // SQL 既定の ASC / NULLS LAST。DuckDB の既定と揃えてある。
        SortKey { expr: load(ty, idx), desc: false, nulls_first: false }
    }

    struct SpecBuilder {
        kind: WindowKind,
        args: Vec<Program>,
        partition_by: Vec<Program>,
        order_by: Vec<SortKey>,
        result_ty: Ty,
    }

    fn spec(kind: WindowKind, result_ty: Ty) -> SpecBuilder {
        SpecBuilder {
            kind,
            args: Vec::new(),
            partition_by: Vec::new(),
            order_by: Vec::new(),
            result_ty,
        }
    }

    impl SpecBuilder {
        fn args(mut self, a: Vec<Program>) -> Self {
            self.args = a;
            self
        }
        fn part(mut self, p: Vec<Program>) -> Self {
            self.partition_by = p;
            self
        }
        fn order(mut self, o: Vec<SortKey>) -> Self {
            self.order_by = o;
            self
        }
        fn build(self) -> WindowSpec {
            // 枠はバインダの既定と同じ決め方: ORDER BY があれば RANGE、
            // 無ければ分割全体。
            let frame = if self.order_by.is_empty() {
                WindowFrame::WholePartition
            } else {
                WindowFrame::RangeUnboundedPreceding
            };
            WindowSpec {
                kind: self.kind,
                args: self.args,
                partition_by: self.partition_by,
                order_by: self.order_by,
                frame,
                result_ty: self.result_ty,
                name: String::from("w"),
            }
        }
        /// ORDER BY を持つが枠は分割全体、という組み合わせ。
        fn build_whole(self) -> WindowSpec {
            let mut s = self.build();
            s.frame = WindowFrame::WholePartition;
            s
        }
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

    // --- 実行ヘルパ ---------------------------------------------------------

    fn drive(steps: Vec<Script>, windows: Vec<WindowSpec>) -> Result<Vec<Vec<Value>>> {
        let mut op = Window::new(Mock::new(steps), windows)?;
        let mut cat = Catalog::new();
        let mut vm = Vm::new();
        let mut rows = Vec::new();
        for guard in 0..100_000 {
            assert!(guard < 99_999, "終わらない");
            let mut ctx =
                ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
            match op.next(&mut ctx)? {
                Step::Ready(b) => {
                    assert!(b.card() <= BATCH_SIZE);
                    for i in 0..b.card() {
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

    fn run(steps: Vec<Script>, windows: Vec<WindowSpec>) -> Vec<Vec<Value>> {
        drive(steps, windows).unwrap()
    }

    /// 出力の `c` 列目を i64 で取り出す。
    fn ints_at(rows: &[Vec<Value>], c: usize) -> Vec<Option<i64>> {
        rows.iter().map(|r| r[c].as_i64()).collect()
    }

    fn dbls_at(rows: &[Vec<Value>], c: usize) -> Vec<Option<f64>> {
        rows.iter().map(|r| r[c].as_f64()).collect()
    }

    /// 1 バッチ・2 列（g, x）の入力。
    fn gx(g: &[Option<i32>], x: &[Option<i32>]) -> Vec<Script> {
        vec![Script::Rows(vec![ints(g), ints(x)])]
    }

    // --- 中断と再開（最重要） -----------------------------------------------

    #[test]
    fn need_io_and_need_codec_mid_input_match_uninterrupted_run() {
        let chunk = |g: &[Option<i32>], x: &[Option<i32>]| Script::Rows(vec![ints(g), ints(x)]);
        let mk = |interrupted: bool| {
            let a = chunk(&[Some(1), Some(1)], &[Some(10), Some(20)]);
            let b = chunk(&[Some(1), Some(2)], &[Some(30), Some(1)]);
            let c = chunk(&[Some(2), Some(1)], &[Some(2), Some(20)]);
            if interrupted {
                // 入力の途中（先頭でも末尾でもない位置）で両方の中断を挟む。
                vec![a, Script::NeedIo, b, Script::NeedCodec, c, Script::NeedIo]
            } else {
                vec![a, b, c]
            }
        };
        let ws = || {
            vec![
                spec(WindowKind::RowNumber, Ty::BigInt)
                    .part(vec![load(Ty::Int, 0)])
                    .order(vec![skey(1, Ty::Int)])
                    .build(),
                spec(WindowKind::Agg(AggKind::Sum), Ty::HugeInt)
                    .args(vec![load(Ty::Int, 1)])
                    .part(vec![load(Ty::Int, 0)])
                    .order(vec![skey(1, Ty::Int)])
                    .build(),
            ]
        };
        let plain = run(mk(false), ws());
        let broken = run(mk(true), ws());
        assert_eq!(plain.len(), 6);
        for c in 0..4 {
            assert_eq!(ints_at(&broken, c), ints_at(&plain, c), "列 {c}");
        }
    }

    #[test]
    fn need_io_before_any_input_is_passed_through() {
        let steps = vec![
            Script::NeedIo,
            Script::NeedCodec,
            Script::Rows(vec![ints(&[Some(1), Some(1)]), ints(&[Some(5), Some(6)])]),
        ];
        let mut op = Window::new(
            Mock::new(steps),
            vec![spec(WindowKind::RowNumber, Ty::BigInt).order(vec![skey(1, Ty::Int)]).build()],
        )
        .unwrap();
        let mut cat = Catalog::new();
        let mut vm = Vm::new();
        let mut ctx =
            ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
        assert!(matches!(op.next(&mut ctx).unwrap(), Step::NeedIo));
        assert!(matches!(op.next(&mut ctx).unwrap(), Step::NeedCodec));
        let b = match op.next(&mut ctx).unwrap() {
            Step::Ready(b) => b,
            _ => panic!("行が来るはず"),
        };
        assert_eq!(b.card(), 2);
        assert_eq!(b.cols[2].value_at(1), Value::I64(2));
    }

    // --- 順位系（DuckDB で照合済み） ----------------------------------------

    /// y = 1,1,2,2,3 に対する row_number / rank / dense_rank。
    #[test]
    fn row_number_rank_dense_rank_with_ties() {
        let steps = gx(&[Some(1); 5], &[Some(1), Some(1), Some(2), Some(2), Some(3)]);
        let ws = vec![
            spec(WindowKind::RowNumber, Ty::BigInt).order(vec![skey(1, Ty::Int)]).build(),
            spec(WindowKind::Rank, Ty::BigInt).order(vec![skey(1, Ty::Int)]).build(),
            spec(WindowKind::DenseRank, Ty::BigInt).order(vec![skey(1, Ty::Int)]).build(),
        ];
        let rows = run(steps, ws);
        assert_eq!(ints_at(&rows, 2), vec![Some(1), Some(2), Some(3), Some(4), Some(5)]);
        assert_eq!(ints_at(&rows, 3), vec![Some(1), Some(1), Some(3), Some(3), Some(5)]);
        assert_eq!(ints_at(&rows, 4), vec![Some(1), Some(1), Some(2), Some(2), Some(3)]);
    }

    /// 複数列 PARTITION BY・DESC NULLS FIRST の ORDER BY を組み合わせる。
    /// これまでの row_number/rank テストは単一列の PARTITION BY・既定
    /// （ASC NULLS LAST）の ORDER BY でしか通っていなかった経路
    /// （`cmp_keys` の複数キーループ・DESC 反転・NULL 位置反転）を確認する。
    /// DuckDB で照合済み:
    ///   PARTITION BY (1,1): x=10,20 → DESC で 20,10 → row_number 2,1
    ///   PARTITION BY (1,2): x=30,40 → DESC で 40,30 → row_number 2,1
    ///   PARTITION BY (2,1): x=NULL,5 → DESC NULLS FIRST で NULL,5 → row_number 1,2
    #[test]
    fn multi_column_partition_by_with_desc_nulls_first_order_by() {
        let steps = vec![Script::Rows(vec![
            ints(&[Some(1), Some(1), Some(1), Some(1), Some(2), Some(2)]), // p1
            ints(&[Some(1), Some(1), Some(2), Some(2), Some(1), Some(1)]), // p2
            ints(&[Some(10), Some(20), Some(30), Some(40), None, Some(5)]), // x
        ])];
        let ws = vec![spec(WindowKind::RowNumber, Ty::BigInt)
            .part(vec![load(Ty::Int, 0), load(Ty::Int, 1)])
            .order(vec![SortKey { expr: load(Ty::Int, 2), desc: true, nulls_first: true }])
            .build()];
        let rows = run(steps, ws);
        assert_eq!(ints_at(&rows, 3), vec![Some(2), Some(1), Some(2), Some(1), Some(1), Some(2)]);
    }

    // --- RANGE のピア群 -----------------------------------------------------

    /// 同順の行は同じ枠を共有する（ROWS ではなく RANGE）。
    /// DuckDB:
    ///   y=1 の 2 行 → sum 30 / count 2 / avg 15
    ///   y=2 の 2 行 → sum 100 / count 4 / avg 25
    ///   y=3        → sum 150 / count 5 / avg 30
    #[test]
    fn range_frame_shares_the_running_value_across_peers() {
        let steps = vec![Script::Rows(vec![
            ints(&[Some(1), Some(1), Some(2), Some(2), Some(3)]),
            ints(&[Some(10), Some(20), Some(30), Some(40), Some(50)]),
        ])];
        let w = |k: WindowKind, ty: Ty| {
            spec(k, ty).args(vec![load(Ty::Int, 1)]).order(vec![skey(0, Ty::Int)]).build()
        };
        let ws = vec![
            w(WindowKind::Agg(AggKind::Sum), Ty::HugeInt),
            spec(WindowKind::Agg(AggKind::CountStar), Ty::BigInt)
                .order(vec![skey(0, Ty::Int)])
                .build(),
            w(WindowKind::Agg(AggKind::Avg), Ty::Double),
            w(WindowKind::Agg(AggKind::Min), Ty::Int),
            w(WindowKind::Agg(AggKind::Max), Ty::Int),
            w(WindowKind::FirstValue, Ty::Int),
            w(WindowKind::LastValue, Ty::Int),
        ];
        let rows = run(steps, ws);
        assert_eq!(ints_at(&rows, 2), vec![Some(30), Some(30), Some(100), Some(100), Some(150)]);
        assert_eq!(ints_at(&rows, 3), vec![Some(2), Some(2), Some(4), Some(4), Some(5)]);
        assert_eq!(
            dbls_at(&rows, 4),
            vec![Some(15.0), Some(15.0), Some(25.0), Some(25.0), Some(30.0)]
        );
        assert_eq!(ints_at(&rows, 5), vec![Some(10); 5], "MIN は枠の先頭から");
        assert_eq!(ints_at(&rows, 6), vec![Some(20), Some(20), Some(40), Some(40), Some(50)]);
        assert_eq!(ints_at(&rows, 7), vec![Some(10); 5], "FIRST_VALUE は分割の先頭");
        assert_eq!(
            ints_at(&rows, 8),
            vec![Some(20), Some(20), Some(40), Some(40), Some(50)],
            "LAST_VALUE はピアの末尾"
        );
    }

    /// FIRST_VALUE/LAST_VALUE を実際に複数パーティションと組み合わせる。
    /// 既存の RANGE テストは `.part(...)` を使わない単一パーティションでしか
    /// 確認していなかった。
    #[test]
    fn first_value_last_value_are_scoped_per_partition() {
        let steps = vec![Script::Rows(vec![
            ints(&[Some(1), Some(1), Some(1), Some(2), Some(2)]), // g
            ints(&[Some(10), Some(20), Some(30), Some(100), Some(200)]), // x
        ])];
        let w = |k: WindowKind| {
            spec(k, Ty::Int)
                .args(vec![load(Ty::Int, 1)])
                .part(vec![load(Ty::Int, 0)])
                .order(vec![skey(1, Ty::Int)])
                .build_whole()
        };
        let ws = vec![w(WindowKind::FirstValue), w(WindowKind::LastValue)];
        let rows = run(steps, ws);
        assert_eq!(
            ints_at(&rows, 2),
            vec![Some(10), Some(10), Some(10), Some(100), Some(100)],
            "FIRST_VALUE はパーティションごとに別々でなければならない"
        );
        assert_eq!(
            ints_at(&rows, 3),
            vec![Some(30), Some(30), Some(30), Some(200), Some(200)],
            "LAST_VALUE もパーティションを跨いで漏れてはいけない"
        );
    }

    /// ORDER BY 無し（枠は分割全体）。全行が同じ値になる。
    #[test]
    fn whole_partition_frame_without_order_by() {
        let steps = gx(&[Some(1); 3], &[Some(1), Some(2), Some(3)]);
        let ws = vec![
            spec(WindowKind::Agg(AggKind::Sum), Ty::HugeInt).args(vec![load(Ty::Int, 1)]).build(),
            spec(WindowKind::Agg(AggKind::CountStar), Ty::BigInt).build(),
            spec(WindowKind::FirstValue, Ty::Int).args(vec![load(Ty::Int, 1)]).build(),
            spec(WindowKind::LastValue, Ty::Int).args(vec![load(Ty::Int, 1)]).build(),
        ];
        let rows = run(steps, ws);
        assert_eq!(ints_at(&rows, 2), vec![Some(6); 3]);
        assert_eq!(ints_at(&rows, 3), vec![Some(3); 3]);
        assert_eq!(ints_at(&rows, 4), vec![Some(1); 3]);
        assert_eq!(ints_at(&rows, 5), vec![Some(3); 3], "枠が分割全体なら最後の行");
    }

    /// ORDER BY があっても枠が分割全体なら累計にならない。
    #[test]
    fn explicit_whole_partition_frame_with_order_by() {
        let steps = gx(&[Some(1); 3], &[Some(3), Some(1), Some(2)]);
        let ws = vec![spec(WindowKind::Agg(AggKind::Sum), Ty::HugeInt)
            .args(vec![load(Ty::Int, 1)])
            .order(vec![skey(1, Ty::Int)])
            .build_whole()];
        assert_eq!(ints_at(&run(steps, ws), 2), vec![Some(6); 3]);
    }

    // --- lag / lead ---------------------------------------------------------

    #[test]
    fn lag_and_lead_at_partition_edges() {
        let steps = gx(&[Some(1); 3], &[Some(1), Some(2), Some(3)]);
        let ws = vec![
            // lag(x)
            spec(WindowKind::Lag, Ty::Int)
                .args(vec![load(Ty::Int, 1)])
                .order(vec![skey(1, Ty::Int)])
                .build(),
            // lead(x, 2, -1)
            spec(WindowKind::Lead, Ty::Int)
                .args(vec![
                    load(Ty::Int, 1),
                    konst(Ty::BigInt, Value::I64(2)),
                    konst(Ty::Int, Value::I32(-1)),
                ])
                .order(vec![skey(1, Ty::Int)])
                .build(),
            // lag(x, 1, -9)
            spec(WindowKind::Lag, Ty::Int)
                .args(vec![
                    load(Ty::Int, 1),
                    konst(Ty::BigInt, Value::I64(1)),
                    konst(Ty::Int, Value::I32(-9)),
                ])
                .order(vec![skey(1, Ty::Int)])
                .build(),
            // lag(x, -1) は lead(x, 1) と同じ。
            spec(WindowKind::Lag, Ty::Int)
                .args(vec![load(Ty::Int, 1), konst(Ty::BigInt, Value::I64(-1))])
                .order(vec![skey(1, Ty::Int)])
                .build(),
            // lag(x, 0) は自分自身。
            spec(WindowKind::Lag, Ty::Int)
                .args(vec![load(Ty::Int, 1), konst(Ty::BigInt, Value::I64(0))])
                .order(vec![skey(1, Ty::Int)])
                .build(),
        ];
        let rows = run(steps, ws);
        assert_eq!(ints_at(&rows, 2), vec![None, Some(1), Some(2)]);
        assert_eq!(ints_at(&rows, 3), vec![Some(3), Some(-1), Some(-1)]);
        assert_eq!(ints_at(&rows, 4), vec![Some(-9), Some(1), Some(2)]);
        assert_eq!(ints_at(&rows, 5), vec![Some(2), Some(3), None]);
        assert_eq!(ints_at(&rows, 6), vec![Some(1), Some(2), Some(3)]);
    }

    /// オフセットが NULL なら結果も NULL（DuckDB と同じ）。
    #[test]
    fn lag_with_null_offset_yields_null() {
        let steps = gx(&[Some(1); 2], &[Some(1), Some(2)]);
        let ws = vec![spec(WindowKind::Lag, Ty::Int)
            .args(vec![load(Ty::Int, 1), konst(Ty::BigInt, Value::Null)])
            .order(vec![skey(1, Ty::Int)])
            .build()];
        assert_eq!(ints_at(&run(steps, ws), 2), vec![None, None]);
    }

    /// lag は分割を跨がない。
    #[test]
    fn lag_does_not_cross_partitions() {
        let steps =
            gx(&[Some(1), Some(1), Some(2), Some(2)], &[Some(1), Some(2), Some(3), Some(4)]);
        let ws = vec![spec(WindowKind::Lag, Ty::Int)
            .args(vec![load(Ty::Int, 1)])
            .part(vec![load(Ty::Int, 0)])
            .order(vec![skey(1, Ty::Int)])
            .build()];
        assert_eq!(ints_at(&run(steps, ws), 2), vec![None, Some(1), None, Some(3)]);
    }

    // --- 分割 ---------------------------------------------------------------

    /// 複数分割。DuckDB の出力と同じ（g=2 は y が NULL の 2 行を含む）。
    #[test]
    fn multiple_partitions_with_null_order_key() {
        // (g, y, x)
        let steps = vec![Script::Rows(vec![
            ints(&[Some(1), Some(1), Some(2), Some(2), Some(2)]),
            ints(&[Some(1), Some(2), Some(1), None, None]),
            ints(&[Some(10), Some(20), Some(1), Some(2), Some(3)]),
        ])];
        let ws = vec![
            spec(WindowKind::Rank, Ty::BigInt)
                .part(vec![load(Ty::Int, 0)])
                .order(vec![skey(1, Ty::Int)])
                .build(),
            spec(WindowKind::Agg(AggKind::Sum), Ty::HugeInt)
                .args(vec![load(Ty::Int, 2)])
                .part(vec![load(Ty::Int, 0)])
                .order(vec![skey(1, Ty::Int)])
                .build(),
        ];
        let rows = run(steps, ws);
        // NULL は ASC/NULLS LAST で末尾、かつ NULL 同士はピア。
        assert_eq!(ints_at(&rows, 3), vec![Some(1), Some(2), Some(1), Some(2), Some(2)]);
        assert_eq!(ints_at(&rows, 4), vec![Some(10), Some(30), Some(1), Some(6), Some(6)]);
    }

    /// PARTITION BY 無し = 分割が 1 つ。
    #[test]
    fn single_partition_without_partition_by() {
        let steps = gx(&[Some(9); 4], &[Some(4), Some(3), Some(2), Some(1)]);
        let ws =
            vec![spec(WindowKind::RowNumber, Ty::BigInt).order(vec![skey(1, Ty::Int)]).build()];
        // 入力行順で出るので、値 4,3,2,1 に対して順位は 4,3,2,1。
        assert_eq!(ints_at(&run(steps, ws), 2), vec![Some(4), Some(3), Some(2), Some(1)]);
    }

    /// NULL の分割キーは独立した 1 分割（GROUP BY と同じ）。
    #[test]
    fn null_partition_keys_form_their_own_partition() {
        let steps = gx(&[None, Some(1), None, Some(1)], &[Some(1), Some(2), Some(3), Some(4)]);
        let ws = vec![
            spec(WindowKind::Agg(AggKind::CountStar), Ty::BigInt)
                .part(vec![load(Ty::Int, 0)])
                .build(),
            spec(WindowKind::Agg(AggKind::Sum), Ty::HugeInt)
                .args(vec![load(Ty::Int, 1)])
                .part(vec![load(Ty::Int, 0)])
                .build(),
        ];
        let rows = run(steps, ws);
        assert_eq!(ints_at(&rows, 2), vec![Some(2); 4]);
        assert_eq!(ints_at(&rows, 3), vec![Some(4), Some(6), Some(4), Some(6)]);
    }

    // --- 行順 ---------------------------------------------------------------

    /// 出力は必ず入力の行順。分割内で並べ替えても戻ってくること。
    #[test]
    fn output_keeps_input_row_order() {
        let g = [Some(1), Some(2), Some(1), Some(2), Some(1)];
        let x = [Some(50), Some(5), Some(10), Some(1), Some(30)];
        let steps = gx(&g, &x);
        let ws = vec![spec(WindowKind::RowNumber, Ty::BigInt)
            .part(vec![load(Ty::Int, 0)])
            .order(vec![skey(1, Ty::Int)])
            .build()];
        let rows = run(steps, ws);
        // 入力列がそのまま並んでいること。
        assert_eq!(ints_at(&rows, 0), g.iter().map(|v| v.map(|x| x as i64)).collect::<Vec<_>>());
        assert_eq!(ints_at(&rows, 1), x.iter().map(|v| v.map(|x| x as i64)).collect::<Vec<_>>());
        // g=1 は 10 < 30 < 50、g=2 は 1 < 5。
        assert_eq!(ints_at(&rows, 2), vec![Some(3), Some(2), Some(1), Some(1), Some(2)]);
    }

    /// 同順の行は入力順を保つ（比較器が行番号で決着するため）。
    #[test]
    fn ties_keep_input_order_in_row_number() {
        let steps = gx(&[Some(1); 4], &[Some(7), Some(7), Some(7), Some(7)]);
        let ws =
            vec![spec(WindowKind::RowNumber, Ty::BigInt).order(vec![skey(1, Ty::Int)]).build()];
        assert_eq!(ints_at(&run(steps, ws), 2), vec![Some(1), Some(2), Some(3), Some(4)]);
    }

    // --- 大きさ -------------------------------------------------------------

    #[test]
    fn more_rows_than_batch_size() {
        const N: usize = BATCH_SIZE * 2 + 37;
        let mut steps = Vec::new();
        let mut i = 0usize;
        while i < N {
            let end = (i + 500).min(N);
            // g は 0/1 の 2 分割、x は通し番号（降順に見えるよう反転）。
            let g: Vec<Option<i32>> = (i..end).map(|k| Some((k % 2) as i32)).collect();
            let x: Vec<Option<i32>> = (i..end).map(|k| Some((N - k) as i32)).collect();
            steps.push(Script::Rows(vec![ints(&g), ints(&x)]));
            i = end;
        }
        let ws = vec![
            spec(WindowKind::RowNumber, Ty::BigInt)
                .part(vec![load(Ty::Int, 0)])
                .order(vec![skey(1, Ty::Int)])
                .build(),
            spec(WindowKind::Agg(AggKind::CountStar), Ty::BigInt)
                .part(vec![load(Ty::Int, 0)])
                .build(),
        ];
        let rows = run(steps, ws);
        assert_eq!(rows.len(), N);
        // x は全体で単調減少なので、分割ごとの順位は末尾ほど小さい。
        let n0 = N.div_ceil(2) as i64;
        let n1 = (N / 2) as i64;
        assert_eq!(rows[0][2].as_i64(), Some(n0));
        assert_eq!(rows[1][2].as_i64(), Some(n1));
        assert_eq!(rows[N - 1][2].as_i64(), Some(1));
        assert_eq!(rows[0][3].as_i64(), Some(n0));
        assert_eq!(rows[1][3].as_i64(), Some(n1));
    }

    #[test]
    fn output_is_chunked_to_batch_size() {
        const N: usize = BATCH_SIZE + 10;
        let x: Vec<Option<i32>> = (0..N as i32).map(Some).collect();
        let steps = vec![Script::Rows(vec![ints(&x)])];
        let mut op = Window::new(
            Mock::new(steps),
            vec![spec(WindowKind::RowNumber, Ty::BigInt).order(vec![skey(0, Ty::Int)]).build()],
        )
        .unwrap();
        let mut cat = Catalog::new();
        let mut vm = Vm::new();
        let mut sizes = Vec::new();
        loop {
            let mut ctx =
                ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
            match op.next(&mut ctx).unwrap() {
                Step::Ready(b) => sizes.push(b.card()),
                Step::NeedIo | Step::NeedCodec => {}
                Step::Done => break,
            }
        }
        assert_eq!(sizes, vec![BATCH_SIZE, 10]);
    }

    // --- 端の条件 -----------------------------------------------------------

    #[test]
    fn empty_input_emits_nothing() {
        let ws =
            || vec![spec(WindowKind::RowNumber, Ty::BigInt).order(vec![skey(0, Ty::Int)]).build()];
        assert!(run(Vec::new(), ws()).is_empty());
        // 0 行のバッチだけが来る場合も同じ。
        assert!(run(vec![Script::Rows(vec![ints(&[])]), Script::NeedIo], ws()).is_empty());
    }

    /// NULL 値は SUM / COUNT(x) から外れるが COUNT(*) には入る。
    #[test]
    fn nulls_in_the_value_are_ignored_by_aggregates() {
        let steps = gx(&[Some(1); 3], &[None, Some(2), None]);
        let ws = vec![
            spec(WindowKind::Agg(AggKind::Sum), Ty::HugeInt).args(vec![load(Ty::Int, 1)]).build(),
            spec(WindowKind::Agg(AggKind::Count), Ty::BigInt).args(vec![load(Ty::Int, 1)]).build(),
            spec(WindowKind::Agg(AggKind::CountStar), Ty::BigInt).build(),
        ];
        let rows = run(steps, ws);
        assert_eq!(ints_at(&rows, 2), vec![Some(2); 3]);
        assert_eq!(ints_at(&rows, 3), vec![Some(1); 3]);
        assert_eq!(ints_at(&rows, 4), vec![Some(3); 3]);
    }

    /// 全部 NULL の枠は SUM が NULL、COUNT は 0。
    #[test]
    fn all_null_frame_sums_to_null() {
        let steps = gx(&[Some(1); 2], &[None, None]);
        let ws = vec![
            spec(WindowKind::Agg(AggKind::Sum), Ty::HugeInt).args(vec![load(Ty::Int, 1)]).build(),
            spec(WindowKind::Agg(AggKind::Count), Ty::BigInt).args(vec![load(Ty::Int, 1)]).build(),
            spec(WindowKind::Agg(AggKind::Avg), Ty::Double).args(vec![load(Ty::Int, 1)]).build(),
        ];
        let rows = run(steps, ws);
        assert!(rows[0][2].is_null());
        assert_eq!(rows[0][3].as_i64(), Some(0));
        assert!(rows[0][4].is_null());
    }

    /// 文字列列でも first/last/min/max が動く（`Value` 経由の積み込み）。
    #[test]
    fn string_values() {
        let steps = vec![Script::Rows(vec![
            ints(&[Some(1), Some(1), Some(1)]),
            strs(&[Some("b"), None, Some("a")]),
        ])];
        let ws = vec![
            spec(WindowKind::FirstValue, Ty::Varchar).args(vec![load(Ty::Varchar, 1)]).build(),
            spec(WindowKind::LastValue, Ty::Varchar).args(vec![load(Ty::Varchar, 1)]).build(),
            spec(WindowKind::Agg(AggKind::Min), Ty::Varchar)
                .args(vec![load(Ty::Varchar, 1)])
                .build(),
        ];
        let rows = run(steps, ws);
        assert_eq!(rows[0][2], Value::Bytes(b"b".to_vec()));
        assert_eq!(rows[0][3], Value::Bytes(b"a".to_vec()));
        assert_eq!(rows[0][4], Value::Bytes(b"a".to_vec()));
    }

    /// 0 列の入力（`count(*) OVER ()` だけを選ぶ経路）。
    #[test]
    fn zero_column_input() {
        struct RowsOnly(usize);
        impl Operator for RowsOnly {
            fn next(&mut self, _ctx: &mut ExecContext) -> Result<Step> {
                if self.0 == 0 {
                    return Ok(Step::Done);
                }
                let n = self.0;
                self.0 = 0;
                Ok(Step::Ready(Batch::rows_only(n)))
            }
        }
        let mut op = Window::new(
            Box::new(RowsOnly(3)),
            vec![spec(WindowKind::Agg(AggKind::CountStar), Ty::BigInt).build()],
        )
        .unwrap();
        let mut cat = Catalog::new();
        let mut vm = Vm::new();
        let mut ctx =
            ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
        let b = match op.next(&mut ctx).unwrap() {
            Step::Ready(b) => b,
            _ => panic!("行が来るはず"),
        };
        assert_eq!(b.card(), 3);
        assert_eq!(b.cols[0].value_at(0), Value::I64(3));
    }

    /// f64 の順序キー。NaN は最大、-0.0 と 0.0 は同順（ピア）。
    #[test]
    fn f64_order_key_is_a_total_order() {
        let steps = vec![Script::Rows(vec![col(
            Ty::Double,
            &[
                Some(Value::F64(f64::NAN)),
                Some(Value::F64(0.0)),
                Some(Value::F64(-0.0)),
                Some(Value::F64(-1.0)),
            ],
        )])];
        let ws = vec![
            spec(WindowKind::Rank, Ty::BigInt).order(vec![skey(0, Ty::Double)]).build(),
            spec(WindowKind::RowNumber, Ty::BigInt).order(vec![skey(0, Ty::Double)]).build(),
        ];
        let rows = run(steps, ws);
        // 並びは -1.0 < -0.0 = 0.0 < NaN。
        assert_eq!(ints_at(&rows, 1), vec![Some(4), Some(2), Some(2), Some(1)]);
        assert_eq!(ints_at(&rows, 2), vec![Some(4), Some(2), Some(3), Some(1)]);
    }

    /// SUM は i128 で累積する（i64 では溢れる合計）。
    #[test]
    fn sum_accumulates_in_i128() {
        let big = i64::MAX;
        let steps = vec![Script::Rows(vec![col(
            Ty::BigInt,
            &[Some(Value::I64(big)), Some(Value::I64(big)), Some(Value::I64(big))],
        )])];
        let ws = vec![spec(WindowKind::Agg(AggKind::Sum), Ty::HugeInt)
            .args(vec![load(Ty::BigInt, 0)])
            .build()];
        let rows = run(steps, ws);
        assert_eq!(rows[0][1], Value::I128(big as i128 * 3));
    }

    #[test]
    fn sum_overflowing_i128_errors() {
        let steps = vec![Script::Rows(vec![col(
            Ty::HugeInt,
            &[Some(Value::I128(i128::MAX)), Some(Value::I128(i128::MAX))],
        )])];
        let ws = vec![spec(WindowKind::Agg(AggKind::Sum), Ty::HugeInt)
            .args(vec![load(Ty::HugeInt, 0)])
            .build()];
        assert_eq!(code_of(drive(steps, ws).map(|_| ())), Some(Code::ValueOutOfRange));
    }

    /// 引数が要るのに無い、は構成側のバグとして弾く。
    #[test]
    fn missing_argument_is_rejected() {
        let steps = gx(&[Some(1)], &[Some(1)]);
        let ws = vec![spec(WindowKind::Lag, Ty::Int).order(vec![skey(1, Ty::Int)]).build()];
        assert_eq!(code_of(drive(steps, ws).map(|_| ())), Some(Code::WrongArgCount));
    }

    /// 入力の selection vector を尊重する。
    #[test]
    fn selection_vector_on_input_is_respected() {
        struct Sel;
        impl Operator for Sel {
            fn next(&mut self, _ctx: &mut ExecContext) -> Result<Step> {
                let mut b = Batch::new(vec![ints(&[Some(5), Some(1), Some(9), Some(3)])]);
                b.sel = Some(vec![1, 3]);
                Ok(Step::Ready(b))
            }
        }
        // 1 回だけ返すよう、2 回目以降は Done になるモックで包む。
        struct Once(Option<Sel>);
        impl Operator for Once {
            fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
                match self.0.take() {
                    Some(mut s) => s.next(ctx),
                    None => Ok(Step::Done),
                }
            }
        }
        let mut op = Window::new(
            Box::new(Once(Some(Sel))),
            vec![spec(WindowKind::RowNumber, Ty::BigInt).order(vec![skey(0, Ty::Int)]).build()],
        )
        .unwrap();
        let mut cat = Catalog::new();
        let mut vm = Vm::new();
        let mut ctx =
            ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
        let b = match op.next(&mut ctx).unwrap() {
            Step::Ready(b) => b,
            _ => panic!("行が来るはず"),
        };
        assert_eq!(b.card(), 2);
        assert_eq!(b.cols[0].value_at(0), Value::I32(1));
        assert_eq!(b.cols[1].value_at(0), Value::I64(1));
        assert_eq!(b.cols[1].value_at(1), Value::I64(2));
    }
}
