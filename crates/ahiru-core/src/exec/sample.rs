//! `USING SAMPLE` / `TABLESAMPLE`。
//!
//! パーセント指定はベルヌーイ法で実装する（各行を独立に確率 `p` で残す。
//! 上流をブロックしない、`exec::Filter` と同じ形のストリーミングオペレータ）。
//! 行数指定は入力を全部読み切ってから一様ランダムに N 行選ぶブロッキング
//! 方式（`exec::sort::Sort` と同じ「蓄積 → 確定 → `BATCH_SIZE` ずつ返す」の
//! 3 相）。`BERNOULLI`/`SYSTEM`/`RESERVOIR` という手法名の構文は受理するが、
//! 実装上の違いは無い（タスクの優先度: パーセント指定 > 行数指定 > 手法の
//! 使い分け、という指示どおりの単純化）。
//!
//! ## 乱数生成器
//! 依存クレート無しの xorshift64* を自前で実装する（`no_std` なので `rand`
//! クレートは使えない）。シードは `plan::SampleSpec::seed`
//! （`USING SAMPLE ... (method, seed)` で明示するか、無ければ
//! `plan::DEFAULT_SAMPLE_SEED`）から決定的に初期化するので、同じクエリを
//! 何度実行しても同じ行が選ばれる。
//!
//! ## `NeedIo`/`NeedCodec` をまたいでも再現性が壊れない理由
//! - ベルヌーイ法（`Bernoulli`）: 1 行ごとに 1 回だけ乱数を引く。中断は
//!   入力側で起きてそのまま上へ素通しするだけ（`Filter` と同じ）なので、
//!   PRNG の呼び出し列は「実際に評価された行の並び」だけで決まり、中断が
//!   どこで起きたかには依存しない。
//! - 行数指定（`RowSample`）: `Buffering` フェーズで蓄積を続け、入力が
//!   `Done` になって初めて乱数で部分集合を選ぶ。中断はその前段（`Sort` と
//!   同じ）でしか起きないので、選び方自体は中断の有無に影響されない。

use crate::exec::sort::vector_bytes;
use crate::exec::{ExecContext, Operator, Step};
use crate::plan::SampleSpec;
use crate::prelude::*;
use crate::vector::{Batch, Ty, Vector, BATCH_SIZE};

/// xorshift64*。`no_std` 環境で依存クレート無しに使える決定的 PRNG。
/// 暗号強度は要らない（サンプリング用途のみ）。
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // 0 は不動点（`0 ^ ... = 0`）になるので下駄を履かせる。
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

    /// `[0, 1)` の一様乱数。上位 53 bit を仮数部として使う。
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// `[0, n)` の一様な添字。`n == 0` では呼ばない前提。
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

// --- パーセント指定（ベルヌーイ法） ------------------------------------------

pub struct Bernoulli {
    input: Box<dyn Operator>,
    /// 残す確率（0.0..=1.0）。
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
                // 全行落ちたバッチは上流に返さず次を引く（`Filter` と同じ規律）。
                continue;
            }
            batch.sel = Some(sel);
            return Ok(Step::Ready(batch));
        }
    }
}

// --- 行数指定（一様ランダムに N 行） ------------------------------------------

/// 溢れ処理を持たないので、これを超えたら `Oom` を返す（`exec::sort::Sort`
/// と同じ上限・同じ理由: wasm の線形メモリと他の抱えているバッファとの
/// 同居を考えて単体では抑える）。
const MAX_BUFFER_BYTES: usize = 256 * 1024 * 1024;

/// 蓄積した入力 1 バッチぶん。
struct Buffered {
    cols: Vec<Vector>,
    rows: usize,
}

enum Phase {
    /// 入力を読んで溜めている。中断を跨いでもこの状態のまま。
    Buffering,
    /// 選ばれた行を `BATCH_SIZE` ずつ返す。
    Emitting,
    Done,
}

pub struct RowSample {
    input: Box<dyn Operator>,
    /// 残す行数。
    target: u64,
    rng: Rng,
    phase: Phase,
    batches: Vec<Buffered>,
    total_rows: u64,
    buffered_bytes: usize,
    /// `Emitting` 以降のみ有効。選ばれた行を列指向で持つ。
    out: Vec<Vector>,
    out_rows: usize,
    pos: usize,
}

impl RowSample {
    pub fn new(input: Box<dyn Operator>, spec: &SampleSpec) -> Self {
        // 端数は四捨五入（`duckdb` の `12.5 ROWS` が 12 行になる例もあれば
        // 四捨五入寄りの例もあり実装依存が伺えるので、ここでは単純な
        // 四捨五入に決め打つ）。負値は構文上出てこない（`sql::parser` が拒否）。
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
        // selection を先に解消しておく。`value_at` を後で行番号のまま使うため。
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

    /// 入力を読み切った。`0..total_rows` から一様ランダムに `k` 個選び、
    /// 選ばれた行を（入力の相対順序を保ったまま）出力ベクタへ移す。
    fn finish(&mut self) {
        let n = self.total_rows;
        let k = self.target.min(n);
        // 部分 Fisher–Yates: 先頭 `k` 個を一様ランダムに選ぶ。
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
                    // 中断はそのまま上へ返す。蓄積した行は `self` に残るので、
                    // 次回の呼び出しはここから入力を引き直す。
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
            assert!(guard < 9_999, "終わらない");
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

    // --- ベルヌーイ法（パーセント指定） --------------------------------------

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
        // 選ばれた行は入力の相対順序のまま。
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
        assert_eq!(
            drive(mk(false)),
            drive(mk(true)),
            "NeedIo をまたいでも結果が変わってはいけない"
        );
    }

    // --- 行数指定（一様ランダムに N 行） --------------------------------------

    #[test]
    fn selects_exactly_the_requested_row_count() {
        let vals: Vec<i32> = (0..1000).collect();
        let steps = vec![Script::Rows(vec![ints(&vals)])];
        let op = Box::new(RowSample::new(Box::new(Mock { steps, pos: 0 }), &spec(true, 100.0, 7)));
        let got = drive(op);
        assert_eq!(got.len(), 100);
        // 重複なし・入力に実在する値のみ・入力順のまま。
        let mut sorted = got.clone();
        sorted.dedup();
        assert_eq!(got, sorted, "重複してはいけない");
        assert!(got.iter().all(|v| vals.contains(v)));
        let mut asc = got.clone();
        asc.sort_unstable();
        assert_eq!(got, asc, "選ばれた行は入力の相対順序のまま");
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
        assert_eq!(
            drive(mk(false)),
            drive(mk(true)),
            "NeedIo をまたいでも結果が変わってはいけない"
        );
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
