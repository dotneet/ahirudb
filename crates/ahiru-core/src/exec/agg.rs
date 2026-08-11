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

use crate::exec::rowkey::{encode_key, ord_f64, pow10, HashIndex};
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

/// 実行時に選ぶ更新規則。`AggKind` と入力の物理型から一度だけ決める。
///
/// `ApproxCountDistinct` には専用の variant を用意しない。v1 は厳密カウント
/// でよく、それは「引数に強制的に DISTINCT の重複除去を掛けた `Count`」と
/// 完全に同じ演算なので、`Op::Count` にそのまま相乗りする
/// （`HashAggregate::new` の `distinct` 配列の構築を見よ）。
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
    /// 標本標準偏差・標本分散。Welford のオンライン algorithm（平均と
    /// 平方偏差和 M2 を 1 パスで更新）で求める。「二乗の総和 − 総和の二乗」
    /// という素朴な式は桁落ちが激しく実データでは信用できないため使わない。
    StdDev,
    Variance,
    /// 連続分布の中央値（`quantile_cont(x, 0.5)` と同じ、線形補間）。
    /// ストリーミングでは正確な値を求められないので、非 NULL 値を全部
    /// 保持して出力時に 1 度だけソートする。
    Median,
    /// 最頻値。(グループ, 値) を鍵にした頻度表で数え、同数は先に見つかった
    /// 方を勝たせる（`>` 判定で更新するため自然にそうなる）。
    Mode,
    /// 区切り文字で連結する文字列集約。到着順に連結する
    /// （SQL 標準も DuckDB も `ORDER BY` 無しでは順序を保証しない）。
    StringAgg,
    /// 値を JSON 風テキストへ集める。LIST 型が無いための代替表現
    /// （DESIGN.md のネスト型の扱いと同じ判断）。NULL も要素として含める
    /// 点だけ他の集約と異なる（DuckDB の `array_agg`/`list` と同じ）。
    ArrayAgg,
}

/// 1 グループ × 1 集約の状態。
struct State {
    /// 非 NULL 入力の個数。`COUNT(*)` では全行数。ArrayAgg では NULL も
    /// 要素として数えるので「積んだ要素数」の意味になる。
    n: i64,
    /// 累積値。`Value::Null` は「まだ非 NULL 入力が無い」。
    /// StringAgg/ArrayAgg では蓄積中のテキスト（`Value::Bytes`）を兼ねる。
    acc: Value,
    /// StdDev/Variance 用の Welford 累積（オンライン平均）。他の演算では
    /// 未使用のまま 0.0。
    mean: f64,
    /// StdDev/Variance 用の Welford 累積（平方偏差和 M2）。
    m2: f64,
    /// Median が出力まで抱える非 NULL 値。
    median_vals: Vec<f64>,
    /// Mode の現在の最多得票数。
    mode_best: i64,
}

impl State {
    fn empty() -> Self {
        State { n: 0, acc: Value::Null, mean: 0.0, m2: 0.0, median_vals: Vec::new(), mode_best: 0 }
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
    /// `ApproxCountDistinct` は DISTINCT 指定の有無によらず常にここへ
    /// 重複除去表を持つ（v1 の実装は厳密カウントなので、これがそのまま
    /// 本体になる）。
    distinct: Vec<Option<HashIndex>>,
    /// Mode の (グループ, 値) → 頻度表添字。DISTINCT の重複除去表と同じ
    /// 「グループ番号を前置」方式で、集約ごとに 1 本の表を全グループで
    /// 共有する（`distinct` と同様、`aggs` と同じ並び）。
    mode_freq: Vec<Option<HashIndex>>,
    /// `mode_freq` の添字ごとの出現数。
    mode_counts: Vec<Vec<i64>>,

    /// MIN/MAX の勝者バイト列・Mode の勝者バイト列・StringAgg/ArrayAgg の
    /// 蓄積テキスト・Median の一時値など、可変長で伸びる状態のバイト数の
    /// 合計。メモリ判定に含める。
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
        let mut mode_freq = Vec::with_capacity(aggs.len());
        let mut mode_counts = Vec::with_capacity(aggs.len());
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
                AggKind::StdDev => Op::StdDev,
                AggKind::Variance => Op::Variance,
                AggKind::Median => Op::Median,
                AggKind::Mode => Op::Mode,
                // 下の `distinct` 構築で強制的に重複除去表を持たせるので、
                // 演算そのものは COUNT(DISTINCT x) と同じ `Op::Count` でよい。
                AggKind::ApproxCountDistinct => Op::Count,
                AggKind::StringAgg => Op::StringAgg,
                AggKind::ArrayAgg => Op::ArrayAgg,
            });
            // DECIMAL は内部が整数なので、AVG/StdDev/Variance/Median では
            // 10^scale で戻す。
            avg_div.push(match ity {
                Ty::Decimal { scale, .. } => pow10(scale),
                _ => 1.0,
            });
            states.push(Vec::new());
            // `COUNT(*)` に DISTINCT は付かない（引数が無い）。
            // ApproxCountDistinct は書き手が DISTINCT を書いたかどうかに
            // 関わらず常に重複除去する（それが定義そのものなので）。
            distinct.push(
                if (a.distinct || a.kind == AggKind::ApproxCountDistinct) && a.arg.is_some() {
                    Some(HashIndex::new())
                } else {
                    None
                },
            );
            mode_freq.push(if a.kind == AggKind::Mode { Some(HashIndex::new()) } else { None });
            mode_counts.push(Vec::new());
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
            mode_freq,
            mode_counts,
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
        let mut n = self.index.approx_bytes();
        n += self.key_bytes + self.acc_bytes;
        n += self.num_groups() * self.aggs.len() * core::mem::size_of::<State>();
        for d in self.distinct.iter().flatten() {
            n += d.approx_bytes();
        }
        for d in self.mode_freq.iter().flatten() {
            n += d.approx_bytes();
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
        // `FILTER (WHERE cond)`。集約ごとに独立した BOOLEAN 列を先に評価して
        // おき、行ループでは真偽を引くだけにする。
        let mut fvs = Vec::with_capacity(self.aggs.len());
        for a in &self.aggs {
            fvs.push(match &a.filter {
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
                // FILTER が偽・NULL の行はこの集約からは無かったことにする
                // （SQL の三値論理: UNKNOWN も除外側）。他の集約は影響しない。
                if let Some(fv) = &fvs[ai] {
                    // `compile_predicate` は結果型が BOOLEAN か NULL であることしか
                    // 保証しない（`WHERE NULL` 相当）。NULL 型は物理表現が Bool と
                    // 限らないので、`.bools()` を呼ぶ前に物理型を確かめる
                    // （`vm::eval_filter` と同じ防御）。
                    let pass = fv.data().phys() == PhysType::Bool
                        && fv.is_valid(row)
                        && fv.bools().get(row);
                    if !pass {
                        continue;
                    }
                }
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
                let valid = col.is_valid(row);
                // SUM/MIN/MAX/AVG/COUNT(x) は NULL を無視する。ArrayAgg だけ
                // NULL も要素として数える（DuckDB の array_agg/list と同じ）。
                if !valid && self.ops[ai] != Op::ArrayAgg {
                    continue;
                }
                if let Some(seen) = &mut self.distinct[ai] {
                    // グループ番号を前置して 1 本の表で全グループを賄う。
                    // ネストした表を持たずに済むが、(グループ, 値) の組の数
                    // だけキーが残るのでメモリはその分増える。
                    // `encode_key` は無効値も flag=0 で符号化するので、
                    // ArrayAgg(DISTINCT x) の NULL 重複除去もこのまま通せる。
                    encode_key(&[col], row, &mut vkey);
                    dkey.clear();
                    dkey.extend_from_slice(&slot.to_le_bytes());
                    dkey.extend_from_slice(&vkey);
                    if !seen.get_or_insert(&dkey).1 {
                        continue;
                    }
                }
                if valid {
                    self.update(ai, g, col, row)?;
                } else {
                    // ここに来るのは ArrayAgg の NULL 行だけ。
                    self.push_array_null(ai, g);
                }
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
            Op::StdDev | Op::Variance => {
                // Welford のオンライン更新。`st.n` はこの関数の先頭で
                // 既に +1 済みなので、そのまま「今回時点での総数」として使える。
                let x = as_f64_generic(col, row, self.avg_div[ai])?;
                let delta = x - st.mean;
                st.mean += delta / st.n as f64;
                let delta2 = x - st.mean;
                st.m2 += delta * delta2;
            }
            Op::Median => {
                // 正確な中央値はストリーミングでは求まらないので、出力まで
                // 非 NULL 値を丸ごと抱える。ソートは push_result で 1 回だけ。
                let x = as_f64_generic(col, row, self.avg_div[ai])?;
                st.median_vals.push(x);
                self.acc_bytes += 8;
            }
            Op::Mode => {
                // DISTINCT の重複除去表と同じ「グループ番号を前置」方式で、
                // 1 本の頻度表を全グループで共有する。
                let mut vkey = Vec::new();
                encode_key(&[col], row, &mut vkey);
                let mut fkey = Vec::with_capacity(4 + vkey.len());
                fkey.extend_from_slice(&(g as u32).to_le_bytes());
                fkey.extend_from_slice(&vkey);
                let freq = match &mut self.mode_freq[ai] {
                    Some(f) => f,
                    // Mode の Op には必ず対応する頻度表がある（契約違反）。
                    None => err!(Internal),
                };
                let (slot, is_new) = freq.get_or_insert(&fkey);
                if is_new {
                    self.mode_counts[ai].push(1);
                    self.acc_bytes += 8;
                } else {
                    self.mode_counts[ai][slot as usize] += 1;
                }
                let cnt = self.mode_counts[ai][slot as usize];
                // `>` なので同数のときは先に見つかった値が勝ったまま残る
                // （DuckDB で観測した挙動と同じ「先着優先」）。
                if cnt > st.mode_best {
                    st.mode_best = cnt;
                    if let Value::Bytes(b) = &st.acc {
                        self.acc_bytes = self.acc_bytes.saturating_sub(b.len());
                    }
                    let v = col.value_at(row);
                    if let Value::Bytes(b) = &v {
                        self.acc_bytes += b.len();
                    }
                    st.acc = v;
                }
            }
            Op::StringAgg => {
                let bytes = match col.data() {
                    Data::Bytes(b) => b.get(row),
                    // バインダが VARCHAR 以外を渡してきた場合の防御。
                    _ => err!(TypeMismatch),
                };
                let sep = &self.aggs[ai].separator;
                let old_len = match &st.acc {
                    Value::Bytes(b) => b.len(),
                    _ => 0,
                };
                let mut buf = match core::mem::replace(&mut st.acc, Value::Null) {
                    Value::Bytes(b) => b,
                    _ => Vec::new(),
                };
                if old_len > 0 {
                    buf.extend_from_slice(sep);
                }
                buf.extend_from_slice(bytes);
                self.acc_bytes += buf.len() - old_len;
                st.acc = Value::Bytes(buf);
            }
            Op::ArrayAgg => {
                let mut text = Vec::new();
                push_json_scalar(col, row, &mut text);
                let old_len = match &st.acc {
                    Value::Bytes(b) => b.len(),
                    _ => 0,
                };
                let buf = match core::mem::replace(&mut st.acc, Value::Null) {
                    Value::Bytes(b) => b,
                    _ => Vec::new(),
                };
                let new_buf = append_array_text(buf, &text);
                self.acc_bytes += new_buf.len() - old_len;
                st.acc = Value::Bytes(new_buf);
            }
        }
        Ok(())
    }

    /// ArrayAgg の NULL 要素を積む。他の集約と違い NULL も読み飛ばさない
    /// （DuckDB の `array_agg`/`list` は NULL を要素として結果に含める）。
    fn push_array_null(&mut self, ai: usize, g: usize) {
        let st = &mut self.states[ai][g];
        st.n += 1;
        let old_len = match &st.acc {
            Value::Bytes(b) => b.len(),
            _ => 0,
        };
        let buf = match core::mem::replace(&mut st.acc, Value::Null) {
            Value::Bytes(b) => b,
            _ => Vec::new(),
        };
        let new_buf = append_array_text(buf, b"null");
        self.acc_bytes += new_buf.len() - old_len;
        st.acc = Value::Bytes(new_buf);
    }

    /// 1 グループぶんの集約結果を出力ベクタへ積む。
    ///
    /// Median だけ `median_vals` を出力直前に 1 度ソートするため `&mut self`
    /// が要る（他の演算はここで状態を変えない）。
    fn push_result(&mut self, ai: usize, g: usize, out: &mut Vector) {
        match self.ops[ai] {
            Op::CountStar | Op::Count => out.push_value(&Value::I64(self.states[ai][g].n)),
            // 非 NULL 入力が 1 つも無いグループは NULL。StringAgg も同じ規則
            // （非 NULL 値を 1 つも足していなければ acc は Null のまま）。
            // Mode の勝者も acc に直接入っている。
            Op::SumInt | Op::SumF64 | Op::Min | Op::Max | Op::StringAgg | Op::Mode => {
                out.push_value(&self.states[ai][g].acc)
            }
            Op::AvgInt => {
                // 整数は i128 で正確に足してから 1 回だけ割る。f64 で足し込むと
                // 桁落ちが累積するため。
                let st = &self.states[ai][g];
                match &st.acc {
                    Value::I128(s) if st.n > 0 => {
                        out.push_value(&Value::F64(*s as f64 / self.avg_div[ai] / st.n as f64))
                    }
                    _ => out.push_null(),
                }
            }
            Op::AvgF64 => {
                let st = &self.states[ai][g];
                match &st.acc {
                    Value::F64(s) if st.n > 0 => out.push_value(&Value::F64(s / st.n as f64)),
                    _ => out.push_null(),
                }
            }
            Op::StdDev | Op::Variance => {
                let st = &self.states[ai][g];
                // 標本分散・標本標準偏差は n < 2 で未定義（DuckDB でも NULL）。
                if st.n < 2 {
                    out.push_null();
                } else {
                    // 浮動小数の丸め誤差で M2 がわずかに負になり得るので 0 で
                    // 下限を切る（全部同じ値のグループで分散がぴったり 0 に
                    // ならず NaN の平方根が出るのを防ぐ）。
                    let var = (st.m2 / (st.n as f64 - 1.0)).max(0.0);
                    let v = if self.ops[ai] == Op::StdDev { f_sqrt(var) } else { var };
                    out.push_value(&Value::F64(v));
                }
            }
            Op::Median => {
                let st = &mut self.states[ai][g];
                if st.median_vals.is_empty() {
                    out.push_null();
                } else {
                    // MIN/MAX と同じ「NaN は最大」の全順序で揃える
                    // （中央値の入力に NaN が混ざること自体まず無いが、
                    // 順序関数を 2 通り持たずに済ませる）。
                    st.median_vals.sort_by(|a, b| ord_f64(*a, *b));
                    let m = st.median_vals.len();
                    // quantile_cont(0.5) と同じ線形補間: h = (m-1)*0.5。
                    // `core` に `f64::floor`/`ceil` が無い（libm 側）ので、
                    // `m-1` が非負整数であることを使って整数演算だけで
                    // floor(h)/ceil(h) を出す（自動的に切り捨てになる）。
                    let lo = (m - 1) / 2;
                    let hi = m / 2;
                    let v = if lo == hi {
                        st.median_vals[lo]
                    } else {
                        let h = (m - 1) as f64 * 0.5;
                        let frac = h - lo as f64;
                        st.median_vals[lo] + (st.median_vals[hi] - st.median_vals[lo]) * frac
                    };
                    out.push_value(&Value::F64(v));
                }
            }
            Op::ArrayAgg => match &self.states[ai][g].acc {
                // 蓄積中は閉じ括弧を持たない（`append_array_text` 参照）ので
                // ここで初めて `]` を足す。
                Value::Bytes(b) => {
                    let mut v = b.clone();
                    v.push(b']');
                    out.push_value(&Value::Bytes(v));
                }
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

/// 数値列の 1 値を f64 として取り出す。DECIMAL は `div`（= 10^scale）で
/// 割って戻す。StdDev/Variance/Median は結果が必ず DOUBLE なので、
/// SUM/AVG のように整数のまま累積してから割る意味が無く、最初から f64 に
/// 寄せてしまう。
fn as_f64_generic(col: &Vector, row: usize, div: f64) -> Result<f64> {
    Ok(match col.data() {
        Data::I32(v) => v[row] as f64 / div,
        Data::I64(v) => v[row] as f64 / div,
        Data::I128(v) => v[row] as f64 / div,
        Data::F64(v) => v[row],
        _ => err!(TypeMismatch),
    })
}

/// 平方根。`core` に `f64::sqrt` は無い（libm 側）。`expr::funcs::f_sqrt` は
/// この用途向けに公開されていない（private のまま）ため、同じ考え方
/// （指数部を半分にした初期値からの Newton 法、5 回反復で倍精度に収束）で
/// 作り直す。
fn f_sqrt(x: f64) -> f64 {
    if x <= 0.0 || !x.is_finite() {
        return if x < 0.0 { f64::NAN } else { x };
    }
    let mut y = f64::from_bits((x.to_bits() + (1023u64 << 52)) >> 1);
    for _ in 0..5 {
        y = 0.5 * (y + x / y);
    }
    y
}

/// ArrayAgg の蓄積バッファへ 1 要素追記する。空なら `[` から書き始め、
/// 閉じ括弧は付けない（`push_result` が出力直前に 1 回だけ足す）。
/// NULL 要素も呼び出し側が `b"null"` を渡すだけで同じ経路を通せる。
fn append_array_text(mut buf: Vec<u8>, text: &[u8]) -> Vec<u8> {
    if buf.is_empty() {
        buf.push(b'[');
    } else {
        buf.extend_from_slice(b", ");
    }
    buf.extend_from_slice(text);
    buf
}

/// ArrayAgg の 1 要素を JSON 風テキストにして `out` へ足す。この処理系は
/// LIST 型を持たないため、DESIGN.md のネスト型の扱い（`format::jsonl` が
/// ネスト値をそのまま VARCHAR のテキストとして持つのと同じ発想）に倣い、
/// 値を JSON 風の文字列へ寄せる。ここに来る値は必ず非 NULL
/// （NULL は呼び出し側が `b"null"` を直接 `append_array_text` に渡す）。
fn push_json_scalar(col: &Vector, row: usize, out: &mut Vec<u8>) {
    match col.data() {
        Data::Bool(b) => out.extend_from_slice(if b.get(row) { b"true" } else { b"false" }),
        Data::I32(v) => push_int_text(out, v[row] as i128, 0),
        Data::I64(v) => push_int_text(out, v[row] as i128, 0),
        Data::I128(v) => {
            let scale = match col.ty() {
                Ty::Decimal { scale, .. } => scale,
                _ => 0,
            };
            push_int_text(out, v[row], scale);
        }
        Data::F64(v) => push_f64_text(out, v[row]),
        Data::Bytes(b) => push_json_string(out, b.get(row)),
    }
}

/// `v` を（DECIMAL なら `scale` 桁の小数点付きで）10 進テキストにする。
/// `expr::kernels::fmt_int` は private で外から呼べないため、
/// ArrayAgg で必要な分だけ同じ考え方で書き直す。
fn push_int_text(out: &mut Vec<u8>, v: i128, scale: u8) {
    let neg = v < 0;
    let mut u = v.unsigned_abs();
    let mut buf = [0u8; 48];
    let mut k = 0usize;
    if u == 0 {
        buf[0] = b'0';
        k = 1;
    }
    while u > 0 {
        buf[k] = b'0' + (u % 10) as u8;
        u /= 10;
        k += 1;
    }
    // 0.05 のように整数部が無い場合の先頭 0 を補う。
    while k <= scale as usize {
        buf[k] = b'0';
        k += 1;
    }
    if neg {
        out.push(b'-');
    }
    for i in (0..k).rev() {
        out.push(buf[i]);
        if scale > 0 && i == scale as usize {
            out.push(b'.');
        }
    }
}

/// f64 の簡易 10 進表記。有効 15 桁に丸めて末尾 0 を落とす。CAST 相当の
/// 完全な往復表現は要らない（ArrayAgg の表示用途のみ）ので、
/// `expr::kernels::fmt_f64`（private で呼べない）ほど厳密には作り込まない。
fn push_f64_text(out: &mut Vec<u8>, x: f64) {
    if x.is_nan() {
        out.extend_from_slice(b"NaN");
        return;
    }
    if x.is_infinite() {
        out.extend_from_slice(if x < 0.0 { b"-Infinity" } else { b"Infinity" });
        return;
    }
    if x == 0.0 {
        out.extend_from_slice(if x.is_sign_negative() { b"-0" } else { b"0" });
        return;
    }
    if x < 0.0 {
        out.push(b'-');
    }
    let v = if x < 0.0 { -x } else { x };
    // v = m * 10^e10 を保ったまま m を [1e14, 1e15) に正規化する。
    let mut m = v;
    let mut e10: i32 = 0;
    while m >= 1e15 {
        m /= 10.0;
        e10 += 1;
    }
    while m < 1e14 {
        m *= 10.0;
        e10 -= 1;
    }
    let mut mant = (m + 0.5) as u64;
    if mant >= 1_000_000_000_000_000 {
        mant /= 10;
        e10 += 1;
    }
    let mut digits = [0u8; 15];
    let mut t = mant;
    for i in (0..15).rev() {
        digits[i] = b'0' + (t % 10) as u8;
        t /= 10;
    }
    let mut k = 15usize;
    while k > 1 && digits[k - 1] == b'0' {
        k -= 1;
    }
    // v = 0.d1..dk × 10^p
    let p = e10 + 15;
    if !(1..=17).contains(&p) {
        // 指数表記。
        out.push(digits[0]);
        if k > 1 {
            out.push(b'.');
            out.extend_from_slice(&digits[1..k]);
        }
        out.push(b'e');
        let e = p - 1;
        if e < 0 {
            out.push(b'-');
        }
        push_int_text(out, if e < 0 { -(e as i128) } else { e as i128 }, 0);
    } else if (p as usize) >= k {
        out.extend_from_slice(&digits[..k]);
        for _ in 0..(p as usize - k) {
            out.push(b'0');
        }
    } else {
        out.extend_from_slice(&digits[..p as usize]);
        out.push(b'.');
        out.extend_from_slice(&digits[p as usize..k]);
    }
}

/// JSON 風の文字列リテラルを書く。フル JSON の文字コーデックまでは要らない
/// ので、往復に困る記号（引用符・バックスラッシュ・制御文字）だけ
/// エスケープする。
fn push_json_string(out: &mut Vec<u8>, s: &[u8]) {
    out.push(b'"');
    for &b in s {
        match b {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            0x00..=0x1f => {
                out.extend_from_slice(b"\\u00");
                out.push(hex_digit(b >> 4));
                out.push(hex_digit(b & 0xf));
            }
            _ => out.push(b),
        }
    }
    out.push(b'"');
}

fn hex_digit(n: u8) -> u8 {
    if n < 10 {
        b'0' + n
    } else {
        b'a' + (n - 10)
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
        Agg {
            kind,
            arg,
            distinct: false,
            name: String::from("a"),
            separator: Vec::new(),
            filter: None,
        }
    }

    fn agg_distinct(kind: AggKind, arg: Program) -> Agg {
        Agg {
            kind,
            arg: Some(arg),
            distinct: true,
            name: String::from("a"),
            separator: Vec::new(),
            filter: None,
        }
    }

    /// 区切り文字付きの StringAgg。
    fn agg_sep(kind: AggKind, arg: Program, sep: &[u8]) -> Agg {
        Agg {
            kind,
            arg: Some(arg),
            distinct: false,
            name: String::from("a"),
            separator: sep.to_vec(),
            filter: None,
        }
    }

    /// `FILTER (WHERE cond)` 付きの集約。
    fn agg_filter(kind: AggKind, arg: Option<Program>, filter: Program) -> Agg {
        Agg {
            kind,
            arg,
            distinct: false,
            name: String::from("a"),
            separator: Vec::new(),
            filter: Some(filter),
        }
    }

    /// 浮動小数の許容誤差付き比較。Welford は丸め誤差を多少出すため。
    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
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

    // --- FILTER (WHERE cond) -------------------------------------------------

    #[test]
    fn filter_restricts_which_rows_feed_the_aggregate() {
        // col0 = value, col1 = 条件列。FILTER (WHERE col1 > 5) を模す。
        let steps = batches(vec![vec![
            i32s(&[Some(1), Some(2), Some(3), Some(4)]),
            i32s(&[Some(10), Some(1), Some(10), Some(1)]),
        ]]);
        let op = build(
            steps,
            vec![],
            vec![
                // FILTER 無し: 全行を数える／足す。
                agg(AggKind::CountStar, None),
                agg(AggKind::Sum, Some(load(Ty::Int, 0))),
                // FILTER あり: col1 > 5 の行（1, 3 番目）だけを数える／足す。
                agg_filter(
                    AggKind::CountStar,
                    None,
                    cmp_const(OpCode::Gt, Ty::Int, 1, Value::I32(5)),
                ),
                agg_filter(
                    AggKind::Sum,
                    Some(load(Ty::Int, 0)),
                    cmp_const(OpCode::Gt, Ty::Int, 1, Value::I32(5)),
                ),
            ],
            None,
        );
        let rows = run(op).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::I64(4), "FILTER 無しの COUNT(*) は全行");
        assert_eq!(rows[0][1], Value::I128(10), "FILTER 無しの SUM は全行 (1+2+3+4)");
        assert_eq!(rows[0][2], Value::I64(2), "FILTER 付きの COUNT(*) は条件を満たす行だけ");
        assert_eq!(rows[0][3], Value::I128(4), "FILTER 付きの SUM は 1 番目と 3 番目 (1+3)");
    }

    #[test]
    fn filter_is_independent_per_aggregate_and_group() {
        let steps = batches(vec![vec![
            strs(&[Some("a"), Some("a"), Some("b"), Some("b")]),
            i32s(&[Some(1), Some(2), Some(3), Some(4)]),
            // 条件列: グループ a は 1 行目だけ真、グループ b は 2 行目だけ真。
            i32s(&[Some(1), Some(0), Some(0), Some(1)]),
        ]]);
        let cond = cmp_const(OpCode::Eq, Ty::Int, 2, Value::I32(1));
        let op = build(
            steps,
            vec![load(Ty::Varchar, 0)],
            vec![agg_filter(AggKind::Sum, Some(load(Ty::Int, 1)), cond)],
            None,
        );
        let rows = sorted(run(op).unwrap());
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][0], Value::Bytes(b"a".to_vec()));
        assert_eq!(rows[0][1], Value::I128(1), "グループ a は 1 行目だけが条件を満たす");
        assert_eq!(rows[1][0], Value::Bytes(b"b".to_vec()));
        assert_eq!(rows[1][1], Value::I128(4), "グループ b は 2 行目だけが条件を満たす");
    }

    #[test]
    fn filter_that_matches_nothing_yields_count_zero_and_sum_null() {
        let steps = batches(vec![vec![i32s(&[Some(1), Some(2)])]]);
        let op = build(
            steps,
            vec![],
            vec![
                agg_filter(
                    AggKind::CountStar,
                    None,
                    cmp_const(OpCode::Gt, Ty::Int, 0, Value::I32(100)),
                ),
                agg_filter(
                    AggKind::Sum,
                    Some(load(Ty::Int, 0)),
                    cmp_const(OpCode::Gt, Ty::Int, 0, Value::I32(100)),
                ),
            ],
            None,
        );
        let rows = run(op).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::I64(0), "COUNT(*) FILTER は該当 0 件なら 0");
        assert!(rows[0][1].is_null(), "SUM FILTER は該当 0 件なら NULL");
    }

    #[test]
    fn filter_applies_to_statistical_aggregates() {
        // 全 8 値 (2,4,4,4,5,5,7,9) のうち FILTER (WHERE col1 > 5) が選ぶのは
        // col1 に 10 を入れた 3 番目・6 番目・8 番目（値 4, 5, 9）。
        // FILTER 無しの StdDev/Median は全 8 値を、FILTER 付きは絞った 3 値だけを見る。
        let steps = batches(vec![vec![
            i32s(&[Some(2), Some(4), Some(4), Some(4), Some(5), Some(5), Some(7), Some(9)]),
            i32s(&[Some(1), Some(1), Some(10), Some(1), Some(1), Some(10), Some(1), Some(10)]),
            // StringAgg は VARCHAR しか受け付けないので、col0 と同じ値をテキストで並行して持つ。
            strs(&[
                Some("2"),
                Some("4"),
                Some("4"),
                Some("4"),
                Some("5"),
                Some("5"),
                Some("7"),
                Some("9"),
            ]),
        ]]);
        let cond = cmp_const(OpCode::Gt, Ty::Int, 1, Value::I32(5));
        let op = build(
            steps,
            vec![],
            vec![
                agg(AggKind::Median, Some(load(Ty::Int, 0))),
                agg_filter(AggKind::Median, Some(load(Ty::Int, 0)), cond.clone()),
                agg(AggKind::StringAgg, Some(load(Ty::Varchar, 2))),
                agg_filter(AggKind::StringAgg, Some(load(Ty::Varchar, 2)), cond),
            ],
            None,
        );
        let rows = run(op).unwrap();
        assert_eq!(rows[0][0], Value::F64(4.5), "FILTER 無しの中央値は全 8 値から");
        assert_eq!(rows[0][1], Value::F64(5.0), "FILTER 付きの中央値は 4,5,9 の 3 値から（中央）");
        // 区切り文字は agg() の既定（空文字）。区切り文字自体のテストは別にある。
        assert_eq!(rows[0][2], Value::Bytes(b"24445579".to_vec()));
        assert_eq!(rows[0][3], Value::Bytes(b"459".to_vec()), "FILTER 付きの StringAgg");
    }

    // --- 複数グループでの状態分離 -----------------------------------------------

    #[test]
    fn statistical_aggregates_keep_separate_state_per_group_across_interleaved_batches() {
        // グループ a / b の行を複数バッチにまたがって交互に流し込み、
        // Welford の mean/m2、中央値バッファ、Mode の頻度表、StringAgg/ArrayAgg の
        // 蓄積バッファがグループ間で混線しないことを確認する。
        let steps = batches(vec![
            vec![
                strs(&[Some("a"), Some("b"), Some("a")]),
                i32s(&[Some(1), Some(10), Some(2)]),
                strs(&[Some("1"), Some("10"), Some("2")]),
            ],
            vec![
                strs(&[Some("b"), Some("a"), Some("b")]),
                i32s(&[Some(20), Some(3), Some(10)]),
                strs(&[Some("20"), Some("3"), Some("10")]),
            ],
        ]);
        let op = build(
            steps,
            vec![load(Ty::Varchar, 0)],
            vec![
                agg(AggKind::StdDev, Some(load(Ty::Int, 1))),
                agg(AggKind::Median, Some(load(Ty::Int, 1))),
                agg(AggKind::Mode, Some(load(Ty::Int, 1))),
                agg(AggKind::StringAgg, Some(load(Ty::Varchar, 2))),
            ],
            None,
        );
        let rows = sorted(run(op).unwrap());
        assert_eq!(rows.len(), 2);
        // グループ a: 1, 2, 3。
        assert_eq!(rows[0][0], Value::Bytes(b"a".to_vec()));
        assert!(close(rows[0][1].as_f64().unwrap(), 1.0), "a の stddev = {:?}", rows[0][1]);
        assert_eq!(rows[0][2], Value::F64(2.0), "a の median");
        assert_eq!(rows[0][3], Value::I32(1), "a の mode（全部 1 回ずつなので先着の 1）");
        // 区切り文字は agg() の既定（空文字）。区切り文字自体のテストは別にある。
        assert_eq!(rows[0][4], Value::Bytes(b"123".to_vec()), "a の StringAgg（到着順）");
        // グループ b: 10, 20, 10。
        assert_eq!(rows[1][0], Value::Bytes(b"b".to_vec()));
        assert!(
            close(rows[1][1].as_f64().unwrap(), 5.773502691896258),
            "b の stddev = {:?}",
            rows[1][1]
        );
        assert_eq!(rows[1][2], Value::F64(10.0), "b の median");
        assert_eq!(rows[1][3], Value::I32(10), "b の mode（10 が 2 回で最頻）");
        assert_eq!(rows[1][4], Value::Bytes(b"102010".to_vec()), "b の StringAgg（到着順）");
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

    // --- StdDev / Variance ---------------------------------------------------

    #[test]
    fn stddev_and_variance_match_hand_verified_welford_result() {
        // 2,4,4,4,5,5,7,9 → 分散 32/7 ≈ 4.571428571428571、
        // 標準偏差 ≈ 2.138089935299395（duckdb の stddev_samp/var_samp と照合済み）。
        let steps = batches(vec![vec![i32s(&[
            Some(2),
            Some(4),
            Some(4),
            Some(4),
            Some(5),
            Some(5),
            Some(7),
            Some(9),
        ])]]);
        let op = build(
            steps,
            vec![],
            vec![
                agg(AggKind::Variance, Some(load(Ty::Int, 0))),
                agg(AggKind::StdDev, Some(load(Ty::Int, 0))),
            ],
            None,
        );
        let rows = run(op).unwrap();
        let var = rows[0][0].as_f64().unwrap();
        let sd = rows[0][1].as_f64().unwrap();
        assert!(close(var, 32.0 / 7.0), "variance = {var}");
        assert!(close(sd, 2.138089935299395), "stddev = {sd}");
    }

    #[test]
    fn stddev_variance_below_two_values_is_null() {
        // n == 0（空入力）。
        let op = build(vec![], vec![], vec![agg(AggKind::StdDev, Some(load(Ty::Int, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::Null);

        // n == 1。標本分散・標本標準偏差は未定義（duckdb でも NULL）。
        let steps = batches(vec![vec![i32s(&[Some(5)])]]);
        let op = build(steps, vec![], vec![agg(AggKind::Variance, Some(load(Ty::Int, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::Null);
    }

    #[test]
    fn variance_of_constant_values_is_exactly_zero() {
        // 丸め誤差で M2 がわずかに負になっても 0 にクランプされることを確認。
        let steps = batches(vec![vec![f64s(&[Some(5.0), Some(5.0), Some(5.0)])]]);
        let op = build(
            steps,
            vec![],
            vec![
                agg(AggKind::Variance, Some(load(Ty::Double, 0))),
                agg(AggKind::StdDev, Some(load(Ty::Double, 0))),
            ],
            None,
        );
        let rows = run(op).unwrap();
        assert_eq!(rows[0][0], Value::F64(0.0));
        assert_eq!(rows[0][1], Value::F64(0.0));
    }

    // --- Median ----------------------------------------------------------------

    #[test]
    fn median_even_and_odd_counts() {
        let steps = batches(vec![vec![i32s(&[Some(1), Some(2), Some(3), Some(4)])]]);
        let op = build(steps, vec![], vec![agg(AggKind::Median, Some(load(Ty::Int, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::F64(2.5), "偶数個は中央 2 値の線形補間");

        let steps = batches(vec![vec![i32s(&[Some(1), Some(2), Some(3), Some(4), Some(5)])]]);
        let op = build(steps, vec![], vec![agg(AggKind::Median, Some(load(Ty::Int, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::F64(3.0), "奇数個はちょうど中央値");
    }

    #[test]
    fn median_of_decimal_matches_duckdb_quantile_cont() {
        // DECIMAL(10,2) の 1.00,2.00,3.00,4.00 → duckdb では median = 2.50
        // (DECIMAL型のまま)。このエンジンは Median の出力型を常に DOUBLE に
        // 決めている（`AggKind::result_ty`）ので、値は 2.5 (DOUBLE) になる。
        let ty = Ty::Decimal { precision: 10, scale: 2 };
        let steps = batches(vec![vec![col(
            ty,
            &[
                Some(Value::I64(100)),
                Some(Value::I64(200)),
                Some(Value::I64(300)),
                Some(Value::I64(400)),
            ],
        )]]);
        let op = build(steps, vec![], vec![agg(AggKind::Median, Some(load(ty, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::F64(2.5));
    }

    #[test]
    fn median_empty_group_is_null() {
        let op = build(vec![], vec![], vec![agg(AggKind::Median, Some(load(Ty::Int, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::Null);
    }

    // --- Mode --------------------------------------------------------------

    #[test]
    fn mode_has_a_clear_winner() {
        let steps = batches(vec![vec![i32s(&[Some(1), Some(2), Some(2), Some(2), Some(3)])]]);
        let op = build(steps, vec![], vec![agg(AggKind::Mode, Some(load(Ty::Int, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::I32(2));
    }

    #[test]
    fn mode_tie_breaks_to_first_encountered_value() {
        // duckdb で観測した挙動: 同数タイのときは先に現れた値が勝つ。
        // (SELECT mode(x) FROM (SELECT unnest([3,2,2,1,1]) AS x) → 2)
        let steps = batches(vec![vec![i32s(&[Some(3), Some(2), Some(2), Some(1), Some(1)])]]);
        let op = build(steps, vec![], vec![agg(AggKind::Mode, Some(load(Ty::Int, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::I32(2), "2 が 1 より先に現れる");
    }

    #[test]
    fn mode_empty_group_is_null() {
        let op = build(vec![], vec![], vec![agg(AggKind::Mode, Some(load(Ty::Int, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::Null);
    }

    #[test]
    fn mode_distinct_dedups_before_counting() {
        // DISTINCT を付けると重複除去が Mode の頻度表より手前で効くので、
        // 全ての値が「1 回だけ」に見える。duckdb で観測した挙動と同じく、
        // 最初に現れた distinct 値が勝つ（SELECT mode(DISTINCT x) FROM
        // (SELECT unnest([1,1,2,3]) AS x) → 1）。
        let steps = batches(vec![vec![i32s(&[Some(1), Some(1), Some(2), Some(3)])]]);
        let op = build(steps, vec![], vec![agg_distinct(AggKind::Mode, load(Ty::Int, 0))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::I32(1));
    }

    // --- Median / Mode の再開 ------------------------------------------------

    #[test]
    fn need_io_mid_input_does_not_change_median_or_mode_result() {
        let data: Vec<Vec<Option<i32>>> = vec![
            vec![Some(5), Some(1), Some(9)],
            vec![Some(3), Some(7)],
            vec![Some(1), Some(1), Some(8)],
        ];
        let make = |interrupt: bool| {
            let mut steps = Vec::new();
            for (i, v) in data.iter().enumerate() {
                if interrupt && i == 1 {
                    steps.push(MockStep::NeedIo);
                }
                steps.push(MockStep::Rows(Batch::new(vec![i32s(v)])));
                if interrupt && i == 1 {
                    steps.push(MockStep::NeedCodec);
                }
            }
            build(
                steps,
                vec![],
                vec![
                    agg(AggKind::Median, Some(load(Ty::Int, 0))),
                    agg(AggKind::Mode, Some(load(Ty::Int, 0))),
                ],
                None,
            )
        };
        let plain = run(make(false)).unwrap();
        let interrupted = run(make(true)).unwrap();
        assert_eq!(plain, interrupted, "NeedIo/NeedCodec をまたいでも結果が変わってはいけない");
        // 値そのものも確認: 1,1,1,3,5,7,8,9 → median = (3+5)/2 = 4.0, mode = 1。
        assert_eq!(plain[0][0], Value::F64(4.0));
        assert_eq!(plain[0][1], Value::I32(1));
    }

    // --- ApproxCountDistinct -------------------------------------------------

    #[test]
    fn approx_count_distinct_matches_count_distinct_exactly() {
        // v1 は厳密カウント。COUNT(DISTINCT x) と同じ結果になるはず。
        let steps = batches(vec![vec![i32s(&[Some(1), Some(1), Some(2), Some(3), None, Some(3)])]]);
        let op = build(
            steps,
            vec![],
            vec![
                agg(AggKind::ApproxCountDistinct, Some(load(Ty::Int, 0))),
                agg_distinct(AggKind::Count, load(Ty::Int, 0)),
            ],
            None,
        );
        let rows = run(op).unwrap();
        assert_eq!(rows[0][0], rows[0][1]);
        assert_eq!(rows[0][0], Value::I64(3));
    }

    #[test]
    fn approx_count_distinct_of_empty_input_is_zero() {
        let op = build(
            vec![],
            vec![],
            vec![agg(AggKind::ApproxCountDistinct, Some(load(Ty::Int, 0)))],
            None,
        );
        assert_eq!(run(op).unwrap()[0][0], Value::I64(0));
    }

    // --- StringAgg -----------------------------------------------------------

    #[test]
    fn string_agg_default_separator_is_empty_and_explicit_separator_is_used() {
        // このファイルは `a.separator` をそのまま読むだけ。省略時に何を
        // 詰めるかはバインダの契約（`AggKind::optional_arg_default`）で、
        // 現状は空文字。duckdb の既定はカンマなので値は異なるが、
        // これはバインダ側の判断でありこのファイルの管轄外。
        let steps = batches(vec![vec![strs(&[Some("a"), Some("b"), Some("c")])]]);
        let op =
            build(steps, vec![], vec![agg(AggKind::StringAgg, Some(load(Ty::Varchar, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::Bytes(b"abc".to_vec()));

        let steps = batches(vec![vec![strs(&[Some("a"), Some("b"), Some("c")])]]);
        let op = build(
            steps,
            vec![],
            vec![agg_sep(AggKind::StringAgg, load(Ty::Varchar, 0), b",")],
            None,
        );
        assert_eq!(run(op).unwrap()[0][0], Value::Bytes(b"a,b,c".to_vec()));
    }

    #[test]
    fn string_agg_skips_nulls_and_all_null_group_is_null() {
        let steps = batches(vec![vec![strs(&[Some("a"), None, Some("b")])]]);
        let op = build(
            steps,
            vec![],
            vec![agg_sep(AggKind::StringAgg, load(Ty::Varchar, 0), b",")],
            None,
        );
        assert_eq!(run(op).unwrap()[0][0], Value::Bytes(b"a,b".to_vec()));

        let steps = batches(vec![vec![strs(&[None, None])]]);
        let op = build(
            steps,
            vec![],
            vec![agg_sep(AggKind::StringAgg, load(Ty::Varchar, 0), b",")],
            None,
        );
        assert_eq!(run(op).unwrap()[0][0], Value::Null, "非 NULL が 1 つも無ければ NULL");
    }

    #[test]
    fn string_agg_distinct_dedups_values() {
        let steps = batches(vec![vec![strs(&[Some("a"), Some("a"), Some("b")])]]);
        let op = build(
            steps,
            vec![],
            vec![Agg {
                kind: AggKind::StringAgg,
                arg: Some(load(Ty::Varchar, 0)),
                distinct: true,
                name: String::from("a"),
                separator: b",".to_vec(),
                filter: None,
            }],
            None,
        );
        let rows = run(op).unwrap();
        let s = match &rows[0][0] {
            Value::Bytes(b) => b.clone(),
            other => panic!("bytes を期待したが {other:?}"),
        };
        // DISTINCT の到着順はハッシュ表の実装依存で厳密には約束しないので、
        // 集合として {a, b} であることだけ確認する。
        let mut parts: Vec<&[u8]> = s.split(|&c| c == b',').collect();
        parts.sort();
        assert_eq!(parts, vec![b"a".as_slice(), b"b".as_slice()]);
    }

    // --- ArrayAgg --------------------------------------------------------------

    #[test]
    fn array_agg_integers_and_varchars() {
        let steps = batches(vec![vec![i32s(&[Some(1), Some(2), Some(3)])]]);
        let op = build(steps, vec![], vec![agg(AggKind::ArrayAgg, Some(load(Ty::Int, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::Bytes(b"[1, 2, 3]".to_vec()));

        let steps = batches(vec![vec![strs(&[Some("a"), Some("b")])]]);
        let op =
            build(steps, vec![], vec![agg(AggKind::ArrayAgg, Some(load(Ty::Varchar, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::Bytes(b"[\"a\", \"b\"]".to_vec()));
    }

    #[test]
    fn array_agg_includes_null_elements_as_literal_null() {
        // ArrayAgg は他の集約と違い NULL も要素として数える
        // （duckdb の array_agg/list も同じ: [1, NULL, 3]）。
        let steps = batches(vec![vec![i32s(&[Some(1), None, Some(3)])]]);
        let op = build(steps, vec![], vec![agg(AggKind::ArrayAgg, Some(load(Ty::Int, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::Bytes(b"[1, null, 3]".to_vec()));
    }

    #[test]
    fn array_agg_of_f64_uses_shortest_decimal_or_exponential_form() {
        // push_f64_text は INT/VARCHAR の array_agg テストでは一切通らない。
        // 通常範囲・指数表記に切り替わる境界（p が 1..=17 の外）・NaN/Infinity
        // をここでまとめて確認する。duckdb の to_json(x) の書式に合わせてある。
        let steps = batches(vec![vec![f64s(&[
            Some(1.5),
            Some(1e20),  // p = 21 > 17 → 指数表記
            Some(1e-10), // p = -9 < 1 → 指数表記
            Some(f64::NAN),
            Some(f64::INFINITY),
        ])]]);
        let op =
            build(steps, vec![], vec![agg(AggKind::ArrayAgg, Some(load(Ty::Double, 0)))], None);
        let s = match &run(op).unwrap()[0][0] {
            Value::Bytes(b) => b.clone(),
            other => panic!("bytes を期待したが {other:?}"),
        };
        assert_eq!(s, b"[1.5, 1e20, 1e-10, NaN, Infinity]".to_vec());
    }

    #[test]
    fn array_agg_of_zero_rows_is_null_not_empty_array() {
        // duckdb でも array_agg(x) は「0 行」なら NULL（`[]` ではない）。
        // NULL だけの行がある場合とは違う（そちらは `[NULL]` になる。
        // 上の `array_agg_includes_null_elements_as_literal_null` 参照）。
        let op = build(vec![], vec![], vec![agg(AggKind::ArrayAgg, Some(load(Ty::Int, 0)))], None);
        assert_eq!(run(op).unwrap()[0][0], Value::Null);
    }

    #[test]
    fn array_agg_distinct_dedups_nulls_too() {
        let steps = batches(vec![vec![i32s(&[Some(1), None, Some(1), None, Some(2)])]]);
        let op =
            build(steps, vec![], vec![agg_distinct(AggKind::ArrayAgg, load(Ty::Int, 0))], None);
        let s = match &run(op).unwrap()[0][0] {
            Value::Bytes(b) => b.clone(),
            other => panic!("bytes を期待したが {other:?}"),
        };
        // 順序は約束しないので要素の集合だけ確認する。
        let inner = core::str::from_utf8(&s[1..s.len() - 1]).unwrap();
        let mut parts: Vec<&str> = inner.split(", ").collect();
        parts.sort();
        assert_eq!(parts, vec!["1", "2", "null"]);
    }

    // --- メモリ上限（新しい演算） ---------------------------------------------

    #[test]
    fn memory_limit_is_enforced_for_array_agg() {
        // MIN/MAX と同じ発想: 大きなバイト列を大量に積んで上限に当てる。
        // ここは単一グループなので、蓄積される JSON 風テキスト自体が
        // 上限を超えるまで伸び続ける。
        let filler = "x".repeat(1024);
        let mut steps = Vec::new();
        for _ in 0..80 {
            let vals: Vec<Option<String>> = (0..1000).map(|_| Some(filler.clone())).collect();
            let refs: Vec<Option<&str>> = vals.iter().map(|s| s.as_deref()).collect();
            steps.push(MockStep::Rows(Batch::new(vec![strs(&refs)])));
        }
        let op =
            build(steps, vec![], vec![agg(AggKind::ArrayAgg, Some(load(Ty::Varchar, 0)))], None);
        assert_eq!(code_of(run(op)), Some(Code::Oom));
    }

    #[test]
    fn memory_limit_is_enforced_for_median() {
        // median_vals は 1 値あたり 8 バイトとして数える。64MiB を超えるには
        // 838 万値強が要るので、余裕を見て 900 万行を流し込む。
        let mut steps = Vec::new();
        for _ in 0..90 {
            let vals: Vec<Option<i32>> = (0..100_000).map(Some).collect();
            steps.push(MockStep::Rows(Batch::new(vec![i32s(&vals)])));
        }
        let op = build(steps, vec![], vec![agg(AggKind::Median, Some(load(Ty::Int, 0)))], None);
        assert_eq!(code_of(run(op)), Some(Code::Oom));
    }
}
