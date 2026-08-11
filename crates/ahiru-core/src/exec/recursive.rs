//! `WITH RECURSIVE`（再帰 CTE）の不動点反復。
//!
//! ## アルゴリズム
//!
//! 1. アンカー（`UNION` の左辺）を最後まで読み切り、その全行を最初の
//!    「作業テーブル」（直前イテレーションの新規行）にする。読みながら
//!    そのまま呼び出し元へも返す（アンカーの結果も最終結果の一部）。
//! 2. 作業テーブルが空でない間、それを入力に再帰項（`UNION` の右辺）を
//!    1 回実行する。再帰項の中の自己参照は `Node::WorkingTable` という
//!    葉ノードとして現れており、実行のたびに `WorkingTableScan` へ
//!    「今回の作業テーブル」を差し込んで物理オペレータ木を組み立て直す
//!    （`super::build_ctx` 参照）。
//! 3. 再帰項が生んだ行のうち（`UNION` なら）まだ出ていないものだけを
//!    「新規行」とし、それを次のイテレーションの作業テーブルにしつつ
//!    呼び出し元へも返す。新規行が 0 件になったら終了。
//!
//! ## 重複排除
//!
//! `UNION ALL` は重複を残すので作業テーブルの構築以外に状態を持たない。
//! `UNION`（DISTINCT）は `exec::setop`/`exec::agg` と同じ
//! `rowkey::encode_key` + `HashIndex` を使い、アンカーから最後の
//! イテレーションまでを通した既出キー集合を 1 つだけ持つ（キー符号化を
//! オペレータごとに作り直さないという、このエンジン一貫の方針）。
//!
//! ## 再開可能性
//!
//! アンカー・各イテレーションの再帰項はどちらも `Step::NeedIo`/`NeedCodec`
//! を返しうる。中断はそのまま上へ返し、`phase`/`working`/`seen` などの
//! 途中状態はすべて `self` に持つので、次の `next()` は同じ場所から
//! 再開する（`exec::setop::SetOp` と同じ流儀）。1 イテレーションの物理
//! オペレータ木は `self.current` に保持し、イテレーションの境界（作業
//! テーブルの入れ替え）以外では作り直さない。
//!
//! ## 安全弁
//!
//! 終端しない再帰 CTE（例: 減少しない `WHERE` や、そもそも停止条件を
//! 書き忘れたもの）を有限時間・有限メモリで確実にエラーへ倒すため、
//! 反復回数とイテレーションあたりの作業テーブルのバイト数の両方に
//! 上限を設ける（下の定数のコメント参照）。

use crate::exec::rowkey::{encode_key, HashIndex};
use crate::exec::sort::vector_bytes;
use crate::exec::{build_ctx, ExecContext, Operator, Step};
use crate::plan::Node;
use crate::prelude::*;
use crate::vector::{Batch, Vector};

/// 不動点反復の回数上限。
///
/// DuckDB 自身はこの種の入力（例: `SELECT n+1 FROM t` に停止条件が無い
/// 再帰項）を無制限に回し続け、メモリを食い潰すまで止まらないことを
/// 実機で確認済み（`duckdb` CLI で 120 秒経っても終了しなかった）。
/// wasm ホストでこれをやると復帰不能になるため、現実的な階層データ
/// （数千〜数万段の組織図・カテゴリツリー）やグラフ探索は十分に収まり、
/// かつ暴走時は数秒で明確なエラーに落ちる値として 100,000 を選んだ。
const MAX_RECURSIVE_ITERATIONS: u32 = 100_000;

/// 1 イテレーションぶんの作業テーブル（直前イテレーションの新規行）が
/// 使ってよいおおよそのバイト数の上限。
///
/// 反復のたびに行が増え続ける（かつ減らない）ケースを、回数の上限に
/// 頼らず早期に検出するための第二の安全弁。`exec::sort::Sort`/
/// `exec::setop::SetOp` と同じ考え方で、厳密なバイト計算はしない。
const MAX_WORKING_BYTES: usize = 256 << 20;

/// `UNION`（重複排除）の既出キー集合が使ってよいおおよそのバイト数の上限。
/// `exec::setop::SetOp`/`exec::mod::DistinctOn` と同じ水準。
const MAX_SEEN_BYTES: usize = 64 << 20;

enum Phase {
    /// アンカーを読んでいる。
    Anchor,
    /// 現在のイテレーションの再帰項（`current`）を読んでいる。
    Iterate,
    Done,
}

pub struct RecursiveCte {
    anchor: Box<dyn Operator>,
    /// 再帰項の論理プラン。イテレーションごとに新しい物理オペレータ木を
    /// 組み立て直すため、実行済みでも所有し続ける（`Node: Clone`）。
    recursive_term: Node,
    phase: Phase,
    /// `Iterate` フェーズでのみ `Some`。
    current: Option<Box<dyn Operator>>,
    /// 直前のイテレーションで新しく増えた行。次のイテレーションの
    /// `Node::WorkingTable` に差し込む。
    working: Vec<Batch>,
    /// 今のイテレーション（またはアンカー）で新しく見つかった行。
    /// フェーズが終わったら `working` に差し替える。
    next_working: Vec<Batch>,
    /// `next_working` が使っているおおよそのバイト数。
    next_working_bytes: usize,
    /// `UNION`（DISTINCT）のときだけ `Some`。`UNION ALL` では重複を見ない。
    seen: Option<HashIndex>,
    keybuf: Vec<u8>,
    /// 完了した反復回数（安全弁）。
    iterations: u32,
}

impl RecursiveCte {
    pub fn new(anchor: Box<dyn Operator>, recursive_term: Node, union_all: bool) -> Self {
        RecursiveCte {
            anchor,
            recursive_term,
            phase: Phase::Anchor,
            current: None,
            working: Vec::new(),
            next_working: Vec::new(),
            next_working_bytes: 0,
            seen: if union_all { None } else { Some(HashIndex::new()) },
            keybuf: Vec::new(),
            iterations: 0,
        }
    }

    /// バッチ 1 つを処理する。`UNION` なら重複行を除き、残った行を
    /// 「次のイテレーションの作業テーブル」にも積む。戻り値は呼び出し元へ
    /// 返す出力（重複だけのバッチ、または 0 行なら `None`）。
    fn process(&mut self, mut batch: Batch) -> Result<Option<Batch>> {
        if batch.card() == 0 {
            return Ok(None);
        }
        // 以降は行番号で引く（重複判定・作業テーブルへの格納とも）ので
        // selection をここで畳む。
        batch.materialize();
        let cols = match &mut self.seen {
            None => batch.cols,
            Some(seen) => {
                let rows = batch.num_rows();
                let refs: Vec<&Vector> = batch.cols.iter().collect();
                let mut sel = Vec::with_capacity(rows);
                let mut keybuf = core::mem::take(&mut self.keybuf);
                for r in 0..rows {
                    encode_key(&refs, r, &mut keybuf);
                    if seen.get_or_insert(&keybuf).1 {
                        sel.push(r as u32);
                    }
                }
                self.keybuf = keybuf;
                ensure!(seen.approx_bytes() <= MAX_SEEN_BYTES, Oom);
                if sel.is_empty() {
                    return Ok(None);
                }
                if sel.len() == rows {
                    batch.cols
                } else {
                    batch.cols.iter().map(|c| c.gather(&sel)).collect()
                }
            }
        };

        let bytes: usize = cols.iter().map(vector_bytes).sum();
        self.next_working_bytes = self.next_working_bytes.saturating_add(bytes);
        ensure!(self.next_working_bytes <= MAX_WORKING_BYTES, Oom);

        let out = Batch::new(cols);
        self.next_working.push(clone_batch(&out));
        Ok(Some(out))
    }

    /// 現在のフェーズの入力を読み切った。次のイテレーションへ進む
    /// （新規行が無ければ `Done`）。
    fn begin_iteration(&mut self) -> Result<()> {
        self.working = core::mem::take(&mut self.next_working);
        self.next_working_bytes = 0;
        self.current = None;
        if self.working.is_empty() {
            self.phase = Phase::Done;
            return Ok(());
        }
        self.iterations += 1;
        ensure!(self.iterations <= MAX_RECURSIVE_ITERATIONS, RecursionLimitExceeded);
        self.current = Some(build_ctx(self.recursive_term.clone(), Some(&self.working))?);
        self.phase = Phase::Iterate;
        Ok(())
    }
}

impl Operator for RecursiveCte {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        loop {
            match self.phase {
                Phase::Anchor => match self.anchor.next(ctx)? {
                    Step::Ready(b) => {
                        if let Some(out) = self.process(b)? {
                            return Ok(Step::Ready(out));
                        }
                    }
                    Step::NeedIo => return Ok(Step::NeedIo),
                    Step::NeedCodec => return Ok(Step::NeedCodec),
                    Step::Done => self.begin_iteration()?,
                },
                Phase::Iterate => {
                    let op = match &mut self.current {
                        Some(op) => op,
                        // `Iterate` に入るのは `begin_iteration` が `current` を
                        // 設定した直後だけなので、必ず `Some`。
                        None => err!(Internal),
                    };
                    match op.next(ctx)? {
                        Step::Ready(b) => {
                            if let Some(out) = self.process(b)? {
                                return Ok(Step::Ready(out));
                            }
                        }
                        Step::NeedIo => return Ok(Step::NeedIo),
                        Step::NeedCodec => return Ok(Step::NeedCodec),
                        Step::Done => self.begin_iteration()?,
                    }
                }
                Phase::Done => return Ok(Step::Done),
            }
        }
    }
}

/// 再帰 CTE の再帰項内での自己参照（`Node::WorkingTable`）。直前の
/// イテレーションで新しく増えた行をそのまま返すだけの葉オペレータ。
/// データは既にメモリ上にあるので、`MemScan` と同じ理由で
/// `NeedIo`/`NeedCodec` は原理的に返らない。
pub struct WorkingTableScan {
    batches: Vec<Batch>,
    pos: usize,
}

impl WorkingTableScan {
    pub fn new(batches: Vec<Batch>) -> Self {
        WorkingTableScan { batches, pos: 0 }
    }
}

impl Operator for WorkingTableScan {
    fn next(&mut self, _ctx: &mut ExecContext) -> Result<Step> {
        if self.pos >= self.batches.len() {
            return Ok(Step::Done);
        }
        let b = core::mem::replace(&mut self.batches[self.pos], Batch::new(Vec::new()));
        self.pos += 1;
        Ok(Step::Ready(b))
    }
}

/// 作業テーブルを複製する。`Node::WorkingTable` が複数箇所（自己結合）に
/// 現れても、それぞれが独立に読み進められるよう、参照のたびに新しい
/// `Vec<Batch>` を作る。
pub(crate) fn clone_batches(src: &[Batch]) -> Vec<Batch> {
    src.iter().map(clone_batch).collect()
}

/// `Batch` は `sel`/`empty_rows` を外から複製する手段を持たないので
/// （`vector::Batch` の非公開フィールド）、selection を持たない前提で
/// 列だけを複製する。呼び出し元はすべて `materialize()` 済みのバッチしか
/// 渡さない。
fn clone_batch(b: &Batch) -> Batch {
    if b.cols.is_empty() {
        Batch::rows_only(b.num_rows())
    } else {
        Batch::new(b.cols.clone())
    }
}
