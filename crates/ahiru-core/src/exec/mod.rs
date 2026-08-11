//! 物理オペレータ（プル型 Volcano、ベクトル化）。
//!
//! プッシュ型のほうが速いが、プル型のほうがコード量が小さく、RowGroup 境界の
//! ステップ実行（DESIGN.md §6）に自然に載るのでプル型を採る。1 回の `next()`
//! で 2048 行を返すため、Volcano の呼び出しオーバーヘッドは相対的に無視できる。

pub mod agg;
pub mod join;
pub mod rowkey;
pub mod sort;

use crate::catalog::Catalog;
use crate::expr::vm::Vm;
use crate::plan::{Node, ScanSpec};
use crate::prelude::*;
use crate::vector::{Batch, Field, Vector, BATCH_SIZE};

/// ホストに要求するバイト範囲。
#[derive(Clone, Copy)]
pub struct IoRequest {
    /// カタログ上のテーブル添字。
    pub table: usize,
    pub offset: u64,
    pub len: u64,
}

/// ホストに展開を依頼する圧縮ブロック。
///
/// wasm コアが内蔵しないコーデック（GZIP / ZSTD）はここを通る。GZIP は
/// ブラウザの `DecompressionStream`、ZSTD は別 wasm モジュールが処理する
/// （DESIGN.md §6）。エンジンから見ればどちらも同じ「ホストに頼む作業」。
#[derive(Clone, Copy)]
pub struct CodecRequest {
    pub table: usize,
    pub codec: crate::parquet::Compression,
    pub offset: u64,
    pub len: u32,
    pub out_len: u32,
}

pub enum Step {
    Ready(Batch),
    /// 必要なバイトが未取得。`ExecContext::io` に要求が積まれている。
    NeedIo,
    /// 内蔵していないコーデックの展開が必要。`ExecContext::codec` に要求がある。
    NeedCodec,
    Done,
}

pub struct ExecContext<'a> {
    pub catalog: &'a mut Catalog,
    pub vm: &'a mut Vm,
    pub io: Vec<IoRequest>,
    pub codec: Vec<CodecRequest>,
}

pub trait Operator {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step>;
}

/// 論理プランから物理オペレータ木を組み立てる。
pub fn build(node: Node) -> Result<Box<dyn Operator>> {
    Ok(match node {
        Node::Scan(spec) => Box::new(Scan::new(*spec)),
        Node::Filter { input, pred } => Box::new(Filter { input: build(*input)?, pred }),
        Node::Project { input, exprs, .. } => Box::new(Project { input: build(*input)?, exprs }),
        Node::Limit { input, limit, offset } => {
            Box::new(Limit { input: build(*input)?, limit, offset, seen: 0, emitted: 0 })
        }
        Node::Aggregate { input, groups, aggs, having, .. } => {
            Box::new(agg::HashAggregate::new(build(*input)?, groups, aggs, having)?)
        }
        Node::Sort { input, keys, limit } => {
            Box::new(sort::Sort::new(build(*input)?, keys, limit)?)
        }
        Node::Join { left, right, kind, left_keys, right_keys, residual, .. } => {
            let lt: Vec<crate::vector::Ty> = left.schema().iter().map(|f| f.ty).collect();
            let rt: Vec<crate::vector::Ty> = right.schema().iter().map(|f| f.ty).collect();
            Box::new(join::HashJoin::new(
                build(*left)?,
                build(*right)?,
                kind,
                left_keys,
                right_keys,
                residual,
                lt,
                rt,
            )?)
        }
    })
}

// --- Scan -------------------------------------------------------------------

pub struct Scan {
    spec: ScanSpec,
    /// 次に読む分割（Parquet なら RowGroup、CSV ならバイトチャンク）。
    split: usize,
    /// 復号済みの分割。
    cur: Option<Decoded>,
}

struct Decoded {
    cols: Vec<Vector>,
    rows: usize,
    /// 次に返す行の先頭。
    pos: usize,
}

impl Scan {
    pub fn new(spec: ScanSpec) -> Self {
        Scan { spec, split: 0, cur: None }
    }
}

impl Operator for Scan {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        loop {
            // 復号済みの分割が残っていればそこから返す。
            if let Some(d) = &mut self.cur {
                if d.pos < d.rows {
                    let n = (d.rows - d.pos).min(BATCH_SIZE);
                    // TODO: 連続範囲のコピーで済むので、専用のスライスを用意すれば
                    // この添字配列は不要になる。
                    let idx: Vec<u32> = (d.pos as u32..(d.pos + n) as u32).collect();
                    let cols = d.cols.iter().map(|c| c.gather(&idx)).collect();
                    d.pos += n;
                    return Ok(Step::Ready(Batch::new(cols)));
                }
                self.cur = None;
            }

            let table = match ctx.catalog.get(self.spec.table) {
                Some(t) => t,
                None => err!(TableNotFound),
            };
            let fmt = &table.format;
            if self.split >= fmt.num_splits() {
                return Ok(Step::Done);
            }

            // 統計で落とせるならバイトを 1 つも取らずに次へ進む。
            // 統計を持たないフォーマットは既定実装で常に通す。
            if !fmt.may_match(self.split, &self.spec.pruners, &self.spec.columns) {
                self.split += 1;
                continue;
            }

            // この分割に必要なバイト範囲を集め、未取得なら一括で要求する。
            // 分割境界でしか I/O を待たないので、オペレータは非同期を意識しない
            // （DESIGN.md §6）。
            let mut ranges = Vec::with_capacity(self.spec.columns.len());
            fmt.split_ranges(self.split, &self.spec.columns, &mut ranges)?;
            let mut missing = Vec::new();
            for (start, end) in &ranges {
                if let Some((o, l)) = table.source.missing(*start, end - start) {
                    missing.push(IoRequest { table: self.spec.table, offset: o, len: l });
                }
            }
            if !missing.is_empty() {
                ctx.io.extend_from_slice(&missing);
                return Ok(Step::NeedIo);
            }

            // 内蔵していないコーデックはホストに展開してもらう。ページヘッダは
            // 非圧縮なので、この時点で必要な作業をすべて洗い出せる。
            let mut tasks = Vec::new();
            fmt.codec_tasks(&table.source, self.split, &self.spec.columns, &mut tasks)?;
            let mut pending = Vec::new();
            for t in &tasks {
                if !table.source.has_decoded(t.offset, t.len) {
                    pending.push(CodecRequest {
                        table: self.spec.table,
                        codec: t.codec,
                        offset: t.offset,
                        len: t.len,
                        out_len: t.out_len,
                    });
                }
            }
            if !pending.is_empty() {
                ctx.codec.extend_from_slice(&pending);
                return Ok(Step::NeedCodec);
            }

            let cols = fmt.read_split(&table.source, self.split, &self.spec.columns)?;
            // フォーマット実装の契約: 射影と同じ個数・同じ長さの列を返す。
            ensure!(cols.len() == self.spec.columns.len(), Internal);
            let rows = cols.first().map_or(0, |c| c.len());
            ensure!(cols.iter().all(|c| c.len() == rows), Internal);

            // 展開済みページはこの分割でしか使わない。抱えたままだと
            // 圧縮前のファイルより大きなメモリを持つことになる。
            if let Some(t) = ctx.catalog.get_mut(self.spec.table) {
                t.source.clear_decoded();
            }

            self.split += 1;
            if rows == 0 {
                // 空の分割（空の CSV チャンクなど）は読み飛ばす。
                continue;
            }
            self.cur = Some(Decoded { cols, rows, pos: 0 });
        }
    }
}

// --- Filter -----------------------------------------------------------------

pub struct Filter {
    input: Box<dyn Operator>,
    pred: crate::expr::Program,
}

impl Operator for Filter {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        loop {
            let mut batch = match self.input.next(ctx)? {
                Step::Ready(b) => b,
                other => return Ok(other),
            };
            let mut sel = Vec::new();
            ctx.vm.eval_filter(&self.pred, &batch, &mut sel)?;
            if sel.is_empty() {
                // 全行落ちたバッチは上流に返さず次を引く。
                continue;
            }
            batch.sel = Some(sel);
            return Ok(Step::Ready(batch));
        }
    }
}

// --- Project ----------------------------------------------------------------

pub struct Project {
    input: Box<dyn Operator>,
    exprs: Vec<crate::expr::Program>,
}

impl Operator for Project {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        let batch = match self.input.next(ctx)? {
            Step::Ready(b) => b,
            other => return Ok(other),
        };
        let rows = batch.card();
        let mut cols = Vec::with_capacity(self.exprs.len());
        for p in &self.exprs {
            cols.push(ctx.vm.eval(p, &batch)?);
        }
        if cols.is_empty() {
            return Ok(Step::Ready(Batch::rows_only(rows)));
        }
        Ok(Step::Ready(Batch::new(cols)))
    }
}

// --- Limit ------------------------------------------------------------------

pub struct Limit {
    input: Box<dyn Operator>,
    limit: Option<u64>,
    offset: u64,
    /// OFFSET 消化のために読み飛ばした行数。
    seen: u64,
    emitted: u64,
}

impl Operator for Limit {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        loop {
            if let Some(l) = self.limit {
                if self.emitted >= l {
                    return Ok(Step::Done);
                }
            }
            let mut batch = match self.input.next(ctx)? {
                Step::Ready(b) => b,
                other => return Ok(other),
            };
            let card = batch.card() as u64;

            // OFFSET をまだ消化しきっていない。
            if self.seen + card <= self.offset {
                self.seen += card;
                continue;
            }
            let skip = self.offset.saturating_sub(self.seen);
            self.seen += card;

            let mut take = card - skip;
            if let Some(l) = self.limit {
                take = take.min(l - self.emitted);
            }
            if take == 0 {
                continue;
            }
            self.emitted += take;

            if skip > 0 || take < card {
                let base: Vec<u32> = match &batch.sel {
                    Some(s) => s[skip as usize..(skip + take) as usize].to_vec(),
                    None => (skip as u32..(skip + take) as u32).collect(),
                };
                batch.sel = Some(base);
            }
            return Ok(Step::Ready(batch));
        }
    }
}

/// 実行結果のスキーマ。
pub fn result_schema(node: &Node) -> Vec<Field> {
    node.schema().to_vec()
}
