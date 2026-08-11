//! 物理オペレータ（プル型 Volcano、ベクトル化）。
//!
//! プッシュ型のほうが速いが、プル型のほうがコード量が小さく、RowGroup 境界の
//! ステップ実行（DESIGN.md §6）に自然に載るのでプル型を採る。1 回の `next()`
//! で 2048 行を返すため、Volcano の呼び出しオーバーヘッドは相対的に無視できる。

pub mod agg;
pub mod join;
pub mod range;
pub mod recursive;
pub mod rowkey;
pub mod sample;
pub mod setop;
pub mod sort;
pub mod unnest;
pub mod window;

use crate::catalog::Catalog;
use crate::exec::rowkey::{encode_key, HashIndex};
use crate::expr::vm::Vm;
use crate::expr::Program;
use crate::plan::{Node, ScanSpec};
use crate::prelude::*;
use crate::vector::{Batch, Field, Vector, BATCH_SIZE};

/// ホストに要求するバイト範囲。
#[derive(Clone, Copy)]
pub struct IoRequest {
    /// カタログ上のテーブル添字。
    pub table: usize,
    /// テーブル内のパート添字（複数ファイルテーブルの何ファイル目か）。
    /// 単一ファイルのテーブルは常に 0。
    pub part: usize,
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
    /// `IoRequest::part` と同じ意味。
    pub part: usize,
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
    build_ctx(node, None)
}

/// `build` の本体。`working` は「再帰 CTE の再帰項を組み立てている最中か」
/// を表し、`Some` のときだけ `Node::WorkingTable` を解決できる
/// （`exec::recursive` 参照）。トップレベルの `build` は常に `None` で呼ぶ。
///
/// 再帰 CTE の入れ子（ある再帰 CTE の再帰項が、既に完成した別の再帰 CTE を
/// 参照する）でも `working` はそのまま素通しするだけで安全: `bind` は
/// `Node::WorkingTable` を「それを生んだ `RecursiveCte` 自身の再帰項」の
/// 中にしか置かないので、内側の木に迷い込んだ外側の作業テーブルが誤って
/// 使われることはない。
fn build_ctx(node: Node, working: Option<&[Batch]>) -> Result<Box<dyn Operator>> {
    Ok(match node {
        Node::Scan(spec) => Box::new(Scan::new(*spec)),
        #[cfg(feature = "ddl")]
        Node::MemScan(spec) => Box::new(MemScan::new(*spec)),
        Node::Filter { input, pred } => {
            Box::new(Filter { input: build_ctx(*input, working)?, pred })
        }
        Node::Project { input, exprs, .. } => {
            Box::new(Project { input: build_ctx(*input, working)?, exprs })
        }
        Node::Limit { input, limit, offset } => Box::new(Limit {
            input: build_ctx(*input, working)?,
            limit,
            offset,
            seen: 0,
            emitted: 0,
        }),
        Node::Aggregate { input, groups, aggs, having, .. } => {
            Box::new(agg::HashAggregate::new(build_ctx(*input, working)?, groups, aggs, having)?)
        }
        Node::Sort { input, keys, limit } => {
            Box::new(sort::Sort::new(build_ctx(*input, working)?, keys, limit)?)
        }
        Node::Join { left, right, kind, left_keys, right_keys, residual, .. } => {
            let lt: Vec<crate::vector::Ty> = left.schema().iter().map(|f| f.ty).collect();
            let rt: Vec<crate::vector::Ty> = right.schema().iter().map(|f| f.ty).collect();
            Box::new(join::HashJoin::new(
                build_ctx(*left, working)?,
                build_ctx(*right, working)?,
                kind,
                left_keys,
                right_keys,
                residual,
                lt,
                rt,
            )?)
        }
        Node::Window { input, windows, .. } => {
            Box::new(window::Window::new(build_ctx(*input, working)?, windows)?)
        }
        Node::SetOp { left, right, op, all, .. } => Box::new(setop::SetOp::new(
            build_ctx(*left, working)?,
            build_ctx(*right, working)?,
            op,
            all,
        )?),
        Node::DistinctOn { input, keys } => {
            Box::new(DistinctOn::new(build_ctx(*input, working)?, keys))
        }
        Node::RecursiveCte { anchor, recursive_term, union_all, .. } => Box::new(
            recursive::RecursiveCte::new(build_ctx(*anchor, working)?, *recursive_term, union_all),
        ),
        Node::WorkingTable { .. } => {
            // `bind` は `Node::WorkingTable` を自分を生んだ `RecursiveCte` の
            // 再帰項の中にしか置かないので、`working` が無いのはバインダの
            // バグ（あるいは `exec::build` をこのノードへ直接呼んだバグ）。
            let batches = match working {
                Some(w) => recursive::clone_batches(w),
                None => err!(Internal),
            };
            Box::new(recursive::WorkingTableScan::new(batches))
        }
        Node::Unnest { input, expr, elem_ty, .. } => {
            Box::new(unnest::Unnest::new(build_ctx(*input, working)?, expr, elem_ty))
        }
        Node::GenerateSeries { start, stop, step, inclusive, .. } => {
            Box::new(range::GenerateSeries::new(start, stop, step, inclusive))
        }
        Node::Sample { input, spec } => {
            let input = build_ctx(*input, working)?;
            if spec.is_rows {
                Box::new(sample::RowSample::new(input, &spec))
            } else {
                Box::new(sample::Bernoulli::new(input, &spec))
            }
        }
    })
}

// --- Scan -------------------------------------------------------------------

pub struct Scan {
    spec: ScanSpec,
    /// 現在読んでいるテーブルパートの添字（複数ファイルテーブルの何ファイル
    /// 目か）。単一ファイルのテーブルは常に 0 のまま終わる。
    part: usize,
    /// 現在のパート内で次に読む分割（Parquet なら RowGroup、CSV なら
    /// バイトチャンク）。パートを跨ぐと 0 に戻る。
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
        Scan { spec, part: 0, split: 0, cur: None }
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

            // 現在のパートを読み切っていたら次のパートへ進む。分割が 0 個の
            // パート（空ファイルなど）も飛ばせるよう while で進める。これで
            // 上位のオペレータ（Filter/Project/...）は複数ファイルテーブルを
            // 意識せずに済む — 分割番号を「テーブル全体で平坦な列」に見せる
            // のがこのループの役目。
            while self.part < table.parts.len()
                && self.split >= table.parts[self.part].format.num_splits()
            {
                self.part += 1;
                self.split = 0;
            }
            if self.part >= table.parts.len() {
                return Ok(Step::Done);
            }
            let part = &table.parts[self.part];
            let fmt = &part.format;

            // 統計で落とせるならバイトを 1 つも取らずに次へ進む。
            // 統計を持たないフォーマットは既定実装で常に通す。
            if !fmt.may_match(self.split, &self.spec.pruners, &self.spec.columns) {
                self.split += 1;
                continue;
            }

            // ページ単位の絞り込み（ColumnIndex/OffsetIndex/Bloom フィルタ）に
            // 使うバイトを集める。対象が無ければ何も積まれない（既定実装）ので、
            // 対応しないフォーマットはここで 1 往復も増えない。
            let mut idx_ranges = Vec::with_capacity(self.spec.columns.len());
            fmt.index_ranges(self.split, &self.spec.pruners, &self.spec.columns, &mut idx_ranges)?;
            let mut idx_missing = Vec::new();
            for (start, end) in &idx_ranges {
                if let Some((o, l)) = part.source.missing(*start, end - start) {
                    idx_missing.push(IoRequest {
                        table: self.spec.table,
                        part: self.part,
                        offset: o,
                        len: l,
                    });
                }
            }
            if !idx_missing.is_empty() {
                ctx.io.extend_from_slice(&idx_missing);
                return Ok(Step::NeedIo);
            }

            // 取得できたインデックスバイトでページ選択を確定する。`false` なら
            // この分割は丸ごと読み飛ばせる（Bloom フィルタでの否定、または
            // 統計上どのページも一致し得ない場合）。`&mut self` が要るので、
            // ここだけ `catalog` を可変で借り直す（`table`/`part`/`fmt` は
            // 以降で改めて取り直す）。
            let keep = match ctx.catalog.get_mut(self.spec.table) {
                Some(t) => match t.parts.get_mut(self.part) {
                    Some(p) => p.format.refine_with_index(
                        &p.source,
                        self.split,
                        &self.spec.pruners,
                        &self.spec.columns,
                    )?,
                    None => err!(Internal),
                },
                None => err!(TableNotFound),
            };
            if !keep {
                self.split += 1;
                continue;
            }

            let table = match ctx.catalog.get(self.spec.table) {
                Some(t) => t,
                None => err!(TableNotFound),
            };
            let part = &table.parts[self.part];
            let fmt = &part.format;

            // この分割に必要なバイト範囲を集め、未取得なら一括で要求する。
            // 分割境界でしか I/O を待たないので、オペレータは非同期を意識しない
            // （DESIGN.md §6）。ページ選択が効いていれば、ここで返る範囲は
            // 列チャンク全体ではなく生存ページだけに絞られる。
            let mut ranges = Vec::with_capacity(self.spec.columns.len());
            fmt.split_ranges(self.split, &self.spec.columns, &mut ranges)?;
            let mut missing = Vec::new();
            for (start, end) in &ranges {
                if let Some((o, l)) = part.source.missing(*start, end - start) {
                    missing.push(IoRequest {
                        table: self.spec.table,
                        part: self.part,
                        offset: o,
                        len: l,
                    });
                }
            }
            if !missing.is_empty() {
                ctx.io.extend_from_slice(&missing);
                return Ok(Step::NeedIo);
            }

            // 内蔵していないコーデックはホストに展開してもらう。ページヘッダは
            // 非圧縮なので、この時点で必要な作業をすべて洗い出せる。
            let mut tasks = Vec::new();
            fmt.codec_tasks(&part.source, self.split, &self.spec.columns, &mut tasks)?;
            let mut pending = Vec::new();
            for t in &tasks {
                if !part.source.has_decoded(t.offset, t.len) {
                    pending.push(CodecRequest {
                        table: self.spec.table,
                        part: self.part,
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

            let cols = fmt.read_split(&part.source, self.split, &self.spec.columns)?;
            // フォーマット実装の契約: 射影と同じ個数・同じ長さの列を返す。
            ensure!(cols.len() == self.spec.columns.len(), Internal);
            let rows = cols.first().map_or(0, |c| c.len());
            ensure!(cols.iter().all(|c| c.len() == rows), Internal);

            // 展開済みページはこの分割でしか使わない。抱えたままだと
            // 圧縮前のファイルより大きなメモリを持つことになる。
            if let Some(t) = ctx.catalog.get_mut(self.spec.table) {
                if let Some(p) = t.parts.get_mut(self.part) {
                    p.source.clear_decoded();
                }
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

// --- MemScan (`ddl`) ---------------------------------------------------------

/// `catalog::MemTable` の走査。データは既にメモリ上にあるので、`Scan` と
/// 違って `NeedIo`/`NeedCodec` は原理的に返らない
/// （DESIGN.md §16「なぜ 4 つに割るのか」）。
#[cfg(feature = "ddl")]
pub struct MemScan {
    spec: crate::plan::MemScanSpec,
    pos: usize,
}

#[cfg(feature = "ddl")]
impl MemScan {
    pub fn new(spec: crate::plan::MemScanSpec) -> Self {
        MemScan { spec, pos: 0 }
    }
}

#[cfg(feature = "ddl")]
impl Operator for MemScan {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        let mt = match ctx.catalog.mem_get(self.spec.table) {
            Some(t) => t,
            None => err!(TableNotFound),
        };
        if self.pos >= mt.rows.len() {
            return Ok(Step::Done);
        }
        let end = (self.pos + BATCH_SIZE).min(mt.rows.len());
        let batch = mt.batch(self.pos, end);
        self.pos = end;
        Ok(Step::Ready(batch))
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

// --- DistinctOn ---------------------------------------------------------------

/// `DISTINCT ON (keys)`。
///
/// 入力の並び順で、キーごとに**最初に見た行だけ**を通すストリーミングフィルタ。
/// 「どの行が先か」を決めるのはこのオペレータの仕事ではなく、呼び出し側
/// （バインダ）が必要なら先に `Sort` を挟んで並びを確定させておく。
/// ブロッキングではないので `NeedIo`/`NeedCodec` はそのまま素通しするだけで
/// 再開できる（`Filter` と同じ単純さ）。状態（`seen`）はバッチをまたいで
/// 保持するが、1 バッチぶんの処理は途中で抜けずに丸ごと終える。
const MAX_DISTINCT_ON_BYTES: usize = 64 << 20;
const DISTINCT_ON_OVERHEAD: usize = 32;

pub struct DistinctOn {
    input: Box<dyn Operator>,
    keys: Vec<Program>,
    seen: HashIndex,
}

impl DistinctOn {
    pub fn new(input: Box<dyn Operator>, keys: Vec<Program>) -> Self {
        DistinctOn { input, keys, seen: HashIndex::new() }
    }
}

impl Operator for DistinctOn {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        loop {
            let batch = match self.input.next(ctx)? {
                Step::Ready(b) => b,
                other => return Ok(other),
            };
            let rows = batch.card();
            let mut kcols = Vec::with_capacity(self.keys.len());
            for p in &self.keys {
                kcols.push(ctx.vm.eval(p, &batch)?);
            }
            let refs: Vec<&Vector> = kcols.iter().collect();
            let mut sel = Vec::with_capacity(rows);
            let mut key = Vec::new();
            for row in 0..rows {
                encode_key(&refs, row, &mut key);
                let (_, is_new) = self.seen.get_or_insert(&key);
                if is_new {
                    let phys = match &batch.sel {
                        Some(s) => s[row],
                        None => row as u32,
                    };
                    sel.push(phys);
                }
            }
            let used = self.seen.key_bytes() + self.seen.len() * DISTINCT_ON_OVERHEAD;
            ensure!(used <= MAX_DISTINCT_ON_BYTES, Oom);
            if sel.is_empty() {
                continue;
            }
            let mut out = batch;
            out.sel = Some(sel);
            return Ok(Step::Ready(out));
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

// --- Values -----------------------------------------------------------------

/// あらかじめ作ったバッチを 1 回だけ返すオペレータ。
///
/// `DESCRIBE` / `SHOW TABLES` / `EXPLAIN` のように、スキャンを伴わずに
/// 結果が確定しているものに使う。専用の実行経路を作らずに済む。
pub struct Values {
    batch: Option<Batch>,
}

impl Values {
    pub fn new(batch: Batch) -> Self {
        Values { batch: Some(batch) }
    }
}

impl Operator for Values {
    fn next(&mut self, _ctx: &mut ExecContext) -> Result<Step> {
        match self.batch.take() {
            Some(b) => Ok(Step::Ready(b)),
            None => Ok(Step::Done),
        }
    }
}

#[cfg(test)]
mod distinct_on_tests {
    use super::*;
    use crate::catalog::Catalog;
    use crate::expr::vm::Vm;
    use crate::expr::{Instr, OpCode, Program};
    use crate::vector::{Ty, Value};

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

    /// `idx` 列をそのまま返すプログラム。
    fn load(idx: u16) -> Program {
        let mut p = Program::new();
        let r = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, crate::vector::PhysType::I32, r, 0, 0, idx));
        p.result = r;
        p.result_ty = Ty::Int;
        p
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

    fn drive(steps: Vec<Script>, keys: Vec<Program>) -> Vec<Vec<Value>> {
        let mut op = DistinctOn::new(Box::new(Mock { steps, pos: 0 }), keys);
        let mut cat = Catalog::new();
        let mut vm = Vm::new();
        let mut rows = Vec::new();
        for guard in 0..10_000 {
            assert!(guard < 9_999, "終わらない");
            let mut ctx =
                ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
            match op.next(&mut ctx).unwrap() {
                Step::Ready(b) => {
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
        rows
    }

    #[test]
    fn keeps_only_the_first_row_per_key_in_arrival_order() {
        // key=1 の行が 3 つ、key=2 の行が 2 つ来る。各キーの最初の行だけ残る。
        let steps = vec![Script::Rows(vec![
            ints(&[Some(1), Some(2), Some(1), Some(2), Some(1)]),
            ints(&[Some(10), Some(20), Some(11), Some(21), Some(12)]),
        ])];
        let rows = drive(steps, vec![load(0)]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec![Value::I32(1), Value::I32(10)]);
        assert_eq!(rows[1], vec![Value::I32(2), Value::I32(20)]);
    }

    #[test]
    fn null_keys_form_their_own_group() {
        let steps = vec![Script::Rows(vec![
            ints(&[None, Some(1), None]),
            ints(&[Some(9), Some(8), Some(7)]),
        ])];
        let rows = drive(steps, vec![load(0)]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec![Value::Null, Value::I32(9)]);
        assert_eq!(rows[1], vec![Value::I32(1), Value::I32(8)]);
    }

    #[test]
    fn multi_column_keys_are_compared_as_a_whole() {
        let steps = vec![Script::Rows(vec![
            ints(&[Some(1), Some(1), Some(1)]),
            ints(&[Some(1), Some(2), Some(1)]),
            ints(&[Some(100), Some(200), Some(300)]),
        ])];
        let rows = drive(steps, vec![load(0), load(1)]);
        // (1,1) と (1,2) は別キー、2 つ目の (1,1) は重複なので落ちる。
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][2], Value::I32(100));
        assert_eq!(rows[1][2], Value::I32(200));
    }

    #[test]
    fn need_io_mid_input_does_not_change_the_result() {
        let make = |interrupt: bool| {
            let mut steps = Vec::new();
            steps.push(Script::Rows(vec![ints(&[Some(1), Some(2)]), ints(&[Some(10), Some(20)])]));
            if interrupt {
                steps.push(Script::NeedIo);
            }
            steps.push(Script::Rows(vec![ints(&[Some(1), Some(3)]), ints(&[Some(11), Some(30)])]));
            steps
        };
        let plain = drive(make(false), vec![load(0)]);
        let interrupted = drive(make(true), vec![load(0)]);
        assert_eq!(plain, interrupted, "NeedIo をまたいでも結果が変わってはいけない");
        assert_eq!(plain.len(), 3);
    }

    #[test]
    fn all_duplicates_in_a_batch_yield_no_output_batch() {
        // 1 バッチが丸ごと重複だけなら、そのバッチは呼び出し元に返らず
        // 次の入力を引きにいく（`Filter` と同じ「空なら次へ」規律）。
        let steps = vec![
            Script::Rows(vec![ints(&[Some(1)]), ints(&[Some(10)])]),
            Script::Rows(vec![ints(&[Some(1), Some(1)]), ints(&[Some(99), Some(98)])]),
            Script::Rows(vec![ints(&[Some(2)]), ints(&[Some(20)])]),
        ];
        let rows = drive(steps, vec![load(0)]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec![Value::I32(1), Value::I32(10)]);
        assert_eq!(rows[1], vec![Value::I32(2), Value::I32(20)]);
    }
}
