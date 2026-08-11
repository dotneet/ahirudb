//! ソートと Top-N。
//!
//! ソートは入力を全部読み切るまで 1 行も返せない**ブロッキング**オペレータ。
//! リモート（レンジ取得）ソースでは入力の途中で `Step::NeedIo` / `NeedCodec`
//! が返るので、蓄積の途中状態はすべて `self` に持ち、中断はそのまま素通しして
//! 次の呼び出しで同じ場所から再開する（DESIGN.md §6）。状態を持たずに
//! 「全部読む」ループを書くと、そこで入力を捨てるか二重に読むかになる。
//!
//! 行は列指向のまま溜める。行ごとに `Value` を作ると 1 セル 1 確保になり、
//! 比較よりも確保のほうが支配的になるため。
//!
//! ## 順序の決め方
//!
//! - キーは指定順に比較し、`desc` は**値の比較結果だけ**を反転する。
//! - NULL の位置は `nulls_first` だけで決まり、`desc` の影響を受けない。
//!   SQL 既定の ASC→NULLS LAST / DESC→NULLS FIRST はバインダがフラグに
//!   落とし済みなので、ここで再適用すると二重に反転する。
//! - F64 は `partial_cmp` が `None` を返しうる。比較器が `None` を「等しい」に
//!   潰すと推移律が壊れるので、そもそも使わずに全順序を持つ `u64` キーへ写す
//!   （`f64_key`）。順序は `-inf < … < -0.0 = 0.0 < … < +inf < NaN`。
//! - 最後にバッファ上の行番号で決着させるので比較は**全順序**になる。
//!   これで安定性（同値キーは入力順のまま）も同時に得られる。
//!
//! ## メモリ
//!
//! 溢れ処理（spill）は持たない。バッファが `MAX_BUFFER_BYTES` を超えたら
//! 静かに巨大化させず `Oom` を返す。`limit` があるときはそもそも上位 n 件しか
//! 抱えないので、50M 行に対する `ORDER BY … LIMIT 10` でも上限に触れない。

use crate::exec::{ExecContext, Operator, Step};
use crate::plan::SortKey;
use crate::prelude::*;
use crate::vector::{Batch, Bitmap, Data, Vector, BATCH_SIZE};

use core::cmp::Ordering;

/// 溢れ処理を持たないので、これを超えたら `Oom` を返す。
/// wasm の線形メモリは 4GiB が上限だが、ホスト側のバッファや復号済みページと
/// 同居するため、ソート単体では 256MiB までに抑える。
const MAX_BUFFER_BYTES: usize = 256 * 1024 * 1024;

enum Phase {
    /// 入力を読んで溜めている。中断を跨いでもこの状態のまま。
    Buffering,
    /// 順序が確定した。`order` を `BATCH_SIZE` ずつ切って返す。
    Emitting,
    Done,
}

pub struct Sort {
    input: Box<dyn Operator>,
    keys: Vec<SortKey>,
    /// Top-N で残す行数。`None` は全件。
    limit: Option<usize>,
    phase: Phase,

    /// 入力列の蓄積。スキーマは入力そのまま（ソートは列を変えない）。
    cols: Vec<Vector>,
    /// ソートキーの蓄積。`keys` と同じ個数・同じ並び。
    key_cols: Vec<Vector>,
    /// 蓄積した行数。列を持たない入力（`COUNT(*)` など）でも行数だけは要る。
    rows: usize,
    /// 最初のバッチで列型を決めたか。0 列の入力があるので `cols` の空判定では代用できない。
    init: bool,

    /// 確定した出力順。`Emitting` 以降のみ有効。
    order: Vec<u32>,
    /// 次に返す `order` の位置。
    pos: usize,
}

impl Sort {
    pub fn new(input: Box<dyn Operator>, keys: Vec<SortKey>, limit: Option<usize>) -> Result<Self> {
        // LIMIT 0 は 1 行も返さない。入力を引く必要すらない。
        let phase = if limit == Some(0) { Phase::Done } else { Phase::Buffering };
        Ok(Sort {
            input,
            keys,
            limit,
            phase,
            cols: Vec::new(),
            key_cols: Vec::new(),
            rows: 0,
            init: false,
            order: Vec::new(),
            pos: 0,
        })
    }

    /// 1 バッチを蓄積へ取り込む。
    fn absorb(&mut self, mut batch: Batch, ctx: &mut ExecContext) -> Result<()> {
        // selection を先に解消しておく。キー評価と列の追記で 2 回 gather する
        // のを避けるため。
        batch.materialize();
        let rows = batch.card();

        let mut kvs = Vec::with_capacity(self.keys.len());
        for k in &self.keys {
            kvs.push(ctx.vm.eval(&k.expr, &batch)?);
        }

        if !self.init {
            self.cols = batch.cols.iter().map(|c| Vector::new(c.ty())).collect();
            self.key_cols = kvs.iter().map(|v| Vector::new(v.ty())).collect();
            self.init = true;
        }
        ensure!(batch.cols.len() == self.cols.len(), Internal);

        // 行番号を u32 に載せるので、そこを超えたら諦める。
        ensure!(self.rows.saturating_add(rows) <= u32::MAX as usize, LimitExceeded);

        for (dst, src) in self.key_cols.iter_mut().zip(kvs.iter()) {
            append(dst, src)?;
        }
        for (dst, src) in self.cols.iter_mut().zip(batch.cols.iter()) {
            append(dst, src)?;
        }
        self.rows += rows;

        // 上限判定より先に圧縮する。Top-N は上限に触れずに済む。
        self.compact()?;
        ensure!(self.buffered_bytes() <= MAX_BUFFER_BYTES, Oom);
        Ok(())
    }

    /// Top-N のときだけ、バッファを上位 `n` 件へ切り詰める。
    ///
    /// 1 行ごとにヒープを触る代わりに `2n` 行まで溜めてから一括で選別する。
    /// 列指向のバッファでは 1 行の差し替えが（可変長列のせいで）安くないので、
    /// まとめて `gather` し直すほうがアロケーションも比較回数も少ない。
    /// 1 回の圧縮で `n` 行以上を捨てるため、償却では 1 行あたり O(log n)。
    fn compact(&mut self) -> Result<()> {
        let n = match self.limit {
            Some(n) => n,
            None => return Ok(()),
        };
        let cap = n.saturating_mul(2).max(BATCH_SIZE);
        if self.rows <= cap {
            return Ok(());
        }
        let mut order: Vec<u32> = (0..self.rows as u32).collect();
        order.sort_by(|&a, &b| cmp_row(&self.keys, &self.key_cols, a, b));
        order.truncate(n);
        // 残す行は**ソート順のまま**書き戻す。こうすると「バッファ添字の昇順は
        // 同値キーにおける入力順と一致する」という不変条件が保たれ、比較器の
        // 添字による決着だけで安定性が出る。
        for c in self.cols.iter_mut() {
            *c = c.gather(&order);
        }
        for c in self.key_cols.iter_mut() {
            *c = c.gather(&order);
        }
        self.rows = order.len();
        Ok(())
    }

    /// 蓄積が使っているおおよそのバイト数。
    fn buffered_bytes(&self) -> usize {
        let mut n = 0usize;
        for v in self.cols.iter().chain(self.key_cols.iter()) {
            n = n.saturating_add(vector_bytes(v));
        }
        n
    }

    /// 入力を読み切った。順序を確定して出力フェーズへ移る。
    fn finish(&mut self) -> Result<()> {
        let mut order: Vec<u32> = (0..self.rows as u32).collect();
        order.sort_by(|&a, &b| cmp_row(&self.keys, &self.key_cols, a, b));
        if let Some(n) = self.limit {
            order.truncate(n);
        }
        self.order = order;
        self.pos = 0;
        // キー列はもう使わない。出力中に抱えている理由がない。
        self.key_cols = Vec::new();
        self.phase = Phase::Emitting;
        Ok(())
    }

    fn emit(&mut self) -> Result<Step> {
        if self.pos >= self.order.len() {
            self.phase = Phase::Done;
            self.cols = Vec::new();
            self.order = Vec::new();
            return Ok(Step::Done);
        }
        let end = (self.pos + BATCH_SIZE).min(self.order.len());
        let idx = &self.order[self.pos..end];
        let out = if self.cols.is_empty() {
            Batch::rows_only(idx.len())
        } else {
            Batch::new(self.cols.iter().map(|c| c.gather(idx)).collect())
        };
        self.pos = end;
        Ok(Step::Ready(out))
    }
}

impl Operator for Sort {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        loop {
            match self.phase {
                Phase::Buffering => match self.input.next(ctx)? {
                    Step::Ready(b) => self.absorb(b, ctx)?,
                    // 中断はそのまま上へ返す。蓄積した行は `self` に残るので、
                    // 次回の呼び出しはここから入力を引き直す（取りこぼしも
                    // 二重取りも起きない）。バイト待ちも展開待ちも扱いは同じ。
                    Step::NeedIo => return Ok(Step::NeedIo),
                    Step::NeedCodec => return Ok(Step::NeedCodec),
                    Step::Done => self.finish()?,
                },
                Phase::Emitting => return self.emit(),
                Phase::Done => return Ok(Step::Done),
            }
        }
    }
}

// --- 比較 -------------------------------------------------------------------

/// 2 行の全順序比較。同値キーは添字で決着させるので、`Equal` は同一行のときだけ。
fn cmp_row(keys: &[SortKey], cols: &[Vector], a: u32, b: u32) -> Ordering {
    let (ai, bi) = (a as usize, b as usize);
    for (k, c) in keys.iter().zip(cols.iter()) {
        let (va, vb) = (c.is_valid(ai), c.is_valid(bi));
        if !va || !vb {
            if !va && !vb {
                continue;
            }
            // NULL の位置は nulls_first だけで決まる。ここで desc を掛けると
            // バインダが入れた既定（ASC→LAST / DESC→FIRST）を二重に適用する。
            let null_is_first = k.nulls_first;
            return if !va {
                if null_is_first {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            } else if null_is_first {
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
    a.cmp(&b)
}

/// 物理型ごとの値比較。NULL の判定は呼び出し側で済ませてある。
fn cmp_data(d: &Data, a: usize, b: usize) -> Ordering {
    match d {
        // false < true。
        Data::Bool(v) => v.get(a).cmp(&v.get(b)),
        Data::I32(v) => v[a].cmp(&v[b]),
        Data::I64(v) => v[a].cmp(&v[b]),
        Data::I128(v) => v[a].cmp(&v[b]),
        Data::F64(v) => f64_key(v[a]).cmp(&f64_key(v[b])),
        // 辞書順。前方一致する場合は短いほうが小さい。
        Data::Bytes(v) => v.get(a).cmp(v.get(b)),
    }
}

/// f64 を順序を保つ `u64` へ写す。
///
/// `partial_cmp` は NaN で `None` を返すため比較器には使えない。ビット表現を
/// 単調写像に通して全順序にする:
/// `-inf < … < -0.0 = 0.0 < … < +inf < NaN`。
///
/// - NaN は「全数値より大きい」1 つの値に潰す。NaN 同士は同値なので、
///   複数の NaN があっても入力順のまま並ぶ（決定的）。
/// - `-0.0` と `0.0` は `=` で等しく、`rowkey::canonical_f64` も同一視する。
///   順序だけ別扱いにすると挙動がちぐはぐになるので同値に揃える。
#[inline]
fn f64_key(v: f64) -> u64 {
    if v.is_nan() {
        return u64::MAX;
    }
    let b = if v == 0.0 { 0 } else { v.to_bits() };
    // 負数はビット列が大きいほど値が小さいので反転する。正数は符号ビットを
    // 立てて負数より上へ持ち上げる。
    if b >> 63 != 0 {
        !b
    } else {
        b | (1 << 63)
    }
}

// --- バッファ操作 -----------------------------------------------------------

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
            // 先頭 offset は既に dst 側に入っているので飛ばす。
            for &o in s.offsets.iter().skip(1) {
                d.offsets.push(shift + (o - first));
            }
        }
        // 同じオペレータから来る列なので、ここに落ちるのは組み立て側のバグ。
        _ => err!(Internal),
    }
    // どちらかに NULL があれば validity を揃える。`dst` 側だけが持っている
    // 場合も伸ばさないと長さが本体とずれ、以降の `is_valid` が範囲外を読む。
    if n > 0 && (src.has_nulls() || dst.has_nulls()) {
        // validity_mut は追記済みの長さまで全 1 で実体化・伸長してくれる。
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
/// `exec::recursive` も再帰 CTE の作業テーブルのバイト数上限判定に使う
/// （メモリ見積もりのロジックを 2 か所に増やさないため）。
pub(crate) fn vector_bytes(v: &Vector) -> usize {
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
    use crate::expr::vm::Vm;
    use crate::expr::{Instr, OpCode, Program};
    use crate::vector::{Ty, Value};

    // --- 組み立てヘルパ -----------------------------------------------------

    /// `col` 番目の列をそのまま返すプログラム。
    fn col_expr(col: u16, ty: Ty) -> Program {
        let mut p = Program::new();
        let r = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, ty.phys(), r, 0, 0, col));
        p.result = r;
        p.result_ty = ty;
        p
    }

    fn key(col: u16, ty: Ty, desc: bool, nulls_first: bool) -> SortKey {
        SortKey { expr: col_expr(col, ty), desc, nulls_first }
    }

    fn vector(ty: Ty, vals: &[Option<Value>]) -> Vector {
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
        vector(Ty::Int, &vals.iter().map(|v| v.map(Value::I32)).collect::<Vec<_>>())
    }

    /// 0..n の連番列。安定性と Top-N の検証用の「行 ID」。
    fn ids(n: usize) -> Vector {
        ints(&(0..n as i32).map(Some).collect::<Vec<_>>())
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
        // テスト用のヘルパなので `Box<dyn Operator>` を返す方が呼び出し側が短い。
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
            // 実際の Scan も再呼び出しでは同じ分割の続きから返す。
            Ok(match &self.steps[i] {
                Script::NeedIo => Step::NeedIo,
                Script::NeedCodec => Step::NeedCodec,
                Script::Rows(cols) => Step::Ready(Batch::new(cols.clone())),
            })
        }
    }

    /// 1 バッチぶんの出力を行ごとの `Value` 列に落とす。
    fn rows_of(b: &Batch) -> Vec<Vec<Value>> {
        (0..b.card()).map(|i| b.cols.iter().map(|c| c.value_at(i)).collect()).collect()
    }

    /// ソートを最後まで回し、バッチごとの行を返す。
    fn drive(steps: Vec<Script>, keys: Vec<SortKey>, limit: Option<usize>) -> Vec<Vec<Vec<Value>>> {
        let mut op = Sort::new(Mock::new(steps), keys, limit).unwrap();
        let mut cat = Catalog::default();
        let mut vm = Vm::new();
        let mut ctx =
            ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
        let mut out = Vec::new();
        for _ in 0..100_000 {
            match op.next(&mut ctx).unwrap() {
                Step::Ready(b) => out.push(rows_of(&b)),
                // ホストが応答したことにして同じオペレータを呼び直す。
                Step::NeedIo | Step::NeedCodec => continue,
                Step::Done => return out,
            }
        }
        panic!("sort did not terminate");
    }

    /// バッチ境界を潰した全行。
    fn flat(steps: Vec<Script>, keys: Vec<SortKey>, limit: Option<usize>) -> Vec<Vec<Value>> {
        drive(steps, keys, limit).into_iter().flatten().collect()
    }

    /// 各行の指定列を i32 として取り出す。
    fn col_i32(rows: &[Vec<Value>], c: usize) -> Vec<Option<i32>> {
        rows.iter()
            .map(|r| match &r[c] {
                Value::I32(v) => Some(*v),
                _ => None,
            })
            .collect()
    }

    // --- 中断と再開（最重要） -----------------------------------------------

    #[test]
    fn need_io_mid_input_matches_uninterrupted_run() {
        let mk = |interrupted: bool| {
            let a = Script::Rows(vec![ints(&[Some(5), Some(1)]), ints(&[Some(0), Some(1)])]);
            let b = Script::Rows(vec![ints(&[Some(3), Some(9)]), ints(&[Some(2), Some(3)])]);
            let c = Script::Rows(vec![ints(&[Some(2)]), ints(&[Some(4)])]);
            if interrupted {
                // 入力の途中（バッチとバッチの間、かつ先頭でも末尾でもない）で
                // 中断を挟む。バイト待ちと展開待ちの両方を混ぜる。
                vec![a, Script::NeedIo, b, Script::NeedCodec, c, Script::NeedIo]
            } else {
                vec![a, b, c]
            }
        };
        let ks = || vec![key(0, Ty::Int, false, false)];
        let plain = flat(mk(false), ks(), None);
        let broken = flat(mk(true), ks(), None);
        assert_eq!(col_i32(&plain, 0), vec![Some(1), Some(2), Some(3), Some(5), Some(9)]);
        // 行が消えたり二重に入ったりしていないこと。
        assert_eq!(col_i32(&broken, 0), col_i32(&plain, 0));
        assert_eq!(col_i32(&broken, 1), col_i32(&plain, 1));
    }

    #[test]
    fn need_io_before_any_input_is_passed_through() {
        // 最初の呼び出しがいきなり中断でも壊れない。
        let steps =
            vec![Script::NeedIo, Script::NeedCodec, Script::Rows(vec![ints(&[Some(2), Some(1)])])];
        let mut op =
            Sort::new(Mock::new(steps), vec![key(0, Ty::Int, false, false)], None).unwrap();
        let mut cat = Catalog::default();
        let mut vm = Vm::new();
        let mut ctx =
            ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
        assert!(matches!(op.next(&mut ctx).unwrap(), Step::NeedIo));
        assert!(matches!(op.next(&mut ctx).unwrap(), Step::NeedCodec));
        let b = match op.next(&mut ctx).unwrap() {
            Step::Ready(b) => b,
            _ => panic!("expected rows"),
        };
        assert_eq!(col_i32(&rows_of(&b), 0), vec![Some(1), Some(2)]);
    }

    #[test]
    fn need_io_during_top_n_compaction() {
        // 圧縮を跨いで中断しても上位 n 件は変わらない。
        let chunk = |base: i32| {
            let vals: Vec<Option<i32>> = (0..1500).map(|i| Some((base + i * 37) % 5000)).collect();
            Script::Rows(vec![ints(&vals)])
        };
        let mk = |interrupted: bool| {
            let mut v = vec![chunk(0), chunk(1)];
            if interrupted {
                v.push(Script::NeedIo);
            }
            v.push(chunk(2));
            v
        };
        let ks = || vec![key(0, Ty::Int, false, false)];
        let plain = flat(mk(false), ks(), Some(5));
        let broken = flat(mk(true), ks(), Some(5));
        assert_eq!(col_i32(&broken, 0), col_i32(&plain, 0));
        assert_eq!(plain.len(), 5);
    }

    // --- 基本の順序 ---------------------------------------------------------

    #[test]
    fn single_key_asc_and_desc() {
        let rows = || vec![Script::Rows(vec![ints(&[Some(3), Some(1), Some(2)])])];
        let asc = flat(rows(), vec![key(0, Ty::Int, false, false)], None);
        assert_eq!(col_i32(&asc, 0), vec![Some(1), Some(2), Some(3)]);
        let desc = flat(rows(), vec![key(0, Ty::Int, true, false)], None);
        assert_eq!(col_i32(&desc, 0), vec![Some(3), Some(2), Some(1)]);
    }

    #[test]
    fn second_key_breaks_ties_with_its_own_direction() {
        // 第 1 キー ASC、第 2 キー DESC。
        let cols = vec![
            ints(&[Some(1), Some(1), Some(0), Some(0)]),
            ints(&[Some(10), Some(20), Some(30), Some(40)]),
        ];
        let rows = flat(
            vec![Script::Rows(cols)],
            vec![key(0, Ty::Int, false, false), key(1, Ty::Int, true, false)],
            None,
        );
        assert_eq!(col_i32(&rows, 0), vec![Some(0), Some(0), Some(1), Some(1)]);
        assert_eq!(col_i32(&rows, 1), vec![Some(40), Some(30), Some(20), Some(10)]);
    }

    #[test]
    fn equal_keys_keep_input_order() {
        // キーは全部同じ。ID 列が入力順のまま出てくること。
        let n = 200;
        let cols = vec![ints(&vec![Some(7); n]), ids(n)];
        let rows = flat(vec![Script::Rows(cols)], vec![key(0, Ty::Int, false, false)], None);
        assert_eq!(col_i32(&rows, 1), (0..n as i32).map(Some).collect::<Vec<_>>());

        // バッチを跨いでも同じ。
        let steps = vec![
            Script::Rows(vec![ints(&[Some(7), Some(7)]), ints(&[Some(0), Some(1)])]),
            Script::Rows(vec![ints(&[Some(7), Some(7)]), ints(&[Some(2), Some(3)])]),
        ];
        let rows = flat(steps, vec![key(0, Ty::Int, true, false)], None);
        assert_eq!(col_i32(&rows, 1), vec![Some(0), Some(1), Some(2), Some(3)]);
    }

    // --- NULL の位置 --------------------------------------------------------

    #[test]
    fn null_placement_follows_flag_not_direction() {
        // 値は 2, NULL, 1（ID は 0, 1, 2）。
        let cols = || vec![ints(&[Some(2), None, Some(1)]), ids(3)];
        let run = |desc: bool, nulls_first: bool| {
            let rows =
                flat(vec![Script::Rows(cols())], vec![key(0, Ty::Int, desc, nulls_first)], None);
            col_i32(&rows, 1)
        };
        // ASC: 1, 2 の順。NULL は旗の側へ。
        assert_eq!(run(false, false), vec![Some(2), Some(0), Some(1)]);
        assert_eq!(run(false, true), vec![Some(1), Some(2), Some(0)]);
        // DESC: 2, 1 の順。NULL の位置は ASC と同じ旗に従う（desc で反転しない）。
        assert_eq!(run(true, false), vec![Some(0), Some(2), Some(1)]);
        assert_eq!(run(true, true), vec![Some(1), Some(0), Some(2)]);
    }

    #[test]
    fn nulls_are_equal_to_each_other_and_fall_through_to_next_key() {
        // 第 1 キーが両方 NULL なら第 2 キーで決まる。
        let cols = vec![ints(&[None, None]), ints(&[Some(9), Some(4)])];
        let rows = flat(
            vec![Script::Rows(cols)],
            vec![key(0, Ty::Int, false, true), key(1, Ty::Int, false, false)],
            None,
        );
        assert_eq!(col_i32(&rows, 1), vec![Some(4), Some(9)]);
    }

    // --- 物理型ごとの比較 ---------------------------------------------------

    fn sorted_ids(col: Vector, n: usize, desc: bool) -> Vec<Option<i32>> {
        let ty = col.ty();
        let rows = flat(vec![Script::Rows(vec![col, ids(n)])], vec![key(0, ty, desc, false)], None);
        col_i32(&rows, 1)
    }

    #[test]
    fn sorts_bool() {
        let c = vector(Ty::Boolean, &[Some(Value::Bool(true)), Some(Value::Bool(false))]);
        // false < true。
        assert_eq!(sorted_ids(c, 2, false), vec![Some(1), Some(0)]);
    }

    #[test]
    fn sorts_i32_i64_i128() {
        let c = ints(&[Some(0), Some(i32::MIN), Some(i32::MAX)]);
        assert_eq!(sorted_ids(c, 3, false), vec![Some(1), Some(0), Some(2)]);

        let c = vector(
            Ty::BigInt,
            &[Some(Value::I64(0)), Some(Value::I64(i64::MIN)), Some(Value::I64(i64::MAX))],
        );
        assert_eq!(sorted_ids(c, 3, false), vec![Some(1), Some(0), Some(2)]);

        let c = vector(
            Ty::HugeInt,
            &[Some(Value::I128(0)), Some(Value::I128(i128::MIN)), Some(Value::I128(i128::MAX))],
        );
        assert_eq!(sorted_ids(c, 3, true), vec![Some(2), Some(0), Some(1)]);
    }

    #[test]
    fn sorts_bytes_lexicographically() {
        let b = |s: &str| Some(Value::Bytes(s.as_bytes().to_vec()));
        // "" < "ab" < "abc" < "b"（前方一致は短いほうが小さい）
        let c = vector(Ty::Varchar, &[b("b"), b("abc"), b(""), b("ab")]);
        assert_eq!(sorted_ids(c, 4, false), vec![Some(2), Some(3), Some(1), Some(0)]);

        // 0x80 以上のバイトも符号なしとして扱う。
        let raw = |v: &[u8]| Some(Value::Bytes(v.to_vec()));
        let c = vector(Ty::Blob, &[raw(&[0xff]), raw(&[0x01]), raw(&[0x80])]);
        assert_eq!(sorted_ids(c, 3, false), vec![Some(1), Some(2), Some(0)]);
    }

    #[test]
    fn f64_total_order_is_documented_and_deterministic() {
        let f = |v: f64| Some(Value::F64(v));
        // 入力順: NaN, 0.0, -0.0, +inf, -inf, 1.0, NaN(負号)
        let c = vector(
            Ty::Double,
            &[
                f(f64::NAN),
                f(0.0),
                f(-0.0),
                f(f64::INFINITY),
                f(f64::NEG_INFINITY),
                f(1.0),
                f(-f64::NAN),
            ],
        );
        // -inf < -0.0 = 0.0 < 1.0 < +inf < NaN。
        // 0.0 と -0.0 は同値なので入力順（ID 1, 2）のまま。NaN 同士も同様。
        assert_eq!(
            sorted_ids(c.clone(), 7, false),
            vec![Some(4), Some(1), Some(2), Some(5), Some(3), Some(0), Some(6)]
        );
        // DESC は完全に逆順（同値の組は入力順のままなので入れ替わらない）。
        assert_eq!(
            sorted_ids(c, 7, true),
            vec![Some(0), Some(6), Some(3), Some(5), Some(1), Some(2), Some(4)]
        );
    }

    #[test]
    fn f64_nan_and_nulls_are_independent() {
        let f = |v: f64| Some(Value::F64(v));
        let c = vector(Ty::Double, &[f(f64::NAN), None, f(1.0)]);
        let ty = c.ty();
        // NaN は「最大の数値」、NULL は旗の側。両者は混ざらない。
        let rows =
            flat(vec![Script::Rows(vec![c.clone(), ids(3)])], vec![key(0, ty, false, true)], None);
        assert_eq!(col_i32(&rows, 1), vec![Some(1), Some(2), Some(0)]);
        let rows = flat(vec![Script::Rows(vec![c, ids(3)])], vec![key(0, ty, false, false)], None);
        assert_eq!(col_i32(&rows, 1), vec![Some(2), Some(0), Some(1)]);
    }

    // --- Top-N --------------------------------------------------------------

    #[test]
    fn limit_zero_emits_nothing() {
        let steps = vec![Script::Rows(vec![ints(&[Some(1), Some(2)])])];
        assert!(drive(steps, vec![key(0, Ty::Int, false, false)], Some(0)).is_empty());
    }

    #[test]
    fn limit_smaller_equal_and_larger_than_input() {
        let rows = || vec![Script::Rows(vec![ints(&[Some(3), Some(1), Some(2)])])];
        let ks = || vec![key(0, Ty::Int, false, false)];
        assert_eq!(col_i32(&flat(rows(), ks(), Some(2)), 0), vec![Some(1), Some(2)]);
        assert_eq!(col_i32(&flat(rows(), ks(), Some(3)), 0), vec![Some(1), Some(2), Some(3)]);
        assert_eq!(col_i32(&flat(rows(), ks(), Some(9)), 0), vec![Some(1), Some(2), Some(3)]);
    }

    #[test]
    fn top_n_over_many_rows_comes_out_in_order() {
        // 5000 行を 2048 行ずつ流し、圧縮を何度も起こす。
        // キーは 0..4999 の並べ替え（gcd(37, 5000) = 1）。
        const N: usize = 5000;
        let mut steps = Vec::new();
        let mut i = 0usize;
        while i < N {
            let end = (i + BATCH_SIZE).min(N);
            let k: Vec<Option<i32>> = (i..end).map(|j| Some(((j * 37) % N) as i32)).collect();
            let id: Vec<Option<i32>> = (i..end).map(|j| Some(j as i32)).collect();
            steps.push(Script::Rows(vec![ints(&k), ints(&id)]));
            i = end;
        }
        let rows = flat(steps, vec![key(0, Ty::Int, false, false)], Some(5));
        assert_eq!(col_i32(&rows, 0), vec![Some(0), Some(1), Some(2), Some(3), Some(4)]);
        // 元の行番号も一致すること（gather がずれていないか）。37 の法 5000 での
        // 逆元は 2973 なので、キー v を出した行は j = v * 2973 mod 5000。
        let expect: Vec<Option<i32>> = (0..5usize).map(|v| Some(((v * 2973) % N) as i32)).collect();
        assert_eq!(col_i32(&rows, 1), expect);
    }

    #[test]
    fn top_n_is_stable_across_compaction() {
        // 全部同じキー。先頭 3 行（入力順）が残ること。
        const N: usize = 5000;
        let mut steps = Vec::new();
        let mut i = 0usize;
        while i < N {
            let end = (i + BATCH_SIZE).min(N);
            let id: Vec<Option<i32>> = (i..end).map(|j| Some(j as i32)).collect();
            steps.push(Script::Rows(vec![ints(&vec![Some(1); end - i]), ints(&id)]));
            i = end;
        }
        let rows = flat(steps, vec![key(0, Ty::Int, false, false)], Some(3));
        assert_eq!(col_i32(&rows, 1), vec![Some(0), Some(1), Some(2)]);
    }

    // --- 出力バッチ ---------------------------------------------------------

    #[test]
    fn splits_output_into_batch_size_chunks() {
        const N: usize = 5000;
        let k: Vec<Option<i32>> = (0..N).map(|j| Some((N - 1 - j) as i32)).collect();
        let steps = vec![Script::Rows(vec![ints(&k)])];
        let batches = drive(steps, vec![key(0, Ty::Int, false, false)], None);
        assert_eq!(batches.iter().map(|b| b.len()).collect::<Vec<_>>(), vec![2048, 2048, 904]);
        // バッチ境界を跨いでも全体として昇順。
        let all: Vec<Vec<Value>> = batches.into_iter().flatten().collect();
        assert_eq!(all.len(), N);
        assert_eq!(col_i32(&all, 0), (0..N as i32).map(Some).collect::<Vec<_>>());
    }

    #[test]
    fn empty_input_emits_nothing() {
        assert!(drive(Vec::new(), vec![key(0, Ty::Int, false, false)], None).is_empty());
        // 0 行のバッチだけが来る場合も同じ。
        let steps = vec![Script::Rows(vec![ints(&[])]), Script::NeedIo];
        assert!(drive(steps, vec![key(0, Ty::Int, false, false)], None).is_empty());
    }

    #[test]
    fn schema_passes_through_unchanged() {
        let cols = vec![
            ints(&[Some(2), Some(1)]),
            vector(Ty::Varchar, &[Some(Value::Bytes(b"b".to_vec())), None]),
            vector(Ty::Double, &[Some(Value::F64(1.5)), Some(Value::F64(2.5))]),
        ];
        let batches = drive(vec![Script::Rows(cols)], vec![key(0, Ty::Int, false, false)], None);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[0][0].len(), 3, "列数はそのまま");
        assert_eq!(batches[0][0][0], Value::I32(1));
        assert_eq!(batches[0][0][1], Value::Null);
        assert_eq!(batches[0][0][2], Value::F64(2.5));
        assert_eq!(batches[0][1][1], Value::Bytes(b"b".to_vec()));
    }

    #[test]
    fn selection_vector_on_input_is_respected() {
        let mut op =
            Sort::new(Mock::new(Vec::new()), vec![key(0, Ty::Int, false, false)], None).unwrap();
        let mut cat = Catalog::default();
        let mut vm = Vm::new();
        let mut ctx =
            ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
        let mut batch = Batch::new(vec![ints(&[Some(5), Some(1), Some(9), Some(3)]), ids(4)]);
        batch.sel = Some(vec![2, 1]);
        op.absorb(batch, &mut ctx).unwrap();
        op.finish().unwrap();
        let b = match op.emit().unwrap() {
            Step::Ready(b) => b,
            _ => panic!("expected rows"),
        };
        assert_eq!(col_i32(&rows_of(&b), 0), vec![Some(1), Some(9)]);
        assert_eq!(col_i32(&rows_of(&b), 1), vec![Some(1), Some(2)]);
    }

    // --- 比較器の性質 -------------------------------------------------------

    #[test]
    fn comparator_is_a_total_order() {
        // NULL / NaN / 同値を混ぜても反対称性と推移律が破れないこと。
        let f = |v: f64| Some(Value::F64(v));
        let c = vector(
            Ty::Double,
            &[f(f64::NAN), None, f(0.0), f(-0.0), f(1.0), None, f(f64::NEG_INFINITY), f(f64::NAN)],
        );
        let keys = vec![SortKey { expr: col_expr(0, Ty::Double), desc: true, nulls_first: true }];
        let cols = vec![c];
        let n = 8u32;
        for a in 0..n {
            assert_eq!(cmp_row(&keys, &cols, a, a), Ordering::Equal);
            for b in 0..n {
                let ab = cmp_row(&keys, &cols, a, b);
                assert_eq!(ab.reverse(), cmp_row(&keys, &cols, b, a), "{a} vs {b}");
                if a != b {
                    assert_ne!(ab, Ordering::Equal, "{a} vs {b}");
                }
                for d in 0..n {
                    if ab == Ordering::Less && cmp_row(&keys, &cols, b, d) == Ordering::Less {
                        assert_eq!(cmp_row(&keys, &cols, a, d), Ordering::Less, "{a}<{b}<{d}");
                    }
                }
            }
        }
    }

    #[test]
    fn appended_vectors_keep_values_and_validity() {
        let mut dst = Vector::new(Ty::Varchar);
        let s1 = vector(Ty::Varchar, &[Some(Value::Bytes(b"ab".to_vec())), None]);
        let s2 = vector(
            Ty::Varchar,
            &[Some(Value::Bytes(b"".to_vec())), Some(Value::Bytes(b"cde".to_vec()))],
        );
        append(&mut dst, &s1).unwrap();
        append(&mut dst, &s2).unwrap();
        assert_eq!(dst.len(), 4);
        assert_eq!(dst.value_at(0), Value::Bytes(b"ab".to_vec()));
        assert_eq!(dst.value_at(1), Value::Null);
        assert_eq!(dst.value_at(2), Value::Bytes(Vec::new()));
        assert_eq!(dst.value_at(3), Value::Bytes(b"cde".to_vec()));

        // NULL が後から来る場合（dst に validity がまだ無い）。
        let mut dst = Vector::new(Ty::Boolean);
        append(&mut dst, &vector(Ty::Boolean, &[Some(Value::Bool(true))])).unwrap();
        append(&mut dst, &vector(Ty::Boolean, &[None, Some(Value::Bool(false))])).unwrap();
        // 逆に NULL の無いベクタが後から来ても validity の長さがずれないこと。
        append(&mut dst, &vector(Ty::Boolean, &[Some(Value::Bool(true))])).unwrap();
        assert_eq!(dst.len(), 4);
        assert_eq!(dst.value_at(0), Value::Bool(true));
        assert_eq!(dst.value_at(1), Value::Null);
        assert_eq!(dst.value_at(2), Value::Bool(false));
        assert_eq!(dst.value_at(3), Value::Bool(true));
    }

    #[test]
    fn nulls_survive_batches_that_have_none() {
        // NULL 入りバッチ → NULL 無しバッチ → NULL 入りバッチ、の順。
        let steps = vec![
            Script::Rows(vec![ints(&[Some(4), None]), ints(&[Some(0), Some(1)])]),
            Script::Rows(vec![ints(&[Some(2), Some(6)]), ints(&[Some(2), Some(3)])]),
            Script::Rows(vec![ints(&[None, Some(1)]), ints(&[Some(4), Some(5)])]),
        ];
        let rows = flat(steps, vec![key(0, Ty::Int, false, false)], None);
        assert_eq!(
            col_i32(&rows, 0),
            vec![Some(1), Some(2), Some(4), Some(6), None, None],
            "NULL は末尾に 2 つだけ"
        );
        assert_eq!(col_i32(&rows, 1), vec![Some(5), Some(2), Some(0), Some(3), Some(1), Some(4)]);
    }

    #[test]
    fn buffer_size_estimate_tracks_appends() {
        // 上限（256MiB）を実際に超えさせるテストは現実的でないので、
        // 見積もり関数が行数とともに増えることだけを確認する。
        let empty = vector(Ty::BigInt, &[]);
        assert_eq!(vector_bytes(&empty), 0);
        let mut big = Vector::new(Ty::BigInt);
        append(&mut big, &vector(Ty::BigInt, &[Some(Value::I64(1)), None])).unwrap();
        // 値 8B × 2 行 + validity。
        assert!(vector_bytes(&big) >= 16);
        let mut s = Vector::new(Ty::Varchar);
        append(&mut s, &vector(Ty::Varchar, &[Some(Value::Bytes(b"abcdef".to_vec()))])).unwrap();
        assert!(vector_bytes(&s) >= 6);
    }
}
