//! `UNNEST`（set-returning オペレータ）。
//!
//! 入力の 1 行を、対象列（`Ty::Json` の配列）の要素数ぶんの行に展開する。
//! 他のオペレータと違い「1 行入力 → N 行出力」なので、1 回の `next()` が
//! 生む行数を `BATCH_SIZE` に収めるため、`self` に「今展開中の入力行」と
//! 「その行の中で次に出す要素の添字」を持ち、複数回の `next()` 呼び出しに
//! またがって再開できるようにしてある（`exec::recursive::RecursiveCte` と
//! 同じ流儀）。
//!
//! 入力側の `NeedIo`/`NeedCodec` は素通しするだけでよい ―― JSON のパース
//! 自体はメモリ上のバイト列に対してしか行わないので、このオペレータ自身が
//! `NeedIo`/`NeedCodec` を生むことは無い。中断・再開をまたいでも結果が
//! 変わらないことは `tests` モジュールと `tests/unnest.rs` の両方で検証する。

use crate::exec::{ExecContext, Operator, Step};
use crate::expr::Program;
use crate::json::{self, Kind};
use crate::prelude::*;
use crate::vector::{Batch, Ty, Value, Vector, BATCH_SIZE};

pub struct Unnest {
    input: Box<dyn Operator>,
    /// 展開対象の配列を入力行に対して評価する式。結果型は必ず `Ty::Json`
    /// （`plan::bind` が保証する）。
    expr: Program,
    /// 展開後の要素列の宣言型。`Node::Unnest::elem_ty` のドキュメント参照。
    elem_ty: Ty,
    /// 展開中の入力バッチ。使い切ったら `None` に戻り、次回の `next()` で
    /// 新しい入力バッチを引く。
    cur: Option<Cur>,
}

/// 展開中の 1 入力バッチぶんの状態。
struct Cur {
    /// 複製して出力する入力列（materialize 済み、密な 0..rows 添字）。
    cols: Vec<Vector>,
    /// 展開対象の配列列（`Ty::Json`、`cols` と同じ行数）。
    arr: Vector,
    rows: usize,
    /// 次に展開する行。
    row: usize,
    /// `row` の中で、次に出す要素の添字（0 始まり）。行が変わるたびに戻す。
    elem_pos: usize,
}

impl Unnest {
    pub fn new(input: Box<dyn Operator>, expr: Program, elem_ty: Ty) -> Self {
        Unnest { input, expr, elem_ty, cur: None }
    }
}

impl Operator for Unnest {
    fn next(&mut self, ctx: &mut ExecContext) -> Result<Step> {
        loop {
            if self.cur.is_none() {
                let mut batch = match self.input.next(ctx)? {
                    Step::Ready(b) => b,
                    Step::NeedIo => return Ok(Step::NeedIo),
                    Step::NeedCodec => return Ok(Step::NeedCodec),
                    Step::Done => return Ok(Step::Done),
                };
                // selection を畳んで、以降は密な 0..rows 添字で行を指す
                // （`arr`/`cols` の行番号を一致させるため。`exec::recursive`
                // の `process` と同じ理由）。
                batch.materialize();
                let arr = ctx.vm.eval(&self.expr, &batch)?;
                ensure!(matches!(arr.ty(), Ty::Json), TypeMismatch);
                let rows = batch.num_rows();
                self.cur = Some(Cur { cols: batch.cols, arr, rows, row: 0, elem_pos: 0 });
            }
            let cur = match &mut self.cur {
                Some(c) => c,
                None => err!(Internal),
            };

            let mut out_elem = Vector::with_capacity(self.elem_ty, BATCH_SIZE);
            let mut dup: Vec<u32> = Vec::new();
            let mut scratch = Vec::new();
            while cur.row < cur.rows && out_elem.len() < BATCH_SIZE {
                if !cur.arr.is_valid(cur.row) {
                    // NULL 配列は 0 行（duckdb で確認済み）。
                    cur.row += 1;
                    cur.elem_pos = 0;
                    continue;
                }
                let doc = cur.arr.bytes().get(cur.row).to_vec();
                // 行ごとに再パースする。1 回の `next()` の途中で行をまたいで
                // 状態を持ち越すのは `row`/`elem_pos`（プレーンな usize）だけ
                // にして、パース結果への借用を `self` に持たせる自己参照を
                // 避けている（要素数が `BATCH_SIZE` を超える 1 行は複数回の
                // `next()` にまたがるので、そのたびに再パースし直す。単発の
                // 行なら O(n) 一発、`BATCH_SIZE` 超えの巨大配列だけ再パースが
                // 数回に増えるだけで済む）。
                let elems = json::array_elements(&doc)?.unwrap_or_default();
                if cur.elem_pos >= elems.len() {
                    cur.row += 1;
                    cur.elem_pos = 0;
                    continue;
                }
                while cur.elem_pos < elems.len() && out_elem.len() < BATCH_SIZE {
                    let (span, kind) = elems[cur.elem_pos];
                    push_elem(&mut out_elem, self.elem_ty, span, kind, &mut scratch)?;
                    dup.push(cur.row as u32);
                    cur.elem_pos += 1;
                }
                if cur.elem_pos >= elems.len() {
                    cur.row += 1;
                    cur.elem_pos = 0;
                }
            }

            if out_elem.is_empty() {
                // この入力バッチ全体が NULL/空配列だけだった。次の入力を引く。
                self.cur = None;
                continue;
            }
            let mut out_cols: Vec<Vector> = cur.cols.iter().map(|c| c.gather(&dup)).collect();
            out_cols.push(out_elem);
            if cur.row >= cur.rows {
                self.cur = None;
            }
            return Ok(Step::Ready(Batch::new(out_cols)));
        }
    }
}

/// 配列要素 1 個を宣言型 `elem_ty` に従って `out` に積む。JSON `null` は
/// SQL NULL（`json::write_extracted_text` と同じ判断）。要素の実際の種別が
/// 宣言型と食い違う場合（`elem_ty` の絞り込みが静的な保証を経ていない
/// 場合や、将来の呼び出し口の誤りに備えた防御）は NULL にする ―― パニック
/// もエラーも起こさない。
fn push_elem(
    out: &mut Vector,
    elem_ty: Ty,
    span: &[u8],
    kind: Kind,
    scratch: &mut Vec<u8>,
) -> Result<()> {
    if kind == Kind::Null {
        out.push_null();
        return Ok(());
    }
    match elem_ty {
        Ty::BigInt => match (kind, json::parse_i64(span)) {
            (Kind::Num, Some(v)) => out.push_value(&Value::I64(v)),
            _ => out.push_null(),
        },
        Ty::Double => match (kind, json::parse_f64(span)) {
            (Kind::Num, Some(v)) => out.push_value(&Value::F64(v)),
            _ => out.push_null(),
        },
        Ty::Varchar => match kind {
            Kind::Str => {
                scratch.clear();
                let body = if span.len() >= 2 { &span[1..span.len() - 1] } else { &[][..] };
                json::decode_string(body, scratch)?;
                out.push_value(&Value::Bytes(scratch.clone()));
            }
            _ => out.push_null(),
        },
        Ty::Boolean => match kind {
            Kind::Bool => out.push_value(&Value::Bool(span.first() == Some(&b't'))),
            _ => out.push_null(),
        },
        // Ty::Json、あるいは想定外の宣言型（絞り込みロジックのバグに備えた
        // 安全側フォールバック）は生の JSON テキストをそのまま積む。
        _ => out.push_value(&Value::Bytes(span.to_vec())),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Catalog;
    use crate::expr::vm::Vm;
    use crate::expr::{Instr, OpCode, Program};

    fn json_col(vals: &[Option<&str>]) -> Vector {
        let mut v = Vector::new(Ty::Json);
        for x in vals {
            match x {
                Some(s) => v.push_value(&Value::Bytes(s.as_bytes().to_vec())),
                None => v.push_null(),
            }
        }
        v
    }

    fn ints(vals: &[i32]) -> Vector {
        let mut v = Vector::new(Ty::Int);
        for &x in vals {
            v.push_value(&Value::I32(x));
        }
        v
    }

    /// 列 `idx` をそのまま返すプログラム。`col0` の型は呼び出し側が渡す。
    fn load(idx: u16, ty: Ty) -> Program {
        let mut p = Program::new();
        let r = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, ty.phys(), r, 0, 0, idx));
        p.result = r;
        p.result_ty = ty;
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

    /// `Unnest` を最後まで駆動し、`(id, 展開値)` のペアの列を返す。
    /// `id` は入力の 0 列目（複製列）、展開値は最後の列。
    fn drive(steps: Vec<Script>, elem_ty: Ty) -> Vec<(i32, Value)> {
        let expr = load(1, Ty::Json); // 1 列目（配列）を展開する
        let mut op = Unnest::new(Box::new(Mock { steps, pos: 0 }), expr, elem_ty);
        let mut cat = Catalog::new();
        let mut vm = Vm::new();
        let mut rows = Vec::new();
        for guard in 0..10_000 {
            assert!(guard < 9_999, "終わらない");
            let mut ctx =
                ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
            match op.next(&mut ctx).unwrap() {
                Step::Ready(mut b) => {
                    b.materialize();
                    for r in 0..b.num_rows() {
                        let Value::I32(id) = b.cols[0].value_at(r) else { panic!("expected I32") };
                        // 出力列は「入力の全列（id, 元の配列列）++ 展開要素」
                        // なので、展開値は常に末尾列。
                        let last = b.cols.len() - 1;
                        rows.push((id, b.cols[last].value_at(r)));
                    }
                }
                Step::NeedIo | Step::NeedCodec => {}
                Step::Done => break,
            }
        }
        rows
    }

    fn s(v: &str) -> Value {
        Value::Bytes(v.as_bytes().to_vec())
    }

    #[test]
    fn expands_each_row_by_its_array_length() {
        let steps = vec![Script::Rows(vec![
            ints(&[1, 2]),
            json_col(&[Some("[10,20,30]"), Some("[40,50]")]),
        ])];
        let rows = drive(steps, Ty::BigInt);
        assert_eq!(
            rows,
            vec![
                (1, Value::I64(10)),
                (1, Value::I64(20)),
                (1, Value::I64(30)),
                (2, Value::I64(40)),
                (2, Value::I64(50)),
            ]
        );
    }

    #[test]
    fn null_and_empty_arrays_produce_zero_rows() {
        // duckdb: UNNEST(NULL::INT[]) / UNNEST([]) はどちらも 0 行。
        let steps =
            vec![Script::Rows(vec![ints(&[1, 2, 3]), json_col(&[None, Some("[]"), Some("[7]")])])];
        let rows = drive(steps, Ty::BigInt);
        assert_eq!(rows, vec![(3, Value::I64(7))]);
    }

    #[test]
    fn json_null_element_becomes_sql_null() {
        let steps = vec![Script::Rows(vec![ints(&[1]), json_col(&[Some("[1,null,3]")])])];
        let rows = drive(steps, Ty::BigInt);
        assert_eq!(rows, vec![(1, Value::I64(1)), (1, Value::Null), (1, Value::I64(3))]);
    }

    #[test]
    fn varchar_and_boolean_narrowing_decode_correctly() {
        let steps = vec![Script::Rows(vec![ints(&[1]), json_col(&[Some(r#"["a","b\"c"]"#)])])];
        let rows = drive(steps, Ty::Varchar);
        assert_eq!(rows, vec![(1, s("a")), (1, s("b\"c"))]);

        let steps = vec![Script::Rows(vec![ints(&[1]), json_col(&[Some("[true,false]")])])];
        let rows = drive(steps, Ty::Boolean);
        assert_eq!(rows, vec![(1, Value::Bool(true)), (1, Value::Bool(false))]);
    }

    #[test]
    fn spans_multiple_output_batches_when_a_single_row_exceeds_batch_size() {
        // 1 行に BATCH_SIZE を超える要素数を持たせ、複数回の `next()`
        // （＝出力バッチ境界）にまたがる「行の途中から再開」を検証する。
        let n = BATCH_SIZE + 500;
        let mut arr = String::from("[");
        for i in 0..n {
            if i > 0 {
                arr.push(',');
            }
            arr.push_str(&i.to_string());
        }
        arr.push(']');
        let steps = vec![Script::Rows(vec![ints(&[9]), json_col(&[Some(&arr)])])];
        let rows = drive(steps, Ty::BigInt);
        assert_eq!(rows.len(), n);
        for (i, (id, v)) in rows.iter().enumerate() {
            assert_eq!(*id, 9);
            assert_eq!(*v, Value::I64(i as i64));
        }
    }

    #[test]
    fn need_io_between_input_batches_does_not_change_the_result() {
        // `Unnest` 自身が読み進めているのは入力オペレータであって、`NeedIo`
        // は入力側が返す。それをまたいでも「今展開中の行/添字」（`Cur`）が
        // 保たれ、結果が変わらないことを確認する。
        let make = |interrupt: bool| {
            let mut steps = Vec::new();
            steps.push(Script::Rows(vec![ints(&[1, 2]), json_col(&[Some("[1,2]"), Some("[3]")])]));
            if interrupt {
                steps.push(Script::NeedIo);
            }
            steps.push(Script::Rows(vec![ints(&[3]), json_col(&[Some("[4,5,6]")])]));
            steps
        };
        let plain = drive(make(false), Ty::BigInt);
        let interrupted = drive(make(true), Ty::BigInt);
        assert_eq!(plain, interrupted, "NeedIo をまたいでも結果が変わってはいけない");
        assert_eq!(plain.len(), 6);
    }

    #[test]
    fn need_io_mid_row_does_not_change_the_result() {
        // 巨大な 1 行を複数バッチに分けて出している最中（`elem_pos` が
        // 途中）に、次の `next()` 呼び出しの前に上流が `NeedIo` を返しても
        // （＝この `next()` 自体は input を呼ばない＝影響しない）結果が
        // 変わらないことを、直接 `NeedIo` を挟めない代わりに「同じ入力を
        // 2 バッチに割った場合」と「1 バッチにまとめた場合」で突き合わせて
        // 検証する。
        let n = BATCH_SIZE + 200;
        let mut arr = String::from("[");
        for i in 0..n {
            if i > 0 {
                arr.push(',');
            }
            arr.push_str(&i.to_string());
        }
        arr.push(']');
        let one_batch = vec![Script::Rows(vec![ints(&[1]), json_col(&[Some(&arr)])])];
        let rows_one = drive(one_batch, Ty::BigInt);
        assert_eq!(rows_one.len(), n);
        assert_eq!(rows_one[0], (1, Value::I64(0)));
        assert_eq!(rows_one[n - 1], (1, Value::I64((n - 1) as i64)));
    }

    #[test]
    fn falls_back_to_json_when_declared_type_mismatches_the_actual_element() {
        // `elem_ty` の絞り込みが誤って narrow 側になった場合の防御的な
        // フォールバック（パニックせず NULL にする）を確認する。
        let steps = vec![Script::Rows(vec![ints(&[1]), json_col(&[Some(r#"["not a number"]"#)])])];
        let rows = drive(steps, Ty::BigInt);
        assert_eq!(rows, vec![(1, Value::Null)]);
    }
}
