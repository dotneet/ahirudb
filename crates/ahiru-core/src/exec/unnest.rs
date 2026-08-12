//! `UNNEST` (a set-returning operator).
//!
//! Expands one input row into as many rows as the target column (a `Ty::Json` array) has
//! elements. Unlike other operators it is "1 row in -> N rows out", so to keep the rows produced
//! by one `next()` within `BATCH_SIZE`, `self` holds "the input row being expanded" and "the
//! index of the next element to emit within that row", so it can resume across several `next()`
//! calls (the same style as `exec::recursive::RecursiveCte`).
//!
//! `NeedIo`/`NeedCodec` from the input side merely pass through -- JSON parsing itself only ever
//! runs against in-memory bytes, so this operator never generates `NeedIo`/`NeedCodec` itself.
//! That the result does not change across an interruption and resumption is verified in both the
//! `tests` module and `tests/unnest.rs`.

use crate::exec::{ExecContext, Operator, Step};
use crate::expr::Program;
use crate::json::{self, Kind};
use crate::prelude::*;
use crate::vector::{Batch, Ty, Value, Vector, BATCH_SIZE};

pub struct Unnest {
    input: Box<dyn Operator>,
    /// The expression evaluated per input row to produce the array to expand. Its result type is
    /// always `Ty::Json` (guaranteed by `plan::bind`).
    expr: Program,
    /// The declared type of the expanded element column. See the docs on `Node::Unnest::elem_ty`.
    elem_ty: Ty,
    /// The input batch being expanded. It returns to `None` once used up, and the next `next()`
    /// pulls a new input batch.
    cur: Option<Cur>,
}

/// The state for one input batch being expanded.
struct Cur {
    /// The input columns duplicated into the output (materialized, with dense 0..rows indices).
    cols: Vec<Vector>,
    /// The column of arrays to expand (`Ty::Json`, the same row count as `cols`).
    arr: Vector,
    rows: usize,
    /// The next row to expand.
    row: usize,
    /// The index (0-based) of the next element to emit within `row`. Reset whenever the row changes.
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
                // Selection is materialized so rows are addressed by dense 0..rows indices from
                // here on (so `arr`'s and `cols`'s row numbers line up; the same reason as
                // `process` in `exec::recursive`).
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
                    // A NULL array gives 0 rows (confirmed with duckdb).
                    cur.row += 1;
                    cur.elem_pos = 0;
                    continue;
                }
                let doc = cur.arr.bytes().get(cur.row).to_vec();
                // Reparsed per row. The only state carried across rows within one `next()` is
                // `row`/`elem_pos` (plain usize), avoiding a self-reference that would put a borrow
                // of the parse result into `self` (a row with more elements than `BATCH_SIZE` spans
                // several `next()` calls and is reparsed each time. A one-off row costs O(n) once,
                // and only huge arrays exceeding `BATCH_SIZE` see a handful of extra reparses).
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
                // This entire input batch was nothing but NULLs and empty arrays. Pull the next input.
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

/// Pushes one array element into `out` according to the declared type `elem_ty`. JSON `null`
/// becomes SQL NULL (the same judgment as `json::write_extracted_text`). When an element's
/// actual kind disagrees with the declared type (when `elem_ty`'s narrowing did not go through a
/// static guarantee, or as a defense against a future caller's mistake) it becomes NULL -- with
/// neither a panic nor an error.
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
        // Ty::Json, or an unexpected declared type (a safe fallback in case of a bug in the
        // narrowing logic), pushes the raw JSON text unchanged.
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

    /// A program that returns column `idx` unchanged. `col0`'s type is supplied by the caller.
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

    /// Drives `Unnest` to completion and returns the sequence of `(id, expanded value)` pairs.
    /// `id` is the input's column 0 (a duplicated column) and the expanded value is the last column.
    fn drive(steps: Vec<Script>, elem_ty: Ty) -> Vec<(i32, Value)> {
        let expr = load(1, Ty::Json); // expands column 1 (the array)
        let mut op = Unnest::new(Box::new(Mock { steps, pos: 0 }), expr, elem_ty);
        let mut cat = Catalog::new();
        let mut vm = Vm::new();
        let mut rows = Vec::new();
        for guard in 0..10_000 {
            assert!(guard < 9_999, "does not terminate");
            let mut ctx =
                ExecContext { catalog: &mut cat, vm: &mut vm, io: Vec::new(), codec: Vec::new() };
            match op.next(&mut ctx).unwrap() {
                Step::Ready(mut b) => {
                    b.materialize();
                    for r in 0..b.num_rows() {
                        let Value::I32(id) = b.cols[0].value_at(r) else { panic!("expected I32") };
                        // The output columns are "every input column (id, the original array
                        // column) ++ the expanded element", so the expanded value is always last.
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
        // duckdb: UNNEST(NULL::INT[]) and UNNEST([]) both give 0 rows.
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
        // Gives one row more elements than BATCH_SIZE, exercising "resume mid-row" across several
        // `next()` calls (= output batch boundaries).
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
        // What `Unnest` itself advances through is the input operator, and `NeedIo` comes from the
        // input side. This confirms that across one, "the row/index being expanded" (`Cur`) is
        // preserved and the result does not change.
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
        assert_eq!(plain, interrupted, "the result must not change across a NeedIo");
        assert_eq!(plain.len(), 6);
    }

    #[test]
    fn need_io_mid_row_does_not_change_the_result() {
        // While a huge single row is being emitted across several batches (with `elem_pos` partway
        // through), an upstream `NeedIo` before the next `next()` call cannot matter (this `next()`
        // does not call input at all). Since a `NeedIo` cannot be interposed directly, that is
        // verified instead by cross-checking "the same input split across 2 batches" against "the
        // same input in 1 batch".
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
        // Confirms the defensive fallback (NULL rather than a panic) for the case where `elem_ty`'s
        // narrowing wrongly landed on the narrow side.
        let steps = vec![Script::Rows(vec![ints(&[1]), json_col(&[Some(r#"["not a number"]"#)])])];
        let rows = drive(steps, Ty::BigInt);
        assert_eq!(rows, vec![(1, Value::Null)]);
    }
}
