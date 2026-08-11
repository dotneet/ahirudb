//! バイトコード VM の実行ループ。
//!
//! 1 命令 = 1 カーネル呼び出し。分岐命令は持たないので、実行は命令列の
//! 単純な線形走査になる（mod.rs の設計判断を参照）。
//!
//! VM 側の責務は 3 つだけ:
//! - selection の解消。`LoadCol` で gather してしまい、以降のカーネルは
//!   常に密なベクタだけを見る（カーネルを selection の有無で 2 倍にしない）。
//! - 定数を長さ 1 のベクタとして持ち込むこと。stride 計算はカーネル側。
//! - 論理型の持ち回り。カーネルは物理型でしか動かないので、結果の論理型
//!   （DECIMAL のスケールなど）は VM が決めて渡す。

use crate::expr::kernels;
use crate::expr::{Instr, OpCode, Program, Reg};
use crate::prelude::*;
use crate::vector::{Batch, PhysType, Ty, Value, Vector};

/// レジスタファイル。クエリ中は使い回してアロケーションを抑える。
pub struct Vm {
    regs: Vec<Vector>,
}

impl Vm {
    pub fn new() -> Self {
        Vm { regs: Vec::new() }
    }

    /// `batch` に対して `p` を評価し、結果ベクタを返す。
    /// 返るベクタの長さは `batch.card()`。
    pub fn eval(&mut self, p: &Program, batch: &Batch) -> Result<Vector> {
        let n = batch.card();
        while self.regs.len() < p.num_regs as usize {
            self.regs.push(Vector::new(Ty::Null));
        }
        for ins in p.instrs.iter() {
            exec(&mut self.regs, ins, p, batch)?;
        }
        let r = p.result as usize;
        ensure!(r < self.regs.len(), Internal);
        // 取り出してレジスタは空に戻す。次回の eval で必ず上書きされる。
        let out = core::mem::replace(&mut self.regs[r], Vector::new(Ty::Null));
        if out.len() == n {
            Ok(out)
        } else if out.len() == 1 {
            // 定数だけの式。行数ぶんに広げてから返す。
            Ok(kernels::broadcast(&out, n))
        } else {
            err!(Internal)
        }
    }

    /// フィルタ用。Bool 結果のうち TRUE の行だけを `out` に積む。
    /// NULL は SQL の意味論どおり偽として扱う。
    pub fn eval_filter(&mut self, p: &Program, batch: &Batch, out: &mut Vec<u32>) -> Result<()> {
        let v = self.eval(p, batch)?;
        ensure!(v.data().phys() == PhysType::Bool, TypeMismatch);
        let bits = v.bools();
        // レジスタは gather 済みなので、結果の i 番目は sel[i] 行目に対応する。
        // 積むのは元のバッチ行番号。
        match &batch.sel {
            Some(sel) => {
                // 添字は結果ビットと selection の両方を引くのに使う。
                #[allow(clippy::needless_range_loop)]
                for i in 0..v.len() {
                    if bits.get(i) && v.is_valid(i) {
                        out.push(sel[i]);
                    }
                }
            }
            None => {
                for i in 0..v.len() {
                    if bits.get(i) && v.is_valid(i) {
                        out.push(i as u32);
                    }
                }
            }
        }
        Ok(())
    }
}

impl Default for Vm {
    fn default() -> Self {
        Vm::new()
    }
}

fn reg(regs: &[Vector], i: Reg) -> Result<&Vector> {
    match regs.get(i as usize) {
        Some(v) => Ok(v),
        None => err!(Internal),
    }
}

/// 物理型に対応する既定の論理型。型情報が失われた場合の受け皿。
fn default_ty(p: PhysType) -> Ty {
    match p {
        PhysType::Bool => Ty::Boolean,
        PhysType::I32 => Ty::Int,
        PhysType::I64 => Ty::BigInt,
        PhysType::F64 => Ty::Double,
        PhysType::I128 => Ty::HugeInt,
        PhysType::Bytes => Ty::Varchar,
    }
}

/// 二項演算の結果論理型。binder が両辺を揃えている前提だが、揃っていない
/// （＝ 統合できない）場合は物理型から決まる既定型に落とす。
fn binary_ty(phys: PhysType, a: Ty, b: Ty) -> Ty {
    match Ty::unify(a, b) {
        Some(t) if t.phys() == phys => t,
        _ => default_ty(phys),
    }
}

/// 定数プールの 1 要素を長さ 1 のベクタにする。NULL 定数は validity 0 の 1 行。
fn const_vector(ty: Ty, v: &Value) -> Result<Vector> {
    let mut out = Vector::new(ty);
    if v.is_null() {
        out.push_null();
    } else {
        ensure!(v.default_ty().phys() == ty.phys(), TypeMismatch);
        out.push_value(v);
    }
    Ok(out)
}

fn exec(regs: &mut [Vector], ins: &Instr, p: &Program, batch: &Batch) -> Result<()> {
    use OpCode::*;
    let out = match ins.op {
        LoadCol => {
            let c = ins.aux as usize;
            ensure!(c < batch.cols.len(), Internal);
            match &batch.sel {
                // ここで selection を解消しておくと、以降のカーネルは
                // selection を一切知らなくて済む。
                Some(sel) => batch.cols[c].gather(sel),
                None => batch.cols[c].clone(),
            }
        }
        LoadConst => {
            let c = ins.aux as usize;
            ensure!(c < p.consts.len(), Internal);
            let (ty, v) = &p.consts[c];
            const_vector(*ty, v)?
        }
        Add | Sub | Mul | Div | Mod | Neg => {
            let a = reg(regs, ins.a)?;
            // 単項の Neg は b を使わない。未初期化レジスタを読まないよう a を渡す。
            let b = if ins.op == Neg { a } else { reg(regs, ins.b)? };
            kernels::arith(ins.op, binary_ty(ins.ty, a.ty(), b.ty()), a, b)?
        }
        Eq | Ne | Lt | Le | Gt | Ge => {
            let a = reg(regs, ins.a)?;
            let b = reg(regs, ins.b)?;
            kernels::compare(ins.op, ins.ty, a, b)?
        }
        And | Or => {
            let a = reg(regs, ins.a)?;
            let b = reg(regs, ins.b)?;
            kernels::logic(ins.op, a, b)?
        }
        Not => kernels::not(reg(regs, ins.a)?)?,
        IsNull => kernels::is_null(reg(regs, ins.a)?, true),
        IsNotNull => kernels::is_null(reg(regs, ins.a)?, false),
        Cast => {
            let c = ins.aux as usize;
            ensure!(c < p.casts.len(), Internal);
            let spec = p.casts[c];
            kernels::cast(spec.from, spec.to, reg(regs, ins.a)?)?
        }
        TryCast => {
            let c = ins.aux as usize;
            ensure!(c < p.casts.len(), Internal);
            let spec = p.casts[c];
            let src = reg(regs, ins.a)?;
            match kernels::try_cast(spec.from, spec.to, src) {
                Ok(v) => v,
                // 組み合わせ自体が変換不能（`InvalidCast` 等）。行単位の失敗は
                // `kernels::cast` がエラーにせず NULL を返すので、ここに来るのは
                // 「その型ペアはそもそも変換できない」場合だけ。エラーを伝播
                // せず、全行 NULL のベクタに落とす。
                Err(_) => {
                    let n = src.len();
                    let mut out = Vector::new(spec.to);
                    for _ in 0..n {
                        out.push_null();
                    }
                    out
                }
            }
        }
        Call => {
            let c = ins.aux as usize;
            ensure!(c < p.calls.len(), Internal);
            let spec = &p.calls[c];
            // 引数は別表にあるので、レジスタから参照を集めてから渡す。
            let mut args = Vec::with_capacity(spec.args.len());
            for &r in &spec.args {
                args.push(reg(regs, r)?);
            }
            crate::expr::funcs::call(spec.func, spec.result_ty, &args)?
        }
        TsAddInterval => kernels::ts_add_interval(reg(regs, ins.a)?, reg(regs, ins.b)?)?,
        IntervalAdd => kernels::interval_add(reg(regs, ins.a)?, reg(regs, ins.b)?)?,
        IntervalNeg => kernels::interval_neg(reg(regs, ins.a)?)?,
        IntervalMul => kernels::interval_mul(reg(regs, ins.a)?, reg(regs, ins.b)?)?,
        Like => kernels::like(reg(regs, ins.a)?, reg(regs, ins.b)?)?,
        Concat => {
            let a = reg(regs, ins.a)?;
            let b = reg(regs, ins.b)?;
            kernels::concat(a, b, binary_ty(PhysType::Bytes, a.ty(), b.ty()))?
        }
        Select => {
            let c = reg(regs, ins.a)?;
            let t = reg(regs, ins.b)?;
            let e = reg(regs, ins.aux)?;
            kernels::pick(Some(c), t, e, binary_ty(ins.ty, t.ty(), e.ty()))?
        }
        Coalesce => {
            let a = reg(regs, ins.a)?;
            let b = reg(regs, ins.b)?;
            kernels::pick(None, a, b, binary_ty(ins.ty, a.ty(), b.ty()))?
        }
    };
    let d = ins.dst as usize;
    ensure!(d < regs.len(), Internal);
    regs[d] = out;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::code_of;
    use crate::vector::Data;

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

    fn ints(vals: &[i32]) -> Vector {
        col(Ty::Int, &vals.iter().map(|v| Some(Value::I32(*v))).collect::<Vec<_>>())
    }

    fn bytes(vals: &[&[u8]]) -> Vector {
        col(Ty::Varchar, &vals.iter().map(|v| Some(Value::Bytes(v.to_vec()))).collect::<Vec<_>>())
    }

    fn bools(vals: &[Option<bool>]) -> Vector {
        col(Ty::Boolean, &vals.iter().map(|v| v.map(Value::Bool)).collect::<Vec<_>>())
    }

    /// `col0 op col1` を評価する。
    fn eval2(op: OpCode, ty: PhysType, a: Vector, b: Vector) -> Vector {
        let mut p = Program::new();
        let r0 = p.alloc_reg();
        let r1 = p.alloc_reg();
        let r2 = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, ty, r0, 0, 0, 0));
        p.push(Instr::with_aux(OpCode::LoadCol, ty, r1, 0, 0, 1));
        p.push(Instr::new(op, ty, r2, r0, r1));
        p.result = r2;
        let batch = Batch::new(vec![a, b]);
        Vm::new().eval(&p, &batch).unwrap()
    }

    /// `col0 op const` / `const op col0` を評価する。`const_first` で順序を入れ替える。
    fn eval_const(
        op: OpCode,
        ty: PhysType,
        a: Vector,
        k: (Ty, Value),
        const_first: bool,
    ) -> Vector {
        let mut p = Program::new();
        let rc = p.alloc_reg();
        let rk = p.alloc_reg();
        let rd = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, ty, rc, 0, 0, 0));
        let ci = p.add_const(k.0, k.1);
        p.push(Instr::with_aux(OpCode::LoadConst, ty, rk, 0, 0, ci));
        if const_first {
            p.push(Instr::new(op, ty, rd, rk, rc));
        } else {
            p.push(Instr::new(op, ty, rd, rc, rk));
        }
        p.result = rd;
        let batch = Batch::new(vec![a]);
        Vm::new().eval(&p, &batch).unwrap()
    }

    fn unary(op: OpCode, ty: PhysType, a: Vector) -> Vector {
        let mut p = Program::new();
        let r0 = p.alloc_reg();
        let r1 = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, ty, r0, 0, 0, 0));
        p.push(Instr::new(op, ty, r1, r0, r0));
        p.result = r1;
        let batch = Batch::new(vec![a]);
        Vm::new().eval(&p, &batch).unwrap()
    }

    fn i32s_of(v: &Vector) -> Vec<i32> {
        v.i32s().to_vec()
    }

    /// Bool ベクタを (値, valid) の組で取り出す。NULL の値は無視する。
    fn tri(v: &Vector) -> Vec<Option<bool>> {
        (0..v.len()).map(|i| if v.is_valid(i) { Some(v.bools().get(i)) } else { None }).collect()
    }

    // --- 算術 ---------------------------------------------------------------

    #[test]
    fn arith_i32_vec_vec() {
        let a = || ints(&[10, -7, 9]);
        let b = || ints(&[3, 2, -4]);
        assert_eq!(i32s_of(&eval2(OpCode::Add, PhysType::I32, a(), b())), vec![13, -5, 5]);
        assert_eq!(i32s_of(&eval2(OpCode::Sub, PhysType::I32, a(), b())), vec![7, -9, 13]);
        assert_eq!(i32s_of(&eval2(OpCode::Mul, PhysType::I32, a(), b())), vec![30, -14, -36]);
        assert_eq!(i32s_of(&eval2(OpCode::Div, PhysType::I32, a(), b())), vec![3, -3, -2]);
        assert_eq!(i32s_of(&eval2(OpCode::Mod, PhysType::I32, a(), b())), vec![1, -1, 1]);
        assert_eq!(i32s_of(&unary(OpCode::Neg, PhysType::I32, a())), vec![-10, 7, -9]);
    }

    #[test]
    fn arith_i32_stride_in_both_positions() {
        // col - 3
        let r = eval_const(
            OpCode::Sub,
            PhysType::I32,
            ints(&[10, 20, 30]),
            (Ty::Int, Value::I32(3)),
            false,
        );
        assert_eq!(i32s_of(&r), vec![7, 17, 27]);
        // 3 - col （定数が左でも同じカーネル）
        let r = eval_const(
            OpCode::Sub,
            PhysType::I32,
            ints(&[10, 20, 30]),
            (Ty::Int, Value::I32(3)),
            true,
        );
        assert_eq!(i32s_of(&r), vec![-7, -17, -27]);
    }

    #[test]
    fn arith_i64_i128_f64() {
        let a = col(Ty::BigInt, &[Some(Value::I64(7)), Some(Value::I64(-9))]);
        let b = col(Ty::BigInt, &[Some(Value::I64(2)), Some(Value::I64(4))]);
        assert_eq!(eval2(OpCode::Mul, PhysType::I64, a, b).i64s(), &[14, -36]);

        let a = col(Ty::HugeInt, &[Some(Value::I128(i128::from(i64::MAX) * 3))]);
        let b = col(Ty::HugeInt, &[Some(Value::I128(3))]);
        assert_eq!(eval2(OpCode::Div, PhysType::I128, a, b).i128s(), &[i128::from(i64::MAX)]);

        let a = col(Ty::Double, &[Some(Value::F64(1.5)), Some(Value::F64(-2.0))]);
        let b = col(Ty::Double, &[Some(Value::F64(0.5)), Some(Value::F64(4.0))]);
        assert_eq!(eval2(OpCode::Add, PhysType::F64, a, b).f64s(), &[2.0, 2.0]);

        // 定数側のストライド（I64 / F64）。
        let r = eval_const(
            OpCode::Add,
            PhysType::I64,
            col(Ty::BigInt, &[Some(Value::I64(1)), Some(Value::I64(2))]),
            (Ty::BigInt, Value::I64(10)),
            false,
        );
        assert_eq!(r.i64s(), &[11, 12]);
        let r = eval_const(
            OpCode::Div,
            PhysType::F64,
            col(Ty::Double, &[Some(Value::F64(1.0)), Some(Value::F64(2.0))]),
            (Ty::Double, Value::F64(4.0)),
            true,
        );
        assert_eq!(r.f64s(), &[4.0, 2.0]);
    }

    #[test]
    fn integer_overflow_wraps() {
        let a = ints(&[i32::MAX]);
        let b = ints(&[1]);
        assert_eq!(i32s_of(&eval2(OpCode::Add, PhysType::I32, a, b)), vec![i32::MIN]);

        let a = col(Ty::BigInt, &[Some(Value::I64(i64::MIN))]);
        let b = col(Ty::BigInt, &[Some(Value::I64(1))]);
        assert_eq!(eval2(OpCode::Sub, PhysType::I64, a, b).i64s(), &[i64::MAX]);
    }

    #[test]
    fn division_edge_cases_yield_null() {
        let r = eval2(OpCode::Div, PhysType::I32, ints(&[10, 5]), ints(&[0, 2]));
        assert!(!r.is_valid(0));
        assert!(r.is_valid(1));
        let r = eval2(OpCode::Mod, PhysType::I32, ints(&[10, 5]), ints(&[0, 2]));
        assert!(!r.is_valid(0));
        assert_eq!(r.i32s()[1], 1);

        // MIN / -1 は 2 の補数で表現できないのでパニックさせず NULL。
        let r = eval2(OpCode::Div, PhysType::I32, ints(&[i32::MIN]), ints(&[-1]));
        assert!(!r.is_valid(0));
        let a = col(Ty::BigInt, &[Some(Value::I64(i64::MIN))]);
        let b = col(Ty::BigInt, &[Some(Value::I64(-1))]);
        assert!(!eval2(OpCode::Div, PhysType::I64, a, b).is_valid(0));
        let a = col(Ty::HugeInt, &[Some(Value::I128(i128::MIN))]);
        let b = col(Ty::HugeInt, &[Some(Value::I128(-1))]);
        assert!(!eval2(OpCode::Mod, PhysType::I128, a, b).is_valid(0));

        // 浮動小数は IEEE のまま。
        let a = col(Ty::Double, &[Some(Value::F64(1.0)), Some(Value::F64(0.0))]);
        let b = col(Ty::Double, &[Some(Value::F64(0.0)), Some(Value::F64(0.0))]);
        let r = eval2(OpCode::Div, PhysType::F64, a, b);
        assert!(r.is_valid(0) && r.f64s()[0] == f64::INFINITY);
        assert!(r.is_valid(1) && r.f64s()[1].is_nan());
    }

    #[test]
    fn null_propagates_through_arithmetic() {
        let a = col(Ty::Int, &[Some(Value::I32(1)), None]);
        let r = eval_const(OpCode::Add, PhysType::I32, a, (Ty::Int, Value::I32(1)), false);
        assert!(r.is_valid(0));
        assert!(!r.is_valid(1));
        assert_eq!(r.i32s()[0], 2);

        // NULL 定数を足すと全行 NULL。
        let a = ints(&[1, 2]);
        let r = eval_const(OpCode::Add, PhysType::I32, a, (Ty::Int, Value::Null), false);
        assert!(!r.is_valid(0) && !r.is_valid(1));
    }

    // --- 比較 ---------------------------------------------------------------

    const ALL_CMP: [OpCode; 6] =
        [OpCode::Eq, OpCode::Ne, OpCode::Lt, OpCode::Le, OpCode::Gt, OpCode::Ge];

    /// (a<b, a=b, a>b) の 3 行に対する 6 比較の期待値。
    const EXPECT: [[bool; 3]; 6] = [
        [false, true, false], // Eq
        [true, false, true],  // Ne
        [true, false, false], // Lt
        [true, true, false],  // Le
        [false, false, true], // Gt
        [false, true, true],  // Ge
    ];

    fn check_cmp(ty: PhysType, a: Vector, b: Vector) {
        for (k, op) in ALL_CMP.iter().enumerate() {
            let r = eval2(*op, ty, a.clone(), b.clone());
            let got: Vec<bool> = (0..3).map(|i| r.bools().get(i)).collect();
            assert_eq!(got, EXPECT[k].to_vec(), "op index {k} on {ty:?}");
            assert_eq!(r.ty(), Ty::Boolean);
        }
    }

    #[test]
    fn comparisons_on_every_phys_type() {
        check_cmp(PhysType::I32, ints(&[1, 2, 3]), ints(&[2, 2, 2]));
        check_cmp(
            PhysType::I64,
            col(Ty::BigInt, &[Some(Value::I64(1)), Some(Value::I64(2)), Some(Value::I64(3))]),
            col(Ty::BigInt, &[Some(Value::I64(2)), Some(Value::I64(2)), Some(Value::I64(2))]),
        );
        check_cmp(
            PhysType::I128,
            col(Ty::HugeInt, &[Some(Value::I128(1)), Some(Value::I128(2)), Some(Value::I128(3))]),
            col(Ty::HugeInt, &[Some(Value::I128(2)), Some(Value::I128(2)), Some(Value::I128(2))]),
        );
        check_cmp(
            PhysType::F64,
            col(Ty::Double, &[Some(Value::F64(1.0)), Some(Value::F64(2.0)), Some(Value::F64(3.0))]),
            col(Ty::Double, &[Some(Value::F64(2.0)), Some(Value::F64(2.0)), Some(Value::F64(2.0))]),
        );
        // Bytes は辞書順。
        let mid = || bytes(&[&b"abd"[..], &b"abd"[..], &b"abd"[..]]);
        check_cmp(PhysType::Bytes, bytes(&[&b"abc"[..], &b"abd"[..], &b"abe"[..]]), mid());
        // 前方一致する長短も辞書順（短いほうが小さい）。
        check_cmp(PhysType::Bytes, bytes(&[&b"ab"[..], &b"abd"[..], &b"abdx"[..]]), mid());
        // Bool は false < true。
        check_cmp(
            PhysType::Bool,
            bools(&[Some(false), Some(true), Some(true)]),
            bools(&[Some(true), Some(true), Some(false)]),
        );
    }

    #[test]
    fn nan_is_unordered_but_not_equal() {
        let a = col(Ty::Double, &[Some(Value::F64(f64::NAN))]);
        let b = col(Ty::Double, &[Some(Value::F64(1.0))]);
        assert!(!eval2(OpCode::Eq, PhysType::F64, a.clone(), b.clone()).bools().get(0));
        assert!(!eval2(OpCode::Lt, PhysType::F64, a.clone(), b.clone()).bools().get(0));
        assert!(!eval2(OpCode::Ge, PhysType::F64, a.clone(), b.clone()).bools().get(0));
        assert!(eval2(OpCode::Ne, PhysType::F64, a, b).bools().get(0));
    }

    #[test]
    fn comparison_with_null_is_null() {
        let a = col(Ty::Int, &[Some(Value::I32(1)), None]);
        let b = ints(&[1, 1]);
        for op in ALL_CMP.iter() {
            let r = eval2(*op, PhysType::I32, a.clone(), b.clone());
            assert!(r.is_valid(0));
            assert!(!r.is_valid(1), "NULL 比較は NULL のまま");
        }
    }

    // --- 三値論理 -----------------------------------------------------------

    const T: Option<bool> = Some(true);
    const F: Option<bool> = Some(false);
    const N: Option<bool> = None;

    #[test]
    fn and_or_truth_tables() {
        let a = bools(&[T, T, T, F, F, F, N, N, N]);
        let b = bools(&[T, F, N, T, F, N, T, F, N]);
        let and = eval2(OpCode::And, PhysType::Bool, a.clone(), b.clone());
        assert_eq!(tri(&and), vec![T, F, N, F, F, F, N, F, N]);
        let or = eval2(OpCode::Or, PhysType::Bool, a.clone(), b.clone());
        assert_eq!(tri(&or), vec![T, T, T, T, F, N, T, N, N]);
    }

    #[test]
    fn not_keeps_null() {
        let r = unary(OpCode::Not, PhysType::Bool, bools(&[T, F, N]));
        assert_eq!(tri(&r), vec![F, T, N]);
    }

    #[test]
    fn logic_with_constant_operand() {
        // NULL AND FALSE = FALSE、NULL OR TRUE = TRUE（定数側 stride 0 でも同じ）。
        let r = eval_const(
            OpCode::And,
            PhysType::Bool,
            bools(&[T, F, N]),
            (Ty::Boolean, Value::Bool(false)),
            false,
        );
        assert_eq!(tri(&r), vec![F, F, F]);
        let r = eval_const(
            OpCode::Or,
            PhysType::Bool,
            bools(&[T, F, N]),
            (Ty::Boolean, Value::Bool(true)),
            true,
        );
        assert_eq!(tri(&r), vec![T, T, T]);
    }

    // --- NULL 述語・選択 ----------------------------------------------------

    #[test]
    fn is_null_and_is_not_null() {
        let c = col(Ty::Int, &[Some(Value::I32(1)), None, Some(Value::I32(3))]);
        let r = unary(OpCode::IsNull, PhysType::I32, c.clone());
        assert_eq!(tri(&r), vec![F, T, F]);
        assert!(!r.has_nulls(), "IsNull の結果は決して NULL にならない");
        let r = unary(OpCode::IsNotNull, PhysType::I32, c);
        assert_eq!(tri(&r), vec![T, F, T]);

        // NULL の無い列でも動く。
        let r = unary(OpCode::IsNull, PhysType::I32, ints(&[1, 2]));
        assert_eq!(tri(&r), vec![F, F]);
    }

    #[test]
    fn select_null_condition_takes_else_branch() {
        let mut p = Program::new();
        let rc = p.alloc_reg();
        let rt = p.alloc_reg();
        let re = p.alloc_reg();
        let rd = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, PhysType::Bool, rc, 0, 0, 0));
        p.push(Instr::with_aux(OpCode::LoadCol, PhysType::I32, rt, 0, 0, 1));
        let ci = p.add_const(Ty::Int, Value::I32(-1));
        p.push(Instr::with_aux(OpCode::LoadConst, PhysType::I32, re, 0, 0, ci));
        p.push(Instr::with_aux(OpCode::Select, PhysType::I32, rd, rc, rt, re));
        p.result = rd;
        let batch = Batch::new(vec![
            bools(&[T, F, N]),
            col(Ty::Int, &[Some(Value::I32(10)), Some(Value::I32(20)), Some(Value::I32(30))]),
        ]);
        let r = Vm::new().eval(&p, &batch).unwrap();
        // 条件が NULL / FALSE のときは else（長さ 1 の定数）を採る。
        assert_eq!(i32s_of(&r), vec![10, -1, -1]);
        assert!(!r.has_nulls());
    }

    #[test]
    fn select_propagates_branch_nullness() {
        let mut p = Program::new();
        let rc = p.alloc_reg();
        let rt = p.alloc_reg();
        let re = p.alloc_reg();
        let rd = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, PhysType::Bool, rc, 0, 0, 0));
        p.push(Instr::with_aux(OpCode::LoadCol, PhysType::I32, rt, 0, 0, 1));
        p.push(Instr::with_aux(OpCode::LoadCol, PhysType::I32, re, 0, 0, 2));
        p.push(Instr::with_aux(OpCode::Select, PhysType::I32, rd, rc, rt, re));
        p.result = rd;
        let batch = Batch::new(vec![
            bools(&[T, F]),
            col(Ty::Int, &[None, Some(Value::I32(2))]),
            col(Ty::Int, &[Some(Value::I32(3)), None]),
        ]);
        let r = Vm::new().eval(&p, &batch).unwrap();
        assert!(!r.is_valid(0) && !r.is_valid(1));
    }

    #[test]
    fn coalesce_picks_first_valid() {
        let a = col(Ty::Int, &[Some(Value::I32(1)), None, None]);
        let b = col(Ty::Int, &[Some(Value::I32(9)), Some(Value::I32(8)), None]);
        let r = eval2(OpCode::Coalesce, PhysType::I32, a, b);
        assert_eq!(i32s_of(&r)[0], 1);
        assert_eq!(i32s_of(&r)[1], 8);
        assert!(!r.is_valid(2));
    }

    // --- Bytes --------------------------------------------------------------

    #[test]
    fn concat_bytes() {
        let r = eval_const(
            OpCode::Concat,
            PhysType::Bytes,
            bytes(&[b"ab", b""]),
            (Ty::Varchar, Value::Bytes(b"!".to_vec())),
            false,
        );
        assert_eq!(r.bytes().get(0), b"ab!");
        assert_eq!(r.bytes().get(1), b"!");
        let r = eval2(OpCode::Concat, PhysType::Bytes, bytes(&[b"x"]), bytes(&[b"y"]));
        assert_eq!(r.bytes().get(0), b"xy");
    }

    #[test]
    fn like_through_vm() {
        let subject = bytes(&[b"hello", b"help", b"yellow", b""]);
        let r = eval_const(
            OpCode::Like,
            PhysType::Bytes,
            subject,
            (Ty::Varchar, Value::Bytes(b"hel%".to_vec())),
            false,
        );
        assert_eq!(tri(&r), vec![T, T, F, F]);

        // NULL 入力は NULL のまま。
        let s = col(Ty::Varchar, &[None]);
        let r = eval_const(
            OpCode::Like,
            PhysType::Bytes,
            s,
            (Ty::Varchar, Value::Bytes(b"%".to_vec())),
            false,
        );
        assert!(!r.is_valid(0));
    }

    // --- キャスト -----------------------------------------------------------

    fn cast_of(from: Ty, to: Ty, v: Vector) -> Result<Vector> {
        let mut p = Program::new();
        let r0 = p.alloc_reg();
        let r1 = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, from.phys(), r0, 0, 0, 0));
        let ci = p.add_cast(from, to);
        p.push(Instr::with_aux(OpCode::Cast, from.phys(), r1, r0, 0, ci));
        p.result = r1;
        p.result_ty = to;
        let batch = Batch::new(vec![v]);
        Vm::new().eval(&p, &batch)
    }

    #[test]
    fn cast_integer_widening_and_narrowing() {
        let r = cast_of(Ty::Int, Ty::BigInt, ints(&[7, -7])).unwrap();
        assert_eq!(r.i64s(), &[7, -7]);
        assert_eq!(r.ty(), Ty::BigInt);
        let r = cast_of(Ty::Int, Ty::HugeInt, ints(&[7])).unwrap();
        assert_eq!(r.i128s(), &[7]);
        // 収まらない縮小はエラーではなくその行だけ NULL。
        let src = col(Ty::BigInt, &[Some(Value::I64(5)), Some(Value::I64(i64::MAX))]);
        let r = cast_of(Ty::BigInt, Ty::Int, src).unwrap();
        assert_eq!(r.i32s()[0], 5);
        assert!(!r.is_valid(1));
    }

    #[test]
    fn cast_between_int_and_float() {
        let r = cast_of(Ty::Int, Ty::Double, ints(&[3, -4])).unwrap();
        assert_eq!(r.f64s(), &[3.0, -4.0]);
        // 浮動小数 → 整数は切り捨てではなく丸める（DuckDB / PostgreSQL と同じ）。
        let src = col(
            Ty::Double,
            &[Some(Value::F64(3.9)), Some(Value::F64(-3.9)), Some(Value::F64(1e30))],
        );
        let r = cast_of(Ty::Double, Ty::Int, src).unwrap();
        assert_eq!(r.i32s()[0], 4);
        assert_eq!(r.i32s()[1], -4);
        assert!(!r.is_valid(2), "範囲外は NULL");

        // ちょうど半端は偶数側へ（銀行丸め）。1.5 → 2、4.5 → 4。
        let src = col(
            Ty::Double,
            &[
                Some(Value::F64(1.5)),
                Some(Value::F64(4.5)),
                Some(Value::F64(-1.5)),
                Some(Value::F64(-4.5)),
            ],
        );
        let r = cast_of(Ty::Double, Ty::Int, src).unwrap();
        assert_eq!(r.i32s(), &[2, 4, -2, -4]);
        // NaN / inf も NULL。
        let src = col(Ty::Double, &[Some(Value::F64(f64::NAN)), Some(Value::F64(f64::INFINITY))]);
        let r = cast_of(Ty::Double, Ty::BigInt, src).unwrap();
        assert!(!r.is_valid(0) && !r.is_valid(1));
    }

    #[test]
    fn cast_decimal_rescale() {
        let d2 = Ty::Decimal { precision: 10, scale: 2 };
        let d4 = Ty::Decimal { precision: 12, scale: 4 };
        // 12.34 → スケール 4 へ（10^2 倍）
        let src = col(Ty::Decimal { precision: 10, scale: 2 }, &[Some(Value::I64(1234))]);
        assert_eq!(cast_of(d2, d4, src).unwrap().i64s(), &[123_400]);
        // 逆向きは 0 から遠ざかる向きに丸める（DuckDB と同じ）。
        // 12.3456 → 12.35、-12.3456 → -12.35。
        let src = col(d4, &[Some(Value::I64(123_456)), Some(Value::I64(-123_456))]);
        assert_eq!(cast_of(d4, d2, src).unwrap().i64s(), &[1235, -1235]);
        // ちょうど半端も 0 から遠ざける。1.235 → 1.24。
        let d3 = Ty::Decimal { precision: 10, scale: 3 };
        let src = col(d3, &[Some(Value::I64(1235)), Some(Value::I64(-1235))]);
        assert_eq!(cast_of(d3, d2, src).unwrap().i64s(), &[124, -124]);
        // DECIMAL → DOUBLE はスケールで割る。
        let src = col(d2, &[Some(Value::I64(1234))]);
        assert_eq!(cast_of(d2, Ty::Double, src).unwrap().f64s(), &[12.34]);
        // DOUBLE → DECIMAL はスケール倍して丸める。12.349 → 12.35。
        let src = col(Ty::Double, &[Some(Value::F64(12.349))]);
        assert_eq!(cast_of(Ty::Double, d2, src).unwrap().i64s(), &[1235]);
        // 整数 → DECIMAL。
        let src = col(Ty::BigInt, &[Some(Value::I64(3))]);
        assert_eq!(cast_of(Ty::BigInt, d2, src).unwrap().i64s(), &[300]);
    }

    #[test]
    fn cast_numeric_to_varchar_and_back() {
        let src = col(Ty::BigInt, &[Some(Value::I64(0)), Some(Value::I64(-1234))]);
        let r = cast_of(Ty::BigInt, Ty::Varchar, src).unwrap();
        assert_eq!(r.bytes().get(0), b"0");
        assert_eq!(r.bytes().get(1), b"-1234");

        let d2 = Ty::Decimal { precision: 10, scale: 2 };
        let src = col(d2, &[Some(Value::I64(1234)), Some(Value::I64(-5))]);
        let r = cast_of(d2, Ty::Varchar, src).unwrap();
        assert_eq!(r.bytes().get(0), b"12.34");
        assert_eq!(r.bytes().get(1), b"-0.05");

        let src = col(Ty::Double, &[Some(Value::F64(1.5)), Some(Value::F64(-0.25))]);
        let r = cast_of(Ty::Double, Ty::Varchar, src).unwrap();
        assert_eq!(r.bytes().get(0), b"1.5");
        assert_eq!(r.bytes().get(1), b"-0.25");

        // VARCHAR → 数値。読めない行だけ NULL。
        let src = bytes(&[b"42", b" -7 ", b"abc", b"1.9"]);
        let r = cast_of(Ty::Varchar, Ty::Int, src).unwrap();
        assert_eq!(r.i32s()[0], 42);
        assert_eq!(r.i32s()[1], -7);
        assert!(!r.is_valid(2));
        assert_eq!(r.i32s()[3], 1);

        let src = bytes(&[b"1.25e2", b"nope"]);
        let r = cast_of(Ty::Varchar, Ty::Double, src).unwrap();
        assert_eq!(r.f64s()[0], 125.0);
        assert!(!r.is_valid(1));

        let src = bytes(&[b"12.345"]);
        assert_eq!(cast_of(Ty::Varchar, d2, src).unwrap().i64s(), &[1234]);
    }

    #[test]
    fn cast_boolean_and_string_forms() {
        let src = bools(&[T, F]);
        assert_eq!(cast_of(Ty::Boolean, Ty::Int, src).unwrap().i32s(), &[1, 0]);
        let src = ints(&[0, 5]);
        let r = cast_of(Ty::Int, Ty::Boolean, src).unwrap();
        assert_eq!(tri(&r), vec![F, T]);
        let src = bools(&[T, F]);
        let r = cast_of(Ty::Boolean, Ty::Varchar, src).unwrap();
        assert_eq!(r.bytes().get(0), b"true");
        assert_eq!(r.bytes().get(1), b"false");
        let src = bytes(&[b"TRUE", b"false", b"1", b"zzz"]);
        let r = cast_of(Ty::Varchar, Ty::Boolean, src).unwrap();
        assert_eq!(tri(&r), vec![T, F, T, N]);
    }

    #[test]
    fn cast_date_timestamp_roundtrip() {
        // 1970-01-03 とその前日以前（負の日数）。
        let src = col(Ty::Date, &[Some(Value::I32(2)), Some(Value::I32(-1))]);
        let ts = cast_of(Ty::Date, Ty::Timestamp, src).unwrap();
        assert_eq!(ts.i64s(), &[2 * 86_400_000_000, -86_400_000_000]);
        let back = cast_of(Ty::Timestamp, Ty::Date, ts).unwrap();
        assert_eq!(back.i32s(), &[2, -1]);
        // 端数のある TIMESTAMP は床除算（前日に落ちない）。
        let src = col(Ty::Timestamp, &[Some(Value::I64(-1))]);
        assert_eq!(cast_of(Ty::Timestamp, Ty::Date, src).unwrap().i32s(), &[-1]);
    }

    #[test]
    fn cast_identity_and_unsupported() {
        let r = cast_of(Ty::Int, Ty::Int, ints(&[1, 2])).unwrap();
        assert_eq!(i32s_of(&r), vec![1, 2]);
        // VARCHAR ↔ BLOB は表現が同じなので複製。
        let r = cast_of(Ty::Varchar, Ty::Blob, bytes(&[b"x"])).unwrap();
        assert_eq!(r.ty(), Ty::Blob);
        // 型未定の NULL は何にでもキャストできる（結果は NULL）。
        let src = col(Ty::Null, &[None]);
        let r = cast_of(Ty::Null, Ty::Varchar, src).unwrap();
        assert!(!r.is_valid(0));
        // 日付の文字列化は funcs のフォーマッタで実装済み。
        let r = cast_of(Ty::Date, Ty::Varchar, col(Ty::Date, &[Some(Value::I32(0))])).unwrap();
        assert_eq!(r.bytes().get(0), b"1970-01-01");
        // DATE ↔ TIME は意味が無いので未対応のまま。黙って壊さずエラーにする。
        let e = cast_of(Ty::Date, Ty::Time, col(Ty::Date, &[Some(Value::I32(0))]));
        assert_eq!(code_of(e), Some(Code::InvalidCast));
        let e = cast_of(Ty::Timestamp, Ty::Double, col(Ty::Timestamp, &[Some(Value::I64(0))]));
        assert_eq!(code_of(e), Some(Code::InvalidCast));
    }

    // --- TRY_CAST -------------------------------------------------------------

    fn try_cast_of(from: Ty, to: Ty, v: Vector) -> Result<Vector> {
        let mut p = Program::new();
        let r0 = p.alloc_reg();
        let r1 = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, from.phys(), r0, 0, 0, 0));
        let ci = p.add_cast(from, to);
        p.push(Instr::with_aux(OpCode::TryCast, from.phys(), r1, r0, 0, ci));
        p.result = r1;
        p.result_ty = to;
        let batch = Batch::new(vec![v]);
        Vm::new().eval(&p, &batch)
    }

    #[test]
    fn try_cast_succeeds_like_cast_when_the_conversion_works() {
        let r = try_cast_of(Ty::Int, Ty::BigInt, ints(&[7, -7])).unwrap();
        assert_eq!(r.i64s(), &[7, -7]);
        let r = try_cast_of(Ty::Varchar, Ty::Int, bytes(&[b"123"])).unwrap();
        assert_eq!(i32s_of(&r), vec![123]);
    }

    #[test]
    fn try_cast_turns_row_level_parse_failure_into_null() {
        // 通常の CAST でも「行単位」の変換失敗（数値として読めない文字列）は
        // 元々エラーにせず NULL にする（`kernels::cast` の契約）。TRY_CAST でも
        // 同じ挙動になることを確かめる。
        let r = try_cast_of(Ty::Varchar, Ty::Int, bytes(&[b"abc", b"42"])).unwrap();
        assert!(!r.is_valid(0), "'abc' は整数として読めないので NULL");
        assert_eq!(r.i32s()[1], 42);
    }

    #[test]
    fn try_cast_turns_unsupported_combination_into_null_instead_of_erroring() {
        // 通常の CAST ならエラーになる組み合わせ（`cast_identity_and_unsupported`
        // 参照）。TRY_CAST はエラーを伝播せず、行数ぶんの NULL に落とす。
        let r = try_cast_of(
            Ty::Timestamp,
            Ty::Double,
            col(Ty::Timestamp, &[Some(Value::I64(0)), None]),
        )
        .unwrap();
        assert_eq!(r.len(), 2);
        assert!(!r.is_valid(0));
        assert!(!r.is_valid(1));
        assert_eq!(r.ty(), Ty::Double);

        let e = cast_of(Ty::Timestamp, Ty::Double, col(Ty::Timestamp, &[Some(Value::I64(0))]));
        assert_eq!(
            code_of(e),
            Some(Code::InvalidCast),
            "普通の CAST は同じ組み合わせでエラーのまま"
        );
    }

    // --- VARCHAR ⇄ JSON -------------------------------------------------------

    #[test]
    fn cast_varchar_to_json_validates_and_json_to_varchar_passes_through() {
        // duckdb: CAST('{"a":1}' AS JSON) は成功し、CAST('not json' AS JSON) は
        // Conversion Error になる（TRY_CAST は NULL）。
        let r =
            cast_of(Ty::Varchar, Ty::Json, bytes(&[br#"{"a":1}"#, b"[1,2]", b"\"x\""])).unwrap();
        assert_eq!(r.ty(), Ty::Json);
        assert_eq!(r.bytes().get(0), br#"{"a":1}"#);
        assert_eq!(r.bytes().get(1), b"[1,2]");

        // 通常の CAST は不正な JSON をその場でエラーにする（他の型と違い、
        // 行単位の失敗を NULL に丸めない例外。`kernels::cast_str_to_json` の
        // doc 参照）。
        let e = cast_of(Ty::Varchar, Ty::Json, bytes(&[b"not json"]));
        assert_eq!(code_of(e), Some(Code::InvalidCast));

        // TRY_CAST はその行だけ NULL にし、他の妥当な行は活かす。
        let r = try_cast_of(Ty::Varchar, Ty::Json, bytes(&[br#"{"a":1}"#, b"not json"])).unwrap();
        assert_eq!(r.ty(), Ty::Json);
        assert!(r.is_valid(0));
        assert_eq!(r.bytes().get(0), br#"{"a":1}"#);
        assert!(!r.is_valid(1));

        // JSON → VARCHAR はテキストをそのまま返す（検証済みなので失敗しない）。
        let j = col(Ty::Json, &[Some(Value::Bytes(br#"{"a":1}"#.to_vec())), None]);
        let back = cast_of(Ty::Json, Ty::Varchar, j).unwrap();
        assert_eq!(back.ty(), Ty::Varchar);
        assert_eq!(back.bytes().get(0), br#"{"a":1}"#);
        assert!(!back.is_valid(1));

        // NULL 行は検証をすり抜けて NULL のまま（空文字列は不正な JSON だが
        // NULL 行なので CAST はエラーにならない）。
        let r = cast_of(Ty::Varchar, Ty::Json, col(Ty::Varchar, &[None])).unwrap();
        assert!(!r.is_valid(0));
    }

    #[test]
    fn json_only_casts_with_varchar_and_blob_are_rejected() {
        // BLOB ⇄ JSON、数値 → JSON などは非対応（`to_json` 関数を使うべき、
        // という設計判断。モジュール doc 参照）。
        let e = cast_of(Ty::Blob, Ty::Json, bytes(&[b"{}"]));
        assert_eq!(code_of(e), Some(Code::InvalidCast));
        let e = cast_of(Ty::Json, Ty::Blob, col(Ty::Json, &[Some(Value::Bytes(b"{}".to_vec()))]));
        assert_eq!(code_of(e), Some(Code::InvalidCast));
        let e = cast_of(Ty::Int, Ty::Json, ints(&[1]));
        assert_eq!(code_of(e), Some(Code::InvalidCast));
    }

    // --- フィルタ・複合プログラム -------------------------------------------

    #[test]
    fn eval_filter_returns_original_row_indices() {
        let mut p = Program::new();
        let r0 = p.alloc_reg();
        let r1 = p.alloc_reg();
        let r2 = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, PhysType::I32, r0, 0, 0, 0));
        let ci = p.add_const(Ty::Int, Value::I32(10));
        p.push(Instr::with_aux(OpCode::LoadConst, PhysType::I32, r1, 0, 0, ci));
        p.push(Instr::new(OpCode::Gt, PhysType::I32, r2, r0, r1));
        p.result = r2;
        p.result_ty = Ty::Boolean;

        let mut batch = Batch::new(vec![col(
            Ty::Int,
            &[
                Some(Value::I32(5)),
                Some(Value::I32(20)),
                None,
                Some(Value::I32(30)),
                Some(Value::I32(1)),
                Some(Value::I32(40)),
            ],
        )]);
        // 事前 selection 付き。行 0,3,5 だけを見る。
        batch.sel = Some(vec![0, 3, 5]);
        let mut out = Vec::new();
        let mut vm = Vm::new();
        vm.eval_filter(&p, &batch, &mut out).unwrap();
        assert_eq!(out, vec![3, 5], "元のバッチ行番号を返す");

        // selection なしなら行番号そのもの。NULL は偽。
        batch.sel = None;
        out.clear();
        vm.eval_filter(&p, &batch, &mut out).unwrap();
        assert_eq!(out, vec![1, 3, 5]);
    }

    #[test]
    fn eval_filter_rejects_non_bool() {
        let mut p = Program::new();
        let r0 = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, PhysType::I32, r0, 0, 0, 0));
        p.result = r0;
        let batch = Batch::new(vec![ints(&[1])]);
        let mut out = Vec::new();
        let e = Vm::new().eval_filter(&p, &batch, &mut out);
        assert_eq!(code_of(e), Some(Code::TypeMismatch));
    }

    #[test]
    fn multi_instruction_program_reuses_registers() {
        // (a + 1) * 2 > b AND c IS NOT NULL
        let mut p = Program::new();
        let ra = p.alloc_reg();
        let rb = p.alloc_reg();
        let rc = p.alloc_reg();
        let rk = p.alloc_reg();
        let rt = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, PhysType::I32, ra, 0, 0, 0));
        p.push(Instr::with_aux(OpCode::LoadCol, PhysType::I32, rb, 0, 0, 1));
        p.push(Instr::with_aux(OpCode::LoadCol, PhysType::I32, rc, 0, 0, 2));
        let c1 = p.add_const(Ty::Int, Value::I32(1));
        let c2 = p.add_const(Ty::Int, Value::I32(2));
        p.push(Instr::with_aux(OpCode::LoadConst, PhysType::I32, rk, 0, 0, c1));
        p.push(Instr::new(OpCode::Add, PhysType::I32, ra, ra, rk)); // ra を再利用
        p.push(Instr::with_aux(OpCode::LoadConst, PhysType::I32, rk, 0, 0, c2));
        p.push(Instr::new(OpCode::Mul, PhysType::I32, ra, ra, rk));
        p.push(Instr::new(OpCode::Gt, PhysType::I32, rt, ra, rb));
        p.push(Instr::new(OpCode::IsNotNull, PhysType::I32, rc, rc, rc));
        p.push(Instr::new(OpCode::And, PhysType::Bool, rt, rt, rc));
        p.result = rt;
        p.result_ty = Ty::Boolean;

        let batch = Batch::new(vec![
            ints(&[1, 10, 5, 0]),
            ints(&[5, 5, 100, 0]),
            col(Ty::Int, &[Some(Value::I32(0)), Some(Value::I32(0)), Some(Value::I32(0)), None]),
        ]);
        // (1+1)*2=4 > 5 → F / (10+1)*2=22 > 5 → T / (5+1)*2=12 > 100 → F / c が NULL → F
        let mut vm = Vm::new();
        let r = vm.eval(&p, &batch).unwrap();
        assert_eq!(tri(&r), vec![F, T, F, F]);

        // 同じ Vm を使い回しても結果は変わらない（レジスタの持ち越し無し）。
        let r2 = vm.eval(&p, &batch).unwrap();
        assert_eq!(tri(&r2), vec![F, T, F, F]);
    }

    #[test]
    fn constant_only_program_broadcasts_to_card() {
        let mut p = Program::new();
        let r0 = p.alloc_reg();
        let ci = p.add_const(Ty::Int, Value::I32(7));
        p.push(Instr::with_aux(OpCode::LoadConst, PhysType::I32, r0, 0, 0, ci));
        p.result = r0;
        let batch = Batch::new(vec![ints(&[0, 0, 0])]);
        let r = Vm::new().eval(&p, &batch).unwrap();
        assert_eq!(i32s_of(&r), vec![7, 7, 7]);

        // NULL 定数も同様。
        let mut p = Program::new();
        let r0 = p.alloc_reg();
        let ci = p.add_const(Ty::Int, Value::Null);
        p.push(Instr::with_aux(OpCode::LoadConst, PhysType::I32, r0, 0, 0, ci));
        p.result = r0;
        let r = Vm::new().eval(&p, &batch).unwrap();
        assert_eq!(r.len(), 3);
        assert!(!r.is_valid(0) && !r.is_valid(2));
    }

    #[test]
    fn empty_batch_yields_empty_result() {
        let mut p = Program::new();
        let r0 = p.alloc_reg();
        let r1 = p.alloc_reg();
        let r2 = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, PhysType::I32, r0, 0, 0, 0));
        let ci = p.add_const(Ty::Int, Value::I32(1));
        p.push(Instr::with_aux(OpCode::LoadConst, PhysType::I32, r1, 0, 0, ci));
        p.push(Instr::new(OpCode::Add, PhysType::I32, r2, r0, r1));
        p.result = r2;
        let batch = Batch::new(vec![Vector::new(Ty::Int)]);
        let r = Vm::new().eval(&p, &batch).unwrap();
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn bad_program_reports_internal_error() {
        // 存在しない列。
        let mut p = Program::new();
        let r0 = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, PhysType::I32, r0, 0, 0, 9));
        p.result = r0;
        let batch = Batch::new(vec![ints(&[1])]);
        assert_eq!(code_of(Vm::new().eval(&p, &batch)), Some(Code::Internal));

        // 物理型が命令と食い違う。
        let mut p = Program::new();
        let r0 = p.alloc_reg();
        let r1 = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, PhysType::I32, r0, 0, 0, 0));
        p.push(Instr::new(OpCode::Add, PhysType::I64, r1, r0, r0));
        p.result = r1;
        assert_eq!(code_of(Vm::new().eval(&p, &batch)), Some(Code::TypeMismatch));
    }

    /// `Data` を直接触る経路（Vector の生成）が壊れていないことの確認。
    #[test]
    fn vector_from_data_shape() {
        let v = Vector::from_data(Ty::Int, Data::I32(vec![1, 2]), None);
        assert_eq!(v.len(), 2);
    }
}
