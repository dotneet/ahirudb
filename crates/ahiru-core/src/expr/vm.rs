//! The bytecode VM's execution loop.
//!
//! One instruction = one kernel call. There are no branch instructions, so execution is a simple
//! linear walk of the instruction sequence (see the design decision in mod.rs).
//!
//! The VM has only three responsibilities:
//! - Resolving selection. `LoadCol` gathers, so every kernel afterwards sees only dense vectors
//!   (kernels are not doubled by the presence of selection).
//! - Bringing constants in as length-1 vectors. Stride computation is the kernels' job.
//! - Carrying the logical types. Kernels work only on physical types, so the result's logical
//!   type (DECIMAL scale and the like) is decided and passed by the VM.

use crate::expr::kernels;
use crate::expr::{Instr, OpCode, Program, Reg};
use crate::prelude::*;
use crate::vector::{Batch, PhysType, Ty, Value, Vector};

/// The register file. Reused across a query to keep allocations down.
pub struct Vm {
    regs: Vec<Vector>,
}

impl Vm {
    pub fn new() -> Self {
        Vm { regs: Vec::new() }
    }

    /// Evaluates `p` against `batch` and returns the result vector.
    /// The returned vector's length is `batch.card()`.
    pub fn eval(&mut self, p: &Program, batch: &Batch) -> Result<Vector> {
        // A program whose register/side-table counters saturated while it was
        // being compiled cannot be executed: its `u16` operand fields would
        // alias two different values onto the same slot, which is a wrong
        // answer rather than a slow one. See `Program::overflow`.
        ensure!(!p.overflow, LimitExceeded);
        let n = batch.card();
        while self.regs.len() < p.num_regs as usize {
            self.regs.push(Vector::new(Ty::Null));
        }
        for ins in p.instrs.iter() {
            exec(&mut self.regs, ins, p, batch)?;
        }
        let r = p.result as usize;
        ensure!(r < self.regs.len(), Internal);
        // Taken out, leaving the register empty. The next eval always overwrites it.
        let out = core::mem::replace(&mut self.regs[r], Vector::new(Ty::Null));
        if out.len() == n {
            Ok(out)
        } else if out.len() == 1 {
            // A constant-only expression. Broadcast to the row count before returning.
            Ok(kernels::broadcast(&out, n))
        } else {
            err!(Internal)
        }
    }

    /// For filtering. Pushes only the TRUE rows of a Bool result into `out`.
    /// NULL counts as false, per SQL semantics.
    pub fn eval_filter(&mut self, p: &Program, batch: &Batch, out: &mut Vec<u32>) -> Result<()> {
        let v = self.eval(p, batch)?;
        ensure!(v.data().phys() == PhysType::Bool, TypeMismatch);
        let bits = v.bools();
        // The registers are already gathered, so result element i corresponds to row sel[i].
        // What is pushed is the original batch row number.
        match &batch.sel {
            Some(sel) => {
                // The index is used to look up both the result bit and the selection.
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

/// The default logical type for a physical type. A fallback for when type information is lost.
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

/// The result logical type of a binary operation. The binder is assumed to have aligned both
/// sides, but when they are not (= cannot be unified) it falls back to the physical type's default.
fn binary_ty(op: OpCode, phys: PhysType, a: Ty, b: Ty) -> Ty {
    // DECIMAL multiplication *adds* the operands' scales -- the kernel just multiplies the
    // raw integers, so the product is already scaled by `s1 + s2`. `Ty::unify` cannot say
    // that: it answers "the one type both sides align to", which for two DECIMALs is
    // `max(s1, s2)`, and labelling the result with that scale renders it 10^min(s1,s2) times
    // too large (`to_json`, `printf`, `CAST(... AS VARCHAR)`, the CSV writer).
    //
    // `plan::compile::decimal_arith` has already widened both operands to the *result's*
    // precision, so that precision paired with the summed scale reconstructs exactly the
    // type the compiler planned. The `phys` check below keeps the fallback honest if some
    // other path ever emits a `Mul` over DECIMALs without that widening.
    if op == OpCode::Mul {
        if let (
            Ty::Decimal { precision: p1, scale: s1 },
            Ty::Decimal { precision: p2, scale: s2 },
        ) = (a, b)
        {
            let t = Ty::decimal(p1.max(p2), s1.saturating_add(s2));
            if t.phys() == phys {
                return t;
            }
        }
    }
    // DATE arithmetic is deliberately compiled on the shared I32 lane even
    // though DATE and INTEGER do not unify. Preserve the logical result here
    // so the arithmetic kernel can reject DuckDB's reserved DATE infinity
    // sentinels; DATE - DATE is the one shape whose result is an INTEGER day
    // count and must not receive the DATE-only check.
    if phys == PhysType::I32 && (a == Ty::Date && b != Ty::Date || b == Ty::Date && a != Ty::Date) {
        return Ty::Date;
    }
    match Ty::unify(a, b) {
        Some(t) if t.phys() == phys => t,
        _ => default_ty(phys),
    }
}

/// Turns one element of the constant pool into a length-1 vector. A NULL constant is one row with validity 0.
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
                // Resolving selection here means no kernel afterwards has to know about selection at all.
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
            // Unary Neg does not use b. `a` is passed so an uninitialized register is never read.
            let b = if ins.op == Neg { a } else { reg(regs, ins.b)? };
            kernels::arith(ins.op, binary_ty(ins.op, ins.ty, a.ty(), b.ty()), a, b)?
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
                // The combination itself is unconvertible (`InvalidCast` and so on). Per-row
                // failures are returned as NULL rather than an error by `kernels::cast`, so
                // reaching here means only "that type pair simply cannot be converted". The error
                // is not propagated; it falls to an all-NULL vector.
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
            // The arguments live in a separate table, so the references are collected from the registers before being passed.
            let mut args = Vec::with_capacity(spec.args.len());
            for &r in &spec.args {
                args.push(reg(regs, r)?);
            }
            match spec.lambda {
                // `list_transform`/`list_filter`/`list_reduce`. They cannot be lowered to one
                // vectorized instruction (the length varies per array element and per row), so they
                // go to a dedicated execution path (see the `expr::funcs::call_lambda` docs).
                Some(li) => {
                    let body = match p.lambdas.get(li as usize) {
                        Some(b) => b,
                        None => err!(Internal),
                    };
                    crate::expr::funcs::call_lambda(spec.func, spec.result_ty, &args, body)?
                }
                None => crate::expr::funcs::call(spec.func, spec.result_ty, &args)?,
            }
        }
        TsAddInterval => kernels::ts_add_interval(reg(regs, ins.a)?, reg(regs, ins.b)?)?,
        IntervalAdd => kernels::interval_add(reg(regs, ins.a)?, reg(regs, ins.b)?)?,
        IntervalNeg => kernels::interval_neg(reg(regs, ins.a)?)?,
        IntervalMul => kernels::interval_mul(reg(regs, ins.a)?, reg(regs, ins.b)?)?,
        Like => kernels::like(reg(regs, ins.a)?, reg(regs, ins.b)?)?,
        Concat => {
            let a = reg(regs, ins.a)?;
            let b = reg(regs, ins.b)?;
            kernels::concat(a, b, binary_ty(ins.op, PhysType::Bytes, a.ty(), b.ty()))?
        }
        Select => {
            let c = reg(regs, ins.a)?;
            let t = reg(regs, ins.b)?;
            let e = reg(regs, ins.aux)?;
            kernels::pick(Some(c), t, e, binary_ty(ins.op, ins.ty, t.ty(), e.ty()))?
        }
        Coalesce => {
            let a = reg(regs, ins.a)?;
            let b = reg(regs, ins.b)?;
            kernels::pick(None, a, b, binary_ty(ins.op, ins.ty, a.ty(), b.ty()))?
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

    // --- Construction helpers -----------------------------------------------

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

    /// Evaluates `col0 op col1`.
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

    /// Evaluates `col0 op const` / `const op col0`. `const_first` swaps the order.
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

    /// Extracts a Bool vector as (value, valid) pairs. The value is ignored for NULLs.
    fn tri(v: &Vector) -> Vec<Option<bool>> {
        (0..v.len()).map(|i| if v.is_valid(i) { Some(v.bools().get(i)) } else { None }).collect()
    }

    // --- Arithmetic ---------------------------------------------------------

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
        // 3 - col (the same kernel even with the constant on the left)
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

        // The constant side's stride (I64 / F64).
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

        // MIN / -1 is not representable in two's complement, so it gives NULL rather than panicking.
        let r = eval2(OpCode::Div, PhysType::I32, ints(&[i32::MIN]), ints(&[-1]));
        assert!(!r.is_valid(0));
        let a = col(Ty::BigInt, &[Some(Value::I64(i64::MIN))]);
        let b = col(Ty::BigInt, &[Some(Value::I64(-1))]);
        assert!(!eval2(OpCode::Div, PhysType::I64, a, b).is_valid(0));
        let a = col(Ty::HugeInt, &[Some(Value::I128(i128::MIN))]);
        let b = col(Ty::HugeInt, &[Some(Value::I128(-1))]);
        assert!(!eval2(OpCode::Mod, PhysType::I128, a, b).is_valid(0));

        // Floating point stays IEEE.
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

        // Adding a NULL constant makes every row NULL.
        let a = ints(&[1, 2]);
        let r = eval_const(OpCode::Add, PhysType::I32, a, (Ty::Int, Value::Null), false);
        assert!(!r.is_valid(0) && !r.is_valid(1));
    }

    // --- Comparison ---------------------------------------------------------

    const ALL_CMP: [OpCode; 6] =
        [OpCode::Eq, OpCode::Ne, OpCode::Lt, OpCode::Le, OpCode::Gt, OpCode::Ge];

    /// The expected results of the six comparisons over the three rows (a<b, a=b, a>b).
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
        // Bytes compare lexicographically.
        let mid = || bytes(&[&b"abd"[..], &b"abd"[..], &b"abd"[..]]);
        check_cmp(PhysType::Bytes, bytes(&[&b"abc"[..], &b"abd"[..], &b"abe"[..]]), mid());
        // A common prefix of differing lengths is lexicographic too (the shorter is smaller).
        check_cmp(PhysType::Bytes, bytes(&[&b"ab"[..], &b"abd"[..], &b"abdx"[..]]), mid());
        // For Bool, false < true.
        check_cmp(
            PhysType::Bool,
            bools(&[Some(false), Some(true), Some(true)]),
            bools(&[Some(true), Some(true), Some(false)]),
        );
    }

    #[test]
    /// NaN participates in a total order (DuckDB's semantics): it is greater than
    /// every other value, including `+inf`, and equal to itself. That is also what
    /// `exec::rowkey` uses for join keys, grouping and `ORDER BY`, so the hash-join
    /// and nested-loop paths of the same query cannot disagree.
    fn nan_sorts_above_everything_and_equals_itself() {
        let nan = col(Ty::Double, &[Some(Value::F64(f64::NAN))]);
        let one = col(Ty::Double, &[Some(Value::F64(1.0))]);
        let inf = col(Ty::Double, &[Some(Value::F64(f64::INFINITY))]);
        let bit = |op, a: Vector, b: Vector| eval2(op, PhysType::F64, a, b).bools().get(0);

        assert!(bit(OpCode::Eq, nan.clone(), nan.clone()));
        assert!(!bit(OpCode::Ne, nan.clone(), nan.clone()));
        assert!(!bit(OpCode::Eq, nan.clone(), one.clone()));
        assert!(bit(OpCode::Ne, nan.clone(), one.clone()));
        assert!(!bit(OpCode::Lt, nan.clone(), one.clone()));
        assert!(bit(OpCode::Ge, nan.clone(), one.clone()));
        assert!(bit(OpCode::Gt, nan.clone(), inf.clone()));
        assert!(bit(OpCode::Lt, inf, nan.clone()));
        // -0.0 and 0.0 stay equal, as IEEE says and as the row-key encoding assumes.
        let zero = col(Ty::Double, &[Some(Value::F64(0.0))]);
        let neg_zero = col(Ty::Double, &[Some(Value::F64(-0.0))]);
        assert!(bit(OpCode::Eq, zero.clone(), neg_zero.clone()));
        assert!(!bit(OpCode::Lt, neg_zero, zero));
    }

    /// A program whose counters saturated during compilation is refused rather
    /// than run with two values aliased onto one register.
    #[test]
    fn an_overflowed_program_is_refused() {
        let mut p = Program::new();
        let r = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, PhysType::I32, r, 0, 0, 0));
        p.result = r;
        let batch = Batch::new(vec![ints(&[1, 2])]);
        assert!(Vm::new().eval(&p, &batch).is_ok());

        p.overflow = true;
        assert_eq!(code_of(Vm::new().eval(&p, &batch)), Some(Code::LimitExceeded));
    }

    #[test]
    fn comparison_with_null_is_null() {
        let a = col(Ty::Int, &[Some(Value::I32(1)), None]);
        let b = ints(&[1, 1]);
        for op in ALL_CMP.iter() {
            let r = eval2(*op, PhysType::I32, a.clone(), b.clone());
            assert!(r.is_valid(0));
            assert!(!r.is_valid(1), "a NULL comparison stays NULL");
        }
    }

    // --- Three-valued logic -------------------------------------------------

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
        // NULL AND FALSE = FALSE, NULL OR TRUE = TRUE (the same with stride 0 on the constant side).
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

    // --- NULL predicates and selection --------------------------------------

    #[test]
    fn is_null_and_is_not_null() {
        let c = col(Ty::Int, &[Some(Value::I32(1)), None, Some(Value::I32(3))]);
        let r = unary(OpCode::IsNull, PhysType::I32, c.clone());
        assert_eq!(tri(&r), vec![F, T, F]);
        assert!(!r.has_nulls(), "IsNull's result is never NULL");
        let r = unary(OpCode::IsNotNull, PhysType::I32, c);
        assert_eq!(tri(&r), vec![T, F, T]);

        // It works on a column with no NULLs too.
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
        // When the condition is NULL / FALSE it takes the else side (a length-1 constant).
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

        // NULL input stays NULL.
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

    // --- Casts --------------------------------------------------------------

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
        // A narrowing that does not fit is not an error; just that row becomes NULL.
        let src = col(Ty::BigInt, &[Some(Value::I64(5)), Some(Value::I64(i64::MAX))]);
        let r = cast_of(Ty::BigInt, Ty::Int, src).unwrap();
        assert_eq!(r.i32s()[0], 5);
        assert!(!r.is_valid(1));

        // Logical integer widths are narrower than their shared physical
        // vectors. Out-of-range signed and unsigned values must not survive
        // a cast merely because they fit I32/I64/I128.
        let src = col(
            Ty::BigInt,
            &[Some(Value::I64(127)), Some(Value::I64(128)), Some(Value::I64(-129))],
        );
        let r = cast_of(Ty::BigInt, Ty::TinyInt, src).unwrap();
        assert!(r.is_valid(0));
        assert!(!r.is_valid(1) && !r.is_valid(2));

        let src = col(Ty::BigInt, &[Some(Value::I64(255)), Some(Value::I64(-1))]);
        let r = cast_of(Ty::BigInt, Ty::UTinyInt, src).unwrap();
        assert!(r.is_valid(0));
        assert!(!r.is_valid(1));
    }

    #[test]
    fn cast_between_int_and_float() {
        let r = cast_of(Ty::Int, Ty::Double, ints(&[3, -4])).unwrap();
        assert_eq!(r.f64s(), &[3.0, -4.0]);
        // Floating point -> integer rounds rather than truncates (the same as DuckDB / PostgreSQL).
        let src = col(
            Ty::Double,
            &[Some(Value::F64(3.9)), Some(Value::F64(-3.9)), Some(Value::F64(1e30))],
        );
        let r = cast_of(Ty::Double, Ty::Int, src).unwrap();
        assert_eq!(r.i32s()[0], 4);
        assert_eq!(r.i32s()[1], -4);
        assert!(!r.is_valid(2), "out of range gives NULL");

        // Exactly halfway goes to the even side (banker's rounding). 1.5 -> 2, 4.5 -> 4.
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
        // NaN / inf give NULL too.
        let src = col(Ty::Double, &[Some(Value::F64(f64::NAN)), Some(Value::F64(f64::INFINITY))]);
        let r = cast_of(Ty::Double, Ty::BigInt, src).unwrap();
        assert!(!r.is_valid(0) && !r.is_valid(1));
    }

    #[test]
    fn cast_decimal_rescale() {
        let d2 = Ty::Decimal { precision: 10, scale: 2 };
        let d4 = Ty::Decimal { precision: 12, scale: 4 };
        // 12.34 -> to scale 4 (multiplied by 10^2)
        let src = col(Ty::Decimal { precision: 10, scale: 2 }, &[Some(Value::I64(1234))]);
        assert_eq!(cast_of(d2, d4, src).unwrap().i64s(), &[123_400]);
        // The other direction rounds away from zero (the same as DuckDB).
        // 12.3456 -> 12.35, -12.3456 -> -12.35.
        let src = col(d4, &[Some(Value::I64(123_456)), Some(Value::I64(-123_456))]);
        assert_eq!(cast_of(d4, d2, src).unwrap().i64s(), &[1235, -1235]);
        // Exactly halfway also goes away from zero. 1.235 -> 1.24.
        let d3 = Ty::Decimal { precision: 10, scale: 3 };
        let src = col(d3, &[Some(Value::I64(1235)), Some(Value::I64(-1235))]);
        assert_eq!(cast_of(d3, d2, src).unwrap().i64s(), &[124, -124]);
        // DECIMAL -> DOUBLE divides by the scale.
        let src = col(d2, &[Some(Value::I64(1234))]);
        assert_eq!(cast_of(d2, Ty::Double, src).unwrap().f64s(), &[12.34]);
        // DOUBLE -> DECIMAL multiplies by the scale and rounds. 12.349 -> 12.35.
        let src = col(Ty::Double, &[Some(Value::F64(12.349))]);
        assert_eq!(cast_of(Ty::Double, d2, src).unwrap().i64s(), &[1235]);
        // Integer -> DECIMAL.
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

        // VARCHAR -> numeric. Only unreadable rows become NULL.
        // Fractional strings round away from zero (same as DECIMAL scale-down),
        // so they agree with `CAST(1.5 AS INTEGER)` rather than truncating.
        let src = bytes(&[b"42", b" -7 ", b"abc", b"1.9", b"1.5", b"-1.5"]);
        let r = cast_of(Ty::Varchar, Ty::Int, src).unwrap();
        assert_eq!(r.i32s()[0], 42);
        assert_eq!(r.i32s()[1], -7);
        assert!(!r.is_valid(2));
        assert_eq!(r.i32s()[3], 2);
        assert_eq!(r.i32s()[4], 2);
        assert_eq!(r.i32s()[5], -2);

        let src = bytes(&[b"1.25e2", b"nope"]);
        let r = cast_of(Ty::Varchar, Ty::Double, src).unwrap();
        assert_eq!(r.f64s()[0], 125.0);
        assert!(!r.is_valid(1));

        let src = bytes(&[b"12.345"]);
        assert_eq!(cast_of(Ty::Varchar, d2, src).unwrap().i64s(), &[1235]);
        let src = bytes(&[b"1.25"]);
        let d1 = Ty::Decimal { precision: 10, scale: 1 };
        assert_eq!(cast_of(Ty::Varchar, d1, src).unwrap().i64s(), &[13]);
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
        let src = bytes(&[b"TRUE", b"false", b"yes", b"n", b"1", b"zzz"]);
        let r = cast_of(Ty::Varchar, Ty::Boolean, src).unwrap();
        assert_eq!(tri(&r), vec![T, F, T, F, T, N]);
    }

    #[test]
    fn cast_date_timestamp_roundtrip() {
        // 1970-01-03 and the day before and earlier (negative day counts).
        let src = col(Ty::Date, &[Some(Value::I32(2)), Some(Value::I32(-1))]);
        let ts = cast_of(Ty::Date, Ty::Timestamp, src).unwrap();
        assert_eq!(ts.i64s(), &[2 * 86_400_000_000, -86_400_000_000]);
        let back = cast_of(Ty::Timestamp, Ty::Date, ts).unwrap();
        assert_eq!(back.i32s(), &[2, -1]);
        // A TIMESTAMP with a remainder uses floor division (so it does not fall to the previous day).
        let src = col(Ty::Timestamp, &[Some(Value::I64(-1))]);
        assert_eq!(cast_of(Ty::Timestamp, Ty::Date, src).unwrap().i32s(), &[-1]);
    }

    #[test]
    fn cast_identity_and_unsupported() {
        let r = cast_of(Ty::Int, Ty::Int, ints(&[1, 2])).unwrap();
        assert_eq!(i32s_of(&r), vec![1, 2]);
        // VARCHAR <-> BLOB share a representation, so it is a copy.
        let r = cast_of(Ty::Varchar, Ty::Blob, bytes(&[b"x"])).unwrap();
        assert_eq!(r.ty(), Ty::Blob);
        // A type-undetermined NULL can be cast to anything (the result is NULL).
        let src = col(Ty::Null, &[None]);
        let r = cast_of(Ty::Null, Ty::Varchar, src).unwrap();
        assert!(!r.is_valid(0));
        // Date stringification is implemented by the formatter in funcs.
        let r = cast_of(Ty::Date, Ty::Varchar, col(Ty::Date, &[Some(Value::I32(0))])).unwrap();
        assert_eq!(r.bytes().get(0), b"1970-01-01");
        // DATE <-> TIME is meaningless and stays unsupported. It errors rather than breaking silently.
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
        // Even under an ordinary CAST, a "per-row" conversion failure (a string unreadable as a
        // number) is not an error and becomes NULL (the `kernels::cast` contract). This confirms
        // TRY_CAST behaves the same.
        let r = try_cast_of(Ty::Varchar, Ty::Int, bytes(&[b"abc", b"42"])).unwrap();
        assert!(!r.is_valid(0), "'abc' is unreadable as an integer, so NULL");
        assert_eq!(r.i32s()[1], 42);
    }

    #[test]
    fn try_cast_turns_unsupported_combination_into_null_instead_of_erroring() {
        // A combination that would error under an ordinary CAST (see
        // `cast_identity_and_unsupported`). TRY_CAST does not propagate the error and falls to as many NULLs as there are rows.
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
            "an ordinary CAST still errors on the same combination"
        );
    }

    // --- VARCHAR <-> JSON -----------------------------------------------------

    #[test]
    fn cast_varchar_to_json_validates_and_json_to_varchar_passes_through() {
        // duckdb: CAST('{"a":1}' AS JSON) succeeds while CAST('not json' AS JSON) is a
        // Conversion Error (TRY_CAST gives NULL).
        let r =
            cast_of(Ty::Varchar, Ty::Json, bytes(&[br#"{"a":1}"#, b"[1,2]", b"\"x\""])).unwrap();
        assert_eq!(r.ty(), Ty::Json);
        assert_eq!(r.bytes().get(0), br#"{"a":1}"#);
        assert_eq!(r.bytes().get(1), b"[1,2]");

        // An ordinary CAST errors on invalid JSON on the spot (unlike other types, an exception
        // that does not round a per-row failure to NULL; see the `kernels::cast_str_to_json` docs).
        let e = cast_of(Ty::Varchar, Ty::Json, bytes(&[b"not json"]));
        assert_eq!(code_of(e), Some(Code::InvalidCast));

        // TRY_CAST makes just that row NULL and keeps the other valid rows.
        let r = try_cast_of(Ty::Varchar, Ty::Json, bytes(&[br#"{"a":1}"#, b"not json"])).unwrap();
        assert_eq!(r.ty(), Ty::Json);
        assert!(r.is_valid(0));
        assert_eq!(r.bytes().get(0), br#"{"a":1}"#);
        assert!(!r.is_valid(1));

        // JSON -> VARCHAR returns the text unchanged (already validated, so it cannot fail).
        let j = col(Ty::Json, &[Some(Value::Bytes(br#"{"a":1}"#.to_vec())), None]);
        let back = cast_of(Ty::Json, Ty::Varchar, j).unwrap();
        assert_eq!(back.ty(), Ty::Varchar);
        assert_eq!(back.bytes().get(0), br#"{"a":1}"#);
        assert!(!back.is_valid(1));

        // NULL rows slip past validation and stay NULL (the empty string is invalid JSON, but a
        // NULL row does not make the CAST error).
        let r = cast_of(Ty::Varchar, Ty::Json, col(Ty::Varchar, &[None])).unwrap();
        assert!(!r.is_valid(0));
    }

    #[test]
    fn json_only_casts_with_varchar_and_blob_are_rejected() {
        // BLOB <-> JSON, numeric -> JSON, and so on are unsupported (the design decision is that
        // `to_json` should be used; see the module docs).
        let e = cast_of(Ty::Blob, Ty::Json, bytes(&[b"{}"]));
        assert_eq!(code_of(e), Some(Code::InvalidCast));
        let e = cast_of(Ty::Json, Ty::Blob, col(Ty::Json, &[Some(Value::Bytes(b"{}".to_vec()))]));
        assert_eq!(code_of(e), Some(Code::InvalidCast));
        let e = cast_of(Ty::Int, Ty::Json, ints(&[1]));
        assert_eq!(code_of(e), Some(Code::InvalidCast));
    }

    // --- Filters and composite programs -------------------------------------

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
        // With a prior selection. Only rows 0, 3, and 5 are looked at.
        batch.sel = Some(vec![0, 3, 5]);
        let mut out = Vec::new();
        let mut vm = Vm::new();
        vm.eval_filter(&p, &batch, &mut out).unwrap();
        assert_eq!(out, vec![3, 5], "returns the original batch row numbers");

        // Without selection they are just row numbers. NULL is false.
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
        p.push(Instr::new(OpCode::Add, PhysType::I32, ra, ra, rk)); // reusing ra
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
        // (1+1)*2=4 > 5 -> F / (10+1)*2=22 > 5 -> T / (5+1)*2=12 > 100 -> F / c is NULL -> F
        let mut vm = Vm::new();
        let r = vm.eval(&p, &batch).unwrap();
        assert_eq!(tri(&r), vec![F, T, F, F]);

        // Reusing the same Vm does not change the result (no register carryover).
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

        // The same for a NULL constant.
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
        // A column that does not exist.
        let mut p = Program::new();
        let r0 = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, PhysType::I32, r0, 0, 0, 9));
        p.result = r0;
        let batch = Batch::new(vec![ints(&[1])]);
        assert_eq!(code_of(Vm::new().eval(&p, &batch)), Some(Code::Internal));

        // The physical type disagrees with the instruction.
        let mut p = Program::new();
        let r0 = p.alloc_reg();
        let r1 = p.alloc_reg();
        p.push(Instr::with_aux(OpCode::LoadCol, PhysType::I32, r0, 0, 0, 0));
        p.push(Instr::new(OpCode::Add, PhysType::I64, r1, r0, r0));
        p.result = r1;
        assert_eq!(code_of(Vm::new().eval(&p, &batch)), Some(Code::TypeMismatch));
    }

    /// Confirms the path that touches `Data` directly (building a Vector) is not broken.
    #[test]
    fn vector_from_data_shape() {
        let v = Vector::from_data(Ty::Int, Data::I32(vec![1, 2]), None);
        assert_eq!(v.len(), 2);
    }
}
