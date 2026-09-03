//! AST expressions -> bytecode.
//!
//! This also serves as type checking. Implicit conversion goes through `Ty::unify` alone,
//! with explicit `Cast` instructions inserted where needed. Execution kernels never have
//! to think about type conversion, so the number of kernels does not grow (DESIGN.md §11).

use crate::expr::{funcs, CallSpec, Instr, OpCode, Program, Reg};
use crate::plan::{AggKind, Scope};
use crate::prelude::*;
use crate::rt::hash::eq_ascii_ci;
use crate::sql::ast::{BinaryOp, Expr, ExprArena, ExprId, UnaryOp};
use crate::vector::{Field, Ty, Value};

/// The expression nesting limit. The parser limits it too; this is a second layer of defense.
const MAX_DEPTH: u32 = 64;

/// An instruction to replace a subexpression with an input column.
///
/// Used when compiling expressions on top of an aggregate. In
/// `SELECT a + 1, count(*) ... GROUP BY a + 1`, both `a + 1` and `count(*)` are already
/// output columns of the aggregate operator, so they are swapped for `LoadCol` to avoid re-evaluation.
#[derive(Clone, Copy)]
pub struct Substitution {
    /// The expression to replace.
    pub expr: ExprId,
    /// The input column number to replace it with.
    pub column: usize,
    /// `true` matches structurally (for matching GROUP BY expressions).
    /// `false` matches only the identical node (for matching aggregate calls).
    pub structural: bool,
}

pub struct Compiler<'a> {
    arena: &'a ExprArena,
    /// The input batch's columns. The index doubles as `LoadCol`'s column number.
    scope: &'a Scope,
    /// The values bound to `?` placeholders.
    params: &'a [Value],
    subs: &'a [Substitution],
    prog: Program,
    depth: u32,
}

/// Compiles a single expression.
pub fn compile(arena: &ExprArena, scope: &Scope, params: &[Value], id: ExprId) -> Result<Program> {
    compile_with_subs(arena, scope, params, &[], id)
}

/// Compiles with replacement instructions. Used on top of an aggregate.
pub fn compile_with_subs(
    arena: &ExprArena,
    scope: &Scope,
    params: &[Value],
    subs: &[Substitution],
    id: ExprId,
) -> Result<Program> {
    let mut c = Compiler { arena, scope, params, subs, prog: Program::new(), depth: 0 };
    let (reg, ty) = c.expr(id)?;
    c.prog.result = reg;
    c.prog.result_ty = ty;
    Ok(c.prog)
}

/// Compiles a predicate. An error if the result is not BOOLEAN.
pub fn compile_predicate(
    arena: &ExprArena,
    scope: &Scope,
    params: &[Value],
    id: ExprId,
) -> Result<Program> {
    compile_predicate_with_subs(arena, scope, params, &[], id)
}

pub fn compile_predicate_with_subs(
    arena: &ExprArena,
    scope: &Scope,
    params: &[Value],
    subs: &[Substitution],
    id: ExprId,
) -> Result<Program> {
    let p = compile_with_subs(arena, scope, params, subs, id)?;
    if p.result_ty == Ty::Null {
        // A bare `NULL` (or `?` bound to null) is UNKNOWN, not a type error.
        // `eval_filter` requires a BOOLEAN physical type, so materialize a
        // BOOLEAN NULL constant rather than leaving `Ty::Null` (I32).
        return Ok(boolean_null_program());
    }
    ensure!(p.result_ty == Ty::Boolean, TypeMismatch);
    Ok(p)
}

/// `WHERE NULL` / a leftover `AND NULL` conjunct: one BOOLEAN NULL row.
fn boolean_null_program() -> Program {
    let mut p = Program::new();
    let k = p.add_const(Ty::Boolean, Value::Null);
    let dst = p.alloc_reg();
    p.push(Instr::with_aux(OpCode::LoadConst, Ty::Boolean.phys(), dst, 0, 0, k));
    p.result = dst;
    p.result_ty = Ty::Boolean;
    p
}

/// A program that just returns the input columns unchanged. Used by `SELECT *` and join keys.
pub fn column_program(scope: &Scope, i: usize) -> Result<Program> {
    let ty = match scope.fields().get(i) {
        Some(f) => f.ty,
        None => err!(Internal),
    };
    let mut p = Program::new();
    let dst = p.alloc_reg();
    p.push(Instr::with_aux(OpCode::LoadCol, ty.phys(), dst, 0, 0, i as u16));
    p.result = dst;
    p.result_ty = ty;
    Ok(p)
}

/// Determines the result type of DATE arithmetic. The returned `bool` says whether the operands should be swapped.
///
/// - `DATE + integer` / `integer + DATE` -> DATE (adds days)
/// - `DATE - integer` -> DATE
/// - `DATE - DATE` -> a day count (DuckDB returns INTEGER)
///
/// Arithmetic between TIMESTAMP and an integer is an error in DuckDB too and is not
/// accepted (the unit -- seconds or microseconds -- cannot be decided).
fn date_arith(op: BinaryOp, lt: Ty, rt: Ty) -> Option<(Ty, bool)> {
    use BinaryOp::*;
    let intish = |t: Ty| t.is_integer() || t == Ty::Null;
    match (op, lt, rt) {
        (Add, Ty::Date, r) if intish(r) => Some((Ty::Date, false)),
        // `1 + DATE` swaps so DATE comes first.
        (Add, l, Ty::Date) if intish(l) => Some((Ty::Date, true)),
        (Sub, Ty::Date, r) if intish(r) => Some((Ty::Date, false)),
        (Sub, Ty::Date, Ty::Date) => Some((Ty::BigInt, false)),
        _ => None,
    }
}

/// Determines the type of multiplication and division involving DECIMAL. `None` if it does not apply.
///
/// Addition and subtraction are correct as "align the scales, then add", so the ordinary
/// common-type path suffices, but **multiplication adds the scales** (`1.25 * 2.5 = 3.125`:
/// s=2 and s=3 give s=5). Aligning to a common type before multiplying would return a value
/// off by a doubled scale. The kernel merely multiplies raw integers, so the correctly
/// scaled result type is decided here, and both operands are widened in physical width only, **keeping their scale**.
///
/// Division falls to DOUBLE, as in DuckDB. Left as integer division the scale would
/// subtract, leaving too few digits and giving 0 in most cases.
///
/// A product whose scale would exceed [`crate::vector::types::MAX_DECIMAL_PRECISION`] is
/// rejected with `ValueOutOfRange` rather than silently clamped: clamping the type without
/// rescaling the raw integer made `0.01::DECIMAL(25,20) * 0.01::DECIMAL(25,20)` report
/// `0.01` instead of `0.0001`. DuckDB raises an out-of-range error naming the needed scale
/// there too, and points at the same workarounds -- cast an operand to DOUBLE, or to a
/// DECIMAL with a smaller scale. The result *precision* is still clamped to the maximum,
/// which is also what DuckDB does (`DECIMAL(20,2) * DECIMAL(19,2)` -> `DECIMAL(38,4)`).
fn decimal_arith(op: BinaryOp, lt: Ty, rt: Ty) -> Result<Option<(Ty, Ty, Ty)>> {
    if !matches!(op, BinaryOp::Mul | BinaryOp::Div) {
        return Ok(None);
    }
    // If neither side is DECIMAL, take the ordinary path.
    if !matches!(lt, Ty::Decimal { .. }) && !matches!(rt, Ty::Decimal { .. }) {
        return Ok(None);
    }
    // With floating point mixed in, fall to DOUBLE (as in DuckDB).
    if matches!(lt, Ty::Float | Ty::Double) || matches!(rt, Ty::Float | Ty::Double) {
        return Ok(Some((Ty::Double, Ty::Double, Ty::Double)));
    }
    let (Some((p1, s1)), Some((p2, s2))) = (lt.as_decimal(), rt.as_decimal()) else {
        return Ok(None);
    };
    if op == BinaryOp::Div {
        return Ok(Some((Ty::Double, Ty::Double, Ty::Double)));
    }
    // Multiplication: precision adds, and so does scale.
    let scale = s1.saturating_add(s2);
    ensure!(scale <= crate::vector::types::MAX_DECIMAL_PRECISION, ValueOutOfRange);
    let res = Ty::decimal(p1.saturating_add(p2), scale);
    let Some((rp, _)) = res.as_decimal() else { return Ok(None) };
    // Both operands keep their scale and widen to the result's physical width.
    Ok(Some((Ty::decimal(rp, s1), Ty::decimal(rp, s2), res)))
}

/// Recognizes the shapes DATE/TIMESTAMP +- INTERVAL, INTERVAL +- INTERVAL, and
/// INTERVAL * integer. `None` if none applies (= take the ordinary `Ty::unify` path).
///
/// The reason it is not on `Ty::unify` is the same as for `date_arith`: INTERVAL has no
/// width ordering against any other type, so the very idea of promoting to a common type does not fit.
enum IntervalOp {
    /// `swap`: whether to swap the operands (the `INTERVAL + DATE` shape).
    /// `negate_b`: whether to negate the INTERVAL side first (the `- INTERVAL` shape).
    TsInterval {
        swap: bool,
        negate_b: bool,
    },
    IntervalInterval {
        negate_b: bool,
    },
    /// `swap`: whether the integer is on the left (`3 * INTERVAL`).
    IntervalMul {
        swap: bool,
    },
}

fn interval_arith(op: BinaryOp, lt: Ty, rt: Ty) -> Option<IntervalOp> {
    use BinaryOp::*;
    let temporal = |t: Ty| matches!(t, Ty::Date | Ty::Timestamp);
    match (op, lt, rt) {
        (Add, l, Ty::Interval) if temporal(l) => {
            Some(IntervalOp::TsInterval { swap: false, negate_b: false })
        }
        (Add, Ty::Interval, r) if temporal(r) => {
            Some(IntervalOp::TsInterval { swap: true, negate_b: false })
        }
        (Sub, l, Ty::Interval) if temporal(l) => {
            Some(IntervalOp::TsInterval { swap: false, negate_b: true })
        }
        (Add, Ty::Interval, Ty::Interval) => Some(IntervalOp::IntervalInterval { negate_b: false }),
        (Sub, Ty::Interval, Ty::Interval) => Some(IntervalOp::IntervalInterval { negate_b: true }),
        (Mul, Ty::Interval, r) if r.is_integer() => Some(IntervalOp::IntervalMul { swap: false }),
        (Mul, l, Ty::Interval) if l.is_integer() => Some(IntervalOp::IntervalMul { swap: true }),
        _ => None,
    }
}

/// Bundles two programs with `AND`.
///
/// `WHERE a AND b` can be compiled as one program at the AST stage, but when a predicate is
/// decomposed, part of it pushed down, and the rest bundled back together, already-compiled
/// programs have to be joined.
/// This is the shared routine for merging the two `Program`s `lhs`/`rhs` into one.
///
/// Register, constant, and cast-table numbers are shifted on the `rhs` side only, to sit
/// after `lhs`'s end (`base`/`kbase`/`cbase`), and `rhs.instrs` is appended to `lhs.instrs`.
/// The caller then only needs to append its own operation (`And`/`Coalesce`/...) as one
/// instruction, using the returned `(result register number, rebased rhs result register number)`.
///
/// `lhs.num_regs` is already increased here by `rhs`'s share, so the caller may call
/// `lhs.alloc_reg()` directly.
///
/// Every side table a `Program` carries has to be rebased here, not just
/// `consts`/`casts`: `OpCode::Call`'s `aux` indexes `Program::calls`, a
/// `CallSpec`'s `args` are register numbers, and `CallSpec::lambda` indexes
/// `Program::lambdas`. Missing any of them makes the merged program reference
/// the wrong side table entry — e.g. `WHERE a = 1 AND upper(s) = 'FOO'` used
/// to merge a call-free `lhs` with a `rhs` holding one `CallSpec`, leaving the
/// `Call` instruction pointing into an empty `calls` table (`Internal` at
/// runtime), while the same conjuncts in the opposite order happened to work.
///
/// Every one of those indices is a `u16` (`Instr` is a packed 12-byte record whose
/// operand fields are `u16` by design, DESIGN.md §9), so the rebase is only legal
/// while each shifted table still fits. Past that, adding `base`/`kbase`/... would
/// wrap and alias two distinct registers or side-table entries onto one -- and in
/// `profile.wasm`, where overflow checks are off, it would do so with no diagnostic
/// at all. So the room is checked up front here, once, for every table; when it is
/// not there the merge is abandoned and `lhs` is poisoned through
/// [`Program::overflow`], which makes `expr::vm::Vm::eval` refuse the program with
/// `LimitExceeded` (the same "clean error, never a wrong answer" rule the rest of
/// the engine follows for resource ceilings). The guard belongs here rather than at
/// the call sites because not every caller had one: `and_programs` checked, but
/// `plan::bind::agg::coalesce_programs` did not.
pub(crate) fn merge_program_bodies(lhs: &mut Program, rhs: Program) -> (Reg, Reg) {
    // Checking the five table sums is enough to make every individual shift below
    // safe: each index inside `rhs` is already bounded by the corresponding `rhs`
    // count, so if the sum fits, so does every shifted index.
    let fits = |a: usize, b: usize| a + b <= u16::MAX as usize;
    if rhs.overflow
        || !fits(lhs.num_regs as usize, rhs.num_regs as usize)
        || !fits(lhs.consts.len(), rhs.consts.len())
        || !fits(lhs.casts.len(), rhs.casts.len())
        || !fits(lhs.calls.len(), rhs.calls.len())
        || !fits(lhs.lambdas.len(), rhs.lambdas.len())
    {
        lhs.overflow = true;
        // `rhs` is dropped unmerged. The caller still gets two in-range register
        // numbers to build its combining instruction from, so the program stays
        // structurally valid -- it simply never runs.
        return (lhs.result, lhs.result);
    }
    let base = lhs.num_regs;
    let kbase = lhs.consts.len() as u16;
    let cbase = lhs.casts.len() as u16;
    let fbase = lhs.calls.len() as u16;
    let lbase = lhs.lambdas.len() as u16;

    lhs.consts.extend(rhs.consts.iter().cloned());
    lhs.casts.extend(rhs.casts.iter().copied());
    // Lambda bodies are self-contained programs with their own register and
    // constant space (`Compiler::lambda_call` compiles them in an isolated
    // scope), so they move over as-is; only the index that points at them
    // shifts.
    lhs.lambdas.extend(rhs.lambdas);
    for c in rhs.calls {
        lhs.calls.push(CallSpec {
            func: c.func,
            args: c.args.iter().map(|r| r + base).collect(),
            result_ty: c.result_ty,
            lambda: c.lambda.map(|l| l + lbase),
        });
    }
    for i in &rhs.instrs {
        let mut i2 = *i;
        i2.dst += base;
        // LoadCol / LoadConst do not use a and b, so shifting them would break things.
        match i2.op {
            OpCode::LoadCol => {}
            OpCode::LoadConst => i2.aux += kbase,
            OpCode::Cast | OpCode::TryCast => {
                i2.a += base;
                i2.aux += cbase;
            }
            // Only for Select is aux the register number of the third operand.
            OpCode::Select => {
                i2.a += base;
                i2.b += base;
                i2.aux += base;
            }
            // `Call` takes no register operands (`a`/`b` are unused); its
            // arguments live in `calls[aux].args`, already rebased above.
            OpCode::Call => i2.aux += fbase,
            _ => {
                i2.a += base;
                i2.b += base;
            }
        }
        lhs.instrs.push(i2);
    }
    lhs.num_regs = base + rhs.num_regs;
    (lhs.result, rhs.result + base)
}

pub fn and_programs(mut lhs: Program, rhs: Program) -> Result<Program> {
    // No register-count guard here: `merge_program_bodies` owns that check for every
    // caller now (it used to be duplicated here, and only here, covering `num_regs`
    // alone while the constant/cast/call/lambda tables could still wrap).
    let (a, b) = merge_program_bodies(&mut lhs, rhs);
    let dst = lhs.alloc_reg();
    lhs.push(Instr::new(OpCode::And, crate::vector::PhysType::Bool, dst, a, b));
    lhs.result = dst;
    lhs.result_ty = Ty::Boolean;
    Ok(lhs)
}

/// Converts the result of an already-compiled program to another type.
/// Used to align the left and right column types of a set operation.
pub fn cast_program(mut p: Program, to: Ty) -> Result<Program> {
    if p.result_ty == to {
        return Ok(p);
    }
    let from = p.result_ty;
    let aux = p.add_cast(from, to);
    let src = p.result;
    let dst = p.alloc_reg();
    p.push(Instr::with_aux(OpCode::Cast, from.phys(), dst, src, 0, aux));
    p.result = dst;
    p.result_ty = to;
    Ok(p)
}

/// Whether two expressions are structurally equal. Used to match `GROUP BY` expressions against subexpressions in SELECT.
///
/// Name comparison is case-insensitive (`GROUP BY a` and `SELECT A` count as the same).
/// Constants are judged by value equality.
pub fn expr_eq(arena: &ExprArena, a: ExprId, b: ExprId) -> bool {
    expr_eq_at(arena, a, b, 0)
}

fn expr_eq_at(arena: &ExprArena, a: ExprId, b: ExprId, depth: u32) -> bool {
    if a == b {
        return true;
    }
    if depth >= MAX_DEPTH {
        return false;
    }
    let d = depth + 1;
    let eq = |x: &ExprId, y: &ExprId| expr_eq_at(arena, *x, *y, d);
    let eq_opt = |x: &Option<ExprId>, y: &Option<ExprId>| match (x, y) {
        (None, None) => true,
        (Some(x), Some(y)) => expr_eq_at(arena, *x, *y, d),
        _ => false,
    };
    let ci = |x: &Option<String>, y: &Option<String>| match (x, y) {
        (None, None) => true,
        (Some(x), Some(y)) => crate::rt::hash::eq_ascii_ci(x.as_bytes(), y.as_bytes()),
        _ => false,
    };
    match (arena.get(a), arena.get(b)) {
        (Expr::Literal(x), Expr::Literal(y)) => x == y,
        (Expr::IntervalLiteral(x), Expr::IntervalLiteral(y)) => x == y,
        (Expr::TypedLiteral(x1, t1), Expr::TypedLiteral(x2, t2)) => x1 == x2 && t1 == t2,
        (Expr::Param(x), Expr::Param(y)) => x == y,
        (
            Expr::ColumnRef { qualifier: q1, name: n1 },
            Expr::ColumnRef { qualifier: q2, name: n2 },
        ) => ci(q1, q2) && crate::rt::hash::eq_ascii_ci(n1.as_bytes(), n2.as_bytes()),
        (Expr::Unary { op: o1, arg: a1 }, Expr::Unary { op: o2, arg: a2 }) => {
            o1 == o2 && eq(a1, a2)
        }
        (Expr::Binary { op: o1, lhs: l1, rhs: r1 }, Expr::Binary { op: o2, lhs: l2, rhs: r2 }) => {
            o1 == o2 && eq(l1, l2) && eq(r1, r2)
        }
        (Expr::Cast { arg: a1, ty: t1, try_: y1 }, Expr::Cast { arg: a2, ty: t2, try_: y2 }) => {
            t1 == t2 && y1 == y2 && eq(a1, a2)
        }
        (Expr::IsNull { arg: a1, negated: n1 }, Expr::IsNull { arg: a2, negated: n2 }) => {
            n1 == n2 && eq(a1, a2)
        }
        (
            Expr::Between { arg: a1, low: l1, high: h1, negated: n1 },
            Expr::Between { arg: a2, low: l2, high: h2, negated: n2 },
        ) => n1 == n2 && eq(a1, a2) && eq(l1, l2) && eq(h1, h2),
        (
            Expr::InList { arg: a1, list: l1, negated: n1 },
            Expr::InList { arg: a2, list: l2, negated: n2 },
        ) => {
            n1 == n2
                && eq(a1, a2)
                && l1.len() == l2.len()
                && l1.iter().zip(l2).all(|(x, y)| eq(x, y))
        }
        (
            Expr::Like { arg: a1, pattern: p1, negated: n1, ci: c1 },
            Expr::Like { arg: a2, pattern: p2, negated: n2, ci: c2 },
        ) => n1 == n2 && c1 == c2 && eq(a1, a2) && eq(p1, p2),
        (
            Expr::Case { operand: o1, whens: w1, else_: e1 },
            Expr::Case { operand: o2, whens: w2, else_: e2 },
        ) => {
            eq_opt(o1, o2)
                && eq_opt(e1, e2)
                && w1.len() == w2.len()
                && w1.iter().zip(w2).all(|((c1, v1), (c2, v2))| eq(c1, c2) && eq(v1, v2))
        }
        (
            Expr::Function { name: n1, args: a1, distinct: d1, star: s1, filter: f1 },
            Expr::Function { name: n2, args: a2, distinct: d2, star: s2, filter: f2 },
        ) => {
            d1 == d2
                && s1 == s2
                && eq_opt(f1, f2)
                && crate::rt::hash::eq_ascii_ci(n1.as_bytes(), n2.as_bytes())
                && a1.len() == a2.len()
                && a1.iter().zip(a2).all(|(x, y)| eq(x, y))
        }
        _ => false,
    }
}

/// Whether this function name may take a lambda (the same fixed set as
/// `sql::parser::is_lambda_func`. The parser reads `->` as a lambda only in the argument
/// positions of these names, and after binding the same check dispatches to `Compiler::lambda_call` here).
fn is_lambda_func(name: &str) -> bool {
    eq_ascii_ci(name.as_bytes(), b"list_transform")
        || eq_ascii_ci(name.as_bytes(), b"list_filter")
        || eq_ascii_ci(name.as_bytes(), b"list_reduce")
}

impl<'a> Compiler<'a> {
    fn emit(&mut self, op: OpCode, ty: Ty, a: Reg, b: Reg) -> Reg {
        let dst = self.prog.alloc_reg();
        self.prog.push(Instr::new(op, ty.phys(), dst, a, b));
        dst
    }

    fn konst(&mut self, ty: Ty, v: Value) -> Reg {
        let k = self.prog.add_const(ty, v);
        let dst = self.prog.alloc_reg();
        self.prog.push(Instr::with_aux(OpCode::LoadConst, ty.phys(), dst, 0, 0, k));
        dst
    }

    /// Aligns a register of type `from` to type `to`.
    fn coerce(&mut self, reg: Reg, from: Ty, to: Ty) -> Result<Reg> {
        self.coerce_with(OpCode::Cast, reg, from, to)
    }

    /// For `TRY_CAST`. Builds the same instruction sequence as `coerce` but emits `TryCast`
    /// instead of `Cast`. A combination that cannot be converted becomes all-NULL at runtime
    /// rather than an error (see `expr::vm::exec`). Per-row conversion failures (out of
    /// range, unparsable) already turn just that row into NULL under either instruction
    /// (the contract of `kernels::cast`), so no distinction is needed here.
    fn try_coerce(&mut self, reg: Reg, from: Ty, to: Ty) -> Result<Reg> {
        self.coerce_with(OpCode::TryCast, reg, from, to)
    }

    /// The shared implementation of `coerce`/`try_coerce`. `op` is either `Cast` or `TryCast`.
    fn coerce_with(&mut self, op: OpCode, reg: Reg, from: Ty, to: Ty) -> Result<Reg> {
        if from == to {
            return Ok(reg);
        }
        // A NULL literal has no type. Rather than converting it, a NULL constant of the
        // target type is built afresh. That way the Cast kernel never has to handle Ty::Null.
        if from == Ty::Null {
            return Ok(self.konst(to, Value::Null));
        }
        let aux = self.prog.add_cast(from, to);
        let dst = self.prog.alloc_reg();
        self.prog.push(Instr::with_aux(op, from.phys(), dst, reg, 0, aux));
        Ok(dst)
    }

    /// Emits `lower(x)` as a one-argument call, for `ILIKE`.
    fn lower_reg(&mut self, r: Reg) -> Result<Reg> {
        let (id, _want, res) = crate::expr::funcs::resolve("lower", &[Ty::Varchar])?;
        let aux = self.prog.add_call(id, vec![r], res);
        let dst = self.prog.alloc_reg();
        self.prog.push(Instr::with_aux(OpCode::Call, res.phys(), dst, 0, 0, aux));
        Ok(dst)
    }

    /// Aligns both operands of a binary operation to a common type.
    fn unify_operands(&mut self, lr: Reg, lt: Ty, rr: Reg, rt: Ty) -> Result<(Reg, Reg, Ty)> {
        let t = Ty::unify_or_mismatch(lt, rt)?;
        let l = self.coerce(lr, lt, t)?;
        let r = self.coerce(rr, rt, t)?;
        Ok((l, r, t))
    }

    /// Lowers the shape `interval_arith` recognized into bytecode.
    fn compile_interval_op(
        &mut self,
        kind: IntervalOp,
        lr: Reg,
        lt: Ty,
        rr: Reg,
        rt: Ty,
    ) -> Result<(Reg, Ty)> {
        match kind {
            IntervalOp::TsInterval { swap, negate_b } => {
                let (ts_r, ts_t, iv_r) = if swap { (rr, rt, lr) } else { (lr, lt, rr) };
                // DATE is moved to TIMESTAMP first. DuckDB also returns TIMESTAMP for
                // DATE +- INTERVAL (since an INTERVAL may carry time components).
                let ts_r = self.coerce(ts_r, ts_t, Ty::Timestamp)?;
                let iv_r = if negate_b {
                    self.emit(OpCode::IntervalNeg, Ty::Interval, iv_r, 0)
                } else {
                    iv_r
                };
                let dst = self.prog.alloc_reg();
                self.prog.push(Instr::new(
                    OpCode::TsAddInterval,
                    crate::vector::PhysType::I64,
                    dst,
                    ts_r,
                    iv_r,
                ));
                Ok((dst, Ty::Timestamp))
            }
            IntervalOp::IntervalInterval { negate_b } => {
                let b =
                    if negate_b { self.emit(OpCode::IntervalNeg, Ty::Interval, rr, 0) } else { rr };
                let dst = self.prog.alloc_reg();
                self.prog.push(Instr::new(
                    OpCode::IntervalAdd,
                    crate::vector::PhysType::I128,
                    dst,
                    lr,
                    b,
                ));
                Ok((dst, Ty::Interval))
            }
            IntervalOp::IntervalMul { swap } => {
                let (iv_r, n_r, n_t) = if swap { (rr, lr, lt) } else { (lr, rr, rt) };
                let n_r = self.coerce(n_r, n_t, Ty::BigInt)?;
                let dst = self.prog.alloc_reg();
                self.prog.push(Instr::new(
                    OpCode::IntervalMul,
                    crate::vector::PhysType::I128,
                    dst,
                    iv_r,
                    n_r,
                ));
                Ok((dst, Ty::Interval))
            }
        }
    }

    fn expr(&mut self, id: ExprId) -> Result<(Reg, Ty)> {
        self.depth += 1;
        ensure!(self.depth <= MAX_DEPTH, ExpressionTooDeep);
        let r = self.expr_inner(id);
        self.depth -= 1;
        r
    }

    fn expr_inner(&mut self, id: ExprId) -> Result<(Reg, Ty)> {
        // Replacement comes first. Aggregate results and GROUP BY keys already exist as columns.
        if let Some(r) = self.substitute(id) {
            return r;
        }
        match self.arena.get(id) {
            Expr::Literal(v) => {
                let ty = v.default_ty();
                Ok((self.konst(ty, v.clone()), ty))
            }
            Expr::IntervalLiteral(packed) => {
                Ok((self.konst(Ty::Interval, Value::I128(*packed)), Ty::Interval))
            }
            Expr::TypedLiteral(v, ty) => Ok((self.konst(*ty, v.clone()), *ty)),
            Expr::ColumnRef { qualifier, name } => self.column(qualifier.as_deref(), name),
            Expr::Star { .. } => err!(SyntaxError),
            Expr::Param(i) => {
                let v = match self.params.get(*i as usize) {
                    Some(v) => v.clone(),
                    None => err!(WrongArgCount),
                };
                let ty = v.default_ty();
                Ok((self.konst(ty, v), ty))
            }
            Expr::Unary { op, arg } => self.unary(*op, *arg),
            Expr::Binary { .. } => self.binary_chain(id),
            Expr::Cast { arg, ty, try_ } => {
                let (r, from) = self.expr(*arg)?;
                if *try_ {
                    Ok((self.try_coerce(r, from, *ty)?, *ty))
                } else {
                    Ok((self.coerce(r, from, *ty)?, *ty))
                }
            }
            Expr::IsNull { arg, negated } => {
                let (r, _) = self.expr(*arg)?;
                let op = if *negated { OpCode::IsNotNull } else { OpCode::IsNull };
                // The input's physical type does not matter. The kernel looks only at validity.
                let dst = self.emit(op, Ty::Boolean, r, 0);
                Ok((dst, Ty::Boolean))
            }
            Expr::Between { arg, low, high, negated } => self.between(*arg, *low, *high, *negated),
            Expr::InList { arg, list, negated } => self.in_list(*arg, list.clone(), *negated),
            Expr::Like { arg, pattern, negated, ci } => {
                let (a, at) = self.expr(*arg)?;
                let (p, pt) = self.expr(*pattern)?;
                let mut a = self.coerce(a, at, Ty::Varchar)?;
                let mut p = self.coerce(p, pt, Ty::Varchar)?;
                // ILIKE ignores case. There is no dedicated kernel; `lower()` is applied to
                // both sides and it falls to an ordinary LIKE (inheriting the same
                // ASCII-only limitation as upper/lower).
                if *ci {
                    a = self.lower_reg(a)?;
                    p = self.lower_reg(p)?;
                }
                let dst = self.emit(OpCode::Like, Ty::Varchar, a, p);
                Ok((self.maybe_not(dst, *negated), Ty::Boolean))
            }
            Expr::Case { operand, whens, else_ } => self.case(*operand, whens.clone(), *else_),
            // Window functions and subqueries arrive only after the binder has rewritten them
            // into dedicated nodes. Anything left here was written somewhere it cannot be.
            Expr::Window { .. } => err!(UnsupportedFeature),
            // `QuantifiedComparison` is either desugared away by the binder
            // (`= ANY`/`<> ALL` into `InSubquery`, `<`/`<=`/`>`/`>=` ANY/ALL
            // into a `MIN`/`MAX`-based `CASE` expression, see
            // `plan::bind::build_quantified_comparison`) or rejected at bind
            // time (correlated subquery, or the unsupported `= ALL`/`<> ANY`
            // combination). What's left here is either a binder bug or a
            // position the binder doesn't scan (e.g. inside a `Lambda` body).
            Expr::ScalarSubquery(_)
            | Expr::Exists { .. }
            | Expr::InSubquery { .. }
            | Expr::QuantifiedComparison { .. } => {
                err!(UnsupportedFeature)
            }
            // `UNNEST` likewise arrives only after the binder rewrites it into
            // `Node::Unnest` + `Substitution`. Anything left here was written somewhere it
            // cannot be (`plan::bind::collect_unnests` detects and rejects that earlier, so
            // in practice this is a net for catching binder bugs).
            Expr::Unnest(_) => err!(UnsupportedFeature),
            // The parser produces this only as the second argument of
            // `list_transform`/`list_filter`/`list_reduce` (see `sql::parser::Parser::call`).
            // Anywhere else (= a bug or a parser oversight) is rejected here.
            Expr::Lambda { .. } => err!(UnsupportedFeature),
            Expr::Function { name, args, distinct, star, filter } => {
                // Aggregate functions arrive only after the binder replaces them. Anything
                // left here was written where aggregation is impossible (WHERE and the like), or is a nested aggregate.
                if AggKind::from_name(name).is_some() {
                    err!(NotAggregate);
                }
                // FILTER is meaningless for a scalar function.
                ensure!(!*distinct && !*star && filter.is_none(), UnsupportedFeature);
                if is_lambda_func(name) {
                    return self.lambda_call(name, args);
                }
                self.scalar_call(name, args)
            }
        }
    }

    /// The value of an argument that is written as an integer literal, `None` otherwise.
    /// The parser keeps a negated literal as `-<literal>` rather than folding it, so that
    /// shape is unwrapped here too (`round(x, -2)`).
    fn const_int(&self, id: ExprId) -> Option<i64> {
        match self.arena.get(id) {
            Expr::Literal(v) => v.as_i64(),
            Expr::Unary { op: UnaryOp::Neg, arg } => match self.arena.get(*arg) {
                Expr::Literal(v) => v.as_i64().and_then(i64::checked_neg),
                _ => None,
            },
            _ => None,
        }
    }

    /// A scalar function call. Type checking and argument conversion are finished here, so
    /// only already-converted vectors are passed at runtime.
    fn scalar_call(&mut self, name: &str, args: &[ExprId]) -> Result<(Reg, Ty)> {
        // `pi()` takes no arguments, and the runtime's row loop derives its row count from the
        // arguments (`expr::funcs::strides`), so a nullary function has nowhere to get one.
        // It is folded to a literal here instead.
        if eq_ascii_ci(name.as_bytes(), b"pi") {
            ensure!(args.is_empty(), WrongArgCount);
            return Ok((self.konst(Ty::Double, Value::F64(core::f64::consts::PI)), Ty::Double));
        }
        let mut regs = Vec::with_capacity(args.len());
        let mut tys = Vec::with_capacity(args.len());
        for a in args {
            let (r, t) = self.expr(*a)?;
            regs.push(r);
            tys.push(t);
        }
        // `typeof` is decided entirely by the argument's static type, which is known right here,
        // so it folds to a literal too. Going through `resolve`/`call` would additionally make
        // `typeof(NULL_column)` come out NULL instead of `'NULL'`, because the row loop
        // propagates a NULL argument to the result.
        if eq_ascii_ci(name.as_bytes(), b"typeof") {
            ensure!(tys.len() == 1, WrongArgCount);
            let v = Value::Bytes(tys[0].name().as_bytes().to_vec());
            return Ok((self.konst(Ty::Varchar, v), Ty::Varchar));
        }
        // A few signatures' *result type* depends on an argument's value, not just its type
        // (`round(<decimal>, d)`'s result scale is `min(s, max(d, 0))`, as in DuckDB), so
        // literal integer arguments are passed along.
        let consts: Vec<Option<i64>> = args.iter().map(|a| self.const_int(*a)).collect();
        let (id, want, res) = crate::expr::funcs::resolve_const(name, &tys, &consts)?;
        ensure!(want.len() == regs.len(), WrongArgCount);
        for i in 0..regs.len() {
            regs[i] = self.coerce(regs[i], tys[i], want[i])?;
        }
        let aux = self.prog.add_call(id, regs, res);
        let dst = self.prog.alloc_reg();
        self.prog.push(Instr::with_aux(OpCode::Call, res.phys(), dst, 0, 0, aux));
        Ok((dst, res))
    }

    /// `list_transform(list, x -> expr)` / `list_filter(list, x -> expr)` /
    /// `list_reduce(list, (acc, x) -> expr [, initial])`.
    ///
    /// This does not ride `scalar_call`'s general path (compiling each argument with
    /// `self.expr()` into an outer-scope register and bundling them into one `Call`
    /// instruction): this bytecode VM assumes vectorized execution, and "evaluate an
    /// expression per array element" is variable-length per row and does not fit. So only the
    /// lambda body is compiled as **a separate small `Program`** embedded in `Program::lambdas`.
    /// At runtime `expr::funcs::call_lambda` repeats `Batch::new` + `Vm::eval` once per array
    /// element (the same idea as `ddl`/`dml` building a one-row batch and running it through
    /// `Vm::eval`; see also the list_transform/list_filter/list_reduce section docs at the
    /// top of the `expr::funcs` module).
    ///
    /// **Known limitation**: a lambda body can reference only its own parameters. It cannot
    /// reference columns of the enclosing SQL scope (a form mixing in an outer column, such
    /// as `list_transform(tags, x -> x || suffix_col)`, is unsupported and gives
    /// `ColumnNotFound`), because the body is always compiled in an isolated `Scope`
    /// containing only the parameters.
    ///
    /// The parameters' type is always `Ty::Json` (the same as `list_extract`'s result; array
    /// elements are all represented as dynamically typed JSON values). In this engine
    /// `Ty::Json` does not `Ty::unify` with any other type (see the `vector::types` docs), so
    /// doing arithmetic or comparison on a parameter in the body requires an explicit
    /// conversion through VARCHAR, as in `CAST(CAST(x AS VARCHAR) AS INTEGER)` (the same as
    /// the existing limitation on `list_extract`'s result, not a lambda-specific
    /// constraint).
    fn lambda_call(&mut self, name: &str, args: &[ExprId]) -> Result<(Reg, Ty)> {
        let is_reduce = eq_ascii_ci(name.as_bytes(), b"list_reduce");
        let func = if eq_ascii_ci(name.as_bytes(), b"list_transform") {
            funcs::F_LIST_TRANSFORM
        } else if eq_ascii_ci(name.as_bytes(), b"list_filter") {
            funcs::F_LIST_FILTER
        } else if is_reduce {
            funcs::F_LIST_REDUCE
        } else {
            err!(FunctionNotFound)
        };
        ensure!(args.len() == 2 || (is_reduce && args.len() == 3), WrongArgCount);

        // The first argument (the list) is compiled in the outer scope as usual.
        let (list_reg, list_ty) = self.expr(args[0])?;
        ensure!(matches!(list_ty, Ty::Json | Ty::Null), TypeMismatch);
        let list_reg =
            if list_ty == Ty::Null { self.konst(Ty::Json, Value::Null) } else { list_reg };

        let (params, body) = match self.arena.get(args[1]) {
            Expr::Lambda { params, body } => (params.clone(), *body),
            // The parser reads the second argument of `list_transform` and friends only as
            // lambda syntax (`x -> expr` / `(a, b) -> expr`), so any other expression (a
            // column reference, say) is a syntax error.
            _ => err!(SyntaxError),
        };
        let want_params = if is_reduce { 2 } else { 1 };
        ensure!(params.len() == want_params, WrongArgCount);

        // `list_reduce`'s third argument (the initial value). When omitted, the list's first
        // element is used (see `expr::funcs::call_list_reduce`). Like the first argument it
        // must be Ty::Json (convert explicitly with `to_json(...)` or similar).
        let mut call_args = vec![list_reg];
        if args.len() == 3 {
            let (init_reg, init_ty) = self.expr(args[2])?;
            ensure!(matches!(init_ty, Ty::Json | Ty::Null), TypeMismatch);
            let init_reg =
                if init_ty == Ty::Null { self.konst(Ty::Json, Value::Null) } else { init_reg };
            call_args.push(init_reg);
        }

        // The body is compiled in an isolated scope containing only the parameters
        // (see "Known limitation" in this method's docs).
        let fields: Vec<Field> =
            params.iter().map(|p| Field::new(p.clone(), Ty::Json, true)).collect();
        let param_scope = Scope::from_fields(fields);
        let body_prog = compile_with_subs(self.arena, &param_scope, self.params, &[], body)?;
        if func == funcs::F_LIST_FILTER {
            ensure!(matches!(body_prog.result_ty, Ty::Boolean | Ty::Null), TypeMismatch);
        }

        let result_ty = Ty::Json;
        let li = self.prog.add_lambda(body_prog);
        let aux = self.prog.add_lambda_call(func, call_args, result_ty, li);
        let dst = self.prog.alloc_reg();
        self.prog.push(Instr::with_aux(OpCode::Call, result_ty.phys(), dst, 0, 0, aux));
        Ok((dst, result_ty))
    }

    fn column(&mut self, qualifier: Option<&str>, name: &str) -> Result<(Reg, Ty)> {
        let i = self.scope.resolve(qualifier, name)?;
        self.load_col(i)
    }

    fn load_col(&mut self, i: usize) -> Result<(Reg, Ty)> {
        let ty = match self.scope.fields().get(i) {
            Some(f) => f.ty,
            None => err!(Internal),
        };
        ensure!(i <= u16::MAX as usize, LimitExceeded);
        let dst = self.prog.alloc_reg();
        self.prog.push(Instr::with_aux(OpCode::LoadCol, ty.phys(), dst, 0, 0, i as u16));
        Ok((dst, ty))
    }

    /// If an expression matches a replacement instruction, it becomes just a read of that input column.
    ///
    /// **Later-added instructions win** (the scan runs in reverse). Both "a scalar subquery's
    /// column" and "a post-aggregation group column" can be registered for the same
    /// expression, and above the aggregate the latter is correct.
    fn substitute(&mut self, id: ExprId) -> Option<Result<(Reg, Ty)>> {
        for s in self.subs.iter().rev() {
            let hit = if s.structural { expr_eq(self.arena, s.expr, id) } else { s.expr == id };
            if hit {
                return Some(self.load_col(s.column));
            }
        }
        None
    }

    fn unary(&mut self, op: UnaryOp, arg: ExprId) -> Result<(Reg, Ty)> {
        let (r, t) = self.expr(arg)?;
        match op {
            UnaryOp::Neg => {
                if t == Ty::Interval {
                    return Ok((self.emit(OpCode::IntervalNeg, Ty::Interval, r, 0), Ty::Interval));
                }
                ensure!(t.is_numeric() || t == Ty::Null, TypeMismatch);
                Ok((self.emit(OpCode::Neg, t, r, 0), t))
            }
            UnaryOp::Not => {
                ensure!(matches!(t, Ty::Boolean | Ty::Null), TypeMismatch);
                let r = self.coerce(r, t, Ty::Boolean)?;
                Ok((self.emit(OpCode::Not, Ty::Boolean, r, 0), Ty::Boolean))
            }
        }
    }

    /// Whether [`Self::substitute`] would replace this node, without emitting anything.
    fn is_substituted(&self, id: ExprId) -> bool {
        self.subs.iter().any(|s| {
            if s.structural {
                expr_eq(self.arena, s.expr, id)
            } else {
                s.expr == id
            }
        })
    }

    /// Compiles a left-deep chain of binary operators without recursing per link.
    ///
    /// `1+1+...+1`, `'a'||'a'||...` and `p AND q AND ...` all parse into `Binary` nodes
    /// nested through their **left** operand only. Recursing into `lhs` would spend one
    /// stack frame and one unit of `MAX_DEPTH` per term, rejecting a perfectly flat
    /// expression of a few dozen terms as "expression nesting too deep". The spine is
    /// descended in a loop and then compiled bottom-up, which visits the operands in
    /// exactly the same order as the recursive version and so emits identical bytecode.
    ///
    /// The descent stops at a node the substitution list replaces (an inner node of the
    /// spine can itself be a GROUP BY expression that already exists as a column); that
    /// node is then compiled by `expr`, which applies the substitution as usual.
    fn binary_chain(&mut self, id: ExprId) -> Result<(Reg, Ty)> {
        let mut spine: Vec<(BinaryOp, ExprId)> = Vec::new();
        let mut cur = id;
        while let Expr::Binary { op, lhs, rhs } = self.arena.get(cur) {
            if cur != id && self.is_substituted(cur) {
                break;
            }
            spine.push((*op, *rhs));
            cur = *lhs;
        }
        let mut acc = self.expr(cur)?;
        for &(op, rhs) in spine.iter().rev() {
            acc = self.binary_rhs(op, acc, rhs)?;
        }
        Ok(acc)
    }

    /// Compiles one binary operator, with its left operand already compiled.
    fn binary_rhs(&mut self, op: BinaryOp, lhs: (Reg, Ty), rhs: ExprId) -> Result<(Reg, Ty)> {
        let (lr, lt) = lhs;
        let (rr, rt) = self.expr(rhs)?;

        if op.is_logical() {
            let l = self.coerce(lr, lt, Ty::Boolean)?;
            let r = self.coerce(rr, rt, Ty::Boolean)?;
            let code = if op == BinaryOp::And { OpCode::And } else { OpCode::Or };
            return Ok((self.emit(code, Ty::Boolean, l, r), Ty::Boolean));
        }

        if op == BinaryOp::Concat {
            // `||` between two JSON operands is *list* concatenation, not text
            // concatenation (`duckdb -c "select [1,2] || [3]"` -> `[1, 2, 3]`).
            // A list has no physical type of its own here — it is a `Ty::Json`
            // value (`docs/DESIGN.md` §5/§8) — so this is the only place the
            // distinction can be made, and it has to be made on the static
            // type. Everything else (including JSON on just one side, e.g.
            // `json_col || 'x'`) keeps the VARCHAR behavior, which is what
            // `'a' || 1` -> `a1` relies on.
            //
            // `Ty::Null` counts as a JSON operand as long as the other side is
            // JSON, so `[1] || NULL` stays a JSON NULL rather than becoming a
            // VARCHAR one; `coerce` turns the untyped NULL into a JSON NULL
            // constant. The result is NULL either way (`duckdb -c "select [1]
            // || NULL::INTEGER[]"` -> NULL), only its type differs.
            //
            // Whether the JSON values are actually *arrays* can only be known
            // at run time; a non-array operand raises `TypeMismatch` there
            // (`funcs::json::list_concat_build`).
            let json_side = |t: Ty| matches!(t, Ty::Json | Ty::Null);
            if json_side(lt) && json_side(rt) && (lt == Ty::Json || rt == Ty::Json) {
                let l = self.coerce(lr, lt, Ty::Json)?;
                let r = self.coerce(rr, rt, Ty::Json)?;
                let aux = self.prog.add_call(funcs::F_LIST_CONCAT_OP, vec![l, r], Ty::Json);
                let dst = self.prog.alloc_reg();
                self.prog.push(Instr::with_aux(OpCode::Call, Ty::Json.phys(), dst, 0, 0, aux));
                return Ok((dst, Ty::Json));
            }
            let l = self.coerce(lr, lt, Ty::Varchar)?;
            let r = self.coerce(rr, rt, Ty::Varchar)?;
            return Ok((self.emit(OpCode::Concat, Ty::Varchar, l, r), Ty::Varchar));
        }

        // Arithmetic between DATE and an integer is treated as days (the same as DuckDB).
        // `unify(Date, Int)` is None, so it does not ride the common-type path.
        if let Some((res, swap)) = date_arith(op, lt, rt) {
            let code = if op == BinaryOp::Add { OpCode::Add } else { OpCode::Sub };
            // DATE - DATE must widen both day counts before subtraction. The
            // result is a BIGINT, and subtracting the finite DATE endpoints
            // can exceed the I32 lane even though each input fits in it.
            if res == Ty::BigInt {
                let l = self.coerce(lr, lt, Ty::BigInt)?;
                let r = self.coerce(rr, rt, Ty::BigInt)?;
                let dst = self.prog.alloc_reg();
                self.prog.push(Instr::new(code, crate::vector::PhysType::I64, dst, l, r));
                return Ok((dst, Ty::BigInt));
            }
            // DATE ± integer. The integer may be BIGINT (CSV inference) or NULL;
            // both are coerced to I32 days so the kernel sees matching physical types.
            let (date_r, date_t, int_r, int_t) =
                if swap { (rr, rt, lr, lt) } else { (lr, lt, rr, rt) };
            let date_r = self.coerce(date_r, date_t, Ty::Date)?;
            let int_r = self.coerce(int_r, int_t, Ty::Int)?;
            let dst = self.prog.alloc_reg();
            self.prog.push(Instr::new(code, crate::vector::PhysType::I32, dst, date_r, int_r));
            return Ok((dst, res));
        }

        // DECIMAL multiplication and division change the scale, so they do not ride the common-type path.
        if let Some((lcast, rcast, res)) = decimal_arith(op, lt, rt)? {
            let l = self.coerce(lr, lt, lcast)?;
            let r = self.coerce(rr, rt, rcast)?;
            let code = if op == BinaryOp::Mul { OpCode::Mul } else { OpCode::Div };
            return Ok((self.emit(code, res, l, r), res));
        }

        // DATE/TIMESTAMP +- INTERVAL, INTERVAL +- INTERVAL, INTERVAL * integer.
        if let Some(kind) = interval_arith(op, lt, rt) {
            return self.compile_interval_op(kind, lr, lt, rr, rt);
        }

        let (l, r, t) = self.unify_operands(lr, lt, rr, rt)?;

        if op.is_comparison() {
            // JSON has no ordering: comparison is byte equality, so a difference in key order
            // or whitespace alone can make two equal documents unequal (no normalizing
            // comparison as in DuckDB -- a known v1 limitation). Ordering those bytes would
            // read as a value order it is not, so only equality is allowed.
            //
            // INTERVAL *does* order: `kernels::compare` routes it to `cmp_interval`, which
            // flattens months/days/microseconds through `exec::rowkey::interval_key` with
            // DuckDB's fixed 1 month = 30 days and 1 day = 24 hours. That is the same key
            // ORDER BY, DISTINCT, GROUP BY and equi-joins already use, so allowing `<`/`<=`/
            // `>`/`>=` here keeps the operators consistent with the rest of the engine
            // instead of contradicting it.
            if t == Ty::Json {
                ensure!(matches!(op, BinaryOp::Eq | BinaryOp::Ne), TypeMismatch);
            }
            let code = match op {
                BinaryOp::Eq => OpCode::Eq,
                BinaryOp::Ne => OpCode::Ne,
                BinaryOp::Lt => OpCode::Lt,
                BinaryOp::Le => OpCode::Le,
                BinaryOp::Gt => OpCode::Gt,
                _ => OpCode::Ge,
            };
            // The instruction is emitted for the comparison's input type; the output is always BOOLEAN.
            let dst = self.prog.alloc_reg();
            self.prog.push(Instr::new(code, t.phys(), dst, l, r));
            return Ok((dst, Ty::Boolean));
        }

        ensure!(t.is_numeric() || t == Ty::Null, TypeMismatch);
        let code = match op {
            BinaryOp::Add => OpCode::Add,
            BinaryOp::Sub => OpCode::Sub,
            BinaryOp::Mul => OpCode::Mul,
            BinaryOp::Div => OpCode::Div,
            BinaryOp::Mod => OpCode::Mod,
            _ => err!(Internal),
        };
        Ok((self.emit(code, t, l, r), t))
    }

    fn between(
        &mut self,
        arg: ExprId,
        low: ExprId,
        high: ExprId,
        negated: bool,
    ) -> Result<(Reg, Ty)> {
        // BETWEEN expands to (a >= lo) AND (a <= hi). There is no dedicated kernel.
        let (ar, at) = self.expr(arg)?;
        let (lr, lt) = self.expr(low)?;
        let (hr, ht) = self.expr(high)?;

        let (a1, l, t1) = self.unify_operands(ar, at, lr, lt)?;
        // JSON has no ordering (`<`/`>` are TypeMismatch); BETWEEN must not silently fall
        // through to a physical byte compare. INTERVAL orders through `cmp_interval`, the same
        // as `<`/`>` above.
        ensure!(t1 != Ty::Json, TypeMismatch);
        let ge = self.prog.alloc_reg();
        self.prog.push(Instr::new(OpCode::Ge, t1.phys(), ge, a1, l));

        let (a2, h, t2) = self.unify_operands(ar, at, hr, ht)?;
        ensure!(t2 != Ty::Json, TypeMismatch);
        let le = self.prog.alloc_reg();
        self.prog.push(Instr::new(OpCode::Le, t2.phys(), le, a2, h));

        let both = self.emit(OpCode::And, Ty::Boolean, ge, le);
        Ok((self.maybe_not(both, negated), Ty::Boolean))
    }

    fn in_list(&mut self, arg: ExprId, list: Vec<ExprId>, negated: bool) -> Result<(Reg, Ty)> {
        ensure!(!list.is_empty(), SyntaxError);
        // IN expands to a chain of Eq joined by OR. Without a dedicated set instruction it is
        // linear in the number of elements. Acceptable for v1.
        let (ar, at) = self.expr(arg)?;
        let mut acc: Option<Reg> = None;
        for item in list {
            let (ir, it) = self.expr(item)?;
            let (a, i, t) = self.unify_operands(ar, at, ir, it)?;
            let eq = self.prog.alloc_reg();
            self.prog.push(Instr::new(OpCode::Eq, t.phys(), eq, a, i));
            acc = Some(match acc {
                None => eq,
                Some(prev) => self.emit(OpCode::Or, Ty::Boolean, prev, eq),
            });
        }
        let r = match acc {
            Some(r) => r,
            None => err!(Internal),
        };
        Ok((self.maybe_not(r, negated), Ty::Boolean))
    }

    fn case(
        &mut self,
        operand: Option<ExprId>,
        whens: Vec<(ExprId, ExprId)>,
        else_: Option<ExprId>,
    ) -> Result<(Reg, Ty)> {
        ensure!(!whens.is_empty(), SyntaxError);

        // Conditions and values are evaluated first, then the result type is decided and Select instructions are stacked from the back.
        let operand = match operand {
            Some(o) => Some(self.expr(o)?),
            None => None,
        };

        let mut conds = Vec::with_capacity(whens.len());
        let mut vals = Vec::with_capacity(whens.len());
        let mut result_ty = Ty::Null;
        for (c, v) in whens {
            let cond = match operand {
                // `CASE x WHEN a ...` is read as `x = a`.
                Some((or, ot)) => {
                    let (cr, ct) = self.expr(c)?;
                    let (l, r, t) = self.unify_operands(or, ot, cr, ct)?;
                    let dst = self.prog.alloc_reg();
                    self.prog.push(Instr::new(OpCode::Eq, t.phys(), dst, l, r));
                    dst
                }
                None => {
                    let (cr, ct) = self.expr(c)?;
                    self.coerce(cr, ct, Ty::Boolean)?
                }
            };
            let (vr, vt) = self.expr(v)?;
            result_ty = Ty::unify_or_mismatch(result_ty, vt)?;
            conds.push(cond);
            vals.push((vr, vt));
        }

        let else_reg = match else_ {
            Some(e) => {
                let (er, et) = self.expr(e)?;
                result_ty = Ty::unify_or_mismatch(result_ty, et)?;
                Some((er, et))
            }
            None => None,
        };

        let mut acc = match else_reg {
            Some((r, t)) => self.coerce(r, t, result_ty)?,
            None => self.konst(result_ty, Value::Null),
        };
        // Stacking from the last WHEN forward nests the conditions in priority order.
        for (cond, (vr, vt)) in conds.into_iter().zip(vals).rev() {
            let v = self.coerce(vr, vt, result_ty)?;
            let dst = self.prog.alloc_reg();
            self.prog.push(Instr::with_aux(OpCode::Select, result_ty.phys(), dst, cond, v, acc));
            acc = dst;
        }
        Ok((acc, result_ty))
    }

    fn maybe_not(&mut self, r: Reg, negated: bool) -> Reg {
        if negated {
            self.emit(OpCode::Not, Ty::Boolean, r, 0)
        } else {
            r
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::code_of;
    use crate::sql::ast::{Expr, UnaryOp};
    use crate::vector::Field;

    fn cols() -> Scope {
        Scope::from_fields(vec![
            Field::new("id", Ty::Int, false),
            Field::new("big", Ty::BigInt, true),
            Field::new("name", Ty::Varchar, true),
        ])
    }

    /// Builds an `id <op> <literal>` expression.
    fn bin(a: &mut ExprArena, op: BinaryOp, col: &str, v: Value) -> ExprId {
        let l = a.push(Expr::ColumnRef { qualifier: None, name: col.into() });
        let r = a.push(Expr::Literal(v));
        a.push(Expr::Binary { op, lhs: l, rhs: r })
    }

    #[test]
    fn comparison_yields_boolean() {
        let mut a = ExprArena::new();
        let id = bin(&mut a, BinaryOp::Gt, "id", Value::I32(5));
        let p = compile(&a, &cols(), &[], id).unwrap();
        assert_eq!(p.result_ty, Ty::Boolean);
        assert!(p.instrs.iter().any(|i| i.op == OpCode::Gt));
    }

    #[test]
    fn widening_cast_is_inserted() {
        // Comparing a BIGINT column with an INTEGER literal aligns to BIGINT.
        let mut a = ExprArena::new();
        let id = bin(&mut a, BinaryOp::Lt, "big", Value::I32(5));
        let p = compile(&a, &cols(), &[], id).unwrap();
        assert_eq!(p.casts.len(), 1);
        assert_eq!(p.casts[0].from, Ty::Int);
        assert_eq!(p.casts[0].to, Ty::BigInt);
    }

    #[test]
    fn null_literal_is_retyped_not_cast() {
        // NULL is rebuilt as a constant of the target type rather than via Cast.
        let mut a = ExprArena::new();
        let id = bin(&mut a, BinaryOp::Eq, "big", Value::Null);
        let p = compile(&a, &cols(), &[], id).unwrap();
        assert!(p.casts.is_empty());
        assert!(p.consts.iter().any(|(t, v)| *t == Ty::BigInt && v.is_null()));
    }

    #[test]
    fn between_expands_to_two_comparisons() {
        let mut a = ExprArena::new();
        let col = a.push(Expr::ColumnRef { qualifier: None, name: "id".into() });
        let lo = a.push(Expr::Literal(Value::I32(1)));
        let hi = a.push(Expr::Literal(Value::I32(9)));
        let id = a.push(Expr::Between { arg: col, low: lo, high: hi, negated: false });
        let p = compile(&a, &cols(), &[], id).unwrap();
        assert!(p.instrs.iter().any(|i| i.op == OpCode::Ge));
        assert!(p.instrs.iter().any(|i| i.op == OpCode::Le));
        assert!(p.instrs.iter().any(|i| i.op == OpCode::And));
    }

    #[test]
    fn between_accepts_interval_and_rejects_json() {
        // BETWEEN expands to `>= AND <=`, so it follows the operators: INTERVAL orders
        // (`interval_compares_by_order_not_just_equality`), JSON does not.
        let mut a = ExprArena::new();
        let col = a.push(Expr::ColumnRef { qualifier: None, name: "iv".into() });
        let lo = a.push(Expr::IntervalLiteral(0));
        let hi = a.push(Expr::IntervalLiteral(1));
        let id = a.push(Expr::Between { arg: col, low: lo, high: hi, negated: false });
        let scope = Scope::from_fields(vec![Field::new("iv", Ty::Interval, true)]);
        assert_eq!(compile(&a, &scope, &[], id).unwrap().result_ty, Ty::Boolean);

        let mut a = ExprArena::new();
        let col = a.push(Expr::ColumnRef { qualifier: None, name: "j".into() });
        let lo_s = a.push(Expr::Literal(Value::Bytes(b"1".to_vec())));
        let lo = a.push(Expr::Cast { arg: lo_s, ty: Ty::Json, try_: false });
        let hi_s = a.push(Expr::Literal(Value::Bytes(b"9".to_vec())));
        let hi = a.push(Expr::Cast { arg: hi_s, ty: Ty::Json, try_: false });
        let id = a.push(Expr::Between { arg: col, low: lo, high: hi, negated: false });
        let scope = Scope::from_fields(vec![Field::new("j", Ty::Json, true)]);
        assert_eq!(code_of(compile(&a, &scope, &[], id)), Some(crate::error::Code::TypeMismatch));
    }

    #[test]
    fn in_list_expands_to_or_chain() {
        let mut a = ExprArena::new();
        let col = a.push(Expr::ColumnRef { qualifier: None, name: "id".into() });
        let list = vec![
            a.push(Expr::Literal(Value::I32(1))),
            a.push(Expr::Literal(Value::I32(2))),
            a.push(Expr::Literal(Value::I32(3))),
        ];
        let id = a.push(Expr::InList { arg: col, list, negated: false });
        let p = compile(&a, &cols(), &[], id).unwrap();
        assert_eq!(p.instrs.iter().filter(|i| i.op == OpCode::Eq).count(), 3);
        assert_eq!(p.instrs.iter().filter(|i| i.op == OpCode::Or).count(), 2);
    }

    #[test]
    fn not_in_wraps_with_not() {
        let mut a = ExprArena::new();
        let col = a.push(Expr::ColumnRef { qualifier: None, name: "id".into() });
        let list = vec![a.push(Expr::Literal(Value::I32(1)))];
        let id = a.push(Expr::InList { arg: col, list, negated: true });
        let p = compile(&a, &cols(), &[], id).unwrap();
        assert!(p.instrs.iter().any(|i| i.op == OpCode::Not));
    }

    #[test]
    fn case_becomes_nested_selects() {
        let mut a = ExprArena::new();
        let c1 = bin(&mut a, BinaryOp::Gt, "id", Value::I32(5));
        let v1 = a.push(Expr::Literal(Value::I32(1)));
        let c2 = bin(&mut a, BinaryOp::Gt, "id", Value::I32(2));
        let v2 = a.push(Expr::Literal(Value::I32(2)));
        let e = a.push(Expr::Literal(Value::I32(0)));
        let id =
            a.push(Expr::Case { operand: None, whens: vec![(c1, v1), (c2, v2)], else_: Some(e) });
        let p = compile(&a, &cols(), &[], id).unwrap();
        assert_eq!(p.instrs.iter().filter(|i| i.op == OpCode::Select).count(), 2);
        assert_eq!(p.result_ty, Ty::Int);
    }

    #[test]
    fn unknown_column_is_reported() {
        let mut a = ExprArena::new();
        let id = a.push(Expr::ColumnRef { qualifier: None, name: "nope".into() });
        assert_eq!(code_of(compile(&a, &cols(), &[], id)), Some(Code::ColumnNotFound));
    }

    #[test]
    fn incomparable_types_are_rejected() {
        let mut a = ExprArena::new();
        let id = bin(&mut a, BinaryOp::Add, "name", Value::I32(1));
        assert_eq!(code_of(compile(&a, &cols(), &[], id)), Some(Code::TypeMismatch));
    }

    #[test]
    fn predicate_must_be_boolean() {
        let mut a = ExprArena::new();
        let id = a.push(Expr::ColumnRef { qualifier: None, name: "id".into() });
        assert_eq!(code_of(compile_predicate(&a, &cols(), &[], id)), Some(Code::TypeMismatch));
    }

    #[test]
    fn bare_null_predicate_is_boolean_unknown() {
        let mut a = ExprArena::new();
        let id = a.push(Expr::Literal(Value::Null));
        let p = compile_predicate(&a, &cols(), &[], id).unwrap();
        assert_eq!(p.result_ty, Ty::Boolean);
    }

    #[test]
    fn not_null_is_boolean_unknown() {
        let mut a = ExprArena::new();
        let n = a.push(Expr::Literal(Value::Null));
        let id = a.push(Expr::Unary { op: UnaryOp::Not, arg: n });
        let p = compile(&a, &cols(), &[], id).unwrap();
        assert_eq!(p.result_ty, Ty::Boolean);
    }

    #[test]
    fn deep_nesting_is_rejected_without_overflowing_the_stack() {
        let mut a = ExprArena::new();
        let mut id = a.push(Expr::Literal(Value::I32(1)));
        // Nested through the *right* operand: `1+(1+(1+...))` really is 500 levels deep.
        for _ in 0..500 {
            let l = a.push(Expr::Literal(Value::I32(1)));
            id = a.push(Expr::Binary { op: BinaryOp::Add, lhs: l, rhs: id });
        }
        assert_eq!(code_of(compile(&a, &cols(), &[], id)), Some(Code::ExpressionTooDeep));
    }

    #[test]
    fn a_flat_left_deep_chain_compiles_however_long_it_is() {
        let mut a = ExprArena::new();
        let mut id = a.push(Expr::Literal(Value::I32(1)));
        // `1+1+...+1` is what a flat 500-term sum parses into: left-associative, so the
        // tree grows down the left operand. It is not nested, so the nesting limit must
        // not reject it (`Compiler::binary_chain` walks the spine iteratively).
        for _ in 0..500 {
            let r = a.push(Expr::Literal(Value::I32(1)));
            id = a.push(Expr::Binary { op: BinaryOp::Add, lhs: id, rhs: r });
        }
        assert!(compile(&a, &cols(), &[], id).is_ok());
    }

    // --- ILIKE / TRY_CAST -------------------------------------------------------

    #[test]
    fn ilike_lowers_both_sides_before_the_like_kernel() {
        let mut a = ExprArena::new();
        let arg = a.push(Expr::ColumnRef { qualifier: None, name: "name".into() });
        let pattern = a.push(Expr::Literal(Value::Bytes(b"A%".to_vec())));
        let id = a.push(Expr::Like { arg, pattern, negated: false, ci: true });
        let p = compile(&a, &cols(), &[], id).unwrap();
        assert_eq!(p.result_ty, Ty::Boolean);
        // Two `lower()` calls (both sides), then one Like.
        assert_eq!(p.calls.len(), 2);
        assert!(p.instrs.iter().any(|i| i.op == OpCode::Like));
        // An ordinary LIKE, unlike ILIKE, does not call lower().
        let id2 = a.push(Expr::Like { arg, pattern, negated: false, ci: false });
        let p2 = compile(&a, &cols(), &[], id2).unwrap();
        assert!(p2.calls.is_empty());
    }

    #[test]
    fn try_cast_emits_try_cast_opcode_not_cast() {
        let mut a = ExprArena::new();
        let arg = a.push(Expr::ColumnRef { qualifier: None, name: "name".into() });
        let id = a.push(Expr::Cast { arg, ty: Ty::Int, try_: true });
        let p = compile(&a, &cols(), &[], id).unwrap();
        assert!(p.instrs.iter().any(|i| i.op == OpCode::TryCast));
        assert!(!p.instrs.iter().any(|i| i.op == OpCode::Cast));

        let id2 = a.push(Expr::Cast { arg, ty: Ty::Int, try_: false });
        let p2 = compile(&a, &cols(), &[], id2).unwrap();
        assert!(p2.instrs.iter().any(|i| i.op == OpCode::Cast));
        assert!(!p2.instrs.iter().any(|i| i.op == OpCode::TryCast));
    }

    #[test]
    fn filter_on_a_non_aggregate_function_is_rejected() {
        // FILTER is meaningless on a scalar function call. It is rejected on the same path as
        // arriving here before aggregate replacement (= written where aggregation is impossible).
        let mut a = ExprArena::new();
        let arg = a.push(Expr::ColumnRef { qualifier: None, name: "id".into() });
        let cond = a.push(Expr::Literal(Value::Bool(true)));
        let id = a.push(Expr::Function {
            name: "abs".into(),
            args: vec![arg],
            distinct: false,
            star: false,
            filter: Some(cond),
        });
        assert_eq!(code_of(compile(&a, &cols(), &[], id)), Some(Code::UnsupportedFeature));
    }

    // --- INTERVAL -------------------------------------------------------------

    fn date_col() -> Scope {
        Scope::from_fields(vec![
            Field::new("d", Ty::Date, false),
            Field::new("ts", Ty::Timestamp, false),
        ])
    }

    fn interval_lit(a: &mut ExprArena, months: i32, days: i32, micros: i64) -> ExprId {
        a.push(Expr::IntervalLiteral(crate::vector::pack_interval(months, days, micros)))
    }

    #[test]
    fn date_plus_interval_promotes_to_timestamp() {
        let mut a = ExprArena::new();
        let d = a.push(Expr::ColumnRef { qualifier: None, name: "d".into() });
        let iv = interval_lit(&mut a, 0, 3, 0);
        let id = a.push(Expr::Binary { op: BinaryOp::Add, lhs: d, rhs: iv });
        let p = compile(&a, &date_col(), &[], id).unwrap();
        assert_eq!(p.result_ty, Ty::Timestamp);
        assert!(p.instrs.iter().any(|i| i.op == OpCode::TsAddInterval));
        // An implicit DATE -> TIMESTAMP cast is interposed.
        assert!(p.casts.iter().any(|c| c.from == Ty::Date && c.to == Ty::Timestamp));
    }

    #[test]
    fn timestamp_minus_interval_negates_then_adds() {
        let mut a = ExprArena::new();
        let ts = a.push(Expr::ColumnRef { qualifier: None, name: "ts".into() });
        let iv = interval_lit(&mut a, 1, 0, 0);
        let id = a.push(Expr::Binary { op: BinaryOp::Sub, lhs: ts, rhs: iv });
        let p = compile(&a, &date_col(), &[], id).unwrap();
        assert_eq!(p.result_ty, Ty::Timestamp);
        assert!(p.instrs.iter().any(|i| i.op == OpCode::IntervalNeg));
        assert!(p.instrs.iter().any(|i| i.op == OpCode::TsAddInterval));
    }

    #[test]
    fn interval_plus_date_is_swapped() {
        let mut a = ExprArena::new();
        let d = a.push(Expr::ColumnRef { qualifier: None, name: "d".into() });
        let iv = interval_lit(&mut a, 0, 1, 0);
        let id = a.push(Expr::Binary { op: BinaryOp::Add, lhs: iv, rhs: d });
        let p = compile(&a, &date_col(), &[], id).unwrap();
        assert_eq!(p.result_ty, Ty::Timestamp);
        assert!(p.instrs.iter().any(|i| i.op == OpCode::TsAddInterval));
    }

    #[test]
    fn interval_add_and_mul_compile() {
        let mut a = ExprArena::new();
        let i1 = interval_lit(&mut a, 1, 0, 0);
        let i2 = interval_lit(&mut a, 0, 3, 0);
        let add = a.push(Expr::Binary { op: BinaryOp::Add, lhs: i1, rhs: i2 });
        let p = compile(&a, &cols(), &[], add).unwrap();
        assert_eq!(p.result_ty, Ty::Interval);
        assert!(p.instrs.iter().any(|i| i.op == OpCode::IntervalAdd));

        let n = a.push(Expr::Literal(Value::I32(2)));
        let mul = a.push(Expr::Binary { op: BinaryOp::Mul, lhs: i1, rhs: n });
        let p = compile(&a, &cols(), &[], mul).unwrap();
        assert_eq!(p.result_ty, Ty::Interval);
        assert!(p.instrs.iter().any(|i| i.op == OpCode::IntervalMul));

        // `n * INTERVAL` takes the same path (with the operands swapped).
        let mul2 = a.push(Expr::Binary { op: BinaryOp::Mul, lhs: n, rhs: i1 });
        let p = compile(&a, &cols(), &[], mul2).unwrap();
        assert!(p.instrs.iter().any(|i| i.op == OpCode::IntervalMul));
    }

    #[test]
    fn interval_compares_by_order_not_just_equality() {
        // `kernels::compare` routes INTERVAL to `cmp_interval`, which flattens
        // months/days/microseconds through `exec::rowkey::interval_key` for *every* mask, so
        // ordering is as well-defined here as equality and as ORDER BY on an interval column.
        // Ordering used to be rejected even though that kernel already existed.
        let mut a = ExprArena::new();
        let i1 = interval_lit(&mut a, 1, 0, 0);
        let i2 = interval_lit(&mut a, 0, 30, 0);
        for op in [BinaryOp::Lt, BinaryOp::Le, BinaryOp::Gt, BinaryOp::Ge, BinaryOp::Eq, BinaryOp::Ne]
        {
            let e = a.push(Expr::Binary { op, lhs: i1, rhs: i2 });
            let p = compile(&a, &cols(), &[], e).unwrap();
            assert_eq!(p.result_ty, Ty::Boolean, "{op:?} should compile to a BOOLEAN");
        }
    }

    #[test]
    fn json_ordering_comparison_is_rejected_but_equality_is_not() {
        // JSON rejects ordering comparison for the same reason as INTERVAL.
        // JSON does not `Ty::unify` with any other type (see the module docs), so the
        // comparison operand must be explicitly CAST to JSON as well.
        let scope = Scope::from_fields(vec![Field::new("doc", Ty::Json, true)]);
        let mut a = ExprArena::new();
        let doc = a.push(Expr::ColumnRef { qualifier: None, name: "doc".into() });
        let lit_str = a.push(Expr::Literal(Value::Bytes(b"{}".to_vec())));
        let lit = a.push(Expr::Cast { arg: lit_str, ty: Ty::Json, try_: false });
        let lt = a.push(Expr::Binary { op: BinaryOp::Lt, lhs: doc, rhs: lit });
        assert_eq!(code_of(compile(&a, &scope, &[], lt)), Some(Code::TypeMismatch));

        let eq = a.push(Expr::Binary { op: BinaryOp::Eq, lhs: doc, rhs: lit });
        let p = compile(&a, &scope, &[], eq).unwrap();
        assert_eq!(p.result_ty, Ty::Boolean);
    }

    #[test]
    fn cast_to_json_is_generic_cast_opcode_no_special_compile_path() {
        // A CAST to JSON merely takes the same path as any other type (`Expr::Cast` ->
        // the `Cast`/`TryCast` opcode); no special branch is needed in compile.rs
        // (the validation lives in `expr::kernels::cast`).
        let scope = Scope::from_fields(vec![Field::new("s", Ty::Varchar, true)]);
        let mut a = ExprArena::new();
        let s = a.push(Expr::ColumnRef { qualifier: None, name: "s".into() });
        let cast = a.push(Expr::Cast { arg: s, ty: Ty::Json, try_: false });
        let p = compile(&a, &scope, &[], cast).unwrap();
        assert_eq!(p.result_ty, Ty::Json);
        assert!(p.instrs.iter().any(|i| i.op == OpCode::Cast));
    }

    #[test]
    fn unary_neg_on_interval_uses_dedicated_kernel() {
        let mut a = ExprArena::new();
        let iv = interval_lit(&mut a, 1, 2, 3);
        let id = a.push(Expr::Unary { op: UnaryOp::Neg, arg: iv });
        let p = compile(&a, &cols(), &[], id).unwrap();
        assert_eq!(p.result_ty, Ty::Interval);
        assert!(p.instrs.iter().any(|i| i.op == OpCode::IntervalNeg));
    }

    // --- Lambdas: list_transform / list_filter / list_reduce ------------------

    fn json_lit(a: &mut ExprArena, text: &str) -> ExprId {
        let s = a.push(Expr::Literal(Value::Bytes(text.as_bytes().to_vec())));
        a.push(Expr::Cast { arg: s, ty: Ty::Json, try_: false })
    }

    fn func_call(a: &mut ExprArena, name: &str, args: Vec<ExprId>) -> ExprId {
        a.push(Expr::Function {
            name: name.into(),
            args,
            distinct: false,
            star: false,
            filter: None,
        })
    }

    #[test]
    fn list_transform_compiles_to_a_lambda_call_instruction() {
        let mut a = ExprArena::new();
        let list = json_lit(&mut a, "[1,2,3]");
        let x = a.push(Expr::ColumnRef { qualifier: None, name: "x".into() });
        let lambda = a.push(Expr::Lambda { params: vec!["x".into()], body: x });
        let id = func_call(&mut a, "list_transform", vec![list, lambda]);
        let p = compile(&a, &cols(), &[], id).unwrap();
        assert_eq!(p.result_ty, Ty::Json);
        let call = p.instrs.iter().find(|i| i.op == OpCode::Call).unwrap();
        let spec = &p.calls[call.aux as usize];
        assert_eq!(spec.func, crate::expr::funcs::F_LIST_TRANSFORM);
        assert!(spec.lambda.is_some());
        assert_eq!(p.lambdas.len(), 1);
    }

    #[test]
    fn list_filter_requires_a_boolean_lambda_body() {
        let mut a = ExprArena::new();
        let list = json_lit(&mut a, "[1,2,3]");
        // The body is `x` (Ty::Json) itself, so it is not BOOLEAN.
        let x = a.push(Expr::ColumnRef { qualifier: None, name: "x".into() });
        let lambda = a.push(Expr::Lambda { params: vec!["x".into()], body: x });
        let id = func_call(&mut a, "list_filter", vec![list, lambda]);
        assert_eq!(code_of(compile(&a, &cols(), &[], id)), Some(Code::TypeMismatch));
    }

    #[test]
    fn list_filter_accepts_a_boolean_lambda_body() {
        let mut a = ExprArena::new();
        let list = json_lit(&mut a, "[1,2,3]");
        let x = a.push(Expr::ColumnRef { qualifier: None, name: "x".into() });
        let one = json_lit(&mut a, "1");
        let pred = a.push(Expr::Binary { op: BinaryOp::Eq, lhs: x, rhs: one });
        let lambda = a.push(Expr::Lambda { params: vec!["x".into()], body: pred });
        let id = func_call(&mut a, "list_filter", vec![list, lambda]);
        let p = compile(&a, &cols(), &[], id).unwrap();
        assert_eq!(p.result_ty, Ty::Json);
    }

    #[test]
    fn lambda_body_cannot_see_outer_scope_columns() {
        // Known limitation: a lambda body can reference only its own parameters (see the
        // `Compiler::lambda_call` docs). `id` is a column of the outer scope and not a
        // parameter, so it gives `ColumnNotFound`.
        let mut a = ExprArena::new();
        let list = json_lit(&mut a, "[1,2,3]");
        let x = a.push(Expr::ColumnRef { qualifier: None, name: "x".into() });
        let outer = a.push(Expr::ColumnRef { qualifier: None, name: "id".into() });
        let body = a.push(Expr::Binary { op: BinaryOp::Add, lhs: x, rhs: outer });
        let lambda = a.push(Expr::Lambda { params: vec!["x".into()], body });
        let id = func_call(&mut a, "list_transform", vec![list, lambda]);
        assert_eq!(code_of(compile(&a, &cols(), &[], id)), Some(Code::ColumnNotFound));
    }

    #[test]
    fn list_transform_rejects_wrong_lambda_param_count() {
        let mut a = ExprArena::new();
        let list = json_lit(&mut a, "[1,2,3]");
        let x = a.push(Expr::ColumnRef { qualifier: None, name: "x".into() });
        // list_transform accepts only a one-parameter lambda.
        let lambda = a.push(Expr::Lambda { params: vec!["x".into(), "y".into()], body: x });
        let id = func_call(&mut a, "list_transform", vec![list, lambda]);
        assert_eq!(code_of(compile(&a, &cols(), &[], id)), Some(Code::WrongArgCount));
    }

    #[test]
    fn list_reduce_needs_two_lambda_params() {
        let mut a = ExprArena::new();
        let list = json_lit(&mut a, "[1,2,3]");
        let x = a.push(Expr::ColumnRef { qualifier: None, name: "x".into() });
        let lambda = a.push(Expr::Lambda { params: vec!["x".into()], body: x });
        let id = func_call(&mut a, "list_reduce", vec![list, lambda]);
        assert_eq!(code_of(compile(&a, &cols(), &[], id)), Some(Code::WrongArgCount));
    }

    #[test]
    fn list_reduce_accepts_two_params_and_an_optional_initial_value() {
        let mut a = ExprArena::new();
        let list = json_lit(&mut a, "[1,2,3]");
        let acc = a.push(Expr::ColumnRef { qualifier: None, name: "acc".into() });
        let lambda = a.push(Expr::Lambda { params: vec!["acc".into(), "x".into()], body: acc });
        let id = func_call(&mut a, "list_reduce", vec![list, lambda]);
        let p = compile(&a, &cols(), &[], id).unwrap();
        assert_eq!(p.result_ty, Ty::Json);

        // A third argument (the initial value) is accepted too.
        let mut a2 = ExprArena::new();
        let list2 = json_lit(&mut a2, "[1,2,3]");
        let acc2 = a2.push(Expr::ColumnRef { qualifier: None, name: "acc".into() });
        let lambda2 = a2.push(Expr::Lambda { params: vec!["acc".into(), "x".into()], body: acc2 });
        let init = json_lit(&mut a2, "0");
        let id2 = func_call(&mut a2, "list_reduce", vec![list2, lambda2, init]);
        let p2 = compile(&a2, &cols(), &[], id2).unwrap();
        assert_eq!(p2.result_ty, Ty::Json);
    }

    #[test]
    fn lambda_as_a_bare_expression_is_unsupported() {
        // The parser only ever builds a lambda as the second argument of list_transform and
        // friends, but `plan::compile` also rejects one in an ordinary expression position, just in case.
        let mut a = ExprArena::new();
        let x = a.push(Expr::ColumnRef { qualifier: None, name: "x".into() });
        let id = a.push(Expr::Lambda { params: vec!["x".into()], body: x });
        assert_eq!(code_of(compile(&a, &cols(), &[], id)), Some(Code::UnsupportedFeature));
    }

    /// `upper(name) = '<lit>'`.
    fn upper_eq(a: &mut ExprArena, lit: &str) -> ExprId {
        let col = a.push(Expr::ColumnRef { qualifier: None, name: "name".into() });
        let up = func_call(a, "upper", vec![col]);
        let s = a.push(Expr::Literal(Value::Bytes(lit.as_bytes().to_vec())));
        a.push(Expr::Binary { op: BinaryOp::Eq, lhs: up, rhs: s })
    }

    /// Merging a call-free `lhs` with a `rhs` that has one must not leave the
    /// `Call` instruction pointing into `lhs`'s (empty) call table. This is the
    /// shape predicate pushdown produces for `WHERE id = 1 AND upper(name) =
    /// 'X'`: the equality is compiled first (and separately consumed into a
    /// scan pruner), then merged with the residual conjunct.
    #[test]
    fn and_programs_rebases_the_call_table_of_the_right_hand_side() {
        let mut a = ExprArena::new();
        let lhs_id = bin(&mut a, BinaryOp::Eq, "id", Value::I32(1));
        let rhs_id = upper_eq(&mut a, "NAME_1");
        let lhs = compile(&a, &cols(), &[], lhs_id).unwrap();
        let rhs = compile(&a, &cols(), &[], rhs_id).unwrap();
        assert!(lhs.calls.is_empty());
        assert_eq!(rhs.calls.len(), 1);
        let upper = rhs.calls[0].func;

        let p = and_programs(lhs, rhs).unwrap();
        assert_eq!(p.calls.len(), 1);
        for i in p.instrs.iter().filter(|i| i.op == OpCode::Call) {
            let spec = p.calls.get(i.aux as usize).expect("call index out of range");
            assert_eq!(spec.func, upper);
            for &r in &spec.args {
                assert!(r < p.num_regs, "argument register {r} out of range");
            }
        }
    }

    /// With calls on both sides the indices must not collide either — before
    /// the rebase existed this silently ran the left-hand function twice
    /// (`WHERE lower(name) = 'a' AND upper(name) = 'A'` returned no rows at
    /// all rather than erroring).
    #[test]
    fn and_programs_keeps_both_sides_call_tables_distinct() {
        let mut a = ExprArena::new();
        let col = a.push(Expr::ColumnRef { qualifier: None, name: "name".into() });
        let low = func_call(&mut a, "lower", vec![col]);
        let lit = a.push(Expr::Literal(Value::Bytes(b"name_1".to_vec())));
        let lhs_id = a.push(Expr::Binary { op: BinaryOp::Eq, lhs: low, rhs: lit });
        let rhs_id = upper_eq(&mut a, "NAME_1");
        let lhs = compile(&a, &cols(), &[], lhs_id).unwrap();
        let rhs = compile(&a, &cols(), &[], rhs_id).unwrap();
        let (lower, upper) = (lhs.calls[0].func, rhs.calls[0].func);
        assert_ne!(lower, upper);

        let p = and_programs(lhs, rhs).unwrap();
        assert_eq!(p.calls.len(), 2);
        let called: Vec<u16> = p
            .instrs
            .iter()
            .filter(|i| i.op == OpCode::Call)
            .map(|i| p.calls[i.aux as usize].func)
            .collect();
        assert_eq!(called, vec![lower, upper]);
    }

    /// `CallSpec::lambda` indexes `Program::lambdas`, so that table has to be
    /// merged and the index rebased too.
    #[test]
    fn and_programs_rebases_the_lambda_table() {
        let mut a = ExprArena::new();
        let lhs_id = bin(&mut a, BinaryOp::Eq, "id", Value::I32(1));
        let list = json_lit(&mut a, "[1,2,3]");
        let x = a.push(Expr::ColumnRef { qualifier: None, name: "x".into() });
        let one = json_lit(&mut a, "1");
        let pred = a.push(Expr::Binary { op: BinaryOp::Eq, lhs: x, rhs: one });
        let lambda = a.push(Expr::Lambda { params: vec!["x".into()], body: pred });
        let filtered = func_call(&mut a, "list_filter", vec![list, lambda]);
        let len = func_call(&mut a, "json_array_length", vec![filtered]);
        let n = a.push(Expr::Literal(Value::I64(1)));
        let rhs_id = a.push(Expr::Binary { op: BinaryOp::Eq, lhs: len, rhs: n });

        let lhs = compile(&a, &cols(), &[], lhs_id).unwrap();
        let rhs = compile(&a, &cols(), &[], rhs_id).unwrap();
        assert!(lhs.lambdas.is_empty());
        assert_eq!(rhs.lambdas.len(), 1);

        let p = and_programs(lhs, rhs).unwrap();
        assert_eq!(p.lambdas.len(), 1);
        for spec in &p.calls {
            if let Some(l) = spec.lambda {
                assert!((l as usize) < p.lambdas.len(), "lambda index out of range");
            }
        }
    }

    /// A `Program` with `n` registers, one instruction, and `k` constants. Registers are
    /// counted, not materialized, so this stays cheap at the `u16` ceiling.
    fn sized_program(n: u16, k: usize) -> Program {
        let mut p = Program::new();
        p.num_regs = n;
        p.result = n.saturating_sub(1);
        p.result_ty = Ty::Boolean;
        for i in 0..k {
            p.consts.push((Ty::BigInt, Value::I64(i as i64)));
        }
        p.push(Instr::new(OpCode::And, crate::vector::PhysType::Bool, p.result, 0, 0));
        p
    }

    /// The register rebase must saturate into [`Program::overflow`] rather than wrap.
    ///
    /// This is the defect the guard on `and_programs` used to hide: the merge itself did
    /// `lhs.num_regs = base + rhs.num_regs` and `i2.a += base` on `u16`s, so in
    /// `profile.wasm` (overflow checks off) a large enough merge aliased two distinct
    /// registers onto one and answered the query wrongly, with no diagnostic.
    #[test]
    fn merging_past_the_register_ceiling_poisons_the_program() {
        let mut lhs = sized_program(60_000, 0);
        let rhs = sized_program(10_000, 0);
        let (a, b) = merge_program_bodies(&mut lhs, rhs);
        assert!(lhs.overflow, "the merged program must be poisoned");
        // `num_regs` must not have wrapped to 4464.
        assert_eq!(lhs.num_regs, 60_000);
        // The returned registers stay inside the program that is actually there.
        assert!(a < lhs.num_regs && b < lhs.num_regs);
    }

    /// The side tables are `u16`-indexed too, and only `num_regs` was ever checked.
    #[test]
    fn merging_past_a_side_table_ceiling_poisons_the_program() {
        let mut lhs = sized_program(4, 40_000);
        let rhs = sized_program(4, 30_000);
        merge_program_bodies(&mut lhs, rhs);
        assert!(lhs.overflow);
        assert_eq!(lhs.consts.len(), 40_000);
    }

    /// An already-poisoned right-hand side poisons the merge, the same way
    /// `Program::add_lambda` propagates a poisoned body.
    #[test]
    fn merging_a_poisoned_right_hand_side_poisons_the_result() {
        let mut lhs = sized_program(4, 0);
        let mut rhs = sized_program(4, 0);
        rhs.overflow = true;
        merge_program_bodies(&mut lhs, rhs);
        assert!(lhs.overflow);
    }

    /// A merge that fits must be unaffected by the guard.
    #[test]
    fn a_merge_that_fits_is_not_poisoned() {
        let mut lhs = sized_program(60_000, 100);
        let rhs = sized_program(5_000, 100);
        merge_program_bodies(&mut lhs, rhs);
        assert!(!lhs.overflow);
        assert_eq!(lhs.num_regs, 65_000);
        assert_eq!(lhs.consts.len(), 200);
    }

    /// `Vm::eval` is the gate the poison flag feeds: a poisoned program never runs.
    #[test]
    fn a_poisoned_merged_program_is_refused_at_evaluation() {
        let mut lhs = sized_program(60_000, 0);
        merge_program_bodies(&mut lhs, sized_program(10_000, 0));
        assert!(lhs.overflow);
        let batch = crate::vector::Batch::rows_only(1);
        let mut vm = crate::expr::vm::Vm::new();
        assert_eq!(code_of(vm.eval(&lhs, &batch)), Some(crate::error::Code::LimitExceeded));
    }
}
