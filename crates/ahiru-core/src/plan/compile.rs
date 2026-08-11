//! AST 式 → バイトコード。
//!
//! ここが型検査も兼ねる。暗黙変換は `Ty::unify` に一本化し、必要な場所に
//! 明示的な `Cast` 命令を挿入する。実行カーネルが型変換を意識しなくて済むので
//! カーネル数を増やさずに済む（DESIGN.md §11）。

use crate::expr::{Instr, OpCode, Program, Reg};
use crate::plan::{AggKind, Scope};
use crate::prelude::*;
use crate::sql::ast::{BinaryOp, Expr, ExprArena, ExprId, UnaryOp};
use crate::vector::{Ty, Value};

/// 式のネスト上限。パーサ側でも制限しているが、二重に守る。
const MAX_DEPTH: u32 = 64;

/// 部分式を入力列で置き換える指示。
///
/// 集約の上で式をコンパイルするときに使う。`SELECT a + 1, count(*) ... GROUP BY a + 1`
/// なら、`a + 1` と `count(*)` はどちらも集約オペレータの出力列になっているので、
/// そこを `LoadCol` に差し替えて再評価を避ける。
#[derive(Clone, Copy)]
pub struct Substitution {
    /// 置き換える対象の式。
    pub expr: ExprId,
    /// 差し替え先の入力列番号。
    pub column: usize,
    /// `true` なら構造の一致で判定する（GROUP BY 式の照合）。
    /// `false` なら同一ノードのときだけ（集約呼び出しの照合）。
    pub structural: bool,
}

pub struct Compiler<'a> {
    arena: &'a ExprArena,
    /// 入力バッチの列。添字がそのまま `LoadCol` の列番号になる。
    scope: &'a Scope,
    /// `?` プレースホルダに束縛された値。
    params: &'a [Value],
    subs: &'a [Substitution],
    prog: Program,
    depth: u32,
}

/// 単一の式をコンパイルする。
pub fn compile(arena: &ExprArena, scope: &Scope, params: &[Value], id: ExprId) -> Result<Program> {
    compile_with_subs(arena, scope, params, &[], id)
}

/// 置き換え指示付きでコンパイルする。集約の上で使う。
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

/// 述語をコンパイルする。結果が BOOLEAN でなければエラー。
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
    ensure!(matches!(p.result_ty, Ty::Boolean | Ty::Null), TypeMismatch);
    Ok(p)
}

/// 入力列をそのまま返すだけのプログラム。`SELECT *` や結合キーで使う。
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

/// 2 つのプログラムを `AND` で束ねる。
///
/// `WHERE a AND b` は AST の段階で 1 本にまとめてコンパイルできるが、
/// 述語を分解して一部だけ押し下げたあとに残りを束ね直す場面では、
/// 既にコンパイル済みのプログラム同士をつなぐ必要がある。
pub fn and_programs(mut lhs: Program, rhs: Program) -> Result<Program> {
    let base = lhs.num_regs;
    let kbase = lhs.consts.len() as u16;
    let cbase = lhs.casts.len() as u16;
    ensure!(base as usize + rhs.num_regs as usize <= u16::MAX as usize, LimitExceeded);

    lhs.consts.extend(rhs.consts.iter().cloned());
    lhs.casts.extend(rhs.casts.iter().copied());
    for i in &rhs.instrs {
        let mut i2 = *i;
        i2.dst += base;
        // LoadCol / LoadConst は a・b を使わないので、ずらすと壊れる。
        match i2.op {
            OpCode::LoadCol => {}
            OpCode::LoadConst => i2.aux += kbase,
            OpCode::Cast => {
                i2.a += base;
                i2.aux += cbase;
            }
            // Select だけ aux が第 3 オペランドのレジスタ番号。
            OpCode::Select => {
                i2.a += base;
                i2.b += base;
                i2.aux += base;
            }
            _ => {
                i2.a += base;
                i2.b += base;
            }
        }
        lhs.instrs.push(i2);
    }
    lhs.num_regs = base + rhs.num_regs;
    let a = lhs.result;
    let b = rhs.result + base;
    let dst = lhs.alloc_reg();
    lhs.push(Instr::new(OpCode::And, crate::vector::PhysType::Bool, dst, a, b));
    lhs.result = dst;
    lhs.result_ty = Ty::Boolean;
    Ok(lhs)
}

/// 2 つの式が構造的に等しいか。`GROUP BY` 式と SELECT 中の部分式の照合に使う。
///
/// 名前の比較は大文字小文字を区別しない（`GROUP BY a` と `SELECT A` を同じと
/// 見なす）。定数は値の一致で判定する。
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
        (Expr::Cast { arg: a1, ty: t1 }, Expr::Cast { arg: a2, ty: t2 }) => t1 == t2 && eq(a1, a2),
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
            Expr::Like { arg: a1, pattern: p1, negated: n1, escape: e1 },
            Expr::Like { arg: a2, pattern: p2, negated: n2, escape: e2 },
        ) => n1 == n2 && e1 == e2 && eq(a1, a2) && eq(p1, p2),
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
            Expr::Function { name: n1, args: a1, distinct: d1, star: s1 },
            Expr::Function { name: n2, args: a2, distinct: d2, star: s2 },
        ) => {
            d1 == d2
                && s1 == s2
                && crate::rt::hash::eq_ascii_ci(n1.as_bytes(), n2.as_bytes())
                && a1.len() == a2.len()
                && a1.iter().zip(a2).all(|(x, y)| eq(x, y))
        }
        _ => false,
    }
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

    /// `from` 型のレジスタを `to` 型へ揃える。
    fn coerce(&mut self, reg: Reg, from: Ty, to: Ty) -> Result<Reg> {
        if from == to {
            return Ok(reg);
        }
        // NULL リテラルは型を持たない。変換するのではなく、目的の型の NULL 定数を
        // 作り直す。こうすると Cast カーネルが Ty::Null を扱わなくて済む。
        if from == Ty::Null {
            return Ok(self.konst(to, Value::Null));
        }
        let aux = self.prog.add_cast(from, to);
        let dst = self.prog.alloc_reg();
        self.prog.push(Instr::with_aux(OpCode::Cast, from.phys(), dst, reg, 0, aux));
        Ok(dst)
    }

    /// 二項演算の両辺を共通型に揃える。
    fn unify_operands(&mut self, lr: Reg, lt: Ty, rr: Reg, rt: Ty) -> Result<(Reg, Reg, Ty)> {
        let t = match Ty::unify(lt, rt) {
            Some(t) => t,
            None => err!(TypeMismatch),
        };
        let l = self.coerce(lr, lt, t)?;
        let r = self.coerce(rr, rt, t)?;
        Ok((l, r, t))
    }

    fn expr(&mut self, id: ExprId) -> Result<(Reg, Ty)> {
        self.depth += 1;
        ensure!(self.depth <= MAX_DEPTH, ExpressionTooDeep);
        let r = self.expr_inner(id);
        self.depth -= 1;
        r
    }

    fn expr_inner(&mut self, id: ExprId) -> Result<(Reg, Ty)> {
        // 置き換えが先。集約結果と GROUP BY キーは既に列として存在している。
        if let Some(r) = self.substitute(id) {
            return r;
        }
        match self.arena.get(id) {
            Expr::Literal(v) => {
                let ty = v.default_ty();
                Ok((self.konst(ty, v.clone()), ty))
            }
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
            Expr::Binary { op, lhs, rhs } => self.binary(*op, *lhs, *rhs),
            Expr::Cast { arg, ty } => {
                let (r, from) = self.expr(*arg)?;
                Ok((self.coerce(r, from, *ty)?, *ty))
            }
            Expr::IsNull { arg, negated } => {
                let (r, _) = self.expr(*arg)?;
                let op = if *negated { OpCode::IsNotNull } else { OpCode::IsNull };
                // 入力の物理型は問わない。カーネルは validity しか見ない。
                let dst = self.emit(op, Ty::Boolean, r, 0);
                Ok((dst, Ty::Boolean))
            }
            Expr::Between { arg, low, high, negated } => self.between(*arg, *low, *high, *negated),
            Expr::InList { arg, list, negated } => self.in_list(*arg, list.clone(), *negated),
            Expr::Like { arg, pattern, negated, escape } => {
                ensure!(escape.is_none(), UnsupportedFeature);
                let (a, at) = self.expr(*arg)?;
                let (p, pt) = self.expr(*pattern)?;
                let a = self.coerce(a, at, Ty::Varchar)?;
                let p = self.coerce(p, pt, Ty::Varchar)?;
                let dst = self.emit(OpCode::Like, Ty::Varchar, a, p);
                Ok((self.maybe_not(dst, *negated), Ty::Boolean))
            }
            Expr::Case { operand, whens, else_ } => self.case(*operand, whens.clone(), *else_),
            Expr::Function { name, args, distinct, star } => {
                let _ = (args, distinct, star);
                // 集約関数はバインダが置き換えてから来る。ここに残っているのは
                // 集約できない位置（WHERE など）に書かれた場合か、集約の入れ子。
                if AggKind::from_name(name).is_some() {
                    err!(NotAggregate);
                }
                // スカラ関数は v1 では未提供。
                err!(FunctionNotFound)
            }
        }
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

    /// 置き換え指示に一致する式なら、その入力列を読むだけにする。
    fn substitute(&mut self, id: ExprId) -> Option<Result<(Reg, Ty)>> {
        for s in self.subs {
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
                ensure!(t.is_numeric() || t == Ty::Null, TypeMismatch);
                Ok((self.emit(OpCode::Neg, t, r, 0), t))
            }
            UnaryOp::Not => {
                ensure!(matches!(t, Ty::Boolean | Ty::Null), TypeMismatch);
                Ok((self.emit(OpCode::Not, Ty::Boolean, r, 0), Ty::Boolean))
            }
        }
    }

    fn binary(&mut self, op: BinaryOp, lhs: ExprId, rhs: ExprId) -> Result<(Reg, Ty)> {
        let (lr, lt) = self.expr(lhs)?;
        let (rr, rt) = self.expr(rhs)?;

        if op.is_logical() {
            let l = self.coerce(lr, lt, Ty::Boolean)?;
            let r = self.coerce(rr, rt, Ty::Boolean)?;
            let code = if op == BinaryOp::And { OpCode::And } else { OpCode::Or };
            return Ok((self.emit(code, Ty::Boolean, l, r), Ty::Boolean));
        }

        if op == BinaryOp::Concat {
            let l = self.coerce(lr, lt, Ty::Varchar)?;
            let r = self.coerce(rr, rt, Ty::Varchar)?;
            return Ok((self.emit(OpCode::Concat, Ty::Varchar, l, r), Ty::Varchar));
        }

        let (l, r, t) = self.unify_operands(lr, lt, rr, rt)?;

        if op.is_comparison() {
            let code = match op {
                BinaryOp::Eq => OpCode::Eq,
                BinaryOp::Ne => OpCode::Ne,
                BinaryOp::Lt => OpCode::Lt,
                BinaryOp::Le => OpCode::Le,
                BinaryOp::Gt => OpCode::Gt,
                _ => OpCode::Ge,
            };
            // 比較の入力型で命令を発行し、出力は必ず BOOLEAN。
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
        // BETWEEN は (a >= lo) AND (a <= hi) に展開する。専用カーネルを持たない。
        let (ar, at) = self.expr(arg)?;
        let (lr, lt) = self.expr(low)?;
        let (hr, ht) = self.expr(high)?;

        let (a1, l, t1) = self.unify_operands(ar, at, lr, lt)?;
        let ge = self.prog.alloc_reg();
        self.prog.push(Instr::new(OpCode::Ge, t1.phys(), ge, a1, l));

        let (a2, h, t2) = self.unify_operands(ar, at, hr, ht)?;
        let le = self.prog.alloc_reg();
        self.prog.push(Instr::new(OpCode::Le, t2.phys(), le, a2, h));

        let both = self.emit(OpCode::And, Ty::Boolean, ge, le);
        Ok((self.maybe_not(both, negated), Ty::Boolean))
    }

    fn in_list(&mut self, arg: ExprId, list: Vec<ExprId>, negated: bool) -> Result<(Reg, Ty)> {
        ensure!(!list.is_empty(), SyntaxError);
        // IN は Eq の OR 連鎖に展開する。専用の集合命令を持たないぶん、
        // 要素数が多いと線形になる。v1 では許容する。
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

        // まず条件と値を評価し、結果型を決めてから Select を後ろから積む。
        let operand = match operand {
            Some(o) => Some(self.expr(o)?),
            None => None,
        };

        let mut conds = Vec::with_capacity(whens.len());
        let mut vals = Vec::with_capacity(whens.len());
        let mut result_ty = Ty::Null;
        for (c, v) in whens {
            let cond = match operand {
                // CASE x WHEN a … は x = a に読み替える。
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
            result_ty = match Ty::unify(result_ty, vt) {
                Some(t) => t,
                None => err!(TypeMismatch),
            };
            conds.push(cond);
            vals.push((vr, vt));
        }

        let else_reg = match else_ {
            Some(e) => {
                let (er, et) = self.expr(e)?;
                result_ty = match Ty::unify(result_ty, et) {
                    Some(t) => t,
                    None => err!(TypeMismatch),
                };
                Some((er, et))
            }
            None => None,
        };

        let mut acc = match else_reg {
            Some((r, t)) => self.coerce(r, t, result_ty)?,
            None => self.konst(result_ty, Value::Null),
        };
        // 後ろの WHEN から前へ積むと、条件の優先順位がそのまま入れ子になる。
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
    use crate::sql::ast::Expr;
    use crate::vector::Field;

    fn cols() -> Scope {
        Scope::from_fields(vec![
            Field::new("id", Ty::Int, false),
            Field::new("big", Ty::BigInt, true),
            Field::new("name", Ty::Varchar, true),
        ])
    }

    /// `id <op> <literal>` の式を組み立てる。
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
        // BIGINT 列と INTEGER リテラルの比較は BIGINT に揃える。
        let mut a = ExprArena::new();
        let id = bin(&mut a, BinaryOp::Lt, "big", Value::I32(5));
        let p = compile(&a, &cols(), &[], id).unwrap();
        assert_eq!(p.casts.len(), 1);
        assert_eq!(p.casts[0].from, Ty::Int);
        assert_eq!(p.casts[0].to, Ty::BigInt);
    }

    #[test]
    fn null_literal_is_retyped_not_cast() {
        // NULL は Cast ではなく目的の型の定数として作り直される。
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
    fn deep_nesting_is_rejected_without_overflowing_the_stack() {
        let mut a = ExprArena::new();
        let mut id = a.push(Expr::Literal(Value::I32(1)));
        for _ in 0..500 {
            let r = a.push(Expr::Literal(Value::I32(1)));
            id = a.push(Expr::Binary { op: BinaryOp::Add, lhs: id, rhs: r });
        }
        assert_eq!(code_of(compile(&a, &cols(), &[], id)), Some(Code::ExpressionTooDeep));
    }
}
