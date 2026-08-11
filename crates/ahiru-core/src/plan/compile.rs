//! AST 式 → バイトコード。
//!
//! ここが型検査も兼ねる。暗黙変換は `Ty::unify` に一本化し、必要な場所に
//! 明示的な `Cast` 命令を挿入する。実行カーネルが型変換を意識しなくて済むので
//! カーネル数を増やさずに済む（DESIGN.md §11）。

use crate::expr::{funcs, Instr, OpCode, Program, Reg};
use crate::plan::{AggKind, Scope};
use crate::prelude::*;
use crate::rt::hash::eq_ascii_ci;
use crate::sql::ast::{BinaryOp, Expr, ExprArena, ExprId, UnaryOp};
use crate::vector::{Field, Ty, Value};

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

/// DATE の加減算の結果型を決める。返り値の `bool` は左右を入れ替えるべきか。
///
/// - `DATE + 整数` / `整数 + DATE` → DATE（日数を足す）
/// - `DATE - 整数` → DATE
/// - `DATE - DATE` → 日数（DuckDB は INTEGER を返す）
///
/// TIMESTAMP と整数の加減算は DuckDB でもエラーなので受け付けない
/// （単位が秒なのかマイクロ秒なのか決められないため）。
fn date_arith(op: BinaryOp, lt: Ty, rt: Ty) -> Option<(Ty, bool)> {
    use BinaryOp::*;
    match (op, lt, rt) {
        (Add, Ty::Date, r) if r.is_integer() => Some((Ty::Date, false)),
        // `1 + DATE` は交換して DATE を左に持ってくる。
        (Add, l, Ty::Date) if l.is_integer() => Some((Ty::Date, true)),
        (Sub, Ty::Date, r) if r.is_integer() => Some((Ty::Date, false)),
        (Sub, Ty::Date, Ty::Date) => Some((Ty::BigInt, false)),
        _ => None,
    }
}

/// DECIMAL が絡む乗除の型を決める。該当しなければ `None`。
///
/// 加減算は「スケールを揃えてから足す」で正しいので通常の共通型経路でよいが、
/// **乗算はスケールが足し算になる**（`1.25 * 2.5 = 3.125`: s=2 と s=3 で s=5）。
/// 共通型に揃えてから掛けると、スケールが 2 倍ずれた値が返ってしまう。
/// カーネルは生の整数として掛けるだけなので、正しいスケールを持つ結果型を
/// ここで決め、両辺は**スケールを変えずに**物理幅だけ広げる。
///
/// 除算は DuckDB と同じく DOUBLE に落とす。整数除算のままだとスケールが
/// 引き算になり、桁が足りずにほとんどの場合 0 になってしまう。
fn decimal_arith(op: BinaryOp, lt: Ty, rt: Ty) -> Option<(Ty, Ty, Ty)> {
    if !matches!(op, BinaryOp::Mul | BinaryOp::Div) {
        return None;
    }
    // どちらかが DECIMAL でなければ通常経路。
    if !matches!(lt, Ty::Decimal { .. }) && !matches!(rt, Ty::Decimal { .. }) {
        return None;
    }
    // 浮動小数が混ざったら DOUBLE に倒す（DuckDB と同じ）。
    if matches!(lt, Ty::Float | Ty::Double) || matches!(rt, Ty::Float | Ty::Double) {
        return Some((Ty::Double, Ty::Double, Ty::Double));
    }
    let (p1, s1) = lt.as_decimal()?;
    let (p2, s2) = rt.as_decimal()?;
    if op == BinaryOp::Div {
        return Some((Ty::Double, Ty::Double, Ty::Double));
    }
    // 乗算: precision は足し算、scale も足し算。
    let res = Ty::decimal(p1.saturating_add(p2), s1.saturating_add(s2));
    let (rp, _) = res.as_decimal()?;
    // 両辺は scale を保ったまま、結果と同じ物理幅へ広げる。
    Some((Ty::decimal(rp, s1), Ty::decimal(rp, s2), res))
}

/// DATE/TIMESTAMP ± INTERVAL, INTERVAL ± INTERVAL, INTERVAL * 整数 の形を
/// 認識する。該当しなければ `None`（＝ 通常の `Ty::unify` 経路へ）。
///
/// `Ty::unify` に載せない理由は `date_arith` と同じ: INTERVAL は他のどの型
/// とも広さの順序を持たないので、共通型への昇格という発想自体が合わない。
enum IntervalOp {
    /// `swap`: 左右を入れ替えるか（`INTERVAL + DATE` の形）。
    /// `negate_b`: 先に INTERVAL 側を符号反転するか（`- INTERVAL` の形）。
    TsInterval {
        swap: bool,
        negate_b: bool,
    },
    IntervalInterval {
        negate_b: bool,
    },
    /// `swap`: 整数が左に来ているか（`3 * INTERVAL`）。
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

/// 2 つのプログラムを `AND` で束ねる。
///
/// `WHERE a AND b` は AST の段階で 1 本にまとめてコンパイルできるが、
/// 述語を分解して一部だけ押し下げたあとに残りを束ね直す場面では、
/// 既にコンパイル済みのプログラム同士をつなぐ必要がある。
/// `lhs`/`rhs` の 2 本の `Program` を 1 本に併合する共通処理。
///
/// レジスタ・定数・キャストテーブルの番号を `rhs` 側だけ `lhs` の末尾に
/// ずらして (`base`/`kbase`/`cbase`)、`rhs.instrs` を `lhs.instrs` に追記する。
/// 呼び出し側は返った `(結果レジスタ番号, リベース後の rhs 結果レジスタ番号)`
/// を使って、末尾に自分の演算（`And`/`Coalesce`/...）を 1 命令足すだけでよい。
///
/// `lhs.num_regs` はここで `rhs` ぶん増やしてあるので、呼び出し側は
/// そのまま `lhs.alloc_reg()` を呼んでよい。
pub(crate) fn merge_program_bodies(lhs: &mut Program, rhs: Program) -> (Reg, Reg) {
    let base = lhs.num_regs;
    let kbase = lhs.consts.len() as u16;
    let cbase = lhs.casts.len() as u16;

    lhs.consts.extend(rhs.consts.iter().cloned());
    lhs.casts.extend(rhs.casts.iter().copied());
    for i in &rhs.instrs {
        let mut i2 = *i;
        i2.dst += base;
        // LoadCol / LoadConst は a・b を使わないので、ずらすと壊れる。
        match i2.op {
            OpCode::LoadCol => {}
            OpCode::LoadConst => i2.aux += kbase,
            OpCode::Cast | OpCode::TryCast => {
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
    (lhs.result, rhs.result + base)
}

pub fn and_programs(mut lhs: Program, rhs: Program) -> Result<Program> {
    ensure!(lhs.num_regs as usize + rhs.num_regs as usize <= u16::MAX as usize, LimitExceeded);
    let (a, b) = merge_program_bodies(&mut lhs, rhs);
    let dst = lhs.alloc_reg();
    lhs.push(Instr::new(OpCode::And, crate::vector::PhysType::Bool, dst, a, b));
    lhs.result = dst;
    lhs.result_ty = Ty::Boolean;
    Ok(lhs)
}

/// 既にコンパイル済みのプログラムの結果を別の型へ変換する。
/// 集合演算で左右の列型を揃えるときに使う。
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
            Expr::Like { arg: a1, pattern: p1, negated: n1, escape: e1, ci: c1 },
            Expr::Like { arg: a2, pattern: p2, negated: n2, escape: e2, ci: c2 },
        ) => n1 == n2 && e1 == e2 && c1 == c2 && eq(a1, a2) && eq(p1, p2),
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

/// ラムダを引数に取りうる関数名か（`sql::parser::is_lambda_func` と同じ
/// 固定集合。パーサ側はこの名前の集合の引数位置でだけ `->` をラムダとして
/// 読み、束縛後にここでも同じ判定で `Compiler::lambda_call` へ振り分ける）。
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

    /// `from` 型のレジスタを `to` 型へ揃える。
    fn coerce(&mut self, reg: Reg, from: Ty, to: Ty) -> Result<Reg> {
        self.coerce_with(OpCode::Cast, reg, from, to)
    }

    /// `TRY_CAST` 用。`coerce` と同じ命令列を組むが `Cast` の代わりに
    /// `TryCast` を発行する。変換できない組み合わせは実行時にエラーではなく
    /// 全行 NULL になる（`expr::vm::exec` 参照）。行単位の変換失敗
    /// （範囲外・パース不能）はどちらの命令でも元々その行だけ NULL になる
    /// （`kernels::cast` の契約）ので、ここで違いを付ける必要はない。
    fn try_coerce(&mut self, reg: Reg, from: Ty, to: Ty) -> Result<Reg> {
        self.coerce_with(OpCode::TryCast, reg, from, to)
    }

    /// `coerce`/`try_coerce` の共通実装。`op` は `Cast`/`TryCast` のどちらか。
    fn coerce_with(&mut self, op: OpCode, reg: Reg, from: Ty, to: Ty) -> Result<Reg> {
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
        self.prog.push(Instr::with_aux(op, from.phys(), dst, reg, 0, aux));
        Ok(dst)
    }

    /// `ILIKE` 用に `lower(x)` を 1 引数呼び出しとして発行する。
    fn lower_reg(&mut self, r: Reg) -> Result<Reg> {
        let (id, _want, res) = crate::expr::funcs::resolve("lower", &[Ty::Varchar])?;
        let aux = self.prog.add_call(id, vec![r], res);
        let dst = self.prog.alloc_reg();
        self.prog.push(Instr::with_aux(OpCode::Call, res.phys(), dst, 0, 0, aux));
        Ok(dst)
    }

    /// 二項演算の両辺を共通型に揃える。
    fn unify_operands(&mut self, lr: Reg, lt: Ty, rr: Reg, rt: Ty) -> Result<(Reg, Reg, Ty)> {
        let t = Ty::unify_or_mismatch(lt, rt)?;
        let l = self.coerce(lr, lt, t)?;
        let r = self.coerce(rr, rt, t)?;
        Ok((l, r, t))
    }

    /// `interval_arith` が認識した形をバイトコードへ落とす。
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
                // DATE は先に TIMESTAMP へ寄せる。DuckDB も DATE ± INTERVAL は
                // TIMESTAMP を返す（INTERVAL が時刻成分を持ちうるため）。
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
        // 置き換えが先。集約結果と GROUP BY キーは既に列として存在している。
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
            Expr::Binary { op, lhs, rhs } => self.binary(*op, *lhs, *rhs),
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
                // 入力の物理型は問わない。カーネルは validity しか見ない。
                let dst = self.emit(op, Ty::Boolean, r, 0);
                Ok((dst, Ty::Boolean))
            }
            Expr::Between { arg, low, high, negated } => self.between(*arg, *low, *high, *negated),
            Expr::InList { arg, list, negated } => self.in_list(*arg, list.clone(), *negated),
            Expr::Like { arg, pattern, negated, escape, ci } => {
                ensure!(escape.is_none(), UnsupportedFeature);
                let (a, at) = self.expr(*arg)?;
                let (p, pt) = self.expr(*pattern)?;
                let mut a = self.coerce(a, at, Ty::Varchar)?;
                let mut p = self.coerce(p, pt, Ty::Varchar)?;
                // ILIKE は大小文字を無視する。専用カーネルは持たず、`lower()` を
                // 両辺にかけてから通常の LIKE に落とす（upper/lower と同じ
                // ASCII 限定の制限をそのまま継承する）。
                if *ci {
                    a = self.lower_reg(a)?;
                    p = self.lower_reg(p)?;
                }
                let dst = self.emit(OpCode::Like, Ty::Varchar, a, p);
                Ok((self.maybe_not(dst, *negated), Ty::Boolean))
            }
            Expr::Case { operand, whens, else_ } => self.case(*operand, whens.clone(), *else_),
            // ウィンドウ関数とサブクエリはバインダが専用ノードに書き換えてから
            // 来る。ここに残っているのは、書ける位置ではないところに書かれた場合。
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
            // `UNNEST` もバインダが `Node::Unnest` + `Substitution` に書き換えて
            // から来る。ここに残っているのは書ける位置ではないところに書かれた
            // 場合（`plan::bind::collect_unnests` が検出して先に拒否するので、
            // 実質的にはバインダのバグ検出用の網）。
            Expr::Unnest(_) => err!(UnsupportedFeature),
            // `list_transform`/`list_filter`/`list_reduce` の第 2 引数としてしか
            // パーサは生成しない（`sql::parser::Parser::call` 参照）。それ以外の
            // 位置に来た（＝バグかパーサの取りこぼし）場合はここで弾く。
            Expr::Lambda { .. } => err!(UnsupportedFeature),
            Expr::Function { name, args, distinct, star, filter } => {
                // 集約関数はバインダが置き換えてから来る。ここに残っているのは
                // 集約できない位置（WHERE など）に書かれた場合か、集約の入れ子。
                if AggKind::from_name(name).is_some() {
                    err!(NotAggregate);
                }
                // FILTER はスカラ関数には意味を持たない。
                ensure!(!*distinct && !*star && filter.is_none(), UnsupportedFeature);
                if is_lambda_func(name) {
                    return self.lambda_call(name, args);
                }
                self.scalar_call(name, args)
            }
        }
    }

    /// スカラ関数呼び出し。型検査と引数の変換はここで済ませ、実行時には
    /// 変換済みのベクタだけを渡す。
    fn scalar_call(&mut self, name: &str, args: &[ExprId]) -> Result<(Reg, Ty)> {
        let mut regs = Vec::with_capacity(args.len());
        let mut tys = Vec::with_capacity(args.len());
        for a in args {
            let (r, t) = self.expr(*a)?;
            regs.push(r);
            tys.push(t);
        }
        let (id, want, res) = crate::expr::funcs::resolve(name, &tys)?;
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
    /// `list_reduce(list, (acc, x) -> expr [, initial])`。
    ///
    /// `scalar_call` の一般経路（各引数を `self.expr()` で外側スコープの
    /// レジスタへコンパイルしてから 1 個の `Call` 命令にまとめる）には乗らない:
    /// このバイトコード VM はベクタ化実行が前提で「配列の要素ごとに式を評価
    /// する」処理は行あたり可変長になり形が合わない。そこでラムダ本体だけを
    /// **別の小さな `Program`** としてコンパイルし `Program::lambdas` に埋め込む。
    /// 実行時は `expr::funcs::call_lambda` が配列の要素数ぶんだけ
    /// `Batch::new` + `Vm::eval` を繰り返す（`ddl`/`dml` が 1 行だけのバッチを
    /// 組んで `Vm::eval` に通すのと同じ発想。`expr::funcs` モジュール冒頭の
    /// list_transform/list_filter/list_reduce セクション doc も参照）。
    ///
    /// **既知の制限**: ラムダ本体は自分のパラメータだけを参照できる。外側の
    /// SQL スコープの列は参照できない（`list_transform(tags, x -> x ||
    /// suffix_col)` のように外側列を混ぜる書き方は非対応で `ColumnNotFound`
    /// になる）。本体は毎回パラメータだけの孤立した `Scope` でコンパイルする
    /// ため。
    ///
    /// パラメータの型は常に `Ty::Json`（`list_extract` の結果と同じ、配列
    /// 要素はすべて動的型付けの JSON 値として表現される）。このエンジンでは
    /// `Ty::Json` は他のどの型とも `Ty::unify` しない（`vector::types` の doc
    /// 参照）ため、本体でパラメータに算術・比較を行うには
    /// `CAST(CAST(x AS VARCHAR) AS INTEGER)` のように一度 VARCHAR を経由して
    /// 明示的に変換する必要がある（`list_extract` の結果に対する既存の制限と
    /// 同じで、ラムダ固有の制約ではない）。
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

        // 第 1 引数（リスト）は通常どおり外側スコープでコンパイルする。
        let (list_reg, list_ty) = self.expr(args[0])?;
        ensure!(matches!(list_ty, Ty::Json | Ty::Null), TypeMismatch);
        let list_reg =
            if list_ty == Ty::Null { self.konst(Ty::Json, Value::Null) } else { list_reg };

        let (params, body) = match self.arena.get(args[1]) {
            Expr::Lambda { params, body } => (params.clone(), *body),
            // パーサは `list_transform` 等の第 2 引数をラムダ構文
            // （`x -> expr` / `(a, b) -> expr`）としてしか読まないので、他の
            // 式（列参照など）が来るのは構文エラー。
            _ => err!(SyntaxError),
        };
        let want_params = if is_reduce { 2 } else { 1 };
        ensure!(params.len() == want_params, WrongArgCount);

        // `list_reduce` の第 3 引数（初期値）。省略時はリストの先頭要素を使う
        // （`expr::funcs::call_list_reduce` 参照）。第 1 引数と同様、Ty::Json
        // でなければならない（`to_json(...)` 等で明示的に変換して渡す）。
        let mut call_args = vec![list_reg];
        if args.len() == 3 {
            let (init_reg, init_ty) = self.expr(args[2])?;
            ensure!(matches!(init_ty, Ty::Json | Ty::Null), TypeMismatch);
            let init_reg =
                if init_ty == Ty::Null { self.konst(Ty::Json, Value::Null) } else { init_reg };
            call_args.push(init_reg);
        }

        // 本体は「パラメータだけ」の孤立したスコープでコンパイルする
        // （このメソッドの doc の「既知の制限」参照）。
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

    /// 置き換え指示に一致する式なら、その入力列を読むだけにする。
    ///
    /// **後から追加した指示を優先する**（逆順に走査する）。同じ式に対して
    /// 「スカラサブクエリの列」と「集約後のグループ列」の両方が登録されうるが、
    /// 集約より上では後者が正しいため。
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

        // DATE と整数の加減算は日数として扱う（DuckDB と同じ）。
        // `unify(Date, Int)` は None なので、共通型経路には乗らない。
        if let Some((res, swap)) = date_arith(op, lt, rt) {
            let (l, r) = if swap { (rr, lr) } else { (lr, rr) };
            let code = if op == BinaryOp::Add { OpCode::Add } else { OpCode::Sub };
            // DATE も INTEGER も物理型は I32 なので、演算自体に変換は要らない。
            let dst = self.prog.alloc_reg();
            self.prog.push(Instr::new(code, crate::vector::PhysType::I32, dst, l, r));
            // 日数差は DuckDB が BIGINT を返すので合わせる。値は同じだが、
            // 型が違うと上位の型解決がずれる。
            if res == Ty::BigInt {
                return Ok((self.coerce(dst, Ty::Int, Ty::BigInt)?, Ty::BigInt));
            }
            return Ok((dst, res));
        }

        // DECIMAL の乗除はスケールが変わるので、共通型に揃える経路には乗せない。
        if let Some((lcast, rcast, res)) = decimal_arith(op, lt, rt) {
            let l = self.coerce(lr, lt, lcast)?;
            let r = self.coerce(rr, rt, rcast)?;
            let code = if op == BinaryOp::Mul { OpCode::Mul } else { OpCode::Div };
            return Ok((self.emit(code, res, l, r), res));
        }

        // DATE/TIMESTAMP ± INTERVAL, INTERVAL ± INTERVAL, INTERVAL * 整数。
        if let Some(kind) = interval_arith(op, lt, rt) {
            return self.compile_interval_op(kind, lr, lt, rr, rt);
        }

        let (l, r, t) = self.unify_operands(lr, lt, rr, rt)?;

        if op.is_comparison() {
            // 大小比較は月・日・マイクロ秒の相対的な重みが定義できない
            // （1 か月は 28〜31 日のどれとも比較しうる）ので、順序比較は
            // 未対応のまま明示的にエラーにする。等価比較はビットパターンの
            // 一致で正しく判定できるので許す。
            // JSON も INTERVAL と同じ理由（大小の順序が定義できない）で
            // 等価比較のみ許す。等価はバイト列一致で判定するので、キー順序や
            // 空白の違いだけで不一致になりうる（DuckDB のような正規化比較は
            // 行わない、v1 の既知の制限）。
            if t == Ty::Interval || t == Ty::Json {
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

    // --- ILIKE / TRY_CAST -------------------------------------------------------

    #[test]
    fn ilike_lowers_both_sides_before_the_like_kernel() {
        let mut a = ExprArena::new();
        let arg = a.push(Expr::ColumnRef { qualifier: None, name: "name".into() });
        let pattern = a.push(Expr::Literal(Value::Bytes(b"A%".to_vec())));
        let id = a.push(Expr::Like { arg, pattern, negated: false, escape: None, ci: true });
        let p = compile(&a, &cols(), &[], id).unwrap();
        assert_eq!(p.result_ty, Ty::Boolean);
        // `lower()` の呼び出しが 2 回（両辺）、その後に Like が 1 回。
        assert_eq!(p.calls.len(), 2);
        assert!(p.instrs.iter().any(|i| i.op == OpCode::Like));
        // ILIKE でない通常の LIKE は lower() を呼ばない。
        let id2 = a.push(Expr::Like { arg, pattern, negated: false, escape: None, ci: false });
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
        // FILTER はスカラ関数呼び出しには意味を持たない。集約置換前にここへ
        // 来た（＝集約できない位置に書かれた）場合と同じ経路でも弾かれる。
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
        // DATE → TIMESTAMP への暗黙キャストが挟まる。
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

        // `n * INTERVAL` も同じ経路（左右入れ替え）。
        let mul2 = a.push(Expr::Binary { op: BinaryOp::Mul, lhs: n, rhs: i1 });
        let p = compile(&a, &cols(), &[], mul2).unwrap();
        assert!(p.instrs.iter().any(|i| i.op == OpCode::IntervalMul));
    }

    #[test]
    fn interval_ordering_comparison_is_rejected_but_equality_is_not() {
        let mut a = ExprArena::new();
        let i1 = interval_lit(&mut a, 1, 0, 0);
        let i2 = interval_lit(&mut a, 0, 30, 0);
        let lt = a.push(Expr::Binary { op: BinaryOp::Lt, lhs: i1, rhs: i2 });
        assert_eq!(code_of(compile(&a, &cols(), &[], lt)), Some(Code::TypeMismatch));

        let eq = a.push(Expr::Binary { op: BinaryOp::Eq, lhs: i1, rhs: i2 });
        let p = compile(&a, &cols(), &[], eq).unwrap();
        assert_eq!(p.result_ty, Ty::Boolean);
    }

    #[test]
    fn json_ordering_comparison_is_rejected_but_equality_is_not() {
        // JSON も INTERVAL と同じ理由で大小比較を拒否する。
        // JSON は他のどの型とも `Ty::unify` しない（モジュール doc 参照）ので、
        // 比較相手も明示的に JSON へ CAST してから渡す。
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
        // JSON への CAST は他の型と同じ経路（`Expr::Cast` → `Cast`/`TryCast`
        // opcode）を通るだけで、compile.rs 側に特別な分岐は要らない
        // （検証は `expr::kernels::cast` にある）。
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

    // --- ラムダ: list_transform / list_filter / list_reduce -------------------

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
        // 本体が `x`（Ty::Json）そのものなので BOOLEAN にならない。
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
        // 既知の制限: ラムダ本体は自分のパラメータだけを参照できる
        // （`Compiler::lambda_call` の doc 参照）。`id` は外側スコープの列で
        // パラメータではないので `ColumnNotFound` になる。
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
        // list_transform はパラメータ 1 個のラムダしか受け付けない。
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

        // 第 3 引数（初期値）付きも受け付ける。
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
        // パーサは list_transform 等の第 2 引数としてしかラムダを作らないが、
        // `plan::compile` 側も念のため通常の式位置に来た場合を拒否する。
        let mut a = ExprArena::new();
        let x = a.push(Expr::ColumnRef { qualifier: None, name: "x".into() });
        let id = a.push(Expr::Lambda { params: vec!["x".into()], body: x });
        assert_eq!(code_of(compile(&a, &cols(), &[], id)), Some(Code::UnsupportedFeature));
    }
}
