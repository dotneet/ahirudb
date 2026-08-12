//! The execution kernels, per physical type.
//!
//! Kernels are written **only against the six physical types** (DESIGN.md §8, §11). Having a
//! kernel per logical type would blow the 1 MiB budget through monomorphization. There are four concrete measures:
//!
//! 1. **No constant-specific kernels.** A constant is represented as a length-1 vector, and each
//!    operand's stride (0 when the length is 1, otherwise 1) is decided outside the loop.
//!    That folds vec-vec / vec-const / const-vec / const-const into one.
//! 2. **The six comparisons fold into one.** A three-way comparison (plus "unordered" for NaN) is
//!    represented in bits and ANDed with a per-operator mask. One comparison kernel per physical type.
//! 3. **Selection is never consulted.** The VM has already gathered via `LoadCol`, so the vectors
//!    a kernel receives are always dense. The presence of selection is not a type parameter.
//! 4. **One arithmetic kernel per physical type.** Operators are distinguished by a `match`
//!    inside the loop (branch prediction handles it, and the code does not grow with the operator count).
//!
//! NULLs are decoupled from "computing values". For most operations the result's validity is the
//! AND of the inputs' validity, and the value's contents are meaningless on NULL rows. The
//! exceptions are three-valued `AND`/`OR` (`logic`) and division by zero (which sets the value and the NULL together).

use crate::expr::{funcs, OpCode};
use crate::prelude::*;
use crate::vector::{
    pack_interval, unpack_interval, Bitmap, BytesData, Data, PhysType, Ty, Vector,
};

/// Microseconds in a day. DATE(I32, days) <-> TIMESTAMP(I64, microseconds).
const MICROS_PER_DAY: i128 = 86_400_000_000;

/// 2^127. Used for range checks on f64 -> i128 (converting i128::MAX to f64 rounds it up).
const I128_LIMIT: f64 = 170_141_183_460_469_231_731_687_303_715_884_105_728.0;

// --- Strides and validity ---------------------------------------------------

/// The stride for reading an operand of length `l` as `n` rows.
/// Length 1 (= a constant) gives 0, so one loop handles both vectors and constants.
fn stride(l: usize, n: usize) -> Result<usize> {
    if l == n {
        Ok(1)
    } else if l == 1 {
        Ok(0)
    } else {
        err!(Internal)
    }
}

/// The row count and strides of a binary operation. If either is empty the result is empty too.
pub fn strides2(la: usize, lb: usize) -> Result<(usize, usize, usize)> {
    let n = if la == 0 || lb == 0 { 0 } else { core::cmp::max(la, lb) };
    Ok((n, stride(la, n)?, stride(lb, n)?))
}

/// The ternary (`Select`) version.
fn strides3(la: usize, lb: usize, lc: usize) -> Result<(usize, usize, usize, usize)> {
    let n =
        if la == 0 || lb == 0 || lc == 0 { 0 } else { core::cmp::max(core::cmp::max(la, lb), lc) };
    Ok((n, stride(la, n)?, stride(lb, n)?, stride(lc, n)?))
}

/// The AND of the inputs' validity. `None` (no bitmap built) when neither has NULLs.
fn combine_validity(a: &Vector, sa: usize, b: &Vector, sb: usize, n: usize) -> Option<Bitmap> {
    if !a.has_nulls() && !b.has_nulls() {
        return None;
    }
    let mut m = Bitmap::with_capacity(n);
    for i in 0..n {
        m.push(a.is_valid(i * sa) && b.is_valid(i * sb));
    }
    Some(m)
}

/// Merges the extra NULLs a kernel produced into the input-derived validity to finish up.
fn finish(ty: Ty, data: Data, validity: Option<Bitmap>, extra: Option<Bitmap>) -> Vector {
    let v = match (validity, extra) {
        (Some(mut a), Some(b)) => {
            a.and_assign(&b);
            Some(a)
        }
        (Some(a), None) => Some(a),
        (None, e) => e,
    };
    let mut out = Vector::from_data(ty, data, v);
    out.compact_validity();
    out
}

// --- Arithmetic ---------------------------------------------------------------
// Integers wrap. Overflow does not panic (if checked semantics are ever needed, a separate OpCode
// can be added). Division by zero and MIN / -1 give NULL (the design decision in mod.rs).

macro_rules! int_arith {
    ($name:ident, $t:ty) => {
        fn $name(
            op: OpCode,
            a: &[$t],
            sa: usize,
            b: &[$t],
            sb: usize,
            n: usize,
            bad: &mut Option<Bitmap>,
        ) -> Vec<$t> {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let x = a[i * sa];
                let y = b[i * sb];
                let v = match op {
                    OpCode::Add => x.wrapping_add(y),
                    OpCode::Sub => x.wrapping_sub(y),
                    OpCode::Mul => x.wrapping_mul(y),
                    OpCode::Neg => (0 as $t).wrapping_sub(x),
                    // Div / Mod. Division by zero and MIN / -1 make the result NULL.
                    _ => {
                        if y == 0 || (y == -1 && x == <$t>::MIN) {
                            funcs::set_null(bad, i, n);
                            0
                        } else if matches!(op, OpCode::Div) {
                            x.wrapping_div(y)
                        } else {
                            x.wrapping_rem(y)
                        }
                    }
                };
                out.push(v);
            }
            out
        }
    };
}

int_arith!(arith_i32, i32);
int_arith!(arith_i64, i64);
int_arith!(arith_i128, i128);

/// Floating point follows IEEE. Division by zero gives inf/NaN rather than NULL.
fn arith_f64(op: OpCode, a: &[f64], sa: usize, b: &[f64], sb: usize, n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let x = a[i * sa];
        let y = b[i * sb];
        out.push(match op {
            OpCode::Add => x + y,
            OpCode::Sub => x - y,
            OpCode::Mul => x * y,
            OpCode::Div => x / y,
            OpCode::Neg => -x,
            _ => x % y,
        });
    }
    out
}

/// `Add`/`Sub`/`Mul`/`Div`/`Mod`/`Neg`. For `Neg` the VM passes a as b as well.
///
/// DECIMAL is computed as a raw integer. `Add`/`Sub` are correct this way once the scales are
/// aligned, and the scale adjustment for `Mul`/`Div` is done by the binder with a `Cast`.
pub fn arith(op: OpCode, out_ty: Ty, a: &Vector, b: &Vector) -> Result<Vector> {
    ensure!(
        matches!(
            op,
            OpCode::Add | OpCode::Sub | OpCode::Mul | OpCode::Div | OpCode::Mod | OpCode::Neg
        ),
        Internal
    );
    let phys = out_ty.phys();
    ensure!(a.data().phys() == phys && b.data().phys() == phys, TypeMismatch);
    let (n, sa, sb) = strides2(a.len(), b.len())?;
    let mut bad = None;
    let data = match phys {
        PhysType::I32 => Data::I32(arith_i32(op, a.i32s(), sa, b.i32s(), sb, n, &mut bad)),
        PhysType::I64 => Data::I64(arith_i64(op, a.i64s(), sa, b.i64s(), sb, n, &mut bad)),
        PhysType::I128 => Data::I128(arith_i128(op, a.i128s(), sa, b.i128s(), sb, n, &mut bad)),
        PhysType::F64 => Data::F64(arith_f64(op, a.f64s(), sa, b.f64s(), sb, n)),
        _ => err!(TypeMismatch),
    };
    Ok(finish(out_ty, data, combine_validity(a, sa, b, sb, n), bad))
}

// --- Comparison ---------------------------------------------------------------
// A three-way comparison code (plus "unordered" for NaN) ANDed with a mask folds the six operators into one.

const C_LT: u8 = 1;
const C_EQ: u8 = 2;
const C_GT: u8 = 4;
/// The state where NaN makes the comparison unordered. Only `<>` treats it as true (per IEEE).
const C_UN: u8 = 8;

fn cmp_mask(op: OpCode) -> Result<u8> {
    Ok(match op {
        OpCode::Eq => C_EQ,
        OpCode::Ne => C_LT | C_GT | C_UN,
        OpCode::Lt => C_LT,
        OpCode::Le => C_LT | C_EQ,
        OpCode::Gt => C_GT,
        OpCode::Ge => C_GT | C_EQ,
        _ => err!(Internal),
    })
}

fn ord_code(o: core::cmp::Ordering) -> u8 {
    match o {
        core::cmp::Ordering::Less => C_LT,
        core::cmp::Ordering::Equal => C_EQ,
        core::cmp::Ordering::Greater => C_GT,
    }
}

macro_rules! int_cmp {
    ($name:ident, $t:ty) => {
        fn $name(a: &[$t], sa: usize, b: &[$t], sb: usize, n: usize, mask: u8) -> Bitmap {
            let mut out = Bitmap::with_capacity(n);
            for i in 0..n {
                let x = a[i * sa];
                let y = b[i * sb];
                let c = if x < y {
                    C_LT
                } else if x > y {
                    C_GT
                } else {
                    C_EQ
                };
                out.push(mask & c != 0);
            }
            out
        }
    };
}

int_cmp!(cmp_i32, i32);
int_cmp!(cmp_i64, i64);
int_cmp!(cmp_i128, i128);

fn cmp_f64(a: &[f64], sa: usize, b: &[f64], sb: usize, n: usize, mask: u8) -> Bitmap {
    let mut out = Bitmap::with_capacity(n);
    for i in 0..n {
        let x = a[i * sa];
        let y = b[i * sb];
        let c = if x < y {
            C_LT
        } else if x > y {
            C_GT
        } else if x == y {
            C_EQ
        } else {
            C_UN
        };
        out.push(mask & c != 0);
    }
    out
}

fn cmp_bool(a: &Bitmap, sa: usize, b: &Bitmap, sb: usize, n: usize, mask: u8) -> Bitmap {
    let mut out = Bitmap::with_capacity(n);
    for i in 0..n {
        let c = ord_code(a.get(i * sa).cmp(&b.get(i * sb)));
        out.push(mask & c != 0);
    }
    out
}

/// Byte sequences compare lexicographically (memcmp order). VARCHAR is treated the same and carries no collation.
fn cmp_bytes(a: &BytesData, sa: usize, b: &BytesData, sb: usize, n: usize, mask: u8) -> Bitmap {
    let mut out = Bitmap::with_capacity(n);
    for i in 0..n {
        let c = ord_code(a.get(i * sa).cmp(b.get(i * sb)));
        out.push(mask & c != 0);
    }
    out
}

/// The six comparisons. The input is `phys` and the output is Bool.
pub fn compare(op: OpCode, phys: PhysType, a: &Vector, b: &Vector) -> Result<Vector> {
    let mask = cmp_mask(op)?;
    ensure!(a.data().phys() == phys && b.data().phys() == phys, TypeMismatch);
    let (n, sa, sb) = strides2(a.len(), b.len())?;
    let bits = match phys {
        PhysType::Bool => cmp_bool(a.bools(), sa, b.bools(), sb, n, mask),
        PhysType::I32 => cmp_i32(a.i32s(), sa, b.i32s(), sb, n, mask),
        PhysType::I64 => cmp_i64(a.i64s(), sa, b.i64s(), sb, n, mask),
        PhysType::I128 => cmp_i128(a.i128s(), sa, b.i128s(), sb, n, mask),
        PhysType::F64 => cmp_f64(a.f64s(), sa, b.f64s(), sb, n, mask),
        PhysType::Bytes => cmp_bytes(a.bytes(), sa, b.bytes(), sb, n, mask),
    };
    Ok(finish(Ty::Boolean, Data::Bool(bits), combine_validity(a, sa, b, sb, n), None))
}

// --- Three-valued logic -------------------------------------------------------

/// `AND`/`OR`. The value and the validity have to be decided together, so it cannot share code with comparison.
///
/// - AND: TRUE when both are TRUE; FALSE when **either** is FALSE (even if the other is NULL).
/// - OR : TRUE when **either** is TRUE (even if the other is NULL); FALSE when both are FALSE.
pub fn logic(op: OpCode, a: &Vector, b: &Vector) -> Result<Vector> {
    let is_and = match op {
        OpCode::And => true,
        OpCode::Or => false,
        _ => err!(Internal),
    };
    ensure!(a.data().phys() == PhysType::Bool && b.data().phys() == PhysType::Bool, TypeMismatch);
    let (n, sa, sb) = strides2(a.len(), b.len())?;
    let (av, bv) = (a.bools(), b.bools());
    let mut vals = Bitmap::with_capacity(n);
    let mut valid = Bitmap::with_capacity(n);
    for i in 0..n {
        let (ia, ib) = (i * sa, i * sb);
        let (pa, pb) = (a.is_valid(ia), b.is_valid(ib));
        let (at, af) = (pa && av.get(ia), pa && !av.get(ia));
        let (bt, bf) = (pb && bv.get(ib), pb && !bv.get(ib));
        let (t, f) = if is_and { (at && bt, af || bf) } else { (at || bt, af && bf) };
        vals.push(t);
        // Neither TRUE nor FALSE means NULL.
        valid.push(t || f);
    }
    Ok(finish(Ty::Boolean, Data::Bool(vals), Some(valid), None))
}

/// `NOT`. NULL stays NULL.
pub fn not(a: &Vector) -> Result<Vector> {
    ensure!(a.data().phys() == PhysType::Bool, TypeMismatch);
    let mut bits = a.bools().clone();
    bits.negate();
    Ok(finish(Ty::Boolean, Data::Bool(bits), a.validity().cloned(), None))
}

/// `IsNull` / `IsNotNull`. The result is never NULL.
pub fn is_null(a: &Vector, want_null: bool) -> Vector {
    let n = a.len();
    let mut bits = Bitmap::with_capacity(n);
    match a.validity() {
        None => bits.push_n(!want_null, n),
        Some(v) => {
            for i in 0..n {
                bits.push(v.get(i) != want_null);
            }
        }
    }
    Vector::from_data(Ty::Boolean, Data::Bool(bits), None)
}

// --- Row copying --------------------------------------------------------------

/// Appends row `i` of `src` to the end of `dst`. The caller has already checked the physical types match.
fn push_row(dst: &mut Data, src: &Data, i: usize) {
    match (dst, src) {
        (Data::Bool(d), Data::Bool(s)) => d.push(s.get(i)),
        (Data::I32(d), Data::I32(s)) => d.push(s[i]),
        (Data::I64(d), Data::I64(s)) => d.push(s[i]),
        (Data::F64(d), Data::F64(s)) => d.push(s[i]),
        (Data::I128(d), Data::I128(s)) => d.push(s[i]),
        (Data::Bytes(d), Data::Bytes(s)) => d.push(s.get(i)),
        _ => debug_assert!(false, "phys type mismatch"),
    }
}

/// A dummy row (a placeholder for a NULL row).
fn push_default(dst: &mut Data) {
    match dst {
        Data::Bool(d) => d.push(false),
        Data::I32(d) => d.push(0),
        Data::I64(d) => d.push(0),
        Data::F64(d) => d.push(0.0),
        Data::I128(d) => d.push(0),
        Data::Bytes(d) => d.push_empty(),
    }
}

/// Broadcasts a length-1 vector to `n` rows. Used when returning the result of a constant-only expression.
pub fn broadcast(v: &Vector, n: usize) -> Vector {
    debug_assert_eq!(v.len(), 1);
    let mut data = Data::with_capacity(v.ty().phys(), n);
    for _ in 0..n {
        push_row(&mut data, v.data(), 0);
    }
    let validity = if v.is_valid(0) { None } else { Some(Bitmap::zeros(n)) };
    Vector::from_data(v.ty(), data, validity)
}

/// `Select` and `Coalesce`. When `cond` is `None` the condition becomes "is `t` valid"
/// (= COALESCE). A NULL or FALSE condition takes the `e` side.
pub fn pick(cond: Option<&Vector>, t: &Vector, e: &Vector, out_ty: Ty) -> Result<Vector> {
    let phys = out_ty.phys();
    ensure!(t.data().phys() == phys && e.data().phys() == phys, TypeMismatch);
    if let Some(c) = cond {
        ensure!(c.data().phys() == PhysType::Bool, TypeMismatch);
    }
    let lc = cond.map_or(t.len(), |c| c.len());
    let (n, sc, st, se) = strides3(lc, t.len(), e.len())?;
    let mut data = Data::with_capacity(phys, n);
    let mut valid = Bitmap::with_capacity(n);
    for i in 0..n {
        let take_t = match cond {
            Some(c) => c.is_valid(i * sc) && c.bools().get(i * sc),
            None => t.is_valid(i * sc),
        };
        let (src, j) = if take_t { (t, i * st) } else { (e, i * se) };
        push_row(&mut data, src.data(), j);
        valid.push(src.is_valid(j));
    }
    Ok(finish(out_ty, data, Some(valid), None))
}

// --- Bytes operations ---------------------------------------------------------

/// `a || b`. Byte-sequence concatenation.
pub fn concat(a: &Vector, b: &Vector, out_ty: Ty) -> Result<Vector> {
    ensure!(a.data().phys() == PhysType::Bytes && b.data().phys() == PhysType::Bytes, TypeMismatch);
    let (n, sa, sb) = strides2(a.len(), b.len())?;
    let (ad, bd) = (a.bytes(), b.bytes());
    let mut out = BytesData::with_capacity(n, ad.data.len() + bd.data.len());
    for i in 0..n {
        out.data.extend_from_slice(ad.get(i * sa));
        out.data.extend_from_slice(bd.get(i * sb));
        // offsets are u32. Results beyond that cannot be handled.
        ensure!(out.data.len() <= u32::MAX as usize, LimitExceeded);
        out.offsets.push(out.data.len() as u32);
    }
    Ok(finish(out_ty, Data::Bytes(out), combine_validity(a, sa, b, sb, n), None))
}

/// Length in bytes of the UTF-8 sequence starting at `s[i]` (`i < s.len()` is required),
/// clamped so it never runs past the end of `s`. A continuation byte or otherwise invalid
/// lead byte falls back to 1 -- `s`/`p` are not assumed to be valid UTF-8 here (this is the
/// same defensive posture as the rest of the untrusted-input-facing code), so this always
/// makes forward progress instead of panicking or looping forever on malformed bytes.
fn utf8_len_at(s: &[u8], i: usize) -> usize {
    let n = match s[i] {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF7 => 4,
        _ => 1,
    };
    n.min(s.len() - i)
}

/// SQL `LIKE`. `%` matches zero or more characters and `_` matches exactly one Unicode
/// character (a UTF-8 sequence, not necessarily one byte) -- matching the Unicode-codepoint
/// convention the rest of the string functions follow (`docs/sql/functions-string.md`).
///
/// `%`'s own literal/backtrack matching stays byte-oriented (cheap, and correct on its own:
/// UTF-8 is self-synchronizing, so byte-for-byte literal comparison never needs character
/// boundaries). The one place alignment actually matters is `_`, plus the backtrack step
/// that retries a `%` with "one more `_`/literal token" -- that step also has to advance a
/// full character, not one byte, or a retry could land mid-character and hand `_` a bogus
/// starting point. `ESCAPE` is unsupported (rejected at parse time, see
/// `docs/sql/limitations.md`), so there's no escape handling to keep in sync here.
///
/// Backtracking is a two-pointer method remembering only "the position of the last `%` seen".
/// It does not recurse, so it consumes no stack, and even a pattern like `%a%a%a...` costs at
/// worst O(|s| * |p|) (naive recursion would go exponential).
fn like_match(s: &[u8], p: &[u8]) -> bool {
    let (mut si, mut pi) = (0usize, 0usize);
    // `star_p == usize::MAX` means "no `%` seen yet".
    let (mut star_p, mut star_s) = (usize::MAX, 0usize);
    while si < s.len() {
        if pi < p.len() && p[pi] == b'_' {
            si += utf8_len_at(s, si);
            pi += 1;
        } else if pi < p.len() && p[pi] == s[si] {
            si += 1;
            pi += 1;
        } else if pi < p.len() && p[pi] == b'%' {
            star_p = pi;
            star_s = si;
            pi += 1;
        } else if star_p != usize::MAX {
            // Feed the previous `%` one more character (not byte -- see doc comment) and retry.
            pi = star_p + 1;
            star_s += utf8_len_at(s, star_s);
            si = star_s;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == b'%' {
        pi += 1;
    }
    pi == p.len()
}

pub fn like(a: &Vector, b: &Vector) -> Result<Vector> {
    ensure!(a.data().phys() == PhysType::Bytes && b.data().phys() == PhysType::Bytes, TypeMismatch);
    let (n, sa, sb) = strides2(a.len(), b.len())?;
    let (ad, bd) = (a.bytes(), b.bytes());
    let mut bits = Bitmap::with_capacity(n);
    for i in 0..n {
        bits.push(like_match(ad.get(i * sa), bd.get(i * sb)));
    }
    Ok(finish(Ty::Boolean, Data::Bool(bits), combine_validity(a, sa, b, sb, n), None))
}

// --- Casts --------------------------------------------------------------------

/// The family of conversions. It follows from the physical type, so no extra branching per logical type.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fam {
    Int,
    Flt,
    Str,
}

fn fam(t: Ty) -> Fam {
    match t.phys() {
        PhysType::F64 => Fam::Flt,
        PhysType::Bytes => Fam::Str,
        _ => Fam::Int,
    }
}

fn dec_scale(t: Ty) -> u8 {
    match t {
        Ty::Decimal { scale, .. } => scale,
        _ => 0,
    }
}

fn pow10_i128(k: u32) -> Option<i128> {
    if k > 38 {
        return None;
    }
    let mut r: i128 = 1;
    for _ in 0..k {
        r *= 10;
    }
    Some(r)
}

fn pow10_f64(k: u8) -> f64 {
    let mut r = 1.0f64;
    for _ in 0..k {
        r *= 10.0;
    }
    r
}

/// Rounding to the nearest even (banker's rounding). `core` has no `f64::round_ties_even`.
///
/// Casting floating point to an integer rounds rather than truncates. The SQL standard calls it
/// implementation-defined, but DuckDB and PostgreSQL both round here. Leaning to the even side
/// at exactly 0.5 matches both as well (`1.5 -> 2`, `4.5 -> 4`).
/// Rounding away from zero would systematically inflate sums over all-positive data, so
/// round-half-to-even is preferable for statistics.
fn f_round(x: f64) -> f64 {
    let t = funcs::f_trunc(x);
    let frac = x - t;
    if frac > 0.5 {
        return t + 1.0;
    }
    if frac < -0.5 {
        return t - 1.0;
    }
    if frac != 0.5 && frac != -0.5 {
        return t;
    }
    // Exactly halfway. If `t` is even it stays; if odd it moves away from zero.
    // By this point |x| is below 2^52, so converting to i64 is safe.
    if (t as i64) % 2 == 0 {
        t
    } else if frac > 0.0 {
        t + 1.0
    } else {
        t - 1.0
    }
}

fn load_i128(d: &Data, i: usize) -> i128 {
    match d {
        Data::Bool(b) => b.get(i) as i128,
        Data::I32(v) => v[i] as i128,
        Data::I64(v) => v[i] as i128,
        Data::I128(v) => v[i],
        Data::F64(v) => v[i] as i128,
        Data::Bytes(_) => 0,
    }
}

/// Writes an integer into the destination physical type. Out of range pushes a default and gives `false` (= NULL).
fn store_i128(d: &mut Data, y: i128) -> bool {
    match d {
        Data::Bool(b) => {
            b.push(y != 0);
            true
        }
        Data::I32(v) => match i32::try_from(y) {
            Ok(z) => {
                v.push(z);
                true
            }
            Err(_) => {
                v.push(0);
                false
            }
        },
        Data::I64(v) => match i64::try_from(y) {
            Ok(z) => {
                v.push(z);
                true
            }
            Err(_) => {
                v.push(0);
                false
            }
        },
        Data::I128(v) => {
            v.push(y);
            true
        }
        Data::F64(v) => {
            v.push(y as f64);
            true
        }
        Data::Bytes(b) => {
            b.push_empty();
            false
        }
    }
}

/// The conversion factors `(mul, div, floor)` between integer families.
/// `floor` is floor division (only for TIMESTAMP->DATE, so dates before the epoch are not off by one).
fn int_conv(from: Ty, to: Ty) -> Result<(i128, i128, bool)> {
    use Ty::*;
    if from.is_temporal() || to.is_temporal() {
        return Ok(match (from, to) {
            (Date, Timestamp) | (Date, Timestamptz) => (MICROS_PER_DAY, 1, false),
            (Timestamp, Date) | (Timestamptz, Date) => (1, MICROS_PER_DAY, true),
            // `Timestamp` and `Timestamptz` have exactly the same physical representation (UTC
            // microseconds), so the value passes straight through
            // (the one-sided cast when `Ty::unify` settles `Date`/`Timestamp` on `Timestamptz`
            // comes through here).
            (Timestamp, Timestamptz) | (Timestamptz, Timestamp) => (1, 1, false),
            // Temporal types and integers keep their raw values. Conversion with BOOLEAN or DECIMAL is meaningless.
            (f, t)
                if (f.is_temporal() && t.is_integer()) || (f.is_integer() && t.is_temporal()) =>
            {
                (1, 1, false)
            }
            _ => err!(InvalidCast),
        });
    }
    // DECIMAL is an integer scaled by 10^scale. Multiply/divide by the scale difference.
    let s1 = dec_scale(from) as i32;
    let s2 = dec_scale(to) as i32;
    if s2 > s1 {
        match pow10_i128((s2 - s1) as u32) {
            Some(p) => Ok((p, 1, false)),
            None => err!(InvalidCast),
        }
    } else if s1 > s2 {
        match pow10_i128((s1 - s2) as u32) {
            Some(p) => Ok((1, p, false)),
            None => err!(InvalidCast),
        }
    } else {
        Ok((1, 1, false))
    }
}

fn rescale_i128(x: i128, mul: i128, div: i128, floor: bool) -> Option<i128> {
    let mut y = x;
    if mul != 1 {
        y = y.checked_mul(mul)?;
    }
    if div != 1 {
        let q = y / div;
        let r = y % div;
        y = if floor {
            // TIMESTAMP -> DATE must be a floor. Truncating would shift pre-epoch values by a day.
            if r != 0 && (y < 0) != (div < 0) {
                q - 1
            } else {
                q
            }
        } else {
            // Reducing DECIMAL scale rounds away from zero (the same as DuckDB). Truncating would
            // systematically underestimate in monetary computation.
            // The comparison is against half rather than `r * 2 >= div` because `r * 2` could
            // overflow i128.
            let half = (div + 1) / 2;
            if r >= half {
                q + 1
            } else if r <= -half {
                q - 1
            } else {
                q
            }
        };
    }
    Some(y)
}

/// Renders an unsigned magnitude plus a scale as decimal. `format!` is unavailable (size).
pub(crate) fn fmt_int(mut u: u128, neg: bool, scale: u8, out: &mut Vec<u8>) {
    let mut buf = [0u8; 48];
    let mut k = 0usize;
    if u == 0 {
        buf[0] = b'0';
        k = 1;
    }
    while u > 0 {
        buf[k] = b'0' + (u % 10) as u8;
        u /= 10;
        k += 1;
    }
    // Supplies the leading 0 when there is no integer part, as with 0.05.
    while k <= scale as usize {
        buf[k] = b'0';
        k += 1;
    }
    if neg {
        out.push(b'-');
    }
    for i in (0..k).rev() {
        out.push(buf[i]);
        if scale > 0 && i == scale as usize {
            out.push(b'.');
        }
    }
}

/// The decimal rendering of an f64. The shortest round-trip representation (Ryu/Grisu) is too
/// heavy in size, so this is an approximate rendering rounded to 15 digits with trailing zeros dropped.
pub(crate) fn fmt_f64(x: f64, out: &mut Vec<u8>) {
    if x.is_nan() {
        out.extend_from_slice(b"NaN");
        return;
    }
    if x < 0.0 {
        out.push(b'-');
    }
    let v = funcs::f_abs(x);
    if v == f64::INFINITY {
        out.extend_from_slice(b"Inf");
        return;
    }
    if v == 0.0 {
        out.push(b'0');
        return;
    }
    // Normalizes m into [1e14, 1e15) while preserving v = m * 10^e10.
    let mut m = v;
    let mut e10: i32 = 0;
    while m >= 1e15 {
        if m >= 1e31 {
            m /= 1e16;
            e10 += 16;
        } else {
            m /= 10.0;
            e10 += 1;
        }
    }
    while m < 1e14 {
        if m < 1e-2 {
            m *= 1e16;
            e10 -= 16;
        } else {
            m *= 10.0;
            e10 -= 1;
        }
    }
    let mut mant = (m + 0.5) as u64;
    if mant >= 1_000_000_000_000_000 {
        mant /= 10;
        e10 += 1;
    }
    let mut digits = [0u8; 15];
    let mut t = mant;
    for i in (0..15).rev() {
        digits[i] = b'0' + (t % 10) as u8;
        t /= 10;
    }
    let mut k = 15usize;
    while k > 1 && digits[k - 1] == b'0' {
        k -= 1;
    }
    // v = 0.d1..dk × 10^p
    let p = e10 + 15;
    if !(-3..=17).contains(&p) {
        // Exponential notation.
        out.push(digits[0]);
        if k > 1 {
            out.push(b'.');
            out.extend_from_slice(&digits[1..k]);
        }
        out.push(b'e');
        let e = p - 1;
        if e < 0 {
            out.push(b'-');
        }
        fmt_int((if e < 0 { -(e as i64) } else { e as i64 }) as u128, false, 0, out);
    } else if p <= 0 {
        out.extend_from_slice(b"0.");
        for _ in 0..(-p) {
            out.push(b'0');
        }
        out.extend_from_slice(&digits[..k]);
    } else if (p as usize) >= k {
        out.extend_from_slice(&digits[..k]);
        for _ in 0..(p as usize - k) {
            out.push(b'0');
        }
    } else {
        out.extend_from_slice(&digits[..p as usize]);
        out.push(b'.');
        out.extend_from_slice(&digits[p as usize..k]);
    }
}

/// Reads a decimal number as `mant * 10^exp`. `None` (= NULL) if it cannot be read.
///
/// The third return value is "whether integer digits were dropped because they did not fit the
/// mantissa". When they were, `mant * 10^exp` is a rounded version of the original, so integer
/// casts that need exactness consult it and fall to NULL (returning the rounded value would turn
/// `CAST('...105727' AS HUGEINT)` into `...105720`). Floating point has only mantissa precision to
/// begin with, so it can ignore this.
///
/// The mantissa is **accumulated on the negative side**. `i128::MIN`'s magnitude is not
/// representable as a positive `i128`, so accumulating positively would make exactly the lower
/// bound (`-170141183460469231731687303715884105728`) unreadable.
fn parse_dec(s: &[u8]) -> Option<(i128, i32, bool)> {
    let mut lo = 0usize;
    let mut hi = s.len();
    while lo < hi && (s[lo] == b' ' || s[lo] == b'\t') {
        lo += 1;
    }
    while hi > lo && (s[hi - 1] == b' ' || s[hi - 1] == b'\t') {
        hi -= 1;
    }
    let s = &s[lo..hi];
    let mut i = 0usize;
    let mut neg = false;
    if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
        neg = s[i] == b'-';
        i += 1;
    }
    // Accumulated on the negative side (see the function's docs).
    let mut mant: i128 = 0;
    let mut exp: i32 = 0;
    let mut inexact = false;
    let mut seen = false;
    while i < s.len() && s[i].is_ascii_digit() {
        seen = true;
        let d = (s[i] - b'0') as i128;
        match mant.checked_mul(10).and_then(|m| m.checked_sub(d)) {
            Some(m) => mant = m,
            // Digits that do not fit the mantissa escape into the exponent (precision is lost, hence inexact).
            None => {
                exp += 1;
                inexact = true;
            }
        }
        i += 1;
    }
    if i < s.len() && s[i] == b'.' {
        i += 1;
        while i < s.len() && s[i].is_ascii_digit() {
            seen = true;
            let d = (s[i] - b'0') as i128;
            // Dropping trailing fractional digits does not change the value as an integer, so
            // inexact is not set here (it does not affect the integer cast's result).
            if let Some(m) = mant.checked_mul(10).and_then(|m| m.checked_sub(d)) {
                mant = m;
                exp -= 1;
            }
            i += 1;
        }
    }
    if !seen {
        return None;
    }
    if i < s.len() && (s[i] == b'e' || s[i] == b'E') {
        i += 1;
        let mut eneg = false;
        if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
            eneg = s[i] == b'-';
            i += 1;
        }
        let mut e: i32 = 0;
        let mut any = false;
        while i < s.len() && s[i].is_ascii_digit() {
            any = true;
            if e < 100_000 {
                e = e * 10 + (s[i] - b'0') as i32;
            }
            i += 1;
        }
        if !any {
            return None;
        }
        exp += if eneg { -e } else { e };
    }
    if i != s.len() {
        return None;
    }
    let mant = if neg {
        mant
    } else {
        match mant.checked_neg() {
            Some(m) => m,
            // Exactly `+|i128::MIN|`. It does not fit on the positive side, so exactly one digit
            // escapes into the exponent (being inexact, the integer cast becomes NULL, and for
            // floating point it makes no difference at f64's precision).
            None => {
                inexact = true;
                exp += 1;
                -(mant / 10)
            }
        }
    };
    Some((mant, exp, inexact))
}

/// `m * 10^e`. Multiplied via binary decomposition, so the error is smaller than multiplying by 10 e times.
fn scale_f64(mut m: f64, e: i32) -> f64 {
    const TAB: [f64; 9] = [1e1, 1e2, 1e4, 1e8, 1e16, 1e32, 1e64, 1e128, 1e256];
    let neg = e < 0;
    let mut k = if neg { -e } else { e };
    if k > 400 {
        k = 400;
    }
    for (j, p) in TAB.iter().enumerate() {
        if (k >> j) & 1 == 1 {
            if neg {
                m /= *p;
            } else {
                m *= *p;
            }
        }
    }
    m
}

fn parse_bool(s: &[u8]) -> Option<bool> {
    let eq =
        |w: &[u8]| s.len() == w.len() && s.iter().zip(w).all(|(a, b)| a.to_ascii_lowercase() == *b);
    if eq(b"true") || eq(b"t") {
        Some(true)
    } else if eq(b"false") || eq(b"f") {
        Some(false)
    } else {
        None
    }
}

/// `VARCHAR -> JSON`. Each row is validated with `crate::json::validate`.
///
/// Only here does the handling of a per-row failure differ between `CAST` and `TRY_CAST`:
/// `lenient == false` (an ordinary `CAST`) makes invalid JSON an error (matching DuckDB's measured
/// behavior where `CAST('not json' AS JSON)` errors). `lenient == true` (`TRY_CAST`) makes just
/// that row NULL (aligning with other types' TRY_CAST and with per-row parse failures).
/// Casts in general are designed so that "a per-row failure is always NULL, with no difference
/// between CAST and TRY_CAST", but JSON has only the binary question of whether the whole document
/// is broken, so this alone is made an exception to match DuckDB's actual behavior.
fn cast_str_to_json(a: &Vector, lenient: bool) -> Result<Vector> {
    let n = a.len();
    let sv = a.bytes();
    let mut out = BytesData::with_capacity(n, sv.data.len());
    let mut bad: Option<Bitmap> = None;
    for i in 0..n {
        if !a.is_valid(i) {
            out.push_empty();
            continue;
        }
        let s = sv.get(i);
        if crate::json::validate(s).is_ok() {
            out.push(s);
        } else if lenient {
            out.push_empty();
            funcs::set_null(&mut bad, i, n);
        } else {
            err!(InvalidCast);
        }
    }
    Ok(finish(Ty::Json, Data::Bytes(out), a.validity().cloned(), bad))
}

/// `VARCHAR -> UUID`. A per-row parse failure makes just that row NULL, by the same convention as
/// `DATE`/`TIME`/`TIMESTAMP` (for both `CAST` and `TRY_CAST`, regardless of `lenient`).
/// Only `VARCHAR -> JSON` is an exception to that convention (see the `cast_str_to_json` docs),
/// and UUID does not follow it.
fn cast_str_to_uuid(a: &Vector) -> Result<Vector> {
    let n = a.len();
    let sv = a.bytes();
    let mut out = BytesData::with_capacity(n, n * 16);
    let mut bad: Option<Bitmap> = None;
    for i in 0..n {
        if !a.is_valid(i) {
            out.push_empty();
            continue;
        }
        match funcs::parse_uuid(sv.get(i)) {
            Some(bytes) => out.push(&bytes),
            None => {
                out.push_empty();
                funcs::set_null(&mut bad, i, n);
            }
        }
    }
    Ok(finish(Ty::Uuid, Data::Bytes(out), a.validity().cloned(), bad))
}

/// `Cast`. Unimplemented combinations give `InvalidCast` rather than silently returning a broken value.
/// A per-row conversion failure (out of range, unparsable) is not an error; just that row becomes NULL.
///
/// `VARCHAR -> JSON` alone is the exception: a per-row failure (not valid JSON) becomes NULL only
/// under `TRY_CAST` (unlike other types, an ordinary `CAST` errors. A deliberate exception matching
/// DuckDB's actual behavior; see the docs on [`try_cast`] for details).
pub fn cast(from: Ty, to: Ty, a: &Vector) -> Result<Vector> {
    cast_impl(from, to, a, false)
}

/// `TRY_CAST`. For most types it is exactly [`cast`] (a per-row conversion failure is contracted to
/// be NULL rather than an error in `cast` anyway). The only difference is `VARCHAR -> JSON`'s
/// per-row validation: `CAST` errors on invalid JSON while `TRY_CAST` makes just that row NULL
/// (when the type pair itself is unsupported, `expr::vm` still catches it and falls to all-NULL as before).
pub fn try_cast(from: Ty, to: Ty, a: &Vector) -> Result<Vector> {
    cast_impl(from, to, a, true)
}

fn cast_impl(from: Ty, to: Ty, a: &Vector, lenient: bool) -> Result<Vector> {
    ensure!(a.data().phys() == from.phys(), TypeMismatch);
    let n = a.len();
    if from == to {
        return Ok(a.clone());
    }
    if from == Ty::Null {
        // A NULL literal with no settled type. Every row is NULL, so no conversion rule is needed.
        let mut out = Vector::new(to);
        for _ in 0..n {
            out.push_null();
        }
        return Ok(out);
    }
    // Under `fam()` JSON would collapse into the same Bytes family as VARCHAR/BLOB, so it is
    // handled separately before reaching the general (Fam, Fam) match. Only VARCHAR <-> JSON is
    // supported (casting from BLOB, numbers, or temporals stays unsupported and gives
    // `InvalidCast`; the design decision is that `to_json` should be used instead).
    if to == Ty::Json {
        ensure!(from == Ty::Varchar, InvalidCast);
        return cast_str_to_json(a, lenient);
    }
    if from == Ty::Json {
        ensure!(to == Ty::Varchar, InvalidCast);
        let mut out = a.clone();
        out.retype(to);
        return Ok(out);
    }
    // UUID is handled separately for the same reason: only VARCHAR <-> UUID is supported.
    // Its physical representation is `Bytes`, the same as VARCHAR/BLOB, but UUID's text form
    // (hyphenated hex) differs from the raw bytes, so it cannot ride the generic `(Fam::Str,
    // Fam::Str)` straight copy (`BLOB <-> UUID` stays unsupported and gives `InvalidCast`; the
    // design decision is that if you want raw bytes you should use `UUID` directly rather than
    // going through `BLOB`).
    if to == Ty::Uuid {
        ensure!(from == Ty::Varchar, InvalidCast);
        return cast_str_to_uuid(a);
    }
    if from == Ty::Uuid {
        ensure!(to == Ty::Varchar, InvalidCast);
        let sv = a.bytes();
        let mut data = Data::with_capacity(PhysType::Bytes, n);
        let mut buf = Vec::new();
        if let Data::Bytes(d) = &mut data {
            for i in 0..n {
                buf.clear();
                if let Ok(raw) = <[u8; 16]>::try_from(sv.get(i)) {
                    funcs::fmt_uuid(&raw, &mut buf);
                }
                d.push(&buf);
            }
        }
        return Ok(finish(to, data, a.validity().cloned(), None));
    }
    let src = a.data();
    let mut data = Data::with_capacity(to.phys(), n);
    let mut bad: Option<Bitmap> = None;
    match (fam(from), fam(to)) {
        (Fam::Int, Fam::Int) => {
            let (mul, div, floor) = int_conv(from, to)?;
            for i in 0..n {
                let ok = match rescale_i128(load_i128(src, i), mul, div, floor) {
                    Some(y) => store_i128(&mut data, y),
                    None => {
                        push_default(&mut data);
                        false
                    }
                };
                if !ok {
                    funcs::set_null(&mut bad, i, n);
                }
            }
        }
        (Fam::Int, Fam::Flt) => {
            ensure!(!from.is_temporal(), InvalidCast);
            let s = dec_scale(from);
            for i in 0..n {
                let mut f = load_i128(src, i) as f64;
                if s > 0 {
                    f /= pow10_f64(s);
                }
                if to == Ty::Float {
                    f = f as f32 as f64;
                }
                push_f64(&mut data, f);
            }
        }
        (Fam::Flt, Fam::Int) => {
            ensure!(!to.is_temporal(), InvalidCast);
            let s = dec_scale(to);
            let sv = a.f64s();
            // The index is used not only to fetch the value but also to name the row for `set_null`.
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let mut f = sv[i];
                if s > 0 {
                    f *= pow10_f64(s);
                }
                let ok = if f.is_finite() && (-I128_LIMIT..I128_LIMIT).contains(&f) {
                    store_i128(&mut data, f_round(f) as i128)
                } else {
                    push_default(&mut data);
                    false
                };
                if !ok {
                    funcs::set_null(&mut bad, i, n);
                }
            }
        }
        (Fam::Flt, Fam::Flt) => {
            let sv = a.f64s();
            // `n` is the row count after strides are applied and is not necessarily `sv.len()`.
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                // DOUBLE -> FLOAT drops to f32 precision (while still stored as f64).
                let f = if to == Ty::Float { sv[i] as f32 as f64 } else { sv[i] };
                push_f64(&mut data, f);
            }
        }
        (Fam::Int, Fam::Str) => {
            let scale = dec_scale(from);
            let is_bool = from == Ty::Boolean;
            let mut buf = Vec::new();
            if let Data::Bytes(d) = &mut data {
                for i in 0..n {
                    let x = load_i128(src, i);
                    buf.clear();
                    if is_bool {
                        buf.extend_from_slice(if x != 0 { &b"true"[..] } else { &b"false"[..] });
                    } else if from.is_temporal() {
                        // Date formatting lives on the funcs side (it pairs with the parser).
                        let y = x as i64;
                        match from {
                            Ty::Date => funcs::fmt_date(y, &mut buf),
                            Ty::Time => funcs::fmt_time(y, &mut buf),
                            Ty::Timestamptz => funcs::fmt_timestamptz(y, &mut buf),
                            _ => funcs::fmt_timestamp(y, &mut buf),
                        }
                    } else {
                        fmt_int(x.unsigned_abs(), x < 0, scale, &mut buf);
                    }
                    d.push(&buf);
                }
            }
        }
        (Fam::Flt, Fam::Str) => {
            let sv = a.f64s();
            let mut buf = Vec::new();
            if let Data::Bytes(d) = &mut data {
                #[allow(clippy::needless_range_loop)]
                for i in 0..n {
                    buf.clear();
                    fmt_f64(sv[i], &mut buf);
                    d.push(&buf);
                }
            }
        }
        (Fam::Str, Fam::Str) => {
            // VARCHAR <-> BLOB. The representation is the same, so it is copied as is.
            let mut out = a.clone();
            out.retype(to);
            return Ok(out);
        }
        (Fam::Str, Fam::Flt) => {
            let sv = a.bytes();
            for i in 0..n {
                match parse_dec(sv.get(i)) {
                    // Digits overflowing the mantissa are outside f64's precision and can be ignored.
                    Some((m, e, _)) => {
                        let mut f = scale_f64(m as f64, e);
                        if to == Ty::Float {
                            f = f as f32 as f64;
                        }
                        push_f64(&mut data, f);
                    }
                    None => {
                        push_default(&mut data);
                        funcs::set_null(&mut bad, i, n);
                    }
                }
            }
        }
        (Fam::Str, Fam::Int) if to.is_temporal() => {
            // An unreadable string is not an error; just that row becomes NULL (as with numeric parsing).
            let sv = a.bytes();
            for i in 0..n {
                let b = sv.get(i);
                let v = match to {
                    Ty::Date => funcs::parse_date(b),
                    Ty::Time => funcs::parse_time(b),
                    Ty::Timestamptz => funcs::parse_timestamptz(b),
                    _ => funcs::parse_timestamp(b),
                };
                let ok = match v {
                    Some(y) => store_i128(&mut data, y as i128),
                    None => {
                        push_default(&mut data);
                        false
                    }
                };
                if !ok {
                    funcs::set_null(&mut bad, i, n);
                }
            }
        }
        (Fam::Str, Fam::Int) => {
            let scale = dec_scale(to) as i32;
            let is_bool = to == Ty::Boolean;
            let sv = a.bytes();
            for i in 0..n {
                let b = sv.get(i);
                let mut ok = false;
                if is_bool {
                    if let Some(v) = parse_bool(b) {
                        ok = store_i128(&mut data, v as i128);
                    }
                }
                if !ok {
                    // For BOOLEAN too, anything but 'true'/'false' is read as a number and counts as true when non-zero.
                    ok = match parse_dec(b) {
                        Some((m, e, inexact)) => {
                            let k = e + scale;
                            // If integer digits were dropped, only a rounded value exists.
                            // It is treated like out of range and becomes NULL (returning the rounded
                            // value would silently mangle the digits). DuckDB errors under CAST and
                            // gives NULL under TRY_CAST. This engine always takes the NULL side.
                            let y = if inexact {
                                None
                            } else if k >= 0 {
                                pow10_i128(k as u32).and_then(|p| m.checked_mul(p))
                            } else if -k > 38 {
                                Some(0)
                            } else {
                                pow10_i128((-k) as u32).map(|p| m / p)
                            };
                            match y {
                                Some(y) => store_i128(&mut data, y),
                                None => {
                                    push_default(&mut data);
                                    false
                                }
                            }
                        }
                        None => {
                            push_default(&mut data);
                            false
                        }
                    };
                }
                if !ok {
                    funcs::set_null(&mut bad, i, n);
                }
            }
        }
    }
    Ok(finish(to, data, a.validity().cloned(), bad))
}

fn push_f64(d: &mut Data, f: f64) {
    if let Data::F64(v) = d {
        v.push(f);
    }
}

// --- INTERVAL operations ------------------------------------------------------
// All three physical types are I128-family, but a carry must never cross a field boundary
// (`1 month + 3 days` must not become `4 ...`), which raw i128 binary arithmetic cannot express.
// That is why these are dedicated kernels.

/// TIMESTAMP(I64, microseconds) + INTERVAL(I128). DATE is cast to TIMESTAMP by the caller before
/// being passed in (DuckDB also returns TIMESTAMP for DATE +- INTERVAL).
pub fn ts_add_interval(a: &Vector, b: &Vector) -> Result<Vector> {
    ensure!(a.data().phys() == PhysType::I64 && b.data().phys() == PhysType::I128, TypeMismatch);
    let (n, sa, sb) = strides2(a.len(), b.len())?;
    let (av, bv) = (a.i64s(), b.i128s());
    let mut out = Vec::with_capacity(n);
    let mut bad = None;
    for i in 0..n {
        let (months, days, micros) = unpack_interval(bv[i * sb]);
        match funcs::add_interval_to_ts(av[i * sa], months, days, micros) {
            Some(v) => out.push(v),
            None => {
                out.push(0);
                funcs::set_null(&mut bad, i, n);
            }
        }
    }
    Ok(finish(Ty::Timestamp, Data::I64(out), combine_validity(a, sa, b, sb, n), bad))
}

/// INTERVAL +- INTERVAL. Field-wise addition (no carrying; DuckDB likewise leaves
/// `1 month + 3 days` as `1 month 3 days` without normalizing).
pub fn interval_add(a: &Vector, b: &Vector) -> Result<Vector> {
    ensure!(a.data().phys() == PhysType::I128 && b.data().phys() == PhysType::I128, TypeMismatch);
    let (n, sa, sb) = strides2(a.len(), b.len())?;
    let (av, bv) = (a.i128s(), b.i128s());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let (m1, d1, u1) = unpack_interval(av[i * sa]);
        let (m2, d2, u2) = unpack_interval(bv[i * sb]);
        out.push(pack_interval(m1.wrapping_add(m2), d1.wrapping_add(d2), u1.wrapping_add(u2)));
    }
    Ok(finish(Ty::Interval, Data::I128(out), combine_validity(a, sa, b, sb, n), None))
}

/// Negating an INTERVAL. Done field-wise (negating the raw 128-bit two's complement would break
/// across field boundaries and cannot be used).
pub fn interval_neg(a: &Vector) -> Result<Vector> {
    ensure!(a.data().phys() == PhysType::I128, TypeMismatch);
    let mut out = Vec::with_capacity(a.len());
    for &packed in a.i128s() {
        let (m, d, u) = unpack_interval(packed);
        out.push(pack_interval(m.wrapping_neg(), d.wrapping_neg(), u.wrapping_neg()));
    }
    Ok(finish(Ty::Interval, Data::I128(out), a.validity().cloned(), None))
}

/// INTERVAL * BIGINT. Field-wise multiplication (no carrying; the same as DuckDB).
pub fn interval_mul(a: &Vector, b: &Vector) -> Result<Vector> {
    ensure!(a.data().phys() == PhysType::I128 && b.data().phys() == PhysType::I64, TypeMismatch);
    let (n, sa, sb) = strides2(a.len(), b.len())?;
    let (av, bv) = (a.i128s(), b.i64s());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let (m, d, u) = unpack_interval(av[i * sa]);
        let k = bv[i * sb];
        let m = (m as i64).wrapping_mul(k) as i32;
        let d = (d as i64).wrapping_mul(k) as i32;
        out.push(pack_interval(m, d, u.wrapping_mul(k)));
    }
    Ok(finish(Ty::Interval, Data::I128(out), combine_validity(a, sa, b, sb, n), None))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[u8]) -> String {
        String::from_utf8(v.to_vec()).unwrap()
    }

    #[test]
    fn like_basics() {
        assert!(like_match(b"abc", b"abc"));
        assert!(!like_match(b"abc", b"abd"));
        assert!(like_match(b"abc", b"a%"));
        assert!(like_match(b"abc", b"%c"));
        assert!(like_match(b"abc", b"a%c"));
        assert!(like_match(b"abc", b"%b%"));
        assert!(like_match(b"abc", b"a_c"));
        assert!(!like_match(b"abc", b"a_"));
        assert!(like_match(b"abc", b"%%%"));
        assert!(like_match(b"", b"%"));
        assert!(like_match(b"", b""));
        assert!(!like_match(b"", b"_"));
        assert!(!like_match(b"abc", b""));
    }

    // Regression test: `_` used to advance by one UTF-8 *byte* instead of one Unicode
    // character, so `'あ' LIKE '_'` (a 3-byte character) returned false. `_` should match
    // exactly one character, matching DuckDB and this project's own documented convention
    // that string matching is Unicode-codepoint-based (docs/sql/functions-string.md).
    #[test]
    fn like_underscore_matches_one_unicode_character() {
        assert!(like_match("あ".as_bytes(), b"_"));
        assert!(!like_match("あい".as_bytes(), b"_"));
        assert!(like_match("あい".as_bytes(), b"__"));
        assert!(like_match("あいう".as_bytes(), "あ_う".as_bytes()));
        // Mixed ASCII / multibyte.
        assert!(like_match("aあb".as_bytes(), b"a_b"));
        assert!(like_match("aあb".as_bytes(), b"___"));
        assert!(!like_match("aあb".as_bytes(), b"a_"));
        // 4-byte character (emoji, outside the BMP).
        assert!(like_match("😀".as_bytes(), b"_"));
        assert!(like_match("a😀b".as_bytes(), b"a_b"));
        assert!(!like_match("😀😀".as_bytes(), b"_"));
        // `_` combined with `%` backtracking must stay character-aligned through the retry
        // step too, not just on the first (non-backtracked) attempt.
        assert!(like_match("aあx".as_bytes(), b"%_x"));
        assert!(!like_match("あx".as_bytes(), b"%__x")); // only one char precedes 'x'
                                                         // `%` itself is still fine matching byte-wise (any sequence of characters).
        assert!(like_match("あいう".as_bytes(), b"%"));
        assert!(like_match("あいう".as_bytes(), "%う".as_bytes()));
        assert!(like_match("あいう".as_bytes(), "あ%".as_bytes()));
    }

    #[test]
    fn like_pathological_is_not_exponential() {
        // A pattern that would go exponential under a naive recursive implementation. Linear backtracking finishes instantly.
        let subject = vec![b'a'; 4096];
        assert!(!like_match(&subject, b"%a%a%a%a%a%a%a%b"));
        let mut ok = subject.clone();
        ok.push(b'b');
        assert!(like_match(&ok, b"%a%a%a%a%a%a%a%b"));
    }

    #[test]
    fn fmt_int_inserts_decimal_point() {
        let mut o = Vec::new();
        fmt_int(12345, false, 2, &mut o);
        assert_eq!(s(&o), "123.45");
        o.clear();
        fmt_int(5, true, 2, &mut o);
        assert_eq!(s(&o), "-0.05");
        o.clear();
        fmt_int(0, false, 0, &mut o);
        assert_eq!(s(&o), "0");
        o.clear();
        fmt_int(i128::MIN.unsigned_abs(), true, 0, &mut o);
        assert_eq!(s(&o), "-170141183460469231731687303715884105728");
    }

    #[test]
    fn fmt_f64_shapes() {
        let f = |x: f64| {
            let mut o = Vec::new();
            fmt_f64(x, &mut o);
            s(&o)
        };
        assert_eq!(f(0.0), "0");
        assert_eq!(f(1.5), "1.5");
        assert_eq!(f(-2.25), "-2.25");
        assert_eq!(f(100.0), "100");
        assert_eq!(f(0.5), "0.5");
        assert_eq!(f(0.001), "0.001");
        assert_eq!(f(f64::INFINITY), "Inf");
        assert_eq!(f(f64::NEG_INFINITY), "-Inf");
        assert_eq!(f(f64::NAN), "NaN");
        assert_eq!(f(1e20), "1e20");
        assert_eq!(f(1.5e-8), "1.5e-8");
    }

    #[test]
    fn parse_dec_forms() {
        assert_eq!(parse_dec(b"123"), Some((123, 0, false)));
        assert_eq!(parse_dec(b" -12.5 "), Some((-125, -1, false)));
        assert_eq!(parse_dec(b"+1e3"), Some((1, 3, false)));
        assert_eq!(parse_dec(b".5"), Some((5, -1, false)));
        assert_eq!(parse_dec(b"1E-2"), Some((1, -2, false)));
        assert_eq!(parse_dec(b""), None);
        assert_eq!(parse_dec(b"abc"), None);
        assert_eq!(parse_dec(b"1.2.3"), None);
        assert_eq!(parse_dec(b"1e"), None);
    }

    /// The i128 extremes. Since the mantissa accumulates on the negative side, even the exact lower
    /// bound reads correctly. Back when 39 digits were truncated at 38, the upper bound rounded to `...105720`.
    #[test]
    fn parse_dec_i128_boundaries() {
        assert_eq!(
            parse_dec(b"170141183460469231731687303715884105727"),
            Some((i128::MAX, 0, false))
        );
        assert_eq!(
            parse_dec(b"-170141183460469231731687303715884105728"),
            Some((i128::MIN, 0, false))
        );
        // Upper + 1 / lower - 1 do not fit the mantissa. A rounded value comes back, but inexact is
        // set and the integer cast side consults it and gives NULL.
        let (_, _, inexact) = parse_dec(b"170141183460469231731687303715884105728").unwrap();
        assert!(inexact);
        let (_, _, inexact) = parse_dec(b"-170141183460469231731687303715884105729").unwrap();
        assert!(inexact);
        // Up to 38 digits it was exact all along.
        assert_eq!(
            parse_dec(b"12345678901234567890123456789012345678"),
            Some((12345678901234567890123456789012345678, 0, false))
        );
        // Dropping the fractional part does not affect the value as an integer, so it is not inexact.
        let long_frac = b"1.000000000000000000000000000000000000000000000005";
        assert_eq!(parse_dec(long_frac).map(|(_, _, x)| x), Some(false));
    }

    // --- INTERVAL -------------------------------------------------------------

    fn ivec(vals: &[(i32, i32, i64)]) -> Vector {
        let mut v = Vector::new(Ty::Interval);
        for &(m, d, u) in vals {
            v.push_value(&crate::vector::Value::I128(pack_interval(m, d, u)));
        }
        v
    }

    fn tsvec(vals: &[i64]) -> Vector {
        let mut v = Vector::new(Ty::Timestamp);
        for &x in vals {
            v.push_value(&crate::vector::Value::I64(x));
        }
        v
    }

    #[test]
    fn ts_add_interval_is_calendar_aware() {
        // 2024-01-31 00:00:00 + 1 month -> 2024-02-29 (a leap year, clamped to month end).
        let jan31 = funcs::days_from_civil(2024, 1, 31) * 86_400_000_000;
        let a = tsvec(&[jan31]);
        let b = ivec(&[(1, 0, 0)]);
        let r = ts_add_interval(&a, &b).unwrap();
        assert_eq!(r.ty(), Ty::Timestamp);
        assert_eq!(r.i64s()[0], funcs::days_from_civil(2024, 2, 29) * 86_400_000_000);
    }

    #[test]
    fn interval_add_sub_do_not_carry_across_fields() {
        let a = ivec(&[(1, 0, 0)]);
        let b = ivec(&[(0, 3, 0)]);
        let r = interval_add(&a, &b).unwrap();
        assert_eq!(unpack_interval(r.i128s()[0]), (1, 3, 0));

        let neg = interval_neg(&b).unwrap();
        assert_eq!(unpack_interval(neg.i128s()[0]), (0, -3, 0));
    }

    #[test]
    fn interval_mul_scales_every_field() {
        let a = ivec(&[(1, 3, 3_600_000_000)]);
        let mut k = Vector::new(Ty::BigInt);
        k.push_value(&crate::vector::Value::I64(2));
        let r = interval_mul(&a, &k).unwrap();
        assert_eq!(unpack_interval(r.i128s()[0]), (2, 6, 7_200_000_000));
    }

    // The design decision "integers wrap; overflow does not panic" (see the `int_arith!` comment at
    // the top of this file) applies consistently to INTERVAL's field-wise operations too.
    // The wrapping behavior at this boundary is pinned down here.
    #[test]
    fn interval_neg_of_i32_min_stays_negative_due_to_two_s_complement_wraparound() {
        // i32::MIN.wrapping_neg() == i32::MIN (a positive i32::MAX+1 is not representable).
        // The same intended wrapping behavior as the ordinary integer Neg kernel.
        let a = ivec(&[(i32::MIN, i32::MIN, i64::MIN)]);
        let r = interval_neg(&a).unwrap();
        assert_eq!(unpack_interval(r.i128s()[0]), (i32::MIN, i32::MIN, i64::MIN));
    }

    #[test]
    fn interval_add_wraps_on_months_and_days_overflow() {
        let a = ivec(&[(i32::MAX, i32::MAX, 0)]);
        let b = ivec(&[(1, 1, 0)]);
        let r = interval_add(&a, &b).unwrap();
        assert_eq!(unpack_interval(r.i128s()[0]), (i32::MIN, i32::MIN, 0));
    }

    #[test]
    fn interval_mul_wraps_without_double_truncation_of_the_multiplier() {
        // The multiplication happens at i64 intermediate precision and is then truncated to i32
        // (`(m as i64).wrapping_mul(k) as i32`). k itself is not truncated to i32 before
        // multiplying, so even for large k the low 32 bits of the final result are consistently the
        // same value (there is no double truncation).
        let a = ivec(&[(1_000_000, 0, 0)]);
        let mut k = Vector::new(Ty::BigInt);
        k.push_value(&crate::vector::Value::I64(10_000));
        let r = interval_mul(&a, &k).unwrap();
        let expect_months = ((1_000_000i64).wrapping_mul(10_000) as i32, 0, 0);
        assert_eq!(unpack_interval(r.i128s()[0]), expect_months);
    }

    #[test]
    fn interval_mul_wraps_on_micros_overflow() {
        let a = ivec(&[(0, 0, i64::MAX)]);
        let mut k = Vector::new(Ty::BigInt);
        k.push_value(&crate::vector::Value::I64(2));
        let r = interval_mul(&a, &k).unwrap();
        assert_eq!(unpack_interval(r.i128s()[0]), (0, 0, i64::MAX.wrapping_mul(2)));
    }

    #[test]
    fn ts_add_interval_propagates_nulls() {
        let mut a = Vector::new(Ty::Timestamp);
        a.push_null();
        a.push_value(&crate::vector::Value::I64(0));
        let b = ivec(&[(0, 1, 0), (0, 1, 0)]);
        let r = ts_add_interval(&a, &b).unwrap();
        assert!(!r.is_valid(0));
        assert!(r.is_valid(1));
    }
}
