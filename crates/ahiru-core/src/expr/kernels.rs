//! 物理型ごとの実行カーネル。
//!
//! カーネルは**物理型 6 種に対してのみ**書く（DESIGN.md §8, §11）。論理型ごとに
//! カーネルを持つと単相化爆発で 1MiB 予算が飛ぶ。サイズを抑える具体策は 4 つ:
//!
//! 1. **定数専用カーネルを作らない。** 定数は長さ 1 のベクタとして表し、
//!    オペランドごとの stride（長さ 1 なら 0、それ以外は 1）をループの外で決める。
//!    これで vec-vec / vec-const / const-vec / const-const の 4 通りが 1 本に畳まれる。
//! 2. **比較 6 種を 1 本に畳む。** 3 値比較（+ NaN 用の「順序なし」）をビットで表し、
//!    演算子ごとのマスクと AND を取る。物理型 1 種につき比較カーネルは 1 本。
//! 3. **selection は見ない。** VM が `LoadCol` で gather 済みなので、カーネルが
//!    受け取るベクタは常に密。selection の有無を型パラメータにしない。
//! 4. **算術は物理型ごとに 1 本。** 演算子はループ内の `match` で分ける
//!    （分岐予測が効くうえ、コードは演算子数ぶん増えない）。
//!
//! NULL は「値の計算」と切り離す。ほとんどの演算では結果の validity は入力の
//! validity の AND で、値の中身は NULL 行では意味を持たない。例外は三値論理の
//! `AND`/`OR`（`logic`）とゼロ除算（値と同時に NULL を立てる）。

use crate::expr::{funcs, OpCode};
use crate::prelude::*;
use crate::vector::{
    pack_interval, unpack_interval, Bitmap, BytesData, Data, PhysType, Ty, Vector,
};

/// 1 日のマイクロ秒。DATE(I32,日) ↔ TIMESTAMP(I64,マイクロ秒)。
const MICROS_PER_DAY: i128 = 86_400_000_000;

/// 2^127。f64 → i128 の範囲判定に使う（i128::MAX を f64 にすると丸め上がるため）。
const I128_LIMIT: f64 = 170_141_183_460_469_231_731_687_303_715_884_105_728.0;

// --- ストライドと validity --------------------------------------------------

/// 長さ `l` のオペランドを `n` 行として読むときの stride。
/// 長さ 1（＝定数）は 0 を返し、同じループで vec/const 両方を扱えるようにする。
fn stride(l: usize, n: usize) -> Result<usize> {
    if l == n {
        Ok(1)
    } else if l == 1 {
        Ok(0)
    } else {
        err!(Internal)
    }
}

/// 2 項演算の行数と stride。どちらかが空なら結果も空。
pub fn strides2(la: usize, lb: usize) -> Result<(usize, usize, usize)> {
    let n = if la == 0 || lb == 0 { 0 } else { core::cmp::max(la, lb) };
    Ok((n, stride(la, n)?, stride(lb, n)?))
}

/// 3 項演算（`Select`）版。
fn strides3(la: usize, lb: usize, lc: usize) -> Result<(usize, usize, usize, usize)> {
    let n =
        if la == 0 || lb == 0 || lc == 0 { 0 } else { core::cmp::max(core::cmp::max(la, lb), lc) };
    Ok((n, stride(la, n)?, stride(lb, n)?, stride(lc, n)?))
}

/// 入力 validity の AND。どちらも NULL 無しなら `None`（ビットマップを作らない）。
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

/// カーネルが作った追加 NULL を入力由来の validity にマージして仕上げる。
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

// --- 算術 -------------------------------------------------------------------
// 整数は wrapping。オーバーフローでパニックさせない（checked 意味論が要るなら
// 将来 OpCode を分ける）。ゼロ除算と MIN / -1 は NULL（mod.rs の設計判断）。

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
                    // Div / Mod。0 除算と MIN / -1 は結果を NULL にする。
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

/// 浮動小数は IEEE 準拠。0 除算は NULL ではなく inf/NaN。
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

/// `Add`/`Sub`/`Mul`/`Div`/`Mod`/`Neg`。`Neg` は VM が b にも a を渡す。
///
/// DECIMAL は生の整数として計算する。`Add`/`Sub` はスケールが揃っていれば
/// これで正しく、`Mul`/`Div` のスケール調整は binder が `Cast` で行う。
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

// --- 比較 -------------------------------------------------------------------
// 3 値比較コード（+ NaN の「順序なし」）とマスクの AND で 6 演算子を 1 本に畳む。

const C_LT: u8 = 1;
const C_EQ: u8 = 2;
const C_GT: u8 = 4;
/// NaN が絡んで順序が付かない状態。`<>` だけがこれを真とする（IEEE 準拠）。
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

/// バイト列は辞書順（memcmp 順）。VARCHAR も同じ扱いで、照合順序は持たない。
fn cmp_bytes(a: &BytesData, sa: usize, b: &BytesData, sb: usize, n: usize, mask: u8) -> Bitmap {
    let mut out = Bitmap::with_capacity(n);
    for i in 0..n {
        let c = ord_code(a.get(i * sa).cmp(b.get(i * sb)));
        out.push(mask & c != 0);
    }
    out
}

/// 6 種の比較。入力は `phys`、出力は Bool。
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

// --- 三値論理 ---------------------------------------------------------------

/// `AND`/`OR`。値と validity を同時に決める必要があるので比較などと共通化できない。
///
/// - AND: 両方 TRUE なら TRUE、**どちらかが** FALSE なら（他方が NULL でも）FALSE。
/// - OR : **どちらかが** TRUE なら（他方が NULL でも）TRUE、両方 FALSE なら FALSE。
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
        // TRUE でも FALSE でも無ければ NULL。
        valid.push(t || f);
    }
    Ok(finish(Ty::Boolean, Data::Bool(vals), Some(valid), None))
}

/// `NOT`。NULL は NULL のまま。
pub fn not(a: &Vector) -> Result<Vector> {
    ensure!(a.data().phys() == PhysType::Bool, TypeMismatch);
    let mut bits = a.bools().clone();
    bits.negate();
    Ok(finish(Ty::Boolean, Data::Bool(bits), a.validity().cloned(), None))
}

/// `IsNull` / `IsNotNull`。結果は決して NULL にならない。
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

// --- 行コピー ---------------------------------------------------------------

/// `src` の行 `i` を `dst` の末尾に足す。物理型の一致は呼び出し側で検査済み。
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

/// ダミー行（NULL 行のプレースホルダ）。
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

/// 長さ 1 のベクタを `n` 行へ広げる。定数だけの式の結果を返すときに使う。
pub fn broadcast(v: &Vector, n: usize) -> Vector {
    debug_assert_eq!(v.len(), 1);
    let mut data = Data::with_capacity(v.ty().phys(), n);
    for _ in 0..n {
        push_row(&mut data, v.data(), 0);
    }
    let validity = if v.is_valid(0) { None } else { Some(Bitmap::zeros(n)) };
    Vector::from_data(v.ty(), data, validity)
}

/// `Select` と `Coalesce`。`cond` が `None` なら「`t` が有効か」を条件にする
/// （＝ COALESCE）。条件が NULL または FALSE なら `e` 側を採る。
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

// --- Bytes 演算 -------------------------------------------------------------

/// `a || b`。バイト列の連結。
pub fn concat(a: &Vector, b: &Vector, out_ty: Ty) -> Result<Vector> {
    ensure!(a.data().phys() == PhysType::Bytes && b.data().phys() == PhysType::Bytes, TypeMismatch);
    let (n, sa, sb) = strides2(a.len(), b.len())?;
    let (ad, bd) = (a.bytes(), b.bytes());
    let mut out = BytesData::with_capacity(n, ad.data.len() + bd.data.len());
    for i in 0..n {
        out.data.extend_from_slice(ad.get(i * sa));
        out.data.extend_from_slice(bd.get(i * sb));
        // offsets は u32。これを超える結果は扱えない。
        ensure!(out.data.len() <= u32::MAX as usize, LimitExceeded);
        out.offsets.push(out.data.len() as u32);
    }
    Ok(finish(out_ty, Data::Bytes(out), combine_validity(a, sa, b, sb, n), None))
}

/// SQL `LIKE`。`%` は 0 文字以上、`_` は**ちょうど 1 バイト**に一致する。
///
/// `_` がコードポイント単位でなくバイト単位なのは、UTF-8 の境界判定を持ち込むと
/// コードが増えるため。ASCII 以外を含む文字列では期待とずれる（既知の制限）。
/// `ESCAPE` 句も未対応。
///
/// バックトラックは「最後に出た `%` の位置」を 1 つ覚えるだけの 2 ポインタ法。
/// 再帰しないのでスタックを消費せず、`%a%a%a...` のようなパターンでも
/// 最悪 O(|s| * |p|) で済む（素朴な再帰だと指数時間になる）。
fn like_match(s: &[u8], p: &[u8]) -> bool {
    let (mut si, mut pi) = (0usize, 0usize);
    // `star_p == usize::MAX` は「まだ `%` を見ていない」。
    let (mut star_p, mut star_s) = (usize::MAX, 0usize);
    while si < s.len() {
        if pi < p.len() && (p[pi] == b'_' || p[pi] == s[si]) {
            si += 1;
            pi += 1;
        } else if pi < p.len() && p[pi] == b'%' {
            star_p = pi;
            star_s = si;
            pi += 1;
        } else if star_p != usize::MAX {
            // 直前の `%` に 1 バイト多く食わせてやり直す。
            pi = star_p + 1;
            star_s += 1;
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

// --- キャスト ---------------------------------------------------------------

/// 変換の族。物理型から決まるので、論理型ごとに分岐を増やさない。
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

/// 最近接偶数への丸め（銀行丸め）。`core` に `f64::round_ties_even` は無い。
///
/// 浮動小数 → 整数のキャストは切り捨てではなく丸める。SQL 標準は実装依存と
/// しているが、DuckDB も PostgreSQL もここは丸める。ちょうど 0.5 のときに
/// 偶数側へ寄せるのも両者に合わせてある（`1.5 → 2`、`4.5 → 4`）。
/// 0 から遠ざける丸めにすると、正の値ばかりのデータで合計が系統的に
/// 上振れするので、統計処理では偶数丸めの方が望ましい。
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
    // ちょうど半端。`t` が偶数ならそのまま、奇数なら 0 から遠ざける。
    // ここに来る時点で |x| は 2^52 未満なので i64 への変換は安全。
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

/// 整数を出力先の物理型へ書く。範囲外なら既定値を積んで `false`（＝ NULL）。
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

/// 整数系どうしの変換係数 `(mul, div, floor)`。
/// `floor` は床除算（TIMESTAMP→DATE のみ。エポック前で 1 日ずれないように）。
fn int_conv(from: Ty, to: Ty) -> Result<(i128, i128, bool)> {
    use Ty::*;
    if from.is_temporal() || to.is_temporal() {
        return Ok(match (from, to) {
            (Date, Timestamp) | (Date, Timestamptz) => (MICROS_PER_DAY, 1, false),
            (Timestamp, Date) | (Timestamptz, Date) => (1, MICROS_PER_DAY, true),
            // `Timestamp` と `Timestamptz` は物理表現が完全に同じ（UTC の
            // マイクロ秒）なので、値はそのまま素通しでよい
            // (`Ty::unify` が `Date`/`Timestamp` を `Timestamptz` に寄せた
            // ときの片側キャストがここを通る)。
            (Timestamp, Timestamptz) | (Timestamptz, Timestamp) => (1, 1, false),
            // 時刻系と整数は生値のまま。BOOLEAN や DECIMAL との変換は意味が無い。
            (f, t)
                if (f.is_temporal() && t.is_integer()) || (f.is_integer() && t.is_temporal()) =>
            {
                (1, 1, false)
            }
            _ => err!(InvalidCast),
        });
    }
    // DECIMAL は 10^scale 倍された整数。スケール差だけ掛け／割りする。
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
            // TIMESTAMP → DATE は床関数でなければならない。切り捨てると
            // エポック以前の値が 1 日ずれる。
            if r != 0 && (y < 0) != (div < 0) {
                q - 1
            } else {
                q
            }
        } else {
            // DECIMAL のスケール縮小は 0 から遠ざかる向きに丸める（DuckDB と
            // 同じ）。切り捨てると金額計算で系統的に過小評価になる。
            // `r * 2 >= div` ではなく半分と比較するのは、`r * 2` が i128 を
            // 溢れさせうるため。
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

/// 符号なし絶対値 + スケールで 10 進表記する。`format!` は使えない（サイズ）。
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
    // 0.05 のように整数部が無い場合の先頭 0 を補う。
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

/// f64 の 10 進表記。最短往復表現（Ryu/Grisu）はサイズが重いので採らず、
/// 15 桁に丸めて末尾 0 を落とす近似表記にする。
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
    // v = m * 10^e10 を保ったまま m を [1e14, 1e15) に正規化する。
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
        // 指数表記。
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

/// `mant * 10^exp` として 10 進数を読む。読めなければ `None`（＝ NULL）。
fn parse_dec(s: &[u8]) -> Option<(i128, i32)> {
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
    let mut mant: i128 = 0;
    let mut exp: i32 = 0;
    let mut ndig = 0u32;
    let mut seen = false;
    while i < s.len() && s[i].is_ascii_digit() {
        seen = true;
        if ndig < 38 {
            mant = mant * 10 + (s[i] - b'0') as i128;
            ndig += 1;
        } else {
            // 桁あふれ分は指数へ逃がす（精度は落ちる）。
            exp += 1;
        }
        i += 1;
    }
    if i < s.len() && s[i] == b'.' {
        i += 1;
        while i < s.len() && s[i].is_ascii_digit() {
            seen = true;
            if ndig < 38 {
                mant = mant * 10 + (s[i] - b'0') as i128;
                ndig += 1;
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
    Some((if neg { -mant } else { mant }, exp))
}

/// `m * 10^e`。2 進分解で掛けるので、10 を e 回掛けるより誤差が小さい。
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

/// `VARCHAR → JSON`。各行を `crate::json::validate` で検証する。
///
/// ここだけ他の CAST と違い、行単位の失敗の扱いが `CAST`/`TRY_CAST` で
/// 分かれる: `lenient == false`（通常の `CAST`）は不正な JSON をエラーに
/// する（DuckDB の実測 `CAST('not json' AS JSON)` がエラーになる挙動に
/// 合わせた）。`lenient == true`（`TRY_CAST`）はその行だけ NULL にする
/// （他の型の TRY_CAST・行単位パース失敗と同じ規則に揃える）。
/// 他の CAST 全般は「行単位の失敗は常に NULL、CAST/TRY_CAST で差が無い」
/// 設計だが、JSON はドキュメント全体が壊れているかどうかの 2 値しか
/// 無いため、DuckDB の実際の挙動に寄せてここだけ例外にした。
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

/// `VARCHAR → UUID`。行単位のパース失敗は `DATE`/`TIME`/`TIMESTAMP` と同じ
/// 慣習でその行だけ NULL にする（`CAST`/`TRY_CAST` の両方、`lenient` に
/// 関わらず）。`VARCHAR → JSON` だけがこの慣習の例外であり（`cast_str_to_json`
/// の doc 参照）、UUID はそちらには倣わない。
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

/// `Cast`。実装していない組み合わせは黙って壊れた値を返さず `InvalidCast`。
/// 行単位の変換失敗（範囲外・パース不能）はエラーにせず、その行だけ NULL。
///
/// `VARCHAR → JSON` だけは例外で、行単位の失敗（妥当な JSON でない）を
/// `TRY_CAST` のときだけ NULL にする（他の型と違い、通常の `CAST` は
/// エラーにする。DuckDB の実際の挙動に合わせた意図的な例外。詳細は
/// [`try_cast`] の doc を参照）。
pub fn cast(from: Ty, to: Ty, a: &Vector) -> Result<Vector> {
    cast_impl(from, to, a, false)
}

/// `TRY_CAST`。ほとんどの型では [`cast`] と完全に同じ（行単位の変換失敗は
/// どのみち `cast` がエラーにせず NULL にする契約のため）。唯一の違いは
/// `VARCHAR → JSON` の行単位検証: `CAST` は不正な JSON をエラーにするが、
/// `TRY_CAST` はその行だけ NULL にする（型ペア自体が非対応の場合は従来どおり
/// `expr::vm` 側が catch して全行 NULL に落とす）。
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
        // 型未定の NULL リテラル。値は全行 NULL なので変換規則は不要。
        let mut out = Vector::new(to);
        for _ in 0..n {
            out.push_null();
        }
        return Ok(out);
    }
    // JSON は fam() だと VARCHAR/BLOB と同じ Bytes 系に丸められてしまうので、
    // 汎用の (Fam, Fam) マッチに乗せる前に個別に弾く。対応するのは
    // VARCHAR ↔ JSON のみ（BLOB や数値・日時からの CAST は非対応のまま
    // `InvalidCast` にする。`to_json` 関数を使うべき、という設計判断）。
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
    // UUID も同じ理由で個別に弾く: 対応するのは VARCHAR ↔ UUID のみ。
    // 物理表現は VARCHAR/BLOB と同じ `Bytes` だが、UUID のテキスト表現
    // （ハイフン付き 16 進）は生バイト列とは異なるので、汎用の `(Fam::Str,
    // Fam::Str)` の単純コピーには乗せられない（`BLOB ↔ UUID` は非対応の
    // まま `InvalidCast` にする。生バイトを扱いたければ `BLOB` 経由ではなく
    // 素直に `UUID` を使う、という設計判断）。
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
            // 添字は値の取得だけでなく `set_null` の行指定にも使う。
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
            // `n` は stride 適用後の行数で `sv.len()` とは限らない。
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                // DOUBLE → FLOAT は f32 の精度へ落とす（保持は f64 のまま）。
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
                        // 日付の書式化は funcs 側に置く（パーサと対で使うため）。
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
            // VARCHAR ↔ BLOB。表現は同じなのでそのまま複製する。
            let mut out = a.clone();
            out.retype(to);
            return Ok(out);
        }
        (Fam::Str, Fam::Flt) => {
            let sv = a.bytes();
            for i in 0..n {
                match parse_dec(sv.get(i)) {
                    Some((m, e)) => {
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
            // 読めない文字列はエラーにせず、その行だけ NULL（数値パースと同じ）。
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
                    // BOOLEAN でも 'true'/'false' 以外は数値として読み、0 以外を真とする。
                    ok = match parse_dec(b) {
                        Some((m, e)) => {
                            let k = e + scale;
                            let y = if k >= 0 {
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

// --- INTERVAL 演算 -----------------------------------------------------------
// 物理型は 3 つとも I128 系だが、フィールド境界を跨ぐ桁上がりを起こしては
// いけない（`1 month + 3 days` は `4 ...` にはならない）ので、生の i128
// 二進演算では表現できない。専用カーネルにしてある理由はこれ。

/// TIMESTAMP(I64, マイクロ秒) + INTERVAL(I128)。DATE は呼び出し側が
/// TIMESTAMP へキャストしてから渡す（DuckDB も DATE ± INTERVAL は
/// TIMESTAMP を返す）。
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

/// INTERVAL ± INTERVAL。フィールドごとの加算（繰り上がりなし。DuckDB も
/// `1 month + 3 days` は `1 month 3 days` のまま正規化しない）。
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

/// INTERVAL の符号反転。フィールドごとに反転する（生の 128bit 二の補数の
/// 反転は桁境界を跨いで壊れるため使えない）。
pub fn interval_neg(a: &Vector) -> Result<Vector> {
    ensure!(a.data().phys() == PhysType::I128, TypeMismatch);
    let mut out = Vec::with_capacity(a.len());
    for &packed in a.i128s() {
        let (m, d, u) = unpack_interval(packed);
        out.push(pack_interval(m.wrapping_neg(), d.wrapping_neg(), u.wrapping_neg()));
    }
    Ok(finish(Ty::Interval, Data::I128(out), a.validity().cloned(), None))
}

/// INTERVAL * BIGINT。フィールドごとの乗算（繰り上がりなし。DuckDB と同じ）。
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

    #[test]
    fn like_pathological_is_not_exponential() {
        // 素朴な再帰実装だと指数時間になるパターン。線形バックトラックなら一瞬。
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
        assert_eq!(parse_dec(b"123"), Some((123, 0)));
        assert_eq!(parse_dec(b" -12.5 "), Some((-125, -1)));
        assert_eq!(parse_dec(b"+1e3"), Some((1, 3)));
        assert_eq!(parse_dec(b".5"), Some((5, -1)));
        assert_eq!(parse_dec(b"1E-2"), Some((1, -2)));
        assert_eq!(parse_dec(b""), None);
        assert_eq!(parse_dec(b"abc"), None);
        assert_eq!(parse_dec(b"1.2.3"), None);
        assert_eq!(parse_dec(b"1e"), None);
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
        // 2024-01-31 00:00:00 + 1 month → 2024-02-29（うるう年、月末クランプ）。
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

    // 「整数は wrapping。オーバーフローで panic させない」という設計
    // （このファイル冒頭の `int_arith!` のコメント参照）は INTERVAL の
    // フィールドごとの演算にも一貫して適用されている。この境界での
    // wrapping 挙動をここで固定しておく。
    #[test]
    fn interval_neg_of_i32_min_stays_negative_due_to_two_s_complement_wraparound() {
        // i32::MIN.wrapping_neg() == i32::MIN（正の i32::MAX+1 は表現できない）。
        // 通常の整数 Neg カーネルと同じ、意図された wrapping 挙動。
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
        // 乗算は i64 の中間精度で行ってから i32 へ切り詰める
        // （`(m as i64).wrapping_mul(k) as i32`）。k 自体を先に i32 へ
        // 切り詰めてから掛けるわけではないので、大きな k でも最終結果の
        // 下位 32bit は一貫して同じ値になる（二重の切り詰めにはならない）。
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
