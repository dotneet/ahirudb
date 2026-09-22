//! Shared `f64` -> shortest-round-trip decimal text: the crate's single float
//! formatter, used by the `CAST(<double> AS VARCHAR)` kernel
//! (`expr::kernels::fmt_f64`), by the CSV and JSONL writers (`write/csv.rs`,
//! `write/jsonl.rs`, which re-export this module through `write/mod.rs`), and
//! through `fmt_f64` by the CLI's cell renderer.
//!
//! It lives under `expr/` rather than `write/` because `expr` is compiled
//! unconditionally while all of `write` is gated behind the opt-in `export`
//! feature, and the cast path needs this formatter in every build.
//!
//! `core` has no float formatting (`core::fmt`'s Display/Debug machinery
//! alone costs 30-60 KB, DESIGN.md §4, so this crate avoids it everywhere,
//! not just here), so this is hand-rolled. What it produces, for any finite
//! `f64`, is the *shortest* decimal digit string that round-trips back to
//! the same `f64` bit pattern, choosing (when more than one digit string of
//! that shortest length round-trips) the one nearest the value's exact
//! binary value, with exact ties broken to the even digit -- matching
//! Ryū/Grisu/Dragon4-style correctly-rounded shortest formatters (Python's
//! `repr` and DuckDB's writer; Rust `std`'s `f64` Display too, except that it
//! breaks an exact tie the other way).
//!
//! This lives in one place, rather than once per writer, because this exact
//! logic was previously written out in full independently in both
//! `write/csv.rs` and `write/jsonl.rs` (~350 lines each): the same intricate
//! exact-big-integer shortest-round-trip algorithm, needing to stay bit-for-
//! bit in sync so CSV and JSONL exports of the same query never disagree on
//! how a float is spelled. That already required three rounds of manual
//! "keep both copies in sync" during development, and any future fix landed
//! on only one copy would silently reintroduce that divergence. Sharing it
//! here makes divergence structurally impossible instead of merely policed.
//!
//! The two writers differ only in how they handle non-finite values (CSV
//! writes `NaN` / `inf` / `-inf`; JSON has no such literal, so JSONL quotes
//! them instead) -- that part stays local to each writer.
//! Everything else, starting from finite-value handling (including zero,
//! sign, and the fixed-vs-exponential notation threshold), is byte-identical
//! between the two formats and lives here as `write_f64_finite`.

// `Vec` and `vec!` come from here, not from a prelude: this crate is `no_std`
// on the wasm target, where neither is in scope by default.
use crate::prelude::*;

/// Writes a finite `f64` (including positive/negative zero) as shortest
/// round-trip decimal text. Callers are responsible for handling non-finite
/// values (`NaN`/infinities) themselves before calling this -- see the
/// module doc for why that part is not shared.
///
/// The digits come from [`shortest_digits`], an exact big-integer
/// implementation of the Steele & White / Burger & Dybvig "free-format"
/// algorithm, and are then laid out by [`write_decimal`].
pub(crate) fn write_f64_finite(out: &mut Vec<u8>, v: f64) {
    write_finite(out, v, Prec::F64)
}

/// The same, for a value whose *logical* type is `FLOAT`.
///
/// FLOAT is held in an `f64` register like every other floating-point value (there
/// is no separate physical type for it -- DESIGN.md's six-physical-type model), but
/// every such value came from an `f32` and is exactly representable as one. Measuring
/// the round-trip against `f64` therefore asks for far more digits than the value
/// actually carries: `1.1::FLOAT` is the `f64` 1.100000023841858, and that is what
/// `CAST(... AS VARCHAR)` and the CSV writer used to print, where DuckDB prints `1.1`.
/// Measuring it against `f32` instead -- the only thing that changes is the width of
/// the rounding interval the digits must land in -- gives the shortest string that
/// round-trips through the type the value really has.
///
/// Callers handle non-finite values themselves, exactly as for [`write_f64_finite`].
pub(crate) fn write_f32_finite(out: &mut Vec<u8>, v: f64) {
    write_finite(out, v, Prec::F32)
}

/// Which floating-point width a digit string has to round-trip through. It changes
/// *only* the rounding interval; digit generation, the nearest-candidate choice and
/// the fixed-vs-exponential rendering are identical (DuckDB likewise spells an `f32`
/// and an `f64` of the same value the same way once the digits are chosen -- verified
/// against the `duckdb` CLI for the notation thresholds).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Prec {
    F64,
    F32,
}

fn write_finite(out: &mut Vec<u8>, v: f64, prec: Prec) {
    if v == 0.0 {
        out.extend_from_slice(if v.is_sign_negative() { b"-0.0" } else { b"0.0" });
        return;
    }
    if v.is_sign_negative() {
        out.push(b'-');
    }
    let (f, e, p, min_e) = decompose(v, prec);
    let (digits, e10) = shortest_digits(f, e, p, min_e);
    write_decimal(out, &digits, e10);
}

/// Splits `|v|` (finite, nonzero) into `(f, e, p, min_e)` with `|v| == f * 2^e`
/// *exactly*, read directly off the IEEE 754 bit layout of the width `prec` names:
/// `p` is that format's significand width (hidden bit included) and `min_e` the
/// exponent of its subnormals. A FLOAT's `f64` is exactly an `f32`, so the `f32`
/// narrowing is lossless.
fn decompose(v: f64, prec: Prec) -> (u64, i32, u32, i32) {
    let (bits, frac_bits, bias) = match prec {
        Prec::F64 => (v.to_bits(), 52u32, 1075i32),
        Prec::F32 => ((v as f32).to_bits() as u64, 23, 150),
    };
    let exp_mask = match prec {
        Prec::F64 => 0x7FF,
        Prec::F32 => 0xFF,
    };
    let biased = ((bits >> frac_bits) & exp_mask) as i32;
    let frac = bits & ((1u64 << frac_bits) - 1);
    let min_e = 1 - bias;
    if biased == 0 {
        // Subnormal: no implicit leading bit, and the exponent is pinned to the
        // smallest normal exponent's value.
        (frac, min_e, frac_bits + 1, min_e)
    } else {
        (frac | (1u64 << frac_bits), biased - bias, frac_bits + 1, min_e)
    }
}

/// The shortest decimal digit string that reads back as `f * 2^e`, and the decimal
/// exponent of its leading digit.
///
/// Every real number strictly inside the rounding interval of `v` -- halfway to the
/// next representable value on either side, with the endpoints themselves included
/// when `f` is even, because a reader rounds an exact halfway point to the even
/// significand -- reads back as `v`. This is the classic free-format algorithm (Steele
/// & White's FPP², in Burger & Dybvig's formulation): scale the value `r / s` and the
/// two half-gaps `m- / s`, `m+ / s` so the first digit sits just below the decimal
/// point, then peel off one digit at a time until the remaining digits are no longer
/// needed to stay inside the interval. When stopping, the last digit is rounded
/// towards whichever of the two candidates is nearer `v`, with an exact tie going to
/// the even digit -- the choice DuckDB and Python's `repr` make (Rust's `std` breaks
/// that one tie the other way).
///
/// All of it runs on exact big integers ([`Big`]), so there is no floating-point
/// rounding anywhere in the decision. An earlier implementation seeded the digits from
/// repeated `f64` multiplications by ten and repaired them by re-parsing candidates;
/// when that normalization overshot a power of ten it re-pinned the 17-digit seed by
/// multiplying it by ten, throwing its last digit away, and so
/// `0.09999999999999999` came out as the non-shortest `0.099999999999999992`.
fn shortest_digits(f: u64, e: i32, p: u32, min_e: i32) -> (Vec<u8>, i32) {
    // The lower half-gap is half as wide as the upper one exactly at a power of two
    // (the value below it has a smaller exponent), except at the bottom of the range.
    let unequal = f == 1u64 << (p - 1) && e > min_e;
    let (mut r, mut s, mut m_plus, mut m_minus);
    if e >= 0 {
        let be = Big::from_u64(1).shl(e as u32);
        if unequal {
            r = Big::from_u64(f).shl(e as u32 + 2);
            s = Big::from_u64(4);
            m_plus = be.clone().shl(1);
            m_minus = be;
        } else {
            r = Big::from_u64(f).shl(e as u32 + 1);
            s = Big::from_u64(2);
            m_plus = be.clone();
            m_minus = be;
        }
    } else if unequal {
        r = Big::from_u64(f).shl(2);
        s = Big::from_u64(1).shl((2 - e) as u32);
        m_plus = Big::from_u64(2);
        m_minus = Big::from_u64(1);
    } else {
        r = Big::from_u64(f).shl(1);
        s = Big::from_u64(1).shl((1 - e) as u32);
        m_plus = Big::from_u64(1);
        m_minus = Big::from_u64(1);
    }
    let even = f.is_multiple_of(2);
    // `high_ok`: is the value just above the kept digits still inside the interval?
    let high_ok = |r: &Big, m_plus: &Big, s: &Big| {
        let hi = r.add(m_plus);
        match hi.cmp(s) {
            core::cmp::Ordering::Greater => true,
            core::cmp::Ordering::Equal => even,
            core::cmp::Ordering::Less => false,
        }
    };
    // An estimate of `k = ceil(log10(v))`, low by at most one: `floor(log2(v))` is
    // `e + bitlen(f) - 1`, and 78913 / 2^18 is log10(2) rounded down.
    let log2 = e + (64 - f.leading_zeros() as i32) - 1;
    let mut k = ((log2 as i64 * 78_913) >> 18) as i32 + 1;
    if k >= 0 {
        s = s.mul_pow10(k as u32);
    } else {
        let n = (-k) as u32;
        r = r.mul_pow10(n);
        m_plus = m_plus.mul_pow10(n);
        m_minus = m_minus.mul_pow10(n);
    }
    // Fix the estimate up so that `(r + m+) / s` is in `[0.1, 1)`: the leading digit
    // then comes out of the first multiplication by ten.
    while high_ok(&r, &m_plus, &s) {
        s = s.mul_small(10);
        k += 1;
    }
    loop {
        let (r10, p10) = (r.clone().mul_small(10), m_plus.clone().mul_small(10));
        if high_ok(&r10, &p10, &s) {
            break;
        }
        r = r10;
        m_plus = p10;
        m_minus = m_minus.mul_small(10);
        k -= 1;
    }
    let mut digits = Vec::new();
    loop {
        r = r.mul_small(10);
        m_plus = m_plus.mul_small(10);
        m_minus = m_minus.mul_small(10);
        let mut d = 0u8;
        while r.cmp(&s) != core::cmp::Ordering::Less {
            r = r.sub(&s);
            d += 1;
        }
        let low = match r.cmp(&m_minus) {
            core::cmp::Ordering::Less => true,
            core::cmp::Ordering::Equal => even,
            core::cmp::Ordering::Greater => false,
        };
        let high = high_ok(&r, &m_plus, &s);
        if !low && !high {
            digits.push(d);
            continue;
        }
        let up = match (low, high) {
            (true, false) => false,
            (false, true) => true,
            // Both candidates round-trip: take the nearer, and the even one on a tie.
            _ => match r.shl(1).cmp(&s) {
                core::cmp::Ordering::Less => false,
                core::cmp::Ordering::Greater => true,
                core::cmp::Ordering::Equal => d % 2 == 1,
            },
        };
        digits.push(d + up as u8);
        break;
    }
    // A rounded-up 9 carries into the digits before it.
    let mut i = digits.len() - 1;
    while digits[i] == 10 {
        digits[i] = 0;
        if i == 0 {
            digits.insert(0, 1);
            k += 1;
            break;
        }
        i -= 1;
        digits[i] += 1;
    }
    while digits.len() > 1 && *digits.last().unwrap_or(&1) == 0 {
        digits.pop();
    }
    for d in digits.iter_mut() {
        *d += b'0';
    }
    (digits, k - 1)
}

/// A minimal little-endian, base-`2^32` unsigned big integer: just the operations
/// [`shortest_digits`] needs (shift, multiply by a small factor or a power of ten,
/// add, subtract, compare). The operands reach about 1100 bits for the most extreme
/// exponents and stay at two or three limbs for everyday values.
#[derive(Clone)]
struct Big {
    limbs: Vec<u32>,
}

impl Big {
    fn from_u64(v: u64) -> Self {
        let mut b = Big { limbs: vec![v as u32, (v >> 32) as u32] };
        b.trim();
        b
    }

    fn trim(&mut self) {
        while self.limbs.len() > 1 && self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
    }

    fn mul_small(mut self, m: u32) -> Self {
        let mut carry: u64 = 0;
        for limb in self.limbs.iter_mut() {
            let prod = (*limb as u64) * (m as u64) + carry;
            *limb = prod as u32;
            carry = prod >> 32;
        }
        if carry > 0 {
            self.limbs.push(carry as u32);
        }
        self.trim();
        self
    }

    /// Multiplies by `10^n`, nine decimal digits at a time.
    fn mul_pow10(mut self, mut n: u32) -> Self {
        while n >= 9 {
            self = self.mul_small(1_000_000_000);
            n -= 9;
        }
        self.mul_small(10u32.pow(n))
    }

    /// Multiplies by `2^n`.
    fn shl(&self, n: u32) -> Self {
        let (words, bits) = ((n / 32) as usize, n % 32);
        let mut limbs = vec![0u32; words];
        let mut carry = 0u32;
        for &l in &self.limbs {
            if bits == 0 {
                limbs.push(l);
            } else {
                limbs.push((l << bits) | carry);
                carry = l >> (32 - bits);
            }
        }
        if carry > 0 {
            limbs.push(carry);
        }
        let mut b = Big { limbs };
        b.trim();
        b
    }

    fn add(&self, other: &Big) -> Self {
        let n = self.limbs.len().max(other.limbs.len());
        let mut limbs = Vec::with_capacity(n + 1);
        let mut carry = 0u64;
        for i in 0..n {
            let a = *self.limbs.get(i).unwrap_or(&0) as u64;
            let b = *other.limbs.get(i).unwrap_or(&0) as u64;
            let sum = a + b + carry;
            limbs.push(sum as u32);
            carry = sum >> 32;
        }
        if carry > 0 {
            limbs.push(carry as u32);
        }
        Big { limbs }
    }

    /// `self - other`, which the caller guarantees is not negative.
    fn sub(mut self, other: &Big) -> Self {
        let mut borrow = 0i64;
        for i in 0..self.limbs.len() {
            let diff = self.limbs[i] as i64 - *other.limbs.get(i).unwrap_or(&0) as i64 - borrow;
            if diff < 0 {
                self.limbs[i] = (diff + (1i64 << 32)) as u32;
                borrow = 1;
            } else {
                self.limbs[i] = diff as u32;
                borrow = 0;
            }
        }
        self.trim();
        self
    }

    fn cmp(&self, other: &Big) -> core::cmp::Ordering {
        if self.limbs.len() != other.limbs.len() {
            return self.limbs.len().cmp(&other.limbs.len());
        }
        for i in (0..self.limbs.len()).rev() {
            if self.limbs[i] != other.limbs[i] {
                return self.limbs[i].cmp(&other.limbs[i]);
            }
        }
        core::cmp::Ordering::Equal
    }
}

/// Renders the significant `digits` (ASCII, leading digit at decimal exponent `e10`) as
/// plain decimal or exponent notation, matching both how this crate's own
/// CSV/JSONL readers accept numbers and (verified against the `duckdb` CLI
/// directly) how DuckDB's CSV/JSON writer formats them -- this project's
/// test suite compares against DuckDB byte for byte.
fn write_decimal(out: &mut Vec<u8>, digits: &[u8], e10: i32) {
    // DuckDB switches to exponent notation for `e10 < -4` or `e10 >= 16`
    // (e.g. `1e-4` stays `0.0001` but `1e-5` becomes `1e-05`; `1234567890123456.0`
    // -- 16 digits, `e10 == 15` -- stays plain but `1e16` becomes `1e+16`).
    if (-4..16).contains(&e10) {
        write_fixed(out, digits, e10);
    } else {
        write_exponential(out, digits, e10);
    }
}

fn write_fixed(out: &mut Vec<u8>, digits: &[u8], e10: i32) {
    if e10 < 0 {
        out.push(b'0');
        out.push(b'.');
        for _ in 0..(-e10 - 1) {
            out.push(b'0');
        }
        out.extend_from_slice(digits);
    } else {
        let int_len = (e10 + 1) as usize;
        if digits.len() <= int_len {
            out.extend_from_slice(digits);
            for _ in 0..(int_len - digits.len()) {
                out.push(b'0');
            }
            out.push(b'.');
            out.push(b'0');
        } else {
            out.extend_from_slice(&digits[..int_len]);
            out.push(b'.');
            out.extend_from_slice(&digits[int_len..]);
        }
    }
}

/// DuckDB always writes an explicit sign (`e+16`, `e-05`) and pads the
/// exponent magnitude to at least 2 digits (`e-05`, not `e-5`; `e+100` is
/// left alone, not padded to a fixed width) -- both verified against the
/// `duckdb` CLI directly.
fn write_exponential(out: &mut Vec<u8>, digits: &[u8], e10: i32) {
    out.push(digits[0]);
    if digits.len() > 1 {
        out.push(b'.');
        out.extend_from_slice(&digits[1..]);
    }
    out.push(b'e');
    out.push(if e10 < 0 { b'-' } else { b'+' });
    let mag = e10.unsigned_abs();
    if mag < 10 {
        out.push(b'0');
    }
    push_int(out, mag as i128);
}

/// A private copy of the digit-writing helper both `csv.rs` and `jsonl.rs`
/// also define for their own (non-float) integer/decimal formatting. Not
/// shared with those: it is used here only to write `write_exponential`'s
/// exponent magnitude, which is unrelated to why it also exists in each
/// writer (formatting plain integers and DECIMAL columns), so pulling it in
/// from either writer would create an arbitrary cross-dependency instead of
/// removing real duplication.
fn push_int(out: &mut Vec<u8>, v: i128) {
    if v < 0 {
        out.push(b'-');
    }
    let mut buf = [0u8; 40];
    let mut n = 0usize;
    let mut u = v.unsigned_abs();
    loop {
        buf[n] = b'0' + (u % 10) as u8;
        n += 1;
        u /= 10;
        if u == 0 {
            break;
        }
    }
    for i in (0..n).rev() {
        out.push(buf[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn written(v: f64) -> String {
        let mut out = Vec::new();
        write_f64_finite(&mut out, v);
        String::from_utf8(out).expect("write_f64_finite output must be valid UTF-8")
    }

    fn written32(v: f32) -> String {
        let mut out = Vec::new();
        write_f32_finite(&mut out, v as f64);
        String::from_utf8(out).expect("write_f32_finite output must be valid UTF-8")
    }

    /// A FLOAT is held in an `f64` register, so measuring the round trip against `f64`
    /// asked for digits the value never had: `1.1::FLOAT` printed as
    /// `1.100000023841858`. The expected strings here were verified against the `duckdb`
    /// CLI directly (`SELECT (<literal>::FLOAT)::VARCHAR`), including the
    /// fixed-vs-exponential switchover, which FLOAT shares with DOUBLE.
    #[test]
    fn f32_precision_gives_the_shortest_f32_round_trip() {
        let cases: &[(f32, &str)] = &[
            (1.1, "1.1"),
            (0.1, "0.1"),
            (1.0, "1.0"),
            (-2.5, "-2.5"),
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            // The fixed/exponential thresholds are the same as DOUBLE's.
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            // f32's extremes: the largest finite value and the smallest subnormal.
            (f32::MAX, "3.4028235e+38"),
            (1e-45, "1e-45"),
            (f32::MIN_POSITIVE, "1.1754944e-38"),
            // Not representable as an f32; it lands on the even neighbour.
            (16777217.0, "16777216.0"),
        ];
        for (v, expect) in cases {
            let got = written32(*v);
            assert_eq!(got, *expect, "{v}: wrote {got:?}, expected {expect:?}");
        }
        // The DOUBLE formatter is unchanged and still spells the same bits the long way.
        assert_eq!(written(1.1f32 as f64), "1.100000023841858");
    }

    /// The f32 form must be a genuine shortest round trip, not merely shorter: every
    /// value has to reparse to the identical `f32`, and to the same significant digits
    /// Rust `std`'s own (correctly rounded, shortest) `f32` Display produces. Same
    /// sampling method and same `std`-as-oracle rationale as the `f64` property test
    /// below; exact ties are the one case `std` is known to break the other way, and
    /// this asserts adjacency there rather than waving the mismatch through.
    #[test]
    fn f32_matches_std_shortest_round_trip() {
        fn digits_of(s: &str) -> Vec<u8> {
            let mantissa = s.split(['e', 'E']).next().unwrap_or(s);
            let mut d: Vec<u8> = mantissa.bytes().filter(u8::is_ascii_digit).collect();
            while d.len() > 1 && d[0] == b'0' {
                d.remove(0);
            }
            while d.len() > 1 && *d.last().unwrap() == b'0' {
                d.pop();
            }
            d
        }

        let mut seed: u32 = 0x9E37_79B9;
        for _ in 0..20000 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = f32::from_bits(seed);
            if !v.is_finite() || v == 0.0 {
                continue;
            }
            let got = written32(v);
            let back: f32 = got
                .parse()
                .unwrap_or_else(|e| panic!("{v}: wrote {got:?}, failed to reparse: {e}"));
            assert_eq!(back.to_bits(), v.to_bits(), "{v}: wrote {got:?}, round-tripped to {back}");
            let std_form = std::format!("{v}");
            let (ours, theirs) = (digits_of(&got), digits_of(&std_form));
            if ours == theirs {
                continue;
            }
            let to_u128 = |d: &[u8]| d.iter().fold(0u128, |a, &b| a * 10 + (b - b'0') as u128);
            assert!(
                ours.len() == theirs.len() && to_u128(&ours).abs_diff(to_u128(&theirs)) == 1,
                "{v}: wrote {got:?}, std wrote {std_form:?} -- not an exact-tie disagreement"
            );
        }
    }

    // Regression test for a real correctness bug found during review: the
    // old `push_f64` (before this shared module existed) took the integer
    // part via a saturating `x as i128` cast and the fraction part via 15
    // iterations of `x *= 10.0`. That silently produced wrong output outside
    // i128's range and outside the multiply loop's 15-digit reach: `1e40`
    // wrote as `i128::MAX` (the saturated cast) and `1e-20` wrote as `0.0`
    // (all 15 fraction digits landed on zero before the first significant
    // one). Every finite magnitude now round-trips, and does so with the
    // same digit string DuckDB would write.
    //
    // The exact expected strings below were verified against the `duckdb`
    // CLI directly (`COPY (SELECT <literal>::DOUBLE a) TO '...'`), since
    // this project's test suite also compares CSV/JSONL output against
    // DuckDB byte for byte (`crates/ahiru-cli/tests/copy.rs`).
    #[test]
    // The `.25`/`.4921875` literals below are written at their full exact
    // decimal precision deliberately (that they name the same `f64` as a
    // shorter literal is exactly the point being tested -- see the comment
    // on the tie cases below), not accidentally over-precise.
    #[allow(clippy::excessive_precision)]
    fn exact_strings_including_tie_regressions() {
        let cases: &[(f64, &str)] = &[
            (1e-20, "1e-20"),
            (1e40, "1e+40"),
            (1e-300, "1e-300"),
            (1e300, "1e+300"),
            (0.1, "0.1"),
            (-0.0, "-0.0"),
            (3.0, "3.0"),
            (1.5e10, "15000000000.0"),
            (-2.5e-15, "-2.5e-15"),
            (123456789.123456, "123456789.123456"),
            // DuckDB-verified exponent-notation thresholds and spelling:
            // explicit sign, exponent magnitude padded to >= 2 digits, and
            // the fixed/exponential switchover at e10 < -4 or e10 >= 16.
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (1234567890123456.0, "1234567890123456.0"),
            // These decimal strings are what the *old* buggy formatter used
            // to emit for exactly `138.0`/`148.5` (see the coordinator's bug
            // report reproduction against `tests/data/basic.csv`); as an
            // `f64` literal each is bit-identical to the short form
            // (`clippy::excessive_precision` -- confirmed with `rustc`, not
            // just asserted).
            (138.0, "138.0"),
            (148.5, "148.5"),
            // Genuine exact-decimal ties (the underlying `f64`'s value is
            // *exactly* halfway between two candidates at the shortest
            // round-tripping length -- these all have a small enough binary
            // exponent that their decimal expansion terminates exactly one
            // digit past what's needed). DuckDB-verified directly (`COPY
            // (SELECT <literal>::DOUBLE) TO ...`): DuckDB (and Python's
            // `repr`, and this crate's own tie-break) round these to even,
            // e.g. `853.25` -> `853.2` (not `853.3`). Notably, Rust `std`'s
            // own `f64` Display disagrees with DuckDB/Python on these
            // specific cases (confirmed with `rustc` directly) -- which is
            // why the property test below excludes exact ties from its
            // `std`-comparison rather than trusting `std` universally.
            (667082108456853.25, "667082108456853.2"),
            (914912890181944.25, "914912890181944.2"),
            (829306832509257.25, "829306832509257.2"),
            (12386969366.4921875, "12386969366.492188"),
        ];
        for (v, expect) in cases {
            let got = written(*v);
            assert_eq!(got, *expect, "{v}: wrote {got:?}, expected {expect:?}");
            // Round-trips back through a plain `f64` parse (both writers'
            // own readers accept this same grammar; see each writer's own
            // round-trip tests for the format-specific parsing path).
            let reparsed: f64 = got
                .parse()
                .unwrap_or_else(|e| panic!("{v}: wrote {got:?}, which failed to reparse: {e}"));
            assert_eq!(
                reparsed.to_bits(),
                v.to_bits(),
                "{v}: wrote {got:?}, round-tripped to {reparsed}"
            );
        }
    }

    // Property-style test: a broad spread of `f64` values (a fixed list plus
    // a deterministic LCG-generated sample of raw bit patterns, so this is
    // reproducible without an external RNG dependency) must all satisfy two
    // properties: (1) round-trip through a plain `f64` parse back to the
    // identical bit pattern, and (2) use the *same* significant-digit
    // sequence as Rust `std`'s `f64` `Display`, which is itself a
    // correctly-rounded shortest-round-trip (Grisu/Dragon4-style) formatter.
    // `std::fmt` is otherwise avoided everywhere in this crate (DESIGN.md
    // §4), but that constraint is about what ships in the `no_std` wasm
    // build, not this native, `std`-only (`#[cfg(test)]`) test binary, so
    // using it purely as a test oracle here does not reintroduce that cost
    // into the shipped artifact.
    //
    // This used to check digit *count* only ("no more digits than std"),
    // which missed a real, systematic bug: when more than one shortest-length
    // digit string round-trips, this writer was picking the low end of that
    // window rather than the one nearest `x`'s exact value (same count,
    // wrong last digit -- e.g. `1.2933663726238106e+51` instead of the
    // correct `...107`, confirmed against DuckDB, Python's `repr`, and this
    // same `std` Display, which all three agree with each other on and
    // disagreed with the old output on ~14% of a 400-value random sample).
    // Comparing the full digit *sequence*, not just its length, is what
    // catches that class of bug.
    #[test]
    fn matches_std_shortest_round_trip_digit_for_digit() {
        // Extracts the bare significant-digit sequence, discarding
        // everything presentational: sign, decimal point, and (for the
        // mantissa) leading/trailing zeros that only place the decimal
        // point/exponent rather than carry precision. This is what lets a
        // `std` Display string (`std` never uses exponent notation, always
        // a bare fixed form) compare directly against this module's output
        // (which may use either notation), despite the two using different
        // conventions for a synthetic trailing `.0`.
        //
        // Stripping trailing zeros is safe *because* both strings are
        // shortest round-tripping representations (by construction for
        // `write_f64_finite`'s output, by contract for `std`'s Display): a
        // significant trailing zero digit is never actually needed by a
        // shortest representation, since dropping it names the same real
        // number (a trailing zero only shifts where the decimal point/
        // exponent implicitly falls) at one digit shorter, and the
        // shortest-search would already have stopped there.
        fn significant_digits(s: &str) -> Vec<u8> {
            let mantissa = s.split(['e', 'E']).next().unwrap_or(s);
            let mut digits: Vec<u8> = mantissa.bytes().filter(u8::is_ascii_digit).collect();
            while digits.len() > 1 && digits[0] == b'0' {
                digits.remove(0);
            }
            while digits.len() > 1 && *digits.last().unwrap() == b'0' {
                digits.pop();
            }
            digits
        }

        fn to_u128(d: &[u8]) -> u128 {
            d.iter().fold(0u128, |acc, &b| acc * 10 + (b - b'0') as u128)
        }

        let mut values: Vec<f64> = std::vec![
            1e-20,
            1e40,
            1e-300,
            1e300,
            0.1,
            -0.0,
            3.0,
            138.0,
            148.5,
            151.5,
            1.5e10,
            -2.5e-15,
            123456789.123456,
            1234567890123456.0,
            9999999999999998.0,
            f64::MIN_POSITIVE,
            f64::MAX,
            f64::EPSILON,
            core::f64::consts::PI,
            core::f64::consts::E,
            // The exact values an independent randomized sweep found this
            // formatter disagreeing with DuckDB/Python/`std` on (a
            // tie-break bias, always toward the digit one *below* the
            // correct nearest one).
            1.3398922278945227e-248,
            1.2933663726238107e+51,
            1.3687101854960292e-12,
            9.114560759530303e-237,
        ];
        // A fixed-seed linear congruential generator (Numerical Recipes'
        // constants) over raw `u64` bit patterns, so this sample is
        // reproducible without pulling in an RNG dependency. Interpreting
        // arbitrary bits as an `f64` naturally spreads across the full
        // exponent range (denormal to huge), which is exactly the range
        // this fix touches; NaN/infinite/zero draws are filtered out below.
        // A pure formatting property test like this is cheap per sample
        // (no SQL parsing/execution), so the sample count is kept generous.
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..5000 {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let v = f64::from_bits(seed);
            if v.is_finite() && v != 0.0 {
                values.push(v);
            }
        }

        for v in values {
            let written = written(v);

            // Property 1: round-trips through a plain `f64` parse.
            let got: f64 = written
                .parse()
                .unwrap_or_else(|e| panic!("{v}: wrote {written:?}, failed to reparse: {e}"));
            assert_eq!(
                got.to_bits(),
                v.to_bits(),
                "{v}: wrote {written:?}, round-tripped to {got} (bits {:x} vs {:x})",
                got.to_bits(),
                v.to_bits()
            );

            // Property 2: the exact same significant-digit sequence as
            // std's shortest round-trip -- not just the same count.
            let std_form = std::format!("{v}");
            let ours = significant_digits(&written);
            let theirs = significant_digits(&std_form);
            if ours == theirs {
                continue;
            }
            // `std`'s Display is not a perfectly reliable oracle for this
            // comparison in one specific, rare situation: a genuine exact
            // tie, where `x`'s value is *precisely* halfway between the two
            // shortest-length candidates (only possible when `x` has few
            // enough fractional binary bits that its decimal expansion
            // terminates exactly one digit past the shortest length -- see
            // the tie regression cases in
            // `exact_strings_including_tie_regressions` above). On those
            // specific cases, confirmed directly with the `duckdb` CLI,
            // `std` disagrees with DuckDB/Python/this crate, all three of
            // which round to even; elsewhere `std` is a faithful oracle. So
            // before treating a mismatch as a failure, check -- from `v`'s
            // exact decimal expansion (`exact_tie`) -- whether it is really
            // one of these ties; if so, this crate's choice must still be the
            // even one, and that is asserted for real rather than the
            // mismatch being silently waved through.
            let (ours_val, theirs_val) = (to_u128(&ours), to_u128(&theirs));
            assert!(
                ours.len() == theirs.len() && ours_val.abs_diff(theirs_val) == 1,
                "{v}: wrote {written:?} (digits {:?}), \
                 std's shortest round-trip is {std_form:?} (digits {:?}) -- \
                 not adjacent single-digit candidates, so this is a real mismatch, not a tie",
                String::from_utf8_lossy(&ours),
                String::from_utf8_lossy(&theirs)
            );
            let lo = ours_val.min(theirs_val);
            assert!(
                exact_tie(v, ours.len()),
                "{v}: wrote {written:?}, std wrote {std_form:?}, and they differ by exactly one \
                 unit but {v} is *not* exactly halfway between them -- a real bug, not the \
                 known std-vs-DuckDB tie-break disagreement"
            );
            let even_val = if lo % 2 == 0 { lo } else { lo + 1 };
            assert_eq!(
                ours_val,
                even_val,
                "{v}: confirmed exact tie between {lo} and {}, but this module wrote {written:?} \
                 ({ours_val}), not the even choice ({even_val}) DuckDB/Python use",
                lo + 1
            );
        }
    }

    /// Whether `v` lies *exactly* halfway between two `len`-significant-digit decimals:
    /// its exact decimal expansion (`std` prints every digit when asked for enough
    /// precision) continues after `len` digits with a single `5` and nothing else.
    fn exact_tie(v: f64, len: usize) -> bool {
        let full = std::format!("{:.1100e}", v.abs());
        let mantissa = full.split('e').next().unwrap();
        let digits: Vec<u8> = mantissa.bytes().filter(u8::is_ascii_digit).collect();
        digits[len] == b'5' && digits[len + 1..].iter().all(|&d| d == b'0')
    }

    /// `std`'s shortest digits and leading-digit exponent (`{:e}` is shortest
    /// round-trip too), laid out by this module's own `write_decimal`, so the whole
    /// rendered string can be compared.
    fn std_rendered(sci: &str) -> String {
        let (m, e) = sci.split_once('e').unwrap();
        let neg = m.starts_with('-');
        let digits: Vec<u8> = m.bytes().filter(u8::is_ascii_digit).collect();
        let mut out = Vec::new();
        if neg {
            out.push(b'-');
        }
        write_decimal(&mut out, &digits, e.parse().unwrap());
        String::from_utf8(out).unwrap()
    }

    /// A broad randomized sweep over raw bit patterns -- every exponent, subnormals
    /// included -- comparing the whole rendered text with `std`'s shortest round trip,
    /// for both widths. `AHIRU_FLOAT_FUZZ` raises the sample count (it was run at
    /// 1,000,000 per width when the formatter was rewritten). The only tolerated
    /// difference is an exact tie, which `std` breaks upward and this module to even.
    #[test]
    fn full_text_matches_std_on_random_bit_patterns() {
        let n: u64 =
            std::env::var("AHIRU_FLOAT_FUZZ").ok().and_then(|v| v.parse().ok()).unwrap_or(20_000);
        let mut seed: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for i in 0..n {
            // Raw bit patterns spread over every binary exponent; every other draw is
            // instead an everyday magnitude (a uniform fraction times 10^-3 .. 10^6).
            let v = if i % 2 == 0 {
                f64::from_bits(next())
            } else {
                (next() >> 11) as f64 / (1u64 << 53) as f64 * 10f64.powi((i % 10) as i32 - 3)
            };
            if v.is_finite() && v != 0.0 {
                let got = written(v);
                let want = std_rendered(&std::format!("{v:e}"));
                if got != want {
                    let len = got.bytes().filter(u8::is_ascii_digit).count();
                    assert!(exact_tie(v, len.min(17)), "{v:e}: wrote {got}, std {want}");
                }
            }
            let w = f32::from_bits(next() as u32);
            if w.is_finite() && w != 0.0 {
                let got = written32(w);
                let want = std_rendered(&std::format!("{w:e}"));
                if got != want {
                    let len = got.bytes().filter(u8::is_ascii_digit).count();
                    assert!(exact_tie(w as f64, len.min(9)), "{w:e}: wrote {got}, std {want}");
                }
            }
        }
    }

    /// The case that exposed the old seed-and-repair algorithm: the double just
    /// below 0.1. Normalizing it by repeated `* 10.0` overshot to exactly `10.0`, the
    /// 17-digit seed was then re-pinned by multiplying it by ten (dropping a digit),
    /// and the shortest length was missed. Values next to every power of ten, where
    /// that overshoot happens, are checked the same way.
    #[test]
    fn values_next_to_powers_of_ten_are_shortest() {
        assert_eq!(written(0.09999999999999999), "0.09999999999999999");
        assert_eq!(written(0.9999999999999999), "0.9999999999999999");
        assert_eq!(written(9.999999999999998), "9.999999999999998");
        for k in -300..300 {
            let p = std::format!("1e{k}").parse::<f64>().unwrap();
            for v in [p, f64::from_bits(p.to_bits() - 1), f64::from_bits(p.to_bits() + 1)] {
                assert_eq!(written(v), std_rendered(&std::format!("{v:e}")), "{v:e}");
            }
        }
    }
}
