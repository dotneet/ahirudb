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
//! Ryū/Grisu/Dragon4-style correctly-rounded shortest formatters (and, by
//! construction, Rust `std`'s own `f64` Display, Python's `repr`, and
//! DuckDB's writer).
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

/// The number of significant decimal digits the first-pass conversion
/// (`normalize_and_correct`) computes.
///
/// 17 significant digits always suffice to round-trip any finite `f64`
/// (Steele & White, "How to Print Floating-Point Numbers Accurately"), which
/// is why this is not tuned down further.
const SIG_DIGITS: u32 = 17;
/// `10^(SIG_DIGITS - 1)`: the smallest `SIG_DIGITS`-digit integer.
const SCALE: u128 = 10_000_000_000_000_000;
/// `SCALE` as an `f64`. `SCALE` itself (`10^16`) exceeds `2^53`, so this cast
/// is not bit-exact, but the round-trip correction in `normalize_and_correct`
/// absorbs that; it only needs to be close.
const SCALE_F: f64 = 1e16;

/// Writes a finite `f64` (including positive/negative zero) as shortest
/// round-trip decimal text. Callers are responsible for handling non-finite
/// values (`NaN`/infinities) themselves before calling this -- see the
/// module doc for why that part is not shared.
///
/// The approach, in two stages:
///
/// 1. `normalize_and_correct`: normalize `v` to a mantissa in `[1, 10)` and a
///    decimal exponent using plain `f64` multiply/divide by 10. That is not
///    exact -- each step rounds, and for extreme exponents (subnormal-to-huge)
///    the rounding compounds over up to ~324 steps -- so the candidate
///    `SIG_DIGITS`-digit integer is then corrected to be exact by
///    round-tripping it back through `f64: FromStr` (`core::num::dec2flt`,
///    already linked in for this crate's own CSV/JSONL number parsing) and
///    nudging it until the reparsed value matches the original bit-for-bit.
/// 2. `shortest_digits`: that alone picks *a* valid `SIG_DIGITS`-digit
///    round-tripping representative, not the *shortest* one (there is
///    usually a range of decimal strings that all round-trip to the same
///    `f64`). This crate's own reader does not care which one it gets, but
///    this project's test suite compares CSV/JSONL output against DuckDB
///    byte for byte, and DuckDB (like Rust's own `std` float `Display` and
///    Python's `repr`, via Ryū/Grisu) always emits the *shortest* decimal
///    string that round-trips, choosing the candidate nearest the true
///    value when more than one of that shortest length round-trips (ties
///    broken to even). `shortest_digits` finds the shortest working length
///    cheaply (rounding the `SIG_DIGITS`-digit candidate down, same as
///    before), but the *nearest*-candidate choice at that length is done via
///    exact big-integer arithmetic (`cmp_midpoint`/`nearest_at_length`)
///    against `x`'s exact binary value (`decompose`), not by comparing
///    reparsed floats -- comparing reparsed floats reintroduces the exact
///    ULP-level bias this is trying to avoid (see `normalize_and_correct`'s
///    doc comment for how that was discovered).
///
/// Together this sidesteps implementing a full from-scratch correctly-rounded
/// *shortest* decimal conversion (Ryū/Grisu/Dragon4-style, meaningfully more
/// code and a lookup table) while still landing on the same output.
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
/// Measuring it against `f32` instead -- the only thing that changes, since the exact
/// binary value the digits are chosen nearest to is the same number either way --
/// gives the shortest string that round-trips through the type the value really has.
///
/// Callers handle non-finite values themselves, exactly as for [`write_f64_finite`].
pub(crate) fn write_f32_finite(out: &mut Vec<u8>, v: f64) {
    write_finite(out, v, Prec::F32)
}

/// Which floating-point width a candidate digit string has to round-trip through.
/// It changes *only* the round-trip test; digit generation, the nearest-candidate
/// choice and the fixed-vs-exponential rendering are identical (DuckDB likewise
/// spells an `f32` and an `f64` of the same value the same way once the digits are
/// chosen -- verified against the `duckdb` CLI for the notation thresholds).
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
    let neg = v.is_sign_negative();
    let x = if neg { -v } else { v };
    if neg {
        out.push(b'-');
    }
    let (d, e10) = normalize_and_correct(x);
    let (mantissa, exp2) = decompose(x);
    let (digits, e10) = shortest_digits(d, e10, x, mantissa, exp2, prec);
    write_decimal(out, &digits, e10);
}

/// Whether the decimal `val * 10^last_digit_exp` reads back as exactly `x` at `prec`'s
/// width. For `Prec::F32` the text is parsed straight into an `f32` rather than parsed
/// as an `f64` and then narrowed: those two disagree when the `f64` rounding lands
/// exactly on an `f32` halfway point (classic double rounding), and this comparison is
/// what the whole search hangs on.
fn round_trips(val: u128, last_digit_exp: i32, x: f64, prec: Prec) -> bool {
    let mut buf = [0u8; 48];
    let n = decimal_text(val, last_digit_exp, &mut buf);
    let Ok(s) = core::str::from_utf8(&buf[..n]) else {
        return false;
    };
    match prec {
        Prec::F64 => s.parse::<f64>() == Ok(x),
        Prec::F32 => s.parse::<f32>() == Ok(x as f32),
    }
}

/// Returns `(d, e10)` such that `x` (positive, finite, nonzero) is the
/// nearest `f64` to `d * 10^(e10 - (SIG_DIGITS - 1))` -- i.e. `d` is a
/// `SIG_DIGITS`-digit integer whose leading digit sits at decimal exponent
/// `e10`. This alone is *a* valid round-tripping representative, not
/// necessarily the shortest one; see `shortest_digits`.
fn normalize_and_correct(x: f64) -> (u128, i32) {
    let mut m = x;
    let mut e10: i32 = 0;
    while m >= 10.0 {
        m /= 10.0;
        e10 += 1;
    }
    while m < 1.0 {
        m *= 10.0;
        e10 -= 1;
    }
    // A last-step rounding overshoot lands exactly on 10.0; rare, but cheap to guard.
    if m >= 10.0 {
        m /= 10.0;
        e10 += 1;
    }
    let mut d = (m * SCALE_F) as u128;
    // The normalize loop above runs up to ~324 times for the most extreme
    // exponents (denormal-to-huge), each step rounding by up to ~0.5 ULP, so
    // its worst-case compounded error is a few hundred units of `d`, not the
    // "handful" a single normalization step alone would suggest. Rather than
    // loop that many times one unit at a time, one ratio-scaled refinement
    // step collapses that down to (empirically, and by construction: this is
    // exactly a single step of Newton's method on `redecode(d, e10) == x`)
    // a handful of units first.
    let got0 = redecode(d, e10 - (SIG_DIGITS as i32 - 1));
    if got0 != x && got0 != 0.0 {
        let refined = (d as f64) * (x / got0);
        if refined.is_finite() && refined >= 1.0 {
            d = refined as u128;
        }
    }
    // Bounded, exact fine-tuning for whatever the refinement step above did
    // not already land exactly on.
    for _ in 0..32 {
        let got = redecode(d, e10 - (SIG_DIGITS as i32 - 1));
        if got == x {
            break;
        }
        if got < x {
            d += 1;
        } else if d > 0 {
            d -= 1;
        } else {
            break;
        }
    }
    // A nudge can cross a power-of-ten boundary either way; re-pin the
    // leading digit at decimal exponent `e10` if so.
    if d >= SCALE * 10 {
        d /= 10;
        e10 += 1;
    } else if d != 0 && d < SCALE {
        d *= 10;
        e10 -= 1;
    }
    // `d` is now *a* `SIG_DIGITS`-digit value that round-trips to `x` -- good
    // enough as a seed for `shortest_digits`, which determines both the
    // shortest working length and (via exact big-integer arithmetic, not
    // more `redecode`-based nudging) the precise nearest-to-`x` digit string
    // at that length on its own. An earlier version of this function tried
    // to *also* center `d` here by expanding to the round-trip window's two
    // edges and averaging them; that was not just unnecessary work but
    // actively wrong: the midpoint of the window of decimals that round-trip
    // to `x` is not the same point as `x`'s own exact value, so rounding
    // that midpoint down to fewer digits does not reliably give the
    // nearest-to-`x` shorter candidate either (found via a 400-value
    // randomized sweep against DuckDB, Python, and Rust `std`'s Display, all
    // three of which agree with each other and disagree with the old
    // midpoint-based choice ~14% of the time, always in the same direction).
    (d, e10)
}

/// Decomposes `x` (positive, finite, nonzero) into `(mantissa, exp2)` such
/// that `x == mantissa * 2^exp2` *exactly* -- read directly off the IEEE 754
/// bit layout, not approximated by any floating-point arithmetic. This is
/// the exact value `shortest_digits`' big-integer nearest-candidate
/// comparison (`cmp_midpoint`) is anchored on, which is what makes it immune
/// to the kind of ULP-level bias a `redecode`-based (parse-and-compare)
/// comparison can reintroduce.
fn decompose(x: f64) -> (u64, i32) {
    let bits = x.to_bits();
    let biased_exp = ((bits >> 52) & 0x7FF) as i32;
    let frac = bits & 0x000F_FFFF_FFFF_FFFF;
    if biased_exp == 0 {
        // Subnormal: no implicit leading 1 bit, and the exponent is pinned
        // to the smallest normal exponent's value rather than decoded from
        // `biased_exp`.
        (frac, -1074)
    } else {
        (frac | (1u64 << 52), biased_exp - 1075)
    }
}

/// Finds the shortest decimal digit string that still round-trips to `x`,
/// choosing (once a length is known to work at all) the candidate nearest
/// `x`'s exact value at that length, starting from the `SIG_DIGITS`-digit
/// `d` (exact for `x`, from `normalize_and_correct`) as a seed.
///
/// Two passes, deliberately kept separate:
///
/// 1. **Which length is shortest?** Tries every length from 1 up to
///    `SIG_DIGITS`, rounding `d` to that many significant digits
///    (`round_to_length`, round-half-to-even) and checking whether that
///    candidate (or an immediate neighbor, in case `d`'s limited precision
///    made that rounding step land on a false tie) reconstructs `x` exactly
///    via `redecode`. This existence check only needs "does *something*
///    round-trip here", so a `redecode`-based (reparse-and-compare) check is
///    fine for it -- the length it finds does not depend on *which*
///    candidate happened to match.
/// 2. **Which candidate, exactly, at that length?** Once a length is known
///    to work, `nearest_at_length` finds the specific digit string nearest
///    `x`'s *exact* value there, via big-integer arithmetic anchored on
///    `x`'s exact binary decomposition (`decompose`/`cmp_midpoint`) rather
///    than by comparing reparsed floats. This split matters: an earlier
///    version picked among the length-1 pass's own candidates directly
///    (whichever `redecode`-verified one it found first), which is provably
///    *not* the same thing as nearest -- confirmed by a 400-value randomized
///    sweep against DuckDB, Python's `repr`, and Rust `std`'s Display (all
///    three independently implement correctly-rounded shortest-round-trip
///    and agreed with each other on every case, and disagreed with the old
///    choice on ~14% of values, always in the same direction: the old code
///    was systematically biased toward the low end of the round-trip
///    window rather than picking the point nearest `x`).
fn shortest_digits(
    d: u128,
    e10: i32,
    x: f64,
    mantissa: u64,
    exp2: i32,
    prec: Prec,
) -> (Vec<u8>, i32) {
    for len in 1..=SIG_DIGITS {
        let (primary, carry) = round_to_length(d, len);
        let cand_e10 = e10 + carry;
        if !any_round_trips_at_length(primary, cand_e10, len, x, prec) {
            continue;
        }
        let (val, final_e10) = nearest_at_length(mantissa, exp2, primary, cand_e10, len);
        let last_digit_exp = final_e10 - (len as i32 - 1);
        if round_trips(val, last_digit_exp, x, prec) {
            return (unsigned_digits(val), final_e10);
        }
        // Defensive fallback, not expected to trigger: if *something* at
        // this length round-trips (just confirmed above), the nearest
        // candidate -- being no farther from `x` than that something is --
        // must round-trip too. If it somehow doesn't, fall through to a
        // longer length rather than emit an unverified value.
    }
    // Unreachable in practice -- `len == SIG_DIGITS` is exact by construction
    // of `d` -- but never leave a path that produces no output.
    (unsigned_digits(d), e10)
}

/// Existence check only: does *any* candidate within one unit of `primary`
/// (at the same `len`-digit length) reconstruct `x` exactly? `d` only has
/// `SIG_DIGITS` digits of precision, so `round_to_length`'s own rounding of
/// it can land on a false tie (see `shortest_digits`'s doc comment); the
/// immediate neighbors cover that without needing to know, at this point,
/// which of them (if more than one matches) is actually nearest `x` -- that
/// is `nearest_at_length`'s job, done separately once a length is confirmed
/// to work at all.
fn any_round_trips_at_length(primary: u128, cand_e10: i32, len: u32, x: f64, prec: Prec) -> bool {
    let lower = if len == 1 { 0 } else { 10u128.pow(len - 1) };
    let upper = 10u128.pow(len);
    let mut candidates: [Option<u128>; 3] = [Some(primary), None, None];
    if primary + 1 < upper {
        candidates[1] = Some(primary + 1);
    }
    if primary > lower {
        candidates[2] = Some(primary - 1);
    }
    let last_digit_exp = cand_e10 - (len as i32 - 1);
    candidates.into_iter().flatten().any(|val| round_trips(val, last_digit_exp, x, prec))
}

/// Finds the `len`-digit decimal (leading digit at decimal exponent
/// `seed_e10`, i.e. last digit at `seed_e10 - (len - 1)`) nearest to `x`'s
/// *exact* value `mantissa * 2^exp2`, starting the search from `seed`
/// (expected to already be within a handful of units of the answer, e.g.
/// from `round_to_length`).
///
/// At each step, `cmp_midpoint` exactly compares `x` against the midpoint
/// between two adjacent decimal candidates (no floating-point
/// re-involved -- see its doc comment), which is enough to walk to the true
/// nearest candidate in a small, bounded number of steps: check whether `x`
/// is above the midpoint between `val` and `val + 1` (if so, the answer is
/// higher -- move up and repeat) or, if not, whether `x` is also below the
/// midpoint between `val - 1` and `val` (if so, the answer is lower -- move
/// down and repeat); once neither holds, `val` is nearest. An exact tie at
/// either boundary is broken to the even candidate, matching Ryū/`std`/
/// DuckDB/Python's convention.
fn nearest_at_length(mantissa: u64, exp2: i32, seed: u128, seed_e10: i32, len: u32) -> (u128, i32) {
    let mut val = seed;
    let k = seed_e10 - (len as i32 - 1);
    for _ in 0..8 {
        match cmp_midpoint(mantissa, exp2, val, k) {
            core::cmp::Ordering::Greater => {
                val += 1;
            }
            core::cmp::Ordering::Equal => {
                if !val.is_multiple_of(2) {
                    val += 1;
                }
                break;
            }
            core::cmp::Ordering::Less => {
                if val == 0 {
                    break;
                }
                match cmp_midpoint(mantissa, exp2, val - 1, k) {
                    core::cmp::Ordering::Less => {
                        val -= 1;
                    }
                    core::cmp::Ordering::Equal => {
                        if (val - 1).is_multiple_of(2) {
                            val -= 1;
                        }
                        break;
                    }
                    core::cmp::Ordering::Greater => break,
                }
            }
        }
    }
    let mut e10 = seed_e10;
    // The walk above can cross a power-of-ten boundary (e.g. `x` nearest to
    // exactly `10^len` at this precision); re-pin to exactly `len` digits.
    let upper = 10u128.pow(len);
    let lower = if len == 1 { 0 } else { 10u128.pow(len - 1) };
    if val >= upper {
        val /= 10;
        e10 += 1;
    } else if val != 0 && val < lower {
        val *= 10;
        e10 -= 1;
    }
    (val, e10)
}

/// Compares `x`'s exact value (`mantissa * 2^exp2`) against the midpoint
/// between the decimal candidates `d_cand` and `d_cand + 1` at scale
/// `10^k` -- i.e. against `(2 * d_cand + 1) * 10^k / 2` -- computed exactly
/// via a small fixed-precision big-integer comparison (`Big`). `Ordering::
/// Less` means `x` is closer to `d_cand`; `Greater` means closer to
/// `d_cand + 1`; `Equal` is an exact tie.
///
/// This is deliberately *not* implemented by reparsing decimal strings back
/// to `f64` and comparing floats: that reintroduces up to half a ULP of
/// rounding error into the very comparison meant to resolve sub-ULP
/// ambiguity, which is exactly the bug this function replaces (see
/// `shortest_digits`'s doc comment).
///
/// Derivation: comparing `2x` against `(2*d_cand+1) * 10^k` is equivalent to
/// comparing `mantissa * 2^(exp2+1)` against `(2*d_cand+1) * 2^k * 5^k`.
/// Negative exponents on either side are cleared by multiplying *both*
/// sides by the same power of 2/5 (which does not change the comparison),
/// leaving two non-negative-integer big numbers to compare directly.
fn cmp_midpoint(mantissa: u64, exp2: i32, d_cand: u128, k: i32) -> core::cmp::Ordering {
    let mut lhs_pow2 = exp2 + 1;
    let mut lhs_pow5: i32 = 0;
    let mut rhs_pow2: i32 = 0;
    let mut rhs_pow5: i32 = 0;
    if k >= 0 {
        rhs_pow2 += k;
        rhs_pow5 += k;
    } else {
        lhs_pow2 += -k;
        lhs_pow5 += -k;
    }
    if lhs_pow2 < 0 {
        rhs_pow2 += -lhs_pow2;
        lhs_pow2 = 0;
    }

    let mut lhs = Big::from_u64(mantissa);
    lhs.mul_pow5(lhs_pow5 as u32);
    lhs.shl(lhs_pow2 as u32);

    let mut rhs = Big::from_u128(2 * d_cand + 1);
    rhs.mul_pow5(rhs_pow5 as u32);
    rhs.shl(rhs_pow2 as u32);

    lhs.cmp(&rhs)
}

/// A minimal growable, little-endian, base-`2^32` unsigned big integer.
///
/// This exists solely to make `cmp_midpoint` exact: comparing the huge
/// integers that appear once `x`'s binary exponent (up to ~1074) and a
/// decimal candidate's power-of-five scaling (up to ~324) are cleared into
/// plain integers is well outside `u128`'s ~38 decimal digits, but the
/// *operations* actually needed -- multiply by a small constant, and
/// compare -- are a small fraction of a general-purpose big-integer
/// library. Not performance-tuned (`mul_pow5`/`shl` multiply one small
/// factor at a time rather than batching into larger chunks): this runs at
/// most a handful of times per `f64` written, not in a hot loop.
#[derive(Clone)]
struct Big {
    limbs: Vec<u32>,
}

impl Big {
    fn from_u64(v: u64) -> Self {
        let mut limbs = vec![v as u32, (v >> 32) as u32];
        Self::trim(&mut limbs);
        Big { limbs }
    }

    fn from_u128(v: u128) -> Self {
        let mut limbs = vec![v as u32, (v >> 32) as u32, (v >> 64) as u32, (v >> 96) as u32];
        Self::trim(&mut limbs);
        Big { limbs }
    }

    fn trim(limbs: &mut Vec<u32>) {
        while limbs.len() > 1 && *limbs.last().unwrap() == 0 {
            limbs.pop();
        }
    }

    fn mul_small(&mut self, m: u32) {
        let mut carry: u64 = 0;
        for limb in self.limbs.iter_mut() {
            let prod = (*limb as u64) * (m as u64) + carry;
            *limb = prod as u32;
            carry = prod >> 32;
        }
        if carry > 0 {
            self.limbs.push(carry as u32);
        }
        Self::trim(&mut self.limbs);
    }

    /// Multiplies by `5^n`, one factor of 5 at a time.
    fn mul_pow5(&mut self, n: u32) {
        for _ in 0..n {
            self.mul_small(5);
        }
    }

    /// Multiplies by `2^n` (a left shift), one bit at a time.
    fn shl(&mut self, n: u32) {
        for _ in 0..n {
            self.mul_small(2);
        }
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

/// Rounds `d` (a `SIG_DIGITS`-digit integer) to `len` significant digits,
/// round-half-to-even. Returns `(rounded, carry)`; `carry` is `1` if
/// rounding up overflowed into one more digit than `len` (e.g.
/// `999... -> 1000...`), meaning the result's leading digit moved up one
/// decimal exponent -- the caller adds `carry` to its exponent, and the
/// returned value is exactly `len` digits long either way.
fn round_to_length(d: u128, len: u32) -> (u128, i32) {
    if len >= SIG_DIGITS {
        return (d, 0);
    }
    let divisor = 10u128.pow(SIG_DIGITS - len);
    let q = d / divisor;
    let r = d % divisor;
    let half = divisor / 2;
    let round_up = r > half || (r == half && q % 2 == 1);
    let q = if round_up { q + 1 } else { q };
    if q >= 10u128.pow(len) {
        (q / 10, 1)
    } else {
        (q, 0)
    }
}

/// Writes `digits(val) * 10^last_digit_exp` into `buf` and returns its length, so a
/// candidate from `normalize_and_correct`/`shortest_digits` can be parsed back and
/// checked against the original value (`round_trips`). `FromStr`'s grammar accepts a
/// bare `<digits>e<exp>` with no decimal point, so this does not need to place one.
fn decimal_text(val: u128, last_digit_exp: i32, buf: &mut [u8; 48]) -> usize {
    let mut n = 0usize;
    for &b in &unsigned_digits(val) {
        buf[n] = b;
        n += 1;
    }
    buf[n] = b'e';
    n += 1;
    if last_digit_exp < 0 {
        buf[n] = b'-';
        n += 1;
    }
    for &b in &unsigned_digits(last_digit_exp.unsigned_abs() as u128) {
        buf[n] = b;
        n += 1;
    }
    n
}

/// Parses `digits(val) * 10^last_digit_exp` back into an `f64`.
fn redecode(val: u128, last_digit_exp: i32) -> f64 {
    let mut buf = [0u8; 48];
    let n = decimal_text(val, last_digit_exp, &mut buf);
    core::str::from_utf8(&buf[..n]).ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(f64::NAN)
}

/// `v`'s decimal digits, most significant first, with no leading zero (`0` itself renders as `"0"`).
fn unsigned_digits(v: u128) -> Vec<u8> {
    let mut buf = [0u8; 40];
    let mut n = 0usize;
    let mut u = v;
    loop {
        buf[n] = b'0' + (u % 10) as u8;
        n += 1;
        u /= 10;
        if u == 0 {
            break;
        }
    }
    buf[..n].iter().rev().copied().collect()
}

/// Renders `0.<digits> * 10^(e10 + 1)` (see `normalize_and_correct`) as
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
            // before treating a mismatch as a failure, check -- via this
            // crate's own already DuckDB-verified exact big-integer
            // comparison (`cmp_midpoint`), not `std` -- whether it is
            // really one of these ties; if so, this crate's choice must
            // still be the even one, and that is asserted for real rather
            // than the mismatch being silently waved through.
            let (ours_val, theirs_val) = (to_u128(&ours), to_u128(&theirs));
            assert!(
                ours.len() == theirs.len() && ours_val.abs_diff(theirs_val) == 1,
                "{v}: wrote {written:?} (digits {:?}), \
                 std's shortest round-trip is {std_form:?} (digits {:?}) -- \
                 not adjacent single-digit candidates, so this is a real mismatch, not a tie",
                String::from_utf8_lossy(&ours),
                String::from_utf8_lossy(&theirs)
            );
            let x = v.abs();
            let (mantissa, exp2) = decompose(x);
            let (_, e10) = normalize_and_correct(x);
            let len = ours.len() as u32;
            let k = e10 - (len as i32 - 1);
            let lo = ours_val.min(theirs_val);
            assert_eq!(
                cmp_midpoint(mantissa, exp2, lo, k),
                core::cmp::Ordering::Equal,
                "{v}: wrote {written:?}, std wrote {std_form:?}, and they differ by exactly one \
                 unit but `cmp_midpoint` says this is *not* an exact tie -- a real bug, not the \
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
}
