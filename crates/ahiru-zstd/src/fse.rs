//! The FSE (Finite State Entropy) decoding table.
//!
//! The table is three arrays: state -> (symbol, and the bit count and base used to form the next state).
//! The state count is a power of two, so indices are masked and stay in bounds even
//! without bounds checks (`sym`/`nb`/`base` are nonetheless always kept the same length).

use crate::bits::{Forward, Reverse};
use crate::prelude::*;
use crate::Error;

/// The largest symbol that can appear in the normalized counts. Up to 255 per the spec.
const MAX_SYMBOLS: usize = 256;

pub struct Table {
    accuracy: u32,
    /// State count - 1. Indices are masked with this.
    mask: u16,
    sym: Vec<u8>,
    nb: Vec<u8>,
    base: Vec<u16>,
}

impl Table {
    /// For RLE mode. The state is always 0 and the extra bit count is 0, returning the same symbol.
    pub fn rle(s: u8) -> Table {
        Table { accuracy: 0, mask: 0, sym: vec![s], nb: vec![0], base: vec![0] }
    }

    pub fn init(&self, r: &mut Reverse) -> u16 {
        (r.read(self.accuracy) as u16) & self.mask
    }

    pub fn peek(&self, st: u16) -> u8 {
        self.sym[(st & self.mask) as usize]
    }

    pub fn update(&self, st: &mut u16, r: &mut Reverse) {
        let i = (*st & self.mask) as usize;
        let n = self.nb[i] as u32;
        *st = (self.base[i].wrapping_add(r.read(n) as u16)) & self.mask;
    }
}

/// The position of the highest set bit. `v == 0` is ruled out by the caller, but
/// this saturates so debug builds do not panic on underflow.
#[inline]
fn highest_bit(v: u32) -> u32 {
    31u32.saturating_sub(v.leading_zeros())
}

/// Builds the decoding table from normalized counts. `norm[i] == -1` means
/// "probability below 1/N", and those are assigned one cell at a time from the end of the table.
fn build(norm: &[i16], n: usize, accuracy: u32) -> Result<Table, Error> {
    let size = 1usize << accuracy;
    let mut sym = vec![0u8; size];
    let mut nb = vec![0u8; size];
    let mut base = vec![0u16; size];
    let mut desc = [0u16; MAX_SYMBOLS];

    let mut high = size;
    for (s, &p) in norm.iter().enumerate().take(n) {
        if p == -1 {
            high -= 1;
            sym[high] = s as u8;
            desc[s] = 1;
        }
    }

    // Cells are assigned scattered by a step rather than linearly. step is coprime
    // with size (always odd, since accuracy >= 5), so the cycle visits every cell.
    let step = (size >> 1) + (size >> 3) + 3;
    let mask = size - 1;
    let mut pos = 0usize;
    for (s, &p) in norm.iter().enumerate().take(n) {
        if p <= 0 {
            continue;
        }
        desc[s] = p as u16;
        for _ in 0..p {
            sym[pos] = s as u8;
            // Unreachable when high == 0 (there would be no positive probability).
            // Even so, cap the iterations so corrupt input cannot loop forever.
            let mut guard = size + 1;
            loop {
                pos = (pos + step) & mask;
                if pos < high {
                    break;
                }
                guard -= 1;
                if guard == 0 {
                    return Err(Error::BadFse);
                }
            }
        }
    }
    // Using up every cell wraps around back to 0.
    if pos != 0 {
        return Err(Error::BadFse);
    }

    for i in 0..size {
        let s = sym[i] as usize;
        let d = desc[s];
        desc[s] = d + 1;
        let bits = accuracy - highest_bit(d as u32);
        nb[i] = bits as u8;
        base[i] = (((d as u32) << bits) - size as u32) as u16;
    }

    Ok(Table { accuracy, mask: mask as u16, sym, nb, base })
}

/// Builds a table from a predefined distribution.
pub fn predefined(norm: &[i16], accuracy: u32) -> Result<Table, Error> {
    build(norm, norm.len(), accuracy)
}

/// Reads an FSE table description. Returns the table and the bytes consumed.
///
/// The description is a forward LSB-first bit sequence that ends rounded up to a byte boundary.
pub fn read_table(
    src: &[u8],
    max_accuracy: u32,
    max_symbol: usize,
) -> Result<(Table, usize), Error> {
    let mut r = Forward::new(src);
    let accuracy = r.read(4)? + 5;
    if accuracy > max_accuracy {
        return Err(Error::BadFse);
    }

    let mut norm = [0i16; MAX_SYMBOLS];
    let mut remaining: i32 = 1 << accuracy;
    let mut sym = 0usize;
    while remaining > 0 && sym <= max_symbol {
        // A variable-length code whose width shrinks with the remaining probability mass.
        let bits = highest_bit(remaining as u32 + 1) + 1;
        let raw = r.read(bits)?;
        let low_mask = (1u32 << (bits - 1)) - 1;
        let threshold = (1u32 << bits) - 1 - (remaining as u32 + 1);
        let val = if (raw & low_mask) < threshold {
            r.rewind1();
            raw & low_mask
        } else if raw > low_mask {
            raw - threshold
        } else {
            raw
        };

        // A value of 0 means -1, i.e. "probability below 1/N".
        let proba = val as i32 - 1;
        remaining -= proba.abs();
        if remaining < 0 {
            return Err(Error::BadFse);
        }
        norm[sym] = proba as i16;
        sym += 1;

        if proba == 0 {
            // A zero-probability symbol is followed by a 2-bit skip count.
            // A value of 3 means reading 2 more bits (chaining up to 3 at a time).
            let mut repeat = r.read(2)?;
            loop {
                let mut i = 0;
                while i < repeat && sym <= max_symbol {
                    norm[sym] = 0;
                    sym += 1;
                    i += 1;
                }
                if repeat == 3 {
                    repeat = r.read(2)?;
                } else {
                    break;
                }
            }
        }
    }
    if remaining != 0 || sym > max_symbol + 1 {
        return Err(Error::BadFse);
    }

    let t = build(&norm, sym, accuracy)?;
    Ok((t, r.bytes_used()))
}

/// Two-state interleaved FSE expansion for Huffman weights.
///
/// The symbol count is not known in advance; it ends the moment the bitstream is
/// overrun (the spec defines it that way). The length of `out` caps it.
pub fn decode_interleaved(t: &Table, stream: &[u8], out: &mut [u8]) -> Result<usize, Error> {
    let mut r = Reverse::new(stream)?;
    let mut s1 = t.init(&mut r);
    let mut s2 = t.init(&mut r);
    if r.off() < 0 {
        return Err(Error::BadFse);
    }

    let mut n = 0usize;
    macro_rules! put {
        ($v:expr) => {{
            if n >= out.len() {
                return Err(Error::BadFse);
            }
            out[n] = $v;
            n += 1;
        }};
    }
    loop {
        put!(t.peek(s1));
        t.update(&mut s1, &mut r);
        if r.off() < 0 {
            put!(t.peek(s2));
            break;
        }
        put!(t.peek(s2));
        t.update(&mut s2, &mut r);
        if r.off() < 0 {
            put!(t.peek(s1));
            break;
        }
    }
    Ok(n)
}
