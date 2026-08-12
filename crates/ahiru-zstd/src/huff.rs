//! The Huffman decoder for literals.
//!
//! The table is fully expanded to 2^max_bits (one symbol per peek). max_bits is at
//! most 11 per the spec, so 2 KiB x 2 suffices -- smaller and faster than walking a tree.

use crate::bits::Reverse;
use crate::fse;
use crate::prelude::*;
use crate::Error;

/// The maximum code length the spec allows.
const MAX_BITS: u32 = 11;

pub struct Huff {
    max_bits: u32,
    mask: usize,
    sym: Vec<u8>,
    nb: Vec<u8>,
}

#[inline]
fn highest_bit(v: u32) -> u32 {
    31u32.saturating_sub(v.leading_zeros())
}

/// Reads a Huffman tree description and returns the table and the bytes consumed.
pub fn read_table(src: &[u8]) -> Result<(Huff, usize), Error> {
    let head = match src.first() {
        Some(v) => *v,
        None => return Err(Error::UnexpectedEof),
    };
    // 256 weights are needed: up to 255 read plus the final one that is reconstructed.
    let mut w = [0u8; 256];
    let n;
    let used;
    if head < 128 {
        // head is the byte count of the FSE stream.
        let size = head as usize;
        let body = src.get(1..1 + size).ok_or(Error::UnexpectedEof)?;
        let (t, hdr) = fse::read_table(body, 6, 255)?;
        let stream = body.get(hdr..).ok_or(Error::UnexpectedEof)?;
        n = fse::decode_interleaved(&t, stream, &mut w[..255])?;
        used = 1 + size;
        stat!(crate::stats::HUF_W_FSE);
    } else {
        // Direct representation. 4 bits per weight, high nibble first.
        n = head as usize - 127;
        let bytes = n.div_ceil(2);
        let body = src.get(1..1 + bytes).ok_or(Error::UnexpectedEof)?;
        for (i, slot) in w.iter_mut().enumerate().take(n) {
            *slot = if i % 2 == 0 { body[i / 2] >> 4 } else { body[i / 2] & 0x0F };
        }
        used = 1 + bytes;
        stat!(crate::stats::HUF_W_DIRECT);
    }
    let h = build(&mut w, n)?;
    Ok((h, used))
}

/// Builds the decoding table from the weight sequence.
///
/// The last symbol's weight is not written; it is reconstructed as "whatever fills
/// the sum up to the next power of two". If that does not work out, the input is corrupt.
fn build(w: &mut [u8; 256], n: usize) -> Result<Huff, Error> {
    if n == 0 || n > 255 {
        return Err(Error::BadHuffman);
    }
    let mut sum: u32 = 0;
    for &x in w.iter().take(n) {
        if x as u32 > MAX_BITS {
            return Err(Error::BadHuffman);
        }
        if x > 0 {
            sum += 1 << (x - 1);
        }
    }
    if sum == 0 {
        return Err(Error::BadHuffman);
    }
    let max_bits = highest_bit(sum) + 1;
    if max_bits > MAX_BITS {
        return Err(Error::BadHuffman);
    }
    let left = (1u32 << max_bits) - sum;
    if !left.is_power_of_two() {
        return Err(Error::BadHuffman);
    }
    w[n] = (highest_bit(left) + 1) as u8;
    let num = n + 1;

    // The code length for weight w is max_bits + 1 - w. Weight 0 means an unused symbol.
    let mut bits = [0u8; 256];
    let mut rank_count = [0u32; MAX_BITS as usize + 1];
    for s in 0..num {
        let b = if w[s] > 0 { max_bits + 1 - w[s] as u32 } else { 0 };
        bits[s] = b as u8;
        if b > 0 {
            rank_count[b as usize] += 1;
        }
    }

    // Codes are packed from 0 in order of decreasing bit length (= increasing weight).
    let size = 1usize << max_bits;
    let mut rank_idx = [0u32; MAX_BITS as usize + 1];
    let mut i = max_bits as usize;
    while i >= 1 {
        rank_idx[i - 1] = rank_idx[i] + rank_count[i] * (1u32 << (max_bits as usize - i));
        i -= 1;
    }
    if rank_idx[0] as usize != size {
        return Err(Error::BadHuffman);
    }

    let mut sym = vec![0u8; size];
    let mut nb = vec![0u8; size];
    for (s, &b) in bits.iter().enumerate().take(num) {
        if b == 0 {
            continue;
        }
        let code = rank_idx[b as usize] as usize;
        let len = 1usize << (max_bits - b as u32);
        if code + len > size {
            return Err(Error::BadHuffman);
        }
        for k in code..code + len {
            sym[k] = s as u8;
            nb[k] = b;
        }
        rank_idx[b as usize] += len as u32;
    }

    Ok(Huff { max_bits, mask: size - 1, sym, nb })
}

impl Huff {
    /// Expands `n` symbols from the backwards stream and appends them to `out`.
    ///
    /// The caller has already guaranteed the appended amount against `out`'s capacity
    /// (the literals region is capped at the 128 KiB block limit).
    pub fn decode_stream(&self, src: &[u8], n: usize, out: &mut Vec<u8>) -> Result<(), Error> {
        let mut r = Reverse::new(src)?;
        for _ in 0..n {
            // One symbol is at least one bit. Running out means corruption.
            if r.off() <= 0 {
                return Err(Error::BadHuffman);
            }
            let i = (r.peek(self.max_bits) as usize) & self.mask;
            out.push(self.sym[i]);
            r.skip(self.nb[i] as u32);
        }
        // The stream is consumed exactly. Both leftover and shortfall are corruption.
        if r.off() != 0 {
            return Err(Error::BadHuffman);
        }
        Ok(())
    }
}
