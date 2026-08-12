//! Two kinds of bit reader.
//!
//! zstd mixes two conventions. Only the FSE table description is an LSB-first
//! stream read front to back; the Huffman and sequences payloads are read
//! backwards from the last byte. Both are confined here so callers above only
//! ever deal in bit counts.

use crate::Error;

/// Treats `src` as a bit sequence whose bit 0 is the least significant bit of
/// byte 0, and reads `n` bits (n <= 32) starting at bit position `off`.
///
/// Out-of-range bits read as 0. When the backwards reader steps into the trailing
/// padding, the spec says to interpret it as zero fill (bounds checking is the caller's job).
fn read_le(src: &[u8], off: usize, n: u32) -> u64 {
    let mut res: u64 = 0;
    let mut got: u32 = 0;
    let mut idx = off / 8;
    let mut shift = (off % 8) as u32;
    while got < n {
        let b = match src.get(idx) {
            Some(v) => *v as u64,
            None => 0,
        };
        res |= (b >> shift) << got;
        got += 8 - shift;
        shift = 0;
        idx += 1;
    }
    if n >= 64 {
        res
    } else {
        res & ((1u64 << n) - 1)
    }
}

/// Forward LSB-first reader. For FSE table descriptions only.
pub struct Forward<'a> {
    src: &'a [u8],
    bit: usize,
}

impl<'a> Forward<'a> {
    pub fn new(src: &'a [u8]) -> Self {
        Forward { src, bit: 0 }
    }

    /// Reads `n` bits (n <= 32). `Err` once the input runs out.
    pub fn read(&mut self, n: u32) -> Result<u32, Error> {
        if n == 0 {
            return Ok(0);
        }
        let end = self.bit + n as usize;
        if end > self.src.len() * 8 {
            return Err(Error::UnexpectedEof);
        }
        let v = read_le(self.src, self.bit, n);
        self.bit = end;
        Ok(v as u32)
    }

    /// Puts one bit back. Used by the table description's "small values are one bit shorter" coding.
    pub fn rewind1(&mut self) {
        self.bit = self.bit.saturating_sub(1);
    }

    /// Bytes consumed, rounded up to a byte boundary.
    pub fn bytes_used(&self) -> usize {
        self.bit.div_ceil(8)
    }
}

/// The backwards reader.
///
/// The highest set bit of the last byte is the stream end marker, and everything
/// above it is padding. `off` is "bits not yet read"; going negative signals that
/// the stream was overrun (which is the only way to learn FSE's symbol count).
pub struct Reverse<'a> {
    src: &'a [u8],
    off: i64,
}

impl<'a> Reverse<'a> {
    pub fn new(src: &'a [u8]) -> Result<Self, Error> {
        let last = match src.last() {
            Some(v) => *v,
            None => return Err(Error::UnexpectedEof),
        };
        // A stream without the marker set is corrupt.
        if last == 0 {
            return Err(Error::BadBitstream);
        }
        let hi = 7 - last.leading_zeros() as i64;
        let off = (src.len() as i64) * 8 - (8 - hi);
        Ok(Reverse { src, off })
    }

    /// Peeks the top `n` bits without consuming. Anything missing is zero-filled at the bottom.
    pub fn peek(&self, n: u32) -> u64 {
        if n == 0 {
            return 0;
        }
        let start = self.off - n as i64;
        if start >= 0 {
            return read_le(self.src, start as usize, n);
        }
        let avail = if self.off > 0 { self.off as u32 } else { 0 };
        let v = read_le(self.src, 0, avail);
        let sh = n - avail;
        if sh >= 64 {
            0
        } else {
            v << sh
        }
    }

    pub fn skip(&mut self, n: u32) {
        self.off -= n as i64;
    }

    pub fn read(&mut self, n: u32) -> u64 {
        let v = self.peek(n);
        self.skip(n);
        v
    }

    /// Bits remaining. Negative means the stream was overrun.
    pub fn off(&self) -> i64 {
        self.off
    }
}
