//! Decoders for Parquet page encodings.
//!
//! Handles the RLE/bit-packing hybrid (definition levels and dictionary indices)
//! and the DELTA_* family. PLAIN is just a fixed-length copy, so it's handled on
//! the `reader.rs` side.
//!
//! Input comes from the network and cannot be trusted. Every read performs bounds
//! checking, and any corruption always returns `Err` (never panics, never loops
//! forever). In particular, even if a "run that produces zero values" repeats
//! consecutively, the header always consumes at least one byte, so the loop is
//! guaranteed to terminate.

use crate::prelude::*;
use crate::vector::{Bitmap, BytesData};

/// Number of bits needed to represent `max_value`.
pub fn bit_width(max_value: u32) -> u8 {
    (u32::BITS - max_value.leading_zeros()) as u8
}

/// Upper bound on the number of values a single stream may declare. Casting the
/// declared value directly to `usize` would truncate on wasm32 (usize = 32 bits),
/// so we reject oversized values up front.
const MAX_VALUES: u64 = 1 << 31;

/// Upper bound on the DELTA_BINARY_PACKED block size. Kept small enough that the
/// miniblock's intermediate calculations never overflow 32 bits.
const MAX_BLOCK: u64 = 1 << 20;

// --- Reading bit streams ----------------------------------------------------

/// An LSB-first bit-stream reader. All Parquet bit-packing uses this order: the
/// first value goes into the low bits of the first byte.
struct BitReader<'a> {
    buf: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    #[inline]
    fn new(buf: &'a [u8]) -> Self {
        BitReader { buf, bit_pos: 0 }
    }

    /// Reads the next `width` bits (0..=64). `width == 0` consumes no bits and
    /// returns 0 (an all-zero run / miniblock).
    fn read(&mut self, width: u8) -> Result<u64> {
        let want = width as u32;
        let mut got = 0u32;
        let mut v = 0u64;
        while got < want {
            let byte = self.bit_pos >> 3;
            ensure!(byte < self.buf.len(), UnexpectedEof, byte);
            let off = (self.bit_pos & 7) as u32;
            let take = core::cmp::min(8 - off, want - got);
            let chunk = ((self.buf[byte] >> off) as u64) & ((1u64 << take) - 1);
            v |= chunk << got;
            got += take;
            self.bit_pos += take as usize;
        }
        Ok(v)
    }
}

/// ULEB128 variable-length integer. Advances `pos`. Treated as corrupted past 10 bytes.
fn uleb128(src: &[u8], pos: &mut usize) -> Result<u64> {
    let mut result = 0u64;
    let mut shift = 0u32;
    loop {
        ensure!(shift < 64, BadCompressedData, *pos);
        ensure!(*pos < src.len(), UnexpectedEof, *pos);
        let b = src[*pos];
        *pos += 1;
        // The tenth byte contributes only bit 63. Without this check a
        // malformed value with any of the other payload bits set would
        // silently wrap in the u64 shift and be interpreted as a valid small
        // number by the decoder.
        ensure!(shift < 63 || b <= 1, BadCompressedData, *pos - 1);
        result |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Ok(result);
        }
        shift += 7;
    }
}

/// Zigzag decoding. Every DELTA_* header uses this format.
#[inline]
fn zigzag(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

// --- RLE / bit-packing hybrid -----------------------------------------------

/// RLE / bit-packing hybrid decoder.
/// Used for both definition levels and dictionary indices.
///
/// A run can hold more values than requested, so leftovers are kept internally to
/// support streaming. Bit-packed runs align to a byte boundary every 8 values
/// (= `bit_width` bytes), so a single 8-value buffer is enough to carry the
/// leftover.
pub struct RleDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    bit_width: u8,
    /// The RLE run waiting to be expanded.
    rle_value: u32,
    rle_left: usize,
    /// Number of unexpanded values remaining in the bit-packed run (always a multiple of 8).
    packed_left: usize,
    /// The most recently expanded 8 values, and the range not yet consumed.
    group: [u32; 8],
    group_pos: usize,
    group_len: usize,
}

impl<'a> RleDecoder<'a> {
    /// `data` is the RLE stream body, with the length prefix removed.
    pub fn new(data: &'a [u8], bit_width: u8) -> Self {
        RleDecoder {
            data,
            pos: 0,
            bit_width,
            rle_value: 0,
            rle_left: 0,
            packed_left: 0,
            group: [0; 8],
            group_pos: 0,
            group_len: 0,
        }
    }

    /// Reads the next run header. A run that produces zero values is technically
    /// legal, but since the header always advances at least one byte, the caller's
    /// loop still terminates.
    fn next_run(&mut self) -> Result<()> {
        let header = uleb128(self.data, &mut self.pos)?;
        let avail = (self.data.len() - self.pos) as u64;
        if header & 1 == 1 {
            // A bit-packed run. The value count is groups x 8, where each group
            // occupies exactly `bit_width` bytes.
            let groups = header >> 1;
            ensure!(groups <= MAX_VALUES >> 3, LimitExceeded, self.pos);
            // The full run length is already known, so check it all at once here.
            ensure!(groups * self.bit_width as u64 <= avail, UnexpectedEof, self.pos);
            self.packed_left = (groups as usize) * 8;
        } else {
            // An RLE run. The value is ceil(bit_width / 8) bytes, little-endian.
            let count = header >> 1;
            ensure!(count <= MAX_VALUES, LimitExceeded, self.pos);
            let nbytes = (self.bit_width as usize).div_ceil(8);
            ensure!(nbytes as u64 <= avail, UnexpectedEof, self.pos);
            let mut v = 0u32;
            for i in 0..nbytes {
                v |= (self.data[self.pos + i] as u32) << (i * 8);
            }
            self.pos += nbytes;
            // RLE stores a whole byte (or bytes) even when the declared bit
            // width is smaller. The unused high bits must be zero; accepting
            // them would turn malformed definition levels into NULLs and
            // could also produce out-of-range dictionary indices.
            if self.bit_width < 32 {
                ensure!(v < (1u32 << self.bit_width), BadCompressedData, self.pos);
            }
            self.rle_value = v;
            self.rle_left = count as usize;
        }
        Ok(())
    }

    /// Expands 8 values from a bit-packed run into `group`.
    fn fill_group(&mut self) -> Result<()> {
        let nbytes = self.bit_width as usize;
        ensure!(nbytes <= self.data.len() - self.pos, UnexpectedEof, self.pos);
        let mut br = BitReader::new(&self.data[self.pos..self.pos + nbytes]);
        for slot in self.group.iter_mut() {
            *slot = br.read(self.bit_width)? as u32;
        }
        self.pos += nbytes;
        self.packed_left -= 8;
        self.group_pos = 0;
        self.group_len = 8;
        Ok(())
    }

    /// Ensures at least one value is available to take. Prepared in order: leftover
    /// -> RLE run -> new run.
    fn ensure_available(&mut self) -> Result<()> {
        loop {
            if self.group_pos < self.group_len || self.rle_left > 0 {
                return Ok(());
            }
            if self.packed_left > 0 {
                self.fill_group()?;
            } else {
                self.next_run()?;
            }
        }
    }

    /// Appends the next `n` values to `out`.
    pub fn read_u32(&mut self, n: usize, out: &mut Vec<u32>) -> Result<()> {
        // Values are stored in u32, so widths beyond 32 bits are not supported.
        ensure!(self.bit_width <= 32, UnsupportedEncoding);
        let mut need = n;
        while need > 0 {
            self.ensure_available()?;
            if self.group_pos < self.group_len {
                let take = core::cmp::min(need, self.group_len - self.group_pos);
                out.extend_from_slice(&self.group[self.group_pos..self.group_pos + take]);
                self.group_pos += take;
                need -= take;
            } else {
                let take = core::cmp::min(need, self.rle_left);
                for _ in 0..take {
                    out.push(self.rle_value);
                }
                self.rle_left -= take;
                need -= take;
            }
        }
        Ok(())
    }

    /// Reads definition levels and appends `level == max_level` (= value is
    /// present) to `out` as a validity bitmap.
    /// Returns the number of values that were present.
    pub fn read_levels_into(
        &mut self,
        n: usize,
        max_level: u32,
        out: &mut Bitmap,
    ) -> Result<usize> {
        ensure!(self.bit_width <= 32, UnsupportedEncoding);
        let mut need = n;
        let mut present = 0usize;
        while need > 0 {
            self.ensure_available()?;
            if self.group_pos < self.group_len {
                let take = core::cmp::min(need, self.group_len - self.group_pos);
                for i in 0..take {
                    let level = self.group[self.group_pos + i];
                    ensure!(level <= max_level, BadCompressedData);
                    let v = level == max_level;
                    out.push(v);
                    present += v as usize;
                }
                self.group_pos += take;
                need -= take;
            } else {
                // A column with no NULLs collapses into one giant RLE run. Since this is
                // the dominant path, fill it word-at-a-time rather than bit-by-bit.
                let take = core::cmp::min(need, self.rle_left);
                ensure!(self.rle_value <= max_level, BadCompressedData);
                let v = self.rle_value == max_level;
                out.push_n(v, take);
                if v {
                    present += take;
                }
                self.rle_left -= take;
                need -= take;
            }
        }
        Ok(present)
    }
}

// --- DELTA_BINARY_PACKED ----------------------------------------------------

/// An output-side abstraction that keeps a single decoder body shared between i32
/// and i64. Accumulation is always done in i64. Because two's-complement makes the
/// lower-32-bit wraparound match i32's `wrapping_add`, an i32 column only needs a
/// final truncation.
trait DeltaSink {
    /// Upper bound on a miniblock's bit width. A declaration exceeding the physical type's width is corrupted.
    const MAX_BITS: u8;
    fn push_delta(&mut self, v: i64);
}

impl DeltaSink for Vec<i32> {
    const MAX_BITS: u8 = 32;
    #[inline]
    fn push_delta(&mut self, v: i64) {
        self.push(v as i32);
    }
}

impl DeltaSink for Vec<i64> {
    const MAX_BITS: u8 = 64;
    #[inline]
    fn push_delta(&mut self, v: i64) {
        self.push(v);
    }
}

/// The body of DELTA_BINARY_PACKED. Returns the number of bytes consumed.
///
/// The final block may be padded, rounding the value count up. Values beyond the
/// header's declared total are discarded, but the miniblock bytes are still
/// consumed (if the consumed byte count were off, the start position of the
/// following byte stream would be wrong).
fn decode_delta<S: DeltaSink>(src: &[u8], n: usize, out: &mut S) -> Result<usize> {
    let mut pos = 0usize;
    let block_size = uleb128(src, &mut pos)?;
    let miniblocks = uleb128(src, &mut pos)?;
    let total = uleb128(src, &mut pos)?;
    let first = zigzag(uleb128(src, &mut pos)?);

    ensure!(block_size > 0 && block_size <= MAX_BLOCK, BadCompressedData, pos);
    ensure!(miniblocks > 0 && miniblocks <= block_size, BadCompressedData, pos);
    ensure!(block_size % miniblocks == 0, BadCompressedData, pos);
    let per_mini = block_size / miniblocks;
    // The number of values per miniblock is a multiple of 32. This guarantees that
    // a miniblock ends on a byte boundary regardless of bit width.
    ensure!(per_mini % 32 == 0, BadCompressedData, pos);
    ensure!(total <= MAX_VALUES, LimitExceeded, pos);
    ensure!(n as u64 <= total, UnexpectedEof, pos);

    let miniblocks = miniblocks as usize;
    let per_mini = per_mini as usize;
    let mut remaining = total as usize; // Number of unprocessed values (including the first)
    let mut emitted = 0usize;
    let mut last = first;

    if remaining > 0 {
        // Only the first value is embedded directly in the header; no delta is added to it.
        if n > 0 {
            out.push_delta(last);
            emitted = 1;
        }
        remaining -= 1;
    }

    while remaining > 0 {
        let min_delta = zigzag(uleb128(src, &mut pos)?);
        ensure!(miniblocks <= src.len() - pos, UnexpectedEof, pos);
        let widths_at = pos;
        pos += miniblocks;
        for k in 0..miniblocks {
            // A miniblock that becomes unneeded in the final block still has its bit
            // width written, but no data byte follows it at all.
            if remaining == 0 {
                break;
            }
            let w = src[widths_at + k];
            ensure!(w <= S::MAX_BITS, BadCompressedData, widths_at + k);
            let nbytes = per_mini * w as usize / 8;
            ensure!(nbytes <= src.len() - pos, UnexpectedEof, pos);
            let take = core::cmp::min(per_mini, remaining);
            if emitted < n {
                let mut br = BitReader::new(&src[pos..pos + nbytes]);
                let emit = core::cmp::min(take, n - emitted);
                for _ in 0..emit {
                    let d = br.read(w)?;
                    // Per spec, delta addition wraps.
                    last = last.wrapping_add(min_delta).wrapping_add(d as i64);
                    out.push_delta(last);
                }
                emitted += emit;
            }
            pos += nbytes;
            remaining -= take;
        }
    }
    Ok(pos)
}

/// DELTA_BINARY_PACKED (INT32). Returns the number of bytes consumed.
/// Needed by things like DELTA_LENGTH_BYTE_ARRAY, where more data follows.
pub fn decode_delta_binary_packed_i32(src: &[u8], n: usize, out: &mut Vec<i32>) -> Result<usize> {
    decode_delta(src, n, out)
}

/// DELTA_BINARY_PACKED (INT64). Returns the number of bytes consumed.
pub fn decode_delta_binary_packed_i64(src: &[u8], n: usize, out: &mut Vec<i64>) -> Result<usize> {
    decode_delta(src, n, out)
}

// --- DELTA_*_BYTE_ARRAY -----------------------------------------------------

/// DELTA_LENGTH_BYTE_ARRAY.
/// The body is concatenated immediately after the length column (DELTA_BINARY_PACKED).
pub fn decode_delta_length_byte_array(src: &[u8], n: usize, out: &mut BytesData) -> Result<()> {
    let mut lens: Vec<i32> = Vec::new();
    let consumed = decode_delta_binary_packed_i32(src, n, &mut lens)?;
    ensure!(lens.len() == n, BadCompressedData);
    let data = &src[consumed..];
    let mut off = 0usize;
    for &l in lens.iter() {
        ensure!(l >= 0, ValueOutOfRange);
        let l = l as usize;
        ensure!(l <= data.len() - off, UnexpectedEof, off);
        out.push(&data[off..off + l]);
        off += l;
    }
    Ok(())
}

/// DELTA_BYTE_ARRAY (common leading prefix + suffix).
/// The prefix lengths and suffix lengths are each laid out as DELTA_BINARY_PACKED,
/// followed by the concatenated suffix bodies.
pub fn decode_delta_byte_array(src: &[u8], n: usize, out: &mut BytesData) -> Result<()> {
    let mut prefixes: Vec<i32> = Vec::new();
    let c1 = decode_delta_binary_packed_i32(src, n, &mut prefixes)?;
    let mut suffixes: Vec<i32> = Vec::new();
    let c2 = decode_delta_binary_packed_i32(&src[c1..], n, &mut suffixes)?;
    ensure!(prefixes.len() == n && suffixes.len() == n, BadCompressedData);

    let data = &src[c1 + c2..];
    let mut off = 0usize;
    // The previous value lives inside `out`, so we just remember its range and copy
    // from our own buffer -- no need for a per-value temporary buffer.
    let mut prev_start = out.data.len();
    let mut prev_len = 0usize;
    for (&p, &s) in prefixes.iter().zip(suffixes.iter()) {
        ensure!(p >= 0 && s >= 0, ValueOutOfRange);
        let p = p as usize;
        let s = s as usize;
        // A prefix longer than the previous value is corruption; without this check we'd copy out of range.
        ensure!(p <= prev_len, BadCompressedData);
        ensure!(s <= data.len() - off, UnexpectedEof, off);
        let start = out.data.len();
        out.data.extend_from_within(prev_start..prev_start + p);
        out.data.extend_from_slice(&data[off..off + s]);
        out.offsets.push(out.data.len() as u32);
        prev_start = start;
        prev_len = p + s;
        off += s;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Test encoders -----------------------------------------------------

    /// Packs LSB-first. Written independently of the decoder; small-width cases are
    /// also cross-checked against the handwritten byte sequences below.
    fn pack(values: &[u32], width: u8) -> Vec<u8> {
        let mut out = vec![0u8; values.len() * width as usize / 8 + 1];
        let mut bit = 0usize;
        for &v in values {
            for i in 0..width as usize {
                if (v >> i) & 1 != 0 {
                    out[(bit + i) / 8] |= 1 << ((bit + i) % 8);
                }
            }
            bit += width as usize;
        }
        out.truncate(bit.div_ceil(8));
        out
    }

    /// A bit-packed run (header + values). `values.len()` must be a multiple of 8.
    fn bp_run(values: &[u32], width: u8) -> Vec<u8> {
        let mut out = vec![(((values.len() / 8) << 1) | 1) as u8];
        out.extend_from_slice(&pack(values, width));
        out
    }

    /// An RLE run (header + value).
    fn rle_run(value: u32, count: usize, width: u8) -> Vec<u8> {
        let mut out = Vec::new();
        let mut hdr = (count << 1) as u64;
        loop {
            let b = (hdr & 0x7f) as u8;
            hdr >>= 7;
            if hdr == 0 {
                out.push(b);
                break;
            }
            out.push(b | 0x80);
        }
        for i in 0..(width as usize).div_ceil(8) {
            out.push((value >> (i * 8)) as u8);
        }
        out
    }

    fn read_all(data: &[u8], width: u8, n: usize) -> Result<Vec<u32>> {
        let mut d = RleDecoder::new(data, width);
        let mut out = Vec::new();
        d.read_u32(n, &mut out)?;
        Ok(out)
    }

    // --- bit_width -----------------------------------------------------------

    #[test]
    fn bit_width_of_max_value() {
        assert_eq!(bit_width(0), 0);
        assert_eq!(bit_width(1), 1);
        assert_eq!(bit_width(2), 2);
        assert_eq!(bit_width(7), 3);
        assert_eq!(bit_width(8), 4);
        assert_eq!(bit_width(255), 8);
        assert_eq!(bit_width(u32::MAX), 32);
    }

    // --- bit-packed run ------------------------------------------------------

    #[test]
    fn bitpacked_width1_handwritten() {
        // Header 0x03 = (1 group << 1) | 1. The value packs 8 into 1 byte.
        // 0b1001_0110 -> from LSB: 0,1,1,0,1,0,0,1
        let data = [0x03u8, 0b1001_0110];
        assert_eq!(read_all(&data, 1, 8).unwrap(), vec![0, 1, 1, 0, 1, 0, 0, 1]);
    }

    #[test]
    fn bitpacked_width3_handwritten() {
        // Packing values 0,1,2,3,4,5,6,7 at 3 bits LSB-first yields 3 bytes.
        //   bit stream: 000 001 010 011 100 101 110 111 (from the low end)
        //   byte0 = 10001000b = 0x88, byte1 = 11000110b = 0xc6, byte2 = 11111010b = 0xfa
        let data = [0x03u8, 0x88, 0xc6, 0xfa];
        assert_eq!(read_all(&data, 3, 8).unwrap(), vec![0, 1, 2, 3, 4, 5, 6, 7]);
        // Also confirm the above byte sequence matches the test encoder.
        assert_eq!(pack(&[0, 1, 2, 3, 4, 5, 6, 7], 3), vec![0x88, 0xc6, 0xfa]);
    }

    #[test]
    fn bitpacked_width8_handwritten() {
        let data = [0x03u8, 1, 2, 3, 250, 251, 252, 253, 254];
        assert_eq!(read_all(&data, 8, 8).unwrap(), vec![1, 2, 3, 250, 251, 252, 253, 254]);
    }

    #[test]
    fn bitpacked_width0_yields_zeros_without_value_bytes() {
        // With width 0 there are no value bytes. A single header byte is enough to read 16 values.
        let data = [0x05u8];
        assert_eq!(read_all(&data, 0, 16).unwrap(), vec![0u32; 16]);
    }

    #[test]
    fn bitpacked_various_widths() {
        for &w in &[0u8, 1, 3, 7, 8, 12, 16, 32] {
            let max = if w == 0 {
                0
            } else if w == 32 {
                u32::MAX
            } else {
                (1u32 << w) - 1
            };
            let vals: Vec<u32> = (0..24u32)
                .map(|i| if w == 0 { 0 } else { i.wrapping_mul(2_654_435_761) & max })
                .collect();
            let data = bp_run(&vals, w);
            assert_eq!(read_all(&data, w, vals.len()).unwrap(), vals, "width {w}");
            // Also exercise the edge values (0 and all-bits-1).
            let edge: Vec<u32> = (0..8).map(|i| if i % 2 == 0 { 0 } else { max }).collect();
            let data = bp_run(&edge, w);
            assert_eq!(read_all(&data, w, 8).unwrap(), edge, "width {w} edge");
        }
    }

    // --- RLE run -------------------------------------------------------------

    #[test]
    fn rle_run_handwritten() {
        // Header 0x0a = 5 << 1 (RLE, 5 values), value is 1 byte.
        let data = [0x0au8, 0x2a];
        assert_eq!(read_all(&data, 8, 5).unwrap(), vec![42u32; 5]);
    }

    #[test]
    fn rle_run_width0_consumes_no_value_bytes() {
        // A width-0 RLE has no value bytes. A single header byte covers 1000 values.
        let data = rle_run(0, 1000, 0);
        assert_eq!(data.len(), 2, "header only (varint, 2 bytes)");
        assert_eq!(read_all(&data, 0, 1000).unwrap(), vec![0u32; 1000]);
    }

    #[test]
    fn rle_run_wide_value() {
        let data = rle_run(0x0123_4567, 3, 32);
        assert_eq!(read_all(&data, 32, 3).unwrap(), vec![0x0123_4567; 3]);
    }

    #[test]
    fn alternating_runs() {
        let mut data = Vec::new();
        data.extend_from_slice(&rle_run(5, 100, 4));
        data.extend_from_slice(&bp_run(&[1, 2, 3, 4, 5, 6, 7, 8], 4));
        data.extend_from_slice(&rle_run(9, 3, 4));
        data.extend_from_slice(&bp_run(&[0, 15, 0, 15, 0, 15, 0, 15], 4));

        let mut expect = vec![5u32; 100];
        expect.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        expect.extend_from_slice(&[9, 9, 9]);
        expect.extend_from_slice(&[0, 15, 0, 15, 0, 15, 0, 15]);
        assert_eq!(read_all(&data, 4, expect.len()).unwrap(), expect);
    }

    // --- Streaming -------------------------------------------------------------

    #[test]
    fn successive_reads_split_a_single_rle_run() {
        let data = rle_run(3, 10, 4);
        let mut d = RleDecoder::new(&data, 4);
        let mut out = Vec::new();
        d.read_u32(4, &mut out).unwrap();
        assert_eq!(out, vec![3u32; 4]);
        d.read_u32(6, &mut out).unwrap();
        assert_eq!(out, vec![3u32; 10]);
        // Requesting more after it's exhausted should hit EOF.
        assert!(d.read_u32(1, &mut out).is_err());
    }

    #[test]
    fn successive_reads_split_a_bitpacked_group() {
        // Read one group (8 values) split into 3 + 2 + 3. Verify the leftover carries over correctly.
        let vals: Vec<u32> = vec![7, 6, 5, 4, 3, 2, 1, 0];
        let data = bp_run(&vals, 3);
        let mut d = RleDecoder::new(&data, 3);
        let mut out = Vec::new();
        d.read_u32(3, &mut out).unwrap();
        d.read_u32(2, &mut out).unwrap();
        d.read_u32(3, &mut out).unwrap();
        assert_eq!(out, vals);
    }

    #[test]
    fn successive_reads_cross_run_boundary() {
        let mut data = bp_run(&[1, 2, 3, 4, 5, 6, 7, 8], 4);
        data.extend_from_slice(&rle_run(12, 5, 4));
        let mut d = RleDecoder::new(&data, 4);
        let mut out = Vec::new();
        // The first request spans across a run boundary.
        d.read_u32(10, &mut out).unwrap();
        assert_eq!(out, vec![1, 2, 3, 4, 5, 6, 7, 8, 12, 12]);
        d.read_u32(3, &mut out).unwrap();
        assert_eq!(&out[10..], &[12, 12, 12]);
    }

    // --- definition level ----------------------------------------------------

    #[test]
    fn read_levels_into_builds_validity() {
        // max_level = 1. Bit-packed with a mix of 0/1, followed by a non-NULL RLE run.
        let mut data = bp_run(&[1, 0, 1, 1, 0, 0, 1, 0], 1);
        data.extend_from_slice(&rle_run(1, 100, 1));
        data.extend_from_slice(&rle_run(0, 4, 1));

        let mut bm = Bitmap::new();
        let mut d = RleDecoder::new(&data, 1);
        let present = d.read_levels_into(112, 1, &mut bm).unwrap();
        assert_eq!(bm.len(), 112);
        assert_eq!(present, 4 + 100);
        let head: Vec<bool> = (0..8).map(|i| bm.get(i)).collect();
        assert_eq!(head, vec![true, false, true, true, false, false, true, false]);
        for i in 8..108 {
            assert!(bm.get(i), "bit {i}");
        }
        for i in 108..112 {
            assert!(!bm.get(i), "bit {i}");
        }
        assert_eq!(bm.count_ones(), present);
    }

    #[test]
    fn read_levels_into_all_present_for_required_column() {
        // A REQUIRED column has max_level = 0, width 0. Every row is valid.
        let data = rle_run(0, 300, 0);
        let mut bm = Bitmap::new();
        let mut d = RleDecoder::new(&data, 0);
        let present = d.read_levels_into(300, 0, &mut bm).unwrap();
        assert_eq!(present, 300);
        assert!(bm.all_set());
        assert_eq!(bm.len(), 300);
    }

    #[test]
    fn read_levels_into_is_streaming() {
        let data = rle_run(1, 200, 1);
        let mut bm = Bitmap::new();
        let mut d = RleDecoder::new(&data, 1);
        assert_eq!(d.read_levels_into(70, 1, &mut bm).unwrap(), 70);
        assert_eq!(d.read_levels_into(130, 1, &mut bm).unwrap(), 130);
        assert_eq!(bm.len(), 200);
        assert!(bm.all_set());
    }

    // --- RLE error cases ---------------------------------------------------------

    #[test]
    fn rle_rejects_bit_width_over_32() {
        let data = [0x03u8, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut out = Vec::new();
        assert!(RleDecoder::new(&data, 33).read_u32(1, &mut out).is_err());
        let mut bm = Bitmap::new();
        assert!(RleDecoder::new(&data, 33).read_levels_into(1, 0, &mut bm).is_err());
    }

    #[test]
    fn rle_rejects_truncated_input() {
        // Empty stream.
        assert!(read_all(&[], 4, 1).is_err());
        // Header only, no value bytes for the RLE.
        assert!(read_all(&[0x0a], 8, 5).is_err());
        // Not enough bit-packed value bytes (declaration for 4 groups, only 1 byte).
        assert!(read_all(&[0x09, 0xff], 8, 8).is_err());
        // The varint never terminates.
        assert!(read_all(&[0x80, 0x80, 0x80], 8, 1).is_err());
    }

    #[test]
    fn rle_rejects_uleb128_overflow_in_the_tenth_byte() {
        // The final payload byte of a u64 ULEB128 value may only be 0 or 1.
        // This otherwise decodes to a one-value RLE run after the overflowing
        // high bits are truncated by a u64 shift.
        let data = [0x82, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x02, 0x00];
        assert!(read_all(&data, 0, 1).is_err());
    }

    #[test]
    fn rle_rejects_values_that_do_not_fit_the_declared_width() {
        // Width 1 has only the values 0 and 1; the high bits in an RLE value
        // are not padding and must not be silently accepted.
        assert!(read_all(&[0x02, 0xff], 1, 1).is_err());

        // Bit-packed values need the same level-range validation. A wider
        // stream can encode a value that exceeds the column's max_level.
        let data = bp_run(&[2, 0, 0, 0, 0, 0, 0, 0], 2);
        let mut levels = Bitmap::new();
        assert!(RleDecoder::new(&data, 2).read_levels_into(8, 1, &mut levels).is_err());
    }

    #[test]
    fn rle_rejects_absurd_declared_counts() {
        // RLE run length near the u64 upper bound.
        let data = [0xfeu8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01, 0x00];
        assert!(read_all(&data, 8, 1).is_err());
        // A huge bit-packed group count.
        let data = [0xffu8, 0xff, 0xff, 0xff, 0xff, 0x0f, 0x00];
        assert!(read_all(&data, 8, 1).is_err());
    }

    #[test]
    fn rle_zero_length_runs_terminate() {
        // A stream consisting only of runs that produce no values. Must hit EOF without looping forever.
        let data = [0x00u8, 0x00, 0x00, 0x01, 0x01, 0x00];
        assert!(read_all(&data, 0, 1).is_err());
        let mut bm = Bitmap::new();
        assert!(RleDecoder::new(&data, 0).read_levels_into(1, 0, &mut bm).is_err());
    }

    // --- DELTA_BINARY_PACKED fixtures ----------------------------------
    //
    // The byte sequences below are exactly what arrow-rs's parquet crate (55.2)'s
    // DeltaBitPackEncoder / DeltaLengthByteArrayEncoder / DeltaByteArrayEncoder
    // encoders produced, pasted here verbatim.
    // D32_SMALL (10 bytes)
    const D32_SMALL: &[u8] = &[0x80, 0x01, 0x04, 0x0a, 0x02, 0x02, 0x00, 0x00, 0x00, 0x00];
    // D32_NEG (36 bytes)
    const D32_NEG: &[u8] = &[
        0x80, 0x01, 0x04, 0x08, 0xc8, 0x01, 0x8b, 0x01, 0x06, 0x00, 0x00, 0x00, 0xbc, 0x8c, 0x7a,
        0x94, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    // D32_CONST (10 bytes)
    const D32_CONST: &[u8] = &[0x80, 0x01, 0x04, 0x28, 0x0e, 0x00, 0x00, 0x00, 0x00, 0x00];
    // D32_ONE (5 bytes)
    const D32_ONE: &[u8] = &[0x80, 0x01, 0x04, 0x01, 0x54];
    // D32_WRAP (14 bytes)
    const D32_WRAP: &[u8] =
        &[0x80, 0x01, 0x04, 0x06, 0xfa, 0xff, 0xff, 0xff, 0x0f, 0x02, 0x00, 0x00, 0x00, 0x00];
    // D32_BIG (497 bytes)
    const D32_BIG: &[u8] = &[
        0x80, 0x01, 0x04, 0xac, 0x02, 0x85, 0x11, 0x6b, 0x0b, 0x0c, 0x0c, 0x0d, 0x00, 0x28, 0x81,
        0x12, 0xde, 0x40, 0x89, 0x5c, 0x78, 0x63, 0x20, 0x28, 0x69, 0x8a, 0x5c, 0x2e, 0xc3, 0x9b,
        0xf0, 0x18, 0x68, 0x45, 0x50, 0xaa, 0x93, 0xa6, 0x7e, 0x45, 0xae, 0x84, 0xb9, 0x6c, 0x6a,
        0x78, 0xeb, 0x9c, 0xf0, 0xce, 0xc7, 0xc0, 0x18, 0x5a, 0x71, 0x8f, 0xa0, 0x54, 0x4c, 0xea,
        0xf4, 0x50, 0x34, 0x95, 0x55, 0x7e, 0x35, 0x5a, 0xc8, 0xd5, 0x5e, 0x12, 0x76, 0x63, 0x5c,
        0x16, 0x68, 0xa6, 0xb6, 0x6c, 0xf0, 0x56, 0x71, 0x3a, 0xf7, 0x75, 0x84, 0x97, 0x7a, 0xce,
        0x37, 0x7f, 0x18, 0xd8, 0x83, 0x62, 0x78, 0x88, 0xac, 0x18, 0x8d, 0xf6, 0xb8, 0x91, 0x40,
        0x59, 0x96, 0x8a, 0xf9, 0x9a, 0xd4, 0x99, 0x9f, 0x1e, 0x3a, 0xa4, 0x68, 0xda, 0xa8, 0xb2,
        0x7a, 0xad, 0xfc, 0x1a, 0xb2, 0x46, 0xbb, 0xb6, 0x90, 0x5b, 0xbb, 0xda, 0xfb, 0xbf, 0x24,
        0x9c, 0xc4, 0x6e, 0x3c, 0xc9, 0xb8, 0xdc, 0xcd, 0x02, 0x7d, 0xd2, 0x4c, 0x1d, 0xd7, 0x96,
        0xbd, 0xdb, 0xe0, 0xad, 0xc0, 0xa9, 0xb8, 0x27, 0x47, 0xe7, 0x32, 0x9d, 0xaf, 0x1b, 0x77,
        0x08, 0xaf, 0xe5, 0x49, 0xbd, 0xbb, 0xc7, 0xf9, 0x82, 0x9f, 0xf9, 0x5b, 0x80, 0x30, 0xb0,
        0x0a, 0xea, 0xc1, 0x4f, 0x48, 0x0c, 0xd3, 0xa1, 0x43, 0x9c, 0x89, 0x58, 0xb1, 0x2f, 0x8a,
        0xc6, 0xe3, 0xc8, 0x1e, 0x23, 0xa4, 0x8d, 0xdc, 0x92, 0x94, 0x49, 0x0b, 0x0c, 0x0c, 0x0d,
        0x00, 0x28, 0x81, 0x12, 0xde, 0x40, 0x89, 0x5c, 0x78, 0x63, 0x20, 0x28, 0x69, 0x8a, 0x5c,
        0x2e, 0xc3, 0x9b, 0xf0, 0x18, 0x68, 0x45, 0x50, 0xaa, 0x93, 0xa6, 0x7e, 0x45, 0xae, 0x84,
        0xb9, 0x6c, 0x6a, 0x78, 0xeb, 0x9c, 0xf0, 0xce, 0xc7, 0xc0, 0x18, 0x5a, 0x71, 0x8f, 0xa0,
        0x54, 0x4c, 0xea, 0xf4, 0x50, 0x34, 0x95, 0x55, 0x7e, 0x35, 0x5a, 0xc8, 0xd5, 0x5e, 0x12,
        0x76, 0x63, 0x5c, 0x16, 0x68, 0xa6, 0xb6, 0x6c, 0xf0, 0x56, 0x71, 0x3a, 0xf7, 0x75, 0x84,
        0x97, 0x7a, 0xce, 0x37, 0x7f, 0x18, 0xd8, 0x83, 0x62, 0x78, 0x88, 0xac, 0x18, 0x8d, 0xf6,
        0xb8, 0x91, 0x40, 0x59, 0x96, 0x8a, 0xf9, 0x9a, 0xd4, 0x99, 0x9f, 0x1e, 0x3a, 0xa4, 0x68,
        0xda, 0xa8, 0xb2, 0x7a, 0xad, 0xfc, 0x1a, 0xb2, 0x46, 0xbb, 0xb6, 0x90, 0x5b, 0xbb, 0xda,
        0xfb, 0xbf, 0x24, 0x9c, 0xc4, 0x6e, 0x3c, 0xc9, 0xb8, 0xdc, 0xcd, 0x02, 0x7d, 0xd2, 0x4c,
        0x1d, 0xd7, 0x96, 0xbd, 0xdb, 0xe0, 0xad, 0xc0, 0xa9, 0xb8, 0x27, 0x47, 0xe7, 0x32, 0x9d,
        0xaf, 0x1b, 0x77, 0x08, 0xaf, 0xe5, 0x49, 0xbd, 0xbb, 0xc7, 0xf9, 0x82, 0x9f, 0xf9, 0x5b,
        0x80, 0x30, 0xb0, 0x0a, 0xea, 0xc1, 0x4f, 0x48, 0x0c, 0xd3, 0xa1, 0x43, 0x9c, 0x89, 0x58,
        0xb1, 0x2f, 0x8a, 0xc6, 0xe3, 0xc8, 0x1e, 0x23, 0xa4, 0x8d, 0xdc, 0x92, 0x94, 0x93, 0x01,
        0x0b, 0x0b, 0x00, 0x00, 0x00, 0x28, 0x81, 0x12, 0xde, 0x40, 0x89, 0x5c, 0x78, 0x63, 0x20,
        0x28, 0x69, 0x8a, 0x5c, 0x2e, 0xc3, 0x9b, 0xf0, 0x18, 0x68, 0x45, 0x50, 0xaa, 0x93, 0xa6,
        0x7e, 0x45, 0xae, 0x84, 0xb9, 0x6c, 0x6a, 0x78, 0xeb, 0x9c, 0xf0, 0xce, 0xc7, 0xc0, 0x18,
        0x5a, 0x71, 0x8f, 0xa0, 0x2c, 0xa6, 0x3a, 0x1f, 0x4a, 0xd3, 0xac, 0xfa, 0x75, 0xb4, 0xc8,
        0x6d, 0xaf, 0x84, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00,
    ];
    // D64_SMALL (351 bytes)
    const D64_SMALL: &[u8] = &[
        0x80, 0x02, 0x04, 0x08, 0x05, 0xff, 0xff, 0xd0, 0x94, 0xb5, 0x74, 0x2a, 0x00, 0x00, 0x00,
        0x03, 0x20, 0x4a, 0xa9, 0xd1, 0x15, 0x80, 0x28, 0xa5, 0x46, 0x07, 0x00, 0xa2, 0x94, 0x1a,
        0x1d, 0x00, 0x88, 0x52, 0x6a, 0x74, 0xfb, 0x2f, 0xef, 0x7d, 0xba, 0x02, 0x00, 0x00, 0x00,
        0x00, 0x70, 0x00, 0xf3, 0xde, 0xa7, 0x2b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    // D64_WRAP (19 bytes)
    const D64_WRAP: &[u8] = &[
        0x80, 0x02, 0x04, 0x04, 0xfc, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01, 0x02,
        0x00, 0x00, 0x00, 0x00,
    ];
    // D64_BIG (216 bytes)
    const D64_BIG: &[u8] = &[
        0x80, 0x02, 0x04, 0xac, 0x02, 0x0b, 0x15, 0x05, 0x05, 0x05, 0x05, 0x81, 0x24, 0x28, 0xda,
        0x90, 0x0c, 0x19, 0x83, 0x98, 0x1c, 0x4d, 0xc0, 0x44, 0x02, 0x08, 0x24, 0x41, 0xd1, 0x86,
        0x64, 0xc8, 0x18, 0xc4, 0xe4, 0x68, 0x02, 0x26, 0x12, 0x40, 0x20, 0x09, 0x8a, 0x36, 0x24,
        0x43, 0xc6, 0x20, 0x26, 0x47, 0x13, 0x30, 0x91, 0x00, 0x02, 0x49, 0x50, 0xb4, 0x21, 0x19,
        0x32, 0x06, 0x31, 0x39, 0x9a, 0x80, 0x89, 0x04, 0x10, 0x48, 0x82, 0xa2, 0x0d, 0xc9, 0x90,
        0x31, 0x88, 0xc9, 0xd1, 0x04, 0x4c, 0x24, 0x80, 0x40, 0x12, 0x14, 0x6d, 0x48, 0x86, 0x8c,
        0x41, 0x4c, 0x8e, 0x26, 0x60, 0x22, 0x01, 0x04, 0x92, 0xa0, 0x68, 0x43, 0x32, 0x64, 0x0c,
        0x62, 0x72, 0x34, 0x01, 0x13, 0x09, 0x20, 0x90, 0x04, 0x45, 0x1b, 0x92, 0x21, 0x63, 0x10,
        0x93, 0xa3, 0x09, 0x98, 0x48, 0x00, 0x81, 0x24, 0x28, 0xda, 0x90, 0x0c, 0x19, 0x83, 0x98,
        0x1c, 0x4d, 0xc0, 0x44, 0x02, 0x08, 0x24, 0x41, 0xd1, 0x86, 0x64, 0xc8, 0x18, 0xc4, 0xe4,
        0x68, 0x02, 0x26, 0x12, 0x40, 0x20, 0x09, 0x8a, 0x36, 0x24, 0x43, 0xc6, 0x20, 0x26, 0x47,
        0x13, 0x30, 0x91, 0x00, 0x02, 0x49, 0x15, 0x05, 0x00, 0x00, 0x00, 0x50, 0xb4, 0x21, 0x19,
        0x32, 0x06, 0x31, 0x39, 0x9a, 0x80, 0x89, 0x04, 0x10, 0x48, 0x82, 0xa2, 0x0d, 0xc9, 0x90,
        0x31, 0x88, 0xc9, 0xd1, 0x04, 0x4c, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    // DLBA (62 bytes)
    const DLBA: &[u8] = &[
        0x80, 0x01, 0x04, 0x06, 0x00, 0x0d, 0x05, 0x00, 0x00, 0x00, 0x68, 0x25, 0xa0, 0x01, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x61, 0x68, 0x65, 0x6c, 0x6c, 0x6f, 0x77, 0x6f, 0x72, 0x6c, 0x64, 0x21, 0x21, 0x74, 0x68,
        0x65, 0x20, 0x71, 0x75, 0x69, 0x63, 0x6b, 0x20, 0x62, 0x72, 0x6f, 0x77, 0x6e, 0x20, 0x66,
        0x6f, 0x78,
    ];
    // DBA (63 bytes)
    const DBA: &[u8] = &[
        0x80, 0x01, 0x04, 0x07, 0x00, 0x07, 0x04, 0x00, 0x00, 0x00, 0x94, 0x06, 0x71, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x01, 0x04, 0x07,
        0x00, 0x05, 0x04, 0x00, 0x00, 0x00, 0x08, 0x22, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x61, 0x68, 0x69, 0x72, 0x75, 0x64, 0x62, 0x21,
        0x7a, 0x7a, 0x7a,
    ];

    fn dec32(src: &[u8], n: usize) -> Vec<i32> {
        let mut out = Vec::new();
        let consumed = decode_delta_binary_packed_i32(src, n, &mut out).unwrap();
        assert_eq!(consumed, src.len(), "consumed bytes");
        out
    }

    fn dec64(src: &[u8], n: usize) -> Vec<i64> {
        let mut out = Vec::new();
        let consumed = decode_delta_binary_packed_i64(src, n, &mut out).unwrap();
        assert_eq!(consumed, src.len(), "consumed bytes");
        out
    }

    #[test]
    fn delta_i32_roundtrip() {
        assert_eq!(dec32(D32_SMALL, 10), (1..=10).collect::<Vec<i32>>());
        assert_eq!(dec32(D32_ONE, 1), vec![42]);
        assert_eq!(dec32(D32_NEG, 8), vec![100, 90, 70, 40, 0, -50, -110, -180]);
        assert_eq!(dec32(D32_CONST, 40), vec![7i32; 40]);
    }

    #[test]
    fn delta_i32_constant_uses_zero_bit_width() {
        // For a constant sequence, min_delta absorbs everything and the bit width
        // becomes 0. Only the header + block header are present; the miniblock data
        // is 0 bytes.
        assert!(D32_CONST.len() < 20, "len = {}", D32_CONST.len());
        assert_eq!(dec32(D32_CONST, 40), vec![7i32; 40]);
    }

    #[test]
    fn delta_i32_large_sequence() {
        let mut expect = Vec::new();
        let mut x: i32 = -1000;
        for i in 0..300i32 {
            x = x.wrapping_add(i * 37 - 91);
            expect.push(x);
        }
        assert_eq!(dec32(D32_BIG, 300), expect);
    }

    #[test]
    fn delta_i32_wraps_on_overflow() {
        assert_eq!(
            dec32(D32_WRAP, 6),
            vec![i32::MAX - 2, i32::MAX - 1, i32::MAX, i32::MIN, i32::MIN + 1, i32::MIN + 2]
        );
    }

    #[test]
    fn delta_i64_roundtrip() {
        assert_eq!(
            dec64(D64_SMALL, 8),
            vec![-3, 0, 5, 5, 5, 1_000_000_000_000, -1_000_000_000_000, 7]
        );
        // Spans multiple blocks, with a length such that the final block is padded.
        let mut expect = Vec::new();
        let mut y: i64 = 5;
        for i in 0..300i64 {
            y = y.wrapping_add((i * i) % 23).wrapping_sub(11);
            expect.push(y);
        }
        assert_eq!(dec64(D64_BIG, 300), expect);
    }

    #[test]
    fn delta_i64_wraps_on_overflow() {
        assert_eq!(dec64(D64_WRAP, 4), vec![i64::MAX - 1, i64::MAX, i64::MIN, i64::MIN + 1]);
    }

    #[test]
    fn delta_partial_read_still_reports_full_consumption() {
        // Even when n is smaller than the header's declared total, bytes consumed cover the whole stream.
        let mut out = Vec::new();
        let consumed = decode_delta_binary_packed_i32(D32_BIG, 5, &mut out).unwrap();
        assert_eq!(out, dec32(D32_BIG, 300)[..5]);
        assert_eq!(consumed, D32_BIG.len());
        // Same for n = 0.
        let mut out = Vec::new();
        let consumed = decode_delta_binary_packed_i32(D32_BIG, 0, &mut out).unwrap();
        assert!(out.is_empty());
        assert_eq!(consumed, D32_BIG.len());
    }

    #[test]
    fn delta_appends_to_existing_vec() {
        let mut out = vec![-1i32];
        decode_delta_binary_packed_i32(D32_SMALL, 10, &mut out).unwrap();
        assert_eq!(out[0], -1);
        assert_eq!(&out[1..], &(1..=10).collect::<Vec<i32>>()[..]);
    }

    // --- DELTA_BINARY_PACKED error cases ----------------------------------

    #[test]
    fn delta_rejects_truncated_stream() {
        for cut in 0..D32_BIG.len() {
            let r = decode_delta_binary_packed_i32(&D32_BIG[..cut], 300, &mut Vec::new());
            assert!(r.is_err(), "cut = {cut}");
        }
        // Same when it's cut off right after the header.
        assert!(decode_delta_binary_packed_i64(&D64_BIG[..5], 200, &mut Vec::new()).is_err());
    }

    #[test]
    fn delta_rejects_bad_header() {
        // block_size = 0
        assert!(
            decode_delta_binary_packed_i32(&[0x00, 0x01, 0x01, 0x00], 1, &mut Vec::new()).is_err()
        );
        // miniblocks = 0
        assert!(decode_delta_binary_packed_i32(
            &[0x80, 0x01, 0x00, 0x01, 0x00],
            1,
            &mut Vec::new()
        )
        .is_err());
        // block_size is not divisible by miniblocks (128 / 7)
        assert!(decode_delta_binary_packed_i32(
            &[0x80, 0x01, 0x07, 0x01, 0x00],
            1,
            &mut Vec::new()
        )
        .is_err());
        // Values per miniblock is not a multiple of 32 (32 / 2 = 16)
        assert!(
            decode_delta_binary_packed_i32(&[0x20, 0x02, 0x01, 0x00], 1, &mut Vec::new()).is_err()
        );
        // block_size exceeds the upper bound
        assert!(decode_delta_binary_packed_i32(
            &[0x80, 0x80, 0x80, 0x80, 0x01, 0x01, 0x01, 0x00],
            1,
            &mut Vec::new()
        )
        .is_err());
        // Empty input
        assert!(decode_delta_binary_packed_i32(&[], 0, &mut Vec::new()).is_err());
    }

    #[test]
    fn delta_rejects_absurd_value_count() {
        // total is huge. Header is 128/4/huge/0.
        let src = [0x80u8, 0x01, 0x04, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f, 0x00];
        assert!(decode_delta_binary_packed_i32(&src, 1, &mut Vec::new()).is_err());
    }

    #[test]
    fn delta_rejects_n_greater_than_declared_total() {
        assert!(decode_delta_binary_packed_i32(D32_SMALL, 11, &mut Vec::new()).is_err());
    }

    #[test]
    fn delta_rejects_bit_width_over_type_width() {
        // block 128 / 4 miniblock / 33 values / first 0, min_delta 0, width 33.
        let src = [0x80u8, 0x01, 0x04, 0x21, 0x00, 0x00, 33, 0, 0, 0];
        assert!(decode_delta_binary_packed_i32(&src, 33, &mut Vec::new()).is_err());
        // For i64, 65 exceeds the upper bound.
        let src = [0x80u8, 0x01, 0x04, 0x21, 0x00, 0x00, 65, 0, 0, 0];
        assert!(decode_delta_binary_packed_i64(&src, 33, &mut Vec::new()).is_err());
    }

    // --- DELTA_LENGTH_BYTE_ARRAY / DELTA_BYTE_ARRAY --------------------------

    fn strings(b: &BytesData) -> Vec<&str> {
        (0..b.len()).map(|i| core::str::from_utf8(b.get(i)).unwrap()).collect()
    }

    #[test]
    fn delta_length_byte_array_roundtrip() {
        let mut out = BytesData::new();
        decode_delta_length_byte_array(DLBA, 6, &mut out).unwrap();
        assert_eq!(strings(&out), vec!["", "a", "hello", "world!!", "", "the quick brown fox"]);
    }

    #[test]
    fn delta_byte_array_roundtrip() {
        let mut out = BytesData::new();
        decode_delta_byte_array(DBA, 7, &mut out).unwrap();
        assert_eq!(strings(&out), vec!["", "ahiru", "ahirudb", "ahirudb!", "ahi", "zzz", "zzz"]);
    }

    #[test]
    fn delta_byte_array_appends_to_existing_buffer() {
        // Confirm the reference range for the previous value doesn't shift even with existing data.
        let mut out = BytesData::new();
        out.push(b"XXXXXXXXXXXX");
        decode_delta_byte_array(DBA, 7, &mut out).unwrap();
        assert_eq!(out.len(), 8);
        assert_eq!(out.get(0), b"XXXXXXXXXXXX");
        assert_eq!(
            strings(&out)[1..],
            vec!["", "ahiru", "ahirudb", "ahirudb!", "ahi", "zzz", "zzz"]
        );
    }

    #[test]
    fn byte_array_rejects_truncated_stream() {
        for cut in 0..DLBA.len() {
            let mut out = BytesData::new();
            assert!(decode_delta_length_byte_array(&DLBA[..cut], 6, &mut out).is_err(), "{cut}");
        }
        for cut in 0..DBA.len() {
            let mut out = BytesData::new();
            assert!(decode_delta_byte_array(&DBA[..cut], 7, &mut out).is_err(), "{cut}");
        }
    }

    #[test]
    fn byte_array_rejects_negative_length() {
        // Hand-assemble a length column: block 128 / 1 miniblock / 1 value / first = -1.
        // zigzag(-1) = 1.
        let src = [0x80u8, 0x01, 0x01, 0x01, 0x01];
        let mut out = BytesData::new();
        assert!(decode_delta_length_byte_array(&src, 1, &mut out).is_err());
    }

    #[test]
    fn byte_array_rejects_prefix_longer_than_previous() {
        // A corrupted input where prefix = [0, 5] but the previous value's length is only 1.
        // prefix column: 128/1/2/0 + block(min_delta=5, width=0)
        let mut src: Vec<u8> = vec![0x80, 0x01, 0x01, 0x02, 0x00, 0x0a, 0x00];
        // suffix column: 128/1/2/1 (first=-1, not 1) + block(min_delta=-1, width=0)
        src.extend_from_slice(&[0x80, 0x01, 0x01, 0x02, 0x02, 0x01, 0x00]);
        src.extend_from_slice(b"ab");
        let mut out = BytesData::new();
        assert!(decode_delta_byte_array(&src, 2, &mut out).is_err());
    }

    #[test]
    fn byte_array_rejects_absurd_lengths() {
        // Declares length 1000, but the body is missing.
        let src = [0x80u8, 0x01, 0x01, 0x01, 0xd0, 0x0f];
        let mut out = BytesData::new();
        assert!(decode_delta_length_byte_array(&src, 1, &mut out).is_err());
    }

    // --- Near-exhaustive robustness tests -------------------------------------

    /// xorshift64. A minimal PRNG to keep the tests deterministic.
    fn rng(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    /// Runs corrupted input through every decoder. The return value doesn't matter.
    /// What's tested is simply that it doesn't panic and does terminate.
    fn poke(src: &[u8]) {
        for &w in &[0u8, 1, 3, 8, 17, 32, 33, 64, 255] {
            let mut v = Vec::new();
            let _ = RleDecoder::new(src, w).read_u32(97, &mut v);
            let mut bm = Bitmap::new();
            let _ = RleDecoder::new(src, w).read_levels_into(97, 1, &mut bm);
        }
        for &n in &[0usize, 1, 97, 5000] {
            let _ = decode_delta_binary_packed_i32(src, n, &mut Vec::new());
            let _ = decode_delta_binary_packed_i64(src, n, &mut Vec::new());
            let _ = decode_delta_length_byte_array(src, n, &mut BytesData::new());
            let _ = decode_delta_byte_array(src, n, &mut BytesData::new());
        }
    }

    #[test]
    fn random_bytes_never_panic_or_hang() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut buf = Vec::new();
        for _ in 0..400 {
            buf.clear();
            let len = (rng(&mut state) % 64) as usize;
            for _ in 0..len {
                buf.push(rng(&mut state) as u8);
            }
            poke(&buf);
        }
    }

    #[test]
    fn mutated_fixtures_never_panic_or_hang() {
        // Corrupt a single byte of a valid stream. This produces a "plausible"
        // corruption where only the header's declared value is off, which is good at
        // catching gaps in bounds checking.
        let mut state = 0x0123_4567_89ab_cdefu64;
        for fixture in [D32_SMALL, D32_NEG, D32_BIG, D64_SMALL, D64_WRAP, DLBA, DBA] {
            for _ in 0..80 {
                let mut m = fixture.to_vec();
                let i = (rng(&mut state) as usize) % m.len();
                m[i] ^= rng(&mut state) as u8;
                poke(&m);
            }
            // Also try streams cut short, across the whole length.
            for cut in 0..fixture.len() {
                poke(&fixture[..cut]);
            }
        }
    }
}
