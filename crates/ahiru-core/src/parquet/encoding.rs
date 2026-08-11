//! Parquet のページエンコーディングのデコーダ。
//!
//! 扱うのは RLE/bit-packing hybrid（definition level と辞書インデックス）と
//! DELTA_* 系。PLAIN は固定長のコピーで済むので `reader.rs` 側で処理する。
//!
//! 入力はネットワーク由来で信用できない。全ての読み取りは境界検査を行い、
//! 破損に対しては必ず `Err` を返す（パニックも無限ループもしない）。
//! 特に「値を 1 個も生まない run」が連続しても、ヘッダは必ず 1 バイト以上
//! 消費するのでループは必ず終端する。

use crate::prelude::*;
use crate::vector::{Bitmap, BytesData};

/// `max_value` を表現するのに必要なビット数。
pub fn bit_width(max_value: u32) -> u8 {
    (u32::BITS - max_value.leading_zeros()) as u8
}

/// 1 ストリームが宣言できる値数の上限。宣言値をそのまま `usize` にすると
/// wasm32 (usize = 32 ビット) で切り捨てが起きるため、先に弾く。
const MAX_VALUES: u64 = 1 << 31;

/// DELTA_BINARY_PACKED のブロックサイズ上限。ミニブロックの一時計算が
/// 32 ビットで溢れない範囲に抑える。
const MAX_BLOCK: u64 = 1 << 20;

// --- ビット列の読み取り -----------------------------------------------------

/// LSB-first のビット列リーダ。Parquet の bit-packing は全てこの順序で、
/// 最初の値が先頭バイトの下位ビットに入る。
struct BitReader<'a> {
    buf: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    #[inline]
    fn new(buf: &'a [u8]) -> Self {
        BitReader { buf, bit_pos: 0 }
    }

    /// 次の `width` ビット（0..=64）を読む。`width == 0` は 1 ビットも
    /// 消費せず 0 を返す（全値ゼロの run / ミニブロック）。
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

/// ULEB128 可変長整数。`pos` を進める。10 バイトを超えたら破損とみなす。
fn uleb128(src: &[u8], pos: &mut usize) -> Result<u64> {
    let mut result = 0u64;
    let mut shift = 0u32;
    loop {
        ensure!(shift < 64, BadCompressedData, *pos);
        ensure!(*pos < src.len(), UnexpectedEof, *pos);
        let b = src[*pos];
        *pos += 1;
        result |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Ok(result);
        }
        shift += 7;
    }
}

/// zigzag デコード。DELTA_* のヘッダは全てこの形式。
#[inline]
fn zigzag(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

// --- RLE / bit-packing hybrid -----------------------------------------------

/// RLE / bit-packing hybrid のデコーダ。
/// definition level と辞書インデックスの両方で使う。
///
/// run が要求より多くの値を持つことがあるので、ストリーミングできるよう
/// 余りを内部に残す。bit-packed は 8 個単位（= `bit_width` バイト単位）で
/// バイト境界に揃うため、持ち越しは 8 個のバッファ 1 つで足りる。
pub struct RleDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    bit_width: u8,
    /// 展開待ちの RLE run。
    rle_value: u32,
    rle_left: usize,
    /// bit-packed run に残っている未展開の値数（常に 8 の倍数）。
    packed_left: usize,
    /// 直近に展開した 8 個と、その未消費範囲。
    group: [u32; 8],
    group_pos: usize,
    group_len: usize,
}

impl<'a> RleDecoder<'a> {
    /// `data` は長さプレフィックスを除いた RLE ストリーム本体。
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

    /// 次の run ヘッダを読む。値を 0 個しか生まない run も仕様上は合法だが、
    /// ヘッダで必ず 1 バイト以上進むので呼び出し側のループは停止する。
    fn next_run(&mut self) -> Result<()> {
        let header = uleb128(self.data, &mut self.pos)?;
        let avail = (self.data.len() - self.pos) as u64;
        if header & 1 == 1 {
            // bit-packed run。値数はグループ数 × 8 で、
            // 1 グループがちょうど `bit_width` バイトを占める。
            let groups = header >> 1;
            ensure!(groups <= MAX_VALUES >> 3, LimitExceeded, self.pos);
            // run 全体の長さは確定しているので、ここでまとめて検査しておく。
            ensure!(groups * self.bit_width as u64 <= avail, UnexpectedEof, self.pos);
            self.packed_left = (groups as usize) * 8;
        } else {
            // RLE run。値は ceil(bit_width / 8) バイトのリトルエンディアン。
            let count = header >> 1;
            ensure!(count <= MAX_VALUES, LimitExceeded, self.pos);
            let nbytes = (self.bit_width as usize).div_ceil(8);
            ensure!(nbytes as u64 <= avail, UnexpectedEof, self.pos);
            let mut v = 0u32;
            for i in 0..nbytes {
                v |= (self.data[self.pos + i] as u32) << (i * 8);
            }
            self.pos += nbytes;
            self.rle_value = v;
            self.rle_left = count as usize;
        }
        Ok(())
    }

    /// bit-packed run から 8 個を展開して `group` に置く。
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

    /// 値を 1 個以上取り出せる状態にする。持ち越し → RLE run →
    /// 新しい run の順に用意する。
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

    /// 次の `n` 個の値を `out` に追記する。
    pub fn read_u32(&mut self, n: usize, out: &mut Vec<u32>) -> Result<()> {
        // 値は u32 に入れるので 32 ビットを超える幅は扱えない。
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

    /// definition level を読み、`level == max_level`（= 値が存在する）を
    /// validity ビットマップとして `out` に追記する。
    /// 返り値は存在した値の個数。
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
                    let v = self.group[self.group_pos + i] == max_level;
                    out.push(v);
                    present += v as usize;
                }
                self.group_pos += take;
                need -= take;
            } else {
                // NULL が無い列は巨大な RLE run 1 本になる。ここが主要な経路
                // なので、ビット単位ではなくワード単位でまとめて埋める。
                let take = core::cmp::min(need, self.rle_left);
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

/// i32 と i64 でデコーダ本体を 1 本に保つための出力側の抽象。
/// 累算は常に i64 で行う。2 の補数では下位 32 ビットの折り返しが i32 での
/// `wrapping_add` と一致するので、i32 列は最後に切り捨てるだけでよい。
trait DeltaSink {
    /// ミニブロックのビット幅の上限。物理型の幅を超える宣言は破損。
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

/// DELTA_BINARY_PACKED の本体。返り値は消費したバイト数。
///
/// 最終ブロックは値数を切り上げてパディングされることがある。ヘッダの
/// 総数を超えた分は値としては捨てるが、ミニブロックのバイトは消費する
/// （消費バイト数が狂うと後続のバイト列の開始位置がずれる）。
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
    // ミニブロックの値数は 32 の倍数。これでビット幅によらずミニブロックが
    // バイト境界で終わることが保証される。
    ensure!(per_mini % 32 == 0, BadCompressedData, pos);
    ensure!(total <= MAX_VALUES, LimitExceeded, pos);
    ensure!(n as u64 <= total, UnexpectedEof, pos);

    let miniblocks = miniblocks as usize;
    let per_mini = per_mini as usize;
    let mut remaining = total as usize; // 未処理の値数（先頭値を含む）
    let mut emitted = 0usize;
    let mut last = first;

    if remaining > 0 {
        // 先頭値だけはヘッダに直接入っていて、デルタを足さない。
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
            // 最終ブロックで不要になったミニブロックはビット幅こそ書かれて
            // いるが、データは 1 バイトも続かない。
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
                    // 仕様上デルタの加算は折り返す。
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

/// DELTA_BINARY_PACKED (INT32)。返り値は消費したバイト数。
/// 後続のデータが続く DELTA_LENGTH_BYTE_ARRAY などで必要になる。
pub fn decode_delta_binary_packed_i32(src: &[u8], n: usize, out: &mut Vec<i32>) -> Result<usize> {
    decode_delta(src, n, out)
}

/// DELTA_BINARY_PACKED (INT64)。返り値は消費したバイト数。
pub fn decode_delta_binary_packed_i64(src: &[u8], n: usize, out: &mut Vec<i64>) -> Result<usize> {
    decode_delta(src, n, out)
}

// --- DELTA_*_BYTE_ARRAY -----------------------------------------------------

/// DELTA_LENGTH_BYTE_ARRAY。
/// 長さの列（DELTA_BINARY_PACKED）の直後に本体が連結されている。
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

/// DELTA_BYTE_ARRAY（前方共通接頭辞 + サフィックス）。
/// 接頭辞長・サフィックス長がそれぞれ DELTA_BINARY_PACKED で並び、
/// その後にサフィックス本体が連結されている。
pub fn decode_delta_byte_array(src: &[u8], n: usize, out: &mut BytesData) -> Result<()> {
    let mut prefixes: Vec<i32> = Vec::new();
    let c1 = decode_delta_binary_packed_i32(src, n, &mut prefixes)?;
    let mut suffixes: Vec<i32> = Vec::new();
    let c2 = decode_delta_binary_packed_i32(&src[c1..], n, &mut suffixes)?;
    ensure!(prefixes.len() == n && suffixes.len() == n, BadCompressedData);

    let data = &src[c1 + c2..];
    let mut off = 0usize;
    // 直前の値は `out` の中にあるので、範囲だけ覚えて自身のバッファから
    // コピーする。値ごとの一時バッファを持たずに済む。
    let mut prev_start = out.data.len();
    let mut prev_len = 0usize;
    for (&p, &s) in prefixes.iter().zip(suffixes.iter()) {
        ensure!(p >= 0 && s >= 0, ValueOutOfRange);
        let p = p as usize;
        let s = s as usize;
        // 接頭辞が直前の値より長いのは破損。ここを通さないと範囲外コピーになる。
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

    // --- テスト用のエンコーダ -----------------------------------------------

    /// LSB-first でパックする。デコーダとは独立に書いてあり、
    /// 幅の小さいケースは下の手書きバイト列とも突き合わせている。
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

    /// bit-packed run（ヘッダ + 値）。`values.len()` は 8 の倍数であること。
    fn bp_run(values: &[u32], width: u8) -> Vec<u8> {
        let mut out = vec![(((values.len() / 8) << 1) | 1) as u8];
        out.extend_from_slice(&pack(values, width));
        out
    }

    /// RLE run（ヘッダ + 値）。
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
        // ヘッダ 0x03 = (1 グループ << 1) | 1。値は 1 バイトに 8 個。
        // 0b1001_0110 -> LSB から 0,1,1,0,1,0,0,1
        let data = [0x03u8, 0b1001_0110];
        assert_eq!(read_all(&data, 1, 8).unwrap(), vec![0, 1, 1, 0, 1, 0, 0, 1]);
    }

    #[test]
    fn bitpacked_width3_handwritten() {
        // 値 0,1,2,3,4,5,6,7 を 3 ビット LSB-first で詰めると 3 バイト。
        //   ビット列: 000 001 010 011 100 101 110 111 (下位から)
        //   byte0 = 10001000b = 0x88, byte1 = 11000110b = 0xc6, byte2 = 11111010b = 0xfa
        let data = [0x03u8, 0x88, 0xc6, 0xfa];
        assert_eq!(read_all(&data, 3, 8).unwrap(), vec![0, 1, 2, 3, 4, 5, 6, 7]);
        // 上のバイト列がテスト用エンコーダと一致することも確認する。
        assert_eq!(pack(&[0, 1, 2, 3, 4, 5, 6, 7], 3), vec![0x88, 0xc6, 0xfa]);
    }

    #[test]
    fn bitpacked_width8_handwritten() {
        let data = [0x03u8, 1, 2, 3, 250, 251, 252, 253, 254];
        assert_eq!(read_all(&data, 8, 8).unwrap(), vec![1, 2, 3, 250, 251, 252, 253, 254]);
    }

    #[test]
    fn bitpacked_width0_yields_zeros_without_value_bytes() {
        // 幅 0 では値バイトが存在しない。ヘッダ 1 バイトだけで 16 個読める。
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
            // 端の値（0 と全ビット 1）も通す。
            let edge: Vec<u32> = (0..8).map(|i| if i % 2 == 0 { 0 } else { max }).collect();
            let data = bp_run(&edge, w);
            assert_eq!(read_all(&data, w, 8).unwrap(), edge, "width {w} edge");
        }
    }

    // --- RLE run -------------------------------------------------------------

    #[test]
    fn rle_run_handwritten() {
        // ヘッダ 0x0a = 5 << 1（RLE, 5 個）、値 1 バイト。
        let data = [0x0au8, 0x2a];
        assert_eq!(read_all(&data, 8, 5).unwrap(), vec![42u32; 5]);
    }

    #[test]
    fn rle_run_width0_consumes_no_value_bytes() {
        // 幅 0 の RLE は値バイトを持たない。ヘッダ 1 バイトで 1000 個。
        let data = rle_run(0, 1000, 0);
        assert_eq!(data.len(), 2, "ヘッダのみ (varint 2 バイト)");
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

    // --- ストリーミング -------------------------------------------------------

    #[test]
    fn successive_reads_split_a_single_rle_run() {
        let data = rle_run(3, 10, 4);
        let mut d = RleDecoder::new(&data, 4);
        let mut out = Vec::new();
        d.read_u32(4, &mut out).unwrap();
        assert_eq!(out, vec![3u32; 4]);
        d.read_u32(6, &mut out).unwrap();
        assert_eq!(out, vec![3u32; 10]);
        // 使い切った後にさらに要求すれば EOF。
        assert!(d.read_u32(1, &mut out).is_err());
    }

    #[test]
    fn successive_reads_split_a_bitpacked_group() {
        // 1 グループ (8 個) を 3 + 2 + 3 に分けて読む。持ち越しが正しく残るか。
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
        // 1 回目の要求が run をまたぐ。
        d.read_u32(10, &mut out).unwrap();
        assert_eq!(out, vec![1, 2, 3, 4, 5, 6, 7, 8, 12, 12]);
        d.read_u32(3, &mut out).unwrap();
        assert_eq!(&out[10..], &[12, 12, 12]);
    }

    // --- definition level ----------------------------------------------------

    #[test]
    fn read_levels_into_builds_validity() {
        // max_level = 1。bit-packed で 0/1 が混ざり、その後 NULL でない RLE run。
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
        // REQUIRED 列は max_level = 0、幅 0。全行 valid になる。
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

    // --- RLE の異常系 ---------------------------------------------------------

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
        // 空ストリーム。
        assert!(read_all(&[], 4, 1).is_err());
        // ヘッダのみで値バイトが無い RLE。
        assert!(read_all(&[0x0a], 8, 5).is_err());
        // bit-packed の値バイトが足りない（4 グループ分の宣言に 1 バイト）。
        assert!(read_all(&[0x09, 0xff], 8, 8).is_err());
        // varint が終わらない。
        assert!(read_all(&[0x80, 0x80, 0x80], 8, 1).is_err());
    }

    #[test]
    fn rle_rejects_absurd_declared_counts() {
        // RLE run 長 = u64 の上限付近。
        let data = [0xfeu8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01, 0x00];
        assert!(read_all(&data, 8, 1).is_err());
        // bit-packed グループ数が巨大。
        let data = [0xffu8, 0xff, 0xff, 0xff, 0xff, 0x0f, 0x00];
        assert!(read_all(&data, 8, 1).is_err());
    }

    #[test]
    fn rle_zero_length_runs_terminate() {
        // 値を生まない run が並ぶだけのストリーム。無限ループせず EOF になる。
        let data = [0x00u8, 0x00, 0x00, 0x01, 0x01, 0x00];
        assert!(read_all(&data, 0, 1).is_err());
        let mut bm = Bitmap::new();
        assert!(RleDecoder::new(&data, 0).read_levels_into(1, 0, &mut bm).is_err());
    }

    // --- DELTA_BINARY_PACKED のフィクスチャ ----------------------------------
    //
    // 以下のバイト列は arrow-rs の parquet クレート (55.2) の
    // DeltaBitPackEncoder / DeltaLengthByteArrayEncoder / DeltaByteArrayEncoder
    // が出力したものをそのまま貼っている。
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
        // 定数列は min_delta が全てを吸収してビット幅 0 になる。
        // ヘッダ + ブロックヘッダだけで、ミニブロックのデータは 0 バイト。
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
        // 複数ブロックにまたがり、最終ブロックがパディングされる長さ。
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
        // n がヘッダの総数より小さくても、消費バイト数はストリーム全体。
        let mut out = Vec::new();
        let consumed = decode_delta_binary_packed_i32(D32_BIG, 5, &mut out).unwrap();
        assert_eq!(out, dec32(D32_BIG, 300)[..5]);
        assert_eq!(consumed, D32_BIG.len());
        // n = 0 でも同じ。
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

    // --- DELTA_BINARY_PACKED の異常系 ----------------------------------------

    #[test]
    fn delta_rejects_truncated_stream() {
        for cut in 0..D32_BIG.len() {
            let r = decode_delta_binary_packed_i32(&D32_BIG[..cut], 300, &mut Vec::new());
            assert!(r.is_err(), "cut = {cut}");
        }
        // ヘッダ直後で切れている場合も同じ。
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
        // block_size が miniblocks で割り切れない (128 / 7)
        assert!(decode_delta_binary_packed_i32(
            &[0x80, 0x01, 0x07, 0x01, 0x00],
            1,
            &mut Vec::new()
        )
        .is_err());
        // ミニブロックあたりの値数が 32 の倍数でない (32 / 2 = 16)
        assert!(
            decode_delta_binary_packed_i32(&[0x20, 0x02, 0x01, 0x00], 1, &mut Vec::new()).is_err()
        );
        // block_size が上限超え
        assert!(decode_delta_binary_packed_i32(
            &[0x80, 0x80, 0x80, 0x80, 0x01, 0x01, 0x01, 0x00],
            1,
            &mut Vec::new()
        )
        .is_err());
        // 空入力
        assert!(decode_delta_binary_packed_i32(&[], 0, &mut Vec::new()).is_err());
    }

    #[test]
    fn delta_rejects_absurd_value_count() {
        // total = 巨大。ヘッダは 128/4/huge/0。
        let src = [0x80u8, 0x01, 0x04, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f, 0x00];
        assert!(decode_delta_binary_packed_i32(&src, 1, &mut Vec::new()).is_err());
    }

    #[test]
    fn delta_rejects_n_greater_than_declared_total() {
        assert!(decode_delta_binary_packed_i32(D32_SMALL, 11, &mut Vec::new()).is_err());
    }

    #[test]
    fn delta_rejects_bit_width_over_type_width() {
        // block 128 / 4 miniblock / 33 値 / first 0、min_delta 0、幅 33。
        let src = [0x80u8, 0x01, 0x04, 0x21, 0x00, 0x00, 33, 0, 0, 0];
        assert!(decode_delta_binary_packed_i32(&src, 33, &mut Vec::new()).is_err());
        // i64 なら 65 が上限超え。
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
        // 既存データがあっても直前値の参照範囲がずれないこと。
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
        // 長さ列を手で組む: block 128 / 1 miniblock / 1 値 / first = -1。
        // zigzag(-1) = 1。
        let src = [0x80u8, 0x01, 0x01, 0x01, 0x01];
        let mut out = BytesData::new();
        assert!(decode_delta_length_byte_array(&src, 1, &mut out).is_err());
    }

    #[test]
    fn byte_array_rejects_prefix_longer_than_previous() {
        // prefix = [0, 5] だが直前の値の長さは 1 しかない、という壊れた入力。
        // prefix 列: 128/1/2/0 + block(min_delta=5, width=0)
        let mut src: Vec<u8> = vec![0x80, 0x01, 0x01, 0x02, 0x00, 0x0a, 0x00];
        // suffix 列: 128/1/2/1 (first=-1 ではなく 1) + block(min_delta=-1, width=0)
        src.extend_from_slice(&[0x80, 0x01, 0x01, 0x02, 0x02, 0x01, 0x00]);
        src.extend_from_slice(b"ab");
        let mut out = BytesData::new();
        assert!(decode_delta_byte_array(&src, 2, &mut out).is_err());
    }

    #[test]
    fn byte_array_rejects_absurd_lengths() {
        // 長さ 1000 を宣言しているのに本体が無い。
        let src = [0x80u8, 0x01, 0x01, 0x01, 0xd0, 0x0f];
        let mut out = BytesData::new();
        assert!(decode_delta_length_byte_array(&src, 1, &mut out).is_err());
    }

    // --- 総当たりに近い頑健性テスト -------------------------------------------

    /// xorshift64。テストを決定的に保つための最小の PRNG。
    fn rng(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    /// 壊れた入力を全デコーダに通す。返り値は問わない。
    /// パニックせず、かつ停止すること自体が検査対象。
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
        // 正しいストリームの 1 バイトを壊す。ヘッダの宣言値だけが狂った
        // 「もっともらしい」破損を作れるので、境界検査の抜けを拾いやすい。
        let mut state = 0x0123_4567_89ab_cdefu64;
        for fixture in [D32_SMALL, D32_NEG, D32_BIG, D64_SMALL, D64_WRAP, DLBA, DBA] {
            for _ in 0..80 {
                let mut m = fixture.to_vec();
                let i = (rng(&mut state) as usize) % m.len();
                m[i] ^= rng(&mut state) as u8;
                poke(&m);
            }
            // 途中で切れたものも全長にわたって試す。
            for cut in 0..fixture.len() {
                poke(&fixture[..cut]);
            }
        }
    }
}
