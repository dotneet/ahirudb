//! リテラル用ハフマンデコーダ。
//!
//! 表は 2^max_bits の完全展開型（1 回の peek で 1 シンボル）。max_bits は
//! 仕様上 11 までなので最大 2 KiB × 2 本で済み、木を辿る実装より小さく速い。

use crate::bits::Reverse;
use crate::fse;
use crate::prelude::*;
use crate::Error;

/// 仕様が許すコード長の上限。
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

/// ハフマン木記述を読み、表と消費バイト数を返す。
pub fn read_table(src: &[u8]) -> Result<(Huff, usize), Error> {
    let head = match src.first() {
        Some(v) => *v,
        None => return Err(Error::UnexpectedEof),
    };
    // weight は最大 255 個 + 末尾 1 個を復元するので 256 要る。
    let mut w = [0u8; 256];
    let n;
    let used;
    if head < 128 {
        // head は FSE ストリームのバイト数。
        let size = head as usize;
        let body = src.get(1..1 + size).ok_or(Error::UnexpectedEof)?;
        let (t, hdr) = fse::read_table(body, 6, 255)?;
        let stream = body.get(hdr..).ok_or(Error::UnexpectedEof)?;
        n = fse::decode_interleaved(&t, stream, &mut w[..255])?;
        used = 1 + size;
        stat!(crate::stats::HUF_W_FSE);
    } else {
        // 直接表現。4 ビット/weight、上位ニブルが先。
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

/// weight 列から復号表を組む。
///
/// 最後のシンボルの weight は書かれておらず、「総和を次の 2 のべきに満たす分」
/// から復元する。ここが合わなければ壊れた入力。
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

    // weight w のコード長は max_bits + 1 - w。weight 0 は不使用シンボル。
    let mut bits = [0u8; 256];
    let mut rank_count = [0u32; MAX_BITS as usize + 1];
    for s in 0..num {
        let b = if w[s] > 0 { max_bits + 1 - w[s] as u32 } else { 0 };
        bits[s] = b as u8;
        if b > 0 {
            rank_count[b as usize] += 1;
        }
    }

    // コードはビット長の長い順（= weight の小さい順）に 0 から詰める。
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
    /// 逆向きストリームから `n` シンボル展開して `out` に追記する。
    ///
    /// 追記量は呼び出し側が `out` の容量で保証済み（リテラル領域は
    /// ブロック上限 128 KiB で切ってある）。
    pub fn decode_stream(&self, src: &[u8], n: usize, out: &mut Vec<u8>) -> Result<(), Error> {
        let mut r = Reverse::new(src)?;
        for _ in 0..n {
            // 1 シンボルは最低 1 ビット。残りが無ければ壊れている。
            if r.off() <= 0 {
                return Err(Error::BadHuffman);
            }
            let i = (r.peek(self.max_bits) as usize) & self.mask;
            out.push(self.sym[i]);
            r.skip(self.nb[i] as u32);
        }
        // ストリームはちょうど使い切る。余り/不足はどちらも破損。
        if r.off() != 0 {
            return Err(Error::BadHuffman);
        }
        Ok(())
    }
}
