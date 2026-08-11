//! ビットリーダ 2 種。
//!
//! zstd は流儀を 2 つ混在させる。FSE テーブル記述だけが前から読む LSB 先行の
//! ストリームで、ハフマンとシーケンスの本体は末尾バイトから前方向へ戻りながら
//! 読む。どちらもここに閉じ込めて、上位からはビット数だけを扱わせる。

use crate::Error;

/// `src` を「バイト 0 の最下位ビットを 0 番」とするビット列と見なし、
/// ビット位置 `off` から `n` ビット (n <= 32) を読む。
///
/// 範囲外は 0 として扱う。逆向きリーダが末尾のパディングを踏むとき、
/// 仕様上そこは 0 埋めとして解釈するため（境界検査は呼び出し側で済ませる）。
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

/// 前向き LSB 先行リーダ。FSE テーブル記述専用。
pub struct Forward<'a> {
    src: &'a [u8],
    bit: usize,
}

impl<'a> Forward<'a> {
    pub fn new(src: &'a [u8]) -> Self {
        Forward { src, bit: 0 }
    }

    /// `n` ビット (n <= 32) 読む。入力が尽きたら `Err`。
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

    /// 1 ビット戻す。テーブル記述の「小さい値は 1 ビット短い」符号で使う。
    pub fn rewind1(&mut self) {
        self.bit = self.bit.saturating_sub(1);
    }

    /// バイト境界に切り上げた消費バイト数。
    pub fn bytes_used(&self) -> usize {
        self.bit.div_ceil(8)
    }
}

/// 逆向きリーダ。
///
/// 末尾バイトの最上位の 1 ビットがストリーム終端マーカで、それより上位は
/// パディング。`off` は「まだ読んでいないビット数」で、負になったら
/// ストリームを踏み越えた合図（FSE のシンボル数はこれでしか分からない）。
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
        // マーカが立っていないストリームは壊れている。
        if last == 0 {
            return Err(Error::BadBitstream);
        }
        let hi = 7 - last.leading_zeros() as i64;
        let off = (src.len() as i64) * 8 - (8 - hi);
        Ok(Reverse { src, off })
    }

    /// 消費せずに上位 `n` ビットを覗く。足りない分は下位を 0 で埋める。
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

    /// 残りビット数。負ならストリームを踏み越えている。
    pub fn off(&self) -> i64 {
        self.off
    }
}
