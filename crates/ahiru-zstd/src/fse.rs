//! FSE (Finite State Entropy) デコード表。
//!
//! 表は「状態 -> (シンボル, 次状態を作るためのビット数とベース)」の 3 本の配列。
//! 状態数は 2 のべきなので添字はマスクで丸められ、境界検査を省いてもバッファ外に
//! 出ない（それでも `sym`/`nb`/`base` の長さは常に同じに保つ）。

use crate::bits::{Forward, Reverse};
use crate::prelude::*;
use crate::Error;

/// 正規化カウントに現れうるシンボルの上限。仕様上 255 まで。
const MAX_SYMBOLS: usize = 256;

pub struct Table {
    accuracy: u32,
    /// 状態数 - 1。添字はこれで丸める。
    mask: u16,
    sym: Vec<u8>,
    nb: Vec<u8>,
    base: Vec<u16>,
}

impl Table {
    /// RLE モード用。状態は常に 0、追加ビットも 0 で同じシンボルを返す。
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

/// 最上位 1 ビットの位置。`v == 0` は呼び出し側で排除済みだが、
/// デバッグビルドで桁溢れパニックを起こさないよう飽和させておく。
#[inline]
fn highest_bit(v: u32) -> u32 {
    31u32.saturating_sub(v.leading_zeros())
}

/// 正規化カウントから復号表を組む。`norm[i] == -1` は「確率 1/N 未満」を表し、
/// 表の末尾から 1 セルずつ割り当てられる。
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

    // セル割り当ては線形ではなく step 幅で散らす。step は size と互いに素
    // (accuracy >= 5 なので必ず奇数) なので巡回で全セルを踏む。
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
            // high == 0 のときはここに来ない（正の確率が無いため）。それでも
            // 壊れた入力で無限ループしないよう回数を切る。
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
    // 全セルを使い切れば一周して 0 に戻る。
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

/// 定義済み分布から表を組む。
pub fn predefined(norm: &[i16], accuracy: u32) -> Result<Table, Error> {
    build(norm, norm.len(), accuracy)
}

/// FSE テーブル記述を読む。返り値は表と消費バイト数。
///
/// 記述は前向き LSB 先行のビット列で、バイト境界に切り上げて終わる。
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
        // 残り確率量に応じてビット幅が縮む可変長符号。
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

        // 値 0 は「確率 1/N 未満」を意味する -1。
        let proba = val as i32 - 1;
        remaining -= proba.abs();
        if remaining < 0 {
            return Err(Error::BadFse);
        }
        norm[sym] = proba as i16;
        sym += 1;

        if proba == 0 {
            // 確率 0 のシンボルの後ろには 2 ビットのスキップ数が続く。
            // 3 なら更に 2 ビット読む（最大 3 個ずつ連鎖）。
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

/// ハフマン weight 用の 2 状態インターリーブ FSE 展開。
///
/// シンボル数は前もって分からず、ビットストリームを踏み越えた時点で終わる
/// （仕様がそう定めている）。`out` の長さで上限を切る。
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
