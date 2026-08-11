//! フレーム / ブロック / シーケンスの復号 (RFC 8878)。
//!
//! 入力は信用できない。宣言された長さ・オフセットは一切信用せず、全ての添字を
//! 境界検査する。破損に対しては必ず `Err` を返す（パニックしない、バッファ外を
//! 読まない、無限ループしない、上限を超えて確保しない）。

use crate::fse;
use crate::huff::Huff;
use crate::prelude::*;
use crate::{huff, xxh64, Error};

const MAGIC: u32 = 0xFD2F_B528;
const SKIP_MAGIC: u32 = 0x184D_2A50;
/// Block_Maximum_Size。仕様上ブロックはこれを超えない（window がより小さい
/// 場合は更に小さいが、こちらは全体をバッファに持つので上限だけ効かせる）。
const BLOCK_MAX: usize = 128 * 1024;

// --- 定義済みテーブル -------------------------------------------------------

// RFC 8878 の記載どおり 16 個ずつ並べる（写し間違いを目視で潰せるように）。
// -1 は「確率 1/N 未満」で、表の末尾から 1 セルずつ埋まる。
#[rustfmt::skip]
const LL_NORM: [i16; 36] = [
     4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
    -1,-1,-1,-1,
];
#[rustfmt::skip]
const ML_NORM: [i16; 53] = [
     1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,
    -1,-1,-1,-1,-1,
];
#[rustfmt::skip]
const OF_NORM: [i16; 29] = [
     1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
];

#[rustfmt::skip]
const LL_BASE: [u32; 36] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28,
    32, 40, 48, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
];
#[rustfmt::skip]
const LL_BITS: [u8; 36] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3,
    4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
];
#[rustfmt::skip]
const ML_BASE: [u32; 53] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 41, 43, 47, 51, 59,
    67, 83, 99, 131, 259, 515, 1027, 2051, 4099, 8195, 16387, 32771, 65539,
];
#[rustfmt::skip]
const ML_BITS: [u8; 53] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16,
];

// --- 小物 -------------------------------------------------------------------

fn u32le(src: &[u8], at: usize) -> Result<u32, Error> {
    let b = src.get(at..at + 4).ok_or(Error::UnexpectedEof)?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn uint_le(src: &[u8], at: usize, n: usize) -> Result<u64, Error> {
    let b = src.get(at..at + n).ok_or(Error::UnexpectedEof)?;
    let mut v = 0u64;
    for (i, x) in b.iter().enumerate() {
        v |= (*x as u64) << (i * 8);
    }
    Ok(v)
}

/// 出力末尾から `offset` 戻った位置を `len` バイト複製する。
///
/// 複製元と複製先は重なりうる（offset < len が繰り返しの符号化そのもの）ので
/// 1 バイトずつ写す。`copy_from_slice` では重なり分を再生成できない。
fn copy_match(out: &mut Vec<u8>, offset: usize, len: usize, limit: usize) -> Result<(), Error> {
    if out.len() + len > limit {
        return Err(Error::LimitExceeded);
    }
    for p in (out.len() - offset..).take(len) {
        let b = out[p];
        out.push(b);
    }
    Ok(())
}

fn append(out: &mut Vec<u8>, src: &[u8], limit: usize) -> Result<(), Error> {
    if out.len() + src.len() > limit {
        return Err(Error::LimitExceeded);
    }
    out.extend_from_slice(src);
    Ok(())
}

// --- フレーム状態 -----------------------------------------------------------

/// フレームを跨がずブロック間で持ち越す状態。
///
/// ハフマン表は Treeless リテラルブロックが、FSE 表は Repeat モードが
/// 前ブロックのものを参照する。繰り返しオフセットも同様。
struct State {
    huff: Option<Huff>,
    ll: Option<fse::Table>,
    of: Option<fse::Table>,
    ml: Option<fse::Table>,
    rep: [u32; 3],
    lits: Vec<u8>,
}

impl State {
    fn new() -> State {
        State {
            huff: None,
            ll: None,
            of: None,
            ml: None,
            // 繰り返しオフセットの初期値はフレーム毎にこの値へ戻る。
            rep: [1, 4, 8],
            lits: Vec::new(),
        }
    }
}

// --- 入口 -------------------------------------------------------------------

/// `src` 全体を（連結された複数フレームとして）展開し `out` に追記する。
pub fn decompress_into(src: &[u8], out: &mut Vec<u8>, max_len: usize) -> Result<(), Error> {
    let limit = out.len().checked_add(max_len).ok_or(Error::LimitExceeded)?;
    let mut pos = 0usize;
    let mut frames = 0usize;
    while pos < src.len() {
        pos = one_frame(src, pos, out, limit)?;
        frames += 1;
    }
    if frames == 0 {
        // フレームが 1 つも無い＝マジックすら無い。
        return Err(Error::UnexpectedEof);
    }
    Ok(())
}

/// 1 フレーム分を処理し、次のフレームの開始位置を返す。
fn one_frame(src: &[u8], mut pos: usize, out: &mut Vec<u8>, limit: usize) -> Result<usize, Error> {
    let magic = u32le(src, pos)?;
    pos += 4;

    // スキッパブルフレームは中身を見ずに読み飛ばす。
    if magic & 0xFFFF_FFF0 == SKIP_MAGIC {
        let n = u32le(src, pos)? as usize;
        pos += 4;
        pos = pos.checked_add(n).ok_or(Error::UnexpectedEof)?;
        if pos > src.len() {
            return Err(Error::UnexpectedEof);
        }
        stat!(crate::stats::SKIPPABLE);
        return Ok(pos);
    }
    if magic != MAGIC {
        return Err(Error::BadMagic);
    }

    let d = *src.get(pos).ok_or(Error::UnexpectedEof)?;
    pos += 1;
    // 予約ビットが立っていたら未知の拡張。読み進めない。
    if d & 0x08 != 0 {
        return Err(Error::BadFrameHeader);
    }
    let fcs_flag = d >> 6;
    let single = d & 0x20 != 0;
    let checksum = d & 0x04 != 0;
    let did_flag = d & 0x03;

    if !single {
        // Window_Descriptor。全体をバッファに持つ実装ではウィンドウ幅を使わず、
        // オフセットは「フレーム内で生成済みのバイト数」で縛るので読み飛ばす。
        if src.get(pos).is_none() {
            return Err(Error::UnexpectedEof);
        }
        pos += 1;
    }
    // 辞書は非対応。黙って壊れた出力を返すより明示的に落とす。
    if did_flag != 0 {
        return Err(Error::DictionaryUnsupported);
    }

    let fcs_size = match fcs_flag {
        0 => usize::from(single),
        1 => 2,
        2 => 4,
        _ => 8,
    };
    let mut fcs: Option<u64> = None;
    if fcs_size > 0 {
        let v = uint_le(src, pos, fcs_size)?;
        pos += fcs_size;
        // 2 バイト形式だけ 256 が下駄として乗る。
        fcs = Some(if fcs_size == 2 { v + 256 } else { v });
    }

    let frame_base = out.len();
    // 宣言サイズが上限を超えるなら展開する前に落とす。
    if let Some(n) = fcs {
        if n > (limit - frame_base) as u64 {
            return Err(Error::LimitExceeded);
        }
    }

    let mut st = State::new();
    loop {
        let h = uint_le(src, pos, 3)? as u32;
        pos += 3;
        let last = h & 1 != 0;
        let btype = (h >> 1) & 3;
        let bsize = (h >> 3) as usize;
        if bsize > BLOCK_MAX {
            return Err(Error::BadBlock);
        }
        match btype {
            0 => {
                let b = src.get(pos..pos + bsize).ok_or(Error::UnexpectedEof)?;
                append(out, b, limit)?;
                pos += bsize;
                stat!(crate::stats::BLOCK_RAW);
            }
            1 => {
                let b = *src.get(pos).ok_or(Error::UnexpectedEof)?;
                pos += 1;
                if out.len() + bsize > limit {
                    return Err(Error::LimitExceeded);
                }
                out.resize(out.len() + bsize, b);
                stat!(crate::stats::BLOCK_RLE);
            }
            2 => {
                if bsize == 0 {
                    return Err(Error::BadBlock);
                }
                let b = src.get(pos..pos + bsize).ok_or(Error::UnexpectedEof)?;
                st.block(b, out, frame_base, limit)?;
                pos += bsize;
                stat!(crate::stats::BLOCK_COMPRESSED);
            }
            _ => return Err(Error::BadBlock),
        }
        if last {
            break;
        }
    }

    if checksum {
        let want = u32le(src, pos)?;
        pos += 4;
        if xxh64::hash(&out[frame_base..]) as u32 != want {
            return Err(Error::ChecksumMismatch);
        }
        stat!(crate::stats::CHECKSUM);
    }
    if let Some(n) = fcs {
        if (out.len() - frame_base) as u64 != n {
            return Err(Error::BadFrameHeader);
        }
    }
    Ok(pos)
}

impl State {
    /// 圧縮ブロック 1 個。リテラル部 → シーケンス部の順。
    fn block(
        &mut self,
        src: &[u8],
        out: &mut Vec<u8>,
        frame_base: usize,
        limit: usize,
    ) -> Result<(), Error> {
        let used = self.literals(src)?;
        let rest = src.get(used..).ok_or(Error::UnexpectedEof)?;
        self.sequences(rest, out, frame_base, limit)
    }

    // --- リテラル部 ---------------------------------------------------------

    /// リテラルを `self.lits` に展開し、消費バイト数を返す。
    fn literals(&mut self, src: &[u8]) -> Result<usize, Error> {
        let b0 = *src.first().ok_or(Error::UnexpectedEof)?;
        let ltype = b0 & 3;
        let sf = (b0 >> 2) & 3;

        self.lits.clear();
        if ltype < 2 {
            // Raw / RLE。ヘッダは 1〜3 バイトで再生サイズだけを持つ。
            let (hdr, regen) = match sf {
                1 => (
                    2,
                    (b0 as usize >> 4) | ((*src.get(1).ok_or(Error::UnexpectedEof)? as usize) << 4),
                ),
                3 => (
                    3,
                    (b0 as usize >> 4)
                        | ((*src.get(1).ok_or(Error::UnexpectedEof)? as usize) << 4)
                        | ((*src.get(2).ok_or(Error::UnexpectedEof)? as usize) << 12),
                ),
                _ => (1, b0 as usize >> 3),
            };
            if regen > BLOCK_MAX {
                return Err(Error::BadLiterals);
            }
            if ltype == 0 {
                let b = src.get(hdr..hdr + regen).ok_or(Error::UnexpectedEof)?;
                self.lits.extend_from_slice(b);
                stat!(crate::stats::LIT_RAW);
                return Ok(hdr + regen);
            }
            let v = *src.get(hdr).ok_or(Error::UnexpectedEof)?;
            self.lits.resize(regen, v);
            stat!(crate::stats::LIT_RLE);
            return Ok(hdr + 1);
        }

        // Compressed / Treeless。ヘッダは 3〜5 バイトのリトルエンディアン語で、
        // 下位 4 ビットの後ろに再生サイズと圧縮サイズが並ぶ。
        let (hdr, w, streams) = match sf {
            0 => (3, 10u32, 1),
            1 => (3, 10, 4),
            2 => (4, 14, 4),
            _ => (5, 18, 4),
        };
        let v = uint_le(src, 0, hdr)?;
        let mask = (1u64 << w) - 1;
        let regen = ((v >> 4) & mask) as usize;
        let comp = ((v >> (4 + w)) & mask) as usize;
        if regen > BLOCK_MAX {
            return Err(Error::BadLiterals);
        }
        let body = src.get(hdr..hdr + comp).ok_or(Error::UnexpectedEof)?;

        let tree_used = if ltype == 2 {
            let (h, n) = huff::read_table(body)?;
            self.huff = Some(h);
            stat!(crate::stats::LIT_COMPRESSED);
            n
        } else {
            // Treeless は前ブロックの表を再利用する。無ければ破損。
            if self.huff.is_none() {
                return Err(Error::BadLiterals);
            }
            stat!(crate::stats::LIT_TREELESS);
            0
        };
        let streams_buf = body.get(tree_used..).ok_or(Error::UnexpectedEof)?;

        // フィールドを分割借用する（表を読みながらリテラルへ書くため）。
        let State { huff, lits, .. } = self;
        let h = huff.as_ref().ok_or(Error::BadLiterals)?;
        lits.reserve(regen);

        if streams == 1 {
            h.decode_stream(streams_buf, regen, lits)?;
            stat!(crate::stats::HUF_1STREAM);
        } else {
            // 4 ストリーム形式。先頭 6 バイトが最初の 3 本のサイズ表。
            let jt = streams_buf.get(..6).ok_or(Error::UnexpectedEof)?;
            let s1 = u16::from_le_bytes([jt[0], jt[1]]) as usize;
            let s2 = u16::from_le_bytes([jt[2], jt[3]]) as usize;
            let s3 = u16::from_le_bytes([jt[4], jt[5]]) as usize;
            let body4 = &streams_buf[6..];
            let s4 = body4.len().checked_sub(s1 + s2 + s3).ok_or(Error::BadLiterals)?;
            // 各ストリームは 1/4 ずつ担当し、最後だけ端数を持つ。
            let seg = regen.div_ceil(4);
            let last = regen.checked_sub(seg * 3).ok_or(Error::BadLiterals)?;
            let mut at = 0usize;
            for (sz, n) in [(s1, seg), (s2, seg), (s3, seg), (s4, last)] {
                let part = body4.get(at..at + sz).ok_or(Error::UnexpectedEof)?;
                h.decode_stream(part, n, lits)?;
                at += sz;
            }
            stat!(crate::stats::HUF_4STREAM);
        }
        if lits.len() != regen {
            return Err(Error::BadLiterals);
        }
        Ok(hdr + comp)
    }

    // --- シーケンス部 -------------------------------------------------------

    fn sequences(
        &mut self,
        src: &[u8],
        out: &mut Vec<u8>,
        frame_base: usize,
        limit: usize,
    ) -> Result<(), Error> {
        let b0 = *src.first().ok_or(Error::UnexpectedEof)?;
        let (mut p, nb) = if b0 == 0 {
            (1usize, 0usize)
        } else if b0 < 128 {
            (1, b0 as usize)
        } else if b0 < 255 {
            let b1 = *src.get(1).ok_or(Error::UnexpectedEof)?;
            (2, (((b0 as usize) - 128) << 8) + b1 as usize)
        } else {
            let b1 = *src.get(1).ok_or(Error::UnexpectedEof)?;
            let b2 = *src.get(2).ok_or(Error::UnexpectedEof)?;
            (3, b1 as usize + ((b2 as usize) << 8) + 0x7F00)
        };

        if nb == 0 {
            // シーケンス無し。リテラルがそのまま出力になる。
            return append(out, &self.lits, limit);
        }

        let modes = *src.get(p).ok_or(Error::UnexpectedEof)?;
        p += 1;
        if modes & 3 != 0 {
            return Err(Error::BadSequence);
        }
        // 表の並びは LL, OF, ML の順（実行順とは違うので注意）。
        p += load_table(&mut self.ll, modes >> 6, src.get(p..).unwrap_or(&[]), Kind::Ll)?;
        p += load_table(&mut self.of, (modes >> 4) & 3, src.get(p..).unwrap_or(&[]), Kind::Of)?;
        p += load_table(&mut self.ml, (modes >> 2) & 3, src.get(p..).unwrap_or(&[]), Kind::Ml)?;

        let bits = src.get(p..).ok_or(Error::UnexpectedEof)?;
        let State { ll, of, ml, rep, lits, .. } = self;
        let ll = ll.as_ref().ok_or(Error::BadSequence)?;
        let of = of.as_ref().ok_or(Error::BadSequence)?;
        let ml = ml.as_ref().ok_or(Error::BadSequence)?;

        let mut r = crate::bits::Reverse::new(bits)?;
        let mut ll_st = ll.init(&mut r);
        let mut of_st = of.init(&mut r);
        let mut ml_st = ml.init(&mut r);
        if r.off() < 0 {
            return Err(Error::BadSequence);
        }

        let mut lit_pos = 0usize;
        for i in 0..nb {
            let of_code = of.peek(of_st) as u32;
            let ll_code = ll.peek(ll_st) as usize;
            let ml_code = ml.peek(ml_st) as usize;
            if of_code > 31 || ll_code >= LL_BASE.len() || ml_code >= ML_BASE.len() {
                return Err(Error::BadSequence);
            }
            // 追加ビットは Offset, Match_Length, Literals_Length の順に並ぶ。
            let off_v = (1u64 << of_code) + r.read(of_code);
            let ml_v = ML_BASE[ml_code] as usize + r.read(ML_BITS[ml_code] as u32) as usize;
            let ll_v = LL_BASE[ll_code] as usize + r.read(LL_BITS[ll_code] as u32) as usize;
            // 状態更新は Literals_Length, Match_Length, Offset の順。
            // 最後のシーケンスの後には更新用のビットが無い。
            if i + 1 < nb {
                ll.update(&mut ll_st, &mut r);
                ml.update(&mut ml_st, &mut r);
                of.update(&mut of_st, &mut r);
            }
            if r.off() < 0 {
                return Err(Error::BadSequence);
            }

            let src_lit = lits.get(lit_pos..lit_pos + ll_v).ok_or(Error::BadSequence)?;
            append(out, src_lit, limit)?;
            lit_pos += ll_v;

            // オフセット値 1..3 は繰り返しオフセットの参照。literals_length が
            // 0 のときだけ 1 つずれる（直前と同じオフセットを再指定できないため）。
            let offset = if off_v > 3 {
                let o = (off_v - 3) as u32;
                rep[2] = rep[1];
                rep[1] = rep[0];
                rep[0] = o;
                o
            } else {
                stat!(crate::stats::REPEAT_OFFSET);
                let rc = off_v as usize - 1 + usize::from(ll_v == 0);
                if rc == 0 {
                    rep[0]
                } else {
                    let o = if rc == 3 { rep[0].wrapping_sub(1) } else { rep[rc] };
                    if rc >= 2 {
                        rep[2] = rep[1];
                    }
                    rep[1] = rep[0];
                    rep[0] = o;
                    o
                }
            } as usize;

            // 直前フレームや辞書には遡れない。生成済みバイト数で縛る。
            if offset == 0 || offset > out.len() - frame_base {
                return Err(Error::BadOffset);
            }
            copy_match(out, offset, ml_v, limit)?;
        }
        // ビットストリームはちょうど使い切る。
        if r.off() != 0 {
            return Err(Error::BadSequence);
        }

        let rest = lits.get(lit_pos..).ok_or(Error::BadSequence)?;
        append(out, rest, limit)
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Ll,
    Of,
    Ml,
}

/// シーケンス用 FSE 表を 1 本読み込む。返り値は消費バイト数。
fn load_table(
    slot: &mut Option<fse::Table>,
    mode: u8,
    src: &[u8],
    kind: Kind,
) -> Result<usize, Error> {
    let (norm, acc, max_sym): (&[i16], u32, usize) = match kind {
        Kind::Ll => (&LL_NORM, 6, 35),
        Kind::Of => (&OF_NORM, 5, 31),
        Kind::Ml => (&ML_NORM, 6, 52),
    };
    let max_acc = match kind {
        Kind::Of => 8,
        _ => 9,
    };
    match mode {
        0 => {
            *slot = Some(fse::predefined(norm, acc)?);
            stat!(crate::stats::SEQ_PREDEFINED);
            Ok(0)
        }
        1 => {
            let s = *src.first().ok_or(Error::UnexpectedEof)?;
            if s as usize > max_sym {
                return Err(Error::BadSequence);
            }
            *slot = Some(fse::Table::rle(s));
            stat!(crate::stats::SEQ_RLE);
            Ok(1)
        }
        2 => {
            let (t, used) = fse::read_table(src, max_acc, max_sym)?;
            *slot = Some(t);
            stat!(crate::stats::SEQ_FSE);
            Ok(used)
        }
        _ => {
            // Repeat: 直前のブロックの表をそのまま使う。無ければ破損。
            if slot.is_none() {
                return Err(Error::BadSequence);
            }
            stat!(crate::stats::SEQ_REPEAT);
            Ok(0)
        }
    }
}

#[cfg(test)]
mod predefined_tests {
    use super::*;

    /// 定義済み分布は写し間違えても合計が 2^Accuracy_Log に収まってしまうことが
    /// あり、`build` の整合検査だけでは検出できない。既知の状態→シンボル対応で
    /// 固定する（RFC 8878 の表から手計算した値）。
    #[test]
    fn predefined_tables_match_spec() {
        let ll = fse::predefined(&LL_NORM, 6).unwrap();
        let ml = fse::predefined(&ML_NORM, 6).unwrap();
        let of = fse::predefined(&OF_NORM, 5).unwrap();

        assert_eq!(ll.peek(0), 0);
        assert_eq!(ll.peek(24), 2);
        assert_eq!(ll.peek(63), 32);
        assert_eq!(of.peek(0), 0);
        assert_eq!(of.peek(31), 24);
        assert_eq!(ml.peek(0), 0);
        assert_eq!(ml.peek(57), 52);
        assert_eq!(ml.peek(63), 46);

        // -1 のシンボルは表末尾に自然順で逆向きに並ぶ。
        for (i, s) in [46u8, 47, 48, 49, 50, 51, 52].iter().enumerate() {
            assert_eq!(ml.peek(63 - i as u16), *s);
        }
    }

    /// baseline / 追加ビット表の長さと端の値。
    #[test]
    fn baseline_tables() {
        assert_eq!(LL_BASE.len(), LL_BITS.len());
        assert_eq!(ML_BASE.len(), ML_BITS.len());
        assert_eq!((LL_BASE[35], LL_BITS[35]), (65536, 16));
        assert_eq!((ML_BASE[52], ML_BITS[52]), (65539, 16));
        assert_eq!((ML_BASE[0], ML_BITS[0]), (3, 0));
    }
}
