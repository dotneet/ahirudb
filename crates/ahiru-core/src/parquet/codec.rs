//! 圧縮コーデック。
//!
//! wasm コアが内蔵するのは SNAPPY・LZ4_RAW・ZSTD（`zstd` フィーチャ、既定で
//! 有効）。ZSTD デコーダは `ahiru-zstd` を素の Rust ライブラリとしてリンク
//! しているだけで、展開結果自体はここに含めていない（13 KB 程度で 1 MiB
//! 予算への影響が小さいため、別 wasm モジュールに分ける手間に見合わない
//! と判断した）。GZIP だけはホストの `DecompressionStream` に委譲する
//! （ブラウザ/Node に既にあるものをわざわざ内蔵する理由が無いため）。
//! `zstd` フィーチャを外せば ZSTD もホスト委譲（`NeedCodec`）に戻る
//! （DESIGN.md §6）。
//!
//! 入力はネットワーク由来で信用できない。宣言された長さ・オフセットは一切
//! 信用せず、全ての添字を境界検査する。破損に対しては必ず `Err` を返す
//! （パニックしない・バッファ外を読まない）。
//!
//! エラーコードの使い分け:
//! - `UnexpectedEof`  … 入力がシーケンスの途中で尽きた
//! - `BadCompressedData` … 構造の破損（不正なオフセット、長さ不一致など）
//! - `LimitExceeded`  … 展開結果が呼び出し側の宣言サイズを超える

use crate::parquet::Compression;
use crate::prelude::*;

/// 内蔵コーデックで展開する。非内蔵コーデックは `UnsupportedCodec` を返し、
/// 呼び出し側がホストへの委譲に切り替える。
///
/// `out_len` はページヘッダが宣言する展開後サイズ。デコーダはこれを超えて
/// 書き込んではならない（信用できない入力に対する上限）。
pub fn decompress(codec: Compression, src: &[u8], out_len: usize) -> Result<Vec<u8>> {
    match codec {
        Compression::Uncompressed => Ok(src.to_vec()),
        Compression::Snappy => {
            let mut out = Vec::with_capacity(out_len);
            snappy_decompress(src, out_len, &mut out)?;
            Ok(out)
        }
        Compression::Lz4Raw => {
            let mut out = Vec::with_capacity(out_len);
            lz4_raw_decompress(src, out_len, &mut out)?;
            Ok(out)
        }
        #[cfg(feature = "zstd")]
        Compression::Zstd => ahiru_zstd::decompress(src, out_len).map_err(map_zstd_err),
        _ => err!(UnsupportedCodec),
    }
}

/// `ahiru-zstd` のエラーコードをこのクレートのものへ落とす。位置情報は
/// 持たない（`ahiru-zstd` はバイト位置を追跡していない）ので `Error::new`
/// を使う。展開結果が宣言サイズを超える／入力が途中で尽きる、以外は
/// すべて「構造が壊れている」の一括りにする（このファイル冒頭の
/// エラーコードの使い分け方針と同じ）。
#[cfg(feature = "zstd")]
fn map_zstd_err(e: ahiru_zstd::Error) -> Error {
    use ahiru_zstd::Error as Z;
    let code = match e {
        Z::UnexpectedEof => Code::UnexpectedEof,
        Z::LimitExceeded => Code::LimitExceeded,
        _ => Code::BadCompressedData,
    };
    Error::new(code)
}

/// `out` の末尾から `offset` バイト戻った位置を `len` バイト複製する。
///
/// 複製元と複製先は重なりうる（連続した繰り返しはこの重なりで符号化される）
/// ので、1 バイトずつ写す。`copy_from_slice` では重なり分を再生成できない。
#[inline]
fn copy_within(out: &mut Vec<u8>, offset: usize, len: usize) {
    // 呼び出し側で 1 <= offset <= 書き込み済みバイト数 を検査済み。
    let mut p = out.len() - offset;
    // `out` に push しながら同じ `out` を読む（重なりの再生成）。
    // イテレータでは表現できないので手動カウンタのままにする。
    #[allow(clippy::explicit_counter_loop)]
    for _ in 0..len {
        let b = out[p];
        out.push(b);
        p += 1;
    }
}

// --- Snappy -----------------------------------------------------------------

/// Snappy raw format（フレーミング無し）を展開して `out` に追記する。
pub fn snappy_decompress(src: &[u8], max_len: usize, out: &mut Vec<u8>) -> Result<()> {
    // `out` には既存データが入っていることがある。オフセットの基準は
    // 「今回の展開で書き始めた位置」であって Vec の先頭ではない。
    let base = out.len();
    let mut ip = 0usize;

    // 前置きの展開後長（LEB128）。32 bit を超える値は仕様上ありえない。
    let mut declared: u64 = 0;
    let mut shift = 0u32;
    loop {
        ensure!(ip < src.len(), UnexpectedEof, ip);
        ensure!(shift < 32, BadCompressedData, ip);
        let b = src[ip];
        ip += 1;
        declared |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    ensure!(declared <= max_len as u64, BadCompressedData, ip);
    let declared = declared as usize;

    while ip < src.len() {
        let tag = src[ip];
        ip += 1;
        if tag & 0x03 == 0 {
            // リテラル。n < 60 なら長さは n+1、そうでなければ n-59 バイトの
            // リトルエンディアン整数が「長さ-1」を表す。
            let n = (tag >> 2) as usize;
            let len = if n < 60 {
                (n + 1) as u64
            } else {
                let w = n - 59;
                ensure!(w <= src.len() - ip, UnexpectedEof, ip);
                let mut v: u64 = 0;
                for i in 0..w {
                    v |= (src[ip + i] as u64) << (8 * i);
                }
                ip += w;
                v + 1
            };
            // 入力に収まるかを先に見るので、この後 usize へ落として安全。
            ensure!(len <= (src.len() - ip) as u64, UnexpectedEof, ip);
            let len = len as usize;
            ensure!(len <= declared - (out.len() - base), LimitExceeded, ip);
            out.extend_from_slice(&src[ip..ip + len]);
            ip += len;
        } else {
            let (len, offset) = match tag & 0x03 {
                1 => {
                    ensure!(ip < src.len(), UnexpectedEof, ip);
                    let lo = src[ip] as usize;
                    ip += 1;
                    (4 + ((tag >> 2) & 0x07) as usize, (((tag >> 5) & 0x07) as usize) << 8 | lo)
                }
                2 => {
                    ensure!(src.len() - ip >= 2, UnexpectedEof, ip);
                    let o = u16::from_le_bytes([src[ip], src[ip + 1]]) as usize;
                    ip += 2;
                    ((tag >> 2) as usize + 1, o)
                }
                _ => {
                    ensure!(src.len() - ip >= 4, UnexpectedEof, ip);
                    let o = u32::from_le_bytes([src[ip], src[ip + 1], src[ip + 2], src[ip + 3]]);
                    ip += 4;
                    ((tag >> 2) as usize + 1, o as usize)
                }
            };
            let written = out.len() - base;
            // オフセットが基準位置より前を指すなら、今回の入力とは無関係な
            // データを覗くことになる。破損として拒否する。
            ensure!(offset > 0 && offset <= written, BadCompressedData, ip);
            ensure!(len <= declared - written, LimitExceeded, ip);
            copy_within(out, offset, len);
        }
    }

    ensure!(out.len() - base == declared, BadCompressedData, ip);
    Ok(())
}

// --- LZ4 block --------------------------------------------------------------

/// LZ4 の長さ拡張。ニブルが 15 のとき、255 が続く限り加算する。
/// `limit` を超えた時点で打ち切るので、0xff の羅列で桁が溢れることはない。
fn lz4_len_ext(src: &[u8], ip: &mut usize, limit: usize) -> Result<usize> {
    let mut extra: u64 = 0;
    loop {
        ensure!(*ip < src.len(), UnexpectedEof, *ip);
        let b = src[*ip];
        *ip += 1;
        extra += b as u64;
        ensure!(extra <= limit as u64, LimitExceeded, *ip);
        if b != 255 {
            return Ok(extra as usize);
        }
    }
}

/// LZ4 block format（フレーミング無し）を展開して `out` に追記する。
pub fn lz4_raw_decompress(src: &[u8], out_len: usize, out: &mut Vec<u8>) -> Result<()> {
    let base = out.len();
    let mut ip = 0usize;

    while ip < src.len() {
        let token = src[ip];
        ip += 1;

        // リテラル部。
        let mut lit = (token >> 4) as usize;
        if lit == 15 {
            lit += lz4_len_ext(src, &mut ip, out_len)?;
        }
        ensure!(lit <= src.len() - ip, UnexpectedEof, ip);
        ensure!(lit <= out_len - (out.len() - base), LimitExceeded, ip);
        out.extend_from_slice(&src[ip..ip + lit]);
        ip += lit;

        // 最終シーケンスはリテラルだけで終わる（マッチ部を持たない）。
        if ip == src.len() {
            break;
        }

        ensure!(src.len() - ip >= 2, UnexpectedEof, ip);
        let offset = u16::from_le_bytes([src[ip], src[ip + 1]]) as usize;
        ip += 2;
        let written = out.len() - base;
        ensure!(offset > 0 && offset <= written, BadCompressedData, ip);

        // マッチ部。最小マッチ長 4 がニブルから引かれている。
        let mut mlen = (token & 0x0f) as usize + 4;
        if token & 0x0f == 15 {
            mlen += lz4_len_ext(src, &mut ip, out_len)?;
        }
        ensure!(mlen <= out_len - written, LimitExceeded, ip);
        copy_within(out, offset, mlen);
    }

    ensure!(out.len() - base == out_len, BadCompressedData, ip);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Code;

    // --- フィクスチャ -------------------------------------------------------
    // snap / lz4_flex クレートで実際に圧縮したバイト列（16 進）。
    // 生成物だけを持ち込み、ワークスペースには依存を足さない。

    const SNAPPY_HELLO: &str = "2b1468656c6c6f2046060048776f726c642c2068656c6c6f20776f726c6421";
    const LZ4_HELLO: &str = "6e68656c6c6f20060063776f726c642c190060776f726c6421";
    // `zstd` CLI で `hello()` と同じ文字列を圧縮したもの
    // （`printf '...' | zstd -q -c | xxd -p`）。
    #[cfg(feature = "zstd")]
    const ZSTD_HELLO: &str =
        "28b52ffd0458dd00009068656c6c6f20776f726c642c776f726c642102003cb312af140157c1b30b";
    const SNAPPY_RUN: &str = concat!(
        "88270061fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe",
        "0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100",
        "fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe01",
        "00fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe",
        "0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100fe0100",
        "fe0100fe0100fe0100fe0100fe0100fe01000d01",
    );
    const LZ4_RUN: &str = "1f610100ffffffffffffffffffffffffffffffffffffff8160616161616161";
    const SNAPPY_PATTERN: &str = concat!(
        "807d1c6162636465666768fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800",
        "fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe08",
        "00fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe",
        "0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800",
        "fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe08",
        "00fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe",
        "0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800",
        "fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe08",
        "00fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe",
        "0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800",
        "fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe08",
        "00fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe",
        "0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800",
        "fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe08",
        "00fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe",
        "0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800",
        "fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe0800fe08",
        "00fe0800fe0800fe0800de0800",
    );
    const LZ4_PATTERN: &str = concat!(
        "8f61626364656667680800ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffff9d60636465666768",
    );
    const SNAPPY_BIG: &str = concat!(
        "f0a2047874686520717569636b2062726f776e20666f78206a756d7073206f76657220011f206c617a792064",
        "6f672e050efe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe",
        "2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe",
        "2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe",
        "2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe",
        "2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe",
        "2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe",
        "2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe",
        "2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe",
        "2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe",
        "2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe",
        "2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe",
        "2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe",
        "2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe",
        "2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe",
        "2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe",
        "2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe",
        "2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe",
        "2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe",
        "2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe",
        "2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe",
        "2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe",
        "2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe",
        "2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe",
        "2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d003a2d006c666f",
        "78206a756d7073206f76657220746865206c617a7920646f672e050e2c717569636b2062726f776e20fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe",
        "2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00",
        "fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d",
        "00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00fe2d00ee2d000d2d",
    );
    const LZ4_BIG: &str = concat!(
        "f01074686520717569636b2062726f776e20666f78206a756d7073206f766572201f00916c617a7920646f67",
        "2e0e000f2d00ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "ffffffffffffffffffffffffffffffff3860206a756d7073",
    );
    const SNAPPY_RANDOM: &str = concat!(
        "e807f4e70375cd254b84e2eaf2a68120674334b26e4be2995473767ff1cc75998d1eabcedb9739656eca98c3",
        "710b6efa89a43b2f146e3b1a47c51424caf617e8dc823a1a1ce60a1687c941a33eafbba02896f3b08a2d4f18",
        "1f71d3644df1158581ff767bdb9408cb99a2a83c03251dadbc2b63ffc09f5e576a9998424b71dd79c0e9c14c",
        "70186ff63df949623dc63c1ed377c68af81c1f7d49e0fbe66ef1d236ee95b08e892dedf09df4580d40adb553",
        "0c928b63ca41bb137272489831af4a8e775e0be327d2a5f67940ba5b30bb17131c99553f21efcf892efc7eca",
        "e6ae90b8d9a3d0949f4a3128260c098c0993e886cdacfa783c4b6c305cae510cafb2e24fa7c5308d991d9a8a",
        "8a0b9305999e9f3afaf0850caebf193e6d402a1785477253882d8f5d1415600321755a508108aa20958e369d",
        "5ef7cdce67e923c435ba67283644873861417762fb05131838c1688da1b17f7088e4f8dfe3690f125e1ddc94",
        "7f79e3f624f54555bcfc292e087c2f3a398c0870babc7a520fff6f257446dec2473da61256c7a7729b1cb987",
        "0fb245468751aa29d8f7a8eeac737c44c1d07043850904ad4693ec3065ad963d409fe930ce3a76cf1a534d5d",
        "110c21d85f1889094256668a9477d8eb777b458b604249df6183ad3609d0a42e279a00703f40136b9da36811",
        "34d6a1e5c7d500d309618c9caa6cc7d886b461d26b6d6bc0262fec26ca72dd4e7258f241dc8eb599aa82c814",
        "243b6fd3c387a8a0f05b6b9faa8ee6570e4bc8f1e8e7ed08d2ff068686f20b57613783ea471613d25abebb06",
        "3accc983c5b6624d8ab4232558034a68b1c254dd05ae1781ef74ae86e472cc15141a4e905d4419599b2f9c69",
        "b1f8892d17c4997dcaf072a3e8f3fcc05bdac10f4c6e24ca59878eb3b512d489b6c482ed7491c1757a500efc",
        "b62ff26421db725acc83eee14f14959ac555ba1dd3cda6de5d86aa67f8299ae03103f93e2cdf6b824fd741e7",
        "ea8737f10d3ab2a006fe1c5f36c1225738d6f9448391effb04d2ed3465aba954e97eb14021439274e3c9beef",
        "3a28f0b043b842c22b3e252ec928ea447306a3031e9c74b3eb36a7dc931f18347761cb3d3433c097feb2604d",
        "2bd40ba8b0b1a2337087251aa71fb001d1786ff7d0da60313eb0f2a521dbdbb37052ac7b293c0c88daea9de8",
        "e84cdfefecd1b623f9c24e21a49c8b7210ea419e729ddcfdbef63c096b73e8369152dd97c8a898cb7e0d2b06",
        "bf6772279a54b62f9e39609022f8802144d448bfdd07c5ccdd017322bd747f3dfc85c194e2d2cd6ef729afba",
        "e01b1ba09dc532e35885264737635a81e872c788dc70c7286061914411f9f62651fc43b8c0004ea6e32d04a4",
        "74db11660447720bf2a308fbdf427f8117b514a4a7fe22020c447aafb5a0d664caf81371ed",
    );
    const LZ4_RANDOM: &str = concat!(
        "f0ffffffdc75cd254b84e2eaf2a68120674334b26e4be2995473767ff1cc75998d1eabcedb9739656eca98c3",
        "710b6efa89a43b2f146e3b1a47c51424caf617e8dc823a1a1ce60a1687c941a33eafbba02896f3b08a2d4f18",
        "1f71d3644df1158581ff767bdb9408cb99a2a83c03251dadbc2b63ffc09f5e576a9998424b71dd79c0e9c14c",
        "70186ff63df949623dc63c1ed377c68af81c1f7d49e0fbe66ef1d236ee95b08e892dedf09df4580d40adb553",
        "0c928b63ca41bb137272489831af4a8e775e0be327d2a5f67940ba5b30bb17131c99553f21efcf892efc7eca",
        "e6ae90b8d9a3d0949f4a3128260c098c0993e886cdacfa783c4b6c305cae510cafb2e24fa7c5308d991d9a8a",
        "8a0b9305999e9f3afaf0850caebf193e6d402a1785477253882d8f5d1415600321755a508108aa20958e369d",
        "5ef7cdce67e923c435ba67283644873861417762fb05131838c1688da1b17f7088e4f8dfe3690f125e1ddc94",
        "7f79e3f624f54555bcfc292e087c2f3a398c0870babc7a520fff6f257446dec2473da61256c7a7729b1cb987",
        "0fb245468751aa29d8f7a8eeac737c44c1d07043850904ad4693ec3065ad963d409fe930ce3a76cf1a534d5d",
        "110c21d85f1889094256668a9477d8eb777b458b604249df6183ad3609d0a42e279a00703f40136b9da36811",
        "34d6a1e5c7d500d309618c9caa6cc7d886b461d26b6d6bc0262fec26ca72dd4e7258f241dc8eb599aa82c814",
        "243b6fd3c387a8a0f05b6b9faa8ee6570e4bc8f1e8e7ed08d2ff068686f20b57613783ea471613d25abebb06",
        "3accc983c5b6624d8ab4232558034a68b1c254dd05ae1781ef74ae86e472cc15141a4e905d4419599b2f9c69",
        "b1f8892d17c4997dcaf072a3e8f3fcc05bdac10f4c6e24ca59878eb3b512d489b6c482ed7491c1757a500efc",
        "b62ff26421db725acc83eee14f14959ac555ba1dd3cda6de5d86aa67f8299ae03103f93e2cdf6b824fd741e7",
        "ea8737f10d3ab2a006fe1c5f36c1225738d6f9448391effb04d2ed3465aba954e97eb14021439274e3c9beef",
        "3a28f0b043b842c22b3e252ec928ea447306a3031e9c74b3eb36a7dc931f18347761cb3d3433c097feb2604d",
        "2bd40ba8b0b1a2337087251aa71fb001d1786ff7d0da60313eb0f2a521dbdbb37052ac7b293c0c88daea9de8",
        "e84cdfefecd1b623f9c24e21a49c8b7210ea419e729ddcfdbef63c096b73e8369152dd97c8a898cb7e0d2b06",
        "bf6772279a54b62f9e39609022f8802144d448bfdd07c5ccdd017322bd747f3dfc85c194e2d2cd6ef729afba",
        "e01b1ba09dc532e35885264737635a81e872c788dc70c7286061914411f9f62651fc43b8c0004ea6e32d04a4",
        "74db11660447720bf2a308fbdf427f8117b514a4a7fe22020c447aafb5a0d664caf81371ed",
    );
    const SNAPPY_EMPTY: &str = "00";
    const LZ4_EMPTY: &str = "00";
    const SNAPPY_ONE: &str = "01007f";
    const LZ4_ONE: &str = "107f";
    const SNAPPY_MIXED: &str = concat!(
        "fe08f43e0175cd254b84e2eaf2a68120674334b26e4be2995473767ff1cc75998d1eabcedb9739656eca98c3",
        "710b6efa89a43b2f146e3b1a47c51424caf617e8dc823a1a1ce60a1687c941a33eafbba02896f3b08a2d4f18",
        "1f71d3644df1158581ff767bdb9408cb99a2a83c03251dadbc2b63ffc09f5e576a9998424b71dd79c0e9c14c",
        "70186ff63df949623dc63c1ed377c68af81c1f7d49e0fbe66ef1d236ee95b08e892dedf09df4580d40adb553",
        "0c928b63ca41bb137272489831af4a8e775e0be327d2a5f67940ba5b30bb17131c99553f21efcf892efc7eca",
        "e6ae90b8d9a3d0949f4a3128260c098c0993e886cdacfa783c4b6c305cae510cafb2e24fa7c5308d991d9a8a",
        "8a0b9305999e9f3afaf0850caebf193e6d402a1785477253882d8f5d1415600321755a508108aa20955a5a5a",
        "5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5afe0a00fe0a00fe0a00fe0a00fe0a00fe0a00fe0a00fe0a00fe0a00fe",
        "0a00a20a00086162636a0300fe0604de0604",
    );
    const LZ4_MIXED: &str = concat!(
        "ffff2275cd254b84e2eaf2a68120674334b26e4be2995473767ff1cc75998d1eabcedb9739656eca98c3710b",
        "6efa89a43b2f146e3b1a47c51424caf617e8dc823a1a1ce60a1687c941a33eafbba02896f3b08a2d4f181f71",
        "d3644df1158581ff767bdb9408cb99a2a83c03251dadbc2b63ffc09f5e576a9998424b71dd79c0e9c14c7018",
        "6ff63df949623dc63c1ed377c68af81c1f7d49e0fbe66ef1d236ee95b08e892dedf09df4580d40adb5530c92",
        "8b63ca41bb137272489831af4a8e775e0be327d2a5f67940ba5b30bb17131c99553f21efcf892efc7ecae6ae",
        "90b8d9a3d0949f4a3128260c098c0993e886cdacfa783c4b6c305cae510cafb2e24fa7c5308d991d9a8a8a0b",
        "9305999e9f3afaf0850caebf193e6d402a1785477253882d8f5d1415600321755a508108aa20955a5a5a5a04",
        "00ffffa73f6162630300080f06045f60576a9998424b",
    );

    fn hex(s: &str) -> Vec<u8> {
        fn nib(c: u8) -> u8 {
            match c {
                b'0'..=b'9' => c - b'0',
                _ => c - b'a' + 10,
            }
        }
        let b = s.as_bytes();
        b.chunks(2).map(|p| nib(p[0]) << 4 | nib(p[1])).collect()
    }

    /// フィクスチャ生成側と同じ線形合同法。非圧縮データの再現に使う。
    fn lcg(n: usize) -> Vec<u8> {
        let mut s: u32 = 0x1234_5678;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 24) as u8
            })
            .collect()
    }

    fn repeat(pat: &[u8], n: usize) -> Vec<u8> {
        pat.iter().cycle().take(n).cloned().collect()
    }

    fn hello() -> Vec<u8> {
        b"hello hello hello hello world, hello world!".to_vec()
    }
    fn run() -> Vec<u8> {
        vec![b'a'; 5000]
    }
    fn pattern() -> Vec<u8> {
        repeat(b"abcdefgh", 16_000)
    }
    fn big() -> Vec<u8> {
        repeat(b"the quick brown fox jumps over the lazy dog. ", 70_000)
    }
    fn mixed() -> Vec<u8> {
        let mut v = lcg(300);
        v.extend(std::iter::repeat_n(b'Z', 700));
        v.extend_from_slice(b"abcabcabcabcabcabcabcabcabcabc");
        v.extend(lcg(120));
        v
    }

    fn snap_ok(fixture: &str, expect: &[u8]) {
        let src = hex(fixture);
        let mut out = Vec::new();
        snappy_decompress(&src, expect.len(), &mut out).unwrap();
        assert_eq!(out.len(), expect.len());
        assert!(out == expect);
        // 宣言サイズに余裕があっても結果は変わらない。
        let mut out2 = Vec::new();
        snappy_decompress(&src, expect.len() + 4096, &mut out2).unwrap();
        assert!(out2 == expect);
    }

    fn lz4_ok(fixture: &str, expect: &[u8]) {
        let src = hex(fixture);
        let mut out = Vec::new();
        lz4_raw_decompress(&src, expect.len(), &mut out).unwrap();
        assert_eq!(out.len(), expect.len());
        assert!(out == expect);
    }

    fn code(e: Error) -> u16 {
        e.code_u16()
    }

    // --- Snappy: 正常系 -----------------------------------------------------

    #[test]
    fn snappy_small_literal_and_copy() {
        snap_ok(SNAPPY_HELLO, &hello());
        snap_ok(SNAPPY_ONE, &[0x7f]);
    }

    #[test]
    fn snappy_empty() {
        snap_ok(SNAPPY_EMPTY, &[]);
        // 空入力そのものは前置きが読めないので EOF。
        let mut out = Vec::new();
        assert_eq!(
            code(snappy_decompress(&[], 0, &mut out).unwrap_err()),
            Code::UnexpectedEof as u16
        );
    }

    #[test]
    fn snappy_overlapping_single_byte_run() {
        // offset=1 の自己参照コピー。素朴な実装が壊れる典型。
        snap_ok(SNAPPY_RUN, &run());
    }

    #[test]
    fn snappy_overlapping_multi_byte_pattern() {
        snap_ok(SNAPPY_PATTERN, &pattern());
    }

    #[test]
    fn snappy_incompressible_all_literal() {
        // 1000 バイトのリテラル → 長さは 2 バイト拡張で符号化される。
        snap_ok(SNAPPY_RANDOM, &lcg(1000));
    }

    #[test]
    fn snappy_mixed_literal_and_copy() {
        snap_ok(SNAPPY_MIXED, &mixed());
    }

    #[test]
    fn snappy_over_64k() {
        // 前置き varint が複数バイトになる領域。
        snap_ok(SNAPPY_BIG, &big());
    }

    #[test]
    fn snappy_three_byte_literal_length() {
        // 64KiB 超のリテラルは長さが 3 バイトになる。実圧縮器は分割するので
        // ここだけは手で組む。
        let data = lcg(70_000);
        let mut src = Vec::new();
        // 前置き: 70000 = 0xf0 0xa2 0x04
        let mut n = data.len() as u64;
        while n >= 0x80 {
            src.push((n as u8) | 0x80);
            n >>= 7;
        }
        src.push(n as u8);
        // タグ: n=62 → 3 バイト長（値は 長さ-1）
        src.push(62 << 2);
        let l = (data.len() - 1) as u32;
        src.extend_from_slice(&l.to_le_bytes()[..3]);
        src.extend_from_slice(&data);

        let mut out = Vec::new();
        snappy_decompress(&src, data.len(), &mut out).unwrap();
        assert!(out == data);
    }

    #[test]
    fn snappy_appends_to_existing_output() {
        // `out` に既存データがあってもオフセットの基準は今回の開始位置。
        let mut out = vec![0xaa; 100];
        snappy_decompress(&hex(SNAPPY_RUN), 5000, &mut out).unwrap();
        assert_eq!(out.len(), 5100);
        assert!(out[..100].iter().all(|&b| b == 0xaa));
        assert!(out[100..].iter().all(|&b| b == b'a'));
    }

    #[test]
    fn snappy_cannot_reach_before_base() {
        // 先頭 1 バイトのリテラルの直後に offset=2 のコピー。基準より前の
        // 既存データを参照しようとするので拒否されなければならない。
        let src = [0x08, 0x00, 0xaa, 0x01, 0x02];
        let mut out = vec![0xaa; 64];
        let e = snappy_decompress(&src, 8, &mut out).unwrap_err();
        assert_eq!(code(e), Code::BadCompressedData as u16);
    }

    // --- Snappy: 異常系 -----------------------------------------------------

    #[test]
    fn snappy_declared_length_over_limit() {
        // 展開後 5000 バイトの正当なストリームでも上限 100 なら拒否。
        let mut out = Vec::new();
        let e = snappy_decompress(&hex(SNAPPY_RUN), 100, &mut out).unwrap_err();
        assert_eq!(code(e), Code::BadCompressedData as u16);
        assert!(out.is_empty());
    }

    #[test]
    fn snappy_truncated_preamble() {
        let mut out = Vec::new();
        // 継続ビットが立ったまま終わる。
        assert!(snappy_decompress(&[0x80], 100, &mut out).is_err());
        assert!(snappy_decompress(&[0x80, 0x80, 0x80, 0x80, 0x80, 0x80], 100, &mut out).is_err());
    }

    #[test]
    fn snappy_truncated_mid_literal() {
        let full = hex(SNAPPY_HELLO);
        for cut in 1..full.len() {
            let mut out = Vec::new();
            let r = snappy_decompress(&full[..cut], 4096, &mut out);
            assert!(r.is_err(), "cut={cut} で成功してしまった");
        }
    }

    #[test]
    fn snappy_truncated_mid_offset() {
        // どれも「宣言 8 / リテラル 1 バイト」の後にコピーが中途で切れる形。
        for src in [
            // 4 バイトオフセットを 2 バイトで打ち切る。
            &[0x08u8, 0x00, 0xaa, 0x0f, 0x01, 0x00][..],
            // 2 バイトオフセットを 1 バイトで打ち切る。
            &[0x08, 0x00, 0xaa, 0x0e, 0x01][..],
            // 1 バイトオフセットの続きが無い。
            &[0x08, 0x00, 0xaa, 0x01][..],
        ] {
            let mut out = Vec::new();
            assert_eq!(
                code(snappy_decompress(src, 64, &mut out).unwrap_err()),
                Code::UnexpectedEof as u16
            );
        }
    }

    #[test]
    fn snappy_zero_offset() {
        let src = [0x08, 0x00, 0xaa, 0x0e, 0x00, 0x00];
        let mut out = Vec::new();
        assert_eq!(
            code(snappy_decompress(&src, 64, &mut out).unwrap_err()),
            Code::BadCompressedData as u16
        );
    }

    #[test]
    fn snappy_offset_beyond_written() {
        // 1 バイトしか書いていないのに offset=9999。
        let src = [0x08, 0x00, 0xaa, 0x0e, 0x0f, 0x27];
        let mut out = Vec::new();
        assert_eq!(
            code(snappy_decompress(&src, 64, &mut out).unwrap_err()),
            Code::BadCompressedData as u16
        );
        // 4 バイトオフセットで u32 上限。usize が 32 bit でも溢れないこと。
        let src = [0x08, 0x00, 0xaa, 0x0f, 0xff, 0xff, 0xff, 0xff];
        let mut out = Vec::new();
        assert!(snappy_decompress(&src, 64, &mut out).is_err());
    }

    #[test]
    fn snappy_output_exceeds_declared() {
        // 宣言 2 バイトなのにリテラル 8 バイトが続く。
        let src = [0x02, 0x1c, 1, 2, 3, 4, 5, 6, 7, 8];
        let mut out = Vec::new();
        assert_eq!(
            code(snappy_decompress(&src, 64, &mut out).unwrap_err()),
            Code::LimitExceeded as u16
        );
        // コピーで超過する場合。
        let src = [0x04, 0x04, 0x08, 0x01, 0x01];
        let mut out = Vec::new();
        assert!(snappy_decompress(&src, 64, &mut out).is_err());
    }

    #[test]
    fn snappy_output_shorter_than_declared() {
        // 宣言 8 バイトだがリテラルは 2 バイトだけ。
        let src = [0x08, 0x04, 0xaa, 0xbb];
        let mut out = Vec::new();
        assert_eq!(
            code(snappy_decompress(&src, 64, &mut out).unwrap_err()),
            Code::BadCompressedData as u16
        );
    }

    #[test]
    fn snappy_arbitrary_bytes_never_panic() {
        // 実データを 1 バイトずつ壊しても Err で済むこと。
        let full = hex(SNAPPY_MIXED);
        for i in 0..full.len() {
            for m in [0x00u8, 0x55, 0xff] {
                let mut s = full.clone();
                s[i] ^= m;
                let mut out = Vec::new();
                let _ = snappy_decompress(&s, 1150, &mut out);
            }
        }
    }

    // --- LZ4: 正常系 --------------------------------------------------------

    #[test]
    fn lz4_small_literal_and_match() {
        lz4_ok(LZ4_HELLO, &hello());
        lz4_ok(LZ4_ONE, &[0x7f]);
    }

    #[test]
    fn lz4_empty() {
        lz4_ok(LZ4_EMPTY, &[]);
        // 入力が空でも out_len が 0 なら成功（シーケンス無し）。
        let mut out = Vec::new();
        lz4_raw_decompress(&[], 0, &mut out).unwrap();
        assert!(out.is_empty());
        // out_len > 0 で入力が空なら足りない。
        let mut out = Vec::new();
        assert!(lz4_raw_decompress(&[], 4, &mut out).is_err());
    }

    #[test]
    fn lz4_overlapping_single_byte_run() {
        lz4_ok(LZ4_RUN, &run());
    }

    #[test]
    fn lz4_overlapping_multi_byte_pattern() {
        lz4_ok(LZ4_PATTERN, &pattern());
    }

    #[test]
    fn lz4_incompressible_all_literal() {
        // リテラル長 1000 → 0xff の連続で拡張される。
        lz4_ok(LZ4_RANDOM, &lcg(1000));
    }

    #[test]
    fn lz4_mixed_literal_and_match() {
        lz4_ok(LZ4_MIXED, &mixed());
    }

    #[test]
    fn lz4_over_64k() {
        // マッチ長が 64KiB を超え、拡張バイトが大量に続く。
        lz4_ok(LZ4_BIG, &big());
    }

    #[test]
    fn lz4_appends_to_existing_output() {
        let mut out = vec![0xaa; 100];
        lz4_raw_decompress(&hex(LZ4_RUN), 5000, &mut out).unwrap();
        assert_eq!(out.len(), 5100);
        assert!(out[..100].iter().all(|&b| b == 0xaa));
        assert!(out[100..].iter().all(|&b| b == b'a'));
    }

    #[test]
    fn lz4_cannot_reach_before_base() {
        // リテラル 1 バイト → offset=2 は基準より前。
        let src = [0x10, 0x00, 0x02, 0x00, 0x00];
        let mut out = vec![0xaa; 64];
        assert_eq!(
            code(lz4_raw_decompress(&src, 8, &mut out).unwrap_err()),
            Code::BadCompressedData as u16
        );
    }

    // --- LZ4: 異常系 --------------------------------------------------------

    #[test]
    fn lz4_truncated_mid_literal() {
        let full = hex(LZ4_HELLO);
        for cut in 1..full.len() {
            let mut out = Vec::new();
            assert!(
                lz4_raw_decompress(&full[..cut], 43, &mut out).is_err(),
                "cut={cut} で成功してしまった"
            );
        }
    }

    #[test]
    fn lz4_truncated_mid_offset() {
        // リテラル 4 バイトの直後にオフセットが 1 バイトしかない。
        let src = [0x40, 1, 2, 3, 4, 0x01];
        let mut out = Vec::new();
        assert_eq!(
            code(lz4_raw_decompress(&src, 64, &mut out).unwrap_err()),
            Code::UnexpectedEof as u16
        );
    }

    #[test]
    fn lz4_truncated_length_extension() {
        // リテラル長ニブル 15 で 0xff が続いたまま入力が尽きる。
        let src = [0xf0, 0xff, 0xff, 0xff];
        let mut out = Vec::new();
        assert!(lz4_raw_decompress(&src, 1 << 20, &mut out).is_err());
        // マッチ長側も同様。
        let src = [0x4f, 1, 2, 3, 4, 0x01, 0x00, 0xff, 0xff];
        let mut out = Vec::new();
        assert!(lz4_raw_decompress(&src, 1 << 20, &mut out).is_err());
    }

    #[test]
    fn lz4_zero_offset() {
        let src = [0x40, 1, 2, 3, 4, 0x00, 0x00, 0x00];
        let mut out = Vec::new();
        assert_eq!(
            code(lz4_raw_decompress(&src, 64, &mut out).unwrap_err()),
            Code::BadCompressedData as u16
        );
    }

    #[test]
    fn lz4_offset_beyond_written() {
        let src = [0x40, 1, 2, 3, 4, 0x64, 0x00, 0x00];
        let mut out = Vec::new();
        assert_eq!(
            code(lz4_raw_decompress(&src, 64, &mut out).unwrap_err()),
            Code::BadCompressedData as u16
        );
    }

    #[test]
    fn lz4_output_exceeds_out_len() {
        // リテラルだけで上限超過。
        let src = [0x80, 1, 2, 3, 4, 5, 6, 7, 8];
        let mut out = Vec::new();
        assert_eq!(
            code(lz4_raw_decompress(&src, 4, &mut out).unwrap_err()),
            Code::LimitExceeded as u16
        );
        // マッチで上限超過（リテラル 4 + マッチ 8 > 8）。
        let src = [0x44, 1, 2, 3, 4, 0x04, 0x00, 0x00];
        let mut out = Vec::new();
        assert!(lz4_raw_decompress(&src, 8, &mut out).is_err());
    }

    #[test]
    fn lz4_output_shorter_than_out_len() {
        let src = [0x40, 1, 2, 3, 4];
        let mut out = Vec::new();
        assert_eq!(
            code(lz4_raw_decompress(&src, 64, &mut out).unwrap_err()),
            Code::BadCompressedData as u16
        );
    }

    #[test]
    fn lz4_arbitrary_bytes_never_panic() {
        let full = hex(LZ4_MIXED);
        for i in 0..full.len() {
            for m in [0x00u8, 0x55, 0xff] {
                let mut s = full.clone();
                s[i] ^= m;
                let mut out = Vec::new();
                let _ = lz4_raw_decompress(&s, 1150, &mut out);
            }
        }
    }

    // --- ディスパッチ -------------------------------------------------------

    #[test]
    fn decompress_dispatch() {
        let raw = decompress(Compression::Uncompressed, b"abc", 3).unwrap();
        assert_eq!(raw, b"abc");

        let s = decompress(Compression::Snappy, &hex(SNAPPY_HELLO), 43).unwrap();
        assert!(s == hello());

        let l = decompress(Compression::Lz4Raw, &hex(LZ4_HELLO), 43).unwrap();
        assert!(l == hello());

        // ZSTD は `zstd` フィーチャが付いていれば内蔵で展開する。
        // 空バイト列はそもそも妥当な ZSTD フレームではないので、
        // 「委譲が要る」ではなく「壊れた入力」のエラーになる。
        #[cfg(feature = "zstd")]
        {
            let z = decompress(Compression::Zstd, &hex(ZSTD_HELLO), 43).unwrap();
            assert_eq!(z, hello());
            assert_ne!(
                code(decompress(Compression::Zstd, b"", 0).unwrap_err()),
                Code::UnsupportedCodec as u16,
                "内蔵しているので UnsupportedCodec にはならないはず"
            );
        }
        // `zstd` を外した構成では旧来どおりホスト委譲（UnsupportedCodec）に戻る。
        #[cfg(not(feature = "zstd"))]
        assert_eq!(
            code(decompress(Compression::Zstd, b"", 0).unwrap_err()),
            Code::UnsupportedCodec as u16
        );
        // GZIP は `zstd` の有無に関わらず常にホスト委譲。
        assert_eq!(
            code(decompress(Compression::Gzip, b"", 0).unwrap_err()),
            Code::UnsupportedCodec as u16
        );
    }
}
