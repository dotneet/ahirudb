//! ahiru-zstd — Zstandard (RFC 8878) 展開専用デコーダ。
//!
//! ahirudb のコアは 1 MiB のサイズ予算を持つため、ZSTD は意図的にコアから外し、
//! 別ロードする wasm サイドモジュールとして配る（`docs/DESIGN.md` §3 / §6）。
//! したがってこのクレートはコアに依存せず、アロケータもパニックハンドラも
//! 自前で持つ。依存クレートはゼロ。
//!
//! 対応範囲は Parquet が書く範囲、つまり「辞書無しの単体フレーム」。
//! 辞書 (Dictionary_ID) が付いた入力は `Err` を返す。
//!
//! ビルド構成:
//! - ネイティブ (既定): `std` フィーチャ有効。テストとデバッグはこちら。
//! - wasm 配布: `--no-default-features` で `no_std`。

#![cfg_attr(not(feature = "std"), no_std)]
// no_std でも alloc は常に使う。
extern crate alloc;

/// 復号中に通った経路を記録する。テストでのみ有効で、配布ビルドでは
/// 呼び出しごと消える（フォーマットの分岐を実際に踏んだか検証するため）。
#[cfg(test)]
macro_rules! stat {
    ($bit:expr) => {
        crate::stats::hit($bit)
    };
}
#[cfg(not(test))]
macro_rules! stat {
    ($bit:expr) => {
        ()
    };
}

mod bits;
mod frame;
mod fse;
mod huff;
mod xxh64;

pub mod rt;

#[cfg(target_arch = "wasm32")]
pub mod abi;

/// クレート内で共通に使う import。
pub(crate) mod prelude {
    #![allow(unused_imports)]

    pub use alloc::vec;
    pub use alloc::vec::Vec;
}

use prelude::*;

/// エラーは数値コードで表現する。`core::fmt` を引かないのが目的で、
/// wasm ABI ではこの値を負数にして返す。
// Debug はネイティブのテストでのみ導出する（wasm で導出すると core::fmt が
// リンクされて数十 KB を失う）。
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
#[repr(i32)]
pub enum Error {
    /// 入力が構造の途中で尽きた。
    UnexpectedEof = 1,
    /// フレームマジックが合わない。
    BadMagic = 2,
    /// フレームヘッダの予約ビット / 宣言サイズの不一致。
    BadFrameHeader = 3,
    /// 辞書付きフレーム。非対応。
    DictionaryUnsupported = 4,
    /// ブロックヘッダが不正（予約タイプ、サイズ超過）。
    BadBlock = 5,
    /// リテラル部の破損。
    BadLiterals = 6,
    /// ハフマン表 / ストリームの破損。
    BadHuffman = 7,
    /// FSE 表の破損。
    BadFse = 8,
    /// シーケンス部の破損。
    BadSequence = 9,
    /// 出力先頭より前を指すオフセット。
    BadOffset = 10,
    /// Content_Checksum 不一致。
    ChecksumMismatch = 11,
    /// 展開結果が呼び出し側の宣言サイズを超える。
    LimitExceeded = 12,
    /// ビットストリームの終端マーカが無い。
    BadBitstream = 13,
}

/// `src` を展開する。`out_len` は呼び出し側が知っている展開後サイズで、
/// デコーダはこれを超えて書き込まない（信用できない入力に対する上限）。
pub fn decompress(src: &[u8], out_len: usize) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    // 宣言サイズ分だけ先に確保する。壊れた入力で青天井に伸びることはない。
    out.try_reserve_exact(out_len).map_err(|_| Error::LimitExceeded)?;
    frame::decompress_into(src, &mut out, out_len)?;
    Ok(out)
}

/// `src` を展開して `out` の末尾に追記する。追記できるのは最大 `max_len` バイト。
pub fn decompress_into(src: &[u8], out: &mut Vec<u8>, max_len: usize) -> Result<(), Error> {
    frame::decompress_into(src, out, max_len)
}

/// 経路カウンタ。テストでフォーマットの分岐を実際に踏んだか検証するためだけの
/// 仕組みなので、配布ビルドには一切残さない。
#[cfg(test)]
pub(crate) mod stats {
    pub const BLOCK_RAW: u32 = 0;
    pub const BLOCK_RLE: u32 = 1;
    pub const BLOCK_COMPRESSED: u32 = 2;
    pub const LIT_RAW: u32 = 3;
    pub const LIT_RLE: u32 = 4;
    pub const LIT_COMPRESSED: u32 = 5;
    pub const LIT_TREELESS: u32 = 6;
    pub const HUF_W_FSE: u32 = 7;
    pub const HUF_W_DIRECT: u32 = 8;
    pub const HUF_1STREAM: u32 = 9;
    pub const HUF_4STREAM: u32 = 10;
    pub const SEQ_PREDEFINED: u32 = 11;
    pub const SEQ_RLE: u32 = 12;
    pub const SEQ_FSE: u32 = 13;
    pub const SEQ_REPEAT: u32 = 14;
    pub const REPEAT_OFFSET: u32 = 15;
    pub const SKIPPABLE: u32 = 16;
    pub const CHECKSUM: u32 = 17;

    use std::cell::Cell;
    // テストは並列に走るのでスレッドローカルに持つ。
    std::thread_local! {
        static SEEN: Cell<u32> = const { Cell::new(0) };
    }
    pub fn hit(bit: u32) {
        SEEN.with(|c| c.set(c.get() | (1 << bit)));
    }
    /// これまでに踏んだ経路のビットマスクを取り出して 0 に戻す。
    pub fn take() -> u32 {
        SEEN.with(|c| c.replace(0))
    }
}

#[cfg(test)]
mod tests;
