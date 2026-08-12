//! ahiru-zstd -- a decompression-only Zstandard (RFC 8878) decoder.
//!
//! The ahirudb core has a 1 MiB size budget, so ZSTD is deliberately kept out of
//! it and shipped as a separately loaded wasm side module (`docs/DESIGN.md` §3 / §6).
//! This crate therefore does not depend on the core and carries its own allocator
//! and panic handler. Zero dependencies.
//!
//! The supported subset is what Parquet writes: single frames without a dictionary.
//! Input carrying a dictionary (Dictionary_ID) returns `Err`.
//!
//! Build configurations:
//! - native (default): the `std` feature is on. Tests and debugging live here.
//! - wasm distribution: `no_std` via `--no-default-features`.

#![cfg_attr(not(feature = "std"), no_std)]
// alloc is always used, even under no_std.
extern crate alloc;

/// Records which paths were taken during decoding. Enabled in tests only; in
/// distribution builds the calls disappear entirely (this exists to verify that
/// format branches were actually exercised).
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

// The wasm ABI entry points (the `ahiru_zstd_*` exported functions) are only
// needed when this crate ships as a standalone wasm module (`standalone`). When
// it is linked into `ahiru-core` as a library, only `ahiru-core`'s own ABI should
// be exposed, so this is left out.
#[cfg(all(target_arch = "wasm32", feature = "standalone"))]
pub mod abi;

/// The imports shared across the crate.
pub(crate) mod prelude {
    #![allow(unused_imports)]

    pub use alloc::vec;
    pub use alloc::vec::Vec;
}

use prelude::*;

/// Errors are expressed as numeric codes. The point is to avoid pulling in
/// `core::fmt`; the wasm ABI returns these values negated.
// Debug is derived only for native tests (deriving it on wasm links `core::fmt`
// and costs tens of KB).
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
#[repr(i32)]
pub enum Error {
    /// Input ran out in the middle of a structure.
    UnexpectedEof = 1,
    /// The frame magic does not match.
    BadMagic = 2,
    /// Reserved bits in the frame header, or a declared-size mismatch.
    BadFrameHeader = 3,
    /// A frame with a dictionary. Unsupported.
    DictionaryUnsupported = 4,
    /// Invalid block header (reserved type, oversized).
    BadBlock = 5,
    /// Corrupt literals section.
    BadLiterals = 6,
    /// Corrupt Huffman table / stream.
    BadHuffman = 7,
    /// Corrupt FSE table.
    BadFse = 8,
    /// Corrupt sequences section.
    BadSequence = 9,
    /// An offset pointing before the start of the output.
    BadOffset = 10,
    /// Content_Checksum mismatch.
    ChecksumMismatch = 11,
    /// The decompressed result exceeds the caller's declared size.
    LimitExceeded = 12,
    /// The bitstream has no end marker.
    BadBitstream = 13,
}

/// Decompresses `src`. `out_len` is the decompressed size the caller knows, and
/// the decoder never writes beyond it (an upper bound against untrusted input).
pub fn decompress(src: &[u8], out_len: usize) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    // Reserve the declared size up front. Corrupt input cannot make this grow without bound.
    out.try_reserve_exact(out_len).map_err(|_| Error::LimitExceeded)?;
    frame::decompress_into(src, &mut out, out_len)?;
    Ok(out)
}

/// Decompresses `src` and appends to the end of `out`. At most `max_len` bytes are appended.
pub fn decompress_into(src: &[u8], out: &mut Vec<u8>, max_len: usize) -> Result<(), Error> {
    frame::decompress_into(src, out, max_len)
}

/// Path counters. Purely a mechanism for verifying in tests that format branches
/// were actually exercised, so nothing of it remains in distribution builds.
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
    // Tests run in parallel, so this is thread-local.
    std::thread_local! {
        static SEEN: Cell<u32> = const { Cell::new(0) };
    }
    pub fn hit(bit: u32) {
        SEEN.with(|c| c.set(c.get() | (1 << bit)));
    }
    /// Takes the bitmask of paths taken so far and resets it to 0.
    pub fn take() -> u32 {
        SEEN.with(|c| c.replace(0))
    }
}

#[cfg(test)]
mod tests;
