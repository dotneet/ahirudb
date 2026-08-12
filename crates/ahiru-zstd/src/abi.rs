//! The wasm ABI.
//!
//! Follows the same conventions as the core (`ahiru-core::abi`): raw pointer plus
//! length, and return values that are nothing but `i32` numeric codes. No message
//! strings (so `core::fmt` is never linked).
//!
//! ```text
//! const src = zstd_alloc(n);  // the host writes the compressed bytes here
//! const dst = zstd_alloc(m);  // m is the decompressed size declared by the page header
//! const r = zstd_decompress(src, n, dst, m);
//! if (r < 0) throw new Error("zstd " + (-r));
//! ```

use crate::prelude::*;

/// Reserves a buffer for the host to write into.
#[no_mangle]
pub extern "C" fn zstd_alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::new();
    if v.try_reserve_exact(len).is_err() {
        return core::ptr::null_mut();
    }
    let p = v.as_mut_ptr();
    core::mem::forget(v);
    p
}

/// Returns a region reserved by `zstd_alloc`.
///
/// # Safety
/// `ptr` must be what `zstd_alloc` returned for the same `len`.
#[no_mangle]
pub unsafe extern "C" fn zstd_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        drop(unsafe { Vec::from_raw_parts(ptr, 0, len) });
    }
}

/// The decompressed byte count on success, or a negative value (an `Error` code) on failure.
///
/// # Safety
/// `src`/`out` must each point at a valid region of `src_len`/`out_cap` bytes.
/// `out` must be what `zstd_alloc` returned (it is temporarily re-wrapped as a
/// `Vec` below, so the layouts have to match).
#[no_mangle]
pub unsafe extern "C" fn zstd_decompress(
    src: *const u8,
    src_len: usize,
    out: *mut u8,
    out_cap: usize,
) -> i32 {
    if src.is_null() || out.is_null() {
        return -(crate::Error::UnexpectedEof as i32);
    }
    let s = unsafe { core::slice::from_raw_parts(src, src_len) };

    // Temporarily wrap the caller-owned buffer as a Vec. `decompress_into` never
    // writes past `out_cap`, so no reallocation happens and the pointer stays put.
    let mut v = unsafe { Vec::from_raw_parts(out, 0, out_cap) };
    let r = crate::decompress_into(s, &mut v, out_cap);
    let n = v.len();
    // The host owns the region, so do not free it.
    core::mem::forget(v);

    match r {
        Ok(()) if n <= i32::MAX as usize => n as i32,
        Ok(()) => -(crate::Error::LimitExceeded as i32),
        Err(e) => -(e as i32),
    }
}
