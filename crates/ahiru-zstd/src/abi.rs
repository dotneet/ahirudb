//! wasm ABI。
//!
//! コア (`ahiru-core::abi`) と同じ流儀に合わせる: 生ポインタ + 長さ、返り値は
//! `i32` の数値コードだけ。メッセージ文字列は持たない（`core::fmt` をリンク
//! させないため）。
//!
//! ```text
//! const src = zstd_alloc(n);  // ホストが圧縮バイト列を書き込む
//! const dst = zstd_alloc(m);  // m はページヘッダが宣言する展開後サイズ
//! const r = zstd_decompress(src, n, dst, m);
//! if (r < 0) throw new Error("zstd " + (-r));
//! ```

use crate::prelude::*;

/// ホストが書き込むためのバッファを確保する。
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

/// `zstd_alloc` で確保した領域を返す。
///
/// # Safety
/// `ptr` は同じ `len` で `zstd_alloc` が返したものでなければならない。
#[no_mangle]
pub unsafe extern "C" fn zstd_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        drop(unsafe { Vec::from_raw_parts(ptr, 0, len) });
    }
}

/// 成功なら展開後バイト数、失敗なら負値（`Error` のコード）。
///
/// # Safety
/// `src`/`out` はそれぞれ `src_len`/`out_cap` バイトの有効な領域を指すこと。
/// `out` は `zstd_alloc` が返したものであること（下で一時的に `Vec` として
/// 包み直すため、レイアウトが一致している必要がある）。
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

    // 呼び出し側が所有するバッファを一時的に Vec として包む。`decompress_into`
    // は `out_cap` を超えて書かないので再確保は起きず、元のポインタのまま。
    let mut v = unsafe { Vec::from_raw_parts(out, 0, out_cap) };
    let r = crate::decompress_into(s, &mut v, out_cap);
    let n = v.len();
    // 領域の所有権はホストにあるので解放しない。
    core::mem::forget(v);

    match r {
        Ok(()) if n <= i32::MAX as usize => n as i32,
        Ok(()) => -(crate::Error::LimitExceeded as i32),
        Err(e) => -(e as i32),
    }
}
