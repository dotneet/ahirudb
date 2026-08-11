//! ランタイム基盤: アロケータ、パニックハンドラ、小さなコンテナ。

pub mod alloc;
pub mod hash;

/// wasm + no_std ビルドでのみグローバルアロケータを差し替える。
#[cfg(all(target_arch = "wasm32", not(feature = "std")))]
#[global_allocator]
static ALLOC: alloc::AhiruAlloc = alloc::AhiruAlloc;

/// no_std ビルドのパニックハンドラ。
///
/// `panic = "abort"` としているため巻き戻しは発生しない。メッセージを
/// 組み立てると `core::fmt` がリンクされてしまうので、何も見ずに trap する。
/// エラーは `Result` で返す設計なので、ここに来るのはバグかメモリ枯渇のみ。
#[cfg(all(target_arch = "wasm32", not(feature = "std"), not(test)))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}
