//! Runtime foundation: the allocator, the panic handler, and small containers.

pub mod alloc;
pub mod hash;

/// The global allocator is only swapped in for wasm + no_std builds.
#[cfg(all(target_arch = "wasm32", not(feature = "std")))]
#[global_allocator]
static ALLOC: alloc::AhiruAlloc = alloc::AhiruAlloc;

/// The panic handler for no_std builds.
///
/// With `panic = "abort"` there is no unwinding. Assembling a message would link
/// `core::fmt`, so this traps without looking at anything. Every error is designed
/// to come back as a `Result`, so reaching here means a bug or memory exhaustion.
#[cfg(all(target_arch = "wasm32", not(feature = "std"), not(test)))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}
