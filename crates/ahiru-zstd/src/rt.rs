//! wasm runtime foundation: the global allocator and the panic handler.
//!
//! The point of a side module is not to bloat the core, so these are carried here
//! rather than depending on `ahiru-core` (which would pull the whole engine into this module).
//!
//! The allocator is a size-class-segregated free list plus a bump. Using it
//! instead of dlmalloc (about 10 KB) keeps this to around 1 KB. wasm is assumed
//! single-threaded, so there is no lock. Freed pages are not returned to the OS
//! (wasm linear memory never shrinks).

use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;
use core::ptr;

/// The global allocator is only swapped in for wasm + no_std builds.
/// It is defined only under `standalone` (when this crate alone is the top level
/// of a wasm module), because when it is linked into `ahiru-core` as a library it
/// would collide with `ahiru-core`'s own allocator.
#[cfg(all(target_arch = "wasm32", not(feature = "std"), feature = "standalone"))]
#[global_allocator]
static ALLOC: ZstdAlloc = ZstdAlloc;

/// The panic handler for no_std builds. Only under `standalone` (for the same
/// reason as the allocator above).
///
/// With `panic = "abort"` there is no unwinding. Assembling a message would link
/// `core::fmt`, so this traps without looking at anything. Every error is designed
/// to come back as a `Result`, so reaching here means a bug or memory exhaustion.
#[cfg(all(target_arch = "wasm32", not(feature = "std"), not(test), feature = "standalone"))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

const PAGE: usize = 65536;
const MIN_SHIFT: usize = 4;
const NUM_CLASSES: usize = 12;
/// 16 << 11 = 32768
const MAX_SMALL: usize = 16 << (NUM_CLASSES - 1);
/// The minimum number of pages reserved by one `memory.grow` (1 MiB).
const GROW_PAGES: usize = 16;

#[repr(C)]
struct FreeNode {
    next: *mut FreeNode,
}

#[repr(C)]
struct LargeNode {
    next: *mut LargeNode,
    size: usize,
}

struct Heap {
    small: [*mut FreeNode; NUM_CLASSES],
    large: *mut LargeNode,
    bump: usize,
    end: usize,
}

struct HeapCell(UnsafeCell<Heap>);
// wasm32 is assumed to run single-threaded.
unsafe impl Sync for HeapCell {}

static HEAP: HeapCell = HeapCell(UnsafeCell::new(Heap {
    small: [ptr::null_mut(); NUM_CLASSES],
    large: ptr::null_mut(),
    bump: 0,
    end: 0,
}));

#[inline]
fn class_of(size: usize) -> usize {
    if size <= 16 {
        0
    } else {
        (usize::BITS - (size - 1).leading_zeros()) as usize - MIN_SHIFT
    }
}

#[inline]
fn round16(n: usize) -> usize {
    (n + 15) & !15
}

impl Heap {
    /// Carves `n` bytes (a multiple of 16) out of the bump region.
    unsafe fn carve(&mut self, n: usize) -> *mut u8 {
        loop {
            match self.bump.checked_add(n) {
                Some(end) if end <= self.end => {
                    let p = self.bump;
                    self.bump = end;
                    return p as *mut u8;
                }
                _ => {
                    if !unsafe { self.grow(n) } {
                        return ptr::null_mut();
                    }
                }
            }
        }
    }

    unsafe fn grow(&mut self, need: usize) -> bool {
        let pages = need.div_ceil(PAGE).max(GROW_PAGES);
        let Some(bytes) = pages.checked_mul(PAGE) else {
            return false;
        };
        let prev = memory_grow(pages);
        if prev == usize::MAX {
            return false;
        }
        let Some(start) = prev.checked_mul(PAGE) else {
            return false;
        };
        let Some(new_end) = start.checked_add(bytes) else {
            return false;
        };
        if start == self.end {
            self.end = new_end;
        } else {
            self.bump = start;
            self.end = new_end;
        }
        true
    }
}

pub struct ZstdAlloc;

unsafe impl GlobalAlloc for ZstdAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let h = unsafe { &mut *HEAP.0.get() };
        let size = layout.size().max(1);
        let align = layout.align();

        if align <= 16 && size <= MAX_SMALL {
            // No header is needed: the class can be recovered from the Layout passed to `dealloc`.
            let c = class_of(size);
            let head = h.small[c];
            if !head.is_null() {
                h.small[c] = unsafe { (*head).next };
                return head as *mut u8;
            }
            return unsafe { h.carve(16 << c) };
        }
        if align > 16 {
            // Unreachable in practice (every type this crate handles has align <= 8).
            // Take from the bump without reuse, and leak on free.
            let raw = unsafe { h.carve(round16(size + align)) };
            if raw.is_null() {
                return raw;
            }
            return ((raw as usize + align - 1) & !(align - 1)) as *mut u8;
        }

        // large: first-fit search of the free list.
        let need = round16(size);
        let mut prev: *mut *mut LargeNode = &mut h.large;
        let mut cur = h.large;
        while !cur.is_null() {
            if unsafe { (*cur).size } >= need {
                unsafe { *prev = (*cur).next };
                let rem = unsafe { (*cur).size } - need;
                if rem > 0 {
                    let rest = unsafe { (cur as *mut u8).add(need) as *mut LargeNode };
                    unsafe {
                        (*rest).next = h.large;
                        (*rest).size = rem;
                    }
                    h.large = rest;
                }
                return cur as *mut u8;
            }
            prev = unsafe { &mut (*cur).next };
            cur = unsafe { (*cur).next };
        }
        unsafe { h.carve(need) }
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        if p.is_null() {
            return;
        }
        let h = unsafe { &mut *HEAP.0.get() };
        let size = layout.size().max(1);
        let align = layout.align();

        if align <= 16 && size <= MAX_SMALL {
            let node = p as *mut FreeNode;
            let c = class_of(size);
            unsafe { (*node).next = h.small[c] };
            h.small[c] = node;
            return;
        }
        if align > 16 {
            return; // leak (effectively unreachable, as noted above)
        }
        let node = p as *mut LargeNode;
        unsafe {
            (*node).next = h.large;
            (*node).size = round16(size);
        }
        h.large = node;
    }

    unsafe fn realloc(&self, p: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let old = layout.size().max(1);
        let align = layout.align();

        // Nothing to do if it still fits the same size class. This pays off often as a Vec grows.
        if align <= 16
            && old <= MAX_SMALL
            && new_size <= MAX_SMALL
            && class_of(old) == class_of(new_size.max(1))
        {
            return p;
        }

        let new_layout = match Layout::from_size_align(new_size, align) {
            Ok(l) => l,
            Err(_) => return ptr::null_mut(),
        };
        let np = unsafe { self.alloc(new_layout) };
        if !np.is_null() {
            unsafe {
                ptr::copy_nonoverlapping(p, np, if old < new_size { old } else { new_size });
                self.dealloc(p, layout);
            }
        }
        np
    }
}

#[cfg(target_arch = "wasm32")]
#[inline]
fn memory_grow(pages: usize) -> usize {
    core::arch::wasm32::memory_grow(0, pages)
}

/// Cannot grow off wasm (native builds use std's allocator).
#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn memory_grow(_pages: usize) -> usize {
    usize::MAX
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_boundaries() {
        assert_eq!(class_of(1), 0);
        assert_eq!(class_of(16), 0);
        assert_eq!(class_of(17), 1);
        assert_eq!(class_of(32), 1);
        assert_eq!(class_of(MAX_SMALL), NUM_CLASSES - 1);
        for c in 0..NUM_CLASSES {
            assert_eq!(class_of(16 << c), c);
        }
    }

    #[test]
    fn round16_works() {
        assert_eq!(round16(0), 0);
        assert_eq!(round16(1), 16);
        assert_eq!(round16(16), 16);
        assert_eq!(round16(17), 32);
    }
}
