//! The global allocator for WASM.
//!
//! A size-class-segregated free list plus a bump. Using it instead of dlmalloc
//! (about 10 KB) keeps this to around 1 KB. wasm is assumed single-threaded, so there is no lock.
//!
//! Design decisions:
//! - 16 B through 32 KB are rounded into 12 size classes, each with its own free list.
//!   No header is needed, since the class can be recovered from the `Layout` passed to `dealloc`.
//! - Larger blocks are rounded to a 16 B boundary and served first-fit from a singly linked list.
//! - Regions are reserved 1 MiB at a time via `memory.grow`, and never released
//!   (wasm linear memory cannot shrink).

use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;
use core::ptr;

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
    /// For statistics. Exposed through `ahiru_heap_used`.
    allocated: usize,
}

const fn empty_heap() -> Heap {
    Heap {
        small: [ptr::null_mut(); NUM_CLASSES],
        large: ptr::null_mut(),
        bump: 0,
        end: 0,
        allocated: 0,
    }
}

#[inline]
fn class_of(size: usize) -> usize {
    if size <= 16 {
        0
    } else {
        (usize::BITS - (size - 1).leading_zeros()) as usize - MIN_SHIFT
    }
}

#[inline]
fn class_size(c: usize) -> usize {
    16 << c
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
            // Contiguous with the previous region, so just extend the end.
            self.end = new_end;
        } else {
            // Otherwise, discard the old remainder. This is unreachable unless
            // something other than us calls memory.grow.
            self.bump = start;
            self.end = new_end;
        }
        true
    }
}

struct HeapCell(UnsafeCell<Heap>);
// wasm32 is assumed to run single-threaded.
unsafe impl Sync for HeapCell {}

static HEAP: HeapCell = HeapCell(UnsafeCell::new(empty_heap()));

pub struct AhiruAlloc;

unsafe impl GlobalAlloc for AhiruAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let h = unsafe { &mut *HEAP.0.get() };
        let size = layout.size().max(1);
        let align = layout.align();

        if align <= 16 && size <= MAX_SMALL {
            let c = class_of(size);
            let head = h.small[c];
            if !head.is_null() {
                h.small[c] = unsafe { (*head).next };
                h.allocated += class_size(c);
                return head as *mut u8;
            }
            let p = unsafe { h.carve(class_size(c)) };
            if !p.is_null() {
                h.allocated += class_size(c);
            }
            return p;
        }

        if align > 16 {
            // Unreachable in practice (align > 16 is almost unheard of for standard Rust types).
            // Take from the bump without reuse, and leak on free.
            let n = round16(size + align);
            let raw = unsafe { h.carve(n) };
            if raw.is_null() {
                return raw;
            }
            h.allocated += n;
            let aligned = (raw as usize + align - 1) & !(align - 1);
            return aligned as *mut u8;
        }

        // large: first-fit search of the free list.
        let need = round16(size);
        let mut prev: *mut *mut LargeNode = &mut h.large;
        let mut cur = h.large;
        while !cur.is_null() {
            let csize = unsafe { (*cur).size };
            if csize >= need {
                unsafe { *prev = (*cur).next };
                // Split the tail back onto the large list so a 64 KiB free
                // block reused for 40 KiB does not lose 24 KiB (and so
                // `heap_used` matches what `dealloc` will subtract).
                let rem = csize - need;
                if rem > 0 {
                    let rest = unsafe { (cur as *mut u8).add(need) as *mut LargeNode };
                    unsafe {
                        (*rest).next = h.large;
                        (*rest).size = rem;
                    }
                    h.large = rest;
                }
                h.allocated += need;
                return cur as *mut u8;
            }
            prev = unsafe { &mut (*cur).next };
            cur = unsafe { (*cur).next };
        }
        let p = unsafe { h.carve(need) };
        if !p.is_null() {
            h.allocated += need;
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        if p.is_null() {
            return;
        }
        let h = unsafe { &mut *HEAP.0.get() };
        let size = layout.size().max(1);
        let align = layout.align();

        if align <= 16 && size <= MAX_SMALL {
            let c = class_of(size);
            let node = p as *mut FreeNode;
            unsafe { (*node).next = h.small[c] };
            h.small[c] = node;
            h.allocated -= class_size(c);
            return;
        }
        if align > 16 {
            return; // leak (effectively unreachable, as noted above)
        }
        let need = round16(size);
        let node = p as *mut LargeNode;
        unsafe {
            (*node).next = h.large;
            (*node).size = need;
        }
        h.large = node;
        h.allocated -= need;
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
                ptr::copy_nonoverlapping(p, np, core::cmp::min(old, new_size));
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

/// Cannot grow off wasm (for unit-testing the allocator).
#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn memory_grow(_pages: usize) -> usize {
    usize::MAX
}

/// How many bytes the heap currently holds. Used by the memory-limit check.
pub fn heap_used() -> usize {
    unsafe { (*HEAP.0.get()).allocated }
}

/// The total bytes reserved so far via `memory.grow`.
pub fn heap_reserved() -> usize {
    let h = unsafe { &*HEAP.0.get() };
    h.end.saturating_sub(h.bump) + h.allocated
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
        assert_eq!(class_of(33), 2);
        assert_eq!(class_of(MAX_SMALL), NUM_CLASSES - 1);
        for c in 0..NUM_CLASSES {
            assert_eq!(class_of(class_size(c)), c);
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
