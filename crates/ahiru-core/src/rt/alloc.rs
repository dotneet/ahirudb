//! The global allocator for WASM.
//!
//! A size-class-segregated free list plus a bump. Using it instead of dlmalloc
//! (about 10 KB) keeps this to around 1 KB. wasm is assumed single-threaded, so there is no lock.
//!
//! Design decisions:
//! - 16 B through 32 KB are rounded into 12 size classes, each with its own free list.
//!   No header is needed, since the class can be recovered from the `Layout` passed to `dealloc`.
//! - Larger blocks are rounded to a 16 B boundary and served best-fit from an
//!   **address-ordered** free list. Freeing coalesces a block with the neighbours
//!   it physically touches, and a block that ends at the bump pointer is handed
//!   back to the bump region so it can serve any size class again.
//!   Without both of those, a workload that repeats the same query grows linear
//!   memory without bound: a LIFO first-fit list carves each new request out of
//!   the previous peak block and never puts the pieces back together.
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
/// The smallest split remainder that can still hold a `LargeNode` and stay on the
/// free list. Every large size is a multiple of 16, so a remainder is either 0 or
/// at least this much.
const LARGE_MIN: usize = 16;

#[repr(C)]
struct FreeNode {
    next: *mut FreeNode,
}

#[repr(C)]
struct LargeNode {
    next: *mut LargeNode,
    size: usize,
}

const _: () = assert!(core::mem::size_of::<LargeNode>() <= LARGE_MIN);

struct Heap {
    small: [*mut FreeNode; NUM_CLASSES],
    /// Free blocks larger than `MAX_SMALL`, linked in ascending address order.
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

    /// Takes `need` bytes (a multiple of 16, greater than `MAX_SMALL`) out of the
    /// large free list, choosing the smallest block that fits. Returns null when
    /// nothing fits.
    ///
    /// Best fit rather than first fit: with an address-ordered list, first fit
    /// keeps shaving the lowest -- and typically largest, since it absorbed its
    /// neighbours -- block, so the block that served the previous peak is gone by
    /// the time the peak comes round again.
    unsafe fn take_large(&mut self, need: usize) -> *mut u8 {
        let mut best: *mut *mut LargeNode = ptr::null_mut();
        let mut best_size = usize::MAX;
        let mut link: *mut *mut LargeNode = &mut self.large;
        loop {
            let cur = unsafe { *link };
            if cur.is_null() {
                break;
            }
            let csize = unsafe { (*cur).size };
            if csize >= need && csize < best_size {
                best = link;
                best_size = csize;
                if csize == need {
                    break;
                }
            }
            link = unsafe { &mut (*cur).next };
        }
        if best.is_null() {
            return ptr::null_mut();
        }
        let block = unsafe { *best };
        let rem = best_size - need;
        if rem >= LARGE_MIN {
            // The tail keeps the block's place in the address order. Keeping a
            // remainder smaller than `MAX_SMALL` here is deliberate: it is not a
            // zombie any more, because freeing either neighbour merges it back.
            let rest = unsafe { (block as *mut u8).add(need) as *mut LargeNode };
            unsafe {
                (*rest).next = (*block).next;
                (*rest).size = rem;
                *best = rest;
            }
        } else {
            unsafe { *best = (*block).next };
        }
        block as *mut u8
    }

    /// Puts a block back on the address-ordered large free list, merging it with
    /// whichever neighbours it physically touches. A block that ends exactly at
    /// the bump pointer is returned to the bump region instead, so the top of the
    /// heap can be handed out to any size class again.
    unsafe fn put_large(&mut self, block: *mut LargeNode, size: usize) {
        let addr = block as usize;
        // Walk to the insertion point, remembering the slot that points at the
        // preceding node so a backward merge can unlink through it.
        let mut prev_link: *mut *mut LargeNode = ptr::null_mut();
        let mut link: *mut *mut LargeNode = &mut self.large;
        loop {
            let cur = unsafe { *link };
            if cur.is_null() || cur as usize >= addr {
                break;
            }
            prev_link = link;
            link = unsafe { &mut (*cur).next };
        }
        let next = unsafe { *link };
        unsafe {
            (*block).next = next;
            (*block).size = size;
            *link = block;
            // Merge forward.
            if !next.is_null() && addr + size == next as usize {
                (*block).size += (*next).size;
                (*block).next = (*next).next;
            }
        }
        // Merge backward. `head`/`owner` then describe the surviving block.
        let mut head = block;
        let mut owner = link;
        if !prev_link.is_null() {
            let prev = unsafe { *prev_link };
            if unsafe { prev as usize + (*prev).size } == addr {
                unsafe {
                    (*prev).size += (*block).size;
                    (*prev).next = (*block).next;
                }
                head = prev;
                owner = prev_link;
            }
        }
        unsafe {
            if head as usize + (*head).size == self.bump {
                self.bump = head as usize;
                *owner = (*head).next;
            }
        }
    }

    unsafe fn alloc_impl(&mut self, layout: Layout) -> *mut u8 {
        let size = layout.size().max(1);
        let align = layout.align();

        if align <= 16 && size <= MAX_SMALL {
            let c = class_of(size);
            let head = self.small[c];
            if !head.is_null() {
                self.small[c] = unsafe { (*head).next };
                self.allocated += class_size(c);
                return head as *mut u8;
            }
            let p = unsafe { self.carve(class_size(c)) };
            if !p.is_null() {
                self.allocated += class_size(c);
            }
            return p;
        }

        if align > 16 {
            // Unreachable in practice (align > 16 is almost unheard of for standard Rust types).
            // Take from the bump without reuse, and leak on free.
            let n = round16(size + align);
            let raw = unsafe { self.carve(n) };
            if raw.is_null() {
                return raw;
            }
            self.allocated += n;
            let aligned = (raw as usize + align - 1) & !(align - 1);
            return aligned as *mut u8;
        }

        let need = round16(size);
        let p = unsafe { self.take_large(need) };
        if !p.is_null() {
            self.allocated += need;
            return p;
        }
        let p = unsafe { self.carve(need) };
        if !p.is_null() {
            self.allocated += need;
        }
        p
    }

    unsafe fn dealloc_impl(&mut self, p: *mut u8, layout: Layout) {
        let size = layout.size().max(1);
        let align = layout.align();

        if align <= 16 && size <= MAX_SMALL {
            let c = class_of(size);
            let node = p as *mut FreeNode;
            unsafe { (*node).next = self.small[c] };
            self.small[c] = node;
            self.allocated -= class_size(c);
            return;
        }
        if align > 16 {
            return; // leak (effectively unreachable, as noted above)
        }
        let need = round16(size);
        unsafe { self.put_large(p as *mut LargeNode, need) };
        self.allocated -= need;
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
        unsafe { h.alloc_impl(layout) }
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        if p.is_null() {
            return;
        }
        let h = unsafe { &mut *HEAP.0.get() };
        unsafe { h.dealloc_impl(p, layout) }
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
    use alloc::vec;
    use alloc::vec::Vec;

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

    /// A heap laid over a plain byte buffer. `memory_grow` is unavailable off
    /// wasm, so the region is fixed up front and a request that would need to
    /// grow simply fails -- which is exactly what makes "did this reuse memory
    /// or ask for more?" observable in a test.
    struct Arena {
        _buf: Vec<u8>,
        heap: Heap,
        base: usize,
    }

    impl Arena {
        fn new(bytes: usize) -> Arena {
            let buf = vec![0u8; bytes + 16];
            let base = (buf.as_ptr() as usize + 15) & !15;
            let mut heap = empty_heap();
            heap.bump = base;
            heap.end = base + bytes;
            Arena { _buf: buf, heap, base }
        }

        fn alloc(&mut self, size: usize) -> *mut u8 {
            let layout = Layout::from_size_align(size, 8).unwrap();
            unsafe { self.heap.alloc_impl(layout) }
        }

        fn dealloc(&mut self, p: *mut u8, size: usize) {
            let layout = Layout::from_size_align(size, 8).unwrap();
            unsafe { self.heap.dealloc_impl(p, layout) };
        }

        /// How far the bump pointer has advanced, i.e. how much of the region has
        /// ever been handed out.
        fn high_water(&self) -> usize {
            self.heap.bump - self.base
        }

        /// The blocks on the large free list, as (offset from base, size).
        fn free_list(&self) -> Vec<(usize, usize)> {
            let mut out = Vec::new();
            let mut cur = self.heap.large;
            while !cur.is_null() {
                unsafe {
                    out.push((cur as usize - self.base, (*cur).size));
                    cur = (*cur).next;
                }
            }
            out
        }
    }

    const BIG: usize = 64 * 1024;

    #[test]
    fn large_blocks_coalesce_when_freed() {
        let mut a = Arena::new(1 << 20);
        let p0 = a.alloc(BIG);
        let p1 = a.alloc(BIG);
        let p2 = a.alloc(BIG);
        // Keep p2 allocated so the freed pair cannot be absorbed by the bump.
        assert!(!p0.is_null() && !p1.is_null() && !p2.is_null());
        a.dealloc(p0, BIG);
        a.dealloc(p1, BIG);
        assert_eq!(a.free_list(), vec![(0, 2 * BIG)]);
        // The merged block satisfies a request neither half could.
        let hw = a.high_water();
        let p3 = a.alloc(2 * BIG);
        assert_eq!(p3, p0);
        assert_eq!(a.high_water(), hw, "should not have taken new bump space");
    }

    #[test]
    fn freeing_the_top_block_gives_it_back_to_the_bump() {
        let mut a = Arena::new(1 << 20);
        let p0 = a.alloc(BIG);
        let p1 = a.alloc(BIG);
        a.dealloc(p1, BIG);
        assert!(a.free_list().is_empty(), "top block should return to the bump");
        assert_eq!(a.high_water(), BIG);
        // ... and is then available to a *small* request too, which the large
        // free list could never serve.
        let s = a.alloc(64);
        assert_eq!(s, p1);
        a.dealloc(s, 64);
        a.dealloc(p0, BIG);
    }

    #[test]
    fn best_fit_leaves_the_biggest_block_alone() {
        let mut a = Arena::new(1 << 20);
        // Lay out [small hole][keep][big hole][keep] so neither hole coalesces.
        let a0 = a.alloc(3 * BIG);
        let k0 = a.alloc(BIG);
        let a1 = a.alloc(8 * BIG);
        let k1 = a.alloc(BIG);
        a.dealloc(a0, 3 * BIG);
        a.dealloc(a1, 8 * BIG);
        assert_eq!(a.free_list().len(), 2);
        // A request that both holes could serve must come out of the smaller one.
        let p = a.alloc(2 * BIG);
        assert_eq!(p, a0);
        assert!(
            a.free_list().iter().any(|&(_, size)| size == 8 * BIG),
            "the large hole must still be intact: {:?}",
            a.free_list()
        );
        let _ = (k0, k1);
    }

    #[test]
    fn split_remainders_stay_reusable() {
        let mut a = Arena::new(1 << 20);
        let p0 = a.alloc(2 * BIG);
        let keep = a.alloc(BIG);
        a.dealloc(p0, 2 * BIG);
        // Leaves a 32 KiB tail, which is below MAX_SMALL and used to be dropped.
        let p1 = a.alloc(2 * BIG - 32 * 1024);
        assert_eq!(p1, p0);
        assert_eq!(a.free_list(), vec![(2 * BIG - 32 * 1024, 32 * 1024)]);
        // Freeing the head merges the remainder back into one whole block.
        a.dealloc(p1, 2 * BIG - 32 * 1024);
        assert_eq!(a.free_list(), vec![(0, 2 * BIG)]);
        a.dealloc(keep, BIG);
    }

    /// The regression this allocator shape exists for: repeating a workload with
    /// varying sizes must reach a steady state instead of asking the region for
    /// more every round. The bound is the one the bug report asks for -- the
    /// high-water mark stays within roughly twice the largest single round's
    /// demand.
    #[test]
    fn repeated_workload_reaches_a_steady_state() {
        let mut a = Arena::new(16 << 20);
        // Something long-lived at the bottom, so the rounds above it cannot all
        // be reclaimed by simply rewinding the bump pointer.
        let pinned = a.alloc(BIG);
        assert!(!pinned.is_null());

        let mut seed: u64 = 0x9e37_79b9;
        let mut rnd = |lo: usize, hi: usize| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + (seed >> 33) as usize % (hi - lo)
        };

        let mut peak_demand = 0usize;
        for round in 0..40 {
            let mut live = Vec::new();
            let mut demand = BIG;
            for _ in 0..6 {
                let s = rnd(33 * 1024, 400 * 1024);
                let p = a.alloc(s);
                assert!(!p.is_null(), "round {round} ran out of arena");
                live.push((p, s));
                demand += round16(s);
            }
            peak_demand = peak_demand.max(demand);
            // Free in a rotated order, so the holes are not a plain LIFO unwind.
            let shift = round % live.len();
            live.rotate_left(shift);
            for (p, s) in live.drain(..) {
                a.dealloc(p, s);
            }
            assert!(
                a.high_water() <= 2 * peak_demand,
                "round {round}: high water {} exceeds 2x peak demand {peak_demand}",
                a.high_water()
            );
        }
        a.dealloc(pinned, BIG);
        assert_eq!(a.heap.allocated, 0);
        // With everything freed the region is whole again, either as one free
        // block or entirely rewound into the bump.
        assert!(a.free_list().len() <= 1, "fragmented: {:?}", a.free_list());
    }

    #[test]
    fn small_classes_reuse_freed_blocks() {
        let mut a = Arena::new(1 << 20);
        let p = a.alloc(100);
        a.dealloc(p, 100);
        let q = a.alloc(120); // same class (128)
        assert_eq!(p, q);
        assert_eq!(a.high_water(), 128);
    }
}
