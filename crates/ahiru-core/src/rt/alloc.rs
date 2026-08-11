//! WASM 用グローバルアロケータ。
//!
//! サイズクラス分離フリーリスト + バンプ。dlmalloc (約 10 KB) の代わりに
//! 使うことで 1 KB 程度に収まる。wasm は単一スレッド前提なのでロックを持たない。
//!
//! 設計方針:
//! - 16 B から 32 KB までを 12 個のサイズクラスに丸め、クラスごとのフリーリストで回す。
//!   クラスは `dealloc` に渡される `Layout` から復元できるのでヘッダを持たない。
//! - それより大きいブロックは 16 B 境界に丸めて単方向リストで first-fit。
//! - 領域は `memory.grow` で 1 MiB ずつ確保する。解放は行わない（wasm の
//!   線形メモリは縮められないため）。

use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;
use core::ptr;

const PAGE: usize = 65536;
const MIN_SHIFT: usize = 4;
const NUM_CLASSES: usize = 12;
/// 16 << 11 = 32768
const MAX_SMALL: usize = 16 << (NUM_CLASSES - 1);
/// 一度の `memory.grow` で確保する最小ページ数 (1 MiB)。
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
    /// 統計用。`ahiru_heap_used` で公開する。
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
    /// バンプ領域から `n` バイト (16 の倍数) を切り出す。
    unsafe fn carve(&mut self, n: usize) -> *mut u8 {
        if self.bump + n > self.end && !unsafe { self.grow(n) } {
            return ptr::null_mut();
        }
        let p = self.bump;
        self.bump += n;
        p as *mut u8
    }

    unsafe fn grow(&mut self, need: usize) -> bool {
        let pages = need.div_ceil(PAGE).max(GROW_PAGES);
        let prev = memory_grow(pages);
        if prev == usize::MAX {
            return false;
        }
        let start = prev * PAGE;
        if start == self.end {
            // 直前の領域と連続しているので末尾を伸ばすだけでよい。
            self.end = start + pages * PAGE;
        } else {
            // 連続していない場合は古い残余を捨てる。自分以外が memory.grow を
            // 呼ばない限りここには来ない。
            self.bump = start;
            self.end = start + pages * PAGE;
        }
        true
    }
}

struct HeapCell(UnsafeCell<Heap>);
// wasm32 は単一スレッドで動かす前提。
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
            // 実際には到達しない (Rust の標準的な型で align > 16 はほぼ無い)。
            // 再利用せずバンプから取り、解放時はリークさせる。
            let n = round16(size + align);
            let raw = unsafe { h.carve(n) };
            if raw.is_null() {
                return raw;
            }
            h.allocated += n;
            let aligned = (raw as usize + align - 1) & !(align - 1);
            return aligned as *mut u8;
        }

        // large: フリーリストを first-fit で探す。
        let need = round16(size);
        let mut prev: *mut *mut LargeNode = &mut h.large;
        let mut cur = h.large;
        while !cur.is_null() {
            let csize = unsafe { (*cur).size };
            if csize >= need {
                unsafe { *prev = (*cur).next };
                h.allocated += csize;
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
            return; // リーク (上記の通り実質到達しない)
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

        // 同じサイズクラスに収まるなら何もしない。Vec の伸長で頻繁に効く。
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

/// wasm 以外 (アロケータの単体テスト用) では成長できない。
#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn memory_grow(_pages: usize) -> usize {
    usize::MAX
}

/// 現在ヒープが保持しているバイト数。メモリ上限チェックに使う。
pub fn heap_used() -> usize {
    unsafe { (*HEAP.0.get()).allocated }
}

/// `memory.grow` で確保済みの総バイト数。
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
