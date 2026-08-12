//! Thrift Compact Protocol writer (`export-parquet` feature).
//!
//! The mirror image of `crate::parquet::thrift`, which reads. Like the
//! reader, this is not a general Thrift runtime: it exposes only the
//! handful of operations the Parquet writer needs (a fixed set of struct,
//! list and scalar shapes), because a generated runtime would cost far
//! more than the whole feature is worth against the 1 MiB budget
//! (DESIGN.md §5).
//!
//! Everything written here comes from our own metadata, never from
//! untrusted input, so there is nothing to validate. The writer still
//! avoids indexing and `unwrap` so the `no_std` build pulls in no panic
//! paths.

use crate::prelude::*;

/// Thrift Compact type tags. Only the ones the Parquet footer and page
/// headers actually use.
pub mod ttype {
    pub const STOP: u8 = 0;
    pub const BOOL_TRUE: u8 = 1;
    pub const BOOL_FALSE: u8 = 2;
    pub const I8: u8 = 3;
    pub const I32: u8 = 5;
    pub const I64: u8 = 6;
    pub const BINARY: u8 = 8;
    pub const LIST: u8 = 9;
    pub const STRUCT: u8 = 12;
}

/// Struct nesting levels the writer tracks. The deepest thing we write is
/// `FileMetaData` > `RowGroup` > `ColumnChunk` > `ColumnMetaData` >
/// `LogicalType`'s inner union member, so this leaves plenty of headroom.
const MAX_DEPTH: usize = 8;

pub struct Writer {
    out: Vec<u8>,
    /// Last field id written at each nesting level. Compact protocol
    /// encodes a field id as a delta from the previous field *of the same
    /// struct*, so entering a struct has to push a fresh context.
    id_stack: [i16; MAX_DEPTH],
    depth: usize,
}

impl Writer {
    pub fn new() -> Self {
        Writer { out: Vec::new(), id_stack: [0; MAX_DEPTH], depth: 0 }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.out
    }

    // --- primitives ---------------------------------------------------------

    fn uvarint(&mut self, mut v: u64) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                self.out.push(b);
                return;
            }
            self.out.push(b | 0x80);
        }
    }

    fn zigzag(&mut self, v: i64) {
        self.uvarint(((v << 1) ^ (v >> 63)) as u64);
    }

    /// Current slot in `id_stack`. Clamped so a (impossible, given the
    /// fixed shapes above) over-deep nesting degrades into reusing the
    /// last slot instead of panicking.
    #[inline]
    fn slot(&self) -> usize {
        if self.depth < MAX_DEPTH {
            self.depth
        } else {
            MAX_DEPTH - 1
        }
    }

    fn field_header(&mut self, id: i16, ty: u8) {
        let slot = self.slot();
        let delta = i32::from(id) - i32::from(self.id_stack[slot]);
        if (1..=15).contains(&delta) {
            self.out.push(((delta as u8) << 4) | ty);
        } else {
            // Out of delta range: type tag with a zero delta nibble,
            // followed by the absolute id as a zigzag varint.
            self.out.push(ty);
            self.zigzag(i64::from(id));
        }
        self.id_stack[slot] = id;
    }

    // --- structs ------------------------------------------------------------

    /// Enter a struct. Every call must be paired with `end_struct`.
    pub fn begin_struct(&mut self) {
        self.depth += 1;
        let slot = self.slot();
        self.id_stack[slot] = 0;
    }

    pub fn end_struct(&mut self) {
        self.out.push(ttype::STOP);
        self.depth = self.depth.saturating_sub(1);
    }

    /// Write a field header for a nested struct and enter it.
    pub fn begin_field_struct(&mut self, id: i16) {
        self.field_header(id, ttype::STRUCT);
        self.begin_struct();
    }

    /// A struct-valued field with no members. `LogicalType`'s union members
    /// that carry no parameters (`STRING`, `DATE`, `JSON`, `UUID`, …) are
    /// all written this way.
    pub fn field_empty_struct(&mut self, id: i16) {
        self.begin_field_struct(id);
        self.end_struct();
    }

    // --- scalar fields ------------------------------------------------------

    pub fn field_bool(&mut self, id: i16, v: bool) {
        // A struct field's boolean value lives in the type nibble itself;
        // no payload follows.
        self.field_header(id, if v { ttype::BOOL_TRUE } else { ttype::BOOL_FALSE });
    }

    /// A Thrift `byte` field (`LogicalType`'s `IntType.bitWidth`). Unlike
    /// I16/I32/I64 this is *not* zigzag-varint encoded: the value is a
    /// single raw byte.
    pub fn field_i8(&mut self, id: i16, v: i8) {
        self.field_header(id, ttype::I8);
        self.out.push(v as u8);
    }

    pub fn field_i32(&mut self, id: i16, v: i32) {
        self.field_header(id, ttype::I32);
        self.zigzag(i64::from(v));
    }

    pub fn field_i64(&mut self, id: i16, v: i64) {
        self.field_header(id, ttype::I64);
        self.zigzag(v);
    }

    pub fn field_binary(&mut self, id: i16, v: &[u8]) {
        self.field_header(id, ttype::BINARY);
        self.uvarint(v.len() as u64);
        self.out.extend_from_slice(v);
    }

    // --- lists --------------------------------------------------------------

    /// Write a list header. The caller then writes exactly `n` elements of
    /// `etype` (structs via `begin_struct`/`end_struct`, scalars via the
    /// `elem_*` helpers).
    pub fn begin_field_list(&mut self, id: i16, etype: u8, n: usize) {
        self.field_header(id, ttype::LIST);
        if n < 15 {
            self.out.push(((n as u8) << 4) | etype);
        } else {
            self.out.push(0xf0 | etype);
            self.uvarint(n as u64);
        }
    }

    pub fn elem_i32(&mut self, v: i32) {
        self.zigzag(i64::from(v));
    }

    pub fn elem_binary(&mut self, v: &[u8]) {
        self.uvarint(v.len() as u64);
        self.out.extend_from_slice(v);
    }
}

impl Default for Writer {
    fn default() -> Self {
        Writer::new()
    }
}
