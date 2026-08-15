//! Thrift Compact Protocol reader (read-only).
//!
//! Parquet's footer and page headers are encoded in this format. There is no
//! general-purpose Thrift runtime here; this only provides the operations
//! that `meta.rs`'s hand-written decoder needs.
//!
//! Input comes from the network and cannot be trusted. Every read is
//! bounds-checked, and corruption always results in `Err` (never panics).

use crate::prelude::*;

/// Thrift Compact type tags.
pub mod ttype {
    pub const STOP: u8 = 0;
    pub const BOOL_TRUE: u8 = 1;
    pub const BOOL_FALSE: u8 = 2;
    pub const I8: u8 = 3;
    pub const I16: u8 = 4;
    pub const I32: u8 = 5;
    pub const I64: u8 = 6;
    pub const DOUBLE: u8 = 7;
    pub const BINARY: u8 = 8;
    pub const LIST: u8 = 9;
    pub const SET: u8 = 10;
    pub const MAP: u8 = 11;
    pub const STRUCT: u8 = 12;
}

/// Upper bound on struct nesting depth. Prevents stack exhaustion from cyclic or excessively deep input.
pub const MAX_DEPTH: usize = 32;

pub struct Thrift<'a> {
    buf: &'a [u8],
    pos: usize,
    /// The previous field ID at each nesting level (for the compact protocol's delta encoding).
    id_stack: [i16; MAX_DEPTH],
    depth: usize,
}

impl<'a> Thrift<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Thrift { buf, pos: 0, id_stack: [0; MAX_DEPTH], depth: 0 }
    }

    /// The current read position. Used for reporting error locations and computing bytes consumed.
    #[inline]
    pub fn pos(&self) -> usize {
        self.pos
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// Enter a struct. Pushes one level onto the field-ID delta-encoding context.
    pub fn enter(&mut self) -> Result<()> {
        ensure!(self.depth + 1 < MAX_DEPTH, NestingTooDeep, self.pos);
        self.depth += 1;
        self.id_stack[self.depth] = 0;
        Ok(())
    }

    /// Leave a struct. Call this after `read_field_begin` returns STOP.
    pub fn leave(&mut self) {
        if self.depth > 0 {
            self.depth -= 1;
        }
    }

    // --- Primitives ----------------------------------------------------------

    #[inline]
    fn byte(&mut self) -> Result<u8> {
        ensure!(self.pos < self.buf.len(), UnexpectedEof, self.pos);
        let b = self.buf[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(n <= self.buf.len() - self.pos, UnexpectedEof, self.pos);
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    /// LEB128 variable-length integer. Exceeding 10 bytes is treated as corruption.
    fn uvarint(&mut self) -> Result<u64> {
        let mut result: u64 = 0;
        let mut shift = 0u32;
        loop {
            ensure!(shift < 64, BadVarint, self.pos);
            let b = self.byte()?;
            result |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
        }
    }

    #[inline]
    fn zigzag(&mut self) -> Result<i64> {
        let u = self.uvarint()?;
        Ok(((u >> 1) as i64) ^ -((u & 1) as i64))
    }

    // --- Fields ----------------------------------------------------------------

    /// Read the next field header. `None` if it's STOP.
    /// The return value is `(type tag, field ID)`.
    pub fn read_field_begin(&mut self) -> Result<Option<(u8, i16)>> {
        let b = self.byte()?;
        if b == ttype::STOP {
            return Ok(None);
        }
        let ftype = b & 0x0f;
        let delta = (b >> 4) as i16;
        let id = if delta == 0 {
            // Followed by an absolute ID (not a delta) as a zigzag varint.
            let v = self.zigzag()?;
            ensure!(v >= i16::MIN as i64 && v <= i16::MAX as i64, BadThrift, self.pos);
            v as i16
        } else {
            self.id_stack[self.depth].wrapping_add(delta)
        };
        self.id_stack[self.depth] = id;
        Ok(Some((ftype, id)))
    }

    /// BOOL. When it appears as a field, the value is embedded in the type
    /// tag. When it appears as a list element, read 1 byte.
    pub fn read_bool(&mut self, ftype: u8) -> Result<bool> {
        match ftype {
            ttype::BOOL_TRUE => Ok(true),
            ttype::BOOL_FALSE => Ok(false),
            _ => Ok(self.byte()? != 0),
        }
    }

    pub fn read_i8(&mut self, _ftype: u8) -> Result<i8> {
        Ok(self.byte()? as i8)
    }

    /// BOOL as a list/set element. Unlike BOOL as a struct field, the
    /// element's type header (the `etype` returned by `read_list_begin`) is a
    /// fixed `BOOL_TRUE` value shared across all elements and carries no
    /// truth value. Each element is instead written as an independent 1 byte,
    /// and the truth value is determined by whether it equals `BOOL_TRUE`
    /// (1) (per the Thrift Compact Protocol spec, the non-field code path of
    /// `TCompactProtocol.writeBoolean`). This is kept separate from
    /// `read_bool(ftype)` because using that for list elements would always
    /// yield "true, consuming 0 bytes".
    pub fn read_bool_elem(&mut self) -> Result<bool> {
        Ok(self.byte()? == ttype::BOOL_TRUE)
    }

    pub fn read_i16(&mut self, _ftype: u8) -> Result<i16> {
        let v = self.zigzag()?;
        ensure!(v >= i16::MIN as i64 && v <= i16::MAX as i64, BadThrift, self.pos);
        Ok(v as i16)
    }

    pub fn read_i32(&mut self, _ftype: u8) -> Result<i32> {
        let v = self.zigzag()?;
        ensure!(v >= i32::MIN as i64 && v <= i32::MAX as i64, BadThrift, self.pos);
        Ok(v as i32)
    }

    pub fn read_i64(&mut self, _ftype: u8) -> Result<i64> {
        self.zigzag()
    }

    pub fn read_double(&mut self, _ftype: u8) -> Result<f64> {
        let b = self.take(8)?;
        Ok(f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    /// Variable-length byte string. Returns a reference straight into the buffer (no copy).
    pub fn read_binary(&mut self, _ftype: u8) -> Result<&'a [u8]> {
        let len = self.uvarint()? as usize;
        ensure!(len <= self.buf.len() - self.pos, UnexpectedEof, self.pos);
        self.take(len)
    }

    /// UTF-8 string. Invalid byte sequences are lossily converted rather than
    /// replaced outright, so that invalid UTF-8 in a schema name doesn't fail the query.
    pub fn read_string(&mut self, ftype: u8) -> Result<String> {
        let b = self.read_binary(ftype)?;
        Ok(match core::str::from_utf8(b) {
            Ok(s) => s.to_owned(),
            Err(_) => {
                let mut s = String::with_capacity(b.len());
                for &c in b {
                    if c.is_ascii() {
                        s.push(c as char);
                    } else {
                        s.push('\u{FFFD}');
                    }
                }
                s
            }
        })
    }

    /// List/set header. Returns `(element type, element count)`.
    /// An element count exceeding `max` is treated as corruption or an attack and rejected.
    pub fn read_list_begin(&mut self, max: usize) -> Result<(u8, usize)> {
        let b = self.byte()?;
        let etype = b & 0x0f;
        let mut size = (b >> 4) as usize;
        if size == 0x0f {
            size = self.uvarint()? as usize;
        }
        ensure!(size <= max, LimitExceeded, self.pos);
        // Each element needs at least 1 byte. Corruption if the declared size exceeds the remaining buffer.
        ensure!(size <= self.remaining(), UnexpectedEof, self.pos);
        Ok((etype, size))
    }

    /// Skip an unknown field.
    pub fn skip(&mut self, ftype: u8) -> Result<()> {
        self.skip_inner(ftype, 0)
    }

    fn skip_inner(&mut self, ftype: u8, depth: usize) -> Result<()> {
        ensure!(depth < MAX_DEPTH, NestingTooDeep, self.pos);
        match ftype {
            ttype::BOOL_TRUE | ttype::BOOL_FALSE => {}
            ttype::I8 => {
                self.byte()?;
            }
            ttype::I16 | ttype::I32 | ttype::I64 => {
                self.zigzag()?;
            }
            ttype::DOUBLE => {
                self.take(8)?;
            }
            ttype::BINARY => {
                self.read_binary(ftype)?;
            }
            ttype::LIST | ttype::SET => {
                let (et, n) = self.read_list_begin(MAX_SKIP_LIST)?;
                // Unlike a struct field, a bool as a list/set element isn't
                // packed into the header nibble -- it's laid out as 1 byte per
                // element (same convention as `read_bool_elem`). The BOOL arm of
                // `skip_inner` assumes "the value is embedded in the struct
                // field's header, so there's no extra byte," so passing it
                // straight through here would silently get the number of bytes
                // to skip wrong, throwing off the rest of the parse.
                if et == ttype::BOOL_TRUE || et == ttype::BOOL_FALSE {
                    for _ in 0..n {
                        self.byte()?;
                    }
                } else {
                    for _ in 0..n {
                        self.skip_inner(et, depth + 1)?;
                    }
                }
            }
            ttype::MAP => {
                let n = self.uvarint()? as usize;
                if n > 0 {
                    ensure!(n <= MAX_SKIP_LIST, LimitExceeded, self.pos);
                    let kv = self.byte()?;
                    let (kt, vt) = (kv >> 4, kv & 0x0f);
                    // Same reason as list/set: a bool as a map element (key or
                    // value, either way) isn't embedded into the header like a struct
                    // field is -- it's laid out as 1 byte per element. Passing it
                    // straight to `skip_inner`'s BOOL arm would treat it as 0 bytes
                    // skipped, throwing off the subsequent parse position.
                    let skip_one = |t: &mut Self, ty: u8, depth: usize| -> Result<()> {
                        if ty == ttype::BOOL_TRUE || ty == ttype::BOOL_FALSE {
                            t.byte()?;
                            Ok(())
                        } else {
                            t.skip_inner(ty, depth)
                        }
                    };
                    for _ in 0..n {
                        skip_one(self, kt, depth + 1)?;
                        skip_one(self, vt, depth + 1)?;
                    }
                }
            }
            ttype::STRUCT => {
                self.enter()?;
                while let Some((ft, _)) = self.read_field_begin()? {
                    self.skip_inner(ft, depth + 1)?;
                }
                self.leave();
            }
            _ => err!(BadThrift, self.pos),
        }
        Ok(())
    }
}

/// Upper bound on list length while skipping.
const MAX_SKIP_LIST: usize = 1 << 24;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_bool_elem_consumes_one_byte_per_element() {
        // 3 elements: BOOLEAN_TRUE(1), BOOLEAN_FALSE(2), BOOLEAN_TRUE(1).
        let buf = [1u8, 2, 1];
        let mut t = Thrift::new(&buf);
        assert!(t.read_bool_elem().unwrap());
        assert!(!t.read_bool_elem().unwrap());
        assert!(t.read_bool_elem().unwrap());
        assert_eq!(t.pos(), 3);
    }

    #[test]
    fn read_bool_elem_treats_any_non_true_tag_as_false() {
        // Per spec, values other than 1/2 shouldn't appear, but this pins down
        // that a value like 0 falls to false as well (only == BOOL_TRUE counts as true).
        let buf = [0u8];
        let mut t = Thrift::new(&buf);
        assert!(!t.read_bool_elem().unwrap());
    }

    #[test]
    fn skip_inner_treats_bool_list_elements_as_one_byte_each() {
        // Skip the 3 elements of a list<bool>, and confirm the byte count
        // hasn't drifted by successfully reading the field right after. A bool
        // struct field has its value embedded in the header nibble, but a
        // list/set element does not -- it occupies 1 byte per element (same
        // convention as `read_bool_elem`). Passing it straight through and
        // treating it as 0 bytes here would throw off every subsequent field.
        let mut buf = Vec::new();
        // Field 1: list<bool>, 3 elements.
        // Field header: delta=1, type=LIST(9) -> 0x19
        buf.push(0x19);
        // List header: size=3, elem_type=BOOL_TRUE(1) -> (3<<4)|1 = 0x31
        buf.push(0x31);
        buf.extend_from_slice(&[1u8, 2, 1]); // raw bytes for the 3 elements
                                             // Field 2: I32 value 42 (delta=1, type=I32(5) -> 0x15, zigzag(42)=84)
        buf.push(0x15);
        buf.push(84);
        buf.push(0); // STOP

        let mut t = Thrift::new(&buf);
        t.enter().unwrap();
        let (ft1, id1) = t.read_field_begin().unwrap().unwrap();
        assert_eq!(id1, 1);
        t.skip(ft1).unwrap();
        let (ft2, id2) = t.read_field_begin().unwrap().unwrap();
        assert_eq!(id2, 2);
        assert_eq!(t.read_i32(ft2).unwrap(), 42, "read position drifted after skipping list<bool>");
        assert!(t.read_field_begin().unwrap().is_none());
        t.leave();
    }

    #[test]
    fn skip_inner_treats_bool_map_key_and_value_elements_as_one_byte_each() {
        // A regression test that skips 2 entries of a map<bool, bool> and
        // confirms the field right after can still be read correctly. The same
        // gap as list/set existed independently for both the map's key and its
        // value (since it could break for just the key side or just the value
        // side, both are checked as bool at once here).
        let mut buf = Vec::new();
        // Field 1: map<bool,bool>, 2 entries.
        // Field header: delta=1, type=MAP(11) -> 0x1B
        buf.push(0x1B);
        buf.push(0x02); // map size = 2
                        // kv-type nibble: kt=vt=BOOL_TRUE(1) -> (1<<4)|1 = 0x11
        buf.push(0x11);
        buf.extend_from_slice(&[1u8, 2, 2, 1]); // raw bytes for key1,val1,key2,val2
                                                // Field 2: I32 value 42 (delta=1, type=I32(5) -> 0x15, zigzag(42)=84)
        buf.push(0x15);
        buf.push(84);
        buf.push(0); // STOP

        let mut t = Thrift::new(&buf);
        t.enter().unwrap();
        let (ft1, id1) = t.read_field_begin().unwrap().unwrap();
        assert_eq!(id1, 1);
        t.skip(ft1).unwrap();
        let (ft2, id2) = t.read_field_begin().unwrap().unwrap();
        assert_eq!(id2, 2);
        assert_eq!(
            t.read_i32(ft2).unwrap(),
            42,
            "read position drifted after skipping map<bool,bool>"
        );
        assert!(t.read_field_begin().unwrap().is_none());
        t.leave();
    }
}
