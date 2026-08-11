//! Thrift Compact Protocol リーダ（読み取り専用）。
//!
//! Parquet のフッタとページヘッダはこの形式でエンコードされている。
//! 汎用 Thrift ランタイムは持たず、`meta.rs` の手書きデコーダが必要とする
//! 操作だけを提供する。
//!
//! 入力はネットワーク由来で信用できない。全ての読み取りは境界検査を行い、
//! 破損に対しては必ず `Err` を返す（パニックしない）。

use crate::prelude::*;

/// Thrift Compact の型タグ。
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

/// 構造体のネスト深さ上限。循環・過深な入力でスタックを枯渇させないため。
pub const MAX_DEPTH: usize = 32;

pub struct Thrift<'a> {
    buf: &'a [u8],
    pos: usize,
    /// 各ネストレベルの直前フィールド ID（compact protocol の差分符号化用）。
    id_stack: [i16; MAX_DEPTH],
    depth: usize,
}

impl<'a> Thrift<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Thrift { buf, pos: 0, id_stack: [0; MAX_DEPTH], depth: 0 }
    }

    /// 現在の読み取り位置。エラー位置の報告と消費バイト数の算出に使う。
    #[inline]
    pub fn pos(&self) -> usize {
        self.pos
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// 構造体に入る。フィールド ID の差分符号化コンテキストを 1 段積む。
    pub fn enter(&mut self) -> Result<()> {
        ensure!(self.depth + 1 < MAX_DEPTH, NestingTooDeep, self.pos);
        self.depth += 1;
        self.id_stack[self.depth] = 0;
        Ok(())
    }

    /// 構造体から出る。`read_field_begin` が STOP を返した後に呼ぶ。
    pub fn leave(&mut self) {
        if self.depth > 0 {
            self.depth -= 1;
        }
    }

    // --- プリミティブ -------------------------------------------------------

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

    /// LEB128 可変長整数。10 バイトを超えたら破損とみなす。
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

    // --- フィールド ---------------------------------------------------------

    /// 次のフィールドヘッダを読む。STOP なら `None`。
    /// 返り値は `(型タグ, フィールド ID)`。
    pub fn read_field_begin(&mut self) -> Result<Option<(u8, i16)>> {
        let b = self.byte()?;
        if b == ttype::STOP {
            return Ok(None);
        }
        let ftype = b & 0x0f;
        let delta = (b >> 4) as i16;
        let id = if delta == 0 {
            // 差分ではなく絶対 ID が zigzag varint で続く。
            let v = self.zigzag()?;
            ensure!(v >= i16::MIN as i64 && v <= i16::MAX as i64, BadThrift, self.pos);
            v as i16
        } else {
            self.id_stack[self.depth].wrapping_add(delta)
        };
        self.id_stack[self.depth] = id;
        Ok(Some((ftype, id)))
    }

    /// BOOL。フィールドとして現れる場合は値が型タグに埋め込まれている。
    /// リスト要素として現れる場合は 1 バイトを読む。
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

    /// 可変長バイト列。バッファへの参照をそのまま返す（コピーしない）。
    pub fn read_binary(&mut self, _ftype: u8) -> Result<&'a [u8]> {
        let len = self.uvarint()? as usize;
        ensure!(len <= self.buf.len() - self.pos, UnexpectedEof, self.pos);
        self.take(len)
    }

    /// UTF-8 文字列。不正なバイト列は置換文字にせず、そのまま lossy 変換する。
    /// スキーマ名に不正な UTF-8 が来てもクエリを失敗させないため。
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

    /// リスト/セットのヘッダ。`(要素型, 要素数)` を返す。
    /// `max` を超える要素数は破損または攻撃とみなして拒否する。
    pub fn read_list_begin(&mut self, max: usize) -> Result<(u8, usize)> {
        let b = self.byte()?;
        let etype = b & 0x0f;
        let mut size = (b >> 4) as usize;
        if size == 0x0f {
            size = self.uvarint()? as usize;
        }
        ensure!(size <= max, LimitExceeded, self.pos);
        // 要素あたり最低 1 バイトは必要。宣言サイズが残りバッファを超えるなら破損。
        ensure!(size <= self.remaining() || etype == ttype::BOOL_TRUE, UnexpectedEof, self.pos);
        Ok((etype, size))
    }

    /// 未知フィールドを読み飛ばす。
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
                for _ in 0..n {
                    self.skip_inner(et, depth + 1)?;
                }
            }
            ttype::MAP => {
                let n = self.uvarint()? as usize;
                if n > 0 {
                    ensure!(n <= MAX_SKIP_LIST, LimitExceeded, self.pos);
                    let kv = self.byte()?;
                    let (kt, vt) = (kv >> 4, kv & 0x0f);
                    for _ in 0..n {
                        self.skip_inner(kt, depth + 1)?;
                        self.skip_inner(vt, depth + 1)?;
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

/// `skip` 中のリスト長上限。
const MAX_SKIP_LIST: usize = 1 << 24;
