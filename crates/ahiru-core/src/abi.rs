//! wasm ABI。
//!
//! ホストとのやり取りは「実行を止めて必要なバイト範囲を返す」ループで表現する。
//! Asyncify を使わないので生成コードが膨らまない（DESIGN.md §6）。
//!
//! ```text
//! for (;;) {
//!   const st = ahiru_query_step(q);
//!   if (st === NEED_IO)     { await fetchAll(ahiru_io_requests(q)); continue; }
//!   if (st === BATCH_READY) { yield decode(ahiru_result(q)); continue; }
//!   break;
//! }
//! ```
//!
//! 文字列は UTF-8 のポインタ + 長さで渡す。エラーはコード（`u32`）だけを返し、
//! メッセージ文字列は JS 側のテーブルで組み立てる。これだけで wasm から
//! 20 KB 前後のメッセージ文字列を追い出せる。

use core::cell::UnsafeCell;

use crate::exec::IoRequest;
use crate::prelude::*;
use crate::session::{Prepared, Query, QueryStep, Session};
use crate::vector::{Batch, Data, Field, PhysType};

pub const STATUS_BATCH_READY: i32 = 0;
pub const STATUS_NEED_IO: i32 = 1;
pub const STATUS_DONE: i32 = 2;
pub const STATUS_ERROR: i32 = 3;
/// 内蔵しないコーデックの展開をホストに依頼する。
pub const STATUS_NEED_CODEC: i32 = 4;

/// 結果バッファの先頭に置くマジック。JS 側のデコーダと同期させる。
const RESULT_MAGIC: u32 = 0x4148_5231; // "AHR1"

struct State {
    sessions: Vec<Option<Session>>,
    queries: Vec<Option<QuerySlot>>,
    last_error: u32,
    /// `ahiru_result` / `ahiru_io_requests` が返すバッファ。
    /// 次の呼び出しまで生存させる必要があるのでここに置く。
    out: Vec<u8>,
}

struct QuerySlot {
    session: usize,
    query: Query,
    io: Vec<IoRequest>,
}

struct Cell(UnsafeCell<Option<State>>);
// wasm32 は単一スレッド。
unsafe impl Sync for Cell {}
static STATE: Cell = Cell(UnsafeCell::new(None));

#[allow(clippy::mut_from_ref)]
fn state() -> &'static mut State {
    let slot = unsafe { &mut *STATE.0.get() };
    if slot.is_none() {
        *slot = Some(State {
            sessions: Vec::new(),
            queries: Vec::new(),
            last_error: 0,
            out: Vec::new(),
        });
    }
    match slot {
        Some(s) => s,
        // 直前に必ず Some を入れているので到達しない。
        None => unreachable!(),
    }
}

fn fail<T>(e: crate::error::Error, fallback: T) -> T {
    state().last_error = e.code_u16() as u32;
    fallback
}

/// 入口ごとに直前のエラーを消す。消さないと、成功した呼び出しの後に
/// 古いコードが読めてしまう。
fn clear_error() {
    state().last_error = 0;
}

fn fail_code<T>(code: crate::error::Code, fallback: T) -> T {
    state().last_error = code as u16 as u32;
    fallback
}

// --- メモリ -----------------------------------------------------------------

/// ホストが書き込むためのバッファを確保する。
#[no_mangle]
pub extern "C" fn ahiru_alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len);
    let p = v.as_mut_ptr();
    core::mem::forget(v);
    p
}

/// `ahiru_alloc` で確保した領域を返す。
///
/// # Safety
/// `ptr` は同じ `len` で `ahiru_alloc` が返したものでなければならない。
#[no_mangle]
pub unsafe extern "C" fn ahiru_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        drop(unsafe { Vec::from_raw_parts(ptr, 0, len) });
    }
}

unsafe fn slice<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    if ptr.is_null() || len == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(ptr, len) }
    }
}

// --- セッション -------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ahiru_session_new() -> i32 {
    let s = state();
    s.sessions.push(Some(Session::new()));
    (s.sessions.len() - 1) as i32
}

#[no_mangle]
pub extern "C" fn ahiru_session_free(h: i32) {
    let s = state();
    if let Some(slot) = s.sessions.get_mut(h as usize) {
        *slot = None;
    }
}

fn session(h: i32) -> Option<&'static mut Session> {
    state().sessions.get_mut(h as usize)?.as_mut()
}

/// フォーマットコード。JS 側の定数と 1:1 で対応する。
fn format_kind(v: u32) -> Result<crate::format::FormatKind> {
    use crate::format::FormatKind::*;
    Ok(match v {
        0 => Auto,
        1 => Parquet,
        2 => Csv,
        3 => Tsv,
        4 => Jsonl,
        _ => err!(UnsupportedFeature),
    })
}

/// ホストがレンジ取得で供給するテーブルを登録する。返り値はテーブル添字。
/// フォーマットは名前（拡張子）から推定する。
///
/// # Safety
/// `name` は `name_len` バイトの有効な UTF-8 を指していること。
#[no_mangle]
pub unsafe extern "C" fn ahiru_register(
    h: i32,
    name: *const u8,
    name_len: usize,
    total_len: u64,
) -> i32 {
    unsafe { ahiru_register_as(h, name, name_len, total_len, 0) }
}

/// フォーマットを明示して登録する。
///
/// 拡張子推定では、テーブル名が拡張子を持つことを強制してしまう
/// （`FROM "logs.csv"` と書かざるを得ない）。名前と読み方を分けられるように、
/// 明示指定の入口を用意する。`format` は 0=Auto, 1=Parquet, 2=Csv, 3=Tsv,
/// 4=Jsonl。
///
/// # Safety
/// `name` は `name_len` バイトの有効な UTF-8 を指していること。
#[no_mangle]
pub unsafe extern "C" fn ahiru_register_as(
    h: i32,
    name: *const u8,
    name_len: usize,
    total_len: u64,
    format: u32,
) -> i32 {
    clear_error();
    let bytes = unsafe { slice(name, name_len) };
    let name = match core::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => return fail_code(crate::error::Code::Internal, -1),
    };
    let kind = match format_kind(format) {
        Ok(k) => k,
        Err(e) => return fail(e, -1),
    };
    match session(h) {
        Some(s) => match s.register_remote_as(name, total_len, kind) {
            Ok(i) => i as i32,
            Err(e) => fail(e, -1),
        },
        None => fail_code(crate::error::Code::Internal, -1),
    }
}

/// 取得したバイト列をセッションに渡す。
///
/// # Safety
/// `data` は `len` バイトの有効な領域を指していること。
#[no_mangle]
pub unsafe extern "C" fn ahiru_provide(
    h: i32,
    table: u32,
    offset: u64,
    data: *const u8,
    len: usize,
) -> i32 {
    clear_error();
    let bytes = unsafe { slice(data, len) }.to_vec();
    match session(h) {
        Some(s) => match s.provide(table as usize, offset, bytes) {
            Ok(()) => 0,
            Err(e) => fail(e, -1),
        },
        None => fail_code(crate::error::Code::Internal, -1),
    }
}

/// ホストが展開した圧縮ブロックを渡す。`offset` と `len` は
/// `STATUS_NEED_CODEC` で返した要求のものと一致していなければならない。
///
/// # Safety
/// `data` は `data_len` バイトの有効な領域を指していること。
#[no_mangle]
pub unsafe extern "C" fn ahiru_provide_codec(
    h: i32,
    table: u32,
    offset: u64,
    len: u32,
    data: *const u8,
    data_len: usize,
) -> i32 {
    clear_error();
    let bytes = unsafe { slice(data, data_len) }.to_vec();
    match session(h) {
        Some(s) => match s.provide_decoded(table as usize, offset, len, bytes) {
            Ok(()) => 0,
            Err(e) => fail(e, -1),
        },
        None => fail_code(crate::error::Code::Internal, -1),
    }
}

// --- クエリ -----------------------------------------------------------------

/// SQL を受け取りクエリハンドルを返す。失敗時は負値。
///
/// # Safety
/// `sql` は `sql_len` バイトの有効な UTF-8 を指していること。
#[no_mangle]
pub unsafe extern "C" fn ahiru_query_start(
    h: i32,
    sql: *const u8,
    sql_len: usize,
    params_ptr: *const u8,
    params_len: usize,
) -> i32 {
    clear_error();
    let bytes = unsafe { slice(sql, sql_len) };
    let sql = match core::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => return fail_code(crate::error::Code::Internal, -1),
    };
    let params = match decode_params(unsafe { slice(params_ptr, params_len) }) {
        Ok(p) => p,
        Err(e) => return fail(e, -1),
    };
    let s = match session(h) {
        Some(s) => s,
        None => return fail_code(crate::error::Code::Internal, -1),
    };
    match s.prepare(sql, &params) {
        Ok(Prepared::Ready(q)) => {
            let st = state();
            st.queries.push(Some(QuerySlot { session: h as usize, query: q, io: Vec::new() }));
            (st.queries.len() - 1) as i32
        }
        // フッタ取得が必要な段階。ホストは要求を満たしてから呼び直す。
        Ok(Prepared::NeedIo(io)) => {
            let st = state();
            st.out = encode_io(&io);
            -2
        }
        Err(e) => fail(e, -1),
    }
}

#[no_mangle]
pub extern "C" fn ahiru_query_step(q: i32) -> i32 {
    clear_error();
    let st = state();
    let slot = match st.queries.get_mut(q as usize).and_then(|s| s.as_mut()) {
        Some(s) => s,
        None => return STATUS_ERROR,
    };
    let sidx = slot.session;
    // セッションとクエリを同時に可変借用できないので、いったん取り出す。
    let mut query = match core::mem::replace(&mut st.queries[q as usize], None) {
        Some(s) => s,
        None => return STATUS_ERROR,
    };
    let session = match st.sessions.get_mut(sidx).and_then(|s| s.as_mut()) {
        Some(s) => s,
        None => return STATUS_ERROR,
    };
    let r = session.step(&mut query.query);
    let status = match r {
        Ok(QueryStep::Batch(mut b)) => {
            // selection vector を実体化してから直列化する。これを忘れると
            // `LIMIT ... OFFSET` が先頭から返ってしまう。
            b.materialize();
            st.out = encode_batch(&b);
            STATUS_BATCH_READY
        }
        Ok(QueryStep::NeedIo(io)) => {
            query.io = io;
            st.out = encode_io(&query.io);
            STATUS_NEED_IO
        }
        Ok(QueryStep::NeedCodec(reqs)) => {
            st.out = encode_codec(&reqs);
            STATUS_NEED_CODEC
        }
        Ok(QueryStep::Done) => STATUS_DONE,
        Err(e) => {
            st.last_error = e.code_u16() as u32;
            STATUS_ERROR
        }
    };
    st.queries[q as usize] = Some(query);
    status
}

#[no_mangle]
pub extern "C" fn ahiru_query_close(q: i32) {
    clear_error();
    let st = state();
    if let Some(slot) = st.queries.get_mut(q as usize) {
        *slot = None;
    }
}

/// 直前の `ahiru_query_step` / `ahiru_query_start` が用意したバッファ。
#[no_mangle]
pub extern "C" fn ahiru_out_ptr() -> *const u8 {
    state().out.as_ptr()
}

#[no_mangle]
pub extern "C" fn ahiru_out_len() -> usize {
    state().out.len()
}

#[no_mangle]
pub extern "C" fn ahiru_last_error() -> u32 {
    state().last_error
}

/// 現在ヒープが保持しているバイト数。
#[no_mangle]
pub extern "C" fn ahiru_heap_used() -> usize {
    crate::rt::alloc::heap_used()
}

// --- 直列化 -----------------------------------------------------------------

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// コーデック展開要求の列。`[count][{table,codec,offset,len,out_len}...]`
fn encode_codec(reqs: &[crate::exec::CodecRequest]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + reqs.len() * 24);
    put_u32(&mut out, reqs.len() as u32);
    for r in reqs {
        put_u32(&mut out, r.table as u32);
        put_u32(&mut out, r.codec as u32);
        put_u64(&mut out, r.offset);
        put_u32(&mut out, r.len);
        put_u32(&mut out, r.out_len);
    }
    out
}

/// パラメータ列を読む。`[count][{tag}{payload}...]`
///
/// tag: 0=NULL, 1=BOOL(1B), 2=I64(8B), 3=F64(8B), 4=BYTES(u32 len + bytes)
fn decode_params(buf: &[u8]) -> Result<Vec<crate::vector::Value>> {
    use crate::vector::Value;
    if buf.is_empty() {
        return Ok(Vec::new());
    }
    ensure!(buf.len() >= 4, UnexpectedEof);
    let n = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    ensure!(n <= 4096, LimitExceeded);
    let mut pos = 4usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        ensure!(pos < buf.len(), UnexpectedEof);
        let tag = buf[pos];
        pos += 1;
        let v = match tag {
            0 => Value::Null,
            1 => {
                ensure!(pos < buf.len(), UnexpectedEof);
                let b = buf[pos] != 0;
                pos += 1;
                Value::Bool(b)
            }
            2 | 3 => {
                ensure!(buf.len() - pos >= 8, UnexpectedEof);
                let mut a = [0u8; 8];
                a.copy_from_slice(&buf[pos..pos + 8]);
                pos += 8;
                if tag == 2 {
                    Value::I64(i64::from_le_bytes(a))
                } else {
                    Value::F64(f64::from_le_bytes(a))
                }
            }
            4 => {
                ensure!(buf.len() - pos >= 4, UnexpectedEof);
                let l = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]])
                    as usize;
                pos += 4;
                ensure!(buf.len() - pos >= l, UnexpectedEof);
                let v = Value::Bytes(buf[pos..pos + l].to_vec());
                pos += l;
                v
            }
            _ => err!(BadThrift),
        };
        out.push(v);
    }
    Ok(out)
}

/// I/O 要求の列。`[count][{table,offset,len}...]`
fn encode_io(io: &[IoRequest]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + io.len() * 20);
    put_u32(&mut out, io.len() as u32);
    for r in io {
        put_u32(&mut out, r.table as u32);
        put_u64(&mut out, r.offset);
        put_u64(&mut out, r.len);
    }
    out
}

/// 1 バッチの列指向表現。
///
/// ```text
/// magic:u32 num_cols:u32 num_rows:u32
/// 列ごとに: phys:u32 validity_len:u32 [validity bytes] data_len:u32 [data bytes]
///           Bytes 型のみ offsets_len:u32 [offsets] が data の前に入る
/// ```
///
/// Arrow IPC 互換にするのは今後の課題。まずは JS 側が TypedArray として
/// そのまま読める最小の形にしてある。
fn encode_batch(b: &Batch) -> Vec<u8> {
    let mut out = Vec::new();
    put_u32(&mut out, RESULT_MAGIC);
    put_u32(&mut out, b.cols.len() as u32);
    put_u32(&mut out, b.card() as u32);

    for c in &b.cols {
        put_u32(&mut out, c.data().phys() as u32);
        match c.validity() {
            Some(v) => {
                let words = v.as_words();
                put_u32(&mut out, (words.len() * 8) as u32);
                for w in words {
                    put_u64(&mut out, *w);
                }
            }
            None => put_u32(&mut out, 0),
        }
        match c.data() {
            Data::Bool(bm) => {
                let words = bm.as_words();
                put_u32(&mut out, (words.len() * 8) as u32);
                for w in words {
                    put_u64(&mut out, *w);
                }
            }
            Data::I32(v) => put_slice(&mut out, v, 4),
            Data::I64(v) => put_slice(&mut out, v, 8),
            Data::F64(v) => put_slice(&mut out, v, 8),
            Data::I128(v) => put_slice(&mut out, v, 16),
            Data::Bytes(bd) => {
                put_u32(&mut out, (bd.offsets.len() * 4) as u32);
                for o in &bd.offsets {
                    put_u32(&mut out, *o);
                }
                put_u32(&mut out, bd.data.len() as u32);
                out.extend_from_slice(&bd.data);
            }
        }
    }
    out
}

/// 固定幅の数値配列をリトルエンディアンで書く。
fn put_slice<T: Copy>(out: &mut Vec<u8>, v: &[T], width: usize) {
    put_u32(out, (v.len() * width) as u32);
    // wasm はリトルエンディアンなので、そのままバイト列として写せる。
    let bytes = unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * width) };
    out.extend_from_slice(bytes);
}

/// 出力スキーマ（列名と型）を書く。`ahiru_query_start` の直後に呼ぶ。
///
/// 返り値は書き込んだバイト数。ハンドルが不正な場合は `-1`（列が 0 個の
/// 正常な場合と区別できるようにするため）。
#[no_mangle]
pub extern "C" fn ahiru_schema(q: i32) -> isize {
    clear_error();
    let st = state();
    let slot = match st.queries.get(q as usize).and_then(|s| s.as_ref()) {
        Some(s) => s,
        None => return fail_code(crate::error::Code::Internal, -1),
    };
    st.out = encode_schema(&slot.query.schema);
    st.out.len() as isize
}

fn encode_schema(fields: &[Field]) -> Vec<u8> {
    let mut out = Vec::new();
    put_u32(&mut out, fields.len() as u32);
    for f in fields {
        put_u32(&mut out, ty_code(f.ty));
        put_u32(&mut out, f.ty.phys() as u32);
        // DECIMAL は precision/scale が無いと値を復元できない。
        // 型コードだけ送ると、スケール前の整数がそのまま出てしまう。
        let (p, sc) = match f.ty {
            crate::vector::Ty::Decimal { precision, scale } => (precision as u32, scale as u32),
            _ => (0, 0),
        };
        put_u32(&mut out, p);
        put_u32(&mut out, sc);
        put_u32(&mut out, f.name.len() as u32);
        out.extend_from_slice(f.name.as_bytes());
    }
    out
}

/// 論理型のコード。JS 側の型名テーブルと 1:1 で対応する。
fn ty_code(t: crate::vector::Ty) -> u32 {
    use crate::vector::Ty::*;
    match t {
        Null => 0,
        Boolean => 1,
        TinyInt => 2,
        SmallInt => 3,
        Int => 4,
        BigInt => 5,
        HugeInt => 6,
        UTinyInt => 7,
        USmallInt => 8,
        UInt => 9,
        UBigInt => 10,
        Float => 11,
        Double => 12,
        Decimal { .. } => 13,
        Varchar => 14,
        Blob => 15,
        Date => 16,
        Time => 17,
        Timestamp => 18,
    }
}

// 物理型コードが JS 側の想定とずれていないかを固定する。
const _: () = {
    assert!(PhysType::Bool as u32 == 0);
    assert!(PhysType::Bytes as u32 == 5);
};
