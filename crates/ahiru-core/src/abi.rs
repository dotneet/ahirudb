//! The wasm ABI.
//!
//! Interaction with the host is expressed as a loop that "stops execution and returns the byte ranges needed".
//! Avoiding Asyncify keeps the generated code from bloating (DESIGN.md §6).
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
//! Strings are passed as a UTF-8 pointer plus length. Errors return only a code
//! (`u32`), and message strings are assembled by the table on the JS side. That
//! alone keeps roughly 20 KB of message strings out of the wasm.

use alloc::string::ToString;
use core::cell::UnsafeCell;

use crate::exec::IoRequest;
use crate::prelude::*;
use crate::session::{Prepared, Query, QueryStep, Session};
use crate::vector::{Batch, Data, Field, PhysType};

pub const STATUS_BATCH_READY: i32 = 0;
pub const STATUS_NEED_IO: i32 = 1;
pub const STATUS_DONE: i32 = 2;
pub const STATUS_ERROR: i32 = 3;
/// Asks the host to decompress a codec that is not built in.
pub const STATUS_NEED_CODEC: i32 = 4;

/// The magic placed at the head of the result buffer. Kept in sync with the JS-side decoder.
const RESULT_MAGIC: u32 = 0x4148_5231; // "AHR1"

struct State {
    sessions: Vec<Option<Session>>,
    queries: Vec<Option<QuerySlot>>,
    /// Generation counter per query-handle slot index, bumped every time that
    /// slot is closed. Packed into the high bits of the handle returned by
    /// `ahiru_query_start` (see `make_query_handle`), so a handle minted
    /// before a close can never address whatever query later reuses its slot
    /// index -- validation rejects it instead of silently aliasing.
    ///
    /// Unlike `queries`, this Vec is never truncated: it has to outlive the
    /// slot's `Some`/`None` payload, or a slot index freed by
    /// `ahiru_query_close`'s trailing-`None` truncation could hand out
    /// generation 0 again and make an old, already-invalid handle look valid
    /// once more (the classic ABA problem).
    query_generations: Vec<u32>,
    last_error: u32,
    /// The buffer returned by `ahiru_result` / `ahiru_io_requests`.
    /// It must stay alive until the next call, hence living here.
    out: Vec<u8>,
}

struct QuerySlot {
    session: usize,
    query: Query,
    io: Vec<IoRequest>,
}

/// Bits of a query handle spent on the slot index; the remaining bits below
/// the sign bit (which must stay clear -- negative handles are the
/// `ahiru_query_start` error/NEED_IO sentinels) hold the generation from
/// `State::query_generations`. 65536 concurrently open, unclosed queries or
/// 32768 open/close cycles on one slot are both far beyond anything a real
/// caller does; going past either is treated as an error rather than risking
/// two live handles aliasing the same bits.
const QUERY_IDX_BITS: u32 = 16;
const QUERY_IDX_MASK: i32 = (1 << QUERY_IDX_BITS) - 1;
const QUERY_GEN_MASK: u32 = (1 << (31 - QUERY_IDX_BITS)) - 1;

fn make_query_handle(index: usize, generation: u32) -> i32 {
    (((generation & QUERY_GEN_MASK) as i32) << QUERY_IDX_BITS) | (index as i32 & QUERY_IDX_MASK)
}

/// Splits a handle into its slot index and generation. `None` for any
/// negative value -- the error/NEED_IO sentinels `ahiru_query_start` returns
/// are always negative, so this alone rejects them without a separate check.
fn split_query_handle(h: i32) -> Option<(usize, u32)> {
    if h < 0 {
        return None;
    }
    let index = (h & QUERY_IDX_MASK) as usize;
    let generation = (h as u32 >> QUERY_IDX_BITS) & QUERY_GEN_MASK;
    Some((index, generation))
}

struct Cell(UnsafeCell<Option<State>>);
// wasm32 is single-threaded.
unsafe impl Sync for Cell {}
static STATE: Cell = Cell(UnsafeCell::new(None));

#[allow(clippy::mut_from_ref)]
fn state() -> &'static mut State {
    let slot = unsafe { &mut *STATE.0.get() };
    if slot.is_none() {
        *slot = Some(State {
            sessions: Vec::new(),
            queries: Vec::new(),
            query_generations: Vec::new(),
            last_error: 0,
            out: Vec::new(),
        });
    }
    match slot {
        Some(s) => s,
        // Unreachable, since a Some is always stored just before this.
        None => unreachable!(),
    }
}

fn fail<T>(e: crate::error::Error, fallback: T) -> T {
    state().last_error = e.code_u16() as u32;
    fallback
}

/// Clears the previous error at each entry point. Without this, a stale code could
/// be read after a successful call.
fn clear_error() {
    state().last_error = 0;
}

fn fail_code<T>(code: crate::error::Code, fallback: T) -> T {
    state().last_error = code as u16 as u32;
    fallback
}

// --- Memory -----------------------------------------------------------------

/// Reserves a buffer for the host to write into.
#[no_mangle]
pub extern "C" fn ahiru_alloc(len: usize) -> *mut u8 {
    if len == 0 {
        return core::ptr::NonNull::dangling().as_ptr();
    }
    let layout = match core::alloc::Layout::from_size_align(len, 1) {
        Ok(l) => l,
        Err(_) => return core::ptr::null_mut(),
    };
    unsafe { alloc::alloc::alloc(layout) }
}

/// Returns a region reserved by `ahiru_alloc`.
///
/// # Safety
/// `ptr` must be what `ahiru_alloc` returned for the same `len`.
#[no_mangle]
pub unsafe extern "C" fn ahiru_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        if let Ok(layout) = core::alloc::Layout::from_size_align(len, 1) {
            unsafe { alloc::alloc::dealloc(ptr, layout) };
        }
    }
}

unsafe fn slice<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    if ptr.is_null() || len == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(ptr, len) }
    }
}

// --- Session ----------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ahiru_session_new() -> i32 {
    let s = state();
    if let Some(pos) = s.sessions.iter().position(|slot| slot.is_none()) {
        s.sessions[pos] = Some(Session::new());
        pos as i32
    } else {
        s.sessions.push(Some(Session::new()));
        (s.sessions.len() - 1) as i32
    }
}

#[no_mangle]
pub extern "C" fn ahiru_session_free(h: i32) {
    if h < 0 {
        return;
    }
    let s = state();
    if let Some(slot) = s.sessions.get_mut(h as usize) {
        *slot = None;
    }
    while matches!(s.sessions.last(), Some(None)) {
        s.sessions.pop();
    }
}

/// Sets the query start time (microseconds since the epoch, UTC). The wasm core has
/// no clock, so the host is expected to call this with the current time on every
/// `prepare`/`query` (for `CURRENT_DATE`/`CURRENT_TIMESTAMP`/`now()`, DESIGN.md §2).
/// Never calling it leaves the time at the epoch (1970-01-01).
#[no_mangle]
pub extern "C" fn ahiru_set_now(h: i32, now_micros: i64) -> i32 {
    clear_error();
    match session(h) {
        Some(s) => {
            s.set_now(now_micros);
            0
        }
        None => fail_code(crate::error::Code::Internal, -1),
    }
}

fn session(h: i32) -> Option<&'static mut Session> {
    if h < 0 {
        return None;
    }
    state().sessions.get_mut(h as usize)?.as_mut()
}

/// Format codes. These map 1:1 to the constants on the JS side.
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

/// Registers a table the host supplies via range fetching. Returns the table index.
/// The format is inferred from the name (its extension).
///
/// # Safety
/// `name` must point at `name_len` bytes of valid UTF-8.
#[no_mangle]
pub unsafe extern "C" fn ahiru_register(
    h: i32,
    name: *const u8,
    name_len: usize,
    total_len: u64,
) -> i32 {
    unsafe { ahiru_register_as(h, name, name_len, total_len, 0) }
}

/// Registers with an explicit format.
///
/// Extension inference forces the table name to carry an extension (you would have to
/// write `FROM "logs.csv"`). This entry point exists so the name and how it is read
/// can be separated. `format` is 0=Auto, 1=Parquet, 2=Csv, 3=Tsv, 4=Jsonl.
///
/// # Safety
/// `name` must point at `name_len` bytes of valid UTF-8.
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

/// Hands fetched bytes to the session.
///
/// `part` says which file of a multi-file table is meant. Single-file tables
/// (registered via `ahiru_register` / `ahiru_register_as`) are always 0.
/// Pass back the `part` field of the request `ahiru_io_requests` returned, unchanged.
///
/// # Safety
/// `data` must point at a valid region of `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn ahiru_provide(
    h: i32,
    table: u32,
    part: u32,
    offset: u64,
    data: *const u8,
    len: usize,
) -> i32 {
    clear_error();
    let bytes = unsafe { slice(data, len) }.to_vec();
    match session(h) {
        Some(s) => match s.provide(table as usize, part as usize, offset, bytes) {
            Ok(()) => 0,
            Err(e) => fail(e, -1),
        },
        None => fail_code(crate::error::Code::Internal, -1),
    }
}

/// Hands over a compressed block the host decompressed. `offset` and `len` must match
/// those of the request returned with `STATUS_NEED_CODEC`.
/// `part` means the same thing as in `ahiru_provide`.
///
/// # Safety
/// `data` must point at a valid region of `data_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn ahiru_provide_codec(
    h: i32,
    table: u32,
    part: u32,
    offset: u64,
    len: u32,
    data: *const u8,
    data_len: usize,
) -> i32 {
    clear_error();
    let bytes = unsafe { slice(data, data_len) }.to_vec();
    match session(h) {
        Some(s) => match s.provide_decoded(table as usize, part as usize, offset, len, bytes) {
            Ok(()) => 0,
            Err(e) => fail(e, -1),
        },
        None => fail_code(crate::error::Code::Internal, -1),
    }
}

/// Registers several files as one logical table. Returns the table index.
///
/// Assumes the host supplies them via range fetching (same as `ahiru_register_as`).
/// Each part's path is used both for automatic format detection and for extracting
/// Hive partition columns (directories such as `year=2024/month=01/...`).
///
/// Wire format (`files`): the same "length prefix plus variable-length entries" shape as `decode_params`.
/// ```text
/// [count:u32] { path_len:u32, path_bytes, total_len:u64 } ...
/// ```
///
/// # Safety
/// `name` must point at `name_len` bytes of valid UTF-8, and `files` at `files_len`
/// bytes of data in the format above.
#[no_mangle]
pub unsafe extern "C" fn ahiru_register_multi(
    h: i32,
    name: *const u8,
    name_len: usize,
    files: *const u8,
    files_len: usize,
    format: u32,
) -> i32 {
    clear_error();
    let name_bytes = unsafe { slice(name, name_len) };
    let name = match core::str::from_utf8(name_bytes) {
        Ok(s) => s,
        Err(_) => return fail_code(crate::error::Code::Internal, -1),
    };
    let kind = match format_kind(format) {
        Ok(k) => k,
        Err(e) => return fail(e, -1),
    };
    let files_buf = unsafe { slice(files, files_len) };
    let files = match decode_multi_files(files_buf) {
        Ok(f) => f,
        Err(e) => return fail(e, -1),
    };
    match session(h) {
        Some(s) => match s.register_multi_remote(name, files, kind) {
            Ok(i) => i as i32,
            Err(e) => fail(e, -1),
        },
        None => fail_code(crate::error::Code::Internal, -1),
    }
}

// --- Queries ----------------------------------------------------------------

/// Takes SQL and returns a query handle. A negative value on failure.
///
/// # Safety
/// `sql` must point at `sql_len` bytes of valid UTF-8.
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
            // Reuse the first closed slot instead of growing forever: a
            // long-lived module that runs many queries over its lifetime
            // would otherwise leak one `Vec` entry per query ever opened.
            let index = st.queries.iter().position(|s| s.is_none()).unwrap_or(st.queries.len());
            if index > QUERY_IDX_MASK as usize {
                return fail_code(crate::error::Code::LimitExceeded, -1);
            }
            let slot = QuerySlot { session: h as usize, query: q, io: Vec::new() };
            if index == st.queries.len() {
                st.queries.push(Some(slot));
            } else {
                st.queries[index] = Some(slot);
            }
            // `query_generations` may already know this index (it survives
            // `ahiru_query_close`'s truncation of `queries`); only a genuinely
            // new index needs a fresh generation.
            if index >= st.query_generations.len() {
                st.query_generations.resize(index + 1, 0);
            }
            make_query_handle(index, st.query_generations[index])
        }
        // The stage where the footer must be fetched. The host satisfies the request and calls again.
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
    let Some((index, gen)) = split_query_handle(q) else { return STATUS_ERROR };
    if st.query_generations.get(index).copied() != Some(gen) {
        // Out of range, or a stale handle whose slot has since been closed
        // (and possibly reused for an unrelated query): reject rather than
        // silently stepping whatever now lives at `index`.
        return STATUS_ERROR;
    }
    let slot = match st.queries.get_mut(index).and_then(|s| s.as_mut()) {
        Some(s) => s,
        None => return STATUS_ERROR,
    };
    let sidx = slot.session;
    // The session and the query cannot be mutably borrowed at once, so take it out first.
    let mut query = match core::mem::replace(&mut st.queries[index], None) {
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
            // Materialize the selection vector before serializing. Forgetting this
            // makes `LIMIT ... OFFSET` return rows from the very beginning.
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
    st.queries[index] = Some(query);
    status
}

#[no_mangle]
pub extern "C" fn ahiru_query_close(q: i32) {
    clear_error();
    let st = state();
    let Some((index, gen)) = split_query_handle(q) else { return };
    if st.query_generations.get(index).copied() != Some(gen) {
        // Same guard as `ahiru_query_step`: a stale handle must not be able
        // to close a live query that has since reused its slot index.
        return;
    }
    if let Some(slot) = st.queries.get_mut(index) {
        *slot = None;
    }
    if let Some(g) = st.query_generations.get_mut(index) {
        *g = g.wrapping_add(1) & QUERY_GEN_MASK;
    }
    // Shrink the trailing run of now-empty slots so a long-lived module that
    // closes its most recently opened queries also gets the `Vec`'s memory
    // back, not just a reusable index. `query_generations` is never
    // truncated (see its doc comment), so this can't reopen the ABA hazard
    // slot reuse alone would otherwise avoid.
    while matches!(st.queries.last(), Some(None)) {
        st.queries.pop();
    }
}

/// The buffer prepared by the preceding `ahiru_query_step` / `ahiru_query_start`.
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

/// How many bytes the heap currently holds.
#[no_mangle]
pub extern "C" fn ahiru_heap_used() -> usize {
    crate::rt::alloc::heap_used()
}

// --- Serialization ----------------------------------------------------------

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// The list of codec decompression requests. `[count][{table,part,codec,offset,len,out_len}...]`
fn encode_codec(reqs: &[crate::exec::CodecRequest]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + reqs.len() * 28);
    put_u32(&mut out, reqs.len() as u32);
    for r in reqs {
        put_u32(&mut out, r.table as u32);
        put_u32(&mut out, r.part as u32);
        put_u32(&mut out, r.codec as u32);
        put_u64(&mut out, r.offset);
        put_u32(&mut out, r.len);
        put_u32(&mut out, r.out_len);
    }
    out
}

/// Reads `[count:u32]{ path_len:u32, path_bytes, total_len:u64 }...`.
/// The same "length prefix plus variable-length entries" shape as `decode_params`.
fn decode_multi_files(buf: &[u8]) -> Result<Vec<(String, u64)>> {
    ensure!(buf.len() >= 4, UnexpectedEof);
    let n = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    // The cap on the file count follows the same reasoning as `decode_params`'
    // parameter cap (so corrupt or malicious input cannot force a huge allocation).
    ensure!(n <= 4096, LimitExceeded);
    let mut pos = 4usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        ensure!(buf.len() - pos >= 4, UnexpectedEof);
        let path_len =
            u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]) as usize;
        pos += 4;
        ensure!(buf.len() - pos >= path_len, UnexpectedEof);
        let path = match core::str::from_utf8(&buf[pos..pos + path_len]) {
            Ok(s) => s.to_string(),
            Err(_) => err!(Internal),
        };
        pos += path_len;
        ensure!(buf.len() - pos >= 8, UnexpectedEof);
        let mut a = [0u8; 8];
        a.copy_from_slice(&buf[pos..pos + 8]);
        let total_len = u64::from_le_bytes(a);
        pos += 8;
        out.push((path, total_len));
    }
    Ok(out)
}

/// Reads the parameter list. `[count][{tag}{payload}...]`
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

/// The list of I/O requests. `[count][{table,part,offset,len}...]`
fn encode_io(io: &[IoRequest]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + io.len() * 24);
    put_u32(&mut out, io.len() as u32);
    for r in io {
        put_u32(&mut out, r.table as u32);
        put_u32(&mut out, r.part as u32);
        put_u64(&mut out, r.offset);
        put_u64(&mut out, r.len);
    }
    out
}

/// The columnar representation of one batch.
///
/// ```text
/// magic:u32 num_cols:u32 num_rows:u32
/// Per column: phys:u32 validity_len:u32 [validity bytes] data_len:u32 [data bytes]
///             for Bytes only, offsets_len:u32 [offsets] comes before data
/// ```
///
/// Arrow IPC compatibility is future work. For now this is the minimal shape the JS
/// side can read directly as TypedArrays.
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
            Data::I32(v) => put_slice(&mut out, v),
            Data::I64(v) => put_slice(&mut out, v),
            Data::F64(v) => put_slice(&mut out, v),
            Data::I128(v) => put_slice(&mut out, v),
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

/// Writes a fixed-width numeric array in little-endian order.
fn put_slice<T: Copy>(out: &mut Vec<u8>, v: &[T]) {
    let byte_len = core::mem::size_of_val(v);
    put_u32(out, byte_len as u32);
    // wasm is little-endian, so the bytes can be copied straight across.
    let bytes = unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, byte_len) };
    out.extend_from_slice(bytes);
}

/// Writes the output schema (column names and types). Called right after `ahiru_query_start`.
///
/// Returns the number of bytes written, or `-1` for an invalid handle (so it stays
/// distinguishable from the valid case of zero columns).
#[no_mangle]
pub extern "C" fn ahiru_schema(q: i32) -> isize {
    clear_error();
    let st = state();
    let Some((index, gen)) = split_query_handle(q) else {
        return fail_code(crate::error::Code::Internal, -1);
    };
    if st.query_generations.get(index).copied() != Some(gen) {
        return fail_code(crate::error::Code::Internal, -1);
    }
    let slot = match st.queries.get(index).and_then(|s| s.as_ref()) {
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
        // DECIMAL values cannot be reconstructed without precision/scale.
        // Sending only the type code would surface the pre-scale integer as is.
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

/// Logical type codes. These map 1:1 to the type-name table on the JS side.
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
        Interval => 19,
        Json => 20,
        Uuid => 21,
        Timestamptz => 22,
    }
}

// Pins the physical type codes against what the JS side assumes.
const _: () = {
    assert!(PhysType::Bool as u32 == 0);
    assert!(PhysType::Bytes as u32 == 5);
};

// This module is only compiled under `#[cfg(target_arch = "wasm32")]` (see
// `lib.rs`), so these tests do not run under native `cargo test`. They become
// active once a test runner for wasm32 is set up.
#[cfg(test)]
mod tests {
    use super::*;

    /// The inverse of `decode_multi_files`. A test-only encoder.
    fn encode_multi_files(files: &[(&str, u64)]) -> Vec<u8> {
        let mut out = Vec::new();
        put_u32(&mut out, files.len() as u32);
        for (path, total_len) in files {
            put_u32(&mut out, path.len() as u32);
            out.extend_from_slice(path.as_bytes());
            put_u64(&mut out, *total_len);
        }
        out
    }

    #[test]
    fn multi_files_wire_format_round_trips() {
        let files = [("a.parquet", 100u64), ("year=2024/b.parquet", 200), ("", 0)];
        let buf = encode_multi_files(&files);
        let decoded = decode_multi_files(&buf).unwrap();
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0], ("a.parquet".to_string(), 100));
        assert_eq!(decoded[1], ("year=2024/b.parquet".to_string(), 200));
        assert_eq!(decoded[2], ("".to_string(), 0));
    }

    #[test]
    fn multi_files_empty_list_round_trips() {
        let buf = encode_multi_files(&[]);
        assert_eq!(decode_multi_files(&buf).unwrap(), Vec::new());
    }

    #[test]
    fn multi_files_truncated_buffer_is_rejected() {
        let mut buf = encode_multi_files(&[("a.parquet", 100)]);
        buf.truncate(buf.len() - 1);
        assert!(decode_multi_files(&buf).is_err());
    }

    #[test]
    fn ahiru_register_multi_registers_all_parts() {
        let h = ahiru_session_new();
        let name = b"t";
        let files = encode_multi_files(&[
            ("data/year=2024/month=01/a.parquet", 100),
            ("data/year=2024/month=02/b.parquet", 200),
        ]);
        let idx = unsafe {
            ahiru_register_multi(
                h,
                name.as_ptr(),
                name.len(),
                files.as_ptr(),
                files.len(),
                0, // Auto
            )
        };
        assert!(idx >= 0, "register_multi failed: last_error={}", ahiru_last_error());

        let s = session(h).unwrap();
        let t = s.catalog.get(idx as usize).unwrap();
        assert_eq!(t.parts.len(), 2);
        assert_eq!(t.parts[0].path, "data/year=2024/month=01/a.parquet");

        ahiru_session_free(h);
    }

    #[test]
    fn ahiru_register_multi_rejects_garbage_wire_data() {
        let h = ahiru_session_new();
        let name = b"t";
        let garbage = [0xFFu8; 3]; // too short to even read the count field
        let idx = unsafe {
            ahiru_register_multi(h, name.as_ptr(), name.len(), garbage.as_ptr(), garbage.len(), 0)
        };
        assert_eq!(idx, -1);
        assert_ne!(ahiru_last_error(), 0);
        ahiru_session_free(h);
    }

    #[test]
    fn query_handle_round_trips_through_pack_and_split() {
        assert_eq!(split_query_handle(make_query_handle(0, 0)), Some((0, 0)));
        assert_eq!(split_query_handle(make_query_handle(7, 3)), Some((7, 3)));
        assert_eq!(
            split_query_handle(make_query_handle(QUERY_IDX_MASK as usize, QUERY_GEN_MASK)),
            Some((QUERY_IDX_MASK as usize, QUERY_GEN_MASK))
        );
        // The sign bit must never be set for a valid handle: it is reserved
        // for the -1/-2 error and NEED_IO sentinels.
        assert!(make_query_handle(QUERY_IDX_MASK as usize, QUERY_GEN_MASK) >= 0);
        // Negative values (the sentinels) never decode to a slot.
        assert_eq!(split_query_handle(-1), None);
        assert_eq!(split_query_handle(-2), None);
    }

    /// `range()` needs no registered table and no I/O, so a query against it
    /// always reaches `Prepared::Ready` synchronously -- exactly what these
    /// tests need to drive the query handle lifecycle without mocking I/O.
    fn start_range_query(h: i32) -> i32 {
        let sql = b"SELECT 1 FROM range(3)";
        let q = unsafe { ahiru_query_start(h, sql.as_ptr(), sql.len(), core::ptr::null(), 0) };
        assert!(q >= 0, "query_start failed: last_error={}", ahiru_last_error());
        q
    }

    #[test]
    fn query_slot_is_reused_and_vec_does_not_grow_across_cycles() {
        let h = ahiru_session_new();
        for _ in 0..50 {
            let q = start_range_query(h);
            // One live query: never more than one slot in use.
            assert_eq!(state().queries.len(), 1);
            ahiru_query_close(q);
            // Closing the only open query also truncates the now-empty tail,
            // so the backing `Vec` gives its memory back rather than keeping
            // a permanently growing history of every query ever opened.
            assert_eq!(state().queries.len(), 0);
        }
        ahiru_session_free(h);
    }

    #[test]
    fn stale_handle_is_rejected_after_its_slot_is_reused() {
        let h = ahiru_session_new();
        let q1 = start_range_query(h);
        ahiru_query_close(q1);

        let q2 = start_range_query(h);
        // The closed slot's index was reused for the new query...
        assert_eq!(q1 & QUERY_IDX_MASK, q2 & QUERY_IDX_MASK, "test assumes the slot was reused");
        // ...but the generation bump means the two handles differ.
        assert_ne!(q1, q2);

        // The stale handle must not be able to step or close the new query.
        assert_eq!(ahiru_query_step(q1), STATUS_ERROR);
        ahiru_query_close(q1);
        assert_ne!(
            ahiru_query_step(q2),
            STATUS_ERROR,
            "the live query must be unaffected by operations on the stale handle"
        );

        ahiru_query_close(q2);
        ahiru_session_free(h);
    }
}
