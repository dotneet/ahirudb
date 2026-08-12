// The JS host layer for ahirudb. A dependency-free ES module (browser / Node 18+).
//
// The entire contract with the wasm side lives in crates/ahiru-core/src/abi.rs.
// When changing status values, the wire format, or error codes, change that file and this one together.
//
// This host is responsible for exactly three things:
//   1. the NEED_IO loop ... coalesce the byte ranges the engine asks for, fetch them in parallel, and hand them back
//   2. decoding the result buffer ... columnar little-endian representation -> JS values
//   3. assembling error messages (the table in errors.js)
//
// --- Handling wasm memory (important) ----------------------------------------
// `ahiru_alloc` / `ahiru_provide` may grow the wasm heap, and the moment it grows
// every existing TypedArray view detaches and becomes zero-length. That is a
// silently-corrupting class of bug, so the policy is fixed:
//
//   (a) Never hold a view onto `memory.buffer` across a wasm call.
//       Always re-take it with `new Uint8Array(memory.buffer)` right after the call.
//   (b) The buffer `ahiru_out_ptr()` points at is rebuilt by the next
//       `ahiru_query_step` / `ahiru_schema`. If a value is needed later, it must be
//       copied to the JS side before the next wasm call.
//   (c) Only when (b) above can be upheld, decode straight from the view and skip
//       the copy (`query()` packs into row objects on the spot so no copy is
//       needed; `stream()` hands batches to the caller so it always `.slice()`s).

import { AhiruError, Code, errorMessage } from './errors.js';

export { AhiruError, Code, errorMessage };

// --- Constants mirroring abi.rs ----------------------------------------------

const STATUS_BATCH_READY = 0;
const STATUS_NEED_IO = 1;
const STATUS_DONE = 2;
const STATUS_ERROR = 3;
/** Asks the host to decompress a codec that is not built in (GZIP / ZSTD). */
const STATUS_NEED_CODEC = 4;

/** The magic "AHR1" at the head of the result buffer. */
const RESULT_MAGIC = 0x41485231;

/** One element of `encode_io`: table:u32 + part:u32 + offset:u64 + len:u64. */
const IO_REQUEST_SIZE = 24;

/** One element of `encode_codec`: table:u32 + part:u32 + codec:u32 + offset:u64 + len:u32 + out_len:u32. */
const CODEC_REQUEST_SIZE = 28;

/** Parquet's Compression enum (parquet/mod.rs). Only the ones not built in are named. */
const CODEC_NAMES = {
  0: 'UNCOMPRESSED',
  1: 'SNAPPY',
  2: 'GZIP',
  3: 'LZO',
  4: 'BROTLI',
  5: 'LZ4',
  6: 'ZSTD',
  7: 'LZ4_RAW',
};
const CODEC_GZIP = 2;
const CODEC_ZSTD = 6;

/** PhysType numbers. Pinned by the `const _: ()` assertions in abi.rs. */
const PHYS_BOOL = 0;
const PHYS_I32 = 1;
const PHYS_I64 = 2;
const PHYS_F64 = 3;
const PHYS_I128 = 4;
const PHYS_BYTES = 5;

/** Logical type code (`ty_code`) -> type name. The index is the code itself. */
const TYPE_NAMES = [
  'NULL', 'BOOLEAN', 'TINYINT', 'SMALLINT', 'INTEGER', 'BIGINT', 'HUGEINT',
  'UTINYINT', 'USMALLINT', 'UINTEGER', 'UBIGINT', 'FLOAT', 'DOUBLE', 'DECIMAL',
  'VARCHAR', 'BLOB', 'DATE', 'TIME', 'TIMESTAMP', 'INTERVAL', 'JSON', 'UUID',
  'TIMESTAMPTZ',
];
const TY_DECIMAL = 13;
const TY_VARCHAR = 14;
const TY_INTERVAL = 19;
const TY_JSON = 20;
const TY_UUID = 21;

/** Adjacency threshold. Gaps narrower than this are coalesced, on the grounds that reading them is cheaper. */
const COALESCE_GAP = 1024 * 1024;

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder('utf-8');

// --- Time helpers ------------------------------------------------------------

/** Turns a TIMESTAMP (microseconds since the epoch, BigInt) into a `Date`. */
export function timestampToDate(micros) {
  // Date has millisecond precision, so the sub-millisecond remainder is dropped here.
  return new Date(Number(BigInt(micros) / 1000n));
}

/** Turns a DATE (days since the epoch, number) into a UTC `Date`. */
export function dateToDate(days) {
  return new Date(Number(days) * 86400000);
}

/**
 * Turns a TIMESTAMPTZ (UTC microseconds since the epoch, BigInt) into a `Date`.
 * The physical representation is identical to TIMESTAMP (this engine has no notion
 * of a session time zone, and values always denote a UTC instant), so this is an alias of `timestampToDate`.
 */
export const timestamptzToDate = timestampToDate;

/**
 * Opens the physical representation of an INTERVAL (months / days / microseconds
 * packed into a single i128) into `{ months, days, micros }`. This is the same
 * computation as `unpack_interval` in `vector::types`, so change both together.
 *
 * Keeping three separate components is the same model DuckDB / PostgreSQL use.
 * Months and days are not collapsed into microseconds because the length of
 * "one month" depends on the reference date (see the docs on `pack_interval`).
 * The three therefore cannot be returned as a single number.
 */
export function unpackInterval(packed) {
  const v = BigInt(packed);
  return {
    months: Number(BigInt.asIntN(32, v >> 96n)),
    days: Number(BigInt.asIntN(32, (v >> 64n) & 0xffffffffn)),
    micros: BigInt.asIntN(64, v),
  };
}

/** Turns 16 bytes into `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`. */
function formatUuid(bytes) {
  let s = '';
  for (let i = 0; i < 16; i++) {
    if (i === 4 || i === 6 || i === 8 || i === 10) s += '-';
    s += bytes[i].toString(16).padStart(2, '0');
  }
  return s;
}

// --- Range cache -------------------------------------------------------------

/**
 * An LRU cache of byte ranges keyed by `(source, offset, len)`.
 *
 * Only exact matches are considered. Searching for partial coverage would be
 * linear, and the engine asks for the same range of the same RowGroup every time,
 * so in practice this hits.
 */
export class MemoryCache {
  #map = new Map();
  #bytes = 0;

  constructor(maxBytes = 64 * 1024 * 1024) {
    this.maxBytes = maxBytes;
  }

  get size() {
    return this.#bytes;
  }

  get(key) {
    const v = this.#map.get(key);
    if (v === undefined) return undefined;
    // Map preserves insertion order, so deleting and reinserting is all LRU takes.
    this.#map.delete(key);
    this.#map.set(key, v);
    return v;
  }

  set(key, bytes) {
    if (bytes.byteLength > this.maxBytes) return; // give up on anything that does not fit on its own
    const old = this.#map.get(key);
    if (old !== undefined) {
      this.#map.delete(key);
      this.#bytes -= old.byteLength;
    }
    this.#map.set(key, bytes);
    this.#bytes += bytes.byteLength;
    for (const k of this.#map.keys()) {
      if (this.#bytes <= this.maxBytes) break;
      this.#bytes -= this.#map.get(k).byteLength;
      this.#map.delete(k);
    }
  }

  clear() {
    this.#map.clear();
    this.#bytes = 0;
  }
}

/** A cache that remembers nothing. `cache: "none"`. */
class NullCache {
  get() {
    return undefined;
  }
  set() {}
  clear() {}
}

function makeCache(spec, maxBytes) {
  if (spec === 'none' || spec === false || spec === null) return new NullCache();
  if (spec === undefined || spec === 'memory') return new MemoryCache(maxBytes);
  // "cache-api" is browser-only. Node has no `caches`, so fall back to memory.
  // (A Cache API version adds asynchronous I/O across Responses, so for now it is
  //  degraded to the same behavior. The correctness of the memory cache comes first.)
  if (spec === 'cache-api') return new MemoryCache(maxBytes);
  if (spec && typeof spec.get === 'function' && typeof spec.set === 'function') return spec;
  throw new TypeError(`unknown cache option: ${String(spec)}`);
}

// --- Format detection --------------------------------------------------------

/** The format argument of `ahiru_register_as`. 1:1 with `format_kind` in abi.rs. */
const FORMAT_CODES = { auto: 0, parquet: 1, csv: 2, tsv: 3, jsonl: 4 };

/**
 * Infers the format from the extension of the registered name.
 * A mirror of `format::FormatKind::detect`, so change both together.
 *
 * The engine performs the same inference, so this exists only to show, on the JS
 * side, what a name would be read as absent an explicit choice. The decision
 * itself is left to the wasm via Auto.
 */
export function detectFormat(name) {
  const path = String(name).split(/[?#]/)[0];
  const dot = path.lastIndexOf('.');
  if (dot < 0) return 'parquet';
  const ext = path.slice(dot + 1).toLowerCase();
  if (ext === 'csv') return 'csv';
  if (ext === 'tsv' || ext === 'tab') return 'tsv';
  if (ext === 'jsonl' || ext === 'ndjson') return 'jsonl';
  if (ext === 'json') return 'json';
  return 'parquet';
}

// --- Byte sources ------------------------------------------------------------

let sourceSeq = 0;

/**
 * Wraps whatever was registered in a common interface.
 * The only requirements are `{ key, size(), read(offset, len) }`.
 *
 * `size()` and `read()` are separate so that registration performs no I/O.
 * The total byte length is needed by `ahiru_register`, so it is deferred to the first query.
 */
function makeSource(spec, fetchImpl) {
  if (typeof spec === 'string' || spec instanceof URL) {
    return urlSource(String(spec), fetchImpl);
  }
  if (spec instanceof ArrayBuffer) return bytesSource(new Uint8Array(spec));
  if (ArrayBuffer.isView(spec)) {
    return bytesSource(new Uint8Array(spec.buffer, spec.byteOffset, spec.byteLength));
  }
  // Blob / File. Node 18+ has Blob too.
  if (spec && typeof spec.arrayBuffer === 'function' && typeof spec.size === 'number') {
    return blobSource(spec);
  }
  // A custom source (tests, OPFS, and so on).
  if (spec && typeof spec.read === 'function') {
    const key = spec.key ?? `custom:${++sourceSeq}`;
    const size = typeof spec.size === 'function' ? () => spec.size() : () => spec.size;
    return { key, size: async () => Number(await size()), read: (o, l) => spec.read(o, l) };
  }
  throw new TypeError('registerParquet: pass one of url / Uint8Array / ArrayBuffer / Blob');
}

function bytesSource(bytes) {
  const key = `bytes:${++sourceSeq}`;
  return {
    key,
    size: async () => bytes.byteLength,
    read: async (offset, len) => bytes.subarray(offset, offset + len),
    // Caching what is already in memory would hold it twice, so suppress that.
    cacheable: false,
  };
}

function blobSource(blob) {
  const key = `blob:${++sourceSeq}`;
  return {
    key,
    size: async () => blob.size,
    read: async (offset, len) =>
      new Uint8Array(await blob.slice(offset, offset + len).arrayBuffer()),
    cacheable: false,
  };
}

/** Network-layer failures are normalized to E504 too (so callers only need to look at code). */
async function request(doFetch, url, init) {
  try {
    return await doFetch(url, init);
  } catch (cause) {
    throw new AhiruError(Code.IO_FAILED, { detail: `fetch ${url} failed`, cause });
  }
}

function urlSource(url, fetchImpl) {
  const doFetch = fetchImpl ?? globalThis.fetch;
  if (typeof doFetch !== 'function') {
    throw new TypeError('no fetch available. Pass an implementation via init({ fetch })');
  }
  return {
    key: `url:${url}`,
    async size() {
      // HEAD first. Some servers do not support it, so fall back to Content-Range.
      try {
        const r = await doFetch(url, { method: 'HEAD' });
        const len = r.headers?.get('content-length');
        if (r.ok && len) return Number(len);
      } catch {
        /* HEAD unavailable. The range request below yields the total length. */
      }
      const r = await request(doFetch, url, { headers: { Range: 'bytes=0-0' } });
      const cr = r.headers?.get('content-range');
      const m = cr && /\/(\d+)\s*$/.exec(cr);
      if (m) return Number(m[1]);
      const len = r.headers?.get('content-length');
      if (r.ok && len) return Number(len);
      throw new AhiruError(Code.IO_FAILED, { detail: `cannot determine size of ${url}` });
    },
    async read(offset, len) {
      const r = await request(doFetch, url, {
        headers: { Range: `bytes=${offset}-${offset + len - 1}` },
      });
      if (!r.ok) {
        throw new AhiruError(Code.IO_FAILED, { detail: `${url} -> HTTP ${r.status}` });
      }
      const buf = new Uint8Array(await r.arrayBuffer());
      // Some servers ignore Range and return the whole thing. Slice out the requested window.
      if (r.status !== 206 && buf.byteLength > len) return buf.subarray(offset, offset + len);
      return buf;
    },
  };
}

// --- Range coalescing --------------------------------------------------------

/**
 * Coalesces nearby ranges into one.
 *
 * Fetching 900 KB once beats fetching 400 KB twice around a 100 KB gap.
 * That is exactly why the engine batches its requests per RowGroup, so do not
 * defeat that intent by issuing them one at a time (DESIGN.md §6).
 */
export function coalesceRanges(ranges, gap = COALESCE_GAP, totalLen = Infinity) {
  const sorted = ranges
    .map((r) => ({ offset: Number(r.offset), len: Number(r.len) }))
    .filter((r) => r.len > 0)
    .sort((a, b) => a.offset - b.offset);
  const out = [];
  for (const r of sorted) {
    const prev = out[out.length - 1];
    if (prev !== undefined && r.offset <= prev.offset + prev.len + gap) {
      const end = Math.max(prev.offset + prev.len, r.offset + r.len);
      prev.len = end - prev.offset;
      continue;
    }
    out.push({ offset: r.offset, len: r.len });
  }
  // Servers answer 416 for requests past the end of the file, so clamp them.
  for (const r of out) r.len = Math.min(r.len, Math.max(0, totalLen - r.offset));
  return out.filter((r) => r.len > 0);
}

// --- Wire format decoding ----------------------------------------------------

/**
 * `encode_io`: [count:u32][{table:u32, part:u32, offset:u64, len:u64}...]
 *
 * `part` says which file of a multi-file table (`ahiru_register_multi`) is meant.
 * Single-file registration (`ahiru_register`/`ahiru_register_as`) is always 0.
 * It must be passed straight back when calling `ahiru_provide` -- `table` alone
 * cannot uniquely identify the file the byte offsets are relative to.
 */
export function decodeIoRequests(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const n = dv.getUint32(0, true);
  const out = [];
  for (let i = 0; i < n; i++) {
    const p = 4 + i * IO_REQUEST_SIZE;
    out.push({
      table: dv.getUint32(p, true),
      part: dv.getUint32(p + 4, true),
      offset: Number(dv.getBigUint64(p + 8, true)),
      len: Number(dv.getBigUint64(p + 16, true)),
    });
  }
  return out;
}

/**
 * `encode_codec`: [count:u32][{table:u32, part:u32, codec:u32, offset:u64, len:u32, out_len:u32}...]
 * `part` means the same thing as in `decodeIoRequests`.
 */
export function decodeCodecRequests(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const n = dv.getUint32(0, true);
  const out = [];
  for (let i = 0; i < n; i++) {
    const p = 4 + i * CODEC_REQUEST_SIZE;
    out.push({
      table: dv.getUint32(p, true),
      part: dv.getUint32(p + 4, true),
      codec: dv.getUint32(p + 8, true),
      offset: Number(dv.getBigUint64(p + 12, true)),
      len: dv.getUint32(p + 20, true),
      outLen: dv.getUint32(p + 24, true),
    });
  }
  return out;
}

/** `encode_schema`: [n:u32][{ty:u32, phys:u32, precision:u32, scale:u32, name_len:u32, name}...] */
export function decodeSchema(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const n = dv.getUint32(0, true);
  const fields = [];
  let p = 4;
  for (let i = 0; i < n; i++) {
    const ty = dv.getUint32(p, true);
    const phys = dv.getUint32(p + 4, true);
    const precision = dv.getUint32(p + 8, true);
    const scale = dv.getUint32(p + 12, true);
    const nameLen = dv.getUint32(p + 16, true);
    p += 20;
    const name = textDecoder.decode(u8.subarray(p, p + nameLen));
    p += nameLen;
    fields.push({
      name,
      type: TYPE_NAMES[ty] ?? `TYPE_${ty}`,
      typeCode: ty,
      physType: phys,
      precision,
      scale,
    });
  }
  return fields;
}

/**
 * Serializes parameters. `[count:u32]` followed by `[tag:u8][payload]` per value.
 * tag: 0=NULL, 1=BOOL(1B), 2=I64(8B LE), 3=F64(8B LE), 4=BYTES(u32 len + bytes)
 */
export function encodeParams(params) {
  if (params === undefined || params === null || params.length === 0) return new Uint8Array(0);
  const parts = [];
  let len = 4;
  const push = (b) => {
    parts.push(b);
    len += b.length;
  };
  for (const v of params) {
    if (v === null || v === undefined) {
      push(Uint8Array.of(0));
    } else if (typeof v === 'boolean') {
      push(Uint8Array.of(1, v ? 1 : 0));
    } else if (typeof v === 'bigint' || typeof v === 'number') {
      const b = new Uint8Array(9);
      const dv = new DataView(b.buffer);
      // Safe integers and BigInt ride on I64; every other number rides on F64.
      if (typeof v === 'bigint' || Number.isSafeInteger(v)) {
        if (typeof v === 'bigint' && BigInt.asIntN(64, v) !== v) {
          throw new AhiruError(Code.VALUE_OUT_OF_RANGE, { detail: `${v} does not fit in i64` });
        }
        b[0] = 2;
        dv.setBigInt64(1, BigInt(v), true);
      } else {
        b[0] = 3;
        dv.setFloat64(1, v, true);
      }
      push(b);
    } else if (typeof v === 'string' || v instanceof Uint8Array || v instanceof ArrayBuffer) {
      const bytes =
        typeof v === 'string'
          ? textEncoder.encode(v)
          : v instanceof ArrayBuffer
            ? new Uint8Array(v)
            : v;
      const head = new Uint8Array(5);
      head[0] = 4;
      new DataView(head.buffer).setUint32(1, bytes.length, true);
      push(head);
      push(bytes);
    } else {
      // Implicitly converting a Date to microseconds would hide an off-by-a-factor mistake. Require it explicitly.
      throw new AhiruError(Code.UNSUPPORTED_FEATURE, {
        detail:
          `cannot bind ${Object.prototype.toString.call(v)}; ` +
          'use null / boolean / number / bigint / string / Uint8Array ' +
          '(pass TIMESTAMP as BigInt microseconds)',
      });
    }
  }
  const out = new Uint8Array(len);
  new DataView(out.buffer).setUint32(0, params.length, true);
  let p = 4;
  for (const part of parts) {
    out.set(part, p);
    p += part.length;
  }
  return out;
}

/**
 * Applying DECIMAL scale.
 *
 * Values arrive as "the integer before scaling". Dropping them into a `number`
 * rounds past 18 digits, so a **string** is returned. Where digits do not matter, call `Number(v)`.
 */
function scaleDecimal(unscaled, scale) {
  const v = BigInt(unscaled);
  if (scale === 0) return v.toString();
  const neg = v < 0n;
  const digits = (neg ? -v : v).toString().padStart(scale + 1, '0');
  const cut = digits.length - scale;
  return `${neg ? '-' : ''}${digits.slice(0, cut)}.${digits.slice(cut)}`;
}

/** The i-th bit of a bitmap (LSB-first). u64 little-endian = LSB-first per byte. */
function bitAt(bits, i) {
  return (bits[i >> 3] >> (i & 7)) & 1;
}

/**
 * If aligned, this is a view onto wasm memory; otherwise a view over a copy.
 * The wasm out buffer only guarantees 4-byte alignment per column, so F64 / I64
 * columns may not land on an 8-byte boundary.
 */
function viewOrCopy(Ctor, u8, byteOffset, count, copy) {
  const abs = u8.byteOffset + byteOffset;
  if (!copy && abs % Ctor.BYTES_PER_ELEMENT === 0) {
    return new Ctor(u8.buffer, abs, count);
  }
  const bytes = u8.slice(byteOffset, byteOffset + count * Ctor.BYTES_PER_ELEMENT);
  return new Ctor(bytes.buffer, 0, count);
}

function readI128(dv, off) {
  const lo = dv.getBigUint64(off, true);
  const hi = dv.getBigInt64(off + 8, true);
  return (hi << 64n) | lo;
}

/**
 * Decodes `encode_batch`.
 *
 * ```text
 * magic:u32 num_cols:u32 num_rows:u32
 * Per column: phys:u32 validity_len:u32 [validity] data_len:u32 [data]
 *             for Bytes only, offsets_len:u32 [offsets] precedes data
 * ```
 *
 * With `copy=false` the returned TypedArrays may point directly at wasm memory.
 * Use it only when they can be fully read before the next wasm call (policy (b)(c) at the top).
 */
export function decodeBatch(u8, schema, copy = true) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const magic = dv.getUint32(0, true);
  if (magic !== RESULT_MAGIC) {
    throw new AhiruError(Code.INTERNAL, {
      detail: `result magic mismatch: 0x${magic.toString(16)}`,
    });
  }
  const numCols = dv.getUint32(4, true);
  const numRows = dv.getUint32(8, true);
  let p = 12;
  const columns = [];

  for (let c = 0; c < numCols; c++) {
    const phys = dv.getUint32(p, true);
    const validityLen = dv.getUint32(p + 4, true);
    p += 8;
    let valid = null;
    if (validityLen > 0) {
      const bits = u8.subarray(p, p + validityLen);
      // Bitmaps are always copied. Expanding to one byte per row is easier for
      // callers to handle, and it stays small (rows/8 -> rows).
      valid = new Uint8Array(numRows);
      for (let i = 0; i < numRows; i++) valid[i] = bitAt(bits, i);
      p += validityLen;
    }

    const field = schema?.[c];
    const ty = field?.typeCode ?? -1;
    let values;

    if (phys === PHYS_BYTES) {
      const offsetsLen = dv.getUint32(p, true);
      p += 4;
      const offsets = viewOrCopy(Uint32Array, u8, p, offsetsLen / 4, false);
      p += offsetsLen;
      const dataLen = dv.getUint32(p, true);
      p += 4;
      const data = u8.subarray(p, p + dataLen);
      p += dataLen;
      values = new Array(numRows);
      for (let i = 0; i < numRows; i++) {
        const s = offsets[i];
        const e = offsets[i + 1];
        // VARCHAR / JSON are UTF-8 strings (JSON's physical representation is the
        // raw text before decoding, so it can be handed over as a string as is --
        // whether to `JSON.parse` is left to the caller); UUID is a hyphenated hex
        // string; everything else (BLOB) is returned as raw bytes.
        if (ty === TY_VARCHAR || ty === TY_JSON) {
          values[i] = textDecoder.decode(data.subarray(s, e));
        } else if (ty === TY_UUID) {
          values[i] = formatUuid(data.subarray(s, e));
        } else {
          values[i] = data.slice(s, e);
        }
      }
      columns.push({ name: field?.name ?? `col${c}`, type: field?.type ?? 'BLOB', typeCode: ty, physType: phys, values, valid });
      continue;
    }

    const dataLen = dv.getUint32(p, true);
    p += 4;
    const dataAt = p;
    p += dataLen;

    switch (phys) {
      case PHYS_BOOL: {
        // Bool is a bitmap too. Expand it into a 0/1 Uint8Array.
        const bits = u8.subarray(dataAt, dataAt + dataLen);
        values = new Uint8Array(numRows);
        for (let i = 0; i < numRows; i++) values[i] = bitAt(bits, i);
        break;
      }
      case PHYS_I32:
        values = viewOrCopy(Int32Array, u8, dataAt, numRows, copy);
        break;
      case PHYS_I64:
        values = viewOrCopy(BigInt64Array, u8, dataAt, numRows, copy);
        break;
      case PHYS_F64:
        values = viewOrCopy(Float64Array, u8, dataAt, numRows, copy);
        break;
      case PHYS_I128: {
        // There is no 128-bit TypedArray, so use an array of BigInt.
        values = new Array(numRows);
        for (let i = 0; i < numRows; i++) values[i] = readI128(dv, dataAt + i * 16);
        break;
      }
      default:
        throw new AhiruError(Code.INTERNAL, { detail: `unknown phys type ${phys}` });
    }
    if (ty === TY_DECIMAL) {
      // The pre-scale integer is not usable as is, so convert it to a string here.
      const scale = field?.scale ?? 0;
      const scaled = new Array(numRows);
      for (let i = 0; i < numRows; i++) scaled[i] = scaleDecimal(values[i], scale);
      values = scaled;
    } else if (ty === TY_INTERVAL) {
      // A packed i128 has no meaning as a number (months sit at the 2^96 place),
      // so it is opened into three components. Same treatment as DECIMAL: types
      // unusable in their physical representation are fixed up here.
      const parts = new Array(numRows);
      for (let i = 0; i < numRows; i++) parts[i] = unpackInterval(values[i]);
      values = parts;
    }
    columns.push({
      name: field?.name ?? `col${c}`,
      type: field?.type ?? TYPE_NAMES[ty] ?? 'UNKNOWN',
      typeCode: ty,
      physType: phys,
      values,
      valid,
    });
  }

  return new Batch(numRows, columns);
}

/** One batch (columnar). What `stream()` hands over. */
export class Batch {
  constructor(numRows, columns) {
    this.numRows = numRows;
    this.columns = columns;
    this.schema = columns.map((c) => ({ name: c.name, type: c.type }));
  }

  #index(k) {
    if (typeof k === 'number') return k;
    const i = this.columns.findIndex((c) => c.name === k);
    if (i < 0) throw new AhiruError(Code.COLUMN_NOT_FOUND, { detail: String(k) });
    return i;
  }

  /** Raw column values (TypedArray or Array). NULL positions hold a dummy value. */
  column(k) {
    return this.columns[this.#index(k)].values;
  }

  /** Whether a row is NULL. Honors the validity bitmap. */
  isNull(k, row) {
    const v = this.columns[this.#index(k)].valid;
    return v !== null && v[row] === 0;
  }

  /** The value at row i, column k. NULL is `null`. */
  get(k, row) {
    const c = this.columns[this.#index(k)];
    if (c.valid !== null && c.valid[row] === 0) return null;
    return c.physType === PHYS_BOOL ? c.values[row] === 1 : c.values[row];
  }

  /** Turns the batch into an array of plain objects. */
  toRows() {
    const rows = new Array(this.numRows);
    for (let r = 0; r < this.numRows; r++) {
      const o = {};
      for (const c of this.columns) {
        o[c.name] =
          c.valid !== null && c.valid[r] === 0
            ? null
            : c.physType === PHYS_BOOL
              ? c.values[r] === 1
              : c.values[r];
      }
      rows[r] = o;
    }
    return rows;
  }
}

// --- Codec delegation --------------------------------------------------------
//
// The core does not carry GZIP / ZSTD. Not carrying them is precisely why the core
// is small (DESIGN.md §6). GZIP goes to `DecompressionStream`, which browsers /
// Node have built in; ZSTD goes to a separate wasm module.

/** GZIP. Costing zero extra bytes is the whole point of this delegation. */
async function gunzip(bytes) {
  if (typeof DecompressionStream !== 'function') {
    throw new AhiruError(Code.UNSUPPORTED_CODEC, {
      detail: 'GZIP needs DecompressionStream (browser or Node 18+)',
    });
  }
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** The ZSTD decoder carried by a separate wasm module. Not loaded until first requested. */
class ZstdModule {
  #exports;

  constructor(instance) {
    this.#exports = instance.exports;
    for (const name of ['zstd_alloc', 'zstd_free', 'zstd_decompress']) {
      if (typeof this.#exports[name] !== 'function') {
        throw new AhiruError(Code.UNSUPPORTED_CODEC, {
          detail: `zstd module does not export ${name}`,
        });
      }
    }
  }

  static async load({ zstdUrl, zstdBinary, zstdModule, fetch: fetchImpl }) {
    let instance;
    if (zstdModule instanceof WebAssembly.Module) {
      instance = await WebAssembly.instantiate(zstdModule, {});
    } else if (zstdBinary || zstdUrl) {
      const bytes = zstdBinary
        ? ArrayBuffer.isView(zstdBinary)
          ? new Uint8Array(zstdBinary.buffer, zstdBinary.byteOffset, zstdBinary.byteLength)
          : new Uint8Array(zstdBinary)
        : await loadWasmBytes(zstdUrl, fetchImpl);
      ({ instance } = await WebAssembly.instantiate(bytes, {}));
    } else {
      throw new AhiruError(Code.UNSUPPORTED_CODEC, {
        detail:
          'ZSTD requires the side module: pass zstdUrl / zstdBinary / zstdModule to AhiruDB.init()',
      });
    }
    return new ZstdModule(instance);
  }

  decompress(src, outLen) {
    const e = this.#exports;
    const srcPtr = e.zstd_alloc(src.length);
    // Memory may grow on every alloc. Re-take the view each time (same policy as the core).
    new Uint8Array(e.memory.buffer).set(src, srcPtr);
    const dstPtr = e.zstd_alloc(outLen);
    const n = e.zstd_decompress(srcPtr, src.length, dstPtr, outLen);
    if (n < 0) {
      e.zstd_free(srcPtr, src.length);
      e.zstd_free(dstPtr, outLen);
      throw new AhiruError(Code.BAD_COMPRESSED_DATA, { detail: `zstd_decompress -> ${n}` });
    }
    // Left inside wasm it would detach on the next alloc, so return a copy.
    const out = new Uint8Array(e.memory.buffer, dstPtr, n).slice();
    e.zstd_free(srcPtr, src.length);
    e.zstd_free(dstPtr, outLen);
    return out;
  }
}

// --- wasm loading ------------------------------------------------------------

const isNode = typeof process !== 'undefined' && process.versions?.node !== undefined;

async function loadWasmBytes(wasmUrl, fetchImpl) {
  const s = String(wasmUrl);
  const remote = /^https?:/i.test(s);
  if (!remote && isNode) {
    const { readFile } = await import('node:fs/promises');
    const { fileURLToPath } = await import('node:url');
    return new Uint8Array(await readFile(s.startsWith('file:') ? fileURLToPath(s) : s));
  }
  const doFetch = fetchImpl ?? globalThis.fetch;
  const r = await doFetch(s);
  if (!r.ok) throw new AhiruError(Code.IO_FAILED, { detail: `${s} -> HTTP ${r.status}` });
  return new Uint8Array(await r.arrayBuffer());
}

// --- Main --------------------------------------------------------------------

export class AhiruDB {
  #exports;
  #memory;
  #session;
  #tables = new Map(); // name(lower) -> record
  #byIndex = new Map(); // wasm table index -> record
  #cache;
  /** When the cache was supplied from outside, close() must not clear it. */
  #ownsCache;
  #fetch;
  #memoryLimit;
  /** Cap on the fetched bytes retained for codec delegation. */
  #residentLimit;
  #closed = false;
  /** The ZSTD side module. Not loaded until the first NEED_CODEC. */
  #zstd = null;
  #zstdOptions;

  constructor(instance, options) {
    this.#zstdOptions = {
      zstdUrl: options.zstdUrl,
      zstdBinary: options.zstdBinary,
      zstdModule: options.zstdModule,
      fetch: options.fetch,
    };
    this.#exports = instance.exports;
    this.#memory = instance.exports.memory;
    this.#ownsCache = typeof options.cache !== 'object' || options.cache === null;
    this.#cache = makeCache(options.cache, options.cacheSize ?? 64 * 1024 * 1024);
    this.#residentLimit = options.cacheSize ?? 64 * 1024 * 1024;
    this.#fetch = options.fetch;
    this.#memoryLimit = options.memoryLimit ?? 0;
    this.#session = this.#exports.ahiru_session_new();
    if (this.#session < 0) throw new AhiruError(Code.INTERNAL, { detail: 'session_new failed' });
  }

  /**
   * Loads the wasm and opens a single session.
   *
   * `wasmUrl` may be a URL or a file path (on Node it is read as a file).
   * If you already have bytes or a compiled module, use `wasmBinary` / `wasmModule`.
   */
  static async init(options = {}) {
    const { wasmUrl, wasmBinary, wasmModule } = options;
    let instance;
    if (wasmModule instanceof WebAssembly.Module) {
      instance = await WebAssembly.instantiate(wasmModule, {});
    } else {
      const bytes = wasmBinary
        ? ArrayBuffer.isView(wasmBinary)
          ? new Uint8Array(wasmBinary.buffer, wasmBinary.byteOffset, wasmBinary.byteLength)
          : new Uint8Array(wasmBinary)
        : await loadWasmBytes(wasmUrl ?? 'ahiru-core.wasm', options.fetch);
      // The core has no imports at all (no_std, panic=abort).
      ({ instance } = await WebAssembly.instantiate(bytes, {}));
    }
    return new AhiruDB(instance, options);
  }

  /**
   * Registers a table. No I/O whatsoever happens here.
   * Fetching the total byte length and reading the footer / header are both deferred to the first query.
   *
   * Without `format`, the engine infers it from the **extension of the registered
   * name** (`format::FormatKind::detect`). When given, it takes precedence, so the
   * name does not need an extension.
   *
   * ```js
   * db.register('logs', url, { format: 'csv' });  // lets you write FROM logs
   * db.register('logs.csv', url);                 // FROM "logs.csv"
   * ```
   *
   * An explicit choice that disagrees with the extension is allowed. Decoupling the
   * name from how it is read is the purpose of this option; blocking that with a check would defeat it.
   */
  register(name, source, { format } = {}) {
    this.#assertOpen();
    if (typeof name !== 'string' || name.length === 0) {
      throw new TypeError('register: a table name is required');
    }
    let code = FORMAT_CODES.auto;
    if (format !== undefined && format !== null) {
      code = FORMAT_CODES[String(format).toLowerCase()];
      // Falling back to Auto on a typo would read it as Parquet, fail with BadMagic,
      // and obscure the cause. Reject unknown names here.
      if (code === undefined || code === FORMAT_CODES.auto) {
        throw new AhiruError(Code.UNSUPPORTED_FEATURE, {
          detail: `unknown format "${format}" (parquet / csv / tsv / jsonl)`,
        });
      }
    }
    const src = makeSource(source, this.#fetch);
    // A duplicate name replaces the previous one (the wasm-side catalog follows the same rule).
    this.#tables.set(name.toLowerCase(), {
      name,
      source: src,
      index: -1,
      size: -1,
      // What it will actually be read as. For Auto, infer it with the engine's rule and show that.
      format: code === FORMAT_CODES.auto ? detectFormat(name) : String(format).toLowerCase(),
      formatCode: code,
      // The retained copy of supplied bytes, and the ranges fetched so far (used once the copy is dropped).
      resident: [],
      fetched: [],
    });
    return this;
  }

  /** Alias for `register`. Accepts formats other than Parquet too. */
  registerParquet(name, source, options) {
    return this.register(name, source, options);
  }

  /** Materializes the entire result in memory and returns it. */
  async query(sql, params) {
    const rows = [];
    // copy=false: the views are only used until the values are moved into row
    // objects, which saves one copy of the column buffers.
    for await (const batch of this.#run(sql, params, false)) {
      // Views pointing directly at wasm memory are being read here, so move
      // everything into row objects before the next step (policy (c) at the top).
      for (const r of batch.toRows()) rows.push(r);
    }
    return rows;
  }

  /** Yields columnar batches in order. The entry point for not putting a large result in one array. */
  stream(sql, params) {
    return this.#run(sql, params, true);
  }

  /** Closes the session. Later calls are errors. */
  close() {
    if (this.#closed) return;
    this.#closed = true;
    this.#exports.ahiru_session_free(this.#session);
    this.#tables.clear();
    this.#byIndex.clear();
    if (this.#ownsCache) this.#cache.clear();
  }

  /** How many bytes the wasm heap currently holds. */
  get heapUsed() {
    return this.#exports.ahiru_heap_used();
  }

  // --- Execution loop -------------------------------------------------------

  async *#run(sql, params, copy) {
    this.#assertOpen();
    await this.#bindTables(sql);

    const q = await this.#start(sql, params);
    try {
      const schema = this.#readSchema(q, sql);
      let lastSignature = null;
      for (;;) {
        const status = this.#exports.ahiru_query_step(q);
        this.#checkMemory(sql);
        if (status === STATUS_BATCH_READY) {
          const out = this.#out();
          yield decodeBatch(out, schema, copy);
          continue;
        }
        if (status === STATUS_NEED_IO) {
          lastSignature = await this.#pump(decodeIoRequests(this.#out()), lastSignature, sql);
          continue;
        }
        if (status === STATUS_NEED_CODEC) {
          await this.#decompress(decodeCodecRequests(this.#out()), sql);
          continue;
        }
        if (status === STATUS_DONE) return;
        throw this.#lastError(sql);
      }
    } finally {
      this.#exports.ahiru_query_close(q);
    }
  }

  /**
   * Starts a query. If the footer is not fetched yet it returns `-2`, so satisfy the request and retry.
   */
  async #start(sql, params) {
    const bytes = textEncoder.encode(sql);
    const pbytes = encodeParams(params);
    let lastSignature = null;
    for (;;) {
      const e = this.#exports;
      // The core has no clock, so the query start time is passed in here for
      // CURRENT_DATE/CURRENT_TIMESTAMP/now() (DESIGN.md §2).
      e.ahiru_set_now(this.#session, BigInt(Date.now()) * 1000n);
      const ptr = e.ahiru_alloc(bytes.length);
      const pptr = pbytes.length > 0 ? e.ahiru_alloc(pbytes.length) : 0;
      // alloc may grow memory, so re-take the view right before writing.
      const mem = new Uint8Array(this.#memory.buffer);
      mem.set(bytes, ptr);
      if (pptr !== 0) mem.set(pbytes, pptr);
      const h = e.ahiru_query_start(this.#session, ptr, bytes.length, pptr, pbytes.length);
      e.ahiru_free(ptr, bytes.length);
      if (pptr !== 0) e.ahiru_free(pptr, pbytes.length);
      if (h >= 0) return h;
      if (h !== -2) throw this.#lastError(sql);
      // -2: not enough bytes to read the footer.
      lastSignature = await this.#pump(decodeIoRequests(this.#out()), lastSignature, sql);
    }
  }

  /**
   * Satisfies codec decompression requests.
   *
   * The compressed blocks should already have been fetched by the preceding NEED_IO,
   * so nothing new is fetched. If they are not on hand, report that the engine's request is wrong rather than silently fetching.
   */
  async #decompress(requests, sql) {
    for (const req of requests) {
      if (req.codec !== CODEC_GZIP && req.codec !== CODEC_ZSTD) {
        throw new AhiruError(Code.UNSUPPORTED_CODEC, {
          sql,
          detail: `${CODEC_NAMES[req.codec] ?? `codec ${req.codec}`} is not handled by the host`,
        });
      }
      // If there is even one ZSTD, load it once before going parallel.
      if (req.codec === CODEC_ZSTD) this.#zstd ??= await ZstdModule.load(this.#zstdOptions);
    }

    // Decompression is independent, so run it in parallel (GZIP is an async stream).
    const outputs = await Promise.all(
      requests.map(async (req) => {
        const rec = this.#byIndex.get(req.table);
        if (rec === undefined) {
          throw new AhiruError(Code.INTERNAL, { sql, detail: `unknown table index ${req.table}` });
        }
        const src = await this.#bytesAt(rec, req.offset, req.len, sql);
        const out =
          req.codec === CODEC_GZIP ? await gunzip(src) : this.#zstd.decompress(src, req.outLen);
        if (out.length !== req.outLen) {
          throw new AhiruError(Code.BAD_COMPRESSED_DATA, {
            sql,
            detail: `expected ${req.outLen} bytes, got ${out.length}`,
          });
        }
        return out;
      }),
    );

    // Handing back to wasm is sequential, because memory moves on every alloc.
    for (let i = 0; i < requests.length; i++) this.#provideCodec(requests[i], outputs[i], sql);
    this.#checkMemory(sql);
  }

  /** Returns decompressed blocks to wasm. */
  #provideCodec(req, bytes, sql) {
    const e = this.#exports;
    const ptr = e.ahiru_alloc(bytes.length);
    if (ptr === 0 && bytes.length > 0) throw new AhiruError(Code.OOM, { sql });
    new Uint8Array(this.#memory.buffer).set(bytes, ptr);
    const rc = e.ahiru_provide_codec(
      this.#session,
      req.table,
      req.part,
      BigInt(req.offset),
      req.len,
      ptr,
      bytes.length,
    );
    e.ahiru_free(ptr, bytes.length);
    if (rc !== 0) throw this.#lastError(sql);
  }

  /**
   * Slices the bytes of a compressed block out of what is on hand.
   *
   * The normal case is that they are in the copy retained by the preceding NEED_IO.
   * Only what was discarded when that copy overflowed gets refetched (which is not
   * I/O if it is still cached). A request for a range never fetched at all is an
   * engine-side inconsistency, so it is reported rather than silently fetched.
   */
  async #bytesAt(rec, offset, len, sql) {
    for (const c of rec.resident) {
      if (c.offset <= offset && offset + len <= c.offset + c.bytes.length) {
        return c.bytes.subarray(offset - c.offset, offset - c.offset + len);
      }
    }
    // Memory / Blob sources need no I/O to slice, so they keep no retained copy.
    if (rec.source.cacheable === false) return rec.source.read(offset, len);
    const everFetched = rec.fetched.some(
      (r) => r.offset <= offset && offset + len <= r.offset + r.len,
    );
    if (everFetched) return this.#read(rec, offset, len);
    throw new AhiruError(Code.INTERNAL, {
      sql,
      detail:
        `codec request for bytes we never fetched: table ${rec.name} [${offset}, ${offset + len})` +
        ' — the engine should ask for these via NEED_IO first',
    });
  }

  /** Reads the output schema. `ahiru_schema` rebuilds the out buffer, so call it before step. */
  #readSchema(q, sql) {
    // A negative value means a bad handle. 0 means "zero columns", so keep them distinct.
    const len = this.#exports.ahiru_schema(q);
    if (len < 0) throw this.#lastError(sql);
    if (len === 0) return [];
    return decodeSchema(this.#out());
  }

  /**
   * Satisfies I/O requests. Coalesce -> fetch in parallel -> `ahiru_provide`.
   * Returns a signature for comparison on the next round.
   */
  async #pump(requests, lastSignature, sql) {
    const signature = requests.map((r) => `${r.table}.${r.part}/${r.offset}+${r.len}`).join(',');

    // `table` alone cannot tell which file of a multi-file table is meant (offsets
    // live in a separate space per file), so requests are grouped by the composite
    // key `table:part`. Single-file registration always has part=0, so this extends
    // safely to multiple files while behaving exactly as before.
    const jobs = [];
    for (const [key, list] of groupBy(requests, (r) => `${r.table}:${r.part}`)) {
      const [table, part] = key.split(':').map(Number);
      const rec = this.#byIndex.get(table);
      if (rec === undefined) {
        throw new AhiruError(Code.INTERNAL, { sql, detail: `unknown table index ${table}` });
      }
      for (const r of coalesceRanges(list, COALESCE_GAP, rec.size)) {
        jobs.push({ rec, table, part, offset: r.offset, len: r.len });
      }
    }

    // Coalesced ranges are fetched in parallel, to avoid stacking round trips.
    const buffers = await Promise.all(jobs.map((j) => this.#read(j.rec, j.offset, j.len)));

    let provided = 0;
    for (let i = 0; i < jobs.length; i++) {
      const { rec, table, part, offset } = jobs[i];
      provided += this.#provide(table, part, offset, buffers[i], sql);
      // Only remote sources keep a retained copy. When codec delegation asks for a
      // compressed block, slicing it out of here avoids a refetch (memory sources just slice).
      //
      // The JS-side registration API (`register`/`registerParquet`) is one table =
      // one file for now, so `part` is always 0, which is consistent with `rec`
      // holding a single retained copy. When multi-file registration
      // (`ahiru_register_multi`) becomes usable from JS, this copy must be split per file.
      if (rec.source.cacheable !== false && !rec.resident.some((c) => c.offset === offset)) {
        rec.resident.push({ offset, bytes: buffers[i] });
        rec.fetched.push({ offset, len: buffers[i].byteLength });
        // Drop oldest first, so this does not grow without bound. If a dropped range
        // is needed later it is refetched from the cache (`#bytesAt`).
        let held = rec.resident.reduce((a, c) => a + c.bytes.byteLength, 0);
        while (held > this.#residentLimit && rec.resident.length > 1) {
          held -= rec.resident.shift().bytes.byteLength;
        }
      }
    }
    this.#checkMemory(sql);

    // Livelock detection: the same request repeats and not one byte has been added.
    if (provided === 0 && signature === lastSignature) {
      throw new AhiruError(Code.IO_FAILED, {
        sql,
        detail: `no progress for ranges [${signature}]`,
      });
    }
    return signature;
  }

  /** Reads a byte range through the cache. */
  async #read(rec, offset, len) {
    const key = `${rec.source.key}:${offset}:${len}`;
    const cacheable = rec.source.cacheable !== false;
    if (cacheable) {
      const hit = this.#cache.get(key);
      if (hit !== undefined) return hit;
    }
    const bytes = await rec.source.read(offset, len);
    const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    if (cacheable) this.#cache.set(key, u8);
    return u8;
  }

  /** Hands fetched bytes to wasm. Returns the length handed over. */
  #provide(table, part, offset, bytes, sql) {
    if (bytes.byteLength === 0) return 0;
    const e = this.#exports;
    const ptr = e.ahiru_alloc(bytes.byteLength);
    if (ptr === 0) throw new AhiruError(Code.OOM, { sql });
    // alloc may have grown memory, so always re-take the view here.
    new Uint8Array(this.#memory.buffer).set(bytes, ptr);
    const rc = e.ahiru_provide(this.#session, table, part, BigInt(offset), ptr, bytes.byteLength);
    e.ahiru_free(ptr, bytes.byteLength);
    if (rc !== 0) throw this.#lastError(sql);
    return bytes.byteLength;
  }

  /**
   * Registers only the tables the SQL actually references.
   *
   * Registration needs the total byte length (= one HEAD round trip for a URL), so
   * identifiers are collected to narrow it down and avoid round trips for unused tables.
   */
  async #bindTables(sql) {
    const mentioned = new Set();
    const add = (s) => {
      mentioned.add(s.toLowerCase());
      // For dotted names like `t.id`, take both the whole thing and each part as
      // candidates (a table name itself may contain a dot, as in `basic.csv`).
      if (s.includes('.')) for (const part of s.split('.')) mentioned.add(part.toLowerCase());
    };
    for (const m of sql.matchAll(/[A-Za-z_][A-Za-z0-9_$.]*|'([^']*)'|"([^"]*)"/g)) {
      add(m[1] ?? m[2] ?? m[0]);
    }
    // `FROM parquet('https://...')` is contracted to register the path itself as the
    // table name (see resolve_from in plan/bind.rs).
    // The function is named parquet, but an extension such as .csv is read as CSV.
    for (const m of sql.matchAll(/parquet\(\s*'([^']*)'/gi)) {
      const path = m[1];
      if (!this.#tables.has(path.toLowerCase())) this.register(path, path);
      add(path);
    }

    for (const [key, rec] of this.#tables) {
      if (rec.index >= 0 || !mentioned.has(key)) continue;
      rec.size = await rec.source.size();
      const e = this.#exports;
      const name = textEncoder.encode(rec.name);
      const ptr = e.ahiru_alloc(name.length);
      new Uint8Array(this.#memory.buffer).set(name, ptr);
      // Older cores have no ahiru_register_as. For Auto the 4-argument version is
      // equivalent, but silently ignoring an explicit choice would read it as another format, so fail.
      const hasRegisterAs = typeof e.ahiru_register_as === 'function';
      if (!hasRegisterAs && rec.formatCode !== FORMAT_CODES.auto) {
        e.ahiru_free(ptr, name.length);
        throw new AhiruError(Code.UNSUPPORTED_FEATURE, {
          detail: `this wasm core has no ahiru_register_as; format="${rec.format}" cannot be honoured`,
        });
      }
      const idx = hasRegisterAs
        ? e.ahiru_register_as(this.#session, ptr, name.length, BigInt(rec.size), rec.formatCode)
        : e.ahiru_register(this.#session, ptr, name.length, BigInt(rec.size));
      e.ahiru_free(ptr, name.length);
      if (idx < 0) throw this.#lastError();
      rec.index = idx;
      this.#byIndex.set(idx, rec);
    }
  }

  // --- Odds and ends --------------------------------------------------------

  /** A view of the current out buffer. Valid only until the next wasm call. */
  #out() {
    const ptr = this.#exports.ahiru_out_ptr();
    const len = this.#exports.ahiru_out_len();
    return new Uint8Array(this.#memory.buffer, ptr, len);
  }

  #lastError(sql) {
    const code = this.#exports.ahiru_last_error();
    return new AhiruError(code, { sql });
  }

  #checkMemory(sql) {
    if (this.#memoryLimit > 0 && this.#exports.ahiru_heap_used() > this.#memoryLimit) {
      throw new AhiruError(Code.LIMIT_EXCEEDED, {
        sql,
        detail: `wasm heap ${this.#exports.ahiru_heap_used()} > memoryLimit ${this.#memoryLimit}`,
      });
    }
  }

  #assertOpen() {
    if (this.#closed) throw new AhiruError(Code.INTERNAL, { detail: 'database is closed' });
  }
}

function groupBy(items, keyOf) {
  const m = new Map();
  for (const it of items) {
    const k = keyOf(it);
    const list = m.get(k);
    if (list === undefined) m.set(k, [it]);
    else list.push(it);
  }
  return m;
}

export default AhiruDB;
