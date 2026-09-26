// The JS host layer for ahirudb. A dependency-free ES module (browser / Node 18+).
//
// The entire contract with the wasm side lives in crates/ahiru-core/src/abi.rs.
// When changing status values, the wire format, or error codes, change that file and this one together.
//
// This host is responsible for:
//   1. the NEED_IO loop ... coalesce the byte ranges the engine asks for, fetch them in parallel, and hand them back
//   2. decoding the result buffer ... columnar little-endian representation -> JS values
//   3. assembling error messages (the table in errors.js)
//   4. binding tables ... every register() is declared to the engine without I/O; the
//      engine asks for the lengths of the tables it resolves and names the SQL
//      string-literal paths nobody registered. The host never parses SQL itself.
//   5. replacing the wasm instance after a trap (see `isWasmTrap`)
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

/** Maximum decompressed bytes accepted for one Parquet page (mirrors codec.rs). */
const MAX_DECOMPRESSED_PAGE_BYTES = 256 * 1024 * 1024;

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
const DEFAULT_CACHE_SIZE = 64 * 1024 * 1024;

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder('utf-8', { fatal: true });

/**
 * Validate byte limits before they reach arithmetic or eviction comparisons.
 * Number.isSafeInteger is intentional: these values are used as exact byte
 * counts, so accepting NaN, fractions, or values above MAX_SAFE_INTEGER would
 * make the limit either ineffective or ambiguous.
 */
function requireNonNegativeSafeInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new AhiruError(Code.VALUE_OUT_OF_RANGE, {
      detail: `${label} must be a non-negative safe integer`,
    });
  }
  return value;
}

function requireSafeInteger(value, label) {
  if (!Number.isSafeInteger(value)) {
    throw new AhiruError(Code.VALUE_OUT_OF_RANGE, {
      detail: `${label} must be a safe integer`,
    });
  }
  return value;
}

/** Normalize the numeric form accepted by BigInt while rejecting lossy numbers. */
function toExactIntegerBigInt(value, label) {
  if (typeof value === 'number' && !Number.isSafeInteger(value)) {
    throw new AhiruError(Code.VALUE_OUT_OF_RANGE, {
      detail: `${label} must be a safe integer when passed as a number`,
    });
  }
  try {
    return BigInt(value);
  } catch {
    throw new AhiruError(Code.VALUE_OUT_OF_RANGE, { detail: `${label} must be an integer` });
  }
}

const MAX_DATE_MILLIS = 8_640_000_000_000_000n;

function dateFromMillis(millis, label) {
  if (millis < -MAX_DATE_MILLIS || millis > MAX_DATE_MILLIS) {
    throw new AhiruError(Code.VALUE_OUT_OF_RANGE, { detail: `${label} is outside Date's range` });
  }
  return new Date(Number(millis));
}

/**
 * Decode ABI text strictly; replacement characters would hide a corrupt wire buffer.
 *
 * `code` says whose fault a failure is. Structural text (schema field names) can
 * only be malformed if the wire buffer itself is, which is an engine bug (E900).
 * A VARCHAR/JSON *value*, though, is whatever bytes the source file holds -- a
 * Parquet writer is free to put non-UTF-8 in a BYTE_ARRAY column -- so that is a
 * data error (E108) and must not be reported as an internal one.
 */
function decodeUtf8(bytes, what, code = Code.INTERNAL) {
  try {
    return textDecoder.decode(bytes);
  } catch {
    throw new AhiruError(code, { detail: `invalid UTF-8 ${what}` });
  }
}

// --- Time helpers ------------------------------------------------------------

/** Turns a TIMESTAMP (microseconds since the epoch, BigInt) into a `Date`. */
export function timestampToDate(micros) {
  // Date has millisecond precision, so the sub-millisecond remainder is dropped
  // here. BigInt division truncates toward zero, which for a negative instant
  // would round *up* (-999us -> 0ms, i.e. forward in time); floor instead so the
  // result is always the millisecond that contains the instant.
  const exact = toExactIntegerBigInt(micros, 'timestamp microseconds');
  let millis = exact / 1000n;
  if (exact < 0n && exact % 1000n !== 0n) {
    millis -= 1n;
  }
  return dateFromMillis(millis, 'timestamp');
}

/** Turns a DATE (days since the epoch, number) into a UTC `Date`. */
export function dateToDate(days) {
  const exactDays = requireSafeInteger(days, 'date days');
  return dateFromMillis(BigInt(exactDays) * 86400000n, 'date');
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
  const v = toExactIntegerBigInt(packed, 'packed interval');
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
  #maxBytes;

  constructor(maxBytes = DEFAULT_CACHE_SIZE) {
    this.maxBytes = maxBytes;
  }

  get maxBytes() {
    return this.#maxBytes;
  }

  set maxBytes(value) {
    this.#maxBytes = requireNonNegativeSafeInteger(value, 'MemoryCache maxBytes');
    // A public limit change takes effect immediately. Without this eviction,
    // lowering the limit leaves the cache over budget until some later write
    // happens (and forever for a read-only workload).
    for (const k of this.#map.keys()) {
      if (this.#bytes <= this.#maxBytes) break;
      this.#bytes -= this.#map.get(k).byteLength;
      this.#map.delete(k);
    }
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
const FORMAT_CODES = { auto: 0, parquet: 1, csv: 2, tsv: 3, jsonl: 4, json: 5 };

/**
 * Flag OR-ed into `ahiru_register_as`'s format argument for a path a SQL string
 * literal referenced (`FORMAT_PATH` in abi.rs): registered case-sensitively.
 */
const FORMAT_PATH = 0x100;

/** `ahiru_register_as` total length meaning "not known yet" (`catalog::SIZE_UNKNOWN`). */
const SIZE_UNKNOWN = 0xffff_ffff_ffff_ffffn;

/** `ahiru_copy_result` flag: gzip the bytes before handing them to `onCopy` (`COPY_GZIP`). */
const COPY_GZIP = 1;

/** `ahiru_query_start`'s "register these paths and call me again" (`START_NEED_TABLES`). */
const START_NEED_TABLES = -3;

/**
 * The table-function family a path reported by `START_NEED_TABLES` came from,
 * by the `format_kind` code the SQL asked for. A bare `FROM 'x.csv'` reports the
 * family its extension implies. Passed to `sqlUrlPolicy` as `functionName`.
 */
const FUNCTION_FAMILY = ['parquet', 'parquet', 'read_csv', 'read_csv', 'read_json', 'read_json'];

/**
 * The format to register a SQL-referenced path with, from the `format_kind` code
 * the SQL asked for. `parquet('x.csv')` keeps the historical "the extension
 * decides" behaviour (Auto); `read_csv('x.tsv')` / `read_json('x.jsonl')` pick
 * the variant the extension names.
 */
function formatForSqlPath(code, path) {
  const detected = detectFormat(path);
  switch (code) {
    case FORMAT_CODES.csv:
      return detected === 'tsv' ? 'tsv' : 'csv';
    case FORMAT_CODES.json:
      return detected === 'jsonl' ? 'jsonl' : 'json';
    case FORMAT_CODES.tsv:
      return 'tsv';
    case FORMAT_CODES.jsonl:
      return 'jsonl';
    default:
      return undefined;
  }
}

/**
 * Resolves a path from SQL the way `fetch()` would: against the page's base URL
 * when there is one (a browser), verbatim otherwise. A protocol-relative
 * `//host/x` or a relative `x.parquet` is a network request to *some* origin in
 * a browser, so the URL policy has to see where it really goes.
 */
function resolveSqlPath(path) {
  const base = globalThis.document?.baseURI ?? globalThis.location?.href;
  if (base === undefined) return path;
  try {
    return new URL(path, base).href;
  } catch {
    return path;
  }
}

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
function makeSource(spec, fetchImpl, options = {}) {
  if (typeof spec === 'string' || spec instanceof URL) {
    return urlSource(String(spec), fetchImpl, options);
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
    return {
      key,
      size: async () => toSafeSize(await size(), 'ByteSource size'),
      read: (o, l) => spec.read(o, l),
      // The caller's `read()` may return a view onto memory it still owns (e.g.
      // `buf.subarray(...)`), unlike the bytes this host produces itself from
      // `fetch`/`Blob`. #read() copies before retaining anything from a source
      // marked this way (cache entries, the resident copy used by codec delegation).
      untrusted: true,
    };
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
    size: async () => toSafeSize(blob.size, 'Blob size'),
    read: async (offset, len) =>
      new Uint8Array(await blob.slice(offset, offset + len).arrayBuffer()),
    cacheable: false,
  };
}

/**
 * Strips whatever should never end up in a log or thrown error: userinfo and the
 * query string (and any fragment). Presigned S3 URLs and API tokens commonly ride
 * in the query string, so an error message must never carry one verbatim.
 */
function redactUrl(url) {
  try {
    const u = new URL(String(url));
    u.username = '';
    u.password = '';
    u.search = '';
    u.hash = '';
    return u.toString();
  } catch {
    // Not a parseable absolute URL (a relative path, or something else). Still
    // drop anything that looks like a query string / fragment as a fallback.
    return String(url).split(/[?#]/)[0];
  }
}

/** Validates a source length before it becomes a BigInt or a range clamp. */
function toSafeSize(value, what) {
  if (typeof value === 'string' && !/^\s*\d+\s*$/.test(value)) {
    throw new AhiruError(Code.VALUE_OUT_OF_RANGE, {
      detail: `${what} ${value} is not a non-negative integer`,
    });
  }
  let n;
  try {
    n = Number(value);
  } catch {
    n = NaN;
  }
  if (!Number.isSafeInteger(n) || n < 0) {
    throw new AhiruError(Code.VALUE_OUT_OF_RANGE, {
      detail: `${what} ${String(value)} is not a non-negative safe integer`,
    });
  }
  return n;
}

/** Network-layer failures are normalized to E504 too (so callers only need to look at code). */
async function request(doFetch, url, init) {
  try {
    return await doFetch(url, init);
  } catch (cause) {
    throw new AhiruError(Code.IO_FAILED, { detail: `fetch ${redactUrl(url)} failed`, cause });
  }
}

/** Normalize failures while consuming a successful response body as well as fetch failures. */
async function responseBytes(response, url) {
  try {
    return new Uint8Array(await response.arrayBuffer());
  } catch (cause) {
    throw new AhiruError(Code.IO_FAILED, {
      detail: `read ${redactUrl(url)} response failed`,
      cause,
    });
  }
}

function urlSource(url, fetchImpl, { rejectRedirects = false } = {}) {
  const doFetch = fetchImpl ?? globalThis.fetch;
  if (typeof doFetch !== 'function') {
    throw new TypeError('no fetch available. Pass an implementation via init({ fetch })');
  }
  return {
    key: `url:${url}`,
    async size() {
      // HEAD first. Some servers do not support it, so fall back to Content-Range.
      let head = null;
      try {
        head = await doFetch(url, {
          method: 'HEAD',
          ...(rejectRedirects ? { redirect: 'error' } : {}),
        });
      } catch {
        /* HEAD unavailable. The range request below yields the total length. */
      }
      const headLen = head?.headers?.get('content-length');
      // Keep validation outside the network-error catch. A malformed successful
      // HEAD response is a server protocol error, not a reason to silently try a
      // second request and potentially hide VALUE_OUT_OF_RANGE.
      if (head?.ok && headLen) return toSafeSize(headLen, 'HTTP Content-Length');
      const r = await request(doFetch, url, {
        headers: { Range: 'bytes=0-0' },
        ...(rejectRedirects ? { redirect: 'error' } : {}),
      });
      if (!r.ok) {
        throw new AhiruError(Code.IO_FAILED, { detail: `${redactUrl(url)} -> HTTP ${r.status}` });
      }
      const cr = r.headers?.get('content-range');
      const m = cr && /\/(\d+)\s*$/.exec(cr);
      if (m) return toSafeSize(m[1], 'HTTP Content-Range length');
      if (r.status === 200) {
        const len = r.headers?.get('content-length');
        if (len) return toSafeSize(len, 'HTTP Content-Length');
      }
      throw new AhiruError(Code.IO_FAILED, { detail: `cannot determine size of ${redactUrl(url)}` });
    },
    async read(offset, len) {
      const r = await request(doFetch, url, {
        headers: { Range: `bytes=${offset}-${offset + len - 1}` },
        ...(rejectRedirects ? { redirect: 'error' } : {}),
      });
      if (!r.ok) {
        throw new AhiruError(Code.IO_FAILED, { detail: `${redactUrl(url)} -> HTTP ${r.status}` });
      }
      const buf = await responseBytes(r, url);
      if (r.status === 206) {
        // A conforming server's Content-Range says exactly which bytes these are.
        // Trust that over just the body length -- otherwise a server that ignores
        // Range but still (wrongly) answers 206 with the whole file would have its
        // bytes inserted at the requested offset, and the caller would read garbage
        // without any error ever being raised.
        const cr = r.headers?.get('content-range');
        const m = cr && /^bytes\s+(\d+)-(\d+)\/(?:\d+|\*)\s*$/i.exec(cr);
        if (!m) {
          // No Content-Range. A body of the requested length starting at offset 0
          // is the only case we can trust without a range header (the request was
          // for the prefix). Anything else — including a prefix-sized body for a
          // non-zero offset — would silently feed the wrong bytes to wasm.
          if (offset === 0 && buf.byteLength === len) return buf;
          throw new AhiruError(Code.IO_FAILED, {
            detail:
              `${redactUrl(url)}: 206 response has no Content-Range` +
              (offset === 0
                ? ` and an unexpected body length (${buf.byteLength} vs the ${len} requested)`
                : ` (cannot verify that the body is offset ${offset})`),
          });
        }
        const start = Number(m[1]);
        const end = Number(m[2]);
        // Slice whenever Content-Range proves the requested window is inside
        // the body. A wider 206 (start matches, body longer than `len`) used
        // to be returned unsliced and then rejected as an over-long read.
        if (start <= offset && offset + len <= start + buf.byteLength) {
          return buf.subarray(offset - start, offset - start + len);
        }
        throw new AhiruError(Code.IO_FAILED, {
          detail:
            `${redactUrl(url)}: 206 response covers [${start}, ${end}], which does not ` +
            `contain the requested [${offset}, ${offset + len})`,
        });
      }
      // Some servers ignore Range and return 200 with the whole thing. Slice out the requested window.
      if (buf.byteLength >= offset + len) return buf.slice(offset, offset + len);
      // A body of the requested length at offset 0 is the prefix we asked for.
      // The same length at a non-zero offset is *not* trustworthy: a Range-unaware
      // server often returns the first `len` bytes of the file (or an error page)
      // and would silently poison the engine with the wrong window.
      if (buf.byteLength === len && offset === 0) return buf;
      throw new AhiruError(Code.IO_FAILED, {
        detail:
          `${redactUrl(url)}: 200 response length ${buf.byteLength} does not cover ` +
          `the requested [${offset}, ${offset + len})`,
      });
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
  requireNonNegativeSafeInteger(gap, 'coalescing gap');
  const sorted = ranges
    .map((r) => {
      const normalized = { offset: Number(r.offset), len: Number(r.len) };
      ensureSafeRange(normalized.offset, normalized.len, 'I/O range');
      return normalized;
    })
    .filter((r) => r.len > 0)
    .sort((a, b) => a.offset - b.offset);
  if (totalLen !== Infinity && (!Number.isSafeInteger(totalLen) || totalLen < 0)) {
    throw new AhiruError(Code.VALUE_OUT_OF_RANGE, {
      detail: `file length ${totalLen} is not a non-negative safe integer`,
    });
  }
  const out = [];
  for (const r of sorted) {
    const prev = out[out.length - 1];
    // Compare the gap by subtraction so adding the coalescing threshold cannot
    // overflow the safe-integer range near a 2^53-sized file.
    if (prev !== undefined && r.offset - (prev.offset + prev.len) <= gap) {
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

/**
 * Records one fetched byte range, merged into the coverage already recorded for
 * the same part. Mutates and returns `ranges`.
 *
 * Codec delegation asks "did we ever fetch [offset, offset+len)?", so what has to
 * be remembered is *coverage*, not a list of individual requests. Remembering
 * `(part, offset)` alone loses a later, longer fetch that happens to start where
 * an earlier short one did -- which is exactly what a page-index-filtered query
 * followed by a full scan produces. Merging also keeps the list from growing
 * once per fetch for the lifetime of the session.
 */
export function recordFetched(ranges, part, offset, len) {
  if (!(len > 0)) return ranges;
  let start = offset;
  let end = offset + len;
  // Absorb every recorded range this one touches or abuts.
  for (let i = ranges.length - 1; i >= 0; i--) {
    const r = ranges[i];
    if (r.part !== part || r.offset > end || r.offset + r.len < start) continue;
    if (r.offset < start) start = r.offset;
    if (r.offset + r.len > end) end = r.offset + r.len;
    ranges.splice(i, 1);
  }
  ranges.push({ part, offset: start, len: end - start });
  return ranges;
}

// --- Wire format decoding ----------------------------------------------------

/**
 * Converts a `u64` (as BigInt) crossing the wasm ABI into a `number`, throwing
 * instead of silently losing precision above `Number.MAX_SAFE_INTEGER`.
 *
 * `ByteSource.read(offset, length)` is a public contract expressed in `number`s
 * (see the type docs / README), and widening it to `bigint` end-to-end would
 * ripple into every host and every user-supplied source. A file whose offsets
 * exceed 2^53 is outside what a browser `fetch`/Range request can address
 * sanely anyway, so failing loudly here is the smaller, safer change.
 */
function toSafeNumber(big, what) {
  if (big > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new AhiruError(Code.VALUE_OUT_OF_RANGE, {
      detail: `${what} ${big} exceeds Number.MAX_SAFE_INTEGER`,
    });
  }
  return Number(big);
}

/**
 * JavaScript numbers can represent each u64 field safely while still losing
 * precision when `offset + len` crosses 2^53. Reject that combination before
 * it reaches a Range header or a custom ByteSource.
 */
function ensureSafeRange(offset, len, what) {
  if (
    !Number.isSafeInteger(offset) ||
    !Number.isSafeInteger(len) ||
    offset < 0 ||
    len < 0 ||
    offset > Number.MAX_SAFE_INTEGER - len
  ) {
    throw new AhiruError(Code.VALUE_OUT_OF_RANGE, {
      detail: `${what} [${offset}, ${offset} + ${len}) exceeds the safe numeric range`,
    });
  }
}

/**
 * `encode_io`: [count:u32][{table:u32, part:u32, offset:u64, len:u64}...]
 *
 * `part` says which file of a multi-file table (`ahiru_register_multi`) is meant.
 * Single-file registration (`ahiru_register`/`ahiru_register_as`) is always 0.
 * It must be passed straight back when calling `ahiru_provide` -- `table` alone
 * cannot uniquely identify the file the byte offsets are relative to.
 *
 * A request at offset `u64::MAX` with length 0 is a size request for a table
 * declared before its length was known; it decodes as `{ table, part, size: true }`
 * and is answered with `ahiru_set_size`.
 */
export function decodeIoRequests(u8) {
  if (u8.byteLength < 4) {
    throw new AhiruError(Code.INTERNAL, { detail: 'malformed I/O request buffer' });
  }
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const n = dv.getUint32(0, true);
  if (n > Math.floor((u8.byteLength - 4) / IO_REQUEST_SIZE)) {
    throw new AhiruError(Code.INTERNAL, { detail: 'truncated I/O request buffer' });
  }
  const out = [];
  for (let i = 0; i < n; i++) {
    const p = 4 + i * IO_REQUEST_SIZE;
    if (dv.getBigUint64(p + 8, true) === SIZE_UNKNOWN && dv.getBigUint64(p + 16, true) === 0n) {
      out.push({ table: dv.getUint32(p, true), part: dv.getUint32(p + 4, true), size: true });
      continue;
    }
    const offset = toSafeNumber(dv.getBigUint64(p + 8, true), 'I/O request offset');
    const len = toSafeNumber(dv.getBigUint64(p + 16, true), 'I/O request length');
    ensureSafeRange(offset, len, 'I/O request');
    out.push({
      table: dv.getUint32(p, true),
      part: dv.getUint32(p + 4, true),
      offset,
      len,
    });
  }
  return out;
}

/**
 * `encode_codec`: [count:u32][{table:u32, part:u32, codec:u32, offset:u64, len:u32, out_len:u32}...]
 * `part` means the same thing as in `decodeIoRequests`.
 */
export function decodeCodecRequests(u8) {
  if (u8.byteLength < 4) {
    throw new AhiruError(Code.INTERNAL, { detail: 'malformed codec request buffer' });
  }
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const n = dv.getUint32(0, true);
  if (n > Math.floor((u8.byteLength - 4) / CODEC_REQUEST_SIZE)) {
    throw new AhiruError(Code.INTERNAL, { detail: 'truncated codec request buffer' });
  }
  const out = [];
  for (let i = 0; i < n; i++) {
    const p = 4 + i * CODEC_REQUEST_SIZE;
    const offset = toSafeNumber(dv.getBigUint64(p + 12, true), 'codec request offset');
    const len = dv.getUint32(p + 20, true);
    ensureSafeRange(offset, len, 'codec request');
    const outLen = dv.getUint32(p + 24, true);
    if (outLen > MAX_DECOMPRESSED_PAGE_BYTES) {
      throw new AhiruError(Code.LIMIT_EXCEEDED, {
        detail: `codec output exceeds the per-page limit (${outLen} > ${MAX_DECOMPRESSED_PAGE_BYTES})`,
      });
    }
    out.push({
      table: dv.getUint32(p, true),
      part: dv.getUint32(p + 4, true),
      codec: dv.getUint32(p + 8, true),
      offset,
      len,
      outLen,
    });
  }
  return out;
}

/**
 * `encode_missing`: [count:u32][{format:u32, path_len:u32, path}...] -- the paths
 * a statement's string literals named that no table is registered under
 * (`START_NEED_TABLES`). `format` is the `FORMAT_CODES` value the SQL asked for.
 */
export function decodeMissingTables(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const n = wireU32(dv, u8, 0, 'missing-table header');
  const out = [];
  let p = 4;
  for (let i = 0; i < n; i++) {
    const format = wireU32(dv, u8, p, 'missing-table format');
    const len = wireU32(dv, u8, p + 4, 'missing-table path length');
    const end = wireEnd(u8, p + 8, len, 'missing-table path');
    out.push({ format, path: decodeUtf8(u8.subarray(p + 8, end), 'missing-table path') });
    p = end;
  }
  return out;
}

/** `encode_schema`: [n:u32][{ty:u32, phys:u32, precision:u32, scale:u32, name_len:u32, name}...] */
function wireError(detail) {
  throw new AhiruError(Code.INTERNAL, { detail });
}

/** Returns the end offset after checking a variable-length wire field. */
function wireEnd(u8, offset, length, what) {
  if (
    !Number.isSafeInteger(offset) ||
    !Number.isSafeInteger(length) ||
    offset < 0 ||
    length < 0 ||
    offset > u8.byteLength - length
  ) {
    wireError(`truncated ${what}`);
  }
  return offset + length;
}

function wireU32(dv, u8, offset, what) {
  wireEnd(u8, offset, 4, what);
  return dv.getUint32(offset, true);
}

/** Physical representation required by one logical type-code/schema tuple. */
function expectedPhysType(ty, precision) {
  switch (ty) {
    case 1:
      return PHYS_BOOL;
    case 0:
    case 2:
    case 3:
    case 4:
    case 7:
    case 8:
    case 16:
      return PHYS_I32;
    case 5:
    case 9:
    case 17:
    case 18:
    case 22:
      return PHYS_I64;
    case 6:
    case 10:
    case 19:
      return PHYS_I128;
    case 11:
    case 12:
      return PHYS_F64;
    case TY_DECIMAL:
      return precision <= 18 ? PHYS_I64 : PHYS_I128;
    case TY_VARCHAR:
    case 15:
    case TY_JSON:
    case TY_UUID:
      return PHYS_BYTES;
    default:
      return -1;
  }
}

export function decodeSchema(u8) {
  if (u8.byteLength < 4) wireError('malformed schema buffer');
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const n = wireU32(dv, u8, 0, 'schema header');
  // Every field has at least the five u32 words below. This guard also keeps a
  // corrupt count from causing a huge allocation before the first bounds check.
  if (n > Math.floor((u8.byteLength - 4) / 20)) wireError('truncated schema fields');
  const fields = [];
  let p = 4;
  for (let i = 0; i < n; i++) {
    wireEnd(u8, p, 20, 'schema field');
    const ty = dv.getUint32(p, true);
    const phys = dv.getUint32(p + 4, true);
    const precision = dv.getUint32(p + 8, true);
    const scale = dv.getUint32(p + 12, true);
    const nameLen = dv.getUint32(p + 16, true);
    p += 20;
    const nameEnd = wireEnd(u8, p, nameLen, 'schema field name');
    if (ty === TY_DECIMAL && (precision < 1 || precision > 38 || scale > precision)) {
      wireError('invalid DECIMAL schema precision/scale');
    }
    if (phys !== expectedPhysType(ty, precision)) {
      wireError('logical/physical schema type mismatch');
    }
    const name = decodeUtf8(u8.subarray(p, nameEnd), 'schema field name');
    p = nameEnd;
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
          '(for a TIMESTAMP, bind BigInt microseconds and write make_timestamp(?), ' +
          'or bind an ISO string and write ?::TIMESTAMP)',
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
  if (u8.byteLength < 12) wireError('malformed result buffer');
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const magic = wireU32(dv, u8, 0, 'result header');
  if (magic !== RESULT_MAGIC) {
    throw new AhiruError(Code.INTERNAL, {
      detail: `result magic mismatch: 0x${magic.toString(16)}`,
    });
  }
  const numCols = wireU32(dv, u8, 4, 'result header');
  const numRows = wireU32(dv, u8, 8, 'result header');
  if (!Array.isArray(schema) || schema.length !== numCols) {
    wireError('result column count does not match schema');
  }
  // A column always has phys:u32, validity_len:u32, and data_len:u32, even
  // when it contains zero rows. Reject an impossible count before looping.
  if (numCols > Math.floor((u8.byteLength - 12) / 12)) {
    wireError('truncated result columns');
  }
  let p = 12;
  const columns = [];

  for (let c = 0; c < numCols; c++) {
    wireEnd(u8, p, 8, 'result column header');
    const phys = dv.getUint32(p, true);
    const validityLen = dv.getUint32(p + 4, true);
    p += 8;
    let valid = null;
    if (validityLen > 0) {
      const minValidityLen = Math.ceil(numRows / 8);
      if (validityLen < minValidityLen) wireError('short result validity bitmap');
      const bits = u8.subarray(p, p + validityLen);
      p = wireEnd(u8, p, validityLen, 'result validity bitmap');
      // Bitmaps are always copied. Expanding to one byte per row is easier for
      // callers to handle, and it stays small (rows/8 -> rows).
      valid = new Uint8Array(numRows);
      for (let i = 0; i < numRows; i++) valid[i] = bitAt(bits, i);
    }

    const field = schema?.[c];
    const ty = field?.typeCode ?? -1;
    const precision = field?.precision ?? 0;
    const scale = field?.scale ?? 0;
    if (
      (ty === TY_DECIMAL &&
        (!Number.isInteger(precision) ||
          !Number.isInteger(scale) ||
          precision < 1 ||
          precision > 38 ||
          scale < 0 ||
          scale > precision)) ||
      phys !== field?.physType ||
      phys !== expectedPhysType(ty, precision)
    ) {
      wireError('result physical type does not match schema');
    }
    let values;

    if (phys === PHYS_BYTES) {
      const offsetsLen = wireU32(dv, u8, p, 'result byte offsets length');
      p += 4;
      const expectedOffsetsLen = (numRows + 1) * Uint32Array.BYTES_PER_ELEMENT;
      if (offsetsLen !== expectedOffsetsLen) wireError('invalid result byte offsets length');
      const offsetsEnd = wireEnd(u8, p, offsetsLen, 'result byte offsets');
      const offsets = viewOrCopy(Uint32Array, u8, p, numRows + 1, false);
      p = offsetsEnd;
      const dataLen = wireU32(dv, u8, p, 'result byte data length');
      p += 4;
      const dataEnd = wireEnd(u8, p, dataLen, 'result byte data');
      const data = u8.subarray(p, dataEnd);
      p = dataEnd;
      for (let i = 0; i <= numRows; i++) {
        if (offsets[i] > dataLen || (i > 0 && offsets[i] < offsets[i - 1])) {
          wireError('invalid result byte offsets');
        }
      }
      values = new Array(numRows);
      for (let i = 0; i < numRows; i++) {
        const s = offsets[i];
        const e = offsets[i + 1];
        if (valid !== null && valid[i] === 0) {
          // NULL values use an empty placeholder in the core. In particular,
          // do not pass a zero-length placeholder to UUID formatting, which
          // requires exactly 16 bytes and would otherwise throw TypeError.
          values[i] = ty === TY_VARCHAR || ty === TY_JSON || ty === TY_UUID ? '' : data.slice(s, e);
          continue;
        }
        // VARCHAR / JSON are UTF-8 strings (JSON's physical representation is the
        // raw text before decoding, so it can be handed over as a string as is --
        // whether to `JSON.parse` is left to the caller); UUID is a hyphenated hex
        // string; everything else (BLOB) is returned as raw bytes.
        if (ty === TY_VARCHAR || ty === TY_JSON) {
          values[i] = decodeUtf8(data.subarray(s, e), 'result text value', Code.INVALID_UTF8);
        } else if (ty === TY_UUID) {
          if (e - s !== 16) wireError('invalid UUID result width');
          values[i] = formatUuid(data.subarray(s, e));
        } else {
          values[i] = data.slice(s, e);
        }
      }
      columns.push({ name: field?.name ?? `col${c}`, type: field?.type ?? 'BLOB', typeCode: ty, physType: phys, values, valid });
      continue;
    }

    const dataLen = wireU32(dv, u8, p, 'result column data length');
    p += 4;
    const dataAt = p;
    const dataEnd = wireEnd(u8, p, dataLen, 'result column data');
    p = dataEnd;

    switch (phys) {
      case PHYS_BOOL: {
        // Bool is a bitmap too. Expand it into a 0/1 Uint8Array.
        const expected = Math.ceil(numRows / 64) * 8;
        if (dataLen !== expected) wireError('invalid BOOLEAN result width');
        const bits = u8.subarray(dataAt, dataAt + dataLen);
        values = new Uint8Array(numRows);
        for (let i = 0; i < numRows; i++) values[i] = bitAt(bits, i);
        break;
      }
      case PHYS_I32:
        if (dataLen !== numRows * Int32Array.BYTES_PER_ELEMENT) {
          wireError('invalid I32 result width');
        }
        values = viewOrCopy(Int32Array, u8, dataAt, numRows, copy);
        break;
      case PHYS_I64:
        if (dataLen !== numRows * BigInt64Array.BYTES_PER_ELEMENT) {
          wireError('invalid I64 result width');
        }
        values = viewOrCopy(BigInt64Array, u8, dataAt, numRows, copy);
        break;
      case PHYS_F64:
        if (dataLen !== numRows * Float64Array.BYTES_PER_ELEMENT) {
          wireError('invalid F64 result width');
        }
        values = viewOrCopy(Float64Array, u8, dataAt, numRows, copy);
        break;
      case PHYS_I128: {
        // There is no 128-bit TypedArray, so use an array of BigInt.
        if (dataLen !== numRows * 16) wireError('invalid I128 result width');
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
  /** Lazily built row-object keys; see `#rowKeys`. */
  #keys;

  constructor(numRows, columns) {
    this.numRows = numRows;
    this.columns = columns;
    this.schema = columns.map((c) => ({ name: c.name, type: c.type }));
  }

  /**
   * The property name each column gets in a row object, made unique.
   *
   * Two output columns can legitimately share a name -- `SELECT id, id, name AS id
   * FROM t` gives three columns all called `id`. Assigning each to `o[c.name]` let
   * the last one win, so `query()` and `toRows()` returned a single `id` holding the
   * final column's value and silently dropped the other two. Duplicates now get
   * `_1`, `_2`, ... appended, the way DuckDB's own clients disambiguate them.
   * `columns` is fixed for the life of a batch, so this is computed once.
   */
  #rowKeys() {
    if (this.#keys === undefined) {
      const seen = new Set();
      this.#keys = this.columns.map((c) => {
        let k = c.name;
        // A generated name can itself collide (a real column literally called
        // `id_1`), so keep counting until the name is free.
        for (let n = 1; seen.has(k); n++) k = `${c.name}_${n}`;
        seen.add(k);
        return k;
      });
    }
    return this.#keys;
  }

  #index(k) {
    if (typeof k === 'number') {
      if (!Number.isInteger(k) || k < 0 || k >= this.columns.length) {
        throw new AhiruError(Code.COLUMN_NOT_FOUND, { detail: String(k) });
      }
      return k;
    }
    const i = this.columns.findIndex((c) => c.name === k);
    if (i < 0) throw new AhiruError(Code.COLUMN_NOT_FOUND, { detail: String(k) });
    return i;
  }

  #row(i) {
    if (!Number.isInteger(i) || i < 0 || i >= this.numRows) {
      throw new AhiruError(Code.VALUE_OUT_OF_RANGE, { detail: `row ${String(i)}` });
    }
    return i;
  }

  /** Raw column values (TypedArray or Array). NULL positions hold a dummy value. */
  column(k) {
    return this.columns[this.#index(k)].values;
  }

  /** Whether a row is NULL. Honors the validity bitmap. */
  isNull(k, row) {
    const v = this.columns[this.#index(k)].valid;
    return v !== null && v[this.#row(row)] === 0;
  }

  /** The value at row i, column k. NULL is `null`. */
  get(k, row) {
    const c = this.columns[this.#index(k)];
    const i = this.#row(row);
    if (c.valid !== null && c.valid[i] === 0) return null;
    return c.physType === PHYS_BOOL ? c.values[i] === 1 : c.values[i];
  }

  /**
   * Turns the batch into an array of plain objects. Columns that share a name are
   * given distinct keys (`id`, `id_1`, `id_2`); see `#rowKeys`.
   */
  toRows() {
    const keys = this.#rowKeys();
    // A column literally named `__proto__` must become an own property. Plain
    // assignment would invoke the `__proto__` setter instead: the value vanished
    // from the row, and an object value (an INTERVAL) became the row's prototype.
    const proto = keys.indexOf('__proto__');
    const rows = new Array(this.numRows);
    for (let r = 0; r < this.numRows; r++) {
      const o = {};
      for (let j = 0; j < this.columns.length; j++) {
        const c = this.columns[j];
        const v =
          c.valid !== null && c.valid[r] === 0
            ? null
            : c.physType === PHYS_BOOL
              ? c.values[r] === 1
              : c.values[r];
        if (j === proto) {
          Object.defineProperty(o, '__proto__', {
            value: v,
            enumerable: true,
            writable: true,
            configurable: true,
          });
        } else {
          o[keys[j]] = v;
        }
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

/** GZIP compression for `COPY ... TO 'x.gz'` (the core has no deflate encoder). */
async function gzip(bytes) {
  if (typeof CompressionStream !== 'function') {
    throw new AhiruError(Code.UNSUPPORTED_CODEC, {
      detail: 'writing a .gz file needs CompressionStream (browser or Node 18+)',
    });
  }
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** GZIP. Costing zero extra bytes is the whole point of this delegation. */
async function gunzip(bytes, maxLen) {
  if (typeof DecompressionStream !== 'function') {
    throw new AhiruError(Code.UNSUPPORTED_CODEC, {
      detail: 'GZIP needs DecompressionStream (browser or Node 18+)',
    });
  }
  let reader;
  try {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    reader = stream.getReader();
    const chunks = [];
    let total = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > maxLen) {
        await reader.cancel().catch(() => undefined);
        throw new AhiruError(Code.LIMIT_EXCEEDED, {
          detail: `GZIP output exceeds the declared page size (${total} > ${maxLen})`,
        });
      }
      chunks.push(value);
    }
    const out = new Uint8Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      out.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return out;
  } catch (cause) {
    if (cause instanceof AhiruError) throw cause;
    throw new AhiruError(Code.BAD_COMPRESSED_DATA, {
      detail: 'invalid GZIP data',
      cause,
    });
  } finally {
    try {
      reader?.releaseLock();
    } catch {
      // The codec failure above is the useful normalized error; do not let a
      // broken custom stream's cleanup mask it.
    }
  }
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
    let srcPtr = 0;
    let dstPtr = 0;
    try {
      srcPtr = e.zstd_alloc(src.length);
      // zstd_alloc returns null on allocation failure (it uses try_reserve_exact,
      // same as the core's #provide checks ahiru_alloc). Writing to a null pointer
      // would corrupt the start of wasm memory instead of failing loudly.
      if (srcPtr === 0) {
        throw new AhiruError(Code.OOM, { detail: 'zstd module: allocation failed for the input buffer' });
      }
      // Memory may grow on every alloc. Re-take the view each time (same policy as the core).
      new Uint8Array(e.memory.buffer).set(src, srcPtr);
      dstPtr = e.zstd_alloc(outLen);
      if (dstPtr === 0) {
        throw new AhiruError(Code.OOM, { detail: 'zstd module: allocation failed for the output buffer' });
      }
      const n = e.zstd_decompress(srcPtr, src.length, dstPtr, outLen);
      if (n < 0) {
        throw new AhiruError(Code.BAD_COMPRESSED_DATA, { detail: `zstd_decompress -> ${n}` });
      }
      if (n > outLen) {
        throw new AhiruError(Code.BAD_COMPRESSED_DATA, {
          detail: `zstd_decompress returned ${n} bytes for a ${outLen}-byte buffer`,
        });
      }
      // Left inside wasm it would detach on the next alloc, so return a copy.
      return new Uint8Array(e.memory.buffer, dstPtr, n).slice();
    } finally {
      // Free both host-owned side-module allocations even if the decoder traps
      // or a later allocation/memory write fails. Keep the frees nested so a
      // broken cleanup export cannot prevent the other allocation from being freed.
      try {
        if (srcPtr !== 0) e.zstd_free(srcPtr, src.length);
      } finally {
        if (dstPtr !== 0) e.zstd_free(dstPtr, outLen);
      }
    }
  }
}

// --- Session lock --------------------------------------------------------------

/**
 * A minimal async mutex, used to serialize every entry point that touches the
 * wasm session.
 *
 * The engine keeps its result buffer and last-error state as module-level
 * singletons (`State` in `abi.rs`), not per-session, so two `ahiru_query_step`
 * calls interleaving on the same instance (e.g. `Promise.all([db.query(a),
 * db.query(b)])`) would silently overwrite each other's output. Queuing callers
 * here instead makes that impossible.
 */
class Mutex {
  #tail = Promise.resolve();

  /** Waits for the lock. Returns a function the caller must call exactly once to release it. */
  async acquire() {
    let release;
    const held = new Promise((resolve) => {
      release = resolve;
    });
    const prev = this.#tail;
    this.#tail = prev.then(() => held);
    await prev;
    return release;
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
  if (typeof doFetch !== 'function') {
    throw new TypeError('no fetch available. Pass an implementation via init({ fetch })');
  }
  // Keep wasm loading on the same normalized error path as table I/O. In
  // particular, a signed wasm URL must not leak its query token in an
  // exception message when a server rejects it or the network fails.
  const r = await request(doFetch, s);
  if (!r.ok) {
    throw new AhiruError(Code.IO_FAILED, { detail: `${redactUrl(s)} -> HTTP ${r.status}` });
  }
  return responseBytes(r, s);
}

// --- Main --------------------------------------------------------------------

/**
 * Whether an exception out of a wasm call means the instance can no longer be
 * trusted. A trap (a stack overflow in deep recursion, `unreachable` from a
 * panic, an out-of-bounds access) unwinds past every Rust epilogue, so the
 * shadow `__stack_pointer` and whatever state was mid-update stay as they were:
 * every later call traps too, or worse, runs on corrupt state. A recursion deep
 * enough to exhaust the host's native stack first surfaces as a RangeError and
 * leaves the instance in the same condition.
 */
function isWasmTrap(e) {
  return (
    e instanceof WebAssembly.RuntimeError ||
    (e instanceof RangeError && /call stack/i.test(String(e.message)))
  );
}

/** Bound on the coalesced ranges remembered per table for codec delegation (`#bytesAt`). */
const MAX_REMEMBERED_JOBS = 4096;

export class AhiruDB {
  /** The compiled module, kept so a trapped instance can be replaced. */
  #module;
  #exports;
  #memory;
  #session;
  /** wasm table index -> registration record, for every table the current instance knows. */
  #byIndex = new Map();
  /**
   * Registrations not declared to the wasm catalog yet: made while a run held
   * the instance (replacing a table under a running scan would corrupt it), or
   * carried over from a trapped instance. The next run declares them first.
   */
  #pending = [];
  #cache;
  /** When the cache was supplied from outside, close() must not clear it. */
  #ownsCache;
  #fetch;
  /** Optional gate for paths discovered in SQL string literals. */
  #sqlUrlPolicy;
  /** Where `COPY ... TO` results go (the engine never writes files itself). */
  #onCopy;
  #memoryLimit;
  /** Cap on the fetched bytes retained for codec delegation. */
  #residentLimit;
  #closed = false;
  /** The ZSTD side module. Not loaded until the first NEED_CODEC. */
  #zstd = null;
  #zstdOptions;
  /** Serializes query()/stream() against each other (see the Mutex doc comment). */
  #sessionLock = new Mutex();
  /** Number of runs that have acquired the session lock and may still call wasm. */
  #activeRuns = 0;
  /**
   * The run currently holding the session lock, so `close()` can take the
   * session back from an iterator that was abandoned mid-stream.
   */
  #currentRun = null;
  /** Prevents close() from freeing the session twice, including deferred close. */
  #sessionFreed = false;
  /** A wasm call trapped: nothing may call into the current instance again. */
  #trapped = false;
  /**
   * Why a trapped instance cannot be replaced (every later call fails with it),
   * or null. Replacing it is only transparent while the session holds nothing
   * that would be lost with it -- in-memory tables and views created by SQL.
   */
  #dead = null;
  /** In-memory tables + views (`ddl`) the session held after the last run. */
  #ddlObjects = 0;

  constructor(instance, options = {}, module = undefined) {
    this.#zstdOptions = {
      zstdUrl: options.zstdUrl,
      zstdBinary: options.zstdBinary,
      zstdModule: options.zstdModule,
      fetch: options.fetch,
    };
    this.#module = module instanceof WebAssembly.Module ? module : undefined;
    this.#ownsCache = typeof options.cache !== 'object' || options.cache === null;
    const cacheSize = requireNonNegativeSafeInteger(
      options.cacheSize ?? DEFAULT_CACHE_SIZE,
      'cacheSize',
    );
    this.#cache = makeCache(options.cache, cacheSize);
    this.#residentLimit = cacheSize;
    this.#fetch = options.fetch;
    if (
      options.sqlUrlPolicy !== undefined &&
      options.sqlUrlPolicy !== false &&
      typeof options.sqlUrlPolicy !== 'function'
    ) {
      throw new TypeError('sqlUrlPolicy must be a function or false');
    }
    this.#sqlUrlPolicy = options.sqlUrlPolicy;
    if (options.onCopy !== undefined && typeof options.onCopy !== 'function') {
      throw new TypeError('onCopy must be a function');
    }
    this.#onCopy = options.onCopy;
    this.#memoryLimit = requireNonNegativeSafeInteger(options.memoryLimit ?? 0, 'memoryLimit');
    this.#bindInstance(instance);
  }

  /**
   * Loads the wasm and opens a single session.
   *
   * `wasmUrl` may be a URL or a file path (on Node it is read as a file).
   * If you already have bytes or a compiled module, use `wasmBinary` / `wasmModule`.
   */
  static async init(options = {}) {
    const { wasmUrl, wasmBinary, wasmModule } = options;
    let module = wasmModule;
    if (!(module instanceof WebAssembly.Module)) {
      const bytes = wasmBinary
        ? ArrayBuffer.isView(wasmBinary)
          ? new Uint8Array(wasmBinary.buffer, wasmBinary.byteOffset, wasmBinary.byteLength)
          : new Uint8Array(wasmBinary)
        : await loadWasmBytes(wasmUrl ?? 'ahiru-core.wasm', options.fetch);
      // Compiled separately from instantiation so a trapped instance can be
      // replaced later without fetching or compiling the module again.
      module = await WebAssembly.compile(bytes);
    }
    // The core has no imports at all (no_std, panic=abort).
    const instance = await WebAssembly.instantiate(module, {});
    return new AhiruDB(instance, options, module);
  }

  /** Adopts a fresh instance and opens its session. */
  #bindInstance(instance) {
    const e = instance.exports;
    this.#exports = e;
    this.#memory = e.memory;
    const session = e.ahiru_session_new();
    if (session < 0) throw new AhiruError(Code.INTERNAL, { detail: 'session_new failed' });
    this.#session = session;
    this.#sessionFreed = false;
    this.#trapped = false;
    this.#ddlObjects = 0;
  }

  /**
   * Registers a table. No I/O whatsoever happens here.
   * Fetching the total byte length and reading the footer / header are both deferred
   * to the first query that actually reads the table.
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
   *
   * The name is an SQL identifier: it replaces an earlier registration whose name
   * differs only in ASCII case, as the engine's catalog does.
   */
  register(name, source, { format } = {}) {
    return this.#register(name, source, format, { path: false, rejectRedirects: false });
  }

  /**
   * `path`: a path a SQL string literal referenced, registered case-sensitively.
   * `now`: declare immediately (the caller is the run holding the instance).
   */
  #register(name, source, format, { path, rejectRedirects, now = false }) {
    this.#assertUsable();
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
          detail: `unknown format "${format}" (parquet / csv / tsv / jsonl / json)`,
        });
      }
    }
    const rec = {
      name,
      path,
      source: makeSource(source, this.#fetch, { rejectRedirects }),
      index: -1,
      size: -1,
      // What it will actually be read as. For Auto, infer it with the engine's rule and show that.
      format: code === FORMAT_CODES.auto ? detectFormat(name) : String(format).toLowerCase(),
      formatCode: code,
      // The retained copy of supplied bytes, and the ranges fetched so far (used once the copy is dropped).
      resident: [],
      fetched: [],
      // The exact coalesced ranges fetched (the range-cache keys), for `#bytesAt`.
      jobs: new JobIndex(),
    };
    // Declaring while idle surfaces registration errors (a name CREATE TABLE
    // already took, a format this build lacks) right here.
    if (now || (this.#activeRuns === 0 && !this.#trapped)) {
      try {
        this.#declare(rec);
      } catch (e) {
        throw this.#asTrap(e);
      }
    } else {
      this.#pending.push(rec);
    }
    return this;
  }

  /** Declares a registration to the wasm catalog (no I/O; the length may still be unknown). */
  #declare(rec) {
    const e = this.#exports;
    if (typeof e.ahiru_set_size !== 'function') {
      throw new AhiruError(Code.UNSUPPORTED_FEATURE, {
        detail: 'this wasm core is older than the JS host (no ahiru_set_size); rebuild it',
      });
    }
    const name = textEncoder.encode(rec.name);
    const ptr = e.ahiru_alloc(name.length) >>> 0;
    if (ptr === 0 && name.length > 0) throw new AhiruError(Code.OOM);
    let idx;
    try {
      new Uint8Array(this.#memory.buffer).set(name, ptr);
      idx = e.ahiru_register_as(
        this.#session,
        ptr,
        name.length,
        // A length learnt by an earlier instance (see #replaceInstance) is reused.
        rec.size >= 0 ? BigInt(rec.size) : SIZE_UNKNOWN,
        rec.formatCode | (rec.path ? FORMAT_PATH : 0),
      );
    } finally {
      e.ahiru_free(ptr, name.length);
    }
    if (idx < 0) throw this.#lastError();
    rec.index = idx;
    // A re-registration reuses its predecessor's index, which drops the stale record.
    this.#byIndex.set(idx, rec);
  }

  /** Declares the registrations queued while the instance was busy. */
  #flushPending() {
    while (this.#pending.length > 0) {
      // A registration the engine rejects is dropped and reported by this run.
      this.#declare(this.#pending.shift());
    }
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

  /**
   * Closes the session. Later calls are errors.
   *
   * A `stream()` iterator that is abandoned part-way through -- `break` without
   * `return()`, or simply dropped -- is suspended at a `yield` and never resumed,
   * so its `finally` never runs and it would hold the session lock for the
   * lifetime of the process. `close()` is therefore the recovery point: it takes
   * the run's query slot and lock back. If that iterator is ever resumed it
   * throws instead of stepping a handle this call already closed.
   */
  close() {
    if (this.#closed) return;
    this.#closed = true;
    this.#byIndex.clear();
    this.#pending = [];
    if (this.#ownsCache) this.#cache.clear();
    this.#abortCurrentRun();
    // A query may still be suspended at a source/fetch await. Do not free the
    // wasm session underneath it: the ABI state is shared by the whole instance,
    // so a later session allocation could otherwise reuse this handle while the
    // old run is still calling query_step/query_close.
    if (this.#activeRuns === 0) this.#freeSession();
  }

  /** Ends the in-flight run, if any, on close()'s behalf. */
  #abortCurrentRun() {
    const run = this.#currentRun;
    if (run === null || run.finished) return;
    run.aborted = true;
    // Close the query here rather than leaving it to a `finally` that may never
    // run; after this the handle is stale and the run must not touch it again.
    if (run.q >= 0) {
      const q = run.q;
      run.q = -1;
      if (!this.#trapped) this.#exports.ahiru_query_close(q);
    }
    run.finish();
  }

  /** How many bytes the wasm heap currently holds (0 while a trapped instance awaits replacement). */
  get heapUsed() {
    if (this.#trapped) return 0;
    // Unsigned: see the note in #out().
    return this.#exports.ahiru_heap_used() >>> 0;
  }

  // --- Execution loop -------------------------------------------------------

  async *#run(sql, params, copy) {
    this.#assertUsable();
    // Every wasm entry point below shares module-level state (the out buffer,
    // last-error) across the whole instance, so only one #run may be in flight
    // at a time. The lock is held for the entire lifetime of this generator,
    // including while the caller sits between `yield`s in stream() -- an early
    // `break`/`return()`/`throw()` on the consumer's side is delivered here as a
    // return/throw injected at the suspended yield, which still runs this `finally`
    // (standard (async) generator semantics), so the lock is always released.
    const release = await this.#sessionLock.acquire();
    // Everything close() needs to unwind this run from the outside. `finish` is
    // idempotent, so whichever of the two gets there first wins.
    const run = { q: -1, aborted: false, finished: false, finish: () => {} };
    run.finish = () => {
      if (run.finished) return;
      run.finished = true;
      if (this.#currentRun === run) this.#currentRun = null;
      this.#activeRuns--;
      release();
      if (this.#closed && this.#activeRuns === 0) this.#freeSession();
    };
    this.#currentRun = run;
    this.#activeRuns++;
    try {
      this.#assertUsable();
      if (this.#trapped) await this.#replaceInstance();
      this.#assertOpen();
      this.#flushPending();

      const q = await this.#start(sql, params);
      run.q = q;
      this.#assertOpen();
      try {
        // `COPY ... TO` has no rows; its bytes go to the onCopy handler.
        if (await this.#deliverCopy(q, sql)) return;
        const schema = this.#readSchema(q, sql);
        let lastSignature = null;
        for (;;) {
          // Re-checked after every suspension point, `yield` included: close()
          // may have taken the session away while this generator was parked.
          this.#assertRunning(run);
          const status = this.#exports.ahiru_query_step(q);
          this.#checkMemory(sql);
          if (status === STATUS_BATCH_READY) {
            const out = this.#out();
            yield decodeBatch(out, schema, copy);
            continue;
          }
          if (status === STATUS_NEED_IO) {
            lastSignature = await this.#pump(decodeIoRequests(this.#out()), lastSignature, sql);
            this.#assertRunning(run);
            continue;
          }
          if (status === STATUS_NEED_CODEC) {
            await this.#decompress(decodeCodecRequests(this.#out()), sql);
            this.#assertRunning(run);
            continue;
          }
          if (status === STATUS_DONE) return;
          throw this.#lastError(sql);
        }
      } catch (e) {
        // Noted before the `finally` below, which must not call into a trapped instance.
        throw this.#asTrap(e, sql);
      } finally {
        // close() already closed it (and freed the session) if it aborted us.
        if (!run.aborted && run.q >= 0 && !this.#trapped) {
          run.q = -1;
          // Closing also releases the bytes this query fetched into the wasm heap
          // (the next query asks for them again, from the range cache).
          this.#exports.ahiru_query_close(q);
          const count = this.#exports.ahiru_ddl_object_count;
          this.#ddlObjects = typeof count === 'function' ? count(this.#session) : 0;
        }
      }
    } catch (e) {
      throw this.#asTrap(e, sql);
    } finally {
      run.finish();
    }
  }

  /**
   * Normalizes an exception thrown out of a run. A wasm trap marks the instance
   * as unusable (it is replaced before the next call, or the database is dead
   * if replacing it would lose SQL-created state) and becomes an AhiruError.
   */
  #asTrap(e, sql) {
    if (!isWasmTrap(e)) return e;
    if (!this.#trapped) {
      this.#trapped = true;
      if (this.#module === undefined) {
        this.#dead = 'the wasm engine trapped and no compiled module is available to replace it';
      } else if (this.#ddlObjects > 0) {
        this.#dead =
          'the wasm engine trapped; replacing it would silently drop the in-memory tables ' +
          'and views created by SQL, so this database cannot be used any more';
      }
    }
    return new AhiruError(Code.INTERNAL, {
      sql,
      detail:
        `the wasm engine trapped (${e.message}); ` +
        (this.#dead === null
          ? 'it is replaced with a fresh instance before the next call'
          : 'create a new AhiruDB'),
      cause: e,
    });
  }

  /**
   * Swaps a trapped instance for a fresh one from the same module. Every
   * registration is declared again; lengths already learnt are reused, and bytes
   * come back from the range cache, so nothing is fetched twice.
   */
  async #replaceInstance() {
    const instance = await WebAssembly.instantiate(this.#module, {});
    this.#assertOpen();
    this.#bindInstance(instance);
    const known = [...this.#byIndex.entries()].sort((a, b) => a[0] - b[0]).map(([, rec]) => rec);
    this.#byIndex.clear();
    this.#pending = [...known, ...this.#pending];
  }

  /**
   * Starts a query. The engine may first ask for bytes (`-2`: footers, table
   * lengths, or the reads a one-shot statement such as `COPY` was waiting on) or
   * for paths its string literals named (`START_NEED_TABLES`); satisfy the
   * request and retry.
   */
  async #start(sql, params) {
    const bytes = textEncoder.encode(sql);
    const pbytes = encodeParams(params);
    let lastSignature = null;
    const asked = new Set();
    for (;;) {
      const e = this.#exports;
      // The core has no clock, so the query start time is passed in here for
      // CURRENT_DATE/CURRENT_TIMESTAMP/now() (DESIGN.md §2).
      e.ahiru_set_now(this.#session, BigInt(Date.now()) * 1000n);
      let ptr = 0;
      let pptr = 0;
      let h;
      try {
        ptr = e.ahiru_alloc(bytes.length) >>> 0;
        if (ptr === 0 && bytes.length > 0) throw new AhiruError(Code.OOM, { sql });
        pptr = pbytes.length > 0 ? e.ahiru_alloc(pbytes.length) >>> 0 : 0;
        if (pptr === 0 && pbytes.length > 0) throw new AhiruError(Code.OOM, { sql });
        // alloc may grow memory, so re-take the view right before writing.
        const mem = new Uint8Array(this.#memory.buffer);
        mem.set(bytes, ptr);
        if (pptr !== 0) mem.set(pbytes, pptr);
        h = e.ahiru_query_start(this.#session, ptr, bytes.length, pptr, pbytes.length);
      } finally {
        try {
          if (ptr !== 0) e.ahiru_free(ptr, bytes.length);
        } finally {
          if (pptr !== 0) e.ahiru_free(pptr, pbytes.length);
        }
      }
      if (h >= 0) return h;
      if (h === START_NEED_TABLES) {
        await this.#registerSqlPaths(decodeMissingTables(this.#out()), sql, asked);
        this.#assertOpen();
        continue;
      }
      if (h !== -2) throw this.#lastError(sql);
      lastSignature = await this.#pump(decodeIoRequests(this.#out()), lastSignature, sql);
      this.#assertOpen();
    }
  }

  /**
   * Registers the paths a statement's string literals named (`FROM 'x.parquet'`,
   * `parquet('https://…')`, `read_csv('…')`) under their exact spelling.
   *
   * The engine reports exactly the references it resolves, so a table name that
   * merely appears as a column, alias, comment, or string value is never bound or
   * fetched. `sqlUrlPolicy`, when configured, sees every such path -- resolved the
   * way `fetch()` would resolve it -- before anything is registered or fetched.
   */
  async #registerSqlPaths(missing, sql, asked) {
    for (const { format, path } of missing) {
      // Registering it did not satisfy the engine; stop instead of looping.
      if (asked.has(path)) {
        throw new AhiruError(Code.TABLE_NOT_FOUND, { sql, detail: redactUrl(path) });
      }
      asked.add(path);
      if (this.#sqlUrlPolicy !== undefined) {
        const url = resolveSqlPath(path);
        const functionName = FUNCTION_FAMILY[format] ?? 'parquet';
        const allowed =
          this.#sqlUrlPolicy === false
            ? false
            : await this.#sqlUrlPolicy(url, { functionName, sql });
        this.#assertOpen();
        if (!allowed) {
          throw new AhiruError(Code.UNSUPPORTED_FEATURE, {
            detail: `SQL URL policy rejected ${redactUrl(url)}`,
          });
        }
      }
      // A policy-gated SQL URL must not escape its allowlist through an HTTP
      // redirect. Keep the historical permissive behavior when no policy was
      // configured at all.
      this.#register(path, path, formatForSqlPath(format, path), {
        path: true,
        rejectRedirects: this.#sqlUrlPolicy !== undefined,
        now: true,
      });
    }
  }

  /**
   * Hands a `COPY ... TO` result to the `onCopy` handler. Returns false when the
   * statement was not a `COPY` (or the core was built without `export`).
   */
  async #deliverCopy(q, sql) {
    const take = this.#exports.ahiru_copy_result;
    if (typeof take !== 'function') return false;
    const n = take(q);
    if (n < 0) throw this.#lastError(sql);
    if (n === 0) return false;
    const out = this.#out();
    const dv = new DataView(out.buffer, out.byteOffset, out.byteLength);
    const pathLen = wireU32(dv, out, 0, 'COPY result');
    const flags = wireU32(dv, out, 4, 'COPY flags');
    const pathEnd = wireEnd(out, 8, pathLen, 'COPY path');
    const path = decodeUtf8(out.subarray(8, pathEnd), 'COPY path');
    // Copied out before anything else can call into wasm and reuse the buffer.
    let bytes = out.slice(pathEnd);
    // `out.csv.gz`: the core wrote plain CSV and leaves the compression to the host,
    // as DuckDB compresses by the file name.
    if (flags & COPY_GZIP) bytes = await gzip(bytes);
    if (this.#onCopy === undefined) {
      throw new AhiruError(Code.UNSUPPORTED_FEATURE, {
        sql,
        detail:
          `COPY ... TO '${redactUrl(path)}' produced ${bytes.byteLength} bytes, but the engine ` +
          'never writes files itself: pass onCopy(path, bytes) to AhiruDB.init() to receive them',
      });
    }
    await this.#onCopy(path, bytes);
    return true;
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
      // If there is even one ZSTD, load it once before decoding.
      if (req.codec === CODEC_ZSTD) this.#zstd ??= await ZstdModule.load(this.#zstdOptions);
    }

    // Decode and provide one page at a time. A split may contain many pages;
    // Promise.all would retain every decompressed block simultaneously and
    // multiply the per-page cap into an unbounded host-side allocation.
    for (const req of requests) {
      const rec = this.#recordAt(req.table, sql);
      const src = await this.#bytesAt(rec, req.part, req.offset, req.len, sql);
      const out =
        req.codec === CODEC_GZIP
          ? await gunzip(src, req.outLen)
          : this.#zstd.decompress(src, req.outLen);
      if (out.length !== req.outLen) {
        throw new AhiruError(Code.BAD_COMPRESSED_DATA, {
          sql,
          detail: `expected ${req.outLen} bytes, got ${out.length}`,
        });
      }
      this.#assertOpen();
      this.#provideCodec(req, out, sql);
    }
    this.#checkMemory(sql);
  }

  /** Returns decompressed blocks to wasm. */
  #provideCodec(req, bytes, sql) {
    const e = this.#exports;
    const ptr = e.ahiru_alloc(bytes.length) >>> 0;
    if (ptr === 0 && bytes.length > 0) throw new AhiruError(Code.OOM, { sql });
    let rc;
    try {
      new Uint8Array(this.#memory.buffer).set(bytes, ptr);
      rc = e.ahiru_provide_codec(
        this.#session,
        req.table,
        req.part,
        BigInt(req.offset),
        req.len,
        ptr,
        bytes.length,
      );
    } finally {
      e.ahiru_free(ptr, bytes.length);
    }
    if (rc !== 0) throw this.#lastError(sql);
  }

  /**
   * Slices the bytes of a compressed block out of what is on hand.
   *
   * The normal case is that they are in the copy retained by the preceding NEED_IO.
   * Past that, the coalesced range that contained them may still be in the range
   * cache under its own key, and is sliced from there. Only what is in neither
   * gets refetched. A request for a range never fetched at all is an engine-side
   * inconsistency, so it is reported rather than silently fetched.
   */
  async #bytesAt(rec, part, offset, len, sql) {
    for (const c of rec.resident) {
      if (c.part === part && c.offset <= offset && offset + len <= c.offset + c.bytes.length) {
        return c.bytes.subarray(offset - c.offset, offset - c.offset + len);
      }
    }
    // Memory / Blob sources need no I/O to slice, so they keep no retained copy.
    if (rec.source.cacheable === false) return rec.source.read(offset, len);
    // A page is only ever a piece of a coalesced fetch, and the cache is keyed by
    // those; asking it for the page's own range always missed, so every page
    // went back to the network even with the whole file cached.
    const cached = this.#fromCachedJobs(rec, part, offset, len);
    if (cached !== undefined) return cached;
    const everFetched = rec.fetched.some(
      (r) => r.part === part && r.offset <= offset && offset + len <= r.offset + r.len,
    );
    if (everFetched) return this.#read(rec, part, offset, len, sql);
    throw new AhiruError(Code.INTERNAL, {
      sql,
      detail:
        `codec request for bytes we never fetched: table ${rec.name} part ${part} [${offset}, ${offset + len})` +
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

  /** The registration behind a wasm table index. */
  #recordAt(table, sql) {
    const rec = this.#byIndex.get(table);
    if (rec === undefined) {
      throw new AhiruError(Code.INTERNAL, { sql, detail: `unknown table index ${table}` });
    }
    return rec;
  }

  /**
   * Satisfies I/O requests. Coalesce -> fetch in parallel -> `ahiru_provide`.
   * Size requests (a table declared before its length was known) are answered
   * alongside, with one `size()` (a HEAD for a URL) per table.
   * Returns a signature for comparison on the next round.
   */
  async #pump(requests, lastSignature, sql) {
    const signature = requests
      .map((r) => `${r.table}.${r.part}/${r.size ? 'size' : `${r.offset}+${r.len}`}`)
      .join(',');
    const sizes = requests.filter((r) => r.size);

    // `table` alone cannot tell which file of a multi-file table is meant (offsets
    // live in a separate space per file), so requests are grouped by the composite
    // key `table:part`. Single-file registration always has part=0, so this extends
    // safely to multiple files while behaving exactly as before.
    const jobs = [];
    for (const [key, list] of groupBy(
      requests.filter((r) => !r.size),
      (r) => `${r.table}:${r.part}`,
    )) {
      const [table, part] = key.split(':').map(Number);
      const rec = this.#recordAt(table, sql);
      for (const r of coalesceRanges(list, COALESCE_GAP, rec.size)) {
        jobs.push({ rec, table, part, offset: r.offset, len: r.len });
      }
    }

    // Sizes and coalesced ranges are fetched in parallel, to avoid stacking round trips.
    const [, buffers] = await Promise.all([
      Promise.all(
        sizes.map(async (r) => {
          const rec = this.#recordAt(r.table, sql);
          if (rec.size < 0) rec.size = await rec.source.size();
        }),
      ),
      Promise.all(jobs.map((j) => this.#read(j.rec, j.part, j.offset, j.len, sql))),
    ]);
    // `size()` and reads are user callbacks and may take arbitrarily long;
    // close() can land while they are in flight.
    this.#assertOpen();

    // How many of the requested ranges were handed over *in full*. Counting bytes alone
    // cannot tell progress from a livelock: a partially satisfied range provides a
    // non-zero count while `Source::insert` keeps only that prefix, so the engine asks
    // for the identical range on the next round, forever.
    let satisfied = 0;
    for (const r of sizes) {
      const size = this.#recordAt(r.table, sql).size;
      if (this.#exports.ahiru_set_size(this.#session, r.table, r.part, BigInt(size)) !== 0) {
        throw this.#lastError(sql);
      }
      satisfied++;
    }
    for (let i = 0; i < jobs.length; i++) {
      const { rec, table, part, offset, len } = jobs[i];
      if (this.#provide(table, part, offset, buffers[i], sql) === len) satisfied++;
      // Only remote sources keep a retained copy. When codec delegation asks for a
      // compressed block, slicing it out of here avoids a refetch (memory sources just slice).
      if (rec.source.cacheable !== false) {
        const len = buffers[i].byteLength;
        // Both records are keyed by coverage, not by the (part, offset) pair
        // alone. A page-index-filtered query fetches a short range, and the next
        // query's longer fetch starts at the same offset; deduplicating on the
        // start alone drops it, and `#bytesAt` then reports bytes we did fetch as
        // never fetched (E900).
        recordFetched(rec.fetched, part, offset, len);
        if (len > 0) rec.jobs.add(part, offset, len);
        const covered = rec.resident.some(
          (c) => c.part === part && c.offset <= offset && offset + len <= c.offset + c.bytes.length,
        );
        if (!covered && len > 0) {
          rec.resident.push({ part, offset, bytes: buffers[i] });
          // Drop oldest first, so this does not grow without bound. If a dropped range
          // is needed later it is refetched from the cache (`#bytesAt`).
          let held = rec.resident.reduce((a, c) => a + c.bytes.byteLength, 0);
          while (held > this.#residentLimit && rec.resident.length > 1) {
            held -= rec.resident.shift().bytes.byteLength;
          }
        }
      }
    }
    this.#checkMemory(sql);

    // Livelock detection, keyed on the request signature: the engine asked for exactly
    // the same set of ranges as last round and not one of them was satisfied in full,
    // so another round would ask the same thing again.
    if (satisfied === 0 && signature === lastSignature) {
      throw new AhiruError(Code.IO_FAILED, {
        sql,
        detail: `no progress for ranges [${signature}]`,
      });
    }
    return signature;
  }

  /**
   * Assembles `[offset, offset + len)` from the cached coalesced ranges this
   * registration fetched earlier (`rec.jobs`), or returns undefined when they do
   * not cover it all.
   */
  #fromCachedJobs(rec, part, offset, len) {
    const end = offset + len;
    const pieces = [];
    for (const j of rec.jobs.overlapping(part, offset, end)) {
      const hit = this.#cache.get(this.#cacheKey(rec, part, j.offset, j.len));
      if (hit instanceof Uint8Array && hit.byteLength === j.len) pieces.push({ at: j.offset, hit });
    }
    pieces.sort((a, b) => a.at - b.at);
    // The common case: a single cached range contains the whole request.
    for (const p of pieces) {
      if (p.at <= offset && end <= p.at + p.hit.byteLength) {
        return p.hit.subarray(offset - p.at, end - p.at);
      }
    }
    const out = new Uint8Array(len);
    let pos = offset;
    for (const p of pieces) {
      const pEnd = p.at + p.hit.byteLength;
      if (p.at > pos) break; // a gap no cached range covers
      if (pEnd <= pos) continue;
      const stop = Math.min(pEnd, end);
      out.set(p.hit.subarray(pos - p.at, stop - p.at), pos - offset);
      pos = stop;
      if (pos === end) return out;
    }
    return undefined;
  }

  /** The range-cache key of `[offset, offset + len)` of one registration's part. */
  #cacheKey(rec, part, offset, len) {
    return `${rec.source.key}:${part}:${offset}:${len}`;
  }

  /** Reads a byte range through the cache. */
  async #read(rec, part, offset, len, sql) {
    const key = this.#cacheKey(rec, part, offset, len);
    const cacheable = rec.source.cacheable !== false;
    if (cacheable) {
      const hit = this.#cache.get(key);
      // A hit is only usable if it is a Uint8Array of exactly the requested length.
      // A cache wrapper (the Cache API, a shared store, a persisted entry from an
      // older build) can hand back a truncated or otherwise wrong-length body; that
      // body would go straight to `ahiru_provide`, where `Source::insert` keeps it as
      // the prefix of the range, the engine would ask for the very same range again,
      // and `query()` would spin forever on a pure microtask loop that never yields
      // to the event loop. Anything else is treated as a miss, so the source is asked
      // directly; a genuinely short `ByteSource.read()` still throws IO_FAILED below.
      if (hit instanceof Uint8Array && hit.byteLength === len) return hit;
      // The fetch that brought these bytes in may have been coalesced differently
      // (a footer probe that happened to cover the last row group, a range next to
      // columns the previous query did not read). Once the engine drops its copy
      // after a query, the next one asks for ranges that exist in the cache only
      // as pieces of those fetches.
      const pieced = this.#fromCachedJobs(rec, part, offset, len);
      if (pieced !== undefined) return pieced;
    }
    const bytes = await rec.source.read(offset, len);
    let u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    // A user-supplied ByteSource may hand back a view aliasing memory it still
    // owns; copy it before this host retains it anywhere. Bytes this host produced
    // itself (fetch / Blob) are already privately owned, so they are not copied again.
    if (rec.source.untrusted) u8 = u8.slice();
    // A short or over-long read must never be handed to wasm: Source::insert keeps
    // the first prefix of overlapping ranges, so a truncated body at `offset` would
    // poison a later full read of the same window, and a non-empty short read also
    // defeats livelock detection (which only trips on zero bytes provided).
    // A zero-length body is left to that livelock path.
    if (u8.byteLength !== 0 && u8.byteLength !== len) {
      throw new AhiruError(Code.IO_FAILED, {
        sql,
        detail: `short read at ${offset}: got ${u8.byteLength} bytes, wanted ${len}`,
      });
    }
    // Never cache a zero-length body. Caching it makes the range permanently
    // unreadable: every retry hits the cache, the source is never asked again,
    // and the livelock detector then fails the query (and every later one, and
    // every other instance sharing the cache) with E504.
    if (cacheable && u8.byteLength !== 0) this.#cache.set(key, u8);
    return u8;
  }

  /** Hands fetched bytes to wasm. Returns the length handed over. */
  #provide(table, part, offset, bytes, sql) {
    if (bytes.byteLength === 0) return 0;
    const e = this.#exports;
    const ptr = e.ahiru_alloc(bytes.byteLength) >>> 0;
    if (ptr === 0) throw new AhiruError(Code.OOM, { sql });
    let rc;
    try {
      // alloc may have grown memory, so always re-take the view here.
      new Uint8Array(this.#memory.buffer).set(bytes, ptr);
      rc = e.ahiru_provide(this.#session, table, part, BigInt(offset), ptr, bytes.byteLength);
    } finally {
      e.ahiru_free(ptr, bytes.byteLength);
    }
    if (rc !== 0) throw this.#lastError(sql);
    return bytes.byteLength;
  }

  // --- Odds and ends --------------------------------------------------------

  /** A view of the current out buffer. Valid only until the next wasm call. */
  #out() {
    // wasm hands back `usize` as a signed i32, so anything at or above 2 GiB
    // arrives negative. `>>> 0` puts it back in the unsigned range; without it a
    // heap that has grown past 2 GiB fails with a bare RangeError from the
    // TypedArray constructor.
    const ptr = this.#exports.ahiru_out_ptr() >>> 0;
    const len = this.#exports.ahiru_out_len() >>> 0;
    return new Uint8Array(this.#memory.buffer, ptr, len);
  }

  #lastError(sql) {
    const e = this.#exports;
    const code = e.ahiru_last_error();
    // Where the engine knows it: a byte offset into the UTF-8 SQL for syntax errors.
    const pos = typeof e.ahiru_last_error_pos === 'function' ? e.ahiru_last_error_pos() >>> 0 : NO_POS;
    return new AhiruError(code, pos === NO_POS ? { sql } : { sql, position: pos });
  }

  #checkMemory(sql) {
    const used = this.#exports.ahiru_heap_used() >>> 0;
    if (this.#memoryLimit > 0 && used > this.#memoryLimit) {
      throw new AhiruError(Code.LIMIT_EXCEEDED, {
        sql,
        detail: `wasm heap ${used} > memoryLimit ${this.#memoryLimit}`,
      });
    }
  }

  #assertOpen() {
    if (this.#closed) throw new AhiruError(Code.INTERNAL, { detail: 'database is closed' });
  }

  /** `#assertOpen`, plus: a trap has not left this database unusable. */
  #assertUsable() {
    this.#assertOpen();
    if (this.#dead !== null) throw new AhiruError(Code.INTERNAL, { detail: this.#dead });
  }

  /** Fails a run that close() unwound while it was suspended. */
  #assertRunning(run) {
    if (run.aborted) {
      throw new AhiruError(Code.INTERNAL, {
        detail: 'this query was aborted by close(); a partially consumed stream() must be ' +
          'return()ed (for-await does this) before closing',
      });
    }
    this.#assertOpen();
  }

  #freeSession() {
    if (this.#sessionFreed) return;
    this.#sessionFreed = true;
    // A trapped instance is never called again; it is simply dropped.
    if (!this.#trapped) this.#exports.ahiru_session_free(this.#session);
  }
}

/** `ahiru_last_error_pos` when the error carries no position. */
const NO_POS = 0xffffffff;

/**
 * The coalesced ranges fetched for one registration (each a range-cache key), most
 * recent last, forgetting the oldest beyond `MAX_REMEMBERED_JOBS`.
 *
 * `#read` consults it on every exact-key cache miss -- which includes every range's
 * first fetch -- so the ranges overlapping a request are found through a per-part
 * array sorted by offset instead of a scan over every remembered range.
 */
class JobIndex {
  /** `part:offset:len` -> job, in recency order (for eviction). */
  #lru = new Map();
  /** part -> { jobs sorted by offset, the longest length ever added }. */
  #byPart = new Map();

  add(part, offset, len) {
    const key = `${part}:${offset}:${len}`;
    if (this.#lru.has(key)) {
      const job = this.#lru.get(key);
      this.#lru.delete(key);
      this.#lru.set(key, job);
      return;
    }
    const job = { part, offset, len };
    this.#lru.set(key, job);
    let p = this.#byPart.get(part);
    if (p === undefined) {
      p = { jobs: [], maxLen: 0 };
      this.#byPart.set(part, p);
    }
    p.jobs.splice(upperBound(p.jobs, offset), 0, job);
    if (len > p.maxLen) p.maxLen = len;
    if (this.#lru.size > MAX_REMEMBERED_JOBS) {
      const [oldKey, old] = this.#lru.entries().next().value;
      this.#lru.delete(oldKey);
      const q = this.#byPart.get(old.part);
      q.jobs.splice(q.jobs.indexOf(old, lowerBound(q.jobs, old.offset)), 1);
    }
  }

  /** The remembered jobs of `part` that overlap `[offset, end)`. */
  *overlapping(part, offset, end) {
    const p = this.#byPart.get(part);
    if (p === undefined) return;
    // Every job starting at or past `end` misses; one starting more than the longest
    // length before `offset` cannot reach it.
    const from = lowerBound(p.jobs, offset - p.maxLen);
    const to = lowerBound(p.jobs, end);
    for (let i = from; i < to; i++) {
      const j = p.jobs[i];
      if (j.offset + j.len > offset) yield j;
    }
  }
}

/** The first index in offset-sorted `jobs` whose offset is `>= at`. */
function lowerBound(jobs, at) {
  let lo = 0;
  let hi = jobs.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (jobs[mid].offset < at) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** The first index in offset-sorted `jobs` whose offset is `> at`. */
function upperBound(jobs, at) {
  let lo = 0;
  let hi = jobs.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (jobs[mid].offset <= at) lo = mid + 1;
    else hi = mid;
  }
  return lo;
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
