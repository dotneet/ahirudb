// ahirudb の JS ホスト層。依存ゼロの ES モジュール（ブラウザ / Node 18+）。
//
// wasm 側との契約はすべて crates/ahiru-core/src/abi.rs にある。ステータス値・
// ワイヤ形式・エラーコードを変えるときは、あちらとこのファイルを同時に直すこと。
//
// このホストが担うのは 3 つだけ:
//   1. NEED_IO ループ … エンジンが要求したバイト範囲を結合して並列取得し、返す
//   2. 結果バッファのデコード … 列指向のリトルエンディアン表現 → JS の値
//   3. エラーメッセージの組み立て（errors.js の表）
//
// --- wasm メモリの扱い（重要）-------------------------------------------------
// `ahiru_alloc` / `ahiru_provide` は wasm のヒープを伸ばす可能性があり、伸びた
// 瞬間に既存の TypedArray ビューは detach して長さ 0 になる。silent に壊れる
// タイプのバグなので、方針を固定する:
//
//   (a) `memory.buffer` へのビューは wasm 呼び出しをまたいで保持しない。
//       必ず呼び出しの直後に `new Uint8Array(memory.buffer)` で取り直す。
//   (b) `ahiru_out_ptr()` が指すバッファは次の `ahiru_query_step` /
//       `ahiru_schema` で作り直される。値を後で使うなら、次の wasm 呼び出しの
//       前に JS 側へコピーし終えていること。
//   (c) 上記 (b) を守れる場合に限り、デコードはビューのまま行いコピーを省く
//       （`query()` はその場で行オブジェクトに詰めるのでコピー不要、
//        `stream()` は呼び出し側にバッチを渡すので必ず `.slice()` する）。

import { AhiruError, Code, errorMessage } from './errors.js';

export { AhiruError, Code, errorMessage };

// --- abi.rs と対応する定数 ---------------------------------------------------

const STATUS_BATCH_READY = 0;
const STATUS_NEED_IO = 1;
const STATUS_DONE = 2;
const STATUS_ERROR = 3;
/** 内蔵しないコーデック（GZIP / ZSTD）の展開をホストに依頼する。 */
const STATUS_NEED_CODEC = 4;

/** 結果バッファ先頭のマジック "AHR1"。 */
const RESULT_MAGIC = 0x41485231;

/** `encode_io` の 1 要素: table:u32 + part:u32 + offset:u64 + len:u64。 */
const IO_REQUEST_SIZE = 24;

/** `encode_codec` の 1 要素: table:u32 + part:u32 + codec:u32 + offset:u64 + len:u32 + out_len:u32。 */
const CODEC_REQUEST_SIZE = 28;

/** Parquet の Compression enum（parquet/mod.rs）。内蔵しないものだけ名前を持つ。 */
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

/** PhysType の数値。abi.rs の `const _: ()` アサーションで固定されている。 */
const PHYS_BOOL = 0;
const PHYS_I32 = 1;
const PHYS_I64 = 2;
const PHYS_F64 = 3;
const PHYS_I128 = 4;
const PHYS_BYTES = 5;

/** 論理型コード（`ty_code`）→ 型名。添字がそのままコード。 */
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

/** 隣接判定のしきい値。この幅未満の穴は「読んだ方が安い」として結合する。 */
const COALESCE_GAP = 1024 * 1024;

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder('utf-8');

// --- 時刻ヘルパ ---------------------------------------------------------------

/** TIMESTAMP（エポックからのマイクロ秒, BigInt）を `Date` にする。 */
export function timestampToDate(micros) {
  // Date はミリ秒精度なので、マイクロ秒の端数はここで落ちる。
  return new Date(Number(BigInt(micros) / 1000n));
}

/** DATE（エポックからの日数, number）を UTC の `Date` にする。 */
export function dateToDate(days) {
  return new Date(Number(days) * 86400000);
}

/**
 * TIMESTAMPTZ（エポックからの UTC マイクロ秒, BigInt）を `Date` にする。
 * 物理表現は TIMESTAMP と同一（このエンジンにセッションタイムゾーンの
 * 概念は無く、値は常に UTC の瞬間を表す）なので `timestampToDate` の別名。
 */
export const timestamptzToDate = timestampToDate;

/**
 * INTERVAL の物理表現（月 / 日 / マイクロ秒を 1 個の i128 に詰めたもの）を
 * `{ months, days, micros }` に開く。`vector::types` の `unpack_interval`
 * と同じ計算なので、あちらを変えたらここも変えること。
 *
 * 3 成分を別々に持つのは DuckDB / PostgreSQL と同じモデルで、月と日を
 * マイクロ秒に潰さないのは「1 か月」の長さが基準日に依存するため
 * （`pack_interval` の doc 参照）。したがってこの 3 つを 1 個の数値に
 * まとめて返すことはできない。
 */
export function unpackInterval(packed) {
  const v = BigInt(packed);
  return {
    months: Number(BigInt.asIntN(32, v >> 96n)),
    days: Number(BigInt.asIntN(32, (v >> 64n) & 0xffffffffn)),
    micros: BigInt.asIntN(64, v),
  };
}

/** 16 バイトを `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` にする。 */
function formatUuid(bytes) {
  let s = '';
  for (let i = 0; i < 16; i++) {
    if (i === 4 || i === 6 || i === 8 || i === 10) s += '-';
    s += bytes[i].toString(16).padStart(2, '0');
  }
  return s;
}

// --- レンジキャッシュ ---------------------------------------------------------

/**
 * `(source, offset, len)` をキーにしたバイト範囲の LRU キャッシュ。
 *
 * 完全一致のみを見る。部分被覆の探索は線形になるうえ、エンジンは同じ
 * RowGroup に対して毎回同じ範囲を要求するので、実用上これで当たる。
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
    // Map は挿入順を保つので、消して入れ直すだけで LRU になる。
    this.#map.delete(key);
    this.#map.set(key, v);
    return v;
  }

  set(key, bytes) {
    if (bytes.byteLength > this.maxBytes) return; // 単体で入らないものは諦める
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

/** 何も覚えないキャッシュ。`cache: "none"`。 */
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
  // "cache-api" はブラウザ限定。Node には `caches` が無いのでメモリへ落とす。
  // （Cache API 版は Response をまたぐ非同期 I/O が増えるので、まずは同じ挙動に
  //   縮退させておく。メモリキャッシュの正しさの方が優先。）
  if (spec === 'cache-api') return new MemoryCache(maxBytes);
  if (spec && typeof spec.get === 'function' && typeof spec.set === 'function') return spec;
  throw new TypeError(`unknown cache option: ${String(spec)}`);
}

// --- フォーマット判定 ---------------------------------------------------------

/** `ahiru_register_as` の format 引数。abi.rs の `format_kind` と 1:1。 */
const FORMAT_CODES = { auto: 0, parquet: 1, csv: 2, tsv: 3, jsonl: 4 };

/**
 * 登録名の拡張子からフォーマットを推定する。
 * `format::FormatKind::detect` の写しなので、あちらを変えたらここも変える。
 *
 * エンジンも同じ推定をするので、これは「明示指定が無いときに何として読まれるか」
 * を JS 側で見せるための写し。判定そのものは Auto で wasm に任せる。
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

// --- バイト供給元 -------------------------------------------------------------

let sourceSeq = 0;

/**
 * 登録されたものを共通のインタフェースに包む。
 * `{ key, size(), read(offset, len) }` の 3 つだけが要件。
 *
 * `size()` と `read()` を分けているのは、登録時に I/O させないため。
 * 総バイト長は `ahiru_register` に必要なので、初回クエリまで遅延する。
 */
function makeSource(spec, fetchImpl) {
  if (typeof spec === 'string' || spec instanceof URL) {
    return urlSource(String(spec), fetchImpl);
  }
  if (spec instanceof ArrayBuffer) return bytesSource(new Uint8Array(spec));
  if (ArrayBuffer.isView(spec)) {
    return bytesSource(new Uint8Array(spec.buffer, spec.byteOffset, spec.byteLength));
  }
  // Blob / File。Node 18+ にも Blob はある。
  if (spec && typeof spec.arrayBuffer === 'function' && typeof spec.size === 'number') {
    return blobSource(spec);
  }
  // 独自の供給元（テストや OPFS など）。
  if (spec && typeof spec.read === 'function') {
    const key = spec.key ?? `custom:${++sourceSeq}`;
    const size = typeof spec.size === 'function' ? () => spec.size() : () => spec.size;
    return { key, size: async () => Number(await size()), read: (o, l) => spec.read(o, l) };
  }
  throw new TypeError('registerParquet: url / Uint8Array / ArrayBuffer / Blob のいずれかを渡すこと');
}

function bytesSource(bytes) {
  const key = `bytes:${++sourceSeq}`;
  return {
    key,
    size: async () => bytes.byteLength,
    read: async (offset, len) => bytes.subarray(offset, offset + len),
    // メモリ上にあるものをキャッシュに積むのは二重持ちなので抑止する。
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

/** ネットワーク層の失敗も E504 に揃える（呼び出し側が code だけ見れば済むように）。 */
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
    throw new TypeError('fetch がありません。init({ fetch }) で実装を渡してください');
  }
  return {
    key: `url:${url}`,
    async size() {
      // まず HEAD。使えないサーバもあるので Content-Range へフォールバックする。
      try {
        const r = await doFetch(url, { method: 'HEAD' });
        const len = r.headers?.get('content-length');
        if (r.ok && len) return Number(len);
      } catch {
        /* HEAD 不可。下のレンジ要求で総長を得る。 */
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
      // Range を無視して全体を返すサーバがある。要求した窓だけを切り出す。
      if (r.status !== 206 && buf.byteLength > len) return buf.subarray(offset, offset + len);
      return buf;
    },
  };
}

// --- レンジ結合 ---------------------------------------------------------------

/**
 * 近接するレンジを 1 本にまとめる。
 *
 * 900 KB を 1 回取る方が、100 KB の穴を挟んだ 400 KB × 2 回より速い。
 * エンジンが RowGroup 単位で要求をまとめて返してくるのはこのためなので、
 * 1 本ずつ投げてその意図を潰さない（DESIGN.md §6）。
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
  // ファイル末尾を越える要求はサーバが 416 を返すので詰めておく。
  for (const r of out) r.len = Math.min(r.len, Math.max(0, totalLen - r.offset));
  return out.filter((r) => r.len > 0);
}

// --- ワイヤ形式のデコード -----------------------------------------------------

/**
 * `encode_io`: [count:u32][{table:u32, part:u32, offset:u64, len:u64}...]
 *
 * `part` は複数ファイルテーブル（`ahiru_register_multi`）の何ファイル目かを
 * 指す。単一ファイル登録（`ahiru_register`/`ahiru_register_as`）は常に 0。
 * `ahiru_provide` を呼ぶときにそのまま渡し戻す必要がある — `table` だけでは
 * バイトオフセットの基準となるファイルを一意に決められない。
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
 * `part` の意味は `decodeIoRequests` と同じ。
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
 * パラメータを直列化する。`[count:u32]` + 値ごとに `[tag:u8][payload]`。
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
      // 安全な整数と BigInt は I64、それ以外の number は F64 に載せる。
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
      // Date を暗黙にマイクロ秒へ直すと桁を間違えても気づけない。明示させる。
      throw new AhiruError(Code.UNSUPPORTED_FEATURE, {
        detail:
          `cannot bind ${Object.prototype.toString.call(v)}; ` +
          'use null / boolean / number / bigint / string / Uint8Array ' +
          '(TIMESTAMP は BigInt のマイクロ秒で渡す)',
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
 * DECIMAL のスケール適用。
 *
 * 値は「スケール前の整数」で届く。`number` に落とすと 18 桁以上で丸まるので、
 * **文字列**で返す。桁が問題にならない用途では `Number(v)` すればよい。
 */
function scaleDecimal(unscaled, scale) {
  const v = BigInt(unscaled);
  if (scale === 0) return v.toString();
  const neg = v < 0n;
  const digits = (neg ? -v : v).toString().padStart(scale + 1, '0');
  const cut = digits.length - scale;
  return `${neg ? '-' : ''}${digits.slice(0, cut)}.${digits.slice(cut)}`;
}

/** ビットマップ（LSB-first）の i 番目。u64 リトルエンディアン = バイト単位の LSB-first。 */
function bitAt(bits, i) {
  return (bits[i >> 3] >> (i & 7)) & 1;
}

/**
 * 整列していれば wasm メモリ上のビュー、していなければコピーの上にビューを作る。
 * wasm の out バッファは列ごとに 4 バイト境界しか保証されないので、
 * F64 / I64 の列は 8 バイト境界に乗らないことがある。
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
 * `encode_batch` をデコードする。
 *
 * ```text
 * magic:u32 num_cols:u32 num_rows:u32
 * 列ごとに: phys:u32 validity_len:u32 [validity] data_len:u32 [data]
 *           Bytes 型のみ data の前に offsets_len:u32 [offsets]
 * ```
 *
 * `copy=false` のときは戻り値の TypedArray が wasm メモリを直接指すことがある。
 * 次の wasm 呼び出しの前に読み切れる場合だけ使うこと（冒頭の方針 (b)(c)）。
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
      // ビットマップは常にコピーする。行あたり 1 バイトに展開した方が
      // 呼び出し側の扱いが楽で、サイズも（行数/8 → 行数）と小さい。
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
        // VARCHAR / JSON は UTF-8 文字列（JSON の物理表現はデコード前の
        // 生テキストそのものなので、そのまま文字列として渡してよい —
        // `JSON.parse` するかどうかは呼び出し側に委ねる）、UUID はハイフン
        // 付き 16 進文字列、それ以外（BLOB）はバイト列のまま返す。
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
        // Bool もビットマップ。0/1 の Uint8Array に展開する。
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
        // 128 ビットの TypedArray は無いので BigInt の配列にする。
        values = new Array(numRows);
        for (let i = 0; i < numRows; i++) values[i] = readI128(dv, dataAt + i * 16);
        break;
      }
      default:
        throw new AhiruError(Code.INTERNAL, { detail: `unknown phys type ${phys}` });
    }
    if (ty === TY_DECIMAL) {
      // スケール前の整数のままでは使えないので、ここで文字列に直す。
      const scale = field?.scale ?? 0;
      const scaled = new Array(numRows);
      for (let i = 0; i < numRows; i++) scaled[i] = scaleDecimal(values[i], scale);
      values = scaled;
    } else if (ty === TY_INTERVAL) {
      // 詰めたままの i128 は数値として意味を持たない（月が 2^96 の位に居る）
      // ので、3 成分に開いて返す。DECIMAL と同じ「物理表現のままでは使えない
      // 型はここで直す」扱い。
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

/** 1 バッチ（列指向）。`stream()` が渡すもの。 */
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

  /** 列の生の値（TypedArray or Array）。NULL の位置にはダミー値が入っている。 */
  column(k) {
    return this.columns[this.#index(k)].values;
  }

  /** 行が NULL かどうか。validity ビットマップを尊重する。 */
  isNull(k, row) {
    const v = this.columns[this.#index(k)].valid;
    return v !== null && v[row] === 0;
  }

  /** i 行 k 列の値。NULL は `null`。 */
  get(k, row) {
    const c = this.columns[this.#index(k)];
    if (c.valid !== null && c.valid[row] === 0) return null;
    return c.physType === PHYS_BOOL ? c.values[row] === 1 : c.values[row];
  }

  /** バッチをプレーンなオブジェクトの配列にする。 */
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

// --- コーデック委譲 -----------------------------------------------------------
//
// コアは GZIP / ZSTD を持たない。持たせないことがコアが小さい理由そのもの
// （DESIGN.md §6）。GZIP はブラウザ / Node が標準で持っている
// `DecompressionStream` に、ZSTD は別 wasm モジュールに投げる。

/** GZIP。追加バイトゼロで済むのがこの委譲の狙い。 */
async function gunzip(bytes) {
  if (typeof DecompressionStream !== 'function') {
    throw new AhiruError(Code.UNSUPPORTED_CODEC, {
      detail: 'GZIP needs DecompressionStream (browser or Node 18+)',
    });
  }
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** 別 wasm モジュールに載せた ZSTD デコーダ。初回要求まで読み込まない。 */
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
    // alloc のたびにメモリが伸びうる。ビューは毎回取り直す（コアと同じ方針）。
    new Uint8Array(e.memory.buffer).set(src, srcPtr);
    const dstPtr = e.zstd_alloc(outLen);
    const n = e.zstd_decompress(srcPtr, src.length, dstPtr, outLen);
    if (n < 0) {
      e.zstd_free(srcPtr, src.length);
      e.zstd_free(dstPtr, outLen);
      throw new AhiruError(Code.BAD_COMPRESSED_DATA, { detail: `zstd_decompress -> ${n}` });
    }
    // wasm 内に置いたままだと次の alloc で detach するのでコピーして返す。
    const out = new Uint8Array(e.memory.buffer, dstPtr, n).slice();
    e.zstd_free(srcPtr, src.length);
    e.zstd_free(dstPtr, outLen);
    return out;
  }
}

// --- wasm ロード --------------------------------------------------------------

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

// --- 本体 --------------------------------------------------------------------

export class AhiruDB {
  #exports;
  #memory;
  #session;
  #tables = new Map(); // name(lower) -> record
  #byIndex = new Map(); // wasm のテーブル添字 -> record
  #cache;
  /** キャッシュを外から渡された場合、close() で消してはいけない。 */
  #ownsCache;
  #fetch;
  #memoryLimit;
  /** コーデック委譲用に抱えておく取得済みバイトの上限。 */
  #residentLimit;
  #closed = false;
  /** ZSTD サイドモジュール。初回の NEED_CODEC まで読み込まない。 */
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
   * wasm を読み込んでセッションを 1 つ開く。
   *
   * `wasmUrl` は URL でもファイルパスでもよい（Node ではファイルとして読む）。
   * 既にバイト列やコンパイル済みモジュールがあるなら `wasmBinary` / `wasmModule`。
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
      // core は import を 1 つも持たない（no_std, panic=abort）。
      ({ instance } = await WebAssembly.instantiate(bytes, {}));
    }
    return new AhiruDB(instance, options);
  }

  /**
   * テーブルを登録する。ここでは I/O を一切行わない。
   * 総バイト長の取得もフッタ / ヘッダの読み込みも、初回クエリまで遅延する。
   *
   * `format` を渡さなければ、エンジンが**登録名の拡張子**から推定する
   * （`format::FormatKind::detect`）。渡した場合はそれが優先されるので、
   * 名前に拡張子を持たせる必要はない。
   *
   * ```js
   * db.register('logs', url, { format: 'csv' });  // FROM logs と書ける
   * db.register('logs.csv', url);                 // FROM "logs.csv"
   * ```
   *
   * 明示指定が拡張子と食い違っていても通す。名前と読み方を切り離せることが
   * このオプションの目的なので、そこを検査で塞いだら意味がない。
   */
  register(name, source, { format } = {}) {
    this.#assertOpen();
    if (typeof name !== 'string' || name.length === 0) {
      throw new TypeError('register: テーブル名が必要です');
    }
    let code = FORMAT_CODES.auto;
    if (format !== undefined && format !== null) {
      code = FORMAT_CODES[String(format).toLowerCase()];
      // 綴り間違いを Auto に落とすと、Parquet として読まれて BadMagic になり
      // 原因が分からなくなる。知らない名前はここで弾く。
      if (code === undefined || code === FORMAT_CODES.auto) {
        throw new AhiruError(Code.UNSUPPORTED_FEATURE, {
          detail: `unknown format "${format}" (parquet / csv / tsv / jsonl)`,
        });
      }
    }
    const src = makeSource(source, this.#fetch);
    // 同名は置き換える（wasm 側の catalog も同じ規則）。
    this.#tables.set(name.toLowerCase(), {
      name,
      source: src,
      index: -1,
      size: -1,
      // 実際に何として読まれるか。Auto ならエンジンと同じ規則で推定して見せる。
      format: code === FORMAT_CODES.auto ? detectFormat(name) : String(format).toLowerCase(),
      formatCode: code,
      // 供給済みバイトの控えと、これまでに取った範囲（控えを捨てた後の判断用）。
      resident: [],
      fetched: [],
    });
    return this;
  }

  /** `register` の別名。Parquet 以外も受け付ける。 */
  registerParquet(name, source, options) {
    return this.register(name, source, options);
  }

  /** 結果を全部メモリに載せて返す。 */
  async query(sql, params) {
    const rows = [];
    // copy=false: 行オブジェクトに詰め替えるまでの間しかビューを使わないので、
    // 列バッファのコピーを 1 回省ける。
    for await (const batch of this.#run(sql, params, false)) {
      // ここでは wasm メモリを直接指すビューを読んでいるので、
      // 次の step の前に行オブジェクトへ移し切る（冒頭の方針 (c)）。
      for (const r of batch.toRows()) rows.push(r);
    }
    return rows;
  }

  /** 列指向のバッチを順に返す。大きな結果を 1 つの配列に載せないための入口。 */
  stream(sql, params) {
    return this.#run(sql, params, true);
  }

  /** セッションを閉じる。以後の呼び出しはエラー。 */
  close() {
    if (this.#closed) return;
    this.#closed = true;
    this.#exports.ahiru_session_free(this.#session);
    this.#tables.clear();
    this.#byIndex.clear();
    if (this.#ownsCache) this.#cache.clear();
  }

  /** 現在 wasm ヒープが保持しているバイト数。 */
  get heapUsed() {
    return this.#exports.ahiru_heap_used();
  }

  // --- 実行ループ -----------------------------------------------------------

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
   * クエリを開始する。フッタ未取得なら `-2` が返るので、要求を満たして再試行する。
   */
  async #start(sql, params) {
    const bytes = textEncoder.encode(sql);
    const pbytes = encodeParams(params);
    let lastSignature = null;
    for (;;) {
      const e = this.#exports;
      // コアは時計を持たないので、CURRENT_DATE/CURRENT_TIMESTAMP/now() 用に
      // クエリ開始時刻をここで渡す（DESIGN.md §2）。
      e.ahiru_set_now(this.#session, BigInt(Date.now()) * 1000n);
      const ptr = e.ahiru_alloc(bytes.length);
      const pptr = pbytes.length > 0 ? e.ahiru_alloc(pbytes.length) : 0;
      // alloc でメモリが伸びうるので、書き込む直前にビューを取り直す。
      const mem = new Uint8Array(this.#memory.buffer);
      mem.set(bytes, ptr);
      if (pptr !== 0) mem.set(pbytes, pptr);
      const h = e.ahiru_query_start(this.#session, ptr, bytes.length, pptr, pbytes.length);
      e.ahiru_free(ptr, bytes.length);
      if (pptr !== 0) e.ahiru_free(pptr, pbytes.length);
      if (h >= 0) return h;
      if (h !== -2) throw this.#lastError(sql);
      // -2: フッタを読むためのバイトが足りない。
      lastSignature = await this.#pump(decodeIoRequests(this.#out()), lastSignature, sql);
    }
  }

  /**
   * コーデック展開要求を満たす。
   *
   * 圧縮ブロックは直前の NEED_IO で取得済みのはずなので、新たな取得はしない。
   * 手元に無い場合は黙って取りに行かず、エンジン側の要求がおかしいと報告する。
   */
  async #decompress(requests, sql) {
    for (const req of requests) {
      if (req.codec !== CODEC_GZIP && req.codec !== CODEC_ZSTD) {
        throw new AhiruError(Code.UNSUPPORTED_CODEC, {
          sql,
          detail: `${CODEC_NAMES[req.codec] ?? `codec ${req.codec}`} is not handled by the host`,
        });
      }
      // ZSTD が 1 つでもあるなら、並列に入る前に 1 回だけ読み込む。
      if (req.codec === CODEC_ZSTD) this.#zstd ??= await ZstdModule.load(this.#zstdOptions);
    }

    // 展開は独立なので並列に回す（GZIP は非同期ストリーム）。
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

    // wasm への受け渡しは逐次。alloc のたびにメモリが動くため。
    for (let i = 0; i < requests.length; i++) this.#provideCodec(requests[i], outputs[i], sql);
    this.#checkMemory(sql);
  }

  /** 展開済みブロックを wasm に返す。 */
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
   * 圧縮ブロックのバイトを手元から切り出す。
   *
   * 直前の NEED_IO で取った控えにあるのが正常系。控えを溢れさせて捨てた分だけは
   * 取り直す（キャッシュに残っていれば I/O にはならない）。一度も取っていない
   * 範囲を要求されたらエンジン側の不整合なので、黙って取りに行かず報告する。
   */
  async #bytesAt(rec, offset, len, sql) {
    for (const c of rec.resident) {
      if (c.offset <= offset && offset + len <= c.offset + c.bytes.length) {
        return c.bytes.subarray(offset - c.offset, offset - c.offset + len);
      }
    }
    // メモリ / Blob 供給元は切り出しに I/O が要らないので、控えを持たない。
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

  /** 出力スキーマを読む。`ahiru_schema` は out バッファを作り直すので step の前に。 */
  #readSchema(q, sql) {
    // 負値はハンドル不正。0 は「列が 0 個」なので区別する。
    const len = this.#exports.ahiru_schema(q);
    if (len < 0) throw this.#lastError(sql);
    if (len === 0) return [];
    return decodeSchema(this.#out());
  }

  /**
   * I/O 要求を満たす。結合 → 並列取得 → `ahiru_provide`。
   * 戻り値は次回の比較用シグネチャ。
   */
  async #pump(requests, lastSignature, sql) {
    const signature = requests.map((r) => `${r.table}.${r.part}/${r.offset}+${r.len}`).join(',');

    // `table` だけでは複数ファイルテーブルの何ファイル目かを区別できない
    // （オフセットはファイルごとに独立した空間）ので、`table:part` の複合
    // キーで束ねる。単一ファイル登録では常に part=0 なので、今まで通りの
    // 挙動のまま複数ファイルにも安全に拡張できる。
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

    // 往復を重ねないよう、結合後のレンジは並列に取る。
    const buffers = await Promise.all(jobs.map((j) => this.#read(j.rec, j.offset, j.len)));

    let provided = 0;
    for (let i = 0; i < jobs.length; i++) {
      const { rec, table, part, offset } = jobs[i];
      provided += this.#provide(table, part, offset, buffers[i], sql);
      // 遠隔供給元だけ控えを持つ。コーデック委譲で圧縮ブロックを求められたとき、
      // ここから切り出せれば取り直さずに済む（メモリ供給元は slice で足りる）。
      //
      // JS 側の登録 API（`register`/`registerParquet`）は今のところ 1 テーブル
      // = 1 ファイルなので `part` は常に 0 であり、`rec` が単一の控えを持つ
      // ことと矛盾しない。複数ファイル登録（`ahiru_register_multi`）を JS から
      // 使えるようにするときは、この控えをファイルごとに分ける必要がある。
      if (rec.source.cacheable !== false && !rec.resident.some((c) => c.offset === offset)) {
        rec.resident.push({ offset, bytes: buffers[i] });
        rec.fetched.push({ offset, len: buffers[i].byteLength });
        // 際限なく抱えないよう、古い順に落とす。落とした範囲が後で要るなら
        // キャッシュから取り直す（`#bytesAt`）。
        let held = rec.resident.reduce((a, c) => a + c.bytes.byteLength, 0);
        while (held > this.#residentLimit && rec.resident.length > 1) {
          held -= rec.resident.shift().bytes.byteLength;
        }
      }
    }
    this.#checkMemory(sql);

    // ライブロック検出: 同じ要求が繰り返され、1 バイトも増えていない。
    if (provided === 0 && signature === lastSignature) {
      throw new AhiruError(Code.IO_FAILED, {
        sql,
        detail: `no progress for ranges [${signature}]`,
      });
    }
    return signature;
  }

  /** キャッシュ越しにバイト範囲を読む。 */
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

  /** 取得したバイト列を wasm に渡す。渡した長さを返す。 */
  #provide(table, part, offset, bytes, sql) {
    if (bytes.byteLength === 0) return 0;
    const e = this.#exports;
    const ptr = e.ahiru_alloc(bytes.byteLength);
    if (ptr === 0) throw new AhiruError(Code.OOM, { sql });
    // alloc でメモリが伸びた可能性があるので、ここで必ず取り直す。
    new Uint8Array(this.#memory.buffer).set(bytes, ptr);
    const rc = e.ahiru_provide(this.#session, table, part, BigInt(offset), ptr, bytes.byteLength);
    e.ahiru_free(ptr, bytes.byteLength);
    if (rc !== 0) throw this.#lastError(sql);
    return bytes.byteLength;
  }

  /**
   * SQL が参照しているテーブルだけを wasm に登録する。
   *
   * 登録には総バイト長が要る（= URL なら HEAD 1 往復）ので、
   * 使いもしないテーブルの分まで往復しないよう、識別子を拾って絞る。
   */
  async #bindTables(sql) {
    const mentioned = new Set();
    const add = (s) => {
      mentioned.add(s.toLowerCase());
      // `t.id` のようなドット付きは、全体と各要素の両方を候補にする
      // （テーブル名自体が `basic.csv` のように点を含みうるため）。
      if (s.includes('.')) for (const part of s.split('.')) mentioned.add(part.toLowerCase());
    };
    for (const m of sql.matchAll(/[A-Za-z_][A-Za-z0-9_$.]*|'([^']*)'|"([^"]*)"/g)) {
      add(m[1] ?? m[2] ?? m[0]);
    }
    // `FROM parquet('https://…')` はパスをそのままテーブル名として登録する
    // 約束になっている（plan/bind.rs の resolve_from を参照）。
    // 名前は parquet だが、拡張子が .csv などなら CSV として読まれる。
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
      // 古いコアには ahiru_register_as が無い。Auto なら 4 引数版で等価だが、
      // 明示指定を黙って無視すると別のフォーマットとして読まれるので落とす。
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

  // --- 小物 -----------------------------------------------------------------

  /** 現在の out バッファのビュー。次の wasm 呼び出しまでしか有効でない。 */
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
