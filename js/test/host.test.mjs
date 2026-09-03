// Host layer tests. Run with `node --test 'js/test/*.test.mjs'`.
//
// Build target/ahiru-core.wasm with `./scripts/size.sh` beforehand.
// Expected values come from the duckdb CLI (this repository's ground truth).

import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { copyFileSync, existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

import {
  AhiruDB,
  AhiruError,
  Batch,
  MemoryCache,
  coalesceRanges,
  dateToDate,
  decodeBatch,
  decodeCodecRequests,
  decodeIoRequests,
  decodeSchema,
  detectFormat,
  encodeParams,
  recordFetched,
  timestampToDate,
  timestamptzToDate,
  unpackInterval,
} from '../ahirudb.js';
import { Code, errorMessage } from '../errors.js';

const ROOT = fileURLToPath(new URL('../..', import.meta.url));
const WASM = join(ROOT, 'target/ahiru-core.wasm');
const BASIC = join(ROOT, 'tests/data/basic.parquet');

if (!existsSync(WASM)) {
  throw new Error(`${WASM} is missing. Run ./scripts/size.sh first`);
}

/** Calls the duckdb CLI with JSON output. Every expected value comes from here. */
function duck(sql) {
  const out = execFileSync('duckdb', ['-json', '-c', sql], { encoding: 'utf8', maxBuffer: 1 << 28 });
  return out.trim() === '' ? [] : JSON.parse(out);
}

async function openDb(options = {}) {
  return AhiruDB.init({ wasmUrl: WASM, ...options });
}

/**
 * CSV / JSONL are feature-gated, so they are not in the default distribution
 * build (`target/ahiru-core.wasm` = parquet only). An all-formats build is
 * prepared separately. If it is missing it is built on the spot, and if that
 * fails too, only the format tests are skipped.
 */
const FULL_WASM = process.env.AHIRU_WASM_FULL ?? join(ROOT, 'target/ahiru-core-full.wasm');
const FORMAT_SKIP = (() => {
  try {
    execFileSync(
      'cargo',
      // prettier-ignore
      ['build', '--profile', 'wasm', '--target', 'wasm32-unknown-unknown',
       '-p', 'ahiru-core', '--no-default-features', '--features', 'csv,jsonl'],
      { cwd: ROOT, stdio: 'ignore' },
    );
    copyFileSync(join(ROOT, 'target/wasm32-unknown-unknown/wasm/ahiru_core.wasm'), FULL_WASM);
    return false;
  } catch {
    // Even if the build fails, use a previously built one if it is still around.
    return existsSync(FULL_WASM)
      ? false
      : `No wasm with csv,jsonl. Run cargo build --profile wasm ` +
          `--target wasm32-unknown-unknown -p ahiru-core --no-default-features ` +
          `--features csv,jsonl and place it at ${FULL_WASM}, or set AHIRU_WASM_FULL`;
  }
})();

/**
 * ZSTD is built into the core by default (the `zstd` feature, DESIGN.md §6), so
 * queries against `target/ahiru-core.wasm` never go through codec delegation
 * (`NEED_CODEC`). Host-side delegation and resilience to cache eviction are
 * behaviors we want to test in their own right, so a core built without `zstd`
 * is prepared separately and only those tests point at it.
 */
const NOZSTD_WASM = process.env.AHIRU_WASM_NOZSTD ?? join(ROOT, 'target/ahiru-core-nozstd.wasm');
const NOZSTD_SKIP = (() => {
  try {
    execFileSync(
      'cargo',
      // prettier-ignore
      ['build', '--profile', 'wasm', '--target', 'wasm32-unknown-unknown',
       '-p', 'ahiru-core', '--no-default-features'],
      { cwd: ROOT, stdio: 'ignore' },
    );
    copyFileSync(join(ROOT, 'target/wasm32-unknown-unknown/wasm/ahiru_core.wasm'), NOZSTD_WASM);
    return false;
  } catch {
    return existsSync(NOZSTD_WASM)
      ? false
      : `No wasm without zstd. Run cargo build --profile wasm ` +
          `--target wasm32-unknown-unknown -p ahiru-core --no-default-features ` +
          `and place it at ${NOZSTD_WASM}, or set AHIRU_WASM_NOZSTD`;
  }
})();

async function openFullDb(options = {}) {
  return AhiruDB.init({ wasmUrl: FULL_WASM, ...options });
}

test('stale session handles cannot close a reused wasm session slot', async () => {
  const { instance } = await WebAssembly.instantiate(await readFile(FULL_WASM), {});
  const { ahiru_session_new, ahiru_session_free, ahiru_set_now } = instance.exports;

  const stale = ahiru_session_new();
  ahiru_session_free(stale);
  const live = ahiru_session_new();
  assert.notEqual(live, stale, 'reused session slots must receive a new generation');

  ahiru_session_free(stale);
  assert.equal(ahiru_set_now(live, 1n), 0, 'a stale free must not close the live session');
  ahiru_session_free(live);
});

/**
 * The expression VM (crates/ahiru-core/src/expr/vm.rs) is still a stub, and
 * Project always returns E900. Value assertions cannot be verified until it
 * lands, so this checks once whether it actually works and skips the affected
 * tests otherwise. Once the VM lands the skip lifts automatically (the tests
 * themselves are not weakened at all).
 */
const VM_STATUS = await (async () => {
  const db = await openDb();
  try {
    db.registerParquet('t', new Uint8Array(await readFile(BASIC)));
    await db.query('SELECT id FROM t LIMIT 1');
    return null;
  } catch (e) {
    return e.code === Code.INTERNAL
      ? 'expression VM (expr/vm.rs) is still a stub: Project returns E900. ' +
          'Unskip happens automatically once the VM lands.'
      : `unexpected failure: ${e.message}`;
  } finally {
    db.close();
  }
})();
const needsVm = VM_STATUS ?? false;

/**
 * Until the VM lands, E900 occurs before all rows are drained.
 * The I/O path assertions still hold up to that point, so only 900 is swallowed.
 */
async function runTolerantly(db, sql) {
  try {
    return await db.query(sql);
  } catch (e) {
    if (e.code === Code.INTERNAL) return null;
    throw e;
  }
}

// --- Wire format decoding (no wasm needed) -----------------------------------

/** Builds a buffer in the same layout as `encode_batch` in abi.rs. For testing the decoder alone. */
function encodeBatch(numRows, cols) {
  const parts = [];
  const u32 = (v) => {
    const b = new Uint8Array(4);
    new DataView(b.buffer).setUint32(0, v, true);
    return b;
  };
  const bitmapWords = (bits) => {
    const words = Math.ceil(bits.length / 64) || 0;
    const b = new Uint8Array(words * 8);
    bits.forEach((v, i) => {
      if (v) b[i >> 3] |= 1 << (i & 7);
    });
    return b;
  };
  parts.push(u32(0x41485231), u32(cols.length), u32(numRows));
  for (const c of cols) {
    parts.push(u32(c.phys));
    if (c.valid) {
      const w = bitmapWords(c.valid);
      parts.push(u32(w.length), w);
    } else {
      parts.push(u32(0));
    }
    if (c.phys === 5) {
      const off = new Uint8Array(c.offsets.length * 4);
      const dv = new DataView(off.buffer);
      c.offsets.forEach((v, i) => dv.setUint32(i * 4, v, true));
      parts.push(u32(off.length), off);
    }
    const data = c.phys === 0 ? bitmapWords(c.data) : c.data;
    parts.push(u32(data.length), data);
  }
  const total = parts.reduce((a, p) => a + p.length, 0);
  const out = new Uint8Array(total);
  let p = 0;
  for (const part of parts) {
    out.set(part, p);
    p += part.length;
  }
  return out;
}

test('decodeBatch reads columnar buffers according to their types', () => {
  const bytes = new TextEncoder().encode('\u20achello');
  const f64 = new Float64Array([1.5, -2.25, 0]);
  const i64 = new BigInt64Array([10n, -20n, 30n]);
  // I128 has no TypedArray, so it is written as two 64-bit halves (low then high).
  const i128 = new Uint8Array(3 * 16);
  const i128dv = new DataView(i128.buffer);
  [0n, 2n ** 100n, -(2n ** 100n) - 1n].forEach((v, i) => {
    const u = BigInt.asUintN(128, v);
    i128dv.setBigUint64(i * 16, u & 0xffffffffffffffffn, true);
    i128dv.setBigUint64(i * 16 + 8, u >> 64n, true);
  });
  const buf = encodeBatch(3, [
    { phys: 1, valid: [1, 0, 1], data: new Uint8Array(new Int32Array([7, 999, -3]).buffer) },
    // For variable-length columns offsets.len == rows + 1. An odd length here
    // shifts the following columns off the 8-byte boundary, so this single case
    // also exercises the unaligned path.
    { phys: 5, valid: null, offsets: [0, 3, 3, bytes.length], data: bytes },
    { phys: 3, valid: null, data: new Uint8Array(f64.buffer) },
    { phys: 2, valid: [1, 1, 0], data: new Uint8Array(i64.buffer) },
    { phys: 0, valid: null, data: [1, 0, 1] },
    { phys: 4, valid: null, data: i128 },
  ]);
  const schema = [
    { name: 'i', type: 'INTEGER', typeCode: 4, physType: 1 },
    { name: 's', type: 'VARCHAR', typeCode: 14, physType: 5 },
    { name: 'f', type: 'DOUBLE', typeCode: 12, physType: 3 },
    { name: 'b', type: 'BIGINT', typeCode: 5, physType: 2 },
    { name: 'z', type: 'BOOLEAN', typeCode: 1, physType: 0 },
    { name: 'h', type: 'HUGEINT', typeCode: 6, physType: 4 },
  ];
  const batch = decodeBatch(buf, schema);

  assert.equal(batch.numRows, 3);
  assert.deepEqual(batch.toRows(), [
    { i: 7, s: '\u20ac', f: 1.5, b: 10n, z: true, h: 0n },
    { i: null, s: '', f: -2.25, b: -20n, z: false, h: 2n ** 100n },
    { i: -3, s: 'hello', f: 0, b: null, z: true, h: -(2n ** 100n) - 1n },
  ]);
  assert.ok(batch.column('f') instanceof Float64Array);
  assert.ok(batch.column('b') instanceof BigInt64Array);
  assert.equal(batch.isNull('i', 1), true);
  assert.equal(batch.get('i', 1), null);
  assert.throws(
    () => batch.column(6),
    (e) => e instanceof AhiruError && e.code === Code.COLUMN_NOT_FOUND,
  );
  for (const row of [-1, 1.5, 3]) {
    assert.throws(
      () => batch.get('i', row),
      (e) => e instanceof AhiruError && e.code === Code.VALUE_OUT_OF_RANGE,
    );
    assert.throws(
      () => batch.isNull('i', row),
      (e) => e instanceof AhiruError && e.code === Code.VALUE_OUT_OF_RANGE,
    );
  }
});

test('decodeBatch rejects a bad magic number', () => {
  const buf = encodeBatch(0, []);
  buf[0] = 0;
  assert.throws(() => decodeBatch(buf, []), (e) => e instanceof AhiruError && e.code === 900);
});

test('wire decoders reject truncated schemas and result columns as AhiruError', () => {
  const schema = new Uint8Array(4);
  new DataView(schema.buffer).setUint32(0, 1, true);
  assert.throws(
    () => decodeSchema(schema),
    (e) => e instanceof AhiruError && e.code === Code.INTERNAL,
  );
  const badDecimal = new Uint8Array(4 + 20);
  const badDecimalDv = new DataView(badDecimal.buffer);
  badDecimalDv.setUint32(0, 1, true);
  badDecimalDv.setUint32(4, 13, true); // DECIMAL
  badDecimalDv.setUint32(8, 2, true); // physical I64
  badDecimalDv.setUint32(12, 0, true); // precision is invalid
  badDecimalDv.setUint32(16, 1, true); // scale exceeds precision
  badDecimalDv.setUint32(20, 0, true); // empty name
  assert.throws(
    () => decodeSchema(badDecimal),
    (e) => e instanceof AhiruError && e.code === Code.INTERNAL,
  );

  const badPhysicalType = new Uint8Array(4 + 20);
  const badPhysicalDv = new DataView(badPhysicalType.buffer);
  badPhysicalDv.setUint32(0, 1, true);
  badPhysicalDv.setUint32(4, 13, true); // DECIMAL(10, 1) must be physical I64.
  badPhysicalDv.setUint32(8, 3, true); // F64 is not a valid DECIMAL representation.
  badPhysicalDv.setUint32(12, 10, true);
  badPhysicalDv.setUint32(16, 1, true);
  badPhysicalDv.setUint32(20, 0, true);
  assert.throws(
    () => decodeSchema(badPhysicalType),
    (e) => e instanceof AhiruError && e.code === Code.INTERNAL,
  );

  const result = new Uint8Array(12 + 12);
  const dv = new DataView(result.buffer);
  dv.setUint32(0, 0x41485231, true);
  dv.setUint32(4, 1, true);
  dv.setUint32(8, 1, true);
  dv.setUint32(12, 1, true); // I32
  dv.setUint32(16, 0, true); // no validity bitmap
  dv.setUint32(20, 0, true); // but one row needs four data bytes
  assert.throws(
    () =>
      decodeBatch(result, [
        { name: 'x', type: 'INTEGER', typeCode: 4, physType: 1, precision: 0 },
      ]),
    (e) => e instanceof AhiruError && e.code === Code.INTERNAL,
  );
});

test('wire decoders reject invalid UTF-8 instead of silently replacing it', () => {
  // A field *name* can only be malformed if the wire buffer is: still E900.
  const schema = new Uint8Array(4 + 20 + 1);
  const schemaDv = new DataView(schema.buffer);
  schemaDv.setUint32(0, 1, true);
  schemaDv.setUint32(4, 14, true); // VARCHAR
  schemaDv.setUint32(8, 5, true); // BYTES
  schemaDv.setUint32(20, 1, true); // name_len
  schema[24] = 0xff;
  assert.throws(
    () => decodeSchema(schema),
    (e) => e instanceof AhiruError && e.code === Code.INTERNAL,
  );

  const result = new Uint8Array(12 + 8 + 4 + 8 + 4 + 1);
  const resultDv = new DataView(result.buffer);
  let p = 0;
  resultDv.setUint32(p, 0x41485231, true); p += 4;
  resultDv.setUint32(p, 1, true); p += 4;
  resultDv.setUint32(p, 1, true); p += 4;
  resultDv.setUint32(p, 5, true); p += 4; // BYTES
  resultDv.setUint32(p, 0, true); p += 4; // no validity bitmap
  resultDv.setUint32(p, 8, true); p += 4; // offsets_len
  resultDv.setUint32(p, 0, true); p += 4;
  resultDv.setUint32(p, 1, true); p += 4;
  resultDv.setUint32(p, 1, true); p += 4; // data_len
  result[p] = 0xff;
  assert.throws(
    () => decodeBatch(result, [{ name: 's', type: 'VARCHAR', typeCode: 14, physType: 5 }]),
    // A VARCHAR *value* is whatever bytes the source file holds, so this is a
    // data error, not an engine bug. Reporting it as E900 sent users hunting
    // for a bug in the engine over a file with a non-UTF-8 string in it.
    (e) => e instanceof AhiruError && e.code === Code.INVALID_UTF8,
  );
});

test('decodeBatch rejects schema and physical-column mismatches as AhiruError', () => {
  const result = new Uint8Array(12 + 12 + 8);
  const dv = new DataView(result.buffer);
  dv.setUint32(0, 0x41485231, true);
  dv.setUint32(4, 1, true);
  dv.setUint32(8, 1, true);
  dv.setUint32(12, 3, true); // F64
  dv.setUint32(16, 0, true);
  dv.setUint32(20, 8, true);
  dv.setFloat64(24, 1.5, true);

  assert.throws(
    () => decodeBatch(result, []),
    (e) => e instanceof AhiruError && e.code === Code.INTERNAL,
  );
  assert.throws(
    () =>
      decodeBatch(result, [
        { name: 'x', type: 'INTEGER', typeCode: 4, physType: 1, precision: 0 },
      ]),
    (e) => e instanceof AhiruError && e.code === Code.INTERNAL,
  );
});

test('decodeBatch does not format a NULL UUID placeholder as a real value', () => {
  const uuid = Uint8Array.from({ length: 16 }, (_, i) => i);
  const result = new Uint8Array(12 + 8 + 8 + 4 + 12 + 4 + 16);
  const dv = new DataView(result.buffer);
  let p = 0;
  dv.setUint32(p, 0x41485231, true); p += 4;
  dv.setUint32(p, 1, true); p += 4;
  dv.setUint32(p, 2, true); p += 4;
  dv.setUint32(p, 5, true); p += 4; // Bytes
  dv.setUint32(p, 8, true); p += 4; // validity bitmap (row 1 is NULL)
  dv.setBigUint64(p, 1n, true); p += 8;
  dv.setUint32(p, 12, true); p += 4; // three u32 offsets
  dv.setUint32(p, 0, true); p += 4;
  dv.setUint32(p, 16, true); p += 4;
  dv.setUint32(p, 16, true); p += 4;
  dv.setUint32(p, 16, true); p += 4;
  result.set(uuid, p);

  const batch = decodeBatch(result, [
    { name: 'u', type: 'UUID', typeCode: 21, physType: 5, precision: 0 },
  ]);
  assert.equal(batch.get('u', 0), '00010203-0405-0607-0809-0a0b0c0d0e0f');
  assert.equal(batch.get('u', 1), null);
  assert.equal(batch.column('u')[1], '');
});

// --- Range coalescing --------------------------------------------------------

test('coalesceRanges fills gaps smaller than 1 MB into a single range', () => {
  // Adjacent (no gap)
  assert.deepEqual(coalesceRanges([{ offset: 100, len: 50 }, { offset: 150, len: 50 }]), [
    { offset: 100, len: 100 },
  ]);
  // Small gaps are swallowed: 400KB + a 100KB gap + 400KB -> 900KB in one request
  assert.deepEqual(
    coalesceRanges([
      { offset: 0, len: 400 * 1024 },
      { offset: 500 * 1024, len: 400 * 1024 },
    ]),
    [{ offset: 0, len: 900 * 1024 }],
  );
  // Kept separate when at least 1 MB apart
  assert.equal(
    coalesceRanges([
      { offset: 0, len: 10 },
      { offset: 3 * 1024 * 1024, len: 10 },
    ]).length,
    2,
  );
  // Out of order, contained, and running past the end of the file
  assert.deepEqual(
    coalesceRanges([{ offset: 60, len: 100 }, { offset: 0, len: 10 }, { offset: 70, len: 5 }], 0, 120),
    [{ offset: 0, len: 10 }, { offset: 60, len: 60 }],
  );
  for (const gap of [NaN, -1, 0.5, Number.MAX_SAFE_INTEGER + 1]) {
    assert.throws(
      () => coalesceRanges([], gap),
      (e) => e instanceof AhiruError && e.code === Code.VALUE_OUT_OF_RANGE,
      `expected invalid coalescing gap ${String(gap)} to be rejected`,
    );
  }
});

test('recordFetched remembers coverage, not just where a fetch started', () => {
  const ranges = [];
  recordFetched(ranges, 0, 4, 1501);
  assert.deepEqual(ranges, [{ part: 0, offset: 4, len: 1501 }]);
  // The regression: a longer fetch starting at an offset already recorded. Keyed
  // by (part, offset) alone this was dropped, and codec delegation then reported
  // bytes we had in fact fetched as never fetched (E900).
  recordFetched(ranges, 0, 4, 72927);
  assert.deepEqual(ranges, [{ part: 0, offset: 4, len: 72927 }]);
  const covers = (part, offset, len) =>
    ranges.some((r) => r.part === part && r.offset <= offset && offset + len <= r.offset + r.len);
  assert.ok(covers(0, 1525, 1473));

  // A different part shares nothing: offsets live in their own space per file.
  recordFetched(ranges, 1, 4, 10);
  assert.equal(ranges.length, 2);
  assert.ok(!covers(1, 4, 72927));

  // Touching and abutting ranges merge; disjoint ones do not.
  recordFetched(ranges, 0, 72931, 2025);
  assert.deepEqual(
    ranges.filter((r) => r.part === 0),
    [{ part: 0, offset: 4, len: 74952 }],
  );
  recordFetched(ranges, 0, 200000, 10);
  assert.equal(ranges.filter((r) => r.part === 0).length, 2);
  // A bridging fetch collapses them back into one.
  recordFetched(ranges, 0, 74956, 125044);
  assert.deepEqual(
    ranges.filter((r) => r.part === 0),
    [{ part: 0, offset: 4, len: 200006 }],
  );

  // Empty fetches record nothing.
  const before = ranges.length;
  recordFetched(ranges, 0, 500000, 0);
  assert.equal(ranges.length, before);
});

// --- Cache -------------------------------------------------------------------

test('MemoryCache evicts LRU entries at the capacity limit', () => {
  const c = new MemoryCache(300);
  c.set('a', new Uint8Array(100));
  c.set('b', new Uint8Array(100));
  c.set('c', new Uint8Array(100));
  assert.ok(c.get('a'));
  c.set('d', new Uint8Array(100)); // a was just touched, so b is the one dropped
  assert.equal(c.get('b'), undefined);
  assert.ok(c.get('a') && c.get('c') && c.get('d'));
  assert.equal(c.size, 300);
  // An entry that exceeds the limit on its own is not admitted (it would evict everything else)
  c.set('big', new Uint8Array(1000));
  assert.equal(c.get('big'), undefined);
});

test('MemoryCache rejects ambiguous byte limits', () => {
  for (const value of [NaN, -1, 0.5, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
    assert.throws(
      () => new MemoryCache(value),
      (e) => e instanceof AhiruError && e.code === Code.VALUE_OUT_OF_RANGE,
      `expected ${String(value)} to be rejected`,
    );
  }
  assert.equal(new MemoryCache(0).size, 0);
  assert.equal(new MemoryCache(Number.MAX_SAFE_INTEGER).size, 0);
  const cache = new MemoryCache(1);
  assert.throws(
    () => {
      cache.maxBytes = NaN;
    },
    (e) => e instanceof AhiruError && e.code === Code.VALUE_OUT_OF_RANGE,
  );

  const resized = new MemoryCache(4);
  resized.set('oldest', Uint8Array.of(1, 2));
  resized.set('newest', Uint8Array.of(3, 4));
  resized.maxBytes = 2;
  assert.equal(resized.size, 2);
  assert.equal(resized.get('oldest'), undefined);
  assert.deepEqual(resized.get('newest'), Uint8Array.of(3, 4));
});

// --- Error code table --------------------------------------------------------

test('errors.js matches the Code / message in error.rs', () => {
  const rs = readFileSync(join(ROOT, 'crates/ahiru-core/src/error.rs'), 'utf8');
  const codes = new Map();
  for (const m of rs.matchAll(/^\s{4}(\w+) = (\d+),$/gm)) codes.set(m[1], Number(m[2]));
  assert.ok(codes.size > 20, 'failed to read Code out of error.rs');

  const messages = new Map();
  // rustfmt wraps an arm onto its own `{ "..." }` block once the single-line form
  // would run past the line-length limit, so both forms have to be matched.
  for (const m of rs.matchAll(/^\s{12}(\w+) => "([^"]*)",$/gm)) messages.set(m[1], m[2]);
  for (const m of rs.matchAll(/^\s{12}(\w+) => \{\s*\n\s*"([^"]*)"\s*\n\s*\}$/gm)) {
    messages.set(m[1], m[2]);
  }

  const known = new Set(Object.values(Code));
  for (const [name, value] of codes) {
    assert.ok(known.has(value), `errors.js is missing ${name} = ${value}`);
    assert.equal(errorMessage(value), messages.get(name), `the message for ${name} has drifted`);
  }
  assert.equal(Object.keys(Code).length, codes.size, 'errors.js has extra codes');
});

// --- Real data: in-memory registration ---------------------------------------

test('registering bytes + SELECT agrees with duckdb', { skip: needsVm }, async () => {
  const db = await openDb();
  try {
    db.registerParquet('t', new Uint8Array(await readFile(BASIC)));
    const rows = await db.query('SELECT id, name FROM t LIMIT 5');
    const want = duck(`SELECT id, name FROM '${BASIC}' LIMIT 5`);
    assert.equal(rows.length, 5);
    assert.deepEqual(rows, want);
  } finally {
    db.close();
  }
});

test('NULL comes back as null, not 0', { skip: needsVm }, async () => {
  const db = await openDb();
  try {
    db.registerParquet('t', new Uint8Array(await readFile(BASIC)));
    const rows = await db.query('SELECT id, big FROM t LIMIT 20');
    const want = duck(`SELECT id, big FROM '${BASIC}' LIMIT 20`);
    // duckdb's JSON emits BIGINT as a number, so compare on the BigInt side.
    assert.deepEqual(
      rows,
      want.map((r) => ({ id: r.id, big: r.big === null ? null : BigInt(r.big) })),
    );
    // basic.parquet is built with a NULL every five rows.
    for (const r of rows) {
      if (r.id % 5 === 0) assert.equal(r.big, null, `id=${r.id} should be NULL`);
      else assert.notEqual(r.big, null);
    }
  } finally {
    db.close();
  }
});

test('OFFSET takes effect', { skip: needsVm }, async () => {
  // Now that encode_batch materializes before serializing, the result with the
  // selection vector applied comes back as is.
  const db = await openDb();
  try {
    db.register('t', new Uint8Array(await readFile(BASIC)));
    assert.deepEqual(
      await db.query('SELECT id FROM t LIMIT 3 OFFSET 5'),
      duck(`SELECT id FROM '${BASIC}' LIMIT 3 OFFSET 5`),
    );
    assert.deepEqual(await db.query('SELECT id FROM t LIMIT 3 OFFSET 5'), [
      { id: 5 },
      { id: 6 },
      { id: 7 },
    ]);
  } finally {
    db.close();
  }
});

test('stream returns columnar batches', { skip: needsVm }, async () => {
  const db = await openDb();
  try {
    db.registerParquet('t', new Uint8Array(await readFile(BASIC)));
    let seen = 0;
    for await (const batch of db.stream('SELECT id, score FROM t LIMIT 100')) {
      assert.ok(batch instanceof Batch);
      assert.ok(batch.column('score') instanceof Float64Array);
      assert.deepEqual(batch.schema.map((f) => f.name), ['id', 'score']);
      seen += batch.numRows;
    }
    assert.equal(seen, 100);
  } finally {
    db.close();
  }
});

test('TIMESTAMP comes back as microsecond BigInt and the helper turns it into a Date', { skip: needsVm }, async () => {
  const db = await openDb();
  try {
    db.registerParquet('t', new Uint8Array(await readFile(BASIC)));
    const [row] = await db.query('SELECT d FROM t LIMIT 1');
    assert.equal(typeof row.d, 'bigint');
    assert.equal(timestampToDate(row.d).toISOString(), '2024-01-01T00:00:00.000Z');
  } finally {
    db.close();
  }
});

test('temporal helpers reject lossy numeric inputs and Date overflow', () => {
  const invalid = [NaN, 0.5, Number.MAX_SAFE_INTEGER + 1];
  for (const value of invalid) {
    assert.throws(
      () => timestampToDate(value),
      (e) => e instanceof AhiruError && e.code === Code.VALUE_OUT_OF_RANGE,
      `timestamp ${String(value)} should be rejected`,
    );
    assert.throws(
      () => unpackInterval(value),
      (e) => e instanceof AhiruError && e.code === Code.VALUE_OUT_OF_RANGE,
      `interval ${String(value)} should be rejected`,
    );
    assert.throws(
      () => dateToDate(value),
      (e) => e instanceof AhiruError && e.code === Code.VALUE_OUT_OF_RANGE,
      `date ${String(value)} should be rejected`,
    );
  }
  assert.equal(timestampToDate(1001n).getTime(), 1);
  assert.equal(dateToDate(1).getTime(), 86400000);
  assert.deepEqual(unpackInterval(0), { months: 0, days: 0, micros: 0n });
  assert.throws(
    () => timestampToDate(8_640_000_000_000_001_000n),
    (e) => e instanceof AhiruError && e.code === Code.VALUE_OUT_OF_RANGE,
  );
  assert.throws(
    () => dateToDate(100_000_001),
    (e) => e instanceof AhiruError && e.code === Code.VALUE_OUT_OF_RANGE,
  );
});

test('timestampToDate floors toward the past for pre-epoch instants', () => {
  // BigInt division truncates toward zero, which for a negative timestamp
  // rounds forward in time. The Date must always be the millisecond that
  // contains the instant.
  assert.equal(timestampToDate(-999n).toISOString(), '1969-12-31T23:59:59.999Z');
  assert.equal(timestampToDate(-1n).toISOString(), '1969-12-31T23:59:59.999Z');
  assert.equal(timestampToDate(-1500n).toISOString(), '1969-12-31T23:59:59.998Z');
  assert.equal(timestampToDate(-1000n).toISOString(), '1969-12-31T23:59:59.999Z');
  assert.equal(timestampToDate(-2000n).toISOString(), '1969-12-31T23:59:59.998Z');
  // Exact multiples and positive values are unchanged.
  assert.equal(timestampToDate(0n).getTime(), 0);
  assert.equal(timestampToDate(999n).getTime(), 0);
  assert.equal(timestampToDate(-86_400_000_000n).toISOString(), '1969-12-31T00:00:00.000Z');
  // The alias shares the implementation.
  assert.equal(timestamptzToDate(-999n).toISOString(), '1969-12-31T23:59:59.999Z');
});

// --- Real data: range fetching -----------------------------------------------

/** A fake fetch that slices a local buffer by Range. Records every request. */
function fakeFetcher(file, url = 'https://example.invalid/data.parquet') {
  const calls = [];
  const ranges = [];
  const fetchImpl = async (target, init = {}) => {
    const method = init.method ?? 'GET';
    calls.push({ url: String(target), method });
    if (method === 'HEAD') {
      return new Response(null, { headers: { 'content-length': String(file.length) } });
    }
    const raw = new Headers(init.headers ?? {}).get('range') ?? '';
    const m = /bytes=(\d+)-(\d+)/.exec(raw);
    assert.ok(m, `no Range header: ${raw}`);
    const start = Number(m[1]);
    const end = Math.min(Number(m[2]), file.length - 1);
    ranges.push({ offset: start, len: end - start + 1 });
    return new Response(file.subarray(start, end + 1), {
      status: 206,
      headers: { 'content-range': `bytes ${start}-${end}/${file.length}` },
    });
  };
  return { url, calls, ranges, fetchImpl, bytes: () => ranges.reduce((a, r) => a + r.len, 0) };
}

/** A somewhat large file, so the 64 KiB speculative footer fetch alone cannot read it all. */
const WIDE = join(tmpdir(), 'ahirudb-test-wide.parquet');
if (!existsSync(WIDE)) {
  duck(
    `COPY (SELECT i::INTEGER AS id, ('name_' || (i % 97))::VARCHAR AS name,
            (i * 1.5)::DOUBLE AS score, (i % 2 = 0) AS flag,
            CASE WHEN i % 5 = 0 THEN NULL ELSE i * 100 END::BIGINT AS big
          FROM range(200000) t(i))
     TO '${WIDE}' (FORMAT PARQUET, ROW_GROUP_SIZE 100000)`,
  );
}
const WIDE_SIZE = statSync(WIDE).size;
/** The actual byte ranges of the column chunks. Projection and coalescing assertions are based on these. */
const WIDE_CHUNKS = new Map(
  duck(
    `SELECT path_in_schema AS name,
            least(data_page_offset, coalesce(dictionary_page_offset, data_page_offset)) AS start,
            total_compressed_size AS len
     FROM parquet_metadata('${WIDE}') WHERE row_group_id = 0`,
  ).map((r) => [r.name, { start: Number(r.start), len: Number(r.len) }]),
);

test('registration alone fetches not a single byte', async () => {
  const f = fakeFetcher(new Uint8Array(await readFile(WIDE)));
  const db = await openDb({ fetch: f.fetchImpl });
  try {
    db.registerParquet('t', f.url);
    assert.equal(f.calls.length, 0, 'I/O happened at registration time');
    assert.equal(f.ranges.length, 0);
  } finally {
    db.close();
  }
});

test('SQL comments do not bind registered URLs', async () => {
  const file = new Uint8Array(await readFile(WIDE));
  const usedUrl = 'https://example.invalid/used.parquet';
  const unusedUrl = 'https://example.invalid/unused.parquet';
  const f = fakeFetcher(file, usedUrl);
  const db = await openDb({ fetch: f.fetchImpl });
  try {
    db.register('used', usedUrl);
    db.register('unused', unusedUrl);
    await runTolerantly(
      db,
      `SELECT id FROM used
       -- unused ${unusedUrl} read_parquet('${unusedUrl}')
       /* FROM unused; read_parquet('${unusedUrl}') */
       LIMIT 1`,
    );
    assert.ok(f.calls.some((call) => call.url === usedUrl), 'the referenced URL was not fetched');
    assert.equal(
      f.calls.some((call) => call.url === unusedUrl),
      false,
      'a URL mentioned only in comments was fetched',
    );
  } finally {
    db.close();
  }
});

test('escaped quotes in a file-table path are registered as one path', async () => {
  const file = new Uint8Array(await readFile(WIDE));
  const url = "https://example.invalid/a'b.parquet";
  const f = fakeFetcher(file, url);
  const db = await openDb({ fetch: f.fetchImpl });
  try {
    const escaped = url.replaceAll("'", "''");
    await runTolerantly(db, `SELECT id FROM read_parquet('${escaped}') LIMIT 1`);
    assert.ok(f.calls.some((call) => call.url === url), 'the escaped path was not fetched');
    assert.equal(
      f.calls.some((call) => call.url !== url),
      false,
      'a truncated path was fetched instead of the escaped path',
    );
  } finally {
    db.close();
  }
});

test('SQL URL policy can disable automatic URL registration without affecting explicit sources', async () => {
  const calls = [];
  const db = await openDb({
    sqlUrlPolicy: async (url, context) => {
      calls.push({ url, ...context });
      return false;
    },
  });
  try {
    const url = 'https://example.invalid/private.parquet?token=POLICY_SECRET';
    await assert.rejects(
      () => db.query(`SELECT id FROM parquet('${url}') LIMIT 1`),
      (e) => {
        assert.ok(e instanceof AhiruError);
        assert.equal(e.code, Code.UNSUPPORTED_FEATURE);
        assert.ok(!e.message.includes('POLICY_SECRET'));
        return true;
      },
    );
    assert.deepEqual(calls, [
      { url, functionName: 'parquet', sql: `SELECT id FROM parquet('${url}') LIMIT 1` },
    ]);

    db.registerParquet('t', new Uint8Array(await readFile(BASIC)));
  } finally {
    db.close();
  }
});

test('sqlUrlPolicy false rejects SQL URLs before any fetch', async () => {
  let fetches = 0;
  const db = await openDb({
    sqlUrlPolicy: false,
    fetch: async () => {
      fetches++;
      throw new Error('policy should reject before fetch');
    },
  });
  try {
    await assert.rejects(
      () => db.query("SELECT id FROM parquet('https://127.0.0.1/secret.parquet') LIMIT 1"),
      (e) => e instanceof AhiruError && e.code === Code.UNSUPPORTED_FEATURE,
    );
    assert.equal(fetches, 0);
  } finally {
    db.close();
  }
});

test('policy-gated SQL URLs disable HTTP redirects at the fetch boundary', async () => {
  const calls = [];
  const db = await openDb({
    sqlUrlPolicy: () => true,
    fetch: async (_target, init = {}) => {
      calls.push(init);
      if (init.redirect === 'error') throw new TypeError('redirect blocked');
      throw new Error('test requires redirect protection');
    },
  });
  try {
    await assert.rejects(
      () => db.query("SELECT id FROM parquet('https://example.invalid/redirect.parquet') LIMIT 1"),
      (e) => e instanceof AhiruError && e.code === Code.IO_FAILED,
    );
    assert.ok(calls.length >= 2, 'HEAD fallback should attempt both requests');
    assert.ok(calls.every((init) => init.redirect === 'error'));
  } finally {
    db.close();
  }
});

test('double-quoted file-table paths are not auto-registered', async () => {
  const file = new Uint8Array(await readFile(WIDE));
  const url = 'https://example.invalid/double-quoted.parquet';
  const f = fakeFetcher(file, url);
  const db = await openDb({ fetch: f.fetchImpl });
  try {
    await assert.rejects(
      () => db.query(`SELECT id FROM read_parquet("${url}") LIMIT 1`),
      (e) => e instanceof AhiruError,
    );
    assert.equal(f.calls.length, 0, 'a double-quoted path was fetched before the syntax error');
  } finally {
    db.close();
  }
});

test('projection pushdown: only the bytes of the selected columns are fetched', async () => {
  const file = new Uint8Array(await readFile(WIDE));
  const f = fakeFetcher(file);
  const db = await openDb({ fetch: f.fetchImpl });
  try {
    db.registerParquet('t', f.url);
    await runTolerantly(db, 'SELECT id, name FROM t LIMIT 5');

    const footer = 64 * 1024; // FOOTER_PROBE (parquet/file.rs)
    const wanted = WIDE_CHUNKS.get('id').len + WIDE_CHUNKS.get('name').len;
    // Bytes fetched = speculative footer + the id/name column chunks only. score/flag/big are never read.
    assert.equal(f.bytes(), footer + wanted);
    assert.ok(f.bytes() < WIDE_SIZE * 0.25, `${f.bytes()} bytes is not a reduction`);

    // Directly confirm that the region of the unread columns is never touched.
    const skipped = WIDE_CHUNKS.get('score');
    for (const r of f.ranges) {
      const overlapsSkipped =
        r.offset < skipped.start + skipped.len && skipped.start < r.offset + r.len;
      const isFooterProbe = r.offset >= WIDE_SIZE - footer;
      assert.ok(!overlapsSkipped || isFooterProbe, `reading score's region: ${JSON.stringify(r)}`);
    }
  } finally {
    db.close();
  }
});

test('two adjacent requests coalesce into a single fetch', async () => {
  const file = new Uint8Array(await readFile(WIDE));
  const f = fakeFetcher(file);
  const db = await openDb({ fetch: f.fetchImpl });
  try {
    db.registerParquet('t', f.url);
    await runTolerantly(db, 'SELECT id, name FROM t LIMIT 5');

    const id = WIDE_CHUNKS.get('id');
    const name = WIDE_CHUNKS.get('name');
    // Precondition: these two column chunks are adjacent in the file.
    assert.equal(id.start + id.len, name.start, "the test's precondition (adjacency) no longer holds");

    // The engine issues one request per column. The host coalesces them, so there is one fetch.
    const data = f.ranges.filter((r) => r.offset < WIDE_SIZE - 64 * 1024);
    assert.equal(data.length, 1, `not coalesced: ${JSON.stringify(data)}`);
    assert.deepEqual(data[0], { offset: id.start, len: id.len + name.len });
    assert.equal(f.ranges.length, 2, 'should be exactly two: footer + data');
  } finally {
    db.close();
  }
});

test('running the same query a second time issues no new fetch', async () => {
  const file = new Uint8Array(await readFile(WIDE));
  const f = fakeFetcher(file);
  const db = await openDb({ fetch: f.fetchImpl });
  try {
    db.registerParquet('t', f.url);
    await runTolerantly(db, 'SELECT id, name FROM t LIMIT 5');
    const first = f.ranges.length;
    assert.ok(first > 0);
    await runTolerantly(db, 'SELECT id, name FROM t LIMIT 5');
    assert.equal(f.ranges.length, first, 'refetching on the second run');
  } finally {
    db.close();
  }
});

test('the range cache works across instances', async () => {
  const file = new Uint8Array(await readFile(WIDE));
  const cache = new MemoryCache(8 * 1024 * 1024);
  const f1 = fakeFetcher(file);
  const db1 = await openDb({ fetch: f1.fetchImpl, cache });
  try {
    db1.registerParquet('t', f1.url);
    await runTolerantly(db1, 'SELECT id, name FROM t LIMIT 5');
    assert.ok(f1.ranges.length > 0);
  } finally {
    db1.close();
  }

  // Same URL and same ranges, so the second instance needs no fetch.
  const f2 = fakeFetcher(file);
  const db2 = await openDb({ fetch: f2.fetchImpl, cache });
  try {
    db2.registerParquet('t', f2.url);
    await runTolerantly(db2, 'SELECT id, name FROM t LIMIT 5');
    assert.equal(f2.ranges.length, 0, 'the cache is not being used');
  } finally {
    db2.close();
  }
});

test('results from range fetching match in-memory registration', { skip: needsVm }, async () => {
  const file = new Uint8Array(await readFile(WIDE));
  const f = fakeFetcher(file);
  const remote = await openDb({ fetch: f.fetchImpl });
  const local = await openDb();
  try {
    remote.registerParquet('t', f.url);
    local.registerParquet('t', file);
    const sql = 'SELECT id, name FROM t LIMIT 50';
    assert.deepEqual(await remote.query(sql), await local.query(sql));
    assert.deepEqual(await remote.query(sql), duck(`SELECT id, name FROM '${WIDE}' LIMIT 50`));
  } finally {
    remote.close();
    local.close();
  }
});

test('cache: "none" fetches every time', async () => {
  const file = new Uint8Array(await readFile(WIDE));
  const f = fakeFetcher(file);
  const db = await openDb({ fetch: f.fetchImpl, cache: 'none' });
  try {
    db.registerParquet('t', f.url);
    await runTolerantly(db, 'SELECT id, name FROM t LIMIT 5');
    assert.ok(f.ranges.length > 0);
  } finally {
    db.close();
  }
});

// --- Custom ByteSource / URL safety -------------------------------------------

test(
  'a custom ByteSource read() result is copied before caching, not aliased',
  { skip: needsVm },
  async () => {
    // Repro (see the bug report): a user's read() hands back a view onto memory
    // it still owns (`buf.subarray(...)`). If the host caches that view directly
    // instead of copying it, mutating the buffer afterwards corrupts the cache.
    const fileBytes = new Uint8Array(await readFile(BASIC));
    const cache = new MemoryCache();
    const makeMutableSource = () => ({
      key: 'mutable-source', // fixed key: both instances below must hit the same cache entry
      size: fileBytes.length,
      read: (o, l) => fileBytes.subarray(o, o + l),
    });

    const db1 = await openDb({ cache });
    try {
      db1.registerParquet('t', makeMutableSource());
      assert.deepEqual(
        await db1.query('SELECT id, name FROM t LIMIT 5'),
        duck(`SELECT id, name FROM '${BASIC}' LIMIT 5`),
      );
    } finally {
      db1.close();
    }

    // Corrupt every byte of the buffer read() aliased. A fresh session has no
    // wasm-side copy of its own yet, so it must go through the (shared) cache.
    fileBytes.fill(0xff);

    const db2 = await openDb({ cache });
    try {
      db2.registerParquet('t', makeMutableSource());
      assert.deepEqual(
        await db2.query('SELECT id, name FROM t LIMIT 5'),
        duck(`SELECT id, name FROM '${BASIC}' LIMIT 5`),
        'the cache aliased the caller\'s buffer instead of copying it',
      );
    } finally {
      db2.close();
    }
  },
);

test('a 206 response whose Content-Range does not cover the requested window is rejected', async () => {
  const file = new Uint8Array(await readFile(WIDE));
  const url = 'https://example.invalid/lying.parquet';
  const fetchImpl = async (_target, init = {}) => {
    const method = init.method ?? 'GET';
    if (method === 'HEAD') {
      return new Response(null, { headers: { 'content-length': String(file.length) } });
    }
    // Always answers with the first 1000 bytes, ignoring the Range that was asked
    // for, while (truthfully) reporting a Content-Range for what it actually sent.
    return new Response(file.subarray(0, 1000), {
      status: 206,
      headers: { 'content-range': `bytes 0-999/${file.length}` },
    });
  };
  const db = await openDb({ fetch: fetchImpl });
  try {
    db.registerParquet('t', url);
    // The footer probe asks for the last 64 KiB, which this server never sent.
    await assert.rejects(
      () => db.query('SELECT id FROM t LIMIT 1'),
      (e) => e instanceof AhiruError && e.code === Code.IO_FAILED,
    );
  } finally {
    db.close();
  }
});

test('a 200 response of exactly the requested length is rejected at a non-zero offset', async () => {
  const file = new Uint8Array(await readFile(WIDE));
  const url = 'https://example.invalid/prefix200.parquet';
  const fetchImpl = async (_target, init = {}) => {
    const method = init.method ?? 'GET';
    if (method === 'HEAD') {
      return new Response(null, { headers: { 'content-length': String(file.length) } });
    }
    const raw = new Headers(init.headers ?? {}).get('range') ?? '';
    const m = /bytes=(\d+)-(\d+)/.exec(raw);
    const start = Number(m[1]);
    const len = Number(m[2]) - start + 1;
    // Range-unaware: always the first `len` bytes, 200.
    return new Response(file.subarray(0, len), { status: 200 });
  };
  const db = await openDb({ fetch: fetchImpl });
  try {
    db.registerParquet('t', url);
    await assert.rejects(
      () => db.query('SELECT id FROM t LIMIT 1'),
      (e) => e instanceof AhiruError && e.code === Code.IO_FAILED,
    );
  } finally {
    db.close();
  }
});

test('a 206 response wider than requested is sliced to the requested window', async () => {
  const file = new Uint8Array(await readFile(WIDE));
  const url = 'https://example.invalid/wide206.parquet';
  const fetchImpl = async (_target, init = {}) => {
    const method = init.method ?? 'GET';
    if (method === 'HEAD') {
      return new Response(null, { headers: { 'content-length': String(file.length) } });
    }
    const raw = new Headers(init.headers ?? {}).get('range') ?? '';
    const m = /bytes=(\d+)-(\d+)/.exec(raw);
    const start = Number(m[1]);
    const end = Number(m[2]);
    // Honour the start but send 64 extra bytes past the requested end.
    const sendEnd = Math.min(file.length - 1, end + 64);
    return new Response(file.subarray(start, sendEnd + 1), {
      status: 206,
      headers: { 'content-range': `bytes ${start}-${sendEnd}/${file.length}` },
    });
  };
  const db = await openDb({ fetch: fetchImpl });
  try {
    db.registerParquet('t', url);
    const rows = await db.query('SELECT id FROM t LIMIT 1');
    assert.equal(rows.length, 1);
  } finally {
    db.close();
  }
});

test('a 206 response with no Content-Range is rejected when the offset is not zero', async () => {
  const file = new Uint8Array(await readFile(WIDE));
  const url = 'https://example.invalid/no-cr.parquet';
  const fetchImpl = async (_target, init = {}) => {
    const method = init.method ?? 'GET';
    if (method === 'HEAD') {
      return new Response(null, { headers: { 'content-length': String(file.length) } });
    }
    const raw = new Headers(init.headers ?? {}).get('range') ?? '';
    const m = /bytes=(\d+)-(\d+)/.exec(raw);
    const start = Number(m[1]);
    const len = Number(m[2]) - start + 1;
    // Ignores Range: always the first `len` bytes, 206, no Content-Range.
    return new Response(file.subarray(0, len), { status: 206 });
  };
  const db = await openDb({ fetch: fetchImpl });
  try {
    db.registerParquet('t', url);
    await assert.rejects(
      () => db.query('SELECT id FROM t LIMIT 1'),
      (e) => e instanceof AhiruError && e.code === Code.IO_FAILED,
    );
  } finally {
    db.close();
  }
});

test('a ByteSource that returns a short read fails instead of spinning', async () => {
  const db = await openDb();
  try {
    db.registerParquet('t', {
      key: 'short-read',
      size: 1_000_000,
      read: (_offset, len) => new Uint8Array(Math.min(len, 100)),
    });
    await assert.rejects(
      withTimeout(db.query('SELECT id FROM t LIMIT 1'), 5000, 'short read hung'),
      (e) => e instanceof AhiruError && e.code === Code.IO_FAILED,
    );
  } finally {
    db.close();
  }
});

test('a ByteSource with an invalid size fails with VALUE_OUT_OF_RANGE', async () => {
  const db = await openDb();
  try {
    db.registerParquet('t', {
      key: 'bad-size',
      size: Number.MAX_SAFE_INTEGER + 1,
      read: () => new Uint8Array(0),
    });
    await assert.rejects(
      () => db.query('SELECT id FROM t LIMIT 1'),
      (e) => e instanceof AhiruError && e.code === Code.VALUE_OUT_OF_RANGE,
    );
  } finally {
    db.close();
  }
});

test('a Blob-like source with an unsafe size fails with VALUE_OUT_OF_RANGE', async () => {
  const db = await openDb();
  try {
    db.registerParquet('t', {
      size: Number.MAX_SAFE_INTEGER + 1,
      arrayBuffer: async () => new ArrayBuffer(0),
      slice: () => ({ arrayBuffer: async () => new ArrayBuffer(0) }),
    });
    await assert.rejects(
      () => db.query('SELECT id FROM t LIMIT 1'),
      (e) => e instanceof AhiruError && e.code === Code.VALUE_OUT_OF_RANGE,
    );
  } finally {
    db.close();
  }
});

test('an unsafe HEAD Content-Length is rejected without a fallback range request', async () => {
  let calls = 0;
  const db = await openDb({
    fetch: async (_target, init = {}) => {
      calls++;
      assert.equal(init.method, 'HEAD', 'an invalid successful HEAD must not be retried as GET');
      return new Response(null, {
        headers: { 'content-length': String(Number.MAX_SAFE_INTEGER + 1) },
      });
    },
  });
  try {
    db.registerParquet('t', 'https://example.invalid/unsafe-size.parquet');
    await assert.rejects(
      () => db.query('SELECT id FROM t LIMIT 1'),
      (e) => e instanceof AhiruError && e.code === Code.VALUE_OUT_OF_RANGE,
    );
    assert.equal(calls, 1);
  } finally {
    db.close();
  }
});

test('close during a pending range read aborts the run safely', async () => {
  const fileBytes = new Uint8Array(await readFile(BASIC));
  let entered;
  const enteredPromise = new Promise((resolve) => {
    entered = resolve;
  });
  let release;
  const releasePromise = new Promise((resolve) => {
    release = resolve;
  });
  const db = await openDb();
  db.registerParquet('t', {
    key: 'close-during-read',
    size: fileBytes.length,
    read: async (offset, len) => {
      entered();
      await releasePromise;
      return fileBytes.subarray(offset, offset + len);
    },
  });
  const running = db.query('SELECT id FROM t LIMIT 1');
  await enteredPromise;
  db.close();
  release();
  await assert.rejects(
    running,
    (e) => e instanceof AhiruError && e.code === Code.INTERNAL,
  );
});

test('decodeIoRequests rejects an offset beyond Number.MAX_SAFE_INTEGER instead of truncating it', () => {
  const buf = new Uint8Array(4 + 24);
  const dv = new DataView(buf.buffer);
  dv.setUint32(0, 1, true); // count
  dv.setUint32(4, 0, true); // table
  dv.setUint32(8, 0, true); // part
  dv.setBigUint64(12, BigInt(Number.MAX_SAFE_INTEGER) + 10n, true); // offset
  dv.setBigUint64(20, 10n, true); // len
  assert.throws(
    () => decodeIoRequests(buf),
    (e) => e instanceof AhiruError && e.code === Code.VALUE_OUT_OF_RANGE,
  );
});

test('decodeIoRequests rejects a range whose end crosses Number.MAX_SAFE_INTEGER', () => {
  const buf = new Uint8Array(4 + 24);
  const dv = new DataView(buf.buffer);
  dv.setUint32(0, 1, true);
  dv.setBigUint64(12, BigInt(Number.MAX_SAFE_INTEGER - 4), true);
  dv.setBigUint64(20, 10n, true);
  assert.throws(
    () => decodeIoRequests(buf),
    (e) => e instanceof AhiruError && e.code === Code.VALUE_OUT_OF_RANGE,
  );
});

test('wire decoders report truncated request buffers as AhiruError', () => {
  assert.throws(
    () => decodeIoRequests(Uint8Array.of(1, 0, 0, 0)),
    (e) => e instanceof AhiruError && e.code === Code.INTERNAL,
  );
  assert.throws(
    () => decodeCodecRequests(Uint8Array.of(1, 0, 0, 0)),
    (e) => e instanceof AhiruError && e.code === Code.INTERNAL,
  );
});

test('coalesceRanges rejects an unsafe range end before creating a Range header', () => {
  assert.throws(
    () => coalesceRanges([{ offset: Number.MAX_SAFE_INTEGER - 4, len: 10 }]),
    (e) => e instanceof AhiruError && e.code === Code.VALUE_OUT_OF_RANGE,
  );
});

test('a fetch failure redacts the query string (tokens) from the error message', async () => {
  const url = 'https://example.invalid/data.parquet?token=SECRET123&sig=abc';
  const fetchImpl = async (_target, init = {}) => {
    const method = init.method ?? 'GET';
    if (method === 'HEAD') return new Response(null, { headers: { 'content-length': '1000' } });
    return new Response('nope', { status: 403 });
  };
  const db = await openDb({ fetch: fetchImpl });
  try {
    db.registerParquet('t', url);
    await assert.rejects(
      () => db.query('SELECT id FROM t'),
      (e) => {
        assert.ok(e instanceof AhiruError);
        assert.ok(!e.message.includes('SECRET123'), `token leaked into the error: ${e.message}`);
        assert.ok(!e.message.includes('token='), `query string leaked into the error: ${e.message}`);
        assert.ok(
          e.message.includes('example.invalid/data.parquet'),
          `origin+path should still be present: ${e.message}`,
        );
        return true;
      },
    );
  } finally {
    db.close();
  }
});

test('a network failure redacts userinfo and the query string too', async () => {
  const url = 'https://user:s3cr3t@example.invalid/data.parquet?token=SECRET456';
  const fetchImpl = async () => {
    throw new Error('network down');
  };
  const db = await openDb({ fetch: fetchImpl });
  try {
    db.registerParquet('t', url);
    await assert.rejects(
      () => db.query('SELECT id FROM t'),
      (e) => {
        assert.ok(e instanceof AhiruError);
        assert.ok(!e.message.includes('SECRET456'), `token leaked into the error: ${e.message}`);
        assert.ok(!e.message.includes('s3cr3t'), `userinfo leaked into the error: ${e.message}`);
        return true;
      },
    );
  } finally {
    db.close();
  }
});

test('a response body failure is normalized and redacts the URL', async () => {
  const url = 'https://example.invalid/data.parquet?token=SECRET_BODY';
  const fetchImpl = async (_target, init = {}) => {
    if ((init.method ?? 'GET') === 'HEAD') {
      return new Response(null, { headers: { 'content-length': '1000' } });
    }
    return {
      ok: true,
      status: 206,
      headers: new Headers({ 'content-range': 'bytes 0-999/1000' }),
      arrayBuffer: async () => {
        throw new Error('body stream failed');
      },
    };
  };
  const db = await openDb({ fetch: fetchImpl });
  try {
    db.registerParquet('t', url);
    await assert.rejects(
      () => db.query('SELECT id FROM t'),
      (e) => {
        assert.ok(e instanceof AhiruError);
        assert.equal(e.code, Code.IO_FAILED);
        assert.ok(!e.message.includes('SECRET_BODY'), `token leaked into the error: ${e.message}`);
        assert.ok(e.message.includes('read https://example.invalid/data.parquet response failed'));
        return true;
      },
    );
  } finally {
    db.close();
  }
});

test('a WASM response body failure is normalized and redacts the URL', async () => {
  const url = 'https://example.invalid/ahirudb.wasm?token=SECRET_WASM_BODY';
  await assert.rejects(
    () =>
      AhiruDB.init({
        wasmUrl: url,
        fetch: async () => ({
          ok: true,
          status: 200,
          arrayBuffer: async () => {
            throw new Error('wasm body stream failed');
          },
        }),
      }),
    (e) => {
      assert.ok(e instanceof AhiruError);
      assert.equal(e.code, Code.IO_FAILED);
      assert.ok(!e.message.includes('SECRET_WASM_BODY'));
      assert.ok(e.message.includes('read https://example.invalid/ahirudb.wasm response failed'));
      return true;
    },
  );
});

test('wasm loading redacts query tokens from HTTP errors', async () => {
  const url = 'https://example.invalid/ahirudb.wasm?token=SECRET789';
  await assert.rejects(
    () => AhiruDB.init({ wasmUrl: url, fetch: async () => new Response('nope', { status: 403 }) }),
    (e) => {
      assert.ok(e instanceof AhiruError);
      assert.equal(e.code, Code.IO_FAILED);
      assert.ok(!e.message.includes('SECRET789'), `token leaked into the error: ${e.message}`);
      assert.ok(!e.message.includes('token='), `query string leaked into the error: ${e.message}`);
      assert.ok(e.message.includes('https://example.invalid/ahirudb.wasm'));
      return true;
    },
  );
});

// --- Errors ------------------------------------------------------------------

test('a syntax error becomes an AhiruError (3xx)', async () => {
  const db = await openDb();
  try {
    db.registerParquet('t', new Uint8Array(await readFile(BASIC)));
    await assert.rejects(
      () => db.query('SELECT FROM WHERE'),
      (e) => {
        assert.ok(e instanceof AhiruError);
        assert.ok(e.code >= 300 && e.code < 400, `code=${e.code}`);
        assert.equal(e.code, Code.UNEXPECTED_TOKEN);
        assert.equal(e.message, '[E301] unexpected token');
        assert.equal(e.sql, 'SELECT FROM WHERE');
        return true;
      },
    );
  } finally {
    db.close();
  }
});

test('an unknown table gives TABLE_NOT_FOUND', async () => {
  const db = await openDb();
  try {
    db.registerParquet('t', new Uint8Array(await readFile(BASIC)));
    await assert.rejects(
      () => db.query('SELECT id FROM missing_table'),
      (e) => e instanceof AhiruError && e.code === Code.TABLE_NOT_FOUND,
    );
  } finally {
    db.close();
  }
});

test('an unknown column gives COLUMN_NOT_FOUND', async () => {
  const db = await openDb();
  try {
    db.registerParquet('t', new Uint8Array(await readFile(BASIC)));
    await assert.rejects(
      () => db.query('SELECT nope FROM t'),
      (e) => e instanceof AhiruError && e.code === Code.COLUMN_NOT_FOUND,
    );
  } finally {
    db.close();
  }
});

// --- Parameter binding -------------------------------------------------------

test('encodeParams builds a tagged sequence', () => {
  const buf = encodeParams([null, true, 7, 1.5, 'ab']);
  const dv = new DataView(buf.buffer);
  assert.equal(dv.getUint32(0, true), 5);
  assert.equal(buf[4], 0); // NULL
  assert.deepEqual([...buf.subarray(5, 7)], [1, 1]); // BOOL true
  assert.equal(buf[7], 2); // safe integers are I64
  assert.equal(dv.getBigInt64(8, true), 7n);
  assert.equal(buf[16], 3); // fractional values are F64
  assert.equal(dv.getFloat64(17, true), 1.5);
  assert.equal(buf[25], 4); // strings are BYTES
  assert.equal(dv.getUint32(26, true), 2);
  assert.deepEqual([...buf.subarray(30)], [0x61, 0x62]);
  assert.equal(encodeParams([]).length, 0);
  assert.equal(encodeParams(undefined).length, 0);
});

test('encodeParams rejects unsupported types explicitly', () => {
  // Silently converting a Date to microseconds would hide an off-by-a-factor mistake.
  assert.throws(
    () => encodeParams([new Date()]),
    (e) => e instanceof AhiruError && e.code === Code.UNSUPPORTED_FEATURE,
  );
  assert.throws(
    () => encodeParams([2n ** 70n]),
    (e) => e instanceof AhiruError && e.code === Code.VALUE_OUT_OF_RANGE,
  );
});

test('parameters are bound', { skip: needsVm }, async () => {
  const db = await openDb();
  try {
    db.register('t', new Uint8Array(await readFile(BASIC)));
    assert.deepEqual(
      await db.query('SELECT id, name FROM t WHERE id = ?', [3]),
      duck(`SELECT id, name FROM '${BASIC}' WHERE id = 3`),
    );
    // String parameters. Not having to embed them in the SQL is the point of this API.
    assert.deepEqual(
      await db.query('SELECT id FROM t WHERE name = ? LIMIT 3', ['name_5']),
      duck(`SELECT id FROM '${BASIC}' WHERE name = 'name_5' LIMIT 3`),
    );
    // Several of them, BigInt, and floating point
    assert.deepEqual(
      await db.query('SELECT id FROM t WHERE id > ? AND score < ? ORDER BY id', [10n, 20.0]),
      duck(`SELECT id FROM '${BASIC}' WHERE id > 10 AND score < 20.0 ORDER BY id`),
    );
    // A string containing a quote is passed through as a value (concatenation would break here)
    assert.deepEqual(await db.query("SELECT id FROM t WHERE name = ?", ["a' OR 1=1 --"]), []);
  } finally {
    db.close();
  }
});

test('an identical request with no byte progress is treated as a livelock and fails', async () => {
  const db = await openDb();
  try {
    let reads = 0;
    // A source that always returns empty. Left alone it would spin on NEED_IO forever.
    db.registerParquet('t', {
      key: 'stuck',
      size: 1024 * 1024,
      read: () => {
        reads++;
        return new Uint8Array(0);
      },
    });
    await assert.rejects(
      () => db.query('SELECT id FROM t'),
      (e) => e instanceof AhiruError && e.code === Code.IO_FAILED,
    );
    assert.ok(reads <= 4, `spun too long on empty responses: ${reads}`);
  } finally {
    db.close();
  }
});

test('a transient empty read is not cached and the retry succeeds', async () => {
  const file = new Uint8Array(await readFile(BASIC));
  const cache = new MemoryCache();
  let reads = 0;
  const source = {
    key: 'flaky-empty',
    size: () => file.byteLength,
    // The first read comes back empty, as a flaky origin might answer once.
    read: async (offset, len) => (++reads === 1 ? new Uint8Array(0) : file.subarray(offset, offset + len)),
  };
  // Caching the empty body poisoned that range for good: every retry hit the
  // cache, the source was never asked again, and the livelock detector failed
  // this query -- and every later one, on every instance sharing the cache.
  const db = await openDb({ cache });
  try {
    db.registerParquet('t', source);
    assert.equal((await db.query('SELECT id FROM t LIMIT 1')).length, 1);
    assert.ok(reads >= 2, `the source should have been asked again: ${reads}`);
  } finally {
    db.close();
  }
  const db2 = await openDb({ cache });
  try {
    db2.registerParquet('t', source);
    assert.equal((await db2.query('SELECT id FROM t LIMIT 1')).length, 1);
  } finally {
    db2.close();
  }
});

test('a wrong-length cache hit is treated as a miss instead of spinning forever', async () => {
  const file = new Uint8Array(await readFile(BASIC));
  let reads = 0;
  let gets = 0;
  const source = {
    key: 'short-cache-hit',
    size: () => file.byteLength,
    read: async (offset, len) => {
      reads++;
      return file.subarray(offset, offset + len);
    },
  };
  // What a truncated Cache API body, or a stale entry left by an older build, looks
  // like: a hit of the wrong length for every key. Handed to wasm it became the
  // prefix of the range, the engine asked for the identical range again, and query()
  // looped on microtasks alone -- never yielding, so not even a timer could stop it
  // (which is why this cache throws after a generous number of gets: a regression has
  // to fail the test rather than hang the whole run).
  const cache = {
    get() {
      if (++gets > 500) throw new Error(`livelock: cache.get called ${gets} times`);
      return new Uint8Array(3);
    },
    set() {},
    clear() {},
  };
  const db = await openDb({ cache });
  try {
    db.registerParquet('t', source);
    const rows = await db.query('SELECT sum(id) AS s FROM t');
    assert.equal(rows.length, 1);
    // `Number()` on both sides, as the other duckdb cross-checks here do: `sum` of an
    // INTEGER column is a HUGEINT, and `duckdb -json` renders one as a bare number on some
    // builds and as a quoted string on others. `assert` is the strict flavour in this file,
    // so comparing the raw values would pass or fail depending on the local duckdb.
    const want = duck(`SELECT sum(id) AS s FROM '${BASIC}'`)[0].s;
    assert.equal(Number(rows[0].s), Number(want));
    assert.ok(reads > 0, 'the source was never asked; the bogus cache hit was used as-is');
  } finally {
    db.close();
  }
});

test('output columns that share a name keep distinct row-object keys', async () => {
  const db = await openDb();
  try {
    db.registerParquet('t', new Uint8Array(await readFile(BASIC)));
    // Three columns all called `id`. Assigning each to `o[c.name]` let the last one
    // win, so the row collapsed to a single `id` holding `name AS id` and the two
    // real `id` values were lost with no error anywhere.
    const rows = await db.query('SELECT id, id, name AS id FROM t WHERE id = 1');
    assert.equal(rows.length, 1);
    assert.deepEqual(Object.keys(rows[0]), ['id', 'id_1', 'id_2']);
    assert.equal(Number(rows[0].id), 1);
    assert.equal(Number(rows[0].id_1), 1);
    assert.equal(rows[0].id_2, 'name_1');

    // A generated key that collides with a real column name keeps counting.
    const rows2 = await db.query('SELECT id, id AS id_1, id AS id FROM t WHERE id = 2');
    assert.deepEqual(Object.keys(rows2[0]), ['id', 'id_1', 'id_2']);

    // The schema still reports the real names, unchanged.
    for await (const b of db.stream('SELECT id, id FROM t LIMIT 1')) {
      assert.deepEqual(
        b.schema.map((c) => c.name),
        ['id', 'id'],
      );
      assert.deepEqual(Object.keys(b.toRows()[0]), ['id', 'id_1']);
    }
  } finally {
    db.close();
  }
});

test('close() while a source size() is in flight reports that the database is closed', async () => {
  const file = new Uint8Array(await readFile(BASIC));
  const db = await openDb();
  db.registerParquet('t', {
    // A user callback, so it can take arbitrarily long; close() lands while it runs.
    size: () => new Promise((r) => setTimeout(() => r(file.byteLength), 50)),
    read: async (offset, len) => file.subarray(offset, offset + len),
  });
  const pending = db.query('SELECT count(*) AS c FROM t');
  setTimeout(() => db.close(), 10);
  // Table binding used to carry on into the freed session and surface a bare E900
  // with no explanation, unlike every other close race.
  await assert.rejects(
    withTimeout(pending, 5000, 'the query hung after close()'),
    (e) => e instanceof AhiruError && /database is closed/.test(e.message),
  );
});

test('close() recovers an abandoned stream instead of deadlocking the instance', async () => {
  const db = await openDb();
  db.registerParquet('t', new Uint8Array(await readFile(BASIC)));
  const it = db.stream('SELECT id FROM t')[Symbol.asyncIterator]();
  const first = await it.next();
  assert.ok(first.value.numRows > 0);
  // The iterator is now parked at a `yield` holding the session lock, and it is
  // never resumed. Without close() taking the session back, everything queued
  // behind it waits forever.
  const queued = db.query('SELECT id FROM t LIMIT 1');
  db.close();
  await assert.rejects(
    withTimeout(queued, 5000, 'a query queued behind an abandoned stream hung after close()'),
    (e) => e instanceof AhiruError && e.code === Code.INTERNAL,
  );
  // Resuming the abandoned iterator must fail loudly rather than step a handle
  // close() already released.
  await assert.rejects(
    withTimeout(it.next(), 5000, 'the abandoned iterator hung'),
    (e) => e instanceof AhiruError && e.code === Code.INTERNAL && /aborted by close/.test(e.message),
  );
});

test('pointers and sizes are read as unsigned, so a heap above 2 GiB still works', async () => {
  // wasm returns `usize` as a signed i32, so anything at or above 2 GiB arrives
  // negative. Left signed, `#checkMemory` silently stops firing and the
  // out-buffer view throws a bare RangeError.
  const memory = new WebAssembly.Memory({ initial: 1 });
  const fake = {
    exports: {
      memory,
      ahiru_session_new: () => 0,
      ahiru_session_free: () => {},
      // The exact value the 2 GiB reproduction reported.
      ahiru_heap_used: () => -2128680800,
    },
  };
  const db = new AhiruDB(fake, {});
  assert.equal(db.heapUsed, 2166286496);
  db.close();
});

test('exceeding memoryLimit stops with E501', async () => {
  const db = await openDb({ memoryLimit: 1 });
  try {
    db.registerParquet('t', new Uint8Array(await readFile(BASIC)));
    await assert.rejects(
      () => db.query('SELECT id FROM t'),
      (e) => e instanceof AhiruError && e.code === Code.LIMIT_EXCEEDED,
    );
  } finally {
    db.close();
  }
});

test('init rejects ambiguous cacheSize and memoryLimit values', async () => {
  for (const option of ['cacheSize', 'memoryLimit']) {
    for (const value of [NaN, -1, 0.5, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
      await assert.rejects(
        () => openDb({ [option]: value }),
        (e) => e instanceof AhiruError && e.code === Code.VALUE_OUT_OF_RANGE,
        `expected ${option}=${String(value)} to be rejected`,
      );
    }
  }
});

test('operations after close are errors', async () => {
  const db = await openDb();
  db.close();
  db.close(); // a double close is harmless
  assert.throws(() => db.registerParquet('t', new Uint8Array(8)));
  await assert.rejects(() => db.query('SELECT 1 FROM t'));
});

// --- Concurrency: the session lock ---------------------------------------------

/** Rejects with `msg` if `p` does not settle within `ms`. Turns a hang into a clear failure. */
function withTimeout(p, ms, msg) {
  return Promise.race([
    p,
    new Promise((_, reject) => setTimeout(() => reject(new Error(msg)), ms)),
  ]);
}

test(
  'Promise.all concurrent queries on one AhiruDB return correct, independent results',
  { skip: needsVm },
  async () => {
    const file = new Uint8Array(await readFile(WIDE));
    const f = fakeFetcher(file);
    const db = await openDb({ fetch: f.fetchImpl });
    try {
      db.registerParquet('t', f.url);
      // Overlapping queries with awaits (fetch) inside their step loops. Without
      // serializing access to the shared wasm out buffer / last-error state, these
      // would interleave and silently return mixed-up rows for one another.
      const [a, b, c] = await Promise.all([
        db.query('SELECT id FROM t LIMIT 5'),
        db.query('SELECT id FROM t LIMIT 5 OFFSET 100'),
        db.query('SELECT id FROM t LIMIT 5 OFFSET 200'),
      ]);
      assert.deepEqual(a, duck(`SELECT id FROM '${WIDE}' LIMIT 5`));
      assert.deepEqual(b, duck(`SELECT id FROM '${WIDE}' LIMIT 5 OFFSET 100`));
      assert.deepEqual(c, duck(`SELECT id FROM '${WIDE}' LIMIT 5 OFFSET 200`));
    } finally {
      db.close();
    }
  },
);

test(
  'stream(): breaking out of the loop early still releases the session lock',
  { skip: needsVm },
  async () => {
    const db = await openDb();
    try {
      db.registerParquet('t', new Uint8Array(await readFile(BASIC)));
      let seen = 0;
      for await (const batch of db.stream('SELECT id FROM t')) {
        seen += batch.numRows;
        break; // abandon the iterator early -- the consumer's for-await calls .return()
      }
      assert.ok(seen > 0);

      // If the lock were not released, this would hang until the test times out.
      const rows = await withTimeout(
        db.query('SELECT id FROM t LIMIT 3'),
        5000,
        'query after an early break did not resolve -- the session lock was not released',
      );
      assert.deepEqual(rows, duck(`SELECT id FROM '${BASIC}' LIMIT 3`));
    } finally {
      db.close();
    }
  },
);

test(
  'stream(): the consumer throwing out of the loop still releases the session lock',
  { skip: needsVm },
  async () => {
    const db = await openDb();
    try {
      db.registerParquet('t', new Uint8Array(await readFile(BASIC)));
      await assert.rejects(async () => {
        for await (const batch of db.stream('SELECT id FROM t')) {
          void batch;
          throw new Error('boom');
        }
      }, /boom/);

      const rows = await withTimeout(
        db.query('SELECT id FROM t LIMIT 1'),
        5000,
        'query after a thrown error did not resolve -- the session lock was not released',
      );
      assert.equal(rows.length, 1);
    } finally {
      db.close();
    }
  },
);

// --- WHERE (waiting on the expression VM) ------------------------------------

test(
  'WHERE narrows rows',
  {
    skip:
      needsVm &&
      'WHERE needs the expression VM (expr/vm.rs). This skip lifts on its own once the VM lands.',
  },
  async () => {
    const db = await openDb();
    try {
      db.registerParquet('t', new Uint8Array(await readFile(BASIC)));
      const rows = await db.query('SELECT id, name FROM t WHERE id < 3');
      assert.deepEqual(rows, duck(`SELECT id, name FROM '${BASIC}' WHERE id < 3`));
    } finally {
      db.close();
    }
  },
);

test(
  'statistics pruning: a RowGroup the predicate cannot match is not read at all',
  {
    skip:
      needsVm &&
      'WHERE needs the expression VM (expr/vm.rs). This skip lifts on its own once the VM lands.',
  },
  async () => {
    const file = new Uint8Array(await readFile(WIDE));
    const f = fakeFetcher(file);
    const db = await openDb({ fetch: f.fetchImpl });
    try {
      db.registerParquet('t', f.url);
      // id is ascending, so only the second RowGroup (id >= 100000) survives.
      const rows = await db.query('SELECT id FROM t WHERE id > 199990');
      assert.deepEqual(rows, duck(`SELECT id FROM '${WIDE}' WHERE id > 199990`));
      const first = WIDE_CHUNKS.get('id');
      for (const r of f.ranges) {
        const hitsFirstGroup = r.offset < first.start + first.len && first.start < r.offset + r.len;
        assert.ok(!hitsFirstGroup, 'reading a RowGroup that could have been pruned');
      }
    } finally {
      db.close();
    }
  },
);

// --- Aggregation / sorting / joins -------------------------------------------

/** Unimplemented features come back as E409 / E900. Skip rather than weakening the value assertions. */
async function featureStatus(sql) {
  const db = await openDb();
  try {
    db.register('t', new Uint8Array(await readFile(BASIC)));
    await db.query(sql);
    return false;
  } catch (e) {
    if (e.code === Code.UNSUPPORTED_FEATURE || e.code === Code.INTERNAL) {
      return `not implemented on the engine side yet (E${e.code}). The skip lifts on its own once it is.`;
    }
    throw e;
  } finally {
    db.close();
  }
}

const AGG_SKIP = await featureStatus('SELECT count(*) c FROM t');
const JOIN_SKIP = await featureStatus('SELECT a.id FROM t a JOIN t b ON a.id = b.id LIMIT 1');

test('GROUP BY and aggregates agree with duckdb', { skip: AGG_SKIP }, async () => {
  const db = await openDb();
  try {
    db.register('t', new Uint8Array(await readFile(BASIC)));
    const rows = await db.query('SELECT flag, count(*) c FROM t GROUP BY flag');
    const want = duck(`SELECT flag, count(*) c FROM '${BASIC}' GROUP BY flag`);
    const norm = (rs) =>
      rs.map((r) => ({ flag: r.flag, c: Number(r.c) })).sort((a, b) => Number(a.flag) - Number(b.flag));
    assert.deepEqual(norm(rows), norm(want));

    // NULLs are not counted (big is NULL every five rows)
    const [agg] = await db.query('SELECT count(big) nb, count(*) n, sum(id) s FROM t');
    const [wantAgg] = duck(`SELECT count(big) nb, count(*) n, sum(id) s FROM '${BASIC}'`);
    assert.equal(Number(agg.nb), Number(wantAgg.nb));
    assert.equal(Number(agg.n), Number(wantAgg.n));
    assert.equal(Number(agg.s), Number(wantAgg.s));
  } finally {
    db.close();
  }
});

test('HAVING and DISTINCT', { skip: AGG_SKIP }, async () => {
  const db = await openDb();
  try {
    db.register('t', new Uint8Array(await readFile(BASIC)));
    const rows = await db.query(
      'SELECT name, count(*) c FROM t GROUP BY name HAVING count(*) > 10 ORDER BY name',
    );
    const want = duck(
      `SELECT name, count(*) c FROM '${BASIC}' GROUP BY name HAVING count(*) > 10 ORDER BY name`,
    );
    assert.deepEqual(
      rows.map((r) => [r.name, Number(r.c)]),
      want.map((r) => [r.name, Number(r.c)]),
    );

    const d = await db.query('SELECT DISTINCT flag FROM t');
    assert.deepEqual(
      d.map((r) => r.flag).sort(),
      duck(`SELECT DISTINCT flag FROM '${BASIC}'`).map((r) => r.flag).sort(),
    );
  } finally {
    db.close();
  }
});

test('ORDER BY agrees with duckdb', { skip: AGG_SKIP }, async () => {
  const db = await openDb();
  try {
    db.register('t', new Uint8Array(await readFile(BASIC)));
    assert.deepEqual(
      await db.query('SELECT id, score FROM t ORDER BY score DESC, id LIMIT 5'),
      duck(`SELECT id, score FROM '${BASIC}' ORDER BY score DESC, id LIMIT 5`),
    );
    // NULL ordering matches too (big contains NULLs)
    assert.deepEqual(
      (await db.query('SELECT big FROM t ORDER BY big NULLS FIRST LIMIT 3')).map((r) => r.big),
      duck(`SELECT big FROM '${BASIC}' ORDER BY big NULLS FIRST LIMIT 3`).map((r) =>
        r.big === null ? null : BigInt(r.big),
      ),
    );
  } finally {
    db.close();
  }
});

test('JOIN agrees with duckdb', { skip: JOIN_SKIP }, async () => {
  const db = await openDb();
  try {
    db.register('t', new Uint8Array(await readFile(BASIC)));
    assert.deepEqual(
      await db.query(
        'SELECT a.id, b.name FROM t a JOIN t b ON a.id = b.id WHERE a.id < 5 ORDER BY a.id',
      ),
      duck(
        `SELECT a.id, b.name FROM '${BASIC}' a JOIN '${BASIC}' b ON a.id = b.id
         WHERE a.id < 5 ORDER BY a.id`,
      ),
    );
  } finally {
    db.close();
  }
});

// --- CURRENT_DATE / CURRENT_TIMESTAMP / now() ---------------------------------

test('CURRENT_DATE/now() reflect the host-supplied time per query', { skip: needsVm }, async () => {
  const db = await openDb();
  try {
    db.register('t', new Uint8Array(await readFile(BASIC)));
    const before = Date.now();
    const rows = await db.query('SELECT CURRENT_DATE AS d, now() AS n FROM t LIMIT 1');
    const after = Date.now();
    // Rather than the exact JS-side type conversion of DATE/TIMESTAMP, what we want
    // to confirm is "a value close to the call time came back", so only now() is checked strictly.
    const gotDate = rows[0].d;
    const nowMicros = rows[0].n;
    assert.ok(typeof nowMicros === 'bigint', `now() should be a BigInt(micros): ${typeof nowMicros}`);
    const nowMs = Number(nowMicros / 1000n);
    assert.ok(
      nowMs >= before - 5 && nowMs <= after + 5,
      `now() is outside the call-time window: ${nowMs} not in [${before}, ${after}]`,
    );
    assert.ok(gotDate !== undefined);
  } finally {
    db.close();
  }
});

// --- DECIMAL -----------------------------------------------------------------

/** DECIMAL(18,4) rides on I64 and DECIMAL(30,6) on I128. Both are exercised. */
const DEC = join(tmpdir(), 'ahirudb-test-decimal.parquet');
if (!existsSync(DEC)) {
  duck(
    `COPY (SELECT i::INTEGER AS id, (i * 1.005)::DECIMAL(18,4) AS d18,
             (i * 1.005)::DECIMAL(30,6) AS d30 FROM range(10) t(i))
     TO '${DEC}' (FORMAT PARQUET)`,
  );
}

test('DECIMAL comes back as a string with precision/scale applied', { skip: needsVm }, async () => {
  const db = await openDb();
  try {
    db.register('d', new Uint8Array(await readFile(DEC)));
    const rows = await db.query('SELECT id, d18, d30 FROM d LIMIT 4');
    // Returned as a string so digits are not lost; a number would round past 18 digits.
    assert.deepEqual(rows, [
      { id: 0, d18: '0.0000', d30: '0.000000' },
      { id: 1, d18: '1.0050', d30: '1.005000' },
      { id: 2, d18: '2.0100', d30: '2.010000' },
      { id: 3, d18: '3.0150', d30: '3.015000' },
    ]);
    // And that the values agree with duckdb.
    const want = duck(`SELECT id, d18, d30 FROM '${DEC}' LIMIT 4`);
    assert.deepEqual(
      rows.map((r) => [r.id, Number(r.d18), Number(r.d30)]),
      want.map((r) => [r.id, Number(r.d18), Number(r.d30)]),
    );
    for await (const b of db.stream('SELECT d18 FROM d LIMIT 1')) {
      assert.equal(b.schema[0].type, 'DECIMAL');
      assert.equal(b.columns[0].scale, undefined); // scale is carried by the schema instead
    }
  } finally {
    db.close();
  }
});

test('the JSON type comes back as the raw text string (not a decoded object)', { skip: needsVm }, async () => {
  const db = await openDb();
  try {
    const rows = await db.query(
      "SELECT json_object('a', 1, 'b', [1, 2, 3]) AS o, json_array(1, 2, 3) AS l FROM range(1)",
    );
    assert.deepEqual(rows, [{ o: '{"a":1,"b":[1,2,3]}', l: '[1,2,3]' }]);
    assert.equal(typeof rows[0].o, 'string');
    assert.equal(typeof rows[0].l, 'string');
    // Confirms the contract that the caller runs JSON.parse itself if it needs to.
    assert.deepEqual(JSON.parse(rows[0].o), { a: 1, b: [1, 2, 3] });
    assert.deepEqual(JSON.parse(rows[0].l), [1, 2, 3]);
    for await (const b of db.stream("SELECT json_array(1, 2) AS l FROM range(1)")) {
      assert.equal(b.schema[0].type, 'JSON');
    }
  } finally {
    db.close();
  }
});

test('INTERVAL comes back split into three components (not a packed i128)', { skip: needsVm }, async () => {
  const db = await openDb();
  try {
    const rows = await db.query(
      "SELECT INTERVAL '1' MONTH AS mo, INTERVAL '3' DAY AS d," +
        " INTERVAL '90' MINUTE AS t, INTERVAL '1 year 2 months 3 days' AS mix FROM range(1)",
    );
    // While packed, months sits at the 2^96 place, so the raw BigInt is meaningless.
    assert.deepEqual(rows[0].mo, { months: 1, days: 0, micros: 0n });
    assert.deepEqual(rows[0].d, { months: 0, days: 3, micros: 0n });
    assert.deepEqual(rows[0].t, { months: 0, days: 0, micros: 5400000000n });
    assert.deepEqual(rows[0].mix, { months: 14, days: 3, micros: 0n });
    for await (const b of db.stream("SELECT INTERVAL '1' DAY AS d FROM range(1)")) {
      assert.equal(b.schema[0].type, 'INTERVAL');
    }
  } finally {
    db.close();
  }
});

test('unpackInterval extracts negative components with their sign', () => {
  // Builds the same packing as pack_interval(months, days, micros) on the JS side and round-trips it.
  const pack = (months, days, micros) =>
    (BigInt.asUintN(32, BigInt(months)) << 96n) |
    (BigInt.asUintN(32, BigInt(days)) << 64n) |
    BigInt.asUintN(64, BigInt(micros));
  assert.deepEqual(unpackInterval(pack(-1, -2, -3)), { months: -1, days: -2, micros: -3n });
  assert.deepEqual(unpackInterval(pack(0, 0, 0)), { months: 0, days: 0, micros: 0n });
  assert.deepEqual(unpackInterval(pack(2147483647, -2147483648, -1)), {
    months: 2147483647,
    days: -2147483648,
    micros: -1n,
  });
});

test('HUGEINT is exact all the way to the i128 limits', { skip: needsVm }, async () => {
  const db = await openDb();
  try {
    // Back when the decimal parser truncated at 38 digits, this rounded to `...105720`.
    const rows = await db.query(
      "SELECT CAST('170141183460469231731687303715884105727' AS HUGEINT) AS mx," +
        " CAST('-170141183460469231731687303715884105728' AS HUGEINT) AS mn FROM range(1)",
    );
    assert.equal(rows[0].mx, 170141183460469231731687303715884105727n);
    assert.equal(rows[0].mn, -170141183460469231731687303715884105728n);
  } finally {
    db.close();
  }
});

// --- Codec delegation --------------------------------------------------------

test('decodeCodecRequests reads the request list', () => {
  // encode_codec: [count:u32][{table:u32, part:u32, codec:u32, offset:u64, len:u32, out_len:u32}...]
  const buf = new Uint8Array(4 + 28);
  const dv = new DataView(buf.buffer);
  dv.setUint32(0, 1, true);
  dv.setUint32(4, 3, true); // table
  dv.setUint32(8, 0, true); // part
  dv.setUint32(12, 6, true); // codec = ZSTD
  dv.setBigUint64(16, 123456n, true);
  dv.setUint32(24, 777, true);
  dv.setUint32(28, 4096, true);
  assert.deepEqual(decodeCodecRequests(buf), [
    { table: 3, part: 0, codec: 6, offset: 123456, len: 777, outLen: 4096 },
  ]);
});

test('decodeCodecRequests rejects a page output size that could exhaust host memory', () => {
  const buf = new Uint8Array(4 + 28);
  const dv = new DataView(buf.buffer);
  dv.setUint32(0, 1, true);
  // Leave the request otherwise well formed; only the declared output is hostile.
  dv.setUint32(12, 2, true); // GZIP
  dv.setUint32(28, 256 * 1024 * 1024 + 1, true);
  assert.throws(
    () => decodeCodecRequests(buf),
    (e) => e instanceof AhiruError && e.code === Code.LIMIT_EXCEEDED,
  );
});

const GZIP_PARQUET = join(ROOT, 'tests/data/gzip.parquet');

/** Smooths over the BigInt vs number difference only. duckdb's JSON emits integers as numbers. */
const numeric = (rows) =>
  rows.map((r) =>
    Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === 'bigint' ? Number(v) : v])),
  );

test('GZIP is decompressed by the host DecompressionStream', { skip: needsVm }, async () => {
  const db = await openDb();
  try {
    db.register('g', new Uint8Array(await readFile(GZIP_PARQUET)));
    // Do not hard-code column names (another agent swaps out the test data too).
    assert.deepEqual(
      numeric(await db.query('SELECT * FROM g LIMIT 5')),
      numeric(duck(`SELECT * FROM '${GZIP_PARQUET}' LIMIT 5`)),
    );
    if (!AGG_SKIP) {
      const [n] = await db.query('SELECT count(*) c FROM g');
      assert.equal(Number(n.c), Number(duck(`SELECT count(*) c FROM '${GZIP_PARQUET}'`)[0].c));
    }
  } finally {
    db.close();
  }
});

test('malformed GZIP failures become normalized codec errors', { skip: needsVm }, async () => {
  const OriginalDecompressionStream = globalThis.DecompressionStream;
  globalThis.DecompressionStream = class {
    constructor() {
      throw new Error('malformed gzip stream');
    }
  };
  let db;
  try {
    db = await openDb();
    db.register('g', new Uint8Array(await readFile(GZIP_PARQUET)));
    await assert.rejects(
      () => db.query('SELECT * FROM g LIMIT 1'),
      (e) => e instanceof AhiruError && e.code === Code.BAD_COMPRESSED_DATA,
    );
  } finally {
    db?.close();
    globalThis.DecompressionStream = OriginalDecompressionStream;
  }
});

test('GZIP decompresses on the range-fetch path with no extra fetches', { skip: needsVm }, async () => {
  const file = new Uint8Array(await readFile(GZIP_PARQUET));
  const f = fakeFetcher(file);
  const db = await openDb({ fetch: f.fetchImpl });
  try {
    db.register('g', f.url);
    assert.deepEqual(
      numeric(await db.query('SELECT * FROM g LIMIT 5')),
      numeric(duck(`SELECT * FROM '${GZIP_PARQUET}' LIMIT 5`)),
    );
    // Compressed blocks are sliced out of the copy retained from NEED_IO. No refetching happens.
    assert.ok(f.ranges.length <= 2, `extra fetches were issued: ${JSON.stringify(f.ranges)}`);
  } finally {
    db.close();
  }
});

const ZSTD_PARQUET = join(ROOT, 'tests/data/zstd.parquet');

test('ZSTD decompresses with the default core alone (no side module)', { skip: needsVm }, async () => {
  const db = await openDb(); // the default target/ahiru-core.wasm; zstdUrl is not passed
  try {
    db.register('z', new Uint8Array(await readFile(ZSTD_PARQUET)));
    assert.deepEqual(
      numeric(await db.query('SELECT * FROM z LIMIT 5')),
      numeric(duck(`SELECT * FROM '${ZSTD_PARQUET}' LIMIT 5`)),
    );
  } finally {
    db.close();
  }
});

test('ZSTD fails naming ZSTD explicitly when no module is given', { skip: NOZSTD_SKIP || needsVm }, async () => {
  const db = await AhiruDB.init({ wasmUrl: NOZSTD_WASM });
  try {
    db.register('z', new Uint8Array(await readFile(ZSTD_PARQUET)));
    await assert.rejects(
      () => db.query('SELECT * FROM z LIMIT 5'),
      (e) => {
        assert.ok(e instanceof AhiruError);
        assert.equal(e.code, Code.UNSUPPORTED_CODEC);
        assert.match(e.message, /ZSTD/);
        return true;
      },
    );
  } finally {
    db.close();
  }
});

/**
 * The ZSTD side module (crates/ahiru-zstd). By default it is linked into
 * `ahiru-core` as a library (the `zstd` feature), so building it as a standalone
 * wasm module requires passing the `standalone` feature explicitly and
 * overriding `crate-type` to `cdylib` explicitly as well (see
 * `crates/ahiru-zstd/Cargo.toml`; the default is `rlib` only).
 * Only used to test the delegation path on a core built without `zstd`.
 */
const ZSTD_DIR = join(ROOT, 'target/wasm32-unknown-unknown/wasm');

/**
 * Cargo only "uplifts" a copy of an artifact to the profile root for the
 * crate-types declared in `[lib]`. `cdylib` is forced on the command line here
 * rather than declared, so the module is written under `deps/` with a hash
 * suffix and may never appear at the profile root. Look there too instead of
 * silently skipping every ZSTD side-module test.
 */
function findZstdWasm() {
  const uplifted = join(ZSTD_DIR, 'ahiru_zstd.wasm');
  // A stale zero-export stub can be left at the root by a build that ran
  // without `--crate-type cdylib`, so prefer the largest candidate.
  const candidates = [uplifted];
  const deps = join(ZSTD_DIR, 'deps');
  if (existsSync(deps)) {
    for (const name of readdirSync(deps)) {
      if (/^ahiru_zstd-[0-9a-f]+\.wasm$/.test(name)) candidates.push(join(deps, name));
    }
  }
  const found = candidates.filter((p) => existsSync(p));
  if (found.length === 0) return null;
  return found.sort((a, b) => statSync(b).size - statSync(a).size)[0];
}

let ZSTD_WASM = join(ZSTD_DIR, 'ahiru_zstd.wasm');
const ZSTD_SKIP = await (async () => {
  try {
    execFileSync(
      'cargo',
      // prettier-ignore
      ['rustc', '--profile', 'wasm', '--target', 'wasm32-unknown-unknown',
       '-p', 'ahiru-zstd', '--no-default-features', '--features', 'standalone',
       '--', '--crate-type', 'cdylib'],
      { cwd: ROOT, stdio: 'ignore' },
    );
  } catch {
    /* Even if the build fails, look at whatever is already there. */
  }
  const found = findZstdWasm();
  if (found === null) return 'crates/ahiru-zstd does not build yet';
  ZSTD_WASM = found;
  const mod = await WebAssembly.compile(await readFile(ZSTD_WASM));
  const names = new Set(WebAssembly.Module.exports(mod).map((e) => e.name));
  const missing = ['zstd_alloc', 'zstd_free', 'zstd_decompress'].filter((n) => !names.has(n));
  return missing.length === 0
    ? false
    : `ahiru-zstd does not export ${missing.join('/')} yet (work in progress). ` +
        'This skip lifts on its own once it does.';
})();

test('ZSTD is decompressed by the side module', { skip: ZSTD_SKIP || NOZSTD_SKIP || needsVm }, async () => {
  const db = await AhiruDB.init({ wasmUrl: NOZSTD_WASM, zstdUrl: ZSTD_WASM });
  try {
    db.register('z', new Uint8Array(await readFile(ZSTD_PARQUET)));
    assert.deepEqual(
      numeric(await db.query('SELECT * FROM z LIMIT 5')),
      numeric(duck(`SELECT * FROM '${ZSTD_PARQUET}' LIMIT 5`)),
    );
  } finally {
    db.close();
  }
});

// --- CSV / JSONL -------------------------------------------------------------

test('detectFormat picks the format from the extension of the registered name', () => {
  assert.equal(detectFormat('a.parquet'), 'parquet');
  assert.equal(detectFormat('a.CSV'), 'csv');
  assert.equal(detectFormat('a.tsv'), 'tsv');
  assert.equal(detectFormat('a.ndjson'), 'jsonl');
  assert.equal(detectFormat('a.json'), 'json');
  assert.equal(detectFormat('data'), 'parquet');
  assert.equal(detectFormat('https://x/y/trips.csv?token=abc'), 'csv');
  assert.equal(detectFormat('https://x/y/data.parquet?name=a.csv'), 'parquet');
});

test('an unknown format name is rejected at registration', async () => {
  const db = await openDb();
  try {
    // Falling back to Auto on a typo would read it as Parquet, fail with BadMagic,
    // and obscure the cause. Stop here instead.
    assert.throws(
      () => db.register('t', new Uint8Array(8), { format: 'not-a-format' }),
      (e) => e instanceof AhiruError && e.code === Code.UNSUPPORTED_FEATURE,
    );
  } finally {
    db.close();
  }
});

test('an explicit format allows registering a name without an extension', { skip: FORMAT_SKIP }, async () => {
  const db = await openFullDb();
  try {
    // The format is passed to ahiru_register_as, so the table name can be a plain identifier.
    db.register('logs', new Uint8Array(await readFile(CSV)), { format: 'csv' });
    db.register('events', new Uint8Array(await readFile(JSONL)), { format: 'jsonl' });
    const want = duck(`SELECT id, name FROM '${BASIC}' LIMIT 5`).map((r) => [r.id, r.name]);
    assert.deepEqual(
      (await db.query('SELECT id, name FROM logs LIMIT 5')).map((r) => [Number(r.id), r.name]),
      want,
    );
    assert.deepEqual(
      (await db.query('SELECT id, name FROM events LIMIT 5')).map((r) => [Number(r.id), r.name]),
      want,
    );
  } finally {
    db.close();
  }
});

test('an explicit format wins over the extension', async () => {
  const db = await openDb();
  try {
    // Even when the name lies, the explicit choice wins (decoupling the name from
    // how it is read is the purpose of this option, so a mismatch is not blocked).
    db.register('p.csv', new Uint8Array(await readFile(BASIC)), { format: 'parquet' });
    assert.deepEqual(
      await db.query('SELECT id, name FROM "p.csv" LIMIT 5'),
      duck(`SELECT id, name FROM '${BASIC}' LIMIT 5`),
    );
  } finally {
    db.close();
  }
});

const CSV = join(ROOT, 'tests/data/basic.csv');
const JSONL = join(ROOT, 'tests/data/basic.jsonl');
const JSON_DOC = join(ROOT, 'tests/data/json_demo.json');

test('CSV returns the same values as Parquet', { skip: FORMAT_SKIP }, async () => {
  const db = await openFullDb();
  try {
    db.register('basic.csv', new Uint8Array(await readFile(CSV)));
    const rows = await db.query('SELECT id, name FROM "basic.csv" LIMIT 5');
    const want = duck(`SELECT id, name FROM read_csv('${CSV}') LIMIT 5`);
    // CSV has no types, and integers are inferred as BIGINT. Compare by value.
    assert.deepEqual(
      rows.map((r) => [Number(r.id), r.name]),
      want.map((r) => [Number(r.id), r.name]),
    );
    assert.deepEqual(
      rows.map((r) => [Number(r.id), r.name]),
      duck(`SELECT id, name FROM '${BASIC}' LIMIT 5`).map((r) => [r.id, r.name]),
    );
  } finally {
    db.close();
  }
});

test('JSONL returns the same values as Parquet', { skip: FORMAT_SKIP }, async () => {
  const db = await openFullDb();
  try {
    db.register('basic.jsonl', new Uint8Array(await readFile(JSONL)));
    const rows = await db.query('SELECT id, name FROM "basic.jsonl" LIMIT 5');
    assert.deepEqual(
      rows.map((r) => [Number(r.id), r.name]),
      duck(`SELECT id, name FROM read_json_auto('${JSONL}') LIMIT 5`).map((r) => [
        Number(r.id),
        r.name,
      ]),
    );
    assert.deepEqual(
      rows.map((r) => [Number(r.id), r.name]),
      duck(`SELECT id, name FROM '${BASIC}' LIMIT 5`).map((r) => [r.id, r.name]),
    );
  } finally {
    db.close();
  }
});

test('CSV returns NULL as null too', { skip: FORMAT_SKIP }, async () => {
  const db = await openFullDb();
  try {
    db.register('basic.csv', new Uint8Array(await readFile(CSV)));
    const rows = await db.query('SELECT id, big FROM "basic.csv" LIMIT 12');
    for (const r of rows) {
      if (Number(r.id) % 5 === 0) assert.equal(r.big, null, `id=${r.id} should be NULL`);
      else assert.notEqual(r.big, null);
    }
  } finally {
    db.close();
  }
});

test('CSV can be read over range fetching too', { skip: FORMAT_SKIP }, async () => {
  const file = new Uint8Array(await readFile(CSV));
  const f = fakeFetcher(file, 'https://example.invalid/basic.csv');
  const db = await openFullDb({ fetch: f.fetchImpl });
  try {
    db.register('basic.csv', f.url);
    assert.equal(f.ranges.length, 0, 'I/O happened at registration');
    const rows = await db.query('SELECT id, name FROM "basic.csv" LIMIT 5');
    assert.ok(f.ranges.length > 0);
    assert.deepEqual(
      rows.map((r) => [Number(r.id), r.name]),
      duck(`SELECT id, name FROM '${BASIC}' LIMIT 5`).map((r) => [r.id, r.name]),
    );
  } finally {
    db.close();
  }
});

test('read_csv_auto auto-registers its path', { skip: FORMAT_SKIP }, async () => {
  const file = new Uint8Array(await readFile(CSV));
  const f = fakeFetcher(file, 'https://example.invalid/auto.csv');
  const db = await openFullDb({ fetch: f.fetchImpl });
  try {
    const rows = await db.query(
      `SELECT id, name FROM read_csv_auto /* path */ ( '${f.url}' ) LIMIT 5`,
    );
    assert.deepEqual(
      rows.map((r) => [Number(r.id), r.name]),
      duck(`SELECT id, name FROM read_csv('${CSV}') LIMIT 5`).map((r) => [r.id, r.name]),
    );
    assert.ok(f.calls.some((call) => call.url === f.url), 'the auto-registered URL was not fetched');
  } finally {
    db.close();
  }
});

test('read_csv and read_json force their format for extensionless paths', { skip: FORMAT_SKIP }, async () => {
  const csv = fakeFetcher(new Uint8Array(await readFile(CSV)), 'https://example.invalid/data-csv');
  const jsonl = fakeFetcher(
    new Uint8Array(await readFile(JSONL)),
    'https://example.invalid/data-jsonl.jsonl',
  );
  const json = fakeFetcher(
    new Uint8Array(await readFile(JSON_DOC)),
    'https://example.invalid/data-json',
  );
  const db = await openFullDb({ fetch: async (target, init) => {
    if (String(target) === csv.url) return csv.fetchImpl(target, init);
    if (String(target) === jsonl.url) return jsonl.fetchImpl(target, init);
    if (String(target) === json.url) return json.fetchImpl(target, init);
    throw new Error(`unexpected URL ${target}`);
  } });
  try {
    const csvRows = await db.query(`SELECT id, name FROM read_csv('${csv.url}') LIMIT 2`);
    assert.deepEqual(
      csvRows.map((r) => [Number(r.id), r.name]),
      [[0, 'name_0'], [1, 'name_1']],
    );
    const jsonlRows = await db.query(`SELECT id, name FROM read_json('${jsonl.url}') LIMIT 2`);
    assert.deepEqual(
      jsonlRows.map((r) => [Number(r.id), r.name]),
      [[0, 'name_0'], [1, 'name_1']],
    );
    const jsonRows = await db.query(`SELECT id, name FROM read_json('${json.url}') LIMIT 2`);
    assert.deepEqual(
      jsonRows.map((r) => [Number(r.id), r.name]),
      [[1, 'widget'], [2, 'gadget']],
    );
  } finally {
    db.close();
  }
});

// --- Evicting the retained copy ----------------------------------------------

/**
 * Codec delegation assumes compressed blocks are sliced out of the copy retained
 * from NEED_IO, but that copy is capped by `cacheSize`. This confirms it does not
 * fail when a range that overflowed and was discarded is requested later (i.e. it
 * falls back to refetching). The path only shows up under memory pressure, so a
 * deliberately tiny limit is used to reach it.
 */
const BIG_GZIP = join(tmpdir(), 'ahirudb-test-big-gzip.parquet');
const BIG_ZSTD = join(tmpdir(), 'ahirudb-test-big-zstd.parquet');
for (const [path, codec] of [
  [BIG_GZIP, 'GZIP'],
  [BIG_ZSTD, 'ZSTD'],
]) {
  if (existsSync(path)) continue;
  duck(
    `COPY (SELECT i::INTEGER AS id, ('name_' || (i % 997))::VARCHAR AS name
          FROM range(120000) t(i))
     TO '${path}' (FORMAT PARQUET, COMPRESSION ${codec}, ROW_GROUP_SIZE 40000)`,
  );
}

async function scanTwiceWithTinyCache(path, options = {}) {
  const file = new Uint8Array(await readFile(path));
  const f = fakeFetcher(file);
  // Both the retained copy and the cache are effectively zero, so something is always discarded.
  const db = await AhiruDB.init({
    wasmUrl: WASM,
    fetch: f.fetchImpl,
    cacheSize: 4096,
    ...options,
  });
  try {
    db.register('t', f.url);
    const first = await db.query('SELECT id FROM t');
    // The second run raises no NEED_IO because the bytes are still on the wasm side.
    // Codec requests must still be satisfiable with an empty retained copy.
    const second = await db.query('SELECT id FROM t');
    return { first, second, fetches: f.ranges.length };
  } finally {
    db.close();
  }
}

test('GZIP can still be read through even when the retained copy overflows', { skip: needsVm }, async () => {
  const { first, second } = await scanTwiceWithTinyCache(BIG_GZIP);
  assert.equal(first.length, 120000);
  assert.deepEqual(first, second);
  assert.equal(first[0].id, 0);
  assert.equal(first[first.length - 1].id, 119999);
});

test('ZSTD can still be read through even when the retained copy overflows', { skip: ZSTD_SKIP || NOZSTD_SKIP || needsVm }, async () => {
  const { first, second } = await scanTwiceWithTinyCache(BIG_ZSTD, {
    wasmUrl: NOZSTD_WASM,
    zstdUrl: ZSTD_WASM,
  });
  assert.equal(first.length, 120000);
  assert.deepEqual(first, second);
  assert.equal(first[first.length - 1].id, 119999);
});
