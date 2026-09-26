// Host tests for table binding, the wasm-heap lifecycle, trap recovery, and the
// smaller ABI surfaces (COPY results, error positions, parameter counts).
// Run with `node --test 'js/test/*.test.mjs'`.
//
// Every test here runs against one core built with the opt-in features too
// (`ddl`, `dml`, `export`, `csv`, `jsonl`), since views, CREATE TABLE and COPY are
// part of what is under test. It is built into its own target directory so it
// never races the builds `host.test.mjs` runs in parallel.

import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

import { AhiruDB, AhiruError, MemoryCache, decodeMissingTables } from '../ahirudb.js';
import { Code } from '../errors.js';

const ROOT = fileURLToPath(new URL('../..', import.meta.url));
const TARGET = join(ROOT, 'target');
const WASM = process.env.AHIRU_WASM_BINDING ?? join(TARGET, 'ahiru-core-binding.wasm');
const FEATURES = 'zstd,csv,jsonl,ddl,dml,export';

const BUILD_SKIP = (() => {
  try {
    const dir = join(TARGET, 'js-binding-test');
    execFileSync(
      'cargo',
      // prettier-ignore
      ['build', '--profile', 'wasm', '--target', 'wasm32-unknown-unknown', '-p', 'ahiru-core',
       '--no-default-features', '--features', FEATURES, '--target-dir', dir],
      { cwd: ROOT, stdio: 'ignore' },
    );
    mkdirSync(TARGET, { recursive: true });
    copyFileSync(join(dir, 'wasm32-unknown-unknown/wasm/ahiru_core.wasm'), WASM);
    return false;
  } catch {
    return existsSync(WASM)
      ? false
      : `No wasm with ${FEATURES}. Build ahiru-core with those features and place it at ` +
          `${WASM}, or set AHIRU_WASM_BINDING`;
  }
})();

const skip = BUILD_SKIP;

/** Calls the duckdb CLI (the repository's ground truth; it also writes the fixtures). */
function duck(sql) {
  execFileSync('duckdb', ['-c', sql], { encoding: 'utf8' });
}

/** Writes a fixture once per temp directory and returns its bytes. */
function fixture(name, selectSql, options = '') {
  const path = join(tmpdir(), `ahirudb-binding-${name}`);
  if (!existsSync(path)) duck(`COPY (${selectSql}) TO '${path}' (FORMAT PARQUET${options})`);
  return new Uint8Array(readFileSync(path));
}

const A = fixture('a.parquet', 'SELECT i::BIGINT AS id FROM range(3) t(i)'); // ids 0..2
const B = fixture('b.parquet', 'SELECT (100 + i)::BIGINT AS id FROM range(5) t(i)'); // ids 100..104
/** Several MiB, several row groups, three columns: enough to see what stays on the heap. */
const BIG = fixture(
  'big.parquet',
  `SELECT i::BIGINT AS id, (i * 0.5)::DOUBLE AS f, repeat('x', 16 + i % 16) AS s
   FROM range(400000) t(i)`,
  ', ROW_GROUP_SIZE 100000',
);
/** GZIP pages are always decompressed by the host (codec delegation). */
const GZ = fixture(
  'gz.parquet',
  'SELECT i::BIGINT AS id, repeat(chr((65 + i % 26)::INTEGER), 40) AS s FROM range(200000) t(i)',
  ', COMPRESSION GZIP, ROW_GROUP_SIZE 50000',
);

async function openDb(options = {}) {
  return AhiruDB.init({ wasmUrl: WASM, ...options });
}

/** A Range-capable fake server over `{ url: bytes }`. Logs every request. */
function fakeServer(files) {
  const log = [];
  const fetchImpl = async (target, init = {}) => {
    const url = String(target);
    const method = init.method ?? 'GET';
    log.push({ url, method });
    const body = files[url];
    if (body === undefined) return new Response('not found', { status: 404 });
    if (method === 'HEAD') {
      return new Response(null, { headers: { 'content-length': String(body.length) } });
    }
    const m = /bytes=(\d+)-(\d+)/.exec(new Headers(init.headers ?? {}).get('range') ?? '');
    assert.ok(m, 'expected a Range request');
    const start = Number(m[1]);
    const end = Math.min(Number(m[2]), body.length - 1);
    return new Response(body.slice(start, end + 1), {
      status: 206,
      headers: { 'content-range': `bytes ${start}-${end}/${body.length}` },
    });
  };
  return { fetchImpl, log, gets: () => log.filter((l) => l.method === 'GET').length };
}

const ids = (rows) => rows.map((r) => Number(r.id));

/**
 * Makes every instance created while `fn` runs trap in `ahiru_query_step` while
 * `arm.on` is set -- the state a real trap (stack overflow, panic) leaves behind,
 * without depending on how deep some query has to be to cause one.
 */
async function withTrapHook(arm, fn) {
  const original = WebAssembly.instantiate;
  WebAssembly.instantiate = async (...args) => {
    const result = await original.apply(WebAssembly, args);
    const instance = result instanceof WebAssembly.Instance ? result : result.instance;
    const exports = { ...instance.exports };
    const step = exports.ahiru_query_step;
    exports.ahiru_query_step = (q) => {
      if (arm.on) {
        arm.on = false;
        arm.instances.add(exports);
        throw new WebAssembly.RuntimeError('memory access out of bounds');
      }
      // A trapped instance must never be called again.
      assert.ok(!arm.instances.has(exports), 'a trapped instance was called again');
      return step(q);
    };
    const fake = { exports };
    return result instanceof WebAssembly.Instance ? fake : { module: result.module, instance: fake };
  };
  try {
    return await fn();
  } finally {
    WebAssembly.instantiate = original;
  }
}

// --- 1. wasm traps -------------------------------------------------------------

test('a trapped instance is replaced and later queries keep working', { skip }, async () => {
  const arm = { on: false, instances: new Set() };
  await withTrapHook(arm, async () => {
    const server = fakeServer({ 'https://h/a.parquet': A });
    const db = await openDb({ fetch: server.fetchImpl });
    try {
      db.register('t', 'https://h/a.parquet');
      assert.deepEqual(ids(await db.query('SELECT id FROM t ORDER BY id')), [0, 1, 2]);
      const before = server.log.length;

      arm.on = true;
      await assert.rejects(db.query('SELECT id FROM t'), (e) => {
        assert.ok(e instanceof AhiruError);
        assert.equal(e.code, Code.INTERNAL);
        assert.match(e.message, /trapped/);
        assert.ok(e.cause instanceof WebAssembly.RuntimeError);
        return true;
      });

      // The next call runs on a fresh instance: the registration is declared
      // again with the length already known, and the bytes come from the cache.
      assert.deepEqual(ids(await db.query('SELECT id FROM t ORDER BY id')), [0, 1, 2]);
      assert.equal(server.log.length, before, 'recovery went back to the network');
      db.register('u', B);
      assert.deepEqual(ids(await db.query('SELECT id FROM u ORDER BY id LIMIT 2')), [100, 101]);
    } finally {
      db.close();
    }
  });
});

test('a trap is fatal only when a fresh instance would lose SQL-created state', { skip }, async () => {
  const arm = { on: false, instances: new Set() };
  await withTrapHook(arm, async () => {
    const db = await openDb();
    try {
      db.register('t', A);
      await db.query('CREATE TABLE m AS SELECT id FROM t');
      arm.on = true;
      await assert.rejects(db.query('SELECT id FROM m'), /trapped.*create a new AhiruDB/);
      await assert.rejects(db.query('SELECT 1 FROM range(1)'), /in-memory tables and views/);
      assert.throws(() => db.register('u', B), /in-memory tables and views/);
    } finally {
      db.close();
    }
  });
});

test('a query deep enough to exhaust the stack does not break the instance', { skip }, async () => {
  const db = await openDb();
  try {
    const n = 3000;
    const ctes = Array.from(
      { length: n },
      (_, i) => `c${i} AS (SELECT ${i} x FROM ${i ? `c${i - 1}` : 'range(1)'})`,
    );
    // Whether the engine rejects this up front (a nesting limit) or traps on the
    // way, it must fail cleanly, and the instance must survive it.
    await assert.rejects(db.query(`WITH ${ctes.join(',')} SELECT * FROM c${n - 1}`), AhiruError);
    assert.deepEqual(await db.query('SELECT 42 AS x FROM range(1)'), [{ x: 42 }]);
  } finally {
    db.close();
  }
});

// --- 2. the wasm heap only holds one query's bytes ------------------------------

test('fetched bytes leave the wasm heap after each query', { skip }, async () => {
  const server = fakeServer({ 'https://h/big.parquet': BIG });
  const db = await openDb({ fetch: server.fetchImpl });
  try {
    db.register('t', 'https://h/big.parquet');
    const [{ n }] = await db.query('SELECT count(*) AS n, sum(length(s)) AS l FROM t');
    assert.equal(n, 400000n);
    assert.ok(
      db.heapUsed < BIG.length / 4,
      `heap holds ${db.heapUsed} bytes after the query (file: ${BIG.length})`,
    );

    // The next query asks for its bytes again, and the range cache answers.
    const gets = server.gets();
    await db.query('SELECT count(*) AS n, sum(length(s)) AS l FROM t');
    assert.equal(server.gets(), gets, 'a repeated query went back to the network');
  } finally {
    db.close();
  }
});

test('memoryLimit bounds each query, not everything read so far', { skip }, async () => {
  const server = fakeServer({ 'https://h/big.parquet': BIG });
  // Enough for any one of these queries, far from enough for all their columns at once.
  const db = await openDb({ fetch: server.fetchImpl, memoryLimit: 24 * 1024 * 1024 });
  try {
    db.register('t', 'https://h/big.parquet');
    for (let round = 0; round < 2; round++) {
      assert.deepEqual(await db.query('SELECT max(f) AS m FROM t'), [{ m: 199999.5 }]);
      assert.deepEqual(await db.query('SELECT max(id) AS m FROM t'), [{ m: 399999n }]);
      assert.deepEqual(await db.query("SELECT count(*) AS c FROM t WHERE s = 'x'"), [{ c: 0n }]);
    }
  } finally {
    db.close();
  }
});

test('codec delegation slices evicted pages out of the cached coalesced range', { skip }, async () => {
  const server = fakeServer({ 'https://h/gz.parquet': GZ });
  // A tiny per-table resident copy forces the cache path; the shared cache holds everything.
  const db = await openDb({
    fetch: server.fetchImpl,
    cache: new MemoryCache(1 << 30),
    cacheSize: 64 * 1024,
  });
  try {
    db.register('g', 'https://h/gz.parquet');
    const first = await db.query('SELECT sum(id) AS s, max(s) AS m FROM g');
    const gets = server.gets();
    assert.deepEqual(await db.query('SELECT sum(id) AS s, max(s) AS m FROM g'), first);
    assert.equal(server.gets(), gets, 'GZIP pages were fetched again despite the cache');
  } finally {
    db.close();
  }
});

// --- 3. names: identifiers fold ASCII case, paths do not ------------------------

test('URLs that differ only in case are different tables', { skip }, async () => {
  const server = fakeServer({ 'http://h/Data.parquet': A, 'http://h/data.parquet': B });
  const db = await openDb({ fetch: server.fetchImpl });
  try {
    const q = (u) => db.query(`SELECT min(id) AS m, count(*) AS c FROM parquet('${u}')`);
    assert.deepEqual(await q('http://h/Data.parquet'), [{ m: 0n, c: 3n }]);
    assert.deepEqual(await q('http://h/data.parquet'), [{ m: 100n, c: 5n }]);
    assert.deepEqual(await q('http://h/Data.parquet'), [{ m: 0n, c: 3n }]);
    assert.deepEqual(
      new Set(server.log.map((l) => l.url)),
      new Set(['http://h/Data.parquet', 'http://h/data.parquet']),
    );
  } finally {
    db.close();
  }
});

test('registered names are identifiers: ASCII case folds, other letters do not', { skip }, async () => {
  const db = await openDb();
  try {
    db.register('Ärger', A);
    db.register('ärger', B);
    assert.deepEqual(await db.query('SELECT min(id) AS m FROM "Ärger"'), [{ m: 0n }]);
    assert.deepEqual(await db.query('SELECT min(id) AS m FROM "ärger"'), [{ m: 100n }]);

    db.register('Trips', A);
    assert.deepEqual(await db.query('SELECT min(id) AS m FROM TRIPS'), [{ m: 0n }]);
    db.register('trips', B); // replaces 'Trips'
    assert.deepEqual(await db.query('SELECT min(id) AS m FROM Trips'), [{ m: 100n }]);
    const names = (await db.query('SHOW TABLES')).map((r) => r.name);
    assert.equal(names.filter((n) => n.toLowerCase() === 'trips').length, 1);
  } finally {
    db.close();
  }
});

test('unquoted identifiers with combining marks or emoji bind', { skip }, async () => {
  for (const name of ['डेटा', 'café', 'data🚀', 'ข้อมูล']) {
    const db = await openDb();
    try {
      db.register(name, A);
      assert.deepEqual(await db.query(`SELECT count(*) AS c FROM ${name}`), [{ c: 3n }], name);
    } finally {
      db.close();
    }
  }
});

// --- 4. sqlUrlPolicy sees every SQL path ---------------------------------------

test('protocol-relative and relative SQL paths go through sqlUrlPolicy', { skip }, async () => {
  const server = fakeServer({});
  const seen = [];
  const policy = (url, context) => {
    seen.push({ url, functionName: context.functionName });
    return false;
  };
  const db = await openDb({ fetch: server.fetchImpl, sqlUrlPolicy: policy });
  try {
    for (const sql of [
      "SELECT count(*) FROM parquet('//evil.example/x.parquet')",
      "SELECT count(*) FROM 'x.parquet'",
      "SELECT count(*) FROM read_csv('../secret.csv')",
    ]) {
      await assert.rejects(db.query(sql), (e) => e.code === Code.UNSUPPORTED_FEATURE);
    }
    assert.deepEqual(seen, [
      { url: '//evil.example/x.parquet', functionName: 'parquet' },
      { url: 'x.parquet', functionName: 'parquet' },
      { url: '../secret.csv', functionName: 'read_csv' },
    ]);

    // Where there is a base URL (a browser page), the policy sees where fetch would go.
    globalThis.location = { href: 'https://app.example/page/' };
    try {
      await assert.rejects(db.query("SELECT count(*) FROM parquet('//evil.example/y.parquet')"));
      assert.equal(seen.at(-1).url, 'https://evil.example/y.parquet');
    } finally {
      delete globalThis.location;
    }
    assert.equal(server.log.length, 0, 'a request was made before the policy check');
  } finally {
    db.close();
  }
  const denyAll = await openDb({ fetch: server.fetchImpl, sqlUrlPolicy: false });
  try {
    await assert.rejects(
      denyAll.query("SELECT count(*) FROM parquet('//evil.example/x.parquet')"),
      (e) => e.code === Code.UNSUPPORTED_FEATURE,
    );
    assert.equal(server.log.length, 0);
  } finally {
    denyAll.close();
  }
});

// --- 5/6. binding follows what the engine resolves -------------------------------

test('views read the current registration of their tables', { skip }, async () => {
  const db = await openDb();
  try {
    db.register('t', A);
    await db.query('CREATE VIEW v AS SELECT * FROM t');
    assert.deepEqual(ids(await db.query('SELECT * FROM v ORDER BY id')), [0, 1, 2]);
    db.register('t', B);
    assert.deepEqual(ids(await db.query('SELECT * FROM v ORDER BY id')), [100, 101, 102, 103, 104]);
  } finally {
    db.close();
  }
});

test('a view can use a table registered after it, reached only through the view', { skip }, async () => {
  const db = await openDb();
  try {
    await db.query('CREATE VIEW v AS SELECT * FROM t');
    db.register('t', A);
    assert.deepEqual(ids(await db.query('SELECT * FROM v ORDER BY id')), [0, 1, 2]);
  } finally {
    db.close();
  }
});

test('a table name used as a column, alias or string is never fetched', { skip }, async () => {
  const requests = [];
  const down = async (url, init = {}) => {
    requests.push(`${init.method ?? 'GET'} ${url}`);
    throw new TypeError('network down');
  };
  const db = await openDb({ fetch: down });
  try {
    db.register('id', 'https://down.example/id.parquet');
    db.register('t', A);
    assert.deepEqual(await db.query('SELECT sum(id) AS s FROM t'), [{ s: 3n }]);
    assert.deepEqual(await db.query("SELECT 'id' AS id FROM t AS id LIMIT 1"), [{ id: 'id' }]);
    assert.deepEqual(requests, []);
    // Reading it for real still reports the failure.
    await assert.rejects(db.query('SELECT * FROM id'), (e) => e.code === Code.IO_FAILED);
  } finally {
    db.close();
  }
});

test('SHOW TABLES lists every registration, queried or not', { skip }, async () => {
  const db = await openDb();
  try {
    db.register('alpha', A);
    db.register('beta', 'https://never.example/b.parquet');
    assert.deepEqual(await db.query('SHOW TABLES'), [{ name: 'alpha' }, { name: 'beta' }]);
  } finally {
    db.close();
  }
});

test('registration errors surface at register() time', { skip }, async () => {
  const db = await openDb();
  try {
    await db.query('CREATE TABLE m (id INTEGER)');
    assert.throws(() => db.register('m', A), (e) => e.code === Code.DUPLICATE_TABLE);
    // The failed registration left nothing behind that breaks unrelated queries.
    assert.deepEqual(await db.query('SELECT 1 AS x FROM range(1)'), [{ x: 1 }]);
  } finally {
    db.close();
  }
});

test('decodeMissingTables reads the START_NEED_TABLES list', () => {
  const path = new TextEncoder().encode('s3://b/Ä.csv');
  const buf = new Uint8Array(4 + 8 + path.length);
  const dv = new DataView(buf.buffer);
  dv.setUint32(0, 1, true);
  dv.setUint32(4, 2, true);
  dv.setUint32(8, path.length, true);
  buf.set(path, 12);
  assert.deepEqual(decodeMissingTables(buf), [{ format: 2, path: 's3://b/Ä.csv' }]);
  assert.throws(() => decodeMissingTables(buf.subarray(0, buf.length - 1)), AhiruError);
});

// --- 8. COPY / CTAS -------------------------------------------------------------

test('COPY ... TO hands its bytes to onCopy, and fails loudly without it', { skip }, async () => {
  const server = fakeServer({ 'https://h/a.parquet': A });
  const copies = [];
  const db = await openDb({
    fetch: server.fetchImpl,
    onCopy: async (path, bytes) => copies.push({ path, text: new TextDecoder().decode(bytes) }),
  });
  try {
    db.register('t', 'https://h/a.parquet');
    // The source is remote: the statement is restarted until its bytes are in.
    assert.deepEqual(await db.query("COPY (SELECT id FROM t ORDER BY id) TO 'out.csv'"), []);
    assert.deepEqual(copies, [{ path: 'out.csv', text: 'id\n0\n1\n2\n' }]);
  } finally {
    db.close();
  }

  const plain = await openDb();
  try {
    plain.register('t', A);
    await assert.rejects(plain.query("COPY (SELECT id FROM t) TO 'out.csv'"), (e) => {
      assert.equal(e.code, Code.UNSUPPORTED_FEATURE);
      assert.match(e.message, /onCopy/);
      return true;
    });
  } finally {
    plain.close();
  }
});

test('CREATE TABLE AS and INSERT ... SELECT read remote tables on the first try', { skip }, async () => {
  const server = fakeServer({ 'https://h/big.parquet': BIG });
  const db = await openDb({ fetch: server.fetchImpl });
  try {
    db.register('t', 'https://h/big.parquet');
    assert.deepEqual(await db.query('CREATE TABLE m AS SELECT id FROM t WHERE id < 250000'), [
      { count: 250000n },
    ]);
    assert.deepEqual(await db.query('INSERT INTO m SELECT id FROM t WHERE id >= 399999'), [
      { count: 1n },
    ]);
    assert.deepEqual(await db.query('SELECT count(*) AS c, max(id) AS m FROM m'), [
      { c: 250001n, m: 399999n },
    ]);
  } finally {
    db.close();
  }
});

// --- 9-11. smaller ABI surfaces -------------------------------------------------

test('a column named __proto__ is an own property of the row', { skip }, async () => {
  const db = await openDb();
  try {
    const [row] = await db.query('SELECT 1 AS "__proto__", 2 AS b FROM range(1)');
    assert.ok(Object.hasOwn(row, '__proto__'));
    assert.equal(row.__proto__, 1);
    assert.deepEqual(Object.keys(row), ['__proto__', 'b']);
    assert.equal(Object.getPrototypeOf(row), Object.prototype);

    const [iv] = await db.query(`SELECT INTERVAL '1 day' AS "__proto__" FROM range(1)`);
    assert.equal(Object.getPrototypeOf(iv), Object.prototype, 'the value became the prototype');
    assert.deepEqual(iv.__proto__, { months: 0, days: 1, micros: 0n });
  } finally {
    db.close();
  }
});

test('extra parameters are rejected like missing ones', { skip }, async () => {
  const db = await openDb();
  try {
    assert.deepEqual(await db.query('SELECT ? AS v FROM range(1)', [1]), [{ v: 1n }]);
    for (const params of [[1, 2], []]) {
      await assert.rejects(db.query('SELECT ? AS v FROM range(1)', params), (e) => {
        return e.code === Code.WRONG_ARG_COUNT;
      });
    }
  } finally {
    db.close();
  }
});

test('engine errors carry their position', { skip }, async () => {
  const db = await openDb();
  try {
    await assert.rejects(db.query('SELECT 1 FROM range(1) WHERE )'), (e) => {
      assert.equal(e.code, Code.UNEXPECTED_TOKEN);
      assert.equal(e.position, 'SELECT 1 FROM range(1) WHERE '.length);
      return true;
    });
    // An error without a position does not claim one.
    await assert.rejects(db.query('SELECT * FROM nowhere'), (e) => {
      assert.equal(e.code, Code.TABLE_NOT_FOUND);
      assert.equal(e.position, undefined);
      return true;
    });
  } finally {
    db.close();
  }
});

test('a TIMESTAMP compares with make_timestamp(?) or an ISO string cast', { skip }, async () => {
  const db = await openDb();
  try {
    const d = new Date('2024-06-01T12:34:56.789Z');
    const micros = BigInt(d.getTime()) * 1000n;
    const sql = (p) => `SELECT TIMESTAMP '2024-06-01 12:34:56.789' = ${p} AS eq FROM range(1)`;
    assert.deepEqual(await db.query(sql('make_timestamp(?)'), [micros]), [{ eq: true }]);
    assert.deepEqual(await db.query(sql('?::TIMESTAMP'), [d.toISOString()]), [{ eq: true }]);
    // A bare bigint is a BIGINT, which a TIMESTAMP does not compare with.
    await assert.rejects(db.query(sql('?'), [micros]), (e) => e.code === Code.TYPE_MISMATCH);
  } finally {
    db.close();
  }
});
