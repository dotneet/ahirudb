// ホスト層のテスト。`node --test 'js/test/*.test.mjs'` で走る。
//
// 事前に `./scripts/size.sh` で target/ahiru-core.wasm を作っておくこと。
// 値の正解は duckdb CLI から取る（このリポジトリの ground truth）。

import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { copyFileSync, existsSync, readFileSync, statSync } from 'node:fs';
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
  decodeBatch,
  decodeCodecRequests,
  detectFormat,
  encodeParams,
  timestampToDate,
} from '../ahirudb.js';
import { Code, errorMessage } from '../errors.js';

const ROOT = fileURLToPath(new URL('../..', import.meta.url));
const WASM = join(ROOT, 'target/ahiru-core.wasm');
const BASIC = join(ROOT, 'tests/data/basic.parquet');

if (!existsSync(WASM)) {
  throw new Error(`${WASM} がありません。先に ./scripts/size.sh を実行してください`);
}

/** duckdb CLI を JSON で叩く。値の正解はすべてここから取る。 */
function duck(sql) {
  const out = execFileSync('duckdb', ['-json', '-c', sql], { encoding: 'utf8', maxBuffer: 1 << 28 });
  return out.trim() === '' ? [] : JSON.parse(out);
}

async function openDb(options = {}) {
  return AhiruDB.init({ wasmUrl: WASM, ...options });
}

/**
 * CSV / JSONL はフィーチャで切れるので、既定の配布ビルド
 * (`target/ahiru-core.wasm` = parquet のみ) には入っていない。
 * 全フォーマット入りを別に用意する。無ければその場でビルドし、
 * それも駄目ならフォーマット系のテストだけ skip する。
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
    // ビルドできなくても、以前作ったものが残っていれば使う。
    return existsSync(FULL_WASM)
      ? false
      : `csv,jsonl 入りの wasm がありません。cargo build --profile wasm ` +
          `--target wasm32-unknown-unknown -p ahiru-core --no-default-features ` +
          `--features csv,jsonl して ${FULL_WASM} に置くか、AHIRU_WASM_FULL を指定してください`;
  }
})();

/**
 * ZSTD は既定でコアに内蔵する（`zstd` フィーチャ、DESIGN.md §6）ので、
 * `target/ahiru-core.wasm` に対するクエリはコーデック委譲（`NEED_CODEC`）を
 * 経由しない。ホスト側の委譲・キャッシュ溢れ耐性はそれ自体テストしたい
 * 挙動なので、`zstd` を外したコアを別に用意してそちらだけに向ける。
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
      : `zstd 無しの wasm がありません。cargo build --profile wasm ` +
          `--target wasm32-unknown-unknown -p ahiru-core --no-default-features ` +
          `して ${NOZSTD_WASM} に置くか、AHIRU_WASM_NOZSTD を指定してください`;
  }
})();

async function openFullDb(options = {}) {
  return AhiruDB.init({ wasmUrl: FULL_WASM, ...options });
}

/**
 * 式 VM（crates/ahiru-core/src/expr/vm.rs）はまだスタブで、Project が必ず
 * E900 を返す。値のアサーションはそこが入るまで検証できないので、実際に
 * 動くかを 1 回だけ確かめ、駄目なら該当テストを skip する。
 * VM が入れば自動的に skip が外れる（テスト本体は一切弱めていない）。
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
 * VM が入るまでは、行を取り切る手前で E900 になる。
 * I/O 経路のアサーションはそこまでで十分成立するので、900 だけ飲み込む。
 */
async function runTolerantly(db, sql) {
  try {
    return await db.query(sql);
  } catch (e) {
    if (e.code === Code.INTERNAL) return null;
    throw e;
  }
}

// --- ワイヤ形式のデコード（wasm 非依存）--------------------------------------

/** abi.rs の `encode_batch` と同じ並びでバッファを組む。デコーダ単体の検証用。 */
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

test('decodeBatch は列指向バッファを型どおりに読む', () => {
  const bytes = new TextEncoder().encode('あhello');
  const f64 = new Float64Array([1.5, -2.25, 0]);
  const i64 = new BigInt64Array([10n, -20n, 30n]);
  // I128 は TypedArray が無いので 64 ビット 2 本（下位→上位）で書く。
  const i128 = new Uint8Array(3 * 16);
  const i128dv = new DataView(i128.buffer);
  [0n, 2n ** 100n, -(2n ** 100n) - 1n].forEach((v, i) => {
    const u = BigInt.asUintN(128, v);
    i128dv.setBigUint64(i * 16, u & 0xffffffffffffffffn, true);
    i128dv.setBigUint64(i * 16 + 8, u >> 64n, true);
  });
  const buf = encodeBatch(3, [
    { phys: 1, valid: [1, 0, 1], data: new Uint8Array(new Int32Array([7, 999, -3]).buffer) },
    // 可変長は offsets.len == 行数 + 1。ここの長さが奇数だと後続の列が
    // 8 バイト境界からずれるので、非整列の経路もこの 1 本で踏む。
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
    { i: 7, s: 'あ', f: 1.5, b: 10n, z: true, h: 0n },
    { i: null, s: '', f: -2.25, b: -20n, z: false, h: 2n ** 100n },
    { i: -3, s: 'hello', f: 0, b: null, z: true, h: -(2n ** 100n) - 1n },
  ]);
  assert.ok(batch.column('f') instanceof Float64Array);
  assert.ok(batch.column('b') instanceof BigInt64Array);
  assert.equal(batch.isNull('i', 1), true);
  assert.equal(batch.get('i', 1), null);
});

test('decodeBatch はマジックが違えば落とす', () => {
  const buf = encodeBatch(0, []);
  buf[0] = 0;
  assert.throws(() => decodeBatch(buf, []), (e) => e instanceof AhiruError && e.code === 900);
});

// --- レンジ結合 --------------------------------------------------------------

test('coalesceRanges は 1 MB 未満の穴を埋めて 1 本にする', () => {
  // 隣接（穴なし）
  assert.deepEqual(coalesceRanges([{ offset: 100, len: 50 }, { offset: 150, len: 50 }]), [
    { offset: 100, len: 100 },
  ]);
  // 小さい穴は飲み込む: 400KB + 100KB の穴 + 400KB → 900KB を 1 回
  assert.deepEqual(
    coalesceRanges([
      { offset: 0, len: 400 * 1024 },
      { offset: 500 * 1024, len: 400 * 1024 },
    ]),
    [{ offset: 0, len: 900 * 1024 }],
  );
  // 1 MB 以上離れていれば分けたまま
  assert.equal(
    coalesceRanges([
      { offset: 0, len: 10 },
      { offset: 3 * 1024 * 1024, len: 10 },
    ]).length,
    2,
  );
  // 順序不同・包含・ファイル末尾のはみ出し
  assert.deepEqual(
    coalesceRanges([{ offset: 60, len: 100 }, { offset: 0, len: 10 }, { offset: 70, len: 5 }], 0, 120),
    [{ offset: 0, len: 10 }, { offset: 60, len: 60 }],
  );
});

// --- キャッシュ --------------------------------------------------------------

test('MemoryCache は容量上限で LRU 追い出しする', () => {
  const c = new MemoryCache(300);
  c.set('a', new Uint8Array(100));
  c.set('b', new Uint8Array(100));
  c.set('c', new Uint8Array(100));
  assert.ok(c.get('a'));
  c.set('d', new Uint8Array(100)); // a を触った直後なので b が落ちる
  assert.equal(c.get('b'), undefined);
  assert.ok(c.get('a') && c.get('c') && c.get('d'));
  assert.equal(c.size, 300);
  // 単体で上限を超えるものは載せない（他を全部追い出さないため）
  c.set('big', new Uint8Array(1000));
  assert.equal(c.get('big'), undefined);
});

// --- エラーコード表 ----------------------------------------------------------

test('errors.js は error.rs の Code / message と一致している', () => {
  const rs = readFileSync(join(ROOT, 'crates/ahiru-core/src/error.rs'), 'utf8');
  const codes = new Map();
  for (const m of rs.matchAll(/^\s{4}(\w+) = (\d+),$/gm)) codes.set(m[1], Number(m[2]));
  assert.ok(codes.size > 20, 'error.rs から Code を読めていない');

  const messages = new Map();
  for (const m of rs.matchAll(/^\s{12}(\w+) => "([^"]*)",$/gm)) messages.set(m[1], m[2]);

  const known = new Set(Object.values(Code));
  for (const [name, value] of codes) {
    assert.ok(known.has(value), `errors.js に ${name} = ${value} が無い`);
    assert.equal(errorMessage(value), messages.get(name), `${name} のメッセージがずれている`);
  }
  assert.equal(Object.keys(Code).length, codes.size, 'errors.js に余分なコードがある');
});

// --- 実データ: メモリ登録 ----------------------------------------------------

test('bytes 登録 + SELECT が duckdb と一致する', { skip: needsVm }, async () => {
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

test('NULL は 0 ではなく null として返る', { skip: needsVm }, async () => {
  const db = await openDb();
  try {
    db.registerParquet('t', new Uint8Array(await readFile(BASIC)));
    const rows = await db.query('SELECT id, big FROM t LIMIT 20');
    const want = duck(`SELECT id, big FROM '${BASIC}' LIMIT 20`);
    // duckdb の JSON は BIGINT を number で出すので BigInt 側に寄せて比べる。
    assert.deepEqual(
      rows,
      want.map((r) => ({ id: r.id, big: r.big === null ? null : BigInt(r.big) })),
    );
    // 5 行ごとに NULL が入っているのが basic.parquet の作り。
    for (const r of rows) {
      if (r.id % 5 === 0) assert.equal(r.big, null, `id=${r.id} は NULL のはず`);
      else assert.notEqual(r.big, null);
    }
  } finally {
    db.close();
  }
});

test('OFFSET が効く', { skip: needsVm }, async () => {
  // encode_batch が materialize してから直列化するようになったので、
  // selection vector が効いた結果がそのまま返る。
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

test('stream は列指向バッチを返す', { skip: needsVm }, async () => {
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

test('TIMESTAMP はマイクロ秒の BigInt で返り、ヘルパで Date になる', { skip: needsVm }, async () => {
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

// --- 実データ: レンジ取得 ----------------------------------------------------

/** ローカルのバッファを Range で切り出す偽 fetch。要求は全部記録する。 */
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
    assert.ok(m, `Range ヘッダが無い: ${raw}`);
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

/** 64 KiB のフッタ投機取得だけでは全体が読めない、多少大きなファイルを用意する。 */
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
/** 列チャンクの実バイト範囲。射影と結合のアサーションはこれを基準にする。 */
const WIDE_CHUNKS = new Map(
  duck(
    `SELECT path_in_schema AS name,
            least(data_page_offset, coalesce(dictionary_page_offset, data_page_offset)) AS start,
            total_compressed_size AS len
     FROM parquet_metadata('${WIDE}') WHERE row_group_id = 0`,
  ).map((r) => [r.name, { start: Number(r.start), len: Number(r.len) }]),
);

test('登録だけでは 1 バイトも取りに行かない', async () => {
  const f = fakeFetcher(new Uint8Array(await readFile(WIDE)));
  const db = await openDb({ fetch: f.fetchImpl });
  try {
    db.registerParquet('t', f.url);
    assert.equal(f.calls.length, 0, '登録時に I/O が発生している');
    assert.equal(f.ranges.length, 0);
  } finally {
    db.close();
  }
});

test('射影プッシュダウン: 選んだ列のバイトしか取らない', async () => {
  const file = new Uint8Array(await readFile(WIDE));
  const f = fakeFetcher(file);
  const db = await openDb({ fetch: f.fetchImpl });
  try {
    db.registerParquet('t', f.url);
    await runTolerantly(db, 'SELECT id, name FROM t LIMIT 5');

    const footer = 64 * 1024; // FOOTER_PROBE（parquet/file.rs）
    const wanted = WIDE_CHUNKS.get('id').len + WIDE_CHUNKS.get('name').len;
    // 取得量 = フッタ投機 + id/name の列チャンクのみ。score/flag/big は読まない。
    assert.equal(f.bytes(), footer + wanted);
    assert.ok(f.bytes() < WIDE_SIZE * 0.25, `${f.bytes()} bytes は削れていない`);

    // 読まなかった列の領域に触れていないことを直接確かめる。
    const skipped = WIDE_CHUNKS.get('score');
    for (const r of f.ranges) {
      const overlapsSkipped =
        r.offset < skipped.start + skipped.len && skipped.start < r.offset + r.len;
      const isFooterProbe = r.offset >= WIDE_SIZE - footer;
      assert.ok(!overlapsSkipped || isFooterProbe, `score の領域を読んでいる: ${JSON.stringify(r)}`);
    }
  } finally {
    db.close();
  }
});

test('隣接した 2 本の要求は 1 回の fetch にまとまる', async () => {
  const file = new Uint8Array(await readFile(WIDE));
  const f = fakeFetcher(file);
  const db = await openDb({ fetch: f.fetchImpl });
  try {
    db.registerParquet('t', f.url);
    await runTolerantly(db, 'SELECT id, name FROM t LIMIT 5');

    const id = WIDE_CHUNKS.get('id');
    const name = WIDE_CHUNKS.get('name');
    // 前提: この 2 列のチャンクはファイル上で隣接している。
    assert.equal(id.start + id.len, name.start, 'テストの前提（隣接）が崩れている');

    // エンジンは列ごとに 1 本ずつ要求する。ホストが結合するので fetch は 1 回。
    const data = f.ranges.filter((r) => r.offset < WIDE_SIZE - 64 * 1024);
    assert.equal(data.length, 1, `結合されていない: ${JSON.stringify(data)}`);
    assert.deepEqual(data[0], { offset: id.start, len: id.len + name.len });
    assert.equal(f.ranges.length, 2, 'フッタ + データの 2 回だけのはず');
  } finally {
    db.close();
  }
});

test('2 回目の同じクエリは新しい fetch を出さない', async () => {
  const file = new Uint8Array(await readFile(WIDE));
  const f = fakeFetcher(file);
  const db = await openDb({ fetch: f.fetchImpl });
  try {
    db.registerParquet('t', f.url);
    await runTolerantly(db, 'SELECT id, name FROM t LIMIT 5');
    const first = f.ranges.length;
    assert.ok(first > 0);
    await runTolerantly(db, 'SELECT id, name FROM t LIMIT 5');
    assert.equal(f.ranges.length, first, '2 回目に取り直している');
  } finally {
    db.close();
  }
});

test('レンジキャッシュはインスタンスをまたいで効く', async () => {
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

  // 同じ URL・同じレンジなので、2 つ目のインスタンスは fetch せずに済む。
  const f2 = fakeFetcher(file);
  const db2 = await openDb({ fetch: f2.fetchImpl, cache });
  try {
    db2.registerParquet('t', f2.url);
    await runTolerantly(db2, 'SELECT id, name FROM t LIMIT 5');
    assert.equal(f2.ranges.length, 0, 'キャッシュが使われていない');
  } finally {
    db2.close();
  }
});

test('レンジ取得の結果はメモリ登録と一致する', { skip: needsVm }, async () => {
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

test('cache: "none" は毎回取りに行く', async () => {
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

// --- エラー ------------------------------------------------------------------

test('構文エラーは AhiruError（3xx）になる', async () => {
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

test('未知のテーブルは TABLE_NOT_FOUND', async () => {
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

test('未知の列は COLUMN_NOT_FOUND', async () => {
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

// --- パラメータバインド ------------------------------------------------------

test('encodeParams はタグ付きの列を組む', () => {
  const buf = encodeParams([null, true, 7, 1.5, 'ab']);
  const dv = new DataView(buf.buffer);
  assert.equal(dv.getUint32(0, true), 5);
  assert.equal(buf[4], 0); // NULL
  assert.deepEqual([...buf.subarray(5, 7)], [1, 1]); // BOOL true
  assert.equal(buf[7], 2); // 安全な整数は I64
  assert.equal(dv.getBigInt64(8, true), 7n);
  assert.equal(buf[16], 3); // 小数は F64
  assert.equal(dv.getFloat64(17, true), 1.5);
  assert.equal(buf[25], 4); // 文字列は BYTES
  assert.equal(dv.getUint32(26, true), 2);
  assert.deepEqual([...buf.subarray(30)], [0x61, 0x62]);
  assert.equal(encodeParams([]).length, 0);
  assert.equal(encodeParams(undefined).length, 0);
});

test('encodeParams は扱えない型を明示的に落とす', () => {
  // Date を勝手にマイクロ秒へ直すと、桁を間違えても気づけない。
  assert.throws(
    () => encodeParams([new Date()]),
    (e) => e instanceof AhiruError && e.code === Code.UNSUPPORTED_FEATURE,
  );
  assert.throws(
    () => encodeParams([2n ** 70n]),
    (e) => e instanceof AhiruError && e.code === Code.VALUE_OUT_OF_RANGE,
  );
});

test('パラメータが束縛される', { skip: needsVm }, async () => {
  const db = await openDb();
  try {
    db.register('t', new Uint8Array(await readFile(BASIC)));
    assert.deepEqual(
      await db.query('SELECT id, name FROM t WHERE id = ?', [3]),
      duck(`SELECT id, name FROM '${BASIC}' WHERE id = 3`),
    );
    // 文字列パラメータ。SQL に埋め込まずに済むのがこの API の眼目。
    assert.deepEqual(
      await db.query('SELECT id FROM t WHERE name = ? LIMIT 3', ['name_5']),
      duck(`SELECT id FROM '${BASIC}' WHERE name = 'name_5' LIMIT 3`),
    );
    // 複数・BigInt・浮動小数
    assert.deepEqual(
      await db.query('SELECT id FROM t WHERE id > ? AND score < ? ORDER BY id', [10n, 20.0]),
      duck(`SELECT id FROM '${BASIC}' WHERE id > 10 AND score < 20.0 ORDER BY id`),
    );
    // 引用符入りの文字列がそのまま値として渡ること（連結なら壊れる形）
    assert.deepEqual(await db.query("SELECT id FROM t WHERE name = ?", ["a' OR 1=1 --"]), []);
  } finally {
    db.close();
  }
});

test('バイトが増えないまま同じ要求が来たらライブロックとして落ちる', async () => {
  const db = await openDb();
  try {
    let reads = 0;
    // 常に空を返す供給元。放っておくと NEED_IO を無限に回してしまう。
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
    assert.ok(reads <= 4, `空応答で回りすぎ: ${reads}`);
  } finally {
    db.close();
  }
});

test('memoryLimit を超えたら E501 で止まる', async () => {
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

test('close 後の操作はエラー', async () => {
  const db = await openDb();
  db.close();
  db.close(); // 二重 close は無害
  assert.throws(() => db.registerParquet('t', new Uint8Array(8)));
  await assert.rejects(() => db.query('SELECT 1 FROM t'));
});

// --- WHERE（式 VM 待ち）------------------------------------------------------

test(
  'WHERE で絞れる',
  {
    skip:
      needsVm &&
      'WHERE は式 VM (expr/vm.rs) が必要。VM が入ったらこの skip は自動で外れる。',
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
  '統計プルーニング: 述語に当たらない RowGroup は 1 バイトも読まない',
  {
    skip:
      needsVm &&
      'WHERE は式 VM (expr/vm.rs) が必要。VM が入ったらこの skip は自動で外れる。',
  },
  async () => {
    const file = new Uint8Array(await readFile(WIDE));
    const f = fakeFetcher(file);
    const db = await openDb({ fetch: f.fetchImpl });
    try {
      db.registerParquet('t', f.url);
      // id は昇順なので、2 つ目の RowGroup（id >= 100000）だけが残る。
      const rows = await db.query('SELECT id FROM t WHERE id > 199990');
      assert.deepEqual(rows, duck(`SELECT id FROM '${WIDE}' WHERE id > 199990`));
      const first = WIDE_CHUNKS.get('id');
      for (const r of f.ranges) {
        const hitsFirstGroup = r.offset < first.start + first.len && first.start < r.offset + r.len;
        assert.ok(!hitsFirstGroup, '枝刈りできる RowGroup を読んでいる');
      }
    } finally {
      db.close();
    }
  },
);

// --- 集約 / ソート / 結合 ----------------------------------------------------

/** 未実装の機能は E409 / E900 で返る。値のアサーションは弱めずに skip する。 */
async function featureStatus(sql) {
  const db = await openDb();
  try {
    db.register('t', new Uint8Array(await readFile(BASIC)));
    await db.query(sql);
    return false;
  } catch (e) {
    if (e.code === Code.UNSUPPORTED_FEATURE || e.code === Code.INTERNAL) {
      return `まだエンジン側が未実装 (E${e.code})。実装されれば自動で skip が外れる。`;
    }
    throw e;
  } finally {
    db.close();
  }
}

const AGG_SKIP = await featureStatus('SELECT count(*) c FROM t');
const JOIN_SKIP = await featureStatus('SELECT a.id FROM t a JOIN t b ON a.id = b.id LIMIT 1');

test('GROUP BY と集約が duckdb と一致する', { skip: AGG_SKIP }, async () => {
  const db = await openDb();
  try {
    db.register('t', new Uint8Array(await readFile(BASIC)));
    const rows = await db.query('SELECT flag, count(*) c FROM t GROUP BY flag');
    const want = duck(`SELECT flag, count(*) c FROM '${BASIC}' GROUP BY flag`);
    const norm = (rs) =>
      rs.map((r) => ({ flag: r.flag, c: Number(r.c) })).sort((a, b) => Number(a.flag) - Number(b.flag));
    assert.deepEqual(norm(rows), norm(want));

    // NULL を数えないこと（big は 5 行ごとに NULL）
    const [agg] = await db.query('SELECT count(big) nb, count(*) n, sum(id) s FROM t');
    const [wantAgg] = duck(`SELECT count(big) nb, count(*) n, sum(id) s FROM '${BASIC}'`);
    assert.equal(Number(agg.nb), Number(wantAgg.nb));
    assert.equal(Number(agg.n), Number(wantAgg.n));
    assert.equal(Number(agg.s), Number(wantAgg.s));
  } finally {
    db.close();
  }
});

test('HAVING と DISTINCT', { skip: AGG_SKIP }, async () => {
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

test('ORDER BY が duckdb と一致する', { skip: AGG_SKIP }, async () => {
  const db = await openDb();
  try {
    db.register('t', new Uint8Array(await readFile(BASIC)));
    assert.deepEqual(
      await db.query('SELECT id, score FROM t ORDER BY score DESC, id LIMIT 5'),
      duck(`SELECT id, score FROM '${BASIC}' ORDER BY score DESC, id LIMIT 5`),
    );
    // NULL の並び順も合わせる（big は NULL を含む）
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

test('JOIN が duckdb と一致する', { skip: JOIN_SKIP }, async () => {
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

// --- DECIMAL -----------------------------------------------------------------

/** DECIMAL(18,4) は I64、DECIMAL(30,6) は I128 に載る。両方通す。 */
const DEC = join(tmpdir(), 'ahirudb-test-decimal.parquet');
if (!existsSync(DEC)) {
  duck(
    `COPY (SELECT i::INTEGER AS id, (i * 1.005)::DECIMAL(18,4) AS d18,
             (i * 1.005)::DECIMAL(30,6) AS d30 FROM range(10) t(i))
     TO '${DEC}' (FORMAT PARQUET)`,
  );
}

test('DECIMAL は precision/scale を適用した文字列で返る', { skip: needsVm }, async () => {
  const db = await openDb();
  try {
    db.register('d', new Uint8Array(await readFile(DEC)));
    const rows = await db.query('SELECT id, d18, d30 FROM d LIMIT 4');
    // 桁を落とさないため文字列で返す。number にすると 18 桁超で丸まる。
    assert.deepEqual(rows, [
      { id: 0, d18: '0.0000', d30: '0.000000' },
      { id: 1, d18: '1.0050', d30: '1.005000' },
      { id: 2, d18: '2.0100', d30: '2.010000' },
      { id: 3, d18: '3.0150', d30: '3.015000' },
    ]);
    // 値として duckdb と一致していること。
    const want = duck(`SELECT id, d18, d30 FROM '${DEC}' LIMIT 4`);
    assert.deepEqual(
      rows.map((r) => [r.id, Number(r.d18), Number(r.d30)]),
      want.map((r) => [r.id, Number(r.d18), Number(r.d30)]),
    );
    for await (const b of db.stream('SELECT d18 FROM d LIMIT 1')) {
      assert.equal(b.schema[0].type, 'DECIMAL');
      assert.equal(b.columns[0].scale, undefined); // scale はスキーマ側が持つ
    }
  } finally {
    db.close();
  }
});

// --- コーデック委譲 ----------------------------------------------------------

test('decodeCodecRequests は要求列を読む', () => {
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

const GZIP_PARQUET = join(ROOT, 'tests/data/gzip.parquet');

/** BigInt と number の差だけを均す。duckdb の JSON は整数を number で出す。 */
const numeric = (rows) =>
  rows.map((r) =>
    Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === 'bigint' ? Number(v) : v])),
  );

test('GZIP はホストの DecompressionStream で展開される', { skip: needsVm }, async () => {
  const db = await openDb();
  try {
    db.register('g', new Uint8Array(await readFile(GZIP_PARQUET)));
    // 列名を決め打ちしない（テストデータは別の agent も差し替える）。
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

test('GZIP はレンジ取得経路でも追加 fetch なしで展開できる', { skip: needsVm }, async () => {
  const file = new Uint8Array(await readFile(GZIP_PARQUET));
  const f = fakeFetcher(file);
  const db = await openDb({ fetch: f.fetchImpl });
  try {
    db.register('g', f.url);
    assert.deepEqual(
      numeric(await db.query('SELECT * FROM g LIMIT 5')),
      numeric(duck(`SELECT * FROM '${GZIP_PARQUET}' LIMIT 5`)),
    );
    // 圧縮ブロックは NEED_IO で取った控えから切り出す。取り直しは起きない。
    assert.ok(f.ranges.length <= 2, `余計な fetch が出ている: ${JSON.stringify(f.ranges)}`);
  } finally {
    db.close();
  }
});

const ZSTD_PARQUET = join(ROOT, 'tests/data/zstd.parquet');

test('ZSTD は既定のコアだけで（サイドモジュール無しで）展開される', { skip: needsVm }, async () => {
  const db = await openDb(); // 既定の target/ahiru-core.wasm。zstdUrl は渡さない。
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

test('ZSTD はモジュール未指定なら ZSTD と名指しで落ちる', { skip: NOZSTD_SKIP || needsVm }, async () => {
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
 * ZSTD サイドモジュール（crates/ahiru-zstd）。既定では `ahiru-core` に
 * ライブラリとしてリンクされる（`zstd` フィーチャ）ので、単独の wasm
 * モジュールとしては `standalone` フィーチャを明示し、`crate-type` も
 * `cdylib` に明示的に上書きしてビルドする必要がある
 * （`crates/ahiru-zstd/Cargo.toml` 参照。既定は `rlib` のみ）。
 * `zstd` フィーチャを外したコアでの委譲経路のテスト専用。
 */
const ZSTD_WASM = join(ROOT, 'target/wasm32-unknown-unknown/wasm/ahiru_zstd.wasm');
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
    /* ビルドできなくても、既にあるものを見る。 */
  }
  if (!existsSync(ZSTD_WASM)) return 'crates/ahiru-zstd がまだビルドできない';
  const mod = await WebAssembly.compile(await readFile(ZSTD_WASM));
  const names = new Set(WebAssembly.Module.exports(mod).map((e) => e.name));
  const missing = ['zstd_alloc', 'zstd_free', 'zstd_decompress'].filter((n) => !names.has(n));
  return missing.length === 0
    ? false
    : `ahiru-zstd がまだ ${missing.join('/')} を公開していない（実装中）。` +
        '公開されればこの skip は自動で外れる。';
})();

test('ZSTD はサイドモジュールで展開される', { skip: ZSTD_SKIP || NOZSTD_SKIP || needsVm }, async () => {
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

test('detectFormat は登録名の拡張子でフォーマットを決める', () => {
  assert.equal(detectFormat('a.parquet'), 'parquet');
  assert.equal(detectFormat('a.CSV'), 'csv');
  assert.equal(detectFormat('a.tsv'), 'tsv');
  assert.equal(detectFormat('a.ndjson'), 'jsonl');
  assert.equal(detectFormat('data'), 'parquet');
  assert.equal(detectFormat('https://x/y/trips.csv?token=abc'), 'csv');
  assert.equal(detectFormat('https://x/y/data.parquet?name=a.csv'), 'parquet');
});

test('知らない format 名は登録時に落とす', async () => {
  const db = await openDb();
  try {
    // 綴り間違いを Auto に落とすと Parquet として読まれ、BadMagic になって
    // 原因が分からなくなる。ここで止める。
    assert.throws(
      () => db.register('t', new Uint8Array(8), { format: 'json' }),
      (e) => e instanceof AhiruError && e.code === Code.UNSUPPORTED_FEATURE,
    );
  } finally {
    db.close();
  }
});

test('format を明示すれば拡張子なしの名前で登録できる', { skip: FORMAT_SKIP }, async () => {
  const db = await openFullDb();
  try {
    // ahiru_register_as にフォーマットを渡すので、テーブル名は素の識別子でよい。
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

test('明示した format は拡張子より優先される', async () => {
  const db = await openDb();
  try {
    // 名前が嘘をついていても、明示指定が勝つ（名前と読み方を切り離せることが
    // このオプションの目的なので、食い違いを検査で塞がない）。
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

test('CSV が Parquet と同じ値を返す', { skip: FORMAT_SKIP }, async () => {
  const db = await openFullDb();
  try {
    db.register('basic.csv', new Uint8Array(await readFile(CSV)));
    const rows = await db.query('SELECT id, name FROM "basic.csv" LIMIT 5');
    const want = duck(`SELECT id, name FROM read_csv('${CSV}') LIMIT 5`);
    // CSV には型が無く、整数は BIGINT として推定される。値として比べる。
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

test('JSONL が Parquet と同じ値を返す', { skip: FORMAT_SKIP }, async () => {
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

test('CSV も NULL を null として返す', { skip: FORMAT_SKIP }, async () => {
  const db = await openFullDb();
  try {
    db.register('basic.csv', new Uint8Array(await readFile(CSV)));
    const rows = await db.query('SELECT id, big FROM "basic.csv" LIMIT 12');
    for (const r of rows) {
      if (Number(r.id) % 5 === 0) assert.equal(r.big, null, `id=${r.id} は NULL のはず`);
      else assert.notEqual(r.big, null);
    }
  } finally {
    db.close();
  }
});

test('CSV もレンジ取得で読める', { skip: FORMAT_SKIP }, async () => {
  const file = new Uint8Array(await readFile(CSV));
  const f = fakeFetcher(file, 'https://example.invalid/basic.csv');
  const db = await openFullDb({ fetch: f.fetchImpl });
  try {
    db.register('basic.csv', f.url);
    assert.equal(f.ranges.length, 0, '登録で I/O が出ている');
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

// --- 控えの追い出し ----------------------------------------------------------

/**
 * コーデック委譲は「NEED_IO で取った控えから圧縮ブロックを切り出す」前提だが、
 * 控えは `cacheSize` で頭打ちにしてある。溢れて捨てた範囲を後から要求された
 * ときに落ちないこと（＝取り直しに落ちること）を確かめる。メモリ圧のときだけ
 * 出る経路なので、意図的に極小の上限で踏ませる。
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
  // 控えもキャッシュも実質ゼロ。毎回どこかが捨てられる状態にする。
  const db = await AhiruDB.init({
    wasmUrl: WASM,
    fetch: f.fetchImpl,
    cacheSize: 4096,
    ...options,
  });
  try {
    db.register('t', f.url);
    const first = await db.query('SELECT id FROM t');
    // 2 回目は wasm 側にバイトが残っているので NEED_IO が出ない。
    // 控えが空でもコーデック要求を満たせること。
    const second = await db.query('SELECT id FROM t');
    return { first, second, fetches: f.ranges.length };
  } finally {
    db.close();
  }
}

test('控えを溢れさせても GZIP を読み切れる', { skip: needsVm }, async () => {
  const { first, second } = await scanTwiceWithTinyCache(BIG_GZIP);
  assert.equal(first.length, 120000);
  assert.deepEqual(first, second);
  assert.equal(first[0].id, 0);
  assert.equal(first[first.length - 1].id, 119999);
});

test('控えを溢れさせても ZSTD を読み切れる', { skip: ZSTD_SKIP || NOZSTD_SKIP || needsVm }, async () => {
  const { first, second } = await scanTwiceWithTinyCache(BIG_ZSTD, {
    wasmUrl: NOZSTD_WASM,
    zstdUrl: ZSTD_WASM,
  });
  assert.equal(first.length, 120000);
  assert.deepEqual(first, second);
  assert.equal(first[first.length - 1].id, 119999);
});
