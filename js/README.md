# ahirudb — JS host layer

An ES module that drives `ahiru-core.wasm`. **Zero dependencies, no build step** —
it runs as-is in both the browser and Node 18+.

```
js/
  ahirudb.js     Core (IO loop / result decoding / caching)
  errors.js      Error code table (kept 1:1 with crates/ahiru-core/src/error.rs)
  ahirudb.d.ts   Type definitions
  test/          Tests for node --test
```

## Usage

```js
import { AhiruDB, timestampToDate } from './js/ahirudb.js';

const db = await AhiruDB.init({
  wasmUrl: '/ahiru-core.wasm',
  memoryLimit: 512 * 1024 * 1024, // wasm heap cap; exceeding it raises E501
  cache: 'memory',                // "memory" | "cache-api" | "none" | a custom implementation
  zstdUrl: '/ahiru-zstd.wasm',    // only needed if ahiru-core.wasm was built without the `zstd` feature
});

// register() does no I/O. Fetching the total length and reading the footer
// are both deferred until the first query.
db.register('trips', 'https://example.com/trips.parquet'); // HTTP Range
db.register('local', bytes);                               // Uint8Array / ArrayBuffer
db.register('picked', fileFromInputElement);               // Blob / File
db.register('logs.csv', csvBytes);                         // CSV / TSV / JSONL

// 1) Fetch all rows
const rows = await db.query('SELECT id, name FROM trips LIMIT 5');
// -> [{ id: 0, name: 'name_0' }, ...]

// 2) Bind parameters (never interpolate values into SQL)
await db.query('SELECT * FROM trips WHERE vendor = ? AND fare > ?', ['VTS', 100]);

// 3) Streaming (columnar batches, 2048 rows by default)
for await (const batch of db.stream('SELECT id, score FROM trips')) {
  batch.numRows;            // row count
  batch.column('score');    // Float64Array
  batch.isNull('score', 3); // inspect the validity bitmap
  batch.toRows();           // array of plain objects
}

db.close();
```

`registerParquet` is an alias for `register` (it can register formats other than Parquet too).

Writing `FROM parquet('https://…/a.parquet')` automatically registers that path
as a table of the same name (this is a contract of `resolve_from` in
`plan/bind.rs`). That's the entry point for Parquet specifically; CSV / JSONL
must be registered first via `register(name, src, { format })` and then
referenced by their plain identifier.

### Format

Passing `format` uses it directly (`ahiru_register_as`). If omitted, the engine
infers it **from the extension of the registered name** (`format::FormatKind::detect`).

```js
db.register('logs', bytes, { format: 'csv' }); // the name can be a plain identifier
await db.query('SELECT * FROM logs');

db.register('logs.csv', bytes);                // let the extension decide
await db.query('SELECT * FROM "logs.csv"');    // reference it quoted on the SQL side
```

`format` accepts `parquet` / `csv` / `tsv` / `jsonl`. An explicit value takes
priority even if it disagrees with the extension (decoupling the name from how
it's read is the whole point of this option, so we don't validate that away).
Only a misspelled value fails, with E409 — falling back to Auto would read it
as Parquet and fail with an opaque `BadMagic` instead.

Extension-based detection recognizes `.csv` / `.tsv` / `.tab` / `.jsonl` /
`.ndjson` / `.json`; anything else is treated as Parquet. `.json` means a
single top-level JSON document (array of objects, or one object) — the
`read_json`/`read_json_auto` shape, distinct from `.jsonl`'s one-object-per-line
— and can currently only be reached through extension detection: there is no
`format: 'json'` option, since `ahiru_register_as`'s explicit-format
parameter doesn't have a wire value for it yet. CSV and JSONL are gated
behind wasm-side features (`--features csv,jsonl`), so they aren't present in
the default distribution build. Registering them against a build that lacks
the feature raises E409.

### Parameters

Accepted types: `null` / `boolean` / `number` / `bigint` / `string` / `Uint8Array`.
Safe integers and `bigint` are sent as I64; other `number` values as F64.
`Date` is not accepted (implicitly converting to microseconds would hide
off-by-a-magnitude bugs). To compare against a TIMESTAMP, pass
`BigInt(d.getTime()) * 1000n`.

## Value mapping

| Logical type | JS |
|---|---|
| BOOLEAN | `boolean` |
| TINYINT / SMALLINT / INTEGER / DATE | `number` |
| BIGINT / TIME / TIMESTAMP | `bigint` (TIMESTAMP is microseconds since epoch) |
| HUGEINT / UBIGINT | `bigint` |
| FLOAT / DOUBLE | `number` |
| DECIMAL | `string` (precision/scale already applied, e.g. `"1.0050"`) |
| VARCHAR | `string` (already UTF-8 decoded) |
| JSON | `string` (raw JSON text, not parsed — call `JSON.parse()` yourself if you want an object/array back) |
| INTERVAL | `{ months: number, days: number, micros: bigint }` |
| BLOB | `Uint8Array` |
| NULL | `null` |

Converting DECIMAL to `number` rounds once it exceeds 18 digits, so it's
returned as a string to avoid losing precision. If an approximation is fine,
just do `Number(row.amount)`.

A helper for turning TIMESTAMP into a `Date` is included; note it rounds down
to millisecond precision.

```js
timestampToDate(row.d);  // BigInt(micros) -> Date
dateToDate(row.day);     // DATE(days)     -> Date
```

## I/O and caching

The engine never blocks. When it runs out of bytes, it returns `NEED_IO`
along with a list of `{table, offset, len}`, and the host is expected to:

1. **Coalesce** — merge ranges whose gap is under 1 MiB. Fetching 900 KB once
   beats two 400 KB fetches around a 100 KB gap. The engine already batches
   requests per RowGroup for this reason, so don't defeat that by firing them
   off one at a time.
2. **Fetch in parallel** — fetch the coalesced ranges together with
   `Promise.all`. For a URL, use `Range: bytes=start-end`; for memory/Blob, slice.
3. **Supply** — hand the bytes back via `ahiru_provide` and continue the loop.
   If the same request repeats with zero bytes gained, that's treated as a
   livelock and raises `E504`.

### Codec delegation

ZSTD is decompressed by the core itself by default (feature `zstd`, ~13 KB —
small enough that splitting it into a separate module wasn't worth the extra
round-trip; DESIGN.md §6). GZIP stays delegated to the host on purpose, since
the browser/Node already ship a decompressor for it at zero extra bytes.
When the engine hits a codec it doesn't handle internally, it returns
`NEED_CODEC` with a list of `{table, codec, offset, len, out_len}`. The host then:

- **GZIP** … uses `DecompressionStream('gzip')`. Available in both browsers
  and Node 18+, so it costs zero extra bytes.
- **ZSTD** … only reaches the host at all if `ahiru-core.wasm` was built with
  the `zstd` feature turned off. In that case the host loads
  `crates/ahiru-zstd` as a separate wasm module **on first request** (via
  `zstdUrl` / `zstdBinary` / `zstdModule`). If none is configured, it raises
  E201 naming ZSTD specifically.
- **Anything else** (BROTLI, etc.) … E201, "unsupported compression codec".

The compressed block was already fetched by the preceding `NEED_IO`, so
**decompression never re-fetches it**. To make that possible, a per-table
cache of already-fetched bytes is kept (bounded by `cacheSize`; oldest entries
are evicted first and re-fetched from source if needed again). If a range that
was never fetched is requested, the host does not silently fetch it — that's
treated as an engine-side inconsistency and raises E900.

The cache key is an exact match on `(source, offset, len)`. `"memory"` is a
capacity-bounded LRU (64 MiB by default, adjustable via `cacheSize`). Passing
a `MemoryCache` instance directly lets multiple `AhiruDB` instances share one
(in that case `close()` does not clear it). `"cache-api"` currently falls back
to the in-memory implementation.

## Handling wasm memory (implementation notes)

`ahiru_alloc` / `ahiru_provide` can grow the wasm heap, and growth detaches any
existing `TypedArray` views at that instant. Since that failure mode is silent,
the following rules are fixed policy:

- Never hold a view into `memory.buffer` across a wasm call. Re-create it with
  `new Uint8Array(memory.buffer)` immediately after the call returns.
- The buffer behind `ahiru_out_ptr()` is overwritten by the next
  `ahiru_query_step` / `ahiru_schema` call. If you need the value later, copy
  it out to the JS side before the next call.
- `query()` copies data straight into row objects as it goes, so no separate
  copy step is needed. `stream()` hands batches to the caller, so column
  buffers are always copied before being yielded.
- Result buffers are only guaranteed to be 4-byte aligned. For `Float64Array` /
  `BigInt64Array` columns that don't happen to land on an 8-byte boundary,
  build them over a copy rather than a view.

## Errors

wasm only ever returns a numeric code; messages are assembled from the table
in `errors.js`. That alone keeps roughly 20 KB of strings out of wasm
(DESIGN.md §10).

```js
try {
  await db.query('SELECT FROM');
} catch (e) {
  e.code;    // 301
  e.message; // "[E301] unexpected token"
  e.sql;     // the SQL that was being executed
}
```

`errors.js` mirrors `Code` and `message()` from
`crates/ahiru-core/src/error.rs`, so **always update both together**. If they
drift apart, the test (`errors.js matches the Code / message in error.rs`) fails.

## Tests

```sh
./scripts/size.sh          # builds target/ahiru-core.wasm
node --test 'js/test/*.test.mjs'
```

Expected values come from the `duckdb` CLI (must be installed). The
range-fetch tests use a ~2 MB Parquet file generated by DuckDB, placed in a
temp directory (if a 64 KiB speculative footer fetch could read the whole
file, it wouldn't actually exercise projection pushdown).

CSV / JSONL tests need a wasm build with `--features csv,jsonl`; the test
suite builds `target/ahiru-core-full.wasm` automatically (override with
`AHIRU_WASM_FULL`). The ZSTD *delegation* tests (the opt-out fallback path)
need a core built with `--no-default-features` (i.e. without `zstd`) plus the
separate `ahiru-zstd` side module; the suite builds both automatically
(override the core with `AHIRU_WASM_NOZSTD`). If any of these builds isn't
available, the corresponding tests are skipped.

> On Node 24, `node --test js/test/` (passing a directory) doesn't work.
> Use a glob as shown above, or run `node --test` with no arguments.

## Limitations

- BROTLI / LZO / framed LZ4 are not decompressed (E201 names the codec). The
  delegation hook lives in the core, so adding support only requires a JS-side change.
- `parquet('...')` is the only SQL table function. There's no syntax for
  referencing CSV / JSONL directly by path — register them first and look them
  up by name.
