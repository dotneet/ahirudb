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
// are both deferred until the first query that actually reads the table.
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

Row objects need one key per column, so **output columns that share a name are
disambiguated**: the first keeps the name and the rest get `_1`, `_2`, ...
appended, as DuckDB's own clients do. `SELECT id, id, name AS id FROM t` yields
`{ id, id_1, id_2 }`. `batch.column(k)` and `batch.get(k, row)` still resolve a
name to the *first* column with it — index them by position to reach the others.
`batch.schema` always reports the real column names, unchanged.

### Finishing a stream

Only one query runs against an instance at a time, and a `stream()` iterator
holds that slot for as long as it is alive. **A stream you stop consuming early
must be finished**, or every later `query()`/`stream()` on the instance waits
for it forever:

```js
for await (const batch of db.stream(sql)) { if (done) break; }  // fine: for-await
                                                               // calls return()
                                                               // on break/throw

const it = db.stream(sql)[Symbol.asyncIterator]();
await it.next();
await it.return();  // required when driving the iterator by hand
```

`close()` is the other way out: it aborts whatever run is in flight, releases
the slot, and closes that query. An abandoned iterator resumed after `close()`
throws rather than stepping a handle that is no longer valid.

### Table names and paths

A registered name is an SQL identifier: `FROM trips`, `FROM TRIPS` and
`FROM "Trips"` all find `register('trips', …)`, and registering `'Trips'`
later replaces it. Only ASCII letters fold, as in the engine (and DuckDB):
`'Ärger'` and `'ärger'` are two tables.

A path in a string literal — `FROM 'https://…/a.parquet'`,
`parquet('…')`, `read_parquet('…')`, `read_csv('…')`, `read_json('…')` — is
registered automatically, as a table named by that exact string, the first
time a statement reads it. Paths are **case-sensitive**:
`parquet('https://h/Data.parquet')` and `parquet('https://h/data.parquet')`
are two different URLs. `read_csv` / `read_json` pick CSV / JSON(L) for an
extensionless path; otherwise the extension decides.

Which tables a statement uses is decided by the engine, not by scanning the
SQL text: registering a table does no I/O, and a table is sized (a `HEAD`
for a URL) and read only when a statement actually resolves it — directly,
or through a view. A registered name that merely appears as a column, alias,
comment or string value costs nothing. `SHOW TABLES` lists every
registration.

### Format

Passing `format` uses it directly (`ahiru_register_as`). If omitted, the engine
infers it **from the extension of the registered name** (`format::FormatKind::detect`).

```js
db.register('logs', bytes, { format: 'csv' }); // the name can be a plain identifier
await db.query('SELECT * FROM logs');

db.register('logs.csv', bytes);                // let the extension decide
await db.query('SELECT * FROM "logs.csv"');    // reference it quoted on the SQL side
```

`format` accepts `parquet` / `csv` / `tsv` / `jsonl` / `json`. An explicit value takes
priority even if it disagrees with the extension (decoupling the name from how
it's read is the whole point of this option, so we don't validate that away).
Only a misspelled value fails, with E409 — falling back to Auto would read it
as Parquet and fail with an opaque `BadMagic` instead.

Extension-based detection recognizes `.csv` / `.tsv` / `.tab` / `.jsonl` /
`.ndjson` / `.json`; anything else is treated as Parquet. `.json` means a
single top-level JSON document (array of objects, or one object) — the
`read_json`/`read_json_auto` shape, distinct from `.jsonl`'s one-object-per-line.
CSV, JSONL, and single-document JSON are gated
behind wasm-side features (`--features csv,jsonl`), so they aren't present in
the default distribution build. Registering them against a build that lacks
the feature raises E409.

### Parameters

Accepted types: `null` / `boolean` / `number` / `bigint` / `string` / `Uint8Array`.
Safe integers and `bigint` are sent as I64 (BIGINT); other `number` values as F64.
`Date` is not accepted (implicitly converting to microseconds would hide
off-by-a-magnitude bugs). A BIGINT does not compare with a TIMESTAMP (E404), so
to compare against one, bind microseconds and convert in SQL, or bind an ISO
string and cast it:

```js
await db.query('SELECT * FROM t WHERE ts > make_timestamp(?)', [BigInt(d.getTime()) * 1000n]);
await db.query('SELECT * FROM t WHERE ts > ?::TIMESTAMP', [d.toISOString()]);
```

The number of values must match the number of placeholders: too many is
E406, like too few.

### COPY ... TO

The engine never writes files. With a wasm core built with the `export`
feature, `COPY (SELECT …) TO 'out.csv'` runs the query and hands the encoded
bytes to the `onCopy` option; `query()` resolves to `[]` once it returns.
Without `onCopy` the statement fails with E409 instead of silently doing
nothing.

```js
const db = await AhiruDB.init({
  wasmUrl: '/ahiru-core-export.wasm',
  onCopy: async (path, bytes) => { await writeSomewhere(path, bytes); },
});
await db.query("COPY (SELECT * FROM trips WHERE fare > 100) TO 'big.parquet'");
```

`COPY`, `CREATE TABLE … AS SELECT` and `INSERT … SELECT` run to completion
inside a single engine call, so they cannot pause for I/O partway. When they
need bytes the engine reports them, the host fetches them and starts the
statement again; each attempt gets further, until everything it reads is in
the wasm heap at once. Fine for modest inputs; for large ones prefer
streaming `SELECT` results.

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

A request can also be a **size request** (offset `2^64-1`, length 0): the
engine is about to read a table it has not been told the length of. The host
calls the source's `size()` (a `HEAD` for a URL, in parallel for several
tables) and answers with `ahiru_set_size`.

Fetched bytes live in the wasm heap only while a query runs: when it
finishes, the engine drops them, and the next query asks again — which the
range cache below answers without touching the network. So `heapUsed` stays
near the size of what one query needs, not everything ever read, and
`memoryLimit` bounds a single query. With `cache: 'none'`, repeating a query
repeats its fetches.

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
decompression does not normally re-fetch it. To make that possible, a per-table
copy of already-fetched bytes is kept (bounded by `cacheSize`; oldest entries
are evicted first). A block whose copy was evicted is sliced out of the
coalesced range that contained it if the range cache still holds that range;
only when neither has it is it fetched again (with `cache: 'none'`, or a cache
too small to keep it). If a range that was never fetched is requested, the host
does not silently fetch it — that's treated as an engine-side inconsistency and
raises E900.

The cache key is an exact match on `(source, offset, len)`. `"memory"` is a
capacity-bounded LRU (64 MiB by default, adjustable via `cacheSize`). Passing
a `MemoryCache` instance directly lets multiple `AhiruDB` instances share one
(in that case `close()` does not clear it). `"cache-api"` currently falls back
to the in-memory implementation.

**The cache assumes registered sources are immutable.** The key has no ETag,
Last-Modified, or version component, so if a URL's content changes between
queries, a shared `MemoryCache`/`"memory"` cache keeps serving the bytes it
fetched the first time for any range it already holds — not the new content.
This is fine for the common case (versioned object storage, content-addressed
paths, files that are written once and read many times), but it means a URL
whose content is mutated in place will read stale after the first query. If
your data can change under a fixed URL, either give each `AhiruDB` its own
cache (the default — don't pass a shared `MemoryCache` instance), use
`cache: 'none'`, or put a version/content-hash in the URL/path itself so a
changed file gets a new cache key.

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
  e.code;     // 301
  e.message;  // "[E301] unexpected token"
  e.sql;      // the SQL that was being executed
  e.position; // 7 -- where the engine located it, when it knows (a byte
              // offset into the UTF-8 SQL for syntax errors)
}
```

### When the engine traps

A pathological query can crash the wasm engine itself (a WebAssembly
*trap*: native stack exhaustion from extreme nesting, or an internal panic).
A trapped instance cannot be trusted again, so that query fails with E900
("the wasm engine trapped …", the original `RuntimeError` as `cause`) and
the host swaps in a fresh instance of the same module before the next call:
registrations are declared again and cached bytes reused, so later queries
work as before. The exception is a database holding in-memory tables or
views created by SQL, which a fresh instance would silently lose: it stays
unusable (every call throws E900) and must be recreated.

`errors.js` mirrors `Code` and `message()` from
`crates/ahiru-core/src/error.rs`, so **always update both together**. If they
drift apart, the test (`errors.js matches the Code / message in error.rs`) fails.

Two codes are raised by this host rather than by wasm and are worth calling out:

- **E108** `invalid UTF-8 in string data` — a VARCHAR / JSON value in the source
  file is not valid UTF-8. The engine passes those bytes through untouched, so
  the decoder here is where it surfaces. It is a property of the data, not an
  engine bug: E900 is reserved for a result buffer that is structurally wrong.
- **E504** `io failed: no progress for ranges [...]` — a `ByteSource` returned
  the same range twice without adding a byte. Empty reads are deliberately not
  cached, so a source that returns an empty body once and then recovers works;
  one that never returns bytes fails here instead of spinning.

## Security

**Running untrusted SQL against a Node process with network access is not
safe.** `FROM 'URL'` / `parquet('URL')` / `read_csv('URL')` (and `register(name, url)`) make
plain HEAD and `Range` HTTP requests from wherever the JS host runs. There is
no URL allowlist and no way to disable URL sources. In a browser this is
constrained by CORS and same-origin policy the same as any other `fetch`; in
Node there is no such boundary — the process can reach anything on its
network, including `http://127.0.0.1/...`, other hosts on a private network,
and cloud instance-metadata endpoints (`http://169.254.169.254/...`). SQL that
embeds a URL is therefore effectively an SSRF primitive if it comes from an
untrusted source (a user-supplied query string, an LLM-generated query, etc.).

If you need to run SQL you don't fully trust in a Node process that also has
access to internal services:

- Don't. Run it in an environment with no route to anything sensitive, or
- Supply your own `ByteSource` implementations (via `register(name, source)`)
  instead of registering URLs at all, so the host never makes an HTTP request
  on the engine's behalf, or
- Set `sqlUrlPolicy: false` (or a synchronous/asynchronous callback that
  allowlists the URL's origin) when constructing the database. The callback
  sees **every** path a SQL string literal names that is not a registered
  table — any scheme, relative and protocol-relative (`//host/x`) paths
  included — resolved the way `fetch()` would resolve it (against
  `document.baseURI` / `location.href` in a browser; verbatim where there is
  no base, so a callback should reject what it cannot parse). It receives
  `(url, { functionName, sql })` before anything is registered or fetched;
  explicit `register(name, url)` calls remain the caller's responsibility.
  When a policy is configured, SQL-discovered URLs also use
  `redirect: "error"`, so an allowed origin cannot redirect the fetch to a
  different host behind the callback's decision.

The default remains permissive for compatibility with the documented
`parquet('URL')` shorthand. For untrusted SQL, explicitly set the policy to
`false` or reject private/non-allowlisted origins in the callback. `AhiruDB.init({
fetch })` also lets a caller enforce a final network-level policy for every URL
request, including explicit registrations and the wasm/ZSTD URLs.

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
- The reader functions take one argument, the path: named options
  (`delim=`, `header=`, ...) and globs are not supported. Register the source
  with an explicit `format` when the extension does not say it.
