# ahirudb

**A SQL engine that queries Parquet, CSV, and JSON directly — in the browser,
in under 1 MiB of WebAssembly.**

Point it at a `.parquet` URL and run SQL against it. Only the byte ranges the
query actually needs are fetched, over HTTP Range requests, and nothing is
uploaded anywhere — the engine runs in the tab.

```js
import { AhiruDB } from './js/ahirudb.js';

const db = await AhiruDB.init({ wasmUrl: '/ahiru-core.wasm' });
db.register('trips', 'https://example.com/trips.parquet');   // no I/O yet

await db.query('SELECT vendor, count(*) c FROM trips GROUP BY 1 ORDER BY c DESC');
```

DuckDB-WASM is the obvious alternative and a far more complete database, but it
weighs tens of MB (roughly 10 MB even after brotli) — enough that shipping it
on a page is a decision, not a detail. Rather than *shrinking* DuckDB, ahirudb
fits a 1 MiB budget by **choosing what to include from the start**, and still
covers a practically DuckDB-shaped subset of SQL: joins, window functions,
recursive CTEs, correlated subqueries, `GROUPING SETS`, `PIVOT`/`UNPIVOT`,
JSON path operators, lambdas, regular expressions.

| Build | raw | gzip -9 | of 1 MiB budget |
|---|---:|---:|---:|
| Parquet only, ZSTD included (default) | 538.1 KiB | 232.8 KiB | 52.5% |
| Plus CSV + JSONL + JSON (everything read-side) | 568.3 KiB | 246.2 KiB | **55.5%** |

Measured by [`./scripts/size.sh`](scripts/size.sh) with `wasm-opt`. CI fails the
build if the fully-loaded configuration exceeds 1 MiB.

## Try it

```bash
./scripts/demo.sh
```

Builds the wasm binary and opens a local page where you can query the bundled
sample files — or drop in your own `.parquet`/`.csv`/`.jsonl`, which never
leaves the browser.

Or from the native CLI, no wasm involved:

```bash
cargo run -p ahiru-cli -- query tests/data/basic.parquet \
  "SELECT name, count(*) c FROM t GROUP BY name ORDER BY c DESC"
```

## What you get

**Reads what you already have.** Parquet (PLAIN / RLE / dictionary / DELTA
encodings; SNAPPY / LZ4_RAW / ZSTD decompressed in-core, GZIP delegated to the
host's `DecompressionStream`), CSV / TSV, JSONL / NDJSON, single-document JSON.
Globs and multi-file tables, Hive-partitioned directories (`year=2024/month=01/`)
exposed as filterable virtual columns, and nested `STRUCT` / `LIST` / `MAP`
columns.

**Reads as little as possible.** Projection pushdown, statistics-based RowGroup
pruning, page-level pruning via ColumnIndex/OffsetIndex, and Split Block Bloom
Filters — so a selective query over a large remote file fetches a handful of
ranges rather than the file.

**SQL that covers real work.** All join types (plus non-equi and correlated
subqueries), `GROUP BY`/`HAVING`/`GROUPING SETS`/`ROLLUP`/`CUBE`, window
functions and `QUALIFY`, `WITH RECURSIVE`, set operations, `PIVOT`/`UNPIVOT`,
`UNNEST`, `DATE`/`TIME`/`TIMESTAMP`/`INTERVAL`/`DECIMAL`/`UUID` with correct
semantics, and a `JSON` type with path operators. Verified against DuckDB:
the end-to-end tests run each query through both engines and compare.

**Runs anywhere JS does.** A zero-dependency, no-build-step ES module drives the
wasm core in the browser and in Node 18+, with range coalescing, LRU byte
caching, parameter binding, and streaming columnar batches. A native CLI covers
scripting and development.

**Writes, if you ask for it.** `CREATE TABLE`/`INSERT`/`UPDATE`/`DELETE` and
`COPY ... TO` (CSV, JSONL, and Parquet) exist as opt-in Cargo features, the
DDL/DML ones operating on in-memory tables only. They're off by default and
compiled out entirely, so the read-only distribution pays nothing for them.

**Limitations are explicit.** No spilling (exceeding the memory cap is a clean
error, never a wrong or partial result), no `ASOF JOIN`, no general `LATERAL`,
no transactions, no explicit window frames. The full list, along with the
places where behavior deliberately differs from DuckDB, is in
[docs/sql/limitations.md](docs/sql/limitations.md).

## How it works

The one idea worth knowing up front: **the engine never blocks on I/O**. When
it needs bytes it doesn't have, it returns the exact ranges it wants and
suspends; the host fetches them however it likes and resumes it. That's what
lets the same core run against an HTTP URL, a `File` picked in a form, or a
local path, without the engine knowing which — and without a threads-and-async
runtime inside the wasm budget.

The rest of the budget is held by the same kind of deliberate choice: `no_std`
with no `core::fmt`, six physical types instead of a general type system, error
*codes* in wasm with the message strings living in the JS host, and optional
features that compile out completely.

## Documentation

| | |
|---|---|
| [docs/sql/](docs/sql/README.md) | SQL reference — what you can write in a query, page by page |
| [js/README.md](js/README.md) | JS host API: registering sources, streaming, caching, error codes |
| [docs/DESIGN.md](docs/DESIGN.md) | Architecture and the reasoning behind every constraint above |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Building, testing, Cargo features, size measurement |

## License

MIT
