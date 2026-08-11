# ahirudb

A lightweight SQL engine that queries Parquet directly, built to run in under 1 MiB of WASM.

DuckDB-WASM weighs tens of MB (roughly 10 MB even after brotli). Rather than
*shrinking* DuckDB, ahirudb fits the 1 MiB budget by **choosing what to include
from the start**.

See [docs/DESIGN.md](docs/DESIGN.md) for the full design.

## Current size

| Configuration | raw | gzip -9 | of budget |
|---|---:|---:|---:|
| `ahiru-core.wasm` (Parquet only) | 377.4 KiB | 169.0 KiB | 36.9% |
| `ahiru-core.wasm` (+ CSV + JSONL) | 404.9 KiB | 181.1 KiB | **39.5%** |
| `ahiru-zstd.wasm` (separate module) | 13.3 KiB | 6.9 KiB | outside budget |

Figures include `wasm-opt`. The `ddl`/`dml`/`export` write-path features are
opt-in and excluded from these default numbers — see [DESIGN.md §16](docs/DESIGN.md).

Measured with `./scripts/size.sh`, which reports a breakdown per configuration
and the incremental cost of adding CSV / JSONL. The gate is judged on the
**fully-loaded configuration** (passing only with a trimmed distribution
wouldn't actually enforce the budget). CI fails if it exceeds 1 MiB.

## Supported features

**Formats (read)**
- Parquet — Thrift Compact metadata, PLAIN / RLE / dictionary / DELTA page encodings, UNCOMPRESSED / SNAPPY / LZ4_RAW natively, GZIP via host delegation, ZSTD via a separate lazily-loaded module
- CSV / TSV (feature `csv`), JSONL/NDJSON (feature `jsonl`), single-document JSON arrays (`read_json`/`read_json_auto`-style)
- glob / multi-file tables, Hive-style partition directories (`year=2024/month=01/...`), automatically exposed and filterable as virtual columns
- Nested Parquet columns (`STRUCT`, `LIST`, `MAP`) — `STRUCT` is flattened into dotted column names where possible; `LIST`/`MAP` (and `STRUCT` containing them) are exposed as a `JSON`-typed column
- Pushdown: projection pushdown, statistics-based RowGroup pruning, page-level pruning via ColumnIndex/OffsetIndex, Split Block Bloom Filters

**SQL**
- Joins: `INNER` / `LEFT` / `RIGHT` / `FULL` / `CROSS`, non-equi, correlated and uncorrelated subqueries (`EXISTS` / `IN` / scalar), `UNNEST` (in the `SELECT` list and, implicitly lateral, in `FROM`)
- Aggregation: `GROUP BY` / `HAVING` / `DISTINCT` / `FILTER (WHERE ...)`, `GROUPING SETS` / `ROLLUP` / `CUBE`, statistical aggregates (`stddev`, `variance`, `median`, `mode`, `approx_count_distinct`), `string_agg` / `array_agg`
- Window functions with `ROWS`/`RANGE` frames, `QUALIFY`
- `WITH` (CTEs, including `WITH RECURSIVE`), `UNION` / `INTERSECT` / `EXCEPT`
- `DISTINCT ON`, `ILIKE`, `TRY_CAST`, `IIF`, regular expressions (`regexp_matches` / `regexp_extract` / `regexp_replace`)
- `JSON` type with path operators (`->`, `->>`, `json_extract`, `json_type`, `json_array_length`, `json_object`, `json_array`, `list_extract`, `map_extract`, ...) — `LIST`/`MAP` values share this same representation
- `DATE` / `TIME` / `TIMESTAMP` / `INTERVAL` arithmetic, `DECIMAL` with correct scale propagation
- `DESCRIBE`, `SHOW TABLES`, `EXPLAIN`

**Write path (opt-in, off by default — see [DESIGN.md §16](docs/DESIGN.md))**
- `CREATE TABLE` / `ALTER TABLE` (`ADD`/`DROP`/`RENAME COLUMN`, `RENAME TO`) / `DROP TABLE`, `CREATE VIEW` / `DROP VIEW` — feature `ddl`, effective only on in-memory tables created this way, never on file-backed tables
- `INSERT` / `UPDATE` / `DELETE` — feature `dml`
- `COPY (SELECT ...) TO 'file' (FORMAT csv|jsonl)` and the underlying `TableSink`/`export_all` API — feature `export`. The core never touches a filesystem itself; `ahiru-cli` performs the actual file write

**Runtime**
- wasm ABI + split-boundary I/O barrier (the engine never blocks; it returns the exact byte ranges it needs and resumes when they're supplied)
- JS host layer (Node and browser): range-request coalescing, LRU byte caching, codec delegation (GZIP via `DecompressionStream`, ZSTD via `ahiru-zstd`)
- Native `ahiru-cli` for development, testing, and scripting

**Known gaps**: `PIVOT`/`UNPIVOT`, `ASOF JOIN`, general `LATERAL` (beyond `UNNEST`), `CREATE MACRO`, sequences/constraints, transactions, `ATTACH`, named parameters. See [docs/DESIGN.md §15](docs/DESIGN.md) for the full list of intentional limitations.

## Usage (native CLI)

Development and testing run natively; debugging through wasm is inefficient.

```bash
cargo run -p ahiru-cli -- schema tests/data/basic.parquet
```

```bash
cargo run -p ahiru-cli -- dump tests/data/basic.parquet 10
```

```bash
cargo run -p ahiru-cli -- query tests/data/basic.parquet "SELECT name, count(*) c FROM t GROUP BY name ORDER BY c DESC"
```

Passing multiple files binds them as `t`, `t2`, ... so they can be joined.

```bash
cargo run -p ahiru-cli -- query tests/data/small_a.parquet tests/data/small_b.parquet "SELECT a.k, b.w FROM t AS a LEFT JOIN t2 AS b ON a.k = b.k ORDER BY a.k"
```

## Tests

```bash
cargo test
```

Test data is generated with the DuckDB CLI.

The SQL end-to-end tests (`crates/ahiru-cli/tests/sql_e2e.rs`) **don't hardcode
expected values — the same query is run against DuckDB and the results are
compared**. Hand-written expected values would silently turn a typo into the
spec, and every new query would need its expectation computed by hand. Using
DuckDB as the reference implementation means adding one line adds one more
verified case. On environments without DuckDB installed, those tests are skipped.

## Size measurement

```bash
./scripts/size.sh
```

With `wasm-opt` (binaryen) and `twiggy` installed, it also reports the
optimized size and a function-by-function breakdown.

## Limitations

Intentional limitations are catalogued in [docs/DESIGN.md §15](docs/DESIGN.md).
In short: no spilling (exceeding the memory budget is a hard error), and the
gaps listed under "Known gaps" above. Rounding conventions (float-to-integer
uses round-half-to-even; DECIMAL scale reduction rounds away from zero) are in
the same section.

## License

MIT
