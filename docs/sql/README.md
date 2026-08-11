# ahirudb SQL reference

ahirudb queries Parquet, CSV/TSV, JSONL/NDJSON, and JSON files directly with
a large, practically-DuckDB-shaped subset of SQL — window functions,
recursive CTEs, correlated subqueries, `GROUPING SETS`, `PIVOT`/`UNPIVOT`,
lambda expressions, regular expressions, and more. This is the end-user
reference for what you can actually write in a query. For the engine's
internal architecture and the reasoning behind its design decisions, see
[DESIGN.md](../DESIGN.md) instead.

## Quickstart

The native `ahiru-cli` binary is the quickest way to try a query — no wasm
involved:

```bash
cargo run -p ahiru-cli -- schema tests/data/basic.parquet
cargo run -p ahiru-cli -- dump tests/data/basic.parquet 10
cargo run -p ahiru-cli -- query tests/data/basic.parquet \
  "SELECT name, count(*) c FROM t GROUP BY name ORDER BY c DESC"
```

Passing multiple files binds them as `t`, `t2`, `t3`, ... so they can be
joined:

```bash
cargo run -p ahiru-cli -- query tests/data/small_a.parquet tests/data/small_b.parquet \
  "SELECT a.k, b.w FROM t AS a LEFT JOIN t2 AS b ON a.k = b.k ORDER BY a.k"
```

From JS (browser/Node/Workers), via the `ahirudb` npm package:

```js
import { AhiruDB } from "ahirudb";

const db = await AhiruDB.init({ wasmUrl: "/ahiru-core.wasm" });
db.register("trips", "https://example.com/trips.parquet");
const rows = await db.query("SELECT vendor, count(*) c FROM trips GROUP BY 1 ORDER BY c DESC");
```

See [DESIGN.md §10](../DESIGN.md) for the full JS API (streaming results,
parameter binding, custom byte sources, memory limits).

## Pages

| Page | Covers |
|---|---|
| [queries.md](queries.md) | `SELECT` end to end: `WHERE`, joins, subqueries, CTEs (incl. `WITH RECURSIVE`), `UNION`/`INTERSECT`/`EXCEPT`, `GROUP BY`/`HAVING`/`GROUPING SETS`/`ROLLUP`/`CUBE`, window functions/`QUALIFY`, `SAMPLE`, `PIVOT`/`UNPIVOT`, `UNNEST`, `generate_series`/`range`, `DESCRIBE`/`SHOW TABLES`/`EXPLAIN` |
| [types.md](types.md) | Data types, `CAST`/`TRY_CAST`, `NULL`/three-valued logic, rounding rules, `INTERVAL` literals |
| [data-sources.md](data-sources.md) | Reading Parquet/CSV/JSONL/JSON, glob and multi-file tables, Hive partitions, nested `STRUCT`/`LIST`/`MAP` |
| [ddl-dml.md](ddl-dml.md) | `CREATE`/`ALTER`/`DROP TABLE`, views, `INSERT`/`UPDATE`/`DELETE`, `COPY ... TO` — all opt-in, in-memory-only features |
| [limitations.md](limitations.md) | What's not supported, and DuckDB-visible behavior differences worth knowing about |

### Function reference

| Page | Covers |
|---|---|
| [functions-string.md](functions-string.md) | Case/trim/pad, substrings/search/split, `LIKE`/`ILIKE`/`GLOB`/`SIMILAR TO`, regular expressions, `printf`/`format` |
| [functions-numeric.md](functions-numeric.md) | Arithmetic, rounding, `greatest`/`least`/`coalesce`/`nullif` |
| [functions-datetime.md](functions-datetime.md) | `CURRENT_DATE`/`now()`, field extraction, truncation, parsing/formatting, date arithmetic |
| [functions-json.md](functions-json.md) | `JSON` path operators and functions, list/map access, lambda expressions (`list_transform`/`list_filter`/`list_reduce`) |
| [functions-aggregate.md](functions-aggregate.md) | Aggregate functions (`sum`, `stddev`, `string_agg`, ...) and window functions (`row_number`, `lag`, ...) |

## What's not covered here

The write path (`CREATE TABLE`/`INSERT`/`UPDATE`/`DELETE`/`COPY TO`) is
opt-in and off by default — see [ddl-dml.md](ddl-dml.md) for which Cargo
features turn it on. Everything else on this page works out of the box in
the default build. For what's deliberately out of scope entirely (`ASOF
JOIN`, transactions, constraints, ...), see
[limitations.md](limitations.md).
