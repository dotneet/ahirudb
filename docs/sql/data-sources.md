# Data sources

[← Back to index](README.md)

ahirudb queries files directly — there's no import/load step. A "table" is
just a name bound to one or more Parquet/CSV/TSV/JSONL/JSON files.

## Formats

| Format | Cargo feature | Extensions |
|---|---|---|
| Parquet | always on | `.parquet` |
| CSV / TSV | `csv` | `.csv`, `.tsv` |
| JSONL / NDJSON | `jsonl` | `.jsonl`, `.ndjson` |
| JSON (single document: a top-level array or object) | `jsonl` | `.json` |

Parquet is the primary target and is always available. CSV/TSV and
JSONL/JSON support are opt-in Cargo features on the core (kept out of the
default wasm build to protect the size budget — see
[DESIGN.md §5](../DESIGN.md)); the native `ahiru-cli` and default JS host
build enable them.

Format is normally inferred from the file name's extension; an unrecognized
extension defaults to Parquet.

### JSON / JSONL shapes

```sql
-- top-level JSON array of objects, or one JSON object per line (JSONL)
SELECT id, name, score FROM t ORDER BY id LIMIT 3;

-- a top-level JSON *object* (not an array) becomes a 1-row table
SELECT a, b FROM t;   -- against a file containing {"a": 1, "b": "hello"}

-- a top-level JSON array of scalars (not objects) exposes one column
-- named "json"
SELECT sum(json) AS total FROM t;   -- against a file containing [1, 2, 3]
```

## Reading via the CLI

The native `ahiru-cli` binds file arguments positionally as `t`, `t2`,
`t3`, ... (see [ddl-dml.md](ddl-dml.md) for how the JS host's `db.register`
API does the equivalent registration):

```bash
cargo run -p ahiru-cli -- query tests/data/basic.parquet \
  "SELECT name, count(*) c FROM t GROUP BY name ORDER BY c DESC"

# a JOIN across two files
cargo run -p ahiru-cli -- query tests/data/small_a.parquet tests/data/small_b.parquet \
  "SELECT a.k, b.w FROM t AS a LEFT JOIN t2 AS b ON a.k = b.k ORDER BY a.k"
```

`ahiru query`'s `+`-joined syntax (`a.parquet+b.parquet+c.parquet`) is a
CLI-only convenience for binding several files as **one** logical
multi-part table — see below.

## Multi-file tables and Hive partitions

Several files with the same schema can be registered as a single logical
table (a "multi-part" table) rather than as separate `t`/`t2`/... tables.
Column *names* across all parts must match exactly — matching column
*positions* with different names is rejected (`TypeMismatch`), since name
matching (not just positional matching) is what makes the union safe.

```bash
# tests/data/multi/{a,b,c}.parquet, all (id INTEGER, name VARCHAR):
# 100 + 150 + 230 = 480 rows total, queried as one table
cargo run -p ahiru-cli -- query "tests/data/multi/a.parquet+tests/data/multi/b.parquet+tests/data/multi/c.parquet" \
  "SELECT count(*) FROM t"
```

**Hive-style partition directories** (`year=2024/month=01/part.parquet`)
work the same way: each partition-key segment in the path (`year`, `month`
above) becomes an extra, automatically-exposed, filterable virtual column,
on top of whatever the file itself contains:

```sql
-- fixture layout:
--   tests/data/hive/year=2024/month=01/part.parquet
--   tests/data/hive/year=2024/month=02/part.parquet
--   tests/data/hive/year=2025/month=01/part.parquet
SELECT count(*) FROM t;                                  -- all partitions, 1000 rows
SELECT count(*) FROM t WHERE year = 2024 AND month = 1;   -- 300 rows, one partition pruned in
```

Registering a glob pattern or a directory of partitions is a host-side (JS
API / CLI-glue) concern — the engine core itself has no filesystem access
(it's `no_std`); the host resolves the file list and passes it to the
multi-part registration call.

## Nested Parquet types

Parquet's nested types (`STRUCT`, `LIST`, `MAP`) don't map onto SQL columns
1:1, so ahirudb takes a deliberate middle path (see
[DESIGN.md §5](../DESIGN.md)):

- **`STRUCT` flattens into dotted column names**, recursively:

  ```sql
  -- duckdb schema: id INTEGER, address STRUCT(city VARCHAR, zip INTEGER)
  SELECT id, address.city, address.zip FROM t ORDER BY id LIMIT 5;

  -- flattening recurses through arbitrarily deep nesting:
  -- duckdb schema: id INTEGER, nested STRUCT(a STRUCT(b STRUCT(c INTEGER)))
  SELECT id, nested.a.b.c FROM t ORDER BY id;
  ```

  A `NULL` struct value nulls out every one of its flattened leaf columns
  for that row.

- **`LIST` and `MAP` (and any `STRUCT` that contains one) become a single
  `JSON`-typed column** instead of flattening, since there's no way to turn
  a variable-length array into a fixed set of columns:

  ```sql
  -- xs is a LIST<INTEGER> column in the source Parquet file
  SELECT id, xs FROM t ORDER BY id;
  -- xs comes back as JSON text, e.g. "[1,2,3]"

  -- a LIST value can itself be NULL, empty, or contain NULL elements —
  -- all three are distinguishable:
  --   NULL      -- the list itself is SQL NULL
  --   "[]"      -- an empty array
  --   "[3,null,6]"  -- a NULL element inside a non-null array

  -- a STRUCT that contains a LIST becomes one JSON column too (not
  -- flattened into struct-dotted-names + a separate list column):
  SELECT id, s FROM t;   -- s: '{"name":"n0","tags":["t0","t1"]}'

  -- MAP renders as a JSON array of {"key": ..., "value": ...} pairs
  -- (works the same way whether keys are strings or numbers):
  SELECT id, m FROM t;   -- m: '[{"key":"a","value":0},{"key":"b","value":0}]'
  ```

Once a `LIST`/`MAP` column is in this `JSON` representation, the full
JSON-path/`list_*`/`map_*`/lambda function surface documented in
[functions-json.md](functions-json.md) applies to it directly — including
`UNNEST` in the `SELECT` list or `FROM` clause (see
[queries.md](queries.md#unnest)).

## Parquet coverage

| Item | Support |
|---|---|
| Container | Parquet v1 and v2 data pages |
| Encodings | PLAIN, RLE, RLE_DICTIONARY, PLAIN_DICTIONARY, DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY |
| Compression | UNCOMPRESSED, SNAPPY, LZ4_RAW, ZSTD (built into the engine); GZIP (delegated to the host — `DecompressionStream` in the browser/Node, or a `gzip` subprocess in the native CLI) |
| Physical types | BOOLEAN, INT32, INT64, FLOAT, DOUBLE, BYTE_ARRAY, FIXED_LEN_BYTE_ARRAY, INT96 (timestamp-compatible) |
| Not supported | `BYTE_STREAM_SPLIT` encoding, encryption, `LZO`/`BROTLI` codecs |

Projection pushdown, RowGroup statistics pruning, page-level pruning
(ColumnIndex/OffsetIndex), and Split Block Bloom Filter probing are all
automatic — the engine only fetches the byte ranges a query actually needs.
Predicates that benefit from this: equality, `BETWEEN`/range comparisons,
and `IN (...)` lists of literals.
