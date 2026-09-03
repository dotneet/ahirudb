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
extension defaults to Parquet. A **local** file name that contains `#` or
`?` is detected by its extension like any other — the URL query/fragment
strip that would otherwise cut `data.csv?v=2` down to `data.csv` only
applies to actual URLs, so a file genuinely named `report#final.csv` reads
as CSV rather than as Parquet.

## Referring to a file from SQL

A quoted path in SQL auto-registers the file it names, but **only in table
position** — directly after `FROM` or `JOIN`, or as the first argument of a
reader function (`read_csv`, `read_json`, `parquet`, ...):

```sql
SELECT count(*) FROM 'data/events.csv';          -- registers and reads the file
SELECT count(*) FROM read_csv('data/events.csv');-- same
SELECT '/etc/passwd';                            -- just a string: '/etc/passwd'
SELECT length('./foo.csv');                      -- just a string: 9
```

A string literal anywhere else stays a string. This is what keeps an
ordinary `'/'` or `'.'` in a comparison or a `split_part` call from walking
the filesystem.

Paths in table position are glob patterns. A backslash escapes a glob
metacharacter, so the literal file `star*.csv` is nameable:

```sql
SELECT * FROM 'star\*.csv';   -- exactly the file called star*.csv
SELECT * FROM 'star*.csv';    -- every file matching star*.csv
```

## Text-format type inference

CSV/TSV column types are sniffed from a leading sample of the file (up to
256 KiB), JSON/JSONL types from the values actually seen. What the sniffer
does with an ambiguous or mixed column:

| Input | Inferred | Why |
|---|---|---|
| `007`, `042` | `VARCHAR` | A leading zero is significant — reading these as `7`/`42` would destroy the padding |
| `true`, `1` | `VARCHAR` (CSV, JSONL) / `JSON` (`.json`) | `BOOLEAN` mixed with a number widens rather than picking one — `true` has no numeric reading. DuckDB gives all three a `JSON` column; the CSV/JSONL readers predate this engine's `JSON` type and carry the raw text in a `VARCHAR` instead |
| `2020-01-02 03:04:05`, `2020-01-03` | `TIMESTAMP` | A bare date in a `TIMESTAMP` column reads as midnight (`2020-01-03 00:00:00`) |
| NDJSON lines that aren't objects (`1`, `"two"`) | one `VARCHAR` column named `json` | There are no object keys to make columns out of; the raw line is the value |
| JSON/NDJSON records that are all `{}` | one column named `json` | Same reason: the records carry no keys, so each record itself is the value. A table with no columns at all could not report a row count |

The sample is the leading 256 KiB, grown (up to 1 MiB, the longest record
the readers accept) when that does not contain one complete record — so a
CSV header, or a first JSONL line, longer than 256 KiB still reads.

A value **outside** the sample that doesn't fit the inferred type raises a
conversion error rather than becoming `NULL` — see
[limitations.md](limitations.md#partially-supported), where the divergence
from DuckDB (which re-sniffs and widens) is spelled out.

An object **key** outside the sample is never silently dropped either, but
the two JSON readers handle it differently, because of how much of the file
each one has in hand:

- **`.json`** (one document) has the whole document resident and walks
  every element anyway, so *every* element contributes its keys: the column
  set is always complete, no matter how late a key first appears. Only the
  types of the columns the sample settled on stay frozen (which is what
  produces the conversion error above). A column discovered past the sample
  has no sample evidence at all, so its type widens over every element it
  appears in.
- **JSONL/NDJSON** is read split by split, and the schema has to be fixed
  before the first split is read, so a complete key set is not knowable up
  front. A key with no column is reported as a `ColumnNotFound` error
  positioned at the offending line, rather than being skipped. DuckDB, which
  can re-sniff, would add the column instead; erroring is the honest answer
  for a streaming reader — the alternative is losing a whole column of data
  without a word.

When several **text** parts are registered as one multi-part table and
their sniffed schemas disagree, the column widens to `VARCHAR`. A part that
saw **no value at all** for a column (a header-only CSV, a column whose
every cell was empty, a JSON key that was always `null`) is not a
disagreement, though — it has nothing to say, so the other parts' type
wins. `a.csv` (`id,v` / `1,10`) plus a header-only `c.csv` keeps
`id BIGINT`, as DuckDB does. Parquet parts stay strict: a declared type
mismatch between two Parquet parts is a `TypeMismatch` error, since a
Parquet schema is a declaration rather than a guess.

### Malformed CSV records

The reader never silently discards part of a record:

- A row with **more** fields than the header is a parse error, as it is in
  DuckDB. Dropping the surplus hid the usual cause — an unquoted delimiter
  inside a value, which shifts every following field.
- Any byte between a **closing quote** and the next delimiter or line
  terminator is a parse error too (`"x"junk,1`), again matching DuckDB.
- A row with **fewer** fields is `NULL`-padded (DuckDB's `null_padding`),
  since nothing the file contains is lost that way.
- A **blank line** is skipped, except in a one-column file, where it is the
  only way to spell an empty record and so reads as one `NULL` row —
  `a\n1\n\n2\n` is three rows, as in DuckDB.
- A file that mixes line terminators, so that a lone `\r` appears among
  `\n`/`\r\n` records, treats that `\r` as a record terminator rather than
  merging the two records into one. (DuckDB refuses to sniff such a file at
  all; this is what its `strict_mode=false` reading gives.) Such a file is
  read as a single split, like a quoted or CR-only one — see
  [limitations.md](limitations.md#performance-adjacent-notes-not-correctness-bugs).

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
multi-part table — see below. A file whose *name* contains a `+` still
opens: `\+` escapes the separator explicitly (`a\+b.csv`), and a spec whose
`+`-separated parts don't all exist while the whole spec names a real file
is treated as that one file anyway.

A glob's results never include directories (`dir/*` yields the files
directly inside `dir`, not its subdirectories), matching the shell and
DuckDB. Use `dir/**` — or just `dir/` — to descend.

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

Partition columns come from the file *path*, so they appear however the
table was registered — a directory, a glob, a `+`-joined list, or a single
explicit file path (`t=sales/year=2024/month=01/part.parquet` exposes
`year` and `month` too, as DuckDB does).

A partition key's **type is decided once for the whole table**, not per
partition: `k=1`/`k=2` gives an `INTEGER` virtual column, and `k=1`/`k=abc`
gives `VARCHAR`, because the partitions disagree and the union has to
widen. A zero-padded value stays `VARCHAR` (`k=007`/`k=42` is a `VARCHAR`
column holding `007` and `42`), the same rule the CSV sniffer uses — so
`007` keeps its padding rather than being flattened to `7`. Typing the key
once means a predicate on it compares the same way in every partition.

If a partition key **also names a real column inside the file**, the file's
column wins and no virtual column is synthesized for that part — Hive and
DuckDB's `WRITE_PARTITION_COLUMNS` both write the key into the data as
well, and this matches what DuckDB reads back. (Two partition keys of the
same name in one path are still an error: there is no data column to
prefer.)

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
| Fixed-width annotations | `DECIMAL` and `UUID` as usual, plus `INTERVAL` (FLBA(12), read as `INTERVAL`) and `FLOAT16` (FLBA(2), widened to `FLOAT`) |
| Not supported | `BYTE_STREAM_SPLIT` encoding, encryption, `LZO`/`BROTLI` codecs |

Projection pushdown, RowGroup statistics pruning, page-level pruning
(ColumnIndex/OffsetIndex), and Split Block Bloom Filter probing are all
automatic — the engine only fetches the byte ranges a query actually needs.
Predicates that benefit from this: equality, `BETWEEN`/range comparisons,
and `IN (...)` lists of literals.

Pruning never changes a query's answer, only how many bytes are read, so
the engine gives it up wherever it cannot compare exactly:

- A literal is rescaled into the column's own representation first — `150`
  becomes the `1500` a `DECIMAL(5,1)` column actually stores, `DATE
  '2024-01-01'` becomes microseconds against a `TIMESTAMP` column. When no
  exact form exists (a fractional literal against a `DECIMAL`, a value too
  wide for the column), that predicate simply does not prune.
- `>` and `>=` on a `FLOAT`/`DOUBLE` column never prune on `max`. Writers
  leave `NaN` out of the statistics, but this engine (like DuckDB) orders
  `NaN` above every other value, so those rows can match a bound that lies
  beyond `max`.
- `DECIMAL` columns stored as `FIXED_LEN_BYTE_ARRAY`/`BYTE_ARRAY`
  (precision 19 and above), `INT96` columns, and string columns (whose
  statistics the writer may truncate) are not pruned at all.
- In a multi-file table, a file whose own column type differs from the
  table's unified type (one file's `DECIMAL(5,1)` next to another's
  `DECIMAL(8,3)`) is read in full — its statistics are in a different
  domain from the predicate's constant.
