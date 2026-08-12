# DDL, DML, and COPY TO

[← Back to index](README.md)

By default, ahirudb is **read-only**: it queries Parquet/CSV/JSONL/JSON
files without ever modifying them. `CREATE TABLE`, `INSERT`/`UPDATE`/
`DELETE`, and `COPY ... TO` are opt-in write-path features, each gated
behind its own Cargo feature and **off by default**. Disabling a feature
removes its code from the build entirely (this is what keeps the default
1 MiB wasm budget intact — see [DESIGN.md §16](../DESIGN.md)).

| Feature | Enables | Depends on |
|---|---|---|
| `ddl` | `CREATE`/`ALTER`/`DROP TABLE`, `CREATE`/`DROP VIEW` | — |
| `dml` | `INSERT`/`UPDATE`/`DELETE` | `ddl` |
| `export` | `COPY (SELECT ...) TO 'path' (FORMAT csv\|jsonl)` | — |
| `export-parquet` | Same, for Parquet output | `export` |

The native `ahiru-cli` binary enables `export` and `export-parquet` by
default; `ddl`/`dml` must be turned on explicitly:

```bash
cargo run -p ahiru-cli --features ahiru-core/ddl,ahiru-core/dml -- \
  query tests/data/basic.parquet "CREATE TABLE t2 (id INTEGER); ..."
```

## In-memory tables only

**`CREATE TABLE`/`INSERT`/`UPDATE`/`DELETE` only ever operate on tables you
created with `CREATE TABLE` yourself.** A table backed by a Parquet/CSV/
JSONL/JSON file — `parquet('...')`, or any name registered as a file source
— stays read-only forever; writing to one fails with a `ReadOnlyTable`
error rather than silently doing nothing or touching the file on disk.

```sql
-- t is a file-backed (Parquet) table
INSERT INTO t VALUES (3);   -- error: ReadOnlyTable
UPDATE t SET id = 9;        -- error: ReadOnlyTable
DELETE FROM t;              -- error: ReadOnlyTable
```

This split exists because a `Source`'s bytes are assumed immutable
everywhere else in the engine (pruning, caching, ...); in-memory tables are
a genuinely separate, row-oriented table type used only by DDL/DML.

## CREATE TABLE / DROP TABLE

```sql
CREATE TABLE accounts (id INTEGER, name VARCHAR, balance DECIMAL(10, 2));
INSERT INTO accounts VALUES (1, 'alice', 100.00), (2, 'bob', 50.00), (3, 'carol', 0.00);
SELECT id, name, balance FROM accounts ORDER BY id;

DROP TABLE accounts;
```

`CREATE TABLE ... AS SELECT` (CTAS) and `INSERT ... SELECT` are also
supported:

```sql
CREATE TABLE snap AS SELECT id, val FROM src WHERE val >= 20;

CREATE TABLE dst (id INTEGER, val INTEGER);
INSERT INTO dst SELECT id, val FROM snap;
```

`IF NOT EXISTS` and `OR REPLACE` are both supported. A `CREATE TABLE` whose
name collides with an existing **file-backed** table always fails
(`DuplicateTable`), regardless of `OR REPLACE` — `OR REPLACE` only ever
replaces another in-memory table:

```sql
CREATE TABLE IF NOT EXISTS u (id INTEGER);
CREATE OR REPLACE TABLE u (id INTEGER, name VARCHAR);
```

A column declared `NOT NULL` is enforced on every subsequent `INSERT`/
`UPDATE`:

```sql
CREATE TABLE t (id INTEGER NOT NULL);
INSERT INTO t VALUES (NULL);   -- error: TypeMismatch
```

## ALTER TABLE

```sql
ALTER TABLE t ADD COLUMN grade INTEGER DEFAULT 100;
ALTER TABLE t ADD COLUMN note VARCHAR;         -- no DEFAULT -> existing rows get NULL

ALTER TABLE t DROP COLUMN b;

ALTER TABLE accounts RENAME COLUMN balance TO bal;
ALTER TABLE accounts RENAME TO ledger;
```

Like `CREATE`/`DROP TABLE`, `ALTER TABLE` on a file-backed table fails with
`ReadOnlyTable`.

## INSERT / UPDATE / DELETE

```sql
INSERT INTO accounts (id, name, balance) VALUES (4, 'dave', 25.00);
INSERT INTO accounts SELECT * FROM accounts WHERE id = 1;   -- INSERT ... SELECT

UPDATE accounts SET balance = balance + 25.00 WHERE id <= 2;
DELETE FROM accounts WHERE balance = 0.00;
```

`UPDATE ... SET` uses simultaneous-assignment semantics (matching DuckDB):
every `SET` expression is evaluated against the row's values *before* the
update, so `UPDATE t SET a = b, b = a` swaps the two columns rather than
collapsing them to the same value.

## CREATE VIEW / DROP VIEW

```sql
CREATE VIEW dst_v AS SELECT id, val FROM dst;
SELECT * FROM dst_v WHERE val > 10;

CREATE OR REPLACE VIEW dst_v AS SELECT id, val, val * 2 AS doubled FROM dst;
DROP VIEW dst_v;
```

A view's query text is stored as-is and re-parsed/re-bound on every
reference (rather than a precompiled plan), so it always reflects the
current definition and current state of whatever it selects from,
including another view — recursive view references are rejected past a
fixed depth limit.

## COPY ... TO (export)

```sql
COPY (SELECT a, b FROM t) TO 'out.csv';
COPY (SELECT a FROM t) TO 'out.txt' (FORMAT csv);
COPY (SELECT a FROM t) TO 'out.jsonl' (FORMAT jsonl);
COPY (SELECT a FROM t) TO 'out.parquet';
COPY (SELECT a FROM t) TO 'out.bin' (FORMAT parquet);
COPY t TO 'out.csv';   -- shorthand for COPY (SELECT * FROM t) TO 'out.csv'
```

`FORMAT` defaults to whatever `path`'s extension implies, and is
case-insensitive when given explicitly. As on the read side, an extension
that isn't recognised (including no extension at all) means Parquet.

### Parquet output

Requires the `export-parquet` feature (on by default in the native CLI,
opt-in for wasm builds). The writer is deliberately plain: one uncompressed
`PLAIN`-encoded data page per column per row group, RLE definition levels,
122,880 rows per row group, and no dictionary, statistics, page index, or
bloom filters. Those are all optional parts of the format, so the output is
readable anywhere (there are DuckDB cross-checks in
`crates/ahiru-cli/tests/copy.rs`) — it is just larger and less prunable
than what a full-featured writer would produce.

SQL types map to their natural Parquet types, with two deliberate
exceptions:

- **INTERVAL** is written as text (`1 year 2 months 3 days 01:02:03`, the
  same rendering the CSV/JSONL exports use), not as the legacy FLBA(12)
  `INTERVAL` type, which cannot represent signed components. It reads back
  as `VARCHAR`.
- **HUGEINT** is written as `DECIMAL(38, 0)`, since Parquet has no 128-bit
  integer type. Values keep their exact value but read back as `DECIMAL`.

The full table is in the module doc of
`crates/ahiru-core/src/write/parquet/mod.rs`.

The engine core itself never touches a filesystem (it's `no_std`) — `COPY`
runs the query to completion in memory and hands the resulting bytes plus
the destination path back to the host, which performs the actual file
write. In the native CLI this happens automatically; a JS host would do
the equivalent via its own file-write API.

**Limitation:** `COPY`/CTAS/`INSERT ... SELECT` are non-resumable — if
reading the source data would require pausing for I/O partway through
(`NEED_IO`/`NEED_CODEC`), the statement fails with `IoFailed` instead of
suspending and resuming. They only work when the source data is already
fully available in memory (typical CLI usage, or a JS caller that
pre-fetched the table).
