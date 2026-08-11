# Data types

[← Back to index](README.md)

ahirudb keeps a strict separation between **logical types** (what SQL sees)
and **physical types** (what the execution engine computes on — only 6 of
them; see [DESIGN.md §8](../DESIGN.md)). As a user you only need the logical
types below; the physical collapse is an internal implementation detail that
never changes query results.

## Logical types

| SQL type | Notes |
|---|---|
| `BOOLEAN` | `TRUE` / `FALSE` / `NULL` |
| `TINYINT`, `SMALLINT`, `INTEGER`, `BIGINT`, `HUGEINT` | Signed integers, 8/16/32/64/128-bit |
| `UTINYINT`, `USMALLINT`, `UINTEGER`, `UBIGINT` | Unsigned integers, 8/16/32/64-bit |
| `FLOAT`, `DOUBLE` | IEEE 754 32-bit / 64-bit float |
| `DECIMAL(precision, scale)` | Fixed-point. `precision` 1–38, `scale` ≤ `precision`. Bare `DECIMAL`/`NUMERIC` (no parens) means `DECIMAL(18, 3)` |
| `VARCHAR` | UTF-8 text. `TEXT`/`STRING`/`CHAR` are accepted as synonyms in `CAST` |
| `BLOB` | Byte string. `BYTEA` is accepted as a synonym in `CAST` |
| `DATE` | Calendar date (days since epoch) |
| `TIME` | Time of day (microsecond resolution) |
| `TIMESTAMP` | Date + time (microsecond resolution). `DATETIME` is accepted as a synonym in `CAST` |
| `INTERVAL` | A span of time — see [Interval literals](#interval-literals) below |
| `JSON` | Dynamically-typed JSON document. Also how Parquet `LIST`/`MAP` values are exposed — see [data-sources.md](data-sources.md#nested-parquet-types) |

There is no dedicated `UUID` type and no `CAST(... AS UUID)` — a Parquet
column with the UUID logical annotation is read as raw bytes/text instead.

## Integer promotion and mixed-type arithmetic

Kernels only exist for 6 physical types, so logical types are promoted onto
one of them:

- `TINYINT`/`SMALLINT`/`INTEGER`/`DATE`/`TIME` share the 32-bit physical lane.
- `BIGINT`/`TIMESTAMP`/`DECIMAL(p ≤ 18)` share the 64-bit lane.
- `HUGEINT`/`DECIMAL(p > 18)`/`INTERVAL` share the 128-bit lane.
- Unsigned integers promote to the next-wider *signed* physical width
  (`UINTEGER` behaves like a 64-bit value internally); only `UBIGINT`
  promotes all the way to the 128-bit lane.

When two different numeric types meet in an expression (`a + b`, `a = b`,
`UNION` of two queries, ...), ahirudb picks a common type to compare/combine
them in:

- `NULL` unifies with anything, becoming the other side's type.
- Otherwise the narrower type widens to the wider one, in this order:
  `BOOLEAN < TINYINT/UTINYINT < SMALLINT/USMALLINT < INTEGER/UINTEGER <
  BIGINT/UBIGINT < DECIMAL < HUGEINT < FLOAT < DOUBLE < DATE < TIME <
  TIMESTAMP < VARCHAR < BLOB`.
- Two `DECIMAL`s unify to a `DECIMAL` wide enough for both: the new
  precision is `max(p1 - s1, p2 - s2) + max(s1, s2) + 1` (the `+1` covers
  carry on addition, matching DuckDB), and the new scale is `max(s1, s2)`.
- A `DECIMAL` combined with `FLOAT`/`DOUBLE` always becomes `DOUBLE` (never a
  wider `DECIMAL`); `FLOAT` combined with any other numeric type always
  becomes `DOUBLE` (never stays `FLOAT`).
- `DATE` combined with `TIMESTAMP` becomes `TIMESTAMP`.
- `VARCHAR` combined with `BLOB` becomes `BLOB`.
- `INTERVAL` and `JSON` only unify with themselves (or `NULL`) — mixing
  either with an unrelated type is a type error.

A plain (unsuffixed) integer literal is `INTEGER` if it fits, else `BIGINT`,
else `HUGEINT`.

## CAST and TRY_CAST

> The `SELECT`s on this page are written as bare expressions for brevity.
> ahirudb requires a `FROM` clause on every `SELECT` (see
> [queries.md](queries.md#overall-shape)) — run any of these directly by
> appending `FROM range(1)`, e.g. `SELECT CAST('42' AS INTEGER) FROM range(1);`.

```sql
SELECT CAST('42' AS INTEGER);
SELECT CAST(1.2345 AS DECIMAL(10, 2));   -- 1.23
SELECT TRY_CAST('not a number' AS INTEGER);  -- NULL, no error
```

`CAST` raises an error on a conversion it can't perform (e.g. `CAST('abc' AS
INTEGER)`, or non-JSON text `CAST AS JSON`). `TRY_CAST` never raises — a
failed conversion becomes `NULL` for that row instead of aborting the query.
This matters most when casting a whole column: a single bad row with `CAST`
fails the entire query, while `TRY_CAST` just nulls that row out.

Accepted `CAST` type-name spellings (case-insensitive):

```
BOOLEAN | BOOL
TINYINT
SMALLINT
INT | INTEGER
BIGINT
HUGEINT
UTINYINT
USMALLINT
UINTEGER
UBIGINT
FLOAT | REAL
DOUBLE
DECIMAL | NUMERIC        -- bare form = DECIMAL(18, 3)
DECIMAL(p, s) | NUMERIC(p, s)
VARCHAR | TEXT | STRING | CHAR
BLOB | BYTEA
DATE
TIME
TIMESTAMP | DATETIME
JSON
```

`INTERVAL` has no `CAST` spelling — the only way to produce one is the
`INTERVAL '...'` literal syntax below.

## NULL and three-valued logic

`NULL` means "unknown," and comparisons/boolean logic follow standard SQL
three-valued logic (`TRUE` / `FALSE` / `UNKNOWN`, where `UNKNOWN` rows are
filtered out by `WHERE`/`JOIN ON`/`HAVING` just like `FALSE`):

- `NULL = NULL` is `NULL` (not `TRUE`) — use `IS NULL` / `IS NOT NULL` to
  test for it directly.
- `AND`/`OR` follow the standard truth tables with a `NULL` operand (e.g.
  `FALSE AND NULL` is `FALSE`, but `TRUE AND NULL` is `NULL`).
- Aggregates other than `COUNT(*)` and `COUNT(expr)` ignore `NULL` inputs;
  an aggregate over zero non-null rows returns `NULL` (`SUM`) or `0`
  (`COUNT`), matching standard SQL.
- Division by zero and integer `MIN_VALUE / -1` return `NULL`, not an
  error. Floats stay IEEE (`inf` / `NaN` are real, distinct values).
- `GROUP BY`/`JOIN` keys and `DISTINCT` treat all `NULL`s as equal to each
  other for grouping purposes (standard SQL grouping semantics, distinct
  from `=` comparison semantics above).

## Rounding and floating-point conventions

These are deliberately matched to DuckDB, since the two diverge easily and
it needs to be explicit:

- Casting a float to an integer type rounds to the **nearest even**
  (`CAST(1.5 AS INTEGER)` → `2`, `CAST(4.5 AS INTEGER)` → `4`).
- Reducing a `DECIMAL`'s scale rounds **away from zero**
  (`CAST(1.235 AS DECIMAL(10,2))` → `1.24`), so monetary rounding doesn't
  systematically under-round.
- Integer arithmetic overflow **wraps** (no error), except `SUM`, which
  accumulates in a 128-bit integer internally and only errors
  (`ValueOutOfRange`) if that itself overflows.
- `-0.0` and `0.0` are treated as identical for grouping/join keys; all
  `NaN` values collapse to one representative for grouping purposes.
  Comparison operators still treat `NaN` as not-equal-to-anything (`false`)
  except `<>`, which is `true`.

## Interval literals

```sql
SELECT INTERVAL '1 year 2 months 3 days';
SELECT INTERVAL 1 DAY;
SELECT CAST('2024-01-01' AS DATE) + INTERVAL 1 DAY;   -- 2024-01-02
```

Accepted units (singular or plural): `YEAR(S)`, `MONTH(S)`, `DAY(S)`,
`HOUR(S)`, `MINUTE(S)`, `SECOND(S)`, `MILLISECOND(S)`, `MICROSECOND(S)`.
Other DuckDB shorthand units (`mon`, `y`, `wk`, ...) are not recognized.

An interval is stored as one packed value combining a month count, a day
count, and a microsecond count (kept separate internally because "1 month"
isn't a fixed number of days — calendar arithmetic needs to know which unit
moved). Displaying one back out looks like:

```
1 year 2 months 3 days 01:02:03
```

with zero-valued components omitted.
