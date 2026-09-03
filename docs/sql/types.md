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
| `DATE` | Calendar date (days since epoch). Literal: `DATE '2024-01-01'` — see [Typed date/time literals](#typed-datetime-literals) below |
| `TIME` | Time of day (microsecond resolution). Literal: `TIME '10:20:30'` |
| `TIMESTAMP` | Date + time (microsecond resolution), no timezone. `DATETIME` is accepted as a synonym in `CAST`. Literal: `TIMESTAMP '2024-01-01 10:20:30'` |
| `TIMESTAMPTZ` | Date + time (microsecond resolution), an instant in UTC. `TIMESTAMP WITH TIME ZONE` is accepted as a synonym in `CAST`. See [TIMESTAMPTZ](#timestamptz) below |
| `INTERVAL` | A span of time — see [Interval literals](#interval-literals) below |
| `JSON` | Dynamically-typed JSON document. Also how Parquet `LIST`/`MAP` values are exposed — see [data-sources.md](data-sources.md#nested-parquet-types) |
| `UUID` | 16-byte UUID, displayed/parsed as `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`. See [UUID](#uuid) below |

## Integer promotion and mixed-type arithmetic

Kernels only exist for 6 physical types, so logical types are promoted onto
one of them:

- `TINYINT`/`SMALLINT`/`INTEGER`/`DATE`/`TIME` share the 32-bit physical lane.
- `BIGINT`/`TIMESTAMP`/`TIMESTAMPTZ`/`DECIMAL(p ≤ 18)` share the 64-bit lane.
- `HUGEINT`/`DECIMAL(p > 18)`/`INTERVAL` share the 128-bit lane.
- `UUID` shares the same byte-buffer lane as `VARCHAR`/`BLOB`, but (like
  `JSON`) never implicitly converts to/from them — see [UUID](#uuid).
- Unsigned integers promote to the next-wider *signed* physical width
  (`UINTEGER` behaves like a 64-bit value internally); only `UBIGINT`
  promotes all the way to the 128-bit lane.

When two different numeric types meet in an expression (`a + b`, `a = b`,
`UNION` of two queries, ...), ahirudb picks a common type to compare/combine
them in:

- `NULL` unifies with anything, becoming the other side's type.
- A **signed** integer combined with an **unsigned** one unifies to the
  smallest *signed* type that holds both domains, regardless of which side
  is written first (matching DuckDB): `TINYINT`+`UTINYINT` → `SMALLINT`,
  `SMALLINT`+`USMALLINT` → `INTEGER`, `INTEGER`+`UINTEGER` → `BIGINT`,
  `BIGINT`+`UBIGINT` → `HUGEINT`. When the signed side is already wide
  enough it simply wins (`BIGINT`+`UTINYINT` → `BIGINT`). Neither operand's
  own type could hold the other's whole range, so this is what keeps
  `0::UTINYINT + (-1)::TINYINT` at `-1` and `1::UBIGINT > (-1)::BIGINT`
  true.
- Otherwise the narrower type widens to the wider one, in this order:
  `BOOLEAN < TINYINT/UTINYINT < SMALLINT/USMALLINT < INTEGER/UINTEGER <
  BIGINT/UBIGINT < DECIMAL < HUGEINT < FLOAT < DOUBLE < DATE < TIME <
  TIMESTAMP < TIMESTAMPTZ < VARCHAR < BLOB`. This ordering does **not**
  decide a `DECIMAL`-with-integer pair, even though `HUGEINT` outranks
  `DECIMAL` in it: the `DECIMAL` rule below is checked first and wins. Nor
  does it decide a signed/unsigned pair — the two share a rank there, and
  the rule above is checked first.
- A `DECIMAL` combined with another `DECIMAL` — or with an **integer**,
  which counts as a `DECIMAL` of scale 0 — unifies to a `DECIMAL` wide
  enough for both: the new precision is
  `max(p1 - s1, p2 - s2) + max(s1, s2) + 1` (the `+1` covers carry on
  addition, matching DuckDB), and the new scale is `max(s1, s2)`. The
  precision is capped at 38.

  Widening the `DECIMAL` side rather than the integer side is what keeps
  the fractional digits: `DECIMAL(4,1) + BIGINT` becomes `DECIMAL(21,1)`
  and `DECIMAL(4,1) + HUGEINT` becomes `DECIMAL(38,1)` (uncapped it would
  be 40), so both of these are `8.5` and not `9`:

  ```sql
  SELECT CAST('7.5' AS DECIMAL(4,1)) + 1::BIGINT;    -- 8.5
  SELECT CAST('7.5' AS DECIMAL(4,1)) + 1::HUGEINT;   -- 8.5
  ```

  `typeof` here reports a bare `DECIMAL` without the precision and scale,
  so it cannot be used to observe the unified width the way DuckDB's
  `typeof` can.
- A `DECIMAL` combined with `FLOAT`/`DOUBLE` always becomes `DOUBLE` (never a
  wider `DECIMAL`); `FLOAT` combined with any other numeric type always
  becomes `DOUBLE` (never stays `FLOAT`).
- `DATE` combined with `TIMESTAMP` becomes `TIMESTAMP`.
- `DATE` or `TIMESTAMP` combined with `TIMESTAMPTZ` becomes `TIMESTAMPTZ`
  (the more specific type wins, matching DuckDB).
- `VARCHAR` combined with `BLOB` becomes `BLOB`.
- `INTERVAL`, `JSON`, and `UUID` only unify with themselves (or `NULL`) —
  mixing any of them with an unrelated type, including `VARCHAR`, is a type
  error. This means `WHERE uuid_col = '...'` needs an explicit
  `CAST('...' AS UUID)` — ahirudb doesn't implicitly coerce a bare string
  literal the way DuckDB does (consistent with how `WHERE date_col =
  '2024-01-01'` already requires an explicit `CAST(... AS DATE)` here).

`DECIMAL` **multiplication and division do not use the common type above**,
because the scale changes:

- `*` **adds** the scales and the precisions: `DECIMAL(4,1) * DECIMAL(3,2)`
  is `DECIMAL(7,3)`, so `1.5 * 1.25` is `1.875` and not `1.88`. The
  precision is capped at 38 (as in DuckDB: `DECIMAL(20,2) * DECIMAL(19,2)`
  is `DECIMAL(38,4)`), but a product whose **scale** would exceed 38 is an
  error (`ValueOutOfRange`) rather than a silently truncated type — again
  matching DuckDB. Cast an operand to `DOUBLE`, or to a `DECIMAL` with a
  smaller scale, when you hit it:

  ```sql
  SELECT 0.01::DECIMAL(25,20) * 0.01::DECIMAL(25,20);          -- error: scale 40 > 38
  SELECT 0.01::DECIMAL(25,20) * 0.01::DECIMAL(25,20)::DOUBLE;  -- 0.0001
  ```
- `/` always falls to `DOUBLE` (as in DuckDB). Integer division of the raw
  scaled values would subtract the scales and lose every fractional digit.

A plain (unsuffixed) integer literal is `INTEGER` if it fits, else `BIGINT`,
else `HUGEINT`.

Numeric literals accept **`_` as a digit separator** between digits
(`1_000_000`, `1_0.5_5`, `1.5e1_0`), and a float may be written in
**leading-dot form** (`.5` is `0.5`) — both matching DuckDB.

A literal with a decimal point or an exponent (`1.005`, `.5`, `1e3`) is
**`DOUBLE`**. DuckDB types the same literal as a `DECIMAL` wide enough to
hold it exactly (`typeof(1.005)` is `DECIMAL(4,3)` there, `DOUBLE` here).
The consequence is that exact-decimal identities hold in DuckDB and not
here:

```sql
SELECT 0.1 + 0.2 = 0.3;                          -- false here, true in DuckDB
SELECT CAST(123456789012345678.005 AS DECIMAL(30,3));
-- 123456789012345680.000 here (the literal was rounded to a DOUBLE first);
-- 123456789012345678.005 in DuckDB
```

Write `CAST('0.1' AS DECIMAL(...))` — or a `DECIMAL`-typed column — when
you need exact decimal arithmetic on constants.

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

`TRY_CAST` never raises — a failed conversion becomes `NULL` for that row
instead of aborting the query.

**`CAST` behaves the same way for numeric and text conversions**, which is
a deliberate divergence from DuckDB: unparseable text and an out-of-range
value both become `NULL` silently rather than failing the query.

```sql
SELECT CAST('abc' AS INTEGER);          -- NULL here; DuckDB raises a conversion error
SELECT CAST(3000000000 AS INTEGER);     -- NULL here; DuckDB raises "value out of range"
SELECT CAST(1e300::DOUBLE AS FLOAT);    -- NULL here (not `inf`); DuckDB raises
```

This is the same convention already documented for `DATE`/`TIME`/
`TIMESTAMP` (see [Typed date/time literals](#typed-datetime-literals)) and
`UUID` (see [UUID](#uuid)) — `CAST` and `TRY_CAST` are indistinguishable
outside the JSON case. It is a known gap rather than a design goal:
matching DuckDB needs a strict-cast flag threaded through the cast kernels
*and* a change to the `LIMIT`/`Project` evaluation order, so that a row a
`LIMIT` would have discarded cannot fail the query on a value the user
never asked to see. Until both land, treat `CAST` as `TRY_CAST` and use
`WHERE ... IS NOT NULL` (or `TRY_CAST` explicitly, to say so) rather than
relying on a cast to reject bad data.

The one conversion that does raise is non-JSON text `CAST AS JSON`.

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
TIMESTAMPTZ | TIMESTAMP WITH TIME ZONE
JSON
UUID
INTERVAL
```

`INTERVAL` is nameable in a type position, and casts round-trip through
text in both directions:

```sql
SELECT CAST('1 day' AS INTERVAL);                        -- 1 day
SELECT CAST(INTERVAL '1 day 02:03:04' AS VARCHAR);       -- '1 day 02:03:04'
SELECT CAST(CAST(INTERVAL '3 months 4 days 05:06:07' AS VARCHAR) AS INTERVAL);
-- 3 months 4 days 05:06:07
```

The text form is the same one the CSV/JSONL/Parquet exports write (see
[Interval literals](#interval-literals) below).

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
- Casting to a `DECIMAL` rounds **away from zero** at every scale, whether it
  is reducing an existing `DECIMAL`'s scale (`CAST(1.235 AS DECIMAL(10,2))` →
  `1.24`) or coming from a float (`CAST(2.5 AS DECIMAL(3,0))` → `3`, not the
  `2` a float-to-*integer* cast gives). Monetary rounding therefore doesn't
  systematically under-round, and `DECIMAL(p, 0)` follows the same rule as
  every other scale of the same type. DuckDB draws the line in the same place.
- Integer arithmetic overflow **wraps** (no error), except `SUM`, which
  accumulates in a 128-bit integer internally and only errors
  (`ValueOutOfRange`) if that itself overflows, and `factorial`/`!`, which
  errors on the same code the moment its `HUGEINT` result itself overflows
  (`factorial(34)` and above — see
  [functions-numeric.md](functions-numeric.md#factorial)).
- `-0.0` and `0.0` are treated as identical for grouping/join keys; all
  `NaN` values collapse to one representative for grouping purposes.
- **`NaN` compares under a total order, not under IEEE rules** (matching
  DuckDB): `NaN` is equal to itself and greater than everything else,
  including `+inf`. So `=`, `<>`, `<`/`>`, `IN`, join conditions,
  `GROUP BY`, `DISTINCT`, and `ORDER BY` all agree with each other, and a
  hash join and a nested-loop join over the same `NaN` keys produce the
  same rows.

  ```sql
  SELECT 'nan'::DOUBLE = 'nan'::DOUBLE;     -- true
  SELECT 'nan'::DOUBLE > 'inf'::DOUBLE;     -- true
  SELECT 'nan'::DOUBLE IN ('nan'::DOUBLE, 1.0);  -- true
  ```

  `ORDER BY` therefore sorts a `DOUBLE` column as
  `-inf < ... < +inf < NaN`, with `NULL`s placed by the usual
  `NULLS FIRST`/`NULLS LAST` rule.
- `CAST(<finite double> AS VARCHAR)` produces the **shortest decimal string
  that round-trips** back to the same `DOUBLE` — `100.0`, `1e+30`,
  `1e-05`, `0.1` — the same formatter the CSV/JSONL writers use, so a
  finite value survives a cast to text and back unchanged.
- A **non-finite** `DOUBLE` casts to the lowercase `nan`, `inf`, `-inf`,
  matching DuckDB, and that is also how the value displays when selected
  bare. The cast round-trips like the finite case does:

  ```sql
  SELECT CAST('inf'::DOUBLE AS VARCHAR);                    -- inf
  SELECT CAST('-inf'::DOUBLE AS VARCHAR);                   -- -inf
  SELECT CAST('nan'::DOUBLE AS VARCHAR);                    -- nan
  SELECT CAST(CAST('inf'::DOUBLE AS VARCHAR) AS DOUBLE);    -- inf (infinite)
  ```

  The CSV writer spells the infinities the same way (`inf`, `-inf`, matching
  DuckDB's own CSV output) but writes `NaN` rather than `nan`. The JSONL
  writer has to quote all three, and writes them longhand as `"NaN"`,
  `"Infinity"`, `"-Infinity"`; see [ddl-dml.md](ddl-dml.md#jsonl-output).

- A **`FLOAT`** casts to the shortest text that round-trips through `FLOAT`,
  not through `DOUBLE`: `CAST(1.1::FLOAT AS VARCHAR)` is `'1.1'`, not the
  `'1.100000023841858'` that spelling out the widened `DOUBLE` would give.
  The same goes for how a `FLOAT` displays in the CLI and how it is written
  by `COPY ... TO` in CSV and JSONL. (DuckDB agrees everywhere except its
  own JSON writer, which prints the widened `DOUBLE` there.)

## Typed date/time literals

```sql
SELECT DATE '2024-01-01';
SELECT TIME '10:20:30';
SELECT TIMESTAMP '2024-01-01 10:20:30';
SELECT TIMESTAMPTZ '2024-01-01 10:20:30+09';   -- 2024-01-01 01:20:30+00

SELECT * FROM trips WHERE pickup >= TIMESTAMP '2024-01-01 00:00:00';
```

A type name written directly in front of a single-quoted string is a
literal of that type. Only the four temporal types above take this form
here; DuckDB generalizes it to every type name (`INTEGER '5'`), which this
engine does not — use `CAST('5' AS INTEGER)` for the rest.

The text accepted is the same as the corresponding `CAST` from `VARCHAR`
(see [CAST and TRY_CAST](#cast-and-try_cast)), including the optional
timezone offset on `TIMESTAMPTZ`.

Two things worth knowing:

- **These are constants, not casts.** The value is converted once while the
  query is parsed, so a typed literal in `WHERE` participates in Parquet
  RowGroup/page/Bloom-filter pruning exactly like a number does. A
  `CAST('2024-01-01' AS DATE)` in the same position does not.
- **A literal that can't be parsed is an error, not `NULL`.** This is the
  one place where the "unreadable text becomes `NULL`" rule used for `CAST`
  does not apply: `SELECT DATE 'nonsense'` fails at parse time (matching
  DuckDB), because a literal is fixed query text and a typo there should be
  reported rather than silently turned into `NULL`.

`DATE`, `TIME`, `TIMESTAMP`, and `TIMESTAMPTZ` are **not** reserved words —
a column named `date` or `time` still works unquoted (`SELECT date FROM t`,
`ORDER BY time`, `SELECT 1 AS date`). The literal form is only recognized
when a string literal immediately follows the type name.

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

**Comparing** two intervals flattens those three components with DuckDB's
fixed conversions — **1 month = 30 days** and **1 day = 24 hours** — and
compares the resulting microsecond spans. There is no anchor date in a
comparison, so there is nothing to ask how long "one month" really is; only
adding an interval to a `DATE`/`TIMESTAMP` uses real calendar arithmetic.
The normalization applies everywhere a value is compared or keyed:

```sql
SELECT INTERVAL 1 DAY = INTERVAL 24 HOUR;   -- true
SELECT INTERVAL 1 MONTH = INTERVAL 30 DAY;  -- true
-- ORDER BY, DISTINCT, GROUP BY, UNION, equi-joins and min/max agree:
-- 23:00:00 < 1 day < 25:00:00
```

`<`, `<=`, `>` and `>=` between two intervals are not accepted yet
(`ORDER BY` on an interval column works).

## UUID

```sql
SELECT CAST('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11' AS UUID) FROM range(1);
```

A `UUID` is stored as its raw 16 bytes (not as text), and is ordered by
plain unsigned byte comparison — matching DuckDB's own ordering. `CAST`
only converts between `UUID` and `VARCHAR` (the usual dashed hex form,
case-insensitive on input, always lowercase on output); there's no direct
`CAST` to/from `BLOB` — go through `VARCHAR` if you need the raw bytes as
text. Reading a Parquet column with the `UUID` logical type annotation
produces a `UUID` column automatically, no `CAST` needed.

A row that isn't valid `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` hex becomes
`NULL` rather than failing the query, for both `CAST` and `TRY_CAST` — the
same convention this engine already uses for `DATE`/`TIME`/`TIMESTAMP`
parse failures (unlike DuckDB, which raises a hard error on `CAST` for an
invalid `UUID` string).

There's no `uuid()` function to generate a random one (see
[limitations.md](limitations.md)) — bring UUIDs in from your data instead.

## TIMESTAMPTZ

```sql
SELECT CAST('2024-01-01 12:00:00+09' AS TIMESTAMPTZ) FROM range(1);
-- 2024-01-01 03:00:00+00
SELECT CURRENT_TIMESTAMP FROM range(1);  -- also TIMESTAMPTZ, matching DuckDB
```

`TIMESTAMPTZ` has the exact same physical representation as `TIMESTAMP`
(microseconds since the epoch) — the difference is purely that a
`TIMESTAMPTZ` value is defined to already be a UTC instant, while a plain
`TIMESTAMP` carries no timezone information at all. Reading a Parquet
`TIMESTAMP` column produces `TIMESTAMPTZ` if the column's `isAdjustedToUTC`
logical-type flag is set, and plain `TIMESTAMP` otherwise (files written
via the legacy `ConvertedType` annotation, which has no such flag, always
read back as `TIMESTAMP`).

`CAST(... AS TIMESTAMPTZ)` from text accepts an optional timezone offset
suffix — `Z`, `+HH`, `+HH:MM`, `-HH`, or `-HH:MM` — and normalizes to UTC.
**There is no session timezone concept in this engine**, so a literal with
no offset is simply assumed to already be UTC (unlike DuckDB, which applies
its configured session timezone). Comparing/mixing `DATE` or `TIMESTAMP`
with `TIMESTAMPTZ` widens to `TIMESTAMPTZ` rather than erroring (see
[Integer promotion and mixed-type arithmetic](#integer-promotion-and-mixed-type-arithmetic)
above). Displaying a `TIMESTAMPTZ` always appends a `+00` suffix, since the
value is always UTC.

with zero-valued components omitted.
