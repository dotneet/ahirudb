# Date and time functions

[← Back to index](README.md)

> Every `SELECT` below is written as a bare expression for brevity. ahirudb
> requires a `FROM` clause on every `SELECT` (see
> [queries.md](queries.md#overall-shape)) — run any of these directly by
> appending `FROM range(1)`, e.g. `SELECT CURRENT_DATE FROM range(1);`.

See [types.md](types.md) for the `DATE`/`TIME`/`TIMESTAMP`/`INTERVAL` types
themselves and interval-literal syntax (`INTERVAL '1 day'`, usable directly
with `+`/`-`: `d + INTERVAL 1 DAY`).

## CURRENT_DATE and now()

```sql
SELECT CURRENT_DATE, CURRENT_TIMESTAMP, now(), today(), current_time;
```

These are the only "clock" access the engine has, and they're handled
specially: the query's start time is passed in once by the host (there's
no clock inside the `no_std` wasm core itself) and substituted into the
query as a constant *before* binding — so `CURRENT_TIMESTAMP` evaluates
**once per query**, not once per row:

```sql
SELECT id, CURRENT_TIMESTAMP FROM t ORDER BY id;   -- same timestamp on every row
```

Because of this substitution, `CURRENT_DATE`/`current_time` are recognized
only in their bare keyword form and take precedence over a same-named
column (`SELECT current_date FROM t` returns the constant, not a `t.
current_date` column, if one existed) — while `today`/`now` as *bare
identifiers without parentheses* are ordinary column references, not
special forms; only the call forms `now()`/`today()` are magic. There is no
general-purpose "read the clock" scalar function beyond these fixed forms.

## Extracting fields

```sql
SELECT year(d), month(d), day(d) FROM t LIMIT 1;
SELECT date_part('year', d) FROM t LIMIT 1;    -- alias: datepart, extract
SELECT extract(year FROM d) FROM t LIMIT 1;
SELECT date_part('epoch', d) FROM t LIMIT 1;   -- BIGINT seconds, floored (not fractional)
```

`year`/`quarter`/`month`/`week`/`day`/`dayofmonth`/`hour`/`minute`/
`second`/`dayofweek`/`dayofyear`/`epoch` are all shorthand for the
equivalent `date_part(...)` call. `week` uses ISO 8601 week numbering
(`date_trunc('week', ...)` treats Monday as the start of the week).

## Truncating, formatting, parsing

```sql
SELECT date_trunc('month', d) FROM t LIMIT 1;   -- always returns TIMESTAMP, even truncating a DATE
SELECT strftime(d, '%Y-%m-%d') FROM t LIMIT 1;  -- only %Y %m %d %H %M %S %% are interpreted
SELECT to_date('2024-05-01');                   -- strict YYYY-MM-DD
SELECT to_timestamp('2024-05-01 10:00:00');     -- YYYY-MM-DD[ T]HH:MM[:SS[.ffffff]]
```

`strftime` only understands `%Y`/`%m`/`%d`/`%H`/`%M`/`%S`/`%%` — it is not
a full strftime implementation; unrecognized specifiers pass through as
literal text rather than erroring. `to_timestamp` here is a **string
parser** (`YYYY-MM-DD` optionally followed by a time), unlike DuckDB's
`to_timestamp`, which takes epoch-seconds as a `DOUBLE` — an intentional,
documented incompatibility with that one DuckDB function name.

## Arithmetic

```sql
SELECT date_add('month', 1, d) FROM t LIMIT 1;   -- date_add(part, n, timestamp)
SELECT date_diff('day', d, d + INTERVAL 5 DAY) FROM t LIMIT 1;
SELECT last_day(d) FROM t LIMIT 1;               -- last day of the month, as a DATE
SELECT CAST('2024-01-01' AS DATE) + INTERVAL 1 DAY;  -- ordinary +/- with an INTERVAL also works
```

`date_add(part, n, timestamp)` deliberately does **not** match DuckDB's
`date_add(timestamp, INTERVAL ...)` signature — there's no scalar-function
overload taking an `INTERVAL` argument directly, since ordinary `+`/`-`
already covers that case (see the last example above). Adding a `year`/
`quarter`/`month` clamps the day-of-month to the target month's length
(e.g. adding 1 month to Jan 31 lands on Feb 29 in a leap year, not
Mar 3). `date_diff` counts calendar-unit **boundaries crossed**, not an
elapsed-time division — 23:00 to 01:00 the next day counts as 1 day
crossed, not 0.
