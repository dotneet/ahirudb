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

A view that contains `now()` / `CURRENT_DATE` is rewritten at **query** time
(not at `CREATE VIEW`), so each `SELECT` from the view sees that query's
start time.

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
SELECT dayname(d), monthname(d) FROM t LIMIT 1;  -- 'Wednesday', 'August' (English only)
SELECT epoch_ms(ts), epoch_us(ts), epoch_ns(ts) FROM t LIMIT 1;
```

`year`/`quarter`/`month`/`week`/`day`/`dayofmonth`/`hour`/`minute`/
`second`/`millisecond`/`microsecond`/`dayofweek`/`isodow`/`dayofyear`/
`century`/`decade`/`millennium`/`isoyear`/`epoch` are all shorthand for the
equivalent `date_part(...)` call. `week` uses ISO 8601 week numbering
(`date_trunc('week', ...)` treats Monday as the start of the week).

Part names are case-insensitive, a trailing `s` is ignored (`years` =
`year`), and DuckDB's abbreviations are accepted:
`y`/`yr`/`yrs`, `mon`/`mons`, `d`/`dayofmonth`, `h`/`hr`/`hrs`,
`m`/`min`/`mins` (`m` is **minute**, not month, as in DuckDB),
`s`/`sec`/`secs`, `ms`/`msec`/`msecs`, `us`/`usec`/`usecs`,
`w`/`weekofyear`, `c`/`cent`/`centuries`, `dec`/`decs`,
`mil`/`mils`/`millennia`, `weekday`, `isoweekday`.

| Part | Notes |
|---|---|
| `millisecond` (`ms`), `microsecond` (`us`) | Include the whole seconds field, matching DuckDB: `millisecond` of `11:59:44.123456` is `44123`, not `123` |
| `dayofweek` (`dow`) | Sunday = 0 … Saturday = 6 |
| `isodow` (`isoweekday`) | Monday = 1 … Sunday = 7 |
| `isoyear` | The ISO 8601 week-numbering year, which can differ from `year` at a year boundary: `isoyear` of 2024-12-30 is 2025 |
| `century` | Years 1–100 are century 1, so 2021 → 21 and 2000 → 20 |
| `millennium` | Same 1-based counting, so 2024 → 3 |
| `decade` | `year / 10`, so 2021 → 202 |

The same part names work with `date_trunc`/`date_diff`/`date_add`
(`isodow` excepted — there is nothing to truncate or add there).

**`century`/`millennium` mean something different in `date_trunc` and
`date_diff` than in `date_part`** — this is DuckDB's own inconsistency and
is matched deliberately. `date_part` counts them 1-based (2024 is century
21), while `date_trunc` and `date_diff` use a plain `year / 100`:
`date_trunc('century', DATE '2024-05-05')` is `2000-01-01` (not
`2001-01-01`) and `date_diff('century', DATE '1900-01-01', DATE
'2024-01-01')` is 1.

`date_diff('microsecond', a, b)` raises `value out of range` instead of
wrapping when the difference does not fit in a `BIGINT` (only the
microsecond unit can overflow; the others divide before subtracting).

`epoch_ms`/`epoch_us`/`epoch_ns` are the sub-second counterparts of the
`epoch` part: the same instant rescaled to milliseconds, microseconds, or
nanoseconds since 1970-01-01. `dayname`/`monthname` return English names
only — the engine carries no locale data.

## Truncating, formatting, parsing

```sql
SELECT date_trunc('month', d) FROM t LIMIT 1;   -- always returns TIMESTAMP, even truncating a DATE
SELECT strftime(d, '%Y-%m-%d') FROM t LIMIT 1;  -- only %Y %m %d %H %M %S %% are interpreted
SELECT to_date('2024-05-01');                   -- strict YYYY-MM-DD
SELECT to_timestamp('2024-05-01 10:00:00');     -- YYYY-MM-DD[ T]HH:MM[:SS[.ffffff]][zone]
SELECT make_date(2024, 2, 29);                  -- 2024-02-29 (a DATE)
SELECT make_timestamp(2024, 8, 14, 13, 45, 30); -- 2024-08-14 13:45:30
```

`make_date(y, m, d)` and `make_timestamp(y, m, d, h, mi, s)` return `NULL`
for an out-of-range component — `make_date(2023, 2, 29)` is `NULL` because
2023 is not a leap year — where DuckDB raises. That follows this engine's
general "prefer `NULL` over erroring mid-scan" policy. `make_timestamp`
accepts, and normalizes, any time of day in `[00:00:00, 24:00:00]` with
minutes below 60 and seconds at or below 60, exactly as DuckDB does:
`make_timestamp(2024, 6, 5, 24, 0, 0)` is the next day's midnight and
`make_timestamp(2024, 6, 5, 7, 8, 60)` is `07:09:00`. Anything past
`24:00:00` is out of range and `NULL`. DuckDB's
single-argument `make_timestamp(microseconds)` overload and its `DOUBLE`
seconds argument (fractional seconds) are not provided; use a `CAST` or add
an `INTERVAL` for those.

**Text → `DATE`/`TIMESTAMP` casts** accept the same shapes as DuckDB:
`YYYY-MM-DD`, optionally followed by `T` or one or more spaces and a
`HH:MM[:SS[.ffffff]]` time, optionally followed by a zone suffix (`Z`,
`[+-]HH[[:]MM]`, or a separate ` UTC` word — the offset is only validated,
never applied, because `TIMESTAMP` has no zone). A trailing `.` with no
digits after the seconds is accepted and means zero. A cast to `DATE`
accepts a timestamp-shaped string and keeps only the date part, so an ISO
timestamp column casts to `DATE` cleanly:
`'2024-01-01T10:00:00'::DATE` is `2024-01-01`. Anything else — including
DuckDB's habit of ignoring arbitrary trailing text in a `DATE` cast
(`'2024-01-01x'::DATE`) — becomes `NULL` here; see
[limitations.md](limitations.md#partially-supported).

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
