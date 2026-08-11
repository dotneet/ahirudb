# Aggregate and window functions

[← Back to index](README.md)

See [queries.md](queries.md#aggregation-group-by-having-grouping-setsrollupcube)
for `GROUP BY`/`HAVING`/`GROUPING SETS`/`ROLLUP`/`CUBE` syntax and
[queries.md](queries.md#window-functions) for `OVER`/`WINDOW`/`QUALIFY`
syntax. This page is the function list.

## Aggregate functions

```sql
SELECT count(*), count(name), count(DISTINCT name) FROM t;
SELECT sum(id), avg(score), min(score), max(score) FROM t;
SELECT stddev(score), variance(score) FROM t;    -- sample stddev/variance
SELECT median(score) FROM t;                      -- continuous/interpolated median
SELECT mode(name) FROM t;                         -- most frequent value
SELECT approx_count_distinct(name) FROM t;
SELECT string_agg(name, ',') FROM t WHERE id < 3;
SELECT array_agg(id) FROM t WHERE id < 3;
```

| Function | Aliases | Notes |
|---|---|---|
| `count(*)` | — | Row count, ignoring `NULL`s in no particular column |
| `count(x)` | — | Count of non-`NULL` `x` values |
| `count(DISTINCT x)` | — | Count of distinct non-`NULL` `x` values |
| `sum(x)` | — | Integer inputs accumulate in a 128-bit integer to avoid overflow; only errors (`ValueOutOfRange`) if *that* itself overflows |
| `min(x)` / `max(x)` | — | Returns the input type unchanged |
| `avg(x)` | `mean(x)` | Result is always `DOUBLE` |
| `stddev(x)` | `stddev_samp(x)` | Sample standard deviation |
| `variance(x)` | `var_samp(x)` | Sample variance |
| `median(x)` | — | Continuous/interpolated median (equivalent to the 0.5 quantile) |
| `mode(x)` | — | Most frequent value; ties broken by first-seen (implementation-defined, matching DuckDB) |
| `approx_count_distinct(x)` | — | Currently an **exact** count internally, despite the name — a HyperLogLog-based approximation is a possible future swap, not a correctness gap today |
| `string_agg(x, sep)` | `group_concat(x, sep)` | `sep` must be a constant literal; defaults to `''` if omitted |
| `array_agg(x)` | `list(x)` | Collects values into a `JSON`-array-shaped result (no separate LIST physical type — see [functions-json.md](functions-json.md)) |

Every aggregate supports `DISTINCT` and `FILTER (WHERE cond)`:

```sql
SELECT count(*) FILTER (WHERE flag) AS n_true FROM t;
SELECT count(DISTINCT name) FILTER (WHERE flag) FROM t;
```

Aggregates other than `COUNT(*)`/`COUNT(x)` ignore `NULL` inputs and return
`NULL` (not `0`) over zero non-null rows. See
[types.md](types.md#null-and-three-valued-logic) for the general NULL
rules.

## Window functions

```sql
SELECT id, row_number() OVER (ORDER BY id) FROM t LIMIT 3;
SELECT id, rank() OVER (ORDER BY score) FROM t LIMIT 3;
SELECT id, dense_rank() OVER (ORDER BY score) FROM t LIMIT 3;
SELECT id, lag(score) OVER (ORDER BY id) FROM t LIMIT 3;
SELECT id, lead(score, 1, 0.0) OVER (ORDER BY id) FROM t LIMIT 3;
SELECT id, first_value(score) OVER (PARTITION BY flag ORDER BY id) FROM t LIMIT 3;
SELECT id, last_value(score) OVER (PARTITION BY flag ORDER BY id) FROM t LIMIT 3;
```

| Function | Notes |
|---|---|
| `row_number()` | No arguments; 1-based position within the partition |
| `rank()` | No arguments; ties share a rank, next rank skips |
| `dense_rank()` | No arguments; ties share a rank, next rank doesn't skip |
| `lag(x[, offset[, default]])` | Up to 3 arguments; `offset` defaults to 1, `default` defaults to `NULL` |
| `lead(x[, offset[, default]])` | Same shape as `lag`, looking forward instead of back |
| `first_value(x)` | First value in the current window frame |
| `last_value(x)` | Last value in the current window frame |

Any aggregate function from the table above can also run as a window
function: `sum(x) OVER (...)`, `count(*) OVER (...)`, `avg(x) OVER
(PARTITION BY ...)`, etc.

Not implemented: `ntile`, `percent_rank`, `cume_dist`, `nth_value`.

Window buffering has the same fixed in-memory cap as other blocking
operators — see [limitations.md](limitations.md#no-spilling).
