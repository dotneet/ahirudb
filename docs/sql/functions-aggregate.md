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
SELECT any_value(name), last(name) FROM t;
SELECT bool_and(flag), bool_or(flag), count_if(flag) FROM t;
SELECT product(score), stddev_pop(score), var_pop(score) FROM t;
SELECT quantile_cont(score, 0.9) FROM t;          -- interpolated 90th percentile
SELECT arg_max(name, score), arg_min(name, score) FROM t;
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
| `stddev_pop(x)` | — | Population standard deviation; defined from one row (the sample version needs two) |
| `var_pop(x)` | — | Population variance; same one-row note |
| `quantile_cont(x, frac)` | `quantile`, `percentile_cont` | Interpolated quantile; `frac` must be a constant literal in `[0, 1]`. DuckDB's `quantile` is the *discrete* version — here all three spellings are continuous, i.e. `quantile(x, 0.5)` equals `median(x)` |
| `string_agg(x, sep)` | `group_concat(x, sep)`, `listagg` | `sep` must be a constant literal; defaults to `','` if omitted |
| `array_agg(x)` | `list(x)` | Collects values into a `JSON`-array-shaped result (no separate LIST physical type — see [functions-json.md](functions-json.md)). Elements are rendered from their **logical** type, exactly as `to_json` renders them: `DECIMAL` keeps its decimal point, `DOUBLE` uses the `CAST(x AS VARCHAR)` spelling, and `DATE`/`TIME`/`TIMESTAMP`/`INTERVAL`/`UUID` come out as quoted text |
| `any_value(x)` | `first(x)`, `arbitrary(x)` | First non-`NULL` value seen. Input order is not guaranteed without `ORDER BY`, so treat it as "some value" |
| `last(x)` | — | Last non-`NULL` value seen; same ordering caveat |
| `bool_and(x)` / `bool_or(x)` | — | `BOOLEAN` input; `NULL`s are skipped, so an all-`NULL` group is `NULL` |
| `count_if(x)` | `countif(x)` | Counts rows where `x` is true; a group with none counts `0`, not `NULL` |
| `product(x)` | — | Accumulated in `DOUBLE` (an exact integer product would overflow immediately) |
| `arg_max(v, k)` / `arg_min(v, k)` | `max_by`, `min_by` | `v` at the row with the largest/smallest `k`. Rows where **either** argument is `NULL` take no part. `DISTINCT` is rejected here (it would have to deduplicate on the pair) |

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
SELECT id, ntile(4) OVER (ORDER BY score) FROM t LIMIT 3;
SELECT id, percent_rank() OVER (ORDER BY score) FROM t LIMIT 3;
SELECT id, cume_dist() OVER (ORDER BY score) FROM t LIMIT 3;
SELECT id, nth_value(score, 2) OVER (ORDER BY id) FROM t LIMIT 3;
```

| Function | Notes |
|---|---|
| `row_number()` | No arguments; 1-based position within the partition |
| `rank()` | No arguments; ties share a rank, next rank skips |
| `dense_rank()` | No arguments; ties share a rank, next rank doesn't skip |
| `percent_rank()` | No arguments; `(rank - 1) / (rows - 1)`, so it spans 0..1. A single-row partition is `0` |
| `cume_dist()` | No arguments; the fraction of the partition at or before this row's peer group |
| `ntile(n)` | Splits the partition into `n` buckets, the first `rows % n` of them one row larger. `n < 1` gives `NULL` (DuckDB errors) |
| `lag(x[, offset[, default]])` | Up to 3 arguments; `offset` defaults to 1, `default` defaults to `NULL` and is cast to `x`'s type |
| `lead(x[, offset[, default]])` | Same shape as `lag`, looking forward instead of back |
| `first_value(x)` | First value in the current window frame |
| `last_value(x)` | Last value in the current window frame |
| `nth_value(x, n)` | The `n`-th row (1-based) of the current frame, or `NULL` if the frame has not reached it yet |

Aggregates that only ever *add* to the frame can also run as window
functions: `sum`, `count`, `count(*)`, `avg`, `min`, `max`, `any_value`,
`first`, `last`, `bool_and`, `bool_or`, `count_if`, and `product`.

Aggregates that would have to **remove** values from the frame as it
advances (`median`, `quantile_cont`, `mode`, `stddev`, `variance`,
`stddev_pop`, `var_pop`, `string_agg`, `array_agg`, `arg_min`, `arg_max`,
`approx_count_distinct`) have no window version and are rejected with
`unsupported SQL feature`.

Window buffering has the same fixed in-memory cap as other blocking
operators — see [limitations.md](limitations.md#no-spilling).
