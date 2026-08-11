# Numeric functions

[← Back to index](README.md)

> Every `SELECT` below is written as a bare expression for brevity. ahirudb
> requires a `FROM` clause on every `SELECT` (see
> [queries.md](queries.md#overall-shape)) — run any of these directly by
> appending `FROM range(1)`, e.g. `SELECT abs(-5) FROM range(1);`.

```sql
SELECT abs(-5);          -- 5
SELECT sign(-3.0);        -- -1
SELECT ceil(1.2);         -- 2      (alias: ceiling)
SELECT floor(1.8);        -- 1
SELECT trunc(1.9);        -- 1      (truncate toward zero)
SELECT round(3.14159, 2); -- 3.14
SELECT round(-1234, -2);  -- -1200  (negative digits round to a power of ten)
SELECT sqrt(16);          -- 4
SELECT mod(10, 3);        -- 1
SELECT pow(2, 10);        -- 1024   (alias: power)
SELECT exp(1);            -- 2.718281828459045
SELECT ln(2.718281828);   -- ~1
SELECT log10(100);        -- 2
SELECT log(1000);         -- ~3     (log is single-argument, always base-10; hand-rolled,
                          --         so expect ordinary floating-point noise, e.g. 2.9999999999999996)
```

| Function | Notes |
|---|---|
| `abs(x)` | Integer overflow case (`abs(i64::MIN)`) returns `NULL` rather than overflowing |
| `sign(x)` | Returns -1/0/1; on floats, preserves signed-zero/NaN pass-through |
| `ceil(x)` / `ceiling(x)`, `floor(x)`, `trunc(x)` | No-op (identity) on integer input; real work only happens on float input |
| `round(x[, d])` | `d` > 0 rounds to `d` decimal places (half-away-from-zero); `d` < 0 rounds to a power of ten; `d` on an integer input with `d ≥ 0` is a no-op |
| `mod(a, b)` | Integer: `b = 0` or `MIN_VALUE % -1` returns `NULL` (no error/panic). Float: plain `%` |
| `sqrt(x)` | Negative input returns `NULL` (DuckDB errors instead — an intentional divergence, matching this engine's general "prefer NULL over erroring mid-scan" policy) |
| `exp(x)` | — |
| `ln(x)` | `x ≤ 0` → `NULL` |
| `log10(x)` / `log(x)` | Base-10; `x ≤ 0` → `NULL`. **The two-argument `log(base, x)` form is not implemented** |
| `pow(x, y)` / `power(x, y)` | — |

All math kernels (`sqrt`, `ln`, `exp`, ...) are hand-rolled (Newton's
method / series expansions) rather than delegating to a system math
library, since the engine is `no_std` and has no libm available in the
wasm build.

## NULL-aware / multi-value helpers

```sql
SELECT greatest(1, 5, 3);        -- 5   (NULL args are skipped; all-NULL -> NULL)
SELECT least(1, 5, 3);           -- 1
SELECT coalesce(NULL, NULL, 7);  -- 7   (first non-NULL argument)
SELECT ifnull(NULL, 7);          -- 7   (exactly 2 args; thin wrapper over coalesce)
SELECT nullif(1, 1);             -- NULL (equal -> NULL)
SELECT nullif(1, NULL);          -- 1    (comparison itself is NULL/unknown, not TRUE -> returns 'a' unchanged)
```

`nullif(a, b)` only returns `NULL` when `a = b` evaluates to exactly
`TRUE`; if either side is `NULL` (making the comparison `UNKNOWN`), it
returns `a` as-is rather than `NULL` — a subtle case worth knowing about if
you're using `nullif` to catch `NULL` inputs specifically (it won't).

See [types.md](types.md#rounding-and-floating-point-conventions) for the
project-wide rounding and overflow conventions (float→integer cast
rounding, `DECIMAL` scale reduction, integer overflow wraparound).
