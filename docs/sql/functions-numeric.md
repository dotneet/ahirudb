# Numeric functions

[← Back to index](README.md)

> Every `SELECT` below is written as a bare expression for brevity. ahirudb
> requires a `FROM` clause on every `SELECT` (see
> [queries.md](queries.md#overall-shape)) — run any of these directly by
> appending `FROM range(1)`, e.g. `SELECT abs(-5) FROM range(1);`.

```sql
SELECT abs(-5);          -- 5
SELECT @(-5);             -- 5      (prefix operator, sugar for abs(x))
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
| `abs(x)` / `@x` (prefix) | Integer overflow case (`abs(i64::MIN)`) returns `NULL` rather than overflowing |
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

## Division and integer division

```sql
SELECT 7 / 2;      -- 3      (both operands integer -> truncates toward zero)
SELECT -7 / 2;     -- -3
SELECT 5.0 / 2;    -- 2.5    (either operand a float -> real-valued division)
SELECT 5 / 0;      -- NULL  (not an error)
SELECT 7 // 2;     -- 3      (// is sugar for /, see below)
SELECT 5 // 0;     -- NULL
```

**ahirudb's plain `/` is already truncating integer division when both
operands are integers** — a divergence from DuckDB, whose `/` always
returns a float (`duckdb`'s `7/2` is `3.5`, not `3`). This predates the
`//` operator and is listed in [limitations.md](limitations.md#partially-supported);
whether `/` should change to match DuckDB is an open question, not a
settled decision.

`//` is DuckDB's own truncating-integer-division operator, and it happens
to be exact sugar for `/` in ahirudb specifically *because* `/` already
behaves that way here: `//` binds at the same precedence as `*`/`/` (so `2
+ 5 // 2` is `2 + (5 // 2)` = `4`) and is left-associative (`5 // 2 // 2` =
`(5 // 2) // 2` = `1`). Division by zero returns `NULL` either way, and if
`/`'s semantics ever change to match DuckDB's, `//` would need to be
revisited to keep its own (DuckDB-matching) truncating behavior.

## Factorial

```sql
SELECT 4!;             -- 24        (postfix operator, sugar for factorial(x))
SELECT factorial(4);   -- 24
SELECT 0!;              -- 1
SELECT (2 + 2)!;        -- 24
SELECT -4!;             -- 1         (`!` applies to the whole `-4`, i.e. (-4)!)
SELECT factorial(-1);   -- 1         (negative n is defined as 1, not an error)
SELECT factorial(33);   -- 8683317618811886495518194401280000000  (largest that fits)
SELECT factorial(34);   -- error: value out of range
SELECT 4!::VARCHAR;     -- '24'
SELECT 2 + 3!;           -- 8         (not 120 -- see the precedence note below)
```

`factorial`/`!` returns `HUGEINT` (128-bit) to match DuckDB. `33!` is the
largest factorial that fits in a `HUGEINT` (`i128::MAX` ≈ `1.7e38`, `33!` ≈
`8.68e36`, `34!` ≈ `2.95e38`); `factorial(34)` and above raise an error
rather than silently wrapping or truncating, the same treatment `SUM`
overflow gets (see [types.md](types.md#rounding-and-floating-point-conventions)).
Only integer input is accepted — a `DOUBLE` argument (`4.5!`,
`factorial(4.0)`) is a type error, matching DuckDB.

**Precedence:** `!` binds looser than the prefix operators (`-`, `~`,
`NOT`) but tighter than every binary operator. The first part is what
makes `-4!` mean `(-4)!` = `1`, not `-(4!)` = `-24` — this holds for any
operand, not just literals (`-x!` for a column `x` behaves the same way).
The second part means a `!` always applies to just its immediately
preceding operand: `2 + 3!` is `2 + (3!)` = `8`, and `x!::VARCHAR` casts
the factorial result (`CAST(x! AS VARCHAR)`), not `x` before it's
factorialized.

This is a deliberate divergence from DuckDB for expressions mixing `!`
with binary operators. DuckDB's own grammar for postfix `!` is internally
inconsistent Postgres legacy — `3! ^ 2` parses (`36.0`) but `2 ^ 3!` is a
syntax error there, and `2 + 3!` silently reads as `(2+3)!` = `120` while
`3! + 1` is rejected outright — so there's no single coherent rule to
match. See [limitations.md](limitations.md#partially-supported) for the
full comparison table.

Note that `!` is *not* a general prefix logical-NOT operator in ahirudb
(there is none — use the `NOT` keyword); a bare `!` only appears postfix,
or as the leading byte of `!=`/`!~`/`!~~`/`!~~*`.

## Bitwise (BIGINT only)

```sql
SELECT 5 & 3;       -- 1   (bit_and)
SELECT 5 | 2;       -- 7   (bit_or)
SELECT 1 << 4;      -- 16  (bit_shift_left)
SELECT 16 >> 2;     -- 4   (bit_shift_right)
SELECT ~5;          -- -6  (bit_not, prefix)
```

The operators are sugar over the named functions (`bit_and(a,b)`,
`bit_or(a,b)`, `bit_shift_left(a,b)`, `bit_shift_right(a,b)`, `bit_not(a)`),
which can also be called directly. All operate on `BIGINT` (64-bit); other
numeric input is cast to `BIGINT` first, matching this engine's usual
"collapse to one working width" simplification for math functions (see the
`log`/`sqrt` notes above). A shift amount that's negative or ≥ 64 returns
`NULL` rather than erroring (DuckDB raises an error there instead — the
same "prefer NULL over erroring mid-scan" divergence as `sqrt` above).

Operator precedence: `&`/`|`/`<<`/`>>` bind tighter than comparison
operators but looser than `+`/`-` (so `1 + 2 & 3` is `(1 + 2) & 3`, and
`1 & 2 = 0` is `(1 & 2) = 0`). Prefix `~` and infix `~`/`!~` share the same
token but never conflict — `~x` (nothing before it) is always bitwise NOT;
`x ~ y`/`x !~ y` (an operand before it) is always the regex-match operator
documented in [queries.md](queries.md#where-operators-and-predicates).
