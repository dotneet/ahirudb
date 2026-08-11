# Known limitations

[← Back to index](README.md)

ahirudb aims to fail loudly and explicitly rather than silently returning a
wrong answer. This page lists what it doesn't do — either because it's an
architectural trade-off (unlikely to change without a deliberate redesign)
or simply not built yet. See [DESIGN.md §15](../DESIGN.md) if you want the
implementation-level reasoning behind each one; this page just states the
user-visible effect.

## Not supported at all

- **`ASOF JOIN`**
- **General `LATERAL`** — only the implicit-lateral form of `UNNEST` in
  `FROM` is supported (`FROM t, UNNEST(t.xs) AS y(x)`); an arbitrary
  `LATERAL (subquery)` is not
- **`CREATE MACRO`**
- **Sequences and constraints** (`PRIMARY KEY`, `FOREIGN KEY`, `CHECK`)
- **Transactions** (`BEGIN`/`COMMIT`/`ROLLBACK`)
- **`ATTACH`** (attaching another database file)
- **Named parameters** — only positional `?` placeholders are supported
- **`UUID` as a first-class type** — no `CAST(... AS UUID)`; a Parquet
  column with the UUID logical annotation is read as raw text instead
- Hashing/fuzzy-match functions commonly found in DuckDB: `md5`, `sha256`,
  `levenshtein`, `jaro`, `soundex`, `uuid()`, `random()`
- `now()` / `current_timestamp` / `current_date` as ordinary **scalar
  functions** you could invoke inside, say, a user-defined function — the
  no-argument keyword forms (`CURRENT_DATE`, `CURRENT_TIMESTAMP`, `now()`,
  `today()`, `current_time`) work as documented in
  [functions-datetime.md](functions-datetime.md#current_date-and-now), but they
  are handled by substituting the query's start time in before binding, not
  by a general clock-reading scalar function

## Partially supported

- **`LIKE`/`ILIKE ... ESCAPE '<char>'`** parses but a custom escape
  character is not actually implemented (the query fails with
  `UnsupportedFeature` at prepare time) — use `LIKE`/`ILIKE` without an
  `ESCAPE` clause, or use `GLOB`/`SIMILAR TO`/`regexp_matches` instead if
  you need to match a literal `%`/`_`.
- **`PIVOT ... ON x`** requires an explicit `IN (...)` value list.
  DuckDB's auto-detect-distinct-values form (`PIVOT t ON x USING agg(y)`
  with no `IN`) is not supported — enumerate the pivot values yourself.
- **`PIVOT`** supports only a single `ON` expression and a single `USING`
  aggregate; DuckDB's multi-column `ON (a, b)` and multiple simultaneous
  `USING sum(a), avg(b)` aggregates aren't supported.
- **`UNPIVOT`** supports only single-column-at-a-time unpivoting; DuckDB's
  `UNPIVOT ... ON (a, b), (c, d)` (unpivoting several columns into several
  value columns at once) isn't supported.
- **Regular expressions**: the engine (a hand-written Thompson NFA, chosen
  to avoid backtracking blowups on adversarial input) does not support
  lookaround, backreferences *inside a pattern*, named capture groups,
  non-greedy quantifiers, `\b`/`\B` word boundaries, or a case-insensitive
  flag.
- **`log(x)`** is single-argument base-10 only; the two-argument
  `log(base, x)` form isn't implemented.
- **`date_add`** has the signature `date_add(part, n, timestamp)`, not
  DuckDB's `date_add(timestamp, INTERVAL ...)` — there's no scalar-function
  overload that takes an `INTERVAL` value directly (interval arithmetic via
  `+`/`-` operators, e.g. `date + INTERVAL 1 DAY`, works fine — see
  [types.md](types.md#interval-literals)).
- **LIST/MAP values have no dedicated physical type** — they're
  represented as `JSON` text under the hood (see
  [data-sources.md](data-sources.md#nested-parquet-types)). This is usually
  invisible, but it means, for example, that list elements need an explicit
  `CAST(... AS VARCHAR)` / `CAST(... AS INTEGER)` round-trip before doing
  arithmetic on them inside a lambda (`list_transform(xs, x -> CAST(CAST(x
  AS VARCHAR) AS INTEGER) + 1)`).
- **JSON equality is byte-comparison**, not semantic comparison — two JSON
  documents that differ only in whitespace (`'{"a": 1}'` vs `'{"a":1}'`)
  compare unequal. Only `=`/`<>` are defined on `JSON`; ordering comparisons
  (`<`, `>`, ...) are a type error.

## No spilling

Every blocking operator (hash aggregate, hash join, sort, `DISTINCT ON`,
sampling, recursive CTEs) runs entirely in memory with a fixed byte cap; it
returns a clean out-of-memory error rather than spilling to disk or
degrading silently. In practice this means a `GROUP BY`/join/sort over data
that doesn't fit the cap fails outright instead of running slowly — it
never produces a partial or wrong result.

## Performance-adjacent notes, not correctness bugs

- **Hash join build-side choice** relies on Parquet row-count metadata,
  which isn't always available for a nested source (a subquery or CTE
  feeding into a join). When it's unavailable, the engine isn't guaranteed
  to pick the smaller side as the build side — results are still correct,
  just potentially using more memory than the optimal choice would.
- **Dictionary-encoded Parquet columns are decoded to plain values before
  execution** — `GROUP BY`/equality on a dictionary-encoded string column
  runs against materialized values, not dictionary codes. This is a
  plausible future optimization that was never built, not a gap in
  correctness.
- **A `CSV` split boundary that lands inside a quoted newline** can be
  mis-resynchronized. This is a known trade-off shared by parallel CSV
  readers generally, and CSV in ahirudb currently reads as a single split,
  which avoids it in practice.
- **Low-selectivity `IN`-list pruning**: predicate pushdown for `WHERE x IN
  (...)` skips whole RowGroups/pages when the candidate values cluster
  together. If a list's values are scattered widely enough that nearly
  every RowGroup contains at least one of them, pruning can't skip
  anything, and the extra statistics lookup becomes a small amount of pure
  overhead rather than a win. This only affects wall-clock time, not
  correctness, and realistic `IN` lists (a batch of IDs from one customer,
  one time window) tend to cluster rather than scatter uniformly.

## Rounding conventions to be aware of

These match DuckDB's behavior deliberately, since the two diverge easily
and query results can be surprising if you're expecting different rules.
See [types.md](types.md#rounding-and-floating-point-conventions) for the
full list — briefly: float→integer casts round to nearest-even, `DECIMAL`
scale reduction rounds away from zero, integer arithmetic overflow wraps
rather than erroring (except `SUM`), and division by zero returns `NULL`
rather than raising an error.
