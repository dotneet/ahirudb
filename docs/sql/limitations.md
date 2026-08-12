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
- **Window function frames** are always the standard default (`RANGE
  UNBOUNDED PRECEDING` through the current row if the window has an `ORDER
  BY`, the whole partition if it doesn't), chosen automatically — there is
  no support for an explicit `ROWS`/`RANGE BETWEEN ...` frame. It's rejected
  at parse time rather than silently substituting the default, since that
  would change the result. See [queries.md](queries.md#window-functions).
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
  AS VARCHAR) AS INTEGER) + 1)`). It also changes what `||` means — see
  [JSON is also the list type](#json-is-also-the-list-type) below.
- **JSON equality is byte-comparison**, not semantic comparison — two JSON
  documents that differ only in whitespace (`'{"a": 1}'` vs `'{"a":1}'`)
  compare unequal. Only `=`/`<>` are defined on `JSON`; ordering comparisons
  (`<`, `>`, ...) are a type error.
- **`TIMESTAMPTZ` has no session timezone concept.** A `CAST(... AS
  TIMESTAMPTZ)` literal with no explicit offset (`+HH`, `+HH:MM`, or `Z`) is
  assumed to already be UTC, rather than being interpreted in a configured
  session timezone the way DuckDB does. See
  [types.md](types.md#timestamptz).
- **`CAST(... AS UUID)` of malformed text becomes `NULL`, not an error** —
  matching this engine's existing convention for `DATE`/`TIME`/`TIMESTAMP`
  parse failures, but different from DuckDB, which raises a hard error on
  plain `CAST` (only `TRY_CAST` returns `NULL` there). See
  [types.md](types.md#uuid).
- **Integer `/` truncates instead of returning a float.** `SELECT 7 / 2`
  is `3` here and `3.5` in DuckDB; `5.0 / 2` is `2.5` in both. This is a
  long-standing divergence that predates the `//` operator (which DuckDB
  defines as truncating integer division, and which is therefore exact
  sugar for `/` here). Whether `/` should change to match DuckDB is an
  open question. See
  [functions-numeric.md](functions-numeric.md#division-and-integer-division).
- **Postfix `!` (factorial) binds tighter than every binary operator
  here**, so `2 + 3!` is `2 + (3!)` = `8`, and `3! + 1` (`(3!) + 1` = `7`)
  parses at all. DuckDB's own precedence for `!` is internally
  inconsistent Postgres legacy, not a coherent rule worth replicating —
  it parses `3! ^ 2` fine (`36.0`) but rejects `2 ^ 3!` as a syntax error,
  and silently reads `2 + 3!` as `(2+3)!` = `120` while rejecting
  `3! + 1` outright. We chose the conventional, self-consistent reading
  on purpose instead: `!` binds looser than the prefix operators (so
  `-x!` is `(-x)!` for any `x`, matching DuckDB) but tighter than every
  binary operator. See [functions-numeric.md](functions-numeric.md#factorial).

## JSON is also the list type

DuckDB has a statically-typed `LIST` and a separate `JSON` type. ahirudb has
only `JSON` (see [DESIGN.md §5/§8](../DESIGN.md) — six physical types is a
load-bearing size constraint), so `[1, 2]` and `CAST('[1,2]' AS JSON)` are
literally the same value of the same type. Three consequences, all of them
around `||`:

- **`||` between two `JSON` operands concatenates them as lists**, matching
  `SELECT [1,2] || [3]` → `[1, 2, 3]` in DuckDB. **An operand that isn't a
  JSON array raises a `TypeMismatch` error at run time.** This is where it
  diverges: DuckDB's `JSON` isn't a list, so `'{"a":1}'::JSON ||
  '{"b":2}'::JSON` there is VARCHAR text concatenation (`{"a":1}{"b":2}`).
  Nothing in the type can distinguish the two cases here, so one behavior
  has to win — and an error is the only one that doesn't silently return a
  wrong answer. (Returning `NULL` was considered and rejected: it would be
  harder to notice than the invalid-JSON string this whole rule replaced.)
  **To concatenate two JSON documents as text, cast out of `JSON` first:**
  `CAST(a AS VARCHAR) || CAST(b AS VARCHAR)`. Mixed operands
  (`[1] || '{"a":1}'::JSON`) raise too, in either order, and so does a
  non-array paired with `NULL` — the error takes priority over `NULL`
  propagation so the result never depends on operand order. `[1] || NULL`
  is still `NULL`. Note that the `list_concat` **function** is *not* strict
  this way: it keeps returning `NULL` for a non-array operand, like every
  other `list_*` function (`list_extract`, `list_slice`,
  `list_transform`).
- **`||` with `JSON` on one side only stays VARCHAR concatenation.** `SELECT
  [1] || 2` is `'[1]2'` here; DuckDB rejects it ("Cannot concatenate types
  INTEGER[] and INTEGER"). It can afford to, because it can tell a list from
  a JSON document, and `json_col || 'suffix'` is legal DuckDB — rejecting it
  here would reject that too.
- **Mixed element types are allowed.** `SELECT ['a'] || [1]` is
  `'["a",1]'`; DuckDB rejects it ("Cannot concatenate lists of types
  VARCHAR[] and INTEGER[]") because its lists are homogeneously typed. JSON
  arrays aren't.

See [functions-json.md](functions-json.md#concatenating-lists) for the
`list_concat` function, which is defined on the same values but handles
`NULL` differently (as DuckDB also does).

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
rather than erroring (except `SUM` and `factorial`/`!`, see
[functions-numeric.md](functions-numeric.md#factorial)), and division by
zero returns `NULL` rather than raising an error.
