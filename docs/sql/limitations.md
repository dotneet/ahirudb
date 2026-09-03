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
  `hash`, `levenshtein`, `jaro`, `soundex`, `uuid()`, `random()`
- **Trigonometric functions** (`sin`/`cos`/`tan`/`asin`/`acos`/`atan`/
  `atan2` and the hyperbolic family). Every float kernel here is
  hand-rolled because the `no_std` wasm build has no libm (see
  [functions-numeric.md](functions-numeric.md)), and a full set of series
  expansions costs more code than the 1 MiB budget wants to spend on them.
- **Unicode-table functions**: `strip_accents`, `nfc_normalize`,
  `length_grapheme` and friends — the tables alone are tens of KB. This is
  the same reason `upper`/`lower` are ASCII-only.
- `strptime` (format-driven timestamp parsing). Use `to_timestamp`/
  `to_date`, which parse the fixed ISO-ish shapes documented in
  [functions-datetime.md](functions-datetime.md#truncating-formatting-parsing).
- `date_sub` — deliberately *not* aliased to `date_diff`, because DuckDB's
  `date_sub` counts complete partitions while `date_diff` counts boundaries
  crossed, and the two disagree over a partial unit.
- `now()` / `current_timestamp` / `current_date` as ordinary **scalar
  functions** you could invoke inside, say, a user-defined function — the
  no-argument keyword forms (`CURRENT_DATE`, `CURRENT_TIMESTAMP`, `now()`,
  `today()`, `current_time`) work as documented in
  [functions-datetime.md](functions-datetime.md#current_date-and-now), but they
  are handled by substituting the query's start time in before binding, not
  by a general clock-reading scalar function

## Partially supported

- **Window function frames** are always the standard default (`RANGE
  UNBOUNDED PRECEDING` through the current row if the window has an `ORDER
  BY`, the whole partition if it doesn't), chosen automatically — there is
  no support for an explicit `ROWS`/`RANGE BETWEEN ...` frame. It's rejected
  at parse time rather than silently substituting the default, since that
  would change the result. See [queries.md](queries.md#window-functions).
- **Typed string literals cover the four temporal types only** —
  `DATE '...'`, `TIME '...'`, `TIMESTAMP '...'`, `TIMESTAMPTZ '...'` work
  (see [types.md](types.md#typed-datetime-literals)), but DuckDB's general
  `<any type> '<text>'` form (`INTEGER '5'`) does not; write
  `CAST('5' AS INTEGER)` instead.
- **`GROUP BY ALL` does not accept `*` in the select list.** DuckDB expands
  the star and groups by every resulting column; here
  `SELECT * ... GROUP BY ALL` fails with `unsupported SQL feature`. List the
  columns explicitly, or use `SELECT DISTINCT *`. (`ORDER BY ALL` *does*
  work with `*` — it is resolved after the star has been expanded.)
- **`ORDER BY ALL` is not accepted on `PIVOT`/`UNPIVOT` statements** — those
  rebuild their output column list while being desugared, so the shorthand
  is rejected rather than silently resolved against the wrong columns. Use
  an explicit `ORDER BY` list there.
- **`PIVOT ... ON x`** requires an explicit `IN (...)` value list.
  DuckDB's auto-detect-distinct-values form (`PIVOT t ON x USING agg(y)`
  with no `IN`) is not supported — enumerate the pivot values yourself.
- **`PIVOT`** supports only a single `ON` expression and a single `USING`
  aggregate; DuckDB's multi-column `ON (a, b)` and multiple simultaneous
  `USING sum(a), avg(b)` aggregates aren't supported.
- **`UNPIVOT`** supports only single-column-at-a-time unpivoting; DuckDB's
  `UNPIVOT ... ON (a, b), (c, d)` (unpivoting several columns into several
  value columns at once) isn't supported.
- **Star expressions**: `COLUMNS(*)`, `COLUMNS('regex')`,
  `COLUMNS(['a','b'])` and the `AS '\1'` capture-group renaming form all
  work (see [queries.md](queries.md#columns)), but four DuckDB star-expression
  features are rejected with `UnsupportedFeature` rather than
  half-implemented: distributing an enclosing expression over the expansion
  (`min(COLUMNS(*))`, `COLUMNS(*) + 1`), `UNPACK(...)` / `*COLUMNS(...)`
  unpacking, the `COLUMNS(c -> ...)` lambda predicate form, and the
  `* LIKE 'col%'` / `* GLOB` / `* SIMILAR TO` star-filtering operators.
  A `COLUMNS(...)` item also can't be table-qualified (`t.COLUMNS(*)`) —
  neither can DuckDB's. Note the `COLUMNS('regex')` match is
  **case-sensitive**, unlike every other column-name comparison here; that
  matches DuckDB.
- **Regular expressions**: the engine (a hand-written Thompson NFA, chosen
  to avoid backtracking blowups on adversarial input) does not support
  lookaround, backreferences *inside a pattern*, named capture groups,
  non-greedy quantifiers, `\b`/`\B` word boundaries, or a case-insensitive
  flag (neither `(?i)` nor the `'i'` flag argument). Matching itself is
  per UTF-8 character — `.`, character classes, and quantifiers each
  consume one whole scalar value — and POSIX bracket expressions
  (`[[:alpha:]]`, `[[:digit:]]`, ...) are supported but match ASCII only.
  See [functions-string.md](functions-string.md#regular-expressions).
- **`quantile`/`percentile_cont`** are the *continuous* (interpolated)
  quantile in all spellings. DuckDB's `quantile` is the discrete version,
  and its `quantile_disc` isn't implemented. A list-valued fraction
  (`quantile_cont(x, [0.25, 0.75])`) isn't supported either — the fraction
  must be a single constant in `[0, 1]`.
- **`make_date`/`make_timestamp`** return `NULL` for an out-of-range
  component where DuckDB raises, and `make_timestamp` takes six integer
  arguments only (no `DOUBLE` seconds, no single-argument microseconds
  overload).
- **`ntile(n)` with `n < 1`** returns `NULL` rather than raising as DuckDB
  does — the same "prefer NULL over erroring mid-scan" policy as `sqrt(-1)`.
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
  on purpose instead: `!` binds tighter than every binary operator, and
  tighter than the prefix operators `~` and `@`. Unary `-` still binds
  tighter than `!` (so `-x!` is `(-x)!` for any `x`, matching DuckDB), but
  `~` and `@` do not:

  ```sql
  SELECT ~5!;        -- -121 here (~(5!) = ~120); DuckDB gives 1 ((~5)! = (-6)! = 1)
  SELECT @(-5)!;     -- 1 here (@((-5)!) = @1); DuckDB gives 120 ((@(-5))! = 5!)
  SELECT @5!;        -- 120 in both, but by different groupings
  ```

  This is a deliberate divergence. DuckDB's ladder puts `~`/`@` above `!`,
  which is what makes `~5!` collapse to `1`; ours puts every prefix
  operator except `-` below the postfix, which keeps `!` reading as an
  ordinary suffix on its operand. See
  [functions-numeric.md](functions-numeric.md#factorial).
- **Unsigned integer arithmetic wraps within its declared unsigned
  width** rather than raising. `SELECT 0::UTINYINT - 1` is `-1` here and
  `SELECT 0::UBIGINT - 1` is `18446744073709551615`; DuckDB raises an
  out-of-range error for both. This follows the same rule as signed
  overflow (see [Rounding conventions](#rounding-conventions-to-be-aware-of)
  below): arithmetic wraps, it doesn't error. The result stays inside the
  declared width — it can no longer leak a negative value out of a
  narrower unsigned column, which it used to.
- **`length`/`len` on a list measures its JSON text, not its elements.**
  `SELECT length([1,2,3])` is `7` (the length of `[1,2,3]`), because lists
  and JSON documents are one logical type here (see
  [JSON is also the list type](#json-is-also-the-list-type) below) and
  `length` is the string function. Use **`array_length`/`list_length`**,
  which return the element count (`3`).
- **`UPDATE`/`DELETE` do not accept a table alias or a subquery in
  `WHERE`.** `UPDATE t AS x SET ...` and `DELETE FROM t AS x` are syntax
  errors; `UPDATE t SET ... WHERE a IN (SELECT ...)` fails with
  `unsupported SQL feature`. Table-*qualified* columns do work
  (`UPDATE t SET b = 1 WHERE t.a = 1`) — it is only the alias binding and
  the correlated/uncorrelated subquery that are missing.
- **Text parts whose sniffed schemas disagree widen to `VARCHAR`; Parquet
  parts stay strict.** Registering `a.csv` (whose column `a` sniffs as
  `BIGINT`) together with `b.csv` (whose `a` holds text and sniffs as
  `VARCHAR`) gives one `VARCHAR` column rather than an error, because a
  sniffed type is a guess about the file, not a declaration by it. A part
  that saw *no value at all* for the column (a header-only file, an all-empty
  column) is not counted as a disagreement — it has nothing to say, so the
  other parts' type wins, as it does in DuckDB. Two Parquet parts whose
  column `a` really is declared as different physical types still fail with
  `TypeMismatch` — there the schema is authoritative, and a silent widening
  would be hiding a mistake rather than tolerating one. See
  [data-sources.md](data-sources.md#text-format-type-inference).
- **A value outside the inference sample that doesn't fit the inferred type
  is an error, not a `NULL`.** If a CSV column sniffs as an integer from
  its first 256 KiB and row 60,001 holds `notanumber`, the query fails with
  `invalid cast`. DuckDB re-sniffs and widens the column instead, so the
  same file counts fine there. Erroring is the deliberate choice: silently
  nulling the row destroyed data, and this engine would rather fail loudly.
  Cast the column explicitly (`CAST(... AS VARCHAR)` at the source, or a
  Parquet conversion) if the input really is mixed.
- **`VALUES` is only a source for `INSERT`.** `INSERT INTO t VALUES (...)`
  works; `(VALUES (1,2)) AS x(a,b)` in a `FROM` clause, and a top-level
  `VALUES` statement, are both syntax errors. Use a real table, a
  `range(n)`-anchored `SELECT`, or a `UNION ALL` chain instead.
- **Statement size caps.** Two separate limits keep a pathological statement
  from exhausting the stack (on wasm a stack overflow is an unrecoverable
  trap, so deep input is always turned into an error instead):
  - *Expression nesting* is capped at 64 levels. This counts genuine
    nesting — parentheses, function arguments, subqueries — not chain
    length: `a AND b AND ... `, `1+1+...`, `'a'||'a'||...` are left-deep but
    flat, and any number of terms is fine.
  - *Left-deep `JOIN` and set-operation chains* are capped at 64 links per
    statement, counted across the whole statement. `A UNION ALL B UNION ALL
    ...` with more than 64 branches, or a chain of more than 64 `JOIN`s, is
    rejected with `expression nesting too deep`. Unlike expressions, these
    are `Box` chains whose *drop* alone recurses once per link, so the cap
    stays low. Nest them differently (union in batches through a CTE) if you
    generate SQL that hits it.
- **`printf('%f', ...)` prints the value's full exact binary expansion**,
  as C's `printf` does. `printf('%f', 1e300)` therefore prints all 301
  integer digits of the `DOUBLE` nearest `1e300`, where DuckDB prints the
  shortest round-trip digits and pads the rest with zeros. Both are
  "correct"; neither is the other. Relatedly, `printf('%f', -0.0)` is
  `-0.000000` here (C's rendering) and `0.000000` in DuckDB.

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

## DML statement atomicity

`INSERT`/`UPDATE`/`DELETE` against an in-memory table
([ddl-dml.md](ddl-dml.md), DESIGN.md §16) are validate-then-apply: every
row's new values are computed and checked (type coercion, `NOT NULL`)
*before* any row in the table is mutated, and the table's rows are only ever
written once every row has passed. Internally, rows are processed in
batches of a fixed internal size for `UPDATE`/`DELETE`; that's purely an
implementation detail and isn't visible in this guarantee — a constraint
violation discovered in a later batch does not leave an earlier batch's
rows already mutated. Concretely:

- `INSERT` evaluates and NOT-NULL-checks every row of the new data first,
  then appends the whole batch to the table in one step.
- `UPDATE` evaluates every `SET` expression and NOT-NULL-checks every
  matched row across the whole statement first, then writes all the
  validated values in one step.
- `DELETE` computes the full "rows to keep" list first, then replaces the
  table's row list in one step.

So a statement that fails partway through — a `NOT NULL` violation, a type
error, or any other row-level check — leaves the target table completely
unchanged, for the whole statement, not just a "this batch" scope.

This is **not** general transactional rollback (ahirudb has no
`BEGIN`/`COMMIT`/`ROLLBACK`, see "Not supported at all" above) — it only
covers row-level constraint violations discovered while evaluating the
statement, which is what a `Result::Err` from validation can actually catch.
It does not, and structurally cannot, cover an abrupt allocator or process
failure partway through evaluating or applying a statement (for example, the
wasm heap running out while buffering the validated rows) — `alloc::Vec`
aborts on allocation failure rather than returning a recoverable error in
this engine's `no_std` build, so there's no `Result` to intercept and no way
to guarantee a defined table state afterward. In practice such a failure
takes down the whole session, not just the DML statement's target table, so
"the table might be left half-updated" isn't really the operative risk in
that scenario — the session itself doesn't survive it either.

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
- **A `CSV`/`TSV` file is always read as a single split once it looks like it
  uses RFC 4180 quoting.** A quoted field's embedded newline is not a record
  boundary, but a fixed-size split cut has no way to tell, from its own
  bytes alone, whether the byte at its boundary sits inside an open quote —
  the field carrying it can start arbitrarily far back in the file, well
  outside what one split step can see. Guessing wrong there used to surface
  as an extra `(NULL, NULL)`-style row (a real, fixed bug). Instead of
  guessing, `ahirudb` inspects the same leading sample already fetched for
  header/type inference (up to 256 KiB) for a `"` byte; if one is found, the
  whole file is read as one split, which sidesteps the ambiguity entirely.

  Be aware what that costs: a split's whole fetch range has to be resident
  before it can be read, so a single-split file is read with its entire data
  region in memory at once, not just the 8 MiB a chunk would need. Alongside
  the loss of split-level I/O parallelism, this means a large quoted CSV can
  exhaust memory where an unquoted file of the same size would stream fine —
  a deliberate trade of a rare wrong answer for a visible failure. Convert
  large quoted CSV inputs to Parquet, or strip the quoting, if you need to
  read them at a size that does not fit in memory. Unquoted CSV/TSV files
  (the common case for large files) are unaffected and still split normally
  (8 MiB chunks by default).

  This has one remaining remote-only case: if a file's first 256 KiB contains
  **no** `"` at all, but a quoted field appears only later, and the file is
  not fully resident at resolve time (HTTP Range / a custom `ByteSource`),
  a later split that contains a `"` is rejected with `unsupported SQL
  feature` rather than being parsed as unquoted CSV. In-memory files (the
  CLI, `register` of a buffer) scan the whole file at resolve time, so a
  late quote still forces a single split and is read correctly. Files that
  quote consistently from early on, or don't quote at all, are unaffected.

  **A CSV/TSV file containing a lone `\r` — one that is not the first half of
  a `\r\n` — is read as a single split for the same reason**, and with the
  same memory consequence. That covers both a CR-only (classic Mac) file and
  one that mixes terminators; a lone `\r` ends a record in either case, so
  the `\n`-based split-boundary resynchronization has no boundary it can
  trust. Pure-LF and pure-CRLF files, and a `\r` inside a quoted field, are
  unaffected and still split normally.
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

The one deliberate divergence is **float summation**. `SUM`/`AVG` over
`DOUBLE` use Neumaier compensated summation, so they recover the low-order
bits a plain running accumulator drops (`sum([1e100, 1.0, -1e100])` is
`1.0` here and `0.0` in DuckDB). The cost is that the compensated result is
the correctly rounded value of the *exact* sum — with ties going to even
when the exact sum lands exactly between two doubles, as it does for
`n = 3` and `n = 6` below — while DuckDB's accumulated error happens to
land on the neighboring double, which prints as the shorter literal:

```sql
SELECT sum(x) FROM (SELECT 0.1 AS x FROM range(3));  -- 0.30000000000000004 here, 0.3 in DuckDB
SELECT sum(x) FROM (SELECT 0.1 AS x FROM range(6));  -- 0.6000000000000001  here, 0.6 in DuckDB
SELECT sum(x) FROM (SELECT 0.1 AS x FROM range(7));  -- 0.7000000000000001  here, 0.7 in DuckDB
```

Both engines are within a ulp of the true sum; neither is wrong. What
matters here is that the value is now the same whichever path computes it:
the blocking aggregate and the window form (`sum(x) OVER ()`) use the same
compensated accumulator and agree on every case above, so a query's answer
does not change when a `GROUP BY` is rewritten as a window.
