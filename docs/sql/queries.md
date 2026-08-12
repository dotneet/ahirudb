# Queries

[← Back to index](README.md)

This page covers `SELECT` end to end: the overall clause shape, joins,
subqueries, CTEs, set operations, aggregation, window functions, and the
table-transforming statements (`PIVOT`/`UNPIVOT`, `SAMPLE`, `UNNEST`).
Scalar/aggregate functions themselves are documented separately — see the
[function reference pages](README.md#function-reference).

## Overall shape

```sql
SELECT [DISTINCT [ON (expr, ...)]] expr [AS alias], ...
FROM <table | parquet('url') | read_json[_auto]('url') | generate_series(...) | range(...) | (subquery)> [alias]
  [ [INNER|LEFT|RIGHT|FULL|CROSS] JOIN <rel> [ON <expr>] ]
  [, LATERAL? UNNEST(<expr>) ...]
  [TABLESAMPLE|USING SAMPLE <n>% | <n> ROWS | (bernoulli|system|reservoir)(...)]
[WHERE <expr>]
[GROUP BY <expr>, ... | ALL | GROUPING SETS (...) | ROLLUP (...) | CUBE (...)]
[HAVING <expr>]
[WINDOW name AS (...), ...]
[QUALIFY <expr>]
[ORDER BY <expr> [ASC|DESC] [NULLS FIRST|LAST], ... | ALL [ASC|DESC] [NULLS FIRST|LAST]]
[LIMIT n] [OFFSET n]
```

plus the query-level forms that wrap or combine a `SELECT`:

```sql
WITH [RECURSIVE] cte_name [(cols...)] AS (<query>), ... <query>
<query> UNION [ALL] | INTERSECT | EXCEPT <query>
PIVOT <rel> ON <expr> [IN (...)] [USING agg(expr)] [GROUP BY ...]
UNPIVOT <rel> ON (col, ...) INTO NAME n VALUE v
DESCRIBE <rel>   SHOW TABLES   EXPLAIN <query>
```

There is no bare `SELECT <expr>` without a `FROM` clause — a query with no
real table needs an anchor. `range(1)`/`generate_series(1)` work for this
(see [Table functions](#table-functions-generate_series--range) below):

```sql
SELECT 1 + 1 FROM range(1);
```

## WHERE, operators, and predicates

Standard comparison/boolean/arithmetic operators, plus:

```sql
-- IN / BETWEEN
SELECT id FROM t WHERE id IN (12345, 999999999, -1) ORDER BY id;
SELECT count(*) FROM t WHERE id BETWEEN 12000 AND 12010;

-- LIKE / ILIKE (case-insensitive LIKE); % and _ are the wildcards
SELECT * FROM t WHERE name LIKE 'name_0';
SELECT * FROM t WHERE name ILIKE 'NAME_0';

-- GLOB (shell-style wildcards: *, ?, [...], [!...])
SELECT DISTINCT name FROM t WHERE name GLOB 'name_[01]' ORDER BY name;

-- ~~ / !~~ / ~~* / !~~* / ~~~: punctuation aliases for LIKE / NOT LIKE /
-- ILIKE / NOT ILIKE / GLOB (see functions-string.md#pattern-matching for
-- the precedence quirk that sets these apart from the keyword forms).
SELECT * FROM t WHERE name ~~ 'name_0';

-- SIMILAR TO (SQL regex, anchored to the whole string)
SELECT DISTINCT name FROM t WHERE name SIMILAR TO 'name_[0-1]' ORDER BY name;

-- ~ / !~ (infix): same as SIMILAR TO, a shorter spelling for a full-string
-- regex match. Prefix ~ (no left-hand operand) is unrelated -- bitwise NOT.
SELECT DISTINCT name FROM t WHERE name ~ 'name_[0-1]' ORDER BY name;
SELECT DISTINCT name FROM t WHERE name !~ 'name_[0-1]' ORDER BY name;

-- IS [NOT] NULL, CASE, COALESCE, IIF
SELECT IIF(flag, 'yes', 'no') FROM t LIMIT 3;

-- ISNULL / NOTNULL: DuckDB's non-standard postfix aliases for IS [NOT]
-- NULL. Soft keywords, like GLOB/SIMILAR above -- a column actually named
-- isnull/notnull still works unquoted (SELECT isnull FROM t).
SELECT * FROM t WHERE name ISNULL;
SELECT * FROM t WHERE name NOTNULL;

-- IS [NOT] UNKNOWN: exactly IS [NOT] NULL. The SQL standard defines it on
-- booleans only, but any operand is accepted and simply null-tested
-- (matching DuckDB): 1 IS UNKNOWN is false. UNKNOWN is a soft keyword, so a
-- column named `unknown` still works unquoted.
SELECT NULL IS UNKNOWN, 1 IS UNKNOWN, NULL IS NOT UNKNOWN;  -- true, false, false

-- ^@ (prefix / starts-with): sugar for starts_with(a, b). See
-- functions-string.md#prefix-operator-starts-with for its precedence.
SELECT * FROM t WHERE name ^@ 'name_1';

-- IS [NOT] TRUE / IS [NOT] FALSE: unlike `= true`, these coerce a
-- non-boolean operand (CAST to BOOLEAN) and never return NULL -- NULL IS
-- TRUE is FALSE, not NULL, even though NULL = TRUE would be NULL.
SELECT 3 IS TRUE, NULL IS TRUE, NULL IS NOT TRUE;   -- true, false, true

-- IS [NOT] DISTINCT FROM: NULL-safe equality/inequality (never UNKNOWN,
-- always TRUE/FALSE -- NULL is treated as equal to NULL and unequal to
-- anything else)
SELECT a, b FROM (VALUES (1,1), (1,2), (1,NULL), (NULL,NULL)) x(a,b)
  WHERE a IS DISTINCT FROM b;

-- :: cast shorthand for CAST(... AS ...). Binds tighter than unary
-- operators: -1::VARCHAR is -(1::VARCHAR), i.e. '-1', not (-1)::VARCHAR
-- misread as negating text.
SELECT '42'::INTEGER, (1 + 2)::VARCHAR;

-- ^ / ** (power, always returns DOUBLE; left-associative: 2^3^2 = 64, not
-- 512), and the bitwise operators & | << >> and prefix ~ (integer only)
SELECT 2 ^ 10, 2 ** 10;
SELECT 5 & 3, 5 | 2, 1 << 4, 16 >> 2, ~5;

-- // (integer division, see functions-numeric.md#division-and-integer-division),
-- @ (absolute value, prefix), ! (factorial, postfix, returns HUGEINT --
-- see functions-numeric.md#factorial)
SELECT 7 // 2, @(-5), 4!;
```

`IN`/`BETWEEN` predicates against literal values are also what drives
RowGroup/page/Bloom-filter pruning on Parquet scans — see
[data-sources.md](data-sources.md#parquet-coverage).

A shift by a negative amount or by more than 63 bits returns `NULL` rather
than erroring (DuckDB itself raises an error there) — consistent with how
this engine already treats other undefined integer arithmetic (division by
zero, etc.; see [types.md](types.md#null-and-three-valued-logic)).

## Joins

```sql
SELECT a.k, b.w FROM t AS a
  LEFT JOIN t2 AS b ON a.k = b.k
  ORDER BY a.k;
```

`INNER`, `LEFT`, `RIGHT`, `FULL`, and `CROSS` are all supported, including
non-equi join conditions (`ON a.x < b.y`).

## Subqueries (scalar, EXISTS, IN, correlated)

The examples below assume two tables, `customers(id, name, region)` and
`orders(id, customer_id, amount, region)` — any table works the same way,
these are just illustrative names.

```sql
-- scalar subquery, correlated on the outer row
SELECT c.id, c.name, (SELECT max(o.amount) FROM orders o WHERE o.customer_id = c.id) AS max_order
FROM customers c ORDER BY c.id;

-- correlated scalar COUNT: 0, not NULL, when there are no matching rows
SELECT c.id, c.name, (SELECT count(*) FROM orders o WHERE o.customer_id = c.id) AS n
FROM customers c ORDER BY c.id;

-- correlated EXISTS
SELECT c.id, c.name FROM customers c
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id AND o.amount > 60)
ORDER BY c.id;

-- correlated NOT EXISTS
SELECT c.id, c.name FROM customers c
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id)
ORDER BY c.id;

-- correlated IN
SELECT c.id, c.name FROM customers c
WHERE c.id IN (SELECT o.customer_id FROM orders o WHERE o.region = c.region)
ORDER BY c.id;
```

`NOT IN (subquery)` follows standard SQL's NULL-aware semantics: if the
subquery's result set contains even one `NULL`, the whole `NOT IN` becomes
`UNKNOWN` (so the outer row is filtered out) rather than silently ignoring
that `NULL`.

## CTEs (WITH, WITH RECURSIVE)

```sql
WITH regional_totals AS (
  SELECT region, sum(amount) AS total FROM orders GROUP BY region
)
SELECT * FROM regional_totals WHERE total > 100;
```

Recursive CTEs (`WITH RECURSIVE`) work through a non-recursive "anchor"
member `UNION ALL`'d (or `UNION`'d, for dedup-until-fixed-point) with a
recursive member that references the CTE itself. Since the anchor is
usually a literal starting value rather than a real table, it needs a
`FROM range(1)` (or similar) to satisfy the "every `SELECT` needs a `FROM`"
rule from above:

```sql
-- Fibonacci sequence
WITH RECURSIVE fib(n, a, b) AS (
  SELECT 0, 0, 1 FROM range(1)
  UNION ALL
  SELECT n + 1, b, a + b FROM fib WHERE n < 10
)
SELECT * FROM fib;

-- hierarchy traversal (self-join against a base table -- no synthetic
-- anchor needed here, since the anchor member reads from a real table)
WITH RECURSIVE tree AS (
  SELECT id, parent_id, name FROM nodes WHERE parent_id IS NULL
  UNION ALL
  SELECT n.id, n.parent_id, n.name FROM nodes n JOIN tree t ON n.parent_id = t.id
)
SELECT * FROM tree ORDER BY id;

-- UNION (not UNION ALL) de-duplicates each step, so it terminates at a
-- fixed point instead of running until the WHERE clause cuts it off
WITH RECURSIVE t(n) AS (
  SELECT 1 FROM range(1)
  UNION
  SELECT (n % 3) + 1 FROM t
)
SELECT * FROM t ORDER BY n;
```

Multiple independent recursive CTEs can appear in the same `WITH`, and can
reference each other's *finished* results:

```sql
WITH RECURSIVE
  a(n) AS (SELECT 1 FROM range(1) UNION ALL SELECT n + 1 FROM a WHERE n < 3),
  b(n) AS (SELECT 100 FROM range(1) UNION ALL SELECT n + 1 FROM b WHERE n < 103)
SELECT a.n, b.n FROM a, b WHERE a.n = b.n - 99 ORDER BY a.n;
```

The recursive working set and its deduplication ("seen") set each have a
fixed in-memory cap — see [limitations.md](limitations.md#no-spilling).

## UNION / INTERSECT / EXCEPT

```sql
SELECT id FROM a UNION SELECT id FROM b;        -- dedups
SELECT id FROM a UNION ALL SELECT id FROM b;    -- keeps duplicates
SELECT id FROM a INTERSECT SELECT id FROM b;
SELECT id FROM a EXCEPT SELECT id FROM b;
```

`EXCEPT` is not associative — `(a EXCEPT b) EXCEPT c` differs from `a
EXCEPT (b EXCEPT c)` — so a chain of set operators is always evaluated
left-to-right, matching standard SQL and DuckDB.

## Aggregation: GROUP BY, HAVING, GROUPING SETS/ROLLUP/CUBE

```sql
SELECT flag, count(*) c FROM t GROUP BY flag ORDER BY flag;

-- FILTER (WHERE ...) restricts one aggregate's input without a subquery
SELECT count(*) FILTER (WHERE flag) AS n_true, count(*) FILTER (WHERE NOT flag) AS n_false FROM t;

-- COUNT(DISTINCT ...)
SELECT count(DISTINCT name) FROM t;

-- DISTINCT ON: one row per key, keeping the first row per ORDER BY
SELECT DISTINCT ON (region) region, amount FROM orders ORDER BY region, amount DESC;
```

`GROUPING SETS`/`ROLLUP`/`CUBE` compute several grouping granularities in
one pass, unioning the results together (rows from a coarser grouping have
`NULL` in the columns that grouping doesn't group by):

```sql
-- two granularities at once: grouped by flag, and the grand total
SELECT flag, count(*) c, sum(id) s
FROM t GROUP BY GROUPING SETS ((flag), ())
ORDER BY flag;

-- ROLLUP (a, b) = GROUPING SETS ((a,b), (a), ()) -- hierarchical subtotals
SELECT flag, id % 3 AS m, count(*) c FROM t GROUP BY ROLLUP (flag, id % 3) ORDER BY 1, 2;

-- CUBE (a, b) = GROUPING SETS ((a,b), (a), (b), ()) -- every combination
SELECT flag, id % 3 AS m, count(*) c FROM t GROUP BY CUBE (flag, id % 3) ORDER BY 1, 2;

-- GROUPING()/GROUPING_ID() tell you which grouping-set produced a row
-- (1 = that column was rolled up to NULL for this row, 0 = it wasn't)
SELECT flag, id % 3 AS m, count(*) c,
       grouping(flag) gf, grouping(id % 3) gm, grouping(flag, id % 3) gid
FROM t GROUP BY CUBE (flag, id % 3) ORDER BY 1, 2;

-- HAVING can reference GROUPING() to pick out just one granularity
SELECT flag, id % 3 AS m, count(*) c
FROM t GROUP BY GROUPING SETS ((flag, id % 3), (flag), ())
HAVING grouping(flag) = 0 ORDER BY 1, 2;
```

### GROUP BY ALL

`GROUP BY ALL` (a DuckDB shorthand) groups by **every select-list
expression that doesn't contain an aggregate**, so you don't have to repeat
the non-aggregated columns:

```sql
SELECT flag, name, count(*) c FROM t GROUP BY ALL ORDER BY ALL;
-- identical to: ... GROUP BY flag, name ORDER BY flag, name, c
```

- "Contains an aggregate" is about the whole expression, not just its top
  level: in `SELECT id % 3, sum(id) + 1 FROM t GROUP BY ALL`, the grouping
  key is `id % 3` alone — `sum(id) + 1` is excluded.
- With no aggregate anywhere in the select list it behaves like
  `SELECT DISTINCT`.
- With nothing *but* aggregates there are no grouping columns, i.e. one
  row for the whole input.
- `ALL` can't be mixed with an explicit list (`GROUP BY ALL, x` is a syntax
  error, as in DuckDB).
- `SELECT * ... GROUP BY ALL` is **not** supported here (DuckDB expands the
  star and groups by every column); it fails with `unsupported SQL
  feature`. Spell the columns out, or use `SELECT DISTINCT *`.

### ORDER BY ALL

`ORDER BY ALL` sorts by every output column, left to right, all in the same
direction:

```sql
SELECT id, name FROM t ORDER BY ALL;              -- = ORDER BY id, name
SELECT id, name FROM t ORDER BY ALL DESC;         -- = ORDER BY id DESC, name DESC
SELECT big FROM t ORDER BY ALL NULLS FIRST;
SELECT * FROM t ORDER BY ALL;                     -- covers the whole star expansion
```

It applies to the final output columns, so aggregate result columns are
included too (`SELECT name, count(*) FROM t GROUP BY ALL ORDER BY ALL`
sorts by `name`, then by the count). Like `GROUP BY ALL`, it can't be
combined with an explicit list. It also works after a set operation
(`... UNION ALL ... ORDER BY ALL`), but is not accepted on
`PIVOT`/`UNPIVOT` statements.

See [functions-aggregate.md](functions-aggregate.md) for the full
aggregate-function list (`sum`, `avg`, `stddev`, `median`, `string_agg`,
...).

## Window functions

```sql
-- inline OVER
SELECT id, sum(score) OVER (PARTITION BY flag ORDER BY id) AS running_total
FROM t WHERE id < 6 ORDER BY id;

-- named window, shared by several calls
SELECT id, flag,
       sum(score) OVER w AS s,
       avg(score) OVER w AS a
FROM t WHERE id < 6
WINDOW w AS (PARTITION BY flag ORDER BY id)
ORDER BY id;

-- named + inline mixed in the same query
SELECT id, flag,
       row_number() OVER w AS rn,
       count(*) OVER () AS total
FROM t WHERE id < 6
WINDOW w AS (PARTITION BY flag ORDER BY id)
ORDER BY id;

-- multiple named windows
SELECT id, rank() OVER w1 AS r1, row_number() OVER w2 AS r2
FROM t WHERE id < 4
WINDOW w1 AS (ORDER BY id), w2 AS (PARTITION BY flag ORDER BY id)
ORDER BY id;
```

Dedicated window functions: `row_number()`, `rank()`, `dense_rank()`,
`lag(x[, offset[, default]])`, `lead(x[, offset[, default]])`,
`first_value(x)`, `last_value(x)`. Any aggregate function (`sum`, `avg`,
`count`, `min`, `max`, `stddev`, ...) can also be used as a window function
via `agg(...) OVER (...)`.

The frame is always the standard default, chosen automatically from whether
`ORDER BY` is present: `RANGE UNBOUNDED PRECEDING` (through the current row)
if it is, the whole partition if it isn't. **An explicit `ROWS`/`RANGE
BETWEEN ...` frame is not supported** — it's rejected at parse time rather
than silently substituting the default, since that would change the query's
result. If you need a specific frame, restructure the query with a subquery
or `LIMIT`/aggregation instead of relying on `OVER (... ROWS BETWEEN ...)`.

`QUALIFY` filters on the *result* of a window function without needing to
wrap the query in a subquery:

```sql
SELECT id, row_number() OVER (ORDER BY id) AS rn
FROM t
QUALIFY rn <= 3;
```

## SAMPLE / TABLESAMPLE

```sql
SELECT range AS x FROM range(20000) USING SAMPLE 10%;
SELECT range AS x FROM range(20000) TABLESAMPLE 10%;      -- same thing

SELECT range AS x FROM range(20000) USING SAMPLE 100 ROWS; -- fixed row count
SELECT range AS x FROM range(20000) USING SAMPLE 50;        -- bare number = rows

SELECT range AS x FROM range(20000) USING SAMPLE BERNOULLI(10%);
SELECT range AS x FROM range(20000) USING SAMPLE reservoir(30);

-- explicit seed for a reproducible sample
SELECT id FROM t USING SAMPLE 20% (bernoulli, 7);
```

`SAMPLE` always applies to the joined `FROM` result, before `WHERE`.
`bernoulli`/`system`/`reservoir` are accepted as method names, but (unlike
DuckDB) all three currently run the same underlying sampling algorithm —
the method name doesn't change behavior. Without an explicit seed, sampling
uses a fixed default seed (so it's deterministic, not random, by default).
Row-count sampling buffers candidates in memory with the same fixed byte
cap as other blocking operators — see
[limitations.md](limitations.md#no-spilling).

## PIVOT / UNPIVOT

```sql
-- PIVOT: turn distinct values of one column into new columns
PIVOT t ON category IN ('a', 'b', 'c') USING sum(amount) GROUP BY region ORDER BY region;

-- USING defaults to count(*) if omitted
PIVOT t ON category IN ('a', 'b', 'c') GROUP BY region ORDER BY region;

-- GROUP BY defaults to "every other column" if omitted
PIVOT t ON category IN ('a', 'b', 'c') USING sum(amount) ORDER BY region;

-- IN-list values can be aliased to control the output column name
PIVOT t ON category IN ('a' AS alpha, 'b' AS beta) USING sum(amount) GROUP BY region ORDER BY region;

-- ON accepts any expression, not just a bare column
PIVOT t ON id % 2 IN (0, 1) USING sum(amount) GROUP BY region ORDER BY region;
```

`PIVOT`'s `ON` clause always needs an explicit `IN (...)` list — see
[limitations.md](limitations.md#partially-supported) for why.

```sql
-- UNPIVOT: turn several columns into name/value row pairs
UNPIVOT t ON q1, q2, q3, q4 INTO NAME quarter VALUE amt ORDER BY id, quarter;

-- INTO NAME/VALUE default to "name"/"value" if omitted
UNPIVOT t ON amount ORDER BY region, category;
```

## UNNEST

`UNNEST` expands a `JSON`-array-valued expression (a Parquet `LIST` column,
or a `list_value(...)`/`json_array(...)` expression) into one row per
element. It can appear in the `SELECT` list, or as an implicit-lateral join
in `FROM`:

```sql
-- SELECT-list form: one output row per array element
SELECT id, UNNEST(xs) AS x FROM t WHERE id < 2;

-- FROM-clause form (implicitly LATERAL — no LATERAL keyword needed):
-- can reference columns from the table(s) that precede it
SELECT t.id, y.x FROM t, UNNEST(t.xs) AS y(x) WHERE t.id < 5;

-- chaining several UNNESTs cross-joins their elements (UNNEST can't be the
-- first FROM item on its own, so range(1) anchors it, same as above)
SELECT a.v, b.v FROM range(1), UNNEST(list_value(1, 2)) AS a(v), UNNEST(list_value(10, 20)) AS b(v);
```

If every element of the array is the same scalar type, `UNNEST` restores
that native type (`BIGINT`, `VARCHAR`, `BOOLEAN`, ...) rather than leaving
the result as `JSON` text; a mixed-type array stays `JSON`. A `NULL` or
empty array produces zero rows (not a row with a `NULL` value).

## Table functions: generate_series / range

```sql
SELECT * FROM range(5);              -- 0,1,2,3,4  (stop excluded)
SELECT * FROM range(0, 100, 5);      -- start, stop, step
SELECT * FROM range(10, 0, -2);      -- negative step

SELECT * FROM generate_series(5);        -- 0,1,2,3,4,5 (stop included)
SELECT * FROM generate_series(1, 10);
SELECT * FROM generate_series(0, 10, 2);

-- aliasing the generated column
SELECT x FROM range(3) AS t(x);
SELECT t.x FROM range(3) AS t(x) WHERE t.x > 0;
```

`range`'s `stop` is exclusive; `generate_series`'s `stop` is inclusive —
the same distinction DuckDB makes. Arguments must currently be literal
integers (no expressions or column references).

## SELECT * modifiers: EXCLUDE / REPLACE / RENAME

```sql
SELECT * EXCLUDE (score, big, d) FROM t WHERE id < 4 ORDER BY id;
SELECT * REPLACE (score * 2 AS score) FROM t WHERE id < 4 ORDER BY id;
SELECT * RENAME (score AS points) FROM t WHERE id < 4 ORDER BY id;
SELECT * EXCLUDE (name, big, d) REPLACE (score * 2 AS score) FROM t WHERE id < 4 ORDER BY id;
SELECT t.* EXCLUDE (name, big, d) FROM t WHERE id < 4 ORDER BY id;   -- qualified star works too
```

All three can combine on the same `*`, but the order is fixed:
`EXCLUDE` -> `REPLACE` -> `RENAME`. Writing them in any other order (e.g.
`RENAME (...) EXCLUDE (...)`) is a parser error.

`RENAME (old AS new, ...)` only relabels the OUTPUT column name — `WHERE`,
`GROUP BY`, and the rest of the query still see the original column name.
`ORDER BY` accepts both the old and the new name. A renamed name is visible
to an enclosing query the same way any other output column is.

Unlike `EXCLUDE`/`REPLACE`, which error on a column that doesn't exist,
`RENAME` of an unknown column is **silently ignored** (no error) — this
asymmetry matches DuckDB's actual behavior. Renaming a column onto a name
that already exists in the output is allowed and produces duplicate output
column names, it is not rejected either.

## Introspection: DESCRIBE, SHOW TABLES, EXPLAIN

```sql
DESCRIBE t;
DESCRIBE parquet('tests/data/basic.parquet');
SHOW TABLES;
EXPLAIN SELECT flag, count(*) FROM t GROUP BY flag;
```
