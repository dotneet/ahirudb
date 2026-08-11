# JSON, list, and map functions

[← Back to index](README.md)

> Every `SELECT` below is written as a bare expression for brevity. ahirudb
> requires a `FROM` clause on every `SELECT` (see
> [queries.md](queries.md#overall-shape)) — run any of these directly by
> appending `FROM range(1)`, e.g. `SELECT json_type('true') FROM range(1);`.

ahirudb has one dynamically-typed `JSON` type, and it does double duty: it's
also how Parquet `LIST`/`MAP` columns (and any `STRUCT` containing one) are
exposed — see [data-sources.md](data-sources.md#nested-parquet-types). There is no
separate, statically-typed `LIST`/`ARRAY`/`MAP` physical type; functions
named `list_*`/`map_*`/`array_*` (matching DuckDB's naming) all operate on
`JSON` values underneath.

## Path operators

```sql
SELECT '{"a":"hi"}' -> '$.a';    -- "hi"  (returns JSON — a string value stays quoted)
SELECT '{"a":"hi"}' ->> '$.a';   -- hi    (returns unquoted VARCHAR)
SELECT '{"a":1}' ->> '$.a' = '1';  -- true (->> binds tighter than =, no parens needed)
```

## Extraction

```sql
SELECT json_extract('{"a":{"b":[1,2,3]}}', '$.a.b[1]');  -- '2'   (array indexing in the path)
SELECT json_extract('[1,2,3]', '$[-1]');                  -- '3'   (negative index = from the end)
SELECT json_extract('{"a":1}', '$.b');                    -- NULL  (missing path -> SQL NULL, not an error)

SELECT json_extract_string('{"a":"hi"}', '$.a');   -- 'hi'  (unquotes the string)
SELECT json_extract_string('{"a":null}', '$.a');   -- NULL  (JSON null -> SQL NULL)

SELECT json_type('{"a":1}');   -- 'OBJECT'
SELECT json_type('[1,2]');     -- 'ARRAY'
SELECT json_type('"x"');       -- 'VARCHAR'
SELECT json_type('true');      -- 'BOOLEAN'
SELECT json_type('null');      -- 'NULL'
SELECT json_type('1.5');       -- 'DOUBLE'

SELECT json_array_length('[1,2,3]');  -- 3
SELECT json_array_length('{"a":1}');  -- 0  (non-array input -> 0, not an error, matching DuckDB)
```

## Construction

```sql
SELECT to_json(1);                           -- '1'
SELECT to_json('hello');                     -- '"hello"'
SELECT to_json(CAST('2024-01-01' AS DATE));  -- '"2024-01-01"'
SELECT to_json(NULL);                        -- SQL NULL, not the JSON literal null

SELECT json_object('a', 1, 'b', 'x');        -- '{"a":1,"b":"x"}'
SELECT json_array(1, 'x', true, NULL);       -- '[1,"x",true,null]'  (a SQL NULL argument becomes JSON null here)
SELECT list_value(1, 2, 3);                  -- '[1,2,3]'  (list_value is an alias for json_array)
SELECT [1, 2, 3];                            -- '[1,2,3]'  (array-literal sugar for list_value)

-- works over table columns, not just literals:
SELECT json_object('id', id, 'flag', flag) FROM t WHERE id IN (0, 1) ORDER BY id;
```

`to_json` accepts `NULL`/`BOOLEAN`/numeric/`DECIMAL`/`VARCHAR`/`DATE`/
`TIME`/`TIMESTAMP`/`JSON`; `BLOB` and `INTERVAL` are not JSON-encodable and
raise a type error.

## Accessing list/map elements

```sql
SELECT list_extract('[10,20,30]', 1);    -- '10'  (1-based, like DuckDB)
SELECT list_extract('[10,20,30]', -1);   -- '30'  (negative index from the end)
SELECT list_extract('[10,20,30]', 0);    -- NULL  (index 0 is invalid)

SELECT map_extract('{"a":1,"b":2}', 'a'); -- '1'
SELECT map_extract('{"a":1}', 'z');       -- NULL  (missing key -> NULL, not an error)
```

`array_extract` is an alias for `list_extract`.

## CAST to/from JSON

```sql
SELECT CAST('{"a":1}' AS JSON);                       -- round-trips
SELECT CAST(CAST('{"a":1}' AS JSON) AS VARCHAR);       -- '{"a":1}'
SELECT TRY_CAST('not json' AS JSON);                   -- NULL (lenient)
-- SELECT CAST('not json' AS JSON) errors instead (InvalidCast)
```

`JSON` equality (`=`/`<>`) is a **byte comparison**, not a semantic one —
`CAST('{"a": 1}' AS JSON) = CAST('{"a":1}' AS JSON)` is `false` because the
two documents differ in whitespace even though they mean the same thing.
Ordering comparisons (`<`, `>`, ...) on `JSON` are a type error; only
equality is defined.

## Lambda expressions and list_transform / list_filter / list_reduce

Lambda syntax (`x -> expr` for a single parameter, `(a, b) -> expr` for
several) is recognized **only** as an argument to `list_transform`,
`list_filter`, and `list_reduce` — anywhere else, `->` still means the JSON
path operator above. A lambda body can only reference its own parameters,
not columns from the surrounding query:

```sql
SELECT list_transform(json_array(1, 2, 3), x -> x + id) FROM t;
-- error: ColumnNotFound -- `id` isn't visible inside the lambda body
```

Because list elements are `JSON`-typed text, arithmetic on them needs an
explicit round-trip cast (`CAST(CAST(x AS VARCHAR) AS INTEGER)`); this is
the single most common idiom you'll see in lambda bodies:

```sql
-- list_transform: map each element
SELECT list_transform(json_array(1, 2, 3), x -> CAST(CAST(x AS VARCHAR) AS INTEGER) + 1);
-- '[2,3,4]'

SELECT list_transform(json_array(1, 2, 3), x -> x);   -- identity transform needs no cast
-- '[1,2,3]'

SELECT list_transform(json_array(1, 2, NULL, 4), x -> CAST(CAST(x AS VARCHAR) AS INTEGER) + 1);
-- '[2,3,null,5]'  (a NULL element passes through)

-- nested lambdas
SELECT list_transform(
  json_array(json_array(1, 2), json_array(3, 4)),
  y -> list_transform(y, x -> CAST(CAST(x AS VARCHAR) AS INTEGER) * 2)
);
-- '[[2,4],[6,8]]'

-- list_filter: keep elements where the predicate is true
SELECT list_filter(json_array(1, 2, 3, 4, 5), x -> CAST(CAST(x AS VARCHAR) AS INTEGER) > 2);
-- '[3,4,5]'

SELECT list_filter(json_array(1, 2, NULL, 4), x -> CAST(CAST(x AS VARCHAR) AS INTEGER) > 1);
-- '[2,4]'  (a NULL element's predicate is NULL/unknown, treated as false -- excluded, not an error)

-- list_reduce: fold, with an optional initial value (3rd argument)
SELECT list_reduce(json_array(1, 2, 3, 4),
  (acc, x) -> CAST(CAST(acc AS VARCHAR) AS INTEGER) + CAST(CAST(x AS VARCHAR) AS INTEGER));
-- '10'  (no initial value: starts from the first element)

SELECT list_reduce(CAST('[]' AS JSON),
  (acc, x) -> CAST(CAST(acc AS VARCHAR) AS INTEGER) + CAST(CAST(x AS VARCHAR) AS INTEGER),
  to_json(100));
-- '100' (empty list + explicit initial value -> the initial value)
```

Notable edge cases:

- `list_transform`/`list_filter`/`list_reduce` on a `NULL` list all return
  `NULL`.
- `list_transform` on non-array `JSON` input (e.g. a JSON object) is
  tolerated and coerced to `NULL`, rather than erroring — a deliberate
  divergence from DuckDB, which is statically typed and can't even express
  that input shape.
- `list_reduce` on an **empty list with no initial value** returns `NULL`
  — DuckDB errors in this case instead ("Cannot perform list_reduce on an
  empty input list"); ahirudb follows its general "coerce to `NULL` rather
  than fail the query" policy here.
- `list_filter`'s lambda body must evaluate to `BOOLEAN`; a non-boolean
  body is rejected at prepare time (`TypeMismatch`).
