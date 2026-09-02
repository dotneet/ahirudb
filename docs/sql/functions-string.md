# String functions

[← Back to index](README.md)

> Every `SELECT` below is written as a bare expression for brevity. ahirudb
> requires a `FROM` clause on every `SELECT` (see
> [queries.md](queries.md#overall-shape)) — run any of these directly by
> appending `FROM range(1)`, e.g. `SELECT upper('abc') FROM range(1);`.

Unless noted otherwise, string functions operate on Unicode codepoints, not
bytes — `length('café')` is 4, not 5.

## Case, trimming, padding

| Function | Example | Notes |
|---|---|---|
| `upper(s)` / `ucase(s)` | `upper('abc')` → `'ABC'` | ASCII-only (no Unicode case tables) |
| `lower(s)` / `lcase(s)` | `lower('ABC')` → `'abc'` | ASCII-only |
| `trim(s[, chars])` | `trim('  x  ')` → `'x'` | Default char set is a plain space (not tab); strips both ends |
| `ltrim(s[, chars])` | `ltrim('  x')` → `'x'` | Left side only |
| `rtrim(s[, chars])` | `rtrim('x  ')` → `'x'` | Right side only |
| `lpad(s, len, fill)` | `lpad('5', 3, '0')` → `'005'` | Pads or truncates to exactly `len` codepoints |
| `rpad(s, len, fill)` | `rpad('5', 3, '0')` → `'500'` | Same, right side |

## Substrings, search, split

```sql
SELECT substring('hello', 2, 3);     -- 'ell'  (1-based)
SELECT substr('hello', -3);          -- 'llo'  (negative start counts from the end)
SELECT split_part('a,b,c', ',', 2);  -- 'b'    (1-based; negative index counts from the end)
SELECT strpos('hello world', 'wor'); -- 7      (aliases: position, instr; 0 if not found)
SELECT starts_with('hello', 'he');   -- true   (alias: prefix)
SELECT ends_with('hello', 'lo');     -- true   (alias: suffix)
SELECT contains('hello', 'ell');     -- true
SELECT left('abcde', 2);             -- 'ab'
SELECT right('abcde', 2);            -- 'de'
SELECT left('abcde', -2);            -- 'abc'  (negative = all but the last 2)
SELECT string_split('a,b,c', ',');   -- ["a","b","c"]  (a LIST, i.e. JSON text)
```

| Function | Aliases | Notes |
|---|---|---|
| `substring(s, start[, len])` | `substr` | 1-based; negative `start` counts from the end; negative `len` reverses the extracted range |
| `split_part(s, delim, index)` | — | 1-based, negative index counts from the end; index `0` or out of range → empty string |
| `strpos(s, sub)` | `position`, `instr` | 1-based position, `0` if not found |
| `starts_with(s, prefix)` | `prefix(s, p)` | boolean |
| `ends_with(s, suffix)` | `suffix(s, s2)` | boolean |
| `contains(s, sub)` | — | boolean |
| `length(s)` | `len`, `char_length`, `character_length` | counts codepoints |
| `left(s, n)` | — | First `n` codepoints; negative `n` drops the last `\|n\|` instead |
| `right(s, n)` | — | Last `n` codepoints; negative `n` drops the first `\|n\|` instead |
| `string_split(s, sep)` | `str_split`, `string_to_array`, `split` | Returns a LIST (JSON text — see [functions-json.md](functions-json.md)). An empty `sep` gives a one-element list holding the whole string (`string_split('abc','')` → `["abc"]`), where DuckDB 1.4 splits into characters (`[a, b, c]`) — a divergence |

`left` and `right` are reserved words (they introduce a join kind), but a
`(` immediately after the keyword is unambiguous, so writing them as
function calls needs no quoting.

## SQL-standard spellings

The SQL standard writes three of the functions above with keywords instead
of commas. Both spellings work and mean exactly the same thing — the
keyword forms are rewritten to the positional calls while the query is
parsed, so there is no difference in behavior or performance.

```sql
SELECT position('wor' IN 'hello world');        -- 7, same as strpos('hello world', 'wor')

SELECT substring('hello' FROM 2);               -- 'ello'
SELECT substring('hello' FROM 2 FOR 3);         -- 'ell'
SELECT substring('hello' FOR 3);                -- 'hel'   (start defaults to 1)

SELECT trim(BOTH 'x' FROM 'xxhixx');            -- 'hi',   same as trim('xxhixx', 'x')
SELECT trim(LEADING 'x' FROM 'xxhixx');         -- 'hixx', same as ltrim('xxhixx', 'x')
SELECT trim(TRAILING 'x' FROM 'xxhixx');        -- 'xxhi', same as rtrim('xxhixx', 'x')
SELECT trim('x' FROM 'xxhixx');                 -- 'hi',   direction defaults to BOTH
SELECT trim(FROM '  hi  ');                     -- 'hi',   char set defaults to a space
```

Note the **argument order flip** in `position`: the standard form names the
string being searched *for* first, the positional form names the string
being searched *in* first. `position('b' IN 'abc')` and `strpos('abc', 'b')`
are the same call.

`BOTH`, `LEADING`, `TRAILING`, and `FOR` are not reserved words — a column
named `leading` or `for` still works unquoted, and `trim(leading, 'x')`
stays an ordinary two-argument call on a column named `leading`.

## Prefix operator (starts-with)

```sql
SELECT 'hello' ^@ 'he';                       -- true
SELECT * FROM t WHERE name ^@ 'name_1';
```

`a ^@ b` is sugar for `starts_with(a, b)`; `NULL` on either side gives
`NULL`. It binds at comparison strength, but reads its right operand more
tightly than `||` does — so `'ab' ^@ 'a' || 'b'` is `('ab' ^@ 'a') || 'b'`,
while `'a' || 'b' ^@ 'a'` is `('a' || 'b') ^@ 'a'`. (Both match DuckDB;
this is the same asymmetry the `~~` operator family has, see
[Pattern matching](#pattern-matching).)

## Other transforms

```sql
SELECT replace('a-b-c', '-', '_');   -- 'a_b_c'  (literal, not regex; empty `from` is a no-op)
SELECT repeat('ab', 3);              -- 'ababab' (capped at 16 MiB output)
SELECT reverse('abc');               -- 'cba'    (codepoint-aware)
SELECT concat('a', NULL, 'b');       -- 'ab'     (NULL args treated as empty string; result is never NULL)
SELECT concat_ws('-', 'a', NULL, 'b'); -- 'a-b'  (a NULL value drops its separator too)
SELECT concat_ws(NULL, 'a', 'b');    -- NULL     (a NULL *separator* does propagate)
SELECT ascii('A');                   -- 65       (codepoint; aliases: unicode, ord)
SELECT chr(9731);                    -- '☃'
SELECT hex('AB');                    -- '4142'   (byte dump; an integer argument uses to_hex)
SELECT to_hex(255);                  -- 'FF'     (negatives use the two's-complement pattern)
```

`hex`/`to_hex` render an integer at **its own declared width**, not
truncated to 64 bits: a `HUGEINT` prints 128 bits, and a `UBIGINT` above
`i64::MAX` prints its true value rather than a sign-extended one.

```sql
SELECT to_hex(-1::HUGEINT);                 -- 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF' (32 nibbles)
SELECT hex(18446744073709551615::UBIGINT);  -- 'FFFFFFFFFFFFFFFF'
```

`concat` is special-cased: unlike most functions, it never returns `NULL`
even when an argument is `NULL` — a `NULL` argument just contributes
nothing to the result. This differs from `||` (the `Concat` binary
operator), which follows standard NULL propagation (`'a' || NULL` is
`NULL`).

`||` is string concatenation *unless both operands are `JSON`*, in which
case it concatenates them as lists (`[1,2] || [3]` → `[1,2,3]`) — see
[functions-json.md](functions-json.md#concatenating-lists).

## Pattern matching

```sql
SELECT 'name_0' LIKE 'name\_0' ESCAPE '\';   -- LIKE: %, _ wildcards
SELECT 'NAME_0' ILIKE 'name_0';              -- ILIKE: case-insensitive LIKE
SELECT 'name_0' GLOB 'name_?';               -- GLOB: shell-style *, ?, [...], [!...]
SELECT 'abc' SIMILAR TO 'a.c';               -- SIMILAR TO: SQL regex, anchored to the whole string

-- Punctuation aliases, from PostgreSQL/DuckDB:
SELECT 'name_0' ~~ 'name_0';                 -- ~~   = LIKE
SELECT 'name_0' !~~ 'x';                     -- !~~  = NOT LIKE
SELECT 'NAME_0' ~~* 'name_0';                -- ~~*  = ILIKE
SELECT 'NAME_0' !~~* 'x';                    -- !~~* = NOT ILIKE
SELECT 'name_0' ~~~ 'name_?';                -- ~~~  = GLOB
```

`GLOB`/`SIMILAR TO` desugar to the `glob(s, pattern)` / `regexp_full_match(s,
pattern)` scalar functions respectively (usable directly too). `GLOB`
matching is byte-oriented, not codepoint-oriented; an unterminated `[` in
the pattern matches nothing rather than erroring, matching DuckDB's
observed behavior.

The punctuation aliases (`~~`/`!~~`/`~~*`/`!~~*`/`~~~`) desugar the same
way the keyword forms do, with one genuinely surprising difference in
**operator precedence**: the keyword forms (`LIKE`/`ILIKE`/`GLOB`) read
their pattern operand loosely enough to swallow a following `||`, but the
punctuation forms read it more tightly, so a following `||` applies to the
*result* instead:

```sql
SELECT 'ab' LIKE 'a' || 'b';   -- true      ('ab' LIKE ('a'||'b'))
SELECT 'ab' ~~   'a' || 'b';   -- 'falseb'  (('ab' ~~ 'a') || 'b')
SELECT 'ab' GLOB 'a' || '*';   -- true      ('ab' GLOB ('a'||'*'))
SELECT 'ab' ~~~  'a' || '*';   -- 'false*'  (('ab' ~~~ 'a') || '*')
```

This matches DuckDB's own behavior exactly (verified against the `duckdb`
CLI) — it isn't an ahirudb-specific quirk, but it's easy to get bitten by
if you assume the punctuation spelling is a drop-in replacement for the
keyword. `ESCAPE` is only accepted after the `LIKE`/`ILIKE` *keywords*, not
after `~~`/`~~*` (DuckDB rejects it there too).

Unlike `~~`/`~~*` above, plain single-`~`/`!~` (one tilde, not two) is
*not* a `LIKE` alias — it's `SIMILAR TO`'s punctuation spelling, documented
under [Regular expressions](#regular-expressions) below and in
[queries.md](queries.md#where-operators-and-predicates).

## Regular expressions

```sql
SELECT regexp_matches('hello world', 'wor');           -- true (boolean test)
SELECT regexp_extract('hello world', '(\w+) (\w+)', 2); -- 'world' (capture group N, 0-based; group 0 = whole match)
SELECT regexp_replace('hello world', 'o', '0');          -- 'hell0 world' (first match only, unless a 'g' flag is passed)
SELECT regexp_full_match('abc', 'a.c');                  -- true (whole-string match; what SIMILAR TO desugars to)
```

The regex engine is a hand-written Thompson-NFA implementation (chosen to
guarantee linear-time matching, with no risk of catastrophic backtracking
on adversarial input — matters since Parquet/query text is treated as
untrusted). As a result, it does **not** support: lookaround
(`(?=...)`/`(?!...)`), backreferences inside the pattern itself
(`\1` in the *pattern*, as opposed to the *replacement* — see below),
named capture groups, non-greedy quantifiers (`*?`, `+?`), `\b`/`\B` word
boundaries, or a case-insensitive flag — neither the inline `(?i)` form nor
the `'i'` flag argument is accepted. `regexp_replace`'s replacement
string does support `\1`/`\2`-style backreferences to captured groups.

**Matching is per UTF-8 character, not per byte.** `.`, a character class,
and a quantifier each consume one whole Unicode scalar value, so
`regexp_replace('日本語abc', '.', 'X', 'g')` is `'XXXXXX'` (six characters
replaced, not twelve bytes), `regexp_matches('日本', '^..$')` is true, and
no regex operation can produce invalid UTF-8 by cutting a character in
half.

**POSIX bracket expressions are supported, and match ASCII only:**

```sql
SELECT regexp_extract('abc123', '[[:alpha:]]+');   -- 'abc'
SELECT regexp_extract('abc123', '[[:digit:]]+');   -- '123'
SELECT regexp_extract('ｱあ漢', '[[:alpha:]]+');     -- ''  (no ASCII letters)
```

`[[:alpha:]]`, `[[:digit:]]`, `[[:alnum:]]`, `[[:space:]]`, `[[:upper:]]`,
`[[:lower:]]`, `[[:punct:]]` and friends are ASCII-restricted for the same
reason `upper`/`lower` are: the Unicode tables cost more than the size
budget allows (see [limitations.md](limitations.md#not-supported-at-all)).

## printf / format

Two formatting styles, both usable directly on table columns:

```sql
SELECT printf('id=%d name=%s', id, name) FROM t LIMIT 2;
SELECT printf('%05d', 3);      -- '00003'
SELECT printf('%.2f', 3.14159); -- '3.14'
SELECT printf('%%');            -- '%'

SELECT format('{}-{}', 42, 'x');     -- '42-x'
SELECT format('{1}-{0}', 'a', 'b');  -- 'b-a'  (positional placeholders)
SELECT format('{{literal}}');        -- '{literal}'
```

`printf` supports the common C conversions (`%d`, `%s`, `%f`, plus width/
precision/`-`-left-align/`0`-pad modifiers) but not `%x`/`%o`, a `*`
dynamic width, or positional `%1$d`-style specifiers; it's also
deliberately more permissive than a strict C `printf` about argument-type
mismatches (e.g. `%d` accepts a `FLOAT` argument) rather than erroring.
`format` supports `{}`/`{n}` placeholders and `{{`/`}}` literal-brace
escapes, but not a format mini-language (`{:.2f}` is not supported — cast
or round the value yourself first).

**`%s` and `{}` render every type exactly as `CAST(... AS VARCHAR)` does**,
including the ones with no plain numeric spelling — `DATE`, `TIME`,
`TIMESTAMP`, `DECIMAL`, `HUGEINT`, `INTERVAL`, and `UUID` all print their
text form rather than the physical value underneath:

```sql
SELECT printf('%s', DATE '2020-01-02');                  -- '2020-01-02'
SELECT printf('%s', TIMESTAMP '2020-01-02 03:04:05');    -- '2020-01-02 03:04:05'
SELECT printf('%s', CAST('1.25' AS DECIMAL(5,2)));       -- '1.25'
SELECT printf('%s', 170141183460469231731687303715884105727::HUGEINT);
-- '170141183460469231731687303715884105727'
SELECT format('{}', INTERVAL '2 months');                -- '2 months'
```

`%d` and `%f`, by contrast, **reject an `INTERVAL` argument** with a
`TypeMismatch` — there is no meaningful single number to print for a value
that carries separate month, day, and microsecond components.

`%f` output is **correctly rounded, half-to-even**, computed from an exact
decimal expansion of the `DOUBLE` rather than by multiplying in floating
point (`printf('%.2f', 0.125)` is `'0.12'`, `printf('%.2f', 0.135)` is
`'0.14'`). The precision in `%.<N>f` is **clamped at 32**, so
`printf('%.40f', 1.0)` yields 32 digits after the point, not 40. With no
precision given, `%f` prints the value's full exact expansion the way C's
`printf` does — see
[limitations.md](limitations.md#partially-supported) for how that differs
from DuckDB on very large magnitudes and on `-0.0`.
