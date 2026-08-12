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
boundaries, or a case-insensitive flag. `regexp_replace`'s replacement
string does support `\1`/`\2`-style backreferences to captured groups.

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
