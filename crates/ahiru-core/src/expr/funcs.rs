//! Scalar functions.
//!
//! `resolve` is called exactly once at planning time and decides, from the name and argument
//! types, "the function ID, each argument's conversion target type, and the result type".
//! Implicit conversion is inserted by the caller as a `Cast` instruction, so at runtime (`call`)
//! only a `match` on the ID remains -- no type checking and no per-type branching.
//!
//! The size policy inherits kernels.rs's four levers and adds three more:
//!
//! 1. **Only one loop per output physical type.** Per-row work is confined to the `match id`
//!    inside `eval_str` / `eval_int` / `eval_f64` / `eval_bool`. A loop per function would grow
//!    the code with the number of functions.
//! 2. **Separate IDs per type family.** `abs` has different IDs for integers and floating point.
//!    Logical types are, as a rule, not consulted at runtime, so the loop has one fewer check.
//!    The exceptions are the handful of functions whose *whole job* is to render or measure a
//!    value the way its logical type dictates, and which therefore cannot be split by ID at all:
//!    `printf`/`format` (whose argument types are only known per call site, see `write_display`)
//!    and `hex`/`to_hex` (which needs the argument's bit width). Those read `Vector::ty()`.
//! 3. **Any-type functions are written by composing kernels.** `coalesce` / `nullif` /
//!    `greatest` / `least` carry no code for the six physical types and lower to combinations of
//!    `kernels::pick` / `compare` / `logic` / `is_null`.
//!
//! ## Strings are counted by code point
//!
//! `length` / `substr` / `strpos` / `lpad` / `rpad` / `reverse` count in **UTF-8 code points**
//! (the same as DuckDB). It only skips continuation bytes (`0b10xxxxxx`), so the difference from
//! a byte-wise implementation is only a few dozen bytes. `upper` / `lower`, by contrast, are
//! **ASCII only**: Unicode case tables run to tens of KB and do not fit the 1 MiB budget.
//!
//! ## NULL
//!
//! The default is "if any argument is NULL, that row's result is NULL". The exceptions are
//! `coalesce` / `ifnull` / `nullif` / `greatest` / `least` (which skip NULLs), `concat` (which
//! treats NULL as the empty string, so the result is never NULL), and
//! `json_array` / `json_object` / `list_value` (which embed a JSON `null` at that position for a
//! NULL argument; `to_json(NULL)` itself is SQL NULL as usual).
//!
//! ## Functions that never reach `resolve`
//!
//! `now()` / `current_timestamp` / `current_date` do not resolve here (`FunctionNotFound`).
//! no_std wasm has no clock of its own, so the host supplies the query's start time
//! (`Session::set_now`) and `sql::now::substitute_now` folds these forms into literals before
//! binding. There is no general-purpose "read the clock" function beyond those fixed spellings.
//!
//! `pi()` and `typeof(x)` are folded to literals too, in
//! `plan::compile::Compiler::scalar_call`: `call`'s row loop takes its row count from the
//! arguments (`strides`), so a zero-argument function has nowhere to get one, and `typeof`'s
//! answer is fully determined by the argument's static type (folding it also keeps
//! `typeof(null_column)` at `'NULL'` instead of letting NULL propagation swallow it).

use crate::expr::vm::Vm;
use crate::expr::{kernels, regex, OpCode, Program};
use crate::prelude::*;
use crate::vector::{Batch, Bitmap, BytesData, Data, PhysType, Ty, Value, Vector};

pub type FuncId = u16;

// --- Function IDs -----------------------------------------------------------
// Grouped by output physical type. `call`'s loop is chosen by this grouping.

// Strings (Bytes output)
const F_UPPER: FuncId = 1;
const F_LOWER: FuncId = 2;
const F_TRIM: FuncId = 3;
const F_LTRIM: FuncId = 4;
const F_RTRIM: FuncId = 5;
const F_SUBSTR: FuncId = 6;
const F_REPLACE: FuncId = 7;
const F_LPAD: FuncId = 8;
const F_RPAD: FuncId = 9;
const F_REVERSE: FuncId = 10;
const F_SPLIT_PART: FuncId = 11;
const F_REPEAT: FuncId = 12;
const F_STRFTIME: FuncId = 13;
const F_CONCAT: FuncId = 14;
const F_REGEXP_EXTRACT: FuncId = 15;
const F_REGEXP_REPLACE: FuncId = 16;
const F_PRINTF: FuncId = 17;
const F_FORMAT: FuncId = 18;

// Integer output (I32 = DATE, I64 = BIGINT / TIMESTAMP)
const F_LENGTH: FuncId = 20;
const F_STRPOS: FuncId = 21;
const F_DATE_PART: FuncId = 22;
const F_DATE_DIFF: FuncId = 23;
const F_ABS_I: FuncId = 24;
const F_SIGN_I: FuncId = 25;
const F_ROUND_I: FuncId = 26;
const F_MOD_I: FuncId = 27;
const F_DATE_TRUNC: FuncId = 28;
const F_DATE_ADD: FuncId = 29;
const F_TO_DATE: FuncId = 30;
const F_TO_TIMESTAMP: FuncId = 31;
const F_LAST_DAY: FuncId = 32;
/// `ceil` / `floor` / `trunc` on a DECIMAL. Integers use `F_IDENT` instead (the
/// operation is the identity there); these drop the fractional digits of the scaled
/// integer, so the result is a `DECIMAL(p, 0)`.
const F_CEIL_I: FuncId = 33;
const F_FLOOR_I: FuncId = 34;
const F_TRUNC_I: FuncId = 35;

// Bool output
const F_STARTS_WITH: FuncId = 40;
const F_ENDS_WITH: FuncId = 41;
const F_CONTAINS: FuncId = 42;
const F_REGEXP_MATCHES: FuncId = 43;
const F_GLOB: FuncId = 44;
/// The substance of `SIMILAR TO`. `sql::parser` desugars `x SIMILAR TO y` into a call to a
/// function of this name (rather than adding a dedicated AST node like `Expr::Like`, everything
/// is done through the existing function-call path).
const F_REGEXP_FULL_MATCH: FuncId = 45;
/// The desugaring target of `s LIKE p ESCAPE 'e'` / `ILIKE ... ESCAPE 'e'`
/// (`sql::parser`). DuckDB implements the clause through an internal function
/// of the same name and signature; the matcher is `kernels::like_escape_match`.
const F_LIKE_ESC: FuncId = 46;

// Floating-point output
const F_ABS_F: FuncId = 50;
const F_SIGN_F: FuncId = 51;
const F_ROUND_F: FuncId = 52;
const F_CEIL_F: FuncId = 53;
const F_FLOOR_F: FuncId = 54;
const F_TRUNC_F: FuncId = 55;
const F_SQRT: FuncId = 56;
const F_EXP: FuncId = 57;
const F_LN: FuncId = 58;
const F_LOG10: FuncId = 59;
const F_POW: FuncId = 60;
const F_MOD_F: FuncId = 61;

// Any type (handled by composing kernels)
const F_IDENT: FuncId = 70;
const F_COALESCE: FuncId = 71;
const F_NULLIF: FuncId = 72;
const F_GREATEST: FuncId = 73;
const F_LEAST: FuncId = 74;

// Bitwise operations (the desugaring target of `&`/`|`/`<<`/`>>`/prefix `~`; see `sql::parser`).
// Fixed to BIGINT (the same reason as the other numeric functions' "HUGEINT/DECIMAL are rounded
// too" simplification; see the `num_ty` docs).
//
// Anything at or above `F_PART_BASE` (100) is taken by `eval_int` as a dynamic offset for
// `date_part` (it reads `id - F_PART_BASE` as the part number), so the free numbers just below
// that are used.
const F_BIT_AND: FuncId = 62;
const F_BIT_OR: FuncId = 63;
const F_BIT_SHL: FuncId = 64;
const F_BIT_SHR: FuncId = 65;
const F_BIT_NOT: FuncId = 66;

// HUGEINT output (I128). `factorial`/postfix `!` (`sql::parser::Parser::
// primary`/`cast_postfix` desugar `!` to this) is currently the only
// function that returns HUGEINT — see `call`'s `PhysType::I128` arm, added
// specifically to support it. Picked 93 to stay clear of both the JSON
// block below (80-92) and `F_PART_BASE` (100, see the comment above).
const F_FACTORIAL: FuncId = 93;

// JSON (Bytes output). `json_object`/`json_array` take a variable number of arguments and must
// embed a JSON `null` for a NULL argument rather than skipping it, so they are handled outside
// `call`'s row loop (`eval_str`), in the same place as `F_CONCAT` and friends.
const F_JSON_EXTRACT: FuncId = 80;
const F_JSON_EXTRACT_STRING: FuncId = 81;
const F_JSON_TYPE: FuncId = 82;
const F_TO_JSON: FuncId = 83;
const F_LIST_EXTRACT: FuncId = 84;
const F_MAP_EXTRACT: FuncId = 85;
const F_JSON_OBJECT: FuncId = 86;
const F_JSON_ARRAY: FuncId = 87;
// `list_transform`/`list_filter`/`list_reduce`. Unlike ordinary scalar functions,
// `plan::compile::Compiler::lambda_call` compiles them using these IDs directly, without going
// through `resolve`/`call` (the second argument is a lambda and has no ordinary `Ty`).
// Execution goes through `call_lambda` rather than `call` (see the `Call` instruction in
// `expr::vm::exec` and `CallSpec::lambda`). They are `pub(crate)` so `plan::compile` can
// reference them directly.
pub(crate) const F_LIST_TRANSFORM: FuncId = 88;
pub(crate) const F_LIST_FILTER: FuncId = 89;
pub(crate) const F_LIST_REDUCE: FuncId = 91;

// JSON (integer output)
const F_JSON_ARRAY_LENGTH: FuncId = 90;

/// The desugaring target of `list_slice`/`array_slice` and `expr[i:j]`. 92 was chosen because
/// 84-91 are already taken (above) and anything at or above `F_PART_BASE` (below, 100) is
/// reserved for date-part extraction -- `eval_int`/`eval_str` and friends begin by treating
/// `id >= F_PART_BASE` as date-part extraction, so using a number at or above 100 for a
/// general-purpose function would collide.
const F_LIST_SLICE: FuncId = 92;

// `list_concat` (and its `list_cat`/`array_concat`/`array_cat` aliases), plus
// the `||` operator when both operands are `JSON`. Two IDs share one body
// (`json::list_concat_build`) because the only difference between them is
// NULL handling, which DuckDB defines differently for the function and the
// operator — see that function's doc comment for the verification commands.
// 94/95 are the next free numbers below `F_PART_BASE` (93 = `F_FACTORIAL`).
//
// `F_LIST_CONCAT_OP` is `pub(crate)` because `plan::compile::binary` emits it
// directly: `resolve` is keyed on a function name, and `||` has none.
const F_LIST_CONCAT: FuncId = 94;
pub(crate) const F_LIST_CONCAT_OP: FuncId = 95;

// --- DuckDB-compatibility additions -----------------------------------------
// 96-199 is one flat block of free IDs (`F_PART_BASE` used to sit at 100 and
// cap this range at four spare numbers; it was moved to 200 so the block could
// grow). Grouped by output physical type, like the blocks above.

// Strings / JSON (Bytes output)
const F_CONCAT_WS: FuncId = 96;
const F_LEFT: FuncId = 97;
const F_RIGHT: FuncId = 98;
const F_CHR: FuncId = 99;
// The integer-path forms of `json_extract` / `->` / `->>`: DuckDB reads an integer path
// argument as a 0-based array subscript instead of an object key (see the `json_extract`
// arm in `resolve`). 100/101 are free -- `F_PART_BASE` used to sit at 100 and was moved
// to 200.
const F_JSON_EXTRACT_IDX: FuncId = 100;
const F_JSON_EXTRACT_STRING_IDX: FuncId = 101;
const F_DAYNAME: FuncId = 110;
const F_MONTHNAME: FuncId = 111;
const F_HEX: FuncId = 112;
const F_TO_HEX: FuncId = 113;
const F_STRING_SPLIT: FuncId = 114;
const F_LIST_SORT: FuncId = 115;
const F_LIST_DISTINCT: FuncId = 116;
const F_LIST_REVERSE: FuncId = 117;
const F_UNHEX: FuncId = 118;

// Integer output
const F_ASCII: FuncId = 120;
const F_GCD: FuncId = 121;
const F_LCM: FuncId = 122;
const F_BIT_COUNT: FuncId = 123;
const F_BIT_XOR: FuncId = 124;
const F_MAKE_DATE: FuncId = 125;
const F_MAKE_TIMESTAMP: FuncId = 126;
const F_EPOCH_MS: FuncId = 127;
const F_EPOCH_US: FuncId = 128;
const F_EPOCH_NS: FuncId = 129;
const F_LIST_POSITION: FuncId = 130;

// Bool output
const F_ISNAN: FuncId = 140;
const F_ISINF: FuncId = 141;
const F_ISFINITE: FuncId = 142;
const F_LIST_CONTAINS: FuncId = 143;

// Floating-point output
const F_LOG2: FuncId = 150;
const F_LOG_BASE: FuncId = 151;
const F_CBRT: FuncId = 152;
const F_RADIANS: FuncId = 153;
const F_DEGREES: FuncId = 154;

/// Shorthands such as `year()`. The part number is embedded in the ID, joining the same
/// extraction function as `date_part` (separate IDs per function, but one body).
///
/// Everything at or above this number is a date part, so it has to stay above every
/// ordinary function ID (see the free-ID block above).
const F_PART_BASE: FuncId = 200;

// --- Date-time parts ---------------------------------------------------------

const P_YEAR: u8 = 0;
const P_QUARTER: u8 = 1;
const P_MONTH: u8 = 2;
const P_WEEK: u8 = 3;
const P_DAY: u8 = 4;
const P_HOUR: u8 = 5;
const P_MINUTE: u8 = 6;
const P_SECOND: u8 = 7;
const P_DOW: u8 = 8;
const P_DOY: u8 = 9;
const P_EPOCH: u8 = 10;
const P_MILLISECOND: u8 = 11;
const P_MICROSECOND: u8 = 12;
const P_ISODOW: u8 = 13;
const P_CENTURY: u8 = 14;
const P_DECADE: u8 = 15;
const P_MILLENNIUM: u8 = 16;
const P_ISOYEAR: u8 = 17;

const PART_NAMES: [&[u8]; 18] = [
    b"year",
    b"quarter",
    b"month",
    b"week",
    b"day",
    b"hour",
    b"minute",
    b"second",
    b"dow",
    b"doy",
    b"epoch",
    b"millisecond",
    b"microsecond",
    b"isodow",
    b"century",
    b"decade",
    b"millennium",
    b"isoyear",
];

/// The alias spellings DuckDB accepts that dropping a trailing `s` from `PART_NAMES` cannot
/// reach: the abbreviations (which share no stem with the full name), the irregular plurals
/// `centuries` / `millennia`, and `ms` / `us`, which end in `s` and so must not go through the
/// plural stripping (`ms` would become `m` = minute).
///
/// Taken from <https://duckdb.org/docs/sql/functions/datepart> and each one checked against
/// `duckdb -c "select date_part('<name>', DATE '2024-05-05')"`. Note `m` is *minute* there, not
/// month. `era`, `julian`, `timezone*` and `yearweek` are deliberately absent -- see
/// `docs/sql/limitations.md`.
const PART_ALIASES: [(&[u8], u8); 36] = [
    (b"y", P_YEAR),
    (b"yr", P_YEAR),
    (b"yrs", P_YEAR),
    (b"mon", P_MONTH),
    (b"mons", P_MONTH),
    (b"d", P_DAY),
    (b"dayofmonth", P_DAY),
    (b"h", P_HOUR),
    (b"hr", P_HOUR),
    (b"hrs", P_HOUR),
    (b"m", P_MINUTE),
    (b"min", P_MINUTE),
    (b"mins", P_MINUTE),
    (b"s", P_SECOND),
    (b"sec", P_SECOND),
    (b"secs", P_SECOND),
    (b"ms", P_MILLISECOND),
    (b"msec", P_MILLISECOND),
    (b"msecs", P_MILLISECOND),
    (b"us", P_MICROSECOND),
    (b"usec", P_MICROSECOND),
    (b"usecs", P_MICROSECOND),
    (b"w", P_WEEK),
    (b"weekofyear", P_WEEK),
    (b"c", P_CENTURY),
    (b"cent", P_CENTURY),
    (b"centuries", P_CENTURY),
    (b"dec", P_DECADE),
    (b"decs", P_DECADE),
    (b"mil", P_MILLENNIUM),
    (b"mils", P_MILLENNIUM),
    (b"millennia", P_MILLENNIUM),
    (b"dayofweek", P_DOW),
    (b"weekday", P_DOW),
    (b"dayofyear", P_DOY),
    (b"isoweekday", P_ISODOW),
];

/// Maps a part name (case-insensitive) to a number. `PART_ALIASES` is consulted first, then
/// `PART_NAMES` with a trailing `s` dropped (`years` = `year`).
fn part_id(s: &[u8]) -> Option<u8> {
    let eq = |w: &[u8], t: &[u8]| {
        w.len() == t.len() && w.iter().zip(t.iter()).all(|(a, b)| a.to_ascii_lowercase() == *b)
    };
    if let Some(&(_, p)) = PART_ALIASES.iter().find(|(nm, _)| eq(s, nm)) {
        return Some(p);
    }
    // A trailing `s` is dropped (`years` = `year`).
    let t = if s.len() > 1 && (s[s.len() - 1] | 0x20) == b's' { &s[..s.len() - 1] } else { s };
    PART_NAMES.iter().position(|nm| eq(t, nm)).map(|i| i as u8)
}

// --- Constants ---------------------------------------------------------------

const US_PER_SEC: i64 = 1_000_000;
const US_PER_MIN: i64 = 60 * US_PER_SEC;
const US_PER_HOUR: i64 = 60 * US_PER_MIN;
const US_PER_DAY: i64 = 24 * US_PER_HOUR;

/// The per-row cap for functions that build strings (`repeat` / `lpad` and so on).
/// It prevents arbitrarily large allocations depending on the arguments.
const MAX_STR: i64 = 1 << 24;

// =========================================================================
// resolve
// =========================================================================

/// Resolves a function from its name and argument types.
///
/// The return value is `(function ID, each argument's conversion target type, result type)`. The
/// caller `Cast`s the arguments to the target types before passing them to `call`. Finishing type
/// checking at compile time reduces runtime branching.
pub fn resolve(name: &str, args: &[Ty]) -> Result<(FuncId, Vec<Ty>, Ty)> {
    resolve_const(name, args, &[])
}

/// [`resolve`], additionally told which arguments are compile-time integer constants
/// (`None` where an argument is not one, and a shorter slice than `args` is fine).
///
/// Exactly one signature needs this: `round(<decimal>, d)`. DuckDB's result scale
/// there is `min(s, max(d, 0))` -- it depends on the *value* of `d`, not on its type --
/// so `round(1.005::DECIMAL(4,3), 2)` is a `DECIMAL(4,2)` printing `1.01`. Without the
/// constant the scale has to stay at the input's, which prints the same number with
/// trailing zeros (`1.010`); that is the fallback for a non-constant `d`.
pub fn resolve_const(
    name: &str,
    args: &[Ty],
    consts: &[Option<i64>],
) -> Result<(FuncId, Vec<Ty>, Ty)> {
    let lower = name.to_ascii_lowercase();
    let n = args.len();
    use Ty::*;
    match lower.as_str() {
        // --- Strings --------------------------------------------------------
        "upper" | "ucase" => fixed(F_UPPER, &[Varchar], n, 1, Varchar),
        "lower" | "lcase" => fixed(F_LOWER, &[Varchar], n, 1, Varchar),
        "trim" => fixed(F_TRIM, &[Varchar, Varchar], n, 1, Varchar),
        "ltrim" => fixed(F_LTRIM, &[Varchar, Varchar], n, 1, Varchar),
        "rtrim" => fixed(F_RTRIM, &[Varchar, Varchar], n, 1, Varchar),
        "substr" | "substring" => fixed(F_SUBSTR, &[Varchar, BigInt, BigInt], n, 2, Varchar),
        "replace" => fixed(F_REPLACE, &[Varchar, Varchar, Varchar], n, 3, Varchar),
        "lpad" => fixed(F_LPAD, &[Varchar, BigInt, Varchar], n, 3, Varchar),
        "rpad" => fixed(F_RPAD, &[Varchar, BigInt, Varchar], n, 3, Varchar),
        "reverse" => fixed(F_REVERSE, &[Varchar], n, 1, Varchar),
        "split_part" => fixed(F_SPLIT_PART, &[Varchar, Varchar, BigInt], n, 3, Varchar),
        "repeat" => fixed(F_REPEAT, &[Varchar, BigInt], n, 2, Varchar),
        "concat" => {
            ensure!(n >= 1, WrongArgCount);
            Ok((F_CONCAT, vec![Varchar; n], Varchar))
        }
        // Unlike `concat`, a NULL argument is **skipped** (no separator emitted for it) and a
        // NULL separator makes the whole row NULL. Verified against duckdb:
        // `select concat_ws('-', 'a', NULL, 'b')` -> `a-b`,
        // `select concat_ws(NULL, 'a', 'b')` -> NULL.
        "concat_ws" => {
            ensure!(n >= 1, WrongArgCount);
            Ok((F_CONCAT_WS, vec![Varchar; n], Varchar))
        }
        // A negative count means "all but the last/first |n| characters" (DuckDB).
        "left" => fixed(F_LEFT, &[Varchar, BigInt], n, 2, Varchar),
        "right" => fixed(F_RIGHT, &[Varchar, BigInt], n, 2, Varchar),
        // Code-point in / code-point out, so `chr(ascii(s))` round-trips a single character.
        "ascii" | "unicode" | "ord" => fixed(F_ASCII, &[Varchar], n, 1, BigInt),
        "chr" => fixed(F_CHR, &[BigInt], n, 1, Varchar),
        // Splits into a LIST, which this engine represents as JSON text (see the module docs).
        "string_split" | "str_split" | "string_to_array" | "split" => {
            fixed(F_STRING_SPLIT, &[Varchar, Varchar], n, 2, Json)
        }
        // `hex` is overloaded in DuckDB: an integer argument is rendered in base 16, anything
        // else is the hex dump of its bytes. `to_hex` is the integer-only spelling.
        //
        // An integer argument keeps its own type rather than being coerced to BIGINT: HUGEINT
        // needs all 128 bits of its pattern, and a UBIGINT above `i64::MAX` would not survive
        // the cast at all (it became NULL). `eval_str` reads the width back off the vector.
        "hex" => {
            ensure!(n == 1, WrongArgCount);
            if args[0].is_integer() {
                Ok((F_TO_HEX, vec![args[0]], Varchar))
            } else {
                Ok((F_HEX, vec![Varchar], Varchar))
            }
        }
        "to_hex" => {
            ensure!(n == 1, WrongArgCount);
            // Non-integer arguments (DOUBLE, and NULL with no settled type) keep the previous
            // "round through BIGINT" leniency.
            let t = if args[0].is_integer() { args[0] } else { BigInt };
            Ok((F_TO_HEX, vec![t], Varchar))
        }
        "unhex" | "from_hex" => fixed(F_UNHEX, &[Varchar], n, 1, Blob),
        "length" | "len" | "char_length" | "character_length" => {
            fixed(F_LENGTH, &[Varchar], n, 1, BigInt)
        }
        "strpos" | "position" | "instr" => fixed(F_STRPOS, &[Varchar, Varchar], n, 2, BigInt),
        "starts_with" | "prefix" => fixed(F_STARTS_WITH, &[Varchar, Varchar], n, 2, Boolean),
        "ends_with" | "suffix" => fixed(F_ENDS_WITH, &[Varchar, Varchar], n, 2, Boolean),
        "contains" => fixed(F_CONTAINS, &[Varchar, Varchar], n, 2, Boolean),
        // The regex engine itself is expr::regex (see the comments at the top of that module for
        // supported syntax and limits). This is only type resolution.
        "regexp_matches" => fixed(F_REGEXP_MATCHES, &[Varchar, Varchar], n, 2, Boolean),
        "regexp_extract" => fixed(F_REGEXP_EXTRACT, &[Varchar, Varchar, BigInt], n, 2, Varchar),
        "regexp_replace" => {
            fixed(F_REGEXP_REPLACE, &[Varchar, Varchar, Varchar, Varchar], n, 3, Varchar)
        }
        // The desugaring target of `SIMILAR TO` (`sql::parser::Parser::similar_to`). The pattern is
        // wrapped in `^(?:...)$` and passed straight to `expr::regex`'s existing engine (see that
        // module's docs for supported syntax and the step cap; no new regex engine is written).
        // DuckDB uses the same pattern syntax as well (`_`/`%` are not special in POSIX regular
        // expressions).
        "regexp_full_match" => fixed(F_REGEXP_FULL_MATCH, &[Varchar, Varchar], n, 2, Boolean),
        // The `GLOB` operator (desugared by `sql::parser::expr_body`). Whether it fully matches a
        // shell glob pattern (`*` `?` `[...]`). See the comments on `glob_match` for the supported
        // syntax.
        "glob" => fixed(F_GLOB, &[Varchar, Varchar], n, 2, Boolean),
        // The desugaring target of `s LIKE p ESCAPE 'e'` / `ILIKE ... ESCAPE`
        // (`sql::parser`; DuckDB implements the clause through a function of
        // the same name and shape). The third argument is exactly one escape
        // character. The matcher itself is `kernels::like_escape_match`.
        "like_escape" => fixed(F_LIKE_ESC, &[Varchar, Varchar, Varchar], n, 3, Boolean),
        // `printf`/`format` differ in the range of format specifiers they support (printf: `%`
        // formats; format: `{}` placeholders), so their IDs are separate. Only the first argument
        // (the format string) is fixed to VARCHAR and the rest are passed with their original types
        // (the same "per-type branching happens at runtime" policy as `json_array`/`greatest`.
        // The format string is not necessarily a constant, so at `resolve` time neither the number
        // nor the kinds of specifiers are known). See the comments on `printf_scan`/`format_scan`
        // for the supported range.
        "printf" | "format" => {
            ensure!(n >= 1, WrongArgCount);
            let mut want = vec![Varchar];
            want.extend_from_slice(&args[1..]);
            let id = if lower == "printf" { F_PRINTF } else { F_FORMAT };
            Ok((id, want, Varchar))
        }

        // --- Numbers -------------------------------------------------------
        // Integers and floating point get separate IDs, since logical types are not consulted at runtime.
        "abs" => num1(args, n, F_ABS_I, F_ABS_F),
        // The result is always a plain integer (-1/0/1), never the argument's own
        // DECIMAL/HUGEINT type; DuckDB narrows it all the way to TINYINT.
        "sign" => {
            ensure!(n == 1, WrongArgCount);
            let t = num_ty(args)?;
            if t == Double {
                Ok((F_SIGN_F, vec![Double], Double))
            } else {
                Ok((F_SIGN_I, vec![t], BigInt))
            }
        }
        // ceil / floor / trunc on integers are the identity. There is no dedicated kernel.
        "ceil" | "ceiling" => num1_whole(args, n, F_CEIL_I, F_CEIL_F),
        "floor" => num1_whole(args, n, F_FLOOR_I, F_FLOOR_F),
        "trunc" => num1_whole(args, n, F_TRUNC_I, F_TRUNC_F),
        "round" => {
            ensure!((1..=2).contains(&n), WrongArgCount);
            let t = num_ty(&args[..1])?;
            let id = if t == Double { F_ROUND_F } else { F_ROUND_I };
            let mut a = vec![t];
            if n == 2 {
                a.push(BigInt);
            }
            // On a DECIMAL the digits below the requested position are dropped from the
            // result's scale too, the way DuckDB does it (`round(x::DECIMAL(4,3), 2)` is a
            // `DECIMAL(4,2)`). That needs `d`'s value, so it only happens when `d` is a
            // literal; otherwise the input's own scale is kept and the same number simply
            // prints with trailing zeros.
            let res = match t {
                Decimal { precision, scale } if n == 2 => {
                    let d = consts.get(1).copied().flatten().unwrap_or(scale as i64);
                    Ty::decimal(precision, d.clamp(0, scale as i64) as u8)
                }
                Decimal { precision, .. } => Ty::decimal(precision, 0),
                _ => t,
            };
            Ok((id, a, res))
        }
        "mod" => {
            ensure!(n == 2, WrongArgCount);
            let t = num_ty(args)?;
            Ok((if t == Double { F_MOD_F } else { F_MOD_I }, vec![t; 2], t))
        }
        "sqrt" => fixed(F_SQRT, &[Double], n, 1, Double),
        "cbrt" => fixed(F_CBRT, &[Double], n, 1, Double),
        "exp" => fixed(F_EXP, &[Double], n, 1, Double),
        "ln" => fixed(F_LN, &[Double], n, 1, Double),
        // DuckDB's `log(x)` is the common logarithm; `log(b, x)` is base `b`.
        "log" => {
            ensure!((1..=2).contains(&n), WrongArgCount);
            if n == 1 {
                Ok((F_LOG10, vec![Double], Double))
            } else {
                Ok((F_LOG_BASE, vec![Double; 2], Double))
            }
        }
        "log10" => fixed(F_LOG10, &[Double], n, 1, Double),
        "log2" => fixed(F_LOG2, &[Double], n, 1, Double),
        "pow" | "power" => fixed(F_POW, &[Double, Double], n, 2, Double),
        "radians" => fixed(F_RADIANS, &[Double], n, 1, Double),
        "degrees" => fixed(F_DEGREES, &[Double], n, 1, Double),
        // Defined on floating point only; an integer can never be NaN or infinite, and DuckDB
        // resolves `isnan(1)` to the FLOAT overload too.
        "isnan" => fixed(F_ISNAN, &[Double], n, 1, Boolean),
        "isinf" => fixed(F_ISINF, &[Double], n, 1, Boolean),
        "isfinite" => fixed(F_ISFINITE, &[Double], n, 1, Boolean),
        "gcd" | "greatest_common_divisor" => fixed(F_GCD, &[BigInt, BigInt], n, 2, BigInt),
        "lcm" | "least_common_multiple" => fixed(F_LCM, &[BigInt, BigInt], n, 2, BigInt),
        "bit_count" => fixed(F_BIT_COUNT, &[BigInt], n, 1, BigInt),
        "xor" => fixed(F_BIT_XOR, &[BigInt, BigInt], n, 2, BigInt),
        // The desugaring target of `&`/`|`/`<<`/`>>`/prefix `~` (see `sql::parser`).
        "bit_and" => fixed(F_BIT_AND, &[BigInt, BigInt], n, 2, BigInt),
        "bit_or" => fixed(F_BIT_OR, &[BigInt, BigInt], n, 2, BigInt),
        "bit_shift_left" => fixed(F_BIT_SHL, &[BigInt, BigInt], n, 2, BigInt),
        "bit_shift_right" => fixed(F_BIT_SHR, &[BigInt, BigInt], n, 2, BigInt),
        "bit_not" => fixed(F_BIT_NOT, &[BigInt], n, 1, BigInt),
        // Postfix `!` (`sql::parser`) desugars here. Returns HUGEINT to
        // match duckdb (`typeof(4!)` -> `HUGEINT`); only integer input is
        // accepted (duckdb itself rejects a DOUBLE argument, e.g.
        // `factorial(4.0)`, as a type error — verified against the
        // `duckdb` CLI). `n < 0` is a valid input (see `numeric::factorial`
        // — it returns 1, not an error), so no range check happens here.
        "factorial" => {
            ensure!(n == 1, WrongArgCount);
            ensure!(
                args[0] == Null || (args[0].is_numeric() && args[0].is_integer()),
                TypeMismatch
            );
            Ok((F_FACTORIAL, vec![BigInt], HugeInt))
        }

        // --- JSON -------------------------------------------------------
        // See `crate::json`'s module docs for the path syntax and known limitations.
        // The `->`/`->>` operators are expanded by `sql::parser` as sugar for
        // `json_extract`/`json_extract_string`.
        // An *integer* path argument is a 0-based array subscript in DuckDB (negative counts
        // from the end), never an object key: `'[1,2]' -> 0` is `1`, `'[1,2]' ->> -1` is `2`,
        // `json_extract('[1,2]', 1)` is `2`, and `'{"0":5}' -> 0` is NULL. Only the static type
        // decides; a VARCHAR path keeps the JSONPath semantics (`'{"0":5}' -> '0'` is `5`).
        "json_extract" => {
            ensure!(n == 2, WrongArgCount);
            if args[1].is_integer() {
                Ok((F_JSON_EXTRACT_IDX, vec![Json, BigInt], Json))
            } else {
                Ok((F_JSON_EXTRACT, vec![Json, Varchar], Json))
            }
        }
        "json_extract_string" => {
            ensure!(n == 2, WrongArgCount);
            if args[1].is_integer() {
                Ok((F_JSON_EXTRACT_STRING_IDX, vec![Json, BigInt], Varchar))
            } else {
                Ok((F_JSON_EXTRACT_STRING, vec![Json, Varchar], Varchar))
            }
        }
        "json_type" => json_path_opt(F_JSON_TYPE, n, Varchar),
        // `array_length`/`list_length` are DuckDB's LIST spellings of the same thing. They ride
        // the existing ID: a LIST here *is* a JSON array (see the module docs).
        "json_array_length" | "array_length" | "list_length" => {
            json_path_opt(F_JSON_ARRAY_LENGTH, n, BigInt)
        }
        // The element argument keeps its own type and is serialized to JSON text per row, then
        // compared byte-wise against each element -- the same equality JSON values already use
        // in this engine (see `docs/sql/limitations.md`).
        "list_contains" | "array_contains" | "list_has" | "array_has" => {
            ensure!(n == 2, WrongArgCount);
            ensure!(json_encodable(args[1]), TypeMismatch);
            Ok((F_LIST_CONTAINS, vec![Json, args[1]], Boolean))
        }
        "list_position" | "list_indexof" | "array_position" | "array_indexof" => {
            ensure!(n == 2, WrongArgCount);
            ensure!(json_encodable(args[1]), TypeMismatch);
            Ok((F_LIST_POSITION, vec![Json, args[1]], BigInt))
        }
        "list_sort" | "array_sort" => fixed(F_LIST_SORT, &[Json], n, 1, Json),
        // Note `list_unique` is deliberately *not* an alias here: in DuckDB it returns the
        // *count* of distinct elements, not the deduplicated list.
        "list_distinct" | "array_distinct" => fixed(F_LIST_DISTINCT, &[Json], n, 1, Json),
        "list_reverse" | "array_reverse" => fixed(F_LIST_REVERSE, &[Json], n, 1, Json),
        // They carry LIST/MAP-family names by DuckDB convention, but really they just read the
        // array/object shape of a Ty::Json (the design decision in the module docs at the top).
        "list_extract" | "array_extract" => fixed(F_LIST_EXTRACT, &[Json, BigInt], n, 2, Json),
        // The desugaring target of `expr[i:j]` (`sql::parser::Parser::subscript`). Inclusive on both
        // ends, 1-based, with negatives counting from the end; unlike `list_extract`, out of range
        // clamps rather than giving NULL (see `crate::json::list_slice`'s module docs).
        "list_slice" | "array_slice" => fixed(F_LIST_SLICE, &[Json, BigInt, BigInt], n, 3, Json),
        "map_extract" => fixed(F_MAP_EXTRACT, &[Json, Varchar], n, 2, Json),
        // Variadic, like DuckDB (`duckdb -c "select list_concat([1]),
        // list_concat([1,2],[3],NULL::INT[],[4])"` -> `[1]`, `[1, 2, 3, 4]`).
        // `list_cat`/`array_concat`/`array_cat` are DuckDB's own aliases for
        // the same function (verified with `duckdb -c "select list_cat([1],
        // [2]), array_concat([1],[2]), array_cat([1],[2])"`), and cost one
        // more match pattern each.
        "list_concat" | "list_cat" | "array_concat" | "array_cat" => {
            ensure!(n >= 1, WrongArgCount);
            Ok((F_LIST_CONCAT, vec![Json; n], Json))
        }
        "to_json" => {
            ensure!(n == 1, WrongArgCount);
            ensure!(json_encodable(args[0]), TypeMismatch);
            Ok((F_TO_JSON, vec![args[0]], Json))
        }
        "json_object" => {
            ensure!(n >= 2 && n.is_multiple_of(2), WrongArgCount);
            let mut want = Vec::with_capacity(n);
            for (i, &a) in args.iter().enumerate() {
                if i % 2 == 0 {
                    want.push(Varchar);
                } else {
                    ensure!(json_encodable(a), TypeMismatch);
                    want.push(a);
                }
            }
            Ok((F_JSON_OBJECT, want, Json))
        }
        // `list_value` is DuckDB's conventional alias for `json_array`.
        "json_array" | "list_value" => {
            ensure!(n >= 1, WrongArgCount);
            for &a in args {
                ensure!(json_encodable(a), TypeMismatch);
            }
            Ok((F_JSON_ARRAY, args.to_vec(), Json))
        }

        // --- Any type -------------------------------------------------------
        "greatest" => anyn(F_GREATEST, args, 1),
        "least" => anyn(F_LEAST, args, 1),
        "coalesce" => anyn(F_COALESCE, args, 1),
        "ifnull" => {
            ensure!(n == 2, WrongArgCount);
            anyn(F_COALESCE, args, 2)
        }
        "nullif" => {
            ensure!(n == 2, WrongArgCount);
            anyn(F_NULLIF, args, 2)
        }

        // --- Date and time ---------------------------------------------------
        // The internal representations are DATE = days (I32) / TIMESTAMP = microseconds (I64) / TIME = microseconds (I64).
        // Arguments settle on TIMESTAMP, so both DATE columns and VARCHAR literals can be passed
        // straight through via the caller's `Cast`.
        //
        // Note: DuckDB 1.4's `date_trunc` returns DATE or TIMESTAMP depending on the part, but
        // `resolve` cannot see the part's **value** (only types arrive), so it always returns
        // TIMESTAMP. For the same reason `date_part('epoch', ..)` returns BIGINT (seconds,
        // truncated) rather than DOUBLE.
        "date_trunc" | "datetrunc" => fixed(F_DATE_TRUNC, &[Varchar, Timestamp], n, 2, Timestamp),
        "date_part" | "datepart" | "extract" => {
            fixed(F_DATE_PART, &[Varchar, Timestamp], n, 2, BigInt)
        }
        // `date_sub` is deliberately not an alias of this: DuckDB's `date_sub` counts *complete*
        // partitions where `date_diff` counts boundaries crossed, so the two disagree over a
        // partial unit (`date_diff('day', '..23:00', '..01:00')` is 1, `date_sub` is 0).
        "date_diff" | "datediff" => {
            fixed(F_DATE_DIFF, &[Varchar, Timestamp, Timestamp], n, 3, BigInt)
        }
        // DuckDB's `date_add` takes an INTERVAL, but this implementation has no INTERVAL type.
        // As a deliberate incompatibility it takes three arguments, `date_add(part, n, ts)`.
        "date_add" => fixed(F_DATE_ADD, &[Varchar, BigInt, Timestamp], n, 3, Timestamp),
        "last_day" => fixed(F_LAST_DAY, &[Timestamp], n, 1, Date),
        "strftime" => fixed(F_STRFTIME, &[Timestamp, Varchar], n, 2, Varchar),
        // English names, matching DuckDB's output exactly (`Monday`, `January`, ...).
        // No locale support -- the engine carries no locale data at all.
        "dayname" => fixed(F_DAYNAME, &[Timestamp], n, 1, Varchar),
        "monthname" => fixed(F_MONTHNAME, &[Timestamp], n, 1, Varchar),
        "make_date" => fixed(F_MAKE_DATE, &[BigInt, BigInt, BigInt], n, 3, Date),
        // `make_timestamp(y, mo, d, h, mi, s)`. DuckDB's single-argument
        // `make_timestamp(microseconds)` overload is not provided (`CAST` covers it).
        // The seconds argument is DOUBLE in DuckDB; here it is BIGINT, so fractional seconds
        // have to be written with `make_timestamp(...) + INTERVAL ... ` instead.
        "make_timestamp" => fixed(
            F_MAKE_TIMESTAMP,
            &[BigInt, BigInt, BigInt, BigInt, BigInt, BigInt],
            n,
            6,
            Timestamp,
        ),
        // The `epoch` shorthand is whole seconds; these are the finer-grained spellings.
        "epoch_ms" => fixed(F_EPOCH_MS, &[Timestamp], n, 1, BigInt),
        "epoch_us" => fixed(F_EPOCH_US, &[Timestamp], n, 1, BigInt),
        "epoch_ns" => fixed(F_EPOCH_NS, &[Timestamp], n, 1, BigInt),
        // DuckDB's `to_timestamp` takes epoch seconds (DOUBLE), but here it is defined as a string
        // parser (the equivalent of `strptime`). A deliberate incompatibility.
        "to_date" => fixed(F_TO_DATE, &[Varchar], n, 1, Date),
        "to_timestamp" => fixed(F_TO_TIMESTAMP, &[Varchar], n, 1, Timestamp),
        "year" => shorthand(P_YEAR, n),
        "quarter" => shorthand(P_QUARTER, n),
        "month" => shorthand(P_MONTH, n),
        "week" => shorthand(P_WEEK, n),
        "day" | "dayofmonth" => shorthand(P_DAY, n),
        "hour" => shorthand(P_HOUR, n),
        "minute" => shorthand(P_MINUTE, n),
        "second" => shorthand(P_SECOND, n),
        "dayofweek" => shorthand(P_DOW, n),
        "dayofyear" => shorthand(P_DOY, n),
        "epoch" => shorthand(P_EPOCH, n),
        "millisecond" => shorthand(P_MILLISECOND, n),
        "microsecond" => shorthand(P_MICROSECOND, n),
        "isodow" => shorthand(P_ISODOW, n),
        "century" => shorthand(P_CENTURY, n),
        "decade" => shorthand(P_DECADE, n),
        "millennium" => shorthand(P_MILLENNIUM, n),
        "isoyear" => shorthand(P_ISOYEAR, n),

        // Not implemented, since there is no clock (see the comments at the top of the module).
        _ => err!(FunctionNotFound),
    }
}

/// A fixed signature. `pat` is the template of argument types; when `n < pat.len()` the leading
/// ones are used (optional arguments).
fn fixed(id: FuncId, pat: &[Ty], n: usize, lo: usize, ret: Ty) -> Result<(FuncId, Vec<Ty>, Ty)> {
    ensure!(n >= lo && n <= pat.len(), WrongArgCount);
    Ok((id, pat[..n].to_vec(), ret))
}

/// The 1-2 argument signature `f(json)` / `f(json, path)`. The common shape for JSON functions
/// such as `json_type`/`json_array_length` that switch between "the whole JSON" and "a path"
/// solely by the presence of the second argument.
fn json_path_opt(id: FuncId, n: usize, ret: Ty) -> Result<(FuncId, Vec<Ty>, Ty)> {
    ensure!((1..=2).contains(&n), WrongArgCount);
    let mut want = vec![Ty::Json];
    if n == 2 {
        want.push(Ty::Varchar);
    }
    Ok((id, want, ret))
}

/// A one-argument numeric function. `int_id` for integers, `flt_id` otherwise.
fn num1(args: &[Ty], n: usize, int_id: FuncId, flt_id: FuncId) -> Result<(FuncId, Vec<Ty>, Ty)> {
    ensure!(n == 1, WrongArgCount);
    let t = num_ty(args)?;
    Ok((if t == Ty::Double { flt_id } else { int_id }, vec![t], t))
}

/// The common type of numeric arguments.
///
/// Integers narrower than 64 bits all settle on BIGINT (one `i64` kernel covers them, and the
/// result cannot overflow one). HUGEINT/UBIGINT keep their full 128-bit width and DECIMAL keeps
/// its scale, so `mod`/`abs`/`round` stay exact there: they used to settle on DOUBLE as well,
/// which silently rounded every value above 2^53 (`mod(<hugeint max>, 10)` answered `8` instead
/// of `7`) and turned exact decimal rounding into binary rounding (`round(1.005::DECIMAL(4,3), 2)`
/// answered `1.0` instead of `1.01`). FLOAT settles on DOUBLE, since the `f64` kernels are the
/// only floating-point ones.
fn num_ty(args: &[Ty]) -> Result<Ty> {
    let mut acc = Ty::Null;
    for &t in args {
        if t == Ty::Null {
            continue;
        }
        ensure!(t.is_numeric(), TypeMismatch);
        let t = match t {
            Ty::HugeInt | Ty::UBigInt | Ty::Decimal { .. } => t,
            _ if t.is_integer() => Ty::BigInt,
            _ => Ty::Double,
        };
        acc = match Ty::unify(acc, t) {
            Some(u) => u,
            None => err!(TypeMismatch),
        };
    }
    // All arguments were type-undetermined NULLs.
    Ok(if acc == Ty::Null { Ty::BigInt } else { acc })
}

/// `ceil` / `floor` / `trunc`: the identity on integers, a scale-dropping kernel on DECIMAL
/// (the result is a `DECIMAL(p, 0)`, as in DuckDB), and the `f64` kernel on floating point.
fn num1_whole(
    args: &[Ty],
    n: usize,
    int_id: FuncId,
    flt_id: FuncId,
) -> Result<(FuncId, Vec<Ty>, Ty)> {
    ensure!(n == 1, WrongArgCount);
    let t = num_ty(args)?;
    Ok(match t {
        Ty::Double => (flt_id, vec![Ty::Double], Ty::Double),
        Ty::Decimal { precision, scale } if scale > 0 => {
            (int_id, vec![t], Ty::decimal(precision, 0))
        }
        _ => (F_IDENT, vec![t], t),
    })
}

/// Whether the type is writable as a value of `to_json`/`json_array`/`json_object`.
/// Supported are NULL/BOOLEAN/integers/floating point/DECIMAL/VARCHAR/DATE/TIME/TIMESTAMP/JSON
/// (embedded as is). BLOB and INTERVAL are unsupported (they have no natural JSON representation,
/// so like CAST they are rejected with `TypeMismatch`).
fn json_encodable(t: Ty) -> bool {
    use Ty::*;
    t.is_numeric() || matches!(t, Null | Boolean | Varchar | Date | Time | Timestamp | Json)
}

/// A variadic any-type function. Every argument settles on a common type.
fn anyn(id: FuncId, args: &[Ty], lo: usize) -> Result<(FuncId, Vec<Ty>, Ty)> {
    ensure!(args.len() >= lo, WrongArgCount);
    let mut t = Ty::Null;
    for &a in args {
        t = match Ty::unify(t, a) {
            Some(u) => u,
            None => err!(TypeMismatch),
        };
    }
    // If all of them are type-undetermined NULLs, some type is picked (DuckDB picks INTEGER too).
    if t == Ty::Null {
        t = Ty::Int;
    }
    Ok((id, vec![t; args.len()], t))
}

/// Shorthands such as `year(ts)`. The part is embedded in the ID.
fn shorthand(part: u8, n: usize) -> Result<(FuncId, Vec<Ty>, Ty)> {
    ensure!(n == 1, WrongArgCount);
    Ok((F_PART_BASE + part as FuncId, vec![Ty::Timestamp], Ty::BigInt))
}

// =========================================================================
// call
// =========================================================================

/// Executes. `args` are already aligned to the types `resolve` returned.
///
/// Every argument is either the same length or length 1 (a constant). Constants are treated as stride 0.
pub fn call(id: FuncId, result_ty: Ty, args: &[&Vector]) -> Result<Vector> {
    // Any-type functions are handled by composing kernels (no per-physical-type code).
    match id {
        F_IDENT => {
            ensure!(args.len() == 1, WrongArgCount);
            return Ok(args[0].clone());
        }
        F_COALESCE => return fold_null(args, result_ty),
        F_NULLIF => return nullif(args, result_ty),
        F_GREATEST | F_LEAST => return extremum(id == F_GREATEST, args, result_ty),
        F_CONCAT => return concat_all(args, result_ty),
        // Like `concat`, `concat_ws` decides NULL per argument rather than per row (a NULL
        // value is skipped entirely, separator included), so it bypasses the row loop too.
        F_CONCAT_WS => return concat_ws_build(args, result_ty),
        // The three regex functions do not ride the per-row eval_* loops. When the pattern column
        // is constant (stride 0) we want to compile once per batch, but carrying that cache into
        // eval_bool/eval_str's shared signature would be unnatural, so like F_CONCAT they are
        // passed to a dedicated function directly under call().
        F_REGEXP_MATCHES => return regex::eval_matches(args),
        F_REGEXP_EXTRACT => return regex::eval_extract(args),
        F_REGEXP_REPLACE => return regex::eval_replace(args),
        // `SIMILAR TO` (`regexp_full_match`) wants the same pattern-column compilation cache, so it
        // too is passed to a dedicated function (see `regexp_full_match_build`. It is almost
        // identical to `regex::eval_matches`, differing only in wrapping the pattern in anchors).
        F_REGEXP_FULL_MATCH => return regexp_full_match_build(args),
        // `json_array`/`json_object` also embed a JSON `null` for a NULL argument rather than
        // skipping it (departing from default NULL propagation for the same reason as `concat`), so
        // they are passed to a dedicated function.
        F_JSON_ARRAY => return json_array_build(args),
        F_JSON_OBJECT => return json_object_build(args),
        // `list_concat` and `||`-on-JSON also decide NULL row by row rather
        // than letting `call()`'s default propagation do it (the function form
        // treats a NULL list as an empty one; the operator form has to keep
        // inspecting the other operands after seeing a NULL, because a
        // non-array one raises), so they get the same bypass as
        // `concat`/`json_array`.
        F_LIST_CONCAT | F_LIST_CONCAT_OP => return list_concat_build(args, id == F_LIST_CONCAT_OP),
        _ => {}
    }
    let (n, s) = strides(args)?;
    // The AND of the inputs' validity. NULL rows skip evaluation entirely (so no placeholder value is read).
    // list_position is the one strict-looking function whose second argument is
    // intentionally still inspected when it is SQL NULL: DuckDB treats NULL as a
    // searchable list element (list_position([1, NULL], NULL) = 2), while a NULL
    // list remains NULL. Keep the normal strict propagation for every other function.
    let valid = if id == F_LIST_POSITION {
        if args[0].has_nulls() {
            let mut m = Bitmap::with_capacity(n);
            for i in 0..n {
                m.push(args[0].is_valid(i * s[0]));
            }
            Some(m)
        } else {
            None
        }
    } else {
        combine(args, &s, n)
    };
    let live = |i: usize| valid.as_ref().is_none_or(|b| b.get(i));
    let mut data = Data::with_capacity(result_ty.phys(), n);
    let mut bad: Option<Bitmap> = None;
    match result_ty.phys() {
        PhysType::Bytes => {
            let mut buf = Vec::new();
            if let Data::Bytes(d) = &mut data {
                for i in 0..n {
                    if !live(i) {
                        d.push_empty();
                        continue;
                    }
                    buf.clear();
                    if eval_str(id, &A { v: args, s: &s, i }, &mut buf)? {
                        // offsets are u32. Results beyond that cannot be handled.
                        ensure!(d.data.len() + buf.len() <= u32::MAX as usize, LimitExceeded);
                        d.push(&buf);
                    } else {
                        d.push_empty();
                        set_null(&mut bad, i, n);
                    }
                }
            }
        }
        PhysType::Bool => {
            if let Data::Bool(d) = &mut data {
                for i in 0..n {
                    let v = if live(i) { eval_bool(id, &A { v: args, s: &s, i })? } else { None };
                    match v {
                        Some(x) => d.push(x),
                        None => {
                            d.push(false);
                            if live(i) {
                                set_null(&mut bad, i, n);
                            }
                        }
                    }
                }
            }
        }
        PhysType::F64 => {
            if let Data::F64(d) = &mut data {
                for i in 0..n {
                    let v = if live(i) { eval_f64(id, &A { v: args, s: &s, i })? } else { None };
                    match v {
                        Some(x) => d.push(x),
                        None => {
                            d.push(0.0);
                            if live(i) {
                                set_null(&mut bad, i, n);
                            }
                        }
                    }
                }
            }
        }
        PhysType::I32 | PhysType::I64 => {
            for i in 0..n {
                let v = if live(i) {
                    eval_int(id, &A { v: args, s: &s, i }, result_ty)?
                } else {
                    Some(0)
                };
                // For I32 output (DATE), out of range also makes just that row NULL.
                let ok = match v {
                    Some(x) => push_int(&mut data, x, result_ty),
                    None => {
                        push_int(&mut data, 0, result_ty);
                        false
                    }
                };
                if !ok && live(i) {
                    set_null(&mut bad, i, n);
                }
            }
        }
        // HUGEINT output (currently only `factorial`). Structurally the
        // same row loop as `I32 | I64` above, but there's no "range check
        // narrows to I32" concern (HUGEINT is the widest integer type), and
        // a genuine overflow (e.g. `factorial(34)`) is a hard query error
        // via `?` — propagated out of `eval_i128`, not turned into a
        // per-row NULL — matching this engine's existing convention for
        // `SUM` overflow (`exec::agg`'s `ValueOutOfRange`, see
        // `docs/sql/limitations.md`'s rounding-conventions note: "integer
        // arithmetic overflow wraps ... except SUM"; `factorial` joins
        // `SUM` as the other place overflow is a hard error rather than a
        // silent wrap).
        PhysType::I128 => {
            if let Data::I128(d) = &mut data {
                for i in 0..n {
                    if !live(i) {
                        d.push(0);
                        continue;
                    }
                    match eval_i128(id, &A { v: args, s: &s, i }, result_ty)? {
                        Some(x) => d.push(x),
                        None => {
                            d.push(0);
                            set_null(&mut bad, i, n);
                        }
                    }
                }
            }
        }
    }
    let mut out = Vector::from_data(result_ty, data, merge(valid, bad));
    out.compact_validity();
    Ok(out)
}

// --- The row loop's underpinnings ---------------------------------------------

/// The row count and each argument's stride. An argument of length 1 gets stride 0 (= a constant).
/// A shared helper used by `expr::regex` too (the same "constant or all the same length" rule).
pub(crate) fn strides(args: &[&Vector]) -> Result<(usize, Vec<usize>)> {
    ensure!(!args.is_empty(), WrongArgCount);
    let mut n = 1usize;
    let mut empty = false;
    for a in args {
        let l = a.len();
        if l == 0 {
            empty = true;
        } else if l > n {
            n = l;
        }
    }
    if empty {
        n = 0;
    }
    let mut s = Vec::with_capacity(args.len());
    for a in args {
        s.push(if a.len() == n {
            1
        } else if a.len() == 1 {
            0
        } else {
            err!(Internal)
        });
    }
    Ok((n, s))
}

/// The AND of every argument's validity. `None` when there are no NULLs. Shared with `expr::regex`.
pub(crate) fn combine(args: &[&Vector], s: &[usize], n: usize) -> Option<Bitmap> {
    if !args.iter().any(|a| a.has_nulls()) {
        return None;
    }
    let mut m = Bitmap::with_capacity(n);
    for i in 0..n {
        let mut ok = true;
        for (k, a) in args.iter().enumerate() {
            ok &= a.is_valid(i * s[k]);
        }
        m.push(ok);
    }
    Some(m)
}

/// Makes row `i` NULL (allocating lazily). `expr::kernels`/`expr::regex` have implementations
/// serving the same role (each kept independently to suit its own call pattern).
pub(crate) fn set_null(m: &mut Option<Bitmap>, i: usize, n: usize) {
    if m.is_none() {
        *m = Some(Bitmap::ones(n));
    }
    if let Some(b) = m {
        b.set(i, false);
    }
}

fn merge(a: Option<Bitmap>, b: Option<Bitmap>) -> Option<Bitmap> {
    match (a, b) {
        (Some(mut x), Some(y)) => {
            x.and_assign(&y);
            Some(x)
        }
        (Some(x), None) => Some(x),
        (None, y) => y,
    }
}

/// Writes an i64 into an integer output. Out of range for DATE (I32) gives `false` (= NULL).
///
/// DuckDB reserves three physical i32 values for DATE special values (the two
/// infinities and the adjacent lower sentinel). AhiruDB does not expose
/// infinity literals, so allowing any sentinel to escape from date arithmetic
/// would turn an out-of-range result into a fictitious finite calendar date.
fn push_int(d: &mut Data, x: i64, ty: Ty) -> bool {
    match d {
        Data::I32(v) => match i32::try_from(x) {
            Ok(z) => {
                if ty == Ty::Date && (z <= i32::MIN + 1 || z == i32::MAX) {
                    v.push(0);
                    return false;
                }
                v.push(z);
                true
            }
            Err(_) => {
                v.push(0);
                false
            }
        },
        Data::I64(v) => {
            v.push(x);
            true
        }
        _ => false,
    }
}

/// The per-row argument accessors. Out of range or a physical type mismatch returns a default.
/// `resolve` guarantees the arguments are correct, but this keeps `call` from panicking even if
/// called directly.
struct A<'a, 'b> {
    v: &'a [&'b Vector],
    s: &'a [usize],
    i: usize,
}

impl<'b> A<'_, 'b> {
    #[inline]
    fn n(&self) -> usize {
        self.v.len()
    }

    #[inline]
    fn at(&self, k: usize) -> Option<(&'b Vector, usize)> {
        match (self.v.get(k), self.s.get(k)) {
            (Some(v), Some(st)) => Some((v, self.i * st)),
            _ => None,
        }
    }

    fn bytes(&self, k: usize) -> &'b [u8] {
        match self.at(k) {
            Some((v, j)) if matches!(v.data(), Data::Bytes(_)) && j < v.len() => v.bytes().get(j),
            _ => &[],
        }
    }

    fn int(&self, k: usize) -> i64 {
        match self.at(k) {
            Some((v, j)) => match v.data() {
                Data::I64(d) => d.get(j).copied().unwrap_or(0),
                Data::I32(d) => d.get(j).copied().unwrap_or(0) as i64,
                _ => 0,
            },
            None => 0,
        }
    }

    /// The DECIMAL scale of argument `k`'s logical type (0 for anything else). Read off the
    /// argument vector rather than passed in: `resolve` has already aligned every argument to
    /// one common type, so this is the scale the whole call works at.
    fn dec_scale(&self, k: usize) -> u32 {
        match self.at(k) {
            Some((v, _)) => match v.ty() {
                Ty::Decimal { scale, .. } => scale as u32,
                _ => 0,
            },
            None => 0,
        }
    }

    /// The full-width integer value. Needed by the few functions that must see HUGEINT/UBIGINT
    /// patterns rather than the BIGINT-shaped view `int` gives (`hex`/`to_hex`).
    fn i128(&self, k: usize) -> i128 {
        match self.at(k) {
            Some((v, j)) => match v.data() {
                Data::I128(d) => d.get(j).copied().unwrap_or(0),
                Data::I64(d) => d.get(j).copied().unwrap_or(0) as i128,
                Data::I32(d) => d.get(j).copied().unwrap_or(0) as i128,
                _ => 0,
            },
            None => 0,
        }
    }

    fn flt(&self, k: usize) -> f64 {
        match self.at(k) {
            Some((v, j)) => match v.data() {
                Data::F64(d) => d.get(j).copied().unwrap_or(0.0),
                _ => 0.0,
            },
            None => 0.0,
        }
    }
}

// --- Submodules --------------------------------------------------------
//
// This file stays a thin entry point: constants, `resolve`, `call`, and the
// `A` argument bundle. The per-physical-type eval bodies (`eval_str`/
// `eval_bool`/`eval_int`/`eval_f64`) and their supporting helpers live in
// `expr::funcs::*` submodules instead (same split strategy as
// `sql::parser` -> `sql::parser/`).

mod bool_ops;
mod datetime;
mod json;
mod lambda;
mod numeric;
mod string;

#[cfg(test)]
mod tests;

// `call()` above dispatches into these submodule bodies.
use bool_ops::eval_bool;
use json::{
    concat_all, concat_ws_build, extremum, fold_null, json_array_build, json_object_build,
    list_concat_build, nullif, regexp_full_match_build,
};
use numeric::{eval_f64, eval_i128, eval_int};
use string::eval_str;

// Other modules (`expr::kernels`/`expr::regex`/`write::csv`/`write::jsonl`/
// `parquet::nested`/`plan::compile`/`expr::vm`) reach these items through
// the `funcs::X` path that existed before this split. Re-export them here
// so that path keeps resolving — only the implementation moved, not the
// public name/path.
pub(crate) use datetime::{
    add_interval_to_ts, days_from_civil, fmt_date, fmt_time, fmt_timestamp, fmt_timestamptz,
    fmt_uuid, parse_date, parse_time, parse_timestamp, parse_timestamptz, parse_uuid,
};
// This one is used only from `write::csv`/`write::jsonl`. `write` itself exists only with
// `export`, and the csv/jsonl inside it are gated further by their own features, so it is
// re-exported only when both hold (with `export` alone neither sink exists and it would warn as
// unused).
#[cfg(all(feature = "export", any(feature = "csv", feature = "jsonl")))]
pub(crate) use datetime::civil_from_days;
pub use lambda::call_lambda;
pub(crate) use numeric::{f_abs, f_trunc};
