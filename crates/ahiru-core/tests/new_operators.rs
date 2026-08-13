//! Integration tests for the operators added to close common SQL-operator
//! gaps versus DuckDB: `IS [NOT] DISTINCT FROM`, `::` (CAST shorthand),
//! `^`/`**` (power), `&`/`|`/`<<`/`>>`/prefix `~` (bitwise), infix
//! `~`/`!~` (regex match, an alias for `SIMILAR TO`'s desugaring),
//! `~~`/`!~~`/`~~*`/`!~~*`/`~~~` (LIKE/ILIKE/GLOB punctuation aliases),
//! `IS [NOT] TRUE/FALSE`, `ISNULL`/`NOTNULL`, `//` (integer division), `@`
//! (absolute value), and `!` (factorial, and the new `factorial()`
//! function it desugars to).
//!
//! Parser-level desugaring is already covered by unit tests in
//! `crates/ahiru-core/src/sql/parser/tests.rs` (`distinct_from_desugars_to_null_safe_equality`,
//! `cast_shorthand_desugars_to_cast`, `power_operator_desugars_to_pow`,
//! `bitwise_operators_desugar_to_bit_functions`,
//! `tilde_operators_desugar_to_regexp_full_match`, `like_alias_operators_desugar_to_like`,
//! `glob_alias_operator_desugars_to_glob_call`, `is_true_desugars_to_cast_and_coalesce`,
//! `is_false_desugars_to_negated_cast_and_coalesce`, `isnull_notnull_desugar_to_is_null`,
//! `integer_division_operator_desugars_to_div`, `at_prefix_desugars_to_abs`,
//! `factorial_postfix_desugars_to_factorial_call`); this file checks the
//! actual evaluated results end to end, including NULL propagation and
//! operators applied to real table columns (constant-only expressions take
//! a different code path in this engine). Expected values are
//! cross-checked against a real `duckdb` CLI.

use ahiru_core::error::{code_of, Code};
use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn session_with_dual() -> Session {
    let mut s = Session::new();
    s.register_bytes_as("dual", b"x\n1\n".to_vec(), FormatKind::Csv).unwrap();
    s
}

fn run(s: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    let mut q = match s.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
    };
    let mut rows = Vec::new();
    loop {
        match s.step(&mut q).unwrap() {
            QueryStep::Batch(mut b) => {
                b.materialize();
                for r in 0..b.num_rows() {
                    rows.push(b.cols.iter().map(|c| c.value_at(r)).collect());
                }
            }
            QueryStep::Done => break,
            QueryStep::NeedIo(_) | QueryStep::NeedCodec(_) => panic!("unexpected suspend"),
        }
    }
    rows
}

// --- IS [NOT] DISTINCT FROM ---------------------------------------------------

/// duckdb: `SELECT a, b, a IS DISTINCT FROM b FROM (VALUES (1,1),(1,2),
///          (1,NULL),(NULL,NULL)) t(a,b)` -> false, true, true, false
#[test]
fn distinct_from_treats_null_as_a_comparable_value() {
    let mut db = Session::new();
    db.register_bytes_as("t", b"a,b\n1,1\n1,2\n1,\n,\n".to_vec(), FormatKind::Csv).unwrap();
    let rows = run(&mut db, "SELECT a IS DISTINCT FROM b FROM t");
    assert_eq!(
        rows,
        vec![
            vec![Value::Bool(false)],
            vec![Value::Bool(true)],
            vec![Value::Bool(true)],
            vec![Value::Bool(false)],
        ]
    );
}

#[test]
fn not_distinct_from_is_the_negation() {
    let mut db = Session::new();
    db.register_bytes_as("t", b"a,b\n1,1\n1,2\n1,\n,\n".to_vec(), FormatKind::Csv).unwrap();
    let rows = run(&mut db, "SELECT a IS NOT DISTINCT FROM b FROM t");
    assert_eq!(
        rows,
        vec![
            vec![Value::Bool(true)],
            vec![Value::Bool(false)],
            vec![Value::Bool(false)],
            vec![Value::Bool(true)],
        ]
    );
}

#[test]
fn distinct_from_can_be_used_in_where() {
    // A practical use: find rows that changed, treating NULL-vs-NULL as
    // "unchanged" (unlike plain `<>`, which would leave those rows out with
    // UNKNOWN instead of FALSE).
    let mut db = Session::new();
    db.register_bytes_as("t", b"id,a,b\n1,1,1\n2,1,2\n3,1,\n4,,\n".to_vec(), FormatKind::Csv)
        .unwrap();
    let rows = run(&mut db, "SELECT id FROM t WHERE a IS DISTINCT FROM b ORDER BY id");
    assert_eq!(rows, vec![vec![Value::I64(2)], vec![Value::I64(3)]]);
}

// --- :: cast shorthand ---------------------------------------------------------

#[test]
fn cast_shorthand_matches_cast() {
    let mut db = session_with_dual();
    let a = run(&mut db, "SELECT '42'::INTEGER FROM dual");
    let b = run(&mut db, "SELECT CAST('42' AS INTEGER) FROM dual");
    assert_eq!(a, b);
    assert_eq!(a, vec![vec![Value::I32(42)]]);
}

#[test]
fn cast_string_fraction_rounds_like_numeric_cast() {
    let mut db = session_with_dual();
    let from_str = run(&mut db, "SELECT CAST('1.5' AS INTEGER) FROM dual");
    let from_num = run(&mut db, "SELECT CAST(1.5 AS INTEGER) FROM dual");
    assert_eq!(from_str, from_num);
    assert_eq!(from_str, vec![vec![Value::I32(2)]]);
    let dec = run(&mut db, "SELECT CAST('1.25' AS DECIMAL(10,1)) FROM dual");
    assert_eq!(dec, vec![vec![Value::I64(13)]]);
}

#[test]
fn between_on_interval_or_json_is_type_mismatch() {
    let mut db = session_with_dual();
    assert_eq!(
        code_of(db.prepare("SELECT 1 FROM dual WHERE INTERVAL '1' DAY BETWEEN INTERVAL '0' DAY AND INTERVAL '2' DAY", &[])),
        Some(Code::TypeMismatch)
    );
    assert_eq!(
        code_of(db.prepare("SELECT 1 FROM dual WHERE CAST('[1]' AS JSON) BETWEEN CAST('[0]' AS JSON) AND CAST('[2]' AS JSON)", &[])),
        Some(Code::TypeMismatch)
    );
}

#[test]
fn cast_shorthand_chains() {
    let mut db = session_with_dual();
    let rows = run(&mut db, "SELECT 3.7::INTEGER::VARCHAR FROM dual");
    assert_eq!(rows, vec![vec![Value::Bytes(b"4".to_vec())]]); // round-half-to-even
}

#[test]
fn cast_shorthand_binds_tighter_than_unary_minus() {
    // duckdb: SELECT -1::VARCHAR -> '-1' (parses as -(1::VARCHAR))
    let mut db = session_with_dual();
    let rows = run(&mut db, "SELECT -1::VARCHAR FROM dual");
    assert_eq!(rows, vec![vec![Value::Bytes(b"-1".to_vec())]]);
}

// --- ^ / ** power ---------------------------------------------------------------

#[test]
fn power_operators_match_pow_function() {
    let mut db = session_with_dual();
    // duckdb: SELECT 2^10, 2**10, power(2,10) -> 1024.0 (double) for all three
    let rows = run(&mut db, "SELECT 2 ^ 10, 2 ** 10, power(2, 10) FROM dual");
    assert_eq!(rows, vec![vec![Value::F64(1024.0), Value::F64(1024.0), Value::F64(1024.0)]]);
}

#[test]
fn power_is_left_associative() {
    // duckdb: SELECT 2^3^2 -> 64.0, not 512.0
    let mut db = session_with_dual();
    let rows = run(&mut db, "SELECT 2 ^ 3 ^ 2 FROM dual");
    assert_eq!(rows, vec![vec![Value::F64(64.0)]]);
}

// --- bitwise operators ----------------------------------------------------------

#[test]
fn bitwise_and_or_shift_match_duckdb() {
    let mut db = session_with_dual();
    // duckdb: SELECT 5 & 3, 5 | 2, 1 << 4, 16 >> 2, ~5 -> 1, 7, 16, 4, -6
    let rows = run(&mut db, "SELECT 5 & 3, 5 | 2, 1 << 4, 16 >> 2, ~5 FROM dual");
    assert_eq!(
        rows,
        vec![vec![Value::I64(1), Value::I64(7), Value::I64(16), Value::I64(4), Value::I64(-6),]]
    );
}

#[test]
fn bitwise_precedence_matches_duckdb() {
    let mut db = session_with_dual();
    // duckdb: SELECT 1 + 2 & 3 -> (1+2)&3 = 3 (bitwise binds looser than +)
    let rows = run(&mut db, "SELECT 1 + 2 & 3 FROM dual");
    assert_eq!(rows, vec![vec![Value::I64(3)]]);
    // duckdb: SELECT 1 & 2 = 0 -> (1&2)=0 = true (bitwise binds tighter than =)
    let rows = run(&mut db, "SELECT 1 & 2 = 0 FROM dual");
    assert_eq!(rows, vec![vec![Value::Bool(true)]]);
}

#[test]
fn bit_not_round_trips() {
    // duckdb: SELECT ~(-1), ~(0) -> 0, -1
    let mut db = session_with_dual();
    let rows = run(&mut db, "SELECT ~(-1), ~0 FROM dual");
    assert_eq!(rows, vec![vec![Value::I64(0), Value::I64(-1)]]);
}

#[test]
fn shift_by_an_out_of_range_amount_is_null_not_an_error() {
    // duckdb errors here ("Left-shift value 64 is out of range"); this
    // engine instead returns NULL, consistent with its existing convention
    // for other undefined integer arithmetic (division by zero, etc. --
    // see docs/sql/types.md's rounding-conventions section).
    let mut db = session_with_dual();
    let rows = run(&mut db, "SELECT 1 << 64, 1 << -1, 1 >> 64 FROM dual");
    assert_eq!(rows, vec![vec![Value::Null, Value::Null, Value::Null]]);
}

#[test]
fn bitwise_functions_are_directly_callable_by_name() {
    let mut db = session_with_dual();
    let rows =
        run(&mut db, "SELECT bit_and(5,3), bit_or(5,2), bit_shift_left(1,4), bit_shift_right(16,2), bit_not(5) FROM dual");
    assert_eq!(
        rows,
        vec![vec![Value::I64(1), Value::I64(7), Value::I64(16), Value::I64(4), Value::I64(-6),]]
    );
}

// --- infix ~ / !~ (regex match) --------------------------------------------------

#[test]
fn tilde_operators_match_regexp_full_match() {
    let mut db = session_with_dual();
    // duckdb: SELECT 'abc' ~ 'a.c', 'abc' !~ 'xyz' -> true, true
    let rows = run(&mut db, "SELECT 'abc' ~ 'a.c', 'abc' !~ 'xyz' FROM dual");
    assert_eq!(rows, vec![vec![Value::Bool(true), Value::Bool(true)]]);
}

#[test]
fn tilde_is_a_full_match_not_a_substring_search() {
    // `~` desugars to regexp_full_match (anchored), same as SIMILAR TO --
    // not a substring search the way LIKE '%...%' would be.
    let mut db = session_with_dual();
    let rows = run(&mut db, "SELECT 'xabcx' ~ 'abc' FROM dual");
    assert_eq!(rows, vec![vec![Value::Bool(false)]]);
    let rows = run(&mut db, "SELECT 'xabcx' ~ '.*abc.*' FROM dual");
    assert_eq!(rows, vec![vec![Value::Bool(true)]]);
}

#[test]
fn tilde_on_table_data() {
    let mut db = Session::new();
    db.register_bytes_as("t", b"id,name\n1,alice\n2,bob\n3,alicia\n".to_vec(), FormatKind::Csv)
        .unwrap();
    let rows = run(&mut db, "SELECT id FROM t WHERE name ~ 'ali.*' ORDER BY id");
    assert_eq!(rows, vec![vec![Value::I64(1)], vec![Value::I64(3)]]);
}

/// Runs `sql` expecting it to fail (either at `prepare` or during `step`)
/// and returns the numeric error code. Panics if the query instead
/// succeeds, since a silently-wrong answer is exactly the bug class these
/// operators were added to close (see the module doc / `docs/sql/
/// limitations.md`'s "fail loudly" preamble).
fn run_err(s: &mut Session, sql: &str) -> u16 {
    let mut q = match s.prepare(sql, &[]) {
        Err(e) => return e.code_u16(),
        Ok(Prepared::NeedIo(_)) => panic!("{sql}: unexpected NeedIo"),
        Ok(Prepared::Ready(q)) => q,
    };
    loop {
        match s.step(&mut q) {
            Err(e) => return e.code_u16(),
            Ok(QueryStep::Batch(_)) => continue,
            Ok(QueryStep::Done) => panic!("{sql}: expected an error but the query succeeded"),
            Ok(QueryStep::NeedIo(_) | QueryStep::NeedCodec(_)) => {
                panic!("{sql}: unexpected suspend")
            }
        }
    }
}

// --- ~~ / !~~ / ~~* / !~~* (LIKE/ILIKE punctuation aliases) --------------

#[test]
fn like_alias_operators_match_like_and_ilike_on_literals() {
    // duckdb: 'abc' ~~ 'a%' -> true, 'abc' !~~ 'a%' -> false,
    //         'ABC' ~~* 'a%' -> true, 'ABC' !~~* 'a%' -> false
    let mut db = session_with_dual();
    let rows = run(
        &mut db,
        "SELECT 'abc' ~~ 'a%', 'abc' !~~ 'a%', 'ABC' ~~* 'a%', 'ABC' !~~* 'a%' FROM dual",
    );
    assert_eq!(
        rows,
        vec![vec![Value::Bool(true), Value::Bool(false), Value::Bool(true), Value::Bool(false),]]
    );
}

#[test]
fn like_alias_null_propagates() {
    // duckdb: SELECT NULL ~~ 'a' -> NULL
    let mut db = session_with_dual();
    let rows = run(&mut db, "SELECT NULL ~~ 'a' FROM dual");
    assert_eq!(rows, vec![vec![Value::Null]]);
}

#[test]
fn like_alias_operators_on_table_data() {
    let mut db = Session::new();
    db.register_bytes_as("t", b"id,name\n1,alice\n2,bob\n3,\n".to_vec(), FormatKind::Csv).unwrap();
    let rows = run(&mut db, "SELECT id, name ~~ 'al%', name !~~ 'al%' FROM t ORDER BY id");
    assert_eq!(
        rows,
        vec![
            vec![Value::I64(1), Value::Bool(true), Value::Bool(false)],
            vec![Value::I64(2), Value::Bool(false), Value::Bool(true)],
            vec![Value::I64(3), Value::Null, Value::Null],
        ]
    );
}

#[test]
fn like_alias_operators_bind_tighter_than_concat() {
    // Real, verified divergence from the `LIKE` keyword's own precedence
    // (see the parser-level test with the same name in `sql/parser/tests.rs`
    // for the desugared trees): `duckdb -c "select 'ab' LIKE 'a' || 'b'"`
    // -> `true` (pattern absorbs the `||`), but `duckdb -c "select 'ab' ~~
    // 'a' || 'b'"` -> `'falseb'` (the `~~` result concatenates with `'b'`).
    let mut db = session_with_dual();
    let rows = run(&mut db, "SELECT 'ab' LIKE 'a' || 'b' FROM dual");
    assert_eq!(rows, vec![vec![Value::Bool(true)]]);
    let rows = run(&mut db, "SELECT 'ab' ~~ 'a' || 'b' FROM dual");
    assert_eq!(rows, vec![vec![Value::Bytes(b"falseb".to_vec())]]);
}

// --- ~~~ (GLOB punctuation alias) -----------------------------------------

#[test]
fn glob_alias_matches_glob_on_literals_and_table_data() {
    // duckdb: SELECT 'ab' ~~~ 'a*' -> true
    let mut db = Session::new();
    db.register_bytes_as("t", b"id,name\n1,alice\n2,bob\n".to_vec(), FormatKind::Csv).unwrap();
    let rows = run(&mut db, "SELECT id FROM t WHERE name ~~~ 'al*' ORDER BY id");
    assert_eq!(rows, vec![vec![Value::I64(1)]]);
}

// --- IS [NOT] TRUE / IS [NOT] FALSE ---------------------------------------

#[test]
fn is_true_and_is_false_match_duckdb_on_literals() {
    // duckdb: 3 IS TRUE -> true, 0 IS TRUE -> false, NULL IS TRUE -> false,
    //         NULL IS NOT TRUE -> true, NULL IS FALSE -> false,
    //         NULL IS NOT FALSE -> true, true IS FALSE -> false,
    //         false IS FALSE -> true
    let mut db = session_with_dual();
    let rows = run(
        &mut db,
        "SELECT 3 IS TRUE, 0 IS TRUE, NULL IS TRUE, NULL IS NOT TRUE, \
         NULL IS FALSE, NULL IS NOT FALSE, true IS FALSE, false IS FALSE FROM dual",
    );
    assert_eq!(
        rows,
        vec![vec![
            Value::Bool(true),
            Value::Bool(false),
            Value::Bool(false),
            Value::Bool(true),
            Value::Bool(false),
            Value::Bool(true),
            Value::Bool(false),
            Value::Bool(true),
        ]]
    );
}

#[test]
fn is_true_and_is_false_never_return_null_on_table_data() {
    // `IS TRUE`/`IS FALSE` always produce TRUE/FALSE, never NULL -- unlike
    // a plain `= true` comparison, which would be NULL for a NULL row.
    let mut db = Session::new();
    db.register_bytes_as("t", b"id,n\n1,3\n2,0\n3,\n".to_vec(), FormatKind::Csv).unwrap();
    let rows = run(&mut db, "SELECT id, n IS TRUE, n IS FALSE FROM t ORDER BY id");
    assert_eq!(
        rows,
        vec![
            vec![Value::I64(1), Value::Bool(true), Value::Bool(false)],
            vec![Value::I64(2), Value::Bool(false), Value::Bool(true)],
            vec![Value::I64(3), Value::Bool(false), Value::Bool(false)],
        ]
    );
}

// --- ISNULL / NOTNULL postfix ---------------------------------------------

#[test]
fn isnull_notnull_match_duckdb_on_literals() {
    // duckdb: 1 ISNULL -> false, NULL ISNULL -> true, NULL NOTNULL -> false
    let mut db = session_with_dual();
    let rows = run(&mut db, "SELECT 1 ISNULL, NULL ISNULL, NULL NOTNULL FROM dual");
    assert_eq!(rows, vec![vec![Value::Bool(false), Value::Bool(true), Value::Bool(false)]]);
}

#[test]
fn isnull_notnull_on_table_data() {
    let mut db = Session::new();
    db.register_bytes_as("t", b"id,n\n1,3\n2,\n".to_vec(), FormatKind::Csv).unwrap();
    let rows = run(&mut db, "SELECT id FROM t WHERE n ISNULL ORDER BY id");
    assert_eq!(rows, vec![vec![Value::I64(2)]]);
    let rows = run(&mut db, "SELECT id FROM t WHERE n NOTNULL ORDER BY id");
    assert_eq!(rows, vec![vec![Value::I64(1)]]);
}

#[test]
fn isnull_stays_usable_as_a_column_name() {
    // `SELECT isnull FROM t` must still resolve `isnull` as a column
    // reference, and `SELECT 1 AS isnull` must still alias a column to
    // that name -- these are soft keywords, not reserved words.
    let mut db = Session::new();
    db.register_bytes_as("t", b"isnull,x\n5,1\n".to_vec(), FormatKind::Csv).unwrap();
    let rows = run(&mut db, "SELECT isnull FROM t");
    assert_eq!(rows, vec![vec![Value::I64(5)]]);
    let mut dual = session_with_dual();
    let rows = run(&mut dual, "SELECT 1 AS isnull FROM dual");
    assert_eq!(rows, vec![vec![Value::I32(1)]]);
}

// --- // integer division ---------------------------------------------------

#[test]
fn integer_division_matches_duckdb_on_literals() {
    // duckdb: 5//2=2, -5//2=-2, 5//-2=-2, -7//2=-3, 5.0//2=2.5, 5//0=NULL
    let mut db = session_with_dual();
    let rows = run(&mut db, "SELECT 5 // 2, -5 // 2, 5 // -2, -7 // 2, 5.0 // 2, 5 // 0 FROM dual");
    assert_eq!(
        rows,
        vec![vec![
            Value::I32(2),
            Value::I32(-2),
            Value::I32(-2),
            Value::I32(-3),
            Value::F64(2.5),
            Value::Null,
        ]]
    );
}

#[test]
fn integer_division_on_table_data_including_divide_by_zero() {
    let mut db = Session::new();
    db.register_bytes_as("t", b"id,n,d\n1,7,2\n2,7,0\n3,,2\n".to_vec(), FormatKind::Csv).unwrap();
    let rows = run(&mut db, "SELECT id, n // d FROM t ORDER BY id");
    assert_eq!(
        rows,
        vec![
            vec![Value::I64(1), Value::I64(3)],
            vec![Value::I64(2), Value::Null],
            vec![Value::I64(3), Value::Null],
        ]
    );
}

// --- @ absolute value --------------------------------------------------------

#[test]
fn at_prefix_matches_abs_on_literals_and_table_data() {
    // duckdb: @(-5) -> 5, @(-5.5) -> 5.5
    let mut db = Session::new();
    db.register_bytes_as("t", b"id,n\n1,-5\n2,5\n3,\n".to_vec(), FormatKind::Csv).unwrap();
    let rows = run(&mut db, "SELECT id, @n FROM t ORDER BY id");
    assert_eq!(
        rows,
        vec![
            vec![Value::I64(1), Value::I64(5)],
            vec![Value::I64(2), Value::I64(5)],
            vec![Value::I64(3), Value::Null],
        ]
    );
    let mut dual = session_with_dual();
    let rows = run(&mut dual, "SELECT @(-5.5) FROM dual");
    assert_eq!(rows, vec![vec![Value::F64(5.5)]]);
}

// --- ! postfix factorial / factorial() --------------------------------------

#[test]
fn factorial_matches_duckdb_on_literals() {
    // duckdb: 4! -> 24, 0! -> 1, (2+2)! -> 24, -4! -> 1
    let mut db = session_with_dual();
    let rows = run(&mut db, "SELECT 4!, 0!, (2 + 2)!, -4! FROM dual");
    assert_eq!(rows, vec![vec![Value::I128(24), Value::I128(1), Value::I128(24), Value::I128(1)]]);
}

#[test]
fn factorial_on_table_data_including_null() {
    let mut db = Session::new();
    db.register_bytes_as("t", b"id,n\n1,5\n2,0\n3,\n".to_vec(), FormatKind::Csv).unwrap();
    let rows = run(&mut db, "SELECT id, n!, factorial(n) FROM t ORDER BY id");
    assert_eq!(
        rows,
        vec![
            vec![Value::I64(1), Value::I128(120), Value::I128(120)],
            vec![Value::I64(2), Value::I128(1), Value::I128(1)],
            vec![Value::I64(3), Value::Null, Value::Null],
        ]
    );
}

#[test]
fn factorial_of_33_is_the_largest_that_fits_and_34_overflows() {
    // duckdb: factorial(33) = 8683317618811886495518194401280000000,
    // factorial(34) -> "Out of Range Error" (34! ~= 2.95e38 > i128::MAX).
    let mut db = session_with_dual();
    let rows = run(&mut db, "SELECT factorial(33) FROM dual");
    assert_eq!(rows, vec![vec![Value::I128(8_683_317_618_811_886_495_518_194_401_280_000_000)]]);
    assert_eq!(run_err(&mut db, "SELECT factorial(34) FROM dual"), 503); // Code::ValueOutOfRange
}

// --- `!` precedence: the `-x!` column-vs-literal bug and its fix ---------
//
// An earlier version of this parser folded `!` into `primary()`'s postfix
// loop (same strength as `::`/`[...]`), which made `-4!` (a negative
// *literal*) desugar correctly by accident -- `prefix()`'s `Tok::Minus`
// arm has a fast path that folds a negative integer literal before `!`
// ever gets a chance to apply, so `-4!` became `factorial(-4)` = `1`,
// matching duckdb. But `-x!` for a non-literal operand (a column, a
// parenthesized expression, ...) went through the *general* unary-minus
// path instead, which builds `Unary::Neg(factorial(x))` = `-(x!)` = `-24`
// -- the same syntax parsing to a different answer depending only on
// whether the operand happened to be a literal. `!` now lives in
// `expr_body`'s infix loop at its own precedence, `BP_BANG` (strictly
// between every binary operator and the prefix operators `-`/`~`/`NOT`,
// all of which read their operand at `BP_UNARY`) -- see `BP_BANG`'s doc in
// `sql::parser` for the full rationale. That fixes both paths uniformly:
// `!` is left alone while a prefix operator reads its operand, so it ends
// up applying to the *result* of the prefix operator, not the bare operand.

#[test]
fn factorial_applies_after_unary_minus_for_columns_too() {
    // duckdb: SELECT -x! FROM t (x=4) -> 1, exactly matching `-4!` -- this
    // is the literal-vs-column divergence described above, now fixed.
    let mut db = Session::new();
    db.register_bytes_as("t", b"x\n4\n".to_vec(), FormatKind::Csv).unwrap();
    let rows = run(&mut db, "SELECT -x! FROM t");
    assert_eq!(rows, vec![vec![Value::I128(1)]]);
}

#[test]
fn factorial_applies_after_bitwise_not_for_literals_and_columns() {
    // duckdb: SELECT ~5! -> 1, matching `(~5)!` (`~(5!)` would be `-121`).
    // Prefix `~` reads its operand at `BP_UNARY` exactly like unary `-`
    // does, so the same rule applies here too.
    let mut db = session_with_dual();
    let rows = run(&mut db, "SELECT ~5! FROM dual");
    assert_eq!(rows, vec![vec![Value::I128(1)]]);
    db.register_bytes_as("t", b"x\n5\n".to_vec(), FormatKind::Csv).unwrap();
    let rows = run(&mut db, "SELECT ~x! FROM t");
    assert_eq!(rows, vec![vec![Value::I128(1)]]);
}

#[test]
fn factorial_and_cast_shorthand_interleave_on_columns() {
    // duckdb: SELECT x!::VARCHAR FROM t (x=4) -> '24'. `!` no longer joins
    // `primary`'s postfix loop, so this `::` is picked up by an explicit
    // `cast_postfix` call made right after `expr_body` folds the `!`.
    let mut db = Session::new();
    db.register_bytes_as("t", b"x\n4\n".to_vec(), FormatKind::Csv).unwrap();
    let rows = run(&mut db, "SELECT x!::VARCHAR FROM t");
    assert_eq!(rows, vec![vec![Value::Bytes(b"24".to_vec())]]);
}

#[test]
fn factorial_precedence_matches_duckdb_where_duckdb_accepts_it() {
    // duckdb: 3! ^ 2 -> 36.0, 3! = 6 -> true. Both are cases where
    // DuckDB's own (internally inconsistent) grammar happens to accept
    // the expression, and we match it exactly here -- on both literal and
    // column operands.
    let mut db = session_with_dual();
    let rows = run(&mut db, "SELECT 3! ^ 2, 3! = 6 FROM dual");
    assert_eq!(rows, vec![vec![Value::F64(36.0), Value::Bool(true)]]);
    let mut t = Session::new();
    t.register_bytes_as("t", b"x\n3\n".to_vec(), FormatKind::Csv).unwrap();
    let rows = run(&mut t, "SELECT x! ^ 2, x! = 6 FROM t");
    assert_eq!(rows, vec![vec![Value::F64(36.0), Value::Bool(true)]]);
}

#[test]
fn factorial_precedence_deliberately_diverges_from_duckdb_on_binary_operators() {
    // duckdb parses `2 + 3!` as `(2+3)!` = `120` and rejects `3! + 1`
    // outright as a syntax error (its postfix `!` grammar is internally
    // inconsistent -- see docs/sql/limitations.md for the full writeup,
    // including the `3! ^ 2` works / `2 ^ 3!` errors pair that proves it).
    // Here `!` always binds to just the immediately preceding operand, on
    // both literal and column operands, giving `2 + 3!` = `2 + (3!)` = `8`
    // and `3! + 1` = `(3!) + 1` = `7` instead.
    let mut db = session_with_dual();
    let rows = run(&mut db, "SELECT 2 + 3!, 3! + 1 FROM dual");
    assert_eq!(rows, vec![vec![Value::I128(8), Value::I128(7)]]);
    let mut t = Session::new();
    t.register_bytes_as("t", b"x\n3\n".to_vec(), FormatKind::Csv).unwrap();
    let rows = run(&mut t, "SELECT 2 + x!, x! + 1 FROM t");
    assert_eq!(rows, vec![vec![Value::I128(8), Value::I128(7)]]);
}
