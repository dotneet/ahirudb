//! Integration tests for the operators added to close common SQL-operator
//! gaps versus DuckDB: `IS [NOT] DISTINCT FROM`, `::` (CAST shorthand),
//! `^`/`**` (power), `&`/`|`/`<<`/`>>`/prefix `~` (bitwise), and infix
//! `~`/`!~` (regex match, an alias for `SIMILAR TO`'s desugaring).
//!
//! Parser-level desugaring is already covered by unit tests in
//! `crates/ahiru-core/src/sql/parser.rs` (`distinct_from_desugars_to_null_safe_equality`,
//! `cast_shorthand_desugars_to_cast`, `power_operator_desugars_to_pow`,
//! `bitwise_operators_desugar_to_bit_functions`,
//! `tilde_operators_desugar_to_regexp_full_match`); this file checks the
//! actual evaluated results end to end. Expected values are cross-checked
//! against a real `duckdb` CLI.

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
