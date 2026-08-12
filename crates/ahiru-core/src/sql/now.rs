//! Pre-binding substitution of
//! `CURRENT_DATE`/`CURRENT_TIMESTAMP`/`CURRENT_TIME`/`now()`/`today()`.
//!
//! The wasm core has no clock (exactly this engine's overall policy of "do on the host
//! what the host can do", DESIGN.md §2). The host passes the query start time once via
//! `Session::set_now`, and this module rewrites the syntax tree before binding. After
//! the rewrite they are plain constants (`Expr::TypedLiteral`), so `plan::bind`/
//! `plan::compile` need no special handling at all.
//!
//! The SQL standard contracts `CURRENT_TIMESTAMP` to be evaluated exactly once (the
//! same value for every row in the query), so rewriting the syntax tree agrees with
//! that semantics naturally. There is no risk of re-evaluation per row.

use crate::rt::hash::eq_ascii_ci;
use crate::sql::ast::{Expr, ExprArena};
use crate::vector::{Ty, Value};

const MICROS_PER_DAY: i64 = 86_400_000_000;

enum Kind {
    Date,
    Timestamp,
    Time,
}

/// Names writable as a bare identifier (without parentheses, as in `CURRENT_DATE`).
/// Limited to these three, which are effectively reserved in the SQL standard. Function
/// forms such as `now`/`today` always carry `()`, so they cannot in principle collide
/// with a column name (the past ROWS/RANGE/QUALIFY incidents were caused by casually
/// treating "bare identifiers without parentheses" as keywords, so this stays careful).
fn bare_kind(name: &str) -> Option<Kind> {
    let b = name.as_bytes();
    if eq_ascii_ci(b, b"current_date") {
        Some(Kind::Date)
    } else if eq_ascii_ci(b, b"current_timestamp") {
        Some(Kind::Timestamp)
    } else if eq_ascii_ci(b, b"current_time") {
        Some(Kind::Time)
    } else {
        None
    }
}

/// Names that only appear in the `name()` form (a call with no arguments). Carrying
/// parentheses means no collision with column names, so this can be broader than `bare_kind`.
fn call_kind(name: &str) -> Option<Kind> {
    let b = name.as_bytes();
    if eq_ascii_ci(b, b"current_date") || eq_ascii_ci(b, b"today") {
        Some(Kind::Date)
    } else if eq_ascii_ci(b, b"current_timestamp") || eq_ascii_ci(b, b"now") {
        Some(Kind::Timestamp)
    } else if eq_ascii_ci(b, b"current_time") {
        Some(Kind::Time)
    } else {
        None
    }
}

/// Replaces the corresponding node in `arena` with a constant built from `now_micros`
/// (microseconds since the epoch, UTC). Like DuckDB's `CURRENT_TIMESTAMP`/`now()`, it
/// returns `Ty::Timestamptz` (physically the same UTC microseconds as `Ty::Timestamp`,
/// but `now_micros` originates from the host's wall clock as a UTC instant, so
/// `Timestamptz` is also the semantically accurate choice).
pub fn substitute_now(arena: &mut ExprArena, now_micros: i64) {
    let date_days = now_micros.div_euclid(MICROS_PER_DAY) as i32;
    let time_micros = now_micros.rem_euclid(MICROS_PER_DAY);
    for node in arena.iter_mut() {
        let kind = match node {
            Expr::ColumnRef { qualifier: None, name } => bare_kind(name),
            Expr::Function { name, args, distinct: false, star: false, filter: None }
                if args.is_empty() =>
            {
                call_kind(name)
            }
            _ => None,
        };
        if let Some(kind) = kind {
            *node = match kind {
                Kind::Date => Expr::TypedLiteral(Value::I32(date_days), Ty::Date),
                Kind::Timestamp => Expr::TypedLiteral(Value::I64(now_micros), Ty::Timestamptz),
                Kind::Time => Expr::TypedLiteral(Value::I64(time_micros), Ty::Time),
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::parser::parse;

    fn literal_of(sql: &str, now_micros: i64) -> Expr {
        let mut p = parse(sql).unwrap();
        substitute_now(&mut p.arena, now_micros);
        let crate::sql::ast::Stmt::Select(q) = &p.stmt else { panic!("expected SELECT") };
        let crate::sql::ast::SetExpr::Select(sel) = &q.body else { panic!("expected select") };
        p.arena.get(sel.items[0].expr).clone_for_test()
    }

    // `Expr` does not derive Clone (with subqueries involved, cloning one expression
    // could clone a whole tree), so only a lightweight test-only copy is permitted.
    impl Expr {
        fn clone_for_test(&self) -> Expr {
            match self {
                Expr::TypedLiteral(v, ty) => Expr::TypedLiteral(v.clone(), *ty),
                Expr::ColumnRef { qualifier, name } => {
                    Expr::ColumnRef { qualifier: qualifier.clone(), name: name.clone() }
                }
                _ => panic!("clone_for_test: unsupported shape"),
            }
        }
    }

    // 2024-01-15 12:30:00 UTC (epoch second 1705321800) in microseconds.
    const T: i64 = 1_705_321_800_000_000;

    #[test]
    fn bare_current_date_and_timestamp_become_typed_literals() {
        assert!(matches!(
            literal_of("SELECT CURRENT_DATE", T),
            Expr::TypedLiteral(Value::I32(19737), Ty::Date)
        ));
        assert!(matches!(
            literal_of("SELECT CURRENT_TIMESTAMP", T),
            Expr::TypedLiteral(Value::I64(t), Ty::Timestamptz) if t == T
        ));
    }

    #[test]
    fn call_forms_become_typed_literals() {
        assert!(matches!(
            literal_of("SELECT now()", T),
            Expr::TypedLiteral(Value::I64(t), Ty::Timestamptz) if t == T
        ));
        assert!(matches!(
            literal_of("SELECT today()", T),
            Expr::TypedLiteral(Value::I32(19737), Ty::Date)
        ));
        assert!(matches!(
            literal_of("SELECT current_time", T),
            Expr::TypedLiteral(Value::I64(micros), Ty::Time)
                if micros == 12 * 3_600_000_000 + 30 * 60_000_000
        ));
    }

    #[test]
    fn unrelated_column_names_are_left_alone() {
        // Only current_date/current_timestamp/current_time are targeted as bare
        // identifiers without parentheses. Other names (which could be column names)
        // are left alone.
        assert!(matches!(
            literal_of("SELECT today", T),
            Expr::ColumnRef { qualifier: None, name } if name == "today"
        ));
        assert!(matches!(
            literal_of("SELECT now", T),
            Expr::ColumnRef { qualifier: None, name } if name == "now"
        ));
    }
}
