//! `CURRENT_DATE`/`CURRENT_TIMESTAMP`/`CURRENT_TIME`/`now()`/`today()` の
//! 束縛前置換。
//!
//! wasm コアは時計を持たない（「ホストでできることはホストでやる」という
//! このエンジン全体の方針そのもの、DESIGN.md §2）。クエリ開始時刻は
//! ホストが `Session::set_now` で一度だけ渡し、束縛前にこのモジュールが
//! 構文木を書き換える。書き換え後はただの定数（`Expr::TypedLiteral`）に
//! なるので、`plan::bind`/`plan::compile` は特別扱いを一切要らない。
//!
//! `CURRENT_TIMESTAMP` は SQL 標準では時刻を 1 回だけ評価する（クエリ内の
//! 全行で同じ値になる）契約なので、構文木を書き換える方式はこの意味論と
//! 自然に一致する。行ごとに再評価される心配がない。

use crate::rt::hash::eq_ascii_ci;
use crate::sql::ast::{Expr, ExprArena};
use crate::vector::{Ty, Value};

const MICROS_PER_DAY: i64 = 86_400_000_000;

enum Kind {
    Date,
    Timestamp,
    Time,
}

/// 裸の識別子（`CURRENT_DATE` のように括弧無し）として書ける名前。
/// SQL 標準で実質予約語相当のこの 3 つだけに絞る。`now`/`today` のような
/// 関数形は必ず `()` を伴うので、列名との衝突が原理的に起きない
/// （過去の ROWS/RANGE/QUALIFY 事故は「括弧を伴わない裸の識別子」を
/// 安易にキーワード扱いしたことが原因だったので、ここは慎重にする）。
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

/// `name()` の形（引数無しの関数呼び出し）でしか現れない名前。括弧を伴う
/// ので列名との衝突が起きない分、`bare_kind` より広く受け付けてよい。
fn call_kind(name: &str) -> Option<Kind> {
    let b = name.as_bytes();
    if eq_ascii_ci(b, b"current_date") {
        Some(Kind::Date)
    } else if eq_ascii_ci(b, b"today") {
        Some(Kind::Date)
    } else if eq_ascii_ci(b, b"current_timestamp") || eq_ascii_ci(b, b"now") {
        Some(Kind::Timestamp)
    } else if eq_ascii_ci(b, b"current_time") {
        Some(Kind::Time)
    } else {
        None
    }
}

/// `arena` 内の該当ノードを `now_micros`（エポックからのマイクロ秒、UTC）
/// を使った定数に置き換える。DuckDB の `CURRENT_TIMESTAMP`/`now()` は
/// `TIMESTAMP WITH TIME ZONE` を返すが、このエンジンにタイムゾーン型は
/// 無いので `Ty::Timestamp`（UTC のマイクロ秒）に落とす簡略化をしている。
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
                Kind::Timestamp => Expr::TypedLiteral(Value::I64(now_micros), Ty::Timestamp),
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

    // `Expr` は導出 Clone を持たない（サブクエリを含むと木ごと複製に
    // なりかねないため）ので、テスト専用に軽い複製だけ許す。
    impl Expr {
        fn clone_for_test(&self) -> Expr {
            match self {
                Expr::TypedLiteral(v, ty) => Expr::TypedLiteral(v.clone(), *ty),
                Expr::ColumnRef { qualifier, name } => {
                    Expr::ColumnRef { qualifier: qualifier.clone(), name: name.clone() }
                }
                _ => panic!("clone_for_test: 未対応の形"),
            }
        }
    }

    // 2024-01-15 12:30:00 UTC (エポック秒 1705321800) をマイクロ秒で。
    const T: i64 = 1_705_321_800_000_000;

    #[test]
    fn bare_current_date_and_timestamp_become_typed_literals() {
        assert!(matches!(
            literal_of("SELECT CURRENT_DATE", T),
            Expr::TypedLiteral(Value::I32(19737), Ty::Date)
        ));
        assert!(matches!(
            literal_of("SELECT CURRENT_TIMESTAMP", T),
            Expr::TypedLiteral(Value::I64(t), Ty::Timestamp) if t == T
        ));
    }

    #[test]
    fn call_forms_become_typed_literals() {
        assert!(matches!(
            literal_of("SELECT now()", T),
            Expr::TypedLiteral(Value::I64(t), Ty::Timestamp) if t == T
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
        // 括弧を伴わない裸の識別子は current_date/current_timestamp/
        // current_time の 3 つだけを対象にする。他の名前（列名でありうる
        // もの）は触らない。
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
