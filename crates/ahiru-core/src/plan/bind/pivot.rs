//! `PIVOT`/`UNPIVOT` desugaring into plain `SELECT` statements. This runs as
//! an AST-level rewrite before binding proper (see the module doc below for
//! why), so it lives alongside the rest of the binder but does not call into
//! `bind_select_in` at all.

use super::refs::each_child;
use super::*;
use crate::sql::ast::{PivotStmt, SelectItem, UnpivotStmt};

// --- PIVOT/UNPIVOT -----------------------------------------------------------
//
// どちらも「計画レベルの構文糖衣展開」として実装する（GROUPING SETS が複数の
// `Node::Aggregate` を `Node::SetOp` で束ねた方針と同じ発想）。ただし
// GROUPING SETS と違い、展開後の形が既存の「`GROUP BY` + 集約 + `FILTER`」
// （PIVOT）や「射影 + `UNION ALL`」（UNPIVOT）とちょうど同じ AST 形に落ちる
// ので、束縛（`bind_select_in`）そのものには一切手を入れず、AST レベルの
// 書き換えだけで完結させる。呼び出し元は `Session::prepare`
// （PIVOT/UNPIVOT の `Stmt` を検出した直後、対象表のスキーマを解決してから
// 呼ぶ）。展開結果は通常の `Stmt::Select` として、そのまま既存の
// `prepare_query` に渡る。
//
// アリーナは束縛時点では不変参照（`&ExprArena`）なので、新しい式ノード
// （`on = 値` の等号や、`FILTER` 付き集約呼び出し）を束縛の最中に作ることは
// できない。そのため展開はパース直後・束縛前の段階で行い、
// `substitute_now`（`sql::now`）と同じように `&mut ExprArena` へ新規ノードを
// 積む。

/// `IN (...)` で明示された 1 つの PIVOT 値がとりうるリテラル型の上限。
/// 文字列・整数・真偽値のみ対応（列名の文字列化に `core::fmt` を使えない
/// ため、浮動小数点数は非対応）。
fn pivot_value_to_column_name(v: &Value) -> Result<String> {
    match v {
        Value::Bytes(b) => match core::str::from_utf8(b) {
            Ok(s) => Ok(String::from(s)),
            Err(_) => err!(UnsupportedFeature),
        },
        Value::Bool(x) => Ok(String::from(if *x { "true" } else { "false" })),
        Value::I32(n) => Ok(i128_to_decimal_string(*n as i128)),
        Value::I64(n) => Ok(i128_to_decimal_string(*n as i128)),
        Value::I128(n) => Ok(i128_to_decimal_string(*n)),
        Value::Null | Value::F64(_) => err!(UnsupportedFeature),
    }
}

/// `core::fmt` を使わない符号付き 10 進数文字列化。`push_u32` の `i128` 版。
fn i128_to_decimal_string(v: i128) -> String {
    if v == 0 {
        return String::from("0");
    }
    let neg = v < 0;
    let mut uv = v.unsigned_abs();
    let mut buf = [0u8; 40];
    let mut n = 0;
    while uv > 0 {
        buf[n] = b'0' + (uv % 10) as u8;
        uv /= 10;
        n += 1;
    }
    let mut s = String::new();
    if neg {
        s.push('-');
    }
    for i in (0..n).rev() {
        s.push(buf[i] as char);
    }
    s
}

/// 式 `id` が（再帰的に）参照する裸の列名をすべて集める。GROUP BY 省略時の
/// デフォルト列（`on`/`using` が参照する列以外の全列）を決めるのに使う。
/// サブクエリ・ウィンドウの中は `each_child` がそもそも辿らない
/// （モジュール doc 参照）ので、それらの中の列参照は対象外になる —
/// PIVOT の `ON`/`USING` にサブクエリやウィンドウ関数が来ることは想定して
/// いないので実害はない。
fn collect_colref_names(
    arena: &ExprArena,
    id: ExprId,
    out: &mut Vec<String>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    if let Expr::ColumnRef { name, .. } = arena.get(id) {
        out.push(name.clone());
        return Ok(());
    }
    let d = depth + 1;
    each_child(arena, id, &mut |c| collect_colref_names(arena, c, out, d))
}

/// `PIVOT` を通常の `SELECT ... GROUP BY ...` へ展開する。
///
/// `from_schema` は対象表（`stmt.from`）の列一覧。`stmt.group_by` が空の
/// ときだけ使う（`on`/`using` が参照する列を除いた残り全部が既定の
/// `GROUP BY` 対象になる — DuckDB と同じ規則）。実データは一切読まない。
pub fn desugar_pivot(
    arena: &mut ExprArena,
    stmt: PivotStmt,
    from_schema: &[Field],
) -> Result<QueryStmt> {
    let PivotStmt { from, on, in_list, using, group_by, order_by, limit, offset } = stmt;

    // 値の自動検出（`IN` 省略）には対象列の実データの DISTINCT が要る。
    // 束縛時点ではスキーマしか読めていない（`Session::prepare` がスキーマ
    // 解決の直後にこの関数を呼ぶ）ので非対応（モジュール doc / `PivotStmt`
    // doc 参照）。
    let in_list = match in_list {
        Some(v) => v,
        None => err!(UnsupportedFeature),
    };
    ensure!(!in_list.is_empty(), SyntaxError);
    ensure!(in_list.len() <= MAX_PIVOT_VALUES, ExpressionTooDeep);

    // `USING` 省略時は DuckDB と同じく既定で `count(*)`。
    let using = if using.is_empty() {
        let f = arena.push(Expr::Function {
            name: String::from("count"),
            args: Vec::new(),
            distinct: false,
            star: true,
            filter: None,
        });
        vec![SelectItem { expr: f, alias: None }]
    } else {
        using
    };
    // 複数集約関数（`USING sum(a), avg(b)`）は列名（`a_sum(a)` 方式）の決定に
    // 式の文字列化が要り、`core::fmt` 禁止の下でのコストに見合わないため
    // 対応しない（`PivotStmt::using` doc 参照）。
    ensure!(using.len() == 1, UnsupportedFeature);
    let (agg_name, agg_args, agg_distinct, agg_star) = match arena.get(using[0].expr) {
        Expr::Function { name, args, distinct, star, filter } => {
            ensure!(filter.is_none(), UnsupportedFeature);
            (name.clone(), args.clone(), *distinct, *star)
        }
        _ => err!(SyntaxError),
    };

    // GROUP BY 対象列。明示が無ければ「on/using が参照する列以外の全列」。
    let group_by_exprs: Vec<ExprId> = if !group_by.is_empty() {
        group_by
    } else {
        let mut excluded = Vec::new();
        collect_colref_names(arena, on, &mut excluded, 0)?;
        for a in &agg_args {
            collect_colref_names(arena, *a, &mut excluded, 0)?;
        }
        from_schema
            .iter()
            .filter(|f| !excluded.iter().any(|e| eq_ascii_ci(e.as_bytes(), f.name.as_bytes())))
            .map(|f| arena.push(Expr::ColumnRef { qualifier: None, name: f.name.clone() }))
            .collect()
    };

    let mut items: Vec<SelectItem> =
        group_by_exprs.iter().map(|&expr| SelectItem { expr, alias: None }).collect();

    // 値ごとに `agg(...) FILTER (WHERE on = 値)` を 1 列作る。既存の
    // `FILTER (WHERE cond)` 付き集約の仕組み（`Expr::Function.filter`、
    // `exec::agg::Agg.filter`）にそのまま乗るので、集約束縛ロジックには
    // 手を入れていない。
    //
    // `IN (...)` に同じ値が 2 回以上現れると（別名の有無に関わらず）
    // `duckdb` は "The value ... was specified multiple times in the IN
    // clause" として拒否する（`duckdb -c "PIVOT ... ON category IN
    // ('a','a') ..."` で確認済み）。ここでチェックしないと、同じ
    // `FILTER` 条件を持つ列が重複した名前で 2 本できてしまう
    // （既知の不具合。`star_exclude_replace.rs` の EXCLUDE/REPLACE 重複
    // チェックや `WINDOW` 句の重複名チェックと同じ判断で、値ベースの
    // 重複だけを拒む — 別名が衝突しても `duckdb` は `_1` サフィックスで
    // 自動的にリネームするだけでエラーにしないので、そちらは追わない）。
    let mut seen_values: Vec<Value> = Vec::with_capacity(in_list.len());
    for (val_expr, alias) in &in_list {
        let lit = match arena.get(*val_expr) {
            Expr::Literal(v) => v.clone(),
            // `TypedLiteral`/式一般は非対応。列名も定数畳み込みも要るため。
            _ => err!(UnsupportedFeature),
        };
        ensure!(!seen_values.contains(&lit), SyntaxError);
        seen_values.push(lit.clone());
        let col_name = match alias {
            Some(a) => a.clone(),
            None => pivot_value_to_column_name(&lit)?,
        };
        let lit_expr = arena.push(Expr::Literal(lit));
        let pred = arena.push(Expr::Binary { op: BinaryOp::Eq, lhs: on, rhs: lit_expr });
        let f = arena.push(Expr::Function {
            name: agg_name.clone(),
            args: agg_args.clone(),
            distinct: agg_distinct,
            star: agg_star,
            filter: Some(pred),
        });
        items.push(SelectItem { expr: f, alias: Some(col_name) });
    }

    Ok(QueryStmt {
        ctes: Vec::new(),
        body: SetExpr::Select(Box::new(SelectStmt {
            items,
            from: Some(from),
            group_by: group_by_exprs,
            ..SelectStmt::empty()
        })),
        // `bind_query_in` の「外側の ORDER BY/LIMIT」経路（列名・序数参照のみ
        // 対応）にそのまま乗せる。展開後の本体は素の単一 SELECT だが、こちら
        // に置いても意味は変わらない（`UNPIVOT` 側は `SetOp` になるので
        // 同じ置き場所で揃えてある）。
        order_by,
        // `ORDER BY ALL` は `PIVOT`/`UNPIVOT` ではパーサが拒否済み
        // （`sql::parser::Parser::pivot_stmt` 参照）。
        order_by_all: None,
        limit,
        offset,
    })
}

/// `PIVOT` の `IN (...)` に書ける値の個数の上限。`CUBE`/`ROLLUP` の
/// `MAX_CUBE_COLS`（`sql::parser`）と同じ「作りすぎを防ぐ安全弁」の役割。
const MAX_PIVOT_VALUES: usize = 128;

/// `UNPIVOT` で同時に畳み込める対象列数の上限。
const MAX_UNPIVOT_COLUMNS: usize = 128;

/// `FromItem` を複製する。PIVOT/UNPIVOT の `from` は `Table`/`File` に
/// しか対応しない（`UNPIVOT` は対象列数ぶん同じ `from` を複製して
/// `UNION ALL` の各枝に配る必要があり、`Subquery`/`Join` はプラン
/// （`Node`）を複製できないため単純にコピーできない。`plan::bind::resolve_from`
/// が `DESCRIBE` 向けに課している制約と同じ）。
fn clone_from_item(f: &FromItem) -> Result<FromItem> {
    match f {
        FromItem::Table { name, alias } => {
            Ok(FromItem::Table { name: name.clone(), alias: alias.clone() })
        }
        FromItem::File { path, format, alias } => {
            Ok(FromItem::File { path: path.clone(), format: *format, alias: alias.clone() })
        }
        // データを持たない計算だけのソースなので、他の派生表と違って複製に
        // 実コストが無い。
        FromItem::GenerateSeries { start, stop, step, inclusive, alias, column_alias } => {
            Ok(FromItem::GenerateSeries {
                start: *start,
                stop: *stop,
                step: *step,
                inclusive: *inclusive,
                alias: alias.clone(),
                column_alias: column_alias.clone(),
            })
        }
        FromItem::Join { .. } | FromItem::Subquery { .. } | FromItem::Unnest { .. } => {
            err!(UnsupportedFeature)
        }
    }
}

/// `UNPIVOT` を `UNION ALL` へ展開する。
///
/// 対象列 1 本につき「対象列以外をそのまま通し、対象列名を文字列リテラル
/// として `name_col` に、対象列の値を `value_col` に出す」`SELECT` を 1 本
/// 作り、全部を `UNION ALL` で束ねる。`from_schema` は対象表の列一覧
/// （素通しする「対象列以外」を決めるのに使う。実データは読まない）。
pub fn desugar_unpivot(
    arena: &mut ExprArena,
    stmt: UnpivotStmt,
    from_schema: &[Field],
) -> Result<QueryStmt> {
    let UnpivotStmt { from, columns, name_col, value_col, order_by, limit, offset } = stmt;
    ensure!(!columns.is_empty(), SyntaxError);
    ensure!(columns.len() <= MAX_UNPIVOT_COLUMNS, ExpressionTooDeep);

    // 対象は修飾子なしの裸の列参照のみ（`UnpivotStmt` doc 参照）。
    let mut target_names: Vec<String> = Vec::with_capacity(columns.len());
    for &c in &columns {
        match arena.get(c) {
            Expr::ColumnRef { qualifier: None, name } => target_names.push(name.clone()),
            _ => err!(UnsupportedFeature),
        }
    }

    // 対象列以外はそのまま素通しする（DuckDB の既定と同じ）。
    let other_names: Vec<String> = from_schema
        .iter()
        .map(|f| f.name.clone())
        .filter(|n| !target_names.iter().any(|t| eq_ascii_ci(t.as_bytes(), n.as_bytes())))
        .collect();

    let mut branches: Vec<SetExpr> = Vec::with_capacity(target_names.len());
    for name in &target_names {
        let mut items: Vec<SelectItem> = Vec::with_capacity(other_names.len() + 2);
        for n in &other_names {
            let e = arena.push(Expr::ColumnRef { qualifier: None, name: n.clone() });
            items.push(SelectItem { expr: e, alias: None });
        }
        let name_lit = arena.push(Expr::Literal(Value::Bytes(name.clone().into_bytes())));
        items.push(SelectItem { expr: name_lit, alias: Some(name_col.clone()) });
        let val_ref = arena.push(Expr::ColumnRef { qualifier: None, name: name.clone() });
        items.push(SelectItem { expr: val_ref, alias: Some(value_col.clone()) });

        let sel = SelectStmt { items, from: Some(clone_from_item(&from)?), ..SelectStmt::empty() };
        branches.push(SetExpr::Select(Box::new(sel)));
    }

    // 左結合の `UNION ALL` 連鎖に束ねる。GROUPING SETS が `Node::SetOp` で
    // 複数の `Node::Aggregate` を束ねたのと同じ発想（モジュール doc 参照）。
    let mut iter = branches.into_iter();
    let mut body = match iter.next() {
        Some(b) => b,
        None => err!(Internal), // columns が空でない限り起きない
    };
    for b in iter {
        body = SetExpr::SetOp {
            op: SetOp::Union,
            all: true,
            left: Box::new(body),
            right: Box::new(b),
        };
    }

    Ok(QueryStmt { ctes: Vec::new(), body, order_by, order_by_all: None, limit, offset })
}
