//! Subquery decorrelation into joins: `EXISTS`/`IN` -> semi/anti joins
//! (`build_semijoin`), scalar-subquery correlation-key extraction
//! (`correlation_keys`), quantified `ANY`/`ALL` comparison desugaring
//! (`build_quantified_comparison`), and the WHERE-clause conjunct
//! classification used to detect correlated equality predicates
//! (`classify_conjunct`/`ConjClass`).

use super::cte::CteScope;
use super::refs::{collect_refs, each_child, push_u32};
use super::*;

/// `EXISTS` / `IN (SELECT ...)` を半結合・反結合に書き換える。
#[allow(clippy::too_many_arguments)]
pub(super) fn build_semijoin(
    catalog: &Catalog,
    arena: &ExprArena,
    params: &[Value],
    ctes: &mut CteScope,
    left: Node,
    scope: &Scope,
    subs: &[Substitution],
    pred: ExprId,
) -> Result<Node> {
    let schema = scope.fields().to_vec();
    match arena.get(pred) {
        Expr::Exists { query, negated } => {
            // `scope` はこの `EXISTS`/`IN` を含むクエリ自身のスコープなので、
            // そのまま「1 段外側」として渡す（相関は 1 階層のみ対応。
            // これより深い相関は内側の `scope` に存在しない列参照になり、
            // 自然に `ColumnNotFound` で失敗する）。
            let plan = bind_query_in(catalog, arena, query, params, ctes, Some(scope))?;
            // 相関が無ければ「右が 1 行でもあるか」だけが問題でキーは要らない。
            // 相関があれば、相関キーが一致する行が 1 つでもあるかを見る。
            // `EXISTS` は集合ではなく行の存在を問うだけなので、`IN`/`NOT IN`
            // のような NULL の三値論理の落とし穴が無く、通常の
            // Semi/Anti（NULL キーは互いに一致しない）でそのまま正しい
            // （DuckDB で確認済み）。
            let kind = if *negated { JoinKind::Anti } else { JoinKind::Semi };
            let (left_keys, right_keys) = correlation_keys(arena, scope, params, &plan)?;
            Ok(Node::Join {
                left: Box::new(left),
                right: Box::new(plan.root),
                kind,
                left_keys,
                right_keys,
                residual: None,
                schema,
            })
        }
        Expr::InSubquery { arg, query, negated } => build_in_style_semijoin(
            catalog, arena, params, ctes, left, scope, subs, schema, *arg, query, *negated,
        ),
        // `= ANY (SELECT ...)` / `<> ALL (SELECT ...)` は `IN`/`NOT IN` と
        // 意味的に全く同じ（`is_semijoin_predicate` の doc 参照）。他の演算子・
        // `SOME`/`ALL` の組み合わせはここには来ない（`is_semijoin_predicate` が
        // 弾いている）。
        Expr::QuantifiedComparison { op, arg, all: _, query } => {
            let negated = matches!(op, BinaryOp::Ne);
            build_in_style_semijoin(
                catalog, arena, params, ctes, left, scope, subs, schema, *arg, query, negated,
            )
        }
        _ => err!(Internal),
    }
}

/// `x [NOT] IN (SELECT ...)` 形の半結合・反結合を組み立てる。`InSubquery` と
/// `= ANY` / `<> ALL` に書き換わった `QuantifiedComparison` の両方から呼ばれる
/// 共通実装（`build_semijoin` 参照）。
#[allow(clippy::too_many_arguments)]
fn build_in_style_semijoin(
    catalog: &Catalog,
    arena: &ExprArena,
    params: &[Value],
    ctes: &mut CteScope,
    left: Node,
    scope: &Scope,
    subs: &[Substitution],
    schema: Vec<Field>,
    arg: ExprId,
    query: &QueryStmt,
    negated: bool,
) -> Result<Node> {
    let plan = bind_query_in(catalog, arena, query, params, ctes, Some(scope))?;
    let k = plan.correlated.len();
    ensure!(plan.root.schema().len() - k == 1, TypeMismatch);
    let rf = plan.root.schema()[0].clone();

    let lp = compile_with_subs(arena, scope, params, subs, arg)?;
    let rscope = Scope::from_fields(plan.root.schema().to_vec());
    let rp = column_program(&rscope, 0)?;
    // キーはバイト列に符号化して突き合わせるので、物理型を揃える。
    let want = Ty::unify_or_mismatch(lp.result_ty, rf.ty)?;
    let mut left_keys = vec![cast_program(lp, want)?];
    let mut right_keys = vec![cast_program(rp, want)?];
    let (corr_left, corr_right) = correlation_keys(arena, scope, params, &plan)?;
    left_keys.extend(corr_left);
    right_keys.extend(corr_right);

    // `NOT IN` は右に NULL が 1 つでもあると三値論理で結果が空になる。
    // 通常の反結合ではこの意味を再現できないので、NULL 対応版
    // （右に NULL キーが 1 つでもあれば無条件に空を返す）を使う。
    // `AntiNullAware` は「右全体に NULL があるか」を見る実装
    // （相関キーごとにスコープされていない）なので、相関がある場合に
    // そのまま使うと、ある外側行の相関先に NULL があるだけで無関係な
    // 別の外側行の結果まで空になりかねない。相関キーごとに絞った
    // null 対応判定を行う実行系のプリミティブが無い以上、相関
    // `NOT IN` は安全に評価できない（対象列の NULL 可能性を束縛時に
    // 正確に判定する手段も無い: SELECT リストの出力列は常に
    // `nullable = true` として扱われるため）。誤った結果を返すより
    // 明確に拒否する。
    ensure!(!(negated && k > 0), UnsupportedFeature);
    let kind = match (negated, rf.nullable) {
        (false, _) => JoinKind::Semi,
        (true, false) => JoinKind::Anti,
        (true, true) => JoinKind::AntiNullAware,
    };
    Ok(Node::Join {
        left: Box::new(left),
        right: Box::new(plan.root),
        kind,
        left_keys,
        right_keys,
        residual: None,
        schema,
    })
}

/// `plan.correlated` の各項目に対応する結合キー `(left, right)` の組を作る。
/// `plan.root` の末尾 `plan.correlated.len()` 列が、`bind_select_in` が
/// 付加した相関キー列（その手前の列数はサブクエリの種類によって違うが、
/// 常に末尾にまとまっている）。相関が無ければ両方とも空を返す。
fn correlation_keys(
    arena: &ExprArena,
    outer_scope: &Scope,
    params: &[Value],
    plan: &Plan,
) -> Result<(Vec<crate::expr::Program>, Vec<crate::expr::Program>)> {
    let k = plan.correlated.len();
    if k == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let rscope = Scope::from_fields(plan.root.schema().to_vec());
    ensure!(rscope.len() >= k, Internal);
    let base = rscope.len() - k;
    let mut left_keys = Vec::with_capacity(k);
    let mut right_keys = Vec::with_capacity(k);
    for (i, &outer_e) in plan.correlated.iter().enumerate() {
        let lp = compile(arena, outer_scope, params, outer_e)?;
        let rp = column_program(&rscope, base + i)?;
        let want = Ty::unify_or_mismatch(lp.result_ty, rp.result_ty)?;
        left_keys.push(cast_program(lp, want)?);
        right_keys.push(cast_program(rp, want)?);
    }
    Ok((left_keys, right_keys))
}

// --- 量化比較（ANY/ALL/SOME、`>`/`<`/`>=`/`<=` 側） --------------------------
//
// `= ANY`/`<> ALL` は `IN`/`NOT IN` と同じ半結合に書き換わる
// （`is_semijoin_predicate`/`build_semijoin` 参照）。残りの 8 通り
// （`<`/`<=`/`>`/`>=` × `ANY`/`ALL`）は「一致する行があるか」ではなく
// 「集合全体との比較」なので半結合には落ちない。代わりに `duckdb` の内部
// 書き換え（`SELECT 5 > ALL(...)` を EXPLAIN すると `NOT (5 <= ANY(...))`
// になることを実測済み）にならい、`x <op> ALL (q)` を
// `NOT (x <negate(op)> ANY (q))` として `ANY` 側 1 通りに帰着させる。
//
// `x <op> ANY (q)` 自体は `MIN`/`MAX` に単純に置き換えるだけでは正しくない
// （空集合・NULL の三値論理が絡む。`ANY` の空集合は常に `FALSE`、`ALL` の
// 空集合は常に `TRUE`、という SQL 標準の落とし穴。モジュールを担当した際に
// `duckdb` CLI の実測で以下の恒真式を確認した）:
//
//   x <op> ANY (q) ==
//     CASE
//       WHEN COUNT(*) = 0        THEN FALSE  -- 空集合
//       WHEN x IS NULL           THEN NULL   -- x が NULL なら全行 UNKNOWN
//       WHEN x <op> extreme(q)   THEN TRUE   -- 非 NULL の実測値だけで判定できる
//       WHEN COUNT(col) < COUNT(*) THEN NULL -- 非 NULL では負けたが NULL 行がある
//       ELSE FALSE
//     END
//
// ここで `extreme` は `op` が `>`/`>=` なら `MIN(col)`、`<`/`<=` なら
// `MAX(col)`（「最も倒しやすい候補」に x を挑ませれば十分、という考え方。
// `x > ANY(q)` は「q のどれか 1 つより大きい」なので、最小値より大きければ
// 必ず存在する）。3 番目の分岐は `extreme` が NULL（＝非 NULL 行が無い）なら
// 比較自体が NULL になり自然に素通りするので、`COUNT(col) > 0` を別途
// 確認する必要は無い。
//
// **対応範囲**: 非相関サブクエリのみ。内側のクエリが外側スコープの列を
// 参照する場合は `UnsupportedFeature` で明確に拒否する（DESIGN.md 冒頭の
// 「黙って壊れるより明確に拒否する」方針）。相関対応は決定的に複雑になる
// （x は外側行ごとに変わるが、`extreme`/`COUNT` は現状 1 度だけ計算する
// 設計なので、相関キーごとの集約に一般化する追加実装が要る）ため、今回は
// 対応を見送った。詳細は `tests/any_all_subquery.rs` のテスト・
// 最終報告を参照。
// `= ALL (q)`/`<> ANY (q)` も同じ理由で対応を見送っている
// （`collect_quantified_comparisons` の doc 参照）:
// この形は「集合の要素がちょうど 1 通りに揃っているか」を問う形で、
// `MIN`/`MAX` 一発にも半結合にも帰着しない。

/// `x <op> ANY (q)` の書き換えに使う `(MAX を使うか, 比較演算子)` を選ぶ。
/// `op` は必ず `Lt`/`Le`/`Gt`/`Ge` のいずれか（呼び出し元で保証）。
fn quantified_any_extreme(op: BinaryOp) -> (bool, BinaryOp) {
    match op {
        BinaryOp::Gt => (false, BinaryOp::Gt), // MIN(q) より大きいか
        BinaryOp::Ge => (false, BinaryOp::Ge), // MIN(q) 以上か
        BinaryOp::Lt => (true, BinaryOp::Lt),  // MAX(q) より小さいか
        BinaryOp::Le => (true, BinaryOp::Le),  // MAX(q) 以下か
        _ => unreachable!("caller only passes order comparisons"),
    }
}

/// 論理否定した比較演算子（`x <op> y` の否定が `x <negate(op)> y` になる方。
/// 左右を入れ替える `BinaryOp::swapped` とは別物）。
/// `x <op> ALL (q)` を `NOT (x <negate(op)> ANY (q))` に落とすために使う。
fn negate_comparison(op: BinaryOp) -> BinaryOp {
    match op {
        BinaryOp::Gt => BinaryOp::Le,
        BinaryOp::Le => BinaryOp::Gt,
        BinaryOp::Lt => BinaryOp::Ge,
        BinaryOp::Ge => BinaryOp::Lt,
        _ => unreachable!("caller only passes order comparisons"),
    }
}

/// 合成した合成名（`__qc{idx}_{suffix}`）。`Scope::resolve` は無修飾で全列を
/// 探すので、ユーザーの列名との衝突を避けるため接頭辞を付ける
/// （既存の `subqN` ラベルと同じ割り切り。完全な衝突回避ではないが、この
/// 接頭辞を実際の列名に使う SQL はまず無い）。
fn quantified_label(idx: usize, suffix: &str) -> String {
    let mut s = String::from("__qc");
    push_u32(&mut s, idx as u32);
    s.push('_');
    s.push_str(suffix);
    s
}

/// `>`/`<`/`>=`/`<=` を伴う量化比較 1 個を、`COUNT(*)`/`COUNT(col)`/
/// `MIN` か `MAX` の 3 集約 1 行と、それを外側の各行に付ける（キー無しの）
/// `LEFT JOIN` へ書き換える。集約は `GROUP BY` 無しなので、`q` が空でも
/// 必ず 1 行返る（`exec::agg` の `empty_input_ungrouped_emits_one_row` と
/// 同じ前提。COUNT は 0、`MIN`/`MAX` は NULL になる）ので、`q` が空集合の
/// ときの分岐もこの 1 行の値だけで CASE 式が正しく判定できる。
///
/// 結合後、3 集約列と x の値を読む小さな `CASE` 式（モジュール冒頭の式）を
/// 別の一時 `ExprArena`（このサブクエリの計算専用。外側の `arena` とは無関係）
/// で組み立て、既存の `compile()` にそのまま通す（比較演算子の型ディスパッチ
/// や `CASE`/`IS NULL` の意味論を再実装しない）。最後に中間列（x・3 集約列）
/// を落として、元の列 + 結果の 1 列だけを残す。
#[allow(clippy::too_many_arguments)]
pub(super) fn build_quantified_comparison(
    catalog: &Catalog,
    arena: &ExprArena,
    params: &[Value],
    ctes: &mut CteScope,
    node: Node,
    scope: &Scope,
    subs: &[Substitution],
    idx: usize,
    op: BinaryOp,
    arg: ExprId,
    all: bool,
    query: &QueryStmt,
) -> Result<Node> {
    let plan = bind_query_in(catalog, arena, query, params, ctes, Some(scope))?;
    // 相関サブクエリは対応範囲外（モジュール冒頭の doc 参照）。
    ensure!(plan.correlated.is_empty(), UnsupportedFeature);
    ensure!(plan.root.schema().len() == 1, TypeMismatch);
    let col_ty = plan.root.schema()[0].ty;

    // `ALL` は `NOT (x <negate(op)> ANY (q))` として `ANY` 側の書き換えに帰着させる。
    let (want_max, cmp_op, wrap_not) = if all {
        let (wm, co) = quantified_any_extreme(negate_comparison(op));
        (wm, co, true)
    } else {
        let (wm, co) = quantified_any_extreme(op);
        (wm, co, false)
    };

    let rscope = Scope::from_fields(plan.root.schema().to_vec());
    let col_prog = column_program(&rscope, 0)?;
    let aggs = vec![
        Agg {
            kind: AggKind::CountStar,
            arg: None,
            distinct: false,
            name: "cnt".into(),
            separator: Vec::new(),
            filter: None,
        },
        Agg {
            kind: AggKind::Count,
            arg: Some(col_prog.clone()),
            distinct: false,
            name: "nonnull".into(),
            separator: Vec::new(),
            filter: None,
        },
        Agg {
            kind: if want_max { AggKind::Max } else { AggKind::Min },
            arg: Some(col_prog),
            distinct: false,
            name: "extreme".into(),
            separator: Vec::new(),
            filter: None,
        },
    ];
    let agg_schema = vec![
        Field::new(quantified_label(idx, "cnt"), Ty::BigInt, false),
        Field::new(quantified_label(idx, "nonnull"), Ty::BigInt, false),
        Field::new(quantified_label(idx, "extreme"), col_ty, true),
    ];
    let agg_node = Node::Aggregate {
        input: Box::new(plan.root),
        groups: Vec::new(),
        aggs,
        schema: agg_schema,
        having: None,
    };

    // x の値を計算し、既存の列すべての後ろに 1 列足す。
    let cur_scope = Scope::from_fields(node.schema().to_vec());
    let orig_len = cur_scope.len();
    let x_prog = compile_with_subs(arena, scope, params, subs, arg)?;
    let x_ty = x_prog.result_ty;
    let x_name = quantified_label(idx, "x");
    let mut exprs = Vec::with_capacity(orig_len + 1);
    for i in 0..orig_len {
        exprs.push(column_program(&cur_scope, i)?);
    }
    exprs.push(x_prog);
    let mut ext_schema = cur_scope.fields().to_vec();
    ext_schema.push(Field::new(x_name, x_ty, true));
    let node = Node::Project { input: Box::new(node), exprs, schema: ext_schema };

    // 集約 1 行をキー無し（＝クロス結合相当）の LEFT JOIN で全行に付ける。
    let mut full_schema = node.schema().to_vec();
    full_schema.extend_from_slice(agg_node.schema());
    let node = Node::Join {
        left: Box::new(node),
        right: Box::new(agg_node),
        kind: JoinKind::Left,
        left_keys: Vec::new(),
        right_keys: Vec::new(),
        residual: None,
        schema: full_schema,
    };

    // 結合後のスキーマ全体を指す一時スコープ上で、CASE 式を組み立てて
    // コンパイルする。末尾 4 列が [x, cnt, nonnull, extreme]。
    let combined_scope = Scope::from_fields(node.schema().to_vec());
    let len = combined_scope.len();
    ensure!(len >= 4, Internal);
    let (x_i, cnt_i, nonnull_i, extreme_i) = (len - 4, len - 3, len - 2, len - 1);

    let mut fa = ExprArena::new();
    let col_ref = |fa: &mut ExprArena, i: usize| {
        fa.push(Expr::ColumnRef { qualifier: None, name: combined_scope.fields()[i].name.clone() })
    };
    let x_ref = col_ref(&mut fa, x_i);
    let cnt_ref = col_ref(&mut fa, cnt_i);
    let nonnull_ref = col_ref(&mut fa, nonnull_i);
    let extreme_ref = col_ref(&mut fa, extreme_i);
    let zero = fa.push(Expr::Literal(Value::I64(0)));
    let cnt_eq0 = fa.push(Expr::Binary { op: BinaryOp::Eq, lhs: cnt_ref, rhs: zero });
    let x_isnull = fa.push(Expr::IsNull { arg: x_ref, negated: false });
    let cmp = fa.push(Expr::Binary { op: cmp_op, lhs: x_ref, rhs: extreme_ref });
    let nonnull_lt_cnt = fa.push(Expr::Binary { op: BinaryOp::Lt, lhs: nonnull_ref, rhs: cnt_ref });
    let v_false_empty = fa.push(Expr::Literal(Value::Bool(false)));
    let v_null_x = fa.push(Expr::Literal(Value::Null));
    let v_true = fa.push(Expr::Literal(Value::Bool(true)));
    let v_null_has_null = fa.push(Expr::Literal(Value::Null));
    let v_false_else = fa.push(Expr::Literal(Value::Bool(false)));
    let case_id = fa.push(Expr::Case {
        operand: None,
        whens: vec![
            (cnt_eq0, v_false_empty),
            (x_isnull, v_null_x),
            (cmp, v_true),
            (nonnull_lt_cnt, v_null_has_null),
        ],
        else_: Some(v_false_else),
    });
    let result_id =
        if wrap_not { fa.push(Expr::Unary { op: UnaryOp::Not, arg: case_id }) } else { case_id };
    let formula = compile(&fa, &combined_scope, params, result_id)?;

    // 中間列（x・cnt・nonnull・extreme）は落とし、元の列 + 結果の 1 列だけ残す。
    let mut exprs2 = Vec::with_capacity(orig_len + 1);
    for i in 0..orig_len {
        exprs2.push(column_program(&combined_scope, i)?);
    }
    exprs2.push(formula);
    let mut out_schema = combined_scope.fields()[..orig_len].to_vec();
    out_schema.push(Field::new(quantified_label(idx, "result"), Ty::Boolean, true));
    Ok(Node::Project { input: Box::new(node), exprs: exprs2, schema: out_schema })
}

// --- 述語の分解 --------------------------------------------------------------

/// AND で連結された述語を要素に分ける。
pub(super) fn split_conjuncts(
    arena: &ExprArena,
    id: ExprId,
    out: &mut Vec<ExprId>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    match arena.get(id) {
        Expr::Binary { op: BinaryOp::And, lhs, rhs } => {
            split_conjuncts(arena, *lhs, out, depth + 1)?;
            split_conjuncts(arena, *rhs, out, depth + 1)?;
        }
        _ => out.push(id),
    }
    Ok(())
}

pub(super) fn and_all(
    arena: &ExprArena,
    scope: &Scope,
    params: &[Value],
    subs: &[Substitution],
    parts: &[ExprId],
) -> Result<crate::expr::Program> {
    ensure!(!parts.is_empty(), Internal);
    let mut prog = compile_predicate_with_subs(arena, scope, params, subs, parts[0])?;
    for &p in &parts[1..] {
        let rhs = compile_predicate_with_subs(arena, scope, params, subs, p)?;
        prog = and_programs(prog, rhs)?;
    }
    Ok(prog)
}

/// 式がサブクエリを含むか。押し下げの可否判定に使う。
pub(super) fn contains_subquery(arena: &ExprArena, id: ExprId, depth: u32) -> bool {
    if depth >= MAX_EXPR_DEPTH {
        return true;
    }
    if matches!(
        arena.get(id),
        Expr::ScalarSubquery(_)
            | Expr::Exists { .. }
            | Expr::InSubquery { .. }
            | Expr::QuantifiedComparison { .. }
    ) {
        return true;
    }
    let mut found = false;
    let _ = each_child(arena, id, &mut |c| {
        if contains_subquery(arena, c, depth + 1) {
            found = true;
        }
        Ok(())
    });
    found
}

/// 述語そのものが `EXISTS` / `IN (SELECT)` か（半結合に書き換えられる形）。
/// `= ANY (SELECT ...)` / `<> ALL (SELECT ...)` は意味的に `IN`/`NOT IN` と
/// 全く同じなので、ここでも同じ半結合として扱う（`build_semijoin` 参照）。
/// `>`/`<`/`>=`/`<=` を伴う量化比較（それ以外の `QuantifiedComparison`）は
/// 半結合には書き換えられない（集合全体との比較なので「一致する行が
/// あるか」だけを見る半結合の形に落ちない）。`bind_select_in` 側の
/// `collect_quantified_comparisons` が別経路で処理する。
pub(super) fn is_semijoin_predicate(arena: &ExprArena, id: ExprId) -> bool {
    match arena.get(id) {
        Expr::Exists { .. } | Expr::InSubquery { .. } => true,
        Expr::QuantifiedComparison { op, all, .. } => {
            matches!((op, all), (BinaryOp::Eq, false) | (BinaryOp::Ne, true))
        }
        _ => false,
    }
}

/// 式の中のスカラサブクエリを集める。
pub(super) fn collect_scalar_subqueries(
    arena: &ExprArena,
    id: ExprId,
    out: &mut Vec<ExprId>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    if matches!(arena.get(id), Expr::ScalarSubquery(_)) {
        if !out.contains(&id) {
            out.push(id);
        }
        return Ok(());
    }
    let d = depth + 1;
    each_child(arena, id, &mut |c| collect_scalar_subqueries(arena, c, out, d))
}

/// 式の中の「集約経由の量化比較」（`>`/`<`/`>=`/`<=` を伴う `ANY`/`ALL`）を
/// 集める。`= ANY`/`<> ALL` は `is_semijoin_predicate` 経由で別に処理される
/// ので、ここでは拾わない（そのまま残ると `build_quantified_comparison` が
/// `MIN`/`MAX` に頼れない演算子で呼ばれてしまう）。`= ALL`/`<> ANY`
/// （どちらも半結合にも `MIN`/`MAX` にも書き換えられない、対応範囲外の
/// 組み合わせ）もここでは拾わない: `bind_select_in` のどの経路にも
/// 引っかからず素通しし、最終的に `plan::compile` の網
/// （`Expr::QuantifiedComparison { .. } => err!(UnsupportedFeature)`）に
/// 落ちて明確なエラーになる。
pub(super) fn collect_quantified_comparisons(
    arena: &ExprArena,
    id: ExprId,
    out: &mut Vec<ExprId>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    if let Expr::QuantifiedComparison { op, .. } = arena.get(id) {
        if matches!(op, BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge)
            && !out.contains(&id)
        {
            out.push(id);
        }
    }
    // `arg` はこの式と同じスコープに属する（`InSubquery` と違って `query` を
    // 持たない代わりに、`arg` の中にさらに別の量化比較が入れ子になりうる:
    // 例 `(a > ANY (q1)) = (b < ANY (q2))`）ので、一致した後も必ず子を辿る
    // （`each_child` は `QuantifiedComparison` については `arg` だけを渡す）。
    let d = depth + 1;
    each_child(arena, id, &mut |c| collect_quantified_comparisons(arena, c, out, d))
}

/// 修飾子の無い列参照ノードを (ノードの ExprId, 名前) の組で集める。
/// QUALIFY の出力別名解決に使う（`each_child` で式木全体を辿るので、
/// どんな深さに書かれていても見つかる）。
pub(super) fn collect_unqualified_colrefs(
    arena: &ExprArena,
    id: ExprId,
    out: &mut Vec<(ExprId, String)>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    if let Expr::ColumnRef { qualifier: None, name } = arena.get(id) {
        out.push((id, name.clone()));
        return Ok(());
    }
    let d = depth + 1;
    each_child(arena, id, &mut |c| collect_unqualified_colrefs(arena, c, out, d))
}

/// この述語が参照する関係が 1 つだけならその添字を返す。
pub(super) fn single_rel_of(
    arena: &ExprArena,
    scope: &Scope,
    ranges: &[(usize, usize)],
    id: ExprId,
) -> Result<Option<usize>> {
    let mut cols = Vec::new();
    // 解決できない参照を含む述語は押し下げない。
    if collect_refs(arena, scope, id, &mut cols).is_err() {
        return Ok(None);
    }
    if cols.is_empty() {
        return Ok(None);
    }
    let mut owner = None;
    for c in cols {
        let r = ranges.iter().position(|(s, e)| c >= *s && c < *e);
        match (owner, r) {
            (_, None) => return Ok(None),
            (None, Some(r)) => owner = Some(r),
            (Some(a), Some(b)) if a == b => {}
            _ => return Ok(None),
        }
    }
    Ok(owner)
}

/// `左の式 = 右の式` の形なら等値結合キーとして取り出す。
pub(super) fn equi_key(
    arena: &ExprArena,
    joined: &Scope,
    left_width: usize,
    id: ExprId,
) -> Result<Option<(ExprId, ExprId)>> {
    let (lhs, rhs) = match arena.get(id) {
        Expr::Binary { op: BinaryOp::Eq, lhs, rhs } => (*lhs, *rhs),
        _ => return Ok(None),
    };
    let mut lc = Vec::new();
    let mut rc = Vec::new();
    if collect_refs(arena, joined, lhs, &mut lc).is_err()
        || collect_refs(arena, joined, rhs, &mut rc).is_err()
    {
        return Ok(None);
    }
    // 定数だけの辺は結合キーにならない（WHERE 相当の条件）。
    if lc.is_empty() || rc.is_empty() {
        return Ok(None);
    }
    let all_left = |v: &[usize]| v.iter().all(|&c| c < left_width);
    let all_right = |v: &[usize]| v.iter().all(|&c| c >= left_width);
    if all_left(&lc) && all_right(&rc) {
        Ok(Some((lhs, rhs)))
    } else if all_right(&lc) && all_left(&rc) {
        Ok(Some((rhs, lhs)))
    } else {
        Ok(None)
    }
}

/// 結合キーの両辺を同じ物理型に揃える。
///
/// `equi_key` は左右をそれぞれ独立にコンパイルするため（`WHERE a.k = d.k`
/// の1本のプログラムとしてコンパイルされる相関等価述語や通常の比較式とは
/// 違い）、`Ty::unify` による暗黙変換が自動では効かない。揃えないと
/// `BIGINT` 列と `DOUBLE` 列のように論理的には比較可能な型同士でも物理表現
/// （`I64` の生ビット列 対 `F64` の生ビット列）が食い違い、ハッシュ結合の
/// キー比較が常に不一致になって行が消える（クラッシュもエラーも起きない
/// ぶん、誤りに気付きにくい）。
pub(super) fn unify_key_types(l: Program, r: Program) -> Result<(Program, Program)> {
    if l.result_ty == r.result_ty {
        return Ok((l, r));
    }
    let t = Ty::unify_or_mismatch(l.result_ty, r.result_ty)?;
    Ok((cast_program(l, t)?, cast_program(r, t)?))
}

// --- 相関サブクエリの WHERE 分類 ---------------------------------------------

/// WHERE の最上位 conjunct（`split_conjuncts` で AND を分解した 1 片）の分類。
pub(super) enum ConjClass {
    /// このクエリ自身のスコープだけで解決できる、普通の述語。
    Local,
    /// `内側の式 = 外側スコープの式` という相関等価述語。
    Correlated { inner: ExprId, outer: ExprId },
}

/// WHERE の 1 つの最上位 conjunct を分類する。
///
/// ローカルスコープだけで解決できるなら `Local`。外側スコープの列を含む
/// `内側 = 外側` の等価述語（両辺どちらも複合式でよい）なら `Correlated`。
/// それ以外の形で外側スコープの列を参照している場合（非等価比較、`OR` の
/// 中、`NOT` で包まれている、など）は結合キーとして取り出せないので
/// `UnsupportedFeature` を返す。不正確な結果を返すより明確に拒否する方針
/// （このプロジェクト全体の方針、DESIGN.md 参照）。
pub(super) fn classify_conjunct(
    arena: &ExprArena,
    local: &Scope,
    outer: &Scope,
    id: ExprId,
) -> Result<ConjClass> {
    // まずローカルだけで解決できるか（＝相関を含まない普通の述語か）を見る。
    let mut tmp = Vec::new();
    if collect_refs(arena, local, id, &mut tmp).is_ok() {
        return Ok(ConjClass::Local);
    }
    if let Expr::Binary { op: BinaryOp::Eq, lhs, rhs } = arena.get(id) {
        let (l, r) = (*lhs, *rhs);
        if is_pure_scope(arena, local, l) && is_pure_outer_only(arena, local, outer, r) {
            return Ok(ConjClass::Correlated { inner: l, outer: r });
        }
        if is_pure_scope(arena, local, r) && is_pure_outer_only(arena, local, outer, l) {
            return Ok(ConjClass::Correlated { inner: r, outer: l });
        }
    }
    // 等価の相関キーとしては取り出せなかった。外側スコープへの参照が
    // どこかに残っているなら、サポート外の相関（非等価・OR の中など）。
    if references_outer(arena, local, outer, id, 0)? {
        err!(UnsupportedFeature);
    }
    // 外側スコープに触れていないなら、相関とは無関係の（別の理由で
    // ローカル解決に失敗した）述語。通常どおりの経路に委ねる。
    Ok(ConjClass::Local)
}

/// 式が丸ごと `scope` だけで解決できるか。
fn is_pure_scope(arena: &ExprArena, scope: &Scope, id: ExprId) -> bool {
    let mut tmp = Vec::new();
    collect_refs(arena, scope, id, &mut tmp).is_ok()
}

/// 式が「ローカルでは解決できないが外側スコープでは解決できる」か
/// （＝純粋に外側だけを参照している）。
fn is_pure_outer_only(arena: &ExprArena, local: &Scope, outer: &Scope, id: ExprId) -> bool {
    if is_pure_scope(arena, local, id) {
        return false;
    }
    is_pure_scope(arena, outer, id)
}

/// 式のどこかに、外側スコープでのみ解決できる列参照があるか。
/// ローカルで解決できる同名列はシャドーイングにより除外する
/// （SQL の名前解決規則: 内側のスコープが外側より優先される）。
fn references_outer(
    arena: &ExprArena,
    local: &Scope,
    outer: &Scope,
    id: ExprId,
    depth: u32,
) -> Result<bool> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    if let Expr::ColumnRef { qualifier, name } = arena.get(id) {
        if local.resolve(qualifier.as_deref(), name).is_ok() {
            return Ok(false);
        }
        return Ok(outer.resolve(qualifier.as_deref(), name).is_ok());
    }
    let mut found = false;
    let d = depth + 1;
    each_child(arena, id, &mut |c| {
        if references_outer(arena, local, outer, c, d)? {
            found = true;
        }
        Ok(())
    })?;
    Ok(found)
}

/// `collect_refs` の相関対応版。ローカルで解決できない列参照が外側スコープで
/// 解決できるなら（相関参照とみなし）黙って無視する。ローカルにも外側にも
/// 無い列参照はこれまでどおりエラーにする。`outer_scope` が `None`（相関で
/// ない通常のクエリ）なら `collect_refs` と完全に同じ挙動。
pub(super) fn collect_refs_tolerant(
    arena: &ExprArena,
    scope: &Scope,
    outer_scope: Option<&Scope>,
    id: ExprId,
    out: &mut Vec<usize>,
) -> Result<()> {
    match outer_scope {
        None => collect_refs(arena, scope, id, out),
        Some(outer) => collect_refs_tolerant_at(arena, scope, outer, id, out, 0),
    }
}

fn collect_refs_tolerant_at(
    arena: &ExprArena,
    scope: &Scope,
    outer_scope: &Scope,
    id: ExprId,
    out: &mut Vec<usize>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    if let Expr::ColumnRef { qualifier, name } = arena.get(id) {
        match scope.resolve(qualifier.as_deref(), name) {
            Ok(i) => out.push(i),
            Err(e) => {
                if outer_scope.resolve(qualifier.as_deref(), name).is_err() {
                    return Err(e);
                }
            }
        }
        return Ok(());
    }
    let d = depth + 1;
    each_child(arena, id, &mut |c| collect_refs_tolerant_at(arena, scope, outer_scope, c, out, d))
}
