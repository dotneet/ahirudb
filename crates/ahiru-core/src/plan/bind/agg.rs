//! Aggregate extraction and rewriting: `GROUP BY`/`GROUPING SETS` support,
//! window-function binding, `UNNEST` element-type narrowing, and the
//! NULL/zero coalescing used when decorrelating correlated `COUNT`
//! aggregates.

use super::refs::{const_program, default_name, each_child};
use super::*;

/// 直近の結合で相関キーとして使った末尾 `k` 列を落とす。
pub(super) fn drop_trailing_columns(node: Node, k: usize) -> Result<Node> {
    let s = Scope::from_fields(node.schema().to_vec());
    ensure!(s.len() >= k, Internal);
    let keep = s.len() - k;
    let mut exprs = Vec::with_capacity(keep);
    let mut schema = Vec::with_capacity(keep);
    for i in 0..keep {
        exprs.push(column_program(&s, i)?);
        schema.push(s.fields()[i].clone());
    }
    Ok(Node::Project { input: Box::new(node), exprs, schema })
}

/// 2 つの `Program` のレジスタ・定数・キャスト表を 1 本に併合し、
/// `Coalesce(a.result, b.result)` を末尾に足す。`and_programs`（`compile.rs`）
/// と同じ併合規則（`compile::merge_program_bodies` を共有）だが、末尾の
/// 合成演算子だけが違う。
fn coalesce_programs(mut a: Program, b: Program) -> Program {
    let ty = a.result_ty;
    let (ra, rb) = crate::plan::compile::merge_program_bodies(&mut a, b);
    let dst = a.alloc_reg();
    a.push(Instr::new(OpCode::Coalesce, ty.phys(), dst, ra, rb));
    a.result = dst;
    a
}

/// `scope` の `i` 列目が NULL なら 0 に補正する `Program` を作る。
/// 相関 GROUP BY 分の疑似グループが存在しない（＝一致する内側行が無い）
/// 場合に `count(*)`/`count(x)` を「0 行を集約 → 0」に補正するために使う
/// （DuckDB で確認済み: 通常の非相関の `count` と同じ結果になるようにする）。
fn coalesce_zero(scope: &Scope, i: usize) -> Result<Program> {
    let ty = scope.fields()[i].ty;
    let col = column_program(scope, i)?;
    let zero = const_program(ty, Value::I64(0));
    Ok(coalesce_programs(col, zero))
}

/// 集約結果ノードの `target` 列（COUNT 系の集約結果）だけを `coalesce_zero`
/// で補正し、他の列（相関 GROUP BY のキー列）はそのまま通す `Project` を被せる。
pub(super) fn coalesce_count_column(node: Node, target: usize) -> Result<Node> {
    let scope = Scope::from_fields(node.schema().to_vec());
    let mut exprs = Vec::with_capacity(scope.len());
    for i in 0..scope.len() {
        if i == target {
            exprs.push(coalesce_zero(&scope, i)?);
        } else {
            exprs.push(column_program(&scope, i)?);
        }
    }
    let schema = scope.fields().to_vec();
    Ok(Node::Project { input: Box::new(node), exprs, schema })
}

// --- 集約の抽出 --------------------------------------------------------------

/// 式の中の集約呼び出しを集める。既出のものは構造の一致で重複を除く。
pub(super) fn collect_aggregates(
    arena: &ExprArena,
    id: ExprId,
    out: &mut Vec<ExprId>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    let d = depth + 1;
    if let Expr::Function { name, args, .. } = arena.get(id) {
        if AggKind::from_name(name).is_some() {
            // 集約の入れ子は SQL として無効。
            for a in args {
                let mut nested = Vec::new();
                collect_aggregates(arena, *a, &mut nested, d)?;
                ensure!(nested.is_empty(), NotAggregate);
            }
            if !out.iter().any(|&e| expr_eq(arena, e, id)) {
                out.push(id);
            }
            return Ok(());
        }
    }
    each_child(arena, id, &mut |c| collect_aggregates(arena, c, out, d))
}

/// `GROUPING(col, ...)` / `GROUPING_ID(col, ...)`。
///
/// 集約関数とは違って引数を評価しない（グルーピングセットごとに定まる
/// ビットマスク定数に置き換わるだけ）ので `collect_aggregates` とは
/// 別に集める。
fn is_grouping_fn(name: &str) -> bool {
    eq_ascii_ci(name.as_bytes(), b"grouping") || eq_ascii_ci(name.as_bytes(), b"grouping_id")
}

/// 式の中の `GROUPING`/`GROUPING_ID` 呼び出しを集める。入れ子は不正
/// （集約と同じ理由: 引数の中で自分自身が現れることは意味を持たない）。
pub(super) fn collect_grouping_calls(
    arena: &ExprArena,
    id: ExprId,
    out: &mut Vec<ExprId>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    let d = depth + 1;
    if let Expr::Function { name, distinct, star, filter, .. } = arena.get(id) {
        if is_grouping_fn(name) {
            ensure!(!*distinct && !*star && filter.is_none(), UnsupportedFeature);
            if !out.iter().any(|&e| expr_eq(arena, e, id)) {
                out.push(id);
            }
            return Ok(());
        }
    }
    each_child(arena, id, &mut |c| collect_grouping_calls(arena, c, out, d))
}

/// 式の中のウィンドウ関数呼び出しを集める。入れ子は不正。
pub(super) fn collect_windows(
    arena: &ExprArena,
    id: ExprId,
    out: &mut Vec<ExprId>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    let d = depth + 1;
    if let Expr::Window { args, partition_by, order_by, .. } = arena.get(id) {
        // ウィンドウ関数の入れ子は SQL として無効。
        for a in args.iter().chain(partition_by) {
            let mut nested = Vec::new();
            collect_windows(arena, *a, &mut nested, d)?;
            ensure!(nested.is_empty(), UnsupportedFeature);
        }
        for o in order_by {
            let mut nested = Vec::new();
            collect_windows(arena, o.expr, &mut nested, d)?;
            ensure!(nested.is_empty(), UnsupportedFeature);
        }
        if !out.iter().any(|&e| expr_eq(arena, e, id)) {
            out.push(id);
        }
        return Ok(());
    }
    each_child(arena, id, &mut |c| collect_windows(arena, c, out, d))
}

/// 式の中の `Expr::Unnest` 呼び出しを集める。集約・ウィンドウと違って
/// 入れ子（`UNNEST(...)` の引数の中にさらに `UNNEST`）を特別扱いする必要は
/// 無い ―― `each_child` が引数の中まで辿るので自然に見つかり、複数見つかれば
/// 呼び出し側が `UnsupportedFeature` で拒否する（`FILTER`/`QUALIFY`
/// 実装時と同じ「集約でも通常の式でもない特殊な式」としての拾い方）。
pub(super) fn collect_unnests(
    arena: &ExprArena,
    id: ExprId,
    out: &mut Vec<ExprId>,
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    let d = depth + 1;
    if matches!(arena.get(id), Expr::Unnest(_)) {
        out.push(id);
        return Ok(());
    }
    each_child(arena, id, &mut |c| collect_unnests(arena, c, out, d))
}

/// UNNEST の要素型を、可能なら `Ty::Json` から実際のスカラ型へ復元する。
///
/// このエンジンでは配列・オブジェクトはすべて `Ty::Json`（JSON テキスト）
/// に統一されており、要素の型は実データを見ないと分からない
/// （`vector::types::Ty::Json` のドキュメント参照）。一方でクエリの出力列の
/// 型はバインド時（実行前）に確定していなければならないので、実データを
/// 読まずに「安全に」ネイティブ型だと判定できる場合だけ絞り込む。
///
/// 唯一データを見ずに判定できるのは、`UNNEST` の対象がその場で
/// `json_array(...)`/`list_value(...)`（DuckDB のリストリテラル相当。この
/// エンジンには `[1,2,3]` のような配列リテラル構文自体が無いので、
/// `duckdb -c "SELECT UNNEST([1,2,3])"` に対応する書き方はこちらになる）を
/// 直接呼んでいるケース: 各引数の**コンパイル時の型**が分かっているので、
/// 全引数が入れ子を持たない同じ非 JSON スカラ型に揃うかどうかを、行データを
/// 一切読まずに判定できる。
///
/// それ以外（テーブルの JSON 列そのものを UNNEST する、通常のケース）は
/// 実データを読まないと判定できないので `Ty::Json` のまま返す
/// （タスクの要求どおり、この場合は絞り込まなくてよい）。
pub(super) fn narrow_unnest_elem_ty(
    arena: &ExprArena,
    scope: &Scope,
    params: &[Value],
    arg: ExprId,
) -> Ty {
    let (name, args) = match arena.get(arg) {
        Expr::Function { name, args, distinct: false, star: false, filter: None } => (name, args),
        _ => return Ty::Json,
    };
    let is_array_ctor =
        eq_ascii_ci(name.as_bytes(), b"json_array") || eq_ascii_ci(name.as_bytes(), b"list_value");
    if !is_array_ctor || args.is_empty() {
        return Ty::Json;
    }
    let mut common: Option<Ty> = None;
    for &a in args {
        let ty = match compile(arena, scope, params, a) {
            Ok(p) => p.result_ty,
            Err(_) => return Ty::Json,
        };
        // 引数自体が JSON（配列・オブジェクトを含みうる）なら、要素が
        // 「入れ子を持たないスカラー」という前提が崩れるので諦める。
        if ty == Ty::Json {
            return Ty::Json;
        }
        common = Some(match common {
            None => ty,
            Some(c) => match Ty::unify(c, ty) {
                Some(u) => u,
                None => return Ty::Json,
            },
        });
    }
    match common {
        Some(t) if t.is_integer() => Ty::BigInt,
        Some(Ty::Float) | Some(Ty::Double) => Ty::Double,
        Some(Ty::Varchar) => Ty::Varchar,
        Some(Ty::Boolean) => Ty::Boolean,
        // DECIMAL/DATE/TIME/TIMESTAMP/NULL 等は JSON テキスト経由での復元を
        // 実装していない（範囲外。精度・書式を保って往復させるにはこの
        // エンジンの JSON シリアライズ規約を専用に決める必要があり、
        // スコープを絞った）。
        _ => Ty::Json,
    }
}

pub(super) fn build_window(
    arena: &ExprArena,
    scope: &Scope,
    params: &[Value],
    subs: &[Substitution],
    windows: &[(String, WindowDef)],
    id: ExprId,
) -> Result<WindowSpec> {
    let (name, args, star, window_ref, inline_partition_by, inline_order_by, inline_frame) =
        match arena.get(id) {
            Expr::Window { name, args, star, window_ref, partition_by, order_by, frame } => {
                (name, args, *star, window_ref, partition_by, order_by, *frame)
            }
            _ => err!(Internal),
        };
    // `OVER w`（名前付き参照）は `WINDOW` 句で定義された実体をここで引く。
    // `WINDOW` 句は構文上 SELECT リストより後に現れるため、パース時点では
    // まだ名前を解決できず、束縛時にこの一箇所でだけ解決する。
    let (partition_by, order_by, frame): (&[ExprId], &[OrderByItem], WindowFrame) = match window_ref
    {
        Some(wname) => {
            let def = windows.iter().find(|(n, _)| eq_ascii_ci(n.as_bytes(), wname.as_bytes()));
            match def {
                Some((_, d)) => (&d.partition_by, &d.order_by, d.frame),
                // 未定義のウィンドウ名を参照した。`duckdb` はパース時に
                // `window "w" does not exist` として拒否するが、このエンジンは
                // `WINDOW` 句を先読みしないため束縛時に検出する。
                None => err!(UnsupportedFeature),
            }
        }
        None => (inline_partition_by, inline_order_by, inline_frame),
    };
    let kind = match WindowKind::from_name(name) {
        Some(k) => k,
        None => err!(FunctionNotFound),
    };

    let mut arg_progs = Vec::with_capacity(args.len());
    for a in args {
        arg_progs.push(compile_with_subs(arena, scope, params, subs, *a)?);
    }
    // `count(*) OVER ()` は引数を取らない集約として扱う。
    let kind = if star {
        ensure!(kind == WindowKind::Agg(AggKind::Count), WrongArgCount);
        WindowKind::Agg(AggKind::CountStar)
    } else {
        kind
    };
    if kind.is_nullary() || kind == WindowKind::Agg(AggKind::CountStar) {
        ensure!(arg_progs.is_empty(), WrongArgCount);
    } else {
        ensure!(!arg_progs.is_empty(), WrongArgCount);
    }

    let mut parts = Vec::with_capacity(partition_by.len());
    for p in partition_by {
        parts.push(compile_with_subs(arena, scope, params, subs, *p)?);
    }
    let mut keys = Vec::with_capacity(order_by.len());
    for o in order_by {
        keys.push(SortKey {
            expr: compile_with_subs(arena, scope, params, subs, o.expr)?,
            desc: o.desc,
            nulls_first: o.nulls_first,
        });
    }

    let arg_ty = arg_progs.first().map_or(Ty::Null, |p| p.result_ty);
    let result_ty = match kind {
        // 順位付けは 1 始まりの通し番号。
        WindowKind::RowNumber | WindowKind::Rank | WindowKind::DenseRank => Ty::BigInt,
        // 値を運ぶだけの関数は入力型をそのまま返す。
        WindowKind::Lag | WindowKind::Lead | WindowKind::FirstValue | WindowKind::LastValue => {
            arg_ty
        }
        // 集約は通常の集約と同じ規則。両方でずれないよう同じ関数を通す。
        WindowKind::Agg(a) => a.result_ty(arg_ty)?,
    };

    let mut label = String::from(name);
    label.push_str("_over");
    Ok(WindowSpec {
        kind,
        args: arg_progs,
        partition_by: parts,
        order_by: keys,
        frame,
        result_ty,
        name: label,
    })
}

/// このクエリが「裸の集約呼び出し 1 つだけを SELECT する」単純な形なら、
/// その集約の種類を返す。相関スカラサブクエリが（内側の相関 GROUP BY
/// 決コリレーションを経由する）集約サブクエリだったかどうかを、束縛結果の
/// `Plan` を介さずに呼び出し側（`bind_select_in` のスカラサブクエリ処理）
/// から再判定するために使う。相関スカラサブクエリの束縛が成功している
/// 時点で、集約を伴うなら必ずこの形（`bind_select_in` の早期リターン経路の
/// `ensure!` 参照）であることが保証されているので、ここでの判定と実際の
/// 束縛結果は必ず一致する。
pub(super) fn as_bare_aggregate(arena: &ExprArena, q: &QueryStmt) -> Option<AggKind> {
    let sel = q.as_simple_select()?;
    if sel.items.len() != 1 {
        return None;
    }
    let mut probe = Vec::new();
    collect_aggregates(arena, sel.items[0].expr, &mut probe, 0).ok()?;
    if probe.len() != 1 || probe[0] != sel.items[0].expr {
        return None;
    }
    match arena.get(probe[0]) {
        Expr::Function { name, star, .. } => {
            let kind = AggKind::from_name(name)?;
            Some(if *star { AggKind::CountStar } else { kind })
        }
        _ => None,
    }
}

pub(super) fn build_agg(
    arena: &ExprArena,
    scope: &Scope,
    params: &[Value],
    id: ExprId,
) -> Result<Agg> {
    let (name, args, distinct, star, filter) = match arena.get(id) {
        Expr::Function { name, args, distinct, star, filter } => {
            (name, args, *distinct, *star, *filter)
        }
        _ => err!(Internal),
    };
    let kind = match AggKind::from_name(name) {
        Some(k) => k,
        None => err!(FunctionNotFound),
    };
    // FILTER の条件は集約前の入力スコープ（args と同じ）で評価する。
    // 集約自身は書けない（`collect_aggregates` が別扱いにしないので、
    // FILTER の中身に集約を書くと通常の「集約の入れ子」検出に掛かって弾かれる）。
    let filter_prog = match filter {
        Some(f) => Some(compile_predicate(arena, scope, params, f)?),
        None => None,
    };
    if star {
        ensure!(kind == AggKind::Count, WrongArgCount);
        ensure!(!distinct, UnsupportedFeature);
        return Ok(Agg {
            kind: AggKind::CountStar,
            arg: None,
            distinct: false,
            name: String::from("count_star()"),
            separator: Vec::new(),
            filter: filter_prog,
        });
    }

    // `string_agg(x, sep)` だけが 2 引数を許す。sep は定数リテラルのみ。
    let max_args = if kind.optional_arg_default().is_some() { 2 } else { 1 };
    ensure!(!args.is_empty() && args.len() <= max_args, WrongArgCount);
    let arg = compile(arena, scope, params, args[0])?;
    let separator = match args.get(1) {
        Some(&sep_id) => match arena.get(sep_id) {
            Expr::Literal(Value::Bytes(b)) => b.clone(),
            _ => err!(UnsupportedFeature),
        },
        None => kind.optional_arg_default().map(|d| d.to_vec()).unwrap_or_default(),
    };
    Ok(Agg {
        kind,
        arg: Some(arg),
        distinct,
        name: agg_name(name, arena, args[0]),
        separator,
        filter: filter_prog,
    })
}

fn agg_name(fname: &str, arena: &ExprArena, arg: ExprId) -> String {
    let mut s = String::from(fname);
    s.push('(');
    s.push_str(&default_name(arena, arg));
    s.push(')');
    s
}

/// GROUP BY に無い裸の列参照を検出する。
pub(super) fn check_grouped(
    arena: &ExprArena,
    scope: &Scope,
    id: ExprId,
    groups: &[ExprId],
    aggs: &[ExprId],
    depth: u32,
) -> Result<()> {
    ensure!(depth < MAX_EXPR_DEPTH, ExpressionTooDeep);
    if groups.iter().any(|&g| expr_eq(arena, g, id)) {
        return Ok(());
    }
    if aggs.iter().any(|&a| expr_eq(arena, a, id)) {
        return Ok(());
    }
    match arena.get(id) {
        Expr::ColumnRef { qualifier, name } => {
            // 入力に存在する列がここまで来た = GROUP BY にも集約にも入っていない。
            if scope.resolve(qualifier.as_deref(), name).is_ok() {
                err!(NotGrouped);
            }
        }
        // スカラサブクエリは集約前の列として横に付けてあるので、集約の上で
        // 裸で参照されると列番号がずれる。列参照と同じ扱いで弾く。
        Expr::ScalarSubquery(_) => err!(NotGrouped),
        _ => {}
    }
    let d = depth + 1;
    each_child(arena, id, &mut |c| check_grouped(arena, scope, c, groups, aggs, d))
}
