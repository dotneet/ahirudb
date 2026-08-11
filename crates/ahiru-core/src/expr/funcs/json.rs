//! JSON construction (`to_json`/`json_array`/`json_object`) and the
//! "any-type" functions that are built by composing `expr::kernels`
//! primitives instead of a per-physical-type loop (`coalesce`/`nullif`/
//! `greatest`/`least`/`concat`), plus `SIMILAR TO` (`regexp_full_match`).
use super::*;

/// Shared by `json_type`/`json_array_length`: if a second (path) argument is
/// present, target that path; otherwise target the whole document.
pub(super) fn json_extract_or_whole<'b>(
    a: &A<'_, 'b>,
) -> Result<Option<(&'b [u8], crate::json::Kind)>> {
    if a.n() >= 2 {
        crate::json::extract(a.bytes(0), a.bytes(1))
    } else {
        Ok(Some(crate::json::whole(a.bytes(0))?))
    }
}

// =========================================================================
// 任意型の関数（kernels の合成）
// =========================================================================

/// `COALESCE` / `IFNULL`。先頭から最初の非 NULL を採る。
pub(super) fn fold_null(args: &[&Vector], ty: Ty) -> Result<Vector> {
    ensure!(!args.is_empty(), WrongArgCount);
    let mut acc = args[0].clone();
    for a in &args[1..] {
        acc = kernels::pick(None, &acc, a, ty)?;
    }
    Ok(acc)
}

/// `NULLIF(a, b)`。`a = b` が **TRUE のときだけ** NULL。
/// どちらかが NULL で比較結果が NULL の行は `a` をそのまま返す
/// （`nullif(1, NULL)` は 1）。
pub(super) fn nullif(args: &[&Vector], ty: Ty) -> Result<Vector> {
    ensure!(args.len() == 2, WrongArgCount);
    let eq = kernels::compare(OpCode::Eq, ty.phys(), args[0], args[1])?;
    // 長さ 1 の NULL は stride 0 で全行に効く。行数ぶん作る必要は無い。
    let mut nul = Vector::new(ty);
    nul.push_null();
    kernels::pick(Some(&eq), &nul, args[0], ty)
}

/// `GREATEST` / `LEAST`。DuckDB と同じく NULL を読み飛ばし、全部 NULL の
/// ときだけ NULL を返す。
///
/// 「acc を残す条件」= `acc IS NOT NULL AND (b IS NULL OR acc > b)` を
/// 三値論理でそのまま組み立てる。物理型ごとの比較は `kernels::compare` が
/// 既に持っているので、ここに型分岐は現れない。
pub(super) fn extremum(want_max: bool, args: &[&Vector], ty: Ty) -> Result<Vector> {
    ensure!(!args.is_empty(), WrongArgCount);
    let op = if want_max { OpCode::Gt } else { OpCode::Lt };
    let mut acc = args[0].clone();
    for b in &args[1..] {
        let c = kernels::compare(op, ty.phys(), &acc, b)?;
        let c = kernels::logic(OpCode::Or, &c, &kernels::is_null(b, true))?;
        let c = kernels::logic(OpCode::And, &kernels::is_null(&acc, false), &c)?;
        acc = kernels::pick(Some(&c), &acc, b, ty)?;
    }
    Ok(acc)
}

/// `concat`。DuckDB と同じく NULL 引数は空文字列として無視するので、
/// 結果は決して NULL にならない。
pub(super) fn concat_all(args: &[&Vector], ty: Ty) -> Result<Vector> {
    let (n, s) = strides(args)?;
    let mut out = BytesData::with_capacity(n, n * 8);
    for i in 0..n {
        for (k, a) in args.iter().enumerate() {
            let j = i * s[k];
            if a.is_valid(j) {
                out.data.extend_from_slice(a.bytes().get(j));
            }
        }
        ensure!(out.data.len() <= u32::MAX as usize, LimitExceeded);
        out.offsets.push(out.data.len() as u32);
    }
    Ok(Vector::from_data(ty, Data::Bytes(out), None))
}

// =========================================================================
// JSON 構築（`to_json` の値直列化を共有する）
// =========================================================================

/// スカラ値 1 個を JSON テキストとして `out` に書く。`to_json`・
/// `json_array`・`json_object` の値部分がこれを共有する。NULL は JSON の
/// `null` になる（コンテナの要素としての規約。`to_json(NULL)` 自体が
/// SQL NULL になるのは、この関数を呼ぶ前の既定の NULL 伝播で処理される）。
///
/// 対応する型は `resolve` の `json_encodable` が絞っているので、想定外の
/// 論理型が来ても（バグでもパニックはしたくないので）安全側に `null` を書く。
pub(super) fn write_json_scalar(v: &Vector, row: usize, out: &mut Vec<u8>) {
    if !v.is_valid(row) {
        out.extend_from_slice(b"null");
        return;
    }
    match v.ty() {
        Ty::Boolean => {
            out.extend_from_slice(if v.bools().get(row) { b"true" } else { b"false" });
        }
        // すでに妥当な JSON テキストのはずなので、そのまま埋め込む。
        Ty::Json => out.extend_from_slice(v.bytes().get(row)),
        Ty::Varchar => crate::json::write_json_string(v.bytes().get(row), out),
        Ty::Date => {
            out.push(b'"');
            fmt_date(v.i32s()[row] as i64, out);
            out.push(b'"');
        }
        Ty::Time => {
            out.push(b'"');
            fmt_time(v.i64s()[row], out);
            out.push(b'"');
        }
        Ty::Timestamp => {
            out.push(b'"');
            fmt_timestamp(v.i64s()[row], out);
            out.push(b'"');
        }
        Ty::Float | Ty::Double => {
            let x = v.f64s()[row];
            if x.is_finite() {
                kernels::fmt_f64(x, out);
            } else {
                // NaN/Infinity は JSON に表現が無いので null にする。
                out.extend_from_slice(b"null");
            }
        }
        t => {
            // 整数系・DECIMAL。`fmt_int` は符号なし絶対値 + scale で書く。
            let scale = match t {
                Ty::Decimal { scale, .. } => scale,
                _ => 0,
            };
            match v.data() {
                Data::I32(d) => {
                    kernels::fmt_int(d[row].unsigned_abs() as u128, d[row] < 0, scale, out)
                }
                Data::I64(d) => {
                    kernels::fmt_int(d[row].unsigned_abs() as u128, d[row] < 0, scale, out)
                }
                Data::I128(d) => kernels::fmt_int(d[row].unsigned_abs(), d[row] < 0, scale, out),
                _ => out.extend_from_slice(b"null"),
            }
        }
    }
}

/// `json_array`/`list_value`。各引数を要素として直列化する。引数どうしの
/// 型を揃える必要はない（`json_array(1, 'x', true)` のような混在を許す）。
pub(super) fn json_array_build(args: &[&Vector]) -> Result<Vector> {
    let (n, s) = strides(args)?;
    let mut out = BytesData::with_capacity(n, n * 8);
    let mut buf = Vec::new();
    for i in 0..n {
        buf.clear();
        buf.push(b'[');
        for (k, a) in args.iter().enumerate() {
            if k > 0 {
                buf.push(b',');
            }
            write_json_scalar(a, i * s[k], &mut buf);
        }
        buf.push(b']');
        ensure!(out.data.len() + buf.len() <= u32::MAX as usize, LimitExceeded);
        out.push(&buf);
    }
    Ok(Vector::from_data(Ty::Json, Data::Bytes(out), None))
}

/// `json_object(key1, val1, key2, val2, ...)`。キーは `resolve` が
/// VARCHAR へ変換済み。NULL キーは意味のある既定が無いので空文字列キーに
/// 落とす（他の値と同じく `null` にはしない: JSON のキーは文字列でなければ
/// ならないため）。
pub(super) fn json_object_build(args: &[&Vector]) -> Result<Vector> {
    let (n, s) = strides(args)?;
    let mut out = BytesData::with_capacity(n, n * 16);
    let mut buf = Vec::new();
    for i in 0..n {
        buf.clear();
        buf.push(b'{');
        for (pi, pair) in args.chunks(2).enumerate() {
            if pi > 0 {
                buf.push(b',');
            }
            let (key_v, val_v) = (pair[0], pair[1]);
            let kj = i * s[pi * 2];
            let vj = i * s[pi * 2 + 1];
            if key_v.is_valid(kj) {
                crate::json::write_json_string(key_v.bytes().get(kj), &mut buf);
            } else {
                buf.extend_from_slice(b"\"\"");
            }
            buf.push(b':');
            write_json_scalar(val_v, vj, &mut buf);
        }
        buf.push(b'}');
        ensure!(out.data.len() + buf.len() <= u32::MAX as usize, LimitExceeded);
        out.push(&buf);
    }
    Ok(Vector::from_data(Ty::Json, Data::Bytes(out), None))
}

// =========================================================================
// SIMILAR TO（regexp_full_match）
// =========================================================================

/// パターンを `^(?:...)$` で包む。`SIMILAR TO` は部分一致ではなく完全一致を
/// 要求する（`duckdb -c "select 'abc' similar to 'a.c', 'Xabc' similar to
/// 'a.c'"` で確認済み: 前者は true、後者は false）ので、`expr::regex` の
/// 既存アンカー (`^`/`$`) に載せ替えるだけで済む。新しい正規表現エンジンは
/// 書かない。
fn wrap_full_match(pattern: &[u8]) -> Vec<u8> {
    let mut w = Vec::with_capacity(pattern.len() + 6);
    w.extend_from_slice(b"^(?:");
    w.extend_from_slice(pattern);
    w.extend_from_slice(b")$");
    w
}

/// `SIMILAR TO`（`regexp_full_match`）。`regex::eval_matches` とほぼ同じ形
/// （パターン列が定数ならバッチ内で 1 回だけコンパイルする）だが、包んだ
/// パターンを渡す点だけが違うので、`expr::regex` 側は一切変更しない。
/// `regex::compile`/`is_match` が持つステップ数上限（ReDoS 対策）・
/// パターン長上限は、包んだ後の文字列にもそのままかかる。
pub(super) fn regexp_full_match_build(args: &[&Vector]) -> Result<Vector> {
    ensure!(args.len() == 2, WrongArgCount);
    let (n, s) = strides(args)?;
    let valid = combine(args, &s, n);
    let live = |i: usize| valid.as_ref().is_none_or(|b| b.get(i));
    let (sv, pv) = (args[0], args[1]);
    let pat_const = s[1] == 0;
    let cached = if pat_const && n > 0 && pv.is_valid(0) {
        Some(regex::compile(&wrap_full_match(pv.bytes().get(0)))?)
    } else {
        None
    };
    let mut bits = Bitmap::with_capacity(n);
    for i in 0..n {
        if !live(i) {
            bits.push(false);
            continue;
        }
        let compiled;
        let prog = match &cached {
            Some(p) => p,
            None => {
                compiled = regex::compile(&wrap_full_match(pv.bytes().get(i * s[1])))?;
                &compiled
            }
        };
        bits.push(regex::is_match(prog, sv.bytes().get(i * s[0]))?);
    }
    let mut out = Vector::from_data(Ty::Boolean, Data::Bool(bits), valid);
    out.compact_validity();
    Ok(out)
}
