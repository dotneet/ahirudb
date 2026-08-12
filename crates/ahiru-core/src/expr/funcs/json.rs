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
// Any-type functions (composed from kernels)
// =========================================================================

/// `COALESCE` / `IFNULL`. Takes the first non-NULL from the front.
pub(super) fn fold_null(args: &[&Vector], ty: Ty) -> Result<Vector> {
    ensure!(!args.is_empty(), WrongArgCount);
    let mut acc = args[0].clone();
    for a in &args[1..] {
        acc = kernels::pick(None, &acc, a, ty)?;
    }
    Ok(acc)
}

/// `NULLIF(a, b)`. NULL **only when `a = b` is TRUE**.
/// A row where either is NULL and the comparison is NULL returns `a` unchanged
/// (`nullif(1, NULL)` is 1).
pub(super) fn nullif(args: &[&Vector], ty: Ty) -> Result<Vector> {
    ensure!(args.len() == 2, WrongArgCount);
    let eq = kernels::compare(OpCode::Eq, ty.phys(), args[0], args[1])?;
    // A length-1 NULL applies to every row with stride 0. There is no need to build one per row.
    let mut nul = Vector::new(ty);
    nul.push_null();
    kernels::pick(Some(&eq), &nul, args[0], ty)
}

/// `GREATEST` / `LEAST`. Like DuckDB it skips NULLs and returns NULL only when everything is NULL.
///
/// Floating-point values use a total order (`NaN` greater than every finite value), matching
/// `MIN`/`MAX`. IEEE `>`/`<` would make `greatest(1, nan)` and `greatest(nan, 1)` disagree.
/// Other types keep the three-valued `acc IS NOT NULL AND (b IS NULL OR acc > b)` construction.
pub(super) fn extremum(want_max: bool, args: &[&Vector], ty: Ty) -> Result<Vector> {
    ensure!(!args.is_empty(), WrongArgCount);
    if ty.phys() == PhysType::F64 {
        return extremum_f64(want_max, args, ty);
    }
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

/// Same total order as `exec::rowkey::ord_f64`: `-inf < … < +inf < NaN`.
fn ord_f64(a: f64, b: f64) -> core::cmp::Ordering {
    use core::cmp::Ordering::*;
    if a < b {
        Less
    } else if a > b {
        Greater
    } else if a == b {
        Equal
    } else {
        match (a.is_nan(), b.is_nan()) {
            (true, true) => Equal,
            (true, false) => Greater,
            _ => Less,
        }
    }
}

fn extremum_f64(want_max: bool, args: &[&Vector], ty: Ty) -> Result<Vector> {
    let (n, s) = strides(args)?;
    let mut out = Vec::with_capacity(n);
    let mut bad: Option<Bitmap> = None;
    for i in 0..n {
        let mut best: Option<f64> = None;
        for (k, a) in args.iter().enumerate() {
            let j = i * s[k];
            if !a.is_valid(j) {
                continue;
            }
            let x = a.f64s()[j];
            best = Some(match best {
                None => x,
                Some(b) => {
                    let prefer_x =
                        if want_max { ord_f64(x, b).is_gt() } else { ord_f64(x, b).is_lt() };
                    if prefer_x {
                        x
                    } else {
                        b
                    }
                }
            });
        }
        match best {
            Some(v) => out.push(v),
            None => {
                out.push(0.0);
                set_null(&mut bad, i, n);
            }
        }
    }
    Ok(Vector::from_data(ty, Data::F64(out), bad))
}

/// `concat`. Like DuckDB it ignores NULL arguments as the empty string, so the result is never NULL.
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
// JSON construction (sharing `to_json`'s value serialization)
// =========================================================================

/// Writes one scalar value into `out` as JSON text. The value parts of `to_json`, `json_array`,
/// and `json_object` all share this. NULL becomes JSON `null` (the convention for a container's
/// element. That `to_json(NULL)` itself becomes SQL NULL is handled by the default NULL
/// propagation before this function is called).
///
/// The supported types are narrowed by `json_encodable` in `resolve`, so an unexpected logical
/// type errs safe and writes `null` (a bug should still never panic).
pub(super) fn write_json_scalar(v: &Vector, row: usize, out: &mut Vec<u8>) {
    if !v.is_valid(row) {
        out.extend_from_slice(b"null");
        return;
    }
    match v.ty() {
        Ty::Boolean => {
            out.extend_from_slice(if v.bools().get(row) { b"true" } else { b"false" });
        }
        // It should already be valid JSON text, so it is embedded as is.
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
                // NaN/Infinity have no JSON representation, so they become null.
                out.extend_from_slice(b"null");
            }
        }
        t => {
            // The integer family and DECIMAL. `fmt_int` writes an unsigned magnitude plus a scale.
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

/// `json_array`/`list_value`. Serializes each argument as an element. The arguments' types need
/// not agree (a mixture such as `json_array(1, 'x', true)` is allowed).
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

/// `list_concat`/`list_cat`/`array_concat`/`array_cat`, and the `||` operator
/// when both of its operands are `JSON` (`plan::compile::binary` emits
/// `F_LIST_CONCAT_OP` for that case). `is_operator` selects between the two,
/// which differ in NULL handling and in what a non-array operand does.
///
/// **NULL.** DuckDB defines the function and the operator differently, and
/// this reproduces both:
///
/// - `duckdb -c "select list_concat([1], NULL::INTEGER[]),
///   list_concat(NULL::INTEGER[], NULL::INTEGER[])"` -> `[1]`, `[]`. The
///   function reads a NULL list as an empty one and never returns NULL.
/// - `duckdb -c "select [1] || NULL::INTEGER[], NULL || [1]"` -> NULL, NULL.
///   The operator propagates NULL like every other binary operator.
///
/// Other DuckDB-verified cases this reproduces: `[1,2] || [3]` -> `[1, 2, 3]`,
/// `[] || [1]` -> `[1]`, `list_concat([1,2],[3],NULL::INT[],[4])` ->
/// `[1, 2, 3, 4]`.
///
/// **A JSON value that is not an array.** DuckDB cannot reach this case:
/// there `LIST` and `JSON` are separate types, so `'{"a":1}'::JSON ||
/// '{"b":2}'::JSON` is *string* concatenation (`{"a":1}{"b":2}`, VARCHAR),
/// and every mixed combination is a binder error (`duckdb -c "select [1] ||
/// '{\"a\":1}'::JSON"` -> "Cannot concatenate types INTEGER[] and JSON - an
/// explicit cast is required"; the same for `'[2]'::JSON` on the right, and
/// with the operands swapped). This engine has no `LIST` physical type — a
/// list *is* a `Ty::Json` value (`docs/DESIGN.md` §5/§8) — so the two cases
/// are indistinguishable at plan time and the behavior has to be chosen:
///
/// - **The operator raises `TypeMismatch`.** That is the code the VARCHAR
///   `||` kernel itself already raises for a run-time type problem
///   (`kernels::concat`), and the code `plan::compile::binary` already raises
///   for the other undefined `JSON` operator (ordering comparison). Returning
///   NULL instead would trade one silent wrong answer for another — and a
///   harder one to notice than the invalid-JSON string (`{"a":1}{"b":2}`)
///   this function was written to replace. `CAST(a AS VARCHAR) || CAST(b AS
///   VARCHAR)` is the documented way to get DuckDB's text concatenation.
/// - **The function still yields SQL NULL**, matching the leniency every
///   other `list_*` function here has for non-array JSON
///   (`list_extract`/`list_slice`/`list_transform`).
///
/// The operator's error takes priority over NULL propagation, so the result
/// does not depend on operand order: `'{"a":1}'::JSON || NULL` and `NULL ||
/// '{"a":1}'::JSON` both raise, rather than one raising and the other
/// returning NULL depending on which side is scanned first. A row is NULL
/// only when every non-NULL operand is a well-formed array. Documented in
/// `docs/sql/limitations.md`.
///
/// Element spans are copied verbatim out of each input, so no element is
/// re-serialized; `crate::json::list_slice(doc, 1, i64::MAX)` gives both the
/// span covering every element and the "is this an array at all" check, and
/// validates the array's structure on the way (a malformed input is an
/// `Err`, not a silently truncated list).
pub(super) fn list_concat_build(args: &[&Vector], is_operator: bool) -> Result<Vector> {
    let (n, s) = strides(args)?;
    let mut out = BytesData::with_capacity(n, n * 8);
    let mut bad: Option<Bitmap> = None;
    let mut buf = Vec::new();
    for i in 0..n {
        buf.clear();
        buf.push(b'[');
        let mut null_row = false;
        for (k, a) in args.iter().enumerate() {
            let j = i * s[k];
            if !a.is_valid(j) {
                // The function reads a NULL list as an empty one; the operator
                // propagates. Keep scanning either way — a later non-array
                // operand still has to raise (see the doc comment).
                null_row |= is_operator;
                continue;
            }
            let doc = a.bytes().get(j);
            let Some((lo, hi)) = crate::json::list_slice(doc, 1, i64::MAX)? else {
                if is_operator {
                    err!(TypeMismatch);
                }
                null_row = true;
                break;
            };
            if lo == hi {
                continue; // Empty list: contributes nothing, not even a comma.
            }
            if buf.len() > 1 {
                buf.push(b',');
            }
            buf.extend_from_slice(&doc[lo..hi]);
        }
        buf.push(b']');
        if null_row {
            out.push_empty();
            set_null(&mut bad, i, n);
        } else {
            ensure!(out.data.len() + buf.len() <= u32::MAX as usize, LimitExceeded);
            out.push(&buf);
        }
    }
    let mut v = Vector::from_data(Ty::Json, Data::Bytes(out), bad);
    v.compact_validity();
    Ok(v)
}

/// `json_object(key1, val1, key2, val2, ...)`. The keys are already converted to VARCHAR by
/// `resolve`. A NULL key has no meaningful default and falls back to the empty-string key (unlike
/// other values it does not become `null`, since a JSON key must be a string).
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
// SIMILAR TO (regexp_full_match)
// =========================================================================

/// Wraps the pattern in `^(?:...)$`. `SIMILAR TO` requires a full match rather than a partial one
/// (confirmed with `duckdb -c "select 'abc' similar to 'a.c', 'Xabc' similar to 'a.c'"`: the
/// former is true and the latter false), so it merely rides `expr::regex`'s existing anchors
/// (`^`/`$`). No new regex engine is written.
fn wrap_full_match(pattern: &[u8]) -> Vec<u8> {
    let mut w = Vec::with_capacity(pattern.len() + 6);
    w.extend_from_slice(b"^(?:");
    w.extend_from_slice(pattern);
    w.extend_from_slice(b")$");
    w
}

/// `SIMILAR TO` (`regexp_full_match`). Almost the same shape as `regex::eval_matches` (compiling
/// once per batch when the pattern column is constant), differing only in passing the wrapped
/// pattern, so `expr::regex` is left entirely unchanged.
/// The step cap (ReDoS protection) and pattern-length cap in `regex::compile`/`is_match` apply to
/// the wrapped string just the same.
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
