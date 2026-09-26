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
    // The arguments keep their own types (the result is the first one's), so they are compared
    // in their common type.
    let (t0, t1) = (args[0].ty(), args[1].ty());
    let t = Ty::unify_or_mismatch(t0, t1)?;
    let (l, r) = (kernels::cast(t0, t, args[0])?, kernels::cast(t1, t, args[1])?);
    let eq = kernels::compare(OpCode::Eq, t.phys(), &l, &r)?;
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

/// `concat_ws(sep, x, ...)`. A NULL value argument is dropped along with the separator that
/// would have preceded it, so `concat_ws('-', 'a', NULL, 'b')` is `a-b` rather than `a--b`.
/// A NULL **separator** makes the whole row NULL (both verified against duckdb).
pub(super) fn concat_ws_build(args: &[&Vector], ty: Ty) -> Result<Vector> {
    let (n, s) = strides(args)?;
    let mut out = BytesData::with_capacity(n, n * 8);
    let mut bad: Option<Bitmap> = None;
    for i in 0..n {
        if !args[0].is_valid(i * s[0]) {
            out.offsets.push(out.data.len() as u32);
            set_null(&mut bad, i, n);
            continue;
        }
        let sep = args[0].bytes().get(i * s[0]);
        let mut first = true;
        for (k, a) in args.iter().enumerate().skip(1) {
            let j = i * s[k];
            if !a.is_valid(j) {
                continue;
            }
            if !first {
                out.data.extend_from_slice(sep);
            }
            out.data.extend_from_slice(a.bytes().get(j));
            first = false;
        }
        ensure!(out.data.len() <= u32::MAX as usize, LimitExceeded);
        out.offsets.push(out.data.len() as u32);
    }
    Ok(Vector::from_data(ty, Data::Bytes(out), bad))
}

// =========================================================================
// LIST functions over the JSON representation
// =========================================================================

/// The 1-based index of `needle` in the array, or 0 when absent. `None` when the argument is
/// not an array at all (the caller turns that into SQL NULL, matching duckdb's behavior for
/// `list_position` on a non-list).
///
/// The needle is serialized to JSON text and compared with `crate::json::cmp_element`, so
/// numbers compare by value (`list_contains([1.0, 2.0], 2)`, as DuckDB's implicit cast to
/// the element type gives) and strings by their decoded text.
pub(super) fn list_find(a: &A) -> Result<Option<i64>> {
    let mut needle = Vec::new();
    if let Some((v, j)) = a.at(1) {
        write_json_scalar(v, j, &mut needle);
    }
    let doc = a.bytes(0);
    let elems = match crate::json::array_elements(doc)? {
        Some(e) => e,
        None => return Ok(None),
    };
    for (k, (span, _)) in elems.iter().enumerate() {
        if crate::json::cmp_element(span, &needle)?.is_eq() {
            return Ok(Some(k as i64 + 1));
        }
    }
    Ok(Some(0))
}

/// `list_sort` / `list_reverse_sort` / `list_distinct` / `list_reverse`. All of them only
/// reorder or drop whole element spans, so one body covers them: parse once, permute the
/// spans, re-emit.
///
/// The sorts order elements with `crate::json::cmp_element` (the order of the typed values:
/// numbers by value, strings by decoded bytes, lists and structs element-wise) and take
/// DuckDB's optional arguments: `list_sort(l, 'ASC'|'DESC', 'NULLS FIRST'|'NULLS LAST')` and
/// `list_reverse_sort(l, 'NULLS FIRST'|'NULLS LAST')`. NULLs go last by default either way,
/// as in DuckDB (`list_reverse_sort([3, NULL, 1])` is `[3, 1, NULL]`).
pub(super) fn list_rearrange(id: FuncId, a: &A, out: &mut Vec<u8>) -> Result<bool> {
    let elems = match crate::json::array_elements(a.bytes(0))? {
        Some(e) => e,
        // Not an array -> NULL, the same judgment as `list_find`.
        None => return Ok(false),
    };
    let mut order: Vec<usize> = (0..elems.len()).collect();
    match id {
        F_LIST_REVERSE => order.reverse(),
        F_LIST_SORT | F_LIST_REVERSE_SORT => {
            let opt = |k: usize, want: &[u8]| -> Result<bool> {
                if a.n() <= k {
                    return Ok(false);
                }
                let s = a.bytes(k);
                if crate::rt::hash::eq_ascii_ci(s, want) {
                    return Ok(true);
                }
                // The other spelling of the same option.
                let other: &[u8] = match want {
                    b"DESC" => b"ASC",
                    _ => b"NULLS LAST",
                };
                ensure!(crate::rt::hash::eq_ascii_ci(s, other), UnsupportedFeature);
                Ok(false)
            };
            let rev = id == F_LIST_REVERSE_SORT;
            let desc = rev || opt(1, b"DESC")?;
            let nulls_first = opt(if rev { 1 } else { 2 }, b"NULLS FIRST")?;
            let null = |k: usize| elems[k].1 == crate::json::Kind::Null;
            order.sort_by(|&x, &y| match (null(x), null(y)) {
                (false, false) => {
                    let o = crate::json::cmp_element(elems[x].0, elems[y].0)
                        .unwrap_or_else(|_| elems[x].0.cmp(elems[y].0));
                    if desc {
                        o.reverse()
                    } else {
                        o
                    }
                }
                (nx, ny) => {
                    if nulls_first {
                        ny.cmp(&nx)
                    } else {
                        nx.cmp(&ny)
                    }
                }
            });
        }
        // Keeps first-occurrence order (duckdb's `list_distinct` does not promise an order, and
        // this way the function needs no comparator). SQL NULL list elements are not retained,
        // matching DuckDB's `list_distinct([NULL, 1, NULL]) = [1]`.
        _ => order.retain(|&k| {
            elems[k].1 != crate::json::Kind::Null && !elems[..k].iter().any(|e| e.0 == elems[k].0)
        }),
    }
    out.push(b'[');
    for (i, &k) in order.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        out.extend_from_slice(elems[k].0);
    }
    out.push(b']');
    Ok(true)
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
        Ty::Blob => write_json_blob(v.bytes().get(row), out),
        Ty::Float | Ty::Double => {
            let x = v.f64s()[row];
            if x.is_finite() && v.ty() == Ty::Float {
                // Every FLOAT is exactly an `f32`, so its shortest text is measured against
                // `f32`, as `CAST(x AS VARCHAR)` does: `[0.1::FLOAT]` is `[0.1]` (as in DuckDB),
                // not the widened double's `[0.10000000149011612]`.
                kernels::fmt_f32(x, out);
            } else {
                write_json_f64(x, out);
            }
        }
        t => {
            // The integer family and DECIMAL.
            let x = match v.data() {
                Data::I32(d) => d[row] as i128,
                Data::I64(d) => d[row] as i128,
                Data::I128(d) => d[row],
                _ => return out.extend_from_slice(b"null"),
            };
            write_json_int(t, x, out);
        }
    }
}

/// A DOUBLE as JSON. Non-finite values are written `NaN`/`Infinity`/`-Infinity`, as DuckDB's
/// `to_json` does (and its JSON reader accepts; so does `crate::json::scan_number`), so a
/// list holding them reads back as the same values.
pub(crate) fn write_json_f64(x: f64, out: &mut Vec<u8>) {
    if x.is_nan() {
        out.extend_from_slice(b"NaN");
    } else if x.is_infinite() {
        out.extend_from_slice(if x < 0.0 { b"-Infinity" } else { b"Infinity" });
    } else {
        kernels::fmt_f64(x, out);
    }
}

/// An integer or DECIMAL (`x` is the scaled value) as JSON. DuckDB's `to_json` writes a
/// DECIMAL of precision 15 or less through DOUBLE (`-0.50::DECIMAL(4,2)` is `-0.5`,
/// `5::DECIMAL(9,0)` is `5.0`) and a wider one as its exact text (`1.50::DECIMAL(16,2)` is
/// `1.50`); every such DECIMAL value is below 2^53, so the double is exact.
pub(crate) fn write_json_int(ty: Ty, x: i128, out: &mut Vec<u8>) {
    let (p, scale) = match ty {
        Ty::Decimal { precision, scale } => (precision, scale),
        _ => (u8::MAX, 0),
    };
    if p <= 15 {
        let mut d = 1.0;
        for _ in 0..scale {
            d *= 10.0;
        }
        write_json_f64(x as f64 / d, out);
    } else {
        kernels::fmt_int(x.unsigned_abs(), x < 0, scale, out);
    }
}

/// A BLOB as JSON: a string holding its VARCHAR form (`\xHH` escapes), as DuckDB's `to_json`.
fn write_json_blob(b: &[u8], out: &mut Vec<u8>) {
    let mut s = Vec::new();
    kernels::escape_blob(b, &mut s);
    crate::json::write_json_string(&s, out);
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

/// `SIMILAR TO` (`regexp_full_match`). Almost the same shape as `regex::eval_matches` (compiling
/// once per batch when the pattern column is constant), differing only in compiling the pattern
/// with `regex::compile_full` instead of `regex::compile`.
///
/// `SIMILAR TO` requires a full match rather than a partial one (confirmed with
/// `duckdb -c "select 'abc' similar to 'a.c', 'Xabc' similar to 'a.c'"`: the former is true and
/// the latter false). This used to be done by wrapping the pattern text in `^(?:...)$`, which
/// silently repaired an unbalanced pattern into a valid one with a different meaning
/// (`'a' SIMILAR TO 'a)|(b'` answered `true` where DuckDB raises an error). `compile_full`
/// parses the pattern on its own and emits the anchors around the compiled program instead.
/// The step cap (ReDoS protection) and pattern-length cap in `regex::compile`/`is_match` apply
/// just the same.
pub(super) fn regexp_full_match_build(args: &[&Vector]) -> Result<Vector> {
    ensure!(args.len() == 2, WrongArgCount);
    let (n, s) = strides(args)?;
    let valid = combine(args, &s, n);
    let live = |i: usize| valid.as_ref().is_none_or(|b| b.get(i));
    let (sv, pv) = (args[0], args[1]);
    let pat_const = s[1] == 0;
    let cached = if pat_const && n > 0 && pv.is_valid(0) {
        Some(regex::compile_full(pv.bytes().get(0))?)
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
                compiled = regex::compile_full(pv.bytes().get(i * s[1]))?;
                &compiled
            }
        };
        bits.push(regex::is_match(prog, sv.bytes().get(i * s[0]))?);
    }
    let mut out = Vector::from_data(Ty::Boolean, Data::Bool(bits), valid);
    out.compact_validity();
    Ok(out)
}
