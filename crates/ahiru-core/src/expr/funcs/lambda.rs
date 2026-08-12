//! `list_transform` / `list_filter` / `list_reduce`.
//!
//! A lambda body has to be evaluated per row and per array element and cannot be lowered to one
//! vectorized instruction. In the same spirit as `ddl`/`dml` "building a one-row batch and running
//! it through `Vm::eval`", it builds a small one-row batch per array element and repeatedly
//! evaluates `body` (already compiled by `plan::compile::Compiler::lambda_call`).
//!
//! A parameter's type is always `Ty::Json` (the same as `list_extract`'s result). JSON `null` is
//! the SQL NULL representation of a list element (that is how `json_array`/`list_value` embed a
//! NULL argument; see the module docs at the top), so it is bound directly as SQL NULL.
use super::json::write_json_scalar;
use super::*;

/// Turns one array element into a length-1 vector for a lambda parameter.
fn lambda_param_vector(span: &[u8], kind: crate::json::Kind) -> Vector {
    let mut v = Vector::new(Ty::Json);
    if kind == crate::json::Kind::Null {
        v.push_null();
    } else {
        v.push_value(&Value::Bytes(span.to_vec()));
    }
    v
}

/// The execution body of `list_transform`/`list_filter`/`list_reduce`. Called from the `Call`
/// instruction in `expr::vm::exec`, and only when `CallSpec::lambda` is `Some`
/// (ordinary scalar functions still go through `call`).
pub fn call_lambda(
    func: FuncId,
    result_ty: Ty,
    args: &[&Vector],
    body: &Program,
) -> Result<Vector> {
    if func == F_LIST_REDUCE {
        return call_list_reduce(args, result_ty, body);
    }
    ensure!(func == F_LIST_TRANSFORM || func == F_LIST_FILTER, Internal);
    ensure!(args.len() == 1, WrongArgCount);
    let list = args[0];
    let n = list.len();
    let mut out = BytesData::with_capacity(n, n * 8);
    let mut bad: Option<Bitmap> = None;
    let mut vm = Vm::new();
    let mut buf = Vec::new();
    for i in 0..n {
        if !list.is_valid(i) {
            out.push_empty();
            set_null(&mut bad, i, n);
            continue;
        }
        let elems = match crate::json::array_elements(list.bytes().get(i))? {
            Some(e) => e,
            // A non-array value is SQL NULL (matching the lenient treatment of non-arrays in the
            // other list_* functions such as `list_extract`).
            None => {
                out.push_empty();
                set_null(&mut bad, i, n);
                continue;
            }
        };
        buf.clear();
        buf.push(b'[');
        let mut first = true;
        for (span, kind) in elems {
            let param = lambda_param_vector(span, kind);
            let elem_batch = Batch::new(vec![param]);
            let r = vm.eval(body, &elem_batch)?;
            match func {
                F_LIST_TRANSFORM => {
                    if !first {
                        buf.push(b',');
                    }
                    first = false;
                    write_json_scalar(&r, 0, &mut buf);
                }
                F_LIST_FILTER => {
                    // `plan::compile::Compiler::lambda_call` has already checked the predicate is
                    // BOOLEAN/NULL. NULL/FALSE are excluded (per SQL's three-valued logic, the same
                    // as duckdb).
                    if r.is_valid(0) && matches!(r.data(), Data::Bool(b) if b.get(0)) {
                        if !first {
                            buf.push(b',');
                        }
                        first = false;
                        buf.extend_from_slice(span);
                    }
                }
                _ => err!(Internal),
            }
        }
        buf.push(b']');
        ensure!(out.data.len() + buf.len() <= u32::MAX as usize, LimitExceeded);
        out.push(&buf);
    }
    let mut v =
        Vector::from_data(result_ty, Data::Bytes(out), merge(list.validity().cloned(), bad));
    v.compact_validity();
    Ok(v)
}

/// `list_reduce(list, (acc, x) -> expr [, initial])`.
///
/// Without `initial`, the first element becomes the initial accumulator (the same as duckdb).
/// For an empty array with no `initial`, duckdb errors while this implementation prefers the same
/// "leniently round to NULL" policy as the other list_* functions and returns SQL NULL
/// (a known incompatibility). Once the accumulation becomes NULL anywhere, folding further cannot
/// change the result, so it breaks off early.
fn call_list_reduce(args: &[&Vector], result_ty: Ty, body: &Program) -> Result<Vector> {
    ensure!(!args.is_empty() && args.len() <= 2, WrongArgCount);
    let (n, s) = strides(args)?;
    let list = args[0];
    let mut out = BytesData::with_capacity(n, n * 8);
    let mut bad: Option<Bitmap> = None;
    let mut vm = Vm::new();
    let mut buf = Vec::new();
    for i in 0..n {
        let li = i * s[0];
        if !list.is_valid(li) {
            out.push_empty();
            set_null(&mut bad, i, n);
            continue;
        }
        let elems = match crate::json::array_elements(list.bytes().get(li))? {
            Some(e) => e,
            None => {
                out.push_empty();
                set_null(&mut bad, i, n);
                continue;
            }
        };
        let mut iter = elems.into_iter();
        let mut acc: Option<Vec<u8>> = if args.len() >= 2 {
            let init = args[1];
            let ij = i * s[1];
            if init.is_valid(ij) {
                Some(init.bytes().get(ij).to_vec())
            } else {
                None
            }
        } else {
            match iter.next() {
                Some((span, kind)) => {
                    if kind == crate::json::Kind::Null {
                        None
                    } else {
                        Some(span.to_vec())
                    }
                }
                None => {
                    out.push_empty();
                    set_null(&mut bad, i, n);
                    continue;
                }
            }
        };
        for (span, kind) in iter {
            let Some(acc_text) = acc.as_ref() else {
                // Already NULL. Folding further keeps it NULL, so it breaks off.
                break;
            };
            let mut acc_v = Vector::new(Ty::Json);
            acc_v.push_value(&Value::Bytes(acc_text.clone()));
            let x_v = lambda_param_vector(span, kind);
            let batch = Batch::new(vec![acc_v, x_v]);
            let r = vm.eval(body, &batch)?;
            if !r.is_valid(0) {
                acc = None;
                break;
            }
            buf.clear();
            write_json_scalar(&r, 0, &mut buf);
            acc = Some(buf.clone());
        }
        match acc {
            Some(text) => {
                ensure!(out.data.len() + text.len() <= u32::MAX as usize, LimitExceeded);
                out.push(&text);
            }
            None => {
                out.push_empty();
                set_null(&mut bad, i, n);
            }
        }
    }
    let mut v =
        Vector::from_data(result_ty, Data::Bytes(out), merge(list.validity().cloned(), bad));
    v.compact_validity();
    Ok(v)
}
