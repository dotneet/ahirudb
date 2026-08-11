//! `list_transform` / `list_filter` / `list_reduce`.
//!
//! ラムダ本体は行ごと・配列要素ごとに評価する必要があり、ベクタ化された
//! 1 命令には落とせない。`ddl`/`dml` が「1 行だけのバッチを作って `Vm::eval`
//! に通す」のと同じ発想で、配列の要素数ぶんだけ小さな 1 行バッチを作って
//! `body`（`plan::compile::Compiler::lambda_call` がコンパイル済み）を
//! 繰り返し評価する。
//!
//! パラメータの型は常に `Ty::Json`（`list_extract` の結果と同じ）。JSON の
//! `null` はリスト要素の SQL NULL 表現（`json_array`/`list_value` が NULL
//! 引数をそう埋め込む。モジュール冒頭 doc 参照）なので、そのまま SQL NULL
//! として束縛する。
use super::json::write_json_scalar;
use super::*;

/// 配列要素 1 個をラムダのパラメータ用の長さ 1 ベクタにする。
fn lambda_param_vector(span: &[u8], kind: crate::json::Kind) -> Vector {
    let mut v = Vector::new(Ty::Json);
    if kind == crate::json::Kind::Null {
        v.push_null();
    } else {
        v.push_value(&Value::Bytes(span.to_vec()));
    }
    v
}

/// `list_transform`/`list_filter`/`list_reduce` の実行本体。`expr::vm::exec`
/// の `Call` 命令から、`CallSpec::lambda` が `Some` のときだけ呼ばれる
/// （通常のスカラ関数は `call` のまま）。
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
            // 配列でない値は SQL NULL（`list_extract` 等、他の list_* 関数の
            // 非配列に対する寛容な扱いに合わせる）。
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
                    // 述語は `plan::compile::Compiler::lambda_call` が
                    // BOOLEAN/NULL であることを検査済み。NULL/FALSE は除外
                    // （SQL の 3 値論理どおり、duckdb と同じ）。
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

/// `list_reduce(list, (acc, x) -> expr [, initial])`。
///
/// `initial` が無ければ先頭要素を初期アキュムレータにする（duckdb と同じ）。
/// 空配列かつ `initial` も無い場合、duckdb はエラーにするがこの実装は他の
/// list_* 関数と同じ「寛容に NULL へ丸める」方針を優先し SQL NULL を返す
/// （既知の非互換）。累積のどこかで NULL になったら、以降は畳んでも結果が
/// 変わらないので早期に打ち切る。
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
                // 既に NULL。これ以降どう畳んでも NULL のままなので打ち切る。
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
