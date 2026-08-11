//! JSONL (NDJSON) 書き出し（`export` フィーチャ）。
//!
//! 1 行 1 JSON オブジェクト。`format/jsonl.rs`（読み取り側）と対称の設計だが、
//! 依存関係は無い方向にしてある（書き出しが読み取りを呼ぶことはあっても
//! 逆はない）。数値・文字列のエスケープはこのファイル内で完結させる。

use crate::expr::funcs::civil_from_days;
use crate::prelude::*;
use crate::vector::{Batch, Field, Ty, Value};
use crate::write::TableSink;

pub struct JsonlSink {
    out: Vec<u8>,
    schema: Vec<Field>,
}

impl JsonlSink {
    pub fn new() -> Self {
        JsonlSink { out: Vec::new(), schema: Vec::new() }
    }
}

impl Default for JsonlSink {
    fn default() -> Self {
        JsonlSink::new()
    }
}

impl TableSink for JsonlSink {
    fn begin(&mut self, schema: &[Field]) -> Result<()> {
        self.schema = schema.to_vec();
        Ok(())
    }

    fn write_batch(&mut self, schema: &[Field], batch: &Batch) -> Result<()> {
        let n = batch.num_rows();
        for r in 0..n {
            self.out.push(b'{');
            for (i, (c, f)) in batch.cols.iter().zip(schema).enumerate() {
                if i > 0 {
                    self.out.push(b',');
                }
                push_string(&mut self.out, f.name.as_bytes());
                self.out.push(b':');
                if c.is_valid(r) {
                    push_value(&mut self.out, &c.value_at(r), f.ty);
                } else {
                    self.out.extend_from_slice(b"null");
                }
            }
            self.out.push(b'}');
            self.out.push(b'\n');
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<Vec<u8>> {
        Ok(core::mem::take(&mut self.out))
    }
}

fn push_value(out: &mut Vec<u8>, v: &Value, ty: Ty) {
    match v {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(b) => out.extend_from_slice(if *b { b"true" } else { b"false" }),
        Value::I32(x) if ty == Ty::Date => push_date_string(out, *x as i64),
        Value::I32(x) => push_int(out, *x as i128),
        Value::I64(x) if ty == Ty::Timestamp => push_timestamp_string(out, *x),
        Value::I64(x) if ty == Ty::Timestamptz => push_timestamptz_string(out, *x),
        // `Ty::Decimal` with precision <= 18 is stored as `Value::I64`, not
        // `Value::I128` (`vector/types.rs`'s doc on `Decimal`). This arm was
        // missing, so a DECIMAL(10,2) value of 12.50 (stored as the I64
        // 1250) wrote out as the bare number `1250` instead of the string
        // `"12.50"` — a real round-trip bug found during QA, symmetric with
        // the `Value::I128` arm below (same reasoning: DECIMAL is written as
        // a string to avoid JSON-number rounding).
        Value::I64(x) if matches!(ty, Ty::Decimal { .. }) => {
            let Ty::Decimal { scale, .. } = ty else { unreachable!() };
            out.push(b'"');
            push_decimal(out, *x as i128, scale);
            out.push(b'"');
        }
        Value::I64(x) => push_int(out, *x as i128),
        Value::I128(x) => match ty {
            // DECIMAL は JSON の数値だと丸め誤差が乗るので文字列で正確に出す。
            // JSON は任意精度の数値を安全に表現する標準の型を持たない。
            Ty::Decimal { scale, .. } => {
                out.push(b'"');
                push_decimal(out, *x, scale);
                out.push(b'"');
            }
            // INTERVAL は JSON にネイティブな型が無いので文字列で出す。
            Ty::Interval => {
                let (months, days, micros) = crate::vector::unpack_interval(*x);
                out.push(b'"');
                crate::vector::fmt_interval(months, days, micros, out);
                out.push(b'"');
            }
            _ => push_int(out, *x),
        },
        Value::F64(x) => push_f64(out, *x),
        // `Ty::Json` values are already-valid UTF-8 JSON text (`vector::Ty::Json`
        // doc; Parquet LIST/MAP/nested-STRUCT columns are exposed this way,
        // DESIGN.md §5). Embed them verbatim so nested arrays/objects come out
        // as real JSON structure, matching `duckdb`'s `COPY ... (FORMAT JSON)`
        // (`{"tags":[1,2,3]}`, not `{"tags":"[1,2,3]"}`). Every other Bytes
        // value (VARCHAR/BLOB) is an opaque string and must be escaped.
        Value::Bytes(b) if ty == Ty::Json => out.extend_from_slice(b),
        // UUID's physical representation is the raw 16 bytes; render as the
        // usual hyphenated hex text, not the opaque escaped-string form used
        // for VARCHAR/BLOB.
        Value::Bytes(b) if ty == Ty::Uuid => {
            let mut hex = Vec::with_capacity(36);
            if let Ok(raw) = <[u8; 16]>::try_from(b.as_slice()) {
                crate::expr::funcs::fmt_uuid(&raw, &mut hex);
            }
            push_string(out, &hex);
        }
        Value::Bytes(b) => push_string(out, b),
    }
}

fn push_int(out: &mut Vec<u8>, v: i128) {
    if v < 0 {
        out.push(b'-');
    }
    let mut buf = [0u8; 40];
    let mut n = 0usize;
    let mut u = v.unsigned_abs();
    loop {
        buf[n] = b'0' + (u % 10) as u8;
        n += 1;
        u /= 10;
        if u == 0 {
            break;
        }
    }
    for i in (0..n).rev() {
        out.push(buf[i]);
    }
}

fn push_decimal(out: &mut Vec<u8>, v: i128, scale: u8) {
    if scale == 0 {
        push_int(out, v);
        return;
    }
    let scale = scale as usize;
    if v < 0 {
        out.push(b'-');
    }
    let mut buf = [0u8; 40];
    let mut n = 0usize;
    let mut u = v.unsigned_abs();
    loop {
        buf[n] = b'0' + (u % 10) as u8;
        n += 1;
        u /= 10;
        if u == 0 {
            break;
        }
    }
    let digits: Vec<u8> = buf[..n].iter().rev().copied().collect();
    if digits.len() <= scale {
        out.push(b'0');
        out.push(b'.');
        for _ in 0..(scale - digits.len()) {
            out.push(b'0');
        }
        out.extend_from_slice(&digits);
    } else {
        let split = digits.len() - scale;
        out.extend_from_slice(&digits[..split]);
        out.push(b'.');
        out.extend_from_slice(&digits[split..]);
    }
}

/// JSON は NaN/Infinity を許さない（RFC 8259）ので `null` に落とす。
/// DuckDB の `TO JSON` も同じ立場を取る。
fn push_f64(out: &mut Vec<u8>, v: f64) {
    if !v.is_finite() {
        out.extend_from_slice(b"null");
        return;
    }
    if v == 0.0 {
        out.extend_from_slice(if v.is_sign_negative() { b"-0.0" } else { b"0.0" });
        return;
    }
    let neg = v < 0.0;
    let mut x = if neg { -v } else { v };
    if neg {
        out.push(b'-');
    }
    // `trunc()` は core に無い（libm 依存）ので `as i128` の
    // 飽和・ゼロ方向丸めキャストで代用する。
    let ip = x as i128;
    push_int(out, ip);
    out.push(b'.');
    // `ip` が i128 の範囲外で飽和した場合（v の絶対値が i128::MAX 超）、
    // `x - ip as f64` は [0,1) に収まらない巨大な残差になる。そのまま桁
    // 抽出すると `x as u8` が 255 に飽和し、`b'0' + d` で u8 加算オーバー
    // フロー（デバッグ panic、release ビルドでは不正なバイトの書き込み）
    // になる。飽和したら小数部の抽出自体を諦める。
    if ip == i128::MAX || ip == i128::MIN {
        out.push(b'0');
    } else {
        x -= ip as f64;
        let mut digits = Vec::with_capacity(15);
        for _ in 0..15 {
            x *= 10.0;
            let d = x as u8;
            digits.push(b'0' + d);
            x -= d as f64;
        }
        while digits.len() > 1 && *digits.last().unwrap() == b'0' {
            digits.pop();
        }
        out.extend_from_slice(&digits);
    }
}

fn push_date_string(out: &mut Vec<u8>, days: i64) {
    out.push(b'"');
    let (y, m, d) = civil_from_days(days);
    push_padded(out, y, 4);
    out.push(b'-');
    push_padded(out, m as i64, 2);
    out.push(b'-');
    push_padded(out, d as i64, 2);
    out.push(b'"');
}

fn push_timestamp_string(out: &mut Vec<u8>, micros: i64) {
    out.push(b'"');
    let days = micros.div_euclid(86_400_000_000);
    let rem = micros.rem_euclid(86_400_000_000);
    let (y, m, d) = civil_from_days(days);
    push_padded(out, y, 4);
    out.push(b'-');
    push_padded(out, m as i64, 2);
    out.push(b'-');
    push_padded(out, d as i64, 2);
    out.push(b' ');
    push_padded(out, rem / 3_600_000_000, 2);
    out.push(b':');
    push_padded(out, rem / 60_000_000 % 60, 2);
    out.push(b':');
    push_padded(out, rem / 1_000_000 % 60, 2);
    out.push(b'"');
}

fn push_timestamptz_string(out: &mut Vec<u8>, micros: i64) {
    // `push_timestamp_string` already writes the closing quote; splice the
    // `+00` suffix in before it rather than duplicating the date/time body.
    push_timestamp_string(out, micros);
    out.truncate(out.len() - 1);
    out.extend_from_slice(b"+00\"");
}

fn push_padded(out: &mut Vec<u8>, v: i64, width: usize) {
    let neg = v < 0;
    let mut buf = [0u8; 20];
    let mut n = 0usize;
    let mut u = v.unsigned_abs();
    loop {
        buf[n] = b'0' + (u % 10) as u8;
        n += 1;
        u /= 10;
        if u == 0 {
            break;
        }
    }
    if neg {
        out.push(b'-');
    }
    for _ in 0..width.saturating_sub(n) {
        out.push(b'0');
    }
    for i in (0..n).rev() {
        out.push(buf[i]);
    }
}

/// JSON 文字列としてエスケープして書く（引用符込み）。
fn push_string(out: &mut Vec<u8>, s: &[u8]) {
    out.push(b'"');
    for &b in s {
        match b {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            0x00..=0x1f => {
                out.extend_from_slice(b"\\u00");
                out.push(hex_digit(b >> 4));
                out.push(hex_digit(b & 0xf));
            }
            _ => out.push(b),
        }
    }
    out.push(b'"');
}

fn hex_digit(n: u8) -> u8 {
    if n < 10 {
        b'0' + n
    } else {
        b'a' + (n - 10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::Session;
    use crate::write::export_all;

    fn run(sql: &str, bytes: Vec<u8>, kind: crate::format::FormatKind) -> Vec<String> {
        let mut s = Session::new();
        s.register_bytes_as("t", bytes, kind).unwrap();
        let mut sink = JsonlSink::new();
        let out = export_all(&mut s, sql, &[], &mut sink).unwrap();
        String::from_utf8(out).unwrap().lines().map(String::from).collect()
    }

    #[test]
    fn basic_object_per_row() {
        let lines = run(
            "SELECT id, name FROM t ORDER BY id",
            b"id,name\n1,alice\n2,bob\n".to_vec(),
            crate::format::FormatKind::Csv,
        );
        assert_eq!(lines, vec![r#"{"id":1,"name":"alice"}"#, r#"{"id":2,"name":"bob"}"#]);
    }

    #[test]
    fn null_becomes_json_null() {
        let lines =
            run("SELECT a, b FROM t", b"a,b\n1,\n".to_vec(), crate::format::FormatKind::Csv);
        assert_eq!(lines, vec![r#"{"a":1,"b":null}"#]);
    }

    #[test]
    fn string_escaping() {
        // CSV は引用符で囲んだフィールドの中に生の改行・タブ・二重引用符
        // （`""` で表す）を持てる。それぞれ JSON の `\n` `\t` `\"` に
        // 変換されることを確認する。バイト列を直接組み立てて、Rust の
        // 文字列エスケープと CSV のクォート規則が混ざって読みにくくなる
        // のを避ける。
        let mut csv = Vec::new();
        csv.extend_from_slice(b"s\n\"line1");
        csv.push(b'\n'); // フィールド内の生の改行
        csv.extend_from_slice(b"line2");
        csv.push(b'\t'); // フィールド内の生のタブ
        csv.extend_from_slice(b"tab\"\"q\"\n"); // `""` は埋め込みの `"` 1 個
        let lines = run("SELECT s FROM t", csv, crate::format::FormatKind::Csv);
        assert_eq!(lines[0], "{\"s\":\"line1\\nline2\\ttab\\\"q\"}");
    }

    #[test]
    fn interval_is_formatted_as_json_string() {
        let lines = run(
            "SELECT INTERVAL '3 days' AS iv FROM t LIMIT 1",
            b"id\n1\n".to_vec(),
            crate::format::FormatKind::Csv,
        );
        assert_eq!(lines, vec![r#"{"iv":"3 days"}"#]);
    }

    #[test]
    fn empty_result_produces_no_lines() {
        let lines = run(
            "SELECT id FROM t WHERE id > 100",
            b"id\n1\n2\n".to_vec(),
            crate::format::FormatKind::Csv,
        );
        assert!(lines.is_empty());
    }

    // Regression test for a real round-trip bug found during QA, symmetric
    // with the one in `write/csv.rs`: DECIMAL with precision <= 18 is
    // stored as `Value::I64` (`vector/types.rs`'s doc on `Ty::Decimal`),
    // but the decimal-scaling + string-quoting logic used to live only on
    // the `Value::I128` arm. A DECIMAL(10,2) column (I64 storage) wrote out
    // as a bare unscaled JSON number (`1250` instead of the quoted string
    // `"12.50"`), silently dropping the decimal point.
    #[test]
    fn decimal_stored_as_i64_keeps_its_decimal_point() {
        let lines = run(
            "SELECT CAST(a AS DECIMAL(10,2)) AS a FROM t",
            b"a\n12.5\n".to_vec(),
            crate::format::FormatKind::Csv,
        );
        assert_eq!(lines, vec![r#"{"a":"12.50"}"#]);
    }

    #[test]
    fn float_and_negative_numbers() {
        let lines =
            run("SELECT a, b FROM t", b"a,b\n-5,-1.5\n".to_vec(), crate::format::FormatKind::Csv);
        assert_eq!(lines, vec![r#"{"a":-5,"b":-1.5}"#]);
    }

    // `push_f64` の整数部飽和ガードの回帰テスト（csv.rs と同種のバグ）。
    // 修正前は `x as i128` が i128::MAX に飽和した後の残差抽出で
    // u8 加算オーバーフローが起き panic していた。
    #[test]
    fn float_formatting_saturates_on_extremely_large_finite_values_without_panicking() {
        let lines = run("SELECT a FROM t", b"a\n1e40\n".to_vec(), crate::format::FormatKind::Csv);
        assert_eq!(lines, vec![format!(r#"{{"a":{}.0}}"#, i128::MAX)]);
    }

    // Regression test for a real bug found during QA: a `Ty::Json` column
    // (produced by Parquet LIST/MAP/nested-STRUCT columns, DESIGN.md §5) is
    // physically stored as raw UTF-8 JSON text (`vector/types.rs`'s doc on
    // `Ty::Json`). The JSONL writer used to treat every `Value::Bytes` as an
    // opaque string via `push_string` regardless of `ty`, so it re-escaped
    // already-valid JSON text into a JSON *string*: `"xs":"[1,2,3]"` instead
    // of the nested value `"xs":[1,2,3]`. Verified against `duckdb`'s own
    // `COPY ... (FORMAT JSON)`, which embeds LIST/STRUCT columns unescaped
    // (`{"id":1,"tags":[1,2,3]}`). Fixed by writing `Ty::Json` bytes through
    // verbatim instead of through `push_string`.
    #[test]
    fn json_typed_column_is_embedded_raw_not_double_encoded() {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/list1.parquet");
        let bytes = std::fs::read(p).unwrap();
        let lines =
            run("SELECT id, xs FROM t WHERE id = 0", bytes, crate::format::FormatKind::Parquet);
        assert_eq!(lines, vec![r#"{"id":0,"xs":[1,2,3]}"#]);
    }

    #[test]
    fn null_json_typed_column_is_still_json_null() {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/list_varied.parquet");
        let bytes = std::fs::read(p).unwrap();
        let lines =
            run("SELECT id, xs FROM t WHERE id = 0", bytes, crate::format::FormatKind::Parquet);
        // list_varied.parquet: row 0's list itself is SQL NULL (see nested_files.rs).
        assert_eq!(lines, vec![r#"{"id":0,"xs":null}"#]);
    }
}
