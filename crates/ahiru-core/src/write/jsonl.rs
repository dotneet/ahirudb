//! JSONL (NDJSON) 書き出し（`export` フィーチャ）。
//!
//! 1 行 1 JSON オブジェクト。`format/jsonl.rs`（読み取り側）と対称の設計だが、
//! 依存関係は無い方向にしてある（書き出しが読み取りを呼ぶことはあっても
//! 逆はない）。数値・文字列のエスケープはこのファイル内で完結させる。

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

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
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
}
