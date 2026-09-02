//! CSV writing (`export` feature).
//!
//! RFC 4180 compliant. Only quotes fields that contain a comma, quote, or
//! newline (smaller and more readable output than quoting every field).

use crate::expr::funcs::{civil_from_days, fmt_time};
use crate::prelude::*;
use crate::vector::{Batch, Field, Ty, Value};
use crate::write::{validate_batch, TableSink};

pub struct CsvSink {
    out: Vec<u8>,
    schema: Vec<Field>,
    /// Whether the header row has been written.
    began: bool,
    delimiter: u8,
}

impl CsvSink {
    pub fn new() -> Self {
        Self::with_delimiter(b',')
    }

    pub fn with_delimiter(delimiter: u8) -> Self {
        CsvSink { out: Vec::new(), schema: Vec::new(), began: false, delimiter }
    }
}

impl Default for CsvSink {
    fn default() -> Self {
        CsvSink::new()
    }
}

impl TableSink for CsvSink {
    fn begin(&mut self, schema: &[Field]) -> Result<()> {
        ensure!(!self.began, Internal);
        self.out.clear();
        self.schema = schema.to_vec();
        for (i, f) in schema.iter().enumerate() {
            if i > 0 {
                self.out.push(self.delimiter);
            }
            push_field(&mut self.out, f.name.as_bytes(), self.delimiter);
        }
        self.out.push(b'\n');
        self.began = true;
        Ok(())
    }

    fn write_batch(&mut self, schema: &[Field], batch: &Batch) -> Result<()> {
        ensure!(self.began, Internal);
        ensure!(self.schema.len() == schema.len(), Internal);
        ensure!(
            self.schema
                .iter()
                .zip(schema)
                .all(|(a, b)| { a.name == b.name && a.ty == b.ty && a.nullable == b.nullable }),
            Internal
        );
        validate_batch(schema, batch)?;
        let n = batch.num_rows();
        for r in 0..n {
            for (i, (c, f)) in batch.cols.iter().zip(schema).enumerate() {
                if i > 0 {
                    self.out.push(self.delimiter);
                }
                if !c.is_valid(r) {
                    // An empty field means NULL. CSV has no standard representation for NULL.
                    continue;
                }
                push_value(&mut self.out, &c.value_at(r), f.ty, self.delimiter);
            }
            self.out.push(b'\n');
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<Vec<u8>> {
        ensure!(self.began, Internal);
        self.began = false;
        Ok(core::mem::take(&mut self.out))
    }
}

/// Quotes a field only if it contains the delimiter, a quote, newline, or CR,
/// or if the value is the empty string.
///
/// Quoting the empty string too is deliberate: `write_batch` represents NULL
/// by writing no field at all (the counterpart of `crate::format::csv`'s read
/// side convention that an unquoted empty field is NULL). Writing an empty
/// string unquoted would produce the exact same output bytes as NULL (empty),
/// so reading it back with this crate's own CSV reader would turn the empty
/// string into NULL (a real round-trip bug that was actually found). Writing
/// `""` lets the read side's "quoted empty = empty string" convention tell them apart.
fn push_field(out: &mut Vec<u8>, s: &[u8], delimiter: u8) {
    let needs_quote =
        s.is_empty() || s.iter().any(|&b| b == delimiter || matches!(b, b'"' | b'\n' | b'\r'));
    if !needs_quote {
        out.extend_from_slice(s);
        return;
    }
    out.push(b'"');
    for &b in s {
        if b == b'"' {
            out.push(b'"');
        }
        out.push(b);
    }
    out.push(b'"');
}

fn push_value(out: &mut Vec<u8>, v: &Value, ty: Ty, delimiter: u8) {
    match v {
        Value::Null => {}
        Value::Bool(b) => out.extend_from_slice(if *b { b"true" } else { b"false" }),
        Value::I32(x) if ty == Ty::Date => push_date(out, *x as i64),
        Value::I32(x) => push_int(out, *x as i128),
        Value::I64(x) if ty == Ty::Time => fmt_time(*x, out),
        Value::I64(x) if ty == Ty::Timestamp => push_timestamp(out, *x),
        Value::I64(x) if ty == Ty::Timestamptz => push_timestamptz(out, *x),
        // `Ty::Decimal` with precision <= 18 is stored as `Value::I64`, not
        // `Value::I128` (`vector/types.rs`'s doc on `Decimal`: "precision <=
        // 18 is held as I64"). This arm used to be missing, so a
        // DECIMAL(10,2) value of 12.50 (stored as the I64 1250) wrote out as
        // the bare integer `1250` instead of `12.50` — a real round-trip bug
        // found during QA, symmetric with the `Value::I128` arm below.
        Value::I64(x) if matches!(ty, Ty::Decimal { .. }) => {
            let Ty::Decimal { scale, .. } = ty else { unreachable!() };
            push_decimal(out, *x as i128, scale);
        }
        Value::I64(x) => push_int(out, *x as i128),
        Value::I128(x) => match ty {
            Ty::Decimal { scale, .. } => push_decimal(out, *x, scale),
            Ty::Interval => {
                let (months, days, micros) = crate::vector::unpack_interval(*x);
                crate::vector::fmt_interval(months, days, micros, out);
            }
            _ => push_int(out, *x),
        },
        Value::F64(x) => push_f64(out, *x),
        Value::Bytes(b) if ty == Ty::Blob => push_blob(out, b, delimiter),
        Value::Bytes(b) if ty == Ty::Uuid => {
            // UUID's physical representation is 16 raw bytes, so convert to text
            // first before deciding whether it needs quoting (it contains hyphens
            // but no comma/quote/newline, so it can always be written unquoted).
            let mut hex = Vec::with_capacity(36);
            if let Ok(raw) = <[u8; 16]>::try_from(b.as_slice()) {
                crate::expr::funcs::fmt_uuid(&raw, &mut hex);
            }
            push_field(out, &hex, delimiter);
        }
        Value::Bytes(b) => push_field(out, b, delimiter),
    }
}

/// Writes a BLOB in DuckDB's textual form: one uppercase `\\xHH` escape per byte.
/// The result is ASCII, so it is safe to emit in a text CSV field even when the
/// underlying value contains arbitrary bytes.
fn push_blob(out: &mut Vec<u8>, bytes: &[u8], delimiter: u8) {
    if bytes.is_empty() {
        // Keep an empty BLOB distinct from NULL, just as an empty VARCHAR is.
        push_field(out, b"", delimiter);
        return;
    }
    for &b in bytes {
        out.extend_from_slice(b"\\x");
        out.push(blob_hex_digit(b >> 4));
        out.push(blob_hex_digit(b & 0x0f));
    }
}

fn blob_hex_digit(n: u8) -> u8 {
    if n < 10 {
        b'0' + n
    } else {
        b'A' + (n - 10)
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
    let neg = v < 0;
    if neg {
        out.push(b'-');
    }
    let digits = {
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
        buf[..n].iter().rev().copied().collect::<Vec<u8>>()
    };
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

/// `core` has no float formatting (`core::fmt`'s Display/Debug machinery
/// alone costs 30-60 KB, DESIGN.md §4, so this crate avoids it everywhere,
/// not just here), so this is hand-rolled. The shortest-round-trip digit
/// generation itself (`normalize_and_correct` / `shortest_digits` /
/// `nearest_at_length` / `cmp_midpoint` / `Big`, and the fixed-vs-exponential
/// rendering) is shared with the JSONL writer -- see `expr/float.rs`'s
/// module doc for why that lives in one place instead of being duplicated
/// per format.
///
/// The only thing that differs between the two writers is how non-finite
/// values are spelled: CSV writes `NaN` / `Infinity` / `-Infinity`, while
/// JSON has no such literal (JSONL writes `null` instead) -- that split is
/// why this function itself is not shared, only what it delegates to below.
fn push_f64(out: &mut Vec<u8>, v: f64) {
    if v.is_nan() {
        out.extend_from_slice(b"NaN");
        return;
    }
    if v.is_infinite() {
        out.extend_from_slice(if v > 0.0 { b"Infinity" } else { b"-Infinity" });
        return;
    }
    super::float::write_f64_finite(out, v);
}

fn push_date(out: &mut Vec<u8>, days: i64) {
    let (y, m, d) = civil_from_days(days);
    push_padded(out, y, 4);
    out.push(b'-');
    push_padded(out, m as i64, 2);
    out.push(b'-');
    push_padded(out, d as i64, 2);
}

fn push_timestamp(out: &mut Vec<u8>, micros: i64) {
    let days = micros.div_euclid(86_400_000_000);
    let rem = micros.rem_euclid(86_400_000_000);
    push_date(out, days);
    out.push(b' ');
    push_padded(out, rem / 3_600_000_000, 2);
    out.push(b':');
    push_padded(out, rem / 60_000_000 % 60, 2);
    out.push(b':');
    push_padded(out, rem / 1_000_000 % 60, 2);
    let sub = rem % 1_000_000;
    if sub != 0 {
        out.push(b'.');
        push_padded(out, sub, 6);
    }
}

fn push_timestamptz(out: &mut Vec<u8>, micros: i64) {
    push_timestamp(out, micros);
    out.extend_from_slice(b"+00");
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::Session;
    use crate::write::export_all;

    fn run_csv(sql: &str, table: &str, bytes: Vec<u8>, kind: crate::format::FormatKind) -> String {
        let mut s = Session::new();
        s.register_bytes_as(table, bytes, kind).unwrap();
        let mut sink = CsvSink::new();
        let out = export_all(&mut s, sql, &[], &mut sink).unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn header_and_rows() {
        let out = run_csv(
            "SELECT id, name FROM t ORDER BY id",
            "t",
            b"id,name\n1,alice\n2,bob\n".to_vec(),
            crate::format::FormatKind::Csv,
        );
        assert_eq!(out, "id,name\n1,alice\n2,bob\n");
    }

    #[test]
    fn quotes_fields_that_need_it() {
        let out = run_csv(
            "SELECT name FROM t",
            "t",
            b"name\n\"a,b\"\n".to_vec(),
            crate::format::FormatKind::Csv,
        );
        assert_eq!(out, "name\n\"a,b\"\n");
    }

    #[test]
    fn embedded_quote_is_doubled() {
        let out = run_csv(
            "SELECT name FROM t",
            "t",
            b"name\n\"a\"\"b\"\n".to_vec(),
            crate::format::FormatKind::Csv,
        );
        assert_eq!(out, "name\n\"a\"\"b\"\n");
    }

    #[test]
    fn null_is_empty_field() {
        let out = run_csv(
            "SELECT a, b FROM t",
            "t",
            b"a,b\n1,\n".to_vec(),
            crate::format::FormatKind::Csv,
        );
        assert_eq!(out, "a,b\n1,\n");
    }

    #[test]
    fn integer_and_float_formatting() {
        let out = run_csv(
            "SELECT a, b FROM t",
            "t",
            b"a,b\n-5,-1.5\n0,0.0\n".to_vec(),
            crate::format::FormatKind::Csv,
        );
        assert_eq!(out, "a,b\n-5,-1.5\n0,0.0\n");
    }

    #[test]
    fn interval_is_formatted_as_text() {
        let out = run_csv(
            "SELECT INTERVAL '1 month 3 days 1 hour 2 minutes 3 seconds' AS iv FROM t LIMIT 1",
            "t",
            b"id\n1\n".to_vec(),
            crate::format::FormatKind::Csv,
        );
        assert_eq!(out, "iv\n1 month 3 days 01:02:03\n");
    }

    #[test]
    fn time_and_blob_are_formatted_as_text() {
        let out = run_csv(
            "SELECT TIME '12:34:56.123456' AS tm, unhex('00a1FEff') AS b FROM t LIMIT 1",
            "t",
            b"id\n1\n".to_vec(),
            crate::format::FormatKind::Csv,
        );
        assert_eq!(out, "tm,b\n12:34:56.123456,\\x00\\xA1\\xFE\\xFF\n");
    }

    #[test]
    fn float_formatting_handles_small_fraction_and_trailing_zero_trim() {
        let out = run_csv(
            "SELECT a FROM t",
            "t",
            b"a\n0.1\n100.0\n-0.5\n".to_vec(),
            crate::format::FormatKind::Csv,
        );
        assert_eq!(out, "a\n0.1\n100.0\n-0.5\n");
    }

    #[test]
    fn non_finite_values_render_as_nan_and_infinity() {
        // CSV's own share of `push_f64`: non-finite handling is the one
        // thing that is not shared with the JSONL writer (JSON has no
        // NaN/Infinity literal, so `write/jsonl.rs` writes `null` instead --
        // see that file's equivalent test). Everything else -- shortest
        // round-trip digit generation, exact-tie regression cases, and the
        // std-Display property test -- is covered once, for both writers,
        // in `expr/float.rs`'s own test module.
        let mut out = Vec::new();
        push_f64(&mut out, f64::NAN);
        assert_eq!(out, b"NaN");
        out.clear();
        push_f64(&mut out, f64::INFINITY);
        assert_eq!(out, b"Infinity");
        out.clear();
        push_f64(&mut out, f64::NEG_INFINITY);
        assert_eq!(out, b"-Infinity");
    }

    #[test]
    fn empty_result_still_writes_header() {
        let out = run_csv(
            "SELECT id FROM t WHERE id > 100",
            "t",
            b"id\n1\n2\n".to_vec(),
            crate::format::FormatKind::Csv,
        );
        assert_eq!(out, "id\n");
    }

    // Regression test for a real round-trip bug found during QA: DECIMAL
    // with precision <= 18 is stored as `Value::I64` (`vector/types.rs`'s
    // doc on `Ty::Decimal`), but `push_value`'s decimal-scaling logic used
    // to live only on the `Value::I128` arm. A DECIMAL(10,2) column (I64
    // storage) wrote out as a bare unscaled integer (`1250` instead of
    // `12.50`), silently dropping the decimal point.
    #[test]
    fn decimal_stored_as_i64_keeps_its_decimal_point() {
        let out = run_csv(
            "SELECT CAST(a AS DECIMAL(10,2)) AS a FROM t ORDER BY a",
            "t",
            b"a\n12.5\n-1\n0\n".to_vec(),
            crate::format::FormatKind::Csv,
        );
        assert_eq!(out, "a\n-1.00\n0.00\n12.50\n");
    }

    // Regression test for a real round-trip bug found during QA: an empty
    // string (`""` in the source CSV) and a SQL NULL both used to serialize
    // as an unquoted empty field, which is indistinguishable from NULL when
    // read back by this crate's own CSV reader (`format::csv`'s
    // `empty_versus_quoted_empty` treats unquoted-empty as NULL and
    // quoted-empty as `""`). Fixed by always quoting an empty VARCHAR value
    // so the writer/reader pair round-trips losslessly, matching how
    // `duckdb`'s CSV writer also quotes empty strings (verified with the
    // `duckdb` CLI: `COPY (SELECT '' AS a) TO ...` produces `""`, not an
    // unquoted empty field).
    #[test]
    fn empty_string_and_null_round_trip_distinctly() {
        let out = run_csv(
            "SELECT a, b FROM t ORDER BY b",
            "t",
            b"a,b\n\"\",1\n,2\n".to_vec(),
            crate::format::FormatKind::Csv,
        );
        assert_eq!(out, "a,b\n\"\",1\n,2\n");

        // Round-trip: read the exported bytes back with this crate's own
        // CSV reader and confirm the empty string survived as `""`, not NULL.
        use crate::format::TableFormat;
        let src = crate::catalog::Source::from_bytes(out.into_bytes());
        let mut fmt = crate::format::csv::CsvFormat::new(b',');
        fmt.resolve(&src).unwrap().unwrap();
        let cols = fmt.read_split(&src, 0, &[0, 1]).unwrap();
        assert_eq!(
            cols[0].value_at(0),
            Value::Bytes(Vec::new()),
            "row with b=1: a must be empty string, not NULL"
        );
        assert_eq!(cols[0].value_at(1), Value::Null, "row with b=2: a must stay NULL");
    }

    #[test]
    fn rejects_a_batch_that_does_not_match_the_started_schema() {
        let mut sink = CsvSink::new();
        let schema = [Field::new("a", Ty::Int, true)];
        sink.begin(&schema).unwrap();
        let batch = Batch::new(Vec::new());
        assert_eq!(
            crate::error::code_of(sink.write_batch(&schema, &batch)),
            Some(crate::error::Code::Internal)
        );

        let mut wrong_type = crate::vector::Vector::new(Ty::BigInt);
        wrong_type.push_value(&Value::I64(1));
        let batch = Batch::new(vec![wrong_type]);
        assert_eq!(
            crate::error::code_of(sink.write_batch(&schema, &batch)),
            Some(crate::error::Code::Internal)
        );
    }

    #[test]
    fn enforces_lifecycle_and_can_be_reused_after_finish() {
        let schema = [Field::new("a", Ty::Int, true)];
        let mut sink = CsvSink::new();
        assert_eq!(crate::error::code_of(sink.finish()), Some(crate::error::Code::Internal));
        sink.begin(&schema).unwrap();
        assert_eq!(crate::error::code_of(sink.begin(&schema)), Some(crate::error::Code::Internal));
        assert_eq!(sink.finish().unwrap(), b"a\n");
        sink.begin(&schema).unwrap();
        assert_eq!(sink.finish().unwrap(), b"a\n");
    }
}
