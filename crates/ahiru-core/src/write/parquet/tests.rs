//! Parquet writer tests.
//!
//! The main lever here is a round trip through this crate's *own* Parquet
//! reader: write a file, register the bytes as a table, query it back. That
//! exercises the footer, page headers, definition levels and PLAIN payloads
//! against an independent implementation (the reader knows nothing about
//! this module), which is a much stronger check than asserting on the bytes
//! the writer happens to emit today.
//!
//! Cross-checking against DuckDB — the other half of "is this really a
//! Parquet file" — lives in `crates/ahiru-cli/tests/copy.rs`, because it
//! needs a real file on disk.

use super::*;
use crate::format::FormatKind;
use crate::session::{Prepared, QueryStep, Session};
use crate::vector::{pack_interval, Value};

/// Drive the sink directly over one batch built from `rows`.
///
/// Going through the sink rather than through `COPY` keeps these tests
/// independent of the `csv`/`jsonl` features (there would otherwise be no
/// way to get a source table into a `Session` to copy *from*).
fn write_rows(fields: &[Field], rows: &[Vec<Value>]) -> Vec<u8> {
    let mut cols: Vec<Vector> = fields.iter().map(|f| Vector::new(f.ty)).collect();
    for row in rows {
        assert_eq!(row.len(), fields.len(), "row width must match the schema");
        for (c, v) in cols.iter_mut().zip(row) {
            c.push_value(v);
        }
    }
    let batch = Batch::new(cols);
    let mut sink = ParquetSink::new();
    sink.begin(fields).unwrap();
    if !rows.is_empty() {
        sink.write_batch(fields, &batch).unwrap();
    }
    sink.finish().unwrap()
}

/// Read written bytes back through the Parquet reader.
fn read_back(bytes: Vec<u8>, sql: &str) -> (Vec<Field>, Vec<Vec<Value>>) {
    let mut s = Session::new();
    s.register_bytes_as("p", bytes, FormatKind::Parquet).unwrap();
    let mut q = match s.prepare(sql, &[]).unwrap() {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("bytes are in memory; no io should be needed"),
    };
    let schema = q.schema.clone();
    let mut rows = Vec::new();
    loop {
        match s.step(&mut q).unwrap() {
            QueryStep::Batch(mut b) => {
                b.materialize();
                for r in 0..b.num_rows() {
                    rows.push(b.cols.iter().map(|c| c.value_at(r)).collect::<Vec<_>>());
                }
            }
            QueryStep::Done => break,
            QueryStep::NeedIo(_) | QueryStep::NeedCodec(_) => panic!("unexpected io request"),
        }
    }
    (schema, rows)
}

fn f(name: &str, ty: Ty) -> Field {
    Field::new(name, ty, true)
}

/// Write one column of one type, read it back, and check both the value
/// and the type it resolved to.
fn round_trip_one(ty: Ty, value: Value) -> (Ty, Value) {
    let fields = [f("c", ty)];
    let bytes = write_rows(&fields, &[vec![value], vec![Value::Null]]);
    let (schema, rows) = read_back(bytes, "SELECT c FROM p");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[1][0], Value::Null, "the NULL row must stay NULL");
    (schema[0].ty, rows[0][0].clone())
}

#[test]
fn integers_round_trip_with_their_width_and_sign() {
    for (ty, v) in [
        (Ty::TinyInt, Value::I32(-128)),
        (Ty::SmallInt, Value::I32(-32768)),
        (Ty::Int, Value::I32(i32::MIN)),
        (Ty::BigInt, Value::I64(i64::MIN)),
        (Ty::UTinyInt, Value::I32(255)),
        (Ty::USmallInt, Value::I32(65535)),
        (Ty::UInt, Value::I64(u32::MAX as i64)),
        (Ty::UBigInt, Value::I128(u64::MAX as i128)),
    ] {
        let (got_ty, got) = round_trip_one(ty, v.clone());
        assert_eq!(got_ty, ty, "{} lost its type", ty.name());
        assert_eq!(got, v, "{} lost its value", ty.name());
    }
}

#[test]
fn floats_and_booleans_round_trip() {
    // 1.5 is exact in f32, so narrowing FLOAT to its Parquet type is
    // lossless here and the assertion stays about the encoding rather than
    // about rounding.
    assert_eq!(round_trip_one(Ty::Float, Value::F64(1.5)), (Ty::Float, Value::F64(1.5)));
    assert_eq!(round_trip_one(Ty::Double, Value::F64(-0.125)), (Ty::Double, Value::F64(-0.125)));
    assert_eq!(round_trip_one(Ty::Boolean, Value::Bool(true)), (Ty::Boolean, Value::Bool(true)));
    assert_eq!(round_trip_one(Ty::Boolean, Value::Bool(false)), (Ty::Boolean, Value::Bool(false)));
}

#[test]
fn strings_and_blobs_round_trip() {
    let s = Value::Bytes(b"hello, \xe3\x81\x82".to_vec());
    assert_eq!(round_trip_one(Ty::Varchar, s.clone()), (Ty::Varchar, s));
    let b = Value::Bytes(vec![0u8, 1, 2, 255]);
    assert_eq!(round_trip_one(Ty::Blob, b.clone()), (Ty::Blob, b));
    // The empty string is a real value, distinct from NULL; PLAIN writes it
    // as a zero length prefix and no payload.
    assert_eq!(
        round_trip_one(Ty::Varchar, Value::Bytes(Vec::new())),
        (Ty::Varchar, Value::Bytes(Vec::new()))
    );
}

#[test]
fn temporal_types_round_trip_in_microseconds() {
    assert_eq!(round_trip_one(Ty::Date, Value::I32(19_000)), (Ty::Date, Value::I32(19_000)));
    assert_eq!(
        round_trip_one(Ty::Time, Value::I64(3_723_000_001)),
        (Ty::Time, Value::I64(3_723_000_001))
    );
    assert_eq!(
        round_trip_one(Ty::Timestamp, Value::I64(-1_000_000)),
        (Ty::Timestamp, Value::I64(-1_000_000))
    );
    // The `isAdjustedToUTC` bit is the only thing separating these two, so
    // this is really a test that the logical type is written at all.
    assert_eq!(
        round_trip_one(Ty::Timestamptz, Value::I64(1_700_000_000_000_000)),
        (Ty::Timestamptz, Value::I64(1_700_000_000_000_000))
    );
}

#[test]
fn decimals_round_trip_on_both_sides_of_the_int64_boundary() {
    // precision <= 18 is stored as INT64, above it as FLBA(16); both have
    // to come back with the same precision and scale.
    let small = Ty::Decimal { precision: 10, scale: 2 };
    assert_eq!(round_trip_one(small, Value::I64(-1250)), (small, Value::I64(-1250)));
    let big = Ty::Decimal { precision: 30, scale: 4 };
    let v = Value::I128(-123_456_789_012_345_678_901_234i128);
    assert_eq!(round_trip_one(big, v.clone()), (big, v));
}

#[test]
fn hugeint_round_trips_as_a_38_digit_decimal() {
    // HUGEINT has no Parquet physical type of its own, so it goes out as
    // DECIMAL(38, 0). The value survives; the type reads back as DECIMAL
    // (documented in the module doc).
    let v = Value::I128(-(10i128.pow(38) - 1));
    let (ty, got) = round_trip_one(Ty::HugeInt, v.clone());
    assert_eq!(ty, Ty::Decimal { precision: 38, scale: 0 });
    assert_eq!(got, v);
}

#[test]
fn rejects_hugeint_outside_decimal38_range() {
    let fields = [f("c", Ty::HugeInt)];
    let mut sink = ParquetSink::new();
    sink.begin(&fields).unwrap();
    let mut column = Vector::new(Ty::HugeInt);
    column.push_value(&Value::I128(10i128.pow(38)));
    let batch = Batch::new(vec![column]);
    assert_eq!(
        crate::error::code_of(sink.write_batch(&fields, &batch)),
        Some(crate::error::Code::ValueOutOfRange)
    );
}

#[test]
fn rejects_i64_decimal_outside_declared_precision() {
    let ty = Ty::Decimal { precision: 2, scale: 0 };
    let fields = [f("c", ty)];
    let mut sink = ParquetSink::new();
    sink.begin(&fields).unwrap();
    let mut column = Vector::new(ty);
    column.push_value(&Value::I64(100));
    let batch = Batch::new(vec![column]);
    assert_eq!(
        crate::error::code_of(sink.write_batch(&fields, &batch)),
        Some(crate::error::Code::ValueOutOfRange)
    );
}

#[test]
fn uuid_round_trips_as_sixteen_raw_bytes() {
    let raw: Vec<u8> = (0u8..16).collect();
    let (ty, got) = round_trip_one(Ty::Uuid, Value::Bytes(raw.clone()));
    assert_eq!(ty, Ty::Uuid);
    assert_eq!(got, Value::Bytes(raw));
}

#[test]
fn json_round_trips_as_text() {
    // The reader maps the JSON logical type to VARCHAR (module doc), so the
    // bytes survive but the type widens.
    let text = b"{\"a\":[1,2]}".to_vec();
    let (ty, got) = round_trip_one(Ty::Json, Value::Bytes(text.clone()));
    assert_eq!(ty, Ty::Varchar);
    assert_eq!(got, Value::Bytes(text));
}

#[test]
fn interval_round_trips_as_its_text_rendering() {
    // Deliberately lossy in type: the legacy FLBA(12) INTERVAL encoding
    // cannot represent signed components and this crate's reader rejects
    // it, so the writer emits the same text the CSV/JSONL sinks do.
    let v = Value::I128(pack_interval(14, 3, 3_723_000_000));
    let (ty, got) = round_trip_one(Ty::Interval, v);
    assert_eq!(ty, Ty::Varchar);
    assert_eq!(got, Value::Bytes(b"1 year 2 months 3 days 01:02:03".to_vec()));
}

#[test]
fn untyped_null_column_round_trips() {
    let fields = [f("c", Ty::Null)];
    let bytes = write_rows(&fields, &[vec![Value::Null], vec![Value::Null]]);
    let (schema, rows) = read_back(bytes, "SELECT c FROM p");
    assert_eq!(schema[0].ty, Ty::Null);
    assert_eq!(rows, vec![vec![Value::Null], vec![Value::Null]]);
}

#[test]
fn alternating_nulls_survive_the_bit_packed_level_path() {
    // Runs shorter than 8 take the bit-packing branch of
    // `encode_def_levels`, which is the branch whose group padding is easy
    // to get wrong.
    let fields = [f("a", Ty::Int)];
    let rows: Vec<Vec<Value>> =
        (0..37).map(|i| vec![if i % 2 == 0 { Value::I32(i) } else { Value::Null }]).collect();
    let bytes = write_rows(&fields, &rows);
    let (_, got) = read_back(bytes, "SELECT a FROM p");
    assert_eq!(got, rows);
}

#[test]
fn long_null_and_value_runs_survive_the_rle_level_path() {
    // Mirror image of the test above: runs of 8 or more take the RLE
    // branch, and a mix of the two exercises the hand-off between them.
    let fields = [f("a", Ty::Int)];
    let mut rows: Vec<Vec<Value>> = Vec::new();
    rows.extend((0..100).map(|i| vec![Value::I32(i)]));
    rows.extend((0..100).map(|_| vec![Value::Null]));
    rows.push(vec![Value::I32(7)]);
    rows.push(vec![Value::Null]);
    rows.extend((0..50).map(|i| vec![Value::I32(i)]));
    let bytes = write_rows(&fields, &rows);
    let (_, got) = read_back(bytes, "SELECT a FROM p");
    assert_eq!(got, rows);
}

#[test]
fn a_column_with_no_nulls_needs_no_validity() {
    let fields = [f("a", Ty::BigInt)];
    let rows: Vec<Vec<Value>> = (0..100).map(|i| vec![Value::I64(i)]).collect();
    let bytes = write_rows(&fields, &rows);
    let (_, got) = read_back(bytes, "SELECT a FROM p");
    assert_eq!(got, rows);
}

#[test]
fn several_columns_keep_their_names_and_order() {
    let fields = [f("id", Ty::Int), f("name", Ty::Varchar), f("flag", Ty::Boolean)];
    let rows = vec![
        vec![Value::I32(1), Value::Bytes(b"a".to_vec()), Value::Bool(true)],
        vec![Value::I32(2), Value::Null, Value::Bool(false)],
    ];
    let bytes = write_rows(&fields, &rows);
    let (schema, got) = read_back(bytes, "SELECT id, name, flag FROM p ORDER BY id");
    let names: Vec<&str> = schema.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, ["id", "name", "flag"]);
    assert_eq!(got, rows);
}

#[test]
fn an_empty_result_still_produces_a_readable_file() {
    let fields = [f("a", Ty::Int), f("b", Ty::Varchar)];
    let bytes = write_rows(&fields, &[]);
    let (schema, rows) = read_back(bytes, "SELECT a, b FROM p");
    assert_eq!(schema.len(), 2);
    assert!(rows.is_empty());
}

#[test]
fn a_result_with_no_columns_is_rejected() {
    // A Parquet row group has to have at least one column, so there is no
    // file to write for a zero-column result.
    let mut sink = ParquetSink::new();
    assert_eq!(
        crate::error::code_of(sink.begin(&[])),
        Some(crate::error::Code::UnsupportedFeature)
    );
}

#[test]
fn rejects_a_batch_that_does_not_match_the_started_schema() {
    let fields = [f("a", Ty::Int)];
    let mut sink = ParquetSink::new();
    sink.begin(&fields).unwrap();
    let batch = Batch::new(Vec::new());
    assert_eq!(
        crate::error::code_of(sink.write_batch(&fields, &batch)),
        Some(crate::error::Code::Internal)
    );

    let mut not_started = ParquetSink::new();
    assert_eq!(
        crate::error::code_of(not_started.write_batch(&[], &Batch::new(Vec::new()))),
        Some(crate::error::Code::Internal)
    );
}

#[test]
fn enforces_lifecycle_and_resets_metadata_when_reused() {
    let fields = [f("a", Ty::Int)];
    let mut sink = ParquetSink::new();
    assert_eq!(crate::error::code_of(sink.finish()), Some(crate::error::Code::Internal));
    sink.begin(&fields).unwrap();
    assert_eq!(crate::error::code_of(sink.begin(&fields)), Some(crate::error::Code::Internal));
    let first = sink.finish().unwrap();
    sink.begin(&fields).unwrap();
    let second = sink.finish().unwrap();
    assert_eq!(first, second);
}

#[test]
fn footer_length_is_bounded_before_u32_framing() {
    assert_eq!(
        crate::error::code_of(validate_footer_len(crate::parquet::file::MAX_FOOTER_LEN + 1)),
        Some(crate::error::Code::LimitExceeded)
    );
}

#[test]
fn rows_past_the_row_group_limit_spill_into_a_second_row_group() {
    let fields = [f("a", Ty::Int)];
    let mut sink = ParquetSink::new();
    sink.begin(&fields).unwrap();
    // Feed batches until the sink has flushed one row group and started
    // another, so the footer has to describe both.
    let total = ROW_GROUP_ROWS + 1000;
    let mut written = 0usize;
    while written < total {
        let n = crate::vector::BATCH_SIZE.min(total - written);
        let mut v = Vector::with_capacity(Ty::Int, n);
        for i in 0..n {
            v.push_value(&Value::I32((written + i) as i32));
        }
        sink.write_batch(&fields, &Batch::new(vec![v])).unwrap();
        written += n;
    }
    let bytes = sink.finish().unwrap();

    let md = crate::parquet::meta::decode_file_metadata(footer_bytes(&bytes)).unwrap();
    assert_eq!(md.row_groups.len(), 2, "should have spilled into a second row group");
    assert_eq!(md.num_rows, total as i64);

    let (_, rows) = read_back(bytes, "SELECT a FROM p WHERE a >= 122878 ORDER BY a LIMIT 3");
    assert_eq!(
        rows,
        vec![vec![Value::I32(122878)], vec![Value::I32(122879)], vec![Value::I32(122880)]]
    );
}

#[test]
fn the_file_is_framed_by_the_parquet_magic() {
    let bytes = write_rows(&[f("a", Ty::Int)], &[vec![Value::I32(1)]]);
    assert!(bytes.len() > 12);
    assert_eq!(&bytes[..4], crate::parquet::MAGIC);
    assert_eq!(&bytes[bytes.len() - 4..], crate::parquet::MAGIC);
    // The footer length must actually point at the start of the footer.
    let md = crate::parquet::meta::decode_file_metadata(footer_bytes(&bytes)).unwrap();
    assert_eq!(md.num_rows, 1);
    assert_eq!(md.schema.len(), 2, "root element plus one leaf");
    assert_eq!(md.created_by.as_deref(), Some("ahirudb"));
}

/// Slice out the footer using the trailing length field, the way a reader
/// locates it.
fn footer_bytes(file: &[u8]) -> &[u8] {
    let n = file.len();
    let len = u32::from_le_bytes([file[n - 8], file[n - 7], file[n - 6], file[n - 5]]) as usize;
    &file[n - 8 - len..n - 8]
}

// --- unit tests on the level encoder ----------------------------------------

/// Decode a level stream back into a validity bitmap using the reader's own
/// RLE decoder, so these tests check interoperability rather than agreement
/// with themselves.
fn decode_levels(stream: &[u8], rows: usize) -> Vec<bool> {
    let len = u32::from_le_bytes([stream[0], stream[1], stream[2], stream[3]]) as usize;
    assert_eq!(len + 4, stream.len(), "length prefix must cover exactly the run bytes");
    let mut bm = Bitmap::with_capacity(rows);
    let mut d = crate::parquet::encoding::RleDecoder::new(&stream[4..], 1);
    d.read_levels_into(rows, u32::from(MAX_DEF_LEVEL), &mut bm).unwrap();
    (0..rows).map(|i| bm.get(i)).collect()
}

fn levels_from(bits: &[bool]) -> Vec<u8> {
    let mut bm = Bitmap::with_capacity(bits.len());
    for &b in bits {
        bm.push(b);
    }
    encode_def_levels(&bm, bits.len())
}

#[test]
fn def_levels_collapse_a_run_with_no_nulls() {
    let bits = vec![true; 1000];
    let stream = levels_from(&bits);
    // One RLE run: 4-byte prefix + a 2-byte varint header + 1 value byte.
    assert_eq!(stream.len(), 7, "a null-free column should not be bit-packed");
    assert_eq!(decode_levels(&stream, bits.len()), bits);
}

#[test]
fn def_levels_bit_pack_short_runs() {
    let bits: Vec<bool> = (0..24).map(|i| i % 2 == 0).collect();
    let stream = levels_from(&bits);
    // 3 groups of 8 = 3 bytes, plus a 1-byte header and the 4-byte prefix.
    assert_eq!(stream.len(), 8, "alternating levels should bit-pack, not RLE");
    assert_eq!(decode_levels(&stream, bits.len()), bits);
}

#[test]
fn def_levels_switch_between_run_kinds() {
    // Short runs, then a long one, then short runs again — the hand-off in
    // both directions, with a tail that is not a whole group of 8.
    let mut bits: Vec<bool> = (0..11).map(|i| i % 2 == 0).collect();
    bits.extend(core::iter::repeat_n(true, 50));
    bits.extend((0..13).map(|i| i % 3 == 0));
    let stream = levels_from(&bits);
    assert_eq!(decode_levels(&stream, bits.len()), bits);
}

#[test]
fn def_levels_handle_a_single_row() {
    for bit in [true, false] {
        let stream = levels_from(&[bit]);
        assert_eq!(decode_levels(&stream, 1), vec![bit]);
    }
}
