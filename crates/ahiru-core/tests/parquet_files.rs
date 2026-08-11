//! 実ファイルに対する統合テスト。
//!
//! テストデータは DuckDB が書いた本物の Parquet。単体テストは自前で組んだ
//! バイト列を使うので、「実際の writer が出すもの」に対する検証をここで担保する。
//! 期待値は `duckdb -c "SELECT ..."` の出力と突き合わせて決めている。

use ahiru_core::parquet::file::open_bytes;
use ahiru_core::parquet::reader::{read_column_chunk, NoPageCache};
use ahiru_core::vector::{Ty, Value, Vector};

fn data(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// ファイル全体を列ごとに読む。
fn read_all(bytes: &[u8]) -> (Vec<String>, Vec<Vector>) {
    let f = open_bytes(bytes).expect("open");
    let names: Vec<String> = f.schema.columns.iter().map(|c| c.name.clone()).collect();
    let mut cols: Vec<Option<Vector>> = (0..f.schema.columns.len()).map(|_| None).collect();

    for rg in &f.meta.row_groups {
        for (i, desc) in f.schema.columns.iter().enumerate() {
            let meta = rg.columns[i].meta.as_ref().expect("column metadata");
            let (start, end) = meta.byte_range();
            let v = read_column_chunk(
                desc,
                meta,
                &bytes[start as usize..end as usize],
                start,
                rg.num_rows as usize,
                &NoPageCache,
            )
            .unwrap_or_else(|e| panic!("column {}: {e}", desc.name));
            match &mut cols[i] {
                // RowGroup ごとの結果を縦に連結する。
                Some(acc) => {
                    let mut merged = Vector::with_capacity(desc.ty, acc.len() + v.len());
                    for k in 0..acc.len() {
                        merged.push_value(&acc.value_at(k));
                    }
                    for k in 0..v.len() {
                        merged.push_value(&v.value_at(k));
                    }
                    *acc = merged;
                }
                None => cols[i] = Some(v),
            }
        }
    }
    (names, cols.into_iter().map(|c| c.unwrap()).collect())
}

#[test]
fn basic_file_matches_duckdb_output() {
    let bytes = data("basic.parquet");
    let (names, cols) = read_all(&bytes);
    assert_eq!(names, ["id", "name", "score", "flag", "big", "d"]);
    assert!(cols.iter().all(|c| c.len() == 1000));

    // duckdb: SELECT * FROM 'basic.parquet' LIMIT 3
    assert_eq!(cols[0].value_at(0), Value::I32(0));
    assert_eq!(cols[1].value_at(0), Value::Bytes(b"name_0".to_vec()));
    assert_eq!(cols[2].value_at(0), Value::F64(0.0));
    assert_eq!(cols[3].value_at(0), Value::Bool(true));
    assert_eq!(cols[4].value_at(0), Value::Null); // i % 5 == 0
    assert_eq!(cols[0].value_at(1), Value::I32(1));
    assert_eq!(cols[2].value_at(1), Value::F64(1.5));
    assert_eq!(cols[4].value_at(1), Value::I64(100));

    // 末尾も確認する。ページ境界の扱いを間違えるとここがずれる。
    assert_eq!(cols[0].value_at(999), Value::I32(999));
    assert_eq!(cols[1].value_at(999), Value::Bytes(b"name_5".to_vec()));
    assert_eq!(cols[4].value_at(999), Value::I64(99_900));
}

#[test]
fn nulls_land_on_the_right_rows() {
    let bytes = data("basic.parquet");
    let (_, cols) = read_all(&bytes);
    let big = &cols[4];
    for i in 0..1000 {
        let expect_null = i % 5 == 0;
        assert_eq!(big.value_at(i) == Value::Null, expect_null, "行 {i} の NULL 判定がずれている");
    }
    assert_eq!((0..1000).filter(|i| big.value_at(*i) == Value::Null).count(), 200);
}

#[test]
fn dictionary_encoded_strings_round_trip() {
    let bytes = data("basic.parquet");
    let (_, cols) = read_all(&bytes);
    // name は PLAIN_DICTIONARY。7 種類の値が循環している。
    for i in 0..1000 {
        let expect = alloc_name(i % 7);
        assert_eq!(cols[1].value_at(i), Value::Bytes(expect), "行 {i}");
    }
}

fn alloc_name(n: usize) -> Vec<u8> {
    let mut v = b"name_".to_vec();
    v.push(b'0' + n as u8);
    v
}

#[test]
fn timestamps_are_normalised_to_micros() {
    let bytes = data("basic.parquet");
    let f = open_bytes(&bytes).unwrap();
    assert_eq!(f.schema.columns[5].ty, Ty::Timestamp);
    let (_, cols) = read_all(&bytes);
    // 2024-01-01T00:00:00Z = 1704067200 秒
    assert_eq!(cols[5].value_at(0), Value::I64(1_704_067_200_000_000));
    // 翌日
    assert_eq!(cols[5].value_at(1), Value::I64(1_704_153_600_000_000));
}

#[test]
fn multiple_row_groups_are_concatenated_in_order() {
    let bytes = data("multi_rg.parquet");
    let f = open_bytes(&bytes).unwrap();
    assert!(f.num_row_groups() > 1, "複数 RowGroup のファイルであること");
    assert_eq!(f.num_rows(), 50_000);

    let (names, cols) = read_all(&bytes);
    assert_eq!(names, ["id", "k", "s", "f"]);
    assert!(cols.iter().all(|c| c.len() == 50_000));

    // RowGroup をまたいでも順序と値が保たれるか。
    for i in [0usize, 1, 8191, 8192, 8193, 24_999, 49_999] {
        assert_eq!(cols[0].value_at(i), Value::I32(i as i32), "id[{i}]");
        assert_eq!(cols[1].value_at(i), Value::I64((i % 97) as i64), "k[{i}]");
        assert_eq!(cols[3].value_at(i), Value::F64(i as f64 * 0.25), "f[{i}]");
    }
}

#[test]
fn uncompressed_pages_are_read_without_a_codec() {
    let bytes = data("plain.parquet");
    let (_, cols) = read_all(&bytes);
    assert_eq!(cols[0].len(), 5000);
    assert_eq!(cols[0].value_at(4999), Value::I32(4999));
}

// duckdb -c "SELECT count(*) FROM 'zstd.parquet'" -> 5000
// duckdb -c "DESCRIBE SELECT * FROM 'zstd.parquet'" -> id INTEGER のみ、0..4999

/// `zstd` フィーチャ（既定で有効）が付いていれば、ZSTD 圧縮ページはコアが
/// その場で展開できる。ホストへの委譲（`NeedCodec`）は経由しない。
#[cfg(feature = "zstd")]
#[test]
fn zstd_pages_are_decompressed_by_the_core_when_the_feature_is_on() {
    let bytes = data("zstd.parquet");
    let (_, cols) = read_all(&bytes);
    assert_eq!(cols[0].len(), 5000);
    for i in 0..5000usize {
        assert_eq!(cols[0].value_at(i), Value::I32(i as i32), "row {i}");
    }
}

/// `zstd` を外した構成では、旧来どおりホスト委譲が必要という明確な
/// `UnsupportedCodec` になる（黙って壊れるより良い）。
#[cfg(not(feature = "zstd"))]
#[test]
fn zstd_is_rejected_with_a_clear_code_when_the_feature_is_off() {
    let bytes = data("zstd.parquet");
    let f = open_bytes(&bytes).expect("フッタは非圧縮なので読める");
    let rg = &f.meta.row_groups[0];
    let meta = rg.columns[0].meta.as_ref().unwrap();
    let (start, end) = meta.byte_range();
    let r = read_column_chunk(
        &f.schema.columns[0],
        meta,
        &bytes[start as usize..end as usize],
        start,
        rg.num_rows as usize,
        &NoPageCache,
    );
    let code = r.err().map(|e| e.code_u16());
    assert_eq!(code, Some(201), "UnsupportedCodec が返るべき");
}

#[test]
fn truncated_file_is_rejected_without_panicking() {
    let bytes = data("basic.parquet");
    // フッタが読めない長さに切り詰める。
    for cut in [0usize, 4, 8, 100, 1000] {
        let r = open_bytes(&bytes[..cut.min(bytes.len())]);
        assert!(r.is_err(), "{cut} バイトで切ったファイルは拒否されるべき");
    }
}

#[test]
fn corrupted_footer_bytes_never_panic() {
    let bytes = data("basic.parquet");
    // フッタ領域のバイトを 1 つずつ壊しても、パニックせず Err か正常終了で済むこと。
    let n = bytes.len();
    for i in (n - 400)..(n - 8) {
        let mut b = bytes.clone();
        b[i] ^= 0xff;
        let _ = open_bytes(&b);
    }
}
