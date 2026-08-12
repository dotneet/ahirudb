//! Real-file integration tests for LIST/MAP (Dremel assembly).
//!
//! Reads end-to-end via `TableFormat` (`resolve` -> `read_split`), also verifying
//! `format::parquet`'s handling of "a nested column can span multiple physical column
//! chunks" (the nested branch of `split_ranges`/`codec_tasks`/`read_split`). Expected
//! values are hand-transcribed from the output of
//! `duckdb -c "SELECT id, to_json(...) FROM '...'"`, adjusted to match this crate's JSON
//! serialization format (`parquet::nested`) -- no whitespace, and MAP as an array of
//! key/value pairs (`[{"key":..,"value":..}, ...]`). See the "LIST/MAP (Dremel
//! assembly)" section of `scripts/gen-testdata.sh` for how they were generated.

use ahiru_core::catalog::Source;
use ahiru_core::format::parquet::ParquetFormat;
use ahiru_core::format::{PruneOp, Pruner, TableFormat};
use ahiru_core::vector::{Ty, Value, Vector};

fn data(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Opens a file and reads every column in `projection`, vertically, across all RowGroups.
fn read_all(bytes: Vec<u8>, projection: &[usize]) -> (Vec<ahiru_core::vector::Field>, Vec<Vector>) {
    let mut fmt = ParquetFormat::new();
    let src = Source::from_bytes(bytes);
    fmt.resolve(&src).unwrap().unwrap();
    let schema = fmt.schema().to_vec();

    let mut cols: Vec<Option<Vector>> = projection.iter().map(|_| None).collect();
    for split in 0..fmt.num_splits() {
        let part = fmt.read_split(&src, split, projection).unwrap();
        for (i, v) in part.into_iter().enumerate() {
            match &mut cols[i] {
                Some(acc) => {
                    let mut merged = Vector::with_capacity(v.ty(), acc.len() + v.len());
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
    (schema, cols.into_iter().map(|c| c.unwrap()).collect())
}

fn json_str(v: &Value) -> Option<String> {
    match v {
        Value::Bytes(b) => Some(String::from_utf8(b.clone()).expect("valid utf8 json")),
        Value::Null => None,
        other => panic!("expected Bytes or Null, got {other:?}"),
    }
}

#[test]
fn list_of_int_matches_duckdb() {
    // duckdb schema: id INTEGER, xs INTEGER[]  (see `scripts/gen-testdata.sh`)
    // duckdb -c "SELECT id, to_json(xs) FROM 'list1.parquet' ORDER BY id;" gives
    // [1,2,3] for all 10 rows.
    let (schema, cols) = read_all(data("list1.parquet"), &[0, 1]);
    assert_eq!(schema[0].ty, Ty::Int);
    assert_eq!(schema[1].ty, Ty::Json);
    assert!(schema[1].nullable);
    assert_eq!(cols[0].len(), 10);
    for i in 0..10 {
        assert_eq!(cols[0].value_at(i), Value::I32(i as i32));
        assert_eq!(json_str(&cols[1].value_at(i)).unwrap(), "[1,2,3]");
    }
}

#[test]
fn list_varied_distinguishes_null_empty_and_inner_nulls() {
    // duckdb -c "SELECT id, to_json(xs) FROM 'list_varied.parquet' ORDER BY id LIMIT 10;"
    //  0 NULL / 1 [] / 2 [2] / 3 [3,null,6] / 4 [4,5,6,7] / 5 NULL / 6 [] / ...
    let (_, cols) = read_all(data("list_varied.parquet"), &[0, 1]);
    assert_eq!(cols[0].len(), 50);
    for i in 0..10u32 {
        let got = cols[1].value_at(i as usize);
        match i % 5 {
            0 => assert_eq!(got, Value::Null, "row {i}: list itself must be SQL NULL"),
            1 => assert_eq!(json_str(&got).unwrap(), "[]", "row {i}: empty array"),
            2 => assert_eq!(json_str(&got).unwrap(), format!("[{i}]"), "row {i}"),
            3 => assert_eq!(
                json_str(&got).unwrap(),
                format!("[{i},null,{}]", i * 2),
                "row {i}: null element inside the array"
            ),
            4 => assert_eq!(
                json_str(&got).unwrap(),
                format!("[{i},{},{},{}]", i + 1, i + 2, i + 3),
                "row {i}"
            ),
            _ => unreachable!(),
        }
    }
}

#[test]
fn list_of_struct_matches_duckdb() {
    // duckdb -c "SELECT id, to_json(items) FROM 'list_of_struct.parquet' ORDER BY id LIMIT 3;"
    // [{"a":0,"b":"s0"},{"a":1,"b":null}] / [{"a":1,"b":"s1"},{"a":2,"b":null}] / ...
    let (schema, cols) = read_all(data("list_of_struct.parquet"), &[0, 1]);
    assert_eq!(schema[1].name, "items");
    assert_eq!(schema[1].ty, Ty::Json);
    for i in 0..3u32 {
        let expect = format!("[{{\"a\":{i},\"b\":\"s{i}\"}},{{\"a\":{},\"b\":null}}]", i + 1);
        assert_eq!(json_str(&cols[1].value_at(i as usize)).unwrap(), expect, "row {i}");
    }
}

#[test]
fn struct_with_list_becomes_one_json_column_not_flattened() {
    // Kept separate from the existing STRUCT flattening (the address.city approach) --
    // column `s` itself becomes a single JSON column (it does not become the
    // dot-separated s.name/s.tags).
    // duckdb -c "SELECT id, to_json(s) FROM 'struct_with_list.parquet' ORDER BY id LIMIT 3;"
    // {"name":"n0","tags":["t0","t1"]} / {"name":"n1","tags":["t1","t2"]} / ...
    let (schema, cols) = read_all(data("struct_with_list.parquet"), &[0, 1]);
    assert_eq!(schema.len(), 2);
    assert_eq!(schema[1].name, "s");
    assert_eq!(schema[1].ty, Ty::Json);
    for i in 0..3u32 {
        let expect = format!("{{\"name\":\"n{i}\",\"tags\":[\"t{i}\",\"t{}\"]}}", i + 1);
        assert_eq!(json_str(&cols[1].value_at(i as usize)).unwrap(), expect, "row {i}");
    }
}

#[test]
fn list_of_struct_with_list_handles_three_levels_of_nesting() {
    // LIST<STRUCT<..., LIST<...>>>: three levels -- array -> struct -> array.
    // list_of_struct/struct_with_list only ever combined up to 2 levels, so this verifies
    // that Dremel assembly correctly stacks repetition/definition levels at 3+ levels too
    // (see `scripts/gen-testdata.sh`).
    // duckdb -c "SELECT id, to_json(items) FROM 'list_of_struct_with_list.parquet' ORDER BY id LIMIT 3;"
    let (schema, cols) = read_all(data("list_of_struct_with_list.parquet"), &[0, 1]);
    assert_eq!(schema[1].name, "items");
    assert_eq!(schema[1].ty, Ty::Json);
    for i in 0..3u32 {
        let expect = format!(
            "[{{\"name\":\"n{i}\",\"tags\":[\"t{i}\",\"t{}\"]}},{{\"name\":\"x\",\"tags\":[]}}]",
            i + 1
        );
        assert_eq!(json_str(&cols[1].value_at(i as usize)).unwrap(), expect, "row {i}");
    }
}

#[test]
fn list_of_list_handles_nested_repetition() {
    // duckdb -c "SELECT id, to_json(xss) FROM 'list_of_list.parquet' ORDER BY id LIMIT 3;"
    // [[0,1],[],[0]] / [[1,2],[],[10]] / [[2,3],[],[20]]
    let (_, cols) = read_all(data("list_of_list.parquet"), &[0, 1]);
    for i in 0..3u32 {
        let expect = format!("[[{i},{}],[],[{}]]", i + 1, i * 10);
        assert_eq!(json_str(&cols[1].value_at(i as usize)).unwrap(), expect, "row {i}");
    }
}

#[test]
fn map_with_string_keys_matches_duckdb_values() {
    // duckdb -c "SELECT id, to_json(m) FROM 'map_basic.parquet' ORDER BY id LIMIT 3;"
    // {"a":0,"b":0,"c":null} / {"a":1,"b":2,"c":null} / {"a":2,"b":4,"c":null}
    // This crate assembles MAP as an array of key/value pairs
    // (see the comment on `parquet::nested`; there's no obligation to match DuckDB's
    // to_json -- we're just verifying the values are correct).
    let (_, cols) = read_all(data("map_basic.parquet"), &[0, 1]);
    for i in 0..3u32 {
        let expect = format!(
            "[{{\"key\":\"a\",\"value\":{i}}},{{\"key\":\"b\",\"value\":{}}},{{\"key\":\"c\",\"value\":null}}]",
            i * 2
        );
        assert_eq!(json_str(&cols[1].value_at(i as usize)).unwrap(), expect, "row {i}");
    }
}

#[test]
fn map_with_non_string_keys_reads_without_special_casing() {
    // duckdb -c "SELECT id, to_json(m) FROM 'map_int_key.parquet' ORDER BY id LIMIT 3;"
    // {"0":"v0","1":"v1"} / {"1":"v1","2":"v2"} / {"2":"v2","3":"v3"}
    let (_, cols) = read_all(data("map_int_key.parquet"), &[0, 1]);
    for i in 0..3u32 {
        let expect = format!(
            "[{{\"key\":{i},\"value\":\"v{i}\"}},{{\"key\":{},\"value\":\"v{}\"}}]",
            i + 1,
            i + 1
        );
        assert_eq!(json_str(&cols[1].value_at(i as usize)).unwrap(), expect, "row {i}");
    }
}

/// Verifies the branch (`None if desc.nested.is_some()` in `format::parquet::read_split`)
/// taken when a LIST column and statistics-based per-page pruning
/// (`refine_with_index`) mix in the same projection.
///
/// Even when page selection is enabled by an equality pruner on `id`, `xs` (the nested
/// column) must correctly fall back to "read the whole column chunk as excluded from page
/// selection, then gather into `kept_ranges` afterward" (if it doesn't fall back, another
/// column's `ColumnPagePlan` would get mistakenly reused).
#[test]
fn list_column_survives_page_pruning_on_a_sibling_column() {
    // Made with pyarrow: id 0..2000, xs is an empty array when id%7==0, otherwise
    // [id, id+1]. Has ColumnIndex/OffsetIndex (see `scripts/gen-testdata.sh`).
    let bytes = data("list_pagetest.parquet");
    let mut fmt = ParquetFormat::new();
    let src = Source::from_bytes(bytes);
    fmt.resolve(&src).unwrap().unwrap();

    let projection = vec![0usize, 1];
    let pruners =
        vec![Pruner { column: 0, op: PruneOp::Eq, value: Value::I32(1234), in_values: Vec::new() }];

    assert!(fmt.may_match(0, &pruners, &projection));
    let mut idx_ranges = Vec::new();
    fmt.index_ranges(0, &pruners, &projection, &mut idx_ranges).unwrap();
    // Since `Source::from_bytes` holds the whole file, there's no need to simulate an
    // extra fetch (`src.get` always succeeds).
    let keep = fmt.refine_with_index(&src, 0, &pruners, &projection).unwrap();
    assert!(keep);

    let mut ranges = Vec::new();
    fmt.split_ranges(0, &projection, &mut ranges).unwrap();
    assert!(!ranges.is_empty());

    let cols = fmt.read_split(&src, 0, &projection).unwrap();
    assert_eq!(cols[0].len(), cols[1].len());

    // If page selection is working, this should be fewer than all 2000 rows (correctness
    // can be verified even if it isn't, so this isn't a mandatory assert).
    let hit = (0..cols[0].len()).find(|&i| cols[0].value_at(i) == Value::I32(1234));
    let hit = hit.expect("id=1234 must survive page pruning (its page must be kept)");
    assert_eq!(json_str(&cols[1].value_at(hit)).unwrap(), "[1234,1235]");

    // For every row that survives page selection, verify that `xs`'s value matches the
    // value derived from `id` (checks that the gather indices haven't drifted).
    for i in 0..cols[0].len() {
        let Value::I32(id) = cols[0].value_at(i) else { panic!("expected I32") };
        let got = json_str(&cols[1].value_at(i));
        if id % 7 == 0 {
            assert_eq!(got.unwrap(), "[]", "id={id}");
        } else {
            assert_eq!(got.unwrap(), format!("[{id},{}]", id + 1), "id={id}");
        }
    }
}
