//! LIST/MAP（Dremel 組み立て）の実ファイル統合テスト。
//!
//! `TableFormat` 経由（`resolve` → `read_split`）でエンドツーエンドに読み、
//! `format::parquet` 側の「入れ子列は複数の物理列チャンクにまたがる」対応
//! （`split_ranges`/`codec_tasks`/`read_split` の nested 分岐）も一緒に
//! 検証する。期待値は `duckdb -c "SELECT id, to_json(...) FROM '...'"` の
//! 出力を基に、このクレートの JSON 直列化（`parquet::nested`）の書式
//! （空白無し、MAP はキー・バリューの配列 `[{"key":..,"value":..}, ...]`）
//! に合わせて手で書き起こしている。詳しい生成方法は
//! `scripts/gen-testdata.sh` の「LIST/MAP（Dremel 組み立て）」節を参照。

use ahiru_core::catalog::Source;
use ahiru_core::format::parquet::ParquetFormat;
use ahiru_core::format::{PruneOp, Pruner, TableFormat};
use ahiru_core::vector::{Ty, Value, Vector};

fn data(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// ファイルを開き、`projection` の全列を全 RowGroup ぶん縦に読む。
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
    // duckdb schema: id INTEGER, xs INTEGER[]  (`scripts/gen-testdata.sh` 参照)
    // duckdb -c "SELECT id, to_json(xs) FROM 'list1.parquet' ORDER BY id;" は
    // 全 10 行とも [1,2,3]。
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
    // 既存の STRUCT フラット化（address.city 方式）とは切り分けられ、
    // s 列そのものが 1 本の JSON 列になる（ドット区切りの s.name/s.tags には
    // ならない）。
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
    // LIST<STRUCT<..., LIST<...>>>: 配列 → 構造体 → 配列の3段。
    // list_of_struct/struct_with_list はどちらも2段までしか組み合わせて
    // いなかったので、Dremel 組み立てが3段以上でも repetition/definition
    // level を正しく積み重ねられるかを確認する（`scripts/gen-testdata.sh`
    // 参照）。
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
    // このクレートは MAP をキー・バリューの配列として組み立てる
    // （`parquet::nested` のコメント参照。DuckDB の to_json と一致させる
    // 義務は無い、値が正しいことを検証する）。
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

/// LIST 列と、統計に基づくページ単位の絞り込み（`refine_with_index`）が
/// 同じ射影に混ざったときの分岐 (`format::parquet::read_split` の
/// `None if desc.nested.is_some()`) を確認する。
///
/// `id` に等号 pruner が効いてページ選択が有効化されても、`xs`（入れ子列）
/// は「ページ選択の対象外として列チャンク全体を読み、`kept_ranges` へ
/// 後から gather する」フォールバックに正しく落ちる必要がある
/// （落ちなければ他列の `ColumnPagePlan` を誤って使い回してしまう）。
#[test]
fn list_column_survives_page_pruning_on_a_sibling_column() {
    // pyarrow 製、id 0..2000、xs は id%7==0 なら空配列、それ以外は
    // [id, id+1]。ColumnIndex/OffsetIndex 付き（`scripts/gen-testdata.sh`
    // 参照）。
    let bytes = data("list_pagetest.parquet");
    let mut fmt = ParquetFormat::new();
    let src = Source::from_bytes(bytes);
    fmt.resolve(&src).unwrap().unwrap();

    let projection = vec![0usize, 1];
    let pruners = vec![Pruner { column: 0, op: PruneOp::Eq, value: Value::I32(1234) }];

    assert!(fmt.may_match(0, &pruners, &projection));
    let mut idx_ranges = Vec::new();
    fmt.index_ranges(0, &pruners, &projection, &mut idx_ranges).unwrap();
    // `Source::from_bytes` はファイル全体を持っているので、追加取得の
    // シミュレーションは要らない（`src.get` は常に成功する）。
    let keep = fmt.refine_with_index(&src, 0, &pruners, &projection).unwrap();
    assert!(keep);

    let mut ranges = Vec::new();
    fmt.split_ranges(0, &projection, &mut ranges).unwrap();
    assert!(!ranges.is_empty());

    let cols = fmt.read_split(&src, 0, &projection).unwrap();
    assert_eq!(cols[0].len(), cols[1].len());

    // ページ選択が効いていれば全 2000 行より少ないはず（効いていなくても
    // 正しさの検証はできるので必須の assert にはしない）。
    let hit = (0..cols[0].len()).find(|&i| cols[0].value_at(i) == Value::I32(1234));
    let hit = hit.expect("id=1234 must survive page pruning (its page must be kept)");
    assert_eq!(json_str(&cols[1].value_at(hit)).unwrap(), "[1234,1235]");

    // ページ選択後に生き残った全行で、xs の値が id から導出した期待値と
    // 一致することを確認する（gather のインデックスがずれていないか）。
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
