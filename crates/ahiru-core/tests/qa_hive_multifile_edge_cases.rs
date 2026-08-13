//! QA pass, continued: Hive partition + multi-file glob edge cases exercised
//! through `Session`/SQL, building on `format::partitioned.rs`'s own unit
//! tests and the existing `crates/ahiru-core/tests/multi_file_smoke.rs`
//! (which this file does not edit — see `partition_column_colliding_with_a_
//! real_file_column_is_rejected` there for the base case this file deepens).

use ahiru_core::error::{code_of, Code};
use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn run_all(sql: &str, sess: &mut Session) -> Vec<Vec<Value>> {
    let mut q = match sess.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: in-memory bytes never need IO"),
    };
    let mut rows = Vec::new();
    loop {
        match sess.step(&mut q).unwrap() {
            QueryStep::Batch(mut b) => {
                b.materialize();
                for r in 0..b.num_rows() {
                    rows.push(b.cols.iter().map(|c| c.value_at(r)).collect());
                }
            }
            QueryStep::NeedIo(_) | QueryStep::NeedCodec(_) => panic!("unexpected suspend"),
            QueryStep::Done => break,
        }
    }
    rows
}

#[test]
fn same_directory_repeating_a_partition_key_is_rejected_not_last_value_wins() {
    // `PartitionedFormat::resolve` folds every `(name, value)` pair into a
    // single schema by checking each new name against the ones already
    // added, so a path with the same key twice (`year=2024/year=2025/f.csv`)
    // collides with itself the same way a partition key colliding with a
    // real file column does (`partition_column_colliding_with_a_real_file_
    // column_is_rejected` in `multi_file_smoke.rs`). Confirm it's rejected
    // rather than silently keeping the last (or first) value.
    let mut sess = Session::new();
    let r = sess.register_multi_bytes(
        "t",
        vec![("data/year=2024/year=2025/f.csv".into(), b"id\n1\n".to_vec())],
        FormatKind::Csv,
    );
    // Registration itself succeeds (paths are just parsed); the collision
    // surfaces once the format is actually resolved, i.e. on first query.
    assert!(r.is_ok());
    let q = sess.prepare("SELECT count(*) FROM t", &[]);
    assert_eq!(code_of(q), Some(Code::DuplicateColumn));
}

#[test]
fn parts_with_different_partition_depth_are_rejected_as_a_schema_mismatch() {
    // Each file's Hive columns are parsed independently from its own path
    // (`session.rs`'s `register_multi`), so a glob mixing a fully-partitioned
    // path (`year=2024/month=01/f.csv`, 2 extra columns) with a shallower one
    // (`year=2025/f.csv`, 1 extra column) produces parts with different
    // schema lengths. `catalog::unify_schema` requires every part to have the
    // same column count, so this must be a clear TypeMismatch, not a
    // silently short/ragged union.
    let mut sess = Session::new();
    sess.register_multi_bytes(
        "t",
        vec![
            ("data/year=2024/month=01/a.csv".into(), b"id\n1\n".to_vec()),
            ("data/year=2025/b.csv".into(), b"id\n2\n".to_vec()),
        ],
        FormatKind::Csv,
    )
    .unwrap();
    let r = sess.prepare("SELECT count(*) FROM t", &[]);
    assert_eq!(code_of(r), Some(Code::TypeMismatch));
}

#[test]
fn partition_values_are_usable_in_where_group_by_and_projection() {
    let mut sess = Session::new();
    sess.register_multi_bytes(
        "t",
        vec![
            ("data/region=us/day=1/f.csv".into(), b"id\n1\n2\n".to_vec()),
            ("data/region=us/day=2/f.csv".into(), b"id\n3\n".to_vec()),
            ("data/region=eu/day=1/f.csv".into(), b"id\n4\n".to_vec()),
        ],
        FormatKind::Csv,
    )
    .unwrap();

    let rows =
        run_all("SELECT region, count(*) AS n FROM t GROUP BY region ORDER BY region", &mut sess);
    assert_eq!(
        rows,
        vec![
            vec![Value::Bytes(b"eu".to_vec()), Value::I64(1)],
            vec![Value::Bytes(b"us".to_vec()), Value::I64(3)],
        ]
    );

    let rows = run_all("SELECT id FROM t WHERE region = 'us' AND day = 1 ORDER BY id", &mut sess);
    assert_eq!(rows, vec![vec![Value::I64(1)], vec![Value::I64(2)]]);
}

#[test]
fn selecting_only_partition_columns_still_returns_the_right_row_count() {
    // `PartitionedFormat::inner_projection` special-cases "projection has no
    // inner columns at all" by forcing column 0 through anyway, purely to
    // learn the row count. Exercise that path through actual SQL.
    let mut sess = Session::new();
    sess.register_multi_bytes(
        "t",
        vec![
            ("data/year=2024/f.csv".into(), b"id,name\n1,a\n2,b\n".to_vec()),
            ("data/year=2025/f.csv".into(), b"id,name\n3,c\n".to_vec()),
        ],
        FormatKind::Csv,
    )
    .unwrap();
    let rows = run_all("SELECT year, count(*) FROM t GROUP BY year ORDER BY year", &mut sess);
    assert_eq!(
        rows,
        vec![vec![Value::I32(2024), Value::I64(2)], vec![Value::I32(2025), Value::I64(1)]]
    );
}

#[test]
fn percent_encoded_partition_values_decode_through_sql_filtering() {
    // `format::partitioned`'s own unit test (`percent_encoded_values_are_
    // decoded`) checks `parse_hive_path` in isolation; this confirms the
    // decoded value is actually usable to filter through the full SQL path.
    let mut sess = Session::new();
    sess.register_multi_bytes(
        "t",
        vec![
            ("data/region=us%20east/f.csv".into(), b"id\n1\n".to_vec()),
            ("data/region=us%20west/f.csv".into(), b"id\n2\n".to_vec()),
        ],
        FormatKind::Csv,
    )
    .unwrap();
    let rows = run_all("SELECT id FROM t WHERE region = 'us east'", &mut sess);
    assert_eq!(rows, vec![vec![Value::I64(1)]]);
}

#[test]
fn large_numeric_partition_value_that_overflows_i32_is_bigint_and_comparable() {
    let mut sess = Session::new();
    sess.register_multi_bytes(
        "t",
        vec![("data/ts=99999999999/f.csv".into(), b"id\n1\n".to_vec())],
        FormatKind::Csv,
    )
    .unwrap();
    let rows = run_all("SELECT id FROM t WHERE ts = 99999999999", &mut sess);
    assert_eq!(rows, vec![vec![Value::I64(1)]]);
}

#[test]
fn mixed_int_and_bigint_partition_values_unify_and_compare() {
    // One part infers `ts` as INT (`1`), the other as BIGINT (`99999999999`).
    // Catalog unify widens to BIGINT; the scan must recast the INT part so
    // ORDER BY / comparisons do not hit a mixed physical type.
    let mut sess = Session::new();
    sess.register_multi_bytes(
        "t",
        vec![
            ("data/ts=1/f.csv".into(), b"id\n1\n".to_vec()),
            ("data/ts=99999999999/f.csv".into(), b"id\n2\n".to_vec()),
        ],
        FormatKind::Csv,
    )
    .unwrap();
    let rows = run_all("SELECT id, ts FROM t ORDER BY ts", &mut sess);
    assert_eq!(
        rows,
        vec![vec![Value::I64(1), Value::I64(1)], vec![Value::I64(2), Value::I64(99999999999)],]
    );
    let rows = run_all("SELECT id FROM t WHERE ts > 10 ORDER BY id", &mut sess);
    assert_eq!(rows, vec![vec![Value::I64(2)]]);
}
