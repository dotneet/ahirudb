//! `UUID` / `TIMESTAMPTZ` の統合テスト。
//!
//! 期待値は `duckdb -c "..."` の実際の出力と突き合わせて決めている。

use ahiru_core::error::{code_of, Code};
use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::{Ty, Value};

fn session_with_dual() -> Session {
    let mut s = Session::new();
    s.register_bytes_as("dual", b"x\n1\n".to_vec(), FormatKind::Csv).unwrap();
    s
}

fn run(s: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    let mut q = match s.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
    };
    let mut rows = Vec::new();
    loop {
        match s.step(&mut q).unwrap() {
            QueryStep::Batch(mut b) => {
                b.materialize();
                for r in 0..b.num_rows() {
                    rows.push(b.cols.iter().map(|c| c.value_at(r)).collect());
                }
            }
            QueryStep::Done => break,
            QueryStep::NeedIo(_) | QueryStep::NeedCodec(_) => panic!("unexpected suspend"),
        }
    }
    rows
}

fn uuid_bytes(s: &str) -> Vec<u8> {
    let hex: String = s.chars().filter(|c| *c != '-').collect();
    (0..16).map(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap()).collect()
}

// --- UUID --------------------------------------------------------------------

#[test]
fn uuid_string_roundtrip_through_cast() {
    let mut db = session_with_dual();
    // duckdb: SELECT CAST(CAST('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11' AS UUID) AS VARCHAR)
    let rows = run(
        &mut db,
        "SELECT CAST(CAST('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11' AS UUID) AS VARCHAR) FROM dual",
    );
    assert_eq!(rows, vec![vec![Value::Bytes(b"a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11".to_vec())]]);
}

#[test]
fn uuid_cast_stores_raw_16_bytes_in_rfc_4122_order() {
    let mut db = session_with_dual();
    let rows =
        run(&mut db, "SELECT CAST('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11' AS UUID) FROM dual");
    assert_eq!(rows, vec![vec![Value::Bytes(uuid_bytes("a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"))]]);
}

#[test]
fn uuid_cast_is_case_insensitive_on_input_and_lowercase_on_output() {
    let mut db = session_with_dual();
    let rows = run(
        &mut db,
        "SELECT CAST(CAST('A0EEBC99-9C0B-4EF8-BB6D-6BB9BD380A11' AS UUID) AS VARCHAR) FROM dual",
    );
    assert_eq!(rows, vec![vec![Value::Bytes(b"a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11".to_vec())]]);
}

#[test]
fn uuid_equality_is_case_insensitive_on_input() {
    let mut db = session_with_dual();
    // duckdb: SELECT CAST('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11' AS UUID) =
    //         CAST('A0EEBC99-9C0B-4EF8-BB6D-6BB9BD380A11' AS UUID) -> true
    let rows = run(
        &mut db,
        "SELECT CAST('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11' AS UUID) = \
         CAST('A0EEBC99-9C0B-4EF8-BB6D-6BB9BD380A11' AS UUID) FROM dual",
    );
    assert_eq!(rows, vec![vec![Value::Bool(true)]]);
}

#[test]
fn uuid_ordering_is_unsigned_byte_wise() {
    // duckdb: SELECT u FROM (VALUES
    //   (CAST('ffffffff-0000-0000-0000-000000000000' AS UUID)),
    //   (CAST('00000000-0000-0000-0000-000000000001' AS UUID)),
    //   (CAST('7fffffff-ffff-ffff-ffff-ffffffffffff' AS UUID))
    // ) t(u) ORDER BY u
    // -> 00000000.../1, then 7fffffff.../ffff, then ffffffff.../0000
    let mut db = Session::new();
    db.register_bytes_as(
        "t",
        b"u\nffffffff-0000-0000-0000-000000000000\n00000000-0000-0000-0000-000000000001\n7fffffff-ffff-ffff-ffff-ffffffffffff\n"
            .to_vec(),
        FormatKind::Csv,
    )
    .unwrap();
    let rows = run(
        &mut db,
        "SELECT CAST(u AS VARCHAR) FROM (SELECT CAST(u AS UUID) AS u FROM t) x ORDER BY u",
    );
    assert_eq!(
        rows,
        vec![
            vec![Value::Bytes(b"00000000-0000-0000-0000-000000000001".to_vec())],
            vec![Value::Bytes(b"7fffffff-ffff-ffff-ffff-ffffffffffff".to_vec())],
            vec![Value::Bytes(b"ffffffff-0000-0000-0000-000000000000".to_vec())],
        ]
    );
}

#[test]
fn malformed_uuid_text_becomes_null_not_an_error() {
    // Matches this engine's established convention for DATE/TIME/TIMESTAMP
    // (see expr::kernels::cast_str_to_uuid doc): a row-level parse failure
    // is NULL, not a hard error, for both CAST and TRY_CAST alike. DuckDB
    // itself errors on plain CAST here, but this engine deliberately
    // diverges the same way it already does for temporal types.
    let mut db = session_with_dual();
    let rows = run(&mut db, "SELECT CAST('not-a-uuid' AS UUID) FROM dual");
    assert_eq!(rows, vec![vec![Value::Null]]);
    let rows = run(&mut db, "SELECT TRY_CAST('not-a-uuid' AS UUID) FROM dual");
    assert_eq!(rows, vec![vec![Value::Null]]);
}

#[test]
fn uuid_missing_a_hyphen_is_rejected() {
    let mut db = session_with_dual();
    let rows = run(&mut db, "SELECT CAST('a0eebc999c0b-4ef8-bb6d-6bb9bd380a11' AS UUID) FROM dual");
    assert_eq!(rows, vec![vec![Value::Null]]);
}

#[test]
fn uuid_to_blob_and_blob_to_uuid_are_not_supported() {
    // Only VARCHAR <-> UUID is supported, mirroring the existing VARCHAR <->
    // JSON special case (expr::kernels::cast_impl doc). Raw-byte access to a
    // UUID should go through VARCHAR, not BLOB. `Cast`'s legality is only
    // checked at execution time (`plan::compile`'s `coerce_with` defers all
    // validation to the `Cast` instruction), so this surfaces as a `step()`
    // error, not a `prepare()` error.
    let mut db = session_with_dual();
    let mut q = match db
        .prepare(
            "SELECT CAST(CAST('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11' AS UUID) AS BLOB) FROM dual",
            &[],
        )
        .unwrap()
    {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("unexpected NeedIo"),
    };
    assert_eq!(code_of(db.step(&mut q)), Some(Code::InvalidCast));
}

#[test]
fn uuid_does_not_implicitly_unify_with_varchar() {
    // Consistent with this engine's existing convention that DATE/TIMESTAMP
    // also don't implicitly coerce a bare string literal (confirmed: `WHERE
    // date_col = '2020-01-01'` is also a TypeMismatch here, unlike DuckDB
    // which does coerce). UUID follows the same "explicit CAST required"
    // rule for consistency, even though DuckDB itself does implicitly
    // coerce a string literal to UUID in a comparison.
    let mut db = session_with_dual();
    let err = db.prepare(
        "SELECT 1 FROM dual WHERE CAST('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11' AS UUID) = 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11'",
        &[],
    );
    assert_eq!(code_of(err), Some(Code::TypeMismatch));
}

#[test]
fn uuid_result_schema_reports_the_uuid_type_name() {
    // `DESCRIBE` only supports `DESCRIBE <table>`, not a derived-table
    // subquery (a pre-existing limitation unrelated to UUID/TIMESTAMPTZ),
    // so check the computed column's type via `Query::schema` directly.
    let mut db = session_with_dual();
    let Prepared::Ready(q) = db
        .prepare("SELECT CAST('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11' AS UUID) AS u FROM dual", &[])
        .unwrap()
    else {
        panic!("unexpected NeedIo");
    };
    assert_eq!(q.schema[0].ty, Ty::Uuid);
    assert_eq!(q.schema[0].ty.name(), "UUID");
}

// --- TIMESTAMPTZ ---------------------------------------------------------------

#[test]
fn timestamptz_without_offset_is_treated_as_utc() {
    // No session timezone concept (docs/sql/types.md), so a bare TIMESTAMPTZ
    // literal with no offset has the same underlying instant as the
    // equivalent TIMESTAMP literal — check this by comparing against the
    // already-well-tested TIMESTAMP parser rather than a hand-computed
    // epoch value.
    let mut db = session_with_dual();
    let tz = run(&mut db, "SELECT CAST('2024-01-01 12:00:00' AS TIMESTAMPTZ) FROM dual");
    let ts = run(&mut db, "SELECT CAST('2024-01-01 12:00:00' AS TIMESTAMP) FROM dual");
    assert_eq!(tz, ts);
}

#[test]
fn timestamptz_with_positive_offset_converts_to_utc() {
    let mut db = session_with_dual();
    // 12:00 at +09:00 is 03:00 UTC.
    let rows = run(
        &mut db,
        "SELECT CAST('2024-01-01 12:00:00+09' AS TIMESTAMPTZ) = CAST('2024-01-01 03:00:00' AS TIMESTAMPTZ) FROM dual",
    );
    assert_eq!(rows, vec![vec![Value::Bool(true)]]);
}

#[test]
fn timestamptz_with_negative_offset_and_minutes_converts_to_utc() {
    let mut db = session_with_dual();
    // 12:00 at -05:30 is 17:30 UTC.
    let rows = run(
        &mut db,
        "SELECT CAST('2024-01-01 12:00:00-05:30' AS TIMESTAMPTZ) = CAST('2024-01-01 17:30:00' AS TIMESTAMPTZ) FROM dual",
    );
    assert_eq!(rows, vec![vec![Value::Bool(true)]]);
}

#[test]
fn timestamptz_accepts_a_z_suffix_as_utc() {
    let mut db = session_with_dual();
    let a = run(&mut db, "SELECT CAST('2024-01-01 12:00:00Z' AS TIMESTAMPTZ) FROM dual");
    let b = run(&mut db, "SELECT CAST('2024-01-01 12:00:00' AS TIMESTAMPTZ) FROM dual");
    assert_eq!(a, b);
}

#[test]
fn timestamptz_display_has_a_plus_00_suffix() {
    let mut db = session_with_dual();
    let rows = run(
        &mut db,
        "SELECT CAST(CAST('2024-01-01 12:00:00+09' AS TIMESTAMPTZ) AS VARCHAR) FROM dual",
    );
    assert_eq!(rows, vec![vec![Value::Bytes(b"2024-01-01 03:00:00+00".to_vec())]]);
}

#[test]
fn timestamp_with_time_zone_spelling_is_accepted() {
    let mut db = session_with_dual();
    let a =
        run(&mut db, "SELECT CAST('2024-01-01 12:00:00' AS TIMESTAMP WITH TIME ZONE) FROM dual");
    let b = run(&mut db, "SELECT CAST('2024-01-01 12:00:00' AS TIMESTAMPTZ) FROM dual");
    assert_eq!(a, b);
}

#[test]
fn timestamptz_unifies_with_timestamp_and_date_by_widening() {
    let mut db = session_with_dual();
    // TIMESTAMP compared against TIMESTAMPTZ should widen to TIMESTAMPTZ and
    // compare by the underlying UTC instant, not error out as a type mismatch.
    let rows = run(
        &mut db,
        "SELECT CAST('2024-01-01 12:00:00' AS TIMESTAMPTZ) = CAST('2024-01-01 12:00:00' AS TIMESTAMP) FROM dual",
    );
    assert_eq!(rows, vec![vec![Value::Bool(true)]]);
    let rows = run(
        &mut db,
        "SELECT CAST('2024-01-01' AS DATE) < CAST('2024-01-01 12:00:00' AS TIMESTAMPTZ) FROM dual",
    );
    assert_eq!(rows, vec![vec![Value::Bool(true)]]);
}

#[test]
fn current_timestamp_is_timestamptz_matching_duckdb() {
    let mut s = session_with_dual();
    s.set_now(1_705_321_800_000_000); // 2024-01-15 12:30:00 UTC
    let Prepared::Ready(q) = s.prepare("SELECT CURRENT_TIMESTAMP AS ts FROM dual", &[]).unwrap()
    else {
        panic!("unexpected NeedIo");
    };
    assert_eq!(q.schema[0].ty, Ty::Timestamptz);
    let rows = run(&mut s, "SELECT CAST(CURRENT_TIMESTAMP AS VARCHAR) FROM dual");
    assert_eq!(rows, vec![vec![Value::Bytes(b"2024-01-15 12:30:00+00".to_vec())]]);
}

#[test]
fn malformed_timestamptz_text_becomes_null_not_an_error() {
    let mut db = session_with_dual();
    let rows = run(&mut db, "SELECT CAST('not-a-timestamp' AS TIMESTAMPTZ) FROM dual");
    assert_eq!(rows, vec![vec![Value::Null]]);
}

#[test]
fn timestamptz_out_of_range_offset_is_rejected() {
    let mut db = session_with_dual();
    let rows = run(&mut db, "SELECT CAST('2024-01-01 12:00:00+25' AS TIMESTAMPTZ) FROM dual");
    assert_eq!(rows, vec![vec![Value::Null]]);
}

#[test]
fn ty_uuid_and_timestamptz_are_isolated_in_the_type_system() {
    // Direct unit-level sanity check on the type-system rules used above,
    // to pin the contract independent of the SQL surface.
    assert_eq!(Ty::Uuid.phys(), ahiru_core::vector::PhysType::Bytes);
    assert_eq!(Ty::Timestamptz.phys(), ahiru_core::vector::PhysType::I64);
    assert!(Ty::Timestamptz.is_temporal());
    assert_eq!(Ty::unify(Ty::Uuid, Ty::Varchar), None);
    assert_eq!(Ty::unify(Ty::Uuid, Ty::Blob), None);
    assert_eq!(Ty::unify(Ty::Uuid, Ty::Uuid), Some(Ty::Uuid));
    assert_eq!(Ty::unify(Ty::Null, Ty::Uuid), Some(Ty::Uuid));
    assert_eq!(Ty::unify(Ty::Timestamp, Ty::Timestamptz), Some(Ty::Timestamptz));
    assert_eq!(Ty::unify(Ty::Date, Ty::Timestamptz), Some(Ty::Timestamptz));
}

// --- COPY TO round trip (feature `export`) --------------------------------

#[cfg(feature = "export")]
mod copy_roundtrip {
    use super::*;

    fn copy_bytes(sess: &mut Session, sql: &str) -> Vec<u8> {
        match sess.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
            Prepared::Ready(q) => q.copy.expect("COPY statement must set Query::copy").data,
            Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
        }
    }

    #[test]
    fn uuid_and_timestamptz_export_as_text_in_csv() {
        let mut sess = session_with_dual();
        let out = copy_bytes(
            &mut sess,
            "COPY (SELECT CAST('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11' AS UUID) AS u, \
             CAST('2024-01-01 12:00:00+09' AS TIMESTAMPTZ) AS ts FROM dual) TO 'out.csv'",
        );
        let text = String::from_utf8(out).unwrap();
        assert_eq!(text, "u,ts\na0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11,2024-01-01 03:00:00+00\n");
    }

    #[test]
    fn uuid_and_timestamptz_export_as_text_in_jsonl() {
        let mut sess = session_with_dual();
        let out = copy_bytes(
            &mut sess,
            "COPY (SELECT CAST('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11' AS UUID) AS u, \
             CAST('2024-01-01 12:00:00+09' AS TIMESTAMPTZ) AS ts FROM dual) TO 'out.jsonl' (FORMAT jsonl)",
        );
        let text = String::from_utf8(out).unwrap();
        assert_eq!(
            text,
            "{\"u\":\"a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11\",\"ts\":\"2024-01-01 03:00:00+00\"}\n"
        );
    }
}

// --- DDL column types (feature `dml`, which implies `ddl`; needs INSERT) ---

#[cfg(feature = "dml")]
mod ddl_column_types {
    use super::*;

    #[test]
    fn create_table_accepts_uuid_and_timestamptz_columns() {
        let mut sess = Session::new();
        run(&mut sess, "CREATE TABLE t (id UUID, seen_at TIMESTAMPTZ)");
        run(
            &mut sess,
            "INSERT INTO t VALUES (CAST('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11' AS UUID), \
             CAST('2024-01-01 12:00:00' AS TIMESTAMPTZ))",
        );
        let rows = run(&mut sess, "SELECT CAST(id AS VARCHAR), CAST(seen_at AS VARCHAR) FROM t");
        assert_eq!(
            rows,
            vec![vec![
                Value::Bytes(b"a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11".to_vec()),
                Value::Bytes(b"2024-01-01 12:00:00+00".to_vec()),
            ]]
        );
    }
}
