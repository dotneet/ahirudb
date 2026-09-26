//! Regression tests for write-path bugs found against DuckDB 1.4.4: JSON values that
//! broke NDJSON lines, UPDATE evaluating SET on rows its WHERE excluded, CTAS copying
//! NOT NULL flags and rejecting duplicate names, DECIMAL type names without
//! precision/scale, DROP TABLE on file tables, compressed COPY destinations, type-name
//! aliases, and column DEFAULTs. Expected values come from the `duckdb` CLI.

#![cfg(feature = "dml")]

use ahiru_core::error::{code_of, Code};
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn run(sess: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    let mut q = match sess.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
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
            QueryStep::Done => break,
            QueryStep::NeedIo(_) | QueryStep::NeedCodec(_) => panic!("unexpected suspend"),
        }
    }
    rows
}

fn exec(sess: &mut Session, sql: &str) {
    sess.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}"));
}

fn s(v: &str) -> Value {
    Value::Bytes(v.as_bytes().to_vec())
}

/// `(column_name, column_type, null)` rows of `DESCRIBE <target>`.
fn describe(sess: &mut Session, target: &str) -> Vec<(String, String, String)> {
    let text = |v: &Value| match v {
        Value::Bytes(b) => String::from_utf8(b.clone()).unwrap(),
        other => panic!("{other:?}"),
    };
    run(sess, &format!("DESCRIBE {target}"))
        .iter()
        .map(|r| (text(&r[0]), text(&r[1]), text(&r[2])))
        .collect()
}

fn names(sess: &mut Session, target: &str) -> Vec<String> {
    describe(sess, target).into_iter().map(|(n, _, _)| n).collect()
}

// --- UPDATE evaluates SET only for the rows it updates ----------------------

#[test]
fn update_does_not_evaluate_set_on_rows_the_where_clause_excludes() {
    let mut sess = Session::new();
    exec(&mut sess, "CREATE TABLE t (s VARCHAR, j JSON)");
    exec(&mut sess, "INSERT INTO t VALUES ('[1]', NULL), ('not json', NULL)");
    let n = run(&mut sess, "UPDATE t SET j = CAST(s AS JSON) WHERE s LIKE '[%'");
    assert_eq!(n, vec![vec![Value::I64(1)]]);
    assert_eq!(
        run(&mut sess, "SELECT s, CAST(j AS VARCHAR) FROM t ORDER BY s"),
        vec![vec![s("[1]"), s("[1]")], vec![s("not json"), Value::Null]]
    );

    // factorial(40) overflows; the n = 40 row is not updated, so it must not abort.
    exec(&mut sess, "CREATE TABLE f (n INTEGER, v HUGEINT)");
    exec(&mut sess, "INSERT INTO f VALUES (5, NULL), (40, NULL)");
    exec(&mut sess, "UPDATE f SET v = factorial(n) WHERE n < 30");
    assert_eq!(
        run(&mut sess, "SELECT n, v FROM f ORDER BY n"),
        vec![vec![Value::I32(5), Value::I128(120)], vec![Value::I32(40), Value::Null]]
    );
}

#[test]
fn update_with_a_sparse_where_updates_exactly_the_matched_rows_across_batches() {
    // The SET results are computed on the narrowed batch; each one must land back on
    // the row it came from, in every batch.
    let mut sess = Session::new();
    exec(&mut sess, "CREATE TABLE t (id INTEGER, v INTEGER)");
    exec(&mut sess, "INSERT INTO t SELECT CAST(range AS INTEGER), 0 FROM range(5000)");
    let n = run(&mut sess, "UPDATE t SET v = id * 2 WHERE id % 3 = 0");
    assert_eq!(n, vec![vec![Value::I64(1667)]]);
    assert_eq!(
        run(
            &mut sess,
            "SELECT count(*) FROM t WHERE v <> CASE WHEN id % 3 = 0 THEN id * 2 ELSE 0 END"
        ),
        vec![vec![Value::I64(0)]]
    );
}

// --- CREATE TABLE AS ---------------------------------------------------------

#[test]
fn ctas_columns_are_always_nullable() {
    let mut sess = Session::new();
    exec(&mut sess, "CREATE TABLE src (a INTEGER NOT NULL, b VARCHAR NOT NULL)");
    exec(&mut sess, "INSERT INTO src VALUES (1, 'x')");
    exec(&mut sess, "CREATE TABLE t AS SELECT * FROM src");
    assert!(describe(&mut sess, "t").iter().all(|(_, _, null)| null == "YES"));
    exec(&mut sess, "INSERT INTO t VALUES (NULL, NULL)");
    assert_eq!(run(&mut sess, "SELECT count(*) FROM t"), vec![vec![Value::I64(2)]]);
}

#[test]
fn ctas_renames_duplicate_output_names_like_duckdb() {
    let mut sess = Session::new();
    exec(&mut sess, "CREATE TABLE a (id INTEGER, x INTEGER)");
    exec(&mut sess, "CREATE TABLE b (id INTEGER, y INTEGER)");
    exec(&mut sess, "INSERT INTO a VALUES (1, 2)");
    exec(&mut sess, "INSERT INTO b VALUES (1, 3)");
    exec(&mut sess, "CREATE TABLE j AS SELECT * FROM a JOIN b ON a.id = b.id");
    assert_eq!(names(&mut sess, "j"), ["id", "x", "id_1", "y"]);
    assert_eq!(
        run(&mut sess, "SELECT id, x, id_1, y FROM j"),
        vec![vec![Value::I32(1), Value::I32(2), Value::I32(1), Value::I32(3)]]
    );

    // Case-insensitive collisions, and a suffix that is already taken.
    exec(&mut sess, "CREATE TABLE k AS SELECT 1 AS a, 2 AS a, 3 AS A, 4 AS a_1 FROM range(1)");
    assert_eq!(names(&mut sess, "k"), ["a", "a_1", "A_2", "a_1_1"]);
    exec(&mut sess, "CREATE TABLE k2 AS SELECT 1 AS a_1, 2 AS a, 3 AS a FROM range(1)");
    assert_eq!(names(&mut sess, "k2"), ["a_1", "a", "a_2"]);

    // An explicit column list still rejects a real duplicate, as DuckDB does.
    let r = sess.prepare("CREATE TABLE bad (c INTEGER, C INTEGER)", &[]);
    assert_eq!(code_of(r), Some(Code::DuplicateColumn));
}

#[test]
fn view_with_duplicate_output_names_exposes_renamed_columns() {
    let mut sess = Session::new();
    exec(&mut sess, "CREATE VIEW v AS SELECT 1 AS a, 2 AS a FROM range(1)");
    assert_eq!(names(&mut sess, "v"), ["a", "a_1"]);
    assert_eq!(run(&mut sess, "SELECT a, a_1 FROM v"), vec![vec![Value::I32(1), Value::I32(2)]]);
}

// --- Type names ----------------------------------------------------------------

#[test]
fn describe_and_typeof_show_decimal_precision_and_scale() {
    let mut sess = Session::new();
    exec(&mut sess, "CREATE TABLE t (d DECIMAL(10,2), e DECIMAL, f NUMERIC(5))");
    let types: Vec<String> = describe(&mut sess, "t").into_iter().map(|(_, t, _)| t).collect();
    assert_eq!(types, ["DECIMAL(10,2)", "DECIMAL(18,3)", "DECIMAL(5,0)"]);
    exec(&mut sess, "INSERT INTO t VALUES (1, 1, 1)");
    assert_eq!(
        run(&mut sess, "SELECT typeof(CAST(1 AS DECIMAL(5,2))), typeof(d), typeof(1) FROM t"),
        vec![vec![s("DECIMAL(5,2)"), s("DECIMAL(10,2)"), s("INTEGER")]]
    );
}

#[test]
fn common_type_name_aliases_are_accepted() {
    // duckdb -c "SELECT typeof(CAST(NULL AS <alias>))"
    let cases = [
        ("VARCHAR(10)", "VARCHAR"),
        ("CHAR(3)", "VARCHAR"),
        ("CHARACTER VARYING(5)", "VARCHAR"),
        ("CHARACTER", "VARCHAR"),
        ("BPCHAR", "VARCHAR"),
        ("NVARCHAR", "VARCHAR"),
        ("DECIMAL(10)", "DECIMAL(10,0)"),
        ("NUMERIC(10)", "DECIMAL(10,0)"),
        ("DEC(5,2)", "DECIMAL(5,2)"),
        ("DOUBLE PRECISION", "DOUBLE"),
        ("INT1", "TINYINT"),
        ("INT2", "SMALLINT"),
        ("INT4", "INTEGER"),
        ("INT8", "BIGINT"),
        ("LONG", "BIGINT"),
        ("SHORT", "SMALLINT"),
        ("SIGNED", "INTEGER"),
        ("INT128", "HUGEINT"),
        ("UINT64", "UBIGINT"),
        ("FLOAT4", "FLOAT"),
        ("FLOAT8", "DOUBLE"),
        ("FLOAT(10)", "FLOAT"),
        ("FLOAT(30)", "DOUBLE"),
        ("BINARY", "BLOB"),
        ("VARBINARY", "BLOB"),
        ("LOGICAL", "BOOLEAN"),
        ("GUID", "UUID"),
        ("TIMESTAMP WITHOUT TIME ZONE", "TIMESTAMP"),
        ("TIME WITHOUT TIME ZONE", "TIME"),
    ];
    let mut sess = Session::new();
    for (alias, want) in cases {
        let rows = run(&mut sess, &format!("SELECT typeof(CAST(NULL AS {alias})) FROM range(1)"));
        assert_eq!(rows, vec![vec![s(want)]], "{alias}");
    }
    exec(&mut sess, "CREATE TABLE t (a VARCHAR(10), b INT8, c DOUBLE PRECISION NOT NULL)");
    let types: Vec<String> = describe(&mut sess, "t").into_iter().map(|(_, t, _)| t).collect();
    assert_eq!(types, ["VARCHAR", "BIGINT", "DOUBLE"]);
    // Modifiers DuckDB rejects stay rejected.
    for bad in ["VARCHAR(5,3)", "DECIMAL(39)", "DECIMAL(0)", "INTEGER(5)"] {
        let r = sess.prepare(&format!("SELECT CAST(NULL AS {bad}) FROM range(1)"), &[]);
        assert!(r.is_err(), "{bad}");
    }
}

// --- DROP TABLE on a file-backed table -------------------------------------------

#[cfg(feature = "csv")]
#[test]
fn drop_table_on_a_file_backed_table_is_read_only() {
    let mut sess = Session::new();
    sess.register_bytes_as("f", b"a\n1\n".to_vec(), ahiru_core::FormatKind::Csv).unwrap();
    assert_eq!(code_of(sess.prepare("DROP TABLE f", &[])), Some(Code::ReadOnlyTable));
    assert_eq!(code_of(sess.prepare("DROP TABLE IF EXISTS f", &[])), Some(Code::ReadOnlyTable));
    assert_eq!(run(&mut sess, "SELECT count(*) FROM f"), vec![vec![Value::I64(1)]]);
    assert_eq!(code_of(sess.prepare("DROP TABLE nope", &[])), Some(Code::TableNotFound));
}

// --- DEFAULT -----------------------------------------------------------------------

#[test]
fn column_defaults_fill_omitted_columns_default_keyword_and_default_values() {
    let mut sess = Session::new();
    exec(
        &mut sess,
        "CREATE TABLE t (a INTEGER DEFAULT 5 NOT NULL, b VARCHAR NOT NULL DEFAULT 'x', c INTEGER)",
    );
    exec(&mut sess, "INSERT INTO t VALUES (DEFAULT, DEFAULT, 1)");
    exec(&mut sess, "INSERT INTO t DEFAULT VALUES");
    exec(&mut sess, "INSERT INTO t (c) VALUES (3)");
    exec(&mut sess, "INSERT INTO t (c, a) VALUES (4, DEFAULT), (5, 7)");
    assert_eq!(
        run(&mut sess, "SELECT a, b, c FROM t"),
        vec![
            vec![Value::I32(5), s("x"), Value::I32(1)],
            vec![Value::I32(5), s("x"), Value::Null],
            vec![Value::I32(5), s("x"), Value::I32(3)],
            vec![Value::I32(5), s("x"), Value::I32(4)],
            vec![Value::I32(7), s("x"), Value::I32(5)],
        ]
    );
}

#[test]
fn column_default_errors() {
    let mut sess = Session::new();
    // A default that does not fit the column type is rejected up front, and no
    // table is left behind.
    let r = sess.prepare("CREATE TABLE u (a TINYINT DEFAULT 1000)", &[]);
    assert_eq!(code_of(r), Some(Code::ValueOutOfRange));
    assert_eq!(code_of(sess.prepare("SELECT * FROM u", &[])), Some(Code::TableNotFound));

    // A NOT NULL column without a default cannot take DEFAULT VALUES.
    exec(&mut sess, "CREATE TABLE w (a INTEGER NOT NULL)");
    assert_eq!(
        code_of(sess.prepare("INSERT INTO w DEFAULT VALUES", &[])),
        Some(Code::TypeMismatch)
    );
    assert_eq!(
        code_of(sess.prepare("INSERT INTO w VALUES (DEFAULT)", &[])),
        Some(Code::TypeMismatch)
    );

    // `ADD COLUMN` accepts the constraints in either order too.
    exec(&mut sess, "ALTER TABLE w ADD COLUMN b INTEGER DEFAULT -8 * 2 NOT NULL");
    exec(&mut sess, "INSERT INTO w (a) VALUES (1)");
    assert_eq!(run(&mut sess, "SELECT a, b FROM w"), vec![vec![Value::I32(1), Value::I32(-16)]]);
}

// --- COPY ... TO -------------------------------------------------------------------

#[cfg(feature = "export")]
fn copy(sess: &mut Session, sql: &str) -> ahiru_core::session::CopyResult {
    match sess.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
        Prepared::Ready(q) => q.copy.expect("no COPY result"),
        Prepared::NeedIo(_) => panic!("unexpected NeedIo"),
    }
}

#[cfg(all(feature = "export", feature = "jsonl"))]
#[test]
fn copy_jsonl_minifies_pretty_printed_json_values() {
    let mut sess = Session::new();
    let r = copy(
        &mut sess,
        "COPY (SELECT CAST('{' || chr(10) || '\"a\" : [1, \" x  y\"]}' AS JSON) AS j FROM range(1)) TO 'x.jsonl'",
    );
    // duckdb writes {"j":{"a":[1," x  y"]}}: whitespace inside strings is kept.
    assert_eq!(String::from_utf8(r.data).unwrap(), "{\"j\":{\"a\":[1,\" x  y\"]}}\n");

    // A pretty-printed JSON file: one output line per record.
    let pretty = b"[\n  {\n    \"id\": 1,\n    \"o\": {\n      \"k\": [1,\n 2],\n \"s\": \"a b\"\n    }\n  }\n]\n";
    sess.register_bytes_as("p", pretty.to_vec(), ahiru_core::FormatKind::Json).unwrap();
    let r = copy(&mut sess, "COPY (SELECT * FROM p) TO 'out.jsonl'");
    assert_eq!(
        String::from_utf8(r.data).unwrap(),
        "{\"id\":1,\"o\":{\"k\":[1,2],\"s\":\"a b\"}}\n"
    );
}

#[cfg(all(feature = "export", feature = "csv", feature = "jsonl"))]
#[test]
fn copy_renames_duplicate_names_for_csv_but_rejects_them_for_jsonl() {
    let mut sess = Session::new();
    let r = copy(&mut sess, "COPY (SELECT 1 AS a, 2 AS a FROM range(1)) TO 'd.csv'");
    assert_eq!(String::from_utf8(r.data).unwrap(), "a,a_1\n1,2\n");
    // DuckDB: "Duplicate struct entry name".
    let r = sess.prepare("COPY (SELECT 1 AS a, 2 AS a FROM range(1)) TO 'd.jsonl'", &[]);
    assert!(r.is_err());
}

#[cfg(all(feature = "export", feature = "csv", feature = "jsonl"))]
#[test]
fn copy_compression_suffix_and_ndjson_format() {
    let mut sess = Session::new();
    // The format comes from the name before `.gz`; the host gzips.
    let r = copy(&mut sess, "COPY (SELECT 1 AS a FROM range(1)) TO 'o.csv.gz'");
    assert!(r.gzip);
    assert_eq!(r.data, b"a\n1\n");
    let r = copy(&mut sess, "COPY (SELECT 1 AS a FROM range(1)) TO 'o.jsonl.gz'");
    assert!(r.gzip);
    assert_eq!(r.data, b"{\"a\":1}\n");
    let r = copy(&mut sess, "COPY (SELECT 1 AS a FROM range(1)) TO 'o.gz'");
    assert!(r.gzip);
    assert_eq!(r.data, b"a\n1\n");
    let r = copy(&mut sess, "COPY (SELECT 1 AS a FROM range(1)) TO 'o.csv'");
    assert!(!r.gzip);
    // `.GZ` is not a compression suffix for DuckDB either.
    assert!(!copy(&mut sess, "COPY (SELECT 1 AS a FROM range(1)) TO 'o.csv.GZ'").gzip);

    // No zstd encoder exists, so this fails instead of writing plain text.
    let r = sess.prepare("COPY (SELECT 1 AS a FROM range(1)) TO 'o.csv.zst'", &[]);
    assert_eq!(code_of(r), Some(Code::UnsupportedCodec));

    let r = copy(&mut sess, "COPY (SELECT 1 AS a FROM range(1)) TO 'o.x' (FORMAT ndjson)");
    assert_eq!(r.data, b"{\"a\":1}\n");
}

#[cfg(feature = "export-parquet")]
#[test]
fn copy_parquet_under_a_compression_suffix_stays_plain_parquet() {
    let mut sess = Session::new();
    for path in ["o.parquet.gz", "o.parquet.zst"] {
        let r = copy(&mut sess, &format!("COPY (SELECT 1 AS a FROM range(1)) TO '{path}'"));
        assert!(!r.gzip, "{path}");
        assert_eq!(&r.data[..4], b"PAR1", "{path}");
    }
}
