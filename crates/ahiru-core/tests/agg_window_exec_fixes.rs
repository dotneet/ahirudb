//! Regression tests for aggregate / window / sort execution bugs where the
//! engine returned a silently wrong answer (`exec::agg`, `exec::window`,
//! `exec::sort`, and the binder spots that feed them).
//!
//! Expected values are cross-checked against the `duckdb` CLI (v1.4); the one
//! deliberate deviation (NaN instead of DuckDB's "out of range" error for
//! `stddev` over non-finite input) is called out inline.

use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn data(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// `w`: ids 1..=10 with a DOUBLE `x` (every row non-NULL, so window results
/// are easy to predict).
fn session() -> Session {
    let mut csv = String::from("id,x\n");
    for i in 1..=10 {
        csv.push_str(&format!("{i},{}.5\n", i * 2));
    }
    let mut s = Session::new();
    s.register_bytes_as("w", csv.into_bytes(), FormatKind::Csv).unwrap();
    s
}

fn run(session: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    let mut q = match session.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
    };
    let mut rows = Vec::new();
    loop {
        match session.step(&mut q).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
            QueryStep::Batch(mut b) => {
                b.materialize();
                for r in 0..b.num_rows() {
                    rows.push(b.cols.iter().map(|c| c.value_at(r)).collect());
                }
            }
            QueryStep::Done => break,
            QueryStep::NeedIo(_) | QueryStep::NeedCodec(_) => {
                panic!("{sql}: unexpected NeedIo/NeedCodec")
            }
        }
    }
    rows
}

fn s(v: &str) -> Value {
    Value::Bytes(v.as_bytes().to_vec())
}

fn f(v: &Value) -> f64 {
    match v {
        Value::F64(x) => *x,
        other => panic!("expected DOUBLE, got {other:?}"),
    }
}

/// One column of every row, as text (`Bytes`) values.
fn col(rows: &[Vec<Value>], c: usize) -> Vec<Value> {
    rows.iter().map(|r| r[c].clone()).collect()
}

// --- stddev / variance over non-finite input ---------------------------------

/// `f64::max` drops a NaN operand, so the variance clamp used to turn the NaN a
/// non-finite input produces into a confident `0.0`. DuckDB raises
/// "STDDEV_SAMP is out of range"; this engine keeps floats IEEE (DESIGN.md §15)
/// and answers NaN.
#[test]
fn stddev_family_over_nan_or_inf_is_nan_not_zero() {
    let mut sess = session();
    for bad in ["'nan'", "'inf'", "'-inf'"] {
        let sql = format!(
            "SELECT stddev(v), variance(v), stddev_pop(v), var_pop(v) FROM \
             (SELECT CASE WHEN id = 3 THEN {bad}::DOUBLE ELSE x END AS v FROM w)"
        );
        let rows = run(&mut sess, &sql);
        for v in &rows[0] {
            assert!(f(v).is_nan(), "{sql}: {v:?}");
        }
    }
    // The clamp itself still holds for identical values.
    let rows = run(&mut sess, "SELECT stddev(1.1::DOUBLE), var_pop(1.1::DOUBLE) FROM w");
    assert_eq!(rows[0], vec![Value::F64(0.0), Value::F64(0.0)]);
}

// --- window count/offset arguments --------------------------------------------

/// The count/offset arguments are cast to BIGINT (as in DuckDB). They used to be
/// read raw: `2::DECIMAL(4,1)` as 20 buckets, a DOUBLE as NULL or a type error.
#[test]
fn window_count_arguments_are_cast_to_bigint() {
    let mut sess = session();
    let rows = run(
        &mut sess,
        "SELECT id, \
           ntile(2::DECIMAL(4,1)) OVER (ORDER BY id), \
           ntile(2.0) OVER (ORDER BY id), \
           ntile(x / x + 1) OVER (ORDER BY id), \
           lag(id, 1::DECIMAL(4,1)) OVER (ORDER BY id), \
           lag(id, 1.0) OVER (ORDER BY id), \
           lead(id, 2.0, 0) OVER (ORDER BY id), \
           nth_value(id, 3::DECIMAL(3,1)) OVER (ORDER BY id) \
         FROM w ORDER BY id",
    );
    // The CSV sniffer types `id` as BIGINT.
    for r in &rows {
        let Value::I64(id) = r[0] else { panic!("{:?}", r[0]) };
        let bucket = Value::I64(if id <= 5 { 1 } else { 2 });
        assert_eq!(r[1], bucket, "ntile(decimal) at {id}");
        assert_eq!(r[2], bucket, "ntile(double literal) at {id}");
        assert_eq!(r[3], bucket, "ntile(double column) at {id}");
        let prev = if id == 1 { Value::Null } else { Value::I64(id - 1) };
        assert_eq!(r[4], prev, "lag(decimal offset) at {id}");
        assert_eq!(r[5], prev, "lag(double offset) at {id}");
        let next2 = Value::I64(if id <= 8 { id + 2 } else { 0 });
        assert_eq!(r[6], next2, "lead(double offset, default) at {id}");
        let third = if id >= 3 { Value::I64(3) } else { Value::Null };
        assert_eq!(r[7], third, "nth_value(decimal) at {id}");
    }
}

// --- LIST ordering --------------------------------------------------------------

/// Lists (JSON array text) used to order by bytes: `max([id])` was `[9]`.
#[test]
fn list_min_max_and_order_by_compare_element_wise() {
    let mut sess = session();
    let rows =
        run(&mut sess, "SELECT max([id]), min([id]), arg_max(id, [id]), arg_min(id, [id]) FROM w");
    assert_eq!(rows[0][0], s("[10]"));
    assert_eq!(rows[0][1], s("[1]"));
    assert_eq!(rows[0][2], Value::I64(10));
    assert_eq!(rows[0][3], Value::I64(1));

    let rows = run(&mut sess, "SELECT [id] FROM w ORDER BY [id] DESC LIMIT 3");
    assert_eq!(col(&rows, 0), vec![s("[10]"), s("[9]"), s("[8]")]);

    // A NULL element sorts after every value; a prefix sorts first.
    let rows = run(
        &mut sess,
        "SELECT [CASE WHEN id % 3 = 0 THEN NULL ELSE id % 4 END, id] AS v FROM w ORDER BY v DESC",
    );
    let got: Vec<Value> = col(&rows, 0);
    let want: Vec<Value> = [
        "[null,9]", "[null,6]", "[null,3]", "[3,7]", "[2,10]", "[2,2]", "[1,5]", "[1,1]", "[0,8]",
        "[0,4]",
    ]
    .iter()
    .map(|v| s(v))
    .collect();
    assert_eq!(got, want);

    // Nested lists compare recursively.
    let rows = run(&mut sess, "SELECT [[id % 3], [id]] AS v FROM w ORDER BY v LIMIT 4");
    assert_eq!(col(&rows, 0), vec![s("[[0],[3]]"), s("[[0],[6]]"), s("[[0],[9]]"), s("[[1],[1]]")]);

    // Window ordering and a window MAX go through the same comparator.
    let rows = run(
        &mut sess,
        "SELECT id, rank() OVER (ORDER BY [id] DESC), max([id]) OVER (ORDER BY id) \
         FROM w ORDER BY id DESC LIMIT 2",
    );
    assert_eq!(rows[0][1], Value::I64(1));
    assert_eq!(rows[0][2], s("[10]"));
    assert_eq!(rows[1][1], Value::I64(2));
    assert_eq!(rows[1][2], s("[9]"));
}

/// The same fix over a real Parquet LIST column. DuckDB:
/// `SELECT min(xs), max(xs) FROM 'list_varied.parquet' WHERE id < 10` -> `[]`, `[9, 10, 11, 12]`.
#[test]
fn parquet_list_column_min_max_and_order() {
    let mut sess = Session::new();
    sess.register_bytes_as("t", data("list_varied.parquet"), FormatKind::Parquet).unwrap();
    let rows = run(&mut sess, "SELECT min(xs), max(xs) FROM t WHERE id < 10");
    assert_eq!(rows[0], vec![s("[]"), s("[9,10,11,12]")]);
    let rows = run(&mut sess, "SELECT xs FROM t WHERE id < 10 AND xs IS NOT NULL ORDER BY xs");
    let want: Vec<Value> =
        ["[]", "[]", "[2]", "[3,null,6]", "[4,5,6,7]", "[7]", "[8,null,16]", "[9,10,11,12]"]
            .iter()
            .map(|v| s(v))
            .collect();
    assert_eq!(col(&rows, 0), want);
}

// --- AVG(DECIMAL) ----------------------------------------------------------------

/// Dividing by 10^scale and then by n rounds twice: three `0.1`s averaged to
/// 0.09999999999999999. DuckDB divides once, by `n * 10^scale`.
#[test]
fn avg_decimal_divides_once() {
    let mut sess = session();
    let rows = run(
        &mut sess,
        "SELECT avg(0.1::DECIMAL(4,1)), avg(CASE WHEN id <= 3 THEN 0.1::DECIMAL(4,1) END) FROM w",
    );
    assert_eq!(rows[0], vec![Value::F64(0.1), Value::F64(0.1)]);
    let rows = run(
        &mut sess,
        "SELECT avg(CASE WHEN id <= 3 THEN 0.1::DECIMAL(4,1) END) OVER () FROM w LIMIT 1",
    );
    assert_eq!(rows[0][0], Value::F64(0.1));
}

// --- count_if ------------------------------------------------------------------------

/// `count_if` is a SUM of booleans in DuckDB: NULL over no non-NULL input,
/// 0 over only-false input. It used to answer 0 for both.
#[test]
fn count_if_is_null_without_non_null_input() {
    let mut sess = session();
    let rows = run(
        &mut sess,
        "SELECT count_if(x > 100), count_if(NULL::BOOLEAN), count_if(x > 5) FILTER (WHERE false), \
         count_if(x > 5) FROM w",
    );
    assert_eq!(rows[0], vec![Value::I64(0), Value::Null, Value::Null, Value::I64(8)]);
    let rows = run(&mut sess, "SELECT count_if(x > 1) FROM w WHERE id > 100");
    assert_eq!(rows[0][0], Value::Null);
    let rows = run(
        &mut sess,
        "SELECT count_if(CASE WHEN id > 2 THEN x > 1 END) OVER (ORDER BY id) FROM w ORDER BY id LIMIT 3",
    );
    assert_eq!(col(&rows, 0), vec![Value::Null, Value::Null, Value::I64(1)]);
}

// --- mode --------------------------------------------------------------------------------

/// Equal keys can render differently (`1 month` = `30 days` = `720 hours` as an
/// INTERVAL key). DuckDB reports the first one seen; this used to report the
/// last one seen.
#[test]
fn mode_reports_the_first_seen_representative() {
    let mut sess = session();
    let rows = run(
        &mut sess,
        "SELECT mode(v)::VARCHAR FROM (SELECT CASE WHEN id % 3 = 1 THEN INTERVAL 1 MONTH \
         WHEN id % 3 = 2 THEN INTERVAL 30 DAYS ELSE INTERVAL 720 HOURS END AS v FROM w)",
    );
    assert_eq!(rows[0][0], s("1 month"));
    // A later value overtaking an earlier one still reports its own first occurrence.
    let rows = run(
        &mut sess,
        "SELECT mode(v)::VARCHAR FROM (SELECT CASE id WHEN 1 THEN INTERVAL 2 DAYS \
         WHEN 2 THEN INTERVAL 1 DAY WHEN 3 THEN INTERVAL 48 HOURS ELSE INTERVAL 24 HOURS END \
         AS v FROM w)",
    );
    assert_eq!(rows[0][0], s("1 day"));
}

// --- quantile_cont ---------------------------------------------------------------------

/// `lo + (hi - lo) * w` is NaN when `lo` is `-inf` (`-inf + inf`); DuckDB
/// answers `-inf`. It also overflows for `-1e308`/`1e308`.
#[test]
fn quantile_cont_interpolates_across_infinities() {
    let mut sess = session();
    let vals = "SELECT CASE id WHEN 1 THEN '-inf'::DOUBLE WHEN 2 THEN -0.0 WHEN 3 THEN 0.0 \
                WHEN 4 THEN 1.0 WHEN 5 THEN 'inf'::DOUBLE ELSE 'nan'::DOUBLE END AS v FROM w \
                WHERE id <= 6";
    let rows = run(
        &mut sess,
        &format!(
            "SELECT quantile_cont(v, 0.1), quantile_cont(v, 0.3), quantile_cont(v, 0.9) FROM ({vals})"
        ),
    );
    assert_eq!(f(&rows[0][0]), f64::NEG_INFINITY);
    assert_eq!(f(&rows[0][1]), 0.0);
    assert!(f(&rows[0][2]).is_nan());

    let rows = run(
        &mut sess,
        "SELECT median(v), quantile_cont(v, 0.25) FROM (SELECT CASE WHEN id = 1 THEN '-inf'::DOUBLE \
         ELSE 1.0 END AS v FROM w WHERE id <= 2)",
    );
    assert_eq!(rows[0], vec![Value::F64(f64::NEG_INFINITY), Value::F64(f64::NEG_INFINITY)]);
    let rows = run(
        &mut sess,
        "SELECT median(v) FROM (SELECT CASE WHEN id = 1 THEN -1e308 ELSE 1e308 END AS v FROM w \
         WHERE id <= 2)",
    );
    assert_eq!(rows[0][0], Value::F64(0.0));
    // A finite case where the two interpolation forms round differently; DuckDB
    // (quantile_cont over -1e300, 0.1, 0.7, 1e300 at 0.3) gives -1.000000000000001e299.
    let rows = run(
        &mut sess,
        "SELECT quantile_cont(v, 0.3) FROM (SELECT CASE id WHEN 1 THEN -1e300 WHEN 2 THEN 0.1 \
         WHEN 3 THEN 0.7 ELSE 1e300 END AS v FROM w WHERE id <= 4)",
    );
    assert_eq!(rows[0][0], Value::F64(-1.000000000000001e299));
}

/// The fraction used to be accepted only as a DOUBLE or BIGINT literal, so the
/// INTEGER literals `0` and `1` failed with "unsupported SQL feature".
#[test]
fn quantile_fraction_accepts_any_constant_numeric() {
    let mut sess = session();
    let rows = run(
        &mut sess,
        "SELECT quantile_cont(x, 0), quantile_cont(x, 1), quantile_cont(x, -0.0), \
         quantile_cont(x, 0.5::DECIMAL(2,1)), percentile_cont(x, 1) FROM w",
    );
    assert_eq!(
        rows[0],
        vec![
            Value::F64(2.5),
            Value::F64(20.5),
            Value::F64(2.5),
            Value::F64(11.5),
            Value::F64(20.5)
        ]
    );
    // Still a constant in [0, 1].
    for bad in ["quantile_cont(x, id)", "quantile_cont(x, NULL)"] {
        assert!(sess.prepare(&format!("SELECT {bad} FROM w"), &[]).is_err(), "{bad}");
    }
    assert!(sess.prepare("SELECT quantile_cont(x, 2) FROM w", &[]).is_err());
}
