//! Extra edge-case coverage for scalar and aggregate functions
//! (`crates/ahiru-core/src/expr/funcs.rs`, `crates/ahiru-core/src/expr/regex.rs`,
//! and the aggregate paths they feed into via `exec::agg`).
//!
//! This file complements the existing unit tests inside `expr::funcs` /
//! `expr::regex` (which are extensive) by exercising things only visible at
//! the SQL/`Session` layer: `GROUP BY`/`FILTER (WHERE ..)` aggregate
//! behavior, DECIMAL scale propagation through real expressions, and a
//! handful of NULL/Unicode/boundary cases for scalar functions run over
//! actual table data rather than hand-built vectors.
//!
//! Expected values are cross-checked against the `duckdb` CLI
//! (`/opt/homebrew/bin/duckdb`) wherever DuckDB's behavior is the intended
//! reference; deviations that are documented, intentional non-goals (see
//! `crates/ahiru-core/src/expr/funcs.rs` module doc and `docs/DESIGN.md`
//! §15) are called out inline instead of asserted against DuckDB's output.

use ahiru_core::format::FormatKind;
use ahiru_core::session::{Prepared, QueryStep, Session};
use ahiru_core::vector::Value;

fn data(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn session_with_basic() -> Session {
    let mut s = Session::new();
    s.register_bytes("t", data("basic.parquet")).unwrap();
    s
}

/// Registers a small CSV-backed table so we get exact, hand-picked values
/// (rather than depending on `basic.parquet`'s generated content) for the
/// aggregate edge cases below. Follows the same CSV-`dual` trick as
/// `tests/generate_series.rs`.
fn session_with_csv(name: &str, csv: &str) -> Session {
    let mut s = Session::new();
    s.register_bytes_as(name, csv.as_bytes().to_vec(), FormatKind::Csv).unwrap();
    s
}

fn run(session: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    let mut q = match session.prepare(sql, &[]).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("{sql}: unexpected NeedIo"),
    };
    let mut rows = Vec::new();
    loop {
        match session.step(&mut q).unwrap() {
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

fn one(session: &mut Session, expr: &str) -> Value {
    let rows = run(session, &format!("SELECT {expr} AS x FROM t LIMIT 1"));
    rows[0][0].clone()
}

fn s(v: &str) -> Value {
    Value::Bytes(v.as_bytes().to_vec())
}

// =============================================================================
// Aggregate functions: statistics (stddev / variance / median / mode /
// approx_count_distinct)
// =============================================================================

#[test]
fn stddev_and_variance_are_sample_statistics() {
    let mut sess = session_with_csv("t", "x\n1\n2\n3\n4\n");
    // duckdb: stddev(x)=1.2909944487358056, variance(x)=1.6666666666666667
    // for [1,2,3,4] (sample, i.e. divides by n-1, matching `stddev_samp`).
    let rows = run(&mut sess, "SELECT stddev(x), variance(x) FROM t");
    let Value::F64(sd) = rows[0][0] else { panic!("not f64: {:?}", rows[0][0]) };
    let Value::F64(var) = rows[0][1] else { panic!("not f64: {:?}", rows[0][1]) };
    assert!((sd - 1.2909944487358056).abs() < 1e-9, "stddev={sd}");
    assert!((var - 1.6666666666666667).abs() < 1e-9, "variance={var}");
}

#[test]
fn stddev_and_variance_need_at_least_two_rows() {
    // duckdb: stddev(x) over a single row is NULL (sample stat is undefined
    // for n=1), and so is variance.
    let mut sess = session_with_csv("t", "x\n5\n");
    let rows = run(&mut sess, "SELECT stddev(x), variance(x) FROM t");
    assert_eq!(rows[0][0], Value::Null);
    assert_eq!(rows[0][1], Value::Null);
}

#[test]
fn stddev_of_identical_values_is_exactly_zero_not_nan() {
    // Regression guard for the Welford-M2-can-go-slightly-negative case
    // (`exec::agg::push_result` clamps M2 to >= 0 before sqrt).
    // duckdb: stddev(x)=0.0, variance(x)=0.0 for [5,5,5].
    let mut sess = session_with_csv("t", "x\n5\n5\n5\n");
    let rows = run(&mut sess, "SELECT stddev(x), variance(x) FROM t");
    assert_eq!(rows[0][0], Value::F64(0.0));
    assert_eq!(rows[0][1], Value::F64(0.0));
}

#[test]
fn median_handles_odd_and_even_counts() {
    // duckdb: median([1,2,3]) = 2.0 (odd count, no interpolation).
    let mut sess = session_with_csv("t", "x\n1\n2\n3\n");
    let rows = run(&mut sess, "SELECT median(x) FROM t");
    assert_eq!(rows[0][0], Value::F64(2.0));

    // duckdb: median([1,1,2,2]) = 1.5 (even count, linear interpolation
    // between the two middle values).
    let mut sess = session_with_csv("t", "x\n1\n1\n2\n2\n");
    let rows = run(&mut sess, "SELECT median(x) FROM t");
    assert_eq!(rows[0][0], Value::F64(1.5));
}

#[test]
fn mode_picks_the_most_frequent_value() {
    // duckdb: mode([1,2,2,3]) = 2.
    let mut sess = session_with_csv("t", "x\n1\n2\n2\n3\n");
    let rows = run(&mut sess, "SELECT mode(x) FROM t");
    assert_eq!(rows[0][0], Value::I64(2));
}

#[test]
fn approx_count_distinct_ignores_nulls() {
    // duckdb: approx_count_distinct([1,NULL,2,NULL]) = 2. v1's implementation
    // is an exact distinct count under the hood (module doc, `exec::agg`),
    // so this must be exact, not merely approximately close. The `id`
    // column keeps the NULL rows from being fully blank CSV lines, which
    // the reader would otherwise skip as "no row" entirely rather than "a
    // row with NULL x" (`format::csv::Scanner::skip_blank_line`) -- which
    // would make this test pass trivially without ever exercising NULL
    // handling.
    let mut sess = session_with_csv("t", "id,x\n1,1\n2,\n3,2\n4,\n");
    let rows = run(&mut sess, "SELECT count(*), approx_count_distinct(x) FROM t");
    assert_eq!(rows[0][0], Value::I64(4), "sanity check: all 4 rows, including NULLs, are read");
    assert_eq!(rows[0][1], Value::I64(2));
}

// =============================================================================
// Aggregate functions: string_agg / array_agg (NULL handling)
// =============================================================================

#[test]
fn string_agg_default_separator_is_comma() {
    let mut sess = session_with_csv("t", "x\na\nb\nc\n");
    // duckdb: string_agg(x) with no separator defaults to ','.
    let rows = run(&mut sess, "SELECT string_agg(x) FROM t");
    assert_eq!(rows[0][0], s("a,b,c"));
}

#[test]
fn string_agg_skips_null_rows_entirely() {
    // duckdb: string_agg(['a', NULL, 'b'], '-') = 'a-b' (NULL rows are
    // dropped, not turned into empty pieces). An `id` column keeps the
    // middle row from being a fully blank CSV line, which the reader would
    // otherwise skip as "no row" rather than "a row with NULL x"
    // (`format::csv::Scanner::skip_blank_line`).
    let mut sess = session_with_csv("t", "id,x\n1,a\n2,\n3,b\n");
    let rows = run(&mut sess, "SELECT string_agg(x, '-') FROM t");
    assert_eq!(rows[0][0], s("a-b"));
}

#[test]
fn string_agg_over_all_null_input_is_null() {
    // duckdb: string_agg(x, '-') over a single NULL row is NULL, not ''.
    let mut sess = session_with_csv("t", "id,x\n1,\n");
    let rows = run(&mut sess, "SELECT string_agg(x, '-') FROM t");
    assert_eq!(rows[0][0], Value::Null);
}

#[test]
fn array_agg_keeps_nulls_as_elements() {
    // `array_agg`/`list` has no LIST physical type in this engine, so the
    // result is a JSON-ish VARCHAR text (module doc in `exec::agg`).
    // NULL rows must still show up as a `null` element rather than being
    // dropped (matches duckdb: array_agg([1,NULL,3]) = [1, NULL, 3]).
    //
    // A second `id` column keeps the middle row from being an entirely
    // blank line: the CSV reader skips fully blank lines as "no row"
    // (`format::csv::Scanner::skip_blank_line`), not as a row with a NULL
    // in its only column, so a single-column NULL row can't be expressed
    // this way.
    let mut sess = session_with_csv("t", "id,x\n1,1\n2,\n3,3\n");
    let rows = run(&mut sess, "SELECT array_agg(x) FROM t");
    assert_eq!(rows[0][0], s("[1, null, 3]"));
}

#[test]
fn array_agg_of_a_single_null_row_is_a_one_element_array_of_null() {
    // duckdb: array_agg over a single NULL row is [NULL], not NULL/empty.
    let mut sess = session_with_csv("t", "id,x\n1,\n");
    let rows = run(&mut sess, "SELECT array_agg(x) FROM t");
    assert_eq!(rows[0][0], s("[null]"));
}

#[test]
fn array_agg_of_strings_escapes_json_specials() {
    let mut sess = session_with_csv("t", "x\nplain\n\"has \"\"quote\"\"\"\n");
    let rows = run(&mut sess, "SELECT array_agg(x) FROM t");
    assert_eq!(rows[0][0], s(r#"["plain", "has \"quote\""]"#));
}

// =============================================================================
// Aggregates over GROUP BY / FILTER / empty and all-NULL cases (basic.parquet)
// =============================================================================

#[test]
fn count_star_is_zero_but_other_aggregates_are_null_over_no_rows() {
    // No GROUP BY + a predicate that matches nothing: duckdb returns exactly
    // one row where COUNT(*)=0 and everything else is NULL (§9/HashAggregate
    // "ungrouped aggregate over empty input still emits one row").
    let mut sess = session_with_basic();
    let rows = run(
        &mut sess,
        "SELECT count(*), count(big), sum(big), avg(big), min(big), max(big), \
         stddev(big) FROM t WHERE id > 100000",
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], Value::I64(0), "COUNT(*) over empty input is 0");
    // COUNT(x) (unlike COUNT(*)) also returns 0, not NULL, over empty input
    // -- it's the other aggregates (SUM/AVG/MIN/MAX/STDDEV) that become
    // NULL when they have no non-NULL input to fold (duckdb reference:
    // `count(x)` over zero matching rows is 0).
    assert_eq!(rows[0][1], Value::I64(0), "COUNT(big) over empty input is 0, not NULL");
    for (i, v) in rows[0].iter().enumerate().skip(2) {
        assert_eq!(*v, Value::Null, "aggregate #{i} over empty input should be NULL");
    }
}

#[test]
fn aggregates_over_an_all_null_column_are_null_but_count_star_is_not() {
    // duckdb over `basic.parquet WHERE id in (0,5,10,15)` (all of which have
    // `big IS NULL`): sum/avg/min/max/stddev are NULL, count(big)=0,
    // count(*)=4.
    let mut sess = session_with_basic();
    let rows = run(
        &mut sess,
        "SELECT count(*), count(big), sum(big), avg(big), min(big), max(big), \
         stddev(big) FROM t WHERE id IN (0, 5, 10, 15)",
    );
    assert_eq!(rows[0][0], Value::I64(4));
    assert_eq!(rows[0][1], Value::I64(0));
    for v in &rows[0][2..] {
        assert_eq!(*v, Value::Null);
    }
}

#[test]
fn filter_where_scopes_the_aggregate_independently_per_group() {
    // duckdb reference (see module-level derivation): grouping ('a',1),
    // ('a',10),('b',NULL) by g, `sum(x) FILTER (WHERE x IS NOT NULL AND x >
    // 5)` gives a=10, b=NULL, while `count(*)` (no FILTER) is unaffected
    // (2 and 1 respectively).
    let mut sess = session_with_csv("t", "g,x\na,1\na,10\nb,\n");
    let rows = run(
        &mut sess,
        "SELECT g, count(*), sum(x) FILTER (WHERE x IS NOT NULL AND x > 5) \
         FROM t GROUP BY g ORDER BY g",
    );
    assert_eq!(rows.len(), 2);
    // SUM over integers accumulates in i128 (docs/DESIGN.md §15: "SUM alone
    // accumulates in i128"), so the result is `Value::I128`, not `I64`.
    assert_eq!(rows[0], vec![s("a"), Value::I64(2), Value::I128(10)]);
    assert_eq!(rows[1], vec![s("b"), Value::I64(1), Value::Null]);
}

#[test]
fn count_distinct_and_having_over_real_table_data() {
    // duckdb: `basic.parquet` has 7 distinct `name` values (name_0..name_6).
    let mut sess = session_with_basic();
    let rows = run(&mut sess, "SELECT count(DISTINCT name) FROM t");
    assert_eq!(rows[0][0], Value::I64(7));

    // HAVING filters groups post-aggregation; `name_6` has 142 rows, all
    // others have 143 (duckdb reference computed earlier).
    let mut rows =
        run(&mut sess, "SELECT name, count(*) c FROM t GROUP BY name HAVING count(*) < 143");
    rows.sort_by_key(|r| format!("{:?}", r[0]));
    assert_eq!(rows, vec![vec![s("name_6"), Value::I64(142)]]);
}

// =============================================================================
// DECIMAL: scale propagation and rounding conventions (docs/DESIGN.md §15)
// =============================================================================

#[test]
fn decimal_scale_down_rounds_away_from_zero() {
    let mut sess = session_with_basic();
    // duckdb: CAST(1.235 AS DECIMAL(10,2)) = 1.24, CAST(-1.235 AS
    // DECIMAL(10,2)) = -1.24 (away from zero on both sides, not banker's
    // rounding -- documented in docs/DESIGN.md §15 as distinct from the
    // float->int cast rule).
    assert_eq!(one(&mut sess, "CAST(1.235 AS DECIMAL(10,2))"), Value::I64(124));
    assert_eq!(one(&mut sess, "CAST(-1.235 AS DECIMAL(10,2))"), Value::I64(-124));
    // Note: `1.005` is deliberately not tested here. This engine parses
    // float literals as f64 (`sql::parser`), and 1.005 has no exact f64
    // representation -- it's actually 1.00499999999999989... (verified:
    // `python3 -c "from decimal import Decimal; print(Decimal(1.005))"`),
    // which is strictly less than the halfway point at scale 2. Rounding
    // that value away from zero correctly yields 1.00, not 1.01 (DuckDB
    // gets 1.01 here only because it parses decimal-looking literals as
    // exact DECIMAL, not through f64, which this engine doesn't do).
}

#[test]
fn float_to_int_cast_rounds_to_nearest_even() {
    // duckdb: CAST(x::DOUBLE AS INTEGER) uses round-half-to-even:
    // 1.5->2, 2.5->2, 3.5->4, 4.5->4, -1.5->-2, -2.5->-2.
    let mut sess = session_with_basic();
    for (x, want) in [("1.5", 2), ("2.5", 2), ("3.5", 4), ("4.5", 4), ("-1.5", -2), ("-2.5", -2)] {
        assert_eq!(
            one(&mut sess, &format!("CAST(CAST({x} AS DOUBLE) AS INTEGER)")),
            Value::I32(want),
            "CAST({x} AS INTEGER)"
        );
    }
}

#[test]
fn decimal_plus_decimal_widens_precision_and_matches_scale() {
    // duckdb: CAST(1.23 AS DECIMAL(10,2)) + CAST(1.2345 AS DECIMAL(10,4)) =
    // 2.4645, typed DECIMAL(13,4) (precision grows by the +1 carry digit
    // rule in `Ty::unify`, scale takes the wider operand's scale).
    let mut sess = session_with_basic();
    let mut q = sess
        .prepare(
            "SELECT CAST(1.23 AS DECIMAL(10,2)) + CAST(1.2345 AS DECIMAL(10,4)) AS x \
             FROM t LIMIT 1",
            &[],
        )
        .unwrap();
    let schema = match &q {
        Prepared::Ready(p) => p.schema.clone(),
        Prepared::NeedIo(_) => panic!("unexpected NeedIo"),
    };
    assert_eq!(schema[0].ty, ahiru_core::vector::Ty::Decimal { precision: 13, scale: 4 });
    let q = match &mut q {
        Prepared::Ready(p) => p,
        _ => unreachable!(),
    };
    let mut got = None;
    loop {
        match sess.step(q).unwrap() {
            QueryStep::Batch(mut b) => {
                b.materialize();
                got = Some(b.cols[0].value_at(0));
            }
            QueryStep::Done => break,
            _ => panic!("unexpected"),
        }
    }
    // Raw storage is the integer 24645 at scale 4 (== 2.4645).
    assert_eq!(got, Some(Value::I64(24645)));
}

#[test]
fn integer_division_and_mod_by_zero_are_null_not_errors() {
    // docs/DESIGN.md §15: "Integer division-by-zero and MIN / -1 return
    // NULL, not an error."
    let mut sess = session_with_basic();
    assert_eq!(one(&mut sess, "5 / 0"), Value::Null);
    assert_eq!(one(&mut sess, "5 % 0"), Value::Null);
    assert_eq!(one(&mut sess, "mod(5, 0)"), Value::Null);
    // Float division stays IEEE (never NULL, produces Infinity/NaN).
    let v = one(&mut sess, "5.0 / 0.0");
    assert_eq!(v, Value::F64(f64::INFINITY));
}

// =============================================================================
// Scalar functions over table data: Unicode, NULL propagation, multi-byte
// delimiters
// =============================================================================

#[test]
fn unicode_string_functions_match_duckdb_over_literals() {
    let mut sess = session_with_basic();
    // duckdb: upper/lower only touch ASCII case; 'é' passes through
    // unchanged both ways for THIS engine (documented ASCII-only limitation
    // in expr::funcs module doc) -- unlike duckdb, which has full Unicode
    // case folding ('café' -> 'CAFÉ'). Assert our documented, intentional
    // behavior explicitly so a regression toward "silently wrong" is caught.
    assert_eq!(one(&mut sess, "upper('café')"), s("CAFé"));
    assert_eq!(one(&mut sess, "lower('CAFÉ')"), s("cafÉ"));
    // length/substr are codepoint-based and do handle 'é' as one codepoint
    // (duckdb: length('café')=4, substr('café',4,1)='é').
    assert_eq!(one(&mut sess, "length('café')"), Value::I64(4));
    assert_eq!(one(&mut sess, "substr('café', 4, 1)"), s("é"));
    // duckdb: instr('あいうえお','う') = 3 (codepoint position).
    assert_eq!(one(&mut sess, "instr('あいうえお', 'う')"), Value::I64(3));
}

#[test]
fn split_part_supports_multibyte_delimiters_and_negative_index() {
    let mut sess = session_with_basic();
    // duckdb: split_part('a::b::c', '::', 2) = 'b', ..., -1) = 'c'.
    assert_eq!(one(&mut sess, "split_part('a::b::c', '::', 2)"), s("b"));
    assert_eq!(one(&mut sess, "split_part('a::b::c', '::', -1)"), s("c"));
    // duckdb: split_part('a,b,,d', ',', 3) = '' (empty piece between two
    // consecutive delimiters is preserved, not skipped).
    assert_eq!(one(&mut sess, "split_part('a,b,,d', ',', 3)"), s(""));
}

#[test]
fn printf_truncates_floats_toward_zero_for_d_unlike_duckdbs_strict_typing() {
    // Documented, intentional deviation (`printf_scan` doc in funcs.rs):
    // DuckDB errors on `%d` with a FLOAT argument; this engine truncates
    // toward zero instead so the query doesn't fail mid-expression.
    let mut sess = session_with_basic();
    assert_eq!(one(&mut sess, "printf('%d', 3.9)"), s("3"));
    assert_eq!(one(&mut sess, "printf('%d', -3.9)"), s("-3"));
}

#[test]
fn json_extract_navigates_arrays_of_objects_and_nested_arrays() {
    let mut sess = session_with_basic();
    // duckdb: json_extract('{"a":[{"b":1},{"b":2}]}', '$.a[1].b') = 2.
    assert_eq!(one(&mut sess, r#"json_extract('{"a":[{"b":1},{"b":2}]}', '$.a[1].b')"#), s("2"));
    // duckdb: json_extract('[1,[2,3],4]', '$[1][0]') = 2.
    assert_eq!(one(&mut sess, r#"json_extract('[1,[2,3],4]', '$[1][0]')"#), s("2"));
}

#[test]
fn json_array_length_of_json_null_literal_is_zero() {
    // duckdb: json_array_length('null') = 0 (JSON `null` is not an array,
    // same "non-array => 0" rule as any other scalar/object).
    let mut sess = session_with_basic();
    assert_eq!(one(&mut sess, "json_array_length('null')"), Value::I64(0));
}

#[test]
fn nullif_and_greatest_least_promote_int_and_double_to_double() {
    let mut sess = session_with_basic();
    // duckdb (with the float literal cast to DOUBLE so both engines agree on
    // the input type -- see module doc: bare float literals are DOUBLE in
    // this engine, not DECIMAL like DuckDB's default literal typing):
    // greatest(1, 2.5::DOUBLE) = 2.5, coalesce(NULL, 1, 2.5::DOUBLE) = 1.0.
    assert_eq!(one(&mut sess, "greatest(1, CAST(2.5 AS DOUBLE))"), Value::F64(2.5));
    assert_eq!(one(&mut sess, "coalesce(NULL, 1, CAST(2.5 AS DOUBLE))"), Value::F64(1.0));
    assert_eq!(one(&mut sess, "nullif('abc', 'abc')"), Value::Null);
    assert_eq!(one(&mut sess, "nullif('abc', 'xyz')"), s("abc"));
}

// =============================================================================
// regexp_* through the SQL layer (group capture, backreferences, global
// replace) -- expr::regex already has extensive unit tests; these confirm
// the same behavior survives the resolve/call/Session round trip.
// =============================================================================

#[test]
fn regexp_extract_with_capture_group_and_replace_with_backreferences() {
    let mut sess = session_with_basic();
    // duckdb: regexp_extract('2024-01-05', '(\d+)-(\d+)-(\d+)', 2) = '01'.
    assert_eq!(one(&mut sess, r#"regexp_extract('2024-01-05', '(\d+)-(\d+)-(\d+)', 2)"#), s("01"));
    // duckdb: regexp_replace('hello world', '(\w+) (\w+)', '\2 \1') =
    // 'world hello'.
    assert_eq!(
        one(&mut sess, r#"regexp_replace('hello world', '(\w+) (\w+)', '\2 \1')"#),
        s("world hello")
    );
    // duckdb: regexp_replace('aaa', 'a', 'b', 'g') = 'bbb' (global flag).
    assert_eq!(one(&mut sess, "regexp_replace('aaa', 'a', 'b', 'g')"), s("bbb"));
}

#[test]
fn regexp_extract_no_match_is_empty_string_not_null() {
    // duckdb: regexp_extract('no match here', '(\d+)', 1) = '' (empty
    // string). NULL is reserved for NULL str/pattern/group arguments
    // themselves (documented in expr::regex::eval_extract).
    let mut sess = session_with_basic();
    let v = one(&mut sess, r#"regexp_extract('no match here', '(\d+)', 1)"#);
    assert_eq!(v, s(""));
    assert_ne!(v, Value::Null);
}

#[test]
fn regexp_matches_and_regexp_extract_over_table_column_not_just_literals() {
    // Exercise the vectorized path (non-constant subject column) rather than
    // only the constant-folded literal path used above.
    let mut sess = session_with_basic();
    let rows = run(
        &mut sess,
        "SELECT id, regexp_matches(name, 'name_[0-3]$'), regexp_extract(name, '_(\\d+)', 1) \
         FROM t WHERE id < 5 ORDER BY id",
    );
    // name_0..name_4: only name_0..name_3 match `name_[0-3]$`.
    assert_eq!(rows.len(), 5);
    for (i, row) in rows.iter().enumerate() {
        let expect_match = i <= 3;
        assert_eq!(row[1], Value::Bool(expect_match), "row {i}");
        assert_eq!(row[2], s(&i.to_string()), "row {i}");
    }
}
