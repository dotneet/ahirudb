//! Integration tests for the array literal `[..]`, `printf`/`format`, and `GLOB`/`SIMILAR TO`.
//!
//! Expected values are decided by cross-checking against the actual output of
//! `duckdb -c "SELECT ..."`.
//! For supported scope and known limitations, see the comments on `sql::parser`
//! (`array_literal`/`similar_to`) and `expr::funcs`
//! (`printf_scan`/`format_scan`/`glob_match`/`regexp_full_match_build`).
//!
//! A `SELECT <expr>` with no `FROM` is unsupported in v1 (`plan::bind`), so we use
//! `tests/data/basic.parquet` with `LIMIT 1` to get exactly one row
//! (same convention as `json_functions.rs`).

use ahiru_core::error::{code_of, Code};
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

// --- Array literal ------------------------------------------------------------

#[test]
fn array_literal_is_sugar_for_list_value() {
    let mut sess = session_with_basic();
    // duckdb: [1,2,3] = list_value(1,2,3) -> true
    assert_eq!(one(&mut sess, "[1, 2, 3] = list_value(1, 2, 3)"), Value::Bool(true));
    // duckdb: [1,2,3][1] -> 1 (1-based; confirmed via list_extract).
    // `list_extract`'s result is designed to return Ty::Json (raw JSON text), so the
    // expected value is also the text "10" (see the module-top doc on `list_extract` and the
    // `list_extract_is_one_based_with_negative_from_end` unit test in `funcs.rs`).
    assert_eq!(one(&mut sess, "list_extract([10, 20, 30], 1)"), s("10"));
    // duckdb: [] is a valid expression (an empty array). json_array_length([]) = 0.
    assert_eq!(one(&mut sess, "json_array_length([])"), Value::I64(0));
    // Mixed types are also allowed (same as `list_value`/`json_array`).
    assert_eq!(one(&mut sess, "to_json([1, 'x', true])"), s(r#"[1,"x",true]"#));
}

#[test]
fn subscript_on_a_non_json_column_binds_fine_but_fails_at_execution() {
    // `expr[i]` is no longer scope-limited to array literals — it now
    // desugars to `list_extract(expr, i)` for any `expr` (see
    // `sql::parser::Parser::subscript`, added alongside this test's rename).
    // `id` is `Ty::Int`, and `list_extract`'s first parameter is `Ty::Json`,
    // so the call still needs an implicit Int -> Json coercion. That
    // coercion is *not* rejected at bind time — `plan::compile::Compiler::
    // coerce` unconditionally emits a `Cast` instruction without checking
    // in advance whether the specific (from, to) pair is even a legal cast
    // (this is pre-existing behavior of the generic function-argument
    // coercion path, not something this change introduced). The rejection
    // instead happens at the `Cast` kernel, which only allows VARCHAR <->
    // JSON (`expr::kernels::cast_impl`), i.e. at *execution* time.
    let mut sess = session_with_basic();
    assert_eq!(code_of(sess.prepare("SELECT id[1] FROM t", &[]).map(|_| ())), None);
    let mut q = match sess.prepare("SELECT id[1] AS x FROM t", &[]).unwrap() {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("unexpected NeedIo"),
    };
    let err = loop {
        match sess.step(&mut q) {
            Ok(QueryStep::Batch(_)) => continue,
            Ok(QueryStep::Done) => panic!("expected a runtime error, got a result"),
            Ok(QueryStep::NeedIo(_)) | Ok(QueryStep::NeedCodec(_)) => panic!("unexpected NeedIo"),
            Err(e) => break e,
        }
    };
    assert_eq!(err.code, Code::InvalidCast);
}

// --- printf / format ----------------------------------------------------------

#[test]
fn printf_matches_duckdb_on_common_specifiers() {
    let mut sess = session_with_basic();
    // duckdb: printf('%d-%s', 42, 'x') = '42-x'
    assert_eq!(one(&mut sess, "printf('%d-%s', 42, 'x')"), s("42-x"));
    // duckdb: printf('%%') = '%'
    assert_eq!(one(&mut sess, "printf('%%')"), s("%"));
    // duckdb: printf('%05d', 3) = '00003', printf('%05d', -3) = '-0003'
    assert_eq!(one(&mut sess, "printf('%05d', 3)"), s("00003"));
    assert_eq!(one(&mut sess, "printf('%05d', -3)"), s("-0003"));
    // duckdb: printf('%-5d|', 3) = '3    |'
    assert_eq!(one(&mut sess, "printf('%-5d|', 3)"), s("3    |"));
    // duckdb: printf('%.2f', 3.14159) = '3.14', printf('%f', 3.5) = '3.500000'
    assert_eq!(one(&mut sess, "printf('%.2f', 3.14159)"), s("3.14"));
    assert_eq!(one(&mut sess, "printf('%f', 3.5)"), s("3.500000"));
    // A NULL argument makes the whole result NULL (the default NULL propagation).
    assert_eq!(one(&mut sess, "printf('%s', NULL)"), Value::Null);
    // A table column can be used as a real argument (works with real data, not just constant folding).
    assert_eq!(one(&mut sess, "printf('id=%d name=%s', id, name)"), s("id=0 name=name_0"));
}

#[test]
fn printf_rejects_unsupported_specifiers_and_short_arg_lists() {
    let mut sess = session_with_basic();
    // The format string isn't necessarily constant (it can be a column too), so mixing in an
    // unsupported conversion character like `%x` cannot be caught at `prepare` (type checking
    // only) time; it only becomes an error at runtime (`step`).
    let mut q = match sess.prepare("SELECT printf('%x', 1) FROM t", &[]).unwrap() {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("unexpected NeedIo"),
    };
    let mut saw_error = false;
    loop {
        match sess.step(&mut q) {
            Ok(QueryStep::Batch(_)) => continue,
            Ok(QueryStep::Done) => break,
            Ok(QueryStep::NeedIo(_)) | Ok(QueryStep::NeedCodec(_)) => panic!("unexpected"),
            Err(e) => {
                assert_eq!(e.code, Code::UnsupportedFeature);
                saw_error = true;
                break;
            }
        }
    }
    assert!(
        saw_error,
        "printf('%x', ..) uses an unsupported conversion character, so it should error"
    );
}

#[test]
fn format_supports_python_style_placeholders() {
    let mut sess = session_with_basic();
    // duckdb: format('{}-{}', 42, 'x') = '42-x'
    assert_eq!(one(&mut sess, "format('{}-{}', 42, 'x')"), s("42-x"));
    // duckdb: format('{{literal}}') = '{literal}'
    assert_eq!(one(&mut sess, "format('{{literal}}')"), s("{literal}"));
    // duckdb: format('{1}-{0}', 'a', 'b') = 'b-a'
    assert_eq!(one(&mut sess, "format('{1}-{0}', 'a', 'b')"), s("b-a"));
}

// --- GLOB / SIMILAR TO ---------------------------------------------------------

#[test]
fn glob_matches_shell_patterns_over_table_rows() {
    let mut sess = session_with_basic();
    // duckdb: 'name_0' GLOB 'name_?' = true, 'name_0' GLOB 'name_1' = false
    assert_eq!(one(&mut sess, "name GLOB 'name_?'"), Value::Bool(true));
    assert_eq!(one(&mut sess, "'name_0' GLOB 'name_1'"), Value::Bool(false));
    // Filtering works correctly against real data too (used in a `WHERE` clause).
    // `basic.parquet` is just a table repeating 7 distinct values `name_0`..`name_6` over 1000
    // rows, so use `DISTINCT` to look at just the candidate set.
    let rows =
        run(&mut sess, "SELECT DISTINCT name FROM t WHERE name GLOB 'name_[01]' ORDER BY name");
    assert_eq!(rows, vec![vec![s("name_0")], vec![s("name_1")]]);
    // `NOT (x GLOB y)` can be written, but `x NOT GLOB y` is a syntax error just like DuckDB.
    assert_eq!(one(&mut sess, "NOT ('abc' GLOB 'x*')"), Value::Bool(true));
    assert_eq!(
        code_of(sess.prepare("SELECT 'abc' NOT GLOB 'x*' FROM t", &[]).map(|_| ())),
        Some(Code::UnexpectedToken)
    );
}

#[test]
fn similar_to_is_full_match_regexp() {
    let mut sess = session_with_basic();
    // duckdb: 'abc' similar to 'a.c' = true, 'Xabc' similar to 'a.c' = false
    // (a full match, not a partial one).
    assert_eq!(one(&mut sess, "'abc' SIMILAR TO 'a.c'"), Value::Bool(true));
    assert_eq!(one(&mut sess, "'Xabc' SIMILAR TO 'a.c'"), Value::Bool(false));
    assert_eq!(one(&mut sess, "'Xabc' NOT SIMILAR TO 'a.c'"), Value::Bool(true));
    // A filter against real data (`basic.parquet` just repeats 7 distinct values
    // `name_0`..`name_6`, so use `DISTINCT` to look at just the candidate set).
    let rows = run(
        &mut sess,
        "SELECT DISTINCT name FROM t WHERE name SIMILAR TO 'name_[0-1]' ORDER BY name",
    );
    assert_eq!(rows, vec![vec![s("name_0")], vec![s("name_1")]]);
    // The ESCAPE clause is rejected here too, since DuckDB itself rejects it as unimplemented.
    assert_eq!(
        code_of(sess.prepare(r"SELECT 'a' SIMILAR TO 'a' ESCAPE '\' FROM t", &[]).map(|_| ())),
        Some(Code::UnsupportedFeature)
    );
}

// --- Additional edge cases ---------------------------------------------------------

#[test]
fn printf_too_few_args_is_a_runtime_error() {
    let mut sess = session_with_basic();
    // duckdb: printf('%d-%d', 1) => Invalid Input Error ("Argument index ...
    // out of range"). Since the format string can also be a column, this engine likewise
    // detects it as a runtime (`step`) error (same reason as
    // `printf_rejects_unsupported_specifiers_and_short_arg_lists`).
    let mut q = match sess.prepare("SELECT printf('%d-%d', 1) FROM t", &[]).unwrap() {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("unexpected NeedIo"),
    };
    let mut saw_error = false;
    loop {
        match sess.step(&mut q) {
            Ok(QueryStep::Batch(_)) => continue,
            Ok(QueryStep::Done) => break,
            Ok(QueryStep::NeedIo(_)) | Ok(QueryStep::NeedCodec(_)) => panic!("unexpected"),
            Err(e) => {
                assert_eq!(e.code, Code::WrongArgCount);
                saw_error = true;
                break;
            }
        }
    }
    assert!(saw_error, "printf('%d-%d', 1) has too few arguments, so it should error");
}

#[test]
fn printf_extra_args_are_ignored_like_duckdb() {
    let mut sess = session_with_basic();
    // duckdb: printf('%d', 1, 2) = '1' (leftover arguments are ignored).
    assert_eq!(one(&mut sess, "printf('%d', 1, 2)"), s("1"));
}

#[test]
fn format_too_few_args_and_out_of_range_index_are_runtime_errors() {
    let mut sess = session_with_basic();
    // duckdb: format('{} {} {}', 1, 2) => an error for too few arguments. Since the format
    // string can also be a column, `prepare` (type checking) itself succeeds here, and it
    // is only detected at runtime (`step`).
    let mut q = match sess.prepare("SELECT format('{} {} {}', 1, 2) FROM t", &[]).unwrap() {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("unexpected NeedIo"),
    };
    let mut saw_error = false;
    loop {
        match sess.step(&mut q) {
            Ok(QueryStep::Batch(_)) => continue,
            Ok(QueryStep::Done) => break,
            Ok(QueryStep::NeedIo(_)) | Ok(QueryStep::NeedCodec(_)) => panic!("unexpected"),
            Err(e) => {
                assert_eq!(e.code, Code::WrongArgCount);
                saw_error = true;
                break;
            }
        }
    }
    assert!(saw_error);
    // duckdb: format('{2}', 1, 2) => index 2 is out of range, an error
    // (0-based, so only {0}/{1} are valid).
    let mut q2 = match sess.prepare("SELECT format('{2}', 1, 2) FROM t", &[]).unwrap() {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => panic!("unexpected NeedIo"),
    };
    let mut saw_error2 = false;
    loop {
        match sess.step(&mut q2) {
            Ok(QueryStep::Batch(_)) => continue,
            Ok(QueryStep::Done) => break,
            Ok(QueryStep::NeedIo(_)) | Ok(QueryStep::NeedCodec(_)) => panic!("unexpected"),
            Err(e) => {
                assert_eq!(e.code, Code::WrongArgCount);
                saw_error2 = true;
                break;
            }
        }
    }
    assert!(saw_error2);
}

#[test]
fn glob_supports_negated_character_classes_and_escaping() {
    let mut sess = session_with_basic();
    // duckdb: 'abc' GLOB '[!a]*' = false (starts with 'a', so it doesn't match the negated
    // class), 'xbc' GLOB '[!a]*' = true.
    assert_eq!(one(&mut sess, "'abc' GLOB '[!a]*'"), Value::Bool(false));
    assert_eq!(one(&mut sess, "'xbc' GLOB '[!a]*'"), Value::Bool(true));
    // duckdb: 'a*b' GLOB 'a\*b' = true (a backslash can escape `*` to be treated
    // literally).
    assert_eq!(one(&mut sess, r"'a*b' GLOB 'a\*b'"), Value::Bool(true));
    // Character-class range specs work too.
    assert_eq!(one(&mut sess, "'c' GLOB '[a-z]'"), Value::Bool(true));
    assert_eq!(one(&mut sess, "'C' GLOB '[a-z]'"), Value::Bool(false));
}

#[test]
fn similar_to_supports_alternation_and_quantifiers() {
    let mut sess = session_with_basic();
    // duckdb: 'abc' similar to '(a|x)bc' = true, 'aaab' similar to
    // 'a{2,3}b' = true, 'ab' similar to 'a+b?' = true.
    assert_eq!(one(&mut sess, "'abc' SIMILAR TO '(a|x)bc'"), Value::Bool(true));
    assert_eq!(one(&mut sess, "'aaab' SIMILAR TO 'a{2,3}b'"), Value::Bool(true));
    assert_eq!(one(&mut sess, "'ab' SIMILAR TO 'a+b?'"), Value::Bool(true));
}

#[test]
fn array_literal_allows_mixed_numeric_types_and_nesting() {
    let mut sess = session_with_basic();
    // Mixing integers and floats also just rides along into JSON without any special
    // conversion (`Ty::Json` is dynamically typed, so unlike duckdb it does not unify numeric
    // types; same policy as the mixed-type test in `array_literal_is_sugar_for_list_value`).
    assert_eq!(one(&mut sess, "to_json([1, 2.5])"), s("[1,2.5]"));
    // An array containing a NULL element.
    assert_eq!(one(&mut sess, "to_json([1, NULL, 3])"), s("[1,null,3]"));
    // An array of arrays (nested).
    assert_eq!(one(&mut sess, "to_json([1, [2, 3]])"), s("[1,[2,3]]"));
}

// --- Interaction: combined with WHERE/JOIN/aggregation ------------------------------

#[test]
fn glob_and_similar_to_work_as_join_conditions() {
    let mut sess = session_with_basic();
    // `GLOB`/`SIMILAR TO` are ordinary bool expressions, so they can also be written in `JOIN ... ON`.
    let rows = run(
        &mut sess,
        "SELECT DISTINCT a.name FROM t a JOIN t b ON a.name GLOB b.name \
         WHERE a.name = 'name_0' ORDER BY a.name",
    );
    assert_eq!(rows, vec![vec![s("name_0")]]);
}

#[test]
fn array_literal_can_feed_an_aggregate_input() {
    let mut sess = session_with_basic();
    // Aggregate the array literal's element count (`json_array_length`)
    // -- combines the new feature (array literals) with the existing aggregation pipeline.
    let rows = run(&mut sess, "SELECT sum(json_array_length([id, id, id])) FROM t WHERE id < 4");
    assert_eq!(rows, vec![vec![Value::I128(12)]]);
}
