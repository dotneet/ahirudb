//! End-to-end tests for the shell surface: flags, output modes, dot commands,
//! stdin scripts, globs and path auto-registration.
//!
//! These drive the real binary rather than any internal API, because that is
//! the whole point of the surface being tested — the interesting behavior is
//! in argument handling, stream plumbing and exit codes, none of which a unit
//! test of a helper function would catch.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn repo_root() -> PathBuf {
    PathBuf::from(format!("{}/../..", env!("CARGO_MANIFEST_DIR")))
}

struct Run {
    stdout: String,
    stderr: String,
    ok: bool,
}

/// Runs the CLI from the repository root, so relative paths in the arguments
/// and in SQL text mean the same thing they would for a user standing there.
fn run(args: &[&str]) -> Run {
    run_with_stdin(args, None)
}

fn run_with_stdin(args: &[&str], stdin: Option<&str>) -> Run {
    let mut c = Command::new(env!("CARGO_BIN_EXE_ahiru"));
    // `-no-init` keeps the suite hermetic: a developer's own `~/.ahirurc`
    // must not be able to change what these tests observe. The one test that
    // cares about startup files passes `-init` explicitly, which wins.
    c.arg("-no-init")
        .args(args)
        .current_dir(repo_root())
        .stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = c.spawn().expect("failed to spawn ahiru");
    if let Some(text) = stdin {
        child.stdin.take().expect("no stdin").write_all(text.as_bytes()).expect("write stdin");
    }
    let out = child.wait_with_output().expect("failed to run ahiru");
    Run {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        ok: out.status.success(),
    }
}

fn tmp_file(tag: &str, ext: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("ahiru_cli_test_{tag}_{}.{ext}", std::process::id()));
    p
}

const BASIC: &str = "tests/data/basic.parquet";

// ---- flags ---------------------------------------------------------------

#[test]
fn dash_c_runs_sql_and_exits() {
    let r = run(&["-c", "SELECT 1 AS x", "-csv"]);
    assert!(r.ok, "{}", r.stderr);
    assert_eq!(r.stdout, "x\n1\n");
}

#[test]
fn several_dash_c_run_in_order() {
    let r = run(&["-csv", "-c", "SELECT 1 AS a", "-c", "SELECT 2 AS b"]);
    assert!(r.ok, "{}", r.stderr);
    assert_eq!(r.stdout, "a\n1\nb\n2\n");
}

#[test]
fn help_exits_zero_and_lists_options() {
    let r = run(&["--help"]);
    assert!(r.ok);
    assert!(r.stdout.contains("-readonly"), "{}", r.stdout);
    assert!(r.stdout.contains("interactive shell"), "{}", r.stdout);
}

#[test]
fn unknown_flag_is_an_error() {
    let r = run(&["-nonesuch"]);
    assert!(!r.ok);
    assert!(r.stderr.contains("unknown option"), "{}", r.stderr);
}

#[test]
fn noheader_and_separator_apply() {
    let r = run(&["-c", "SELECT 1 AS a, 2 AS b", "-list", "-noheader", "-separator", "|"]);
    assert!(r.ok, "{}", r.stderr);
    assert_eq!(r.stdout, "1|2\n");
}

#[test]
fn nullvalue_applies() {
    let r = run(&["-c", "SELECT NULL AS a", "-list", "-noheader", "-nullvalue", "(nil)"]);
    assert!(r.ok, "{}", r.stderr);
    assert_eq!(r.stdout, "(nil)\n");
}

#[test]
fn timer_reports_wall_time_on_stderr() {
    let r = run(&["-timer", "-csv", "-c", "SELECT 1"]);
    assert!(r.ok, "{}", r.stderr);
    assert!(r.stderr.contains("Run Time (s):"), "{}", r.stderr);
}

#[test]
fn readonly_rejects_writes() {
    let r = run(&["-readonly", "-c", "CREATE TABLE x (a INT)"]);
    assert!(!r.ok);
    assert!(r.stderr.contains("read-only"), "{}", r.stderr);
}

#[test]
fn bail_stops_at_the_first_error() {
    let r = run(&["-bail", "-csv", "-c", "SELECT nosuchfunc(); SELECT 1 AS after"]);
    assert!(!r.ok);
    assert!(!r.stdout.contains("after"), "second statement should not have run: {}", r.stdout);
}

#[test]
fn without_bail_a_failing_statement_does_not_stop_the_script() {
    let r = run(&["-csv", "-c", "SELECT nosuchfunc(); SELECT 1 AS after"]);
    assert!(!r.ok, "the run must still report failure");
    assert!(r.stdout.contains("after"), "{}", r.stdout);
}

// ---- output modes --------------------------------------------------------

#[test]
fn csv_quotes_only_when_needed() {
    let r = run(&["-csv", "-c", "SELECT 'a,b' AS x, 'plain' AS y"]);
    assert!(r.ok, "{}", r.stderr);
    assert_eq!(r.stdout, "x,y\n\"a,b\",plain\n");
}

#[test]
fn json_mode_emits_typed_literals() {
    let r = run(&["-json", "-c", "SELECT 1 AS n, 'x' AS s, NULL AS z"]);
    assert!(r.ok, "{}", r.stderr);
    let t: String = r.stdout.chars().filter(|c| !c.is_whitespace()).collect();
    assert_eq!(t, "[{\"n\":1,\"s\":\"x\",\"z\":null}]");
}

#[test]
fn markdown_mode_has_a_separator_row() {
    let r = run(&["-markdown", "-c", "SELECT 1 AS a, 2 AS b"]);
    assert!(r.ok, "{}", r.stderr);
    let lines: Vec<&str> = r.stdout.lines().collect();
    assert!(lines[0].starts_with('|') && lines[0].contains(" a "), "{:?}", lines);
    assert!(lines[1].contains("---"), "{:?}", lines);
    assert!(lines[2].contains(" 1 "), "{:?}", lines);
}

#[test]
fn insert_mode_emits_statements() {
    let r = run(&["-insert", "-c", "SELECT 1 AS a, 'it''s' AS b"]);
    assert!(r.ok, "{}", r.stderr);
    assert_eq!(r.stdout.trim(), "INSERT INTO tbl VALUES (1, 'it''s');");
}

#[test]
fn duckbox_mode_draws_a_box_with_a_footer() {
    let r = run(&["-duckbox", "-c", "SELECT 1 AS a"]);
    assert!(r.ok, "{}", r.stderr);
    assert!(r.stdout.contains('│'), "{}", r.stdout);
    assert!(r.stdout.contains("┌"), "{}", r.stdout);
    assert!(r.stdout.contains("1 row"), "{}", r.stdout);
}

#[test]
fn duckbox_elides_the_middle_when_maxrows_is_exceeded() {
    let sql = "SELECT range AS x FROM range(10)";
    let r = run(&["-duckbox", "-maxrows", "4", "-c", sql]);
    assert!(r.ok, "{}", r.stderr);
    assert!(r.stdout.contains('·'), "expected an elision row:\n{}", r.stdout);
    assert!(r.stdout.contains("10 rows"), "{}", r.stdout);
}

#[test]
fn trash_mode_prints_nothing() {
    let r = run(&["-trash", "-c", "SELECT 1"]);
    assert!(r.ok, "{}", r.stderr);
    assert_eq!(r.stdout, "");
}

// ---- input plumbing ------------------------------------------------------

#[test]
fn a_piped_script_runs_statement_by_statement() {
    let r = run_with_stdin(&["-csv"], Some("SELECT 1 AS a;\nSELECT 2 AS b;\n"));
    assert!(r.ok, "{}", r.stderr);
    assert_eq!(r.stdout, "a\n1\nb\n2\n");
}

#[test]
fn a_piped_script_may_switch_modes_mid_stream() {
    let r = run_with_stdin(&[], Some(".mode csv\nSELECT 1 AS a;\n"));
    assert!(r.ok, "{}", r.stderr);
    assert_eq!(r.stdout, "a\n1\n");
}

#[test]
fn dash_f_runs_a_script_file() {
    let p = tmp_file("script", "sql");
    std::fs::write(&p, ".mode csv\nSELECT 7 AS seven;\n").unwrap();
    let r = run(&["-f", p.to_str().unwrap()]);
    let _ = std::fs::remove_file(&p);
    assert!(r.ok, "{}", r.stderr);
    assert_eq!(r.stdout, "seven\n7\n");
}

#[test]
fn init_file_runs_before_the_command() {
    let p = tmp_file("init", "sql");
    std::fs::write(&p, ".mode csv\n").unwrap();
    let r = run(&["-init", p.to_str().unwrap(), "-c", "SELECT 3 AS three"]);
    let _ = std::fs::remove_file(&p);
    assert!(r.ok, "{}", r.stderr);
    assert_eq!(r.stdout, "three\n3\n");
}

#[test]
fn dot_read_runs_a_script_file() {
    let p = tmp_file("read", "sql");
    std::fs::write(&p, "SELECT 42 AS answer;\n").unwrap();
    let r = run(&["-csv", "-c", &format!(".read {}", p.display())]);
    let _ = std::fs::remove_file(&p);
    assert!(r.ok, "{}", r.stderr);
    assert_eq!(r.stdout, "answer\n42\n");
}

// ---- file arguments ------------------------------------------------------

#[test]
fn positional_file_is_registered_as_t() {
    let r = run(&["-csv", BASIC, "-c", "SELECT count(*) AS n FROM t"]);
    assert!(r.ok, "{}", r.stderr);
    assert!(r.stdout.lines().nth(1).is_some(), "{}", r.stdout);
}

#[test]
fn named_file_argument_registers_under_that_name() {
    let r = run(&["-csv", &format!("basic={BASIC}"), "-c", "SELECT count(*) AS n FROM basic"]);
    assert!(r.ok, "{}", r.stderr);
    let n: i64 = r.stdout.lines().nth(1).unwrap().trim().parse().unwrap();
    assert!(n > 0);
}

#[test]
fn a_path_in_sql_is_registered_without_being_named_on_the_command_line() {
    let r = run(&["-csv", "-c", &format!("SELECT count(*) AS n FROM '{BASIC}'")]);
    assert!(r.ok, "{}", r.stderr);
    let n: i64 = r.stdout.lines().nth(1).unwrap().trim().parse().unwrap();
    assert!(n > 0);
}

#[test]
fn read_parquet_of_an_unregistered_path_works_too() {
    let r = run(&["-csv", "-c", &format!("SELECT count(*) AS n FROM read_parquet('{BASIC}')")]);
    assert!(r.ok, "{}", r.stderr);
    let n: i64 = r.stdout.lines().nth(1).unwrap().trim().parse().unwrap();
    assert!(n > 0);
}

#[test]
fn a_quoted_glob_argument_becomes_one_table() {
    let one = run(&["-csv", "tests/data/multi/a.parquet", "-c", "SELECT count(*) AS n FROM t"]);
    assert!(one.ok, "{}", one.stderr);
    let n1: i64 = one.stdout.lines().nth(1).unwrap().trim().parse().unwrap();
    let all = run(&["-csv", "tests/data/multi/*.parquet", "-c", "SELECT count(*) AS n FROM t"]);
    assert!(all.ok, "{}", all.stderr);
    let n2: i64 = all.stdout.lines().nth(1).unwrap().trim().parse().unwrap();
    assert!(n2 > n1, "the glob covers three files, one file has fewer rows: {n2} vs {n1}");
}

/// A whole directory becomes one table, which is what makes a Hive-partitioned
/// layout usable without naming every part.
#[test]
fn a_directory_argument_becomes_one_table() {
    let r = run(&["-csv", "d=tests/data/multi", "-c", "SELECT count(*) AS n FROM d"]);
    assert!(r.ok, "{}", r.stderr);
    let n: i64 = r.stdout.lines().nth(1).unwrap().trim().parse().unwrap();
    assert!(n > 0);
}

#[test]
fn a_glob_written_in_sql_is_registered() {
    let r = run(&["-csv", "-c", "SELECT count(*) AS n FROM 'tests/data/multi/*.parquet'"]);
    assert!(r.ok, "{}", r.stderr);
    let n: i64 = r.stdout.lines().nth(1).unwrap().trim().parse().unwrap();
    assert!(n > 0);
}

// ---- dot commands --------------------------------------------------------

#[test]
fn dot_tables_lists_registered_tables() {
    let r = run(&[&format!("basic={BASIC}"), "-c", ".tables"]);
    assert!(r.ok, "{}", r.stderr);
    assert!(r.stdout.contains("basic"), "{}", r.stdout);
}

#[test]
fn dot_schema_prints_create_table_text() {
    let r = run(&[&format!("basic={BASIC}"), "-c", ".schema basic"]);
    assert!(r.ok, "{}", r.stderr);
    assert!(r.stdout.contains("CREATE TABLE basic("), "{}", r.stdout);
}

#[test]
fn dot_import_registers_a_file() {
    let r = run(&["-csv", "-c", &format!(".import {BASIC} b"), "-c", "SELECT count(*) FROM b"]);
    assert!(r.ok, "{}", r.stderr);
}

#[test]
fn dot_output_redirects_to_a_file() {
    let p = tmp_file("output", "csv");
    let r = run(&[
        "-csv",
        "-c",
        &format!(".output {}", p.display()),
        "-c",
        "SELECT 5 AS five",
        "-c",
        ".output",
    ]);
    assert!(r.ok, "{}", r.stderr);
    let written = std::fs::read_to_string(&p).unwrap();
    let _ = std::fs::remove_file(&p);
    assert_eq!(written, "five\n5\n");
    assert_eq!(r.stdout, "", "nothing should have reached stdout");
}

#[test]
fn dot_once_redirects_exactly_one_statement() {
    let p = tmp_file("once", "csv");
    let r = run(&[
        "-csv",
        "-c",
        &format!(".once {}", p.display()),
        "-c",
        "SELECT 1 AS a",
        "-c",
        "SELECT 2 AS b",
    ]);
    assert!(r.ok, "{}", r.stderr);
    let written = std::fs::read_to_string(&p).unwrap();
    let _ = std::fs::remove_file(&p);
    assert_eq!(written, "a\n1\n");
    assert_eq!(r.stdout, "b\n2\n");
}

/// Dot-command output goes to the same sink as query results, so a schema
/// dump can be redirected to a file like anything else.
#[test]
fn dot_output_captures_dot_command_output_too() {
    let p = tmp_file("output_dot", "txt");
    let r = run(&[
        &format!("basic={BASIC}"),
        "-c",
        &format!(".output {}", p.display()),
        "-c",
        ".tables",
        "-c",
        ".output",
    ]);
    assert!(r.ok, "{}", r.stderr);
    let written = std::fs::read_to_string(&p).unwrap();
    let _ = std::fs::remove_file(&p);
    assert_eq!(written, "basic\n");
    assert_eq!(r.stdout, "");
}

#[test]
fn dot_log_redirects_diagnostics() {
    let p = tmp_file("log", "txt");
    let r = run(&[
        "-csv",
        "-c",
        &format!(".log {}", p.display()),
        "-c",
        "SELECT 1 AS a",
        "-c",
        ".log off",
    ]);
    assert!(r.ok, "{}", r.stderr);
    let written = std::fs::read_to_string(&p).unwrap();
    let _ = std::fs::remove_file(&p);
    assert!(written.contains("(1 rows)"), "{written}");
    assert!(!r.stderr.contains("(1 rows)"), "{}", r.stderr);
}

/// `.excel` writes CSV to a temporary file and hands it to the system opener.
/// `PATH` is emptied so nothing actually pops up on a developer's screen; the
/// file must still be written, and the CLI must say what it did.
#[test]
fn dot_excel_writes_csv_even_when_no_opener_exists() {
    let out = Command::new(env!("CARGO_BIN_EXE_ahiru"))
        .args(["-no-init", BASIC, "-c", ".excel", "-c", "SELECT id FROM t LIMIT 1"])
        .current_dir(repo_root())
        .env("PATH", "/nonexistent")
        .output()
        .expect("failed to run ahiru");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let path = stderr
        .split("(wrote ")
        .nth(1)
        .and_then(|s| s.split(" \u{2014}").next())
        .unwrap_or_else(|| panic!("no `(wrote ...)` line in stderr: {stderr}"));
    let written = std::fs::read_to_string(path).unwrap();
    let _ = std::fs::remove_file(path);
    assert_eq!(written, "id\n0\n");
}

#[test]
fn dot_print_and_dot_show_work() {
    let r = run(&["-c", ".print hello", "-c", ".show"]);
    assert!(r.ok, "{}", r.stderr);
    assert!(r.stdout.contains("hello"), "{}", r.stdout);
    assert!(r.stdout.contains("mode:"), "{}", r.stdout);
}

#[test]
fn dot_shell_runs_a_command() {
    let r = run(&["-c", ".shell echo from-the-shell"]);
    assert!(r.ok, "{}", r.stderr);
    assert!(r.stdout.contains("from-the-shell"), "{}", r.stdout);
}

#[test]
fn unknown_dot_command_reports_an_error() {
    let r = run(&["-c", ".nosuchthing"]);
    assert!(!r.ok);
    assert!(r.stderr.contains("unknown command"), "{}", r.stderr);
}

#[test]
fn dot_help_lists_commands() {
    let r = run(&["-c", ".help"]);
    assert!(r.ok, "{}", r.stderr);
    assert!(r.stdout.contains(".mode"), "{}", r.stdout);
    assert!(r.stdout.contains(".maxrows"), "{}", r.stdout);
}

#[test]
fn dot_exit_sets_the_exit_code() {
    let r = run(&["-c", ".exit 3", "-c", "SELECT 1"]);
    assert!(!r.ok);
    assert!(!r.stdout.contains('1'), "statements after .exit must not run: {}", r.stdout);
}

// ---- SUMMARIZE -----------------------------------------------------------

#[test]
fn summarize_describes_every_column() {
    let r = run(&["-csv", BASIC, "-c", "SUMMARIZE t"]);
    assert!(r.ok, "{}", r.stderr);
    let lines: Vec<&str> = r.stdout.lines().collect();
    assert!(lines[0].starts_with("column_name,column_type,min,max,approx_unique"), "{:?}", lines);
    assert!(lines.len() > 1, "expected one row per column: {:?}", lines);
}

#[test]
fn summarize_accepts_a_table_function_call() {
    let r = run(&["-csv", "-c", "SUMMARIZE read_csv('tests/data/orders.csv')"]);
    assert!(r.ok, "{}", r.stderr);
    assert!(r.stdout.contains("column_name"), "{}", r.stdout);
    assert!(r.stdout.lines().count() > 1, "{}", r.stdout);
}

#[test]
fn summarize_of_a_subquery_says_what_to_do_instead() {
    let r = run(&["-csv", BASIC, "-c", "SUMMARIZE (SELECT id FROM t)"]);
    assert!(!r.ok);
    assert!(r.stderr.contains("does not accept a subquery"), "{}", r.stderr);
}

#[test]
fn summarize_accepts_a_path_literal() {
    let r = run(&["-csv", "-c", &format!("SUMMARIZE '{BASIC}'")]);
    assert!(r.ok, "{}", r.stderr);
    assert!(r.stdout.contains("column_name"), "{}", r.stdout);
}

// ---- legacy surface ------------------------------------------------------

#[test]
fn legacy_query_still_prints_tsv_with_a_header() {
    let r = run(&["query", BASIC, "SELECT 1 AS a, 2 AS b"]);
    assert!(r.ok, "{}", r.stderr);
    assert_eq!(r.stdout, "a\tb\n1\t2\n");
    assert!(r.stderr.contains("(1 rows)"), "{}", r.stderr);
}

#[test]
fn legacy_dump_still_works() {
    let r = run(&["dump", BASIC, "2"]);
    assert!(r.ok, "{}", r.stderr);
    assert_eq!(r.stdout.lines().count(), 3, "header + 2 rows: {}", r.stdout);
}

#[test]
fn legacy_schema_still_works() {
    let r = run(&["schema", BASIC]);
    assert!(r.ok, "{}", r.stderr);
    assert!(r.stdout.contains("columns:"), "{}", r.stdout);
    assert!(r.stdout.contains("row groups:"), "{}", r.stdout);
}

#[test]
fn ddl_tables_visible_in_dot_commands() {
    let r = run(&[
        "-c",
        "CREATE TABLE users (id INT, name VARCHAR)",
        "-c",
        ".tables",
        "-c",
        ".schema users",
    ]);
    assert!(r.ok, "{}", r.stderr);
    assert!(r.stdout.contains("users"), "{}", r.stdout);
    assert!(r.stdout.contains("CREATE TABLE users ("), "{}", r.stdout);
    assert!(r.stdout.contains("id integer"), "{}", r.stdout);
    assert!(r.stdout.contains("name varchar"), "{}", r.stdout);
}

#[test]
fn insert_mode_blob_hex() {
    let r = run(&["-mode", "insert", "-c", "SELECT unhex('deadbeef') AS b"]);
    assert!(r.ok, "{}", r.stderr);
    assert!(r.stdout.contains("X'deadbeef'"), "{}", r.stdout);
}
