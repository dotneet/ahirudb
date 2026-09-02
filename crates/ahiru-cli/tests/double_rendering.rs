//! The CLI must spell a DOUBLE the same way the engine does.
//!
//! `render::render` used to use Rust's `Display` for `f64`. That is shortest-round-trip
//! too, but it never switches to exponential notation, so `1e30` printed as
//! `1000000000000000000000000000000` in a result table while `CAST(1e30::DOUBLE AS
//! VARCHAR)` -- and a CSV/JSONL export of the same value -- printed `1e+30`. The renderer
//! now goes through `ahiru_core::expr::kernels::fmt_f64`, the engine's one formatter.
//!
//! Every output mode funnels through `render::render`, so the mode sweep below is what
//! actually establishes that: one shared implementation, ten identical spellings.

use std::process::{Command, Stdio};

fn run(args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_ahiru"))
        .arg("-no-init")
        .args(args)
        .stdin(Stdio::null())
        .output()
        .expect("failed to run ahiru");
    assert!(out.status.success(), "ahiru failed:\n{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// The bare value and the same value cast to VARCHAR, in one row, in csv mode.
fn bare_and_cast(expr: &str) -> (String, String) {
    let sql = format!("SELECT ({expr}) AS a, CAST(({expr}) AS VARCHAR) AS b");
    let text = run(&["-csv", "-noheader", "-c", &sql]);
    let line = text.lines().next().unwrap_or_default().to_string();
    let (a, b) = line.split_once(',').expect("two columns");
    (a.to_string(), b.trim().to_string())
}

/// Both spellings agree, and both are the one DuckDB produces.
fn assert_spelled(expr: &str, want: &str) {
    let (bare, cast) = bare_and_cast(expr);
    assert_eq!(bare, want, "bare display of {expr}");
    assert_eq!(cast, want, "CAST({expr} AS VARCHAR)");
}

#[test]
fn large_and_small_magnitudes_use_exponential_notation() {
    // duckdb -csv -c "SELECT 1e30::DOUBLE": 1e+30 (this printed
    // 1000000000000000000000000000000 before the fix).
    assert_spelled("1e30::DOUBLE", "1e+30");
    // duckdb: 1e+16 / 1e-07
    assert_spelled("1e16::DOUBLE", "1e+16");
    assert_spelled("1e-7::DOUBLE", "1e-07");
    // duckdb: -1e+30
    assert_spelled("-1e30::DOUBLE", "-1e+30");
}

#[test]
fn ordinary_magnitudes_keep_fixed_notation() {
    // duckdb: 0.1 / 1000000000000000.0 / 123456789.0 / 0.0
    assert_spelled("0.1::DOUBLE", "0.1");
    assert_spelled("1e15::DOUBLE", "1000000000000000.0");
    assert_spelled("123456789.0::DOUBLE", "123456789.0");
    assert_spelled("0.0::DOUBLE", "0.0");
}

/// Non-finite doubles, in the one spelling DuckDB uses.
///
/// Before this pair of fixes there were two: the CLI printed Rust's `inf` / `-inf` /
/// `NaN` while `CAST(x AS VARCHAR)` printed `Inf` / `-Inf` / `NaN`, and neither matched
/// `duckdb -csv -c "SELECT 'inf'::DOUBLE"`, which prints `inf`.
///
/// The CSV and JSONL writers keep their own spellings (`NaN` / `Infinity` / `-Infinity`)
/// on purpose -- each has its own reader to satisfy -- so this is not a crate-wide
/// rename, only the display/cast path.
#[test]
fn non_finite_doubles_match_the_cast_spelling() {
    // duckdb: nan / inf / -inf
    assert_spelled("'nan'::DOUBLE", "nan");
    assert_spelled("'inf'::DOUBLE", "inf");
    assert_spelled("'-inf'::DOUBLE", "-inf");
}

/// The text cast has to stay reversible: whatever `CAST(x AS VARCHAR)` writes,
/// `CAST(... AS DOUBLE)` has to read back as the same value.
#[test]
fn non_finite_doubles_round_trip_through_varchar() {
    let text = run(&[
        "-csv",
        "-noheader",
        "-c",
        "SELECT CAST(CAST('inf'::DOUBLE AS VARCHAR) AS DOUBLE) AS a, \
         CAST(CAST('-inf'::DOUBLE AS VARCHAR) AS DOUBLE) AS b, \
         isnan(CAST(CAST('nan'::DOUBLE AS VARCHAR) AS DOUBLE)) AS c",
    ]);
    assert_eq!(text.lines().next().unwrap_or_default(), "inf,-inf,true");
}

/// Rendering is mode-independent by design (`render.rs`'s module doc), so a value that
/// changed spelling has to change it everywhere at once.
#[test]
fn every_output_mode_spells_a_double_the_same_way() {
    const SQL: &str = "SELECT 1e30::DOUBLE AS d";
    for mode in [
        "-csv",
        "-tsv",
        "-json",
        "-jsonlines",
        "-markdown",
        "-box",
        "-duckbox",
        "-line",
        "-list",
        "-insert",
    ] {
        let text = run(&[mode, "-c", SQL]);
        assert!(text.contains("1e+30"), "{mode} did not render 1e+30; got:\n{text}");
        assert!(
            !text.contains("1000000000000000000000000000000"),
            "{mode} still renders the expanded form; got:\n{text}"
        );
    }
}
