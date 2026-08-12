//! The native CLI.
//!
//! Development and testing happen here. Debugging through wasm is inefficient, so
//! the split is "develop and test natively, use wasm only for size measurement"
//! (DESIGN.md §12).

mod args;
mod engine;
mod glob;
mod lineedit;
mod output;
mod render;
mod repl;
mod summarize;

use std::io::{IsTerminal, Read};
use std::process::ExitCode;

use ahiru_core::parquet::file::open_bytes;
use ahiru_core::{FormatKind, Session};

use args::{Job, Legacy, Options, Parsed};
use engine::R;
use repl::Shell;

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let opts = match args::parse(&argv) {
        Parsed::Options(o) => *o,
        Parsed::Help => {
            print!("{}", args::USAGE);
            return ExitCode::SUCCESS;
        }
        Parsed::Error(e) => {
            eprintln!("error: {e}\n");
            eprint!("{}", args::USAGE);
            return ExitCode::from(2);
        }
    };
    match run(opts, &argv) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Whether the command line explicitly named `-init`/`--init` (as opposed to
/// `opts.init` holding the implicit `~/.ahirurc` default). `args::parse`
/// doesn't record this distinction on `Options`, so it is re-derived here
/// from the raw argv with just enough of the same flag shape (`-x`/`--x`,
/// `--` stops flag parsing, `-no-init` cancels a preceding `-init`) to track
/// this one flag.
fn init_flag_explicit(argv: &[String]) -> bool {
    const VALUE_FLAGS: &[&str] = &[
        "c",
        "command",
        "cmd",
        "f",
        "file",
        "init",
        "separator",
        "nullvalue",
        "maxrows",
        "maxwidth",
    ];
    let mut explicit = false;
    let mut only_files = false;
    let mut i = 0;
    while i < argv.len() {
        let a = argv[i].as_str();
        if only_files || !a.starts_with('-') || a == "-" {
            i += 1;
            continue;
        }
        if a == "--" {
            only_files = true;
            i += 1;
            continue;
        }
        let flag = a.trim_start_matches('-');
        match flag {
            "init" => explicit = true,
            "no-init" | "noinit" => explicit = false,
            _ => {}
        }
        i += 1;
        if VALUE_FLAGS.contains(&flag) {
            i += 1; // skip the flag's value token
        }
    }
    explicit
}

fn run(opts: Options, argv: &[String]) -> R<()> {
    if let Some(l) = opts.legacy.clone() {
        return run_legacy(l, opts);
    }
    let mut sh = Shell::new(&opts);
    for f in &opts.files {
        sh.engine.register_arg(f)?;
    }
    // `~/.ahirurc` (or `-init FILE`). A missing default file is not an error;
    // a missing explicitly-named one is.
    if let Some(p) = &opts.init {
        if p.exists() {
            let text = std::fs::read_to_string(p)?;
            sh.run_script(&text)?;
        } else if init_flag_explicit(argv) {
            return Err(format!("{}: no such file or directory", p.display()).into());
        }
    }

    if opts.jobs.is_empty() {
        let stdin = std::io::stdin();
        if stdin.is_terminal() && !opts.batch {
            sh.interactive()?;
        } else {
            let mut text = String::new();
            std::io::stdin().lock().read_to_string(&mut text)?;
            sh.run_script(&text)?;
        }
        return sh.into_result();
    }

    for j in &opts.jobs {
        match j {
            Job::Sql(sql) => sh.run_script(sql)?,
            Job::File(p) => {
                let text = std::fs::read_to_string(p)
                    .map_err(|e| format!("{}: {e}", p.to_string_lossy()))?;
                sh.run_script(&text)?;
            }
        }
    }
    sh.into_result()
}

fn run_legacy(l: Legacy, opts: Options) -> R<()> {
    match l {
        Legacy::Schema(path) => cmd_schema(&path),
        Legacy::Dump(path, n) => {
            let mut sh = Shell::new(&opts);
            sh.engine.register_arg(&path)?;
            sh.run_statement(&format!("SELECT * FROM t LIMIT {n}"))?;
            sh.into_result()
        }
        Legacy::Query(files, sql) => {
            let mut sh = Shell::new(&opts);
            for f in &files {
                sh.engine.register_arg(f)?;
            }
            sh.run_statement(&sql)?;
            sh.into_result()
        }
    }
}

fn cmd_schema(path: &str) -> R<()> {
    let bytes = std::fs::read(path)?;
    let kind = FormatKind::detect(path);

    // Print the format-independent schema first.
    let mut s = Session::new();
    s.register_bytes_as("t", bytes.clone(), kind)?;
    let table = s.catalog.index_of("t").ok_or("table not registered")?;
    match s.catalog.get_mut(table).ok_or("table not registered")?.resolve()? {
        Ok(()) => {}
        Err(_) => return Err("could not resolve even with the whole file in memory".into()),
    }
    let t = s.catalog.get(table).ok_or("table not registered")?;
    println!("format: {kind:?}");
    println!("\ncolumns:");
    for (i, f) in t.schema().iter().enumerate() {
        println!(
            "  [{i}] {:<24} {:<12} {}",
            f.name,
            f.ty.name(),
            if f.nullable { "NULL" } else { "NOT NULL" }
        );
    }
    println!("splits: {}", t.num_splits());

    // For Parquet only, print an additional breakdown, since the physical layout is worth seeing.
    if kind == FormatKind::Parquet {
        print_parquet_details(&bytes)?;
    }
    Ok(())
}

fn print_parquet_details(bytes: &[u8]) -> R<()> {
    let f = open_bytes(bytes)?;
    println!("\nrows: {}", f.num_rows());
    if let Some(c) = &f.meta.created_by {
        println!("created by: {c}");
    }
    println!("\nrow groups:");
    for (i, rg) in f.meta.row_groups.iter().enumerate() {
        println!("  [{i}] rows={} bytes={}", rg.num_rows, rg.total_byte_size);
        for (j, cc) in rg.columns.iter().enumerate() {
            if let Some(m) = &cc.meta {
                let (start, end) = m.byte_range();
                println!(
                    "        col {j}: codec={:?} enc={:?} range=[{start},{end}) values={}",
                    m.codec, m.encodings, m.num_values
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn init_flag_explicit_detects_the_flag_and_its_spellings() {
        assert!(init_flag_explicit(&args(&["-init", "/tmp/x"])));
        assert!(init_flag_explicit(&args(&["--init", "/tmp/x"])));
        assert!(!init_flag_explicit(&args(&["-c", "SELECT 1"])));
        assert!(!init_flag_explicit(&args(&["-no-init"])));
        // Later flags win, matching `args::parse`'s left-to-right semantics.
        assert!(!init_flag_explicit(&args(&["-init", "/tmp/x", "-no-init"])));
        assert!(init_flag_explicit(&args(&["-no-init", "-init", "/tmp/x"])));
        // `--` stops flag parsing, so a bare `-init` after it is a file argument.
        assert!(!init_flag_explicit(&args(&["--", "-init"])));
        // The value of an unrelated flag must not be mistaken for `-init`.
        assert!(!init_flag_explicit(&args(&["-c", "-init"])));
    }

    /// An explicitly named `-init FILE` that does not exist must be an error
    /// (with a nonzero exit code via `main`), not silently ignored.
    #[test]
    fn explicit_missing_init_file_is_an_error() {
        let p = std::env::temp_dir()
            .join(format!("ahiru_main_test_missing_init_{}.sql", std::process::id()));
        let _ = std::fs::remove_file(&p);

        let opts = Options {
            init: Some(p.clone()),
            jobs: vec![Job::Sql("SELECT 1".to_string())],
            ..Options::default()
        };
        let argv = args(&["-init", p.to_str().expect("tmp path is valid UTF-8"), "-c", "SELECT 1"]);

        let err = run(opts, &argv).expect_err("an explicitly named missing -init file must error");
        assert!(
            err.to_string().contains(&p.to_string_lossy().to_string()),
            "error should name the missing file: {err}"
        );
    }

    /// `~/.ahirurc` (what `opts.init` defaults to when `-init` is never
    /// given) staying absent must remain silent.
    #[test]
    fn implicit_missing_default_init_is_silent() {
        let p = std::env::temp_dir()
            .join(format!("ahiru_main_test_missing_default_init_{}.sql", std::process::id()));
        let _ = std::fs::remove_file(&p);

        let opts = Options {
            init: Some(p), // stands in for the unset `~/.ahirurc` default
            jobs: vec![Job::Sql("SELECT 1".to_string())],
            ..Options::default()
        };
        let argv = args(&["-c", "SELECT 1"]); // no `-init` on the command line

        assert!(run(opts, &argv).is_ok());
    }
}
