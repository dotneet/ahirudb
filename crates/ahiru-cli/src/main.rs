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
    match run(opts) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(opts: Options) -> R<()> {
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
