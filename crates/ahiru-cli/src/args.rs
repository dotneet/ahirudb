//! Command-line parsing.
//!
//! Flags follow the DuckDB CLI's spelling, which uses a single dash for long
//! names (`-csv`, `-readonly`, `-maxrows`); the GNU-style `--csv` is accepted
//! for all of them too, since muscle memory goes both ways.

use std::path::PathBuf;

use crate::output::{Mode, Settings};

/// One unit of work handed to the engine before the REPL starts (or instead
/// of it).
#[derive(Debug, Clone, PartialEq)]
pub enum Job {
    /// `-c SQL`
    Sql(String),
    /// `-f FILE`
    File(PathBuf),
}

/// The legacy subcommands, kept working as they were.
#[derive(Debug, Clone, PartialEq)]
pub enum Legacy {
    Schema(String),
    Dump(String, usize),
    Query(Vec<String>, String),
}

#[derive(Debug, Clone)]
pub struct Options {
    pub legacy: Option<Legacy>,
    /// Positional file arguments: `path`, `name=path`, `a+b`, globs, dirs.
    pub files: Vec<String>,
    pub jobs: Vec<Job>,
    pub settings: Settings,
    /// `-init FILE`; `None` after `-no-init`.
    pub init: Option<PathBuf>,
    pub readonly: bool,
    pub timer: bool,
    pub echo: bool,
    pub bail: bool,
    /// `-batch` forces non-interactive even on a terminal.
    pub batch: bool,
    /// True when the user pinned the output mode explicitly, so the REPL
    /// doesn't override it with `duckbox`.
    pub mode_set: bool,
    /// True when `-separator` was given, so `.mode` leaves it alone.
    pub sep_explicit: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            legacy: None,
            files: Vec::new(),
            jobs: Vec::new(),
            settings: Settings { mode: Mode::Duckbox, ..Settings::default() },
            init: default_init_path(),
            readonly: false,
            timer: false,
            echo: false,
            bail: false,
            batch: false,
            mode_set: false,
            sep_explicit: false,
        }
    }
}

/// `~/.ahirurc`, the counterpart of DuckDB's `~/.duckdbrc`.
pub fn default_init_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".ahirurc"))
}

/// `~/.ahiru_history`.
pub fn history_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".ahiru_history"))
}

pub const USAGE: &str = "\
usage:
  ahiru [OPTIONS] [FILE|NAME=FILE]...          interactive shell (or stdin script)
  ahiru [OPTIONS] -c 'SQL' [FILE]...           run SQL and exit
  ahiru schema <file>                          show the schema (RowGroups too, for Parquet)
  ahiru dump   <file> [n]                      show the first n rows (default 20)
  ahiru query  <file>... <sql>                 run SQL (tables are t, t2, ... and the file name)

file arguments:
  path                 registered as t, t2, ... and under its own path text
  name=path            registered under `name`
  'dir/'               every supported file below it, as one table
  'data/*.parquet'     glob, as one table (quote it to keep the shell out)
  a.parquet+b.parquet  several files as one table (`\\+` for a literal `+`)

options:
  -c, --command SQL    run SQL (repeatable)
  -f, --file FILE      run a SQL script (repeatable)
  -init FILE           run FILE at startup (default: ~/.ahirurc)
  -no-init             skip the startup file
  -csv -tsv -json -jsonlines -markdown -box -duckbox -line -list -insert -trash
                       output mode (default: duckbox interactive, tsv otherwise)
  -noheader            omit the header row
  -separator SEP       column separator for csv/tsv/list modes
  -nullvalue STR       text printed for NULL (default: NULL)
  -maxrows N           rows shown in duckbox mode before eliding (0 = all)
  -maxwidth N          table width in boxed modes before eliding columns
  -readonly            reject INSERT/UPDATE/DELETE/CREATE/DROP/COPY
  -timer               print query wall time
  -echo                print each statement before running it
  -bail                stop at the first error
  -batch               never enter the interactive shell
  -h, --help           this message

supported formats: .parquet / .csv / .tsv / .jsonl / .ndjson / .json
detected from the file extension.
";

pub enum Parsed {
    Options(Box<Options>),
    Help,
    Error(String),
}

pub fn parse(argv: &[String]) -> Parsed {
    let mut o = Options::default();
    let mut positional: Vec<String> = Vec::new();
    let mut i = 0;
    // `--` stops flag parsing; everything after it is a file argument.
    let mut only_files = false;
    // Once `-separator` is given it wins over any later mode's default.
    let mut sep_explicit = false;

    while i < argv.len() {
        let a = argv[i].as_str();
        if only_files || !a.starts_with('-') || a == "-" {
            positional.push(a.to_string());
            i += 1;
            continue;
        }
        if a == "--" {
            only_files = true;
            i += 1;
            continue;
        }
        // Accept both `-csv` and `--csv`.
        let flag = a.trim_start_matches('-');
        let take_value = |i: &mut usize, name: &str| -> Result<String, String> {
            *i += 1;
            argv.get(*i).cloned().ok_or_else(|| format!("{name} needs a value"))
        };
        macro_rules! value {
            ($name:expr) => {
                match take_value(&mut i, $name) {
                    Ok(v) => v,
                    Err(e) => return Parsed::Error(e),
                }
            };
        }
        match flag {
            "h" | "help" => return Parsed::Help,
            "c" | "command" | "cmd" => o.jobs.push(Job::Sql(value!("-c"))),
            "f" | "file" => o.jobs.push(Job::File(PathBuf::from(value!("-f")))),
            "init" => o.init = Some(PathBuf::from(value!("-init"))),
            "no-init" | "noinit" => o.init = None,
            "separator" => {
                o.settings.separator = value!("-separator");
                sep_explicit = true;
            }
            "nullvalue" => o.settings.null = value!("-nullvalue"),
            "maxrows" => {
                let v = value!("-maxrows");
                match v.parse() {
                    Ok(n) => o.settings.maxrows = n,
                    Err(_) => return Parsed::Error(format!("-maxrows: not a number: {v}")),
                }
            }
            "maxwidth" => {
                let v = value!("-maxwidth");
                match v.parse() {
                    Ok(n) => o.settings.maxwidth = n,
                    Err(_) => return Parsed::Error(format!("-maxwidth: not a number: {v}")),
                }
            }
            "noheader" | "no-header" => o.settings.header = false,
            "header" => o.settings.header = true,
            "readonly" | "read-only" => o.readonly = true,
            "timer" => o.timer = true,
            "echo" => o.echo = true,
            "bail" => o.bail = true,
            "batch" => o.batch = true,
            "mode" => {
                let v = value!("-mode");
                match Mode::parse(&v) {
                    Some(m) => {
                        o.settings.mode = m;
                        o.mode_set = true;
                        if !sep_explicit {
                            o.settings.separator = m.default_separator().to_string();
                        }
                    }
                    None => return Parsed::Error(format!("unknown mode: {v}")),
                }
            }
            other => match Mode::parse(other) {
                Some(m) => {
                    o.settings.mode = m;
                    o.mode_set = true;
                    if !sep_explicit {
                        o.settings.separator = m.default_separator().to_string();
                    }
                }
                None => return Parsed::Error(format!("unknown option: {a}")),
            },
        }
        i += 1;
    }

    // Legacy subcommands keep their exact old shape, including TSV output.
    match positional.first().map(String::as_str) {
        Some("schema") if positional.len() == 2 => {
            o.legacy = Some(Legacy::Schema(positional[1].clone()));
        }
        Some("dump") if positional.len() >= 2 && positional.len() <= 3 => {
            let n = positional.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
            o.legacy = Some(Legacy::Dump(positional[1].clone(), n));
        }
        Some("query") if positional.len() >= 3 => {
            let last = positional.len() - 1;
            o.legacy = Some(Legacy::Query(positional[1..last].to_vec(), positional[last].clone()));
        }
        Some("schema") | Some("dump") | Some("query") => {
            return Parsed::Error(format!("wrong number of arguments for `{}`", positional[0]));
        }
        _ => o.files = positional,
    }
    o.sep_explicit = sep_explicit;
    if o.legacy.is_some() && !o.mode_set {
        o.settings.mode = Mode::Tsv;
        if !sep_explicit {
            o.settings.separator = Mode::Tsv.default_separator().to_string();
        }
    }
    Parsed::Options(Box::new(o))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    fn opts(v: &[&str]) -> Options {
        match parse(&args(v)) {
            Parsed::Options(o) => *o,
            Parsed::Help => panic!("unexpected help"),
            Parsed::Error(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn legacy_query_still_parses() {
        let o = opts(&["query", "a.parquet", "b.parquet", "SELECT 1"]);
        assert_eq!(
            o.legacy,
            Some(Legacy::Query(vec!["a.parquet".into(), "b.parquet".into()], "SELECT 1".into()))
        );
    }

    #[test]
    fn commands_and_files_separate() {
        let o = opts(&["-c", "SELECT 1", "-c", "SELECT 2", "x.parquet"]);
        assert_eq!(o.jobs, vec![Job::Sql("SELECT 1".into()), Job::Sql("SELECT 2".into())]);
        assert_eq!(o.files, vec!["x.parquet".to_string()]);
    }

    #[test]
    fn double_dash_stops_flag_parsing() {
        let o = opts(&["--", "-weird-name.csv"]);
        assert_eq!(o.files, vec!["-weird-name.csv".to_string()]);
    }

    #[test]
    fn unknown_option_is_an_error() {
        assert!(matches!(parse(&args(&["-nope"])), Parsed::Error(_)));
    }
}
