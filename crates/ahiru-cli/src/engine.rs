//! Session wrapper: file registration and query execution.
//!
//! `ahiru-core` is `no_std` and never touches a filesystem, so everything that
//! reads bytes off disk — resolving a path argument, expanding a glob, walking
//! a Hive-partitioned directory, writing a `COPY ... TO` result — lives here.
//! The core only ever sees `(name, bytes)` pairs.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ahiru_core::vector::{Ty, Value};
use ahiru_core::{FormatKind, Prepared, QueryStep, Session};

use crate::glob;
use crate::output::Writer;

pub type R<T> = Result<T, Box<dyn std::error::Error>>;

/// Extensions the CLI knows how to register. Used when expanding a directory
/// argument, where we must not hand unrelated files (`_SUCCESS`, `.crc`, ...)
/// to the core.
pub const SUPPORTED_EXTS: &[&str] = &["parquet", "csv", "tsv", "jsonl", "ndjson", "json"];

pub fn is_supported_path(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| SUPPORTED_EXTS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Outcome of one statement.
pub struct Outcome {
    pub rows: usize,
    /// `COPY ... TO 'path'`: bytes written and where.
    pub copied: Option<(String, usize)>,
    pub elapsed: Duration,
}

pub struct Engine {
    s: Session,
    /// `(table index, part index) -> original bytes`, for codec delegation.
    sources: HashMap<(usize, usize), Vec<u8>>,
    /// Every name the catalog currently knows, so repeated references to the
    /// same path in later statements don't re-read the file.
    registered: HashSet<String>,
    /// Table names in registration order, for `.tables`.
    names: Vec<String>,
    /// Next auto-assigned short alias index (`t`, `t2`, `t3`, ...).
    next_alias: usize,
    pub readonly: bool,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub fn new() -> Engine {
        Engine {
            s: Session::new(),
            sources: HashMap::new(),
            registered: HashSet::new(),
            names: Vec::new(),
            next_alias: 0,
            readonly: false,
        }
    }

    /// Table names registered so far (including file table aliases, in-memory tables, and views).
    pub fn table_names(&self) -> Vec<String> {
        let mut out = self.names.clone();
        for name in self.s.catalog.mem_names() {
            if !out.iter().any(|n| n.eq_ignore_ascii_case(name)) {
                out.push(name.to_string());
            }
        }
        for name in self.s.catalog.view_names() {
            if !out.iter().any(|n| n.eq_ignore_ascii_case(name)) {
                out.push(name.to_string());
            }
        }
        out
    }

    /// Registers one command-line file argument. Accepted forms:
    ///
    /// - `path`                  — auto alias `t`, `t2`, ...
    /// - `name=path`             — explicit alias
    /// - `a.parquet+b.parquet`   — several files as one table (legacy syntax)
    /// - `dir/`                  — every supported file below it, one table
    /// - `data/*.parquet`        — glob, one table
    ///
    /// Every form is *also* registered under its literal path text so that
    /// `FROM 'data/x.parquet'` / `read_parquet('data/x.parquet')` resolve.
    /// Returns the alias the table got.
    pub fn register_arg(&mut self, arg: &str) -> R<String> {
        let (alias, spec) = match split_alias_spec(arg) {
            (Some(name), rest) => (Some(name.to_string()), rest),
            (None, rest) => (None, rest),
        };
        let alias = match alias {
            Some(a) => a,
            None => {
                self.next_alias += 1;
                if self.next_alias == 1 {
                    "t".to_string()
                } else {
                    format!("t{}", self.next_alias)
                }
            }
        };
        self.register_as(&alias, spec)?;
        Ok(alias)
    }

    /// Registers `spec` (path / glob / directory / `+`-joined list) as `name`.
    pub fn register_as(&mut self, name: &str, spec: &str) -> R<()> {
        let paths = resolve_spec(spec)?;
        if paths.is_empty() {
            return Err(format!("no files matched: {spec}").into());
        }
        // Detect per file. A directory / glob can mix extensions; applying the
        // first path's format to every part would read CSV as Parquet (or vice versa).
        if paths.len() == 1 {
            let kind = FormatKind::detect(&paths[0].to_string_lossy());
            let bytes = std::fs::read(&paths[0])?;
            let a = self.s.register_bytes_as(name, bytes.clone(), kind)?;
            self.sources.insert((a, 0), bytes.clone());
            self.remember(name);
            // Also reachable under its own path text (`parquet('...')`).
            let lit = paths[0].to_string_lossy().into_owned();
            if lit != name {
                let b = self.s.register_bytes(&lit, bytes.clone())?;
                self.sources.insert((b, 0), bytes);
                self.registered.insert(lit);
            }
        } else {
            let mut files = Vec::with_capacity(paths.len());
            for p in &paths {
                files.push((p.to_string_lossy().into_owned(), std::fs::read(p)?));
            }
            let a = self.s.register_multi_bytes(name, files.clone(), FormatKind::Auto)?;
            for (p, (_, bytes)) in files.into_iter().enumerate() {
                self.sources.insert((a, p), bytes);
            }
            self.remember(name);
        }
        // A multi-file table is deliberately *not* also registered under the
        // spec text: that would hold every byte twice, and the only way to
        // reach it that way is to write the glob in SQL as well, which
        // `autoregister` handles on demand.
        Ok(())
    }

    fn remember(&mut self, name: &str) {
        if self.registered.insert(name.to_string()) {
            self.names.push(name.to_string());
        }
    }

    /// Registers any file path that appears as a string literal in `sql` and
    /// isn't registered yet, so `SELECT * FROM 'data/x.parquet'` works without
    /// naming the file on the command line first. Paths that don't exist are
    /// left alone — the engine's own "table not found" error is the better
    /// message for a typo than anything we could produce here.
    pub fn autoregister(&mut self, sql: &str) {
        for lit in string_literals(sql) {
            if self.registered.contains(&lit) {
                continue;
            }
            let looks_like_path = glob::is_pattern(&lit)
                || Path::new(&lit).is_dir()
                || (is_supported_path(Path::new(&lit)) && Path::new(&lit).exists());
            if !looks_like_path {
                continue;
            }
            // A failure here is not fatal: the statement may not have meant
            // the literal as a table at all (a `COPY ... TO` target, say).
            let _ = self.register_as(&lit.clone(), &lit);
        }
    }

    /// Runs one statement, streaming rows into `out`.
    pub fn run(&mut self, sql: &str, out: &mut Writer) -> R<Outcome> {
        if self.readonly {
            if let Some(word) = write_statement_keyword(sql) {
                return Err(format!("{word} is not allowed in read-only mode (-readonly)").into());
            }
        }
        self.autoregister(sql);
        let start = Instant::now();

        // The core has no clock, so the query start time is passed in here for
        // CURRENT_DATE/CURRENT_TIMESTAMP/now() (DESIGN.md §2; the same role as the
        // JS host's `ahiru_set_now` call).
        let now_micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as i64)
            .unwrap_or(0);
        self.s.set_now(now_micros);

        let mut q = match self.s.prepare(sql, &[])? {
            Prepared::Ready(q) => q,
            // The whole file is in memory, so this is unreachable.
            Prepared::NeedIo(_) => return Err("unexpected io request".into()),
        };
        // `COPY (SELECT ...) TO 'path'`: `ahiru-core` is no_std and cannot touch the
        // filesystem, so writing the real file is handled here (the native CLI).
        // `ahiru-core`'s job ends at returning the bytes and the destination path on
        // the `Query` (see the `ahiru_core::write` module docs and
        // `ahiru_core::session::Query::copy_result`).
        if let Some(c) = q.copy.take() {
            let n = c.data.len();
            std::fs::write(&c.path, &c.data)?;
            return Ok(Outcome {
                rows: 0,
                copied: Some((c.path.clone(), n)),
                elapsed: start.elapsed(),
            });
        }

        let types: Vec<Ty> = q.schema.iter().map(|f| f.ty).collect();
        out.begin(&q.schema)?;
        let mut rows = 0usize;
        let mut vals: Vec<Value> = Vec::with_capacity(types.len());
        loop {
            match self.s.step(&mut q)? {
                QueryStep::Batch(mut b) => {
                    b.materialize();
                    for r in 0..b.num_rows() {
                        vals.clear();
                        vals.extend(b.cols.iter().map(|c| c.value_at(r)));
                        out.row(&vals)?;
                        rows += 1;
                    }
                }
                QueryStep::NeedIo(_) => return Err("unexpected io request".into()),
                // Codec delegation. ZSTD is now part of `ahiru-core`'s default
                // features, so this is normally not reached (the core decompresses
                // it in-process). What remains is GZIP (deliberately not built into
                // the core, DESIGN.md §6), plus ZSTD when built without the `zstd` feature.
                QueryStep::NeedCodec(reqs) => {
                    for r in reqs {
                        let src = source_bytes(&self.sources, r.table, r.part, r.offset, r.len)?;
                        let decoded = decompress_host(r.codec, src, r.out_len as usize)?;
                        self.s.provide_decoded(r.table, r.part, r.offset, r.len, decoded)?;
                    }
                }
                QueryStep::Done => break,
            }
        }
        out.finish()?;
        Ok(Outcome { rows, copied: None, elapsed: start.elapsed() })
    }

    /// Column names and types of `table`, for `.schema` and `SUMMARIZE`.
    pub fn columns(&mut self, table: &str) -> R<Vec<(String, String, bool)>> {
        self.describe_columns(table)
    }

    /// Column names and types of an arbitrary `FROM` item, via `DESCRIBE`.
    ///
    /// Unlike [`Engine::columns`] this accepts anything the parser accepts
    /// after `FROM` — a table name, a quoted path, `read_csv('...')` — which
    /// is what `SUMMARIZE` needs, since its target is written by the user in
    /// whatever form they like.
    pub fn describe_columns(&mut self, target: &str) -> R<Vec<(String, String, bool)>> {
        self.autoregister(target);
        let mut q = match self.s.prepare(&format!("DESCRIBE {target}"), &[])? {
            Prepared::Ready(q) => q,
            Prepared::NeedIo(_) => return Err("unexpected io request".into()),
        };
        let mut out = Vec::new();
        loop {
            match self.s.step(&mut q)? {
                QueryStep::Batch(mut b) => {
                    b.materialize();
                    for r in 0..b.num_rows() {
                        let cell = |i: usize| match b.cols[i].value_at(r) {
                            Value::Bytes(v) => String::from_utf8_lossy(&v).into_owned(),
                            _ => String::new(),
                        };
                        out.push((cell(0), cell(1), cell(2) == "YES"));
                    }
                }
                QueryStep::NeedIo(_) => return Err("unexpected io request".into()),
                QueryStep::NeedCodec(reqs) => {
                    for r in reqs {
                        let src = source_bytes(&self.sources, r.table, r.part, r.offset, r.len)?;
                        let decoded = decompress_host(r.codec, src, r.out_len as usize)?;
                        self.s.provide_decoded(r.table, r.part, r.offset, r.len, decoded)?;
                    }
                }
                QueryStep::Done => break,
            }
        }
        Ok(out)
    }
}

/// Splits a CLI file argument into `(optional alias, spec)`.
///
/// `name=path` is only used when `name` is a bare identifier *and* the
/// whole argument is not itself an existing file, directory, or matching
/// glob. Otherwise `year=2024/part.parquet` (a Hive-style relative path)
/// would be read as alias `year` and file `2024/part.parquet`.
fn split_alias_spec(arg: &str) -> (Option<&str>, &str) {
    match arg.split_once('=') {
        Some((name, rest)) if is_identifier(name) && !spec_exists(arg) => (Some(name), rest),
        _ => (None, arg),
    }
}

fn spec_exists(spec: &str) -> bool {
    if glob::is_pattern(spec) {
        !glob::expand(spec).is_empty()
    } else {
        Path::new(spec).exists()
    }
}

/// True for a bare SQL identifier, used to tell `name=path` from a path that
/// merely contains `=`.
fn is_identifier(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with(|c: char| c.is_ascii_digit())
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Expands one file spec into the concrete files that make up the table.
fn resolve_spec(spec: &str) -> R<Vec<PathBuf>> {
    let mut out = Vec::new();
    // Arguments joined with `+` are bundled into one logical table (see usage).
    for part in spec.split('+') {
        if part.is_empty() {
            continue;
        }
        if glob::is_pattern(part) {
            let mut m = glob::expand(part);
            m.sort();
            if m.is_empty() {
                return Err(format!("no files matched: {part}").into());
            }
            out.extend(m);
        } else {
            let p = Path::new(part);
            if p.is_dir() {
                let mut m = glob::walk_dir(p);
                m.sort();
                if m.is_empty() {
                    return Err(format!("no supported files under: {part}").into());
                }
                out.extend(m);
            } else {
                out.push(p.to_path_buf());
            }
        }
    }
    Ok(out)
}

/// Single-quoted string literals in `sql`, with `''` escapes resolved.
fn string_literals(sql: &str) -> Vec<String> {
    let b: Vec<char> = sql.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            '-' if i + 1 < b.len() && b[i + 1] == '-' => {
                while i < b.len() && b[i] != '\n' {
                    i += 1;
                }
            }
            '"' => {
                i += 1;
                while i < b.len() && b[i] != '"' {
                    i += 1;
                }
                i += 1;
            }
            '\'' => {
                i += 1;
                let mut s = String::new();
                while i < b.len() {
                    if b[i] == '\'' {
                        if i + 1 < b.len() && b[i + 1] == '\'' {
                            s.push('\'');
                            i += 2;
                            continue;
                        }
                        break;
                    }
                    s.push(b[i]);
                    i += 1;
                }
                i += 1;
                out.push(s);
            }
            _ => i += 1,
        }
    }
    out
}

/// The leading keyword of `sql` if it mutates state, for `-readonly`.
fn write_statement_keyword(sql: &str) -> Option<&'static str> {
    let mut s = sql.trim_start();
    // Skip leading comments so `-- note\nINSERT ...` is still caught.
    loop {
        if let Some(rest) = s.strip_prefix("--") {
            s = rest.split_once('\n').map(|(_, r)| r).unwrap_or("").trim_start();
        } else if let Some(rest) = s.strip_prefix("/*") {
            s = rest.split_once("*/").map(|(_, r)| r).unwrap_or("").trim_start();
        } else {
            break;
        }
    }
    let word: String =
        s.chars().take_while(|c| c.is_ascii_alphabetic()).collect::<String>().to_ascii_uppercase();
    match word.as_str() {
        "INSERT" => Some("INSERT"),
        "UPDATE" => Some("UPDATE"),
        "DELETE" => Some("DELETE"),
        "CREATE" => Some("CREATE"),
        "DROP" => Some("DROP"),
        "ALTER" => Some("ALTER"),
        "COPY" => Some("COPY"),
        _ => None,
    }
}

fn source_bytes(
    sources: &HashMap<(usize, usize), Vec<u8>>,
    table: usize,
    part: usize,
    offset: u64,
    len: u32,
) -> Result<&[u8], String> {
    let b = sources
        .get(&(table, part))
        .ok_or_else(|| String::from("unknown table/part for codec request"))?;
    let s = usize::try_from(offset)
        .map_err(|_| String::from("codec request offset does not fit in usize"))?;
    let e = s
        .checked_add(len as usize)
        .ok_or_else(|| String::from("codec request range overflows usize"))?;
    if e > b.len() {
        return Err(String::from("codec request out of range"));
    }
    Ok(&b[s..e])
}

/// Host-side codecs.
///
/// - ZSTD links `ahiru-zstd` directly. `ahiru-core`'s default build has ZSTD built
///   in (the `zstd` feature), so this is normally not reached. It is the fallback
///   for builds that drop `zstd`, as with `--no-default-features`.
/// - GZIP is handed to the system `gzip`. The CLI is for development, so an external process is fine.
///   On wasm, `DecompressionStream('gzip')` plays the same role.
fn decompress_host(
    codec: ahiru_core::parquet::Compression,
    src: &[u8],
    out_len: usize,
) -> R<Vec<u8>> {
    use ahiru_core::parquet::Compression;
    match codec {
        Compression::Zstd => ahiru_zstd::decompress(src, out_len)
            .map_err(|e| format!("zstd decompression failed: {e:?}").into()),
        Compression::Gzip => {
            let out = gunzip(src)?;
            if out.len() != out_len {
                return Err(format!(
                    "gzip decompressed to {} bytes, expected {out_len}",
                    out.len()
                )
                .into());
            }
            Ok(out)
        }
        other => Err(format!("{other:?} is unsupported on the host side too").into()),
    }
}

fn gunzip(src: &[u8]) -> R<Vec<u8>> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut c = Command::new("gzip")
        .arg("-dc")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    // Feed stdin on a separate thread while `wait_with_output` drains stdout.
    // Writing the complete input first can deadlock when gzip emits more than
    // the OS pipe buffer: the parent blocks on stdin while the child blocks on
    // a full stdout pipe.
    let mut stdin = c.stdin.take().ok_or("no stdin")?;
    let input = src.to_vec();
    let feeder = std::thread::spawn(move || stdin.write_all(&input));
    let out = c.wait_with_output()?;
    feeder.join().map_err(|_| "gzip stdin feeder panicked")??;
    if !out.status.success() {
        return Err("gzip failed".into());
    }
    Ok(out.stdout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_string_literals() {
        assert_eq!(string_literals("SELECT * FROM 'a.parquet'"), vec!["a.parquet"]);
        assert_eq!(string_literals("SELECT 'it''s'"), vec!["it's"]);
        assert_eq!(string_literals("-- 'no'\nSELECT 'yes'"), vec!["yes"]);
        assert_eq!(string_literals("SELECT \"'x'\" FROM t"), Vec::<String>::new());
    }

    #[test]
    fn detects_write_statements() {
        assert_eq!(write_statement_keyword("  insert into t values (1)"), Some("INSERT"));
        assert_eq!(write_statement_keyword("-- c\n COPY t TO 'x'"), Some("COPY"));
        assert_eq!(write_statement_keyword("SELECT 1"), None);
    }

    #[test]
    fn identifier_check_keeps_hive_paths_intact() {
        assert!(is_identifier("sales"));
        assert!(!is_identifier("data/year=2024"));
    }

    #[test]
    fn split_alias_does_not_eat_an_existing_hive_relative_path() {
        let dir = std::env::temp_dir().join(format!("ahiru-hive-{}", std::process::id()));
        let hive = dir.join("year=2024");
        std::fs::create_dir_all(&hive).unwrap();
        let file = hive.join("part.csv");
        std::fs::write(&file, b"id\n1\n").unwrap();
        let cwd = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&dir).expect("chdir");
        let got = split_alias_spec("year=2024/part.csv");
        std::env::set_current_dir(&cwd).expect("restore cwd");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(got, (None, "year=2024/part.csv"));
    }

    #[test]
    fn split_alias_still_accepts_name_eq_path() {
        assert_eq!(split_alias_spec("sales=data.parquet"), (Some("sales"), "data.parquet"));
    }

    #[test]
    fn gzip_rejects_a_length_mismatch() {
        use std::io::Write;
        use std::process::{Command, Stdio};
        let mut c = Command::new("gzip")
            .arg("-c")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("gzip");
        c.stdin.take().expect("stdin").write_all(b"hello").unwrap();
        let out = c.wait_with_output().unwrap();
        assert!(out.status.success());
        assert!(decompress_host(ahiru_core::parquet::Compression::Gzip, &out.stdout, 1).is_err());
        assert_eq!(
            decompress_host(ahiru_core::parquet::Compression::Gzip, &out.stdout, 5).unwrap(),
            b"hello"
        );
    }

    #[test]
    fn source_bytes_rejects_unrepresentable_and_overflowing_ranges() {
        let mut sources = HashMap::new();
        sources.insert((0, 0), vec![1, 2, 3]);

        assert!(source_bytes(&sources, 0, 0, u64::MAX, 1).is_err());
        assert!(source_bytes(&sources, 0, 0, 2, 2).is_err());
        assert_eq!(source_bytes(&sources, 0, 0, 1, 2).unwrap(), &[2, 3]);
    }

    #[test]
    fn gunzip_drains_output_while_feeding_large_input() {
        // A highly-compressible payload expands far beyond a normal pipe
        // buffer. The old write-all-then-wait sequence deadlocked here because
        // gzip blocked on stdout before it could consume all of stdin.
        use std::io::Write;
        use std::process::{Command, Stdio};

        let plain = vec![b'x'; 1024 * 1024];
        let mut c = Command::new("gzip")
            .arg("-c")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("gzip");
        c.stdin.take().expect("stdin").write_all(&plain).unwrap();
        let compressed = c.wait_with_output().unwrap();
        assert!(compressed.status.success());

        let restored = gunzip(&compressed.stdout).expect("large gzip stream should complete");
        assert_eq!(restored, plain);
    }
}
