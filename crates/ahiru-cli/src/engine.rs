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
        let mut s = Session::new();
        // `COPY`, `CREATE TABLE AS` and `INSERT ... SELECT` complete inside
        // `Session::prepare`, so there is no step boundary at which the loop in
        // `run` could answer a `NEED_CODEC`. Handing the core the same
        // decompressor up front lets those statements read a GZIP-compressed
        // Parquet source too, instead of failing (see `Session::set_codec_hook`).
        s.set_codec_hook(core_codec_hook);
        Engine {
            s,
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
        //
        // Every registration goes through the multi-part call, even for a single file: that is the
        // only path that parses `key=value` directories into Hive partition columns. Routing one
        // file around it used to make `t=data/year=2025` (a directory holding exactly one file)
        // expose only the file's own columns, while the sibling `year=2024` -- with two files --
        // exposed `year` and `month`. DuckDB adds the partition columns for a single explicit file
        // path too.
        let mut files = Vec::with_capacity(paths.len());
        for p in &paths {
            files.push((p.to_string_lossy().into_owned(), read_file(p)?));
        }
        let a = self.s.register_multi_bytes(name, files.clone(), FormatKind::Auto)?;
        for (p, (_, bytes)) in files.iter().enumerate() {
            self.sources.insert((a, p), bytes.clone());
        }
        self.remember(name);
        if let [(lit, bytes)] = files.as_slice() {
            // A single file is also reachable under its own path text (`parquet('...')`).
            if lit != name {
                let b = self.s.register_multi_bytes(
                    lit,
                    vec![(lit.clone(), bytes.clone())],
                    FormatKind::Auto,
                )?;
                self.sources.insert((b, 0), bytes.clone());
                self.registered.insert(lit.clone());
            }
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

    /// Registers any file path that appears as a string literal *in table
    /// position* in `sql` and isn't registered yet, so
    /// `SELECT * FROM 'data/x.parquet'` works without naming the file on the
    /// command line first. Paths that don't exist are left alone — the
    /// engine's own "table not found" error is the better message for a typo
    /// than anything we could produce here.
    ///
    /// Only table-position literals are considered (see
    /// [`table_source_literals`]). An ordinary value literal must never reach
    /// the filesystem probe below: `replace(p, '/', '-')` would otherwise hit
    /// the `is_dir` branch and walk the whole root filesystem.
    pub fn autoregister(&mut self, sql: &str) {
        for lit in table_source_literals(sql) {
            if self.registered.contains(&lit) || is_degenerate_path(&lit) {
                continue;
            }
            // Probe the unescaped path: `'star\*.csv'` is not a glob (the
            // metacharacter is escaped), and the file it names on disk is
            // `star*.csv`.
            let probe = literal_path(&lit);
            let looks_like_path = glob::is_pattern(&lit)
                || probe.is_dir()
                || (is_supported_path(&probe) && probe.exists());
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
        // Autoregister the whole statement, not the bare target: a target of
        // `'x.parquet'` only reads as a table reference with the `DESCRIBE`
        // keyword in front of it.
        let sql = format!("DESCRIBE {target}");
        self.autoregister(&sql);
        let mut q = match self.s.prepare(&sql, &[])? {
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
        literal_path(spec).exists()
    }
}

/// True for a bare SQL identifier, used to tell `name=path` from a path that
/// merely contains `=`.
fn is_identifier(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with(|c: char| c.is_ascii_digit())
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Reads a file, naming it in the error. `std::io::Error` carries no path, so
/// a bare `?` here yields "No such file or directory (os error 2)" with no
/// hint about *which* file the user got wrong.
fn read_file(p: &Path) -> R<Vec<u8>> {
    std::fs::read(p)
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{}: {e}", p.display()).into() })
}

/// Resolves one non-glob spec component to a path on disk.
///
/// A spec is glob syntax even when it has no metacharacters left, so `\`
/// escapes a character that should be matched literally: `star\*.csv` names
/// the file `star*.csv`. A path that exists exactly as written still wins, so
/// a name that genuinely contains a backslash keeps resolving.
fn literal_path(part: &str) -> PathBuf {
    let raw = PathBuf::from(part);
    if !part.contains('\\') || raw.exists() {
        return raw;
    }
    let unescaped = PathBuf::from(glob::unescape_literal(part));
    if unescaped.exists() {
        unescaped
    } else {
        raw
    }
}

/// Splits a spec on the `+` that joins several files into one logical table.
///
/// A `+` preceded by a backslash is part of the file name, not a separator, so
/// `a\+b.csv` names the single file `a+b.csv`. And when the parts of a split do
/// not all exist while the whole spec does, the `+` was never a separator at
/// all: a file simply called `a+b.csv` still opens with no escaping needed.
fn split_spec(spec: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let bytes = spec.as_bytes();
    for i in 0..bytes.len() {
        if bytes[i] != b'+' || (i > 0 && bytes[i - 1] == b'\\') {
            continue;
        }
        parts.push(&spec[start..i]);
        start = i + 1;
    }
    parts.push(&spec[start..]);
    if parts.len() > 1 && !parts.iter().all(|p| p.is_empty() || spec_exists(p)) && spec_exists(spec)
    {
        return vec![spec];
    }
    parts
}

/// Expands one file spec into the concrete files that make up the table.
fn resolve_spec(spec: &str) -> R<Vec<PathBuf>> {
    let mut out = Vec::new();
    // Arguments joined with `+` are bundled into one logical table (see usage).
    for part in split_spec(spec) {
        if part.is_empty() {
            continue;
        }
        if glob::is_pattern(part) {
            let mut m = glob::expand(part);
            // A pattern's last component can match a directory (`g/*` where `g`
            // holds a subdirectory). Reading one as a table part fails with
            // "Is a directory" and used to abort the whole registration; the
            // shell and DuckDB both leave directories out of a glob's results.
            // `g/**` still reaches the files inside them, since `**` descends.
            m.retain(|p| !p.is_dir());
            m.sort();
            if m.is_empty() {
                return Err(format!("no files matched: {part}").into());
            }
            out.extend(m);
        } else {
            let p = literal_path(part);
            let p = p.as_path();
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

/// Table functions whose first argument names a file (or a glob, or a list of
/// files). Matched case-insensitively against the identifier in front of `(`.
const READER_FUNCTIONS: &[&str] = &[
    "PARQUET",
    "READ_PARQUET",
    "CSV",
    "READ_CSV",
    "READ_CSV_AUTO",
    "JSON",
    "READ_JSON",
    "READ_JSON_AUTO",
    "JSONL",
    "NDJSON",
    "READ_JSONL",
    "READ_NDJSON",
];

/// Keywords after which a string literal names a table (`FROM 'x.csv'`,
/// `t JOIN 'y.csv'`, `COPY t FROM 'z.csv'`, `DESCRIBE 'x.csv'`).
const FROM_ITEM_STARTERS: &[&str] = &["FROM", "JOIN", "DESCRIBE"];

/// Keywords that end a `FROM` item list, after which a literal is an ordinary
/// value again. `AS` is deliberately absent: it introduces an alias, and the
/// item list may continue past it (`FROM 'a.csv' AS a, 'b.csv'`).
const FROM_ITEM_ENDERS: &[&str] = &[
    "WHERE",
    "GROUP",
    "HAVING",
    "WINDOW",
    "QUALIFY",
    "ORDER",
    "LIMIT",
    "OFFSET",
    "ON",
    "USING",
    "SELECT",
    "VALUES",
    "SET",
    "RETURNING",
    "UNION",
    "INTERSECT",
    "EXCEPT",
    "WITH",
    "INSERT",
    "UPDATE",
    "DELETE",
    "INTO",
    "TO",
];

/// One parenthesis level while scanning for table-position literals.
#[derive(Clone, Copy, Default)]
struct Frame {
    /// A `FROM` / `JOIN` / `DESCRIBE` item list is open at this level.
    from_item: bool,
    /// This level is the argument list of a [`READER_FUNCTIONS`] call, so its
    /// first argument is a path.
    reader_args: bool,
    /// Argument index within the current parenthesis level (comma-separated).
    arg: usize,
    /// This level is a `[...]` list sitting in a path position, so *every*
    /// literal in it is a path (`read_parquet(['a.parquet', 'b.parquet'])`).
    list: bool,
}

impl Frame {
    /// Whether a string literal directly at this level names a table.
    fn is_path_position(&self) -> bool {
        if self.list {
            true
        } else if self.reader_args {
            self.arg == 0
        } else {
            self.from_item
        }
    }
}

/// Single-quoted string literals in `sql` that sit in *table position*, with
/// `''` escapes resolved.
///
/// A literal is in table position when it follows `FROM` / `JOIN` /
/// `DESCRIBE` at the same parenthesis depth, or when it is the first argument
/// of a file-reading table function (`read_csv('x.csv', delim='.')` — the
/// `'.'` is an option value, not a path). Every other literal is an ordinary
/// value: `SELECT '/'`, `replace(p, '/', '-')` and `string_split(x, '.')` must
/// not be mistaken for data sources, or [`Engine::autoregister`] would try to
/// register `/` and walk the entire filesystem.
fn table_source_literals(sql: &str) -> Vec<String> {
    let cs: Vec<char> = sql.chars().collect();
    let mut out = Vec::new();
    let mut stack = vec![Frame::default()];
    let mut prev_word: Option<String> = None;
    let mut i = 0;
    while i < cs.len() {
        match cs[i] {
            '-' if cs.get(i + 1) == Some(&'-') => {
                while i < cs.len() && cs[i] != '\n' {
                    i += 1;
                }
            }
            '/' if cs.get(i + 1) == Some(&'*') => {
                i += 2;
                while i < cs.len() && !(cs[i] == '*' && cs.get(i + 1) == Some(&'/')) {
                    i += 1;
                }
                i = (i + 2).min(cs.len());
            }
            '"' => {
                i += 1;
                while i < cs.len() && cs[i] != '"' {
                    i += 1;
                }
                i += 1;
                prev_word = None;
            }
            '\'' => {
                i += 1;
                let mut s = String::new();
                while i < cs.len() {
                    if cs[i] == '\'' {
                        if cs.get(i + 1) == Some(&'\'') {
                            s.push('\'');
                            i += 2;
                            continue;
                        }
                        break;
                    }
                    s.push(cs[i]);
                    i += 1;
                }
                i += 1;
                if stack.last().copied().unwrap_or_default().is_path_position() {
                    out.push(s);
                }
                prev_word = None;
            }
            '(' => {
                let reader = prev_word.as_deref().is_some_and(|w| READER_FUNCTIONS.contains(&w));
                stack.push(Frame { reader_args: reader, ..Frame::default() });
                prev_word = None;
                i += 1;
            }
            '[' => {
                let list = stack.last().copied().unwrap_or_default().is_path_position();
                stack.push(Frame { list, ..Frame::default() });
                prev_word = None;
                i += 1;
            }
            ')' | ']' => {
                if stack.len() > 1 {
                    stack.pop();
                }
                prev_word = None;
                i += 1;
            }
            ',' => {
                if let Some(f) = stack.last_mut() {
                    f.arg += 1;
                }
                prev_word = None;
                i += 1;
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                i += 1;
                while i < cs.len() && (cs[i].is_ascii_alphanumeric() || cs[i] == '_') {
                    i += 1;
                }
                let w: String = cs[start..i].iter().collect::<String>().to_ascii_uppercase();
                // `IS [NOT] DISTINCT FROM` is a comparison operator, not a
                // FROM clause; its right operand is a value.
                let distinct_from = w == "FROM" && prev_word.as_deref() == Some("DISTINCT");
                if let Some(f) = stack.last_mut() {
                    if !distinct_from && FROM_ITEM_STARTERS.contains(&w.as_str()) {
                        f.from_item = true;
                    } else if FROM_ITEM_ENDERS.contains(&w.as_str()) {
                        f.from_item = false;
                    }
                }
                prev_word = Some(w);
            }
            c => {
                i += 1;
                // Whitespace keeps `prev_word`: `IS DISTINCT FROM` and
                // `read_csv ('x')` both need to see the word before it.
                if !c.is_whitespace() {
                    prev_word = None;
                }
            }
        }
    }
    out
}

/// True for a literal that cannot usefully name a data source: an empty
/// string, or a path built only from separators and `.` / `..` components.
///
/// Such a literal is almost always an ordinary value (`'/'` as a delimiter,
/// `'.'` as a split character). Probing it would resolve to a directory —
/// the filesystem root or the working directory — and walking that is both
/// pointless and, at `/`, effectively unbounded.
fn is_degenerate_path(lit: &str) -> bool {
    lit.is_empty() || lit.split('/').all(|c| c.is_empty() || c == "." || c == "..")
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

/// [`decompress_host`] in the shape `ahiru-core` can call back into.
///
/// `COPY`, `CREATE TABLE AS` and `INSERT ... SELECT` finish inside
/// `Session::prepare`, so the codec loop in [`Engine::run`] never gets a turn
/// on them; the core calls this instead (registered via
/// `Session::set_codec_hook`). The streaming path is unchanged and still
/// answers `NEED_CODEC` in that loop.
///
/// The core is `no_std` and its errors are numeric codes, so the host's
/// descriptive `String` error collapses to `BadCompressedData` (the bytes did
/// not decompress) or `UnsupportedCodec` (this host has no such decoder).
fn core_codec_hook(
    codec: ahiru_core::parquet::Compression,
    src: &[u8],
    out_len: usize,
) -> ahiru_core::error::Result<Vec<u8>> {
    use ahiru_core::error::{Code, Error};
    use ahiru_core::parquet::Compression;
    if !matches!(codec, Compression::Zstd | Compression::Gzip) {
        return Err(Error::new(Code::UnsupportedCodec));
    }
    decompress_host(codec, src, out_len).map_err(|_| Error::new(Code::BadCompressedData))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_table_position_literals() {
        assert_eq!(table_source_literals("SELECT * FROM 'a.parquet'"), vec!["a.parquet"]);
        assert_eq!(
            table_source_literals("SELECT * FROM t JOIN 'b.csv' ON t.a = b.a"),
            vec!["b.csv"]
        );
        assert_eq!(table_source_literals("DESCRIBE 'a.parquet'"), vec!["a.parquet"]);
        assert_eq!(table_source_literals("COPY t FROM 'a.csv'"), vec!["a.csv"]);
        assert_eq!(table_source_literals("SELECT * FROM 'it''s.csv'"), vec!["it's.csv"]);
        assert_eq!(table_source_literals("-- FROM 'no'\nSELECT * FROM 'yes.csv'"), vec!["yes.csv"]);
        assert_eq!(
            table_source_literals("SELECT * FROM 'a.csv' AS a, 'b.csv' AS b"),
            vec!["a.csv", "b.csv"]
        );
    }

    #[test]
    fn ignores_value_literals() {
        // The bug this guards: any of these used to be probed with `is_dir`,
        // and `'/'` sent the CLI walking the whole filesystem.
        for sql in [
            "SELECT '/'",
            "SELECT '.' FROM range(1)",
            "SELECT replace(path, '/', '-') FROM t",
            "SELECT string_split(x, '.') FROM 'a.csv'",
            "SELECT * FROM t WHERE p = '/'",
            "SELECT * FROM t ORDER BY replace(x, '/', '-')",
            "SELECT 1 IS DISTINCT FROM '/'",
            "SELECT \"'/'\" FROM t",
        ] {
            let got = table_source_literals(sql);
            assert!(!got.iter().any(|l| l == "/" || l == "."), "{sql} yielded {got:?}");
        }
    }

    #[test]
    fn reader_function_takes_only_its_first_argument() {
        assert_eq!(table_source_literals("SELECT * FROM read_csv('a.csv')"), vec!["a.csv"]);
        assert_eq!(
            table_source_literals("SELECT * FROM read_csv('a.csv', delim='.')"),
            vec!["a.csv"]
        );
        assert_eq!(
            table_source_literals("SELECT * FROM read_parquet(['a.parquet', 'b.parquet'])"),
            vec!["a.parquet", "b.parquet"]
        );
        // A plain scalar function in a FROM-adjacent position is not a reader.
        assert_eq!(
            table_source_literals("SELECT * FROM t WHERE x = concat('/', 'a')"),
            Vec::<String>::new()
        );
    }

    #[test]
    fn degenerate_paths_are_never_probed() {
        assert!(is_degenerate_path(""));
        assert!(is_degenerate_path("/"));
        assert!(is_degenerate_path("."));
        assert!(is_degenerate_path(".."));
        assert!(is_degenerate_path("//"));
        assert!(is_degenerate_path("./../"));
        assert!(!is_degenerate_path("a.csv"));
        assert!(!is_degenerate_path("./data"));
        assert!(!is_degenerate_path("/tmp"));
    }

    #[test]
    fn literal_path_unescapes_glob_escapes_when_needed() {
        let dir = std::env::temp_dir().join(format!("ahiru-esc-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let star = dir.join("star*.csv");
        std::fs::write(&star, b"v\n1\n").unwrap();

        let spec = format!("{}/star\\*.csv", dir.display());
        assert_eq!(literal_path(&spec), star);
        // A path without a backslash is passed through untouched.
        let plain = format!("{}/plain.csv", dir.display());
        assert_eq!(literal_path(&plain), PathBuf::from(&plain));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_error_names_the_file() {
        let msg = read_file(Path::new("no/such/file.parquet")).unwrap_err().to_string();
        assert!(msg.contains("no/such/file.parquet"), "{msg}");
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
