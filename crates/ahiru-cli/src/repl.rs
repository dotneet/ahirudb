//! The shell: statement splitting, dot commands, and the interactive loop.
//!
//! One `Shell` owns the engine, the current output settings and the current
//! output sink. Everything the CLI can do after startup goes through it, so a
//! `-c` one-liner, a `-f` script, a piped stdin and the interactive prompt all
//! share exactly one code path (`run_script`).

use std::fs::File;
use std::io::{self, Write};

use crate::args::Options;
use crate::engine::{Engine, R};
use crate::glob;
use crate::lineedit::{Editor, Line};
use crate::output::{Mode, Settings, Writer};
use crate::summarize;

/// Where results go: `.output` / `.once`, or standard output.
enum Sink {
    Stdout(io::Stdout),
    File(File),
}

impl Sink {
    fn as_write(&mut self) -> &mut dyn Write {
        match self {
            Sink::Stdout(s) => s,
            Sink::File(f) => f,
        }
    }

    fn flush(&mut self) {
        let _ = self.as_write().flush();
    }
}

pub struct Shell {
    pub engine: Engine,
    settings: Settings,
    /// True once the user pinned a mode explicitly, so the interactive loop
    /// leaves it alone instead of switching to `duckbox`.
    mode_set: bool,
    timer: bool,
    echo: bool,
    bail: bool,
    failed: bool,
    out: Sink,
    /// `.once FILE`: replaces `out` for exactly one statement.
    once: Option<Sink>,
    /// Settings forced for the `.once` statement only (used by `.excel`,
    /// which always writes CSV whatever the current mode is).
    once_settings: Option<Settings>,
    /// `.excel`: the temporary file to hand to the system opener once the
    /// next statement has been written to it.
    excel: Option<std::path::PathBuf>,
    /// `.log FILE`: diagnostics (errors, row counts, timings) go here.
    log: Option<File>,
    prompt: String,
    prompt_cont: String,
    excel_calls: usize,
    /// Set once `-separator` or `.separator` has been used, after which
    /// `.mode` stops imposing the mode's own default separator.
    sep_explicit: bool,
}

impl Shell {
    pub fn new(opts: &Options) -> Shell {
        let mut engine = Engine::new();
        engine.readonly = opts.readonly;
        Shell {
            engine,
            settings: opts.settings.clone(),
            mode_set: opts.mode_set,
            timer: opts.timer,
            echo: opts.echo,
            bail: opts.bail,
            failed: false,
            out: Sink::Stdout(io::stdout()),
            once: None,
            once_settings: None,
            excel: None,
            log: None,
            prompt: "ahiru> ".to_string(),
            prompt_cont: "   ...> ".to_string(),
            excel_calls: 0,
            sep_explicit: opts.sep_explicit,
        }
    }

    /// Writes a diagnostic (row counts, timings, errors) to the log target.
    fn diag(&mut self, msg: &str) {
        match &mut self.log {
            Some(f) => {
                let _ = writeln!(f, "{msg}");
            }
            None => eprintln!("{msg}"),
        }
    }

    /// Writes dot-command output to the current result sink, so `.output`
    /// and `.once` capture it the same way they capture query results.
    fn emit(&mut self, text: &str) {
        let sink = match self.once {
            Some(ref mut s) => s,
            None => &mut self.out,
        };
        let w = sink.as_write();
        let _ = w.write_all(text.as_bytes());
        let _ = w.flush();
    }

    /// Runs one SQL statement (no dot-command handling).
    pub fn run_statement(&mut self, sql: &str) -> R<()> {
        let trimmed = sql.trim();
        if trimmed.is_empty() {
            return Ok(());
        }
        if self.echo {
            let msg = trimmed.to_string();
            self.diag(&msg);
        }
        // `SUMMARIZE t` is rewritten here rather than in the engine: every
        // aggregate it needs exists already, and the wasm build pays nothing.
        let rewritten = match summarize::parse(trimmed) {
            Some(target) => {
                let cols = self.engine.describe_columns(&target).map_err(|e| {
                    if target.starts_with('(') {
                        // `DESCRIBE` only takes a `FROM` item, so a subquery
                        // target cannot be described. Say so, rather than
                        // passing the engine's generic message through.
                        "SUMMARIZE does not accept a subquery; \
                         use a table name, a path, or a read_*() call"
                            .to_string()
                    } else {
                        format!("SUMMARIZE {target}: {e}")
                    }
                })?;
                Some(summarize::summarize_sql(&target, &cols))
            }
            None => None,
        };
        let sql = rewritten.as_deref().unwrap_or(trimmed);
        // `SELECT 1 + 1` with no `FROM` is the most common thing anyone types
        // at a SQL prompt, and this engine requires a `FROM` on every `SELECT`
        // (docs/sql/queries.md). Supply the one-row table it wants. `LIMIT` /
        // `OFFSET` / `ORDER BY` still need the dummy FROM *before* those clauses.
        let dummied;
        let sql = if let Some(rewritten) = dummy_from(sql) {
            dummied = rewritten;
            dummied.as_str()
        } else {
            sql
        };

        let settings = self.once_settings.clone().unwrap_or_else(|| self.settings.clone());
        let mode = settings.mode;
        let sink = match self.once {
            Some(ref mut s) => s,
            None => &mut self.out,
        };
        let result = {
            let mut w = Writer::new(sink.as_write(), settings);
            self.engine.run(sql, &mut w)
        };
        if let Some(mut s) = self.once.take() {
            s.flush();
            self.once_settings = None;
            if let Some(p) = self.excel.take() {
                // Nothing useful was written if the statement failed; opening
                // an empty spreadsheet would just be noise.
                if result.is_ok() {
                    open_externally(&p);
                } else {
                    let _ = std::fs::remove_file(&p);
                }
            }
        } else {
            self.out.flush();
        }
        let outcome = result?;
        if let Some((path, bytes)) = &outcome.copied {
            let msg = format!("(copied {bytes} bytes to {path})");
            self.diag(&msg);
        } else if mode != Mode::Duckbox && mode != Mode::Trash {
            // Duckbox prints its own footer; everything else keeps the row
            // count the CLI has always printed to stderr.
            let msg = format!("({} rows)", outcome.rows);
            self.diag(&msg);
        }
        if self.timer {
            let msg = format!("Run Time (s): real {:.3}", outcome.elapsed.as_secs_f64());
            self.diag(&msg);
        }
        Ok(())
    }

    /// Runs a script: several statements, `.dot` commands, comments.
    pub fn run_script(&mut self, text: &str) -> R<()> {
        for piece in split_script(text) {
            let r = match piece {
                Piece::Dot(line) => self.run_dot(&line),
                Piece::Sql(sql) => self.run_statement(&sql),
            };
            if let Err(e) = r {
                let msg = format!("error: {e}");
                self.diag(&msg);
                self.failed = true;
                if self.bail {
                    return Err("stopped at the first error (-bail)".into());
                }
            }
        }
        Ok(())
    }

    /// Runs a final statement left in the buffer with no trailing `;` when
    /// input ends (Ctrl-D on the interactive prompt, or EOF partway through a
    /// piped script). Mirrors the per-statement error handling in
    /// `interactive`'s main loop, so a failing final statement is diagnosed
    /// and still marks the session as failed -- previously the result of
    /// this last `run_script` call was discarded, so the process could exit
    /// 0 even though the last statement errored.
    fn run_trailing(&mut self, script: &str) {
        if let Err(e) = self.run_script(script) {
            let msg = format!("error: {e}");
            self.diag(&msg);
            self.failed = true;
        }
    }

    /// The interactive loop.
    pub fn interactive(&mut self) -> R<()> {
        if !self.mode_set {
            self.settings.mode = Mode::Duckbox;
        }
        if self.settings.maxwidth == 0 {
            self.settings.maxwidth = terminal_width();
        }
        let mut ed = Editor::new(crate::args::history_path());
        let mut buf = String::new();
        if ed.is_interactive() {
            println!("ahirudb — type .help for dot commands, .exit to quit");
        }
        loop {
            let prompt =
                if buf.trim().is_empty() { self.prompt.clone() } else { self.prompt_cont.clone() };
            let line = {
                let names = self.engine.table_names();
                let complete = |word: &str| complete_word(word, &names);
                ed.read_line(&prompt, &complete)
            };
            match line {
                Line::Eof => break,
                Line::Interrupted => {
                    buf.clear();
                    continue;
                }
                Line::Text(t) => {
                    if buf.trim().is_empty() && t.trim_start().starts_with('.') {
                        ed.add_history(t.trim());
                        buf.clear();
                        if let Err(e) = self.run_dot(t.trim()) {
                            let msg = format!("error: {e}");
                            self.diag(&msg);
                            self.failed = true;
                        }
                        continue;
                    }
                    buf.push_str(&t);
                    buf.push('\n');
                    if !ends_statement(&buf) {
                        continue;
                    }
                    let script = std::mem::take(&mut buf);
                    ed.add_history(script.trim());
                    if let Err(e) = self.run_script(&script) {
                        let msg = format!("error: {e}");
                        self.diag(&msg);
                        self.failed = true;
                        if self.bail {
                            break;
                        }
                    }
                }
            }
        }
        // A trailing statement without a terminating `;`, then Ctrl-D.
        if !buf.trim().is_empty() {
            let script = std::mem::take(&mut buf);
            self.run_trailing(&script);
        }
        ed.save_history();
        Ok(())
    }

    /// Final status: `Err` if any statement in a script failed.
    pub fn into_result(mut self) -> R<()> {
        self.out.flush();
        if self.failed {
            Err("one or more statements failed".into())
        } else {
            Ok(())
        }
    }

    /// A counter so two `.excel` invocations in one session don't collide.
    fn excel_seq(&mut self) -> usize {
        self.excel_calls += 1;
        self.excel_calls
    }

    /// The `.show` report.
    fn settings_text(&self) -> String {
        let s = &self.settings;
        let on = |b: bool| if b { "on" } else { "off" };
        let mut t = String::new();
        t.push_str(&format!("      mode: {}\n", s.mode.name()));
        t.push_str(&format!("   headers: {}\n", on(s.header)));
        t.push_str(&format!(" nullvalue: {:?}\n", s.null));
        t.push_str(&format!(" separator: {:?}\n", s.separator));
        t.push_str(&format!("   maxrows: {}\n", s.maxrows));
        t.push_str(&format!("  maxwidth: {}\n", s.maxwidth));
        t.push_str(&format!("     timer: {}\n", on(self.timer)));
        t.push_str(&format!("      echo: {}\n", on(self.echo)));
        t.push_str(&format!("      bail: {}\n", on(self.bail)));
        t.push_str(&format!("  readonly: {}\n", on(self.engine.readonly)));
        t.push_str(&format!("    tables: {}\n", self.engine.table_names().join(" ")));
        t
    }

    // ---- dot commands -----------------------------------------------------

    fn run_dot(&mut self, line: &str) -> R<()> {
        let args = tokenize(line);
        let Some(cmd) = args.first() else { return Ok(()) };
        let cmd = cmd.trim_start_matches('.');
        let rest = &args[1..];
        let arg = |i: usize| rest.get(i).map(String::as_str);
        match cmd {
            "exit" | "quit" => {
                let code = arg(0).and_then(|s| s.parse::<u8>().ok()).unwrap_or(0);
                self.out.flush();
                if let Some(mut s) = self.once.take() {
                    s.flush();
                }
                std::process::exit(code as i32);
            }
            "help" => {
                let text = help_text(arg(0));
                self.emit(&text);
                Ok(())
            }
            "mode" => match arg(0) {
                None => {
                    let m = self.settings.mode.name().to_string();
                    self.emit(&format!("current output mode: {m}\n"));
                    Ok(())
                }
                Some(m) => {
                    let mode = Mode::parse(m).ok_or_else(|| format!("unknown output mode: {m}"))?;
                    self.settings.mode = mode;
                    self.mode_set = true;
                    if !self.sep_explicit {
                        self.settings.separator = mode.default_separator().to_string();
                    }
                    if let Some(t) = arg(1) {
                        self.settings.insert_table = t.to_string();
                    }
                    Ok(())
                }
            },
            "headers" => {
                self.settings.header = on_off(arg(0), self.settings.header)?;
                Ok(())
            }
            "nullvalue" => {
                self.settings.null = arg(0).unwrap_or("").to_string();
                Ok(())
            }
            "separator" => {
                self.settings.separator = unescape(arg(0).ok_or(".separator needs a separator")?);
                self.sep_explicit = true;
                Ok(())
            }
            "maxrows" => {
                self.settings.maxrows = parse_usize(arg(0), ".maxrows")?;
                Ok(())
            }
            "maxwidth" => {
                self.settings.maxwidth = parse_usize(arg(0), ".maxwidth")?;
                Ok(())
            }
            "timer" => {
                self.timer = on_off(arg(0), self.timer)?;
                Ok(())
            }
            "echo" => {
                self.echo = on_off(arg(0), self.echo)?;
                Ok(())
            }
            "bail" => {
                self.bail = on_off(arg(0), self.bail)?;
                Ok(())
            }
            "output" => {
                self.out.flush();
                self.out = match arg(0) {
                    None | Some("stdout") => Sink::Stdout(io::stdout()),
                    Some(p) => Sink::File(File::create(p).map_err(|e| format!("{p}: {e}"))?),
                };
                Ok(())
            }
            "once" => {
                let p = arg(0).ok_or(".once needs a file name")?;
                self.once = Some(Sink::File(File::create(p).map_err(|e| format!("{p}: {e}"))?));
                Ok(())
            }
            "excel" => {
                let p = std::env::temp_dir().join(format!(
                    "ahiru_{}_{}.csv",
                    std::process::id(),
                    self.excel_seq()
                ));
                self.once = Some(Sink::File(
                    File::create(&p).map_err(|e| format!("{}: {e}", p.display()))?,
                ));
                self.once_settings = Some(Settings {
                    mode: Mode::Csv,
                    separator: ",".to_string(),
                    header: true,
                    ..self.settings.clone()
                });
                self.excel = Some(p);
                Ok(())
            }
            "log" => {
                self.log = match arg(0) {
                    None | Some("off") | Some("stderr") => None,
                    Some(p) => Some(File::create(p).map_err(|e| format!("{p}: {e}"))?),
                };
                Ok(())
            }
            "read" => {
                let p = arg(0).ok_or(".read needs a file name")?;
                let text = std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?;
                self.run_script(&text)
            }
            "tables" => {
                let pat = arg(0);
                let mut text = String::new();
                for n in self.engine.table_names() {
                    if pat.map(|p| glob::matches(p, &n)).unwrap_or(true) {
                        text.push_str(&n);
                        text.push('\n');
                    }
                }
                self.emit(&text);
                Ok(())
            }
            "schema" => {
                let pat = arg(0);
                for n in self.engine.table_names() {
                    if !pat.map(|p| glob::matches(p, &n)).unwrap_or(true) {
                        continue;
                    }
                    match self.engine.columns(&n) {
                        Ok(cols) => {
                            let text = create_table_text(&n, &cols);
                            self.emit(&text);
                        }
                        Err(e) => {
                            let msg = format!("-- {n}: {e}");
                            self.diag(&msg);
                        }
                    }
                }
                Ok(())
            }
            "databases" => {
                // There is no persistent database to attach: every table is a
                // file registered at startup or with `.import`.
                let n = self.engine.table_names().len();
                self.emit(&format!("main: (in-memory; {n} table(s) registered)\n"));
                Ok(())
            }
            "import" => {
                let file = arg(0).ok_or(".import needs a file and a table name")?;
                let table = arg(1).ok_or(".import needs a table name")?;
                self.engine.register_as(table, file)
            }
            "open" => Err("no persistent database in ahirudb: register a file instead \
                 (`.import FILE NAME`, or pass files on the command line)"
                .into()),
            "cd" => {
                let d = arg(0).ok_or(".cd needs a directory")?;
                std::env::set_current_dir(d).map_err(|e| format!("{d}: {e}"))?;
                Ok(())
            }
            "print" => {
                let text = format!("{}\n", rest.join(" "));
                self.emit(&text);
                Ok(())
            }
            "prompt" => {
                if let Some(p) = arg(0) {
                    self.prompt = p.to_string();
                }
                if let Some(p) = arg(1) {
                    self.prompt_cont = p.to_string();
                }
                Ok(())
            }
            "shell" | "system" => {
                if rest.is_empty() {
                    return Err(".shell needs a command".into());
                }
                let joined = line.trim_start().trim_start_matches('.');
                let joined = joined.split_once(char::is_whitespace).map(|(_, r)| r).unwrap_or("");
                let st = std::process::Command::new("sh").arg("-c").arg(joined).status()?;
                if !st.success() {
                    return Err(format!("command failed: {joined}").into());
                }
                Ok(())
            }
            "show" => {
                let text = self.settings_text();
                self.emit(&text);
                Ok(())
            }
            other => Err(format!("unknown command: .{other} (try .help)").into()),
        }
    }
}

fn create_table_text(name: &str, cols: &[(String, String, bool)]) -> String {
    let mut s = format!("CREATE TABLE {name}(\n");
    for (i, (n, ty, nullable)) in cols.iter().enumerate() {
        s.push_str(&format!(
            "  \"{}\" {}{}{}\n",
            n.replace('"', "\"\""),
            ty,
            if *nullable { "" } else { " NOT NULL" },
            if i + 1 == cols.len() { "" } else { "," }
        ));
    }
    s.push_str(");\n");
    s
}

fn parse_usize(v: Option<&str>, what: &str) -> Result<usize, String> {
    let v = v.ok_or_else(|| format!("{what} needs a number"))?;
    v.parse().map_err(|_| format!("{what}: not a number: {v}"))
}

fn on_off(v: Option<&str>, current: bool) -> Result<bool, String> {
    match v {
        None => Ok(!current),
        Some(s) => match s.to_ascii_lowercase().as_str() {
            "on" | "true" | "yes" | "1" => Ok(true),
            "off" | "false" | "no" | "0" => Ok(false),
            other => Err(format!("expected on or off, got: {other}")),
        },
    }
}

/// `\t`, `\n`, `\\` in a `.separator` argument.
fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('t') => out.push('\t'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('\\') => out.push('\\'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

/// Splits a dot-command line into words, honouring quotes so that
/// `.output 'my results.csv'` keeps its space.
fn tokenize(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    let mut any = false;
    for c in line.chars() {
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => cur.push(c),
            None if c == '\'' || c == '"' => {
                quote = Some(c);
                any = true;
            }
            None if c.is_whitespace() => {
                if !cur.is_empty() || any {
                    out.push(std::mem::take(&mut cur));
                    any = false;
                }
            }
            None => cur.push(c),
        }
    }
    if !cur.is_empty() || any {
        out.push(cur);
    }
    out
}

/// Completion candidates for the word under the cursor.
fn complete_word(word: &str, tables: &[String]) -> Vec<String> {
    const KEYWORDS: &[&str] = &[
        "SELECT",
        "FROM",
        "WHERE",
        "GROUP BY",
        "ORDER BY",
        "LIMIT",
        "HAVING",
        "JOIN",
        "LEFT JOIN",
        "INNER JOIN",
        "UNION ALL",
        "WITH",
        "DISTINCT",
        "COUNT",
        "SUM",
        "AVG",
        "MIN",
        "MAX",
        "SUMMARIZE",
        "DESCRIBE",
        "EXPLAIN",
        "SHOW TABLES",
        "PIVOT",
        "UNPIVOT",
        "QUALIFY",
    ];
    const DOTS: &[&str] = &[
        ".bail",
        ".cd",
        ".databases",
        ".echo",
        ".exit",
        ".headers",
        ".help",
        ".import",
        ".log",
        ".maxrows",
        ".maxwidth",
        ".mode",
        ".nullvalue",
        ".once",
        ".output",
        ".print",
        ".prompt",
        ".quit",
        ".read",
        ".schema",
        ".separator",
        ".shell",
        ".show",
        ".system",
        ".tables",
        ".timer",
    ];
    if word.starts_with('.') {
        return DOTS.iter().filter(|d| d.starts_with(word)).map(|d| d.to_string()).collect();
    }
    let upper = word.to_ascii_uppercase();
    let mut out: Vec<String> = tables.iter().filter(|t| t.starts_with(word)).cloned().collect();
    out.extend(
        KEYWORDS
            .iter()
            .filter(|k| k.starts_with(&upper) && !upper.is_empty())
            .map(|k| k.to_string()),
    );
    out
}

/// One unit of a script.
enum Piece {
    Dot(String),
    Sql(String),
}

/// Splits a script into dot commands and SQL statements.
///
/// A dot command is a line whose first non-blank character is `.` *and* which
/// starts at a statement boundary — inside an unterminated statement, a
/// leading `.` is just SQL (a decimal point at the start of a continuation
/// line, say).
fn split_script(text: &str) -> Vec<Piece> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for line in text.lines() {
        if cur.trim().is_empty() && line.trim_start().starts_with('.') {
            cur.clear();
            out.push(Piece::Dot(line.trim().to_string()));
            continue;
        }
        cur.push_str(line);
        cur.push('\n');
        // A line may hold several statements: peel them off one `;` at a time.
        while let Some(end) = statement_end(&cur) {
            let stmt: String = cur.chars().take(end).collect();
            let rest: String = cur.chars().skip(end + 1).collect();
            if !stmt.trim().is_empty() {
                out.push(Piece::Sql(stmt));
            }
            cur = rest;
        }
    }
    if !cur.trim().is_empty() {
        out.push(Piece::Sql(cur));
    }
    out
}

/// True for a `SELECT` that needs a `FROM range(1)` inserted.
///
/// This engine requires a `FROM` clause on every `SELECT`
/// (docs/sql/queries.md), which a shell user does not expect when typing
/// `SELECT 1 + 1`. The rewrite fires for a statement that starts with `SELECT`
/// and whose top level has no `FROM` and no set operation. Trailing clauses
/// (`WHERE`, `ORDER BY`, `LIMIT`, …) stay after the inserted `FROM range(1)`.
fn needs_dummy_from(sql: &str) -> bool {
    const BLOCKERS: &[&str] = &["FROM", "UNION", "INTERSECT", "EXCEPT"];
    let words = top_level_words(sql);
    match words.first() {
        Some((_, w)) if w == "SELECT" => {}
        _ => return false,
    }
    !words.iter().any(|(_, w)| BLOCKERS.contains(&w.as_str()))
}

/// Rewrites a clauseless `SELECT` so the engine's required `FROM` is present.
/// Trailing clauses (`WHERE` / `GROUP BY` / `ORDER BY` / `LIMIT` / …) stay
/// after the inserted `FROM range(1)`. `AS order` / `AS limit` are aliases,
/// not clauses, and must not be treated as insert points.
fn dummy_from(sql: &str) -> Option<String> {
    if !needs_dummy_from(sql) {
        return None;
    }
    let words = top_level_words(sql);
    let insert_at =
        words.iter().find_map(|&(start, ref w)| clause_starts_at(sql, start, w).then_some(start));
    Some(match insert_at {
        Some(i) => format!("{} FROM range(1) {}", sql[..i].trim_end(), &sql[i..]),
        None => format!("{sql} FROM range(1)"),
    })
}

/// Whether the top-level word at `start` begins a SELECT tail clause.
fn clause_starts_at(sql: &str, start: usize, word: &str) -> bool {
    match word {
        "WHERE" | "GROUP" | "HAVING" | "WINDOW" | "QUALIFY" | "USING" => true,
        "ORDER" => next_word_is(sql, start + word.len(), "BY"),
        "LIMIT" | "OFFSET" => looks_like_limit_arg(sql, start + word.len()),
        _ => false,
    }
}

fn next_word_is(sql: &str, from: usize, want: &str) -> bool {
    let rest = sql[from.min(sql.len())..].trim_start();
    rest.len() >= want.len()
        && rest[..want.len()].eq_ignore_ascii_case(want)
        && rest[want.len()..].starts_with(|c: char| !c.is_ascii_alphanumeric() && c != '_')
}

fn looks_like_limit_arg(sql: &str, from: usize) -> bool {
    let rest = sql[from.min(sql.len())..].trim_start();
    rest.starts_with(|c: char| c.is_ascii_digit())
        || rest.starts_with('?')
        || next_word_is(sql, from, "ALL")
}

/// Upper-cased bare words of `sql` that sit at parenthesis depth 0, outside
/// string literals, quoted identifiers and comments.
fn top_level_words(sql: &str) -> Vec<(usize, String)> {
    let cs: Vec<(usize, char)> = sql.char_indices().collect();
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut i = 0;
    while i < cs.len() {
        match cs[i].1 {
            '-' if cs.get(i + 1).map(|c| c.1) == Some('-') => {
                while i < cs.len() && cs[i].1 != '\n' {
                    i += 1;
                }
            }
            '/' if cs.get(i + 1).map(|c| c.1) == Some('*') => {
                i += 2;
                while i < cs.len() && !(cs[i].1 == '*' && cs.get(i + 1).map(|c| c.1) == Some('/')) {
                    i += 1;
                }
                i += 2;
            }
            '\'' | '"' => {
                let q = cs[i].1;
                i += 1;
                while i < cs.len() {
                    if cs[i].1 == q {
                        if cs.get(i + 1).map(|c| c.1) == Some(q) {
                            i += 2;
                            continue;
                        }
                        break;
                    }
                    i += 1;
                }
                i += 1;
            }
            '(' => {
                depth += 1;
                i += 1;
            }
            ')' => {
                depth -= 1;
                i += 1;
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start_byte = cs[i].0;
                i += 1;
                while i < cs.len() && (cs[i].1.is_ascii_alphanumeric() || cs[i].1 == '_') {
                    i += 1;
                }
                if depth == 0 {
                    let end_byte = if i < cs.len() { cs[i].0 } else { sql.len() };
                    out.push((start_byte, sql[start_byte..end_byte].to_ascii_uppercase()));
                }
            }
            _ => i += 1,
        }
    }
    out
}

/// True when `text` contains a statement-terminating `;`.
fn ends_statement(text: &str) -> bool {
    statement_end(text).is_some()
}

/// Character index of the first `;` that is not inside a string literal,
/// quoted identifier or comment.
fn statement_end(text: &str) -> Option<usize> {
    let cs: Vec<char> = text.chars().collect();
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
                i += 2;
            }
            '\'' | '"' => {
                let q = cs[i];
                i += 1;
                while i < cs.len() {
                    if cs[i] == q {
                        if cs.get(i + 1) == Some(&q) {
                            i += 2;
                            continue;
                        }
                        break;
                    }
                    i += 1;
                }
                i += 1;
            }
            ';' => return Some(i),
            _ => i += 1,
        }
    }
    None
}

/// Hands `path` to the system's file opener (`.excel`). Best effort: a
/// desktop-less machine simply has neither command, and the file is still
/// written and reported.
fn open_externally(path: &std::path::Path) {
    let opener = if cfg!(target_os = "macos") { "open" } else { "xdg-open" };
    let r = std::process::Command::new(opener).arg(path).status();
    match r {
        Ok(st) if st.success() => {}
        _ => eprintln!("(wrote {} — could not launch `{opener}`)", path.display()),
    }
}

/// Terminal width from `stty size`, or 0 when it can't be determined.
fn terminal_width() -> usize {
    if let Ok(v) = std::env::var("COLUMNS") {
        if let Ok(n) = v.trim().parse::<usize>() {
            return n;
        }
    }
    let out = std::process::Command::new("stty")
        .arg("size")
        .stdin(std::process::Stdio::inherit())
        .output();
    let Ok(out) = out else { return 0 };
    let text = String::from_utf8_lossy(&out.stdout);
    text.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0)
}

fn help_text(pattern: Option<&str>) -> String {
    const HELP: &[(&str, &str)] = &[
        (".bail on|off", "Stop after hitting an error"),
        (".cd DIR", "Change the working directory"),
        (".databases", "Show what is registered (there is no persistent database)"),
        (".echo on|off", "Print each statement before running it"),
        (".excel", "Write the next result to a temporary CSV and open it"),
        (".exit [CODE]", "Exit with CODE (default 0)"),
        (".headers on|off", "Toggle the header row"),
        (".help [PATTERN]", "Show this text"),
        (".import FILE TABLE", "Register FILE as TABLE"),
        (".log FILE|off", "Send diagnostics to FILE instead of stderr"),
        (".maxrows N", "Rows shown in boxed modes before eliding (0 = all)"),
        (".maxwidth N", "Table width in boxed modes before eliding columns"),
        (
            ".mode MODE [TABLE]",
            "duckbox box csv tsv json jsonlines line list markdown insert trash",
        ),
        (".nullvalue STR", "Text printed for NULL"),
        (".once FILE", "Send the next result to FILE"),
        (".output [FILE]", "Send results to FILE, or back to stdout"),
        (".print STRING...", "Print STRING"),
        (".prompt MAIN [CONT]", "Change the prompts"),
        (".quit", "Exit"),
        (".read FILE", "Run the SQL script in FILE"),
        (".schema [PATTERN]", "Show CREATE TABLE text for matching tables"),
        (".separator SEP", "Column separator for csv/tsv/list modes"),
        (".shell CMD...", "Run CMD in a system shell"),
        (".show", "Show current settings"),
        (".system CMD...", "Same as .shell"),
        (".tables [PATTERN]", "List registered tables"),
        (".timer on|off", "Print query wall time"),
    ];
    let mut s = String::new();
    for (cmd, desc) in HELP {
        if let Some(p) = pattern {
            if !cmd.contains(p.trim_start_matches('.')) {
                continue;
            }
        }
        s.push_str(&format!("{cmd:<22} {desc}\n"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(text: &str) -> Vec<String> {
        split_script(text)
            .into_iter()
            .map(|p| match p {
                Piece::Dot(d) => format!("dot:{d}"),
                Piece::Sql(s) => format!("sql:{}", s.trim()),
            })
            .collect()
    }

    #[test]
    fn splits_statements_and_dot_commands() {
        assert_eq!(
            pieces(".mode csv\nSELECT 1; SELECT 2;\nSELECT 3"),
            vec!["dot:.mode csv", "sql:SELECT 1", "sql:SELECT 2", "sql:SELECT 3"]
        );
    }

    #[test]
    fn semicolons_inside_literals_do_not_split() {
        assert_eq!(pieces("SELECT ';' AS x;"), vec!["sql:SELECT ';' AS x"]);
        assert_eq!(pieces("SELECT 'it''s; ok';"), vec!["sql:SELECT 'it''s; ok'"]);
        assert_eq!(pieces("SELECT 1 -- ;\n;"), vec!["sql:SELECT 1 -- ;"]);
        assert_eq!(pieces("/* ; */ SELECT 1;"), vec!["sql:/* ; */ SELECT 1"]);
    }

    #[test]
    fn multiline_statement_is_one_piece() {
        assert_eq!(pieces("SELECT 1,\n  2\nFROM t;"), vec!["sql:SELECT 1,\n  2\nFROM t"]);
    }

    #[test]
    fn tokenizer_keeps_quoted_arguments_whole() {
        assert_eq!(tokenize(".output 'my file.csv'"), vec![".output", "my file.csv"]);
        assert_eq!(tokenize(".mode  csv"), vec![".mode", "csv"]);
        assert_eq!(tokenize(".nullvalue ''"), vec![".nullvalue", ""]);
    }

    #[test]
    fn separator_escapes() {
        assert_eq!(unescape("\\t"), "\t");
        assert_eq!(unescape("|"), "|");
        assert_eq!(unescape("\\\\"), "\\");
    }

    #[test]
    fn on_off_toggles_when_no_argument() {
        assert!(on_off(None, false).unwrap());
        assert!(!on_off(Some("off"), true).unwrap());
        assert!(on_off(Some("ON"), false).unwrap());
        assert!(on_off(Some("maybe"), false).is_err());
    }

    #[test]
    fn dummy_from_only_for_clauseless_selects() {
        assert!(needs_dummy_from("SELECT 1 + 1"));
        assert!(needs_dummy_from("select upper('a'), now()"));
        // A nested FROM is fine — the outer SELECT still has none.
        assert!(needs_dummy_from("SELECT (SELECT max(x) FROM t)"));
        assert!(!needs_dummy_from("SELECT * FROM t"));
        assert!(!needs_dummy_from("SELECT 1 UNION SELECT 2"));
        assert!(!needs_dummy_from("DESCRIBE t"));
        // A `from` inside a literal is not a clause keyword.
        assert!(needs_dummy_from("SELECT 'from'"));
        // Trailing LIMIT/ORDER still need the dummy FROM, inserted before them.
        assert!(needs_dummy_from("SELECT 1 LIMIT 5"));
        assert!(needs_dummy_from("SELECT 1 AS x ORDER BY x"));
        assert_eq!(
            dummy_from("SELECT 1 LIMIT 5").as_deref(),
            Some("SELECT 1 FROM range(1) LIMIT 5")
        );
        assert_eq!(
            dummy_from("SELECT 1 AS x ORDER BY x").as_deref(),
            Some("SELECT 1 AS x FROM range(1) ORDER BY x")
        );
        assert_eq!(dummy_from("SELECT 1 + 1").as_deref(), Some("SELECT 1 + 1 FROM range(1)"));
        // `AS order` / `AS limit` are aliases, not ORDER BY / LIMIT clauses.
        assert_eq!(
            dummy_from("SELECT 1 AS order").as_deref(),
            Some("SELECT 1 AS order FROM range(1)")
        );
        assert_eq!(
            dummy_from("SELECT 1 AS limit").as_deref(),
            Some("SELECT 1 AS limit FROM range(1)")
        );
        assert_eq!(
            dummy_from("SELECT 1 WHERE false").as_deref(),
            Some("SELECT 1 FROM range(1) WHERE false")
        );
    }

    #[test]
    fn completion_offers_dots_tables_and_keywords() {
        let tables = vec!["trips".to_string()];
        assert!(complete_word(".ta", &tables).contains(&".tables".to_string()));
        assert!(complete_word("tr", &tables).contains(&"trips".to_string()));
        assert!(complete_word("sel", &tables).contains(&"SELECT".to_string()));
    }

    #[test]
    fn trailing_statement_without_semicolon_at_eof_marks_failure() {
        let opts = Options::default();
        let mut sh = Shell::new(&opts);
        // Mirrors what `interactive()` does when input ends mid-statement
        // (Ctrl-D at the prompt, or EOF partway through a piped script): an
        // unterminated final statement is still run, via `run_trailing`.
        sh.run_trailing("SELECT nosuchfunc()");
        assert!(sh.into_result().is_err(), "a failing trailing statement must be reported");
    }

    #[test]
    fn trailing_statement_without_semicolon_at_eof_succeeds_when_ok() {
        let opts = Options::default();
        let mut sh = Shell::new(&opts);
        sh.run_trailing("SELECT 1");
        assert!(sh.into_result().is_ok());
    }
}
