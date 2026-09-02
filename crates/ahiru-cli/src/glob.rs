//! Filename globbing and directory walking.
//!
//! The shell already expands globs it sees, but the CLI also needs to expand
//! patterns that arrive *quoted* — either quoted on the command line to keep
//! several files in one table, or written inside SQL
//! (`FROM 'data/*.parquet'`), where the shell never sees them at all.

use std::path::{Path, PathBuf};

/// True if `s` contains a glob metacharacter.
pub fn is_pattern(s: &str) -> bool {
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                // An escaped character is never itself a metacharacter; just
                // skip past it (if there is one — a trailing lone backslash
                // has nothing to escape and is not a metacharacter either).
                chars.next();
            }
            '*' | '?' | '[' => return true,
            _ => {}
        }
    }
    false
}

/// Extensions recognized as loadable data files, matched case-insensitively.
const DATA_EXTENSIONS: &[&str] = &["parquet", "csv", "tsv", "jsonl", "ndjson", "json"];

/// True if `path`'s extension is one of the supported data file extensions.
fn has_data_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| DATA_EXTENSIONS.iter().any(|want| e.eq_ignore_ascii_case(want)))
}

/// True if `name` (a file or directory name, not a full path) is hidden, i.e.
/// starts with `.` — this also covers `.` and `..` themselves.
fn is_hidden(name: &str) -> bool {
    name.starts_with('.')
}

/// True if the directory entry at `path` is a symlink. Used to avoid
/// following symlinked directories while recursing, so a cyclic link can't
/// send the walk into an infinite loop.
fn is_symlink(path: &Path) -> bool {
    std::fs::symlink_metadata(path).map(|m| m.file_type().is_symlink()).unwrap_or(false)
}

/// Matches one path component against one pattern component.
///
/// Supports `*` (any run of characters, including none), `?` (exactly one
/// character), `[abc]` / `[a-z]` / `[!abc]` character classes, and `\`
/// escaping of the next character. Operates on `char`s (not bytes) so
/// non-ASCII names are handled correctly.
///
/// Implemented as the classic iterative two-pointer wildcard match: walk
/// `pattern` and `text` in lockstep, and on a mismatch backtrack to the most
/// recent `*` (retrying it against one more character of `text`) instead of
/// recursing — this keeps the algorithm O(pattern * text) in the worst case
/// and uses no call stack proportional to input size.
pub fn matches(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();

    let mut pi = 0usize; // index into `p`
    let mut ti = 0usize; // index into `t`
                         // Position of the most recent unmatched `*` in `p`, and the position in
                         // `t` we last retried it at (i.e. one past the last attempt).
    let mut star_pi: Option<usize> = None;
    let mut star_ti = 0usize;

    while ti < t.len() {
        if pi < p.len() {
            match p[pi] {
                '*' => {
                    // Record the star and optimistically match zero chars.
                    star_pi = Some(pi);
                    star_ti = ti;
                    pi += 1;
                    continue;
                }
                '?' => {
                    pi += 1;
                    ti += 1;
                    continue;
                }
                '[' => {
                    if let Some((matched, next_pi)) = match_class(&p, pi, t[ti]) {
                        if matched {
                            pi = next_pi;
                            ti += 1;
                            continue;
                        }
                        // Class parsed fine but didn't match this char: fall
                        // through to backtrack handling below.
                    } else {
                        // Malformed class (no closing `]`): treat `[` as a
                        // literal character.
                        if t[ti] == '[' {
                            pi += 1;
                            ti += 1;
                            continue;
                        }
                    }
                }
                '\\' => {
                    let lit = p.get(pi + 1).copied().unwrap_or('\\');
                    if t[ti] == lit {
                        pi += if p.get(pi + 1).is_some() { 2 } else { 1 };
                        ti += 1;
                        continue;
                    }
                }
                c => {
                    if c == t[ti] {
                        pi += 1;
                        ti += 1;
                        continue;
                    }
                }
            }
        }
        // Mismatch (or pattern exhausted while text remains): backtrack to
        // the last `*`, consuming one more character of `text` with it.
        if let Some(sp) = star_pi {
            star_ti += 1;
            ti = star_ti;
            pi = sp + 1;
        } else {
            return false;
        }
    }

    // Text is exhausted; any remaining pattern must be trailing `*`s.
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

/// Parses and evaluates a `[...]` character class starting at `p[start]`
/// (which must be `[`). Returns `Some((matched, index_after_class))` on a
/// well-formed class (with a closing `]`), or `None` if there is no closing
/// `]` at all (malformed — caller falls back to treating `[` literally).
fn match_class(p: &[char], start: usize, c: char) -> Option<(bool, usize)> {
    debug_assert_eq!(p[start], '[');
    let mut i = start + 1;
    let negate = matches!(p.get(i), Some('!'));
    if negate {
        i += 1;
    }
    let body_start = i;
    // Find the closing `]`. A `]` immediately at the start of the body (right
    // after `[` or `[!`) is treated as a literal member, not the closer —
    // standard glob behavior.
    let mut j = body_start;
    let mut first = true;
    let close = loop {
        match p.get(j) {
            None => return None,
            Some(']') if !first => break j,
            _ => {
                first = false;
                j += 1;
            }
        }
    };

    let mut found = false;
    let mut k = body_start;
    while k < close {
        // A single member, possibly the start of a `a-z` range.
        let lo = p[k];
        if k + 2 < close && p[k + 1] == '-' {
            let hi = p[k + 2];
            if lo <= c && c <= hi {
                found = true;
            }
            k += 3;
        } else {
            if lo == c {
                found = true;
            }
            k += 1;
        }
    }

    let matched = if negate { !found } else { found };
    Some((matched, close + 1))
}

/// One in-progress `expand` result: a filesystem path to check/read from,
/// paired with the display path (built with the pattern's own separators and
/// prefix, never an added `./`).
struct Partial {
    fs_path: PathBuf,
    display: PathBuf,
}

/// Every existing path matching `pattern`. Empty if nothing matches.
///
/// The pattern is split on `/` into components which are walked one at a
/// time: literal components are appended without touching the filesystem,
/// `**` recurses through zero or more directory levels, and any other
/// component lists its parent directory and keeps entries whose name
/// [`matches()`]. Only the final component is allowed to match plain files —
/// every earlier component must resolve to a directory to descend into.
pub fn expand(pattern: &str) -> Vec<PathBuf> {
    if pattern.is_empty() {
        return Vec::new();
    }

    let absolute = pattern.starts_with('/');
    let comps: Vec<&str> = pattern.split('/').filter(|c| !c.is_empty()).collect();
    if comps.is_empty() {
        return Vec::new();
    }

    // The filesystem walk always starts from a real, readable location (`/`
    // or `.`), but the *display* path only gets that prefix when the
    // pattern was itself absolute — a relative pattern like `data/*.parquet`
    // must yield `data/x.parquet`, not `./data/x.parquet`.
    let start_fs: PathBuf = if absolute { PathBuf::from("/") } else { PathBuf::from(".") };
    let start_display: PathBuf = if absolute { PathBuf::from("/") } else { PathBuf::new() };
    let mut current = vec![Partial { fs_path: start_fs, display: start_display }];

    for (idx, comp) in comps.iter().enumerate() {
        let is_last = idx == comps.len() - 1;
        let mut next = Vec::new();

        if *comp == "**" {
            for base in &current {
                collect_recursive(&base.fs_path, &base.display, is_last, &mut next);
            }
        } else if is_pattern(comp) {
            for base in &current {
                list_matching(&base.fs_path, &base.display, comp, is_last, &mut next);
            }
        } else {
            // Literal component: append without reading the directory. Only
            // keep it if the resulting path actually exists, so a literal
            // segment in the middle of a pattern (e.g. `data/2024/*.csv`)
            // correctly yields nothing when `data/2024` doesn't exist.
            let lit = unescape_literal(comp);
            for base in &current {
                let fs_path = base.fs_path.join(&lit);
                let display = base.display.join(&lit);
                if is_last {
                    if fs_path.exists() || is_symlink(&fs_path) {
                        next.push(Partial { fs_path, display });
                    }
                } else if fs_path.is_dir() {
                    next.push(Partial { fs_path, display });
                }
            }
        }

        current = next;
    }

    let mut out: Vec<PathBuf> = current.into_iter().map(|p| p.display).collect();
    dedup_unsorted(&mut out);
    out
}

/// Removes duplicate paths without requiring the input to be sorted first
/// (the caller is responsible for sorting the final result).
fn dedup_unsorted(paths: &mut Vec<PathBuf>) {
    let mut seen = std::collections::HashSet::with_capacity(paths.len());
    paths.retain(|p| seen.insert(p.clone()));
}

/// Strips backslash-escaping from a literal (non-metacharacter) glob
/// component so `\[literal\]` in a pattern becomes `[literal]` on disk.
pub fn unescape_literal(comp: &str) -> String {
    let mut out = String::with_capacity(comp.len());
    let mut chars = comp.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(next) = chars.next() {
                out.push(next);
                continue;
            }
        }
        out.push(c);
    }
    out
}

/// Lists `fs_dir` and appends entries whose file name matches `comp` (a
/// pattern component). If `is_last` is false, only directories are kept
/// (they become the base for the next pattern component); if true, both
/// files and directories are kept as final results, matching typical shell
/// glob behavior.
fn list_matching(
    fs_dir: &Path,
    display_dir: &Path,
    comp: &str,
    is_last: bool,
    out: &mut Vec<Partial>,
) {
    let comp_is_hidden = comp.starts_with('.');
    let entries = match std::fs::read_dir(fs_dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else { continue };
        if is_hidden(name_str) && !comp_is_hidden {
            continue;
        }
        if !matches(comp, name_str) {
            continue;
        }
        if !is_last {
            match entry.file_type() {
                Ok(ft) if ft.is_dir() => {}
                _ => continue,
            }
        }
        out.push(Partial { fs_path: entry.path(), display: display_dir.join(name_str) });
    }
}

/// Handles a `**` pattern component: recurses through the directory tree
/// rooted at `fs_dir`, treating each level (including zero levels, i.e.
/// `fs_dir` itself) as a candidate match for `**`. If `is_last` is true,
/// every directory (and, per typical `**` semantics, every file) visited is
/// a result; if false, `**` is followed immediately by more pattern
/// components, so only directories are pushed as bases for the next
/// component.
///
/// Symlinked directories are not followed, so a cyclic symlink cannot send
/// this into an infinite loop.
fn collect_recursive(fs_dir: &Path, display_dir: &Path, is_last: bool, out: &mut Vec<Partial>) {
    // `**` matches zero levels: the directory itself is always a candidate,
    // whether it's the final result (`is_last`) or the base for the next
    // pattern component.
    out.push(Partial { fs_path: fs_dir.to_path_buf(), display: display_dir.to_path_buf() });

    let entries = match std::fs::read_dir(fs_dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else { continue };
        if is_hidden(name_str) {
            continue;
        }
        let fs_path = entry.path();
        let is_dir = matches!(entry.file_type(), Ok(ft) if ft.is_dir());
        let display_path = display_dir.join(name_str);
        if is_dir {
            if is_symlink(&fs_path) {
                continue;
            }
            collect_recursive(&fs_path, &display_path, is_last, out);
        } else if is_last {
            // `**` also matches files directly, same as `find`.
            out.push(Partial { fs_path, display: display_path });
        }
    }
}

/// Every supported data file below `dir`, recursively (Hive layouts).
///
/// This is what lets a Hive-partitioned directory (e.g.
/// `sales/year=2024/month=01/part.parquet`) be queried as a single table:
/// the walk descends arbitrarily deep, collecting every file whose extension
/// is one of `parquet`, `csv`, `tsv`, `jsonl`, `ndjson`, or `json`
/// (case-insensitive). Hidden files/directories and symlinked directories
/// are skipped, and unreadable directories are silently ignored.
pub fn walk_dir(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk_dir_into(dir, dir, &mut out);
    out
}

fn walk_dir_into(fs_dir: &Path, display_dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(fs_dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else { continue };
        if is_hidden(name_str) {
            continue;
        }
        let fs_path = entry.path();
        let display_path = display_dir.join(name_str);
        let is_dir = matches!(entry.file_type(), Ok(ft) if ft.is_dir());
        if is_dir {
            if is_symlink(&fs_path) {
                continue;
            }
            walk_dir_into(&fs_path, &display_path, out);
        } else if has_data_extension(&fs_path) {
            out.push(display_path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- is_pattern -------------------------------------------------

    #[test]
    fn is_pattern_detects_metacharacters() {
        assert!(is_pattern("*.parquet"));
        assert!(is_pattern("a?c"));
        assert!(is_pattern("[abc]"));
        assert!(is_pattern("data/[!x]/*.csv"));
        assert!(is_pattern("prefix*"));
    }

    #[test]
    fn is_pattern_false_for_plain_paths() {
        assert!(!is_pattern("data/file.parquet"));
        assert!(!is_pattern(""));
        assert!(!is_pattern("just-a-name.csv"));
    }

    #[test]
    fn is_pattern_escaped_metachar_does_not_count() {
        assert!(!is_pattern(r"\*.parquet"));
        assert!(!is_pattern(r"a\?c"));
        assert!(!is_pattern(r"\[abc\]"));
        // A metacharacter elsewhere in the string still counts even if one
        // instance is escaped.
        assert!(is_pattern(r"\**"));
    }

    // ---- matches ------------------------------------------------------

    #[test]
    fn matches_literal() {
        assert!(matches("file.csv", "file.csv"));
        assert!(!matches("file.csv", "other.csv"));
    }

    #[test]
    fn matches_star() {
        assert!(matches("*.parquet", "data.parquet"));
        assert!(matches("*.parquet", ".parquet"));
        assert!(matches("a*b", "ab"));
        assert!(matches("a*b", "axxxb"));
        assert!(!matches("a*b", "axxxc"));
        assert!(matches("*", ""));
        assert!(matches("*", "anything"));
        assert!(matches("*a*b*", "xaxxbx"));
        assert!(!matches("*a*b*", "xxxx"));
    }

    #[test]
    fn matches_question_mark() {
        assert!(matches("a?c", "abc"));
        assert!(!matches("a?c", "ac"));
        assert!(!matches("a?c", "abbc"));
        assert!(matches("???", "xyz"));
        assert!(!matches("???", "xy"));
    }

    #[test]
    fn matches_char_class() {
        assert!(matches("[a-c]x", "ax"));
        assert!(matches("[a-c]x", "bx"));
        assert!(matches("[a-c]x", "cx"));
        assert!(!matches("[a-c]x", "dx"));
        assert!(matches("[abc]x", "ax"));
        assert!(!matches("[abc]x", "dx"));
    }

    #[test]
    fn matches_char_class_negated() {
        assert!(matches("[!a-c]x", "dx"));
        assert!(!matches("[!a-c]x", "ax"));
        assert!(matches("[!abc]x", "zx"));
        assert!(!matches("[!abc]x", "bx"));
    }

    #[test]
    fn matches_char_class_literal_bracket_close() {
        // `]` right after `[` or `[!` is a literal member, not the closer.
        assert!(matches("[]a]x", "]x"));
        assert!(matches("[]a]x", "ax"));
    }

    #[test]
    fn matches_escape() {
        assert!(matches(r"a\*c", "a*c"));
        assert!(!matches(r"a\*c", "abc"));
        assert!(matches(r"a\?c", "a?c"));
        assert!(matches(r"a\[c", "a[c"));
    }

    #[test]
    fn matches_combination() {
        assert!(matches("*.[ct]sv", "data.csv"));
        assert!(matches("*.[ct]sv", "data.tsv"));
        assert!(!matches("*.[ct]sv", "data.jsv"));
        assert!(matches("report_????.csv", "report_2024.csv"));
    }

    #[test]
    fn matches_non_ascii() {
        assert!(matches(
            "*.parquet",
            "\u{3b4}\u{3b5}\u{3b4}\u{3bf}\u{3bc}\u{3ad}\u{3bd}\u{3b1}.parquet"
        ));
        assert!(matches(
            "\u{3b4}\u{3b5}?\u{3bf}\u{3bc}\u{3ad}\u{3bd}\u{3b1}.csv",
            "\u{3b4}\u{3b5}\u{3b4}\u{3bf}\u{3bc}\u{3ad}\u{3bd}\u{3b1}.csv"
        ));
        assert!(matches("[\u{3b1}-\u{3c9}]x", "\u{3b2}x"));
        assert!(matches("😀*", "😀🎉"));
    }

    #[test]
    fn matches_no_backtrack_stack_overflow() {
        // A pattern with many stars against a long non-matching text should
        // return quickly without recursing (would stack overflow if this
        // were implemented recursively without care).
        let pattern = "*".repeat(50) + "x";
        let text = "a".repeat(20_000);
        assert!(!matches(&pattern, &text));
    }

    // ---- expand / walk_dir (filesystem) -------------------------------

    /// Builds a small unique scratch directory tree for a filesystem test,
    /// returning its path. The directory is removed by `Scratch::drop`.
    struct Scratch {
        dir: PathBuf,
    }

    impl Scratch {
        fn new(test_name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "ahiru_glob_test_{}_{}",
                std::process::id(),
                test_name
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("create scratch dir");
            Scratch { dir }
        }

        fn file(&self, rel: &str) {
            let path = self.dir.join(rel);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).expect("create parent dir");
            }
            std::fs::write(&path, b"x").expect("write scratch file");
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn sorted(mut v: Vec<PathBuf>) -> Vec<PathBuf> {
        v.sort();
        v
    }

    #[test]
    fn expand_plain_star() {
        let s = Scratch::new("plain_star");
        s.file("a.csv");
        s.file("b.csv");
        s.file("c.txt");

        let pattern = s.dir.join("*.csv");
        let pattern_str = pattern.to_str().unwrap();
        let got = sorted(expand(pattern_str));
        let want = sorted(vec![s.dir.join("a.csv"), s.dir.join("b.csv")]);
        assert_eq!(got, want);
    }

    #[test]
    fn expand_literal_directory_prefix() {
        let s = Scratch::new("literal_prefix");
        s.file("sub/a.parquet");
        s.file("sub/b.parquet");
        s.file("other/c.parquet");

        let pattern = s.dir.join("sub/*.parquet");
        let got = sorted(expand(pattern.to_str().unwrap()));
        let want = sorted(vec![s.dir.join("sub/a.parquet"), s.dir.join("sub/b.parquet")]);
        assert_eq!(got, want);
    }

    #[test]
    fn expand_relative_pattern_keeps_relative_spelling() {
        let s = Scratch::new("relative_spelling");
        s.file("data/x.parquet");

        let cwd = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&s.dir).expect("chdir into scratch");
        let got = expand("data/*.parquet");
        std::env::set_current_dir(&cwd).expect("restore cwd");

        assert_eq!(got, vec![PathBuf::from("data/x.parquet")]);
    }

    #[test]
    fn expand_recursive_double_star() {
        let s = Scratch::new("recursive");
        s.file("year=2024/month=01/part.parquet");
        s.file("year=2024/month=02/part.parquet");
        s.file("year=2025/month=01/part.parquet");
        s.file("top.parquet");

        let pattern = s.dir.join("**/*.parquet");
        let got = sorted(expand(pattern.to_str().unwrap()));
        let want = sorted(vec![
            s.dir.join("year=2024/month=01/part.parquet"),
            s.dir.join("year=2024/month=02/part.parquet"),
            s.dir.join("year=2025/month=01/part.parquet"),
            s.dir.join("top.parquet"),
        ]);
        assert_eq!(got, want);
    }

    #[test]
    fn expand_skips_hidden_files() {
        let s = Scratch::new("hidden");
        s.file("visible.csv");
        s.file(".hidden.csv");

        let pattern = s.dir.join("*.csv");
        let got = expand(pattern.to_str().unwrap());
        assert_eq!(got, vec![s.dir.join("visible.csv")]);

        // But a pattern that itself starts with `.` does pick up dotfiles.
        let dot_pattern = s.dir.join(".*.csv");
        let got_dot = expand(dot_pattern.to_str().unwrap());
        assert_eq!(got_dot, vec![s.dir.join(".hidden.csv")]);
    }

    #[test]
    fn expand_no_duplicates_from_overlapping_matches() {
        let s = Scratch::new("no_dup");
        s.file("a.csv");
        s.file("b.csv");

        // `**` at the root matches the root itself (zero levels) once; make
        // sure results are still deduplicated even so.
        let pattern = s.dir.join("**/*.csv");
        let got = expand(pattern.to_str().unwrap());
        let mut seen = std::collections::HashSet::new();
        for p in &got {
            assert!(seen.insert(p.clone()), "duplicate result: {p:?}");
        }
    }

    #[test]
    fn walk_dir_finds_hive_layout_and_skips_success_marker() {
        let s = Scratch::new("walk_hive");
        s.file("sales/year=2024/month=01/part.parquet");
        s.file("sales/year=2024/month=02/part.parquet");
        s.file("sales/year=2025/month=01/part.parquet");
        s.file("sales/_SUCCESS");

        let got = sorted(walk_dir(&s.dir.join("sales")));
        let want = sorted(vec![
            s.dir.join("sales/year=2024/month=01/part.parquet"),
            s.dir.join("sales/year=2024/month=02/part.parquet"),
            s.dir.join("sales/year=2025/month=01/part.parquet"),
        ]);
        assert_eq!(got, want);
    }

    #[test]
    fn walk_dir_skips_hidden_and_unsupported_extensions() {
        let s = Scratch::new("walk_skip");
        s.file("data.parquet");
        s.file("data.csv");
        s.file("data.JSON");
        s.file("notes.txt");
        s.file(".hidden.parquet");
        s.file(".hidden_dir/inside.parquet");

        let got = sorted(walk_dir(&s.dir));
        let want = sorted(vec![
            s.dir.join("data.parquet"),
            s.dir.join("data.csv"),
            s.dir.join("data.JSON"),
        ]);
        assert_eq!(got, want);
    }

    // ---- sanity check against the repo's real sample data -------------

    #[test]
    fn walk_dir_over_real_hive_sample_data() {
        let root = format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), "tests/data/hive");
        let root = PathBuf::from(root);
        if !root.is_dir() {
            // The sample data is optional context for this crate; don't
            // fail the suite if the checkout doesn't have it.
            return;
        }
        let got = walk_dir(&root);
        assert!(!got.is_empty(), "expected to find files under {root:?}");
        for p in &got {
            assert!(has_data_extension(p), "unexpected non-data file matched: {p:?}");
        }
    }

    #[test]
    fn expand_glob_over_real_multi_sample_data() {
        let root = format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), "tests/data/multi");
        if !PathBuf::from(&root).is_dir() {
            return;
        }
        let pattern = format!("{root}/*.parquet");
        let got = expand(&pattern);
        assert!(!got.is_empty(), "expected to find parquet files under {root}");
    }
}
