//! Interactive line editing: history, cursor movement, tab completion.
//!
//! The workspace has no third-party dependencies (see `docs/DESIGN.md`), so
//! there is no readline crate here. Raw mode is obtained by shelling out to
//! `stty`, and the escape sequences are handled by hand. When stdin is not a
//! terminal, or raw mode cannot be entered, this degrades to plain
//! `read_line` — which is what a piped or redirected session wants anyway.

use std::io::{IsTerminal, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

pub enum Line {
    Text(String),
    /// Ctrl-C: abandon whatever is being typed, keep the session.
    Interrupted,
    /// Ctrl-D on an empty line, or EOF on a pipe.
    Eof,
}

pub struct Editor {
    history: Vec<String>,
    path: Option<PathBuf>,
    /// Whether both stdin and stdout are terminals. Computed once at
    /// construction; `stty` failures downgrade this to `false` permanently
    /// the first time raw mode fails to engage.
    interactive: bool,
}

/// A cursor-and-contents pair for the line currently being edited.
///
/// Kept separate from `Editor` (and from any terminal I/O) so the character-
/// and word-level editing logic can be unit tested without a TTY. All
/// indices here are *character* indices, not byte offsets — callers that
/// need byte offsets into the underlying `String` must convert.
#[derive(Default, Debug, PartialEq, Eq)]
struct Buffer {
    /// The line content, stored as chars for O(1) char-indexed edits.
    chars: Vec<char>,
    /// Cursor position, in chars, in `0..=chars.len()`.
    cursor: usize,
}

impl Buffer {
    fn from_str(s: &str) -> Buffer {
        let chars: Vec<char> = s.chars().collect();
        let cursor = chars.len();
        Buffer { chars, cursor }
    }

    fn as_string(&self) -> String {
        self.chars.iter().collect()
    }

    fn insert_char(&mut self, c: char) {
        self.chars.insert(self.cursor, c);
        self.cursor += 1;
    }

    /// Deletes the char immediately before the cursor (backspace). Returns
    /// `true` if something was deleted.
    fn delete_before(&mut self) -> bool {
        if self.cursor == 0 {
            return false;
        }
        self.cursor -= 1;
        self.chars.remove(self.cursor);
        true
    }

    /// Deletes the char under the cursor (delete/Ctrl-D). Returns `true` if
    /// something was deleted.
    fn delete_at_cursor(&mut self) -> bool {
        if self.cursor >= self.chars.len() {
            return false;
        }
        self.chars.remove(self.cursor);
        true
    }

    fn move_left(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
        }
    }

    fn move_right(&mut self) {
        if self.cursor < self.chars.len() {
            self.cursor += 1;
        }
    }

    fn move_home(&mut self) {
        self.cursor = 0;
    }

    fn move_end(&mut self) {
        self.cursor = self.chars.len();
    }

    /// Deletes from the start of the line up to (not including) the cursor,
    /// leaving the cursor at 0. Returns the deleted text (unused for now,
    /// but handy for a future kill-ring).
    fn kill_to_start(&mut self) -> String {
        let removed: String = self.chars.drain(0..self.cursor).collect();
        self.cursor = 0;
        removed
    }

    /// Deletes from the cursor to the end of the line.
    fn kill_to_end(&mut self) -> String {
        let removed: String = self.chars.drain(self.cursor..).collect();
        removed
    }

    /// Index of the start of the previous word, treating runs of
    /// non-whitespace as words (mirrors typical shell/readline behavior).
    fn prev_word_boundary(&self) -> usize {
        let mut i = self.cursor;
        while i > 0 && self.chars[i - 1].is_whitespace() {
            i -= 1;
        }
        while i > 0 && !self.chars[i - 1].is_whitespace() {
            i -= 1;
        }
        i
    }

    /// Index of the start of the next word.
    fn next_word_boundary(&self) -> usize {
        let mut i = self.cursor;
        let len = self.chars.len();
        while i < len && self.chars[i].is_whitespace() {
            i += 1;
        }
        while i < len && !self.chars[i].is_whitespace() {
            i += 1;
        }
        i
    }

    fn move_word_left(&mut self) {
        self.cursor = self.prev_word_boundary();
    }

    fn move_word_right(&mut self) {
        self.cursor = self.next_word_boundary();
    }

    /// Deletes the word before the cursor (Ctrl-W).
    fn delete_word_before(&mut self) -> String {
        let start = self.prev_word_boundary();
        let removed: String = self.chars.drain(start..self.cursor).collect();
        self.cursor = start;
        removed
    }

    /// Replaces the `[start, end)` char range with `text`, leaving the
    /// cursor immediately after the inserted text.
    fn replace_range(&mut self, start: usize, end: usize, text: &str) {
        let replacement: Vec<char> = text.chars().collect();
        let inserted_len = replacement.len();
        self.chars.splice(start..end, replacement);
        self.cursor = start + inserted_len;
    }
}

/// Finds the char-index start of the "word" ending at `cursor`, for tab
/// completion purposes. Words are split on whitespace and the SQL
/// punctuation that commonly precedes an identifier: `(`, `,`, `'`, `"`.
fn word_before_cursor(chars: &[char], cursor: usize) -> usize {
    let mut i = cursor;
    while i > 0 {
        let c = chars[i - 1];
        if c.is_whitespace() || matches!(c, '(' | ',' | '\'' | '"') {
            break;
        }
        i -= 1;
    }
    i
}

/// Longest common prefix shared by every string in `items`. Empty if
/// `items` is empty or the strings share no common prefix.
fn longest_common_prefix(items: &[String]) -> String {
    let mut iter = items.iter();
    let first = match iter.next() {
        Some(s) => s,
        None => return String::new(),
    };
    let mut prefix_len = first.chars().count();
    let first_chars: Vec<char> = first.chars().collect();
    for item in iter {
        let item_chars: Vec<char> = item.chars().collect();
        let mut common = 0;
        while common < prefix_len
            && common < item_chars.len()
            && first_chars[common] == item_chars[common]
        {
            common += 1;
        }
        prefix_len = common;
    }
    first_chars[..prefix_len].iter().collect()
}

/// Escapes a history entry for one-line-per-entry storage: literal
/// backslashes become `\\` and literal newlines become `\n` (the two-char
/// sequence), so a multi-line SQL statement round-trips as a single file
/// line.
fn escape_history_entry(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            _ => out.push(c),
        }
    }
    out
}

/// Reverses [`escape_history_entry`].
fn unescape_history_entry(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    // Unknown escape: keep it verbatim rather than losing data.
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Maximum number of history entries retained by `save_history`.
const HISTORY_LIMIT: usize = 1000;

/// RAII guard that puts the terminal into raw mode on construction and
/// restores it on drop, so a panic mid-edit can never leave the user's
/// terminal in raw mode. `enable` returns `None` if raw mode could not be
/// entered (no `stty`, or it failed), in which case the caller should fall
/// back to dumb-mode reading.
struct RawModeGuard {
    /// The output of `stty -g`, used to restore exact settings. Falls back
    /// to `stty sane` if this wasn't captured.
    saved: Option<String>,
}

impl RawModeGuard {
    /// Attempts to enter raw mode. Returns `None` on any failure (missing
    /// `stty`, non-zero exit, etc.) without leaving the terminal modified.
    fn enable() -> Option<RawModeGuard> {
        // Capture current settings so we can restore them verbatim.
        let saved = Command::new("stty")
            .arg("-g")
            .stdin(Stdio::inherit())
            .output()
            .ok()
            .filter(|out| out.status.success())
            .and_then(|out| String::from_utf8(out.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let status =
            Command::new("stty").arg("raw").arg("-echo").stdin(Stdio::inherit()).status().ok()?;
        if !status.success() {
            return None;
        }
        Some(RawModeGuard { saved })
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let result = if let Some(saved) = &self.saved {
            Command::new("stty").arg(saved).stdin(Stdio::inherit()).status()
        } else {
            Command::new("stty").arg("sane").stdin(Stdio::inherit()).status()
        };
        // Best effort: if even `stty sane` fails there is nothing more we
        // can do from here, and panicking in a `Drop` during unwinding
        // would abort the process.
        let _ = result;
    }
}

impl Editor {
    /// `history_path` is loaded now and appended to by `save_history`.
    pub fn new(history_path: Option<PathBuf>) -> Editor {
        let mut history = Vec::new();
        if let Some(path) = &history_path {
            if let Ok(contents) = std::fs::read_to_string(path) {
                for line in contents.lines() {
                    if line.is_empty() {
                        continue;
                    }
                    history.push(unescape_history_entry(line));
                }
            }
        }
        let interactive = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
        Editor { history, path: history_path, interactive }
    }

    /// True when the editor is driving a real terminal (raw mode available).
    pub fn is_interactive(&self) -> bool {
        self.interactive
    }

    /// Reads one line. `complete` maps the word before the cursor to its
    /// candidate completions.
    pub fn read_line(&mut self, prompt: &str, complete: &dyn Fn(&str) -> Vec<String>) -> Line {
        if !self.interactive {
            return self.read_line_dumb();
        }
        match RawModeGuard::enable() {
            Some(guard) => {
                let result = self.read_line_raw(prompt, complete);
                drop(guard);
                result
            }
            None => {
                // `stty` is missing or failed: fall back permanently, since
                // it is unlikely to start working mid-session.
                self.interactive = false;
                self.read_line_dumb()
            }
        }
    }

    pub fn add_history(&mut self, line: &str) {
        if line.is_empty() {
            return;
        }
        if self.history.last().map(|s| s.as_str()) == Some(line) {
            return;
        }
        self.history.push(line.to_string());
    }

    pub fn save_history(&self) {
        let Some(path) = &self.path else { return };
        let start = self.history.len().saturating_sub(HISTORY_LIMIT);
        let mut out = String::new();
        for entry in &self.history[start..] {
            out.push_str(&escape_history_entry(entry));
            out.push('\n');
        }
        let _ = std::fs::write(path, out);
    }

    /// Plain, non-interactive line reading: no prompt echo control, no
    /// escape codes, just `read_line` semantics on stdin. Used both when
    /// stdin/stdout aren't terminals and as the permanent fallback when
    /// `stty` is unavailable.
    fn read_line_dumb(&mut self) -> Line {
        let mut line = String::new();
        match std::io::stdin().read_line(&mut line) {
            Ok(0) => Line::Eof,
            Ok(_) => {
                if line.ends_with('\n') {
                    line.pop();
                    if line.ends_with('\r') {
                        line.pop();
                    }
                }
                Line::Text(line)
            }
            Err(_) => Line::Eof,
        }
    }

    /// Full raw-mode line editor: reads bytes one at a time from stdin,
    /// interprets control sequences, and repaints after each edit.
    fn read_line_raw(&mut self, prompt: &str, complete: &dyn Fn(&str) -> Vec<String>) -> Line {
        let mut buf = Buffer::default();
        // `history_cursor` walks the history; `self.history.len()` denotes
        // the "bottom" (not-yet-committed) line, which is `pending`.
        let mut history_cursor = self.history.len();
        let mut pending = String::new();

        let stdin = std::io::stdin();
        let mut lock = stdin.lock();
        let mut stdout = std::io::stdout();

        redraw(&mut stdout, prompt, &buf);

        loop {
            let byte = match read_one_byte(&mut lock) {
                Some(b) => b,
                None => {
                    // EOF mid-line (e.g. terminal hung up): treat like Ctrl-D.
                    return Line::Eof;
                }
            };

            match byte {
                b'\r' | b'\n' => {
                    let _ = write!(stdout, "\r\n");
                    let _ = stdout.flush();
                    return Line::Text(buf.as_string());
                }
                0x03 => {
                    // Ctrl-C
                    let _ = write!(stdout, "\r\n");
                    let _ = stdout.flush();
                    return Line::Interrupted;
                }
                0x04 => {
                    // Ctrl-D
                    if buf.chars.is_empty() {
                        let _ = write!(stdout, "\r\n");
                        let _ = stdout.flush();
                        return Line::Eof;
                    }
                    buf.delete_at_cursor();
                }
                0x7f | 0x08 => {
                    // Backspace
                    buf.delete_before();
                }
                0x01 => buf.move_home(), // Ctrl-A
                0x05 => buf.move_end(),  // Ctrl-E
                0x15 => {
                    buf.kill_to_start(); // Ctrl-U
                }
                0x0b => {
                    buf.kill_to_end(); // Ctrl-K
                }
                0x17 => {
                    buf.delete_word_before(); // Ctrl-W
                }
                0x0c => {
                    // Ctrl-L: clear screen and repaint at top-left.
                    let _ = write!(stdout, "\x1b[H\x1b[2J");
                    let _ = stdout.flush();
                }
                0x09 => {
                    // Tab: completion.
                    handle_completion(&mut buf, complete, &mut stdout, prompt);
                }
                0x1b => {
                    // Escape sequence.
                    match read_escape_sequence(&mut lock) {
                        EscapeSeq::Left => buf.move_left(),
                        EscapeSeq::Right => buf.move_right(),
                        EscapeSeq::Up => {
                            if history_cursor > 0 {
                                if history_cursor == self.history.len() {
                                    pending = buf.as_string();
                                }
                                history_cursor -= 1;
                                buf = Buffer::from_str(&self.history[history_cursor]);
                            }
                        }
                        EscapeSeq::Down => {
                            if history_cursor < self.history.len() {
                                history_cursor += 1;
                                if history_cursor == self.history.len() {
                                    buf = Buffer::from_str(&pending);
                                } else {
                                    buf = Buffer::from_str(&self.history[history_cursor]);
                                }
                            }
                        }
                        EscapeSeq::Home => buf.move_home(),
                        EscapeSeq::End => buf.move_end(),
                        EscapeSeq::Delete => {
                            buf.delete_at_cursor();
                        }
                        EscapeSeq::WordLeft => buf.move_word_left(),
                        EscapeSeq::WordRight => buf.move_word_right(),
                        EscapeSeq::Unknown => {}
                    }
                }
                b if b < 0x20 => {
                    // Other control characters: ignore.
                }
                first_byte => {
                    if let Some(c) = read_utf8_char(&mut lock, first_byte) {
                        buf.insert_char(c);
                    }
                }
            }

            redraw(&mut stdout, prompt, &buf);
        }
    }
}

/// Reads exactly one byte from `r`. Returns `None` at EOF or on error.
fn read_one_byte<R: Read>(r: &mut R) -> Option<u8> {
    let mut b = [0u8; 1];
    match r.read(&mut b) {
        Ok(1) => Some(b[0]),
        _ => None,
    }
}

/// Reads the remaining bytes of a UTF-8 sequence that started with
/// `first_byte`, and decodes the resulting char. Returns `None` if the
/// sequence is malformed (in which case the malformed bytes are simply
/// dropped — there is no good recovery for this at the terminal layer).
fn read_utf8_char<R: Read>(r: &mut R, first_byte: u8) -> Option<char> {
    let extra = if first_byte & 0b1110_0000 == 0b1100_0000 {
        1
    } else if first_byte & 0b1111_0000 == 0b1110_0000 {
        2
    } else if first_byte & 0b1111_1000 == 0b1111_0000 {
        3
    } else if first_byte < 0x80 {
        0
    } else {
        // Invalid leading byte (stray continuation byte, etc.).
        return None;
    };
    let mut bytes = vec![first_byte];
    for _ in 0..extra {
        bytes.push(read_one_byte(r)?);
    }
    std::str::from_utf8(&bytes).ok()?.chars().next()
}

enum EscapeSeq {
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    Delete,
    WordLeft,
    WordRight,
    Unknown,
}

/// Parses an escape sequence after the initial `\x1b` has already been
/// consumed. Handles the common `CSI` (`ESC [ ...`) forms used by arrow
/// keys, Home/End, Delete, and Ctrl/Alt + arrow (word movement), plus the
/// bare `ESC b` / `ESC f` (Alt-B / Alt-F) word-movement forms. Anything
/// else is reported as `Unknown` without consuming more than necessary.
fn read_escape_sequence<R: Read>(r: &mut R) -> EscapeSeq {
    let Some(b1) = read_one_byte(r) else {
        return EscapeSeq::Unknown;
    };
    match b1 {
        b'b' => return EscapeSeq::WordLeft,
        b'f' => return EscapeSeq::WordRight,
        b'[' | b'O' => {}
        _ => return EscapeSeq::Unknown,
    }
    let Some(b2) = read_one_byte(r) else {
        return EscapeSeq::Unknown;
    };
    match b2 {
        b'D' => EscapeSeq::Left,
        b'C' => EscapeSeq::Right,
        b'A' => EscapeSeq::Up,
        b'B' => EscapeSeq::Down,
        b'H' => EscapeSeq::Home,
        b'F' => EscapeSeq::End,
        b'1' | b'3' | b'5' | b'7' => {
            // Numeric CSI sequences: `ESC [ <digits> ~` (Delete, Home, End,
            // ...) or `ESC [ 1 ; 5 <letter>` (Ctrl+arrow). Consume up to the
            // terminator and classify what we can.
            let mut digits = vec![b2];
            let mut terminator;
            loop {
                let Some(b) = read_one_byte(r) else {
                    return EscapeSeq::Unknown;
                };
                terminator = b;
                if b == b'~' || b.is_ascii_alphabetic() {
                    break;
                }
                digits.push(b);
            }
            if terminator == b'~' {
                match digits.first() {
                    Some(b'3') => EscapeSeq::Delete,
                    Some(b'1') | Some(b'7') => EscapeSeq::Home,
                    _ => EscapeSeq::Unknown,
                }
            } else if terminator == b'D' {
                EscapeSeq::WordLeft
            } else if terminator == b'C' {
                EscapeSeq::WordRight
            } else {
                EscapeSeq::Unknown
            }
        }
        b'4' => {
            // `ESC [ 4 ~` = End on some terminals.
            match read_one_byte(r) {
                Some(b'~') => EscapeSeq::End,
                _ => EscapeSeq::Unknown,
            }
        }
        _ => EscapeSeq::Unknown,
    }
}

/// Repaints the current line: return to column 0, write the prompt and
/// buffer, clear to end of line, then position the cursor.
fn redraw<W: Write>(stdout: &mut W, prompt: &str, buf: &Buffer) {
    let content: String = buf.chars.iter().collect();
    let _ = write!(stdout, "\r{prompt}{content}\x1b[K");
    // Move cursor back to `prompt.len() + buf.cursor` columns from start.
    // We already wrote the whole buffer, so move left by the number of
    // chars after the cursor, then reposition with an absolute move from
    // column 0 for simplicity and correctness with wide prompts.
    let _ = write!(stdout, "\r");
    let prompt_width = crate::output::str_width(prompt);
    let buf_prefix: String = buf.chars[..buf.cursor].iter().collect();
    let col = prompt_width + crate::output::str_width(&buf_prefix);
    if col > 0 {
        let _ = write!(stdout, "\x1b[{col}C");
    }
    let _ = stdout.flush();
}

/// Handles a Tab keypress: completes the word before the cursor.
fn handle_completion<W: Write>(
    buf: &mut Buffer,
    complete: &dyn Fn(&str) -> Vec<String>,
    stdout: &mut W,
    prompt: &str,
) {
    let start = word_before_cursor(&buf.chars, buf.cursor);
    let word: String = buf.chars[start..buf.cursor].iter().collect();
    let candidates = complete(&word);
    match candidates.len() {
        0 => {}
        1 => {
            buf.replace_range(start, buf.cursor, &candidates[0]);
        }
        _ => {
            let common = longest_common_prefix(&candidates);
            if common.len() > word.len() {
                buf.replace_range(start, buf.cursor, &common);
            } else {
                // No progress from the prefix: show the candidates.
                let _ = write!(stdout, "\r\n");
                let mut col = 0;
                for candidate in &candidates {
                    if col + candidate.len() + 2 > 80 {
                        let _ = write!(stdout, "\r\n");
                        col = 0;
                    }
                    let _ = write!(stdout, "{candidate}  ");
                    col += candidate.len() + 2;
                }
                let _ = write!(stdout, "\r\n");
                let _ = stdout.flush();
                redraw(stdout, prompt, buf);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_insert_and_cursor_advances() {
        let mut buf = Buffer::default();
        buf.insert_char('a');
        buf.insert_char('b');
        buf.insert_char('c');
        assert_eq!(buf.as_string(), "abc");
        assert_eq!(buf.cursor, 3);
    }

    #[test]
    fn buffer_insert_utf8_multibyte_as_one_char() {
        let mut buf = Buffer::default();
        for c in "héllo\u{1F600}".chars() {
            buf.insert_char(c);
        }
        assert_eq!(buf.as_string(), "héllo\u{1F600}");
        // 6 chars, not counting bytes.
        assert_eq!(buf.cursor, 6);
        assert_eq!(buf.chars.len(), 6);
    }

    #[test]
    fn buffer_backspace_deletes_one_char_not_one_byte() {
        let mut buf = Buffer::from_str("caf\u{e9}"); // "café", cursor at end
        assert!(buf.delete_before());
        assert_eq!(buf.as_string(), "caf");
    }

    #[test]
    fn buffer_backspace_at_start_is_noop() {
        let mut buf = Buffer::from_str("abc");
        buf.move_home();
        assert!(!buf.delete_before());
        assert_eq!(buf.as_string(), "abc");
    }

    #[test]
    fn buffer_delete_at_cursor() {
        let mut buf = Buffer::from_str("abc");
        buf.move_home();
        assert!(buf.delete_at_cursor());
        assert_eq!(buf.as_string(), "bc");
        assert_eq!(buf.cursor, 0);
    }

    #[test]
    fn buffer_delete_at_cursor_at_end_is_noop() {
        let mut buf = Buffer::from_str("abc");
        assert!(!buf.delete_at_cursor());
    }

    #[test]
    fn buffer_kill_to_start_and_end() {
        let mut buf = Buffer::from_str("hello world");
        buf.cursor = 5; // after "hello"
        let killed = buf.kill_to_start();
        assert_eq!(killed, "hello");
        assert_eq!(buf.as_string(), " world");
        assert_eq!(buf.cursor, 0);

        let mut buf2 = Buffer::from_str("hello world");
        buf2.cursor = 5;
        let killed2 = buf2.kill_to_end();
        assert_eq!(killed2, " world");
        assert_eq!(buf2.as_string(), "hello");
    }

    #[test]
    fn buffer_word_movement() {
        let mut buf = Buffer::from_str("select * from t");
        buf.move_home();
        buf.move_word_right();
        assert_eq!(buf.cursor, 6); // "select"
        buf.move_word_right();
        assert_eq!(buf.cursor, 8); // "select *"
        buf.move_word_left();
        assert_eq!(buf.cursor, 7); // start of "*"
    }

    #[test]
    fn buffer_delete_word_before() {
        let mut buf = Buffer::from_str("select * from t");
        let removed = buf.delete_word_before();
        assert_eq!(removed, "t");
        assert_eq!(buf.as_string(), "select * from ");

        let mut buf2 = Buffer::from_str("select foo");
        let removed2 = buf2.delete_word_before();
        assert_eq!(removed2, "foo");
        assert_eq!(buf2.as_string(), "select ");
    }

    #[test]
    fn buffer_replace_range() {
        let mut buf = Buffer::from_str("select fr");
        buf.replace_range(7, 9, "from");
        assert_eq!(buf.as_string(), "select from");
        assert_eq!(buf.cursor, 11);
    }

    #[test]
    fn word_before_cursor_splits_on_whitespace_and_punctuation() {
        let chars: Vec<char> = "select foo(bar, baz".chars().collect();
        let cursor = chars.len();
        let start = word_before_cursor(&chars, cursor);
        let word: String = chars[start..cursor].iter().collect();
        assert_eq!(word, "baz");

        let chars2: Vec<char> = "select foo(ba".chars().collect();
        let start2 = word_before_cursor(&chars2, chars2.len());
        let word2: String = chars2[start2..].iter().collect();
        assert_eq!(word2, "ba");
    }

    #[test]
    fn word_before_cursor_at_start_of_line() {
        let chars: Vec<char> = "sel".chars().collect();
        let start = word_before_cursor(&chars, chars.len());
        assert_eq!(start, 0);
    }

    #[test]
    fn longest_common_prefix_basic() {
        let items = vec!["select".to_string(), "sel".to_string(), "selection".to_string()];
        assert_eq!(longest_common_prefix(&items), "sel");
    }

    #[test]
    fn longest_common_prefix_no_common() {
        let items = vec!["abc".to_string(), "xyz".to_string()];
        assert_eq!(longest_common_prefix(&items), "");
    }

    #[test]
    fn longest_common_prefix_single_item() {
        let items = vec!["only".to_string()];
        assert_eq!(longest_common_prefix(&items), "only");
    }

    #[test]
    fn longest_common_prefix_empty_input() {
        let items: Vec<String> = vec![];
        assert_eq!(longest_common_prefix(&items), "");
    }

    #[test]
    fn longest_common_prefix_respects_char_boundaries() {
        // Ensure we don't split multi-byte chars: both share "café".
        let items = vec!["café1".to_string(), "café2".to_string()];
        assert_eq!(longest_common_prefix(&items), "café");
    }

    #[test]
    fn history_escape_round_trip_simple() {
        let original = "select 1;";
        let escaped = escape_history_entry(original);
        assert_eq!(escaped, "select 1;");
        assert_eq!(unescape_history_entry(&escaped), original);
    }

    #[test]
    fn history_escape_round_trip_multiline() {
        let original = "select *\nfrom t\nwhere x = 1;";
        let escaped = escape_history_entry(original);
        assert!(!escaped.contains('\n'));
        assert_eq!(unescape_history_entry(&escaped), original);
    }

    #[test]
    fn history_escape_round_trip_backslash() {
        let original = "select '\\n' as literal_backslash_n";
        let escaped = escape_history_entry(original);
        assert_eq!(unescape_history_entry(&escaped), original);
    }

    #[test]
    fn history_escape_handles_mixed_content() {
        let original = "a\\b\nc\\nd";
        let escaped = escape_history_entry(original);
        assert_eq!(unescape_history_entry(&escaped), original);
    }

    #[test]
    fn editor_add_history_skips_empty_and_consecutive_duplicates() {
        let mut editor = Editor { history: Vec::new(), path: None, interactive: false };
        editor.add_history("select 1;");
        editor.add_history("");
        // Consecutive duplicate: skipped.
        editor.add_history("select 1;");
        editor.add_history("select 2;");
        editor.add_history("select 2;");
        assert_eq!(editor.history, vec!["select 1;", "select 2;"]);
        // Not consecutive anymore: this one is kept.
        editor.add_history("select 1;");
        assert_eq!(editor.history, vec!["select 1;", "select 2;", "select 1;"]);
    }

    #[test]
    fn editor_history_load_and_save_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("ahiru_lineedit_test_{}.history", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let mut editor = Editor::new(Some(path.clone()));
        editor.add_history("select 1;");
        editor.add_history("select *\nfrom t;");
        editor.save_history();

        let reloaded = Editor::new(Some(path.clone()));
        assert_eq!(reloaded.history, vec!["select 1;", "select *\nfrom t;"]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn editor_save_history_caps_at_limit() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("ahiru_lineedit_test_cap_{}.history", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let mut editor =
            Editor { history: Vec::new(), path: Some(path.clone()), interactive: false };
        for i in 0..1500 {
            editor.add_history(&format!("stmt {i}"));
        }
        editor.save_history();

        let reloaded = Editor::new(Some(path.clone()));
        assert_eq!(reloaded.history.len(), 1000);
        assert_eq!(reloaded.history[0], "stmt 500");
        assert_eq!(reloaded.history[999], "stmt 1499");

        let _ = std::fs::remove_file(&path);
    }
}
