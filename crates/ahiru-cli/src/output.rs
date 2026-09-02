//! Result rendering: output modes, `.headers`, `.nullvalue`, `.separator`,
//! `.maxrows`, `.maxwidth`.
//!
//! Rows arrive one at a time and most modes stream them straight out; only the
//! boxed modes (`Box`/`Duckbox`) buffer because they cannot know the column
//! widths until every row has been seen.

use std::io::{self, Write};

use ahiru_core::vector::{Field, Ty, Value};

/// Output format, i.e. DuckDB's `.mode`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// Boxed table with a row-count footer. The interactive default.
    Duckbox,
    /// Boxed table, no footer, no truncation.
    Box,
    Csv,
    /// Tab separated. The non-interactive default (what the CLI always did).
    Tsv,
    /// One JSON array of objects.
    Json,
    /// One JSON object per line.
    JsonLines,
    /// `column = value` blocks, one row per block.
    Line,
    /// Values joined by `separator` (`.mode list`).
    List,
    Markdown,
    /// `INSERT INTO <table> VALUES (...);` statements.
    Insert,
    /// Discards output (`.mode trash`), for timing a query.
    Trash,
}

impl Mode {
    /// Parses a `.mode` argument, case-insensitively. Accepts a few aliases
    /// for the same mode (e.g. `tabs` for `Tsv`, `md` for `Markdown`).
    /// Returns `None` for anything unrecognized so the caller can report
    /// "unknown option".
    pub fn parse(s: &str) -> Option<Mode> {
        Some(match s.to_ascii_lowercase().as_str() {
            "duckbox" => Mode::Duckbox,
            "box" => Mode::Box,
            "csv" => Mode::Csv,
            "tsv" | "tabs" => Mode::Tsv,
            "json" => Mode::Json,
            "jsonlines" | "ndjson" => Mode::JsonLines,
            "line" | "lines" => Mode::Line,
            "list" => Mode::List,
            "markdown" | "md" => Mode::Markdown,
            "insert" => Mode::Insert,
            "trash" => Mode::Trash,
            _ => return None,
        })
    }

    /// The column separator this mode expects when the user hasn't chosen
    /// one: `,` for CSV, `|` for `list` (matching DuckDB), a tab otherwise.
    pub fn default_separator(self) -> &'static str {
        match self {
            Mode::Csv => ",",
            Mode::List => "|",
            _ => "\t",
        }
    }

    /// Canonical lowercase name, as accepted back by `parse`.
    pub fn name(self) -> &'static str {
        match self {
            Mode::Duckbox => "duckbox",
            Mode::Box => "box",
            Mode::Csv => "csv",
            Mode::Tsv => "tsv",
            Mode::Json => "json",
            Mode::JsonLines => "jsonlines",
            Mode::Line => "line",
            Mode::List => "list",
            Mode::Markdown => "markdown",
            Mode::Insert => "insert",
            Mode::Trash => "trash",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub mode: Mode,
    pub header: bool,
    pub separator: String,
    pub null: String,
    /// Max rows shown in `duckbox` mode before the middle is elided.
    /// 0 disables truncation. `box` never truncates rows (it has no footer
    /// to report the true count), matching DuckDB.
    pub maxrows: usize,
    /// Max total width of a boxed table before middle columns are elided.
    /// 0 disables truncation.
    pub maxwidth: usize,
    /// Table name used by `Mode::Insert`.
    pub insert_table: String,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            mode: Mode::Tsv,
            header: true,
            separator: "\t".to_string(),
            null: crate::render::DEFAULT_NULL.to_string(),
            maxrows: 40,
            maxwidth: 0,
            insert_table: "tbl".to_string(),
        }
    }
}

/// One column in the boxed-mode layout: either a real schema column or the
/// single synthetic "elided columns went here" marker.
enum Col {
    Real(usize),
    Elision,
}

/// One row in the boxed-mode layout: either a buffered data row or the
/// synthetic "elided rows went here" marker (rendered as a row of `·`).
enum DisplayRow<'a> {
    Data(&'a [String]),
    Elision,
}

/// Writes one result set in the configured mode.
pub struct Writer<'a> {
    out: &'a mut dyn Write,
    settings: Settings,
    names: Vec<String>,
    types: Vec<Ty>,
    /// True row count, kept even when the boxed modes only display a subset.
    rows: usize,
    /// Rendered cells, buffered for `Box`/`Duckbox` (need column widths).
    box_rows: Vec<Vec<String>>,
}

impl<'a> Writer<'a> {
    pub fn new(out: &'a mut dyn Write, settings: Settings) -> Writer<'a> {
        Writer {
            out,
            settings,
            names: Vec::new(),
            types: Vec::new(),
            rows: 0,
            box_rows: Vec::new(),
        }
    }

    /// Starts a result set. Must be called before `row`.
    pub fn begin(&mut self, schema: &[Field]) -> io::Result<()> {
        self.names = schema.iter().map(|f| f.name.clone()).collect();
        self.types = schema.iter().map(|f| f.ty).collect();
        self.rows = 0;
        self.box_rows.clear();
        match self.settings.mode {
            Mode::Tsv | Mode::List => self.write_header_delim(),
            Mode::Csv => self.write_header_csv(),
            Mode::Markdown => self.write_header_markdown(),
            _ => Ok(()),
        }
    }

    pub fn row(&mut self, values: &[Value]) -> io::Result<()> {
        self.rows += 1;
        match self.settings.mode {
            Mode::Trash => Ok(()),
            Mode::Tsv | Mode::List => self.write_row_delim(values),
            Mode::Csv => self.write_row_csv(values),
            Mode::Line => self.write_row_line(values),
            Mode::Markdown => self.write_row_markdown(values),
            Mode::Insert => self.write_row_insert(values),
            Mode::JsonLines => self.write_row_jsonlines(values),
            Mode::Json => self.write_row_json(values),
            Mode::Box | Mode::Duckbox => {
                let cells: Vec<String> = values
                    .iter()
                    .zip(&self.types)
                    .map(|(v, ty)| crate::render::render(v, *ty, &self.settings.null))
                    .collect();
                self.box_rows.push(cells);
                Ok(())
            }
        }
    }

    /// Flushes anything buffered and writes the footer, if the mode has one.
    pub fn finish(&mut self) -> io::Result<()> {
        match self.settings.mode {
            Mode::Box => self.finish_box(false)?,
            Mode::Duckbox => self.finish_box(true)?,
            Mode::Json => self.finish_json()?,
            _ => {}
        }
        self.out.flush()
    }

    // ---- Tsv / List (identical shape; only `separator` differs) ----

    fn write_header_delim(&mut self) -> io::Result<()> {
        if !self.settings.header {
            return Ok(());
        }
        writeln!(self.out, "{}", self.names.join(&self.settings.separator))
    }

    fn write_row_delim(&mut self, values: &[Value]) -> io::Result<()> {
        let cells: Vec<String> = values
            .iter()
            .zip(&self.types)
            .map(|(v, ty)| crate::render::render(v, *ty, &self.settings.null))
            .collect();
        writeln!(self.out, "{}", cells.join(&self.settings.separator))
    }

    // ---- Csv: same shape as Tsv/List, plus RFC 4180 quoting ----

    fn write_header_csv(&mut self) -> io::Result<()> {
        if !self.settings.header {
            return Ok(());
        }
        let cells: Vec<String> =
            self.names.iter().map(|n| csv_quote(n, &self.settings.separator)).collect();
        writeln!(self.out, "{}", cells.join(&self.settings.separator))
    }

    fn write_row_csv(&mut self, values: &[Value]) -> io::Result<()> {
        let cells: Vec<String> = values
            .iter()
            .zip(&self.types)
            .map(|(v, ty)| {
                let text = crate::render::render(v, *ty, &self.settings.null);
                csv_quote(&text, &self.settings.separator)
            })
            .collect();
        writeln!(self.out, "{}", cells.join(&self.settings.separator))
    }

    // ---- Line: `col = value` blocks, blank line between (not after) rows ----

    fn write_row_line(&mut self, values: &[Value]) -> io::Result<()> {
        if self.rows > 1 {
            writeln!(self.out)?;
        }
        let width = self.names.iter().map(|n| n.chars().count()).max().unwrap_or(0);
        for (name, (v, ty)) in self.names.iter().zip(values.iter().zip(&self.types)) {
            let cell = crate::render::render(v, *ty, &self.settings.null);
            writeln!(self.out, "{name:width$} = {cell}")?;
        }
        Ok(())
    }

    // ---- Markdown ----

    fn write_header_markdown(&mut self) -> io::Result<()> {
        if !self.settings.header {
            return Ok(());
        }
        let cells: Vec<String> = self.names.iter().map(|n| md_escape(n)).collect();
        writeln!(self.out, "| {} |", cells.join(" | "))?;
        let seps: Vec<&str> = self.names.iter().map(|_| "---").collect();
        writeln!(self.out, "|{}|", seps.join("|"))
    }

    fn write_row_markdown(&mut self, values: &[Value]) -> io::Result<()> {
        let cells: Vec<String> = values
            .iter()
            .zip(&self.types)
            .map(|(v, ty)| md_escape(&crate::render::render(v, *ty, &self.settings.null)))
            .collect();
        writeln!(self.out, "| {} |", cells.join(" | "))
    }

    // ---- Insert ----

    fn write_row_insert(&mut self, values: &[Value]) -> io::Result<()> {
        let parts: Vec<String> = values
            .iter()
            .zip(&self.types)
            .map(|(v, ty)| {
                if matches!(v, Value::Null) {
                    "NULL".to_string()
                } else if *ty == Ty::Blob {
                    if let Value::Bytes(b) = v {
                        const HEX: &[u8; 16] = b"0123456789abcdef";
                        let mut hex = String::with_capacity(b.len() * 2);
                        for byte in b {
                            hex.push(HEX[(byte >> 4) as usize] as char);
                            hex.push(HEX[(byte & 0xf) as usize] as char);
                        }
                        format!("X'{hex}'")
                    } else {
                        "NULL".to_string()
                    }
                } else if matches!(ty, Ty::Float | Ty::Double)
                    && matches!(v, Value::F64(f) if !f.is_finite())
                {
                    // Bare `inf`/`NaN` tokens are parsed as column names. A
                    // quoted spelling is cast back to the target floating
                    // type when the generated INSERT statement is replayed.
                    let text = crate::render::render(v, *ty, "NULL");
                    format!("'{}'", text.replace('\'', "''"))
                } else if *ty == Ty::Boolean || crate::render::is_numeric(*ty) {
                    crate::render::render(v, *ty, "NULL")
                } else {
                    let text = crate::render::render(v, *ty, "NULL");
                    format!("'{}'", text.replace('\'', "''"))
                }
            })
            .collect();
        writeln!(
            self.out,
            "INSERT INTO {} VALUES ({});",
            insert_target(&self.settings.insert_table),
            parts.join(", ")
        )
    }

    // ---- Json / JsonLines ----

    fn write_row_jsonlines(&mut self, values: &[Value]) -> io::Result<()> {
        let frag = self.json_row_fragment(values);
        writeln!(self.out, "{frag}")
    }

    /// Streams one row into a JSON array. Prefixing rows after the first with a
    /// comma keeps the exact output shape without retaining every rendered row
    /// until `finish`.
    fn write_row_json(&mut self, values: &[Value]) -> io::Result<()> {
        let frag = self.json_row_fragment(values);
        if self.rows == 1 {
            write!(self.out, "[\n\t{frag}")
        } else {
            write!(self.out, ",\n\t{frag}")
        }
    }

    /// Renders one row as a compact `{"col":value,...}` fragment. Shared by
    /// `Json` (streamed as an array) and `JsonLines` (one object per line).
    fn json_row_fragment(&self, values: &[Value]) -> String {
        let mut out = String::from("{");
        for (i, (v, ty)) in values.iter().zip(&self.types).enumerate() {
            if i > 0 {
                out.push(',');
            }
            json_escape_into(&self.names[i], &mut out);
            out.push(':');
            out.push_str(&json_cell_value(v, *ty));
        }
        out.push('}');
        out
    }

    fn finish_json(&mut self) -> io::Result<()> {
        if self.rows == 0 {
            writeln!(self.out, "[]")
        } else {
            writeln!(self.out, "\n]")
        }
    }

    // ---- Box / Duckbox ----

    /// Renders the buffered rows as a Unicode box-drawing table.
    ///
    /// `duck` selects between the two boxed modes: `Duckbox` additionally
    /// shows a second header line with each column's type, a row-count
    /// footer, and (when `settings.maxrows`/`maxwidth` are set) truncates
    /// rows/columns that don't fit; plain `Box` shows neither and never
    /// truncates.
    fn finish_box(&mut self, duck: bool) -> io::Result<()> {
        let n = self.names.len();
        let total_rows = self.box_rows.len();

        // --- Row truncation: split `maxrows` between the start and end, with
        // a single elision marker row spliced into the middle. For an odd
        // limit, the start gets the extra row so exactly `maxrows` data rows
        // remain visible (including when the limit is one). ---
        let row_trunc = duck && self.settings.maxrows > 0 && total_rows > self.settings.maxrows;
        let tail_rows = self.settings.maxrows / 2;
        let head_rows = tail_rows + self.settings.maxrows % 2;
        let display_rows: Vec<DisplayRow> = if row_trunc {
            let mut v = Vec::with_capacity(self.settings.maxrows + 1);
            v.extend(self.box_rows[..head_rows].iter().map(|r| DisplayRow::Data(r.as_slice())));
            v.push(DisplayRow::Elision);
            v.extend(
                self.box_rows[total_rows - tail_rows..]
                    .iter()
                    .map(|r| DisplayRow::Data(r.as_slice())),
            );
            v
        } else {
            self.box_rows.iter().map(|r| DisplayRow::Data(r.as_slice())).collect()
        };
        let shown_rows = if row_trunc { self.settings.maxrows } else { total_rows };

        // --- Column content widths, across the header, the type line (duck
        // only) and every displayed data row. ---
        let mut col_width: Vec<usize> = self.names.iter().map(|n| str_width(n)).collect();
        if duck {
            for (w, ty) in col_width.iter_mut().zip(&self.types) {
                *w = (*w).max(str_width(&ty.name().to_ascii_lowercase()));
            }
        }
        for row in &display_rows {
            if let DisplayRow::Data(cells) = row {
                for (w, cell) in col_width.iter_mut().zip(cells.iter()) {
                    *w = (*w).max(str_width(cell));
                }
            }
        }
        for w in &mut col_width {
            *w = (*w).max(1);
        }

        // --- Column truncation: drop columns from the middle of the
        // surviving set (index `len/2`) one at a time until the table fits
        // `maxwidth`, always keeping at least the first and last column.
        // This does not always yield the narrowest possible table when many
        // columns must go, but the removal order is simple and predictable:
        // it eats into the table from the center outward. ---
        let mut keep: Vec<usize> = (0..n).collect();
        if duck && self.settings.maxwidth > 0 {
            while keep.len() > 2
                && table_width(&keep, &col_width, keep.len() < n) > self.settings.maxwidth
            {
                let mid = keep.len() / 2;
                keep.remove(mid);
            }
        }
        let col_trunc = keep.len() < n;
        let shown_cols = keep.len();

        // Splice a single "…" elision column in where columns were dropped.
        let mut visible: Vec<Col> = Vec::with_capacity(keep.len() + 1);
        for (i, &orig) in keep.iter().enumerate() {
            visible.push(Col::Real(orig));
            if i + 1 < keep.len() && keep[i + 1] != orig + 1 {
                visible.push(Col::Elision);
            }
        }

        let col_disp_width = |c: &Col| -> usize {
            match c {
                Col::Real(i) => col_width[*i],
                Col::Elision => 1,
            }
        };

        if visible.is_empty() {
            if duck {
                let footer = format!("{total_rows} row{}", if total_rows == 1 { "" } else { "s" });
                writeln!(self.out, "{footer}")?;
            }
            return Ok(());
        }

        // --- Borders ---
        let mut top = String::from("┌");
        let mut mid = String::from("├");
        let mut bot = String::from("└");
        for (i, c) in visible.iter().enumerate() {
            let w = col_disp_width(c);
            let bar = "─".repeat(w + 2);
            top.push_str(&bar);
            mid.push_str(&bar);
            bot.push_str(&bar);
            let last = i + 1 == visible.len();
            top.push_str(if last { "┐" } else { "┬" });
            mid.push_str(if last { "┤" } else { "┼" });
            bot.push_str(if last { "┘" } else { "┴" });
        }
        writeln!(self.out, "{top}")?;

        // --- Header (and, for duckbox, the type line) ---
        let mut header = String::from("│");
        let mut type_line = String::from("│");
        for c in &visible {
            let w = col_disp_width(c);
            match c {
                Col::Real(i) => {
                    let right = crate::render::is_numeric(self.types[*i]);
                    header.push_str(&pad_cell(&self.names[*i], w, right));
                    type_line.push_str(&pad_cell(
                        &self.types[*i].name().to_ascii_lowercase(),
                        w,
                        right,
                    ));
                }
                Col::Elision => {
                    header.push_str(&pad_center("…", w));
                    type_line.push_str(&pad_center("…", w));
                }
            }
            header.push('│');
            type_line.push('│');
        }
        writeln!(self.out, "{header}")?;
        if duck {
            writeln!(self.out, "{type_line}")?;
        }
        writeln!(self.out, "{mid}")?;

        // --- Data rows ---
        for row in &display_rows {
            let mut line = String::from("│");
            for c in &visible {
                let w = col_disp_width(c);
                match (row, c) {
                    (DisplayRow::Elision, _) => line.push_str(&pad_center("·", w)),
                    (DisplayRow::Data(_), Col::Elision) => line.push_str(&pad_center("…", w)),
                    (DisplayRow::Data(cells), Col::Real(i)) => {
                        let right = crate::render::is_numeric(self.types[*i]);
                        line.push_str(&pad_cell(&cells[*i], w, right));
                    }
                }
                line.push('│');
            }
            writeln!(self.out, "{line}")?;
        }
        writeln!(self.out, "{bot}")?;

        // --- Footer (duckbox only) ---
        if duck {
            let mut footer = format!("{total_rows} row{}", if total_rows == 1 { "" } else { "s" });
            if row_trunc {
                footer.push_str(&format!(" ({shown_rows} shown)"));
            }
            writeln!(self.out, "{footer}")?;
            if col_trunc {
                writeln!(self.out, "({n} columns, {shown_cols} shown)")?;
            }
        }
        Ok(())
    }
}

/// Render the target of generated INSERT statements as a safe SQL identifier.
/// Keep ordinary names unquoted for compatibility, but quote qualified parts
/// that contain punctuation, spaces, quotes, or a reserved word. The table name
/// comes from `.mode insert TABLE`, so it is user-controlled output text rather
/// than trusted SQL.
fn insert_target(name: &str) -> String {
    name.split('.').map(insert_identifier_part).collect::<Vec<_>>().join(".")
}

fn insert_identifier_part(part: &str) -> String {
    let mut chars = part.chars();
    let simple = chars.next().is_some_and(|c| c == '_' || c.is_ascii_alphabetic())
        && chars.all(|c| c == '_' || c.is_ascii_alphanumeric());
    const RESERVED: &[&str] = &[
        "as",
        "by",
        "in",
        "is",
        "on",
        "or",
        "all",
        "and",
        "asc",
        "end",
        "not",
        "case",
        "cast",
        "desc",
        "else",
        "from",
        "full",
        "join",
        "last",
        "left",
        "like",
        "null",
        "show",
        "then",
        "true",
        "when",
        "with",
        "cross",
        "false",
        "first",
        "group",
        "ilike",
        "inner",
        "limit",
        "nulls",
        "order",
        "outer",
        "right",
        "union",
        "where",
        "escape",
        "except",
        "exists",
        "having",
        "offset",
        "select",
        "tables",
        "window",
        "between",
        "explain",
        "qualify",
        "describe",
        "distinct",
        "intersect",
        "if",
        "to",
        "add",
        "drop",
        "view",
        "alter",
        "table",
        "column",
        "create",
        "rename",
        "default",
        "replace",
        "set",
        "into",
        "delete",
        "insert",
        "update",
        "values",
        "copy",
    ];
    let lower = part.to_ascii_lowercase();
    let reserved = RESERVED.contains(&lower.as_str());
    if simple && !reserved {
        return part.to_string();
    }
    format!("\"{}\"", part.replace('"', "\"\""))
}

/// Quotes a cell per RFC 4180 if it contains the separator, a `"`, or a
/// line-ending character; doubles any internal `"`.
fn csv_quote(s: &str, separator: &str) -> String {
    let needs_quoting =
        s.contains(separator) || s.contains('"') || s.contains('\r') || s.contains('\n');
    if !needs_quoting {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        if c == '"' {
            out.push('"');
        }
        out.push(c);
    }
    out.push('"');
    out
}

/// Escapes `|` for a Markdown table cell.
fn md_escape(s: &str) -> String {
    s.replace('|', "\\|")
}

/// The JSON fragment to embed for one cell: `null` for SQL NULL (never
/// `settings.null` — JSON has its own null spelling), a bare literal for
/// numeric/boolean types, raw (unescaped) text for a `Ty::Json` column whose
/// text looks like actual JSON, and a JSON string otherwise.
fn json_cell_value(v: &Value, ty: Ty) -> String {
    if matches!(v, Value::Null) {
        return "null".to_string();
    }
    if matches!(v, Value::F64(x) if !x.is_finite()) {
        return "null".to_string();
    }
    if ty == Ty::Boolean || crate::render::is_numeric(ty) {
        return crate::render::render(v, ty, "null");
    }
    let text = crate::render::render(v, ty, "null");
    if ty == Ty::Json && looks_like_json(&text) {
        return text;
    }
    let mut out = String::new();
    json_escape_into(&text, &mut out);
    out
}

/// Cheap plausibility check for whether a `Ty::Json` column's text is
/// actually JSON (so it can be spliced in raw) rather than something else
/// stored under a JSON column that wouldn't parse (so it must be quoted as a
/// string instead). Checks only the leading token, not a full parse.
fn looks_like_json(s: &str) -> bool {
    match s.chars().next() {
        Some(c) if c == '{' || c == '[' || c == '"' || c == '-' || c.is_ascii_digit() => true,
        _ => s.starts_with("true") || s.starts_with("false") || s.starts_with("null"),
    }
}

/// Appends a JSON-escaped, quoted string to `out`.
fn json_escape_into(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Total rendered width of a boxed table given the set of columns to keep
/// (original indices) and each column's content width, optionally including
/// one synthetic 1-wide "…" elision column. Each column contributes its
/// content width plus 2 (one padding space either side) plus 1 (its right
/// border); there's one more border on the left of the whole table.
fn table_width(keep: &[usize], col_width: &[usize], elided: bool) -> usize {
    let mut total = 1;
    for &i in keep {
        total += col_width[i] + 3;
    }
    if elided {
        total += 1 + 3;
    }
    total
}

/// True for a code point that advances the cursor by nothing: the common
/// combining-mark blocks, the zero-width format controls, and the variation
/// selectors.
///
/// This is deliberately a short list of whole blocks rather than a full
/// Unicode width table — the CLI only needs the cases a user is likely to
/// paste into a terminal (`e` + U+0301, Devanagari/Arabic vowel signs,
/// emoji variation selectors).
fn is_zero_width(cp: u32) -> bool {
    matches!(
        cp,
        // Combining diacritical marks (base, extended, supplement, symbols)
        0x0300..=0x036f | 0x1ab0..=0x1aff | 0x1dc0..=0x1dff | 0x20d0..=0x20f0
        // Combining Cyrillic, Hebrew points, Arabic marks
        | 0x0483..=0x0489
        | 0x0591..=0x05bd | 0x05bf | 0x05c1..=0x05c2 | 0x05c4..=0x05c5 | 0x05c7
        | 0x0610..=0x061a | 0x064b..=0x065f | 0x0670 | 0x06d6..=0x06dc
        // Devanagari and Thai above/below signs
        | 0x0900..=0x0902 | 0x093a | 0x093c | 0x0941..=0x0948 | 0x0951..=0x0957
        | 0x0e31 | 0x0e34..=0x0e3a | 0x0e47..=0x0e4e
        // Zero-width format controls and bidi embedding controls
        | 0x200c..=0x200f | 0x202a..=0x202e
        // Combining half marks and variation selectors
        | 0xfe00..=0xfe0f | 0xfe20..=0xfe2f | 0xe0100..=0xe01ef
    )
}

/// Returns the display column width of a single character in a monospace terminal.
pub(crate) fn char_width(c: char) -> usize {
    let cp = c as u32;
    // Zero-advance: control characters, plus the combining marks and format
    // controls that render on top of the previous cell. Counting a combining
    // mark as 1 makes a boxed cell claim more columns than the terminal draws,
    // and the border comes out short.
    if cp < 0x20 || (0x7f..0xa0).contains(&cp) || cp == 0x200b || cp == 0xfeff || is_zero_width(cp)
    {
        0
    } else if (0x1100..=0x115f).contains(&cp)
        || (0x2329..=0x232a).contains(&cp)
        || (0x2e80..=0x303e).contains(&cp)
        || (0x3040..=0xa4cf).contains(&cp)
        || (0xac00..=0xd7a3).contains(&cp)
        || (0xf900..=0xfaff).contains(&cp)
        || (0xfe10..=0xfe19).contains(&cp)
        || (0xfe30..=0xfe6f).contains(&cp)
        || (0xff00..=0xff60).contains(&cp)
        || (0xffe0..=0xffe6).contains(&cp)
        || (0x1f300..=0x1f64f).contains(&cp)
        || (0x1f680..=0x1f6ff).contains(&cp)
        || (0x1f900..=0x1f9ff).contains(&cp)
        || (0x20000..=0x3ffff).contains(&cp)
    {
        2
    } else {
        1
    }
}

/// Returns the display column width of a string in a monospace terminal.
pub(crate) fn str_width(s: &str) -> usize {
    s.chars().map(char_width).sum()
}

/// Pads `text` to `width` with one space of padding on each side, right- or
/// left-aligned within that width.
fn pad_cell(text: &str, width: usize, right_align: bool) -> String {
    let len = str_width(text);
    let fill = width.saturating_sub(len);
    if right_align {
        format!(" {}{text} ", " ".repeat(fill))
    } else {
        format!(" {text}{} ", " ".repeat(fill))
    }
}

/// Pads a short marker (`·` or `…`) centered within `width`, for elision rows
/// and the elision column.
fn pad_center(marker: &str, width: usize) -> String {
    let len = str_width(marker);
    let fill = width.saturating_sub(len);
    let left = fill / 2;
    let right = fill - left;
    format!(" {}{marker}{} ", " ".repeat(left), " ".repeat(right))
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use super::*;

    struct SharedBuffer(Rc<RefCell<Vec<u8>>>);

    impl Write for SharedBuffer {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.borrow_mut().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn field(name: &str, ty: Ty) -> Field {
        Field::new(name, ty, true)
    }

    fn writer<'a>(out: &'a mut Vec<u8>, settings: Settings) -> Writer<'a> {
        Writer::new(out, settings)
    }

    fn as_str(buf: &[u8]) -> &str {
        std::str::from_utf8(buf).unwrap()
    }

    #[test]
    fn mode_parse_round_trip() {
        let modes = [
            Mode::Duckbox,
            Mode::Box,
            Mode::Csv,
            Mode::Tsv,
            Mode::Json,
            Mode::JsonLines,
            Mode::Line,
            Mode::List,
            Mode::Markdown,
            Mode::Insert,
            Mode::Trash,
        ];
        for m in modes {
            assert_eq!(Mode::parse(m.name()), Some(m), "round trip for {}", m.name());
        }
        // Case-insensitivity and aliases.
        assert_eq!(Mode::parse("DUCKBOX"), Some(Mode::Duckbox));
        assert_eq!(Mode::parse("Tabs"), Some(Mode::Tsv));
        assert_eq!(Mode::parse("ndjson"), Some(Mode::JsonLines));
        assert_eq!(Mode::parse("md"), Some(Mode::Markdown));
        assert_eq!(Mode::parse("lines"), Some(Mode::Line));
        assert_eq!(Mode::parse("bogus"), None);
    }

    #[test]
    fn tsv_is_byte_exact() {
        let schema = [field("name", Ty::Varchar), field("n", Ty::Int)];
        let mut buf = Vec::new();
        let mut w = writer(&mut buf, Settings { mode: Mode::Tsv, ..Settings::default() });
        w.begin(&schema).unwrap();
        w.row(&[Value::Bytes(b"alice".to_vec()), Value::I32(1)]).unwrap();
        w.row(&[Value::Bytes(b"bob".to_vec()), Value::I32(2)]).unwrap();
        w.finish().unwrap();
        assert_eq!(as_str(&buf), "name\tn\nalice\t1\nbob\t2\n");
    }

    #[test]
    fn tsv_no_header_when_disabled() {
        let schema = [field("n", Ty::Int)];
        let mut buf = Vec::new();
        let mut w =
            writer(&mut buf, Settings { mode: Mode::Tsv, header: false, ..Settings::default() });
        w.begin(&schema).unwrap();
        w.row(&[Value::I32(1)]).unwrap();
        w.finish().unwrap();
        assert_eq!(as_str(&buf), "1\n");
    }

    #[test]
    fn csv_quotes_special_cells() {
        let schema = [field("a", Ty::Varchar), field("b", Ty::Varchar)];
        let mut buf = Vec::new();
        let mut w = writer(
            &mut buf,
            Settings { mode: Mode::Csv, separator: ",".to_string(), ..Settings::default() },
        );
        w.begin(&schema).unwrap();
        w.row(&[Value::Bytes(b"has,comma".to_vec()), Value::Bytes(b"has\"quote".to_vec())])
            .unwrap();
        w.row(&[Value::Bytes(b"plain".to_vec()), Value::Null]).unwrap();
        w.finish().unwrap();
        assert_eq!(as_str(&buf), "a,b\n\"has,comma\",\"has\"\"quote\"\nplain,NULL\n");
    }

    #[test]
    fn json_escapes_and_types_bare_literals() {
        let schema = [
            field("s", Ty::Varchar),
            field("n", Ty::Int),
            field("b", Ty::Boolean),
            field("z", Ty::Varchar),
        ];
        let mut buf = Vec::new();
        let mut w = writer(&mut buf, Settings { mode: Mode::Json, ..Settings::default() });
        w.begin(&schema).unwrap();
        w.row(&[
            Value::Bytes(b"line\nbreak".to_vec()),
            Value::I32(42),
            Value::Bool(true),
            Value::Null,
        ])
        .unwrap();
        w.finish().unwrap();
        let out = as_str(&buf);
        assert_eq!(out, "[\n\t{\"s\":\"line\\nbreak\",\"n\":42,\"b\":true,\"z\":null}\n]\n");
    }

    #[test]
    fn json_replaces_non_finite_floats_with_null() {
        let schema = [
            field("nan", Ty::Double),
            field("pos_inf", Ty::Double),
            field("neg_inf", Ty::Double),
            field("finite", Ty::Double),
        ];
        let mut buf = Vec::new();
        let mut w = writer(&mut buf, Settings { mode: Mode::Json, ..Settings::default() });
        w.begin(&schema).unwrap();
        w.row(&[
            Value::F64(f64::NAN),
            Value::F64(f64::INFINITY),
            Value::F64(f64::NEG_INFINITY),
            Value::F64(1.5),
        ])
        .unwrap();
        w.finish().unwrap();

        assert_eq!(
            as_str(&buf),
            "[\n\t{\"nan\":null,\"pos_inf\":null,\"neg_inf\":null,\"finite\":1.5}\n]\n"
        );
    }

    #[test]
    fn json_empty_result_set() {
        let schema = [field("n", Ty::Int)];
        let mut buf = Vec::new();
        let mut w = writer(&mut buf, Settings { mode: Mode::Json, ..Settings::default() });
        w.begin(&schema).unwrap();
        w.finish().unwrap();
        assert_eq!(as_str(&buf), "[]\n");
    }

    #[test]
    fn json_streams_rows_before_finish() {
        let schema = [field("n", Ty::Int)];
        let observed = Rc::new(RefCell::new(Vec::new()));
        let mut sink = SharedBuffer(Rc::clone(&observed));
        let mut w = Writer::new(&mut sink, Settings { mode: Mode::Json, ..Settings::default() });

        w.begin(&schema).unwrap();
        w.row(&[Value::I32(1)]).unwrap();
        assert_eq!(as_str(&observed.borrow()), "[\n\t{\"n\":1}");

        w.row(&[Value::I32(2)]).unwrap();
        assert_eq!(as_str(&observed.borrow()), "[\n\t{\"n\":1},\n\t{\"n\":2}");

        w.finish().unwrap();
        assert_eq!(as_str(&observed.borrow()), "[\n\t{\"n\":1},\n\t{\"n\":2}\n]\n");
    }

    #[test]
    fn json_lines_object_per_line() {
        let schema = [field("n", Ty::Int)];
        let mut buf = Vec::new();
        let mut w = writer(&mut buf, Settings { mode: Mode::JsonLines, ..Settings::default() });
        w.begin(&schema).unwrap();
        w.row(&[Value::I32(1)]).unwrap();
        w.row(&[Value::I32(2)]).unwrap();
        w.finish().unwrap();
        assert_eq!(as_str(&buf), "{\"n\":1}\n{\"n\":2}\n");
    }

    #[test]
    fn markdown_escapes_pipes() {
        let schema = [field("a", Ty::Varchar)];
        let mut buf = Vec::new();
        let mut w = writer(&mut buf, Settings { mode: Mode::Markdown, ..Settings::default() });
        w.begin(&schema).unwrap();
        w.row(&[Value::Bytes(b"a|b".to_vec())]).unwrap();
        w.finish().unwrap();
        assert_eq!(as_str(&buf), "| a |\n|---|\n| a\\|b |\n");
    }

    #[test]
    fn insert_quotes_strings_and_bares_numbers() {
        let schema = [field("s", Ty::Varchar), field("n", Ty::Int)];
        let mut buf = Vec::new();
        let mut w = writer(
            &mut buf,
            Settings { mode: Mode::Insert, insert_table: "t".to_string(), ..Settings::default() },
        );
        w.begin(&schema).unwrap();
        w.row(&[Value::Bytes(b"it's".to_vec()), Value::I32(5)]).unwrap();
        w.row(&[Value::Null, Value::I32(6)]).unwrap();
        w.finish().unwrap();
        assert_eq!(
            as_str(&buf),
            "INSERT INTO t VALUES ('it''s', 5);\nINSERT INTO t VALUES (NULL, 6);\n"
        );
    }

    #[test]
    fn insert_quotes_non_finite_floats_for_sql_replay() {
        let schema = [
            field("nan", Ty::Double),
            field("pos_inf", Ty::Double),
            field("neg_inf", Ty::Float),
            field("finite", Ty::Double),
        ];
        let mut buf = Vec::new();
        let mut w = writer(
            &mut buf,
            Settings { mode: Mode::Insert, insert_table: "t".to_string(), ..Settings::default() },
        );
        w.begin(&schema).unwrap();
        w.row(&[
            Value::F64(f64::NAN),
            Value::F64(f64::INFINITY),
            Value::F64(f64::NEG_INFINITY),
            Value::F64(1.5),
        ])
        .unwrap();
        w.finish().unwrap();
        assert_eq!(as_str(&buf), "INSERT INTO t VALUES ('NaN', 'inf', '-inf', 1.5);\n");
    }

    #[test]
    fn insert_quotes_unsafe_table_names() {
        let schema = [field("n", Ty::Int)];
        let mut buf = Vec::new();
        let mut w = writer(
            &mut buf,
            Settings {
                mode: Mode::Insert,
                insert_table: "x; DROP TABLE y; --".to_string(),
                ..Settings::default()
            },
        );
        w.begin(&schema).unwrap();
        w.row(&[Value::I32(1)]).unwrap();
        w.finish().unwrap();
        assert_eq!(as_str(&buf), "INSERT INTO \"x; DROP TABLE y; --\" VALUES (1);\n");
        assert_eq!(insert_target("else"), "\"else\"");
        assert_eq!(insert_target("main.safe_name"), "main.safe_name");
        assert_eq!(insert_target("main.has\"quote"), "main.\"has\"\"quote\"");
    }

    #[test]
    fn box_layout_2x2() {
        let schema = [field("a", Ty::Varchar), field("b", Ty::Int)];
        let mut buf = Vec::new();
        let mut w = writer(&mut buf, Settings { mode: Mode::Box, ..Settings::default() });
        w.begin(&schema).unwrap();
        w.row(&[Value::Bytes(b"x".to_vec()), Value::I32(1)]).unwrap();
        w.row(&[Value::Bytes(b"yy".to_vec()), Value::I32(22)]).unwrap();
        w.finish().unwrap();
        let expected = "\
┌────┬────┐
│ a  │  b │
├────┼────┤
│ x  │  1 │
│ yy │ 22 │
└────┴────┘
";
        assert_eq!(as_str(&buf), expected);
    }

    #[test]
    fn duckbox_row_truncation_footer() {
        let schema = [field("n", Ty::Int)];
        let mut buf = Vec::new();
        let mut w =
            writer(&mut buf, Settings { mode: Mode::Duckbox, maxrows: 4, ..Settings::default() });
        w.begin(&schema).unwrap();
        for i in 1..=6 {
            w.row(&[Value::I32(i)]).unwrap();
        }
        w.finish().unwrap();
        let out = as_str(&buf);
        assert!(out.lines().any(|l| l == "6 rows (4 shown)"), "footer missing in:\n{out}");
        // First two and last two rows are shown, with a `·` marker between.
        assert!(out.contains("1"));
        assert!(out.contains("6"));
        assert!(out.contains('·'));
    }

    #[test]
    fn duckbox_odd_row_limit_shows_exact_requested_count() {
        let schema = [field("n", Ty::Int)];
        let mut buf = Vec::new();
        let mut w =
            writer(&mut buf, Settings { mode: Mode::Duckbox, maxrows: 3, ..Settings::default() });
        w.begin(&schema).unwrap();
        for i in 1..=6 {
            w.row(&[Value::I32(i)]).unwrap();
        }
        w.finish().unwrap();

        let out = as_str(&buf);
        assert!(out.lines().any(|line| line == "6 rows (3 shown)"), "footer missing in:\n{out}");
        let cells: Vec<&str> =
            out.lines().filter_map(|line| line.split('│').nth(1).map(str::trim)).collect();
        assert!(cells.contains(&"1"));
        assert!(cells.contains(&"2"));
        assert!(cells.contains(&"6"));
    }

    #[test]
    fn duckbox_one_row_limit_still_shows_data() {
        let schema = [field("n", Ty::Int)];
        let mut buf = Vec::new();
        let mut w =
            writer(&mut buf, Settings { mode: Mode::Duckbox, maxrows: 1, ..Settings::default() });
        w.begin(&schema).unwrap();
        w.row(&[Value::I32(7)]).unwrap();
        w.row(&[Value::I32(8)]).unwrap();
        w.finish().unwrap();

        let out = as_str(&buf);
        assert!(out.lines().any(|line| line == "2 rows (1 shown)"), "footer missing in:\n{out}");
        assert!(out
            .lines()
            .filter_map(|line| line.split('│').nth(1).map(str::trim))
            .any(|cell| cell == "7"));
    }

    #[test]
    fn duckbox_no_truncation_has_plain_footer() {
        let schema = [field("n", Ty::Int)];
        let mut buf = Vec::new();
        let mut w = writer(&mut buf, Settings { mode: Mode::Duckbox, ..Settings::default() });
        w.begin(&schema).unwrap();
        w.row(&[Value::I32(1)]).unwrap();
        w.finish().unwrap();
        let out = as_str(&buf);
        assert!(out.lines().any(|l| l == "1 row"), "footer missing in:\n{out}");
    }

    #[test]
    fn trash_writes_nothing() {
        let schema = [field("n", Ty::Int)];
        let mut buf = Vec::new();
        let mut w = writer(&mut buf, Settings { mode: Mode::Trash, ..Settings::default() });
        w.begin(&schema).unwrap();
        w.row(&[Value::I32(1)]).unwrap();
        w.finish().unwrap();
        assert!(buf.is_empty());
    }
}
