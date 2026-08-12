//! Result rendering: output modes, `.headers`, `.nullvalue`, `.separator`,
//! `.maxrows`, `.maxwidth`.
//!
//! Rows arrive one at a time and most modes stream them straight out; only the
//! boxed modes (`Box`/`Duckbox`) and `Json` buffer, since they cannot know the
//! column widths (boxed modes) or whether they are writing the first/last row
//! (`Json`, which needs commas between array elements and `[]` for an empty
//! result) until every row has been seen.

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
    /// Max rows shown in the boxed modes before the middle is elided.
    /// 0 disables truncation.
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
    /// Pre-rendered `{"col":value,...}` fragments, buffered for `Json` (needs
    /// to know the last row to place commas, and whether there were any rows
    /// at all to choose between `[]` and a multi-line array).
    json_rows: Vec<String>,
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
            json_rows: Vec::new(),
        }
    }

    /// Starts a result set. Must be called before `row`.
    pub fn begin(&mut self, schema: &[Field]) -> io::Result<()> {
        self.names = schema.iter().map(|f| f.name.clone()).collect();
        self.types = schema.iter().map(|f| f.ty).collect();
        self.rows = 0;
        self.box_rows.clear();
        self.json_rows.clear();
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
            Mode::Json => {
                let frag = self.json_row_fragment(values);
                self.json_rows.push(frag);
                Ok(())
            }
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
            self.settings.insert_table,
            parts.join(", ")
        )
    }

    // ---- Json / JsonLines ----

    fn write_row_jsonlines(&mut self, values: &[Value]) -> io::Result<()> {
        let frag = self.json_row_fragment(values);
        writeln!(self.out, "{frag}")
    }

    /// Renders one row as a compact `{"col":value,...}` fragment. Shared by
    /// `Json` (buffered, assembled into an array in `finish_json`) and
    /// `JsonLines` (written straight out, one per line).
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
        if self.json_rows.is_empty() {
            return writeln!(self.out, "[]");
        }
        writeln!(self.out, "[")?;
        let last = self.json_rows.len() - 1;
        for (i, r) in self.json_rows.iter().enumerate() {
            if i == last {
                writeln!(self.out, "\t{r}")?;
            } else {
                writeln!(self.out, "\t{r},")?;
            }
        }
        writeln!(self.out, "]")
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

        // --- Row truncation: keep the first and last `maxrows/2` rows, with
        // a single elision marker row spliced into the middle. ---
        let row_trunc = duck && self.settings.maxrows > 0 && total_rows > self.settings.maxrows;
        let half = self.settings.maxrows / 2;
        let display_rows: Vec<DisplayRow> = if row_trunc {
            let mut v = Vec::with_capacity(half * 2 + 1);
            v.extend(self.box_rows[..half].iter().map(|r| DisplayRow::Data(r.as_slice())));
            v.push(DisplayRow::Elision);
            v.extend(
                self.box_rows[total_rows - half..].iter().map(|r| DisplayRow::Data(r.as_slice())),
            );
            v
        } else {
            self.box_rows.iter().map(|r| DisplayRow::Data(r.as_slice())).collect()
        };
        let shown_rows = if row_trunc { half * 2 } else { total_rows };

        // --- Column content widths, across the header, the type line (duck
        // only) and every displayed data row. ---
        let mut col_width: Vec<usize> = self.names.iter().map(|n| n.chars().count()).collect();
        if duck {
            for (w, ty) in col_width.iter_mut().zip(&self.types) {
                *w = (*w).max(ty.name().to_ascii_lowercase().chars().count());
            }
        }
        for row in &display_rows {
            if let DisplayRow::Data(cells) = row {
                for (w, cell) in col_width.iter_mut().zip(cells.iter()) {
                    *w = (*w).max(cell.chars().count());
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

/// Pads `text` to `width` with one space of padding on each side, right- or
/// left-aligned within that width.
fn pad_cell(text: &str, width: usize, right_align: bool) -> String {
    let len = text.chars().count();
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
    let len = marker.chars().count();
    let fill = width.saturating_sub(len);
    let left = fill / 2;
    let right = fill - left;
    format!(" {}{marker}{} ", " ".repeat(left), " ".repeat(right))
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn json_empty_result_set() {
        let schema = [field("n", Ty::Int)];
        let mut buf = Vec::new();
        let mut w = writer(&mut buf, Settings { mode: Mode::Json, ..Settings::default() });
        w.begin(&schema).unwrap();
        w.finish().unwrap();
        assert_eq!(as_str(&buf), "[]\n");
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
