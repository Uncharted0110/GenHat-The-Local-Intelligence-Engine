//! Tabular data parsers — CSV and XLSX/XLS/ODS.
//!
//! Strategy:
//!   - Header row is read once and prepended to every section so the LLM
//!     always sees column names alongside row data.
//!   - Rows are batched (ROWS_PER_SECTION) into text sections; the existing
//!     RecursiveCharacterChunker then handles final chunk sizing.
//!   - Each row is rendered as "Row N: col1=val1, col2=val2, ..."
//!   - Multiple XLSX sheets produce independent section groups.

use std::path::Path;
use super::{ParsedDocument, TextBlock};

/// Number of data rows to include in a single text section.
/// Keeps sections comfortably below the 1536-char chunk ceiling.
const ROWS_PER_SECTION: usize = 50;

// ── CSV ──────────────────────────────────────────────────────────────────────

pub fn parse_csv(path: &Path) -> Result<ParsedDocument, String> {
    let title = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("table.csv")
        .to_string();

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)
        .map_err(|e| format!("Failed to open CSV: {e}"))?;

    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| format!("CSV header error: {e}"))?
        .iter()
        .map(|s| s.to_owned())
        .collect();

    let header_line = if headers.is_empty() {
        String::new()
    } else {
        format!("Columns: {}\n", headers.join(", "))
    };

    let mut all_rows: Vec<Vec<String>> = Vec::new();
    for result in rdr.records() {
        let record = result.map_err(|e| format!("CSV read error: {e}"))?;
        all_rows.push(record.iter().map(|s| s.to_owned()).collect());
    }

    let sections = rows_to_sections(&headers, &header_line, &all_rows, None);
    Ok(ParsedDocument::text_only(title, sections))
}

// ── XLSX / XLS / ODS ─────────────────────────────────────────────────────────

pub fn parse_xlsx(path: &Path) -> Result<ParsedDocument, String> {
    use calamine::{open_workbook_auto, Reader};

    let title = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("spreadsheet.xlsx")
        .to_string();

    let mut workbook = open_workbook_auto(path)
        .map_err(|e| format!("Failed to open spreadsheet: {e}"))?;

    let sheet_names: Vec<String> = workbook.sheet_names().to_vec();
    if sheet_names.is_empty() {
        return Err("Spreadsheet has no sheets".into());
    }

    let mut all_sections: Vec<TextBlock> = Vec::new();

    for sheet_name in &sheet_names {
        let range = match workbook.worksheet_range(sheet_name) {
            Ok(r) => r,
            Err(e) => {
                log::warn!("Skipping sheet '{}': {}", sheet_name, e);
                continue;
            }
        };

        if range.is_empty() {
            continue;
        }

        let mut rows_iter = range.rows();

        // First row = headers
        let header_row = match rows_iter.next() {
            Some(r) => r,
            None => continue,
        };

        let headers: Vec<String> = header_row
            .iter()
            .map(|cell| cell_to_string(cell))
            .collect();

        let header_line = if headers.is_empty() {
            format!("Sheet: {sheet_name}\n")
        } else {
            format!("Sheet: {sheet_name}\nColumns: {}\n", headers.join(", "))
        };

        let all_rows: Vec<Vec<String>> = rows_iter
            .map(|row| row.iter().map(|cell| cell_to_string(cell)).collect())
            .collect();

        let sections = rows_to_sections(
            &headers,
            &header_line,
            &all_rows,
            Some(sheet_name.as_str()),
        );
        all_sections.extend(sections);
    }

    if all_sections.is_empty() {
        return Err("Spreadsheet produced no parseable content".into());
    }

    Ok(ParsedDocument::text_only(title, all_sections))
}

// ── Shared helpers ────────────────────────────────────────────────────────────

/// Convert rows into text sections with ROWS_PER_SECTION rows each.
/// `header_line` is prepended to every section so each chunk is self-contained.
fn rows_to_sections(
    headers: &[String],
    header_line: &str,
    rows: &[Vec<String>],
    sheet: Option<&str>,
) -> Vec<TextBlock> {
    if rows.is_empty() {
        return Vec::new();
    }

    let total = rows.len();
    let mut sections = Vec::new();
    let mut chunk_start = 0usize;

    while chunk_start < total {
        let chunk_end = (chunk_start + ROWS_PER_SECTION).min(total);
        let mut buf = header_line.to_owned();

        for (i, row) in rows[chunk_start..chunk_end].iter().enumerate() {
            let row_num = chunk_start + i + 1; // 1-based
            let mut pairs = String::new();
            for (col_idx, val) in row.iter().enumerate() {
                if val.is_empty() {
                    continue; // skip empty cells
                }
                if !pairs.is_empty() {
                    pairs.push_str(", ");
                }
                let col_name = headers.get(col_idx).map(|s| s.as_str()).unwrap_or("?");
                pairs.push_str(&format!("{col_name}={val}"));
            }
            if !pairs.is_empty() {
                buf.push_str(&format!("Row {row_num}: {pairs}\n"));
            }
        }

        let trimmed = buf.trim().to_string();
        if !trimmed.is_empty() {
            let metadata = match sheet {
                Some(s) => format!("sheet:{s}:rows:{}-{}", chunk_start + 1, chunk_end),
                None => format!("rows:{}-{}", chunk_start + 1, chunk_end),
            };
            sections.push(TextBlock {
                text: trimmed,
                metadata,
            });
        }

        chunk_start = chunk_end;
    }

    sections
}

/// Convert a calamine cell value to a plain string.
fn cell_to_string(cell: &calamine::Data) -> String {
    use calamine::Data;
    match cell {
        Data::Int(n) => n.to_string(),
        Data::Float(f) => {
            // Avoid scientific notation for common spreadsheet values
            if f.abs() < 1e15 && f.fract() == 0.0 {
                format!("{}", *f as i64)
            } else {
                format!("{f}")
            }
        }
        Data::String(s) => s.clone(),
        Data::Bool(b) => b.to_string(),
        Data::DateTime(dt) => {
            // Excel serial date stored as f64; format as-is for now
            format!("{dt}")
        }
        Data::DateTimeIso(s) => s.clone(),
        Data::DurationIso(s) => s.clone(),
        Data::Error(e) => format!("[Error: {e:?}]"),
        Data::Empty => String::new(),
    }
}
