//! Document parsers — extract text and media from various file formats.

pub mod pdf;
pub mod docx;
pub mod pptx;
pub mod text;
pub mod audio;
pub mod tabular;

use std::path::{Path, PathBuf};

// ── Element-level types (multimodal) ──

/// The kind of content extracted from a document.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum ElementKind {
    /// Plain text (paragraph, heading, list, etc.)
    Text,
    /// An embedded image extracted from the document.
    Image,
    /// A table rendered as a PNG image.
    Table,
}

/// A single element (text block or media asset) extracted from a document.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ParsedElement {
    /// What kind of content this element represents.
    pub kind: ElementKind,
    /// Text content: the paragraph text for `Text`, or the surrounding context
    /// caption for `Image`/`Table` elements.
    pub text: String,
    /// For `Image`/`Table`: path to the extracted PNG file on disk.
    /// `None` for `Text` elements.
    pub media_path: Option<PathBuf>,
    /// Optional metadata (page number, slide number, etc.)
    pub metadata: String,
}

/// Parsed output from any document parser (multimodal).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ParsedDocument {
    /// Human-readable title (filename or extracted title).
    pub title: String,
    /// All extracted elements: text blocks, images, and tables.
    pub elements: Vec<ParsedElement>,
    /// Legacy text-only sections (for backward compatibility with chunker).
    pub sections: Vec<TextBlock>,
}

/// A single block of text from a document (legacy, used by chunker).
#[derive(Debug, Clone, serde::Serialize)]
pub struct TextBlock {
    /// The text content.
    pub text: String,
    /// Optional metadata (page number, slide number, timestamp, etc.)
    pub metadata: String,
}

// ── Helper constructors ──

impl ParsedElement {
    /// Create a text element.
    pub fn text(text: impl Into<String>, metadata: impl Into<String>) -> Self {
        Self {
            kind: ElementKind::Text,
            text: text.into(),
            media_path: None,
            metadata: metadata.into(),
        }
    }

    /// Create an image element with surrounding context as caption.
    pub fn image(
        context_caption: impl Into<String>,
        media_path: PathBuf,
        metadata: impl Into<String>,
    ) -> Self {
        Self {
            kind: ElementKind::Image,
            text: context_caption.into(),
            media_path: Some(media_path),
            metadata: metadata.into(),
        }
    }

    /// Create a table element (rendered as PNG) with surrounding context as caption.
    pub fn table(
        context_caption: impl Into<String>,
        media_path: PathBuf,
        metadata: impl Into<String>,
    ) -> Self {
        Self {
            kind: ElementKind::Table,
            text: context_caption.into(),
            media_path: Some(media_path),
            metadata: metadata.into(),
        }
    }
}

impl ParsedDocument {
    /// Build a text-only ParsedDocument (convenience for parsers that don't
    /// extract media yet, like text.rs and audio.rs).
    pub fn text_only(title: String, sections: Vec<TextBlock>) -> Self {
        let elements = sections
            .iter()
            .map(|s| ParsedElement::text(&s.text, &s.metadata))
            .collect();
        Self {
            title,
            elements,
            sections,
        }
    }

    /// Collect all text-bearing elements into legacy TextBlock sections.
    /// Used by the pipeline to feed into the chunker.
    pub fn text_sections(&self) -> Vec<&TextBlock> {
        self.sections.iter().collect()
    }

    /// Return only media elements (images and tables).
    pub fn media_elements(&self) -> Vec<&ParsedElement> {
        self.elements
            .iter()
            .filter(|e| e.kind == ElementKind::Image || e.kind == ElementKind::Table)
            .collect()
    }
}

/// Size heuristic filter constants for media extraction.
pub const MIN_IMAGE_WIDTH: u32 = 50;
pub const MIN_IMAGE_HEIGHT: u32 = 50;
pub const MIN_IMAGE_BYTES: usize = 5_000; // 5 KB

/// Parse a document at the given path, dispatching to the correct parser
/// based on file extension. The `media_dir` is where extracted images/tables
/// are saved as PNG files.
pub fn parse_document(path: &Path) -> Result<ParsedDocument, String> {
    parse_document_with_media(path, None)
}

/// Parse a document, optionally extracting media to the given directory.
pub fn parse_document_with_media(
    path: &Path,
    media_dir: Option<&Path>,
) -> Result<ParsedDocument, String> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "pdf" => pdf::parse(path, media_dir),
        "docx" => docx::parse(path, media_dir),
        "pptx" => pptx::parse(path, media_dir),
        "csv" => tabular::parse_csv(path),
        "xlsx" | "xls" | "ods" => tabular::parse_xlsx(path),
        "txt" | "md" | "json" | "toml" | "yaml" | "yml" | "rs" | "py" | "js" | "ts"
        | "c" | "cpp" | "h" | "java" | "go" | "rb" | "sh" | "bat" | "html" | "css" | "xml"
        | "log" => text::parse(path),
        "mp3" | "wav" | "m4a" | "ogg" | "flac" | "aac" | "wma" | "webm" => {
            // Audio files need async transcription — return a placeholder.
            Ok(ParsedDocument::text_only(
                path.file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("audio")
                    .to_string(),
                vec![TextBlock {
                    text: format!("[Audio file pending transcription: {}]", path.display()),
                    metadata: "audio:pending".into(),
                }],
            ))
        }
        _ => Err(format!("Unsupported file type: .{ext}")),
    }
}
