//! Watched-paths manager — incremental file-system scanner for auto-discovery.
//!
//! A user can register any number of directories as "watched". On each app
//! startup (and on an explicit user-triggered scan) the scanner walks those
//! directories recursively, compares file modification times + SHA-256 hashes
//! against what is recorded in the RAG DB, and re-ingests only the files that
//! are new or changed.
//!
//! The scan runs in a background task and emits Tauri events so the frontend
//! can show a progress indicator:
//!   `rag:scan_progress`  — `ScanProgress` struct (JSON)

use std::path::PathBuf;
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use sha2::{Digest, Sha256};
use tauri::Emitter;
use walkdir::WalkDir;

use crate::rag::db::RagDb;
use crate::rag::pipeline::RagPipeline;

// ── Supported file extensions (must stay in sync with parsers/mod.rs) ─────────

/// File extensions that the RAG pipeline can parse.
const SUPPORTED_EXTENSIONS: &[&str] = &[
    // Documents
    "pdf", "docx", "pptx",
    // Tabular
    "csv", "xlsx", "xls", "ods",
    // Text / source
    "txt", "md", "json", "toml", "yaml", "yml",
    "rs", "py", "js", "ts", "c", "cpp", "h", "java", "go", "rb", "sh",
    "bat", "html", "css", "xml", "log",
    // Audio
    "mp3", "wav", "m4a", "ogg", "flac", "aac", "wma", "webm",
];

fn is_supported(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| SUPPORTED_EXTENSIONS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

// ── Progress event payload ────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct ScanProgress {
    /// Human-readable status message.
    pub status: String,
    /// Number of files found so far.
    pub found: usize,
    /// Number of new/changed files ingested so far.
    pub ingested: usize,
    /// Number of files skipped (unchanged).
    pub skipped: usize,
    /// Number of files that failed ingestion.
    pub errors: usize,
    /// Whether the scan has finished.
    pub done: bool,
}

// ── Scan result summary ───────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct ScanResult {
    pub ingested: usize,
    pub skipped: usize,
    pub errors: usize,
    pub total_files: usize,
}

// ── Hash helper ──────────────────────────────────────────────────────────────

/// Compute SHA-256 of a file and return as lowercase hex string.
fn sha256_of_file(path: &std::path::Path) -> Option<String> {
    let data = std::fs::read(path).ok()?;
    let digest = Sha256::digest(&data);
    Some(hex::encode(digest))
}

/// Get file mtime as seconds since Unix epoch. Returns 0 on failure.
fn mtime_secs(path: &std::path::Path) -> i64 {
    std::fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ── WatchedPathsManager ───────────────────────────────────────────────────────

pub struct WatchedPathsManager {
    pub db: Arc<RagDb>,
}

impl WatchedPathsManager {
    pub fn new(db: Arc<RagDb>) -> Self {
        Self { db }
    }

    pub fn add_path(&self, workspace_id: &str, path: &str) -> Result<(), String> {
        // Validate the path is an existing directory before registering.
        let p = PathBuf::from(path);
        if !p.exists() {
            return Err(format!("Path does not exist: {path}"));
        }
        if !p.is_dir() {
            return Err(format!("Path is not a directory: {path}"));
        }
        self.db.insert_watched_path(workspace_id, path)
    }

    pub fn remove_path(&self, workspace_id: &str, path: &str) -> Result<(), String> {
        self.db.delete_watched_path(workspace_id, path)
    }

    pub fn list_paths(
        &self,
        workspace_id: &str,
    ) -> Result<Vec<crate::rag::db::WatchedPathRecord>, String> {
        self.db.list_watched_paths(workspace_id)
    }

    /// Walk all watched dirs for `workspace_id`, compare hashes, and re-ingest
    /// any new or changed files.  Emits `rag:scan_progress` Tauri events.
    pub async fn scan_diff(
        &self,
        pipeline: Arc<RagPipeline>,
        workspace_id: &str,
        app_handle: tauri::AppHandle,
    ) -> ScanResult {
        let watched = match self.db.list_watched_paths(workspace_id) {
            Ok(p) => p,
            Err(e) => {
                log::error!("scan_diff: failed to list watched paths: {e}");
                return ScanResult {
                    ingested: 0,
                    skipped: 0,
                    errors: 0,
                    total_files: 0,
                };
            }
        };

        if watched.is_empty() {
            emit_progress(
                &app_handle,
                ScanProgress {
                    status: "No watched paths configured.".into(),
                    found: 0,
                    ingested: 0,
                    skipped: 0,
                    errors: 0,
                    done: true,
                },
            );
            return ScanResult {
                ingested: 0,
                skipped: 0,
                errors: 0,
                total_files: 0,
            };
        }

        emit_progress(
            &app_handle,
            ScanProgress {
                status: format!("Scanning {} watched director{}…", watched.len(), if watched.len() == 1 { "y" } else { "ies" }),
                found: 0,
                ingested: 0,
                skipped: 0,
                errors: 0,
                done: false,
            },
        );

        // Collect all candidate files (blocking walk on spawn_blocking)
        let mut candidate_paths: Vec<PathBuf> = Vec::new();
        for record in &watched {
            let dir = PathBuf::from(&record.path);
            if !dir.is_dir() {
                log::warn!("Watched path is no longer a directory: {}", dir.display());
                continue;
            }
            for entry in WalkDir::new(&dir)
                .follow_links(false)
                .into_iter()
                .flatten()
            {
                let path = entry.path().to_path_buf();
                if path.is_file() && is_supported(&path) {
                    candidate_paths.push(path);
                }
            }
        }

        let total = candidate_paths.len();
        let mut ingested = 0usize;
        let mut skipped = 0usize;
        let mut errors = 0usize;

        emit_progress(
            &app_handle,
            ScanProgress {
                status: format!("Found {total} supported files — checking for changes…"),
                found: total,
                ingested: 0,
                skipped: 0,
                errors: 0,
                done: false,
            },
        );

        for path in &candidate_paths {
            let path_str = path.to_string_lossy().to_string();

            // Check mtime first (cheap)
            let current_mtime = mtime_secs(path);

            match self.db.get_document_hash_by_path(&path_str) {
                Ok(Some((doc_id, stored_hash, stored_mtime))) => {
                    if stored_mtime != 0 && stored_mtime == current_mtime {
                        // mtime unchanged → skip without hashing
                        skipped += 1;
                        continue;
                    }

                    // mtime changed → compute hash to confirm actual content change
                    let current_hash = match sha256_of_file(path) {
                        Some(h) => h,
                        None => {
                            log::warn!("Could not hash file: {path_str}");
                            errors += 1;
                            continue;
                        }
                    };

                    if current_hash == stored_hash && !stored_hash.is_empty() {
                        // Content unchanged (hash matches) — update mtime in DB
                        let _ = self.db.update_document_hash(doc_id, &current_hash, current_mtime);
                        skipped += 1;
                        continue;
                    }

                    // Content changed → delete old record and re-ingest
                    log::info!("File changed, re-ingesting: {path_str}");
                    if let Err(e) = pipeline.delete_document(doc_id).await {
                        log::warn!("Could not delete stale doc {doc_id}: {e}");
                    }

                    match pipeline.ingest_document(path).await {
                        Ok(_status) => {
                            // Record hash + mtime for the newly ingested doc
                            let _ = self
                                .db
                                .set_new_document_hash(&path_str, &current_hash, current_mtime);
                            ingested += 1;
                        }
                        Err(e) => {
                            log::warn!("Re-ingestion failed for {path_str}: {e}");
                            errors += 1;
                        }
                    }
                }
                Ok(None) => {
                    // New file — ingest it
                    log::info!("New file discovered, ingesting: {path_str}");

                    let current_hash = sha256_of_file(path).unwrap_or_default();

                    match pipeline.ingest_document(path).await {
                        Ok(_status) => {
                            let _ = self
                                .db
                                .set_new_document_hash(&path_str, &current_hash, current_mtime);
                            ingested += 1;
                        }
                        Err(e) => {
                            log::warn!("Ingestion failed for {path_str}: {e}");
                            errors += 1;
                        }
                    }
                }
                Err(e) => {
                    log::warn!("DB error checking {path_str}: {e}");
                    errors += 1;
                }
            }

            // Emit progress every 10 files
            let processed = ingested + skipped + errors;
            if processed % 10 == 0 {
                emit_progress(
                    &app_handle,
                    ScanProgress {
                        status: format!("Processing… ({processed}/{total})"),
                        found: total,
                        ingested,
                        skipped,
                        errors,
                        done: false,
                    },
                );
            }
        }

        let result = ScanResult {
            ingested,
            skipped,
            errors,
            total_files: total,
        };

        emit_progress(
            &app_handle,
            ScanProgress {
                status: format!(
                    "Scan complete — {ingested} ingested, {skipped} unchanged, {errors} errors"
                ),
                found: total,
                ingested,
                skipped,
                errors,
                done: true,
            },
        );

        result
    }
}

fn emit_progress(app: &tauri::AppHandle, progress: ScanProgress) {
    if let Err(e) = app.emit("rag:scan_progress", &progress) {
        log::debug!("Failed to emit rag:scan_progress: {e}");
    }
}
