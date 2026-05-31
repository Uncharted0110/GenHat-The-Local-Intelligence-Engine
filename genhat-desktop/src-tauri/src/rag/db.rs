//! SQLite vector store for RAG with r2d2 connection pooling.
//!
//! Schema:
//!   documents  — one row per ingested file
//!   chunks     — one row per text chunk, with embeddings stored as BLOB
//!
//! Uses an r2d2 connection pool (8 connections) for parallel read access
//! under WAL mode. Vector search is delegated to the in-memory VectorIndex
//! (see vecindex.rs) for sub-linear ANN search; the brute-force fallback
//! method is retained for testing.

use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rayon::prelude::*;
use rusqlite::params;
use std::path::{Path, PathBuf};

// ── Types ──

/// A stored document record.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DocumentRecord {
    pub id: i64,
    pub path: String,
    pub title: String,
    pub doc_type: String,
    pub chunk_count: i64,
    pub enriched_count: i64,
    pub created_at: String,
}

/// A stored chunk record.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChunkRecord {
    pub id: i64,
    pub doc_id: i64,
    pub chunk_index: i32,
    pub text: String,
    pub enriched_text: Option<String>,
    pub metadata: String,
    pub confidence: Option<f64>,
}

/// A watched directory path registered for auto-scanning.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WatchedPathRecord {
    pub id: i64,
    pub workspace_id: String,
    pub path: String,
    pub added_at: String,
}

/// A stored media asset (image or table extracted from a document).
#[derive(Debug, Clone, serde::Serialize)]
pub struct MediaAssetRecord {
    pub id: i64,
    pub doc_id: i64,
    /// "image" or "table"
    pub asset_type: String,
    /// Path to the extracted PNG file on disk.
    pub file_path: String,
    /// Context-aware caption (surrounding text from document).
    pub caption: String,
    /// Source metadata (e.g. "page:3:image:2").
    pub metadata: String,
    /// SHA-256 hash of the caption embedding (for dedup).
    pub caption_hash: Option<String>,
}

/// Handle to the RAG database backed by an r2d2 connection pool.
#[derive(Clone)]
pub struct RagDb {
    pool: Pool<SqliteConnectionManager>,
    pub path: PathBuf,
}

// ── Helpers ──

/// Serialize f32 embedding to bytes (little-endian).
pub fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for &v in embedding {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Deserialize bytes back to f32 embedding.
pub fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        dot / denom
    }
}

/// Dot product between two vectors.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
    }
    dot
}

// ── Connection Pool Initializer ──

/// Configures each pooled SQLite connection with WAL mode and synchronous=NORMAL.
#[derive(Debug)]
struct WalModeCustomizer;

impl r2d2::CustomizeConnection<rusqlite::Connection, rusqlite::Error> for WalModeCustomizer {
    fn on_acquire(&self, conn: &mut rusqlite::Connection) -> Result<(), rusqlite::Error> {
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;
        Ok(())
    }
}

// ── Implementation ──

impl RagDb {
    /// Get a connection from the pool.
    pub(crate) fn conn(
        &self,
    ) -> Result<r2d2::PooledConnection<SqliteConnectionManager>, String> {
        self.pool.get().map_err(|e| format!("DB pool error: {e}"))
    }

    /// Open (or create) the RAG database at the given path.
    pub fn open(db_path: &Path) -> Result<Self, String> {
        // Ensure parent directory exists
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create DB directory: {e}"))?;
        }

        // Ensure the SQLite file itself exists before creating the pool.
        // Some sqlite open modes used by pooled connection managers may not
        // create the file automatically on first connection.
        if !db_path.exists() {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(db_path)
                .map_err(|e| {
                    format!(
                        "Failed to create DB file {}: {e}",
                        db_path.display()
                    )
                })?;
        }

        let manager = SqliteConnectionManager::file(db_path);
        let pool = Pool::builder()
            .max_size(8) // 8 connections for parallel read access
            .connection_customizer(Box::new(WalModeCustomizer))
            .build(manager)
            .map_err(|e| format!("Failed to create DB pool: {e}"))?;

        let db = Self {
            pool,
            path: db_path.to_path_buf(),
        };

        db.create_tables()?;
        db.create_raptor_tables()?;
        db.create_media_tables()?;
        db.create_watched_paths_tables()?;
        db.migrate_documents_table()?;
        Ok(db)
    }

    fn create_tables(&self) -> Result<(), String> {
        let conn = self.conn()?;
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS documents (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                path         TEXT NOT NULL UNIQUE,
                title        TEXT NOT NULL,
                doc_type     TEXT NOT NULL,
                chunk_count  INTEGER NOT NULL DEFAULT 0,
                enriched_count INTEGER NOT NULL DEFAULT 0,
                created_at   TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id       INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index  INTEGER NOT NULL,
                text         TEXT NOT NULL,
                enriched_text TEXT,
                metadata     TEXT NOT NULL DEFAULT '',
                embedding    BLOB,
                enriched_embedding BLOB,
                confidence   REAL,
                UNIQUE(doc_id, chunk_index)
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            ",
        )
        .map_err(|e| format!("Failed to create tables: {e}"))
    }

    fn create_watched_paths_tables(&self) -> Result<(), String> {
        let conn = self.conn()?;
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS watched_paths (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id TEXT NOT NULL,
                path         TEXT NOT NULL,
                added_at     TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(workspace_id, path)
            );
            CREATE INDEX IF NOT EXISTS idx_watched_workspace ON watched_paths(workspace_id);
            ",
        )
        .map_err(|e| format!("Failed to create watched_paths table: {e}"))
    }

    /// Add `file_hash` and `file_mtime` columns to documents if not present.
    /// Safe to run on every startup (ignores already-existing-column errors).
    fn migrate_documents_table(&self) -> Result<(), String> {
        let conn = self.conn()?;

        // Check for file_hash column
        let has_hash: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('documents') WHERE name = 'file_hash'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(0)
            > 0;

        if !has_hash {
            conn.execute_batch(
                "ALTER TABLE documents ADD COLUMN file_hash TEXT DEFAULT NULL;",
            )
            .map_err(|e| format!("Migration (file_hash): {e}"))?;
        }

        // Check for file_mtime column
        let has_mtime: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('documents') WHERE name = 'file_mtime'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(0)
            > 0;

        if !has_mtime {
            conn.execute_batch(
                "ALTER TABLE documents ADD COLUMN file_mtime INTEGER DEFAULT NULL;",
            )
            .map_err(|e| format!("Migration (file_mtime): {e}"))?;
        }

        Ok(())
    }

    fn create_media_tables(&self) -> Result<(), String> {
        let conn = self.conn()?;
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS media_assets (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id       INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                asset_type   TEXT NOT NULL CHECK(asset_type IN ('image', 'table')),
                file_path    TEXT NOT NULL,
                caption      TEXT NOT NULL DEFAULT '',
                metadata     TEXT NOT NULL DEFAULT '',
                caption_hash TEXT,
                embedding    BLOB,
                created_at   TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_media_doc ON media_assets(doc_id);
            CREATE INDEX IF NOT EXISTS idx_media_type ON media_assets(asset_type);
            ",
        )
        .map_err(|e| format!("Failed to create media tables: {e}"))
    }

    // ── Document CRUD ──

    /// Insert a new document. Returns the document ID.
    pub fn insert_document(
        &self,
        path: &str,
        title: &str,
        doc_type: &str,
        chunk_count: i64,
    ) -> Result<i64, String> {
        let conn = self.conn()?;
        conn.execute(
            "INSERT OR REPLACE INTO documents (path, title, doc_type, chunk_count) VALUES (?1, ?2, ?3, ?4)",
            params![path, title, doc_type, chunk_count],
        )
        .map_err(|e| format!("Insert doc error: {e}"))?;
        Ok(conn.last_insert_rowid())
    }

    /// Get all documents.
    pub fn list_documents(&self) -> Result<Vec<DocumentRecord>, String> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT id, path, title, doc_type, chunk_count, enriched_count, created_at
                 FROM documents ORDER BY created_at DESC",
            )
            .map_err(|e| format!("Query error: {e}"))?;

        let docs = stmt
            .query_map([], |row| {
                Ok(DocumentRecord {
                    id: row.get(0)?,
                    path: row.get(1)?,
                    title: row.get(2)?,
                    doc_type: row.get(3)?,
                    chunk_count: row.get(4)?,
                    enriched_count: row.get(5)?,
                    created_at: row.get(6)?,
                })
            })
            .map_err(|e| format!("Query error: {e}"))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(docs)
    }

    /// Delete a document and all its chunks.
    pub fn delete_document(&self, doc_id: i64) -> Result<(), String> {
        let conn = self.conn()?;
        conn.execute("DELETE FROM chunks WHERE doc_id = ?1", params![doc_id])
            .map_err(|e| format!("Delete chunks error: {e}"))?;
        conn.execute("DELETE FROM documents WHERE id = ?1", params![doc_id])
            .map_err(|e| format!("Delete doc error: {e}"))?;
        Ok(())
    }

    /// Get the stored hash and mtime for an already-ingested document path.
    /// Returns None if the document is not in the DB or has no hash recorded.
    pub fn get_document_hash_by_path(
        &self,
        path: &str,
    ) -> Result<Option<(i64, String, i64)>, String> {
        let conn = self.conn()?;
        let result = conn.query_row(
            "SELECT id, COALESCE(file_hash, ''), COALESCE(file_mtime, 0)
             FROM documents WHERE path = ?1",
            params![path],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?, row.get::<_, i64>(2)?)),
        );
        match result {
            Ok((id, hash, mtime)) => Ok(Some((id, hash, mtime))),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(format!("Query error: {e}")),
        }
    }

    /// Update the hash and mtime for a document that is already in the DB.
    pub fn update_document_hash(
        &self,
        doc_id: i64,
        file_hash: &str,
        file_mtime: i64,
    ) -> Result<(), String> {
        let conn = self.conn()?;
        conn.execute(
            "UPDATE documents SET file_hash = ?1, file_mtime = ?2 WHERE id = ?3",
            params![file_hash, file_mtime, doc_id],
        )
        .map_err(|e| format!("Update doc hash error: {e}"))
        .map(|_| ())
    }

    /// Set hash and mtime when inserting a new document (called right after insert_document).
    pub fn set_new_document_hash(
        &self,
        path: &str,
        file_hash: &str,
        file_mtime: i64,
    ) -> Result<(), String> {
        let conn = self.conn()?;
        conn.execute(
            "UPDATE documents SET file_hash = ?1, file_mtime = ?2 WHERE path = ?3",
            params![file_hash, file_mtime, path],
        )
        .map_err(|e| format!("Set doc hash error: {e}"))
        .map(|_| ())
    }

    // ── Watched Paths CRUD ──

    /// Register a path for a workspace. Silently succeeds if already present.
    pub fn insert_watched_path(
        &self,
        workspace_id: &str,
        path: &str,
    ) -> Result<(), String> {
        let conn = self.conn()?;
        conn.execute(
            "INSERT OR IGNORE INTO watched_paths (workspace_id, path) VALUES (?1, ?2)",
            params![workspace_id, path],
        )
        .map_err(|e| format!("Insert watched path error: {e}"))
        .map(|_| ())
    }

    /// Remove a watched path for a workspace.
    pub fn delete_watched_path(
        &self,
        workspace_id: &str,
        path: &str,
    ) -> Result<(), String> {
        let conn = self.conn()?;
        conn.execute(
            "DELETE FROM watched_paths WHERE workspace_id = ?1 AND path = ?2",
            params![workspace_id, path],
        )
        .map_err(|e| format!("Delete watched path error: {e}"))
        .map(|_| ())
    }

    /// List all watched paths for a workspace.
    pub fn list_watched_paths(&self, workspace_id: &str) -> Result<Vec<WatchedPathRecord>, String> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT id, workspace_id, path, added_at
                 FROM watched_paths WHERE workspace_id = ?1 ORDER BY added_at ASC",
            )
            .map_err(|e| format!("Query error: {e}"))?;
        let records = stmt
            .query_map(params![workspace_id], |row| {
                Ok(WatchedPathRecord {
                    id: row.get(0)?,
                    workspace_id: row.get(1)?,
                    path: row.get(2)?,
                    added_at: row.get(3)?,
                })
            })
            .map_err(|e| format!("Query error: {e}"))?;
        Ok(records.filter_map(|r| r.ok()).collect())
    }

    /// Check if a document (by path) is already ingested.
    pub fn document_exists(&self, path: &str) -> Result<bool, String> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare("SELECT COUNT(*) FROM documents WHERE path = ?1")
            .map_err(|e| format!("Query error: {e}"))?;
        let count: i64 = stmt
            .query_row(params![path], |row| row.get(0))
            .map_err(|e| format!("Query error: {e}"))?;
        Ok(count > 0)
    }

    /// Count total ingested documents.
    pub fn document_count(&self) -> Result<i64, String> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare("SELECT COUNT(*) FROM documents")
            .map_err(|e| format!("Query error: {e}"))?;
        let count: i64 = stmt
            .query_row([], |row| row.get(0))
            .map_err(|e| format!("Query error: {e}"))?;
        Ok(count)
    }

    // ── Chunk CRUD ──

    /// Insert a batch of chunks for a document. Returns the chunk IDs.
    pub fn insert_chunks(
        &self,
        doc_id: i64,
        chunks: &[(usize, String, String)], // (chunk_index, text, metadata)
    ) -> Result<Vec<i64>, String> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "INSERT INTO chunks (doc_id, chunk_index, text, metadata) VALUES (?1, ?2, ?3, ?4)",
            )
            .map_err(|e| format!("Prepare error: {e}"))?;

        let mut ids = Vec::with_capacity(chunks.len());
        for (idx, text, meta) in chunks {
            stmt.execute(params![doc_id, *idx as i32, text, meta])
                .map_err(|e| format!("Insert chunk error: {e}"))?;
            ids.push(conn.last_insert_rowid());
        }

        Ok(ids)
    }

    /// Store an embedding for a chunk, with optional confidence score.
    pub fn set_chunk_embedding(
        &self,
        chunk_id: i64,
        embedding: &[f32],
        confidence: Option<f64>,
    ) -> Result<(), String> {
        let conn = self.conn()?;
        let bytes = embedding_to_bytes(embedding);
        conn.execute(
            "UPDATE chunks SET embedding = ?1, confidence = ?2 WHERE id = ?3",
            params![bytes, confidence, chunk_id],
        )
        .map_err(|e| format!("Set embedding error: {e}"))
        .map(|_| ())
    }

    /// Store enriched text and optionally its embedding for a chunk.
    pub fn set_chunk_enrichment(
        &self,
        chunk_id: i64,
        enriched_text: &str,
        enriched_embedding: Option<&Vec<f32>>,
    ) -> Result<(), String> {
        let conn = self.conn()?;
        let bytes = enriched_embedding.map(|e| embedding_to_bytes(e));
        conn.execute(
            "UPDATE chunks SET enriched_text = ?1, enriched_embedding = ?2 WHERE id = ?3",
            params![enriched_text, bytes, chunk_id],
        )
        .map_err(|e| format!("Set enrichment error: {e}"))
        .map(|_| ())?;

        // Increment enriched_count on the parent document
        conn.execute(
            "UPDATE documents SET enriched_count = enriched_count + 1 WHERE id = (SELECT doc_id FROM chunks WHERE id = ?1)",
            params![chunk_id],
        )
        .map_err(|e| format!("Update enriched count error: {e}"))
        .map(|_| ())
    }

    /// Get chunk IDs that haven't been enriched yet (across all documents).
    /// Returns up to `limit` IDs.
    pub fn unenriched_chunk_ids(&self, limit: usize) -> Result<Vec<i64>, String> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT id FROM chunks WHERE enriched_text IS NULL AND embedding IS NOT NULL ORDER BY id LIMIT ?1",
            )
            .map_err(|e| format!("Query error: {e}"))?;
        let ids = stmt
            .query_map(params![limit as i64], |row| row.get::<_, i64>(0))
            .map_err(|e| format!("Query error: {e}"))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(ids)
    }

    /// Get all chunk IDs belonging to a document.
    pub fn get_chunk_ids_for_doc(&self, doc_id: i64) -> Result<Vec<i64>, String> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare("SELECT id FROM chunks WHERE doc_id = ?1 ORDER BY chunk_index")
            .map_err(|e| format!("Query error: {e}"))?;
        let ids = stmt
            .query_map(params![doc_id], |row| row.get::<_, i64>(0))
            .map_err(|e| format!("Query error: {e}"))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(ids)
    }

    /// Get a chunk by ID.
    pub fn get_chunk(&self, chunk_id: i64) -> Result<ChunkRecord, String> {
        let conn = self.conn()?;
        conn.query_row(
            "SELECT id, doc_id, chunk_index, text, enriched_text, metadata, confidence
             FROM chunks WHERE id = ?1",
            params![chunk_id],
            |row| {
                Ok(ChunkRecord {
                    id: row.get(0)?,
                    doc_id: row.get(1)?,
                    chunk_index: row.get(2)?,
                    text: row.get(3)?,
                    enriched_text: row.get(4)?,
                    metadata: row.get(5)?,
                    confidence: row.get(6)?,
                })
            },
        )
        .map_err(|e| format!("Get chunk error: {e}"))
    }

    // ── Vector Search ──

    /// Brute-force KNN search over all chunks.
    /// Returns (chunk_id, similarity) pairs, sorted descending by similarity.
    /// If `use_enriched` is true, prefer enriched embeddings.
    pub fn vector_search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        use_enriched: bool,
    ) -> Result<Vec<(i64, f32)>, String> {
        let conn = self.conn()?;
        let sql = if use_enriched {
            "SELECT id, COALESCE(enriched_embedding, embedding)
             FROM chunks WHERE COALESCE(enriched_embedding, embedding) IS NOT NULL"
        } else {
            "SELECT id, embedding
             FROM chunks WHERE embedding IS NOT NULL"
        };
        let mut stmt = conn
            .prepare(sql)
            .map_err(|e| format!("Query error: {e}"))?;

        let mut scored: Vec<(i64, f32)> = stmt
            .query_map([], |row| {
                let id: i64 = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((id, blob))
            })
            .map_err(|e| format!("Query error: {e}"))?
            .filter_map(|r| r.ok())
            .map(|(id, blob)| {
                let emb = bytes_to_embedding(&blob);
                let sim = dot_product(query_embedding, &emb);
                (id, sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored)
    }

    /// Get full chunk records by IDs (preserving order).
    pub fn get_chunks_by_ids(&self, ids: &[i64]) -> Result<Vec<ChunkRecord>, String> {
        if ids.is_empty() {
            return Ok(vec![]);
        }
        let conn = self.conn()?;
        let placeholders: Vec<String> = ids.iter().map(|_| "?".to_string()).collect();
        let sql = format!(
            "SELECT id, doc_id, chunk_index, text, enriched_text, metadata, confidence
             FROM chunks WHERE id IN ({})",
            placeholders.join(",")
        );
        let mut stmt = conn.prepare(&sql).map_err(|e| format!("Prepare error: {e}"))?;

        let params: Vec<Box<dyn rusqlite::types::ToSql>> =
            ids.iter().map(|id| Box::new(*id) as Box<dyn rusqlite::types::ToSql>).collect();

        let rows: Vec<ChunkRecord> = stmt
            .query_map(rusqlite::params_from_iter(params.iter()), |row| {
                Ok(ChunkRecord {
                    id: row.get(0)?,
                    doc_id: row.get(1)?,
                    chunk_index: row.get(2)?,
                    text: row.get(3)?,
                    enriched_text: row.get(4)?,
                    metadata: row.get(5)?,
                    confidence: row.get(6)?,
                })
            })
            .map_err(|e| format!("Query error: {e}"))?
            .filter_map(|r| r.ok())
            .collect();

        // Re-order to match input ID order
        let mut ordered = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(rec) = rows.iter().find(|r| r.id == *id) {
                ordered.push(rec.clone());
            }
        }
        Ok(ordered)
    }

    /// Fetch adjacent (prev/next) chunks for context window expansion.
    /// Given a list of (doc_id, chunk_index) pairs, returns all chunks whose
    /// chunk_index is (chunk_index - 1) or (chunk_index + 1) within the same document.
    pub fn get_adjacent_chunks(
        &self,
        refs: &[(i64, i32)],
    ) -> Result<Vec<ChunkRecord>, String> {
        if refs.is_empty() {
            return Ok(vec![]);
        }
        let conn = self.conn()?;
        let mut result: Vec<ChunkRecord> = Vec::new();

        for &(doc_id, chunk_index) in refs {
            let prev_idx = chunk_index - 1;
            let next_idx = chunk_index + 1;
            let mut stmt = conn
                .prepare(
                    "SELECT id, doc_id, chunk_index, text, enriched_text, metadata, confidence
                     FROM chunks
                     WHERE doc_id = ?1 AND chunk_index IN (?2, ?3)
                     ORDER BY chunk_index ASC",
                )
                .map_err(|e| format!("Prepare error: {e}"))?;

            let rows = stmt
                .query_map(params![doc_id, prev_idx, next_idx], |row| {
                    Ok(ChunkRecord {
                        id: row.get(0)?,
                        doc_id: row.get(1)?,
                        chunk_index: row.get(2)?,
                        text: row.get(3)?,
                        enriched_text: row.get(4)?,
                        metadata: row.get(5)?,
                        confidence: row.get(6)?,
                    })
                })
                .map_err(|e| format!("Query error: {e}"))?;

            for row in rows.flatten() {
                // Avoid duplicates when multiple selected chunks share the same neighbor
                if !result.iter().any(|r| r.id == row.id) {
                    result.push(row);
                }
            }
        }
        Ok(result)
    }

    /// Get the document title for a chunk.
    pub fn doc_title_for_chunk(&self, chunk_id: i64) -> Result<String, String> {
        let conn = self.conn()?;
        conn.query_row(
            "SELECT d.title FROM documents d
             JOIN chunks c ON c.doc_id = d.id
             WHERE c.id = ?1",
            params![chunk_id],
            |row| row.get::<_, String>(0),
        )
        .map_err(|e| format!("Query error: {e}"))
    }

    // ── Bulk Embedding Access (for VectorIndex) ──

    /// Get all chunk embeddings (prefer enriched). For loading VectorIndex on startup.
    pub fn get_all_embeddings(&self) -> Result<Vec<(i64, Vec<f32>)>, String> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT id, COALESCE(enriched_embedding, embedding) FROM chunks
                 WHERE COALESCE(enriched_embedding, embedding) IS NOT NULL",
            )
            .map_err(|e| format!("Query error: {e}"))?;
        // SQLite access remains single-threaded: collect raw blobs first.
        let raw_rows: Vec<(i64, Vec<u8>)> = stmt
            .query_map([], |row| {
                let id: i64 = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((id, blob))
            })
            .map_err(|e| format!("Query error: {e}"))?
            .filter_map(|r| r.ok())
            .collect();

        // Decode blobs in parallel across CPU cores.
        let results: Vec<(i64, Vec<f32>)> = raw_rows
            .into_par_iter()
            .map(|(id, blob)| (id, bytes_to_embedding(&blob)))
            .collect();

        Ok(results)
    }

    /// Get embeddings for all chunks of a specific document (prefer enriched).
    pub fn get_chunk_embeddings_for_doc(&self, doc_id: i64) -> Result<Vec<(i64, Vec<f32>)>, String> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT id, COALESCE(enriched_embedding, embedding) FROM chunks
                 WHERE doc_id = ?1 AND COALESCE(enriched_embedding, embedding) IS NOT NULL",
            )
            .map_err(|e| format!("Query error: {e}"))?;
        let results = stmt
            .query_map(params![doc_id], |row| {
                let id: i64 = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((id, bytes_to_embedding(&blob)))
            })
            .map_err(|e| format!("Query error: {e}"))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(results)
    }

    // ── Media Asset CRUD ──

    /// Insert a media asset record. Returns the asset ID.
    pub fn insert_media_asset(
        &self,
        doc_id: i64,
        asset_type: &str,
        file_path: &str,
        caption: &str,
        metadata: &str,
    ) -> Result<i64, String> {
        let conn = self.conn()?;
        conn.execute(
            "INSERT INTO media_assets (doc_id, asset_type, file_path, caption, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![doc_id, asset_type, file_path, caption, metadata],
        )
        .map_err(|e| format!("Insert media asset error: {e}"))?;
        Ok(conn.last_insert_rowid())
    }

    /// Store the caption embedding for a media asset.
    pub fn set_media_embedding(
        &self,
        asset_id: i64,
        embedding: &[f32],
    ) -> Result<(), String> {
        let conn = self.conn()?;
        let bytes = embedding_to_bytes(embedding);
        conn.execute(
            "UPDATE media_assets SET embedding = ?1 WHERE id = ?2",
            params![bytes, asset_id],
        )
        .map_err(|e| format!("Set media embedding error: {e}"))
        .map(|_| ())
    }

    /// Get all media assets for a document.
    pub fn get_media_for_doc(&self, doc_id: i64) -> Result<Vec<MediaAssetRecord>, String> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT id, doc_id, asset_type, file_path, caption, metadata, caption_hash
                 FROM media_assets WHERE doc_id = ?1 ORDER BY id",
            )
            .map_err(|e| format!("Query error: {e}"))?;

        let assets = stmt
            .query_map(params![doc_id], |row| {
                Ok(MediaAssetRecord {
                    id: row.get(0)?,
                    doc_id: row.get(1)?,
                    asset_type: row.get(2)?,
                    file_path: row.get(3)?,
                    caption: row.get(4)?,
                    metadata: row.get(5)?,
                    caption_hash: row.get(6)?,
                })
            })
            .map_err(|e| format!("Query error: {e}"))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(assets)
    }

    /// Get a media asset by ID.
    pub fn get_media_asset(&self, asset_id: i64) -> Result<MediaAssetRecord, String> {
        let conn = self.conn()?;
        conn.query_row(
            "SELECT id, doc_id, asset_type, file_path, caption, metadata, caption_hash
             FROM media_assets WHERE id = ?1",
            params![asset_id],
            |row| {
                Ok(MediaAssetRecord {
                    id: row.get(0)?,
                    doc_id: row.get(1)?,
                    asset_type: row.get(2)?,
                    file_path: row.get(3)?,
                    caption: row.get(4)?,
                    metadata: row.get(5)?,
                    caption_hash: row.get(6)?,
                })
            },
        )
        .map_err(|e| format!("Get media asset error: {e}"))
    }

    /// Get all media asset embeddings (for loading into VectorIndex).
    /// Returns (negative_asset_id, embedding) — negative IDs distinguish media from chunks.
    pub fn get_all_media_embeddings(&self) -> Result<Vec<(i64, Vec<f32>)>, String> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT id, embedding FROM media_assets WHERE embedding IS NOT NULL",
            )
            .map_err(|e| format!("Query error: {e}"))?;
        let results = stmt
            .query_map([], |row| {
                let id: i64 = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                // Use negative IDs for media assets to distinguish from chunk IDs
                Ok((-id, bytes_to_embedding(&blob)))
            })
            .map_err(|e| format!("Query error: {e}"))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(results)
    }

    /// Get all media assets that were stored without embeddings (failed during ingestion).
    /// Returns (asset_id, caption) pairs.
    pub fn get_unembedded_media(&self) -> Result<Vec<(i64, String)>, String> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT id, caption FROM media_assets WHERE embedding IS NULL AND caption != ''",
            )
            .map_err(|e| format!("Query error: {e}"))?;
        let results = stmt
            .query_map([], |row| {
                let id: i64 = row.get(0)?;
                let caption: String = row.get(1)?;
                Ok((id, caption))
            })
            .map_err(|e| format!("Query error: {e}"))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(results)
    }

    /// Delete all media assets for a document (also deletes files from disk).
    pub fn delete_media_for_doc(&self, doc_id: i64) -> Result<Vec<String>, String> {
        let conn = self.conn()?;

        // Collect file paths first
        let mut stmt = conn
            .prepare("SELECT file_path FROM media_assets WHERE doc_id = ?1")
            .map_err(|e| format!("Query error: {e}"))?;
        let paths: Vec<String> = stmt
            .query_map(params![doc_id], |row| row.get::<_, String>(0))
            .map_err(|e| format!("Query error: {e}"))?
            .filter_map(|r| r.ok())
            .collect();

        conn.execute("DELETE FROM media_assets WHERE doc_id = ?1", params![doc_id])
            .map_err(|e| format!("Delete media error: {e}"))?;

        Ok(paths)
    }

    /// Search media assets by caption embedding similarity.
    /// Returns (asset_id, similarity) pairs sorted descending.
    pub fn media_vector_search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<(i64, f32)>, String> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare("SELECT id, embedding FROM media_assets WHERE embedding IS NOT NULL")
            .map_err(|e| format!("Query error: {e}"))?;

        let mut scored: Vec<(i64, f32)> = stmt
            .query_map([], |row| {
                let id: i64 = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((id, blob))
            })
            .map_err(|e| format!("Query error: {e}"))?
            .filter_map(|r| r.ok())
            .map(|(id, blob)| {
                let emb = bytes_to_embedding(&blob);
                let sim = dot_product(query_embedding, &emb);
                (id, sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored)
    }
}
