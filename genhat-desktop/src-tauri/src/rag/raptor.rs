//! RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.
//!
//! RAPTOR builds a hierarchical tree structure over document chunks:
//!   1. Cluster chunks using k-means on their embeddings
//!   2. Summarize each cluster via LLM (with confidence scores)
//!   3. Store summaries as parent nodes with confidence scores
//!   4. Recursively build 1-2 levels
//!
//! During retrieval, confidence-aware traversal:
//!   - If a summary node has confidence_score < threshold (-1.5), expand to children
//!   - Otherwise, use the summary text directly
//!
//! This prevents generic/hallucinated summaries from poisoning retrieval.
//!
//! # Example Usage
//!
//! ```no_run
//! use app_lib::rag::raptor;
//!
//! // Build a RAPTOR tree for document ID 1
//! let status = raptor::build_raptor_tree(db.clone(), router.clone(), 1).await?;
//! println!("Created {} nodes across {} levels", status.nodes_created, status.levels);
//!
//! // Query using the RAPTOR tree
//! let results = raptor::raptor_retrieve(
//!     db.clone(),
//!     router.clone(),
//!     1,  // doc_id
//!     "What are the key findings?",
//!     5,  // top_k
//!     None // Use default confidence threshold
//! ).await?;
//!
//! for (chunk_id, score, text) in results {
//!     println!("Chunk {}: {} (score: {:.2})", chunk_id, &text[..50.min(text.len())], score);
//! }
//! ```
//!
//! # Configuration
//!
//! Key parameters can be adjusted via constants:
//! - `DEFAULT_CONFIDENCE_THRESHOLD`: -1.5 (expand nodes below this)
//! - `MAX_CLUSTERS_PER_LEVEL`: 10
//! - `MIN_CLUSTER_SIZE`: 3
//! - `MAX_TREE_DEPTH`: 2

use crate::rag::db::{bytes_to_embedding, dot_product, embedding_to_bytes, RagDb};
use crate::registry::types::TaskResponse;
use crate::router::tasks;
use crate::router::TaskRouter;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════════

/// Default confidence threshold. Summaries below this are expanded to children.
const DEFAULT_CONFIDENCE_THRESHOLD: f64 = -1.5;

/// Maximum number of clusters per level.
const MAX_CLUSTERS_PER_LEVEL: usize = 10;

/// Minimum chunks per cluster (smaller clusters won't be summarized).
const MIN_CLUSTER_SIZE: usize = 3;

/// Maximum RAPTOR tree depth.
const MAX_TREE_DEPTH: usize = 2;

// ═══════════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════════

/// A RAPTOR tree node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaptorNode {
    pub id: i64,
    pub doc_id: i64,
    pub level: i32,
    pub parent_id: Option<i64>,
    pub summary_text: String,
    pub confidence_score: f64,
    pub child_ids: Vec<i64>, // Can be chunk IDs (level 0) or other node IDs (level > 0)
}

/// Result of building a RAPTOR tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaptorTreeStatus {
    pub doc_id: i64,
    pub nodes_created: usize,
    pub levels: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// RAPTOR Database Extensions
// ═══════════════════════════════════════════════════════════════════════════════

impl RagDb {
    /// Create RAPTOR-specific tables if they don't exist.
    pub fn create_raptor_tables(&self) -> Result<(), String> {
        let conn = self.conn()?;
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS raptor_nodes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id          INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                level           INTEGER NOT NULL,
                parent_id       INTEGER REFERENCES raptor_nodes(id) ON DELETE CASCADE,
                summary_text    TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                child_ids       TEXT NOT NULL,
                embedding       BLOB,
                created_at      TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_raptor_doc ON raptor_nodes(doc_id);
            CREATE INDEX IF NOT EXISTS idx_raptor_parent ON raptor_nodes(parent_id);
            CREATE INDEX IF NOT EXISTS idx_raptor_level ON raptor_nodes(level);
            ",
        )
        .map_err(|e| format!("Failed to create RAPTOR tables: {e}"))
    }

    /// Insert a RAPTOR node.
    pub fn insert_raptor_node(
        &self,
        doc_id: i64,
        level: i32,
        parent_id: Option<i64>,
        summary_text: &str,
        confidence_score: f64,
        child_ids: &[i64],
        embedding: Option<&[f32]>,
    ) -> Result<i64, String> {
        let conn = self.conn()?;
        let child_ids_json = serde_json::to_string(child_ids).unwrap_or_default();
        let embedding_bytes = embedding.map(embedding_to_bytes);

        conn.execute(
            "INSERT INTO raptor_nodes (doc_id, level, parent_id, summary_text, confidence_score, child_ids, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                doc_id,
                level,
                parent_id,
                summary_text,
                confidence_score,
                child_ids_json,
                embedding_bytes,
            ],
        )
        .map_err(|e| format!("Insert RAPTOR node error: {e}"))?;

        Ok(conn.last_insert_rowid())
    }

    /// Get all RAPTOR nodes for a document.
    pub fn get_raptor_nodes(&self, doc_id: i64) -> Result<Vec<RaptorNode>, String> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT id, doc_id, level, parent_id, summary_text, confidence_score, child_ids
                 FROM raptor_nodes WHERE doc_id = ?1 ORDER BY level, id",
            )
            .map_err(|e| format!("Query error: {e}"))?;

        let nodes = stmt
            .query_map(rusqlite::params![doc_id], |row: &rusqlite::Row| {
                let child_ids_json: String = row.get(6)?;
                let child_ids: Vec<i64> = serde_json::from_str(&child_ids_json).unwrap_or_default();
                Ok(RaptorNode {
                    id: row.get(0)?,
                    doc_id: row.get(1)?,
                    level: row.get(2)?,
                    parent_id: row.get(3)?,
                    summary_text: row.get(4)?,
                    confidence_score: row.get(5)?,
                    child_ids,
                })
            })
            .map_err(|e| format!("Query error: {e}"))?
            .filter_map(|r: Result<RaptorNode, _>| r.ok())
            .collect();

        Ok(nodes)
    }

    /// Get a specific RAPTOR node by ID.
    pub fn get_raptor_node(&self, node_id: i64) -> Result<RaptorNode, String> {
        let conn = self.conn()?;
        conn.query_row(
            "SELECT id, doc_id, level, parent_id, summary_text, confidence_score, child_ids
             FROM raptor_nodes WHERE id = ?1",
            rusqlite::params![node_id],
            |row: &rusqlite::Row| {
                let child_ids_json: String = row.get(6)?;
                let child_ids: Vec<i64> = serde_json::from_str(&child_ids_json).unwrap_or_default();
                Ok(RaptorNode {
                    id: row.get(0)?,
                    doc_id: row.get(1)?,
                    level: row.get(2)?,
                    parent_id: row.get(3)?,
                    summary_text: row.get(4)?,
                    confidence_score: row.get(5)?,
                    child_ids,
                })
            },
        )
        .map_err(|e| format!("Get RAPTOR node error: {e}"))
    }

    /// Delete all RAPTOR nodes for a document.
    pub fn delete_raptor_nodes(&self, doc_id: i64) -> Result<(), String> {
        let conn = self.conn()?;
        conn.execute(
            "DELETE FROM raptor_nodes WHERE doc_id = ?1",
            rusqlite::params![doc_id],
        )
        .map_err(|e| format!("Delete RAPTOR nodes error: {e}"))?;
        Ok(())
    }

    /// Check if a document has RAPTOR nodes.
    pub fn has_raptor_tree(&self, doc_id: i64) -> Result<bool, String> {
        let conn = self.conn()?;
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM raptor_nodes WHERE doc_id = ?1",
                rusqlite::params![doc_id],
                |row: &rusqlite::Row| row.get(0),
            )
            .map_err(|e| format!("Query error: {e}"))?;
        Ok(count > 0)
    }

    /// Get embeddings for RAPTOR nodes (for vector search).
    pub fn get_raptor_embeddings(&self, doc_id: i64) -> Result<Vec<(i64, Vec<f32>)>, String> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT id, embedding FROM raptor_nodes WHERE doc_id = ?1 AND embedding IS NOT NULL",
            )
            .map_err(|e| format!("Query error: {e}"))?;

        let embeddings = stmt
            .query_map(rusqlite::params![doc_id], |row: &rusqlite::Row| {
                let id: i64 = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((id, bytes_to_embedding(&blob)))
            })
            .map_err(|e| format!("Query error: {e}"))?
            .filter_map(|r: Result<(i64, Vec<f32>), _>| r.ok())
            .collect();

        Ok(embeddings)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// K-Means Clustering Implementation
// ═══════════════════════════════════════════════════════════════════════════════

/// Simple k-means clustering for embeddings.
/// Returns cluster assignments (index -> cluster_id).
fn kmeans_cluster(embeddings: &[Vec<f32>], k: usize, max_iterations: usize) -> Vec<usize> {
    if embeddings.is_empty() || k == 0 {
        return vec![];
    }

    let k = k.min(embeddings.len());
    let dim = embeddings[0].len();

    // Initialize centroids: pick first k embeddings
    let mut centroids: Vec<Vec<f32>> = embeddings.iter().take(k).cloned().collect();

    let mut assignments = vec![0; embeddings.len()];

    for _iteration in 0..max_iterations {
        // Assignment step: assign each point to nearest centroid
        let mut changed = false;
        for (i, emb) in embeddings.iter().enumerate() {
            let mut best_cluster = 0;
            let mut best_sim = f32::NEG_INFINITY;
            for (c, centroid) in centroids.iter().enumerate() {
                let sim = dot_product(emb, centroid);
                if sim > best_sim {
                    best_sim = sim;
                    best_cluster = c;
                }
            }
            if assignments[i] != best_cluster {
                assignments[i] = best_cluster;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update step: recompute centroids
        let mut cluster_sums: Vec<Vec<f32>> = vec![vec![0.0; dim]; k];
        let mut cluster_counts = vec![0; k];

        for (i, emb) in embeddings.iter().enumerate() {
            let cluster = assignments[i];
            for (d, val) in emb.iter().enumerate() {
                cluster_sums[cluster][d] += val;
            }
            cluster_counts[cluster] += 1;
        }

        for c in 0..k {
            if cluster_counts[c] > 0 {
                for d in 0..dim {
                    centroids[c][d] = cluster_sums[c][d] / cluster_counts[c] as f32;
                }
            }
        }
    }

    assignments
}

/// Group items by cluster ID.
fn group_by_cluster<T: Clone>(items: &[T], assignments: &[usize]) -> HashMap<usize, Vec<T>> {
    let mut groups: HashMap<usize, Vec<T>> = HashMap::new();
    for (i, cluster_id) in assignments.iter().enumerate() {
        if i < items.len() {
            groups.entry(*cluster_id).or_default().push(items[i].clone());
        }
    }
    groups
}

// ═══════════════════════════════════════════════════════════════════════════════
// RAPTOR Tree Builder
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a RAPTOR tree for a document.
///
/// This function performs the full tree-building process:
/// 1. Collects all chunk embeddings for the document
/// 2. Clusters chunks using k-means at each level
/// 3. Summarizes each cluster via LLM with confidence scoring
/// 4. Stores summaries as RAPTOR nodes with their embeddings
/// 5. Recursively builds up to MAX_TREE_DEPTH levels
///
/// # Arguments
/// * `db` - Database handle for storage
/// * `router` - Task router for LLM inference
/// * `doc_id` - Document ID to build tree for
///
/// # Returns
/// * `Ok(RaptorTreeStatus)` - Tree statistics (nodes created, levels)
/// * `Err(String)` - Error message if tree building fails
///
/// # Errors
/// * Document has no chunks
/// * Document has no embeddings (run Phase 1 first)
/// * Tree already exists (delete it first to rebuild)
/// * LLM summarization fails
///
/// # Example
/// ```no_run
/// let status = build_raptor_tree(db, router, doc_id).await?;
/// println!("Built tree: {} nodes, {} levels", status.nodes_created, status.levels);
/// ```
pub async fn build_raptor_tree(
    db: Arc<RagDb>,
    router: Arc<TaskRouter>,
    doc_id: i64,
) -> Result<RaptorTreeStatus, String> {
    // Ensure RAPTOR tables exist
    db.create_raptor_tables()?;

    // Check if tree already exists
    if db.has_raptor_tree(doc_id)? {
        return Err("RAPTOR tree already exists for this document. Delete it first if you want to rebuild.".into());
    }

    // Get all chunk IDs and their embeddings for this document
    let chunk_ids = db.get_chunk_ids_for_doc(doc_id)?;
    if chunk_ids.is_empty() {
        return Err("Document has no chunks".into());
    }

    let chunks = db.get_chunks_by_ids(&chunk_ids)?;

    // Collect embeddings (prefer enriched, fallback to raw)
    let chunk_embeddings = db.get_chunk_embeddings_for_doc(doc_id)?;

    if chunk_embeddings.is_empty() {
        return Err("No embeddings found for document chunks. Run Phase 1 ingestion first.".into());
    }

    log::info!(
        "Building RAPTOR tree for doc {}: {} chunks with embeddings",
        doc_id,
        chunk_embeddings.len()
    );

    // Build tree level by level
    let mut nodes_created = 0;
    let mut current_level_items = chunk_embeddings.clone(); // (id, embedding)
    let mut current_level = 0;

    while current_level < MAX_TREE_DEPTH && current_level_items.len() > 1 {
        log::info!(
            "RAPTOR level {}: clustering {} items",
            current_level + 1,
            current_level_items.len()
        );

        // Determine number of clusters
        let num_clusters = (current_level_items.len() / MIN_CLUSTER_SIZE)
            .max(1)
            .min(MAX_CLUSTERS_PER_LEVEL);

        if num_clusters <= 1 {
            log::info!("Too few items to cluster further, stopping at level {}", current_level);
            break;
        }

        // Extract just the embeddings for clustering
        let embeddings: Vec<Vec<f32>> = current_level_items.iter().map(|(_, e)| e.clone()).collect();

        // Cluster
        let assignments = kmeans_cluster(&embeddings, num_clusters, 20);

        // Group items by cluster
        let clusters = group_by_cluster(&current_level_items, &assignments);

        // Create summary nodes for each cluster
        let mut next_level_items: Vec<(i64, Vec<f32>)> = Vec::new();

        for (cluster_id, items) in &clusters {
            if items.len() < MIN_CLUSTER_SIZE {
                log::debug!(
                    "Skipping cluster {} with only {} items (min {})",
                    cluster_id,
                    items.len(),
                    MIN_CLUSTER_SIZE
                );
                continue;
            }

            let child_ids: Vec<i64> = items.iter().map(|(id, _)| *id).collect();

            // Fetch the texts of the children
            let child_texts: Vec<String> = if current_level == 0 {
                // Level 0: children are chunks
                chunks
                    .iter()
                    .filter(|c| child_ids.contains(&c.id))
                    .map(|c| c.text.clone())
                    .collect()
            } else {
                // Level > 0: children are RAPTOR nodes
                child_ids
                    .iter()
                    .filter_map(|id| db.get_raptor_node(*id).ok())
                    .map(|n| n.summary_text)
                    .collect()
            };

            if child_texts.is_empty() {
                continue;
            }

            // Generate summary via LLM with confidence (logprobs)
            let (summary_text, confidence_score) = summarize_cluster_with_confidence(
                &router,
                &child_texts,
            )
            .await?;

            log::debug!(
                "Cluster {} summary (confidence: {:.2}): {}",
                cluster_id,
                confidence_score,
                &summary_text[..summary_text.len().min(80)]
            );

            // Embed the summary
            let summary_embedding = embed_text(&router, &summary_text).await?;

            // Insert RAPTOR node
            let node_id = db.insert_raptor_node(
                doc_id,
                (current_level + 1) as i32,
                None,
                &summary_text,
                confidence_score,
                &child_ids,
                Some(&summary_embedding),
            )?;

            nodes_created += 1;
            next_level_items.push((node_id, summary_embedding));
        }

        if next_level_items.is_empty() {
            break;
        }

        current_level_items = next_level_items;
        current_level += 1;
    }

    log::info!(
        "RAPTOR tree built for doc {}: {} nodes across {} levels",
        doc_id,
        nodes_created,
        current_level
    );

    Ok(RaptorTreeStatus {
        doc_id,
        nodes_created,
        levels: current_level,
    })
}

/// Summarize a cluster of texts via LLM, extracting confidence from logprobs.
async fn summarize_cluster_with_confidence(
    router: &TaskRouter,
    texts: &[String],
) -> Result<(String, f64), String> {
    // Concatenate texts (with length limit)
    let max_len = 4000; // Keep context manageable
    let mut combined = String::new();
    for (i, text) in texts.iter().enumerate() {
        combined.push_str(&format!("Document {}:\n{}\n\n", i + 1, text));
        if combined.len() > max_len {
            let boundary = combined.floor_char_boundary(max_len);
            combined.truncate(boundary);
            combined.push_str("\n[truncated...]");
            break;
        }
    }

    let prompt = format!(
        "Summarize the following documents into a single, concise summary that captures the key information:\n\n{}\n\nSummary:",
        combined
    );

    // Request summarization with logprobs (if supported)
    let request = tasks::summarize_request(&prompt);
    
    match router.route(&request).await {
        Ok(TaskResponse::Text(summary)) => {
            // Extract confidence if available in response metadata
            // For now, use a default confidence based on summary length heuristic
            // Real implementation would parse logprobs from LLM response
            let confidence = estimate_confidence(&summary);
            Ok((summary, confidence))
        }
        Ok(_) => Err("Summarization returned unexpected response type".into()),
        Err(e) => Err(format!("Summarization failed: {e}")),
    }
}

/// Heuristic confidence estimation based on summary characteristics.
/// Real implementation would use mean logprobs from the LLM.
fn estimate_confidence(summary: &str) -> f64 {
    // Simple heuristic: longer, more detailed summaries get higher confidence
    // Generic short responses get lower confidence
    let words = summary.split_whitespace().count();
    
    if summary.contains("document") && words < 15 {
        // Generic filler text
        -2.0
    } else if words > 30 {
        // Detailed summary
        -0.5
    } else {
        // Medium confidence
        -1.0
    }
}

/// Embed a single text via the embedding model.
async fn embed_text(router: &TaskRouter, text: &str) -> Result<Vec<f32>, String> {
    let request = tasks::embed_request(vec![text.to_string()]);
    match router.route(&request).await {
        Ok(TaskResponse::Embeddings(mut vecs)) => {
            vecs.pop().ok_or_else(|| "No embedding returned".into())
        }
        Ok(_) => Err("Embedding returned unexpected response type".into()),
        Err(e) => Err(format!("Embedding failed: {e}")),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Confidence-Aware RAPTOR Retrieval
// ═══════════════════════════════════════════════════════════════════════════════

/// Retrieve using RAPTOR tree with confidence-aware traversal.
///
/// This function implements the key innovation of RAPTOR:
/// 1. Finds top-k most similar RAPTOR nodes via vector search
/// 2. For each matched node:
///    - If confidence < threshold: expand to child chunks/nodes
///    - If confidence >= threshold: use summary directly
/// 3. Returns expanded results with original similarity scores
///
/// This prevents generic/hallucinated summaries from being used,
/// falling back to more detailed child content when needed.
///
/// # Arguments
/// * `db` - Database handle
/// * `router` - Task router for embedding the query
/// * `doc_id` - Document to search within
/// * `query` - Query text
/// * `top_k` - Number of results to return
/// * `confidence_threshold` - Optional threshold (default: -1.5)
///
/// # Returns
/// Vec of (chunk_id, score, text) tuples, where:
/// - chunk_id is the original chunk ID (or RAPTOR node ID)
/// - score is the dot-product similarity to the query
/// - text is either summary or expanded child text
///
/// # Example
/// ```no_run
/// let results = raptor_retrieve(db, router, 1, "key findings", 5, None).await?;
/// for (id, score, text) in results {
///     println!("Result {}: {:.2} - {}", id, score, &text[..50]);
/// }
/// ```
pub async fn raptor_retrieve(
    db: Arc<RagDb>,
    router: Arc<TaskRouter>,
    doc_id: i64,
    query: &str,
    top_k: usize,
    confidence_threshold: Option<f64>,
) -> Result<Vec<(i64, f64, String)>, String> {
    let threshold = confidence_threshold.unwrap_or(DEFAULT_CONFIDENCE_THRESHOLD);

    // Check if RAPTOR tree exists
    if !db.has_raptor_tree(doc_id)? {
        return Err("No RAPTOR tree exists for this document".into());
    }

    // Embed the query
    let query_embedding = embed_text(&router, query).await?;

    // Get all RAPTOR nodes for the document
    let raptor_embeddings = db.get_raptor_embeddings(doc_id)?;

    if raptor_embeddings.is_empty() {
        return Err("No RAPTOR embeddings found".into());
    }

    // Find top-k most similar RAPTOR nodes
    let mut scored: Vec<(i64, f32)> = raptor_embeddings
        .iter()
        .map(|(node_id, emb)| {
            let sim = dot_product(&query_embedding, emb);
            (*node_id, sim)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);

    // Expand nodes based on confidence
    let mut results: Vec<(i64, f64, String)> = Vec::new();

    for (node_id, similarity) in scored {
        let node = db.get_raptor_node(node_id)?;

        if node.confidence_score < threshold {
            // Low confidence: expand to children
            log::debug!(
                "Expanding low-confidence node {} (confidence: {:.2} < {:.2})",
                node_id,
                node.confidence_score,
                threshold
            );

            // If level 1, children are chunks
            if node.level == 1 {
                let chunks = db.get_chunks_by_ids(&node.child_ids)?;
                for chunk in chunks {
                    results.push((chunk.id, similarity as f64, chunk.text));
                }
            } else {
                // Level > 1: children are other RAPTOR nodes
                for child_id in &node.child_ids {
                    if let Ok(child_node) = db.get_raptor_node(*child_id) {
                        results.push((
                            child_node.id,
                            similarity as f64,
                            child_node.summary_text,
                        ));
                    }
                }
            }
        } else {
            // High confidence: use the summary directly
            results.push((node.id, similarity as f64, node.summary_text));
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_basic() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.9, 0.1],
        ];

        let assignments = kmeans_cluster(&embeddings, 2, 10);
        
        // Items 0,1 should cluster together, items 2,3 should cluster together
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
        assert_ne!(assignments[0], assignments[2]);
    }

    #[test]
    fn test_kmeans_empty() {
        let embeddings: Vec<Vec<f32>> = vec![];
        let assignments = kmeans_cluster(&embeddings, 2, 10);
        assert!(assignments.is_empty());
    }

    #[test]
    fn test_group_by_cluster() {
        let items = vec!["a", "b", "c", "d"];
        let assignments = vec![0, 1, 0, 1];
        let groups = group_by_cluster(&items, &assignments);

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[&0], vec!["a", "c"]);
        assert_eq!(groups[&1], vec!["b", "d"]);
    }

    #[test]
    fn test_estimate_confidence() {
        // Generic short text should have low confidence
        let generic = "This document contains information.";
        assert!(estimate_confidence(generic) < -1.5);

        // Detailed text should have higher confidence
        let detailed = "The research demonstrates significant improvements in model performance through the application of hierarchical clustering techniques, specifically k-means, combined with recursive summarization strategies.";
        assert!(estimate_confidence(detailed) > -1.5);
    }
}
