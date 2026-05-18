//! In-memory IVF (Inverted File) vector index for fast approximate nearest neighbor search.
//!
//! Replaces brute-force O(N) scan in SQLite with:
//!   - In-memory pre-loaded embeddings (no SQL/blob overhead per search)
//!   - IVF partitioning for sub-linear search when >128 vectors
//!   - Automatic index rebuild after threshold insertions
//!
//! For small corpora (<128 vectors), falls back to exact brute-force in-memory
//! search which is still much faster than the SQL-based scan.

use crate::rag::db::{dot_product, RagDb};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::RwLock;
use wide::f32x8;

/// TurboQuant-style symmetric per-vector int8 quantization.
///
/// This compresses each vector from f32 to i8 with a scalar scale factor,
/// reducing memory and improving cache locality for search.
#[derive(Clone)]
struct QuantizedVector {
    data: Vec<i8>,
    scale: f32,
}

impl QuantizedVector {
    fn from_f32(values: &[f32]) -> Self {
        let max_abs = values
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, |a, b| a.max(b));

        if max_abs < 1e-12 {
            return Self {
                data: vec![0; values.len()],
                scale: 1.0,
            };
        }

        let scale = max_abs / 127.0;
        let mut data = Vec::with_capacity(values.len());

        for &v in values {
            let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
            data.push(q);
        }

        Self { data, scale }
    }

    fn dequantize(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|&q| (q as f32) * self.scale)
            .collect()
    }

    fn dot_with_query(&self, query: &[f32]) -> f32 {
        if self.data.len() != query.len() {
            return 0.0;
        }

        let mut i = 0usize;
        let mut acc = f32x8::splat(0.0);

        while i + 8 <= self.data.len() {
            let qv = f32x8::from([
                self.data[i] as f32,
                self.data[i + 1] as f32,
                self.data[i + 2] as f32,
                self.data[i + 3] as f32,
                self.data[i + 4] as f32,
                self.data[i + 5] as f32,
                self.data[i + 6] as f32,
                self.data[i + 7] as f32,
            ]);
            let qq = f32x8::from([
                query[i],
                query[i + 1],
                query[i + 2],
                query[i + 3],
                query[i + 4],
                query[i + 5],
                query[i + 6],
                query[i + 7],
            ]);
            acc += qv * qq;
            i += 8;
        }

        let mut dot_q = acc.reduce_add();
        while i < self.data.len() {
            dot_q += (self.data[i] as f32) * query[i];
            i += 1;
        }

        dot_q * self.scale
    }

    fn compressed_bytes(&self) -> usize {
        self.data.len() + std::mem::size_of::<f32>()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Number of IVF partitions (Voronoi cells).
const IVF_NUM_CELLS: usize = 32;

/// Number of partitions to probe during search (higher = better recall, slower).
/// Rule of thumb: ~50% of IVF_NUM_CELLS for good recall on homogeneous corpora.
/// At 4 probes/32 cells only 12.5% of vectors are scanned, collapsing recall to near-random.
const IVF_NUM_PROBES: usize = 16;

/// Minimum vector count before IVF partitioning is used.
const MIN_VECTORS_FOR_IVF: usize = 128;

/// Number of new insertions before automatic IVF rebuild.
const REBUILD_THRESHOLD: usize = 100;

/// Maximum k-means iterations when building IVF centroids.
const KMEANS_MAX_ITER: usize = 20;

// ═══════════════════════════════════════════════════════════════════════════════
// VectorIndex
// ═══════════════════════════════════════════════════════════════════════════════

/// In-memory vector index with optional IVF partitioning.
///
/// For corpora under [`MIN_VECTORS_FOR_IVF`] vectors, performs exact brute-force
/// search in memory (already much faster than SQL blob deserialization).
/// Above that threshold, k-means IVF partitioning gives sub-linear search.
pub struct VectorIndex {
    /// All vectors, keyed by chunk_id.
    vectors: RwLock<HashMap<i64, QuantizedVector>>,
    /// IVF centroids (one per partition).
    centroids: RwLock<Vec<Vec<f32>>>,
    /// Partition assignments: centroids[i] owns partition_members[i].
    partition_members: RwLock<Vec<Vec<i64>>>,
    /// Insertions since last IVF rebuild.
    pending_inserts: AtomicUsize,
}

impl VectorIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self {
            vectors: RwLock::new(HashMap::new()),
            centroids: RwLock::new(Vec::new()),
            partition_members: RwLock::new(Vec::new()),
            pending_inserts: AtomicUsize::new(0),
        }
    }

    /// Load all existing embeddings from the database into memory,
    /// including both chunk embeddings and media caption embeddings.
    pub fn load_from_db(db: &RagDb) -> Result<Self, String> {
        let all_embeddings = db.get_all_embeddings()?;
        let quantized_chunks: Vec<(i64, QuantizedVector, usize, usize)> = all_embeddings
            .into_par_iter()
            .map(|(id, emb)| {
                let float_bytes = emb.len() * std::mem::size_of::<f32>();
                let q = QuantizedVector::from_f32(&emb);
                let quantized_bytes = q.compressed_bytes();
                (id, q, float_bytes, quantized_bytes)
            })
            .collect();

        let mut map = HashMap::with_capacity(quantized_chunks.len());
        let mut float_bytes = 0usize;
        let mut quantized_bytes = 0usize;
        for (id, q, f_bytes, q_bytes) in quantized_chunks {
            float_bytes += f_bytes;
            quantized_bytes += q_bytes;
            map.insert(id, q);
        }
        let chunk_count = map.len();

        // Also load media caption embeddings (stored with negative IDs)
        let media_embeddings = db.get_all_media_embeddings().unwrap_or_default();
        let media_count = media_embeddings.len();
        let quantized_media: Vec<(i64, QuantizedVector, usize, usize)> = media_embeddings
            .into_par_iter()
            .map(|(neg_id, emb)| {
                let float_bytes = emb.len() * std::mem::size_of::<f32>();
                let q = QuantizedVector::from_f32(&emb);
                let quantized_bytes = q.compressed_bytes();
                (neg_id, q, float_bytes, quantized_bytes)
            })
            .collect();
        for (neg_id, q, f_bytes, q_bytes) in quantized_media {
            float_bytes += f_bytes;
            quantized_bytes += q_bytes;
            map.insert(neg_id, q);
        }

        let saved_pct = if float_bytes == 0 {
            0.0
        } else {
            100.0 * (1.0 - (quantized_bytes as f64 / float_bytes as f64))
        };

        log::info!(
            "VectorIndex loaded: {} chunk vectors + {} media vectors = {} total (vec mem: {:.2} MB -> {:.2} MB, saved {:.1}%)",
            chunk_count,
            media_count,
            map.len(),
            float_bytes as f64 / (1024.0 * 1024.0),
            quantized_bytes as f64 / (1024.0 * 1024.0),
            saved_pct
        );

        let index = Self {
            vectors: RwLock::new(map),
            centroids: RwLock::new(Vec::new()),
            partition_members: RwLock::new(Vec::new()),
            pending_inserts: AtomicUsize::new(0),
        };

        Ok(index)
    }

    /// Insert or update a vector for a chunk.
    pub fn insert(&self, chunk_id: i64, embedding: Vec<f32>) {
        {
            let mut vecs = self.vectors.write().unwrap();
            vecs.insert(chunk_id, QuantizedVector::from_f32(&embedding));
        }
        self.pending_inserts.fetch_add(1, Ordering::Relaxed);
    }

    /// Remove a vector by chunk_id.
    pub fn remove(&self, chunk_id: i64) {
        let mut vecs = self.vectors.write().unwrap();
        vecs.remove(&chunk_id);
        // IVF partitions become slightly stale but search still works (skips missing IDs)
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.vectors.read().unwrap().len()
    }

    /// Dimension of stored embeddings (from the first vector), or 0 if empty.
    pub fn embedding_dim(&self) -> usize {
        self.vectors
            .read()
            .unwrap()
            .values()
            .next()
            .map(|v| v.data.len())
            .unwrap_or(0)
    }

    /// Approximate memory used by in-memory vector payloads (quantized vectors + IVF metadata).
    pub fn estimated_memory_bytes(&self) -> usize {
        let vecs = self.vectors.read().unwrap();
        let vec_bytes: usize = vecs.values().map(|v| v.compressed_bytes()).sum();

        let centroids = self.centroids.read().unwrap();
        let centroid_bytes: usize = centroids
            .iter()
            .map(|c| c.len() * std::mem::size_of::<f32>())
            .sum();

        let partitions = self.partition_members.read().unwrap();
        let partition_bytes: usize = partitions
            .iter()
            .map(|p| p.len() * std::mem::size_of::<i64>())
            .sum();

        vec_bytes + centroid_bytes + partition_bytes
    }

    /// Search for the top-k nearest neighbors by dot-product similarity.
    /// Uses IVF partitioning when available, falls back to brute-force in memory.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(i64, f32)> {
        let vecs = self.vectors.read().unwrap();
        if vecs.is_empty() {
            return vec![];
        }

        let centroids = self.centroids.read().unwrap();
        let partitions = self.partition_members.read().unwrap();

        let mut scored: Vec<(i64, f32)> =
            if centroids.is_empty() || vecs.len() < MIN_VECTORS_FOR_IVF {
                // Brute-force over all in-memory vectors
                vecs.iter()
                    .map(|(&id, emb)| (id, emb.dot_with_query(query)))
                    .collect()
            } else {
                // IVF: search only the nearest partitions
                let probe_indices = nearest_centroids(query, &centroids, IVF_NUM_PROBES);
                let mut seen = HashSet::new();
                let mut results = Vec::new();
                for &ci in &probe_indices {
                    if ci < partitions.len() {
                        for &chunk_id in &partitions[ci] {
                            if seen.insert(chunk_id) {
                                if let Some(emb) = vecs.get(&chunk_id) {
                                    results.push((chunk_id, emb.dot_with_query(query)));
                                }
                            }
                        }
                    }
                }
                results
            };

        // Partial sort: O(n) to find top-k, then sort only those
        if scored.len() > top_k {
            scored.select_nth_unstable_by(top_k, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(top_k);
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Rebuild IVF partitions if enough new vectors have been inserted.
    pub fn rebuild_if_needed(&self) {
        let pending = self.pending_inserts.load(Ordering::Relaxed);
        let should_bootstrap = if pending == 0 {
            let vec_len = self.vectors.read().unwrap().len();
            let centroids_empty = self.centroids.read().unwrap().is_empty();
            vec_len >= MIN_VECTORS_FOR_IVF && centroids_empty
        } else {
            false
        };

        if pending >= REBUILD_THRESHOLD || should_bootstrap {
            self.rebuild_ivf();
        }
    }

    /// Force rebuild of IVF partitions using k-means clustering.
    pub fn rebuild_ivf(&self) {
        let vecs = self.vectors.read().unwrap();
        if vecs.len() < MIN_VECTORS_FOR_IVF {
            // Too few vectors — clear partitions and use brute-force
            *self.centroids.write().unwrap() = Vec::new();
            *self.partition_members.write().unwrap() = Vec::new();
            self.pending_inserts.store(0, Ordering::Relaxed);
            return;
        }

        let num_cells = IVF_NUM_CELLS.min(vecs.len() / 4).max(2);

        // Collect (id, embedding_ref) pairs
        let items: Vec<(i64, Vec<f32>)> = vecs
            .iter()
            .map(|(&id, emb)| (id, emb.dequantize()))
            .collect();
        let embeddings: Vec<&Vec<f32>> = items.iter().map(|(_, e)| e).collect();

        // K-means clustering
        let (new_centroids, assignments) = kmeans(&embeddings, num_cells, KMEANS_MAX_ITER);

        // Build partition membership lists
        let mut new_partitions: Vec<Vec<i64>> = vec![Vec::new(); new_centroids.len()];
        for (i, &cluster) in assignments.iter().enumerate() {
            if cluster < new_partitions.len() {
                new_partitions[cluster].push(items[i].0);
            }
        }

        // Store results
        *self.centroids.write().unwrap() = new_centroids;
        *self.partition_members.write().unwrap() = new_partitions;
        self.pending_inserts.store(0, Ordering::Relaxed);

        log::info!(
            "VectorIndex IVF rebuilt: {} vectors across {} partitions",
            vecs.len(),
            num_cells
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// K-Means Helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Simple k-means clustering. Returns (centroids, per-item assignments).
fn kmeans(embeddings: &[&Vec<f32>], k: usize, max_iter: usize) -> (Vec<Vec<f32>>, Vec<usize>) {
    if embeddings.is_empty() || k == 0 {
        return (vec![], vec![]);
    }

    let k = k.min(embeddings.len());
    let dim = embeddings[0].len();

    // Initialize centroids evenly spread across the data
    let step = (embeddings.len() / k).max(1);
    let mut centroids: Vec<Vec<f32>> = (0..k)
        .map(|i| embeddings[(i * step).min(embeddings.len() - 1)].clone())
        .collect();

    let mut assignments = vec![0usize; embeddings.len()];

    for _ in 0..max_iter {
        // Assignment step
        let mut changed = false;
        for (i, emb) in embeddings.iter().enumerate() {
            let mut best = 0;
            let mut best_sim = f32::NEG_INFINITY;
            for (c, centroid) in centroids.iter().enumerate() {
                let sim = dot_product(emb, centroid);
                if sim > best_sim {
                    best_sim = sim;
                    best = c;
                }
            }
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update step
        let mut sums: Vec<Vec<f32>> = vec![vec![0.0; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, emb) in embeddings.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for (d, val) in emb.iter().enumerate() {
                sums[c][d] += val;
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim {
                    centroids[c][d] = sums[c][d] / counts[c] as f32;
                }
            }
        }
    }

    (centroids, assignments)
}

/// Find the `n` nearest centroids to a query vector.
fn nearest_centroids(query: &[f32], centroids: &[Vec<f32>], n: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, dot_product(query, c)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.iter().take(n).map(|(i, _)| *i).collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_index() {
        let idx = VectorIndex::new();
        assert_eq!(idx.len(), 0);
        let results = idx.search(&[1.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_insert_and_search() {
        let idx = VectorIndex::new();
        idx.insert(1, vec![1.0, 0.0, 0.0]);
        idx.insert(2, vec![0.0, 1.0, 0.0]);
        idx.insert(3, vec![0.9, 0.1, 0.0]);

        let results = idx.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1); // Exact match first
        assert_eq!(results[1].0, 3); // Near match second
    }

    #[test]
    fn test_remove() {
        let idx = VectorIndex::new();
        idx.insert(1, vec![1.0, 0.0, 0.0]);
        idx.insert(2, vec![0.0, 1.0, 0.0]);
        assert_eq!(idx.len(), 2);

        idx.remove(1);
        assert_eq!(idx.len(), 1);

        let results = idx.search(&[1.0, 0.0, 0.0], 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 2);
    }

    #[test]
    fn test_ivf_rebuild() {
        let idx = VectorIndex::new();
        // Insert enough vectors to trigger IVF
        for i in 0..200i64 {
            let angle = (i as f32) * std::f32::consts::PI / 100.0;
            idx.insert(i, vec![angle.cos(), angle.sin(), 0.0]);
        }

        idx.rebuild_ivf();

        // Search should still find correct nearest neighbors
        let results = idx.search(&[1.0, 0.0, 0.0], 3);
        assert_eq!(results.len(), 3);
        assert!(results[0].1 > 0.9); // Top result should be very similar
    }

    #[test]
    fn test_rebuild_if_needed() {
        let idx = VectorIndex::new();
        // Below threshold — should not rebuild
        for i in 0..50i64 {
            idx.insert(i, vec![1.0, 0.0, 0.0]);
        }
        idx.rebuild_if_needed();
        assert!(idx.centroids.read().unwrap().is_empty()); // Too few for IVF

        // Above both thresholds
        for i in 50..250i64 {
            idx.insert(i, vec![0.0, 1.0, 0.0]);
        }
        idx.rebuild_if_needed();
        assert!(!idx.centroids.read().unwrap().is_empty()); // IVF built
    }
}
