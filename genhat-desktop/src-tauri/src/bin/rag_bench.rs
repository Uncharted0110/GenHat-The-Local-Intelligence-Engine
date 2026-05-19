//! NELA RAG Benchmark CLI
//!
//! Measures retrieval recall, latency, and index efficiency of the NELA RAG
//! pipeline components.  Runs without the Tauri runtime; uses the lower-level
//! `app_lib::rag::*` APIs directly.
//!
//! # Usage
//!
//! ```sh
//! # Step 1 – ingest a corpus of .txt files into a fresh workspace
//! rag-bench ingest \
//!   --workspace-dir /tmp/nela-bench \
//!   --corpus-dir    ./docs \
//!   --embed-model   ./models/bge-base.gguf
//!
//! # Step 2 – run benchmarks against the ingested workspace
//! rag-bench bench \
//!   --workspace-dir /tmp/nela-bench \
//!   --qa-file       ./qa_pairs.json \
//!   --embed-model   ./models/bge-base.gguf \
//!   --output        bench_results.json
//!
//! # Or do both in one shot
//! rag-bench run \
//!   --workspace-dir /tmp/nela-bench \
//!   --corpus-dir    ./docs \
//!   --qa-file       ./qa_pairs.json \
//!   --embed-model   ./models/bge-base.gguf
//! ```
//!
//! llama-server is auto-detected from the NELA repo's `bin/llama-lin/` directory.
//! Override with `--llama-server <path>`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures_util::stream::{self, StreamExt};

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use app_lib::rag::{
    chunker::{chunk_text, chunk_text_default, Chunk, ChunkerConfig},
    db::{dot_product, RagDb},
    fusion::{rrf_fuse, rrf_fuse_with_k},
    raptor::RaptorNode,
    search::BM25Index,
    vecindex::VectorIndex,
};

// ── CLI definition ────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "rag-bench", about = "NELA RAG Pipeline Benchmark Tool")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Ingest a corpus of .txt documents into a fresh benchmark workspace.
    Ingest(IngestArgs),
    /// Run recall, latency, and index benchmarks against an ingested workspace.
    Bench(BenchArgs),
    /// Ingest corpus then immediately run all benchmarks (combines ingest + bench).
    Run(RunArgs),
    /// Incrementally ingest corpus and measure recall at each corpus-size checkpoint.
    Scale(ScaleArgs),
    /// Run E2E answer-quality eval (EM + token F1) against an already-ingested workspace.
    /// Skips recall/latency benchmarks — use when you only want LLM answer scoring.
    Eval(EvalArgs),
    /// Benchmark retrieval on a BEIR-format dataset (NDCG@10, MAP, Recall@100).
    /// Expects <beir-dir>/corpus.jsonl, queries.jsonl, and qrels/test.tsv.
    BeirBench(BeirBenchArgs),
    /// Ablation study: chunk size × overlap grid search over hybrid recall.
    AblateChunking(AblateChunkingArgs),
    /// Ablation study: RRF k-constant sensitivity [10,30,60,100,200].
    AblateRrfK(AblateRrfKArgs),
    /// Ablation study: embedding model quantization comparison (multiple GGUF models).
    AblateQuant(AblateQuantArgs),
}

#[derive(Args, Clone)]
struct IngestArgs {
    /// Directory to create the benchmark workspace in (rag.db + bm25_index).
    #[arg(long)]
    workspace_dir: PathBuf,

    /// Directory of .txt documents to ingest.
    #[arg(long)]
    corpus_dir: PathBuf,

    /// Path to the GGUF embedding model (e.g. bge-base-en-v1.5-q8_0.gguf).
    #[arg(long)]
    embed_model: PathBuf,

    /// Path to the llama-server binary [auto-detected from repo if not provided].
    #[arg(long)]
    llama_server: Option<PathBuf>,

    /// Port to run the embedding server on.
    #[arg(long, default_value_t = 12345)]
    embed_port: u16,

    /// Build a RAPTOR hierarchical summary tree after standard ingestion.
    /// Requires --llm-model.  Significantly increases ingest time.
    #[arg(long, default_value_t = false)]
    raptor: bool,

    /// GGUF chat model used for RAPTOR cluster summarization.
    #[arg(long)]
    llm_model: Option<PathBuf>,

    /// Port for the RAPTOR LLM summarization server.
    #[arg(long, default_value_t = 12346)]
    llm_port: u16,
}

#[derive(Args, Clone)]
struct BenchArgs {
    /// Workspace directory (must have been prepared with `ingest` first).
    #[arg(long)]
    workspace_dir: PathBuf,

    /// JSON file with QA pairs.  Format: [{question, relevant_keywords, doc_title?}]
    #[arg(long)]
    qa_file: PathBuf,

    /// Path to the GGUF embedding model.
    #[arg(long)]
    embed_model: PathBuf,

    /// Path to the llama-server binary [auto-detected from repo if not provided].
    #[arg(long)]
    llama_server: Option<PathBuf>,

    /// Port to run the embedding server on.
    #[arg(long, default_value_t = 12345)]
    embed_port: u16,

    /// Comma-separated k values for Recall@k (e.g. "5,10").
    #[arg(long, default_value = "5,10")]
    top_k: String,

    /// Path for the output JSON results file.
    #[arg(long, default_value = "bench_results.json")]
    output: PathBuf,

    /// Also run RAPTOR confidence-gate ablation (requires pre-built RAPTOR trees
    /// in the workspace, e.g. by ingesting via NELA first).
    #[arg(long, default_value_t = false)]
    raptor: bool,

    /// Path to a GGUF chat/generation model for end-to-end answer quality evaluation.
    /// If omitted, the E2E eval is skipped.
    #[arg(long)]
    llm_model: Option<PathBuf>,

    /// Port for the LLM generation server (must differ from --embed-port).
    #[arg(long, default_value_t = 12346)]
    llm_port: u16,

    /// Maximum number of QA pairs to run through the LLM for E2E eval.
    /// Use 500 for publication-quality results; lower for quick iteration.
    #[arg(long, default_value_t = 500)]
    e2e_count: usize,

    /// Also run a no-RAG baseline: same questions sent to the LLM without any
    /// retrieved context.  Requires --llm-model.  Results are printed alongside
    /// the RAG E2E score so the retrieval gain is immediately visible.
    #[arg(long, default_value_t = false)]
    no_rag_baseline: bool,

    /// Bootstrap resampling iterations for E2E confidence intervals.
    #[arg(long, default_value_t = 1000)]
    bootstrap_samples: usize,

    /// Random seed for reproducible bootstrap sampling.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Args)]
struct RunArgs {
    #[arg(long)]
    workspace_dir: PathBuf,

    #[arg(long)]
    corpus_dir: PathBuf,

    #[arg(long)]
    qa_file: PathBuf,

    #[arg(long)]
    embed_model: PathBuf,

    #[arg(long)]
    llama_server: Option<PathBuf>,

    #[arg(long, default_value_t = 12345)]
    embed_port: u16,

    #[arg(long, default_value = "5,10")]
    top_k: String,

    #[arg(long, default_value = "bench_results.json")]
    output: PathBuf,

    #[arg(long, default_value_t = false)]
    raptor: bool,

    /// Path to a GGUF chat/generation model for end-to-end answer quality evaluation.
    #[arg(long)]
    llm_model: Option<PathBuf>,

    /// Port for the LLM generation server.
    #[arg(long, default_value_t = 12346)]
    llm_port: u16,

    /// Maximum number of QA pairs to run through the LLM for E2E eval.
    #[arg(long, default_value_t = 500)]
    e2e_count: usize,

    /// Also run a no-RAG baseline: same questions sent to the LLM without any
    /// retrieved context.  Requires --llm-model.  Results are printed alongside
    /// the RAG E2E score so the retrieval gain is immediately visible.
    #[arg(long, default_value_t = false)]
    no_rag_baseline: bool,

    /// Bootstrap resampling iterations for E2E confidence intervals.
    #[arg(long, default_value_t = 1000)]
    bootstrap_samples: usize,

    /// Random seed for reproducible bootstrap sampling.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Args)]
struct ScaleArgs {
    /// Workspace directory (created if absent; reused across scale checkpoints).
    #[arg(long)]
    workspace_dir: PathBuf,

    /// Directory of .txt documents to ingest.
    #[arg(long)]
    corpus_dir: PathBuf,

    /// JSON file with QA pairs (same format as used by `bench`).
    #[arg(long)]
    qa_file: PathBuf,

    /// Path to the GGUF embedding model.
    #[arg(long)]
    embed_model: PathBuf,

    /// Path to the llama-server binary [auto-detected if not provided].
    #[arg(long)]
    llama_server: Option<PathBuf>,

    /// Port to run the embedding server on.
    #[arg(long, default_value_t = 12345)]
    embed_port: u16,

    /// Comma-separated list of corpus sizes (number of docs) to test at each checkpoint.
    #[arg(long, default_value = "100,500,1000,2000")]
    sizes: String,

    /// Number of QA pairs to sample at each checkpoint (controls eval speed vs accuracy).
    #[arg(long, default_value_t = 500)]
    qa_sample: usize,

    /// Path for scale results output file.
    #[arg(long, default_value = "scale_results.json")]
    output: PathBuf,
}

#[derive(Args)]
struct EvalArgs {
    /// Existing benchmark workspace directory (must already be ingested).
    #[arg(long)]
    workspace_dir: PathBuf,

    /// JSON file with QA pairs (must include 'answers' field — see prepare_squad.py).
    #[arg(long)]
    qa_file: PathBuf,

    /// Path to the GGUF embedding model (same model used during ingest).
    #[arg(long)]
    embed_model: PathBuf,

    /// Path to the GGUF chat/generation model.
    #[arg(long)]
    llm_model: PathBuf,

    /// Path to the llama-server binary [auto-detected from repo if not provided].
    #[arg(long)]
    llama_server: Option<PathBuf>,

    /// Port for the embedding server.
    #[arg(long, default_value_t = 12345)]
    embed_port: u16,

    /// Port for the LLM generation server (must differ from --embed-port).
    #[arg(long, default_value_t = 12346)]
    llm_port: u16,

    /// Maximum number of QA pairs to evaluate.
    #[arg(long, default_value_t = 500)]
    count: usize,

    /// Bootstrap resampling iterations for confidence intervals.
    #[arg(long, default_value_t = 1000)]
    bootstrap_samples: usize,

    /// Random seed for reproducible bootstrap sampling.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Output JSON file for E2E results.
    #[arg(long, default_value = "e2e_results.json")]
    output: PathBuf,
}

#[derive(Args)]
struct BeirBenchArgs {
    /// Workspace directory for the ingested BEIR corpus (created if absent).
    #[arg(long)]
    workspace_dir: PathBuf,

    /// Directory containing BEIR dataset: corpus.jsonl, queries.jsonl, qrels/test.tsv.
    #[arg(long)]
    beir_dir: PathBuf,

    /// Path to the GGUF embedding model.
    #[arg(long)]
    embed_model: PathBuf,

    /// Path to the llama-server binary [auto-detected if not provided].
    #[arg(long)]
    llama_server: Option<PathBuf>,

    /// Port for the embedding server.
    #[arg(long, default_value_t = 12345)]
    embed_port: u16,

    /// Output JSON file.
    #[arg(long, default_value = "beir_results.json")]
    output: PathBuf,
}

#[derive(Args)]
struct AblateChunkingArgs {
    /// Base workspace directory (fresh sub-dirs created per grid point, then removed).
    #[arg(long)]
    workspace_dir: PathBuf,

    /// Directory of .txt documents to ingest.
    #[arg(long)]
    corpus_dir: PathBuf,

    /// JSON file with QA pairs.
    #[arg(long)]
    qa_file: PathBuf,

    /// Path to the GGUF embedding model.
    #[arg(long)]
    embed_model: PathBuf,

    /// Path to the llama-server binary [auto-detected if not provided].
    #[arg(long)]
    llama_server: Option<PathBuf>,

    /// Port for the embedding server.
    #[arg(long, default_value_t = 12345)]
    embed_port: u16,

    /// Comma-separated chunk sizes (characters) to test.
    #[arg(long, default_value = "512,1024,1536,2048")]
    chunk_sizes: String,

    /// Comma-separated overlap sizes (characters) to test.
    #[arg(long, default_value = "64,128,256")]
    overlaps: String,

    /// Output JSON file.
    #[arg(long, default_value = "chunking_ablation.json")]
    output: PathBuf,
}

#[derive(Args)]
struct AblateRrfKArgs {
    /// Workspace directory (must have been prepared with `ingest` first).
    #[arg(long)]
    workspace_dir: PathBuf,

    /// JSON file with QA pairs.
    #[arg(long)]
    qa_file: PathBuf,

    /// Path to the GGUF embedding model.
    #[arg(long)]
    embed_model: PathBuf,

    /// Path to the llama-server binary [auto-detected if not provided].
    #[arg(long)]
    llama_server: Option<PathBuf>,

    /// Port for the embedding server.
    #[arg(long, default_value_t = 12345)]
    embed_port: u16,

    /// Comma-separated RRF k values to test.
    #[arg(long, default_value = "10,30,60,100,200")]
    rrf_k_values: String,

    /// Output JSON file.
    #[arg(long, default_value = "rrf_k_ablation.json")]
    output: PathBuf,
}

#[derive(Args)]
struct AblateQuantArgs {
    /// Base workspace directory.
    #[arg(long)]
    workspace_dir: PathBuf,

    /// Directory of .txt documents to ingest.
    #[arg(long)]
    corpus_dir: PathBuf,

    /// JSON file with QA pairs.
    #[arg(long)]
    qa_file: PathBuf,

    /// Comma-separated paths to GGUF embedding models to compare.
    #[arg(long)]
    embed_models: String,

    /// Path to the llama-server binary [auto-detected if not provided].
    #[arg(long)]
    llama_server: Option<PathBuf>,

    /// Port for the embedding server.
    #[arg(long, default_value_t = 12345)]
    embed_port: u16,

    /// Output JSON file.
    #[arg(long, default_value = "quant_ablation.json")]
    output: PathBuf,
}

// ── QA / result data types ────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
struct QAPair {
    question: String,
    /// Optional content-word filter for relevance scoring.  Missing or empty → doc-title
    /// match alone determines relevance (compatible with bare SQuAD QA files).
    #[serde(default)]
    relevant_keywords: Vec<String>,
    /// If set, only chunks from a document whose title contains this string are
    /// treated as relevant (case-insensitive substring match).
    #[serde(default)]
    doc_title: Option<String>,
    /// Gold answers for end-to-end evaluation (SQuAD format: all acceptable answers).
    #[serde(default)]
    answers: Option<Vec<String>>,
}

#[derive(Serialize)]
struct RecallResult {
    config: String,
    /// Recall@k values keyed by "recall@<k>".
    recall: HashMap<String, f64>,
    /// Mean Reciprocal Rank across all QA pairs.
    mrr: f64,
    /// Mean query latency across all QA pairs (ms).
    avg_latency_ms: f64,
    /// Per-question details (useful for error analysis).
    per_question: Vec<PerQuestionResult>,
}

#[derive(Clone, Serialize)]
struct PerQuestionResult {
    question: String,
    hit_at_k: HashMap<String, bool>,
    /// 1-based rank of the first relevant result, None if not found in top-K.
    first_relevant_rank: Option<usize>,
    latency_ms: f64,
}

#[derive(Serialize)]
struct LatencyBreakdown {
    embed_ms: f64,
    bm25_ms: f64,
    vector_ms: f64,
    rrf_ms: f64,
    expand_ms: f64,
    total_ms: f64,
}

#[derive(Serialize)]
struct IndexStats {
    vector_count: usize,
    memory_mb: f64,
    /// True when IVF partitioning is active (>= 128 vectors).
    ivf_active: bool,
    /// Raw f32 memory estimate for comparison (unquantized).
    raw_f32_estimate_mb: f64,
    /// Compression ratio (raw / quantized).
    compression_ratio: f64,
}

#[derive(Serialize, Deserialize)]
struct IngestTiming {
    doc_title: String,
    size_bytes: u64,
    char_count: usize,
    chunk_count: usize,
    embed_ms: u64,
    total_ms: u64,
}

#[derive(Serialize)]
struct RaptorResult {
    config: String,
    threshold_label: String,
    recall: HashMap<String, f64>,
    avg_latency_ms: f64,
    nodes_evaluated: usize,
    nodes_expanded: usize,
}

#[derive(Clone, Serialize)]
struct E2EPerQuestion {
    question: String,
    gold_answers: Vec<String>,
    predicted: String,
    exact_match: bool,
    f1: f64,
    latency_ms: f64,
}

#[derive(Clone, Serialize)]
struct E2EResult {
    /// Fraction of predictions that exactly match a gold answer (normalized).
    exact_match: f64,
    /// Mean token-level F1 over all evaluated questions.
    f1: f64,
    avg_latency_ms: f64,
    sample_count: usize,
    per_question: Vec<E2EPerQuestion>,
}

#[derive(Serialize, Clone)]
struct ScalePoint {
    doc_count: usize,
    vector_count: usize,
    recall_5_hybrid: f64,
    recall_5_bm25: f64,
    recall_5_vector: f64,
    avg_latency_ms_hybrid: f64,
    memory_mb: f64,
}

#[derive(Serialize)]
struct ScaleResults {
    timestamp: String,
    embed_model: String,
    points: Vec<ScalePoint>,
}

#[derive(Serialize)]
struct BenchResults {
    timestamp: String,
    workspace_dir: String,
    document_count: i64,
    chunk_count: usize,
    recall: Vec<RecallResult>,
    /// Latency breakdown measured on the hybrid+expand configuration.
    latency_hybrid_expand: LatencyBreakdown,
    index_stats: IndexStats,
    ingest_timing: Vec<IngestTiming>,
    raptor: Option<Vec<RaptorResult>>,
    e2e: Option<E2EResult>,
    /// No-RAG baseline: LLM answers from parametric knowledge only (no retrieval).
    no_rag_baseline: Option<E2EResult>,
    /// E2E result with 95% bootstrap confidence intervals and latency percentiles.
    e2e_ci: Option<E2EWithCI>,
}

// ── Extended result types for new benchmarks ─────────────────────────────────

/// E2E result with 95% bootstrap CIs and latency percentiles.
#[derive(Debug, Serialize)]
struct E2EWithCI {
    exact_match: f64,
    em_ci_low: f64,
    em_ci_high: f64,
    f1: f64,
    f1_ci_low: f64,
    f1_ci_high: f64,
    avg_latency_ms: f64,
    p50_latency_ms: f64,
    p95_latency_ms: f64,
    p99_latency_ms: f64,
    n: usize,
    bootstrap_samples: usize,
}

/// BEIR retrieval metrics for one configuration.
#[derive(Debug, Serialize)]
struct BeirMetrics {
    config: String,
    ndcg_at_10: f64,
    map: f64,
    recall_at_100: f64,
    mrr: f64,
    avg_latency_ms: f64,
}

/// BEIR benchmark output file.
#[derive(Serialize)]
struct BeirBenchOutput {
    timestamp: String,
    dataset_dir: String,
    query_count: usize,
    doc_count: usize,
    results: Vec<BeirMetrics>,
}

/// One point in the chunk-size × overlap ablation grid.
#[derive(Debug, Serialize)]
struct ChunkAblationPoint {
    chunk_size: usize,
    overlap: usize,
    n_chunks_total: usize,
    recall_5: f64,
    recall_10: f64,
    mrr: f64,
    avg_query_ms: f64,
    ingest_total_ms: u64,
}

/// One RRF k-constant sensitivity measurement.
#[derive(Debug, Serialize)]
struct RrfKPoint {
    rrf_k: f64,
    recall_5: f64,
    recall_10: f64,
    mrr: f64,
    avg_latency_ms: f64,
}

/// One quantization/model ablation measurement.
#[derive(Debug, Serialize)]
struct QuantAblationPoint {
    model_name: String,
    recall_5: f64,
    recall_10: f64,
    mrr: f64,
    avg_embed_ms: f64,
    avg_query_ms: f64,
}

/// BEIR query (queries.jsonl format).
#[derive(Debug, Deserialize)]
struct BeirQuery {
    #[serde(rename = "_id")]
    id: String,
    text: String,
}

/// BEIR corpus document (corpus.jsonl format).
#[derive(Debug, Deserialize)]
struct BeirDoc {
    #[serde(rename = "_id")]
    id: String,
    #[serde(default)]
    title: String,
    text: String,
}

// ── Embedding server management ───────────────────────────────────────────────

/// Return the last `max_lines` lines of a file as a string, for error diagnostics.
fn read_file_tail(path: &std::path::Path, max_lines: usize) -> String {
    std::fs::read_to_string(path)
        .map(|s| {
            let lines: Vec<&str> = s.lines().collect();
            let start = lines.len().saturating_sub(max_lines);
            lines[start..].join("\n")
        })
        .unwrap_or_else(|_| "(no output captured)".to_string())
}

struct EmbedServer {
    process: Mutex<Child>,
    port: u16,
    client: Client,
}

impl EmbedServer {
    async fn start(server_bin: &Path, model: &Path, port: u16) -> Result<Self> {
        println!(
            "[bench] Starting embedding server (port {}, model {}) …",
            port,
            model.display()
        );

        // Prepend the server binary's directory to LD_LIBRARY_PATH so the
        // dynamically-linked llama-server can find libggml*.so siblings.
        let lib_dir = server_bin
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_default();
        let existing_ld = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
        let new_ld = if existing_ld.is_empty() {
            lib_dir.to_string_lossy().into_owned()
        } else {
            format!("{}:{}", lib_dir.display(), existing_ld)
        };

        // Capture stderr to a temp file so failure messages include the actual error.
        let stderr_log = std::env::temp_dir()
            .join(format!("rag_bench_embed_{}.log", port));
        let stderr_file = std::fs::File::create(&stderr_log)
            .unwrap_or_else(|_| std::fs::OpenOptions::new()
                .write(true).open("/dev/null").unwrap());

        let mut process = Command::new(server_bin)
            .args([
                "--model",
                model.to_str().context("Invalid model path encoding")?,
                "--port",
                &port.to_string(),
                "--embeddings",
                // 2048 tokens covers the largest ablation chunk size (2048 chars ≈ 512 tokens)
                // plus metadata overhead; 512 was too small and caused ctx-exceeded errors.
                "--ctx-size",
                "2048",
                "--batch-size",
                "2048",
                // ubatch-size (physical micro-batch) must be >= the longest single input in
                // tokens.  SciFact / BEIR documents can exceed 512 tokens per chunk, so match
                // this to ctx-size to avoid "input too large" errors.
                "--ubatch-size",
                "2048",
                // Offload all layers to GPU if available (falls back to CPU silently).
                // --log-disable is intentionally omitted so GPU detection messages
                // ("found X CUDA devices", "offloaded N layers") are captured in the
                // stderr log and surfaced as a startup diagnostic below.
                "--n-gpu-layers",
                "99",
                "--no-warmup",
            ])
            .env("LD_LIBRARY_PATH", &new_ld)
            .stdout(Stdio::null())
            .stderr(Stdio::from(stderr_file))
            .spawn()
            .with_context(|| {
                format!(
                    "Failed to spawn llama-server at '{}'",
                    server_bin.display()
                )
            })?;

        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()?;

        let health_url = format!("http://127.0.0.1:{}/health", port);
        let deadline = Instant::now() + Duration::from_secs(90);
        let start_t = Instant::now();
        loop {
            // Detect instant crash (missing library, unsupported flag, OOM, etc.)
            if let Ok(Some(status)) = process.try_wait() {
                let tail = read_file_tail(&stderr_log, 40);
                bail!(
                    "Embedding server (port {}) exited prematurely after {:.1}s ({}).\n\
                     llama-server output:\n{}",
                    port, start_t.elapsed().as_secs_f64(), status, tail
                );
            }
            if Instant::now() >= deadline {
                let tail = read_file_tail(&stderr_log, 40);
                bail!(
                    "Embedding server did not become healthy within 90s (port {}).\n\
                     Check that the model file is valid and the port is free.\n\
                     llama-server output:\n{}",
                    port, tail
                );
            }
            match client.get(&health_url).send().await {
                Ok(r) if r.status().is_success() => break,
                _ => tokio::time::sleep(Duration::from_millis(500)).await,
            }
        }

        println!(
            "[bench] Embedding server ready ({:.1}s warm-up)",
            start_t.elapsed().as_secs_f64()
        );

        // Scan the startup log for GPU/CUDA lines and print diagnostics.
        let log_content = std::fs::read_to_string(&stderr_log).unwrap_or_default();
        let log_lower = log_content.to_ascii_lowercase();

        let gpu_lines: Vec<&str> = log_content
            .lines()
            .filter(|l| {
                let lo = l.to_ascii_lowercase();
                lo.contains("cuda") || lo.contains("gpu") || lo.contains("metal")
                    || lo.contains("offload") || lo.contains("layers to")
                    || lo.contains("no devices") || lo.contains("vulkan")
            })
            .take(20)
            .collect();

        if gpu_lines.is_empty() {
            println!("[bench] GPU: no CUDA/GPU messages in server log — likely running on CPU.");
            println!("[bench]      Verify: ldd {} | grep -i cuda", server_bin.display());
        } else {
            for line in &gpu_lines {
                println!("[bench] GPU: {}", line.trim());
            }
            // Warn when CUDA was detected but there is no evidence of GPU use.
            // Newer llama.cpp uses "fitting params to device memory" instead of
            // "offloaded N layers" — accept either as a GPU-active signal.
            let has_offload = log_lower.contains("offload")
                || log_lower.contains("fitting params to device");
            if !has_offload {
                println!(
                    "[bench] GPU: WARNING — CUDA detected but no GPU-offload signal found."
                );
                println!(
                    "[bench]      Model may be running on CPU despite --n-gpu-layers 99."
                );
                println!(
                    "[bench]      Full server log: {}",
                    stderr_log.display()
                );
                println!(
                    "[bench]      Monitor live GPU use: nvidia-smi dmon -s u -d 1"
                );
            }
        }

        Ok(Self {
            process: Mutex::new(process),
            port,
            client,
        })
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        // BGE models are BERT-based with a hard 512-token architectural limit.
        // Default chunks are 1536 chars; Wikipedia sport/number-heavy text tokenises
        // at ~2.5 chars/token, yielding 600+ tokens. Cap at 1200 chars (~480 tokens
        // at 2.5 chars/token) to stay safely under the 512-token model limit.
        const BGE_MAX_CHARS: usize = 1200;
        let texts: Vec<String> = texts
            .into_iter()
            .map(|t| {
                if t.len() <= BGE_MAX_CHARS {
                    t
                } else {
                    let boundary = t.floor_char_boundary(BGE_MAX_CHARS);
                    t[..boundary].to_string()
                }
            })
            .collect();

        // Use the OpenAI-compatible /v1/embeddings endpoint (same as the main app backend).
        let url = format!("http://127.0.0.1:{}/v1/embeddings", self.port);
        let body = serde_json::json!({ "input": texts });

        // Retry up to 3 times with exponential backoff to handle brief server
        // unreadiness after the health check passes.
        let mut last_err = String::new();
        for attempt in 1u32..=3 {
            if attempt > 1 {
                let wait_ms = 500u64 * (1 << (attempt - 2)); // 500ms, 1000ms
                tokio::time::sleep(Duration::from_millis(wait_ms)).await;
            }

            let resp: serde_json::Value = match self
                .client
                .post(&url)
                .json(&body)
                .send()
                .await
            {
                Err(e) => { last_err = format!("HTTP request failed: {e}"); continue; }
                Ok(r) => match r.json().await {
                    Err(e) => { last_err = format!("JSON parse failed: {e}"); continue; }
                    Ok(v) => v,
                },
            };

            // If the server returned an error object, surface it clearly.
            if let Some(err_obj) = resp.get("error") {
                let msg = err_obj
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or_else(|| err_obj.as_str().unwrap_or("unknown server error"));
                last_err = format!("Server error: {msg}");
                continue;
            }

            // OpenAI-compatible response: { "data": [ { "embedding": [f32...], "index": 0 }, ... ] }
            let items = match resp["data"].as_array() {
                Some(a) => a,
                None => {
                    last_err = format!(
                        "Embedding response missing 'data' array; raw response: {}",
                        &resp.to_string()[..resp.to_string().len().min(300)]
                    );
                    continue;
                }
            };

            let embeddings = items
                .iter()
                .map(|item| {
                    item["embedding"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                            .collect::<Vec<f32>>()
                    })
                    .unwrap_or_default()
                })
                .collect();

            return Ok(embeddings);
        }

        bail!("Embedding failed after 3 attempts: {}", last_err)
    }

    fn stop(&self) {
        if let Ok(mut child) = self.process.lock() {
            let _ = child.kill();
        }
    }
}

impl Drop for EmbedServer {
    fn drop(&mut self) {
        self.stop();
    }
}

// ── LLM generation server (for E2E answer quality eval) ─────────────────────────

struct ChatServer {
    process: Mutex<Child>,
    port: u16,
    client: Client,
}

impl ChatServer {
    /// `parallel` is the number of concurrent KV-cache slots.  Use 1 for sequential
    /// E2E eval (avoids dividing ctx_size across unused slots).  Use ≥4 for RAPTOR
    /// parallel cluster summarisation during ingest.
    async fn start(server_bin: &Path, model: &Path, port: u16, parallel: usize) -> Result<Self> {
        let lib_dir = server_bin.parent().map(|p| p.to_path_buf()).unwrap_or_default();
        let existing_ld = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
        let new_ld = if existing_ld.is_empty() {
            lib_dir.to_string_lossy().into_owned()
        } else {
            format!("{}:{}", lib_dir.display(), existing_ld)
        };
        println!(
            "[bench] Starting LLM server (port {}, model {}) …",
            port,
            model.display()
        );
        let process = Command::new(server_bin)
            .args([
                "--model",
                model.to_str().context("Invalid LLM model path encoding")?,
                "--port",
                &port.to_string(),
                "--ctx-size",
                "4096",
                "--n-gpu-layers",
                "99",
                "--parallel",
                &parallel.to_string(),
                "--no-warmup",
                "--log-disable",
            ])
            .env("LD_LIBRARY_PATH", &new_ld)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| {
                format!("Failed to spawn llama-server at '{}'", server_bin.display())
            })?;

        let client = Client::builder().timeout(Duration::from_secs(120)).build()?;
        let health_url = format!("http://127.0.0.1:{}/health", port);
        let deadline = Instant::now() + Duration::from_secs(90);
        loop {
            if Instant::now() >= deadline {
                bail!("LLM server did not become healthy within 90 s on port {}.", port);
            }
            match client.get(&health_url).send().await {
                Ok(r) if r.status().is_success() => break,
                _ => tokio::time::sleep(Duration::from_millis(500)).await,
            }
        }
        println!("[bench] LLM server ready");
        Ok(Self { process: Mutex::new(process), port, client })
    }

    async fn chat_complete(&self, system: &str, user: &str) -> Result<String> {
        let url = format!("http://127.0.0.1:{}/v1/chat/completions", self.port);
        let body = serde_json::json!({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user}
            ],
            "temperature": 0.0,
            "max_tokens": 64,
            // Disable Qwen3 thinking mode at the chat-template level.
            // Do NOT set reasoning_budget:0 — it causes the model to emit an
            // unclosed <think> tag on long RAG contexts, which strip_think_tags
            // then discards entirely, giving EM=0 F1=0 for all RAG queries.
            "chat_template_kwargs": {"enable_thinking": false}
        });
        let resp: serde_json::Value = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Chat HTTP request failed")?
            .json()
            .await
            .context("Failed to parse chat completion response as JSON")?;
        let raw = resp["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_else(|| {
                // Surface unexpected response shapes for debugging
                static DUMPED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !DUMPED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("[chat-debug] unexpected response JSON: {}", resp);
                }
                ""
            })
            .trim()
            .to_string();
        Ok(strip_think_tags(&raw))
    }

    fn stop(&self) {
        if let Ok(mut child) = self.process.lock() {
            let _ = child.kill();
        }
    }
}

impl Drop for ChatServer {
    fn drop(&mut self) {
        self.stop();
    }
}

// ── llama-server binary auto-detection ───────────────────────────────────────

fn find_llama_server() -> Option<PathBuf> {
    // Check $PATH first
    if let Ok(out) = Command::new("which").arg("llama-server").output() {
        if out.status.success() {
            let p = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim().to_string());
            if p.exists() {
                return Some(p);
            }
        }
    }

    // Try candidate paths relative to the current working directory.
    // When `cargo run` is called from genhat-desktop/, the binary lands in
    // src-tauri/target/{profile}/rag-bench; from there `../../bin/llama-lin/`
    // walks back to src-tauri/bin/.
    let cwd = std::env::current_dir().unwrap_or_default();
    let candidates = [
        "genhat-desktop/src-tauri/bin/llama-lin/llama-server",
        "src-tauri/bin/llama-lin/llama-server",
        "bin/llama-lin/llama-server",
        "../../bin/llama-lin/llama-server",
        "../../../bin/llama-lin/llama-server",
        "../../../../bin/llama-lin/llama-server",
    ];

    for c in &candidates {
        let p = cwd.join(c);
        if p.exists() {
            return Some(p);
        }
    }

    // Try relative to the compiled executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for c in &candidates {
                let p = dir.join(c);
                if p.exists() {
                    return Some(p);
                }
            }
        }
    }

    None
}

fn resolve_server_bin(user_provided: &Option<PathBuf>) -> Result<PathBuf> {
    if let Some(p) = user_provided {
        if p.exists() {
            return Ok(p.clone());
        }
        bail!("llama-server not found at: {}", p.display());
    }
    find_llama_server().ok_or_else(|| {
        anyhow::anyhow!(
            "llama-server binary not found.\n\
             Pass --llama-server <path> or ensure the NELA repo's \
             bin/llama-lin/llama-server is accessible from the working directory."
        )
    })
}

// ── Oracle ────────────────────────────────────────────────────────────────────

/// A chunk is "relevant" to a QA pair when:
/// 1. Its text contains at least one of the `relevant_keywords` (case-insensitive).
/// 2. If `doc_title` is set, the chunk's document title contains it as a substring.
fn is_relevant(chunk_text: &str, chunk_doc_title: &str, qa: &QAPair) -> bool {
    let lower_text = chunk_text.to_lowercase();
    // When no keywords are provided, skip the keyword gate entirely.
    let keyword_hit = if qa.relevant_keywords.is_empty() {
        true
    } else {
        qa.relevant_keywords
            .iter()
            .any(|kw| lower_text.contains(&kw.to_lowercase()))
    };
    let doc_hit = qa
        .doc_title
        .as_ref()
        .map(|dt| {
            chunk_doc_title
                .to_lowercase()
                .contains(&dt.to_lowercase())
        })
        .unwrap_or(true);
    keyword_hit && doc_hit
}

// ── Ingestion ─────────────────────────────────────────────────────────────────

async fn ingest_corpus(
    args: &IngestArgs,
    server: &EmbedServer,
    db: &RagDb,
    bm25: &BM25Index,
    vec_index: &VectorIndex,
) -> Result<Vec<IngestTiming>> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&args.corpus_dir)
        .with_context(|| format!("Cannot read corpus dir: {}", args.corpus_dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("txt"))
        .collect();
    entries.sort();

    if entries.is_empty() {
        bail!(
            "No .txt files found in corpus dir: {}",
            args.corpus_dir.display()
        );
    }

    println!("[bench] Ingesting {} .txt files …", entries.len());
    let mut timings = Vec::new();

    for path in &entries {
        let t0 = Instant::now();

        let path_str = path.to_string_lossy().to_string();
        let title = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Skip already-ingested documents (idempotent)
        if db.document_exists(&path_str).unwrap_or(false) {
            println!("[bench]   skip (already ingested): {}", title);
            continue;
        }

        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Cannot read file: {}", path.display()))?;
        let size_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let char_count = text.len();

        let chunks = chunk_text_default(&text);
        let chunk_count = chunks.len();
        if chunk_count == 0 {
            println!("[bench]   WARNING: no chunks from '{}', skipping", title);
            continue;
        }

        // Insert document record
        let doc_id = db
            .insert_document(&path_str, &title, "txt", chunk_count as i64)
            .map_err(|e| anyhow::anyhow!("insert_document failed: {}", e))?;

        // Insert chunk records
        let chunk_data: Vec<(usize, String, String)> = chunks
            .iter()
            .map(|c| (c.index, c.text.clone(), c.metadata.clone()))
            .collect();
        let chunk_ids = db
            .insert_chunks(doc_id, &chunk_data)
            .map_err(|e| anyhow::anyhow!("insert_chunks failed: {}", e))?;

        // Index in BM25
        let bm25_batch: Vec<(i64, String, String)> = chunk_ids
            .iter()
            .zip(chunks.iter())
            .map(|(&id, c)| (id, c.text.clone(), title.clone()))
            .collect();
        bm25.add_chunks_batch(&bm25_batch)
            .map_err(|e| anyhow::anyhow!("BM25 add_chunks_batch failed: {}", e))?;

        // Embed chunks via embedding server
        let t_embed = Instant::now();
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let embeddings = server
            .embed(texts)
            .await
            .with_context(|| format!("Embedding failed for '{}'", title))?;
        let embed_ms = t_embed.elapsed().as_millis() as u64;

        // Store embeddings in DB + vector index
        for (i, emb) in embeddings.iter().enumerate() {
            if i < chunk_ids.len() && !emb.is_empty() {
                let _ = db.set_chunk_embedding(chunk_ids[i], emb, None);
                vec_index.insert(chunk_ids[i], emb.clone());
            }
        }

        let total_ms = t0.elapsed().as_millis() as u64;
        println!(
            "[bench]   '{}' → {} chunks | embed {}ms | total {}ms",
            title, chunk_count, embed_ms, total_ms
        );

        timings.push(IngestTiming {
            doc_title: title,
            size_bytes,
            char_count,
            chunk_count,
            embed_ms,
            total_ms,
        });
    }

    vec_index.rebuild_if_needed();

    println!(
        "[bench] Ingestion done: {} docs ingested, {} vectors in index.",
        timings.len(),
        vec_index.len()
    );
    Ok(timings)
}

// ── Recall + latency benchmark ────────────────────────────────────────────────

async fn run_recall_bench(
    qa_pairs: &[QAPair],
    top_ks: &[usize],
    db: &RagDb,
    bm25: &BM25Index,
    vec_index: &VectorIndex,
    server: &EmbedServer,
) -> Result<(Vec<RecallResult>, LatencyBreakdown)> {
    // Pre-build doc_id → title map for fast oracle evaluation
    let docs = db.list_documents().map_err(|e| anyhow::anyhow!("list_documents failed: {}", e))?;
    let doc_title_map: HashMap<i64, String> =
        docs.iter().map(|d| (d.id, d.title.clone())).collect();

    let max_k = *top_ks.iter().max().unwrap_or(&10);

    // ── Retrieval configurations ──────────────────────────────────────────
    struct Config {
        name: &'static str,
        use_bm25: bool,
        use_vector: bool,
        use_expand: bool,
    }
    let configs = [
        Config {
            name: "bm25_only",
            use_bm25: true,
            use_vector: false,
            use_expand: false,
        },
        Config {
            name: "vector_only",
            use_bm25: false,
            use_vector: true,
            use_expand: false,
        },
        Config {
            name: "hybrid",
            use_bm25: true,
            use_vector: true,
            use_expand: false,
        },
        Config {
            name: "hybrid_expand",
            use_bm25: true,
            use_vector: true,
            use_expand: true,
        },
    ];

    // Per-stage latency accumulators (measured on hybrid_expand)
    let mut lat_embed = 0.0_f64;
    let mut lat_bm25 = 0.0_f64;
    let mut lat_vec = 0.0_f64;
    let mut lat_rrf = 0.0_f64;
    let mut lat_expand = 0.0_f64;
    let mut lat_n = 0u64;

    let mut recall_results = Vec::new();

    for cfg in &configs {
        let mut per_question = Vec::new();
        let mut total_latency = 0.0;

        for qa in qa_pairs {
            let t_total = Instant::now();

            // ── Stage 1: embed query ──────────────────────────────────────
            let t = Instant::now();
            let emb_resp = server.embed(vec![qa.question.clone()]).await?;
            let query_emb = emb_resp.into_iter().next().unwrap_or_default();
            let d_embed = t.elapsed().as_secs_f64() * 1000.0;

            // ── Stage 2: BM25 ─────────────────────────────────────────────
            let t = Instant::now();
            let bm25_res: Vec<(i64, f32)> = if cfg.use_bm25 {
                bm25.search(&qa.question, max_k).unwrap_or_default()
            } else {
                vec![]
            };
            let d_bm25 = t.elapsed().as_secs_f64() * 1000.0;

            // ── Stage 3: vector search ────────────────────────────────────
            let t = Instant::now();
            let vec_res: Vec<(i64, f32)> = if cfg.use_vector && !query_emb.is_empty() {
                vec_index.search(&query_emb, max_k)
            } else {
                vec![]
            };
            let d_vec = t.elapsed().as_secs_f64() * 1000.0;

            // ── Stage 4: RRF fusion ───────────────────────────────────────
            let t = Instant::now();
            let fused_ids: Vec<i64> = match (cfg.use_bm25, cfg.use_vector) {
                (true, true) => {
                    let fused = rrf_fuse(&[bm25_res, vec_res]);
                    fused.iter().take(max_k).map(|r| r.chunk_id).collect()
                }
                (true, false) => bm25_res.iter().take(max_k).map(|(id, _)| *id).collect(),
                (false, true) => vec_res.iter().take(max_k).map(|(id, _)| *id).collect(),
                (false, false) => vec![],
            };
            let d_rrf = t.elapsed().as_secs_f64() * 1000.0;

            // ── Stage 5: context expansion ────────────────────────────────
            let t = Instant::now();
            let final_ids: Vec<i64> = if cfg.use_expand && !fused_ids.is_empty() {
                let chunks = db.get_chunks_by_ids(&fused_ids).unwrap_or_default();
                let refs: Vec<(i64, i32)> =
                    chunks.iter().map(|c| (c.doc_id, c.chunk_index)).collect();
                let neighbors = db.get_adjacent_chunks(&refs).unwrap_or_default();
                let mut ids = fused_ids.clone();
                for n in &neighbors {
                    if !ids.contains(&n.id) {
                        ids.push(n.id);
                    }
                }
                ids
            } else {
                fused_ids.clone()
            };
            let d_expand = t.elapsed().as_secs_f64() * 1000.0;

            // Accumulate latency for the hybrid+expand config
            if cfg.name == "hybrid_expand" {
                lat_embed += d_embed;
                lat_bm25 += d_bm25;
                lat_vec += d_vec;
                lat_rrf += d_rrf;
                lat_expand += d_expand;
                lat_n += 1;
            }

            let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
            total_latency += total_ms;

            // ── Oracle evaluation ─────────────────────────────────────────
            let mut hit_at_k: HashMap<String, bool> = HashMap::new();
            let mut first_relevant_rank: Option<usize> = None;

            if !final_ids.is_empty() {
                let fetched = db.get_chunks_by_ids(&final_ids).unwrap_or_default();

                // Find the 1-based rank of the first relevant result (for MRR)
                first_relevant_rank = final_ids.iter().enumerate().find_map(|(i, id)| {
                    fetched.iter().find(|c| c.id == *id).and_then(|c| {
                        let title = doc_title_map
                            .get(&c.doc_id)
                            .map(|s| s.as_str())
                            .unwrap_or("");
                        if is_relevant(&c.text, title, qa) { Some(i + 1) } else { None }
                    })
                });

                for &k in top_ks {
                    let top = &final_ids[..final_ids.len().min(k)];
                    let hit = top.iter().any(|id| {
                        fetched.iter().find(|c| c.id == *id).map_or(false, |c| {
                            let title = doc_title_map
                                .get(&c.doc_id)
                                .map(|s| s.as_str())
                                .unwrap_or("");
                            is_relevant(&c.text, title, qa)
                        })
                    });
                    hit_at_k.insert(format!("recall@{}", k), hit);
                }
            } else {
                for &k in top_ks {
                    hit_at_k.insert(format!("recall@{}", k), false);
                }
            }

            per_question.push(PerQuestionResult {
                question: qa.question.clone(),
                hit_at_k,
                first_relevant_rank,
                latency_ms: total_ms,
            });
        }

        // Aggregate recall across all questions
        let mut recall: HashMap<String, f64> = HashMap::new();
        let n = qa_pairs.len().max(1);
        for &k in top_ks {
            let key = format!("recall@{}", k);
            let hits = per_question
                .iter()
                .filter(|q| *q.hit_at_k.get(&key).unwrap_or(&false))
                .count();
            recall.insert(key, hits as f64 / n as f64);
        }

        // MRR: mean of (1/rank) for questions that had a relevant result, 0 otherwise
        let mrr = per_question
            .iter()
            .map(|q| q.first_relevant_rank.map(|r| 1.0 / r as f64).unwrap_or(0.0))
            .sum::<f64>()
            / n as f64;

        let avg_latency_ms = total_latency / n as f64;

        let recall_str: String = top_ks
            .iter()
            .map(|&k| format!("recall@{}={:.3}", k, recall.get(&format!("recall@{}", k)).copied().unwrap_or(0.0)))
            .collect::<Vec<_>>()
            .join("  ");
        println!(
            "[bench] {:>16}  {}  mrr={:.3}  latency={:.1}ms",
            cfg.name, recall_str, mrr, avg_latency_ms,
        );

        recall_results.push(RecallResult {
            config: cfg.name.to_string(),
            recall,
            mrr,
            avg_latency_ms,
            per_question,
        });
    }

    let n_f = lat_n.max(1) as f64;
    let latency = LatencyBreakdown {
        embed_ms: lat_embed / n_f,
        bm25_ms: lat_bm25 / n_f,
        vector_ms: lat_vec / n_f,
        rrf_ms: lat_rrf / n_f,
        expand_ms: lat_expand / n_f,
        total_ms: (lat_embed + lat_bm25 + lat_vec + lat_rrf + lat_expand) / n_f,
    };

    Ok((recall_results, latency))
}

// ── RAPTOR confidence-gate ablation ──────────────────────────────────────────

async fn run_raptor_bench(
    qa_pairs: &[QAPair],
    top_ks: &[usize],
    db: &RagDb,
    server: &EmbedServer,
) -> Result<Vec<RaptorResult>> {
    let docs = db.list_documents().map_err(|e| anyhow::anyhow!("list_documents failed: {}", e))?;

    // Collect all RAPTOR nodes + embeddings across every document
    let mut all_nodes: Vec<RaptorNode> = Vec::new();
    let mut all_embeddings: Vec<(i64, Vec<f32>)> = Vec::new();

    for doc in &docs {
        if !db.has_raptor_tree(doc.id).unwrap_or(false) {
            continue;
        }
        all_nodes.extend(db.get_raptor_nodes(doc.id).unwrap_or_default());
        all_embeddings.extend(db.get_raptor_embeddings(doc.id).unwrap_or_default());
    }

    if all_nodes.is_empty() {
        println!(
            "[bench] No RAPTOR trees found in workspace — skipping RAPTOR ablation.\n\
             (Hint: ingest documents via NELA first, wait for background enrichment to finish.)"
        );
        return Ok(vec![]);
    }

    println!(
        "[bench] RAPTOR: {} nodes, {} with embeddings",
        all_nodes.len(),
        all_embeddings.len()
    );

    let node_map: HashMap<i64, &RaptorNode> =
        all_nodes.iter().map(|n| (n.id, n)).collect();

    // Three ablation points:
    //   - "gated"      threshold = -1.5 : NELA default — expand only low-confidence summaries
    //   - "trust_all"  threshold = -∞   : never expand (score never < -∞ → trust every summary)
    //   - "expand_all" threshold = +∞   : always expand (score always < +∞ → fall back to raw chunks)
    let threshold_configs: &[(&str, &str, f64)] = &[
        ("raptor_gated", "-1.5 (NELA default)", -1.5),
        ("raptor_trust_all", "-inf (trust all summaries)", f64::NEG_INFINITY),
        ("raptor_expand_all", "+inf (always expand)", f64::INFINITY),
    ];

    let max_k = *top_ks.iter().max().unwrap_or(&10);
    let mut results = Vec::new();

    for &(config_name, threshold_label, threshold) in threshold_configs {
        let mut hits: HashMap<String, usize> =
            top_ks.iter().map(|&k| (format!("recall@{}", k), 0)).collect();
        let mut total_lat = 0.0_f64;
        let mut total_evaluated = 0usize;
        let mut total_expanded = 0usize;

        for qa in qa_pairs {
            let t0 = Instant::now();

            let emb_resp = server.embed(vec![qa.question.clone()]).await?;
            let query_emb = emb_resp.into_iter().next().unwrap_or_default();
            if query_emb.is_empty() {
                continue;
            }

            // Score RAPTOR nodes by dot product
            let mut scored: Vec<(i64, f32)> = all_embeddings
                .iter()
                .map(|(id, emb)| (*id, dot_product(&query_emb, emb)))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(max_k);

            // Apply confidence gate
            let mut final_texts: Vec<String> = Vec::new();
            for (node_id, _score) in &scored {
                if let Some(node) = node_map.get(node_id) {
                    total_evaluated += 1;
                    if node.confidence_score < threshold {
                        // Low-confidence summary: expand to child chunks
                        total_expanded += 1;
                        if let Ok(children) = db.get_chunks_by_ids(&node.child_ids) {
                            for c in children {
                                final_texts.push(c.text);
                            }
                        }
                    } else {
                        // High-confidence: use the summary directly
                        final_texts.push(node.summary_text.clone());
                    }
                }
            }

            let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
            total_lat += latency_ms;

            // Evaluate recall (keyword oracle, no doc_title filter for summaries)
            for &k in top_ks {
                let top = &final_texts[..final_texts.len().min(k)];
                let hit = top.iter().any(|text| {
                    let lower = text.to_lowercase();
                    qa.relevant_keywords
                        .iter()
                        .any(|kw| lower.contains(&kw.to_lowercase()))
                });
                if hit {
                    *hits.get_mut(&format!("recall@{}", k)).unwrap() += 1;
                }
            }
        }

        let n = qa_pairs.len().max(1);
        let recall: HashMap<String, f64> =
            hits.iter().map(|(k, v)| (k.clone(), *v as f64 / n as f64)).collect();

        let recall_str: String = top_ks
            .iter()
            .map(|&k| format!("recall@{}={:.3}", k, recall.get(&format!("recall@{}", k)).copied().unwrap_or(0.0)))
            .collect::<Vec<_>>()
            .join("  ");
        println!(
            "[bench] {:>20}  {}  expanded={}/{}",
            config_name, recall_str, total_expanded, total_evaluated,
        );

        results.push(RaptorResult {
            config: config_name.to_string(),
            threshold_label: threshold_label.to_string(),
            recall,
            avg_latency_ms: total_lat / n as f64,
            nodes_evaluated: total_evaluated,
            nodes_expanded: total_expanded,
        });
    }

    Ok(results)
}

// ── SQuAD-style answer scoring ────────────────────────────────────────────────

/// Strip `<think>...</think>` blocks from model output (Qwen3/DeepSeek thinking mode).
/// Used as a defensive post-processing step even when thinking is disabled via API params.
fn strip_think_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(start) = rest.find("<think>") {
        out.push_str(&rest[..start]);
        if let Some(end) = rest[start..].find("</think>") {
            // Normal case: closed tag — skip the entire <think>…</think> block.
            rest = &rest[start + end + "</think>".len()..];
        } else {
            // Unclosed tag (e.g. truncated by reasoning_budget) — skip the
            // <think> opener and keep whatever came after it as the answer.
            rest = &rest[start + "<think>".len()..];
            break;
        }
    }
    out.push_str(rest);
    out.trim().to_string()
}

fn normalize_answer(s: &str) -> String {
    let lower = s.to_lowercase();
    let no_punct: String = lower
        .chars()
        .map(|c| if c.is_ascii_punctuation() { ' ' } else { c })
        .collect();
    no_punct
        .split_whitespace()
        .filter(|w| !matches!(*w, "a" | "an" | "the"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn exact_match_score(predicted: &str, gold_answers: &[String]) -> bool {
    if gold_answers.is_empty() {
        return false;
    }
    let norm_pred = normalize_answer(predicted);
    gold_answers.iter().any(|g| normalize_answer(g) == norm_pred)
}

fn token_f1_score(predicted: &str, gold: &str) -> f64 {
    let pred_tokens: Vec<String> = normalize_answer(predicted)
        .split_whitespace()
        .map(str::to_string)
        .collect();
    let gold_tokens: Vec<String> = normalize_answer(gold)
        .split_whitespace()
        .map(str::to_string)
        .collect();
    if pred_tokens.is_empty() && gold_tokens.is_empty() {
        return 1.0;
    }
    if pred_tokens.is_empty() || gold_tokens.is_empty() {
        return 0.0;
    }
    let mut pred_counts: HashMap<&str, usize> = HashMap::new();
    for t in &pred_tokens {
        *pred_counts.entry(t.as_str()).or_insert(0) += 1;
    }
    let mut gold_counts: HashMap<&str, usize> = HashMap::new();
    for t in &gold_tokens {
        *gold_counts.entry(t.as_str()).or_insert(0) += 1;
    }
    let overlap: usize = pred_counts
        .iter()
        .map(|(t, &pc)| pc.min(*gold_counts.get(t).unwrap_or(&0)))
        .sum();
    if overlap == 0 {
        return 0.0;
    }
    let precision = overlap as f64 / pred_tokens.len() as f64;
    let recall = overlap as f64 / gold_tokens.len() as f64;
    2.0 * precision * recall / (precision + recall)
}

fn best_f1_over_golds(predicted: &str, gold_answers: &[String]) -> f64 {
    if gold_answers.is_empty() {
        return 0.0;
    }
    gold_answers
        .iter()
        .map(|g| token_f1_score(predicted, g))
        .fold(0.0_f64, f64::max)
}

// ── E2E answer quality benchmark ──────────────────────────────────────────────

async fn run_e2e_bench(
    qa_pairs: &[QAPair],
    sample_count: usize,
    db: &RagDb,
    bm25: &BM25Index,
    vec_index: &VectorIndex,
    embed_server: &EmbedServer,
    chat_server: &ChatServer,
) -> Result<E2EResult> {
    let qa_with_answers: Vec<&QAPair> = qa_pairs
        .iter()
        .filter(|q| q.answers.as_ref().map(|a| !a.is_empty()).unwrap_or(false))
        .take(sample_count)
        .collect();

    if qa_with_answers.is_empty() {
        bail!(
            "No QA pairs with an 'answers' field found.\n\
             Regenerate qa_pairs.json with prepare_squad.py (which emits the 'answers' field)."
        );
    }

    println!(
        "[bench] E2E eval: {} QA pairs with gold answers",
        qa_with_answers.len()
    );

    let system_prompt = "You are a precise question-answering assistant. \
        Answer the question using ONLY the provided context. \
        Give a short, direct answer (1–5 words). \
        If the answer is not present in the context, respond with exactly: unanswerable";

    let max_k = 5usize;
    let mut per_question: Vec<E2EPerQuestion> = Vec::new();
    let mut total_lat = 0.0_f64;

    for qa in &qa_with_answers {
        let t0 = Instant::now();

        // Hybrid retrieval: embed + BM25 + RRF
        let emb_resp = embed_server.embed(vec![qa.question.clone()]).await?;
        let query_emb = emb_resp.into_iter().next().unwrap_or_default();
        let bm25_res = bm25.search(&qa.question, max_k).unwrap_or_default();
        let vec_res = if !query_emb.is_empty() {
            vec_index.search(&query_emb, max_k)
        } else {
            vec![]
        };
        let fused = rrf_fuse(&[bm25_res, vec_res]);
        let ids: Vec<i64> = fused.iter().take(max_k).map(|r| r.chunk_id).collect();
        let chunks = db.get_chunks_by_ids(&ids).unwrap_or_default();
        let context = chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n---\n\n");

        let user_msg = format!("Context:\n{}\n\nQuestion: {}\nAnswer:", context, qa.question);
        let predicted = match chat_server.chat_complete(system_prompt, &user_msg).await {
            Ok(s) => s,
            Err(e) => {
                if per_question.is_empty() {
                    // Surface the first failure so we can diagnose it
                    eprintln!("[e2e-error] chat_complete failed: {:#}", e);
                }
                String::new()
            }
        };

        let gold_answers = qa.answers.clone().unwrap_or_default();
        let em = exact_match_score(&predicted, &gold_answers);
        let f1 = best_f1_over_golds(&predicted, &gold_answers);
        let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
        total_lat += latency_ms;

        // Debug: print first 5 predictions so we can diagnose model output
        if per_question.len() < 5 {
            eprintln!(
                "[e2e-debug #{:02}] Q: {}\n  gold={:?}\n  pred={:?}  em={}  f1={:.3}  ctx_chars={}",
                per_question.len() + 1,
                &qa.question,
                &gold_answers,
                &predicted,
                em,
                f1,
                context.len(),
            );
        }

        per_question.push(E2EPerQuestion {
            question: qa.question.clone(),
            gold_answers,
            predicted,
            exact_match: em,
            f1,
            latency_ms,
        });
    }

    let n = per_question.len().max(1);
    let agg_em = per_question.iter().filter(|q| q.exact_match).count() as f64 / n as f64;
    let agg_f1 = per_question.iter().map(|q| q.f1).sum::<f64>() / n as f64;
    println!(
        "[bench]   e2e_hybrid  EM={:.3}  F1={:.3}  latency={:.1}ms  n={}",
        agg_em,
        agg_f1,
        total_lat / n as f64,
        n
    );

    Ok(E2EResult {
        exact_match: agg_em,
        f1: agg_f1,
        avg_latency_ms: total_lat / n as f64,
        sample_count: n,
        per_question,
    })
}

// ── No-RAG baseline ───────────────────────────────────────────────────────────

/// Ask the LLM the same questions as E2E eval but with NO retrieved context.
/// Measures what the model knows from parametric (training) knowledge alone.
/// Comparing this to run_e2e_bench reveals the actual recall contribution.
async fn run_norag_eval(
    qa_pairs: &[QAPair],
    sample_count: usize,
    chat_server: &ChatServer,
) -> Result<E2EResult> {
    let qa_with_answers: Vec<&QAPair> = qa_pairs
        .iter()
        .filter(|q| q.answers.as_ref().map(|a| !a.is_empty()).unwrap_or(false))
        .take(sample_count)
        .collect();

    if qa_with_answers.is_empty() {
        bail!(
            "No QA pairs with an 'answers' field found for no-RAG baseline.\n\
             Regenerate qa_pairs.json with prepare_squad.py (which emits the 'answers' field)."
        );
    }

    println!(
        "[bench] No-RAG baseline: {} QA pairs (no context)",
        qa_with_answers.len()
    );

    let system_prompt = "You are a precise question-answering assistant. \
        Answer the question from your own knowledge. \
        Give a short, direct answer (1–5 words). \
        If you do not know, respond with exactly: unanswerable";

    let mut per_question: Vec<E2EPerQuestion> = Vec::new();
    let mut total_lat = 0.0_f64;

    for qa in &qa_with_answers {
        let t0 = Instant::now();

        // No retrieval — question only
        let user_msg = format!("Question: {}\nAnswer:", qa.question);
        let predicted = chat_server
            .chat_complete(system_prompt, &user_msg)
            .await
            .unwrap_or_default();

        let gold_answers = qa.answers.clone().unwrap_or_default();
        let em = exact_match_score(&predicted, &gold_answers);
        let f1 = best_f1_over_golds(&predicted, &gold_answers);
        let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
        total_lat += latency_ms;

        per_question.push(E2EPerQuestion {
            question: qa.question.clone(),
            gold_answers,
            predicted,
            exact_match: em,
            f1,
            latency_ms,
        });
    }

    let n = per_question.len().max(1);
    let agg_em = per_question.iter().filter(|q| q.exact_match).count() as f64 / n as f64;
    let agg_f1 = per_question.iter().map(|q| q.f1).sum::<f64>() / n as f64;
    println!(
        "[bench]   no_rag_baseline  EM={:.3}  F1={:.3}  latency={:.1}ms  n={}",
        agg_em,
        agg_f1,
        total_lat / n as f64,
        n
    );

    Ok(E2EResult {
        exact_match: agg_em,
        f1: agg_f1,
        avg_latency_ms: total_lat / n as f64,
        sample_count: n,
        per_question,
    })
}

// ── Scale degradation command ─────────────────────────────────────────────────

async fn cmd_scale(args: ScaleArgs) -> Result<()> {
    let server_bin = resolve_server_bin(&args.llama_server)?;
    std::fs::create_dir_all(&args.workspace_dir)
        .with_context(|| format!("Cannot create workspace dir: {}", args.workspace_dir.display()))?;

    let sizes: Vec<usize> = args
        .sizes
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if sizes.is_empty() {
        bail!("--sizes must contain at least one integer, e.g. '100,500,1000,2000'");
    }

    // Collect and sort all corpus files
    let mut all_files: Vec<PathBuf> = std::fs::read_dir(&args.corpus_dir)
        .with_context(|| format!("Cannot read corpus dir: {}", args.corpus_dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("txt"))
        .collect();
    all_files.sort();

    let max_size = *sizes.iter().max().unwrap();
    if all_files.len() < max_size {
        println!(
            "[scale] WARNING: corpus has {} docs but max size is {} — capping at {}",
            all_files.len(),
            max_size,
            all_files.len()
        );
    }
    all_files.truncate(max_size);

    let qa_json = std::fs::read_to_string(&args.qa_file)
        .with_context(|| format!("Cannot read QA file: {}", args.qa_file.display()))?;
    let all_qa: Vec<QAPair> = serde_json::from_str(&qa_json).context("Invalid QA file")?;
    if all_qa.is_empty() {
        bail!("QA file is empty");
    }

    let top_ks = vec![5usize, 10];
    let mut server = EmbedServer::start(&server_bin, &args.embed_model, args.embed_port).await?;
    let mut points: Vec<ScalePoint> = Vec::new();

    for &target_size in &sizes {
        let target = target_size.min(all_files.len());
        println!("\n[scale] ── Checkpoint: {} docs ──", target);

        // Each checkpoint gets a completely fresh, isolated workspace so that
        // doc counts are accurate and indexes don't carry state from previous
        // checkpoints.  A shared workspace would skip already-ingested docs
        // (document_exists = true) and always report the maximum corpus size.
        let cp_dir = args.workspace_dir.join(format!("cp_{:06}", target));
        std::fs::create_dir_all(&cp_dir)
            .with_context(|| format!("Cannot create checkpoint dir: {}", cp_dir.display()))?;

        let cp_db_path = cp_dir.join("rag.db");
        let cp_index_dir = cp_dir.join("bm25_index");
        let cp_db =
            RagDb::open(&cp_db_path).map_err(|e| anyhow::anyhow!("RagDb (cp): {}", e))?;
        let cp_bm25 =
            BM25Index::open(&cp_index_dir).map_err(|e| anyhow::anyhow!("BM25 (cp): {}", e))?;
        let cp_vec = VectorIndex::load_from_db(&cp_db)
            .map_err(|e| anyhow::anyhow!("VecIndex (cp): {}", e))?;

        // Ingest exactly 'target' corpus files into the fresh workspace
        for path in all_files.iter().take(target) {
            let path_str = path.to_string_lossy().to_string();
            let title = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("?")
                .to_string();
            let text = match std::fs::read_to_string(path) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("[scale]   WARN: cannot read {}: {}", path.display(), e);
                    continue;
                }
            };
            let chunks = chunk_text_default(&text);
            if chunks.is_empty() {
                continue;
            }
            let doc_id = cp_db
                .insert_document(&path_str, &title, "txt", chunks.len() as i64)
                .map_err(|e| anyhow::anyhow!("insert_document: {}", e))?;
            let chunk_data: Vec<(usize, String, String)> = chunks
                .iter()
                .map(|c| (c.index, c.text.clone(), c.metadata.clone()))
                .collect();
            let chunk_ids = cp_db
                .insert_chunks(doc_id, &chunk_data)
                .map_err(|e| anyhow::anyhow!("insert_chunks: {}", e))?;
            let bm25_batch: Vec<(i64, String, String)> = chunk_ids
                .iter()
                .zip(chunks.iter())
                .map(|(&id, c)| (id, c.text.clone(), title.clone()))
                .collect();
            cp_bm25
                .add_chunks_batch(&bm25_batch)
                .map_err(|e| anyhow::anyhow!("BM25 batch: {}", e))?;
            let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
            let embeddings = server
                .embed(texts)
                .await
                .with_context(|| format!("Embedding failed for '{}'", title))?;
            for (i, emb) in embeddings.iter().enumerate() {
                if i < chunk_ids.len() && !emb.is_empty() {
                    let _ = cp_db.set_chunk_embedding(chunk_ids[i], emb, None);
                    cp_vec.insert(chunk_ids[i], emb.clone());
                }
            }
        }

        cp_vec.rebuild_if_needed();
        let doc_count = cp_db.document_count().unwrap_or(0) as usize;
        let vec_count = cp_vec.len();
        println!("[scale] Ingested: {} docs, {} vectors", doc_count, vec_count);

        // Filter QA pairs to those whose source document is in this checkpoint's
        // index so recall is measured on an answerable subset.
        let ingested_docs = cp_db.list_documents().unwrap_or_default();
        let ingested_titles: std::collections::HashSet<String> = ingested_docs
            .iter()
            .map(|d| d.title.to_lowercase())
            .collect();

        let eligible: Vec<QAPair> = all_qa
            .iter()
            .filter(|qa| {
                qa.doc_title
                    .as_ref()
                    .map(|dt| ingested_titles.contains(&dt.to_lowercase()))
                    .unwrap_or(true)
            })
            .take(args.qa_sample)
            .cloned()
            .collect();

        if eligible.is_empty() {
            println!("[scale]   (no QA pairs match ingested docs at this checkpoint — skipping eval)");
            let quant_mb = cp_vec.estimated_memory_bytes() as f64 / (1024.0 * 1024.0);
            points.push(ScalePoint {
                doc_count,
                vector_count: vec_count,
                recall_5_hybrid: 0.0,
                recall_5_bm25: 0.0,
                recall_5_vector: 0.0,
                avg_latency_ms_hybrid: 0.0,
                memory_mb: quant_mb,
            });
            // Remove the checkpoint dir to save disk space
            let _ = std::fs::remove_dir_all(&cp_dir);
            continue;
        }
        println!("[scale]   Evaluating {} eligible QA pairs", eligible.len());
        let (recall_results, _) =
            run_recall_bench(&eligible, &top_ks, &cp_db, &cp_bm25, &cp_vec, &server).await?;

        let get_r5 = |cfg: &str| {
            recall_results
                .iter()
                .find(|r| r.config == cfg)
                .and_then(|r| r.recall.get("recall@5").copied())
                .unwrap_or(0.0)
        };
        let hybrid_lat = recall_results
            .iter()
            .find(|r| r.config == "hybrid")
            .map(|r| r.avg_latency_ms)
            .unwrap_or(0.0);
        let quant_mb = cp_vec.estimated_memory_bytes() as f64 / (1024.0 * 1024.0);

        points.push(ScalePoint {
            doc_count,
            vector_count: vec_count,
            recall_5_hybrid: get_r5("hybrid"),
            recall_5_bm25: get_r5("bm25_only"),
            recall_5_vector: get_r5("vector_only"),
            avg_latency_ms_hybrid: hybrid_lat,
            memory_mb: quant_mb,
        });

        // Remove the checkpoint dir to save disk space
        let _ = std::fs::remove_dir_all(&cp_dir);
    }

    server.stop();

    let results = ScaleResults {
        timestamp: chrono::Utc::now().to_rfc3339(),
        embed_model: args.embed_model.to_string_lossy().to_string(),
        points: points.clone(),
    };

    let json = serde_json::to_string_pretty(&results)?;
    std::fs::write(&args.output, &json)
        .with_context(|| format!("Cannot write scale results to {}", args.output.display()))?;

    println!("\n── Scale Degradation Results ──────────────────────────────────────────────────");
    println!(
        "{:>8}  {:>8}  {:>12}  {:>12}  {:>14}  {:>10}  {:>9}",
        "Docs", "Vectors", "BM25 R@5", "Vector R@5", "Hybrid R@5", "Lat(ms)", "Mem(MB)"
    );
    println!("{}", "─".repeat(82));
    for p in &results.points {
        println!(
            "{:>8}  {:>8}  {:>12.3}  {:>12.3}  {:>14.3}  {:>10.1}  {:>9.2}",
            p.doc_count,
            p.vector_count,
            p.recall_5_bm25,
            p.recall_5_vector,
            p.recall_5_hybrid,
            p.avg_latency_ms_hybrid,
            p.memory_mb
        );
    }
    println!("\n[bench] Scale results saved → {}", args.output.display());
    Ok(())
}

// ── Summary printer ───────────────────────────────────────────────────────────

fn print_summary(r: &BenchResults) {
    println!("\n╔══ NELA RAG Benchmark Results ══╗");
    println!(
        "║ Corpus : {} docs, {} vectors",
        r.document_count, r.chunk_count
    );
    println!(
        "║ Index  : {:.2} MB quantized  ({:.2} MB raw est., {:.1}× compression, IVF={})",
        r.index_stats.memory_mb,
        r.index_stats.raw_f32_estimate_mb,
        r.index_stats.compression_ratio,
        if r.index_stats.ivf_active { "ON" } else { "OFF (brute-force)" }
    );
    println!("╚══════════════════════════════════╝");

    println!("\n── Recall@k ──────────────────────────────────────────────────");
    // Collect all k values from data and display them all
    let mut all_ks: Vec<usize> = r.recall.first()
        .map(|rr| {
            let mut ks: Vec<usize> = rr.recall.keys()
                .filter_map(|k| k.strip_prefix("recall@").and_then(|n| n.parse().ok()))
                .collect();
            ks.sort_unstable();
            ks
        })
        .unwrap_or_else(|| vec![5, 10]);

    let k_header: String = all_ks.iter().map(|k| format!("{:>10}", format!("Recall@{}", k))).collect::<Vec<_>>().join("  ");
    println!("{:<20}  {}  {:>8}  {:>14}", "Config", k_header, "MRR", "Avg Latency(ms)");
    println!("{}", "─".repeat(44 + all_ks.len() * 12));
    for rr in &r.recall {
        let k_vals: String = all_ks.iter()
            .map(|k| format!("{:>10.3}", rr.recall.get(&format!("recall@{}", k)).copied().unwrap_or(0.0)))
            .collect::<Vec<_>>()
            .join("  ");
        println!(
            "{:<20}  {}  {:>8.3}  {:>14.1}",
            rr.config,
            k_vals,
            rr.mrr,
            rr.avg_latency_ms
        );
    }

    println!("\n── Latency breakdown (hybrid+expand, per query) ──────────────");
    let l = &r.latency_hybrid_expand;
    println!("  Embed query : {:6.1} ms", l.embed_ms);
    println!("  BM25 search : {:6.1} ms", l.bm25_ms);
    println!("  Vec search  : {:6.1} ms", l.vector_ms);
    println!("  RRF fusion  : {:6.1} ms", l.rrf_ms);
    println!("  Ctx expand  : {:6.1} ms", l.expand_ms);
    println!("  ─────────────────────────");
    println!("  Total       : {:6.1} ms", l.total_ms);

    if !r.ingest_timing.is_empty() {
        println!("\n── Ingestion timing ──────────────────────────────────────────");
        println!(
            "{:<30}  {:>8}  {:>8}  {:>10}  {:>10}",
            "Document", "Chunks", "KB", "Embed(ms)", "Total(ms)"
        );
        println!("{}", "─".repeat(72));
        for it in &r.ingest_timing {
            println!(
                "{:<30}  {:>8}  {:>8}  {:>10}  {:>10}",
                &it.doc_title[..it.doc_title.len().min(30)],
                it.chunk_count,
                it.size_bytes / 1024,
                it.embed_ms,
                it.total_ms,
            );
        }

        // Aggregate throughput
        let total_time_ms: u64 = r.ingest_timing.iter().map(|t| t.total_ms).sum();
        let total_docs = r.ingest_timing.len();
        let total_chars: usize = r.ingest_timing.iter().map(|t| t.char_count).sum();
        let total_kb: u64 = r.ingest_timing.iter().map(|t| t.size_bytes).sum::<u64>() / 1024;
        let secs = total_time_ms as f64 / 1000.0;
        let docs_per_sec = if secs > 0.0 { total_docs as f64 / secs } else { 0.0 };
        let chars_per_sec = if secs > 0.0 { total_chars as f64 / secs } else { 0.0 };
        println!("{}", "─".repeat(72));
        println!(
            "Throughput: {} docs in {:.1}s  →  {:.1} docs/s  {:.0} chars/s  {} KB total",
            total_docs, secs, docs_per_sec, chars_per_sec, total_kb
        );
    }

    if let Some(raptor) = &r.raptor {
        println!("\n── RAPTOR confidence-gate ablation ───────────────────────────");
        println!(
            "{:<22}  {:>10}  {:>10}  {:>10}  {:>10}",
            "Config", "Recall@5", "Recall@10", "Expanded", "Latency(ms)"
        );
        println!("{}", "─".repeat(70));
        for rr in raptor {
            println!(
                "{:<22}  {:>10.3}  {:>10.3}  {:>10}  {:>10.1}",
                rr.config,
                rr.recall.get("recall@5").copied().unwrap_or(0.0),
                rr.recall.get("recall@10").copied().unwrap_or(0.0),
                rr.nodes_expanded,
                rr.avg_latency_ms,
            );
        }
    }

    if let Some(e2e) = &r.e2e {
        println!("\n── E2E Answer Quality ────────────────────────────────────────────────────────");
        println!(
            "  {:<22}  {:>10}  {:>10}  {:>14}  {:>8}",
            "Config", "Exact Match", "Token F1", "Avg Lat (ms)", "n"
        );
        println!("{}", "─".repeat(72));
        println!(
            "  {:<22}  {:>10}  {:>10}  {:>14.1}  {:>8}",
            "hybrid_rag",
            format!("{:.1}%", e2e.exact_match * 100.0),
            format!("{:.1}%", e2e.f1 * 100.0),
            e2e.avg_latency_ms,
            e2e.sample_count,
        );

        if let Some(norag) = &r.no_rag_baseline {
            println!(
                "  {:<22}  {:>10}  {:>10}  {:>14.1}  {:>8}",
                "no_rag_baseline",
                format!("{:.1}%", norag.exact_match * 100.0),
                format!("{:.1}%", norag.f1 * 100.0),
                norag.avg_latency_ms,
                norag.sample_count,
            );
            let em_gain = (e2e.exact_match - norag.exact_match) * 100.0;
            let f1_gain = (e2e.f1 - norag.f1) * 100.0;
            println!("{}", "─".repeat(72));
            println!(
                "  RAG gain:  EM {:+.1}pp   F1 {:+.1}pp",
                em_gain, f1_gain
            );
        }
    }

    if let Some(ci) = &r.e2e_ci {
        println!("\n── E2E Bootstrap 95% CI (n={}, B={}) ─────────────────────────────────────",
            ci.n, ci.bootstrap_samples);
        println!("  EM: {:.1}%  [{:.1}%, {:.1}%]",
            ci.exact_match * 100.0, ci.em_ci_low * 100.0, ci.em_ci_high * 100.0);
        println!("  F1: {:.1}%  [{:.1}%, {:.1}%]",
            ci.f1 * 100.0, ci.f1_ci_low * 100.0, ci.f1_ci_high * 100.0);
        println!("  Latency  p50={:.1}ms  p95={:.1}ms  p99={:.1}ms",
            ci.p50_latency_ms, ci.p95_latency_ms, ci.p99_latency_ms);
    }
}

// ── Pure-Rust metrics helpers ─────────────────────────────────────────────────

/// NDCG@k: gain = rel / log2(rank + 2).  Relevance values from qrels (u8).
fn ndcg_at_k(ranked_ids: &[String], qrels: &HashMap<String, u8>, k: usize) -> f64 {
    let dcg: f64 = ranked_ids.iter().take(k).enumerate().map(|(i, id)| {
        let rel = *qrels.get(id).unwrap_or(&0) as f64;
        rel / (i as f64 + 2.0).log2()
    }).sum();
    let mut ideal: Vec<f64> = qrels.values().map(|&r| r as f64).collect();
    ideal.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let idcg: f64 = ideal.iter().take(k).enumerate()
        .map(|(i, &r)| r / (i as f64 + 2.0).log2()).sum();
    if idcg == 0.0 { 0.0 } else { dcg / idcg }
}

/// Mean Average Precision over the full ranked list.
fn average_precision(ranked_ids: &[String], qrels: &HashMap<String, u8>) -> f64 {
    let total_rel = qrels.values().filter(|&&r| r > 0).count();
    if total_rel == 0 { return 0.0; }
    let mut found = 0usize;
    let mut sum = 0.0_f64;
    for (i, id) in ranked_ids.iter().enumerate() {
        if qrels.get(id).copied().unwrap_or(0) > 0 {
            found += 1;
            sum += found as f64 / (i + 1) as f64;
        }
    }
    sum / total_rel as f64
}

/// Percentile via linear interpolation on a pre-sorted slice.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let n = sorted.len();
    if n == 1 { return sorted[0]; }
    let rank = p / 100.0 * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    sorted[lo] + (rank - lo as f64) * (sorted[hi] - sorted[lo])
}

/// Bootstrap 95% CI for per-sample scores.
/// Uses a deterministic LCG (no external crate) seeded with `seed`.
fn bootstrap_ci(scores: &[f64], n_samples: usize, seed: u64) -> (f64, f64) {
    if scores.is_empty() { return (0.0, 0.0); }
    let n = scores.len();
    let mut rng = seed;
    // Knuth multiplicative LCG
    let step = |s: u64| -> (u64, usize) {
        let s2 = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s2, (s2 >> 33) as usize % n)
    };
    let mut means: Vec<f64> = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let mut sum = 0.0;
        for _ in 0..n {
            let (r, idx) = step(rng);
            rng = r;
            sum += scores[idx];
        }
        means.push(sum / n as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (percentile_sorted(&means, 2.5), percentile_sorted(&means, 97.5))
}

/// Build E2EWithCI from a finished E2EResult.
fn compute_e2e_with_ci(e2e: &E2EResult, bootstrap_samples: usize, seed: u64) -> E2EWithCI {
    let em_scores: Vec<f64> = e2e.per_question.iter()
        .map(|q| if q.exact_match { 1.0 } else { 0.0 }).collect();
    let f1_scores: Vec<f64> = e2e.per_question.iter().map(|q| q.f1).collect();
    let mut lat_sorted: Vec<f64> = e2e.per_question.iter().map(|q| q.latency_ms).collect();
    lat_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (em_lo, em_hi) = bootstrap_ci(&em_scores, bootstrap_samples, seed);
    let (f1_lo, f1_hi) = bootstrap_ci(&f1_scores, bootstrap_samples, seed.wrapping_add(1));
    E2EWithCI {
        exact_match: e2e.exact_match,
        em_ci_low: em_lo,
        em_ci_high: em_hi,
        f1: e2e.f1,
        f1_ci_low: f1_lo,
        f1_ci_high: f1_hi,
        avg_latency_ms: e2e.avg_latency_ms,
        p50_latency_ms: percentile_sorted(&lat_sorted, 50.0),
        p95_latency_ms: percentile_sorted(&lat_sorted, 95.0),
        p99_latency_ms: percentile_sorted(&lat_sorted, 99.0),
        n: e2e.sample_count,
        bootstrap_samples,
    }
}

// ── RAPTOR CLI builder (standalone, no TaskRouter) ────────────────────────────

/// Minimal k-means for RAPTOR clustering (cosine similarity via dot product).
fn bench_kmeans(embeddings: &[Vec<f32>], k: usize) -> Vec<usize> {
    if embeddings.is_empty() || k == 0 { return vec![]; }
    let k = k.min(embeddings.len());
    let dim = embeddings[0].len();
    let mut centroids: Vec<Vec<f32>> = embeddings.iter().take(k).cloned().collect();
    let mut assignments = vec![0usize; embeddings.len()];
    for _ in 0..20 {
        let mut changed = false;
        for (i, emb) in embeddings.iter().enumerate() {
            let best = centroids.iter().enumerate()
                .map(|(c, cent)| (c, dot_product(emb, cent)))
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(c, _)| c)
                .unwrap_or(0);
            if assignments[i] != best { assignments[i] = best; changed = true; }
        }
        if !changed { break; }
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, emb) in embeddings.iter().enumerate() {
            let c = assignments[i];
            for (d, v) in emb.iter().enumerate() { sums[c][d] += v; }
            counts[c] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim { centroids[c][d] = sums[c][d] / counts[c] as f32; }
            }
        }
    }
    assignments
}

/// Build a RAPTOR tree for one document using CLI embed + chat servers.
/// Returns number of RAPTOR nodes created (0 if tree already exists or no embeddings).
async fn build_raptor_tree_cli(
    db: &RagDb,
    doc_id: i64,
    embed_server: &EmbedServer,
    chat_server: &ChatServer,
) -> Result<usize> {
    const MIN_CLUSTER: usize = 2;
    const MAX_CLUSTERS: usize = 10;
    const MAX_DEPTH: usize = 2;

    if db.has_raptor_tree(doc_id).unwrap_or(false) { return Ok(0); }

    let chunk_embeddings = db.get_chunk_embeddings_for_doc(doc_id)
        .map_err(|e| anyhow::anyhow!("get_chunk_embeddings_for_doc: {}", e))?;
    if chunk_embeddings.is_empty() { return Ok(0); }

    let chunk_ids_all: Vec<i64> = chunk_embeddings.iter().map(|(id, _)| *id).collect();
    let chunks = db.get_chunks_by_ids(&chunk_ids_all).unwrap_or_default();
    let chunk_text_map: HashMap<i64, String> =
        chunks.iter().map(|c| (c.id, c.text.clone())).collect();

    let mut nodes_created = 0usize;
    let mut current_items: Vec<(i64, Vec<f32>)> = chunk_embeddings;
    let mut current_level = 0usize;

    while current_level < MAX_DEPTH && current_items.len() > 1 {
        let n_clusters = (current_items.len() / MIN_CLUSTER).max(1).min(MAX_CLUSTERS);
        if n_clusters <= 1 { break; }

        let embs: Vec<Vec<f32>> = current_items.iter().map(|(_, e)| e.clone()).collect();
        let assignments = bench_kmeans(&embs, n_clusters);

        let mut groups: HashMap<usize, Vec<(i64, Vec<f32>)>> = HashMap::new();
        for (i, (id, emb)) in current_items.iter().enumerate() {
            groups.entry(assignments[i]).or_default().push((*id, emb.clone()));
        }

        let mut next_items: Vec<(i64, Vec<f32>)> = Vec::new();
        for (_cid, items) in &groups {
            if items.len() < MIN_CLUSTER { continue; }
            let child_ids: Vec<i64> = items.iter().map(|(id, _)| *id).collect();

            let child_texts: Vec<String> = child_ids.iter()
                .filter_map(|id| {
                    if current_level == 0 {
                        chunk_text_map.get(id).cloned()
                    } else {
                        db.get_raptor_node(*id).ok().map(|n| n.summary_text)
                    }
                })
                .collect();
            if child_texts.is_empty() { continue; }

            let max_len = 4000usize;
            let mut combined = String::new();
            for (i, t) in child_texts.iter().enumerate() {
                combined.push_str(&format!("Passage {}:\n{}\n\n", i + 1, t));
                if combined.len() > max_len {
                    // Truncate at a valid UTF-8 char boundary to avoid panic on multibyte chars
                    let boundary = combined.floor_char_boundary(max_len);
                    combined.truncate(boundary);
                    break;
                }
            }
            let prompt = format!(
                "Summarize the following passages into one concise paragraph:\n\n{}\nSummary:",
                combined
            );
            let summary = chat_server.chat_complete(
                "You are a precise summarization assistant. Produce a single concise paragraph.",
                &prompt,
            ).await.unwrap_or_else(|_| combined.chars().take(200).collect());

            let emb_vec = embed_server.embed(vec![summary.clone()]).await
                .unwrap_or_default()
                .into_iter().next().unwrap_or_default();
            if emb_vec.is_empty() { continue; }

            // Confidence: shift mean child similarity into a [-2, 0] range
            let conf: f64 = items.iter()
                .map(|(_, e)| dot_product(&emb_vec, e) as f64)
                .sum::<f64>() / items.len() as f64 - 2.0;

            let node_id = db.insert_raptor_node(
                doc_id,
                (current_level + 1) as i32,
                None,
                &summary,
                conf,
                &child_ids,
                Some(&emb_vec),
            ).map_err(|e| anyhow::anyhow!("insert_raptor_node: {}", e))?;
            nodes_created += 1;
            next_items.push((node_id, emb_vec));
        }

        if next_items.is_empty() { break; }
        current_items = next_items;
        current_level += 1;
    }
    Ok(nodes_created)
}

// ── BEIR-format retrieval benchmark ─────────────────────────────────────────

fn load_beir_queries(path: &Path) -> Result<Vec<BeirQuery>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Cannot read BEIR queries: {}", path.display()))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).context("Invalid query JSON line"))
        .collect()
}

fn load_beir_corpus(path: &Path) -> Result<Vec<BeirDoc>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Cannot read BEIR corpus: {}", path.display()))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).context("Invalid corpus JSON line"))
        .collect()
}

/// Returns: query_id → { doc_id → relevance }.
fn load_beir_qrels(path: &Path) -> Result<HashMap<String, HashMap<String, u8>>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Cannot read BEIR qrels: {}", path.display()))?;
    let mut qrels: HashMap<String, HashMap<String, u8>> = HashMap::new();
    for line in text.lines() {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 3 || parts[0] == "query-id" { continue; }
        let rel: u8 = parts[2].trim().parse().unwrap_or(0);
        qrels.entry(parts[0].to_string()).or_default()
            .insert(parts[1].to_string(), rel);
    }
    Ok(qrels)
}

async fn cmd_beir_bench(args: BeirBenchArgs) -> Result<()> {
    let server_bin = resolve_server_bin(&args.llama_server)?;

    let corpus_path = args.beir_dir.join("corpus.jsonl");
    let queries_path = args.beir_dir.join("queries.jsonl");
    let qrels_path = args.beir_dir.join("qrels").join("test.tsv");
    for p in &[&corpus_path, &queries_path, &qrels_path] {
        if !p.exists() { bail!("BEIR file not found: {}", p.display()); }
    }

    println!("[beir] Loading dataset from {}", args.beir_dir.display());
    let corpus = load_beir_corpus(&corpus_path)?;
    let queries = load_beir_queries(&queries_path)?;
    let qrels = load_beir_qrels(&qrels_path)?;
    println!("[beir] {} docs  {} queries  {} with qrels",
        corpus.len(), queries.len(), qrels.len());

    std::fs::create_dir_all(&args.workspace_dir)
        .with_context(|| format!("Cannot create workspace: {}", args.workspace_dir.display()))?;

    let db = RagDb::open(&args.workspace_dir.join("rag.db"))
        .map_err(|e| anyhow::anyhow!("RagDb: {}", e))?;
    let bm25 = BM25Index::open(&args.workspace_dir.join("bm25_index"))
        .map_err(|e| anyhow::anyhow!("BM25: {}", e))?;
    let vec_index = VectorIndex::load_from_db(&db)
        .map_err(|e| anyhow::anyhow!("VecIndex: {}", e))?;
    let mut server = EmbedServer::start(&server_bin, &args.embed_model, args.embed_port).await?;

    // Ingest corpus (BEIR _id used as document title for deduplication & oracle lookup)
    let already_cached = corpus.iter()
        .filter(|d| db.document_exists(&format!("beir:{}", d.id)).unwrap_or(false))
        .count();
    let to_ingest = corpus.len() - already_cached;
    println!("[beir] Ingesting corpus ({} docs, {} already cached)...", corpus.len(), already_cached);
    let mut ingested_count = 0usize;
    let mut skipped_embed = 0usize;
    let progress_step = (to_ingest / 10).max(50); // ~10 progress lines regardless of corpus size
    let ingest_t0 = Instant::now();

    // --- Parallel ingestion ---
    // Collect docs that still need ingesting (skip already-cached ones).
    // Embed requests to llama-server are fired in parallel batches of
    // EMBED_CONCURRENCY; DB/BM25/vector writes remain sequential since
    // those stores are not concurrency-safe.
    //
    // BM25 checkpointing: add_chunks_batch() commits (flushes Tantivy segment)
    // on every call. Tantivy segment merging is O(total_docs) per commit, so
    // committing per batch (~8 docs) causes O(N²) slowdown on large corpora
    // (fiqa: ~170k chunks → 7k commits → 1 doc/s at the end).
    //
    // Fix: accumulate BM25 data into bm25_pending and flush every
    // BM25_CHECKPOINT_DOCS docs. For fiqa this is ~114 commits instead of 7000
    // (60× fewer), while still checkpointing to disk so a crash only loses at
    // most BM25_CHECKPOINT_DOCS worth of BM25 data (SQLite + embeddings are
    // safe — written per-doc). On re-run, SQLite-cached docs would be skipped,
    // so without checkpointing BM25 would be permanently empty for that
    // workspace after any crash.
    const BM25_CHECKPOINT_DOCS: usize = 500;
    let mut bm25_pending: Vec<(i64, String, String)> = Vec::new();
    const EMBED_CONCURRENCY: usize = 8;

    // Pre-compute chunking for every pending doc so the embed futures only do I/O.
    struct PendingDoc<'a> {
        doc: &'a BeirDoc,
        path_key: String,
        chunks: Vec<app_lib::rag::chunker::Chunk>,
    }
    let pending: Vec<PendingDoc> = corpus.iter()
        .filter_map(|doc| {
            let path_key = format!("beir:{}", doc.id);
            if db.document_exists(&path_key).unwrap_or(false) { return None; }
            let full_text = if doc.title.is_empty() {
                doc.text.clone()
            } else {
                format!("{}\n\n{}", doc.title, doc.text)
            };
            let chunks = chunk_text_default(&full_text);
            if chunks.is_empty() { return None; }
            Some(PendingDoc { doc, path_key, chunks })
        })
        .collect();

    for batch in pending.chunks(EMBED_CONCURRENCY) {
        // Fire all embed requests in this batch concurrently.
        let embed_futs: Vec<_> = batch.iter()
            .map(|pd| {
                let texts: Vec<String> = pd.chunks.iter().map(|c| c.text.clone()).collect();
                server.embed(texts)
            })
            .collect();
        let embed_results = futures_util::future::join_all(embed_futs).await;

        // Write results sequentially (DB/BM25/vec_index are not thread-safe).
        for (pd, emb_result) in batch.iter().zip(embed_results) {
            let embeddings = match emb_result {
                Ok(embs) => embs,
                Err(e) => {
                    eprintln!("[beir] WARN: skipping doc {} — embed failed: {}", pd.doc.id, e);
                    skipped_embed += 1;
                    continue;
                }
            };

            let db_doc_id = db.insert_document(&pd.path_key, &pd.doc.id, "beir", pd.chunks.len() as i64)
                .map_err(|e| anyhow::anyhow!("insert_document: {}", e))?;
            let chunk_data: Vec<(usize, String, String)> = pd.chunks.iter()
                .map(|c| (c.index, c.text.clone(), c.metadata.clone())).collect();
            let chunk_ids = db.insert_chunks(db_doc_id, &chunk_data)
                .map_err(|e| anyhow::anyhow!("insert_chunks: {}", e))?;
            let bm25_for_doc: Vec<(i64, String, String)> = chunk_ids.iter().zip(pd.chunks.iter())
                .map(|(&id, c)| (id, c.text.clone(), pd.doc.id.clone())).collect();
            bm25_pending.extend(bm25_for_doc);
            for (i, emb) in embeddings.iter().enumerate() {
                if i < chunk_ids.len() && !emb.is_empty() {
                    let _ = db.set_chunk_embedding(chunk_ids[i], emb, None);
                    vec_index.insert(chunk_ids[i], emb.clone());
                }
            }

            ingested_count += 1;
            // Periodic BM25 checkpoint — flush accumulated chunks to Tantivy
            // every BM25_CHECKPOINT_DOCS docs so a crash loses at most one
            // checkpoint window of BM25 data.
            if ingested_count % BM25_CHECKPOINT_DOCS == 0 && !bm25_pending.is_empty() {
                bm25.add_chunks_batch(&bm25_pending)
                    .map_err(|e| anyhow::anyhow!("BM25 checkpoint: {}", e))?;
                bm25_pending.clear();
            }
            if progress_step == 0 || ingested_count % progress_step == 0 || ingested_count == to_ingest {
                let elapsed = ingest_t0.elapsed().as_secs_f64();
                let rate = ingested_count as f64 / elapsed.max(0.001);
                let eta_s = if rate > 0.0 { (to_ingest - ingested_count) as f64 / rate } else { 0.0 };
                println!(
                    "[beir] ingest {}/{} docs  ({:.0} docs/s  ETA {:.0}s)",
                    ingested_count, to_ingest, rate, eta_s
                );
            }
        }
    }
    vec_index.rebuild_if_needed();

    // Tail flush — commit any BM25 docs accumulated since the last checkpoint.
    if !bm25_pending.is_empty() {
        println!("[beir] Committing BM25 tail ({} chunks)…", bm25_pending.len());
        bm25.add_chunks_batch(&bm25_pending)
            .map_err(|e| anyhow::anyhow!("BM25 tail commit: {}", e))?;
    }

    let total_elapsed = ingest_t0.elapsed().as_secs_f64();
    if skipped_embed > 0 {
        eprintln!("[beir] WARN: {} doc(s) skipped due to embed failure (too large for model context)", skipped_embed);
    }
    println!(
        "[beir] Corpus ready: {} docs total  ({} newly ingested, {} skipped in {:.1}s)",
        db.document_count().unwrap_or(0), ingested_count, skipped_embed, total_elapsed
    );

    // Build db_doc_id → BEIR id map for retrieval oracle
    let db_docs = db.list_documents().unwrap_or_default();
    let db_id_to_beir: HashMap<i64, String> = db_docs.iter()
        .filter_map(|d| d.path.strip_prefix("beir:").map(|s| (d.id, s.to_string())))
        .collect();

    struct BeirCfg { name: &'static str, bm25: bool, vec: bool }
    let configs = [
        BeirCfg { name: "bm25_only",   bm25: true,  vec: false },
        BeirCfg { name: "vector_only", bm25: false, vec: true  },
        BeirCfg { name: "hybrid",      bm25: true,  vec: true  },
    ];
    let max_k = 100usize;
    let mut results: Vec<BeirMetrics> = Vec::new();

    for cfg in &configs {
        let (mut ndcg_sum, mut ap_sum, mut r100_hits, mut mrr_sum, mut lat_sum) =
            (0.0_f64, 0.0_f64, 0usize, 0.0_f64, 0.0_f64);
        let mut eval_n = 0usize;

        for q in &queries {
            let qrel = match qrels.get(&q.id) { Some(r) if !r.is_empty() => r, _ => continue };
            let t0 = Instant::now();
            let emb_resp = server.embed(vec![q.text.clone()]).await?;
            let query_emb = emb_resp.into_iter().next().unwrap_or_default();
            let bm25_res: Vec<(i64, f32)> = if cfg.bm25 {
                bm25.search(&q.text, max_k).unwrap_or_default()
            } else { vec![] };
            let vec_res: Vec<(i64, f32)> = if cfg.vec && !query_emb.is_empty() {
                vec_index.search(&query_emb, max_k)
            } else { vec![] };
            let fused_ids: Vec<i64> = match (cfg.bm25, cfg.vec) {
                (true, true)  => rrf_fuse(&[bm25_res, vec_res]).iter().take(max_k).map(|r| r.chunk_id).collect(),
                (true, false) => bm25_res.iter().take(max_k).map(|(id, _)| *id).collect(),
                (false, true) => vec_res.iter().take(max_k).map(|(id, _)| *id).collect(),
                _             => vec![],
            };
            lat_sum += t0.elapsed().as_secs_f64() * 1000.0;
            eval_n += 1;

            // Deduplicate to one BEIR doc entry per ranked position
            let fetched = db.get_chunks_by_ids(&fused_ids).unwrap_or_default();
            let mut ranked_beir: Vec<String> = Vec::new();
            for id in &fused_ids {
                if let Some(c) = fetched.iter().find(|c| c.id == *id) {
                    if let Some(bid) = db_id_to_beir.get(&c.doc_id) {
                        if !ranked_beir.contains(bid) { ranked_beir.push(bid.clone()); }
                    }
                }
            }

            ndcg_sum += ndcg_at_k(&ranked_beir, qrel, 10);
            ap_sum   += average_precision(&ranked_beir, qrel);
            let hit = ranked_beir.iter().take(100).any(|id| qrel.get(id).copied().unwrap_or(0) > 0);
            if hit { r100_hits += 1; }
            mrr_sum += ranked_beir.iter().enumerate()
                .find_map(|(i, id)| if qrel.get(id).copied().unwrap_or(0) > 0 { Some(1.0 / (i + 1) as f64) } else { None })
                .unwrap_or(0.0);
        }

        let n = eval_n.max(1);
        let m = BeirMetrics {
            config: cfg.name.to_string(),
            ndcg_at_10: ndcg_sum / n as f64,
            map: ap_sum / n as f64,
            recall_at_100: r100_hits as f64 / n as f64,
            mrr: mrr_sum / n as f64,
            avg_latency_ms: lat_sum / n as f64,
        };
        println!("[beir] {:>14}  NDCG@10={:.3}  MAP={:.3}  R@100={:.3}  MRR={:.3}  lat={:.1}ms",
            m.config, m.ndcg_at_10, m.map, m.recall_at_100, m.mrr, m.avg_latency_ms);
        results.push(m);
    }

    server.stop();
    let out = BeirBenchOutput {
        timestamp: chrono::Utc::now().to_rfc3339(),
        dataset_dir: args.beir_dir.to_string_lossy().to_string(),
        query_count: queries.len(),
        doc_count: corpus.len(),
        results,
    };
    std::fs::write(&args.output, serde_json::to_string_pretty(&out)?)
        .with_context(|| format!("Cannot write BEIR results to {}", args.output.display()))?;
    println!("[beir] Results saved → {}", args.output.display());
    Ok(())
}

// ── Chunking ablation ─────────────────────────────────────────────────────────

/// Ingest a corpus directory using a custom ChunkerConfig (for ablation).
async fn ingest_corpus_with_config(
    corpus_dir: &Path,
    server: &EmbedServer,
    db: &RagDb,
    bm25: &BM25Index,
    vec_index: &VectorIndex,
    config: &ChunkerConfig,
) -> Result<(usize, u64)> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(corpus_dir)
        .with_context(|| format!("Cannot read corpus: {}", corpus_dir.display()))?
        .filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("txt"))
        .collect();
    entries.sort();
    if entries.is_empty() { bail!("No .txt files in {}", corpus_dir.display()); }

    let t0 = Instant::now();
    let mut total_chunks = 0usize;
    for path in &entries {
        let title = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?").to_string();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Cannot read: {}", path.display()))?;
        let path_str = path.to_string_lossy().to_string();
        let chunks = chunk_text(&text, config);
        if chunks.is_empty() { continue; }

        let doc_id = db.insert_document(&path_str, &title, "txt", chunks.len() as i64)
            .map_err(|e| anyhow::anyhow!("insert_document: {}", e))?;
        let chunk_data: Vec<(usize, String, String)> = chunks.iter()
            .map(|c| (c.index, c.text.clone(), c.metadata.clone())).collect();
        let chunk_ids = db.insert_chunks(doc_id, &chunk_data)
            .map_err(|e| anyhow::anyhow!("insert_chunks: {}", e))?;
        bm25.add_chunks_batch(&chunk_ids.iter().zip(chunks.iter())
            .map(|(&id, c)| (id, c.text.clone(), title.clone())).collect::<Vec<_>>())
            .map_err(|e| anyhow::anyhow!("BM25: {}", e))?;

        let embeddings = server.embed(chunks.iter().map(|c| c.text.clone()).collect()).await
            .with_context(|| format!("Embedding '{}' failed", title))?;
        for (i, emb) in embeddings.iter().enumerate() {
            if i < chunk_ids.len() && !emb.is_empty() {
                let _ = db.set_chunk_embedding(chunk_ids[i], emb, None);
                vec_index.insert(chunk_ids[i], emb.clone());
            }
        }
        total_chunks += chunks.len();
    }
    vec_index.rebuild_if_needed();
    Ok((total_chunks, t0.elapsed().as_millis() as u64))
}

async fn cmd_ablate_chunking(args: AblateChunkingArgs) -> Result<()> {
    let server_bin = resolve_server_bin(&args.llama_server)?;
    let chunk_sizes: Vec<usize> = args.chunk_sizes.split(',')
        .filter_map(|s| s.trim().parse().ok()).collect();
    let overlaps: Vec<usize> = args.overlaps.split(',')
        .filter_map(|s| s.trim().parse().ok()).collect();
    if chunk_sizes.is_empty() || overlaps.is_empty() {
        bail!("--chunk-sizes and --overlaps must each contain at least one value");
    }

    let qa_json = std::fs::read_to_string(&args.qa_file)
        .with_context(|| format!("Cannot read QA file: {}", args.qa_file.display()))?;
    let qa_pairs: Vec<QAPair> = serde_json::from_str(&qa_json).context("Invalid QA file")?;
    let top_ks = vec![5usize, 10];
    let mut server = EmbedServer::start(&server_bin, &args.embed_model, args.embed_port).await?;
    let mut points: Vec<ChunkAblationPoint> = Vec::new();
    let total = chunk_sizes.len() * overlaps.len();
    let mut idx = 0usize;

    for &cs in &chunk_sizes {
        for &ov in &overlaps {
            if ov >= cs { continue; }
            idx += 1;
            println!("[ablate-chunk] [{}/{}] chunk_size={}  overlap={}", idx, total, cs, ov);
            let cp_dir = args.workspace_dir.join(format!("cs{}ov{}", cs, ov));
            std::fs::create_dir_all(&cp_dir)?;
            let cp_db = RagDb::open(&cp_dir.join("rag.db"))
                .map_err(|e| anyhow::anyhow!("RagDb: {}", e))?;
            let cp_bm25 = BM25Index::open(&cp_dir.join("bm25_index"))
                .map_err(|e| anyhow::anyhow!("BM25: {}", e))?;
            let cp_vec = VectorIndex::load_from_db(&cp_db)
                .map_err(|e| anyhow::anyhow!("VecIndex: {}", e))?;

            let config = ChunkerConfig { chunk_size: cs, overlap: ov, ..Default::default() };
            let (n_chunks, ingest_ms) = ingest_corpus_with_config(
                &args.corpus_dir, &server, &cp_db, &cp_bm25, &cp_vec, &config,
            ).await?;

            let (recall_results, _) =
                run_recall_bench(&qa_pairs, &top_ks, &cp_db, &cp_bm25, &cp_vec, &server).await?;

            let hybrid = recall_results.iter().find(|r| r.config == "hybrid");
            let r5  = hybrid.and_then(|r| r.recall.get("recall@5").copied()).unwrap_or(0.0);
            let r10 = hybrid.and_then(|r| r.recall.get("recall@10").copied()).unwrap_or(0.0);
            let mrr = hybrid.map(|r| r.mrr).unwrap_or(0.0);
            let lat = hybrid.map(|r| r.avg_latency_ms).unwrap_or(0.0);
            println!("[ablate-chunk]   R@5={:.3}  R@10={:.3}  MRR={:.3}  chunks={}  ingest={}ms",
                r5, r10, mrr, n_chunks, ingest_ms);
            points.push(ChunkAblationPoint {
                chunk_size: cs, overlap: ov, n_chunks_total: n_chunks,
                recall_5: r5, recall_10: r10, mrr, avg_query_ms: lat,
                ingest_total_ms: ingest_ms,
            });
            let _ = std::fs::remove_dir_all(&cp_dir);
        }
    }

    server.stop();
    std::fs::write(&args.output, serde_json::to_string_pretty(&points)?)
        .with_context(|| format!("Cannot write chunking ablation to {}", args.output.display()))?;
    println!("\n── Chunking Ablation ────────────────────────────────────────────");
    println!("{:>10} {:>8} {:>8} {:>8} {:>8} {:>10}", "ChunkSize", "Overlap", "R@5", "R@10", "MRR", "Ingest(ms)");
    println!("{}", "─".repeat(60));
    for p in &points {
        println!("{:>10} {:>8} {:>8.3} {:>8.3} {:>8.3} {:>10}",
            p.chunk_size, p.overlap, p.recall_5, p.recall_10, p.mrr, p.ingest_total_ms);
    }
    println!("\n[ablate-chunk] Results saved → {}", args.output.display());
    Ok(())
}

// ── RRF-k ablation ────────────────────────────────────────────────────────────

/// Hybrid recall benchmark with a specific RRF k constant.
async fn run_recall_hybrid_rrf_k(
    qa_pairs: &[QAPair],
    top_ks: &[usize],
    db: &RagDb,
    bm25: &BM25Index,
    vec_index: &VectorIndex,
    server: &EmbedServer,
    rrf_k: f64,
) -> Result<RecallResult> {
    let docs = db.list_documents().map_err(|e| anyhow::anyhow!("{}", e))?;
    let doc_title_map: HashMap<i64, String> =
        docs.iter().map(|d| (d.id, d.title.clone())).collect();
    let max_k = *top_ks.iter().max().unwrap_or(&10);
    let mut per_question: Vec<PerQuestionResult> = Vec::new();
    let mut total_lat = 0.0_f64;

    for qa in qa_pairs {
        let t0 = Instant::now();
        let query_emb = server.embed(vec![qa.question.clone()]).await?
            .into_iter().next().unwrap_or_default();
        let bm25_res = bm25.search(&qa.question, max_k).unwrap_or_default();
        let vec_res  = if !query_emb.is_empty() { vec_index.search(&query_emb, max_k) } else { vec![] };
        let fused = rrf_fuse_with_k(&[bm25_res, vec_res], rrf_k);
        let fused_ids: Vec<i64> = fused.iter().take(max_k).map(|r| r.chunk_id).collect();
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        total_lat += total_ms;

        let fetched = db.get_chunks_by_ids(&fused_ids).unwrap_or_default();
        let first_rank = fused_ids.iter().enumerate().find_map(|(i, id)| {
            fetched.iter().find(|c| c.id == *id).and_then(|c| {
                let title = doc_title_map.get(&c.doc_id).map(|s| s.as_str()).unwrap_or("");
                if is_relevant(&c.text, title, qa) { Some(i + 1) } else { None }
            })
        });
        let mut hit_at_k = HashMap::new();
        for &k in top_ks {
            let hit = fused_ids.iter().take(k).any(|id| {
                fetched.iter().find(|c| c.id == *id).map_or(false, |c| {
                    let t = doc_title_map.get(&c.doc_id).map(|s| s.as_str()).unwrap_or("");
                    is_relevant(&c.text, t, qa)
                })
            });
            hit_at_k.insert(format!("recall@{}", k), hit);
        }
        per_question.push(PerQuestionResult {
            question: qa.question.clone(), hit_at_k,
            first_relevant_rank: first_rank, latency_ms: total_ms,
        });
    }

    let n = qa_pairs.len().max(1);
    let mut recall = HashMap::new();
    for &k in top_ks {
        let key = format!("recall@{}", k);
        let hits = per_question.iter().filter(|q| *q.hit_at_k.get(&key).unwrap_or(&false)).count();
        recall.insert(key, hits as f64 / n as f64);
    }
    let mrr = per_question.iter()
        .map(|q| q.first_relevant_rank.map(|r| 1.0 / r as f64).unwrap_or(0.0))
        .sum::<f64>() / n as f64;
    Ok(RecallResult {
        config: format!("hybrid_k{}", rrf_k as u64),
        recall, mrr,
        avg_latency_ms: total_lat / n as f64,
        per_question,
    })
}

async fn cmd_ablate_rrf_k(args: AblateRrfKArgs) -> Result<()> {
    let server_bin = resolve_server_bin(&args.llama_server)?;
    let rrf_k_vals: Vec<f64> = args.rrf_k_values.split(',')
        .filter_map(|s| s.trim().parse::<f64>().ok()).collect();
    if rrf_k_vals.is_empty() {
        bail!("--rrf-k-values must contain at least one value, e.g. '10,30,60,100,200'");
    }

    let db = RagDb::open(&args.workspace_dir.join("rag.db"))
        .map_err(|e| anyhow::anyhow!("RagDb: {}", e))?;
    let bm25 = BM25Index::open(&args.workspace_dir.join("bm25_index"))
        .map_err(|e| anyhow::anyhow!("BM25: {}", e))?;
    let vec_index = VectorIndex::load_from_db(&db)
        .map_err(|e| anyhow::anyhow!("VecIndex: {}", e))?;
    vec_index.rebuild_if_needed();

    let qa_json = std::fs::read_to_string(&args.qa_file)
        .with_context(|| format!("Cannot read QA file: {}", args.qa_file.display()))?;
    let qa_pairs: Vec<QAPair> = serde_json::from_str(&qa_json).context("Invalid QA file")?;
    let top_ks = vec![5usize, 10];
    let mut server = EmbedServer::start(&server_bin, &args.embed_model, args.embed_port).await?;
    let mut points: Vec<RrfKPoint> = Vec::new();

    for &k in &rrf_k_vals {
        println!("[ablate-rrf-k] k={}", k);
        let res = run_recall_hybrid_rrf_k(&qa_pairs, &top_ks, &db, &bm25, &vec_index, &server, k).await?;
        let r5  = res.recall.get("recall@5").copied().unwrap_or(0.0);
        let r10 = res.recall.get("recall@10").copied().unwrap_or(0.0);
        println!("[ablate-rrf-k]   R@5={:.3}  R@10={:.3}  MRR={:.3}  lat={:.1}ms",
            r5, r10, res.mrr, res.avg_latency_ms);
        points.push(RrfKPoint { rrf_k: k, recall_5: r5, recall_10: r10, mrr: res.mrr, avg_latency_ms: res.avg_latency_ms });
    }

    server.stop();
    std::fs::write(&args.output, serde_json::to_string_pretty(&points)?)
        .with_context(|| format!("Cannot write RRF-k ablation to {}", args.output.display()))?;
    println!("\n── RRF k Ablation ───────────────────────────────────────────────");
    println!("{:>8} {:>8} {:>8} {:>8} {:>12}", "RRF-k", "R@5", "R@10", "MRR", "Latency(ms)");
    println!("{}", "─".repeat(52));
    for p in &points {
        println!("{:>8} {:>8.3} {:>8.3} {:>8.3} {:>12.1}",
            p.rrf_k, p.recall_5, p.recall_10, p.mrr, p.avg_latency_ms);
    }
    println!("\n[ablate-rrf-k] Results saved → {}", args.output.display());
    Ok(())
}

// ── Quantization ablation ─────────────────────────────────────────────────────

async fn cmd_ablate_quant(args: AblateQuantArgs) -> Result<()> {
    let server_bin = resolve_server_bin(&args.llama_server)?;
    let model_paths: Vec<PathBuf> = args.embed_models.split(',')
        .map(|s| PathBuf::from(s.trim())).collect();
    if model_paths.is_empty() { bail!("--embed-models must contain at least one path"); }

    let qa_json = std::fs::read_to_string(&args.qa_file)
        .with_context(|| format!("Cannot read QA file: {}", args.qa_file.display()))?;
    let qa_pairs: Vec<QAPair> = serde_json::from_str(&qa_json).context("Invalid QA file")?;
    let top_ks = vec![5usize, 10];
    let mut points: Vec<QuantAblationPoint> = Vec::new();

    for model_path in &model_paths {
        if !model_path.exists() {
            eprintln!("[ablate-quant] Model not found, skipping: {}", model_path.display());
            continue;
        }
        let model_name = model_path.file_stem().and_then(|s| s.to_str())
            .unwrap_or("?").to_string();
        println!("[ablate-quant] Model: {}", model_name);

        let cp_dir = args.workspace_dir.join(format!("quant_{}", model_name));
        std::fs::create_dir_all(&cp_dir)?;
        let cp_db = RagDb::open(&cp_dir.join("rag.db"))
            .map_err(|e| anyhow::anyhow!("RagDb: {}", e))?;
        let cp_bm25 = BM25Index::open(&cp_dir.join("bm25_index"))
            .map_err(|e| anyhow::anyhow!("BM25: {}", e))?;
        let cp_vec = VectorIndex::load_from_db(&cp_db)
            .map_err(|e| anyhow::anyhow!("VecIndex: {}", e))?;

        let mut server = EmbedServer::start(&server_bin, model_path, args.embed_port).await?;
        let config = ChunkerConfig::default();
        let _ = ingest_corpus_with_config(
            &args.corpus_dir, &server, &cp_db, &cp_bm25, &cp_vec, &config,
        ).await?;

        // Sample embed latency per query
        let sample: Vec<String> = qa_pairs.iter().take(50).map(|q| q.question.clone()).collect();
        let t_e = Instant::now();
        let _ = server.embed(sample.clone()).await?;
        let avg_embed_ms = t_e.elapsed().as_secs_f64() * 1000.0 / sample.len().max(1) as f64;

        let (recall_results, _) =
            run_recall_bench(&qa_pairs, &top_ks, &cp_db, &cp_bm25, &cp_vec, &server).await?;
        server.stop();

        let hybrid = recall_results.iter().find(|r| r.config == "hybrid");
        let r5  = hybrid.and_then(|r| r.recall.get("recall@5").copied()).unwrap_or(0.0);
        let r10 = hybrid.and_then(|r| r.recall.get("recall@10").copied()).unwrap_or(0.0);
        let mrr = hybrid.map(|r| r.mrr).unwrap_or(0.0);
        let lat = hybrid.map(|r| r.avg_latency_ms).unwrap_or(0.0);
        println!("[ablate-quant]   R@5={:.3}  R@10={:.3}  MRR={:.3}  embed_ms/q={:.2}",
            r5, r10, mrr, avg_embed_ms);
        points.push(QuantAblationPoint {
            model_name, recall_5: r5, recall_10: r10, mrr,
            avg_embed_ms, avg_query_ms: lat,
        });
        let _ = std::fs::remove_dir_all(&cp_dir);
    }

    std::fs::write(&args.output, serde_json::to_string_pretty(&points)?)
        .with_context(|| format!("Cannot write quant ablation to {}", args.output.display()))?;
    println!("\n── Quantization Ablation ─────────────────────────────────────────────────────");
    println!("{:<40} {:>8} {:>8} {:>8} {:>12}", "Model", "R@5", "R@10", "MRR", "EmbedMs/q");
    println!("{}", "─".repeat(80));
    for p in &points {
        println!("{:<40} {:>8.3} {:>8.3} {:>8.3} {:>12.2}",
            &p.model_name[..p.model_name.len().min(40)],
            p.recall_5, p.recall_10, p.mrr, p.avg_embed_ms);
    }
    println!("\n[ablate-quant] Results saved → {}", args.output.display());
    Ok(())
}

// ── Command handlers ──────────────────────────────────────────────────────────

async fn cmd_ingest(args: IngestArgs) -> Result<()> {
    let server_bin = resolve_server_bin(&args.llama_server)?;
    std::fs::create_dir_all(&args.workspace_dir)
        .with_context(|| format!("Cannot create workspace dir: {}", args.workspace_dir.display()))?;

    let db_path = args.workspace_dir.join("rag.db");
    let index_dir = args.workspace_dir.join("bm25_index");

    let db = RagDb::open(&db_path).map_err(|e| anyhow::anyhow!("Failed to open RagDb: {}", e))?;
    let bm25 = BM25Index::open(&index_dir).map_err(|e| anyhow::anyhow!("Failed to open BM25 index: {}", e))?;
    let vec_index = VectorIndex::load_from_db(&db).map_err(|e| anyhow::anyhow!("Failed to load VectorIndex: {}", e))?;

    let mut server =
        EmbedServer::start(&server_bin, &args.embed_model, args.embed_port).await?;

    let timings = ingest_corpus(&args, &server, &db, &bm25, &vec_index).await?;

    // RAPTOR: build summary tree after corpus ingestion
    if args.raptor {
        if let Some(ref llm_path) = args.llm_model {
            db.create_raptor_tables().map_err(|e| anyhow::anyhow!("create_raptor_tables: {}", e))?;
            match ChatServer::start(&server_bin, llm_path, args.llm_port, 8).await {
                Ok(chat_server) => {
                    let docs = db.list_documents().unwrap_or_default();
                    println!("[ingest] Building RAPTOR trees for {} documents...", docs.len());
                    // Wrap servers in Arc so they can be shared across 4 concurrent futures.
                    // EmbedServer and ChatServer are Sync (process wrapped in Mutex).
                    let db_arc = Arc::new(&db);
                    let embed_arc = Arc::new(&server);
                    let chat_arc = Arc::new(chat_server);
                    let total_nodes: usize = stream::iter(docs)
                        .map(|doc| {
                            let db_ref = Arc::clone(&db_arc);
                            let embed_ref = Arc::clone(&embed_arc);
                            let chat_ref = Arc::clone(&chat_arc);
                            async move {
                                match build_raptor_tree_cli(*db_ref, doc.id, *embed_ref, &chat_ref).await {
                                    Ok(n) => n,
                                    Err(e) => { eprintln!("[ingest] RAPTOR skipped doc {}: {:#}", doc.id, e); 0 }
                                }
                            }
                        })
                        .buffer_unordered(8)
                        .fold(0usize, |acc, n| async move { acc + n })
                        .await;
                    chat_arc.stop();
                    println!("[ingest] RAPTOR complete: {} nodes created", total_nodes);
                }
                Err(e) => eprintln!("[ingest] WARNING: Could not start LLM server for RAPTOR: {:#}", e),
            }
        } else {
            eprintln!("[ingest] WARNING: --raptor requires --llm-model to be specified (skipped)");
        }
    }

    server.stop();

    let out = args.workspace_dir.join("ingest_timings.json");
    std::fs::write(&out, serde_json::to_string_pretty(&timings)?)
        .with_context(|| format!("Cannot write ingest timings to {}", out.display()))?;
    println!("[bench] Ingest timings saved → {}", out.display());
    Ok(())
}

async fn cmd_bench(args: BenchArgs) -> Result<()> {
    let server_bin = resolve_server_bin(&args.llama_server)?;

    let db_path = args.workspace_dir.join("rag.db");
    let index_dir = args.workspace_dir.join("bm25_index");

    let db = RagDb::open(&db_path).map_err(|e| anyhow::anyhow!("Failed to open RagDb: {}", e))?;
    let bm25 = BM25Index::open(&index_dir).map_err(|e| anyhow::anyhow!("Failed to open BM25 index: {}", e))?;
    let vec_index = VectorIndex::load_from_db(&db).map_err(|e| anyhow::anyhow!("Failed to load VectorIndex: {}", e))?;
    vec_index.rebuild_if_needed();

    let doc_count = db.document_count().unwrap_or(0);
    let vec_count = vec_index.len();
    println!(
        "[bench] Workspace: {} docs, {} vectors in index",
        doc_count, vec_count
    );

    let qa_json = std::fs::read_to_string(&args.qa_file)
        .with_context(|| format!("Cannot read QA file: {}", args.qa_file.display()))?;
    let qa_pairs: Vec<QAPair> =
        serde_json::from_str(&qa_json).context("Invalid QA file — expected JSON array")?;
    println!("[bench] Loaded {} QA pairs", qa_pairs.len());

    let top_ks: Vec<usize> = args
        .top_k
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if top_ks.is_empty() {
        bail!("--top-k must contain at least one valid integer, e.g. '5,10'");
    }

    let mut server =
        EmbedServer::start(&server_bin, &args.embed_model, args.embed_port).await?;

    // Subsample QA pairs for the recall + RAPTOR benchmarks to match --e2e-count.
    // Running the full corpus (87k pairs) takes many hours; 500 representative pairs
    // produce publication-quality Recall@k and latency numbers.
    let recall_n = args.e2e_count.min(qa_pairs.len());
    let step = if recall_n == 0 { 1 } else { qa_pairs.len().max(1) / recall_n };
    let qa_bench: Vec<QAPair> = qa_pairs
        .iter()
        .step_by(step.max(1))
        .take(recall_n)
        .cloned()
        .collect();
    println!(
        "[bench] Recall benchmark sample: {} / {} QA pairs (step {})",
        qa_bench.len(), qa_pairs.len(), step
    );

    println!("\n[bench] ── Running recall + latency benchmarks ──");
    let (recall_results, latency) =
        run_recall_bench(&qa_bench, &top_ks, &db, &bm25, &vec_index, &server).await?;

    let raptor_results = if args.raptor {
        println!("\n[bench] ── Running RAPTOR ablation ──");
        let r = run_raptor_bench(&qa_bench, &top_ks, &db, &server).await?;
        if r.is_empty() { None } else { Some(r) }
    } else {
        None
    };

    // E2E + no-RAG baseline (both share one ChatServer instance to avoid double startup cost)
    let (e2e_result, no_rag_result) = if let Some(ref llm_model) = args.llm_model {
        match ChatServer::start(&server_bin, llm_model, args.llm_port, 1).await {
            Ok(mut chat_server) => {
                // RAG-augmented E2E
                println!("\n[bench] ── Running E2E answer quality eval (with RAG) ──");
                let e2e = match run_e2e_bench(
                    &qa_pairs, args.e2e_count,
                    &db, &bm25, &vec_index, &server, &chat_server,
                ).await {
                    Ok(r) => Some(r),
                    Err(e) => { eprintln!("[bench] E2E eval failed: {:#}", e); None }
                };

                // No-RAG baseline (optional)
                let norag = if args.no_rag_baseline {
                    println!("\n[bench] ── Running no-RAG baseline eval ──");
                    match run_norag_eval(&qa_pairs, args.e2e_count, &chat_server).await {
                        Ok(r) => Some(r),
                        Err(e) => { eprintln!("[bench] No-RAG baseline failed: {:#}", e); None }
                    }
                } else {
                    None
                };

                chat_server.stop();
                (e2e, norag)
            }
            Err(e) => {
                eprintln!("[bench] Failed to start LLM server: {:#}", e);
                (None, None)
            }
        }
    } else {
        if args.no_rag_baseline {
            eprintln!("[bench] WARNING: --no-rag-baseline requires --llm-model (skipped)");
        }
        (None, None)
    };

    server.stop();

    // Load ingest timings produced during `ingest` (may be absent)
    let timing_path = args.workspace_dir.join("ingest_timings.json");
    let ingest_timing: Vec<IngestTiming> = timing_path
        .exists()
        .then(|| {
            std::fs::read_to_string(&timing_path)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
        })
        .flatten()
        .unwrap_or_default();

    // Estimate raw f32 memory for compression-ratio reporting.
    // Use actual embedding dim from the index (falls back to 768 if empty).
    let actual_dim = vec_index.embedding_dim();
    let avg_dim = if actual_dim > 0 { actual_dim as f64 } else { 768.0 };
    let raw_mb = vec_count as f64 * avg_dim * 4.0 / (1024.0 * 1024.0);
    let quant_mb = vec_index.estimated_memory_bytes() as f64 / (1024.0 * 1024.0);
    let compression = if quant_mb > 0.0 { raw_mb / quant_mb } else { 1.0 };

    let results = BenchResults {
        timestamp: chrono::Utc::now().to_rfc3339(),
        workspace_dir: args.workspace_dir.to_string_lossy().to_string(),
        document_count: doc_count,
        chunk_count: vec_count,
        recall: recall_results,
        latency_hybrid_expand: latency,
        index_stats: IndexStats {
            vector_count: vec_count,
            memory_mb: quant_mb,
            ivf_active: vec_count >= 128,
            raw_f32_estimate_mb: raw_mb,
            compression_ratio: compression,
        },
        ingest_timing,
        raptor: raptor_results,
        e2e: e2e_result.clone(),
        no_rag_baseline: no_rag_result,
        e2e_ci: e2e_result.as_ref().map(|e| compute_e2e_with_ci(e, args.bootstrap_samples, args.seed)),
    };

    let json = serde_json::to_string_pretty(&results)?;
    std::fs::write(&args.output, &json)
        .with_context(|| format!("Cannot write results to {}", args.output.display()))?;

    print_summary(&results);
    println!("\n[bench] Results saved → {}", args.output.display());
    Ok(())
}

// ── Standalone E2E eval command ──────────────────────────────────────────────

async fn cmd_eval(args: EvalArgs) -> Result<()> {
    let server_bin = resolve_server_bin(&args.llama_server)?;

    let db_path = args.workspace_dir.join("rag.db");
    let index_dir = args.workspace_dir.join("bm25_index");

    let db = RagDb::open(&db_path)
        .map_err(|e| anyhow::anyhow!("Failed to open RagDb: {}", e))?;
    let bm25 = BM25Index::open(&index_dir)
        .map_err(|e| anyhow::anyhow!("Failed to open BM25 index: {}", e))?;
    let vec_index = VectorIndex::load_from_db(&db)
        .map_err(|e| anyhow::anyhow!("Failed to load VectorIndex: {}", e))?;

    let doc_count = db.document_count().unwrap_or(0);
    let vec_count = vec_index.len();
    println!(
        "[eval] Workspace: {} docs, {} vectors in index",
        doc_count, vec_count
    );

    let qa_json = std::fs::read_to_string(&args.qa_file)
        .with_context(|| format!("Cannot read QA file: {}", args.qa_file.display()))?;
    let qa_pairs: Vec<QAPair> =
        serde_json::from_str(&qa_json).context("Invalid QA file — expected JSON array")?;
    println!("[eval] Loaded {} QA pairs", qa_pairs.len());

    let mut embed_server =
        EmbedServer::start(&server_bin, &args.embed_model, args.embed_port).await?;
    let mut chat_server =
        ChatServer::start(&server_bin, &args.llm_model, args.llm_port, 1).await?;

    println!("\n[eval] Running E2E answer quality eval ──");
    let result = run_e2e_bench(
        &qa_pairs,
        args.count,
        &db,
        &bm25,
        &vec_index,
        &embed_server,
        &chat_server,
    ).await;

    chat_server.stop();
    embed_server.stop();

    let e2e = result?;
    let e2e_ci = compute_e2e_with_ci(&e2e, args.bootstrap_samples, args.seed);
    let output_val = serde_json::json!({ "raw": &e2e, "ci": &e2e_ci });
    std::fs::write(&args.output, serde_json::to_string_pretty(&output_val)?)
        .with_context(|| format!("Cannot write results to {}", args.output.display()))?;

    println!("\n── E2E Answer Quality (hybrid retrieval) ─────────────────────────────────────");
    println!(
        "  Exact Match : {:.1}%  ({}/{} answered)",
        e2e.exact_match * 100.0,
        e2e.per_question.iter().filter(|q| q.exact_match).count(),
        e2e.sample_count
    );
    println!("  Token F1    : {:.1}%", e2e.f1 * 100.0);
    println!("  Avg Latency : {:.1} ms", e2e.avg_latency_ms);
    println!("\n── 95% Bootstrap CI (n={}, B={}) ────────────────────────────────────────────",
        e2e_ci.n, e2e_ci.bootstrap_samples);
    println!("  EM: {:.1}%  [{:.1}%, {:.1}%]",
        e2e_ci.exact_match * 100.0, e2e_ci.em_ci_low * 100.0, e2e_ci.em_ci_high * 100.0);
    println!("  F1: {:.1}%  [{:.1}%, {:.1}%]",
        e2e_ci.f1 * 100.0, e2e_ci.f1_ci_low * 100.0, e2e_ci.f1_ci_high * 100.0);
    println!("  Latency  p50={:.1}ms  p95={:.1}ms  p99={:.1}ms",
        e2e_ci.p50_latency_ms, e2e_ci.p95_latency_ms, e2e_ci.p99_latency_ms);
    println!("\n[eval] Results saved → {}", args.output.display());
    Ok(())
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Ingest(args) => cmd_ingest(args).await,
        Commands::Bench(args) => cmd_bench(args).await,
        Commands::Scale(args) => cmd_scale(args).await,
        Commands::Eval(args) => cmd_eval(args).await,
        Commands::BeirBench(args) => cmd_beir_bench(args).await,
        Commands::AblateChunking(args) => cmd_ablate_chunking(args).await,
        Commands::AblateRrfK(args) => cmd_ablate_rrf_k(args).await,
        Commands::AblateQuant(args) => cmd_ablate_quant(args).await,
        Commands::Run(args) => {
            // Decompose into ingest + bench args sharing the same workspace/model
            let ingest = IngestArgs {
                workspace_dir: args.workspace_dir.clone(),
                corpus_dir: args.corpus_dir,
                embed_model: args.embed_model.clone(),
                llama_server: args.llama_server.clone(),
                embed_port: args.embed_port,
                raptor: false,
                llm_model: None,
                llm_port: 12346,
            };
            if let Err(e) = cmd_ingest(ingest).await {
                eprintln!("[bench] Ingest phase failed: {:#}", e);
                std::process::exit(1);
            }
            let bench = BenchArgs {
                workspace_dir: args.workspace_dir,
                qa_file: args.qa_file,
                embed_model: args.embed_model,
                llama_server: args.llama_server,
                embed_port: args.embed_port,
                top_k: args.top_k,
                output: args.output,
                raptor: args.raptor,
                llm_model: args.llm_model,
                llm_port: args.llm_port,
                e2e_count: args.e2e_count,
                no_rag_baseline: args.no_rag_baseline,
                bootstrap_samples: args.bootstrap_samples,
                seed: args.seed,
            };
            cmd_bench(bench).await
        }
    };

    if let Err(e) = result {
        eprintln!("[bench] Error: {:#}", e);
        std::process::exit(1);
    }
}
