//! RAG (Retrieval-Augmented Generation) subsystem.
//!
//! Provides a fully local, on-device RAG pipeline with:
//!   - **db**       — sqlite-vec vector store + document/chunk metadata
//!   - **search**   — Tantivy BM25 full-text search (MmapDirectory)
//!   - **fusion**   — Reciprocal Rank Fusion (RRF) for hybrid retrieval
//!   - **chunker**  — Recursive character + semantic chunking
//!   - **parsers**  — Per-format document extractors (PDF, DOCX, PPTX, TXT, audio)
//!   - **pipeline** — Orchestrates ingestion (progressive) and retrieval
//!   - **raptor**   — Recursive Abstractive Processing for Tree-Organized Retrieval
//!
//! Key design:
//!   - Progressive Ingestion: Phase 1 stores raw chunks instantly,
//!     Phase 2 enriches in background via the TaskRouter.
//!   - Hybrid search: BM25 (Tantivy) + Vector KNN (sqlite-vec) merged with RRF.
//!   - RAPTOR: On-demand hierarchical summarization with confidence-aware retrieval.
//!   - All inference delegated to existing TaskRouter / ProcessManager.

pub mod db;
pub mod search;
pub mod fusion;
pub mod chunker;
pub mod parsers;
pub mod pipeline;
pub mod raptor;
pub mod vecindex;
pub mod watchman;
