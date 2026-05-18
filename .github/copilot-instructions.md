# NELA Repository Instructions

## Required .github Sync Policy

The customization files under `.github/` are part of the repository contract.

- Update `.github/copilot-instructions.md`, `.github/instructions/*.instructions.md`, and `.github/agents/*.agent.md` whenever code, architecture, workflows, or developer commands change.
- Keep `.github/agents/genhat.md` synchronized when architectural or behavior details in that reference become stale.
- If a user asks to revert a change, also revert or delete the matching updates made in these `.github` files in the same revert scope.
- Do not leave stale references to removed files, removed features, or old command paths.

## Repository Snapshot

- `genhat-desktop/` is the main desktop app (React + TypeScript + Vite + Tauri v2).
- `genhat-desktop/src/` contains frontend app logic and components.
- `genhat-desktop/src-tauri/src/` contains Rust backend modules, commands, routing, and RAG pipeline code.
- Text chat supports both document-grounding paths: KB-ingested RAG retrieval and direct file-to-prompt attachments, controlled by a RAG on/off toggle (default off = direct prompting).
- Runtime model parameters panel is hidden by default and opened explicitly by the user.
- Disk-scanned model sync preserves user-applied runtime params (for example `ctx_size`, `max_tokens`, `flash_attn`) instead of resetting them during model-list refreshes.
- `benchmark/` contains runtime benchmark scripts, plotting tools, and the RAG retrieval quality benchmark CLI.
- `benchmark/prepare_squad.py` downloads SQuAD 1.1 and produces a corpus and QA-pairs file for `rag-bench`. Each QA pair includes an `answers` array (all acceptable gold answers) for E2E evaluation.
- `scripts/` contains end-to-end benchmark orchestration:
  - `download_datasets.py` — download SQuAD, TriviaQA RC, BEIR subsets, and BGE embed models.
    - Flags: `--squad-only`, `--trivia-only`, `--beir-only`, `--models-only`
    - TriviaQA RC (Wikipedia domain) written to `benchmark/trivia_qa/` (same format as SQuAD: `corpus/*.txt` + `qa_pairs.json`)
    - NQ is available as `BeIR/nq` but its 2.68M-doc corpus is impractical; excluded from default BEIR loop.
  - `baseline_llamaindex.py` — LlamaIndex + BGE + llama-server baseline (EM + F1).
  - `baseline_chromadb.py` — ChromaDB + sentence-transformers + llama-server baseline.
  - `generate_paper_assets.py` — read all result JSONs, emit LaTeX tables + matplotlib PDFs.
  - `run_all_benchmarks.sh` — full end-to-end runner (ingest → bench × 2 datasets → BEIR → ablations → baselines → assets). Steps [0/10]–[assets]. TriviaQA steps are soft-skipped if `benchmark/trivia_qa/` is absent.
  - `requirements_benchmark.txt` — pinned Python deps for the above scripts.
- `The-Bare/` contains standalone experiments/prototypes.
- `genhat-desktop/src-tauri/src/bin/rag_bench.rs` is a standalone Rust CLI (`rag-bench`) that benchmarks the NELA RAG pipeline (recall@k, latency breakdown, IVF memory stats, RAPTOR ablation, E2E answer quality, scale degradation) without the Tauri runtime. Subcommands: `ingest`, `bench`, `run`, `scale`, `eval`, `beir-bench`, `ablate-chunking`, `ablate-rrf-k`, `ablate-quant`.
  - `ingest --raptor [--llm-model <gguf>]` — ingest corpus; optionally build RAPTOR tree.
  - `bench --e2e-count 500 --bootstrap-samples 1000 [--no-rag-baseline]` — retrieval + E2E with bootstrap 95% CIs.
  - `eval --count 500 --bootstrap-samples 1000` — standalone E2E eval outputting `{ "raw": ..., "ci": ... }`.
  - `beir-bench --beir-dir <dir>` — BEIR NDCG@10/MAP/Recall@100/MRR across bm25/vector/hybrid configs.
  - `ablate-chunking --chunk-sizes 512,1024,1536,2048 --overlaps 64,128,256` — grid chunking ablation.
  - `ablate-rrf-k --rrf-k-values 10,30,60,100,200` — RRF fusion constant sweep.
  - `ablate-quant --embed-models <path1,path2>` — per-model quantisation ablation.
  - `scale --sizes 100,500,1000,2000 [--qa-sample 500]` — scale degradation (recall/latency vs corpus size).

## Validation Commands

- Frontend build and lint: `cd genhat-desktop && npm run lint && npm run build`
- Rust compile check: `cd genhat-desktop/src-tauri && cargo check`
- RAG benchmark binary check: `cd genhat-desktop/src-tauri && cargo check --bin rag-bench`
- Build RAG benchmark binary: `cd genhat-desktop/src-tauri && cargo build --release --bin rag-bench`
- Desktop dev run: `cd genhat-desktop && npx tauri dev`

## Change Hygiene

- Keep changes minimal and scoped to the request.
- Prefer updating existing patterns in nearby code instead of introducing new conventions.
- Verify changed files with targeted checks before finalizing.
