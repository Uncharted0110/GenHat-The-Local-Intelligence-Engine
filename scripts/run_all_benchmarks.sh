#!/usr/bin/env bash
# run_all_benchmarks.sh — End-to-end NELA benchmark runner for IEEE paper submission.
#
# Pre-conditions:
#   1. llama-server binary present at: genhat-desktop/src-tauri/bin/llama-lin/llama-server
#   2. Embedding model at:  models/embedding/bge-base-en-v1.5-q8_0/bge-base-en-v1.5-q8_0.gguf
#   3. LLM model at:        models/LLM/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf
#   4. SQuAD corpus at:     benchmark/squad_bench_large/corpus/
#   5. TriviaQA corpus at:  benchmark/trivia_qa/corpus/  (optional — skipped with warning if absent)
#   6. BEIR datasets at:    benchmark/beir/              (optional — skipped with warning if absent)
#   7. rag-bench binary built: genhat-desktop/src-tauri/target/release/rag-bench
#      (auto-built via cargo if missing; cargo must be on PATH in that case)
#   8. Python deps installed: pip install -r scripts/requirements_benchmark.txt
#      (optional — baselines skipped with warning if unavailable)
#
# Usage:
#   bash scripts/run_all_benchmarks.sh [--skip-ingest] [--skip-baselines] [--server <path>]
#                                      [--ablate-max-docs <N>] [--ablate-max-qa <N>]
#
# --server <path>         Override the llama-server binary (use a CUDA build on GPU machines).
#   Example: bash scripts/run_all_benchmarks.sh --server /usr/local/bin/llama-server
#
# --ablate-max-docs <N>   Cap ablation stages (6, 8) to the first N corpus documents
#                         (sorted alphabetically). QA pairs are automatically filtered
#                         to those answerable from the subset. Reduces RAM and runtime.
#
# --ablate-max-qa <N>     Further cap QA pairs per grid point (applied after doc-title filter).
#                         When --ablate-max-docs is set without --ablate-max-qa the script
#                         defaults to 500 for fast ablations. Use 0 to disable the auto-cap.
#   Example: bash scripts/run_all_benchmarks.sh --ablate-max-docs 100 --ablate-max-qa 500
#
# Each run writes to results/<RUN_ID>/  (timestamped, never overwritten).
# A symlink results/latest → current run is maintained for convenience.
# Ingest workspaces live under workspace/  (expensive; reused across runs).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$ROOT/genhat-desktop/src-tauri/target/release/rag-bench"
SERVER="$ROOT/genhat-desktop/src-tauri/bin/llama-lin/llama-server"
EMBED="$ROOT/models/embedding/bge-base-en-v1.5-q8_0/bge-base-en-v1.5-q8_0.gguf"
LLM="$ROOT/models/LLM/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"
QA="$ROOT/benchmark/squad_bench_large/qa_pairs.json"
CORPUS="$ROOT/benchmark/squad_bench_large/corpus"
QA_TRIVIA="$ROOT/benchmark/trivia_qa/qa_pairs.json"
CORPUS_TRIVIA="$ROOT/benchmark/trivia_qa/corpus"
BEIR="$ROOT/benchmark/beir"
EMBED_SMALL="$ROOT/models/embedding/bge-small-en-v1.5-q8_0/bge-small-en-v1.5-q8_0.gguf"
# Workspaces persist across runs (ingest is expensive); live outside timestamped results.
WS="$ROOT/workspace/nela_ws"
WS_TRIVIA="$ROOT/workspace/trivia_ws"
# BEIR workspaces are expensive to re-ingest (~5k-40k docs each), so they live
# in a stable path that persists across runs and is reused when already populated.
WS_BEIR="$ROOT/workspace/beir_ws"
# Ablation workspaces also persist: each grid-point sub-dir is reused on resume
# so interrupted runs continue from the last completed config point.
WS_ABLATE_CHUNK="$ROOT/workspace/ablate_chunking_ws"
WS_ABLATE_QUANT="$ROOT/workspace/ablate_quant_ws"
# Per-run outputs land in a timestamped subdirectory — never overwritten.
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RESULTS="$ROOT/results/$RUN_ID"
SKIP_INGEST=0
SKIP_BASELINES=0
SERVER_OVERRIDE=""
# Limit ablation stages (6, 8) to first N corpus documents.
# Leave empty to use the full corpus. Example: --ablate-max-docs 100
ABLATE_MAX_DOCS="100"
# Further cap QA pairs per ablation grid point (after doc-title filter).
# When ABLATE_MAX_DOCS is set this defaults to 500 if not overridden (see below).
ABLATE_MAX_QA="500"

# ── helpers ───────────────────────────────────────────────────────────────────
die()  { echo "[ERROR] $*" >&2; exit 1; }
warn() { echo "[WARN]  $*" >&2; }
# tick <label> — log elapsed seconds since last tick to stdout + timing.txt
_TICK_T=$SECONDS
tick() { local s=$((SECONDS - _TICK_T)); _TICK_T=$SECONDS
         printf "[%s] [timing] %-38s %ds\n" "$(date +%H:%M:%S)" "$1" "$s" \
           | tee -a "$RESULTS/timing.txt"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-ingest)       SKIP_INGEST=1 ; shift ;;
    --skip-baselines)    SKIP_BASELINES=1 ; shift ;;
    --server)            SERVER_OVERRIDE="$2" ; shift 2 ;;
    --ablate-max-docs)   ABLATE_MAX_DOCS="$2" ; shift 2 ;;
    --ablate-max-qa)     ABLATE_MAX_QA="$2" ; shift 2 ;;
    *) shift ;;
  esac
done

# When --ablate-max-docs is set without an explicit --ablate-max-qa, default to 500
# so ablations stay fast (the doc-title filter already keeps pairs correct).
[[ -n "$ABLATE_MAX_DOCS" && -z "$ABLATE_MAX_QA" ]] && ABLATE_MAX_QA="500"
# Allow the user to disable the auto-cap by passing --ablate-max-qa 0.
[[ "$ABLATE_MAX_QA" == "0" ]] && ABLATE_MAX_QA=""

[[ -n "$SERVER_OVERRIDE" ]] && SERVER="$SERVER_OVERRIDE"

# ── preflight: hard fails ─────────────────────────────────────────────────────
# These components are required for every stage; abort immediately if missing.
[[ -f "$SERVER" && -x "$SERVER" ]] \
  || die "llama-server not found or not executable: $SERVER"

# Warn if the binary appears to be CPU-only (no CUDA/Vulkan in its shared-lib deps)
if command -v ldd &>/dev/null; then
  if ! ldd "$SERVER" 2>/dev/null | grep -qiE 'cuda|libcuda|vulkan|libvulkan'; then
    warn "llama-server has no CUDA/Vulkan deps — it will run on CPU only."
    warn "For GPU acceleration pass a CUDA build: --server /path/to/cuda/llama-server"
  fi
fi
[[ -f "$EMBED" ]] \
  || die "Base embedding model not found: $EMBED"
[[ -f "$LLM" ]] \
  || die "LLM model not found: $LLM  (needed for E2E + RAPTOR)"
[[ -f "$QA" ]] \
  || die "QA pairs file not found: $QA  (run scripts/download_datasets.py first)"
[[ -d "$CORPUS" ]] \
  || die "Corpus directory not found: $CORPUS  (run: cd benchmark && python3 prepare_squad.py --out-dir squad_bench_large --max-contexts 500 --max-qa 2000)"

# rag-bench binary — hard fail if missing AND cargo is unavailable
if [[ ! -f "$BENCH" ]]; then
  command -v cargo &>/dev/null \
    || die "rag-bench binary not found and cargo is not on PATH. Build first: cd genhat-desktop/src-tauri && cargo build --release --bin rag-bench"
fi

# ── preflight: soft fails (warn, set skip flags) ──────────────────────────────
SKIP_BEIR=0
SKIP_TRIVIA=0
SKIP_QUANT_SMALL=0
SKIP_PYTHON=0

[[ -f "$QA_TRIVIA" && -d "$CORPUS_TRIVIA" ]] \
  || { warn "TriviaQA data not found; TriviaQA steps will be skipped. Run: python3 scripts/download_datasets.py --trivia-only"; SKIP_TRIVIA=1; }

[[ -d "$BEIR" ]] \
  || { warn "BEIR directory not found ($BEIR); BEIR benchmark will be skipped. Run scripts/download_datasets.py --beir-only to fetch it."; SKIP_BEIR=1; }

[[ -f "$EMBED_SMALL" ]] \
  || { warn "Small embedding model not found ($EMBED_SMALL); quant ablation will use base model only."; SKIP_QUANT_SMALL=1; }

if [[ "$SKIP_BASELINES" -eq 0 ]]; then
  python3 -c "import llama_index, chromadb, sentence_transformers" 2>/dev/null \
    || { warn "Python baseline deps not importable (llama_index / chromadb / sentence_transformers). Run: pip install -r scripts/requirements_benchmark.txt  Baselines will be skipped."; SKIP_PYTHON=1; }
fi

mkdir -p "$RESULTS" "$ROOT/workspace"
# Symlink results/latest → this run so scripts/tools can always find the newest output
ln -sfn "$RESULTS" "$ROOT/results/latest"

# ── Run provenance ────────────────────────────────────────────────────────────
GIT_HASH="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")"
echo "$GIT_HASH" > "$RESULTS/git_hash.txt"
cat > "$RESULTS/run_metadata.json" <<_META
{
  "run_id": "$RUN_ID",
  "git_hash": "$GIT_HASH",
  "args": "$*",
  "host": "$(hostname)"
}
_META

echo "================================================================"
echo "  NELA Benchmark Suite — IEEE Paper Edition"
echo "  Run ID : $RUN_ID"
echo "  Git    : $GIT_HASH"
echo "  Results: $RESULTS"
echo "================================================================"

# ── 0. Build rag-bench if needed ─────────────────────────────────────────────
if [[ ! -f "$BENCH" ]]; then
  echo "[0/10] Building rag-bench release binary …"
  cd "$ROOT/genhat-desktop/src-tauri"
  cargo build --release --bin rag-bench \
    || die "cargo build --release --bin rag-bench failed"
  cd "$ROOT"
fi
tick "0: build"

# ── 1. Ingest SQuAD corpus ────────────────────────────────────────────────────
if [[ "$SKIP_INGEST" -eq 0 ]]; then
  echo "[1/10] Ingesting SQuAD corpus with RAPTOR …"
  "$BENCH" ingest \
    --workspace-dir "$WS" \
    --corpus-dir "$CORPUS" \
    --embed-model "$EMBED" \
    --llama-server "$SERVER" \
    --raptor \
    --llm-model "$LLM"
else
  echo "[1/10] Skipping SQuAD ingest (--skip-ingest)"
fi
tick "1: SQuAD ingest"

# ── 2. Ingest TriviaQA corpus ─────────────────────────────────────────────────
if [[ "$SKIP_TRIVIA" -eq 1 ]]; then
  echo "[2/10] Skipping TriviaQA ingest (data not found)"
elif [[ "$SKIP_INGEST" -eq 0 ]]; then
  echo "[2/10] Ingesting TriviaQA corpus with RAPTOR …"
  "$BENCH" ingest \
    --workspace-dir "$WS_TRIVIA" \
    --corpus-dir "$CORPUS_TRIVIA" \
    --embed-model "$EMBED" \
    --llama-server "$SERVER" \
    --raptor \
    --llm-model "$LLM"
else
  echo "[2/10] Skipping TriviaQA ingest (--skip-ingest)"
fi
tick "2: TriviaQA ingest"

# ── 3. SQuAD bench (retrieval + E2E + bootstrap CI) ──────────────────────────
echo "[3/10] SQuAD bench (n=500, B=1000) …"
"$BENCH" bench \
  --workspace-dir "$WS" \
  --qa-file "$QA" \
  --embed-model "$EMBED" \
  --llama-server "$SERVER" \
  --llm-model "$LLM" \
  --raptor \
  --top-k "1,5,10,20" \
  --e2e-count 500 \
  --bootstrap-samples 1000 \
  --seed 42 \
  --no-rag-baseline \
  --output "$RESULTS/bench_results.json"
tick "3: SQuAD bench"

# ── 4. TriviaQA bench (retrieval + E2E + bootstrap CI) ───────────────────────
if [[ "$SKIP_TRIVIA" -eq 1 ]]; then
  echo "[4/10] Skipping TriviaQA bench (data not found)"
else
  echo "[4/10] TriviaQA bench (n=500, B=1000) …"
  "$BENCH" bench \
    --workspace-dir "$WS_TRIVIA" \
    --qa-file "$QA_TRIVIA" \
    --embed-model "$EMBED" \
    --llama-server "$SERVER" \
    --llm-model "$LLM" \
    --raptor \
    --top-k "1,5,10,20" \
    --e2e-count 500 \
    --bootstrap-samples 1000 \
    --seed 42 \
    --no-rag-baseline \
    --output "$RESULTS/trivia_bench_results.json" \
    || warn "TriviaQA bench failed; continuing"
fi
tick "4: TriviaQA bench"

# ── 5. BEIR evaluation ────────────────────────────────────────────────────────
# Note: NQ (Natural Questions) is available as BeIR/nq but its 2.68M-doc corpus
#       is impractical to ingest; it is excluded from the default BEIR loop.
if [[ "$SKIP_BEIR" -eq 1 ]]; then
  echo "[5/10] Skipping BEIR benchmark (directory not found)"
else
  echo "[5/10] Running BEIR benchmark (scifact / nfcorpus / fiqa) …"
  for DATASET in scifact nfcorpus fiqa; do
    DS_DIR="$BEIR/$DATASET"
    if [[ -d "$DS_DIR" ]]; then
      echo "  → $DATASET"
      "$BENCH" beir-bench \
        --workspace-dir "$WS_BEIR/${DATASET}_ws" \
        --beir-dir "$DS_DIR" \
        --embed-model "$EMBED" \
        --llama-server "$SERVER" \
        --output "$RESULTS/beir_${DATASET}.json" \
        || warn "BEIR $DATASET benchmark failed; continuing"
    else
      warn "BEIR subset '$DATASET' not found at $DS_DIR; skipping"
    fi
  done
fi
tick "5: BEIR bench"

# ── 6. Chunking ablation ──────────────────────────────────────────────────────
echo "[6/10] Chunking ablation …"
"$BENCH" ablate-chunking \
  --workspace-dir "$WS_ABLATE_CHUNK" \
  --corpus-dir "$CORPUS" \
  --qa-file "$QA" \
  --embed-model "$EMBED" \
  --llama-server "$SERVER" \
  --chunk-sizes "512,1024,1536,2048" \
  --overlaps "64,128,256" \
  ${ABLATE_MAX_DOCS:+--max-docs $ABLATE_MAX_DOCS} \
  ${ABLATE_MAX_QA:+--max-qa $ABLATE_MAX_QA} \
  --output "$RESULTS/chunking_ablation.json"
tick "6: chunking ablation"

# ── 7. RRF-k ablation ────────────────────────────────────────────────────────
echo "[7/10] RRF-k ablation …"
"$BENCH" ablate-rrf-k \
  --workspace-dir "$WS" \
  --qa-file "$QA" \
  --embed-model "$EMBED" \
  --llama-server "$SERVER" \
  --rrf-k-values "10,30,60,100,200" \
  ${ABLATE_MAX_QA:+--max-qa $ABLATE_MAX_QA} \
  --output "$RESULTS/rrf_k_ablation.json"
tick "7: RRF-k ablation"

# ── 8. Quantisation ablation ──────────────────────────────────────────────────
echo "[8/10] Quantisation ablation …"
if [[ "$SKIP_QUANT_SMALL" -eq 1 ]]; then
  QUANT_MODELS="$EMBED"
  warn "Running quant ablation with base model only (small model missing)"
else
  QUANT_MODELS="$EMBED,$EMBED_SMALL"
fi
"$BENCH" ablate-quant \
  --workspace-dir "$WS_ABLATE_QUANT" \
  --corpus-dir "$CORPUS" \
  --qa-file "$QA" \
  --embed-models "$QUANT_MODELS" \
  --llama-server "$SERVER" \
  ${ABLATE_MAX_DOCS:+--max-docs $ABLATE_MAX_DOCS} \
  ${ABLATE_MAX_QA:+--max-qa $ABLATE_MAX_QA} \
  --output "$RESULTS/quant_ablation.json"
tick "8: quant ablation"

# ── 9. Scale degradation ─────────────────────────────────────────────────────
echo "[9/10] Scale degradation …"
"$BENCH" scale \
  --workspace-dir "$RESULTS/scale_ws" \
  --corpus-dir "$CORPUS" \
  --qa-file "$QA" \
  --embed-model "$EMBED" \
  --llama-server "$SERVER" \
  --sizes "50,100,200,442" \
  --qa-sample 500 \
  --output "$RESULTS/scale_results.json"
tick "9: scale degradation"

# ── 10. Python baselines ─────────────────────────────────────────────────────
if [[ "$SKIP_BASELINES" -eq 1 || "$SKIP_PYTHON" -eq 1 ]]; then
  echo "[10/10] Skipping Python baselines"
else
  echo "[10/10] Python baselines — starting LLM server on port 12346 …"

  # rag-bench manages its own server lifecycle and stops it after each stage.
  # The baselines need their own server instance for the duration of stage 10.
  BASELINE_PORT=12346
  LIB_DIR="$(dirname "$SERVER")"
  LD_LIBRARY_PATH="${LIB_DIR}:${LD_LIBRARY_PATH:-}" "$SERVER" \
    --model "$LLM" \
    --port "$BASELINE_PORT" \
    --ctx-size 4096 \
    --n-gpu-layers 99 \
    --no-warmup \
    --log-disable &
  BASELINE_LLM_PID=$!

  echo "  Waiting for LLM server (up to 90s) …"
  BASELINE_HEALTHY=0
  for _i in $(seq 1 90); do
    if curl -sf "http://localhost:${BASELINE_PORT}/health" > /dev/null 2>&1; then
      BASELINE_HEALTHY=1; break
    fi
    sleep 1
  done

  if [[ "$BASELINE_HEALTHY" -eq 0 ]]; then
    warn "LLM server did not become healthy in 90s — skipping Python baselines"
    kill "$BASELINE_LLM_PID" 2>/dev/null || true
  else
    LLM_URL="http://localhost:${BASELINE_PORT}/v1"
    echo "  LLM server ready at $LLM_URL"

    echo "  → LlamaIndex baseline"
    python3 "$ROOT/scripts/baseline_llamaindex.py" \
      --corpus-dir "$CORPUS" \
      --qa-file "$QA" \
      --llm-url "$LLM_URL" \
      --count 500 \
      --output "$RESULTS/llamaindex_baseline.json" \
      || warn "LlamaIndex baseline failed; continuing"

    echo "  → ChromaDB baseline"
    python3 "$ROOT/scripts/baseline_chromadb.py" \
      --corpus-dir "$CORPUS" \
      --qa-file "$QA" \
      --llm-url "$LLM_URL" \
      --count 500 \
      --output "$RESULTS/chromadb_baseline.json" \
      || warn "ChromaDB baseline failed; continuing"

    kill "$BASELINE_LLM_PID" 2>/dev/null || true
    wait "$BASELINE_LLM_PID" 2>/dev/null || true
  fi
fi
tick "10: Python baselines"

# ── Paper assets ──────────────────────────────────────────────────────────────
echo "[assets] Generating paper figures and LaTeX tables …"
python3 "$ROOT/scripts/generate_paper_assets.py" --results-dir "$RESULTS" \
  || warn "Paper asset generation failed; check Python deps (matplotlib, numpy)"

tick "assets"
echo ""
echo "================================================================"
echo "  All benchmarks complete."
echo "  Run ID  : $RUN_ID"
echo "  Results : $RESULTS/"
echo "  Latest  : $ROOT/results/latest"
echo "  Timing  : $RESULTS/timing.txt"
echo "  Paper   : $RESULTS/paper/"
echo "  Total   : ${SECONDS}s"
echo "================================================================"
