# NELA Benchmark Suite

This folder contains two independent benchmark tools:

1. **`rag-bench`** — Rust CLI that measures RAG pipeline retrieval quality and latency
   (recall@k, latency breakdown, IVF memory stats, RAPTOR ablation)
2. **`run_benchmark.py`** — Python application-level benchmark (startup timing, CPU/RAM over time)

---

## rag-bench — RAG Retrieval Quality Benchmarks

`rag-bench` is a standalone Rust binary that directly exercises the NELA RAG components
(`RagDb`, `BM25Index`, `VectorIndex`, `rrf_fuse`) without the Tauri runtime.
It is intended for researchers validating retrieval quality claims.

### Scenarios covered

| ID  | Scenario                       | Configs measured                                    |
|-----|--------------------------------|-----------------------------------------------------|
| A1  | Recall@k                       | BM25-only, Vector-only, Hybrid, Hybrid+expand       |
| D1  | Query latency breakdown        | Embed / BM25 / Vector / RRF / Expand per stage      |
| B1  | Ingestion timing               | Per-file: chunk count, embed time, total time       |
| C2  | IVF memory efficiency          | Quantized MB vs raw f32 estimate, compression ratio |
| A4  | RAPTOR confidence-gate ablation| threshold=-1.5 vs +∞ vs -∞ (requires pre-built trees) |

### Step 1 — Prepare a test corpus (SQuAD)

```bash
cd benchmark
python3 prepare_squad.py \
  --out-dir    squad_bench \
  --max-contexts 100 \
  --max-qa     400
```

This downloads SQuAD 1.1 dev set (~10 MB), writes 100 `.txt` context files to
`squad_bench/docs/`, and produces `squad_bench/qa_pairs.json` with 400 QA pairs.

### Step 2 — Run the full benchmark

```bash
cd genhat-desktop

cargo run --release --bin rag-bench -- run \
  --workspace-dir /tmp/nela-bench \
  --corpus-dir    ../benchmark/squad_bench/docs \
  --qa-file       ../benchmark/squad_bench/qa_pairs.json \
  --embed-model   /path/to/bge-base-en-v1.5-q8_0.gguf \
  --top-k         5,10 \
  --output        ../benchmark/rag_results.json
```

`llama-server` is auto-detected from `src-tauri/bin/llama-lin/llama-server`.
Override with `--llama-server <path>` if needed.

### Step 2 (alternative) — Ingest once, bench many times

```bash
# Ingest corpus once
cargo run --release --bin rag-bench -- ingest \
  --workspace-dir /tmp/nela-bench \
  --corpus-dir    ../benchmark/squad_bench/docs \
  --embed-model   /path/to/bge.gguf

# Run benchmarks against the same workspace (fast, no re-embedding)
cargo run --release --bin rag-bench -- bench \
  --workspace-dir /tmp/nela-bench \
  --qa-file       ../benchmark/squad_bench/qa_pairs.json \
  --embed-model   /path/to/bge.gguf \
  --output        ../benchmark/rag_results.json
```

### RAPTOR ablation (A4)

RAPTOR trees must be pre-built.  The easiest way is to ingest the corpus via the NELA
desktop app (Phase 2 background enrichment auto-builds RAPTOR), then point
`--workspace-dir` at that existing NELA workspace:

```bash
cargo run --release --bin rag-bench -- bench \
  --workspace-dir ~/.local/share/nela/workspaces/<your-workspace> \
  --qa-file       ../benchmark/squad_bench/qa_pairs.json \
  --embed-model   /path/to/bge.gguf \
  --raptor \
  --output        ../benchmark/rag_raptor_results.json
```

### QA pairs format

```json
[
  {
    "question": "What causes the tides?",
    "relevant_keywords": ["gravitational", "Moon", "tidal"],
    "doc_title": "optional_partial_doc_title"
  }
]
```

The oracle marks a chunk as relevant if it contains **any** keyword (case-insensitive)
and, if `doc_title` is set, if the chunk's document title contains that substring.

### Plotting results

```bash
python3 benchmark/plot_results.py --input benchmark/rag_results.json
```

---

## run_benchmark.py — Application-Level Benchmark

- Startup timing (cold start)
- Process-tree resource use over time (CPU, RSS, process count)
- Extended collector metrics (best-effort; full Linux `/proc` support, graceful fallback on Windows/macOS)
- Per-model load time + memory deltas (parsed from runtime logs)
- Disk footprint (models + app binary)
- Graceful shutdown timing (launch mode)
- Aggregated series statistics (min/max/mean/median/p95/p99/stddev)

The suite generates both PNG charts and interactive HTML charts.

---

## 1) Setup

From repository root:

```bash
python3 -m venv .venv-benchmark
source .venv-benchmark/bin/activate
pip install -r benchmark/requirements.txt
```

Optional Linux tools (recommended for deeper validation):

```bash
sudo apt-get install -y smem sysstat psmisc procps
```

- `smem` for extra memory validation
- `pidstat` (from `sysstat`) for CPU profiling
- `pstree` (from `psmisc`) for process tree snapshots

---

## 2) Quick Start (Launch Mode)

Launches GenHat and benchmarks from startup.

```bash
python3 benchmark/run_benchmark.py \
  --repo-root . \
  --mode launch \
  --launch-cmd "cd genhat-desktop && npx tauri dev" \
  --profile standard \
  --interactive \
  --shutdown-after-benchmark
```

### Run until the app exits

```bash
python3 benchmark/run_benchmark.py \
  --repo-root . \
  --mode launch \
  --launch-cmd "cd genhat-desktop && npx tauri dev" \
  --profile standard \
  --run-until-exit
```

### Fixed duration mode

```bash
python3 benchmark/run_benchmark.py \
  --repo-root . \
  --mode launch \
  --launch-cmd "cd genhat-desktop && npx tauri dev" \
  --profile quick \
  --duration-s 180
```

---

## 3) Attach Mode

Attach mode is useful when you launch the app in a separate known-good terminal and only want benchmark observation:

```bash
python3 benchmark/run_benchmark.py \
  --repo-root . \
  --mode attach \
  --attach-name app \
  --profile standard
```

If name matching fails, use a PID:

```bash
python3 benchmark/run_benchmark.py \
  --repo-root . \
  --mode attach \
  --attach-pid <PID>
```

For better model event parsing in attach mode, provide a live tauri log file:

```bash
python3 benchmark/run_benchmark.py \
  --repo-root . \
  --mode attach \
  --attach-pid <PID> \
  --tauri-log-file /path/to/tauri.log
```

---

## 4) Profiles

`--profile` sets default timing/sampling values unless explicitly overridden.

- `quick`: short check, faster turnaround
- `standard`: balanced default for day-to-day benchmarking
- `long`: denser sampling and longer steady-state windows

You can still override any profile default directly:

- `--sample-interval-s`
- `--extended-sample-interval-s`
- `--idle-window-s`
- `--model-load-window-s`

---

## 5) Outputs

Each run creates a timestamped folder in `benchmark/results/<timestamp>/`:

- `metrics.json`: core metrics, capabilities, and aggregate stats
- `events.json`: structured event timeline parsed from logs
- `samples.csv`: time-series core samples (`rss_mb`, `cpu_percent`, `cpu_user_percent`, `cpu_system_percent`, `cpu_percent_normalized`, `process_count`)
- `extended_samples.csv`: extended samples (collector-dependent)
- `model_metrics.csv`: per-model load time + memory delta
- `percentile_metrics.csv`: per-series window stats (`min/max/mean/median/p95/p99/stddev`)
- `tauri_runtime.log`: captured runtime logs
- `plots/` PNG outputs:
  - `rss_over_time.png`
  - `cpu_over_time.png`
  - `process_count_over_time.png`
  - `memory_breakdown_over_time.png`
  - `io_rates_over_time.png`
  - `fault_rates_over_time.png`
  - `threads_fds_over_time.png`
  - `llama_server_count_over_time.png` (backend process telemetry)
  - `model_load_time.png`
  - `model_memory_delta.png`
  - `summary_metrics.png`
- `plots/` HTML outputs (interactive):
  - `rss_over_time.html`
  - `cpu_over_time.html`
  - `process_count_over_time.html`
  - `memory_breakdown_over_time.html`
  - `io_rates_over_time.html`
  - `fault_rates_over_time.html`
  - `threads_fds_over_time.html`
  - `backend_process_count_over_time.html`
  - `model_load_time.html`
  - `model_memory_delta.html`
  - `summary_metrics.html`
  - `dashboard.html`

---

## 6) Metrics Coverage

### Timing

- Cold start time: launch timestamp to readiness marker/regex
- Shutdown time (launch mode): `SIGTERM` to process tree exit

### Process tree time series

- CPU% over time
- CPU user/system split over time
- CPU normalized by logical core count
- RSS MB over time
- Process count over time

### Extended collectors (best-effort)

Linux provides full `/proc` collection where available:

- PSS/USS/shared memory
- I/O bytes and rates
- Minor/major faults and rates
- Voluntary/involuntary context switches
- Open file descriptor counts

Windows and macOS runs still collect core process telemetry and write explicit capability flags so missing collectors do not fail the benchmark.

### Per-model metrics

- Model spawn to ready load time
- RSS delta (ready minus spawn)

### Aggregate statistics

For `rss_mb`, `cpu_percent`, and `process_count`, both full-run and idle-window stats include:

- `min`
- `max`
- `mean`
- `median`
- `p95`
- `p99`
- `stddev`

---

## 7) Useful Flags

- `--profile quick|standard|long`
- `--duration-s <seconds>`
- `--interactive`
- `--run-until-exit`
- `--sample-interval-s` / `--extended-sample-interval-s`
- `--no-smaps-rollup`, `--no-proc-io`, `--no-proc-faults`, `--no-proc-fds`, `--no-proc-ctx-switches`
- `--sanitize-launch-env`

---

## 8) Notes

- This folder now benchmarks application behavior only.
- If `[BENCH]` markers exist in logs, they are used when present, but they are not required.
- If optional Linux tools are not installed, the benchmark still runs.
