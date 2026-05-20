#!/usr/bin/env python3
"""
Generate all publication figures for the NELA paper.
Outputs PNG files to conference_latex_template/figures/

Usage:
    cd /path/to/nela
    python3 scripts/generate_figures.py

Verified field names from results/*.json:
  bench_results.json:   recall[].config, recall_5/10/20, mrr, avg_latency_ms
                        latency_hybrid_expand: embed_ms, bm25_ms, vector_ms, rrf_ms, expand_ms, total_ms
                        e2e: exact_match, f1, avg_latency_ms; per_question[].latency_ms
                        e2e_ci: em_ci_low/high, f1_ci_low/high, p50/p95/p99_latency_ms
                        raptor[].config / recall / avg_latency_ms
  scale_results.json:   points[]: doc_count, recall_5_hybrid, recall_5_bm25, recall_5_vector,
                                   avg_latency_ms_hybrid, memory_mb
  chunking_ablation.json: list of {chunk_size, overlap, recall_5, recall_10, mrr, avg_query_ms}
  rrf_k_ablation.json:  list of {rrf_k (float), recall_5, recall_10, mrr, avg_latency_ms}
  quant_ablation.json:  list of {model_name, recall_5, recall_10, mrr, avg_embed_ms, avg_query_ms}
  beir_*.json:          results[]: {config, ndcg_at_10, map, recall_at_100, mrr, avg_latency_ms}
  trivia_bench_results.json: same shape as bench_results but no hybrid recall config and no
                              raptor_expand_all
  llamaindex_baseline.json: {exact_match, f1, avg_latency_ms, per_question[].latency_ms}
  chromadb_baseline.json:   same shape
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "conference_latex_template" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

def load(name):
    return json.loads((RESULTS / name).read_text())

# ── IEEE-style aesthetics ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

BLUE   = "#2271B2"
GREEN  = "#3DB561"
ORANGE = "#E69F00"
RED    = "#D55E00"
PURPLE = "#7B2D8B"
GREY   = "#999999"

# ──────────────────────────────────────────────────────────────────────────
# 1. fig_retrieval.png  –  Retrieval performance grouped bar
# ──────────────────────────────────────────────────────────────────────────
def fig_retrieval():
    bench = load("bench_results.json")
    configs = {r["config"]: r for r in bench["recall"]}

    labels  = ["BM25-only", "Vector-only", "Hybrid", "Hybrid+Expand"]
    keys    = ["bm25_only", "vector_only", "hybrid", "hybrid_expand"]
    metrics = ["recall@5", "recall@10", "mrr"]
    mnames  = ["Recall@5", "Recall@10", "MRR"]
    colors  = [BLUE, GREEN, ORANGE]

    x = np.arange(len(labels))
    w = 0.25
    offsets = [-w, 0, w]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for i, (metric, mname, color) in enumerate(zip(metrics, mnames, colors)):
        vals = []
        for k in keys:
            if metric == "mrr":
                vals.append(configs[k]["mrr"])
            else:
                vals.append(configs[k]["recall"][metric])
        bars = ax.bar(x + offsets[i], vals, w, label=mname, color=color,
                      alpha=0.9, linewidth=0.4, edgecolor="white")
        # annotate only Hybrid bar
        idx = 2  # hybrid
        ax.text(x[idx] + offsets[i], vals[idx] + 0.004,
                f"{vals[idx]:.3f}", ha="center", va="bottom",
                fontsize=6.5, fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0.62, 1.0)
    ax.set_title("Retrieval Performance on SQuAD 1.1\n(442 docs, 14 027 chunks, n=500)")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_retrieval.png")
    plt.close(fig)
    print("  ✓ fig_retrieval.png")


# ──────────────────────────────────────────────────────────────────────────
# 2. fig_latency.png  –  Query latency breakdown (stacked + E2E)
# ──────────────────────────────────────────────────────────────────────────
def fig_latency():
    bench = load("bench_results.json")
    lb = bench["latency_hybrid_expand"]

    # retrieval pipeline
    stages  = ["Embed", "BM25", "Vector ANN", "RRF fusion", "Expand"]
    vals_ms = [lb["embed_ms"], lb["bm25_ms"], lb["vector_ms"],
               lb["rrf_ms"],   lb["expand_ms"]]
    colors  = [RED, BLUE, GREEN, ORANGE, PURPLE]
    total_r = lb["total_ms"]

    # E2E breakdown
    e2e_avg  = bench["e2e"]["avg_latency_ms"]
    llm_ms   = e2e_avg - total_r

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.4))

    # left: retrieval breakdown
    ax = axes[0]
    left = 0
    for s, v, c in zip(stages, vals_ms, colors):
        pct = 100 * v / total_r
        ax.barh(0, v, left=left, height=0.55, color=c, label=s)
        if pct > 4:
            ax.text(left + v/2, 0, f"{v:.2f}\n({pct:.0f}%)",
                    ha="center", va="center", fontsize=6.5, color="white",
                    fontweight="bold")
        left += v
    ax.set_xlim(0, total_r * 1.08)
    ax.set_yticks([])
    ax.set_xlabel("Latency (ms)")
    ax.set_title(f"Retrieval pipeline\n(total={total_r:.2f} ms)", fontsize=8.5)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.85)
    ax.set_ylim(-0.5, 0.5)

    # right: E2E breakdown (retrieval vs LLM)
    ax2 = axes[1]
    e2e_parts = [total_r, llm_ms]
    e2e_cols  = [BLUE, ORANGE]
    e2e_lbls  = [f"Retrieval\n{total_r:.1f} ms", f"LLM gen\n{llm_ms:.1f} ms"]
    left = 0
    for v, c, lbl in zip(e2e_parts, e2e_cols, e2e_lbls):
        ax2.barh(0, v, left=left, height=0.55, color=c)
        ax2.text(left + v/2, 0, lbl, ha="center", va="center",
                 fontsize=7, color="white", fontweight="bold")
        left += v
    ax2.set_xlim(0, e2e_avg * 1.08)
    ax2.set_yticks([])
    ax2.set_xlabel("Latency (ms)")
    ax2.set_title(f"E2E latency\n(p50={bench['e2e_ci']['p50_latency_ms']:.1f} ms)", fontsize=8.5)

    fig.suptitle("Query Latency Breakdown — NELA Hybrid+Expand", fontsize=9.5, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig_latency.png")
    plt.close(fig)
    print("  ✓ fig_latency.png")


# ──────────────────────────────────────────────────────────────────────────
# 3. fig_e2e.png  –  E2E answer quality 4-system bar (EM + F1)
# ──────────────────────────────────────────────────────────────────────────
def fig_e2e():
    bench   = load("bench_results.json")
    llama   = load("llamaindex_baseline.json")
    chroma  = load("chromadb_baseline.json")

    systems = ["No-RAG\n(Qwen3.5-0.8B)", "ChromaDB\n+BGE", "LlamaIndex\n+BGE", "NELA\nHybrid RAG"]
    em_vals = [
        bench["no_rag_baseline"]["exact_match"] * 100,
        chroma["exact_match"] * 100,
        llama["exact_match"] * 100,
        bench["e2e"]["exact_match"] * 100,
    ]
    f1_vals = [
        bench["no_rag_baseline"]["f1"] * 100,
        chroma["f1"] * 100,
        llama["f1"] * 100,
        bench["e2e"]["f1"] * 100,
    ]

    ci = bench["e2e_ci"]
    em_lo = (ci["exact_match"] - ci["em_ci_low"]) * 100
    em_hi = (ci["em_ci_high"] - ci["exact_match"]) * 100
    f1_lo = (ci["f1"] - ci["f1_ci_low"]) * 100
    f1_hi = (ci["f1_ci_high"] - ci["f1"]) * 100

    x = np.arange(len(systems))
    w = 0.35

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    b1 = ax.bar(x - w/2, em_vals, w, color=BLUE,  alpha=0.9, label="Exact Match (EM)", linewidth=0)
    b2 = ax.bar(x + w/2, f1_vals, w, color=GREEN, alpha=0.9, label="Token F1",         linewidth=0)

    # CI error bars on NELA only
    ax.errorbar(x[-1] - w/2, em_vals[-1], yerr=[[em_lo],[em_hi]],
                fmt="none", ecolor="black", capsize=4, linewidth=1.2)
    ax.errorbar(x[-1] + w/2, f1_vals[-1], yerr=[[f1_lo],[f1_hi]],
                fmt="none", ecolor="black", capsize=4, linewidth=1.2)

    # annotate NELA
    ax.text(x[-1] - w/2, em_vals[-1] + 1.5, f"{em_vals[-1]:.1f}%",
            ha="center", fontsize=7.5, fontweight="bold", color=BLUE)
    ax.text(x[-1] + w/2, f1_vals[-1] + 1.5, f"{f1_vals[-1]:.1f}%",
            ha="center", fontsize=7.5, fontweight="bold", color=GREEN)

    # annotate others
    for i in range(len(systems)-1):
        ax.text(x[i] - w/2, em_vals[i] + 0.8, f"{em_vals[i]:.1f}%", ha="center", fontsize=6.5, color=BLUE)
        ax.text(x[i] + w/2, f1_vals[i] + 0.8, f"{f1_vals[i]:.1f}%", ha="center", fontsize=6.5, color=GREEN)

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=7.5)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 80)
    ax.set_title(f"End-to-End Answer Quality — SQuAD 1.1 ($n$=500)\n95% CIs on NELA via bootstrap")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_e2e.png")
    plt.close(fig)
    print("  ✓ fig_e2e.png")


# ──────────────────────────────────────────────────────────────────────────
# 4. fig_e2e_latency_dist.png  –  E2E latency CDF (from per-question data)
# ──────────────────────────────────────────────────────────────────────────
def fig_e2e_latency_dist():
    bench  = load("bench_results.json")
    llama  = load("llamaindex_baseline.json")
    chroma = load("chromadb_baseline.json")

    nela_lats   = sorted([q["latency_ms"] for q in bench["e2e"]["per_question"]])
    norag_lats  = sorted([q["latency_ms"] for q in bench["no_rag_baseline"]["per_question"]])
    llama_lats  = sorted([q["latency_ms"] for q in llama["per_question"]])
    chroma_lats = sorted([q["latency_ms"] for q in chroma["per_question"]])

    def cdf_xy(data):
        n = len(data)
        return data, [(i+1)/n for i in range(n)]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for lats, label, color, ls in [
        (nela_lats,   "NELA",       BLUE,   "-"),
        (norag_lats,  "No-RAG",     GREY,   ":"),
        (llama_lats,  "LlamaIndex", ORANGE, "--"),
        (chroma_lats, "ChromaDB",   RED,    "-."),
    ]:
        xs, ys = cdf_xy(lats)
        ax.plot(xs, ys, color=color, linestyle=ls, linewidth=1.5, label=label)

    ci = bench["e2e_ci"]
    ax.axvline(ci["p50_latency_ms"], color=BLUE, linewidth=0.8, linestyle=":")
    ax.axvline(ci["p95_latency_ms"], color=BLUE, linewidth=0.8, linestyle=":")
    ax.text(ci["p50_latency_ms"]+1, 0.48, "p50", fontsize=6.5, color=BLUE)
    ax.text(ci["p95_latency_ms"]+1, 0.93, "p95", fontsize=6.5, color=BLUE)

    ax.set_xlabel("E2E Latency (ms)")
    ax.set_ylabel("CDF")
    # clip at 2× NELA p99 to avoid chromadb tail dominating the x-axis
    ax.set_xlim(0, bench["e2e_ci"]["p99_latency_ms"] * 3.0)
    ax.set_ylim(0, 1.02)
    ax.set_title("E2E Latency CDF — All Systems ($n$=500)")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_e2e_latency_dist.png")
    plt.close(fig)
    print("  ✓ fig_e2e_latency_dist.png")


# ──────────────────────────────────────────────────────────────────────────
# 5. fig_scale.png  –  Scale degradation (2-panel)
# ──────────────────────────────────────────────────────────────────────────
def fig_scale():
    sc = load("scale_results.json")
    pts = sc["points"]
    docs  = [p["doc_count"] for p in pts]
    r5_h  = [p["recall_5_hybrid"] for p in pts]
    r5_b  = [p["recall_5_bm25"]   for p in pts]
    r5_v  = [p["recall_5_vector"] for p in pts]
    lats  = [p["avg_latency_ms_hybrid"] for p in pts]
    mems  = [p["memory_mb"] for p in pts]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.0), sharex=True)

    ax1.plot(docs, r5_h, "o-", color=GREEN,  linewidth=1.8, markersize=5, label="Hybrid")
    ax1.plot(docs, r5_b, "s--", color=BLUE,  linewidth=1.2, markersize=4, label="BM25-only")
    ax1.plot(docs, r5_v, "^--", color=ORANGE,linewidth=1.2, markersize=4, label="Vector-only")
    for x, y in zip(docs, r5_h):
        ax1.text(x, y+0.002, f"{y:.3f}", ha="center", fontsize=6.5, color=GREEN, fontweight="bold")
    ax1.set_ylabel("Recall@5")
    ax1.set_ylim(0.82, 1.0)
    ax1.legend(loc="lower left", fontsize=7)
    ax1.set_title("Scale Degradation — SQuAD 1.1")

    ax2_lat = ax2
    ax2_mem = ax2.twinx()
    l1, = ax2_lat.plot(docs, lats, "o-", color=RED,  linewidth=1.8, markersize=5, label="Latency (ms)")
    l2, = ax2_mem.plot(docs, mems, "D--", color=BLUE, linewidth=1.2, markersize=4, label="IVF Mem (MB)")
    for x, y in zip(docs, lats):
        ax2_lat.text(x, y+0.05, f"{y:.2f}", ha="center", fontsize=6, color=RED)
    ax2_lat.set_ylabel("Avg. Query Latency (ms)", color=RED)
    ax2_mem.set_ylabel("IVF Index Memory (MB)",   color=BLUE)
    ax2_lat.tick_params(axis="y", labelcolor=RED)
    ax2_mem.tick_params(axis="y", labelcolor=BLUE)
    ax2_lat.set_xlabel("Corpus Size (documents)")
    ax2_lat.set_xticks(docs)
    ax2_lat.legend(handles=[l1, l2], loc="upper left", fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT / "fig_scale.png")
    plt.close(fig)
    print("  ✓ fig_scale.png")


# ──────────────────────────────────────────────────────────────────────────
# 6. fig_beir.png  –  BEIR zero-shot NDCG@10 grouped bar
# ──────────────────────────────────────────────────────────────────────────
def fig_beir():
    datasets = ["SciFact", "NFCorpus", "FiQA-2018"]
    files    = ["beir_scifact.json", "beir_nfcorpus.json", "beir_fiqa.json"]
    cfgs     = ["bm25_only", "vector_only", "hybrid"]
    cfg_lbls = ["BM25-only", "Vector-only", "Hybrid"]
    colors   = [BLUE, GREEN, ORANGE]

    x = np.arange(len(datasets))
    w = 0.25
    offsets = [-w, 0, w]

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.6))

    # NDCG@10
    ax = axes[0]
    for i, (cfg, lbl, col) in enumerate(zip(cfgs, cfg_lbls, colors)):
        vals = []
        for fn in files:
            d = load(fn)
            res = {r["config"]: r for r in d["results"]}
            vals.append(res[cfg]["ndcg_at_10"])
        ax.bar(x + offsets[i], vals, w, color=col, alpha=0.9, label=lbl, linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=12, ha="right", fontsize=7.5)
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(0, 0.85)
    ax.set_title("BEIR Zero-Shot NDCG@10")
    ax.legend(fontsize=7, loc="upper right")

    # Recall@100
    ax2 = axes[1]
    for i, (cfg, lbl, col) in enumerate(zip(cfgs, cfg_lbls, colors)):
        vals = []
        for fn in files:
            d = load(fn)
            res = {r["config"]: r for r in d["results"]}
            vals.append(res[cfg]["recall_at_100"])
        ax2.bar(x + offsets[i], vals, w, color=col, alpha=0.9, label=lbl, linewidth=0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=12, ha="right", fontsize=7.5)
    ax2.set_ylabel("Recall@100")
    ax2.set_ylim(0, 1.08)
    ax2.set_title("BEIR Zero-Shot Recall@100")
    ax2.legend(fontsize=7, loc="lower right")

    fig.suptitle("BEIR Zero-Shot Retrieval — 3 Corpora", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_beir.png")
    plt.close(fig)
    print("  ✓ fig_beir.png")


# ──────────────────────────────────────────────────────────────────────────
# 7. fig_chunking.png  –  Chunking ablation heatmap + line
# ──────────────────────────────────────────────────────────────────────────
def fig_chunking():
    data = load("chunking_ablation.json")
    sizes    = sorted(set(d["chunk_size"] for d in data))
    overlaps = sorted(set(d["overlap"] for d in data))

    r5_grid = np.zeros((len(sizes), len(overlaps)))
    for d in data:
        i = sizes.index(d["chunk_size"])
        j = overlaps.index(d["overlap"])
        r5_grid[i, j] = d["recall_5"]

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.5))

    # heatmap
    ax = axes[0]
    im = ax.imshow(r5_grid, cmap="RdYlGn", vmin=0.920, vmax=0.950, aspect="auto")
    ax.set_xticks(range(len(overlaps)))
    ax.set_yticks(range(len(sizes)))
    ax.set_xticklabels(overlaps, fontsize=8)
    ax.set_yticklabels(sizes, fontsize=8)
    ax.set_xlabel("Overlap")
    ax.set_ylabel("Chunk Size")
    ax.set_title("Recall@5 Heatmap")
    for i in range(len(sizes)):
        for j in range(len(overlaps)):
            ax.text(j, i, f"{r5_grid[i,j]:.3f}", ha="center", va="center", fontsize=7,
                    color="black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Recall@5")

    # MRR line per chunk size
    ax2 = axes[1]
    marker_styles = ["o", "s", "^", "D"]
    col_list = [BLUE, GREEN, ORANGE, RED]
    for idx, (sz, mk, col) in enumerate(zip(sizes, marker_styles, col_list)):
        mrr_vals = [d["mrr"] for d in data if d["chunk_size"] == sz]
        ax2.plot(overlaps, mrr_vals, marker=mk, color=col, linewidth=1.4,
                 markersize=4.5, label=f"size={sz}")
    # mark default
    ax2.axvline(128, color="grey", linewidth=0.8, linestyle=":", label="default overlap")
    ax2.set_xlabel("Overlap")
    ax2.set_ylabel("MRR")
    ax2.set_xticks(overlaps)
    ax2.set_ylim(0.77, 0.84)
    ax2.set_title("MRR vs Overlap")
    ax2.legend(fontsize=6.5)

    fig.suptitle("Chunking Strategy Ablation (Hybrid)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_chunking.png")
    plt.close(fig)
    print("  ✓ fig_chunking.png")


# ──────────────────────────────────────────────────────────────────────────
# 8. fig_rrf_k.png  –  RRF k constant sweep
# ──────────────────────────────────────────────────────────────────────────
def fig_rrf_k():
    data = load("rrf_k_ablation.json")
    ks   = [int(d["rrf_k"]) for d in data]
    r5   = [d["recall_5"] for d in data]
    mrr  = [d["mrr"] for d in data]

    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    ax2 = ax.twinx()

    l1, = ax.plot(ks, r5,  "o-", color=BLUE,   linewidth=1.8, markersize=5, label="Recall@5")
    l2, = ax2.plot(ks, mrr, "s--", color=ORANGE, linewidth=1.4, markersize=5, label="MRR")

    for x, y in zip(ks, r5):
        ax.text(x, y+0.0006, f"{y:.3f}", ha="center", fontsize=6.5, color=BLUE)

    # mark default k=60
    ax.axvline(60, color=GREY, linewidth=1, linestyle=":", label="default k=60")

    ax.set_xlabel("RRF fusion constant $k$")
    ax.set_ylabel("Recall@5", color=BLUE)
    ax2.set_ylabel("MRR", color=ORANGE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax.set_ylim(0.930, 0.955)
    ax2.set_ylim(0.800, 0.835)
    ax.set_title("RRF Constant Ablation ($k$ sweep)")
    ax.legend(handles=[l1, l2, mpatches.Patch(color=GREY, label="default k=60")],
              loc="upper right", fontsize=7)
    ax.set_xticks(ks)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rrf_k.png")
    plt.close(fig)
    print("  ✓ fig_rrf_k.png")


# ──────────────────────────────────────────────────────────────────────────
# 9. fig_quant.png  –  Embedding model quantization comparison
# ──────────────────────────────────────────────────────────────────────────
def fig_quant():
    data = load("quant_ablation.json")

    short_names = ["BGE-base-Q8\n(768-dim)", "BGE-small-Q8\n(384-dim)"]
    r5s    = [d["recall_5"]    for d in data]
    r10s   = [d["recall_10"]   for d in data]
    mrrs   = [d["mrr"]         for d in data]
    emb_ms = [d["avg_embed_ms"] for d in data]

    x = np.arange(len(short_names))
    w = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.5))

    # quality metrics
    ax = axes[0]
    for i, (vals, label, color) in enumerate(
            zip([r5s, r10s, mrrs], ["Recall@5","Recall@10","MRR"], [BLUE, GREEN, ORANGE])):
        bars = ax.bar(x + (i-1)*w, vals, w, color=color, alpha=0.9, label=label, linewidth=0)
        for xi, v in zip(x + (i-1)*w, vals):
            ax.text(xi, v+0.002, f"{v:.3f}", ha="center", fontsize=6, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=7)
    ax.set_ylim(0.72, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Quality Metrics")
    ax.legend(fontsize=6.5, loc="lower right")

    # embed latency
    ax2 = axes[1]
    bar_colors = [BLUE, GREEN]
    bars = ax2.bar(x, emb_ms, 0.45, color=bar_colors, alpha=0.9, linewidth=0)
    for xi, v in zip(x, emb_ms):
        ax2.text(xi, v+0.015, f"{v:.2f} ms", ha="center", fontsize=8, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, fontsize=7)
    ax2.set_ylim(0, 2.6)
    ax2.set_ylabel("Avg. Embed Latency (ms)")
    ax2.set_title("Embed Latency")

    fig.suptitle("Embedding Model Quantization Ablation", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_quant.png")
    plt.close(fig)
    print("  ✓ fig_quant.png")


# ──────────────────────────────────────────────────────────────────────────
# 10. fig_raptor.png  –  RAPTOR gating vs flat retrieval
# ──────────────────────────────────────────────────────────────────────────
def fig_raptor():
    bench  = load("bench_results.json")
    trivia = load("trivia_bench_results.json")

    # SQuAD flat hybrid
    sq_hybrid  = next(r for r in bench["recall"] if r["config"] == "hybrid")
    sq_r5_flat = sq_hybrid["recall"]["recall@5"]

    # SQuAD RAPTOR configs
    sq_raptor    = {r["config"]: r for r in bench["raptor"]}
    sq_gated     = sq_raptor["raptor_gated"]["recall"]["recall@5"]
    sq_exp_all   = sq_raptor["raptor_expand_all"]["recall"]["recall@5"]

    # TriviaQA: no hybrid config, only bm25_only and vector_only
    tv_vec     = next(r for r in trivia["recall"] if r["config"] == "vector_only")
    tv_r5_flat = tv_vec["recall"]["recall@5"]   # 0.680

    # TriviaQA RAPTOR: only raptor_gated and raptor_trust_all available
    tv_raptor  = {r["config"]: r for r in trivia["raptor"]}
    tv_gated   = tv_raptor["raptor_gated"]["recall"]["recall@5"]       # 0.278
    tv_trust   = tv_raptor["raptor_trust_all"]["recall"]["recall@5"]   # 0.274

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.8))

    # SQuAD panel
    ax = axes[0]
    configs_sq = ["Flat Hybrid", "RAPTOR\ngated (τ=−1.5)", "RAPTOR\nexpand-all"]
    vals_sq    = [sq_r5_flat, sq_gated, sq_exp_all]
    ax.bar(np.arange(3), vals_sq, 0.55, color=[GREEN, RED, ORANGE], alpha=0.9, linewidth=0)
    for xi, v in enumerate(vals_sq):
        ax.text(xi, v+0.005, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(configs_sq, fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall@5")
    ax.set_title("SQuAD 1.1")

    # TriviaQA panel
    ax2 = axes[1]
    configs_tv = ["Flat Vector", "RAPTOR\ngated (τ=−1.5)", "RAPTOR\ntrust-all"]
    vals_tv    = [tv_r5_flat, tv_gated, tv_trust]
    ax2.bar(np.arange(3), vals_tv, 0.55, color=[GREEN, RED, PURPLE], alpha=0.9, linewidth=0)
    for xi, v in enumerate(vals_tv):
        ax2.text(xi, v+0.005, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(configs_tv, fontsize=7)
    ax2.set_ylim(0, 0.82)
    ax2.set_ylabel("Recall@5")
    ax2.set_title("TriviaQA-RC")

    fig.suptitle("RAPTOR Hierarchical Retrieval vs Flat Retrieval", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_raptor.png")
    plt.close(fig)
    print("  ✓ fig_raptor.png")


# ──────────────────────────────────────────────────────────────────────────
# 11. fig_trivia.png  –  TriviaQA cross-dataset comparison bar
# ──────────────────────────────────────────────────────────────────────────
def fig_trivia():
    bench  = load("bench_results.json")
    trivia = load("trivia_bench_results.json")

    # SQuAD numbers
    sq_norag_em  = bench["no_rag_baseline"]["exact_match"] * 100
    sq_norag_f1  = bench["no_rag_baseline"]["f1"] * 100
    sq_rag_em    = bench["e2e"]["exact_match"] * 100
    sq_rag_f1    = bench["e2e"]["f1"] * 100

    # TriviaQA numbers
    tv_norag_em  = trivia["no_rag_baseline"]["exact_match"] * 100
    tv_norag_f1  = trivia["no_rag_baseline"]["f1"] * 100
    tv_rag_em    = trivia["e2e"]["exact_match"] * 100
    tv_rag_f1    = trivia["e2e"]["f1"] * 100

    # TriviaQA CIs
    tci = trivia["e2e_ci"]

    x = np.arange(2)
    w = 0.22
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.8))

    for ax, (norag_em, norag_f1, rag_em, rag_f1, title, ci_obj) in zip(
        axes,
        [(sq_norag_em, sq_norag_f1, sq_rag_em, sq_rag_f1, "SQuAD 1.1", bench["e2e_ci"]),
         (tv_norag_em, tv_norag_f1, tv_rag_em, tv_rag_f1, "TriviaQA-RC", tci)]
    ):
        groups = ["No-RAG", "NELA RAG"]
        em_v   = [norag_em, rag_em]
        f1_v   = [norag_f1, rag_f1]
        em_err = [0, (ci_obj["exact_match"] - ci_obj["em_ci_low"]) * 100]
        em_err_hi = [0, (ci_obj["em_ci_high"] - ci_obj["exact_match"]) * 100]

        xi = np.arange(2)
        b1 = ax.bar(xi - w/2, em_v, w*0.9, color=BLUE,  alpha=0.9, label="EM",      linewidth=0)
        b2 = ax.bar(xi + w/2, f1_v, w*0.9, color=GREEN, alpha=0.9, label="Token F1", linewidth=0)
        ax.errorbar(xi[-1] - w/2, em_v[-1], yerr=[[em_err[-1]],[em_err_hi[-1]]],
                    fmt="none", ecolor="black", capsize=3.5, linewidth=1)
        for i, (ev, fv) in enumerate(zip(em_v, f1_v)):
            ax.text(xi[i] - w/2, ev + 1, f"{ev:.1f}%", ha="center", fontsize=7, color=BLUE)
            ax.text(xi[i] + w/2, fv + 1, f"{fv:.1f}%", ha="center", fontsize=7, color=GREEN)
        ax.set_xticks(xi)
        ax.set_xticklabels(groups, fontsize=8)
        ax.set_ylim(0, max(f1_v) * 1.22)
        ax.set_ylabel("Score (%)")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Cross-Dataset Generalization: EM & F1", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_trivia.png")
    plt.close(fig)
    print("  ✓ fig_trivia.png")


# ──────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Generating figures → {OUT}")
    fig_retrieval()
    fig_latency()
    fig_e2e()
    fig_e2e_latency_dist()
    fig_scale()
    fig_beir()
    fig_chunking()
    fig_rrf_k()
    fig_quant()
    fig_raptor()
    fig_trivia()
    print("Done — all 11 figures saved.")
