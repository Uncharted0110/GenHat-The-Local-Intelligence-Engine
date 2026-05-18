#!/usr/bin/env python3
"""
NELA RAG Benchmark Visualization Script
Generates all paper figures from bench_results.json, scale_results.json,
and e2e_results.json.

Usage:
    python benchmark/visualize_results.py

Output:
    conference_latex_template/figures/fig_retrieval.png
    conference_latex_template/figures/fig_latency.png
    conference_latex_template/figures/fig_scale.png
    conference_latex_template/figures/fig_e2e.png
    conference_latex_template/figures/fig_f1_dist.png
    conference_latex_template/figures/fig_rank_dist.png
    conference_latex_template/figures/fig_latency_box.png
    conference_latex_template/figures/fig_memory.png
    conference_latex_template/figures/fig_recall_vs_latency.png
    conference_latex_template/figures/fig_e2e_latency_dist.png
"""

import json
import os
import sys
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.ticker import MultipleLocator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_JSON   = REPO_ROOT / "genhat-desktop/src-tauri/bench_results.json"
SCALE_JSON   = REPO_ROOT / "genhat-desktop/src-tauri/scale_results.json"
E2E_JSON     = REPO_ROOT / "genhat-desktop/src-tauri/e2e_results.json"
FIGURES_DIR  = REPO_ROOT / "conference_latex_template/figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Colour palette (colour-blind friendly)
PALETTE = {
    "bm25":          "#2196F3",   # blue
    "vector":        "#FF9800",   # orange
    "hybrid":        "#4CAF50",   # green
    "hybrid_expand": "#9C27B0",   # purple
    "rag":           "#4CAF50",
    "no_rag":        "#F44336",
    "embed":         "#E53935",
    "bm25_comp":     "#1E88E5",
    "vector_comp":   "#43A047",
    "rrf":           "#FB8C00",
    "expand":        "#8E24AA",
}

CONFIG_LABELS = {
    "bm25_only":      "BM25-only",
    "vector_only":    "Vector-only",
    "hybrid":         "Hybrid",
    "hybrid_expand":  "Hybrid+Expand",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading bench_results.json …", flush=True)
with open(BENCH_JSON) as f:
    bench = json.load(f)

print("Loading scale_results.json …", flush=True)
with open(SCALE_JSON) as f:
    scale = json.load(f)

print("Loading e2e_results.json …", flush=True)
with open(E2E_JSON) as f:
    e2e = json.load(f)

# ---------------------------------------------------------------------------
# Extract top-level retrieval summary from bench_results
# ---------------------------------------------------------------------------
recalls = bench["recall"]   # list of config dicts
latency_breakdown = bench["latency_hybrid_expand"]
index_stats       = bench["index_stats"]
e2e_bench         = bench["e2e"]          # paired RAG result
no_rag            = bench["no_rag_baseline"]

configs   = [r["config"] for r in recalls]
r5_vals   = [r["recall"]["recall@5"]  for r in recalls]
r10_vals  = [r["recall"]["recall@10"] for r in recalls]
mrr_vals  = [r["mrr"] for r in recalls]
lat_vals  = [r["avg_latency_ms"] for r in recalls]

# Per-question data for box plots / distributions
per_q_by_config = {r["config"]: r["per_question"] for r in recalls}

# ---------------------------------------------------------------------------
# Fig 1 — Retrieval performance: R@5, R@10, MRR grouped bar chart
# ---------------------------------------------------------------------------
def fig_retrieval():
    n      = len(configs)
    x      = np.arange(n)
    width  = 0.24
    cols   = [PALETTE["bm25"], PALETTE["vector"], PALETTE["hybrid"],
              PALETTE["hybrid_expand"]]

    fig, ax = plt.subplots(figsize=(7, 4.2))

    offsets = [-width, 0, width]
    metrics = [r5_vals, r10_vals, mrr_vals]
    labels  = ["Recall@5", "Recall@10", "MRR"]
    hatches = ["", "//", "xx"]
    bar_cols = ["#42A5F5", "#66BB6A", "#FFA726"]

    bars_list = []
    for i, (metric, label, hatch, col) in enumerate(
            zip(metrics, labels, hatches, bar_cols)):
        bars = ax.bar(x + offsets[i], metric, width,
                      label=label, color=col, hatch=hatch,
                      edgecolor="white", linewidth=0.6, alpha=0.9)
        bars_list.append(bars)

    ax.set_xlabel("Retrieval Configuration")
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Performance on SQuAD 1.1\n(10,570 QA pairs, 2,067 documents)")
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs], rotation=10, ha="right")
    ax.set_ylim(0.74, 1.01)
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.7)

    # Annotate best bar in each group
    for bars in bars_list:
        best_val = max(b.get_height() for b in bars)
        for bar in bars:
            if abs(bar.get_height() - best_val) < 1e-9:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{bar.get_height():.3f}",
                        ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold")

    fig.tight_layout()
    path = FIGURES_DIR / "fig_retrieval.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Fig 2 — Query latency breakdown (horizontal stacked bar)
# ---------------------------------------------------------------------------
def fig_latency():
    stages = ["Embed", "BM25", "Vector ANN", "RRF", "Expand"]
    times  = [
        latency_breakdown["embed_ms"],
        latency_breakdown["bm25_ms"],
        latency_breakdown["vector_ms"],
        latency_breakdown["rrf_ms"],
        latency_breakdown["expand_ms"],
    ]
    total  = latency_breakdown["total_ms"]
    pcts   = [t / total * 100 for t in times]

    stage_cols = [PALETTE["embed"], PALETTE["bm25_comp"], PALETTE["vector_comp"],
                  PALETTE["rrf"], PALETTE["expand"]]

    fig, ax = plt.subplots(figsize=(7, 2.6))
    left = 0.0
    for stage, t, pct, col in zip(stages, times, pcts, stage_cols):
        bar = ax.barh(0, t, left=left, color=col, edgecolor="white",
                      linewidth=0.8, height=0.5, label=stage)
        if pct > 2.5:
            ax.text(left + t / 2, 0,
                    f"{stage}\n{t:.1f} ms\n({pct:.1f}%)",
                    ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
        elif pct > 0.5:
            ax.text(left + t / 2, 0,
                    f"{t:.2f}",
                    ha="center", va="center",
                    fontsize=7.5, color="white")
        left += t

    ax.set_xlim(0, total * 1.04)
    ax.set_yticks([])
    ax.set_xlabel("Latency (ms)")
    ax.set_title(f"Query Latency Breakdown — Hybrid+Expand (total = {total:.1f} ms)")
    ax.legend(loc="upper right", ncol=5, fontsize=8.5, framealpha=0.9)
    ax.axvline(total, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(total + 0.3, 0, f"{total:.1f} ms", va="center", fontsize=9)

    fig.tight_layout()
    path = FIGURES_DIR / "fig_latency.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Fig 3 — Scale degradation (two sub-panels)
# ---------------------------------------------------------------------------
def fig_scale():
    pts        = scale["points"]
    doc_counts = [p["doc_count"]           for p in pts]
    r5_hyb     = [p["recall_5_hybrid"]     for p in pts]
    r5_bm      = [p["recall_5_bm25"]       for p in pts]
    r5_vec     = [p["recall_5_vector"]     for p in pts]
    lat_hyb    = [p["avg_latency_ms_hybrid"] for p in pts]
    mem_mb     = [p["memory_mb"]           for p in pts]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 6.5), sharex=True)
    fig.suptitle("Scale Degradation (SQuAD 1.1 Corpus Checkpoints)", y=1.01)

    # Panel 1 — Recall@5
    ax1.plot(doc_counts, r5_hyb, "o-",  color=PALETTE["hybrid"],
             linewidth=2, markersize=7, label="Hybrid", zorder=3)
    ax1.plot(doc_counts, r5_bm,  "s--", color=PALETTE["bm25"],
             linewidth=1.5, markersize=6, label="BM25-only", zorder=2)
    ax1.plot(doc_counts, r5_vec, "^--", color=PALETTE["vector"],
             linewidth=1.5, markersize=6, label="Vector-only", zorder=2)

    for x, y in zip(doc_counts, r5_hyb):
        ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=8.5,
                     color=PALETTE["hybrid"])

    ax1.set_ylabel("Recall@5")
    ax1.set_ylim(0.82, 1.01)
    ax1.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax1.legend(loc="lower left", framealpha=0.9)
    ax1.grid(axis="both", linestyle="--", alpha=0.35, linewidth=0.7)

    # Panel 2 — Latency and memory (dual axis)
    ax2.plot(doc_counts, lat_hyb, "o-", color="#E53935",
             linewidth=2, markersize=7, label="Hybrid latency (ms)")
    ax2.set_ylabel("Avg. Query Latency (ms)", color="#E53935")
    ax2.tick_params(axis="y", labelcolor="#E53935")

    ax2b = ax2.twinx()
    ax2b.spines["right"].set_visible(True)
    ax2b.plot(doc_counts, mem_mb, "D--", color="#1565C0",
              linewidth=1.8, markersize=6, label="IVF memory (MB)")
    ax2b.set_ylabel("IVF Index Memory (MB)", color="#1565C0")
    ax2b.tick_params(axis="y", labelcolor="#1565C0")

    for x, y in zip(doc_counts, lat_hyb):
        ax2.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=8.5,
                     color="#E53935")

    ax2.set_xlabel("Corpus Size (documents)")
    ax2.set_xticks(doc_counts)
    ax2.set_xticklabels([str(d) for d in doc_counts])
    ax2.grid(axis="both", linestyle="--", alpha=0.35, linewidth=0.7)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", framealpha=0.9)

    fig.tight_layout()
    path = FIGURES_DIR / "fig_scale.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Fig 4 — E2E: RAG vs no-RAG grouped bar
# ---------------------------------------------------------------------------
def fig_e2e():
    systems = ["No-RAG\n(Qwen3.5-0.8B)", "NELA\nHybrid RAG"]
    em_vals  = [no_rag["exact_match"] * 100, e2e_bench["exact_match"] * 100]
    f1_vals  = [no_rag["f1"] * 100,          e2e_bench["f1"] * 100]

    x      = np.arange(len(systems))
    width  = 0.32

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bars_em = ax.bar(x - width / 2, em_vals, width,
                     label="Exact Match (EM)", color="#42A5F5",
                     edgecolor="white", linewidth=0.8)
    bars_f1 = ax.bar(x + width / 2, f1_vals, width,
                     label="Token F1",        color="#66BB6A",
                     edgecolor="white", linewidth=0.8)

    for bar in list(bars_em) + list(bars_f1):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{bar.get_height():.1f}%",
                ha="center", va="bottom", fontsize=9.5)

    # Annotate gain arrow
    em_gain = em_vals[1] - em_vals[0]
    f1_gain = f1_vals[1] - f1_vals[0]
    ax.annotate(
        f"+{em_gain:.1f} pp",
        xy=(0.5 - width / 2, (em_vals[0] + em_vals[1]) / 2),
        xytext=(0.5 - width / 2 + 0.05, (em_vals[0] + em_vals[1]) / 2 + 5),
        fontsize=8.5, color="#1565C0",
        arrowprops=dict(arrowstyle="-|>", color="#1565C0", lw=1.2),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=10)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 90)
    ax.set_title(
        "End-to-End Answer Quality\n(SQuAD 1.1, n=50, Matched Sample)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.7)

    fig.tight_layout()
    path = FIGURES_DIR / "fig_e2e.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Fig 5 — Per-question Token F1 distribution (E2E eval run histogram)
# ---------------------------------------------------------------------------
def fig_f1_dist():
    f1_scores = [q["f1"] for q in e2e["per_question"]]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    n, bins, patches = ax.hist(f1_scores, bins=20, range=(0, 1),
                                color="#66BB6A", edgecolor="white",
                                linewidth=0.7, alpha=0.85)

    # Colour the 0 bucket red to highlight failures
    patches[0].set_facecolor("#EF5350")

    mean_f1 = float(np.mean(f1_scores))
    ax.axvline(mean_f1, color="#1565C0", linestyle="--", linewidth=1.5,
               label=f"Mean F1 = {mean_f1:.3f}")

    ax.set_xlabel("Token F1 Score")
    ax.set_ylabel("Number of Questions")
    ax.set_title(f"Per-Question Token F1 Distribution\n(NELA Hybrid RAG, n={len(f1_scores)}, eval run)")
    ax.set_xlim(-0.05, 1.05)
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.7)

    # Annotation: % perfect
    perfect = sum(1 for s in f1_scores if s == 1.0)
    ax.text(0.98, 0.97,
            f"F1 = 1.0: {perfect}/{len(f1_scores)} ({perfect/len(f1_scores)*100:.0f}%)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#2E7D32")

    fig.tight_layout()
    path = FIGURES_DIR / "fig_f1_dist.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Fig 6 — First-relevant-rank distribution (MRR analysis, hybrid config)
# ---------------------------------------------------------------------------
def fig_rank_dist():
    pq = per_q_by_config["hybrid"]
    ranks = []
    missed = 0
    for q in pq:
        r = q.get("first_relevant_rank")
        if r is not None:
            ranks.append(r)
        else:
            missed += 1

    max_rank = 10
    counts   = [0] * (max_rank + 1)
    for r in ranks:
        if 1 <= r <= max_rank:
            counts[r] += 1
    if missed:
        counts[0] = missed  # rank=0 means not found in top-10

    x_labels = [f"Rank {i}" if i > 0 else "Not Found" for i in range(max_rank + 1)]
    x_vals   = list(range(max_rank + 1))
    bar_cols = ["#EF5350"] + ["#42A5F5"] * 5 + ["#90CAF9"] * 5

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(x_vals, counts, color=bar_cols, edgecolor="white",
                  linewidth=0.7)

    total = len(pq)
    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + total * 0.003,
                    f"{cnt}\n({cnt/total*100:.1f}%)",
                    ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_ylabel("Number of Questions")
    ax.set_title("First-Relevant-Rank Distribution — Hybrid Config\n"
                 f"(MRR = 0.853, n = {total:,} QA pairs)")
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.7)

    not_found_patch = mpatches.Patch(color="#EF5350", label="Not found in top-10")
    rank1_patch     = mpatches.Patch(color="#42A5F5", label="Found rank 1–5")
    rank610_patch   = mpatches.Patch(color="#90CAF9", label="Found rank 6–10")
    ax.legend(handles=[not_found_patch, rank1_patch, rank610_patch],
              framealpha=0.9)

    fig.tight_layout()
    path = FIGURES_DIR / "fig_rank_dist.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Fig 7 — Per-query latency box plot (all 4 configs)
# ---------------------------------------------------------------------------
def fig_latency_box():
    latency_data = [
        [q["latency_ms"] for q in per_q_by_config[cfg]]
        for cfg in configs
    ]
    labels = [CONFIG_LABELS[c] for c in configs]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(latency_data, labels=labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.8),
                    flierprops=dict(marker=".", markersize=3,
                                   markerfacecolor="gray", alpha=0.4),
                    notch=False, whis=1.5)

    box_cols = [PALETTE["bm25"], PALETTE["vector"],
                PALETTE["hybrid"], PALETTE["hybrid_expand"]]
    for patch, col in zip(bp["boxes"], box_cols):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    ax.set_xlabel("Retrieval Configuration")
    ax.set_ylabel("Query Latency (ms)")
    ax.set_title("Per-Query Retrieval Latency Distribution\n(10,570 queries per configuration)")
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(
        base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9]))
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7, which="both")

    # Add mean markers
    for i, data in enumerate(latency_data, start=1):
        mean_val = float(np.mean(data))
        ax.plot(i, mean_val, "D", color="white", markersize=7,
                markeredgecolor="black", markeredgewidth=1, zorder=5)

    mean_patch = mpatches.Patch(facecolor="white", edgecolor="black",
                                label="Mean (diamond marker)")
    ax.legend(handles=[mean_patch], loc="upper right", framealpha=0.9)

    fig.tight_layout()
    path = FIGURES_DIR / "fig_latency_box.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Fig 8 — Memory compression bar (raw F32 vs IVF quantized)
# ---------------------------------------------------------------------------
def fig_memory():
    categories = ["Full-Precision\n(F32)", "IVF Scalar-\nQuantized"]
    values     = [index_stats["raw_f32_estimate_mb"], index_stats["memory_mb"]]
    bar_cols   = ["#EF9A9A", "#A5D6A7"]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    bars = ax.bar(categories, values, color=bar_cols,
                  edgecolor="white", linewidth=0.8, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{val:.2f} MB",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ratio = index_stats["compression_ratio"]
    ax.annotate(
        f"{ratio:.2f}× compression",
        xy=(0.5, values[1] + 0.15),
        fontsize=10, ha="center", color="#2E7D32", fontweight="bold",
    )

    ax.set_ylabel("Memory (MB)")
    ax.set_title(f"Vector Index Memory\n(2,138 vectors × 768 dim)")
    ax.set_ylim(0, values[0] * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.7)

    fig.tight_layout()
    path = FIGURES_DIR / "fig_memory.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Fig 9 — Recall@5 vs Avg Latency scatter (all configs)
# ---------------------------------------------------------------------------
def fig_recall_vs_latency():
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    plot_cols = [PALETTE["bm25"], PALETTE["vector"],
                 PALETTE["hybrid"], PALETTE["hybrid_expand"]]

    for cfg, r5, lat, col in zip(configs, r5_vals, lat_vals, plot_cols):
        ax.scatter(lat, r5, color=col, s=120, zorder=4,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(CONFIG_LABELS[cfg], (lat, r5),
                    textcoords="offset points",
                    xytext=(5, 5), fontsize=9, color=col)

    ax.set_xlabel("Average Query Latency (ms)")
    ax.set_ylabel("Recall@5")
    ax.set_title("Retrieval Quality vs. Latency Trade-off\n(each point = one retrieval configuration)")
    ax.set_ylim(0.88, 0.97)
    ax.set_xlim(45, 85)
    ax.grid(linestyle="--", alpha=0.4, linewidth=0.7)

    fig.tight_layout()
    path = FIGURES_DIR / "fig_recall_vs_latency.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Fig 10 — E2E answer latency distribution (per question, eval run)
# ---------------------------------------------------------------------------
def fig_e2e_latency_dist():
    latencies_s = [q["latency_ms"] / 1000 for q in e2e["per_question"]]
    em_flags    = [q["exact_match"] for q in e2e["per_question"]]

    correct_lat   = [l for l, m in zip(latencies_s, em_flags) if m]
    incorrect_lat = [l for l, m in zip(latencies_s, em_flags) if not m]

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, max(latencies_s) * 1.05, 20)
    ax.hist(correct_lat,   bins=bins, color="#66BB6A", alpha=0.75,
            label=f"Correct (EM=1, n={len(correct_lat)})", edgecolor="white")
    ax.hist(incorrect_lat, bins=bins, color="#EF5350", alpha=0.75,
            label=f"Incorrect (EM=0, n={len(incorrect_lat)})", edgecolor="white")

    mean_s = float(np.mean(latencies_s))
    ax.axvline(mean_s, color="black", linestyle="--", linewidth=1.4,
               label=f"Mean = {mean_s:.1f} s")

    ax.set_xlabel("Generation Latency (s)")
    ax.set_ylabel("Number of Questions")
    ax.set_title("End-to-End Generation Latency by Answer Correctness\n"
                 f"(Hybrid RAG + Qwen3.5-0.8B, n={len(latencies_s)}, eval run)")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.7)

    fig.tight_layout()
    path = FIGURES_DIR / "fig_e2e_latency_dist.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Fig 11 — Scale: all three configs side-by-side at each checkpoint
# ---------------------------------------------------------------------------
def fig_scale_configs():
    pts        = scale["points"]
    doc_counts = [p["doc_count"]           for p in pts]
    r5_hyb     = [p["recall_5_hybrid"]     for p in pts]
    r5_bm      = [p["recall_5_bm25"]       for p in pts]
    r5_vec     = [p["recall_5_vector"]     for p in pts]

    n      = len(doc_counts)
    x      = np.arange(n)
    width  = 0.25

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - width, r5_bm,  width, label="BM25-only",
           color=PALETTE["bm25"],   edgecolor="white", alpha=0.85)
    ax.bar(x,         r5_vec, width, label="Vector-only",
           color=PALETTE["vector"], edgecolor="white", alpha=0.85)
    ax.bar(x + width, r5_hyb, width, label="Hybrid",
           color=PALETTE["hybrid"], edgecolor="white", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in doc_counts])
    ax.set_xlabel("Corpus Size (documents)")
    ax.set_ylabel("Recall@5")
    ax.set_ylim(0.80, 1.02)
    ax.set_title("Recall@5 vs. Corpus Size\n(all three retrieval configurations)")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.7)

    for i, (bm, vec, hyb) in enumerate(zip(r5_bm, r5_vec, r5_hyb)):
        ax.text(i - width, bm  + 0.003, f"{bm:.3f}",  ha="center",
                va="bottom", fontsize=7.5)
        ax.text(i,         vec + 0.003, f"{vec:.3f}", ha="center",
                va="bottom", fontsize=7.5)
        ax.text(i + width, hyb + 0.003, f"{hyb:.3f}", ha="center",
                va="bottom", fontsize=7.5, fontweight="bold")

    fig.tight_layout()
    path = FIGURES_DIR / "fig_scale_configs.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Run all figures
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    figures = [
        ("fig_retrieval",         fig_retrieval),
        ("fig_latency",           fig_latency),
        ("fig_scale",             fig_scale),
        ("fig_e2e",               fig_e2e),
        ("fig_f1_dist",           fig_f1_dist),
        ("fig_rank_dist",         fig_rank_dist),
        ("fig_latency_box",       fig_latency_box),
        ("fig_memory",            fig_memory),
        ("fig_recall_vs_latency", fig_recall_vs_latency),
        ("fig_e2e_latency_dist",  fig_e2e_latency_dist),
        ("fig_scale_configs",     fig_scale_configs),
    ]

    print(f"\nGenerating {len(figures)} figures → {FIGURES_DIR}\n")
    errors = []
    for name, fn in figures:
        try:
            print(f"[{name}]")
            fn()
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            errors.append((name, exc))

    print(f"\nDone. {len(figures) - len(errors)}/{len(figures)} figures generated.")
    if errors:
        print("Failed:")
        for name, exc in errors:
            print(f"  {name}: {exc}")
        sys.exit(1)
