#!/usr/bin/env python3
"""
generate_paper_assets.py — Read all rag-bench result JSONs and emit publication-ready assets.

Outputs:
  results/paper/table_e2e.tex         — E2E comparison table (NELA vs baselines)
  results/paper/table_beir.tex        — BEIR NDCG@10 table
  results/paper/table_ablate.tex      — Chunking + RRF-k ablation table
  results/paper/fig_latency_cdf.pdf   — Latency CDF (NELA vs baselines)
  results/paper/fig_recall_bar.pdf    — Recall@5/10 grouped bar chart
  results/paper/fig_chunking.pdf      — Chunking ablation heatmap
  results/paper/fig_rrf_k.pdf         — RRF-k ablation line plot
  results/paper/fig_quant.pdf         — Quantisation ablation bar chart
  results/paper/fig_scale.pdf         — Scale degradation recall/latency vs corpus size
  results/paper/fig_beir.pdf          — BEIR per-dataset NDCG@10 grouped bars

Usage:
  python scripts/generate_paper_assets.py --results-dir results/
"""

import argparse
import json
import pathlib
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS = pathlib.Path("results")
OUT = pathlib.Path("results/paper")


# ── helpers ───────────────────────────────────────────────────────────────────

def load_json(path: pathlib.Path) -> Any:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def savefig(name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [fig] {OUT / name}")


def write_tex(name: str, content: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(content, encoding="utf-8")
    print(f"  [tex] {OUT / name}")


# ── E2E table ─────────────────────────────────────────────────────────────────

def table_e2e(results_dir: pathlib.Path) -> None:
    rows = []

    nela_bench = load_json(results_dir / "bench_results.json")
    if nela_bench and nela_bench.get("e2e"):
        e2e = nela_bench["e2e"]
        ci = nela_bench.get("e2e_ci", {})
        rows.append({
            "system": r"\textbf{NELA (ours)}",
            "em": e2e["exact_match"] * 100,
            "em_lo": (ci.get("em_ci_low", 0)) * 100,
            "em_hi": (ci.get("em_ci_high", 0)) * 100,
            "f1": e2e["f1"] * 100,
            "f1_lo": (ci.get("f1_ci_low", 0)) * 100,
            "f1_hi": (ci.get("f1_ci_high", 0)) * 100,
            "lat": e2e.get("avg_latency_ms", 0),
            "n": e2e.get("sample_count", "-"),
        })
        if nela_bench.get("no_rag_baseline"):
            nr = nela_bench["no_rag_baseline"]
            rows.append({
                "system": "No-RAG baseline",
                "em": nr["exact_match"] * 100,
                "em_lo": 0, "em_hi": 0,
                "f1": nr["f1"] * 100,
                "f1_lo": 0, "f1_hi": 0,
                "lat": nr.get("avg_latency_ms", 0),
                "n": nr.get("sample_count", "-"),
            })

    for system, fname in [("LlamaIndex + BGE", "llamaindex_baseline.json"),
                           ("ChromaDB + BGE", "chromadb_baseline.json")]:
        d = load_json(results_dir / fname)
        if d:
            rows.append({
                "system": system,
                "em": d["exact_match"] * 100,
                "em_lo": 0, "em_hi": 0,
                "f1": d["f1"] * 100,
                "f1_lo": 0, "f1_hi": 0,
                "lat": d.get("avg_latency_ms", 0),
                "n": d.get("n", "-"),
            })

    if not rows:
        print("  [table_e2e] No data, skipping.")
        return

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{End-to-end Answer Quality on SQuAD 1.1 ($n=500$)}",
        r"\label{tab:e2e}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"System & EM (\%) & F1 (\%) & Latency (ms) & $n$ \\",
        r"\midrule",
    ]
    for r in rows:
        em_str = f"{r['em']:.1f}"
        f1_str = f"{r['f1']:.1f}"
        if r["em_lo"] and r["em_hi"]:
            em_str += f" [{r['em_lo']:.1f}--{r['em_hi']:.1f}]"
        if r["f1_lo"] and r["f1_hi"]:
            f1_str += f" [{r['f1_lo']:.1f}--{r['f1_hi']:.1f}]"
        lines.append(f"{r['system']} & {em_str} & {f1_str} & {r['lat']:.0f} & {r['n']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("table_e2e.tex", "\n".join(lines))


# ── BEIR helpers ─────────────────────────────────────────────────────────────

def load_beir_datasets(results_dir: pathlib.Path) -> dict:
    """Load all beir_<dataset>.json files and return {dataset: {config: metrics}}."""
    data: dict = {}
    for path in sorted(results_dir.glob("beir_*.json")):
        d = load_json(path)
        if not d or not d.get("results"):
            continue
        # Derive a clean dataset name from the file name (beir_scifact.json → scifact)
        name = path.stem[len("beir_"):]
        data[name] = {row["config"]: row for row in d["results"]}
    return data


# ── BEIR table ────────────────────────────────────────────────────────────────

def table_beir(results_dir: pathlib.Path) -> None:
    datasets = load_beir_datasets(results_dir)
    if not datasets:
        print("  [table_beir] No data, skipping.")
        return
    configs = sorted({cfg for rows in datasets.values() for cfg in rows})
    col_spec = "l" + "c" * len(configs)
    header = "Dataset & " + " & ".join(c.capitalize() for c in configs) + r" \\"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{BEIR NDCG@10 per dataset and retrieval configuration}",
        r"\label{tab:beir}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for ds, rows in sorted(datasets.items()):
        cells = " & ".join(
            f"{rows[c]['ndcg_at_10']:.3f}" if c in rows else "--"
            for c in configs
        )
        lines.append(f"{ds} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("table_beir.tex", "\n".join(lines))


# ── Chunking ablation table ───────────────────────────────────────────────────

def table_ablate(results_dir: pathlib.Path) -> None:
    d = load_json(results_dir / "chunking_ablation.json")
    if not d:
        print("  [table_ablate] No chunking data, skipping.")
        return
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Chunking Ablation: Recall@5 vs Chunk Size and Overlap}",
        r"\label{tab:chunking}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"Chunk Size & Overlap & Recall@5 & Recall@10 & MRR \\",
        r"\midrule",
    ]
    for pt in d:
        lines.append(
            f"{pt['chunk_size']} & {pt['overlap']} & {pt['recall_5']:.3f} "
            f"& {pt['recall_10']:.3f} & {pt['mrr']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("table_ablate.tex", "\n".join(lines))


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_latency_cdf(results_dir: pathlib.Path) -> None:
    systems: dict[str, list[float]] = {}

    bench = load_json(results_dir / "bench_results.json")
    if bench and bench.get("e2e"):
        lats = [q.get("latency_ms", 0) for q in bench["e2e"].get("per_question", [])]
        if lats:
            systems["NELA (ours)"] = lats

    for label, fname in [("LlamaIndex", "llamaindex_baseline.json"),
                          ("ChromaDB", "chromadb_baseline.json")]:
        d = load_json(results_dir / fname)
        if d:
            lats = [q.get("latency_ms", 0) for q in d.get("per_question", [])]
            if lats:
                systems[label] = lats

    if not systems:
        print("  [fig_latency_cdf] No latency data, skipping.")
        return

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for label, lats in systems.items():
        s = sorted(lats)
        y = np.linspace(0, 1, len(s))
        ax.plot(s, y, label=label)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("End-to-End Latency CDF")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    savefig("fig_latency_cdf.pdf")


def fig_recall_bar(results_dir: pathlib.Path) -> None:
    bench = load_json(results_dir / "bench_results.json")
    if not bench or not bench.get("retrieval"):
        print("  [fig_recall_bar] No retrieval data, skipping.")
        return
    configs = []
    r5 = []
    r10 = []
    for cfg in bench["retrieval"]:
        configs.append(cfg["config"])
        r5.append(cfg["recall"].get("recall@5", 0))
        r10.append(cfg["recall"].get("recall@10", 0))

    x = np.arange(len(configs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(x - width/2, r5, width, label="Recall@5")
    ax.bar(x + width/2, r10, width, label="Recall@10")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha="right")
    ax.set_ylabel("Recall")
    ax.set_title("Retrieval Recall by Configuration")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    savefig("fig_recall_bar.pdf")


def fig_chunking(results_dir: pathlib.Path) -> None:
    d = load_json(results_dir / "chunking_ablation.json")
    if not d:
        print("  [fig_chunking] No data, skipping.")
        return
    chunk_sizes = sorted(set(pt["chunk_size"] for pt in d))
    overlaps = sorted(set(pt["overlap"] for pt in d))
    matrix = np.zeros((len(overlaps), len(chunk_sizes)))
    for pt in d:
        i = overlaps.index(pt["overlap"])
        j = chunk_sizes.index(pt["chunk_size"])
        matrix[i][j] = pt["recall_5"]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(chunk_sizes)))
    ax.set_xticklabels(chunk_sizes)
    ax.set_yticks(range(len(overlaps)))
    ax.set_yticklabels(overlaps)
    ax.set_xlabel("Chunk Size (tokens)")
    ax.set_ylabel("Overlap (tokens)")
    ax.set_title("Recall@5 Chunking Ablation")
    plt.colorbar(im, ax=ax, label="Recall@5")
    savefig("fig_chunking.pdf")


def fig_rrf_k(results_dir: pathlib.Path) -> None:
    d = load_json(results_dir / "rrf_k_ablation.json")
    if not d:
        print("  [fig_rrf_k] No data, skipping.")
        return
    ks = [pt["rrf_k"] for pt in d]
    r5 = [pt["recall_5"] for pt in d]
    r10 = [pt["recall_10"] for pt in d]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(ks, r5, marker="o", label="Recall@5")
    ax.plot(ks, r10, marker="s", label="Recall@10")
    ax.set_xlabel("RRF $k$ constant")
    ax.set_ylabel("Recall")
    ax.set_title("RRF $k$ Ablation")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    savefig("fig_rrf_k.pdf")


def fig_quant(results_dir: pathlib.Path) -> None:
    d = load_json(results_dir / "quant_ablation.json")
    if not d:
        print("  [fig_quant] No data, skipping.")
        return
    names = [pt["model_name"] for pt in d]
    r5 = [pt["recall_5"] for pt in d]
    lats = [pt["avg_embed_ms"] for pt in d]

    x = np.arange(len(names))
    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    ax2 = ax1.twinx()
    ax1.bar(x, r5, color="steelblue", alpha=0.8, label="Recall@5")
    ax2.plot(x, lats, color="darkorange", marker="o", label="Embed Latency (ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_ylabel("Recall@5")
    ax2.set_ylabel("Avg Embed Latency (ms)")
    ax1.set_title("Quantisation Ablation")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88))
    savefig("fig_quant.pdf")


def fig_scale(results_dir: pathlib.Path) -> None:
    d = load_json(results_dir / "scale_results.json")
    if not d:
        print("  [fig_scale] No scale data, skipping.")
        return
    pts = d.get("points", d.get("checkpoints", d)) if isinstance(d, dict) else d
    sizes = [p["doc_count"] for p in pts]
    r5 = [p.get("recall_5_hybrid", p.get("recall_5", p.get("recall", {}).get("recall@5", 0))) for p in pts]
    lats = [p.get("avg_latency_ms_hybrid", p.get("avg_latency_ms", 0)) for p in pts]

    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    ax2 = ax1.twinx()
    ax1.plot(sizes, r5, marker="o", color="steelblue", label="Recall@5")
    ax2.plot(sizes, lats, marker="s", color="darkorange", label="Latency (ms)")
    ax1.set_xlabel("Corpus Size (docs)")
    ax1.set_ylabel("Recall@5")
    ax2.set_ylabel("Avg Query Latency (ms)")
    ax1.set_title("Scale Degradation")
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    ax1.grid(True, linestyle="--", alpha=0.4)
    savefig("fig_scale.pdf")


def fig_beir(results_dir: pathlib.Path) -> None:
    datasets = load_beir_datasets(results_dir)
    if not datasets:
        print("  [fig_beir] No BEIR data, skipping.")
        return
    configs = sorted({cfg for rows in datasets.values() for cfg in rows})
    ds_names = sorted(datasets.keys())

    x = np.arange(len(ds_names))
    width = 0.8 / max(len(configs), 1)
    fig, ax = plt.subplots(figsize=(max(5, len(ds_names) * 1.8), 3.5))
    for i, cfg in enumerate(configs):
        ndcg = [datasets[ds].get(cfg, {}).get("ndcg_at_10", 0.0) for ds in ds_names]
        ax.bar(x + (i - len(configs) / 2 + 0.5) * width, ndcg, width, label=cfg.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, rotation=15, ha="right")
    ax.set_ylabel("NDCG@10")
    ax.set_title("BEIR Retrieval Quality")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    savefig("fig_beir.pdf")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results", type=pathlib.Path)
    args = parser.parse_args()

    r = args.results_dir
    print(f"[assets] Reading results from {r} …")

    table_e2e(r)
    table_beir(r)
    table_ablate(r)
    fig_latency_cdf(r)
    fig_recall_bar(r)
    fig_chunking(r)
    fig_rrf_k(r)
    fig_quant(r)
    fig_scale(r)
    fig_beir(r)

    print(f"\n[assets] All assets written to {OUT}/")


if __name__ == "__main__":
    main()
