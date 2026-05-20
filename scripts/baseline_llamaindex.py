#!/usr/bin/env python3
"""
baseline_llamaindex.py — LlamaIndex RAG baseline using BGE embeddings + Qwen via llama-server.

Evaluates on the same SQuAD QA pairs used by rag-bench, computing Exact Match and Token F1
so results are directly comparable to the NELA rag-bench output.

Requires:
  - A running llama-server (or OpenAI-compatible server) on --llm-url
  - pip install -r scripts/requirements_benchmark.txt

Usage:
  python scripts/baseline_llamaindex.py \\
    --corpus-dir benchmark/squad_bench_large/corpus \\
    --qa-file benchmark/squad_bench_large/qa_pairs.json \\
    --embed-model models/embedding/bge-base-en-v1.5-q8_0/bge-base-en-v1.5-q8_0.gguf \\
    --llm-url http://localhost:12346/v1 \\
    --count 500 \\
    --output results/llamaindex_baseline.json
"""

import argparse
import json
import pathlib
import re
import string
import sys
import time
from typing import Any


def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def exact_match(pred: str, golds: list[str]) -> bool:
    p = normalize(pred)
    return any(p == normalize(g) for g in golds)


def token_f1(pred: str, golds: list[str]) -> float:
    pred_tokens = normalize(pred).split()
    best = 0.0
    for gold in golds:
        gold_tokens = normalize(gold).split()
        common = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in set(pred_tokens))
        if common == 0:
            continue
        prec = common / len(pred_tokens) if pred_tokens else 0.0
        rec = common / len(gold_tokens) if gold_tokens else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        best = max(best, f1)
    return best


def build_index(corpus_dir: pathlib.Path, embed_model_path: str) -> Any:
    try:
        from llama_index.core import SimpleDirectoryReader, VectorStoreIndex  # type: ignore
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore
        from llama_index.core import Settings  # type: ignore
    except ImportError:
        sys.exit("ERROR: llama-index packages not installed. Run: pip install -r scripts/requirements_benchmark.txt")

    print(f"[llamaindex] Loading corpus from {corpus_dir} …")
    docs = SimpleDirectoryReader(str(corpus_dir)).load_data()
    print(f"[llamaindex] Loaded {len(docs)} documents. Building index …")

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    # Do NOT set Settings.llm = None here — index building never calls the LLM, and
    # setting it to None bakes MockLLM into the index's service context before run_eval
    # can configure it, resulting in empty query responses (EM=F1=0).

    t0 = time.time()
    index = VectorStoreIndex.from_documents(docs)
    elapsed = time.time() - t0
    print(f"[llamaindex] Index built in {elapsed:.1f}s")
    return index


def run_eval(index: Any, qa_file: pathlib.Path, llm_url: str, count: int) -> dict[str, Any]:
    try:
        import openai  # type: ignore
    except ImportError:
        sys.exit("ERROR: openai package not installed.")

    # Bypass LlamaIndex's LLM integration entirely.
    # Every attempt to use OpenAILike inside LlamaIndex hits one of two problems:
    #   1. system_prompt="/no_think" + ChatPromptTemplate → double system message → HTTP 400
    #   2. No system_prompt → Qwen3 generates unclosed <think> block → exhausts max_tokens → empty pred
    # Solution: use LlamaIndex only for embedding + retrieval, then call the LLM directly
    # with extra_body={"chat_template_kwargs": {"enable_thinking": False}} — the same
    # mechanism used by baseline_chromadb.py and rag_bench.rs.
    client = openai.OpenAI(base_url=llm_url, api_key="dummy")

    with open(qa_file) as f:
        qa_pairs = json.load(f)

    qa_pairs = [q for q in qa_pairs if q.get("answers")][:count]
    print(f"[llamaindex] Evaluating {len(qa_pairs)} QA pairs …")

    retriever = index.as_retriever(similarity_top_k=5)
    results = []
    latencies = []

    for i, qa in enumerate(qa_pairs):
        t0 = time.time()

        nodes = retriever.retrieve(qa["question"])
        context = "\n\n".join(n.text for n in nodes)

        resp = client.chat.completions.create(
            model="local",
            messages=[{
                "role": "user",
                "content": (
                    "Context information is below.\n"
                    "---------------------\n"
                    f"{context}\n"
                    "---------------------\n"
                    "Given the context information and not prior knowledge, answer with a short phrase only — do not explain, do not write full sentences.\n"
                    f"Query: {qa['question']}\n"
                    "Answer:"
                ),
            }],
            max_tokens=256,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        latency_ms = (time.time() - t0) * 1000.0

        raw = resp.choices[0].message.content or ""
        if i == 0:
            print(f"[llamaindex] DEBUG first raw response: {raw[:300]!r}")
        # Defensive stripping of any residual think blocks
        pred = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        em = exact_match(pred, qa["answers"])
        f1 = token_f1(pred, qa["answers"])
        results.append({"question": qa["question"], "pred": pred, "gold": qa["answers"],
                         "exact_match": em, "f1": f1, "latency_ms": latency_ms})
        latencies.append(latency_ms)

        if (i + 1) % 50 == 0:
            running_em = sum(r["exact_match"] for r in results) / len(results)
            running_f1 = sum(r["f1"] for r in results) / len(results)
            print(f"  [{i+1}/{len(qa_pairs)}] EM={running_em:.3f}  F1={running_f1:.3f}")

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    def pct(p: float) -> float:
        idx = int(p / 100 * n)
        return latencies_sorted[min(idx, n - 1)]

    overall_em = sum(r["exact_match"] for r in results) / n
    overall_f1 = sum(r["f1"] for r in results) / n

    return {
        "system": "llamaindex_bge_base_q8",
        "n": n,
        "exact_match": overall_em,
        "f1": overall_f1,
        "avg_latency_ms": sum(latencies) / n,
        "p50_latency_ms": pct(50),
        "p95_latency_ms": pct(95),
        "p99_latency_ms": pct(99),
        "per_question": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True, type=pathlib.Path)
    parser.add_argument("--qa-file", required=True, type=pathlib.Path)
    parser.add_argument("--embed-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--llm-url", default="http://localhost:12346/v1")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--output", default="results/llamaindex_baseline.json", type=pathlib.Path)
    args = parser.parse_args()

    index = build_index(args.corpus_dir, args.embed_model)
    results = run_eval(index, args.qa_file, args.llm_url, args.count)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[llamaindex] EM={results['exact_match']*100:.1f}%  F1={results['f1']*100:.1f}%")
    print(f"[llamaindex] Results → {args.output}")


if __name__ == "__main__":
    main()
