#!/usr/bin/env python3
"""
baseline_chromadb.py — ChromaDB RAG baseline using sentence-transformers + local LLM.

Evaluates on the same SQuAD QA pairs used by rag-bench so results are comparable to
NELA rag-bench output.

Requires:
  - A running llama-server (OpenAI-compatible) on --llm-url
  - pip install -r scripts/requirements_benchmark.txt

Usage:
  python scripts/baseline_chromadb.py \\
    --corpus-dir benchmark/squad_bench_large/corpus \\
    --qa-file benchmark/squad_bench_large/qa_pairs.json \\
    --llm-url http://localhost:12346/v1 \\
    --count 500 \\
    --output results/chromadb_baseline.json
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


def build_chroma_collection(corpus_dir: pathlib.Path) -> Any:
    try:
        import chromadb  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        sys.exit("ERROR: chromadb / sentence-transformers not installed.")

    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    client = chromadb.Client()
    col = client.create_collection("nela_baseline")

    ids = []
    docs = []
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        # Chunk naively at 512 chars
        chunks = [text[i:i+512] for i in range(0, len(text), 512)]
        for j, chunk in enumerate(chunks):
            ids.append(f"{path.stem}__{j}")
            docs.append(chunk)

    batch = 256
    print(f"[chromadb] Embedding {len(docs)} chunks in batches of {batch} …")
    for start in range(0, len(docs), batch):
        end = start + batch
        batch_docs = docs[start:end]
        batch_ids = ids[start:end]
        embeddings = model.encode(batch_docs, normalize_embeddings=True).tolist()
        col.add(ids=batch_ids, documents=batch_docs, embeddings=embeddings)

    print(f"[chromadb] Collection ready with {col.count()} chunks.")
    return col, model


def run_eval(col: Any, model: Any, qa_file: pathlib.Path, llm_url: str, count: int) -> dict[str, Any]:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        sys.exit("ERROR: openai package not installed.")

    client = OpenAI(base_url=llm_url, api_key="dummy")

    with open(qa_file) as f:
        qa_pairs = json.load(f)
    qa_pairs = [q for q in qa_pairs if q.get("answers")][:count]
    print(f"[chromadb] Evaluating {len(qa_pairs)} QA pairs …")

    results = []
    latencies = []

    for i, qa in enumerate(qa_pairs):
        t0 = time.time()
        q_emb = model.encode([qa["question"]], normalize_embeddings=True).tolist()
        hits = col.query(query_embeddings=q_emb, n_results=5)
        context = "\n\n".join(hits["documents"][0])
        prompt = (
            f"Answer the question using only the context below.\n\n"
            f"Context:\n{context}\n\nQuestion: {qa['question']}\nAnswer:"
        )
        chat = client.chat.completions.create(
            model="local",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
            # Disable Qwen3 thinking at the chat-template level.
            # Do NOT use budget_tokens=0 alone — on long RAG contexts it leaves an
            # unclosed <think> tag that strips the entire response (EM=F1=0).
            # chat_template_kwargs is the llama.cpp-native way to fully suppress thinking.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = chat.choices[0].message.content if chat.choices else ""
        # Strip any residual <think>...</think> blocks as a defensive post-processing step.
        pred = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        latency_ms = (time.time() - t0) * 1000.0

        em = exact_match(pred, qa["answers"])
        f1 = token_f1(pred, qa["answers"])
        results.append({"question": qa["question"], "pred": pred, "gold": qa["answers"],
                         "exact_match": em, "f1": f1, "latency_ms": latency_ms})
        latencies.append(latency_ms)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(qa_pairs)}]  EM={sum(r['exact_match'] for r in results)/len(results):.3f}"
                  f"  F1={sum(r['f1'] for r in results)/len(results):.3f}")

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    def pct(p: float) -> float:
        idx = int(p / 100 * n)
        return latencies_sorted[min(idx, n - 1)]

    return {
        "system": "chromadb_bge_base_openai_llm",
        "n": n,
        "exact_match": sum(r["exact_match"] for r in results) / n,
        "f1": sum(r["f1"] for r in results) / n,
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
    parser.add_argument("--llm-url", default="http://localhost:12346/v1")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--output", default="results/chromadb_baseline.json", type=pathlib.Path)
    args = parser.parse_args()

    col, embed_model = build_chroma_collection(args.corpus_dir)
    results = run_eval(col, embed_model, args.qa_file, args.llm_url, args.count)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[chromadb] EM={results['exact_match']*100:.1f}%  F1={results['f1']*100:.1f}%")
    print(f"[chromadb] Results → {args.output}")


if __name__ == "__main__":
    main()
