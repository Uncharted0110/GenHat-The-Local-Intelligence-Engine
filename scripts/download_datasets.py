#!/usr/bin/env python3
"""
download_datasets.py — Download and prepare benchmark datasets.

Downloads:
  - SQuAD 1.1 QA pairs  →  benchmark/squad_bench_large/
  - BEIR subsets (scifact, nfcorpus, fiqa)  →  benchmark/beir/
  - BGE embed model (Q8) via huggingface-hub  →  models/embedding/

Usage:
  python scripts/download_datasets.py [--squad-only] [--beir-only] [--models-only]
"""

import argparse
import json
import os
import pathlib
import re
import sys
import urllib.request
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent


_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "not",
    "no", "it", "its", "they", "them", "their", "we", "you", "he", "she",
}


def _extract_keywords(answer_text: str, n: int = 3) -> list[str]:
    """Return up to n distinctive content words from an answer span."""
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*[A-Za-z]", answer_text)
    content = [t for t in tokens if t.lower() not in _STOPWORDS and len(t) >= 4]
    content.sort(key=len, reverse=True)
    seen: list[str] = []
    for t in content:
        if t.lower() not in {s.lower() for s in seen}:
            seen.append(t)
        if len(seen) >= n:
            break
    return seen or [answer_text.strip()]


def download_squad(out_dir: pathlib.Path) -> None:
    """Download SQuAD 1.1 dev set via direct URL and write qa_pairs.json + corpus docs."""
    SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
    cache_path = out_dir / ".cache" / "train-v1.1.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f"[squad] Using cached file: {cache_path}")
    else:
        print(f"[squad] Downloading SQuAD 1.1 from {SQUAD_URL} …")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SQUAD_URL, cache_path)
        print(f"[squad] Saved → {cache_path}")

    with open(cache_path, encoding="utf-8") as f:
        squad = json.load(f)

    qa_pairs: list[dict[str, Any]] = []
    corpus_docs: dict[str, str] = {}  # doc_key → context text

    for article in squad.get("data", []):
        raw_title: str = article.get("title", "unknown")
        doc_key = re.sub(r"[^A-Za-z0-9_-]", "_", raw_title)[:60]
        for para in article.get("paragraphs", []):
            context: str = para.get("context", "").strip()
            if len(context) < 50:
                continue
            # Concatenate all paragraphs so each file is the full article,
            # giving enough chunks for RAPTOR tree construction.
            if doc_key in corpus_docs:
                corpus_docs[doc_key] += "\n\n" + context
            else:
                corpus_docs[doc_key] = context
            for qa in para.get("qas", []):
                question: str = qa.get("question", "").strip()
                answers_raw = qa.get("answers", [])
                answers: list[str] = list(dict.fromkeys(
                    a["text"].strip() for a in answers_raw if a.get("text", "").strip()
                ))
                if not question or not answers:
                    continue
                keywords = _extract_keywords(answers[0])
                qa_pairs.append({
                    "question": question,
                    "relevant_keywords": keywords,
                    "doc_title": doc_key,
                    "answers": answers,
                })

    with open(out_dir / "qa_pairs.json", "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

    corpus_dir = out_dir / "corpus"
    corpus_dir.mkdir(exist_ok=True)
    for key, text in corpus_docs.items():
        (corpus_dir / f"{key}.txt").write_text(text, encoding="utf-8")

    print(f"[squad] Wrote {len(qa_pairs)} QA pairs  → {out_dir / 'qa_pairs.json'}")
    print(f"[squad] Wrote {len(corpus_docs)} corpus docs → {corpus_dir}")


def download_trivia_qa(out_dir: pathlib.Path, max_qa: int = 3000) -> None:
    """Download TriviaQA RC (Wikipedia domain, validation split) and write corpus/*.txt + qa_pairs.json.

    Uses ``trivia_qa`` 'rc' config from HuggingFace datasets.  Caps at *max_qa*
    completed QA pairs; loads up to ``max_qa * 2`` raw entries from the validation
    split (≈ 7,600 total) to account for entries with no Wikipedia context.
    Each Wikipedia article referenced by a question becomes one corpus document.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        sys.exit("ERROR: 'datasets' not installed. Run: pip install datasets")

    out_dir.mkdir(parents=True, exist_ok=True)
    # Load a generous slice — validation has ~7,600 entries; we cap at max_qa pairs
    raw_limit = min(max_qa * 2, 6000)
    print(f"[trivia] Downloading TriviaQA rc validation[:{raw_limit}] (target {max_qa} QA pairs) …")
    ds = load_dataset("trivia_qa", "rc", split=f"validation[:{raw_limit}]")

    qa_pairs: list[dict[str, Any]] = []
    corpus_docs: dict[str, str] = {}  # doc_key → concatenated Wikipedia text

    for entry in ds:
        if len(qa_pairs) >= max_qa:
            break

        question: str = entry["question"].strip()
        answer_val: str = entry["answer"]["value"].strip()
        aliases: list[str] = list(dict.fromkeys(
            a.strip() for a in entry["answer"]["aliases"] if a.strip()
        ))
        answers: list[str] = list(dict.fromkeys([answer_val] + aliases)) if answer_val else []
        if not question or not answers:
            continue

        # Build corpus from entity_pages (Wikipedia articles)
        pages = entry.get("entity_pages") or {}
        titles: list[str] = pages.get("title") or []
        contexts: list[str] = pages.get("wiki_context") or []

        doc_title_for_qa: str | None = None
        for title, context in zip(titles, contexts):
            context = (context or "").strip()
            if len(context) < 50:
                continue
            doc_key = re.sub(r"[^A-Za-z0-9_-]", "_", title)[:60]
            # Concatenate additional context for the same Wikipedia article
            if doc_key in corpus_docs:
                corpus_docs[doc_key] += "\n\n" + context
            else:
                corpus_docs[doc_key] = context
            if doc_title_for_qa is None:
                doc_title_for_qa = doc_key

        if doc_title_for_qa is None:
            continue  # no usable Wikipedia context for this question

        keywords = _extract_keywords(answers[0])
        qa_pairs.append({
            "question": question,
            "relevant_keywords": keywords,
            "doc_title": doc_title_for_qa,
            "answers": answers,
        })

    with open(out_dir / "qa_pairs.json", "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

    corpus_dir = out_dir / "corpus"
    corpus_dir.mkdir(exist_ok=True)
    for key, text in corpus_docs.items():
        (corpus_dir / f"{key}.txt").write_text(text, encoding="utf-8")

    print(f"[trivia] Wrote {len(qa_pairs)} QA pairs  → {out_dir / 'qa_pairs.json'}")
    print(f"[trivia] Wrote {len(corpus_docs)} corpus docs → {corpus_dir}")


def download_beir(dataset: str, out_dir: pathlib.Path) -> None:
    """Download a BEIR dataset via HuggingFace and write corpus.jsonl + queries.jsonl + qrels/test.tsv."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        sys.exit("ERROR: 'datasets' not installed. Run: pip install datasets")

    ds_dir = out_dir / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    print(f"[beir/{dataset}] Downloading …")

    # Corpus — JSONL, one doc per line with _id / title / text
    corpus_ds = load_dataset("BeIR/" + dataset, "corpus", split="corpus")
    with open(ds_dir / "corpus.jsonl", "w") as f:
        for row in corpus_ds:
            f.write(json.dumps({"_id": row["_id"], "title": row.get("title", ""), "text": row["text"]}) + "\n")

    # Queries — JSONL (rag_bench expects queries.jsonl, not .tsv)
    try:
        query_ds = load_dataset("BeIR/" + dataset, "queries", split="queries")
        with open(ds_dir / "queries.jsonl", "w") as f:
            for row in query_ds:
                f.write(json.dumps({"_id": row["_id"], "text": row["text"]}) + "\n")
    except Exception as e:
        print(f"[beir/{dataset}] WARNING: Could not download queries: {e}")

    # Qrels — rag_bench expects qrels/test.tsv with header "query-id\tcorpus-id\tscore"
    try:
        qrel_ds = load_dataset("BeIR/" + dataset + "-qrels", split="test")
        qrels_dir = ds_dir / "qrels"
        qrels_dir.mkdir(exist_ok=True)
        with open(qrels_dir / "test.tsv", "w") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            for row in qrel_ds:
                f.write(f"{row['query-id']}\t{row['corpus-id']}\t{row['score']}\n")
    except Exception as e:
        print(f"[beir/{dataset}] WARNING: Could not download qrels: {e}")

    print(f"[beir/{dataset}] Done → {ds_dir}")


def download_embed_models(models_dir: pathlib.Path) -> None:
    """Download BGE embedding models (Q8_0 GGUF) from HuggingFace hub."""
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        sys.exit("ERROR: 'huggingface-hub' not installed.")

    models = [
        {
            "repo": "CompendiumLabs/bge-base-en-v1.5-gguf",
            "file": "bge-base-en-v1.5-q8_0.gguf",
            "dir": models_dir / "embedding" / "bge-base-en-v1.5-q8_0",
        },
        {
            "repo": "CompendiumLabs/bge-small-en-v1.5-gguf",
            "file": "bge-small-en-v1.5-q8_0.gguf",
            "dir": models_dir / "embedding" / "bge-small-en-v1.5-q8_0",
        },
    ]
    for m in models:
        out = pathlib.Path(m["dir"])
        out.mkdir(parents=True, exist_ok=True)
        dest = out / m["file"]
        if dest.exists():
            print(f"[models] Skipping {m['file']} (already exists)")
            continue
        print(f"[models] Downloading {m['file']} from {m['repo']} …")
        hf_hub_download(repo_id=m["repo"], filename=m["file"], local_dir=str(out))
        print(f"[models] Saved → {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--squad-only",  action="store_true", help="Download SQuAD 1.1 only")
    parser.add_argument("--trivia-only", action="store_true", help="Download TriviaQA RC only")
    parser.add_argument("--beir-only",   action="store_true", help="Download BEIR subsets only")
    parser.add_argument("--models-only", action="store_true", help="Download embedding models only")
    args = parser.parse_args()

    run_all = not (args.squad_only or args.trivia_only or args.beir_only or args.models_only)

    if run_all or args.squad_only:
        download_squad(ROOT / "benchmark" / "squad_bench_large")

    if run_all or args.trivia_only:
        download_trivia_qa(ROOT / "benchmark" / "trivia_qa")

    if run_all or args.beir_only:
        beir_dir = ROOT / "benchmark" / "beir"
        # scifact / nfcorpus / fiqa — small enough for full BEIR-bench.
        # NOTE: 'nq' (Natural Questions) is also a valid BEIR dataset (BeIR/nq)
        #       but its corpus has ~2.68M documents and is impractical to ingest
        #       with the current beir-bench pipeline; excluded from default list.
        for dataset in ("scifact", "nfcorpus", "fiqa"):
            download_beir(dataset, beir_dir)

    if run_all or args.models_only:
        download_embed_models(ROOT / "models")


if __name__ == "__main__":
    main()
