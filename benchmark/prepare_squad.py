#!/usr/bin/env python3
"""
prepare_squad.py – Download a subset of SQuAD 1.1 and prepare it as a
NELA rag-bench corpus.

Output layout
─────────────
  <out_dir>/
    docs/
      <title>_<idx>.txt    ← one file per context paragraph
    qa_pairs.json           ← QA pairs in rag-bench format

QA pairs JSON schema
────────────────────
  [
    {
      "question": "...",
      "relevant_keywords": ["word1", "word2"],   <- key words from the answer span
      "doc_title": "<title>_<idx>"               <- filename stem of the source doc
    },
    ...
  ]

Usage
─────
  python3 prepare_squad.py                     # uses defaults
  python3 prepare_squad.py --out-dir ./squad --max-contexts 80 --max-qa 300
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

# ── SQuAD source ──────────────────────────────────────────────────────────────
SQUAD_URL = (
    "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
)

# Common English stopwords (trimmed list for keyword extraction)
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "not",
    "no", "nor", "so", "yet", "both", "either", "neither", "than", "that",
    "this", "which", "who", "whom", "whose", "what", "when", "where",
    "why", "how", "it", "its", "they", "them", "their", "we", "our",
    "you", "your", "he", "his", "she", "her", "i", "my", "me", "us",
}


def extract_keywords(answer_text: str, n: int = 3) -> list[str]:
    """Return up to *n* content words from the answer span.

    Strategy:
    1. Tokenise (split on non-alphanumeric boundaries).
    2. Remove stopwords and short tokens (< 4 chars).
    3. Return the longest remaining words (most distinctive).
    """
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*[A-Za-z]", answer_text)
    content = [t for t in tokens if t.lower() not in STOPWORDS and len(t) >= 4]
    # Sort by length descending (longer = more distinctive), then take top n
    content.sort(key=len, reverse=True)
    seen: list[str] = []
    for t in content:
        if t.lower() not in {s.lower() for s in seen}:
            seen.append(t)
        if len(seen) >= n:
            break
    # Fall back to the full answer text if no content words were extracted
    if not seen:
        seen = [answer_text.strip()]
    return seen


def sanitise_title(raw: str) -> str:
    """Turn an article title into a safe filename stem."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", raw)[:50]


def download_squad(cache_path: Path) -> dict:
    if cache_path.exists():
        print(f"[prepare] Using cached SQuAD file: {cache_path}")
    else:
        print(f"[prepare] Downloading SQuAD dev set from:\n  {SQUAD_URL}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SQUAD_URL, cache_path)
        print(f"[prepare] Saved → {cache_path}")

    with open(cache_path, encoding="utf-8") as f:
        return json.load(f)


def build_corpus(
    data: dict,
    out_dir: Path,
    max_contexts: int,
    max_qa: int,
) -> None:
    docs_dir = out_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    qa_pairs: list[dict] = []
    context_count = 0

    for article in data.get("data", []):
        raw_title = article.get("title", "unknown")
        title_stem = sanitise_title(raw_title)

        for para_idx, para in enumerate(article.get("paragraphs", [])):
            if context_count >= max_contexts:
                break

            context: str = para.get("context", "").strip()
            if len(context) < 100:
                continue  # skip very short paragraphs

            doc_title = f"{title_stem}_{para_idx:03d}"
            doc_path = docs_dir / f"{doc_title}.txt"

            # Write context as a plain-text document
            doc_path.write_text(context, encoding="utf-8")
            context_count += 1

            # Convert each QA pair in this paragraph
            for qa in para.get("qas", []):
                if len(qa_pairs) >= max_qa:
                    break

                question: str = qa.get("question", "").strip()
                if not question:
                    continue

                answers = qa.get("answers", [])
                if not answers:
                    continue
                # Use the first (usually canonical) answer
                answer_text: str = answers[0].get("text", "").strip()
                if not answer_text:
                    continue

                keywords = extract_keywords(answer_text)

                # Collect all unique answer texts for E2E exact-match / F1 evaluation
                all_answers = list(
                    dict.fromkeys(
                        a.get("text", "").strip()
                        for a in answers
                        if a.get("text", "").strip()
                    )
                )

                qa_pairs.append(
                    {
                        "question": question,
                        "relevant_keywords": keywords,
                        "doc_title": doc_title,
                        "answers": all_answers,
                    }
                )

            if len(qa_pairs) >= max_qa:
                break

        if context_count >= max_contexts or len(qa_pairs) >= max_qa:
            break

    qa_path = out_dir / "qa_pairs.json"
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

    print(
        f"[prepare] Done.\n"
        f"  Documents : {context_count} files in {docs_dir}\n"
        f"  QA pairs  : {len(qa_pairs)} entries in {qa_path}\n"
        f"\n"
        f"Next steps:\n"
        f"  1. Start the NELA embedding model server (or let rag-bench do it).\n"
        f"  2. Run:  cargo run --bin rag-bench -- run \\\n"
        f"             --workspace-dir /tmp/nela-bench \\\n"
        f"             --corpus-dir    {docs_dir} \\\n"
        f"             --qa-file       {qa_path} \\\n"
        f"             --embed-model   <path/to/bge.gguf>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a SQuAD-based corpus for rag-bench."
    )
    parser.add_argument(
        "--out-dir",
        default="squad_bench",
        help="Output directory for docs/ and qa_pairs.json (default: squad_bench/)",
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=100,
        help="Maximum number of context paragraphs to use as documents (default: 100)",
    )
    parser.add_argument(
        "--max-qa",
        type=int,
        default=400,
        help="Maximum number of QA pairs to include (default: 400)",
    )
    parser.add_argument(
        "--cache-dir",
        default=".squad_cache",
        help="Directory to cache the raw SQuAD JSON download (default: .squad_cache/)",
    )
    args = parser.parse_args()

    cache_path = Path(args.cache_dir) / "dev-v1.1.json"
    squad_data = download_squad(cache_path)

    build_corpus(
        data=squad_data,
        out_dir=Path(args.out_dir),
        max_contexts=args.max_contexts,
        max_qa=args.max_qa,
    )


if __name__ == "__main__":
    main()
