"""Realistic user questions: does retrieval find the right guidebook page?

The JSQuAD benchmark is easy (its questions reuse the paragraph's wording). These are
questions written the way a foreign resident would actually ask: casual Japanese,
casual English, and follow-ups that only make sense with the previous message.
Each has the guidebook page(s) that answer it (data/eval/realistic_questions.json).

Usage:  python -m app.realistic_eval              (search with the question only)
        python -m app.realistic_eval --rewrite    (also search with an LLM-rewritten query)
"""
import json
import sys
from collections import defaultdict

from app.documents import GUIDEBOOK_TITLE, load_or_build_app_index
from app.jsquad import ROOT
from app.retriever import Retriever

QUESTIONS_FILE = ROOT / "data" / "eval" / "realistic_questions.json"


def retrieved_pages(chunks) -> list[int]:
    """'生活・就労ガイドブック/p17' -> 17 (Wikipedia chunks are ignored)."""
    return [int(c.source.split("/p")[1]) for c in chunks if c.source.startswith(GUIDEBOOK_TITLE)]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    use_rewrite = "--rewrite" in sys.argv
    if use_rewrite:
        from app.generator import rewrite_query
    items = [q for q in json.loads(QUESTIONS_FILE.read_text(encoding="utf-8")) if "expected_pages" in q]
    retriever = Retriever(load_or_build_app_index())
    print(f"Mode: {'question + rewritten query' if use_rewrite else 'question only (baseline)'}\n")

    hits = defaultdict(list)
    for item in items:
        history = item.get("history", [])
        if use_rewrite:
            rewritten = rewrite_query(item["question"], history)
            chunks = retriever.search_many([item["question"], rewritten], k=5)
        else:
            rewritten = ""
            chunks = retriever.search(item["question"], k=5)
        pages = retrieved_pages(chunks)
        hit = any(p in item["expected_pages"] for p in pages)
        hits[item["type"]].append(hit)
        print(f"{'✅' if hit else '❌'} [{item['type']:8}] {item['question'][:40]:40} -> pages {pages}")
        if rewritten:
            print(f"      rewritten: {rewritten}")

    print()
    for kind, results in hits.items():
        print(f"{kind:9} Recall@5: {sum(results)}/{len(results)} = {sum(results) / len(results):.0%}")
    total = [h for results in hits.values() for h in results]
    print(f"{'overall':9} Recall@5: {sum(total)}/{len(total)} = {sum(total) / len(total):.0%}")
