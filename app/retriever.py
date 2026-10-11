"""Step 4b: keyword search (BM25 + MeCab) and hybrid search (RRF).

Semantic search (FAISS) finds chunks by meaning; BM25 finds them by exact
words. Hybrid search runs both and fuses the two rankings with Reciprocal
Rank Fusion, which uses only ranks, since the two kinds of score aren't comparable.
"""
import numpy as np
from fugashi import Tagger
from rank_bm25 import BM25Okapi

from app.chunking import Chunk, normalize
from app.index import VectorIndex

# Grammar words that appear in nearly every sentence and carry no topic:
# particles (を, は), auxiliary verbs (た, ます), punctuation and spaces.
SKIP_POS = {"助詞", "助動詞", "補助記号", "記号", "空白"}

_tagger = Tagger()  # MeCab, with the unidic-lite dictionary


def tokenize(text: str) -> list[str]:
    """在留カードを紛失した -> [在留, カード, 紛失, する]"""
    tokens = []
    for word in _tagger(normalize(text)):
        if word.feature.pos1 in SKIP_POS:
            continue
        # The lemma (dictionary form) makes なくした / なくして / なくす the same token.
        # Words missing from the dictionary have no lemma, so fall back to the word as written.
        tokens.append(word.feature.lemma or word.surface)
    return tokens


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion: score = sum of 1 / (k + rank) over the rankings that contain the item."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, position in enumerate(ranking, start=1):
            scores[position] = scores.get(position, 0.0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


class Retriever:
    def __init__(self, vector_index: VectorIndex):
        self.vector_index = vector_index
        self.chunks = vector_index.chunks
        # BM25 is fast to build (a few seconds), so we rebuild it at startup instead of saving it.
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

    def semantic(self, question: str, k: int) -> list[int]:
        positions, _ = self.vector_index.search_positions(question, k)
        return positions

    def keyword(self, question: str, k: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(question))
        best = np.argsort(-scores)[:k]  # positions sorted by score, highest first
        return [int(p) for p in best if scores[p] > 0]  # score 0 = no shared words = not found

    def search(self, question: str, k: int = 5, mode: str = "hybrid", candidates: int = 20) -> list[Chunk]:
        if mode == "semantic":
            positions = self.semantic(question, k)
        elif mode == "keyword":
            positions = self.keyword(question, k)
        else:
            # Each search gives its top 20, so a chunk ranked 6th in one list still gets its vote.
            positions = rrf([self.semantic(question, candidates), self.keyword(question, candidates)])[:k]
        return [self.chunks[p] for p in positions]

    def search_many(self, queries: list[str], k: int = 5, candidates: int = 20) -> list[Chunk]:
        """Hybrid search with several phrasings of the same question (e.g. the user's words
        and a rewritten query), all fused with RRF. If one phrasing misses, the others still vote."""
        rankings = []
        for query in dict.fromkeys(queries):  # skip duplicates, keep order
            rankings += [self.semantic(query, candidates), self.keyword(query, candidates)]
        return [self.chunks[p] for p in rrf(rankings)[:k]]


if __name__ == "__main__":
    import sys

    from app.jsquad import load_jsquad, load_or_build_index, sample_questions

    sys.stdout.reconfigure(encoding="utf-8")

    print("--- MeCab tokenize ---")
    for s in ["在留カードを紛失した", "在留カードをなくしました"]:
        print(f"{s}  ->  {tokenize(s)}")

    chunks, questions = load_jsquad()
    retriever = Retriever(load_or_build_index(chunks))

    for n in [50, 500]:
        sample = sample_questions(questions, n)
        print(f"\n--- {n} questions ---")
        print(f"{'mode':10} {'Recall@1':>9} {'Recall@5':>9}")
        for mode in ["semantic", "keyword", "hybrid"]:
            at_1 = at_5 = 0
            for question, gold_source, _ in sample:
                sources = [c.source for c in retriever.search(question, k=5, mode=mode)]
                at_1 += bool(sources) and sources[0] == gold_source
                at_5 += gold_source in sources
            print(f"{mode:10} {at_1 / n:>9.1%} {at_5 / n:>9.1%}")
