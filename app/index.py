"""Step 4a: the vector index (FAISS).

FAISS stores only vectors, by position (0, 1, 2, ...). We keep the chunks in
a list in the same order, so FAISS position i is self.chunks[i].
On disk an index is three files:
    index.faiss   the vectors
    chunks.json   the chunk texts and sources, same order
    meta.json     which embedding model built it (checked on load)
"""
import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from app.chunking import Chunk
from app.embeddings import MODEL_NAME, embed_passages, embed_query


class VectorIndex:
    def __init__(self, chunks: list[Chunk], index: faiss.Index):
        self.chunks = chunks
        self.index = index

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "VectorIndex":
        vectors = embed_passages([c.text for c in chunks])  # shape (n_chunks, 384)
        # Flat = exact search over every vector. IP = inner (dot) product, which
        # equals cosine similarity because our vectors are normalized.
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors.astype(np.float32))
        return cls(chunks, index)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        (directory / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False), encoding="utf-8"
        )
        (directory / "meta.json").write_text(json.dumps({"model": MODEL_NAME, "chunks": len(self.chunks)}))

    @classmethod
    def load(cls, directory: Path) -> "VectorIndex":
        meta = json.loads((directory / "meta.json").read_text())
        # Vectors from different models live in different "coordinate systems",
        # so comparing them gives garbage without any error. Refuse instead.
        if meta["model"] != MODEL_NAME:
            raise ValueError(
                f"Index was built with {meta['model']} but the app uses {MODEL_NAME}. Rebuild the index."
            )
        chunks = [Chunk(**c) for c in json.loads((directory / "chunks.json").read_text(encoding="utf-8"))]
        return cls(chunks, faiss.read_index(str(directory / "index.faiss")))

    def search_positions(self, question: str, k: int = 5) -> tuple[list[int], list[float]]:
        """Positions of the top-k chunks (best first) and their scores. Used by hybrid search."""
        q = embed_query(question).astype(np.float32).reshape(1, -1)  # FAISS expects a batch: shape (1, 384)
        scores, positions = self.index.search(q, k)
        return positions[0].tolist(), scores[0].tolist()

    def search(self, question: str, k: int = 5) -> list[tuple[Chunk, float]]:
        positions, scores = self.search_positions(question, k)
        return [(self.chunks[p], s) for p, s in zip(positions, scores)]


if __name__ == "__main__":
    # Mini test on JSQuAD: is the paragraph each question came from in the top 5?
    import sys

    from app.jsquad import load_jsquad, load_or_build_index, sample_questions

    sys.stdout.reconfigure(encoding="utf-8")

    chunks, questions = load_jsquad()
    index = load_or_build_index(chunks)
    print(f"Index: {len(index.chunks)} chunks")
    sample = sample_questions(questions, 50)

    hits_at_1 = hits_at_5 = 0
    for question, gold_source, _ in sample:
        results = index.search(question, k=5)
        sources = [chunk.source for chunk, _ in results]
        hits_at_1 += sources[0] == gold_source
        hits_at_5 += gold_source in sources

    print(f"\nRecall@1: {hits_at_1}/50 = {hits_at_1 / 50:.0%}   (correct paragraph ranked first)")
    print(f"Recall@5: {hits_at_5}/50 = {hits_at_5 / 50:.0%}   (correct paragraph anywhere in top 5)")
