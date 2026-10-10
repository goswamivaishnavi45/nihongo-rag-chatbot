"""Step 3: turn text into vectors (embeddings).

multilingual-e5 maps text in ~100 languages to 384 numbers, so that texts
with similar meaning get similar vectors, even across languages.
It was trained with a "query: " label in front of questions and a
"passage: " label in front of documents, so we must add them too.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"

_model = None  # loaded once on first use (loading takes a few seconds)


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _encode(texts: list[str]) -> np.ndarray:
    # normalize_embeddings=True scales every vector to length 1, so
    # cosine similarity becomes a simple dot product.
    return get_model().encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def embed_passages(texts: list[str]) -> np.ndarray:
    """Document chunks -> array of shape (len(texts), 384)."""
    return _encode(["passage: " + t for t in texts])


def embed_query(text: str) -> np.ndarray:
    """A user question -> array of shape (384,)."""
    return _encode(["query: " + text])[0]


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    question = "在留カードをなくしました。どうすればいいですか？"  # I lost my residence card. What should I do?
    passages = {
        "A": "在留カードを紛失した場合は、14日以内に届け出てください。",  # If you lose your residence card, report within 14 days.
        "B": "手数料は無料です。",                                        # The fee is free.
        "C": "富士山の高さは3,776メートルです。",                          # Mt. Fuji is 3,776 m tall.
        "D": "I lost my residence card, what do I do?",                  # English
    }

    q_vec = embed_query(question)
    p_vecs = embed_passages(list(passages.values()))

    print(f"question vector: {q_vec.shape[0]} numbers, first 5 = {np.round(q_vec[:5], 3)}")
    print(f"vector length (should be 1.0): {np.linalg.norm(q_vec):.3f}\n")

    scores = p_vecs @ q_vec  # dot product of each passage with the question = cosine similarity
    for label, score in sorted(zip(passages, scores), key=lambda x: -x[1]):
        print(f"{label}  {score:.3f}  {passages[label]}")
