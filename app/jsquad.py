"""Load JSQuAD (Japanese Wikipedia QA, CC BY-SA 4.0) as chunks plus labelled questions.

Each question is paired with the source of the paragraph it was written from,
so we can check retrieval automatically: is that paragraph in the top k?
"""
import json
import random
from pathlib import Path

from app.chunking import Chunk, chunk_text

ROOT = Path(__file__).resolve().parent.parent
JSQUAD_FILE = ROOT / "data" / "eval" / "jsquad-valid-v1.3.json"
INDEX_DIR = ROOT / "data" / "index" / "jsquad"


def load_jsquad() -> tuple[list[Chunk], list[tuple[str, str, list[str]]]]:
    """Returns (chunks, questions). Each question is (question text, source of its paragraph, correct answers)."""
    data = json.loads(JSQUAD_FILE.read_text(encoding="utf-8"))["data"]
    chunks, questions = [], []
    for article in data:
        for i, para in enumerate(article["paragraphs"]):
            source = f"{article['title']}/p{i}"
            # Paragraphs start with "title [SEP] ". Use "title: " so the title stays a prefix
            # (a line break would make join_wrapped_lines() glue it onto the next line).
            text = para["context"].replace(" [SEP] ", ": ")
            chunks += chunk_text(text, source=source)
            questions += [(qa["question"], source, [a["text"] for a in qa["answers"]]) for qa in para["qas"]]
    return chunks, questions


def load_or_build_index(chunks: list[Chunk]):
    """Load the saved JSQuAD index, or embed everything and save it (takes about a minute)."""
    from app.index import VectorIndex

    if (INDEX_DIR / "meta.json").exists():
        return VectorIndex.load(INDEX_DIR)
    print(f"Embedding {len(chunks)} chunks (one time only)...")
    index = VectorIndex.build(chunks)
    index.save(INDEX_DIR)
    return index


def sample_questions(questions: list, n: int) -> list:
    """The same n questions every run (fixed seed), spread across all articles."""
    return random.Random(0).sample(questions, n)
