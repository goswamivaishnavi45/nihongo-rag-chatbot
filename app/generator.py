"""Step 5: generation. Retrieved chunks + question -> answer with citations (Gemini).

The chunks are numbered [1]..[k] in the prompt, so the model can cite them.
The system prompt is what keeps the model grounded: answer only from the
passages, cite them, and say "I don't know" instead of guessing.
"""
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.chunking import Chunk

load_dotenv()  # reads GEMINI_API_KEY from .env, so the key never appears in the code

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages you are given.

Rules:
- Use only facts stated in the passages. Do not use outside knowledge, even if you know the answer.
- After each fact, cite the passage it came from in square brackets, like [2].
- If the passages do not contain the answer, do not guess. Reply "わかりません。" if the question is in Japanese, or "I don't know based on the documents." if it is in English.
- Answer in the same language as the question.
- Keep the answer short and direct."""

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client()  # picks up GEMINI_API_KEY from the environment
    return _client


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(f"[{i}] (source: {c.source})\n{c.text}" for i, c in enumerate(chunks, start=1))
    return f"Context passages:\n\n{context}\n\nQuestion: {question}"


# Evaluation only: JSQuAD's correct answers are short phrases (小笠原諸島), so a full
# sentence with citations would never match exactly, even when it's right.
SHORT_ANSWER_RULE = """
- EVALUATION MODE: reply with ONLY the shortest exact phrase from the passages that answers the question. No sentence, no citation, no punctuation."""


def topics_rule(topics: list[str]) -> str:
    """Lets the chatbot answer questions about itself ("what do you know?") instead of saying "I don't know"."""
    return (
        "\n- If the user asks what you know, what topics you cover, or what you can help with, "
        "don't use the passages: say you answer questions from these document topics and list them: "
        + ", ".join(topics)
    )


def generate(question: str, chunks: list[Chunk], short: bool = False, topics: list[str] | None = None) -> str:
    system = SYSTEM_PROMPT + (SHORT_ANSWER_RULE if short else "") + (topics_rule(topics) if topics else "")
    response = get_client().models.generate_content(
        model=MODEL,
        contents=build_prompt(question, chunks),
        config=types.GenerateContentConfig(system_instruction=system),
    )
    # .text is None when the model returns no text (e.g. blocked by a safety filter)
    return response.text or "(no answer returned)"


if __name__ == "__main__":
    import sys

    from app.jsquad import load_jsquad, load_or_build_index
    from app.retriever import Retriever

    sys.stdout.reconfigure(encoding="utf-8")

    chunks, _ = load_jsquad()
    retriever = Retriever(load_or_build_index(chunks))

    tests = [
        ("日本で梅雨がないのは北海道とどこか。", "JSQuAD question; correct answer is 小笠原諸島 (Ogasawara Islands)"),
        ("Which parts of Japan have no rainy season?", "same question in English: cross-lingual test"),
        ("How much does it cost to renew a work visa in Tokyo?", "NOT in the documents: should say I don't know"),
    ]
    for question, note in tests:
        top = retriever.search(question, k=5)
        print(f"\nQ: {question}\n   ({note})")
        print(f"   retrieved: {[c.source for c in top]}")
        print(f"A: {generate(question, top)}")
