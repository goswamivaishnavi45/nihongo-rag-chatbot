"""Step 5: generation. Retrieved chunks + question -> answer with citations (Gemini).

The chunks are numbered [1]..[k] in the prompt, so the model can cite them.
The system prompt is what keeps the model grounded: answer only from the
passages, cite them, and say "I don't know" instead of guessing.
"""
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

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


def build_prompt(question: str, chunks: list[Chunk], history: list[str] | None = None) -> str:
    context = "\n\n".join(f"[{i}] (source: {c.source})\n{c.text}" for i, c in enumerate(chunks, start=1))
    earlier = ""
    if history:  # lets the model understand follow-ups like "where do I apply?"
        earlier = "Earlier messages from the user, oldest first:\n" + "\n".join(f"- {h}" for h in history) + "\n\n"
    return f"Context passages:\n\n{context}\n\n{earlier}Question: {question}"


def call_model(system: str, contents: str) -> str:
    # Google's servers occasionally return temporary 5xx or 429 rate-limit errors.
    # Retry with exponential backoff.
    for attempt in range(4):
        try:
            response = get_client().models.generate_content(
                model=MODEL, contents=contents, config=types.GenerateContentConfig(system_instruction=system)
            )
            break
        except (errors.ServerError, errors.ClientError) as e:
            # 429 rate limit / quota exhaustion can be retried with a wait
            is_429 = getattr(e, "code", None) == 429 or "429" in str(e)
            if isinstance(e, errors.ClientError) and not is_429:
                raise
            if attempt == 3:
                raise
            time.sleep(3 * (2**attempt))  # wait 3s, 6s, 12s
    # .text is None when the model returns no text (e.g. blocked by a safety filter)
    return response.text or ""


# Query rewriting. Users write casually ("I changed my job"); the guidebook uses formal
# administrative Japanese (所属機関に関する届出). Rewriting the question into the document's
# style before searching closes that gap, and resolves follow-ups using earlier messages.
# The rules are deliberately general: putting specific test-set wordings here would
# inflate the evaluation score without helping real users (overfitting to the test set).
REWRITE_PROMPT = """You turn a user's message into a search query for an official Japanese government guidebook for foreign residents living in Japan.

Rules:
- Write ONE standalone search query in Japanese, using the formal administrative vocabulary that official Japanese government documents use.
- If the message is a follow-up ("where do I apply?", "how much is it?"), use the earlier messages to make the query complete.
- Keep the user's actual intent. Do not answer the question.
- Reply with only the query, nothing else."""


def rewrite_query(question: str, history: list[str] | None = None) -> str:
    earlier = "".join(f"Earlier message: {h}\n" for h in history or [])
    rewritten = call_model(REWRITE_PROMPT, f"{earlier}Latest message: {question}").strip()
    return rewritten or question  # if the rewrite comes back empty, search with the original


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


def generate(
    question: str,
    chunks: list[Chunk],
    short: bool = False,
    topics: list[str] | None = None,
    history: list[str] | None = None,
) -> str:
    system = SYSTEM_PROMPT + (SHORT_ANSWER_RULE if short else "") + (topics_rule(topics) if topics else "")
    return call_model(system, build_prompt(question, chunks, history)) or "(no answer returned)"


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
