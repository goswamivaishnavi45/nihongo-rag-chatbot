"""Step 6: end-to-end evaluation on JSQuAD.

For each question: retrieve the top 5 chunks (hybrid), ask Gemini for a short
answer, and compare it with JSQuAD's correct answers.

    Recall@5      was the correct paragraph retrieved?
    Exact Match   is the answer exactly a correct answer (after normalization)?
    Character F1  how many characters overlap (partial credit). Characters, not
                  words, because Japanese has no spaces between words.

Usage:  python -m app.evaluate [number of questions, default 200]
Every answer is saved to data/eval/results_<model>_<n>.jsonl for spot checks.
"""
import json
import re
import sys
import time
import unicodedata
from collections import Counter

from google.genai import errors

from app.generator import MODEL, generate
from app.jsquad import ROOT, load_jsquad, load_or_build_index, sample_questions
from app.retriever import Retriever


def normalize_answer(text: str) -> str:
    """NFKC, then drop citations like [2], punctuation and spaces: '小笠原諸島です。[2]' -> '小笠原諸島です'"""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\[\d+\]", "", text)
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith(("P", "Z")))


def char_f1(prediction: str, gold: str) -> float:
    pred, gold = normalize_answer(prediction), normalize_answer(gold)
    common = sum((Counter(pred) & Counter(gold)).values())  # characters shared, counting repeats
    if common == 0:
        return 0.0
    precision = common / len(pred)  # how much of the answer is correct
    recall = common / len(gold)     # how much of the correct answer we gave
    return 2 * precision * recall / (precision + recall)


def is_dont_know(answer: str) -> bool:
    return "わかりません" in answer or "don't know" in answer.lower()


def generate_with_retry(question, chunks, attempts=6) -> str:
    """The free tier limits requests per minute; on 429/503, wait and try again.
    A per-DAY quota won't reset for hours, so stop instead of retrying."""
    for attempt in range(attempts):
        try:
            return generate(question, chunks, short=True)
        except errors.APIError as e:
            if "PerDay" in str(e):
                sys.exit("\nDaily API quota used up. Answers so far are saved; rerun later to continue.")
            if getattr(e, "code", None) not in (429, 503) or attempt == attempts - 1:
                raise
            wait = 10 * 2**attempt
            print(f"   rate limited ({e.code}), waiting {wait}s...")
            time.sleep(wait)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200

    chunks, questions = load_jsquad()
    retriever = Retriever(load_or_build_index(chunks))
    sample = sample_questions(questions, n)

    # Each answer is appended to the file as soon as we have it, so a crash or quota
    # stop loses nothing, and a rerun skips questions that are already answered.
    out = ROOT / "data" / "eval" / f"results_{MODEL}_{n}.jsonl"
    results = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()] if out.exists() else []
    if results:
        print(f"Resuming: {len(results)}/{n} already answered")

    for i, (question, gold_source, gold_answers) in enumerate(sample, start=1):
        if i <= len(results):
            continue
        top = retriever.search(question, k=5)
        answer = generate_with_retry(question, top)
        result = {
            "question": question,
            "gold_answers": gold_answers,
            "answer": answer,
            "retrieved": gold_source in [c.source for c in top],
            "exact_match": max(normalize_answer(answer) == normalize_answer(g) for g in gold_answers),
            "char_f1": max(char_f1(answer, g) for g in gold_answers),
            "dont_know": is_dont_know(answer),
        }
        results.append(result)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        if i % 20 == 0:
            print(f"{i}/{n} done")

    def summary(rows, label):
        if not rows:
            return
        k = len(rows)
        print(f"{label:34} n={k:<4} EM={sum(r['exact_match'] for r in rows) / k:6.1%}  "
              f"charF1={sum(r['char_f1'] for r in rows) / k:6.1%}  "
              f"don't-know={sum(r['dont_know'] for r in rows) / k:5.1%}")

    print(f"\nModel: {MODEL}   Retrieval: hybrid top 5")
    print(f"Recall@5: {sum(r['retrieved'] for r in results) / n:.1%}\n")
    summary(results, "All questions")
    summary([r for r in results if r["retrieved"]], "  correct paragraph retrieved")
    summary([r for r in results if not r["retrieved"]], "  correct paragraph NOT retrieved")
    print(f"\nEvery answer saved to {out.relative_to(ROOT)}")
