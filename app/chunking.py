"""Step 2: Japanese-aware text normalization and chunking.

Japanese has no spaces between words, so we can't split on whitespace.
Instead we split the text into sentences on 。 ! ? and line breaks, then pack
whole sentences into chunks of about CHUNK_SIZE characters, repeating the
last sentence of each chunk at the start of the next one (overlap).
"""
import re
import unicodedata
from dataclasses import dataclass

# A sentence ends right after 。 ! ? or at a line break.
# (After NFKC normalization, full-width ！？ have already become ! ?)
SENTENCE_END = re.compile(r"(?<=[。!?])|\n+")

# A line break that is NOT after 。!? and NOT part of a blank line is just a
# page wrap (common in PDFs), so the two lines belong to the same sentence.
WRAPPED_LINE = re.compile(r"(?<![。!?\n])\n(?!\n)")


@dataclass
class Chunk:
    id: str      # e.g. "visa_guide.txt#3"
    source: str  # the file this chunk came from
    text: str


def normalize(text: str) -> str:
    """Make equivalent characters identical: ＡＢＣ１２３ -> ABC123, ｶﾀｶﾅ -> カタカナ."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n")    # Windows line endings
    text = re.sub(r"[ \t]+", " ", text)  # collapse runs of spaces/tabs
    text = re.sub(r" *\n *", "\n", text)  # drop spaces around line breaks
    return text.strip()


def join_wrapped_lines(text: str) -> str:
    """Rejoin sentences that a PDF split across lines: '14日以内に\\n届け出て' -> '14日以内に届け出て'.

    Japanese needs no space when joining. Trade-off: headings and bullet points
    rarely end in 。, so they get glued onto the next line.
    """
    return WRAPPED_LINE.sub("", text)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


def cut_long_sentences(sentences: list[str], chunk_size: int) -> list[str]:
    """Edge case: a sentence longer than a whole chunk is cut into pieces.

    The pieces are full-size, so sentence overlap can't be used between them.
    Instead each piece repeats the last chunk_size // 8 characters of the
    previous one (50 characters at the default size of 400).
    """
    overlap = chunk_size // 8
    step = chunk_size - overlap
    pieces = []
    for sent in sentences:
        if len(sent) <= chunk_size:
            pieces.append(sent)
            continue
        for start in range(0, len(sent), step):
            pieces.append(sent[start:start + chunk_size])
            if start + chunk_size >= len(sent):  # this piece reached the end
                break
    return pieces


def chunk_text(text: str, source: str, chunk_size: int = 400) -> list[Chunk]:
    text = join_wrapped_lines(normalize(text))
    sentences = cut_long_sentences(split_sentences(text), chunk_size)

    chunks: list[Chunk] = []
    box: list[str] = []  # sentences in the chunk we're currently filling
    box_len = 0

    for sent in sentences:
        # The next sentence doesn't fit: close this chunk and start a new one.
        if box and box_len + len(sent) > chunk_size:
            chunks.append(Chunk(f"{source}#{len(chunks)}", source, "".join(box)))
            box = [box[-1]]  # overlap: carry the last sentence over
            box_len = len(box[0])
            if box_len + len(sent) > chunk_size:  # overlap + next sentence too big: skip overlap
                box, box_len = [], 0
        box.append(sent)
        box_len += len(sent)

    if box:
        chunks.append(Chunk(f"{source}#{len(chunks)}", source, "".join(box)))
    return chunks


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # so Windows terminals can print Japanese

    SAMPLE = (
        "在留カードは常に携帯してください。"                          # Always carry your residence card.
        "紛失した場合は１４日以内に届け出てください。"                  # If you lose it, report within 14 days.  (note full-width １４)
        "引っ越しをした場合は、新しい住所地の市区町村役場で届け出が必要です。"  # If you move, register at your new city office.
        "届け出の期限は、引っ越してから１４日以内です。"                # The deadline is 14 days after moving.
        "ご不明な点は、出入国在留管理庁にお問い合わせください！"          # Questions? Contact the Immigration Services Agency!
    )

    print("--- normalize ---")
    print("before:", "１４日以内", "  after:", normalize("１４日以内"))

    print("\n--- sentences ---")
    for s in split_sentences(normalize(SAMPLE)):
        print(f"{len(s):3d} chars | {s}")

    print("\n--- chunks (chunk_size=60, small so we get several) ---")
    for c in chunk_text(SAMPLE, source="sample.txt", chunk_size=60):
        print(f"{c.id}  ({len(c.text)} chars)\n    {c.text}")

    # Fix 1: text extracted from a PDF, where one sentence wrapped onto two lines.
    PDF_TEXT = (
        "在留カードを紛失した場合は、14日以内に\n"  # line wraps mid-sentence here
        "出入国在留管理局に届け出てください。\n"
        "\n"                                      # blank line = paragraph break
        "手数料は無料です。"                       # The fee is free.
    )
    print("\n--- fix 1: PDF line wraps ---")
    print("BEFORE (every line break splits):")
    for s in split_sentences(normalize(PDF_TEXT)):
        print("   ", s)
    print("AFTER (wrapped lines joined first):")
    for s in split_sentences(join_wrapped_lines(normalize(PDF_TEXT))):
        print("   ", s)

    # Fix 2: one sentence longer than the chunk size, cut into overlapping pieces.
    LONG = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよ。"
    print(f"\n--- fix 2: a {len(LONG)}-char sentence, chunk_size=16 (overlap = 16 // 8 = 2) ---")
    for c in chunk_text(LONG, source="long.txt", chunk_size=16):
        print(f"{c.id}  {c.text}")
