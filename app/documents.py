"""Real-world documents: the Immigration Services Agency's guidebook for foreign residents.

生活・就労ガイドブック (Guidebook on Living and Working), Japanese edition, published by
出入国在留管理庁. We index the Japanese PDF. The English edition has the same page
layout, so it works as an answer key: a citation of page N can be checked on page N
of the English PDF. The PDFs aren't stored in git; they're downloaded from the official site.
"""
import re
import urllib.request
from collections import Counter

from app.chunking import Chunk, chunk_text
from app.jsquad import ROOT, load_jsquad

GUIDEBOOK_TITLE = "生活・就労ガイドブック"
GUIDEBOOK_URL = "https://www.moj.go.jp/isa/content/930004717.pdf"  # Japanese edition
GUIDEBOOK_PDF = ROOT / "data" / "docs" / "guidebook_ja.pdf"
APP_INDEX_DIR = ROOT / "data" / "index" / "app"


def ensure_guidebook() -> None:
    if not GUIDEBOOK_PDF.exists():
        print(f"Downloading {GUIDEBOOK_URL} ...")
        GUIDEBOOK_PDF.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(GUIDEBOOK_URL, headers={"User-Agent": "Mozilla/5.0"})
        GUIDEBOOK_PDF.write_bytes(urllib.request.urlopen(request).read())


def extract_pages(pdf_path) -> list[str]:
    """Text of each page, with boilerplate removed.

    Boilerplate = lines repeated on many pages (the side menu 第１章…第12章 appears on
    142 of 157 pages) plus lines that are only a page number. Real content never
    repeats on 20%+ of pages, so this is safe to drop.
    """
    from pypdf import PdfReader

    pages = [page.extract_text() or "" for page in PdfReader(pdf_path).pages]
    line_counts = Counter(line.strip() for page in pages for line in set(page.splitlines()))
    boilerplate = {line for line, n in line_counts.items() if n >= 0.2 * len(pages)}
    cleaned = []
    for page in pages:
        # Table-of-contents pages ("交通ルール ......... 96") only list section titles,
        # which would compete with the real content in search, so skip them (keep page numbering).
        if len(re.findall(r"\.{5,}", page)) >= 5:
            cleaned.append("")
            continue
        lines = [l for l in page.splitlines() if l.strip() not in boilerplate and not re.fullmatch(r"\s*\d+\s*", l)]
        cleaned.append("\n".join(lines))
    return cleaned


def load_guidebook_chunks() -> list[Chunk]:
    ensure_guidebook()
    chunks = []
    for page_number, text in enumerate(extract_pages(GUIDEBOOK_PDF), start=1):
        chunks += chunk_text(text, source=f"{GUIDEBOOK_TITLE}/p{page_number}")
    return chunks


def load_or_build_app_index():
    """The chatbot's index: the guidebook plus the JSQuAD Wikipedia articles.
    (Evaluation keeps its own JSQuAD-only index, so earlier results stay comparable.)"""
    from app.index import VectorIndex

    if (APP_INDEX_DIR / "meta.json").exists():
        return VectorIndex.load(APP_INDEX_DIR)
    chunks = load_guidebook_chunks() + load_jsquad()[0]
    print(f"Embedding {len(chunks)} chunks (one time only)...")
    index = VectorIndex.build(chunks)
    index.save(APP_INDEX_DIR)
    return index


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    chunks = load_guidebook_chunks()
    sizes = sorted(len(c.text) for c in chunks)
    print(f"{len(chunks)} guidebook chunks, median {sizes[len(sizes) // 2]} chars, longest {sizes[-1]}")
    for c in [c for c in chunks if c.source.endswith("/p21")][:2]:
        print(f"\n{c.id}\n{c.text}")
