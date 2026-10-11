---
title: Japanese RAG Chatbot
emoji: 🗾
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
---

# Japanese RAG Chatbot 日本語RAGチャットボット

A chatbot that answers questions from Japanese documents, citing the passage each answer came from. Ask in Japanese **or English**: retrieval works across languages, and the answer comes back in the language you asked in.

**[English](#english) | [日本語](#日本語)**

<!-- Add a demo GIF here: ![demo](docs/demo.gif) -->

---

## English

### Why this project
Japanese is harder than English for search: there are no spaces between words, the same character can be written in full-width or half-width form, and official documents use formal vocabulary (紛失, "loss") that users don't type (なくした, "lost"). This project handles each of those, and it's evaluated with metrics instead of eyeballing, because I don't read Japanese myself.

### How it works
```
                         ┌─ semantic search: multilingual-e5 embeddings + FAISS  (meaning)
question ── top 20 each ─┤                                                        ├─ RRF fusion ─ top 5 ─ Gemini ─ answer + [n] citations
                         └─ keyword search: BM25 on MeCab tokens                  (exact words)
```

| Stage | What it does | Japanese-specific handling |
|---|---|---|
| Chunking (`app/chunking.py`) | ~400-character chunks of whole sentences, one-sentence overlap | NFKC normalization (１４→14), splits on 。!?, rejoins PDF line wraps |
| Embeddings (`app/embeddings.py`) | `intfloat/multilingual-e5-small`, runs on CPU | Multilingual: English questions find Japanese passages |
| Vector index (`app/index.py`) | FAISS `IndexFlatIP` (exact cosine search), saved with the model name | — |
| Keyword search (`app/retriever.py`) | BM25 | MeCab (fugashi + unidic-lite) tokenization, grammar words dropped by part of speech, verbs reduced to dictionary form |
| Fusion (`app/retriever.py`) | Reciprocal Rank Fusion of both top-20 lists | — |
| Generation (`app/generator.py`) | Gemini, told to answer only from the passages, cite them, and say "I don't know" otherwise | Answers in the question's language |
| API and UI (`app/main.py`, `static/`) | FastAPI + a single-page chat UI with expandable sources | — |

### Results
Evaluated on [JSQuAD](https://github.com/yahoojapan/JGLUE) (Japanese Wikipedia QA): 59 articles indexed as 1,164 chunks. Questions are a fixed-seed random sample.

**Retrieval (500 questions):**

| Method | Recall@1 | Recall@5 |
|---|---|---|
| Semantic only | 87.4% | 96.6% |
| BM25 only | 87.0% | 97.0% |
| **Hybrid (RRF)** | **88.4%** | **98.2%** |

Hybrid search missed 9 questions against 17 for semantic search alone.

**End to end (200 questions, `gemini-3.8-flash`):** Recall@5 99.0%, character F1 96.7%, exact match 92.5%.

**Error analysis:** of the 15 exact-match misses, about 8 were correct answers with a few words more or fewer (約31万8千人 vs 31万8千人), 2 were the same answer written differently (二八条 vs 28条), about 3 were genuinely wrong, and 2 were "I don't know", one of them correctly so because retrieval had missed the paragraph.

**Caveats:** JSQuAD questions were written while looking at the paragraph and reuse its wording, which makes retrieval easier than with real users' questions. The numbers show the pipeline works; they aren't a claim about arbitrary real-world documents.

### Run it locally
You need Python 3.12 and a free Gemini API key from [Google AI Studio](https://aistudio.google.com) ("Get API key").

```bash
git clone https://github.com/goswamivaishnavi45/nihongo-rag-chatbot.git
cd nihongo-rag-chatbot
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
echo GEMINI_API_KEY=your-key > .env      # free key from https://aistudio.google.com
uvicorn app.main:app --port 8000         # first start builds the index (~1 minute)
```
Then open http://127.0.0.1:8000. Interactive API docs are at `/docs`.

**Streamlit version** (streaming answers, conversation memory):
```bash
streamlit run streamlit_app.py
```
On Streamlit Community Cloud, add `GEMINI_API_KEY` (and optionally `GEMINI_MODEL`) under the app's **Secrets**.

**With Docker:**
```bash
docker build -t japanese-rag .
docker run -p 7860:7860 --env-file .env japanese-rag
```

**Reproduce the evaluation:**
```bash
python -m app.retriever          # retrieval: semantic vs BM25 vs hybrid
python -m app.evaluate 200       # end to end; resumes if interrupted
```

### Project structure
```
app/
  chunking.py     Japanese normalization and sentence-aware chunking
  embeddings.py   multilingual-e5 embeddings ("query: " / "passage: " prefixes)
  index.py        FAISS vector index: build, save, load (checks the model name), search
  retriever.py    MeCab + BM25 keyword search, and hybrid search with RRF
  generator.py    Gemini prompt with grounding rules and citations
  jsquad.py       loads the JSQuAD documents and labelled questions
  evaluate.py     end-to-end evaluation: Recall@5, exact match, character F1
  main.py         FastAPI app: /api/ask, /api/topics, /api/health, rate limiting
static/index.html chat page
data/eval/        JSQuAD data and saved evaluation results
Dockerfile        container image (CPU PyTorch; model and index built in)
```
Each module in `app/` can be run directly to see a demo, e.g. `python -m app.chunking`.

### Design decisions
- **Hybrid retrieval:** semantic search handles synonyms and English questions; BM25 handles exact names and numbers. RRF combines them using ranks only, because their scores are on different scales.
- **Grounded answers:** the model may only use the retrieved passages and must say "I don't know" otherwise. Asked about visa fees, it found Japan's immigration law article, saw that the fee wasn't in it, and declined instead of guessing.
- **Public demo protection:** the API limits each visitor to 5 questions a minute and the whole demo to 300 a day, so the hosted key can't be abused. Both are configurable with `MAX_QUESTIONS_PER_MINUTE` and `MAX_QUESTIONS_PER_DAY`.
- **Safe display:** documents and model output are shown with `textContent`, never `innerHTML`, so HTML inside them can't run (XSS).

### Limitations and next steps
- The knowledge base is 59 Wikipedia articles. Next: practical documents such as city guides for foreign residents.
- Answers arrive all at once. Next: stream them token by token.
- Exact match undercounts correct answers. Next: normalize kanji numerals before scoring, and add an LLM-as-judge for full-sentence answers.

### Credits
- Data: JSQuAD from [JGLUE](https://github.com/yahoojapan/JGLUE) (Yahoo Japan), CC BY-SA 4.0. Text from Japanese Wikipedia.
- Embedding model: [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) (MIT).
- Tokenizer: MeCab via [fugashi](https://github.com/polm/fugashi) with unidic-lite.

---

## 日本語

日本語の文書をもとに質問に答え、回答の根拠となった文書を引用して示すチャットボットです。質問は日本語でも英語でもできます。

### 仕組み
- **チャンク分割:** NFKC正規化を行い、「。」「!」「?」で文に分けたうえで、約400文字ずつのチャンクにまとめます。PDFの改行で途切れた文もつなぎ直します。
- **ハイブリッド検索:** 多言語埋め込みモデル(multilingual-e5)とFAISSによる意味検索と、MeCabで形態素解析したBM25によるキーワード検索を、RRF(Reciprocal Rank Fusion)で統合します。
- **回答生成:** Geminiが検索された文書だけを使って回答し、根拠を [1] のように引用します。文書に答えがない場合は「わかりません」と答えます。

### 評価結果(JSQuAD)
- 検索(500問): ハイブリッド検索の Recall@5 は 98.2%(意味検索のみ 96.6%、BM25のみ 97.0%)
- 回答(200問): 文字単位F1 96.7%、完全一致 92.5%
- JSQuADの質問は本文の表現をそのまま使っていることが多く、実際のユーザーの質問より検索しやすい点に注意が必要です。

### 実行方法
上の「Run it locally」を参照してください。`.env` に `GEMINI_API_KEY` を設定して `uvicorn app.main:app` で起動します。
