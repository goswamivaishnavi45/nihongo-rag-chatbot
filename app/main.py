"""Step 7: the web API (FastAPI) and chat page.

Run:  uvicorn app.main:app --port 8000      then open http://127.0.0.1:8000

The embedding model, FAISS index and BM25 take ~30s to load, so they are
loaded ONCE at startup and kept in memory, not reloaded for every question.
"""
import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.generator import MODEL, generate
from app.documents import load_or_build_app_index
from app.retriever import Retriever

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
state = {}  # holds the retriever once it's loaded
log = logging.getLogger("uvicorn.error")

# A public demo spends MY Gemini credit, so cap usage: per visitor per minute, and in total per day.
MAX_PER_MINUTE = int(os.getenv("MAX_QUESTIONS_PER_MINUTE", "5"))
MAX_PER_DAY = int(os.getenv("MAX_QUESTIONS_PER_DAY", "300"))
_recent = defaultdict(deque)  # visitor IP -> times of their recent questions
_daily = {"day": "", "count": 0}
_lock = threading.Lock()  # requests run in parallel threads, so update the counters one at a time


def check_rate_limit(http_request: Request) -> None:
    # Behind a hosting proxy the real visitor IP is the first entry of X-Forwarded-For.
    ip = http_request.headers.get("x-forwarded-for", http_request.client.host).split(",")[0].strip()
    now, today = time.time(), time.strftime("%Y-%m-%d")
    with _lock:
        recent = _recent[ip]
        while recent and now - recent[0] > 60:
            recent.popleft()
        if len(recent) >= MAX_PER_MINUTE:
            raise HTTPException(429, "Too many questions. Please wait a minute.")
        if _daily["day"] != today:
            _daily.update(day=today, count=0)
        if _daily["count"] >= MAX_PER_DAY:
            raise HTTPException(429, "The demo's daily question limit is reached. Please try again tomorrow.")
        recent.append(now)
        _daily["count"] += 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The chatbot searches the immigration guidebook plus the JSQuAD Wikipedia articles.
    index = load_or_build_app_index()
    state["retriever"] = Retriever(index)
    # Document titles, so users (and the bot) can see what it knows. Sources look like "梅雨/p0".
    state["topics"] = sorted({c.source.split("/")[0].replace("_", " ") for c in index.chunks})
    yield  # the server runs while we're paused here


app = FastAPI(title="Japanese RAG Chatbot", lifespan=lifespan)


# Pydantic models: FastAPI checks every request against these and rejects bad input automatically.
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class Source(BaseModel):
    number: int  # matches the [n] citations in the answer
    source: str
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/")
def chat_page():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL, "chunks": len(state["retriever"].chunks)}


@app.get("/api/topics")
def topics() -> list[str]:
    return state["topics"]


# Plain "def", not "async def": search and the Gemini call block while they wait, so
# FastAPI runs this in a worker thread and other users' requests aren't frozen meanwhile.
@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest, http_request: Request):
    check_rate_limit(http_request)
    top = state["retriever"].search(request.question, k=5)
    try:
        answer = generate(request.question, top, topics=state["topics"])
    except Exception:
        # Log the details for me; don't show internal error messages to visitors.
        log.exception("Gemini call failed")
        raise HTTPException(status_code=502, detail="The language model is unavailable. Please try again.")
    return AskResponse(
        answer=answer,
        sources=[Source(number=i, source=c.source, text=c.text) for i, c in enumerate(top, start=1)],
    )
