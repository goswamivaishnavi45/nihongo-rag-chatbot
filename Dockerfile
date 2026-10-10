# Japanese RAG chatbot: FastAPI + multilingual-e5 + FAISS + BM25 (MeCab) + Gemini.
# Build:  docker build -t japanese-rag .
# Run:    docker run -p 7860:7860 --env-file .env japanese-rag      (then open http://localhost:7860)
FROM python:3.12-slim

# Hugging Face Spaces runs containers as user 1000, so build everything as that user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PORT=7860
WORKDIR /home/user/app

# CPU-only PyTorch: the default download includes GPU libraries we'd never use (gigabytes).
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Download the embedding model and build the index while BUILDING the image,
# so the container starts in seconds instead of downloading and embedding on every start.
RUN python -c "from app.jsquad import load_jsquad, load_or_build_index; load_or_build_index(load_jsquad()[0])"

EXPOSE 7860
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/api/health')"
# GEMINI_API_KEY is NOT in the image. It's passed in at run time (--env-file, or a Space secret).
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
