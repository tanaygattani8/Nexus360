# ── Stage 1: build the React UI ──────────────────────────────────────────────
FROM node:20-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-fund --no-audit
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python backend, serving the built UI ────────────────────────────
FROM python:3.12-slim
WORKDIR /app

# CPU-only torch first (much smaller than the default CUDA build),
# then the rest of the requirements see torch already satisfied.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Bake the embedding + reranker models into the image so cold starts
# don't re-download ~200MB.
ENV HF_HOME=/app/.cache
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY backend/ ./backend/
COPY --from=ui /ui/dist ./frontend/dist

# Hugging Face Spaces runs containers as UID 1000; the app also needs to
# write backend/memory.db and backend/qdrant_local/ at runtime.
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

WORKDIR /app/backend
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
