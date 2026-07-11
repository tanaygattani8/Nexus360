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

# RAG_LITE=true builds a small image with BM25-only retrieval (no torch, no
# ML models) — fits free hosts with 512MB RAM like Render's free tier.
# Render exposes service env vars as Docker build args automatically.
ARG RAG_LITE=false
ENV RAG_LITE=${RAG_LITE}
ENV HF_HOME=/app/.cache

COPY requirements.txt .
RUN if [ "$RAG_LITE" = "true" ]; then \
      grep -v "sentence-transformers" requirements.txt > /tmp/req.txt && \
      pip install --no-cache-dir -r /tmp/req.txt; \
    else \
      # CPU-only torch first (much smaller than the default CUDA build), \
      # then bake the models in so cold starts don't re-download ~200MB \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir -r requirements.txt && \
      python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"; \
    fi

COPY backend/ ./backend/
COPY --from=ui /ui/dist ./frontend/dist

# Non-root user (required by some hosts); the app writes backend/memory.db
# and backend/qdrant_local/ at runtime.
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

WORKDIR /app/backend
EXPOSE 7860
# $PORT is set by Render; defaults to 7860 elsewhere
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
