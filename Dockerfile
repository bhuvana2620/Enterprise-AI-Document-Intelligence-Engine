FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Hugging Face Spaces exposes port 7860.
ENV PORT=7860

# Internal backend URL for Streamlit.
ENV API_BASE_URL=http://127.0.0.1:8000

# Product mode defaults.
ENV EMBEDDING_PROVIDER=sentence_transformer
ENV EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ENV EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
ENV EMBEDDING_DEVICE=cpu
ENV EMBEDDING_DIMENSION=384
ENV EMBEDDING_BATCH_SIZE=8
ENV EMBEDDING_MAX_SEQ_LENGTH=512

ENV ENABLE_RERANKER=true
ENV RERANKER_MODEL=BAAI/bge-reranker-base
ENV ENABLE_HYBRID_SEARCH=false

ENV ENABLE_OCR=true
ENV OCR_MIN_TEXT_LENGTH=200

ENV LLM_PROVIDER=auto
ENV MOCK_LLM_MODE=false
ENV GEMINI_MODEL=gemini-3.1-flash-lite
ENV GEMINI_FALLBACK_MODELS=gemini-2.5-flash-lite,gemini-2.0-flash-lite
ENV XAI_BASE_URL=https://api.x.ai/v1
ENV XAI_MODEL=grok-4.3
ENV LLM_MAX_RETRIES=3
ENV SHOW_LLM_FALLBACK_REASON=true

# Reduce CPU thread explosion.
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# Hugging Face cache.
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    supervisor \
    tesseract-ocr \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install -r requirements.txt

ENV PATH="/opt/venv/bin:$PATH"

COPY . .

# Preload models during image build.
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder

print("Downloading semantic embedding model...")
SentenceTransformer("BAAI/bge-small-en-v1.5")

print("Downloading reranker model...")
CrossEncoder("BAAI/bge-reranker-base")

print("Model preload complete.")
PY

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 7860

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]