# =====================================================================
# Enterprise AI Document Intelligence Engine
# Dockerfile for Hugging Face Spaces (Docker SDK)
# =====================================================================
#
# Hugging Face Spaces requirements this Dockerfile satisfies:
#   - Exposes exactly one port: 7860 (HF's default expected port)
#   - Runs as a non-root user (HF Spaces enforces this)
#   - Single CMD that brings up both backend (internal) and frontend
#     (public) via start.sh
#
# =====================================================================
# Stage 1: Build dependencies
# =====================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv ${VIRTUAL_ENV}

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
      --timeout 180 \
      --retries 8 \
      --prefer-binary \
      -r requirements.txt

# =====================================================================
# Stage 2: Runtime image
# =====================================================================
FROM python:3.11-slim AS runner

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# tesseract-ocr -> OCR fallback for scanned/image PDFs
# libglib2.0-0 / libgl1 -> required by PyMuPDF / Pillow image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tesseract-ocr \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as a non-root user.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser app.py /app/app.py
COPY --chown=appuser:appuser start.sh /app/start.sh
COPY --chown=appuser:appuser requirements.txt /app/requirements.txt

RUN mkdir -p /app/data/uploads /app/.cache/huggingface && \
    chown -R appuser:appuser /app && \
    chmod +x /app/start.sh

USER appuser

# Hugging Face Spaces expects the app to listen on 7860.
EXPOSE 7860

ENV BACKEND_PORT=8000
ENV FRONTEND_PORT=7860

CMD ["./start.sh"]
