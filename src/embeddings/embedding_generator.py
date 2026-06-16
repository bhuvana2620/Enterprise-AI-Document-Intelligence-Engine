# src/embeddings/embedding_generator.py

import os
import sys
import gc
import json
import resource
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ---------------------------------------------------
# Embedding Configuration
# ---------------------------------------------------
# Product default: real semantic embeddings
# Lightweight fallback: set EMBEDDING_PROVIDER=hash
EMBEDDING_PROVIDER = os.getenv(
    "EMBEDDING_PROVIDER",
    "sentence_transformer"
).lower().strip()

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
)

EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu").lower().strip()

EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
EMBEDDING_MAX_SEQ_LENGTH = int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH", "512"))

_model = None


# ---------------------------------------------------
# Logging Helpers
# ---------------------------------------------------
def log_step(message: str) -> None:
    print(message, flush=True)


def log_memory(label: str) -> None:
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mb = rss / 1024 / 1024 if sys.platform == "darwin" else rss / 1024
        print(f"[MEMORY] {label}: {mb:.2f} MB", flush=True)
    except Exception as e:
        print(f"[MEMORY] Could not read memory usage: {e}", flush=True)


# ---------------------------------------------------
# Model Loading
# ---------------------------------------------------
def preload_embedding_model() -> None:
    """
    Called during FastAPI startup.

    This makes model-loading failures visible early instead of during the first upload.
    """
    log_step("[EMBED] Preloading embedding model at startup...")
    get_embedding_model()
    log_step("[EMBED] Embedding model preload complete.")


def get_embedding_model():
    """
    Lazily load and cache the embedding model.

    Supported providers:
    - sentence_transformer
    - sentence-transformer
    - semantic
    - bge
    - hash
    - free
    - lightweight
    """
    global _model

    if _model is not None:
        return _model

    log_step("[EMBED] Initializing embedding provider")
    log_step(f"[EMBED] provider={EMBEDDING_PROVIDER}")
    log_step(f"[EMBED] model={EMBEDDING_MODEL_NAME}")
    log_step(f"[EMBED] device={EMBEDDING_DEVICE}")
    log_step(f"[EMBED] embedding_dimension={EMBEDDING_DIMENSION}")
    log_step(f"[EMBED] batch_size={EMBEDDING_BATCH_SIZE}")
    log_step(f"[EMBED] max_seq_length={EMBEDDING_MAX_SEQ_LENGTH}")
    log_memory("before embedding provider init")

    # ---------------------------------------------------
    # Lightweight Hash Provider
    # ---------------------------------------------------
    # Keep this only as an emergency fallback.
    if EMBEDDING_PROVIDER in {"hash", "free", "lightweight"}:
        log_step("[EMBED] Using lightweight hash embedding provider")

        from src.embeddings.hash_embedding import HashEmbeddingModel

        _model = HashEmbeddingModel(dimension=EMBEDDING_DIMENSION)

        log_step("[EMBED] Hash embedding provider ready")
        log_memory("after hash embedding provider init")

        return _model

    # ---------------------------------------------------
    # Production Semantic Provider
    # ---------------------------------------------------
    if EMBEDDING_PROVIDER not in {
        "sentence_transformer",
        "sentence-transformer",
        "semantic",
        "bge",
    }:
        raise ValueError(
            f"Unsupported EMBEDDING_PROVIDER='{EMBEDDING_PROVIDER}'. "
            "Use 'sentence_transformer' for production or 'hash' for fallback."
        )

    log_step("[EMBED] Step 1: before torch/sentence-transformers import")
    log_memory("before embedding model imports")

    try:
        import torch

        torch.set_num_threads(1)

        try:
            torch.set_num_interop_threads(1)
        except Exception as e:
            print(f"[WARN] Could not set torch interop threads: {e}", flush=True)

    except Exception as e:
        print(f"[WARN] Could not configure torch threads: {e}", flush=True)

    from sentence_transformers import SentenceTransformer

    log_step("[EMBED] Step 2: SentenceTransformer import OK")
    log_memory("after SentenceTransformer import")

    log_step(f"[EMBED] Step 3: loading semantic embedding model: {EMBEDDING_MODEL_NAME}")

    _model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device=EMBEDDING_DEVICE
    )

    _model.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH

    log_step("[EMBED] Step 4: semantic embedding model loaded")
    log_step(f"[EMBED] model_name={EMBEDDING_MODEL_NAME}")
    log_step(f"[EMBED] max_seq_length={_model.max_seq_length}")
    log_memory("after semantic embedding model load")

    return _model


# ---------------------------------------------------
# Encoding Helpers
# ---------------------------------------------------
def embedding_to_list(value: Any) -> list[float]:
    """
    Convert numpy arrays, tensors, or plain iterables into Python float lists.
    """
    if hasattr(value, "tolist"):
        value = value.tolist()

    return [float(x) for x in value]


def validate_embedding_dimensions(embedding_lists: list[list[float]]) -> None:
    """
    Ensure embeddings match the Pinecone index dimension.
    """
    if not embedding_lists:
        raise ValueError("No embeddings were generated.")

    actual_dimension = len(embedding_lists[0])

    if actual_dimension != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Embedding dimension mismatch. "
            f"Expected EMBEDDING_DIMENSION={EMBEDDING_DIMENSION}, "
            f"but model produced dimension={actual_dimension}. "
            f"Check Pinecone index dimension and embedding model."
        )


def encode_texts(model, raw_texts: list[str]):
    """
    Encode text chunks using the active provider.
    """
    if EMBEDDING_PROVIDER in {"hash", "free", "lightweight"}:
        try:
            return model.encode(
                raw_texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        except TypeError:
            return model.encode(raw_texts)

    return model.encode(
        raw_texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )


# ---------------------------------------------------
# Public Embedding API
# ---------------------------------------------------
def generate_embeddings(chunks: list[dict]) -> list[list[float]]:
    """
    Generate embeddings for cleaned document chunks.
    """
    if not chunks:
        raise ValueError("No chunks provided for embedding.")

    log_step("[EMBED] Step 5: generate_embeddings started")
    log_step(f"[EMBED] provider={EMBEDDING_PROVIDER}")
    log_step(f"[EMBED] model={EMBEDDING_MODEL_NAME}")
    log_step(f"[EMBED] chunk_count={len(chunks)}, batch_size={EMBEDDING_BATCH_SIZE}")
    log_memory("generate_embeddings start")

    raw_texts = [
        c.get("text", "")
        for c in chunks
        if c.get("text", "").strip()
    ]

    if not raw_texts:
        raise ValueError("No non-empty chunk texts available for embedding.")

    log_step(f"[EMBED] non_empty_text_count={len(raw_texts)}")

    model = get_embedding_model()

    log_step("[EMBED] Step 6: before model.encode")
    log_memory("before model.encode")

    embeddings = encode_texts(model, raw_texts)

    log_step("[EMBED] Step 7: model.encode OK")
    log_memory("after model.encode")

    embedding_lists = [
        embedding_to_list(e)
        for e in embeddings
    ]

    validate_embedding_dimensions(embedding_lists)

    del embeddings
    gc.collect()

    log_step("[EMBED] Step 8: embeddings converted to lists")
    log_step(f"[EMBED] validated_dimension={len(embedding_lists[0])}")
    log_memory("after embeddings converted")

    return embedding_lists


def embedding_pipeline(file_path: str, source_name: str, category: str):
    """
    Full document embedding pipeline:

    file path
    → load document text
    → chunk text
    → generate embeddings
    → return chunks + vectors
    """
    log_step("[PIPELINE] Step 1: embedding_pipeline started")
    log_step(
        f"[PIPELINE] file_path={file_path}, "
        f"source_name={source_name}, category={category}"
    )
    log_memory("embedding_pipeline start")

    log_step("[PIPELINE] Step 2: before load_document import")
    from src.ingestion.load_documents import load_document
    log_step("[PIPELINE] Step 3: load_document import OK")
    log_memory("after load_document import")

    log_step("[PIPELINE] Step 4: before chunk_text import")
    from src.ingestion.chunker import chunk_text
    log_step("[PIPELINE] Step 5: chunk_text import OK")
    log_memory("after chunk_text import")

    log_step("[PIPELINE] Step 6: before load_document execution")
    text = load_document(file_path)
    log_step(f"[PIPELINE] Step 7: load_document OK, text_length={len(text)}")
    log_memory("after load_document")

    if not text or not text.strip():
        raise ValueError("Loaded document text is empty.")

    log_step("[PIPELINE] Step 8: before chunk_text execution")
    chunks = chunk_text(
        text,
        source_name=source_name,
        category=category
    )
    log_step(f"[PIPELINE] Step 9: chunk_text OK, chunks={len(chunks)}")
    log_memory("after chunk_text")

    if not chunks:
        raise ValueError("No chunks generated from document.")

    log_step("[PIPELINE] Step 10: before generate_embeddings")
    embeddings = generate_embeddings(chunks)
    log_step(f"[PIPELINE] Step 11: generate_embeddings OK, embeddings={len(embeddings)}")
    log_memory("embedding_pipeline complete")

    return chunks, embeddings


# ---------------------------------------------------
# Local Test Runner
# ---------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test embedding pipeline.")
    parser.add_argument("--file", required=True)
    parser.add_argument("--category", default="test")
    args = parser.parse_args()

    chunks, embeddings = embedding_pipeline(
        file_path=args.file,
        source_name=os.path.basename(args.file),
        category=args.category
    )

    print(
        f"Chunks: {len(chunks)}, "
        f"Embeddings: {len(embeddings)}, "
        f"Dims: {len(embeddings[0]) if embeddings else 0}"
    )

    if chunks:
        print(json.dumps(chunks[0], indent=4))