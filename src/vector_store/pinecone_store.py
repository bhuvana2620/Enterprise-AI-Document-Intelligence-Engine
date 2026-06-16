# src/vector_store/pinecone_store.py

import os
import sys
import hashlib
import resource
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


# ---------------------------------------------------
# Environment Setup
# ---------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ai-document-intelligence")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "clean-v1")

EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
UPSERT_BATCH_SIZE = int(os.getenv("PINECONE_UPSERT_BATCH_SIZE", "50"))

ENABLE_HYBRID_SEARCH = (
    os.getenv("ENABLE_HYBRID_SEARCH", "false").strip().lower() == "true"
)


# ---------------------------------------------------
# Debug / Memory Helpers
# ---------------------------------------------------
def log_step(message: str) -> None:
    print(message, flush=True)


def log_memory(label: str) -> None:
    """
    Logs rough max RSS memory usage.

    Linux reports ru_maxrss in KB.
    macOS reports ru_maxrss in bytes.
    Render/Railway are Linux, so this is useful in deploy logs.
    """
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        if sys.platform == "darwin":
            mb = rss / 1024 / 1024
        else:
            mb = rss / 1024

        print(f"[MEMORY] {label}: {mb:.2f} MB", flush=True)

    except Exception as e:
        print(f"[MEMORY] Could not read memory usage: {e}", flush=True)


# ---------------------------------------------------
# Lazy Pinecone Setup
# ---------------------------------------------------
_pc: Optional[Pinecone] = None
_index = None
_sparse_encoder = None


def get_pinecone_client() -> Pinecone:
    global _pc

    if not PINECONE_API_KEY:
        raise ValueError("Missing PINECONE_API_KEY in environment variables.")

    if _pc is None:
        log_step("[PINECONE] Creating Pinecone client")
        _pc = Pinecone(api_key=PINECONE_API_KEY)

    return _pc


def get_existing_index_names() -> list[str]:
    pc = get_pinecone_client()
    indexes = pc.list_indexes()

    if hasattr(indexes, "names"):
        return indexes.names()

    names = []
    for idx in indexes:
        if isinstance(idx, dict):
            names.append(idx["name"])
        else:
            names.append(idx.name)

    return names


def ensure_index():
    pc = get_pinecone_client()

    existing_indexes = get_existing_index_names()

    if INDEX_NAME not in existing_indexes:
        log_step(f"[PINECONE] Creating Pinecone index: {INDEX_NAME}")

        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )

        log_step(f"[PINECONE] Created Pinecone index: {INDEX_NAME}")

    else:
        log_step(f"[PINECONE] Pinecone index already exists: {INDEX_NAME}")

    return pc.Index(INDEX_NAME)


def get_index():
    global _index

    if _index is None:
        log_step("[PINECONE] Initializing Pinecone index handle")
        _index = ensure_index()
        log_memory("after Pinecone index handle initialized")

    return _index


def get_sparse_encoder():
    """
    Load sparse encoder only when hybrid search is enabled.

    This avoids unnecessary sparse encoding and avoids sending sparse_values
    into a dense-only Pinecone index.
    """
    global _sparse_encoder

    if not ENABLE_HYBRID_SEARCH:
        return None

    if _sparse_encoder is None:
        log_step("[SPARSE] Loading NativeSparseEncoder")
        from src.embeddings.sparse_encoder import NativeSparseEncoder

        _sparse_encoder = NativeSparseEncoder()
        log_step("[SPARSE] NativeSparseEncoder loaded")
        log_memory("after sparse encoder loaded")

    return _sparse_encoder


# ---------------------------------------------------
# Clear Namespace
# ---------------------------------------------------
def clear_namespace(namespace: str) -> dict:
    """
    Delete all vectors from a Pinecone namespace.

    Used to clean up browser-session data so old uploaded documents
    do not remain searchable or consume storage.
    """

    if not namespace or not namespace.strip():
        raise ValueError("Namespace is required to clear vectors.")

    target_namespace = namespace.strip()

    log_step(f"[PINECONE] Clearing namespace: {target_namespace}")

    pinecone_index = get_index()

    pinecone_index.delete(
        delete_all=True,
        namespace=target_namespace
    )

    log_step(f"[PINECONE] Cleared Pinecone namespace: {target_namespace}")

    return {
        "status": "cleared",
        "index_name": INDEX_NAME,
        "namespace": target_namespace
    }


# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def embedding_to_list(embedding: Any) -> list[float]:
    if hasattr(embedding, "tolist"):
        return embedding.tolist()

    return list(embedding)


def stable_chunk_id(source: str, chunk_index: int, text: str) -> str:
    """
    Stable ID prevents duplicate records when re-indexing same document.
    """
    safe_source = (
        source.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )

    digest = hashlib.sha256(
        f"{source}:{chunk_index}:{text[:500]}".encode("utf-8")
    ).hexdigest()[:16]

    return f"{safe_source}-chunk-{chunk_index}-{digest}"


def batch_items(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


# ---------------------------------------------------
# Store Embeddings
# ---------------------------------------------------
def store_embeddings(
    chunks: list[dict],
    embeddings: list,
    namespace: Optional[str] = None
) -> dict:
    """
    Store document chunks in Pinecone with clean metadata.

    Each vector record contains:
    - id
    - dense embedding values
    - metadata: text, source, category, chunk_index

    If ENABLE_HYBRID_SEARCH=true:
    - sparse_values are also included.
    """

    target_namespace = namespace or PINECONE_NAMESPACE

    log_step("[STORE] Step 1: store_embeddings started")
    log_step(f"[STORE] chunks={len(chunks)}, embeddings={len(embeddings)}")
    log_step(f"[STORE] namespace={target_namespace}")
    log_step(f"[STORE] ENABLE_HYBRID_SEARCH={ENABLE_HYBRID_SEARCH}")
    log_memory("store_embeddings start")

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunks and embeddings count mismatch: "
            f"chunks={len(chunks)}, embeddings={len(embeddings)}"
        )

    sparse_encoder = get_sparse_encoder()
    vectors = []

    log_step("[STORE] Step 2: building Pinecone vectors")

    for chunk_dict, embedding in zip(chunks, embeddings):
        metadata = chunk_dict.get("metadata", {}).copy()

        text = chunk_dict.get("text") or metadata.get("text")
        if not text:
            raise ValueError("Chunk is missing text. Cannot index empty chunk.")

        source = metadata.get("source", "unknown_source")
        category = metadata.get("category", "uncategorized")
        chunk_index = int(metadata.get("chunk_index", len(vectors)))

        clean_metadata = {
            "text": text,
            "source": source,
            "category": category,
            "chunk_index": chunk_index
        }

        dense_values = embedding_to_list(embedding)

        if len(dense_values) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {EMBEDDING_DIMENSION}, "
                f"got {len(dense_values)}."
            )

        vector = {
            "id": stable_chunk_id(source, chunk_index, text),
            "values": dense_values,
            "metadata": clean_metadata
        }

        if ENABLE_HYBRID_SEARCH and sparse_encoder is not None:
            vector["sparse_values"] = sparse_encoder.encode_text(text)

        vectors.append(vector)

    log_step(f"[STORE] Step 3: built {len(vectors)} vectors")
    log_memory("after vector build")

    pinecone_index = get_index()

    log_step("[STORE] Step 4: starting Pinecone upsert")

    total_upserted = 0

    for batch_number, batch in enumerate(
        batch_items(vectors, UPSERT_BATCH_SIZE),
        start=1
    ):
        log_step(
            f"[STORE] Upserting batch {batch_number} "
            f"with {len(batch)} vectors"
        )

        pinecone_index.upsert(
            vectors=batch,
            namespace=target_namespace
        )

        total_upserted += len(batch)
        log_step(f"[STORE] Batch {batch_number} upsert complete")
        log_memory(f"after upsert batch {batch_number}")

    log_step(
        f"[STORE] Stored {total_upserted} vectors into "
        f"index='{INDEX_NAME}', namespace='{target_namespace}'."
    )
    try:
        stats = pinecone_index.describe_index_stats()
        stats_dict = stats.to_dict() if hasattr(stats, "to_dict") else stats
        namespaces = stats_dict.get("namespaces", {})

        log_step(f"[PINECONE] Stats after upsert: {namespaces}")

    except Exception as e:
        log_step(f"[WARN] Could not read Pinecone stats after upsert: {e}")

    return {
        "index_name": INDEX_NAME,
        "namespace": target_namespace,
        "upserted_count": total_upserted
    }


# ---------------------------------------------------
# Full Document Indexing Pipeline
# ---------------------------------------------------
def index_document(
    file_path: str,
    category: str = "university_records",
    namespace: Optional[str] = None,
    source_name: Optional[str] = None
) -> dict:
    """
    Full pipeline:
    file -> load -> clean -> chunk -> embeddings -> Pinecone upsert
    """

    log_step("[INDEX] Step 1: index_document started")
    log_memory("index_document start")

    resolved_path = Path(file_path).expanduser().resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"File not found: {resolved_path}")

    source_name = source_name or resolved_path.name
    target_namespace = namespace or PINECONE_NAMESPACE

    log_step(f"[INDEX] file_path={resolved_path}")
    log_step(f"[INDEX] source_name={source_name}")
    log_step(f"[INDEX] category={category}")
    log_step(f"[INDEX] namespace={target_namespace}")

    log_step("[INDEX] Step 2: before embedding_pipeline import")
    log_memory("before embedding_pipeline import")

    from src.embeddings.embedding_generator import embedding_pipeline

    log_step("[INDEX] Step 3: embedding_pipeline import OK")
    log_memory("after embedding_pipeline import")

    log_step("[INDEX] Step 4: before embedding_pipeline execution")

    chunks, embeddings = embedding_pipeline(
        file_path=str(resolved_path),
        source_name=source_name,
        category=category
    )

    log_step("[INDEX] Step 5: embedding_pipeline execution OK")
    log_step(f"[INDEX] chunks={len(chunks)}, embeddings={len(embeddings)}")
    log_memory("after embedding_pipeline execution")

    if chunks is None or len(chunks) == 0:
        raise ValueError("No chunks generated from document.")

    if embeddings is None or len(embeddings) == 0:
        raise ValueError("No embeddings generated from document.")

    log_step("[INDEX] Step 6: before store_embeddings")

    result = store_embeddings(
        chunks=chunks,
        embeddings=embeddings,
        namespace=target_namespace
    )

    log_step("[INDEX] Step 7: store_embeddings OK")
    log_memory("index_document complete")

    return result


# ---------------------------------------------------
# CLI Runner
# ---------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index document into Pinecone.")
    parser.add_argument("--file", required=True, help="Path to PDF, TXT, or DOCX file.")
    parser.add_argument("--category", default="university_records")
    parser.add_argument("--namespace", default=None)

    args = parser.parse_args()

    result = index_document(
        file_path=args.file,
        category=args.category,
        namespace=args.namespace
    )

    print("\nIndexing complete.")
    print(result)