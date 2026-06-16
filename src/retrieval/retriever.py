# src/retrieval/retriever.py

import os
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------
# Dynamic Workspace Path Injection
# ---------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ---------------------------------------------------
# Optional .env Loading
# ---------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except Exception as e:
    print(f"[WARN] dotenv load skipped: {e}", flush=True)


# ---------------------------------------------------
# Config
# ---------------------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ai-document-intelligence")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "clean-v1")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "false").lower() == "true"
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "false").lower() == "true"

EMBEDDING_MAX_SEQ_LENGTH = int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH", "256"))


# ---------------------------------------------------
# Lazy Globals
# ---------------------------------------------------
_pc = None
_index = None
_sparse_encoder = None
_reranker_model = None


# ---------------------------------------------------
# Lazy Pinecone Client
# ---------------------------------------------------
def get_pinecone_index():
    """
    Lazy Pinecone initialization.

    This prevents Pinecone client setup during module import.
    """
    global _pc, _index

    if _index is not None:
        return _index

    if not PINECONE_API_KEY:
        raise ValueError("Missing PINECONE_API_KEY environment variable.")

    print("[RETRIEVER] Initializing Pinecone client...", flush=True)

    from pinecone import Pinecone

    _pc = Pinecone(api_key=PINECONE_API_KEY)
    _index = _pc.Index(PINECONE_INDEX_NAME)

    print("[RETRIEVER] Pinecone index initialized.", flush=True)

    return _index


# ---------------------------------------------------
# Lazy Embedding Model
# ---------------------------------------------------
def get_embedding_model():
    """
    Reuse the shared embedding model from embedding_generator.py.

    This avoids loading a second SentenceTransformer instance in retriever.py,
    which can cause Render memory restarts.
    """
    print("[RETRIEVER] Reusing shared embedding model...", flush=True)

    from src.embeddings.embedding_generator import get_embedding_model as shared_get_embedding_model

    return shared_get_embedding_model()


# ---------------------------------------------------
# Lazy Sparse Encoder
# ---------------------------------------------------
def get_sparse_encoder():
    """
    Lazy-load sparse encoder only if hybrid search is enabled.
    """
    global _sparse_encoder

    if _sparse_encoder is not None:
        return _sparse_encoder

    print("[RETRIEVER] Loading sparse encoder...", flush=True)

    from src.embeddings.sparse_encoder import NativeSparseEncoder

    _sparse_encoder = NativeSparseEncoder()

    print("[RETRIEVER] Sparse encoder loaded.", flush=True)

    return _sparse_encoder


# ---------------------------------------------------
# Lazy Reranker
# ---------------------------------------------------
def get_reranker_model():
    """
    Lazy-load CrossEncoder only when ENABLE_RERANKER=true.

    On Render free tier, keep ENABLE_RERANKER=false.
    """
    global _reranker_model

    if _reranker_model is not None:
        return _reranker_model

    if not ENABLE_RERANKER:
        return None

    print(f"[RETRIEVER] Loading reranker model: {RERANKER_MODEL_NAME}", flush=True)

    try:
        import torch
        torch.set_num_threads(1)
    except Exception as e:
        print(f"[WARN] Could not configure torch threads for reranker: {e}", flush=True)

    from sentence_transformers import CrossEncoder

    _reranker_model = CrossEncoder(
        RERANKER_MODEL_NAME,
        device="cpu"
    )

    print("[RETRIEVER] Reranker model loaded.", flush=True)

    return _reranker_model


# ---------------------------------------------------
# Generate query embedding
# ---------------------------------------------------
def generate_query_embedding(query: str) -> list[float]:
    """
    Convert user query into a dense embedding vector using the shared model.
    """
    model = get_embedding_model()

    embedding = model.encode(
        query,
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embedding.tolist()


# ---------------------------------------------------
# Normalize Pinecone result object
# ---------------------------------------------------
def extract_matches(results) -> list:
    """
    Pinecone SDK responses can behave like dicts or objects depending on version.
    This helper makes retrieval safer.
    """
    if isinstance(results, dict):
        return results.get("matches", [])

    return getattr(results, "matches", [])


# ---------------------------------------------------
# Retrieval
# ---------------------------------------------------
def retrieve_hybrid_chunks(
    query: str,
    category_filter: Optional[str] = None,
    top_k: int = 3,
    alpha: float = 0.7,
    namespace: Optional[str] = None
) -> list[dict]:
    """
    Retrieval pipeline.

    Stage 1:
    Dense Pinecone retrieval by default.

    Optional:
    Hybrid sparse search only when ENABLE_HYBRID_SEARCH=true.

    Optional:
    CrossEncoder reranking only when ENABLE_RERANKER=true.
    """

    target_namespace = namespace or PINECONE_NAMESPACE
    candidate_pool_size = max(15, top_k * 3)

    index = get_pinecone_index()
    query_embedding = generate_query_embedding(query)

    filter_dict = {"category": {"$eq": category_filter}} if category_filter else None

    # ---------------------------------------------------
    # Stage 1: Pinecone retrieval
    # ---------------------------------------------------
    if ENABLE_HYBRID_SEARCH:
        try:
            sparse_encoder = get_sparse_encoder()
            sparse_vector = sparse_encoder.encode_text(query)

            print("[RETRIEVER] Running hybrid query...", flush=True)

            results = index.query(
                vector=query_embedding,
                sparse_vector=sparse_vector,
                top_k=candidate_pool_size,
                include_metadata=True,
                filter=filter_dict,
                namespace=target_namespace
            )

        except Exception as e:
            print(f"[WARN] Hybrid query failed. Falling back to dense-only search: {e}", flush=True)

            results = index.query(
                vector=query_embedding,
                top_k=candidate_pool_size,
                include_metadata=True,
                filter=filter_dict,
                namespace=target_namespace
            )
    else:
        print("[RETRIEVER] Running dense-only query...", flush=True)

        results = index.query(
            vector=query_embedding,
            top_k=candidate_pool_size,
            include_metadata=True,
            filter=filter_dict,
            namespace=target_namespace
        )

    candidates = []

    for match in extract_matches(results):
        if isinstance(match, dict):
            metadata = match.get("metadata", {}) or {}
            raw_score = match.get("score", 0.0)
        else:
            metadata = getattr(match, "metadata", {}) or {}
            raw_score = getattr(match, "score", 0.0)

        text = metadata.get("text", "")

        if not text or not text.strip():
            continue

        candidates.append({
            "text": text,
            "source": metadata.get("source", "Unknown Document"),
            "category": metadata.get("category", "Uncategorized"),
            "chunk_index": metadata.get("chunk_index"),
            "raw_score": float(raw_score),
            "score": float(raw_score)
        })

    if not candidates:
        return []

    # ---------------------------------------------------
    # Stage 2: Optional CrossEncoder reranking
    # ---------------------------------------------------
    reranker_model = get_reranker_model()

    if reranker_model is not None:
        print("[RETRIEVER] Running CrossEncoder reranking...", flush=True)

        pairs = [[query, candidate["text"]] for candidate in candidates]
        rerank_scores = reranker_model.predict(pairs)

        for idx, score in enumerate(rerank_scores):
            candidates[idx]["score"] = float(score)

        candidates = sorted(
            candidates,
            key=lambda item: item["score"],
            reverse=True
        )
    else:
        candidates = sorted(
            candidates,
            key=lambda item: item["raw_score"],
            reverse=True
        )

    return candidates[:top_k]


# ---------------------------------------------------
# Downstream Interface Wrapper
# ---------------------------------------------------
def retrieve_chunks(
    query: str,
    category_filter: Optional[str] = None,
    top_k: int = 3,
    namespace: Optional[str] = None
) -> list[dict]:
    """
    Stable interface used by FastAPI, tests, and generation modules.
    """
    return retrieve_hybrid_chunks(
        query=query,
        category_filter=category_filter,
        top_k=top_k,
        alpha=0.7,
        namespace=namespace
    )


# ---------------------------------------------------
# Main Test Runner
# ---------------------------------------------------
if __name__ == "__main__":
    test_query = "What happened in the university library?"

    print(
        f"\nRunning retrieval against "
        f"index='{PINECONE_INDEX_NAME}', namespace='{PINECONE_NAMESPACE}'...\n"
    )

    chunks = retrieve_chunks(
        query=test_query,
        category_filter=None,
        top_k=3
    )

    print("===== RETRIEVED CHUNKS =====\n")

    if not chunks:
        print("No chunks retrieved. Check namespace, index, and whether documents were indexed.")
    else:
        for i, chunk in enumerate(chunks, start=1):
            print(f"MATCH CHUNK {i}")
            print(f"Score: {chunk['score']:.4f}")
            print(f"Raw Pinecone Score: {chunk['raw_score']:.4f}")
            print(f"Source: {chunk['source']}")
            print(f"Category: {chunk['category']}")
            print(f"Chunk Index: {chunk.get('chunk_index')}")
            print(f"Text Excerpt:\n{chunk['text'][:400]}...")
            print("\n" + "=" * 70 + "\n")
