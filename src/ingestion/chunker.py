# src/ingestion/chunker.py

import os
from typing import List, Dict


DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))


def normalize_text(text: str) -> str:
    """
    Normalize whitespace while preserving document meaning.
    """
    if not text:
        return ""

    return " ".join(text.split())


def split_with_overlap(
    text: str,
    chunk_size: int,
    chunk_overlap: int
) -> list[str]:
    """
    Lightweight pure-Python splitter.

    This avoids heavy splitter dependency during Render startup/upload.
    """
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Try to avoid cutting in the middle of a sentence/phrase.
        if end < text_length:
            sentence_break = max(
                text.rfind(". ", start, end),
                text.rfind("; ", start, end),
                text.rfind(", ", start, end),
                text.rfind(" ", start, end),
            )

            # Avoid creating very tiny chunks.
            if sentence_break > start + int(chunk_size * 0.5):
                end = sentence_break + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def chunk_text(
    text: str,
    source_name: str,
    category: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Dict]:
    """
    Splits document text into retrieval-friendly chunks.

    Keeps the same output structure your embedding and Pinecone code expects.
    """

    cleaned_text = normalize_text(text)

    if not cleaned_text:
        return []

    raw_chunks = split_with_overlap(
        cleaned_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    structured_chunks = []

    for index, chunk_string in enumerate(raw_chunks):
        cleaned_chunk = chunk_string.strip()

        if not cleaned_chunk:
            continue

        structured_chunks.append({
            "text": cleaned_chunk,
            "metadata": {
                "text": cleaned_chunk,
                "source": source_name,
                "category": category,
                "chunk_index": index,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
        })

    return structured_chunks


if __name__ == "__main__":
    import json

    sample_text = """
    Artificial intelligence is transforming modern software systems.
    RAG systems combine retrieval and generation.
    Embeddings help semantic search.
    Chunking is critical for retrieval quality.
    """ * 50

    chunks = chunk_text(
        text=sample_text,
        source_name="sample_policy.pdf",
        category="policy_test"
    )

    print(f"\nTotal chunks created: {len(chunks)}\n")

    if chunks:
        print(json.dumps(chunks[0], indent=4))
