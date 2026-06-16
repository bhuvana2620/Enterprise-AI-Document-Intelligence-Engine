# src/embeddings/hash_embedding.py

import hashlib
import re
from typing import Union, List

import numpy as np


class HashEmbeddingModel:
    """
    Lightweight deterministic embedding model for free deployments.

    This avoids torch, transformers, and sentence-transformers on low-memory
    environments like Render free tier.

    It produces 384-dimensional vectors using token hashing.
    Retrieval quality is more keyword-style than semantic BGE embeddings,
    but it allows upload, indexing, Pinecone search, and query flow to work
    end-to-end for free.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.max_seq_length = 256

    def _tokenize(self, text: str) -> list[str]:
        if not text:
            return []

        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)

        tokens = self._tokenize(text)

        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            value = int(digest, 16)

            index = value % self.dimension
            sign = 1.0 if ((value // self.dimension) % 2 == 0) else -1.0

            vector[index] += sign

        norm = np.linalg.norm(vector)

        if norm > 0:
            vector = vector / norm

        return vector.astype(np.float32)

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 1,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        **kwargs
    ):
        if isinstance(sentences, str):
            return self._embed_one(sentences)

        if not sentences:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = np.vstack([
            self._embed_one(sentence)
            for sentence in sentences
        ])

        return embeddings