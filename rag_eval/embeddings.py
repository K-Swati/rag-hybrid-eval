"""Embedding backends.

* ``HashingEmbedder`` - fully offline. Word and character n-gram features are
  hashed into a fixed-size vector (the "hashing trick"). It is a *lexical*
  embedder: it handles typos and word forms ("retry"/"retried") but not true
  synonyms. It exists so the whole project, tests and CI run with no API key.
* ``OpenAIEmbedder`` - real semantic embeddings (``text-embedding-3-small`` by default).
"""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = 2048) -> None:
        half = dim // 2
        self.dim = half * 2
        self._words = HashingVectorizer(
            n_features=half, analyzer="word", stop_words="english", norm=None, alternate_sign=True,
            token_pattern=r"(?u)\b\w[\w\-]*\b",
        )
        self._chars = HashingVectorizer(
            n_features=half, analyzer="char_wb", ngram_range=(3, 5), norm=None, alternate_sign=True
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = hstack([self._words.transform(texts), self._chars.transform(texts)]).toarray().astype(np.float32)
        matrix = np.sign(matrix) * np.log1p(np.abs(matrix))  # sublinear term frequency
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, model: str | None = None, client=None, batch_size: int = 96) -> None:
        self.model = model or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        self.dim = 1536 if "3-small" in self.model else 3072 if "3-large" in self.model else 1536
        self.batch_size = batch_size
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self._client = client

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = self._client.embeddings.create(model=self.model, input=texts[i : i + self.batch_size])
            vectors.extend(item.embedding for item in resp.data)
        out = np.asarray(vectors, dtype=np.float32)
        if out.size:
            self.dim = out.shape[1]
        return out


def get_embedder(name: str | None = None) -> Embedder:
    name = (name or os.environ.get("EMBEDDINGS", "hashing")).lower()
    if name == "hashing":
        return HashingEmbedder()
    if name == "openai":
        return OpenAIEmbedder()
    raise ValueError(f"unknown embedder {name!r}; choose 'hashing' or 'openai'")
