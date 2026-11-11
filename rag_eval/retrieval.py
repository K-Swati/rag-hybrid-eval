"""Vector search (Qdrant), keyword search (BM25), hybrid fusion and reranking."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from rag_eval.chunking import Chunk
from rag_eval.embeddings import Embedder

_TOKEN = re.compile(r"[a-z0-9][a-z0-9_\-\.]*[a-z0-9]|[a-z0-9]")
STOPWORDS = frozenset(
    "a an and are as at be by do does for from how i in is it my of on or the to what when which who will with can "
    "you your should".split()
)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def content_terms(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class VectorIndex:
    """Dense retrieval backed by Qdrant (in-memory by default, or a server via ``url``)."""

    def __init__(self, embedder: Embedder, collection: str = "chunks", url: str | None = None) -> None:
        self.embedder = embedder
        self.collection = collection
        self.client = QdrantClient(url=url) if url else QdrantClient(":memory:")
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        vectors = self.embedder.embed([c.text for c in chunks])
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection, vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE)
        )
        points = [
            PointStruct(id=i, vector=vectors[i].tolist(), payload={"chunk_id": c.chunk_id, "doc_id": c.doc_id})
            for i, c in enumerate(chunks)
        ]
        self.client.upsert(self.collection, points=points)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        vector = self.embedder.embed([query])[0].tolist()
        res = self.client.query_points(self.collection, query=vector, limit=k).points
        return [Hit(self._chunks[p.id], float(p.score)) for p in res]


class BM25Index:
    """Keyword retrieval (Okapi BM25) - strong on exact identifiers such as error codes."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._bm25 = BM25Okapi([tokenize(c.text) or ["_"] for c in chunks])

    def search(self, query: str, k: int = 5) -> list[Hit]:
        assert self._bm25 is not None, "call build() first"
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:k]
        return [Hit(self._chunks[i], float(scores[i])) for i in order if scores[i] > 0]


def reciprocal_rank_fusion(rankings: list[list[Hit]], k: int = 60, top: int | None = None) -> list[Hit]:
    """Merge ranked lists: ``score(d) = sum 1 / (k + rank_i(d))``. Needs no score normalisation."""
    fused: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            fused[hit.chunk.chunk_id] = fused.get(hit.chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunks[hit.chunk.chunk_id] = hit.chunk
    ordered = sorted(fused.items(), key=lambda kv: -kv[1])
    return [Hit(chunks[cid], score) for cid, score in ordered[:top]]


class LexicalReranker:
    """Cheap second-stage reranker: query-term coverage blended with the first-stage rank.

    A stand-in for a cross-encoder. ``rerank`` has the same signature a
    cross-encoder wrapper would, so it can be swapped without touching callers.
    """

    def __init__(self, weight: float = 0.5) -> None:
        self.weight = weight

    def rerank(self, query: str, hits: list[Hit], top: int | None = None) -> list[Hit]:
        terms = set(content_terms(query))
        if not hits or not terms:
            return hits[:top]
        scored = []
        for rank, hit in enumerate(hits):
            tokens = set(tokenize(hit.chunk.text))
            coverage = len(terms & tokens) / len(terms)
            prior = 1.0 / (1 + rank)
            scored.append(Hit(hit.chunk, self.weight * coverage + (1 - self.weight) * prior))
        scored.sort(key=lambda h: -h.score)
        return scored[:top]


MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, qdrant_url: str | None = None) -> None:
        self.chunks = chunks
        self.dense = VectorIndex(embedder, url=qdrant_url)
        self.dense.build(chunks)
        self.sparse = BM25Index()
        self.sparse.build(chunks)
        self.reranker = LexicalReranker()

    def search(self, query: str, mode: str = "hybrid", k: int = 5, pool: int = 20) -> list[Hit]:
        if mode == "dense":
            return self.dense.search(query, k)
        if mode == "bm25":
            return self.sparse.search(query, k)
        if mode in ("hybrid", "hybrid_rerank"):
            fused = reciprocal_rank_fusion([self.dense.search(query, pool), self.sparse.search(query, pool)], top=pool)
            if mode == "hybrid":
                return fused[:k]
            return self.reranker.rerank(query, fused, top=k)
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
