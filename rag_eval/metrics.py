"""Retrieval and generation metrics.

Relevance is judged at the chunk level: a chunk is relevant to a question when
it contains the question's gold ``key`` phrase. That keeps the benchmark
independent of how the corpus was chunked, so strategies can be compared fairly.
"""

from __future__ import annotations

import math
import re

from rag_eval.retrieval import content_terms


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("`", "")).strip()


def is_relevant(chunk_text: str, key: str) -> bool:
    return normalize(key) in normalize(chunk_text)


def precision_at_k(relevance: list[bool], k: int) -> float:
    top = relevance[:k]
    return sum(top) / k if k else 0.0


def hit_rate_at_k(relevance: list[bool], k: int) -> float:
    return 1.0 if any(relevance[:k]) else 0.0


def reciprocal_rank(relevance: list[bool]) -> float:
    for rank, rel in enumerate(relevance, start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevance: list[bool], k: int) -> float:
    """Binary-gain nDCG. The ideal ranking puts all relevant results first."""
    gains = [1.0 if r else 0.0 for r in relevance[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(gains, reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def groundedness(answer: str, context: str) -> float:
    """Share of the answer's content words that appear in the retrieved context (hallucination proxy)."""
    terms = content_terms(answer)
    if not terms:
        return 0.0
    ctx = set(content_terms(context))
    return sum(t in ctx for t in terms) / len(terms)


def answer_correct(answer: str, key: str) -> bool:
    return is_relevant(answer, key)
