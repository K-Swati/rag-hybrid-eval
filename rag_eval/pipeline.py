"""Service layer for the document path: chunk -> index -> retrieve -> generate.

``RagPipeline`` answers questions over one chunking strategy.
``PipelineRegistry`` holds one pipeline per strategy, built on first use and
sharing a single embedder and generator, so callers (the API, the UI) can
switch strategy per request.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from rag_eval.core.chunking import STRATEGIES, Chunk, Document, chunk_documents
from rag_eval.core.embeddings import Embedder, get_embedder
from rag_eval.core.generate import Generator, get_generator
from rag_eval.core.retrieval import FINAL_STAGE, MODES, Hit, Retriever


def load_corpus(path: str | Path) -> list[Document]:
    """Every ``*.md`` file in a folder as a document, e.g. a knowledge base written by ``rag_eval db export-md``."""
    files = sorted(Path(path).glob("*.md"))
    if not files:
        raise FileNotFoundError(f"no .md documents found in {path}")
    return [Document(f.stem, f.read_text(encoding="utf-8")) for f in files]


def hit_to_dict(hit: Hit, rank: int) -> dict:
    c = hit.chunk
    return {"rank": rank, "chunk_id": c.chunk_id, "doc_id": c.doc_id, "section": c.section,
            "score": round(hit.score, 4), "text": c.text}


@dataclass
class Response:
    question: str
    answer: str
    sources: list[dict]
    contexts: list[str]
    strategy: str = ""
    mode: str = ""
    generator: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)
    trace: dict[str, list[dict]] | None = None


class RagPipeline:
    def __init__(
        self,
        docs: list[Document],
        strategy: str = "heading",
        embedder: Embedder | None = None,
        generator: Generator | None = None,
        qdrant_url: str | None = None,
    ) -> None:
        self.strategy = strategy
        self.chunks: list[Chunk] = chunk_documents(docs, strategy)
        self.embedder = embedder or get_embedder()
        self.generator = generator or get_generator()
        self.retriever = Retriever(self.chunks, self.embedder, qdrant_url, collection=f"chunks_{strategy}")

    def retrieve(self, question: str, mode: str = "hybrid_rerank", k: int = 3) -> list[Hit]:
        return self.retriever.search(question, mode=mode, k=k)

    def ask(self, question: str, mode: str = "hybrid_rerank", k: int = 3, trace: bool = False) -> Response:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
        t0 = time.perf_counter()
        stages = None
        if trace:
            stages = self.retriever.trace(question, k=k)
            hits = stages[FINAL_STAGE[mode]][:k]
        else:
            hits = self.retrieve(question, mode, k)
        t1 = time.perf_counter()
        answer = self.generator.generate(question, hits)
        t2 = time.perf_counter()
        return Response(
            question=question,
            answer=answer.text,
            sources=[hit_to_dict(h, i) for i, h in enumerate(hits, start=1)],
            contexts=[h.chunk.text for h in hits],
            strategy=self.strategy,
            mode=mode,
            generator=answer.generator,
            timings_ms={"retrieve": round((t1 - t0) * 1000, 1), "generate": round((t2 - t1) * 1000, 1)},
            trace=None if stages is None else {
                name: [hit_to_dict(h, i) for i, h in enumerate(hits_, start=1)] for name, hits_ in stages.items()
            },
        )


class PipelineRegistry:
    """One ``RagPipeline`` per chunking strategy, built lazily and thread-safely."""

    def __init__(
        self,
        docs: list[Document],
        embedder: Embedder,
        generator: Generator,
        default_strategy: str = "heading",
        qdrant_url: str | None = None,
    ) -> None:
        if default_strategy not in STRATEGIES:
            raise ValueError(f"unknown chunking strategy {default_strategy!r}; choose from {sorted(STRATEGIES)}")
        self.docs = docs
        self.embedder = embedder
        self.generator = generator
        self.default_strategy = default_strategy
        self.qdrant_url = qdrant_url
        self._pipelines: dict[str, RagPipeline] = {}
        self._lock = threading.Lock()

    @classmethod
    def of(cls, pipeline: RagPipeline, docs: list[Document] | None = None) -> PipelineRegistry:
        """Wrap an already-built pipeline (handy in tests)."""
        reg = cls(docs or [], pipeline.embedder, pipeline.generator, pipeline.strategy)
        reg._pipelines[pipeline.strategy] = pipeline
        return reg

    def get(self, strategy: str | None = None) -> RagPipeline:
        strategy = strategy or self.default_strategy
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown chunking strategy {strategy!r}; choose from {sorted(STRATEGIES)}")
        with self._lock:
            if strategy not in self._pipelines:
                self._pipelines[strategy] = RagPipeline(
                    self.docs, strategy, self.embedder, self.generator, self.qdrant_url
                )
            return self._pipelines[strategy]

    @property
    def built(self) -> list[str]:
        return sorted(self._pipelines)
