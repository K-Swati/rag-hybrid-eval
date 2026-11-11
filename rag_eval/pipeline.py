"""End-to-end RAG pipeline: load -> chunk -> index -> retrieve -> generate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_eval.chunking import Chunk, Document, chunk_documents
from rag_eval.embeddings import Embedder, get_embedder
from rag_eval.generate import Generator, get_generator
from rag_eval.retrieval import Hit, Retriever

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = ROOT / "data" / "corpus"
DEFAULT_QA = ROOT / "data" / "eval" / "qa.json"


def load_corpus(path: str | Path = DEFAULT_CORPUS) -> list[Document]:
    files = sorted(Path(path).glob("*.md"))
    if not files:
        raise FileNotFoundError(f"no .md documents found in {path}")
    return [Document(f.stem, f.read_text(encoding="utf-8")) for f in files]


@dataclass
class Response:
    question: str
    answer: str
    sources: list[dict]
    contexts: list[str]


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
        self.retriever = Retriever(self.chunks, self.embedder, qdrant_url)

    def retrieve(self, question: str, mode: str = "hybrid_rerank", k: int = 3) -> list[Hit]:
        return self.retriever.search(question, mode=mode, k=k)

    def ask(self, question: str, mode: str = "hybrid_rerank", k: int = 3) -> Response:
        hits = self.retrieve(question, mode, k)
        answer = self.generator.generate(question, hits)
        return Response(
            question=question,
            answer=answer.text,
            sources=[
                {"chunk_id": h.chunk.chunk_id, "doc_id": h.chunk.doc_id, "section": h.chunk.section, "score": round(h.score, 4)}
                for h in hits
            ],
            contexts=[h.chunk.text for h in hits],
        )
