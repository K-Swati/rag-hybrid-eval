"""FastAPI service: ``POST /ask`` and ``GET /health``."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_eval.pipeline import DEFAULT_CORPUS, RagPipeline, load_corpus
from rag_eval.retrieval import MODES


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: str = "hybrid_rerank"
    k: int = Field(default=3, ge=1, le=10)


class Source(BaseModel):
    chunk_id: str
    doc_id: str
    section: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


def create_app(pipeline: RagPipeline | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline or RagPipeline(
            load_corpus(os.environ.get("CORPUS_DIR", DEFAULT_CORPUS)),
            strategy=os.environ.get("CHUNKING", "heading"),
            qdrant_url=os.environ.get("QDRANT_URL") or None,
        )
        yield

    app = FastAPI(title="RAG hybrid search", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        p: RagPipeline = app.state.pipeline
        return {"status": "ok", "chunks": len(p.chunks), "strategy": p.strategy, "embedder": p.embedder.name}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        if req.mode not in MODES:
            raise HTTPException(422, f"mode must be one of {list(MODES)}")
        resp = app.state.pipeline.ask(req.question, mode=req.mode, k=req.k)
        return AskResponse(answer=resp.answer, sources=[Source(**s) for s in resp.sources])

    return app


app = create_app()
