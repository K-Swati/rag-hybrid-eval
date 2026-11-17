"""Request and response models: the HTTP contract, kept apart from the route logic."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Mode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]
Strategy = Literal["fixed", "recursive", "heading"]
RouteChoice = Literal["auto", "sql", "docs"]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    route: RouteChoice = "auto"  # auto = let the router decide
    mode: Mode = "hybrid_rerank"
    k: int = Field(default=3, ge=1, le=10)
    strategy: Strategy | None = None  # None = the server's default chunking strategy
    trace: bool = False  # also return every retrieval stage (dense, bm25, fused, reranked)


class RouteInfo(BaseModel):
    target: Literal["sql", "docs"]
    reason: str
    by: str


class Source(BaseModel):
    rank: int
    chunk_id: str
    doc_id: str
    title: str = ""
    section: str
    score: float
    text: str
    relevant: bool | None = None  # set when the question is a gold question


class Check(BaseModel):
    """How the answer scores, when the question is one of the gold questions."""

    id: str
    expected_route: str
    route_ok: bool
    correct: bool | None = None  # sql: result matches the reference query; docs: answer contains the key phrase
    retrieved: bool | None = None  # docs: a source contains the key phrase
    key: str | None = None


class AskResponse(BaseModel):
    question: str
    route: RouteInfo
    answer: str
    generator: str
    sql: str | None = None
    sql_explanation: str | None = None
    sql_writer: str | None = None
    columns: list[str] = []
    rows: list[list[Any]] = []
    truncated: bool = False
    error: str | None = None
    sources: list[Source] = []
    trace: dict[str, list[Source]] | None = None
    strategy: str | None = None
    mode: str | None = None
    notes: list[str] = []
    timings_ms: dict[str, float] = {}
    check: Check | None = None


class EvalRunRequest(BaseModel):
    k: int = Field(default=3, ge=1, le=10)
    generation: bool = True


class EvalStatus(BaseModel):
    state: Literal["idle", "running", "done", "failed"]
    done: int = 0
    total: int = 0
    k: int | None = None
    generation: bool | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
