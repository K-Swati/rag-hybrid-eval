"""Composition root: builds every component from ``Settings``, in one place.

The API and the CLI both call ``build_components``, so they always run the same
system. Swapping an implementation (router, SQL writer, generator, embedder)
means changing a setting, not code.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_eval.assistant import Assistant, build_vocabulary, load_knowledge_base
from rag_eval.config import Settings
from rag_eval.core.embeddings import Embedder, get_embedder
from rag_eval.core.generate import Generator, get_generator
from rag_eval.core.router import KeywordRouter, OpenAIRouter, Router
from rag_eval.core.text_to_sql import OpenAISqlWriter, RuleSqlWriter, SqlWriter
from rag_eval.evaluate import load_questions
from rag_eval.pipeline import PipelineRegistry
from rag_eval.warehouse.db import Warehouse
from rag_eval.warehouse.export import KbDoc


@dataclass
class Components:
    settings: Settings
    warehouse: Warehouse
    kb: list[KbDoc]
    questions: list[dict]
    embedder: Embedder
    generator: Generator
    router: Router
    writer: SqlWriter
    registry: PipelineRegistry
    assistant: Assistant


def make_router(name: str, warehouse: Warehouse) -> Router:
    if name == "keywords":
        return KeywordRouter()
    if name == "openai":
        return OpenAIRouter([t.name for t in warehouse.catalog()])
    raise ValueError(f"unknown router {name!r}; choose 'keywords' or 'openai'")


def make_writer(name: str, warehouse: Warehouse) -> SqlWriter:
    if name == "rules":
        return RuleSqlWriter(build_vocabulary(warehouse))
    if name == "openai":
        return OpenAISqlWriter(warehouse.schema_text())
    raise ValueError(f"unknown SQL writer {name!r}; choose 'rules' or 'openai'")


def build_components(settings: Settings, warehouse: Warehouse | None = None, index: bool = True) -> Components:
    """Connect to Postgres, export the knowledge base and build both answer paths."""
    warehouse = warehouse or Warehouse(settings.database_url)
    warehouse.ping()  # fail fast with a clear message if Postgres is not running
    kb = load_knowledge_base(warehouse)
    embedder, generator = get_embedder(settings.embeddings), get_generator(settings.generator)
    registry = PipelineRegistry([d.to_document() for d in kb], embedder, generator, settings.chunking,
                                settings.qdrant_url)
    if index:
        registry.get()  # index the default strategy now, not on the first question
    router, writer = make_router(settings.router, warehouse), make_writer(settings.sql_writer, warehouse)
    return Components(settings, warehouse, kb, load_questions(settings.questions_path), embedder, generator,
                      router, writer, registry, Assistant(warehouse, registry, router, writer))
