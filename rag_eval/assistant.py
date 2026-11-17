"""Service layer: answer any question about the company from Postgres.

    question ─▶ Router ─┬─ sql  ─▶ SqlWriter ─▶ guards ─▶ Postgres (read-only) ─▶ narrate
                        └─ docs ─▶ RagPipeline (hybrid search over the exported markdown) ─▶ generate

``Assistant.ask`` returns one ``Reply`` shape for both paths, so the API and
the UI do not need to know which path ran.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rag_eval.core.router import Route, Router
from rag_eval.core.sql import UnsafeSQL
from rag_eval.core.text_to_sql import SqlWriter, Vocabulary
from rag_eval.pipeline import PipelineRegistry
from rag_eval.warehouse.db import Warehouse
from rag_eval.warehouse.export import KbDoc, export_documents


@dataclass
class Reply:
    question: str
    route: Route
    answer: str
    generator: str = ""
    # sql path
    sql: str | None = None
    sql_explanation: str | None = None
    sql_writer: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    truncated: bool = False
    error: str | None = None
    # docs path
    sources: list[dict] = field(default_factory=list)
    trace: dict | None = None
    strategy: str | None = None
    mode: str | None = None
    notes: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


def load_knowledge_base(warehouse: Warehouse) -> list[KbDoc]:
    return export_documents(warehouse.fetch_all())


def build_vocabulary(warehouse: Warehouse) -> Vocabulary:
    t = warehouse.fetch_all(("products", "regions", "distributors", "plants"))
    return Vocabulary(
        products=[(p["sku"], p["name"], p["category"]) for p in t["products"]],
        regions=[(r["region_id"], r["name"]) for r in t["regions"]],
        distributors=[(d["distributor_id"], d["name"]) for d in t["distributors"]],
        plants=[(p["plant_id"], p["name"], p["city"]) for p in t["plants"]],
    )


class Assistant:
    def __init__(self, warehouse: Warehouse, registry: PipelineRegistry, router: Router, writer: SqlWriter) -> None:
        self.warehouse, self.registry, self.router, self.writer = warehouse, registry, router, writer

    def ask(self, question: str, route: str = "auto", mode: str = "hybrid_rerank", k: int = 3,
            strategy: str | None = None, trace: bool = False) -> Reply:
        t0 = time.perf_counter()
        if route in ("sql", "docs"):
            decided = Route(route, f"You chose the {'SQL' if route == 'sql' else 'documents'} path.", "user")
        else:
            decided = self.router.route(question)
        timings = {"route": round((time.perf_counter() - t0) * 1000, 1)}
        if decided.target == "sql":
            reply = self._sql(question, decided, timings)
            if reply is not None:
                return reply
            note = (f"The {self.writer.name} SQL writer could not turn this into a query, so it was answered from the "
                    "documents instead." + (" Set LLM=openai for full text-to-SQL." if self.writer.name == "rules" else ""))
            return self._docs(question, decided, mode, k, strategy, trace, timings, [note])
        return self._docs(question, decided, mode, k, strategy, trace, timings, [])

    def _sql(self, question: str, route: Route, timings: dict) -> Reply | None:
        t0 = time.perf_counter()
        plan = self.writer.write(question)
        timings["write_sql"] = round((time.perf_counter() - t0) * 1000, 1)
        if plan is None:
            return None
        reply = Reply(question, route, "", sql=plan.sql, sql_explanation=plan.explanation, sql_writer=plan.writer,
                      timings_ms=timings)
        try:
            result = self.warehouse.query(plan.sql)
        except UnsafeSQL as exc:
            reply.answer, reply.error = "The generated query was blocked because it was not a read-only SELECT.", str(exc)
            return reply
        except Exception as exc:  # syntax errors, unknown columns, timeouts: show them instead of failing the request
            reply.answer = "The query could not run on the database."
            reply.error = str(exc).strip().splitlines()[0]
            return reply
        timings["query"] = result.ms
        t1 = time.perf_counter()
        reply.columns, reply.rows, reply.truncated = result.columns, result.rows, result.truncated
        reply.answer = self.writer.narrate(question, plan, result)
        reply.generator = self.writer.name
        timings["narrate"] = round((time.perf_counter() - t1) * 1000, 1)
        if result.truncated:
            reply.notes.append(f"Showing the first {len(result.rows)} rows.")
        return reply

    def _docs(self, question, route, mode, k, strategy, trace, timings, notes) -> Reply:
        resp = self.registry.get(strategy).ask(question, mode=mode, k=k, trace=trace)
        return Reply(question, route, resp.answer, generator=resp.generator, sources=resp.sources, trace=resp.trace,
                     strategy=resp.strategy, mode=resp.mode, notes=notes, timings_ms={**timings, **resp.timings_ms})
