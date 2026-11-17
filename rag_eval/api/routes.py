"""HTTP endpoints. Thin: validate, call the service layer, shape the response."""

from __future__ import annotations

import json
from statistics import mean

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from rag_eval.api.schemas import AskRequest, AskResponse, Check, EvalRunRequest, EvalStatus, RouteInfo, Source
from rag_eval.api.state import AppState
from rag_eval.core import metrics as M
from rag_eval.core.chunking import STRATEGIES, chunk_documents, count_tokens
from rag_eval.core.retrieval import FINAL_STAGE, MODES
from rag_eval.core.sql import QueryResult, same_result

router = APIRouter()
EXAMPLES = {"sql": ["s01", "s02", "s05", "s06", "s13", "s20"], "docs": ["d01", "d06", "d16", "d18", "d14", "d17"]}


def _state(request: Request) -> AppState:
    return request.app.state.rag


# ---------- system ----------

@router.get("/health", tags=["system"])
def health(request: Request) -> dict:
    c = _state(request).c
    tables = c.warehouse.catalog()
    return {"status": "ok", "tables": len(tables), "rows": sum(t.rows for t in tables), "documents": len(c.kb),
            "chunks": len(c.registry.get().chunks)}


@router.get("/meta", tags=["system"])
def meta(request: Request) -> dict:
    """Everything a client needs to render its controls."""
    c = _state(request).c
    by_id = {q["id"]: q["question"] for q in c.questions}
    tables = c.warehouse.catalog()
    return {
        "company": "Frostfield Creamery",
        "modes": list(MODES), "final_stage": FINAL_STAGE, "strategies": sorted(STRATEGIES),
        "default_strategy": c.registry.default_strategy,
        "llm": c.settings.llm, "router": c.router.name, "sql_writer": c.writer.name,
        "generator": c.generator.name, "embedder": c.embedder.name,
        "database": c.warehouse.safe_url,
        "vector_store": "qdrant server" if c.settings.qdrant_url else "qdrant in-memory",
        "tables": len(tables), "rows": sum(t.rows for t in tables), "documents": len(c.kb),
        "questions": len(c.questions),
        "examples": {kind: [by_id[i] for i in ids if i in by_id] for kind, ids in EXAMPLES.items()},
    }


# ---------- question answering ----------

@router.post("/ask", response_model=AskResponse, tags=["assistant"])
def ask(req: AskRequest, request: Request) -> AskResponse:
    s = _state(request)
    reply = s.c.assistant.ask(req.question, route=req.route, mode=req.mode, k=req.k, strategy=req.strategy,
                              trace=req.trace)
    gold = s.gold_for(req.question)
    titles = {d.doc_id: d.title for d in s.c.kb}

    def source(d: dict) -> Source:
        rel = M.is_relevant(d["text"], gold["key"]) if gold and gold["route"] == "docs" else None
        return Source(**d, title=titles.get(d["doc_id"], d["doc_id"]), relevant=rel)

    sources = [source(d) for d in reply.sources]
    check = None
    if gold:
        check = Check(id=gold["id"], expected_route=gold["route"], route_ok=reply.route.target == gold["route"])
        if gold["route"] == "docs":
            check.key = gold["key"]
            check.retrieved = any(x.relevant for x in sources)
            check.correct = M.answer_correct(reply.answer, gold["key"])
        elif reply.sql and not reply.error:
            expected = s.c.warehouse.query(gold["reference_sql"])
            check.correct = same_result(expected, QueryResult(reply.columns, reply.rows))
        else:
            check.correct = False
    trace = {name: [source(d) for d in hits] for name, hits in reply.trace.items()} if reply.trace else None
    return AskResponse(
        question=reply.question, route=RouteInfo(**reply.route.__dict__), answer=reply.answer,
        generator=reply.generator, sql=reply.sql, sql_explanation=reply.sql_explanation, sql_writer=reply.sql_writer,
        columns=reply.columns, rows=reply.rows, truncated=reply.truncated, error=reply.error, sources=sources,
        trace=trace, strategy=reply.strategy, mode=reply.mode, notes=reply.notes, timings_ms=reply.timings_ms,
        check=check,
    )


# ---------- data ----------

@router.get("/data/tables", tags=["data"])
def data_tables(request: Request) -> dict:
    """The data dictionary: every table with its description, row count, columns and relationships."""
    tables = _state(request).c.warehouse.catalog()
    return {
        "tables": [{"name": t.name, "description": t.description, "rows": t.rows,
                    "columns": [c.__dict__ for c in t.columns]} for t in tables],
        "relationships": [{"from": f"{t.name}.{c.name}", "to": c.references}
                          for t in tables for c in t.columns if c.references],
    }


@router.get("/data/tables/{name}", tags=["data"])
def data_table_rows(name: str, request: Request, limit: int = 25, offset: int = 0) -> dict:
    wh = _state(request).c.warehouse
    try:
        table = wh.table(name)
    except KeyError:
        raise HTTPException(404, f"no table {name!r}") from None
    limit = max(1, min(limit, 200))
    result = wh.rows(name, limit=limit, offset=max(0, offset))
    return {"name": name, "description": table.description, "total": table.rows, "offset": offset, "limit": limit,
            "columns": result.columns, "rows": result.rows}


@router.get("/data/docs", tags=["data"])
def data_docs(request: Request) -> dict:
    """The knowledge base the document path searches, exported from Postgres."""
    c = _state(request).c
    docs = [d.to_document() for d in c.kb]
    stats = {}
    for name in sorted(STRATEGIES):
        chunks = chunk_documents(docs, name)
        stats[name] = {"chunks": len(chunks), "avg_tokens": round(mean(count_tokens(x.text) for x in chunks), 1)}
    asked = {}
    for q in c.questions:
        if q["route"] == "docs":
            asked[q["doc"]] = asked.get(q["doc"], 0) + 1
    return {
        "documents": [{"doc_id": d.doc_id, "title": d.title, "category": d.category, "source": d.source,
                       "words": count_tokens(d.text), "sections": d.text.count("\n## ") + d.text.startswith("## "),
                       "questions": asked.get(d.doc_id, 0)} for d in c.kb],
        "strategies": stats,
    }


@router.get("/data/docs/{doc_id}", tags=["data"])
def data_doc(doc_id: str, request: Request, strategy: str = "heading") -> dict:
    c = _state(request).c
    doc = next((d for d in c.kb if d.doc_id == doc_id), None)
    if doc is None:
        raise HTTPException(404, f"no document {doc_id!r}")
    if strategy not in STRATEGIES:
        raise HTTPException(422, f"strategy must be one of {sorted(STRATEGIES)}")
    chunks = chunk_documents([doc.to_document()], strategy)
    return {
        "doc_id": doc.doc_id, "title": doc.title, "category": doc.category, "source": doc.source, "text": doc.text,
        "strategy": strategy,
        "chunks": [{"chunk_id": x.chunk_id, "section": x.section, "text": x.text, "tokens": count_tokens(x.text)}
                   for x in chunks],
        "questions": [q for q in c.questions if q.get("doc") == doc.doc_id],
    }


# ---------- evaluation ----------

@router.get("/eval/questions", tags=["evaluation"])
def eval_questions(request: Request) -> list[dict]:
    return _state(request).c.questions


@router.get("/eval/results", tags=["evaluation"])
def eval_results(request: Request) -> dict:
    """The latest report written by ``python -m rag_eval eval`` or ``POST /eval/run``."""
    path = _state(request).c.settings.reports_dir / "report.json"
    if not path.exists():
        raise HTTPException(404, "no benchmark report yet; run `python -m rag_eval eval` or POST /eval/run")
    return json.loads(path.read_text(encoding="utf-8"))


@router.post("/eval/run", status_code=202, response_model=EvalStatus, tags=["evaluation"])
def eval_run(req: EvalRunRequest, request: Request):
    job = _state(request).job
    if not job.start(k=req.k, generation=req.generation):
        return JSONResponse(job.status(), status_code=409)
    return job.status()


@router.get("/eval/status", response_model=EvalStatus, tags=["evaluation"])
def eval_status(request: Request) -> dict:
    return _state(request).job.status()
