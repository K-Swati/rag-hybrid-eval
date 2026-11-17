"""Benchmark on the gold questions in ``data/eval/questions.json``.

Three scores, each answering one question about the system:

* routing   - did the question go to the right path (SQL or documents)?
* sql       - for numeric questions, did the generated query return the same
              result as the hand-written reference query?
* retrieval - for document questions, did search find a chunk containing the
              answer? Measured for every chunking strategy x retrieval mode.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from rag_eval.config import ROOT
from rag_eval.core import metrics as M
from rag_eval.core.chunking import STRATEGIES, Document, chunk_documents, count_tokens
from rag_eval.core.embeddings import Embedder
from rag_eval.core.generate import Generator
from rag_eval.core.retrieval import MODES, Retriever
from rag_eval.core.router import Router
from rag_eval.core.sql import same_result
from rag_eval.core.text_to_sql import SqlWriter
from rag_eval.warehouse.db import Warehouse

DEFAULT_QUESTIONS = ROOT / "data" / "eval" / "questions.json"


@dataclass
class Row:
    strategy: str
    mode: str
    chunks: int
    avg_tokens: float
    hit_rate: float
    precision: float
    mrr: float
    ndcg: float
    answer_accuracy: float | None = None
    groundedness: float | None = None


def load_questions(path: str | Path = DEFAULT_QUESTIONS) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_questions(docs: list[Document], questions: list[dict]) -> list[str]:
    """Each document question's key phrase must be in its document; each SQL question needs a reference query."""
    by_id = {d.doc_id: d.text for d in docs}
    problems = []
    for q in questions:
        if q["route"] == "docs":
            if q.get("doc") not in by_id:
                problems.append(f"{q['id']}: unknown doc {q.get('doc')!r}")
            elif not M.is_relevant(by_id[q["doc"]], q["key"]):
                problems.append(f"{q['id']}: key {q['key']!r} not found in {q['doc']}")
        elif q["route"] == "sql":
            if not q.get("reference_sql"):
                problems.append(f"{q['id']}: missing reference_sql")
        else:
            problems.append(f"{q['id']}: route must be 'sql' or 'docs'")
    return problems


def evaluate_retrieval(docs: list[Document], questions: list[dict], embedder: Embedder, k: int = 3,
                       generator: Generator | None = None, strategies: list[str] | None = None,
                       modes: list[str] | None = None, progress: Callable[[], None] | None = None,
                       ) -> tuple[list[Row], list[dict]]:
    """Score every chunking strategy x retrieval mode on the document questions."""
    qa = [q for q in questions if q["route"] == "docs"]
    rows: list[Row] = []
    misses: list[dict] = []
    for strategy in strategies or list(STRATEGIES):
        chunks = chunk_documents(docs, strategy)
        retriever = Retriever(chunks, embedder, collection=f"eval_{strategy}")
        for mode in modes or list(MODES):
            hit, prec, mrr, ndcg, acc, grd = [], [], [], [], [], []
            for item in qa:
                hits = retriever.search(item["question"], mode=mode, k=k)
                rel = [M.is_relevant(h.chunk.text, item["key"]) for h in hits]
                hit.append(M.hit_rate_at_k(rel, k))
                prec.append(M.precision_at_k(rel, k))
                mrr.append(M.reciprocal_rank(rel))
                ndcg.append(M.ndcg_at_k(rel, k))
                if not any(rel):
                    misses.append({"strategy": strategy, "mode": mode, "id": item["id"], "question": item["question"]})
                if generator is not None:
                    answer = generator.generate(item["question"], hits)
                    acc.append(float(M.answer_correct(answer.text, item["key"])))
                    grd.append(M.groundedness(answer.text, " ".join(h.chunk.text for h in hits)))
            rows.append(Row(
                strategy, mode, len(chunks), round(mean(count_tokens(c.text) for c in chunks), 1),
                round(mean(hit), 3), round(mean(prec), 3), round(mean(mrr), 3), round(mean(ndcg), 3),
                round(mean(acc), 3) if acc else None, round(mean(grd), 3) if grd else None,
            ))
            if progress:
                progress()
    return rows, misses


def evaluate_questions(questions: list[dict], router: Router, writer: SqlWriter, warehouse: Warehouse,
                       progress: Callable[[], None] | None = None) -> list[dict]:
    """Per question: the route it took and, for SQL questions, whether the generated query was right."""
    out = []
    for q in questions:
        route = router.route(q["question"])
        item = {"id": q["id"], "question": q["question"], "expected_route": q["route"], "route": route.target,
                "route_ok": route.target == q["route"], "reason": route.reason}
        if q["route"] == "sql":
            plan = writer.write(q["question"])
            item["sql"] = plan.sql if plan else None
            if plan is None:
                item.update(sql_ok=False, error=f"the {writer.name} SQL writer does not handle this question")
            else:
                try:
                    expected = warehouse.query(q["reference_sql"])
                    got = warehouse.query(plan.sql)
                    item["sql_ok"] = same_result(expected, got)
                    if not item["sql_ok"]:
                        item["error"] = f"result differs from the reference query ({len(got.rows)} vs {len(expected.rows)} rows)"
                except Exception as exc:
                    item.update(sql_ok=False, error=str(exc).strip().splitlines()[0])
        out.append(item)
        if progress:
            progress()
    return out


def _score(n: int, total: int) -> dict:
    return {"correct": n, "total": total, "accuracy": round(n / total, 3) if total else None}


def run_benchmark(docs: list[Document], questions: list[dict], router: Router, writer: SqlWriter,
                  warehouse: Warehouse, embedder: Embedder, generator: Generator | None, k: int = 3,
                  default_strategy: str = "heading", default_mode: str = "hybrid_rerank",
                  progress: Callable[[int, int], None] | None = None) -> dict:
    problems = validate_questions(docs, questions)
    if problems:
        raise ValueError("invalid question set:\n" + "\n".join(problems))
    total = len(questions) + len(STRATEGIES) * len(MODES)
    done = 0

    def tick() -> None:
        nonlocal done
        done += 1
        if progress:
            progress(done, total)

    items = evaluate_questions(questions, router, writer, warehouse, tick)
    rows, misses = evaluate_retrieval(docs, questions, embedder, k, generator, progress=tick)
    docs_q = [q for q in questions if q["route"] == "docs"]
    default = next(r for r in rows if r.strategy == default_strategy and r.mode == default_mode)
    best = max(rows, key=lambda r: (r.hit_rate, r.mrr))
    missed = {(m["strategy"], m["mode"], m["id"]) for m in misses}
    for item in items:
        if item["expected_route"] == "docs":
            item["retrieved"] = (default_strategy, default_mode, item["id"]) not in missed
    sql_items = [i for i in items if i["expected_route"] == "sql"]
    return {
        "meta": {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017
                 "k": k, "embedder": embedder.name, "generator": generator.name if generator else None,
                 "router": router.name, "sql_writer": writer.name, "questions": len(questions),
                 "sql_questions": len(sql_items), "docs_questions": len(docs_q),
                 "default_strategy": default_strategy, "default_mode": default_mode},
        "summary": {
            "routing": _score(sum(i["route_ok"] for i in items), len(items)),
            "sql": _score(sum(bool(i.get("sql_ok")) for i in sql_items), len(sql_items)),
            "retrieval": {"default": asdict(default), "best": asdict(best)},
        },
        "questions": items,
        "retrieval_grid": [asdict(r) for r in rows],
        "misses": misses,
    }


def to_markdown(report: dict) -> str:
    m, s = report["meta"], report["summary"]
    k = m["k"]

    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x * 100:.0f}%"

    d, b = s["retrieval"]["default"], s["retrieval"]["best"]
    lines = [
        "# Benchmark results", "",
        f"{m['questions']} gold questions ({m['sql_questions']} numeric, {m['docs_questions']} document). "
        f"Router: `{m['router']}`, SQL writer: `{m['sql_writer']}`, embeddings: `{m['embedder']}`, k = {k}.", "",
        "| Score | Result | What it means |", "|---|---|---|",
        f"| Routing | {pct(s['routing']['accuracy'])} ({s['routing']['correct']}/{s['routing']['total']}) | questions sent to the right path |",
        f"| SQL answers | {pct(s['sql']['accuracy'])} ({s['sql']['correct']}/{s['sql']['total']}) | numeric questions whose query matched the reference result |",
        f"| Document retrieval | {pct(d['hit_rate'])} Hit@{k} | document questions where the answer was in the top {k} chunks ({m['default_strategy']} + {m['default_mode']}) |",
        f"| Best retrieval setup | {pct(b['hit_rate'])} Hit@{k}, MRR {b['mrr']:.2f} | {b['strategy']} + {b['mode']} |",
        "", "## Retrieval: chunking x mode", "",
        f"| Chunking | Retrieval | Chunks | Hit@{k} | MRR | nDCG@{k} | Answer acc. |", "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(report["retrieval_grid"], key=lambda r: (-r["hit_rate"], -r["mrr"])):
        acc = "n/a" if r["answer_accuracy"] is None else f"{r['answer_accuracy']:.2f}"
        lines.append(f"| {r['strategy']} | {r['mode']} | {r['chunks']} | {r['hit_rate']:.2f} | {r['mrr']:.2f} | "
                     f"{r['ndcg']:.2f} | {acc} |")
    wrong = [i for i in report["questions"] if not i["route_ok"] or i.get("sql_ok") is False]
    if wrong:
        lines += ["", "## Questions that went wrong", ""]
        for i in wrong:
            why = f"routed to {i['route']}" if not i["route_ok"] else i.get("error", "wrong result")
            lines.append(f"- **{i['id']}** {i['question']}: {why}")
    return "\n".join(lines) + "\n"


def write_reports(report: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "results.md").write_text(to_markdown(report), encoding="utf-8")
