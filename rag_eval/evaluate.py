"""Benchmark every chunking strategy x retrieval mode on the gold question set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from rag_eval import metrics as M
from rag_eval.chunking import STRATEGIES, chunk_documents, count_tokens
from rag_eval.embeddings import Embedder, get_embedder
from rag_eval.generate import Generator
from rag_eval.pipeline import DEFAULT_CORPUS, DEFAULT_QA, load_corpus
from rag_eval.retrieval import MODES, Retriever


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


def load_qa(path: str | Path = DEFAULT_QA) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_dataset(corpus_dir: str | Path, qa: list[dict]) -> list[str]:
    """Every gold key phrase must appear in its gold document, otherwise the benchmark is wrong."""
    docs = {d.doc_id: d.text for d in load_corpus(corpus_dir)}
    problems = []
    for q in qa:
        if q["doc"] not in docs:
            problems.append(f"{q['id']}: unknown doc {q['doc']!r}")
        elif not M.is_relevant(docs[q["doc"]], q["key"]):
            problems.append(f"{q['id']}: key {q['key']!r} not found in {q['doc']}")
    return problems


def evaluate(
    corpus_dir: str | Path = DEFAULT_CORPUS,
    qa_path: str | Path = DEFAULT_QA,
    k: int = 3,
    strategies: list[str] | None = None,
    modes: list[str] | None = None,
    embedder: Embedder | None = None,
    generator: Generator | None = None,
) -> tuple[list[Row], list[dict]]:
    docs = load_corpus(corpus_dir)
    qa = load_qa(qa_path)
    problems = validate_dataset(corpus_dir, qa)
    if problems:
        raise ValueError("invalid evaluation set:\n" + "\n".join(problems))
    embedder = embedder or get_embedder()
    rows: list[Row] = []
    misses: list[dict] = []
    for strategy in strategies or list(STRATEGIES):
        chunks = chunk_documents(docs, strategy)
        retriever = Retriever(chunks, embedder)
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
            rows.append(
                Row(
                    strategy, mode, len(chunks), round(mean(count_tokens(c.text) for c in chunks), 1),
                    round(mean(hit), 3), round(mean(prec), 3), round(mean(mrr), 3), round(mean(ndcg), 3),
                    round(mean(acc), 3) if acc else None, round(mean(grd), 3) if grd else None,
                )
            )
    return rows, misses


def to_markdown(rows: list[Row], k: int, n_questions: int, embedder_name: str, generation: bool) -> str:
    head = ["Chunking", "Retrieval", "Chunks", "Avg tokens", f"Hit@{k}", f"P@{k}", "MRR", f"nDCG@{k}"]
    if generation:
        head += ["Answer acc.", "Grounded"]
    lines = [
        f"Embeddings: `{embedder_name}` | questions: {n_questions} | k = {k}",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "|".join(["---"] * len(head)) + "|",
    ]
    best = max(rows, key=lambda r: (r.mrr, r.ndcg))
    for r in sorted(rows, key=lambda r: (-r.mrr, -r.ndcg)):
        cells = [r.strategy, r.mode, str(r.chunks), str(r.avg_tokens), f"{r.hit_rate:.3f}", f"{r.precision:.3f}",
                 f"{r.mrr:.3f}", f"{r.ndcg:.3f}"]
        if generation:
            cells += [f"{r.answer_accuracy:.3f}", f"{r.groundedness:.3f}"]
        line = "| " + " | ".join(cells) + " |"
        lines.append(line.replace(r.strategy, f"**{r.strategy}**", 1) if r is best else line)
    return "\n".join(lines) + "\n"


def write_reports(rows: list[Row], misses: list[dict], out_dir: str | Path, **kw) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps([r.__dict__ for r in rows], indent=2), encoding="utf-8")
    (out / "results.md").write_text(to_markdown(rows, **kw), encoding="utf-8")
    (out / "misses.json").write_text(json.dumps(misses, indent=2), encoding="utf-8")
