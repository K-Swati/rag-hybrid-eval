"""Command line: ``python -m rag_eval ask|eval|serve``."""

from __future__ import annotations

import argparse
import json
import sys

from rag_eval.chunking import STRATEGIES
from rag_eval.embeddings import get_embedder
from rag_eval.evaluate import evaluate, load_qa, write_reports
from rag_eval.generate import get_generator
from rag_eval.pipeline import DEFAULT_CORPUS, RagPipeline, load_corpus
from rag_eval.retrieval import MODES


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rag_eval", description="Hybrid-search RAG with an evaluation harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.add_argument("--strategy", choices=sorted(STRATEGIES), default="heading")
    ask.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    ask.add_argument("-k", type=int, default=3)
    ask.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ask.add_argument("--json", action="store_true", help="print the full response as JSON")

    ev = sub.add_parser("eval", help="benchmark chunking x retrieval on the gold set")
    ev.add_argument("-k", type=int, default=3)
    ev.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ev.add_argument("--reports", default="reports")
    ev.add_argument("--generation", action="store_true", help="also score generated answers")

    sv = sub.add_parser("serve", help="run the HTTP API")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=8000)

    args = p.parse_args(argv)

    if args.cmd == "ask":
        pipe = RagPipeline(load_corpus(args.corpus), strategy=args.strategy)
        resp = pipe.ask(args.question, mode=args.mode, k=args.k)
        if args.json:
            print(json.dumps(resp.__dict__, indent=2))
        else:
            print(resp.answer)
            for s in resp.sources:
                print(f"  - {s['doc_id']} / {s['section'] or s['chunk_id']}  (score {s['score']})")
        return 0

    if args.cmd == "eval":
        embedder = get_embedder()
        generator = get_generator() if args.generation else None
        rows, misses = evaluate(args.corpus, k=args.k, embedder=embedder, generator=generator)
        write_reports(rows, misses, args.reports, k=args.k, n_questions=len(load_qa()),
                      embedder_name=embedder.name, generation=generator is not None)
        print(open(f"{args.reports}/results.md", encoding="utf-8").read())
        return 0

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("rag_eval.api:app", host=args.host, port=args.port)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
