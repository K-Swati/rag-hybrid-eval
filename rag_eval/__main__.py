"""Command line: ``python -m rag_eval ask | eval | serve | db``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_eval.config import Settings


def main(argv: list[str] | None = None) -> int:
    settings = Settings.from_env()
    p = argparse.ArgumentParser(prog="rag_eval", description="Ask anything about Frostfield Creamery's data")
    sub = p.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.add_argument("--route", choices=["auto", "sql", "docs"], default="auto")
    ask.add_argument("--json", action="store_true", help="print the full reply as JSON")

    ev = sub.add_parser("eval", help="run the benchmark on the gold questions")
    ev.add_argument("-k", type=int, default=3)
    ev.add_argument("--reports", default=str(settings.reports_dir))
    ev.add_argument("--no-generation", action="store_true", help="skip scoring generated document answers")

    sv = sub.add_parser("serve", help="run the HTTP API and the web UI")
    sv.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 inside a container")
    sv.add_argument("--port", type=int, default=8000)

    db = sub.add_parser("db", help="database utilities")
    dbs = db.add_subparsers(dest="db_cmd", required=True)
    dbs.add_parser("seed-sql", help="regenerate db/init/02_seed.sql from rag_eval/warehouse/seed.py")
    init = dbs.add_parser("init", help="create the icecream database on a Postgres you run yourself")
    init.add_argument("--admin-url", required=True, help="e.g. postgresql://postgres:postgres@localhost:5432/postgres")
    init.add_argument("--reset", action="store_true", help="drop and recreate the database if it exists")
    loc = dbs.add_parser("local", help="create or reseed the local Postgres in .pgdata/ (no Docker)")
    loc.add_argument("--reset", action="store_true", help="drop and reseed the icecream database")
    exp = dbs.add_parser("export-md", help="write the knowledge base as markdown files")
    exp.add_argument("--out", default="data/kb")

    args = p.parse_args(argv)

    if args.cmd == "db":
        return _db(args, settings)

    from rag_eval.warehouse.db import WarehouseUnavailable
    from rag_eval.wiring import build_components

    settings = _with_reachable_database(settings)
    if settings is None:
        return 2

    if args.cmd == "serve":
        import uvicorn

        print(f"UI:  http://{'localhost' if args.host in ('0.0.0.0', '127.0.0.1') else args.host}:{args.port}/")
        print(f"API: http://localhost:{args.port}/docs")
        uvicorn.run("rag_eval.api:app", host=args.host, port=args.port)
        return 0

    try:
        c = build_components(settings)
    except WarehouseUnavailable as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.cmd == "ask":
        reply = c.assistant.ask(args.question, route=args.route)
        if args.json:
            print(json.dumps(reply.__dict__, indent=2, default=lambda o: o.__dict__))
            return 0
        print(f"[{reply.route.target}] {reply.route.reason}\n")
        print(reply.answer)
        if reply.sql:
            print("\n" + reply.sql)
        for s in reply.sources:
            print(f"  - {s['doc_id']} / {s['section'] or s['chunk_id']}  (score {s['score']})")
        for note in reply.notes:
            print(f"\nNote: {note}")
        return 0

    if args.cmd == "eval":
        from rag_eval.evaluate import run_benchmark, write_reports

        report = run_benchmark([d.to_document() for d in c.kb], c.questions, c.router, c.writer, c.warehouse,
                               c.embedder, None if args.no_generation else c.generator, k=args.k,
                               default_strategy=c.registry.default_strategy)
        write_reports(report, args.reports)
        print((Path(args.reports) / "results.md").read_text(encoding="utf-8"))
        return 0
    return 1


def _with_reachable_database(settings: Settings) -> Settings | None:
    """Use DATABASE_URL if it was set; otherwise use the default, or start a local Postgres.

    With no DATABASE_URL and nothing listening on the default (docker compose) address,
    a local Postgres is started from ``.pgdata/`` so ``serve``, ``ask`` and ``eval`` just work.
    """
    import dataclasses
    import os

    from rag_eval.warehouse.db import Warehouse, WarehouseUnavailable

    if os.environ.get("DATABASE_URL"):
        return settings
    try:
        Warehouse(settings.database_url).ping()
        return settings
    except WarehouseUnavailable:
        pass
    from rag_eval.warehouse.local import start_local_postgres

    print("No Postgres on localhost:5432 and DATABASE_URL is not set: starting a local one in .pgdata/ ...",
          file=sys.stderr)
    try:
        url = start_local_postgres()
    except Exception as exc:  # noqa: BLE001 - report any startup failure plainly
        print(f"Could not start a local Postgres: {exc}\n"
              "Start one with `docker compose up -d postgres`, or set DATABASE_URL.", file=sys.stderr)
        return None
    os.environ["DATABASE_URL"] = url  # the API builds its own Settings from the environment
    return dataclasses.replace(settings, database_url=url)


def _db(args, settings: Settings) -> int:
    root = Path(__file__).resolve().parent.parent
    if args.db_cmd == "local":
        from rag_eval.warehouse.local import start_local_postgres

        url = start_local_postgres(reset=args.reset)
        print(f"local Postgres ready ({'reseeded' if args.reset else 'data kept in .pgdata/'}); it stops when this exits")
        print(f"DATABASE_URL={url}")
        return 0
    if args.db_cmd == "seed-sql":
        from rag_eval.warehouse.seed import to_sql

        path = root / "db" / "init" / "02_seed.sql"
        path.write_text(to_sql(), encoding="utf-8")
        print(f"wrote {path}")
        return 0
    if args.db_cmd == "init":
        from rag_eval.warehouse.db import init_database

        url = init_database(args.admin_url, reset=args.reset)
        print(f"created {url.split('@')[-1]}; the app connects as rag_reader (see DATABASE_URL in .env.example)")
        return 0
    if args.db_cmd == "export-md":
        from rag_eval.assistant import load_knowledge_base
        from rag_eval.warehouse.db import Warehouse

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        docs = load_knowledge_base(Warehouse(settings.database_url))
        for d in docs:
            (out / f"{d.doc_id}.md").write_text(d.text, encoding="utf-8")
        print(f"wrote {len(docs)} markdown files to {out}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
