# Frostfield Assistant: ask anything about an ice cream company's data

[![CI](https://github.com/K-Swati/rag-hybrid-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/K-Swati/rag-hybrid-eval/actions/workflows/ci.yml)

An assistant that answers questions about **Frostfield Creamery**, a fictional Indian ice cream maker (FMCG), from a single **Postgres** database.

- **Number questions** ("Which region had the highest revenue in 2024?") become a read-only **SQL** query.
- **Questions about policies, products, suppliers and incidents** ("Why was the Raspberry Sorbet discontinued?") are answered with **hybrid RAG search** (dense + BM25 + rerank) over a markdown knowledge base exported from the same database.
- A **router** picks the path and says why. Every answer shows its working: the SQL and result table, or the source passages.
- A **benchmark** of 48 questions with known answers measures routing, SQL correctness and retrieval.

It runs fully **offline** by default, with no API key. Set `LLM=openai` to let a model route questions, write the SQL and phrase the answers.

## Quick start

You need Python 3.12 (3.9 also works). Docker is optional.

```bash
cd rag-hybrid-eval
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m rag_eval serve               # http://localhost:8000
```

If nothing is listening on localhost:5432 and `DATABASE_URL` is not set, `serve`, `ask` and `eval` start a local Postgres with `pgserver` (real Postgres binaries installed by pip). The first run creates and seeds it in `.pgdata/` (git-ignored); later runs reuse it. It stops when the app stops. To reseed it, run `python -m rag_eval db local --reset`.

With Docker, run `docker compose up -d postgres` first and the app uses that instead, or `docker compose up` to run everything in containers.

To use another Postgres 14+:

```bash
python -m rag_eval db init --admin-url postgresql://postgres:postgres@localhost:5432/postgres
export DATABASE_URL=postgresql://rag_reader:rag_reader@localhost:5432/icecream
```

## The web UI

| Tab | What it is for |
|---|---|
| **Ask** | Type a question or click an example. The answer card shows which path answered it and why: the result table and SQL for numbers, or the source passages with the matching words highlighted for documents. **Options** lets you force a path, change the search mode or chunking, and show the search steps. |
| **Data** | Browse everything the assistant knows: every table with its column meanings and rows, and every document, as readable text or split into the chunks search sees. |
| **Benchmark** | Three headline scores in plain words, every question with ✓/✗, and a comparison of search setups. **Run benchmark** re-runs it in the background. |

## The data

All of it is in Postgres. `db/init/` is loaded by Docker on first start.

| Table | Rows | What it holds |
|---|---|---|
| `products` | 18 | SKUs: cups, cones, sticks, tubs, kulfi, sorbets; MRP, cost, launch and discontinue dates |
| `sales` | 8,573 | Monthly units and revenue per SKU × distributor × channel, Jan 2024 to Dec 2025 |
| `production_batches` | 1,564 | Production runs per plant, with QC status |
| `quality_incidents` | 13 | What happened, root cause, corrective action |
| `promotions` | 9 | Mechanic, discount and outcome |
| `distributors`, `regions`, `plants`, `suppliers`, `product_ingredients` | | Master data |
| `documents` | 8 | Policies and SOPs in markdown: cold chain, recalls, returns, credit, pricing, launches, allergens, onboarding |

The data is generated deterministically by `rag_eval/warehouse/seed.py` and follows explainable rules: a summer peak from April to June, 12% growth in 2025, quick commerce growing fastest, promotions lifting volume, and tier C distributors never stocking tubs (as the onboarding policy says). After changing the generator, run `python -m rag_eval db seed-sql` and then `docker compose down -v && docker compose up -d postgres`.

Every table and column has a Postgres `COMMENT`. Those comments are the data dictionary: the Data tab displays them, and the OpenAI SQL writer reads them.

## Offline or OpenAI

| Part | Offline (default) | `LLM=openai` |
|---|---|---|
| Router | keyword cues, with the matched cues as the reason | model classifies; falls back to keywords on error |
| SQL writer | templates for totals, counts, rankings, trends and filters (product, category, region, channel, distributor, plant, year, quarter, month) | model writes SQL from the data dictionary |
| Answers | template sentence (SQL) / best-matching sentence (documents) | model writes the answer from the rows or passages |

The offline SQL writer declines questions it cannot express, such as growth rates, shares and averages. It says so, and the question is answered from the documents instead. It never guesses.

## Safety

Generated SQL passes four independent checks:
1. A guard that allows a single `SELECT` only (`core/sql.check_select`).
2. A read-only transaction.
3. A 5-second statement timeout.
4. A database role, `rag_reader`, that can only read.

The UI and API have **no authentication**. `serve` binds to 127.0.0.1, and Compose publishes ports on localhost only. Put it behind an auth proxy before exposing it, because `POST /eval/run` can trigger many OpenAI calls when `LLM=openai`.

## Benchmark

```bash
python -m rag_eval eval      # writes reports/results.md and reports/report.json
```

Offline results with the default settings (see [reports/results.md](reports/results.md)):

| Score | Result |
|---|---|
| Routing | 48/48 |
| SQL answers matching the reference query | 18/20 (the 2 misses are a growth question and a share question, which need `LLM=openai`) |
| Document questions with the answer in the top 3 | 28/28 |

These are good numbers, but read them with care. The keyword router and the SQL templates were written alongside these questions, so they measure coverage of the question set more than how the system handles questions it has never seen. Add your own questions to `data/eval/questions.json`: `route` plus `reference_sql` for SQL questions, or `doc` plus a `key` phrase for document questions.

## Commands

```
python -m rag_eval serve [--port 8000]          web UI + API (docs at /docs)
python -m rag_eval ask "question" [--route sql|docs] [--json]
python -m rag_eval eval [-k 3] [--no-generation]
python -m rag_eval db seed-sql                  regenerate db/init/02_seed.sql
python -m rag_eval db init --admin-url URL      load the database into your own Postgres
python -m rag_eval db export-md [--out data/kb] write the knowledge base as .md files
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q     # use python -m pytest; plain pytest can't find the package
ruff check .
```

Database tests start a throwaway Postgres with `pgserver` (installed by requirements.txt), so no Docker is needed, or use `TEST_ADMIN_URL` if it is set (CI uses a Postgres service container). See [ARCHITECTURE.md](ARCHITECTURE.md) for how the code is organised.

## License

MIT. See [LICENSE](LICENSE).
