# RAG Hybrid Search with an Evaluation Harness

A retrieval-augmented generation (RAG) service that answers questions over product documentation, plus a benchmark that **measures** how chunking strategy and retrieval method change quality, instead of guessing.

- **Vector search** on Qdrant, **keyword search** with BM25, merged with **Reciprocal Rank Fusion**, then an optional **reranker**
- **Three chunking strategies** compared head-to-head: fixed sliding window, recursive, heading-aware (with metadata)
- **Evaluation harness**: hit rate, precision@k, MRR, nDCG@k, plus answer accuracy and groundedness
- **Runs fully offline** (no API key) for tests and CI, and switches to **OpenAI embeddings and LLM** with two environment variables
- FastAPI service, Docker, GitHub Actions CI

## Architecture

```
 markdown docs ──> chunker (fixed | recursive | heading)
                      │
          ┌───────────┴───────────┐
   embeddings ──> Qdrant        BM25 index
          └───────────┬───────────┘
              Reciprocal Rank Fusion
                      │
                optional reranker
                      │
        generator (extractive | OpenAI)  ──>  answer + cited sources
```

## Quick start

```bash
pip install -r requirements-dev.txt
pytest -q                                  # 28 tests, fully offline
python -m rag_eval ask "How long is an OAuth access token valid?"
python -m rag_eval eval --generation       # writes reports/results.md
python -m rag_eval serve                   # http://localhost:8000/docs
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Which header tells me how long to wait after being throttled?"}'
```

Docker (with a real Qdrant server): `docker compose up --build`

### Using real models

```bash
export OPENAI_API_KEY=sk-...
export EMBEDDINGS=openai GENERATOR=openai
python -m rag_eval eval --generation
```

## Evaluation

The gold set is 36 questions over 11 documents. A chunk counts as relevant if it contains the question's gold answer phrase, so the benchmark does not depend on how the corpus was chunked. `tests/` checks that every gold phrase really exists in its document.

Results from `python -m rag_eval eval --generation` with the **offline** embedder and extractive generator (k = 3):

| Chunking | Retrieval | Hit@3 | MRR | nDCG@3 | Answer acc. |
|---|---|---|---|---|---|
| heading | dense | 0.972 | 0.889 | 0.910 | 0.667 |
| heading | hybrid + rerank | 0.944 | 0.889 | 0.903 | 0.639 |
| recursive | hybrid + rerank | 0.944 | 0.884 | 0.900 | 0.722 |
| recursive | dense | 0.972 | 0.810 | 0.852 | 0.722 |
| fixed | hybrid + rerank | 0.944 | 0.829 | 0.856 | 0.278 |
| fixed | dense | 0.917 | 0.810 | 0.838 | 0.250 |
| heading | bm25 | 0.889 | 0.769 | 0.800 | 0.556 |

Full table: [`reports/results.md`](reports/results.md).

### What the numbers show

1. **Chunking matters more than retrieval method here.** Heading-aware chunking has the best MRR (0.889) in every retrieval mode, because each section is one self-contained idea with its title attached.
2. **Retrieval quality can hide a generation problem.** Fixed windows retrieve almost as well (hit@3 about 0.92 to 0.94) but answer accuracy collapses to about 0.25, because windows cut sentences in half and the answer is split across chunks.
3. **BM25 alone is the weakest.** Questions are phrased differently from the docs, so exact keyword match misses them.

### Honest limitations

- **Hybrid search does not clearly beat dense search on this benchmark.** With 11 short documents, 36 questions and a *lexical* hashing embedder, differences of one or two questions are noise. Hybrid and reranking are expected to help most with real semantic embeddings and larger corpora, so re-run with `EMBEDDINGS=openai` before drawing conclusions.
- The default embedder is lexical (word and character n-grams). It handles word forms, not true synonyms.
- The default generator is extractive (it returns the best-matching sentence), so its answer accuracy is a floor, not what an LLM achieves.
- The reranker is a lightweight term-coverage scorer. It exposes the same `rerank(query, hits)` interface as a cross-encoder, so a real one can be dropped in.
- "Tokens" are whitespace words, not model tokens.

## Project layout

```
rag_eval/
  chunking.py    fixed / recursive / heading strategies with metadata
  embeddings.py  offline hashing embedder + OpenAI embedder
  retrieval.py   Qdrant dense index, BM25, RRF fusion, reranker
  generate.py    extractive baseline + OpenAI generator (cited, abstains)
  metrics.py     precision@k, hit rate, MRR, nDCG, groundedness
  evaluate.py    benchmark grid + dataset validation + reports
  api.py         FastAPI: POST /ask, GET /health
data/corpus/     11 sample docs (a fictional product, "Nimbus")
data/eval/       36 gold questions
tests/           28 tests
```

## Ideas for extension

- Replace the reranker with a cross-encoder (for example `bge-reranker`)
- Add an LLM-as-judge faithfulness metric next to the groundedness proxy
- Index a real corpus and compare chunk sizes on it
