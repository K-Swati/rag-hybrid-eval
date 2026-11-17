# Benchmark results

48 gold questions (20 numeric, 28 document). Router: `keywords`, SQL writer: `rules`, embeddings: `hashing`, k = 3.

| Score | Result | What it means |
|---|---|---|
| Routing | 100% (48/48) | questions sent to the right path |
| SQL answers | 90% (18/20) | numeric questions whose query matched the reference result |
| Document retrieval | 100% Hit@3 | document questions where the answer was in the top 3 chunks (heading + hybrid_rerank) |
| Best retrieval setup | 100% Hit@3, MRR 0.98 | heading + dense |

## Retrieval: chunking x mode

| Chunking | Retrieval | Chunks | Hit@3 | MRR | nDCG@3 | Answer acc. |
|---|---|---|---|---|---|---|
| heading | dense | 99 | 1.00 | 0.98 | 0.98 | 0.79 |
| heading | hybrid_rerank | 99 | 1.00 | 0.96 | 0.97 | 0.75 |
| heading | hybrid | 99 | 1.00 | 0.96 | 0.97 | 0.75 |
| recursive | dense | 72 | 1.00 | 0.91 | 0.93 | 0.79 |
| recursive | hybrid_rerank | 72 | 1.00 | 0.89 | 0.92 | 0.75 |
| recursive | hybrid | 72 | 0.96 | 0.88 | 0.90 | 0.75 |
| heading | bm25 | 99 | 0.93 | 0.91 | 0.92 | 0.75 |
| fixed | dense | 98 | 0.93 | 0.86 | 0.88 | 0.29 |
| fixed | hybrid_rerank | 98 | 0.93 | 0.84 | 0.87 | 0.29 |
| recursive | bm25 | 72 | 0.89 | 0.88 | 0.88 | 0.75 |
| fixed | hybrid | 98 | 0.89 | 0.83 | 0.84 | 0.29 |
| fixed | bm25 | 98 | 0.86 | 0.81 | 0.83 | 0.29 |

## Questions that went wrong

- **s18** Which distributor grew revenue the most from 2024 to 2025?: the rules SQL writer does not handle this question
- **s19** What share of 2025 revenue came from quick commerce?: the rules SQL writer does not handle this question
