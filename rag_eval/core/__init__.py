"""Core building blocks: chunking, embeddings, retrieval, generation, metrics.

This layer is pure library code. It never imports from ``rag_eval.pipeline``,
``rag_eval.evaluate``, ``rag_eval.api`` or ``rag_eval.config``
(``tests/test_architecture.py`` enforces this), so every piece can be unit
tested and reused on its own.
"""
