import pytest

from rag_eval.chunking import chunk_documents
from rag_eval.embeddings import HashingEmbedder, OpenAIEmbedder
from rag_eval.pipeline import load_corpus
from rag_eval.retrieval import (
    BM25Index,
    Hit,
    LexicalReranker,
    Retriever,
    reciprocal_rank_fusion,
    tokenize,
)


@pytest.fixture(scope="module")
def retriever():
    chunks = chunk_documents(load_corpus(), "heading")
    return Retriever(chunks, HashingEmbedder())


def test_hashing_embedder_shape_unit_norm_and_determinism():
    e = HashingEmbedder()
    v = e.embed(["rate limit", "rate limit", "totally different words"])
    assert v.shape == (3, e.dim)
    assert pytest.approx(1.0, abs=1e-5) == float((v[0] ** 2).sum())
    assert (v[0] == v[1]).all()
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])


def test_bm25_finds_exact_identifier(retriever):
    top = retriever.sparse.search("E1001", k=1)[0]
    assert "E1001" in top.chunk.text


def test_every_mode_returns_ranked_hits(retriever):
    for mode in ("dense", "bm25", "hybrid", "hybrid_rerank"):
        hits = retriever.search("How long is an access token valid?", mode=mode, k=3)
        assert 1 <= len(hits) <= 3
        assert any("3600 seconds" in h.chunk.text for h in hits), mode


def test_unknown_mode_raises(retriever):
    with pytest.raises(ValueError):
        retriever.search("x", mode="magic")


def test_rrf_rewards_agreement():
    chunks = chunk_documents(load_corpus(), "heading")[:3]
    a = [Hit(chunks[0], 1), Hit(chunks[1], 1), Hit(chunks[2], 1)]
    b = [Hit(chunks[1], 1), Hit(chunks[0], 1)]
    fused = reciprocal_rank_fusion([a, b])
    assert {fused[0].chunk.chunk_id, fused[1].chunk.chunk_id} == {chunks[0].chunk_id, chunks[1].chunk_id}
    assert fused[-1].chunk.chunk_id == chunks[2].chunk_id


def test_reranker_prefers_term_coverage():
    chunks = chunk_documents(load_corpus(), "heading")
    weak = next(c for c in chunks if c.doc_id == "billing-faq")
    strong = next(c for c in chunks if c.doc_id == "authentication" and "3600" in c.text)
    out = LexicalReranker(weight=0.9).rerank("access token expire 3600 seconds", [Hit(weak, 1), Hit(strong, 0.5)])
    assert out[0].chunk is strong


def test_tokenizer_keeps_codes_and_paths():
    assert "e1001" in tokenize("Error E1001!")
    assert "x-nimbus-signature" in tokenize("header X-Nimbus-Signature")


def test_empty_bm25_query_returns_nothing():
    idx = BM25Index()
    idx.build(chunk_documents(load_corpus(), "heading"))
    assert idx.search("zzzz-not-a-term", k=3) == []


def test_openai_embedder_batches_and_uses_model():
    calls = []

    class Item:
        def __init__(self, v):
            self.embedding = v

    class Fake:
        class embeddings:  # noqa: N801
            @staticmethod
            def create(model, input):
                calls.append((model, list(input)))
                return type("R", (), {"data": [Item([float(len(t)), 1.0]) for t in input]})

    emb = OpenAIEmbedder(model="m", client=Fake, batch_size=2)
    out = emb.embed(["a", "bb", "ccc"])
    assert out.shape == (3, 2) and emb.dim == 2
    assert [len(c[1]) for c in calls] == [2, 1] and calls[0][0] == "m"
