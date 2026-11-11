import pytest
from fastapi.testclient import TestClient

from rag_eval.api import create_app
from rag_eval.embeddings import HashingEmbedder
from rag_eval.evaluate import evaluate, load_qa, validate_dataset
from rag_eval.generate import NO_ANSWER, ExtractiveGenerator, OpenAIGenerator
from rag_eval.pipeline import DEFAULT_CORPUS, RagPipeline, load_corpus


@pytest.fixture(scope="module")
def pipe():
    return RagPipeline(load_corpus(), "heading", HashingEmbedder(), ExtractiveGenerator())


def test_gold_dataset_is_consistent():
    qa = load_qa()
    assert len(qa) >= 30
    assert validate_dataset(DEFAULT_CORPUS, qa) == []
    assert len({q["id"] for q in qa}) == len(qa)


@pytest.mark.parametrize(
    "question,expected",
    [
        ("How long is an OAuth access token valid?", "3600 seconds"),
        ("What does error E1001 indicate?", "clock is skewed"),
        ("When will API v1 stop working?", "March 31"),
    ],
)
def test_ask_returns_grounded_answer(pipe, question, expected):
    resp = pipe.ask(question)
    assert expected in resp.answer
    assert resp.sources and resp.contexts


def test_extractive_generator_abstains_without_context():
    assert ExtractiveGenerator().generate("anything", []).text == NO_ANSWER


def test_openai_generator_builds_cited_prompt():
    seen = {}

    class Fake:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kw):
                    seen.update(kw)
                    msg = type("M", (), {"content": " 3600 seconds [1] "})
                    return type("R", (), {"choices": [type("C", (), {"message": msg})]})

    hits = RagPipeline(load_corpus(), "heading", HashingEmbedder()).retrieve("token expiry", "dense", 2)
    out = OpenAIGenerator(model="m", client=Fake).generate("token expiry?", hits)
    assert out.text == "3600 seconds [1]"
    assert seen["temperature"] == 0 and seen["model"] == "m"
    assert "[1]" in seen["messages"][1]["content"] and "ONLY" in seen["messages"][0]["content"]


def test_evaluate_runs_grid_and_scores_are_valid():
    rows, _ = evaluate(k=3, strategies=["heading"], modes=["dense", "hybrid"], embedder=HashingEmbedder(),
                       generator=ExtractiveGenerator())
    assert len(rows) == 2
    for r in rows:
        assert 0 <= r.mrr <= 1 and 0 <= r.ndcg <= 1 and r.hit_rate >= 0.8
        assert r.answer_accuracy is not None and r.answer_accuracy >= 0.6


def test_evaluate_rejects_broken_dataset(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\n## S\nhello world")
    qa = tmp_path / "qa.json"
    qa.write_text('[{"id":"q1","question":"?","doc":"a","key":"missing phrase"}]')
    with pytest.raises(ValueError, match="invalid evaluation set"):
        evaluate(tmp_path, qa, embedder=HashingEmbedder())


def test_api_health_ask_and_validation(pipe):
    with TestClient(create_app(pipe)) as client:
        assert client.get("/health").json()["status"] == "ok"
        body = client.post("/ask", json={"question": "How many requests per minute on Pro?"}).json()
        assert "600 requests per minute" in body["answer"]
        assert body["sources"][0]["doc_id"]
        assert client.post("/ask", json={"question": "ok?", "mode": "nope"}).status_code == 422
        assert client.post("/ask", json={"question": "x"}).status_code == 422
