"""HTTP API and UI, against a real Postgres."""

import pytest
from fastapi.testclient import TestClient

from rag_eval.api import create_app
from rag_eval.config import Settings
from rag_eval.wiring import build_components


@pytest.fixture(scope="module")
def client(db_url, tmp_path_factory):
    settings = Settings(database_url=db_url, reports_dir=tmp_path_factory.mktemp("reports"))
    with TestClient(create_app(build_components(settings))) as c:
        yield c


def ask(client, question, **kw):
    r = client.post("/ask", json={"question": question, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_ui_is_served(client):
    html = client.get("/").text
    assert all(f'id="view-{v}"' in html for v in ("ask", "data", "benchmark"))


def test_meta_and_health(client):
    m = client.get("/meta").json()
    assert m["tables"] == 11 and m["documents"] == 14 and m["router"] == "keywords"
    assert m["examples"]["sql"] and m["examples"]["docs"]
    assert "rag_reader:***@" in m["database"]  # never leak the password
    assert client.get("/health").json()["status"] == "ok"


def test_numeric_question_goes_to_sql(client):
    r = ask(client, "Which region had the highest revenue in 2024?")
    assert r["route"]["target"] == "sql" and r["sql"].startswith("SELECT")
    assert r["columns"] == ["region", "net_revenue_inr"] and r["rows"][0][0] == "North"
    assert r["answer"].startswith("North had the highest net revenue in 2024")
    assert r["check"] == {"id": "s02", "expected_route": "sql", "route_ok": True, "correct": True,
                          "retrieved": None, "key": None}


def test_policy_question_goes_to_documents(client):
    r = ask(client, "How quickly must the recall committee be convened?")
    assert r["route"]["target"] == "docs" and r["sql"] is None
    assert r["sources"][0]["title"] == "Product Recall Procedure"
    assert r["check"]["retrieved"] and r["check"]["correct"]


def test_route_can_be_forced_and_trace_returned(client):
    r = ask(client, "Which region had the highest revenue in 2024?", route="docs", trace=True)
    assert r["route"] == {"target": "docs", "reason": "You chose the documents path.", "by": "user"}
    assert set(r["trace"]) == {"dense", "bm25", "fused", "reranked"}
    assert r["check"]["route_ok"] is False


def test_unsupported_sql_falls_back_with_a_note(client):
    r = ask(client, "What is our average monthly revenue?")
    assert r["route"]["target"] == "sql" and r["sql"] is None and r["sources"]
    assert "could not turn this into a query" in r["notes"][0]


def test_bad_input_is_422(client):
    assert client.post("/ask", json={"question": "hi there", "mode": "magic"}).status_code == 422
    assert client.post("/ask", json={"question": "x"}).status_code == 422


def test_data_endpoints(client):
    t = client.get("/data/tables").json()
    assert {x["name"] for x in t["tables"]} >= {"sales", "products", "documents"}
    assert {"from": "sales.sku", "to": "products.sku"} in t["relationships"]
    rows = client.get("/data/tables/sales?limit=3&offset=3").json()
    assert len(rows["rows"]) == 3 and rows["rows"][0][0] == 4 and rows["total"] > 8000
    assert client.get("/data/tables/pg_user").status_code == 404
    docs = client.get("/data/docs").json()
    assert len(docs["documents"]) == 14 and set(docs["strategies"]) == {"fixed", "heading", "recursive"}
    d = client.get("/data/docs/cold-chain-sop?strategy=heading").json()
    assert d["title"] == "Cold Chain SOP" and len(d["chunks"]) == 5 and d["questions"]
    assert client.get("/data/docs/nope").status_code == 404


def test_benchmark_runs_in_the_background(client):
    assert client.get("/eval/results").status_code == 404
    r = client.post("/eval/run", json={"k": 3, "generation": False})
    assert r.status_code == 202
    client.app.state.rag.job.wait(120)
    status = client.get("/eval/status").json()
    assert status["state"] == "done", status
    report = client.get("/eval/results").json()
    assert report["summary"]["routing"]["accuracy"] == 1.0
    assert report["summary"]["sql"]["correct"] >= 18
    assert report["summary"]["retrieval"]["default"]["hit_rate"] >= 0.9
