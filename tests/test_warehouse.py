"""Seed data, Postgres schema, read-only access and the end-to-end benchmark."""

from datetime import date

import psycopg
import pytest

from rag_eval.assistant import build_vocabulary, load_knowledge_base
from rag_eval.config import ROOT
from rag_eval.core.router import KeywordRouter
from rag_eval.core.sql import UnsafeSQL
from rag_eval.core.text_to_sql import RuleSqlWriter
from rag_eval.evaluate import evaluate_questions, load_questions, validate_questions
from rag_eval.warehouse.db import Warehouse
from rag_eval.warehouse.export import export_documents
from rag_eval.warehouse.seed import build_tables, to_sql

TABLES = build_tables()


# ---------- seed data (no database) ----------

def test_committed_seed_file_is_up_to_date():
    committed = (ROOT / "db" / "init" / "02_seed.sql").read_text(encoding="utf-8")
    assert committed == to_sql(), "run `python -m rag_eval db seed-sql`"


def test_seed_follows_its_business_rules():
    products = {p["sku"]: p for p in TABLES["products"]}
    tiers = {d["distributor_id"]: d["cold_chain_tier"] for d in TABLES["distributors"]}
    for s in TABLES["sales"]:
        p = products[s["sku"]]
        assert not (p["category"] == "tub" and tiers[s["distributor_id"]] == "C"), "tier C may not stock tubs"
        assert date.fromisoformat(s["sale_month"]) >= date.fromisoformat(p["launch_date"]).replace(day=1)
        if p["discontinued_on"]:
            assert date.fromisoformat(s["sale_month"]) <= date.fromisoformat(p["discontinued_on"])
        assert abs(s["gross_revenue_inr"] - s["discount_inr"] - s["net_revenue_inr"]) < 0.011


def test_knowledge_base_contains_every_gold_answer():
    docs = [d.to_document() for d in export_documents(TABLES)]
    assert not validate_questions(docs, load_questions())


def test_export_has_one_section_per_record():
    docs = {d.doc_id: d for d in export_documents(TABLES)}
    assert docs["product-catalogue"].text.count("\n## ") == len(TABLES["products"])
    assert docs["quality-incidents"].text.count("\n## ") == len(TABLES["quality_incidents"])
    assert {d.category for d in docs.values()} == {"policy", "catalogue", "records"}


# ---------- Postgres ----------

@pytest.fixture(scope="module")
def wh(db_url):
    return Warehouse(db_url)


def test_catalog_reads_the_data_dictionary(wh):
    tables = {t.name: t for t in wh.catalog()}
    assert len(tables) == 11 and all(t.description for t in tables.values())
    assert tables["sales"].rows == len(TABLES["sales"])
    fk = {c.name: c.references for c in tables["sales"].columns}
    assert fk["sku"] == "products.sku" and fk["distributor_id"] == "distributors.distributor_id"
    assert "net_revenue_inr" in wh.schema_text()


def test_reader_role_cannot_write(db_url, wh):
    with pytest.raises(UnsafeSQL):
        wh.query("UPDATE sales SET units = 0")
    with psycopg.connect(db_url) as conn, pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        conn.execute("UPDATE sales SET units = 0")


def test_query_caps_rows(wh):
    r = wh.query("SELECT sale_id FROM sales", max_rows=10)
    assert len(r.rows) == 10 and r.truncated


def test_export_from_postgres_matches_export_from_seed(wh):
    assert [d.text for d in load_knowledge_base(wh)] == [d.text for d in export_documents(TABLES)]


def test_offline_router_and_sql_writer_on_gold_questions(wh):
    items = evaluate_questions(load_questions(), KeywordRouter(), RuleSqlWriter(build_vocabulary(wh)), wh)
    assert all(i["route_ok"] for i in items)
    sql = [i for i in items if i["expected_route"] == "sql"]
    wrong = {i["id"] for i in sql if not i["sql_ok"]}
    assert wrong <= {"s18", "s19"}, wrong  # growth and share questions need the OpenAI writer
