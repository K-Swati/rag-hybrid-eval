"""SQL guard, result handling, router and rule-based SQL writer. No database needed."""

import json

import pytest

from rag_eval.config import ROOT, Settings
from rag_eval.core.router import KeywordRouter
from rag_eval.core.sql import QueryResult, UnsafeSQL, check_select, format_inr, format_value, indian_group, same_result
from rag_eval.core.text_to_sql import RuleSqlWriter, SqlPlan, Vocabulary, narrate_result
from rag_eval.warehouse.seed import DISTRIBUTORS, PLANTS, PRODUCTS, REGIONS

QUESTIONS = json.loads((ROOT / "data" / "eval" / "questions.json").read_text(encoding="utf-8"))
VOCAB = Vocabulary(
    products=[(p["sku"], p["name"], p["category"]) for p in PRODUCTS],
    regions=[(r["region_id"], r["name"]) for r in REGIONS],
    distributors=[(d["distributor_id"], d["name"]) for d in DISTRIBUTORS],
    plants=[(p["plant_id"], p["name"], p["city"]) for p in PLANTS],
)


# ---------- guard ----------

@pytest.mark.parametrize("sql", [
    "SELECT 1", "select * from sales;", "WITH t AS (SELECT 1) SELECT * FROM t",
    "SELECT 'delete from sales; drop table x' AS s",  # keywords inside a string are fine
])
def test_guard_accepts_single_select(sql):
    assert not check_select(sql).endswith(";")


@pytest.mark.parametrize("sql, why", [
    ("DELETE FROM sales", "SELECT"), ("SELECT 1; DROP TABLE sales", "one statement"),
    ("SELECT * INTO copy FROM sales", "INTO"), ("SELECT pg_sleep(10)", "PG_SLEEP"),
    ("WITH d AS (DELETE FROM sales RETURNING *) SELECT * FROM d", "DELETE"), ("", "empty"),
    ("SELECT 1 /* ; */ ; UPDATE sales SET units = 0", "one statement"),
])
def test_guard_rejects_anything_else(sql, why):
    with pytest.raises(UnsafeSQL, match=why):
        check_select(sql)


# ---------- results ----------

def test_indian_formatting():
    assert indian_group(12345678) == "1,23,45,678"
    assert indian_group(999) == "999"
    assert format_inr(49_900_000) == "₹4.99 crore"
    assert format_inr(682_000) == "₹6.82 lakh"
    assert format_value(2025, "year") == "2025"
    assert format_value("2025-05-01", "month") == "May 2025"


def test_same_result_ignores_aliases_order_and_extra_columns():
    expected = QueryResult(["name", "sum"], [["North", 10.0], ["West", 5.0]])
    got = QueryResult(["sku", "region", "net_revenue_inr"], [["x", "West", 5.04], ["y", "North", 9.96]])
    assert same_result(expected, got)
    assert not same_result(expected, QueryResult(["region", "rev"], [["North", 10.0]]))
    assert not same_result(expected, QueryResult(["region", "rev"], [["North", 10.0], ["West", 6.0]]))


# ---------- router ----------

def test_keyword_router_on_every_gold_question():
    wrong = [q["id"] for q in QUESTIONS if KeywordRouter().route(q["question"]).target != q["route"]]
    assert not wrong


def test_router_explains_itself():
    r = KeywordRouter().route("Total revenue by region in 2025")
    assert r.target == "sql" and "revenue" in r.reason and r.by == "keywords"
    assert KeywordRouter().route("How many days of credit do we give?").target == "docs"  # a time limit, not a count


# ---------- rule-based SQL writer ----------

def test_writer_builds_filters_and_breakdowns():
    plan = RuleSqlWriter(VOCAB).write("How many units of Mango Kulfi did we sell in the West region in 2024?")
    assert "SUM(s.units)" in plan.sql and "'FF-KUL-MNG-60'" in plan.sql and "'WEST'" in plan.sql and "2024" in plan.sql
    assert "GROUP BY" not in plan.sql
    plan = RuleSqlWriter(VOCAB).write("Top 3 products by revenue in Q2 2025")
    assert "GROUP BY product" in plan.sql and "LIMIT 3" in plan.sql and "QUARTER" in plan.sql
    plan = RuleSqlWriter(VOCAB).write("Which region had the lowest revenue?")
    assert "ASC" in plan.sql and "LIMIT 1" in plan.sql


def test_writer_declines_what_it_cannot_do():
    w = RuleSqlWriter(VOCAB)
    assert w.write("What share of revenue came from quick commerce?") is None
    assert w.write("Which distributor grew the most?") is None
    assert w.write("Tell me a joke") is None


def test_writer_output_passes_the_guard():
    w = RuleSqlWriter(VOCAB)
    for q in QUESTIONS:
        if q["route"] == "sql" and (plan := w.write(q["question"])):
            check_select(plan.sql)


def test_narration():
    plan = SqlPlan("", "", "rules", {"label": "net revenue", "scope": "in 2025", "limit": 1, "desc": True})
    assert narrate_result(plan, QueryResult(["net_revenue_inr"], [[49_900_000.0]])) == "Net revenue in 2025: ₹4.99 crore."
    top = narrate_result(plan, QueryResult(["region", "net_revenue_inr"], [["North", 16_300_000.0]]))
    assert top == "North had the highest net revenue in 2025: ₹1.63 crore."
    assert narrate_result(plan, QueryResult(["x"], [])) == "No matching rows in 2025."


# ---------- settings ----------

def test_llm_switch_sets_all_three_components():
    s = Settings.from_env({"LLM": "openai"})
    assert (s.router, s.sql_writer, s.generator) == ("openai", "openai", "openai")
    s = Settings.from_env({"LLM": "openai", "SQL_WRITER": "rules"})
    assert s.sql_writer == "rules"
    assert Settings.from_env({}).router == "keywords"
    with pytest.raises(ValueError):
        Settings.from_env({"LLM": "maybe"})
