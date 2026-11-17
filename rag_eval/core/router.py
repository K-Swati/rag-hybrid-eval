"""Decides whether a question needs numbers from SQL or text from the documents.

* ``sql``  - totals, counts, rankings and trends computed from the warehouse tables
* ``docs`` - policies, procedures, descriptions and explanations found by hybrid RAG search

``KeywordRouter`` works offline and explains itself by listing the cues it
matched. ``OpenAIRouter`` asks a model and falls back to keywords on failure.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Literal, Protocol

Target = Literal["sql", "docs"]


@dataclass(frozen=True)
class Route:
    target: Target
    reason: str
    by: str  # keywords | openai | user


class Router(Protocol):
    name: str

    def route(self, question: str) -> Route: ...


# (pattern, weight, label shown to the user)
SQL_CUES = [
    (r"\bhow many\b(?! (?:hours?|days?|minutes?|weeks?|months?|years?|litres?|times?)\b)", 2, "how many"),
    (r"\bhow much\b(?! (?:time|notice)\b)", 1, "how much"),
    (r"\b(?:total|sum|overall)\b", 2, "total"),
    (r"\b(?:number of|count)\b", 2, "count"),
    (r"\b(?:average|avg|mean)\b", 1, "average"),
    (r"\btop\s*\d*\b|\bbottom\s+\d+\b", 2, "top / bottom N"),
    (r"\b(?:highest|lowest|most|least|best[- ]selling|worst[- ]selling|biggest|smallest|largest|rank)\b", 1, "ranking"),
    (r"\b(?:revenue|turnover|sales)\b", 2, "revenue / sales"),
    (r"\b(?:units?|volume|sold|sells?|selling)\b", 2, "units sold"),
    (r"\b(?:margin|profit)\b", 2, "margin"),
    (r"\b(?:monthly|per month|by month|each month|trend|over time)\b", 2, "trend"),
    (r"\b(?:by|per|each|across)\s+(?:region|product|channel|category|distributor|plant|flavour|month|year|quarter|severity)", 2, "breakdown"),
    (r"\b(?:compare|vs\.?|versus|growth|grew|share)\b", 2, "comparison"),
    (r"\bq[1-4]\b|\bquarter\b", 1, "quarter"),
    (r"\b20\d\d\b", 1, "year"),
    (r"\b(?:produced|production)\b", 2, "production"),
]
DOCS_CUES = [
    (r"\b(?:policy|policies|procedure|sop|playbook|process|rule|rules|guideline)\b", 3, "a policy or procedure"),
    (r"\b(?:why|explain|describe|tell me about)\b", 3, "an explanation"),
    (r"\b(?:what happens|what happened|what should|what must|how should|how do we|how often|how quickly|how long)\b", 3, "how something is done"),
    (r"\bhow many (?:hours?|days?|minutes?|weeks?|months?|years?|times?)\b", 3, "a time limit"),
    (r"\b(?:must|allowed|need|needs|required|requirements?|approve|approval|sign-off)\b", 2, "a rule"),
    (r"\b(?:root cause|caused|cause|corrective|went wrong|incident with|mechanic|outcome)\b", 3, "a cause or outcome"),
    (r"\b(?:who|supplies|supplier for|ingredients?|contains?|allergens?|sweeteners?|made with)\b", 2, "who supplies or what it contains"),
    (r"\b(?:temperature|shelf life|deposit|credit hold|credit period|days of credit|claim|recall|fssai|mrp|capacity|sales head|onboarding)\b", 2, "a policy term"),
    (r"\bwhat is (?:the|our)\b", 1, "a fact lookup"),
    (r"\bwhen (?:is|are|must|should|was)\b", 2, "a date or deadline"),
]


def _matches(question: str, cues) -> tuple[int, list[str]]:
    score, labels = 0, []
    for pattern, weight, label in cues:
        if re.search(pattern, question, re.IGNORECASE):
            score += weight
            if label not in labels:
                labels.append(label)
    return score, labels


class KeywordRouter:
    name = "keywords"

    def route(self, question: str) -> Route:
        sql, sql_labels = _matches(question, SQL_CUES)
        docs, docs_labels = _matches(question, DOCS_CUES)
        if sql > docs:
            return Route("sql", f"Asks for a number ({', '.join(sql_labels)}), so it is computed with SQL.", self.name)
        if docs_labels:
            why = ", ".join(docs_labels)
            return Route("docs", f"Looks for {why}, so the answer is looked up in the documents.", self.name)
        return Route("docs", "No numeric cue found, so the answer is looked up in the documents.", self.name)


ROUTER_PROMPT = """You route questions for an ice cream company's assistant.
Reply with JSON: {"target": "sql" | "docs", "reason": "<one short sentence>"}.
- "sql": the answer is a number, list or ranking computed from tables: sales, revenue, units, discounts, margin,
  production, counts of products/distributors/incidents/promotions, trends, comparisons.
- "docs": the answer is text: policies, procedures, time limits and rules in SOPs, product descriptions, prices,
  ingredients, suppliers, what happened in an incident and why, how a promotion worked.
Tables: {tables}."""


class OpenAIRouter:
    name = "openai"

    def __init__(self, tables: list[str], model: str | None = None, client=None) -> None:
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.tables = tables
        self.fallback = KeywordRouter()
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self._client = client

    def route(self, question: str) -> Route:
        try:
            resp = self._client.chat.completions.create(
                model=self.model, temperature=0, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": ROUTER_PROMPT.replace("{tables}", ", ".join(self.tables))},
                          {"role": "user", "content": question}],
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            if data.get("target") in ("sql", "docs"):
                return Route(data["target"], str(data.get("reason") or ""), self.name)
        except Exception:  # network, quota or parsing: keep answering offline
            pass
        return self.fallback.route(question)
