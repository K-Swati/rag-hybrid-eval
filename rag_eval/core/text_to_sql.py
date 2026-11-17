"""Turns a numeric question into one read-only SELECT, and the result into a sentence.

* ``RuleSqlWriter`` works offline. It recognises a metric (revenue, units,
  discount, margin, production, counts), filters (product, category, region,
  channel, distributor, plant, year, quarter, month) and up to two breakdowns,
  and builds the query from templates. Questions outside that (growth rates,
  shares, averages) return ``None`` and the caller says so.
* ``OpenAISqlWriter`` sends the data dictionary to a model and lets it write
  the SQL. Either way the query then passes the guards in ``core.sql``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Protocol

from rag_eval.core.sql import QueryResult, format_value


@dataclass(frozen=True)
class Vocabulary:
    """Names the rule writer can recognise, read from the warehouse at start-up."""

    products: list[tuple[str, str, str]]  # (sku, name, category)
    regions: list[tuple[str, str]]  # (region_id, name)
    distributors: list[tuple[str, str]]  # (distributor_id, name)
    plants: list[tuple[str, str, str]]  # (plant_id, name, city)


@dataclass
class SqlPlan:
    sql: str
    explanation: str
    writer: str
    meta: dict = field(default_factory=dict)


class SqlWriter(Protocol):
    name: str

    def write(self, question: str) -> SqlPlan | None: ...

    def narrate(self, question: str, plan: SqlPlan, result: QueryResult) -> str: ...


def _q(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
     "december"], start=1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})  # "jan", "feb", ...
ORDINAL = {"first": 1, "second": 2, "third": 3, "fourth": 4, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
CATEGORIES = {"cup": "cup", "cone": "cone", "stick": "stick", "bar": "stick", "lolly": "stick", "tub": "tub",
              "kulfi": "kulfi", "sorbet": "sorbet"}
CHANNELS = [
    (r"modern[ -]trade|supermarkets?|hypermarkets?", "modern_trade", "modern trade"),
    (r"general[ -]trade|corner shops?|kiranas?", "general_trade", "general trade"),
    (r"quick[ -]commerce|q-commerce|delivery apps?|instant delivery", "quick_commerce", "quick commerce"),
    (r"horeca|hotels?|restaurants?|cafes?", "horeca", "HoReCa"),
]
UNSUPPORTED = re.compile(r"\b(grow|grew|growth|share|percent|percentage|average|avg|mean|increase|decrease|change|"
                         r"ratio|per cent)\b|%")

# metric key -> (sql over alias s/p/b, column alias, label)
SALES_METRICS = {
    "margin": ("ROUND(SUM(s.net_revenue_inr - s.units * p.unit_cost_inr), 2)", "gross_margin_inr", "gross margin"),
    "discount": ("ROUND(SUM(s.discount_inr), 2)", "discount_inr", "discount given"),
    "gross": ("ROUND(SUM(s.gross_revenue_inr), 2)", "gross_revenue_inr", "gross revenue"),
    "revenue": ("ROUND(SUM(s.net_revenue_inr), 2)", "net_revenue_inr", "net revenue"),
    "units": ("SUM(s.units)", "units", "units sold"),
}


@dataclass
class _Parse:
    text: str  # working copy; matched phrases are blanked out so they are not matched twice
    filters: list[tuple[str, str, str]] = field(default_factory=list)  # (kind, sql, phrase)
    joins: set[str] = field(default_factory=set)

    def take(self, pattern: str) -> re.Match | None:
        m = re.search(pattern, self.text)
        if m:
            self.text = self.text[: m.start()] + " " * (m.end() - m.start()) + self.text[m.end():]
        return m


_SCOPE_ORDER = ["product", "category", "distributor", "region", "plant", "channel", "severity", "qc", "status",
                "tier", "season", "month", "quarter", "year"]


def _scope(filters: list[tuple[str, str, str]]) -> str:
    """Filters as a readable phrase: 'for Mango Kulfi in the West region in May 2025'."""
    by_kind = {kind: phrase for kind, _, phrase in filters}
    if "year" in by_kind and ("month" in by_kind or "quarter" in by_kind):
        part = "month" if "month" in by_kind else "quarter"
        by_kind[part] = f"{by_kind[part]} {by_kind.pop('year')[3:]}"
    order = {k: i for i, k in enumerate(_SCOPE_ORDER)}
    return " ".join(by_kind[k] for k in sorted(by_kind, key=lambda k: order.get(k, 99)))


class RuleSqlWriter:
    name = "rules"

    def __init__(self, vocab: Vocabulary) -> None:
        self.vocab = vocab

    # ---------- entity recognition ----------

    def _products(self, p: _Parse) -> list[tuple[str, str]]:
        aliases = []
        for sku, name, _ in self.vocab.products:
            lname = name.lower().replace("&", "and")
            aliases += [(lname, sku, name), (sku.lower(), sku, name)]
            words = lname.split()
            if len(words) >= 3:
                aliases.append((" ".join(words[1:]), sku, name))
        found: dict[str, str] = {}
        for alias, sku, name in sorted(aliases, key=lambda a: -len(a[0])):
            if p.take(rf"\b{re.escape(alias)}s?\b"):
                found[sku] = name
        return list(found.items())

    def _parse_common(self, p: _Parse, date_col: str) -> None:
        # time
        years = sorted(set(re.findall(r"\b(20\d\d)\b", p.text)))
        compare = bool(re.search(r"\b(vs\.?|versus|compare|compared|against)\b", p.text)) and len(years) > 1
        p.compare_years = compare  # type: ignore[attr-defined]
        q = re.search(r"\bq([1-4])\b", p.text) or re.search(r"\b(first|second|third|fourth|1st|2nd|3rd|4th) quarter\b", p.text)
        month_hit = None
        for word, num in sorted(MONTHS.items(), key=lambda kv: -len(kv[0])):
            if re.search(rf"\b{word}\b", p.text) and not (word == "may" and re.search(r"\bmay (we|i|be)\b", p.text)):
                month_hit = (word, num)
                break
        if years and not compare:
            if len(years) == 1:
                p.filters.append(("year", f"EXTRACT(YEAR FROM {date_col}) = {years[0]}", f"in {years[0]}"))
            else:
                p.filters.append(("year", f"EXTRACT(YEAR FROM {date_col}) IN ({', '.join(years)})",
                                  f"in {' and '.join(years)}"))
        for y in years:
            p.take(rf"\b{y}\b")
        if q:
            n = int(q.group(1)) if q.group(1).isdigit() else ORDINAL[q.group(1)]
            p.filters.append(("quarter", f"EXTRACT(QUARTER FROM {date_col}) = {n}", f"in Q{n}"))
            p.take(re.escape(q.group(0)))
        elif month_hit:
            word, num = month_hit
            p.filters.append(("month", f"EXTRACT(MONTH FROM {date_col}) = {num}", f"in {word.title()}"))
            p.take(rf"\b{word}\b")
        elif p.take(r"\bsummer\b"):
            p.filters.append(("season", f"EXTRACT(MONTH FROM {date_col}) BETWEEN 4 AND 6", "in summer (April to June)"))

    def _parse_places(self, p: _Parse, region_alias: str = "r", dist_alias: str = "d") -> None:
        for did, name in self.vocab.distributors:
            if p.take(rf"\b{re.escape(name.lower())}\b|\b{did.lower()}\b"):
                p.filters.append(("distributor", f"{dist_alias}.distributor_id = {_q(did)}", f"for {name}"))
                p.joins.add("d")
        for rid, name in self.vocab.regions:
            if p.take(rf"\b{name.lower()}(?:ern)?(?: region)?\b(?!-)"):
                p.filters.append(("region", f"{region_alias}.region_id = {_q(rid)}", f"in the {name} region"))
                p.joins.update({"d", "r"})

    def _parse_plants(self, p: _Parse) -> None:
        for pid, name, city in self.vocab.plants:
            if p.take(rf"\b{re.escape(name.lower())}\b|\b{re.escape(city.lower())}(?: plant)?\b|\b{pid.lower()}\b"):
                p.filters.append(("plant", f"pl.plant_id = {_q(pid)}", f"at the {name}"))
                p.joins.add("pl")

    def _parse_products(self, p: _Parse, alias: str) -> None:
        prods = self._products(p)
        if prods:
            skus = ", ".join(_q(s) for s, _ in prods)
            names = " and ".join(n for _, n in prods)
            p.filters.append(("product", f"{alias}.sku IN ({skus})" if len(prods) > 1 else f"{alias}.sku = {skus}",
                              f"for {names}"))
        for word, cat in CATEGORIES.items():
            if p.take(rf"\b{word}s?\b") and not any(f[0] == "category" for f in p.filters):
                plural = cat if cat == "kulfi" else f"{cat}s"
                p.filters.append(("category", f"p.category = {_q(cat)}", f"for {plural}"))
                p.joins.add("p")

    def _parse_channel(self, p: _Parse, alias: str = "s") -> None:
        for pattern, value, label in CHANNELS:
            if p.take(rf"\b(?:{pattern})\b"):
                p.filters.append(("channel", f"{alias}.channel = {_q(value)}", f"through {label}"))

    @staticmethod
    def _dims(p: _Parse, allowed: dict[str, tuple[str, str, set[str]]]) -> list[tuple[str, str, str, bool]]:
        """Breakdowns asked for: (key, sql, alias, singular)."""
        found = []
        words = {
            "region": r"regions?|region-wise", "product": r"products?|skus?|items?|flavours? sold",
            "category": r"categor(?:y|ies)|formats?", "flavour": r"flavou?rs?", "channel": r"channels?",
            "distributor": r"distributors?", "plant": r"plants?|factor(?:y|ies)", "month": r"months?|monthly",
            "quarter": r"quarters?|quarterly", "year": r"years?|yearly|annual", "severity": r"severit(?:y|ies)",
            "tier": r"tiers?",
        }
        for key, (sql, alias, _joins) in allowed.items():
            m = re.search(rf"\b({words[key]})\b", p.text)
            if m and not (key in ("month", "year", "quarter") and re.search(rf"\bin (the )?(last|this|that) {m.group(1)}\b", p.text)):
                found.append((m.start(), key, sql, alias, not m.group(1).endswith(("s", "ies")) and m.group(1) not in ("monthly", "quarterly", "yearly", "annual")))
        if re.search(r"\btrend\b|\bover time\b", p.text) and not any(f[1] == "month" for f in found) and "month" in allowed:
            found.append((len(p.text), "month", *allowed["month"][:2], False))
        found.sort()
        return [(k, s, a, single) for _, k, s, a, single in found[:2]]

    # ---------- query families ----------

    def write(self, question: str) -> SqlPlan | None:
        text = " " + question.lower().replace("&", "and").replace("’", "'") + " "
        text = re.sub(r"\bproduct (categor|format)", r"\1", text)
        if UNSUPPORTED.search(text):
            return None
        counting = re.search(r"\b(how many|number of|count)\b", text)
        if counting and re.search(r"\bincidents?\b", text) or re.search(r"\bincidents? (by|per)\b", text):
            return self._incidents(text)
        if counting and re.search(r"\b(products|skus)\b", text) and not re.search(r"\bunits?\b", text):
            return self._count_products(text)
        if counting and re.search(r"\bdistributors\b", text):
            return self._count_distributors(text)
        if counting and re.search(r"\bpromotions?|promos?\b", text):
            return self._count_promotions(text)
        if re.search(r"\b(produce|produced|production|manufactured|batch|batches)\b", text):
            return self._production(text)
        if re.search(r"\b(revenue|sales|sold|sell|sells|selling|units?|volume|discounts?|margin|profit|turnover|earn(ed)?)\b", text):
            return self._sales(text)
        return None

    def _finish(self, metric_sql, metric_alias, label, from_sql, p: _Parse, dims, order_desc, limit,
                intent, extra_where=()) -> SqlPlan:
        where = [f[1] for f in p.filters] + list(extra_where)
        cols = [f"{s} AS {a}" for _, s, a, _ in dims] + [f"{metric_sql} AS {metric_alias}"]
        sql = f"SELECT {', '.join(cols)}\nFROM {from_sql}"
        if where:
            sql += "\nWHERE " + "\n  AND ".join(where)
        if dims:
            sql += "\nGROUP BY " + ", ".join(a for _, _, a, _ in dims)
            time_dim = dims[0][0] in ("month", "quarter", "year")
            if time_dim and limit is None:
                sql += "\nORDER BY " + ", ".join(a for _, _, a, _ in dims)
            else:
                sql += f"\nORDER BY {metric_alias} {'DESC' if order_desc else 'ASC'}" + "".join(f", {a}" for _, _, a, _ in dims)
            if limit:
                sql += f"\nLIMIT {limit}"
        scope = _scope(p.filters)
        by = " and ".join(k for k, *_ in dims)
        explanation = f"Adds up {label}" + (f" {scope}" if scope else "") + (f", broken down by {by}" if by else "")
        if limit:
            explanation += f", keeping the {'top' if order_desc else 'bottom'} {limit}"
        return SqlPlan(sql, explanation + ".", self.name, {
            "intent": intent, "label": label, "metric": metric_alias, "scope": scope,
            "dims": [k for k, *_ in dims], "desc": order_desc, "limit": limit})

    @staticmethod
    def _ranking(p: _Parse, dims) -> tuple[bool, int | None]:
        t = p.text
        desc = not re.search(r"\b(lowest|least|worst|smallest|bottom|fewest|weakest)\b", t)
        m = re.search(r"\b(?:top|bottom)\s+(\d+)\b", t)
        if m:
            return desc, int(m.group(1))
        if re.search(r"\b(top|bottom)\b", t):
            return desc, 5
        if dims and dims[0][3] and re.search(r"\b(which|what)\b", t) and re.search(
                r"\b(highest|lowest|most|least|best|worst|biggest|largest|smallest|top|fewest|max|min)\b|best[- ]selling", t):
            return desc, 1
        if dims and re.search(r"\b(best|worst)[- ]selling\b", t):
            return desc, 1 if dims[0][3] else 5
        return desc, None

    def _sales(self, text: str) -> SqlPlan:
        p = _Parse(text)
        self._parse_common(p, "s.sale_month")
        self._parse_places(p)
        self._parse_products(p, "s")
        self._parse_channel(p)
        if re.search(r"\b(margin|profit)\b", p.text):
            metric = "margin"
        elif re.search(r"\bdiscounts?\b", p.text):
            metric = "discount"
        elif re.search(r"\bgross\b", p.text):
            metric = "gross"
        elif re.search(r"\b(revenue|turnover|earn(ed)?|rupees|inr|value)\b", p.text):
            metric = "revenue"
        elif re.search(r"\b(units?|volume|sold|packs?|quantity|how many)\b", p.text):
            metric = "units"
        else:
            metric = "revenue"
        allowed = {
            "region": ("r.name", "region", {"d", "r"}), "product": ("p.name", "product", {"p"}),
            "category": ("p.category", "category", {"p"}), "flavour": ("p.flavour", "flavour", {"p"}),
            "channel": ("s.channel", "channel", set()), "distributor": ("d.name", "distributor", {"d"}),
            "month": ("s.sale_month", "month", set()), "quarter": ("TO_CHAR(s.sale_month, 'YYYY-\"Q\"Q')", "quarter", set()),
            "year": ("EXTRACT(YEAR FROM s.sale_month)::int", "year", set()),
        }
        dims = self._dims(p, allowed)
        if getattr(p, "compare_years", False) and not any(d[0] == "year" for d in dims):
            dims = (dims + [("year", allowed["year"][0], "year", False)])[:2]
        for key, *_ in dims:
            p.joins |= allowed[key][2]
        if metric == "margin" or any(f[0] in ("category",) for f in p.filters):
            p.joins.add("p")
        desc, limit = self._ranking(p, dims)
        if any(f[0] == "region" for f in p.filters):
            p.joins.update({"d", "r"})
        from_sql = "sales s"
        if "p" in p.joins:
            from_sql += "\nJOIN products p ON p.sku = s.sku"
        if "d" in p.joins:
            from_sql += "\nJOIN distributors d ON d.distributor_id = s.distributor_id"
        if "r" in p.joins:
            from_sql += "\nJOIN regions r ON r.region_id = d.region_id"
        msql, alias, label = SALES_METRICS[metric]
        return self._finish(msql, alias, label, from_sql, p, dims, desc, limit, f"sales.{metric}")

    def _production(self, text: str) -> SqlPlan:
        p = _Parse(text)
        self._parse_common(p, "b.produced_on")
        self._parse_plants(p)
        self._parse_products(p, "b")
        qc = p.take(r"\b(rejected|held|passed)\b")
        if qc:
            p.filters.append(("qc", f"b.qc_status = {_q(qc.group(1))}", f"with QC status {qc.group(1)}"))
        allowed = {"plant": ("pl.name", "plant", {"pl"}), "product": ("p.name", "product", {"p"}),
                   "category": ("p.category", "category", {"p"}), "month": ("DATE_TRUNC('month', b.produced_on)::date", "month", set()),
                   "year": ("EXTRACT(YEAR FROM b.produced_on)::int", "year", set())}
        dims = self._dims(p, allowed)
        for key, *_ in dims:
            p.joins |= allowed[key][2]
        desc, limit = self._ranking(p, dims)
        from_sql = "production_batches b"
        if "p" in p.joins:
            from_sql += "\nJOIN products p ON p.sku = b.sku"
        if "pl" in p.joins:
            from_sql += "\nJOIN plants pl ON pl.plant_id = b.plant_id"
        if re.search(r"\bbatch(es)?\b", p.text) and not re.search(r"\bunits\b", p.text):
            return self._finish("COUNT(*)", "batches", "production batches", from_sql, p, dims, desc, limit,
                                "production.batches")
        return self._finish("SUM(b.units)", "units_produced", "units produced", from_sql, p, dims, desc, limit,
                            "production.units")

    def _incidents(self, text: str) -> SqlPlan:
        p = _Parse(text)
        self._parse_common(p, "q.reported_on")
        self._parse_plants(p)
        self._parse_products(p, "b")
        sev = p.take(r"\b(high|medium|low)(?:[- ]severity)?\b")
        if sev:
            p.filters.append(("severity", f"q.severity = {_q(sev.group(1))}", f"with {sev.group(1)} severity"))
        for word, cat in [("temperature", "temperature_excursion"), ("foreign", "foreign_matter"),
                          ("packaging", "packaging"), ("allergen", "allergen"), ("texture", "texture"),
                          ("label", "labelling")]:
            if p.take(rf"\b{word}\w*\b"):
                p.filters.append(("category", f"q.category = {_q(cat)}", f"about {cat.replace('_', ' ')}"))
        st = p.take(r"\b(open|closed)\b")
        if st:
            p.filters.append(("status", f"q.status = {_q(st.group(1))}", f"still {st.group(1)}"))
        allowed = {"severity": ("q.severity", "severity", set()), "category": ("q.category", "category", set()),
                   "plant": ("pl.name", "plant", {"pl"}), "product": ("p.name", "product", {"p"}),
                   "year": ("EXTRACT(YEAR FROM q.reported_on)::int", "year", set())}
        dims = self._dims(p, allowed)
        for key, *_ in dims:
            p.joins |= allowed[key][2]
        desc, limit = self._ranking(p, dims)
        from_sql = "quality_incidents q\nJOIN production_batches b ON b.batch_id = q.batch_id"
        if "p" in p.joins:
            from_sql += "\nJOIN products p ON p.sku = b.sku"
        if "pl" in p.joins:
            from_sql += "\nJOIN plants pl ON pl.plant_id = b.plant_id"
        return self._finish("COUNT(*)", "incidents", "quality incidents", from_sql, p, dims, desc, limit,
                            "count.incidents")

    def _count_products(self, text: str) -> SqlPlan:
        p = _Parse(text)
        status = p.take(r"\b(active|discontinued)\b")
        extra = [f"p.status = {_q(status.group(1))}"] if status else []
        self._parse_products(p, "p")
        allowed = {"category": ("p.category", "category", set()), "flavour": ("p.flavour", "flavour", set())}
        dims = self._dims(p, allowed)
        label = f"{status.group(1)} products" if status else "products"
        return self._finish("COUNT(*)", "products", label, "products p", p, dims, True, None,
                            "count.products", extra)

    def _count_distributors(self, text: str) -> SqlPlan:
        p = _Parse(text)
        self._parse_places(p, region_alias="d")
        tier = p.take(r"\btier ([abc])\b")
        if tier:
            p.filters.append(("tier", f"d.cold_chain_tier = {_q(tier.group(1).upper())}", f"in tier {tier.group(1).upper()}"))
        allowed = {"region": ("r.name", "region", {"r"}), "tier": ("d.cold_chain_tier", "tier", set())}
        dims = self._dims(p, allowed)
        from_sql = "distributors d" + ("\nJOIN regions r ON r.region_id = d.region_id" if dims and dims[0][0] == "region" else "")
        return self._finish("COUNT(*)", "distributors", "distributors", from_sql, p, dims, True, None,
                            "count.distributors")

    def _count_promotions(self, text: str) -> SqlPlan:
        p = _Parse(text)
        self._parse_common(p, "pr.start_date")
        self._parse_products(p, "pr")
        self._parse_channel(p, "pr")
        allowed = {"channel": ("pr.channel", "channel", set()), "year": ("EXTRACT(YEAR FROM pr.start_date)::int", "year", set())}
        dims = self._dims(p, allowed)
        from_sql = "promotions pr" + ("\nJOIN products p ON p.sku = pr.sku" if "p" in p.joins else "")
        return self._finish("COUNT(*)", "promotions", "promotions", from_sql, p, dims, True, None,
                            "count.promotions")

    # ---------- answer text ----------

    def narrate(self, question: str, plan: SqlPlan, result: QueryResult) -> str:
        return narrate_result(plan, result)


def narrate_result(plan: SqlPlan, result: QueryResult) -> str:
    """A plain sentence for a result, without an LLM."""
    m = plan.meta
    label, scope = m.get("label", "the result"), m.get("scope", "")
    scope = f" {scope}" if scope else ""
    if not result.rows:
        return f"No matching rows{scope}."
    metric = result.columns[-1]
    if result.scalar is not None or len(result.columns) == 1:
        value = result.rows[0][-1]
        if value is None:
            return f"No matching rows{scope}."
        return f"{label[0].upper() + label[1:]}{scope}: {format_value(value, metric)}."
    dims = result.columns[:-1]

    def name(row) -> str:
        return " ".join(format_value(v, dims[i]) for i, v in enumerate(row[:-1])).replace("_", " ")

    if len(result.rows) == 1 and m.get("limit") == 1:
        word = "highest" if m.get("desc", True) else "lowest"
        top = name(result.rows[0])
        return f"{top[:1].upper() + top[1:]} had the {word} {label}{scope}: {format_value(result.rows[0][-1], metric)}."
    if dims and dims[0] in ("month", "quarter", "year") and len(dims) == 1 and len(result.rows) > 2:
        hi = max(result.rows, key=lambda r: r[-1] or 0)
        lo = min(result.rows, key=lambda r: r[-1] or 0)
        total = sum(r[-1] or 0 for r in result.rows)
        return (f"{label[0].upper() + label[1:]}{scope} by {dims[0]}: peak in {name(hi)} ({format_value(hi[-1], metric)}), "
                f"lowest in {name(lo)} ({format_value(lo[-1], metric)}). Total {format_value(total, metric)}.")
    shown = result.rows[:6]
    parts = ", ".join(f"{name(r)} {format_value(r[-1], metric)}" for r in shown)
    more = f", and {len(result.rows) - len(shown)} more in the table" if len(result.rows) > len(shown) else ""
    return f"{label[0].upper() + label[1:]}{scope} by {' and '.join(dims)}: {parts}{more}."


SQL_PROMPT = """You write one PostgreSQL SELECT query that answers a question about Frostfield Creamery,
an Indian ice cream company. Reply with JSON: {"sql": "...", "explanation": "<one plain-English sentence>"}.
Rules:
- One SELECT (or WITH ... SELECT) only. Never modify data.
- Use only the tables and columns below. Join through the foreign keys shown with ->.
- "Revenue" means sales.net_revenue_inr unless gross revenue is asked for. Money is in rupees (INR).
- sales.sale_month is the first day of each month; filter years with EXTRACT(YEAR FROM sale_month).
- Use short snake_case column aliases (e.g. region, net_revenue_inr) and ROUND money to 2 decimals.
- Return at most 50 rows. Order rankings by the measure, trends by time.
If the question cannot be answered from these tables, reply {"sql": "", "explanation": "<why>"}.

Data dictionary:
{schema}"""

NARRATE_PROMPT = """You answer a business question from the result of a SQL query. Use only the rows given.
Write one to three plain sentences with the key numbers. Write rupee amounts with ₹ and lakh/crore
(1 lakh = 100,000; 1 crore = 10,000,000). Do not mention SQL or tables. If the result is empty, say no data matched."""


class OpenAISqlWriter:
    name = "openai"

    def __init__(self, schema_text: str, model: str | None = None, client=None) -> None:
        self.schema_text = schema_text
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self._client = client

    def write(self, question: str) -> SqlPlan | None:
        resp = self._client.chat.completions.create(
            model=self.model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SQL_PROMPT.replace("{schema}", self.schema_text)},
                      {"role": "user", "content": question}],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        sql = str(data.get("sql") or "").strip()
        if not sql:
            return None
        return SqlPlan(sql, str(data.get("explanation") or ""), self.name, {"label": "the result"})

    def narrate(self, question: str, plan: SqlPlan, result: QueryResult) -> str:
        table = json.dumps({"columns": result.columns, "rows": result.rows[:40], "truncated": result.truncated},
                           default=str)
        resp = self._client.chat.completions.create(
            model=self.model, temperature=0,
            messages=[{"role": "system", "content": NARRATE_PROMPT},
                      {"role": "user", "content": f"Question: {question}\nResult: {table}"}],
        )
        return (resp.choices[0].message.content or "").strip()
