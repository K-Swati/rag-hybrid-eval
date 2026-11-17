"""Turns warehouse rows into the markdown knowledge base the document (RAG) path searches.

Postgres stays the single source of truth. Policies are stored as markdown in
the ``documents`` table and exported as they are. Descriptive records
(products, suppliers, plants, distributors, promotions, quality incidents) are
rendered as one markdown document per table, with one ``##`` section per row,
so the heading chunker gives one chunk per record. Sales and production
numbers are left out on purpose: those questions go to SQL.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from rag_eval.core.chunking import Document


@dataclass(frozen=True)
class KbDoc:
    doc_id: str
    title: str
    category: str  # policy | catalogue | records
    source: str  # where it came from, e.g. "documents table" or "products table"
    text: str

    def to_document(self) -> Document:
        return Document(self.doc_id, self.text)


def _inr(x) -> str:
    return f"₹{float(x):,.2f}".replace(".00", "")


def _label(s: str) -> str:
    return s.replace("_", " ")


def export_documents(t: dict[str, list[dict]]) -> list[KbDoc]:
    """``t`` maps table name to rows (dicts), as returned by ``Warehouse.fetch_all`` or ``seed.build_tables``."""
    regions = {r["region_id"]: r for r in t["regions"]}
    suppliers = {s["supplier_id"]: s for s in t["suppliers"]}
    products = {p["sku"]: p for p in t["products"]}
    plants = {p["plant_id"]: p for p in t["plants"]}
    batches = {b["batch_id"]: b for b in t["production_batches"]}
    ingredients = defaultdict(list)
    supplies = defaultdict(set)
    for i in t["product_ingredients"]:
        ingredients[i["sku"]].append(i)
        supplies[i["supplier_id"]].add(products[i["sku"]]["name"])

    docs = [KbDoc(d["doc_id"], d["title"], "policy", "documents table", str(d["body_md"]))
            for d in sorted(t["documents"], key=lambda d: d["doc_id"])]

    lines = ["# Product Catalogue", "",
             "Every product Frostfield Creamery makes or has made, with prices and ingredients.", ""]
    for p in sorted(t["products"], key=lambda p: (p["category"], p["name"])):
        status = "Active" if p["status"] == "active" else f"Discontinued on {p['discontinued_on']}"
        ing = ", ".join(f"{i['ingredient']} ({suppliers[i['supplier_id']]['name']})" for i in ingredients[p["sku"]])
        lines += [f"## {p['name']}",
                  f"SKU {p['sku']}. {p['description']}",
                  f"Format: {p['category']}, {p['pack_size_ml']} ml. Flavour: {p['flavour']}. "
                  f"MRP {_inr(p['mrp_inr'])}, unit cost {_inr(p['unit_cost_inr'])}. Launched {p['launch_date']}. "
                  f"Status: {status}.",
                  f"Main ingredients and suppliers: {ing}.", ""]
    docs.append(KbDoc("product-catalogue", "Product Catalogue", "catalogue", "products, product_ingredients tables",
                      "\n".join(lines)))

    lines = ["# Supplier Directory", "", "Who supplies our raw materials and packaging, and how they perform.", ""]
    for s in sorted(t["suppliers"], key=lambda s: s["supplier_id"]):
        cert = "holds an FSSAI licence" if s["fssai_certified"] else "does not currently hold a valid food-contact certificate"
        lines += [f"## {s['name']}",
                  f"Supplier {s['supplier_id']}, {s['category']} supplier based in {s['city']}. "
                  f"Rating {s['rating']} out of 5; {cert}; payment terms {s['payment_terms_days']} days.",
                  str(s["notes"]),
                  f"Used in: {', '.join(sorted(supplies[s['supplier_id']]))}.", ""]
    docs.append(KbDoc("supplier-directory", "Supplier Directory", "catalogue", "suppliers table", "\n".join(lines)))

    lines = ["# Plants and Regions", "", "Our manufacturing plants and the sales regions they serve.", ""]
    for p in sorted(t["plants"], key=lambda p: p["plant_id"]):
        r = regions[p["region_id"]]
        lines += [f"## {p['name']}",
                  f"Plant {p['plant_id']} in {p['city']}, opened {p['opened_on']}, capacity "
                  f"{int(p['capacity_litres_per_day']):,} litres a day. It supplies the {r['name']} region, "
                  f"whose sales head is {r['sales_head']} (head office {r['head_office_city']}).", ""]
    docs.append(KbDoc("plants-and-regions", "Plants and Regions", "catalogue", "plants, regions tables",
                      "\n".join(lines)))

    lines = ["# Distributor Directory", "", "The distributors we sell through.", ""]
    for d in sorted(t["distributors"], key=lambda d: d["distributor_id"]):
        lines += [f"## {d['name']}",
                  f"Distributor {d['distributor_id']} in {d['city']} ({regions[d['region_id']]['name']} region), "
                  f"onboarded {d['onboarded_on']}. Cold chain tier {d['cold_chain_tier']}; credit limit "
                  f"{_inr(d['credit_limit_inr'])}.", ""]
    docs.append(KbDoc("distributor-directory", "Distributor Directory", "catalogue", "distributors table",
                      "\n".join(lines)))

    lines = ["# Promotions", "", "Every trade and consumer promotion, with its mechanic and outcome.", ""]
    for p in sorted(t["promotions"], key=lambda p: p["start_date"]):
        where = regions[p["region_id"]]["name"] + " region" if p["region_id"] else "nationwide"
        lines += [f"## {p['name']}",
                  f"Promotion {p['promo_id']} on {products[p['sku']]['name']}, {_label(p['channel'])} channel, {where}, "
                  f"from {p['start_date']} to {p['end_date']}, {float(p['discount_pct']):g}% off.",
                  str(p["notes"]), ""]
    docs.append(KbDoc("promotions", "Promotions", "records", "promotions table", "\n".join(lines)))

    lines = ["# Quality Incidents", "", "Every quality and food safety incident, its root cause and the fix.", ""]
    for q in sorted(t["quality_incidents"], key=lambda q: q["reported_on"]):
        b = batches[q["batch_id"]]
        lines += [f"## {q['incident_id']}: {products[b['sku']]['name']}",
                  f"Reported {q['reported_on']}, severity {q['severity']}, {_label(q['category'])}, status {q['status']}. "
                  f"Batch {q['batch_id']} made at the {plants[b['plant_id']]['name']}; batch QC status {b['qc_status']}.",
                  f"What happened: {q['description']}",
                  f"Root cause: {q['root_cause']}",
                  f"Corrective action: {q['corrective_action']}", ""]
    docs.append(KbDoc("quality-incidents", "Quality Incidents", "records", "quality_incidents table",
                      "\n".join(lines)))
    return docs
