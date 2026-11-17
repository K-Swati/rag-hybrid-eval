"""Policies and SOPs of Frostfield Creamery, seeded into the ``documents`` table.

Fictional company, written to be realistic for an Indian ice cream maker. The
gold questions in ``data/eval/questions.json`` quote phrases from these texts,
so edit both together.
"""

from __future__ import annotations

# (doc_id, title, category, owner, updated_on, body)
POLICIES: list[tuple[str, str, str, str, str, str]] = [
    ("cold-chain-sop", "Cold Chain SOP", "operations", "Head of Supply Chain", "2025-02-10", """
# Cold Chain SOP

## Storage temperatures
All finished ice cream is stored at -18 °C or colder in plant cold rooms and distributor freezers. Reefer trucks must hold -20 °C or colder for the whole trip. Kulfi and sorbet follow the same limits as other formats.

## Temperature monitoring
Every cold room, reefer truck and tier A distributor freezer carries a data logger that records the temperature every 5 minutes. Logs are uploaded to the supply chain dashboard daily and kept for 2 years.

## Temperature excursions
If product warms above -12 °C for more than 30 minutes, the affected stock is quarantined and may not be sold until Quality Assurance releases it. Stock that has visibly melted and refrozen is destroyed, never reworked.

## Loading and transit
Trucks are pre-cooled for at least 45 minutes before loading. Doors stay open for no more than 10 minutes per drop. Routes longer than 8 hours need a reefer with a backup compressor.

## Distributor freezer checks
Area sales officers check distributor freezers weekly and record the reading in the field app. A freezer that fails two checks in a row is replaced at the distributor's cost.
"""),
    ("product-recall-procedure", "Product Recall Procedure", "quality", "Head of Quality Assurance", "2024-11-05", """
# Product Recall Procedure

## When to recall
A recall is started for any confirmed food safety risk: an undeclared allergen, foreign matter, microbial contamination or a temperature failure that reached consumers. Quality defects with no safety risk, such as texture, are handled as market withdrawals.

## Recall committee
The recall committee is convened within 2 hours of a confirmed risk. It is chaired by the Head of Quality Assurance and includes the plant head, the regional sales head and legal counsel.

## Regulator notification
The company notifies FSSAI within 24 hours of deciding to recall, and shares the batch numbers, quantities and distribution list.

## Traceability
Every pack carries its batch_id. Using production and sales records, the team must trace 100% of a batch to distributors within 4 hours.

## Mock recalls
Each plant runs a mock recall twice a year. A mock recall fails if traceability takes longer than 4 hours.
"""),
    ("returns-and-damages-policy", "Returns and Damages Policy", "sales", "Head of Sales Operations", "2025-01-15", """
# Returns and Damages Policy

## Claim window
Distributors may claim for damaged or short-shipped stock within 48 hours of delivery. Claims need photos and the delivery challan number.

## What is accepted
Crushed packs, leaking tubs and short shipments are accepted. Stock that melted after delivery because of the distributor's freezer is not accepted.

## Credit notes
Approved claims are settled by credit note within 7 working days. Claims above ₹50,000 need approval from the regional sales manager.

## Expired stock
Ice cream has a shelf life of 12 months. Expired stock is not taken back unless the distributor received it with less than 3 months of shelf life left.
"""),
    ("trade-credit-policy", "Trade Credit Policy", "finance", "Chief Financial Officer", "2024-08-01", """
# Trade Credit Policy

## Credit periods
General trade distributors get 21 days of credit. Modern trade and quick commerce accounts get 45 days. HoReCa accounts pay within 30 days.

## Credit limits
Each distributor has a credit limit set at onboarding, based on 1.5 times their expected peak-month purchases. Limits are reviewed every quarter.

## Overdue accounts
A distributor is put on credit hold when any invoice is overdue by more than 15 days. Orders resume once the overdue amount is cleared.

## Interest
Overdue amounts attract interest at 18% a year, charged monthly.
"""),
    ("pricing-and-discount-policy", "Pricing and Discount Policy", "sales", "Head of Revenue Management", "2025-02-01", """
# Pricing and Discount Policy

## Distributor margin
Distributors buy at 72% of MRP. Retailer margin is built into the MRP.

## Trade discounts
Sales managers may give a trade discount of up to 12% without approval. Anything above 12% needs the regional sales head.

## Promotions
Promotions with a discount above 20% need CFO sign-off. Every promotion must have an end date and is reviewed within 2 weeks of ending.

## Price revisions
MRP is revised at most once a year, in February, before the summer season. Distributors get 30 days' notice of a price change.
"""),
    ("new-product-launch-playbook", "New Product Launch Playbook", "marketing", "Head of Marketing", "2024-09-20", """
# New Product Launch Playbook

## Stage gates
Every new product passes four gates: concept, recipe, pilot and national launch. Brand, R&D, supply chain and finance sign each gate.

## Consumer testing
A recipe moves to pilot only if at least 85% of the sensory panel rate it 4 out of 5 or higher.

## Shelf-life testing
New recipes are stored at -18 °C and tested monthly for 9 months before launch, checking ice crystals, flavour and colour.

## City pilot
Pilots run in 2 cities for 8 weeks. A pilot passes if repeat purchase is above 25% in modern trade.

## Launch timing
National launches are timed for February or March, ahead of the April to June peak season.
"""),
    ("food-safety-and-allergens", "Food Safety and Allergen Control", "quality", "Head of Quality Assurance", "2025-03-12", """
# Food Safety and Allergen Control

## Allergens we handle
Our plants handle milk, tree nuts (pistachio, almond, cashew), gluten (cone wafers) and soy (chocolate coatings). Every pack lists the allergens it contains and a may-contain warning for nuts.

## Line changeover
After running any nut product, a line gets a full CIP (clean-in-place) wash and an allergen swab test before a nut-free product such as sorbet is made on it. The swab result is recorded against the batch.

## Vegan and sugar-free claims
Sorbets are labelled vegan only when made on a dedicated or fully cleaned line. Sugar-free products use sucralose and maltitol and carry the statutory laxative-effect warning.

## Metal detection
Every pack passes a metal detector at the end of the line. Detector checks with test pieces run every hour.
"""),
    ("distributor-onboarding", "Distributor Onboarding", "sales", "Head of Sales Operations", "2024-06-18", """
# Distributor Onboarding

## Requirements
A new distributor needs at least 2 deep freezers of 400 litres each, a GST registration and an FSSAI licence. Tier A distributors also need a generator backup and at least one reefer vehicle.

## Security deposit
New distributors pay a refundable security deposit of ₹2,00,000 before the first order.

## Timeline
Onboarding takes 3 weeks: one week for documents, one for a freezer audit and one for training on the field app.

## Cold chain tiers
Distributors are graded A, B or C on their cold chain. Tier C distributors may not stock tubs, because tubs are the most sensitive to heat shock.
"""),
]
