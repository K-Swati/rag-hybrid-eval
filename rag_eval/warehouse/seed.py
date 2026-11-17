"""Deterministic demo data for Frostfield Creamery, a fictional Indian ice cream maker.

``build_tables()`` returns every table as a list of row dicts. ``to_sql()``
renders them as ``db/init/02_seed.sql``. The same seed always gives the same
rows, so the SQL file, the knowledge base and the benchmark stay in step
(``tests/test_warehouse.py`` checks the committed file is up to date).

The numbers follow simple, explainable rules: summer peak (April to June),
12% growth in 2025, quick commerce growing fastest, promotions lifting volume,
tier C distributors not stocking tubs (see the onboarding policy), and products
selling only between their launch and discontinue dates.
"""

from __future__ import annotations

import random
from datetime import date

from rag_eval.warehouse.knowledge import POLICIES

SEED = 2024
MONTHS = [date(y, m, 1) for y in (2024, 2025) for m in range(1, 13)]
SEASON = [0.55, 0.65, 0.95, 1.35, 1.60, 1.45, 1.00, 0.90, 0.95, 1.05, 0.80, 0.60]  # Jan..Dec
DISTRIBUTOR_PRICE = 0.72  # share of MRP the distributor pays (pricing policy)
BASE_TRADE_DISCOUNT = 0.02

REGIONS = [
    {"region_id": "NORTH", "name": "North", "head_office_city": "New Delhi", "sales_head": "Arjun Mehta"},
    {"region_id": "WEST", "name": "West", "head_office_city": "Mumbai", "sales_head": "Priya Kulkarni"},
    {"region_id": "SOUTH", "name": "South", "head_office_city": "Bengaluru", "sales_head": "Karthik Raman"},
    {"region_id": "EAST", "name": "East", "head_office_city": "Kolkata", "sales_head": "Ananya Bose"},
]
REGION_FACTOR = {"NORTH": 1.20, "WEST": 1.10, "SOUTH": 1.00, "EAST": 0.70}

PLANTS = [
    {"plant_id": "P01", "name": "Sonipat Plant", "city": "Sonipat", "region_id": "NORTH",
     "capacity_litres_per_day": 60000, "opened_on": "2012-04-01"},
    {"plant_id": "P02", "name": "Pune Plant", "city": "Pune", "region_id": "WEST",
     "capacity_litres_per_day": 45000, "opened_on": "2015-09-15"},
    {"plant_id": "P03", "name": "Hosur Plant", "city": "Hosur", "region_id": "SOUTH",
     "capacity_litres_per_day": 40000, "opened_on": "2018-01-10"},
    {"plant_id": "P04", "name": "Kalyani Plant", "city": "Kalyani", "region_id": "EAST",
     "capacity_litres_per_day": 25000, "opened_on": "2021-11-01"},
]
PLANT_OF_REGION = {p["region_id"]: p["plant_id"] for p in PLANTS}

SUPPLIERS = [
    {"supplier_id": "S01", "name": "Anand Valley Dairy Cooperative", "category": "dairy", "city": "Anand", "rating": 4.6,
     "fssai_certified": True, "payment_terms_days": 15,
     "notes": "Main supplier of milk and fresh cream for all four plants. Supplies 70% of our dairy under a 3-year contract that runs to March 2027. Very reliable; never missed a delivery in 2024."},
    {"supplier_id": "S02", "name": "Green Meadows Milk Products", "category": "dairy", "city": "Karnal", "rating": 3.9,
     "fssai_certified": True, "payment_terms_days": 30,
     "notes": "Backup dairy supplier, mainly skimmed milk powder. Fat content ran below specification twice in summer 2025, so every lot is now tested on arrival."},
    {"supplier_id": "S03", "name": "Sahyadri Sugars", "category": "sugar", "city": "Kolhapur", "rating": 4.2,
     "fssai_certified": True, "payment_terms_days": 30,
     "notes": "Sole sugar supplier. Prices are fixed every six months. Also supplies the sucralose and maltitol blend for the sugar-free range."},
    {"supplier_id": "S04", "name": "Ratnagiri Fruit Pulp Works", "category": "fruit", "city": "Ratnagiri", "rating": 4.4,
     "fssai_certified": True, "payment_terms_days": 45,
     "notes": "Alphonso mango pulp, strawberry and raspberry purée. Mango pulp is bought once a year in May and stored frozen. Raspberry purée costs rose 40% in 2025, one reason the Raspberry Sorbet was discontinued."},
    {"supplier_id": "S05", "name": "Kashmir Saffron and Nut Traders", "category": "flavour", "city": "Srinagar", "rating": 3.6,
     "fssai_certified": True, "payment_terms_days": 30,
     "notes": "Saffron, pistachio and almonds. Delivered pistachio with shell fragments in August 2025 (incident QI-2025-07); now on a corrective action plan with an extra sieving step."},
    {"supplier_id": "S06", "name": "CocoaCraft Ingredients", "category": "cocoa", "city": "Bengaluru", "rating": 4.1,
     "fssai_certified": True, "payment_terms_days": 30,
     "notes": "Cocoa powder, chocolate chips and the soy-lecithin chocolate coating for bars and cones."},
    {"supplier_id": "S07", "name": "Polypack Containers", "category": "packaging", "city": "Daman", "rating": 3.4,
     "fssai_certified": False, "payment_terms_days": 60,
     "notes": "Cups, tubs and lids. A batch of 500 ml tub lids did not seal properly in April 2025 (incident QI-2025-03). Food-contact certificate renewal is pending."},
    {"supplier_id": "S08", "name": "ConeCraft Wafers", "category": "packaging", "city": "Hosur", "rating": 4.3,
     "fssai_certified": True, "payment_terms_days": 30,
     "notes": "Wafer cones and cone sleeves, delivered to all plants from its Hosur factory. Cones contain gluten."},
]

# sku, name, category, flavour, ml, mrp, cost, launch, discontinued_on, monthly base units per distributor-channel, description
_PRODUCTS = [
    ("FF-CUP-VAN-100", "Classic Vanilla Cup", "cup", "vanilla", 100, 30, 13.0, "2019-03-01", None, 900,
     "Our best-known product: a 100 ml cup of vanilla ice cream made with fresh cream and real vanilla."),
    ("FF-CUP-CHO-100", "Double Chocolate Cup", "cup", "chocolate", 100, 35, 15.0, "2019-03-01", None, 780,
     "Chocolate ice cream with chocolate chips, in a 100 ml cup."),
    ("FF-CUP-BSC-100", "Butterscotch Crunch Cup", "cup", "butterscotch", 100, 35, 14.5, "2020-02-01", None, 650,
     "Butterscotch ice cream with caramelised cashew crunch, in a 100 ml cup. Contains tree nuts."),
    ("FF-CON-CHO-120", "Choco Fudge Cone", "cone", "chocolate", 120, 50, 22.0, "2019-04-01", None, 820,
     "Wafer cone filled with chocolate ice cream and a fudge core, topped with chocolate coating."),
    ("FF-CON-STR-120", "Strawberry Swirl Cone", "cone", "strawberry", 120, 50, 21.0, "2021-03-01", None, 540,
     "Wafer cone with vanilla and strawberry swirl ice cream."),
    ("FF-CON-PST-120", "Pista Royale Cone", "cone", "pistachio", 120, 60, 27.0, "2025-03-01", None, 480,
     "Premium pistachio cone with roasted pistachio pieces. Launched in March 2025. Contains tree nuts."),
    ("FF-STK-MNG-70", "Mango Duet Stick", "stick", "mango", 70, 25, 10.0, "2020-04-01", None, 760,
     "Alphonso mango ice cream on a stick with a mango sorbet coating."),
    ("FF-STK-ORG-60", "Orange Bar", "stick", "orange", 60, 10, 3.8, "2018-04-01", None, 1300,
     "Low-price orange ice lolly for general trade. Our highest-volume SKU."),
    ("FF-STK-CHO-70", "Choco Bar", "stick", "chocolate", 70, 30, 12.5, "2018-04-01", None, 980,
     "Vanilla ice cream on a stick with a crackling chocolate coating."),
    ("FF-TUB-VAN-500", "Classic Vanilla Tub", "tub", "vanilla", 500, 140, 62.0, "2019-03-01", None, 320,
     "Family pack of Classic Vanilla, 500 ml."),
    ("FF-TUB-BSC-750", "Butterscotch Family Tub", "tub", "butterscotch", 750, 220, 98.0, "2020-10-01", None, 210,
     "750 ml butterscotch tub for families and parties. Contains tree nuts."),
    ("FF-TUB-CCH-500", "Cookies and Cream Tub", "tub", "cookies and cream", 500, 180, 81.0, "2022-02-01", None, 240,
     "Vanilla ice cream with chocolate cookie pieces, 500 ml. Contains gluten."),
    ("FF-TUB-SFV-500", "Sugar-Free Vanilla Tub", "tub", "vanilla", 500, 200, 95.0, "2025-01-01", None, 110,
     "Vanilla ice cream sweetened with sucralose and maltitol, launched in January 2025 for diabetic and health-conscious buyers."),
    ("FF-KUL-MLI-60", "Malai Kulfi", "kulfi", "malai", 60, 30, 12.0, "2019-04-01", None, 700,
     "Traditional slow-cooked milk kulfi on a stick."),
    ("FF-KUL-KES-60", "Kesar Pista Kulfi", "kulfi", "saffron pistachio", 60, 40, 17.5, "2019-04-01", None, 560,
     "Kulfi flavoured with Kashmiri saffron and pistachio. Contains tree nuts."),
    ("FF-KUL-MNG-60", "Mango Kulfi", "kulfi", "mango", 60, 35, 14.5, "2021-04-01", None, 600,
     "Kulfi made with Alphonso mango pulp."),
    ("FF-SOR-LMN-100", "Lemon Mint Sorbet", "sorbet", "lemon mint", 100, 45, 17.0, "2024-10-01", None, 260,
     "Vegan, dairy-free lemon and mint sorbet, launched in October 2024."),
    ("FF-SOR-RSP-100", "Raspberry Sorbet", "sorbet", "raspberry", 100, 50, 20.0, "2023-03-01", "2025-06-30", 200,
     "Vegan raspberry sorbet. Discontinued at the end of June 2025 after raspberry purée costs rose and sales stayed low."),
]
PRODUCTS = [
    {"sku": s, "name": n, "category": c, "flavour": f, "pack_size_ml": ml, "mrp_inr": mrp, "unit_cost_inr": cost,
     "launch_date": launch, "status": "discontinued" if disc else "active", "discontinued_on": disc,
     "description": desc}
    for s, n, c, f, ml, mrp, cost, launch, disc, _, desc in _PRODUCTS
]
BASE_UNITS = {p[0]: p[9] for p in _PRODUCTS}

_INGREDIENTS = {
    "cup": [("milk", "S01", 55), ("cream", "S01", 20), ("sugar", "S03", 14), ("cup and lid", "S07", 6)],
    "cone": [("milk", "S01", 60), ("cream", "S01", 22), ("sugar", "S03", 16), ("wafer cone", "S08", 12)],
    "stick": [("milk", "S01", 38), ("sugar", "S03", 11), ("stick and wrapper", "S07", 3)],
    "tub": [("milk", "S01", 270), ("cream", "S01", 110), ("sugar", "S03", 70), ("tub and lid", "S07", 28)],
    "kulfi": [("milk", "S01", 48), ("milk powder", "S02", 6), ("sugar", "S03", 9)],
    "sorbet": [("sugar", "S03", 22), ("stabiliser", "S03", 1)],
}
_EXTRA = {
    "chocolate": [("cocoa", "S06", 8)], "butterscotch": [("cashew crunch", "S05", 6)],
    "strawberry": [("strawberry puree", "S04", 12)], "pistachio": [("pistachio", "S05", 9)],
    "mango": [("alphonso mango pulp", "S04", 18)], "orange": [("orange flavour", "S06", 1)],
    "cookies and cream": [("cookie pieces", "S06", 40)], "saffron pistachio": [("saffron", "S05", 0.1), ("pistachio", "S05", 5)],
    "lemon mint": [("lemon juice", "S04", 15)], "raspberry": [("raspberry puree", "S04", 25)],
    "malai": [], "vanilla": [("vanilla extract", "S06", 1)],
}

DISTRIBUTORS = [
    {"distributor_id": "D01", "name": "Capital Cold Distributors", "region_id": "NORTH", "city": "New Delhi", "cold_chain_tier": "A", "credit_limit_inr": 4500000, "onboarded_on": "2016-03-01"},
    {"distributor_id": "D02", "name": "Punjab Frozen Foods", "region_id": "NORTH", "city": "Ludhiana", "cold_chain_tier": "B", "credit_limit_inr": 2500000, "onboarded_on": "2018-05-10"},
    {"distributor_id": "D03", "name": "Lucknow Ice Traders", "region_id": "NORTH", "city": "Lucknow", "cold_chain_tier": "B", "credit_limit_inr": 2000000, "onboarded_on": "2020-02-15"},
    {"distributor_id": "D04", "name": "Bombay Freeze Logistics", "region_id": "WEST", "city": "Mumbai", "cold_chain_tier": "A", "credit_limit_inr": 5000000, "onboarded_on": "2015-11-01"},
    {"distributor_id": "D05", "name": "Ahmedabad Chill Agencies", "region_id": "WEST", "city": "Ahmedabad", "cold_chain_tier": "A", "credit_limit_inr": 3000000, "onboarded_on": "2017-04-20"},
    {"distributor_id": "D06", "name": "Nagpur Cool Mart", "region_id": "WEST", "city": "Nagpur", "cold_chain_tier": "C", "credit_limit_inr": 900000, "onboarded_on": "2023-03-01"},
    {"distributor_id": "D07", "name": "Namma Frozen Distributors", "region_id": "SOUTH", "city": "Bengaluru", "cold_chain_tier": "A", "credit_limit_inr": 4000000, "onboarded_on": "2018-01-15"},
    {"distributor_id": "D08", "name": "Chennai Polar Supplies", "region_id": "SOUTH", "city": "Chennai", "cold_chain_tier": "B", "credit_limit_inr": 2800000, "onboarded_on": "2018-06-01"},
    {"distributor_id": "D09", "name": "Kochi Coastal Cold Chain", "region_id": "SOUTH", "city": "Kochi", "cold_chain_tier": "C", "credit_limit_inr": 1100000, "onboarded_on": "2022-08-01"},
    {"distributor_id": "D10", "name": "Bengal Frost Agencies", "region_id": "EAST", "city": "Kolkata", "cold_chain_tier": "A", "credit_limit_inr": 2600000, "onboarded_on": "2021-12-01"},
    {"distributor_id": "D11", "name": "Bhubaneswar Ice Depot", "region_id": "EAST", "city": "Bhubaneswar", "cold_chain_tier": "B", "credit_limit_inr": 1200000, "onboarded_on": "2022-04-01"},
    {"distributor_id": "D12", "name": "Guwahati Northeast Frozen", "region_id": "EAST", "city": "Guwahati", "cold_chain_tier": "C", "credit_limit_inr": 800000, "onboarded_on": "2023-10-01"},
]
# The two channels each distributor serves, with the share of its volume.
CHANNELS = {
    "D01": [("modern_trade", 0.55), ("quick_commerce", 0.45)], "D02": [("general_trade", 0.8), ("horeca", 0.2)],
    "D03": [("general_trade", 0.85), ("horeca", 0.15)], "D04": [("modern_trade", 0.5), ("quick_commerce", 0.5)],
    "D05": [("modern_trade", 0.6), ("general_trade", 0.4)], "D06": [("general_trade", 0.9), ("horeca", 0.1)],
    "D07": [("quick_commerce", 0.55), ("modern_trade", 0.45)], "D08": [("general_trade", 0.7), ("horeca", 0.3)],
    "D09": [("general_trade", 0.85), ("horeca", 0.15)], "D10": [("modern_trade", 0.5), ("general_trade", 0.5)],
    "D11": [("general_trade", 0.9), ("horeca", 0.1)], "D12": [("general_trade", 1.0), ("horeca", 0.0)],
}
CHANNEL_GROWTH_2025 = {"modern_trade": 1.08, "general_trade": 1.05, "quick_commerce": 1.60, "horeca": 1.10}

PROMOTIONS = [
    {"promo_id": "PR-2024-01", "name": "Summer Cone Fest 2024", "sku": "FF-CON-CHO-120", "channel": "modern_trade", "region_id": None,
     "start_date": "2024-04-01", "end_date": "2024-05-31", "discount_pct": 15.0,
     "notes": "Buy 2 get 1 free on Choco Fudge Cones in supermarkets nationwide. Lifted cone volume by about 30% in April and May."},
    {"promo_id": "PR-2024-02", "name": "Mango Mania 2024", "sku": "FF-KUL-MNG-60", "channel": "general_trade", "region_id": "WEST",
     "start_date": "2024-05-01", "end_date": "2024-06-30", "discount_pct": 10.0,
     "notes": "Mango Kulfi at a 10% trade discount in western corner shops, timed with the Alphonso season. Free branded freezer stickers for shops that stocked 2 cartons."},
    {"promo_id": "PR-2024-03", "name": "Diwali Family Tub Offer", "sku": "FF-TUB-BSC-750", "channel": "modern_trade", "region_id": None,
     "start_date": "2024-10-15", "end_date": "2024-11-10", "discount_pct": 18.0,
     "notes": "Festive price-off on the Butterscotch Family Tub for Diwali gatherings, supported by in-store sampling."},
    {"promo_id": "PR-2024-04", "name": "Sorbet Launch Sampling", "sku": "FF-SOR-LMN-100", "channel": "modern_trade", "region_id": "SOUTH",
     "start_date": "2024-10-01", "end_date": "2024-12-31", "discount_pct": 12.0,
     "notes": "Launch support for Lemon Mint Sorbet in Bengaluru and Chennai supermarkets: 12% off plus 40,000 free sample spoons."},
    {"promo_id": "PR-2025-01", "name": "Quick Commerce Midnight Deals", "sku": "FF-TUB-VAN-500", "channel": "quick_commerce", "region_id": None,
     "start_date": "2025-03-01", "end_date": "2025-06-30", "discount_pct": 20.0,
     "notes": "20% off Classic Vanilla Tubs ordered between 10 pm and 2 am on delivery apps. The largest promotion of 2025; it needed CFO sign-off."},
    {"promo_id": "PR-2025-02", "name": "Summer Cone Fest 2025", "sku": "FF-CON-PST-120", "channel": "modern_trade", "region_id": None,
     "start_date": "2025-04-01", "end_date": "2025-05-31", "discount_pct": 15.0,
     "notes": "Buy 2 get 1 free, this year on the new Pista Royale Cone to drive trial after its March 2025 launch."},
    {"promo_id": "PR-2025-03", "name": "Kulfi Nights North", "sku": "FF-KUL-KES-60", "channel": "general_trade", "region_id": "NORTH",
     "start_date": "2025-05-01", "end_date": "2025-06-30", "discount_pct": 8.0,
     "notes": "Extra margin for northern corner shops that kept Kesar Pista Kulfi in the front freezer."},
    {"promo_id": "PR-2025-04", "name": "Sugar-Free Awareness Month", "sku": "FF-TUB-SFV-500", "channel": "modern_trade", "region_id": None,
     "start_date": "2025-11-01", "end_date": "2025-11-30", "discount_pct": 10.0,
     "notes": "Tie-up with diabetes awareness month, with 10% off the Sugar-Free Vanilla Tub and pharmacy-aisle displays."},
    {"promo_id": "PR-2025-05", "name": "HoReCa Dessert Partner", "sku": "FF-CUP-CHO-100", "channel": "horeca", "region_id": "SOUTH",
     "start_date": "2025-07-01", "end_date": "2025-09-30", "discount_pct": 12.0,
     "notes": "Volume discount for southern restaurants that serve Double Chocolate Cups as a dessert option."},
]
PROMO_UPLIFT = 1.30

# Hand-written incidents: (id, sku, plant, month, day, severity, category, status, qc_status, description, root cause, action)
_INCIDENTS = [
    ("QI-2024-01", "FF-KUL-KES-60", "P04", date(2024, 5, 1), 14, "medium", "temperature_excursion", "closed", "rejected",
     "Ice crystals and shrinkage found in Kesar Pista Kulfi delivered to Bengal Frost Agencies in Kolkata.",
     "A reefer truck compressor failed on the route from Kalyani; the data logger showed -9 °C for 3 hours.",
     "The batch was destroyed. Reefers on routes longer than 8 hours now carry a backup compressor."),
    ("QI-2024-02", "FF-CON-CHO-120", "P01", date(2024, 4, 1), 22, "low", "texture", "closed", "passed",
     "Soggy wafer cones reported by supermarkets in Delhi.",
     "Cones were filled before the chocolate barrier coating had set, so moisture reached the wafer.",
     "Coating cure time on the Sonipat cone line raised from 40 to 60 seconds."),
    ("QI-2024-03", "FF-STK-ORG-60", "P03", date(2024, 6, 1), 5, "high", "foreign_matter", "closed", "rejected",
     "A consumer in Chennai found a piece of blue plastic in an Orange Bar.",
     "A worn scraper blade on the Hosur stick moulding line was shedding plastic.",
     "Batch withdrawn from the market. Scraper blades are now metal-detectable and inspected every shift."),
    ("QI-2024-04", "FF-SOR-RSP-100", "P02", date(2024, 9, 1), 11, "low", "labelling", "closed", "passed",
     "Raspberry Sorbet packs printed with the wrong best-before date format.",
     "The date coder template was not updated after a software upgrade.",
     "Template change control added; packs were over-stickered before dispatch."),
    ("QI-2024-05", "FF-TUB-VAN-500", "P01", date(2024, 12, 1), 3, "medium", "packaging", "closed", "held",
     "Dented Classic Vanilla Tubs arriving at Capital Cold Distributors.",
     "Cartons were stacked 8 high in the cold room instead of the maximum of 5.",
     "Stacking limits painted on cold room floors and added to the warehouse checklist."),
    ("QI-2025-01", "FF-SOR-LMN-100", "P03", date(2025, 2, 1), 18, "high", "allergen", "closed", "rejected",
     "Lemon Mint Sorbet, sold as vegan and nut-free, tested positive for pistachio protein.",
     "The Hosur line ran Kesar Pista Kulfi before the sorbet and the allergen swab after cleaning was skipped.",
     "Voluntary recall of the batch; FSSAI was notified within 24 hours. Swab results must now be recorded before the line is released."),
    ("QI-2025-02", "FF-CUP-VAN-100", "P02", date(2025, 3, 1), 9, "low", "texture", "closed", "passed",
     "Classic Vanilla Cups from Pune tasted icy and less creamy.",
     "Skimmed milk powder from Green Meadows Milk Products was below fat specification.",
     "Incoming milk powder is now tested for fat content on every lot."),
    ("QI-2025-03", "FF-TUB-CCH-500", "P02", date(2025, 4, 1), 16, "medium", "packaging", "closed", "held",
     "Lids on Cookies and Cream Tubs popping off, leading to freezer burn.",
     "A batch of 500 ml tub lids from Polypack Containers was 0.3 mm too wide.",
     "Lids are sampled on arrival with a go/no-go gauge. Polypack issued a credit for the faulty lot."),
    ("QI-2025-04", "FF-STK-MNG-70", "P01", date(2025, 5, 1), 27, "medium", "temperature_excursion", "closed", "held",
     "Mango Duet Sticks melted and refroze at Lucknow Ice Traders during a power cut in a heatwave.",
     "The distributor's cold room had no generator backup and lost power for 6 hours.",
     "Stock destroyed at the distributor's cost. Lucknow Ice Traders must install a generator to stay tier B."),
    ("QI-2025-05", "FF-CON-STR-120", "P03", date(2025, 6, 1), 12, "low", "texture", "closed", "passed",
     "Strawberry swirl not visible in some Strawberry Swirl Cones.",
     "The ripple pump on the Hosur cone line was running at half speed.",
     "Pump speed is now logged at the start of every run."),
    ("QI-2025-06", "FF-KUL-MLI-60", "P01", date(2025, 7, 1), 8, "low", "labelling", "closed", "passed",
     "Malai Kulfi wrappers missing the Hindi product name.",
     "The printer used an old artwork file.",
     "Artwork files are now checked against the approved list before every print run."),
    ("QI-2025-07", "FF-CON-PST-120", "P01", date(2025, 8, 1), 21, "high", "foreign_matter", "closed", "rejected",
     "Hard shell fragments found in Pista Royale Cones by two consumers in Delhi.",
     "Pistachio from Kashmir Saffron and Nut Traders contained shell pieces.",
     "Batch withdrawn. The supplier added a sieving step and is on a corrective action plan; we X-ray every nut lot."),
    ("QI-2025-08", "FF-TUB-SFV-500", "P02", date(2025, 11, 1), 6, "medium", "labelling", "open", "held",
     "Sugar-Free Vanilla Tub packs missing the statutory laxative-effect warning for maltitol.",
     "A new pack design went to print without regulatory review.",
     "Batch on hold while packs are re-labelled. Regulatory review added as a mandatory artwork step."),
]


def _round(x: float) -> float:
    return round(x + 1e-9, 2)


def _active(product: dict, month: date) -> bool:
    launch = date.fromisoformat(product["launch_date"]).replace(day=1)
    if month < launch:
        return False
    disc = product["discontinued_on"]
    return disc is None or month <= date.fromisoformat(disc)


def _promo_for(sku: str, channel: str, region: str, month: date) -> dict | None:
    for p in PROMOTIONS:
        start, end = date.fromisoformat(p["start_date"]), date.fromisoformat(p["end_date"])
        if (p["sku"] == sku and p["channel"] == channel and p["region_id"] in (None, region)
                and start.replace(day=1) <= month <= end):
            return p
    return None


def build_tables() -> dict[str, list[dict]]:
    rng = random.Random(SEED)
    sales: list[dict] = []
    produced: dict[tuple[str, str, date], int] = {}  # (sku, plant, month) -> units sold, drives production
    for month in MONTHS:
        season = SEASON[month.month - 1]
        for product in PRODUCTS:
            if not _active(product, month):
                continue
            price = product["mrp_inr"] * DISTRIBUTOR_PRICE
            for dist in DISTRIBUTORS:
                if product["category"] == "tub" and dist["cold_chain_tier"] == "C":
                    continue  # onboarding policy: tier C distributors may not stock tubs
                region = dist["region_id"]
                for channel, share in CHANNELS[dist["distributor_id"]]:
                    if share == 0:
                        continue
                    growth = CHANNEL_GROWTH_2025[channel] if month.year == 2025 else 1.0
                    units = BASE_UNITS[product["sku"]] * share * season * REGION_FACTOR[region] * growth
                    units *= rng.uniform(0.9, 1.1)
                    promo = _promo_for(product["sku"], channel, region, month)
                    if promo:
                        units *= PROMO_UPLIFT
                    units = int(round(units))
                    gross = _round(units * price)
                    disc_rate = BASE_TRADE_DISCOUNT + (promo["discount_pct"] / 100 if promo else 0.0)
                    discount = _round(gross * disc_rate)
                    sales.append({
                        "sale_id": len(sales) + 1, "sale_month": month.isoformat(), "sku": product["sku"],
                        "distributor_id": dist["distributor_id"], "channel": channel, "units": units,
                        "gross_revenue_inr": gross, "discount_inr": discount,
                        "net_revenue_inr": _round(gross - discount),
                    })
                    key = (product["sku"], PLANT_OF_REGION[region], month)
                    produced[key] = produced.get(key, 0) + units

    incident_qc = {(sku, plant, month): qc for _, sku, plant, month, _, _, _, _, qc, *_ in _INCIDENTS}
    batches: list[dict] = []
    batch_of: dict[tuple[str, str, date], str] = {}
    per_plant_month: dict[tuple[str, date], int] = {}
    for (sku, plant, month), units in sorted(produced.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
        n = per_plant_month[(plant, month)] = per_plant_month.get((plant, month), 0) + 1
        batch_id = f"B{month:%y%m}-{plant}-{n:02d}"
        batch_of[(sku, plant, month)] = batch_id
        batches.append({
            "batch_id": batch_id, "plant_id": plant, "sku": sku,
            "produced_on": month.replace(day=1 + (n * 3) % 27).isoformat(),
            "units": int(round(units * 1.04)),  # small safety stock on top of what sold
            "qc_status": incident_qc.get((sku, plant, month), "passed"),
        })

    incidents = [
        {"incident_id": iid, "batch_id": batch_of[(sku, plant, month)], "reported_on": month.replace(day=day).isoformat(),
         "severity": sev, "category": cat, "description": desc, "root_cause": cause, "corrective_action": action,
         "status": status}
        for iid, sku, plant, month, day, sev, cat, status, _, desc, cause, action in _INCIDENTS
    ]

    ingredients = []
    for p in PRODUCTS:
        for name, supplier, grams in _INGREDIENTS[p["category"]] + _EXTRA[p["flavour"]]:
            if p["sku"] == "FF-TUB-SFV-500" and name == "sugar":
                name = "sucralose and maltitol blend"
            ingredients.append({"sku": p["sku"], "ingredient": name, "supplier_id": supplier,
                                "grams_per_unit": float(grams)})

    documents = [
        {"doc_id": d, "title": t, "category": c, "owner": o, "updated_on": u, "body_md": body.strip() + "\n"}
        for d, t, c, o, u, body in POLICIES
    ]
    return {
        "regions": REGIONS, "plants": PLANTS, "suppliers": SUPPLIERS, "products": PRODUCTS,
        "product_ingredients": ingredients, "distributors": DISTRIBUTORS, "sales": sales,
        "production_batches": batches, "promotions": PROMOTIONS, "quality_incidents": incidents,
        "documents": documents,
    }


TABLE_ORDER = ["regions", "plants", "suppliers", "products", "product_ingredients", "distributors", "sales",
               "production_batches", "promotions", "quality_incidents", "documents"]


def _literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def to_sql(tables: dict[str, list[dict]] | None = None, batch: int = 500) -> str:
    tables = tables or build_tables()
    out = ["-- Generated by `python -m rag_eval db seed-sql`. Do not edit by hand; edit rag_eval/warehouse/seed.py.",
           "BEGIN;"]
    for name in TABLE_ORDER:
        rows = tables[name]
        cols = list(rows[0])
        for i in range(0, len(rows), batch):
            values = ",\n".join("(" + ", ".join(_literal(r[c]) for c in cols) + ")" for r in rows[i : i + batch])
            out.append(f"INSERT INTO {name} ({', '.join(cols)}) VALUES\n{values};")
    out.append("COMMIT;")
    return "\n".join(out) + "\n"
