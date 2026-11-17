-- Frostfield Creamery: a fictional Indian ice cream maker (FMCG).
-- Schema for the warehouse the assistant answers questions from.
--
-- The COMMENT ON statements are the data dictionary. The web UI's Data tab and
-- the SQL writer both read them from the database, so keep them accurate.
-- Docker runs the files in this folder in name order on first start.

CREATE TABLE regions (
    region_id        text PRIMARY KEY,
    name             text NOT NULL UNIQUE,
    head_office_city text NOT NULL,
    sales_head       text NOT NULL
);
COMMENT ON TABLE regions IS 'The four sales regions of India the company sells in.';
COMMENT ON COLUMN regions.region_id IS 'Short code: NORTH, WEST, SOUTH or EAST.';
COMMENT ON COLUMN regions.name IS 'Region name, e.g. North.';
COMMENT ON COLUMN regions.sales_head IS 'Regional sales manager responsible for the region.';

CREATE TABLE plants (
    plant_id                text PRIMARY KEY,
    name                    text NOT NULL,
    city                    text NOT NULL,
    region_id               text NOT NULL REFERENCES regions,
    capacity_litres_per_day integer NOT NULL CHECK (capacity_litres_per_day > 0),
    opened_on               date NOT NULL
);
COMMENT ON TABLE plants IS 'Manufacturing plants. Each plant serves the distributors in its region.';
COMMENT ON COLUMN plants.capacity_litres_per_day IS 'Maximum ice cream output per day, in litres.';

CREATE TABLE suppliers (
    supplier_id        text PRIMARY KEY,
    name               text NOT NULL,
    category           text NOT NULL CHECK (category IN ('dairy', 'sugar', 'fruit', 'flavour', 'cocoa', 'packaging')),
    city               text NOT NULL,
    rating             numeric(2, 1) NOT NULL CHECK (rating BETWEEN 1 AND 5),
    fssai_certified    boolean NOT NULL,
    payment_terms_days integer NOT NULL,
    notes              text NOT NULL
);
COMMENT ON TABLE suppliers IS 'Raw material and packaging suppliers.';
COMMENT ON COLUMN suppliers.category IS 'What they supply: dairy, sugar, fruit, flavour, cocoa or packaging.';
COMMENT ON COLUMN suppliers.rating IS 'Internal supplier score from 1 (poor) to 5 (excellent), reviewed every quarter.';
COMMENT ON COLUMN suppliers.fssai_certified IS 'Holds a valid FSSAI (Food Safety and Standards Authority of India) licence.';
COMMENT ON COLUMN suppliers.notes IS 'Procurement team notes: reliability, issues, contract details.';

CREATE TABLE products (
    sku            text PRIMARY KEY,
    name           text NOT NULL UNIQUE,
    category       text NOT NULL CHECK (category IN ('cup', 'cone', 'stick', 'tub', 'kulfi', 'sorbet')),
    flavour        text NOT NULL,
    pack_size_ml   integer NOT NULL,
    mrp_inr        numeric(8, 2) NOT NULL,
    unit_cost_inr  numeric(8, 2) NOT NULL,
    launch_date    date NOT NULL,
    status         text NOT NULL CHECK (status IN ('active', 'discontinued')),
    discontinued_on date,
    description    text NOT NULL
);
COMMENT ON TABLE products IS 'Product catalogue: one row per SKU (stock keeping unit).';
COMMENT ON COLUMN products.category IS 'Format: cup, cone, stick, tub, kulfi or sorbet.';
COMMENT ON COLUMN products.mrp_inr IS 'Maximum retail price per unit, in rupees.';
COMMENT ON COLUMN products.unit_cost_inr IS 'Cost to make one unit (ingredients, packaging, plant), in rupees.';
COMMENT ON COLUMN products.status IS 'active or discontinued.';

CREATE TABLE product_ingredients (
    sku            text NOT NULL REFERENCES products,
    ingredient     text NOT NULL,
    supplier_id    text NOT NULL REFERENCES suppliers,
    grams_per_unit numeric(6, 1) NOT NULL,
    PRIMARY KEY (sku, ingredient)
);
COMMENT ON TABLE product_ingredients IS 'Bill of materials: the main ingredients of each SKU and who supplies them.';

CREATE TABLE distributors (
    distributor_id   text PRIMARY KEY,
    name             text NOT NULL,
    region_id        text NOT NULL REFERENCES regions,
    city             text NOT NULL,
    cold_chain_tier  text NOT NULL CHECK (cold_chain_tier IN ('A', 'B', 'C')),
    credit_limit_inr numeric(12, 2) NOT NULL,
    onboarded_on     date NOT NULL
);
COMMENT ON TABLE distributors IS 'Distributors who buy from the company and sell on to retailers.';
COMMENT ON COLUMN distributors.cold_chain_tier IS 'A = generator-backed freezers and reefer fleet, B = freezers only, C = basic storage.';

CREATE TABLE sales (
    sale_id          bigint PRIMARY KEY,
    sale_month       date NOT NULL,
    sku              text NOT NULL REFERENCES products,
    distributor_id   text NOT NULL REFERENCES distributors,
    channel          text NOT NULL CHECK (channel IN ('modern_trade', 'general_trade', 'quick_commerce', 'horeca')),
    units            integer NOT NULL CHECK (units >= 0),
    gross_revenue_inr numeric(14, 2) NOT NULL,
    discount_inr     numeric(14, 2) NOT NULL,
    net_revenue_inr  numeric(14, 2) NOT NULL
);
COMMENT ON TABLE sales IS 'Monthly sell-in: units and revenue per SKU, distributor and channel. Covers Jan 2024 to Dec 2025.';
COMMENT ON COLUMN sales.sale_month IS 'First day of the month the sales belong to.';
COMMENT ON COLUMN sales.channel IS 'modern_trade (supermarkets), general_trade (corner shops), quick_commerce (10-minute delivery apps) or horeca (hotels, restaurants, cafes).';
COMMENT ON COLUMN sales.gross_revenue_inr IS 'Units times the price charged to the distributor, before discounts.';
COMMENT ON COLUMN sales.discount_inr IS 'Promotional and trade discounts given.';
COMMENT ON COLUMN sales.net_revenue_inr IS 'gross_revenue_inr minus discount_inr. Use this for "revenue" unless gross is asked for.';
CREATE INDEX sales_month_idx ON sales (sale_month);
CREATE INDEX sales_sku_idx ON sales (sku);

CREATE TABLE production_batches (
    batch_id    text PRIMARY KEY,
    plant_id    text NOT NULL REFERENCES plants,
    sku         text NOT NULL REFERENCES products,
    produced_on date NOT NULL,
    units       integer NOT NULL CHECK (units > 0),
    qc_status   text NOT NULL CHECK (qc_status IN ('passed', 'held', 'rejected'))
);
COMMENT ON TABLE production_batches IS 'Production runs. Every unit sold can be traced back to a batch.';
COMMENT ON COLUMN production_batches.qc_status IS 'Quality check result: passed, held (under investigation) or rejected (destroyed).';

CREATE TABLE promotions (
    promo_id     text PRIMARY KEY,
    name         text NOT NULL,
    sku          text NOT NULL REFERENCES products,
    channel      text NOT NULL,
    region_id    text REFERENCES regions,
    start_date   date NOT NULL,
    end_date     date NOT NULL,
    discount_pct numeric(4, 1) NOT NULL,
    notes        text NOT NULL
);
COMMENT ON TABLE promotions IS 'Trade and consumer promotions. region_id NULL means the promotion ran nationally.';
COMMENT ON COLUMN promotions.discount_pct IS 'Discount off the distributor price, in percent.';
COMMENT ON COLUMN promotions.notes IS 'Mechanic, objective and outcome of the promotion.';

CREATE TABLE quality_incidents (
    incident_id       text PRIMARY KEY,
    batch_id          text NOT NULL REFERENCES production_batches,
    reported_on       date NOT NULL,
    severity          text NOT NULL CHECK (severity IN ('low', 'medium', 'high')),
    category          text NOT NULL,
    description       text NOT NULL,
    root_cause        text NOT NULL,
    corrective_action text NOT NULL,
    status            text NOT NULL CHECK (status IN ('open', 'closed'))
);
COMMENT ON TABLE quality_incidents IS 'Quality and food safety incidents, linked to the batch involved.';
COMMENT ON COLUMN quality_incidents.category IS 'temperature_excursion, foreign_matter, packaging, allergen, texture or labelling.';

CREATE TABLE documents (
    doc_id     text PRIMARY KEY,
    title      text NOT NULL,
    category   text NOT NULL,
    owner      text NOT NULL,
    updated_on date NOT NULL,
    body_md    text NOT NULL
);
COMMENT ON TABLE documents IS 'Company policies and standard operating procedures (SOPs), stored as markdown.';
COMMENT ON COLUMN documents.body_md IS 'The full document in markdown. Searched by the document (RAG) path, not by SQL.';
