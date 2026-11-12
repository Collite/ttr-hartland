-- S5 — PoS efficiency by area manager (design 04 §S5): per region, the return on its displays, (gain − cost) ÷ cost,
-- from the block below. The metric is Spearman's rho between the regions' returns and their managers' pos_skill, on
-- average ranks (VCH and SMS, both 0.5, share theirs; s5.yaml, and a test keeps the pairs below in step). NULL when
-- a region's coverage is under 0.9 (the block below; G2 S2.4, NH-D17).
WITH skill (region, pos_skill) AS (
    VALUES ('PHA', 1.4), ('JMV', 1.3), ('SJC', 1.0), ('ZSC', 0.6), ('VCH', 0.5), ('SMS', 0.5)
),
-- region return: begin
-- Per region, up to the world's last order day: the gain of its displays and their cost (NH-D17). A loan's gain is
-- the Customer Margin a day at its store (at its customer, when the loan names no store) during the loan, less that
-- of the 56 days before × q, times the loan's days. q is its controls' margin a day during over before: the stores of
-- its store's region, its customer's channel and its store's class, street type and size class, but its own and any
-- store with a display out in the 56 days before or during the loan — the same stores in both windows. A loan is
-- supported by 5 controls or more whose margin before is positive; otherwise, or without a store, its gain is not
-- known and counts as 0, while its cost counts. `coverage` is the supported share of a region's loan cost: under 0.9
-- in any region, both metrics are NULL. A store that bought nothing before gains all it buys during: kept, and
-- reported apart (`new_store_gain`). `raw_gain` is S2.3's gain, q = 1, a diagnostic (and the counts). Only loans lent from 2024-02-26
-- count, so their 56 days before are inside the history. The cost is the asset's over 36 months for the loan's days,
-- plus the region's give-aways from the same day on.
as_of AS (
    SELECT MAX(ordered_on) AS day, DATE '2024-02-26' AS first_day FROM sales_order
),
held AS (
    SELECT pl.store_id, pl.lent_on, LEAST(COALESCE(pl.returned_on, as_of.day), as_of.day) AS until
    FROM pos_loan pl
    CROSS JOIN as_of
    WHERE pl.store_id IS NOT NULL
),
outlet AS (
    SELECT DISTINCT dp.customer_id, dp.store_id, c.channel, st.store_class, st.street_type, st.size_class, a.region_id
    FROM delivery_point dp
    JOIN customer c ON c.customer_id = dp.customer_id
    JOIN store st ON st.store_id = dp.store_id
    JOIN area a ON a.area_id = st.area_id
),
daily AS (
    SELECT customer_id, store_id, ordered_on, SUM(customer_margin) AS margin
    FROM sales_pl
    GROUP BY customer_id, store_id, ordered_on
),
loan AS (
    SELECT
        pl.pos_loan_id,
        r.code AS region,
        pl.customer_id,
        pl.store_id,
        pl.lent_on,
        LEAST(COALESCE(pl.returned_on, as_of.day), as_of.day) AS until,
        LEAST(COALESCE(pl.returned_on, as_of.day), as_of.day) - pl.lent_on + 1 AS days,
        pa.cost
    FROM pos_loan pl
    CROSS JOIN as_of
    JOIN pos_asset pa ON pa.pos_asset_id = pl.pos_asset_id
    JOIN region r ON r.region_id = pa.region_id
    WHERE pl.lent_on BETWEEN as_of.first_day AND as_of.day
),
treated AS (
    SELECT
        loan.pos_loan_id,
        COALESCE(SUM(d.margin) FILTER (WHERE d.ordered_on >= loan.lent_on), 0) / loan.days AS during,
        COALESCE(SUM(d.margin) FILTER (WHERE d.ordered_on < loan.lent_on), 0) / 56.0 AS before
    FROM loan
    LEFT JOIN daily d ON d.customer_id = loan.customer_id
        AND (loan.store_id IS NULL OR d.store_id = loan.store_id)
        AND d.ordered_on BETWEEN loan.lent_on - 56 AND loan.until
    GROUP BY loan.pos_loan_id, loan.days
),
control AS (
    SELECT loan.pos_loan_id, o.customer_id, o.store_id, COUNT(*) OVER (PARTITION BY loan.pos_loan_id) AS controls
    FROM loan
    JOIN outlet t ON t.customer_id = loan.customer_id AND t.store_id = loan.store_id
    JOIN outlet o ON o.region_id = t.region_id
        AND o.channel = t.channel
        AND o.store_class = t.store_class
        AND o.street_type = t.street_type
        AND o.size_class = t.size_class
        AND o.store_id <> t.store_id
    WHERE NOT EXISTS (
        SELECT 1
        FROM held h
        WHERE h.store_id = o.store_id AND h.lent_on <= loan.until AND h.until >= loan.lent_on - 56
    )
),
cohort AS (
    SELECT
        c.pos_loan_id,
        c.controls,
        COALESCE(SUM(d.margin) FILTER (WHERE d.ordered_on >= loan.lent_on), 0) / loan.days AS during,
        COALESCE(SUM(d.margin) FILTER (WHERE d.ordered_on < loan.lent_on), 0) / 56.0 AS before
    FROM control c
    JOIN loan ON loan.pos_loan_id = c.pos_loan_id
    LEFT JOIN daily d ON d.customer_id = c.customer_id
        AND d.store_id = c.store_id
        AND d.ordered_on BETWEEN loan.lent_on - 56 AND loan.until
    GROUP BY c.pos_loan_id, c.controls, loan.days
),
margin AS (
    SELECT
        loan.pos_loan_id,
        loan.region,
        loan.cost * loan.days / (36 * 30.4) AS cost,
        (t.during - t.before) * loan.days AS raw_gain,
        CASE
            WHEN c.controls >= 5 AND c.before > 0 THEN (t.during - t.before * c.during / c.before) * loan.days
        END AS gain,
        t.before = 0 AS new_store
    FROM loan
    JOIN treated t ON t.pos_loan_id = loan.pos_loan_id
    LEFT JOIN cohort c ON c.pos_loan_id = loan.pos_loan_id
),
giveaway AS (
    SELECT r.code AS region, SUM(h.cost) AS cost
    FROM pos_handout h
    JOIN visit v ON v.visit_id = h.visit_id
    JOIN store st ON st.store_id = v.store_id
    JOIN area a ON a.area_id = st.area_id
    JOIN region r ON r.region_id = a.region_id
    CROSS JOIN as_of
    WHERE v.day BETWEEN as_of.first_day AND as_of.day
    GROUP BY r.code
),
region_return AS (
    SELECT
        m.region,
        COALESCE(SUM(m.gain), 0) AS gain,
        SUM(m.cost) + COALESCE(MAX(g.cost), 0) AS cost,
        SUM(m.raw_gain) AS raw_gain,
        COALESCE(SUM(m.gain) FILTER (WHERE m.new_store), 0) AS new_store_gain,
        COALESCE(SUM(m.cost) FILTER (WHERE m.gain IS NOT NULL), 0) / NULLIF(SUM(m.cost), 0) AS coverage,
        COUNT(*) AS loans,
        COUNT(*) FILTER (WHERE m.gain IS NULL) AS unknown,
        COUNT(*) FILTER (WHERE m.new_store) AS new_stores
    FROM margin m
    LEFT JOIN giveaway g ON g.region = m.region
    GROUP BY m.region
)
-- region return: end
SELECT CASE WHEN (SELECT MIN(coverage) FROM region_return) >= 0.9 THEN corr(rr, rs) END AS metric
FROM (
    SELECT
        RANK() OVER (ORDER BY pos_return) + (COUNT(*) OVER (PARTITION BY pos_return) - 1) / 2.0 AS rr,
        RANK() OVER (ORDER BY pos_skill) + (COUNT(*) OVER (PARTITION BY pos_skill) - 1) / 2.0 AS rs
    FROM (
        SELECT (r.gain - r.cost) / NULLIF(r.cost, 0) AS pos_return, s.pos_skill
        FROM region_return r
        JOIN skill s ON s.region = r.region
    ) returns
) ranks
