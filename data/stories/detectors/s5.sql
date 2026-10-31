-- S5 — PoS efficiency by area manager (design 04 §S5): per region, the return on its displays — (the incremental
-- customer margin of the stores with an active loan: their daily margin during the loan minus that of the 8 weeks
-- before, times the loan's days) minus the PoS cost (the asset's cost over 36 months for the loan's months, and the
-- region's give-aways), over the PoS cost. The metric is Spearman's rho between the regions' returns and their
-- managers' pos_skill (s5.yaml; a test keeps the pairs below in step).
WITH skill (region, pos_skill) AS (
    VALUES ('PHA', 1.4), ('JMV', 1.3), ('SJC', 1.0), ('ZSC', 0.6), ('VCH', 0.5), ('SMS', 0.5)
),
loan AS (
    SELECT
        r.code AS region,
        pl.customer_id,
        pl.lent_on,
        LEAST(COALESCE(pl.returned_on, DATE '2026-10-31'), DATE '2026-10-31') AS until,
        pa.cost
    FROM pos_loan pl
    JOIN pos_asset pa ON pa.pos_asset_id = pl.pos_asset_id
    JOIN region r ON r.region_id = pa.region_id
),
margin AS (
    SELECT
        loan.region,
        loan.customer_id,
        loan.until - loan.lent_on + 1 AS days,
        loan.cost * (loan.until - loan.lent_on + 1) / (36 * 30.4) AS cost,
        SUM(s.customer_margin) FILTER (WHERE s.ordered_on >= loan.lent_on) / (loan.until - loan.lent_on + 1)
            - SUM(s.customer_margin) FILTER (WHERE s.ordered_on < loan.lent_on) / 56.0 AS daily_gain
    FROM loan
    JOIN sales_pl s ON s.customer_id = loan.customer_id AND s.ordered_on BETWEEN loan.lent_on - 56 AND loan.until
    GROUP BY loan.region, loan.customer_id, loan.lent_on, loan.until, loan.cost
),
giveaway AS (
    SELECT r.code AS region, SUM(h.cost) AS cost
    FROM pos_handout h
    JOIN visit v ON v.visit_id = h.visit_id
    JOIN store st ON st.store_id = v.store_id
    JOIN area a ON a.area_id = st.area_id
    JOIN region r ON r.region_id = a.region_id
    GROUP BY r.code
),
return_by_region AS (
    SELECT
        m.region,
        (SUM(COALESCE(m.daily_gain, 0) * m.days) - SUM(m.cost) - COALESCE(MAX(g.cost), 0))
            / NULLIF(SUM(m.cost) + COALESCE(MAX(g.cost), 0), 0) AS pos_return
    FROM margin m
    LEFT JOIN giveaway g ON g.region = m.region
    GROUP BY m.region
)
SELECT corr(rr, rs) AS metric
FROM (
    SELECT RANK() OVER (ORDER BY r.pos_return) AS rr, RANK() OVER (ORDER BY s.pos_skill) AS rs
    FROM return_by_region r
    JOIN skill s ON s.region = r.region
) ranks
