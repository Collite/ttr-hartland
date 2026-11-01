-- S5 — PoS efficiency by area manager (design 04 §S5): per region, the return on its displays, (gain − cost) ÷ cost,
-- from the block below. The metric is Spearman's rho between the regions' returns and their managers' pos_skill, on
-- average ranks (VCH and SMS, both 0.5, share theirs; s5.yaml, and a test keeps the pairs below in step).
WITH skill (region, pos_skill) AS (
    VALUES ('PHA', 1.4), ('JMV', 1.3), ('SJC', 1.0), ('ZSC', 0.6), ('VCH', 0.5), ('SMS', 0.5)
),
-- region return: begin
-- Per region, up to the world's last order day: the gain of its displays and their cost. A loan's gain is the customer
-- margin at the display's store (at the customer, when the loan names no store) per day during the loan, minus that of
-- the 8 weeks before, times the loan's days: a store that bought nothing before gains all it buys, and a loan without
-- orders keeps its cost. Only loans lent from 2024-02-26 count, so their 8 weeks before are inside the history. The
-- cost is the asset's over 36 months for the loan's days, plus the region's give-aways.
as_of AS (
    SELECT MAX(ordered_on) AS day FROM sales_order
),
loan AS (
    SELECT
        pl.pos_loan_id,
        r.code AS region,
        pl.customer_id,
        pl.store_id,
        pl.lent_on,
        LEAST(COALESCE(pl.returned_on, as_of.day), as_of.day) AS until,
        pa.cost
    FROM pos_loan pl
    CROSS JOIN as_of
    JOIN pos_asset pa ON pa.pos_asset_id = pl.pos_asset_id
    JOIN region r ON r.region_id = pa.region_id
    WHERE pl.lent_on BETWEEN DATE '2024-02-26' AND as_of.day
),
margin AS (
    SELECT
        loan.pos_loan_id,
        loan.region,
        loan.cost * (loan.until - loan.lent_on + 1) / (36 * 30.4) AS cost,
        (
            COALESCE(SUM(s.customer_margin) FILTER (WHERE s.ordered_on >= loan.lent_on), 0)
                / (loan.until - loan.lent_on + 1)
            - COALESCE(SUM(s.customer_margin) FILTER (WHERE s.ordered_on < loan.lent_on), 0) / 56.0
        ) * (loan.until - loan.lent_on + 1) AS gain
    FROM loan
    LEFT JOIN sales_pl s ON s.customer_id = loan.customer_id
        AND (loan.store_id IS NULL OR s.store_id = loan.store_id)
        AND s.ordered_on BETWEEN loan.lent_on - 56 AND loan.until
    GROUP BY loan.pos_loan_id, loan.region, loan.lent_on, loan.until, loan.cost
),
giveaway AS (
    SELECT r.code AS region, SUM(h.cost) AS cost
    FROM pos_handout h
    JOIN visit v ON v.visit_id = h.visit_id
    JOIN store st ON st.store_id = v.store_id
    JOIN area a ON a.area_id = st.area_id
    JOIN region r ON r.region_id = a.region_id
    CROSS JOIN as_of
    WHERE v.day BETWEEN DATE '2024-01-01' AND as_of.day
    GROUP BY r.code
),
region_return AS (
    SELECT m.region, SUM(m.gain) AS gain, SUM(m.cost) + COALESCE(MAX(g.cost), 0) AS cost
    FROM margin m
    LEFT JOIN giveaway g ON g.region = m.region
    GROUP BY m.region
)
-- region return: end
SELECT corr(rr, rs) AS metric
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
