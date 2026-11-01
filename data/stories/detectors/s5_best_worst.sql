-- S5, the spread (design 04 §S5): gain ÷ cost of the displays in the region of the most skilled manager (PHA, 1.4)
-- over that in the region of the least skilled one who is not the generous one (VCH, 0.5). Gain ÷ cost is 1 + s5.sql's
-- return: positive unless a region's displays destroy margin, and then the metric is NULL, which fails. The threshold,
-- 3, is provisional: 04 set it for the return, and gain ÷ cost reads the same spread smaller. G2 S2.3 derives it from
-- the mechanics (s5.yaml) once T2's placements and T4's give-aways exist, at T6, before the first build. The block
-- below is s5.sql's, byte for byte (a test keeps it so).
WITH
-- region return: begin
-- Per region, up to the world's last order day: the gain of its displays and their cost. A loan's gain is the customer
-- margin at the display's store (at the customer, when the loan names no store) per day during the loan, minus that of
-- the 8 weeks before, times the loan's days: a store that bought nothing before gains all it buys, and a loan without
-- orders keeps its cost. Only loans lent from 2024-02-26 count, so their 8 weeks before are inside the history. The
-- cost is the asset's over 36 months for the loan's days, plus the region's give-aways from the same day on.
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
    WHERE v.day BETWEEN DATE '2024-02-26' AND as_of.day
    GROUP BY r.code
),
region_return AS (
    SELECT m.region, SUM(m.gain) AS gain, SUM(m.cost) + COALESCE(MAX(g.cost), 0) AS cost
    FROM margin m
    LEFT JOIN giveaway g ON g.region = m.region
    GROUP BY m.region
)
-- region return: end
SELECT MAX(gain / NULLIF(cost, 0)) FILTER (WHERE region = 'PHA')
    / NULLIF(GREATEST(MAX(gain / NULLIF(cost, 0)) FILTER (WHERE region = 'VCH'), 0), 0) AS metric
FROM region_return
