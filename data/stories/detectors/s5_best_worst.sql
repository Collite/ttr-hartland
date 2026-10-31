-- S5, the spread (design 04 §S5): the PoS return of the most skilled manager's region (PHA, 1.4) ÷ that of the least
-- skilled one who is not the generous one (VCH, 0.5) — ≥ 3 with the story on. The return as in s5.sql.
WITH loan AS (
    SELECT
        r.code AS region,
        pl.customer_id,
        pl.lent_on,
        LEAST(COALESCE(pl.returned_on, DATE '2026-10-31'), DATE '2026-10-31') AS until,
        pa.cost
    FROM pos_loan pl
    JOIN pos_asset pa ON pa.pos_asset_id = pl.pos_asset_id
    JOIN region r ON r.region_id = pa.region_id
    WHERE r.code IN ('PHA', 'VCH')
),
margin AS (
    SELECT
        loan.region,
        loan.until - loan.lent_on + 1 AS days,
        loan.cost * (loan.until - loan.lent_on + 1) / (36 * 30.4) AS cost,
        SUM(s.customer_margin) FILTER (WHERE s.ordered_on >= loan.lent_on) / (loan.until - loan.lent_on + 1)
            - SUM(s.customer_margin) FILTER (WHERE s.ordered_on < loan.lent_on) / 56.0 AS daily_gain
    FROM loan
    JOIN sales_pl s ON s.customer_id = loan.customer_id AND s.ordered_on BETWEEN loan.lent_on - 56 AND loan.until
    GROUP BY loan.region, loan.customer_id, loan.lent_on, loan.until, loan.cost
),
return_by_region AS (
    SELECT region, (SUM(COALESCE(daily_gain, 0) * days) - SUM(cost)) / NULLIF(SUM(cost), 0) AS pos_return
    FROM margin
    GROUP BY region
)
SELECT MAX(pos_return) FILTER (WHERE region = 'PHA') / NULLIF(MAX(pos_return) FILTER (WHERE region = 'VCH'), 0)
    AS metric
FROM return_by_region
