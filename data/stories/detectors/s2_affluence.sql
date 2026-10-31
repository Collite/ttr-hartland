-- S2, the affluence term (design 04 §S2): within the TV-supported promotions, the promoted line's uplift — its daily
-- cases during the promotion ÷ in the 8 weeks before, minus 1 — in the richest quarter of the okresy ÷ that in the
-- poorest quarter (by okres.affluence_index; Praha counts as its 22 districts).
WITH tv AS (
    SELECT pr.promotion_id, pr.target_product_line_id AS line_id, pr.valid_from, pr.valid_to
    FROM promotion pr
    JOIN campaign ca ON ca.campaign_id = pr.campaign_id
    WHERE ca.has_tv AND pr.target_product_line_id IS NOT NULL
),
quartile AS (
    SELECT k.okres_id, NTILE(4) OVER (ORDER BY k.affluence_index) AS q
    FROM okres k
    WHERE NOT EXISTS (SELECT 1 FROM okres child WHERE child.parent_okres_id = k.okres_id)
),
cases AS (
    SELECT
        tv.promotion_id,
        quartile.q,
        SUM(l.qty_cases) FILTER (WHERE o.ordered_on >= tv.valid_from) / (tv.valid_to - tv.valid_from + 1) AS during,
        SUM(l.qty_cases) FILTER (WHERE o.ordered_on < tv.valid_from) / 56.0 AS before
    FROM tv
    JOIN sales_order o ON o.ordered_on BETWEEN tv.valid_from - 56 AND tv.valid_to
    JOIN sales_order_line l ON l.sales_order_id = o.sales_order_id
    JOIN sku s ON s.sku_id = l.sku_id
    JOIN product p ON p.product_id = s.product_id AND p.product_line_id = tv.line_id
    JOIN delivery_point dp ON dp.delivery_point_id = o.delivery_point_id
    JOIN quartile ON quartile.okres_id = dp.okres_id
    GROUP BY tv.promotion_id, quartile.q, tv.valid_from, tv.valid_to
)
SELECT AVG(during / NULLIF(before, 0) - 1) FILTER (WHERE q = 4)
    / NULLIF(AVG(during / NULLIF(before, 0) - 1) FILTER (WHERE q = 1), 0) AS metric
FROM cases
