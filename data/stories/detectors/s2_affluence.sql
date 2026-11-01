-- S2, the affluence term (design 04 §S2): within the TV-supported promotions, the promoted line's lift — its daily
-- cases during the promotion ÷ in the 8 weeks before — in the richest quarter of the okresy ÷ that in the poorest
-- quarter (by okres.affluence_index; Praha counts as its 22 districts). Both sides count only the clean days, when no
-- other promotion of the line ran: Gold's September 1+1 falls in its Christmas promotion's 8 weeks before. A lift, not
-- an uplift (lift − 1), so the denominator stays positive. Promotions count from 2024-02-26 (their 8 weeks before
-- inside the history) and up to the world's last order day. The cases are the orders delivered to a store, or to a web
-- consumer's town, in that place's okres: a chain that orders to its own DC (world.yaml big_retail.dc_level_orders)
-- would put its national volume in the okres of the DC.
WITH as_of AS (
    SELECT MAX(ordered_on) AS day FROM sales_order
),
tv AS (
    SELECT pr.promotion_id, pr.target_product_line_id AS line_id, pr.valid_from,
           LEAST(pr.valid_to, as_of.day) AS valid_to
    FROM promotion pr
    CROSS JOIN as_of
    JOIN campaign ca ON ca.campaign_id = pr.campaign_id
    WHERE ca.has_tv AND pr.valid_from BETWEEN DATE '2024-02-26' AND as_of.day
),
clean_day AS (
    SELECT tv.promotion_id, tv.line_id, d.day, d.day >= tv.valid_from AS during
    FROM tv
    CROSS JOIN LATERAL (
        SELECT g::date AS day FROM generate_series(tv.valid_from - 56, tv.valid_to, interval '1 day') g
    ) d
    WHERE NOT EXISTS (
        SELECT 1
        FROM promotion other
        WHERE other.target_product_line_id = tv.line_id AND other.promotion_id <> tv.promotion_id
          AND d.day BETWEEN other.valid_from AND other.valid_to
    )
),
span AS (
    SELECT
        promotion_id,
        COUNT(*) FILTER (WHERE during) AS during_days,
        COUNT(*) FILTER (WHERE NOT during) AS before_days
    FROM clean_day
    GROUP BY promotion_id
),
quartile AS (
    SELECT k.okres_id, NTILE(4) OVER (ORDER BY k.affluence_index) AS q
    FROM okres k
    WHERE NOT EXISTS (SELECT 1 FROM okres child WHERE child.parent_okres_id = k.okres_id)
),
cases AS (
    SELECT
        cd.promotion_id,
        quartile.q,
        SUM(l.qty_cases) FILTER (WHERE cd.during) AS during_cases,
        SUM(l.qty_cases) FILTER (WHERE NOT cd.during) AS before_cases
    FROM clean_day cd
    JOIN sales_order o ON o.ordered_on = cd.day
    JOIN sales_order_line l ON l.sales_order_id = o.sales_order_id
    JOIN sku s ON s.sku_id = l.sku_id
    JOIN product p ON p.product_id = s.product_id AND p.product_line_id = cd.line_id
    JOIN delivery_point dp ON dp.delivery_point_id = o.delivery_point_id
    JOIN quartile ON quartile.okres_id = dp.okres_id
    WHERE dp.store_id IS NOT NULL OR o.channel = 'web'
    GROUP BY cd.promotion_id, quartile.q
),
lift AS (
    SELECT c.q, (c.during_cases / NULLIF(sp.during_days, 0)) / NULLIF(c.before_cases / NULLIF(sp.before_days, 0), 0)
        AS lift
    FROM cases c
    JOIN span sp ON sp.promotion_id = c.promotion_id
)
SELECT AVG(lift) FILTER (WHERE q = 4) / NULLIF(AVG(lift) FILTER (WHERE q = 1), 0) AS metric
FROM lift
