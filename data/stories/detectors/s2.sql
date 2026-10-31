-- S2 — TV promotions (design 04 §S2): the mean weekly uplift (promotion_result) of the TV-supported promotions on
-- premium lines ÷ that of the TV-supported promotions on the other lines. A promotion is TV-supported when its
-- campaign has a TV line.
WITH promo AS (
    SELECT pr.promotion_id, pl.is_premium, COALESCE(ca.has_tv, false) AS has_tv, AVG(r.uplift_pct) AS uplift
    FROM promotion pr
    JOIN promotion_result r ON r.promotion_id = pr.promotion_id
    JOIN product_line pl ON pl.product_line_id = pr.target_product_line_id
    LEFT JOIN campaign ca ON ca.campaign_id = pr.campaign_id
    GROUP BY pr.promotion_id, pl.is_premium, ca.has_tv
)
SELECT AVG(uplift) FILTER (WHERE has_tv AND is_premium)
    / NULLIF(AVG(uplift) FILTER (WHERE has_tv AND NOT is_premium), 0) AS metric
FROM promo
