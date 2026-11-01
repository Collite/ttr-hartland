-- S2 — TV promotions (design 04 §S2): the TV-supported promotions' mean weekly uplift (promotion_result) on premium
-- lines ÷ that on the other lines, mechanic by mechanic (price-off 20 %, price-off 10 %, 1+1: the mechanic and its
-- discount), averaged over the mechanics both sides ran. The calendar runs more of the stronger mechanics on the premium
-- side, so one pooled mean would start 1.13× ahead before any premium factor. A promotion is TV-supported when its
-- campaign has a TV line.
WITH promo AS (
    SELECT pr.promotion_id, pr.mechanic, pr.discount_pct, pl.is_premium, AVG(r.uplift_pct) AS uplift
    FROM promotion pr
    JOIN campaign ca ON ca.campaign_id = pr.campaign_id
    JOIN promotion_result r ON r.promotion_id = pr.promotion_id
    JOIN product_line pl ON pl.product_line_id = pr.target_product_line_id
    WHERE ca.has_tv
    GROUP BY pr.promotion_id, pr.mechanic, pr.discount_pct, pl.is_premium
),
by_mechanic AS (
    SELECT
        mechanic,
        discount_pct,
        AVG(uplift) FILTER (WHERE is_premium) / NULLIF(AVG(uplift) FILTER (WHERE NOT is_premium), 0) AS ratio
    FROM promo
    GROUP BY mechanic, discount_pct
)
SELECT AVG(ratio) AS metric
FROM by_mechanic
