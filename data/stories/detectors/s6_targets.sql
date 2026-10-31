-- S6, the product managers (design 04 §S6): the share of the disciplined product manager's promotions whose realised
-- uplift lands within ±10 % of the target, minus that of the undisciplined one (s6.yaml `pinned_product_managers`:
-- the managers of GOLD and CLASSIC; a test keeps the codes below in step) — ≥ 0.4 with the story on.
WITH pinned (line, disciplined) AS (
    VALUES ('GOLD', true), ('CLASSIC', false)
),
result AS (
    SELECT pr.promotion_id, pinned.disciplined, pr.target_uplift_pct, AVG(r.uplift_pct) AS uplift
    FROM promotion pr
    JOIN product_line pl ON pl.product_line_id = pr.target_product_line_id
    JOIN pinned ON pinned.line = pl.code
    JOIN promotion_result r ON r.promotion_id = pr.promotion_id
    GROUP BY pr.promotion_id, pinned.disciplined, pr.target_uplift_pct
)
SELECT AVG(CASE WHEN ABS(uplift - target_uplift_pct) <= 0.10 * target_uplift_pct THEN 1.0 ELSE 0.0 END)
        FILTER (WHERE disciplined)
    - AVG(CASE WHEN ABS(uplift - target_uplift_pct) <= 0.10 * target_uplift_pct THEN 1.0 ELSE 0.0 END)
        FILTER (WHERE NOT disciplined) AS metric
FROM result
