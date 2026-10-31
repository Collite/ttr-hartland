-- A-10 — the promotion that cannibalised: 1 when PR-2025-10-CLASSIC-200 lifted its SKU by 30 % or more and still
-- lost customer margin overall.
SELECT COUNT(*)
FROM (
    SELECT pr.promotion_id
    FROM promotion pr
    JOIN promotion_result r ON r.promotion_id = pr.promotion_id
    WHERE pr.code = 'PR-2025-10-CLASSIC-200'
    GROUP BY pr.promotion_id
    HAVING AVG(r.uplift_pct) >= 30 AND SUM(r.incremental_customer_margin) < 0
) cannibal
