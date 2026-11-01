-- S4 — the shops cluster (design 04 §S4), the demo's detector: per store, the line with the most cases in 2025; per
-- street type of s4.yaml `leading_line`, the share of its ordering stores led by that line. The metric is the
-- smallest share over the three types (≥ 0.8 with the story on). The k-means detector of 04 (adjusted Rand index
-- ≥ 0.6 against street_type) is the pandas twin's, G1 S1.2. The pairs below are s4.yaml's (a test keeps them so).
WITH expected (street_type, line) AS (
    VALUES ('transport_hub', 'SMESI'), ('village', 'POLEVKY'), ('high_street', 'GOLD')
),
store_line AS (
    SELECT dp.store_id, pl.code AS line, SUM(l.qty_cases) AS cases
    FROM sales_order_line l
    JOIN sales_order o ON o.sales_order_id = l.sales_order_id
    JOIN delivery_point dp ON dp.delivery_point_id = o.delivery_point_id
    JOIN sku s ON s.sku_id = l.sku_id
    JOIN product p ON p.product_id = s.product_id
    JOIN product_line pl ON pl.product_line_id = p.product_line_id
    WHERE o.ordered_on BETWEEN DATE '2025-01-01' AND DATE '2025-12-31' AND dp.store_id IS NOT NULL
      AND NOT l.is_free_goods
    GROUP BY dp.store_id, pl.code
),
leader AS (
    SELECT DISTINCT ON (store_id) store_id, line
    FROM store_line
    ORDER BY store_id, cases DESC, line
)
SELECT MIN(share) AS metric
FROM (
    SELECT e.street_type, AVG(CASE WHEN leader.line = e.line THEN 1.0 ELSE 0.0 END) AS share
    FROM expected e
    JOIN store st ON st.street_type = e.street_type
    JOIN leader ON leader.store_id = st.store_id
    GROUP BY e.street_type
) shares
