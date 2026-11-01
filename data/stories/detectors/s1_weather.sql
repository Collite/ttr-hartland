-- S1, the weather term (design 04 §S1): Pearson's r of a region's weekly dried-soup cases (each region's cases ÷ its
-- own mean, so the large regions do not dominate) against its weekly mean temperature, 2024–2025. Cold weeks sell
-- soup: the story gives r ≤ −0.5. The cases are the orders delivered where they are sold, as in s2_affluence.sql.
WITH weekly_cases AS (
    SELECT a.region_id, date_trunc('week', o.ordered_on)::date AS week, SUM(l.qty_cases) AS cases
    FROM sales_order_line l
    JOIN sales_order o ON o.sales_order_id = l.sales_order_id
    JOIN delivery_point dp ON dp.delivery_point_id = o.delivery_point_id
    JOIN area_okres ao ON ao.okres_id = dp.okres_id
    JOIN area a ON a.area_id = ao.area_id
    JOIN sku s ON s.sku_id = l.sku_id
    JOIN product p ON p.product_id = s.product_id
    JOIN product_line pl ON pl.product_line_id = p.product_line_id
    WHERE pl.code = 'POLEVKY' AND o.ordered_on BETWEEN DATE '2024-01-01' AND DATE '2025-12-31'
      AND (dp.store_id IS NOT NULL OR o.channel IN ('web', 'own_shops'))
    GROUP BY a.region_id, 2
),
weekly_temp AS (
    SELECT region_id, date_trunc('week', day)::date AS week, AVG(temp_mean_c) AS temp
    FROM weather_daily
    WHERE day BETWEEN DATE '2024-01-01' AND DATE '2025-12-31'
    GROUP BY region_id, 2
),
normalised AS (
    SELECT region_id, week, cases / AVG(cases) OVER (PARTITION BY region_id) AS cases FROM weekly_cases
)
SELECT corr(c.cases, t.temp) AS metric
FROM normalised c
JOIN weekly_temp t ON t.region_id = c.region_id AND t.week = c.week
