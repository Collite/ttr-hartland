-- S3 — the reps differ (design 04 §S3). Per rep — the person, across the area moves, over the history — the four
-- observed rates: conversion (sales visits at a customer that took an order ÷ those visits), acquisition (customers
-- won in the rep's area while assigned ÷ the days assigned), churn (customers lost likewise) and cross-sell (customer
-- × lines bought for the first time likewise). The composite is the mean of the four ranks (churn reversed). The
-- metric counts the pinned placements that hold: the two stars in the top five, the two weak reps in the bottom five
-- — 4 with the story on. The pins are anomalies.yaml's, by the area of the rep's first assignment (a test keeps the
-- codes below in step).
WITH rep AS (
    SELECT DISTINCT ON (a.employee_id) a.employee_id, ar.code AS slot
    FROM assignment a
    JOIN area ar ON ar.area_id = a.area_id
    WHERE a.kind = 'rep_area'
    ORDER BY a.employee_id, a.valid_from
),
pinned (slot, profile) AS (
    VALUES ('PHA-03', 'star'), ('JMV-04', 'star'), ('ZSC-05', 'weak'), ('SMS-07', 'weak')
),
duty AS (
    SELECT a.employee_id, a.area_id, GREATEST(a.valid_from, DATE '2024-01-01') AS from_day, a.valid_to AS to_day
    FROM assignment a
    WHERE a.kind = 'rep_area' AND a.valid_to >= DATE '2024-01-01'
),
exposure AS (
    SELECT employee_id, SUM(LEAST(to_day, DATE '2026-10-31') - from_day + 1) AS days FROM duty GROUP BY employee_id
),
conversion AS (
    SELECT v.employee_id, AVG(CASE WHEN v.outcome = 'order' THEN 1.0 ELSE 0.0 END) AS rate
    FROM visit v
    WHERE v.kind = 'sales' AND v.customer_id IS NOT NULL AND v.day >= DATE '2024-01-01'
    GROUP BY v.employee_id
),
won AS (
    SELECT d.employee_id, COUNT(*) AS n
    FROM customer c
    JOIN store st ON st.store_id = c.store_id
    JOIN duty d ON d.area_id = st.area_id AND c.since BETWEEN d.from_day AND d.to_day
    GROUP BY d.employee_id
),
lost AS (
    SELECT d.employee_id, COUNT(*) AS n
    FROM customer c
    JOIN store st ON st.store_id = c.store_id
    JOIN duty d ON d.area_id = st.area_id AND c.until BETWEEN d.from_day AND d.to_day
    GROUP BY d.employee_id
),
first_line AS (
    SELECT o.customer_id, p.product_line_id, MIN(o.ordered_on) AS first_on
    FROM sales_order o
    JOIN sales_order_line l ON l.sales_order_id = o.sales_order_id
    JOIN sku s ON s.sku_id = l.sku_id
    JOIN product p ON p.product_id = s.product_id
    WHERE o.channel = 'small_retail' AND NOT l.is_free_goods
    GROUP BY o.customer_id, p.product_line_id
),
added AS (
    SELECT d.employee_id, COUNT(*) AS n
    FROM first_line f
    JOIN customer c ON c.customer_id = f.customer_id
    JOIN store st ON st.store_id = c.store_id
    JOIN duty d ON d.area_id = st.area_id AND f.first_on BETWEEN d.from_day AND d.to_day
    WHERE f.first_on > c.since + 90
    GROUP BY d.employee_id
),
rates AS (
    SELECT
        rep.employee_id,
        rep.slot,
        COALESCE(conversion.rate, 0) AS conversion,
        COALESCE(won.n, 0)::numeric / exposure.days AS acquisition,
        COALESCE(lost.n, 0)::numeric / exposure.days AS churn,
        COALESCE(added.n, 0)::numeric / exposure.days AS cross_sell
    FROM rep
    JOIN exposure ON exposure.employee_id = rep.employee_id
    LEFT JOIN conversion ON conversion.employee_id = rep.employee_id
    LEFT JOIN won ON won.employee_id = rep.employee_id
    LEFT JOIN lost ON lost.employee_id = rep.employee_id
    LEFT JOIN added ON added.employee_id = rep.employee_id
),
ranked AS (
    SELECT
        slot,
        RANK() OVER (ORDER BY conversion DESC) + RANK() OVER (ORDER BY acquisition DESC)
            + RANK() OVER (ORDER BY churn) + RANK() OVER (ORDER BY cross_sell DESC) AS score
    FROM rates
),
composite AS (
    SELECT slot, RANK() OVER (ORDER BY score) AS place, COUNT(*) OVER () AS reps FROM ranked
)
SELECT COUNT(*) FILTER (
    WHERE (p.profile = 'star' AND c.place <= 5) OR (p.profile = 'weak' AND c.place > c.reps - 5)
) AS metric
FROM pinned p
JOIN composite c ON c.slot = p.slot
