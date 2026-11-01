-- S3 — the reps differ (design 04 §S3). Per rep — the person, across the area moves, from 2024-01-01 to the world's
-- last order day — the four observed rates of small retail, each over its base:
--   conversion  = sales visits at a customer that took an order ÷ those visits;
--   acquisition = customers won in the rep's area while assigned ÷ the potential store-days there: the area's open
--                 small-retail stores (independents, franchises and station shops; no key-account chain, no HoReCa)
--                 × the days, less the days they were customers;
--   churn       = customers lost likewise ÷ the active customer-days there;
--   cross-sell  = customer × lines bought for the first time likewise ÷ the active customer-days. A line counts only
--                 when it comes more than 90 days after the customer's start or the history's (2024-01-01), whichever
--                 is later, so the first orders of the history are not "new" lines.
-- The composite is the mean of the four ranks (churn reversed). The metric counts the pinned placements that hold: the
-- two stars in the top five, the two weak reps in the bottom five — 4 with the story on. The pins are anomalies.yaml's,
-- by the area the rep holds on the warm-up's first day, 2023-01-01 (a test keeps the codes below in step).
WITH as_of AS (
    SELECT MAX(ordered_on) AS day FROM sales_order
),
slot AS (
    SELECT a.employee_id, ar.code AS slot
    FROM assignment a
    JOIN area ar ON ar.area_id = a.area_id
    WHERE a.kind = 'rep_area' AND DATE '2023-01-01' BETWEEN a.valid_from AND a.valid_to
),
pinned (slot, profile) AS (
    VALUES ('PHA-03', 'star'), ('JMV-04', 'star'), ('ZSC-05', 'weak'), ('SMS-07', 'weak')
),
duty AS (
    SELECT
        a.employee_id,
        a.area_id,
        GREATEST(a.valid_from, DATE '2024-01-01') AS from_day,
        LEAST(a.valid_to, as_of.day) AS to_day
    FROM assignment a
    CROSS JOIN as_of
    WHERE a.kind = 'rep_area' AND a.valid_to >= DATE '2024-01-01' AND a.valid_from <= as_of.day
),
market AS (
    SELECT st.store_id, st.area_id, st.opened_on, st.closed_on
    FROM store st
    LEFT JOIN chain ch ON ch.chain_id = st.chain_id
    WHERE st.store_class <> 'horeca' AND NOT COALESCE(ch.is_key_account, false)
),
patron AS (
    SELECT c.customer_id, c.since, c.until, m.area_id
    FROM customer c
    JOIN market m ON m.store_id = c.store_id
    WHERE c.channel = 'small_retail' AND c.since IS NOT NULL
),
store_days AS (
    SELECT d.employee_id, SUM(GREATEST(
        LEAST(d.to_day, COALESCE(m.closed_on, d.to_day)) - GREATEST(d.from_day, COALESCE(m.opened_on, d.from_day)) + 1,
        0
    )) AS days
    FROM duty d
    JOIN market m ON m.area_id = d.area_id
    GROUP BY d.employee_id
),
active_days AS (
    SELECT d.employee_id, SUM(GREATEST(
        LEAST(d.to_day, COALESCE(p.until, d.to_day)) - GREATEST(d.from_day, p.since) + 1, 0
    )) AS days
    FROM duty d
    JOIN patron p ON p.area_id = d.area_id
    GROUP BY d.employee_id
),
conversion AS (
    SELECT v.employee_id, AVG(CASE WHEN v.outcome = 'order' THEN 1.0 ELSE 0.0 END) AS rate
    FROM visit v
    CROSS JOIN as_of
    WHERE v.kind = 'sales' AND v.customer_id IS NOT NULL AND v.day BETWEEN DATE '2024-01-01' AND as_of.day
    GROUP BY v.employee_id
),
won AS (
    SELECT d.employee_id, COUNT(*) AS n
    FROM patron p
    JOIN duty d ON d.area_id = p.area_id AND p.since BETWEEN d.from_day AND d.to_day
    GROUP BY d.employee_id
),
lost AS (
    SELECT d.employee_id, COUNT(*) AS n
    FROM patron p
    JOIN duty d ON d.area_id = p.area_id AND p.until BETWEEN d.from_day AND d.to_day
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
    JOIN patron p ON p.customer_id = f.customer_id
    JOIN duty d ON d.area_id = p.area_id AND f.first_on BETWEEN d.from_day AND d.to_day
    WHERE f.first_on > GREATEST(p.since, DATE '2024-01-01') + 90
    GROUP BY d.employee_id
),
rates AS (
    SELECT
        r.employee_id,
        COALESCE(conversion.rate, 0) AS conversion,
        COALESCE(won.n, 0)::numeric / NULLIF(sd.days - COALESCE(ad.days, 0), 0) AS acquisition,
        COALESCE(lost.n, 0)::numeric / NULLIF(ad.days, 0) AS churn,
        COALESCE(added.n, 0)::numeric / NULLIF(ad.days, 0) AS cross_sell
    FROM (SELECT DISTINCT employee_id FROM duty) r
    LEFT JOIN store_days sd ON sd.employee_id = r.employee_id
    LEFT JOIN active_days ad ON ad.employee_id = r.employee_id
    LEFT JOIN conversion ON conversion.employee_id = r.employee_id
    LEFT JOIN won ON won.employee_id = r.employee_id
    LEFT JOIN lost ON lost.employee_id = r.employee_id
    LEFT JOIN added ON added.employee_id = r.employee_id
),
ranked AS (
    SELECT
        employee_id,
        RANK() OVER (ORDER BY conversion DESC NULLS LAST) + RANK() OVER (ORDER BY acquisition DESC NULLS LAST)
            + RANK() OVER (ORDER BY churn NULLS LAST) + RANK() OVER (ORDER BY cross_sell DESC NULLS LAST) AS score
    FROM rates
),
composite AS (
    SELECT employee_id, RANK() OVER (ORDER BY score) AS place, COUNT(*) OVER () AS reps FROM ranked
)
SELECT COUNT(*) FILTER (
    WHERE (p.profile = 'star' AND c.place <= 5) OR (p.profile = 'weak' AND c.place > c.reps - 5)
) AS metric
FROM pinned p
JOIN slot s ON s.slot = p.slot
JOIN composite c ON c.employee_id = s.employee_id
