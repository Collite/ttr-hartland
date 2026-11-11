-- A-6 — the competitor's winter promotion, end to end (G2 S2.3 T7): Kavalír's promotion seen at visits in East Bohemia
-- (region VCH) from week 45 to week 50 of 2025 — only the `promotion` kind: Kavalír is strong in East Bohemia, so its
-- listings, displays and price cuts are seen there all year — the competitors' facings beside Gold in the region's
-- observations up from week 45, and the region's Gold down in weeks 46–50. The count is the sightings, and none unless
-- the other two show: the facings of weeks 45–50 over those of the 8 weeks before at least 1.25 (half the planted
-- +50 %), and the dip — the region's customers' Gold cases a day in weeks 46–50 over the same customers' in the 8 weeks
-- before, ÷ that ratio of the other regions' customers (the season's) — at most 0.94 (half the planted −12 %).
WITH sighting AS (
    SELECT COUNT(*) AS n
    FROM competitor_sighting cs
    JOIN competitor k ON k.competitor_id = cs.competitor_id
    JOIN visit v ON v.visit_id = cs.visit_id
    JOIN store st ON st.store_id = v.store_id
    JOIN area a ON a.area_id = st.area_id
    JOIN region r ON r.region_id = a.region_id
    WHERE k.code = 'KAVALIR' AND cs.activity_kind = 'promotion' AND r.code = 'VCH'
        AND v.day BETWEEN DATE '2025-11-03' AND DATE '2025-12-14'
),
facings AS (
    SELECT
        AVG(o.competitor_facings) FILTER (WHERE v.day BETWEEN DATE '2025-11-03' AND DATE '2025-12-14')
            / NULLIF(AVG(o.competitor_facings) FILTER (WHERE v.day BETWEEN DATE '2025-09-08' AND DATE '2025-11-02'), 0)
            AS lift
    FROM visit_observation o
    JOIN product_line pl ON pl.product_line_id = o.product_line_id
    JOIN visit v ON v.visit_id = o.visit_id
    JOIN store st ON st.store_id = v.store_id
    JOIN area a ON a.area_id = st.area_id
    JOIN region r ON r.region_id = a.region_id
    WHERE pl.code = 'GOLD' AND r.code = 'VCH'
),
gold AS (
    SELECT
        r.code = 'VCH' AS east,
        s.customer_id,
        COALESCE(SUM(s.cases) FILTER (WHERE s.ordered_on >= DATE '2025-11-10'), 0) / 35.0 AS during,
        SUM(s.cases) FILTER (WHERE s.ordered_on < DATE '2025-11-10') / 56.0 AS before
    FROM sales_pl s
    JOIN product_line pl ON pl.product_line_id = s.product_line_id
    JOIN area a ON a.area_id = s.area_id
    JOIN region r ON r.region_id = a.region_id
    WHERE pl.code = 'GOLD' AND s.ordered_on BETWEEN DATE '2025-09-15' AND DATE '2025-12-14'
    GROUP BY r.code = 'VCH', s.customer_id
),
dip AS (
    SELECT
        (SUM(during) FILTER (WHERE east) / NULLIF(SUM(before) FILTER (WHERE east), 0))
            / NULLIF(SUM(during) FILTER (WHERE NOT east) / NULLIF(SUM(before) FILTER (WHERE NOT east), 0), 0)
            AS ratio
    FROM gold
    WHERE before > 0
)
SELECT CASE WHEN f.lift >= 1.25 AND d.ratio <= 0.94 THEN s.n ELSE 0 END AS metric
FROM sighting s
CROSS JOIN facings f
CROSS JOIN dip d
