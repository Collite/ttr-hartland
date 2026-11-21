-- Extend the NH-D25 rank statement's CTEs. Reuse the audited rows and economic disclosure.
-- Count coverage in the effective calendar month, one day after the decision.
effective_calendar AS (
    SELECT month::date AS month
    FROM evaluation e
    CROSS JOIN generate_series(e.first_day::timestamp, e.last_day::timestamp, INTERVAL '1 month') AS m (month)
),
month_counts AS (
    SELECT
        region_id,
        date_trunc('month', (decision_day + 1)::timestamp)::date AS month,
        COUNT(*) AS rounds,
        MIN(chosen) AS chosen,
        MIN(unchosen) AS unchosen,
        SUM(unchosen) AS total_unchosen
    FROM round_quality
    GROUP BY region_id, 2
),
monthly_support AS (
    SELECT
        r.region_id,
        COUNT(m.month) = (SELECT COUNT(*) FROM effective_calendar)
            AND COUNT(k.month) = (SELECT COUNT(*) FROM effective_calendar)
            AND COALESCE(BOOL_AND(m.rounds = 1 AND m.chosen >= 2 AND m.unchosen >= 2), false)
            AND COALESCE(SUM(m.total_unchosen), 0) >= 100 AS supported
    FROM region r
    LEFT JOIN month_counts m USING (region_id)
    LEFT JOIN effective_calendar k ON k.month = m.month
    GROUP BY r.region_id
),
monthly_regions AS (
    SELECT
        r.*,
        CASE
            WHEN r.integrity NOT IN ('valid', 'support') THEN r.integrity
            WHEN m.supported THEN 'valid'
            ELSE 'support'
        END AS monthly_integrity
    FROM region_quality r
    JOIN region k ON k.code = r.region
    JOIN monthly_support m ON m.region_id = k.region_id
)
SELECT
    (SELECT json_agg(to_jsonb(r) - 'integrity' - 'monthly_integrity'
        || jsonb_build_object('integrity', monthly_integrity)) FROM monthly_regions r) AS regions,
    (SELECT json_agg(round_quality) FROM round_quality) AS rounds
