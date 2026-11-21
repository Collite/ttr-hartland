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
),
-- Use exact, common integer weights for comparisons and ties. The display division is the final step.
supported_world AS (
    SELECT COUNT(*) = 6 AND COUNT(DISTINCT k.region) = 6
        AND BOOL_AND(r.monthly_integrity = 'valid') AS valid
    FROM monthly_regions r
    LEFT JOIN skill k USING (region)
),
q_terms AS (
    SELECT
        row_number() OVER (ORDER BY r.region, r.decision_day) AS n,
        r.region,
        (r.decision_day + 1)::date AS month,
        round(r.q * denominator) AS numerator,
        denominator
    FROM round_quality r
    CROSS JOIN supported_world w
    CROSS JOIN LATERAL (SELECT 2 * r.chosen::numeric * r.unchosen AS denominator) d
    WHERE w.valid
),
q_denominator(n, common) AS (
    SELECT 0::bigint, 1::numeric
    UNION ALL
    SELECT t.n, lcm(d.common, t.denominator)
    FROM q_denominator d
    JOIN q_terms t ON t.n = d.n + 1
),
q_weight AS (
    SELECT t.region, t.month, numerator * div(common, denominator) AS weight, common
    FROM q_terms t
    CROSS JOIN (SELECT common FROM q_denominator ORDER BY n DESC LIMIT 1) d
),
omissions(omitted) AS (
    SELECT ARRAY[]::date[]
    UNION ALL
    SELECT ARRAY[m.month] FROM effective_calendar m
    UNION ALL
    SELECT ARRAY[a.month, b.month] FROM effective_calendar a JOIN effective_calendar b ON a.month < b.month
),
scores AS (
    SELECT o.omitted, q.region, SUM(q.weight) AS raw, COUNT(*) * MIN(q.common) AS total
    FROM omissions o
    CROSS JOIN q_weight q
    WHERE q.month <> ALL(o.omitted)
    GROUP BY o.omitted, q.region
),
ranks AS (
    SELECT
        s.*,
        2 * rank() OVER (PARTITION BY s.omitted ORDER BY s.raw)
            + count(*) OVER (PARTITION BY s.omitted, s.raw) - 1 - 7 AS x,
        2 * rank() OVER (PARTITION BY s.omitted ORDER BY k.pos_skill)
            + count(*) OVER (PARTITION BY s.omitted, k.pos_skill) - 1 - 7 AS y
    FROM scores s JOIN skill k USING (region)
),
metrics AS (
    SELECT
        omitted,
        json_object_agg(region, raw / total) AS regions,
        SUM(x * y) AS cov,
        SUM(x * x) * SUM(y * y) AS var,
        SUM(raw) FILTER (WHERE region IN ('PHA', 'JMV')) AS high,
        SUM(raw) FILTER (WHERE region IN ('ZSC', 'VCH', 'SMS')) AS low,
        MIN(total) AS total
    FROM ranks
    GROUP BY omitted
),
bars AS (
    SELECT
        *,
        CASE WHEN var = 0 THEN 0 ELSE cov / sqrt(var::double precision) END AS rank,
        high / (2 * total) AS q_high,
        low / (3 * total) AS q_low,
        (3 * high - 2 * low) / (6 * total) AS difference,
        var > 0 AND cov >= 0 AND cov * cov >= 0.80 * 0.80 * var AS rank_holds,
        high >= 0.70 * 2 * total AS q_high_holds,
        3 * high - 2 * low >= 0.10 * 6 * total AS difference_holds
    FROM metrics
),
cases AS (
    SELECT
        omitted, regions, rank, q_high, q_low, difference, rank_holds, q_high_holds, difference_holds,
        CASE WHEN rank_holds AND q_high_holds AND difference_holds THEN 'PASS' ELSE 'REJECTED' END AS status,
        NOT rank_holds AND NOT difference_holds AS control_rejects,
        CASE WHEN %s::text = 'story' THEN
            CASE WHEN NOT rank_holds THEN 'rank' WHEN NOT q_high_holds THEN 'q_high'
                WHEN NOT difference_holds THEN 'difference' END
        ELSE
            CASE WHEN rank_holds THEN 'rank' WHEN difference_holds THEN 'difference' END
        END AS failure
    FROM bars
)
SELECT
    (SELECT json_agg(to_jsonb(r) - 'integrity' - 'monthly_integrity'
        || jsonb_build_object('integrity', monthly_integrity)) FROM monthly_regions r) AS regions,
    (SELECT json_agg(round_quality) FROM round_quality) AS rounds,
    (SELECT json_agg(cases ORDER BY cardinality(omitted), omitted) FROM cases) AS cases,
    (SELECT json_build_object(
        'status', 'INCONCLUSIVE',
        'q_high', avg(quality) FILTER (WHERE region IN ('PHA', 'JMV')),
        'q_low', avg(quality) FILTER (WHERE region IN ('ZSC', 'VCH', 'SMS'))
    ) FROM monthly_regions) AS inconclusive
