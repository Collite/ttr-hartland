-- NH-D28/v1 extends the audited monthly CTEs through the per-region scores.
-- Each sum below has the same regional denominator. Compare the group bars before division.
group_totals AS (
    SELECT
        omitted,
        json_object_agg(region, raw / total) AS regions,
        SUM(raw) FILTER (WHERE region IN ('PHA', 'JMV')) AS high,
        SUM(raw) FILTER (WHERE region IN ('ZSC', 'VCH', 'SMS')) AS low,
        MIN(total) AS total
    FROM scores
    GROUP BY omitted
),
group_bars AS (
    SELECT
        omitted, regions,
        high / (2 * total) AS q_high,
        low / (3 * total) AS q_low,
        (3 * high - 2 * low) / (6 * total) AS difference,
        high >= 0.70 * 2 * total AS q_high_holds,
        3 * high - 2 * low >= 0.10 * 6 * total AS difference_holds
    FROM group_totals
),
group_mode AS (
    SELECT %s::text AS mode
),
cases AS (
    SELECT
        b.*,
        CASE
            WHEN mode = 'story' AND q_high_holds AND difference_holds THEN 'PASS'
            WHEN mode IN ('skill-neutral', 'full-null') AND difference_holds THEN 'FAIL'
            ELSE 'REJECTED'
        END AS status,
        mode IN ('skill-neutral', 'full-null') AND NOT difference_holds AS control_rejects,
        array_remove(ARRAY[
            CASE WHEN mode = 'story' AND NOT q_high_holds THEN 'q_high' END,
            CASE WHEN (mode = 'story' AND NOT difference_holds)
                OR (mode IN ('skill-neutral', 'full-null') AND difference_holds) THEN 'difference' END
        ], NULL) AS failures
    FROM group_bars b
    CROSS JOIN group_mode
)
