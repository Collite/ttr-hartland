-- S5 in the pilot, the group claim (G2 S2.7; design08 §8): the mean gain ÷ cost of the high-skill regions (PHA, JMV)
-- over that of the weak-skill ones (ZSC, VCH, SMS), equal weights, the groups fixed before any build. NULL unless every
-- region is valid (the block's `integrity`), and NULL when the weak group's mean is not positive: no ratio then, and
-- the economic claim is rejected (`checks.s5_pilot` reports the sign). The block below is s5_rank.sql's, byte for
-- byte (a test keeps it so). S2.8 makes this s5_best_worst.sql, its historical name (design08 §9).
WITH
-- region return: begin
-- The pilot's observed regional return (design08 §6), `checks.s5_pilot.regions` its twin. The population is the
-- pilot revision's complete pairs: one programme and one holdout customer of one region, both with a baseline. W is
-- the assignment's window, observed to the last order day (`as_of`). A customer's outcome is its whole Customer Margin
-- in W (`sales_pl`, every line and outlet; a day without an order adds 0) less W's days × its stored baseline a day;
-- a region's gain is its programme customers' outcomes less its holdouts'. Its cost is its programme customers'
-- loans, each asset's cost over `life_days` for the loan's days in W from lent_on to the day before it came back (or
-- to as_of, so a placement pending past the last day costs nothing), and every give-away handed to them in W.
-- Coverage: of the loan cost on the region's assets, and of that plus the give-aways at small-retail customers of its
-- stores, the share held by customers of complete pairs. `integrity` is the first failure, the build's before the
-- region's: more than one window, a build short of W, a broken pair, a display or give-away for a small-retail
-- customer outside the programme, an asset out twice on one day, a loan of another region's asset, a region without
-- a pair, a cost not positive, a coverage below `coverage`; else valid.
pilot (revision, life_days, coverage) AS (
    VALUES ('s5-pilot-v1', 36 * 30.4, 0.90)
),
as_of AS (
    SELECT MAX(ordered_on) AS day FROM sales_order
),
assigned AS (
    SELECT
        a.customer_id,
        a.region_id,
        a.pair_id,
        a.arm,
        a.outside_reason,
        a.assigned_on,
        a.window_from,
        a.window_until,
        a.baseline_customer_margin,
        a.baseline_days
    FROM pos_pilot_assignment a
    JOIN pilot ON pilot.revision = a.pilot_revision
),
w AS (
    SELECT
        MIN(window_from) AS first_day,
        MIN(window_until) AS last_day,
        MIN(assigned_on) AS assigned_on,
        COUNT(DISTINCT window_from) = 1 AND COUNT(DISTINCT window_until) = 1 AND COUNT(DISTINCT assigned_on) = 1
            AS one
    FROM assigned
),
pair AS (
    SELECT pair_id, MIN(region_id) AS region_id
    FROM assigned
    WHERE pair_id IS NOT NULL
    GROUP BY pair_id
    HAVING COUNT(*) = 2
        AND COUNT(*) FILTER (WHERE arm = 'programme') = 1
        AND COUNT(*) FILTER (WHERE arm = 'holdout') = 1
        AND COUNT(DISTINCT region_id) = 1
        AND COUNT(baseline_customer_margin) = 2
        AND COUNT(baseline_days) = 2
        AND MIN(baseline_days) > 0
),
paired AS (
    SELECT a.customer_id, a.arm, p.region_id, a.baseline_customer_margin, a.baseline_days
    FROM assigned a
    JOIN pair p ON p.pair_id = a.pair_id
),
outcome AS (
    SELECT
        p.customer_id,
        p.arm,
        p.region_id,
        COALESCE(SUM(s.customer_margin), 0) AS margin,
        (w.last_day - w.first_day + 1) * p.baseline_customer_margin / p.baseline_days AS expected
    FROM paired p
    CROSS JOIN w
    LEFT JOIN sales_pl s ON s.customer_id = p.customer_id AND s.ordered_on BETWEEN w.first_day AND w.last_day
    GROUP BY p.customer_id, p.arm, p.region_id, p.baseline_customer_margin, p.baseline_days, w.first_day, w.last_day
),
loan AS (
    SELECT
        l.pos_loan_id,
        l.pos_asset_id,
        l.customer_id,
        pa.region_id,
        l.lent_on,
        GREATEST(l.lent_on, w.first_day) AS first_day,
        LEAST(COALESCE(l.returned_on - 1, as_of.day), as_of.day, w.last_day) AS last_day,
        pa.cost
    FROM pos_loan l
    JOIN pos_asset pa ON pa.pos_asset_id = l.pos_asset_id
    CROSS JOIN w
    CROSS JOIN as_of
),
used AS (
    SELECT
        loan.*,
        GREATEST(loan.last_day - loan.first_day + 1, 0) AS days,
        loan.cost * GREATEST(loan.last_day - loan.first_day + 1, 0) / pilot.life_days AS asset_cost
    FROM loan
    CROSS JOIN pilot
),
handed AS (
    SELECT v.customer_id, ar.region_id, h.cost
    FROM pos_handout h
    JOIN visit v ON v.visit_id = h.visit_id
    JOIN customer c ON c.customer_id = v.customer_id
    JOIN store st ON st.store_id = v.store_id
    JOIN area ar ON ar.area_id = st.area_id
    CROSS JOIN w
    WHERE c.channel = 'small_retail' AND v.day BETWEEN w.first_day AND w.last_day
),
fault AS (
    SELECT
        (
            SELECT COUNT(*)
            FROM assigned a
            WHERE (a.arm IN ('programme', 'holdout') OR a.pair_id IS NOT NULL)
                AND a.customer_id NOT IN (SELECT customer_id FROM paired)
        ) AS pairs,
        (
            SELECT COUNT(*)
            FROM (
                SELECT used.customer_id
                FROM used
                CROSS JOIN w
                WHERE used.lent_on BETWEEN w.assigned_on AND w.last_day
                UNION ALL
                SELECT handed.customer_id
                FROM handed
            ) treated
            JOIN customer c ON c.customer_id = treated.customer_id
            LEFT JOIN assigned a ON a.customer_id = treated.customer_id
            WHERE c.channel = 'small_retail' AND a.arm IS DISTINCT FROM 'programme'
        ) AS contamination,
        (
            SELECT COUNT(*)
            FROM used x
            JOIN used y ON y.pos_asset_id = x.pos_asset_id AND y.pos_loan_id > x.pos_loan_id
            WHERE x.days > 0 AND y.days > 0 AND x.first_day <= y.last_day AND y.first_day <= x.last_day
        ) AS overlap,
        (
            SELECT COUNT(*)
            FROM used
            JOIN assigned a ON a.customer_id = used.customer_id
            WHERE used.days > 0 AND used.region_id <> a.region_id
        ) AS astray
),
gains AS (
    SELECT
        region_id,
        COUNT(*) FILTER (WHERE arm = 'programme') AS pairs,
        SUM(margin) FILTER (WHERE arm = 'programme') AS programme_margin,
        SUM(margin) FILTER (WHERE arm = 'holdout') AS holdout_margin,
        SUM(expected) FILTER (WHERE arm = 'programme') - SUM(expected) FILTER (WHERE arm = 'holdout')
            AS baseline_adjustment
    FROM outcome
    GROUP BY region_id
),
displays AS (
    SELECT p.region_id, SUM(u.asset_cost) AS asset_cost
    FROM used u
    JOIN paired p ON p.customer_id = u.customer_id AND p.arm = 'programme'
    GROUP BY p.region_id
),
giveaways AS (
    SELECT p.region_id, SUM(h.cost) AS handout_cost
    FROM handed h
    JOIN paired p ON p.customer_id = h.customer_id AND p.arm = 'programme'
    GROUP BY p.region_id
),
held AS (
    SELECT
        u.region_id,
        SUM(u.asset_cost) AS loans,
        COALESCE(SUM(u.asset_cost) FILTER (WHERE p.customer_id IS NOT NULL), 0) AS loans_paired
    FROM used u
    LEFT JOIN paired p ON p.customer_id = u.customer_id
    GROUP BY u.region_id
),
handed_out AS (
    SELECT
        h.region_id,
        SUM(h.cost) AS handouts,
        COALESCE(SUM(h.cost) FILTER (WHERE p.customer_id IS NOT NULL), 0) AS handouts_paired
    FROM handed h
    LEFT JOIN paired p ON p.customer_id = h.customer_id
    GROUP BY h.region_id
),
cohort AS (
    SELECT
        a.region_id,
        COUNT(*) FILTER (WHERE a.outside_reason = 'unpaired') AS unpaired,
        COUNT(*) FILTER (WHERE a.outside_reason = 'cross_region') AS cross_region,
        COUNT(*) FILTER (WHERE p.customer_id IS NOT NULL AND (c.since IS NULL OR c.since >= a.assigned_on))
            AS potential_at_entry,
        COUNT(*) FILTER (WHERE p.customer_id IS NOT NULL AND c.until <= a.window_until) AS lost
    FROM assigned a
    LEFT JOIN paired p ON p.customer_id = a.customer_id
    LEFT JOIN customer c ON c.customer_id = a.customer_id
    GROUP BY a.region_id
),
region_sums AS (
    SELECT
        r.code AS region,
        COALESCE(g.pairs, 0) AS pairs,
        g.programme_margin,
        g.holdout_margin,
        g.baseline_adjustment,
        g.programme_margin - g.holdout_margin - g.baseline_adjustment AS gain,
        COALESCE(d.asset_cost, 0) AS asset_cost,
        COALESCE(gv.handout_cost, 0) AS handout_cost,
        COALESCE(d.asset_cost, 0) + COALESCE(gv.handout_cost, 0) AS cost,
        hl.loans_paired / NULLIF(hl.loans, 0) AS loan_coverage,
        (COALESCE(hl.loans_paired, 0) + COALESCE(ho.handouts_paired, 0))
            / NULLIF(COALESCE(hl.loans, 0) + COALESCE(ho.handouts, 0), 0) AS cost_coverage,
        COALESCE(k.unpaired, 0) AS unpaired,
        COALESCE(k.cross_region, 0) AS cross_region,
        COALESCE(k.potential_at_entry, 0) AS potential_at_entry,
        COALESCE(k.lost, 0) AS lost
    FROM region r
    LEFT JOIN gains g ON g.region_id = r.region_id
    LEFT JOIN displays d ON d.region_id = r.region_id
    LEFT JOIN giveaways gv ON gv.region_id = r.region_id
    LEFT JOIN held hl ON hl.region_id = r.region_id
    LEFT JOIN handed_out ho ON ho.region_id = r.region_id
    LEFT JOIN cohort k ON k.region_id = r.region_id
),
region_return AS (
    SELECT
        s.*,
        CASE WHEN s.cost > 0 THEN s.gain / s.cost END AS gain_per_cost,
        CASE WHEN s.cost > 0 THEN s.gain / s.cost - 1 END AS net_return,
        CASE
            WHEN NOT w.one THEN 'window'
            WHEN as_of.day IS NULL OR as_of.day < w.last_day THEN 'short'
            WHEN f.pairs > 0 THEN 'pairs'
            WHEN f.contamination > 0 THEN 'contamination'
            WHEN f.overlap > 0 THEN 'overlap'
            WHEN f.astray > 0 THEN 'astray'
            WHEN MIN(s.pairs) OVER () = 0 THEN 'regions'
            WHEN NOT s.cost > 0 THEN 'cost'
            WHEN NOT COALESCE(s.loan_coverage >= pilot.coverage AND s.cost_coverage >= pilot.coverage, false)
                THEN 'coverage'
            ELSE 'valid'
        END AS integrity
    FROM region_sums s
    CROSS JOIN w
    CROSS JOIN as_of
    CROSS JOIN fault f
    CROSS JOIN pilot
)
-- region return: end
SELECT CASE WHEN valid AND e_low > 0 THEN e_high / e_low END AS metric
FROM (
    SELECT
        COUNT(*) = 6 AND BOOL_AND(integrity = 'valid') AS valid,
        AVG(gain_per_cost) FILTER (WHERE region IN ('PHA', 'JMV')) AS e_high,
        AVG(gain_per_cost) FILTER (WHERE region IN ('ZSC', 'VCH', 'SMS')) AS e_low
    FROM region_return
) claim
