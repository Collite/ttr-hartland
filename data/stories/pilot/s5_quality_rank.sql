-- S5 in the pilot, the rank (G2 S2.9; NH-D25 design09 §§3, 5): Spearman's rho between the six regions' decision quality
-- Q and their area managers' declared pos_skill, on average ranks (VCH and SMS share theirs). The labels are s5.yaml's
-- story, fixed and read only here, after Q: a control world is ranked against them too, never against its own. Six
-- equal Q are no association, 0. NULL unless every region is valid (the block's `integrity`): an incomplete or
-- unsupported build is never a pass. Forecast-based placement quality only: no measured gain or payback. S2.11 makes
-- this s5.sql (design09 §7).
WITH skill (region, pos_skill) AS (
    VALUES ('PHA', 1.4), ('JMV', 1.3), ('SJC', 1.0), ('ZSC', 0.6), ('VCH', 0.5), ('SMS', 0.5)
),
-- region quality: begin
-- The observed quality of the pilot's placement decisions and its economic disclosure (design09 §§3–4), its twin
-- `checks.s5_decision.regions`. A round is a region × a month-end decision day whose placements are effective the
-- next day, inside W: the 24 month-ends 2023-12-31 … 2025-11-30, each reported, empty or not (`round_quality`). Its
-- pool is every decision of the round with a forecast material and manager and a positive forecast gain, cost and
-- score, whatever its outcome — a no_budget or no_asset candidate the draw reached stays in it; the initial no_asset,
-- nonpositive_score and ineligible ones are counted apart. Chosen: selected or cancelled (a cancellation keeps its
-- score); the rest of the pool is unchosen. q = the share of (chosen, unchosen) pairs whose chosen score is the
-- greater, a tie half — from average ranks, (2 × the chosen's rank sum − k(k + 1)) ÷ (2k(n − k)) — and NULL without
-- a chosen or an unchosen one; a region's Q is the mean q of its comparable rounds, its chosen and unchosen counted
-- over those rounds. The score is the stored one, checked against its stored gain ÷ cost and theirs against their
-- stored inputs (each within half its last digit: the generator rounds floats). No skill, uplift or replay is read.
-- The disclosure, per region of the programme customers' assignment, in W to the last order day (`as_of`): their
-- whole Customer Margin, observed and unattributed (`sales_pl`); their displays' use — each loan's asset cost over
-- `life_days` for its days from lent_on to the day before it came back (a warm-up loan's from W's first day, a loan
-- pending past the last day none); every give-away handed to them; the customers without an order line and their
-- cost; the fleet — every asset of the region, carried from acquisition to the day before its write-off — its cost,
-- the part no loan used and the assets never lent in W.
-- `integrity` is the first failure, the build's before the region's: W not the assignment's one window, a build short
-- of W, a row of another revision, a duplicate decision key or id or a loan chosen twice, an unknown outcome, an
-- impossible decision (fields, rounding, a day not a month-end), a decision not linked to its programme customer,
-- region, delivery point, manager, material or loan, a loan since the assignment without its selection or a
-- cancellation with a loan, a cost of a missing or non-programme small-retail customer, a loan without its asset, a
-- give-away without its visit, an asset out twice on one day, a loan of another region's asset, a region without
-- programme customers; then a region short of `least_rounds` comparable rounds or of `least_chosen` and
-- `least_unchosen` decisions over them; else valid.
evaluation (
    pilot_revision, policy_revision, first_day, last_day, baseline_days, term_days, forecast_days, life_days,
    least_rounds, least_chosen, least_unchosen
) AS (
    VALUES ('s5-pilot-v1', 'nh-d21-b-r1', DATE '2024-01-01', DATE '2025-12-31', 56, 182.625, 183, 36 * 30.4, 12, 100, 100)
),
as_of AS (
    SELECT MAX(ordered_on) AS day FROM sales_order
),
assigned AS (
    SELECT a.customer_id, a.region_id, a.arm, a.assigned_on, a.window_from, a.window_until
    FROM pos_pilot_assignment a
    JOIN evaluation e ON e.pilot_revision = a.pilot_revision
),
w AS (
    SELECT
        COUNT(DISTINCT a.window_from) = 1 AND COUNT(DISTINCT a.window_until) = 1
            AND MIN(a.window_from) = MIN(e.first_day) AND MIN(a.window_until) = MIN(e.last_day) AS one,
        MIN(a.assigned_on) AS assigned_on
    FROM assigned a
    CROSS JOIN evaluation e
),
programme AS (
    SELECT customer_id, region_id FROM assigned WHERE arm = 'programme'
),
decided AS (
    SELECT
        d.pos_placement_decision_id,
        d.pilot_revision,
        d.policy_revision,
        d.decision_day,
        d.customer_id,
        d.delivery_point_id,
        d.region_id,
        d.manager_id,
        d.forecast_material_id,
        d.outcome,
        d.loan_id,
        d.score,
        d.forecast_gain > 0 AND d.forecast_cost > 0 AND d.score > 0 AND d.forecast_material_id IS NOT NULL
            AND d.manager_id IS NOT NULL AS viable,
        d.decision_day + 1 BETWEEN e.first_day AND e.last_day AS inside,
        COALESCE(
            CASE
                WHEN d.outcome IN ('selected', 'cancelled', 'no_budget')
                    THEN d.score > 0 AND d.manager_id IS NOT NULL AND (d.loan_id IS NOT NULL) = (d.outcome = 'selected')
                WHEN d.outcome = 'no_asset'
                    THEN d.loan_id IS NULL AND d.manager_id IS NOT NULL
                        AND (d.score > 0 OR d.score = 0 AND d.forecast_material_id IS NULL)
                WHEN d.outcome = 'nonpositive_score'
                    THEN d.loan_id IS NULL AND d.manager_id IS NOT NULL AND d.score = 0 AND d.forecast_gain = 0
                        AND d.forecast_cost > 0
                WHEN d.outcome = 'ineligible'
                    THEN d.loan_id IS NULL AND d.score = 0 AND (d.manager_id IS NULL OR NOT d.forecast_cost > 0)
            END
            AND (d.forecast_material_id IS NULL) = (d.forecast_gain IS NULL)
            AND (d.forecast_gain IS NULL) = (d.forecast_cost IS NULL)
            AND (d.forecast_cost IS NULL) = (d.asset_cost IS NULL)
            AND d.score >= 0
            AND (d.score = 0 OR ABS(d.score - d.forecast_gain / d.forecast_cost) <= 0.0000000050001)
            AND (
                d.forecast_material_id IS NULL
                OR ABS(d.forecast_gain - GREATEST(d.prior56_line_margin / e.baseline_days, 0) * d.catalogue_fit * e.term_days)
                    <= 0.005000001
                AND ABS(
                    d.forecast_cost
                    - (d.asset_cost * e.term_days / e.life_days + d.prior56_handout_cost / e.baseline_days * e.term_days)
                ) <= 0.005000001
            )
            AND d.forecast_days = e.forecast_days
            AND EXTRACT(DAY FROM d.decision_day + 1) = 1,
            false
        ) AS sound
    FROM pos_placement_decision d
    CROSS JOIN evaluation e
),
calendar AS (
    SELECT r.region_id, r.code AS region, m.month::date - 1 AS decision_day
    FROM region r
    CROSS JOIN evaluation e
    CROSS JOIN generate_series(e.first_day::timestamp, e.last_day::timestamp, INTERVAL '1 month') AS m (month)
),
ranked AS (
    SELECT
        region_id,
        decision_day,
        outcome IN ('selected', 'cancelled') AS chosen,
        RANK() OVER (PARTITION BY region_id, decision_day ORDER BY score) AS low,
        COUNT(*) OVER (PARTITION BY region_id, decision_day, score) AS ties
    FROM decided
    WHERE inside AND viable
),
pooled AS (
    SELECT
        region_id,
        decision_day,
        COUNT(*) AS viable,
        COUNT(*) FILTER (WHERE chosen) AS chosen,
        SUM(2 * low + ties - 1) FILTER (WHERE chosen) AS twice_ranks
    FROM ranked
    GROUP BY region_id, decision_day
),
counted AS (
    SELECT
        region_id,
        decision_day,
        COUNT(*) AS decisions,
        COUNT(*) FILTER (WHERE outcome = 'selected') AS selected,
        COUNT(*) FILTER (WHERE outcome = 'cancelled') AS cancelled,
        COUNT(*) FILTER (WHERE outcome = 'no_budget') AS no_budget,
        COUNT(*) FILTER (WHERE outcome = 'no_asset' AND score > 0) AS no_asset,
        COUNT(*) FILTER (WHERE outcome = 'no_asset' AND NOT score > 0) AS no_asset_initial,
        COUNT(*) FILTER (WHERE outcome = 'nonpositive_score') AS nonpositive_score,
        COUNT(*) FILTER (WHERE outcome = 'ineligible') AS ineligible
    FROM decided
    WHERE inside
    GROUP BY region_id, decision_day
),
round_quality AS (
    SELECT
        k.region_id,
        k.region,
        k.decision_day,
        COALESCE(c.decisions, 0) AS decisions,
        COALESCE(p.viable, 0) AS viable,
        COALESCE(p.chosen, 0) AS chosen,
        COALESCE(p.viable, 0) - COALESCE(p.chosen, 0) AS unchosen,
        CASE
            WHEN p.chosen > 0 AND p.viable > p.chosen
                THEN (p.twice_ranks - p.chosen * (p.chosen + 1)) / (2.0 * p.chosen * (p.viable - p.chosen))
        END AS q
    FROM calendar k
    LEFT JOIN pooled p ON p.region_id = k.region_id AND p.decision_day = k.decision_day
    LEFT JOIN counted c ON c.region_id = k.region_id AND c.decision_day = k.decision_day
),
loan AS (
    SELECT
        l.pos_loan_id,
        l.pos_asset_id,
        l.customer_id,
        l.lent_on,
        pa.pos_asset_id IS NOT NULL AS has_asset,
        pa.region_id AS asset_region_id,
        GREATEST(l.lent_on, e.first_day) AS first_day,
        LEAST(COALESCE(l.returned_on - 1, as_of.day), as_of.day, e.last_day) AS last_day,
        pa.cost
    FROM pos_loan l
    LEFT JOIN pos_asset pa ON pa.pos_asset_id = l.pos_asset_id
    CROSS JOIN evaluation e
    CROSS JOIN as_of
),
used AS (
    SELECT
        loan.*,
        GREATEST(loan.last_day - loan.first_day + 1, 0) AS days,
        COALESCE(loan.cost, 0) * GREATEST(loan.last_day - loan.first_day + 1, 0) / e.life_days AS use_cost
    FROM loan
    CROSS JOIN evaluation e
),
handed AS (
    SELECT h.pos_handout_id, h.cost, v.visit_id IS NOT NULL AS has_visit, v.customer_id, v.day
    FROM pos_handout h
    LEFT JOIN visit v ON v.visit_id = h.visit_id
),
spending AS (
    SELECT u.customer_id, u.days, u.use_cost AS cost FROM used u WHERE u.days > 0
    UNION ALL
    SELECT h.customer_id, 1, h.cost
    FROM handed h
    CROSS JOIN evaluation e
    WHERE h.has_visit AND h.day BETWEEN e.first_day AND e.last_day
),
fault AS (
    SELECT
        (
            SELECT COUNT(*)
            FROM decided d
            CROSS JOIN evaluation e
            WHERE d.pilot_revision IS DISTINCT FROM e.pilot_revision OR d.policy_revision IS DISTINCT FROM e.policy_revision
        ) + (
            SELECT COUNT(*)
            FROM pos_pilot_assignment a
            CROSS JOIN evaluation e
            WHERE a.pilot_revision IS DISTINCT FROM e.pilot_revision
        ) AS revision,
        (SELECT COUNT(*) - COUNT(DISTINCT pos_placement_decision_id) FROM decided)
            + (
                SELECT COUNT(*)
                FROM (
                    SELECT 1
                    FROM decided
                    GROUP BY pilot_revision, decision_day, delivery_point_id
                    HAVING COUNT(*) > 1
                ) twice
            )
            + (SELECT COUNT(loan_id) - COUNT(DISTINCT loan_id) FROM decided) AS duplicate,
        (
            SELECT COUNT(*)
            FROM decided
            WHERE outcome IS NULL
                OR outcome NOT IN ('selected', 'no_budget', 'no_asset', 'nonpositive_score', 'ineligible', 'cancelled')
        ) AS outcome,
        (SELECT COUNT(*) FROM decided WHERE NOT sound) AS fields,
        (
            SELECT COUNT(*)
            FROM decided d
            LEFT JOIN programme p ON p.customer_id = d.customer_id
            LEFT JOIN region r ON r.region_id = d.region_id
            LEFT JOIN delivery_point dp ON dp.delivery_point_id = d.delivery_point_id
            LEFT JOIN store st ON st.store_id = dp.store_id
            LEFT JOIN area ar ON ar.area_id = st.area_id
            LEFT JOIN employee em ON em.employee_id = d.manager_id
            LEFT JOIN pos_material pm ON pm.pos_material_id = d.forecast_material_id
            LEFT JOIN pos_loan l ON l.pos_loan_id = d.loan_id
            LEFT JOIN pos_asset pa ON pa.pos_asset_id = l.pos_asset_id
            WHERE p.customer_id IS NULL
                OR p.region_id IS DISTINCT FROM d.region_id
                OR r.region_id IS NULL
                OR dp.customer_id IS DISTINCT FROM d.customer_id
                OR ar.region_id IS DISTINCT FROM d.region_id
                OR d.manager_id IS NOT NULL AND em.employee_id IS NULL
                OR d.forecast_material_id IS NOT NULL AND pm.pos_material_id IS NULL
                OR d.loan_id IS NOT NULL AND (
                    l.pos_loan_id IS NULL
                    OR l.customer_id IS DISTINCT FROM d.customer_id
                    OR l.lent_on IS DISTINCT FROM d.decision_day + 1
                    OR pa.pos_asset_id IS NOT NULL AND pa.region_id IS DISTINCT FROM d.region_id
                )
        ) AS links,
        (
            SELECT COUNT(*)
            FROM pos_loan l
            CROSS JOIN w
            WHERE l.lent_on >= w.assigned_on
                AND NOT EXISTS (SELECT 1 FROM decided d WHERE d.outcome = 'selected' AND d.loan_id = l.pos_loan_id)
        ) + (
            SELECT COUNT(*)
            FROM decided d
            WHERE d.outcome = 'cancelled'
                AND EXISTS (
                    SELECT 1 FROM pos_loan l WHERE l.customer_id = d.customer_id AND l.lent_on = d.decision_day + 1
                )
        ) AS loans,
        (
            SELECT COUNT(*)
            FROM spending s
            LEFT JOIN customer c ON c.customer_id = s.customer_id
            LEFT JOIN programme p ON p.customer_id = s.customer_id
            WHERE c.customer_id IS NULL OR c.channel = 'small_retail' AND p.customer_id IS NULL
        ) AS ownership,
        (SELECT COUNT(*) FROM loan WHERE NOT has_asset) AS asset,
        (SELECT COUNT(*) FROM handed WHERE NOT has_visit) AS visit,
        (
            SELECT COUNT(*)
            FROM used x
            JOIN used y ON y.pos_asset_id = x.pos_asset_id AND y.pos_loan_id > x.pos_loan_id
            WHERE x.days > 0 AND y.days > 0 AND x.first_day <= y.last_day AND y.first_day <= x.last_day
        ) AS overlap,
        (
            SELECT COUNT(*)
            FROM used u
            JOIN programme p ON p.customer_id = u.customer_id
            WHERE u.days > 0 AND u.has_asset AND u.asset_region_id <> p.region_id
        ) AS astray
),
round_sums AS (
    SELECT
        region_id,
        COUNT(*) AS rounds,
        COUNT(q) AS comparable_rounds,
        SUM(decisions) AS decisions,
        SUM(viable) AS viable,
        COALESCE(SUM(chosen) FILTER (WHERE q IS NOT NULL), 0) AS chosen,
        COALESCE(SUM(unchosen) FILTER (WHERE q IS NOT NULL), 0) AS unchosen,
        AVG(q) AS quality
    FROM round_quality
    GROUP BY region_id
),
outcomes AS (
    SELECT
        region_id,
        SUM(selected) AS selected,
        SUM(cancelled) AS cancelled,
        SUM(no_budget) AS no_budget,
        SUM(no_asset) AS no_asset,
        SUM(no_asset_initial) AS no_asset_initial,
        SUM(nonpositive_score) AS nonpositive_score,
        SUM(ineligible) AS ineligible
    FROM counted
    GROUP BY region_id
),
sold AS (
    SELECT s.customer_id, SUM(s.customer_margin) AS margin, COUNT(*) AS lines
    FROM sales_pl s
    JOIN programme p ON p.customer_id = s.customer_id
    CROSS JOIN evaluation e
    WHERE s.ordered_on BETWEEN e.first_day AND e.last_day
    GROUP BY s.customer_id
),
lent AS (
    SELECT customer_id, SUM(use_cost) AS cost FROM used GROUP BY customer_id
),
given AS (
    SELECT h.customer_id, SUM(h.cost) AS cost
    FROM handed h
    CROSS JOIN evaluation e
    WHERE h.has_visit AND h.day BETWEEN e.first_day AND e.last_day
    GROUP BY h.customer_id
),
spent AS (
    SELECT
        p.region_id,
        COALESCE(s.margin, 0) AS margin,
        COALESCE(s.lines, 0) AS lines,
        COALESCE(l.cost, 0) AS asset_use_cost,
        COALESCE(g.cost, 0) AS handout_cost
    FROM programme p
    LEFT JOIN sold s ON s.customer_id = p.customer_id
    LEFT JOIN lent l ON l.customer_id = p.customer_id
    LEFT JOIN given g ON g.customer_id = p.customer_id
),
economics AS (
    SELECT
        region_id,
        COUNT(*) AS programme_customers,
        SUM(margin) AS observed_margin,
        SUM(asset_use_cost) AS asset_use_cost,
        SUM(handout_cost) AS handout_cost,
        COUNT(*) FILTER (WHERE lines = 0) AS no_sales_customers,
        COALESCE(SUM(asset_use_cost + handout_cost) FILTER (WHERE lines = 0), 0) AS no_sales_cost
    FROM spent
    GROUP BY region_id
),
carried AS (
    SELECT
        pa.pos_asset_id,
        pa.region_id,
        pa.cost,
        GREATEST(
            LEAST(COALESCE(pa.written_off_on - 1, as_of.day), as_of.day, e.last_day) - GREATEST(pa.acquired_on, e.first_day)
                + 1,
            0
        ) AS days
    FROM pos_asset pa
    CROSS JOIN evaluation e
    CROSS JOIN as_of
),
fleet AS (
    SELECT
        c.region_id,
        SUM(c.cost * c.days / e.life_days) AS fleet_cost,
        COUNT(*) FILTER (WHERE c.days > 0 AND COALESCE(u.days, 0) = 0) AS idle_assets
    FROM carried c
    CROSS JOIN evaluation e
    LEFT JOIN (SELECT pos_asset_id, SUM(days) AS days FROM used GROUP BY pos_asset_id) u
        ON u.pos_asset_id = c.pos_asset_id
    GROUP BY c.region_id
),
fleet_used AS (
    SELECT asset_region_id AS region_id, SUM(use_cost) AS cost FROM used WHERE has_asset GROUP BY asset_region_id
),
region_sums AS (
    SELECT
        r.code AS region,
        COALESCE(rs.rounds, 0) AS rounds,
        COALESCE(rs.comparable_rounds, 0) AS comparable_rounds,
        COALESCE(rs.decisions, 0) AS decisions,
        COALESCE(rs.viable, 0) AS viable,
        COALESCE(rs.chosen, 0) AS chosen,
        COALESCE(rs.unchosen, 0) AS unchosen,
        COALESCE(o.selected, 0) AS selected,
        COALESCE(o.cancelled, 0) AS cancelled,
        COALESCE(o.no_budget, 0) AS no_budget,
        COALESCE(o.no_asset, 0) AS no_asset,
        COALESCE(o.no_asset_initial, 0) AS no_asset_initial,
        COALESCE(o.nonpositive_score, 0) AS nonpositive_score,
        COALESCE(o.ineligible, 0) AS ineligible,
        rs.quality,
        COALESCE(ec.programme_customers, 0) AS programme_customers,
        COALESCE(ec.observed_margin, 0) AS observed_margin,
        COALESCE(ec.asset_use_cost, 0) AS asset_use_cost,
        COALESCE(ec.handout_cost, 0) AS handout_cost,
        COALESCE(ec.asset_use_cost, 0) + COALESCE(ec.handout_cost, 0) AS programme_cost,
        COALESCE(ec.no_sales_customers, 0) AS no_sales_customers,
        COALESCE(ec.no_sales_cost, 0) AS no_sales_cost,
        COALESCE(fl.fleet_cost, 0) AS fleet_cost,
        COALESCE(fl.fleet_cost, 0) - COALESCE(fu.cost, 0) AS idle_cost,
        COALESCE(fl.idle_assets, 0) AS idle_assets
    FROM region r
    LEFT JOIN round_sums rs ON rs.region_id = r.region_id
    LEFT JOIN outcomes o ON o.region_id = r.region_id
    LEFT JOIN economics ec ON ec.region_id = r.region_id
    LEFT JOIN fleet fl ON fl.region_id = r.region_id
    LEFT JOIN fleet_used fu ON fu.region_id = r.region_id
),
region_quality AS (
    SELECT
        s.*,
        CASE
            WHEN NOT w.one THEN 'window'
            WHEN as_of.day IS NULL OR as_of.day < e.last_day THEN 'short'
            WHEN f.revision > 0 THEN 'revision'
            WHEN f.duplicate > 0 THEN 'duplicate'
            WHEN f.outcome > 0 THEN 'outcome'
            WHEN f.fields > 0 THEN 'fields'
            WHEN f.links > 0 THEN 'links'
            WHEN f.loans > 0 THEN 'loans'
            WHEN f.ownership > 0 THEN 'ownership'
            WHEN f.asset > 0 THEN 'asset'
            WHEN f.visit > 0 THEN 'visit'
            WHEN f.overlap > 0 THEN 'overlap'
            WHEN f.astray > 0 THEN 'astray'
            WHEN MIN(s.programme_customers) OVER () = 0 THEN 'regions'
            WHEN NOT (
                s.comparable_rounds >= e.least_rounds AND s.chosen >= e.least_chosen AND s.unchosen >= e.least_unchosen
            ) THEN 'support'
            ELSE 'valid'
        END AS integrity
    FROM region_sums s
    CROSS JOIN w
    CROSS JOIN as_of
    CROSS JOIN fault f
    CROSS JOIN evaluation e
)
-- region quality: end
SELECT
    CASE
        WHEN COUNT(*) <> 6 OR COUNT(pos_skill) <> 6 OR NOT BOOL_AND(valid) THEN NULL
        WHEN MIN(rq) = MAX(rq) THEN 0
        ELSE corr(rq, rs)
    END AS metric
FROM (
    SELECT
        r.integrity = 'valid' AS valid,
        s.pos_skill,
        RANK() OVER (ORDER BY r.quality) + (COUNT(*) OVER (PARTITION BY r.quality) - 1) / 2.0 AS rq,
        RANK() OVER (ORDER BY s.pos_skill) + (COUNT(*) OVER (PARTITION BY s.pos_skill) - 1) / 2.0 AS rs
    FROM region_quality r
    LEFT JOIN skill s ON s.region = r.region
) ranks
