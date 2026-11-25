-- Append to the common S3 CTEs. Parameters are evaluator-only story labels and the null-mode flag.
, linked_order AS (
    SELECT visit_id, COUNT(*) AS n
    FROM sales_order
    WHERE visit_id IS NOT NULL
    GROUP BY visit_id
), linkage AS (
    SELECT v.employee_id, SUM(COALESCE(o.n, 0))::bigint AS linked_orders,
           COUNT(*) FILTER (WHERE COALESCE(o.n, 0) <> CASE WHEN v.outcome = 'order' THEN 1 ELSE 0 END)
               AS link_mismatches
    FROM sales_visit v
    LEFT JOIN linked_order o USING (visit_id)
    GROUP BY v.employee_id
), territory_history AS (
    SELECT employee_id, jsonb_agg(jsonb_build_array(area_id, from_day, to_day) ORDER BY from_day, area_id) AS periods
    FROM duty
    GROUP BY employee_id
), labels AS (
    SELECT key::bigint AS employee_id, value AS skills
    FROM jsonb_each(%(labels)s)
), measures AS (
    SELECT o.employee_id, m.measure, m.n, m.d, (l.skills ->> m.measure)::double precision AS skill
    FROM observed o
    LEFT JOIN labels l USING (employee_id)
    CROSS JOIN LATERAL (VALUES
        ('conversion', o.converting_visits, o.sales_visits),
        ('acquisition', o.wins, o.potential_days),
        ('retention', -o.losses, o.active_days),
        ('cross_sell', o.first_paid_lines, o.active_days)
    ) m(measure, n, d)
), skill_ranks AS (
    SELECT employee_id, measure,
           RANK() OVER (PARTITION BY measure ORDER BY skill)
             + (COUNT(*) OVER (PARTITION BY measure, skill) - 1) / 2.0 AS skill_rank
    FROM measures
    WHERE skill IS NOT NULL
), observed_ranks AS (
    -- Compare ratios by exact cross multiplication. One pass counts both higher rates and ties.
    SELECT m.employee_id, m.measure,
           1 + COUNT(*) FILTER (WHERE p.n::numeric * m.d > m.n::numeric * p.d) AS rank,
           (COUNT(*) FILTER (WHERE p.n::numeric * m.d < m.n::numeric * p.d))
             + (COUNT(*) FILTER (WHERE p.n::numeric * m.d = m.n::numeric * p.d) + 1) / 2.0 AS average_rank
    FROM measures m
    JOIN measures p ON p.measure = m.measure AND p.d > 0
    WHERE m.d > 0
    GROUP BY m.employee_id, m.measure
), correlations AS (
    SELECT measure,
           CASE WHEN COUNT(*) = 50 THEN corr(average_rank, skill_rank ORDER BY employee_id) END AS rho
    FROM observed_ranks
    JOIN skill_ranks USING (employee_id, measure)
    GROUP BY measure
), decisions AS (
    SELECT measure, rho, CASE WHEN %(null)s THEN abs(rho) < 0.30 ELSE rho >= 0.70 END AS passed
    FROM correlations
)
SELECT
    (SELECT day FROM as_of) AS last_order,
    (SELECT jsonb_agg(jsonb_build_array(employee_id, slot) ORDER BY employee_id, slot) FROM slot) AS slots,
    (SELECT jsonb_agg(to_jsonb(result) - 'employee_id' || jsonb_build_object('employee_id', employee_id::text)
                      ORDER BY employee_id)
     FROM (
        SELECT o.*, c.place, p.profile, t.periods,
               COALESCE(l.linked_orders, 0) AS linked_orders, COALESCE(l.link_mismatches, 0) AS link_mismatches,
               (SELECT jsonb_object_agg(measure, rank) FROM observed_ranks r WHERE r.employee_id = o.employee_id)
                   AS ranks
        FROM observed o
        JOIN composite c USING (employee_id)
        JOIN territory_history t USING (employee_id)
        LEFT JOIN linkage l USING (employee_id)
        LEFT JOIN slot s USING (employee_id)
        LEFT JOIN pinned p USING (slot)
     ) result) AS reps,
    (SELECT jsonb_object_agg(measure, jsonb_build_array(rho, passed)) FROM decisions) AS correlations
