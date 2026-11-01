-- A-7 — the reps who skip visits: reps who made 65 % or fewer of their planned visits in 2025 (planted: three at 55 %,
-- against the average 90 %).
SELECT COUNT(*)
FROM (
    SELECT vp.employee_id
    FROM visit_plan vp
    LEFT JOIN visit v ON v.visit_plan_id = vp.visit_plan_id
    WHERE vp.planned_day BETWEEN DATE '2025-01-01' AND DATE '2025-12-31'
    GROUP BY vp.employee_id
    HAVING AVG(CASE WHEN v.visit_id IS NULL THEN 0.0 ELSE 1.0 END) <= 0.65
) skipping
