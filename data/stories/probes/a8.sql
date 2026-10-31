-- A-8 — the rep who records nothing: reps whose visits go unrecorded 60 % of the time or more (planted: 70 %).
SELECT COUNT(*)
FROM (
    SELECT employee_id
    FROM visit
    GROUP BY employee_id
    HAVING AVG(CASE WHEN recorded THEN 0.0 ELSE 1.0 END) >= 0.6
) silent
