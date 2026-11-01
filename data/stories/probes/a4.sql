-- A-4 — the Monday carrier: Modřice's carriers whose Monday routes have a stop 45 minutes or more late on at least
-- half of them (the planted one: 60 %).
SELECT COUNT(*)
FROM (
    SELECT r.carrier_id
    FROM route_current r
    JOIN site dc ON dc.site_id = r.dc_site_id
    JOIN LATERAL (
        SELECT bool_or(s.actual_arrival - s.planned_arrival >= INTERVAL '45 minutes') AS late
        FROM route_stop s
        WHERE s.route_id = r.route_id
    ) stops ON true
    WHERE dc.code = 'DC-MODRICE' AND EXTRACT(ISODOW FROM r.day) = 1
    GROUP BY r.carrier_id
    HAVING AVG(CASE WHEN stops.late THEN 1.0 ELSE 0.0 END) >= 0.5
) monday
