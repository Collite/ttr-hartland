-- route_current — the route version current at the end of its day: the morning plan, or its last re-plan (a version
-- nothing supersedes), with its stops' outcomes. Every delivered stop belongs to one of these (05 §7).
SELECT
    r.route_id,
    r.dc_site_id,
    r.day,
    r.carrier_id,
    r.vehicle_id,
    r.version,
    r.planned_departure,
    r.planned_km,
    r.planned_stops,
    r.planned_cost,
    r.actual_km,
    r.actual_cost,
    r.planner,
    r.status,
    COUNT(s.route_stop_id) AS stops,
    COUNT(s.route_stop_id) FILTER (WHERE s.status = 'delivered') AS stops_delivered,
    COUNT(s.route_stop_id) FILTER (WHERE s.status = 'late') AS stops_late,
    COUNT(s.route_stop_id) FILTER (WHERE s.status = 'failed') AS stops_failed,
    COUNT(s.route_stop_id) FILTER (WHERE s.status = 'moved') AS stops_moved
FROM route r
LEFT JOIN route_stop s ON s.route_id = r.route_id
WHERE r.superseded_by_route_id IS NULL
GROUP BY r.route_id
