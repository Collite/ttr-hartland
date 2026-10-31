-- A-5 — bad-weather days cluster the events: 1 when the share of routes with an event, among the routes with a stop
-- in a region on one of its severe days, is at least 12 % (planted 15 %) and at least twice the share on the others.
WITH route_day AS (
    SELECT
        r.route_id,
        EXISTS (
            SELECT 1
            FROM route_stop s
            JOIN delivery_point dp ON dp.delivery_point_id = s.delivery_point_id
            JOIN area_okres ao ON ao.okres_id = dp.okres_id
            JOIN area a ON a.area_id = ao.area_id
            JOIN weather_daily w ON w.region_id = a.region_id AND w.day = r.day
            WHERE s.route_id = r.route_id AND w.severe
        ) AS severe,
        EXISTS (SELECT 1 FROM delivery_event e WHERE e.route_id = r.route_id) AS has_event
    FROM route_current r
),
shares AS (
    SELECT AVG(CASE WHEN has_event THEN 1.0 ELSE 0.0 END) FILTER (WHERE severe) AS severe,
           AVG(CASE WHEN has_event THEN 1.0 ELSE 0.0 END) FILTER (WHERE NOT severe) AS calm
    FROM route_day
)
SELECT COUNT(*) FROM shares WHERE severe >= 0.12 AND severe >= 2 * calm
