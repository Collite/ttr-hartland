-- A-6 — the competitor's winter promotion: sightings of a Kavalír promotion at visits in East Bohemia (region VCH)
-- from week 45 to week 50 of 2025. Only the `promotion` kind: Kavalír is strong in East Bohemia, so its listings,
-- displays and price cuts are seen there all year.
SELECT COUNT(*)
FROM competitor_sighting cs
JOIN competitor k ON k.competitor_id = cs.competitor_id
JOIN visit v ON v.visit_id = cs.visit_id
JOIN store st ON st.store_id = v.store_id
JOIN area a ON a.area_id = st.area_id
JOIN region r ON r.region_id = a.region_id
WHERE k.code = 'KAVALIR' AND cs.activity_kind = 'promotion' AND r.code = 'VCH'
  AND v.day BETWEEN DATE '2025-11-03' AND DATE '2025-12-14'
