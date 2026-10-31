-- A-3 — the chronic stock-out: the months of 2025 in which Gold 200 g ran out at Jirny in the month's last week.
SELECT COUNT(DISTINCT date_trunc('month', so.from_day))
FROM stockout so
JOIN sku s ON s.sku_id = so.sku_id
JOIN site dc ON dc.site_id = so.dc_site_id
WHERE s.sku_code = '2931000000180' AND dc.code = 'DC-JIRNY'
  AND so.from_day BETWEEN DATE '2025-01-01' AND DATE '2025-12-31'
  AND so.from_day > (date_trunc('month', so.from_day) + INTERVAL '1 month - 8 days')::date
