-- A-2 — the novelty with the wrong initial stock: the pumpkin cream pouch runs out at Modřice within 8 weeks of its
-- launch on 2025-09-01.
SELECT COUNT(*)
FROM stockout so
JOIN sku s ON s.sku_id = so.sku_id
JOIN site dc ON dc.site_id = so.dc_site_id
WHERE s.sku_code = '2931000001088' AND dc.code = 'DC-MODRICE'
  AND so.from_day BETWEEN DATE '2025-09-01' AND DATE '2025-10-27'
