-- A-9 — two chains delist Espresso Decaf 100 g: the chains whose listing of it ends in March 2026.
SELECT COUNT(DISTINCT cl.chain_id)
FROM channel_listing cl
JOIN sku s ON s.sku_id = cl.sku_id
WHERE s.sku_code = '2931000000326' AND cl.chain_id IS NOT NULL
  AND cl.valid_to BETWEEN DATE '2026-03-01' AND DATE '2026-03-31'
