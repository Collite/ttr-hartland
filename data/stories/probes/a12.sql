-- A-12 — the Liberec shop: 1 when its Premium-soup units per receipt are at least twice the other shops' mean (planted
-- 2.5×). Per receipt means over all the shop's receipts, so more buyers count as well as bigger baskets.
WITH receipts AS (
    SELECT r.site_id, COUNT(*) AS n
    FROM shop_receipt r
    GROUP BY r.site_id
),
premium AS (
    SELECT r.site_id, SUM(rl.qty) AS units
    FROM shop_receipt r
    JOIN shop_receipt_line rl ON rl.shop_receipt_id = r.shop_receipt_id
    JOIN sku s ON s.sku_id = rl.sku_id
    JOIN product p ON p.product_id = s.product_id
    JOIN product_line pl ON pl.product_line_id = p.product_line_id
    WHERE pl.code = 'PREMIUM'
    GROUP BY r.site_id
),
per_shop AS (
    SELECT si.code AS shop, COALESCE(pr.units, 0) / rc.n AS per_receipt
    FROM receipts rc
    JOIN site si ON si.site_id = rc.site_id
    LEFT JOIN premium pr ON pr.site_id = rc.site_id
)
SELECT COUNT(*)
FROM per_shop l
WHERE l.shop = 'SH-LIBEREC'
  AND l.per_receipt >= 2 * (SELECT AVG(per_receipt) FROM per_shop WHERE shop <> 'SH-LIBEREC')
