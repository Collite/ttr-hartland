-- A-12 — the Liberec shop: 1 when its Premium soups sold per receipt are at least twice the other shops' mean
-- (planted 2.5×).
WITH per_shop AS (
    SELECT si.code AS shop, SUM(rl.qty) / COUNT(DISTINCT r.shop_receipt_id) AS per_receipt
    FROM shop_receipt r
    JOIN site si ON si.site_id = r.site_id
    JOIN shop_receipt_line rl ON rl.shop_receipt_id = r.shop_receipt_id
    JOIN sku s ON s.sku_id = rl.sku_id
    JOIN product p ON p.product_id = s.product_id
    JOIN product_line pl ON pl.product_line_id = p.product_line_id
    WHERE pl.code = 'PREMIUM'
    GROUP BY si.code
)
SELECT COUNT(*)
FROM per_shop l
WHERE l.shop = 'SH-LIBEREC'
  AND l.per_receipt >= 2 * (SELECT AVG(per_receipt) FROM per_shop WHERE shop <> 'SH-LIBEREC')
