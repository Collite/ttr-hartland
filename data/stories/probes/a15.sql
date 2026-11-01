-- A-15 — the Chrudim batch: quality complaints about dried soups in the three weeks from 2025-11-17 (planted: 40,
-- Moravia first).
SELECT COUNT(*)
FROM complaint c
JOIN sku s ON s.sku_id = c.sku_id
JOIN product p ON p.product_id = s.product_id
JOIN product_line pl ON pl.product_line_id = p.product_line_id
WHERE c.kind = 'quality' AND pl.code = 'POLEVKY' AND c.received_on BETWEEN DATE '2025-11-17' AND DATE '2025-12-07'
