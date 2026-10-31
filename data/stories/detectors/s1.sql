-- S1 — seasonality (design 04 §S1): per line, the mean weekly cases of its peak weeks ÷ those of its trough weeks,
-- 2024–2025, free goods left out. The metric is the smallest ratio over the lines with a curve (s1.yaml `curves`:
-- Decaf is flat, the Professional lines follow their channel). The weeks below are s1.yaml's (a test keeps them so).
WITH curve (line, peak, trough) AS (
    VALUES
        ('CLASSIC', ARRAY[45,46,47,48,49,50,51,52,53,1,2,3,4,5,6,7,8], ARRAY[27,28,29,30,31,32,33]),
        ('GOLD', ARRAY[46,47,48,49,50,51,52,2,3,4,5,6,7,8], ARRAY[27,28,29,30,31,32,33]),
        ('ESPRESSO', ARRAY[46,47,48,49,50,51,52,2,3,4,5,6,7,8], ARRAY[27,28,29,30,31,32,33]),
        ('SMESI', ARRAY[48,49,50,51,52,36,37,38,39,40], ARRAY[26,27,28,29,30,31,32]),
        ('POLEVKY', ARRAY[42,43,44,45,46,47,48,49,50,51,52,53,1,2,3,4,5,6,7,8,9,10], ARRAY[24,25,26,27,28,29,30,31,32,33,34]),
        ('PREMIUM', ARRAY[42,43,44,45,46,47,48,49,50,51,52,53,1,2,3,4,5,6,7,8,9,10], ARRAY[24,25,26,27,28,29,30,31,32,33,34]),
        ('KELIMKY', ARRAY[40,41,42,43,44,45,46,47,48,49,50,51,52,53,1,2,3,4,5,6,7,8,9,10,11,12], ARRAY[24,25,26,27,28,29,30,31,32,33,34]),
        ('NUDLE', ARRAY[36,37,38,39,40,2,3,4,5,6], ARRAY[28,29,30,31,32])
),
weekly AS (
    SELECT pl.code AS line, c.iso_year, c.week, SUM(l.qty_cases) AS cases
    FROM sales_order_line l
    JOIN sales_order o ON o.sales_order_id = l.sales_order_id
    JOIN calendar c ON c.day = o.ordered_on
    JOIN sku s ON s.sku_id = l.sku_id
    JOIN product p ON p.product_id = s.product_id
    JOIN product_line pl ON pl.product_line_id = p.product_line_id
    WHERE o.ordered_on BETWEEN DATE '2024-01-01' AND DATE '2025-12-31' AND NOT l.is_free_goods
    GROUP BY pl.code, c.iso_year, c.week
),
ratio AS (
    SELECT
        w.line,
        AVG(w.cases) FILTER (WHERE w.week = ANY (cv.peak))
            / NULLIF(AVG(w.cases) FILTER (WHERE w.week = ANY (cv.trough)), 0) AS peak_over_trough
    FROM weekly w
    JOIN curve cv ON cv.line = w.line
    GROUP BY w.line
)
SELECT MIN(peak_over_trough) AS metric FROM ratio
