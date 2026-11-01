-- A-11 — the key-account retro bonus: Tesco's credit notes issued in January 2026 that carry retro discount rows (the
-- 2025 bonus, booked in the wrong month unless accrued).
SELECT COUNT(DISTINCT i.invoice_id)
FROM invoice i
JOIN customer c ON c.customer_id = i.customer_id
JOIN chain ch ON ch.chain_id = c.chain_id
JOIN invoice_line il ON il.invoice_id = i.invoice_id
JOIN sales_order_line_discount d ON d.sales_order_line_id = il.sales_order_line_id
WHERE ch.code = 'TESCO' AND i.kind = 'credit_note' AND d.kind = 'retro'
  AND i.issued_on BETWEEN DATE '2026-01-01' AND DATE '2026-01-31'
