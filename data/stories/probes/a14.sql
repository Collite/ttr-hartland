-- A-14 — the credit-blocked series: customers with an order held over the credit limit in June 2026 that was later
-- released (the status moves on once the customer pays).
SELECT COUNT(DISTINCT o.customer_id)
FROM sales_order o
WHERE o.hold_reason = 'credit_limit' AND o.ordered_on BETWEEN DATE '2026-06-01' AND DATE '2026-06-30'
  AND o.status IN ('released', 'shipped', 'invoiced')
