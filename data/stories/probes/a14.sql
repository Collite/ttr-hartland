-- A-14 — the credit-blocked series: a regular customer crosses its credit limit. Customers with at least two orders
-- held over the limit in June 2026, every one later released, and no order held, for either reason, in the 12 months
-- before. The holds themselves are company policy and run in the null world too (an order over the limit, or with an
-- invoice more than 21 days overdue, is held), so the probe looks for the planted signature, not for any hold. At
-- least two, not three: a small-retail customer orders about every 14 days. If the null world still finds one at G5,
-- S5.2 records the exception in anomalies.yaml; the threshold is not tuned.
WITH june AS (
    SELECT o.customer_id
    FROM sales_order o
    WHERE o.hold_reason = 'credit_limit' AND o.ordered_on BETWEEN DATE '2026-06-01' AND DATE '2026-06-30'
    GROUP BY o.customer_id
    HAVING COUNT(*) >= 2 AND bool_and(o.status IN ('released', 'shipped', 'invoiced'))
)
SELECT COUNT(*)
FROM june j
WHERE NOT EXISTS (
    SELECT 1
    FROM sales_order o
    WHERE o.customer_id = j.customer_id AND o.hold_reason IS NOT NULL
      AND o.ordered_on BETWEEN DATE '2025-06-01' AND DATE '2026-05-31'
)
