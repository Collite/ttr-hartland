-- A-1 — the chronic late payers: customers who paid at least six invoices, every one of them 30 days or more after
-- its due date. Three are planted (anomalies.yaml pins: two HoReCa, one convenience franchise).
SELECT COUNT(*)
FROM (
    SELECT i.customer_id
    FROM invoice i
    JOIN payment_allocation pa ON pa.invoice_id = i.invoice_id
    JOIN payment p ON p.payment_id = pa.payment_id
    WHERE i.kind = 'invoice'
    GROUP BY i.customer_id
    HAVING COUNT(DISTINCT i.invoice_id) >= 6 AND MIN(p.received_on - i.due_on) >= 30
) chronic
