-- receivable — an invoice with what is paid and what is open, as of the world's last day: the latest invoice or
-- payment in the database (not the wall clock, so a rebuild gives the same rows). Days overdue and the ageing bucket
-- count from the due date; a credit note is a negative receivable.
WITH as_of AS (
    SELECT COALESCE(
        GREATEST((SELECT MAX(issued_on) FROM invoice), (SELECT MAX(received_on) FROM payment)),
        CURRENT_DATE
    ) AS day
)
SELECT
    i.invoice_id,
    i.invoice_no,
    i.kind,
    i.customer_id,
    c.channel,
    i.issued_on,
    i.due_on,
    i.gross_amount,
    COALESCE(paid.amount, 0) AS paid_amount,
    i.gross_amount - COALESCE(paid.amount, 0) AS open_amount,
    as_of.day AS as_of,
    CASE WHEN i.gross_amount - COALESCE(paid.amount, 0) > 0 THEN GREATEST(as_of.day - i.due_on, 0) ELSE 0 END
        AS days_overdue,
    CASE
        WHEN i.gross_amount - COALESCE(paid.amount, 0) <= 0 THEN 'paid'
        WHEN as_of.day <= i.due_on THEN 'not_due'
        WHEN as_of.day - i.due_on <= 30 THEN '1_30'
        WHEN as_of.day - i.due_on <= 60 THEN '31_60'
        WHEN as_of.day - i.due_on <= 90 THEN '61_90'
        ELSE 'over_90'
    END AS ageing_bucket
FROM invoice i
CROSS JOIN as_of
LEFT JOIN customer c ON c.customer_id = i.customer_id
LEFT JOIN (
    SELECT invoice_id, SUM(amount) AS amount FROM payment_allocation GROUP BY invoice_id
) paid ON paid.invoice_id = i.invoice_id
WHERE i.kind <> 'intercompany'
