-- return_request — the return request with its state derived from its lines (the automation extension's view, taken
-- over with its columns and its rule; `document_id` and `source` appended): `received` while any line is received (or
-- none has arrived yet), `closed` when every line is rejected, not_returned or closed, `decided` in between. The key
-- of the return flow is the line, so the header keeps no state of its own.
SELECT
    h.rr_request_number,
    h.rr_customer_sk,
    h.rr_order_number,
    h.rr_channel,
    h.rr_requested_date,
    CASE
        WHEN COUNT(l.rl_line_id) = 0 OR BOOL_OR(l.rl_state = 'received') THEN 'received'
        WHEN BOOL_AND(l.rl_state IN ('rejected', 'not_returned', 'closed')) THEN 'closed'
        ELSE 'decided'
    END::text AS rr_state,
    h.document_id,
    h.source
FROM return_request_hdr h
LEFT JOIN return_request_line l ON l.rl_request_number = h.rr_request_number
GROUP BY h.rr_request_number
