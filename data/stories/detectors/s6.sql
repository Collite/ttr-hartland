-- S6 — discount efficiency (design 04 §S6): per pinned rep, the customer margin gained per discount crown — (the
-- margin of their customer × lines with an agreed discount, during the agreement minus the 8 weeks before, per day,
-- times its days) ÷ the agreed discounts on those lines. The metric: the disciplined reps' ratio ÷ the
-- undisciplined reps' (s6.yaml `pinned_reps`, ≥ 1.0 disciplined; a test keeps the codes below in step).
WITH pinned (slot, disciplined) AS (
    VALUES ('SJC-02', true), ('ZSC-02', true), ('SMS-02', false), ('VCH-05', false)
),
rep AS (
    SELECT DISTINCT ON (a.employee_id) a.employee_id, ar.code AS slot
    FROM assignment a
    JOIN area ar ON ar.area_id = a.area_id
    WHERE a.kind = 'rep_area'
    ORDER BY a.employee_id, a.valid_from
),
agreement AS (
    SELECT pa.price_agreement_id, pa.customer_id, pa.product_line_id, pa.valid_from,
           LEAST(pa.valid_to, DATE '2026-10-31') AS valid_to, pinned.disciplined
    FROM price_agreement pa
    JOIN rep ON rep.employee_id = pa.agreed_by_employee_id
    JOIN pinned ON pinned.slot = rep.slot
    WHERE pa.kind = 'agreed_discount' AND pa.product_line_id IS NOT NULL
),
gain AS (
    SELECT
        ag.disciplined,
        (SUM(s.customer_margin) FILTER (WHERE s.ordered_on >= ag.valid_from) / (ag.valid_to - ag.valid_from + 1)
            - COALESCE(SUM(s.customer_margin) FILTER (WHERE s.ordered_on < ag.valid_from), 0) / 56.0)
            * (ag.valid_to - ag.valid_from + 1) AS margin_gain,
        SUM(s.discount_agreed) AS discount
    FROM agreement ag
    JOIN sales_pl s ON s.customer_id = ag.customer_id AND s.product_line_id = ag.product_line_id
        AND s.ordered_on BETWEEN ag.valid_from - 56 AND ag.valid_to
    GROUP BY ag.price_agreement_id, ag.disciplined, ag.valid_from, ag.valid_to
)
SELECT (SUM(margin_gain) FILTER (WHERE disciplined) / NULLIF(SUM(discount) FILTER (WHERE disciplined), 0))
    / NULLIF(SUM(margin_gain) FILTER (WHERE NOT disciplined) / NULLIF(SUM(discount) FILTER (WHERE NOT disciplined), 0), 0)
    AS metric
FROM gain
