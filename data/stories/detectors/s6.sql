-- S6 — discount efficiency (design 04 §S6): per pinned rep, the customer margin gained per discount crown. The gain of
-- an agreed discount on a customer × line is the margin per day during the agreement minus that of the 8 weeks before,
-- times its days; the crowns are the agreed discounts granted during it (not its predecessor's, in the weeks before).
-- Only agreements from 2024-02-26 count, so their 8 weeks before are inside the history; each runs at most to the
-- world's last order day. The metric is the disciplined reps' gain per crown minus the undisciplined reps': a
-- difference, not a ratio, because the undisciplined one is negative by design (flat 5–8 % to low-elasticity lines
-- loses margin). 04's "> 1 … < 0.5" gives at least 0.5; equal reps give about 0. The pins are s6.yaml's
-- `pinned_reps` (≥ 1.0 disciplined), by the area the rep holds on the warm-up's first day, 2023-01-01 (a test keeps
-- the codes below in step).
WITH as_of AS (
    SELECT MAX(ordered_on) AS day FROM sales_order
),
pinned (slot, disciplined) AS (
    VALUES ('SJC-02', true), ('ZSC-02', true), ('SMS-02', false), ('VCH-05', false)
),
slot AS (
    SELECT a.employee_id, ar.code AS slot
    FROM assignment a
    JOIN area ar ON ar.area_id = a.area_id
    WHERE a.kind = 'rep_area' AND DATE '2023-01-01' BETWEEN a.valid_from AND a.valid_to
),
agreement AS (
    SELECT pa.price_agreement_id, pa.customer_id, pa.product_line_id, pa.valid_from,
           LEAST(pa.valid_to, as_of.day) AS valid_to, pinned.disciplined
    FROM price_agreement pa
    CROSS JOIN as_of
    JOIN slot ON slot.employee_id = pa.agreed_by_employee_id
    JOIN pinned ON pinned.slot = slot.slot
    WHERE pa.kind = 'agreed_discount' AND pa.product_line_id IS NOT NULL
      AND pa.valid_from BETWEEN DATE '2024-02-26' AND as_of.day
),
gain AS (
    SELECT
        ag.disciplined,
        (
            COALESCE(SUM(s.customer_margin) FILTER (WHERE s.ordered_on >= ag.valid_from), 0)
                / (ag.valid_to - ag.valid_from + 1)
            - COALESCE(SUM(s.customer_margin) FILTER (WHERE s.ordered_on < ag.valid_from), 0) / 56.0
        ) * (ag.valid_to - ag.valid_from + 1) AS margin_gain,
        COALESCE(SUM(s.discount_agreed) FILTER (WHERE s.ordered_on >= ag.valid_from), 0) AS discount
    FROM agreement ag
    LEFT JOIN sales_pl s ON s.customer_id = ag.customer_id AND s.product_line_id = ag.product_line_id
        AND s.ordered_on BETWEEN ag.valid_from - 56 AND ag.valid_to
    GROUP BY ag.price_agreement_id, ag.disciplined, ag.valid_from, ag.valid_to
)
SELECT SUM(margin_gain) FILTER (WHERE disciplined) / NULLIF(SUM(discount) FILTER (WHERE disciplined), 0)
    - SUM(margin_gain) FILTER (WHERE NOT disciplined) / NULLIF(SUM(discount) FILTER (WHERE NOT disciplined), 0)
    AS metric
FROM gain
