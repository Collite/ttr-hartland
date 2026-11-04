-- S6 — discount efficiency (design 04 §S6, amended 2026-11-04): per pinned rep, the incremental Customer Margin per
-- agreed-discount crown, after the discount. Customer Margin is sales_pl's: net revenue − delivery − warehouse, no
-- COGS (before G3/G4 write the costs it is the net revenue). An agreed discount's lift is its line's share of the
-- customer's cases (against its other lines) in the window — its first day to a fortnight past its last, the
-- small-retail cadence, so the lifted demand ordered after the end counts, at list — ÷ the same share in the 8 weeks
-- before: all the customer's lines ride the same orders, so the orders' timing cancels. Its gain is the line's Customer
-- Margin in the window less what the line would have earned without the lift at list, margin − (margin + crowns) ÷
-- lift, the crowns being the line's agreed discounts in the window. Counted: the agreements from 2024-02-26 (their 8
-- weeks before lie inside the history) whose window ends by the world's last order day, with no other agreement of the
-- customer-line ending in the 8 weeks before or starting in the fortnight after, and with cases of the line and of the
-- others both before and in the window.
-- The metric is the disciplined reps' gain per crown minus the undisciplined reps': a difference, not a ratio, because
-- the undisciplined one may be negative (5–8 % to whoever asks). 04's "> 1 … < 0.5" gives at least 0.5; equal reps
-- give about 0. The pins are s6.yaml's `pinned_reps` (≥ 1.0 disciplined), by the area the rep holds on the warm-up's
-- first day, 2023-01-01 (a test keeps the codes below in step). The pandas twin is checks/stories.py's `measured`.
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
    SELECT pa.price_agreement_id, pa.customer_id, pa.product_line_id, pa.valid_from, pa.valid_to, pinned.disciplined
    FROM price_agreement pa
    CROSS JOIN as_of
    JOIN slot ON slot.employee_id = pa.agreed_by_employee_id
    JOIN pinned ON pinned.slot = slot.slot
    WHERE pa.kind = 'agreed_discount' AND pa.product_line_id IS NOT NULL
      AND pa.valid_from >= DATE '2024-02-26' AND pa.valid_to + 14 <= as_of.day
      AND NOT EXISTS (
          SELECT FROM price_agreement other
          WHERE other.customer_id = pa.customer_id AND other.product_line_id = pa.product_line_id
            AND other.price_agreement_id <> pa.price_agreement_id
            AND (other.valid_to BETWEEN pa.valid_from - 56 AND pa.valid_from - 1
                 OR other.valid_from BETWEEN pa.valid_to + 1 AND pa.valid_to + 14)
      )
),
window_sums AS (
    SELECT
        ag.disciplined,
        SUM(s.cases) FILTER (WHERE s.product_line_id = ag.product_line_id AND s.ordered_on < ag.valid_from)
            AS line_before,
        SUM(s.cases) FILTER (WHERE s.product_line_id <> ag.product_line_id AND s.ordered_on < ag.valid_from)
            AS others_before,
        SUM(s.cases) FILTER (WHERE s.product_line_id = ag.product_line_id AND s.ordered_on >= ag.valid_from)
            AS line_window,
        SUM(s.cases) FILTER (WHERE s.product_line_id <> ag.product_line_id AND s.ordered_on >= ag.valid_from)
            AS others_window,
        SUM(s.customer_margin) FILTER (WHERE s.product_line_id = ag.product_line_id AND s.ordered_on >= ag.valid_from)
            AS margin,
        SUM(s.discount_agreed) FILTER (WHERE s.product_line_id = ag.product_line_id AND s.ordered_on >= ag.valid_from)
            AS crowns
    FROM agreement ag
    JOIN sales_pl s ON s.customer_id = ag.customer_id
        AND s.ordered_on BETWEEN ag.valid_from - 56 AND ag.valid_to + 14
    GROUP BY ag.price_agreement_id, ag.disciplined
),
gain AS (
    SELECT
        disciplined,
        margin - (margin + crowns) * (line_before / others_before) / (line_window / others_window) AS margin_gain,
        crowns AS discount
    FROM window_sums
    WHERE line_before > 0 AND others_before > 0 AND line_window > 0 AND others_window > 0
)
SELECT SUM(margin_gain) FILTER (WHERE disciplined) / NULLIF(SUM(discount) FILTER (WHERE disciplined), 0)
    - SUM(margin_gain) FILTER (WHERE NOT disciplined) / NULLIF(SUM(discount) FILTER (WHERE NOT disciplined), 0)
    AS metric
FROM gain
