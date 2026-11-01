-- trade_spend — the trade spend by what caused it: a month × a channel × a discount kind × its code (the promotion's,
-- the agreement's or the tier's) × whose budget bears it (contracts §4; NH-D6). Joining `promotion` → `campaign`
-- gives the spend by campaign and mechanic. Money in CZK without VAT.
SELECT
    date_trunc('month', o.ordered_on)::date AS month,
    o.channel,
    x.kind,
    x.code,
    x.promotion_id,
    x.price_agreement_id,
    x.budget,
    COUNT(*) AS lines,
    SUM(l.qty_cases) AS cases,
    SUM(x.amount_line) AS amount
FROM sales_order_line_discount x
JOIN sales_order_line l ON l.sales_order_line_id = x.sales_order_line_id
JOIN sales_order o ON o.sales_order_id = l.sales_order_id
GROUP BY 1, o.channel, x.kind, x.code, x.promotion_id, x.price_agreement_id, x.budget
