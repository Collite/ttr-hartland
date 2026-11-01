-- sales_pl — an order line with its P/L, from gross revenue to the product margin (contracts §4).
-- The `sales` entity, the `sales` cubelet and the H-L hand SQL all read it. The trade-spend columns are sums over the
-- line's discount list by kind; `retro` rolls into discount_individual (contracts §4, the review-189 note), while the
-- GL keeps it on its own account. Money in CZK without VAT.
--
-- The rep is whoever answers for the sale on the order day: in small retail (station shops included, a sub-channel)
-- the rep assigned to the store's area; in the KAM channels (big retail, gas KA, vending, HoReCa) the KAM assigned to
-- the customer; else (own shops, the web, an area or customer without one) whoever took the order. A rep still sees
-- the chain stores of their area, through `area_id`, not through this column. The area: the store's, else the
-- delivery point's okres's.
SELECT
    l.sales_order_line_id AS line_id,
    o.sales_order_id AS order_id,
    o.ordered_on,
    delivered.delivered_on,
    o.channel,
    c.sub_channel,
    o.customer_id,
    o.delivery_point_id,
    dp.store_id,
    l.sku_id,
    p.product_line_id,
    pl.brand_id,
    CASE
        WHEN o.channel = 'small_retail' THEN COALESCE(rep.employee_id, o.taken_by_employee_id)
        ELSE COALESCE(kam.employee_id, o.taken_by_employee_id)
    END AS rep_employee_id,
    COALESCE(st.area_id, ao.area_id) AS area_id,
    ok.code AS okres_code,
    o.dc_site_id,
    l.qty_cases AS cases,
    l.qty_pieces AS pieces,
    l.list_price_case,
    m.gross_revenue,
    d.discount_volume,
    d.discount_cross_sell,
    d.discount_agreed,
    d.discount_individual,
    d.discount_promotion,
    d.free_goods_value,
    m.gross_revenue - d.total AS net_revenue,
    m.delivery_cost,
    m.warehouse_cost,
    m.gross_revenue - d.total - m.delivery_cost - m.warehouse_cost AS customer_margin,
    m.cogs,
    m.marketing_cost,
    m.gross_revenue - d.total - m.delivery_cost - m.warehouse_cost - m.cogs - m.marketing_cost AS product_margin,
    l.vat_rate,
    ROUND((m.gross_revenue - d.total) * l.vat_rate / 100, 2) AS vat
FROM sales_order_line l
JOIN sales_order o ON o.sales_order_id = l.sales_order_id
JOIN customer c ON c.customer_id = o.customer_id
JOIN delivery_point dp ON dp.delivery_point_id = o.delivery_point_id
JOIN okres ok ON ok.okres_id = dp.okres_id
LEFT JOIN area_okres ao ON ao.okres_id = dp.okres_id
LEFT JOIN store st ON st.store_id = dp.store_id
JOIN sku s ON s.sku_id = l.sku_id
JOIN product p ON p.product_id = s.product_id
JOIN product_line pl ON pl.product_line_id = p.product_line_id
CROSS JOIN LATERAL (
    SELECT
        ROUND(l.list_price_case * l.qty_cases, 2) AS gross_revenue,
        ROUND(COALESCE(l.delivery_cost_case, 0) * l.qty_cases, 2) AS delivery_cost,
        ROUND(COALESCE(l.warehouse_cost_case, 0) * l.qty_cases, 2) AS warehouse_cost,
        ROUND((l.purchase_price_case + l.inbound_cost_case) * l.qty_cases, 2) AS cogs,
        ROUND(COALESCE(l.marketing_cost_case, 0) * l.qty_cases, 2) AS marketing_cost
) m
CROSS JOIN LATERAL (
    SELECT
        COALESCE(SUM(x.amount_line) FILTER (WHERE x.kind = 'volume'), 0) AS discount_volume,
        COALESCE(SUM(x.amount_line) FILTER (WHERE x.kind = 'cross_sell'), 0) AS discount_cross_sell,
        COALESCE(SUM(x.amount_line) FILTER (WHERE x.kind = 'agreed'), 0) AS discount_agreed,
        COALESCE(SUM(x.amount_line) FILTER (WHERE x.kind IN ('individual', 'retro')), 0) AS discount_individual,
        COALESCE(SUM(x.amount_line) FILTER (WHERE x.kind = 'promotion'), 0) AS discount_promotion,
        COALESCE(SUM(x.amount_line) FILTER (WHERE x.kind = 'free_goods'), 0) AS free_goods_value,
        COALESCE(SUM(x.amount_line), 0) AS total
    FROM sales_order_line_discount x
    WHERE x.sales_order_line_id = l.sales_order_line_id
) d
LEFT JOIN LATERAL (
    SELECT (MAX(sh.delivered_at) AT TIME ZONE 'Europe/Prague')::date AS delivered_on
    FROM shipment sh
    WHERE sh.sales_order_id = o.sales_order_id
) delivered ON true
LEFT JOIN LATERAL (
    SELECT a.employee_id
    FROM assignment a
    WHERE a.kind = 'rep_area' AND a.area_id = st.area_id AND o.ordered_on BETWEEN a.valid_from AND a.valid_to
    ORDER BY a.valid_from DESC
    LIMIT 1
) rep ON true
LEFT JOIN LATERAL (
    SELECT a.employee_id
    FROM assignment a
    WHERE a.kind = 'kam_customer' AND a.customer_id = o.customer_id
      AND o.ordered_on BETWEEN a.valid_from AND a.valid_to
    ORDER BY a.valid_from DESC
    LIMIT 1
) kam ON true
