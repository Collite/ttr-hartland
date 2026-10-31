-- A-13 — the rejected prompt pattern: customers who rejected three unsupplied-items prompts naming the noodle line
-- and got none after the third.
WITH noodle AS (
    SELECT rp.customer_id, rp.generated_on, rp.outcome,
           COUNT(*) FILTER (WHERE rp.outcome = 'rejected') OVER (
               PARTITION BY rp.customer_id ORDER BY rp.generated_on, rp.rep_prompt_id
           ) AS rejected_so_far
    FROM rep_prompt rp
    WHERE rp.kind = 'unsupplied_items' AND rp.payload LIKE '%"NUDLE"%'
)
SELECT COUNT(*)
FROM (
    SELECT customer_id
    FROM noodle
    GROUP BY customer_id
    HAVING MAX(rejected_so_far) >= 3
       AND COUNT(*) FILTER (WHERE rejected_so_far >= 3 AND outcome <> 'rejected') = 0
) stopped
