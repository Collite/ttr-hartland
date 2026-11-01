-- A-13 — the rejected prompt pattern: customers who rejected three unsupplied-items prompts naming the noodle line and
-- got no noodle prompt after the third rejection: the prompt that brought the running count of rejections to 3 is
-- their last one. A customer who goes on getting (and rejecting) them is not the pattern.
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
    HAVING COUNT(*) FILTER (WHERE rejected_so_far >= 3) = 1
) stopped
