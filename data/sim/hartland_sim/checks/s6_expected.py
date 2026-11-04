"""S6 on paper (G1 S1.5 T2, Bora's S6 rulings 11-04): what `s6.sql`'s measure should give a granter of a discipline,
from `s6.yaml` and the lines' caps alone — no build. `elasticity.high` is derived from it (`test_s6_expected`), never
from a build; G5 S5.2 re-checks it on G4's cost to serve (the margin share).

A granter answers asks on every line of `pricing.max_agreed_discount_pct`, one weight a line at its full cap (the
concession share 1), on customer-lines of the high elasticity at `high_share`, else the low. It grants with `grant`'s
chance and sizes as `grant` does, at the mean of its range: from `disciplined_from` the mean of
`disciplined_share_of_max` × the cap, below it the mean of `undisciplined_discount_pct`, at most the cap. An agreement
lifts the line's demand by 1 + e × d for its `LIFTED` days, and an order at the small-retail cadence brings
`AFTER_START` days of unlifted demand bought at the discount (the first order inside it carries the days since the
last). Per cell, in the line's margin at list a day: crowns ∝ d × (L × (1 + e × d) + U), the gain ∝ L × e × d × m −
the crowns (the lifted cases' margin, the margin share m of the price, less what the discount paid); the ratio is the
cells' gain over their crowns, each weighted by its share and its chance — crown-weighted, as `s6.sql` sums them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hartland_sim.loop.pricing import chance

LIFTED = 45  # the agreement's days of lifted demand: from mid-quarter to its end, on average
AFTER_START = 7  # the days of unlifted demand the first order inside it buys at the discount: half a fortnight


def expected_ratio(discipline: float, margin_share: float, s6: Mapping[str, Any], caps: Mapping[str, float]) -> float:
    """The expected incremental Customer Margin per agreed-discount crown of a granter of `discipline` (module
    docstring), its Customer Margin `margin_share` of the price; `s6` is `s6.yaml`'s parameters, `caps` the lines'
    `max_agreed_discount_pct`."""
    spec = s6["elasticity"]
    mix = ((spec["high"], spec["high_share"]), (spec["low"], 1 - spec["high_share"]))
    gain = crowns = 0.0
    for cap in caps.values():
        if discipline >= s6["disciplined_from"]:
            pct = cap * sum(s6["disciplined_share_of_max"]) / 2
        else:
            pct = min(sum(s6["undisciplined_discount_pct"]) / 2, cap)
        d = pct / 100
        for e, share in mix:
            weight = share * chance(e, discipline, s6)
            paid = d * (LIFTED * (1 + e * d) + AFTER_START)
            gain += weight * (LIFTED * e * d * margin_share - paid)
            crowns += weight * paid
    return gain / crowns
