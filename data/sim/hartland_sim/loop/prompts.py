"""Step 2's rep prompts (G2 S2.2, `05` §3 step 2): the rule-based `rep_prompt`s of a visit plan — generated before its
visit, cut to the customer's lifecycle phase, classified after it — and the purchase history they read.

**The history** (`Purchases`, `World.purchases`): for each small-retail customer (the reps' customers, the tiered
channel) the last day it bought each SKU, and its cases of the mandatory promotions' measured SKUs over the rules'
period; the channel's net revenue of each SKU by month, its promotional sales left out (R-7 step 4). The orders step
books every order into it (`bought`), the warm-up's too, so 2024-01-01's prompts read 2023's purchases. On a month's
first prompt day (`refresh`) it takes the month's figures: each SKU's net revenue per buying point a month over the
twelve months before (the revenue ÷ the customers who bought it in `lookback_days`, 0 when two or fewer did), and
each store cluster's (`04` S4's street types) share of its active customers who bought it in `lookback_days` (the AI
tip). It is pickled with the `World`, so a prolong reads what a straight build does.

**The prompts** (`generate`, on a written day: the warm-up makes none, so the memory starts with the data) for each
planned visit (`visit_plan`) to a small-retail customer or potential one in a phase with a cap — never a key account's
station, whose rep merchandises — by `world.yaml policies.prompt_rules`:
- `mandatory_promotion` (R-6, PO-POVINNA-AKCE): a running mandatory promotion whose eligibility admits the customer
  (`pricing.admits`) and whose measured SKUs — its condition SKU, else its target SKU, else its line's — the customer
  bought less of over `period_days` than `below_share_of_threshold` × its `condition_qty` (one case without one): one
  prompt a promotion;
- `unsupplied_items` (R-7, PO-NEODEBIRANE), T2's steps in their order: the SKUs the customer's listing group lists on
  the day and it has not bought in `lookback_days` (`listed_not_bought`), by product (`group_by_product`), by
  importance (`order_by_importance`), within an equal one by the average revenue per buying point
  (`tie_break_by_avg_revenue_per_buying_point`), without the products rejected in the last `rejection_memory_days`
  and the lines it rejected `stop_after_rejections` times (`drop_rejected`, A-13), without the lines excluded that
  month (`drop_seasonally_excluded`): one prompt for each of the first `max_products` products, with up to
  `max_items` of its SKUs;
- `ai_tip` (R-7's "AI tip", PO-TIP): of what is left after those drops, the first SKU in that order of `importance`
  that `peer_share` of the store's cluster bought, of a line no listing prompt offers; flagged `ai_tip`;
- `pos_placement` (R-2, PO-STOJAN): the store's line with the largest `World.display_uplift`, at least
  `min_display_uplift`, while its region's manager has display budget (`World.display_budget`) — both G2 S2.3's,
  empty until then, so S2.2 generates none;
- the debt and competitor-response rules generate nothing yet: the invoices are G4's, the counter-measures H-D's.
Each has a JSON `payload` (keys sorted; the SKUs by catalogue code, the line, the product or the promotion, the
figures it was chosen on) and a Czech `explanation`. A visit's set is ranked by priority, then the kinds' order and
each kind's own; the phase's cap (`policies.prompt_caps`, the contact-frequency `lifecycle_rule`'s
`prompts_per_visit`) shows the first ones, and every mandatory one whatever the cap; the rest stay `open`, with
their priority, for H-A's "more of the same theme".

**The classification** (`classify`, after the visit and its order; stream `prompt_outcome` by the prompt): a planned
visit not made leaves its prompts open, with no visit; a made one gives them its `visit_id`. Its prompts beyond the
cap stay open, and so do all of an unrecorded visit's (A-8). A visit that dealt with no one (closed, refused, gone)
discussed nothing, and one shorter than `prompt_outcomes.short_visit_minutes` by its recorded times (as they are, a
compressed day's too) only those of `short_visit_priority` or more urgent: the others are `not_discussed`. A
discussed prompt with one of its SKUs on the visit's order is `successful`; A-13's store rejects every listing prompt
of its line, with its reason; of the others the rep rejects `rejected` (a mandatory prompt cannot be: it is
unsuccessful instead) and logs `unsuccessful` ones, each with a reason of `policies.prompt_reasons`, and leaves the
rest open — it classifies only some (R-11). A classified prompt has the day as `classified_on`. A rejected listing
prompt is remembered: its product for `rejection_memory_days`, its line's rejections towards the stop.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np

from hartland_sim.config import PromptOutcomes, PromptReasons, PromptRule, StoriesConfig, WorldConfig
from hartland_sim.ids import row_id
from hartland_sim.io.reference import read
from hartland_sim.loop.demand import REFUSES, BasketReady, same
from hartland_sim.loop.field import UNREACHED, Listing, Planned, Visit, kind_of
from hartland_sim.loop.pricing import TIERED, Priced, Rules, admits, running
from hartland_sim.rng import uniforms
from hartland_sim.setup.rows import Row, row
from hartland_sim.world import Promotion, World

KINDS = ("mandatory_promotion", "debt", "unsupplied_items", "pos_placement", "competitor_response", "ai_tip")
Array = np.ndarray[Any, Any]


def month_index(day: date) -> int:
    return day.year * 12 + day.month - 1


# ── the history ────────────────────────────────────────────────────────────────


@dataclass(eq=False)
class Purchases:
    """The purchase history and the prompts' memory (module docstring): arrays and dicts filled in a fixed order."""

    rows: Array  # (C,) each customer's row here, in the World's order; −1 outside the tiered channel
    last_bought: Array  # (R, S) the ordinal of the day the customer last bought the SKU; 0 never
    recent: dict[int, list[tuple[int, int, float]]]  # customer_id → (ordinal, sku_id, cases) of its measured SKUs
    revenue: Array  # (12, S) the channel's net revenue by the month's slot (its index mod 12), promotions left out
    months: Array  # (12,) the month index (year × 12 + month − 1) a slot holds; −1 none
    average: Array  # (S,) net revenue per buying point a month, of `month` (R-7 step 4)
    clusters: tuple[str, ...]  # the street types, `peers`' rows
    peers: Array  # (K, S) a cluster's share of active customers who bought the SKU, of `month`
    rejected: dict[int, dict[str, int]]  # customer_id → a product's code → the ordinal of its last rejection
    rejections: dict[int, dict[str, int]]  # customer_id → a line's code → its listing prompts rejected
    month: int = -1  # the month index `average` and `peers` are of

    def __eq__(self, other: object) -> bool:
        return same(self, other)

    __hash__ = None  # type: ignore[assignment]

    @classmethod
    def of(cls, world: World) -> Purchases:
        """The opening's: nothing bought yet."""
        demand = world.demand
        if demand is None:
            raise ValueError("the World has no demand state (setup makes it)")
        tiered = [c for c, customer in enumerate(world.customers.values()) if customer.channel in TIERED]
        rows = np.full(len(world.customers), -1, dtype=np.int64)
        rows[tiered] = np.arange(len(tiered))
        skus = len(demand.catalogue.sku_ids)
        return cls(
            rows,
            np.zeros((len(tiered), skus), dtype=np.int64),
            {},
            np.zeros((12, skus)),
            np.full(12, -1, dtype=np.int64),
            np.zeros(skus),
            (),
            np.zeros((0, skus)),
            {},
            {},
        )

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state.pop("_measured", None)
        return state


def measured(promotion: Promotion, rules: Rules) -> tuple[int, ...]:
    """The SKUs (sku_id) whose purchase a mandatory promotion's prompt measures: its condition SKU, else its target
    SKU, else its line's."""
    code = promotion.condition_sku or promotion.target_sku
    if code is not None:
        return (rules.sku_ids[code],)
    return tuple(sku for sku, line in rules.sku_line.items() if line == promotion.line)


def _measured(world: World, purchases: Purchases, rules: Rules) -> frozenset[int]:
    """Every mandatory promotion's measured SKUs, kept until `World.promotions` is another dict or length."""
    held = purchases.__dict__.get("_measured")
    if held is None or held[0] is not world.promotions or held[1] != len(world.promotions):
        found = frozenset(sku for p in world.promotions.values() if p.mandatory for sku in measured(p, rules))
        held = purchases.__dict__["_measured"] = (world.promotions, len(world.promotions), found)
    return held[2]


def period_days(config: WorldConfig) -> int:
    """The longest period a mandatory-promotion rule measures over: what `Purchases.recent` keeps."""
    return max(
        (int(rule.parameters["period_days"]) for rule in config.policies.prompt_rules if rule.kind == KINDS[0]),
        default=0,
    )


def bought(world: World, rules: Rules, customer_id: int, day: date, priced: Sequence[Priced]) -> None:
    """An order of `customer_id` on `day`, its lines `priced`, into the history (module docstring); a customer
    outside the tiered channel has none. The free lines are no purchase."""
    purchases, demand = world.purchases, world.demand
    if purchases is None or demand is None:
        return
    r = int(purchases.rows[rules.customers[customer_id]])
    if r < 0:
        return
    today, month = day.toordinal(), month_index(day)
    slot = month % 12
    if purchases.months[slot] != month:
        purchases.revenue[slot] = 0.0
        purchases.months[slot] = month
    watched = _measured(world, purchases, rules)
    column = sku_columns(demand.catalogue.sku_ids)
    kept = [held for held in purchases.recent.get(customer_id, ()) if held[0] > today - period_days(rules.world)]
    for line in priced:
        if line.is_free_goods:
            continue
        s = column[line.sku_id]
        purchases.last_bought[r, s] = today
        if not any(discount.kind == "promotion" for discount in line.discounts):
            purchases.revenue[slot, s] += line.invoice_price_case * line.cases
        if line.sku_id in watched:
            kept.append((today, line.sku_id, line.cases))
    if kept:
        purchases.recent[customer_id] = kept
    else:
        purchases.recent.pop(customer_id, None)


_COLUMNS: list[tuple[Array, dict[int, int]]] = []  # the last catalogue's: a run reads one


def sku_columns(sku_ids: Array) -> dict[int, int]:
    """sku_id → its column in the catalogue's order, kept while the catalogue's array is the same."""
    if not _COLUMNS or _COLUMNS[0][0] is not sku_ids:
        _COLUMNS[:] = [(sku_ids, {int(sku): s for s, sku in enumerate(sku_ids)})]
    return _COLUMNS[0][1]


def refresh(world: World, book: Rulebook, day: date) -> None:
    """The month's figures (module docstring), once a month: on its first day that asks."""
    purchases, demand = world.purchases, world.demand
    if purchases is None or demand is None:
        return
    month = month_index(day)
    if purchases.month == month:
        return
    today = day.toordinal()
    held = purchases.last_bought >= today - book.lookback  # (R, S); never bought is 0
    buyers = held.sum(axis=0)
    slots = np.isin(purchases.months, [month - k for k in range(1, 13)])
    revenue = purchases.revenue[slots].sum(axis=0)
    purchases.average = np.where(buyers > 2, revenue / np.maximum(buyers, 1) / 12, 0.0)
    customers = np.flatnonzero(purchases.rows >= 0)
    streets = []
    for c in customers.tolist():
        store = demand.spots[int(demand.customer_start[c])].store
        streets.append(store.street_type if store is not None else "")
    clusters = tuple(sorted(set(streets)))
    street = np.array([clusters.index(s) for s in streets], dtype=np.int64)
    active = (demand.customer_since[customers] <= today) & (today <= demand.customer_until[customers])
    peers = np.zeros((len(clusters), held.shape[1]))
    for k in range(len(clusters)):
        mine = (street == k) & active
        if mine.any():
            peers[k] = held[purchases.rows[customers[mine]]].mean(axis=0)
    purchases.clusters, purchases.peers, purchases.month = clusters, peers, month


# ── the rules ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Product:
    code: str
    name: str  # Czech, `product.csv`
    line: int  # its line's index


@dataclass(frozen=True)
class Rulebook:
    """What the prompts read beside the `World`, for a run (never pickled; built again from the configs): the rules by
    kind with their row ids, the caps by phase, how the reps classify, the catalogue's SKUs and products, the lines'
    seasonal exclusions, the customers' and points' indices, A-13 while enabled."""

    rules: Mapping[str, PromptRule]  # by kind
    rule_ids: Mapping[str, int]  # kind → prompt_rule_id
    caps: Mapping[str, int]  # phase → prompts shown
    outcomes: PromptOutcomes
    reasons: PromptReasons
    customers: Mapping[int, int]  # customer_id → its index in the World's order
    points: Mapping[int, int]  # delivery_point_id → its index in the demand's points
    sku_codes: tuple[str, ...]  # catalogue's order
    sku_names: tuple[str, ...]
    sku_product: tuple[int, ...]  # the product's index, by SKU column
    importance: tuple[int, ...]  # 10/5/1/0, by SKU column
    products: tuple[Product, ...]
    line_codes: tuple[str, ...]
    line_names: tuple[str, ...]
    excluded: Mapping[int, frozenset[int]]  # month → the lines (indices) excluded that month (R-7)
    a13: tuple[int, str] | None  # A-13's line (its index) and reason

    @classmethod
    def of(cls, world: World, config: WorldConfig, stories: StoriesConfig) -> Rulebook:
        demand = world.demand
        if demand is None:
            raise ValueError("the World has no demand state (setup makes it)")
        start = config.history.warm_up_from
        rules = {rule.kind: rule for rule in config.policies.prompt_rules}
        rule_ids = {rule.kind: row_id("prompt_rule", code=rule.code, valid_from=start) for rule in rules.values()}
        line_codes = tuple(line.code for line in demand.lines)
        line_index = {code: i for i, code in enumerate(line_codes)}
        names = {line["code"]: line["name_cs"] for line in read("product_line")}
        products = tuple(Product(p["code"], p["name_cs"], line_index[p["product_line"]]) for p in read("product"))
        product_index = {product.code: k for k, product in enumerate(products)}
        catalogue = {sku["sku_code"]: sku for sku in read("catalogue")}
        goods = demand.catalogue
        excluded = {
            month: frozenset(
                line_index[code] for code, months in config.policies.seasonal_exclusion.items() if month in months
            )
            for month in range(1, 13)
        }
        a13 = stories.anomaly("A-13")
        return cls(
            rules,
            rule_ids,
            {str(phase): cap for phase, cap in config.policies.prompt_caps.items()},
            config.policies.prompt_outcomes,
            config.policies.prompt_reasons,
            {int(customer): c for c, customer in enumerate(demand.customer_ids)},
            {int(point): p for p, point in enumerate(demand.point_ids)},
            tuple(goods.sku_codes),
            tuple(catalogue[code]["name_cs"] for code in goods.sku_codes),
            tuple(product_index[catalogue[code]["product"]] for code in goods.sku_codes),
            tuple(int(weight) - 1 for weight in goods.weight),
            products,
            line_codes,
            tuple(names[code] for code in line_codes),
            excluded,
            (line_index[str(a13.parameters["line"])], str(a13.parameters["reason_cs"])) if a13.enabled else None,
        )

    def parameter(self, kind: str, name: str) -> Any:
        return self.rules[kind].parameters[name]

    @property
    def lookback(self) -> int:
        return int(self.parameter("unsupplied_items", "lookback_days"))


# ── the prompts ────────────────────────────────────────────────────────────────


@dataclass
class Prompt:
    """A `rep_prompt` as the day makes it: generated for a planned visit, then classified (module docstring)."""

    rep_prompt_id: int
    visit_plan_id: int
    employee_id: int
    customer_id: int
    kind: str
    prompt_rule_id: int
    priority: int
    shown: bool  # within the cap
    skus: tuple[int, ...]  # the sku_ids it offers; none for a display
    line: str  # its line's code
    product: str | None  # the product's code (a listing prompt, a tip)
    payload: str
    explanation: str
    visit_id: int | None = None
    outcome: str = "open"
    reason: str | None = None
    classified_on: date | None = None


@dataclass(frozen=True)
class Draft:
    """A prompt before its set is ranked: what it offers and says, its order within its kind."""

    kind: str
    skus: tuple[int, ...]  # SKU columns
    line: int
    product: int | None  # the product's index
    payload: dict[str, Any]
    explanation: str


def czk(amount: float) -> str:
    """Whole crowns, thousands apart by a space: 1 800."""
    return f"{round(amount):,}".replace(",", " ")


def cases(amount: float) -> str:
    """Cases as Czech writes them: 4, or 2,5."""
    return f"{amount:.0f}" if float(amount).is_integer() else f"{amount:.1f}".replace(".", ",")


Offer = tuple[int, tuple[int, ...]]  # a product (its index) and its SKUs (columns), in the order offered


def listed_not_bought(listed: Array, last_bought: Array, day: date, lookback: int) -> Array:
    """(S,) bool: the SKUs `listed` (the customer's listing group's on `day`) that its `last_bought` (ordinals, 0
    never) puts before the last `lookback` days (R-7 step 1)."""
    return np.asarray(listed & (last_bought < day.toordinal() - lookback))


def group_by_product(candidates: Array, sku_product: Sequence[int]) -> list[Offer]:
    """The `candidates` (S,) by product, the products and their SKUs in the catalogue's order (R-7 step 2)."""
    out: dict[int, list[int]] = {}
    for s in np.flatnonzero(candidates).tolist():
        out.setdefault(sku_product[s], []).append(s)
    return [(k, tuple(skus)) for k, skus in out.items()]


def order_by_importance(offers: Sequence[Offer], importance: Sequence[int]) -> list[Offer]:
    """Each product's SKUs by importance (10/5/1/0), then the products by their first SKU's (R-7 step 3); stable."""
    out = [(k, tuple(sorted(skus, key=lambda s: -importance[s]))) for k, skus in offers]
    return sorted(out, key=lambda offer: -importance[offer[1][0]])


def tie_break_by_avg_revenue_per_buying_point(
    offers: Sequence[Offer], importance: Sequence[int], average: Sequence[float]
) -> list[Offer]:
    """Within an equal importance, by the channel's average revenue per buying point (R-7 step 4: `Purchases.average`),
    the SKUs of a product and the products alike; stable."""

    def key(s: int) -> tuple[float, float]:
        return (-importance[s], -average[s])

    out = [(k, tuple(sorted(skus, key=key))) for k, skus in offers]
    return sorted(out, key=lambda offer: key(offer[1][0]))


def drop_rejected(
    offers: Sequence[Offer],
    book: Rulebook,
    rejected: Mapping[str, int],
    rejections: Mapping[str, int],
    day: date,
) -> list[Offer]:
    """Without the products the customer rejected (`rejected`: a product's code → the day, an ordinal) in the last
    `rejection_memory_days` and those of a line it rejected (`rejections`: a line's code → how often)
    `stop_after_rejections` times (R-7 step 6, A-13)."""
    memory = int(book.parameter("unsupplied_items", "rejection_memory_days"))
    stop = int(book.parameter("unsupplied_items", "stop_after_rejections"))
    today = day.toordinal()
    out = []
    for k, skus in offers:
        product = book.products[k]
        if rejections.get(book.line_codes[product.line], 0) >= stop:
            continue
        if product.code in rejected and today - rejected[product.code] < memory:
            continue
        out.append((k, skus))
    return out


def drop_seasonally_excluded(offers: Sequence[Offer], book: Rulebook, day: date) -> list[Offer]:
    """Without the products of the lines excluded in `day`'s month (R-7 step 5, `policies.seasonal_exclusion`)."""
    excluded = book.excluded[day.month]
    return [(k, skus) for k, skus in offers if book.products[k].line not in excluded]


@dataclass(frozen=True)
class Briefing:
    """What a day's prompts read once (`generate`): the day, what each listing group lists, the mandatory promotions
    running, the month's averages (`Purchases.average`) as a list."""

    day: date
    live: Array  # (G, S)
    mandatory: tuple[tuple[int, Promotion], ...]
    average: list[float]


def unsupplied(
    world: World, book: Rulebook, briefing: Briefing, row: int, group: int, customer_id: int
) -> tuple[list[Draft], list[Offer]]:
    """R-7 for one customer (module docstring), T2's steps in their order: its listing prompts, and every product left
    after the drops (the tip reads them)."""
    purchases = world.purchases
    assert purchases is not None
    day, last = briefing.day, purchases.last_bought[row]
    offers = group_by_product(listed_not_bought(briefing.live[group], last, day, book.lookback), book.sku_product)
    offers = order_by_importance(offers, book.importance)
    offers = tie_break_by_avg_revenue_per_buying_point(offers, book.importance, briefing.average)
    rejected, rejections = purchases.rejected.get(customer_id, {}), purchases.rejections.get(customer_id, {})
    left = drop_seasonally_excluded(drop_rejected(offers, book, rejected, rejections, day), book, day)
    out = []
    for k, skus in left[: int(book.parameter("unsupplied_items", "max_products"))]:
        product = book.products[k]
        offered = skus[: int(book.parameter("unsupplied_items", "max_items"))]
        reasons = [
            {
                "sku": book.sku_codes[s],
                "importance": int(book.importance[s]),
                "avg_revenue_czk_month": round(float(purchases.average[s]), 2),
                "last_bought": date.fromordinal(int(last[s])).isoformat() if last[s] else None,
            }
            for s in offered
        ]
        best = offered[0]
        said = f"Nabídněte {product.name}: v listingu kanálu, neodebíráno {book.lookback} dní"
        said += f", důležitost {book.importance[best]}"
        if purchases.average[best] > 0:
            said += f", srovnatelné prodejny odebírají v průměru {czk(purchases.average[best])} Kč/měsíc"
        payload = {
            "line": book.line_codes[product.line],
            "product": product.code,
            "skus": [book.sku_codes[s] for s in offered],
            "reasons": reasons,
        }
        out.append(Draft("unsupplied_items", offered, product.line, k, payload, said + "."))
    return out, left


def tip(world: World, book: Rulebook, street: str, left: Sequence[Offer], offered: Sequence[Draft]) -> list[Draft]:
    """The AI tip (module docstring) among the products `left` after R-7's drops, by the cluster of the store (its
    `street` type): the first SKU in their order."""
    purchases = world.purchases
    assert purchases is not None
    if street not in purchases.clusters:
        return []
    importance = int(book.parameter("ai_tip", "importance"))
    share = float(book.parameter("ai_tip", "peer_share"))
    peers = purchases.peers[purchases.clusters.index(street)]
    lines = {draft.line for draft in offered}
    for k, skus in left:
        if book.products[k].line in lines:
            continue
        for s in skus:
            if book.importance[s] != importance or peers[s] < share:
                continue
            product = book.products[k]
            payload = {
                "ai_tip": True,
                "line": book.line_codes[product.line],
                "product": product.code,
                "skus": [book.sku_codes[s]],
                "peer_share": round(float(peers[s]), 3),
            }
            said = (
                f"Tip mimo pravidla: {book.sku_names[s]} – podíl podobných prodejen, které výrobek za poslední rok "
                f"odebíraly: {round(100 * float(peers[s]))} %."
            )
            return [Draft("ai_tip", (s,), product.line, k, payload, said)]
    return []


def mandatory(
    world: World, book: Rulebook, rules: Rules, briefing: Briefing, customer_id: int, point: int
) -> list[Draft]:
    """R-6 for one customer (module docstring): a prompt for each running mandatory promotion it qualifies for."""
    purchases, demand = world.purchases, world.demand
    assert purchases is not None and demand is not None
    if not briefing.mandatory:
        return []
    day = briefing.day
    period = int(book.parameter("mandatory_promotion", "period_days"))
    below = float(book.parameter("mandatory_promotion", "below_share_of_threshold"))
    customer, spot = world.customers[customer_id], demand.spots[point]
    held = [(sku, n) for ordinal, sku, n in purchases.recent.get(customer_id, ()) if ordinal > day.toordinal() - period]
    column = sku_columns(demand.catalogue.sku_ids)
    out = []
    for _, promotion in briefing.mandatory:
        if not admits(promotion, spot, customer, day):
            continue
        skus = measured(promotion, rules)
        threshold = float(promotion.condition_qty or 1)
        got = float(sum(n for sku, n in held if sku in skus))
        if got >= below * threshold:
            continue
        line = book.line_codes.index(promotion.line)
        what = book.sku_names[column[skus[0]]] if len(skus) == 1 else book.line_names[line]
        payload = {
            "promotion": promotion.code,
            "line": promotion.line,
            "skus": [book.sku_codes[column[sku]] for sku in skus],
            "bought_cases": round(got, 3),
            "threshold_cases": threshold,
            "period_days": period,
        }
        said = (
            f"Povinná akce {promotion.name or promotion.code}: nabídněte ji. Za posledních {period} dní odebráno "
            f"{cases(got)} krt. ({what}), akce počítá s {cases(threshold)} krt."
        )
        out.append(Draft("mandatory_promotion", tuple(column[sku] for sku in skus), line, None, payload, said))
    return out


def display(world: World, book: Rulebook, delivery_point_id: int, region: str) -> list[Draft]:
    """R-2's placement (module docstring): the store's line with the largest display uplift over the bar, while its
    region has display budget."""
    if world.display_budget.get(region, 0.0) <= 0:
        return []
    bar = float(book.parameter("pos_placement", "min_display_uplift"))
    found = [
        (uplift, line)
        for (point, line), uplift in world.display_uplift.items()
        if point == delivery_point_id and uplift >= bar
    ]
    if not found:
        return []
    uplift, code = max(found)
    line = book.line_codes.index(code)
    payload = {"line": code, "display_uplift": round(uplift, 3)}
    said = f"Navrhněte stojan na {book.line_names[line]}: odhadovaný nárůst prodeje {round(100 * uplift)} %."
    return [Draft("pos_placement", (), line, None, payload, said)]


def generate(world: World, book: Rulebook, rules: Rules, day: date, planned: Sequence[Planned]) -> list[Prompt]:
    """The day's prompts for `planned` (module docstring), each visit's set ranked and cut to its phase's cap."""
    purchases, demand = world.purchases, world.demand
    if purchases is None or demand is None or not planned:
        return []
    refresh(world, book, day)
    briefing = Briefing(
        day,
        demand.catalogue.live(day),
        tuple((pid, p) for pid, p in running(world, day) if p.mandatory and "mandatory_promotion" in book.rules),
        purchases.average.tolist(),
    )
    out: list[Prompt] = []
    for p in planned:
        t = p.tourplan
        customer = world.customers[t.customer_id]
        cap = book.caps.get(customer.phase_on(day), 0)
        row = int(purchases.rows[book.customers[t.customer_id]])
        if row < 0 or not cap or kind_of(customer, day) == "merchandising":
            continue
        point = book.points[t.delivery_point_id]
        drafts = mandatory(world, book, rules, briefing, t.customer_id, point)
        listing: list[Draft] = []
        left: list[Offer] = []
        if "unsupplied_items" in book.rules:
            listing, left = unsupplied(world, book, briefing, row, int(demand.point_group[point]), t.customer_id)
        drafts += listing
        if "pos_placement" in book.rules:
            drafts += display(world, book, t.delivery_point_id, demand.spots[point].region)
        if "ai_tip" in book.rules:
            store = demand.spots[point].store
            drafts += tip(world, book, store.street_type if store else "", left, listing)
        drafts.sort(key=lambda d: (book.rules[d.kind].priority, KINDS.index(d.kind)))  # stable: each kind's own order
        seq: dict[str, int] = {}
        for rank, d in enumerate(drafts, start=1):
            seq[d.kind] = seq.get(d.kind, 0) + 1
            rule = book.rules[d.kind]
            key = {"employee": t.employee, "customer": customer.code, "generated_on": day, "kind": d.kind}
            out.append(
                Prompt(
                    row_id("rep_prompt", **key, seq=seq[d.kind]),
                    p.visit_plan_id,
                    t.employee_id,
                    t.customer_id,
                    d.kind,
                    book.rule_ids[d.kind],
                    rule.priority,
                    rule.mandatory or rank <= cap,
                    tuple(int(demand.catalogue.sku_ids[s]) for s in d.skus),
                    book.line_codes[d.line],
                    book.products[d.product].code if d.product is not None else None,
                    json.dumps(d.payload, ensure_ascii=False, sort_keys=True),
                    d.explanation,
                )
            )
    return out


# ── the classification ─────────────────────────────────────────────────────────


def _minutes(v: Visit) -> float | None:
    if v.time_in is None or v.time_out is None:
        return None
    return (v.time_out - v.time_in) / timedelta(minutes=1)


def classify(
    world: World,
    book: Rulebook,
    day: date,
    prompts: Sequence[Prompt],
    visits: Sequence[Visit],
    events: Sequence[BasketReady],
) -> dict[int, Listing]:
    """The day's prompts after their visits and orders (module docstring); what each visit's note says of its listing
    prompts, by `visit_id`: A-13's refusal, else the first one classified that the rep offered (not its own
    rejection)."""
    if not prompts:
        return {}
    purchases = world.purchases
    assert purchases is not None
    made = {v.visit_plan_id: v for v in visits if v.visit_plan_id is not None}
    ordered: dict[int, set[int]] = {}
    for event in events:
        if event.visit is not None:
            ordered[event.visit[0]] = {sku for sku, _, _ in event.lines}
    ids = np.array([p.rep_prompt_id for p in prompts], dtype=np.int64)
    draws = uniforms("prompt_outcome", "outcome", ids=ids).tolist()
    picks = uniforms("prompt_outcome", "reason", ids=ids).tolist()
    odds, short = book.outcomes, book.outcomes.short_visit_minutes
    names = {product.code: product.name for product in book.products}
    listings: dict[int, Listing] = {}
    for p, draw, pick in zip(prompts, draws, picks, strict=True):
        v = made.get(p.visit_plan_id)
        if v is None:
            continue  # the visit was not made
        p.visit_id = v.visit_id
        if not p.shown or not v.recorded:
            continue
        minutes = _minutes(v)
        if v.outcome in UNREACHED or (
            minutes is not None and minutes < short and p.priority > odds.short_visit_priority
        ):
            p.outcome, p.classified_on = "not_discussed", day
            continue
        customer = world.customers[p.customer_id]
        planted = (
            book.a13 is not None
            and REFUSES in customer.traits
            and p.kind == "unsupplied_items"
            and p.line == book.line_codes[book.a13[0]]
        )
        if set(p.skus) & ordered.get(v.visit_id, set()):
            p.outcome = "successful"
        elif planted:
            assert book.a13 is not None
            p.outcome, p.reason = "rejected", book.a13[1]
        elif draw < odds.rejected and not book.rules[p.kind].mandatory:
            p.outcome, p.reason = "rejected", _pick(book.reasons.rejected, pick)
        elif draw < odds.rejected + odds.unsuccessful:
            p.outcome, p.reason = "unsuccessful", _pick(book.reasons.unsuccessful, pick)
        else:
            continue  # the rep left it unclassified
        p.classified_on = day
        if p.outcome == "rejected" and p.kind == "unsupplied_items":
            remember(world, p, day)
        if p.kind == "unsupplied_items" and p.product:
            said = listings.get(v.visit_id)
            first = said is None and p.outcome != "rejected"  # a rep's own rejection offered nothing to say
            if first or (planted and not (said and said.planted)):  # A-13's refusal is the one said
                listings[v.visit_id] = Listing(names[p.product], p.outcome == "successful", p.reason, planted)
    return listings


def _pick(reasons: Sequence[str], draw: float) -> str:
    return reasons[min(int(draw * len(reasons)), len(reasons) - 1)]


def remember(world: World, p: Prompt, day: date) -> None:
    """A rejected listing prompt (module docstring): its product's day, its line's count."""
    purchases = world.purchases
    assert purchases is not None
    assert p.product is not None
    purchases.rejected.setdefault(p.customer_id, {})[p.product] = day.toordinal()
    held = purchases.rejections.setdefault(p.customer_id, {})
    held[p.line] = held.get(p.line, 0) + 1


def prompt_row(p: Prompt, day: date) -> Row:
    return row(
        "rep_prompt",
        rep_prompt_id=p.rep_prompt_id,
        employee_id=p.employee_id,
        customer_id=p.customer_id,
        visit_id=p.visit_id,
        generated_on=day,
        kind=p.kind,
        prompt_rule_id=p.prompt_rule_id,
        priority=p.priority,
        payload=p.payload,
        explanation=p.explanation,
        outcome=p.outcome,
        outcome_reason=p.reason,
        classified_on=p.classified_on,
        generated_by="rules",
    )
