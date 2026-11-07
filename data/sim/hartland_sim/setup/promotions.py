"""The promotion calendar from its templates (⚑NH-8): `data/reference/promotion_templates.yaml` holds ~25 shapes, one
promotion each a year, and a year's calendar is every template instantiated for that year — so the codes are stable
(`world.yaml promotions.code`: `PR-<year>-<month>-<line>-<tag>`) and a rebuild gives the same promotions. Setup seeds
the history's years from it into the `World` (`seeded`; `05` §2 step 6); `prolong` goes past the committed years the
same way (G5 S5.3).

A template starts on the Monday of its ISO `start_week` and runs `weeks` weeks; the start must fall in its `month`,
which the code carries. Exclusions pair templates by `<line>-<tag>`; each year's calendar excludes the pair both ways.
Two promotions of one line never stack on an order: where their days and their eligibility meet, they exclude each
other (`clashes` finds the pairs that do not).

**The rows** (`write`, `05` §2 step 6): the calendar's promotions that run on a written day or later, to the end of
the history's last year — December's that runs into the first written January among them — each with its campaign
(`campaign`: promotions hang under campaigns, one each, coded KP- for PR-; TV and outdoor when TV-supported, digital
when TV-supported or sold on the web, always in-store; its media budget `world.yaml promotions.campaign_budget`), its
scope rules (`promotion_eligibility`) and its exclusions both ways (`promotion_exclusion`). The product manager of
the line on its first day owns both. Its `estimated_budget` is what Marketing expects it to cost
(`estimated_budget`); its status is the one it has on the history's last day.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from hartland_sim.config import ConfigError, WorldConfig, build, read_yaml, validate
from hartland_sim.ids import row_id
from hartland_sim.io.reference import REFERENCE_DIR
from hartland_sim.loop.demand import Demand, eligible
from hartland_sim.setup.rows import Tables, row
from hartland_sim.world import Promotion, World

TEMPLATES_FILE = REFERENCE_DIR / "promotion_templates.yaml"


@dataclass(frozen=True)
class Template:
    line: str
    tag: str
    month: int
    start_week: int
    weeks: int
    mechanic: str
    target_uplift_pct: float
    eligibility: Mapping[str, tuple[str, ...]]  # scope kind (channel, chain, region, store_class, phase) → values
    name_cs: str
    goal_cs: str
    tv: bool = False
    discount_pct: float | None = None
    condition_sku: str | None = None
    condition_qty: int | None = None
    reward_sku: str | None = None
    reward_qty: int | None = None
    target_sku: str | None = None
    mandatory: bool = False
    pre_stocking: bool = False

    @property
    def key(self) -> str:
        return f"{self.line}-{self.tag}"


@dataclass(frozen=True)
class Templates:
    templates: tuple[Template, ...]
    exclusions: tuple[tuple[str, str], ...]

    @classmethod
    def load(cls, path: Path = TEMPLATES_FILE) -> Templates:
        data = read_yaml(path)
        validate(data, "promotion_templates", path.name)
        templates = build(cls, data)
        keys = [template.key for template in templates.templates]
        problems = [f"{path.name}: {key} twice" for key in sorted({k for k in keys if keys.count(k) > 1})]
        problems += [
            f"{path.name}: exclusion {pair} names no template"
            for pair in templates.exclusions
            if not set(pair) <= set(keys)
        ]
        if problems:
            raise ConfigError("\n".join(problems))
        return templates


@dataclass(frozen=True)
class PlannedPromotion:
    """A template in a year: its code and dates, and the codes it may not combine with."""

    code: str
    template: Template
    valid_from: date
    valid_to: date
    excludes: tuple[str, ...]


def calendar(templates: Templates, year: int, code: str) -> list[PlannedPromotion]:
    """The year's promotions in start order; `code` is `world.yaml promotions.code`."""
    codes = {t.key: code.format(year=year, month=t.month, line=t.line, tag=t.tag) for t in templates.templates}
    partners: dict[str, list[str]] = {}
    for first, second in templates.exclusions:
        partners.setdefault(first, []).append(codes[second])
        partners.setdefault(second, []).append(codes[first])
    planned = []
    for template in templates.templates:
        start = date.fromisocalendar(year, template.start_week, 1)
        if start.month != template.month:
            raise ValueError(f"{template.key}: week {template.start_week} of {year} starts in month {start.month}")
        end = start + timedelta(weeks=template.weeks) - timedelta(days=1)
        planned.append(
            PlannedPromotion(codes[template.key], template, start, end, tuple(sorted(partners.get(template.key, ()))))
        )
    return sorted(planned, key=lambda promotion: (promotion.valid_from, promotion.code))


def clashes(promotions: list[PlannedPromotion]) -> list[tuple[str, str]]:
    """The pairs of promotions of one line that share a day and an eligible order, without excluding each other."""
    found = []
    for first, second in itertools.combinations(promotions, 2):
        if (
            first.template.line == second.template.line
            and first.valid_from <= second.valid_to
            and second.valid_from <= first.valid_to
            and _may_meet(first.template, second.template)
            and second.code not in first.excludes
        ):
            found.append((first.code, second.code))
    return found


def _may_meet(first: Template, second: Template) -> bool:
    """An order could qualify for both: on every scope kind that both restrict, their values meet."""
    kinds = first.eligibility.keys() & second.eligibility.keys()
    return all(set(first.eligibility[kind]) & set(second.eligibility[kind]) for kind in kinds)


def seeded(templates: Templates, years: Iterable[int], code: str) -> dict[int, Promotion]:
    """The `World`'s promotions of `years` from the calendar, by `promotion_id`, in start order within each year: what
    the demand model lifts (S1.2) and the pricing prices (S1.3)."""
    out = {}
    for year in years:
        for planned in calendar(templates, year, code):
            template = planned.template
            promotion_id = row_id("promotion", code=planned.code)
            out[promotion_id] = Promotion(
                promotion_id=promotion_id,
                code=planned.code,
                line=template.line,
                mechanic=template.mechanic,
                valid_from=planned.valid_from,
                valid_to=planned.valid_to,
                tv=template.tv,
                eligibility=tuple((kind, tuple(values)) for kind, values in template.eligibility.items()),
                discount_pct=template.discount_pct,
                target_sku=template.target_sku,
                condition_sku=template.condition_sku,
                reward_sku=template.reward_sku,
                condition_qty=template.condition_qty,
                reward_qty=template.reward_qty,
                mandatory=template.mandatory,
                excludes=planned.excludes,
                name=template.name_cs,
            )
    return out


def status(promotion: Promotion, day: date) -> str:
    """A promotion's status on `day` (`status_of`)."""
    return status_of(promotion.valid_from, promotion.valid_to, day)


def status_of(valid_from: date, valid_to: date, day: date) -> str:
    """The status on `day` of a promotion running `valid_from`–`valid_to`: approved before it starts, running, closed
    after it ends (a build's end restatuses its rows by it, `loop.build.restatus`)."""
    if day < valid_from:
        return "approved"
    return "running" if day <= valid_to else "closed"


def estimated_budget(promotion: Promotion, template: Template, demand: Demand, phases: list[str] | None) -> float:
    """What Marketing expects `promotion` to cost, to the thousand: the expected gross a day of what it discounts or
    gives — a price-off's and a 1 + 1's line (or target SKU), a conditional one's reward SKU, a free-goods-other's
    condition SKU — at the points its eligibility admits whose customer buys then (the opening's customers, at flat
    terms, on its first day's listings and prices), over its days, lifted by its target uplift, × the share given
    away: the discount, m of n + m, or the reward's value per the condition's. `phases` are the customers' phases
    on its first day, in the `World`'s order, for a promotion with a phase scope."""
    goods = demand.catalogue
    first = promotion.valid_from
    share, _, _ = goods.shares(first, len(demand.lines))
    today = first.toordinal()
    buying = (demand.customer_since <= today) & (today <= demand.customer_until)
    points = eligible(promotion, demand, phases) & buying[demand.point_customer]
    if promotion.mechanic == "conditional_price_off":
        named = promotion.reward_sku
    elif promotion.mechanic == "free_goods_other":
        named = promotion.condition_sku
    else:
        named = promotion.target_sku
    codes = list(goods.sku_codes)
    if named is not None:
        skus = [codes.index(named)]
    else:
        skus = [s for s, line in enumerate(goods.line) if demand.lines[int(line)].code == promotion.line]
    groups = demand.point_group[points]
    carried = demand.carried[demand.point_customer[points]]
    daily = 0.0
    for s in skus:
        line = int(goods.line[s])
        daily += float((demand.base[points, line] * carried[:, line] * share[groups, s] * goods.price[groups, s]).sum())
    if promotion.mechanic in ("price_off", "conditional_price_off"):
        given = (promotion.discount_pct or 0) / 100
    elif promotion.mechanic == "free_goods_same":
        given = (promotion.reward_qty or 1) / ((promotion.condition_qty or 1) + (promotion.reward_qty or 1))
    else:
        reward, condition = codes.index(promotion.reward_sku or ""), codes.index(promotion.condition_sku or "")
        given = (
            (promotion.reward_qty or 0)
            * goods.price[0, reward]
            / ((promotion.condition_qty or 1) * goods.price[0, condition])
        )
    days = (promotion.valid_to - first).days + 1
    return float(round(daily * days * (1 + template.target_uplift_pct / 100) * given, -3))


def write(out: Tables, templates: Templates, state: World, world: WorldConfig) -> None:
    """The calendar's written promotions into `out` (module docstring); `state` is the opening `World`, whose
    promotions (`seeded`) and demand it reads."""
    demand = state.demand
    if demand is None:
        raise ValueError("the World has no demand state (setup makes it)")
    first = world.history.first_day
    years = range(world.history.warm_up_from.year, world.history.until.year + 1)
    planned = [p for year in years for p in calendar(templates, year, world.promotions.code) if p.valid_to >= first]
    written = {promotion.code for promotion in planned}
    end = world.history.until
    budget = world.promotions.campaign_budget
    for one in planned:
        template = one.template
        promotion_id = row_id("promotion", code=one.code)
        promotion = state.promotions[promotion_id]
        duty = state.duty("pm_line", template.line, one.valid_from)
        if duty is None:
            raise ValueError(f"{one.code}: no product manager holds {template.line} on {one.valid_from}")
        channels = dict(promotion.eligibility).get("channel", ())
        campaign = "KP-" + one.code.removeprefix("PR-")
        campaign_id = row_id("campaign", code=campaign)
        out["campaign"].append(
            row(
                "campaign",
                campaign_id=campaign_id,
                code=campaign,
                name=template.name_cs,
                product_line_id=row_id("product_line", code=template.line),
                valid_from=one.valid_from,
                valid_to=one.valid_to,
                has_tv=template.tv,
                has_outdoor=template.tv,
                has_digital=template.tv or "web" in channels,
                has_instore=True,
                budget=budget["tv" if template.tv else "other"],
                owner_employee_id=duty.employee_id,
                document_id=None,
            )
        )
        phases = None
        if "phase" in template.eligibility:
            phases = [customer.phase_on(one.valid_from) for customer in state.customers.values()]
        out["promotion"].append(
            row(
                "promotion",
                promotion_id=promotion_id,
                code=one.code,
                name=template.name_cs,
                campaign_id=campaign_id,
                goal=template.goal_cs,
                target_product_line_id=row_id("product_line", code=template.line),
                target_sku_id=_sku(template.target_sku),
                target_uplift_pct=template.target_uplift_pct,
                target_period_weeks=template.weeks,
                estimated_budget=estimated_budget(promotion, template, demand, phases),
                valid_from=one.valid_from,
                valid_to=one.valid_to,
                mechanic=template.mechanic,
                discount_pct=template.discount_pct,
                condition_sku_id=_sku(template.condition_sku),
                condition_qty=template.condition_qty,
                reward_sku_id=_sku(template.reward_sku),
                reward_qty=template.reward_qty,
                is_mandatory=template.mandatory,
                is_pre_stocking=template.pre_stocking,
                owner_employee_id=duty.employee_id,
                status=status(promotion, end),
                document_id=None,
            )
        )
        for kind, values in template.eligibility.items():
            for value in values:
                out["promotion_eligibility"].append(
                    row(
                        "promotion_eligibility",
                        promotion_eligibility_id=row_id(
                            "promotion_eligibility", promotion=one.code, scope_kind=kind, scope_value=value
                        ),
                        promotion_id=promotion_id,
                        scope_kind=kind,
                        scope_value=value,
                    )
                )
        for other in one.excludes:
            if other in written:
                out["promotion_exclusion"].append(
                    row(
                        "promotion_exclusion",
                        promotion_id=promotion_id,
                        excluded_promotion_id=row_id("promotion", code=other),
                    )
                )


def _sku(code: str | None) -> int | None:
    return row_id("sku", sku_code=code) if code else None
