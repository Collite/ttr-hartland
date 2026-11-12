"""Setup step 6 (`05` §2), the policies of the opening: stock policies, the rep-prompt and lifecycle rules, the
discount budgets of the opening quarter, the PoS materials and the regions' asset pools, the carriers and their
vehicles, the price lists.

- **`stock_policy`** per SKU × DC (`world.yaml policies.stock`): days of cover, Professional packs 15/45, the rest
  10/35, from the SKU's launch to the day before its delisting.
- **`prompt_rule`** (`policies.prompt_rules`, R-5…R-7, R-12) and **`lifecycle_rule`** (`policies.lifecycle`, R-4):
  per phase, the visit cycle with the prompts a visit carries and how a customer enters the phase, the share of a
  line's maximum agreed discount the rep may grant, the lines offered first. `parameters` holds JSON text (keys
  sorted, review-196 ⚑C). Valid from the warm-up's first day, owned by the sales director.
- **`discount_budget`**: a rep's or KAM's quarter is 2 % (`pricing.budgets`) of its planned gross. The opening
  quarter has no `sales_plan` yet (S1.4 rolls it at the warm-up's first month-end), so its planned gross is
  `opening_quarter_gross` per customer the rep's area or the KAM's accounts hold active on the quarter's first day.
  The `World` holds the budgets (the loop accrues `used_amount`); the rows are rendered from it (`budget_row`).
- **`pos_material`** (`pos_material.csv`) and the regions' pools (`World.assets`, G2 S2.3's `loop.pos`): each region
  holds `assets_per_region` of each asset type, bought in one of the three yearly purchases (`world.yaml
  field.pos_acquisitions`) up to the warm-up's first day, drawn by the asset (stream `pos`) — new in the last one,
  then good or worn as it ages — and each purchase's `asset_register` row (`loop.pos.register_row`), its region's
  name and cost centre from the run's PoS terms (`loop.pos.Terms`, made here once). `pos_asset` is rendered from the
  `World` (`setup.opening.state_rows`): the loop writes assets off and buys more.
- **`carrier`** (`carrier.csv`) and **`vehicle`**: each carrier's fleet by class, the capacities of `delivery.capacity`,
  a Czech plate drawn per vehicle (stream `route`) in the region of its DC.
- **The price lists** (`loop.pricing`): the `World` gets the versions in force on the warm-up's first day — the
  standard list of Q1-2023 at the catalogue's prices (every line's price index 1.0) and each channel's of Jan-2023.
  Setup writes no price-list row: the period ends roll the lists from there and write every version from the first
  written period on, so 2023-12-31's quarter-end and month-end write SC-2024-Q1 and Jan-2024's (S1.4, ⚑R213-A).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.dates import quarter_start
from hartland_sim.ids import row_id
from hartland_sim.io.reference import REFERENCE_DIR, Row, read
from hartland_sim.loop.demand import CHANNELS, lines
from hartland_sim.loop.pos import Terms, register_row
from hartland_sim.loop.pricing import channel_list, standard_list
from hartland_sim.rng import rng
from hartland_sim.setup.portfolio import Portfolio
from hartland_sim.setup.reference import Geography
from hartland_sim.setup.rows import Tables, row
from hartland_sim.world import OPEN, Asset, Budget, PriceList

PURCHASES = 3  # the yearly purchases the opening pools were bought in
PLATE_REGION = {"DC-JIRNY": "S", "DC-MODRICE": "B", None: "A"}  # Středočeský, Jihomoravský, Praha
PLATE_LETTERS = "ABCDEFHJKLMNPRSTUVXYZ"  # Czech plates use no G, O, Q, W
VEHICLE_CLASSES = {"vans": "van", "trucks_7_5t": "truck_7_5t", "trucks_12t": "truck_12t"}


def as_json(parameters: Mapping[str, object]) -> str:
    """A rule's parameters as the JSON text its `parameters` column holds: keys sorted, Czech kept as is."""
    return json.dumps(_plain(parameters), ensure_ascii=False, sort_keys=True)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, date):
        return value.isoformat()
    return value


def stock_policies(out: Tables, world: WorldConfig, portfolio: Portfolio) -> dict[tuple[int, int], tuple[float, float]]:
    """`stock_policy` into `out`; each (sku_id, dc_site_id)'s opening (min, max) days of cover."""
    covers = {}
    for sku in portfolio.skus:
        cover = world.policies.stock["professional" if sku.is_professional else "default"]
        for dc in world.sites.dcs:
            dc_id = row_id("site", code=dc)
            out["stock_policy"].append(
                row(
                    "stock_policy",
                    stock_policy_id=row_id("stock_policy", sku=sku.code, dc=dc, valid_from=sku.launched_on),
                    sku_id=sku.sku_id,
                    dc_site_id=dc_id,
                    min_days_cover=cover.min_days,
                    max_days_cover=cover.max_days,
                    valid_from=sku.launched_on,
                    valid_to=sku.last_day,
                )
            )
            covers[sku.sku_id, dc_id] = (float(cover.min_days), float(cover.max_days))
    return covers


def rules(out: Tables, world: WorldConfig, owner: int) -> None:
    """`prompt_rule` and `lifecycle_rule` into `out`, valid from the warm-up's first day."""
    start = world.history.warm_up_from
    for rule in world.policies.prompt_rules:
        out["prompt_rule"].append(
            row(
                "prompt_rule",
                prompt_rule_id=row_id("prompt_rule", code=rule.code, valid_from=start),
                code=rule.code,
                kind=rule.kind,
                parameters=as_json(rule.parameters),
                priority=rule.priority,
                is_mandatory=rule.mandatory,
                is_adaptable=rule.adaptable,
                valid_from=start,
                valid_to=OPEN,
                owner_employee_id=owner,
                document_id=None,
            )
        )
    life = world.policies.lifecycle
    entry: dict[str, dict[str, object]] = {
        "potential": {"max_orders": 0, "lookback_months": life.lookback_months},
        "occasional": {"min_orders": 1, "lookback_months": life.lookback_months},
        "regular": {"min_order_months": life.regular_min_order_months, "lookback_months": life.lookback_months},
        "vip": {"min_net_revenue": life.vip_min_net_revenue, "lookback_months": life.lookback_months},
    }
    for phase in life.phases:
        kinds: dict[str, dict[str, object]] = {
            "contact_frequency": {
                "cycle_weeks": life.contact_weeks[phase],
                "prompts_per_visit": world.policies.prompt_caps[phase],
                "entry": entry[phase],
            },
            "concessions": {"max_agreed_discount_share": life.concession_share[phase]},
            "listing_sequence": {"lines": list(life.listing_sequence[phase])},
        }
        for kind, parameters in kinds.items():
            out["lifecycle_rule"].append(
                row(
                    "lifecycle_rule",
                    lifecycle_rule_id=row_id("lifecycle_rule", phase=phase, rule_kind=kind, valid_from=start),
                    phase=phase,
                    rule_kind=kind,
                    parameters=as_json(parameters),
                    valid_from=start,
                    valid_to=OPEN,
                    document_id=None,
                )
            )


def price_lists(world: WorldConfig, portfolio: Portfolio) -> tuple[PriceList, dict[str, PriceList]]:
    """The standard list and the channels' lists in force on the warm-up's first day, for the `World`; no rows."""
    start = world.history.warm_up_from
    opening = standard_list(portfolio.skus, start)
    return opening, {channel: channel_list(opening, portfolio.skus, world, channel, start) for channel in CHANNELS}


def budgets(world: WorldConfig, planned: Iterable[tuple[int, str, int]]) -> dict[tuple[int, date], Budget]:
    """The opening quarter's budgets by (employee_id, quarter): for each (employee_id, channel, customers) of
    `planned`, the reps' or KAMs' share of `opening_quarter_gross` × customers."""
    quarter = quarter_start(world.history.warm_up_from)
    shares = world.pricing.budgets
    gross = shares.opening_quarter_gross
    totals: dict[int, float] = {}
    pct: dict[int, float] = {}
    for employee, channel, customers in planned:
        totals[employee] = totals.get(employee, 0.0) + gross[channel] * customers
        pct[employee] = (
            shares.rep_discount_pct_of_planned_gross
            if channel == "small_retail"
            else shares.kam_discount_pct_of_planned_gross
        )
    return {(employee, quarter): Budget(round(total * pct[employee] / 100, 2)) for employee, total in totals.items()}


def budget_row(employee: int, quarter: date, budget: Budget) -> Row:
    """The `discount_budget` row of `employee`'s `quarter`."""
    return row(
        "discount_budget",
        employee_id=employee,
        quarter=quarter,
        budget_amount=budget.budget_amount,
        used_amount=budget.used_amount,
    )


def pos(
    out: Tables, world: WorldConfig, stories: StoriesConfig, geography: Geography, root: Path = REFERENCE_DIR
) -> tuple[dict[int, Asset], dict[str, int], Terms]:
    """`pos_material` and the purchases' `asset_register` rows into `out`; the regions' pools (by pos_asset_id), each
    pool's last asset number (`World.counters`, by its prefix) and the PoS terms they were made with."""
    first = world.history.warm_up_from
    month = world.field.pos_acquisitions.month
    last = date(first.year if (first.month, first.day) >= (month, 1) else first.year - 1, month, 1)
    purchases = [date(last.year - k, month, 1) for k in range(PURCHASES - 1, -1, -1)]
    regions = list(dict.fromkeys(geography.region_of.values()))
    terms = Terms.of(world, stories, lines(root), root)
    materials = read("pos_material", root)
    for material in materials:
        line = material["product_line"]
        out["pos_material"].append(
            row(
                "pos_material",
                pos_material_id=row_id("pos_material", code=material["code"]),
                code=material["code"],
                name=material["name"],
                kind=material["kind"],
                unit_cost=material["unit_cost"],
                product_line_id=row_id("product_line", code=line) if line else None,
            )
        )
    assets: dict[int, Asset] = {}
    counters: dict[str, int] = {}
    for region in regions:
        spent = dict.fromkeys(purchases, 0.0)
        for material in materials:
            prefix = f"{material['code']}-{region}"
            for n in range(1, material["assets_per_region"] + 1):
                asset_no = f"{prefix}-{n:04d}"
                draw = rng("pos", "asset", asset_no)
                acquired = purchases[int(draw.integers(0, len(purchases)))]
                age = (first - acquired).days / 365.25
                condition = "new" if acquired == last else ("worn" if draw.random() < min(0.6, 0.12 * age) else "good")
                cost = float(material["unit_cost"])
                assets[row_id("pos_asset", asset_no=asset_no)] = Asset(
                    material["code"], asset_no, region, acquired, cost, condition
                )
                spent[acquired] += cost
                counters[prefix] = n
        for acquired, cost in spent.items():
            if cost:
                out["asset_register"].append(
                    register_row(
                        region,
                        terms.region_names[region],
                        terms.cost_centres[region],
                        acquired,
                        cost,
                        world.field.pos_asset_life_months,
                    )
                )
    return assets, counters, terms


def fleet(out: Tables, world: WorldConfig, root: Path = REFERENCE_DIR) -> None:
    """`carrier` and `vehicle` into `out`."""
    plates: list[str] = []
    for carrier in read("carrier", root):
        out["carrier"].append(
            row(
                "carrier",
                carrier_id=row_id("carrier", code=carrier["code"]),
                code=carrier["code"],
                name=carrier["name"],
                dc_site_id=row_id("site", code=carrier["dc"]) if carrier["dc"] else None,
                rate_base=carrier["rate_base"],
                rate_per_km=carrier["rate_per_km"],
                rate_per_stop=carrier["rate_per_stop"],
            )
        )
        for column, vehicle_class in VEHICLE_CLASSES.items():
            capacity = world.delivery.capacity[vehicle_class]
            for n in range(1, carrier[column] + 1):
                plate = _plate(carrier["code"], vehicle_class, n, PLATE_REGION[carrier["dc"]], plates)
                out["vehicle"].append(
                    row(
                        "vehicle",
                        vehicle_id=row_id("vehicle", plate=plate),
                        carrier_id=row_id("carrier", code=carrier["code"]),
                        plate=plate,
                        vehicle_class=vehicle_class,
                        capacity_cases=capacity.cases,
                        capacity_kg=capacity.kg,
                    )
                )


def _plate(carrier: str, vehicle_class: str, n: int, region: str, taken: list[str]) -> str:
    """A Czech plate (`5S2 4170`): a digit, the region's letter, a letter or digit, four digits; new in `taken`."""
    draw = rng("route", "plate", carrier, vehicle_class, n)
    while True:
        third = str(draw.choice(list(PLATE_LETTERS + "0123456789")))
        plate = f"{int(draw.integers(1, 10))}{region}{third} {int(draw.integers(0, 10000)):04d}"
        if plate not in taken:
            taken.append(plate)
            return plate
