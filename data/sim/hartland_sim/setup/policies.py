"""Setup step 6 (`05` §2), the policies of the opening: stock policies, the rep-prompt and lifecycle rules, the
discount budgets of the opening quarter, the PoS materials and the regions' asset pools, the carriers and their
vehicles. The price lists, promotions and campaigns are S1.3's.

- **`stock_policy`** per SKU × DC (`world.yaml policies.stock`): days of cover, Professional packs 15/45, the rest
  10/35, from the SKU's launch to the day before its delisting.
- **`prompt_rule`** (`policies.prompt_rules`, R-5…R-7, R-12) and **`lifecycle_rule`** (`policies.lifecycle`, R-4):
  per phase, the visit cycle with the prompts a visit carries and how a customer enters the phase, the share of a
  line's maximum agreed discount the rep may grant, the lines offered first. `parameters` holds JSON text (keys
  sorted, review-196 ⚑C). Valid from the warm-up's first day, owned by the sales director.
- **`discount_budget`**: a rep's or KAM's quarter is 2 % (`pricing.budgets`) of its planned gross. The opening
  quarter has no `sales_plan` yet (S1.4 rolls it at the warm-up's first month-end), so its planned gross is
  `opening_quarter_gross` per customer the rep's area or the KAM's accounts hold active on the quarter's first day.
- **`pos_material`** (`pos_material.csv`) and **`pos_asset`**: each region holds `assets_per_region` of each asset
  type, acquired on a day drawn between 2019 and the warm-up (stream `pos`), new in its first year, then good or worn
  as it ages; held by the region, lent by G2 S2.3.
- **`carrier`** (`carrier.csv`) and **`vehicle`**: each carrier's fleet by class, the capacities of `delivery.capacity`,
  a Czech plate drawn per vehicle (stream `route`) in the region of its DC.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from pathlib import Path

from hartland_sim.config import WorldConfig
from hartland_sim.ids import row_id
from hartland_sim.io.reference import REFERENCE_DIR, read
from hartland_sim.rng import rng
from hartland_sim.setup.portfolio import Portfolio
from hartland_sim.setup.reference import Geography
from hartland_sim.setup.rows import Tables, row
from hartland_sim.world import OPEN, Budget

ASSETS_FROM = date(2019, 1, 1)
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


def budgets(
    out: Tables,
    world: WorldConfig,
    planned: Iterable[tuple[int, str, int]],
) -> dict[tuple[int, date], Budget]:
    """`discount_budget` of the opening quarter into `out`: for each (employee_id, channel, customers) of `planned`,
    the reps' or KAMs' share of `opening_quarter_gross` × customers."""
    quarter = _quarter(world.history.warm_up_from)
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
    state = {}
    for employee, total in totals.items():
        amount = round(total * pct[employee] / 100, 2)
        out["discount_budget"].append(
            row("discount_budget", employee_id=employee, quarter=quarter, budget_amount=amount, used_amount=0.0)
        )
        state[employee, quarter] = Budget(amount)
    return state


def _quarter(day: date) -> date:
    return date(day.year, 3 * ((day.month - 1) // 3) + 1, 1)


def pos(out: Tables, world: WorldConfig, geography: Geography, root: Path = REFERENCE_DIR) -> None:
    """`pos_material` and the regions' `pos_asset` pools into `out`."""
    last = world.history.warm_up_from - timedelta(days=1)
    regions = list(dict.fromkeys(geography.region_of.values()))
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
    for region in regions:
        for material in materials:
            for n in range(1, material["assets_per_region"] + 1):
                asset_no = f"{material['code']}-{region}-{n:04d}"
                draw = rng("pos", "asset", asset_no)
                acquired = ASSETS_FROM + timedelta(days=int(draw.integers(0, (last - ASSETS_FROM).days + 1)))
                age = (last - acquired).days / 365.25
                worn = min(0.6, 0.12 * age)
                condition = "new" if age < 1 else ("worn" if draw.random() < worn else "good")
                out["pos_asset"].append(
                    row(
                        "pos_asset",
                        pos_asset_id=row_id("pos_asset", asset_no=asset_no),
                        pos_material_id=row_id("pos_material", code=material["code"]),
                        asset_no=asset_no,
                        acquired_on=acquired,
                        cost=material["unit_cost"],
                        region_id=row_id("region", code=region),
                        condition=condition,
                        written_off_on=None,
                    )
                )


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
