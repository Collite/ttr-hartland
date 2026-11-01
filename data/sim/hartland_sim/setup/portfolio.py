"""Setup step 4 (`05` §2): the portfolio — `business`, `brand`, `product_line`, `product`, `sku` from the catalogue
(`brand.csv`, `product_line.csv`, `product.csv`, `catalogue.csv`; launches and delistings dated there), what each
channel may buy (`channel_listing`), the lines' seasonal exclusions and the factories' terms (`supplier_terms`).

**Listings** (`world.yaml channels.listings`, R-7): a channel lists the retail packs of its `retail` lines and the
Professional packs of its `professional` lines (`all` = every line), each SKU from its launch to the day before its
delisting, numbered by importance (10/5/1/0) then code — the `listing_sequence` a new customer is offered in. The
big-retail chains list centrally (`01` §4.3), so big retail's rows are per chain; the other channels' rows name no
chain. A-9 ends two chains' listing of one SKU in its month. `special_retail` (the station shops, a sub-channel of
small retail) has a listing of its own in `world.yaml` but no row shape: `channel_listing` has no sub-channel
column, so the station shops read small retail's rows (Found while doing).

**Supplier terms** (`world.yaml suppliers`): each SKU's factories (`catalogue.csv suppliers`), with the factory's
lead time, reliability and minimum lot; a SKU two coffee factories make splits by `coffee_split`, one made by a single
factory takes all of it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from hartland_sim.config import Channel, Listing, StoriesConfig, WorldConfig
from hartland_sim.ids import row_id
from hartland_sim.io.reference import REFERENCE_DIR, Row, read
from hartland_sim.setup.rows import Tables, row
from hartland_sim.world import OPEN

# The channels of the data (the inventory's `channel` list): world.yaml's `special_retail` is a sub-channel.
CHANNELS: tuple[Channel, ...] = ("big_retail", "small_retail", "gas_ka", "own_shops", "web", "vending", "horeca")


@dataclass(frozen=True)
class Sku:
    code: str
    sku_id: int
    product: str
    line: str
    is_professional: bool
    importance: int
    units_per_case: int
    base_price_case: float
    cogs_case: float
    suppliers: tuple[str, ...]
    launched_on: date
    delisted_on: date | None

    @property
    def last_day(self) -> date:
        """The last day the SKU is sold: the day before its delisting, or open."""
        return self.delisted_on - timedelta(days=1) if self.delisted_on else OPEN


@dataclass(frozen=True)
class Portfolio:
    skus: tuple[Sku, ...]  # catalogue.csv's order
    lines: Mapping[str, Row]  # line code → product_line.csv's row

    def sku(self, code: str) -> Sku:
        (found,) = [sku for sku in self.skus if sku.code == code]
        return found


def build(out: Tables, world: WorldConfig, stories: StoriesConfig, root: Path = REFERENCE_DIR) -> Portfolio:
    brands = read("brand", root)
    for business, name in dict.fromkeys((brand["business"], brand["business_name"]) for brand in brands):
        out["business"].append(row("business", business_id=row_id("business", code=business), code=business, name=name))
    for brand in brands:
        out["brand"].append(
            row(
                "brand",
                brand_id=row_id("brand", code=brand["code"]),
                business_id=row_id("business", code=brand["business"]),
                code=brand["code"],
                name=brand["name"],
            )
        )
    lines = read("product_line", root)
    for line in lines:
        out["product_line"].append(
            row(
                "product_line",
                product_line_id=row_id("product_line", code=line["code"]),
                brand_id=row_id("brand", code=line["brand"]),
                code=line["code"],
                name=line["name_cs"],
                is_premium=line["is_premium"],
                is_professional=line["is_professional"],
                max_agreed_discount_pct=world.pricing.max_agreed_discount_pct[line["code"]],
            )
        )
    products = read("product", root)
    line_of = {product["code"]: product["product_line"] for product in products}
    for product in products:
        out["product"].append(
            row(
                "product",
                product_id=row_id("product", code=product["code"]),
                product_line_id=row_id("product_line", code=product["product_line"]),
                code=product["code"],
                name=product["name_cs"],
            )
        )
    skus = []
    for item in read("catalogue", root):
        sku = Sku(
            code=item["sku_code"],
            sku_id=row_id("sku", sku_code=item["sku_code"]),
            product=item["product"],
            line=line_of[item["product"]],
            is_professional=item["is_professional"],
            importance=item["importance"],
            units_per_case=item["units_per_case"],
            base_price_case=item["base_price_case"],
            cogs_case=item["cogs_case"],
            suppliers=tuple(item["suppliers"].split(";")),
            launched_on=item["launched_on"],
            delisted_on=item["delisted_on"],
        )
        skus.append(sku)
        out["sku"].append(
            row(
                "sku",
                sku_id=sku.sku_id,
                sku_code=sku.code,
                product_id=row_id("product", code=sku.product),
                name=item["name_cs"],
                pack_format=item["pack_format"],
                net_weight_g=item["net_weight_g"],
                units_per_case=item["units_per_case"],
                case_weight_kg=item["case_weight_kg"],
                shelf_life_days=item["shelf_life_days"],
                vat_rate=item["vat_rate"],
                importance=item["importance"],
                is_professional=item["is_professional"],
                launched_on=sku.launched_on,
                delisted_on=sku.delisted_on,
                launch_document_id=None,
                delist_document_id=None,
            )
        )
    portfolio = Portfolio(tuple(skus), {line["code"]: line for line in lines})
    out["channel_listing"] += listings(world, stories, portfolio)
    for code, months in world.policies.seasonal_exclusion.items():
        out["seasonal_exclusion"] += [
            row("seasonal_exclusion", product_line_id=row_id("product_line", code=code), month_of_year=month)
            for month in sorted(months)
        ]
    out["supplier_terms"] += supplier_terms(world, portfolio)
    return portfolio


def listed(listing: Listing, sku: Sku) -> bool:
    lines = listing.professional if sku.is_professional else listing.retail
    return lines == "all" or sku.line in lines


def listings(world: WorldConfig, stories: StoriesConfig, portfolio: Portfolio) -> list[Row]:
    """`channel_listing`: per channel (per chain in big retail) the SKUs it may buy, in listing sequence."""
    a9 = stories.anomaly("A-9")
    a9_end = _month_end(a9.parameters["month"]) if a9.enabled else None
    rows = []
    for channel in CHANNELS:
        chains: Sequence[str | None] = world.channels.big_retail.accounts if channel == "big_retail" else (None,)
        skus = sorted(
            (sku for sku in portfolio.skus if listed(world.channels.listings[channel], sku)),
            key=lambda sku: (-sku.importance, sku.code),
        )
        for chain in chains:
            for sequence, sku in enumerate(skus, start=1):
                valid_to = sku.last_day
                if a9_end and chain in a9.parameters["chains"] and sku.code == a9.parameters["sku"]:
                    valid_to = min(valid_to, a9_end)
                rows.append(
                    row(
                        "channel_listing",
                        channel_listing_id=row_id(
                            "channel_listing",
                            channel=channel,
                            chain=chain or "",
                            sku=sku.code,
                            valid_from=sku.launched_on,
                        ),
                        channel=channel,
                        chain_id=row_id("chain", code=chain) if chain else None,
                        sku_id=sku.sku_id,
                        valid_from=sku.launched_on,
                        valid_to=valid_to,
                        listing_sequence=sequence,
                    )
                )
    return rows


def supplier_terms(world: WorldConfig, portfolio: Portfolio) -> list[Row]:
    suppliers = world.suppliers
    rows = []
    for sku in portfolio.skus:
        for supplier in sku.suppliers:
            share = suppliers.coffee_split[supplier] if len(sku.suppliers) > 1 else 1.0
            rows.append(
                row(
                    "supplier_terms",
                    supplier_entity_id=row_id("legal_entity", code=supplier),
                    sku_id=sku.sku_id,
                    valid_from=sku.launched_on,
                    valid_to=sku.last_day,
                    lead_time_days=suppliers.lead_time_days[supplier],
                    min_lot_cases=suppliers.min_lot_cases[supplier],
                    reliability_pct=suppliers.reliability_pct[supplier],
                    sourcing_share=share,
                )
            )
    return rows


def _month_end(day: date) -> date:
    following = date(day.year + day.month // 12, day.month % 12 + 1, 1)
    return following - timedelta(days=1)
