"""Setup step 1 (`05` §2): the reference tables, from `data/reference/`, with their ids — `legal_entity`, `kraj`,
`okres` (with the affluence index of ⚑NH-17, Prague's 22 districts under Praha), `site`, `calendar`, `cost_centre`,
`area`, `area_okres`, and `region` once `org` has named the area managers (`regions`). The weather series is S1.2's,
the chart of accounts G4's.

`Geography` is what the later steps look up: each okres's area, each area's region, each region's DC (`world.yaml
sites.dcs`), the leaves a customer can sit in (every okres but Praha, which its districts replace).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hartland_sim.config import WorldConfig
from hartland_sim.ids import row_id
from hartland_sim.io.reference import REFERENCE_DIR, Row, read
from hartland_sim.setup.affluence import load as affluence
from hartland_sim.setup.rows import Tables, row


@dataclass(frozen=True)
class Geography:
    okres: Mapping[str, Row]  # code → okres.csv's row
    area_of: Mapping[str, str]  # okres code → area code
    region_of: Mapping[str, str]  # area code → region code
    dc_of: Mapping[str, str]  # region code → its DC's site code
    sites: Mapping[str, Row]  # code → site.csv's row
    leaves: tuple[str, ...]  # the okres codes a customer can sit in, in okres.csv's order

    def dc_for(self, okres: str) -> str:
        return self.dc_of[self.region_of[self.area_of[okres]]]


def build(out: Tables, world: WorldConfig, root: Path = REFERENCE_DIR) -> Geography:
    """Every reference table but `region` into `out`; the lookups."""
    for entity in read("legal_entity", root):
        out["legal_entity"].append(
            row(
                "legal_entity",
                legal_entity_id=row_id("legal_entity", code=entity["code"]),
                code=entity["code"],
                name=entity["name"],
                ico=entity["ico"],
                role=entity["role"],
            )
        )
    for kraj in read("kraj", root):
        out["kraj"].append(row("kraj", kraj_id=row_id("kraj", code=kraj["code"]), code=kraj["code"], name=kraj["name"]))
    index = affluence(root)
    okresy = read("okres", root)
    for okres in okresy:
        parent = okres["parent_okres"]
        out["okres"].append(
            row(
                "okres",
                okres_id=row_id("okres", code=okres["code"]),
                code=okres["code"],
                name=okres["name"],
                kraj_id=row_id("kraj", code=okres["kraj"]),
                parent_okres_id=row_id("okres", code=parent) if parent else None,
                affluence_index=index[okres["code"]],
                population=okres["population"],
            )
        )
    sites = read("site", root)
    for site in sites:
        out["site"].append(
            row(
                "site",
                site_id=row_id("site", code=site["code"]),
                code=site["code"],
                kind=site["kind"],
                legal_entity_id=row_id("legal_entity", code=site["legal_entity"]),
                name=site["name"],
                town=site["town"],
                okres_id=row_id("okres", code=site["okres"]),
                lat=site["lat"],
                lon=site["lon"],
            )
        )
    out["calendar"].extend(row("calendar", **day) for day in read("calendar", root))
    for centre in read("cost_centre", root):
        out["cost_centre"].append(
            row(
                "cost_centre",
                cost_centre_id=row_id("cost_centre", code=centre["code"]),
                code=centre["code"],
                name=centre["name"],
                unit=centre["unit"],
                site_id=row_id("site", code=centre["site"]) if centre["site"] else None,
            )
        )
    areas = read("area", root)
    for area in areas:
        out["area"].append(
            row(
                "area",
                area_id=row_id("area", code=area["code"]),
                code=area["code"],
                name=area["name"],
                region_id=row_id("region", code=area["region"]),
            )
        )
    links = read("area_okres", root)
    for link in links:
        out["area_okres"].append(
            row("area_okres", area_id=row_id("area", code=link["area"]), okres_id=row_id("okres", code=link["okres"]))
        )
    parents = {okres["parent_okres"] for okres in okresy}
    return Geography(
        okres={okres["code"]: okres for okres in okresy},
        area_of={link["okres"]: link["area"] for link in links},
        region_of={area["code"]: area["region"] for area in areas},
        dc_of={region: dc for dc, regions in world.sites.dcs.items() for region in regions},
        sites={site["code"]: site for site in sites},
        leaves=tuple(okres["code"] for okres in okresy if okres["code"] not in parents),
    )


def regions(out: Tables, managers: Mapping[str, int], root: Path = REFERENCE_DIR) -> None:
    """`region` into `out`, each with its area manager's `employee_id` (`managers`: region code → id)."""
    for region in read("region", root):
        out["region"].append(
            row(
                "region",
                region_id=row_id("region", code=region["code"]),
                code=region["code"],
                name=region["name"],
                area_manager_employee_id=managers[region["code"]],
            )
        )
