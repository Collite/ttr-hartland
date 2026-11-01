"""Setup step 3 (`05` §2): the organisation — `org_unit` (the Center's five units and their teams, `org_unit.csv`),
`employee` (one per post of `world.yaml salesforce`, with a Czech name), `assignment` (the duties: a rep's area with
the two moves of `04` S3, an area manager's region, a product manager's line, a dispatcher's DC; a KAM's customers
are `customers`'), and the hidden skills each actor carries in the `World`, never in a table (NH-16). The `employee`
and `assignment` rows are rendered from the `World`'s actors and duties (`employee_row`, `assignment`,
`opening.state_rows`).

**Posts.** Every person sits on a post with a stable key (`rep:PHA-03`, `kam:horeca:2`, `pm:GOLD`, …), so a draw
keyed by the post does not move when another post is added, unless an earlier post takes its name or its surname: a
taken name is drawn again, and the code counts the surname in post order, so a post added before it can move a
post's name, code and, through `id_key: [code]`, its `employee_id`. The order: management, the area managers by
region, the reps by `area.csv rep_slot`, the head of key accounts, the KAMs by channel, the product managers by line,
the dispatchers by DC, supply, finance, marketing, the shop managers by shop. Each reports to its post's manager: a rep
to the region's area manager, a KAM to the head of key accounts, the field and the key accounts to the sales director,
the product managers and marketing to the marketing director, finance to the finance director, supply and dispatch to
the managing director.

**Names** (stream `org`): a woman on 45 % of the posts; a first name of that gender and a surname in its form, drawn
again until the full name is new. `employee.code` is the surname folded to ASCII, lower case, and a two-digit count
of that surname in post order (`novakova01`) — the Keycloak username (contracts §6). Hired on a day drawn between
2008 and the day before the warm-up; nobody has left yet (turnover is the month-end's).

**The area moves** (`stories/s3.yaml area_move_days`): on each day every rep moves to another area of its region —
the region's areas, in `rep_slot` order, shift by a drawn step (stream `org`), never back onto the first one after the
second move — so a skill follows the person, not the territory.

**Skills** (stream `skills`), each only while its story is enabled (else 1.0, the null world): a rep's four S3 skills
are the pin's when its first area is a pin's `slot` (`anomalies.yaml pins.reps`), else each drawn in S3's
`skill_range`; A-7's `skips_visits` and A-8's `records_nothing` follow the pins while those anomalies are enabled. A
rep's, KAM's or product manager's S6 `discount_discipline` is pinned (`pinned_reps` by first area, `pinned_product_
managers` by line) or drawn in `discipline_range`. An area manager's S5 `pos_skill` and `budget_factor` are
`s5.yaml`'s by region.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from hartland_sim.config import StoriesConfig, WorldConfig
from hartland_sim.ids import row_id
from hartland_sim.io.reference import REFERENCE_DIR, Row, read
from hartland_sim.rng import rng
from hartland_sim.setup.reference import Geography, regions
from hartland_sim.setup.rows import Tables, row
from hartland_sim.world import OPEN, Actor, Duty, Skills

HIRED_FROM = date(2008, 1, 1)
FEMALE_SHARE = 0.45
CEO = "management:OU-HCZ"
SALES = "management:OU-OBC"
MARKETING = "management:OU-MKT"
FINANCE = "management:OU-FIN"
HEAD_KA = "head_of_key_accounts:1"
# The directors, in world.yaml's `management` count: the managing director first.
DIRECTORS = ("OU-HCZ", "OU-FIN", "OU-OBC", "OU-MKT")
KAM_TEAM = {"big_retail": "OU-KA-RET", "gas_ka": "OU-KA-CS", "vending": "OU-AUT", "horeca": "OU-GAS"}
# The column of an assignment row that holds the target of its kind (each target table's id_key is its code).
TARGET_COLUMN = {
    "rep_area": "area_id",
    "area_manager_region": "region_id",
    "kam_customer": "customer_id",
    "pm_line": "product_line_id",
    "dispatcher_dc": "site_id",
}


@dataclass(frozen=True)
class Post:
    slot: str
    role: str
    org_unit: str
    manager: str | None  # the manager's post
    target: str | None = None  # a rep's first area, an area manager's region, a PM's line, a dispatcher's DC


@dataclass
class Org:
    actors: dict[str, Actor]  # by post, in post order
    duties: list[Duty]
    posts: list[Post]
    codes: dict[int, str] = field(default_factory=dict)  # employee_id → code

    def by_role(self, role: str) -> list[tuple[Post, Actor]]:
        return [(post, self.actors[post.slot]) for post in self.posts if post.role == role]

    def kams(self, channel: str) -> list[Actor]:
        return [self.actors[post.slot] for post in self.posts if post.slot.startswith(f"kam:{channel}:")]


def posts(world: WorldConfig, geography: Geography, root: Path = REFERENCE_DIR) -> list[Post]:
    staff = world.salesforce
    if staff.management < len(DIRECTORS):
        raise ValueError(f"salesforce.management is {staff.management}; the org chart has {len(DIRECTORS)} directors")
    out = [Post(f"management:{unit}", "management", unit, None if unit == "OU-HCZ" else CEO) for unit in DIRECTORS]
    extra = staff.management - len(DIRECTORS)
    out += [Post(f"management:OU-HCZ:{n}", "management", "OU-HCZ", CEO) for n in range(2, extra + 2)]
    region_codes = [region["code"] for region in read("region", root)]
    if staff.area_managers != len(region_codes):
        raise ValueError(f"salesforce.area_managers is {staff.area_managers}, the regions {len(region_codes)}")
    out += [Post(f"area_manager:{r}", "area_manager", f"OU-TP-{r}", SALES, r) for r in region_codes]
    areas = sorted(read("area", root), key=lambda area: area["rep_slot"])
    if staff.reps != len(areas):
        raise ValueError(f"salesforce.reps is {staff.reps}, the areas {len(areas)}")
    out += [
        Post(f"rep:{a['code']}", "rep", f"OU-TP-{a['region']}", f"area_manager:{a['region']}", a["code"]) for a in areas
    ]
    out += [Post(HEAD_KA, "head_of_key_accounts", "OU-KA-RET", SALES)]
    for channel, count in staff.kams.items():
        out += [Post(f"kam:{channel}:{n}", "kam", KAM_TEAM[channel], HEAD_KA) for n in range(1, count + 1)]
    lines = read("product_line", root)
    if staff.product_managers != len(lines):
        raise ValueError(f"salesforce.product_managers is {staff.product_managers}, the lines {len(lines)}")
    out += [
        Post(f"pm:{line['code']}", "product_manager", f"OU-PM-{line['brand']}", MARKETING, line["code"])
        for line in lines
    ]
    dcs = list(world.sites.dcs)
    per_dc, extra = divmod(staff.dispatchers, len(dcs))
    if extra:
        raise ValueError(f"salesforce.dispatchers {staff.dispatchers} do not split over {len(dcs)} DCs")
    for dc in dcs:
        unit = f"OU-DIS-{dc.removeprefix('DC-')}"
        out += [Post(f"dispatcher:{dc}:{n}", "dispatcher", unit, CEO, dc) for n in range(1, per_dc + 1)]
    out += [Post(f"buyer:{n}", "buyer", "OU-ZAS", CEO) for n in range(1, staff.supply + 1)]
    out += [
        Post(f"finance:{n}", "finance", unit, FINANCE)
        for n, unit in _spread(staff.finance, "OU-FIN", ("OU-HR", "OU-IT"))
    ]
    out += [
        Post(f"marketing:{n}", "marketing", unit, MARKETING)
        for n, unit in _spread(staff.marketing, "OU-MKT", ("OU-WEB",))
    ]
    shops = [site["code"] for site in geography.sites.values() if site["kind"] == "shop"]
    if staff.shop_managers != len(shops):
        raise ValueError(f"salesforce.shop_managers is {staff.shop_managers}, the shops {len(shops)}")
    out += [Post(f"shop_manager:{shop}", "shop_manager", "OU-VP", SALES, shop) for shop in shops]
    return out


def _spread(count: int, main: str, teams: Sequence[str]) -> list[tuple[int, str]]:
    """`count` people: two in each of `teams` (when there are enough), the rest in `main`."""
    each = 2 if count >= 2 * len(teams) + 1 else 0
    units = [main] * (count - each * len(teams)) + [team for team in teams for _ in range(each)]
    return list(enumerate(units, start=1))


def build(
    out: Tables,
    world: WorldConfig,
    stories: StoriesConfig,
    geography: Geography,
    root: Path = REFERENCE_DIR,
) -> Org:
    """`org_unit` and `region` into `out`; the actors and their duties (but a KAM's, which are `customers`')."""
    units = read("org_unit", root)
    for unit in units:
        out["org_unit"].append(
            row(
                "org_unit",
                org_unit_id=row_id("org_unit", code=unit["code"]),
                code=unit["code"],
                name=unit["name"],
                parent_org_unit_id=row_id("org_unit", code=unit["parent"]) if unit["parent"] else None,
                cost_centre_id=row_id("cost_centre", code=unit["cost_centre"]),
            )
        )
    every = posts(world, geography, root)
    names = _Names(read("names_first", root), read("names_last", root))
    last_day = world.history.warm_up_from - timedelta(days=1)
    skills = _Skills(stories)
    actors: dict[str, Actor] = {}
    seen: Counter[str] = Counter()
    for post in every:
        first, last = names.draw(post.slot)
        login = _ascii(last)
        seen[login] += 1
        code = f"{login}{seen[login]:02d}"
        hired = _day(rng("org", "hired", post.slot), HIRED_FROM, last_day)
        manager = actors[post.manager].employee_id if post.manager else None
        employee = row_id("employee", code=code)
        actor = Actor(employee, code, first, last, post.role, post.org_unit, hired, manager, None, skills.of(post))
        actors[post.slot] = actor
    org = Org(actors, [], every, {actor.employee_id: actor.code for actor in actors.values()})
    moves = tuple(stories.story("S3").parameters["area_move_days"])
    if len(moves) != world.salesforce.rep_area_moves:
        raise ValueError(
            f"s3.yaml has {len(moves)} area_move_days, world.yaml rep_area_moves {world.salesforce.rep_area_moves}"
        )
    for post, actor in org.by_role("rep"):
        assert post.target is not None
        for area, valid_from, valid_to in rep_areas(post.target, actor.hired_on, moves, geography):
            org.duties.append(Duty(actor.employee_id, "rep_area", area, valid_from, valid_to))
    for role, kind in (
        ("area_manager", "area_manager_region"),
        ("product_manager", "pm_line"),
        ("dispatcher", "dispatcher_dc"),
    ):
        for post, actor in org.by_role(role):
            assert post.target is not None
            org.duties.append(Duty(actor.employee_id, kind, post.target, actor.hired_on))
    regions(out, {post.target: actor.employee_id for post, actor in org.by_role("area_manager") if post.target}, root)
    return org


def employee_row(actor: Actor) -> Row:
    """The `employee` row of `actor`."""
    return row(
        "employee",
        employee_id=actor.employee_id,
        code=actor.code,
        first_name=actor.first_name,
        last_name=actor.last_name,
        org_unit_id=row_id("org_unit", code=actor.org_unit),
        role=actor.role,
        manager_employee_id=actor.manager_id,
        hired_on=actor.hired_on,
        left_on=actor.left_on,
    )


def assignment(duty: Duty, codes: Mapping[int, str]) -> Row:
    """The `assignment` row of `duty`; its target in the column of its kind, the others empty."""
    target_table = TARGET_COLUMN[duty.kind].removesuffix("_id")
    values: dict[str, Any] = dict.fromkeys(TARGET_COLUMN.values())
    values[TARGET_COLUMN[duty.kind]] = row_id(target_table, code=duty.target)
    return row(
        "assignment",
        assignment_id=row_id(
            "assignment",
            employee=codes[duty.employee_id],
            kind=duty.kind,
            target=duty.target,
            valid_from=duty.valid_from,
        ),
        employee_id=duty.employee_id,
        kind=duty.kind,
        valid_from=duty.valid_from,
        valid_to=duty.valid_to,
        **values,
    )


def rep_areas(first: str, hired: date, moves: Sequence[date], geography: Geography) -> list[tuple[str, date, date]]:
    """The areas a rep holds, from `first` (held since `hired`), moving on each of `moves` within its region."""
    region = geography.region_of[first]
    ring = [area for area in geography.region_of if geography.region_of[area] == region]
    steps = area_steps(region, len(ring), len(moves))
    position = ring.index(first)
    starts = [hired, *moves]
    out = []
    for k, start in enumerate(starts):
        end = moves[k] - timedelta(days=1) if k < len(moves) else OPEN
        out.append((ring[position], start, end))
        if k < len(steps):
            position = (position + steps[k]) % len(ring)
    return out


def area_steps(region: str, size: int, moves: int) -> list[int]:
    """Each move's shift along the region's ring of areas (stream `org`): never 0, and the moves together never bring
    a rep back to its first area while the ring has room."""
    steps: list[int] = []
    for k in range(moves):
        allowed = [step for step in range(1, size) if size <= 2 or (sum(steps) + step) % size != 0]
        steps.append(int(rng("org", "area_move", region, k).choice(allowed)))
    return steps


def _day(generator: Any, first: date, last: date) -> date:
    return first + timedelta(days=int(generator.integers(0, (last - first).days + 1)))


def _ascii(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text)
    return "".join(char for char in folded if not unicodedata.combining(char)).lower()


class _Names:
    """Czech first names by gender and surnames in both forms (`names_first.csv`, `names_last.csv`)."""

    def __init__(self, first: Sequence[Row], last: Sequence[Row]) -> None:
        self.first = {gender: [n["nominative"] for n in first if n["gender"] == gender] for gender in ("m", "f")}
        self.last = {"m": [n["male"] for n in last], "f": [n["female"] for n in last]}
        self.taken: list[tuple[str, str]] = []

    def draw(self, slot: str) -> tuple[str, str]:
        generator = rng("org", "name", slot)
        while True:
            gender = "f" if generator.random() < FEMALE_SHARE else "m"
            name = (str(generator.choice(self.first[gender])), str(generator.choice(self.last[gender])))
            if name not in self.taken:
                self.taken.append(name)
                return name


class _Skills:
    """The hidden skills of a post's actor, from the stories (module docstring)."""

    def __init__(self, stories: StoriesConfig) -> None:
        self.s3 = stories.story("S3")
        self.s5 = stories.story("S5")
        self.s6 = stories.story("S6")
        self.pins = {pin.slot: pin for pin in stories.pins.reps}
        self.a7 = stories.anomaly("A-7").enabled
        self.a8 = stories.anomaly("A-8").enabled

    def of(self, post: Post) -> Skills:
        skills = Skills()
        if post.role == "rep":
            assert post.target is not None
            self._rep(skills, post.target)
        if post.role in ("rep", "kam", "product_manager") and self.s6.enabled:
            pinned = self.s6.parameters["pinned_reps" if post.role == "rep" else "pinned_product_managers"]
            if post.role != "kam" and post.target in pinned:
                skills.discount_discipline = float(pinned[post.target])
            else:
                low, high = self.s6.parameters["discipline_range"]
                skills.discount_discipline = float(rng("skills", "discipline", post.slot).uniform(low, high))
        if post.role == "area_manager" and self.s5.enabled:
            skills.pos_skill = float(self.s5.parameters["pos_skill"][post.target])
            skills.pos_budget_factor = float(self.s5.parameters["budget_factor"].get(post.target, 1.0))
        return skills

    def _rep(self, skills: Skills, area: str) -> None:
        pin = self.pins.get(area)
        if pin is not None:
            skills.skips_visits = pin.skips_visits and self.a7
            skills.records_nothing = pin.records_nothing and self.a8
        if not self.s3.enabled:
            return
        if pin is not None:
            skills.conversion, skills.acquisition = pin.conversion, pin.acquisition
            skills.retention, skills.cross_sell = pin.retention, pin.cross_sell
            skills.profile = pin.profile
            return
        low, high = self.s3.parameters["skill_range"]
        drawn = rng("skills", "rep", area).uniform(low, high, 4)
        skills.conversion, skills.acquisition, skills.retention, skills.cross_sell = (float(v) for v in drawn)
