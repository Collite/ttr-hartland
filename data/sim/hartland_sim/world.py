"""The simulation's state and its daily checkpoint (contracts §2, architecture §2).

`World` holds everything the day loop carries from one day to the next — customers with their condition, phase
episodes and basket, stock per SKU × DC, open orders, shipments and invoices, tourplans, loans, running promotions,
budgets, the actors' hidden skills and their duties, what the reps' customers bought by SKU and the prompts' memory.
Setup (`setup.build`) makes the opening `World`; the loop moves it a day at a time. The tables that hold the state the
loop changes (`customer`, `lifecycle_phase`, `employee`, `assignment`, `discount_budget`) are rendered from the `World`
when a build ends (`setup.opening.state_rows`), so it holds every value of their rows: the customer's name, the
actor's name and manager.

`Checkpoint` is the pickled `World` after a day, hashed, stored as `_sim_checkpoint(day, sha256, blob)`; `prolong`
resumes from the last one, and the 2023-12-31 checkpoint (the end of the warm-up year) lets a null-world or
re-parameterised build start from the same opening. `Meta` is `_sim_meta(key, value)`: the master seed, the hashes of
the inputs, the last day done and the build id. Neither table is in the inventory (contracts §1): the generator
writes both beside the CSVs, `load-world` creates them.

**Deterministic.** The same inputs pickle to the same bytes on the same path: the state holds lists, tuples, dicts
and dataclasses only — never a `set`, whose order follows `PYTHONHASHSEED` — filled in a fixed order, and the pickle
protocol is pinned (review-188). A world restored from a checkpoint and moved on pickles *equal* to the world a
straight build reaches, not byte-identical: pickle's memo writes one string shared by two fields otherwise than two
equal strings, and a restore shares what the straight build did not (review-201 N7). So two checkpoints are compared
by `restore()`, never by their bytes or sha256; the tables, not `_sim_checkpoint`, are a build's identity. The memo
stays (no `Pickler.fast`): without it a restore would turn one shared object into two.

**The trust boundary.** The checkpoint's sha256 is integrity, not authenticity: whoever can write `_sim_checkpoint`
can run code on the `prolong` host when it unpickles the blob. That is acceptable while only the owner role writes
the `hartland` database (contracts §8); the readonly and DocWH roles cannot write it.
"""

from __future__ import annotations

import csv
import hashlib
import os
import pickle
from collections.abc import Iterable
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hartland_sim.loop.demand import Demand
    from hartland_sim.loop.pos import Measure
    from hartland_sim.loop.prompts import Purchases

PROTOCOL = 5
CHECKPOINT_FILE = "_sim_checkpoint.csv"
CHECKPOINT_HEADER = ("day", "sha256", "blob")
META_FILE = "_sim_meta.csv"
OPEN = date(9999, 12, 31)  # a validity without an end (`03` §0)


@dataclass
class Skills:
    """An actor's hidden skills (`04` S3, S5, S6) — the answer key's, never a table (NH-16). 1.0 is the average."""

    conversion: float = 1.0  # a rep: P(an order | a visit) × this (S3)
    acquisition: float = 1.0  # a rep: potential → customer (S3)
    retention: float = 1.0  # a rep: the churn hazard ÷ this (S3)
    cross_sell: float = 1.0  # a rep: a new line bought (S3)
    discount_discipline: float = 1.0  # a rep, a KAM, a product manager (S6)
    pos_skill: float = 1.0  # an area manager (S5)
    pos_budget_factor: float = 1.0  # an area manager: × the region's PoS budget (S5's generous one 1.5)
    skips_visits: bool = False  # A-7
    records_nothing: bool = False  # A-8
    profile: str | None = None  # a pinned rep's profile (star, weak, hunter, …); None when drawn


@dataclass
class Actor:
    """A person of the Center, with the skills the stories give them."""

    employee_id: int
    code: str
    first_name: str
    last_name: str
    role: str
    org_unit: str
    hired_on: date
    manager_id: int | None = None  # the manager's employee_id
    left_on: date | None = None
    skills: Skills = field(default_factory=Skills)


@dataclass
class Duty:
    """An `assignment` row as the loop reads it: who answers for `target` (an area, region, customer, line or site
    code, by `kind`) from `valid_from` to `valid_to`, both included."""

    employee_id: int
    kind: str
    target: str
    valid_from: date
    valid_to: date = OPEN

    def on(self, day: date) -> bool:
        return self.valid_from <= day <= self.valid_to


@dataclass
class Condition:
    """A retail customer's conditions for a month (`customer_condition`, S1.3/S1.4): the tiers its previous month
    reached (`loop.pricing.condition`)."""

    month: date
    volume_tier: str | None = None
    volume_pct: float = 0.0
    cross_sell_tier: str | None = None
    cross_sell_pct: float = 0.0
    net_revenue: float = 0.0  # the previous month's
    lines_bought: int = 0  # the previous month's product lines
    both_brands: bool = False  # the previous month bought coffee and culinary


@dataclass(frozen=True)
class Episode:
    """A lifecycle phase from `valid_from` until the next episode starts (`lifecycle_phase`, R-4)."""

    phase: str  # potential · occasional · regular · vip · lost
    valid_from: date
    reason: str  # the row's `reason`, in Czech


@dataclass
class Customer:
    """A trading partner's state: who it is (its name and codes, so its row can be written) and where it stands."""

    customer_id: int
    code: str
    name: str
    channel: str
    sub_channel: str | None
    status: str  # potential · active · lost
    since: date | None
    until: date | None
    okres: str | None
    area: str | None
    store: str | None  # the store's osm_id (a retail customer, a HoReCa place)
    chain: str | None
    episodes: list[Episode]  # in order, the last one open
    traits: tuple[str, ...] = ()  # the planted cases' (`anomalies.yaml pins.customers`)
    condition: Condition | None = None

    def active_on(self, day: date) -> bool:
        """A customer from its `since` until its `until`, both included."""
        if self.status == "potential" or self.since is None or self.since > day:
            return False
        return self.until is None or day <= self.until

    def phase_on(self, day: date) -> str:
        """The phase of the last episode that starts on or before `day`; `potential` before the first."""
        phase = "potential"
        for episode in self.episodes:
            if episode.valid_from > day:
                break
            phase = episode.phase
        return phase


@dataclass
class Stock:
    """One SKU at one DC."""

    on_hand: float
    reserved: float = 0.0
    in_transit: float = 0.0


@dataclass
class Budget:
    """A rep's or KAM's discount budget for a quarter (`discount_budget`). `reserved` is what the quarter's agreements
    expect to cost (`loop.pricing.grant`) and is not yet posted: invoicing moves it to `used_amount` (G4 S4.1)."""

    budget_amount: float
    used_amount: float = 0.0
    reserved: float = 0.0


@dataclass
class Bought:
    """What a customer bought in a month, as its orders book it (`loop.orders.book`, S1.4): its net revenue without
    VAT, its orders, and the product lines of its paid lines, a bit each in `Demand.lines`' order. The month-end reads
    the next month's tiers and the phases from it."""

    net_revenue: float = 0.0
    orders: int = 0
    lines: int = 0


@dataclass(frozen=True)
class Promotion:
    """A promotion as the loop reads it (`promotion` + its `promotion_eligibility` rows): setup seeds the committed
    calendar (`setup.promotions.seeded`), S1.3 prices it on the order lines, a prolong reads the database's (G5)."""

    promotion_id: int
    code: str
    line: str  # the product line's code
    mechanic: str  # price_off · conditional_price_off · free_goods_same · free_goods_other
    valid_from: date
    valid_to: date
    tv: bool  # its campaign has a TV line (S2)
    eligibility: tuple[tuple[str, tuple[str, ...]], ...]  # (scope kind, its values), every kind must hold
    discount_pct: float | None = None
    target_sku: str | None = None  # catalogue sku_code; None = the whole line
    condition_sku: str | None = None
    reward_sku: str | None = None
    condition_qty: float | None = None  # cases of the condition SKU (or, 1+1, of the target) a reward needs
    reward_qty: float | None = None  # cases free per `condition_qty`
    mandatory: bool = False  # the reps must offer it (S2.2's prompt)
    name: str = ""  # its template's Czech name (the visit notes cite it, G2 S2.1)
    excludes: tuple[str, ...] = ()  # the codes it may not combine with on an order (`promotion_exclusion`)


@dataclass(frozen=True)
class Price:
    """A SKU's price in a price list version: per case and per piece; the standard list's also the purchase price (the
    factories' standard price), a channel list's whether Marketing's hero override set it."""

    case: float
    piece: float
    purchase_case: float | None = None
    is_override: bool = False


@dataclass(frozen=True)
class PriceList:
    """A version of a price list (`01` §5.1; CN-D31's price-list object — prices with their validity): the standard
    list (`channel` None, quarterly, with its `code`) or a channel's (monthly: the standard × `factor`). `prices` holds
    the SKUs sold in the version's period, by `sku_id` in the catalogue's order."""

    channel: str | None
    code: str | None
    valid_from: date
    valid_to: date
    factor: float
    prices: dict[int, Price]


@dataclass(frozen=True)
class Agreement:
    """A `price_agreement` on a customer's line (`loop.pricing.grant`): a rep's agreed discount (small retail) or a
    KAM's individual price (the other channels), a percentage off the line — or, for one SKU, its own price. The
    demand lifts the line by `discount_pct` (S6), so an individual SKU price carries its percentage too."""

    customer_id: int
    line: str
    discount_pct: float
    valid_from: date
    valid_to: date
    code: str = ""
    kind: str = "agreed_discount"  # agreed_discount · individual_price
    employee_id: int | None = None  # who agreed it
    sku_id: int | None = None  # None: the whole line
    price_case: float | None = None  # an individual SKU price
    budget_charge: float | None = None  # what it reserved of the granter's budget


@dataclass(frozen=True)
class OpenInvoice:
    """An invoice not yet paid, as the credit check reads it (`loop.pricing.credit_check`): G4 S4.1 opens them at
    invoicing and closes them on payment."""

    invoice_id: int
    customer_id: int
    issued_on: date
    due_on: date
    open_amount: float  # CZK without VAT, as the credit limits are


@dataclass
class Asset:
    """A PoS asset of a region's pool (`pos_asset`, G2 S2.3, `loop.pos`): in stock, out on `loan`, or written off."""

    material: str  # its pos_material code
    asset_no: str
    region: str  # the region whose pool holds it
    acquired_on: date
    cost: float
    condition: str  # new · good · worn · damaged
    written_off_on: date | None = None
    loan: int | None = None  # the pos_loan_id it is out on


@dataclass
class Loan:
    """A display lent to a store (`pos_loan`, G2 S2.3, `loop.pos`): its lines' demand at the store's delivery point
    × (1 + `uplift`) from `lent_on` until the day before `returned_on`; it is due back after `until`. A month-end's
    placement is one from the month-end on, and is cancelled — gone — when its customer is not active on `lent_on`
    (`loop.pos.revalidate`, G2 S2.5)."""

    pos_asset_id: int
    customer_id: int
    delivery_point_id: int
    store_id: int
    region: str
    lines: tuple[str, ...]  # the product lines it carries
    uplift: float  # the store's display_uplift (S5)
    lent_on: date
    until: date  # the last day of its term
    placed_by: int  # the rep's employee_id
    approved_by: int  # the area manager's
    returned_on: date | None = None
    condition_on_return: str | None = None


@dataclass
class Tourplan:
    """A rep's call on a store (`tourplan`, G2 S2.1, `loop.field`): every `cycle_weeks` weeks on `weekday` (1 Monday …
    5 Friday), in the weeks whose index (`loop.field.week_of`) ≡ `week` (mod `cycle_weeks`), from `valid_from` to
    `valid_to`, both included. The store's `weekday` and `week` are drawn once per store, so they follow it from rep to
    rep; a re-cut closes the tourplan in place (`valid_to`) and opens the next."""

    employee_id: int
    employee: str  # the rep's code, for the row's business key
    store: str  # osm_id
    store_id: int
    customer_id: int  # the store's customer (a potential one too), or the key account whose station it is
    delivery_point_id: int  # the store's: its customer's (or the key account's) point there
    area: str
    store_class: str
    cycle_weeks: int
    weekday: int
    week: int  # 0–7: the store's week in an 8-week cycle, taken mod `cycle_weeks`
    valid_from: date
    valid_to: date = OPEN
    calls: int = 0  # G2 S2.3: its sales visits that dealt with someone, and of them those that converted — the
    converted: int = 0  # store's under its tourplans, across reps (a re-cut carries them on); the give-aways follow


@dataclass
class World:
    """The state after `day`, the last simulated day (setup's world: the day before the warm-up)."""

    day: date
    customers: dict[int, Customer] = field(default_factory=dict)  # by customer_id, in setup order
    actors: dict[int, Actor] = field(default_factory=dict)  # by employee_id
    duties: list[Duty] = field(default_factory=list)  # append-only; a duty closes by its `valid_to`, set in place
    stock: dict[tuple[int, int], Stock] = field(default_factory=dict)  # (sku_id, dc_site_id)
    budgets: dict[tuple[int, date], Budget] = field(default_factory=dict)  # (employee_id, the quarter's first day)
    # Filled by the loop's stages; their item types arrive with them.
    open_orders: dict[int, object] = field(default_factory=dict)  # S1.4
    open_shipments: dict[int, object] = field(default_factory=dict)  # G3
    open_invoices: dict[int, OpenInvoice] = field(default_factory=dict)  # G4: by invoice_id
    tourplans: dict[int, Tourplan] = field(default_factory=dict)  # G2 S2.1: by tourplan_id, closed ones too
    tourplan_of: dict[str, int] = field(default_factory=dict)  # G2 S2.1: a store's (osm_id) open tourplan
    # G2 S2.3: the regions' PoS assets by pos_asset_id, every loan by pos_loan_id (returned ones too), each delivery
    # point's running loan (`lent`: delivery_point_id → pos_loan_id), and each area's conversion of the month: its
    # stores' median and the share of its calls at the stores above it (`loop.pos.medians`).
    assets: dict[int, Asset] = field(default_factory=dict)
    loans: dict[int, Loan] = field(default_factory=dict)
    lent: dict[int, int] = field(default_factory=dict)
    conversion: dict[str, tuple[float, float]] = field(default_factory=dict)
    promotions: dict[int, Promotion] = field(default_factory=dict)  # the calendar's, by promotion_id (setup; S1.3)
    agreements: dict[int, Agreement] = field(default_factory=dict)  # S1.3: by price_agreement_id (`add_agreement`)
    demand: Demand | None = None  # S1.2: the points, their base and baskets (`loop.demand`)
    # S1.3: the price lists in force on `day` — the standard one and each channel's; S1.4's period ends roll them
    # (the month-end the channel lists, the quarter-end the standard list by `price_index`).
    standard_list: PriceList | None = None
    channel_lists: dict[str, PriceList] = field(default_factory=dict)  # by channel
    price_index: dict[str, float] = field(default_factory=dict)  # S1.4: each line's standard price ÷ its base price
    # S1.4: what the tiered customers bought, by month (its first day) and customer_id, for the month-ends — the
    # last `policies.lifecycle.lookback_months`; and the planned gross of the quarter ahead by budget target (a rep
    # area's code, a KAM's account's), which a quarter's last month-end sets for the quarter-end's budgets.
    bought: dict[date, dict[int, Bought]] = field(default_factory=dict)
    planned_gross: dict[str, float] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)  # a numbered code's last number, by its prefix (AG-2024)
    purchases: Purchases | None = None  # G2 S2.2: what the reps' customers bought, by SKU, and the prompts' memory
    # G2 S2.3 (`loop.pos`) fills these two, which S2.2's display prompts read: a display's uplift at a small-retail
    # store's delivery point, by delivery_point_id then the line it would carry, and the display budget (CZK) a
    # region's manager has left of the month's; and `measure`, the small-retail points' cases by line and month.
    display_uplift: dict[int, dict[str, float]] = field(default_factory=dict)
    display_budget: dict[str, float] = field(default_factory=dict)
    measure: Measure | None = None

    def duty(self, kind: str, target: str, day: date) -> Duty | None:
        """Who holds `kind` for `target` on `day` (the rep of an area, the manager of a region, …): the first such
        duty in `duties`' order, found through an index by (kind, target) that is built again whenever `duties` is
        another list or another length, and never pickled. So `duties` is append-only: a duty ends by setting its
        `valid_to` in place (the index holds the same object), never by replacing or reordering the list's items."""
        index = self.__dict__.get("_duty_index")
        if index is None or index[0] is not self.duties or index[1] != len(self.duties):
            by_target: dict[tuple[str, str], list[Duty]] = {}
            for held in self.duties:
                by_target.setdefault((held.kind, held.target), []).append(held)
            index = (self.duties, len(self.duties), by_target)
            self.__dict__["_duty_index"] = index
        for held in index[2].get((kind, target), ()):
            if held.on(day):
                return held
        return None

    def add_agreement(self, agreement_id: int, agreement: Agreement) -> None:
        """The one way an agreement comes in: never over another (`ValueError`), and never removed after — it ends by
        its `valid_to`. The (customer, line) index of `agreements_on` takes it too, when built."""
        if agreement_id in self.agreements:
            raise ValueError(f"agreement {agreement_id} ({agreement.code}) is in the World already")
        self.agreements[agreement_id] = agreement
        index = self.__dict__.get("_agreement_index")
        if index is not None and index[0] is self.agreements and index[1] == len(self.agreements) - 1:
            index[2].setdefault((agreement.customer_id, agreement.line), []).append((agreement_id, agreement))
            self.__dict__["_agreement_index"] = (self.agreements, len(self.agreements), index[2])

    def agreements_on(self, customer_id: int, line: str, day: date) -> list[tuple[int, Agreement]]:
        """The (price_agreement_id, agreement)s of `customer_id`'s `line` running on `day`, in `agreements`' order,
        through an index by (customer, line) that `add_agreement` keeps. It is built again only for a restored `World`
        (it is never pickled) or another `agreements` dict, or as a guard when its length is not the dict's."""
        index = self.__dict__.get("_agreement_index")
        if index is None or index[0] is not self.agreements or index[1] != len(self.agreements):
            by_line: dict[tuple[int, str], list[tuple[int, Agreement]]] = {}
            for agreement_id, agreement in self.agreements.items():
                by_line.setdefault((agreement.customer_id, agreement.line), []).append((agreement_id, agreement))
            index = (self.agreements, len(self.agreements), by_line)
            self.__dict__["_agreement_index"] = index
        found = index[2].get((customer_id, line), ())
        return [(i, a) for i, a in found if a.valid_from <= day <= a.valid_to]

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state.pop("_duty_index", None)
        state.pop("_agreement_index", None)
        return state


def unordered_containers(value: object, path: str = "world") -> list[str]:
    """The paths inside `value` that hold a `set` or `frozenset` (empty for a deterministic state)."""
    if isinstance(value, set | frozenset):
        return [path]
    if isinstance(value, dict):
        return [p for key, item in value.items() for p in unordered_containers(item, f"{path}[{key!r}]")]
    if isinstance(value, list | tuple):
        return [p for i, item in enumerate(value) for p in unordered_containers(item, f"{path}[{i}]")]
    if is_dataclass(value) and not isinstance(value, type):
        return [p for f in fields(value) for p in unordered_containers(getattr(value, f.name), f"{path}.{f.name}")]
    return []


@dataclass(frozen=True)
class Checkpoint:
    """One row of `_sim_checkpoint`: the `World` after `day`, pickled, with the sha256 of the pickle."""

    day: date
    sha256: str
    blob: bytes

    @classmethod
    def of(cls, world: World, day: date | None = None) -> Checkpoint:
        problems = unordered_containers(world)
        if problems:
            raise TypeError(f"the world holds unordered containers, its pickle would not be stable: {problems[:3]}")
        blob = pickle.dumps(world, protocol=PROTOCOL)
        return cls(day or world.day, hashlib.sha256(blob).hexdigest(), blob)

    def restore(self) -> World:
        """The `World` again; `ValueError` when the blob is not the one its sha256 names."""
        if hashlib.sha256(self.blob).hexdigest() != self.sha256:
            raise ValueError(f"the checkpoint of {self.day} does not match its sha256")
        world = pickle.loads(self.blob)  # the trust boundary: the module docstring
        if not isinstance(world, World):
            raise ValueError(f"the checkpoint of {self.day} holds a {type(world).__name__}, not a World")
        return world

    def save(self, run: Path, keep: Iterable[date] = ()) -> None:
        """Write this checkpoint to `run/_sim_checkpoint.csv`, keeping the rows of the `keep` days already there (the
        warm-up's last day) and dropping the others. The blob is bytea's hex text (`\\x…`), as COPY reads it.

        A kept row is copied as its text, never decoded and encoded again. The file is written beside the old one and
        moved into its place, so a failed write leaves the old file whole."""
        days = frozenset(keep)
        kept = [row for row in _texts(run) if date.fromisoformat(row[0]) in days and row[0] != self.day.isoformat()]
        rows = sorted([*kept, (self.day.isoformat(), self.sha256, "\\x" + self.blob.hex())])
        run.mkdir(parents=True, exist_ok=True)
        path = run / CHECKPOINT_FILE
        partial = path.with_name(f"{CHECKPOINT_FILE}.partial")
        with partial.open("w", encoding="utf-8", newline="") as out:
            writer = csv.writer(out, lineterminator="\n")
            writer.writerow(CHECKPOINT_HEADER)
            writer.writerows(rows)
        os.replace(partial, path)

    @classmethod
    def load(cls, run: Path, day: date | None = None) -> Checkpoint:
        """The checkpoint of `day` in `run`, or the last one; `FileNotFoundError` when there is none. Only that row's
        blob is decoded."""
        rows = sorted(row for row in _texts(run) if day is None or row[0] == day.isoformat())
        if not rows:
            raise FileNotFoundError(f"no checkpoint{f' of {day}' if day else ''} in {run}")
        return _decoded(rows[-1])


def load_all(run: Path) -> list[Checkpoint]:
    """Every checkpoint in `run/_sim_checkpoint.csv`, in day order (none when the file is missing)."""
    return [_decoded(row) for row in sorted(_texts(run))]


def _decoded(row: tuple[str, str, str]) -> Checkpoint:
    day, sha256, blob = row
    return Checkpoint(date.fromisoformat(day), sha256, bytes.fromhex(blob.removeprefix("\\x")))


def _texts(run: Path) -> list[tuple[str, str, str]]:
    """The rows of `run/_sim_checkpoint.csv` as their text: (day, sha256, the blob's hex)."""
    path = run / CHECKPOINT_FILE
    if not path.is_file():
        return []
    csv.field_size_limit(2**31 - 1)
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        if tuple(next(reader, ())) != CHECKPOINT_HEADER:
            raise ValueError(f"{path} does not start with {','.join(CHECKPOINT_HEADER)}")
        return [(day, sha256, blob) for day, sha256, blob in reader]


@dataclass(frozen=True)
class Meta:
    """`_sim_meta`: what `load-world` and `prolong` check a database against (contracts §2).

    `world_sha256` hashes `data/world.yaml` as `stories_sha256` hashes the stories; `build_id` names the inputs
    (the seed, the world, the stories, the inventory, the reference files, the package version, the generator's
    revision), so two builds of the same inputs carry the same id and their CSVs stay byte-identical."""

    master_seed: int
    stories_sha256: str
    world_sha256: str
    last_day_done: date
    build_id: str

    def rows(self) -> list[tuple[str, str]]:
        return [
            ("master_seed", str(self.master_seed)),
            ("stories_sha256", self.stories_sha256),
            ("world_sha256", self.world_sha256),
            ("last_day_done", self.last_day_done.isoformat()),
            ("build_id", self.build_id),
        ]

    def save(self, run: Path) -> None:
        """Written beside the old file and moved into its place, so a failed write leaves the old one whole."""
        run.mkdir(parents=True, exist_ok=True)
        path = run / META_FILE
        partial = path.with_name(f"{META_FILE}.partial")
        with partial.open("w", encoding="utf-8", newline="") as out:
            writer = csv.writer(out, lineterminator="\n")
            writer.writerow(["key", "value"])
            writer.writerows(self.rows())
        os.replace(partial, path)

    @classmethod
    def load(cls, run: Path) -> Meta:
        with (run / META_FILE).open(encoding="utf-8", newline="") as source:
            values = {row["key"]: row["value"] for row in csv.DictReader(source)}
        return cls(
            int(values["master_seed"]),
            values["stories_sha256"],
            values["world_sha256"],
            date.fromisoformat(values["last_day_done"]),
            values["build_id"],
        )


def digest(root: Path, paths: Iterable[Path], *extra: str) -> str:
    """The sha256 of the files at `paths` (each with its path below `root`), then of `extra`, in the order given."""
    out = hashlib.sha256()
    for path in paths:
        out.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    for text in extra:
        out.update(text.encode() + b"\0")
    return out.hexdigest()
