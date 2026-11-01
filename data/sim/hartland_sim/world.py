"""The simulation's state and its daily checkpoint (contracts §2, architecture §2).

`World` holds everything the day loop carries from one day to the next — customers with their condition, phase
episodes and basket, stock per SKU × DC, open orders, shipments and invoices, tourplans, loans, running promotions,
budgets, the actors' hidden skills and their duties. Setup (`setup.build`) makes the opening `World`; the loop moves
it a day at a time. The tables that hold the state the loop changes (`customer`, `lifecycle_phase`, `employee`,
`assignment`, `discount_budget`) are rendered from the `World` when a build ends (`setup.opening.state_rows`), so it
holds every value of their rows: the customer's name, the actor's name and manager.

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
    """A retail customer's conditions for a month (`customer_condition`, S1.3/S1.4)."""

    month: date
    volume_tier: str | None = None
    volume_pct: float = 0.0
    cross_sell_tier: str | None = None
    cross_sell_pct: float = 0.0


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
    basket: dict[int, float] = field(default_factory=dict)  # product_line_id → latent cases not yet ordered

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
    """A rep's or KAM's discount budget for a quarter (`discount_budget`)."""

    budget_amount: float
    used_amount: float = 0.0


@dataclass
class World:
    """The state after `day`, the last simulated day (setup's world: the day before the warm-up)."""

    day: date
    customers: dict[int, Customer] = field(default_factory=dict)  # by customer_id, in setup order
    actors: dict[int, Actor] = field(default_factory=dict)  # by employee_id
    duties: list[Duty] = field(default_factory=list)
    stock: dict[tuple[int, int], Stock] = field(default_factory=dict)  # (sku_id, dc_site_id)
    budgets: dict[tuple[int, date], Budget] = field(default_factory=dict)  # (employee_id, the quarter's first day)
    # Filled by the loop's stages; their item types arrive with them.
    open_orders: dict[int, object] = field(default_factory=dict)  # S1.4
    open_shipments: dict[int, object] = field(default_factory=dict)  # G3
    open_invoices: dict[int, object] = field(default_factory=dict)  # G4
    tourplans: dict[int, object] = field(default_factory=dict)  # G2: by employee_id
    loans: dict[int, object] = field(default_factory=dict)  # G2: by pos_asset_id
    promotions: dict[int, object] = field(default_factory=dict)  # S1.3: the running ones, by promotion_id

    def duty(self, kind: str, target: str, day: date) -> Duty | None:
        """Who holds `kind` for `target` on `day` (the rep of an area, the manager of a region, …)."""
        held = [duty for duty in self.duties if duty.kind == kind and duty.target == target and duty.on(day)]
        return held[0] if held else None


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
    (the seed, the world, the stories, the inventory, the reference files, the package version), so two builds of the
    same inputs carry the same id and their CSVs stay byte-identical."""

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
        run.mkdir(parents=True, exist_ok=True)
        with (run / META_FILE).open("w", encoding="utf-8", newline="") as out:
            writer = csv.writer(out, lineterminator="\n")
            writer.writerow(["key", "value"])
            writer.writerows(self.rows())

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


def digest(paths: Iterable[Path], *extra: str) -> str:
    """The sha256 of the files at `paths` (each with its name), then of `extra`, in the order given."""
    out = hashlib.sha256()
    for path in paths:
        out.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    for text in extra:
        out.update(text.encode() + b"\0")
    return out.hexdigest()
