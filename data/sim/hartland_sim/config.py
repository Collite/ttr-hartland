"""The three configs the generator reads — each a typed view over a committed, schema-checked file.

- `WorldConfig` ← `data/world.yaml`: every number of `01-company.md` (sites, channels, salesforce, pricing,
  promotions, policies, finance, calendar, dials, `master_seed`); validated by `data/schemas/world.schema.json`.
- `InventoryConfig` ← `data/inventory.yaml`: one entry per table — the one source of the DDL, the load order and
  the db layer (contracts §1); validated by `data/schemas/inventory.schema.json`, then as a whole graph.
- `StoriesConfig` ← `data/stories/s1…s6.yaml` + `anomalies.yaml`: the answer key (contracts §3); validated by
  `data/schemas/story.schema.json` and `anomalies.schema.json`, then as a whole.

A missing or mistyped key fails at load with its path, never later inside the simulation: `ConfigError` lists every
problem of the file at once, each as `<file>: <path>: <what>`. The typed view is read-only — mappings are
`MappingProxyType`, lists are tuples, dates are `date`s.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from jsonschema import Draft202012Validator

DATA_DIR = Path(__file__).resolve().parents[2]
WORLD_FILE = DATA_DIR / "world.yaml"
SCHEMA_DIR = DATA_DIR / "schemas"


class ConfigError(ValueError):
    """A config file that breaks its schema; the message lists every problem, each with its path."""


# ── loading, validating, building ────────────────────────────────────────────


def read_yaml(path: Path) -> Any:
    """The YAML at `path` with its dates as ISO strings (the schema speaks JSON); `ConfigError` when it is not YAML."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"{path.name}: not YAML ({error})") from None
    return _isodates(data)


def _isodates(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _isodates(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_isodates(item) for item in value]
    return value


def validate(data: Any, schema_name: str, what: str) -> None:
    """Every problem of `data` against `data/schemas/<schema_name>.schema.json`, raised as one `ConfigError`."""
    schema = json.loads((SCHEMA_DIR / f"{schema_name}.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)
    problems = sorted(_problem(error, data) for error in validator.iter_errors(data))
    if problems:
        raise ConfigError("\n".join(f"{what}: {problem}" for problem in problems))


def _problem(error: Any, data: Any) -> str:
    path = _path(error.absolute_path, data)
    if error.validator == "required":
        missing = [name for name in error.validator_value if name not in error.instance]
        return f"{'.'.join([*path, missing[0]])}: missing"
    return f"{'.'.join(path) or '(top)'}: {error.message}"


def _path(parts: Any, data: Any) -> list[str]:
    """The path's parts as text; a list item with a `name` is called by it (`tables.customer.columns.code`)."""
    out = []
    node = data
    for part in parts:
        node = node[part] if isinstance(node, dict | list) else None
        name = node.get("name") if isinstance(part, int) and isinstance(node, dict) else None
        out.append(name if isinstance(name, str) else str(part))
    return out


def build[T](cls: type[T], data: Any) -> T:
    """`data` (already validated) as the frozen dataclass `cls`, nested types and all."""
    return typing.cast(T, _convert(cls, data))


def _convert(tp: Any, value: Any) -> Any:
    if tp is Any:
        return _frozen(value)
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        hints = typing.get_type_hints(tp)
        names = {field.name for field in dataclasses.fields(tp)}
        return tp(**{name: _convert(hints[name], value[name]) for name in names if name in value})
    if origin is types.UnionType or origin is typing.Union:
        options = typing.get_args(tp)
        if value is None and type(None) in options:
            return None
        errors = []
        for option in options:
            if option is type(None):
                continue
            try:
                return _convert(option, value)
            except (TypeError, ValueError) as error:
                errors.append(error)
        raise TypeError(f"{value!r} is none of {options}: {errors}")
    if origin is Literal:
        if value not in typing.get_args(tp):
            raise ValueError(f"{value!r} is not one of {typing.get_args(tp)}")
        return value
    if origin in (Mapping, dict) or tp is Mapping:
        key_type, item_type = typing.get_args(tp)
        return types.MappingProxyType({_convert(key_type, k): _convert(item_type, v) for k, v in value.items()})
    if origin is tuple:
        args = typing.get_args(tp)
        if not isinstance(value, list):
            raise TypeError(f"{value!r} is not a list")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], item) for item in value)
        return tuple(_convert(arg, item) for arg, item in zip(args, value, strict=True))
    if tp is date:
        return date.fromisoformat(value)
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"{value!r} is not a number")
        return float(value)
    if tp in (int, str, bool):
        if not isinstance(value, tp) or (tp is int and isinstance(value, bool)):
            raise TypeError(f"{value!r} is not {tp.__name__}")
        return value
    raise TypeError(f"no conversion to {tp!r}")


def _frozen(value: Any) -> Any:
    if isinstance(value, dict):
        return types.MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_frozen(item) for item in value)
    return value


# ── WorldConfig (data/world.yaml) ────────────────────────────────────────────

Channel = Literal["big_retail", "small_retail", "special_retail", "gas_ka", "own_shops", "web", "vending", "horeca"]
Phase = Literal["potential", "occasional", "regular", "vip"]


@dataclass(frozen=True)
class History:
    warm_up_from: date
    first_day: date
    until: date


@dataclass(frozen=True)
class Company:
    center: str
    factories: tuple[str, ...]
    currency: str
    timezone: str
    locale: str
    fiscal_year_start_month: int


@dataclass(frozen=True)
class Sites:
    hq: str
    dcs: Mapping[str, tuple[str, ...]]  # DC site code → the sales regions it serves
    own_shops: int


@dataclass(frozen=True)
class Suppliers:
    coffee_split: Mapping[str, float]
    fallback: str
    lead_time_days: Mapping[str, int]
    reliability_pct: Mapping[str, float]


@dataclass(frozen=True)
class ChannelSettings:
    order_grain: Literal["case", "piece"]
    order_cycle_days: int
    accounts: tuple[str, ...] = ()
    dc_level_orders: tuple[str, ...] = ()
    customers: int | None = None
    active_share_at_start: float | None = None
    since_from: date | None = None
    since_until: date | None = None
    consumers: int | None = None
    operators: int | None = None
    direct_customers: int | None = None
    wholesalers: int | None = None
    potential: int | None = None


@dataclass(frozen=True)
class Listing:
    retail: Literal["all"] | tuple[str, ...]
    professional: Literal["all"] | tuple[str, ...]


@dataclass(frozen=True)
class Channels:
    big_retail: ChannelSettings
    small_retail: ChannelSettings
    special_retail: ChannelSettings
    gas_ka: ChannelSettings
    own_shops: ChannelSettings
    web: ChannelSettings
    vending: ChannelSettings
    horeca: ChannelSettings
    listings: Mapping[Channel, Listing]


@dataclass(frozen=True)
class Salesforce:
    reps: int
    area_managers: int
    head_of_key_accounts: int
    kams: Mapping[str, int]
    product_managers: int
    dispatchers: int
    supply: int
    finance: int
    marketing: int
    management: int
    shop_managers: int
    rep_area_moves: int


@dataclass(frozen=True)
class HeroOverride:
    sku: str
    channel: Channel
    factor: float


@dataclass(frozen=True)
class VolumeTier:
    code: str
    min_net_revenue: float
    pct: float


@dataclass(frozen=True)
class CrossSellTier:
    code: str
    min_lines: int
    pct: float


@dataclass(frozen=True)
class Tiers:
    volume: tuple[VolumeTier, ...]
    cross_sell: tuple[CrossSellTier, ...]
    cross_sell_both_brands_pct: float
    cross_sell_max_pct: float


@dataclass(frozen=True)
class Budgets:
    rep_discount_pct_of_planned_gross: float
    kam_discount_pct_of_planned_gross: float


@dataclass(frozen=True)
class Pricing:
    channel_factor: Mapping[Channel, float]
    hero_overrides: tuple[HeroOverride, ...]
    tiers: Tiers
    max_agreed_discount_pct: Mapping[str, float]  # product line code → %
    budgets: Budgets
    retro_bonus_pct: Mapping[str, float]  # chain code or "default" → %


@dataclass(frozen=True)
class Promotions:
    per_year: int
    tv_share: float
    weeks: tuple[int, int]
    mechanics: tuple[str, ...]
    exclusion_pairs_per_year: tuple[int, int]
    key_account_exclusive_per_year: int
    code: str


@dataclass(frozen=True)
class Cover:
    min_days: int
    max_days: int


@dataclass(frozen=True)
class Lifecycle:
    phases: tuple[Phase, ...]
    lookback_months: int
    regular_min_order_months: int
    vip_min_net_revenue: float


@dataclass(frozen=True)
class Policies:
    stock: Mapping[str, Cover]  # "default" · "professional"
    credit_limit: Mapping[Channel, int]
    lifecycle: Lifecycle
    prompt_caps: Mapping[Phase, int]
    seasonal_exclusion: Mapping[str, tuple[int, ...]]  # product line code → months


@dataclass(frozen=True)
class PaymentBehaviour:
    on_time: float
    late_14: float
    late_30: float
    late_more: float


@dataclass(frozen=True)
class Hold:
    overdue_days_over: int
    exposure_over_limit: bool


@dataclass(frozen=True)
class Vat:
    goods: float
    served: float


@dataclass(frozen=True)
class Finance:
    payment_terms_days: Mapping[Channel, int]
    payment_behaviour: PaymentBehaviour
    hold: Hold
    dunning_days_overdue: tuple[int, ...]
    return_line_share: float
    vat_pct: Vat


@dataclass(frozen=True)
class CoffeeJump:
    starts: date
    business: str
    pct: float


@dataclass(frozen=True)
class Calendar:
    standard_price_list: str
    channel_price_list: str
    price_drift_pct: tuple[float, float]
    coffee_jump: CoffeeJump


@dataclass(frozen=True)
class Field:
    cycle_weeks: Mapping[str, int]  # store class (or "potential") → weeks
    adherence: float
    visit_minutes: tuple[int, int]


@dataclass(frozen=True)
class Delivery:
    carriers_per_dc: int
    parcel_carriers: int
    inbound_carriers: int
    events_per_route_day: float


@dataclass(frozen=True)
class Dials:
    receipts_per_shop_day: int
    observation_lines: Literal["all", "store_class"]
    stock_movement_picks: Literal["per_line", "per_order"]


@dataclass(frozen=True)
class WorldConfig:
    """`data/world.yaml`, validated and typed."""

    master_seed: int
    history: History
    company: Company
    sites: Sites
    suppliers: Suppliers
    channels: Channels
    salesforce: Salesforce
    pricing: Pricing
    promotions: Promotions
    policies: Policies
    finance: Finance
    calendar: Calendar
    field: Field
    delivery: Delivery
    dials: Dials

    @classmethod
    def load(cls, path: Path = WORLD_FILE) -> WorldConfig:
        data = read_yaml(path)
        validate(data, "world", path.name)
        return build(cls, data)


# ── InventoryConfig (data/inventory.yaml) ────────────────────────────────────

INVENTORY_FILE = DATA_DIR / "inventory.yaml"

Writer = Literal["setup", "daily", "month_end", "quarter_end", "year_end", "documents"]


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    nullable: bool = False
    ref: str | None = None  # "<table>.<column>"
    coded: str | None = None  # a name of `InventoryConfig.codes`
    label_cs: str | None = None


@dataclass(frozen=True)
class ForeignKey:
    column: str
    table: str
    target: str


@dataclass(frozen=True)
class Table:
    name: str
    group: str
    grain: str
    columns: tuple[Column, ...]
    pk: tuple[str, ...]
    load_order: int
    writer: tuple[Writer, ...]
    rows_expected: int
    serves: tuple[str, ...]
    id_key: tuple[str, ...] | None = None
    unique: tuple[tuple[str, ...], ...] = ()
    indexes: tuple[tuple[str, ...], ...] = ()

    @property
    def fks(self) -> tuple[ForeignKey, ...]:
        """The foreign keys, one per column with a `ref`."""
        return tuple(
            ForeignKey(column.name, *column.ref.split(".", 1)) for column in self.columns if column.ref is not None
        )

    def column(self, name: str) -> Column:
        (found,) = [column for column in self.columns if column.name == name]
        return found


@dataclass(frozen=True)
class View:
    name: str
    kind: Literal["materialized_view", "view"]
    group: str
    grain: str
    sql_file: str  # relative to data/ddl/
    load_order: int
    serves: tuple[str, ...]


@dataclass(frozen=True)
class InventoryConfig:
    """`data/inventory.yaml`, validated: the codes, the tables in `load_order`, the views (contracts §1).

    Beyond the schema, `load` checks what only the whole file can tell: names unique, every `ref` naming a table's
    key column, every `coded` a code list, the key and index columns existing, and every foreign key pointing to a
    lower `load_order` (or to its own table) — so the load order is a topological order of the graph."""

    schema: str
    codes: Mapping[str, Mapping[str, str]]  # name → value → Czech label
    tables: tuple[Table, ...]  # in load_order
    views: tuple[View, ...]  # in load_order

    @classmethod
    def load(cls, path: Path = INVENTORY_FILE) -> InventoryConfig:
        data = read_yaml(path)
        validate(data, "inventory", path.name)
        for table in data["tables"]:
            if isinstance(table["writer"], str):
                table["writer"] = [table["writer"]]
        data["tables"].sort(key=lambda table: table["load_order"])
        data["views"].sort(key=lambda view: view["load_order"])
        inventory = build(cls, data)
        problems = inventory.problems()
        if problems:
            raise ConfigError("\n".join(f"{path.name}: {problem}" for problem in problems))
        return inventory

    def table(self, name: str) -> Table:
        (found,) = [table for table in self.tables if table.name == name]
        return found

    def problems(self) -> list[str]:
        problems = []
        names = [table.name for table in self.tables] + [view.name for view in self.views]
        problems += [f"{name}: named twice" for name in sorted({n for n in names if names.count(n) > 1})]
        orders = [table.load_order for table in self.tables] + [view.load_order for view in self.views]
        problems += [f"load_order {order}: used twice" for order in sorted({o for o in orders if orders.count(o) > 1})]
        by_name = {table.name: table for table in self.tables}
        for table in self.tables:
            problems += self._table_problems(table, by_name)
        last_table = max(table.load_order for table in self.tables)
        problems += [
            f"{view.name}: a view loads after every table" for view in self.views if view.load_order <= last_table
        ]
        return problems

    def _table_problems(self, table: Table, by_name: Mapping[str, Table]) -> list[str]:
        problems = []
        columns = [column.name for column in table.columns]
        problems += [
            f"{table.name}.{name}: named twice" for name in sorted({n for n in columns if columns.count(n) > 1})
        ]
        named = [
            ("pk", table.pk),
            *(("unique", key) for key in table.unique),
            *(("index", key) for key in table.indexes),
        ]
        for what, key in named:
            problems += [f"{table.name}: {what} names no column {name!r}" for name in key if name not in columns]
        if table.id_key is not None and len(table.pk) > 1:
            problems += [f"{table.name}: an id_key with a composite pk {list(table.pk)}"]
        for column in table.columns:
            if column.coded is not None and column.coded not in self.codes:
                problems.append(f"{table.name}.{column.name}: coded {column.coded!r} is no code list")
            if column.coded is not None and column.type != "text":
                problems.append(f"{table.name}.{column.name}: coded, but {column.type}")
        for fk in table.fks:
            target = by_name.get(fk.table)
            if target is None:
                problems.append(f"{table.name}.{fk.column}: ref {fk.table}.{fk.target} names no table")
            elif target.pk != (fk.target,):
                problems.append(f"{table.name}.{fk.column}: ref {fk.table}.{fk.target} is not {fk.table}'s primary key")
            elif target.column(fk.target).type != table.column(fk.column).type:
                problems.append(f"{table.name}.{fk.column}: {table.column(fk.column).type}, its ref {fk.target} not")
            elif target.load_order >= table.load_order and target is not table:
                problems.append(
                    f"{table.name}.{fk.column}: ref {fk.table} loads at {target.load_order}, not before"
                    f" {table.load_order}"
                )
        return problems


# ── StoriesConfig (data/stories/) ────────────────────────────────────────────

STORIES_DIR = DATA_DIR / "stories"

Direction = Literal[">=", "<=", ">", "<", "between"]

# A story passes when `metric direction threshold`; the null world must give the opposite.
OPPOSITE: Mapping[str, str] = types.MappingProxyType({">=": "<", "<=": ">", ">": "<=", "<": ">="})


@dataclass(frozen=True)
class Detector:
    sql: str  # relative to data/stories/
    metric: str
    threshold: float | tuple[float, float]
    direction: Direction
    null_world: Direction


@dataclass(frozen=True)
class Story:
    story: str
    title_cs: str
    enabled: bool
    seed_stream: str
    parameters: Mapping[str, Any]
    detector: Detector
    also: tuple[Detector, ...] = ()

    @property
    def detectors(self) -> tuple[Detector, ...]:
        return (self.detector, *self.also)


@dataclass(frozen=True)
class Probe:
    sql: str  # relative to data/stories/
    expect: str  # ">= n" · "= n" · "rows"


@dataclass(frozen=True)
class Anomaly:
    id: str
    title_cs: str
    enabled: bool
    parameters: Mapping[str, Any]
    probe: Probe


@dataclass(frozen=True)
class RepPin:
    """A pinned rep of S3, by the area of their first assignment (`slot`), with the four skills and the traits."""

    slot: str
    profile: str
    conversion: float
    acquisition: float
    retention: float
    cross_sell: float
    skips_visits: bool = False
    records_nothing: bool = False


@dataclass(frozen=True)
class CustomerPin:
    """A customer a planted case needs, by a rule resolved at setup: the `rank`-th matching customer in OSM order."""

    anomaly: str
    trait: str
    channel: str
    area: str
    rank: int
    chain: str | None = None
    store_class: str | None = None


@dataclass(frozen=True)
class Pins:
    reps: tuple[RepPin, ...]
    customers: tuple[CustomerPin, ...]


@dataclass(frozen=True)
class StoriesConfig:
    """`data/stories/`, validated: the six stories (`s1.yaml` … `s6.yaml`, `data/schemas/story.schema.json`), the
    planted cases and the pins (`anomalies.yaml`, `anomalies.schema.json`) — contracts §3.

    Beyond the schemas, `load` checks that each file holds the story its name says, the six are there once each, the
    anomalies' ids are unique, every SQL file exists, and every `null_world` is the opposite of its `direction`.
    `sha256` hashes the YAML files, for `_sim_meta.stories_sha256` (contracts §2)."""

    stories: tuple[Story, ...]
    anomalies: tuple[Anomaly, ...]
    pins: Pins
    sha256: str

    @classmethod
    def load(cls, directory: Path = STORIES_DIR) -> StoriesConfig:
        problems: list[str] = []
        stories = []
        for path in sorted(directory.glob("s[0-9]*.yaml")):
            try:
                data = read_yaml(path)
                validate(data, "story", path.name)
            except ConfigError as error:
                problems.append(str(error))
                continue
            if data["story"] != path.stem.upper():
                problems.append(f"{path.name}: holds story {data['story']}")
            stories.append(build(Story, data))
        try:
            data = read_yaml(directory / "anomalies.yaml")
            validate(data, "anomalies", "anomalies.yaml")
            anomalies = tuple(build(Anomaly, item) for item in data["anomalies"])
            pins = build(Pins, data["pins"])
        except ConfigError as error:
            problems.append(str(error))
            anomalies, pins = (), Pins((), ())
        ids = [story.story for story in stories]
        if sorted(ids) != [f"S{n}" for n in range(1, 7)] and not problems:
            problems.append(f"the stories are {sorted(ids)}, not S1…S6 once each")
        anomaly_ids = [anomaly.id for anomaly in anomalies]
        problems += [
            f"anomalies.yaml: {id} twice" for id in sorted({i for i in anomaly_ids if anomaly_ids.count(i) > 1})
        ]
        for story in stories:
            for detector in story.detectors:
                problems += _detector_problems(directory, f"s{story.story[1:]}.yaml", detector)
        for anomaly in anomalies:
            if not (directory / anomaly.probe.sql).is_file():
                problems.append(f"anomalies.yaml: {anomaly.id}: no {anomaly.probe.sql}")
        if problems:
            raise ConfigError("\n".join(problems))
        digest = hashlib.sha256()
        for path in sorted(directory.glob("*.yaml")):
            digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
        return cls(tuple(sorted(stories, key=lambda story: story.story)), anomalies, pins, digest.hexdigest())

    def story(self, id: str) -> Story:
        (found,) = [story for story in self.stories if story.story == id]
        return found

    def anomaly(self, id: str) -> Anomaly:
        (found,) = [anomaly for anomaly in self.anomalies if anomaly.id == id]
        return found


def _detector_problems(directory: Path, name: str, detector: Detector) -> list[str]:
    problems = []
    if not (directory / detector.sql).is_file():
        problems.append(f"{name}: no {detector.sql}")
    if (detector.direction == "between") != isinstance(detector.threshold, tuple):
        problems.append(f"{name}: {detector.sql}: `between` takes [low, high], the others one number")
    if detector.direction != "between" and OPPOSITE[detector.direction] != detector.null_world:
        problems.append(
            f"{name}: {detector.sql}: the null world must be the opposite of {detector.direction!r},"
            f" not {detector.null_world!r}"
        )
    return problems
