"""The three configs the generator reads — each a typed view over a committed, schema-checked file.

- `WorldConfig` ← `data/world.yaml`: every number of `01-company.md` (sites, channels, salesforce, pricing,
  promotions, policies, finance, calendar, dials, `master_seed`); validated by `data/schemas/world.schema.json`.
- `StoriesConfig` ← `data/stories/s1…s6.yaml` + `anomalies.yaml`: the answer key (contracts §3). G0 S0.4 T3.
- `InventoryConfig` ← `data/inventory.yaml`: one entry per table — the one source of the DDL, the load order and
  the db layer (contracts §1). G0 S0.4 T4.

A missing or mistyped key fails at load with its path, never later inside the simulation: `ConfigError` lists every
problem of the file at once, each as `<file>: <path>: <what>`. The typed view is read-only — mappings are
`MappingProxyType`, lists are tuples, dates are `date`s.
"""

from __future__ import annotations

import dataclasses
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
    problems = sorted(_problem(error) for error in validator.iter_errors(data))
    if problems:
        raise ConfigError("\n".join(f"{what}: {problem}" for problem in problems))


def _problem(error: Any) -> str:
    path = [str(part) for part in error.absolute_path]
    if error.validator == "required":
        missing = [name for name in error.validator_value if name not in error.instance]
        return f"{'.'.join([*path, missing[0]])}: missing"
    return f"{'.'.join(path) or '(top)'}: {error.message}"


def build[T](cls: type[T], data: Any) -> T:
    """`data` (already validated) as the frozen dataclass `cls`, nested types and all."""
    return typing.cast(T, _convert(cls, data))


def _convert(tp: Any, value: Any) -> Any:
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


# ── the other two (G0 S0.4 T3, T4) ───────────────────────────────────────────


@dataclass(frozen=True)
class StoriesConfig:
    """`data/stories/`, validated: the six stories, the anomalies, the pins (contracts §3; G0 S0.4 T3)."""

    @classmethod
    def load(cls, directory: Path) -> StoriesConfig:
        raise NotImplementedError("G0 S0.4 T3")


@dataclass(frozen=True)
class InventoryConfig:
    """`data/inventory.yaml`, validated: tables, columns, keys, `load_order` (contracts §1; G0 S0.4 T4)."""

    @classmethod
    def load(cls, path: Path) -> InventoryConfig:
        raise NotImplementedError("G0 S0.4 T4")
