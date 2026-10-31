"""The promotion calendar from its templates (⚑NH-8): `data/reference/promotion_templates.yaml` holds ~25 shapes, one
promotion each a year, and a year's calendar is every template instantiated for that year — so the codes are stable
(`world.yaml promotions.code`: `PR-<year>-<month>-<line>-<tag>`) and a rebuild gives the same promotions. Setup seeds
the history's years from it (`05` §2 step 6); `prolong` goes past the committed years the same way (G5 S5.3).

A template starts on the Monday of its ISO `start_week` and runs `weeks` weeks; the start must fall in its `month`,
which the code carries. Exclusions pair templates by `<line>-<tag>`; each year's calendar excludes the pair both ways.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from hartland_sim.config import ConfigError, build, read_yaml, validate
from hartland_sim.io.reference import REFERENCE_DIR

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
