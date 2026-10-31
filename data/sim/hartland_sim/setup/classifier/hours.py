"""`opening_hours` → the weekly hours and whether the store opens on Sunday (G0 S0.3 T4; `01` §7, `04` S4).

A small, strict reader of the OSM `opening_hours` forms the Czech store set uses: rules separated by `;`, each an
optional weekday selector (`Mo-Fr`, `Sa,Su`, `Mo-We,Fr`) and either time spans (`07:00-12:00,13:00-17:00`, past
midnight as `22:00-06:00` or `-24:00`) or `off`/`closed`; `24/7`. A later rule replaces the days it names, as in OSM.
Public and school holidays (`PH`, `SH`) are left out of a selector, and a rule for them alone is skipped: the week is
an ordinary one. Anything else — months, dates, weeks, comments, `sunrise` — makes the value unknown (`None`), and so
does a week with no open hour at all (`off`, `closed`, `Mo off`): the store takes its class's hours
(`setup.market`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DAYS = ("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su")
_DAY = "|".join(DAYS)
_SELECTOR = re.compile(rf"(?:{_DAY})(?:-(?:{_DAY}))?(?:,(?:{_DAY})(?:-(?:{_DAY}))?)*")
_SPAN = re.compile(r"(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})\+?")
# "Mo-Fr 07:00-18:00, Sa 08:00-12:00": a comma after a time, before a weekday, meant a `;`.
_RULE_AFTER_COMMA = re.compile(rf"(?<=\d|f|d),\s*(?=(?:{_DAY}|PH|SH)\b)")
_SPACE_IN_SELECTOR = re.compile(rf"(?<={_DAY}),\s+(?=(?:{_DAY}|PH|SH)\b)")  # "Sa, Su 08:00-12:00"


@dataclass(frozen=True)
class Week:
    weekly_hours: float
    sunday_open: bool


def parse(value: str) -> Week | None:
    """The week of an `opening_hours` value, or None when it cannot be read."""
    value = value.strip()
    if not value:
        return None
    if value == "24/7":
        return Week(168.0, True)
    minutes = [0] * 7
    value = _RULE_AFTER_COMMA.sub(";", _SPACE_IN_SELECTOR.sub(",", value))
    for rule in (part.strip() for part in value.split(";")):
        if not rule:
            continue
        days, times = _split(rule)
        if days is None:
            return None
        if not days:  # a public- or school-holiday rule
            continue
        if times in ("off", "closed"):
            open_minutes = 0
        else:
            spans = [_SPAN.fullmatch(span) for span in times.split(",")]
            if not times or not all(spans):
                return None
            open_minutes = sum(_length(span) for span in spans if span)
            if open_minutes > 24 * 60:
                return None
        for day in days:
            minutes[day] = open_minutes
    if not any(minutes):
        return None
    return Week(round(sum(minutes) / 60, 2), minutes[6] > 0)


def _split(rule: str) -> tuple[list[int] | None, str]:
    """A rule → its weekday indexes (every day when it names none; none for a holiday rule) and its time part."""
    head, _, rest = rule.partition(" ")
    parts = head.split(",")
    days = [part for part in parts if part not in ("PH", "SH")]  # "Su,PH off": the Sunday part
    if parts and not days:
        return [], ""
    head = ",".join(days)
    if _SELECTOR.fullmatch(head):
        return _days(head), rest.strip().replace(" ", "")
    if head[:1].isdigit() or head in ("off", "closed"):
        return list(range(7)), rule.replace(" ", "")
    return None, ""


def _days(selector: str) -> list[int]:
    days: list[int] = []
    for part in selector.split(","):
        first, _, last = part.partition("-")
        start, end = DAYS.index(first), DAYS.index(last or first)
        days += [day % 7 for day in range(start, end + 1 if end >= start else end + 8)]
    return days


def _length(span: re.Match[str]) -> int:
    open_h, open_m, close_h, close_m = (int(group) for group in span.groups())
    start, end = open_h * 60 + open_m, close_h * 60 + close_m
    if start > 24 * 60 or end > 24 * 60 or open_m > 59 or close_m > 59:
        return 24 * 60 + 1  # rejected by the caller's day limit
    return end - start if end > start else end + 24 * 60 - start
