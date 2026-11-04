"""Calendar arithmetic, one copy for the setup and the loop: a quarter's first and last day, a month's last, months
ahead."""

from __future__ import annotations

from datetime import date, timedelta


def quarter_start(day: date) -> date:
    return date(day.year, 3 * ((day.month - 1) // 3) + 1, 1)


def month_end(day: date) -> date:
    following = date(day.year + day.month // 12, day.month % 12 + 1, 1)
    return following - timedelta(days=1)


def quarter_end(day: date) -> date:
    first = quarter_start(day)
    return month_end(date(first.year, first.month + 2, 1))


def add_months(first: date, months: int) -> date:
    """The first day of the month `months` after (or, negative, before) the month holding `first`."""
    n = first.year * 12 + first.month - 1 + months
    return date(n // 12, n % 12 + 1, 1)
