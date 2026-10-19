"""Calendar helpers on top of the standard library `datetime`."""

from __future__ import annotations

import calendar
from datetime import date, timedelta


def days_between(start: date, end: date) -> int:
    """Whole days from `start` to `end` (negative if `end` is earlier)."""
    return (end - start).days


def is_weekend(d: date) -> bool:
    """True on Saturday and Sunday."""
    return d.weekday() >= 5


def add_business_days(d: date, n: int) -> date:
    """The date `n` business days (Monday to Friday) after `d`. `n` must be >= 0."""
    if n < 0:
        raise ValueError("n must be >= 0")
    current = d
    while n > 0:
        current += timedelta(days=1)
        if not is_weekend(current):
            n -= 1
    return current


def days_in_month(year: int, month: int) -> int:
    """Number of days in a month, leap years included."""
    return calendar.monthrange(year, month)[1]


def quarter(d: date) -> int:
    """Calendar quarter, 1 to 4."""
    return (d.month - 1) // 3 + 1


def iso_week_label(d: date) -> str:
    """ISO week label like "2025-W01". The ISO year can differ from d.year near New Year.

    Looks unused, but the export job imports it to name weekly report files.
    """
    iso_year, week, _ = d.isocalendar()
    return f"{iso_year}-W{week:02d}"
