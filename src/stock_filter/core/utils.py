from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Iterable


_YYYYMMDD_RE = re.compile(r"^\d{8}$")


def validate_yyyymmdd(value: str) -> str:
    """Validate YYYYMMDD format and that it is a real calendar date."""
    if not _YYYYMMDD_RE.match(value):
        raise ValueError(f"Expected YYYYMMDD, got: {value!r}")
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as e:
        raise ValueError(f"Invalid date: {value!r}") from e
    return value


def yyyymmdd_to_date(value: str) -> date:
    validate_yyyymmdd(value)
    return datetime.strptime(value, "%Y%m%d").date()


def date_to_yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def today_yyyymmdd() -> str:
    return date_to_yyyymmdd(date.today())


def years_ago_yyyymmdd(years: int, *, from_yyyymmdd: str | None = None) -> str:
    """Calendar-year subtraction (handles Feb 29)."""
    base = date.today() if from_yyyymmdd is None else yyyymmdd_to_date(from_yyyymmdd)
    try:
        shifted = base.replace(year=base.year - years)
    except ValueError:
        # e.g. Feb 29 -> Feb 28
        shifted = base.replace(year=base.year - years, day=28)
    return date_to_yyyymmdd(shifted)


def normalize_universe_name(value: str) -> str:
    v = value.strip().lower()
    aliases = {
        "kospi100": "kospi100",
        "kospi_100": "kospi100",
        "kospi-100": "kospi100",
        "kosdaq100": "kosdaq100",
        "kosdaq_100": "kosdaq100",
        "kosdaq-100": "kosdaq100",
    }
    if v not in aliases:
        raise ValueError(f"Unsupported universe: {value!r} (expected: kospi100 or kosdaq100)")
    return aliases[v]


def chunked(items: Iterable[str], n: int) -> list[list[str]]:
    buf: list[str] = []
    out: list[list[str]] = []
    for x in items:
        buf.append(x)
        if len(buf) >= n:
            out.append(buf)
            buf = []
    if buf:
        out.append(buf)
    return out