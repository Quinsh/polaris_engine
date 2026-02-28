from __future__ import annotations

import csv
from dataclasses import dataclass
from importlib import resources
from typing import Iterable, TextIO

from stock_filter.core.types import UniverseMember


@dataclass(frozen=True)
class StaticCsvUniverseConfig:
    # Package resource folder: stock_filter/universe/static/*.csv
    package: str = "stock_filter.universe.static"


class StaticCsvUniverseProvider:
    """Universe provider that reads a packaged CSV file.

    This is a pragmatic fallback for when index constituent lookup fails.
    The packaged CSVs are intentionally SMALL SAMPLES (not complete universes).

    CSV format:
      - optional leading comment lines starting with '#'
      - header row: ticker,name
      - subsequent rows: 6-digit ticker, optional name
    """

    def __init__(self, *, config: StaticCsvUniverseConfig | None = None) -> None:
        self.config = config or StaticCsvUniverseConfig()

    def _open_csv(self, universe: str) -> TextIO:
        filename = f"{universe}.csv"
        return resources.files(self.config.package).joinpath(filename).open("r", encoding="utf-8")

    @staticmethod
    def _non_comment_lines(f: Iterable[str]) -> list[str]:
        lines: list[str] = []
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            lines.append(line)
        return lines

    def get_members(self, *, universe: str, asof: str, with_names: bool = False) -> list[UniverseMember]:
        # 'asof' is ignored for static universes (this is only a fallback).
        members: list[UniverseMember] = []
        with self._open_csv(universe) as f:
            lines = self._non_comment_lines(f)
            reader = csv.DictReader(lines)
            for row in reader:
                ticker = (row.get("ticker") or "").strip()
                name = (row.get("name") or "").strip() or None
                if not ticker:
                    continue
                members.append(UniverseMember(ticker=ticker, name=name if with_names else None))
        return members