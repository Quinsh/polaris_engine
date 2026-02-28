# src/stock_filter/universe/service.py
from __future__ import annotations

from stock_filter.core.types import UniverseMember
from stock_filter.universe.krx_index import KrxIndexUniverseProvider
from stock_filter.universe.static_csv import StaticCsvUniverseProvider


class UniverseService:
    """Resolves universe members via KRX index constituents, with a CSV fallback."""

    def __init__(
        self,
        *,
        primary: KrxIndexUniverseProvider,
        fallback: StaticCsvUniverseProvider,
    ) -> None:
        self.primary = primary
        self.fallback = fallback

    def get_members(self, *, universe: str, asof: str, with_names: bool = False) -> list[UniverseMember]:
        try:
            members = self.primary.get_members(universe=universe, asof=asof, with_names=with_names)
            if members:
                return members
        except Exception:
            # Primary source (KRX index via pykrx) can break due to upstream changes or throttling.
            pass

        return self.fallback.get_members(universe=universe, asof=asof, with_names=with_names)