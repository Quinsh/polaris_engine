from __future__ import annotations

from typing import Protocol

from stock_filter.core.types import UniverseMember


class UniverseProvider(Protocol):
    def get_members(self, *, universe: str, asof: str, with_names: bool = False) -> list[UniverseMember]:
        ...