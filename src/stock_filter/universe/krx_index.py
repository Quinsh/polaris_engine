from __future__ import annotations

from dataclasses import dataclass

from stock_filter.core.types import UniverseMember
from stock_filter.datasource.pykrx_client import PykrxClient


@dataclass(frozen=True)
class KrxIndexUniverseConfig:
    # The index names we try to resolve via pykrx index metadata.
    kospi100_index_name: str = "코스피 100"
    kosdaq100_index_name: str = "코스닥 100"


class KrxIndexUniverseProvider:
    """Universe provider backed by KRX index constituents via pykrx.

    Strategy:
    - Get available index tickers for a given date
    - Find the index ticker whose name matches the target (e.g., "코스피 100")
    - Use get_index_portfolio_deposit_file(index_ticker, date) to get constituent stock tickers
    """

    def __init__(self, *, client: PykrxClient, config: KrxIndexUniverseConfig | None = None) -> None:
        self.client = client
        self.config = config or KrxIndexUniverseConfig()

    def _resolve_index_ticker(self, *, universe: str, asof: str) -> str | None:
        if universe == "kospi100":
            index_tickers = self.client.get_index_tickers(asof, market=None)
            target = self.config.kospi100_index_name
        elif universe == "kosdaq100":
            index_tickers = self.client.get_index_tickers(asof, market="KOSDAQ")
            target = self.config.kosdaq100_index_name
        else:
            raise ValueError(f"Unsupported universe: {universe}")

        target_norm = target.replace(" ", "").strip()

        # First pass: exact match after normalizing spaces.
        for idx in index_tickers:
            try:
                name = self.client.get_index_name(idx)
            except Exception:  # noqa: BLE001
                continue
            if name.replace(" ", "").strip() == target_norm:
                return idx

        # Second pass: containment match (more permissive).
        for idx in index_tickers:
            try:
                name = self.client.get_index_name(idx)
            except Exception:  # noqa: BLE001
                continue
            if target_norm in name.replace(" ", "").strip():
                return idx

        return None

    def get_members(self, *, universe: str, asof: str, with_names: bool = False) -> list[UniverseMember]:
        idx = self._resolve_index_ticker(universe=universe, asof=asof)
        if idx is None:
            return []

        tickers = self.client.get_index_constituents(idx, date=asof)
        members: list[UniverseMember] = []

        if not with_names:
            return [UniverseMember(ticker=t) for t in tickers]

        # Name lookup is optional and can be slow; best-effort only.
        for t in tickers:
            try:
                name = self.client.get_ticker_name(t)
            except Exception:  # noqa: BLE001
                name = None
            members.append(UniverseMember(ticker=t, name=name))
        return members