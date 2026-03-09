from __future__ import annotations

from stock_filter.app import interactive


class _Member:
    def __init__(self, ticker: str) -> None:
        self.ticker = ticker


def test_load_static_100_tickers_includes_index_etfs(monkeypatch) -> None:
    class FakeProvider:
        def get_members(self, *, universe: str, asof: str, with_names: bool = False):
            if universe == "kospi100":
                return [_Member("005930")]
            if universe == "kosdaq100":
                return [_Member("035420")]
            return []

    monkeypatch.setattr(interactive, "StaticCsvUniverseProvider", FakeProvider)
    interactive.TICKER_MARKET_MAP.clear()

    tickers = interactive._load_static_100_tickers("kospi,kosdaq")

    assert "226490" in tickers
    assert "229200" in tickers
    assert interactive.TICKER_MARKET_MAP["226490"] == "KOSPI"
    assert interactive.TICKER_MARKET_MAP["229200"] == "KOSDAQ"
