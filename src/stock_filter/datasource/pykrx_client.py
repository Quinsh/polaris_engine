from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

try:
    from pykrx import stock as _stock  # type: ignore
except Exception:  # pragma: no cover
    _stock = None


@dataclass(frozen=True)
class PykrxClientConfig:
    pass


class PykrxClient:
    """Thin wrapper around pykrx functions used by this prototype."""

    def __init__(self, config: PykrxClientConfig | None = None) -> None:
        if _stock is None:
            raise RuntimeError(
                "pykrx is not installed or failed to import. Install with: pip install pykrx"
            )
        self._stock = _stock
        self.config = config or PykrxClientConfig()

    def get_ohlcv(self, *, ticker: str, start: str, end: str, freq: str = "d") -> pd.DataFrame:
        if freq == "d":
            return self._stock.get_market_ohlcv(start, end, ticker)
        return self._stock.get_market_ohlcv(start, end, ticker, freq)

    def get_ticker_name(self, ticker: str) -> str:
        return self._stock.get_market_ticker_name(ticker)

    def get_market_tickers(self, *, date: str, market: str = "KOSPI") -> list[str]:
        # Documented usage: stock.get_market_ticker_list("20190225", market="KOSDAQ")
        return self._stock.get_market_ticker_list(date, market=market)

    def get_index_tickers(self, date: str, market: str | None = None) -> list[str]:
        if market is None:
            return self._stock.get_index_ticker_list(date)
        return self._stock.get_index_ticker_list(date, market)

    def get_index_name(self, index_ticker: str) -> str:
        return self._stock.get_index_ticker_name(index_ticker)

    def get_index_constituents(self, index_ticker: str, date: str | None = None) -> list[str]:
        if date is None:
            return self._stock.get_index_portfolio_deposit_file(index_ticker)
        return self._stock.get_index_portfolio_deposit_file(index_ticker, date)