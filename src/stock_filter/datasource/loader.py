from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from stock_filter.datasource.cache import CacheKey, ParquetCache
from stock_filter.datasource.normalize import normalize_ohlcv
from stock_filter.datasource.pykrx_client import PykrxClient


@dataclass
class FetchStats:
    total: int
    fetched: int = 0
    from_cache: int = 0
    failed: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)  # (ticker, error)


class DataLoader:
    def __init__(
        self,
        *,
        client: PykrxClient,
        cache: ParquetCache,
        sleep_seconds: float = 0.25,
        max_retries: int = 3,
        backoff_base_seconds: float = 1.0,
    ) -> None:
        self.client = client
        self.cache = cache
        self.sleep_seconds = sleep_seconds
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds

    def fetch_one(self, *, ticker: str, start: str, end: str, freq: str) -> tuple[pd.DataFrame, bool]:
        key = CacheKey(ticker=ticker, start=start, end=end, freq=freq)
        if self.cache.exists(key):
            return self.cache.load(key), True

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.client.get_ohlcv(ticker=ticker, start=start, end=end, freq=freq)
                df = normalize_ohlcv(raw, ticker=ticker)
                self.cache.save(key, df)
                # Polite delay to reduce chances of temporary blocking.
                time.sleep(self.sleep_seconds)
                return df, False
            except Exception as e:  # noqa: BLE001
                last_err = e
                # Exponential backoff with jitter
                sleep_for = (self.backoff_base_seconds * (2 ** (attempt - 1))) + random.random() * 0.25
                time.sleep(sleep_for)

        assert last_err is not None
        raise last_err

    def fetch_many(self, *, tickers: Iterable[str], start: str, end: str, freq: str) -> FetchStats:
        tickers_list = list(tickers)
        stats = FetchStats(total=len(tickers_list))

        for t in tickers_list:
            try:
                _df, cached = self.fetch_one(ticker=t, start=start, end=end, freq=freq)
                if cached:
                    stats.from_cache += 1
                else:
                    stats.fetched += 1
            except Exception as e:  # noqa: BLE001
                stats.failed += 1
                stats.failures.append((t, repr(e)))

        return stats