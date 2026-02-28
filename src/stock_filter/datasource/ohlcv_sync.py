from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from stock_filter.core.utils import date_to_yyyymmdd, today_yyyymmdd, yyyymmdd_to_date, years_ago_yyyymmdd
from stock_filter.datasource.normalize import normalize_ohlcv
from stock_filter.datasource.ohlcv_series_store import OhlcvSeriesStore
from stock_filter.datasource.pykrx_client import PykrxClient


@dataclass
class SyncStats:
    total: int
    written: int = 0
    updated: int = 0
    skipped: int = 0
    bootstrapped: int = 0
    failed: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)  # (ticker, error)


class OhlcvSync:
    def __init__(
        self,
        *,
        client: PykrxClient,
        store: OhlcvSeriesStore,
        sleep_seconds: float = 0.25,
        max_retries: int = 3,
        backoff_base_seconds: float = 1.0,
    ) -> None:
        self.client = client
        self.store = store
        self.sleep_seconds = sleep_seconds
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds

    def _fetch_ohlcv_with_retry(self, *, ticker: str, start: str, end: str, freq: str) -> pd.DataFrame:
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.client.get_ohlcv(ticker=ticker, start=start, end=end, freq=freq)
                df = normalize_ohlcv(raw, ticker=ticker)
                time.sleep(self.sleep_seconds)
                return df
            except Exception as e:  # noqa: BLE001
                last_err = e
                sleep_for = (self.backoff_base_seconds * (2 ** (attempt - 1))) + random.random() * 0.25
                time.sleep(sleep_for)
        assert last_err is not None
        raise last_err

    def overwrite_last_years(
        self,
        *,
        tickers: list[str],
        years: int = 3,
        freq: str = "d",
        asof: str | None = None,
        limit: int | None = None,
    ) -> SyncStats:
        end = today_yyyymmdd() if asof is None else asof
        start = years_ago_yyyymmdd(years, from_yyyymmdd=end)

        tickers_u = sorted(set(tickers))
        if limit is not None:
            tickers_u = tickers_u[: max(0, int(limit))]

        stats = SyncStats(total=len(tickers_u))

        for t in tickers_u:
            try:
                df = self._fetch_ohlcv_with_retry(ticker=t, start=start, end=end, freq=freq)
                if df.empty:
                    raise RuntimeError("OHLCV returned empty dataframe")
                self.store.save(ticker=t, freq=freq, df=df)
                stats.written += 1
            except Exception as e:  # noqa: BLE001
                stats.failed += 1
                stats.failures.append((t, repr(e)))

        return stats

    def update_to_today(
        self,
        *,
        tickers: list[str],
        freq: str = "d",
        asof: str | None = None,
        bootstrap_years: int = 3,
        limit: int | None = None,
    ) -> SyncStats:
        end = today_yyyymmdd() if asof is None else asof
        end_date = yyyymmdd_to_date(end)

        tickers_u = sorted(set(tickers))
        if limit is not None:
            tickers_u = tickers_u[: max(0, int(limit))]

        stats = SyncStats(total=len(tickers_u))

        for t in tickers_u:
            try:
                existing = self.store.exists(ticker=t, freq=freq)
                if existing:
                    last = self.store.last_date(ticker=t, freq=freq)
                    if last is None:
                        start = years_ago_yyyymmdd(bootstrap_years, from_yyyymmdd=end)
                        stats.bootstrapped += 1
                        old_df = None
                    else:
                        start_date = (pd.Timestamp(last).date() + timedelta(days=1))
                        if start_date > end_date:
                            stats.skipped += 1
                            continue
                        start = date_to_yyyymmdd(start_date)
                        old_df = self.store.load(ticker=t, freq=freq)
                else:
                    start = years_ago_yyyymmdd(bootstrap_years, from_yyyymmdd=end)
                    stats.bootstrapped += 1
                    old_df = None

                new_df = self._fetch_ohlcv_with_retry(ticker=t, start=start, end=end, freq=freq)
                if new_df.empty:
                    stats.skipped += 1
                    continue

                if old_df is None or old_df.empty:
                    self.store.save(ticker=t, freq=freq, df=new_df)
                    stats.updated += 1
                    continue

                combined = pd.concat([old_df, new_df], ignore_index=True)

                if "ticker" in combined.columns:
                    combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
                else:
                    combined = combined.drop_duplicates(subset=["date"], keep="last")

                combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
                combined = combined.sort_values("date").reset_index(drop=True)

                self.store.save(ticker=t, freq=freq, df=combined)
                stats.updated += 1

            except Exception as e:  # noqa: BLE001
                stats.failed += 1
                stats.failures.append((t, repr(e)))

        return stats