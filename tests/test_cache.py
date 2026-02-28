from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from stock_filter.datasource.cache import CacheKey, ParquetCache


def test_parquet_cache_roundtrip(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)

    df = pd.DataFrame(
        {
            "ticker": ["005930", "005930"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [100, 101],
            "high": [110, 111],
            "low": [90, 91],
            "close": [105, 106],
            "volume": [1000, 2000],
        }
    )

    key = CacheKey(ticker="005930", start="20240101", end="20240131", freq="d")
    assert cache.exists(key) is False

    cache.save(key, df)
    assert cache.exists(key) is True

    loaded = cache.load(key)
    assert_frame_equal(df, loaded, check_dtype=False)