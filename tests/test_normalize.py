from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_filter.datasource.normalize import normalize_ohlcv


def test_normalize_ohlcv_from_fixture_csv() -> None:
    fixture = Path(__file__).parent / "fixtures" / "pykrx_ohlcv_raw.csv"
    raw = pd.read_csv(fixture, comment="#")

    # Simulate the common pykrx shape: date in index named '날짜'
    raw["날짜"] = pd.to_datetime(raw["날짜"])
    raw = raw.set_index("날짜")

    out = normalize_ohlcv(raw, ticker="000000")

    assert list(out.columns[:2]) == ["ticker", "date"]
    assert set(["open", "high", "low", "close", "volume"]).issubset(out.columns)
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert (out["ticker"] == "000000").all()
    assert out.shape[0] == 3