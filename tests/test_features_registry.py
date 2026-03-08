from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_filter.features import compute_features


def test_feature_registry_computes_expected_columns() -> None:
    fixture = Path(__file__).parent / "fixtures" / "ohlcv_series_sample.csv"
    df = pd.read_csv(fixture)
    df["date"] = pd.to_datetime(df["date"])

    out = compute_features(df, ["returns_1d", "volume_sma_20"])

    assert "returns_1d" in out.columns
    assert "volume_sma_20" in out.columns

    # 2024-01-03 close 121 vs 2024-01-02 close 110 => +10%
    assert out.loc[2, "returns_1d"] == pytest.approx(0.1)

    # First 19 entries cannot have a 20-day SMA with min_periods=20
    assert out["volume_sma_20"].iloc[:19].isna().all()
    # Last day SMA: (19 * 100 + 300) / 20 = 110
    assert out["volume_sma_20"].iloc[-1] == 110.0


def test_price_above_sma_200_feature() -> None:
    dates = pd.date_range("2023-01-01", periods=205, freq="D")
    close = [100.0] * 204 + [120.0]
    df = pd.DataFrame(
        {
            "ticker": ["005930"] * 205,
            "date": dates,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1000] * 205,
        }
    )

    out = compute_features(df, ["price_above_sma_200"])

    assert "price_above_sma_200" in out.columns
    # first 199 rows do not have SMA(200), comparison should be False -> 0.0
    assert (out["price_above_sma_200"].iloc[:199] == 0.0).all()
    # final close=120 > SMA(200)~100.1, so feature should be 1.0
    assert out["price_above_sma_200"].iloc[-1] == 1.0


def test_green_avwap_direction_features() -> None:
    dates = pd.date_range("2023-01-01", periods=220, freq="D")

    rising_close = [100.0 + i * 0.2 for i in range(220)]
    rising_df = pd.DataFrame(
        {
            "ticker": ["005930"] * 220,
            "date": dates,
            "open": rising_close,
            "high": [c + 1.0 for c in rising_close],
            "low": [c - 1.5 for c in rising_close],
            "close": rising_close,
            "volume": [1000] * 220,
        }
    )

    above = compute_features(rising_df, ["price_above_green_avwap"])
    assert "price_above_green_avwap" in above.columns
    assert above["price_above_green_avwap"].iloc[-1] == 1.0

    # Deterministic path known to produce a final close below green AVWAP.
    rng = np.random.default_rng(7)
    falling_close = np.cumsum(rng.normal(0.0, 1.0, 220)) + 100.0
    falling_high = falling_close + rng.uniform(0.1, 2.0, 220)
    falling_low = falling_close - rng.uniform(0.1, 2.0, 220)
    falling_vol = rng.integers(1, 5000, 220)

    falling_df = pd.DataFrame(
        {
            "ticker": ["005930"] * 220,
            "date": dates,
            "open": falling_close,
            "high": falling_high,
            "low": falling_low,
            "close": falling_close,
            "volume": falling_vol,
        }
    )

    below = compute_features(falling_df, ["price_below_green_avwap"])
    assert "price_below_green_avwap" in below.columns
    assert below["price_below_green_avwap"].iloc[-1] == 1.0
