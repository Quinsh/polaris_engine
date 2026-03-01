from __future__ import annotations

from pathlib import Path

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
