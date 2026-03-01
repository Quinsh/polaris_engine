from __future__ import annotations

import pandas as pd

from stock_filter.analytics.types import Signal
from stock_filter.signals.registry import register_signal


@register_signal("unusual_volume_simple", required_features=["volume_sma_20"])
def unusual_volume_simple(df: pd.DataFrame) -> Signal | None:
    required = ["ticker", "date", "volume", "volume_sma_20"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"unusual_volume_simple missing required columns: {missing}")

    row = df.iloc[-1]
    sma = row["volume_sma_20"]
    volume = row["volume"]
    if pd.isna(sma) or pd.isna(volume) or float(sma) <= 0:
        return None

    score = float(volume) / float(sma)
    if score <= 2.0:
        return None

    asof = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
    ticker = str(row["ticker"])
    return Signal(
        name="unusual_volume_simple",
        ticker=ticker,
        asof=asof,
        score=score,
        details={"volume": float(volume), "volume_sma_20": float(sma)},
    )
