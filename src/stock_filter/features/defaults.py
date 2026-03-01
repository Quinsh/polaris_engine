from __future__ import annotations

import pandas as pd

from stock_filter.features.registry import register_feature


@register_feature("returns_1d")
def returns_1d(df: pd.DataFrame) -> pd.Series:
    if "close" not in df.columns:
        raise ValueError("returns_1d requires 'close' column")
    return df["close"].pct_change()


@register_feature("volume_sma_20")
def volume_sma_20(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        raise ValueError("volume_sma_20 requires 'volume' column")
    return df["volume"].rolling(window=20, min_periods=20).mean()


@register_feature("price_above_sma_200")
def price_above_sma_200(df: pd.DataFrame) -> pd.Series:
    if "close" not in df.columns:
        raise ValueError("price_above_sma_200 requires 'close' column")

    sma_200 = df["close"].rolling(window=200, min_periods=200).mean()
    return (df["close"] > sma_200).astype(float)
