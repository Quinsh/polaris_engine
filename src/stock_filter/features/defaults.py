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



def _wma(series: pd.Series, length: int) -> pd.Series:
    weights = pd.Series(range(1, length + 1), dtype=float)
    return series.rolling(length, min_periods=length).apply(
        lambda x: float((pd.Series(x).reset_index(drop=True) * weights).sum() / weights.sum()),
        raw=False,
    )


def _green_avwap(df: pd.DataFrame) -> pd.Series:
    required = ["high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"green_avwap requires columns: {missing}")

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")

    # Pine defaults from the provided script.
    length_rsi = 64
    length_stoch = 48
    smooth_k = 4
    lower_band = 20.0
    upper_band = 80.0
    lower_reversal = 20.0
    upper_reversal = 80.0

    src = (high + low + close) / 3.0
    rsi1 = src.diff()
    gain = rsi1.clip(lower=0.0)
    loss = -rsi1.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length_rsi, min_periods=length_rsi, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length_rsi, min_periods=length_rsi, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    lowest_rsi = rsi.rolling(length_stoch, min_periods=length_stoch).min()
    highest_rsi = rsi.rolling(length_stoch, min_periods=length_stoch).max()
    stoch = ((rsi - lowest_rsi) / (highest_rsi - lowest_rsi)) * 100.0
    k = _wma(stoch, smooth_k)
    d = k.rolling(smooth_k, min_periods=smooth_k).mean()

    n = len(df)
    lo_avwap: list[float] = [float("nan")] * n

    hi = float(high.iloc[0]) if n else float("nan")
    lo = float(low.iloc[0]) if n else float("nan")
    phi = hi
    plo = lo
    state = 0

    hi_s = lo_s = hi_v = lo_v = 0.0
    hi_s_next = lo_s_next = hi_v_next = lo_v_next = 0.0

    for i in range(n):
        h = float(high.iloc[i]) if pd.notna(high.iloc[i]) else float("nan")
        l = float(low.iloc[i]) if pd.notna(low.iloc[i]) else float("nan")
        v = float(volume.iloc[i]) if pd.notna(volume.iloc[i]) else 0.0
        kd = float(k.iloc[i]) if pd.notna(k.iloc[i]) else float("nan")
        dd = float(d.iloc[i]) if pd.notna(d.iloc[i]) else float("nan")

        if (pd.notna(dd) and dd < lower_band) or (pd.notna(h) and h > phi):
            phi = h
            hi_s_next = 0.0
            hi_v_next = 0.0
        if (pd.notna(dd) and dd > upper_band) or (pd.notna(l) and l < plo):
            plo = l
            lo_s_next = 0.0
            lo_v_next = 0.0

        if pd.notna(h) and h > hi:
            hi = h
            hi_s = 0.0
            hi_v = 0.0
        if pd.notna(l) and l < lo:
            lo = l
            lo_s = 0.0
            lo_v = 0.0

        vwap_hi = h
        vwap_lo = l

        hi_s += vwap_hi * v
        lo_s += vwap_lo * v
        hi_v += v
        lo_v += v
        hi_s_next += vwap_hi * v
        lo_s_next += vwap_lo * v
        hi_v_next += v
        lo_v_next += v

        if pd.notna(dd):
            if state != -1 and dd < lower_band:
                state = -1
            elif state != +1 and dd > upper_band:
                state = +1

        if pd.notna(kd) and pd.notna(dd):
            if hi > phi and state == +1 and kd < dd and kd < lower_reversal:
                hi = phi
                hi_s = hi_s_next
                hi_v = hi_v_next
            if lo < plo and state == -1 and kd > dd and kd > upper_reversal:
                lo = plo
                lo_s = lo_s_next
                lo_v = lo_v_next

        lo_avwap[i] = lo_s / lo_v if lo_v > 0 else float("nan")

    return pd.Series(lo_avwap, index=df.index, dtype=float)


@register_feature("price_above_green_avwap")
def price_above_green_avwap(df: pd.DataFrame) -> pd.Series:
    if "close" not in df.columns:
        raise ValueError("price_above_green_avwap requires 'close' column")

    green_avwap = _green_avwap(df)
    close = pd.to_numeric(df["close"], errors="coerce")
    return (close > green_avwap).astype(float)


@register_feature("price_below_green_avwap")
def price_below_green_avwap(df: pd.DataFrame) -> pd.Series:
    if "close" not in df.columns:
        raise ValueError("price_below_green_avwap requires 'close' column")

    green_avwap = _green_avwap(df)
    close = pd.to_numeric(df["close"], errors="coerce")
    return (close < green_avwap).astype(float)
