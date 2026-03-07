from __future__ import annotations

import math

import pandas as pd

from stock_filter.analytics.types import Signal
from stock_filter.signals.registry import register_signal


def _rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _mfi_from_source(source: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    prev = source.shift(1)
    money_flow = source * volume
    positive_flow = money_flow.where(source > prev, 0.0)
    negative_flow = money_flow.where(source < prev, 0.0)

    pos_sum = positive_flow.rolling(length, min_periods=length).sum()
    neg_sum = negative_flow.rolling(length, min_periods=length).sum()
    money_ratio = pos_sum / neg_sum
    return 100 - (100 / (1 + money_ratio))


def _highestbars_offset(values: pd.Series, lookback: int) -> pd.Series:
    offsets: list[float] = []
    for i in range(len(values)):
        start = max(0, i - lookback + 1)
        window = values.iloc[start : i + 1]
        if window.isna().all():
            offsets.append(float("nan"))
            continue
        vmax = window.max(skipna=True)
        local_positions = [j for j, v in enumerate(window.tolist()) if pd.notna(v) and v == vmax]
        if not local_positions:
            offsets.append(float("nan"))
            continue
        # TradingView-style: nearest (most recent) bar for ties.
        rel_idx = local_positions[-1]
        offset = len(window) - 1 - rel_idx
        offsets.append(float(offset))
    return pd.Series(offsets, index=values.index)


def _lowestbars_offset(values: pd.Series, lookback: int) -> pd.Series:
    offsets: list[float] = []
    for i in range(len(values)):
        start = max(0, i - lookback + 1)
        window = values.iloc[start : i + 1]
        if window.isna().all():
            offsets.append(float("nan"))
            continue
        vmin = window.min(skipna=True)
        local_positions = [j for j, v in enumerate(window.tolist()) if pd.notna(v) and v == vmin]
        if not local_positions:
            offsets.append(float("nan"))
            continue
        rel_idx = local_positions[-1]
        offset = len(window) - 1 - rel_idx
        offsets.append(float(offset))
    return pd.Series(offsets, index=values.index)


def _stoch_k(close: pd.Series, high: pd.Series, low: pd.Series, period_k: int, smooth_k: int) -> pd.Series:
    ll = low.rolling(period_k, min_periods=period_k).min()
    hh = high.rolling(period_k, min_periods=period_k).max()
    raw_k = ((close - ll) / (hh - ll)) * 100.0
    return raw_k.rolling(smooth_k, min_periods=smooth_k).mean()


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


def _rsi_mfi_signal(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    required = ["close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"rsi_mfi missing required columns: {missing}")

    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    length = 14
    smooth_len = 9

    rsi_val = _rsi(close, length)
    mfi_val = _mfi_from_source(close, volume, length)
    rvs = mfi_val - rsi_val
    rvs_sig = rvs.ewm(span=smooth_len, adjust=False, min_periods=smooth_len).mean()
    return rsi_val, rvs, rvs_sig


@register_signal("rsi_mfi_smart_buy", required_features=[])
def rsi_mfi_smart_buy(df: pd.DataFrame) -> Signal | None:
    required = ["ticker", "date", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"rsi_mfi_smart_buy missing required columns: {missing}")

    rsi_val, rvs, rvs_sig = _rsi_mfi_signal(df)
    row = df.iloc[-1]
    rsi_last = rsi_val.iloc[-1]
    rvs_last = rvs.iloc[-1]
    rvs_sig_last = rvs_sig.iloc[-1]

    is_accumulation = bool(pd.notna(rsi_last) and pd.notna(rvs_last) and rsi_last < 30 and rvs_last > 5)
    if not is_accumulation:
        return None

    asof = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
    ticker = str(row["ticker"])
    return Signal(
        name="rsi_mfi_smart_buy",
        ticker=ticker,
        asof=asof,
        score=float(rvs_last),
        details={"rsi": float(rsi_last), "rvs": float(rvs_last), "rvs_signal": float(rvs_sig_last)},
    )


@register_signal("rsi_mfi_smart_sell", required_features=[])
def rsi_mfi_smart_sell(df: pd.DataFrame) -> Signal | None:
    required = ["ticker", "date", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"rsi_mfi_smart_sell missing required columns: {missing}")

    rsi_val, rvs, rvs_sig = _rsi_mfi_signal(df)
    row = df.iloc[-1]
    rsi_last = rsi_val.iloc[-1]
    rvs_last = rvs.iloc[-1]
    rvs_sig_last = rvs_sig.iloc[-1]

    is_distribution = bool(pd.notna(rsi_last) and pd.notna(rvs_last) and rsi_last > 70 and rvs_last < -5)
    if not is_distribution:
        return None

    asof = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
    ticker = str(row["ticker"])
    return Signal(
        name="rsi_mfi_smart_sell",
        ticker=ticker,
        asof=asof,
        score=float(abs(rvs_last)),
        details={"rsi": float(rsi_last), "rvs": float(rvs_last), "rvs_signal": float(rvs_sig_last)},
    )


def _mfi_divergence(df: pd.DataFrame) -> tuple[str, dict[str, float]] | None:
    required = ["close", "high", "low", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"mfi_divergence missing required columns: {missing}")

    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")

    mfi_len = 13
    stoch_len = 14
    stoch_smooth = 3
    ob = 75.0
    os = 25.0
    xbars = 10

    src = (high + low + close) / 3.0
    prev_src = src.shift(1)
    upper = (volume * src.where((src - prev_src) > 0, 0.0)).rolling(mfi_len, min_periods=mfi_len).sum()
    lower = (volume * src.where((src - prev_src) < 0, 0.0)).rolling(mfi_len, min_periods=mfi_len).sum()
    mfi = 100 - (100 / (1 + (upper / lower)))

    k = _stoch_k(close, high, low, stoch_len, stoch_smooth)
    hb = _highestbars_offset(mfi, xbars)
    lb = _lowestbars_offset(mfi, xbars)

    n = len(df)
    if n < 5:
        return None

    max_price = [math.nan] * n
    max_mfi = [math.nan] * n
    min_price = [math.nan] * n
    min_mfi = [math.nan] * n
    divbear = [False] * n
    divbull = [False] * n

    for i in range(n):
        c = float(close.iloc[i]) if pd.notna(close.iloc[i]) else math.nan
        r = float(mfi.iloc[i]) if pd.notna(mfi.iloc[i]) else math.nan

        prev_max_price = max_price[i - 1] if i > 0 else math.nan
        prev_max_mfi = max_mfi[i - 1] if i > 0 else math.nan
        prev_min_price = min_price[i - 1] if i > 0 else math.nan
        prev_min_mfi = min_mfi[i - 1] if i > 0 else math.nan

        if pd.notna(hb.iloc[i]) and float(hb.iloc[i]) == 0.0:
            max_price[i] = c
            max_mfi[i] = r
        else:
            max_price[i] = c if pd.isna(prev_max_price) else prev_max_price
            max_mfi[i] = r if pd.isna(prev_max_mfi) else prev_max_mfi

        if pd.notna(lb.iloc[i]) and float(lb.iloc[i]) == 0.0:
            min_price[i] = c
            min_mfi[i] = r
        else:
            min_price[i] = c if pd.isna(prev_min_price) else prev_min_price
            min_mfi[i] = r if pd.isna(prev_min_mfi) else prev_min_mfi

        if pd.notna(c):
            if pd.notna(max_price[i]) and c > max_price[i]:
                max_price[i] = c
            if pd.notna(min_price[i]) and c < min_price[i]:
                min_price[i] = c
        if pd.notna(r):
            if pd.notna(max_mfi[i]) and r > max_mfi[i]:
                max_mfi[i] = r
            if pd.notna(min_mfi[i]) and r < min_mfi[i]:
                min_mfi[i] = r

        if i >= 2:
            cond_bear = (
                pd.notna(max_price[i - 1])
                and pd.notna(max_price[i - 2])
                and max_price[i - 1] > max_price[i - 2]
                and pd.notna(mfi.iloc[i - 1])
                and pd.notna(max_mfi[i])
                and mfi.iloc[i - 1] < max_mfi[i]
                and pd.notna(mfi.iloc[i])
                and mfi.iloc[i] <= mfi.iloc[i - 1]
            )
            cond_bull = (
                pd.notna(min_price[i - 1])
                and pd.notna(min_price[i - 2])
                and min_price[i - 1] < min_price[i - 2]
                and pd.notna(mfi.iloc[i - 1])
                and pd.notna(min_mfi[i])
                and mfi.iloc[i - 1] > min_mfi[i]
                and pd.notna(mfi.iloc[i])
                and mfi.iloc[i] >= mfi.iloc[i - 1]
            )
            divbear[i] = bool(cond_bear)
            divbull[i] = bool(cond_bull)

    i = n - 1
    if i < 1 or pd.isna(k.iloc[i]) or pd.isna(k.iloc[i - 1]):
        return None

    is_ob = bool(k.iloc[i] > ob or k.iloc[i - 1] > ob)
    is_os = bool(k.iloc[i] < os or k.iloc[i - 1] < os)

    if divbear[i] and is_ob:
        details = {
            "direction_sign": -1.0,
            "mfi": float(mfi.iloc[i]),
            "stoch_k": float(k.iloc[i]),
        }
        return "bearish", details

    if divbull[i] and is_os:
        details = {
            "direction_sign": 1.0,
            "mfi": float(mfi.iloc[i]),
            "stoch_k": float(k.iloc[i]),
        }
        return "bullish", details

    return None


@register_signal("mfi_divergence", required_features=[])
def mfi_divergence(df: pd.DataFrame) -> Signal | None:
    required = ["ticker", "date", "close", "high", "low", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"mfi_divergence missing required columns: {missing}")

    out = _mfi_divergence(df)
    if out is None:
        return None

    direction, details = out
    score = 1.0
    row = df.iloc[-1]
    asof = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
    ticker = str(row["ticker"])
    return Signal(
        name="mfi_divergence",
        ticker=ticker,
        asof=asof,
        score=float(score),
        details={"direction": direction, **details},
    )


@register_signal("mfi_bullish_divergence", required_features=[])
def mfi_bullish_divergence(df: pd.DataFrame) -> Signal | None:
    signal = mfi_divergence(df)
    if signal is None:
        return None
    if signal.details.get("direction") != "bullish":
        return None
    return Signal(
        name="mfi_bullish_divergence",
        ticker=signal.ticker,
        asof=signal.asof,
        score=1.0,
        details=signal.details,
    )


@register_signal("mfi_bearish_divergence", required_features=[])
def mfi_bearish_divergence(df: pd.DataFrame) -> Signal | None:
    signal = mfi_divergence(df)
    if signal is None:
        return None
    if signal.details.get("direction") != "bearish":
        return None
    return Signal(
        name="mfi_bearish_divergence",
        ticker=signal.ticker,
        asof=signal.asof,
        score=1.0,
        details=signal.details,
    )
