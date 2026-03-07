from __future__ import annotations

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
