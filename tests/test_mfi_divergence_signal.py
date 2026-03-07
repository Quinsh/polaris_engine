from __future__ import annotations

import numpy as np
import pandas as pd

from stock_filter.signals import detect_signals


def _make_df(close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray) -> pd.DataFrame:
    n = len(close)
    return pd.DataFrame(
        {
            "ticker": ["005930"] * n,
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _find_divergence_case(direction: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    for _ in range(4000):
        n = 70
        if direction == "bearish":
            close = np.cumsum(rng.normal(0.2, 1.0, n)) + 100
            close[-4:] = [close[-5] + 0.8, close[-5] + 1.4, close[-5] + 2.0, close[-5] + 2.2]
        else:
            close = np.cumsum(rng.normal(-0.2, 1.0, n)) + 140
            close[-4:] = [close[-5] - 0.8, close[-5] - 1.4, close[-5] - 2.0, close[-5] - 2.2]

        high = close + rng.uniform(0.1, 1.1, n)
        low = close - rng.uniform(0.1, 1.1, n)
        volume = rng.integers(100, 1000, n).astype(float)
        volume[-6:] = [120, 110, 100, 95, 90, 85]

        df = _make_df(close, high, low, volume)
        signals = detect_signals(df, ["mfi_divergence"])
        if signals and signals[0].details.get("direction") == direction:
            return df

    raise AssertionError(f"failed to find {direction} divergence case")


def test_mfi_divergence_detects_bearish_and_bullish() -> None:
    bear_df = _find_divergence_case("bearish", seed=7)
    bull_df = _find_divergence_case("bullish", seed=11)

    bear_signal = detect_signals(bear_df, ["mfi_divergence"])[0]
    bull_signal = detect_signals(bull_df, ["mfi_divergence"])[0]

    assert bear_signal.details["direction"] == "bearish"
    assert bull_signal.details["direction"] == "bullish"
    assert bear_signal.score == 1.0
    assert bull_signal.score == 1.0


def test_mfi_divergence_split_signals_match_direction() -> None:
    bear_df = _find_divergence_case("bearish", seed=7)
    bull_df = _find_divergence_case("bullish", seed=11)

    bear_only = detect_signals(bear_df, ["mfi_bearish_divergence"])
    bull_only = detect_signals(bull_df, ["mfi_bullish_divergence"])
    wrong_bear = detect_signals(bear_df, ["mfi_bullish_divergence"])
    wrong_bull = detect_signals(bull_df, ["mfi_bearish_divergence"])

    assert bear_only and bear_only[0].name == "mfi_bearish_divergence"
    assert bull_only and bull_only[0].name == "mfi_bullish_divergence"
    assert not wrong_bear
    assert not wrong_bull


def test_mfi_divergence_score_is_binary_when_signal_exists() -> None:
    df = _find_divergence_case("bearish", seed=7)
    base_signal = detect_signals(df, ["mfi_divergence"])[0]

    stronger = df.copy()
    stronger.loc[stronger.index[-2], "close"] += 2.0
    stronger.loc[stronger.index[-1], "close"] += 3.0
    stronger.loc[stronger.index[-2:], "high"] = stronger.loc[stronger.index[-2:], "close"] + 1.0
    stronger.loc[stronger.index[-2:], "low"] = stronger.loc[stronger.index[-2:], "close"] - 0.3
    stronger.loc[stronger.index[-4:], "volume"] = [80.0, 70.0, 60.0, 50.0]

    stronger_signal = detect_signals(stronger, ["mfi_divergence"])[0]

    assert stronger_signal.details["direction"] == "bearish"
    assert stronger_signal.score == base_signal.score == 1.0
