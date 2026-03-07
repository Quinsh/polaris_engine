from __future__ import annotations

import pandas as pd

from stock_filter.signals import detect_signals


def _build_df(close_values: list[float], volumes: list[float]) -> pd.DataFrame:
    n = len(close_values)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "ticker": ["005930"] * n,
            "date": dates,
            "open": close_values,
            "high": close_values,
            "low": close_values,
            "close": close_values,
            "volume": volumes,
        }
    )


def test_rsi_mfi_smart_buy_detected() -> None:
    close_values = [
        98, 95, 92, 93, 91, 90, 88, 85, 84, 82, 81, 78, 76, 74, 73, 72, 70, 71, 68, 65,
        66, 63, 60, 57, 55, 54, 52, 50, 47, 45, 42, 40, 37, 35, 36, 33, 31, 28, 26, 25,
        23, 20, 18, 17, 14, 11, 12, 10, 7, 5, 2, 3, 0, 1, -2, -5, -6, -9, -10, -12,
    ]
    volumes = [
        833, 856, 816, 5700, 462, 205, 136, 212, 244, 258, 215, 718, 312, 903, 525, 437, 870, 1878, 193, 264,
        4326, 402, 948, 895, 428, 640, 742, 839, 497, 525, 989, 392, 840, 268, 708, 325, 419, 599, 658, 460,
        317, 212, 611, 457, 614, 198, 5376, 818, 815, 542, 782, 5364, 332, 4020, 995, 500, 649, 337, 588, 334,
    ]
    df = _build_df(close_values, volumes)

    signals = detect_signals(df, ["rsi_mfi_smart_buy", "rsi_mfi_smart_sell"])
    assert len(signals) == 1
    assert signals[0].name == "rsi_mfi_smart_buy"
    assert signals[0].score > 5.0


def test_rsi_mfi_smart_sell_detected() -> None:
    close_values = [
        101, 103, 105, 104, 107, 108, 109, 111, 113, 114, 116, 119, 121, 124, 127, 129, 128, 131, 132, 134,
        135, 138, 140, 143, 146, 149, 151, 152, 151, 154, 155, 158, 160, 163, 165, 168, 170, 172, 174, 176,
        177, 180, 182, 184, 185, 184, 187, 188, 190, 193, 194, 196, 198, 200, 202, 204, 203, 202, 204, 205,
    ]
    volumes = [
        751, 751, 477, 3216, 944, 680, 273, 292, 189, 652, 875, 875, 346, 389, 670, 363, 1572, 232, 489, 218,
        971, 278, 688, 529, 354, 452, 214, 481, 1152, 427, 832, 851, 765, 589, 347, 965, 101, 369, 216, 138,
        313, 511, 778, 261, 869, 3090, 345, 213, 726, 394, 128, 815, 726, 967, 131, 279, 996, 5610, 942, 873,
    ]
    df = _build_df(close_values, volumes)

    signals = detect_signals(df, ["rsi_mfi_smart_buy", "rsi_mfi_smart_sell"])
    assert len(signals) == 1
    assert signals[0].name == "rsi_mfi_smart_sell"
    assert signals[0].score > 5.0
