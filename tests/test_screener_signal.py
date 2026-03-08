from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_filter.features import compute_features
from stock_filter.screening import Screener
from stock_filter.signals import detect_signals


def test_screener_pass_and_fail_with_unusual_volume_signal() -> None:
    fixture = Path(__file__).parent / "fixtures" / "ohlcv_series_sample.csv"
    df = pd.read_csv(fixture)
    df["date"] = pd.to_datetime(df["date"])

    enriched = compute_features(df, ["volume_sma_20"])
    signals = detect_signals(enriched, ["unusual_volume_simple"])

    assert len(signals) == 1
    assert signals[0].name == "unusual_volume_simple"
    assert signals[0].score > 2.0

    pass_rules = {
        "signal_rules": [{"name": "unusual_volume_simple", "min_score": 2.0}],
        "feature_rules": [],
    }
    fail_rules = {
        "signal_rules": [{"name": "unusual_volume_simple", "min_score": 3.5}],
        "feature_rules": [],
    }

    pass_result = Screener(pass_rules).evaluate(enriched, signals)
    fail_result = Screener(fail_rules).evaluate(enriched, signals)

    assert pass_result.passed is True
    assert fail_result.passed is False
    assert any(r.startswith("signal_rule_failed:unusual_volume_simple") for r in fail_result.reasons)


def test_screener_feature_rule_for_price_above_sma_200() -> None:
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

    enriched = compute_features(df, ["price_above_sma_200"])

    pass_rules = {"feature_rules": [{"name": "price_above_sma_200", "op": ">=", "value": 1}], "signal_rules": []}
    fail_rules = {"feature_rules": [{"name": "price_above_sma_200", "op": "==", "value": 0}], "signal_rules": []}

    pass_result = Screener(pass_rules).evaluate(enriched, [])
    fail_result = Screener(fail_rules).evaluate(enriched, [])

    assert pass_result.passed is True
    assert fail_result.passed is False
    assert any(r.startswith("feature_rule_failed:price_above_sma_200") for r in fail_result.reasons)



def test_screener_feature_rule_for_price_above_green_avwap() -> None:
    dates = pd.date_range("2023-01-01", periods=220, freq="D")
    close = [100.0 + i * 0.2 for i in range(220)]
    df = pd.DataFrame(
        {
            "ticker": ["005930"] * 220,
            "date": dates,
            "open": close,
            "high": [c + 1.0 for c in close],
            "low": [c - 1.5 for c in close],
            "close": close,
            "volume": [1000] * 220,
        }
    )

    enriched = compute_features(df, ["price_above_green_avwap"])

    pass_rules = {"feature_rules": [{"name": "price_above_green_avwap", "op": ">=", "value": 1}], "signal_rules": []}
    fail_rules = {"feature_rules": [{"name": "price_above_green_avwap", "op": "==", "value": 0}], "signal_rules": []}

    pass_result = Screener(pass_rules).evaluate(enriched, [])
    fail_result = Screener(fail_rules).evaluate(enriched, [])

    assert pass_result.passed is True
    assert fail_result.passed is False
    assert any(r.startswith("feature_rule_failed:price_above_green_avwap") for r in fail_result.reasons)


def test_screener_feature_rule_for_price_above_yearly_avwap() -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    close = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0]
    df = pd.DataFrame(
        {
            "ticker": ["005930"] * 6,
            "date": dates,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1000] * 6,
        }
    )

    enriched = compute_features(df, ["price_above_yearly_avwap"])

    pass_rules = {"feature_rules": [{"name": "price_above_yearly_avwap", "op": ">=", "value": 1}], "signal_rules": []}
    fail_rules = {"feature_rules": [{"name": "price_above_yearly_avwap", "op": "==", "value": 0}], "signal_rules": []}

    pass_result = Screener(pass_rules).evaluate(enriched, [])
    fail_result = Screener(fail_rules).evaluate(enriched, [])

    assert pass_result.passed is True
    assert fail_result.passed is False
    assert any(r.startswith("feature_rule_failed:price_above_yearly_avwap") for r in fail_result.reasons)
