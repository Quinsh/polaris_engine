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
