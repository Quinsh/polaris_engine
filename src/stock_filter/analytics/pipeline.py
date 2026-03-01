from __future__ import annotations

from pathlib import Path

from stock_filter.analytics.types import ScreenResult
from stock_filter.features import compute_features
from stock_filter.screening import Screener
from stock_filter.signals import SIGNAL_REGISTRY, detect_signals
from stock_filter.storage import load_series


def run_screen_pipeline(
    *,
    ticker: str,
    freq: str,
    root_dir: str | Path,
    config: dict,
) -> ScreenResult:
    df = load_series(ticker=ticker, freq=freq, root_dir=root_dir)

    requested_features = list(config.get("features", []))
    requested_signals = list(config.get("signals", []))

    required_features = set(requested_features)
    for signal_name in requested_signals:
        detector = SIGNAL_REGISTRY.get(signal_name)
        if detector is None:
            raise KeyError(f"Signal not registered: {signal_name}")
        required_features.update(detector.required_features)

    df_with_features = compute_features(df, sorted(required_features)) if required_features else df
    signals = detect_signals(df_with_features, requested_signals)

    screener = Screener(config.get("rules", {}))
    return screener.evaluate(df_with_features, signals)
