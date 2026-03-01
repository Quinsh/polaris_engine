from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import pandas as pd

from stock_filter.analytics.types import Signal


@dataclass(frozen=True)
class SignalDetector:
    name: str
    required_features: tuple[str, ...]
    detect: Callable[[pd.DataFrame], Signal | None]


SIGNAL_REGISTRY: dict[str, SignalDetector] = {}


def register_signal(name: str, *, required_features: list[str] | tuple[str, ...] | None = None):
    required = tuple(required_features or ())

    def decorator(func: Callable[[pd.DataFrame], Signal | None]) -> Callable[[pd.DataFrame], Signal | None]:
        SIGNAL_REGISTRY[name] = SignalDetector(name=name, required_features=required, detect=func)
        return func

    return decorator


def detect_signals(df: pd.DataFrame, signal_names: Iterable[str]) -> list[Signal]:
    found: list[Signal] = []
    for name in signal_names:
        if name not in SIGNAL_REGISTRY:
            raise KeyError(f"Signal not registered: {name}")
        signal = SIGNAL_REGISTRY[name].detect(df)
        if signal is not None:
            found.append(signal)
    return found
