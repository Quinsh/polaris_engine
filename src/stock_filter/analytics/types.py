from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Signal:
    name: str
    ticker: str
    asof: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalSet:
    items: list[Signal] = field(default_factory=list)


@dataclass(frozen=True)
class ScreenResult:
    ticker: str
    asof: str | None
    passed: bool
    reasons: list[str] = field(default_factory=list)
    signal_scores: dict[str, float] = field(default_factory=dict)
