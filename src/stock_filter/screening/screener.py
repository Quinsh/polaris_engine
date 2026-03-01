from __future__ import annotations

import pandas as pd

from stock_filter.analytics.types import ScreenResult, Signal


class Screener:
    """Apply config-driven feature and signal rules to a ticker result."""

    def __init__(self, rules: dict) -> None:
        self.rules = rules or {}

    def evaluate(self, df: pd.DataFrame, signals: list[Signal]) -> ScreenResult:
        if df.empty:
            return ScreenResult(ticker="", asof=None, passed=False, reasons=["empty_series"])

        last = df.iloc[-1]
        ticker = str(last.get("ticker", ""))
        asof = pd.Timestamp(last.get("date")).strftime("%Y-%m-%d") if "date" in df.columns else None

        reasons: list[str] = []
        signal_scores = {s.name: float(s.score) for s in signals}

        for rule in self.rules.get("feature_rules", []):
            name = rule["name"]
            op = rule.get("op", ">=")
            threshold = float(rule["value"])
            if name not in df.columns:
                reasons.append(f"missing_feature:{name}")
                continue

            value = last[name]
            if pd.isna(value):
                reasons.append(f"feature_nan:{name}")
                continue

            if not self._compare(float(value), threshold, op):
                reasons.append(f"feature_rule_failed:{name}:{value}{op}{threshold}")

        for rule in self.rules.get("signal_rules", []):
            name = rule["name"]
            min_score = float(rule.get("min_score", 0.0))
            score = signal_scores.get(name)
            if score is None:
                reasons.append(f"missing_signal:{name}")
                continue
            if score < min_score:
                reasons.append(f"signal_rule_failed:{name}:{score}<{min_score}")

        passed = len(reasons) == 0
        return ScreenResult(
            ticker=ticker,
            asof=asof,
            passed=passed,
            reasons=reasons,
            signal_scores=signal_scores,
        )

    @staticmethod
    def _compare(left: float, right: float, op: str) -> bool:
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == "==":
            return left == right
        raise ValueError(f"Unsupported operator: {op}")
