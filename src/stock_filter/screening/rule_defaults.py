# Default rule thresholds for screening. Add an entry here when you add a new
# feature or signal that should be used as a filter by default.

# Feature rules: name -> { "o4th_resistance_test": ">=" | ">" | "<" | "<=" | "==", "value": number }
DEFAULT_FEATURE_RULES: dict[str, dict] = {
    "price_above_sma_200": {"op": ">=", "value": 1},
}

# Signal rules: name -> { "min_score": number }
DEFAULT_SIGNAL_RULES: dict[str, dict] = {
    "unusual_volume_simple": {"min_score": 2.0},
    # Chart patterns (score typically 0–1 confidence); adjust after implementing logic
    "vcp": {"min_score": 0.5},
    "double_bottom": {"min_score": 0.5},
    "bullish_pennant": {"min_score": 0.5},
    "support_test": {"min_score": 0.5},
    "4th_resistance_test": {"min_score": 0.5},
    "wave2": {"min_score": 0.5},
    "rsi_mfi_smart_buy": {"min_score": 5.0},
    "rsi_mfi_smart_sell": {"min_score": 5.0},
}


def build_feature_rules(feature_names: list[str]) -> list[dict]:
    """Build feature_rules list for selected features that have a default rule."""
    return [
        {"name": name, **DEFAULT_FEATURE_RULES[name]}
        for name in feature_names
        if name in DEFAULT_FEATURE_RULES
    ]


def build_signal_rules(signal_names: list[str]) -> list[dict]:
    """Build signal_rules list for selected signals that have a default rule."""
    return [
        {"name": name, **DEFAULT_SIGNAL_RULES[name]}
        for name in signal_names
        if name in DEFAULT_SIGNAL_RULES
    ]
