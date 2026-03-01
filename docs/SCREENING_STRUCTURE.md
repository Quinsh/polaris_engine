# Screening pipeline: file structure and where to add code

## High-level flow

```
OHLCV series (from cache)
    → load_series()
    → compute_features(df, feature_names)   →  df with extra columns
    → detect_signals(df, signal_names)       →  list[Signal]
    → Screener.evaluate(df, signals)         →  ScreenResult (passed/failed + scores)
```

- **Features**: per-row series (e.g. `returns_1d`, `volume_sma_20`, `price_above_sma_200`). One value per date.
- **Signals**: detectors that look at the series (often the latest bar or a window) and return a **Signal** (name, score, details) or `None`. Used for filtering (e.g. “unusual volume”, **chart patterns**).
- **Rules**: thresholds applied in the screener (e.g. feature `>= 1`, signal `min_score >= 2.0`).

---

## Where things live

| Layer | Directory / file | Purpose |
|-------|-------------------|---------|
| **Entry** | `app/interactive.py` | Menu, prompts, space-separated feature/signal choice, runs pipeline. |
| **Pipeline** | `analytics/pipeline.py` | Loads series, computes features, detects signals, runs screener. |
| **Types** | `analytics/types.py` | `Signal`, `ScreenResult`. |
| **Features** | `features/registry.py` | `FEATURE_REGISTRY`, `register_feature`, `compute_features`. |
| **Features (impl)** | `features/defaults.py` | Built-in features: returns_1d, volume_sma_20, price_above_sma_200. |
| **Signals** | `signals/registry.py` | `SIGNAL_REGISTRY`, `register_signal`, `detect_signals`. |
| **Signals (impl)** | `signals/defaults.py` | Built-in signals (e.g. unusual_volume_simple). |
| **Signals (patterns)** | `signals/patterns.py` | **Chart patterns: VCP, double bottom, bullish pennant, etc.** |
| **Screening** | `screening/screener.py` | Applies feature_rules and signal_rules to pass/fail. |
| **Rule defaults** | `screening/rule_defaults.py` | Default thresholds for features/signals (for the interactive app). |

---

## Where to add pattern filters (VCP, double bottom, bullish pennant)

Patterns are **signals**: they take a DataFrame (OHLCV + optional features), look at recent price/volume, and return a `Signal` with a score (e.g. confidence 0–1) or `None`.

1. **Implement detectors** in **`signals/patterns.py`**  
   - One function per pattern (e.g. `vcp`, `double_bottom`, `bullish_pennant`).  
   - Each returns `Signal | None`, registered with `@register_signal("name", required_features=[...])`.  
   - Use `required_features=[]` if you only need OHLCV; add names (e.g. `"volume_sma_20"`) if the pattern uses them.

2. **Make them load** in **`signals/__init__.py`**  
   - Import the patterns module so `register_signal` runs and they appear in `SIGNAL_REGISTRY` (e.g. `from stock_filter.signals import patterns as _patterns  # noqa: F401`).

3. **Optional defaults for screening** in **`screening/rule_defaults.py`**  
   - Add entries to `DEFAULT_SIGNAL_RULES` (e.g. `"vcp": {"min_score": 0.7}`) so the interactive app can filter by that pattern with a default threshold.

No changes are needed in `app/interactive.py` or `analytics/pipeline.py`; they already use the registries.

---

## Signal detector contract

- **Input**: `df: pd.DataFrame` with at least OHLCV (`open`, `high`, `low`, `close`, `volume`) and any `required_features` columns.
- **Output**: `Signal(name=..., ticker=..., asof=..., score=..., details=...)` if the pattern is detected, else `None`.
- **Score**: Float used by the screener (e.g. `min_score`). For patterns, often a confidence 0–1.
- **required_features**: List of feature names that must be computed before this detector runs (can be `[]` for OHLCV-only).

---

## Feature vs signal (when to use which)

- **Feature**: A new column for every row (e.g. SMA, returns). Use when you need a series for other signals or rules.
- **Signal**: A single “detection” per ticker (one `Signal` or `None`). Use for “is this pattern present?” or “how strong is this pattern?” — e.g. VCP, double bottom, bullish pennant.

So: **VCP, double bottom, bullish pennant → implement as signals in `signals/patterns.py`.**
