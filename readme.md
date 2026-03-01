<!-- README.md -->
# stockfilter (prototype)

This repository is a minimal, production-shaped prototype for a Korean stock screening engine.

Scope (for this prototype):
- Select a universe: `kospi100` or `kosdaq100`
- Fetch raw OHLCV using `pykrx`
- Normalize columns to a consistent English schema
- Cache per-ticker OHLCV to local Parquet files (skip refetch if cached)
- Provide a CLI: `stockfilter universe` and `stockfilter fetch`

Out of scope (intentionally not implemented yet):
- Indicators (MFI, ATR, etc.)
- Pattern detection (VCP, pennants, cup & handle, etc.)
- Filtering logic / ranking

## Requirements

- Python 3.11+
- `pykrx` (scraping-based; may break if upstream websites change)
- `pyarrow` is required for Parquet caching.

## Install (dev)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```

## Screening skeleton (features + signals + rules)

The repository now includes an additive screening framework for cached OHLCV series under:
`{out}/ohlcv_series/{freq}/{ticker}.parquet`.

### CLI: screen

Single ticker:

```bash
stockfilter screen --ticker 005930 --freq d --out data/cache
```

Universe mode from static CSVs (`kospi100`/`kosdaq100`):

```bash
stockfilter screen --markets kospi,kosdaq --freq d --out data/cache --limit 20
```

Behavior:
- Computes requested features.
- Runs requested signal detectors.
- Applies configurable screening rules.
- Prints ranked pass list by signal score (descending).
- Missing ticker series are skipped with warnings.

### Add a new feature

1. Create a function in `src/stock_filter/features/defaults.py` (or another imported module):

```python
from stock_filter.features.registry import register_feature

@register_feature("my_feature")
def my_feature(df):
    return df["close"].pct_change(5)
```

2. Reference `"my_feature"` in pipeline config (`features` list).

### Add a new signal

1. Create a detector and declare dependencies:

```python
from stock_filter.signals.registry import register_signal

@register_signal("my_signal", required_features=["my_feature"])
def my_signal(df):
    ...
```

2. Return `Signal` or `None`.
3. Reference it in pipeline config (`signals`) and screener rules (`signal_rules`).
