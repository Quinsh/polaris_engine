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