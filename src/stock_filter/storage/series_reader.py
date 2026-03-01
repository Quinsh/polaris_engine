from __future__ import annotations

from pathlib import Path

import pandas as pd

_REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")


def load_series(ticker: str, freq: str, root_dir: str | Path) -> pd.DataFrame:
    """Load a cached per-ticker OHLCV parquet series with normalized schema.

    Expected path:
      {root_dir}/ohlcv_series/{freq}/{ticker}.parquet
    """
    path = Path(root_dir) / "ohlcv_series" / freq / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Series parquet not found for ticker={ticker}, freq={freq}: {path}")

    df = pd.read_parquet(path)
    if df.empty:
        cols = ["ticker", "date", "open", "high", "low", "close", "volume"]
        return pd.DataFrame(columns=cols)

    out = df.copy()
    if "ticker" not in out.columns:
        out.insert(0, "ticker", ticker)
    else:
        out["ticker"] = out["ticker"].astype(str).str.zfill(6)

    out["ticker"] = out["ticker"].fillna(str(ticker)).astype(str)

    missing = [c for c in _REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(
            f"Missing required OHLCV columns for ticker={ticker}, freq={freq}: {missing}"
        )

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume", "value", "change"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    desired = ["ticker", "date", "open", "high", "low", "close", "volume", "value", "change"]
    cols = [c for c in desired if c in out.columns]
    extras = [c for c in out.columns if c not in cols]
    return out[cols + extras].sort_values("date").reset_index(drop=True)
