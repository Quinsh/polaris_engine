from __future__ import annotations

from typing import Mapping

import pandas as pd


_KO_TO_EN: Mapping[str, str] = {
    "날짜": "date",
    "시가": "open",
    "고가": "high",
    "저가": "low",
    "종가": "close",
    "거래량": "volume",
    "거래대금": "value",
    "등락률": "change",
}


def normalize_ohlcv(df: pd.DataFrame, *, ticker: str | None = None) -> pd.DataFrame:
    """Normalize a pykrx OHLCV dataframe into a consistent schema.

    Expected pykrx shape (from pykrx README):
    - index: 날짜 (DatetimeIndex)
    - columns include: 시가, 고가, 저가, 종가, 거래량, 거래대금, 등락률

    Output columns (when available):
    ticker, date, open, high, low, close, volume, value, change
    """
    if df is None or df.empty:
        cols = ["date", "open", "high", "low", "close", "volume", "value", "change"]
        if ticker is not None:
            cols = ["ticker"] + cols
        return pd.DataFrame(columns=cols)

    out = df.copy()

    # If the date is stored in the index (common pykrx case), expose it as a column.
    if "date" not in out.columns and "날짜" not in out.columns:
        if out.index.name in ("날짜", "date") or isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index()

    # Common reset_index column name is the original index name ("날짜") or "index".
    if "date" not in out.columns and "날짜" not in out.columns and "index" in out.columns:
        out = out.rename(columns={"index": "date"})

    out = out.rename(columns=dict(_KO_TO_EN))

    if "date" not in out.columns:
        raise ValueError("Could not locate date column/index in OHLCV dataframe.")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    for col in ("open", "high", "low", "close", "volume", "value", "change"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Optional: add ticker as a first column.
    if ticker is not None and "ticker" not in out.columns:
        out.insert(0, "ticker", ticker)

    desired = ["ticker", "date", "open", "high", "low", "close", "volume", "value", "change"]
    cols = [c for c in desired if c in out.columns]
    extras = [c for c in out.columns if c not in cols]
    out = out[cols + extras]

    out = out.sort_values("date").reset_index(drop=True)
    return out