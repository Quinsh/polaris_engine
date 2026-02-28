from __future__ import annotations

from pathlib import Path
import pandas as pd


class OhlcvSeriesStore:
    """
    Stores one continuous OHLCV series per ticker.

    Path:
      {root}/ohlcv_series/{freq}/{ticker}.parquet
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path_for(self, *, ticker: str, freq: str) -> Path:
        return self.root / "ohlcv_series" / freq / f"{ticker}.parquet"

    def exists(self, *, ticker: str, freq: str) -> bool:
        return self.path_for(ticker=ticker, freq=freq).exists()

    def load(self, *, ticker: str, freq: str) -> pd.DataFrame:
        return pd.read_parquet(self.path_for(ticker=ticker, freq=freq))

    def save(self, *, ticker: str, freq: str, df: pd.DataFrame) -> Path:
        path = self.path_for(ticker=ticker, freq=freq)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path

    def last_date(self, *, ticker: str, freq: str) -> pd.Timestamp | None:
        path = self.path_for(ticker=ticker, freq=freq)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path, columns=["date"])
        except Exception:
            # fallback if parquet reader can't do column projection
            df = pd.read_parquet(path)
            if "date" not in df.columns:
                return None
            df = df[["date"]]

        if df.empty:
            return None

        s = pd.to_datetime(df["date"], errors="coerce")
        if s.isna().all():
            return None
        return s.max()