from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CacheKey:
    ticker: str
    start: str
    end: str
    freq: str  # 'd' | 'm' | 'y'


class ParquetCache:
    """Very small Parquet cache.

    Layout:
      {root}/ohlcv/{freq}/{ticker}/{start}_{end}.parquet
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path_for(self, key: CacheKey) -> Path:
        return (
            self.root
            / "ohlcv"
            / key.freq
            / key.ticker
            / f"{key.start}_{key.end}.parquet"
        )

    def exists(self, key: CacheKey) -> bool:
        return self.path_for(key).exists()

    def load(self, key: CacheKey) -> pd.DataFrame:
        path = self.path_for(key)
        return pd.read_parquet(path)

    def save(self, key: CacheKey, df: pd.DataFrame) -> Path:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path