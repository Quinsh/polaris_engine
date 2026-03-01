from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd

FeatureFunc = Callable[[pd.DataFrame], pd.Series]
FEATURE_REGISTRY: dict[str, FeatureFunc] = {}


def register_feature(name: str) -> Callable[[FeatureFunc], FeatureFunc]:
    def decorator(func: FeatureFunc) -> FeatureFunc:
        FEATURE_REGISTRY[name] = func
        return func

    return decorator


def compute_features(df: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for name in feature_names:
        if name not in FEATURE_REGISTRY:
            raise KeyError(f"Feature not registered: {name}")
        out[name] = FEATURE_REGISTRY[name](out)
    return out
