"""Backward-looking target lags and rolling statistics."""

from __future__ import annotations

import pandas as pd

from src.config import LAG_HOURS, ROLLING_WINDOWS, TARGET_COL


def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    target = df[TARGET_COL]
    features = pd.DataFrame(index=df.index)
    for lag in LAG_HOURS:
        features[f"lag_{lag}"] = target.shift(lag)

    shifted = target.shift(1)
    for window in ROLLING_WINDOWS:
        rolling = shifted.rolling(window=window, min_periods=window)
        features[f"rolling_mean_{window}"] = rolling.mean()
        features[f"rolling_std_{window}"] = rolling.std()
    return features
