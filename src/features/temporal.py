"""Calendar features that are known at prediction time."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Temporal features require a DatetimeIndex")

    index = df.index
    features = pd.DataFrame(index=index)
    features["hour"] = index.hour.astype("int8")
    features["dayofweek"] = index.dayofweek.astype("int8")
    features["month"] = index.month.astype("int8")
    features["dayofyear"] = index.dayofyear.astype("int16")
    features["is_weekend"] = (index.dayofweek >= 5).astype("int8")
    features["is_holiday"] = df.get("is_holiday", pd.Series(0, index=index)).fillna(0).astype("int8")

    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
    features["dow_sin"] = np.sin(2 * np.pi * features["dayofweek"] / 7)
    features["dow_cos"] = np.cos(2 * np.pi * features["dayofweek"] / 7)
    features["month_sin"] = np.sin(2 * np.pi * (features["month"] - 1) / 12)
    features["month_cos"] = np.cos(2 * np.pi * (features["month"] - 1) / 12)
    features["rush_hour"] = (
        ((features["hour"].between(6, 9)) | (features["hour"].between(15, 18)))
        & (features["is_weekend"] == 0)
    ).astype("int8")
    return features
