"""Weather features and simple, interpretable interactions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_weather_features(df: pd.DataFrame, temporal: pd.DataFrame | None = None) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["temp_c"] = df["temp"] - 273.15
    features["rain_1h"] = df["rain_1h"].clip(lower=0)
    features["snow_1h"] = df["snow_1h"].clip(lower=0)
    features["clouds_all"] = df["clouds_all"].clip(lower=0, upper=100)
    features["has_rain"] = (features["rain_1h"] > 0).astype("int8")
    features["has_snow"] = (features["snow_1h"] > 0).astype("int8")
    features["heavy_cloud"] = (features["clouds_all"] >= 75).astype("int8")
    features["weather_observed"] = df.get(
        "weather_observed", pd.Series(1, index=df.index)
    ).fillna(0).astype("int8")
    features["weather_main"] = df["weather_main"].fillna("Unknown").astype(str)

    if temporal is not None:
        rush = temporal["rush_hour"].reindex(features.index).fillna(0)
        features["rain_during_rush"] = features["has_rain"] * rush
        features["snow_during_rush"] = features["has_snow"] * rush
        features["temp_rush_interaction"] = features["temp_c"] * rush

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    return features
