"""Shared preprocessing and time-series validation helpers."""

from __future__ import annotations

from typing import Iterable, List

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(feature_names: Iterable[str], scale_numeric: bool) -> ColumnTransformer:
    feature_names = list(feature_names)
    categorical: List[str] = [name for name in feature_names if name == "weather_main"]
    numeric = [name for name in feature_names if name not in categorical]

    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    transformers = [("numeric", Pipeline(numeric_steps), numeric)]
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def time_series_cv() -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=4, gap=24)
