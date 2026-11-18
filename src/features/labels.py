"""Train-derived congestion labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.config import CONGESTION_LABELS, THRESHOLDS_PATH


def fit_thresholds(train_target: pd.Series) -> Dict[str, float]:
    low, high = train_target.dropna().quantile([1 / 3, 2 / 3]).tolist()
    return {"low_to_moderate": float(low), "moderate_to_high": float(high)}


def apply_thresholds(target: pd.Series, thresholds: Dict[str, float]) -> pd.Series:
    values = pd.cut(
        target,
        bins=[-np.inf, thresholds["low_to_moderate"], thresholds["moderate_to_high"], np.inf],
        labels=False,
        include_lowest=True,
    )
    return pd.Series(values, index=target.index, name="congestion_class").astype("int8")


def save_thresholds(thresholds: Dict[str, float], path: Path = THRESHOLDS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**thresholds, "labels": CONGESTION_LABELS, "source": "training quantiles only"}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
