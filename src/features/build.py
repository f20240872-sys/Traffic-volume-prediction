"""Assemble consistent feature matrices for every experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.config import DATA_PROCESSED_DIR, RESULTS_DIR, TARGET_COL
from src.features.lags import build_lag_features
from src.features.temporal import build_temporal_features
from src.features.weather import build_weather_features
from src.split import chronological_split, load_processed_dataset


FEATURE_SET_PATH = RESULTS_DIR / "feature_sets.json"


def assemble_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    temporal = build_temporal_features(df)
    weather = build_weather_features(df, temporal)
    lags = build_lag_features(df)
    frame = pd.concat([temporal, weather, lags, df[[TARGET_COL]]], axis=1)

    groups = {
        "temporal": list(temporal.columns),
        "weather": list(weather.columns),
        "lags": list(lags.columns),
    }
    return frame, groups


def feature_sets(groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    t, w, lag = groups["temporal"], groups["weather"], groups["lags"]
    return {
        "T": t,
        "W": w,
        "L": lag,
        "T+W": t + w,
        "T+L": t + lag,
        "T+W+L": t + w + lag,
    }


def common_evaluation_mask(frame: pd.DataFrame, lag_columns: List[str]) -> pd.Series:
    """Require the target and all lag features for fair feature-set comparisons."""
    return frame[TARGET_COL].notna() & frame[lag_columns].notna().all(axis=1)


def build_feature_splits(
    data: pd.DataFrame | None = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]]]:
    data = load_processed_dataset() if data is None else data.sort_index()
    full, groups = assemble_feature_frame(data)
    sets = feature_sets(groups)
    train_raw, val_raw, test_raw = chronological_split(data)
    mask = common_evaluation_mask(full, groups["lags"])

    splits = {
        "train": full.loc[train_raw.index][mask.loc[train_raw.index]].copy(),
        "val": full.loc[val_raw.index][mask.loc[val_raw.index]].copy(),
        "test": full.loc[test_raw.index][mask.loc[test_raw.index]].copy(),
    }
    return splits, sets


def save_feature_splits(output_dir: Path = DATA_PROCESSED_DIR) -> Dict[str, pd.DataFrame]:
    splits, sets = build_feature_splits()
    output_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_parquet(output_dir / f"features_{name}.parquet")

    metadata = {
        "feature_sets": sets,
        "categorical_features": ["weather_main"],
        "split_rows": {name: len(frame) for name, frame in splits.items()},
        "evaluation_rule": "All rows have a target and every lag/rolling feature.",
    }
    FEATURE_SET_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Feature rows:", metadata["split_rows"])
    return splits


if __name__ == "__main__":
    save_feature_splits()
