"""Deterministic data cleaning for the hourly traffic series."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_COLS,
    CLEAN_DATA_PATH,
    DATA_PROCESSED_DIR,
    DATETIME_COL,
    NUMERIC_WEATHER_COLS,
    RAW_DATA_PATH,
    TABLES_DIR,
    TARGET_COL,
)
from src.load import parse_and_sort_datetime, read_raw_csv


MAX_WEATHER_FORWARD_FILL_HOURS = 6
RAIN_OUTLIER_LIMIT_MM = 500.0


def _first_named_holiday(values: pd.Series) -> str:
    named = values[(values.notna()) & (values.astype(str) != "None")]
    return str(named.iloc[0]) if not named.empty else "None"


def _mode_or_missing(values: pd.Series) -> object:
    valid = values.dropna()
    if valid.empty:
        return np.nan
    mode = valid.mode()
    return mode.iloc[0] if not mode.empty else valid.iloc[0]


def collapse_weather_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Collapse multiple weather descriptions recorded for the same hour."""
    duplicate_rows = int(df.duplicated(subset=[DATETIME_COL]).sum())
    rules: Dict[str, object] = {TARGET_COL: "mean"}
    rules.update({col: "mean" for col in NUMERIC_WEATHER_COLS if col in df.columns})
    rules.update(
        {
            col: (_first_named_holiday if col == "holiday" else _mode_or_missing)
            for col in CATEGORICAL_COLS
            if col in df.columns
        }
    )
    hourly = df.groupby(DATETIME_COL).agg(rules).sort_index()
    return hourly, duplicate_rows


def build_hourly_grid(hourly: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Put observations on a continuous grid without filling the target."""
    index = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h", name=DATETIME_COL)
    grid = hourly.reindex(index)
    return grid, int(len(grid) - len(hourly))


def propagate_holiday_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Expand midnight-only holiday markers to all hours on the holiday date."""
    result = df.copy()
    holiday = result["holiday"].fillna("None").astype(str)
    named = holiday[holiday != "None"]
    date_to_name = {timestamp.date(): name for timestamp, name in named.items()}
    result["holiday"] = [date_to_name.get(timestamp.date(), "None") for timestamp in result.index]
    result["is_holiday"] = (result["holiday"] != "None").astype("int8")
    return result


def apply_cleaning_rules(grid: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Apply domain rules and causal weather filling; never impute traffic volume."""
    clean = grid.copy()
    clean["weather_observed"] = clean["temp"].notna().astype("int8")

    zero_kelvin = int((clean["temp"] <= 0).sum())
    rain_outliers = int((clean["rain_1h"] > RAIN_OUTLIER_LIMIT_MM).sum())
    clean.loc[clean["temp"] <= 0, "temp"] = np.nan
    clean.loc[clean["rain_1h"] > RAIN_OUTLIER_LIMIT_MM, "rain_1h"] = np.nan

    weather_columns = [
        col
        for col in NUMERIC_WEATHER_COLS + ["weather_main", "weather_description"]
        if col in clean.columns
    ]
    clean[weather_columns] = clean[weather_columns].ffill(limit=MAX_WEATHER_FORWARD_FILL_HOURS)
    clean = propagate_holiday_dates(clean)

    audit = {
        "zero_kelvin_replaced": zero_kelvin,
        "rain_outliers_replaced": rain_outliers,
        "target_values_imputed": 0,
        "remaining_missing_target": int(clean[TARGET_COL].isna().sum()),
        "remaining_missing_weather_rows": int(clean["temp"].isna().sum()),
    }
    return clean, audit


def build_quality_summary(
    raw_rows: int,
    unique_hours: int,
    grid_rows: int,
    duplicate_rows: int,
    missing_hours: int,
    audit: Dict[str, int],
) -> pd.DataFrame:
    records = [
        ("Raw rows", raw_rows),
        ("Unique observed hours", unique_hours),
        ("Duplicate timestamp rows collapsed", duplicate_rows),
        ("Hourly grid rows", grid_rows),
        ("Missing hours added", missing_hours),
        ("Zero Kelvin values replaced", audit["zero_kelvin_replaced"]),
        ("Extreme rain values replaced", audit["rain_outliers_replaced"]),
        ("Traffic targets imputed", audit["target_values_imputed"]),
        ("Hours without traffic target", audit["remaining_missing_target"]),
        ("Hours still missing weather", audit["remaining_missing_weather_rows"]),
    ]
    return pd.DataFrame(records, columns=["Metric", "Value"])


def run_clean_pipeline(
    raw_path: Path = RAW_DATA_PATH,
    output_path: Path = CLEAN_DATA_PATH,
) -> pd.DataFrame:
    raw = parse_and_sort_datetime(read_raw_csv(raw_path))
    hourly, duplicate_rows = collapse_weather_rows(raw)
    grid, missing_hours = build_hourly_grid(hourly)
    clean, audit = apply_cleaning_rules(grid)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(output_path)

    summary = build_quality_summary(
        raw_rows=len(raw),
        unique_hours=len(hourly),
        grid_rows=len(grid),
        duplicate_rows=duplicate_rows,
        missing_hours=missing_hours,
        audit=audit,
    )
    summary.to_csv(TABLES_DIR / "data_quality_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved cleaned hourly data to {output_path}")
    return clean


if __name__ == "__main__":
    run_clean_pipeline()
