from pathlib import Path
from typing import Tuple
import pandas as pd

from src.config import (
    CLEAN_DATA_PATH,
    DATA_PROCESSED_DIR,
    DATETIME_COL,
    HOURLY_GRID_PATH,
    TARGET_COL,
    TRAIN_RATIO,
    VAL_RATIO,
)

CLEAN_PARQUET_PATH = CLEAN_DATA_PATH


def load_processed_dataset() -> pd.DataFrame:
    """Load clean_hourly.parquet if Person B has pushed it; otherwise fall back to hourly_grid.csv."""
    if CLEAN_PARQUET_PATH.exists():
        print(f"[Data] Loading cleaned dataset: {CLEAN_PARQUET_PATH}")
        df = pd.read_parquet(CLEAN_PARQUET_PATH)
    elif HOURLY_GRID_PATH.exists():
        print(f"[Data] clean_hourly.parquet not found yet. Falling back to: {HOURLY_GRID_PATH}")
        df = pd.read_csv(HOURLY_GRID_PATH)
    else:
        raise FileNotFoundError("Neither clean_hourly.parquet nor hourly_grid.csv found. Run python -m src.load first.")

    if not isinstance(df.index, pd.DatetimeIndex):
        if DATETIME_COL in df.columns:
            df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
            df = df.set_index(DATETIME_COL)
        else:
            df.index = pd.to_datetime(df.index)
            df.index.name = DATETIME_COL

    return df.sort_index()


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-indexed DataFrame chronologically into Train (70%), Val (15%), and Test (15%)."""
    df = df.sort_index()
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def print_split_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Print a clean date-range and row-count table to copy-paste into team chat and slides."""
    total_rows = len(train_df) + len(val_df) + len(test_df)
    splits = [
        ("Train (70%)", train_df),
        ("Validation (15%)", val_df),
        ("Test (15%)", test_df),
    ]

    print("\n" + "=" * 78)
    print(f"{'Split':<18} | {'Start Timestamp':<20} | {'End Timestamp':<20} | {'Hours':>7} | {'Valid Target':>12}")
    print("-" * 78)
    for name, part in splits:
        valid_target = part[TARGET_COL].notna().sum() if TARGET_COL in part.columns else len(part)
        print(
            f"{name:<18} | {str(part.index.min()):<20} | {str(part.index.max()):<20} | "
            f"{len(part):>7,} | {valid_target:>12,}"
        )
    print("=" * 78)
    print(f"Total Hourly Grid Rows: {total_rows:,}\n")


def run_split_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute split, print report, and save train/val/test CSVs for downstream modeling."""
    df = load_processed_dataset()
    train_df, val_df, test_df = chronological_split(df)
    print_split_report(train_df, val_df, test_df)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(DATA_PROCESSED_DIR / "train.csv")
    val_df.to_csv(DATA_PROCESSED_DIR / "val.csv")
    test_df.to_csv(DATA_PROCESSED_DIR / "test.csv")
    print(f"[Done] Saved train.csv, val.csv, and test.csv to {DATA_PROCESSED_DIR}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    run_split_pipeline()
