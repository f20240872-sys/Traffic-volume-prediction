from pathlib import Path
import pandas as pd

from src.config import (
    CATEGORICAL_COLS,
    DATA_PROCESSED_DIR,
    DATETIME_COL,
    HOURLY_GRID_PATH,
    NUMERIC_WEATHER_COLS,
    RAW_DATA_PATH,
    TARGET_COL,
)


def detect_compression(filepath: Path) -> str | None:
    """Detect gzip or zip compression even if the file extension is .csv."""
    with open(filepath, "rb") as f:
        header = f.read(4)
    if header.startswith(b"\x1f\x8b"):
        return "gzip"
    if header.startswith(b"PK\x03\x04"):
        return "zip"
    return None


def read_raw_csv(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw UCI traffic dataset while preserving 'None' in holiday column."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. Place Metro_Interstate_Traffic_Volume.csv in data/raw/."
        )

    compression = detect_compression(filepath)
    df = pd.read_csv(
        filepath,
        compression=compression,
        keep_default_na=False,
        na_values=["", "NA", "NaN", "null", "NULL"],
    )
    return df


def parse_and_sort_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date_time column to pandas Timestamp and sort chronologically."""
    df = df.copy()
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors="coerce")
    invalid_count = df[DATETIME_COL].isna().sum()
    if invalid_count > 0:
        print(f"[Audit] Dropping {invalid_count} rows with unparseable timestamps.")
        df = df.dropna(subset=[DATETIME_COL])

    df = df.sort_values(DATETIME_COL).reset_index(drop=True)
    return df


def collapse_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse duplicate timestamps (caused by multiple weather tags in the same hour).
    - Numeric columns (traffic_volume + weather metrics): averaged via mean
    - Categorical columns (holiday, weather_main, weather_description): first entry kept
    """
    dup_count = df.duplicated(subset=[DATETIME_COL]).sum()
    print(f"[Audit] Raw rows: {len(df):,} | Duplicate timestamp rows: {dup_count:,}")

    agg_rules = {TARGET_COL: "mean"}
    for col in NUMERIC_WEATHER_COLS:
        if col in df.columns:
            agg_rules[col] = "mean"
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            agg_rules[col] = "first"

    df_unique = (
        df.groupby(DATETIME_COL, as_index=True)
        .agg(agg_rules)
        .sort_index()
    )
    print(f"[Audit] Unique hourly timestamps after collapsing: {len(df_unique):,}")
    return df_unique


def reindex_to_hourly_grid(df_unique: pd.DataFrame) -> pd.DataFrame:
    """Reindex onto a continuous hourly grid so missing hours appear explicitly as NaN."""
    start_ts = df_unique.index.min()
    end_ts = df_unique.index.max()
    full_hourly_index = pd.date_range(start=start_ts, end=end_ts, freq="h")
    full_hourly_index.name = DATETIME_COL

    df_grid = df_unique.reindex(full_hourly_index)
    missing_hours = len(df_grid) - len(df_unique)

    print(f"[Audit] Date range: {start_ts} -> {end_ts}")
    print(f"[Audit] Full hourly grid rows: {len(df_grid):,} (Missing hours added: {missing_hours:,})")
    print("\n[Audit] Missing values per column on hourly grid:")
    print(df_grid.isna().sum().to_string())

    return df_grid


def run_load_pipeline(
    raw_path: Path = RAW_DATA_PATH,
    output_path: Path = HOURLY_GRID_PATH,
) -> pd.DataFrame:
    """Execute full Day 1 loading, deduplication, and hourly grid reindexing."""
    df_raw = read_raw_csv(raw_path)
    df_parsed = parse_and_sort_datetime(df_raw)
    df_unique = collapse_duplicate_timestamps(df_parsed)
    df_grid = reindex_to_hourly_grid(df_unique)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_grid.to_csv(output_path, index=True)
    print(f"\n[Done] Saved hourly grid dataset to: {output_path}")
    return df_grid


if __name__ == "__main__":
    run_load_pipeline()