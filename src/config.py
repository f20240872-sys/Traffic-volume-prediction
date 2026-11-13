from pathlib import Path

# Project directory paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RAW_DATA_PATH = DATA_RAW_DIR / "Metro_Interstate_Traffic_Volume.csv"
HOURLY_GRID_PATH = DATA_PROCESSED_DIR / "hourly_grid.csv"

# Reproducibility & chronological split ratios (70 / 15 / 15)
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Dataset schema
DATETIME_COL = "date_time"
TARGET_COL = "traffic_volume"

NUMERIC_WEATHER_COLS = ["temp", "rain_1h", "snow_1h", "clouds_all"]
CATEGORICAL_COLS = ["holiday", "weather_main", "weather_description"]

# Congestion classification labels (thresholds derived strictly on train split)
CONGESTION_LABELS = ["Low", "Moderate", "High"]