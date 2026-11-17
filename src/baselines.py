import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import FIGURES_DIR, TARGET_COL
from src.split import chronological_split, load_processed_dataset

sns.set_theme(style="whitegrid", context="talk")


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create purely calendar-based features from the DatetimeIndex (zero leakage)."""
    feats = pd.DataFrame(index=df.index)
    feats["hour"] = df.index.hour
    feats["dayofweek"] = df.index.dayofweek
    feats["month"] = df.index.month
    feats["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    if "holiday" in df.columns:
        feats["is_holiday"] = (df["holiday"].fillna("None") != "None").astype(int)
    else:
        feats["is_holiday"] = 0

    return feats


def compute_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, and R2 for continuous traffic volume predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def plot_baseline_comparison(results_df: pd.DataFrame) -> None:
    """Save a bar chart comparing Validation MAE and RMSE across the 3 baselines."""
    val_df = results_df[results_df["Split"] == "Validation"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(data=val_df, x="Model", y="MAE", palette="Blues_d", ax=axes[0])
    axes[0].set_title("Validation MAE (Lower is Better)", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("MAE (vehicles/hour)")
    axes[0].tick_params(axis="x", rotation=15)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.1f", padding=3, fontsize=10)

    sns.barplot(data=val_df, x="Model", y="RMSE", palette="Oranges_d", ax=axes[1])
    axes[1].set_title("Validation RMSE (Lower is Better)", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("RMSE (vehicles/hour)")
    axes[1].tick_params(axis="x", rotation=15)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.1f", padding=3, fontsize=10)

    plt.suptitle("Day 3 Baseline Benchmarks (Validation Set)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "05_baseline_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Baseline comparison figure -> {out_path}")


def run_baselines() -> pd.DataFrame:
    """Train and evaluate the 3 Day-3 baselines without future data leakage."""
    df = load_processed_dataset()

    # Strictly backward-looking 168-hour (same hour last week) lag on the continuous grid
    df["lag_168"] = df[TARGET_COL].shift(168)

    train_df, val_df, test_df = chronological_split(df)

    # Fit baselines strictly on valid training rows
    train_valid = train_df.dropna(subset=[TARGET_COL]).copy()
    train_feats = extract_temporal_features(train_valid)
    y_train = train_valid[TARGET_COL]

    # 1. Hour-of-Week Train Mean lookup table (7 days x 24 hours = 168 buckets)
    how_lookup = (
        train_valid.assign(dayofweek=train_valid.index.dayofweek, hour=train_valid.index.hour)
        .groupby(["dayofweek", "hour"])[TARGET_COL]
        .mean()
        .rename("pred_how_mean")
    )

    # 2. Linear Regression on simple one-hot temporal features
    cat_cols = ["hour", "dayofweek", "month"]
    num_cols = ["is_weekend", "is_holiday"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    linreg_pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", LinearRegression()),
        ]
    )
    linreg_pipeline.fit(train_feats, y_train)

    # Evaluate on Validation and Test using the exact mask where target AND lag_168 exist
    records = []
    for split_name, split_df in [("Validation", val_df), ("Test", test_df)]:
        valid_mask = split_df[TARGET_COL].notna() & split_df["lag_168"].notna()
        eval_df = split_df.loc[valid_mask].copy()
        y_true = eval_df[TARGET_COL]

        # Model 1: Same Hour Last Week (lag_168)
        pred_lag168 = eval_df["lag_168"].values

        # Model 2: Hour-of-Week Train Mean
        eval_how = eval_df.assign(dayofweek=eval_df.index.dayofweek, hour=eval_df.index.hour)
        pred_how = eval_how.join(how_lookup, on=["dayofweek", "hour"])["pred_how_mean"].values

        # Model 3: Linear Regression (Temporal Features)
        eval_feats = extract_temporal_features(eval_df)
        pred_linreg = np.clip(linreg_pipeline.predict(eval_feats), 0.0, None)

        models = [
            ("1. Same Hour Last Week (lag_168)", pred_lag168),
            ("2. Hour-of-Week Train Mean", pred_how),
            ("3. Linear Regression (Temporal)", pred_linreg),
        ]

        for model_name, preds in models:
            m = compute_regression_metrics(y_true, preds)
            records.append(
                {
                    "Split": split_name,
                    "Model": model_name,
                    "Evaluated_Hours": len(y_true),
                    "MAE": round(m["MAE"], 2),
                    "RMSE": round(m["RMSE"], 2),
                    "R2": round(m["R2"], 4),
                }
            )

    results_df = pd.DataFrame(records)

    print("\n" + "=" * 86)
    print("DAY 3 BASELINE BENCHMARK RESULTS (Scored on Hours with Valid Target & lag_168)")
    print("=" * 86)
    print(results_df.to_string(index=False))
    print("=" * 86 + "\n")

    reports_dir = FIGURES_DIR.parent
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "baseline_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"[Saved] Baseline metrics table -> {csv_path}")

    plot_baseline_comparison(results_df)
    return results_df


if __name__ == "__main__":
    run_baselines()