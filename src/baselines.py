"""Validation-only reference models used before model tuning."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATA_PROCESSED_DIR, FIGURES_DIR, METRICS_DIR, TARGET_COL
from src.metrics import evaluate_regression


sns.set_theme(style="whitegrid", context="talk")


def load_feature_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(DATA_PROCESSED_DIR / "features_train.parquet")
    val = pd.read_parquet(DATA_PROCESSED_DIR / "features_val.parquet")
    return train, val


def temporal_linear_pipeline() -> Pipeline:
    categorical = ["hour", "dayofweek", "month"]
    numeric = [
        "dayofyear", "is_weekend", "is_holiday", "hour_sin", "hour_cos",
        "dow_sin", "dow_cos", "month_sin", "month_cos", "rush_hour",
    ]
    preprocess = ColumnTransformer(
        [
            ("calendar", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", StandardScaler(), numeric),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", LinearRegression())])


def run_baselines() -> pd.DataFrame:
    train, val = load_feature_data()
    y_train = train[TARGET_COL]
    y_val = val[TARGET_COL]

    hour_of_week = (
        train.assign(hour_of_week=train.index.dayofweek * 24 + train.index.hour)
        .groupby("hour_of_week")[TARGET_COL]
        .mean()
    )
    val_hour_of_week = val.index.dayofweek * 24 + val.index.hour

    linear = temporal_linear_pipeline()
    linear_features = [
        "hour", "dayofweek", "month", "dayofyear", "is_weekend", "is_holiday",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "rush_hour",
    ]
    linear.fit(train[linear_features], y_train)

    predictions = {
        "Same hour last week": val["lag_168"].to_numpy(),
        "Hour-of-week train mean": pd.Series(val_hour_of_week, index=val.index).map(hour_of_week).to_numpy(),
        "Temporal linear regression": linear.predict(val[linear_features]).clip(min=0),
    }
    rows = [
        {"Split": "Validation", **evaluate_regression(y_val, pred, name)}
        for name, pred in predictions.items()
    ]
    results = pd.DataFrame(rows)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(METRICS_DIR / "baseline_validation.csv", index=False)

    long = results.melt(
        id_vars=["Model", "Split"],
        value_vars=["MAE", "RMSE"],
        var_name="Metric",
        value_name="Value",
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    sns.barplot(data=long, x="Model", y="Value", hue="Metric", ax=ax)
    ax.set_title("Validation performance of reference models", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Error (vehicles/hour)")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "06_baseline_comparison.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    run_baselines()
