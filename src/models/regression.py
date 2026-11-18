"""Tune regression models on train and compare them on validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.config import (
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    TARGET_COL,
)
from src.metrics import evaluate_regression
from src.models.common import make_preprocessor, time_series_cv
from src.models.mlp_torch import train_mlp


sns.set_theme(style="whitegrid", context="talk")


def load_training_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_parquet(DATA_PROCESSED_DIR / "features_train.parquet")
    val = pd.read_parquet(DATA_PROCESSED_DIR / "features_val.parquet")
    metadata = json.loads((RESULTS_DIR / "feature_sets.json").read_text(encoding="utf-8"))
    features = metadata["feature_sets"]["T+W+L"]
    return train, val, features


def model_searches(features: list[str]) -> Dict[str, object]:
    cv = time_series_cv()
    tree_preprocess = make_preprocessor(features, scale_numeric=False)
    linear = Pipeline(
        [
            ("preprocess", make_preprocessor(features, scale_numeric=True)),
            ("model", LinearRegression()),
        ]
    )

    decision_tree = RandomizedSearchCV(
        Pipeline([("preprocess", tree_preprocess), ("model", DecisionTreeRegressor(random_state=RANDOM_SEED))]),
        {
            "model__max_depth": [6, 10, 14, None],
            "model__min_samples_leaf": [3, 10, 25, 50],
            "model__max_features": [None, "sqrt", 0.8],
        },
        n_iter=10,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    random_forest = RandomizedSearchCV(
        Pipeline(
            [
                ("preprocess", make_preprocessor(features, scale_numeric=False)),
                ("model", RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=1)),
            ]
        ),
        {
            "model__n_estimators": [180, 260, 360],
            "model__max_depth": [10, 16, None],
            "model__min_samples_leaf": [1, 3, 8],
            "model__max_features": ["sqrt", 0.7, 1.0],
        },
        n_iter=8,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    xgboost = RandomizedSearchCV(
        Pipeline(
            [
                ("preprocess", make_preprocessor(features, scale_numeric=False)),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        random_state=RANDOM_SEED,
                        n_jobs=1,
                        tree_method="hist",
                    ),
                ),
            ]
        ),
        {
            "model__n_estimators": [250, 400, 600],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.03, 0.06, 0.10],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        },
        n_iter=8,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    return {
        "Linear Regression": linear,
        "Decision Tree": decision_tree,
        "Random Forest": random_forest,
        "XGBoost": xgboost,
    }


def _best_params(model: object) -> Dict[str, object]:
    params = getattr(model, "best_params_", {})
    return {key.replace("model__", ""): value for key, value in params.items()}


def run_regression_experiments() -> pd.DataFrame:
    train, val, features = load_training_inputs()
    rows = []
    predictions = pd.DataFrame(index=val.index)
    predictions["actual"] = val[TARGET_COL]
    parameters: Dict[str, object] = {}

    for name, model in model_searches(features).items():
        print(f"Fitting {name}...")
        model.fit(train[features], train[TARGET_COL])
        pred = np.clip(model.predict(val[features]), 0, None)
        predictions[name] = pred
        rows.append({"Split": "Validation", **evaluate_regression(val[TARGET_COL], pred, name)})
        parameters[name] = _best_params(model)

    print("Fitting PyTorch MLP...")
    mlp_pred, mlp_details = train_mlp(train, val, features, TARGET_COL, RANDOM_SEED)
    predictions["PyTorch MLP"] = mlp_pred
    rows.append({"Split": "Validation", **evaluate_regression(val[TARGET_COL], mlp_pred, "PyTorch MLP")})
    parameters["PyTorch MLP"] = mlp_details

    results = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    selected = results.iloc[0]["Model"]
    parameters["selected_model"] = selected

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(METRICS_DIR / "regression_validation.csv", index=False)
    predictions.to_parquet(PREDICTIONS_DIR / "regression_validation.parquet")
    (RESULTS_DIR / "regression_best_params.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8"
    )

    plot_data = results.melt(
        id_vars=["Model"], value_vars=["MAE", "RMSE"], var_name="Metric", value_name="Value"
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=plot_data, x="Model", y="Value", hue="Metric", ax=ax)
    ax.set_title("Regression model comparison on validation", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Error (vehicles/hour)")
    ax.tick_params(axis="x", rotation=18)
    ax.legend(title="")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "07_regression_validation.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(results.to_string(index=False))
    print(f"Selected regression model: {selected}")
    return results


if __name__ == "__main__":
    run_regression_experiments()
