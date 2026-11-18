"""Congestion classification with train-only thresholds and validation tuning."""

from __future__ import annotations

import json
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import (
    CONGESTION_LABELS,
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    TABLES_DIR,
    TARGET_COL,
)
from src.features.labels import apply_thresholds, fit_thresholds, save_thresholds
from src.metrics import evaluate_classification
from src.models.common import make_preprocessor, time_series_cv


sns.set_theme(style="whitegrid", context="talk")


def classifier_searches(features: list[str]) -> Dict[str, object]:
    cv = time_series_cv()
    logistic = RandomizedSearchCV(
        Pipeline(
            [
                ("preprocess", make_preprocessor(features, scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1500,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        {"model__C": [0.05, 0.1, 0.5, 1.0, 3.0, 10.0]},
        n_iter=6,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    tree = RandomizedSearchCV(
        Pipeline(
            [
                ("preprocess", make_preprocessor(features, scale_numeric=False)),
                ("model", DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_SEED)),
            ]
        ),
        {
            "model__max_depth": [5, 8, 12, None],
            "model__min_samples_leaf": [5, 15, 30, 60],
            "model__max_features": [None, "sqrt", 0.8],
        },
        n_iter=8,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    forest = RandomizedSearchCV(
        Pipeline(
            [
                ("preprocess", make_preprocessor(features, scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        {
            "model__n_estimators": [180, 280, 380],
            "model__max_depth": [10, 16, None],
            "model__min_samples_leaf": [1, 3, 8],
            "model__max_features": ["sqrt", 0.7],
        },
        n_iter=7,
        scoring="f1_macro",
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
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=3,
                        eval_metric="mlogloss",
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
        n_iter=7,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    return {
        "Logistic Regression": logistic,
        "Decision Tree": tree,
        "Random Forest": forest,
        "XGBoost": xgboost,
    }


def _plot_thresholds(train_target: pd.Series, thresholds: Dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.histplot(train_target, bins=45, color="#2c7fb8", ax=ax)
    ax.axvline(thresholds["low_to_moderate"], color="#fdae61", linewidth=3, label="Low / Moderate")
    ax.axvline(thresholds["moderate_to_high"], color="#d7191c", linewidth=3, label="Moderate / High")
    ax.set_title("Training traffic distribution and congestion thresholds", fontweight="bold")
    ax.set_xlabel("Traffic volume (vehicles/hour)")
    ax.set_ylabel("Observed hours")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "04_volume_distribution.png", dpi=240)
    plt.close(fig)


def run_classification_experiments() -> pd.DataFrame:
    train = pd.read_parquet(DATA_PROCESSED_DIR / "features_train.parquet")
    val = pd.read_parquet(DATA_PROCESSED_DIR / "features_val.parquet")
    metadata = json.loads((RESULTS_DIR / "feature_sets.json").read_text(encoding="utf-8"))
    features = metadata["feature_sets"]["T+W+L"]

    thresholds = fit_thresholds(train[TARGET_COL])
    save_thresholds(thresholds)
    y_train = apply_thresholds(train[TARGET_COL], thresholds)
    y_val = apply_thresholds(val[TARGET_COL], thresholds)

    distribution = []
    for split_name, labels in [("Train", y_train), ("Validation", y_val)]:
        counts = labels.value_counts().sort_index()
        for class_id, class_name in enumerate(CONGESTION_LABELS):
            distribution.append(
                {
                    "Split": split_name,
                    "Class": class_name,
                    "Count": int(counts.get(class_id, 0)),
                    "Share": round(float((labels == class_id).mean()), 4),
                }
            )
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(distribution).to_csv(TABLES_DIR / "class_distribution.csv", index=False)

    rows = []
    predictions = pd.DataFrame(index=val.index)
    predictions["actual"] = y_val
    best_params: Dict[str, object] = {}
    fitted = {}

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train[features], y_train)
    dummy_pred = dummy.predict(val[features])
    rows.append({"Split": "Validation", **evaluate_classification(y_val, dummy_pred, "Majority class")})

    for name, model in classifier_searches(features).items():
        print(f"Fitting {name} classifier...")
        model.fit(train[features], y_train)
        pred = model.predict(val[features]).astype(int)
        predictions[name] = pred
        rows.append({"Split": "Validation", **evaluate_classification(y_val, pred, name)})
        best_params[name] = {
            key.replace("model__", ""): value for key, value in model.best_params_.items()
        }
        fitted[name] = model

    results = pd.DataFrame(rows).sort_values("Macro_F1", ascending=False).reset_index(drop=True)
    best_params["selected_model"] = results.iloc[0]["Model"]
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(METRICS_DIR / "classification_validation.csv", index=False)
    predictions.to_parquet(PREDICTIONS_DIR / "classification_validation.parquet")
    (RESULTS_DIR / "classification_best_params.json").write_text(
        json.dumps(best_params, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, (name, model) in zip(axes.flat, fitted.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_val,
            predictions[name],
            display_labels=CONGESTION_LABELS,
            cmap="Blues",
            colorbar=False,
            ax=ax,
        )
        ax.set_title(name)
    fig.suptitle("Validation confusion matrices", fontsize=18, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "09_confusion_matrices.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    _plot_thresholds(train[TARGET_COL], thresholds)

    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    run_classification_experiments()
