"""Fixed-parameter feature ablation on the validation set."""

from __future__ import annotations

import json
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.config import (
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    TARGET_COL,
)
from src.metrics import evaluate_regression
from src.models.common import make_preprocessor


sns.set_theme(style="whitegrid", context="talk")


MODEL_ORDER = ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"]
FEATURE_SET_ORDER = ["T", "W", "L", "T+W", "T+L", "T+W+L"]


def fixed_model(name: str, features: list[str], params: Dict[str, object]) -> Pipeline:
    scale = name == "Linear Regression"
    if name == "Linear Regression":
        estimator = LinearRegression()
    elif name == "Decision Tree":
        estimator = DecisionTreeRegressor(random_state=RANDOM_SEED, **params)
    elif name == "Random Forest":
        estimator = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1, **params)
    elif name == "XGBoost":
        estimator = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            tree_method="hist",
            **params,
        )
    else:
        raise KeyError(name)
    return Pipeline(
        [
            ("preprocess", make_preprocessor(features, scale_numeric=scale)),
            ("model", estimator),
        ]
    )


def run_ablation() -> pd.DataFrame:
    train = pd.read_parquet(DATA_PROCESSED_DIR / "features_train.parquet")
    val = pd.read_parquet(DATA_PROCESSED_DIR / "features_val.parquet")
    metadata = json.loads((RESULTS_DIR / "feature_sets.json").read_text(encoding="utf-8"))
    sets = metadata["feature_sets"]
    tuned = json.loads((RESULTS_DIR / "regression_best_params.json").read_text(encoding="utf-8"))

    rows = []
    for set_name in FEATURE_SET_ORDER:
        features = sets[set_name]
        for model_name in MODEL_ORDER:
            print(f"Ablation: {set_name} with {model_name}")
            model = fixed_model(model_name, features, tuned.get(model_name, {}))
            model.fit(train[features], train[TARGET_COL])
            prediction = model.predict(val[features]).clip(min=0)
            rows.append(
                {
                    "Feature_Set": set_name,
                    **evaluate_regression(val[TARGET_COL], prediction, model_name),
                }
            )

    results = pd.DataFrame(rows)
    results["Feature_Set"] = pd.Categorical(
        results["Feature_Set"], categories=FEATURE_SET_ORDER, ordered=True
    )
    results = results.sort_values(["Feature_Set", "Model"]).reset_index(drop=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(METRICS_DIR / "ablation_validation.csv", index=False)
    best = results.loc[results["MAE"].idxmin()]
    (RESULTS_DIR / "ablation_best.json").write_text(
        json.dumps(
            {
                "model": str(best["Model"]),
                "feature_set": str(best["Feature_Set"]),
                "validation_mae": float(best["MAE"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=results,
        x="Feature_Set",
        y="MAE",
        hue="Model",
        hue_order=MODEL_ORDER,
        ax=ax,
    )
    ax.set_title("Feature ablation with fixed model settings", fontweight="bold")
    ax.set_xlabel("Feature set")
    ax.set_ylabel("Validation MAE (vehicles/hour)")
    ax.legend(title="", ncol=2)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "08_ablation.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    run_ablation()
