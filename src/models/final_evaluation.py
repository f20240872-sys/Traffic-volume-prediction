"""One final refit on train+validation followed by test evaluation and analysis."""

from __future__ import annotations

import json
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
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
    THRESHOLDS_PATH,
)
from src.features.labels import apply_thresholds
from src.metrics import evaluate_classification, evaluate_regression
from src.models.ablation import FEATURE_SET_ORDER, MODEL_ORDER, fixed_model
from src.models.common import make_preprocessor
from src.models.mlp_torch import fit_predict_mlp_fixed


sns.set_theme(style="whitegrid", context="talk")


def fixed_classifier(name: str, features: list[str], params: Dict[str, object]) -> Pipeline:
    scale = name == "Logistic Regression"
    if name == "Logistic Regression":
        estimator = LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            **params,
        )
    elif name == "Decision Tree":
        estimator = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_SEED, **params)
    elif name == "Random Forest":
        estimator = RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1, **params
        )
    elif name == "XGBoost":
        estimator = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
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


def _feature_group(feature: str, sets: Dict[str, list[str]]) -> str:
    for group_name, label in [("temporal", "Temporal"), ("weather", "Weather"), ("lags", "Lags")]:
        if feature in sets[group_name]:
            return label
    return "Other"


def compute_validation_importance(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    params: Dict[str, object],
    groups: Dict[str, list[str]],
) -> pd.DataFrame:
    model = fixed_model("Random Forest", features, params)
    model.fit(train[features], train[TARGET_COL])
    importance = permutation_importance(
        model,
        val[features],
        val[TARGET_COL],
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    table = pd.DataFrame(
        {
            "Feature": features,
            "Permutation_MAE_Increase": importance.importances_mean,
            "Permutation_Std": importance.importances_std,
        }
    )
    encoded_names = model.named_steps["preprocess"].get_feature_names_out()
    encoded_importance = model.named_steps["model"].feature_importances_
    tree_importance = {feature: 0.0 for feature in features}
    for encoded_name, value in zip(encoded_names, encoded_importance):
        cleaned = encoded_name.split("__", 1)[-1]
        match = next(
            (
                feature
                for feature in features
                if cleaned == feature or cleaned.startswith(f"{feature}_")
            ),
            None,
        )
        if match is not None:
            tree_importance[match] += float(value)
    table["Tree_Importance"] = table["Feature"].map(tree_importance)
    table["Group"] = table["Feature"].map(lambda value: _feature_group(value, groups))
    return table.sort_values("Permutation_MAE_Increase", ascending=False).reset_index(drop=True)


def _plot_importance(importance: pd.DataFrame) -> None:
    top = importance.head(15).sort_values("Permutation_MAE_Increase")
    grouped = (
        importance.assign(Positive=importance["Permutation_MAE_Increase"].clip(lower=0))
        .groupby("Group", as_index=False)["Positive"]
        .sum()
    )
    grouped["Share"] = grouped["Positive"] / grouped["Positive"].sum()
    grouped.to_csv(TABLES_DIR / "grouped_feature_importance.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(data=top, x="Permutation_MAE_Increase", y="Feature", hue="Group", dodge=False, ax=axes[0])
    axes[0].set_title("Top validation permutation importance", fontweight="bold")
    axes[0].set_xlabel("Increase in MAE after permutation")
    axes[0].set_ylabel("")
    axes[0].legend(title="")
    sns.barplot(data=grouped, x="Group", y="Share", color="#2c7fb8", ax=axes[1])
    axes[1].set_title("Importance by feature group", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Share of positive importance")
    axes[1].set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "10_feature_importance.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_regression_diagnostics(test: pd.DataFrame, predictions: pd.DataFrame, best_column: str) -> None:
    analysis = test[["hour", "weather_main", TARGET_COL]].copy()
    analysis["prediction"] = predictions[best_column]
    analysis["residual"] = analysis[TARGET_COL] - analysis["prediction"]
    analysis["absolute_error"] = analysis["residual"].abs()

    by_hour = analysis.groupby("hour", as_index=False).agg(
        MAE=("absolute_error", "mean"), Mean_Residual=("residual", "mean"), Hours=("residual", "size")
    )
    by_weather = analysis.groupby("weather_main", as_index=False).agg(
        MAE=("absolute_error", "mean"), Mean_Residual=("residual", "mean"), Hours=("residual", "size")
    ).sort_values("Hours", ascending=False)
    by_hour.to_csv(TABLES_DIR / "residuals_by_hour.csv", index=False)
    by_weather.to_csv(TABLES_DIR / "residuals_by_weather.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.lineplot(data=by_hour, x="hour", y="MAE", marker="o", ax=axes[0])
    axes[0].set_title("Test error by hour", fontweight="bold")
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("MAE (vehicles/hour)")
    weather_plot = by_weather.head(8).sort_values("MAE")
    sns.barplot(data=weather_plot, x="MAE", y="weather_main", color="#6baed6", ax=axes[1])
    axes[1].set_title("Test error by weather condition", fontweight="bold")
    axes[1].set_xlabel("MAE (vehicles/hour)")
    axes[1].set_ylabel("")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "11_residual_analysis.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    sample = analysis.iloc[: 24 * 7]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(sample.index, sample[TARGET_COL], label="Actual", linewidth=2)
    ax.plot(sample.index, sample["prediction"], label="Predicted", linewidth=2)
    ax.set_title("Predicted and actual traffic for the first eligible test week", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Traffic volume (vehicles/hour)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "12_predicted_vs_actual.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def run_final_evaluation() -> None:
    train = pd.read_parquet(DATA_PROCESSED_DIR / "features_train.parquet")
    val = pd.read_parquet(DATA_PROCESSED_DIR / "features_val.parquet")
    test = pd.read_parquet(DATA_PROCESSED_DIR / "features_test.parquet")
    train_val = pd.concat([train, val]).sort_index()
    metadata = json.loads((RESULTS_DIR / "feature_sets.json").read_text(encoding="utf-8"))
    feature_sets = metadata["feature_sets"]
    all_features = feature_sets["T+W+L"]
    regression_params = json.loads((RESULTS_DIR / "regression_best_params.json").read_text(encoding="utf-8"))
    classification_params = json.loads((RESULTS_DIR / "classification_best_params.json").read_text(encoding="utf-8"))
    thresholds = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))

    importance = compute_validation_importance(
        train,
        val,
        feature_sets["T+L"],
        regression_params["Random Forest"],
        {key: metadata["feature_sets"][code] for key, code in [("temporal", "T"), ("weather", "W"), ("lags", "L")]},
    )
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(TABLES_DIR / "feature_importance.csv", index=False)
    _plot_importance(importance)

    regression_rows = []
    regression_predictions = pd.DataFrame(index=test.index)
    regression_predictions["actual"] = test[TARGET_COL]
    regression_predictions["Same hour last week"] = test["lag_168"]
    regression_rows.append(
        {"Split": "Test", **evaluate_regression(test[TARGET_COL], test["lag_168"], "Same hour last week")}
    )
    for name in MODEL_ORDER:
        print(f"Final regression refit: {name}")
        model = fixed_model(name, all_features, regression_params.get(name, {}))
        model.fit(train_val[all_features], train_val[TARGET_COL])
        pred = model.predict(test[all_features]).clip(min=0)
        regression_predictions[name] = pred
        regression_rows.append({"Split": "Test", **evaluate_regression(test[TARGET_COL], pred, name)})

    mlp_epochs = int(regression_params["PyTorch MLP"]["epochs_run"])
    mlp_pred = fit_predict_mlp_fixed(train_val, test, all_features, TARGET_COL, mlp_epochs, RANDOM_SEED)
    regression_predictions["PyTorch MLP"] = mlp_pred
    regression_rows.append(
        {"Split": "Test", **evaluate_regression(test[TARGET_COL], mlp_pred, "PyTorch MLP")}
    )
    regression_results = pd.DataFrame(regression_rows).sort_values("MAE").reset_index(drop=True)

    ablation_rows = []
    for set_name in FEATURE_SET_ORDER:
        features = feature_sets[set_name]
        for name in MODEL_ORDER:
            model = fixed_model(name, features, regression_params.get(name, {}))
            model.fit(train_val[features], train_val[TARGET_COL])
            pred = model.predict(test[features]).clip(min=0)
            ablation_rows.append(
                {"Feature_Set": set_name, **evaluate_regression(test[TARGET_COL], pred, name)}
            )
    pd.DataFrame(ablation_rows).to_csv(METRICS_DIR / "ablation_test.csv", index=False)

    best_ablation = json.loads((RESULTS_DIR / "ablation_best.json").read_text(encoding="utf-8"))
    best_model = fixed_model(
        best_ablation["model"],
        feature_sets[best_ablation["feature_set"]],
        regression_params[best_ablation["model"]],
    )
    best_model.fit(train_val[feature_sets[best_ablation["feature_set"]]], train_val[TARGET_COL])
    best_prediction = best_model.predict(test[feature_sets[best_ablation["feature_set"]]]).clip(min=0)
    regression_predictions["Selected RF (T+L)"] = best_prediction
    selected_metrics = evaluate_regression(
        test[TARGET_COL], best_prediction, "Random Forest (T+L selected)"
    )
    regression_results = pd.concat(
        [regression_results, pd.DataFrame([{"Split": "Test", **selected_metrics}])],
        ignore_index=True,
    ).sort_values("MAE").reset_index(drop=True)
    regression_results.to_csv(METRICS_DIR / "regression_test.csv", index=False)
    regression_predictions.to_parquet(PREDICTIONS_DIR / "regression_test.parquet")
    _plot_regression_diagnostics(test, regression_predictions, "Selected RF (T+L)")

    y_train_val = apply_thresholds(train_val[TARGET_COL], thresholds)
    y_test = apply_thresholds(test[TARGET_COL], thresholds)
    classification_rows = []
    classification_predictions = pd.DataFrame(index=test.index)
    classification_predictions["actual"] = y_test
    fitted_classifiers = {}

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train_val[all_features], y_train_val)
    dummy_pred = dummy.predict(test[all_features])
    classification_rows.append(
        {"Split": "Test", **evaluate_classification(y_test, dummy_pred, "Majority class")}
    )
    for name in ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]:
        print(f"Final classifier refit: {name}")
        model = fixed_classifier(name, all_features, classification_params[name])
        model.fit(train_val[all_features], y_train_val)
        pred = model.predict(test[all_features]).astype(int)
        classification_predictions[name] = pred
        classification_rows.append(
            {"Split": "Test", **evaluate_classification(y_test, pred, name)}
        )
        fitted_classifiers[name] = model

    classification_results = pd.DataFrame(classification_rows).sort_values(
        "Macro_F1", ascending=False
    ).reset_index(drop=True)
    classification_results.to_csv(METRICS_DIR / "classification_test.csv", index=False)
    classification_predictions.to_parquet(PREDICTIONS_DIR / "classification_test.parquet")

    class_distribution_path = TABLES_DIR / "class_distribution.csv"
    distribution = pd.read_csv(class_distribution_path)
    test_distribution = pd.DataFrame(
        [
            {
                "Split": "Test",
                "Class": class_name,
                "Count": int((y_test == class_id).sum()),
                "Share": round(float((y_test == class_id).mean()), 4),
            }
            for class_id, class_name in enumerate(CONGESTION_LABELS)
        ]
    )
    pd.concat([distribution[distribution["Split"] != "Test"], test_distribution], ignore_index=True).to_csv(
        class_distribution_path, index=False
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, name in zip(axes.flat, fitted_classifiers):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            classification_predictions[name],
            display_labels=CONGESTION_LABELS,
            cmap="Blues",
            colorbar=False,
            ax=ax,
        )
        ax.set_title(name)
    fig.suptitle("Final test confusion matrices", fontsize=18, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "09_confusion_matrices.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "test_rows": len(test),
        "test_start": str(test.index.min()),
        "test_end": str(test.index.max()),
        "regression_selection": best_ablation,
        "classification_selection": classification_params["selected_model"],
    }
    (RESULTS_DIR / "final_test_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("\nRegression test results")
    print(regression_results.to_string(index=False))
    print("\nClassification test results")
    print(classification_results.to_string(index=False))


if __name__ == "__main__":
    run_final_evaluation()
