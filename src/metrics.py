from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: Optional[str] = None,
) -> Dict[str, float]:
    """Compute MAE, RMSE, and R2 for continuous traffic volume predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    results = {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
    }
    if model_name is not None:
        results = {"Model": model_name, **results}
    return results


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: Optional[str] = None,
) -> Dict[str, float]:
    """Compute Macro-F1 and Balanced Accuracy for 3-class congestion levels."""
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    results = {
        "Macro_F1": round(macro_f1, 4),
        "Balanced_Accuracy": round(bal_acc, 4),
    }
    if model_name is not None:
        results = {"Model": model_name, **results}
    return results


def results_to_dataframe(results: List[Dict[str, float]]) -> pd.DataFrame:
    """Convert a list of metric dictionaries into a clean comparison DataFrame."""
    return pd.DataFrame(results)