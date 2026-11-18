import pandas as pd

from src.features.labels import apply_thresholds, fit_thresholds


def test_thresholds_depend_only_on_the_training_target():
    train = pd.Series([10, 20, 30, 40, 50, 60])
    validation = pd.Series([1, 1000])
    thresholds = fit_thresholds(train)
    labels_before = apply_thresholds(validation, thresholds)
    validation.iloc[:] = [5000, 9000]
    labels_after = apply_thresholds(validation, thresholds)
    assert thresholds == fit_thresholds(train)
    assert labels_before.tolist() != labels_after.tolist()
    assert set(thresholds) == {"low_to_moderate", "moderate_to_high"}
