import numpy as np
import pandas as pd

from src.features.lags import build_lag_features


def test_lags_and_rolling_windows_do_not_use_current_or_future_targets():
    index = pd.date_range("2024-01-01", periods=220, freq="h")
    original = pd.DataFrame({"traffic_volume": np.arange(220, dtype=float)}, index=index)
    changed = original.copy()
    changed.loc[index[200], "traffic_volume"] = 999999.0

    original_features = build_lag_features(original)
    changed_features = build_lag_features(changed)

    pd.testing.assert_series_equal(
        original_features.loc[index[199]],
        changed_features.loc[index[199]],
        check_names=False,
    )
    assert original_features.loc[index[10], "rolling_mean_3"] == np.mean([7.0, 8.0, 9.0])
    assert original_features.loc[index[168], "lag_168"] == 0.0
