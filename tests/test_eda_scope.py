import pandas as pd

from src.eda_weather import training_observations


def test_eda_uses_only_the_training_period():
    index = pd.date_range("2024-01-01", periods=20, freq="h")
    frame = pd.DataFrame({"traffic_volume": range(20)}, index=index)
    train = training_observations(frame)
    assert len(train) == 14
    assert train.index.max() < index[14]
