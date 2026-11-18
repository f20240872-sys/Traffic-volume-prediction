import numpy as np
import pandas as pd

from src.features.temporal import build_temporal_features
from src.features.weather import build_weather_features


def sample_frame():
    index = pd.date_range("2024-01-01 06:00", periods=2, freq="h")
    return pd.DataFrame(
        {
            "temp": [273.15, 283.15],
            "rain_1h": [0.0, 2.0],
            "snow_1h": [0.0, 0.0],
            "clouds_all": [20, 90],
            "weather_main": ["Clear", "Rain"],
            "is_holiday": [0, 1],
            "weather_observed": [1, 1],
        },
        index=index,
    )


def test_temporal_cycles_and_rush_hour_are_deterministic():
    temporal = build_temporal_features(sample_frame())
    assert temporal["hour"].tolist() == [6, 7]
    assert temporal["rush_hour"].tolist() == [1, 1]
    assert np.all(np.abs(temporal["hour_sin"]) <= 1)


def test_weather_features_use_celsius_and_interactions():
    frame = sample_frame()
    temporal = build_temporal_features(frame)
    weather = build_weather_features(frame, temporal)
    assert np.allclose(weather["temp_c"], [0.0, 10.0])
    assert weather["has_rain"].tolist() == [0, 1]
    assert weather["rain_during_rush"].tolist() == [0, 1]
