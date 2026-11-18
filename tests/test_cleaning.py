import numpy as np
import pandas as pd

from src.clean import apply_cleaning_rules, build_hourly_grid, collapse_weather_rows


def test_cleaning_never_fills_the_target():
    index = pd.date_range("2024-01-01", periods=3, freq="h", name="date_time")
    grid = pd.DataFrame(
        {
            "traffic_volume": [100.0, np.nan, 120.0],
            "temp": [273.0, np.nan, 275.0],
            "rain_1h": [0.0, np.nan, 0.0],
            "snow_1h": [0.0, np.nan, 0.0],
            "clouds_all": [20.0, np.nan, 30.0],
            "holiday": ["New Years Day", np.nan, np.nan],
            "weather_main": ["Clear", np.nan, "Clouds"],
            "weather_description": ["clear", np.nan, "cloudy"],
        },
        index=index,
    )
    clean, audit = apply_cleaning_rules(grid)
    assert pd.isna(clean.loc[index[1], "traffic_volume"])
    assert audit["target_values_imputed"] == 0
    assert clean["is_holiday"].tolist() == [1, 1, 1]


def test_duplicate_weather_rows_collapse_to_one_hour():
    raw = pd.DataFrame(
        {
            "date_time": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "traffic_volume": [100, 100],
            "temp": [270.0, 272.0],
            "rain_1h": [0.0, 0.0],
            "snow_1h": [0.0, 0.0],
            "clouds_all": [10, 20],
            "holiday": ["None", "None"],
            "weather_main": ["Clear", "Clouds"],
            "weather_description": ["clear", "scattered clouds"],
        }
    )
    hourly, duplicate_rows = collapse_weather_rows(raw)
    grid, missing_hours = build_hourly_grid(hourly)
    assert duplicate_rows == 1
    assert len(grid) == 1
    assert missing_hours == 0
    assert grid.iloc[0]["temp"] == 271.0
