from sklearn.model_selection import TimeSeriesSplit

from src.models.common import make_preprocessor, time_series_cv


def test_cross_validation_is_time_ordered_with_a_gap():
    cv = time_series_cv()
    assert isinstance(cv, TimeSeriesSplit)
    assert cv.gap == 24
    assert cv.n_splits == 4


def test_weather_category_is_encoded_inside_the_pipeline():
    processor = make_preprocessor(["hour", "weather_main"], scale_numeric=True)
    names = [name for name, _, _ in processor.transformers]
    assert names == ["numeric", "categorical"]
