from src.models.ablation import FEATURE_SET_ORDER, MODEL_ORDER


def test_ablation_covers_six_sets_and_four_models():
    assert FEATURE_SET_ORDER == ["T", "W", "L", "T+W", "T+L", "T+W+L"]
    assert MODEL_ORDER == ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"]
    assert len(FEATURE_SET_ORDER) * len(MODEL_ORDER) == 24
