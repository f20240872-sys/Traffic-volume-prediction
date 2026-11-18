from src.baselines import temporal_linear_pipeline


def test_linear_baseline_keeps_scaling_inside_the_pipeline():
    pipeline = temporal_linear_pipeline()
    assert list(pipeline.named_steps) == ["preprocess", "model"]
    names = [name for name, _, _ in pipeline.named_steps["preprocess"].transformers]
    assert names == ["calendar", "numeric"]
