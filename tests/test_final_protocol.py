import json
from pathlib import Path


def test_validation_outputs_do_not_contain_test_rows():
    root = Path(__file__).resolve().parents[1]
    for name in ["baseline_validation.csv", "regression_validation.csv", "classification_validation.csv"]:
        text = (root / "results" / "metrics" / name).read_text(encoding="utf-8")
        assert "Test" not in text


def test_threshold_manifest_records_train_only_source():
    root = Path(__file__).resolve().parents[1]
    thresholds = json.loads((root / "results" / "thresholds.json").read_text(encoding="utf-8"))
    assert thresholds["source"] == "training quantiles only"
