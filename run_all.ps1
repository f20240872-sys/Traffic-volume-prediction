$ErrorActionPreference = "Stop"

python -m src.clean
python -m src.split
python -m src.eda_temporal
python -m src.eda_weather
python -m src.features.build
python -m src.baselines
python -m src.models.regression
python -m src.models.ablation
python -m src.models.classification
python -m src.models.final_evaluation
python -m pytest -q
