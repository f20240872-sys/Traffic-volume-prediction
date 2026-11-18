# Traffic volume prediction and congestion analysis

This project predicts hourly traffic on westbound I-94 near Minneapolis and converts the forecast into three congestion levels. It uses the UCI Metro Interstate Traffic Volume dataset, a chronological train/validation/test design, and a shared set of rows for every feature comparison.

## Main results

The final test period runs from 6 November 2017 to 30 September 2018 and contains 5,587 eligible hourly observations.

| Regression model | MAE | RMSE | R² |
|---|---:|---:|---:|
| XGBoost | 158.57 | 252.60 | 0.9836 |
| Random Forest, temporal + lag features | 158.58 | 261.54 | 0.9824 |
| Random Forest, all features | 166.30 | 274.76 | 0.9806 |
| PyTorch MLP | 196.54 | 327.18 | 0.9725 |
| Same hour last week | 360.39 | 703.19 | 0.8728 |

The Random Forest congestion classifier reached a test Macro-F1 of 0.9402 and balanced accuracy of 0.9399. Congestion thresholds were fixed from the training target at 2,228 and 4,510 vehicles per hour.

## Method

The raw file contains 48,204 weather rows but only 40,575 unique timestamps. The cleaning stage collapses duplicate weather descriptions, creates a complete hourly grid, replaces ten zero-Kelvin readings and one impossible rain value, and carries weather observations forward for no more than six hours. It never fills the traffic target.

The hourly grid is split chronologically:

| Split | Date range | Grid rows |
|---|---|---:|
| Train | 2 Oct 2012 to 13 Dec 2016 | 36,785 |
| Validation | 13 Dec 2016 to 6 Nov 2017 | 7,883 |
| Test | 6 Nov 2017 to 30 Sep 2018 | 7,883 |

Modeling uses only rows with a target and a complete lag history, leaving 5,621 training, 5,701 validation, and 5,587 test rows. Features are grouped as:

- **Temporal:** calendar fields, cyclical encodings, weekend, holiday, and rush-hour flags.
- **Weather:** temperature, precipitation, cloud cover, weather condition, and rush-hour interactions.
- **Lags:** 1, 2, 24, and 168-hour lags plus shifted rolling means and standard deviations.

The ablation study compares six feature sets across linear regression, decision tree, random forest, and XGBoost without retuning. Lags account for about 62% of positive permutation importance. Weather features add little once recent traffic and the calendar pattern are known.

## Leakage safeguards

- Splits follow time order. Cross-validation uses `TimeSeriesSplit` with a 24-hour gap.
- Rolling features call `shift(1)` before every window calculation.
- Imputers, encoders, and scalers live inside model pipelines and fit on training folds only.
- Congestion cutoffs come from the training target and are reused unchanged.
- All feature sets use the same eligible timestamps.
- Model and feature choices use validation results. The dedicated final stage refits on train plus validation and then evaluates test.

The first Day 3 commit in the repository history contained exploratory test metrics. The current pipeline removes that access from baseline and tuning stages; the final test artifacts now live only in `results/metrics/*_test.csv` and `results/predictions/*_test.parquet`.

## Reproducing the project

Python 3.8 or newer is supported. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
./run_all.sh
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
.\run_all.ps1
```

The complete run cleans the data, regenerates figures and feature files, tunes on validation, performs the fixed ablation and classification studies, runs the final test stage, and executes the tests. Expect the CPU run to take several minutes.

## Repository layout

```text
data/raw/                 source CSV
data/processed/           cleaned hourly data and feature splits
src/features/             temporal, weather, lag, and label builders
src/models/               regression, ablation, classification, final evaluation
scripts/                  numbered entry points for each project stage
results/metrics/          validation and final test tables
results/predictions/      timestamped predictions
reports/figures/          presentation-ready figures
reports/tables/           data quality, importance, and residual summaries
reports/slides/           final presentation
tests/                    leakage and pipeline tests
```

## Limitations

The source series has a ten-month gap in 2014–2015 and many shorter outages. Requiring a complete 168-hour history reduces the usable training sample. This is a single traffic counter, so performance should not be assumed to transfer to other roads. Weather categories are coarse, and the data has no incident, construction, or event information.
