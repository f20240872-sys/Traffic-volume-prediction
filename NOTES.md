# Project notes

- Raw holiday labels appear only at midnight. Cleaning expands each named holiday across its calendar date.
- The long 2014–2015 data outage remains explicit. Traffic volume is never interpolated across it.
- The primary regression choice from validation was Random Forest with temporal and lag features. XGBoost on the full set produced a nearly identical final test MAE.
- The final classifier selection by validation Macro-F1 was XGBoost. Random Forest was slightly better on the held-out test set.
- The test set should not drive further model or feature changes. Any later experiments belong in a new evaluation cycle.
