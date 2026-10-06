# Reproducibility protocol

## Environment

Use either `environment.yml` with conda or `requirements.txt` with Python 3.11.

## Execution order

The final notebook is arranged so that all helper functions are defined before they are called. In particular, the logistic helper (`logistic`, `fit_logistic`, `fit_linear_recent`, and `project_fit`) is defined before the long-horizon scenario analysis.

## External data

At runtime, the notebook queries the World Bank WDI API. A working internet connection is required. The analysis should be rerun in a clean environment rather than relying on stale notebook state.

## Randomness

`RANDOM_STATE = 42`. This seed is applied to the random forest, XGBoost, permutation importance, and clustered bootstrap components.

## Temporal design

The index is constructed for 2002–2024 after the 100-country reporting threshold. Forecasting uses a chronological train/validation/test design, with training through 2017, validation from 2018–2021, and the test period after 2021.

## Interpretation of missing ARIMA/ETS results

The final rerun found zero valid common test observations for both ARIMA and Holt/ETS. This is a coverage result, not evidence that those methods are intrinsically poor forecasting methods. They should therefore not be ranked against the supervised models using aggregate error metrics in this dataset.

## Interpretation of long-horizon scenarios

The 2049 Kazakhstan values are conditional scenario outputs. The index is relative and bounded to a 0–100 scale; these values are not percentages and are not forecasts of cybersecurity capacity.
