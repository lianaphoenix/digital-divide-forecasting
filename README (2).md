# Forecasting the Digital Divide: A Comparative Machine Learning Study of ICT Access Trajectories

Reproducibility repository for the study **“Forecasting the Digital Divide: A Comparative Machine Learning Study of ICT Access Trajectories.”**

The project constructs a relative 0–100 Connectivity Index from World Bank World Development Indicators (WDI), evaluates short-horizon forecasting models, performs robustness diagnostics, and generates long-horizon scenario trajectories for Kazakhstan and comparison countries.

## Main results from the final rerun

- Index construction period: **2002–2024** after the reporting-coverage filter.
- Modeling sample: **3,732 country-year observations from 196 countries**.
- Chronological split: training through **2017**, validation **2018–2021**, test **after 2021**.
- Common test set: **537 observations**.
- Best aggregate model among the valid models: **Linear Regression**.
- Naive persistence: RMSE **4.582**, MAE **2.371**.
- Linear Regression: RMSE **4.300**, MAE **2.234**.
- Random Forest: RMSE **4.353**, MAE **2.468**.
- XGBoost: RMSE **4.454**, MAE **2.523**.
- ARIMA and Holt/ETS produced **0 valid common test observations** in the final rerun and therefore are not ranked as successful alternatives.
- Linear regression reduced RMSE by about **6.1%** relative to persistence on the common test set.
- Kazakhstan 2049 scenario endpoints: conservative recent-linear **96.72**, full-history logistic **77.35**, accelerated pre-2018 logistic **77.70**.

These projected values are **relative Connectivity Index positions**, not percentages of people connected and not measures of cybersecurity capacity.

## Data and index construction

The repository uses these World Bank WDI indicators:

| Variable | WDI code |
|---|---|
| Internet use (% of population) | `IT.NET.USER.ZS` |
| Mobile subscriptions (per 100 people) | `IT.CEL.SETS.P2` |
| Fixed broadband subscriptions (per 100 people) | `IT.NET.BBND.P2` |
| Access to electricity (% of population) | `EG.ELC.ACCS.ZS` |
| GDP per capita | `NY.GDP.PCAP.CD` |
| Secondary-school enrollment | `SE.SEC.ENRR` |
| Urban population (% of total) | `SP.URB.TOTL.IN.ZS` |

The baseline Connectivity Index uses within-year min–max normalization and weights of **0.40 / 0.20 / 0.20 / 0.20** for Internet use, mobile subscriptions, fixed broadband, and electricity access. Mobile subscriptions are capped at **150 per 100 people** before normalization. Missing index components cause the available weights to be renormalized.

A year is retained for index construction only when every index component has at least **100 reporting countries**. Aggregate entities are removed.

## Repository structure

```text
Forecasting-Digital-Divide/
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
├── notebooks/
│   └── final_analysis.ipynb
├── src/
│   └── forecasting_digital_divide.py
├── data/
│   └── README.md
├── outputs/
│   ├── README.md
│   ├── model_results_summary.csv
│   ├── weight_sensitivity_summary.csv
│   └── kazakhstan_2049_scenarios.csv
├── docs/
│   └── reproducibility.md
├── paper/
│   └── Forecasting_Digital_Divide_Final_Rerun_Revised.tex
└── .github/
    └── workflows/
        └── reproducibility.yml
```

## Quick start

### 1. Create the environment

Using conda:

```bash
conda env create -f environment.yml
conda activate digital-divide-forecasting
```

Or using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the analysis

The notebook downloads the WDI indicators directly from the World Bank API. Internet access is therefore required.

```bash
jupyter notebook notebooks/final_analysis.ipynb
```

Run all cells from top to bottom. The corrected repository notebook places the long-horizon helper definitions before the scenario cell that uses them.

The equivalent executable Python script is:

```bash
python src/forecasting_digital_divide.py
```

The script and notebook write generated CSV tables and PNG figures to the **current working directory**. For an organized run, execute the script from `outputs/generated/` or change the working directory in your own workflow.

## Reproducibility notes

1. **Source data are not committed as raw WDI downloads.** The analysis retrieves public WDI data from the World Bank API using the indicator codes documented above. This keeps the repository small and avoids redistributing a separately downloaded copy of the database.
2. **The analysis is deterministic where model settings allow it.** The main random seed is `42`; random forest, XGBoost, permutation importance, and clustered bootstrap use this seed.
3. **The final rerun uses historical normalized index values as the forecasting target.** It does not invent future cross-country minima and maxima.
4. **Forward filling is past-only.** No backward filling or interpolation is used for the selected socioeconomic predictors.
5. **The long-horizon logistic model is a scenario generator, not a probability model or causal law.**
6. **Cybersecurity governance is an application context only.** The forecasting pipeline does not estimate governance capacity or a connectivity–cyber-risk causal effect.

## Important interpretation

The Connectivity Index is a relative comparative measure. A value of 77 or 97 does **not** mean that 77% or 97% of people are connected. It is a position on a 0–100 index generated from within-year cross-country normalization.

The study primarily captures first-level access/infrastructure. It does not directly measure digital skills, quality of use, downstream socioeconomic benefits, cybersecurity incidents, institutional coordination, incident-response capacity, or governance effectiveness.

## Outputs generated by the full pipeline

The final notebook generates, among others:

- `model_results_aggregate.csv`
- `arima_ets_selection.csv`
- `horizon_diagnostics.csv`
- `significance_comparison.csv`
- `xgboost_standard_importance.csv`
- `permutation_importance.csv`
- `feature_correlations.csv`
- `feature_vif.csv`
- `lag_ablation_results.csv`
- `weight_sensitivity.csv`
- `weight_trajectory_sensitivity.csv`
- `kazakhstan_scenarios.csv`
- `data_filtering_summary.csv`
- `excluded_year_component_coverage.csv`
- `ffill_audit.csv`
- PNG diagnostic figures.

## Citation

If you use this repository, please cite the associated paper. A machine-readable citation is provided in `CITATION.cff`.

## License

The repository code and documentation are released under the MIT License. The underlying World Bank data remain subject to the World Bank's applicable terms.
