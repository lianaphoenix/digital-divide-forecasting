# Forecasting the Digital Divide

Reproducibility repository for **“Forecasting the Digital Divide: A Comparative Machine Learning Study of ICT Access Trajectories.”** It builds a relative 0–100 Connectivity Index from World Bank data, compares forecasting models, and explores long-term scenarios for Kazakhstan and other countries.

## Results

The final rerun covers **196 countries and 3,732 country-year observations** (2002–2024). On a common test set of 537 observations, linear regression had the lowest RMSE among the valid models:

| Model | RMSE | MAE |
| --- | ---: | ---: |
| Persistence baseline | 4.582 | 2.371 |
| Linear regression | **4.300** | **2.234** |
| Random forest | 4.353 | 2.468 |
| XGBoost | 4.454 | 2.523 |

Training ended in 2017, validation covered 2018–2021, and testing used later years. ARIMA and Holt/ETS had no valid observations on the common test set, so they are not ranked here. The Kazakhstan 2049 scenario endpoints range from **77.35 to 96.72**. These are positions on a relative index, **not percentages of people connected** or forecasts of cybersecurity capacity.

## Run

Install dependencies with `pip install -r requirements.txt`, or create the Conda environment with `conda env create -f environment.yml` and activate it with `conda activate digital-divide-forecasting`. Python 3.11 is recommended.

From the repository root, run either:

```bash
python src/forecasting_digital_divide.py
# or
jupyter notebook notebooks/final_analysis.ipynb
```

The analysis downloads WDI data at runtime, so it needs internet access. Generated CSV and PNG files are written to the current working directory. To keep the root clean when running the script:

```bash
mkdir -p outputs/generated
cd outputs/generated
python ../../src/forecasting_digital_divide.py
```

## Repository contents

- `src/` — executable analysis script.
- `notebooks/` — equivalent Jupyter analysis.
- `data/` — indicator list and data notes.
- `outputs/` — selected results and processed panel from the final rerun.
- `docs/reproducibility.md` — methodology and reproduction notes.
- `paper/` — LaTeX manuscript and bibliography.

The index combines internet use, mobile subscriptions, fixed broadband, and electricity access using within-year normalization. It measures relative access and infrastructure; it does not directly measure digital skills or quality of use. See [data notes](data/README.md) and [reproducibility notes](docs/reproducibility.md) for details.

## Citation and license

Citation metadata is in [CITATION.cff](CITATION.cff). Code and documentation are released under the [MIT License](LICENSE); World Bank data are subject to their own terms.
