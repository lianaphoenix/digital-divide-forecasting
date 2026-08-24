Machine learning pipeline for forecasting a composite Connectivity Index across countries, with short-term (1–3 year) predictive models and long-term (25-year, through 2049) structural projections. Built to support the paper "Forecasting the Digital Divide: A Comparative Machine Learning Study of ICT Access Trajectories," submitted to SIDe'26.

What this does?

Most digital-divide research relies on internet penetration as a single proxy for digital access. This project instead builds a composite Connectivity Index — combining internet penetration, mobile subscriptions, fixed broadband, and electricity access — and uses it to:

1. Forecast 1–3 years ahead using four models (naive persistence, linear regression, random forest, XGBoost), with statistical significance testing (paired t-test, Wilcoxon signed-rank, bootstrap CI) on the results.
2. Project 25 years ahead (through 2049) using per-country logistic (S-curve) growth models, since recursive multi-step ML forecasting over that horizon would require independently forecasting every input feature and compounds error at each step.

The primary country comparison is Kazakhstan and Uzbekistan against Germany, China, and Japan as reference cases, though the underlying panel covers ~199–213 countries and the short-term models are trained on the full dataset.

All data comes from the World Bank World Development Indicators (WDI), pulled live via the public API — no manual download needed. 

WDI data can also be browsed directly at data.worldbank.org or via the World Bank Data Catalog. The script fetches everything programmatically, so no dataset files need to be downloaded in advance — though a saved copy of the cleaned panel (digital_divide_panel.csv) is included in this repo for reproducibility, matching the version used in the paper's results.

Years with fewer than 100 reporting countries, and any year where any single component has thin coverage are automatically dropped before modeling, since sparse reporting years distort the per-year normalization used to build the index.

How to run?
No local setup required — this runs entirely in a free hosted notebook.

Open a new notebook at kaggle.com/code or colab.research.google.com.
Copy digital_divide_forecast.py into the notebook, either as one cell or split at the # CELL markers into separate cells.
Run top to bottom. No GPU needed; the full pipeline runs in under a minute.
Outputs (CSV files and PNG charts) save into the notebook's working directory.

Methodology summary
Composite index: internet penetration (40%), mobile subscriptions (20%, capped), fixed broadband (20%), electricity access (20%), each min-max normalized within its reporting year.
Short-term models: trained on a strict time-based train/validation/test split (not random) to avoid information leakage from future years.
Significance testing: paired t-test, Wilcoxon signed-rank test, and a 2,000-resample bootstrap confidence interval on the RMSE difference between linear regression and the naive persistence baseline.
Long-term projection: per-country logistic growth curve f(t) = L / (1 + e^(-k(t - t0))) fit on historical index values, following standard technology-diffusion modeling, with a linear-trend fallback for countries with insufficient data for a stable curve fit.

For projection and results, you may reach me at rakhimzhanovaliana@gmail.com

Citation

If you use this pipeline or its results, please cite:
@inproceedings{rakhimzhanova2026forecasting,
  author    = {Rakhimzhanova, Liana},
  title     = {Forecasting the Digital Divide: A Comparative Machine Learning Study of ICT Access Trajectories},
  booktitle = {Proceedings of SIDe'26: Secure Intelligent Digital Ecosystems},
  year      = {2026}
