# Outputs

This directory contains the compact results and processed panel from the final rerun. The complete set of generated diagnostics is produced by running `notebooks/final_analysis.ipynb` or `src/forecasting_digital_divide.py` from the repository root.

The full pipeline writes model comparisons, ARIMA/ETS selection diagnostics, horizon diagnostics, clustered bootstrap results, feature-importance diagnostics, VIF/correlation tables, lag ablation, weight sensitivity, country trajectory sensitivity, Kazakhstan scenarios, filtering summaries, and figures. Run it from `outputs/generated/` to keep those artifacts separate from the committed summaries.
