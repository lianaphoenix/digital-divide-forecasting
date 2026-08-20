import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import curve_fit

pd.set_option("display.max_columns", None)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

INDICATORS = {
    "internet_pct": "IT.NET.USER.ZS",
    "mobile_subs": "IT.CEL.SETS.P2",
    "fixed_broadband": "IT.NET.BBND.P2",
    "electricity_access": "EG.ELC.ACCS.ZS",
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "secondary_enroll": "SE.SEC.ENRR",
    "urban_pop": "SP.URB.TOTL.IN.ZS",
}

def fetch_indicator(indicator_code, per_page=20000):
    url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator_code}"
    params = {"format": "json", "per_page": per_page}
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()[1]
    rows = []
    for rec in data:
        if rec["value"] is not None:
            rows.append({
                "country": rec["country"]["value"],
                "country_code": rec["countryiso3code"],
                "year": int(rec["date"]),
                "value": rec["value"],
            })
    return pd.DataFrame(rows)

print("Fetching WDI indicators...")
frames = {}
for name, code in INDICATORS.items():
    df = fetch_indicator(code)
    df = df.rename(columns={"value": name})
    frames[name] = df
    print(f"  {name}: {len(df)} rows, most recent year = {df['year'].max()}")

panel = frames["internet_pct"][["country", "country_code", "year"]].copy()
for name, df in frames.items():
    panel = panel.merge(
        df[["country_code", "year", name]],
        on=["country_code", "year"],
        how="left",
    )
print(panel.shape)

AGGREGATE_KEYWORDS = [
    "World", "income", "IDA", "IBRD", "OECD", "Euro area", "Arab World",
    "Fragile", "Least developed", "Africa Eastern", "Africa Western",
    "Central Europe", "East Asia", "Europe &", "Latin America &",
    "Middle East &", "North America", "South Asia", "Sub-Saharan",
    "Small states", "Heavily indebted", "Pacific island", "European Union",
]
mask_aggregate = panel["country"].str.contains("|".join(AGGREGATE_KEYWORDS), case=False, na=False)
panel = panel[~mask_aggregate].copy()
print(f"After removing aggregates: {panel.shape}")

MIN_COUNTRIES_PER_YEAR = 100
COMPOSITE_SOURCE_COLS = ["internet_pct", "mobile_subs", "fixed_broadband", "electricity_access"]

coverage = panel.groupby("year")[COMPOSITE_SOURCE_COLS].apply(lambda df: df.notna().sum())

valid_years = coverage[(coverage >= MIN_COUNTRIES_PER_YEAR).all(axis=1)].index
dropped_years = coverage[~(coverage >= MIN_COUNTRIES_PER_YEAR).all(axis=1)]
if len(dropped_years):
    print(f"Dropping years where any component has < {MIN_COUNTRIES_PER_YEAR} countries reporting:")
    print(dropped_years)
panel = panel[panel["year"].isin(valid_years)].copy()
print(f"After dropping thin years: {panel.shape}")

COMPONENTS = ["internet_pct", "mobile_subs", "fixed_broadband", "electricity_access"]
WEIGHTS = {
    "internet_pct": 0.40,
    "mobile_subs": 0.20,
    "fixed_broadband": 0.20,
    "electricity_access": 0.20,
}

def minmax_normalize_by_year(df, col):
    """Normalize within each year's cross-section of countries, so the index
    reflects relative standing each year rather than being distorted by
    trend-level shifts over decades."""
    def _norm(group):
        lo, hi = group.min(), group.max()
        if hi - lo < 1e-9:
            return group * 0
        return 100 * (group - lo) / (hi - lo)
    return df.groupby("year")[col].transform(_norm)

panel["mobile_subs_capped"] = panel["mobile_subs"].clip(upper=150)

norm_components = {}
for comp in COMPONENTS:
    src_col = "mobile_subs_capped" if comp == "mobile_subs" else comp
    norm_components[comp] = minmax_normalize_by_year(panel, src_col)

components_df = pd.DataFrame(norm_components)
weights_series = pd.Series(WEIGHTS)

def weighted_available(row):
    available = row.dropna()
    if available.empty:
        return np.nan
    w = weights_series[available.index]
    return (available * w).sum() / w.sum()

panel["connectivity_index"] = components_df.apply(weighted_available, axis=1)

panel.loc[panel["internet_pct"].isna(), "connectivity_index"] = np.nan

print(panel[["country", "year", "connectivity_index"]].dropna().tail())

panel = panel.sort_values(["country_code", "year"])

FEATURES = ["gdp_per_capita", "electricity_access", "mobile_subs", "secondary_enroll", "urban_pop"]
TARGET = "connectivity_index"

panel["target_lag1"] = panel.groupby("country_code")[TARGET].shift(1)
panel["target_lag2"] = panel.groupby("country_code")[TARGET].shift(2)

MODEL_FEATURES = FEATURES + ["target_lag1", "target_lag2"]

panel[FEATURES] = panel.groupby("country_code")[FEATURES].transform(lambda s: s.ffill())
model_df = panel.dropna(subset=[TARGET, "target_lag1", "target_lag2"] + FEATURES).copy()
LAST_ACTUAL_YEAR = int(model_df["year"].max())
print(f"Modeling dataset: {model_df.shape}. Last actual data year available: {LAST_ACTUAL_YEAR}")

TRAIN_END_YEAR = LAST_ACTUAL_YEAR - 7
VAL_END_YEAR = LAST_ACTUAL_YEAR - 3

train = model_df[model_df["year"] <= TRAIN_END_YEAR]
val = model_df[(model_df["year"] > TRAIN_END_YEAR) & (model_df["year"] <= VAL_END_YEAR)]
test = model_df[model_df["year"] > VAL_END_YEAR]

X_train, y_train = train[MODEL_FEATURES], train[TARGET]
X_val, y_val = val[MODEL_FEATURES], val[TARGET]
X_test, y_test = test[MODEL_FEATURES], test[TARGET]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

baseline_pred = test["target_lag1"]
lr = LinearRegression().fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("Naive persistence  — RMSE: %.3f  MAE: %.3f" % (
    rmse(y_test, baseline_pred), mean_absolute_error(y_test, baseline_pred)))
print("Linear regression   — RMSE: %.3f  MAE: %.3f" % (
    rmse(y_test, lr_pred), mean_absolute_error(y_test, lr_pred)))

xgb = XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_pred = xgb.predict(X_test)
print("XGBoost              — RMSE: %.3f  MAE: %.3f" % (
    rmse(y_test, xgb_pred), mean_absolute_error(y_test, xgb_pred)))

rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest         — RMSE: %.3f  MAE: %.3f" % (
    rmse(y_test, rf_pred), mean_absolute_error(y_test, rf_pred)))

from scipy import stats

errors_naive = np.abs(y_test.values - baseline_pred.values)
errors_lr = np.abs(y_test.values - lr_pred)

diff = errors_naive - errors_lr
t_stat, t_pvalue = stats.ttest_rel(errors_naive, errors_lr)
w_stat, w_pvalue = stats.wilcoxon(errors_naive, errors_lr)

print(f"Mean |error| — naive: {errors_naive.mean():.3f}, linear: {errors_lr.mean():.3f}")
print(f"Paired t-test:      t = {t_stat:.3f}, p = {t_pvalue:.4f}")
print(f"Wilcoxon signed-rank: W = {w_stat:.1f}, p = {w_pvalue:.4f}")
if t_pvalue < 0.05:
    print("-> Difference is statistically significant at alpha = 0.05")
else:
    print("-> NOT statistically significant at alpha = 0.05 "
          "(report the RMSE/MAE gap as descriptive, not as a proven improvement)")

rng = np.random.default_rng(42)
n = len(y_test)
boot_diffs = []
y_test_arr = y_test.values
for _ in range(2000):
    idx = rng.integers(0, n, n)
    rmse_naive_b = rmse(y_test_arr[idx], baseline_pred.values[idx])
    rmse_lr_b = rmse(y_test_arr[idx], lr_pred[idx])
    boot_diffs.append(rmse_naive_b - rmse_lr_b)

boot_diffs = np.array(boot_diffs)
ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
print(f"\nBootstrap 95% CI for RMSE(naive) - RMSE(linear): [{ci_low:.3f}, {ci_high:.3f}]")
if ci_low > 0:
    print("-> CI excludes 0: linear regression's RMSE improvement is likely real")
else:
    print("-> CI includes 0: cannot rule out that the RMSE gap is due to chance")

plt.figure(figsize=(7, 4))
plt.hist(boot_diffs, bins=40, edgecolor="white")
plt.axvline(0, color="red", linestyle="--", label="No difference")
plt.axvline(ci_low, color="gray", linestyle=":", label="95% CI bounds")
plt.axvline(ci_high, color="gray", linestyle=":")
plt.xlabel("RMSE(naive) - RMSE(linear)  [positive = linear better]")
plt.ylabel("Bootstrap resamples")
plt.title("Bootstrap Distribution: Linear Regression vs. Naive Persistence")
plt.legend()
plt.tight_layout()
plt.savefig("bootstrap_significance.png", dpi=200)
plt.show()

importances = pd.Series(xgb.feature_importances_, index=MODEL_FEATURES).sort_values()
plt.figure(figsize=(7, 4))
importances.plot(kind="barh")
plt.title("XGBoost Feature Importance — Connectivity Index")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=200)
plt.show()

def logistic(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

FORECAST_END_YEAR = LAST_ACTUAL_YEAR + 25

def fit_logistic_for_country(country_df, cap=100):
    """Fit a logistic curve to one country's connectivity_index history.
    Falls back to a capped linear trend if the fit fails (e.g. too few
    points or a non-S-shaped series)."""
    years = country_df["year"].values.astype(float)
    values = country_df["connectivity_index"].values.astype(float)
    if len(years) < 5:
        return None

    try:
        p0 = [cap, 0.15, np.median(years)]
        bounds = ([values.max(), 0.001, years.min() - 30],
                  [cap + 1e-6, 2.0, years.max() + 60])
        popt, _ = curve_fit(logistic, years, values, p0=p0, bounds=bounds, maxfev=10000)
        return {"type": "logistic", "params": popt}
    except Exception:

        recent = country_df.sort_values("year").tail(10)
        if len(recent) < 2:
            return None
        slope, intercept = np.polyfit(recent["year"], recent["connectivity_index"], 1)
        return {"type": "linear", "params": (slope, intercept), "cap": cap}

def project_country(country_df, forecast_years, cap=100):
    fit = fit_logistic_for_country(country_df, cap=cap)
    if fit is None:
        return None
    if fit["type"] == "logistic":
        L, k, t0 = fit["params"]
        preds = logistic(forecast_years, L, k, t0)
    else:
        slope, intercept = fit["params"]
        preds = slope * forecast_years + intercept
    return np.clip(preds, 0, cap)

forecast_years = np.arange(LAST_ACTUAL_YEAR + 1, FORECAST_END_YEAR + 1)

long_horizon_results = {}
for country in model_df["country"].unique():
    cdf = model_df[model_df["country"] == country][["year", "connectivity_index"]].dropna()
    preds = project_country(cdf, forecast_years)
    if preds is not None:
        long_horizon_results[country] = preds

long_horizon_df = pd.DataFrame(long_horizon_results, index=forecast_years).T
long_horizon_df.index.name = "country"
long_horizon_df.to_csv("connectivity_index_2050_projection.csv")
print(f"Projected {len(long_horizon_df)} countries out to {FORECAST_END_YEAR}.")
print(long_horizon_df.iloc[:5, [0, 9, 24]])

COUNTRIES_TO_PLOT = ["Kazakhstan", "Uzbekistan", "Germany", "China", "Japan"]

fig, ax = plt.subplots(figsize=(10, 6))
for c in COUNTRIES_TO_PLOT:
    hist = model_df[model_df["country"] == c].sort_values("year")
    ax.plot(hist["year"], hist["connectivity_index"], marker="o", label=f"{c} — actual")

    if c in long_horizon_df.index:
        proj = long_horizon_df.loc[c]
        ax.plot(forecast_years, proj.values, linestyle="--", label=f"{c} — 25yr projection")

ax.axvline(LAST_ACTUAL_YEAR, color="gray", linestyle=":", linewidth=1)
ax.text(LAST_ACTUAL_YEAR, 5, " last observed data", fontsize=8, color="gray")
ax.set_xlabel("Year")
ax.set_ylabel("Connectivity Index (0-100)")
ax.set_title(f"Connectivity Index: Historical + Logistic Projection to {FORECAST_END_YEAR}")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("connectivity_25yr_projection.png", dpi=200)
plt.show()

model_df.to_csv("digital_divide_panel.csv", index=False)

results_table = pd.DataFrame({
    "Model": ["Naive persistence", "Linear Regression", "Random Forest", "XGBoost"],
    "RMSE": [rmse(y_test, baseline_pred), rmse(y_test, lr_pred), rmse(y_test, rf_pred), rmse(y_test, xgb_pred)],
    "MAE": [
        mean_absolute_error(y_test, baseline_pred),
        mean_absolute_error(y_test, lr_pred),
        mean_absolute_error(y_test, rf_pred),
        mean_absolute_error(y_test, xgb_pred),
    ],
})
results_table.to_csv("results_table.csv", index=False)
print(results_table)

print("\nDone. Files saved: digital_divide_panel.csv, results_table.csv, "
      "feature_importance.png, connectivity_index_2050_projection.csv, "
      "connectivity_25yr_projection.png")
