import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from scipy import stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import Holt
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option("display.max_columns", None)

RANDOM_STATE = 42
MIN_COUNTRIES_PER_YEAR = 100
MOBILE_CAP = 150
BOOTSTRAP_REPS = 2000
FORECAST_HORIZON = 25
COUNTRIES_TO_PLOT = ["Kazakhstan", "Uzbekistan", "Germany", "China", "Japan"]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def save_csv(df, filename, **kwargs):
    df.to_csv(filename, **kwargs)


def savefig(filename):
    plt.savefig(filename, dpi=200, bbox_inches="tight")



INDICATORS = {
    "internet_pct": "IT.NET.USER.ZS",
    "mobile_subs": "IT.CEL.SETS.P2",
    "fixed_broadband": "IT.NET.BBND.P2",
    "electricity_access": "EG.ELC.ACCS.ZS",
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "secondary_enroll": "SE.SEC.ENRR",
    "urban_pop": "SP.URB.TOTL.IN.ZS",
}


def fetch_indicator(code, per_page=20000):
    url = f"https://api.worldbank.org/v2/country/all/indicator/{code}"
    r = requests.get(url, params={"format": "json", "per_page": per_page}, timeout=60)
    r.raise_for_status()
    payload = r.json()
    if len(payload) < 2 or payload[1] is None:
        return pd.DataFrame(columns=["country", "country_code", "year", "value"])
    rows = []
    for rec in payload[1]:
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
    d = fetch_indicator(code).rename(columns={"value": name})
    frames[name] = d
    print(f"  {name}: {len(d):,} rows; latest={d['year'].max() if not d.empty else None}")

panel = frames["internet_pct"][["country", "country_code", "year"]].copy()
for name, d in frames.items():
    panel = panel.merge(d[["country_code", "year", name]],
                        on=["country_code", "year"], how="left")

AGGREGATE_KEYWORDS = [
    "World", "income", "IDA", "IBRD", "OECD", "Euro area", "Arab World",
    "Fragile", "Least developed", "Africa Eastern", "Africa Western",
    "Central Europe", "East Asia", "Europe &", "Latin America &",
    "Middle East &", "North America", "South Asia", "Sub-Saharan",
    "Small states", "Heavily indebted", "Pacific island", "European Union",
]
mask_aggregate = panel["country"].str.contains("|".join(AGGREGATE_KEYWORDS), case=False, na=False)
panel = panel.loc[~mask_aggregate].copy()

COMPOSITE_SOURCE_COLS = ["internet_pct", "mobile_subs", "fixed_broadband", "electricity_access"]
coverage = panel.groupby("year")[COMPOSITE_SOURCE_COLS].apply(lambda x: x.notna().sum())
valid_years = coverage[(coverage >= MIN_COUNTRIES_PER_YEAR).all(axis=1)].index
excluded = coverage.loc[~(coverage >= MIN_COUNTRIES_PER_YEAR).all(axis=1)].copy()
panel = panel[panel["year"].isin(valid_years)].copy()

filter_summary = pd.DataFrame({
    "rule": ["minimum_reporting_countries_per_year_for_each_index_component"],
    "threshold": [MIN_COUNTRIES_PER_YEAR],
    "first_year": [int(panel.year.min())],
    "last_year": [int(panel.year.max())],
    "n_valid_years": [int(panel.year.nunique())],
    "n_excluded_years": [int(len(excluded))],
    "excluded_years": [", ".join(map(str, excluded.index.astype(int).tolist()))],
})
save_csv(filter_summary, "data_filtering_summary.csv", index=False)
excluded.to_csv("excluded_year_component_coverage.csv")
print("Excluded years:", excluded.index.astype(int).tolist())



COMPONENTS = COMPOSITE_SOURCE_COLS
BASELINE_WEIGHTS = {
    "internet_pct": 0.40,
    "mobile_subs": 0.20,
    "fixed_broadband": 0.20,
    "electricity_access": 0.20,
}

# The last specification is deliberately different from equal weights.
# It gives more weight to fixed broadband + electricity as infrastructure,
# while retaining a smaller mobile-subscription component.
WEIGHT_SCENARIOS = {
    "baseline_40_20_20_20": BASELINE_WEIGHTS,
    "equal_25_25_25_25": {
        "internet_pct": .25, "mobile_subs": .25,
        "fixed_broadband": .25, "electricity_access": .25,
    },
    "internet_dominant_50_16_7_16_7_16_7": {
        "internet_pct": .50, "mobile_subs": 1/6,
        "fixed_broadband": 1/6, "electricity_access": 1/6,
    },
    "infrastructure_balanced_30_15_25_30": {
        "internet_pct": .30, "mobile_subs": .15,
        "fixed_broadband": .25, "electricity_access": .30,
    },
}


def minmax_by_year(df, col):
    def normalize(s):
        lo, hi = s.min(), s.max()
        if pd.isna(lo) or pd.isna(hi) or hi - lo < 1e-12:
            return pd.Series(np.nan, index=s.index)
        return 100 * (s - lo) / (hi - lo)
    return df.groupby("year")[col].transform(normalize)


panel["mobile_subs_capped"] = panel["mobile_subs"].clip(upper=MOBILE_CAP)
norm_components = {}
for comp in COMPONENTS:
    src = "mobile_subs_capped" if comp == "mobile_subs" else comp
    norm_components[comp] = minmax_by_year(panel, src)
components_df = pd.DataFrame(norm_components, index=panel.index)


def build_index(norm_df, weights):
    w = pd.Series(weights, dtype=float)
    weighted = norm_df.mul(w, axis=1)
    denom = norm_df.notna().mul(w, axis=1).sum(axis=1)
    out = weighted.sum(axis=1, min_count=1) / denom.replace(0, np.nan)
    return out


for name, weights in WEIGHT_SCENARIOS.items():
    panel[f"index__{name}"] = build_index(components_df, weights)
panel["connectivity_index"] = panel["index__baseline_40_20_20_20"]
panel.loc[panel["internet_pct"].isna(), "connectivity_index"] = np.nan

# Explicitly record that the index is a relative within-year standing.
index_diag = []
for year, g in panel.groupby("year"):
    row = {"year": int(year), "n_countries": int(len(g))}
    for comp in COMPONENTS:
        src = "mobile_subs_capped" if comp == "mobile_subs" else comp
        s = g[src].dropna()
        row[f"{comp}_min"] = s.min() if len(s) else np.nan
        row[f"{comp}_max"] = s.max() if len(s) else np.nan
        row[f"{comp}_n"] = len(s)
    index_diag.append(row)
save_csv(pd.DataFrame(index_diag), "index_normalization_diagnostics.csv", index=False)



FEATURES = ["gdp_per_capita", "electricity_access", "mobile_subs",
            "secondary_enroll", "urban_pop"]
TARGET = "connectivity_index"

panel = panel.sort_values(["country_code", "year"]).copy()
panel["target_lag1"] = panel.groupby("country_code")[TARGET].shift(1)
panel["target_lag2"] = panel.groupby("country_code")[TARGET].shift(2)

# Forward-fill is past-only: no backward fill and no interpolation.
# We audit every filled value and record whether it crosses a global split.
pre_ffill = panel[FEATURES].copy()
for f in FEATURES:
    panel[f] = panel.groupby("country_code")[f].ffill()

ffill_rows = []
for f in FEATURES:
    changed = pre_ffill[f].isna() & panel[f].notna()
    for idx in panel.index[changed]:
        r = panel.loc[idx]
        ffill_rows.append({"row_index": idx, "country": r.country, "year": int(r.year), "feature": f})
ffill_diag = pd.DataFrame(ffill_rows)

model_df = panel.dropna(subset=[TARGET, "target_lag1", "target_lag2"] + FEATURES).copy()
LAST_ACTUAL_YEAR = int(model_df.year.max())
TRAIN_END_YEAR = LAST_ACTUAL_YEAR - 7
VAL_END_YEAR = LAST_ACTUAL_YEAR - 3

if not ffill_diag.empty:
    ffill_diag["split"] = np.select(
        [ffill_diag.year <= TRAIN_END_YEAR,
         ffill_diag.year <= VAL_END_YEAR],
        ["train", "validation"], default="test")
    ffill_diag["crosses_split_boundary"] = False
    # A forward fill can only use an earlier value. Mark rows in val/test that
    # were filled and therefore depend on a value from an earlier split.
    for i, r in ffill_diag.iterrows():
        prior = panel[(panel.country == r.country) & (panel.year < r.year)][r.feature].dropna()
        if prior.empty:
            continue
        prior_year = int(panel[(panel.country == r.country) & (panel.year < r.year) &
                               panel[r.feature].notna()].year.max())
        prior_split = "train" if prior_year <= TRAIN_END_YEAR else ("validation" if prior_year <= VAL_END_YEAR else "test")
        ffill_diag.loc[i, "source_year"] = prior_year
        ffill_diag.loc[i, "source_split"] = prior_split
        ffill_diag.loc[i, "crosses_split_boundary"] = prior_split != r.split
else:
    ffill_diag = pd.DataFrame(columns=["row_index", "country", "year", "feature", "split",
                                       "source_year", "source_split", "crosses_split_boundary"])
save_csv(ffill_diag, "ffill_diagnostics.csv", index=False)

MODEL_FEATURES = FEATURES + ["target_lag1", "target_lag2"]
train = model_df[model_df.year <= TRAIN_END_YEAR].copy()
val = model_df[(model_df.year > TRAIN_END_YEAR) & (model_df.year <= VAL_END_YEAR)].copy()
test = model_df[model_df.year > VAL_END_YEAR].copy()

X_train, y_train = train[MODEL_FEATURES], train[TARGET]
X_val, y_val = val[MODEL_FEATURES], val[TARGET]
X_test, y_test = test[MODEL_FEATURES], test[TARGET]

print(f"Year range: {model_df.year.min()}-{LAST_ACTUAL_YEAR}")
print(f"Train: {len(train):,}; validation: {len(val):,}; test: {len(test):,}")



# Group 1: persistence baseline.
# Group 2: multivariate supervised models (linear, RF, XGBoost).
# Group 3: univariate time-series benchmarks (ARIMA, Holt/ETS).

baseline_pred = test["target_lag1"].to_numpy()
lr = LinearRegression().fit(X_train, y_train)
lr_pred = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=300, max_depth=8,
                           random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

xgb = XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=.05,
    subsample=.8, colsample_bytree=.8, random_state=RANDOM_STATE,
    objective="reg:squarederror",
)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_pred = xgb.predict(X_test)

ml_preds = {
    "Naive persistence": baseline_pred,
    "Linear Regression": lr_pred,
    "Random Forest": rf_pred,
    "XGBoost": xgb_pred,
}



ARIMA_ORDERS = [(0,1,0), (1,1,0), (0,1,1), (1,1,1)]
HOLT_SPECS = ["holt", "holt_damped"]


def series_for_country(df, country):
    s = df[df.country == country][["year", TARGET]].dropna().drop_duplicates("year").sort_values("year")
    return s.set_index("year")[TARGET].astype(float)


def arima_forecast(series, horizon, order):
    if len(series) < 8:
        return None
    try:
        fitted = ARIMA(series, order=order,
                       enforce_stationarity=False,
                       enforce_invertibility=False).fit()
        return np.asarray(fitted.forecast(horizon), dtype=float)
    except Exception:
        return None


def holt_forecast(series, horizon, damped):
    if len(series) < 6:
        return None
    try:
        fitted = Holt(series, damped_trend=damped).fit(optimized=True)
        return np.asarray(fitted.forecast(horizon), dtype=float)
    except Exception:
        return None


def choose_univariate_model(series, kind):
    tr = series[series.index <= TRAIN_END_YEAR]
    va = series[(series.index > TRAIN_END_YEAR) & (series.index <= VAL_END_YEAR)]
    if len(va) == 0:
        return (1,1,0) if kind == "ARIMA" else False
    candidates = ARIMA_ORDERS if kind == "ARIMA" else HOLT_SPECS
    best, best_score = None, np.inf
    for candidate in candidates:
        pred = (arima_forecast(tr, len(va), candidate) if kind == "ARIMA"
                else holt_forecast(tr, len(va), candidate == "holt_damped"))
        if pred is None or len(pred) != len(va):
            continue
        score = rmse(va.values, pred)
        if np.isfinite(score) and score < best_score:
            best, best_score = candidate, score
    return best if best is not None else ((1,1,0) if kind == "ARIMA" else False)


arima_records, holt_records, selection_records = [], [], []
for country in test.country.unique():
    s = series_for_country(model_df, country)
    if s.empty:
        continue
    order = choose_univariate_model(s, "ARIMA")
    damped = choose_univariate_model(s, "HOLT")
    fit_s = s[s.index <= VAL_END_YEAR]
    test_rows = test[test.country == country].sort_values("year")
    if fit_s.empty or test_rows.empty:
        continue
    max_year = int(test_rows.year.max())
    horizon = max_year - VAL_END_YEAR
    ap = arima_forecast(fit_s, horizon, order)
    hp = holt_forecast(fit_s, horizon, bool(damped))
    if ap is None or hp is None:
        continue
    year_map_a = dict(zip(range(VAL_END_YEAR+1, max_year+1), ap))
    year_map_h = dict(zip(range(VAL_END_YEAR+1, max_year+1), hp))
    for _, row in test_rows.iterrows():
        y = int(row.year)
        if y in year_map_a:
            arima_records.append({"row_index": row.name, "country": country, "year": y,
                                  "prediction": year_map_a[y]})
        if y in year_map_h:
            holt_records.append({"row_index": row.name, "country": country, "year": y,
                                 "prediction": year_map_h[y]})
    selection_records.append({"country": country, "arima_order": str(order),
                              "holt_damped": bool(damped)})

arima_df = pd.DataFrame(arima_records)
holt_df = pd.DataFrame(holt_records)
selection_df = pd.DataFrame(selection_records)
save_csv(selection_df, "arima_ets_selection.csv", index=False)

# Evaluate ARIMA/ETS on the same common observations available for each model.
# For the headline aggregate comparison, report both model coverage and metrics.
model_results = []
for name, pred in ml_preds.items():
    model_results.append({"Model": name, "N": len(y_test),
                          "RMSE": rmse(y_test, pred), "MAE": mae(y_test, pred)})

for name, d in [("ARIMA", arima_df), ("Holt/ETS", holt_df)]:
    if d.empty:
        model_results.append({"Model": name, "N": 0, "RMSE": np.nan, "MAE": np.nan})
    else:
        idx = d.row_index.values
        y = model_df.loc[idx, TARGET].to_numpy()
        p = d.prediction.to_numpy()
        model_results.append({"Model": name, "N": len(y), "RMSE": rmse(y,p), "MAE": mae(y,p)})

results_table = pd.DataFrame(model_results)
save_csv(results_table, "model_results_aggregate.csv", index=False)



horizon_rows = []
for h in [1, 2, 3]:
    origins = sorted(y for y in model_df.year.unique()
                     if y >= TRAIN_END_YEAR and y+h <= LAST_ACTUAL_YEAR)
    # Use the last three feasible origins in the held-out portion as a compact
    # horizon diagnostic; all observations remain strictly chronological.
    origins = [o for o in origins if o >= VAL_END_YEAR][:3]
    if not origins:
        continue
    for origin in origins:
        actual = model_df[model_df.year == origin+h][["country", TARGET]].copy()
        lag = model_df[model_df.year == origin][["country", TARGET]].rename(columns={TARGET:"pred"})
        merged = actual.merge(lag, on="country", how="inner")
        if len(merged):
            horizon_rows.append({"method":"Naive persistence", "horizon":h,
                                 "origin":origin, "N":len(merged),
                                 "RMSE":rmse(merged[TARGET], merged.pred),
                                 "MAE":mae(merged[TARGET], merged.pred)})

horizon_table = pd.DataFrame(horizon_rows)
save_csv(horizon_table, "model_results_by_horizon.csv", index=False)


errors_naive = np.abs(y_test.to_numpy() - baseline_pred)
errors_lr = np.abs(y_test.to_numpy() - lr_pred)
t_stat, t_p = stats.ttest_rel(errors_naive, errors_lr)
w_stat, w_p = stats.wilcoxon(errors_naive, errors_lr)

rng = np.random.default_rng(RANDOM_STATE)
country_ids = test.country.unique()
cluster_boot = []
for _ in range(BOOTSTRAP_REPS):
    sampled = rng.choice(country_ids, size=len(country_ids), replace=True)
    idxs = np.concatenate([test.index[test.country == c].to_numpy() for c in sampled])
    yy = model_df.loc[idxs, TARGET].to_numpy()
    nn = test.loc[idxs, "target_lag1"].to_numpy()
    ll = lr.predict(test.loc[idxs, MODEL_FEATURES])
    cluster_boot.append(rmse(yy, nn) - rmse(yy, ll))
cluster_boot = np.asarray(cluster_boot)
ci_low, ci_high = np.percentile(cluster_boot, [2.5,97.5])

significance = pd.DataFrame([{
    "comparison":"Naive persistence vs Linear Regression",
    "paired_t":t_stat, "paired_t_p":t_p,
    "wilcoxon_W":w_stat, "wilcoxon_p":w_p,
    "cluster_bootstrap_rmse_difference_low":ci_low,
    "cluster_bootstrap_rmse_difference_high":ci_high,
    "bootstrap_definition":"country-cluster resampling with all country-year test rows retained"
}])
save_csv(significance, "significance_tests.csv", index=False)
save_csv(pd.DataFrame({"rmse_difference":cluster_boot}), "bootstrap_rmse_difference.csv", index=False)



# Standard XGBoost importance remains a descriptive diagnostic.
std_imp = pd.DataFrame({"feature":MODEL_FEATURES,
                        "gain_share":xgb.feature_importances_})

# Permutation importance on the held-out test set. This is still subject to
# correlated-feature masking, so it is interpreted jointly with ablation/VIF.
perm = permutation_importance(xgb, X_test, y_test,
                              n_repeats=30, random_state=RANDOM_STATE,
                              scoring="neg_root_mean_squared_error")
perm_df = pd.DataFrame({
    "feature":MODEL_FEATURES,
    "permutation_mean":perm.importances_mean,
    "permutation_std":perm.importances_std,
})
perm_df = perm_df.sort_values("permutation_mean", ascending=False)
save_csv(std_imp, "xgboost_standard_importance.csv", index=False)
save_csv(perm_df, "permutation_importance.csv", index=False)

# Correlations and VIF.
corr_df = model_df[MODEL_FEATURES + [TARGET]].corr()
corr_df.to_csv("feature_correlations.csv")
X_vif = model_df[MODEL_FEATURES].replace([np.inf,-np.inf],np.nan).dropna()
# Standardize only for numerical stability; VIF itself is scale invariant.
Xv = X_vif.copy()
vif_rows=[]
for i, f in enumerate(MODEL_FEATURES):
    try:
        vif_rows.append({"feature":f, "VIF":variance_inflation_factor(Xv.values, i)})
    except Exception:
        vif_rows.append({"feature":f, "VIF":np.nan})
save_csv(pd.DataFrame(vif_rows), "feature_vif.csv", index=False)

# Lag ablation.
ablation_specs = {
    "full": MODEL_FEATURES,
    "no_lag1": [f for f in MODEL_FEATURES if f != "target_lag1"],
    "no_lag2": [f for f in MODEL_FEATURES if f != "target_lag2"],
    "no_lags": FEATURES,
}
ablation_rows=[]
for name, feats in ablation_specs.items():
    m = XGBRegressor(n_estimators=300,max_depth=4,learning_rate=.05,
                     subsample=.8,colsample_bytree=.8,random_state=RANDOM_STATE,
                     objective="reg:squarederror")
    m.fit(train[feats], y_train, eval_set=[(val[feats],y_val)], verbose=False)
    p = m.predict(test[feats])
    ablation_rows.append({"specification":name,"features":";".join(feats),
                          "RMSE":rmse(y_test,p),"MAE":mae(y_test,p)})
ablation_df=pd.DataFrame(ablation_rows)
save_csv(ablation_df,"lag_ablation_results.csv",index=False)



def logistic(t,L,k,t0):
    z=np.clip(-k*(np.asarray(t)-t0),-700,700)
    return L/(1+np.exp(z))


def fit_logistic(df):
    d=df.dropna().sort_values("year")
    if len(d)<6: return None
    t=d.year.to_numpy(float); y=d[TARGET].to_numpy(float)
    try:
        p0=[min(100,max(y.max(),1)),.15,np.median(t)]
        pars,_=curve_fit(logistic,t,y,p0=p0,
                         bounds=([0,.001,t.min()-30],[100,2,t.max()+60]),maxfev=30000)
        return {"type":"logistic","params":pars}
    except Exception:
        return None


def fit_linear_recent(df):
    d=df.dropna().sort_values("year")
    if len(d)<2: return None
    x=d.year.to_numpy(float); y=d[TARGET].to_numpy(float)
    slope,intercept=np.polyfit(x,y,1)
    return {"type":"linear","params":(slope,intercept)}


def project_fit(fit, years):
    if fit is None: return np.full(len(years),np.nan)
    if fit["type"]=="logistic":
        return np.clip(logistic(years,*fit["params"]),0,100)
    slope,intercept=fit["params"]
    return np.clip(slope*np.asarray(years)+intercept,0,100)

forecast_years=np.arange(LAST_ACTUAL_YEAR+1,LAST_ACTUAL_YEAR+FORECAST_HORIZON+1)

# Diagnostic: does logistic fit the historical observations better than a simple
# linear trend? This does not prove an S-curve; it tests whether logistic is a
# useful descriptive approximation for each focal country.
fit_rows=[]
for country in COUNTRIES_TO_PLOT:
    c=model_df[model_df.country==country][["year",TARGET]].dropna()
    if c.empty: continue
    lf=fit_logistic(c); lin=fit_linear_recent(c)
    if lf:
        lp=project_fit(lf,c.year.to_numpy())
        lrmse=rmse(c[TARGET],lp)
    else: lrmse=np.nan
    if lin:
        rp=project_fit(lin,c.year.to_numpy())
        rrmse=rmse(c[TARGET],rp)
    else: rrmse=np.nan
    fit_rows.append({"country":country,"n_observations":len(c),
                     "logistic_in_sample_RMSE":lrmse,
                     "linear_in_sample_RMSE":rrmse,
                     "logistic_better_in_sample": bool(lrmse<rrmse) if np.isfinite(lrmse+rrmse) else np.nan})
save_csv(pd.DataFrame(fit_rows),"logistic_fit_diagnostics.csv",index=False)

# Kazakhstan scenarios are deliberately model-based sensitivity scenarios:
# 1 conservative = recent linear trend (2018-last)
# 2 baseline = full-history logistic
# 3 accelerated = pre-stabilization logistic through 2017, carried forward as
# a historical high-growth calibration rather than a claim about future policy.
kaz=model_df[model_df.country=="Kazakhstan"][["year",TARGET]].dropna().sort_values("year")
scenario_specs={
    "conservative_recent_linear_2018_last": ("linear",kaz[kaz.year>=2018]),
    "baseline_full_history_logistic": ("logistic",kaz),
    "accelerated_pre_stabilization_logistic": ("logistic",kaz[kaz.year<=2017]),
}
scenario_rows=[]
for name,(kind,d) in scenario_specs.items():
    fit=fit_linear_recent(d) if kind=="linear" else fit_logistic(d)
    pred=project_fit(fit,forecast_years)
    for yr,pv in zip(forecast_years,pred):
        scenario_rows.append({"scenario":name,"calibration_start":int(d.year.min()) if len(d) else np.nan,
                              "calibration_end":int(d.year.max()) if len(d) else np.nan,
                              "year":int(yr),"projection":pv})
scenario_df=pd.DataFrame(scenario_rows)
save_csv(scenario_df,"kazakhstan_scenarios.csv",index=False)

weight_rows=[]
trajectory_rows=[]


def model_df_for_index(index_series):
    d=panel[["country","country_code","year"]+FEATURES].copy()
    d[TARGET]=index_series
    d=d.sort_values(["country_code","year"])
    d["target_lag1"]=d.groupby("country_code")[TARGET].shift(1)
    d["target_lag2"]=d.groupby("country_code")[TARGET].shift(2)
    d[FEATURES]=d.groupby("country_code")[FEATURES].ffill()
    return d.dropna(subset=[TARGET,"target_lag1","target_lag2"]+FEATURES)

baseline_series=panel["connectivity_index"]
baseline_model=model_df_for_index(baseline_series)
base_test=baseline_model[baseline_model.year>VAL_END_YEAR]

for name, weights in WEIGHT_SCENARIOS.items():
    idx=build_index(components_df,weights)
    d=model_df_for_index(idx)
    cor=idx.corr(baseline_series)
    t=d[d.year>VAL_END_YEAR]
    # Same linear model for sensitivity; this is not a new claim about the best model.
    tr=d[d.year<=TRAIN_END_YEAR]
    lr_s=LinearRegression().fit(tr[MODEL_FEATURES],tr[TARGET])
    pred_s=lr_s.predict(t[MODEL_FEATURES])
    row={"weight_scenario":name,**{f"w_{k}":v for k,v in weights.items()},
         "index_corr_with_baseline":cor,
         "test_N":len(t),"linear_RMSE":rmse(t[TARGET],pred_s),"linear_MAE":mae(t[TARGET],pred_s)}
    for c in ["Kazakhstan","Uzbekistan"]:
        b=baseline_model[baseline_model.country==c].set_index("year")[TARGET]
        s=d[d.country==c].set_index("year")[TARGET]
        common=b.index.intersection(s.index)
        row[f"{c}_mean_abs_index_change"]=(b.loc[common]-s.loc[common]).abs().mean()
    weight_rows.append(row)

    # Long-run trajectories for focal countries under each weighting scheme.
    for country in COUNTRIES_TO_PLOT:
        c=d[d.country==country][["year",TARGET]].sort_values("year")
        if c.empty: continue
        fit=fit_logistic = None
        # full-history logistic for weight sensitivity only
        try:
            yy=c[TARGET].to_numpy(float); tt=c.year.to_numpy(float)
            p0=[min(100,max(yy.max(),1)),.15,np.median(tt)]
            pars,_=curve_fit(logistic,tt,yy,p0=p0,
                             bounds=([0,.001,tt.min()-30],[100,2,tt.max()+60]),maxfev=20000)
            pred=logistic(np.arange(LAST_ACTUAL_YEAR+1,LAST_ACTUAL_YEAR+FORECAST_HORIZON+1),*pars)
        except Exception:
            pred=np.full(FORECAST_HORIZON,np.nan)
        for yr,pv in zip(range(LAST_ACTUAL_YEAR+1,LAST_ACTUAL_YEAR+FORECAST_HORIZON+1),pred):
            trajectory_rows.append({"weight_scenario":name,"country":country,"year":yr,"projection":pv})

weight_table=pd.DataFrame(weight_rows)
save_csv(weight_table,"weight_sensitivity_results.csv",index=False)
trajectory_table=pd.DataFrame(trajectory_rows)
save_csv(trajectory_table,"weight_sensitivity_country_trajectories.csv",index=False)



plt.figure(figsize=(8,5))
plt.bar(results_table.Model,results_table.RMSE)
plt.ylabel("RMSE")
plt.xticks(rotation=25,ha="right")
plt.title("Short-term model comparison")
plt.tight_layout(); savefig("model_comparison_rmse.png"); plt.close()

plt.figure(figsize=(8,5))
p=perm_df.sort_values("permutation_mean")
plt.barh(p.feature,p.permutation_mean,xerr=p.permutation_std)
plt.xlabel("Permutation importance: RMSE increase")
plt.title("XGBoost permutation importance")
plt.tight_layout(); savefig("permutation_importance.png"); plt.close()

plt.figure(figsize=(8,5))
for name in WEIGHT_SCENARIOS:
    s=trajectory_table[(trajectory_table.weight_scenario==name)&(trajectory_table.country=="Kazakhstan")]
    plt.plot(s.year,s.projection,label=name)
plt.xlabel("Year"); plt.ylabel("Connectivity Index (relative 0-100)")
plt.title("Kazakhstan: sensitivity to index weights")
plt.legend(fontsize=7); plt.tight_layout(); savefig("kazakhstan_weight_sensitivity.png"); plt.close()

plt.figure(figsize=(9,6))
for name in scenario_specs:
    s=scenario_df[scenario_df.scenario==name]
    plt.plot(s.year,s.projection,label=name)
plt.plot(kaz.year,kaz[TARGET],marker="o",label="historical index")
plt.axvline(LAST_ACTUAL_YEAR,linestyle=":",linewidth=1)
plt.xlabel("Year"); plt.ylabel("Connectivity Index (relative 0-100)")
plt.title("Kazakhstan: model-based long-horizon scenarios")
plt.legend(fontsize=8); plt.tight_layout(); savefig("kazakhstan_scenarios.png"); plt.close()

# Baseline full-history logistic trajectories.
projection_rows=[]
for country in COUNTRIES_TO_PLOT:
    c=model_df[model_df.country==country][["year",TARGET]].dropna()
    fit=fit_logistic(c)
    pred=project_fit(fit,forecast_years)
    for yr,pv in zip(forecast_years,pred):
        projection_rows.append({"country":country,"year":int(yr),"projection":pv})
projection_df=pd.DataFrame(projection_rows)
save_csv(projection_df,"comparative_country_25yr_projection.csv",index=False)

plt.figure(figsize=(10,6))
for country in COUNTRIES_TO_PLOT:
    h=model_df[model_df.country==country].sort_values("year")
    p=projection_df[projection_df.country==country]
    plt.plot(h.year,h[TARGET],label=f"{country} actual")
    plt.plot(p.year,p.projection,linestyle="--",label=f"{country} scenario")
plt.axvline(LAST_ACTUAL_YEAR,linestyle=":",linewidth=1)
plt.xlabel("Year"); plt.ylabel("Connectivity Index (relative 0-100)")
plt.title(f"Connectivity trajectories to {LAST_ACTUAL_YEAR+FORECAST_HORIZON}: scenario extrapolation")
plt.legend(fontsize=7,ncol=2); plt.tight_layout(); savefig("connectivity_25yr_projection.png"); plt.close()



proxy = model_df.groupby("year")[["secondary_enroll","gdp_per_capita","urban_pop",TARGET]].agg(["mean","median","count"])
proxy.to_csv("digital_divide_contextual_proxies.csv")

save_csv(model_df,"digital_divide_panel.csv",index=False)




print("\n"+"="*72)
print("REVIEWER-RESPONSE RERUN COMPLETE")
print("="*72)
print(f"Valid WDI years: {panel.year.min()}-{panel.year.max()}")
print(f"Excluded years: {excluded.index.astype(int).tolist()}")
print(f"Modeling observations: {len(model_df):,}; countries: {model_df.country.nunique()}")
print(f"Train <= {TRAIN_END_YEAR}; validation {TRAIN_END_YEAR+1}-{VAL_END_YEAR}; test > {VAL_END_YEAR}")
print("\nAggregate model results:")
print(results_table.to_string(index=False))
print("\nHorizon-specific persistence diagnostics:")
print(horizon_table.to_string(index=False) if not horizon_table.empty else "No feasible horizon diagnostics")
print("\nWeight sensitivity:")
print(weight_table.to_string(index=False))
print("\nKazakhstan scenario endpoints:")
print(scenario_df[scenario_df.year==LAST_ACTUAL_YEAR+FORECAST_HORIZON].to_string(index=False))
print("\nDone. All new statistics should be copied into the manuscript only after this run completes.")

