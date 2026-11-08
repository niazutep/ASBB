"""ASBB (Adaptive Seasonal Baseline Blend) and the classical baselines it is compared against.

ASBB forecasts each month as (1-w)*HistoricalMean + w*HoltWinters + b, where w, b, and the
calibration window K in {6,12,18,24} are chosen jointly at every forecast origin by a grid search
over the trailing months. See Section 4.5 of the paper for the full specification.

Set the ASBB_DATA_DIR environment variable to the directory holding the rebuilt CISA advisory
panel (vuln_df_ot_all.parquet). This repository does not redistribute the raw advisory text; see
the top-level README for how the panel is built from CISA's own public CSAF repository.
"""
import os
import numpy as np, pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

DATA_DIR = os.environ.get("ASBB_DATA_DIR", "./data_raw")
VULN_PATH = os.path.join(DATA_DIR, "vuln_df_ot_all.parquet")

# ---------------- panel construction ----------------
def merge_vendor_mapping(vuln_df_ot_all, vendor_cwe_df=None):
    df = vuln_df_ot_all.copy()
    df['Advisory Release Date'] = pd.to_datetime(df['Advisory Release Date'], errors='coerce', utc=True)
    df['Vuln Release Date'] = pd.to_datetime(df['Vuln Release Date'], errors='coerce', utc=True)
    df['Event Date'] = df['Advisory Release Date'].fillna(df['Vuln Release Date'])
    df['Event Month'] = df['Event Date'].dt.tz_localize(None).dt.to_period('M').dt.to_timestamp()
    if 'CVSS Score' in df.columns:
        df['CVSS Score'] = pd.to_numeric(df['CVSS Score'], errors='coerce')
    df['Vendor'] = 'Unknown Vendor'
    return df

def build_monthly_panel(df):
    gcols = ['Vendor','CWE ID','CWE Name','Event Month']
    agg = df.groupby(gcols).agg(n_events=('CVE','nunique')).reset_index()
    agg['Year'] = agg['Event Month'].dt.year
    agg['Month'] = agg['Event Month'].dt.month
    return agg

def add_temporal_features(panel, group_cols=['Vendor','CWE ID']):
    out = []
    for key, sub in panel.groupby(group_cols, dropna=False):
        sub = sub.set_index('Event Month').asfreq('MS')
        sub['n_events'] = sub['n_events'].fillna(0)
        if 'CWE Name' in sub.columns: sub['CWE Name'] = sub['CWE Name'].ffill().bfill()
        if 'CWE ID' in sub.columns:   sub['CWE ID'] = sub['CWE ID'].ffill().bfill()
        out.append(sub.reset_index())
    feat = pd.concat(out, ignore_index=True)
    feat['Year'] = feat['Event Month'].dt.year
    feat['Month'] = feat['Event Month'].dt.month
    return feat

def load_panel():
    v = pd.read_parquet(VULN_PATH)
    return add_temporal_features(build_monthly_panel(merge_vendor_mapping(v)))

# ---------------- forecasters ----------------
def fc_naive(train, steps):
    last = float(train.iloc[-1]) if len(train) else 0.0
    idx = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(), periods=steps, freq="MS")
    return pd.Series([last]*steps, index=idx)

def fc_snaive(train, steps, season=6):
    idx = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(), periods=steps, freq="MS")
    if len(train) < season: return fc_naive(train, steps)
    vals = []
    for ts in idx:
        src = ts - pd.DateOffset(months=season)
        vals.append(float(train.get(src, train.iloc[-1])))
    return pd.Series(vals, index=idx)

def fc_holt_winters(train, steps):
    model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12)
    return model.fit(optimized=True).forecast(steps)

def fc_sarimax(train, steps, order=(1,1,1), seasonal_order=(1,1,1,12)):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    v = fit.get_forecast(steps).predicted_mean
    v.index = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(), periods=steps, freq="MS")
    return v

def fc_histmean(train, steps):
    idx = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(), periods=steps, freq="MS")
    return pd.Series([float(train.mean())]*steps, index=idx)

def _rolling_preds_fast_hm(y, season=12, start_idx=0):
    """Trailing one-step predictions used to score each (K, w) candidate during calibration:
    the historical mean (expanding) and Holt-Winters (fitted once, read off its fitted values)."""
    cs = y.cumsum(); n = np.arange(1, len(y) + 1)
    pred_hm_full = pd.Series(cs.shift(1).values / (n - 1), index=y.index)
    try:
        res = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=season,
                                   initialization_method="estimated").fit(optimized=True, use_brute=False)
        pred_hw_full = res.fittedvalues.shift(1)
    except Exception:
        pred_hw_full = pd.Series(index=y.index, dtype=float)
    idx = y.index[start_idx+1:]
    return pred_hm_full.loc[idx], pred_hw_full.loc[idx]

def fc_asbb(train, steps, season=12, calib_months_grid=(6,12,18,24), loss="mae", bias_cap=None):
    """ASBB: (1-w)*HistMean + w*HoltWinters + b, with w, b, and the calibration window K chosen
    jointly by grid search at each forecast origin. See Section 4.5 of the paper."""
    y = train.asfreq("MS").fillna(0)
    if bias_cap is None: bias_cap = max(1.0, 0.1 * float(y.mean()))
    f_hm_h = fc_histmean(y, steps=steps)
    try: f_hw_h = fc_holt_winters(y, steps=steps)
    except Exception: return f_hm_h.clip(lower=0)
    T = len(y); min_start = season
    valid_grid = [m for m in calib_months_grid if T >= (min_start + m + 1)]
    if not valid_grid: return (0.5*f_hm_h + 0.5*f_hw_h).clip(lower=0)
    best_cfg = None
    for calib_m in valid_grid:
        start_idx = T - (calib_m + 1)
        pred_hm, pred_hw = _rolling_preds_fast_hm(y, season=season, start_idx=start_idx)
        target = y.loc[pred_hm.index]
        dfc = pd.DataFrame({"y": target, "hm": pred_hm, "hw": pred_hw}).dropna()
        if len(dfc) < max(6, season // 2): continue
        ws = np.linspace(0.0, 1.0, 51); best = (np.inf, None, None)
        for w in ws:
            yhat = (1 - w) * dfc["hm"] + w * dfc["hw"]
            b = float((dfc["y"] - yhat).median())
            if b > bias_cap: b = bias_cap
            elif b < -bias_cap: b = -bias_cap
            yhat_b = yhat + b
            L = np.mean(np.abs(dfc["y"].values - yhat_b.values))
            if L < best[0]: best = (L, w, b)
        if best[1] is not None and (best_cfg is None or best[0] < best_cfg[0]):
            best_cfg = (best[0], best[1], best[2], calib_m)
    if best_cfg is None: w, b = 0.5, 0.0
    else: _, w, b, _ = best_cfg
    return (((1 - w) * f_hm_h + w * f_hw_h) + b).clip(lower=0)

MODELS = ["Naive","SeasonalNaive","HoltWinters","SARIMAX","ASBB"]

# ---------------- per-CWE series builder ----------------
# Each CWE's series is zero-filled only between its own first and last observed month (this
# bounding happens per-CWE inside add_temporal_features above, not here), not across the panel's
# full global range. This is why the number of evaluated CWEs grows across hold-out years as more
# CWEs accumulate at least min_data_points months of history (Section 6, Discussion and Limitations).
def build_series_all_vendors(panel, cwe_id, test_year=2024, min_data_points=12):
    s = (panel[panel["CWE ID"] == cwe_id].groupby("Event Month")["n_events"].sum().asfreq("MS").fillna(0))
    if len(s) < min_data_points: return s, None, None
    train = s[:f'{test_year-1}-12']; test = s[f'{test_year}-01':f'{test_year}-12']
    return s, train, test

def forecasts_fixed_origin(train, steps, snaive_season=6, with_hm=True):
    out = {"Naive": fc_naive(train, steps), "SeasonalNaive": fc_snaive(train, steps, season=snaive_season),
           "ASBB": fc_asbb(train, steps)}
    try: out["HoltWinters"] = fc_holt_winters(train, steps)
    except Exception: pass
    try: out["SARIMAX"] = fc_sarimax(train, steps)
    except Exception: pass
    if with_hm: out["HistMean"] = fc_histmean(train, steps)
    return out

def score(fc, test):
    fc, y = fc.align(test, join="inner")
    if len(y) == 0: return np.nan, np.nan
    e = y.values - fc.values
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e**2)))
