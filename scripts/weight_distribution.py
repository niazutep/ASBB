"""Reports the distribution of ASBB's selected blend weight w across every forecast origin.
Instruments fc_asbb to also return the selected w (the main evaluation loop discards it after
scoring), and reruns the same walk-forward loop as walk_forward_eval.py, but only for ASBB, to
collect w at every forecast origin. Also sanity-checks the resulting MAE against the already
published per-CWE results to confirm this instrumented run reproduces them exactly.
"""
import sys, os, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asbb_eval as A

def fc_asbb_with_w(train, steps, season=12, calib_months_grid=(6, 12, 18, 24), bias_cap=None):
    """Same selection logic as A.fc_asbb, modified only to also return the selected w."""
    y = train.asfreq("MS").fillna(0)
    if bias_cap is None: bias_cap = max(1.0, 0.1 * float(y.mean()))
    f_hm_h = A.fc_histmean(y, steps=steps)
    try: f_hw_h = A.fc_holt_winters(y, steps=steps)
    except Exception: return f_hm_h.clip(lower=0), np.nan
    T = len(y); min_start = season
    valid_grid = [m for m in calib_months_grid if T >= (min_start + m + 1)]
    if not valid_grid: return (0.5 * f_hm_h + 0.5 * f_hw_h).clip(lower=0), 0.5
    best_cfg = None
    for calib_m in valid_grid:
        start_idx = T - (calib_m + 1)
        pred_hm, pred_hw = A._rolling_preds_fast_hm(y, season=season, start_idx=start_idx)
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
    return (((1 - w) * f_hm_h + w * f_hw_h) + b).clip(lower=0), w

panel = A.load_panel()
cwes = (panel.groupby("CWE ID")["n_events"].sum().sort_values(ascending=False).index.tolist())
YEARS = [2021, 2022, 2023, 2024]

def work(args):
    cwe, year = args
    s, train, test = A.build_series_all_vendors(panel, cwe, year)
    if train is None or test is None or len(train) == 0 or len(test) == 0: return None
    fc_vals, ws = [], []
    for t in test.index:
        hist = s[:t - pd.offsets.MonthBegin()]
        try:
            fc, w = fc_asbb_with_w(hist, 1)
            fc_vals.append(float(fc.iloc[0])); ws.append(w)
        except Exception:
            fc_vals.append(np.nan); ws.append(np.nan)
    e = test.values - np.array(fc_vals)
    mae = float(np.mean(np.abs(e))) if not np.isnan(e).any() else np.nan
    return {"CWE": cwe, "year": year, "n_test": len(test), "MAE": mae, "w_values": ws}

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    t0 = time.time()
    rows = []
    with Pool(24) as p:
        for y in YEARS:
            out = [r for r in p.map(work, [(c, y) for c in cwes], chunksize=1) if r]
            rows.extend(out)
            print(y, "done, elapsed", round(time.time() - t0), "s", flush=True)
    df = pd.DataFrame(rows)
    df.to_json("outputs/weight_distribution_raw.json", orient="records", indent=1)

    # restrict to the same CWE-years used in the paper's reported tables (excludes CWE-years where
    # SARIMAX's own fit was unstable; see outputs/per_cwe_results_{year}.csv's SARIMAX_MAE/RMSE)
    frames = []
    for y in YEARS:
        official = pd.read_csv(f"outputs/per_cwe_results_{y}.csv")
        official = official[~((official.SARIMAX_MAE > 100) | (official.SARIMAX_RMSE > 100))]
        keep = set(official.CWE)
        frames.append(df[(df.year == y) & (df.CWE.isin(keep))])
    filtered = pd.concat(frames)
    all_w = np.array([w for row in filtered.w_values for w in row if w is not None and not (isinstance(w, float) and np.isnan(w))])
    out = {
        "n_forecast_origins": int(len(all_w)),
        "w_mean": round(float(all_w.mean()), 3),
        "w_median": round(float(np.median(all_w)), 3),
        "pct_w_below_0.1": round(100 * float((all_w < 0.1).mean()), 1),
        "pct_w_below_0.3": round(100 * float((all_w < 0.3).mean()), 1),
        "pct_w_above_0.7": round(100 * float((all_w > 0.7).mean()), 1),
        "pct_w_above_0.9": round(100 * float((all_w > 0.9).mean()), 1),
    }
    check = {}
    for y in YEARS:
        pub = pd.read_csv(f"outputs/per_cwe_results_{y}.csv")[["CWE", "ASBB_MAE"]]
        mine = df[df.year == y][["CWE", "MAE"]].rename(columns={"MAE": "ASBB_MAE_rerun"})
        m = pub.merge(mine, on="CWE", how="inner")
        check[y] = {"n_matched": len(m), "max_abs_diff": round(float((m.ASBB_MAE - m.ASBB_MAE_rerun).abs().max()), 6)}
    out["sanity_check_vs_published_MAE"] = check
    json.dump(out, open("outputs/weight_distribution_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("DONE", round(time.time() - t0), "s total")
