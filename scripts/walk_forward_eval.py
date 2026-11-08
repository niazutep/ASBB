"""True one-step-ahead walk-forward evaluation: for each eligible CWE and each hold-out year,
refit every model every month on data strictly before that month, and score the one-step forecast
against the actual value. Produces the per-CWE results used to build Tables 2-5 of the paper.

Writes outputs/per_cwe_results_{year}.csv (per-CWE MAE/RMSE for every model) and
outputs/forecasts_{year}.pkl (the raw per-month forecast values) for each hold-out year.
"""
import sys, os, time, pickle
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asbb_eval as A

panel = A.load_panel()
cwes = (panel.groupby("CWE ID")["n_events"].sum().sort_values(ascending=False).index.tolist())
YEARS = [2021, 2022, 2023, 2024]
ALL = ["Naive","SeasonalNaive","HoltWinters","SARIMAX","ASBB","HistMean"]

def one_step(hist):
    out = {}
    out["Naive"] = A.fc_naive(hist,1)
    out["SeasonalNaive"] = A.fc_snaive(hist,1,season=12)
    out["HistMean"] = A.fc_histmean(hist,1)
    for name, fn in (("HoltWinters", A.fc_holt_winters), ("SARIMAX", A.fc_sarimax), ("ASBB", A.fc_asbb)):
        try: out[name] = fn(hist,1)
        except Exception: pass
    return out

def work(args):
    cwe, year = args
    s, train, test = A.build_series_all_vendors(panel, cwe, year)
    if train is None or test is None or len(train) == 0 or len(test) == 0: return None
    rows = {m: [] for m in ALL}
    for t in test.index:
        hist = s[:t - pd.offsets.MonthBegin()]
        o = one_step(hist)
        for m in ALL: rows[m].append(float(o[m].iloc[0]) if m in o else np.nan)
    fc = {m: pd.Series(rows[m], index=test.index) for m in ALL}
    row = {"CWE": cwe, "n_train": len(train), "n_test": len(test), "train_mean": float(train.mean()), "nz_train": float((train>0).mean())}
    for m in ALL:
        e = test.values - fc[m].values
        if np.isnan(e).any(): row[m+"_MAE"] = row[m+"_RMSE"] = np.nan
        else: row[m+"_MAE"] = float(np.mean(np.abs(e))); row[m+"_RMSE"] = float(np.sqrt(np.mean(e**2)))
    return row, {"y": test.values, "idx": test.index, **{m: fc[m].values for m in ALL}}

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    t0 = time.time()
    with Pool(24) as p:
        for y in YEARS:
            out = [r for r in p.map(work, [(c,y) for c in cwes], chunksize=1) if r]
            df = pd.DataFrame([r[0] for r in out])
            df.to_csv(f"outputs/per_cwe_results_{y}.csv", index=False)
            pickle.dump({r[0]["CWE"]: r[1] for r in out}, open(f"outputs/forecasts_{y}.pkl","wb"))
            print(y, "CWEs", len(df), "elapsed", round(time.time()-t0), "s", flush=True)
    print("DONE", round(time.time()-t0), "s total", flush=True)
