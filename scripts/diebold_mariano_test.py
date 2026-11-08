"""Diebold-Mariano test (Harvey-Leybourne-Newbold small-sample correction, h=1) comparing ASBB
against each of the five baselines, per CWE, for every hold-out year. Produces the significance
results reported in Section 5.2 of the paper. Expects outputs/forecasts_{year}.pkl and
outputs/per_cwe_results_{year}.csv to already exist (produced by walk_forward_eval.py).
"""
import pickle, numpy as np, pandas as pd
from scipy import stats

def dm_test(e1, e2, loss='abs', h=1):
    d = np.abs(e1) - np.abs(e2) if loss == 'abs' else e1**2 - e2**2
    T = len(d)
    if T < 4: return np.nan, np.nan
    d_mean = np.mean(d); gamma0 = np.var(d, ddof=1); nw_var = max(gamma0, 1e-10)
    dm_stat = d_mean / np.sqrt(nw_var / T)
    correction = np.sqrt((T + 1 - 2*h + h*(h-1)/T) / T)
    dm_corr = dm_stat * correction
    return dm_corr, 2 * stats.t.sf(np.abs(dm_corr), df=T-1)

BASE = ["Naive","SeasonalNaive","HoltWinters","SARIMAX","HistMean"]

def run(year):
    fcs = pickle.load(open(f"outputs/forecasts_{year}.pkl","rb")); raw = pd.read_csv(f"outputs/per_cwe_results_{year}.csv").set_index("CWE")
    recs = []
    for cwe, o in fcs.items():
        if raw.loc[cwe,"n_train"] < 12 or raw.loc[cwe,"n_test"] < 4: continue
        y = o["y"]; a = o["ASBB"]
        if np.isnan(a).any(): continue
        e_a = y - a
        for b in BASE:
            if np.isnan(o[b]).any(): continue
            e_b = y - o[b]
            for loss in ("abs","sq"):
                dm, p = dm_test(e_a, e_b, loss)
                if np.isnan(dm): continue
                recs.append(dict(year=year, CWE=cwe, Baseline=b, Loss=loss, DM=dm, p=p,
                                 asbb_better=(dm<0) and (p<0.05), base_better=(dm>0) and (p<0.05)))
    return pd.DataFrame(recs)

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    allr = pd.concat([run(y) for y in (2021,2022,2023,2024)])
    allr.to_csv("outputs/dm_results.csv", index=False)
    g = allr.groupby(["year","Baseline","Loss"]).agg(CWEs_tested=("CWE","count"), ASBB_better=("asbb_better","sum"),
                                                       baseline_better=("base_better","sum")).reset_index()
    g["pct_ASBB_better"] = (100*g.ASBB_better/g.CWEs_tested).round(1)
    g["pct_baseline_better"] = (100*g.baseline_better/g.CWEs_tested).round(1)
    g.to_csv("outputs/dm_summary.csv", index=False)
    order = {"Naive":0,"SeasonalNaive":1,"HoltWinters":2,"SARIMAX":3,"HistMean":4}
    g["o"] = g.Baseline.map(order)
    print("=== Diebold-Mariano significance, 2024 ===")
    print(g[g.year==2024].sort_values(["o","Loss"]).drop(columns="o").to_string(index=False))
    print("\n=== all years, abs loss ===")
    print(g[g.Loss=="abs"].sort_values(["year","o"]).drop(columns="o")[["year","Baseline","CWEs_tested","ASBB_better","pct_ASBB_better","baseline_better","pct_baseline_better"]].to_string(index=False))
