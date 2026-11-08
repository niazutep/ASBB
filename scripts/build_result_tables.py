"""Builds the paper's main result tables and statistics from the per-CWE walk-forward results:
the multi-year macro table, the 2024 per-CWE table, the family-level breakdown, the Forecastability
Index, and the Seasonal-Naive tie rate. Expects outputs/per_cwe_results_{year}.csv to already exist
(produced by walk_forward_eval.py), and the family mapping in data/family/cwe_auto_family_mapping.csv.
"""
import pandas as pd, numpy as np, json, os

DATA_DIR = os.environ.get("ASBB_DATA_DIR", "./data")
fam = pd.read_csv(os.path.join(DATA_DIR, "family", "cwe_auto_family_mapping.csv")).set_index("CWE ID")["family"]
M5 = ["Naive","SeasonalNaive","HoltWinters","SARIMAX","ASBB"]; M6 = M5+["HistMean"]
OUT = {}

def load(y):
    d = pd.read_csv(f"outputs/per_cwe_results_{y}.csv")
    return d[~((d.SARIMAX_MAE>100)|(d.SARIMAX_RMSE>100))].copy()

rows = []
for y in (2021,2022,2023,2024):
    k = load(y); r = {"year":y, "CWEs":len(k)}
    for m in M6: r[m+"_MAE"] = k[m+"_MAE"].mean(); r[m+"_RMSE"] = k[m+"_RMSE"].mean()
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv("outputs/multiyear_macro_table.csv", index=False)
print("=== Multi-year macro table ===")
print(T.round(3).to_string(index=False))
OUT["hist_lower_mae_years"] = [int(r.year) for _,r in T.iterrows() if r.HistMean_MAE<r.ASBB_MAE]
OUT["hist_lower_rmse_years"] = [int(r.year) for _,r in T.iterrows() if r.HistMean_RMSE<r.ASBB_RMSE]

k = load(2024); k.to_csv("outputs/per_cwe_2024.csv", index=False)
OUT["table_2024"] = {m: [round(k[m+"_MAE"].mean(),3), round(k[m+"_RMSE"].mean(),3)] for m in M6}
OUT["n_cwe_2024"] = len(k)
print("\n=== 2024 per-CWE macro table ===")
for m in M6: print(f"  {m:15s} MAE {k[m+'_MAE'].mean():.3f}  RMSE {k[m+'_RMSE'].mean():.3f}")

# CWE-by-CWE ASBB vs HistMean
for y in (2021,2022,2023,2024):
    kk = load(y)
    a_better = int((kk.ASBB_MAE < kk.HistMean_MAE).sum()); h_better = int((kk.HistMean_MAE < kk.ASBB_MAE).sum())
    print(f"  {y}: ASBB lower MAE on {a_better} CWEs, HistMean lower on {h_better} CWEs (of {len(kk)})")

# Family-level breakdown, 2024
k["family"] = k.CWE.map(fam).fillna("Other")
cols = [m+"_MAE" for m in M6]
F = k.groupby("family")[cols].mean(); F["n"] = k.groupby("family").size(); F = F.drop(index="Other", errors="ignore")
F.to_csv("outputs/family_level_2024.csv")
print("\n=== Family-level breakdown, 2024 ===")
print(F.round(3).to_string())
wins_asbb_of5 = sum(1 for f,r in F.iterrows() if r["ASBB_MAE"] == min(r[m+"_MAE"] for m in M5))
wins_hist_over_asbb = sum(1 for f,r in F.iterrows() if r["HistMean_MAE"] < r["ASBB_MAE"])
OUT["families_asbb_lowest_of5"] = wins_asbb_of5; OUT["n_families"] = len(F)
OUT["families_hist_lower_than_asbb"] = wins_hist_over_asbb
print(f"\nASBB lowest of five models in {wins_asbb_of5} of {len(F)} families")
print(f"HistMean lower than ASBB in {wins_hist_over_asbb} of {len(F)} families")
print("macro over families:", {m: round(float(F[m+'_MAE'].mean()),3) for m in M6})

# Tie rate with Seasonal-Naive
tie = (k.ASBB_MAE.round(6) == k.SeasonalNaive_MAE.round(6)).mean()
print(f"\nASBB exactly ties Seasonal-Naive MAE on {100*tie:.1f}% of CWEs")
OUT["sn_tie_rate_2024"] = round(float(tie),3)

# Forecastability Index (FI), 2024, rank-percentile over HW and ASBB improvement vs Seasonal-Naive
d = k.copy()
with np.errstate(divide="ignore", invalid="ignore"):
    d["dHW"] = 100*(d.SeasonalNaive_MAE - d.HoltWinters_MAE)/d.SeasonalNaive_MAE
    d["dASBB"] = 100*(d.SeasonalNaive_MAE - d.ASBB_MAE)/d.SeasonalNaive_MAE
rn = lambda x: (x.rank()-1)/(len(x.dropna())-1) if x.notna().any() else x*0
d["FI"] = pd.concat([rn(d.dHW), rn(d.dASBB)], axis=1).mean(axis=1)
d = d.sort_values("FI", ascending=False); d.to_csv("outputs/forecastability_index_2024.csv", index=False)
OUT["fi_top25"] = d.CWE.head(25).tolist()
OUT["fi_gt08_n"] = int((d.FI>0.8).sum()); OUT["fi_mean"] = round(float(d.FI.mean()),3)
print(f"\nFI: mean {OUT['fi_mean']}, {OUT['fi_gt08_n']} CWEs above 0.8")
print("top 25 by FI:", OUT["fi_top25"][:10], "...")

json.dump(OUT, open("outputs/result_tables_summary.json","w"), indent=1, default=str)
