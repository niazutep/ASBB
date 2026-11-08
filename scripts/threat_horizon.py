"""Builds the 60-month threat-horizon figure (Fig. 4 in the paper): ASBB is fit to each family's
full monthly series through 2023 and projected five years ahead, with split-conformal prediction
bands. Also writes the year-ahead comparison against January-September 2025 actuals used in the
Section 6 discussion of the long-horizon projection's accuracy.
"""
import sys, os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asbb_eval as A
DATA_DIR = os.environ.get("ASBB_DATA_DIR", "./data")
fam = pd.read_csv(os.path.join(DATA_DIR, "family", "cwe_auto_family_mapping.csv")).set_index("CWE ID")["family"]
panel = A.load_panel(); panel["family"] = panel["CWE ID"].map(fam).fillna("Other")

def split_conformal_band(train_series, model_fc_func, horizon=12, calib_years=("2019-01","2023-12"), alpha=0.2):
    s = train_series.asfreq("MS").fillna(0)
    calib_start, calib_end = pd.to_datetime(calib_years[0]), pd.to_datetime(calib_years[1])
    if len(s) < 24: raise ValueError("Too few observations (<24 months) for split-conformal setup.")
    pre_calib = s[:(calib_start - pd.offsets.MonthBegin())]
    steps_calib = (calib_end.to_period("M") - calib_start.to_period("M")).n + 1
    try: fc_calib = model_fc_func(pre_calib, steps_calib)
    except Exception: fc_calib = A.fc_histmean(pre_calib, steps_calib)
    y_calib = s[calib_start:calib_end]
    resid = (y_calib - fc_calib.reindex(y_calib.index).ffill()).abs(); q = resid.quantile(1 - alpha)
    try: fc_final = model_fc_func(s, horizon)
    except Exception: fc_final = A.fc_histmean(s, horizon)
    return fc_final, (fc_final - q).clip(lower=0), fc_final + q, float(q)

def fc_asbb_multi(train, steps, season=12, calib_months_grid=(6, 12, 18, 24), bias_cap=None):
    """ASBB extended to horizons beyond one month: (1-w)*HistMean + w*HoltWinters + b. The
    historical mean needs no special multi-horizon handling, since it is a flat forecast at any
    horizon."""
    y = train.asfreq("MS").fillna(0)
    if bias_cap is None: bias_cap = max(1.0, 0.1 * float(y.mean()))
    f_hm_h = A.fc_histmean(y, steps)
    try: f_hw_h = A.fc_holt_winters(y, steps=steps)
    except Exception: return f_hm_h.clip(lower=0)
    T = len(y); valid_grid = [m for m in calib_months_grid if T >= (season + m + 1)]
    if not valid_grid: return (0.5 * f_hm_h + 0.5 * f_hw_h).clip(lower=0)
    best_cfg = None
    for calib_m in valid_grid:
        pred_hm, pred_hw = A._rolling_preds_fast_hm(y, season=season, start_idx=T - (calib_m + 1))
        dfc = pd.DataFrame({"y": y.loc[pred_hm.index], "hm": pred_hm, "hw": pred_hw}).dropna()
        if len(dfc) < max(6, season // 2): continue
        best = (np.inf, None, None)
        for w in np.linspace(0.0, 1.0, 51):
            yhat = (1 - w) * dfc["hm"] + w * dfc["hw"]; b = float((dfc["y"] - yhat).median()); b = max(-bias_cap, min(bias_cap, b))
            L = np.mean(np.abs(dfc["y"].values - (yhat + b).values))
            if L < best[0]: best = (L, w, b)
        if best[1] is not None and (best_cfg is None or best[0] < best_cfg[0]): best_cfg = (best[0], best[1], best[2], calib_m)
    w, b = (0.5, 0.0) if best_cfg is None else (best_cfg[1], best_cfg[2])
    return (((1 - w) * f_hm_h + w * f_hw_h) + b).clip(lower=0)

months = pd.date_range("2017-01-01", "2025-09-01", freq="MS")
fs = panel[panel.family != "Other"].groupby(["family", "Event Month"]).n_events.sum().unstack(0).reindex(months).fillna(0)
top10 = fs.loc["2017":"2024"].sum().sort_values(ascending=False).head(10).index.tolist()
res = {}
for f in top10:
    s = fs[f]; train = s[:"2023-12"]
    fc, lo, hi, q = split_conformal_band(train, fc_asbb_multi, horizon=60)
    ann = lambda x: x.groupby(x.index.year).sum()
    hist = ann(train); act24 = float(s["2024-01":"2024-12"].sum())
    ytd_a = float(s['2025-01':'2025-09'].sum()); ytd_f = float(fc['2025-01':'2025-09'].sum()); ytd_naive = float(s['2024-01':'2024-09'].sum())
    res[f] = dict(ytd_act=ytd_a, ytd_fc=ytd_f, ytd_prev=ytd_naive, hist=hist.to_dict(), act24=act24,
                  fc=ann(fc).to_dict(), lo=ann(lo).to_dict(), hi=ann(hi).to_dict(), q=q)
rank = sorted(top10, key=lambda f: res[f]["fc"][2028], reverse=True); top3 = rank[:3]
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "xtick.labelsize": 7,
                      "ytick.labelsize": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42})
HC = "#2C5F8A"; FCC = "#C0392B"; BC = "#F1948A"; AC = "#117A65"; WB = "#FEF9E7"
fig, axes = plt.subplots(2, 5, figsize=(7.1, 3.5)); plt.subplots_adjust(hspace=0.62, wspace=0.42, left=0.06, right=0.995, top=0.9, bottom=0.24)
short = {"Authentication & Credentials": "Auth. &\nCredentials", "Injection & Neutralization": "Injection &\nNeutral.",
         "Access Control & Permissions": "Access Ctrl. &\nPerms.", "Information Disclosure": "Info.\nDisclosure",
         "Path & File Operations": "Path & File\nOps.", "Cross-boundary Communication": "Cross-boundary\nComm.",
         "Concurrency & Synchronization": "Concurrency\n& Sync.", "Resource Lifecycle": "Resource\nLifecycle",
         "Memory Safety": "Memory\nSafety", "Input Validation": "Input\nValidation", "Code Quality": "Code\nQuality"}
for ax, f in zip(axes.flat, rank):
    r = res[f]; hy = sorted(r["hist"])
    if f in top3:
        ax.set_facecolor(WB)
        for sp in ax.spines.values(): sp.set_edgecolor("#E74C3C"); sp.set_linewidth(1.2)
    ax.plot(hy, [r["hist"][y] for y in hy], color=HC, lw=1.3, marker="o", ms=2.2, zorder=3)
    ax.plot([2023, 2024], [r["hist"][2023], r["act24"]], color=HC, lw=1.0, zorder=2); ax.plot(2024, r["act24"], color=AC, marker="D", ms=3.5, ls="", zorder=5)
    fy = [2025, 2026, 2027, 2028]; ax.plot([2024] + fy, [r["act24"]] + [r["fc"][y] for y in fy], color=FCC, lw=1.2, ls="--", marker="o", ms=2, zorder=4)
    ax.fill_between(fy, [r["lo"][y] for y in fy], [r["hi"][y] for y in fy], color=BC, alpha=0.35, zorder=1)
    ax.plot([2024], [r["fc"][2024]], color=FCC, marker="x", ms=3.5, ls="", zorder=5)
    ax.axvline(2024.5, color="grey", lw=0.6, ls=":", alpha=0.6)
    if f in top3: ax.annotate(f"#{top3.index(f)+1}", xy=(0.96, 0.93), xycoords="axes fraction", ha="right", va="top", fontsize=7, color="#C0392B", fontweight="bold")
    ax.set_title(short.get(f, f), fontsize=7.5, fontweight="bold", pad=2); ax.set_xlim(2016.5, 2029); ax.set_xticks([2018, 2023, 2028]); ax.set_xticklabels(["2018", "2023", "2028"])
    ax.set_ylim(bottom=0); ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=3)); ax.grid(axis="y", alpha=0.25, lw=0.4)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
h = [plt.Line2D([], [], color=HC, marker="o", ms=2.5, lw=1.3, label="History 2017-2023"), plt.Line2D([], [], color=AC, marker="D", ms=3.5, ls="", label="2024 actual"),
     plt.Line2D([], [], color=FCC, marker="x", ms=3.5, ls="", label="ASBB forecast for 2024"),
     plt.Line2D([], [], color=FCC, ls="--", lw=1.2, label="ASBB forecast 2025-2028"), mp.Patch(color=BC, alpha=0.5, label="80% conformal band (summed monthly)"),
     mp.Patch(facecolor=WB, edgecolor="#E74C3C", label="Top 3 by 2028 forecast")]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.03))
os.makedirs("outputs/figs", exist_ok=True)
fig.savefig("outputs/figs/threat_horizon.pdf", bbox_inches="tight"); plt.close(fig)
out = {"top10_order_by_2028_forecast": rank, "top3": top3,
       "families": {f: {"hist_2023": res[f]["hist"][2023], "actual_2024": res[f]["act24"], "asbb_fc_2024": round(res[f]["fc"][2024], 1),
                         "fc_2025": round(res[f]["fc"][2025], 1), "fc_2028": round(res[f]["fc"][2028], 1), "lo_2028": round(res[f]["lo"][2028], 1),
                         "hi_2028": round(res[f]["hi"][2028], 1), "jan_sep_2025_actual": res[f]["ytd_act"],
                         "jan_sep_2025_asbb_forecast": round(res[f]["ytd_fc"], 1), "jan_sep_2024_actual_repeat_last_year_baseline": res[f]["ytd_prev"]} for f in rank}}
json.dump(out, open("outputs/threat_horizon_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
