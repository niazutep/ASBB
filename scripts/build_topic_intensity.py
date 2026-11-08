"""Builds a monthly, family-level Topic Intensity Index from the fitted topic model (produced by
topic_model_fit.py). For each month, each topic's share of that month's documents is its
proportion (the outlier topic is excluded from the denominator, matching Section 4.8's methodology).
A topic is assigned to a family if its best-cosine CWE match (cosine >= 0.65, the paper's own
threshold) falls in that family. A family's Topic Intensity Index each month is the sum of its
aligned topics' proportions that month. This index is the exogenous regressor tested (and found
not to improve forecast accuracy) in the Section 6 discussion of the semantic signal as a predictor.
"""
import os, pickle, pandas as pd, numpy as np
DATA_DIR = os.environ.get("ASBB_DATA_DIR", "./data")
fam = pd.read_csv(os.path.join(DATA_DIR, "family", "cwe_auto_family_mapping.csv")).set_index("CWE ID")["family"]
ref = pickle.load(open("outputs/topic_model_fit.pkl", "rb"))

doc = pd.DataFrame({"topic": ref["topics"], "month": ref["months"]})
doc = doc[doc.topic != -1]  # exclude the outlier cluster, as in Section 4.8
month_totals = doc.groupby("month").size()  # denominator: all non-outlier docs that month
prop = doc.groupby(["month", "topic"]).size().unstack(fill_value=0)
prop = prop.div(month_totals, axis=0)  # monthly proportion per topic

best = ref["best"]; best = best[best.cosine >= 0.65]
topic_to_family = {}
for _, r in best.iterrows():
    f = fam.get(r.CWE_ID)
    if f: topic_to_family.setdefault(r.topic_id, f)  # a topic's single best-match family
print(f"{len(prop.columns)} topics with monthly proportions; {len(topic_to_family)} aligned (cosine>=0.65) to a family")

idx = pd.date_range("2017-01-01", "2025-09-01", freq="MS")
prop.index = pd.to_datetime(prop.index); prop = prop.reindex(idx, fill_value=0.0)

families = sorted(set(topic_to_family.values()))
T = pd.DataFrame(index=idx, columns=families, dtype=float).fillna(0.0)
for topic, f in topic_to_family.items():
    if topic in prop.columns: T[f] += prop[topic]
os.makedirs("outputs", exist_ok=True)
T.to_csv("outputs/topic_intensity_by_family.csv")
print(f"{len(families)} families with a topic-intensity series:", families)
print(T.describe().T[["mean", "std", "max"]].round(4))
