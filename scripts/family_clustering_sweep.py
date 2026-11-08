"""Sweeps HDBSCAN's min_cluster_size parameter to select the automated CWE-family clustering
(Section 4.3 of the paper). CWEs are embedded from their MITRE name, description, and extended
description, reduced with UMAP, and clustered with HDBSCAN; the configuration with the highest
silhouette score among those assigning at least 70% of CWEs is selected (min_cluster_size=9,
yielding 15 families in the paper).
"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import umap, hdbscan, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
DATA_DIR = os.environ.get("ASBB_DATA_DIR", "./data_raw")
D = DATA_DIR if DATA_DIR.endswith("/") else DATA_DIR + "/"
vuln_df_ot_all=pd.read_parquet(D+"vuln_df_ot_all.parquet")
cwe_df=(vuln_df_ot_all[["CWE ID","CWE Name"]].dropna(subset=["CWE Name"]).drop_duplicates().reset_index(drop=True))
cwe_df=cwe_df[(cwe_df["CWE Name"].astype(str).str.strip()!="")&(~cwe_df["CWE Name"].astype(str).str.upper().isin(["N/A","NONE"]))].reset_index(drop=True)
m=pd.read_csv(D+"CWE_0915.csv"); m["CWE ID"]=m["CWE-ID"].astype(str).map(lambda x:x if x.startswith("CWE-") else f"CWE-{x}")
mitre_df=m.rename(columns={"Name":"MITRE_Name","Description":"MITRE_Description","Extended Description":"MITRE_Extended"})[["CWE ID","MITRE_Name","MITRE_Description","MITRE_Extended"]]
cwe_enriched=cwe_df.merge(mitre_df,on="CWE ID",how="left")
def build_embed_text(row):
    parts=[]; name=row["MITRE_Name"] if pd.notna(row["MITRE_Name"]) else row["CWE Name"]; parts.append(f"{row['CWE ID']} {name}")
    desc=row.get("MITRE_Description","")
    if pd.notna(desc) and len(str(desc).strip())>10: parts.append(str(desc).strip())
    ext=row.get("MITRE_Extended","")
    if pd.notna(ext) and len(str(ext).strip())>10: parts.append(str(ext).strip()[:300])
    return " ".join(parts)
cwe_enriched["embed_text"]=cwe_enriched.apply(build_embed_text,axis=1)
print("CWEs:",len(cwe_enriched),"| with MITRE description:",int((cwe_enriched.MITRE_Description.fillna("").str.len()>10).sum()))
sbert=SentenceTransformer("paraphrase-mpnet-base-v2")
emb=sbert.encode(cwe_enriched["embed_text"].tolist(),normalize_embeddings=True,batch_size=64,show_progress_bar=False)
cwe_umap=umap.UMAP(n_neighbors=15,n_components=5,min_dist=0.0,metric="cosine",random_state=42).fit_transform(emb)
rows=[]
for mcs in range(5,30):
    np.random.seed(42)
    labels=hdbscan.HDBSCAN(min_cluster_size=mcs,min_samples=3,metric="euclidean",cluster_selection_method="eom").fit_predict(cwe_umap)
    n_clusters=len(set(labels))-(1 if -1 in labels else 0); pct=(labels!=-1).sum()/len(labels)*100
    assigned=set(cwe_enriched["CWE ID"][labels!=-1])
    cov=len(vuln_df_ot_all[vuln_df_ot_all["CWE ID"].isin(assigned)])/len(vuln_df_ot_all[vuln_df_ot_all["CWE ID"].isin(cwe_enriched["CWE ID"])])*100
    mask=labels!=-1; sil=silhouette_score(cwe_umap[mask],labels[mask]) if (n_clusters>=3 and mask.sum()>n_clusters) else -1
    rows.append(dict(min_cluster_size=mcs,n_clusters=n_clusters,pct_assigned=pct,adv_coverage=cov,silhouette=sil))
os.makedirs("outputs", exist_ok=True)
R=pd.DataFrame(rows); R.to_csv("outputs/family_clustering_sweep_results.csv",index=False)
viable=R[(R.silhouette>0)&(R.pct_assigned>=70)]; best=viable.loc[viable.silhouette.idxmax()]
print(R.round(3).to_string(index=False)); print("\nSELECTED:",best.round(3).to_dict())
print("PAPER   : mcs=9, 15 families, silhouette 0.556, 84.4% CWEs assigned, 95.3% advisory coverage")
