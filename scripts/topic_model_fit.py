"""Fits BERTopic on the advisory text corpus and aligns each topic to its highest-cosine CWE
match, producing the topic-CWE pairs analyzed in Section 5.5 of the paper. Encoder:
paraphrase-mpnet-base-v2. UMAP(random_state=42), HDBSCAN(min_cluster_size=15, min_samples=5).
Saves everything needed to rebuild the paper's topic-evolution and co-evolution figures.

Requires bertopic, sentence-transformers, umap-learn, hdbscan, and torch to be installed
(pip install bertopic sentence-transformers umap-learn hdbscan torch).
"""
import os, sys, pickle, json, numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer, util
import umap, hdbscan
from bertopic import BERTopic
DATA_DIR = os.environ.get("ASBB_DATA_DIR", "./data_raw")
D = DATA_DIR if DATA_DIR.endswith("/") else DATA_DIR + "/"
vuln_df_ot_all=pd.read_parquet(D+"vuln_df_ot_all.parquet")
def build_advisory_corpus(vuln_df):
    df = vuln_df.copy()
    for col in ["Advisory Release Date", "Vuln Release Date"]:
        if col in df.columns: df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    df["Event Date"] = df["Advisory Release Date"].fillna(df.get("Vuln Release Date"))
    df["Year"] = df["Event Date"].dt.year
    text_fields = []
    for c in ["Advisory Title", "Vuln Title", "CWE Name"]:
        if c in df.columns: text_fields.append(df[c].fillna(""))
    text = pd.Series([""] * len(df))
    for f in text_fields: text = (text + " " + f.astype(str)).str.strip()
    corpus = pd.DataFrame({"advisory_id": df.get("Advisory ID", pd.Series(np.arange(len(df))).astype(str)), "year": df["Year"],
        "month": df["Event Date"].dt.to_period("M").astype(str), "text": text.str.replace(r"\s+", " ", regex=True)}).dropna(subset=["year", "text"])
    return corpus[corpus["text"].str.len() > 15]
corpus=build_advisory_corpus(vuln_df_ot_all); print("Corpus size:",len(corpus))
sbert=SentenceTransformer("paraphrase-mpnet-base-v2")
umap_model=umap.UMAP(n_neighbors=15,n_components=5,min_dist=0.0,metric="cosine",random_state=42)
hdb_model=hdbscan.HDBSCAN(min_cluster_size=15,min_samples=5,metric="euclidean",cluster_selection_method="eom",prediction_data=True)
topic_model=BERTopic(embedding_model=sbert,umap_model=umap_model,hdbscan_model=hdb_model,top_n_words=12,language="english",calculate_probabilities=True,verbose=False)
docs=corpus["text"].tolist(); years=corpus["year"].astype(int).tolist(); months=corpus["month"].tolist()
topics,probs=topic_model.fit_transform(docs)
print("topics found:",len(set(topics))-1,"| outliers:",sum(1 for t in topics if t==-1))
# Topic-CWE alignment
cwe_df=(vuln_df_ot_all[["CWE ID","CWE Name"]].dropna(subset=["CWE Name"]).drop_duplicates().reset_index(drop=True))
cwe_df=cwe_df[(cwe_df["CWE Name"].astype(str).str.strip()!="")&(~cwe_df["CWE Name"].astype(str).str.upper().isin(["N/A","NONE"]))].reset_index(drop=True)
cwe_df["cwe_text"]=cwe_df["CWE ID"].fillna("")+" "+cwe_df["CWE Name"]
topic_rows=[{"topic_id":t,"topic_text":" ".join(w for w,_ in ws)} for t,ws in topic_model.get_topics().items() if t!=-1]
topic_df=pd.DataFrame(topic_rows).sort_values("topic_id").reset_index(drop=True)
topic_emb=sbert.encode(topic_df["topic_text"].tolist(),normalize_embeddings=True); cwe_emb=sbert.encode(cwe_df["cwe_text"].tolist(),normalize_embeddings=True)
sim=util.cos_sim(torch.tensor(topic_emb),torch.tensor(cwe_emb)); recs=[]
for i,row in topic_df.iterrows():
    vals,idxs=torch.topk(sim[i],k=3)
    for rank,(v,j) in enumerate(zip(vals.tolist(),idxs.tolist()),start=1):
        recs.append({"topic_id":row["topic_id"],"topic_text":row["topic_text"],"rank":rank,"cosine":float(v),"CWE_ID":cwe_df.iloc[j]["CWE ID"],"CWE_Name":cwe_df.iloc[j]["CWE Name"]})
matched_df=pd.DataFrame(recs).sort_values(["topic_id","rank"]).reset_index(drop=True)
best=matched_df.drop_duplicates(subset=["topic_id"],keep="first").copy()
# Year-topic prevalence (excludes the outlier cluster and the four largest topics, Section 4.8)
ta=pd.DataFrame({"topic":topics,"year":years}); yt=ta[ta.topic>3].groupby(["year","topic"]).size().reset_index(name="count")
yt=yt.merge(yt.groupby("year")["count"].sum().rename("year_total"),on="year"); yt["prop"]=yt["count"]/yt["year_total"]
os.makedirs("outputs", exist_ok=True)
pickle.dump({"topics":topics,"years":years,"months":months,"best":best,"matched_df":matched_df,"year_topic":yt,"topic_words":topic_model.get_topics(),"corpus_n":len(corpus)},open("outputs/topic_model_fit.pkl","wb"))
# Reproduction check against the paper's saved topic-CWE pairs
ref=pd.read_csv(os.path.join(os.environ.get("ASBB_DATA_DIR", "./data"), "topic_pairs", "topic_cwe_xcorr_permtest.csv"))
m=ref.merge(best[["topic_id","CWE_ID","cosine"]].rename(columns={"topic_id":"topic","CWE_ID":"cwe_new","cosine":"cos_new"}),on="topic",how="left")
same=(m.best_cwe==m.cwe_new)&((m.cosine-m.cos_new).abs()<0.01)
print(f"REPRODUCTION: {int(same.sum())}/{len(ref)} of the paper's topic-CWE pairs reappear with the same topic id, CWE and cosine (within 0.01)")
print(f"  same topic id and CWE: {int((m.best_cwe==m.cwe_new).sum())}/{len(ref)}")
print(m[m.bh_significant][["topic","best_cwe","cosine","cwe_new","cos_new"]].round(3).to_string(index=False))
