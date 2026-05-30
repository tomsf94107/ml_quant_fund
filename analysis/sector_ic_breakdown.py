"""Per-sector rank-IC decomposition. Tests whether signal concentrates in
certain sectors (your 'tech is better' instinct) in CURRENT rank-IC terms,
not the old AUC framing. Reuses SECTOR_ETF_MAP to tag each name.

For each sector: per-date rank-IC computed WITHIN that sector's names only,
then summarized (median IC, t-stat, n_dates, n_names). A sector with t>3 and
enough names is a concentrate-here candidate; t~0 sectors are dilution."""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from features.builder import build_feature_dataframe, SECTOR_ETF_MAP
from models.classifier import FEATURE_COLUMNS, XGB_PARAMS
from analysis.walk_forward import purged_kfold_indices
from xgboost import XGBClassifier

def build_pooled(tickers, horizon=5):
    dfs=[]
    for i,tk in enumerate(tickers,1):
        if i%50==0: print(f"  [{i}/{len(tickers)}]",flush=True)
        try:
            df=build_feature_dataframe(tk,start_date="2020-01-01",training_mode=True)
            if df.empty or len(df)<200: continue
            df=df.copy()
            df["actual_return"]=df["close"].shift(-horizon)/df["close"]-1.0
            df["actual_up"]=(df["actual_return"]>0).astype(int)
            df["prediction_date"]=pd.to_datetime(df["date"]); df["ticker"]=tk
            df["sector"]=SECTOR_ETF_MAP.get(tk,"XLK")
            dfs.append(df)
        except Exception: pass
    return pd.concat(dfs,ignore_index=True).dropna(subset=["actual_return"])

def ic_tstat(sub):
    """per-date rank-IC summary for a subset (one sector)."""
    feat=[c for c in FEATURE_COLUMNS if c in sub.columns]
    X=sub[feat].fillna(0.0).values.astype(np.float32)
    y=sub["actual_up"].astype(int).values
    dates=sub["prediction_date"]; ret=sub["actual_return"].values
    recs=[]
    for tr,te in purged_kfold_indices(dates,n_folds=5,embargo=5):
        if len(np.unique(y[tr]))<2 or len(np.unique(y[te]))<2: continue
        m=XGBClassifier(**XGB_PARAMS); m.fit(X[tr],y[tr])
        p=m.predict_proba(X[te])[:,1]
        g=pd.DataFrame({"date":dates.iloc[te].values,"pred":p,"ret":ret[te]})
        for d,grp in g.groupby("date"):
            if len(grp)>=3 and grp["pred"].nunique()>1 and grp["ret"].nunique()>1:
                ic=grp["pred"].rank().corr(grp["ret"].rank())
                if pd.notna(ic): recs.append(ic)
    if len(recs)<10: return None
    a=np.array(recs); ir=a.mean()/a.std() if a.std()>0 else 0
    return dict(median=np.median(a), t=ir*np.sqrt(len(a)),
                pos=float((a>0).mean()), n_dates=len(a))

tickers=[t.strip().upper() for t in (ROOT/"tickers_expanded.txt").read_text().splitlines()
         if t.strip() and not t.startswith("#")]
print(f"building pooled panel for {len(tickers)} names (warm cache from prior run)...",flush=True)
pooled=build_pooled(tickers,horizon=5)
print(f"pooled: {len(pooled):,} rows, {pooled['ticker'].nunique()} names\n",flush=True)

print(f"{'sector':6s} {'n_names':>7s} {'med_IC':>8s} {'t_stat':>7s} {'pos':>5s} {'n_dates':>7s}")
print("-"*48)
results={}
for sec in sorted(pooled["sector"].unique()):
    sub=pooled[pooled["sector"]==sec]
    nn=sub["ticker"].nunique()
    r=ic_tstat(sub)
    if r:
        results[sec]=r
        flag=" <-- t>2" if r["t"]>2 else (" <-- t>3 STRONG" if r["t"]>3 else "")
        print(f"{sec:6s} {nn:>7d} {r['median']:>+8.4f} {r['t']:>+7.2f} {r['pos']:>5.2f} {r['n_dates']:>7d}{flag}")
    else:
        print(f"{sec:6s} {nn:>7d}  (insufficient)")
print("\nREAD: sectors with t>2-3 = where signal concentrates -> candidates to")
print("concentrate the book. t~0 sectors = dilution (drop from cross-section).")
