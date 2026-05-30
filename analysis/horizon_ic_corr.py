"""Test whether pooling horizons is REAL breadth or redundancy.
Re-scores h3 and h5, extracts per-date rank-IC, correlates the two series.
corr <0.5 = independent bets (real breadth). corr >0.8 = same bet (fake breadth)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from features.builder import build_feature_dataframe
from models.classifier import FEATURE_COLUMNS, XGB_PARAMS
from analysis.walk_forward import purged_kfold_indices
from xgboost import XGBClassifier

def per_date_ic(tickers, horizon, n_folds=5, embargo=5):
    dfs=[]
    for tk in tickers:
        try:
            df=build_feature_dataframe(tk, start_date="2020-01-01", training_mode=True)
            if df.empty or len(df)<200: continue
            df=df.copy()
            df["actual_return"]=df["close"].shift(-horizon)/df["close"]-1.0
            df["actual_up"]=(df["actual_return"]>0).astype(int)
            df["prediction_date"]=pd.to_datetime(df["date"]); df["ticker"]=tk
            dfs.append(df)
        except Exception: pass
    pooled=pd.concat(dfs,ignore_index=True).dropna(subset=["actual_return"])
    feat=[c for c in FEATURE_COLUMNS if c in pooled.columns]
    X=pooled[feat].fillna(0.0).values.astype(np.float32)
    y=pooled["actual_up"].astype(int).values
    dates=pooled["prediction_date"]; ret=pooled["actual_return"].values
    recs={}
    for tr,te in purged_kfold_indices(dates,n_folds=n_folds,embargo=embargo):
        if len(np.unique(y[tr]))<2 or len(np.unique(y[te]))<2: continue
        m=XGBClassifier(**XGB_PARAMS); m.fit(X[tr],y[tr])
        p=m.predict_proba(X[te])[:,1]
        g=pd.DataFrame({"date":dates.iloc[te].values,"pred":p,"ret":ret[te]})
        for d,grp in g.groupby("date"):
            if len(grp)>=5 and grp["pred"].nunique()>1 and grp["ret"].nunique()>1:
                recs[d]=grp["pred"].rank().corr(grp["ret"].rank())
    return pd.Series(recs)

tickers=[t.strip().upper() for t in (ROOT/"tickers.txt").read_text().splitlines()
         if t.strip() and not t.startswith("#")]
print(f"scoring h3 + h5 on {len(tickers)} tickers (warm cache)...",flush=True)
ic3=per_date_ic(tickers,3); ic5=per_date_ic(tickers,5)
common=ic3.index.intersection(ic5.index)
c=ic3[common].corr(ic5[common])
print(f"\n=== HORIZON IC CORRELATION ===")
print(f"  h3 dates={len(ic3)}, h5 dates={len(ic5)}, common={len(common)}")
print(f"  corr(h3_IC, h5_IC) = {c:.3f}")
print(f"  h3 median IC={ic3.median():.4f}  h5 median IC={ic5.median():.4f}")
print(f"\n  READ: <0.5 = independent bets, pooling h3+h5 = REAL breadth")
print(f"        >0.8 = same bet, pooling = FAKE breadth (lever 1 dead)")
