import sys; sys.path.insert(0, ".")
from pathlib import Path
import numpy as np, pandas as pd, sqlite3
from analysis.alpha_fitness import _load_panel, _merge_outcomes

PANEL_DIR = Path("data/alpha_panel"); DB = Path("accuracy.db")

def daily_ic_t(df, col):
    tmp = df[["date", col, "actual_return"]].dropna().rename(columns={col:"a","actual_return":"r"})
    if len(tmp)==0 or tmp["a"].std()==0 or tmp["a"].nunique()<3: return None
    daily = tmp.groupby("date", group_keys=False).apply(
        lambda g: g["a"].rank().corr(g["r"].rank()) if len(g)>=5 and g["a"].nunique()>=3 else np.nan,
        include_groups=False).dropna()
    if len(daily) < 10: return None
    ic, sd, nd = daily.mean(), daily.std(), len(daily)
    t = (ic/sd*np.sqrt(nd)) if sd and sd>0 else 0.0
    return {"rank_ic":round(float(ic),5),"ic_t":round(float(t),2),"n_days":int(nd)}

def residualize(df, ycol, xcols):
    out = pd.Series(index=df.index, dtype=float)
    for dt, g in df.groupby("date"):
        sub = g[[ycol]+xcols].dropna()
        if len(sub) < 10: continue
        X = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in xcols])
        y = sub[ycol].values
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            out.loc[sub.index] = y - X @ beta
        except Exception: continue
    return out

# skew_change from options_skew_history (skew_25d is 100% populated)
con = sqlite3.connect(DB)
sk = pd.read_sql("SELECT date, ticker, skew_25d FROM options_skew_history ORDER BY ticker, date", con)
con.close()
sk["date"] = pd.to_datetime(sk["date"])
sk = sk.dropna(subset=["skew_25d"]).drop_duplicates(["ticker","date"])
sk["skew_change"] = sk.groupby("ticker")["skew_25d"].diff()
sk = sk.dropna(subset=["skew_change"])
print(f"skew_change rows: {len(sk)}, dates: {sk['date'].nunique()}, tickers: {sk['ticker'].nunique()}")

TF = "__cs_rank"
CONTROLS = ["return_1d","return_3d","return_5d","macd","short_pct_float","pc_ratio_snap","iv_skew_snap"]

for HORIZON in (3, 5):
    m = _merge_outcomes(_load_panel(PANEL_DIR), DB, HORIZON)
    m = m.merge(sk[["ticker","date","skew_change"]], on=["ticker","date"], how="inner")
    ctrl = [c+TF for c in CONTROLS if (c+TF) in m.columns]
    print(f"\n{'='*64}\nHORIZON {HORIZON}d  skew_change  (PRELIM: ~76-day history only)\n  controls: {[c.replace(TF,'') for c in ctrl]}\n{'='*64}")
    raw = daily_ic_t(m, "skew_change")
    corrs = {}
    for cc in ctrl:
        sub = m[["skew_change", cc]].dropna()
        if len(sub) > 50:
            corrs[cc.replace(TF,'')] = round(float(sub["skew_change"].rank().corr(sub[cc].rank())),3)
    m["__resid"] = residualize(m, "skew_change", ctrl)
    res = daily_ic_t(m, "__resid")
    print(f"  raw IC:           {raw}")
    print(f"  corr to controls: {corrs}")
    print(f"  RESIDUAL IC:      {res}")
    if raw and res:
        maxcorr = max((abs(v) for v in corrs.values()), default=0)
        keep = abs(res["ic_t"])>3 and (abs(raw["ic_t"])-abs(res["ic_t"]))/abs(raw["ic_t"])<0.5
        v = "PROMISING (needs more history)" if (keep and maxcorr<0.4) else "REDUNDANT" if maxcorr>=0.4 else "WEAK"
        print(f"  max|corr|={maxcorr}  -> {v}  [PRELIM — not full validation]")
