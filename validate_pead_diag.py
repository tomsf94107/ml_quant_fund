import sys; sys.path.insert(0, ".")
from pathlib import Path
import numpy as np, pandas as pd, sqlite3
from analysis.alpha_fitness import _load_panel, _merge_outcomes

PANEL_DIR = Path("data/alpha_panel"); DB = Path("accuracy.db")
DECAY_DAYS = 60

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

con = sqlite3.connect(DB)
ec = pd.read_sql("SELECT ticker, report_date, actual_eps, est_eps FROM earnings_cache "
                 "WHERE actual_eps IS NOT NULL AND est_eps IS NOT NULL", con)
con.close()
ec["report_date"] = pd.to_datetime(ec["report_date"], errors="coerce")
ec = ec.dropna(subset=["report_date"])
ec["sue"] = ((ec["actual_eps"] - ec["est_eps"]) / ec["est_eps"].abs().clip(lower=0.01)).clip(-5,5)
ec = ec.sort_values(["ticker","report_date"])

def build_pead(panel_dates_by_ticker):
    rows = []
    ec_by_t = {}
    for t, g in ec.groupby("ticker"):
        ec_by_t[t] = (g["report_date"].values.astype("datetime64[ns]"), g["sue"].values.astype(float))
    for tkr, dates in panel_dates_by_ticker.items():
        ev = ec_by_t.get(tkr)
        if ev is None: continue
        rep_dates, sues = ev
        for d in np.asarray(dates).astype("datetime64[ns]"):
            elapsed = (d - rep_dates) / np.timedelta64(1, "D")
            mask = (elapsed >= 0) & (elapsed <= DECAY_DAYS)
            if not mask.any(): continue
            i = np.where(mask)[0][-1]
            rows.append((tkr, pd.Timestamp(d), float(sues[i]) * (1.0 - elapsed[i]/DECAY_DAYS)))
    return pd.DataFrame(rows, columns=["ticker","date","pead_drift"])

TF = "__cs_rank"
MOM_ONLY    = ["return_1d","return_3d","return_5d","macd"]
MOM_PLUS_SF = MOM_ONLY + ["short_pct_float"]
SF_ONLY     = ["short_pct_float"]

for HORIZON in (3, 5):
    m = _merge_outcomes(_load_panel(PANEL_DIR), DB, HORIZON)
    pead = build_pead({t: g["date"].values for t, g in m.groupby("ticker")})
    m = m.merge(pead, on=["ticker","date"], how="inner")
    print(f"\n{'='*64}\nHORIZON {HORIZON}d  pead_drift — staged residualization\n{'='*64}")
    print(f"  raw IC:                 {daily_ic_t(m, 'pead_drift')}")
    for label, ctrl in [("vs momentum only", MOM_ONLY), ("vs short_float only", SF_ONLY), ("vs momentum+short_float", MOM_PLUS_SF)]:
        cc = [c+TF for c in ctrl if (c+TF) in m.columns]
        m["__r"] = residualize(m, "pead_drift", cc)
        print(f"  residual {label:24s}: {daily_ic_t(m, '__r')}")
