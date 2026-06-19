import sys; sys.path.insert(0, ".")
from pathlib import Path
import numpy as np, pandas as pd, sqlite3
from analysis.alpha_fitness import _load_panel, _merge_outcomes

PANEL_DIR = Path("data/alpha_panel"); DB = Path("accuracy.db")
DECAY_DAYS = 60   # PEAD drift window

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

# 1. SUE from earnings_cache
con = sqlite3.connect(DB)
ec = pd.read_sql("SELECT ticker, report_date, actual_eps, est_eps FROM earnings_cache "
                 "WHERE actual_eps IS NOT NULL AND est_eps IS NOT NULL", con)
con.close()
ec["report_date"] = pd.to_datetime(ec["report_date"], errors="coerce")
ec = ec.dropna(subset=["report_date"])
# SUE = (actual - est)/|est|, clipped to tame tiny-denominator blowups
ec["sue"] = (ec["actual_eps"] - ec["est_eps"]) / ec["est_eps"].abs().clip(lower=0.01)
ec["sue"] = ec["sue"].clip(-5, 5)
ec = ec.sort_values(["ticker","report_date"])
print(f"earnings events with SUE: {len(ec)}, tickers: {ec['ticker'].nunique()}")

# 2. For each (ticker, panel-date), find most recent earnings within DECAY_DAYS,
#    feature = sue * linear_decay(days_since / DECAY_DAYS)
def build_pead(panel_dates_by_ticker):
    rows = []
    # build per-ticker arrays with explicit dtypes (report_date as datetime64[ns], sue as float)
    ec_by_t = {}
    for t, g in ec.groupby("ticker"):
        rd = g["report_date"].values.astype("datetime64[ns]")
        sv = g["sue"].values.astype(float)
        ec_by_t[t] = (rd, sv)
    for tkr, dates in panel_dates_by_ticker.items():
        ev = ec_by_t.get(tkr)
        if ev is None: continue
        rep_dates, sues = ev
        dates = np.asarray(dates).astype("datetime64[ns]")
        for d in dates:
            elapsed = (d - rep_dates) / np.timedelta64(1, "D")
            mask = (elapsed >= 0) & (elapsed <= DECAY_DAYS)
            if not mask.any(): continue
            i = np.where(mask)[0][-1]  # most recent within window
            decay = 1.0 - (elapsed[i] / DECAY_DAYS)  # linear decay
            rows.append((tkr, pd.Timestamp(d), float(sues[i]) * decay))
    return pd.DataFrame(rows, columns=["ticker","date","pead_drift"])

TF = "__cs_rank"
CONTROLS = ["return_1d","return_3d","return_5d","macd","short_pct_float","pc_ratio_snap"]

for HORIZON in (3, 5):
    m = _merge_outcomes(_load_panel(PANEL_DIR), DB, HORIZON)
    pdbt = {t: g["date"].values for t, g in m.groupby("ticker")}
    pead = build_pead(pdbt)
    m = m.merge(pead, on=["ticker","date"], how="inner")
    ctrl = [c+TF for c in CONTROLS if (c+TF) in m.columns]
    print(f"\n{'='*64}\nHORIZON {HORIZON}d  pead_drift (SUE x 60d linear decay)\n  merged rows: {len(m)}  controls: {[c.replace(TF,'') for c in ctrl]}\n{'='*64}")
    raw = daily_ic_t(m, "pead_drift")
    corrs = {}
    for cc in ctrl:
        sub = m[["pead_drift", cc]].dropna()
        if len(sub) > 100:
            corrs[cc.replace(TF,'')] = round(float(sub["pead_drift"].rank().corr(sub[cc].rank())),3)
    m["__resid"] = residualize(m, "pead_drift", ctrl)
    res = daily_ic_t(m, "__resid")
    print(f"  raw IC:           {raw}")
    print(f"  corr to controls: {corrs}")
    print(f"  RESIDUAL IC:      {res}")
    if raw and res:
        maxcorr = max((abs(v) for v in corrs.values()), default=0)
        keep = abs(res["ic_t"])>3 and (abs(raw["ic_t"])-abs(res["ic_t"]))/abs(raw["ic_t"])<0.5
        v = "ADMIT" if (keep and maxcorr<0.4) else "REDUNDANT" if maxcorr>=0.4 else "WEAK"
        print(f"  max|corr|={maxcorr}  -> {v}")
