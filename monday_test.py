import sys; sys.path.insert(0, ".")
from pathlib import Path
import numpy as np, pandas as pd
from analysis.alpha_fitness import _load_panel, _merge_outcomes
PANEL_DIR = Path("data/alpha_panel"); DB = Path("accuracy.db")
def daily_ic_t(df, col):
    tmp = df[["date", col, "actual_return"]].dropna().rename(columns={col:"a","actual_return":"r"})
    if len(tmp)==0 or tmp["a"].std()==0 or tmp["a"].nunique()<3: return None
    daily = tmp.groupby("date").apply(lambda g: g["a"].rank().corr(g["r"].rank()) if len(g)>=5 and g["a"].nunique()>=3 else np.nan).dropna()
    if len(daily)<2: return None
    ic, sd, nd = daily.mean(), daily.std(), len(daily)
    t = (ic/sd*np.sqrt(nd)) if sd and sd>0 else 0.0
    return {"rank_ic":round(float(ic),5),"ic_t":round(float(t),2),"n_days":int(nd)}
for HORIZON in (3,5):
    m = _merge_outcomes(_load_panel(PANEL_DIR), DB, HORIZON)
    m["dow"] = pd.to_datetime(m["date"]).dt.dayofweek
    cols = [c for c in m.columns if c.startswith("monday_sentiment__")]
    best_col, best = None, None
    for c in cols:
        r = daily_ic_t(m, c)
        if r and (best is None or abs(r["ic_t"])>abs(best["ic_t"])): best, best_col = r, c
    print(f"\n===== monday_sentiment h={HORIZON} =====")
    print(f"strongest transform: {best_col}")
    print(f"  Monday rows {int((m.dow==0).sum())}  non-Monday {int((m.dow!=0).sum())}")
    print(f"  ALL days:         {daily_ic_t(m, best_col)}")
    print(f"  Mondays EXCLUDED: {daily_ic_t(m[m.dow!=0], best_col)}")
    print(f"  Mondays ONLY:     {daily_ic_t(m[m.dow==0], best_col)}")
