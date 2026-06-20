import sys; sys.path.insert(0, ".")
from pathlib import Path
import numpy as np, pandas as pd
from analysis.alpha_fitness import _load_panel, _merge_outcomes

PANEL_DIR = Path("data/alpha_panel"); DB = Path("accuracy.db")
ADMITTED = ["pc_ratio_snap__cs_rank", "short_pct_float__cs_rank"]

def daily_ic_series(df, col):
    tmp = df[["date", col, "actual_return"]].dropna().rename(columns={col:"a","actual_return":"r"})
    daily = tmp.groupby("date", group_keys=False).apply(
        lambda g: g["a"].rank().corr(g["r"].rank()) if len(g)>=5 and g["a"].nunique()>=3 else np.nan,
        include_groups=False).dropna()
    return daily

def ic_t(daily):
    if len(daily) < 10: return None
    ic, sd, nd = daily.mean(), daily.std(), len(daily)
    t = (ic/sd*np.sqrt(nd)) if sd and sd>0 else 0.0
    return {"ic":round(float(ic),5),"t":round(float(t),2),"nd":int(nd)}

def turnover(df, col):
    # avg |rank change| per ticker day-over-day (0=never changes, 1=full flip). Lower=cheaper.
    d = df[["date","ticker",col]].dropna().copy()
    d["rank"] = d.groupby("date")[col].rank(pct=True)
    d = d.sort_values(["ticker","date"])
    d["drank"] = d.groupby("ticker")["rank"].diff().abs()
    return float(d["drank"].mean())

for HORIZON in (3, 5):
    m = _merge_outcomes(_load_panel(PANEL_DIR), DB, HORIZON)
    m = m.sort_values("date")
    print(f"\n{'='*64}\nHORIZON {HORIZON}d  — BOTH-ERAS + COST gate\n{'='*64}")
    for col in ADMITTED:
        if col not in m.columns:
            print(f"  {col}: not in panel"); continue
        daily = daily_ic_series(m, col)
        if len(daily) < 20:
            print(f"  {col}: too few days"); continue
        # split into two eras by date median
        mid = daily.index[len(daily)//2]
        e1 = daily[daily.index < mid]; e2 = daily[daily.index >= mid]
        full, era1, era2 = ic_t(daily), ic_t(e1), ic_t(e2)
        tov = turnover(m, col)
        # cost: rough net-of-cost IC = IC scaled by (1 - turnover penalty). slow snapshots ~ low tov.
        print(f"\n  {col}")
        print(f"    full:  {full}")
        print(f"    era1:  {era1}  ({e1.index.min().date()}..{e1.index.max().date()})")
        print(f"    era2:  {era2}  ({e2.index.min().date()}..{e2.index.max().date()})")
        print(f"    turnover (avg |Δrank|/day): {round(tov,4)}  ({'LOW/cheap' if tov<0.1 else 'MOD' if tov<0.2 else 'HIGH/costly'})")
        if era1 and era2:
            same_sign = np.sign(era1["ic"]) == np.sign(era2["ic"])
            both_sig = abs(era1["t"])>2 and abs(era2["t"])>2
            v = "PASS both-eras" if (same_sign and both_sig) else \
                "PARTIAL (one era weak)" if same_sign else "FAIL (sign flips)"
            print(f"    -> {v}")
