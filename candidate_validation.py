import sys; sys.path.insert(0, ".")
from pathlib import Path
import numpy as np, pandas as pd
from analysis.alpha_fitness import _load_panel, _merge_outcomes
PANEL_DIR = Path("data/alpha_panel"); DB = Path("accuracy.db")
CANDIDATES = ["pc_ratio_snap", "short_pct_float", "monday_sentiment"]
FIXED_TRANSFORMS = ["__cs_rank", "__cs_zscore"]
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
for HORIZON in (3, 5):
    m = _merge_outcomes(_load_panel(PANEL_DIR), DB, HORIZON)
    m["dow"] = pd.to_datetime(m["date"]).dt.dayofweek
    print(f"\n{'='*64}\nHORIZON {HORIZON}d   (h=5 = overlap-inflated upper bounds)\n{'='*64}")
    for base in CANDIDATES:
        for tf in FIXED_TRANSFORMS:
            col = base + tf
            if col not in m.columns:
                print(f"  {col}: NOT IN PANEL"); continue
            allr = daily_ic_t(m, col); exr = daily_ic_t(m[m.dow!=0], col)
            print(f"\n  {col}")
            print(f"    ALL days:         {allr}")
            print(f"    Mondays EXCLUDED: {exr}")
            if allr and exr and abs(allr["ic_t"])>3:
                v = "ROBUST (survives ex-Monday)" if abs(exr["ic_t"])>3 else "WEAKENS off-Monday"
                print(f"    -> {v}")
