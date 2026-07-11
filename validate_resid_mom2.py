import sys; sys.path.insert(0, ".")
from pathlib import Path
import numpy as np, pandas as pd
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

CANDIDATES = ["sector_rel_ret__ts_mean__w10", "sector_rel_ret__ts_decay_linear__w10",
              "sector_rel_ret__ts_decay_linear__w20", "sector_rel_ret__ts_mean__w20"]
TF = "__cs_rank"
CONTROLS = ["return_1d","return_3d","return_5d","return_20d","macd","rsi_14","short_pct_float","pc_ratio_snap"]

for HORIZON in (3, 5):
    m = _merge_outcomes(_load_panel(PANEL_DIR), DB, HORIZON)
    ctrl = [c+TF for c in CONTROLS if (c+TF) in m.columns]
    print(f"\n{'='*64}\nHORIZON {HORIZON}d  residual momentum (windowed sector_rel_ret)\n{'='*64}")
    for col in CANDIDATES:
        if col not in m.columns:
            print(f"  {col}: not in panel"); continue
        raw = daily_ic_t(m, col)
        corrs = {}
        for cc in ctrl:
            sub = m[[col, cc]].dropna()
            if len(sub) > 100:
                corrs[cc.replace(TF,'')] = round(float(sub[col].rank().corr(sub[cc].rank())),3)
        m["__r"] = residualize(m, col, ctrl)
        res = daily_ic_t(m, "__r")
        maxc = max((abs(v) for v in corrs.values()), default=0)
        topc = sorted(corrs.items(), key=lambda x:-abs(x[1]))[:3]
        print(f"\n  {col}")
        print(f"     raw: {raw}")
        print(f"     top corr: {topc}")
        print(f"     residual: {res}  max|corr|={maxc}")
        if raw and res:
            keep = abs(res["ic_t"])>3 and (abs(raw["ic_t"])-abs(res["ic_t"]))/abs(raw["ic_t"])<0.5
            v = "ADMIT" if (keep and maxc<0.4) else "REDUNDANT" if maxc>=0.4 else "WEAK"
            print(f"     -> {v}")
