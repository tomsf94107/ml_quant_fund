import sys; sys.path.insert(0, ".")
from pathlib import Path
import numpy as np, pandas as pd
from analysis.alpha_fitness import _load_panel, _merge_outcomes

PANEL_DIR = Path("data/alpha_panel"); DB = Path("accuracy.db")
CANDIDATES = ["pc_ratio_snap", "short_pct_float", "monday_sentiment"]
TF = "__cs_rank"
MOM_BASES = ["return_1d", "return_3d", "return_5d", "rsi_14", "macd"]

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
        except Exception:
            continue
    return out

for HORIZON in (3, 5):
    m = _merge_outcomes(_load_panel(PANEL_DIR), DB, HORIZON)
    mom_cols = [b+TF for b in MOM_BASES if (b+TF) in m.columns]
    print(f"\n{'='*66}\nHORIZON {HORIZON}d   vs momentum: {[c.replace(TF,'') for c in mom_cols]}\n{'='*66}")
    for base in CANDIDATES:
        col = base + TF
        if col not in m.columns:
            print(f"  {col}: NOT IN PANEL"); continue
        raw = daily_ic_t(m, col)
        corrs = {}
        for mc in mom_cols:
            sub = m[[col, mc]].dropna()
            if len(sub) > 100:
                corrs[mc.replace(TF,'')] = round(float(sub[col].rank().corr(sub[mc].rank())), 3)
        mres = m.copy()
        mres["__resid"] = residualize(mres, col, mom_cols)
        res = daily_ic_t(mres, "__resid")
        print(f"\n  {base}")
        print(f"    raw IC:           {raw}")
        print(f"    corr to momentum: {corrs}")
        print(f"    RESIDUAL IC:      {res}")
        if raw and res:
            keep = abs(res["ic_t"]) > 3 and (abs(raw["ic_t"]) - abs(res["ic_t"])) / abs(raw["ic_t"]) < 0.5
            maxcorr = max((abs(v) for v in corrs.values()), default=0)
            verdict = "ADMIT (survives, decorrelated)" if (keep and maxcorr < 0.4) else \
                      "REDUNDANT (momentum in disguise)" if maxcorr >= 0.4 else \
                      "WEAKENS (residual IC collapses)"
            print(f"    max|corr|={maxcorr}  -> {verdict}")
