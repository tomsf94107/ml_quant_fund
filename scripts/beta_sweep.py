#!/usr/bin/env python3
"""Book-level beta for every gated alpha, TRAIN WINDOW ONLY.

Diagnostic. The 120 test dates are spent; nothing here reads them. Calls
build_books ONCE PER ALPHA so one thin alpha cannot truncate the others
(book_build defect B) -- each alpha reports its own n.
"""
import argparse, math, os, sqlite3, sys
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--survivors", default="train_survivors_purged.csv")
ap.add_argument("--horizon", type=int, default=5)
ap.add_argument("--test-dates", type=int, default=120)
ap.add_argument("--decile", type=int, default=10)
ap.add_argument("--cost-bps", type=float, default=10.0)
ap.add_argument("--min-names", type=int, default=30)
ap.add_argument("--vol-window", type=int, default=20)
ap.add_argument("--legs", choices=["long","ls"], default="long")
ap.add_argument("--csv", default="beta_sweep.csv")
ap.add_argument("--root", default=os.path.expanduser("~/ML_Quant_Fund"))
a = ap.parse_args()

sys.path.insert(0, a.root)
from pathlib import Path
import numpy as np, pandas as pd
from analysis.alpha_fitness import _load_panel, _merge_outcomes
from analysis.book_build import build_books

surv = pd.read_csv(a.survivors)
alphas = list(dict.fromkeys(surv["alpha"].tolist()))
print(f"# {len(alphas)} alphas from {a.survivors}")

m = _merge_outcomes(_load_panel(Path(a.root)/"data"/"alpha_panel"),
                    Path(a.root)/"accuracy.db", a.horizon)
missing = [x for x in alphas if x not in m.columns]
assert not missing, f"FATAL: not in panel: {missing[:5]}"

con = sqlite3.connect(os.path.join(a.root, "prices.db"), timeout=30)
px = defaultdict(list)
for t, d, c in con.execute("SELECT ticker,d,close FROM raw_bars WHERE close>0 "
                           "ORDER BY ticker,d"):
    px[t.upper()].append((d, float(c)))
con.close()
vol, W = {}, a.vol_window
for t, rows in px.items():
    cs = [c for _, c in rows]; ds = [d for d, _ in rows]
    rets = [0.0] + [cs[i]/cs[i-1]-1.0 for i in range(1, len(cs))]
    for i in range(W, len(cs)):
        w = rets[i-W:i]; mu = sum(w)/W
        sd = math.sqrt(sum((x-mu)**2 for x in w)/(W-1))
        if sd > 0: vol[(t, ds[i])] = sd

dates = sorted(m["date"].unique())
split = dates[len(dates) - a.test_dates]
reb_tr = [d for d in dates[::a.horizon] if d < split]
print(f"# panel {len(dates)} dates | split {str(split)[:10]} | "
      f"train rebalances available {len(reb_tr)}")
assert max(reb_tr) < split, "FATAL: train rebalance at or past split"

spy = dict(px["SPY"])
di = {str(d)[:10]: i for i, d in enumerate(dates)}
def spy_fwd(d0):
    i = di[str(d0)[:10]]
    if i + a.horizon >= len(dates): return np.nan
    x, y = str(dates[i])[:10], str(dates[i+a.horizon])[:10]
    return spy[y]/spy[x]-1.0 if x in spy and y in spy else np.nan

out = []
for j, col in enumerate(alphas, 1):
    base = col.split("__")[0]
    nm = m[["date", "ticker", "actual_return", col]]
    books, rd, dg = build_books(nm, {base: col}, vol, reb_tr,
                                decile=a.decile, cost_bps=a.cost_bps,
                                min_names=a.min_names, legs=a.legs)
    r = np.array(books.get(base, []), dtype=float)
    if len(r) < 20:
        print(f"[{j}/{len(alphas)}] {col:46s} SKIP n={len(r)}"); continue
    s = np.array([spy_fwd(d) for d in rd], dtype=float)
    k = ~np.isnan(s); r, s = r[k], s[k]
    n = len(r)
    c_, bt = np.linalg.lstsq(np.c_[np.ones(n), s], r, rcond=None)[0]
    res = r - (c_ + bt*s)
    r2 = 1 - (res**2).sum()/((r-r.mean())**2).sum()
    t = math.sqrt(r2/(1-r2)*(n-2)) if 0 < r2 < 1 else float("nan")
    ann = math.sqrt(252.0/a.horizon)
    out.append(dict(alpha=col, base=base, n=n, beta=round(bt,3),
                    r2=round(r2,3), beta_t=round(t,2),
                    sh_raw=round(r.mean()/r.std(ddof=1)*ann,2),
                    sh_hedged=round((r-bt*s).mean()/(r-bt*s).std(ddof=1)*ann,2),
                    breaks=dg["break_by"].get(base,0)))
    print(f"[{j}/{len(alphas)}] {col:46s} n={n:3d} beta={bt:6.2f} R2={r2:5.2f}")

df = pd.DataFrame(out).sort_values("r2", ascending=False)
df.to_csv(a.csv, index=False)
print(f"\n{df.to_string(index=False)}\n# wrote {a.csv}  ({len(df)} of {len(alphas)})")
