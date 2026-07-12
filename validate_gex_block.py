#!/usr/bin/env python3
"""
validate_gex_block.py -- the honest significance test for the GEX->vol result.

WHAT SURVIVED SO FAR
  corr(gex_z, forward 20d vol) = -0.50, and the PARTIAL correlation after removing
  trailing vol is -0.52 -- 104% retained. GEX is not a volatility proxy: it carries
  incremental information. Direction matches the mechanism (high dealer gamma ->
  dealers dampen -> lower forward vol), monotonically across 3/5/10/20d.

WHY THAT IS NOT YET ENOUGH
  1. OVERLAP. fv20 on day t covers t+1..t+20; on day t+1 it covers t+2..t+21. They
     share 19 of 20 days. 192 daily observations are NOT 192 independent ones --
     effective n is closer to 10. THIS IS THE BUG THAT PRODUCED PEAD's t=-20.
  2. THE i.i.d. SHUFFLE NULL IS TOO EASY. Shuffling destroys the autocorrelation
     that BOTH series have. A shuffled series is a much weaker opponent than the
     real one, so 7.1 sigma against it is an overstatement.

THE FIX
  A) NON-OVERLAPPING windows: sample every h-th day. Fewer points, but each is
     genuinely independent. If the effect survives here it is real.
  B) STATIONARY BLOCK BOOTSTRAP null: resample in blocks so the surrogate keeps the
     autocorrelation structure. That is the honest opponent.
  C) Report both, and say plainly which horizons survive and which do not.
"""
import sqlite3, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, ".")

con = sqlite3.connect("accuracy.db")
gx = pd.read_sql("SELECT ticker, date, net_gamma FROM options_greeks "
                 "WHERE net_gamma IS NOT NULL", con); con.close()
con = sqlite3.connect("prices.db")
px = pd.read_sql("SELECT ticker, date, adj_close c FROM daily_prices "
                 "WHERE date >= '2025-06-01'", con); con.close()

gx["date"] = pd.to_datetime(gx["date"]); px["date"] = pd.to_datetime(px["date"])
gx = gx.sort_values(["ticker","date"])
g = gx.groupby("ticker")["net_gamma"]
gx["z"] = ((gx["net_gamma"] - g.transform(lambda s: s.rolling(60, min_periods=20).mean()))
           / g.transform(lambda s: s.rolling(60, min_periods=20).std()))
gx = gx.replace([np.inf,-np.inf], np.nan).dropna(subset=["z"])
agg = gx.groupby("date")["z"].mean().rename("gex").reset_index()

px = px.sort_values(["ticker","date"]); px["ret"] = px.groupby("ticker")["c"].pct_change()
mkt = px.groupby("date")["ret"].mean().rename("mkt").reset_index()
d = agg.merge(mkt, on="date").sort_values("date").reset_index(drop=True)
d["vol20"] = d["mkt"].rolling(20).std()*np.sqrt(252)

print(f"  {len(d)} trading days\n")
print("="*82)
print("  A) NON-OVERLAPPING WINDOWS -- every h-th day. Each point is independent.")
print("="*82)
print(f"  {'fwd':>5}{'overlap n':>11}{'INDEP n':>9}{'corr':>9}{'p':>9}   verdict")
for w in (3,5,10,20):
    d[f"fv{w}"] = d["mkt"].rolling(w).std().shift(-w)*np.sqrt(252)
    s = d.dropna(subset=[f"fv{w}","gex"]).reset_index(drop=True)
    ind = s.iloc[::w]                      # every w-th row: no shared forward days
    if len(ind) < 8:
        print(f"  {w:>4}d{len(s):>11}{len(ind):>9}   too few independent points"); continue
    r, p = stats.spearmanr(ind["gex"], ind[f"fv{w}"])
    v = ("REAL" if p < 0.05 else "not sig" if p < 0.20 else "nothing")
    print(f"  {w:>4}d{len(s):>11}{len(ind):>9}{r:>+9.3f}{p:>9.3f}   {v}")

print()
print("="*82)
print("  B) BLOCK BOOTSTRAP NULL -- surrogate keeps the autocorrelation")
print("="*82)
rng = np.random.default_rng(7)
def block_perm(x, L, rng):
    n = len(x); out = []
    while len(out) < n:
        i = rng.integers(0, n)
        out.extend(x[i:i+L] if i+L <= n else np.concatenate([x[i:], x[:L-(n-i)]]))
    return np.array(out[:n])

print(f"  {'fwd':>5}{'real':>9}{'null mean':>11}{'null sd':>9}{'sigma':>8}{'p':>8}   verdict")
for w in (3,5,10,20):
    s = d.dropna(subset=[f"fv{w}","gex"]).reset_index(drop=True)
    real = stats.spearmanr(s["gex"], s[f"fv{w}"])[0]
    L = max(w, 20)                          # block >= the forward window
    null = np.array([stats.spearmanr(block_perm(s["gex"].values, L, rng), s[f"fv{w}"])[0]
                     for _ in range(500)])
    sd = null.std()
    sig = (real - null.mean())/sd if sd > 0 else np.nan
    pv = (np.abs(null - null.mean()) >= abs(real - null.mean())).mean()
    v = "REAL" if pv < 0.05 else "not sig"
    print(f"  {w:>4}d{real:>+9.3f}{null.mean():>+11.3f}{sd:>9.3f}{sig:>+8.1f}{pv:>8.3f}   {v}")

print()
print("  A block bootstrap preserves the autocorrelation an i.i.d. shuffle destroys.")
print("  It is a much harder opponent -- and the honest one. If the effect clears")
print("  BOTH the non-overlapping test and the block null, it is real.")
