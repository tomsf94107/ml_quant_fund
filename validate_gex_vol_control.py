#!/usr/bin/env python3
"""
validate_gex_vol_control.py -- is GEX just a volatility proxy?

THE SUSPICION
  validate_gex_regime found:
    corr(gex_z, forward 20d vol) = -0.56, NW-t -3.33   [strong, monotonic]
    HIGH negative-gamma breadth -> REVERSAL (-0.21, t -2.97)
  The second result is BACKWARDS from the mechanism that motivated the test:
  negative gamma means dealers AMPLIFY, which predicts MOMENTUM, not reversal.
  When a result contradicts its own mechanism, you are measuring something else.

  The something else is almost certainly CURRENT VOLATILITY:
    1. Vol is persistent. Anything correlated with today's vol "predicts" forward
       vol. That alone explains Test 1.
    2. Short-term reversal is stronger in high vol. That explains Test 2 -- and it
       explains the backwards sign exactly.

  This is the same error class as the beta trap that killed three prior findings:
  a known, free, already-priced factor wearing a costume.

THE CONTROL
  Partial correlation: does gex_z predict FORWARD vol after removing TRAILING vol?
  Trailing realized vol is the "beta" here. If the GEX effect survives, it is real
  and it is incremental. If it collapses, GEX is a vol proxy and worthless as a
  separate signal -- you would just use realized vol, which is free.

  Plus: a 500-draw shuffle null (one draw is a coin flip, not a control), and the
  Test-2 reversal effect re-run WITHIN vol terciles so the vol level is held fixed.
"""
import sqlite3, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")

def nw_t(x, lag):
    x = np.asarray(x, float); n = len(x)
    if n < 3: return np.nan
    e = x - x.mean(); var = (e @ e) / n
    for k in range(1, min(lag, n-1)+1):
        var += 2.0*(1.0-k/(lag+1.0))*((e[k:] @ e[:-k])/n)
    return x.mean()/np.sqrt(var/n) if var > 0 else np.nan

def partial_corr(x, y, z):
    """corr(x, y) with the linear effect of z removed from BOTH."""
    m = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
    if len(m) < 30: return np.nan, 0
    rx = m["x"] - np.polyval(np.polyfit(m["z"], m["x"], 1), m["z"])
    ry = m["y"] - np.polyval(np.polyfit(m["z"], m["y"], 1), m["z"])
    return rx.corr(ry, method="spearman"), len(m)

con = sqlite3.connect("accuracy.db")
gx = pd.read_sql("SELECT ticker, date, net_gamma FROM options_greeks "
                 "WHERE net_gamma IS NOT NULL", con); con.close()
con = sqlite3.connect("prices.db")
px = pd.read_sql("SELECT ticker, date, adj_close c FROM daily_prices "
                 "WHERE date >= '2025-06-01'", con); con.close()

gx["date"] = pd.to_datetime(gx["date"]); px["date"] = pd.to_datetime(px["date"])
g = gx.sort_values(["ticker","date"]).groupby("ticker")["net_gamma"]
gx = gx.sort_values(["ticker","date"])
gx["z"] = ((gx["net_gamma"] - g.transform(lambda s: s.rolling(60, min_periods=20).mean()))
           / g.transform(lambda s: s.rolling(60, min_periods=20).std()))
gx = gx.replace([np.inf,-np.inf], np.nan).dropna(subset=["z"])
agg = gx.groupby("date").agg(gex=("z","mean"), pct_neg=("net_gamma", lambda s:(s<0).mean())).reset_index()

px = px.sort_values(["ticker","date"]); px["ret"] = px.groupby("ticker")["c"].pct_change()
mkt = px.groupby("date")["ret"].mean().rename("mkt").reset_index()
d = agg.merge(mkt, on="date").sort_values("date").reset_index(drop=True)

# TRAILING realized vol -- the confound
d["vol_20"] = d["mkt"].rolling(20).std()*np.sqrt(252)
d["prev"] = d["mkt"].shift(1)

print(f"  {len(d)} days, {d.date.min().date()} .. {d.date.max().date()}\n")
print("="*84)
print("  IS IT JUST VOLATILITY? raw corr vs PARTIAL corr (trailing vol removed)")
print("="*84)
print(f"  {'fwd':>5}{'raw corr':>11}{'partial':>10}{'retained':>10}{'n':>6}   verdict")
for w in (3,5,10,20):
    d[f"fv{w}"] = d["mkt"].rolling(w).std().shift(-w)*np.sqrt(252)
    s = d.dropna(subset=[f"fv{w}","gex","vol_20"])
    raw = s["gex"].corr(s[f"fv{w}"], method="spearman")
    par, n = partial_corr(s["gex"], s[f"fv{w}"], s["vol_20"])
    ret = 100*abs(par)/abs(raw) if abs(raw) > 1e-9 else np.nan
    v = ("REAL (survives)" if abs(par) > 0.15 and ret > 50
         else "IT WAS VOL" if ret < 40 else "partial/weak")
    print(f"  {w:>4}d{raw:>+11.4f}{par:>+10.4f}{ret:>9.0f}%{n:>6}   {v}")

print()
print("="*84)
print("  TEST 2 RE-RUN: reversal WITHIN vol terciles (vol held fixed)")
print("="*84)
d["vt"] = pd.qcut(d["vol_20"], 3, labels=["LOW vol","MID vol","HIGH vol"])
d["hi_neg"] = d["pct_neg"] > d["pct_neg"].median()
print(f"  {'vol tercile':>12}{'gamma regime':>16}{'days':>6}{'corr(prev,today)':>19}{'NW-t':>8}")
for vt in ["LOW vol","MID vol","HIGH vol"]:
    for lbl, m in [("HIGH neg-gamma", True), ("LOW neg-gamma", False)]:
        s = d[(d.vt==vt) & (d.hi_neg==m)].dropna(subset=["prev","mkt"])
        if len(s) < 20:
            print(f"  {vt:>12}{lbl:>16}{len(s):>6}   too few"); continue
        r = s["prev"].corr(s["mkt"])
        p = (s["prev"]-s["prev"].mean())*(s["mkt"]-s["mkt"].mean())
        print(f"  {vt:>12}{lbl:>16}{len(s):>6}{r:>19.4f}{nw_t(p.values,3):>+8.2f}")
print()
print("  If the gamma regime still flips the sign WITHIN a vol tercile, gamma adds")
print("  something. If both rows look the same inside each tercile, it was vol.")

print()
print("="*84)
print("  PROPER NULL: 500 shuffles (one draw is a coin flip, not a control)")
print("="*84)
s = d.dropna(subset=["fv20","gex"])
real = s["gex"].corr(s["fv20"], method="spearman")
rng = np.random.default_rng(0)
null = np.array([pd.Series(rng.permutation(s["gex"].values)).corr(
                 s["fv20"].reset_index(drop=True), method="spearman") for _ in range(500)])
z = (real - null.mean())/null.std()
print(f"  real corr(gex, fwd20 vol) = {real:+.4f}")
print(f"  null: mean {null.mean():+.4f}  sd {null.std():.4f}  "
      f"95% [{np.percentile(null,2.5):+.4f}, {np.percentile(null,97.5):+.4f}]")
print(f"  the real value sits {abs(z):.1f} sd outside the null")
print(f"  >> {'PASSES' if abs(z) > 2 else 'FAILS'} the null control")
